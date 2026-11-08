import argparse
import datetime
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
TORCH_HOME = "/leonardo_work/IscrC_CropAI/RS/.cache/torch"
os.environ["TORCH_HOME"] = TORCH_HOME
os.makedirs(os.path.join(TORCH_HOME, "hub", "checkpoints"), exist_ok=True)
import random
import sys
sys.path.append(".")
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from WeCLIP.datasets import iSAID
from omegaconf import OmegaConf
from tqdm import tqdm

from WeCLIP.utils.losses import get_aff_loss, DiceLoss
from WeCLIP.utils import evaluate
from WeCLIP.utils.AverageMeter import AverageMeter
from WeCLIP.utils.camutils import cams_to_affinity_label
from WeCLIP.utils.optimizer import PolyWarmupAdamW
from WeCLIP.WeCLIP_model.model_attn_aff_isaid_sam import WeCLIP


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='/leonardo_work/IscrC_CropAI/RS/WeCLIP/configs/isaid_attn_seg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default="work_dir_iSAID_sam", type=str, help="work_dir")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size")


def validate(model=None, data_loader=None, cfg=None):

    model.eval()
    
    preds, gts, cams, aff_gts = [], [], [], []
    num = 1
    seg_hist = np.zeros((16, 16))
    cam_hist = np.zeros((16, 16))
    for _, data in tqdm(enumerate(data_loader),
                        total=len(data_loader), ncols=100, ascii=" >="):
        name, inputs, labels, cls_label = data

        inputs = inputs.cuda()
        labels = labels.cuda()
        cls_label = cls_label.cuda()

        b,c,h,w = inputs.shape
        if (h//cfg.clip_init.clip_flag==0) and (w//cfg.clip_init.clip_flag)==0:
            inputs = inputs
        else:
            new_h = round(h/cfg.clip_init.clip_flag)*cfg.clip_init.clip_flag
            new_w = round(w/cfg.clip_init.clip_flag)*cfg.clip_init.clip_flag
            inputs = F.interpolate(inputs, size=(new_h, new_w), mode='bilinear', align_corners=False)

        segs_clip, segs_dino, cam, attn_loss = model(inputs, cls_label, mode='val')
        segs = 0.5*(segs_dino+segs_clip)
        
        resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

        preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
        cams += list(cam.cpu().numpy().astype(np.int16))
        gts += list(labels.cpu().numpy().astype(np.int16))

        num+=1

        if num % 1000 ==0:
            seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist)
            cam_hist, cam_score = evaluate.scores(gts, cams, cam_hist)
            preds, gts, cams, aff_gts = [], [], [], []

    seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist)
    cam_hist, cam_score = evaluate.scores(gts, cams, cam_hist)
    model.train()
    return seg_score, cam_score


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_logger(filename='test.log'):
    ## setup logger
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def safe_cross_entropy(pred, target, ignore_index=255):
    valid = (target != ignore_index)
    if valid.sum() == 0:
        return None
    return F.cross_entropy(pred, target.long(), ignore_index=ignore_index)


def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    bg_loss = safe_cross_entropy(pred, bg_label, ignore_index)

    fg_label = label.clone()
    fg_label[label == 0] = ignore_index
    fg_loss = safe_cross_entropy(pred, fg_label, ignore_index)

    losses = []
    if bg_loss is not None:
        losses.append(bg_loss)
    if fg_loss is not None:
        losses.append(fg_loss)

    if len(losses) == 0:
        return pred.new_tensor(0.0)

    return sum(losses) / len(losses)


def object_consistency_loss(seg_logits, mask_bank, mask_labels):

    prob = F.softmax(seg_logits, dim=1)

    total_loss = seg_logits.new_tensor(0.0)
    count = 0

    for b in range(len(mask_bank)):
        for mask, fg_cls in zip(mask_bank[b], mask_labels[b]):
            mask = mask.to(seg_logits.device).bool()

            if mask.sum() < 16:
                continue

            cls_id = int(fg_cls) + 1  # foreground class -> segmentation class id

            pix_prob = prob[b, :, mask].transpose(0, 1).clamp_min(1e-6)  # [N, C]
            mean_prob = pix_prob.mean(dim=0)
            mean_prob_detached = mean_prob.detach()

            cls_loss = -torch.log(mean_prob[cls_id].clamp_min(1e-6))

            cons_loss = F.kl_div(
                pix_prob.log(),
                mean_prob_detached.unsqueeze(0).expand_as(pix_prob),
                reduction='batchmean'
            )

            total_loss = total_loss + cls_loss + 0.1 * cons_loss
            count += 1

    if count == 0:
        return seg_logits.new_tensor(0.0)

    return total_loss / count


def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w
    mask  = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius+1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius+1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask


def train(cfg):

    num_workers = 0
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = iSAID.iSAIDClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        # resize_range=cfg.dataset.resize_range,
        resize_range=cfg.dataset.resize_range,
        img_fliplr=True,
        crop_size=cfg.dataset.crop_size,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    val_dataset = iSAID.iSAIDSegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split='val',
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.samples_per_gpu,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        # prefetch_factor=4
        )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )

    model = WeCLIP(
        num_classes=cfg.dataset.num_classes,
        clip_model=cfg.clip_init.clip_pretrain_path,
        dino_model=cfg.dino_init.dino_model,
        embedding_dim=cfg.clip_init.embedding_dim,
        in_channels=cfg.clip_init.in_channels,
        dataset_root_path=cfg.dataset.root_dir,
        clip_flag=cfg.clip_init.clip_flag,
        device='cuda'
    )

    logging.info('\nNetwork config: \n%s'%(model))
    param_groups = model.get_param_groups()
    model.cuda()
    
    mask_size = int(cfg.dataset.crop_size // 16)
    attn_mask = get_mask_by_radius(h=mask_size, w=mask_size, radius=args.radius)
    writer = SummaryWriter(cfg.work_dir.tb_logger_dir)

    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[3],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter,
        max_iter = cfg.train.max_iters,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )
    logging.info('\nOptimizer: \n%s' % optimizer)
    

    LAMBDA_SELF = 0.3  

    train_loader_iter = iter(train_loader)

    avg_meter = AverageMeter()

    criterion_dice = DiceLoss().cuda()

    for n_iter in range(cfg.train.max_iters):
        
        try:
            # img_name, inputs, cls_labels, img_box = next(train_loader_iter)
            img_name, inputs, cls_labels, img_box, crops = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            # img_name, inputs, cls_labels, img_box = next(train_loader_iter)
            img_name, inputs, cls_labels, img_box, crops = next(train_loader_iter)
        
        segs_clip, segs_dino, cam, attn_pred, norm_cam, IS_cam, pgt_sam, sam_conf = model(inputs.cuda(), cls_labels)

        pseudo_label = cam

        segs= 0.5*segs_clip+0.5*segs_dino
        segs = F.interpolate(segs, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        segs_clip = F.interpolate(segs_clip, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        segs_dino = F.interpolate(segs_dino, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        # proj_feats = F.interpolate(proj_feats, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)  # [B, Dp, H, W]

        pred_clip_max, pred_label_clip = torch.max(F.softmax(segs_clip, dim=1), dim=1)
        pred_dino_max, pred_label_dino = torch.max(F.softmax(segs_dino, dim=1), dim=1)
        pred_max, pred_label_seg = torch.max(F.softmax(segs, dim=1), dim=1)

        fts_cam = cam.clone()
            
        aff_label = cams_to_affinity_label(fts_cam, mask=attn_mask, ignore_index=cfg.dataset.ignore_index)
        attn_loss, pos_count, neg_count = get_aff_loss(attn_pred, aff_label)
        
        seg_prob = F.softmax(segs.detach(), dim=1)

        cls_labels = cls_labels.cuda()
        valid_cls = torch.cat([torch.ones(cls_labels.size(0), 1, device=cls_labels.device), cls_labels], dim=1).bool()  # [B, 1 + K_fg]
        seg_prob = seg_prob.masked_fill(~valid_cls[:, :, None, None], 0.0)

        pred_max, pred_label_seg = seg_prob.max(dim=1)
        
        if n_iter > 18000:
            pseudo_label[pred_max>0.75] = pred_label_seg[pred_max>0.75]

        seg_loss = get_seg_loss(segs, pseudo_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        seg_loss2 = criterion_dice(segs, pseudo_label.type(torch.long))

        seg_clip_loss2 = get_seg_loss(segs_clip, pred_label_dino.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        seg_dino_loss2 = get_seg_loss(segs_dino, pred_label_clip.type(torch.long), ignore_index=cfg.dataset.ignore_index)

        # prob = F.softmax(segs, dim=1)  # [B, C, H, W], C=N_CLASSES

        cls_labels = cls_labels.cuda()

        IS_cam = IS_cam / (F.adaptive_max_pool2d(IS_cam, (1, 1)) + 1e-5)
        self_loss = torch.mean(torch.abs(norm_cam - IS_cam))

        # loss_cpm = F.cross_entropy(segs, pgt_sam, ignore_index=255)

        sam_loss = segs_clip.sum() * 0.0

        if pgt_sam is not None:
            seg_sam = F.interpolate(segs_clip, size=pgt_sam.shape[-2:], mode='bilinear', align_corners=False,)
            sam_loss = F.cross_entropy(seg_sam, pgt_sam.long(), ignore_index=255,)

        loss = 1 * seg_loss + 0.1 * attn_loss + 0.1 * (seg_clip_loss2 + seg_dino_loss2) + 0.5 * seg_loss2 + LAMBDA_SELF * self_loss + 0.1 * sam_loss

        avg_meter.add({'seg_loss': seg_loss.item(), 'attn_loss': attn_loss.item(), 'sam_loss': sam_loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (n_iter + 1) % cfg.train.log_iters == 0:
            
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            preds = torch.argmax(segs,dim=1).cpu().numpy().astype(np.int16)
            gts = pseudo_label.cpu().numpy().astype(np.int16)
            seg_mAcc = (preds==gts).sum()/preds.size

            logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e;, pseudo_seg_loss: %.4f, attn_loss: %.4f, sam_loss: %.4f, pseudo_seg_mAcc: %.4f"%(n_iter+1, delta, eta, cur_lr, avg_meter.pop('seg_loss'), avg_meter.pop('attn_loss'), avg_meter.pop('sam_loss'), seg_mAcc))

            writer.add_scalars('train/loss',  {"seg_loss": seg_loss.item(), "attn_loss": attn_loss.item(), "sam_loss": sam_loss.item()}, global_step=n_iter)

        if (n_iter +1) % cfg.train.eval_iters == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "WeCLIP_model_iter_%d.pth"%(n_iter+1))
            logging.info('Validating...')
            if (n_iter + 1) > 30000:
                torch.save(model.state_dict(), ckpt_name)
            seg_score, cam_score = validate(model=model, data_loader=val_loader, cfg=cfg)
            logging.info("cams score:")
            logging.info(cam_score)
            logging.info("segs score:")
            logging.info(seg_score)

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.crop_size = args.crop_size

    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.tb_logger_dir, timestamp)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)

    setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp+'.log'))
    logging.info('\nargs: %s' % args)
    logging.info('\nconfigs: %s' % cfg)

    setup_seed(1)
    train(cfg=cfg)
