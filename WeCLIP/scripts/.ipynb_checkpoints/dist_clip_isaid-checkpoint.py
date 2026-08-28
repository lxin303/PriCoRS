import argparse
import datetime
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import sys
sys.path.append(".")
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from wsss.datasets import iSAID
from omegaconf import OmegaConf
from tqdm import tqdm

from WeCLIP.utils.losses import get_aff_loss, DiceLoss, JointLoss, CTCLoss_neg
from WeCLIP.utils import evaluate
from WeCLIP.utils.AverageMeter import AverageMeter
from WeCLIP.utils.imutils import cam_to_label
from WeCLIP.utils.camutils import cams_to_affinity_label
from WeCLIP.utils.optimizer import PolyWarmupAdamW
from WeCLIP.WeCLIP_model.model_attn_aff_isaid import WeCLIP
from WeCLIP.WeCLIP_model.context_proto import FIFOQueue, OnlineKMeans, ContextPrototypeAware, l2_normalize


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='/root/autodl-tmp/WeCLIP/configs/isaid_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size")
parser.add_argument("--temp", default=0.5, type=float, help="temp")

def normalize_map_per_img_per_cls(M: torch.Tensor):
    # M: [B, C, H, W]
    B, C, H, W = M.shape
    m = M.reshape(B * C, H, W)                     # ← 替代 .view
    m_min = torch.amin(m, dim=(1, 2), keepdim=True)
    m_max = torch.amax(m, dim=(1, 2), keepdim=True)
    m = (m - m_min) / (m_max - m_min + 1e-6)
    return m.reshape(B, C, H, W)  


def instance_proto_from_cam(z_b: torch.Tensor, Mn: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Args:
        z_b: [Dp, H, W] feature map
        Mn:  [H, W]     prob map in [0,1]
        tau: float      hard threshold

    Returns:
        proto: [Dp]     masked average over (H, W)
    """
    mask = (Mn > tau).float()
    if mask.sum() < 1:
        # 退化处理：取 top-20% 像素
        k = max(1, int(0.2 * mask.numel()))
        vals, idx = torch.topk(Mn.flatten(), k)
        mask = torch.zeros_like(Mn).flatten()
        mask[idx] = 1.0
        mask = mask.view_as(Mn)
    num = (z_b * mask).sum(dim=(1,2))
    den = mask.sum() + 1e-6
    return num / den  # [Dp]


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

        segs_clip, segs_dino, cam, attn_loss = model(inputs, cls_label, img_box=None,crops=None, mode='val')
        # segs = 0.5*(segs_dino+segs_clip)
        segs = segs_clip
        resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

        # resized_cam = F.interpolate(cam, size=labels.shape[1:], mode='bilinear', align_corners=False)
        # cam = cam_to_label(resized_cam, cls_label, bkg_thre=0.5, high_thre=0.7, low_thre=0.25, ignore_index=255)

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


def get_seg_loss(pred, label, ignore_index=255):
    
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)
        
    # 安全 CE（避免全 ignore 时除零）
    # bg_loss = _ce_safe(pred, bg_label, ignore_index)
    # fg_loss = _ce_safe(pred, fg_label, ignore_index)
    return (bg_loss + fg_loss) * 0.5


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

    num_workers = 10
    
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
        #shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=4)
    
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
                "lr": 2e-5,
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
    
    FG_START = 1
    N_CLASSES = cfg.dataset.num_classes
    FG_CLASSES = N_CLASSES - FG_START  # = num_classes-1

    TAU = 0.3
    TOP_K = 5
    NP_PER_CLASS = 10
    QUEUE_MAX = 2048
    TEMP = 0.07
    LAMBDA_SELF = getattr(cfg.train, "lambda_self", 0.3)  # 可放到yaml

    # 每个前景类一个队列与kmeans
    # support_bank = {c: FIFOQueue(QUEUE_MAX) for c in range(FG_START, N_CLASSES)}
    # kmeans_state = {c: OnlineKMeans(NP_PER_CLASS, dim=256, momentum=0.5) for c in range(FG_START, N_CLASSES)}
    # cpa = ContextPrototypeAware(dim=256, top_k=TOP_K, temperature=TEMP, weighted=False, learnable_maps=False).cuda()

    train_loader_iter = iter(train_loader)

    avg_meter = AverageMeter()

    criterion_dice = DiceLoss().cuda()
    JOINT_loss = JointLoss().cuda()
    
    for n_iter in range(cfg.train.max_iters):
        
        try:
            # img_name, inputs, cls_labels, img_box = next(train_loader_iter)
            img_name, inputs, cls_labels, img_box, crops = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            # img_name, inputs, cls_labels, img_box = next(train_loader_iter)
            img_name, inputs, cls_labels, img_box, crops = next(train_loader_iter)
        segs_clip, segs_dino, foreground_seg, cam, attn_pred, norm_cam, IS_cam = model(inputs.cuda(), cls_labels)
        
        pseudo_label = cam

        # segs= 0.5*segs_clip+0.5*segs_dino
        segs = segs_clip 
        segs = F.interpolate(segs, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        segs_clip = F.interpolate(segs_clip, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        # segs_dino = F.interpolate(segs_dino, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        foreground_seg = F.interpolate(foreground_seg, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        # proj_feats = F.interpolate(proj_feats, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)  # [B, Dp, H, W]

        pred_clip_max, pred_label_clip = torch.max(F.softmax(segs_clip, dim=1), dim=1)
        # pred_dino_max, pred_label_dino = torch.max(F.softmax(segs_dino, dim=1), dim=1)
        pred_max, pred_label_seg = torch.max(F.softmax(segs, dim=1), dim=1)

        fts_cam = cam.clone()
            
        aff_label = cams_to_affinity_label(fts_cam, mask=attn_mask, ignore_index=cfg.dataset.ignore_index)
        attn_loss, pos_count, neg_count = get_aff_loss(attn_pred, aff_label)
        
        if n_iter > 18000:
            pseudo_label[pred_max>0.75] = pred_label_seg[pred_max>0.75]

        seg_loss = get_seg_loss(segs, pseudo_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        seg_loss2 = criterion_dice(segs, pseudo_label.type(torch.long))
        # seg_loss_d = get_seg_loss(segs, dino_pseudo_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        joint_loss = JOINT_loss(foreground_seg, segs, pseudo_label.type(torch.long))
        
        # seg_clip_loss2 = get_seg_loss(segs_clip, pred_label_dino.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        # seg_dino_loss2 = get_seg_loss(segs_dino, pred_label_clip.type(torch.long), ignore_index=cfg.dataset.ignore_index)

        prob = F.softmax(segs, dim=1)  # [B, C, H, W], C=N_CLASSES

        cls_labels = cls_labels.cuda()

        # B, C, H, W = prob.shape
        # M_fg = prob[:, FG_START:, :, :]                     # [B, FG_CLASSES, H, W]，去掉背景通道
        # M_tilde_fg = torch.zeros_like(M_fg)                 # PACAM 同形
        # present_mask = cls_labels > 0 
        # batch_PI_per_cls: dict[int, list[torch.Tensor]] = {c: [] for c in range(FG_START, N_CLASSES)}

        # with torch.no_grad():
        #     for b in range(B):
        #         z_b = proj_feats[b]  # [Dp, H, W]
        #         for idx in range(FG_CLASSES):
        #             if not present_mask[b, idx]:
        #                 continue
        #             Mn = M_fg[b, idx]                               # [H, W] 第 c 类概率图
        #             PI = instance_proto_from_cam(z_b, Mn, TAU)      # [Dp]
        #             c_abs = idx + FG_START
        #             support_bank[c_abs].push(PI)
        #             batch_PI_per_cls[c_abs].append(PI)

        # for c in range(FG_START, N_CLASSES):
        #     X = support_bank[c].as_tensor()    # [M, Dp] or None
        #     if X is not None and len(X) >= TOP_K:
        #         kmeans_state[c].partial_fit(X)

        # for b in range(B):
        #     z_b = proj_feats[b]  # [Dp, H, W]
        #     for idx in range(FG_CLASSES):
        #         if not present_mask[b, idx]:
        #             continue
        #         c_abs = idx + FG_START
        #         P_cand = kmeans_state[c_abs].get_prototypes()  # [Np, Dp] (cpu) or None
        #         if P_cand is None:
        #             continue
        #         # 当前样本该类的“代表原型”可用 PI；为简洁，用分割概率最强位置近似 PI：
        #         with torch.no_grad():
        #             Mn = M_fg[b, idx]  # [H, W]
        #             PI = instance_proto_from_cam(z_b, Mn, TAU)  # [Dp]
        #         topP, w_top, _ = cpa._positiveness_topk(PI, P_cand)  # Top-K 软邻居
        #         pacam = cpa.pacam(z_b, topP, w_top)                  # [H, W]
        #         M_tilde_fg[b, idx] = pacam

        # 补背景，构造 M 与 M_tilde
        # Mb = prob[:, :1]                 # 等价于 prob[:, 0:1]
        # M  = torch.cat([prob[:, 1:], Mb], dim=1)                     # [B, FG_CLASSES+1, H, W]
        # Mb_t = 1.0 - M_tilde_fg.max(dim=1, keepdim=True).values
        # M_tilde = torch.cat([M_tilde_fg, Mb_t], dim=1)
        # M = prob[:, 1:, :, :]                 # [B, FG, H, W]
        # M_tilde = M_tilde_fg   

        # 归一化并做一致性 L1
        # M_n = normalize_map_per_img_per_cls(M.detach())      # 原路不反传
        # M_tn = normalize_map_per_img_per_cls(M_tilde)        # 反传自监督分支
        # mask_fg = (cls_labels > 0).float()[..., None, None]   # [B, FG, 1, 1]
        # valid = mask_fg.sum() * M.shape[-1] * M.shape[-2] + 1e-6
        # self_loss = ((M_n - M_tn).abs() * mask_fg).sum() / valid

        IS_cam = IS_cam / (F.adaptive_max_pool2d(IS_cam, (1, 1)) + 1e-5)
        self_loss = torch.mean(torch.abs(norm_cam - IS_cam))

        loss = 1 * seg_loss + 0.1 * attn_loss + 1 * seg_loss2 + LAMBDA_SELF * self_loss + 0.3 * joint_loss 

        avg_meter.add({'seg_loss': seg_loss.item(), 'attn_loss': attn_loss.item(), 'self_loss': self_loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (n_iter + 1) % cfg.train.log_iters == 0:
            
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            preds = torch.argmax(segs,dim=1).cpu().numpy().astype(np.int16)
            gts = pseudo_label.cpu().numpy().astype(np.int16)
            seg_mAcc = (preds==gts).sum()/preds.size

            logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e;, pseudo_seg_loss: %.4f, attn_loss: %.4f, self_loss: %.4f, pseudo_seg_mAcc: %.4f"%(n_iter+1, delta, eta, cur_lr, avg_meter.pop('seg_loss'), avg_meter.pop('attn_loss'), avg_meter.pop('self_loss'), seg_mAcc))

            writer.add_scalars('train/loss',  {"seg_loss": seg_loss.item(), "attn_loss": attn_loss.item(), "self_loss": self_loss.item()}, global_step=n_iter)

        if (n_iter +1) % cfg.train.eval_iters == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "WeCLIP_model_iter_%d.pth"%(n_iter+1))
            logging.info('Validating...')
            if (n_iter + 1) > 26000:
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
