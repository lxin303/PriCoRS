import argparse
import os
import sys
sys.path.append(".")

from utils.dcrf import DenseCRF
from utils.imutils import encode_cmap

# Use the same iSAID color encoding convention as CTFA when available.
try:
    from utils.imutils import encode_cmap4iSAID
except ImportError:
    def encode_cmap4iSAID(label):
        # Fallback palette matching CTFA's iSAID visualization.
        color_table = {
            (0, 127, 255): 14,
            (0, 0, 127): 9,
            (0, 127, 127): 8,
            (0, 0, 63): 1,
            (0, 100, 155): 15,
            (0, 63, 127): 4,
            (0, 0, 255): 11,
            (0, 63, 255): 3,
            (0, 63, 63): 2,
            (0, 127, 63): 7,
            (0, 63, 0): 6,
            (0, 0, 191): 10,
            (0, 63, 191): 5,
            (0, 127, 191): 13,
            (0, 191, 127): 12,
        }
        cmap = np.zeros((256, 3), dtype=np.uint8)
        for color, index in color_table.items():
            cmap[index] = np.array(color[::-1], dtype=np.uint8)
        return cmap[label.astype(np.int16), :]

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

TORCH_HOME = "/leonardo_work/IscrC_CropAI/RS/.cache/torch"
os.environ["TORCH_HOME"] = TORCH_HOME
os.makedirs(os.path.join(TORCH_HOME, "hub", "checkpoints"), exist_ok=True)

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
import imageio.v2 as imageio

from utils import evaluate
from WeCLIP.WeCLIP_model.model_attn_aff_isaid_sam import WeCLIP
from WeCLIP.datasets import iSAID

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='/leonardo_work/IscrC_CropAI/RS/WeCLIP/configs/isaid_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--work_dir", default="results", type=str, help="work_dir")
parser.add_argument("--bkg_score", default=0.45, type=float, help="bkg_score")
parser.add_argument("--resize_long", default=512, type=int, help="resize the long side")
parser.add_argument("--eval_set", default="val", type=str, help="eval_set")  # val
parser.add_argument("--model_path",
                    default="/leonardo_work/IscrC_CropAI/RS/work_dir_iSAID_sam/checkpoints/2026-08-25-01-39/WeCLIP_model_iter_52000.pth",
                    type=str,
                    help="model_path")


color_table = {
    (0, 127, 255): 14,  # plane
    (0, 0, 127): 9,  # small_vehicle
    (0, 127, 127): 8,  # large_vehicle
    (0, 0, 63): 1,  # ship
    (0, 100, 155): 15,  # harbor
    (0, 63, 127): 4,  # tennis_court
    (0, 0, 255): 11,  # swimming_pool
    (0, 63, 255): 3,  # baseball_diamond
    (0, 63, 63): 2,  # storage_tank
    (0, 127, 63): 7,  # bridge
    (0, 63, 0): 6,  # Ground_Track_Field
    (0, 0, 191): 10,  # helicopter
    (0, 63, 191): 5,  # basketball_court
    (0, 127, 191): 13,  # Soccer_ball_field
    (0, 191, 127): 12  # Roundabout
}


def _to_name_str(name):
    # DataLoader batch_size=1 时 name 可能是 ['xxx'] 或 'xxx'
    if isinstance(name, (list, tuple)):
        name = name[0]
    return str(name)


def _save_mask_pair(pred, name_str, raw_dir, rgb_dir):
    """Save class-index mask and iSAID color visualization."""
    pred = np.squeeze(pred).astype(np.uint8)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)
    imageio.imwrite(os.path.join(raw_dir, name_str + ".png"), pred)
    imageio.imwrite(
        os.path.join(rgb_dir, name_str + ".png"),
        encode_cmap4iSAID(pred).astype(np.uint8),
    )


def validate(model, dataset, test_scales=None):
    _preds, _gts, _msc_preds, cams = [], [], [], []

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False
    )

    model.cuda(0)
    model.eval()

    num = 0
    _preds_hist = np.zeros((16, 16))
    _msc_preds_hist = np.zeros((16, 16))
    _cams_hist = np.zeros((16, 16))

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
        num += 1

        name, inputs, labels, cls_labels = data
        name_str = _to_name_str(name)

        inputs = inputs.cuda()
        labels = labels.cuda()
        cls_labels = cls_labels.cuda()

        # resize long side to args.resize_long
        _, _, h, w = inputs.shape
        ratio = args.resize_long / max(h, w)
        _h, _w = int(h * ratio), int(w * ratio)
        inputs = F.interpolate(inputs, size=(_h, _w), mode='bilinear', align_corners=False)

        segs_list = []
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        cls_labels_cat = torch.cat([cls_labels, cls_labels], dim=0)

        segs_clip, segs_dino, cam, attn_loss = model(inputs_cat, cls_labels_cat, mode='val')
        segs_cat = 0.5 * (segs_dino + segs_clip)

        cam = cam[0].unsqueeze(0)
        segs = segs_cat[0].unsqueeze(0)

        _segs = (segs_cat[0, ...] + segs_cat[1, ...].flip(-1)) / 2
        segs_list.append(_segs)

        _, _, hh, ww = segs_cat.shape

        if test_scales is None:
            test_scales = [1.0]

        for s in test_scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, scale_factor=s, mode='bilinear', align_corners=False)
                _inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                # 你原代码这里用了 model(inputs, cls_labels)；保持不大改结构，按你意图用 _inputs_cat + cls_labels_cat 更合理
                segs_clip_s, segs_dino_s, cam_s, attn_loss_s = model(_inputs_cat, cls_labels_cat, mode='val')
                segs_cat_s = 0.5 * (segs_dino_s + segs_clip_s)

                segs_cat_s = F.interpolate(segs_cat_s, size=(hh, ww), mode='bilinear', align_corners=False)
                _segs_s = (segs_cat_s[0, ...] + segs_cat_s[1, ...].flip(-1)) / 2
                segs_list.append(_segs_s)

        msc_segs = torch.mean(torch.stack(segs_list, dim=0), dim=0).unsqueeze(0)

        resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        seg_preds = torch.argmax(resized_segs, dim=1)

        resized_msc_segs = F.interpolate(msc_segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        msc_seg_preds = torch.argmax(resized_msc_segs, dim=1)

        cams += list(cam.detach().cpu().numpy().astype(np.int16))
        _preds += list(seg_preds.detach().cpu().numpy().astype(np.int16))
        _msc_preds += list(msc_seg_preds.detach().cpu().numpy().astype(np.int16))
        _gts += list(labels.detach().cpu().numpy().astype(np.int16))

        # Save the direct multi-scale segmentation result (before CRF).
        # This gives the same two-file representation used by CTFA:
        #   1) class-index PNG
        #   2) RGB/cmap PNG for qualitative figures
        _save_mask_pair(
            msc_seg_preds.detach().cpu().numpy()[0],
            name_str,
            args.raw_segs_dir,
            args.raw_segs_rgb_dir,
        )

        if num % 100 == 0:
            _preds_hist, seg_score = evaluate.scores(_gts, _preds, _preds_hist, num_classes=16)
            _msc_preds_hist, msc_seg_score = evaluate.scores(_gts, _msc_preds, _msc_preds_hist, num_classes=16)
            _cams_hist, cam_score = evaluate.scores(_gts, cams, _cams_hist, num_classes=16)
            _preds, _gts, _msc_preds, cams = [], [], [], []

        os.makedirs(os.path.join(args.work_dir, "logit"), exist_ok=True)
        np.save(os.path.join(args.work_dir, "logit", name_str + ".npy"),
                {"segs": segs.detach().cpu().numpy(),
                 "msc_segs": msc_segs.detach().cpu().numpy()})

    return _gts, _preds, _msc_preds, cams, _preds_hist, _msc_preds_hist, _cams_hist


def crf_proc(dataset, config):
    print("crf post-processing...")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    post_processor = DenseCRF(
        iter_max=10,
        pos_xy_std=3,
        pos_w=3,
        bi_xy_std=64,
        bi_rgb_std=5,
        bi_w=4,
    )

    preds, gts = [], []

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        name, inputs, labels, cls_labels = data
        name_str = _to_name_str(name)

        label = labels.squeeze().numpy()

        logit_path = os.path.join(args.work_dir, "logit", name_str + ".npy")
        if not os.path.isfile(logit_path):
            # 防御：如果 logits 丢了就跳过（正常修复后不应该发生）
            continue

        logit_dict = np.load(logit_path, allow_pickle=True).item()
        logit = logit_dict['msc_segs']

        image_path = os.path.join(config.dataset.root_dir, args.eval_set, "images", name_str + ".png")
        image = imageio.imread(image_path).astype(np.float32)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image_u8 = image.astype(np.uint8)
        prob = post_processor(image_u8, prob)
        pred = np.argmax(prob, axis=0)

        # Save final CRF result in CTFA-like format.
        _save_mask_pair(
            pred,
            name_str,
            args.segs_dir,
            args.segs_rgb_dir,
        )

        preds.append(pred)
        gts.append(label)

    hist = np.zeros((16, 16))
    hist, score = evaluate.scores(gts, preds, hist, 16)
    print(score)


def main(cfg):
    val_dataset = iSAID.iSAIDSegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=args.eval_set,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    WeCLIP_model = WeCLIP(
        num_classes=cfg.dataset.num_classes,
        clip_model=cfg.clip_init.clip_pretrain_path,
        embedding_dim=cfg.clip_init.embedding_dim,
        in_channels=cfg.clip_init.in_channels,
        dataset_root_path=cfg.dataset.root_dir,
        device='cuda'
    )

    trained_state_dict = torch.load(args.model_path, map_location="cpu")
    WeCLIP_model.load_state_dict(state_dict=trained_state_dict, strict=False)
    WeCLIP_model.eval()

    gts, preds, msc_preds, cams, preds_hist, msc_preds_hist, cams_hist = validate(
        model=WeCLIP_model,
        dataset=val_dataset,
        test_scales=[1, 0.75]
    )
    torch.cuda.empty_cache()

    preds_hist, seg_score = evaluate.scores(gts, preds, preds_hist, 16)
    msc_preds_hist, msc_seg_score = evaluate.scores(gts, msc_preds, msc_preds_hist, 16)
    cams_hist, cam_score = evaluate.scores(gts, cams, cams_hist, 16)

    print("cams score:")
    print(cam_score)
    print("segs score:")
    print(seg_score)
    print("msc segs score:")
    print(msc_seg_score)

    crf_proc(dataset=val_dataset, config=cfg)
    return True


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    print(cfg)
    print(args)

    # results/<split>/...
    args.work_dir = os.path.join(args.work_dir, args.eval_set)

    # Logits for CRF
    os.makedirs(os.path.join(args.work_dir, "logit"), exist_ok=True)

    # Direct multi-scale prediction (before CRF)
    args.raw_segs_dir = os.path.join(args.work_dir, "seg_preds_raw")
    args.raw_segs_rgb_dir = os.path.join(args.work_dir, "seg_preds_rgb_raw")

    # Final CRF prediction -- same naming idea as the CTFA script
    args.segs_dir = os.path.join(args.work_dir, "seg_preds")
    args.segs_rgb_dir = os.path.join(args.work_dir, "seg_preds_rgb")

    for d in [
        args.raw_segs_dir,
        args.raw_segs_rgb_dir,
        args.segs_dir,
        args.segs_rgb_dir,
    ]:
        os.makedirs(d, exist_ok=True)

    print("Result directories:")
    print("  raw masks:      ", args.raw_segs_dir)
    print("  raw RGB:        ", args.raw_segs_rgb_dir)
    print("  CRF masks:      ", args.segs_dir)
    print("  CRF RGB:        ", args.segs_rgb_dir)

    main(cfg=cfg)
