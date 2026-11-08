import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
import datetime
import os
import random
import torch.nn.functional as F
from WeCLIP.utils import imutils

from WeCLIP.WeCLIP_model.PAR import PAR


# def disperse_v8(feats, index, cls_token, thres_cls_token, thres_patch_token):
#     # Compute the similarity
#     A = (feats @ feats.transpose(1, 2)).squeeze()
#     _A = A[index, :]

#     a = (cls_token @ feats.transpose(1, 2)).squeeze()

#     # print(torch.min(a), torch.max(a))
#     _a = a > thres_cls_token

#     _A = _A > thres_patch_token

#     dis_res = _A.sum(0) * _a

#     return dis_res

def disperse_v8(feats, index, cls_token, thres_cls_token, thres_patch_token):
    """
    feats: [1, N, C] 
    index: [K]  (long)
    cls_token: [1, C] 
    """
    S = (feats @ feats.transpose(1, 2)).squeeze(0)         # [N,N]
    a = (cls_token @ feats.transpose(1, 2)).squeeze(0)     # [N]

    anchor_rows = S[index, :]                              # [K,N]
    mask_patch = (anchor_rows > thres_patch_token).float() # [K,N]
    mask_cls   = (a > thres_cls_token).float()             # [N]

    dis_res = mask_patch.sum(0) * mask_cls                 # [N]
    return dis_res


def disperse_bg(feats, index, bg_thres):
    """
    feats: [1, N, C]
    index: [K]
    """
    S = (feats @ feats.transpose(1, 2)).squeeze(0)         # [N,N]
    anchor_rows = S[index, :]                              # [K,N]
    mask_patch = (anchor_rows > bg_thres).float()          # [K,N]
    dis_res = mask_patch.sum(0)                            # [N]
    return dis_res


@torch.no_grad()
def get_pseudo_labels_rv_cam_10_4(
    model_dino, inputs, cls, cams_list, reps, refine_mod, scales, img_box, memobank,
    thres_cls_token=0.2, thres_patch_token=0.02, thres_bg_token=0.03,
    thres_bg_aff_l=0.1, thres_bg_aff_h=0.5, patch_size=16, crop_size=320,
    topk=0.7, momentum=0.99, num_class=16
):
    """
    约定：
      - cams_list: [B, C_nonbg, Hc, Wc]，已经是 PAR refine 后且映射到 15 个前景通道（不含背景）
      - cls: [B, 15] 的 0/1 多标签
    """
    device = inputs.device
    b, _, h, w = inputs.shape
    ps = patch_size

    # DINO half 前向；相似度/亲和一律 float32
    model_dino = model_dino.half().eval()
    inputs = inputs.half()

    # 拼背景标签通道
    labels = torch.cat([torch.ones(b, 1, device=device), cls.to(device)], dim=1)  # [B, 1+15]

    final_res_list = []
    tau_cam = 0.05        # 更松的 CAM 种子阈值
    margin_top2 = 0.05    # “几乎平手”的软边际

    for i in range(b):
        target_h, target_w = int(h / ps), int(w / ps)
        reps_i = F.interpolate(reps[i].unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False)
        reps_i_flat = reps_i.reshape(1, reps_i.size(1), -1)  # [1, D, Th*Tw]
        cls_valid_num = int(torch.count_nonzero(labels[i]).item())

        res_list = []
        fallback_maps_by_class = {}

        for sc in scales:
            _H, _W = int(sc * h), int(sc * w)
            th, tw = int(_H / ps), int(_W / ps)

            _inputs = F.interpolate(inputs[i].unsqueeze(0), size=(_H, _W), mode='bilinear', align_corners=False)

            # 前景 CAM（15通道）与存在掩码
            cams_fg = cams_list[i].unsqueeze(0).to(device).half()      # [1, 15, Hc, Wc]
            cams_fg = cams_fg * cls[i].view(1, -1, 1, 1).half()

            # 背景通道
            _bg_cam = (1.0 - torch.clamp(cams_fg.max(dim=1, keepdim=True).values, 0.0, 1.0)).half()  # [1,1,Hc,Wc]
            cams = torch.cat([_bg_cam, cams_fg], dim=1)  # [1, 1+15, Hc, Wc]

            # 下采到 token 网格
            cams_ds = F.interpolate(cams, size=(th, tw), mode='bilinear', align_corners=False)  # [1,C,th,tw]
            cams_max_index = cams_ds.argmax(dim=1)  # [1, th, tw]

            # DINO qkv（float32）
            _, _, qkv = model_dino.get_intermediate_feat(_inputs, n=1)
            qkv = qkv[0]
            q = qkv[0].transpose(1, 2).reshape(1, th * tw + 1, -1).float()
            k = qkv[1].transpose(1, 2).reshape(1, th * tw + 1, -1).float()

            feats = F.normalize(k[:, 1:, :], dim=-1, p=2)      # [1, th*tw, C]
            cls_token = F.normalize(q[:, 0, :], dim=-1, p=2)   # [1, C]

            class_indices = torch.nonzero(labels[i], as_tuple=False).flatten().tolist()
            C_here = cams_ds.size(1)
            k_top = 2 if C_here >= 2 else 1

            # 预先取每位置 top2（用于软赢家/runner-up）
            vals_top2, idx_top2 = torch.topk(cams_ds[0], k=k_top, dim=0)  # [k_top, th, tw], [k_top, th, tw]

            for ci in class_indices:
                ci = int(ci)
                scores_map = cams_ds[0, ci]  # [th,tw]

                is_winner = (idx_top2[0] == ci)  # [th,tw]
                if k_top == 2:
                    is_runner_close = (idx_top2[1] == ci) & ((vals_top2[0] - vals_top2[1].clamp_min(0)) <= margin_top2)
                else:
                    is_runner_close = torch.zeros_like(is_winner, dtype=torch.bool)

                # 软赢家 或 CAM 超阈值
                seed_mask = is_winner | is_runner_close | (scores_map > tau_cam)
                scores = (scores_map * seed_mask.float()).view(-1)      # [th*tw]
                num_tokens = scores_map.numel()

                # 选择锚点（更多且更分散）
                if int((scores > 0).sum().item()) == 0:
                    # 兜底：从 CAM 选全局 top-k，并加少量随机探索
                    k_anchor = max(128, round(num_tokens * 0.04))
                    k_anchor = min(k_anchor, num_tokens)
                    base_scores = scores_map.view(-1)
                    top_vals, top_idx = torch.topk(base_scores, k=k_anchor)
                    extra = max(0, round(k_anchor * 0.25))
                    if extra > 0:
                        prob = base_scores - base_scores.min()
                        s = prob.sum()
                        if s <= 1e-6:
                            prob = torch.ones_like(base_scores) / base_scores.numel()
                        else:
                            prob = prob / s
                        rand_idx = torch.multinomial(prob, num_samples=extra, replacement=False)
                        anchor_index = torch.unique(torch.cat([top_idx, rand_idx], dim=0))[:k_anchor]
                    else:
                        anchor_index = top_idx
                else:
                    nz = int((scores > 0).sum().item())
                    k_anchor = max(32, round(nz * topk))
                    k_anchor = min(k_anchor, scores.numel())
                    vals, idx = torch.topk(scores, k=k_anchor)
                    jitter = torch.rand_like(vals) * 1e-3
                    _, order = torch.topk(vals + jitter, k=k_anchor)
                    anchor_index = idx[order]

                # 扩散
                if ci != 0:
                    res = disperse_v8(
                        feats=feats, index=anchor_index, cls_token=cls_token,
                        thres_cls_token=thres_cls_token, thres_patch_token=thres_patch_token
                    ).reshape(th, tw).unsqueeze(0).unsqueeze(0).float()
                else:
                    res = disperse_bg(
                        feats=feats, index=anchor_index, bg_thres=thres_bg_token
                    ).reshape(th, tw).unsqueeze(0).unsqueeze(0).float()

                # 归一化
                res = res + F.adaptive_max_pool2d(-res, (1, 1))
                res = res / (F.adaptive_max_pool2d(res, (1, 1)) + 1e-5)

                # —— 兜底#1：该尺度扩散结果全 0 → 用 CAM 替代 —— 
                if res.max().item() <= 1e-8:
                    alt = scores_map.float()
                    alt = alt - alt.min()
                    m = alt.max()
                    if m > 1e-6:
                        alt = alt / m
                    else:
                        alt.zero_()
                    res = alt.unsqueeze(0).unsqueeze(0)  # [1,1,th,tw]
                    print(f"[fallback#1] image {i}, class {ci}, scale {sc}: use CAM map as res")

                # 记录用于兜底#2的 CAM 上采样
                up_cam = F.interpolate(scores_map.unsqueeze(0).unsqueeze(0),
                                       size=(target_h, target_w), mode='bilinear', align_corners=False)[0, 0]
                fallback_maps_by_class.setdefault(ci, []).append(up_cam)  # [Th,Tw]

                # 上采到统一 token 网格
                res = F.interpolate(res, size=(target_h, target_w), mode='bilinear', align_corners=False)
                res_list.append(res)  # [1,1,Th,Tw]

        # 跨尺度融合到 [C, Th, Tw]
        final_res_i = torch.zeros(num_class, target_h, target_w, device=device).float()
        class_indices = torch.nonzero(labels[i], as_tuple=False).flatten().tolist()
        n_cls_i = len(class_indices)

        for j, ci in enumerate(class_indices):
            ci = int(ci)
            tmp = []
            for k_sc in range(len(scales)):
                idx = j + k_sc * n_cls_i
                tmp.append(res_list[idx])  # [1,1,Th,Tw]

            a = torch.sum(torch.stack(tmp, dim=0), dim=0)  # [1,1,Th,Tw]
            a = a + F.adaptive_max_pool2d(-a, (1, 1))
            a = a / (F.adaptive_max_pool2d(a, (1, 1)) + 1e-5)
            a = a[0, 0]  # [Th,Tw]

            # —— 兜底#2：融合后仍全 0 → 用“各尺度 CAM 的和” —— 
            if a.max().item() <= 1e-8 and ci in fallback_maps_by_class and len(fallback_maps_by_class[ci]) > 0:
                a_fb = torch.stack(fallback_maps_by_class[ci], dim=0).sum(0)  # [Th,Tw]
                a_fb = a_fb - a_fb.min()
                m = a_fb.max()
                if m > 1e-6:
                    a = a_fb / m
                else:
                    a = a_fb
                print(f"[fallback#2] image {i}, class {ci}: use sum(CAM) across scales")

            if ci != 0:
                # 记忆库门控（温和指数）
                memo_rep = memobank[ci - 1].to(device=device, dtype=torch.float32).view(-1, 1, 1).expand(-1, target_h, target_w)
                reps_i_n = F.normalize(reps_i[0].float(), dim=0, p=2)
                memo_rep = F.normalize(memo_rep, dim=0, p=2)

                if cls_valid_num > 2:
                    sim = (memo_rep * reps_i_n).sum(dim=0)  # [Th,Tw]
                    sim = sim.clamp_min(0.0)
                    sim = sim ** 1.2
                    sim = sim + F.adaptive_max_pool2d(-sim.unsqueeze(0).unsqueeze(0), (1, 1))[0, 0]
                    sim = sim / (F.adaptive_max_pool2d(sim.unsqueeze(0).unsqueeze(0), (1, 1))[0, 0] + 1e-5)
                    a = sim * a

                # 更新记忆库（top-8 聚合）
                flat_a = a.reshape(-1)
                k_anchor = min(8, flat_a.numel())
                _, anchor_reps_idx = torch.topk(flat_a, k=k_anchor)
                reps_i_c = reps_i_flat.permute(0, 2, 1)[:, anchor_reps_idx, :][0].mean(0).float()
                memobank[ci - 1] = (momentum * memobank[ci - 1].float() + (1 - momentum) * reps_i_c).detach()
            else:
                # 背景夹紧
                a = torch.clamp(a, min=thres_bg_aff_l, max=thres_bg_aff_h)

            final_res_i[ci] = a  # [Th,Tw]

        final_res_list.append(final_res_i)

    final_res = torch.stack(final_res_list, dim=0)  # [B,C,Th,Tw]
    final_res = torch.nan_to_num(final_res, nan=0.0, posinf=0.0, neginf=0.0)

    # ======= 诊断打印（refinement 之前） =======
    per_class_max = final_res.float().amax(dim=(2, 3))  # [B,C]
    printable = [[f"{v:.4f}" for v in row] for row in per_class_max.cpu().tolist()]
    print("per-class max (每图每类最大值):")
    for bi, row in enumerate(printable):
        print(f"  Image {bi}: {row}")

    argmax_map = final_res.argmax(dim=1)  # [B,Th,Tw]
    hist = torch.bincount(argmax_map.view(-1), minlength=num_class)
    print("argmax分布:", hist.cpu().tolist())

    fg_max = final_res[:, 1:, :, :].amax().item() if num_class > 1 else 0.0
    bg_max = final_res[:, 0, :, :].amax().item()
    print(f"前景最大值={fg_max:.4f}, 背景最大值={bg_max:.4f}")

    # ==================== refinement ====================
    down_scale = 2
    Hs, Ws = h // down_scale, w // down_scale
    final_res_small = F.interpolate(final_res, size=(Hs, Ws), mode='bilinear', align_corners=False)  # [B,C,Hs,Ws]

    has_fg_signal = (fg_max > 1e-6) and ((argmax_map != 0).any().item())
    if not has_fg_signal:
        print("[warn] no foreground signal -> skip refine_mod()")
        _refined_cams = final_res_small
    else:
        _inputs_small = F.interpolate(inputs, size=(Hs, Ws), mode='bilinear', align_corners=False)
        input_res = imutils.denormalize_img2(_inputs_small)
        try:
            _refined_cams = refine_mod(input_res, final_res_small)  # [B,C,Hs,Ws]
        except Exception as e:
            print(f"[warn] refine_mod failed: {repr(e)} -> fallback to raw scores")
            _refined_cams = final_res_small

    _refined_cams = torch.nan_to_num(_refined_cams, nan=0.0, posinf=0.0, neginf=0.0)
    _refined_cams = F.interpolate(_refined_cams, size=(h, w), mode='bilinear', align_corners=False)
    _pseudo_label = torch.argmax(_refined_cams, dim=1)  # [B,H,W]

    pseudo_label = torch.full_like(_pseudo_label, 255)
    for idx, coord in enumerate(img_box):
        r0, r1, c0, c1 = map(int, coord)
        pseudo_label[idx, r0:r1, c0:c1] = _pseudo_label[idx, r0:r1, c0:c1]

    return pseudo_label