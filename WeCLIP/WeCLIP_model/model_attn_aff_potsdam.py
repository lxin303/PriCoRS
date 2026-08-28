import torch
import torch.nn as nn
from WeCLIP.WeCLIP_model.segformer_head import SegFormerHead
import numpy as np
import WeCLIP.clip as clip
from wsss.clip.clip_text import class_names_potsdam, BACKGROUND_CATEGORY_potsdam
from WeCLIP.pytorch_grad_cam import GradCAM
from WeCLIP.clip.clip_tool import perform_cam_potsdam, generate_cam_label, generate_clip_fts
# from torchvision.transforms import Compose, Normalize
from .Decoder.TransDecoder import DecoderTransformer
from WeCLIP.WeCLIP_model.PAR import PAR
import torch.nn.functional as F
from WeCLIP.WeCLIP_model.load_attr import attr_aggregate 
from scipy import ndimage as ndi
# from skimage.feature import peak_local_max

    
def reshape_transform(tensor, height=20, width=20):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            print(texts)
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.t()


def _refine_cams(ref_mod, images, cams, valid_key):
    images = images.unsqueeze(0)
    cams = cams.unsqueeze(0)

    refined_cams = ref_mod(images.float(), cams.float())
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label.squeeze(0)


class WeCLIP(nn.Module):
    def __init__(self, num_classes=None, clip_model=None, dino_model=None, embedding_dim=256, 
                 in_channels=512, dataset_root_path=None, clip_flag=16, device='cuda'):
        
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.dino_fts_fuse_dim = 384
        self.clip_flag = clip_flag

        self.encoder, _ = clip.load(clip_model, device=device)

        for name, param in self.encoder.named_parameters():
            if clip_flag == 14 and '23' not in name:
                param.requires_grad=False
            if clip_flag == 16 and "11" not in name:
                 param.requires_grad=False

        for name, param in self.encoder.named_parameters():
            print(name, param.requires_grad)

        self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        for name, param in self.dino_encoder.named_parameters():
            param.requires_grad = False

        self.in_channels = in_channels

        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels, embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=11)
        self.dino_decoder_fts_fuse = SegFormerHead(in_channels=[self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim], embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=1)
        
        self.decoder = DecoderTransformer(width=self.embedding_dim, layers=3, heads=8, output_dim=self.num_classes)
        
        self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY_potsdam, ['an overhead satellite origami {}.'],
                                               self.encoder) 
        self.fg_text_features = zeroshot_classifier(class_names_potsdam, ['an overhead satellite origami {}.'],
                                               self.encoder) # (5, 512) 

        # self.fg_text_features_attr, self.attr_flag = attr_aggregate(self.fg_text_features, self.encoder,'potsdam', num_atrr_clusters=56, json_file='/root/autodl-tmp/WeCLIP/attributes_text/gtp4.0_cluster_potsdam.json')

        self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]

        self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers, reshape_transform=reshape_transform)
        self.encoder.eval()
        
        self.par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda()
        self.iter_num = 0
        self.require_all_fts = True
        self.proj_dim = 256  
        
        self.cam_bg_thres = 1
        
        self.proto_dim = self.embedding_dim        
        self.momentum = 0.9                          
        self.proto_mix_alpha = 0.5            
        self.memobank = F.normalize(
            torch.randn(self.num_classes - 1, self.proto_dim, device=device),
            dim=1
        )
        self.use_bg_proto = False


    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)
        for param in list(self.dino_decoder_fts_fuse.parameters()):
            param_groups[3].append(param)
        # for param in list(self.proto_head.parameters()):
        #     param_groups[3].append(param)
        return param_groups    
    

    def get_seed(self, norm_cam, label, feature):
        # feature: [B, C, Hf, Wf]
        label = label.cuda()
        B, C, Hf, Wf = feature.shape

        # 1) 拼背景通道并对齐到 feature 网格
        bg = (1 - norm_cam.max(dim=1, keepdim=True)[0]).clamp_min(0.)
        norm_cam = torch.cat([bg, norm_cam], dim=1)                             # [B, 1+K, H', W']
        norm_cam = F.interpolate(norm_cam, size=(Hf, Wf), mode='bilinear', align_corners=False)

        # 2) 图像级标签也拼背景（与 seeds 的类别维对齐）
        assert label.shape[0] == B
        label = torch.cat(
            [torch.ones(B, 1, device=label.device, dtype=label.dtype), label], dim=1
        ).float()                                                                # [B, 1+K]

        n, c, h, w = norm_cam.shape

        # 3) 余弦亲和（像素/patch 两两）
        feat_seq = feature.view(n, C, h * w)
        feat_seq = feat_seq / (feat_seq.norm(dim=1, keepdim=True) + 1e-5)
        correlation = torch.matmul(feat_seq.transpose(2, 1), feat_seq)           # [B, HW, HW]
        correlation = F.relu(correlation, inplace=True).unsqueeze(1)             # [B, 1, HW, HW]

        # 4) mIoU 打分
        cam_flatten = norm_cam.view(n, c, h * w).unsqueeze(2)                    # [B, C, 1, HW]
        inter = (correlation * cam_flatten).sum(-1)                              # [B, C, HW]
        union = correlation.sum(-1) + cam_flatten.sum(-1) - inter
        miou = (inter / (union + 1e-6)).view(n, c, h, w)                         # [B, C, H, W]

        # 背景降权（可根据数据调）
        miou[:, 0] = miou[:, 0] * 0.5

        probs = F.softmax(miou, dim=1)                                           # [B, C, H, W]
        belonging = miou.argmax(1)                                               # [B, H, W]

        # 5) 生成 one-hot 种子，并用图像级标签掩码
        seeds = torch.zeros((n, c, h, w), device=feature.device, dtype=feature.dtype)
        seeds = seeds.scatter_(1, belonging.unsqueeze(1), 1.0)                   # [B, C, H, W]
        seeds = seeds * label.unsqueeze(-1).unsqueeze(-1)                        # [B, C, H, W]
        return seeds, probs, norm_cam


    def get_prototype(self, seeds, feature):
        n,c,h,w = feature.shape
        seeds = F.interpolate(seeds, feature.shape[2:], mode='nearest')
        crop_feature = seeds.unsqueeze(2) * feature.unsqueeze(1)  # seed:[n,21,1,h,w], feature:[n,1,c,h,w], crop_feature:[n,21,c,h,w]
        prototype = F.adaptive_avg_pool2d(crop_feature.view(-1,c,h,w), (1,1)).view(n, self.num_classes, c, 1, 1) # prototypes:[n,21,c,1,1]
        return prototype


    def reactivate(self, prototype, feature):
        IS_cam = F.relu(torch.cosine_similarity(feature.unsqueeze(1), prototype, dim=2)) # feature:[n,1,c,h,w], prototypes:[n,21,c,1,1], crop_feature:[n,21,h,w]
        IS_cam = F.interpolate(IS_cam, feature.shape[2:], mode='bilinear', align_corners=True)
        return IS_cam
    
    
    def minmax_norm(self, x: torch.Tensor, eps: float = 1e-5):
        # x: [B, C, H, W]
        x_flat = x.flatten(2)  # [B, C, HW]
        x_min = x_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)  # [B,C,1,1]
        x_max = x_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)  # [B,C,1,1]
        return (x - x_min) / (x_max - x_min + eps)


    def _global_proto_tensor(self, B: int, device: torch.device):
        # memobank: [K_fg, C] -> [B, 1+K_fg, C, 1, 1]
        bg = torch.zeros(1, 1, self.proto_dim, 1, 1, device=device)
        fg = self.memobank.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1,K_fg,C,1,1]
        proto = torch.cat([bg, fg], dim=1)  # [1,K_all,C,1,1]
        return proto.expand(B, -1, -1, -1, -1)


    def _proto_similarity(self, fts: torch.Tensor):
        # fts: [B, C, Hf, Wf]
        B = fts.shape[0]
        proto = self._global_proto_tensor(B, fts.device)  # [B, K_all, C, 1, 1]
        # S: [B, K_all, Hf, Wf]，和论文一样 ReLU(cos)
        S = F.relu(torch.cosine_similarity(fts.unsqueeze(1), proto, dim=2))
        return S


    def update_and_mix_prototypes(self, prototypes: torch.Tensor, seeds: torch.Tensor, fts: torch.Tensor):
    
        B, K_all, C, _, _ = prototypes.shape
        device = fts.device

        with torch.no_grad():
            # 原型去掉空间维，并做 L2 归一化
            proto_local = prototypes.squeeze(-1).squeeze(-1)            # [B, K_all, C]
            proto_local = F.normalize(proto_local, dim=-1)

            # 每类像素计数，避免空类污染
            pixel_count = seeds.sum(dim=(-1, -2))                       # [B, K_all]
            exist_mask = (pixel_count > 0)                              # [B, K_all]

            for k in range(1, self.num_classes):  # 5 是背景
                valid = exist_mask[:, k]
                if valid.any():
                    p_mean = proto_local[valid, k].mean(dim=0)          # [C]
                    p_mean = F.normalize(p_mean, dim=0)
                    # memobank 中按 (k-1) 存前景类
                    self.memobank[k - 1] = F.normalize(
                        self.momentum * self.memobank[k - 1] + (1.0 - self.momentum) * p_mean,
                        dim=0
                    )

        if self.use_bg_proto:
            bg_global = torch.zeros(1, 1, self.proto_dim, 1, 1, device=device)
        else:
            bg_global = torch.zeros(1, 1, self.proto_dim, 1, 1, device=device)

        fg_global = self.memobank.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, K_fg, C, 1, 1]
        proto_global = torch.cat([bg_global, fg_global], dim=1)             # [1, K_all, C, 1, 1]
        proto_global = proto_global.expand(B, -1, -1, -1, -1)               # [B, K_all, C, 1, 1]

        # 局部/全局融合并归一化
        proto_mix = self.proto_mix_alpha * prototypes + (1.0 - self.proto_mix_alpha) * proto_global
        proto_mix = F.normalize(proto_mix.squeeze(-1).squeeze(-1), dim=-1).unsqueeze(-1).unsqueeze(-1)

        # --------- 3) 用融合原型进行再激活（余弦相似 + ReLU） ---------
        IS_cam = self.reactivate(proto_mix, fts) 

        return proto_mix, IS_cam


    def forward(self, img, img_labels=None, mode='train'):
        
        cam_list = []
        all_cam_list = []
        b, c, h, w = img.shape
        self.iter_num += 1

        fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True, clip_flag=self.clip_flag)

        with torch.no_grad():
            dino_img_h, dino_img_w = (h//14)*14, (w//14)*14
            dino_img = F.interpolate(img, size=(dino_img_h, dino_img_w), mode='bilinear', align_corners=False)
            dino_ftses = self.dino_encoder.forward_features(dino_img)
            dino_fts = dino_ftses['x_norm_patchtokens']
            
        fts_all_stack = torch.stack(fts_all, dim=0)  # (11, hw, b, c)
        attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(1, 0, 2, 3)

        if self.require_all_fts == True:
            cam_fts_all = fts_all_stack[-2].unsqueeze(0).permute(2, 1, 0, 3)  # (1, hw, 1, c)
        else:
            cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)

        all_img_tokens = fts_all_stack[:, 1:, ...]
        img_tokens_channel = all_img_tokens.size(-1)
        all_img_tokens = all_img_tokens.permute(0, 2, 3, 1)
        all_img_tokens = all_img_tokens.reshape(-1, b, img_tokens_channel, h // 16, w // 16)  # (11, b, c, h, w)
        # all_img_tokens = all_img_tokens[-1].unsqueeze(0)

        fts = self.decoder_fts_fuse(all_img_tokens)
        _, _, fts_h, fts_w = fts.shape

        if isinstance(dino_fts, list):
            for d_i, dino_fts_single in enumerate(dino_fts):
                dino_fts_single = dino_fts_single.reshape([b, dino_img_h // 14, dino_img_w // 14, -1]).permute(0, 3, 1, 2)
                dino_fts[d_i] = dino_fts_single
            
            dino_fts = torch.stack(dino_fts)
            dino_fts = self.dino_decoder_fts_fuse(dino_fts)
            dino_h, dino_w = dino_img_h // 14, dino_img_w // 14
        else:
            dino_fts = dino_fts.reshape([b, dino_img_h//14, dino_img_w//14, -1]).permute(0,3,1,2)
            _, _, dino_h, dino_w = dino_fts.shape #32
            dino_fts = self.dino_decoder_fts_fuse(dino_fts.unsqueeze(0))
        
        dino_fts = F.interpolate(dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)

        seg_clip, seg_attn_weight_list_clip = self.decoder(fts)
        seg_dino, seg_attn_weight_list_dino = self.decoder(dino_fts)

        clip_dino_fts = torch.cat([fts, dino_fts], dim=1)
        seg_dino_prob = F.softmax(0.5*seg_dino+0.5*seg_clip, dim=1)
        seg_dino_prob = seg_dino_prob.detach()

        attn_fts = F.interpolate(clip_dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)
        f_b, f_c, f_h, f_w = attn_fts.shape
        attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
        attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
        attn_pred = torch.sigmoid(attn_pred)

        for i, img_label in enumerate(img_labels):
            img_i = img[i]
            cam_fts = cam_fts_all[i]
            cam_attn = attn_weight_stack[i]
            seg_attn = attn_pred.unsqueeze(0)[:, i, :, :]

            require_seg_trans = True
            seg_dino_cam = seg_dino_prob[i]

            cam_refined_list, keys, w, h = perform_cam_potsdam(img_label, img_i, cam_fts, cam_attn, seg_attn,
                                                        self.fg_text_features, self.bg_text_features,
                                                        self.grad_cam, mode=mode, require_seg_trans=require_seg_trans,
                                                        seg_dino_cam=seg_dino_cam, clip_flag = self.clip_flag) #[B, 20,20]
            if len(keys) == 0:
                cams = torch.ones((1, h, w), device=img_i.device)  # 只有背景通道
                valid_key = torch.tensor([0], dtype=torch.long, device=img_i.device)
            else:
                cam_dict = generate_cam_label(cam_refined_list, keys, w, h)
                cams = cam_dict['refined_cam'].to(img_i.device)     # [K,H',W']
                bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdim=True)[0], self.cam_bg_thres)
                cams = torch.cat([bg_score, cams], dim=0)           # [1+K,H',W']
                valid_key = np.pad(np.array(cam_dict['keys'], dtype=np.int64) + 1, (1, 0), mode='constant')
                # valid_key = np.pad(np.array(cam_dict['keys'], dtype=np.int64), (1, 0),mode='constant', constant_values=5)                
                valid_key = torch.from_numpy(valid_key).cuda()
            
            with torch.no_grad():
                cam_labels = _refine_cams(self.par, img[i], cams, valid_key)
            cams_full = torch.zeros((self.num_classes-1, h, w), dtype=cams.dtype, device=img_i.device)
            
            if len(keys) != 0: 
                for idx, class_id in enumerate(cam_dict['keys']):#[3,7]
                    cams_full[class_id] = cams[idx + 1]

            cam_list.append(cam_labels)
            all_cam_list.append(cams_full)
        
        all_cam_labels = torch.stack(cam_list, dim=0) #[B, 320, 320]
        # proj_feats = F.normalize(self.proto_head(fts), dim=1)
        all_cam_no_refine_labels = torch.stack(all_cam_list, dim=0) #[B, 5, 320, 320]

        # first update mambank
        seeds_0, probs_0, norm_cam_0 = self.get_seed(all_cam_no_refine_labels.clone(), img_labels.clone(), fts.detach())
        prototypes_0 = self.get_prototype(seeds_0, fts)
        _, IS_cam_0 = self.update_and_mix_prototypes(prototypes_0, seeds_0, fts)

        S = self._proto_similarity(fts)  # 包含 bg+fg

        Hc, Wc = all_cam_no_refine_labels.shape[-2:]
        S_up = F.interpolate(S, size=(Hc, Wc), mode='bilinear', align_corners=False)  # [B, K_all, Hc, Wc]

        Mf_fg = all_cam_no_refine_labels.clamp_min(0.)
        Mf_bg = (1 - Mf_fg.max(dim=1, keepdim=True)[0]).clamp_min(0.)   # [B,1,Hc,Wc]
        Mf_all = torch.cat([Mf_bg, Mf_fg], dim=1)                       # [B,K_all,Hc,Wc]

        M_proto_all = self.minmax_norm(Mf_all * (S_up ** 2), eps=1e-5)

        M_proto_fg = M_proto_all[:, 1:]  # [B,K_fg,Hc,Wc]

        seeds, probs, norm_cam = self.get_seed(M_proto_fg.clone(), img_labels.clone(), fts.detach())
        prototypes = self.get_prototype(seeds, fts)
        proto_mix, IS_cam = self.update_and_mix_prototypes(prototypes, seeds, fts)

        if mode=='train':
            return seg_clip, seg_dino, all_cam_labels, attn_pred, norm_cam, IS_cam
            # return seg_clip, seg_dino, all_cam_labels, attn_pred
        else:
            return seg_clip, seg_dino, all_cam_labels, attn_pred

