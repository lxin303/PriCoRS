import os
import torch
import torch.nn as nn
from WeCLIP.WeCLIP_model.segformer_head import SegFormerHead
import numpy as np
import WeCLIP.clip as clip
from WeCLIP.clip.clip_text import class_names_isaid, BACKGROUND_CATEGORY
from WeCLIP.pytorch_grad_cam import GradCAM
from WeCLIP.clip.clip_tool import perform_cam, generate_cam_label, generate_clip_fts
from .Decoder.TransDecoder import DecoderTransformer
from WeCLIP.WeCLIP_model.PAR import PAR
import torch.nn.functional as F
from WeCLIP.WeCLIP_model.load_attr import attr_aggregate 
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from segment_anything import sam_model_registry


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
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.t()


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


@torch.no_grad()
def torch_kmeans_cosine(x: torch.Tensor, k: int, iters: int = 30):
    """
    x: [N, D], assumed normalized
    returns:
        centers: [k, D]
        assign:  [N]
    """
    N, D = x.shape
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= N:
        return x.clone(), torch.arange(N, device=x.device)

    # farthest-point style init under cosine distance
    idx0 = torch.randint(0, N, (1,), device=x.device).item()
    centers = [x[idx0]]

    min_dist = 1 - (x @ centers[0].unsqueeze(1)).squeeze(1)
    for _ in range(1, k):
        idx = torch.argmax(min_dist).item()
        centers.append(x[idx])
        dist_new = 1 - (x @ centers[-1].unsqueeze(1)).squeeze(1)
        min_dist = torch.minimum(min_dist, dist_new)

    centers = torch.stack(centers, dim=0)

    for _ in range(iters):
        sim = x @ centers.t()
        assign = sim.argmax(dim=1)

        new_centers = []
        for i in range(k):
            mask = (assign == i)
            if mask.any():
                c = x[mask].mean(dim=0)
                c = l2norm(c, dim=0)
            else:
                c = x[torch.randint(0, N, (1,), device=x.device).item()]
            new_centers.append(c)

        new_centers = torch.stack(new_centers, dim=0)
        if torch.allclose(new_centers, centers, atol=1e-4):
            centers = new_centers
            break
        centers = new_centers

    sim = x @ centers.t()
    assign = sim.argmax(dim=1)
    return centers, assign


def _refine_cams(ref_mod, images, cams, valid_key):
    images = images.unsqueeze(0)
    cams = cams.unsqueeze(0)

    refined_cams = ref_mod(images.float(), cams.float())
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_cams.squeeze(0), refined_label.squeeze(0)


class WeCLIP(nn.Module):
    def __init__(self, num_classes=None, clip_model=None, dino_model=None, embedding_dim=256, 
                 in_channels=512, dataset_root_path=None, clip_flag=16, device='cuda',
                 use_anyup=True,
                 anyup_repo_path='/leonardo_work/IscrC_CropAI/RS/third_party/anyup',
                 anyup_output_size=80,
                 anyup_cam_blend=0.5,
                 anyup_similarity_floor=0.25,
                 anyup_similarity_power=2.0,
                 anyup_q_chunk_size=10):
        
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.dino_fts_fuse_dim = 1024
        self.clip_flag = clip_flag

        self.encoder, _ = clip.load(clip_model, device=device)

        for name, param in self.encoder.named_parameters():
            if clip_flag == 14 and '23' not in name:
                param.requires_grad=False
            if clip_flag == 16 and "11" not in name:
                 param.requires_grad=False

        for name, param in self.encoder.named_parameters():
            print(name, param.requires_grad)

        self.dino_encoder = torch.hub.load("/leonardo_work/IscrC_CropAI/RS/third_party/dinov2", "dinov2_vitl14", source="local", pretrained=True)

        for name, param in self.dino_encoder.named_parameters():
            param.requires_grad = False

        # frozen raw DINO features -> 80x80 -> DINO prototypes/similarity -> CAM calibration.
        # It is deliberately kept outside the segmentation decoder and global affinity path.
        self.use_anyup = bool(use_anyup)
        self.anyup_output_size = int(anyup_output_size)
        self.anyup_cam_blend = float(anyup_cam_blend)
        self.anyup_similarity_floor = float(anyup_similarity_floor)
        self.anyup_similarity_power = float(anyup_similarity_power)
        self.anyup_q_chunk_size = anyup_q_chunk_size

        if self.use_anyup:
            if not os.path.isdir(anyup_repo_path):
                raise FileNotFoundError(f'AnyUp repository was not found at: {anyup_repo_path}. ''Clone wimmerth/anyup there or pass anyup_repo_path explicitly.')

            # use_natten=False avoids adding NATTEN as a required dependency.
            self.anyup = torch.hub.load(anyup_repo_path, 'anyup_multi_backbone', source='local', use_natten=False)
            self.anyup.eval()
            for param in self.anyup.parameters():
                param.requires_grad = False
        else:
            self.anyup = None

        self.in_channels = in_channels

        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels, embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=11)
        self.dino_decoder_fts_fuse = SegFormerHead(in_channels=[self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim], embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=1)
        
        self.decoder = DecoderTransformer(width=self.embedding_dim, layers=3, heads=8, output_dim=self.num_classes)
        
        self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['an overhead satellite origami {}.'],
                                               self.encoder) 
        self.fg_text_features = zeroshot_classifier(class_names_isaid, ['an overhead satellite origami {}.'],
                                               self.encoder)  # (20, 512)

        self.fg_text_features_attr, self.attr_flag = attr_aggregate(self.fg_text_features, self.encoder,'isaid', num_atrr_clusters=112, json_file='/leonardo_work/IscrC_CropAI/RS/WeCLIP/attributes_text/gpt4.0_cluster_isaid.json')

        self.use_cluster_tpe = True
        self.cluster_json_file = '/leonardo_work/IscrC_CropAI/RS/WeCLIP/attributes_text/gpt4.0_cluster_isaid.json'

        self.num_semantic_clusters = min(6, self.num_classes - 1)
        self.num_global_text_prototypes = 64
        self.num_cluster_text_prototypes = 24

        self.cluster_temperature = 10.0
        self.cluster_top_m = 2
        self.global_proto_topk = 6
        self.cluster_proto_topk = 6

        self.lambda_cluster_anchor = 0.30
        self.lambda_global_proto = 0.25
        self.lambda_cluster_proto = 0.40
        self.lambda_attr_blend = 0.45

        self._init_cluster_aware_tpe_from_json(
            json_file=self.cluster_json_file,
            classnames=class_names_isaid
        )

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

        text_dim = self.fg_text_features_cluster.shape[1]

        self.text_proto_proj = nn.Sequential(
            nn.Linear(text_dim, self.proto_dim, bias=False),
            nn.LayerNorm(self.proto_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.proto_dim, self.proto_dim, bias=True),)

        self.text_proto_fuse_beta = 0.05
        self.text_proto_warmup_start = 18000
        self.text_proto_warmup_end = 24000

        self.th_multi = 0.5
        sam_path = '/leonardo_work/IscrC_CropAI/RS/pretrained/sam_vit_h.pth'
        self.net_sam = sam_model_registry['vit_h'](checkpoint=sam_path)

        # SAM configuration
        self.use_sam = True
        self.sam_start_iter = 6000
        self.sam_interval = 4

        self.sam_input_size = 1024

        self.sam_component_thres = 0.45
        self.sam_seed_thres = 0.65

        self.sam_min_component_area = 12
        self.sam_max_component_ratio = 0.50

        self.sam_min_mask_area = 16
        self.sam_max_mask_ratio = 0.60

        self.sam_accept_score = 0.22
        self.sam_pixel_score = 0.22
        self.sam_bg_score = 0.15

        self.sam_ignore_index = 255
        self.sam_max_components = 12

        self.net_sam.eval()

        for param in self.net_sam.parameters():
            param.requires_grad = False


    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)
        for param in list(self.dino_decoder_fts_fuse.parameters()):
            param_groups[3].append(param)
        for param in list(self.text_proto_proj.parameters()):
            param_groups[3].append(param)
        return param_groups    
    

    @torch.no_grad()
    def _encode_text_list(self, texts, batch_size=128):
        device = next(self.encoder.parameters()).device
        feats = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            token = clip.tokenize(batch_texts).to(device)
            emb = self.encoder.encode_text(token).float()
            emb = l2norm(emb, dim=1)
            feats.append(emb)
        return torch.cat(feats, dim=0)


    @torch.no_grad()
    def _topk_weighted_proto(self, query: torch.Tensor, bank: torch.Tensor, topk: int):

        if bank is None or bank.numel() == 0:
            return l2norm(query, dim=1)

        k = min(topk, bank.shape[0])
        sim = query @ bank.t()                  # [N, M]
        val, idx = sim.topk(k, dim=-1)          # [N, k]
        weights = F.softmax(self.cluster_temperature * val, dim=-1)
        selected = bank[idx]                    # [N, k, D]
        proto = (weights.unsqueeze(-1) * selected).sum(dim=1)
        proto = l2norm(proto, dim=1)
        return proto


    @torch.no_grad()
    def _build_cluster_proto_banks(self, desc_feats, desc_class_ids, class_cluster_assign):

        cluster_proto_banks = []
        desc_cluster_ids = class_cluster_assign[desc_class_ids]  # [N_desc]

        for g in range(self.num_semantic_clusters):
            feats_g = desc_feats[desc_cluster_ids == g]  # [Ng, D]
            if feats_g.numel() == 0:
                cluster_proto_banks.append(None)
                continue

            k_g = min(self.num_cluster_text_prototypes, feats_g.shape[0])
            if feats_g.shape[0] <= k_g:
                proto_g = feats_g
            else:
                proto_g, _ = torch_kmeans_cosine(feats_g, k_g, iters=25)
            proto_g = l2norm(proto_g, dim=1)
            cluster_proto_banks.append(proto_g)

        return cluster_proto_banks


    @torch.no_grad()
    def _init_cluster_aware_tpe_from_json(self, json_file, classnames):

        base = self.fg_text_features.detach().float()          # [K, D]
        attr = self.fg_text_features_attr.detach().float()     # [K, D]

        base = l2norm(base, dim=1)
        attr = l2norm(attr, dim=1)

        cache = torch.load(f'/leonardo_work/IscrC_CropAI/RS/WeCLIP/attributes_text/isaid_desc_gpt4.0_cluster_112_embedding_bank.pth', map_location=base.device)

        desc_feats = cache['desc_feats'].to(device=base.device, dtype=torch.float32)
        desc_class_ids = cache['desc_class_ids'].to(device=base.device, dtype=torch.long)
        class_desc_centers = cache['class_desc_centers'].to(device=base.device, dtype=torch.float32)
        desc_feats = l2norm(desc_feats.float(), dim=1)
        class_desc_centers = l2norm(class_desc_centers.float(), dim=1)

        class_semantic_centers = l2norm(
            0.25 * base + 0.25 * attr + 0.50 * class_desc_centers,
            dim=1
        )  # [K, D]

        # semantic clustering over categories
        G = min(self.num_semantic_clusters, class_semantic_centers.shape[0])
        cluster_centers, class_cluster_assign = torch_kmeans_cosine(
            class_semantic_centers, G, iters=30
        )

        # global description prototype bank
        k_global = min(self.num_global_text_prototypes, desc_feats.shape[0])
        if desc_feats.shape[0] <= k_global:
            global_desc_bank = desc_feats
        else:
            global_desc_bank, _ = torch_kmeans_cosine(desc_feats, k_global, iters=25)
        global_desc_bank = l2norm(global_desc_bank, dim=1)

        # cluster-specific description prototype banks
        cluster_proto_banks = self._build_cluster_proto_banks(
            desc_feats, desc_class_ids, class_cluster_assign
        )

        # store buffers / attrs
        self.register_buffer("text_base_bank", base)
        self.register_buffer("text_attr_bank", attr)
        self.register_buffer("text_desc_centers", class_desc_centers)
        self.register_buffer("text_class_semantic_centers", class_semantic_centers)
        self.register_buffer("text_cluster_centers", cluster_centers)
        self.register_buffer("text_class_cluster_assign", class_cluster_assign)
        self.register_buffer("text_global_desc_bank", global_desc_bank)

        # variable-length banks are stored as python list
        self.text_cluster_proto_banks = cluster_proto_banks

        # final cluster-aware class text
        cluster_text = self._build_cluster_aware_text_features()
        self.register_buffer("fg_text_features_cluster", cluster_text)


    @torch.no_grad()
    def _build_cluster_aware_text_features(self):
        """
        T*_c = Norm(
            t_c
            + λ1 * cluster_anchor
            + λ2 * global_desc_proto
            + λ3 * cluster_desc_proto
        )
        then blend with original attr-enriched text.
        """
        t = self.text_base_bank                  # [K, D]
        attr = self.text_attr_bank              # [K, D]
        centers = self.text_cluster_centers     # [G, D]
        global_bank = self.text_global_desc_bank  # [Bg, D]

        # 1) cluster routing
        q = F.softmax(self.cluster_temperature * (t @ centers.t()), dim=-1)   # [K, G]
        cluster_anchor = q @ centers                                           # [K, D]
        cluster_anchor = l2norm(cluster_anchor, dim=1)

        # 2) global description prototype retrieval
        proto_global = self._topk_weighted_proto(
            t, global_bank, self.global_proto_topk
        )  # [K, D]

        # 3) cluster-aware description prototype retrieval
        proto_cluster_list = []
        top_m = min(self.cluster_top_m, centers.shape[0])

        for c in range(t.shape[0]):
            top_clusters = q[c].topk(top_m, dim=-1).indices.tolist()
            banks = []
            for g in top_clusters:
                bank_g = self.text_cluster_proto_banks[g]
                if bank_g is not None and bank_g.numel() > 0:
                    banks.append(bank_g)

            if len(banks) == 0:
                local_bank = global_bank
            else:
                local_bank = torch.cat(banks, dim=0)

            proto_c = self._topk_weighted_proto(
                t[c:c+1], local_bank, self.cluster_proto_topk
            )[0]
            proto_cluster_list.append(proto_c)

        proto_cluster = torch.stack(proto_cluster_list, dim=0)  # [K, D]
        proto_cluster = l2norm(proto_cluster, dim=1)

        # 4) hierarchical enrichment
        out = (
            t + self.lambda_cluster_anchor * cluster_anchor + self.lambda_global_proto * proto_global + self.lambda_cluster_proto * proto_cluster
        )
        out = l2norm(out, dim=1)

        # keep consistency with your current attr aggregation branch
        out = l2norm(
            (1.0 - self.lambda_attr_blend) * out + self.lambda_attr_blend * attr,
            dim=1
        )

        out = out.to(dtype=self.fg_text_features_attr.dtype)
        return out
    

    def get_seed(self, norm_cam, label, feature):
        # feature: [B, C, Hf, Wf]
        label = label.cuda()
        B, C, Hf, Wf = feature.shape

        bg = (1 - norm_cam.max(dim=1, keepdim=True)[0]).clamp_min(0.)
        norm_cam = torch.cat([bg, norm_cam], dim=1)                             # [B, 1+K, H', W']
        norm_cam = F.interpolate(norm_cam, size=(Hf, Wf), mode='bilinear', align_corners=False)

        assert label.shape[0] == B
        label = torch.cat(
            [torch.ones(B, 1, device=label.device, dtype=label.dtype), label], dim=1
        ).float()                                                                # [B, 1+K]

        n, c, h, w = norm_cam.shape

        feat_seq = feature.view(n, C, h * w)
        feat_seq = feat_seq / (feat_seq.norm(dim=1, keepdim=True) + 1e-5)
        correlation = torch.matmul(feat_seq.transpose(2, 1), feat_seq)           # [B, HW, HW]
        correlation = F.relu(correlation, inplace=True).unsqueeze(1)             # [B, 1, HW, HW]

        cam_flatten = norm_cam.view(n, c, h * w).unsqueeze(2)                    # [B, C, 1, HW]
        inter = (correlation * cam_flatten).sum(-1)                              # [B, C, HW]
        union = correlation.sum(-1) + cam_flatten.sum(-1) - inter
        miou = (inter / (union + 1e-6)).view(n, c, h, w)                         # [B, C, H, W]

        miou[:, 0] = miou[:, 0] * 0.5

        probs = F.softmax(miou, dim=1)                                           # [B, C, H, W]
        belonging = miou.argmax(1)                                               # [B, H, W]

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


    @torch.no_grad()
    def calibrate_cam_with_anyup_dino(self, guidance_image, dino_raw, cams_fg, img_labels):
        """
        Args:
            guidance_image: ImageNet-normalized image used by DINO/AnyUp [B,3,H,W].
            dino_raw: frozen raw DINO patch feature map [B,C,h,w].
            cams_fg: foreground CAMs [B,K,Hc,Wc].
            img_labels: image-level foreground labels [B,K].

        Returns:
            calibrated_cams: residual-blended foreground CAMs [B,K,Hc,Wc].
            dino_similarity: class-wise high-resolution similarity [B,K,Hu,Wu].
            dino_highres: AnyUp DINO features [B,C,Hu,Wu].
        """
        if not self.use_anyup:
            return cams_fg, None, None

        self.anyup.eval()
        output_size = (self.anyup_output_size, self.anyup_output_size)

        dino_highres = self.anyup(
            guidance_image,
            dino_raw,
            output_size=output_size,
            q_chunk_size=self.anyup_q_chunk_size,
        )
        dino_highres = F.normalize(dino_highres.float(), dim=1, eps=1e-6)

        cam_highres = F.interpolate(
            cams_fg.float(),
            size=output_size,
            mode='bilinear',
            align_corners=False,
        ).clamp_min(0.0)

        # Remove classes that are absent according to image-level supervision.
        valid_classes = img_labels.to(
            device=cam_highres.device,
            dtype=cam_highres.dtype,
        ).unsqueeze(-1).unsqueeze(-1)
        cam_highres = cam_highres * valid_classes

        # Soft CAM-weighted DINO class prototypes, all in the original frozen DINO space.
        proto_weights = cam_highres.square()
        weight_sum = proto_weights.sum(dim=(-1, -2), keepdim=False)  # [B,K]
        prototypes = torch.einsum(
            'bkhw,bchw->bkc',
            proto_weights,
            dino_highres,
        )
        prototypes = prototypes / weight_sum.unsqueeze(-1).clamp_min(1e-6)
        prototypes = F.normalize(prototypes, dim=-1, eps=1e-6)

        dino_similarity = torch.einsum(
            'bkc,bchw->bkhw',
            prototypes,
            dino_highres,
        ).clamp_min(0.0)
        dino_similarity = self.minmax_norm(dino_similarity, eps=1e-6)

        valid_prototypes = (weight_sum > 1e-6).to(
            dino_similarity.dtype
        ).unsqueeze(-1).unsqueeze(-1)
        dino_similarity = (
            dino_similarity * valid_prototypes * valid_classes
        )

        # Residual structural calibration: similarity can refine a CAM but cannot
        # completely erase its semantic evidence.
        structural_weight = (
            self.anyup_similarity_floor
            + (1.0 - self.anyup_similarity_floor)
            * dino_similarity.pow(self.anyup_similarity_power)
        )
        structural_cam = self.minmax_norm(
            cam_highres * structural_weight,
            eps=1e-6,
        )
        structural_cam = structural_cam * valid_classes

        structural_cam = F.interpolate(
            structural_cam,
            size=cams_fg.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )

        blend = min(max(self.anyup_cam_blend, 0.0), 1.0)
        calibrated_cams = (
            (1.0 - blend) * cams_fg.float()
            + blend * structural_cam
        ).clamp(0.0, 1.0)
        calibrated_cams = calibrated_cams * valid_classes

        return calibrated_cams.to(cams_fg.dtype), dino_similarity, dino_highres


    def cams_to_dense_labels(self, cams_fg, img_labels):
        """Convert full foreground CAMs to dense labels for CAM evaluation/training."""
        valid_classes = img_labels.to(device=cams_fg.device, dtype=cams_fg.dtype).unsqueeze(-1).unsqueeze(-1)
        cams_fg = cams_fg * valid_classes
        bg = (1.0 - cams_fg.max(dim=1, keepdim=True).values).clamp_min(0.0)
        scores = torch.cat([bg, cams_fg], dim=1)
        return scores.argmax(dim=1)
    

    def _get_text_proto_beta(self):

        beta = float(self.text_proto_fuse_beta)

        if self.iter_num < self.text_proto_warmup_start:
            return 0.0

        if self.iter_num >= self.text_proto_warmup_end:
            return beta

        ratio = (
            float(self.iter_num - self.text_proto_warmup_start)
            / float(self.text_proto_warmup_end - self.text_proto_warmup_start + 1e-6))

        return beta * ratio


    def _project_text_to_proto(self):
    
        text_feat = self.fg_text_features_cluster.float()
        text_proto = self.text_proto_proj(text_feat)
        text_proto = l2norm(text_proto, dim=1)
        return text_proto


    def fuse_proto_with_text(self, proto_mix, img_labels):

        beta = self._get_text_proto_beta()

        if beta <= 0:
            return proto_mix

        B, K_all, C, _, _ = proto_mix.shape
        K_fg = self.num_classes - 1
        device = proto_mix.device

        # [K_fg, C]
        text_proto = self._project_text_to_proto()
        text_proto = text_proto.to(device=device, dtype=proto_mix.dtype)

        # [1, K_fg, C, 1, 1] -> [B, K_fg, C, 1, 1]
        text_proto = text_proto.view(1, K_fg, C, 1, 1).expand(B, -1, -1, -1, -1)

        proto_bg = proto_mix[:, :1]
        proto_fg = proto_mix[:, 1:]

        valid_mask = img_labels.to(device=device).bool().view(B, K_fg, 1, 1, 1)

        proto_fg_fused = l2norm((1.0 - beta) * proto_fg + beta * text_proto, dim=2)
        proto_fg_out = torch.where(valid_mask, proto_fg_fused, proto_fg)

        proto_out = torch.cat([proto_bg, proto_fg_out], dim=1)
        proto_out = l2norm(proto_out.squeeze(-1).squeeze(-1), dim=2).unsqueeze(-1).unsqueeze(-1)

        return proto_out


    def _np_kmeans(self, x: np.ndarray, k: int, iters: int = 20):
        """
        Simple numpy K-means for superpixel region clustering.

        Args:
            x: [N, D] numpy feature matrix
            k: number of clusters
            iters: number of K-means iterations

        Returns:
            assign: [N] cluster id for each row
        """
        n, d = x.shape
        if n == 0:
            return np.zeros((0,), dtype=np.int64)

        k = int(min(max(k, 1), n))

        # Deterministic farthest-point initialization.
        mean = x.mean(axis=0, keepdims=True)
        first = np.argmax(((x - mean) ** 2).sum(axis=1))
        centers = [x[first]]

        min_dist = ((x - centers[0]) ** 2).sum(axis=1)
        for _ in range(1, k):
            idx = np.argmax(min_dist)
            centers.append(x[idx])
            dist_new = ((x - centers[-1]) ** 2).sum(axis=1)
            min_dist = np.minimum(min_dist, dist_new)

        centers = np.stack(centers, axis=0)

        for _ in range(iters):
            dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
            assign = dist.argmin(axis=1)

            new_centers = centers.copy()
            for c in range(k):
                mask = assign == c
                if mask.any():
                    new_centers[c] = x[mask].mean(axis=0)

            if np.allclose(new_centers, centers, atol=1e-4):
                centers = new_centers
                break

            centers = new_centers

        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        assign = dist.argmin(axis=1).astype(np.int64)
        return assign


    @torch.no_grad()
    def _image_tensor_to_rgb_np(self, img_i: torch.Tensor):
        """
        Convert image tensor [3,H,W] to numpy RGB image in [0,1].
        This is robust to 0-255 tensors, 0-1 tensors, or normalized tensors.
        """
        img_np = img_i.detach().float().cpu()

        if img_np.dim() != 3:
            raise ValueError(f"Expected img_i with shape [C,H,W], got {tuple(img_np.shape)}")

        img_np = img_np[:3].permute(1, 2, 0).numpy()
        img_np = np.nan_to_num(img_np)

        v_min = float(img_np.min())
        v_max = float(img_np.max())

        if v_max > 2.0:
            img_np = img_np / 255.0
        else:
            img_np = (img_np - v_min) / (v_max - v_min + 1e-6)

        img_np = np.clip(img_np, 0.0, 1.0)
        return img_np


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
        """
        按图原型 -> 以 EMA 方式更新全局原型库（仅前景类）-> 融合局部与全局原型 -> 再激活得到 IS_cam。

        Args:
            prototypes: [B, K_all, C, 1, 1]   # 由 get_prototype 得到(含背景在第0类)
            seeds:      [B, K_all, H, W]      # 与 prototypes 对应的种子图(含背景)
            fts:        [B, C, H, W]          # 与原型同一特征空间的特征图

        Returns:
            proto_mix:  [B, K_all, C, 1, 1]   # 融合后的原型
            IS_cam:     [B, K_all, H, W]      # 用融合原型对 fts 再激活得到的类响应图
        """
        B, K_all, C, _, _ = prototypes.shape
        device = fts.device

        with torch.no_grad():
            # 原型去掉空间维，并做 L2 归一化
            proto_local = prototypes.squeeze(-1).squeeze(-1)            # [B, K_all, C]
            proto_local = F.normalize(proto_local, dim=-1)

            # 每类像素计数，避免空类污染
            pixel_count = seeds.sum(dim=(-1, -2))                       # [B, K_all]
            exist_mask = (pixel_count > 0)                              # [B, K_all]

            for k in range(1, self.num_classes):  # 0 是背景
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
    

    @staticmethod
    def normalize_map(x, eps=1e-6):
        x = x.detach().float()
        x_min = x.amin()
        x_max = x.amax()
        return (x - x_min) / (x_max - x_min + eps)

    def denorm_for_sam(self, x):
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=x.device,
            dtype=x.dtype,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=x.device,
            dtype=x.dtype,
        ).view(1, 3, 1, 1)

        return (x * std + mean).clamp(0, 1)

    @torch.no_grad()
    def build_sam_prompts(self, raw_cam, refined_cam, image_h, image_w):

        raw_cam = self.normalize_map(raw_cam)
        refined_cam = self.normalize_map(refined_cam)

        # raw CAM 保证语义 precision
        # refined CAM 提供完整性
        proposal = torch.sqrt((raw_cam * refined_cam).clamp_min(0.0))

        binary = (proposal >= self.sam_component_thres).cpu().numpy()

        component_map, num_components = ndi.label(binary)

        components = []
        component_areas = []
        
        for component_id in range(1, num_components + 1):
            component_np = component_map == component_id
            area = int(component_np.sum())

            area_ratio = area / float(image_h * image_w)

            if area < self.sam_min_component_area:
                continue
            
            if area_ratio > self.sam_max_component_ratio:
                continue

            component = torch.from_numpy(component_np).to(device=proposal.device,dtype=torch.bool,)

            component_score = proposal[component].mean().item()

            components.append((component_score, component))

            component_areas.append(area)

        # 防止小目标极多时 SAM 调用爆炸
        components.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        components = components[:self.sam_max_components]

        scale_x = self.sam_input_size / float(image_w)
        scale_y = self.sam_input_size / float(image_h)

        prompts = []

        for component_id, (_, component) in enumerate(components):
            ys, xs = torch.where(component)

            if ys.numel() == 0:
                continue

            values = proposal[component]

            # 第一个正点：component 内 CAM 最大的位置
            first_idx = values.argmax()

            pos_y = ys[first_idx]
            pos_x = xs[first_idx]

            # 第二个正点：与第一个点距离较远，同时 CAM 足够高
            distance = (
                (ys.float() - pos_y.float()).square()
                + (xs.float() - pos_x.float()).square()
            )

            valid_second = values >= self.sam_seed_thres

            if valid_second.any():
                distance_valid = distance.masked_fill(
                    ~valid_second,
                    -1.0,
                )
                second_idx = distance_valid.argmax()
            else:
                second_idx = distance.argmax()

            pos2_y = ys[second_idx]
            pos2_x = xs[second_idx]

            def to_sam_xy(x, y):
                return torch.tensor(
                    [
                        float(x) * scale_x,
                        float(y) * scale_y,
                    ],
                    device=proposal.device,
                    dtype=torch.float32,
                )

            # prompt 1：单正点
            prompts.append(
                {
                    'points': torch.stack(
                        [to_sam_xy(pos_x, pos_y)]
                    ),
                    'labels': torch.tensor(
                        [1],
                        device=proposal.device,
                        dtype=torch.float32,
                    ),
                    'component': component,
                    'component_id': component_id,
                }
            )

            # prompt 2：两个正点
            if pos2_x != pos_x or pos2_y != pos_y:
                prompts.append(
                    {
                        'points': torch.stack(
                            [
                                to_sam_xy(pos_x, pos_y),
                                to_sam_xy(pos2_x, pos2_y),
                            ]
                        ),
                        'labels': torch.tensor(
                            [1, 1],
                            device=proposal.device,
                            dtype=torch.float32,
                        ),
                        'component': component,
                        'component_id': component_id,
                    }
                )

            # 构造 component 外围 ring，用于寻找负点
            component_np = component.cpu().numpy()

            dilated = ndi.binary_dilation(
                component_np,
                iterations=8,
            )

            ring_np = np.logical_and(
                dilated,
                np.logical_not(component_np),
            )

            ring = torch.from_numpy(
                ring_np
            ).to(
                device=proposal.device,
                dtype=torch.bool,
            )

            if ring.any():
                # 在外围 ring 中选 proposal 最低的位置作为负点
                ring_score = proposal.masked_fill(
                    ~ring,
                    float('inf'),
                )

                neg_flat = ring_score.argmin()

                neg_y = neg_flat // image_w
                neg_x = neg_flat % image_w

                prompts.append(
                    {
                        'points': torch.stack(
                            [
                                to_sam_xy(pos_x, pos_y),
                                to_sam_xy(neg_x, neg_y),
                            ]
                        ),
                        'labels': torch.tensor(
                            [1, 0],
                            device=proposal.device,
                            dtype=torch.float32,
                        ),
                        'component': component,
                        'component_id': component_id,
                    }
                )

        return prompts

    @torch.no_grad()
    def score_sam_mask(self, mask, sam_iou, raw_cam, refined_cam, dino_feature, component):
        """
        mask:         [H, W] bool
        raw_cam:      [H, W]
        refined_cam:  [H, W]
        dino_feature: [C, Hf, Wf]
        component:    [H, W] bool
        """

        mask_area = int(mask.sum().item())

        if mask_area < self.sam_min_mask_area:
            return None

        area_ratio = mask_area / float(mask.numel())

        if area_ratio > self.sam_max_mask_ratio:
            return None

        raw_cam = self.normalize_map(raw_cam)
        refined_cam = self.normalize_map(refined_cam)

        # mask 内部语义响应
        raw_precision = raw_cam[mask].mean()
        refined_precision = refined_cam[mask].mean()

        # SAM mask 对原始 CAM component 的覆盖程度
        intersection_area = (mask & component).sum().float()

        component_recall = (intersection_area / component.sum().float().clamp_min(1.0))

        component_precision = (intersection_area / mask.sum().float().clamp_min(1.0))

        region_features = dino_feature[:, mask]

        if region_features.shape[1] <= 1:
            dino_coherence = mask.new_tensor(0.0, dtype=torch.float32,)
        else:
            region_proto = region_features.mean(dim=1)
            region_proto = F.normalize(region_proto, dim=0)

            dino_coherence = (region_features.transpose(0, 1) @ region_proto)
            dino_coherence = (dino_coherence.clamp_min(0.0).mean())

        # 几何平均
        terms = torch.stack(
            [raw_precision.clamp(1e-4, 1.0),
            refined_precision.clamp(1e-4, 1.0),
            component_recall.clamp(1e-4, 1.0),
            component_precision.clamp(1e-4, 1.0),
            dino_coherence.clamp(1e-4, 1.0),
            sam_iou.float().clamp(1e-4, 1.0)]
        )

        score = terms.log().mean().exp()

        return score


    @torch.no_grad()
    def generate_sam_pseudo_labels(self, img, img_labels, raw_cams, refined_cams, dino_fts):
        
        self.net_sam.eval()
        batch_size, _, image_h, image_w = img.shape

        dino_up_all = F.interpolate(dino_fts, size=(image_h, image_w), mode='bilinear', align_corners=False,)

        dino_up_all = F.normalize(dino_up_all, dim=1)

        # 恢复成 SAM 需要的 RGB 图像
        img_sam = self.denorm_for_sam(img)

        img_sam = F.interpolate(
            img_sam,
            size=(
                self.sam_input_size,
                self.sam_input_size,
            ),
            mode='bilinear',
            align_corners=False,
        )

        img_sam = (
            img_sam * 255.0
        ).round().to(torch.uint8)

        # SAM image encoder 每张图只运行一次
        features_sam = self.net_sam(
            run_encoder_only=True,
            transformed_image=img_sam,
            original_image_size=(image_h, image_w),
        )

        sam_conf = torch.zeros_like(raw_cams, dtype=torch.float32,)
        accepted_per_image = [0 for _ in range(batch_size)]
        
        for batch_idx in range(batch_size):
            valid_classes = (img_labels[batch_idx].nonzero(as_tuple=False)[:, 0].tolist())

            for class_idx in valid_classes:
                raw_cam = raw_cams[
                    batch_idx,
                    class_idx,
                ]

                refined_cam = refined_cams[
                    batch_idx,
                    class_idx,
                ]

                prompts = self.build_sam_prompts(
                    raw_cam=raw_cam,
                    refined_cam=refined_cam,
                    image_h=image_h,
                    image_w=image_w,
                )

                candidates = []

                for prompt_id, prompt in enumerate(prompts):
                    point_coords = prompt[
                        'points'
                    ].unsqueeze(0)

                    point_labels = prompt[
                        'labels'
                    ].unsqueeze(0)

                    output_sam = self.net_sam(
                        run_decoder_only=True,
                        features_sam=features_sam[
                            batch_idx
                        ].unsqueeze(0),
                        original_image_size=(
                            image_h,
                            image_w,
                        ),
                        point_coords=point_coords,
                        point_labels=point_labels,
                    )

                    masks_raw = output_sam[0][0]

                    if masks_raw.dtype == torch.bool:
                        masks = masks_raw
                    else:
                        masks = masks_raw > 0

                    iou_scores = output_sam[1][0]

                    for mask_idx in range(masks.shape[0]):
                        candidate_mask = masks[mask_idx]

                        base_score = self.score_sam_mask(
                            mask=candidate_mask,
                            sam_iou=iou_scores[mask_idx],
                            raw_cam=raw_cam,
                            refined_cam=refined_cam,
                            dino_feature=dino_up_all[batch_idx],
                            component=prompt['component'],)

                        if base_score is None:
                            continue
                        
                        candidates.append(
                            {
                                'mask': candidate_mask,
                                'base_score': base_score,
                                'component_id': prompt['component_id'],
                                'prompt_id': prompt_id,
                            }
                        )

                # 多 prompt consistency
                for candidate_idx, candidate in enumerate(candidates):
                    overlap_list = []

                    for other_idx, other in enumerate(candidates):
                        if candidate_idx == other_idx:
                            continue

                        # Different objects/components must not be compared.
                        if (
                            candidate['component_id']
                            != other['component_id']
                        ):
                            continue

                        # The three multimask outputs from the same SAM call
                        # are not independent prompt evidence.
                        if candidate['prompt_id'] == other['prompt_id']:
                            continue

                        intersection = (
                            candidate['mask']
                            & other['mask']
                        ).sum().float()

                        union = (
                            candidate['mask']
                            | other['mask']
                        ).sum().float().clamp_min(1.0)

                        overlap_list.append(
                            intersection / union
                        )

                    if len(overlap_list) > 0:
                        overlap_tensor = torch.stack(overlap_list)
                        topk = min(2, overlap_tensor.numel())
                        consistency = overlap_tensor.topk(topk).values.mean()
                    else:
                        consistency = candidate['base_score'].new_tensor(0.5)

                    final_score = (candidate['base_score'] * (0.5 + 0.5 * consistency))

                    if final_score < self.sam_accept_score:
                        continue
                    
                    accepted_per_image[batch_idx] += 1

                    candidate_mask = candidate['mask']

                    current_score = sam_conf[
                        batch_idx,
                        class_idx,
                    ][candidate_mask]

                    new_score = torch.full_like(current_score,float(final_score),)

                    sam_conf[batch_idx, class_idx,][candidate_mask] = torch.maximum(current_score,new_score,)

        print('SAM accepted per image:', accepted_per_image,)

        # 每个像素选择得分最高的类别
        best_score, best_class = sam_conf.max(dim=1)

        pseudo_label = torch.full((batch_size,image_h,image_w,),fill_value=self.sam_ignore_index,device=img.device,dtype=torch.long,)

        # 高置信前景
        foreground_mask = (best_score >= self.sam_pixel_score)

        pseudo_label[foreground_mask] = (best_class[foreground_mask] + 1)

        # 高置信背景
        raw_norm = self.minmax_norm(raw_cams)
        refined_norm = self.minmax_norm(refined_cams)

        raw_fg_score = raw_norm.max(dim=1).values
        refined_fg_score = refined_norm.max(dim=1).values

        foreground_evidence = torch.maximum(raw_fg_score, refined_fg_score)

        background_mask = (foreground_evidence <= self.sam_bg_score)& (~foreground_mask)

        pseudo_label[background_mask] = 0

        for batch_idx in range(batch_size):
            if accepted_per_image[batch_idx] == 0:
                pseudo_label[batch_idx].fill_(self.sam_ignore_index)

        sam_fg = (
            (pseudo_label > 0)
            & (pseudo_label != self.sam_ignore_index)
        )

        raw_norm = self.minmax_norm(raw_cams)
        refined_norm = self.minmax_norm(refined_cams)

        proposal = torch.sqrt(
            (raw_norm * refined_norm).clamp_min(0.0)
        )

        cam_fg = (
            proposal.max(dim=1).values
            >= self.sam_component_thres
        )

        intersection = (
            sam_fg & cam_fg
        ).sum(dim=(1, 2)).float()

        union = (
            sam_fg | cam_fg
        ).sum(dim=(1, 2)).float()

        sam_cam_iou = (
            intersection
            / union.clamp_min(1.0)
        )

        print(
            'SAM-CAM IoU:',
            'mean=', float(sam_cam_iou.mean().item()),
            'min=', float(sam_cam_iou.min().item()),
            'max=', float(sam_cam_iou.max().item()),
        )

        return pseudo_label, sam_conf


    def forward(self, img, img_labels=None, mode='train'):
        
        cam_list = []
        all_cam_list = []
        all_refine_cam_list = []
        b, c, h, w = img.shape
        if mode == 'train':
            self.iter_num += 1

        fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True, clip_flag=self.clip_flag)

        self.dino_encoder.eval()
        with torch.no_grad():
            dino_img_h, dino_img_w = (h//14)*14, (w//14)*14
            dino_img = F.interpolate(img, size=(dino_img_h, dino_img_w), mode='bilinear', align_corners=False)
            dino_ftses = self.dino_encoder.forward_features(dino_img)
            dino_fts = dino_ftses['x_norm_patchtokens']

            dino_raw_tokens = dino_fts[-1] if isinstance(dino_fts, list) else dino_fts
            dino_raw = dino_raw_tokens.reshape(b, dino_img_h // 14, dino_img_w // 14, -1,).permute(0, 3, 1, 2).contiguous()
            
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
            
            cam_refined_list, keys, w, h = perform_cam(img_label, img_i, cam_fts, cam_attn, seg_attn,
                                                        self.fg_text_features_cluster, self.bg_text_features,
                                                        self.grad_cam, mode=mode, require_seg_trans=require_seg_trans,
                                                        seg_dino_cam=seg_dino_cam, clip_flag = self.clip_flag)
            # cam_refined_list, keys, w, h = perform_cam(img_label, img_i, cam_fts, cam_attn, seg_attn,
            #                                             self.fg_text_features_attr, self.bg_text_features,
            #                                             self.grad_cam, mode=mode, require_seg_trans=require_seg_trans,
            #                                             seg_dino_cam=seg_dino_cam, clip_flag = self.clip_flag) #[B, 20,20]
            if len(keys) == 0:
                cams = torch.ones((1, h, w), device=img_i.device)  # 只有背景通道
                valid_key = torch.tensor([0], dtype=torch.long, device=img_i.device)
            else:
                cam_dict = generate_cam_label(cam_refined_list, keys, w, h)
                cams = cam_dict['refined_cam'].to(img_i.device)     # [K,H',W']
                bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdim=True)[0], self.cam_bg_thres)
                cams = torch.cat([bg_score, cams], dim=0)           # [1+K,H',W']
                valid_key = np.pad(np.array(cam_dict['keys'], dtype=np.int64) + 1, (1, 0), mode='constant')
                valid_key = torch.from_numpy(valid_key).cuda()


            with torch.no_grad():
                refined_cams, cam_labels = _refine_cams(self.par, img[i], cams, valid_key)
            cams_full = torch.zeros((self.num_classes-1, h, w), dtype=cams.dtype, device=img_i.device)
            cams_refine_full = torch.zeros((self.num_classes-1, h, w), dtype=cams.dtype, device=img_i.device)

            if len(keys) != 0: 
                for idx, class_id in enumerate(cam_dict['keys']):#[3,7]
                    cams_full[class_id] = cams[idx+1]
                    cams_refine_full[class_id] = refined_cams[idx+1]

            cam_list.append(cam_labels)
            all_cam_list.append(cams_full)
            all_refine_cam_list.append(cams_refine_full)

        all_cam_labels = torch.stack(cam_list, dim=0) #[B, 320, 320]
        all_cam_no_refine_labels = torch.stack(all_cam_list, dim=0) #[B, 15, 320, 320]
        all_cam_refine_labels = torch.stack(all_refine_cam_list, dim=0) #[B, 15, 320, 320]

        # Frozen raw DINO -> AnyUp 80x80 -> DINO prototype similarity -> CAM.
        # Decoder and affinity stay at their original low resolution.
        if self.use_anyup:
            all_cam_no_refine_labels, anyup_dino_similarity, dino_highres = (
                self.calibrate_cam_with_anyup_dino(
                    guidance_image=dino_img,
                    dino_raw=dino_raw,
                    cams_fg=all_cam_no_refine_labels,
                    img_labels=img_labels,
                )
            )
            all_cam_labels = self.cams_to_dense_labels(all_cam_no_refine_labels, img_labels)

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

        proto_mix = self.fuse_proto_with_text(proto_mix=proto_mix, img_labels=img_labels)

        IS_cam = self.reactivate(proto_mix, fts)

        if mode != 'train':
            return seg_clip, seg_dino, all_cam_labels, attn_pred

        pgt_sam = None
        sam_conf = None

        should_run_sam = (
            self.use_sam
            and self.iter_num >= self.sam_start_iter
            and self.iter_num % self.sam_interval == 0
        )

        if should_run_sam:
            with torch.no_grad():
                pgt_sam, sam_conf = self.generate_sam_pseudo_labels(
                    img=img,
                    img_labels=img_labels,
                    raw_cams=all_cam_no_refine_labels,
                    refined_cams=all_cam_refine_labels,
                    dino_fts=dino_fts.detach(),
                )

        return seg_clip, seg_dino, all_cam_labels, attn_pred, norm_cam, IS_cam, pgt_sam, sam_conf
        # return seg_clip, seg_dino, all_cam_labels, attn_pred, norm_cam, IS_cam

