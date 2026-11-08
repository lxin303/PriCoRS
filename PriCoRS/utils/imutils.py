import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from torchvision import transforms
import random

color_table = {
    (255, 127, 0): 14,  # plane
    (127, 0, 0): 9,  # small_vehicle
    (127, 127, 0): 8,  # large_vehicle
    (63, 0, 0): 1,  # ship
    (155, 100, 0): 15,  # harbor
    (127, 63, 0): 4,  # tennis_court
    (255, 0, 0): 11,  # swimming_pool
    (255, 63, 0): 3,  # baseball_diamond
    (63, 63, 0): 2,  # stroage_tank
    (63, 127, 0): 7,  # bridge
    (0, 63, 0): 6,  # Ground_Track_Field
    (191, 0, 0): 10,  # helicopter
    (191, 63, 0): 5,  # basketball_court
    (191, 127, 0): 13,  # Soccer_ball_field
    (127, 191, 0): 12  # Roundabout
}

NUM_CLASSES = 16
BACKGROUND_ID = 0
BACKGROUND_COLOR = (0, 0, 0)   
IGNORE_INDEX = 255        

PALETTE = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
PALETTE[BACKGROUND_ID] = np.array(BACKGROUND_COLOR, dtype=np.uint8)
for (r, g, b), cid in color_table.items():
    if 0 <= cid < NUM_CLASSES:
        PALETTE[cid] = (r, g, b)

def encode_cmap(mask: np.ndarray) -> np.ndarray:
    """
    输入: mask HxW, uint8, 值为类别ID或 IGNORE_INDEX
    输出: RGB HxWx3, uint8,与 GT 的颜色一致
    """
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    valid = mask != IGNORE_INDEX
    out[valid] = PALETTE[mask[valid]]
    # 给 ignore 一个可视化颜色（灰色）；如不需要可改成背景色
    out[~valid] = np.array([128, 128, 128], dtype=np.uint8)
    return out

def denormalize_img(imgs=None, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    _imgs = torch.zeros_like(imgs)
    _imgs[:,0,:,:] = imgs[:,0,:,:] * std[0] + mean[0]
    _imgs[:,1,:,:] = imgs[:,1,:,:] * std[1] + mean[1]
    _imgs[:,2,:,:] = imgs[:,2,:,:] * std[2] + mean[2]
    _imgs = _imgs.type(torch.uint8)

    return _imgs

def denormalize_img2(imgs=None):
    #_imgs = torch.zeros_like(imgs)
    imgs = denormalize_img(imgs)

    return imgs / 255.0


def tensorboard_image(imgs=None, cam=None,):
    ## images

    _imgs = denormalize_img(imgs=imgs)
    grid_imgs = torchvision.utils.make_grid(tensor=_imgs, nrow=2)

    cam = F.interpolate(cam, size=_imgs.shape[2:], mode='bilinear', align_corners=False)
    cam = cam.cpu()
    cam_max = cam.max(dim=1)[0]
    cam_heatmap = plt.get_cmap('jet')(cam_max.numpy())[:,:,:,0:3]*255
    cam_cmap = torch.from_numpy(cam_heatmap).permute([0, 3, 1, 2])
    cam_img = cam_cmap*0.5 + _imgs.cpu()*0.5
    grid_cam = torchvision.utils.make_grid(tensor=cam_img.type(torch.uint8), nrow=2)

    return grid_imgs, grid_cam


def tensorboard_edge(edge=None, n_row=2):
    ## 
    edge = F.interpolate(edge, size=[224, 224], mode='bilinear', align_corners=False)[:,0,...]
    edge = edge.cpu()
    edge_heatmap = plt.get_cmap('viridis')(edge.numpy())[:,:,:,0:3]*255
    edge_cmap = torch.from_numpy(edge_heatmap).permute([0, 3, 1, 2])

    grid_edge = torchvision.utils.make_grid(tensor=edge_cmap.type(torch.uint8), nrow=n_row)

    return grid_edge


def cam_to_label(cam, cls_label, img_box=None, bkg_thre=None, high_thre=None, low_thre=None, ignore_mid=False,
                 ignore_index=None):
    b, c, h, w = cam.shape
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    # For deepglobe dataset training, please remember to annotate the above codes to avoid list out of range: _pseudo_label += 1
    _pseudo_label[cam_value <= bkg_thre] = 0

    if img_box is None:
        return _pseudo_label
    if ignore_mid:
        _pseudo_label[cam_value <= high_thre] = ignore_index
        _pseudo_label[cam_value <= low_thre] = 0
    pseudo_label = torch.ones_like(_pseudo_label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1],
                                                                  coord[2]:coord[3]]
    
    return valid_cam, pseudo_label


# def _refine_cams(ref_mod, images, cams, valid_key, orig_size):
#     images = images.unsqueeze(0)
#     cams = cams.unsqueeze(0)

#     refined_cams = ref_mod(images.float(), cams.float())
#     refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
#     refined_label = refined_cams.argmax(dim=1)
#     refined_label = valid_key[refined_label]

#     return refined_label.squeeze(0)


def _refine_cams(ref_mod, images, cams, valid_key, orig_size):
    refined_cams = ref_mod(images.float(), cams.float())
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label

def refine_cams_with_bkg_v2(ref_mod=None, images=None, cams=None, cls_labels=None, high_thre=None, low_thre=None,
                            ignore_index=False, img_box=None, down_scale=2):
    # For deepglobe dataset training, since no explicit backgroudn category is given, cls_labels no longer needs to cat with bkg_cls. So does the cams.
    # For instance, you can change the code like this:
    '''
     b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)
    refined_label = torch.ones(size=(b, h, w)) * ignore_index
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()

     cams_with_bkg_h = cams
    _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)
    cams_with_bkg_l = cams
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)
    '''
    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b, 1, h, w)) * high_thre
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b, 1, h, w)) * low_thre
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * ignore_index
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()

    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)

    for idx, coord in enumerate(img_box):
        valid_key = torch.nonzero(cls_labels[idx, ...])[:, 0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_h,
                                        valid_key=valid_key, orig_size=(h, w))
        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_l,
                                        valid_key=valid_key, orig_size=(h, w))

        refined_label_h[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_h[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]
        refined_label_l[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_l[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = ignore_index
    refined_label[(refined_label_h + refined_label_l) == 0] = 0
    
    return refined_label


def cam_to_roi_mask2(cam, cls_label, hig_thre=None, low_thre=None):
    b, c, h, w = cam.shape
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    valid_cam = cls_label_rep * cam
    cam_value, _ = valid_cam.max(dim=1, keepdim=False)
    # _pseudo_label += 1
    roi_mask = torch.ones_like(cam_value, dtype=torch.int16)
    roi_mask[cam_value <= low_thre] = 0
    roi_mask[cam_value >= hig_thre] = 2
    return roi_mask


def crop_from_roi_neg(images, roi_mask=None, crop_num=8, crop_size=96):
    crops = []

    b, c, h, w = images.shape

    temp_crops = torch.zeros(size=(b, crop_num, c, crop_size, crop_size)).to(images.device)
    flags = torch.ones(size=(b, crop_num + 2)).to(images.device)
    margin = crop_size // 2

    for i1 in range(b):
        roi_index = (roi_mask[i1, margin:(h - margin), margin:(w - margin)] <= 1).nonzero()
        if roi_index.shape[0] < crop_num:
            roi_index = (roi_mask[i1, margin:(h - margin),
                         margin:(w - margin)] >= 0).nonzero()  ## if NULL then random crop
        rand_index = torch.randperm(roi_index.shape[0])
        crop_index = roi_index[rand_index[:crop_num], :]

        for i2 in range(crop_num):
            h0, w0 = crop_index[i2, 0], crop_index[i2, 1]  # centered at (h0, w0)
            temp_crops[i1, i2, ...] = images[i1, :, h0:(h0 + crop_size), w0:(w0 + crop_size)]
            temp_mask = roi_mask[i1, h0:(h0 + crop_size), w0:(w0 + crop_size)]
            if temp_mask.sum() / (crop_size * crop_size) <= 0.2:
                ## if ratio of uncertain regions < 0.2 then negative
                flags[i1, i2 + 2] = 0

    _crops = torch.chunk(temp_crops, chunks=crop_num, dim=1, )
    crops = [c[:, 0] for c in _crops]

    return crops, flags


def label_to_aff_mask(cam_label, ignore_index=255):
    b, h, w = cam_label.shape

    _cam_label = cam_label.reshape(b, 1, -1)
    _cam_label_rep = _cam_label.repeat([1, _cam_label.shape[-1], 1])
    _cam_label_rep_t = _cam_label_rep.permute(0, 2, 1)
    aff_label = (_cam_label_rep == _cam_label_rep_t).type(torch.long)

    for i in range(b):
        aff_label[i, :, _cam_label_rep[i, 0, :] == ignore_index] = ignore_index
        aff_label[i, _cam_label_rep[i, 0, :] == ignore_index, :] = ignore_index
    aff_label[:, range(h * w), range(h * w)] = ignore_index
    return aff_label


def get_rand_aug():
    aug = transforms.Compose([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return aug


def augment_data(images, labels):
    for i in range(images.shape[0]):
        augment = get_rand_aug()
        images[i] = augment(images[i])
        if random.random() < 0.5:  # Random Flipping
            images[i] = torch.flip(images[i], dims=[2])
            labels[i] = torch.flip(labels[i], dims=[2])

    return images, labels


def crop_from_pseudo_label_1(images, pseudo_mask=None, crop_num=8, crop_size=96):
    b, c, h, w = images.shape

    temp_crops = torch.zeros(size=(b, crop_num, c, crop_size, crop_size)).to(images.device)
    margin = crop_size // 2

    # Exp
    crop_pseudo_label = torch.zeros(size=(b, crop_num, 1, crop_size, crop_size)).to(images.device)

    for i1 in range(b):
        roi_index = (
                (pseudo_mask[i1, margin:(h - margin), margin:(w - margin)] != 0) &
                (pseudo_mask[i1, margin:(h - margin), margin:(w - margin)] != 255)
        ).nonzero()  # 不是背景不是255就是ROI

        if roi_index.shape[0] < crop_num:
            roi_index = (pseudo_mask[i1, margin:(h - margin),
                         margin:(w - margin)] >= 0).nonzero()  ## if NULL then random crop
        rand_index = torch.randperm(roi_index.shape[0])
        crop_index = roi_index[rand_index[:crop_num], :]

        for i2 in range(crop_num):
            h0, w0 = crop_index[i2, 0], crop_index[i2, 1]  # centered at (h0, w0)
            temp_crops[i1, i2, ...] = images[i1, :, h0:(h0 + crop_size), w0:(w0 + crop_size)]
            crop_pseudo_label[i1, i2, ...] = pseudo_mask[i1, h0:(h0 + crop_size), w0:(w0 + crop_size)]

    return temp_crops, crop_pseudo_label
                                 

def tensorboard_attn(attns=None, size=[224,224], n_pix=0, n_row=4):
    n = len(attns)
    imgs = []
    for idx, attn in enumerate(attns):
        
        b, hw, _ = attn.shape
        h = w = int(np.sqrt(hw))

        attn_ = attn.clone()# - attn.min()
        #attn_ = attn_ / attn_.max()
        _n_pix = int(h * n_pix) * (w + 1)
        attn_ = attn_[:, _n_pix, :].reshape(b, 1, h, w)

        attn_ = F.interpolate(attn_, size=size, mode='bilinear', align_corners=True)

        attn_ = attn_.cpu()[:,0,:,:]

        def minmax_norm(x):
            for i in range(x.shape[0]):
                x[i,...] = x[i,...] - x[i,...].min()
                x[i,...] = x[i,...] / x[i,...].max()
            return x
        attn_ = minmax_norm(attn_)

        attn_heatmap = plt.get_cmap('viridis')(attn_.numpy())[:,:,:,0:3]*255
        attn_heatmap = torch.from_numpy(attn_heatmap).permute([0, 3, 2, 1])
        imgs.append(attn_heatmap)
    attn_img = torch.cat(imgs, dim=0)
    
    grid_attn = torchvision.utils.make_grid(tensor=attn_img.type(torch.uint8), nrow=n_row).permute(0,2,1)

    return grid_attn

def tensorboard_attn2(attns=None, size=[224,224], n_pixs=[0.0, 0.3, 0.6, 0.9], n_row=4, with_attn_pred=True):
    n = len(attns)
    attns_top_layers = []
    attns_last_layer = []
    grid_attns = []
    if with_attn_pred:
        _attns_top_layers = attns[:-3]
        _attns_last_layer = attns[-3:-1]
    else:
        _attns_top_layers = attns[:-2]
        _attns_last_layer = attns[-2:]
    
    attns_top_layers = [_attns_top_layers[i][:,0,...] for i in range(len(_attns_top_layers))]
    if with_attn_pred:
        attns_top_layers.append(attns[-1])
    grid_attn_top_case0 = tensorboard_attn(attns_top_layers, n_pix=n_pixs[0], n_row=n_row)
    grid_attn_top_case1 = tensorboard_attn(attns_top_layers, n_pix=n_pixs[1], n_row=n_row)
    grid_attn_top_case2 = tensorboard_attn(attns_top_layers, n_pix=n_pixs[2], n_row=n_row)
    grid_attn_top_case3 = tensorboard_attn(attns_top_layers, n_pix=n_pixs[3], n_row=n_row)
    grid_attns.append(grid_attn_top_case0)
    grid_attns.append(grid_attn_top_case1)
    grid_attns.append(grid_attn_top_case2)
    grid_attns.append(grid_attn_top_case3)
    
    for attn in _attns_last_layer:
        for i in range(attn.shape[1]):
            attns_last_layer.append(attn[:,i,:,:])
    grid_attn_last_case0 = tensorboard_attn(attns_last_layer, n_pix=n_pixs[0], n_row=2*n_row)
    grid_attn_last_case1 = tensorboard_attn(attns_last_layer, n_pix=n_pixs[1], n_row=2*n_row)
    grid_attn_last_case2 = tensorboard_attn(attns_last_layer, n_pix=n_pixs[2], n_row=2*n_row)
    grid_attn_last_case3 = tensorboard_attn(attns_last_layer, n_pix=n_pixs[3], n_row=2*n_row)
    grid_attns.append(grid_attn_last_case0)
    grid_attns.append(grid_attn_last_case1)
    grid_attns.append(grid_attn_last_case2)
    grid_attns.append(grid_attn_last_case3)

    return grid_attns

def tensorboard_label(labels=None):
    ## labels
    labels_cmap = encode_cmap(np.squeeze(labels))
    # print(labels_cmap.shape)
    # labels_cmap = torch.from_numpy(labels_cmap).permute([0, 3, 1, 2])
    labels_cmap = torch.from_numpy(labels_cmap).unsqueeze(0).permute([0, 3, 1, 2])
    grid_labels = torchvision.utils.make_grid(tensor=labels_cmap, nrow=2)

    return grid_labels


def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap