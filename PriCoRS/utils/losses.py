import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("./wrapper/bilateralfilter/build/lib.linux-x86_64-3.6")
# from bilateralfilter import bilateralfilter, bilateralfilter_batch

def get_aff_loss(inputs, targets):

    pos_label = (targets == 1).type(torch.int16)
    pos_count = pos_label.sum() + 1
    neg_label = (targets == 0).type(torch.int16)
    neg_count = neg_label.sum() + 1
    #inputs = torch.sigmoid(input=inputs)

    pos_loss = torch.sum(pos_label * (1 - inputs)) / pos_count
    neg_loss = torch.sum(neg_label * (inputs)) / neg_count

    return 0.5 * pos_loss + 0.5 * neg_loss, pos_count, neg_count


def get_seg_loss(pred, label, ignore_index=255):
    
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5


def get_energy_loss(img, logit, label, img_box, loss_layer, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):

    pred_prob = F.softmax(logit, dim=1)
    crop_mask = torch.zeros_like(pred_prob[:,0,...])

    for idx, coord in enumerate(img_box):
        crop_mask[idx, coord[0]:coord[1], coord[2]:coord[3]] = 1

    _img = torch.zeros_like(img)
    _img[:,0,:,:] = img[:,0,:,:] * std[0] + mean[0]
    _img[:,1,:,:] = img[:,1,:,:] * std[1] + mean[1]
    _img[:,2,:,:] = img[:,2,:,:] * std[2] + mean[2]

    loss = loss_layer(_img, pred_prob, crop_mask, label.type(torch.uint8).unsqueeze(1), )

    return loss.cuda()


class DenseEnergyLossFunction(Function):
    
    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs, unlabel_region):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape
        Gate = ROIs.clone().to(ROIs.device)

        ROIs = ROIs.unsqueeze_(1).repeat(1,ctx.K,1,1)

        seg_max = torch.max(segmentations, dim=1)[0]
        Gate = Gate - seg_max
        Gate[unlabel_region] = 1
        Gate[Gate < 0] = 0
        Gate = Gate.unsqueeze_(1).repeat(1, ctx.K, 1, 1)

        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs
        
        densecrf_loss = 0.0
        images = images.cpu().numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        Gate = Gate.cpu().numpy().flatten()
        AS = np.multiply(AS, Gate)
        densecrf_loss -= np.dot(segmentations, AS)
    
        # averaged by the number of images
        densecrf_loss /= ctx.N
        
        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2*grad_output*torch.from_numpy(ctx.AS)/ctx.N
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None, None  


class DenseEnergyLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseEnergyLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor
    
    def forward(self, images, segmentations, ROIs, seg_label):
        """ scale imag by scale_factor """
        scaled_images = F.interpolate(images,scale_factor=self.scale_factor) 
        scaled_segs = F.interpolate(segmentations,scale_factor=self.scale_factor,mode='bilinear',align_corners=False)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1),scale_factor=self.scale_factor).squeeze(1)
        scaled_seg_label = F.interpolate(seg_label,scale_factor=self.scale_factor,mode='nearest')
        unlabel_region = (scaled_seg_label.long() == 255).squeeze(1)

        return self.weight*DenseEnergyLossFunction.apply(
                scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy*self.scale_factor, scaled_ROIs, unlabel_region)
    
    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )


# def dice_loss(output, target, eps=1e-7):
#     eps = 1e-7
#     # valid_target = target[target != 255]
#     # valid_output = output[target.unsqueeze(1).expand_as(output) != 255]
#     target[target==255] = output.shape[1]
#     # convert target to onehot
#     targ_onehot = torch.eye(output.shape[1]+1).cuda()[target].permute(0,3,1,2).float()
#     targ_onehot = targ_onehot[:,:-1,:,:]
#     # convert logits to probs
#     pred = F.softmax(output, dim=1)

#     target_ex = target.unsqueeze(1).expand_as(pred)
#     valid_region = torch.ones_like(pred)
#     valid_region[target_ex == 255] = 0
#     # sum over HW
#     inter = (pred*valid_region * targ_onehot).sum(axis=[0,2,3])
#     union = (pred*valid_region + targ_onehot*valid_region).sum(axis=[0,2,3])
#     # mean over C
#     dice = (2. * inter / (union + eps)).mean()
#     return 1. - dice


def dice_loss(output, target, ignore_index=255, eps=1e-7):
    target = target.clone().long()

    valid_mask = target != ignore_index

    if valid_mask.sum() == 0:
        return output.new_tensor(0.0)

    target_safe = target.clone()
    target_safe[~valid_mask] = 0

    pred = F.softmax(output, dim=1)

    target_onehot = F.one_hot(target_safe, num_classes=output.shape[1],).permute(0, 3, 1, 2).float()

    valid_mask = valid_mask.unsqueeze(1).float()

    pred = pred * valid_mask
    target_onehot = target_onehot * valid_mask

    inter = (pred * target_onehot).sum(dim=(0, 2, 3))
    union = (pred + target_onehot).sum(dim=(0, 2, 3))

    valid_class = union > eps

    if valid_class.sum() == 0:
        return output.new_tensor(0.0)

    dice = (2.0 * inter[valid_class] + eps) / (union[valid_class] + eps)

    return 1.0 - dice.mean()


# def dice_loss(output, target, ignore_index=255, eps=1e-7):
#     """
#     output: [B, C, H, W], logits
#     target: [B, H, W], class index, ignore_index=255
#     """
#     B, C, H, W = output.shape

#     target = target.clone().long()
#     valid_mask = target != ignore_index

#     target_safe = target.clone()
#     target_safe[~valid_mask] = 0

#     pred = F.softmax(output, dim=1)

#     target_onehot = F.one_hot(target_safe, num_classes=C)
#     target_onehot = target_onehot.permute(0, 3, 1, 2).float()

#     valid_mask = valid_mask.unsqueeze(1).float()

#     pred = pred * valid_mask
#     target_onehot = target_onehot * valid_mask

#     inter = (pred * target_onehot).sum(dim=(0, 2, 3))
#     union = (pred + target_onehot).sum(dim=(0, 2, 3))

#     valid_class = union > eps

#     if valid_class.sum() == 0:
#         return output.new_tensor(0.0)

#     dice = (2.0 * inter[valid_class] + eps) / (union[valid_class] + eps)
#     return 1.0 - dice.mean()


class JointLoss(nn.Module):
    def __init__(self, ignore_index=255, ratio=0.2):
        super(JointLoss, self).__init__()

        self.ignore_index = ignore_index
        self.ratio = ratio

    def forward(self, cls_pred, binary_pred, cls_true):

        valid_mask = (cls_true != self.ignore_index)
        fgp = torch.sigmoid(binary_pred)
        clsp = torch.softmax(cls_pred, dim=1)
        joint_prob = torch.clone(clsp)
        joint_prob[:, 0, :, :] = fgp[:, 0, :, :] * clsp[:, 0, :, :]
        joint_prob[:, 1:, :, :] = fgp[:, 1:, :, :] * clsp[:, 1:, :, :]
        Z = torch.sum(joint_prob, dim=1, keepdim=True)
        p_ci = joint_prob / Z
        p_ci = torch.clamp(p_ci, min=1e-7, max=1 - 1e-7)
        losses = F.nll_loss(torch.log(p_ci), cls_true.long(), ignore_index=self.ignore_index, reduction='none')
        return losses.sum() / valid_mask.sum()
    

class DiceLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, output, targ):
        """
        output is NCHW, targ is NHW
        """
        return dice_loss(output, targ)

    def activation(self, output):
        return F.softmax(output, dim=1)
    
    def decodes(self, output):
        return output.argmax(1)        