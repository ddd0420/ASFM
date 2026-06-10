import torch
import torch.fft
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from skimage.measure import label

def gaussian_ramp_up(iter_num, iterations, wight_gaussian):
    t = iter_num / iterations
    if t <= 1:
        return np.exp(-5 * (1 - iter_num / iterations)**2) * wight_gaussian
    else:
        return wight_gaussian

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)      
    return probs

def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i] #== c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)          
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

        batch_array = np.array(batch_list)
        batch_tensor = torch.tensor(batch_array).cuda()

    return batch_tensor


# mask image modeling
def ABCP(unlabeled_batch, ema_output, labeled_batch, label_batch, p_size, mask_num, iter_num):

    probs = torch.softmax(ema_output, dim=1)  # 在类别维度上计算 softmax
    pseudo_label = torch.argmax(probs, dim=1)  # 获取每个像素的类别预测
    confidence, _ = torch.max(probs, dim=1)  # 选择最大概率作为置信度 # torch.Size([24, 256, 256])
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # 添加小常数以避免 log(0)
    pseudo_label_bcp = get_ACDC_masks(ema_output, nms=1)

    # volume_batch torch.Size([12, 1, 256, 256]), entropy  torch.Size([12, 16, 4096])
    # volume_patch torch.Size([12, 256*256/p_size*p_size, p_size * p_size])
    unlabel_patch = rearrange(unlabeled_batch.squeeze(1), 'b (h p1) (w p2) -> b (h w)(p1 p2) ', p1=p_size, p2=p_size)
    patch_uncertainty = rearrange(entropy, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=p_size, p2=p_size)
    # patch_confidence = rearrange(confidence, 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=p_size, p2=p_size)
    patch_pseudo_label = rearrange(pseudo_label_bcp, 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=p_size, p2=p_size)
    label_patch = rearrange(labeled_batch.squeeze(1), 'b (h p1) (w p2) -> b (h w)(p1 p2) ', p1=p_size, p2=p_size)
    label_patch_gt = rearrange(label_batch, 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=p_size, p2=p_size)

    # 计算每个 patch 中前景类别的数量
    batch_size, num_patches, patch_size_squared = unlabel_patch.shape  # torch.Size([12, 64, 1024])
    total_class_num = ema_output.shape[1]  # torch.Size([24, 4, 256, 256])

    # 计算每个 patch 中前景像素的总数量（不包括背景）, 假设景像素的类别是0
    gt_foreground_mask = label_patch_gt != 0  # 创建一个掩码，标记前景像素
    gt_foreground_pixels = gt_foreground_mask.sum(dim=2)  # 每个 patch 中前景像素的总数

    pseudo_foreground_mask = patch_pseudo_label != 0  # 创建一个掩码，标记前景像素
    pseudo_foreground_pixels = pseudo_foreground_mask.sum(dim=2)  # 每个 patch 中前景像素的总数

    # 计算每个 patch 中前景像素的类别数量
    gt_foreground_classes = torch.zeros((batch_size, num_patches), dtype=torch.int)
    pseudo_foreground_classes = torch.zeros((batch_size, num_patches), dtype=torch.int)
    for b in range(batch_size):
        for p in range(num_patches):
            # 使用 unique 函数来获取每个 patch 中的前景类别
            current_patch_labels = label_patch_gt[b, p, gt_foreground_mask[b, p]]
            gt_unique_classes = current_patch_labels.unique()
            gt_foreground_classes[b, p] = (gt_unique_classes != 0).sum()  # 只计算非背景类别

            current_patch_pseudos = patch_pseudo_label[b, p, pseudo_foreground_mask[b, p]]
            pseudo_unique_classes = current_patch_pseudos.unique()
            pseudo_foreground_classes[b, p] = (pseudo_unique_classes != 0).sum()  # 只计算非背景类别

    gt_classes = gt_foreground_classes.clone().detach().cuda()
    pseudo_classes = pseudo_foreground_classes.clone().detach().cuda()

    # 计算前景类别数量所占比例.  torch.Size([12, 16])
    gt_foreground_ratios = (gt_foreground_pixels / patch_size_squared).clone().detach().cuda()
    pseudo_foreground_ratios = (pseudo_foreground_pixels / patch_size_squared).clone().detach().cuda()

    # 计算每个 patch 的 category_value. m*(n*log n), m=fg_pixels/total, n=category
    pseudo_classes_value = pseudo_classes * torch.exp(gt_foreground_ratios + 1e-6)
    gt_classes_value = gt_classes * torch.exp(pseudo_foreground_ratios + 1e-6)
    # gt_classes_value = gt_foreground_ratios * (gt_classes * torch.log(gt_classes + 1e-6))

    # 计算每个 unlabeled patch 的 mean uncertainty
    patch_mean_uncertainty = torch.mean(patch_uncertainty.detach(), dim=2)  # torch.Size([24, 64, 1024])
    # patch_mean_confidence = torch.mean(patch_confidence.detach(), dim=2)  # torch.Size([24, 64, 1024])

    # 计算每个 unlabeled patch 的混合分数
    unlabeled_patch_values = patch_mean_uncertainty * pseudo_classes_value  # torch.Size([24, 64])

    # 对 patch_value 进行从大到小排序,descending=True 表示降序
    patch_sorted_values, patch_sorted_indices = torch.sort(unlabeled_patch_values, dim=1, descending=True)
    # gt_sorted_values, gt_sorted_indices = torch.sort(gt_foreground_ratios, dim=1, descending=True)

    # 选择取其中的哪几个patch
    gaussian_w = gaussian_ramp_up(iter_num, iterations=10000, wight_gaussian=1) 
    sorted_start = int(gaussian_w * (num_patches-mask_num-1)) # 起始索引
    sorted_end = sorted_start + mask_num  # 结束索引

    if sorted_end > num_patches:
        print("The selected range is incorrect")

    patch_sorted_values = patch_sorted_values[:, sorted_start:sorted_end]
    patch_sorted_indices = patch_sorted_indices[:, sorted_start:sorted_end]
    
    # gt 的每次都取最高的
    gt_sorted_values, gt_sorted_indices = gt_classes_value.topk(mask_num, dim=1)

    h1 = int(np.sqrt(num_patches))
    # patch_zeros = torch.ones_like(unlabel_patch)
    # 对每个 batch 和每个 top k 进行替换
    for batch in range(batch_size):  # batch size
        for k in range(patch_sorted_indices.size(1)):  # mask_topk_num
            # 获取当前索引
            index = patch_sorted_indices[batch, k]
            index_gt = gt_sorted_indices[batch, k]
            # print('index = ', index)
            # print('index2 = ', index_gt)

            # 将 label_patch 中的patch替换到 unlabel_patch 中的对应位置
            unlabel_patch[batch, index] = label_patch[batch, index_gt] 
            patch_pseudo_label[batch, index] = label_patch_gt[batch, index_gt] 

    masked_unlabel_patch = rearrange(unlabel_patch, 'b (h w)(p1 p2) -> b (h p1) (w p2)',
                                  h=h1, w=h1, p1=p_size, p2=p_size)
    masked_pseudo_label = rearrange(patch_pseudo_label, 'b (h w)(p1 p2) -> b (h p1) (w p2)',
                                  h=h1, w=h1, p1=p_size, p2=p_size)

    masked_unlabel_patch = masked_unlabel_patch.unsqueeze(1)
    masked_pseudo_label = masked_pseudo_label

    return masked_unlabel_patch, masked_pseudo_label

