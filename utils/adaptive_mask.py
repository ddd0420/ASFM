import torch
import torch.fft
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
from einops import rearrange
import torch.nn.functional as F


# mask image modeling
def MaskGenerator(output, volume_batch, p_size, mask_num):

    pseudo_label = torch.argmax(output, dim=1)  # 获取每个像素的类别预测
    probs = F.softmax(output, dim=1)  # 在类别维度上计算 softmax
    confidence, _ = torch.max(probs, dim=1)  # 选择最大概率作为置信度 # torch.Size([24, 256, 256])
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # 添加小常数以避免 log(0)

    # volume_batch torch.Size([24, 1, 256, 256]), entropy torch.Size([24, 256, 256])
    # volume_patch torch.Size([24, 256*256/p_size*p_size, p_size * p_size])
    volume_patch = rearrange(volume_batch.squeeze(1), 'b (h p1) (w p2) -> b (h w)(p1 p2) ', p1=p_size, p2=p_size)
    patch_uncertainty = rearrange(entropy, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=p_size, p2=p_size)
    patch_pseudo_label = rearrange(pseudo_label, 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=p_size, p2=p_size)
    patch_confidence = rearrange(confidence, 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=p_size, p2=p_size)

    # 计算每个 patch 中前景类别的数量
    batch_size, num_patches, patch_size_squared = volume_patch.shape  # torch.Size([12, 64, 1024])
    total_class_num = output.shape[1]  # torch.Size([24, 4, 256, 256])
    category_counts = torch.zeros((batch_size, num_patches, total_class_num), dtype=torch.int)

    for i in range(total_class_num):  # 遍历每个类别
        category_counts[:, :, i] = (patch_pseudo_label == i).sum(dim=2)
    # 假设类别 1、2、3 为前景类别，类别 0 为背景
    foreground_counts = category_counts[:, :, 1:].sum(dim=2)  # 统计前景类别的数量
    # 计算前景类别所占比例
    foreground_ratios = (foreground_counts / patch_size_squared).clone().detach().cuda()
    # class_value = -foreground_ratios * torch.log(foreground_ratios + 1e-6)  # 计算 class_value # torch.Size([24, 64]
    # region_impurity = torch.sum(-dist * torch.log(dist + 1e-6), dim=1, keepdim=True) / math.log(4)  # [1, 1, h, w]

    # 计算每个 patch 的 uncertainty
    patch_mean_uncertainty = torch.mean(patch_uncertainty.detach(), dim=2)  # torch.Size([24, 64, 1024])
    patch_mean_confidence = torch.mean(patch_confidence.detach(), dim=2)  # torch.Size([24, 64, 1024])
    patch_value = patch_mean_uncertainty * foreground_ratios  # torch.Size([24, 64])

    # topk_value, topk_indices = patch_value.topk(mask_num, dim=1)
    topk_value, topk_indices = patch_value.topk(mask_num, dim=1)

    aaa = int(256*256 / patch_size_squared)
    h1 = int(np.sqrt(aaa))
    volume_patch = volume_patch.clone()

    # 首先，创建一个与 patch_zeros 形状相同的替换张量
    patch_zeros = torch.zeros_like(volume_patch)

    # 对每个 batch 和每个 top k 进行替换
    for batch in range(topk_indices.size(0)):  # 24 个 batch
        for k in range(topk_indices.size(1)):  # 4 个 top k
            # 获取当前索引
            index = topk_indices[batch, k]
            # 将 patch_zeros 中的样本替换到 volume_patch 中的对应位置
            volume_patch[batch, index] = patch_zeros[batch, k]  # 替换对应位置的 patch

    masked_topk_batch = rearrange(volume_patch,'b (h w)(p1 p2) -> b (h p1) (w p2)', 
                                                h=h1, w=h1, p1=p_size, p2=p_size)
    masked_topk_batch = masked_topk_batch.unsqueeze(1)

    return masked_topk_batch
