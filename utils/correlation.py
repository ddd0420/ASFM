import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np

# Crop
def generate_mask(pred):
    batch_size, channel, pred_x, pred_y = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
    patch_x, patch_y = int(pred_x//3), int(pred_y//3)
    w = np.random.randint(0, pred_x - patch_x)
    h = np.random.randint(0, pred_y - patch_y)
    return w,patch_x,h,patch_y

#Correlation Calculate (h/3,w/3)
def Cal_Correlation(pred):
    h_out, w_out = pred.shape[2], pred.shape[3]
    h_in = h_out
    w_in = w_out
    out = F.interpolate(pred.detach(), (h_in, w_in), mode='bilinear', align_corners=True)
    pred1 = rearrange(out, 'n c h w -> n c (h w)')
    pred2 = pred1.clone()
    corr_map = torch.matmul(pred1.transpose(1, 2), pred2) / torch.sqrt(torch.tensor(pred2.shape[1]).float())
    corr_map = F.softmax(corr_map, dim=-1)
    return corr_map

# 显存需求太大，考虑将prediction直接crop后传入
def Local_Correlation(pred1,pred2):
    w, patch_x, h, patch_y = generate_mask(pred1)
    crop_region1 = pred1[:,:,w:w+patch_x, h:h+patch_y]
    crop_region2 = pred2[:,:,w:w+patch_x, h:h+patch_y]
    corr_1 = Cal_Correlation(crop_region1)
    corr_1 = normalize_corr_map(corr_1)
    corr_2 = Cal_Correlation(crop_region2)
    corr_2 = normalize_corr_map(corr_2)
    return corr_1, corr_2

def normalize_corr_map(corr_map):
    n, h, w = corr_map.shape
    corr_map = rearrange(corr_map, 'n h w -> n (h w)')
    range_ = torch.max(corr_map, dim=1, keepdim=True)[0] - torch.min(corr_map, dim=1, keepdim=True)[0]
    temp_map = ((- torch.min(corr_map, dim=1, keepdim=True)[0]) + corr_map) / (range_ + 1e-9)
    norm_corr_map = rearrange(temp_map, 'n (h w) -> n h w', n=n, h=h, w=w)
    return norm_corr_map

# pred1 = torch.randn((2,4,256,256))
# pred2 = torch.randn((2,4,256,256))
# corr1, corr2 = Local_Correlation(pred1,pred2)
# loss = torch.mean(corr1-corr2)
# print(loss)

