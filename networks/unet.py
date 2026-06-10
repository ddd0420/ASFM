# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

    
class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=False):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Decoder_MS(nn.Module):
    def __init__(self, params):
        super(Decoder_MS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)  
        self.downsample = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)

    def forward(self, feature, down_scale=False, up_scale=False):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)

        if down_scale:
            output = self.upsample(output)
            return output

        if up_scale:
            output = self.downsample(output)
            output = self.avg_pool(output)
            return output
 
        return output


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)  # shape torch.Size([6, 16, 256, 256])
        x1 = self.down1(x0)  # torch.Size([6, 32, 128, 128])
        x2 = self.down2(x1)  # torch.Size([6, 64, 64, 64])
        x3 = self.down3(x2)  # torch.Size([6, 128, 32, 32])
        x4 = self.down4(x3)  # torch.Size([6, 256, 16, 16])

        return [x0, x1, x2, x3, x4]



class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output


class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output

class UNet_FEA(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_FEA, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return feature, output


# DownBlock for reconstruction
# class Down_Block(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout):
#         super(Down_Block, self).__init__()
#         self.down = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x):
#         return self.down(x)


class Decoder_re(nn.Module):
    def __init__(self, params):
        super(Decoder_re, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        # up sampling and conv block
        self.up1 = nn.ConvTranspose2d(self.ft_chns[4], self.ft_chns[3], kernel_size=2, stride=2)
        self.conv1 = ConvBlock(self.ft_chns[3] + self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)

        self.up2 = nn.ConvTranspose2d(self.ft_chns[3], self.ft_chns[2], kernel_size=2, stride=2)
        self.conv2 = ConvBlock(self.ft_chns[2] + self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)

        self.up3 = nn.ConvTranspose2d(self.ft_chns[2], self.ft_chns[1], kernel_size=2, stride=2)
        self.conv3 = ConvBlock(self.ft_chns[1] + self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)

        self.up4 = nn.ConvTranspose2d(self.ft_chns[1], self.ft_chns[0], kernel_size=2, stride=2)
        self.conv4 = ConvBlock(self.ft_chns[0] + self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.final_conv = nn.Conv2d(self.ft_chns[0], 1, kernel_size=1)  # output channel is 1

    def forward(self, encoded_features):
        x = self.up1(encoded_features[4])  # torch.Size([6, 256, 16, 16])
        x = self.conv1(torch.cat((x, encoded_features[3]), dim=1))  # 
        x = self.up2(x)
        x = self.conv2(torch.cat((x, encoded_features[2]), dim=1))  
        x = self.up3(x)
        x = self.conv3(torch.cat((x, encoded_features[1]), dim=1))  
        x = self.up4(x)
        x = self.conv4(torch.cat((x, encoded_features[0]), dim=1)) 
        x_reconstructed = self.final_conv(x)  
        return x_reconstructed

class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(ConvBlock2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.conv(x)

class UpBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock2, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock2(in_channels, out_channels))

    def forward(self, x):
        return self.up(x)

class Decoder_re2(nn.Module):
    def __init__(self, params):
        super(Decoder_re2, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up4 = UpBlock2(self.ft_chns[4], self.ft_chns[3])
        self.up3 = UpBlock2(self.ft_chns[3], self.ft_chns[2])
        self.up2 = UpBlock2(self.ft_chns[2], self.ft_chns[1])
        self.up1 = UpBlock2(self.ft_chns[1], self.ft_chns[0])
        self.out_conv = ConvBlock2(self.ft_chns[0], 1) # output channel is 1

    def forward(self, features):
        x4, x3, x2, x1, x0 = features

        x = self.up4(x4)
        x = self.up3(x + x3)
        x = self.up2(x + x2)
        x = self.up1(x + x1)
        x_reconstructed = self.out_conv(x)
        print('x_reconstructed.shape', x_reconstructed.shape)
        return x_reconstructed

class UNet_RE(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_RE, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'down_scale': False,
                  'up_scale': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.decoder_re = Decoder_re(params)

    def forward(self, x, reconstructed=False):
        feature = self.encoder(x)
        
        if reconstructed:
            output_re = self.decoder_re(feature)
            return  output_re

        output = self.decoder(feature)
        return output



def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x










class UNet_MS(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_MS, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'down_scale': False,
                  'up_scale': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder_MS(params)
        self.decoder_down = Decoder_MS(params)
        self.decoder_up = Decoder_MS(params)

    def forward(self, x, down_scale=False, up_scale=False):
        feature = self.encoder(x)

        if down_scale:
            feature_small = self.decoder_down(feature, down_scale=True)
            return feature_small

        if up_scale:
            feature_large = self.decoder_up(feature, up_scale=True)
            return feature_large

        output = self.decoder(feature)

        return output
