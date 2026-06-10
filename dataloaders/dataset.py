import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
# import matplotlib.pyplot as plt
from PIL import Image
# import pywt
from copy import deepcopy

class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        
        # UniMatch valtest, inference during training not after training    
        elif self.split == "valtest":
            with open(self._base_dir + "/valtest.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample


class GLASDataSets(Dataset):
    """ GLAS Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/test.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} {} samples".format(len(self.sample_list), self.split))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            image = Image.open(self._base_dir + "/image/" + case + ".bmp").convert('L')
            label = Image.open(self._base_dir + "/mask_01/" + case + ".bmp").convert('L')
            image = np.array(image)
            label = np.array(label)
            # image = cv2.imread(self._base_dir + "/image/" + case + ".bmp", 0) # 0 means read image with gray style
            # label = cv2.imread(self._base_dir + "/mask/" + case + ".bmp", 0)
        else:
            image = Image.open(self._base_dir + "/image/" + case + ".bmp").convert('L')
            label = Image.open(self._base_dir + "/mask_01/" + case + ".bmp").convert('L')
            image = np.array(image)
            label = np.array(label)

        sample = {'image': image, 'label': label}
        # print('image.shape = ',image.shape)

        if self.split == "train":
            sample = self.transform(sample)
        # sample["idx"] = idx
        sample['case'] = case
        return sample


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)
    return jitter(image)


class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample

def wavelet_mask(image, mask_ratio_h = 0.2, mask_ratio_l = 0.2, wavelet_type = 'db2', random_mask=0.8):
    # 执行小波变换，获取高频图像和低频图像，小波变换后图像尺寸变小
    # wavelet_type='haar, db2, bior1.5, bior2.4, coif1, dmey'
    cA, (cH, cV, cD) = pywt.dwt2(image, wavelet_type)
    
    # random.random()函数生成0到1的随机符点数
    if random.random() < random_mask:
        # 随机屏蔽高频信息
        mask_ratio_high = mask_ratio_h  # 屏蔽高频信息的比例
        mask_h = np.random.choice([0, 1], size=cH.shape, p=[mask_ratio_high, 1 - mask_ratio_high])
        cH_masked = cH * mask_h
        cV_masked = cV * mask_h
        cD_masked = cD * mask_h

        # 随机屏蔽低频信息
        mask_ratio_low = mask_ratio_l  # 屏蔽低频信息的比例
        mask_l = np.random.choice([0, 1], size=cA.shape, p=[mask_ratio_low, 1 - mask_ratio_low])
        cA_masked = cA * mask_l

        # 重构图像
        low_freq_image_masked = pywt.idwt2((cA_masked, (None, None, None)), wavelet_type)
        high_freq_image_masked = pywt.idwt2((None, (cH_masked, cV_masked, cD_masked)), wavelet_type)
        reconstructed_image = low_freq_image_masked + high_freq_image_masked

    else:
        # 原始图像
        high_freq_image_masked = cH + cV + cD
        low_freq_image_masked = cA

    return high_freq_image_masked, low_freq_image_masked



class RandomGenerator_wave_mask(object):
    def __init__(self, output_size, random_mask, wavelet_type, mask_ratio_h, mask_ratio_l):
        self.output_size = output_size
        self.random_mask = random_mask
        self.wavelet_type = wavelet_type
        self.mask_ratio_h = mask_ratio_h
        self.mask_ratio_l = mask_ratio_l

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)  # label.shape (216, 256)
         
        # wavelet transformation
        image_high_freq, image_low_freq = wavelet_mask(image, self.mask_ratio_h, self.mask_ratio_l, 
                                                       self.wavelet_type, self.random_mask)
        # print('===image_low_freq.shape', image_low_freq.shape)
        # print('---label.shape', image.shape)
        
        x, y = image_low_freq.shape
        labx, laby = label.shape

        image_h = zoom(image_high_freq, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image_l = zoom(image_low_freq, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        image = zoom(image, (self.output_size[0] / labx, self.output_size[1] / laby), order=0)
        label = zoom(label, (self.output_size[0] / labx, self.output_size[1] / laby), order=0)

        # convert image to tensor
        imageH = torch.from_numpy(image_h.astype(np.float32)).unsqueeze(0)
        imageL = torch.from_numpy(image_l.astype(np.float32)).unsqueeze(0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        
        sample = {'image': image, 'imageH': imageH, 'imageL': imageL, 'label': label}
        # sample = {'imageH': imageH, 'imageL': imageL, 'label': label}
        return sample
    




class RandomGenerator_FFT(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # # get High and low frequency with FFT
        # high_frequency, low_frequency = FFT_HL(image)

        # 进行二维傅立叶变换
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)

        # 提取高频和低频成分
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        fshift[crow - 30: crow + 30, ccol - 30: ccol + 30] = 0  # 去除低频成分
        ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(ishift).real
        # 分别保存高频和低频图像
        low_frequency = image - img_back
        high_frequency = img_back

        # convert image to tensor
        imageH = torch.from_numpy(high_frequency.astype(np.float32)).unsqueeze(0)
        imageL = torch.from_numpy(low_frequency.astype(np.float32)).unsqueeze(0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        
        sample = {'image': image, 'imageH': imageH, 'imageL': imageL, 'label': label}
        # sample = {'imageH': imageH, 'imageL': imageL, 'label': label}
        return sample


import numpy as np
import random

# 传入要执行FFT的图像，要屏蔽频率信息的比率
def random_fft_mask(image, rato):
    # 执行二维傅里叶变换
    f_image = np.fft.fft2(image)

    # 创建一个与傅里叶变换后频谱大小相同的掩码
    mask = np.ones_like(f_image)
    mask_rato = rato
    # 随机屏蔽一部分高频信息
    num_freq_to_mask = int(mask_rato * mask.size)  # 假设要屏蔽10%的频率信息
    high_freq_indices = list(zip(*np.where(np.abs(f_image) > np.mean(np.abs(f_image)))))[1:]
    random.shuffle(high_freq_indices)
    for i in range(num_freq_to_mask):
        mask[high_freq_indices[i]] = 0

    # 随机屏蔽一部分低频信息
    num_freq_to_mask = int(mask_rato * mask.size)  # 假设要屏蔽10%的频率信息
    low_freq_indices = list(zip(*np.where(np.abs(f_image) <= np.mean(np.abs(f_image)))))[1:]
    random.shuffle(low_freq_indices)
    for i in range(num_freq_to_mask):
        mask[low_freq_indices[i]] = 0

    # 对经过屏蔽的傅里叶变换执行逆变换
    masked_image = np.fft.ifft2(f_image * mask).real

    return masked_image  

# 另一种屏蔽高低频方法
import numpy as np
import cv2

# def random_freq_mask(image, low_freq_ratio=0.5, high_freq_ratio=0.5):
#     fft_image = np.fft.fft2(image)
#     mag_spectrum = np.abs(fft_image)
    
#     low_freq_mask = np.random.choice([0, 1], size=mag_spectrum.shape, p=[low_freq_ratio, 1-low_freq_ratio])
#     high_freq_mask = np.random.choice([0, 1], size=mag_spectrum.shape, p=[high_freq_ratio, 1-high_freq_ratio])
    
#     masked_fft = fft_image * high_freq_mask * low_freq_mask
#     masked_image = np.fft.ifft2(masked_fft).real
    
#     return masked_image

def random_freq_mask(image, high_freq_ratio=0.5, low_freq_ratio=0.5, random_mask=0.8):
    if random.random() < random_mask:
        # print('==random run')
        fft_image = np.fft.fft2(image)
        mag_spectrum = np.abs(fft_image)
        
        high_freq_mask = np.random.choice([0, 1], size=mag_spectrum.shape, p=[high_freq_ratio, 1-high_freq_ratio])
        low_freq_mask = np.random.choice([0, 1], size=mag_spectrum.shape, p=[low_freq_ratio, 1-low_freq_ratio])
        
        masked_fft = fft_image * high_freq_mask * low_freq_mask
        masked_image = np.fft.ifft2(masked_fft).real
    else:
        masked_image = image

    return masked_image

class RandomGenerator_FFT_Mask(object):
    def __init__(self, output_size, mask_ratio_h, mask_ratio_l, random_mask):
        self.output_size = output_size
        self.mask_ratio_h = mask_ratio_h
        self.mask_ratio_l = mask_ratio_l
        self.random_mask = random_mask

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # image = deepcopy(image)
        # give image for FFT and FFT rato
        image = random_freq_mask(image, self.mask_ratio_h, self.mask_ratio_l, self.random_mask)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample

  
class RandomGenerator_FFT_Mask2(object):
    def __init__(self, output_size, mask_ratio_h, mask_ratio_l, random_mask):
        self.output_size = output_size
        self.mask_ratio_h = mask_ratio_h
        self.mask_ratio_l = mask_ratio_l
        self.random_mask = random_mask

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        # print('===image.shape', image.shape)
        # print('===label.shape', label.shape)
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # image = deepcopy(image)
        # give image for FFT and FFT rato
        image_masked = random_freq_mask(image, self.mask_ratio_h, self.mask_ratio_l, self.random_mask)

        image_masked = torch.from_numpy(image_masked.astype(np.float32)).unsqueeze(0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "image_masked": image_masked, "label": label}
        return sample




class RandomGenerator_wavelet(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # image = np.array(image)
        wavelet_type = 'db2' # default='db2', help='haar, db2, bior1.5, bior2.4, coif1, dmey'

        LL, (LH, HL, HH) = pywt.dwt2(image, wavelet_type)

        LL = (LL - LL.min()) / (LL.max() - LL.min()) * 255

        # LL = Image.fromarray(LL.astype(np.uint8))
        # LL.save(L_path)

        LH = (LH - LH.min()) / (LH.max() - LH.min()) * 255
        HL = (HL - HL.min()) / (HL.max() - HL.min()) * 255
        HH = (HH - HH.min()) / (HH.max() - HH.min()) * 255

        merge_H = HH + HL + LH
        merge_H = (merge_H-merge_H.min()) / (merge_H.max()-merge_H.min()) * 255

        # merge_H = Image.fromarray(merge1.astype(np.uint8))
        # merge_H.save(H_path)
        x, y = merge_H.shape
        imgx, imgy = image.shape
        labx, laby = label.shape

        L = zoom(LL, (self.output_size[0] / x, self.output_size[1] / y), order=0) # reshape
        H = zoom(merge_H, (self.output_size[0] / x, self.output_size[1] / y), order=0) # reshape
        image = zoom(image, (self.output_size[0] / imgx, self.output_size[1] / imgy), order=0)
        label = zoom(label, (self.output_size[0] / labx, self.output_size[1] / laby), order=0) # image.shape (256, 256)
    

        imageL = torch.from_numpy(L.astype(np.float32)).unsqueeze(0)
        imageH = torch.from_numpy(H.astype(np.float32)).unsqueeze(0)
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        
        sample = {'image': image, 'imageH': imageH, 'imageL': imageL, 'label': label}

        return sample



class RandomGenerator_IMG_DB(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        # divide high and low frequency with wavelet type DB2
        LL, (LH, HL, HH) = pywt.dwt2(image, 'db2') # help='haar, db2, bior1.5, bior2.4, coif1, dmey'
        merge1 = HH + HL + LH
        x, y = merge1.shape
        imgx, imgy = image.shape
        labx, laby = label.shape

        L = zoom(LL, (self.output_size[0] / x, self.output_size[1] / y), order=0) # reshape
        H = zoom(merge1, (self.output_size[0] / x, self.output_size[1] / y), order=0) # reshape
        image = zoom(image, (self.output_size[0] / imgx, self.output_size[1] / imgy), order=0)
        label = zoom(label, (self.output_size[0] / labx, self.output_size[1] / laby), order=0) # image.shape (256, 256)
        # print('===L.shape',L.shape)
        # print('===H.shape',H.shape)
        # print('===image.shape',image.shape)

        imageL = torch.from_numpy(L.astype(np.float32)).unsqueeze(0)
        imageH = torch.from_numpy(H.astype(np.float32)).unsqueeze(0)
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        
        sample = {'image': image, 'imageH': imageH, 'imageL': imageL, 'label': label}

        return sample



class RandomGenerator_IMG_FFT(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # # get High and low frequency with FFT
        # high_frequency, low_frequency = FFT_HL(image)

        # 进行二维傅立叶变换
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
            
        # 提取高频和低频成分
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        fshift[crow - 30: crow + 30, ccol - 30: ccol + 30] = 0  # 去除低频成分
        ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(ishift).real
        # 分别保存高频和低频图像
        low_frequency = image - img_back
        high_frequency = img_back

        # convert image to tensor
        imageH = torch.from_numpy(high_frequency.astype(np.float32)).unsqueeze(0)
        imageL = torch.from_numpy(low_frequency.astype(np.float32)).unsqueeze(0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        
        sample = {'image': image, 'imageH': imageH, 'imageL': imageL, 'label': label}

        return sample

class WeakStrongAugment(object):
    """returns weakly and strongly augmented images
    Args:
        object (tuple): output size of network
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:  
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        # weak augmentation is rotation / flip
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y),order=0)  
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # strong augmentation is color jitter
        image_strong, label_strong = cutout_gray(image,label,p=0.5)
        image_strong = color_jitter(image_strong).type("torch.FloatTensor")
        # image_strong = blur(image, p=0.5)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # image_strong = torch.from_numpy(image_strong.astype(np.float32)).unsqueeze(0)

        label = torch.from_numpy(label.astype(np.uint8))
        label_strong = torch.from_numpy(label_strong.astype(np.uint8))
        sample = {
            "image": image,
            "image_strong": image_strong,
            "label": label,
            "label_strong": label_strong}
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)






class WeakStrongAugment_init(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size, shuffle = True):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        self.shuffle = shuffle
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        if self.shuffle:
            primary_iter = iterate_once(self.primary_indices)
        else:
            primary_iter = self.primary_indices 
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
