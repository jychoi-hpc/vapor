import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
#mean = np.array([0.485, 0.456, 0.406])
#std = np.array([0.229, 0.224, 0.225])
mean = np.array([0.121008])
std = np.array([0.217191])
#(0.1140694549214342, 0.11539442)

class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                # transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                # transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        #self.files = sorted(glob.glob(root + "/*.*"))
        self.hr_files = sorted(glob.glob(root + "-hr/*.*"))
        self.lr_files = sorted(glob.glob(root + "-lr/*.*"))

    def __getitem__(self, index):
        img_lr = Image.open(self.lr_files[index % len(self.lr_files)])
        img_hr = Image.open(self.hr_files[index % len(self.hr_files)])

        img_lr = self.lr_transform(img_lr)
        img_hr = self.hr_transform(img_hr)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.hr_files)
