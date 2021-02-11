"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 srgan.py'
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="xgc_images-d3d_coarse_v2", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
parser.add_argument('--nchannel', help='num. of channels', type=int, default=3)
group = parser.add_mutually_exclusive_group()
group.add_argument('--N20', help='N20 model', action='store_const', dest='model', const='N20')
group.add_argument('--N200', help='N200 model', action='store_const', dest='model', const='N200')
group.add_argument('--N1000', help='N1000 model', action='store_const', dest='model', const='N1000')
parser.set_defaults(model='N20')
opt = parser.parse_args()
print(opt)

os.makedirs("images-%s"%opt.dataset_name, exist_ok=True)
os.makedirs("saved_models-%s"%opt.dataset_name, exist_ok=True)

cuda = torch.cuda.is_available()

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorResNet(in_channels=opt.channels, out_channels=opt.channels)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
modelfile = 'xgc-vgg19-ch%d-%s.torch'%(opt.nchannel, opt.model)
feature_extractor = XGCFeatureExtractor(modelfile)

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()
#criterion_content = torch.nn.MSELoss()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset("./%s" % opt.dataset_name, hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))
        #print (imgs_lr.shape, imgs_lr.min(), imgs_lr.max(), imgs_lr.mean())

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        # Content loss
        # _gen_hr = torch.cat((gen_hr, gen_hr, gen_hr), dim=1)
        # _imgs_hr = torch.cat((imgs_hr, imgs_hr, imgs_hr), dim=1)
        # gen_features = feature_extractor(_gen_hr)
        # real_features = feature_extractor(_imgs_hr)
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())
        #loss_content = criterion_content(gen_hr, imgs_hr)

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n"
            % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_lr, gen_hr), -1)
            save_image(img_grid, "images-%s/%d.png" % (opt.dataset_name, batches_done), normalize=False)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models-%s/generator_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models-%s/discriminator_%d.pth" % (opt.dataset_name, epoch))
