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
import logging

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
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from (default: %(default)s)")
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training (default: %(default)s)")
parser.add_argument("--dataset_name", type=str, default="xgc_images-d3d_coarse_v2", help="name of the dataset (default: %(default)s)")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches (default: %(default)s)")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate (default: %(default)s)")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient (default: %(default)s)")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient (default: %(default)s)")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay (default: %(default)s)")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation (default: %(default)s)")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height (default: %(default)s)")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width (default: %(default)s)")
parser.add_argument("--channels", type=int, default=1, help="number of image channels (default: %(default)s)")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between saving image samples (default: %(default)s)")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints (default: %(default)s)")
parser.add_argument('--nchannel', type=int, default=3, help='num. of channels (default: %(default)s)')
parser.add_argument('--modelfile', help='modelfile (default: %(default)s)')
group = parser.add_mutually_exclusive_group()
group.add_argument('--N20', help='N20 model', action='store_const', dest='model', const='N20')
group.add_argument('--N200', help='N200 model', action='store_const', dest='model', const='N200')
group.add_argument('--N1000', help='N1000 model', action='store_const', dest='model', const='N1000')
parser.set_defaults(model='N20')
opt = parser.parse_args()

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.DEBUG)

logging.info("Command: {0}\n".format(" ".join([x for x in sys.argv]))) 
logging.debug("All settings used:") 
for k,v in sorted(vars(opt).items()): 
    logging.debug("\t{0}: {1}".format(k,v))

imgdir = 'srgan_images-ch%d-%s-%s'%(opt.nchannel, opt.model, opt.dataset_name)
modeldir = 'srgan_models-ch%d-%s-%s'%(opt.nchannel, opt.model, opt.dataset_name)
os.makedirs(imgdir, exist_ok=True)
os.makedirs(modeldir, exist_ok=True)

cuda = torch.cuda.is_available()

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorResNet(in_channels=opt.channels, out_channels=opt.channels)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
modelfile = 'xgc-vgg19-ch%d-%s.torch'%(opt.nchannel, opt.model) if opt.modelfile is None else opt.modelfile
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
    generator.load_state_dict(torch.load("%s/generator_%d.pth"%(modeldir, opt.epoch)))
    discriminator.load_state_dict(torch.load("%s/discriminator_%d.pth"%(modeldir, opt.epoch)))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset("%s" % opt.dataset_name, hr_shape=hr_shape),
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

        lr_shape = tuple(imgs_lr.shape[2:])
        n = (64-lr_shape[0])//2
        imgs_lr = nn.ZeroPad2d(n)(imgs_lr)

        n = (256-hr_shape[0])//2
        imgs_hr = nn.ZeroPad2d(n)(imgs_hr)

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
        # valid.shape: torch.Size([16, 1, 16, 16])
        # fake.shape: torch.Size([16, 1, 16, 16])
        # discriminator(gen_hr).shape: torch.Size([16, 1, 16, 16])
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        # Content loss
        if opt.nchannel == 3:
            _gen_hr = torch.cat((gen_hr, gen_hr, gen_hr), dim=1)
            _imgs_hr = torch.cat((imgs_hr, imgs_hr, imgs_hr), dim=1)
            gen_features = feature_extractor(_gen_hr)
            real_features = feature_extractor(_imgs_hr)
        else:
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
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4, mode='bicubic')
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_lr, imgs_hr, gen_hr), -1)
            save_image(img_grid, "%s/%d.png" % (imgdir, batches_done), normalize=False)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "%s/generator_%d.pth" % (modeldir, epoch))
        torch.save(discriminator.state_dict(), "%s/discriminator_%d.pth" % (modeldir, epoch))
