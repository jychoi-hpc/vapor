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

import adios2 as ad2

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from (default: %(default)s)")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training (default: %(default)s)")
parser.add_argument("--dataset_name", type=str, default="nstx", help="name of the dataset (default: %(default)s)")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches (default: %(default)s)")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate (default: %(default)s)")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient (default: %(default)s)")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient (default: %(default)s)")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay (default: %(default)s)")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation (default: %(default)s)")
parser.add_argument("--hr_height", type=int, default=64, help="high res. image height (default: %(default)s)")
parser.add_argument("--hr_width", type=int, default=80, help="high res. image width (default: %(default)s)")
# parser.add_argument("--channels", type=int, default=1, help="number of image channels (default: %(default)s)")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between saving image samples (default: %(default)s)")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints (default: %(default)s)")
parser.add_argument('--nchannel', type=int, default=1, help='num. of channels (default: %(default)s)')
parser.add_argument('--modelfile', help='modelfile (default: %(default)s)')
parser.add_argument("--nframes", type=int, default=16_000, help="number of frames to load")
parser.add_argument('--nofeatureloss', help='no feature loss', action='store_true')
group = parser.add_mutually_exclusive_group()
group.add_argument('--VGG', help='use VGG 3-channel model', action='store_const', dest='model', const='VGG')
group.add_argument('--N1024', help='use XGC 1-channel N1024 model', action='store_const', dest='model', const='N1024')
parser.set_defaults(model='N1024')
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
logging.debug('imgdir: %s'%imgdir)
logging.debug('modeldir: %s'%modeldir)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hr_shape = (opt.hr_height, opt.hr_width)
print ('hr_shape:', hr_shape)

# Initialize generator and discriminator
generator = GeneratorResNet(in_channels=opt.nchannel, out_channels=opt.nchannel)
discriminator = Discriminator(input_shape=(opt.nchannel, *hr_shape))
if opt.model == 'VGG':
    assert (opt.nchannel == 3)
    feature_extractor = FeatureExtractor()
else:
    modelfile = 'nstx-vgg19-ch%d-%s.torch'%(opt.nchannel, opt.model) if opt.modelfile is None else opt.modelfile
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
    fname0 = "%s/generator_%d.pth"%(modeldir, opt.epoch)
    fname1 = "%s/discriminator_%d.pth"%(modeldir, opt.epoch)
    logging.debug ('Loading: %s %s'%(fname0, fname1))
    generator.load_state_dict(torch.load(fname0))
    discriminator.load_state_dict(torch.load(fname1))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

mean = 0.121008*2
std = 0.217191

# %%
offset = 159065
length = opt.nframes
with ad2.open('nstx_data_ornl_demo_v2.bp','r') as f:
    start=(offset,0,0) 
    count=(length,64,80)
    gpiData = f.read('gpiData', start=start, count=count)
print (gpiData.shape)

X = gpiData.astype(np.float32)
xmin = np.min(X, axis=(1,2))
xmax = np.max(X, axis=(1,2))
X = (X-xmin[:,np.newaxis,np.newaxis])/(xmax-xmin)[:,np.newaxis,np.newaxis]

X_lr, X_hr, = torch.tensor(X[:,np.newaxis,::4,::4]), torch.tensor(X[:,np.newaxis,:,:])
if opt.nchannel == 3:
    X_lr = torch.cat((X_lr,X_lr,X_lr), axis=1)
    X_hr = torch.cat((X_hr,X_hr,X_hr), axis=1)
training_data = torch.utils.data.TensorDataset(X_lr, X_hr)
dataloader = torch.utils.data.DataLoader(training_data, batch_size=opt.batch_size, shuffle=True)

# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    abs_list = list()
    for i, imgs in enumerate(dataloader):

        # Configure model input
        imgs_lr = imgs[0].to(device)
        imgs_hr = imgs[1].to(device)
        # print ('imgs_lr, imgs_hr:', list(imgs_lr.shape), list(imgs_hr.shape))
        
        # n2 = (64-imgs_lr.shape[3])//2
        # n1 = (64-imgs_lr.shape[2])//2
        # imgs_lr = F.pad(imgs_lr, (n2,n2,n1,n1), "constant", -mean/std)

        # n2 = (256-imgs_hr.shape[3])//2
        # n1 = (256-imgs_hr.shape[2])//2
        # imgs_hr = F.pad(imgs_hr, (n2,n2,n1,n1), "constant", -mean/std)
        # print (imgs_lr.shape, imgs_lr.min(), imgs_lr.max(), imgs_lr.mean())

        # Adversarial ground truths
        output_shape = discriminator.output_shape
        # (2021/03) Hard coding for now
        # output_shape = (1,8,10)
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)
        # print ('imgs_lr', imgs_lr.min().item(), imgs_lr.max().item(), imgs_lr.mean().item())
        # print ('gen_hr', gen_hr.min().item(), gen_hr.max().item(), gen_hr.mean().item())
        
        # Adversarial loss
        # valid.shape: torch.Size([16, 1, 16, 16])
        # fake.shape: torch.Size([16, 1, 16, 16])
        # discriminator(gen_hr).shape: torch.Size([16, 1, 16, 16])
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())
        #loss_content = criterion_content(gen_hr, imgs_hr)

        # Total loss
        if opt.nofeatureloss:
            loss_G = loss_GAN
        else:
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
        abserr = torch.max(torch.abs(gen_hr.detach()-imgs_hr.detach())).item()
        abs_list.append(abserr)

        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n"
            % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            _imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4, mode='nearest')
            _gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            _imgs_lr = make_grid(_imgs_lr, nrow=1, normalize=True)
            _imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
            img_grid = torch.cat((_imgs_lr, _imgs_hr, _gen_hr), -1)
            save_image(img_grid, "%s/%d.png" % (imgdir, batches_done), normalize=False)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "%s/generator_%d.pth" % (modeldir, epoch))
        torch.save(discriminator.state_dict(), "%s/discriminator_%d.pth" % (modeldir, epoch))
        logging.debug ('ABS error: %g %g %g'%(np.min(abs_list), np.mean(abs_list), np.max(abs_list)))
