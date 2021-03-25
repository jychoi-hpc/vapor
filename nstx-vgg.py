from __future__ import print_function, division
import os
import time

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from skimage.transform import resize

from torchvision import datasets, models, transforms


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import adios2 as ad2
import argparse
import logging
import sys

from sklearn.model_selection import train_test_split

# %%
def visualize_model(model, dataloaders, num_images=6):
    def _imshow(inp, title=None):
        inp = inp.numpy().transpose((1, 2, 0))
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)
    
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=[16,8])

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(preds[j]))
                _imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    # plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    
def show_databatch(inputs, classes):
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[x.item() for x in classes])

def digitizing(x, nbins):
    """
    idx ranges from 1 ... nbins
    nan is 0
    """
    assert(nbins>1)
    _x = x[~np.isnan(x)]
    _, bins = np.histogram(_x, bins=nbins)
    idx = np.digitize(x, bins, right=True)
    idx = np.where(idx==0, 1, idx)
    idx = np.where(idx==nbins+1, 0, idx)
    return (idx, bins)

class XGCFDataset(Dataset):
    def __init__(self, expdir, step_list, nchannel, nclass, shape):

        self.nchannel = nchannel
        self.hr_height, self.hr_height = shape

        lx = list()
        ly = list()
        for istep in step_list:
            fname = os.path.join(expdir, 'restart_dir/xgc.f0.%05d.bp'%istep)
            logging.debug("Reading: %s"%(fname))
            with ad2.open(fname,'r') as f:
                i_f = f.read('i_f')
            i_f = np.moveaxis(i_f,1,2)
            i_f = i_f.astype(np.float32)
            nphi, nnodes, nx, ny = i_f.shape

            for iphi in range(nphi):
                for i in range(nnodes):
                    X = i_f[iphi,i,:,:]
                    X = (X - np.min(X))/(np.max(X)-np.min(X))
                    # X = X[np.newaxis,:,:]
                    # X = np.vstack([X,X,X])
                    lx.append(X)
                    ly.append(nclass[i])
        
        self.X = np.array(lx)
        self.y = np.array(ly)
        self.mean = np.mean(self.X)
        self.std = np.std(self.X)
        logging.debug("Dataset mean, std: %f %f"%(self.mean, self.std))

        self.transform = transforms.Compose(
            [
                # transforms.ToPILImage(),
                # transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                # transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )

    def __getitem__(self, index):
        i = index
        X  = self.X[i,:]
        ## 0: Nearest-neighbor, 1: Bi-linear, 2: Bi-quadratic, 3: Bi-cubic
        X = resize(X, (self.hr_height, self.hr_height), order=0, anti_aliasing=False) 
        # (2021/02) Mnay problems for using PIL with numpy
        # im = transforms.ToPILImage()(X)
        # im = transforms.Resize((self.hr_height, self.hr_height), Image.BICUBIC)(im)
        # X = np.array(im)
        X = transforms.ToTensor()(X)
        X = transforms.Normalize(self.mean, self.std)(X)
        if self.nchannel == 3:
            X = torch.cat([X, X, X])
        y = torch.tensor(self.y[i])
        # print ('X:', X.shape, X.min(), X.max(), i)
        # import pdb; pdb.set_trace()
        return (X, y)

    def __len__(self):
        return len(self.X)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--nframes", type=int, default=16_000, help="number of frames to load")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    # parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    # parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--restart', help='restart', action='store_true')
    # parser.add_argument('--timesteps', help='timesteps', nargs='+', type=int, default=[420,])
    parser.add_argument('--nchannel', help='num. of channels', type=int, default=1)
    # parser.add_argument("--hr_height", type=int, default=256, help="hr_height")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument('--regen', help='regen', action='store_true')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--N1024', help='N1024 model', action='store_const', dest='model', const='N1024')
    parser.set_defaults(model='N1024')
    opt = parser.parse_args()
    print(opt)

    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.DEBUG)

    logging.info("Command: {0}\n".format(" ".join([x for x in sys.argv]))) 
    logging.debug("All settings used:") 
    for k,v in sorted(vars(opt).items()): 
        logging.debug("\t{0}: {1}".format(k,v))

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

    if opt.model == 'N1024':
        ## 1024 classes
        if opt.regen:
            import time
            from sklearn import manifold

            n_neighbors=30
            _X = X.reshape([len(X),-1])
            print ("Embedding ...")
            t0 = time.time()
            X_iso = manifold.Isomap(n_neighbors=n_neighbors, n_components=1).fit_transform(_X)
            print ("ISOMAP (time %.2fs)"%(time.time() - t0))

            od = np.argsort(X_iso[:,0])
            ood = np.argsort(od)
            label = np.digitize(range(len(od)), np.linspace(0, len(od), 1024+1, dtype=np.int))-1
            print (min(label), max(label))
            np.save('nstx-label-%s.npy'%opt.model, label[ood])
            nclass = label
        else:
            nclass = np.load('nstx-label-%s.npy'%opt.model)[:length]
        print (nclass.shape)

    unique, counts = np.unique(nclass, return_counts=True)
    fcls = dict(zip(unique, counts)) 
    num_classes = 1024
    print ("fcls, max nclasses", len(fcls), num_classes)
    # plt.figure()
    # plt.bar(unique, counts)

    # %%
    dat = X
    nnodes, nx, ny = dat.shape
    X = X.reshape((nnodes,1,nx,ny))
    
    # %%
    X_train, X_test, y_train, y_test = train_test_split(X, nclass, test_size=0.2)

    X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
    X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)
    if opt.nchannel == 3:
        X_train = torch.cat((X_train,X_train,X_train), axis=1)
        X_test = torch.cat((X_test,X_test,X_test), axis=1)

    training_data = torch.utils.data.TensorDataset(X_train, y_train)
    validation_data = torch.utils.data.TensorDataset(X_test, y_test)

    training_loader = torch.utils.data.DataLoader(training_data, batch_size=opt.batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=opt.batch_size, shuffle=True)

    # %%
    vgg_based = torchvision.models.vgg19(pretrained=True)

    if opt.nchannel == 1:
        # Modify to use a single channel (gray)
        vgg_based.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    # Modify the last layer
    number_features = vgg_based.classifier[6].in_features
    features = list(vgg_based.classifier.children())[:-1] # Remove last layer
    features.extend([torch.nn.Linear(number_features, num_classes)])
    vgg_based.classifier = torch.nn.Sequential(*features)
    model = vgg_based
    for name, param in model.named_parameters():
        print (name, param.shape, param.requires_grad)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # %%
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = opt.n_epochs

    modelfile = 'nstx-vgg19-ch%d-%s.torch'%(opt.nchannel, opt.model)
    if opt.restart:
        model = torch.load(modelfile)
    model.to(device)
    since = time.time()    
    for epoch in range(num_epochs):
        
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)

        model.train()
        # Iterate over data.
        loss_train = 0.0
        acc_train = 0
        for i, data in enumerate(training_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            ns = torch.normal(mean=0.0, std=inputs*0.01)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs  = model(inputs+ns)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                #scheduler.step()
            
            loss_train += loss.item() * inputs.size(0)
            acc_train += torch.sum(preds == labels.data)
            # print ('model:', model.features[0].weight.sum().item())
            if (i+1) % 1 == 0:
                # print ('model:', model.features[0].weight.sum().item())
                print('[Epoch {:d}/{:d}] [Batch {:d}/{:d}] {} loss: {:.4f}'.format(epoch, num_epochs, i, len(training_loader), 'Train', loss.item()))
                
            if (i+1) % 10 == 0:
                print('Acc: ', torch.sum(preds == labels.data).item()/len(preds))
                print("Label:")
                print(labels)
                print("Pred:")
                print(preds)

        if (epoch+1) % 100 == 0:
            print('Saving:', modelfile)
            torch.save(model, modelfile)

        avg_loss = loss_train / len(training_loader)
        avg_acc = acc_train.double() / len(training_loader)
        #print('{} Loss: {:.4f} Acc: {:.4f}'.format('Epoch', epoch_loss, epoch_acc))
        torch.save(model, modelfile)
        
        print("Validation:")
        model.eval()    
        # Iterate over data.
        loss_val = 0.0
        acc_val = 0
        for i, data in enumerate(validation_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs  = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss_val += loss.item() * inputs.size(0)
            acc_val += torch.sum(preds == labels.data)
            if (i+1) % 1 == 0:
                print('[Epoch {:d}/{:d}] [Batch {:d}/{:d}] {} loss: {:.4f}'.format(epoch, num_epochs, i, len(validation_loader), 'Test', loss.item()))
            if (i+1) % 10 == 0:
                print('Acc: ', torch.sum(preds == labels.data).item()/len(preds))
                print("Label:")
                print(labels)
                print("Pred:")
                print(preds)

        avg_loss_val = loss_val / len(validation_loader)
        avg_acc_val = acc_val.double() / len(validation_loader)
        #print('{} Loss: {:.4f} Acc: {:.4f}'.format('Test', epoch_loss, epoch_acc))
        print()
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print("Label:")
        print(labels)
        print("Pred:")
        print(preds)
        print()    

        torch.save(model, modelfile)
        print('Done.')
