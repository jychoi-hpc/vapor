from __future__ import print_function, division
import os
import time

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

from torchvision import datasets, models, transforms


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import adios2 as ad2
import argparse
import logging
import sys

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    # parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    # parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--restart', help='restart', action='store_true')
    opt = parser.parse_args()
    print(opt)

    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.DEBUG)

    logging.info("Command: {0}\n".format(" ".join([x for x in sys.argv]))) 
    logging.debug("All settings used:") 
    for k,v in sorted(vars(opt).items()): 
        logging.debug("\t{0}: {1}".format(k,v))

    # %%
    # psi_surf: psi value of each surface
    # surf_len: # of nodes of each surface
    # surf_idx: list of node index of each surface
    with ad2.open('d3d_coarse_v2/xgc.mesh.bp', 'r') as f:
        nnodes = int(f.read('n_n', ))
        ncells = int(f.read('n_t', ))
        rz = f.read('rz')
        conn = f.read('nd_connect_list')
        psi = f.read('psi')
        nextnode = f.read('nextnode')
        epsilon = f.read('epsilon')
        node_vol = f.read('node_vol')
        node_vol_nearest = f.read('node_vol_nearest')
        psi_surf = f.read('psi_surf')
        surf_idx = f.read('surf_idx')
        surf_len = f.read('surf_len')
        theta = f.read('theta')

    r = rz[:,0]
    z = rz[:,1]
    print (nnodes)

    # %%
    with ad2.open('d3d_coarse_v2/restart_dir/xgc.f0.00420.bp','r') as f:
        i_f = f.read('i_f')
    i_f = np.moveaxis(i_f,1,2)
    print (i_f.shape)    

    # %%
    """
    ## 20 classes
    fmax = list()
    for i in range(len(psi_surf)):
        n = surf_len[i]
        k = surf_idx[i,:n]-1
        fmax.append(np.max(i_f[:,k,:,:]))

    lx = np.log10(fmax)
    # plt.figure(figsize=[16,4])
    # plt.plot(lx,'-x')

    bins = np.linspace(min(lx), max(lx), 9)
    inds = np.digitize(lx, bins)
    # for i in range(len(lx)):
    #     plt.text(i, lx[i], str(inds[i]))
        
    inds = inds*2
    nclass = np.zeros(len(rz), dtype=np.int)
    for i in range(len(psi_surf)):
        n = surf_len[i]
        k = surf_idx[i,:n]-1
        nclass[k] = inds[i]

    for i in range(len(rz)):
        if r[i]>r[0]: nclass[i] += 1
    """
    
    ## 202 classes
    nclass = np.zeros(len(rz), dtype=np.int)
    for i in range(len(psi_surf)):
        n = surf_len[i]
        k = surf_idx[i,:n]-1
        nclass[k] = i*2

    for i in range(len(rz)):
        if r[i]>r[0]: 
            nclass[i] = nclass[i]+1

    unique, counts = np.unique(nclass, return_counts=True)
    fcls = dict(zip(unique, counts)) 
    print ("fcls, nclasses", len(fcls), max(unique)+1)
    # plt.figure()
    # plt.bar(unique, counts)

    # %%
    dat = i_f[0,:,:,:].astype(np.float32)
    nnodes, nx, ny = dat.shape
    lx = list()
    ly = list()
    lp = list()
    for iphi in range(i_f.shape[0]):
        dat = i_f[iphi,:,:,:].astype(np.float32)
        for i in range(nnodes):
            X = dat[i,:,:]
            X = (X - np.min(X))/(np.max(X)-np.min(X))
            X = X[np.newaxis,:,:]
            lx.append(X)
            ly.append(nclass[i])
            lp.append(1/len(fcls)/fcls[nclass[i]])
    print (len(lx), len(ly))
    lx[0].shape, ly[0]

    # %%
    batch_size=opt.batch_size

    dataset = torch.utils.data.TensorDataset(torch.tensor(lx), torch.tensor(ly))

    training_data, validation_data = random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
    logging.debug("Split: %d %d"%(len(training_data), len(validation_data))) 

    p_training_data = list()
    for _, y in training_data:
        i = y.item()
        p_training_data.append(1/len(fcls)/fcls[i])

    p_validation_data = list()
    for _, y in validation_data:
        i = y.item()
        p_validation_data.append(1/len(fcls)/fcls[i])

    training_sample_size = len(fcls)*batch_size*100
    validation_sample_size = len(fcls)*batch_size*10
    logging.debug("Sample: %d %d"%(training_sample_size, validation_sample_size)) 
    sampler=WeightedRandomSampler(p_training_data, training_sample_size, replacement=True)
    training_loader = DataLoader(training_data, batch_size=batch_size, pin_memory=True, sampler=sampler, drop_last=True)

    sampler2=WeightedRandomSampler(p_validation_data, validation_sample_size, replacement=False)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, pin_memory=True, sampler=sampler2, drop_last=True)

    # %%
    vgg_based = torchvision.models.vgg19(pretrained=False)

    # Modify the last layer
    vgg_based.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    number_features = vgg_based.classifier[6].in_features
    features = list(vgg_based.classifier.children())[:-1] # Remove last layer
    num_classes = max(unique)+1
    features.extend([torch.nn.Linear(number_features, num_classes)])
    vgg_based.classifier = torch.nn.Sequential(*features)
    model = vgg_based

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # %%
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = opt.n_epochs

    if opt.restart:
        model = torch.load('xgc-vgg19.torch')
    model.to(device)
    since = time.time()    
    for epoch in range(num_epochs):
        
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print('-' * 10)

        model.train()
        # Iterate over data.
        loss_train = 0.0
        acc_train = 0
        for i, data in enumerate(training_loader):
            inputs , labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            ns = torch.normal(mean=0.0, std=inputs*0.01)
            #print (model.features[0].weight.sum().item())
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs  = model(inputs) #+ns)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                #scheduler.step()
            
            loss_train += loss.item() * inputs.size(0)
            acc_train += torch.sum(preds == labels.data)
            if (i+1) % 100 == 0:
                print('[{:d}/{:d}] {} loss: {:.4f}'.format(i, len(training_loader), 'Train', loss.item()))
        
        avg_loss = loss_train / training_sample_size
        avg_acc = acc_train.double() / training_sample_size
        #print('{} Loss: {:.4f} Acc: {:.4f}'.format('Epoch', epoch_loss, epoch_acc))
        
        model.eval()    
        # Iterate over data.
        loss_val = 0.0
        acc_val = 0
        for i, data in enumerate(validation_loader):
            inputs , labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs  = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss_val += loss.item() * inputs.size(0)
            acc_val += torch.sum(preds == labels.data)
            if (i+1) % 100 == 0:
                print('[{:d}/{:d}] {} loss: {:.4f}'.format(i, len(validation_loader), 'Test', loss.item()))

        avg_loss_val = loss_val / validation_sample_size
        avg_acc_val = acc_val.double() / validation_sample_size
        #print('{} Loss: {:.4f} Acc: {:.4f}'.format('Test', epoch_loss, epoch_acc))
        print()
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print("Label")
        print(labels)
        print("Pred")
        print(preds)
        print()    

        torch.save(model, 'xgc-vgg19.torch')
        print('Done.')
