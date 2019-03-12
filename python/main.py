from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from Cityscapes_loader import CityScapesDataset
from CamVid_loader import CamVidDataset
from VOC_loader import VOCSeg
from VOC_loader import RandomCrop, RandomHorizontalFlip, ToTensor, CenterCrop, NormalizeVOC
from torchvision import transforms
import copy

from matplotlib import pyplot as plt
import numpy as np
import time
import datetime
import sys
import os

import logging
from logging.config import fileConfig

# global variables
fileConfig('./logging_conf.ini')
logger = logging.getLogger('main')


# 20 classes and background for VOC segmentation
n_classes = 20 + 1
batch_size = 4
epochs = 1
lr = 1e-4
#momentum = 0
w_decay = 1e-5
step_size = 5
gamma = 0.5
configs = "FCNs-CrossEntropyLoss_batch{}_training_epochs{}_Adam_scheduler-step{}-gamma{}_lr{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, w_decay)
print('Configs: ')
print(configs)

if sys.argv[1] == 'VOC':
    data_set_type = 'VOC'
else:
    data_set_type = 'Cityscpaes'

# create dir for model
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)

score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)

use_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if use_gpu else 'cpu')
IU_scores    = np.zeros((epochs, n_classes))
pixel_scores = np.zeros(epochs)
# global variables

def get_dataset_dataloader(data_set_type, batch_size):
    if data_set_type == 'VOC':
        data_transforms = {
            'train': transforms.Compose([
                RandomCrop(224),
                RandomHorizontalFlip(),
                ToTensor(),
                NormalizeVOC([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),

            'val': transforms.Compose([
                CenterCrop(224),
                ToTensor(),
                NormalizeVOC([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
    else:
        data_transforms = {}

    data_set = {
        phase: VOCSeg('VOC/', '2012', image_set=phase, download=False,\
        transform=data_transforms[phase]) 
        for phase in ['train', 'val']
        }
    

    data_loader = {
        'train': DataLoader(data_set['train'], batch_size=batch_size, shuffle=True),
        'val': DataLoader(data_set['val'], batch_size=batch_size, shuffle=False)
    }

    return data_set, data_loader

def get_fcn_model(num_classes, use_gpu):
    vgg_model = VGGNet(requires_grad=True, remove_fc=True)
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=num_classes)

    if use_gpu:
        ts = time.time()
        vgg_model = vgg_model.cuda()
        fcn_model = fcn_model.cuda()
        num_gpu = list(range(torch.cuda.device_count()))
        fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
        
        print("Finish cuda loading, time elapsed {}".format(time.time() - ts))
    
    return fcn_model


def time_stamp() -> str:
    ts = time.time()
    time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return time_stamp

# Borrows and modifies iou() from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions    
def iou(pred, target, n_classes):
    ious = np.zeros(n_classes)
    for cl in range(n_classes):
        pred_inds = (pred == cl)
        target_inds = (target == cl)
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            # if there is no ground truth, do not include in evaluation
            ious[cl] = float('nan')  
        else:
            ious[cl] = float(intersection) / max(union, 1)

    return ious.reshape((1, n_classes)), np.nanmean(ious)

def pixelwise_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total


def train(data_set_type, num_classes, batch_size, epochs, use_gpu, learning_rate, w_decay):
    fcn_model = get_fcn_model(num_classes, use_gpu)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=fcn_model.parameters(), lr=learning_rate, weight_decay=w_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 5 epochs

    data_set, data_loader = get_dataset_dataloader(data_set_type, batch_size)

    since = time.time()
    best_model_wts = copy.deepcopy(fcn_model.state_dict())
    best_acc = 0.0

    epoch_loss = np.zeros(epochs)
    epoch_acc = np.zeros(epochs)
    epoch_iou = np.zeros((epochs, num_classes))
    epoch_mean_iou = np.zeros(epochs)

    for epoch in range(epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, epochs))
        logger.info('-' * 28)
        
        
        for phase in ['val', 'train']:
            if phase == 'train':
                fcn_model.train()
                logger.info(phase)
            else:
                fcn_model.eval()
                logger.info(phase)
        
            running_loss = 0.0
            running_acc = 0.0
            running_mean_iou = 0.0
            running_iou = np.zeros((1, num_classes))
            
            batch_counter = 0
            for imgs, targets in data_loader[phase]:
                batch_counter += 1
                logger.debug('Batch {}'.format(batch_counter))
                imgs = Variable(imgs).float()
                imgs = imgs.to(device)
                targets = Variable(targets).type(torch.LongTensor)
                targets = targets.to(device)

                # zero the learnable parameters gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = fcn_model(imgs)
                    loss = criterion(outputs, targets)
                    preds = torch.argmax(outputs, dim=1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # computes loss and acc for current iteration
                ious, mean_ious = iou(preds, targets, n_classes)
                
                running_loss += loss * imgs.size(0)
                running_acc += pixelwise_acc(preds, targets) * imgs.size(0)
                running_iou += ious * imgs.size(0)
                running_mean_iou += mean_ious * imgs.size(0)
                logger.debug('Batch {} running loss: {}'.format(batch_counter, running_loss))
            
            epoch_loss[epoch] = running_loss / len(data_set[phase])
            epoch_acc[epoch] = running_acc / len(data_set[phase])
            epoch_iou[epoch] = running_iou / len(data_set[phase])
            epoch_mean_iou[epoch] = running_mean_iou / len(data_set[phase])

            
            logger.info('{} loss: {:.4f}, acc: {:.4f}, mean iou: {}'.format(phase,\
                epoch_loss[epoch], epoch_acc[epoch], epoch_mean_iou[epoch]))

            if phase == 'val' and epoch_acc[epoch] > best_acc:
                best_acc = epoch_acc[epoch]
                best_model_wts = copy.deepcopy(fcn_model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    logger.info('Training completed in {:0.f}m {:0.f}s'.format(int(time_elapsed / 60),\
        time_elapsed % 60))
    
    # load best model weights
    model.load_state_dict(best_model_wts)

    # save numpy results
    np.save(os.path.join(score_dir, 'epoch_accuracy'), epoch_acc)
    np.save(os.path.join(score_dir, 'epoch_mean_iou'), epoch_mean_iou)
    np.save(os.path.join(score_dir, 'epoch_iou'), epoch_iou)

    return model

if __name__ == "__main__":
    fcn_model = train(data_set_type, n_classes, batch_size, epochs, use_gpu, lr, w_decay)
    torch.save(fcn_model.state_dict(), os.path.join(score_dir, 'trained_model'))




