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

from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os

# global variables

# 20 classes and background for VOC segmentation
n_classes = 20 + 1
batch_size = 4
epochs = 50
lr = 1e-4
momentum = 0
w_decay = 1e-5
step_size = 50
gamma = 0.5
configs = "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print('Configs: ')
print(configs)

if sys.argv[1] == 'VOC':
    data_set = 'VOC'
else:
    data_set = 'Cityscpaes'

# create dir for model
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

if data_set == 'VOC':
    pass
else:
    pass