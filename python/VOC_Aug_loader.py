import torch
import os
import scipy.io as spio
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class VOCSegAug(Dataset):
    def __init__(self, root_dir, data_set='train', transform=None):
        self.root_dir = os.path.expanduser(root_dir)
        self.data_set = data_set
        self.transform = transform
        
        base_dir = os.path.join(self.root_dir, 'dataset')
        img_dir = os.path.join(base_dir, 'img')
        target_dir = os.path.join(base_dir, 'cls')
        data_set_txt = self.data_set.strip() + '.txt'
        
        if not os.path.exists(data_set_txt):
            raise ValueError('Wrong data_set entered! Please use "train" or "val".')
        
        with open(data_set_txt, 'r') as f:
            img_name_list = [x.strip() for x in f.readlines()]
        
        self.img_path_list = [os.path.join(img_dir, x + '.jpg') for x in img_name_list]
        self.target_path_list = [os.path.join(target_dir, x + '.mat') for x in img_name_list]
        assert len(self.img_path_list) == len(self.target_path_list)
        
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):
        img = Image.open(self.img_path_list[index]).convert('RGB')
        target = spio.loadmat(self.target_path_list[index], \
                              mat_dtype=True, squeeze_me=True, struct_as_record=False)
        # extract the ground-true classification label from the mat dict
        target = target['GTcls'].Segmentation
        
        sample = (img, target)
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample