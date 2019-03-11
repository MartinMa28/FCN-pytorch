from torchvision import datasets, transforms
from torchvision.transforms import functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class VOCSeg(datasets.VOCSegmentation):
    """
    Overrides VOCSegmentation's __getitem__() to make identitical transformations
    for images and their dense labels.
    
    For example, when doing random crop, we hope to deploy the exact same random crop
    for both the image and its dense labels.
    """
    
    # class variables for VOC color maps
    voc_colors = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
    voc_classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
    
    def get_color_map(self, colors, classes):
        color_map = {}
        for ind, color in enumerate(colors):
            # makes colors hashable
            color = str(color[0]).zfill(3) + '/' + str(color[1]).zfill(3) + '/' + str(color[2]).zfill(3)
            color_map[color] = (ind, classes[ind])
        
        return color_map
        
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        # Original targets are palettized images, refering to colors by index (0 - 255).
        # Firstly, converts palettized images to RGB images.
        target = Image.open(self.masks[index]).convert('RGB')
        target = np.array(target)
        
        # Secondly, get the VOC classification indices by mapping RGB colors to colormap
        height, width, _ = target.shape
        index_target = np.zeros((height, width))
        color_map = self.get_color_map(self.voc_colors, self.voc_classes)
        for h in range(height):
            for w in range(width):
                color = target[h, w]
                color = str(color[0]).zfill(3) + '/' + str(color[1]).zfill(3) + '/' + str(color[2]).zfill(3)
                ind = color_map.get(color, (0, 'background'))[0]
                index_target[h, w] = ind
    
        sample = (img, index_target)
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample


# callable classes to transform both images and their dense labels (target)
class Rescale():
    """
    Rescale the image in a sample to a given size.
    Args: output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, sample):
        # sample: (img, target)
        img, target = sample
        assert img.size == target.size
        h, w = img.size
        
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        
        img = Image.resize((new_h, new_w))
        target = Image.resize((new_h, new_w))
        
        return img, target

class RandomCrop():
    """
    Crop randomly the image in a sample.
    Note that PIL Image instances are casted to numpy ndarrays in this step.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    
    def __call__(self, sample):
        img, target = sample
        # convert PIL.Image to np.ndarray
        img = np.asarray(img)
        target = np.asarray(target)
        assert img.shape[:2] == target.shape[:2]
        h, w = img.shape[:2]
        
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        img = img[top: top + new_h, left: left + new_w]
        target = target[top: top + new_h, left: left + new_w]
        
        return img, target


class CenterCrop():
    """
    Crops the center of the image and its dense labels in a sample.
    Note that PIL Image instances are casted to numpy ndarrays in this step.
    
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    
    def __call__(self, sample):
        img, target = sample
        # convert PIL Image to numpy.ndarray
        img = np.asarray(img)
        target = np.asarray(target)
        assert img.shape[:2] == target.shape[:2]
        h, w = img.shape[:2]
        
        new_h, new_w = self.output_size
        
        top = int((h - new_h) / 2)
        left = int((w - new_w) / 2)
        
        img = img[top: top + new_h, left: left + new_w]
        target = target[top: top + new_h, left: left + new_w]
        
        return img, target

    
class RandomHorizontalFlip():
    """
    Randomly flips the image and its dense labels in the horizontal direction.
    Args:
        the rate of flips: 0 ~ 1
    """
    def __init__(self, flip_rate=0.5):
        assert isinstance(flip_rate, (int, float))
        self.flip_rate = flip_rate
    
    def __call__(self, sample):
        img, target = sample
        if np.random.random() < self.flip_rate:
            img = np.fliplr(img).copy()
            target = np.fliplr(target).copy()
        
        return img, target

    
class ToTensor():
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        img, target = sample
        
        # swap color axis
        # numpy image: H x W x C
        # torch Tensor image: C x H x W
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        target = torch.from_numpy(target).to(torch.int64)
        if isinstance(img, torch.ByteTensor):
            img = img.to(dtype=torch.float32)
        
        return img, target

    
class NormalizeVOC():
    """
    Normalizes the input images, because the models were pretrained with 
    the hard-coded normalization values.
    """
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
    
    def __call__(self, sample):
        """
        Args:
            sample (Tensor, Tensor): Tensor image of size (C, H, W) to be normalized.
            Tensor image's dense label tensor will not be normalized.

        Returns:
            (Tensor, Tensor): Normalized Tensor image and its dense labels.
        """
        img, target = sample
        return F.normalize(img, self.mean, self.std, self.inplace), target