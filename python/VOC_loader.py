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
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        sample = (img, target)
        
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
        target = torch.from_numpy(target)
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