from torchvision import datasets, transforms
from torchvision.transforms import functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

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
        
    @classmethod
    def get_color_map(cls, colors, classes):
        color_map = {}
        for ind, color in enumerate(colors):
            # makes colors hashable
            color = str(color[0]).zfill(3) + '/' + str(color[1]).zfill(3) + '/' + str(color[2]).zfill(3)
            color_map[color] = (ind, classes[ind])
        
        return color_map
        
    @classmethod
    def create_target_folder(cls, root):
        target_folder = os.path.join(root, 'VOCdevkit/VOC2012/SegmentationTargets/')
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        return target_folder

    @classmethod
    def preprocess_targets(cls, target_folder, target_img_path):
        img_name = os.path.basename(target_img_path).split('.')[0]
        print('processing ' + img_name)
        if os.path.exists(os.path.join(target_folder, img_name) + '.npy'):
            print('skip ' + img_name)
            return None
        
        # Original targets are palettized images, refering to colors by index (0 - 255).
        # Firstly, converts palettized images to RGB images.
        target = Image.open(target_img_path).convert('RGB')
        target = np.array(target)

        # Secondly, get the VOC classification indices by mapping RGB colors to colormap
        height, width, _ = target.shape
        index_target = np.zeros((height, width))
        color_map = cls.get_color_map(cls.voc_colors, cls.voc_classes)

        for h in range(height):
            for w in range(width):
                color = target[h, w]
                color = str(color[0]).zfill(3) + '/' + str(color[1]).zfill(3) + '/' + str(color[2]).zfill(3)
                ind = color_map.get(color, (0, 'background'))[0]
                index_target[h, w] = ind
        
        np.save(os.path.join(target_folder, img_name), index_target)

    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None,
                 target_transform=None):
        super().__init__(root, year, image_set, download, transform, target_transform)
        self.target_folder = os.path.join(self.root, 'VOCdevkit/VOC2012/SegmentationTargets')


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target_name = os.path.basename(self.masks[index]).split('.')[0]
        target_name = target_name + '.npy'
        
        target = np.load(os.path.join(self.target_folder, target_name))
    
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
        if img.shape[0] < 224 or img.shape[1] < 224:
            # if this image is smaller than 224 x 224, discards it by zeroing out
            # this image and its dense labels
            img = np.zeros((224, 224, 3))
            target = np.zeros((224, 224))

            return img, target
        
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
        if img.shape[0] < 224 or img.shape[1] < 224:
            # if this image is smaller than 224 x 224, discards it by zeroing out
            # this image and its dense labels
            img = np.zeros((224, 224, 3))
            target = np.zeros((224, 224))

            return img, target

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
        img = torch.from_numpy(img).float()
        #target = torch.from_numpy(target).to(torch.int64)
        target = torch.from_numpy(target).float()
        # if isinstance(img, torch.ByteTensor):
        #     img = img.to(dtype=torch.float32)
        
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