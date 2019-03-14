from VOC_loader import VOCSeg
from concurrent.futures import ThreadPoolExecutor
import os

if __name__ == "__main__":
    
    for phase in ['train', 'val']:
        print('processing {} data'.format(phase))
        target_folder = VOCSeg.create_target_folder('VOC/')
        
        def _preprocess_target(target_img_path):
            VOCSeg.preprocess_targets(target_folder, target_img_path)

        target_list_path = 'VOC/VOCdevkit/VOC2012/ImageSets/Segmentation/' + phase + '.txt'
        mask_dir = 'VOC/VOCdevkit/VOC2012/SegmentationClass/'
        with open(target_list_path, "r") as f:
            target_list = [mask_dir + x.strip() + '.png' for x in f.readlines()]

        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(_preprocess_target, target_list)
