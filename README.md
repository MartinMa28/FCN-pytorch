

## 🚘 The easiest implementation of fully convolutional networks

- Task: __semantic segmentation__, it's a very important task for automated driving

- The model is based on CVPR '15 best paper honorable mentioned [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)

## Results
### Trials
<img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='result/trials.png' padding='5px' height="150px"></img>

### Training Procedures
<img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='result/result.gif' padding='5px' height="150px"></img>


## Performance

I train with two popular benchmark dataset: Pascal VOC 2012 and the augmented Pascal VOC segmentation

|dataset|n_class|mean IOU|
|---|---|---
|VOC|21|28%
|VOCAug|21|39%

## Training

### Install packages
```bash
conda install -r requirements.txt
```

download [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) 

download [Augmented Pascal VOC](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)

<!-- and download [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) dataset (recommended) or [Cityscapes](https://www.cityscapes-dataset.com/) dataset -->

### Run the code
- default dataset is Pascal VOC

create a directory named "VOC", and put data into it, then run python codes:  
Note that the pre-processing script may take for a while, because it's going to generate pixelwise labels for each image in the training set and the validation set. There doesn't have a decent progress bar to reveal how much it has done, since I hope to make it run in multi-threads. The year of images should be in the range from 2007 to 2012, so you could estimate the amount of work from the file name.

```python
# for Pascal VOC
python3 python/VOC_preprocessing.py 
python3 python/main.py VOC
```

```python
# For augmented Pascal VOC, it doesn't need any preprocessing in advance.
python3 python/main.py VOCAug
```

<!-- - or train with CityScapes

create a directory named "CityScapes", and put data into it, then run python codes:
```python
python3 python/CityScapes_utils.py 
python3 python/train.py CityScapes
``` -->

<!-- ## Author
Po-Chih Huang / [@pochih](https://pochih.github.io/) -->