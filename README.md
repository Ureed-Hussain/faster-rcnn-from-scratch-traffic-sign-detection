# Faster R-CNN from Scratch for Traffic Sign Detection

This repository presents a complete from-scratch implementation of Faster R-CNN for traffic sign detection on the German Traffic Sign Detection Benchmark (GTSDB) dataset. Unlike my previous repositories, which relied on TorchVision’s high-level Faster R-CNN abstractions, this project reimplements all major architectural components of Faster R-CNN explicitly, including the backbone, neck, Region Proposal Network (RPN), RoI feature extraction, and detection heads.

This work represents the final stage of a structured exploration of Faster R-CNN architectures for traffic sign detection:

1. Baseline Faster R-CNN using TorchVision (reference implementation)
2. Faster R-CNN with Custom Neck, isolating the effect of neck design

The earlier repositories serve as correctness and performance references, while this project prioritizes architectural transparency, learning, and implementation-level understanding.

## Project Motivation

Modern object detection frameworks often hide critical implementation details behind high-level APIs. While convenient, this abstraction can obscure how proposal generation, feature alignment, and multi-task loss optimization actually work together in two-stage detectors.

The goal of this repository is to:

* Fully understand Faster R-CNN internals by reimplementing them
* Gain fine-grained control over architectural and training choices
* Validate theoretical understanding through empirical results
* Build a modular, readable, and extensible detection framework
Performance is important, but clarity and correctness take precedence over pretrained convenience.

## Dataset Description, German Traffic Sign Detection Benchmark (GTSDB)

The GTSDB dataset is part of the IJCNN 2013 Traffic Sign Detection benchmark and contains real-world traffic scene images with annotated traffic signs. The dataset provides high-resolution images and standardized ground-truth annotations for evaluating object detection models.

###  Image Format

* Images are stored in PPM format with a resolution of 1360 × 800 pixels.

* Each image contains 0–6 traffic signs, appearing at sizes between 16×16 and 128×128 pixels.

* Signs may vary in perspective, lighting, and environment, making the dataset suitable for training robust detectors.

### Annotation Format

Annotations are provided in a semicolon-separated CSV file (gt.txt), where each entry contains:

* Filename

* Bounding box: x1; y1; x2; y2

* Class ID: integer representing the traffic sign category

#### Example fields:

image_xx.ppm; left; top; right; bottom; class_id

The dataset follows the class ID definitions described in the official ReadMe.txt file of the GTSDB package.


### Dataset Splits

The GTSDB dataset includes the following official splits (IJCNN 2013):

* FullIJCNN2013.zip → 900 total images

* TrainIJCNN2013.zip → 600 training images

* TestIJCNN2013.zip → 300 test images (no ground truth)

* gt.txt → ground-truth annotations for training and evaluation

In the project, after cleaning and filtering bounding boxes, I load:

* Total samples used: 506 images

* Train split: 404 images

* Validation split: 102 images

### Dataset Download

The dataset is downloaded automatically when running the data upload section in the notebooks. Users only need to upload their kaggle.json file, after which the code handles authentication, dataset download, and extraction.

### Data Visualization
I visualize samples from the dataset with bounding boxes and class IDs overlaid on the images.
Each numeric label corresponds to a traffic sign class defined in the GTSDB label specification.

<p align="center">
    <img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/7d012d07-54af-4780-8517-7cd8e31cc608" />
</p>

### Data Augmentation
To improve robustness, I apply lightweight augmentations using Albumentations, ensuring consistency between images and bounding boxes:

~~~python
train_aug = A.Compose([
    A.Rotate(limit=8, p=0.3),
    A.RandomBrightnessContrast(p=0.3),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
~~~

These augmentations help improve model robustness by introducing small rotations and brightness/contrast variations.

<p align="center">
    <img width="950" height="290" alt="image-1" src="https://github.com/user-attachments/assets/982e42bc-1f4d-426a-9130-194b3301dbd6" />
</p>


### Train/Validation Split
I split the dataset into 80% training and 20% validation using a reproducible train_test_split based on image indices. Both subsets share the same annotations, with the training set using augmentations and the validation set kept clean for unbiased evaluation.
~~~
Loaded 506 images with bounding boxes. (transforms=no)
Total samples: 506 | Train: 404 | Val: 102
Loaded 404 images with bounding boxes. (transforms=yes)
Loaded 102 images with bounding boxes. (transforms=no)
~~~

## Model Description

### Overall Architecture

This implementation follows the canonical Faster R-CNN design:

1. Backbone: Convolutional feature extractor
2. Neck: Multi-scale feature aggregation
3. Region Proposal Network (RPN): Generates candidate object regions
4. RoI Feature Extraction: Aligns proposal features
5. Detection Head: Classification and bounding box regression

All components are implemented explicitly and wired together manually, exposing intermediate tensors, losses, and training dynamics.
