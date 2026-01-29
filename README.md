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


<p align="center">
    <img width="500" height="600" alt="Strach_R-CNN" src="https://github.com/user-attachments/assets/88cfbf47-0095-4cdd-b4b7-0605b34c696d" />
</p>

### 1. Backbone Network

The backbone is a custom wrapper around ResNet50, initialized with ImageNet-pretrained weights. Instead of producing a single feature map, the backbone explicitly exposes intermediate feature maps from stages C2–C5, corresponding to strides 4, 8, 16, and 32. These multi-scale feature maps serve as the foundation for pyramid construction in the subsequent neck module and allow explicit control over feature resolution and channel dimensions.

~~~python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.ops import boxes as box_ops

# SECTION 1: BACKBONE (ResNet50 Wrapper)

class ResNetBackbone(nn.Module):
    """
    Wrap ResNet50 to output multi-scale feature maps (C2-C5)
    """
    def __init__(self, pretrained=True):
        super().__init__()

        resnet = torchvision.models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1  # C2 (stride 4)
        self.layer2 = resnet.layer2  # C3 (stride 8)
        self.layer3 = resnet.layer3  # C4 (stride 16)
        self.layer4 = resnet.layer4  # C5 (stride 32)

        self.out_channels = {
            'c2': 256,
            'c3': 512,
            'c4': 1024,
            'c5': 2048
        }

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return {
            'c2': c2,
            'c3': c3,
            'c4': c4,
            'c5': c5
        }

print(" Section 1: Custom Backbone defined")
~~~

### 2. Neck (Feature Aggregation)
The neck is a custom-built Feature Pyramid Network that performs top-down feature propagation with lateral connections. Feature maps from the backbone (C2–C5) are projected into a unified channel space using 1×1 lateral convolutions, merged via nearest-neighbor upsampling, and refined with 3×3 smoothing convolutions. An additional P6 level is generated via strided convolution for improved large-scale proposal coverage. This explicit FPN implementation replaces high-level abstractions and exposes all pyramid construction details.

~~~python
# SECTION 2: CUSTOM FPN (NECK)

class CustomFPN(nn.Module):
    """
    Custom Feature Pyramid Network
    """
    def __init__(self, in_channels_list=[256, 512, 1024, 2048], out_channels=256):
        super().__init__()

        self.out_channels = out_channels

        # Lateral convs
        self.lateral_c5 = nn.Conv2d(in_channels_list[3], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        self.lateral_c3 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral_c2 = nn.Conv2d(in_channels_list[0], out_channels, 1)

        # Smoothing convs
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # P6 for RPN
        self.p6_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        c2, c3, c4, c5 = features['c2'], features['c3'], features['c4'], features['c5']

        # Top-down pathway
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')

        # Smoothing
        p5 = self.smooth_p5(p5)
        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)
        p2 = self.smooth_p2(p2)

        # P6
        p6 = self.p6_conv(p5)

        return {
            'p2': p2,
            'p3': p3,
            'p4': p4,
            'p5': p5,
            'p6': p6
        }

print(" Section 2: Custom FPN defined")
~~~

### 3. Region Proposal Network (RPN)

The Region Proposal Network is implemented explicitly using a shared convolutional head followed by parallel objectness and bounding box regression layers. Operating over all FPN levels (P2–P6), the RPN generates anchor-aligned predictions, decodes bounding box deltas, applies clipping, filtering, and non-maximum suppression, and produces a final set of region proposals. Anchor assignment, sampling, IoU-based labeling, and RPN loss computation are all handled manually for full transparency.

~~~python
# SECTION 3: CUSTOM RPN HEAD

class CustomRPNHead(nn.Module):
    """
    Region Proposal Network Head
    """
    def __init__(self, in_channels=256, num_anchors=3):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.objectness = nn.Conv2d(in_channels, num_anchors * 1, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)

        for layer in [self.conv, self.objectness, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        objectness = []
        bbox_reg = []

        for feature in features:
            t = F.relu(self.conv(feature))
            objectness.append(self.objectness(t))
            bbox_reg.append(self.bbox_pred(t))

        return objectness, bbox_reg



print(" Section 3: Custom RPN Head defined")
~~~

### 4. RoI Feature Extraction and Detection Head
Region-wise features are extracted using MultiScale RoIAlign over FPN levels P2–P5, producing fixed-size feature tensors for each proposal. These features are processed by fully connected layers with dropout, followed by separate classification and bounding box regression heads. Class-specific bounding box refinement, proposal matching, loss computation, and inference-time post-processing (score thresholding and NMS) are implemented explicitly, completing the end-to-end Faster R-CNN detection pipeline.

~~~python
# SECTION 4: CUSTOM ROI HEADS

class CustomROIHeads(nn.Module):
    """
    ROI Heads for final detection
    """
    def __init__(self, in_channels=256, num_classes=44, representation_size=1024):
        super().__init__()

        from torchvision.ops import MultiScaleRoIAlign
        self.roi_align = MultiScaleRoIAlign(
            featmap_names=['p2', 'p3', 'p4', 'p5'],
            output_size=7,
            sampling_ratio=2
        )

        self.fc1 = nn.Linear(in_channels * 7 * 7, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.cls_score = nn.Linear(representation_size, num_classes)
        self.bbox_pred = nn.Linear(representation_size, num_classes * 4)

        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.cls_score, self.bbox_pred]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, features, proposals, image_shapes):
        box_features = self.roi_align(features, proposals, image_shapes)
        box_features = box_features.flatten(start_dim=1)

        x = F.relu(self.fc1(box_features))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        class_logits = self.cls_score(x)
        box_regression = self.bbox_pred(x)

        return class_logits, box_regression


print(" Section 4: Custom ROI Heads defined")
~~~
## Training Setup and Model Configuration

The from-scratch Faster R-CNN model is constructed and trained using a fully custom training pipeline with explicit control over all architectural and optimization components. Training is executed on GPU when available, with automatic fallback to CPU to ensure portability.

The model is instantiated with a ResNet50 backbone initialized with ImageNet-pretrained weights, while all other components—including the Feature Pyramid Network (FPN), Region Proposal Network (RPN), and RoI heads—are trained from scratch. The detector is configured to predict 44 classes, corresponding to the 43 traffic sign categories in GTSDB plus a background class. The RoI head uses a 1024-dimensional feature representation to balance expressive capacity and training stability.

Mixed Precision Training is enabled using automatic mixed precision (AMP) with gradient scaling, reducing memory usage and improving training throughput while maintaining numerical stability.

Overall, this training setup ensures that model behavior, convergence dynamics, and loss contributions from each Faster R-CNN component can be examined in detail, aligning with the project’s goal of architectural transparency and full pipeline understanding.


<p align="center">
    <img width="800" height="400" alt="Custom_R-cnn_curve" src="https://github.com/user-attachments/assets/039aa22f-052d-45ed-a7c5-7e9a96947710" />
</p>

## Testing and Evaluation

Model evaluation is performed using a fully custom testing pipeline designed to go beyond default TorchVision inference and provide fine-grained control over detection behavior. Testing is conducted on the validation split using carefully tuned confidence thresholds and multiple non-maximum suppression (NMS) stages to balance recall and precision in a dense, multi-class traffic sign environment.

During inference, predictions are filtered using class-agnostic NMS and score-based pruning, followed by IoU-based matching between predicted boxes and ground-truth annotations. True positives, false positives, and false negatives are computed explicitly at an IoU threshold of 0.5, enabling precise calculation of Precision, Recall, F1-score, and mAP (11-point interpolation). This explicit matching strategy ensures metric transparency and avoids reliance on black-box evaluation utilities.

In addition to quantitative metrics, the evaluation pipeline includes visual inspection with ground-truth-aligned labeling, where matched predictions are shown in green and false positives in red. This qualitative analysis helps validate spatial localization quality, error modes, and class confusion patterns—crucial for understanding model behavior in real-world traffic scenes

| Metric | Value |
|------|------|
| Precision | 86.79% |
| Recall | 86.25% |
| mAP | 74.15% |

### Qualitative Result

The following visualization shows predicted bounding boxes closely matching the ground truth, demonstrating strong localization and classification performance.

<p align="center">
    <img width="800" height="1000" alt="Custom_r-cnn_results" src="https://github.com/user-attachments/assets/cbf6b2fa-4114-4d34-bdf0-1b1e4fc64f99" />
</p>



