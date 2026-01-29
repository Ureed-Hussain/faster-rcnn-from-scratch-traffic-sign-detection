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
