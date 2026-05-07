# GR4CIL: Gap-compensated Routing for Class-Incremental Learning

This repository provides a demo implementation of **GR4CIL**, proposed in the submitted paper:

**GR4CIL: Gap-compensated Routing for Class-Incremental Learning**

The current anonymous release is a lightweight demo version using **CIFAR-100** for class-incremental learning experiments.

## Overview

GR4CIL is designed for class-incremental learning. The method aims to improve incremental classification by combining task-specific adaptation, gap-compensated classification, and routing-based inference.

This demo code provides a simplified experimental pipeline for reproducing the main workflow on CIFAR-100.

## Environment Setup

Please install the required packages using:

```bash
pip install -r requirements.txt
```

## Prepare Pretrained Models

Please download the pretrained CLIP model before running the code.

First, create the directory:

```
mkdir -p pretrain_weights/clip_vit_base_patch16
```

Then download the pretrained model files from:

```
https://huggingface.co/openai/clip-vit-base-patch16/tree/main
```

Place all downloaded files into:

```
pretrain_weights/clip_vit_base_patch16/
```

## Dataset

This demo version uses **CIFAR-100**. Please run the code to prepare the dataset.

```
python data_generate.py
```

## Running the Demo

After installing the environment and preparing the pretrained CLIP weights, run:

```
python main.py
```

This command starts the CIFAR-100 class-incremental learning demo.

## Notes

This anonymous release focuses on the CIFAR-100 demo for illustrating and verifying the main GR4CIL pipeline. Additional scripts for all benchmark datasets will be provided in the de-anonymized final release.

## License and Usage Terms

This anonymous release is provided solely for the purpose of reviewing the submitted paper:

**GR4CIL: Gap-compensated Routing for Class-Incremental Learning**

Permission is granted to NeurIPS reviewers, area chairs, senior area chairs, and conference organizers to access, run, and locally modify this code only for the purpose of evaluating the submitted paper during the review process.

Redistribution, public posting, sublicensing, or commercial use of this anonymous review version is not permitted. A standard open-source license may be provided with the de-anonymized camera-ready release.

## Third-party Assets

- **CIFAR-100**: This demo uses CIFAR-100 for class-incremental learning experiments. Users should download and use the dataset from its original source and comply with the original dataset terms. Please cite the original CIFAR technical report when using this dataset.

- **CLIP ViT-B/16**: This demo uses the pretrained CLIP ViT-B/16 model. The pretrained model weights are not included in this repository. Users should download the model from the original OpenAI/Hugging Face source and comply with the corresponding license, model card, and terms of use.

- **Python packages and dependencies**: Third-party packages listed in `requirements.txt` are governed by their own licenses and terms. Users are responsible for complying with the licenses of all dependencies.
