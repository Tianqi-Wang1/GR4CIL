# GR4CIL: Gap-compensated Routing for CLIP-based Class Incremental Learning

This repository provides a demo implementation of **GR4CIL**, proposed in the submitted paper:

**GR4CIL: Gap-compensated Routing for CLIP-based Class Incremental Learning**

The current anonymous release is a lightweight demo version using **CIFAR-100** for class-incremental learning experiments.

## Overview

GR4CIL is designed for CLIP-based class-incremental learning. The method aims to improve incremental classification by combining task-specific adaptation, gap-compensated classification, and routing-based inference.

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

This anonymous repository is provided for review purposes. The released code is a demo version intended to illustrate the main training and evaluation pipeline of GR4CIL.

More complete code, additional datasets, and full experimental scripts will be released in the final version.