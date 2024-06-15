# Human-Centric-MLLM (HERM)
A Multimodal Large Language Model on human-centric tasks.

## Introduction
By tuning on self-created human-centric annotations, our model can excel in a wide range of human-centric vision-language tasks, greatly surpassing the existing MLLMs on human-centric understanding.

<div align=center>
<img src="./figs/examples.png" alt="overview" style="zoom: 80%">
</div>

## Installation
- Pre-requisites: Python 3.10, CUDA>=11.6 (We used 11.7)
- Install PyTorch 
```sh
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
```
- Install [Flash-attention](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features)
- Install the required packages:
```sh
pip install -r requirements.txt
```

## Dataset Preparations
TBD

## Training
We conduct a two-stage training: The first stage is pre-training on human-centric caption and grounding tasks, and the second stage is instruction tuning on free-style human-centric question-answering pairs.

- Stage 1: Pre-training
```sh
CUDA_VISIBLE_DEVICES=<your device numbers> torchrun \
  --master_port <your port> --nproc_per_node <your process numbers> \
  train.py --cfg-path train_configs/hcm_multitask/minigptv2_hcm_multitask.yaml
```
- Stage 2: Instruction tuning
```sh
CUDA_VISIBLE_DEVICES=<your device numbers> torchrun \
  --master_port <your port> --nproc_per_node <your process numbers> \
  train.py --cfg-path train_configs/hcm_multitask/minigptv2_hcm_instruct_tuning.yaml
```

## Inference

## Citation
