
# Distributed Data Parallel (DDP) in PyTorch

This repository contains a series of tutorials and code examples for implementing Distributed Data Parallel (DDP) training in PyTorch. The aim is to provide a thorough understanding of how to set up and run distributed training jobs on single and multi-GPU setups, as well as across multiple nodes. Additionally, we cover fault-tolerant training and training complex models like GPT using DDP.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Tutorials](#tutorials)
  - [Single GPU Training](#single-gpu-training)
  - [Multi-GPU Training on a Single Node](#multi-gpu-training-on-a-single-node)
  - [Fault-Tolerant Training with `torchrun`](#fault-tolerant-training-with-torchrun)
  - [Multi-Node Training](#multi-node-training)
  - [Training GPT Model with DDP](#training-gpt-model-with-ddp)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Running on Single Node](#running-on-single-node)
  - [Running on Multiple Nodes](#running-on-multiple-nodes)

## Introduction

Welcome to the Distributed Data Parallel (DDP) in PyTorch tutorial series. This repository provides code examples and explanations on how to implement DDP in PyTorch for efficient model training. We will start with simple examples and gradually move to more complex setups, including multi-node training and training a GPT model.

## Getting Started

### Prerequisites

- Python 3.8 or later
- PyTorch 1.9.0 or later
- CUDA 10.2 or later (if using GPUs)
- AWS CLI (for multi-node setup with AWS)
- Slurm (for multi-node setup with Slurm)

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/jhuboo/ddp-pytorch-tutorial.git
cd ddp-pytorch-tutorial
pip install -r requirements.txt
```

## Tutorials

### Single GPU Training

We start with a basic example of training a model on a single GPU. The code is provided in the `single_gpu` directory.

### Multi-GPU Training on a Single Node

Next, we demonstrate how to scale the training to multiple GPUs on a single machine using DDP. The code is in the `multi_gpu_single_node` directory.

### Fault-Tolerant Training with `torchrun`

We introduce fault-tolerant training using `torchrun`, which helps in resuming training from the last saved snapshot in case of failures. The code is in the `fault_tolerant_training` directory.

### Multi-Node Training

We cover how to distribute training across multiple machines. The setup includes using both `torchrun` and Slurm workload manager. The code is in the `multi_node_training` directory.

### Training GPT Model with DDP

Finally, we implement DDP for training a GPT model using Andrej Karpathy’s minGPT repository. The code is in the `gpt_training` directory.

## Project Structure

```plaintext
ddp-pytorch-tutorial/
├── single_gpu.py
├── multigpu.py
├── multigpu_torchrun.py
├── multinode.py
├── datautils.py
├── requirements.txt
└── README.md
```

## Usage

### Running on Single Node

For single GPU training:

```bash
python single_gpu.py
```

For multi-GPU training on a single node:

```bash
torchrun --standalone --nproc_per_node=4 multigpu.py
```

### Running on Multiple Nodes

Using `torchrun` for multi-node training:

```bash
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=node0:29500 multigpu_torchrun.py
```

Using Slurm for multi-node training:

1. Edit the `slurm_scripts` with appropriate settings.
2. Submit the Slurm job:


### Training GPT Model with DDP

```bash
cd minGPT-ddp/mingpt
torchrun --standalone --nproc_per_node=4 main.py
```
