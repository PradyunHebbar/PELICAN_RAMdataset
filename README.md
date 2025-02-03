#Changes to PELICAN ATLASweights
  trainclassifier file --> lossfn = nn.crossentropy(reduction = "none")
  src.trainer.trainer --> load weights and multiple to lossfn in train function


# PELICAN Network for Particle Physics

    Permutation Equivariant, Lorentz Invariant/Covariant Aggregator Network for applications in Particle Physics.
    At the moment it includes two main variants: a classifier (for e.g. top-tagging) and a 4-momentum regression network. 

arXiv link: https://arxiv.org/abs/2211.00454

# PELICAN: Flexible Dataset Loading Fork

This repository is a fork of the original PELICAN code by Abogatskiy. It incorporates additional functionality to enhance dataset loading flexibility for distributed training on multi-GPU systems.

## Overview

In many large-scale training scenarios, loading entire datasets into RAM can quickly exhaust available system memory—especially when running multiple processes in parallel. This fork introduces a new command-line argument, `--ram_split`, which allows you to specify which dataset splits should be loaded into memory. For example, you can choose to load only the test and validation splits into RAM (using `--ram_split=test,valid`) while streaming the training data directly from disk. This flexibility can help optimize memory usage and performance.

## Key Features

### Configurable Dataset RAM Loading

Use the `--ram_split` argument to control which dataset splits are loaded entirely into memory. Options include:
- `"all"`: Load all splits into RAM
- `"none"`: Do not load any split into RAM
- A comma-separated list (e.g., `train,test`, `test,valid`): Load only the specified splits into RAM

### Distributed Training Support

Optimized for distributed training environments on nodes with multiple GPUs. Each process only loads the necessary data, helping to prevent out-of-memory (OOM) errors.

### Tribute to the Original PELICAN Code

This project is based on and inspired by the excellent work of Abogatskiy.

Alexander Bogatskiy, Flatiron Institute

Jan T. Offermann, University of Chicago 

Timothy Hoffman, University of Chicago

Xiaoyang Liu, University of Chicago

# Author of Repository

Pradyun Hebbar (pradyun.hebbar@gmail.com)