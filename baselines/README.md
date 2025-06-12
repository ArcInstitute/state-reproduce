# State-Sets Baselines Reproduction

This directory contains the implementation of the baselines from the State-Sets paper.

## Overview

This project implements and evaluates several baseline models for perturbation modeling:
- LowRankLinear (LRM)
- CPA (Compositional Perturbation Autoencoder)
- scGPT
- scVI

The models are evaluated on two major datasets:
- Replogle dataset (with 4-fold cross-validation)
- Tahoe dataset

## Installation

### 1. Pre-requisites

#### PyTorch and CUDA Setup
First, install the appropriate PyTorch version. This project was tested with:
- PyTorch 2.4.1
- CUDA 12.1

Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) to get the correct installation command for your system.

#### torch-scatter Installation
Install the compatible version of torch-scatter based on your PyTorch and CUDA versions. The package is already included in the requirements.txt.

### 1.1 (Optional) scGPT Requirements
To enable fine-tuning of the original pre-trained [scGPT](https://github.com/bowang-lab/scGPT) model, install flash-attention v1.0.9:
```bash
pip install flash-attn==1.0.9
```
In case of any issues, please refer to the [flash-attention](https://github.com/Dao-AILab/flash-attention) repository.

### 2. Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/Arcinstitute/state-sets-reproduce.git
cd state-sets-reproduce/baselines
```

2. Create and activate a virtual environment using uv:
```bash
uv venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Usage

### Training Baselines

The repository includes training scripts for each model-dataset combination. The scripts are located in the `scripts/` directory.

To train the baselines, you can use the `train.sh` script. The script takes three arguments:
- `<model>`: the model to train (e.g. `lrlm`, `cpa`, `scgpt`, `scvi`)
- `<dataset>`: the dataset to train on
- `<fold_id>`: the fold id to train on (only for Replogle)

For tahoe, the `<fold_id>` is not needed.

### Available Training Scripts

| Model | Dataset | Command Example |
|-------|---------|--------|
| LowRankLinear | Replogle | `sh scripts/train.sh lrlm replogle 1` |
| CPA | Replogle | `sh scripts/train.sh cpa replogle 1` |
| scGPT | Replogle | `sh scripts/train.sh scgpt replogle 1` |
| scVI | Replogle | `sh scripts/train.sh scvi replogle 1` |
| LowRankLinear | Tahoe | `sh scripts/train.sh lrlm tahoe` |
| CPA | Tahoe | `sh scripts/train.sh cpa tahoe` |
| scGPT | Tahoe | `sh scripts/train.sh scgpt tahoe` |
| scVI | Tahoe | `sh scripts/train.sh scvi tahoe` |
