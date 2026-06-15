# Configuration Guide

This directory contains two types of YAML files:

- `configs/<dataset>_qwen3vl_lora.yaml`: experiment-level training configs
- `configs/datasets/<dataset>.yaml`: dataset schema and path configs

## 1. Experiment config

Example: `configs/weibo_qwen3vl_lora.yaml`

Main sections:

- `paths`
  - `dataset_config`: points to a dataset YAML under `configs/datasets/`
  - `model_path`: local path or Hugging Face path to the Qwen3-VL checkpoint
  - `output_dir`: where training artifacts are saved
- `data`
  - train/eval/test split names
  - optional sample caps for quick debugging
- `model`
  - dtype, attention backend, gradient checkpointing
- `lora`
  - LoRA rank, alpha, dropout, target modules
- `head`
  - classifier head mode and hyperparameters
- `training`
  - epochs, batch size, accumulation, learning rate, etc.
- `evaluation`
  - save/eval strategy and best-model metric
- `wandb`
  - optional Weights & Biases logging

## 2. Dataset config

Example: `configs/datasets/weibo.yaml`

These files define:

- CSV/TSV/JSON split file locations
- image root location
- text/id/label column names
- raw-label to canonical-label mapping

## Before running

Please update at least the following fields:

1. `paths.model_path`
2. `paths.dataset_config`
3. `dataset.data_root` inside the dataset YAML

Most configs in this repo use internal absolute paths as placeholders, so they need to be changed for a new machine.

