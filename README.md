# LADER

Lightweight training and evaluation code for multimodal fake-news detection with a `Qwen3-VL` backbone, LoRA tuning, and configurable classification heads.

This repository is designed to be simple to read, easy to adapt, and straightforward to reproduce on new datasets.

## Highlights

- `Qwen3-VL`-based multimodal binary classification
- LoRA-based efficient fine-tuning
- Config-driven dataset schema mapping
- Multiple classifier-head variants through YAML configs
- Hugging Face `Trainer` training/evaluation pipeline

## Repository Structure

```text
MoeDet/
├── configs/
│   ├── <dataset>_qwen3vl_lora.yaml
│   └── datasets/
│       └── <dataset>.yaml
├── datasets/
├── data_spec.py
├── dataset.py
├── modeling.py
├── metrics.py
├── train_classifier.py
├── evaluate_classifier.py
├── train.sh
├── eval.sh
└── README.md
```

## Environment

### Python dependencies

Install the basic dependencies with:

```bash
pip install -r requirements.txt
```

### Recommended runtime

- Python `3.10+`
- PyTorch with CUDA support
- NVIDIA GPU(s)

### Optional acceleration

Some configs use:

- `bf16`
- `flash_attention_2`

If your environment does not support them, update the corresponding config fields before training.

## Data Preparation

This codebase is schema-driven. Each dataset has:

1. an **experiment config** under `configs/`
2. a **dataset schema config** under `configs/datasets/`

Before running, you usually need to change:

- `paths.model_path` in `configs/<dataset>_qwen3vl_lora.yaml`
- `dataset.data_root` in `configs/datasets/<dataset>.yaml`

See:

- `configs/README.md`
- `datasets/README.md`

## Label Convention

Internally, the classifier always uses:

- `0 = real`
- `1 = fake`

Raw dataset labels are mapped to this convention using `output_label_map` in each dataset YAML.

## Quick Start

### Option A: run from the parent directory

Train:

```bash
python MoeDet/train_classifier.py --config MoeDet/configs/weibo_qwen3vl_lora.yaml
```

Evaluate:

```bash
python MoeDet/evaluate_classifier.py \
  --config MoeDet/configs/weibo_qwen3vl_lora.yaml \
  --checkpoint-dir outputs/moedet/weibo_qwen3vl_lora_baseline
```

### Option B: run from inside `MoeDet/`

Train:

```bash
cd MoeDet
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash train.sh weibo
```

Multi-GPU:

```bash
cd MoeDet
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash train.sh weibo
```

Evaluate:

```bash
cd MoeDet
CUDA_VISIBLE_DEVICES=0 bash eval.sh weibo outputs/moedet/weibo_qwen3vl_lora_baseline
```

## Training Outputs

After training, the output directory typically contains:

- `classifier_model.pt`
- `processor/`
- `used_config.yaml`
- `eval_metrics.json`
- `test_metrics.json`
- `test_predictions.csv`
- `summary.json`

The exact output directory is controlled by:

- `paths.output_dir` in the experiment config

## Config Tips

### Model path

Set:

```yaml
paths:
  model_path: /path/to/Qwen3-VL-8B-Instruct
```

### Dataset path

Set:

```yaml
dataset:
  data_root: /path/to/your/dataset
```

### WandB logging

Enable or disable in:

```yaml
wandb:
  enabled: true
```

## Reproducing a Run

For a reproducible experiment:

1. copy an existing config under `configs/`
2. update dataset/model paths
3. set the desired `wandb.run_name`
4. train with `train.sh` or `train_classifier.py`
5. evaluate with `eval.sh` or `evaluate_classifier.py`
6. archive the generated `used_config.yaml`

Because `used_config.yaml` is saved after training, it is the easiest way to track the exact config used for a released checkpoint.

## Common Issues

### 1. Model path not found

Check:

- `paths.model_path`
- whether the local model weights are available

### 2. Dataset file not found

Check:

- `dataset.data_root`
- split filenames under `dataset.splits`
- `image_root`

### 3. CUDA / flash attention mismatch

If your runtime does not support the configured attention backend, update:

```yaml
model:
  attn_implementation: none
```

### 4. DDP unused parameter errors

If you disable parts of a head via config switches in future variants, set:

```yaml
training:
  ddp_find_unused_parameters: true
```

## Engineering Notes for Open-Sourcing

This repository intentionally keeps the core pipeline compact:

- dataset logic is separated into `data_spec.py` and `dataset.py`
- model/head logic is centralized in `modeling.py`
- training and evaluation entrypoints are explicit and easy to trace

For public release, we recommend:

- removing private absolute paths from configs before publishing
- keeping only small example metadata in `datasets/`
- excluding checkpoints and outputs with `.gitignore`
- documenting exact backbone and dataset sources in your paper/project page

## Citation

If you use this codebase in your work, please cite the corresponding paper once the bibliographic information is available.

```bibtex
@misc{moedet,
  title        = {LADER},
  author       = {Anonymous},
  year         = {2026},
  note         = {Code release}
}
```

