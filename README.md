# LADER

Code for multimodal fake-news detection with a `Qwen3-VL` backbone and LoRA fine-tuning.

This repository provides a simple training and evaluation pipeline for image-text fake-news classification. The code supports configurable dataset schemas, model paths, training settings, and classifier heads through YAML files.

## Datasets

This repository supports the multimodal fake-news detection datasets used in the GLPN-LLM setting:

- Weibo
- Twitter
- PHEME

Due to dataset redistribution restrictions, raw datasets are not included in this repository. Please obtain the datasets from their original sources and place them under your local data directory.

A typical data structure is:

```text
data/
├── weibo/
├── twitter/
└── pheme/
```

The internal label convention is:

```text
0 = real
1 = fake
```

Please check and update the label mapping in:

```text
configs/datasets/<dataset>.yaml
```


## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Recommended environment:

```text
Python >= 3.10
PyTorch with CUDA support
NVIDIA GPU(s)
```

Some configs may use `bf16` or `flash_attention_2`. If your environment does not support them, please modify the corresponding fields in the YAML config.

## Configuration

Before training, update the model path:

```yaml
paths:
  model_path: /path/to/Qwen3-VL-8B-Instruct
```

Update the dataset path:

```yaml
dataset:
  data_root: /path/to/your/dataset
```

For example:

```yaml
dataset:
  data_root: /data/weibo
```

The main configs are located in:

```text
configs/
configs/datasets/
```

## Training

Run from inside the repository:

```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash train.sh weibo
```

Multi-GPU training:

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash train.sh weibo
```

For other datasets:

```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash train.sh twitter
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash train.sh pheme
```

You can also run training directly with Python:

```bash
python train_classifier.py --config configs/weibo_qwen3vl_lora.yaml
```

## Evaluation

Evaluate a trained checkpoint:

```bash
cd MoeDet
CUDA_VISIBLE_DEVICES=0 bash eval.sh weibo outputs/moedet/weibo_qwen3vl_lora_baseline
```

For other datasets:

```bash
cd MoeDet
CUDA_VISIBLE_DEVICES=0 bash eval.sh twitter outputs/moedet/twitter_qwen3vl_lora_baseline
CUDA_VISIBLE_DEVICES=0 bash eval.sh pheme outputs/moedet/pheme_qwen3vl_lora_baseline
```

You can also run evaluation directly with Python:

```bash
python evaluate_classifier.py \
  --config configs/weibo_qwen3vl_lora.yaml \
  --checkpoint-dir outputs/moedet/weibo_qwen3vl_lora_baseline
```


## Citation

If you use this codebase, please cite our paper once the bibliographic information is available.

```bibtex
@misc{lader2026,
  title  = {LADER},
  author = {Anonymous},
  year   = {2026},
  note   = {Code release}
}
```
