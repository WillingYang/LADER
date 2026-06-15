# LADER

Code for multimodal fake-news detection with a `Qwen3-VL` backbone and LoRA fine-tuning.

This repository provides a simple training and evaluation pipeline for image-text fake-news classification. The code supports configurable dataset schemas, model paths, training settings, and classifier heads through YAML files.

## Datasets

This repository supports the multimodal fake-news detection datasets used in the GLPN-LLM setting:

- Weibo
- Twitter
- PHEME


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
