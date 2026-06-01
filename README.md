# MoeDet baseline

This directory contains a classification-style baseline for multimodal fake-news detection:

- backbone: `Qwen3-VL-8B(-Instruct)`
- representation: last hidden layer, last valid token
- head: binary classifier
- tuning: LoRA
- trainer: Hugging Face `Trainer`
- logging: `wandb`

## Label convention

- internal label id `0` = `real`
- internal label id `1` = `fake`

Raw dataset labels are remapped through each dataset's `output_label_map`, so datasets like Fakeddit can keep their original `0=fake, 1=real` storage while the classifier still trains on a unified internal convention.

## Train

```bash
python MoeDet/train_classifier.py --config MoeDet/configs/weibo_qwen3vl_lora.yaml
```

Or from inside `MoeDet/`:

```bash
cd MoeDet
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash train.sh weibo
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash train.sh weibo
```

## Evaluate

```bash
python MoeDet/evaluate_classifier.py \
  --config MoeDet/configs/weibo_qwen3vl_lora.yaml \
  --checkpoint-dir outputs/moedet/weibo_qwen3vl_lora_baseline
```

Or from inside `MoeDet/`:

```bash
cd MoeDet
CUDA_VISIBLE_DEVICES=0 bash eval.sh weibo outputs/moedet/weibo_qwen3vl_lora_baseline
```

## Notes

- Dataset configs now live inside `MoeDet/configs/datasets/`, so `MoeDet` can run independently.
- The first baseline keeps a single classifier head so we can later replace it with an MoE head without changing the rest of the training stack.
