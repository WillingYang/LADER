#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from transformers import Trainer, TrainingArguments

from MoeDet.dataset import CollatorConfig, MultimodalNewsDataset, QwenVLClassificationCollator, build_model_rows
from MoeDet.data_spec import load_dataset_spec
from MoeDet.metrics import binary_metrics, save_metrics, save_predictions
from MoeDet.modeling import (
    ClassifierConfig,
    QwenVLForFakeNewsClassification,
    load_processor_and_backbone,
    maybe_apply_lora,
    resolve_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Qwen3-VL fake-news classifier.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default=None)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    split = args.split or str(config["data"].get("test_split", "test"))

    dataset_spec = load_dataset_spec(Path(config["paths"]["dataset_config"]))
    rows = build_model_rows(
        dataset_spec,
        split=split,
        max_samples=config["data"].get("max_test_samples"),
    )

    processor, backbone, _ = load_processor_and_backbone(
        model_path=Path(config["paths"]["model_path"]),
        torch_dtype=resolve_dtype(str(config["model"].get("torch_dtype", "bfloat16"))),
        attn_implementation=config["model"].get("attn_implementation"),
        trust_remote_code=bool(config["model"].get("trust_remote_code", True)),
    )
    backbone = maybe_apply_lora(
        backbone=backbone,
        enabled=bool(config.get("lora", {}).get("enabled", True)),
        r=int(config.get("lora", {}).get("r", 16)),
        alpha=int(config.get("lora", {}).get("alpha", 32)),
        dropout=float(config.get("lora", {}).get("dropout", 0.05)),
        target_modules=list(config.get("lora", {}).get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
    )
    model = QwenVLForFakeNewsClassification(
        backbone=backbone,
        config=ClassifierConfig(
            dropout=float(config["head"].get("dropout", 0.1)),
            num_labels=2,
            pooling=str(config["head"].get("pooling", "last_token")),
        ),
    )
    state_dict = torch.load(args.checkpoint_dir / "classifier_model.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    collator = QwenVLClassificationCollator(
        processor=processor,
        config=CollatorConfig(
            system_prompt=str(config["prompt"].get("system_prompt", "")),
            user_prompt_prefix=str(config["prompt"].get("user_prompt_prefix", "")),
            add_generation_prompt=bool(config["prompt"].get("add_generation_prompt", True)),
            max_length=config["prompt"].get("max_length"),
        ),
    )

    ta_signature = inspect.signature(TrainingArguments.__init__)
    supported = set(ta_signature.parameters.keys())
    eval_kwargs: Dict[str, Any] = {
        "output_dir": str(args.checkpoint_dir / "eval_tmp"),
        "per_device_eval_batch_size": int(config["training"].get("per_device_eval_batch_size", 1)),
        "dataloader_num_workers": int(config["training"].get("dataloader_num_workers", 0)),
        "remove_unused_columns": False,
        "report_to": [],
    }
    eval_kwargs = {key: value for key, value in eval_kwargs.items() if key in supported}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**eval_kwargs),
        data_collator=collator,
    )
    output = trainer.predict(MultimodalNewsDataset(rows))
    logits = np.asarray(output.predictions)
    labels = np.asarray(output.label_ids)
    preds = logits.argmax(axis=-1)
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)

    metrics = binary_metrics(labels, preds, probs)
    save_metrics(metrics, args.checkpoint_dir / f"{split}_metrics.json")
    prediction_rows = []
    for row, pred, score in zip(rows, preds.tolist(), probs[:, 1].tolist()):
        prediction_rows.append(
            {
                "sample_id": row["sample_id"],
                "gold_label": row["label"],
                "pred_label": int(pred),
                "fake_probability": float(score),
                "image_path": row["image_path"],
                "text": row["text"],
            }
        )
    save_predictions(prediction_rows, args.checkpoint_dir / f"{split}_predictions.csv")
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
