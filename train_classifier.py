#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from MoeDet.dataset import (
    CollatorConfig,
    MultimodalNewsDataset,
    QwenVLClassificationCollator,
    build_model_rows,
    stratified_train_val_split,
)
from MoeDet.data_spec import load_dataset_spec
from MoeDet.metrics import compute_metrics_from_eval_pred, save_metrics, save_predictions
from MoeDet.modeling import (
    ClassifierConfig,
    QwenVLForFakeNewsClassification,
    infer_image_token_id,
    load_processor_and_backbone,
    maybe_apply_lora,
    resolve_dtype,
)


def using_test_as_eval(config: Dict[str, Any]) -> bool:
    data_cfg = config["data"]
    eval_split = data_cfg.get("eval_split")
    test_split = data_cfg.get("test_split")
    return bool(eval_split) and bool(test_split) and str(eval_split) == str(test_split)


class SplitAwareTrainer(Trainer):
    def __init__(self, *args: Any, eval_metric_prefix: str = "eval", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.eval_metric_prefix = eval_metric_prefix

    def evaluate(self, *args: Any, metric_key_prefix: Optional[str] = None, **kwargs: Any) -> Dict[str, float]:
        if metric_key_prefix is None:
            metric_key_prefix = self.eval_metric_prefix
        return super().evaluate(*args, metric_key_prefix=metric_key_prefix, **kwargs)

    def _determine_best_metric(self, metrics: Dict[str, float], trial: Any) -> bool:
        if self.args.metric_for_best_model is None:
            return False

        metric_to_check = self.args.metric_for_best_model
        try:
            metric_value = metrics[metric_to_check]
        except KeyError as exc:
            available = sorted(metrics.keys())
            raise KeyError(
                f"The `metric_for_best_model` training argument is set to '{metric_to_check}', "
                f"which is not found in the evaluation metrics. The available evaluation metrics are: {available}. "
                "Consider changing the `metric_for_best_model` via the TrainingArguments."
            ) from exc

        operator = np.greater if self.args.greater_is_better else np.less
        if self.state.best_metric is None or operator(metric_value, self.state.best_metric):
            self.state.best_metric = metric_value
            self.state.best_model_checkpoint = os.path.join(
                self._get_output_dir(trial=trial),
                f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}",
            )
            return True
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Qwen3-VL fake-news classifier with LoRA.")
    parser.add_argument("--config", type=Path, required=True, help="Training config YAML.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_rows(
    dataset_cfg_path: Path,
    train_split: str,
    eval_split: Optional[str],
    test_split: Optional[str],
    max_train_samples: Optional[int],
    max_eval_samples: Optional[int],
    max_test_samples: Optional[int],
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    dataset_spec = load_dataset_spec(dataset_cfg_path)
    train_rows = build_model_rows(dataset_spec, split=train_split, max_samples=max_train_samples)
    eval_rows: List[Dict[str, Any]]

    if eval_split:
        eval_rows = build_model_rows(dataset_spec, split=eval_split, max_samples=max_eval_samples)
    else:
        train_rows, eval_rows = stratified_train_val_split(train_rows, val_ratio=val_ratio, seed=seed)
        if max_eval_samples is not None:
            eval_rows = eval_rows[:max_eval_samples]

    test_rows: List[Dict[str, Any]] = []
    if test_split:
        test_rows = build_model_rows(dataset_spec, split=test_split, max_samples=max_test_samples)

    return {"train": train_rows, "eval": eval_rows, "test": test_rows}


def build_training_args(config: Dict[str, Any], output_dir: Path) -> TrainingArguments:
    train_cfg = config["training"]
    eval_cfg = config.get("evaluation", {})
    wandb_cfg = config.get("wandb", {})
    report_to = ["wandb"] if wandb_cfg.get("enabled", False) else []
    signature = inspect.signature(TrainingArguments.__init__)
    supported = set(signature.parameters.keys())

    metric_prefix = "test" if using_test_as_eval(config) else "eval"
    metric_for_best_model = str(eval_cfg.get("metric_for_best_model", "macro_f1"))
    if not metric_for_best_model.startswith(f"{metric_prefix}_"):
        metric_for_best_model = f"{metric_prefix}_{metric_for_best_model}"

    kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 3)),
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(train_cfg.get("learning_rate", 2e-4)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
        "logging_steps": int(train_cfg.get("logging_steps", 10)),
        "save_strategy": str(eval_cfg.get("save_strategy", "epoch")),
        "save_total_limit": int(eval_cfg.get("save_total_limit", 2)),
        "load_best_model_at_end": bool(eval_cfg.get("load_best_model_at_end", True)),
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": True,
        "fp16": bool(train_cfg.get("fp16", False)),
        "bf16": bool(train_cfg.get("bf16", False)),
        "dataloader_num_workers": int(train_cfg.get("dataloader_num_workers", 0)),
        "remove_unused_columns": False,
        "report_to": report_to,
        "run_name": wandb_cfg.get("run_name"),
    }

    eval_strategy_value = str(eval_cfg.get("eval_strategy", "epoch"))
    if "eval_strategy" in supported:
        kwargs["eval_strategy"] = eval_strategy_value
    elif "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = eval_strategy_value

    if "overwrite_output_dir" in supported:
        kwargs["overwrite_output_dir"] = True

    warmup_ratio = train_cfg.get("warmup_ratio", 0.03)
    if "warmup_ratio" in supported:
        kwargs["warmup_ratio"] = float(warmup_ratio)
    elif "warmup_steps" in supported:
        kwargs["warmup_steps"] = int(train_cfg.get("warmup_steps", 0))

    ddp_find_unused = train_cfg.get("ddp_find_unused_parameters", False)
    if "ddp_find_unused_parameters" in supported:
        kwargs["ddp_find_unused_parameters"] = bool(ddp_find_unused)

    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in supported and value is not None
    }
    return TrainingArguments(**filtered_kwargs)


def maybe_configure_wandb(config: Dict[str, Any]) -> None:
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        os.environ.setdefault("WANDB_DISABLED", "true")
        return
    if wandb_cfg.get("project"):
        os.environ["WANDB_PROJECT"] = str(wandb_cfg["project"])
    if wandb_cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = str(wandb_cfg["entity"])
    if wandb_cfg.get("mode"):
        os.environ["WANDB_MODE"] = str(wandb_cfg["mode"])


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    maybe_configure_wandb(config)

    paths_cfg = config["paths"]
    dataset_cfg_path = Path(paths_cfg["dataset_config"])
    model_path = Path(paths_cfg["model_path"])
    output_dir = Path(paths_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config["data"]
    rows = resolve_rows(
        dataset_cfg_path=dataset_cfg_path,
        train_split=str(data_cfg.get("train_split", "train")),
        eval_split=data_cfg.get("eval_split"),
        test_split=data_cfg.get("test_split"),
        max_train_samples=data_cfg.get("max_train_samples"),
        max_eval_samples=data_cfg.get("max_eval_samples"),
        max_test_samples=data_cfg.get("max_test_samples"),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=int(config.get("seed", 42)),
    )

    processor, backbone, loader_name = load_processor_and_backbone(
        model_path=model_path,
        torch_dtype=resolve_dtype(str(config["model"].get("torch_dtype", "bfloat16"))),
        attn_implementation=config["model"].get("attn_implementation"),
        trust_remote_code=bool(config["model"].get("trust_remote_code", True)),
    )
    print(f"Loaded backbone with: {loader_name}")
    image_token_id = infer_image_token_id(backbone, processor)

    if bool(config["model"].get("gradient_checkpointing", True)) and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()

    lora_cfg = config.get("lora", {})
    backbone = maybe_apply_lora(
        backbone=backbone,
        enabled=bool(lora_cfg.get("enabled", True)),
        r=int(lora_cfg.get("r", 16)),
        alpha=int(lora_cfg.get("alpha", 32)),
        dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
    )
    if hasattr(backbone, "print_trainable_parameters"):
        backbone.print_trainable_parameters()

    model = QwenVLForFakeNewsClassification(
        backbone=backbone,
        config=ClassifierConfig(
            dropout=float(config["head"].get("dropout", 0.1)),
            num_labels=2,
            mode=str(config["head"].get("mode", "last_token_linear")),
            pooling=str(config["head"].get("pooling", "last_token")),
            cross_attn_layers=int(config["head"].get("cross_attn_layers", 1)),
            cross_attn_heads=int(config["head"].get("cross_attn_heads", 8)),
            cross_attn_dropout=float(config["head"].get("cross_attn_dropout", 0.1)),
            cross_attn_ffn_mult=int(config["head"].get("cross_attn_ffn_mult", 4)),
            fusion_hidden_dim=int(config["head"].get("fusion_hidden_dim", 1024)),
            moe_enabled=bool(config.get("moe", {}).get("enabled", False)),
            num_experts=int(config.get("moe", {}).get("num_experts", 4)),
            router_hidden_dim=int(config.get("moe", {}).get("router_hidden_dim", 512)),
            router_dropout=float(config.get("moe", {}).get("router_dropout", 0.1)),
            router_temperature=float(config.get("moe", {}).get("router_temperature", 1.0)),
            expert_hidden_dim=int(config.get("moe", {}).get("expert_hidden_dim", 1024)),
            expert_dropout=float(config.get("moe", {}).get("expert_dropout", 0.1)),
            load_balance_weight=float(config.get("moe", {}).get("load_balance_weight", 0.01)),
            entropy_weight=float(config.get("moe", {}).get("entropy_weight", 0.001)),
        ),
    )

    collator = QwenVLClassificationCollator(
        processor=processor,
        config=CollatorConfig(
            system_prompt=str(config["prompt"].get("system_prompt", "")),
            user_prompt_prefix=str(config["prompt"].get("user_prompt_prefix", "")),
            add_generation_prompt=bool(config["prompt"].get("add_generation_prompt", True)),
            max_length=config["prompt"].get("max_length"),
            image_token_id=image_token_id,
            debug_masks=bool(config["prompt"].get("debug_masks", False)),
            debug_max_tokens=int(config["prompt"].get("debug_max_tokens", 256)),
        ),
    )

    eval_metric_prefix = "test" if using_test_as_eval(config) else "eval"

    trainer = SplitAwareTrainer(
        model=model,
        args=build_training_args(config, output_dir),
        train_dataset=MultimodalNewsDataset(rows["train"]),
        eval_dataset=MultimodalNewsDataset(rows["eval"]),
        data_collator=collator,
        compute_metrics=compute_metrics_from_eval_pred,
        eval_metric_prefix=eval_metric_prefix,
    )

    trainer.train()

    processor.save_pretrained(output_dir / "processor")
    (output_dir / "used_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    torch.save(trainer.model.state_dict(), output_dir / "classifier_model.pt")

    eval_metrics = trainer.evaluate(metric_key_prefix=eval_metric_prefix)
    save_metrics(eval_metrics, output_dir / "eval_metrics.json")

    if rows["test"]:
        test_output = trainer.predict(MultimodalNewsDataset(rows["test"]))
        logits = np.asarray(test_output.predictions)
        labels = np.asarray(test_output.label_ids)
        preds = logits.argmax(axis=-1)
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)

        from MoeDet.metrics import binary_metrics, summarized_metrics

        test_metrics = summarized_metrics(binary_metrics(labels, preds, probs))
        save_metrics(test_metrics, output_dir / "test_metrics.json")
        prediction_rows = []
        for row, pred, score in zip(rows["test"], preds.tolist(), probs[:, 1].tolist()):
            prediction_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "text": row["text"],
                    "gold_label": row["label"],
                    "pred_label": int(pred),
                    "fake_probability": float(score),
                    "image_path": row["image_path"],
                }
            )
        save_predictions(prediction_rows, output_dir / "test_predictions.csv")

    summary = {
        "train_size": len(rows["train"]),
        "eval_size": len(rows["eval"]),
        "test_size": len(rows["test"]),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
