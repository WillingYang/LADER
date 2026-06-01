from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor
from transformers.modeling_outputs import SequenceClassifierOutput


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_model_loader_candidates() -> List[Tuple[str, Any]]:
    import transformers

    loader_names = [
        "Qwen3VLForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ]
    candidates: List[Tuple[str, Any]] = []
    for name in loader_names:
        loader = getattr(transformers, name, None)
        if loader is not None:
            candidates.append((name, loader))
    return candidates


def load_processor_and_backbone(
    model_path: Path,
    torch_dtype: torch.dtype,
    attn_implementation: Optional[str],
    trust_remote_code: bool,
) -> Tuple[Any, Any, str]:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    load_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation and attn_implementation.lower() != "none":
        load_kwargs["attn_implementation"] = attn_implementation

    errors: List[str] = []
    for loader_name, loader in get_model_loader_candidates():
        try:
            model = loader.from_pretrained(model_path, **load_kwargs)
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is not None:
                for config_obj in (getattr(model, "config", None), getattr(model, "generation_config", None)):
                    if config_obj is None:
                        continue
                    if getattr(tokenizer, "bos_token_id", None) is not None:
                        config_obj.bos_token_id = tokenizer.bos_token_id
                    if getattr(tokenizer, "eos_token_id", None) is not None:
                        config_obj.eos_token_id = tokenizer.eos_token_id
                    if getattr(tokenizer, "pad_token_id", None) is not None:
                        config_obj.pad_token_id = tokenizer.pad_token_id
            return processor, model, loader_name
        except Exception as exc:
            errors.append(f"{loader_name}: {exc}")

    joined = "\n".join(errors)
    raise RuntimeError(f"Failed to load backbone from {model_path}.\nTried:\n{joined}")


def infer_hidden_size(backbone: Any) -> int:
    config = getattr(backbone, "config", None)
    candidates = [
        getattr(config, "hidden_size", None),
        getattr(getattr(config, "text_config", None), "hidden_size", None),
        getattr(getattr(config, "language_config", None), "hidden_size", None),
    ]
    for value in candidates:
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Could not infer hidden size from backbone config.")


def maybe_apply_lora(
    backbone: Any,
    enabled: bool,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: List[str],
) -> Any:
    if not enabled:
        return backbone
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
    )
    return get_peft_model(backbone, lora_config)


@dataclass
class ClassifierConfig:
    dropout: float = 0.1
    num_labels: int = 2
    pooling: str = "last_token"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dropout": self.dropout,
            "num_labels": self.num_labels,
            "pooling": self.pooling,
        }


class QwenVLForFakeNewsClassification(nn.Module):
    def __init__(self, backbone: Any, config: ClassifierConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        hidden_size = infer_hidden_size(backbone)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(hidden_size, config.num_labels)

    def _pool_hidden(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.config.pooling == "mean":
            if attention_mask is None:
                return hidden_states.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            summed = (hidden_states * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return summed / denom

        if attention_mask is None:
            return hidden_states[:, -1, :]

        last_indices = attention_mask.long().sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, last_indices, :]

    def forward(self, labels: Optional[torch.Tensor] = None, sample_ids: Optional[List[str]] = None, **inputs: Any) -> SequenceClassifierOutput:
        outputs = self.backbone(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]
        pooled = self._pool_hidden(hidden, inputs.get("attention_mask"))
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
