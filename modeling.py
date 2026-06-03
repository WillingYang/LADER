from __future__ import annotations

import math
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


def infer_image_token_id(backbone: Any, processor: Any) -> Optional[int]:
    config_candidates = [
        getattr(getattr(backbone, "config", None), "image_token_id", None),
        getattr(getattr(getattr(backbone, "config", None), "text_config", None), "image_token_id", None),
        getattr(getattr(getattr(backbone, "config", None), "language_config", None), "image_token_id", None),
    ]
    for value in config_candidates:
        if isinstance(value, int) and value >= 0:
            return value

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return None

    token_candidates: List[Any] = [
        getattr(processor, "image_token", None),
        getattr(tokenizer, "image_token", None),
        "<|image_pad|>",
        "<image>",
        "<|vision_pad|>",
        "<|img|>",
    ]
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for token in token_candidates:
        if not token:
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id != unk_token_id:
            return token_id
    return None


@dataclass
class ClassifierConfig:
    dropout: float = 0.1
    num_labels: int = 2
    mode: str = "last_token_linear"
    pooling: str = "last_token"
    cross_attn_layers: int = 1
    cross_attn_heads: int = 8
    cross_attn_dropout: float = 0.1
    cross_attn_ffn_mult: int = 4
    fusion_hidden_dim: int = 1024
    moe_enabled: bool = False
    num_experts: int = 4
    router_hidden_dim: int = 512
    router_dropout: float = 0.1
    router_temperature: float = 1.0
    expert_hidden_dim: int = 1024
    expert_dropout: float = 0.1
    load_balance_weight: float = 0.01
    entropy_weight: float = 0.001

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dropout": self.dropout,
            "num_labels": self.num_labels,
            "mode": self.mode,
            "pooling": self.pooling,
            "cross_attn_layers": self.cross_attn_layers,
            "cross_attn_heads": self.cross_attn_heads,
            "cross_attn_dropout": self.cross_attn_dropout,
            "cross_attn_ffn_mult": self.cross_attn_ffn_mult,
            "fusion_hidden_dim": self.fusion_hidden_dim,
            "moe_enabled": self.moe_enabled,
            "num_experts": self.num_experts,
            "router_hidden_dim": self.router_hidden_dim,
            "router_dropout": self.router_dropout,
            "router_temperature": self.router_temperature,
            "expert_hidden_dim": self.expert_hidden_dim,
            "expert_dropout": self.expert_dropout,
            "load_balance_weight": self.load_balance_weight,
            "entropy_weight": self.entropy_weight,
        }


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float, ffn_mult: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.kv_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        ffn_hidden = hidden_size * max(ffn_mult, 1)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_size),
        )

    def forward(
        self,
        query: torch.Tensor,
        hidden_states: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(query),
            self.kv_norm(hidden_states),
            self.kv_norm(hidden_states),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query = query + attn_out
        query = query + self.ffn(self.ffn_norm(query))
        return query


class QueryCrossAttentionEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, dropout: float, ffn_mult: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    ffn_mult=ffn_mult,
                )
                for _ in range(max(num_layers, 1))
            ]
        )

    def forward(
        self,
        query: torch.Tensor,
        hidden_states: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = query
        for layer in self.layers:
            output = layer(output, hidden_states, key_padding_mask=key_padding_mask)
        return output


class MoEFusionHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, config: ClassifierConfig) -> None:
        super().__init__()
        router_input_dim = hidden_size * 3
        expert_input_dim = hidden_size * 3
        self.num_experts = max(config.num_experts, 1)
        self.temperature = max(float(config.router_temperature), 1e-6)
        self.load_balance_weight = float(config.load_balance_weight)
        self.entropy_weight = float(config.entropy_weight)

        self.router = nn.Sequential(
            nn.LayerNorm(router_input_dim),
            nn.Linear(router_input_dim, config.router_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.router_dropout),
            nn.Linear(config.router_hidden_dim, self.num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(expert_input_dim),
                    nn.Linear(expert_input_dim, config.expert_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.expert_dropout),
                    nn.Linear(config.expert_hidden_dim, num_labels),
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(
        self,
        pooled_repr: torch.Tensor,
        img_repr: torch.Tensor,
        txt_repr: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        router_input = torch.cat([pooled_repr, img_repr, txt_repr], dim=-1)
        expert_input = router_input

        router_logits = self.router(router_input) / self.temperature
        router_probs = torch.softmax(router_logits, dim=-1)

        expert_logits = torch.stack([expert(expert_input) for expert in self.experts], dim=1)
        logits = torch.sum(router_probs.unsqueeze(-1) * expert_logits, dim=1)

        avg_probs = router_probs.mean(dim=0)
        uniform = torch.full_like(avg_probs, 1.0 / self.num_experts)
        load_balance_loss = torch.sum((avg_probs - uniform) ** 2)
        entropy = -(router_probs.clamp_min(1e-8) * router_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        aux_loss = self.load_balance_weight * load_balance_loss - self.entropy_weight * entropy
        return logits, aux_loss, router_probs


class QwenVLForFakeNewsClassification(nn.Module):
    def __init__(self, backbone: Any, config: ClassifierConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        hidden_size = infer_hidden_size(backbone)
        self.dropout = nn.Dropout(config.dropout)
        self.hidden_size = hidden_size

        if self.config.mode == "dual_query_moe":
            self.img_query = nn.Parameter(torch.randn(1, 1, hidden_size) / math.sqrt(hidden_size))
            self.txt_query = nn.Parameter(torch.randn(1, 1, hidden_size) / math.sqrt(hidden_size))
            self.img_encoder = QueryCrossAttentionEncoder(
                hidden_size=hidden_size,
                num_layers=self.config.cross_attn_layers,
                num_heads=self.config.cross_attn_heads,
                dropout=self.config.cross_attn_dropout,
                ffn_mult=self.config.cross_attn_ffn_mult,
            )
            self.txt_encoder = QueryCrossAttentionEncoder(
                hidden_size=hidden_size,
                num_layers=self.config.cross_attn_layers,
                num_heads=self.config.cross_attn_heads,
                dropout=self.config.cross_attn_dropout,
                ffn_mult=self.config.cross_attn_ffn_mult,
            )
            if self.config.moe_enabled:
                self.moe_head = MoEFusionHead(hidden_size=hidden_size, num_labels=config.num_labels, config=config)
                self.fusion_head = None
            else:
                fusion_dim = hidden_size * 3
                self.moe_head = None
                self.fusion_head = nn.Sequential(
                    nn.LayerNorm(fusion_dim),
                    nn.Linear(fusion_dim, self.config.fusion_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(self.config.fusion_hidden_dim, config.num_labels),
                )
        else:
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

    @staticmethod
    def _ensure_non_empty_mask(token_mask: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if token_mask is None or attention_mask is None:
            return token_mask
        token_mask = token_mask.bool().clone()
        fallback = attention_mask.bool()
        for row_index in range(token_mask.size(0)):
            if not token_mask[row_index].any():
                token_mask[row_index] = fallback[row_index]
        return token_mask

    def forward(self, labels: Optional[torch.Tensor] = None, sample_ids: Optional[List[str]] = None, **inputs: Any) -> SequenceClassifierOutput:
        outputs = self.backbone(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]
        attention_mask = inputs.get("attention_mask")

        aux_loss = None
        if self.config.mode == "dual_query_moe":
            pooled = self._pool_hidden(hidden, attention_mask)
            image_token_mask = self._ensure_non_empty_mask(inputs.get("image_token_mask"), attention_mask)
            text_token_mask = self._ensure_non_empty_mask(inputs.get("text_token_mask"), attention_mask)
            batch_size = hidden.size(0)

            img_query = self.img_query.expand(batch_size, -1, -1)
            txt_query = self.txt_query.expand(batch_size, -1, -1)

            img_repr = self.img_encoder(
                img_query,
                hidden,
                key_padding_mask=(~image_token_mask) if image_token_mask is not None else None,
            ).squeeze(1)
            txt_repr = self.txt_encoder(
                txt_query,
                hidden,
                key_padding_mask=(~text_token_mask) if text_token_mask is not None else None,
            ).squeeze(1)

            if self.moe_head is not None:
                logits, aux_loss, _ = self.moe_head(pooled, img_repr, txt_repr)
            else:
                fused = torch.cat([pooled, img_repr, txt_repr], dim=-1)
                logits = self.fusion_head(fused)
        else:
            pooled = self._pool_hidden(hidden, attention_mask)
            logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            if aux_loss is not None:
                loss = loss + aux_loss

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
