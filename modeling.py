from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    layer_expert_count: int = 4
    layer_top_k: int = 2
    layer_router_hidden_dim: int = 512
    layer_router_dropout: float = 0.1
    layer_router_temperature: float = 1.0
    layer_teacher_temperature: float = 1.0
    layer_decay: float = 0.5
    layer_expert_aux_weight: float = 0.2
    layer_router_loss_weight: float = 0.2
    residual_gate_hidden_dim: int = 512
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
            "layer_expert_count": self.layer_expert_count,
            "layer_top_k": self.layer_top_k,
            "layer_router_hidden_dim": self.layer_router_hidden_dim,
            "layer_router_dropout": self.layer_router_dropout,
            "layer_router_temperature": self.layer_router_temperature,
            "layer_teacher_temperature": self.layer_teacher_temperature,
            "layer_decay": self.layer_decay,
            "layer_expert_aux_weight": self.layer_expert_aux_weight,
            "layer_router_loss_weight": self.layer_router_loss_weight,
            "residual_gate_hidden_dim": self.residual_gate_hidden_dim,
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


class LayerwiseExpertFusionHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, config: ClassifierConfig) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.expert_count = max(int(config.layer_expert_count), 1)
        self.top_k = max(1, min(int(config.layer_top_k), self.expert_count))
        self.router_temperature = max(float(config.layer_router_temperature), 1e-6)
        self.teacher_temperature = max(float(config.layer_teacher_temperature), 1e-6)
        self.layer_decay = max(float(config.layer_decay), 0.0)
        self.expert_aux_weight = max(float(config.layer_expert_aux_weight), 0.0)
        self.router_loss_weight = max(float(config.layer_router_loss_weight), 0.0)
        expert_dim = int(config.fusion_hidden_dim)

        self.expert_projector = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, expert_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.final_projector = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, expert_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.shared_expert_head = nn.Linear(expert_dim, num_labels)
        self.router = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, int(config.layer_router_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(config.layer_router_dropout)),
            nn.Linear(int(config.layer_router_hidden_dim), self.expert_count),
        )
        gate_hidden = max(int(config.residual_gate_hidden_dim), expert_dim)
        self.residual_gate = nn.Sequential(
            nn.LayerNorm(expert_dim * 2),
            nn.Linear(expert_dim * 2, gate_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(gate_hidden, expert_dim),
        )
        self.final_classifier = nn.Linear(expert_dim, num_labels)

    @staticmethod
    def _compute_even_layer_indices(total_layers: int, num_experts: int) -> List[int]:
        if total_layers <= 0:
            return [0]
        if num_experts <= 1:
            return [total_layers - 1]
        positions = torch.linspace(0, total_layers - 1, steps=num_experts)
        indices = positions.round().to(torch.int64).tolist()
        deduped: List[int] = []
        for index in indices:
            index = max(0, min(total_layers - 1, int(index)))
            if index not in deduped:
                deduped.append(index)
        while len(deduped) < num_experts:
            candidate = total_layers - 1 - len(deduped)
            candidate = max(0, candidate)
            if candidate not in deduped:
                deduped.append(candidate)
            else:
                break
        return deduped[:num_experts]

    def _layer_prior_logits(self, num_layers: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if num_layers <= 1 or self.layer_decay <= 0.0:
            return torch.zeros(num_layers, device=device, dtype=dtype)
        depth_positions = torch.arange(num_layers, device=device, dtype=dtype)
        depth_positions = depth_positions / max(num_layers - 1, 1)
        return self.layer_decay * depth_positions

    def _select_topk_probs(self, router_probs: torch.Tensor) -> torch.Tensor:
        if self.top_k >= router_probs.size(-1):
            return router_probs
        topk_values, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        masked_probs = torch.zeros_like(router_probs)
        masked_probs.scatter_(dim=-1, index=topk_indices, src=topk_values)
        denom = masked_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return masked_probs / denom

    def forward(
        self,
        layer_hidden_states: List[torch.Tensor],
        pooled_repr: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        batch_size = pooled_repr.size(0)
        selected_indices = self._compute_even_layer_indices(len(layer_hidden_states), self.expert_count)
        selected_states = [layer_hidden_states[index] for index in selected_indices]
        layer_stack = torch.stack(selected_states, dim=1)
        projected_layers = self.expert_projector(layer_stack)
        expert_logits = self.shared_expert_head(projected_layers)

        prior_logits = self._layer_prior_logits(projected_layers.size(1), pooled_repr.device, pooled_repr.dtype)
        router_logits = self.router(pooled_repr) / self.router_temperature
        router_logits = router_logits + prior_logits.unsqueeze(0)
        router_probs = torch.softmax(router_logits, dim=-1)
        topk_router_probs = self._select_topk_probs(router_probs)

        fused_layer = torch.sum(topk_router_probs.unsqueeze(-1) * projected_layers, dim=1)
        final_proj = self.final_projector(pooled_repr)
        residual_gate = torch.sigmoid(self.residual_gate(torch.cat([final_proj, fused_layer], dim=-1)))
        fused_repr = final_proj + residual_gate * fused_layer
        logits = self.final_classifier(fused_repr)

        aux_loss: Optional[torch.Tensor] = None
        extras: Dict[str, torch.Tensor] = {
            "router_probs": router_probs,
            "topk_router_probs": topk_router_probs,
            "selected_layer_indices": torch.tensor(selected_indices, device=pooled_repr.device, dtype=torch.long),
        }

        if labels is not None:
            expanded_labels = labels.unsqueeze(1).expand(batch_size, projected_layers.size(1)).reshape(-1)
            expert_loss_per_sample = F.cross_entropy(
                expert_logits.reshape(-1, self.num_labels),
                expanded_labels,
                reduction="none",
            ).view(batch_size, projected_layers.size(1))
            teacher_probs = torch.softmax((-expert_loss_per_sample / self.teacher_temperature) + prior_logits.unsqueeze(0), dim=-1)
            router_log_probs = torch.log(router_probs.clamp_min(1e-12))
            router_loss = F.kl_div(router_log_probs, teacher_probs.detach(), reduction="batchmean")
            expert_aux_loss = expert_loss_per_sample.mean()
            aux_loss = (self.router_loss_weight * router_loss) + (self.expert_aux_weight * expert_aux_loss)
            extras["teacher_probs"] = teacher_probs
        return logits, aux_loss, extras


class QwenVLForFakeNewsClassification(nn.Module):
    def __init__(self, backbone: Any, config: ClassifierConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        hidden_size = infer_hidden_size(backbone)
        self.dropout = nn.Dropout(config.dropout)
        self.hidden_size = hidden_size
        self.layerwise_head: Optional[LayerwiseExpertFusionHead] = None

        if self.config.mode == "layerwise_topk_router":
            self.layerwise_head = LayerwiseExpertFusionHead(hidden_size=hidden_size, num_labels=config.num_labels, config=config)
        elif self.config.mode == "dual_query_moe":
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
        if self.config.mode == "layerwise_topk_router":
            pooled = self._pool_hidden(hidden, attention_mask)
            all_hidden_states = list(outputs.hidden_states[1:]) if len(outputs.hidden_states) > 1 else [hidden]
            pooled_layer_states = [self._pool_hidden(layer_hidden, attention_mask) for layer_hidden in all_hidden_states]
            logits, aux_loss, _ = self.layerwise_head(pooled_layer_states, pooled, labels=labels)
        elif self.config.mode == "dual_query_moe":
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
