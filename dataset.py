from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image

from .data_spec import DatasetSpec


DEFAULT_SYSTEM_PROMPT = "You are a multimodal misinformation detection encoder."
DEFAULT_USER_PROMPT_PREFIX = (
    "Classify the multimodal news item as REAL or FAKE based on the text claim, image authenticity, and text-image consistency."
)
DEFAULT_NEWS_START_MARKER = "<<<NEWS_TEXT_START>>>"
DEFAULT_NEWS_END_MARKER = "<<<NEWS_TEXT_END>>>"


def stratified_train_val_split(
    rows: Sequence[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not 0.0 < val_ratio < 1.0:
        return list(rows), []

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        label = int(row["label"])
        grouped.setdefault(label, []).append(dict(row))

    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    rng = random.Random(seed)

    for label_rows in grouped.values():
        rng.shuffle(label_rows)
        val_count = max(1, int(round(len(label_rows) * val_ratio)))
        if val_count >= len(label_rows):
            val_count = max(1, len(label_rows) - 1)
        val_rows.extend(label_rows[:val_count])
        train_rows.extend(label_rows[val_count:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def build_model_rows(
    dataset_spec: DatasetSpec,
    split: str,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    df = dataset_spec.load_split(split, max_samples=max_samples)
    dataset_spec.validate_dataframe(df, include_label=True)

    rows: List[Dict[str, Any]] = []
    for raw_row in df.to_dict(orient="records"):
        text = dataset_spec.resolve_text(raw_row)
        label = dataset_spec.resolve_canonical_label(raw_row)
        image_path = dataset_spec.resolve_image(raw_row, split=split)
        sample_id = str(raw_row.get(dataset_spec.id_column))

        if not text or label is None or image_path is None:
            continue

        rows.append(
            {
                "sample_id": sample_id,
                "text": text,
                "label": int(label),
                "image_path": str(image_path),
                "split": split,
            }
        )
    return rows


def build_chat_messages(
    text: str,
    system_prompt: str,
    user_prompt_prefix: str,
    news_start_marker: str,
    news_end_marker: str,
) -> List[Dict[str, Any]]:
    user_text = (
        f"{user_prompt_prefix.strip()}\n\n"
        f"News text:\n{news_start_marker}\n{text.strip()}\n{news_end_marker}\n\n"
        "Return the classification for this item."
    ).strip()
    return [
        {"role": "system", "content": system_prompt.strip()},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
    ]


@dataclass
class CollatorConfig:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt_prefix: str = DEFAULT_USER_PROMPT_PREFIX
    add_generation_prompt: bool = True
    max_length: Optional[int] = None
    news_start_marker: str = DEFAULT_NEWS_START_MARKER
    news_end_marker: str = DEFAULT_NEWS_END_MARKER
    image_token_id: Optional[int] = None
    debug_masks: bool = False
    debug_max_tokens: int = 256


class MultimodalNewsDataset(torch.utils.data.Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return dict(self.rows[index])


class QwenVLClassificationCollator:
    def __init__(self, processor: Any, config: CollatorConfig) -> None:
        self.processor = processor
        self.config = config
        self._debug_printed = False
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            self.news_start_ids = tokenizer(self.config.news_start_marker, add_special_tokens=False).input_ids
            self.news_end_ids = tokenizer(self.config.news_end_marker, add_special_tokens=False).input_ids
        else:
            self.news_start_ids = []
            self.news_end_ids = []

    @staticmethod
    def _find_subsequence(sequence: List[int], pattern: List[int]) -> int:
        if not pattern or len(pattern) > len(sequence):
            return -1
        last = len(sequence) - len(pattern) + 1
        for index in range(last):
            if sequence[index : index + len(pattern)] == pattern:
                return index
        return -1

    def _build_image_token_mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.config.image_token_id is None:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        return (input_ids == int(self.config.image_token_id)) & attention_mask.bool()

    def _build_text_token_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row_index in range(input_ids.size(0)):
            ids = input_ids[row_index].tolist()
            valid_len = int(attention_mask[row_index].long().sum().item())
            ids = ids[:valid_len]

            start_index = self._find_subsequence(ids, self.news_start_ids)
            end_index = self._find_subsequence(ids, self.news_end_ids)

            if start_index >= 0 and end_index > start_index:
                start = start_index + len(self.news_start_ids)
                end = end_index
                batch_mask[row_index, start:end] = True
            else:
                batch_mask[row_index] = attention_mask[row_index].bool() & (~image_token_mask[row_index])

        batch_mask &= attention_mask.bool()
        batch_mask &= ~image_token_mask
        return batch_mask

    @staticmethod
    def _ensure_non_empty_mask(
        token_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        fallback_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        token_mask = token_mask.clone()
        valid_mask = attention_mask.bool()
        for row_index in range(token_mask.size(0)):
            if token_mask[row_index].any():
                continue
            if fallback_mask is not None and fallback_mask[row_index].any():
                token_mask[row_index] = fallback_mask[row_index]
            else:
                token_mask[row_index] = valid_mask[row_index]
        return token_mask

    def _maybe_print_mask_debug(
        self,
        batch: Dict[str, Any],
        texts: Sequence[str],
        sample_ids: Sequence[str],
    ) -> None:
        if not self.config.debug_masks or self._debug_printed:
            return
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            print("[mask_debug] tokenizer unavailable; skip debug print.")
            self._debug_printed = True
            return

        input_ids = batch["input_ids"][0].tolist()
        attention_mask = batch["attention_mask"][0].bool().tolist()
        text_mask = batch["text_token_mask"][0].bool().tolist()
        image_mask = batch["image_token_mask"][0].bool().tolist()
        valid_len = sum(1 for value in attention_mask if value)
        valid_ids = input_ids[:valid_len]
        valid_text_mask = text_mask[:valid_len]
        valid_image_mask = image_mask[:valid_len]
        overlap = sum(1 for text_flag, image_flag in zip(valid_text_mask, valid_image_mask) if text_flag and image_flag)

        tokens = tokenizer.convert_ids_to_tokens(valid_ids)
        printable_limit = min(len(valid_ids), int(self.config.debug_max_tokens))

        print("[mask_debug] ==================================================")
        print(f"[mask_debug] sample_id={sample_ids[0]}")
        print(
            f"[mask_debug] valid_tokens={valid_len} "
            f"text_tokens={sum(valid_text_mask)} "
            f"image_tokens={sum(valid_image_mask)} "
            f"overlap={overlap}"
        )
        print(f"[mask_debug] image_token_id={self.config.image_token_id}")
        print(f"[mask_debug] news_start_marker={self.config.news_start_marker}")
        print(f"[mask_debug] news_end_marker={self.config.news_end_marker}")
        print("[mask_debug] first_prompt_text=")
        print(texts[0][:1000])
        print("[mask_debug] token table (idx | txt_mask | img_mask | token)")
        for idx in range(printable_limit):
            token = tokens[idx].replace("\n", "\\n")
            print(f"[mask_debug] {idx:04d} | {int(valid_text_mask[idx])} | {int(valid_image_mask[idx])} | {token}")

        selected_text_ids = [token_id for token_id, keep in zip(valid_ids, valid_text_mask) if keep]
        selected_image_ids = [token_id for token_id, keep in zip(valid_ids, valid_image_mask) if keep]
        print("[mask_debug] decoded_text_mask_tokens=")
        print(tokenizer.decode(selected_text_ids[:printable_limit], skip_special_tokens=False))
        print("[mask_debug] decoded_image_mask_tokens=")
        print(tokenizer.decode(selected_image_ids[:printable_limit], skip_special_tokens=False))
        print("[mask_debug] ==================================================")
        self._debug_printed = True

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        texts: List[str] = []
        images: List[Image.Image] = []
        labels: List[int] = []
        sample_ids: List[str] = []

        for feature in features:
            messages = build_chat_messages(
                text=feature["text"],
                system_prompt=self.config.system_prompt,
                user_prompt_prefix=self.config.user_prompt_prefix,
                news_start_marker=self.config.news_start_marker,
                news_end_marker=self.config.news_end_marker,
            )
            chat_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=self.config.add_generation_prompt,
            )
            texts.append(chat_text)
            images.append(Image.open(feature["image_path"]).convert("RGB"))
            labels.append(int(feature["label"]))
            sample_ids.append(str(feature["sample_id"]))

        try:
            processor_kwargs: Dict[str, Any] = {
                "text": texts,
                "images": images,
                "padding": True,
                "return_tensors": "pt",
            }
            max_length = self.config.max_length
            if max_length is not None:
                processor_kwargs["truncation"] = True
                processor_kwargs["max_length"] = max_length
            batch = self.processor(**processor_kwargs)
        except ValueError as exc:
            message = str(exc)
            if "Mismatch in `image` token count" not in message:
                raise
            batch = self.processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
        finally:
            for image in images:
                image.close()

        attention_mask = batch["attention_mask"]
        input_ids = batch["input_ids"]
        image_token_mask = self._build_image_token_mask(input_ids, attention_mask)
        text_token_mask = self._build_text_token_mask(input_ids, attention_mask, image_token_mask)
        image_token_mask = self._ensure_non_empty_mask(image_token_mask, attention_mask, fallback_mask=attention_mask.bool() & (~text_token_mask))
        text_token_mask = self._ensure_non_empty_mask(text_token_mask, attention_mask, fallback_mask=attention_mask.bool() & (~image_token_mask))

        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        batch["sample_ids"] = sample_ids
        batch["image_token_mask"] = image_token_mask
        batch["text_token_mask"] = text_token_mask
        self._maybe_print_mask_debug(batch, texts, sample_ids)
        return batch
