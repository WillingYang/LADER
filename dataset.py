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
) -> List[Dict[str, Any]]:
    user_text = (
        f"{user_prompt_prefix.strip()}\n\n"
        f"News text:\n{text.strip()}\n\n"
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

        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        batch["sample_ids"] = sample_ids
        return batch
