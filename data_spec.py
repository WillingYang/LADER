from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml


IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
)

CANONICAL_LABEL_TO_ID = {"real": 0, "fake": 1}


@dataclass
class DatasetSpec:
    name: str
    splits: Dict[str, Path]
    image_root: Path
    split_image_roots: Dict[str, Path]
    id_column: str
    image_id_column: Optional[str]
    image_path_column: Optional[str]
    text_column: str
    fallback_text_column: Optional[str]
    label_column: Optional[str]
    metadata_fields: List[str]
    image_extensions: Tuple[str, ...]
    output_label_map: Dict[Any, str]
    raw_config: Dict[str, Any]

    def split_path(self, split: str) -> Path:
        if split not in self.splits:
            raise KeyError(f"Unknown split '{split}' for dataset '{self.name}'.")
        return self.splits[split]

    def image_root_for_split(self, split: Optional[str]) -> Path:
        if split and split in self.split_image_roots:
            return self.split_image_roots[split]
        return self.image_root

    def load_split(self, split: str, max_samples: Optional[int] = None) -> pd.DataFrame:
        path = self.split_path(split)
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            df = pd.read_json(path)
        else:
            sep = "\t" if suffix in {".tsv", ".txt"} else ","
            df = pd.read_csv(path, sep=sep, low_memory=False)
        if max_samples is not None:
            df = df.head(max_samples).copy()
        else:
            df = df.copy()
        return df

    def required_columns(self, include_label: bool) -> List[str]:
        columns = [self.id_column, self.text_column]
        if self.fallback_text_column:
            columns.append(self.fallback_text_column)
        if include_label and self.label_column:
            columns.append(self.label_column)
        deduped: List[str] = []
        for column in columns:
            if column and column not in deduped:
                deduped.append(column)
        return deduped

    def validate_dataframe(self, df: pd.DataFrame, include_label: bool = True) -> None:
        missing = [column for column in self.required_columns(include_label) if column not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns for dataset '{self.name}': {missing}")

    def resolve_text(self, row: Dict[str, Any]) -> str:
        primary = normalize_text(row.get(self.text_column))
        if primary:
            return primary
        if self.fallback_text_column:
            return normalize_text(row.get(self.fallback_text_column))
        return ""

    def resolve_canonical_label(self, row: Dict[str, Any]) -> Optional[int]:
        if not self.label_column:
            return None
        raw_label = row.get(self.label_column)
        if raw_label is None or (isinstance(raw_label, float) and pd.isna(raw_label)):
            return None

        mapped = None
        if raw_label in self.output_label_map:
            mapped = self.output_label_map[raw_label]
        else:
            raw_text = str(raw_label).strip()
            if raw_text in self.output_label_map:
                mapped = self.output_label_map[raw_text]
            else:
                lowered = raw_text.lower()
                if lowered in CANONICAL_LABEL_TO_ID:
                    mapped = lowered

        if mapped is None:
            return None
        lowered = str(mapped).strip().lower()
        return CANONICAL_LABEL_TO_ID.get(lowered)

    def resolve_image(self, row: Dict[str, Any], split: Optional[str] = None) -> Optional[Path]:
        image_root = self.image_root_for_split(split)

        if self.image_path_column:
            raw_image_path = row.get(self.image_path_column)
            resolved = resolve_image_reference(image_root, raw_image_path)
            if resolved is not None:
                return resolved

        sample_id = normalize_text(row.get(self.id_column))
        if self.image_id_column:
            image_id = normalize_text(row.get(self.image_id_column))
            if image_id:
                resolved = resolve_image_path(image_root, image_id, self.image_extensions)
                if resolved is not None:
                    return resolved

        if sample_id:
            return resolve_image_path(image_root, sample_id, self.image_extensions)
        return None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def resolve_image_reference(image_root: Path, raw_image_path: Any) -> Optional[Path]:
    if raw_image_path is None:
        return None
    text = str(raw_image_path).strip()
    if not text:
        return None

    candidate = Path(text)
    if candidate.is_absolute() and candidate.exists() and candidate.is_file():
        return candidate

    candidate = image_root / text.lstrip("/\\")
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def resolve_image_path(
    image_root: Path,
    sample_id: str,
    image_extensions: Sequence[str] = IMAGE_EXTENSIONS,
) -> Optional[Path]:
    direct_path = image_root / sample_id
    if direct_path.exists() and direct_path.is_file():
        return direct_path
    for extension in image_extensions:
        candidate = image_root / f"{sample_id}{extension}"
        if candidate.exists() and candidate.is_file():
            return candidate
    matches = sorted(image_root.glob(f"{sample_id}.*"))
    for match in matches:
        if match.is_file():
            return match
    return None


def parse_yaml_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_output_label_map(raw_mapping: Dict[Any, Any]) -> Dict[Any, str]:
    parsed: Dict[Any, str] = {}
    for key, value in raw_mapping.items():
        normalized_value = str(value).strip().lower()
        try:
            parsed[int(key)] = normalized_value
        except (TypeError, ValueError):
            parsed[str(key).strip()] = normalized_value
    return parsed


def resolve_config_path(
    value: Any,
    *,
    project_root: Path,
    base_root: Optional[Path] = None,
) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    if base_root is not None:
        return base_root / candidate
    return project_root / candidate


def load_dataset_spec(config_path: Path) -> DatasetSpec:
    raw = parse_yaml_config(config_path)
    dataset = raw["dataset"]
    project_root = config_path.parents[2] if len(config_path.parents) >= 3 else config_path.parent
    data_root = resolve_config_path(
        dataset.get("data_root"),
        project_root=project_root,
    )
    splits_root = resolve_config_path(
        dataset.get("splits_root"),
        project_root=project_root,
        base_root=data_root,
    )
    image_root = resolve_config_path(
        dataset["image_root"],
        project_root=project_root,
        base_root=data_root,
    )
    if image_root is None:
        raise ValueError(f"dataset.image_root is required in {config_path}")

    return DatasetSpec(
        name=str(dataset["name"]),
        splits={
            name: resolve_config_path(path, project_root=project_root, base_root=splits_root or data_root)
            for name, path in dataset["splits"].items()
        },
        image_root=image_root,
        split_image_roots={
            name: resolve_config_path(path, project_root=project_root, base_root=data_root)
            for name, path in dataset.get("split_image_roots", {}).items()
        },
        id_column=str(dataset["id_column"]),
        image_id_column=str(dataset["image_id_column"]) if dataset.get("image_id_column") else None,
        image_path_column=str(dataset["image_path_column"]) if dataset.get("image_path_column") else None,
        text_column=str(dataset["text_column"]),
        fallback_text_column=str(dataset["fallback_text_column"]) if dataset.get("fallback_text_column") else None,
        label_column=str(dataset["label_column"]) if dataset.get("label_column") else None,
        metadata_fields=[str(item) for item in dataset.get("metadata_fields", [])],
        image_extensions=tuple(str(item) for item in dataset.get("image_extensions", IMAGE_EXTENSIONS)),
        output_label_map=parse_output_label_map(dataset.get("output_label_map", {})),
        raw_config=raw,
    )
