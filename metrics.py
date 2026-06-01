from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def binary_metrics(labels: Iterable[int], preds: Iterable[int], probs: Optional[np.ndarray] = None) -> Dict[str, Any]:
    y_true = np.asarray(list(labels), dtype=np.int64)
    y_pred = np.asarray(list(preds), dtype=np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    precision_fake = tp / max(tp + fp, 1)
    recall_fake = tp / max(tp + fn, 1)
    f1_fake = 2 * precision_fake * recall_fake / max(precision_fake + recall_fake, 1e-12)

    precision_real = tn / max(tn + fn, 1)
    recall_real = tn / max(tn + fp, 1)
    f1_real = 2 * precision_real * recall_real / max(precision_real + recall_real, 1e-12)

    metrics: Dict[str, Any] = {
        "accuracy": accuracy,
        "precision_fake": precision_fake,
        "recall_fake": recall_fake,
        "f1_fake": f1_fake,
        "precision_real": precision_real,
        "recall_real": recall_real,
        "f1_real": f1_real,
        "macro_f1": (f1_fake + f1_real) / 2.0,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "num_samples": int(len(y_true)),
    }

    if probs is not None and probs.ndim == 2 and probs.shape[1] == 2:
        try:
            from sklearn.metrics import roc_auc_score

            metrics["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
        except Exception:
            pass
    return metrics


def compute_metrics_from_eval_pred(eval_pred: Any) -> Dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    preds = logits.argmax(axis=-1)
    probs = softmax(logits)
    metrics = binary_metrics(labels, preds, probs)
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "f1_fake": metrics["f1_fake"],
        "f1_real": metrics["f1_real"],
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def save_predictions(
    rows: List[Dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
