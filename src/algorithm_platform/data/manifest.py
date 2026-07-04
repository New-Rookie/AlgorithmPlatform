from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

REQUIRED_COLUMNS = [
    "video_path",
    "start",
    "end",
    "label",
    "split",
    "station_id",
    "sop_id",
    "employee_id",
]


@dataclass
class LabelMaps:
    label2id: Dict[str, int]
    id2label: Dict[int, str]


def validate_manifest(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    if df.empty:
        raise ValueError("Manifest is empty")
    if (df["end"].astype(float) <= df["start"].astype(float)).any():
        raise ValueError("Manifest contains rows with end <= start")


def load_manifest(path: str, split: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_manifest(df)
    if split is not None:
        df = df[df["split"].astype(str) == split].reset_index(drop=True)
    return df


def build_label_maps(df: pd.DataFrame) -> LabelMaps:
    labels = sorted(df["label"].astype(str).unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return LabelMaps(label2id=label2id, id2label=id2label)


def save_label_maps(maps: LabelMaps, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": maps.label2id, "id2label": maps.id2label}, f, ensure_ascii=False, indent=2)


def load_label_maps(model_dir: str) -> LabelMaps:
    path = os.path.join(model_dir, "label_map.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    label2id = {str(k): int(v) for k, v in raw["label2id"].items()}
    id2label = {int(k): str(v) for k, v in raw["id2label"].items()}
    return LabelMaps(label2id=label2id, id2label=id2label)
