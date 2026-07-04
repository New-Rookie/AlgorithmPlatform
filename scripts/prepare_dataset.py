from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from src.algorithm_platform.utils.config import load_config


def resolve_video_path(videos_root: str, value: str) -> str:
    p = Path(str(value))
    if p.is_absolute() and p.exists():
        return str(p)
    return str(Path(videos_root) / str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset manifest from yaml config")
    parser.add_argument("--config", required=True, help="Path to dataset yaml config")
    args = parser.parse_args()

    cfg = load_config(args.config).raw
    paths = cfg["paths"]
    cols = cfg["columns"]
    subset = cfg.get("subset", {})
    metadata = cfg.get("metadata", {})
    split_cfg = cfg.get("split", {})

    df = pd.read_csv(paths["annotations"])
    required = [cols["video"], cols["start"], cols["end"], cols["label"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Annotation file missing required columns: {missing}. Available columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["video_path"] = df[cols["video"]].astype(str).apply(lambda x: resolve_video_path(paths["videos_root"], x))
    out["start"] = df[cols["start"]].astype(float)
    out["end"] = df[cols["end"]].astype(float)
    out["label"] = df[cols["label"]].astype(str)

    split_col = cols.get("split")
    if split_col and split_col in df.columns:
        out["split"] = df[split_col].astype(str)
    else:
        out["split"] = "train"

    station_col = cols.get("station")
    if station_col and station_col in df.columns:
        out["station_id"] = df[station_col].astype(str)
    else:
        out["station_id"] = "station_public"

    out["sop_id"] = metadata.get("sop_id", "sop_public")
    out["employee_id"] = metadata.get("employee_id", "emp_public")
    out = out[out["end"] > out["start"]].reset_index(drop=True)

    labels: List[str] = subset.get("labels") or []
    if labels:
        out = out[out["label"].isin(labels)].reset_index(drop=True)

    if subset.get("require_existing_videos", True):
        out = out[out["video_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)

    max_recordings = subset.get("max_recordings")
    if max_recordings is not None and len(out):
        unique_videos = out["video_path"].drop_duplicates().sample(
            n=min(int(max_recordings), out["video_path"].nunique()),
            random_state=int(split_cfg.get("seed", 42)),
        )
        out = out[out["video_path"].isin(set(unique_videos))].reset_index(drop=True)

    max_rows = subset.get("max_rows")
    if max_rows is not None and len(out):
        max_rows = int(max_rows)
        if subset.get("balanced", True) and out["label"].nunique() > 1:
            per_label = max(1, max_rows // out["label"].nunique())
            out = (
                out.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(n=min(per_label, len(g)), random_state=int(split_cfg.get("seed", 42))))
                .reset_index(drop=True)
            )
            if len(out) > max_rows:
                out = out.sample(n=max_rows, random_state=int(split_cfg.get("seed", 42))).reset_index(drop=True)
        else:
            out = out.sample(n=min(max_rows, len(out)), random_state=int(split_cfg.get("seed", 42))).reset_index(drop=True)

    if out.empty:
        raise ValueError("No rows left after filtering. Check downloaded videos, labels, and subset config.")

    if out["split"].nunique() == 1 and out["split"].iloc[0] == "train" and len(out) >= 10:
        val_size = float(split_cfg.get("val_size", 0.15))
        test_size = float(split_cfg.get("test_size", 0.10))
        seed = int(split_cfg.get("seed", 42))
        total = val_size + test_size
        indices = list(range(len(out)))
        stratify = out["label"] if out["label"].nunique() > 1 and out["label"].value_counts().min() >= 2 else None
        train_idx, temp_idx = train_test_split(indices, test_size=total, random_state=seed, stratify=stratify)
        relative_test = test_size / total if total else 0.0
        val_idx, test_idx = train_test_split(temp_idx, test_size=relative_test, random_state=seed)
        out["split"] = "train"
        out.loc[val_idx, "split"] = "val"
        out.loc[test_idx, "split"] = "test"

    output_dir = paths["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")
    label_path = os.path.join(output_dir, "label_map.json")
    out.to_csv(manifest_path, index=False)

    labels_sorted = sorted(out["label"].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels_sorted)}
    id2label = {idx: label for label, idx in label2id.items()}
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote label map: {label_path}")
    print(f"Rows: {len(out)} | Videos: {out['video_path'].nunique()} | Labels: {len(labels_sorted)}")
    print(f"Splits: {out['split'].value_counts().to_dict()}")
    print(f"Labels: {labels_sorted}")


if __name__ == "__main__":
    main()
