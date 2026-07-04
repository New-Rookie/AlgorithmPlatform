from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def resolve_video_path(videos_root: str, value: str) -> str:
    p = Path(value)
    if p.is_absolute() and p.exists():
        return str(p)
    candidate = Path(videos_root) / value
    if candidate.exists():
        return str(candidate)
    # Keep a useful path even if the user has not copied videos yet.
    return str(candidate)


def parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None or not str(value).strip():
        return None
    return [item.strip() for item in str(value).split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Assembly101-like annotations into AlgorithmPlatform manifest.csv")
    parser.add_argument("--videos-root", required=True, help="Root directory containing Assembly101 videos")
    parser.add_argument("--annotations", required=True, help="CSV annotation file")
    parser.add_argument("--output-dir", required=True, help="Output directory for manifest.csv and label_map.json")
    parser.add_argument("--video-col", default="video_path")
    parser.add_argument("--start-col", default="start")
    parser.add_argument("--end-col", default="end")
    parser.add_argument("--label-col", default="action")
    parser.add_argument("--split-col", default=None, help="Optional existing split column")
    parser.add_argument("--station-col", default=None, help="Optional station column; otherwise generated")
    parser.add_argument("--sop-id", default="sop_assembly101")
    parser.add_argument("--employee-id", default="emp_public")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)

    # Subset controls: important for Assembly101 because the full dataset is large.
    parser.add_argument("--labels", default=None, help="Comma-separated label subset, e.g. take,place,attach")
    parser.add_argument("--max-rows", type=int, default=None, help="Maximum annotation rows after filtering")
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum unique videos after filtering")
    parser.add_argument("--require-existing-videos", action="store_true", help="Drop rows whose resolved video files do not exist locally")
    parser.add_argument("--balanced", action="store_true", help="Sample rows as evenly as possible across labels")
    args = parser.parse_args()

    df = pd.read_csv(args.annotations)
    required = [args.video_col, args.start_col, args.end_col, args.label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Annotation file missing selected columns: {missing}; available columns={list(df.columns)}")

    out = pd.DataFrame()
    out["video_path"] = df[args.video_col].astype(str).apply(lambda x: resolve_video_path(args.videos_root, x))
    out["start"] = df[args.start_col].astype(float)
    out["end"] = df[args.end_col].astype(float)
    out["label"] = df[args.label_col].astype(str)

    if args.split_col and args.split_col in df.columns:
        out["split"] = df[args.split_col].astype(str)
    else:
        out["split"] = "train"

    if args.station_col and args.station_col in df.columns:
        out["station_id"] = df[args.station_col].astype(str)
    else:
        out["station_id"] = "station_public"

    out["sop_id"] = args.sop_id
    out["employee_id"] = args.employee_id
    out = out[out["end"] > out["start"]].reset_index(drop=True)

    labels = parse_csv_list(args.labels)
    if labels:
        out = out[out["label"].isin(labels)].reset_index(drop=True)

    if args.require_existing_videos:
        out = out[out["video_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)

    if args.max_videos is not None:
        unique_videos = out["video_path"].drop_duplicates().sample(
            n=min(args.max_videos, out["video_path"].nunique()), random_state=args.seed
        )
        out = out[out["video_path"].isin(set(unique_videos))].reset_index(drop=True)

    if args.max_rows is not None:
        if args.balanced and out["label"].nunique() > 1:
            per_label = max(1, args.max_rows // out["label"].nunique())
            out = (
                out.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(n=min(per_label, len(g)), random_state=args.seed))
                .reset_index(drop=True)
            )
            if len(out) > args.max_rows:
                out = out.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
        else:
            out = out.sample(n=min(args.max_rows, len(out)), random_state=args.seed).reset_index(drop=True)

    if out.empty:
        raise ValueError("No rows left after filtering. Check labels, paths, max-videos, and require-existing-videos.")

    # If the source did not provide splits, create splits after filtering.
    if out["split"].nunique() == 1 and out["split"].iloc[0] == "train":
        indices = list(range(len(out)))
        test_total = args.val_size + args.test_size
        if len(out) >= 10 and test_total > 0:
            stratify = out["label"] if out["label"].nunique() > 1 and out["label"].value_counts().min() >= 2 else None
            train_idx, temp_idx = train_test_split(indices, test_size=test_total, random_state=args.seed, stratify=stratify)
            relative_test = args.test_size / test_total if test_total > 0 else 0.0
            val_idx, test_idx = train_test_split(temp_idx, test_size=relative_test, random_state=args.seed)
            out["split"] = "train"
            out.loc[val_idx, "split"] = "val"
            out.loc[test_idx, "split"] = "test"

    label_values = sorted(out["label"].unique().tolist())
    label2id = {label: i for i, label in enumerate(label_values)}
    id2label = {i: label for label, i in label2id.items()}

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    label_path = os.path.join(args.output_dir, "label_map.json")
    out.to_csv(manifest_path, index=False)
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote label map: {label_path}")
    print(f"Rows: {len(out)} | Unique videos: {out['video_path'].nunique()} | Labels: {len(label_values)}")
    print(f"Splits: {out['split'].value_counts().to_dict()}")
    print(f"Top labels: {out['label'].value_counts().head(20).to_dict()}")


if __name__ == "__main__":
    main()
