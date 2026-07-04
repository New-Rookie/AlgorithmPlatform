from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split

REPO_ID = "cvml-nus/assembly101"

VIEW_FILES = {
    "v1": ["C10095_rgb.mp4"],
    "v2": ["C10115_rgb.mp4"],
    "v3": ["C10118_rgb.mp4"],
    "v4": ["C10119_rgb.mp4"],
    "v5": ["C10379_rgb.mp4"],
    "v6": ["C10390_rgb.mp4"],
    "v7": ["C10395_rgb.mp4"],
    "v8": ["C10404_rgb.mp4"],
}
VIEW_FILES["fixed"] = sorted({x for k, v in VIEW_FILES.items() for x in v if k.startswith("v")})
VIEW_FILES["all"] = VIEW_FILES["fixed"]

VIDEO_COL_CANDIDATES = ["video_path", "video", "video_name", "recording", "recording_name", "path", "filename"]
START_COL_CANDIDATES = ["start", "start_time", "start_sec", "start_frame", "t_start"]
END_COL_CANDIDATES = ["end", "end_time", "end_sec", "end_frame", "t_end"]
LABEL_COL_CANDIDATES = ["action", "label", "action_label", "verb", "coarse_label", "fine_label", "class"]


def read_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def find_first_file(root: Path, suffixes=(".csv", ".txt")) -> Optional[Path]:
    files = []
    for suffix in suffixes:
        files.extend(root.rglob(f"*{suffix}"))
    files = [p for p in files if p.is_file()]
    if not files:
        return None
    csv_files = [p for p in files if p.suffix.lower() == ".csv"]
    return sorted(csv_files or files, key=lambda p: len(str(p)))[0]


def infer_column(columns: List[str], candidates: List[str], role: str) -> str:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for c in columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    raise ValueError(f"Cannot infer {role} column. Available columns: {columns}")


def normalize_recording(value: str) -> str:
    parts = Path(str(value)).parts
    if "recordings" in parts:
        idx = parts.index("recordings")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if len(parts) >= 2:
        return parts[-2]
    return Path(str(value)).stem


def resolve_video_path(videos_root: str, recording: str, view_file: str) -> str:
    return str(Path(videos_root) / "recordings" / recording / view_file)


def expand_views(views: str) -> List[str]:
    result: List[str] = []
    for v in views.split(","):
        v = v.strip()
        if not v:
            continue
        if v in VIEW_FILES:
            result.extend(VIEW_FILES[v])
        elif v.endswith(".mp4"):
            result.append(v)
        else:
            raise ValueError(f"Unknown view {v!r}. Use v1-v8, fixed, all, or exact mp4 filename.")
    return sorted(set(result))


def download_annotations(local_dir: str) -> Path:
    print("[1/5] Downloading annotations only...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=["annotations/*", "annotations/**/*"],
        resume_download=True,
    )
    ann_root = Path(local_dir) / "annotations"
    annotation_file = find_first_file(ann_root, suffixes=(".csv",))
    if annotation_file is None:
        raise FileNotFoundError(f"No CSV annotation file found under {ann_root}")
    print(f"Found annotation file: {annotation_file}")
    return annotation_file


def select_subset(
    annotation_file: Path,
    num_labels: int,
    max_recordings: int,
    min_clips_per_recording: int,
    preferred_labels: Optional[List[str]],
) -> Tuple[pd.DataFrame, Dict[str, str], List[str], List[str]]:
    print("[2/5] Inspecting annotations and selecting subset...")
    df = pd.read_csv(annotation_file)
    columns = list(df.columns)
    video_col = infer_column(columns, VIDEO_COL_CANDIDATES, "video")
    start_col = infer_column(columns, START_COL_CANDIDATES, "start")
    end_col = infer_column(columns, END_COL_CANDIDATES, "end")
    label_col = infer_column(columns, LABEL_COL_CANDIDATES, "label")

    df = df.copy()
    df[label_col] = df[label_col].astype(str)
    if preferred_labels:
        labels = [l for l in preferred_labels if l in set(df[label_col])]
        if not labels:
            raise ValueError(f"None of preferred labels exist in annotation file: {preferred_labels}")
    else:
        label_counts = df[label_col].value_counts()
        labels = label_counts.head(num_labels).index.astype(str).tolist()

    filtered = df[df[label_col].isin(labels)].copy()
    filtered["recording_name"] = filtered[video_col].astype(str).apply(normalize_recording)

    rec_summary = (
        filtered.groupby("recording_name")
        .agg(clips=(label_col, "size"), labels=(label_col, "nunique"))
        .reset_index()
    )
    rec_summary = rec_summary[rec_summary["clips"] >= min_clips_per_recording]
    rec_summary = rec_summary.sort_values(["labels", "clips"], ascending=False).head(max_recordings)
    recordings = rec_summary["recording_name"].astype(str).tolist()
    if not recordings:
        raise ValueError("No recordings selected. Reduce min_clips_per_recording or check selected labels.")

    selected = filtered[filtered["recording_name"].isin(recordings)].reset_index(drop=True)
    col_map = {"video": video_col, "start": start_col, "end": end_col, "label": label_col}
    print(f"Selected labels: {labels}")
    print(f"Selected recordings: {len(recordings)}")
    print(f"Selected annotation rows before video filtering: {len(selected)}")
    return selected, col_map, labels, recordings


def download_videos(local_dir: str, recordings: List[str], view_files: List[str], dry_run: bool) -> None:
    print("[3/5] Downloading selected recording videos...")
    patterns = []
    for recording in recordings:
        for vf in view_files:
            patterns.append(f"recordings/{recording}/{vf}")
    print(f"Views: {view_files}")
    print(f"File patterns: {len(patterns)}")
    for p in patterns[:20]:
        print("  ", p)
    if dry_run:
        print("Dry-run mode: videos not downloaded.")
        return
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=patterns,
        resume_download=True,
    )


def build_manifest(
    selected: pd.DataFrame,
    col_map: Dict[str, str],
    labels: List[str],
    recordings: List[str],
    view_file: str,
    videos_root: str,
    output_dir: str,
    max_rows: int,
    balanced: bool,
    seed: int,
) -> None:
    print("[4/5] Building manifest.csv...")
    out = pd.DataFrame()
    out["recording_name"] = selected["recording_name"].astype(str)
    out["video_path"] = out["recording_name"].apply(lambda r: resolve_video_path(videos_root, r, view_file))
    out["start"] = selected[col_map["start"]].astype(float)
    out["end"] = selected[col_map["end"]].astype(float)
    out["label"] = selected[col_map["label"]].astype(str)
    out["station_id"] = "station_public"
    out["sop_id"] = "sop_assembly101"
    out["employee_id"] = "emp_public"

    out = out[(out["end"] > out["start"]) & (out["video_path"].apply(lambda p: Path(p).exists()))].reset_index(drop=True)
    if out.empty:
        raise ValueError("No manifest rows left after checking downloaded video files. Check recording names and view selection.")

    if max_rows and len(out) > max_rows:
        if balanced and out["label"].nunique() > 1:
            per_label = max(1, max_rows // out["label"].nunique())
            out = (
                out.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(n=min(per_label, len(g)), random_state=seed))
                .reset_index(drop=True)
            )
            if len(out) > max_rows:
                out = out.sample(n=max_rows, random_state=seed).reset_index(drop=True)
        else:
            out = out.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    indices = list(range(len(out)))
    if len(out) >= 10:
        stratify = out["label"] if out["label"].value_counts().min() >= 2 else None
        train_idx, temp_idx = train_test_split(indices, test_size=0.25, random_state=seed, stratify=stratify)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.4, random_state=seed)
        out["split"] = "train"
        out.loc[val_idx, "split"] = "val"
        out.loc[test_idx, "split"] = "test"
    else:
        out["split"] = "train"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(output_dir) / "manifest.csv"
    label_map_path = Path(output_dir) / "label_map.json"
    summary_path = Path("outputs/dataset/bootstrap_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    labels_sorted = sorted(out["label"].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels_sorted)}
    id2label = {idx: label for label, idx in label2id.items()}
    out.to_csv(manifest_path, index=False)
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "rows": len(out),
                "videos": out["video_path"].nunique(),
                "labels": labels_sorted,
                "splits": out["split"].value_counts().to_dict(),
                "recordings": recordings,
                "view_file": view_file,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Manifest: {manifest_path}")
    print(f"Label map: {label_map_path}")
    print(f"Summary: {summary_path}")
    print(f"Rows: {len(out)} | Videos: {out['video_path'].nunique()} | Labels: {labels_sorted}")
    print(f"Splits: {out['split'].value_counts().to_dict()}")


def patch_dataset_config(config_path: str, annotation_file: Path, labels: List[str]) -> None:
    print("[5/5] Updating dataset config with detected annotation path and selected labels...")
    cfg = read_yaml(config_path)
    cfg["paths"]["annotations"] = str(annotation_file)
    cfg["subset"]["labels"] = labels
    write_yaml(config_path, cfg)
    print(f"Updated: {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command Assembly101 subset bootstrap: annotations -> selected videos -> manifest")
    parser.add_argument("--config", default="configs/dataset_assembly101.yaml")
    parser.add_argument("--local-dir", default="data/raw/assembly101")
    parser.add_argument("--output-dir", default="data/processed/assembly101")
    parser.add_argument("--views", default="v1", help="v1-v8, fixed, all, or exact mp4 filename; start with v1")
    parser.add_argument("--num-labels", type=int, default=6, help="Auto-select top N labels when --labels is not provided")
    parser.add_argument("--labels", default=None, help="Optional comma-separated labels to force")
    parser.add_argument("--max-recordings", type=int, default=80)
    parser.add_argument("--max-rows", type=int, default=12000)
    parser.add_argument("--min-clips-per-recording", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Download annotations, select subset, print video patterns, but do not download videos or build manifest")
    args = parser.parse_args()

    view_files = expand_views(args.views)
    if len(view_files) != 1:
        print("Warning: manifest will use the first view file only. Multiple-view training is not implemented in V1.")
    manifest_view_file = view_files[0]

    annotation_file = download_annotations(args.local_dir)
    preferred_labels = [x.strip() for x in args.labels.split(",") if x.strip()] if args.labels else None
    selected, col_map, labels, recordings = select_subset(
        annotation_file=annotation_file,
        num_labels=args.num_labels,
        max_recordings=args.max_recordings,
        min_clips_per_recording=args.min_clips_per_recording,
        preferred_labels=preferred_labels,
    )
    download_videos(args.local_dir, recordings, view_files, dry_run=args.dry_run)
    if args.dry_run:
        return
    build_manifest(
        selected=selected,
        col_map=col_map,
        labels=labels,
        recordings=recordings,
        view_file=manifest_view_file,
        videos_root=args.local_dir,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        balanced=True,
        seed=args.seed,
    )
    patch_dataset_config(args.config, annotation_file, labels)
    print("Done. Next: python scripts/train.py --config configs/train.yaml")


if __name__ == "__main__":
    main()
