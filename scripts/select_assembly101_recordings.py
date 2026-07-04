from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Select Assembly101 recording names before downloading a partial subset")
    parser.add_argument("--annotations", required=True, help="Annotation CSV")
    parser.add_argument("--video-col", default="video_path", help="Column containing recording/video path/name")
    parser.add_argument("--label-col", default="action", help="Action label column")
    parser.add_argument("--labels", default=None, help="Optional comma-separated label subset")
    parser.add_argument("--max-recordings", type=int, default=80)
    parser.add_argument("--min-clips-per-recording", type=int, default=10)
    parser.add_argument("--output", default="outputs/dataset/selected_recordings.txt")
    parser.add_argument("--summary-output", default="outputs/dataset/selected_recordings_summary.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.annotations)
    for col in [args.video_col, args.label_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found. Available columns: {list(df.columns)}")

    if args.labels:
        labels = [x.strip() for x in args.labels.split(",") if x.strip()]
        df = df[df[args.label_col].astype(str).isin(labels)].copy()

    # Normalize a path such as recordings/<recording>/C10095_rgb.mp4 or <recording>/C10095_rgb.mp4
    # into the recording folder name.
    def to_recording(value: str) -> str:
        parts = Path(str(value)).parts
        if "recordings" in parts:
            idx = parts.index("recordings")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        if len(parts) >= 2:
            return parts[-2]
        return Path(str(value)).stem

    df["recording_name"] = df[args.video_col].astype(str).apply(to_recording)
    summary = (
        df.groupby("recording_name")
        .agg(clips=(args.label_col, "size"), labels=(args.label_col, "nunique"))
        .reset_index()
    )
    summary = summary[summary["clips"] >= args.min_clips_per_recording]
    summary = summary.sort_values(["labels", "clips"], ascending=False).head(args.max_recordings)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary["recording_name"].to_csv(args.output, index=False, header=False)
    summary.to_csv(args.summary_output, index=False)

    print(f"Selected recordings: {len(summary)}")
    print(f"Wrote: {args.output}")
    print(f"Wrote summary: {args.summary_output}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
