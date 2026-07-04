from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Assembly101-like annotation CSV before preparing a training subset")
    parser.add_argument("--annotations", required=True, help="CSV annotation file")
    parser.add_argument("--video-col", default="video_path")
    parser.add_argument("--label-col", default="action")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--output", default="outputs/dataset/annotation_summary.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.annotations)
    for col in [args.video_col, args.label_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found. Available columns: {list(df.columns)}")

    summary = (
        df.groupby(args.label_col)
        .agg(
            clips=(args.label_col, "size"),
            videos=(args.video_col, "nunique"),
        )
        .sort_values(["clips", "videos"], ascending=False)
        .reset_index()
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)

    print(f"Rows: {len(df)}")
    print(f"Unique videos: {df[args.video_col].nunique()}")
    print(f"Unique labels: {df[args.label_col].nunique()}")
    print(f"Wrote summary: {args.output}")
    print("\nTop labels:")
    print(summary.head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
