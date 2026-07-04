from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.algorithm_platform.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dataset annotation distribution from config")
    parser.add_argument("--config", required=True, help="Path to dataset yaml config")
    args = parser.parse_args()

    cfg = load_config(args.config).raw
    annotations = cfg["paths"]["annotations"]
    video_col = cfg["columns"]["video"]
    label_col = cfg["columns"]["label"]
    output = cfg.get("outputs", {}).get("annotation_summary", "outputs/dataset/annotation_summary.csv")

    df = pd.read_csv(annotations)
    for col in [video_col, label_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found. Available columns: {list(df.columns)}")

    summary = (
        df.groupby(label_col)
        .agg(clips=(label_col, "size"), videos=(video_col, "nunique"))
        .sort_values(["clips", "videos"], ascending=False)
        .reset_index()
    )

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)

    print(f"Rows: {len(df)}")
    print(f"Unique videos: {df[video_col].nunique()}")
    print(f"Unique labels: {df[label_col].nunique()}")
    print(f"Wrote summary: {output}")
    print("\nTop labels:")
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
