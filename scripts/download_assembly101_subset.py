from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Please install huggingface_hub first: pip install huggingface_hub") from exc

REPO_ID = "cvml-nus/assembly101"

FIXED_VIEW_FILES = [
    "C10095_rgb.mp4",  # v1
    "C10115_rgb.mp4",  # v2
    "C10118_rgb.mp4",  # v3
    "C10119_rgb.mp4",  # v4
    "C10379_rgb.mp4",  # v5
    "C10390_rgb.mp4",  # v6
    "C10395_rgb.mp4",  # v7
    "C10404_rgb.mp4",  # v8
]

EGO_VIEW_FILES = [
    "HMC_21176875_mono10bit.mp4",
    "HMC_21176623_mono10bit.mp4",
    "HMC_21110305_mono10bit.mp4",
    "HMC_21179183_mono10bit.mp4",
    "HMC_84346135_mono10bit.mp4",
    "HMC_84347414_mono10bit.mp4",
    "HMC_84355350_mono10bit.mp4",
    "HMC_84358933_mono10bit.mp4",
]

VIEW_ALIASES = {
    "v1": ["C10095_rgb.mp4"],
    "v2": ["C10115_rgb.mp4"],
    "v3": ["C10118_rgb.mp4"],
    "v4": ["C10119_rgb.mp4"],
    "v5": ["C10379_rgb.mp4"],
    "v6": ["C10390_rgb.mp4"],
    "v7": ["C10395_rgb.mp4"],
    "v8": ["C10404_rgb.mp4"],
    "fixed": FIXED_VIEW_FILES,
    "ego": EGO_VIEW_FILES,
    "egocentric": EGO_VIEW_FILES,
    "all": FIXED_VIEW_FILES + EGO_VIEW_FILES,
}


def read_recordings(path: str) -> List[str]:
    recordings = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                recordings.append(line)
    if not recordings:
        raise ValueError(f"No recording names found in {path}")
    return recordings


def expand_views(views: Iterable[str]) -> List[str]:
    files: List[str] = []
    for view in views:
        view = view.strip()
        if not view:
            continue
        if view in VIEW_ALIASES:
            files.extend(VIEW_ALIASES[view])
        elif view.endswith(".mp4"):
            files.append(view)
        else:
            raise ValueError(f"Unknown view {view!r}. Use v1-v8, fixed, ego, all, or an exact mp4 filename.")
    return sorted(set(files))


def build_patterns(recordings: List[str], view_files: List[str], include_annotations: bool) -> List[str]:
    patterns: List[str] = []
    if include_annotations:
        patterns.extend(["annotations/*", "annotations/**/*"])
    for recording in recordings:
        for view_file in view_files:
            patterns.append(f"recordings/{recording}/{view_file}")
    return patterns


def main() -> None:
    parser = argparse.ArgumentParser(description="Partially download Assembly101 recordings from Hugging Face")
    parser.add_argument("--recordings-file", required=True, help="Text file with one recording folder name per line")
    parser.add_argument("--local-dir", default="data/raw/assembly101", help="Local output directory")
    parser.add_argument("--views", default="v1", help="Comma-separated views: v1-v8, fixed, ego, all, or exact mp4 filenames")
    parser.add_argument("--include-annotations", action="store_true", help="Also download annotations folder")
    parser.add_argument("--dry-run", action="store_true", help="Only print allow_patterns; do not download")
    args = parser.parse_args()

    recordings = read_recordings(args.recordings_file)
    view_files = expand_views(args.views.split(","))
    allow_patterns = build_patterns(recordings, view_files, include_annotations=args.include_annotations)

    print(f"Repo: {REPO_ID}")
    print(f"Recordings: {len(recordings)}")
    print(f"Views/files per recording: {view_files}")
    print(f"Total file patterns: {len(allow_patterns)}")
    print("\nFirst patterns:")
    for p in allow_patterns[:20]:
        print("  ", p)

    if args.dry_run:
        print("\nDry run only. No files downloaded.")
        return

    Path(args.local_dir).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=args.local_dir,
        allow_patterns=allow_patterns,
        resume_download=True,
    )
    print(f"Downloaded selected subset to: {args.local_dir}")


if __name__ == "__main__":
    main()
