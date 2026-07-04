from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

from src.algorithm_platform.data.video import probe_video, sample_video_clip


@dataclass
class VideoWindow:
    stream_id: str
    video_path: str
    start: float
    end: float
    center: float
    frames: list


def iter_video_windows(
    video_path: str,
    stream_id: str,
    window_seconds: float,
    stride_seconds: float,
    num_frames: int,
) -> Iterator[VideoWindow]:
    info = probe_video(video_path)
    t = 0.0
    while t + window_seconds <= info.duration:
        start = t
        end = t + window_seconds
        frames = sample_video_clip(video_path, start, end, num_frames)
        yield VideoWindow(
            stream_id=stream_id,
            video_path=video_path,
            start=start,
            end=end,
            center=(start + end) / 2.0,
            frames=frames,
        )
        t += stride_seconds


def batched(items: List, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
