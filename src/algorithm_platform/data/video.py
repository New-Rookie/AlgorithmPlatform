from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    duration: float


def probe_video(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return VideoInfo(path=path, fps=fps, frame_count=frame_count, duration=duration)


def _safe_read_frame(cap: cv2.VideoCapture, frame_idx: int):
    frame_idx = max(0, int(frame_idx))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def sample_video_clip(
    path: str,
    start_sec: float,
    end_sec: float,
    num_frames: int,
) -> List[np.ndarray]:
    """Uniformly sample RGB frames from a time interval.

    The function pads with the last available frame when the segment is too short.
    It intentionally returns a Python list because HuggingFace VideoMAE processors
    accept a list of HWC RGB frames.
    """
    info = probe_video(path)
    if info.frame_count <= 0:
        raise ValueError(f"Video has no readable frames: {path}")

    start_sec = max(0.0, float(start_sec))
    end_sec = max(start_sec + 1.0 / max(info.fps, 1.0), float(end_sec))
    start_frame = int(start_sec * info.fps)
    end_frame = min(int(end_sec * info.fps), max(info.frame_count - 1, 0))

    if end_frame <= start_frame:
        end_frame = min(start_frame + 1, max(info.frame_count - 1, 0))

    indices = np.linspace(start_frame, end_frame, num=num_frames).astype(int).tolist()
    cap = cv2.VideoCapture(path)
    frames: List[np.ndarray] = []
    last = None
    for idx in indices:
        frame = _safe_read_frame(cap, idx)
        if frame is None:
            frame = last
        if frame is not None:
            frames.append(frame)
            last = frame
    cap.release()

    if not frames:
        raise ValueError(f"Failed to sample frames from {path}")

    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]


def save_clip(path: str, output_path: str, start_sec: float, end_sec: float) -> None:
    """Save a subclip using OpenCV without requiring ffmpeg."""
    info = probe_video(path)
    cap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, fourcc, info.fps or 25.0, (width, height))

    start_frame = max(0, int(start_sec * info.fps))
    end_frame = min(info.frame_count - 1, int(end_sec * info.fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(max(0, end_frame - start_frame + 1)):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    writer.release()
    cap.release()
