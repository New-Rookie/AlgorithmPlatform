from __future__ import annotations

import csv
import os
from typing import Dict, Any

from src.algorithm_platform.data.video import save_clip
from src.algorithm_platform.models.action_recognizer import VideoMAEActionRecognizer
from src.algorithm_platform.streaming.events import JsonlEventWriter, RecognitionEvent
from src.algorithm_platform.streaming.fsm import SOPFSM
from src.algorithm_platform.streaming.window import iter_video_windows, batched
from src.algorithm_platform.utils.config import load_config


def _load_sop(path: str) -> Dict[str, Any]:
    return load_config(path).raw


def run_single_inference(config_path: str) -> None:
    cfg = load_config(config_path).raw
    runtime = cfg["runtime"]
    paths = cfg["paths"]
    station = cfg["station"]
    sop = _load_sop(cfg["sop_config"])

    os.makedirs(os.path.dirname(paths["output_csv"]), exist_ok=True)
    os.makedirs(os.path.dirname(paths["event_jsonl"]), exist_ok=True)
    os.makedirs(paths["clip_dir"], exist_ok=True)

    recognizer = VideoMAEActionRecognizer(paths["model_dir"], device=runtime.get("device", "auto"))
    fsm = SOPFSM(
        steps=sop["steps"],
        threshold=float(sop.get("thresholds", {}).get("confirm_confidence", 0.70)),
        uncertain_threshold=float(sop.get("thresholds", {}).get("uncertain_confidence", 0.55)),
    )
    event_writer = JsonlEventWriter(paths["event_jsonl"])

    windows = list(iter_video_windows(
        video_path=paths["video_path"],
        stream_id="single_stream",
        window_seconds=float(runtime["window_seconds"]),
        stride_seconds=float(runtime["stride_seconds"]),
        num_frames=int(runtime["num_frames"]),
    ))

    with open(paths["output_csv"], "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "stream_id", "station_id", "employee_id", "sop_id", "start", "end", "center",
            "action", "confidence", "fsm_status", "expected_step_id", "event_type", "message"
        ])
        writer.writeheader()
        for batch in batched(windows, int(runtime.get("batch_size", 4))):
            preds = recognizer.predict_batch([w.frames for w in batch])
            for w, pred in zip(batch, preds):
                update = fsm.update(pred.label, pred.confidence, w.center)
                row = {
                    "stream_id": w.stream_id,
                    "station_id": station["station_id"],
                    "employee_id": station["employee_id"],
                    "sop_id": station["sop_id"],
                    "start": w.start,
                    "end": w.end,
                    "center": w.center,
                    "action": pred.label,
                    "confidence": pred.confidence,
                    "fsm_status": update.status,
                    "expected_step_id": update.expected_step_id,
                    "event_type": update.event_type,
                    "message": update.message,
                }
                writer.writerow(row)

                if update.event_type:
                    clip_path = None
                    if runtime.get("save_event_clips", True):
                        clip_path = os.path.join(paths["clip_dir"], f"single_{int(w.center * 1000)}_{update.event_type}.mp4")
                        save_clip(
                            paths["video_path"],
                            clip_path,
                            max(0.0, w.center - float(runtime.get("pre_event_seconds", 10))),
                            w.center + float(runtime.get("post_event_seconds", 10)),
                        )
                    event_writer.write(RecognitionEvent(
                        stream_id=w.stream_id,
                        station_id=station["station_id"],
                        employee_id=station["employee_id"],
                        sop_id=station["sop_id"],
                        timestamp=w.center,
                        action=pred.label,
                        confidence=pred.confidence,
                        fsm_status=update.status,
                        expected_step_id=update.expected_step_id,
                        event_type=update.event_type,
                        message=update.message,
                        evidence_clip=clip_path,
                        metadata=update.metadata,
                    ))

    print(f"Single inference finished. Predictions: {paths['output_csv']} | Events: {paths['event_jsonl']}")
