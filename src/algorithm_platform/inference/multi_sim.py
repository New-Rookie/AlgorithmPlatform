from __future__ import annotations

import threading
import time
from typing import List

from src.algorithm_platform.streaming.window import iter_video_windows, batched
from src.algorithm_platform.models.action_recognizer import VideoMAEActionRecognizer
from src.algorithm_platform.streaming.fsm import SOPFSM
from src.algorithm_platform.streaming.events import JsonlEventWriter, RecognitionEvent
from src.algorithm_platform.utils.config import load_config


def run_multi_inference(config_path: str) -> None:
    cfg = load_config(config_path).raw
    runtime = cfg["runtime"]
    streams = cfg["streams"]
    sop = load_config(cfg["sop_config"]).raw

    recognizer = VideoMAEActionRecognizer(cfg["paths"]["model_dir"], device=runtime.get("device", "auto"))

    def worker(stream):
        fsm = SOPFSM(
            steps=sop["steps"],
            threshold=float(sop.get("thresholds", {}).get("confirm_confidence", 0.70)),
            uncertain_threshold=float(sop.get("thresholds", {}).get("uncertain_confidence", 0.55)),
        )

        writer = JsonlEventWriter(cfg["paths"]["event_jsonl"])

        windows = iter_video_windows(
            video_path=stream["video_path"],
            stream_id=stream["stream_id"],
            window_seconds=float(runtime["window_seconds"]),
            stride_seconds=float(runtime["stride_seconds"]),
            num_frames=int(runtime["num_frames"]),
        )

        batch = []
        for w in windows:
            batch.append(w)
            if len(batch) >= int(runtime["batch_size"]):
                preds = recognizer.predict_batch([b.frames for b in batch])
                for w_, p in zip(batch, preds):
                    update = fsm.update(p.label, p.confidence, w_.center)
                    if update.event_type:
                        writer.write(RecognitionEvent(
                            stream_id=stream["stream_id"],
                            station_id=stream["station_id"],
                            employee_id=stream["employee_id"],
                            sop_id=stream["sop_id"],
                            timestamp=w_.center,
                            action=p.label,
                            confidence=p.confidence,
                            fsm_status=update.status,
                            expected_step_id=update.expected_step_id,
                            event_type=update.event_type,
                            message=update.message,
                            evidence_clip=None,
                            metadata=update.metadata,
                        ))
                batch = []

    threads: List[threading.Thread] = []
    for s in streams:
        t = threading.Thread(target=worker, args=(s,), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("Multi-stream inference finished.")
