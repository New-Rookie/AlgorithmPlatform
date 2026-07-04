from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from src.algorithm_platform.data.manifest import load_label_maps


@dataclass
class ActionPrediction:
    label: str
    confidence: float
    probabilities: dict


class VideoMAEActionRecognizer:
    def __init__(self, model_dir: str, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.processor = VideoMAEImageProcessor.from_pretrained(model_dir)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_dir).to(device)
        self.model.eval()
        self.label_maps = load_label_maps(model_dir)

    @torch.no_grad()
    def predict_batch(self, clips: List[List[np.ndarray]]) -> List[ActionPrediction]:
        if not clips:
            return []
        inputs = self.processor(clips, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        logits = self.model(pixel_values=pixel_values).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        results: List[ActionPrediction] = []
        for row in probs:
            idx = int(row.argmax())
            label = self.label_maps.id2label.get(idx, str(idx))
            results.append(
                ActionPrediction(
                    label=label,
                    confidence=float(row[idx]),
                    probabilities={self.label_maps.id2label.get(i, str(i)): float(v) for i, v in enumerate(row)},
                )
            )
        return results
