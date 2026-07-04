from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RecognitionEvent:
    stream_id: str
    station_id: str
    employee_id: str
    sop_id: str
    timestamp: float
    action: str
    confidence: float
    fsm_status: str
    expected_step_id: Optional[str]
    event_type: Optional[str]
    message: str
    evidence_clip: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JsonlEventWriter:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def write(self, event: RecognitionEvent) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
