from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FSMUpdate:
    status: str
    current_step_id: Optional[str]
    expected_step_id: Optional[str]
    action: str
    confidence: float
    timestamp: float
    message: str
    event_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SOPFSM:
    """Small online SOP state machine.

    This is intentionally rule-based for V1. It validates model outputs against a
    pseudo-SOP sequence and emits structured statuses that can be stored or sent
    to a later LLM/reporting layer.
    """

    def __init__(self, steps: List[Dict[str, Any]], threshold: float = 0.70, uncertain_threshold: float = 0.55):
        self.steps = steps
        self.threshold = threshold
        self.uncertain_threshold = uncertain_threshold
        self.index = 0
        self.history: List[Dict[str, Any]] = []

    def current_step(self) -> Optional[Dict[str, Any]]:
        return self.steps[self.index] if self.index < len(self.steps) else None

    def reset(self) -> None:
        self.index = 0
        self.history.clear()

    def update(self, action: str, confidence: float, timestamp: float) -> FSMUpdate:
        step = self.current_step()
        if step is None:
            return FSMUpdate(
                status="FINISHED",
                current_step_id=None,
                expected_step_id=None,
                action=action,
                confidence=confidence,
                timestamp=timestamp,
                message="SOP already finished",
            )

        expected_actions = [str(a).lower() for a in step.get("actions", [])]
        action_norm = str(action).lower()
        step_id = step.get("id")

        if confidence < self.uncertain_threshold:
            return FSMUpdate(
                status="UNCERTAIN",
                current_step_id=step_id,
                expected_step_id=step_id,
                action=action,
                confidence=confidence,
                timestamp=timestamp,
                message="Model confidence below uncertain threshold",
                event_type="low_confidence",
            )

        if action_norm in expected_actions and confidence >= self.threshold:
            self.history.append({"timestamp": timestamp, "action": action, "step_id": step_id, "confidence": confidence})
            self.index += 1
            return FSMUpdate(
                status="OK",
                current_step_id=step_id,
                expected_step_id=step_id,
                action=action,
                confidence=confidence,
                timestamp=timestamp,
                message=f"Matched SOP step {step_id}",
            )

        # If the action matches a future step, emit possible skip.
        future_match = None
        for future_idx in range(self.index + 1, len(self.steps)):
            future_actions = [str(a).lower() for a in self.steps[future_idx].get("actions", [])]
            if action_norm in future_actions:
                future_match = self.steps[future_idx]
                break

        if future_match is not None:
            return FSMUpdate(
                status="DEVIATION",
                current_step_id=step_id,
                expected_step_id=step_id,
                action=action,
                confidence=confidence,
                timestamp=timestamp,
                message=f"Action appears to skip from {step_id} to {future_match.get('id')}",
                event_type="possible_skip",
                metadata={"future_step_id": future_match.get("id")},
            )

        return FSMUpdate(
            status="DEVIATION",
            current_step_id=step_id,
            expected_step_id=step_id,
            action=action,
            confidence=confidence,
            timestamp=timestamp,
            message=f"Action does not match expected step {step_id}",
            event_type="unexpected_action",
        )
