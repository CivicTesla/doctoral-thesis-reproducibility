from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotionTubePoint:
    agent_id: int
    version_id: int
    upload_time: int
    future_time_abs: int
    k: int
    y: int
    x: int
    dy: int
    dx: int
    radius: float
    freshness: float
    weight: float


@dataclass(frozen=True)
class IntentionMessage:
    agent_id: int
    version_id: int
    upload_time: int
    delivery_time: int
    horizon: int
    tube_points: list[MotionTubePoint]
    policy_name: str
    uploaded_tube_points: int
