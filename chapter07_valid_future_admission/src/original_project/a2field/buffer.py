from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .messages import IntentionMessage, MotionTubePoint


@dataclass
class BufferStats:
    active_intention_count: int = 0
    mean_intention_age: float = 0.0
    max_intention_age: float = 0.0
    stale_intention_ratio: float = 0.0
    expired_intention_count: int = 0
    overwrite_count: int = 0


@dataclass
class IntentionBuffer:
    mode: str = "rolling_overwrite"
    freshness_tau: float = 10.0
    ttl: int = 30
    stale_age: int = 10
    pending: list[IntentionMessage] = field(default_factory=list)
    active_append: list[IntentionMessage] = field(default_factory=list)
    active_latest: dict[int, IntentionMessage] = field(default_factory=dict)
    expired_intention_count: int = 0
    overwrite_count: int = 0

    def enqueue_message(self, message: IntentionMessage) -> None:
        self.pending.append(message)

    def deliver_messages(self, current_time: int) -> int:
        delivered = [m for m in self.pending if m.delivery_time <= current_time]
        self.pending = [m for m in self.pending if m.delivery_time > current_time]
        for msg in delivered:
            if self.mode == "append_only":
                self.active_append.append(msg)
            elif self.mode in {"rolling_overwrite", "overwrite_with_decay_TTL"}:
                if msg.agent_id in self.active_latest:
                    self.overwrite_count += 1
                self.active_latest[msg.agent_id] = msg
            else:
                raise ValueError(f"Unknown buffer mode {self.mode!r}")
        return len(delivered)

    def update_buffer(self, current_time: int) -> None:
        if self.mode != "overwrite_with_decay_TTL":
            return
        expired = [aid for aid, msg in self.active_latest.items() if current_time - msg.upload_time > self.ttl]
        for aid in expired:
            del self.active_latest[aid]
        self.expired_intention_count += len(expired)

    def _messages(self) -> list[IntentionMessage]:
        if self.mode == "append_only":
            return list(self.active_append)
        return list(self.active_latest.values())

    def get_active_messages(self) -> list[IntentionMessage]:
        return self._messages()

    def get_active_intentions(self, current_time: int) -> list[MotionTubePoint]:
        points: list[MotionTubePoint] = []
        for msg in self._messages():
            age = max(0, current_time - msg.upload_time)
            factor = 1.0
            if self.mode == "overwrite_with_decay_TTL":
                factor = float(np.exp(-age / max(self.freshness_tau, 1e-12)))
            for p in msg.tube_points:
                points.append(
                    MotionTubePoint(
                        agent_id=p.agent_id,
                        version_id=p.version_id,
                        upload_time=p.upload_time,
                        future_time_abs=p.future_time_abs,
                        k=p.k,
                        y=p.y,
                        x=p.x,
                        dy=p.dy,
                        dx=p.dx,
                        radius=p.radius,
                        freshness=p.freshness * factor,
                        weight=p.weight,
                    )
                )
        return points

    def get_buffer_stats(self, current_time: int) -> BufferStats:
        messages = self._messages()
        if not messages:
            return BufferStats(
                expired_intention_count=self.expired_intention_count,
                overwrite_count=self.overwrite_count,
            )
        ages = np.array([max(0, current_time - msg.upload_time) for msg in messages], dtype=float)
        return BufferStats(
            active_intention_count=len(messages),
            mean_intention_age=float(np.mean(ages)),
            max_intention_age=float(np.max(ages)),
            stale_intention_ratio=float(np.mean(ages > self.stale_age)),
            expired_intention_count=self.expired_intention_count,
            overwrite_count=self.overwrite_count,
        )
