from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pathing import astar
from .scenarios import Scenario


@dataclass(frozen=True)
class Agent:
    agent_id: int
    start: tuple[int, int]
    goal: tuple[int, int]
    path: list[tuple[int, int]]


@dataclass
class RollingAgent:
    agent_id: int
    start: tuple[int, int]
    goal: tuple[int, int]
    current_path: list[tuple[int, int]]
    path_index: int = 0
    version_id: int = 0
    can_upload: bool = True
    active: bool = True

    @property
    def current_position(self) -> tuple[int, int]:
        if not self.current_path:
            return self.start
        return self.current_path[min(self.path_index, len(self.current_path) - 1)]

    def step(self) -> None:
        if not self.current_path:
            self.active = False
            return
        if self.path_index < len(self.current_path) - 1:
            self.path_index += 1
        else:
            self.active = False

    def get_future_path(self, current_time: int, horizon: int) -> list[tuple[int, int]]:
        del current_time
        if not self.current_path:
            return []
        points: list[tuple[int, int]] = []
        for k in range(1, horizon + 1):
            idx = min(self.path_index + k, len(self.current_path) - 1)
            points.append(self.current_path[idx])
        return points

    def generate_intention_message(
        self,
        current_time: int,
        horizon: int,
        base_radius: float,
        width_mode: str,
        version_id: int,
        width_growth: float = 0.0,
        max_radius: float = 2.0,
        beta: float = 0.03,
        delivery_time: int | None = None,
        policy_name: str = "full_upload",
    ):
        from .messages import IntentionMessage, MotionTubePoint

        points: list[MotionTubePoint] = []
        for k in range(1, horizon + 1):
            idx = min(self.path_index + k, len(self.current_path) - 1)
            y, x = self.current_path[idx]
            nxt = self.current_path[min(idx + 1, len(self.current_path) - 1)]
            radius = base_radius
            if width_mode in {"linear_growth", "capped_growth"}:
                radius = base_radius + max(0.0, width_growth) * k
                if width_mode == "capped_growth":
                    radius = min(radius, max_radius)
            points.append(
                MotionTubePoint(
                    agent_id=self.agent_id,
                    version_id=version_id,
                    upload_time=current_time,
                    future_time_abs=current_time + k,
                    k=k,
                    y=y,
                    x=x,
                    dy=nxt[0] - y,
                    dx=nxt[1] - x,
                    radius=radius,
                    freshness=1.0,
                    weight=float(np.exp(-beta * k)),
                )
            )
        return IntentionMessage(
            agent_id=self.agent_id,
            version_id=version_id,
            upload_time=current_time,
            delivery_time=current_time if delivery_time is None else delivery_time,
            horizon=horizon,
            tube_points=points,
            policy_name=policy_name,
            uploaded_tube_points=len(points),
        )


def generate_agents(
    scenario: Scenario,
    n_agents: int,
    rng: np.random.Generator,
    max_resamples: int = 50,
) -> list[Agent]:
    agents: list[Agent] = []
    for agent_id in range(n_agents):
        path: list[tuple[int, int]] = []
        start = goal = (0, 0)
        for _ in range(max_resamples):
            start = scenario.sample_start(rng)
            goal = scenario.sample_goal(rng)
            if start == goal:
                continue
            path = astar(scenario.grid, start, goal)
            if path:
                break
        if path:
            agents.append(Agent(agent_id, start, goal, path))
    return agents


def generate_rolling_agents(
    scenario: Scenario,
    n_agents: int,
    rng: np.random.Generator,
    max_resamples: int = 50,
) -> list[RollingAgent]:
    static_agents = generate_agents(scenario, n_agents, rng, max_resamples=max_resamples)
    return [
        RollingAgent(
            agent_id=a.agent_id,
            start=a.start,
            goal=a.goal,
            current_path=a.path,
        )
        for a in static_agents
    ]
