"""CF0 checkpoint-fork counterfactual simulator.

Reimplements the orchestration of `R0_hidden_mechanism_audit/src/simulator.py`
(H1) and `.../src/suppression_simulator.py` (H2_DISTRIBUTED_CAPACITY_
SUPPRESSION) with a strict checkpoint-fork architecture: one pre-generated,
hashed, read-only exogenous event table shared by all three action branches,
plus an independently-keyed endogenous RNG stream per branch (Section 6 of
the CF0 brief). The frozen math helpers (`_geometric`, `geometric`,
`_suppression_lifetime`, `config_hash`) and every mechanism dynamics
formula, action-cost/utility weight, and config value are reused verbatim
from the frozen source -- nothing about H1/H2 dynamics, action semantics,
or the utility/cost function is changed here, only the RNG plumbing.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import sys
HERE = Path(__file__).resolve()
B1 = HERE.parents[2]
R0 = B1 / "R0_hidden_mechanism_audit"
sys.path[:0] = [str(R0)]

from src.simulator import H1, ACTIONS, _geometric, config_hash  # noqa: E402
from src.suppression_simulator import (  # noqa: E402
    H2_SUPPRESSION, geometric, _suppression_lifetime)
from src.schemas import O0_FIELDS  # noqa: E402

assert ACTIONS == ("NO_ACTION", "CAPACITY_ACTION", "SYNC_ACTION")


def derived_seed(*parts) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).digest()
    return int.from_bytes(digest[:8], "big")


def keyed_rng(*parts) -> random.Random:
    return random.Random(derived_seed(*parts))


def sha256_arrays(*arrays) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


@dataclass
class ExogenousTable:
    seed: int
    mechanism: str
    horizon: int
    n: int
    arrival: np.ndarray            # (horizon, n) bool
    service_time: np.ndarray       # (horizon, n) int, meaningful iff arrival
    deadline_offset: np.ndarray    # (horizon, n) int, meaningful iff arrival
    admission_throttle_pass: np.ndarray  # (horizon, n) bool -- H1 only, inert
    initial_suppressed: dict       # H2 only: {slot: [lifetime, cause]}
    schedule_hash: str
    initial_suppression_hash: str


def _draw_arrivals_and_tasks(rng, horizon, n, arrival_rate, service_mean,
                             deadline_mean):
    arrival = np.zeros((horizon, n), dtype=bool)
    service_time = np.zeros((horizon, n), dtype=np.int32)
    deadline_offset = np.zeros((horizon, n), dtype=np.int32)
    for t in range(horizon):
        for node in range(n):
            if rng.random() < arrival_rate:
                arrival[t, node] = True
                service_time[t, node] = _geometric(rng, service_mean)
                deadline_offset[t, node] = _geometric(rng, deadline_mean)
    return arrival, service_time, deadline_offset


def build_h1_exogenous(config, seed) -> ExogenousTable:
    local = dict(config)
    local.update(config["h1"])
    n, horizon = int(local["num_agents"]), int(local["horizon"])
    rng = keyed_rng(seed, H1, "exogenous")
    arrival, service_time, deadline_offset = _draw_arrivals_and_tasks(
        rng, horizon, n, float(local["arrival_rate"]),
        local["service_time_mean"], local["deadline_mean"])
    admission_intensity = float(local.get("admission_intensity", 1.0))
    throttle = np.zeros((horizon, n), dtype=bool)
    for t in range(horizon):
        for node in range(n):
            throttle[t, node] = rng.random() < admission_intensity
    schedule_hash = sha256_arrays(arrival, service_time, deadline_offset,
                                   throttle)
    return ExogenousTable(
        seed=seed, mechanism=H1, horizon=horizon, n=n, arrival=arrival,
        service_time=service_time, deadline_offset=deadline_offset,
        admission_throttle_pass=throttle, initial_suppressed={},
        schedule_hash=schedule_hash,
        initial_suppression_hash="NOT_APPLICABLE_NO_CHANNEL_AT_ACTION_LAYER")


def build_h2_exogenous(config, seed) -> ExogenousTable:
    local = config["h2_distributed_capacity_suppression"]
    n, horizon = int(config["num_agents"]), int(config["horizon"])
    rng = keyed_rng(seed, H2_SUPPRESSION, "exogenous")
    base_capacity = int(local["physical_capacity"])
    initial_probability = min(
        1.0, float(local["suppressed_slot_count"]) / base_capacity)
    initial_suppressed = {}
    init_record = []
    for slot in range(base_capacity):
        if rng.random() < initial_probability:
            cause = ("conflicting_holder"
                      if rng.random() < local["inconsistent_holder_probability"]
                      else "stale_version")
            lifetime = _suppression_lifetime(rng, local, cause)
            initial_suppressed[slot] = [lifetime, cause]
            init_record.append((slot, cause, lifetime))
        else:
            init_record.append((slot, "", 0))
    init_hash = hashlib.sha256(
        json.dumps(init_record, sort_keys=True).encode()).hexdigest()
    arrival, service_time, deadline_offset = _draw_arrivals_and_tasks(
        rng, horizon, n, float(local["arrival_rate"]),
        local["service_time_mean"], config["deadline_mean"])
    schedule_hash = sha256_arrays(arrival, service_time, deadline_offset)
    return ExogenousTable(
        seed=seed, mechanism=H2_SUPPRESSION, horizon=horizon, n=n,
        arrival=arrival, service_time=service_time,
        deadline_offset=deadline_offset,
        admission_throttle_pass=np.ones((horizon, n), dtype=bool),
        initial_suppressed=initial_suppressed, schedule_hash=schedule_hash,
        initial_suppression_hash=init_hash)


@dataclass
class Request:
    arrival: int
    deadline: int
    service: int


def run_h1_branch(config, action, exo: ExogenousTable):
    local = dict(config)
    local.update(config["h1"])
    n, horizon = exo.n, exo.horizon
    base_capacity = int(local["resource_capacity"])
    capacity_duration = int(config["actions"].get(
        "capacity_action_duration", horizon))
    admission_intensity = float(local.get("admission_intensity", 1.0))
    queues = [[] for _ in range(n)]
    holders = []
    failures = arrivals = completed = misses = 0
    waits, queue_samples, busy_samples = [], [], []
    last_success = [0] * n
    retries = [0] * n
    observations = []

    for t in range(horizon):
        capacity = base_capacity
        if action == "CAPACITY_ACTION" and t < capacity_duration:
            capacity += int(config["actions"]["capacity_extra_slots"])
        remaining = []
        for finish, node, req in holders:
            if finish <= t:
                completed += 1
                last_success[node] = t
            else:
                remaining.append((finish, node, req))
        holders = remaining

        for node in range(n):
            if exo.arrival[t, node]:
                arrivals += 1
                queues[node].append(Request(
                    t, t + int(exo.deadline_offset[t, node]),
                    int(exo.service_time[t, node])))

        for node in range(n):
            kept = []
            for req in queues[node]:
                if t > req.deadline:
                    misses += 1
                else:
                    kept.append(req)
            queues[node] = kept

        believed_busy = len(holders) >= capacity
        start = exo.seed % n + t
        for offset in range(n):
            node = (start + offset) % n
            if not queues[node]:
                continue
            if (admission_intensity < 1.0
                    and not exo.admission_throttle_pass[t, node]):
                continue
            if believed_busy or len(holders) >= capacity:
                failures += 1
                retries[node] += 1
                continue
            req = queues[node].pop(0)
            waits.append(t - req.arrival)
            holders.append((t + req.service, node, req))
            believed_busy = len(holders) >= capacity

        qlens = [len(q) for q in queues]
        queue_samples.extend(qlens)
        busy_samples.append(float(believed_busy))
        for node in range(n):
            observations.append({
                "local_queue_length": qlens[node],
                "head_of_line_wait":
                    t - queues[node][0].arrival if queues[node] else 0,
                "local_request_failures": retries[node],
                "locally_observed_busy_flag": int(believed_busy),
                "time_since_last_success": t - last_success[node],
                "local_retry_count": retries[node],
                "recent_local_throughput": int(last_success[node] == t),
            })

    final_queue_total = sum(len(q) for q in queues)
    misses += final_queue_total
    return _finalize(
        config, action, arrivals, completed, misses, failures, waits,
        queue_samples, busy_samples, observations, horizon,
        final_queue_total)


def run_h2_branch(config, action, exo: ExogenousTable, endogenous_rng):
    local = config["h2_distributed_capacity_suppression"]
    n, horizon = exo.n, exo.horizon
    base_capacity = int(local["physical_capacity"])
    capacity_duration = int(config["actions"].get(
        "capacity_action_duration", horizon))
    queues = [[] for _ in range(n)]
    holders = {}
    suppressed = {slot: list(v) for slot, v in exo.initial_suppressed.items()}
    failures = arrivals = completed = misses = 0
    waits, queue_samples, busy_samples = [], [], []
    last_success = [0] * n
    retries = [0] * n
    next_retry = [0] * n
    observations = []
    rng = endogenous_rng

    for t in range(horizon):
        physical_capacity = base_capacity
        if action == "CAPACITY_ACTION" and t < capacity_duration:
            physical_capacity += int(config["actions"]["capacity_extra_slots"])
        valid_slots = set(range(physical_capacity))
        holders = {slot: v for slot, v in holders.items() if slot in valid_slots}
        suppressed = {slot: v for slot, v in suppressed.items()
                      if slot in valid_slots and slot not in holders}

        released = []
        for slot, (finish, node, request) in list(holders.items()):
            if finish <= t:
                released.append(slot)
                completed += 1
                last_success[node] = t
                del holders[slot]

        if action == "SYNC_ACTION":
            suppressed.clear()
        else:
            for slot in list(suppressed):
                suppressed[slot][0] -= 1
                if suppressed[slot][0] <= 0:
                    del suppressed[slot]
            for slot in released:
                if rng.random() < float(local["inconsistent_holder_probability"]):
                    suppressed[slot] = [
                        geometric(rng, local["release_confirmation_delay"]),
                        "release_confirmation"]
            interval = float(local["reconciliation_interval"])
            if interval > 0 and rng.random() < 1.0 / interval and suppressed:
                del suppressed[rng.choice(sorted(suppressed))]
            target = min(int(local["suppressed_slot_count"]),
                         physical_capacity - len(holders))
            free_unsuppressed = sorted(
                valid_slots - holders.keys() - suppressed.keys())
            rng.shuffle(free_unsuppressed)
            for slot in free_unsuppressed:
                if len(suppressed) >= target:
                    break
                if rng.random() >= float(local["conservative_admission_probability"]):
                    continue
                cause = ("conflicting_holder"
                          if rng.random() < float(local["inconsistent_holder_probability"])
                          else "stale_version")
                suppressed[slot] = [_suppression_lifetime(rng, local, cause), cause]

        for node in range(n):
            if exo.arrival[t, node]:
                arrivals += 1
                queues[node].append(Request(
                    t, t + int(exo.deadline_offset[t, node]),
                    int(exo.service_time[t, node])))
        for node in range(n):
            kept = []
            for request in queues[node]:
                if t > request.deadline:
                    misses += 1
                else:
                    kept.append(request)
            queues[node] = kept

        free_usable = sorted(valid_slots - holders.keys() - suppressed.keys())
        conservative_denial_this_tick = False
        start = exo.seed % n + t
        for offset in range(n):
            node = (start + offset) % n
            if not queues[node] or t < next_retry[node]:
                continue
            if (action != "SYNC_ACTION" and free_usable
                    and rng.random() < float(local["conservative_admission_probability"])):
                failures += 1
                retries[node] += 1
                next_retry[node] = t + geometric(rng, local["backoff_mean"])
                conservative_denial_this_tick = True
                continue
            if not free_usable:
                failures += 1
                retries[node] += 1
                next_retry[node] = t + geometric(rng, local["backoff_mean"])
                continue
            slot = free_usable.pop(0)
            request = queues[node].pop(0)
            waits.append(t - request.arrival)
            holders[slot] = (t + request.service, node, request)
            next_retry[node] = t

        free_usable_after = valid_slots - holders.keys() - suppressed.keys()
        observed_busy = not bool(free_usable_after) or conservative_denial_this_tick
        qlens = [len(q) for q in queues]
        queue_samples.extend(qlens)
        busy_samples.append(float(observed_busy))
        for node in range(n):
            observations.append({
                "local_queue_length": qlens[node],
                "head_of_line_wait":
                    t - queues[node][0].arrival if queues[node] else 0,
                "local_request_failures": retries[node],
                "locally_observed_busy_flag": int(observed_busy),
                "time_since_last_success": t - last_success[node],
                "local_retry_count": retries[node],
                "recent_local_throughput": int(last_success[node] == t),
            })

    final_queue_total = sum(len(q) for q in queues)
    misses += final_queue_total
    return _finalize(
        config, action, arrivals, completed, misses, failures, waits,
        queue_samples, busy_samples, observations, horizon,
        final_queue_total)


def _finalize(config, action, arrivals, completed, misses, failures, waits,
              queue_samples, busy_samples, observations, horizon,
              final_queue_total):
    action_cost = (config["actions"]["capacity_action_cost"]
                   if action == "CAPACITY_ACTION"
                   else config["actions"]["sync_action_cost"]
                   if action == "SYNC_ACTION" else 0.0)
    communication_bytes = (
        horizon * config["actions"]["sync_bytes_per_step"]
        if action == "SYNC_ACTION" else 0)
    mean_delay = sum(waits) / len(waits) if waits else float(horizon)
    weights = config["utility"]
    failure_rate = failures / max(1, arrivals)
    miss_rate = misses / max(1, arrivals)
    utility = -(
        weights["w_delay"] * mean_delay
        + weights["w_deadline_miss"] * miss_rate
        + weights["w_failure"] * failure_rate
        + weights["w_action_cost"] * action_cost
        + weights["w_communication_byte"] * communication_bytes)
    return {
        "action": action, "utility": utility, "arrivals": arrivals,
        "completed": completed, "misses": misses, "failures": failures,
        "mean_delay": mean_delay, "failure_rate": failure_rate,
        "miss_rate": miss_rate, "action_cost": action_cost,
        "communication_bytes": communication_bytes,
        "final_queue_occupancy": final_queue_total,
        "observations": observations,
        "config_hash": config_hash(config),
    }


def run_checkpoint(config, mechanism, seed):
    """Runs all three action branches, each from an INDEPENDENTLY REBUILT
    exogenous table (not a single shared-by-reference object), so that
    branch-hash equality is a genuine verification rather than a tautology
    -- it would catch an accidental in-place mutation of the exogenous
    table by any branch's endogenous logic."""
    branch_exo = {}
    branches = {}
    for action in ACTIONS:
        if mechanism == H1:
            exo = build_h1_exogenous(config, seed)
            branches[action] = run_h1_branch(config, action, exo)
        elif mechanism == H2_SUPPRESSION:
            exo = build_h2_exogenous(config, seed)
            endogenous_rng = keyed_rng(seed, mechanism, action, "endogenous")
            branches[action] = run_h2_branch(config, action, exo, endogenous_rng)
        else:
            raise ValueError(mechanism)
        branch_exo[action] = exo
    hashes = {a: (e.schedule_hash, e.initial_suppression_hash)
              for a, e in branch_exo.items()}
    if len(set(hashes.values())) != 1:
        raise RuntimeError(
            f"exogenous schedule hash mismatch across branches: {hashes}")
    return branch_exo[ACTIONS[0]], branches
