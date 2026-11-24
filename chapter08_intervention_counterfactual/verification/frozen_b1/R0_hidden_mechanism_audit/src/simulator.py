from dataclasses import dataclass
import hashlib
import json
import math
import random

from .metrics import percentile
from .schemas import O0_FIELDS

H1 = "H1_CAPACITY_SHORTAGE"
H2 = "H2_STALE_RELEASE"
ACTIONS = ("NO_ACTION", "CAPACITY_ACTION", "SYNC_ACTION")

@dataclass
class Request:
    arrival: int
    deadline: int
    service: int
    contention_wait: int = 0
    stale_state_wait: int = 0
    retry_wait: int = 0

def _geometric(rng, mean):
    p = 1.0 / float(mean)
    return max(1, math.ceil(math.log(max(1e-15, 1.0 - rng.random())) /
                            math.log(1.0 - p)))

def config_hash(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

def mechanism_at(schedule, t):
    """Return the last mechanism whose start time is not after t."""
    eligible = [(start, mechanism) for start, mechanism in schedule if start <= t]
    if not eligible:
        raise ValueError("schedule must start no later than t=0")
    return max(eligible)[1]

def simulate(config, mechanism, action, seed, collect_instrumentation=False):
    if mechanism not in (H1, H2) or action not in ACTIONS:
        raise ValueError("unsupported mechanism/action")
    rng = random.Random(seed)
    local = dict(config)
    local.update(config["h1" if mechanism == H1 else "h2"])
    n, horizon = int(local["num_agents"]), int(local["horizon"])
    capacity = int(local["resource_capacity"])
    base_capacity = capacity
    capacity_duration = int(config["actions"].get(
        "capacity_action_duration", horizon))
    queues = [[] for _ in range(n)]
    holders = []
    stale_until = []
    failures = arrivals = completed = misses = 0
    waits, queue_samples, busy_samples = [], [], []
    last_success = [0] * n
    retries = [0] * n
    next_retry = [0] * n
    observations = []
    delay_components = []
    busy_instrumentation = []

    for t in range(horizon):
        capacity = base_capacity
        if action == "CAPACITY_ACTION" and t < capacity_duration:
            capacity += int(config["actions"]["capacity_extra_slots"])
        remaining = []
        for finish, node, req in holders:
            if finish <= t:
                completed += 1
                last_success[node] = t
                if mechanism == H2 and action != "SYNC_ACTION":
                    loss_rate = float(local.get("release_loss_rate", 0.0))
                    lost = loss_rate > 0.0 and rng.random() < loss_rate
                    retention = (int(local.get("stale_state_retention",
                                               local["release_message_delay"]))
                                 if lost else int(local["release_message_delay"]))
                    stale_until.append(t + retention)
            else:
                remaining.append((finish, node, req))
        holders = remaining
        stale_until = [x for x in stale_until if x > t]
        if action == "SYNC_ACTION":
            stale_until.clear()

        for node in range(n):
            if rng.random() < float(local["arrival_rate"]):
                arrivals += 1
                service = _geometric(rng, local["service_time_mean"])
                deadline = t + _geometric(rng, local["deadline_mean"])
                queues[node].append(Request(t, deadline, service))

        for node in range(n):
            kept = []
            for req in queues[node]:
                if t > req.deadline:
                    misses += 1
                else:
                    kept.append(req)
            queues[node] = kept

        believed_busy = min(capacity, len(holders) + len(stale_until)) >= capacity
        start = seed % n + t
        for offset in range(n):
            node = (start + offset) % n
            if not queues[node]:
                continue
            admission_intensity = (float(local.get("admission_intensity", 1.0))
                                   if mechanism == H1 else 1.0)
            if admission_intensity < 1.0 and rng.random() >= admission_intensity:
                continue
            if mechanism == H2 and t < next_retry[node]:
                continue
            if believed_busy or len(holders) >= capacity:
                failures += 1
                retries[node] += 1
                if mechanism == H2:
                    next_retry[node] = t + int(local.get("retry_timing", 1))
                continue
            req = queues[node].pop(0)
            waits.append(t - req.arrival)
            if collect_instrumentation:
                delay_components.append({
                    "seed": seed,
                    "queue_wait": t - req.arrival,
                    "contention_wait": req.contention_wait,
                    "stale_state_wait": req.stale_state_wait,
                    "service_time": req.service,
                    "retry_wait": req.retry_wait,
                    # A diagnostic view of stale wait after a physical release;
                    # it overlaps stale_state_wait and is not added again.
                    "post_release_wait": req.stale_state_wait,
                })
            holders.append((t + req.service, node, req))
            next_retry[node] = t
            believed_busy = min(capacity, len(holders) + len(stale_until)) >= capacity

        qlens = [len(q) for q in queues]
        queue_samples.extend(qlens)
        busy_samples.append(float(believed_busy))
        for node in range(n):
            obs = {
                "local_queue_length": qlens[node],
                "head_of_line_wait": t - queues[node][0].arrival if queues[node] else 0,
                "local_request_failures": retries[node],
                "locally_observed_busy_flag": int(believed_busy),
                "time_since_last_success": t - last_success[node],
                "local_retry_count": retries[node],
                "recent_local_throughput": int(last_success[node] == t),
            }
            assert tuple(obs) == O0_FIELDS
            observations.append(obs)
        if collect_instrumentation:
            physical_busy = int(len(holders) >= capacity)
            stale_busy = int(believed_busy and not physical_busy)
            busy_instrumentation.append({
                "time": t,
                "locally_observed_busy": int(believed_busy),
                "physical_resource_busy": physical_busy,
                "stale_busy_belief": stale_busy,
                "admitted_occupancy": len(holders),
                "active_service_occupancy": len(holders) / capacity,
            })

            # Attribute exactly one unit of elapsed queue wait for every request
            # left queued at the end of this tick. Categories are exclusive.
            for node, queue in enumerate(queues):
                if mechanism == H2 and t < next_retry[node]:
                    category = "retry_wait"
                elif stale_busy:
                    category = "stale_state_wait"
                else:
                    category = "contention_wait"
                for req in queue:
                    setattr(req, category, getattr(req, category) + 1)

    misses += sum(len(q) for q in queues)
    action_cost = (config["actions"]["capacity_action_cost"] if action == "CAPACITY_ACTION"
                   else config["actions"]["sync_action_cost"] if action == "SYNC_ACTION"
                   else 0.0)
    communication_bytes = (horizon * config["actions"]["sync_bytes_per_step"]
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
        + weights["w_communication_byte"] * communication_bytes
    )
    result = {
        "seed": seed, "mechanism": mechanism, "action": action,
        "config_hash": config_hash(config), "utility": utility,
        "mean_local_queue": sum(queue_samples) / len(queue_samples),
        "p90_local_queue": percentile(queue_samples, .9),
        "mean_delay": mean_delay, "p90_delay": percentile(waits, .9),
        "request_failure_rate": failure_rate,
        "busy_fraction": sum(busy_samples) / len(busy_samples),
        "throughput": completed / horizon, "deadline_miss_rate": miss_rate,
        "communication_bytes": communication_bytes, "arrivals": arrivals,
        "completed": completed, "observations": observations,
    }
    if collect_instrumentation:
        result["delay_components"] = delay_components
        result["busy_instrumentation"] = busy_instrumentation
    return result
