"""Simulator for H2 distributed capacity suppression.

This is intentionally separate from the audited H2_STALE_RELEASE path so old
results remain reproducible. Physical slots are never removed: coordination
conflicts mark otherwise free slots unusable until their stochastic lifetime
expires or SYNC_ACTION reconciles the state.
"""
from dataclasses import dataclass
import math
import random

from .metrics import percentile
from .schemas import O0_FIELDS
from .simulator import ACTIONS, config_hash

H2_SUPPRESSION = "H2_DISTRIBUTED_CAPACITY_SUPPRESSION"

@dataclass
class SuppressionRequest:
    arrival: int
    deadline: int
    service: int
    service_wait: int = 0
    retry_wait: int = 0
    suppression_wait: int = 0
    stale_state_wait: int = 0

def geometric(rng, mean):
    mean = float(mean)
    if mean <= 1.0:
        return 1
    p = 1.0 / mean
    return max(1, math.ceil(
        math.log(max(1e-15, 1.0 - rng.random())) / math.log(1.0 - p)))

def _suppression_lifetime(rng, config, cause):
    mean = (config["stale_version_lifetime"] if cause == "stale_version"
            else config["reservation_retention_duration"])
    return geometric(rng, mean)

def simulate_suppression(config, action, seed, collect_instrumentation=False):
    if action not in ACTIONS:
        raise ValueError("unsupported action")
    local = config["h2_distributed_capacity_suppression"]
    rng = random.Random(seed)
    n, horizon = int(config["num_agents"]), int(config["horizon"])
    base_capacity = int(local["physical_capacity"])
    capacity_duration = int(config["actions"].get(
        "capacity_action_duration", horizon))
    queues = [[] for _ in range(n)]
    holders = {}  # physical slot -> (finish, node, request)
    suppressed = {}  # physical slot -> [remaining lifetime, cause]
    initial_probability = min(
        1.0, float(local["suppressed_slot_count"]) / base_capacity)
    for slot in range(base_capacity):
        if rng.random() < initial_probability:
            cause = ("conflicting_holder"
                     if rng.random() < local["inconsistent_holder_probability"]
                     else "stale_version")
            suppressed[slot] = [_suppression_lifetime(rng, local, cause), cause]

    failures = arrivals = completed = misses = 0
    waits, queue_samples, busy_samples = [], [], []
    last_success = [0] * n
    retries = [0] * n
    next_retry = [0] * n
    observations, delay_components, capacity_trace = [], [], []
    cause_ticks = {"stale_version": 0, "conflicting_holder": 0,
                   "release_confirmation": 0,
                   "conservative_admission": 0}

    for t in range(horizon):
        physical_capacity = base_capacity
        if action == "CAPACITY_ACTION" and t < capacity_duration:
            physical_capacity += int(config["actions"]["capacity_extra_slots"])
        valid_slots = set(range(physical_capacity))
        holders = {slot: value for slot, value in holders.items()
                   if slot in valid_slots}
        suppressed = {slot: value for slot, value in suppressed.items()
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
                        "release_confirmation",
                    ]
            interval = float(local["reconciliation_interval"])
            if interval > 0 and rng.random() < 1.0 / interval and suppressed:
                # Background reconciliation resolves a random view, without a
                # fixed periodic signature.
                del suppressed[rng.choice(sorted(suppressed))]

            target = min(int(local["suppressed_slot_count"]),
                         physical_capacity - len(holders))
            free_unsuppressed = sorted(
                valid_slots - holders.keys() - suppressed.keys())
            rng.shuffle(free_unsuppressed)
            for slot in free_unsuppressed:
                if len(suppressed) >= target:
                    break
                if (rng.random()
                        >= float(local["conservative_admission_probability"])):
                    continue
                cause = ("conflicting_holder"
                         if rng.random()
                         < float(local["inconsistent_holder_probability"])
                         else "stale_version")
                suppressed[slot] = [
                    _suppression_lifetime(rng, local, cause), cause]

        for node in range(n):
            if rng.random() < float(local["arrival_rate"]):
                arrivals += 1
                queues[node].append(SuppressionRequest(
                    arrival=t,
                    deadline=t + geometric(rng, config["deadline_mean"]),
                    service=geometric(rng, local["service_time_mean"]),
                ))
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
        start = seed % n + t
        for offset in range(n):
            node = (start + offset) % n
            if not queues[node] or t < next_retry[node]:
                continue
            if (action != "SYNC_ACTION" and free_usable
                    and rng.random()
                    < float(local["conservative_admission_probability"])):
                failures += 1
                retries[node] += 1
                next_retry[node] = t + geometric(
                    rng, local["backoff_mean"])
                cause_ticks["conservative_admission"] += 1
                conservative_denial_this_tick = True
                continue
            if not free_usable:
                failures += 1
                retries[node] += 1
                next_retry[node] = t + geometric(
                    rng, local["backoff_mean"])
                continue
            slot = free_usable.pop(0)
            request = queues[node].pop(0)
            waits.append(t - request.arrival)
            if collect_instrumentation:
                delay_components.append({
                    "seed": seed, "queue_wait": t - request.arrival,
                    "service_wait": request.service_wait,
                    "retry_wait": request.retry_wait,
                    "suppression_wait": request.suppression_wait,
                    "stale_state_wait": request.stale_state_wait,
                    "total_delay": t - request.arrival,
                })
            holders[slot] = (t + request.service, node, request)
            next_retry[node] = t

        free_usable = valid_slots - holders.keys() - suppressed.keys()
        observed_busy = not bool(free_usable) or conservative_denial_this_tick
        qlens = [len(queue) for queue in queues]
        queue_samples.extend(qlens)
        busy_samples.append(float(observed_busy))
        for node in range(n):
            observation = {
                "local_queue_length": qlens[node],
                "head_of_line_wait":
                    t - queues[node][0].arrival if queues[node] else 0,
                "local_request_failures": retries[node],
                "locally_observed_busy_flag": int(observed_busy),
                "time_since_last_success": t - last_success[node],
                "local_retry_count": retries[node],
                "recent_local_throughput": int(last_success[node] == t),
            }
            assert tuple(observation) == O0_FIELDS
            observations.append(observation)

        if collect_instrumentation:
            capacity_trace.append({
                "time": t, "physical_capacity": physical_capacity,
                "active_service_occupancy": len(holders),
                "suppressed_slots": len(suppressed),
                "effective_usable_capacity":
                    physical_capacity - len(suppressed),
                "currently_free_usable_slots": len(free_usable),
            })
            for _, cause in suppressed.values():
                cause_ticks[cause] += 1
            physical_full = len(holders) >= physical_capacity
            stale_present = any(
                cause in ("stale_version", "release_confirmation")
                for _, cause in suppressed.values())
            for node, queue in enumerate(queues):
                if t < next_retry[node]:
                    category = "retry_wait"
                elif physical_full:
                    category = "service_wait"
                elif stale_present:
                    category = "stale_state_wait"
                else:
                    category = "suppression_wait"
                for request in queue:
                    setattr(request, category,
                            getattr(request, category) + 1)

    misses += sum(len(queue) for queue in queues)
    mean_delay = statistics_mean(waits, float(horizon))
    failure_rate = failures / max(1, arrivals)
    miss_rate = misses / max(1, arrivals)
    action_cost = (config["actions"]["capacity_action_cost"]
                   if action == "CAPACITY_ACTION"
                   else config["actions"]["sync_action_cost"]
                   if action == "SYNC_ACTION" else 0.0)
    communication_bytes = (
        horizon * config["actions"]["sync_bytes_per_step"]
        if action == "SYNC_ACTION" else 0)
    weights = config["utility"]
    utility = -(
        weights["w_delay"] * mean_delay
        + weights["w_deadline_miss"] * miss_rate
        + weights["w_failure"] * failure_rate
        + weights["w_action_cost"] * action_cost
        + weights["w_communication_byte"] * communication_bytes
    )
    result = {
        "seed": seed, "mechanism": H2_SUPPRESSION, "action": action,
        "config_hash": config_hash(config), "utility": utility,
        "mean_local_queue": statistics_mean(queue_samples, 0.0),
        "p90_local_queue": percentile(queue_samples, .9),
        "mean_delay": mean_delay, "p90_delay": percentile(waits, .9),
        "request_failure_rate": failure_rate,
        "busy_fraction": statistics_mean(busy_samples, 0.0),
        "throughput": completed / horizon, "deadline_miss_rate": miss_rate,
        "communication_bytes": communication_bytes, "arrivals": arrivals,
        "completed": completed, "observations": observations,
        "physical_capacity": base_capacity,
        "offered_load_requirement":
            n * float(local["arrival_rate"])
            * float(local["service_time_mean"]),
    }
    if collect_instrumentation:
        result["delay_components"] = delay_components
        result["capacity_trace"] = capacity_trace
        result["suppression_cause_ticks"] = cause_ticks
    return result

def statistics_mean(values, empty):
    return sum(values) / len(values) if values else empty
