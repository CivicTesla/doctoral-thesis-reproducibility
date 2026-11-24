"""Message-faithful ordinary-evidence primitives for B1 R0 Phase 5."""

from dataclasses import dataclass
import math

import numpy as np

from .phase4a_features import _stats
from .schemas import O0_FIELDS


REPORT_FIELDS = (
    "queue", "head_of_line_wait", "failure_count", "busy_flag",
    "retry_count", "time_since_last_success", "recent_throughput",
    "active_requester",
)
SOURCE_INDEX = {
    "queue": 0, "head_of_line_wait": 1, "failure_count": 2,
    "busy_flag": 3, "time_since_last_success": 4, "retry_count": 5,
    "recent_throughput": 6,
}
DEFAULT_BITS = {
    "queue": 16, "head_of_line_wait": 16, "failure_count": 16,
    "busy_flag": 1, "retry_count": 16, "time_since_last_success": 16,
    "recent_throughput": 8, "active_requester": 1,
}
BOOLEAN_FIELDS = {"busy_flag", "active_requester"}
NODE_ID_BYTES = 1
TIMESTAMP_BYTES = 2
HEADER_BYTES = 2
TOPK_TRIGGER_BYTES = 2


@dataclass(frozen=True)
class CommunicationScheme:
    scheme_id: str
    fields: tuple
    node_strategy: str = "all"
    node_count: int = 6
    period: int = 1
    imputation: str = "locf"
    # Zero selects the field-specific default encoding table.
    quantization_bits: int = 0
    separate_field_messages: bool = False
    message_delay: int = 0
    packet_loss: float = 0.0
    selection_trigger: str = "queue"


def field_value(o0_tick, field):
    if field == "active_requester":
        return (o0_tick[:, SOURCE_INDEX["queue"]] > 0).astype(float)
    return o0_tick[:, SOURCE_INDEX[field]].astype(float)


def payload_bytes(fields, bits=None, separate=False):
    bits = bits or DEFAULT_BITS
    if separate:
        return sum(math.ceil(bits[name] / 8) for name in fields)
    return math.ceil(sum(bits[name] for name in fields) / 8)


def protocol_bytes(fields, bits=None, separate=False):
    messages = len(fields) if separate else 1
    return payload_bytes(fields, bits, separate) + messages * (
        NODE_ID_BYTES + TIMESTAMP_BYTES + HEADER_BYTES)


def fit_quantization_ranges(o0, run_mask, fields):
    """Fit clipping ranges on training runs only."""
    if np.asarray(run_mask).dtype != bool:
        raise ValueError("run_mask must be boolean")
    selected = np.asarray(o0)[run_mask]
    if not len(selected):
        raise ValueError("empty quantization training set")
    ranges = {}
    for field in fields:
        values = []
        for run in selected:
            for tick in run:
                values.extend(field_value(tick, field))
        low, high = np.quantile(values, [.01, .99])
        if high <= low:
            high = low + 1.0
        ranges[field] = (float(low), float(high))
    return ranges


def quantize(values, field, bit_count, ranges):
    if field in BOOLEAN_FIELDS or bit_count == 1:
        return (np.asarray(values) > 0).astype(float)
    low, high = ranges[field]
    levels = 2 ** int(bit_count) - 1
    clipped = np.clip(values, low, high)
    code = np.rint((clipped - low) * levels / (high - low))
    return low + code * (high - low) / levels


def event_trigger(previous, current, fields, thresholds):
    """Causal local trigger using only previous and current observations."""
    if previous is None:
        return True
    return any(abs(float(current[name]) - float(previous[name]))
               > float(thresholds.get(name, 0.0)) for name in fields)


def selected_nodes(o0_tick, strategy, count, tick, rng,
                   trigger_field="queue"):
    n = len(o0_tick)
    count = min(int(count), n)
    if strategy == "all":
        return list(range(n)), 0
    if strategy == "round_robin":
        return [int((tick + offset) % n) for offset in range(count)], 0
    if strategy == "random":
        return sorted(rng.choice(n, count, replace=False).tolist()), 0
    if strategy == "top":
        # Exact top-k needs one trigger report from every node. This overhead is
        # charged explicitly; it is not treated as locally free.
        values = field_value(o0_tick, trigger_field)
        order = sorted(range(n), key=lambda node: (-values[node], node))
        overhead = n * (TOPK_TRIGGER_BYTES + NODE_ID_BYTES
                        + TIMESTAMP_BYTES + HEADER_BYTES)
        return order[:count], overhead
    if strategy == "one_fixed":
        return [0], 0
    raise ValueError(strategy)


def transmit_run(o0_run, scheme, quantization_ranges, seed=0):
    """Apply selection, sampling, quantization, loss, and delay to messages.

    Returns central received values only. Feature construction must consume this
    output, never the original ``o0_run``.
    """
    o0_run = np.asarray(o0_run)
    horizon, nodes, _ = o0_run.shape
    field_count = len(scheme.fields)
    received = np.full((horizon, nodes, field_count), np.nan)
    arrival_mask = np.zeros((horizon, nodes, field_count), dtype=bool)
    payload_by_tick = np.zeros(horizon, dtype=int)
    protocol_by_tick = np.zeros(horizon, dtype=int)
    message_count_by_tick = np.zeros(horizon, dtype=int)
    # Instrumentation-only masks. They are not consumed by the simulator,
    # receiver feature pipeline, byte accounting, or message hash.
    generation_mask = np.zeros((horizon, nodes), dtype=bool)
    drop_mask = np.zeros((horizon, nodes), dtype=bool)
    rng = np.random.default_rng(seed)
    scheduled = {}
    bits = {
        field: (DEFAULT_BITS[field] if scheme.quantization_bits == 0
                else 1 if field in BOOLEAN_FIELDS
                else int(scheme.quantization_bits))
        for field in scheme.fields
    }
    for tick in range(horizon):
        for node, values in scheduled.pop(tick, []):
            received[tick, node, :] = values
            arrival_mask[tick, node, :] = True
        if tick % int(scheme.period):
            continue
        nodes_sent, selection_cost = selected_nodes(
            o0_run[tick], scheme.node_strategy, scheme.node_count, tick, rng,
            scheme.selection_trigger)
        protocol_by_tick[tick] += selection_cost
        if selection_cost:
            message_count_by_tick[tick] += nodes
        for node in nodes_sent:
            generation_mask[tick, node] = True
            dropped = rng.random() < float(scheme.packet_loss)
            drop_mask[tick, node] = dropped
            if dropped:
                continue
            values = np.asarray([
                field_value(o0_run[tick], field)[node]
                for field in scheme.fields], dtype=float)
            if scheme.quantization_bits != 0:
                values = np.asarray([
                    quantize([value], field, bits[field],
                             quantization_ranges)[0]
                    for field, value in zip(scheme.fields, values)])
            arrival = tick + int(scheme.message_delay)
            if arrival < horizon:
                if arrival == tick:
                    received[tick, node, :] = values
                    arrival_mask[tick, node, :] = True
                else:
                    scheduled.setdefault(arrival, []).append((node, values))
            payload_by_tick[tick] += payload_bytes(
                scheme.fields, bits, scheme.separate_field_messages)
            protocol_by_tick[tick] += protocol_bytes(
                scheme.fields, bits, scheme.separate_field_messages)
            message_count_by_tick[tick] += (
                len(scheme.fields) if scheme.separate_field_messages else 1)
    return {
        "received": received, "arrival_mask": arrival_mask,
        "payload_by_tick": payload_by_tick,
        "protocol_by_tick": protocol_by_tick,
        "message_count_by_tick": message_count_by_tick,
        "generation_mask": generation_mask,
        "drop_mask": drop_mask,
    }


def impute_received(received, mode):
    output = np.asarray(received, float).copy()
    horizon, nodes, fields = output.shape
    for node in range(nodes):
        for field in range(fields):
            last = None
            previous = None
            for tick in range(horizon):
                if np.isfinite(output[tick, node, field]):
                    previous, last = last, output[tick, node, field]
                elif mode == "unknown":
                    output[tick, node, field] = -1.0
                elif mode == "locf":
                    output[tick, node, field] = 0.0 if last is None else last
                elif mode == "extrapolate":
                    if last is None:
                        output[tick, node, field] = 0.0
                    elif previous is None:
                        output[tick, node, field] = last
                    else:
                        output[tick, node, field] = last + (last - previous)
                else:
                    raise ValueError(mode)
    return output


def central_summary(transmission, scheme, include_derived_o1=False):
    """Construct features exclusively from received/imputed messages."""
    values = impute_received(transmission["received"], scheme.imputation)
    output = {}
    by_name = {name: values[:, :, index]
               for index, name in enumerate(scheme.fields)}
    for name, matrix in by_name.items():
        stats = _stats(matrix.ravel().tolist())
        output.update({f"{stat}_{name}": value
                       for stat, value in stats.items()})
    if include_derived_o1:
        derived = {}
        if "queue" in by_name:
            derived["aggregate_mean_queue"] = np.mean(by_name["queue"], axis=1)
            derived["number_of_active_requesters"] = np.sum(
                by_name["queue"] > 0, axis=1)
        elif "active_requester" in by_name:
            derived["number_of_active_requesters"] = np.sum(
                by_name["active_requester"] > 0, axis=1)
        if "busy_flag" in by_name:
            derived["aggregate_busy_fraction"] = np.mean(
                by_name["busy_flag"], axis=1)
        if "failure_count" in by_name:
            derived["aggregate_failure_rate"] = np.mean(
                by_name["failure_count"], axis=1)
        if "number_of_active_requesters" in derived:
            derived["simple_contention_graph_degree"] = np.maximum(
                0, derived["number_of_active_requesters"] - 1)
        for name, series in derived.items():
            stats = _stats(series.tolist())
            output.update({f"{stat}_{name}": value
                           for stat, value in stats.items()})
    if not output:
        raise ValueError("central feature set is empty")
    return output
