"""Causal event-triggered ordinary-evidence messages for Phase 5.3."""

from dataclasses import dataclass
import math

import numpy as np

from .phase4a_features import _stats
from .phase5_communication import (
    BOOLEAN_FIELDS, DEFAULT_BITS, SOURCE_INDEX, field_value)


EVENT_HEADER_BYTES = 2
EVENT_NODE_ID_BYTES = 1
EVENT_TIMESTAMP_BYTES = 2
EVENT_SEQUENCE_BYTES = 2
EVENT_FIELD_BITMAP_BYTES = 1
EVENT_ENVELOPE_BYTES = (
    EVENT_HEADER_BYTES + EVENT_NODE_ID_BYTES + EVENT_TIMESTAMP_BYTES
    + EVENT_SEQUENCE_BYTES + EVENT_FIELD_BITMAP_BYTES)


@dataclass(frozen=True)
class EventScheme:
    scheme_id: str
    trigger: str
    fields: tuple
    threshold: float = 0.0
    direction: str = "both"
    reconstruction: str = "locf"
    failure_encoding: str = "cumulative"


def _crossed(previous, current, threshold, direction):
    upward = previous < threshold <= current
    downward = previous >= threshold > current
    if direction == "up":
        return upward
    if direction == "down":
        return downward
    if direction == "both":
        return upward or downward
    raise ValueError(direction)


def should_trigger(scheme, current, previous_local, last_transmitted):
    """Use only current local state and locally stored past state."""
    if previous_local is None or last_transmitted is None:
        return True
    if scheme.trigger == "failure_increase":
        return current["failure_count"] > last_transmitted["failure_count"]
    if scheme.trigger == "busy_change":
        return current["busy_flag"] != last_transmitted["busy_flag"]
    if scheme.trigger == "queue_crossing":
        return _crossed(
            previous_local["queue"], current["queue"],
            scheme.threshold, scheme.direction)
    if scheme.trigger == "queue_delta":
        return abs(current["queue"] - last_transmitted["queue"]) >= scheme.threshold
    if scheme.trigger == "time_success_crossing":
        return _crossed(
            previous_local["time_since_last_success"],
            current["time_since_last_success"], scheme.threshold,
            scheme.direction)
    if scheme.trigger == "combined":
        return (
            current["failure_count"] > last_transmitted["failure_count"]
            or current["busy_flag"] != last_transmitted["busy_flag"]
            or _crossed(previous_local["queue"], current["queue"],
                        scheme.threshold, "both")
            or _crossed(previous_local["time_since_last_success"],
                        current["time_since_last_success"],
                        scheme.threshold, "both"))
    raise ValueError(scheme.trigger)


def _local_state(o0_tick, node):
    return {
        "queue": float(field_value(o0_tick, "queue")[node]),
        "head_of_line_wait":
            float(field_value(o0_tick, "head_of_line_wait")[node]),
        "failure_count":
            float(field_value(o0_tick, "failure_count")[node]),
        "busy_flag": float(field_value(o0_tick, "busy_flag")[node]),
        "time_since_last_success":
            float(field_value(o0_tick, "time_since_last_success")[node]),
        "recent_throughput":
            float(field_value(o0_tick, "recent_throughput")[node]),
        "active_requester":
            float(field_value(o0_tick, "active_requester")[node]),
    }


def _encoded_payload_bytes(scheme):
    if (scheme.failure_encoding == "event_bit"
            and scheme.fields == ("failure_count",)):
        return 1
    if (scheme.failure_encoding == "increment"
            and scheme.fields == ("failure_count",)):
        # uint8 increment plus reset/overflow flags packed in one status byte.
        return 2
    return math.ceil(sum(DEFAULT_BITS[field] for field in scheme.fields) / 8)


def transmit_event_run(o0_run, scheme, message_delay=0, packet_loss=0.0,
                       seed=0):
    """Create received evidence under a lossless, zero-delay event channel.

    Loss and delay are intentionally deferred to Phase 5.5+. Sequence numbers
    make later gap detection possible. Cumulative counts saturate at uint16;
    increments saturate at uint8 and carry an overflow flag. Counter reset is
    represented by the status byte and resets the central accumulator.
    """
    o0_run = np.asarray(o0_run)
    horizon, nodes, _ = o0_run.shape
    fields = scheme.fields
    received = np.full((horizon, nodes, len(fields)), np.nan)
    age = np.full((horizon, nodes, len(fields)), np.nan)
    generation_tick = np.full((horizon, nodes, len(fields)), np.nan)
    payload = np.zeros(horizon, dtype=int)
    protocol = np.zeros(horizon, dtype=int)
    messages = np.zeros(horizon, dtype=int)
    previous_local = [None] * nodes
    last_transmitted = [None] * nodes
    central_failure = np.zeros(nodes, dtype=float)
    sequences = np.zeros(nodes, dtype=int)
    rng = np.random.default_rng(seed)
    scheduled = {}
    for tick in range(horizon):
        for node, values, generated in scheduled.pop(tick, []):
            received[tick, node, :] = values
            generation_tick[tick, node, :] = generated
            age[tick, node, :] = tick - generated
        for node in range(nodes):
            current = _local_state(o0_run[tick], node)
            fire = should_trigger(
                scheme, current, previous_local[node],
                last_transmitted[node])
            if fire:
                values = []
                for field in fields:
                    value = current[field]
                    if field == "failure_count":
                        previous = (
                            0.0 if last_transmitted[node] is None
                            else last_transmitted[node]["failure_count"])
                        reset = value < previous
                        if scheme.failure_encoding == "event_bit":
                            value = 1.0 if value > previous else 0.0
                            central_failure[node] = (
                                0.0 if reset else central_failure[node])
                            central_failure[node] += value
                            value = central_failure[node]
                        elif scheme.failure_encoding == "increment":
                            increment = value if reset else max(0.0, value - previous)
                            encoded = min(255.0, increment)
                            central_failure[node] = (
                                encoded if reset else central_failure[node] + encoded)
                            value = central_failure[node]
                        else:
                            value = min(65535.0, max(0.0, value))
                    values.append(value)
                last_transmitted[node] = current.copy()
                sequences[node] = (sequences[node] + 1) % 65536
                pbytes = _encoded_payload_bytes(scheme)
                payload[tick] += pbytes
                protocol[tick] += pbytes + EVENT_ENVELOPE_BYTES
                messages[tick] += 1
                if rng.random() >= float(packet_loss):
                    arrival = tick + int(message_delay)
                    if arrival < horizon:
                        if arrival == tick:
                            received[tick, node, :] = values
                            generation_tick[tick, node, :] = tick
                            age[tick, node, :] = 0.0
                        else:
                            scheduled.setdefault(arrival, []).append(
                                (node, values, tick))
            previous_local[node] = current
        # Age uses receipt history only. It never inspects an unreported value.
        for node in range(nodes):
            for field_index in range(len(fields)):
                if not np.isfinite(age[tick, node, field_index]):
                    prior = (-1 if tick == 0
                             else age[tick - 1, node, field_index])
                    age[tick, node, field_index] = (
                        prior + 1 if np.isfinite(prior) and prior >= 0
                        else -1)
    return {
        "received": received, "age": age,
        "generation_tick": generation_tick,
        "payload_by_tick": payload, "protocol_by_tick": protocol,
        "message_count_by_tick": messages,
    }


def reconstruct_event(received, age, mode):
    output = np.asarray(received, float).copy()
    horizon, nodes, fields = output.shape
    for node in range(nodes):
        for field in range(fields):
            last = None
            for tick in range(horizon):
                if np.isfinite(output[tick, node, field]):
                    last = output[tick, node, field]
                elif mode == "unknown":
                    output[tick, node, field] = -1.0
                elif mode in ("locf", "age_tagged"):
                    output[tick, node, field] = 0.0 if last is None else last
                else:
                    raise ValueError(mode)
    return output


def event_central_summary(transmission, scheme):
    values = reconstruct_event(
        transmission["received"], transmission["age"],
        scheme.reconstruction)
    output = {}
    by_name = {field: values[:, :, index]
               for index, field in enumerate(scheme.fields)}
    for name, matrix in by_name.items():
        output.update({
            f"{stat}_{name}": value
            for stat, value in _stats(matrix.ravel().tolist()).items()})
    if scheme.reconstruction == "age_tagged":
        for index, name in enumerate(scheme.fields):
            output.update({
                f"{stat}_age_{name}": value
                for stat, value in _stats(
                    transmission["age"][:, :, index].ravel().tolist()).items()})
    # Derive only aggregates whose required reports were actually received.
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
        output.update({
            f"{stat}_{name}": value
            for stat, value in _stats(series.tolist()).items()})
    return output


def maximum_silence_duration(message_count_by_tick):
    longest = current = 0
    for count in message_count_by_tick:
        if count == 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
