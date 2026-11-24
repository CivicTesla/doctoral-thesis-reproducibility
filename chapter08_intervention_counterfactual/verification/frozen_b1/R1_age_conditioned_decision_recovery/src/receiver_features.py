"""Receiver-derived age, missingness, and silence features."""

import numpy as np


def arrival_matrix(transmission):
    if "arrival_mask" in transmission:
        return np.any(np.asarray(transmission["arrival_mask"], bool), axis=2)
    return np.any(np.isfinite(np.asarray(transmission["received"], float)),
                  axis=2)


def receiver_state_series(transmission, initial_arrival_age=0):
    """Use arrivals and receiver time only; never inspect source state."""
    arrivals = arrival_matrix(transmission)
    horizon, nodes = arrivals.shape
    age = np.full((horizon, nodes), -1.0)
    missing = np.ones((horizon, nodes), dtype=float)
    silence = np.zeros((horizon, nodes), dtype=float)
    last_arrival = np.full(nodes, -1, dtype=int)
    for tick in range(horizon):
        for node in range(nodes):
            if arrivals[tick, node]:
                last_arrival[node] = tick
                age[tick, node] = float(initial_arrival_age)
                missing[tick, node] = 0.0
                silence[tick, node] = 0.0
            elif last_arrival[node] >= 0:
                elapsed = tick - last_arrival[node]
                age[tick, node] = float(initial_arrival_age + elapsed)
                missing[tick, node] = 0.0
                silence[tick, node] = float(elapsed)
            else:
                silence[tick, node] = float(tick + 1)
    return {
        "arrivals": arrivals.astype(float),
        "age": age, "missing": missing, "silence": silence,
    }


def _summary(prefix, values):
    flat = np.asarray(values, float).ravel()
    return {
        f"rx_mean_{prefix}": float(np.mean(flat)),
        f"rx_std_{prefix}": float(np.std(flat)),
        f"rx_p75_{prefix}": float(np.quantile(flat, .75)),
        f"rx_p90_{prefix}": float(np.quantile(flat, .90)),
        f"rx_max_{prefix}": float(np.max(flat)),
    }


def explicit_receiver_features(transmission, initial_arrival_age=0):
    groups = receiver_feature_groups(transmission, initial_arrival_age)
    return {key: value for group in groups.values()
            for key, value in group.items()}


def receiver_feature_groups(transmission, initial_arrival_age=0):
    series = receiver_state_series(transmission, initial_arrival_age)
    age = _summary("age", series["age"])
    mask = _summary("missing", series["missing"])
    arrivals = series["arrivals"]
    any_arrival = np.any(arrivals > 0, axis=1)
    system_silence = np.zeros(len(any_arrival), dtype=float)
    current = 0
    for tick, arrived in enumerate(any_arrival):
        current = 0 if arrived else current + 1
        system_silence[tick] = current
    silence = {
        **_summary("node_silence", series["silence"]),
        **_summary("system_silence", system_silence),
        "rx_max_current_silence": float(np.max(series["silence"][-1])),
        "rx_arrival_rate": float(np.mean(arrivals)),
        "rx_ticks_with_any_arrival_fraction": float(np.mean(any_arrival)),
    }
    return {"age": age, "mask": mask, "silence": silence}


def selected_receiver_features(transmission, groups, initial_arrival_age=0):
    available = receiver_feature_groups(transmission, initial_arrival_age)
    return {key: value for group in groups
            for key, value in available[group].items()}


def append_receiver_features(ordinary, transmission, initial_arrival_age=0):
    return {
        **ordinary,
        **explicit_receiver_features(transmission, initial_arrival_age),
    }
