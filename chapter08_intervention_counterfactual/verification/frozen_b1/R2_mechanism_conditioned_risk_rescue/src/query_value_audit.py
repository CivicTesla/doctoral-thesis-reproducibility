"""Pure guards for the read-only Gate 1A query-replay audit."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

SEALED_R1 = set(range(2000, 2030))
FORBIDDEN_R0 = set(range(120, 150))
ALLOWED_FIELDS = (
    "local_request_failures",
    "local_queue_length",
    "head_of_line_wait",
    "locally_observed_busy_flag",
    "time_since_last_success",
    "recent_local_throughput",
)
ORACLE_METHODS = {"Q1", "Q2", "Q3", "Q4"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_seed_safety(seeds) -> None:
    used = {int(seed) for seed in seeds}
    if used & SEALED_R1:
        raise ValueError("sealed R1 final seed loaded")
    if used & FORBIDDEN_R0:
        raise ValueError("R0 final seed loaded")


def validate_o0_fields(fields) -> tuple[str, ...]:
    fields = tuple(str(value) for value in fields)
    missing = set(ALLOWED_FIELDS) - set(fields)
    if missing:
        raise ValueError(f"missing O0 query fields: {sorted(missing)}")
    if "local_retry_count" in ALLOWED_FIELDS:
        raise AssertionError("failure/retry alias was double counted")
    return ALLOWED_FIELDS


def update_only_queried_fields(
        stale_state, fresh_tick, node: int, field_indices) -> np.ndarray:
    """Return a copy changing only explicitly queried node-field cells."""
    output = np.asarray(stale_state, float).copy()
    fresh = np.asarray(fresh_tick, float)
    for field in field_indices:
        output[node, int(field)] = fresh[node, int(field)]
    return output


def select_node(prequery_state, field_index: int, eligible=None) -> int:
    """A deterministic selector that can inspect only the pre-query state."""
    values = np.asarray(prequery_state, float)[:, int(field_index)]
    if eligible is None:
        eligible = np.ones(len(values), dtype=bool)
    eligible = np.asarray(eligible, bool)
    if not eligible.any():
        eligible = np.ones(len(values), dtype=bool)
    candidates = np.flatnonzero(eligible)
    return int(candidates[np.argmax(values[candidates])])


def assert_no_future_tick(decision_tick: int, source_tick: int) -> None:
    if int(source_tick) > int(decision_tick):
        raise ValueError("query reads a future tick")


def label_method(method: str) -> str:
    return "NOT_DEPLOYABLE" if method in ORACLE_METHODS else "DEPLOYABLE"


def validate_fold_reuse(g0b_columns) -> None:
    required = {
        "fold", "outer_train_seeds", "outer_test_seeds",
        "tau_h1", "tau_h2", "theta_default", "theta_flip",
    }
    missing = required - set(g0b_columns)
    if missing:
        raise ValueError(f"Gate 0B fold parameters incomplete: {missing}")


def require_replay_inputs(prediction_columns, trace_seed_min,
                          trace_seed_max, query_request_schema: bool):
    """Return pre-registered blockers without attempting reconstruction."""
    blockers = []
    required = {"decision_timestamp", "tick_trace_reference"}
    missing = sorted(required - set(prediction_columns))
    if missing:
        blockers.append("missing_prediction_columns:" + "|".join(missing))
    if trace_seed_min is None or trace_seed_max is None:
        blockers.append("gate1_tick_trace_missing")
    elif trace_seed_min > 3000 or trace_seed_max < 3399:
        blockers.append("gate1_tick_trace_seed_coverage_missing")
    if not query_request_schema:
        blockers.append("query_request_response_framing_undefined")
    return blockers


def bootstrap_seed_indices(seeds, replicates: int, random_seed: int = 8107):
    unique = np.asarray(sorted(set(int(seed) for seed in seeds)))
    rng = np.random.default_rng(random_seed)
    return [rng.choice(unique, len(unique), replace=True)
            for _ in range(int(replicates))]

