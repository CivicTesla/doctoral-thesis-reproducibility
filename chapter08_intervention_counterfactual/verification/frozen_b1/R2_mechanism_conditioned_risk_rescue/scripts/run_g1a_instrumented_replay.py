#!/usr/bin/env python3
"""Recover Gate 1 development traces without fitting or changing behavior."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
R2 = HERE.parents[1]
B1 = HERE.parents[2]
R1 = B1 / "R1_age_conditioned_decision_recovery"
R0 = B1 / "R0_hidden_mechanism_audit"
sys.path[:0] = [
    str(R2 / "src"), str(R1 / "src"), str(R1 / "scripts"),
    str(R0), str(R0 / "scripts"),
]

from conservative_gate import (  # noqa: E402
    apply_direction_thresholds, c3_orientation)
from gate0_config import load_gate0_config  # noqa: E402
from protocol import assert_fixed_schemes  # noqa: E402
from query_value_audit import assert_seed_safety, sha256  # noqa: E402
from receiver_features import (  # noqa: E402
    receiver_state_series, selected_receiver_features)
from rejection_diagnosis import recover_score  # noqa: E402
from run_gate0_pilot import simulate_records  # noqa: E402
from run_gate1_formal import (  # noqa: E402
    ABLATIONS, message_hash, parameter_matched, prediction_row)
from run_phase5_final import transmission  # noqa: E402
from src.config import load_config  # noqa: E402
from src.phase5_final import frozen_scheme, predict_frozen  # noqa: E402
from src.simulator import H1  # noqa: E402
from src.suppression_simulator import H2_SUPPRESSION  # noqa: E402


DEFAULT_OUT = R2 / "G1A_instrumented_replay_data"
GATE1 = R1 / "results/gate1_formal_age_necessity"
G0B = R2 / "G0B_conservative_confidence_gate"
TOL = 1e-12
COMPARE_METHODS = (
    "M0_ZERO_DELAY_FROZEN", "M1_CHANNEL_AUGMENTATION",
    "M4_AGE_MASK_SILENCE", "M5_COST_SENSITIVE_REJECT")


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def serialized_probability(artifact, feature_map):
    order = artifact["feature_order"]
    x = np.asarray([float(feature_map[name]) for name in order])
    mean = np.asarray(artifact["scaler_mean"], float)
    scale = np.asarray(artifact["scaler_scale"], float)
    weights = np.asarray(artifact["coefficients_intercept_first"], float)
    score = np.clip(weights[0] + ((x - mean) / scale) @ weights[1:], -30, 30)
    return float(1 / (1 + np.exp(-score)))


def condition_name(delay, loss):
    return f"delay_{int(delay)}_loss_{str(float(loss)).replace('.', 'p')}"


def historical_record_index(record):
    mechanism_offset = 0 if record["mechanism"] == H1 else 1
    if record["split"] == "train":
        return (int(record["seed"]) - 3000) * 2 + mechanism_offset
    if record["split"] == "validation":
        return 400 + (int(record["seed"]) - 3200) * 2 + mechanism_offset
    if record["split"] == "development_test":
        return 600 + (int(record["seed"]) - 3300) * 2 + mechanism_offset
    raise ValueError(record["split"])


def last_received_state(sent, decision_tick, initial_age):
    received = np.asarray(sent["received"], float)
    nodes, fields = received.shape[1:]
    last = np.full((nodes, fields), np.nan)
    age = np.full(nodes, -1, dtype=int)
    for node in range(nodes):
        arrivals = np.flatnonzero(np.any(
            np.isfinite(received[:decision_tick + 1, node]), axis=1))
        if len(arrivals):
            tick = int(arrivals[-1])
            last[node] = received[tick, node]
            age[node] = int(initial_age + decision_tick - tick)
    return last, age


def load_c3_parameters():
    frame = pd.read_csv(G0B / "05_selected_parameters_by_fold.csv")
    frame = frame[frame.method == "C3_UNCERTAINTY_REJECT"].copy()
    return {int(row.fold): row for _, row in frame.iterrows()}


def c3_decision(probability, score, fitted):
    risk, direction, confident = c3_orientation(
        [score], [1.0 - probability],
        float(fitted.tau_h1), float(fitted.tau_h2))
    accepted = apply_direction_thresholds(
        risk, direction, confident, {
            "theta_default": float(fitted.theta_default),
            "theta_flip": float(fitted.theta_flip),
        })[0]
    if not accepted:
        return "REJECT", "REJECT"
    action = "CAPACITY_ACTION" if int(direction[0]) == 1 else "SYNC_ACTION"
    return "ACCEPT", action


def archive_index():
    frame = pd.read_csv(GATE1 / "per_run_predictions.csv")
    frame = frame[
        frame.method.isin(COMPARE_METHODS)
        & np.isclose(frame.protocol_bytes_per_tick, 5.10)].copy()
    frame["key"] = list(zip(
        frame.run_id, frame.method, frame.message_delay.astype(int),
        frame.packet_loss.astype(float)))
    if frame.key.duplicated().any():
        raise ValueError("archived comparison key is not unique")
    return {row.key: row for _, row in frame.iterrows()}


def compare_row(archived, generated):
    key = (generated["run_id"], generated["method"],
           int(generated["message_delay"]), float(generated["packet_loss"]))
    old = archived[key]
    checks = {
        "mechanism_match": old.mechanism == generated["mechanism"],
        "prediction_match":
            int(old.predicted_label) == int(generated["predicted_label"]),
        "probability_match": math.isclose(
            float(old.prediction_probability),
            float(generated["prediction_probability"]), abs_tol=TOL,
            rel_tol=0.0),
        "action_match": old.selected_action == generated["selected_action"],
        "regret_match": math.isclose(
            float(old.wrong_transfer_regret),
            float(generated["wrong_transfer_regret"]), abs_tol=TOL,
            rel_tol=0.0),
        "message_hash_match": old.message_hash == generated["message_hash"],
    }
    return {
        "run_id": generated["run_id"], "seed": generated["seed"],
        "mechanism": generated["mechanism"],
        "method": generated["method"],
        "delay": generated["message_delay"],
        "packet_loss": generated["packet_loss"], **checks,
        "all_match": all(checks.values()),
        "probability_abs_difference": abs(
            float(old.prediction_probability)
            - float(generated["prediction_probability"])),
        "regret_abs_difference": abs(
            float(old.wrong_transfer_regret)
            - float(generated["wrong_transfer_regret"])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output
    traces = output / "traces"
    traces.mkdir(parents=True, exist_ok=True)

    config = load_gate0_config(R1 / "configs/gate1_formal.yaml")
    train = list(range(config["train_seeds"]["start"],
                       config["train_seeds"]["end"] + 1))
    validation = list(range(config["validation_seeds"]["start"],
                            config["validation_seeds"]["end"] + 1))
    dev = list(range(config["development_test_seeds"]["start"],
                     config["development_test_seeds"]["end"] + 1))
    if args.smoke:
        train, validation, dev = train[:2], validation[:1], dev[:1]
    all_seeds = train + validation + dev
    assert_seed_safety(all_seeds)

    frozen_gate1 = json.loads(
        (GATE1 / "gate1_frozen_receiver_models.json").read_text())
    frozen_phase5 = assert_fixed_schemes(json.loads(
        (R0 / "results/phase5_minimal_distributed_evidence/"
         "development_frozen_schemes.json").read_text()))
    f2 = frozen_phase5["F2"]
    scheme = frozen_scheme(f2)
    include_o1 = f2["scheme_definition"]["include_derived_o1"]
    ranges = json.loads(
        (R0 / "results/phase5_minimal_distributed_evidence/"
         "frozen_scheme_manifest.json").read_text())["quantization_ranges"]
    h1 = load_config(R0 / "results/phase3c_redesign/matched_config_h1.yaml")
    h2 = load_config(R0 / "results/phase3c_redesign/matched_config_h2.yaml")
    records = simulate_records(h1, h2, {
        "train": set(train), "validation": set(validation),
        "pilot_test": set(dev)})
    for record in records:
        if record["split"] == "pilot_test":
            record["split"] = "development_test"
            record["run_id"] = record["run_id"].replace(
                "pilot_test_", "development_test_", 1)

    o0_path = traces / "gate1_development_o0.npz"
    np.savez_compressed(
        o0_path, values=np.asarray([row["o0"] for row in records],
                                   dtype=np.int32),
        run_id=np.asarray([row["run_id"] for row in records]),
        seed=np.asarray([row["seed"] for row in records], dtype=np.int32),
        mechanism=np.asarray([row["mechanism"] for row in records]),
        fields=np.asarray([
            "local_queue_length", "head_of_line_wait",
            "local_request_failures", "locally_observed_busy_flag",
            "time_since_last_success", "local_retry_count",
            "recent_local_throughput"]))

    archived = archive_index()
    c3_parameters = load_c3_parameters()
    reject = frozen_gate1["cost_sensitive_reject"]
    h1_wrong = float(reject["h1_wrong_regret_estimate"])
    h2_wrong = float(reject["h2_wrong_regret_estimate"])
    reject_cost = float(reject["reject_cost"])
    conditions = config["primary_conditions"] + config["auxiliary_conditions"]
    equivalence, states, trace_manifest = [], [], []
    record_count = len(records)

    for condition in conditions:
        delay, loss = int(condition["delay"]), float(condition["loss"])
        name = condition_name(delay, loss)
        received_rows, arrival_rows = [], []
        generation_rows, drop_rows = [], []
        hashes = []
        pending_states = []
        for index, record in enumerate(records):
            historical_index = historical_record_index(record)
            ordinary, cost, sent = transmission(
                record["o0"], scheme, ranges, include_o1, delay, loss,
                seed=(record["seed"] * 100003 + delay * 1009
                      + int(loss * 1000) + historical_index))
            groups = {
                name_: selected_receiver_features(
                    sent, (name_,), initial_arrival_age=delay)
                for name_ in ("age", "mask", "silence")}
            m1_features = dict(ordinary)
            m4_features = dict(ordinary)
            for values in groups.values():
                m4_features.update(values)
            p0 = predict_frozen(f2["classifier"], ordinary)
            p1 = serialized_probability(
                frozen_gate1["augmentation_baseline"], m1_features)
            p4 = serialized_probability(
                frozen_gate1["age_aware_model"], m4_features)
            example = {
                "record": record, "delay": delay, "loss": loss,
                "message_hash": message_hash(sent), "cost": cost}
            generated = [
                prediction_row(example, "M0_ZERO_DELAY_FROZEN", p0,
                               h1_wrong, h2_wrong, reject_cost, 5.10),
                prediction_row(example, "M1_CHANNEL_AUGMENTATION", p1,
                               h1_wrong, h2_wrong, reject_cost, 5.10),
                prediction_row(example, "M4_AGE_MASK_SILENCE", p4,
                               h1_wrong, h2_wrong, reject_cost, 5.10),
                prediction_row(example, "M5_COST_SENSITIVE_REJECT", p4,
                               h1_wrong, h2_wrong, reject_cost, 5.10),
            ]
            equivalence.extend(compare_row(archived, row)
                               for row in generated)
            received_rows.append(np.asarray(sent["received"], np.float32))
            arrival_rows.append(np.asarray(sent["arrival_mask"], bool))
            generation_rows.append(np.asarray(sent["generation_mask"], bool))
            drop_rows.append(np.asarray(sent["drop_mask"], bool))
            hashes.append(example["message_hash"])

            if delay in (1, 2) and loss == 0.0:
                decision_tick = len(record["o0"]) - 1
                last, ages = last_received_state(sent, decision_tick, delay)
                probability_h1 = 1.0 - p4
                score = float(recover_score(
                    [p4], h1_wrong, h2_wrong)[0])
                fold = (int(record["seed"]) - 3000) % 5
                accept, selected = c3_decision(
                    p4, score, c3_parameters[fold])
                pending_states.append({
                    "run_id": record["run_id"], "seed": record["seed"],
                    "mechanism": record["mechanism"], "delay": delay,
                    "packet_loss": loss,
                    "config_id": "PHASE3C_FIXED_MATCHED_CONFIG_PAIR",
                    "decision_timestamp": decision_tick,
                    "outer_fold": fold,
                    "received_message_state_at_decision":
                        json.dumps(np.asarray(
                            sent["received"][decision_tick]).tolist()),
                    "per_node_O0_at_decision":
                        json.dumps(record["o0"][decision_tick].tolist()),
                    "per_node_message_age_at_decision":
                        json.dumps(ages.tolist()),
                    "per_node_last_received_value":
                        json.dumps(last.tolist()),
                    "mechanism_probability_H1": probability_h1,
                    "mechanism_probability_H2": p4,
                    "predicted_mechanism":
                        H2_SUPPRESSION if p4 >= .5 else H1,
                    "rejection_score": score,
                    "C3_accept_or_reject": accept,
                    "selected_action": selected,
                    "oracle_action": record["oracle_action"],
                    "wrong_transfer_regret":
                        record["wrong_transfer_regret"]
                        if selected not in ("REJECT", record["oracle_action"])
                        else 0.0,
                    "potential_wrong_transfer_regret":
                        record["wrong_transfer_regret"],
                    "tick_trace_reference": "",
                    "trace_sha256": "",
                    "message_hash": example["message_hash"],
                })

        trace_path = traces / f"f2_{name}.npz"
        np.savez_compressed(
            trace_path,
            received=np.asarray(received_rows, np.float32),
            arrival_mask=np.asarray(arrival_rows, bool),
            generation_mask=np.asarray(generation_rows, bool),
            drop_mask=np.asarray(drop_rows, bool),
            message_hash=np.asarray(hashes),
            run_id=np.asarray([row["run_id"] for row in records]))
        trace_hash = sha256(trace_path)
        trace_reference = (
            str(trace_path.relative_to(R2))
            if trace_path.is_relative_to(R2) else str(trace_path))
        o0_reference = (
            str(o0_path.relative_to(R2))
            if o0_path.is_relative_to(R2) else str(o0_path))
        trace_manifest.append({
            "condition": name, "delay": delay, "packet_loss": loss,
            "runs": record_count,
            "trace_reference": trace_reference,
            "trace_sha256": trace_hash,
            "o0_reference": o0_reference,
            "o0_sha256": sha256(o0_path),
        })
        for index, row in enumerate(pending_states):
            row["tick_trace_reference"] = (
                f"{trace_reference}#index={index}")
            row["trace_sha256"] = trace_hash
            states.append(row)

    equivalence_fields = list(equivalence[0])
    write_csv(output / "02_instrumented_replay_equivalence.csv",
              equivalence, equivalence_fields)
    write_csv(output / "03_decision_state_manifest.csv", states)
    write_csv(output / "01_replay_manifest.csv", trace_manifest)
    all_match = all(row["all_match"] for row in equivalence)
    decision = (
        "G1A_REPLAY_DATA_RECOVERED" if all_match and states
        else "INSTRUMENTED_REPLAY_NOT_EQUIVALENT")
    (output / "06_recovery_decision.txt").write_text(decision + "\n")
    summary = {
        "status": decision, "smoke": args.smoke,
        "development_seed_min": min(all_seeds),
        "development_seed_max": max(all_seeds),
        "development_seed_count": len(all_seeds),
        "mechanism_runs": len(records),
        "comparison_rows": len(equivalence),
        "comparison_mismatches":
            sum(not row["all_match"] for row in equivalence),
        "decision_state_rows": len(states),
        "decision_timestamp": 499,
        "models_refit": False,
        "sealed_seeds_loaded": False,
        "r0_final_seeds_loaded": False,
        "original_prediction_archive_sha256": sha256(
            GATE1 / "per_run_predictions.csv"),
        "frozen_receiver_manifest_sha256": sha256(
            GATE1 / "gate1_frozen_receiver_models.json"),
    }
    (output / "07_recovery_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not all_match:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
