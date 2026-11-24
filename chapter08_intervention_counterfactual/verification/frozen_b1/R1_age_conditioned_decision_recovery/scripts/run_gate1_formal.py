#!/usr/bin/env python3
"""Formal development-only Gate 1 age-conditioning and ablation audit."""

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
R0 = HERE.parent / "R0_hidden_mechanism_audit"
sys.path[:0] = [
    str(HERE / "src"), str(HERE / "scripts"),
    str(R0), str(R0 / "scripts")]

from decision import transfer_or_reject
from gate0_config import load_gate0_config
from protocol import R0_FINAL_HOLDOUT, assert_fixed_schemes
from receiver_features import selected_receiver_features
from run_gate0_pilot import (
    observation_tensor, simulate_records)
from run_phase4a_audit import metric_values
from run_phase5_final import transmission
from src.config import load_config
from src.phase5_final import frozen_scheme, predict_frozen
from src.simple_models import LogisticModel
from src.simulator import H1
from src.suppression_simulator import H2_SUPPRESSION


ABLATIONS = {
    "M1_CHANNEL_AUGMENTATION": (),
    "M2_AGE_ONLY": ("age",),
    "MASK_ONLY": ("mask",),
    "SILENCE_ONLY": ("silence",),
    "M3_AGE_MASK": ("age", "mask"),
    "AGE_SILENCE": ("age", "silence"),
    "MASK_SILENCE": ("mask", "silence"),
    "M4_AGE_MASK_SILENCE": ("age", "mask", "silence"),
}


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fields:
        raise ValueError(f"empty output: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_range(mapping):
    return list(range(int(mapping["start"]), int(mapping["end"]) + 1))


def assert_gate1_isolation(config):
    groups = {
        "train": set(seed_range(config["train_seeds"])),
        "validation": set(seed_range(config["validation_seeds"])),
        "development_test":
            set(seed_range(config["development_test_seeds"])),
        "sealed_final_test":
            set(seed_range(config["sealed_final_test_seeds"])),
    }
    forbidden = set(seed_range(config["forbidden_r0_seeds"]))
    assert forbidden == set(R0_FINAL_HOLDOUT)
    for name, values in groups.items():
        if values & forbidden:
            raise ValueError(f"{name} uses R0 final holdout")
    names = list(groups)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if groups[left] & groups[right]:
                raise ValueError(f"overlap: {left}, {right}")
    return groups


def fit_receiver(feature_maps, labels, l2):
    names = sorted(feature_maps[0])
    if any(sorted(row) != names for row in feature_maps):
        raise ValueError("feature schema changed")
    x = np.asarray([[row[name] for name in names]
                    for row in feature_maps], float)
    mean, scale = np.mean(x, axis=0), np.std(x, axis=0)
    scale[scale == 0] = 1.0
    model = LogisticModel(l2=l2).fit((x - mean) / scale, labels)
    return {"feature_order": names, "mean": mean, "scale": scale,
            "model": model, "l2": l2}


def predict_receiver(artifact, feature_map):
    x = np.asarray([[feature_map[name]
                     for name in artifact["feature_order"]]], float)
    z = (x - artifact["mean"]) / artifact["scale"]
    return float(artifact["model"].predict_proba(z)[0])


def serialize_receiver(artifact):
    return {
        "model": "logistic_regression", "l2": artifact["l2"],
        "feature_order": artifact["feature_order"],
        "scaler_mean": artifact["mean"].tolist(),
        "scaler_scale": artifact["scale"].tolist(),
        "coefficients_intercept_first":
            artifact["model"].weights.tolist(),
        "decision_threshold": .5,
    }


def parameter_matched(ordinary, target_dimension):
    output = dict(ordinary)
    names = sorted(ordinary)
    needed = target_dimension - len(output)
    for index in range(needed):
        output[f"parameter_control_duplicate_{index:03d}"] = float(
            ordinary[names[index % len(names)]])
    if len(output) != target_dimension:
        raise ValueError("parameter-count control mismatch")
    return output


def message_hash(sent):
    digest = hashlib.sha256()
    for name in ("received", "payload_by_tick", "protocol_by_tick"):
        digest.update(np.asarray(sent[name]).tobytes())
    return digest.hexdigest()


def construct_examples(records, frozen, ranges, conditions):
    scheme = frozen_scheme(frozen)
    include_o1 = frozen["scheme_definition"]["include_derived_o1"]
    output = []
    for record_index, record in enumerate(records):
        for condition in conditions:
            delay, loss = int(condition["delay"]), float(condition["loss"])
            ordinary, cost, sent = transmission(
                record["o0"], scheme, ranges, include_o1, delay, loss,
                seed=(record["seed"] * 100003 + delay * 1009
                      + int(loss * 1000) + record_index))
            groups = {
                name: selected_receiver_features(
                    sent, (name,), initial_arrival_age=delay)
                for name in ("age", "mask", "silence")}
            all_map = dict(ordinary)
            for values in groups.values():
                all_map.update(values)
            features = {}
            for method, selected in ABLATIONS.items():
                value = dict(ordinary)
                for name in selected:
                    value.update(groups[name])
                features[method] = value
            features["PARAMETER_MATCHED_ORDINARY"] = parameter_matched(
                ordinary, len(all_map))
            output.append({
                "record": record, "delay": delay, "loss": loss,
                "ordinary": ordinary, "features": features,
                "cost": cost, "message_hash": message_hash(sent),
            })
    return output


def train_artifacts(examples, l2):
    output = {}
    for method in list(ABLATIONS) + ["PARAMETER_MATCHED_ORDINARY"]:
        rows = [row for row in examples
                if row["record"]["split"] == "train"]
        output[method] = fit_receiver(
            [row["features"][method] for row in rows],
            np.asarray([row["record"]["true_label"] for row in rows], int),
            l2)
    return output


def prediction_row(example, method, probability, h1_wrong, h2_wrong,
                   reject_cost, nominal_bytes):
    record = example["record"]
    predicted = int(probability >= .5)
    classification_error = int(predicted != record["true_label"])
    decision, selected = "TRANSFER", (
        "SYNC_ACTION" if predicted else "CAPACITY_ACTION")
    wrong_transfer = classification_error
    rejection_cost = 0.0
    if method == "M5_COST_SENSITIVE_REJECT":
        choice = transfer_or_reject(
            probability, h1_wrong, h2_wrong, reject_cost)
        decision, selected = choice["decision"], choice["selected_action"]
        if decision == "REJECT":
            wrong_transfer = 0
            rejection_cost = reject_cost
    wrong_regret = wrong_transfer * record["wrong_transfer_regret"]
    return {
        "run_id": record["run_id"], "seed": record["seed"],
        "split": record["split"], "mechanism": record["mechanism"],
        "method": method, "message_delay": example["delay"],
        "packet_loss": example["loss"],
        "message_hash": example["message_hash"],
        "true_label": record["true_label"],
        "predicted_label": predicted,
        "prediction_probability": probability,
        "classification_error": classification_error,
        "decision": decision, "selected_action": selected,
        "wrong_transfer": wrong_transfer,
        "wrong_transfer_regret": wrong_regret,
        "rejection_cost": rejection_cost,
        "total_decision_cost": wrong_regret + rejection_cost,
        "protocol_bytes_per_tick": nominal_bytes,
        "effective_received_bytes_per_tick":
            example["cost"]["effective_received_bytes_per_tick"],
    }


def all_predictions(examples, artifacts, frozen, h1_wrong, h2_wrong,
                    reject_cost, nominal_bytes):
    rows = []
    for example in examples:
        frozen_p = predict_frozen(
            frozen["classifier"], example["ordinary"])
        rows.append(prediction_row(
            example, "M0_ZERO_DELAY_FROZEN", frozen_p,
            h1_wrong, h2_wrong, reject_cost, nominal_bytes))
        probabilities = {}
        for method, artifact in artifacts.items():
            probability = predict_receiver(
                artifact, example["features"][method])
            probabilities[method] = probability
            rows.append(prediction_row(
                example, method, probability, h1_wrong, h2_wrong,
                reject_cost, nominal_bytes))
        rows.append(prediction_row(
            example, "M5_COST_SENSITIVE_REJECT",
            probabilities["M4_AGE_MASK_SILENCE"],
            h1_wrong, h2_wrong, reject_cost, nominal_bytes))
    return rows


def balanced_accuracy(rows, error_name="classification_error"):
    recalls = []
    for truth in (0, 1):
        values = [row for row in rows if row["true_label"] == truth]
        recalls.append(1 - statistics.mean(
            int(row[error_name]) for row in values))
    return statistics.mean(recalls)


def aggregate(rows, wrong_reference):
    output = []
    keys = sorted({
        (row["split"], row["method"], row["message_delay"],
         row["packet_loss"]) for row in rows})
    for split, method, delay, loss in keys:
        values = [row for row in rows
                  if (row["split"], row["method"], row["message_delay"],
                      row["packet_loss"]) == (split, method, delay, loss)]
        y = np.asarray([row["true_label"] for row in values], int)
        p = np.asarray([row["prediction_probability"]
                        for row in values], float)
        metrics = metric_values(y, p)
        mechanism = {}
        for label, name in ((0, "h1"), (1, "h2")):
            selected = [row for row in values
                        if row["true_label"] == label]
            mechanism[f"{name}_error_rate"] = statistics.mean(
                row["classification_error"] for row in selected)
            mechanism[f"{name}_wrong_transfer_rate"] = statistics.mean(
                row["wrong_transfer"] for row in selected)
            mechanism[f"{name}_regret"] = statistics.mean(
                row["wrong_transfer_regret"] for row in selected)
        transferred = [row for row in values
                       if row["decision"] == "TRANSFER"]
        accepted_ba = (
            balanced_accuracy(transferred)
            if transferred
            and {row["true_label"] for row in transferred} == {0, 1}
            else math.nan)
        mean_regret = statistics.mean(
            row["wrong_transfer_regret"] for row in values)
        output.append({
            "split": split, "method": method,
            "message_delay": delay, "packet_loss": loss, **metrics,
            **mechanism,
            "mean_wrong_transfer_regret": mean_regret,
            "mean_total_decision_cost": statistics.mean(
                row["total_decision_cost"] for row in values),
            "utility_gain_retention":
                1 - mean_regret / wrong_reference,
            "transfer_coverage": len(transferred) / len(values),
            "accepted_case_balanced_accuracy": accepted_ba,
            "protocol_bytes_per_tick": statistics.mean(
                row["protocol_bytes_per_tick"] for row in values),
            "run_count": len(values),
        })
    return output


def paired_rows(rows, method_a, method_b, delay, loss):
    selected = [row for row in rows
                if row["split"] == "development_test"
                and row["message_delay"] == delay
                and row["packet_loss"] == loss
                and row["method"] in (method_a, method_b)]
    by_key = {}
    for row in selected:
        by_key.setdefault((row["seed"], row["mechanism"]), {})[
            row["method"]] = row
    if any(set(value) != {method_a, method_b}
           for value in by_key.values()):
        raise ValueError("paired alignment failure")
    return [(value[method_a], value[method_b])
            for value in by_key.values()]


def bootstrap_comparison(pairs, replicates, seed):
    seeds = sorted({left["seed"] for left, _ in pairs})
    grouped = {
        value: [(left, right) for left, right in pairs
                if left["seed"] == value] for value in seeds}
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        picked = rng.choices(seeds, k=len(seeds))
        sample = [pair for value in picked for pair in grouped[value]]
        left = [pair[0] for pair in sample]
        right = [pair[1] for pair in sample]
        samples.append({
            "ba_difference":
                balanced_accuracy(right) - balanced_accuracy(left),
            "regret_difference": statistics.mean(
                row["wrong_transfer_regret"] for row in right)
                - statistics.mean(
                    row["wrong_transfer_regret"] for row in left),
        })
    return samples


def bootstrap_summary(comparison, delay, samples):
    output = []
    for metric in ("ba_difference", "regret_difference"):
        values = sorted(row[metric] for row in samples)
        output.append({
            "comparison": comparison, "message_delay": delay,
            "metric": metric, "replicates": len(values),
            "mean": statistics.mean(values),
            "ci95_low": values[int(.025 * len(values))],
            "ci95_high": values[min(len(values) - 1,
                                    int(.975 * len(values)))],
        })
    return output


def permutation_test(pairs, replicates, seed):
    observed_ba = (
        balanced_accuracy([right for _, right in pairs])
        - balanced_accuracy([left for left, _ in pairs]))
    observed_regret = (
        statistics.mean(right["wrong_transfer_regret"]
                        for _, right in pairs)
        - statistics.mean(left["wrong_transfer_regret"]
                          for left, _ in pairs))
    seeds = sorted({left["seed"] for left, _ in pairs})
    rng = random.Random(seed)
    ba_null, regret_null = [], []
    for _ in range(replicates):
        swapped = {
            value: rng.random() < .5 for value in seeds}
        left_rows, right_rows = [], []
        for left, right in pairs:
            if swapped[left["seed"]]:
                left, right = right, left
            left_rows.append(left)
            right_rows.append(right)
        ba_null.append(
            balanced_accuracy(right_rows) - balanced_accuracy(left_rows))
        regret_null.append(
            statistics.mean(row["wrong_transfer_regret"]
                            for row in right_rows)
            - statistics.mean(row["wrong_transfer_regret"]
                              for row in left_rows))
    return {
        "observed_ba_difference": observed_ba,
        "ba_two_sided_p": (
            1 + sum(abs(value) >= abs(observed_ba) for value in ba_null)
        ) / (replicates + 1),
        "observed_regret_difference": observed_regret,
        "regret_two_sided_p": (
            1 + sum(abs(value) >= abs(observed_regret)
                    for value in regret_null)
        ) / (replicates + 1),
        "replicates": replicates,
    }


def mcnemar(pairs):
    b = sum(left["classification_error"]
            and not right["classification_error"] for left, right in pairs)
    c = sum(right["classification_error"]
            and not left["classification_error"] for left, right in pairs)
    n = b + c
    if not n:
        p = 1.0
    else:
        tail = sum(math.comb(n, index) for index in range(0, min(b, c) + 1))
        p = min(1.0, 2 * tail / (2 ** n))
    return {"m1_wrong_m4_correct": b, "m1_correct_m4_wrong": c,
            "discordant": n, "exact_two_sided_p": p}


def seed_influence(pairs):
    seeds = sorted({left["seed"] for left, _ in pairs})
    output = []
    full_ba = (
        balanced_accuracy([right for _, right in pairs])
        - balanced_accuracy([left for left, _ in pairs]))
    full_regret = (
        statistics.mean(right["wrong_transfer_regret"]
                        for _, right in pairs)
        - statistics.mean(left["wrong_transfer_regret"]
                          for left, _ in pairs))
    for excluded in seeds:
        remaining = [pair for pair in pairs
                     if pair[0]["seed"] != excluded]
        ba = (
            balanced_accuracy([right for _, right in remaining])
            - balanced_accuracy([left for left, _ in remaining]))
        regret = (
            statistics.mean(right["wrong_transfer_regret"]
                            for _, right in remaining)
            - statistics.mean(left["wrong_transfer_regret"]
                              for left, _ in remaining))
        output.append({
            "excluded_seed": excluded,
            "ba_difference": ba, "regret_difference": regret,
            "ba_influence": ba - full_ba,
            "regret_influence": regret - full_regret,
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_gate0_config(args.config)
    groups = assert_gate1_isolation(config)
    if args.smoke:
        groups = {
            **groups,
            "train": set(sorted(groups["train"])[:4]),
            "validation": set(sorted(groups["validation"])[:3]),
            "development_test":
                set(sorted(groups["development_test"])[:3]),
        }
    if groups["sealed_final_test"] & set().union(
            groups["train"], groups["validation"],
            groups["development_test"]):
        raise RuntimeError("sealed final test opened")
    output = HERE / "results/gate1_formal_age_necessity"
    if args.smoke:
        output = Path("/tmp/b1_r1_gate1_smoke")
        output.mkdir(parents=True, exist_ok=True)
    r0_output = R0 / "results/phase5_minimal_distributed_evidence"
    frozen_manifest = json.loads(
        (r0_output / "development_frozen_schemes.json").read_text())
    fixed = assert_fixed_schemes(frozen_manifest)
    if config["fixed_primary_scheme"] != "F2":
        raise ValueError("Gate 1 primary scheme changed")
    ranges = json.loads(
        (r0_output / "frozen_scheme_manifest.json").read_text())[
            "quantization_ranges"]
    h1 = load_config(R0 / "results/phase3c_redesign/matched_config_h1.yaml")
    h2 = load_config(R0 / "results/phase3c_redesign/matched_config_h2.yaml")
    records = simulate_records(h1, h2, {
        "train": groups["train"],
        "validation": groups["validation"],
        "pilot_test": groups["development_test"],
    })
    for record in records:
        if record["split"] == "pilot_test":
            record["split"] = "development_test"
            record["run_id"] = record["run_id"].replace(
                "pilot_test_", "development_test_", 1)
    conditions = (
        config["primary_conditions"] + config["auxiliary_conditions"])
    f2_examples = construct_examples(
        records, fixed["F2"], ranges, conditions)
    f2_artifacts = train_artifacts(
        f2_examples, config["receiver"]["l2"])
    train_records = [row for row in records if row["split"] == "train"]
    h1_wrong = statistics.mean(
        row["wrong_transfer_regret"] for row in train_records
        if row["mechanism"] == H1)
    h2_wrong = statistics.mean(
        row["wrong_transfer_regret"] for row in train_records
        if row["mechanism"] == H2_SUPPRESSION)
    f2_predictions = all_predictions(
        f2_examples, f2_artifacts, fixed["F2"],
        h1_wrong, h2_wrong, config["receiver"]["reject_cost"], 5.10)
    wrong_reference = statistics.mean(
        row["wrong_transfer_regret"] for row in records
        if row["split"] == "development_test")
    summaries = aggregate(f2_predictions, wrong_reference)

    # F4 is a fixed negative control with only augmentation and full explicit RX.
    f4_examples = construct_examples(
        records, fixed["F4"], ranges, conditions)
    f4_artifacts = train_artifacts(
        f4_examples, config["receiver"]["l2"])
    f4_all = all_predictions(
        f4_examples, f4_artifacts, fixed["F4"],
        h1_wrong, h2_wrong, config["receiver"]["reject_cost"], 4.49)
    f4_predictions = [
        row for row in f4_all
        if row["method"] in (
            "M1_CHANNEL_AUGMENTATION", "M4_AGE_MASK_SILENCE")]
    f4_summary = aggregate(f4_predictions, wrong_reference)

    comparisons, bootstrap_rows, permutation_rows = [], [], []
    mcnemar_rows, influence_rows = [], []
    formal_replicates = config["statistics"]["bootstrap_replicates"]
    permutation_replicates = config["statistics"]["permutation_replicates"]
    if args.smoke:
        formal_replicates = permutation_replicates = 50
    for delay in (1, 2):
        for right_method in (
                "M4_AGE_MASK_SILENCE", "M5_COST_SENSITIVE_REJECT"):
            pairs = paired_rows(
                f2_predictions, "M1_CHANNEL_AUGMENTATION",
                right_method, delay, 0.0)
            comparison = f"{right_method}-M1"
            left, right = [p[0] for p in pairs], [p[1] for p in pairs]
            row = {
                "comparison": comparison, "message_delay": delay,
                "paired_run_count": len(pairs),
                "paired_ba_difference":
                    balanced_accuracy(right) - balanced_accuracy(left),
                "paired_regret_difference": statistics.mean(
                    value["wrong_transfer_regret"] for value in right)
                    - statistics.mean(
                        value["wrong_transfer_regret"] for value in left),
            }
            comparisons.append(row)
            samples = bootstrap_comparison(
                pairs, formal_replicates, 1700 + delay)
            bootstrap_rows.extend(
                bootstrap_summary(comparison, delay, samples))
            permutation_rows.append({
                "comparison": comparison, "message_delay": delay,
                **permutation_test(
                    pairs, permutation_replicates, 2200 + delay)})
            mcnemar_rows.append({
                "comparison": comparison, "message_delay": delay,
                **mcnemar(pairs)})
            for influence in seed_influence(pairs):
                influence_rows.append({
                    "comparison": comparison, "message_delay": delay,
                    **influence})

    primary = [row for row in summaries
               if row["split"] == "development_test"
               and row["packet_loss"] == 0
               and row["message_delay"] in (1, 2)]
    gate_pass = True
    for delay in (1, 2):
        m1 = next(row for row in primary
                  if row["message_delay"] == delay
                  and row["method"] == "M1_CHANNEL_AUGMENTATION")
        m4 = next(row for row in primary
                  if row["message_delay"] == delay
                  and row["method"] == "M4_AGE_MASK_SILENCE")
        m5 = next(row for row in primary
                  if row["message_delay"] == delay
                  and row["method"] == "M5_COST_SENSITIVE_REJECT")
        parameter = next(row for row in primary
                         if row["message_delay"] == delay
                         and row["method"] == "PARAMETER_MATCHED_ORDINARY")
        ba_boot = next(row for row in bootstrap_rows
                       if row["comparison"] ==
                       "M4_AGE_MASK_SILENCE-M1"
                       and row["message_delay"] == delay
                       and row["metric"] == "ba_difference")
        regret_boot = next(row for row in bootstrap_rows
                           if row["comparison"] ==
                           "M4_AGE_MASK_SILENCE-M1"
                           and row["message_delay"] == delay
                           and row["metric"] == "regret_difference")
        h2_reduction = (
            (m1["h2_regret"] - m4["h2_regret"]) / m1["h2_regret"]
            if m1["h2_regret"] else 0)
        parameter_replication = (
            parameter["balanced_accuracy"]
            >= m4["balanced_accuracy"]
               - config["gate"]["parameter_control_tolerance"])
        gate_pass &= (
            m4["balanced_accuracy"] - m1["balanced_accuracy"]
            >= config["gate"]["minimum_ba_improvement"]
            and ba_boot["ci95_low"] > 0
            and h2_reduction
            >= config["gate"]["minimum_h2_regret_reduction_fraction"]
            and regret_boot["ci95_high"] < 0
            and m4["protocol_bytes_per_tick"] == 5.10
            and m5["transfer_coverage"]
            >= config["gate"]["minimum_reject_coverage"]
            and m5["h2_wrong_transfer_rate"]
            < m4["h2_wrong_transfer_rate"]
            and not parameter_replication)

    m4_differences = [
        row["paired_ba_difference"] for row in comparisons
        if row["comparison"] == "M4_AGE_MASK_SILENCE-M1"]
    if gate_pass:
        status = "GATE1_AGE_CONDITIONING_NECESSARY"
    elif all(value > 0 for value in m4_differences):
        status = "GATE1_INCONCLUSIVE"
    elif all(abs(value) < .03 for value in m4_differences):
        status = "GATE1_CHANNEL_AUGMENTATION_SUFFICIENT"
    else:
        status = "GATE1_CURRENT_FORMULATION_FAILED"

    f4_primary = [row for row in f4_summary
                  if row["split"] == "development_test"
                  and row["packet_loss"] == 0
                  and row["message_delay"] in (1, 2)]
    f4_gains = []
    for delay in (1, 2):
        base = next(row for row in f4_primary
                    if row["message_delay"] == delay
                    and row["method"] == "M1_CHANNEL_AUGMENTATION")
        age = next(row for row in f4_primary
                   if row["message_delay"] == delay
                   and row["method"] == "M4_AGE_MASK_SILENCE")
        f4_gains.append(age["balanced_accuracy"]
                        - base["balanced_accuracy"])
    interpretation_labels = []
    if gate_pass and max(f4_gains) < .03:
        interpretation_labels.append("GATE1_AGE_SIGNAL_F2_SPECIFIC")

    frozen_models = {
        "status": status, "interpretation_labels": interpretation_labels,
        "selection_data": "development_only",
        "fixed_scheme": "F2_PERIOD_20_EXPLICIT_UNKNOWN",
        "protocol_bytes_per_tick": 5.10,
        "augmentation_baseline":
            serialize_receiver(f2_artifacts["M1_CHANNEL_AUGMENTATION"]),
        "age_aware_model":
            serialize_receiver(f2_artifacts["M4_AGE_MASK_SILENCE"]),
        "cost_sensitive_reject": {
            "posterior_model": "age_aware_model",
            "reject_cost": config["receiver"]["reject_cost"],
            "h1_wrong_regret_estimate": h1_wrong,
            "h2_wrong_regret_estimate": h2_wrong,
        },
    }
    sealed_manifest = {
        "sealed": True, "evaluated": False,
        "seeds": seed_range(config["sealed_final_test_seeds"]),
        "forbidden_r0_seeds": seed_range(config["forbidden_r0_seeds"]),
        "development_seeds": sorted(
            groups["train"] | groups["validation"]
            | groups["development_test"]),
        "models_frozen_before_holdout": True,
    }
    if not args.smoke:
        write_csv(output / "per_run_predictions.csv",
                  f2_predictions + f4_predictions)
        write_csv(output / "paired_comparisons.csv", comparisons)
        write_csv(output / "feature_ablation.csv", summaries)
        write_csv(output / "f4_negative_control.csv", f4_summary)
        write_csv(output / "bootstrap_results.csv", bootstrap_rows)
        write_csv(output / "paired_permutation_results.csv",
                  permutation_rows)
        write_csv(output / "mcnemar_results.csv", mcnemar_rows)
        write_csv(output / "seed_influence.csv", influence_rows)
        coverage = [row for row in summaries
                    if row["method"] == "M5_COST_SENSITIVE_REJECT"]
        write_csv(output / "coverage_regret_tradeoff.csv", coverage)
        (output / "gate1_frozen_receiver_models.json").write_text(
            json.dumps(frozen_models, indent=2, sort_keys=True) + "\n")
        (output / "gate1_sealed_holdout_manifest.json").write_text(
            json.dumps(sealed_manifest, indent=2, sort_keys=True) + "\n")
        report = f"""# B1 R1 Gate 1 formal development audit

Formal Gate 1 status: `{status}`

Interpretation labels: `{interpretation_labels}`

F2 remained fixed at 5.10 protocol bytes/tick. All receiver methods used the
same message hashes for each run and channel condition. R0 seeds 120--149 were
forbidden and R1 final-test seeds 2000--2029 remain sealed.

See the paired, bootstrap, permutation, McNemar, ablation, negative-control,
coverage, and seed-influence CSV files for the formal development evidence.

Gate 1 stops here. The sealed final test was not opened.
"""
        (output / "gate1_report.md").write_text(report)
    else:
        write_csv(output / "smoke_summary.csv", summaries)
    print(status if not args.smoke else f"SMOKE:{status}")


if __name__ == "__main__":
    main()
