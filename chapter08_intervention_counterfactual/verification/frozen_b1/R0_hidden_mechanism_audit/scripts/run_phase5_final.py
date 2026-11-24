#!/usr/bin/env python3
"""Phase 5.5--5.7 frozen holdout, channel, and configuration audit."""

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from run_phase4a_audit import metric_values
from run_phase5_3_4 import (
    bootstrap_summary, maximum_silence_duration, paired_seed_bootstrap,
    seed_influence)
from src.config import load_config
from src.phase5_communication import (
    CommunicationScheme, central_summary, transmit_run)
from src.phase5_event import (
    EventScheme, event_central_summary, transmit_event_run)
from src.phase5_final import (
    fit_frozen_family, frozen_scheme, predict_frozen, predict_refit,
    received_age_statistics)
from src.schemas import O0_FIELDS
from src.simulator import H1, simulate
from src.suppression_simulator import H2_SUPPRESSION, simulate_suppression


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fields:
        raise ValueError(f"empty output: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True,
        capture_output=True, check=True)
    return result.stdout.strip()


def transmission(o0_run, scheme, ranges, include_o1, delay=0, loss=0.0,
                 seed=0):
    if isinstance(scheme, EventScheme):
        sent = transmit_event_run(
            o0_run, scheme, message_delay=delay, packet_loss=loss, seed=seed)
        features = event_central_summary(sent, scheme)
        received_messages = int(np.sum(np.any(
            np.isfinite(sent["received"]), axis=2)))
    else:
        altered = dataclasses.replace(
            scheme, message_delay=int(delay), packet_loss=float(loss))
        sent = transmit_run(o0_run, altered, ranges, seed=seed)
        features = central_summary(sent, altered, include_o1)
        received_messages = int(np.sum(np.any(
            sent["arrival_mask"], axis=2)))
    messages = sent["message_count_by_tick"]
    generated_messages = int(np.sum(messages))
    protocol = sent["protocol_by_tick"]
    payload = sent["payload_by_tick"]
    received_fraction = (
        received_messages / generated_messages if generated_messages else 0)
    age_mean, age_p90 = received_age_statistics(sent)
    if not isinstance(scheme, EventScheme) and received_messages:
        # Periodic messages carry their generation timestamp. Their age on
        # receipt is exactly the configured channel delay.
        age_mean = age_p90 = float(delay)
    return features, {
        "payload_bytes_total": int(np.sum(payload)),
        "protocol_bytes_total": int(np.sum(protocol)),
        "mean_payload_bytes_per_tick": float(np.mean(payload)),
        "mean_protocol_bytes_per_tick": float(np.mean(protocol)),
        "p90_protocol_bytes_per_tick": float(np.quantile(protocol, .9)),
        "max_bytes_per_tick": int(np.max(protocol)),
        "message_count": generated_messages,
        "received_message_count": received_messages,
        "effective_received_bytes_per_tick":
            float(np.mean(protocol)) * received_fraction,
        "maximum_silence_duration": maximum_silence_duration(messages),
        "message_age_mean": age_mean,
        "message_age_p90": age_p90,
    }, sent


def evaluate_indices(o0, manifest, outcomes, indices, frozen, ranges,
                     delay=0, loss=0.0, condition="holdout"):
    scheme = frozen_scheme(frozen)
    include_o1 = frozen["scheme_definition"]["include_derived_o1"]
    rows = []
    for index in indices:
        features, cost, _ = transmission(
            o0[index], scheme, ranges, include_o1, delay, loss,
            seed=930000 + index + 1009 * int(delay)
                 + int(round(loss * 100)) * 100003)
        probability = predict_frozen(frozen["classifier"], features)
        truth = int(manifest[index]["mechanism"] != H1)
        prediction = int(probability >= .5)
        wrong = int(prediction != truth)
        regret = float(outcomes[index]["wrong_transfer_regret"]) * wrong
        rows.append({
            "run_id": manifest[index]["run_id"],
            "seed": int(manifest[index]["seed"]),
            "mechanism": manifest[index]["mechanism"],
            "scheme_id": frozen["scheme_id"],
            "freeze_id": frozen["freeze_id"],
            "condition": condition,
            "message_delay": delay, "packet_loss": loss,
            "true_label": truth, "predicted_label": prediction,
            "prediction_probability": probability,
            "selected_action":
                "CAPACITY_ACTION" if prediction == 0 else "SYNC_ACTION",
            "oracle_action": outcomes[index]["oracle_action"],
            "wrong_action": wrong,
            "wrong_transfer_regret": regret,
            **cost,
        })
    return rows


def summarize(rows, wrong_reference, full_regret):
    y = np.asarray([int(row["true_label"]) for row in rows])
    p = np.asarray([float(row["prediction_probability"]) for row in rows])
    metrics = metric_values(y, p)
    regrets = np.asarray([
        float(row["wrong_transfer_regret"]) for row in rows])
    mean_regret = float(np.mean(regrets))
    gain_denominator = wrong_reference - full_regret
    return {
        **metrics,
        "wrong_action_rate":
            statistics.mean(int(row["wrong_action"]) for row in rows),
        "mean_wrong_transfer_regret": mean_regret,
        "median_wrong_transfer_regret": float(np.median(regrets)),
        "p90_wrong_transfer_regret": float(np.quantile(regrets, .9)),
        "mean_payload_bytes_per_tick": statistics.mean(
            float(row["mean_payload_bytes_per_tick"]) for row in rows),
        "mean_protocol_bytes_per_tick": statistics.mean(
            float(row["mean_protocol_bytes_per_tick"]) for row in rows),
        "p90_protocol_bytes_per_tick": statistics.mean(
            float(row["p90_protocol_bytes_per_tick"]) for row in rows),
        "total_bytes_per_run": statistics.mean(
            float(row["protocol_bytes_total"]) for row in rows),
        "effective_received_bytes_per_tick": statistics.mean(
            float(row["effective_received_bytes_per_tick"]) for row in rows),
        "message_age_mean": _nanmean(
            [row["message_age_mean"] for row in rows]),
        "message_age_p90": _nanmean(
            [row["message_age_p90"] for row in rows]),
        "maximum_silence_duration": max(
            int(row["maximum_silence_duration"]) for row in rows),
        "strict_regret_fraction":
            mean_regret / wrong_reference if wrong_reference else math.nan,
        "utility_gain_retention":
            ((wrong_reference - mean_regret) / gain_denominator
             if gain_denominator else math.nan),
        "identification_absolute_pass":
            metrics["balanced_accuracy"] >= .85,
    }


def _nanmean(values):
    values = [float(value) for value in values
              if not math.isnan(float(value))]
    return statistics.mean(values) if values else math.nan


def mechanism_rows(rows):
    output = []
    for mechanism in (H1, H2_SUPPRESSION):
        values = [row for row in rows if row["mechanism"] == mechanism]
        output.append({
            "mechanism": mechanism, "runs": len(values),
            "error_rate": statistics.mean(
                int(row["wrong_action"]) for row in values),
            "mean_regret": statistics.mean(
                float(row["wrong_transfer_regret"]) for row in values),
        })
    return output


def bootstrap_final(scheme_id, rows, replicates):
    converted = []
    for row in rows:
        converted.append({
            **row, "split": "final_holdout",
            "mean_protocol_bytes_per_tick":
                row["mean_protocol_bytes_per_tick"],
        })
    return bootstrap_summary(
        scheme_id, "final_holdout",
        paired_seed_bootstrap(converted, replicates, seed=5517))


def observation_tensor(result, config):
    observations = result["observations"]
    horizon, nodes = int(config["horizon"]), int(config["num_agents"])
    return np.asarray([
        [list(observations[tick * nodes + node].values())
         for node in range(nodes)]
        for tick in range(horizon)], dtype=np.int32)


def candidate_configs(base_h1, base_h2, table, count=8):
    output = []
    seen = set()
    for row in table:
        if row["action_separation_holds"].lower() != "true":
            continue
        h1, h2 = json.loads(json.dumps(base_h1)), json.loads(json.dumps(base_h2))
        h1["h1"].update({
            "resource_capacity": int(float(row["h1_resource_capacity"])),
            "arrival_rate": float(row["h1_arrival_rate"]),
            "service_time_mean": float(row["h1_service_time_mean"]),
            "admission_intensity": float(row["h1_admission_intensity"]),
        })
        h2["h2_distributed_capacity_suppression"].update({
            "physical_capacity": int(float(row["h2_physical_capacity"])),
            "stale_version_lifetime":
                int(float(row["h2_stale_version_lifetime"])),
            "inconsistent_holder_probability":
                float(row["h2_inconsistent_holder_probability"]),
            "conservative_admission_probability":
                float(row["h2_conservative_admission_probability"]),
            "reservation_retention_duration":
                int(float(row["h2_reservation_retention_duration"])),
            "backoff_mean": float(row["h2_backoff_mean"]),
            "suppressed_slot_count":
                int(float(row["h2_suppressed_slot_count"])),
            "reconciliation_interval":
                int(float(row["h2_reconciliation_interval"])),
            "release_confirmation_delay":
                int(float(row["h2_release_confirmation_delay"])),
        })
        signature = json.dumps(
            {"h1": h1["h1"],
             "h2": h2["h2_distributed_capacity_suppression"]},
            sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        output.append((f"PAIR_{int(row['rank']):02d}", h1, h2))
        if len(output) == count:
            break
    if len(output) != count:
        raise ValueError("fewer than eight fixed Phase 3C candidate pairs")
    return output


def generate_loco_data(pairs, seeds):
    records = []
    for pair_id, h1, h2 in pairs:
        for seed in range(seeds):
            for mechanism, config, simulator in (
                (H1, h1, lambda action: simulate(h1, H1, action, seed)),
                (H2_SUPPRESSION, h2,
                 lambda action: simulate_suppression(h2, action, seed))):
                no_action = simulator("NO_ACTION")
                capacity = simulator("CAPACITY_ACTION")
                sync = simulator("SYNC_ACTION")
                preferred = ("CAPACITY_ACTION" if mechanism == H1
                             else "SYNC_ACTION")
                wrong = ("SYNC_ACTION" if mechanism == H1
                         else "CAPACITY_ACTION")
                utilities = {
                    "CAPACITY_ACTION": capacity["utility"],
                    "SYNC_ACTION": sync["utility"],
                }
                records.append({
                    "pair_id": pair_id, "seed": seed,
                    "mechanism": mechanism,
                    "truth": int(mechanism != H1),
                    "o0": observation_tensor(no_action, config),
                    "wrong_transfer_regret":
                        utilities[preferred] - utilities[wrong],
                })
    return records


def loco_audit(records, frozen_defs, ranges):
    rows = []
    pairs = sorted({row["pair_id"] for row in records})
    for frozen in frozen_defs:
        scheme = frozen_scheme(frozen)
        include_o1 = frozen["scheme_definition"]["include_derived_o1"]
        feature_maps, costs = [], []
        for index, record in enumerate(records):
            features, cost, _ = transmission(
                record["o0"], scheme, ranges, include_o1,
                seed=770000 + index)
            feature_maps.append(features)
            costs.append(cost)
        names = frozen["classifier"]["feature_order"]
        x = np.asarray([
            [features[name] for name in names] for features in feature_maps])
        y = np.asarray([row["truth"] for row in records], int)
        for held in pairs:
            train = np.asarray([
                row["pair_id"] != held for row in records])
            test = ~train
            model, mean, scale = fit_frozen_family(
                frozen["classifier"], x[train], y[train])
            probability = predict_refit(model, mean, scale, x[test])
            metrics = metric_values(y[test], probability)
            positions = np.where(test)[0]
            predictions = (probability >= .5).astype(int)
            regret = statistics.mean(
                records[pos]["wrong_transfer_regret"]
                * int(prediction != records[pos]["truth"])
                for pos, prediction in zip(positions, predictions))
            rows.append({
                "freeze_id": frozen["freeze_id"],
                "scheme_id": frozen["scheme_id"],
                "held_out_pair": held,
                "train_pair_count": 7,
                "held_out_runs": int(np.sum(test)),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "roc_auc": metrics["roc_auc"], "f1": metrics["f1"],
                "wrong_transfer_regret": regret,
                "mean_protocol_bytes_per_tick": statistics.mean(
                    costs[pos]["mean_protocol_bytes_per_tick"]
                    for pos in positions),
                "config_identifier_used_as_feature": False,
                "hyperparameters_reselected": False,
            })
    return rows


def final_verdict(summary_rows, robust_rows, loco_rows):
    f0 = next(row for row in summary_rows if row["freeze_id"] == "F0")
    survivors = []
    for row in summary_rows:
        if row["freeze_id"] == "F0":
            continue
        config = [item for item in loco_rows
                  if item["freeze_id"] == row["freeze_id"]]
        perturbed = [
            item for item in robust_rows
            if item["freeze_id"] == row["freeze_id"]
            and (float(item["message_delay"]) >= 2
                 or float(item["packet_loss"]) >= .1)
            and (float(item["balanced_accuracy"]) >= .65
                 or float(item["utility_gain_retention"]) >= .50)]
        identification = float(row["balanced_accuracy"]) >= .85
        decision = float(row["utility_gain_retention"]) >= .90
        passed = (
            (identification or decision)
            and float(row["mean_protocol_bytes_per_tick"])
                <= .25 * float(f0["mean_protocol_bytes_per_tick"])
            and statistics.mean(
                float(item["balanced_accuracy"]) for item in config) >= .80
            and min(float(item["balanced_accuracy"])
                    for item in config) >= .70
            and bool(perturbed))
        passed = passed and (
            float(row.get("maximum_absolute_seed_ba_influence", 1)) <= .05)
        if passed:
            survivors.append({
                "freeze_id": row["freeze_id"],
                "identification_preserving": identification,
                "decision_preserving": decision,
            })
    if survivors:
        return "MINIMAL_ORDINARY_EVIDENCE_SURVIVES", survivors
    near_full = [
        row for row in summary_rows
        if row["freeze_id"] != "F0"
        and float(row["mean_protocol_bytes_per_tick"])
            > .75 * float(f0["mean_protocol_bytes_per_tick"])
        and (float(row["balanced_accuracy"]) >= .85
             or float(row["utility_gain_retention"]) >= .90)]
    any_candidate_preserves = any(
        row["freeze_id"] != "F0"
        and (float(row["balanced_accuracy"]) >= .85
             or float(row["utility_gain_retention"]) >= .90)
        for row in summary_rows)
    f0_config = [item for item in loco_rows
                 if item["freeze_id"] == "F0"]
    f0_perturbed = [
        item for item in robust_rows
        if item["freeze_id"] == "F0"
        and (float(item["message_delay"]) >= 2
             or float(item["packet_loss"]) >= .1)
        and (float(item["balanced_accuracy"]) >= .65
             or float(item["utility_gain_retention"]) >= .50)]
    f0_complete = (
        float(f0["balanced_accuracy"]) >= .85
        and bool(f0_config)
        and statistics.mean(float(item["balanced_accuracy"])
                            for item in f0_config) >= .80
        and min(float(item["balanced_accuracy"])
                for item in f0_config) >= .70
        and bool(f0_perturbed))
    if f0_complete and (near_full or not any_candidate_preserves):
        return "FULL_AGGREGATION_NEEDED", []
    return "PHASE_5_INCONCLUSIVE", []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--h1-config", required=True)
    parser.add_argument("--h2-config", required=True)
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--loco-seeds", type=int, default=30)
    parser.add_argument("--expected-code-commit", required=True)
    args = parser.parse_args()
    output = Path(args.results)
    commit = current_commit()
    if commit != args.expected_code_commit:
        raise RuntimeError(
            f"code commit changed: expected {args.expected_code_commit}, got {commit}")
    frozen_path = output / "development_frozen_schemes.json"
    holdout_manifest_path = output / "dataset_manifest.csv"
    frozen_hash_before = sha256(frozen_path)
    holdout_hash_before = sha256(holdout_manifest_path)
    frozen_manifest = json.loads(frozen_path.read_text())
    if frozen_manifest["final_holdout_evaluated"]:
        raise RuntimeError("final holdout was already opened")
    frozen_defs = frozen_manifest["frozen_schemes"]
    if [row["freeze_id"] for row in frozen_defs] != [
            "F0", "F1", "F2", "F3", "F4"]:
        raise RuntimeError("frozen scheme set changed")
    o0 = np.load(output / "per_node_per_tick_o0.npz")["values"]
    manifest = read_csv(holdout_manifest_path)
    outcomes = read_csv(output / "run_outcomes.csv")
    indices = [i for i, row in enumerate(manifest)
               if row["split"] == "final_holdout"]
    if len(indices) != 60 or sorted({
            int(manifest[i]["seed"]) for i in indices}) != list(range(120, 150)):
        raise RuntimeError("holdout membership changed")
    ranges = json.loads(
        (output / "frozen_scheme_manifest.json").read_text())[
            "quantization_ranges"]
    opened_at = datetime.now(timezone.utc).isoformat()
    opening_record = {
        "final_holdout_opened": True,
        "final_holdout_opened_at": opened_at,
        "final_holdout_code_commit": commit,
        "holdout_seeds": list(range(120, 150)),
        "frozen_scheme_manifest_sha256_before_open": frozen_hash_before,
        "holdout_manifest_sha256_before_open": holdout_hash_before,
        "audit_completed": False,
    }
    (output / "final_holdout_opening_record.json").write_text(
        json.dumps(opening_record, indent=2, sort_keys=True) + "\n")

    # One-shot untouched holdout application of the five saved artifacts.
    holdout_rows = []
    for frozen in frozen_defs:
        holdout_rows.extend(evaluate_indices(
            o0, manifest, outcomes, indices, frozen, ranges))
    write_csv(output / "final_holdout_per_run_results.csv", holdout_rows)
    wrong_reference = statistics.mean(
        float(outcomes[i]["wrong_transfer_regret"]) for i in indices)
    f0_rows = [row for row in holdout_rows if row["freeze_id"] == "F0"]
    f0_regret = statistics.mean(
        float(row["wrong_transfer_regret"]) for row in f0_rows)
    summaries = []
    bootstrap_rows, influence_rows, mechanism_summary = [], [], []
    for frozen in frozen_defs:
        rows = [row for row in holdout_rows
                if row["freeze_id"] == frozen["freeze_id"]]
        summary = {
            "freeze_id": frozen["freeze_id"],
            "scheme_id": frozen["scheme_id"],
            **summarize(rows, wrong_reference, f0_regret),
        }
        summaries.append(summary)
        bootstrap_rows.extend(bootstrap_final(
            frozen["scheme_id"], rows, args.bootstrap))
        influence_rows.extend(seed_influence(
            frozen["scheme_id"],
            [{**row, "split": "final_holdout"} for row in rows]))
        for item in mechanism_rows(rows):
            mechanism_summary.append({
                "freeze_id": frozen["freeze_id"],
                "scheme_id": frozen["scheme_id"], **item})
    f0_ba = next(row for row in summaries
                 if row["freeze_id"] == "F0")["balanced_accuracy"]
    for row in summaries:
        row["within_0.05_of_f0"] = (
            float(row["balanced_accuracy"]) >= float(f0_ba) - .05)
        row["strict_regret_pass"] = (
            float(row["strict_regret_fraction"]) <= .10)
        row["utility_gain_retention_pass"] = (
            float(row["utility_gain_retention"]) >= .90)
    write_csv(output / "final_holdout_summary.csv", summaries)
    write_csv(output / "final_holdout_bootstrap.csv", bootstrap_rows)
    write_csv(output / "final_holdout_seed_influence.csv", influence_rows)
    write_csv(output / "final_holdout_mechanism_specific.csv",
              mechanism_summary)
    for row in summaries:
        relevant = [
            item for item in influence_rows
            if item["scheme_id"] == row["scheme_id"]]
        row["maximum_absolute_seed_ba_influence"] = max(
            abs(float(item["ba_influence"])) for item in relevant)
        row["maximum_absolute_seed_regret_influence"] = max(
            abs(float(item["regret_influence"])) for item in relevant)
    # Rewrite with influence diagnostics used by the final rule.
    write_csv(output / "final_holdout_summary.csv", summaries)

    robustness_per_run, robustness_summary = [], []
    for frozen in frozen_defs:
        for delay in (0, 1, 2, 5, 10):
            for loss in (0.0, .05, .1, .2, .4):
                rows = evaluate_indices(
                    o0, manifest, outcomes, indices, frozen, ranges,
                    delay, loss, f"delay={delay},loss={loss}")
                robustness_per_run.extend(rows)
                robustness_summary.append({
                    "freeze_id": frozen["freeze_id"],
                    "scheme_id": frozen["scheme_id"],
                    "message_delay": delay, "packet_loss": loss,
                    **summarize(rows, wrong_reference, f0_regret),
                })
    write_csv(output / "delay_loss_per_run_results.csv", robustness_per_run)
    write_csv(output / "delay_loss_summary.csv", robustness_summary)
    thresholds = []
    for frozen in frozen_defs:
        values = [row for row in robustness_summary
                  if row["freeze_id"] == frozen["freeze_id"]]
        below_ba = [row for row in values
                    if float(row["balanced_accuracy"]) < .85]
        below_utility = [row for row in values
                         if float(row["utility_gain_retention"]) < .90]
        thresholds.append({
            "freeze_id": frozen["freeze_id"],
            "scheme_id": frozen["scheme_id"],
            "first_below_ba_0.85": (
                f"d={below_ba[0]['message_delay']},p={below_ba[0]['packet_loss']}"
                if below_ba else "NONE"),
            "first_below_utility_retention_0.90": (
                f"d={below_utility[0]['message_delay']},p={below_utility[0]['packet_loss']}"
                if below_utility else "NONE"),
        })
    write_csv(output / "robustness_thresholds.csv", thresholds)

    pairs = candidate_configs(
        load_config(args.h1_config), load_config(args.h2_config),
        read_csv(args.candidate_table))
    loco_records = generate_loco_data(pairs, args.loco_seeds)
    loco_rows = loco_audit(loco_records, frozen_defs, ranges)
    write_csv(output / "leave_one_config_pair_out_results.csv", loco_rows)
    loco_summary = []
    for frozen in frozen_defs:
        values = [row for row in loco_rows
                  if row["freeze_id"] == frozen["freeze_id"]]
        loco_summary.append({
            "freeze_id": frozen["freeze_id"],
            "scheme_id": frozen["scheme_id"],
            "mean_balanced_accuracy": statistics.mean(
                float(row["balanced_accuracy"]) for row in values),
            "minimum_balanced_accuracy": min(
                float(row["balanced_accuracy"]) for row in values),
            "maximum_regret": max(
                float(row["wrong_transfer_regret"]) for row in values),
            "mean_protocol_bytes_per_tick": statistics.mean(
                float(row["mean_protocol_bytes_per_tick"]) for row in values),
        })
    write_csv(output / "configuration_generalization_summary.csv",
              loco_summary)

    verdict, survivors = final_verdict(
        summaries, robustness_summary, loco_rows)
    audit = {
        "phase": "Phase 5.5-5.7",
        "verdict": verdict, "survivors": survivors,
        "branch": subprocess.run(
            ["git", "branch", "--show-current"], cwd=HERE, text=True,
            capture_output=True).stdout.strip(),
        "code_commit": commit,
        "phase52_commit": "fc4a492",
        "phase53_54_results_commit": "a928796",
        "frozen_scheme_manifest_sha256_before_open": frozen_hash_before,
        "holdout_manifest_sha256_before_open": holdout_hash_before,
        "final_holdout_evaluated": True,
        "final_holdout_opened_at": opened_at,
        "final_holdout_code_commit": commit,
        "holdout_seeds": list(range(120, 150)),
        "frozen_scheme_count": 5,
        "scheme_or_threshold_changed_after_open": False,
    }
    (output / "final_audit_manifest.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n")
    opening_record["audit_completed"] = True
    opening_record["verdict"] = verdict
    (output / "final_holdout_opening_record.json").write_text(
        json.dumps(opening_record, indent=2, sort_keys=True) + "\n")
    frozen_manifest.update({
        "final_holdout_evaluated": True,
        "final_holdout_opened_at": opened_at,
        "final_holdout_code_commit": commit,
        "pre_open_manifest_sha256": frozen_hash_before,
    })
    frozen_path.write_text(
        json.dumps(frozen_manifest, indent=2, sort_keys=True) + "\n")
    report = f"""# Phase 5.5--5.7 final audit

Final local verdict: `{verdict}`

The five Phase 5.3--5.4 schemes were applied once to the untouched 60-run
holdout at commit `{commit}`. The holdout was opened at `{opened_at}`.
No scheme, threshold, feature, reconstruction rule, classifier family,
hyperparameter, scaler, or decision threshold was changed.

## Holdout

See `final_holdout_summary.csv`, `final_holdout_bootstrap.csv`, and
`final_holdout_mechanism_specific.csv`. Both identification definitions and
both decision-preservation definitions are reported; neither is suppressed.

## Channel robustness

All 125 scheme-by-delay-by-loss conditions were applied to actual messages
before central reconstruction. See `delay_loss_summary.csv` and
`robustness_thresholds.csv`.

## Configuration generalization

The frozen scheme definitions and model hyperparameters were evaluated by
leave-one-configuration-pair-out over eight fixed Phase 3C candidate pairs.
Only the frozen model family was refit on the other seven pairs. Configuration
identifiers were never features.

## Survivors

`{json.dumps(survivors, sort_keys=True)}`

This is a local verdict for the constructed mechanism pair, not a universal
theory of distributed evidence or safe policy transfer.
"""
    (output / "phase5_final_report.md").write_text(report)
    print(verdict)


if __name__ == "__main__":
    main()
