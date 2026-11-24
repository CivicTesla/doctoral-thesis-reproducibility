#!/usr/bin/env python3
"""Deterministic five-seed receiver-only Gate 0 pilot."""

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
R0 = HERE.parent / "R0_hidden_mechanism_audit"
sys.path[:0] = [
    str(R0), str(R0 / "scripts"), str(HERE / "src")]

from run_phase4a_audit import metric_values
from run_phase5_final import transmission
from src.config import load_config
from src.phase5_final import frozen_scheme, predict_frozen
from src.schemas import O0_FIELDS
from src.simple_models import LogisticModel
from src.simulator import H1, simulate
from src.suppression_simulator import H2_SUPPRESSION, simulate_suppression

from decision import transfer_or_reject
from gate0_config import load_gate0_config
from protocol import assert_fixed_schemes, assert_seed_isolation
from receiver_features import append_receiver_features


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fields:
        raise ValueError(f"empty output: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def observation_tensor(result, config):
    observations = result["observations"]
    horizon, nodes = int(config["horizon"]), int(config["num_agents"])
    output = np.asarray([
        [list(observations[tick * nodes + node].values())
         for node in range(nodes)]
        for tick in range(horizon)], dtype=np.int32)
    if output.shape != (horizon, nodes, len(O0_FIELDS)):
        raise ValueError("unexpected O0 shape")
    return output


def simulate_records(h1, h2, groups):
    rows = []
    for split in ("train", "validation", "pilot_test"):
        for seed in sorted(groups[split]):
            for mechanism, config, simulator in (
                    (H1, h1,
                     lambda action: simulate(h1, H1, action, seed)),
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
                    "CAPACITY_ACTION": float(capacity["utility"]),
                    "SYNC_ACTION": float(sync["utility"]),
                }
                rows.append({
                    "run_id": f"{split}_{mechanism}_seed{seed}",
                    "seed": seed, "split": split,
                    "mechanism": mechanism,
                    "true_label": int(mechanism != H1),
                    "o0": observation_tensor(no_action, config),
                    "oracle_action": preferred,
                    "wrong_action": wrong,
                    "wrong_transfer_regret":
                        utilities[preferred] - utilities[wrong],
                })
    return rows


def fit_logistic(feature_maps, labels, l2):
    names = sorted(feature_maps[0])
    if any(sorted(row) != names for row in feature_maps):
        raise ValueError("receiver feature schema changed")
    x = np.asarray([[row[name] for name in names]
                    for row in feature_maps], float)
    mean, scale = np.mean(x, axis=0), np.std(x, axis=0)
    scale[scale == 0] = 1.0
    model = LogisticModel(l2=l2).fit((x - mean) / scale, labels)
    return {"names": names, "mean": mean, "scale": scale, "model": model}


def receiver_probability(artifact, feature_map):
    x = np.asarray([[feature_map[name] for name in artifact["names"]]], float)
    return float(artifact["model"].predict_proba(
        (x - artifact["mean"]) / artifact["scale"])[0])


def generate_features(records, schemes, ranges, conditions):
    rows = []
    for frozen in schemes.values():
        scheme = frozen_scheme(frozen)
        include_o1 = frozen["scheme_definition"]["include_derived_o1"]
        for record_index, record in enumerate(records):
            for condition in conditions:
                delay, loss = int(condition["delay"]), float(condition["loss"])
                ordinary, cost, sent = transmission(
                    record["o0"], scheme, ranges, include_o1, delay, loss,
                    seed=(record["seed"] * 100003 + delay * 1009
                          + int(loss * 1000) + record_index))
                explicit = append_receiver_features(
                    ordinary, sent, initial_arrival_age=delay)
                rows.append({
                    "record": record, "freeze_id": frozen["freeze_id"],
                    "scheme_id": frozen["scheme_id"],
                    "delay": delay, "loss": loss,
                    "ordinary": ordinary, "explicit": explicit,
                    "cost": cost, "frozen": frozen,
                })
    return rows


def prediction_row(item, receiver, probability, reject_cost,
                   h1_wrong_regret, h2_wrong_regret):
    record, cost = item["record"], item["cost"]
    prediction = int(probability >= .5)
    wrong = int(prediction != record["true_label"])
    selected = "SYNC_ACTION" if prediction else "CAPACITY_ACTION"
    decision = "TRANSFER"
    rejection_cost = 0.0
    if receiver == "ASYMMETRIC_TRANSFER_REJECT":
        choice = transfer_or_reject(
            probability, h1_wrong_regret, h2_wrong_regret, reject_cost)
        decision, selected = choice["decision"], choice["selected_action"]
        if decision == "REJECT":
            wrong = 0
            rejection_cost = reject_cost
    wrong_regret = record["wrong_transfer_regret"] * wrong
    return {
        "run_id": record["run_id"], "seed": record["seed"],
        "split": record["split"], "mechanism": record["mechanism"],
        "freeze_id": item["freeze_id"], "scheme_id": item["scheme_id"],
        "receiver": receiver, "message_delay": item["delay"],
        "packet_loss": item["loss"],
        "true_label": record["true_label"],
        "predicted_label": prediction,
        "prediction_probability": probability,
        "decision": decision, "selected_action": selected,
        "oracle_action": record["oracle_action"],
        "wrong_transfer": wrong,
        "wrong_transfer_regret": wrong_regret,
        "rejection_cost": rejection_cost,
        "total_decision_cost": wrong_regret + rejection_cost,
        "mean_protocol_bytes_per_tick":
            cost["mean_protocol_bytes_per_tick"],
        "effective_received_bytes_per_tick":
            cost["effective_received_bytes_per_tick"],
        "maximum_silence_duration": cost["maximum_silence_duration"],
    }


def summarize(rows):
    output = []
    keys = sorted({
        (row["split"], row["freeze_id"], row["receiver"],
         row["message_delay"], row["packet_loss"]) for row in rows})
    for split, freeze_id, receiver, delay, loss in keys:
        values = [
            row for row in rows
            if (row["split"], row["freeze_id"], row["receiver"],
                row["message_delay"], row["packet_loss"])
            == (split, freeze_id, receiver, delay, loss)]
        y = np.asarray([row["true_label"] for row in values], int)
        p = np.asarray([row["prediction_probability"]
                        for row in values], float)
        metrics = metric_values(y, p)
        h2 = [row for row in values if row["mechanism"] == H2_SUPPRESSION]
        output.append({
            "split": split, "freeze_id": freeze_id,
            "receiver": receiver, "message_delay": delay,
            "packet_loss": loss, **metrics,
            "wrong_transfer_rate": statistics.mean(
                row["wrong_transfer"] for row in values),
            "mean_wrong_transfer_regret": statistics.mean(
                row["wrong_transfer_regret"] for row in values),
            "mean_total_decision_cost": statistics.mean(
                row["total_decision_cost"] for row in values),
            "transfer_coverage": statistics.mean(
                row["decision"] == "TRANSFER" for row in values),
            "h2_error_rate": statistics.mean(
                row["predicted_label"] != row["true_label"] for row in h2),
            "h2_wrong_transfer_regret": statistics.mean(
                row["wrong_transfer_regret"] for row in h2),
            "mean_protocol_bytes_per_tick": statistics.mean(
                row["mean_protocol_bytes_per_tick"] for row in values),
        })
    return output


def counterexamples(features, prediction_rows):
    balance = {}
    for item in features:
        key = (item["record"]["mechanism"], item["freeze_id"],
               item["delay"], item["loss"])
        balance[key] = balance.get(key, 0) + 1
    counts = list(balance.values())
    channel_balanced = len(set(counts)) == 1
    decisions = [
        row for row in prediction_rows
        if row["receiver"] == "ASYMMETRIC_TRANSFER_REJECT"
        and row["split"] == "pilot_test"]
    minimum_coverage = min(
        (statistics.mean(
            row["decision"] == "TRANSFER" for row in decisions
            if row["freeze_id"] == freeze_id)
         for freeze_id in ("F1", "F2", "F4")), default=0)
    return [{
        "check": "mechanism_channel_assignment_balanced",
        "passed": channel_balanced,
        "value": min(counts) if counts else 0,
        "interpretation": "same run count per mechanism/scheme/condition",
    }, {
        "check": "reject_all_collapse_absent",
        "passed": minimum_coverage > 0,
        "value": minimum_coverage,
        "interpretation": "minimum pilot-test transfer coverage",
    }]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_gate0_config(args.config)
    groups = assert_seed_isolation(config)
    output = HERE / "results/gate0_pilot"
    output.mkdir(parents=True, exist_ok=True)
    r0_output = R0 / "results/phase5_minimal_distributed_evidence"
    frozen_path = r0_output / "development_frozen_schemes.json"
    frozen_manifest = json.loads(frozen_path.read_text())
    schemes = assert_fixed_schemes(frozen_manifest)
    ranges = json.loads(
        (r0_output / "frozen_scheme_manifest.json").read_text())[
            "quantization_ranges"]
    h1 = load_config(R0 / "results/phase3c_redesign/matched_config_h1.yaml")
    h2 = load_config(R0 / "results/phase3c_redesign/matched_config_h2.yaml")
    records = simulate_records(h1, h2, groups)
    all_conditions = []
    for name in ("train_channel_conditions", "pilot_test_conditions"):
        for condition in config[name]:
            if condition not in all_conditions:
                all_conditions.append(condition)
    features = generate_features(records, schemes, ranges, all_conditions)

    train_features = [row for row in features
                      if row["record"]["split"] == "train"
                      and {"delay": row["delay"], "loss": row["loss"]}
                      in config["train_channel_conditions"]]
    artifacts = {}
    for freeze_id in schemes:
        selected = [row for row in train_features
                    if row["freeze_id"] == freeze_id]
        labels = np.asarray(
            [row["record"]["true_label"] for row in selected], int)
        artifacts[(freeze_id, "CHANNEL_RANDOMIZED_ORDINARY")] = fit_logistic(
            [row["ordinary"] for row in selected], labels,
            config["receiver"]["l2"])
        artifacts[(freeze_id, "EXPLICIT_AGE_MASK_SILENCE")] = fit_logistic(
            [row["explicit"] for row in selected], labels,
            config["receiver"]["l2"])

    train_records = [row for row in records if row["split"] == "train"]
    h1_wrong = statistics.mean(
        row["wrong_transfer_regret"] for row in train_records
        if row["mechanism"] == H1)
    h2_wrong = statistics.mean(
        row["wrong_transfer_regret"] for row in train_records
        if row["mechanism"] == H2_SUPPRESSION)
    prediction_rows = []
    for item in features:
        frozen_probability = predict_frozen(
            item["frozen"]["classifier"], item["ordinary"])
        prediction_rows.append(prediction_row(
            item, "ZERO_DELAY_FROZEN", frozen_probability,
            config["receiver"]["reject_cost"], h1_wrong, h2_wrong))
        ordinary_probability = receiver_probability(
            artifacts[(item["freeze_id"],
                       "CHANNEL_RANDOMIZED_ORDINARY")],
            item["ordinary"])
        prediction_rows.append(prediction_row(
            item, "CHANNEL_RANDOMIZED_ORDINARY", ordinary_probability,
            config["receiver"]["reject_cost"], h1_wrong, h2_wrong))
        explicit_probability = receiver_probability(
            artifacts[(item["freeze_id"],
                       "EXPLICIT_AGE_MASK_SILENCE")],
            item["explicit"])
        prediction_rows.append(prediction_row(
            item, "EXPLICIT_AGE_MASK_SILENCE", explicit_probability,
            config["receiver"]["reject_cost"], h1_wrong, h2_wrong))
        prediction_rows.append(prediction_row(
            item, "ASYMMETRIC_TRANSFER_REJECT", explicit_probability,
            config["receiver"]["reject_cost"], h1_wrong, h2_wrong))
    summary_rows = summarize(prediction_rows)
    counterexample_rows = counterexamples(features, prediction_rows)

    f2 = [row for row in summary_rows
          if row["split"] == "pilot_test" and row["freeze_id"] == "F2"
          and row["packet_loss"] == 0 and row["message_delay"] in (1, 2)]
    comparisons = []
    promising = True
    for delay in (1, 2):
        ordinary = next(
            row for row in f2
            if row["message_delay"] == delay
            and row["receiver"] == "CHANNEL_RANDOMIZED_ORDINARY")
        explicit = next(
            row for row in f2
            if row["message_delay"] == delay
            and row["receiver"] == "EXPLICIT_AGE_MASK_SILENCE")
        decision = next(
            row for row in f2
            if row["message_delay"] == delay
            and row["receiver"] == "ASYMMETRIC_TRANSFER_REJECT")
        comparisons.append({
            "message_delay": delay,
            "ordinary_ba": ordinary["balanced_accuracy"],
            "explicit_ba": explicit["balanced_accuracy"],
            "ba_improvement":
                explicit["balanced_accuracy"] - ordinary["balanced_accuracy"],
            "ordinary_h2_regret": ordinary["h2_wrong_transfer_regret"],
            "explicit_h2_regret": explicit["h2_wrong_transfer_regret"],
            "decision_total_cost": decision["mean_total_decision_cost"],
            "decision_transfer_coverage": decision["transfer_coverage"],
            "communication_bytes_equal":
                ordinary["mean_protocol_bytes_per_tick"]
                == explicit["mean_protocol_bytes_per_tick"],
        })
        promising &= (
            comparisons[-1]["ba_improvement"]
            >= config["gate0"]["practical_ba_margin"]
            and comparisons[-1]["explicit_h2_regret"]
            <= comparisons[-1]["ordinary_h2_regret"]
            and comparisons[-1]["decision_transfer_coverage"]
            >= config["gate0"]["minimum_transfer_coverage"]
            and comparisons[-1]["communication_bytes_equal"])
    if not all(row["passed"] for row in counterexample_rows):
        status = "GATE0_COUNTEREXAMPLE_DOMINATES"
    elif promising:
        status = "GATE0_AGE_SIGNAL_PROMISING"
    elif all(abs(row["ba_improvement"])
             < config["gate0"]["practical_ba_margin"]
             for row in comparisons):
        status = "GATE0_AGE_SIGNAL_NOT_NECESSARY"
    else:
        status = "GATE0_PILOT_INCONCLUSIVE"

    manifest = {
        "status": status, "pilot_only": True,
        "r0_formal_verdict": "PHASE_5_INCONCLUSIVE",
        "r0_interpretations": [
            "LOW_RATE_DISTRIBUTED_EVIDENCE_EXISTS",
            "ZERO_DELAY_FROZEN_INFERENCE_IS_NOT_ROBUST"],
        "r0_frozen_manifest_sha256":
            hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "r0_final_holdout_seeds_used": False,
        "sealed_final_test_evaluated": False,
        "sealed_final_test_seeds":
            config["sealed_final_test_seeds"],
        "fixed_schemes": list(schemes),
        "train_wrong_regret": {"H1": h1_wrong, "H2": h2_wrong},
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True,
            capture_output=True).stdout.strip(),
    }
    np.savez_compressed(
        output / "pilot_o0.npz",
        values=np.asarray([row["o0"] for row in records]),
        run_ids=np.asarray([row["run_id"] for row in records]))
    write_csv(output / "per_run_predictions.csv", prediction_rows)
    write_csv(output / "receiver_comparison.csv", summary_rows)
    write_csv(output / "f2_delay_recovery.csv", comparisons)
    write_csv(output / "counterexample_checks.csv", counterexample_rows)
    (output / "pilot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report = f"""# B1 R1 Gate 0 small pilot

Pilot status: `{status}`

This is a five-seed-per-split necessity pilot, not a formal R1 result. It used
new seeds only, did not access R0 seeds 120--149, and left R1 final-test seeds
2000--2029 sealed.

The transmitter and byte cost were fixed for F1, F2, and F4. Explicit age,
missingness, and silence were derived at the receiver from arrived messages
and the receiver clock.

See `f2_delay_recovery.csv` for the primary delay-1 and delay-2 comparison,
`receiver_comparison.csv` for all schemes and conditions, and
`counterexample_checks.csv` for leakage and rejection-collapse checks.
"""
    (output / "gate0_pilot_report.md").write_text(report)
    print(status)


if __name__ == "__main__":
    main()
