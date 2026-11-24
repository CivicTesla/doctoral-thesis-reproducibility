#!/usr/bin/env python3
"""F0 Section 14: risk-direction prediction.

For each outer fold, per-error-type costs (directional action advantage,
NO_ACTION opportunity cost, coverage-transition cost) are estimated from
TRAINING folds only. The held-out TEST fold contributes only observable
confusion/coverage-transition COUNTS (never its own regret values) to
combine with those training-estimated per-unit costs into a predicted
regret-change sign. The true test-fold regret change is computed
separately and used only afterward to score prediction accuracy: it is
never fed back into the prediction itself.

Usage: risk_prediction.py <pair1_dir> <pair2_dir> <out_csv>
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REJECT_COST = 5.0


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def transition_predict(train, test, key_col, a_col_accept, a_col_regret,
                        b_col_accept, b_col_regret, mech_mask):
    """train/test: dict of column arrays. Returns (predicted_delta,
    actual_delta) for the mechanism selected by mech_mask."""
    tr_a_acc, tr_b_acc = train[a_col_accept][mech_mask(train)], train[b_col_accept][mech_mask(train)]
    tr_a_reg, tr_b_reg = train[a_col_regret][mech_mask(train)], train[b_col_regret][mech_mask(train)]
    both = tr_a_acc & tr_b_acc
    corrected_costs = (tr_a_reg[both] - tr_b_reg[both])[tr_b_reg[both] < tr_a_reg[both]]
    new_costs = (tr_b_reg[both] - tr_a_reg[both])[tr_b_reg[both] > tr_a_reg[both]]
    mean_corrected = float(corrected_costs.mean()) if len(corrected_costs) else 0.0
    mean_new = float(new_costs.mean()) if len(new_costs) else 0.0
    mean_loss = REJECT_COST - float(tr_a_reg[tr_a_acc & ~tr_b_acc].mean()) if (tr_a_acc & ~tr_b_acc).any() else 0.0
    mean_gain = float(tr_b_reg[~tr_a_acc & tr_b_acc].mean()) - REJECT_COST if (~tr_a_acc & tr_b_acc).any() else 0.0

    m = mech_mask(test)
    te_a_acc, te_b_acc = test[a_col_accept][m], test[b_col_accept][m]
    n_correct_transitions = int((te_a_acc & te_b_acc).sum())
    n_loss = int((te_a_acc & ~te_b_acc).sum())
    n_gain = int((~te_a_acc & te_b_acc).sum())
    predicted_delta = (
        n_correct_transitions * 0.5 * (mean_new - mean_corrected)
        + n_loss * mean_loss + n_gain * mean_gain) / max(1, m.sum())

    actual_delta = float((test[b_col_regret][m] - test[a_col_regret][m]).mean())
    return predicted_delta, actual_delta


def pair1_arrays(df, method):
    sub = df[df.method == method].reset_index(drop=True)
    return {
        "mechanism": sub.mechanism.to_numpy(),
        "accept": sub.accepted.to_numpy().astype(bool),
        "regret": sub.system_regret.to_numpy(dtype=float),
        "outer_fold": sub.outer_fold.to_numpy(),
    }


def pair2_arrays(rows, method):
    return {
        "y": np.array([r["y"] for r in rows]),
        "accept": np.array([r[f"{method}_accept"] for r in rows]),
        "regret": np.array([r[f"{method}_regret"] for r in rows], dtype=float),
        "outer_fold": np.array([r["seed"] % 5 for r in rows]),
    }


def fold_split(arrays, fold):
    train_mask = arrays["outer_fold"] != fold
    test_mask = ~train_mask
    train = {k: v[train_mask] for k, v in arrays.items()}
    test = {k: v[test_mask] for k, v in arrays.items()}
    return train, test


def main():
    pair1_dir, pair2_dir, out_csv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    p1_df = pd.read_csv(pair1_dir / "per_sample_strict_action_outcomes.csv")
    p2_rows = pd.read_csv(pair2_dir / "out_of_fold_predictions.csv").to_dict("records")

    results = []
    # ---- Pair 1: T0 -> T4 ----
    a = pair1_arrays(p1_df, "T0")
    a_key = {"accept": "accept", "regret": "regret"}
    b = pair1_arrays(p1_df, "T4")
    for mech, label in (("H1_CAPACITY_SHORTAGE", "H1"),
                        ("H2_DISTRIBUTED_CAPACITY_SUPPRESSION", "H2")):
        for fold in range(5):
            train_a, test_a = fold_split(a, fold)
            train_b, test_b = fold_split(b, fold)
            train = {"accept_a": train_a["accept"], "regret_a": train_a["regret"],
                     "accept_b": train_b["accept"], "regret_b": train_b["regret"],
                     "mechanism": train_a["mechanism"]}
            test = {"accept_a": test_a["accept"], "regret_a": test_a["regret"],
                    "accept_b": test_b["accept"], "regret_b": test_b["regret"],
                    "mechanism": test_a["mechanism"]}
            pred, actual = transition_predict(
                train, test, "mechanism", "accept_a", "regret_a", "accept_b", "regret_b",
                lambda d, mech=mech: d["mechanism"] == mech)
            results.append({"pair": "pair1", "comparison": "T4-T0", "mechanism": label,
                            "fold": fold, "predicted_sign": int(np.sign(pred)),
                            "actual_sign": int(np.sign(actual)),
                            "correct": int(np.sign(pred)) == int(np.sign(actual))})

    # ---- Pair 2: M0 -> M1 ----
    a2 = pair2_arrays(p2_rows, "M0")
    b2 = pair2_arrays(p2_rows, "M1")
    for yval, label in ((0, "H3"), (1, "H4")):
        for fold in range(5):
            train_a, test_a = fold_split(a2, fold)
            train_b, test_b = fold_split(b2, fold)
            train = {"accept_a": train_a["accept"], "regret_a": train_a["regret"],
                     "accept_b": train_b["accept"], "regret_b": train_b["regret"],
                     "mechanism": train_a["y"]}
            test = {"accept_a": test_a["accept"], "regret_a": test_a["regret"],
                    "accept_b": test_b["accept"], "regret_b": test_b["regret"],
                    "mechanism": test_a["y"]}
            pred, actual = transition_predict(
                train, test, "mechanism", "accept_a", "regret_a", "accept_b", "regret_b",
                lambda d, yval=yval: d["mechanism"] == yval)
            results.append({"pair": "pair2", "comparison": "M1-M0", "mechanism": label,
                            "fold": fold, "predicted_sign": int(np.sign(pred)),
                            "actual_sign": int(np.sign(actual)),
                            "correct": int(np.sign(pred)) == int(np.sign(actual))})

    write_csv(out_csv, results)
    mech_acc = np.mean([r["correct"] for r in results])

    total_rows = []
    for pair, comparison in (("pair1", "T4-T0"), ("pair2", "M1-M0")):
        subset = [r for r in results if r["pair"] == pair and r["comparison"] == comparison]
        for fold in range(5):
            fold_rows = [r for r in subset if r["fold"] == fold]
            total_pred_sign = int(np.sign(sum(r["predicted_sign"] for r in fold_rows)))
            total_actual_sign = int(np.sign(sum(r["actual_sign"] for r in fold_rows)))
            total_rows.append({"pair": pair, "fold": fold,
                               "total_predicted_sign": total_pred_sign,
                               "total_actual_sign": total_actual_sign,
                               "correct": total_pred_sign == total_actual_sign})
    total_acc = np.mean([r["correct"] for r in total_rows])
    print(f"mechanism_sign_accuracy={mech_acc:.4f} (target >=0.80)")
    print(f"total_sign_accuracy={total_acc:.4f} (target >=0.75)")
    return mech_acc, total_acc


if __name__ == "__main__":
    main()
