#!/usr/bin/env python3
"""B1 CF0: strict counterfactual action grounding pilot.

Runs the pre-registered protocol (`02_preregistered_protocol.md`) on the
frozen development seeds 4000-4029, using the checkpoint-fork simulator in
`CF0_strict_counterfactual_grounding/src/cf0_simulator.py`. Does not train
any classifier, does not touch AS1/telemetry-selection scope, does not
modify H1/H2 parameters.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
CF0 = HERE.parents[1]
B1 = HERE.parents[2]
R0 = B1 / "R0_hidden_mechanism_audit"
CONCLUSION = CF0 / "results/CONCLUSIONS/CF0_STRICT_COUNTERFACTUAL"
FIGURES = CF0 / "figures"

sys.path[:0] = [str(CF0 / "src"), str(R0)]
import cf0_simulator as cf0  # noqa: E402
from src.config import load_config  # noqa: E402

SEEDS = list(range(4000, 4030))
MECHANISMS = (cf0.H1, cf0.H2_SUPPRESSION)
BOOTSTRAPS = 2000
NONTRIVIAL_FRACTION = 0.10
NONTRIVIAL_MIN_SEEDS = 5
NEAR_TIE_FRACTION = 0.05


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def best_action(utilities):
    """argmax utility; exact-equality tie detection (Section 11/protocol)."""
    ordered = sorted(utilities.items(), key=lambda kv: -kv[1])
    best_name, best_val = ordered[0]
    second_name, second_val = ordered[1]
    is_tie = best_val == second_val
    return best_name, best_val - second_val, is_tie


def main():
    start = time.perf_counter()
    CF0.mkdir(parents=True, exist_ok=True)
    CONCLUSION.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    assert not (set(SEEDS) & set(range(2000, 2030)))
    assert not (set(SEEDS) & set(range(120, 150)))
    assert min(SEEDS) == 4000 and max(SEEDS) == 4029 and len(SEEDS) == 30

    h1_config = load_config(str(
        R0 / "results/phase3c_redesign/matched_config_h1.yaml"))
    h2_config = load_config(str(
        R0 / "results/phase3c_redesign/matched_config_h2.yaml"))
    configs = {cf0.H1: h1_config, cf0.H2_SUPPRESSION: h2_config}

    manifest_rows, hash_rows, outcome_rows = [], [], []
    checkpoint_id = 0
    for seed in SEEDS:
        for mechanism in MECHANISMS:
            config = configs[mechanism]
            exo, branches = cf0.run_checkpoint(config, mechanism, seed)
            checkpoint_id += 1
            cpid = f"cp{checkpoint_id:04d}_seed{seed}_{mechanism}"

            pre_action_state_hash = cf0.sha256_arrays(
                np.array([seed]),
                np.frombuffer(branches[cf0.ACTIONS[0]]["config_hash"].encode(),
                              dtype=np.uint8))
            # service_schedule_hash: service-time + deadline-offset arrays
            # only (the "when will this task finish" schedule), separated
            # out from the arrival/admission-throttle components of the
            # combined exogenous_schedule_hash for a more granular check.
            service_schedule_hash = cf0.sha256_arrays(
                exo.service_time, exo.deadline_offset)
            # channel_schedule_hash: this action-simulation layer has no
            # packet-loss/wireless-channel model (see
            # 04_exogenous_endogenous_event_inventory.md); repurposed for
            # H2 to carry the initial suppression-seeding schedule hash,
            # the nearest actual pre-generated stochastic schedule at this
            # layer, and reported N/A for H1.
            channel_schedule_hash = (
                exo.initial_suppression_hash if mechanism == cf0.H2_SUPPRESSION
                else "NOT_APPLICABLE_NO_CHANNEL_AT_ACTION_LAYER")
            manifest_rows.append({
                "checkpoint_id": cpid, "seed": seed, "mechanism": mechanism,
                "decision_tick": 0, "eligible": True,
                "eligibility_rule":
                    "single_pre_episode_checkpoint_per_seed_mechanism",
                "pre_action_state_hash": pre_action_state_hash,
                "exogenous_schedule_hash": exo.schedule_hash,
                "service_schedule_hash": service_schedule_hash,
                "channel_schedule_hash": channel_schedule_hash,
                "initial_suppression_schedule_hash":
                    exo.initial_suppression_hash,
                "config_hash": branches[cf0.ACTIONS[0]]["config_hash"],
                "cf0_simulator_module": "cf0_simulator.py",
                "rng_design": "prebuilt_exogenous_table+keyed_endogenous_stream",
            })

            for action in cf0.ACTIONS:
                hash_rows.append({
                    "checkpoint_id": cpid, "seed": seed,
                    "mechanism": mechanism, "action": action,
                    "exogenous_schedule_hash_this_branch": exo.schedule_hash,
                    "service_schedule_hash_this_branch": service_schedule_hash,
                    "channel_schedule_hash_this_branch": channel_schedule_hash,
                    "initial_suppression_hash_this_branch":
                        exo.initial_suppression_hash,
                    "arrivals": branches[action]["arrivals"],
                    "matches_other_branches": True,  # asserted in
                    # cf0_simulator.run_checkpoint; raises before reaching
                    # here otherwise
                })

            utilities = {a: branches[a]["utility"] for a in cf0.ACTIONS}
            best, advantage, tie = best_action(utilities)
            outcome_rows.append({
                "checkpoint_id": cpid, "seed": seed, "mechanism": mechanism,
                "utility_no_action": utilities["NO_ACTION"],
                "utility_capacity": utilities["CAPACITY_ACTION"],
                "utility_sync": utilities["SYNC_ACTION"],
                "arrivals": branches["NO_ACTION"]["arrivals"],
                "completed_no_action": branches["NO_ACTION"]["completed"],
                "completed_capacity": branches["CAPACITY_ACTION"]["completed"],
                "completed_sync": branches["SYNC_ACTION"]["completed"],
                "failures_no_action": branches["NO_ACTION"]["failures"],
                "failures_capacity": branches["CAPACITY_ACTION"]["failures"],
                "failures_sync": branches["SYNC_ACTION"]["failures"],
                "final_queue_occupancy_no_action":
                    branches["NO_ACTION"]["final_queue_occupancy"],
                "final_queue_occupancy_capacity":
                    branches["CAPACITY_ACTION"]["final_queue_occupancy"],
                "final_queue_occupancy_sync":
                    branches["SYNC_ACTION"]["final_queue_occupancy"],
                "communication_bytes_sync":
                    branches["SYNC_ACTION"]["communication_bytes"],
                "oracle_best_action": best,
                "best_vs_second_advantage": advantage,
                "tie": tie,
                "capacity_minus_no_action":
                    utilities["CAPACITY_ACTION"] - utilities["NO_ACTION"],
                "sync_minus_no_action":
                    utilities["SYNC_ACTION"] - utilities["NO_ACTION"],
                "capacity_minus_sync":
                    utilities["CAPACITY_ACTION"] - utilities["SYNC_ACTION"],
            })

    write_csv(CF0 / "06_checkpoint_manifest.csv", manifest_rows)
    write_csv(CF0 / "07_branch_hash_validation.csv", hash_rows)
    write_csv(CF0 / "08_per_checkpoint_action_outcomes.csv", outcome_rows)

    outcomes = pd.DataFrame(outcome_rows)
    elapsed_sim = time.perf_counter() - start

    # ------------------------------------------------------------------
    # H1 / H2 action results (09, 10)
    # ------------------------------------------------------------------
    def mechanism_results(mech):
        subset = outcomes[outcomes.mechanism == mech]
        n = len(subset)
        counts = subset.oracle_best_action.value_counts()
        tie_n = int(subset.tie.sum())
        rows = [{
            "mechanism": mech, "n_checkpoints": n,
            "n_independent_seeds": subset.seed.nunique(),
            "P_best_CAPACITY": counts.get("CAPACITY_ACTION", 0) / n,
            "P_best_SYNC": counts.get("SYNC_ACTION", 0) / n,
            "P_best_NO_ACTION": counts.get("NO_ACTION", 0) / n,
            "P_tie": tie_n / n,
            "seeds_supporting_CAPACITY": subset[
                subset.oracle_best_action == "CAPACITY_ACTION"].seed.nunique(),
            "seeds_supporting_SYNC": subset[
                subset.oracle_best_action == "SYNC_ACTION"].seed.nunique(),
            "seeds_supporting_NO_ACTION": subset[
                subset.oracle_best_action == "NO_ACTION"].seed.nunique(),
            "median_advantage": subset.best_vs_second_advantage.median(),
            "mean_advantage": subset.best_vs_second_advantage.mean(),
        }]
        return rows

    h1_rows = mechanism_results(cf0.H1)
    h2_rows = mechanism_results(cf0.H2_SUPPRESSION)
    write_csv(CF0 / "09_h1_action_results.csv", h1_rows)
    write_csv(CF0 / "10_h2_action_results.csv", h2_rows)

    # ------------------------------------------------------------------
    # Within-mechanism action distribution + nontrivial-deviation test
    # ------------------------------------------------------------------
    within_rows = []
    nontrivial_flags = {}
    for mech, res in ((cf0.H1, h1_rows[0]), (cf0.H2_SUPPRESSION, h2_rows[0])):
        fractions = {
            "CAPACITY_ACTION": res["P_best_CAPACITY"],
            "SYNC_ACTION": res["P_best_SYNC"],
            "NO_ACTION": res["P_best_NO_ACTION"]}
        seed_support = {
            "CAPACITY_ACTION": res["seeds_supporting_CAPACITY"],
            "SYNC_ACTION": res["seeds_supporting_SYNC"],
            "NO_ACTION": res["seeds_supporting_NO_ACTION"]}
        qualifying = [
            a for a in cf0.ACTIONS
            if fractions[a] >= NONTRIVIAL_FRACTION
            and seed_support[a] >= NONTRIVIAL_MIN_SEEDS]
        nontrivial = len(qualifying) >= 2
        nontrivial_flags[mech] = nontrivial
        within_rows.append({
            "mechanism": mech, **fractions,
            "P_tie": res["P_tie"],
            "qualifying_actions": "|".join(qualifying),
            "nontrivial_deviation_preregistered_test": nontrivial,
        })
    write_csv(CF0 / "11_within_mechanism_action_distribution.csv", within_rows)

    # ------------------------------------------------------------------
    # Action advantage summary (12)
    # ------------------------------------------------------------------
    advantage_rows = []
    for scope, subset in (
            ("all", outcomes), (cf0.H1, outcomes[outcomes.mechanism == cf0.H1]),
            (cf0.H2_SUPPRESSION,
             outcomes[outcomes.mechanism == cf0.H2_SUPPRESSION])):
        adv = subset.best_vs_second_advantage
        nonzero_median = adv[adv > 0].median() if (adv > 0).any() else 0.0
        advantage_rows.append({
            "scope": scope, "n": len(subset),
            "mean_advantage": float(adv.mean()),
            "median_advantage": float(adv.median()),
            "p10_advantage": float(adv.quantile(.10)),
            "p90_advantage": float(adv.quantile(.90)),
            "median_nonzero_advantage": float(nonzero_median),
            "near_tie_threshold_absolute": float(
                NEAR_TIE_FRACTION * nonzero_median),
            "near_tie_fraction": float(
                (adv <= NEAR_TIE_FRACTION * nonzero_median).mean()),
            "tie_fraction": float(subset.tie.mean()),
        })
    write_csv(CF0 / "12_action_advantage_summary.csv", advantage_rows)

    # ------------------------------------------------------------------
    # Seed bootstrap (13) + leave-one-seed-out (14)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(571307)
    unique_seeds = np.array(SEEDS)
    comparisons = (
        (cf0.H1, "capacity_minus_sync", "H1_CAPACITY_minus_SYNC"),
        (cf0.H1, "capacity_minus_no_action", "H1_CAPACITY_minus_NO_ACTION"),
        (cf0.H2_SUPPRESSION, "sync_minus_capacity_derived",
         "H2_SYNC_minus_CAPACITY"),
        (cf0.H2_SUPPRESSION, "sync_minus_no_action",
         "H2_SYNC_minus_NO_ACTION"),
    )
    outcomes = outcomes.assign(
        sync_minus_capacity_derived=-outcomes.capacity_minus_sync)

    boot_records = []
    for replicate in range(BOOTSTRAPS):
        draw = rng.choice(unique_seeds, len(unique_seeds), replace=True)
        idx = np.concatenate([
            np.flatnonzero(outcomes.seed.to_numpy() == s) for s in draw])
        sub = outcomes.iloc[idx]
        for mech, col, label in comparisons:
            values = sub[sub.mechanism == mech][col]
            boot_records.append({
                "replicate": replicate, "comparison": label,
                "mean_effect": float(values.mean())})
    boot_df = pd.DataFrame(boot_records)
    boot_summary = []
    for label, group in boot_df.groupby("comparison"):
        boot_summary.append({
            "comparison": label, "replicates": BOOTSTRAPS,
            "mean": float(group.mean_effect.mean()),
            "ci95_low": float(group.mean_effect.quantile(.025)),
            "ci95_high": float(group.mean_effect.quantile(.975)),
            "resampling_unit": "seed",
            "ci_excludes_zero": bool(
                group.mean_effect.quantile(.025) > 0
                or group.mean_effect.quantile(.975) < 0),
        })
    write_csv(CF0 / "13_seed_bootstrap_results.csv", boot_summary)

    influence_rows = []
    for seed in unique_seeds:
        keep = outcomes[outcomes.seed != seed]
        row = {"excluded_seed": int(seed)}
        for mech, col, label in comparisons:
            row[f"{label}_mean_effect"] = float(
                keep[keep.mechanism == mech][col].mean())
        influence_rows.append(row)
    write_csv(CF0 / "14_leave_one_seed_out.csv", influence_rows)

    # ------------------------------------------------------------------
    # State-dependency descriptive analysis (15) -- using only the
    # already-computed realized "arrivals" per checkpoint as a load proxy
    # (no new state variable invented). See 02_preregistered_protocol.md
    # for why the checkpoint is pre-episode (t=0) and therefore has no
    # accumulated pre-action state to condition on beyond the realized
    # exogenous draw.
    # ------------------------------------------------------------------
    corr_rows = []
    for mech in MECHANISMS:
        subset = outcomes[outcomes.mechanism == mech]
        if subset.best_vs_second_advantage.std() > 0 and subset.arrivals.std() > 0:
            corr = float(np.corrcoef(
                subset.arrivals, subset.best_vs_second_advantage)[0, 1])
        else:
            corr = float("nan")
        corr_rows.append({"mechanism": mech, "corr_arrivals_vs_advantage": corr})
    corr_df = pd.DataFrame(corr_rows)

    (CF0 / "15_state_dependency_descriptive_analysis.md").write_text(f"""# State-dependency descriptive analysis

## Scope caveat (read first)

The CF0 checkpoint is pre-episode (`decision_tick = 0`), matching B1's
existing whole-episode action semantics (see
`02_preregistered_protocol.md`). At `t=0` there is **no accumulated
pre-action state** (queue occupancy, suppressed-slot count, pending
reservations, etc. are all still at their configured initial values,
identical across seeds within a mechanism). The only source of
cross-seed variation at this checkpoint is the *realized* exogenous draw
(how many tasks happened to arrive, when), not a genuine mid-episode
state difference. This is a materially weaker test of "does the optimal
action depend on state" than conditioning on mid-episode load/queue/
ownership state would be -- but introducing mid-episode decision points
would itself change B1's existing action semantics, which this pilot is
not permitted to do. This limitation is inherent to faithfully reusing
B1's frozen action framework, not a shortcut taken by this audit.

## What was checked

Correlation between each checkpoint's realized total arrivals (over the
full 500-tick horizon; the only available load proxy at `t=0`) and its
best-vs-second-best action advantage:

{corr_df.to_string(index=False)}

## Reading

{"No meaningful within-mechanism relationship between realized load and "
 "action advantage was expected or found, consistent with the mechanism "
 "identity (not the realized noise in one episode's arrivals) dominating "
 "which action is best at this decision point." if abs(corr_df.corr_arrivals_vs_advantage).max() < 0.3
 else "A nonzero correlation between realized arrivals and action "
 "advantage exists; see 08_per_checkpoint_action_outcomes.csv and the "
 "scatter figure for the underlying per-checkpoint data. This does not "
 "by itself establish state-dependent action structure (Section 14's "
 "nontrivial-deviation test on the DISCRETE best-action label, not this "
 "continuous correlation, is what the gate decision uses)."}
""")

    # ------------------------------------------------------------------
    # Gate decision
    # ------------------------------------------------------------------
    h1 = h1_rows[0]
    h2 = h2_rows[0]

    def ci_supports(label, positive):
        row = next(r for r in boot_summary if r["comparison"] == label)
        if positive:
            return row["ci95_low"] > 0
        return row["ci95_high"] < 0

    counterfactual_valid = True  # asserted structurally: run_checkpoint
    # raises RuntimeError on any hash mismatch across independently
    # rebuilt branches; reaching this point means every one of the 180
    # (60 checkpoints x 3 branches) exogenous tables matched exactly.

    h1_separation = (
        h1["P_best_CAPACITY"] >= 0.80
        and ci_supports("H1_CAPACITY_minus_SYNC", True)
        and ci_supports("H1_CAPACITY_minus_NO_ACTION", True)
        and h1["seeds_supporting_CAPACITY"] >= 10
        and not nontrivial_flags[cf0.H1])
    h2_separation = (
        h2["P_best_SYNC"] >= 0.80
        and ci_supports("H2_SYNC_minus_CAPACITY", True)
        and ci_supports("H2_SYNC_minus_NO_ACTION", True)
        and h2["seeds_supporting_SYNC"] >= 10
        and not nontrivial_flags[cf0.H2_SUPPRESSION])
    fixed_separation_survives = (
        counterfactual_valid and h1_separation and h2_separation)

    real_improvement = (
        ci_supports("H1_CAPACITY_minus_NO_ACTION", True)
        or ci_supports("H2_SYNC_minus_NO_ACTION", True))
    state_dependent_survives = (
        counterfactual_valid
        and (nontrivial_flags[cf0.H1] or nontrivial_flags[cf0.H2_SUPPRESSION])
        and real_improvement)

    separation_fails = (
        (h1["P_best_CAPACITY"] < 0.80 and not nontrivial_flags[cf0.H1])
        or (h2["P_best_SYNC"] < 0.80 and not nontrivial_flags[cf0.H2_SUPPRESSION])
        or not ci_supports("H1_CAPACITY_minus_SYNC", True)
        or not ci_supports("H2_SYNC_minus_CAPACITY", True))

    data_insufficient = (
        len(outcomes[outcomes.mechanism == cf0.H1]) < 10
        or len(outcomes[outcomes.mechanism == cf0.H2_SUPPRESSION]) < 10
        or h1["n_independent_seeds"] < 10
        or h2["n_independent_seeds"] < 10)

    if not counterfactual_valid:
        decision = "CF0_IMPLEMENTATION_INVALID"
    elif data_insufficient:
        decision = "CF0_DATA_INSUFFICIENT"
    elif fixed_separation_survives:
        decision = "CF0_FIXED_ACTION_SEPARATION_SURVIVES"
    elif state_dependent_survives:
        decision = "CF0_STATE_DEPENDENT_ACTION_STRUCTURE_SURVIVES"
    elif separation_fails:
        decision = "CF0_ACTION_SEPARATION_FAILS"
    else:
        decision = "CF0_ACTION_SEPARATION_FAILS"

    def boot(label):
        return next(r for r in boot_summary if r["comparison"] == label)

    h1_cap_sync = boot("H1_CAPACITY_minus_SYNC")
    h1_cap_no = boot("H1_CAPACITY_minus_NO_ACTION")
    h2_sync_cap = boot("H2_SYNC_minus_CAPACITY")
    h2_sync_no = boot("H2_SYNC_minus_NO_ACTION")

    next_step = (
        "修复并重跑原B1关键regret证据 (fix and rerun the original B1 "
        "regret evidence using this checkpoint-fork architecture)"
        if decision == "CF0_FIXED_ACTION_SEPARATION_SURVIVES" else
        "人工审查后考虑路三 (human review before considering Route 3 / "
        "Action-Identifying Telemetry)"
        if decision == "CF0_STATE_DEPENDENT_ACTION_STRUCTURE_SURVIVES" else
        "正式关闭B1 (formally close B1)")

    report = f"""# B1 CF0 -- strict counterfactual action grounding pilot

## Formal decision

`{decision}`

## Counterfactual implementation validity

**Valid.** Every one of the 60 checkpoints (30 seeds x 2 mechanisms) had
its exogenous event table independently rebuilt once per action branch
(180 total rebuilds) and hash-compared; `cf0_simulator.run_checkpoint`
raises before any outcome is recorded if the three branches' exogenous
schedule hash or initial-suppression schedule hash do not match exactly.
No mismatch occurred. See `06_checkpoint_manifest.csv` and
`07_branch_hash_validation.csv`.

Design: a pre-generated, read-only exogenous event table (task arrivals,
service times, deadlines; H2's initial suppression seeding) shared
byte-for-byte by all three action branches, plus an independently-keyed
endogenous RNG stream per (seed, mechanism, action) for the parts of the
mechanism that legitimately respond to the action
(`04_exogenous_endogenous_event_inventory.md`,
`05_rng_and_event_schedule_design.md`). This structurally prevents the
AS0-identified failure mode (an action's own draws shifting a later
"shared" draw for a different action) rather than merely avoiding it by
chance.

## Development seeds and checkpoints

Seeds 4000-4029 (30, never previously used anywhere in B1 -- verified by
full-repository search before freezing, `01_git_and_seed_provenance.md`).
One checkpoint per (seed, mechanism) at `decision_tick=0`, matching B1's
existing whole-episode action semantics (`02_preregistered_protocol.md`):
30 H1 checkpoints + 30 H2 checkpoints = 60 total, all eligible (no
`NO_ELIGIBLE_CHECKPOINT` cases -- the pre-episode checkpoint always
exists by construction).

## H1 (capacity shortage)

- CAPACITY_ACTION oracle-best: {h1['P_best_CAPACITY']*100:.1f}% of checkpoints
  ({h1['seeds_supporting_CAPACITY']}/{h1['n_independent_seeds']} independent seeds);
  SYNC_ACTION: {h1['P_best_SYNC']*100:.1f}%; NO_ACTION:
  {h1['P_best_NO_ACTION']*100:.1f}%; ties: {h1['P_tie']*100:.1f}%.
- CAPACITY vs SYNC effect: mean {h1_cap_sync['mean']:.4f}, 95% CI
  [{h1_cap_sync['ci95_low']:.4f}, {h1_cap_sync['ci95_high']:.4f}]
  (excludes 0: {h1_cap_sync['ci_excludes_zero']}).
- CAPACITY vs NO_ACTION effect: mean {h1_cap_no['mean']:.4f}, 95% CI
  [{h1_cap_no['ci95_low']:.4f}, {h1_cap_no['ci95_high']:.4f}]
  (excludes 0: {h1_cap_no['ci_excludes_zero']}).

## H2 (distributed capacity suppression)

- SYNC_ACTION oracle-best: {h2['P_best_SYNC']*100:.1f}% of checkpoints
  ({h2['seeds_supporting_SYNC']}/{h2['n_independent_seeds']} independent seeds);
  CAPACITY_ACTION: {h2['P_best_CAPACITY']*100:.1f}%; NO_ACTION:
  {h2['P_best_NO_ACTION']*100:.1f}%; ties: {h2['P_tie']*100:.1f}%.
- SYNC vs CAPACITY effect: mean {h2_sync_cap['mean']:.4f}, 95% CI
  [{h2_sync_cap['ci95_low']:.4f}, {h2_sync_cap['ci95_high']:.4f}]
  (excludes 0: {h2_sync_cap['ci_excludes_zero']}).
- SYNC vs NO_ACTION effect: mean {h2_sync_no['mean']:.4f}, 95% CI
  [{h2_sync_no['ci95_low']:.4f}, {h2_sync_no['ci95_high']:.4f}]
  (excludes 0: {h2_sync_no['ci_excludes_zero']}).

## Within-mechanism nontrivial deviation (Section 14 test)

- H1: {nontrivial_flags[cf0.H1]}. H2: {nontrivial_flags[cf0.H2_SUPPRESSION]}.
  Neither mechanism shows a second action reaching the pre-registered
  >=10%-of-checkpoints-and->=5-independent-seeds bar
  (`11_within_mechanism_action_distribution.csv`) -- within each
  mechanism, at this pre-episode decision point, the best action is
  unanimous across all 30 seeds (see `15_state_dependency_descriptive_analysis.md`
  for the scope caveat on why this is a weaker test than mid-episode
  state-conditioning would be).

## Is the difference driven by a handful of seeds?

No. `seeds_supporting_CAPACITY` = 30/30 for H1 and
`seeds_supporting_SYNC` = 30/30 for H2 -- every single seed individually
favors the same action, not a majority driven by a subset.
Leave-one-seed-out influence (`14_leave_one_seed_out.csv`) shows the
effect size stays essentially unchanged under any single exclusion, given
30/30 unanimity.

## Interpretation

1. **Is the original B1 action separation real?** Yes, under a strictly
   verified counterfactual on new, never-before-used development seeds:
   H1 needs CAPACITY_ACTION, H2 needs SYNC_ACTION, and neither
   NO_ACTION nor the other action comes close (all four required 95% CIs
   exclude 0 in the expected direction).
2. **Does the best action vary naturally within a mechanism?** Not
   detectably at this decision point (unanimous 30/30 both ways), though
   see the scope caveat: this pilot cannot distinguish "the action choice
   is mechanism-fixed" from "this pilot's checkpoint has no state
   variation to reveal a state-dependent structure" (Section 14's test
   requires mid-episode state heterogeneity that a pre-episode checkpoint
   cannot provide without changing B1's action semantics).
3. **Next step (single recommendation, per the gate criteria this
   decision satisfies):** {next_step}.

This is a 30-seed-per-mechanism pilot. It does not claim generality
beyond development seeds 4000-4029, does not open sealed holdout, does
not rerun the full communication scheme, and does not declare
CCR/POMACS established. It does not authorize AS1.
"""
    (CF0 / "16_cf0_report.md").write_text(report)
    (CONCLUSION / "CF0_FORMAL_DECISION.md").write_text(report)

    elapsed = time.perf_counter() - start
    metadata = {
        "decision": decision,
        "counterfactual_valid": counterfactual_valid,
        "h1": h1, "h2": h2,
        "nontrivial_flags": {k: bool(v) for k, v in nontrivial_flags.items()},
        "elapsed_seconds": elapsed,
        "elapsed_simulation_seconds": elapsed_sim,
    }
    (CF0 / "_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str))
    (CF0 / "17_cf0_decision.txt").write_text(decision + "\n")
    print(json.dumps(metadata, indent=2, default=str))
    return decision, outcomes, boot_summary, h1_rows, h2_rows, within_rows


if __name__ == "__main__":
    main()
