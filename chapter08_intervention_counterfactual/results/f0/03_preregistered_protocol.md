# F0 preregistered protocol

Frozen before any main-scale run. Any deviation must be logged as a protocol
violation, not silently applied.

## Seeds (see `05_seed_provenance.md`)

- Smoke: 6990-6994 (5 seeds, discarded, never enters a formal result).
- Main: 7000-7399 (400 seeds).
- Holdout: 7400-7499 (100 seeds), run exactly once, after main results and
  claim wording are frozen and committed.

## Pair 1 (H1/H2, T0/T4/R4)

- Action simulation: `cf0_simulator.run_checkpoint(config, mechanism, seed)`
  unchanged, `matched_config_h1.yaml` / `matched_config_h2.yaml` unchanged.
  This gives, per (seed, mechanism), a strict 3-action branch
  (NO_ACTION/CAPACITY_ACTION/SYNC_ACTION) with hash-verified shared
  exogenous schedule -- exactly CF0's own protocol, extended from 30 to 400
  (main) + 100 (holdout) seeds.
- T0/T4/R4 telemetry: `run_g1a_instrumented_replay.py`'s own functions
  (`simulate_records`, `transmission`) reused unchanged, but pointed at the
  new seed set in place of the historical train/validation/dev split. Only
  the **2 primary conditions** (delay=1, delay=2; loss=0.0) are run for new
  seeds -- the 4 auxiliary conditions in the original protocol existed only
  to cross-check against the archived Gate1 prediction file, which has no
  counterpart for brand-new seeds, so running them would produce no
  evaluable output (confirmed before running any new seed; see
  `00_external_validation_closure.md`-style scoping note in
  `04_pre_holdout_claim_freeze.md`).
- T0: frozen `gate1_frozen_receiver_models.json` age-aware model applied
  as-is (`serialized_probability`) to the new seeds' own decision-state
  features. Never refit -- T0 was never a per-fold method.
- T4: `run_g0_temporal_audit.fit_classifier` refit per outer fold on the new
  seed set (5 folds, `fold = (seed - 7000) % 5` for main, mirroring the old
  `(seed - 3000) % 5`), using the same static+convolution feature
  construction and the same window-length selection procedure recorded in
  the old `02_window_manifest.csv` protocol (the selection *procedure* is
  reused; new seeds get their own selected window per fold, which is a
  fold-specific fitted choice, not a hyperparameter change).
- R4: `run_g0b_risk_constrained_convolution.fit_classifier_costsensitive` +
  candidate selection + fallback-to-T0 rule, same protocol, refit per fold
  on the new seed set.
- Strict regret: CF1's own substitution (`run_cf1_regret.decide`,
  `REJECT_COST=5.0`, h1/h2 wrong-regret constants from the same frozen
  `gate1_frozen_receiver_models.json`) applied to the new checkpoint-fork
  outcomes, unchanged.

## Pair 2 (H3/H4, M0/M1/M2)

- `P2_radio_compute_pair_pilot/src/pilot.py` reused unchanged (`CFG`,
  `task_table`, `schedules`, `step`, `outcome`, `features`, `fit_logistic`),
  exactly as P3 reused P2 byte-for-byte, extended from P3's 100 seeds to
  400 (main) + 100 (holdout).
- M0/M1/M2 fit/threshold protocol from `P3_independent_risk_validation/
  src/run_p3.py` (`metric`, `decomposition`, outer 5-fold
  `fold = seed % 5`, M2's 20th-percentile training-fold threshold) reused
  unchanged.

## Metrics and comparisons

Pair 1: BA, coverage, accepted accuracy, total/H1/H2 strict regret,
H1<->H2 error counts, mean cost per directional error, directional cost
ratio. Comparisons: T4-T0, R4-T4, R4-T0. Claims P1-A..E per Section 11 of
the F0 brief.

Pair 2: BA, coverage, H3/H4 coverage, total/H3/H4 strict regret, directional
cost ratio. Comparisons: M1-M0, M2-M1, M2-M0. Claims P2-A..E per Section 12.

## Harm-mass identity

For every method transition in both pairs: Corrected Harm Mass, New Harm
Mass, Coverage Loss Mass, Coverage Gain Mass, reconstructed against
`n_h * delta_regret_h`; maximum absolute reconstruction error must be
< 1e-9 (Section 13), else `F0_HARM_MASS_RECONSTRUCTION_FAILED`.

## Risk-direction prediction

Outer-fold training-cost-only prediction of regret-change sign (mechanism
and total level) for both pairs jointly. Pass bar: mechanism-level sign
accuracy >= 80%, total-risk sign accuracy >= 75% (Section 14). Holdout signs
predicted from main-scale costs before holdout results are revealed.

## Bootstrap

2000 seed-level bootstrap replicates; seed is the resampling unit and pulls
both pairs / all mechanisms / all actions / all prediction rows for that
seed together. Leave-one-seed-out reported alongside every bootstrap CI.

## Formal decision set

Exactly one of `F0_FINAL_SCALE_VALIDATION_SURVIVES`,
`F0_FINAL_SCALE_VALIDATION_PARTIAL`, `F0_PAIR1_CLAIM_WEAKENS`,
`F0_PAIR2_CLAIM_WEAKENS`, `F0_RISK_CRITERION_PREDICTION_FAILS`,
`F0_COUNTERFACTUAL_INTEGRITY_FAILED`, `F0_HPC_EXECUTION_INCOMPLETE`, per
Section 20 of the F0 brief.

## Execution environment note

No Slurm cluster is reachable from this environment (verified:
`sbatch`/`squeue`/`sacct` all absent). Per user decision, the full main
(400-seed) and holdout (100-seed) matrices are executed locally on a single
4-core workstation, honestly labeled as local execution rather than a
Slurm submission -- both mechanism simulators involved (`cf0_simulator.py`
action layer + G1A/F2 telemetry layer; `pilot.py`) are lightweight synthetic
Python, not heavy network simulators, and the old-seed reproduction run
above (Task 20) demonstrates ~6 minutes per 400-seed batch for the heavier
Pair 1 pipeline alone. The HPC script suite (`hpc/`) is still written in
full, portable form for use on a real Slurm cluster if one becomes
available later.
