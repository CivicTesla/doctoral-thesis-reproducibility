# Git and artifact provenance

- Base commit: `9c09d7c` (`Establish strict counterfactual action
  grounding for B1`, on `b1-cf0-strict-counterfactual-grounding`).
- New branch: `b1-cf1-strict-regret-regeneration`, created from a clean
  B1 working tree (only harmless `__pycache__` directories untracked).
- No file under A1/A2/A3/C1/D1 or any other project is touched.
- No sealed seed (2000–2029) or forbidden R0 seed (120–149) is loaded —
  asserted programmatically in `run_cf1_checkpoints.py` before any
  simulation.

## Frozen artifacts reused (see `02_frozen_model_manifest.csv` for hashes)

- **T0**: `R1_age_conditioned_decision_recovery/results/
  gate1_formal_age_necessity/gate1_frozen_receiver_models.json`
  (age-aware model). Predictions read directly from
  `mechanism_probability_H2` in the decision-state manifest — never
  refit.
- **T4**: `R3_temporal_mechanism_evidence/scripts/run_g0_temporal_audit.py`
  (unmodified), re-invoked using the window per outer fold already
  recorded in `02_window_manifest.csv`.
- **R4**: `R3_temporal_mechanism_evidence/scripts/
  run_g0b_risk_constrained_convolution.py` (unmodified), re-invoked using
  the `(w_H2, b_H2)` selection per outer fold already recorded in
  `05_selected_candidate_by_outer_fold.csv`.
- **Checkpoint-fork simulator**: `CF0_strict_counterfactual_grounding/
  src/cf0_simulator.py` (unmodified, imported directly) — the same
  313/313-tested architecture CF0 validated on seeds 4000–4029, applied
  here to the 800 (seed, mechanism) pairs seeds 3000–3399 actually needed.
- **Cost constants**: `h1_wrong_regret_estimate`, `h2_wrong_regret_estimate`,
  `reject_cost` read from the same frozen
  `gate1_frozen_receiver_models.json` used throughout B1 — reused only to
  reproduce the existing accept/reject *decision*, never as the regret
  *magnitude* (that role is now filled by real per-checkpoint strict
  counterfactual utilities).

Per-row prediction reproduction (`04_prediction_reproduction.csv`):
balanced accuracy, coverage, and (old, non-strict) total regret for T0,
T4, and R4 all match the archived `07_method_summary.csv` values to
floating-point exactness (`abs_difference < 1e-9` on every one of the 9
checks), confirming the regenerated per-row predictions are identical to
the original run before any new regret computation is layered on top.
