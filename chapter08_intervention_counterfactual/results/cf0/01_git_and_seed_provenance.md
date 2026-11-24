# Git and seed provenance

- Base commit: `6308f19` (`Audit action non-degeneracy and counterfactual
  validity for B1`, on `b1-as0-action-nondegeneracy-audit`).
- New branch: `b1-cf0-strict-counterfactual-grounding`, created from a
  clean B1 working tree (only harmless `__pycache__` directories
  untracked; verified via `git status --short` restricted to the B1
  R0–R3 and AS0 directories before branching).
- No file under A1/A2/A3/C1/D1 or any other project is touched by this
  pilot.

## Frozen development seeds: 4000–4029

30 seeds, one block, used identically for both mechanisms (60
seed×mechanism checkpoints total before any eligibility filtering).

Verified unused anywhere in the repository before freezing:

- No YAML config (`R0_hidden_mechanism_audit/configs/*.yaml`,
  `results/*/matched_config_h*.yaml`,
  `R1_age_conditioned_decision_recovery/configs/*.yaml`) defines a seed
  range touching 4000–4029. Existing pools are: R0 pilot/KPI-matching
  seeds `0–4`, `100–104`, `200–202`, `300–302` (forbidden: `120–149`);
  R1 gate0 pilot seeds `1000–1004`/`1100–1104`/`1200–1204`; R1/R2/R3
  gate1-formal development seeds `3000–3399` (sealed: `2000–2029`).
- A full-repository search of every CSV column containing `seed` in its
  name for values in `[4000, 4029]` returned zero matches.

This range is frozen **before** any CF0 simulation is run and is not
revisited after seeing results.
