# Frozen scientific claims entering F0

These are the standing B1 results F0 must validate at final scale without
modifying mechanism, method, feature, threshold, or claim wording (per the
task brief's 10-point standing state):

1. H1/H2 strict action separation holds (CF0/CF1).
2. CF1 critical-mechanism risk inversion holds (T4 BA gain + T4 H2 regret
   worse than T0, seed-bootstrap CI entirely positive).
3. T4 improves BA but significantly worsens high-cost H2 regret.
4. R4 protects H2 but harms H1 -- risk transfer, not free improvement.
5. P0 harm-mass risk decomposition holds (exact identity reconstruction).
6. H3/H4 independent mechanism pair holds (P2/P3).
7. P3: under near-symmetric costs, accuracy-risk alignment holds (M1 BA
   gain aligns with M1 regret improvement, no inversion).
8. REJECT -> NO_ACTION carries an opportunity cost (M2 vs M1).
9. Simu5G external validation is incomplete (frozen closed, Section 00).
10. No further simulation platforms after F0.

## Frozen, not to be re-derived

- H1/H2/H3/H4 mechanism dynamics, action definitions, and action-cost/utility
  weights (Section 3 of the F0 brief).
- T0/T4/R4 feature definitions, window/convolution definitions, per-fold
  outer-CV protocol, cost-sensitive fitting, candidate-selection and
  fallback-to-T0 rule, REJECT/coverage rule.
- M0/M1/M2 feature definitions, logistic regression fitting, M2's
  training-fold threshold protocol, REJECT -> NO_ACTION rule.
- Bootstrap unit (seed), delay handling (T0/T4/R4 delay in {1,2}; delay=0
  absent from the historical evaluation sample and remains absent for F0's
  extension, since the frozen decision-state generation only ever runs the
  two primary conditions for T0/T4/R4 -- see `03_preregistered_protocol.md`).

## Explicitly permitted for F0

Re-fitting T4/R4 per outer training fold on the new seed set, using the
frozen algorithm/protocol unchanged (`run_g0_temporal_audit.fit_classifier`,
`run_g0b_risk_constrained_convolution.fit_classifier_costsensitive`) --
this produces new fold-specific fitted coefficients, which is distinct from,
and does not constitute, a hyperparameter or protocol change. T0 remains a
direct read of the frozen `gate1_frozen_receiver_models.json` age-aware
model applied to the new seeds' own decision-state features (T0 was never a
per-fold-refit method in the original protocol, so it is not refit here
either -- see `01_frozen_artifact_manifest.csv` notes on
`gate1_frozen_receiver_models.json`).
