# Pre-holdout claim freeze

Written and committed BEFORE seeds 7400-7499 (holdout) are ever run, per
Section 6/23. Everything below is frozen: claim wording, metric
definitions, model protocols, threshold protocol, figures/tables to
report, success/failure criteria. Holdout may only confirm or weaken these
-- it may not change any of them.

## Main-scale (7000-7399, 400 seeds) results

See `11_pair1_method_results.csv`, `13_pair1_pairwise_results.csv`,
`14_pair1_matched_coverage.csv`, `16_pair2_method_results.csv`,
`17_pair2_pairwise_results.csv`, `18_harm_mass_results.csv`,
`19_risk_direction_prediction.csv`, `20_main_bootstrap.csv`,
`22_leave_one_seed_out.csv`, `24_final_claim_survival_matrix.csv`.

## Main-scale formal decision

`F0_PAIR2_CLAIM_WEAKENS`

## Pair 1: fully replicates the old CF1 pattern at 400-seed scale

- T4 BA gain over T0: +0.0409, 95% CI [0.0138, 0.0688] -- entirely > 0. **SURVIVES.**
- T4 H2 regret worse than T0: +1.232, CI [0.328, 2.156] -- entirely > 0. **SURVIVES.**
- T4-T0 total regret: -0.240, CI [-0.728, 0.281] -- crosses 0, as
  pre-registered (Claim P1-C never required significance; this matches
  the old CF1 finding that total regret does not significantly move).
  **NOT_CLAIMED, as expected.**
- R4 protects H2 relative to T4: -1.089, CI [-1.765, -0.483] -- entirely
  < 0. R4 harms H1 relative to T4: +1.458, CI [1.190, 1.735] -- entirely
  > 0. Both direction-stable -> **risk transfer SURVIVES.**
- Matched-coverage: T4-matched-to-R4's-coverage H2 regret (3.623) is far
  worse than R4's actual H2 regret (1.984) -- H2 protection is not just a
  coverage effect. R4's actual total regret (3.027) is worse than
  matched-T4's (2.949) -- R4 buys H2 protection at a net cost, matching
  risk-transfer, not free improvement. **SURVIVES.**

## Pair 2: accuracy-risk alignment WEAKENS at full scale

- M1 BA gain over M0: +0.193, CI [0.162, 0.224] -- entirely > 0. **SURVIVES.**
- M1-M0 total regret: -16.60, CI [-19.40, -13.95] -- entirely < 0 (improved). SURVIVES in isolation.
- M1-M0 H4 regret: -44.26, CI [-49.68, -38.87] -- entirely < 0 (strongly improved).
- **M1-M0 H3 regret: +11.06, CI [5.35, 16.76] -- entirely > 0 (significantly
  WORSE).** This fails Claim P2-B, which requires total *and* H3 *and* H4
  regret to all improve. By the exact same criterion the original P3
  100-seed pilot used to declare survival
  (`h3["ci95_low"] <= 0 and h4["ci95_low"] <= 0`, from
  `P3_independent_risk_validation/src/run_p3.py`), this main-scale result
  fails: H3's regret-change CI lower bound is 5.35, not <= 0.
- Total regret still improves only because H4's large improvement
  outweighs H3's significant worsening -- this is an asymmetric-harm
  pattern for M1 itself, not the clean "near-symmetric accuracy-risk
  alignment" the smaller pilot found. The 100-seed P3 pilot did not have
  the statistical power to detect this H3-specific reversal.
- M2 vs M1 total regret: +9.25, CI [7.32, 11.28] -- entirely > 0.
  REJECT -> NO_ACTION opportunity cost **SURVIVES.**
- Harm-mass identity: max absolute reconstruction error 3.6e-12 (< 1e-9). **SURVIVES.**

## Risk-direction prediction (Section 14)

Mechanism-level sign accuracy: 90.0% (pre-registered pass bar: >= 80%). **PASSES.**
Total-risk sign accuracy: 80.0% (pre-registered pass bar: >= 75%). **PASSES.**

## Forbidden after this point

No seeds may be added or removed. No mechanism, feature, threshold,
window, candidate grid, fold protocol, or cost constant may change. The
H3 finding above stands as reported regardless of what holdout shows;
holdout may only be reported as CONFIRMS or WEAKENS FURTHER relative to
this frozen main-scale finding, never used to retroactively soften or
remove it.

## Allowed paper wording after this freeze

- Pair 1: "critical-mechanism risk inversion" and "risk transfer" claims
  stand as originally stated, now confirmed at 400-seed scale.
- Pair 2: the accuracy-risk-alignment claim must be narrowed. Permitted:
  "M1 improves overall balanced accuracy and total regret, driven
  primarily by H4; H3 regret significantly worsens under M1 at full
  scale, an asymmetric-harm pattern not visible in the original 100-seed
  pilot." Forbidden: any wording claiming M1 improves H3 and H4 regret
  symmetrically, or that Pair 2 is free of mechanism-specific harm.
