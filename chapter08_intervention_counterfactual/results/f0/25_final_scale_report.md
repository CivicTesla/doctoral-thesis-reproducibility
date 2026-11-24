# B1 F0 -- final formal decision

`F0_FINAL_SCALE_VALIDATION_PARTIAL`

## Summary

Pair 1 (H1/H2, T0/T4/R4): core critical-mechanism risk-inversion and
risk-transfer claims are direction-stable across main (400 seeds) and
holdout (100 seeds), though some holdout CIs are only marginally
significant (expected given holdout is 1/4 the size of main). T4's BA
gain over T0, solidly significant at main scale, is not statistically
significant at holdout scale (CI crosses zero) -- reported as a genuine
weakening of that specific supporting claim, not suppressed.

Pair 2 (H3/H4, M0/M1/M2): M1's aggregate BA gain and total-regret
improvement are stable across main and holdout. But the per-mechanism
breakdown is NOT reproducible: at main scale M1 significantly *worsens*
H3 regret while massively improving H4; at holdout M1 significantly
*improves* H3 regret with no significant H4 effect. This sign
inconsistency in the mechanism-level decomposition means the original
"near-symmetric accuracy-risk alignment" claim from the 100-seed P3
pilot does not survive final-scale validation with an independent
holdout, even though the top-line aggregate metrics do.

Harm-mass identity reconstructs exactly in both main and holdout.
Risk-direction prediction meets its pre-registered thresholds.

Simu5G external validation remains not established (frozen closed,
`00_external_validation_closure.md`).

See `24_final_claim_survival_matrix.csv` for the per-claim table and
`23_main_vs_holdout_comparison.csv` for the main-vs-holdout CI comparison
underlying every verdict above.
