# Registered R0 analysis

**Status:** `WITNESS_PASS`.

All intervals use the pre-locked methods in `EXECUTION_LOCK.md`.  This report does not interpret a coupling contrast as a population marginal mean action-value bias.

## Endpoint diagnostic

- KAPPA_PLUS: endpoint-difference SD 2.652014 dB; canonical disagreement 428/2000 = 0.2140 (95% CI 0.1966, 0.2325).
- KAPPA_ZERO: endpoint-difference SD 5.165566 dB; canonical disagreement 900/2000 = 0.4500 (95% CI 0.4283, 0.4719).
- KAPPA_MINUS: endpoint-difference SD 6.931047 dB; canonical disagreement 1319/2000 = 0.6595 (95% CI 0.6384, 0.6799).

## Primary status evidence

- `KAPPA_PLUS_MINUS_KAPPA_ZERO`: `terminal_absolute_contrast`; estimate -63021941.18785264; 95% CI [-68038921.40083823, -58004960.97486706].
- `KAPPA_PLUS_MINUS_KAPPA_ZERO`: `terminal_selection_disagreement`; estimate 0.449; 95% CI [0.4273196809003545, 0.4708758579223333].
- `KAPPA_PLUS_MINUS_KAPPA_MINUS`: `terminal_absolute_contrast`; estimate -109844118.13281462; 95% CI [-116031790.30949444, -103656445.9561348].
- `KAPPA_PLUS_MINUS_KAPPA_MINUS`: `terminal_selection_disagreement`; estimate 0.562; 95% CI [0.5401577267273363, 0.5836045594097885].
- `KAPPA_ZERO_MINUS_KAPPA_MINUS`: `terminal_absolute_contrast`; estimate -46822176.94496197; 95% CI [-52917679.23339927, -40726674.65652467].
- `KAPPA_ZERO_MINUS_KAPPA_MINUS`: `terminal_selection_disagreement`; estimate 0.294; 95% CI [0.274443321398105, 0.31434650208177106].

Detailed arm results are in `summary.json`; all locked paired contrasts are in `paired_contrasts.csv`.
