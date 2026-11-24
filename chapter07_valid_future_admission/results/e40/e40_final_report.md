# E40 final report

## Status

E40 completed: all required campaign files were present before figures were generated.
All seven qualitative claim checks: PASS.

## Algorithm change

Raw DC-tranche scores for Prefix Diverse VFA and PD no-rollout are min-max normalized over the current candidate pool before prefix confidence and soft diversity. AoI, residual, RGSA, FQ, Crit-TopK, Route-ref., valid-future filtering, and field weighting retain their specified paths.

## Verification

The normalized DC diversity input is nonnegative; the denominator is monotone. See `e40_invariant_check_report.md` and `e40_negative_score_postfix_audit.md`.

## Campaigns

Pilot, main validation, budget sweep, channel stress, online learning, runtime, large-fleet proportional validation, and fixed-scarcity validation completed under E40 output paths.

## Preservation

Old results and TeX were not overwritten.

## Recommended next action

Review the old-vs-new comparison and patch suggestions, then explicitly authorize a TeX update if the claim checks are acceptable.
