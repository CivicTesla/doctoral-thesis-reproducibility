# B1 CF1 — scope and nonclaims

## What CF1 is

A regeneration of T0/T4/R4 control regret using the CF0-verified strict
checkpoint-fork architecture, answering exactly one question: once "was
the chosen action wrong, and by how much" is computed from a real,
hash-verified counterfactual instead of the AS0-invalidated same-seed
independent-rerun numbers, do the old accuracy–control inversion and
risk-transfer conclusions survive?

## What CF1 is not

- Not a retraining of T0, T4, or R4. All three methods' per-row
  predictions (mechanism probability, accept/reject decision) are reused
  exactly as produced by the frozen scripts
  (`R3_temporal_mechanism_evidence/scripts/run_g0_temporal_audit.py`,
  `run_g0b_risk_constrained_convolution.py`), re-invoked read-only with
  their own already-recorded window/candidate selections — never
  re-selected, re-tuned, or re-fit against new data.
- Not a change to any feature, kernel, threshold, coverage rule, reject
  rule, risk weighting, fold, seed grouping, communication scheme, or
  H1/H2 mechanism. BA and coverage are asserted bit-identical to the
  archived G0/G0B values before any regret conclusion is drawn; the
  script halts if they are not.
- Not Route 3, not AS1, not new algorithm development, not paper writing.
- Not a claim that CCR/POMACS is finished. This is a regret-regeneration
  pilot restricted to the same seeds (3000–3399) and samples (1600 rows,
  delay ∈ {1, 2}) T0/T4/R4 were already evaluated on.

## Historical results

All regret figures computed by the old (AS0-invalidated) pipeline are
preserved unchanged and are explicitly re-labeled here:

```text
HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY
```

This includes the old T0/T4/R4 action regret, H1/H2 regret, mechanism-
switching regret, risk-transfer numbers, accuracy–control inversion
numbers, matched-coverage regret, and high-cost error regret. No old CSV
or report in `R3_temporal_mechanism_evidence/`, `AS0_action_nondegeneracy_
audit/`, or `CF0_strict_counterfactual_grounding/` is edited, appended
to, or overwritten. CF1's own results live entirely under
`CF1_strict_regret_regeneration/`.
