# B1 CF0 — scope and nonclaims

## What CF0 is

A single, limited, exit-capable pilot answering one question: under a
**strict** paired counterfactual (one frozen pre-action checkpoint, three
action branches sharing identical exogenous history), what are the real
action effects for H1 and H2, and does the best action vary naturally
within a mechanism?

## What CF0 is not

- Not classifier training. No logistic regression, random forest, CNN,
  Transformer, GNN, or RL model is fit anywhere in this pilot.
- Not telemetry compression, field selection, or message scheduling.
- Not AS1 (`AS1_ACTION_IDENTIFYING_TELEMETRY_AUDIT`). CF0 cannot authorize
  entry into AS1; at most it recommends human review of AS1 eligibility.
- Not a rerun of T0/T4/R4 or any prior gate's regret evidence.
- Not a mechanism-parameter change. H1/H2 dynamics, action semantics,
  action cost, utility weights, load, delay, loss, and task distribution
  are all reused byte-for-byte from the frozen R0 configs
  (`matched_config_h1.yaml`, `matched_config_h2.yaml`).
- Not a claim of generality. This is a 30-seed-per-mechanism pilot on new
  development seeds (4000–4029), not a full-scale confirmatory run, and
  not a claim that CCR/POMACS scope is established.

## What is allowed to change

Per the governing brief's Section 3, the only substantive addition this
round is: strict checkpoint branching, independent (pre-generated shared
exogenous + per-branch endogenous) random streams, and counterfactual
provenance instrumentation, layered on top of the existing H1/H2
simulators without altering their mechanism dynamics, action semantics,
cost function, or configuration.

## Historical results

AS0's conclusion (`AS0_COUNTERFACTUAL_REPLAY_INVALID`) and all prior
action-outcome numbers derived from the old (non-forked, single shared-RNG)
replay are preserved unchanged and are explicitly re-labeled here:

```text
HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY
```

No old CSV, report, or decision file is edited, appended to, or
overwritten by this pilot. CF0's own results live entirely under
`CF0_strict_counterfactual_grounding/`.
