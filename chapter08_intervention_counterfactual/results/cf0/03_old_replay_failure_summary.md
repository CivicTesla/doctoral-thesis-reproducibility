# Old replay failure summary (recap of AS0)

All figures in this section are carried over from
`AS0_action_nondegeneracy_audit/` unchanged and are re-labeled:

```text
HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY
```

## What AS0 found

1. The existing `oracle_action` field
   (`R2_mechanism_conditioned_risk_rescue/G1A_instrumented_replay_data/`
   `03_decision_state_manifest.csv`) is hardcoded as the mechanism-only
   mapping in `run_gate0_pilot.simulate_records()`
   (`preferred = "CAPACITY_ACTION" if mechanism == H1 else
   "SYNC_ACTION"`), not derived from any utility comparison.
2. Each action's outcome was produced by an independent, full,
   from-scratch `simulate()`/`simulate_suppression()` call for the whole
   `t = 0..horizon-1` episode — there was no shared pre-action checkpoint
   to fork from.
3. Both `simulator.py::simulate()` and
   `suppression_simulator.py::simulate_suppression()` use a single
   unnamed `random.Random(seed)` stream for every exogenous and
   action-conditioned draw.
4. For the H2 mechanism (`H2_DISTRIBUTED_CAPACITY_SUPPRESSION`), this was
   shown empirically to desync the exogenous arrival process itself: in
   100% of the 400 H2 development checkpoints tested, arrival counts
   differed across `NO_ACTION`/`CAPACITY_ACTION`/`SYNC_ACTION` for the
   identical seed. For H1 the specific matched config happened to avoid
   this (via `admission_intensity: 1.0`), but with no structural
   guarantee and no saved hash to verify it.

Formal decision: `AS0_COUNTERFACTUAL_REPLAY_INVALID`.

## What CF0 changes

CF0 does not patch the old replay or reinterpret its numbers. It builds a
new, independently-verified checkpoint-fork implementation
(`CF0_strict_counterfactual_grounding/src/`) on new development seeds
(4000–4029), with pre-generated shared exogenous event tables and
per-branch endogenous randomness, so that "only the action differs" can
actually be asserted and hashed rather than assumed.
