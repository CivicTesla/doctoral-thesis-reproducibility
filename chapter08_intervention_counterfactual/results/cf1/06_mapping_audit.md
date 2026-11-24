# Mapping audit

## Why delay collapses to a many-to-one mapping

The strict counterfactual checkpoint fork (`cf0_simulator.py`, reused
unchanged from CF0) operates on the base H1/H2 action simulation
(`simulate()` / `simulate_suppression()`), which is entirely upstream of
and independent from the F2 communication scheme. Delay and packet loss
are properties of how the receiver *observes* the O0 state produced by
that simulation (`transmission()` in `run_phase5_final.py`), not
properties of the action simulation itself -- the same (seed, mechanism)
produces the same action-branch utilities regardless of which delay
condition a downstream row's F2 features were built from. Each old
`(seed, mechanism)` pair therefore has exactly **one** strict checkpoint,
shared by its delay=1 and delay=2 rows. This is a property of the frozen
simulator architecture, not a simplification introduced by this audit.

## Mapping results

- Total old prediction rows: 1600
- Mapped: 1600
- Unmapped: 0
- Mapping rate: 1.0000 (pre-registered minimum: 0.95)
- By mechanism: {'H1_CAPACITY_SHORTAGE': 1.0, 'H2_DISTRIBUTED_CAPACITY_SUPPRESSION': 1.0}
- By delay: {1: 1.0, 2: 1.0}

No mechanism, delay, configuration, or model-error type was systematically
excluded (see per-group rates above, all 1.0 in this run).
