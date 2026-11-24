# State-dependency descriptive analysis

## Scope caveat (read first)

The CF0 checkpoint is pre-episode (`decision_tick = 0`), matching B1's
existing whole-episode action semantics (see
`02_preregistered_protocol.md`). At `t=0` there is **no accumulated
pre-action state** (queue occupancy, suppressed-slot count, pending
reservations, etc. are all still at their configured initial values,
identical across seeds within a mechanism). The only source of
cross-seed variation at this checkpoint is the *realized* exogenous draw
(how many tasks happened to arrive, when), not a genuine mid-episode
state difference. This is a materially weaker test of "does the optimal
action depend on state" than conditioning on mid-episode load/queue/
ownership state would be -- but introducing mid-episode decision points
would itself change B1's existing action semantics, which this pilot is
not permitted to do. This limitation is inherent to faithfully reusing
B1's frozen action framework, not a shortcut taken by this audit.

## What was checked

Correlation between each checkpoint's realized total arrivals (over the
full 500-tick horizon; the only available load proxy at `t=0`) and its
best-vs-second-best action advantage:

                          mechanism  corr_arrivals_vs_advantage
               H1_CAPACITY_SHORTAGE                   -0.350969
H2_DISTRIBUTED_CAPACITY_SUPPRESSION                    0.461682

## Reading

A nonzero correlation between realized arrivals and action advantage exists; see 08_per_checkpoint_action_outcomes.csv and the scatter figure for the underlying per-checkpoint data. This does not by itself establish state-dependent action structure (Section 14's nontrivial-deviation test on the DISCRETE best-action label, not this continuous correlation, is what the gate decision uses).
