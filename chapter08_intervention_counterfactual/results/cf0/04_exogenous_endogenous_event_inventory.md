# Exogenous / endogenous event inventory

Classified from the frozen source (`R0_hidden_mechanism_audit/src/simulator.py`
and `.../src/suppression_simulator.py`), before any new code is written.
This classification is fixed before implementation and is not revisited
after seeing pilot results.

## H1 (`simulator.py::simulate`, matched config: `admission_intensity: 1.0`)

### Exogenous (must be identical across all 3 action branches)

| Event | Original line | Why exogenous |
|---|---|---|
| Task arrival (per tick, per node) | `if rng.random() < arrival_rate` | Unconditional every tick/node, no `action` or capacity term in the condition. |
| Service time (if arrival) | `service = _geometric(rng, service_time_mean)` | Drawn immediately after an arrival, before any admission/capacity logic runs. |
| Deadline offset (if arrival) | `deadline = t + _geometric(rng, deadline_mean)` | Same as above. |
| Admission throttle (per tick, per node) | `if admission_intensity < 1.0 and rng.random() >= admission_intensity` | The intensity value itself does not depend on `action`; only *whether the branch reaches this line* depended on action-driven queue occupancy in the old code (the AS0 desync mechanism for H1-shaped configs). CF0 pre-generates this draw unconditionally per (tick, node) so it no longer gates on branch-specific queue state. Inert under the current config (`admission_intensity == 1.0`), kept for correctness under other configs. |

### Endogenous (legitimately differs by action; uses a per-branch stream)

| Event | Why endogenous |
|---|---|
| Which node's queued request is admitted this tick | Depends on `capacity` (action-dependent: `CAPACITY_ACTION` adds `capacity_extra_slots`) and `believed_busy`, both state, not randomness — no RNG draw here, deterministic given state. |
| Completion timing, `holders`, `believed_busy` | Deterministic function of admitted requests and `service` (itself exogenous) — differs by action only through *which* requests got admitted, not through new randomness. |

H1 has no other action-conditioned RNG draws in the current mechanism
(`mechanism == H2` release-loss block in `simulator.py` is dead code for
this pipeline — `simulate_records` never calls `simulate()` with
`mechanism=H2`).

## H2 (`suppression_simulator.py::simulate_suppression`)

### Exogenous (must be identical across all 3 action branches)

| Event | Original line(s) | Why exogenous |
|---|---|---|
| Initial suppression seeding (t=0, per physical slot) | lines 55–60: `if rng.random() < initial_probability: cause = ...; suppressed[slot] = [...]` | Runs once, before the tick loop, unconditionally on `action`. |
| Task arrival (per tick, per node) | `if rng.random() < arrival_rate` | Unconditional every tick/node. |
| Service time / deadline (if arrival) | `geometric(rng, service_time_mean)`, `t + geometric(rng, deadline_mean)` | Drawn immediately after arrival, before any action-gated logic. |

### Endogenous (legitimately differs by action; uses a per-branch stream)

| Event | Original line(s) | Why endogenous |
|---|---|---|
| Suppression-state clear vs. decay/reconciliation/re-suppression | `if action == "SYNC_ACTION": suppressed.clear() else: [decay, rng.random() < 1/interval reconciliation, rng.choice, rng.shuffle, per-slot re-suppression draws]` | Explicitly gated on `action`; this **is** the mechanism's causal response to synchronization, not noise. |
| Re-suppression after a release | `for slot in released: if rng.random() < inconsistent_holder_probability: suppressed[slot] = [...]` | Inside the same `else` (non-`SYNC_ACTION`) block — action-gated. |
| Conservative-admission denial | `if action != "SYNC_ACTION" and free_usable and rng.random() < conservative_admission_probability` | Explicitly action-gated. |
| Backoff duration after a denial | `next_retry[node] = t + geometric(rng, backoff_mean)` | Only drawn when a denial occurs, and denials are action-dependent (see above). |

### Not applicable at this layer

- **Channel/packet-loss variates**: the action-level H1/H2 simulators
  operate on true O0 state, before the F2 receiver/communication scheme
  is applied. There is no channel model inside `simulate()` /
  `simulate_suppression()`. `07_branch_hash_validation.csv`'s
  `channel_schedule_hash` column is repurposed (documented there) to
  carry the H2 initial-suppression-seeding schedule hash, the nearest
  actual pre-generated stochastic schedule this layer has; for H1 it is
  reported as `NOT_APPLICABLE_NO_CHANNEL_AT_ACTION_LAYER`.

This split is unambiguous for both mechanisms under the frozen matched
configs, so `CF0_IMPLEMENTATION_INVALID` is not triggered by Section 7's
"cannot cleanly distinguish" clause.
