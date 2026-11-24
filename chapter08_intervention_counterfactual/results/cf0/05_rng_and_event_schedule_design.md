# RNG and event-schedule design

Implements Section 6's "recommended: pre-generated exogenous event table"
approach, combined with stable-keyed per-branch endogenous streams for the
parts classified endogenous in `04_exogenous_endogenous_event_inventory.md`.

## Stream derivation

All streams are seeded deterministically from a purpose tag, never from
sequential consumption of a single shared generator:

```python
def derived_seed(*parts) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).digest()
    return int.from_bytes(digest[:8], "big")

def keyed_rng(*parts) -> random.Random:
    return random.Random(derived_seed(*parts))
```

## Exogenous event table (one per seed, per mechanism — shared read-only by
all three action branches)

Generated **once**, before any action branch runs, using
`keyed_rng(seed, mechanism, "exogenous")`:

- H1: for `t in 0..horizon-1`, `node in 0..n-1`: `arrival[t,node]` (bool),
  and if true, `service_time[t,node]`, `deadline[t,node]`; plus
  `admission_throttle_pass[t,node]` (bool, drawn unconditionally,
  independent of whether it is ever consulted by a branch).
- H2: initial suppression seeding at t=0 for each of the `base_capacity`
  physical slots (`suppressed_at_init[slot]`, `cause[slot]`,
  `lifetime[slot]` where seeded); plus the same arrival/service/deadline
  table as H1 (H2's own arrival_rate/service_time_mean).

This table is serialized to a canonical byte representation and hashed
(`sha256`) once. All three action branches read from the identical frozen
table object — they never draw from it, so no branch can shift its
position.

## Endogenous per-branch streams

Each action branch gets its own independently-keyed stream:

```python
keyed_rng(seed, mechanism, action, "endogenous")
```

Used only for the draws classified endogenous in the event inventory
(suppression maintenance, conservative-admission denial, backoff
duration, and H2's release-triggered re-suppression). Because each
branch's endogenous stream is independent of the others' *and* of the
shared exogenous table, an action's effect on how many endogenous draws
occur can never shift another branch's exogenous events — the specific
bug identified in AS0 (H2 arrivals diverging by action) is structurally
impossible under this design, not just empirically rare.

## Why not literal byte-identical pre-action *state*

At the chosen checkpoint (`t=0`, before any tick executes — see
`02_preregistered_protocol.md` for why), "pre-action state" is just
`(config, seed)`, with no accumulated queues/holders/suppression state
yet other than the exogenous table's initial suppression seeding, which
is itself part of the shared, hashed, read-only exogenous table. The
meaningful identity check is therefore on the **exogenous schedule
hash**, not on a nontrivial mid-episode state snapshot; this is recorded
in `06_checkpoint_manifest.csv` / `07_branch_hash_validation.csv`.
