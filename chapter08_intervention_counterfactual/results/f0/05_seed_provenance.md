# Seed provenance and freeze (B1 F0)

## Method

Full-repository search (`grep -rEn "range\([0-9]{3,5}, ?[0-9]{3,5}\)"` across all
`.py` files, plus a search for any bare 4-digit numbers in `.md`/`.csv`
provenance docs) for every seed range B1 has ever used, before selecting any
F0 seed.

## All historically-used seed ranges found

| Range | Used by |
|---|---|
| 120-149 | R0 Phase 5 final holdout (`forbidden_r0_seeds`, sealed) |
| 2000-2029 | R1/R2/CF0/CF1 sealed final test seeds (never opened) |
| 3000-3199 | Gate1 train split |
| 3200-3299 | Gate1 validation split |
| 3300-3399 | Gate1 development_test split |
| (3000-3399 combined) | R3 G0/G0B outer-CV development sample; CF1 T0/T4/R4 evaluation sample |
| 4000-4029 | CF0 checkpoint-fork pilot (30 dev seeds) |
| 5000-5029 | P2 radio-compute pilot |
| 5100-5199 | P3 independent validation (new seeds, P2 protocol reused) |
| 6000-6009 | (Simu5G scenario dev, P4A) |
| 6100-6159 | (Simu5G scenario dev, P4A) |
| 6200-6219 | P4B-R1 radio operating point audit traces |

No occurrence of any seed in 7000-7499, or in the smoke range 6990-6994, was
found anywhere in the repository (the one false-positive hit inspected --
`02_instrumented_replay_equivalence.csv` containing the substring "7000" --
was confirmed to be a `message_hash` hex fragment, not a seed column value;
that file's `seed` column spans exactly 3000-3399).

## Frozen F0 seed ranges

| Purpose | Range | Count |
|---|---|---|
| Smoke (never used in a formal result) | 6990-6994 | 5 |
| Main scale | 7000-7399 | 400 |
| Final untouched holdout | 7400-7499 | 100 |

These are disjoint from every range in the table above and from each other.
Frozen before any F0 run; not to be replaced, appended to, or dropped after
seeing results (Section 5/6 of the F0 brief).

## Same-seed joint generation

Every main/holdout seed generates strict 3-action outcomes for all four
mechanisms (H1, H2, H3, H4) so that seed-level bootstrap/leave-one-seed-out
analysis can treat (Pair 1, Pair 2) jointly per seed, per Section 5.
