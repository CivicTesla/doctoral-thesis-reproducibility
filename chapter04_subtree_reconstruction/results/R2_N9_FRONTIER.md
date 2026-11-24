# R2 nine-vertex frontier

## First order-four size

```text
min {|V(T)| : k(T)=4} = 9.
```

The lower bound is the eight-vertex Helly theorem.  The upper bound is the
R1 result `k(K1,8)=4`.

On nine vertices any order-three collision has volume-eight minimum trade
plus exactly one common row: zero common rows are ruled out by the raw-Helly
obstruction.  In the parity core, the common row must contain all four
selected direction labels.  Otherwise the odd leg still has four
pairwise-intersecting supports with empty total intersection.  Thus its core
padding pattern is necessarily `1111`.

## Exact constrained classification at order nine

The source audit enumerates all 47 unlabeled nine-vertex trees, all bijective
placements of the eight even and odd core rows, all possible common-row
patterns, and only connectedness of the four core supports.  It finds exactly
seven host shapes; each is realized with common pattern `1111`:

| shape | description |
|---|---|
| `DS(3,3)^+` | double star with three leaves at each centre, with one pendant edge subdivided |
| `DS(3,4)` | double star with three and four leaves at its two centres |
| `DS(2,4)^+` | double star with two and four leaves, with one pendant edge subdivided |
| `C(6;1,1,1,1,1)` | a degree-six centre with five leaf branches and one three-vertex cherry branch |
| `S6^(2)` | `K1,6` with two distinct pendant edges subdivided |
| `S7^(1)` | `K1,7` with one pendant edge subdivided |
| `K1,8` | the eight-leaf star |

The standard four-column parity trade plus the common all-one row explicitly
realizes every listed host.  Conversely, minimum-trade classification shows
that any order-three collision projects to this core, so a host excluded by
the constrained audit cannot have one.  Thus the table is an exact finite
classification, conditional only on the transparent exhaustive constrained
test in `src/trade_realizability.py`, not on an arbitrary-family census.

Consequently, among nine-vertex trees: the path has `k=2`; these seven hosts
have `k=4`; and every other host has `k=3`.
