# R2 eight-vertex frontier

## Theorem

No eight-vertex tree has `k(T)=4`.  Hence every eight-vertex nonpath has
`k(T)=3`, while `P8` has `k(P8)=2`.

## Structural proof

By `R2_MINIMUM_TRADE.md`, an order-three collision would be an unpadded
minimum `[3]`-trade.  Select one label from each of its four independent
directions.  On one trade leg the eight host rows have the odd-parity patterns
on four coordinates, up to coordinate flips and permutation.  Its four
coordinate supports have these facts:

```text
each support has size 4;
every pair intersects in 2 vertices;
their fourfold intersection is empty.
```

The first two lines make the four connected supports pairwise intersecting.
The Helly property for subtrees of a tree then forces a common vertex,
contradicting the third line.  Coordinate flips merely exchange the even and
odd legs, so one leg always gives this contradiction. `square`

This proof is stronger than the constrained computation: a raw minimum
`[3]`-trade is not bi-realizable on **any** tree.
