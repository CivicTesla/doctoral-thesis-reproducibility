# R4 human nine-vertex classification

This note humanizes the last R2 branch-state calculation. It uses the frozen
reduction: a nine-vertex order-three collision is a parity [3]-trade plus one
common all-one vertex h. The raw trade is excluded by subtree Helly, and the
padding row is forced to be 1111.

Delete h. By the odd-side analysis in R4_PARITY_POSETS.md, the remaining eight
vertices are isolated vertices and rooted stars. Apply the even-side criterion
there. The allowed star-size multisets and the corresponding unrooted tree are:

| odd-star sizes | host |
|---|---|
| none | Q1 = K1,8 |
| 1 | Q2 |
| 2 | Q4 |
| 3 | Q5 = DS(3,4) |
| 1,1 | Q3 |
| 2,1 | Q7 |
| 3,1 | Q6 |

Here h is adjacent to the root of every star and to every isolate. Reading
degrees gives exactly the canonical descriptions in R3_SEVEN_HOSTS.md. The
excluded cases are equally explicit: three nontrivial stars exhaust the unique
empty label, and two stars of sizes at least two both require the unique [4]
label.

Conversely the constructions in the proof of the even-side criterion, with
the stated odd-side triple/singleton assignments, realize every row on the
displayed host. Rooted monotonicity makes all four supports connected.

Thus a nine-vertex tree admits the padded affine parity trade if and only if it
is Q1,...,Q7. Coupled with the already frozen minimum-trade reduction, this is
a fully human proof of the nine-vertex classification; the old enumerator is
now only an independent check, not a logical dependency.
