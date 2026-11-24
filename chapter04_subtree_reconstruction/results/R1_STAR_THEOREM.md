# R1 exact star order

## Theorem

For every `d>=1`,

```text
k(K1,d) = max(2, floor(log_2 d)+1).
```

Here the leaves are host vertices, not column labels.

## Centre-containing reduction

Let `c` be the centre.  A connected subset of a star with cardinality at
least two contains `c`, hence is uniquely `{c} union S_i` for a leaf set
`S_i`.  Let `I` index these non-singleton labelled columns and set

```text
R_l = {i in I : l in A_i}       (l a leaf).
```

For nonempty `J subseteq I`,

```text
| intersection_(i in J) A_i | - 1 = #{l : J subseteq R_l}.
```

Thus order-`t` signature equality gives equal leaf-row containment moments
through degree `t`; the known number `d` supplies degree zero.  If `2^t>d`,
the trade lemma recovers the multiset of all `d` leaf rows, up to leaf
permutation.

## Attaching singleton columns

The diagonal separates `I` from the singleton columns.  For a singleton
label `j`, its pairwise profile against `I` is

```text
p_j=(|A_j intersection A_i| : i in I).
```

It is the all-one profile `u` for `{c}`, and `R_l` for `{l}`.  Among singleton
labels, a pairwise intersection of one is exactly the equivalence relation
"located at the same host vertex"; this also handles repeated singleton
labels.

The recovered leaf profile multiplicities provide enough vertices for every
singleton equivalence class of a non-all-one profile.  Classes of profile
`u` may be assigned to the centre or to any leaf with `R_l=u`.  Such vertices
have identical incidence across `I`, so arbitrary matching preserves every
labelled column.  After assigning singleton classes, match the remaining
vertices inside each profile class.  This is an active-ground-set isomorphism.

This explicitly covers: a centre singleton; leaf singletons; repeated
singletons; unused leaves (zero leaf rows); repeated leaf rows; leaf rows
equal to `u`; and the case `I=empty`.  In the last case the singleton
equality classes alone determine the active incidence family.  The argument
uses only degree two after the leaf rows are known.

Taking `t=max(2,floor(log_2 d)+1)` proves the upper bound.  For `d=1,2`
the star is a path, and for `d=3` it is the frozen claw positive host; all
have order two, agreeing with the formula.

## Parity-trade lower bound

For any `t>=1`, use `t+1` labelled centre-containing columns.  On `2^t`
leaves let the row blocks be respectively

```text
E = {C subseteq [t+1] : |C| is even},
O = {C subseteq [t+1] : |C| is odd}.
```

For every `S` with `|S|<=t`, fixing `S` leaves at least one free coordinate;
the two parities occur equally often.  Hence `(E,O)` is a `[t]`-trade of
volume `2^t`.  Put `A_i={c} union {l : i in R_l}` using `E` on one side and
`O` on the other.  The two star families have identical `Sigma_<=t`.  They
are not C2-isomorphic: every singleton support occurs as an active leaf row
on the odd side and on no even-side leaf (while the common centre row is the
full support).  Extra leaves for `d>2^t` are padded identically by zero rows.

For `d>=4`, set `t=floor(log_2 d)`.  Since `d>=2^t`, this collision gives
`k(K1,d)>t`; the upper bound gives equality.  This proves the theorem.

## Small sanity check

An exact integer check verified the parity construction for `t=1,2,3,4`:
both legs have volumes `2,4,8,16`, respectively, and all containment moments
through the stated order agree.  The R0 star-orbit computations also found no
order-three collision at `r=3` for `d=3,4,5,6`, consistent with the theorem.
