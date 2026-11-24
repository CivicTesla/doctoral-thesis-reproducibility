# R1 trade reduction

## Padded-row proposition

Fix a host with `n` vertices and an indexed family `A=(A_1,...,A_r)`.  Put

```text
R_v(A) = { i in [r] : v in A_i }.
```

Keep all `n` rows, including `R_v=empty`.  For every nonempty `S subseteq [r]`,

```text
| intersection_(i in S) A_i | = #{v : S subseteq R_v(A)}.
```

Thus, if `Sigma_<=t(A)=Sigma_<=t(B)`, then the two row multisets have equal
containment counts for every nonempty `S` of size at most `t`.  Their order-zero
counts also agree, because each padded multiset has exactly `n` rows.

Cancel each common row type with its minimum multiplicity.  If a difference
remains, write its positive and negative parts as `T_+` and `T_-`.  They are
disjoint **multisets** of subsets of `[r]` and satisfy

```text
#{C in T_+ : S subseteq C} = #{C in T_- : S subseteq C}
                                      (|S|<=t),
```

including `S=empty`.  Hence they are a nonempty signed/multiset `[t]`-trade of
volume at most `n`.  Empty blocks are allowed in this statement.  This is an
exact reduction, not merely an analogy.

## Multiset trade lemma (self-contained scope audit)

Let `f:2^[r] -> Z` be nonzero and suppose

```text
sum_(C supseteq S) f(C)=0                 for every |S|<=t.
```

Then

```text
sum_C |f(C)| >= 2^(t+1),
sum_C f(C)^+ = sum_C f(C)^- >= 2^t.
```

Proof.  Induct on the number of coordinates, with the `t=0` case following
from `sum f=0` and integrality.  Split at a coordinate `j` into the functions
`P(C)=f(C)` and `Q(C)=f(C union {j})` on the remaining coordinates.  The
conditions say that `P+Q` is a `[t]`-trade function, while both `P` and `Q`
are `[t-1]`-trade functions.  If exactly one of `P,Q` vanishes, the other is a
nonzero `[t]`-trade on one fewer coordinate, so the coordinate-induction
hypothesis applies.  If both are nonzero, the order-`t-1` induction gives
`||P||_1,||Q||_1 >= 2^t`; their supports lie in the two coordinate slices, so
`||f||_1=||P||_1+||Q||_1 >= 2^(t+1)`.  Order zero gives `sum f=0`, hence the
positive and negative `l1` masses agree. `square`

This proof permits repeated blocks (integer coefficients of arbitrary
magnitude), variable block sizes, and the empty block.  It therefore covers
the cancelled incidence-row difference without importing any unstated
uniformity or simplicity assumption.

## Relation to the C2 equivalence notion

If the padded row multisets agree, choose a bijection of all host vertices
matching equal row types.  It maps every labelled set `A_i` to `B_i`.
Restricting it to nonzero rows gives the active-ground-set isomorphism used by
C2.  Therefore padded-row recovery is stronger than the frozen C2
reconstruction conclusion; no host automorphism is asserted or needed.
