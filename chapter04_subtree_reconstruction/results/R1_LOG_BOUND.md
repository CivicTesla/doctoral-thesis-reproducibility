# R1 logarithmic universal bound

## Theorem

For every finite `n`-vertex host (in particular, every finite host tree),

```text
k(T) <= max(2, floor(log_2 n)+1).
```

Connectedness of the columns is not used in this generic fixed-host theorem.

## Proof

Let

```text
t = floor(log_2 n)+1,
```

so `2^t>n`.  Pad both incidence matrices to all `n` host vertices.  Equality
of `Sigma_<=t` supplies equal containment counts at every nonempty labelled
coordinate set of size at most `t`; the common padded row count supplies the
order-zero count.  By the reduction in `R1_TRADE_REDUCTION.md`, unequal row
multisets would give a nonempty signed/multiset `[t]`-trade of volume at most
`n`.  The multiset trade lemma gives volume at least `2^t`, a contradiction.

The padded row multisets therefore agree, which implies the C2 active-ground-
set isomorphism after deletion of zero rows.  The outer maximum only enforces
the definition's convention `k>=2`.

## Scope and sharpness

This is strictly sharper than R0's `2n` bound because fixed-host padding
restores the zeroth moment.  It is a universal identification guarantee, not
a claim that every host attains the bound.  The parity trades in
`R1_STAR_THEOREM.md` show that the logarithmic scale cannot be improved for
all fixed-host set systems, and is attained by star hosts.
