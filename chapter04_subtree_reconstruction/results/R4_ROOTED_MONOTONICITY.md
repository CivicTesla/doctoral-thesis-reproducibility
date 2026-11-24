# R4 rooted monotonicity lemma

Let C be a nonempty connected vertex set of a tree T. Give every vertex of C
the row [m]. Each component of T-C has a unique edge to C; root it at the
endpoint adjacent to C, so its parent is closer to C.

## Lemma (rooted monotonicity)

For rows R_v subseteq [m] on T-C, put

~~~
S_i = C union {v outside C : i in R_v}.
~~~

Then all S_i are connected if and only if, on every oriented edge p -> q of
T-C,

~~~
R_q subseteq R_p.
~~~

### Proof

If S_i is connected and i is in R_q, the unique path from q to C passes
through p. Since q and every vertex of C lie in S_i, so does p; hence i is in
R_p. This holds for every i, giving R_q subseteq R_p.

Conversely, take v in S_i-C. Monotonicity puts i in the row of every vertex
on the path from v to the root of its component. The next edge joins that root
to C, whose every row contains i. Thus the path from v to C lies in S_i;
because C is connected, S_i is connected.

The boundary edge needs no extra condition: its endpoint in C already has the
all-one row. This proves the stated iff, including the converse.

## Poset formulation

The rooted forest has its ancestor partial order. A realization is an
assignment of a prescribed multiset of Boolean-lattice labels to its vertices
such that labels weakly decrease by inclusion away from the root. Equivalently
it is an order-preserving map after reversing the ancestor order. It is
deliberately not called an order embedding: labels may repeat, unrelated
vertices need not be comparable, and equality is allowed. Empty labels are
allowed. Multiple common all-one rows are simply vertices of the connected core
C.

This is a local term, **rooted Boolean-lattice realization**, used below to
avoid suggesting a new standard notion of poset embedding.
