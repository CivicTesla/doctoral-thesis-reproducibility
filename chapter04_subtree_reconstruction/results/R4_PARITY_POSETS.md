# R4 parity trade posets

Use directions [4]. The two sides of the minimum affine [3]-trade are

~~~
E = {empty} union C([4],2) union {[4]},
O = C([4],1) union C([4],3).
~~~

Both have eight rows. Add one common all-one vertex h, delete it, and root the
remaining forest at h.

## Odd side

The only strict inclusions among distinct labels in O are

~~~
triple superset singleton, when the singleton is contained in the triple.
~~~

Consequently every nontrivial component is a star rooted at one of the four
triples, with one to three singleton leaves; all other components are isolated
vertices. In particular there are no paths of length at least two in the rooted
forest.

## Even side

The available strict inclusions are

~~~
[4] superset 2-set superset empty,
[4] superset empty.
~~~

Let the nontrivial odd-star component sizes be a_1,...,a_r, where a_j is its
number of leaves. An even-side labelling exists exactly for

~~~
r = 0;
r = 1 with a_1 in {1,2,3};
r = 2 with (a_1,a_2) in {(1,1),(2,1),(3,1)},
~~~

up to order.

Necessity is immediate. A star with at least two children needs [4] at its
root; a one-child star not using [4] needs a 2-set above the unique empty
label. Thus at most two nontrivial stars occur, and two stars cannot both have
at least two leaves. For sufficiency, place [4] atop the larger star, put
distinct 2-sets on its children, and, if a second star is present, put a
remaining 2-set above empty. Put all unused labels on isolated vertices.

The odd labels can be assigned simultaneously: choose distinct singleton
leaves, and give a j-leaf star a triple containing those j singletons. For the
displayed cases this is immediate (for a three-leaf star the triple is forced;
the one remaining singleton serves the second one-leaf star).
