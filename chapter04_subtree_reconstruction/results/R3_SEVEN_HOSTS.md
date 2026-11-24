# R3 canonical nine-vertex hosts

The R2 hosts are named in this fixed order. Edge lists use vertices `0,...,8`.

| host | graph description | degree sequence | edge list | graph6 |
|---|---|---|---|---|
| `Q1` | `K1,8` | `8,1^8` | `01,12,13,14,15,16,17,18` | `HiPAA@?` |
| `Q2` | `K1,7` with one pendant edge subdivided | `7,2,1^7` | `01,02,13,14,15,16,17,18` | `HqPAA@?` |
| `Q3` | `K1,6` with two distinct pendant edges subdivided | `6,2^2,1^6` | `01,02,13,15,16,17,18,34` | `HqDAA@?` |
| `Q4` | degree-six centre with five leaf branches and one cherry branch | `6,3,1^7` | `01,02,03,14,15,16,17,18` | `HsPAA@?` |
| `Q5` | double star `DS(3,4)` | `5,4,1^7` | `01,02,03,04,15,16,17,18` | `Hs&#96;AA@?` |
| `Q6` | `DS(3,3)` with one pendant edge subdivided | `4,4,2,1^6` | `01,02,03,04,15,17,18,56` | `Hs&#96;?I@?` |
| `Q7` | `DS(2,4)` with one pendant edge subdivided | `5,3,2,1^6` | `01,02,03,14,16,17,18,45` | `HsOIA@?` |

Here `DS(a,b)` is the tree made from a centre edge by attaching `a` leaves
to one endpoint and `b` to the other. Superscript `+` in informal R2 notes
meant subdivision of exactly one pendant edge; the descriptions above remove
that ambiguity.

The graph6 strings are identifiers for these labelled edge-list representatives,
not an additional isomorphism convention.
