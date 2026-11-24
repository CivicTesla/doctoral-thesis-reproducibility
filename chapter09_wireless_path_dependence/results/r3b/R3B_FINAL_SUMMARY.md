# R3b final identification-hierarchy summary

## Status and provenance

* Branch: `b2-r3b-identification-hierarchy`.
* Protocol commit: `c0ca73b9891fd99ab6a6ee6e7fb4ec8c362782fc`.
* Theory and deterministic calculation commit:
  `4441b6199ba81c4f8a64a99948a1c2e88de9859d`.
* Audit commit: `83e93cc6486c8f10ddf7a08afca16371c0d29e02`.
* Final status: **IDENTIFICATION_HIERARCHY_PASS_WITH_QUALIFICATIONS**.
* Sign-convention verdict: **SIGN_CONVENTION_PASS**.

No ns-3 run, stochastic world generation, replication, R0/R1/R1b/R2 change,
or manuscript edit occurred.

## Selector and four information levels

The frozen implementation selects A iff
\(\Delta+S_T^b-S_T^a\ge0\).  R3/R3b abstract notation uses \(V_j=-S_T^j\)
only to write the same event as \(\Delta+V_a-V_b\ge0\).

* C0 preserves unconditional endpoint marginals only.
* C1 preserves one shared factual prefix and both complete branch laws; it is
  the R3 all-admissible-completion class.
* C2 is the declared fixed-\(\kappa\), conditionally jointly Gaussian
  residual subclass of C1.
* C3 is one named completion: PLUS, ZERO, or MINUS.

The hierarchy is correctly read as contraction of feasible terminal
decision-probability sets.  C1 induces endpoint couplings allowed by C0, but
C0 is not itself a complete-process object.

## Level 0 and attainment language

For \(\Delta\ge0\), C0 has the sharp Makarov infimum/supremum
\[
 [\,2\Phi(\Delta/(2\sigma))-1,\ 1\,].
\]
At zero margin this is \([0,1]\).  The lower endpoint remains a sharp
infimum, not an unqualified universally attained minimum.  Its robust margin
is finite: \(2\sigma\Phi^{-1}((1+q)/2)\).

C1 retains the R3 standard-Borel, bridge-kernel, and measurable
epsilon-selection qualification.  Its values remain sharp process-tight
infimum/supremum limits, not a universal exact-attainment theorem.

For C2,
\[
 \operatorname{Var}(S_T^b-S_T^a)
 =\sigma^2[(R_b-R_a)^2+q_a^2+q_b^2-2\kappa q_aq_b],
\]
and \(P_\kappa(\mathrm{select\ A})=\Phi(\Delta/s_\kappa)\).  MINUS
(\(\kappa=-1\)) exactly attains the C2 lower endpoint; PLUS
(\(\kappa=+1\)) exactly attains its upper endpoint.  Thus
**GAUSSIAN_SUBCLASS_PASS** for the declared fixed-\(\kappa\) subclass.

## Frozen 10/60 m hierarchy

| eta | C0 interval | C1 interval | C2 interval | PLUS / ZERO / MINUS |
|---:|---:|---:|---:|---:|
| 0.0 | [0.000000, 1.000000] | [0.148272, 0.851728] | [0.500000, 0.500000] | 0.500000 / 0.500000 / 0.500000 |
| 0.5 | [0.130446, 1.000000] | [0.241057, 0.920377] | [0.575634, 0.691462] | 0.691462 / 0.599489 / 0.575634 |
| 1.0 | [0.257428, 1.000000] | [0.350058, 0.963467] | [0.648573, 0.841345] | 0.841345 / 0.692887 / 0.648573 |
| 2.0 | [0.488754, 1.000000] | [0.573973, 0.995539] | [0.777251, 0.977250] | 0.977250 / 0.843298 / 0.777251 |
| 3.0 | [0.675547, 1.000000] | [0.755771, 0.999759] | [0.873774, 0.998650] | 0.998650 / 0.934753 / 0.873774 |

All C2 and C3 values lie inside C1, and all C1 values lie inside C0 on the
frozen grid.  At \(\eta=0\), retaining the shared prefix reduces width from
1 to 0.703455: an absolute identification-width reduction of 0.296545
(29.6545%).  C2 removes the remaining C1 width at this zero-margin point.

## Robust-margin hierarchy (dB)

| q | PLUS | ZERO | MINUS = C2 | C1 all completion | C0 endpoint marginal |
|---:|---:|---:|---:|---:|---:|
| 0.90 | 3.367399 | 6.680663 | 8.827409 | 11.236922 | 13.158829 |
| 0.95 | 4.322010 | 8.574538 | 11.329856 | 13.423428 | 15.679712 |
| 0.99 | 6.112701 | 12.127133 | 16.024032 | 17.684697 | 20.606634 |

Normalized values are retained in `robust_margin_hierarchy.csv`.  These are
class- and branch-pair-conditional certificates, not physical path-loss
safety margins.

## Boundaries and manuscript-use facts

The position-indexed field changes the frozen bent-trajectory forward laws and
is outside C0/C1/C2/C3.  Per-world cross-model disagreement would require a
further declared cross-model joint law.  Terminal ranking and
trajectory-integrated utility remain distinct estimands.  Recommended notation
and factual scope statements are recorded in the corresponding R3b audits.

The wireless literature audit verifies wireless what-if/counterfactual
motivation, including Ruah--Simeone--Al-Hashimi (JSAC 2023), Hou et al. (TCCN
2025), and an ns-3-based what-if digital-twin paper (IWCMC 2024).  It finds
**NO DIRECT PRIOR-MISUSE EVIDENCE FOUND**: these sources should motivate the
application, not substantiate an accusation that prior work used an undeclared
path-dependent cross-branch coupling.
