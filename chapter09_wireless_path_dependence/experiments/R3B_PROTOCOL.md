# R3b identification-hierarchy protocol freeze

## Scope

R3b is theory and deterministic analytic computation only.  It must not run
ns-3, generate worlds, alter R0/R1/R1b/R2, tune a parameter, or edit the
manuscript.

## Frozen selector and numerical pair

The inherited terminal convention is
\[
 P_A=P_0-S_A+\Delta/2,\qquad P_B=P_0-S_B-\Delta/2.
\]
Because the terminal rate map is strictly increasing, select A iff
\(\Delta+S_B-S_A\ge0\).  Equivalent abstract notation is
\(V_j=-S_j\), so that \(\Delta+V_a-V_b\ge0\).  R3b may not reverse this
sign convention.

The frozen pair is \(\sigma=4\) dB, \(d=37\) m, \(L_a=10\) m, and
\(L_b=60\) m.  Set \(R_j=e^{-L_j/d}\), \(q_j=\sqrt{1-R_j^2}\), and use
\(g_{\min}\sigma=2.6275955203\) dB.  The preregistered grid is
\(\eta\in\{0,0.5,1,2,3\}\), \(\Delta=\eta g_{\min}\sigma\).

## Information classes

* **C0:** all endpoint couplings that retain only the two unconditional
  terminal marginals.
* **C1:** the R3 class: a common factual prefix and both complete branch laws,
  with all process-law-preserving admissible completions.
* **C2:** C1 restricted to conditional jointly Gaussian residual endpoints
  \((E_a,E_b)\), each standard normal, with a fixed
  \(\kappa=\operatorname{Corr}(E_a,E_b)\in[-1,1]\), then lifted through
  the established bridge kernels.
* **C3:** a declared member of C2: PLUS (\(\kappa=1\)), ZERO
  (\(\kappa=0\)), or MINUS (\(\kappa=-1\)).

R3b will compare feasible decision-probability sets; it will not make a naive
literal set-containment claim between C0 endpoint objects and C1 complete
process objects.

## Frozen outputs

R3b will derive C0, reuse C1, prove/qualify C2, tabulate the identification
and robust-margin hierarchies, produce one deterministic figure, and audit
signs, topology/attainment, notation, scope, and literature motivation.  The
requested robust levels are \(q\in\{0.90,0.95,0.99\}\).
