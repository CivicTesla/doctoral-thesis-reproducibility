# R3 final theory-and-deterministic-analysis summary

## Status and provenance

* Branch: `b2-r3-sharp-decision-band`.
* Protocol freeze: `6a9347eaf5ec5e0617609c6628fc2e3da57ae5ea`.
* Theory/numerical-band commit: `c9a2e32bf052e2377c75c6e77434239be680b3b2`.
* Proof and positioning audit: `83dd06c8b4daead626bf7aa46891a28d8b0f0527`.
* Final status: **DECISION_BAND_PASS_WITH_QUALIFICATIONS**.

No ns-3 executable was run, no stochastic world was generated, and no R0,
R1, R1b, R2, or manuscript artifact was changed.

## General result

For a common prefix `H0`, conditional terminal CDFs \(F_{a,h},F_{b,h}\), and
\(A_\Delta=\{\Delta+V_a-V_b\ge0\}\), the exact sharp conditional bounds are
the Makarov difference bounds.  In the continuous form used below,
\[
 \ell_h(\Delta)=\sup_x[F_{b,h}(x+\Delta)-F_{a,h}(x)]_+,
\quad
 u_h(\Delta)=1-\sup_x[F_{a,h}(x)-F_{b,h}(x+\Delta)]_+.
\]
The unconditional sharp infimum and supremum are respectively the expectations
of these two functions over the *same* factual prefix.  For atoms, the exact
left-limit CDF formulation in `R3_GENERAL_THEOREM.md` is required.

Under standard-Borel conditional laws, measurable epsilon-optimal endpoint
coupling kernels, and the established bridge kernels, these are process-tight
sharp limits: a sequence of full, branch-law-preserving admissible process
completions approaches each endpoint.  R3 deliberately does not assert that
every general conditional model has one globally measurable exact endpoint
attainer.

## Gaussian 10/60 m specialization

With \(\sigma=4\) dB, \(d=37\) m, \(L_a=10\) m, \(L_b=60\) m, and
\(g_{\min}\sigma=2.6275955203\) dB, deterministic Gaussian optimization and
quadrature give:

| \(\eta\) | \(\Delta\) (dB) | lower \(P(\mathrm{select\ A})\) | upper \(P(\mathrm{select\ A})\) | width | PLUS | ZERO | MINUS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.000000 | 0.148272 | 0.851728 | 0.703455 | 0.500000 | 0.500000 | 0.500000 |
| 0.5 | 1.313798 | 0.241057 | 0.920377 | 0.679321 | 0.691462 | 0.599489 | 0.575634 |
| 1.0 | 2.627596 | 0.350058 | 0.963467 | 0.613408 | 0.841345 | 0.692887 | 0.648573 |
| 2.0 | 5.255191 | 0.573973 | 0.995539 | 0.421566 | 0.977250 | 0.843298 | 0.777251 |
| 3.0 | 7.882787 | 0.755771 | 0.999759 | 0.243989 | 0.998650 | 0.934753 | 0.873774 |

Every PLUS/ZERO/MINUS value lies in the sharp band on the saved dense grid.
They are named residual-coupling conventions, not the full identified set.
`decision_band.pdf` plots the band and those three curves only; it does not
put the location-indexed FIELD model inside the band.

## Completion-robust certificate and width diagnostic

The lower bound is nondecreasing in \(\Delta\), because the decision event is
increasing for every fixed completion.  For this specialization, it tends to
one.  The numerical robust certificates are:

| target \(q\) | robust margin (dB) | normalized margin \(\eta\) |
|---:|---:|---:|
| 0.90 | 11.236922 | 4.276504 |
| 0.95 | 13.423428 | 5.108635 |
| 0.99 | 17.684697 | 6.730372 |

The identification width is 0.703455 at zero margin and tends to zero as the
margin tends to infinity.  For the frozen pair it falls below 0.20, 0.10, and
0.05 at \(\eta=3.311977\), \(4.276483\), and \(5.108635\), respectively.
These values are pair-specific diagnostics, not universal eta thresholds.

## Boundary with existing theory and field semantics

R3 bounds a ranking/difference event.  Existing Theorem 3 bounds service
indicator disagreement; the two fixed-marginal extremal problems are distinct.
The position-indexed exponential field changes the bent-trajectory forward
laws and remains a separate model reference, not an admissible completion in
the trajectory-law band.

## Positioning and limitation

The extremal CDF bound is classical Makarov theory, with the
Frank--Nelsen--Schweizer copula treatment as a directly relevant source.
The B2-specific content is the conditional shared-prefix specialization,
complete-process preservation and bridge lifting, wireless Gaussian
specialization, and completion-robust certificate.  The general theorem must
retain its standard-Borel/regular-kernel/measurable-epsilon-selection
qualification; no stronger universal exact-attainment claim is made.
