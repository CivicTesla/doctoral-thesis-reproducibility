# R0 execution lock

**Purpose.** `PRECOMMIT.md` already freezes the channel, geometry, mesh,
worlds, and three Gaussian completion arms.  This pre-execution document fixes
only the operational link task that its phrase “identical fixed communication
task” leaves unspecified.  It contains no result and makes no change to a
scientific parameter in `PRECOMMIT.md`.

## Scalar link task

This R0 witness deliberately uses the precommitted scalar
`ThreeGppUmaPropagationLossModel`, not a spectrum/PDU-level PHY simulation.
Accordingly, no Wi-Fi/NR MAC or packet scheduler is implied.  At every explicit
mesh evaluation, the native ns-3 model supplies deterministic fixed-LOS UMa
received power and `BridgeShadowingPathLoss` subtracts the one already
pre-generated bridge shadowing value.  This is an operational link-budget
service task, not an additional channel model.

Fixed execution constants are:

| Field | Locked value |
|---|---:|
| Transmit power | 30 dBm |
| Bandwidth | 100 MHz |
| Thermal-noise density | -174 dBm/Hz |
| Receiver noise figure | 7 dB |
| Receiver noise power | -87 dBm |
| Endpoint observation window | 1.0 s |
| Route speed for integrated accounting | 1 m/s |
| Route-service duration of a mesh interval | its incoming mesh displacement in seconds |
| Outage threshold (secondary only) | SNR < 0 dB |
| Tie rule for rate-based best route | deterministic A-before-B |

For received power \(P\) in dBm, define

\[
\gamma(P)=10^{(P-(-87))/10},\qquad
R(P)=10^8\log_2(1+\gamma(P))\;\text{bit/s}.
\]

Traffic is permanently backlogged, so delivered bits over duration \(h\) are
\(hR(P)\).  This formula is the fixed operational service proxy; it introduces
no dB cost, learned model, packet-loss tuning, or action-specific parameter.

## Registered outputs and analysis units

For each `(world_id, completion_arm, branch)`, record all mesh powers/rates and
these branch outcomes:

* **Primary terminal outcomes:** endpoint received power, endpoint achieved
  rate, and endpoint delivered bits over exactly 1.0 s.
* **Primary paired intervention outcomes:** terminal A-minus-B delivered-bit
  contrast, its absolute value, and the rate-based realized best route under
  the locked tie rule.
* **Secondary integrated outcomes:** incoming-segment-duration-weighted
  route delivered bits, route-average achieved rate, and fraction of route
  time with SNR < 0 dB.

Every explicit mesh evaluation is retained in `trajectory_results.csv` with
position, arclength, incoming duration, injected shadowing, received power,
and achieved rate.  This is raw execution provenance, not an additional
outcome or analysis mesh.

The three pre-registered completion comparisons are PLUS–ZERO, PLUS–MINUS,
and ZERO–MINUS.  For branch metrics and signed/absolute paired contrasts, use
the same world ID across arms and report mean paired differences with a
two-sided normal 95% interval \(\bar d\pm1.96\,s_d/\sqrt{2000}\).  For binary
selection-disagreement proportions, report the exact observed proportion and
the Wilson 95% interval.  No unlisted outcome, window, threshold, or
transformation can replace a primary metric.

## Cross-arm base-randomness alignment

For each registered world, deterministic substreams generate shared \(S_0\),
an innovation \(E_a\), an independent innovation \(Z\), and branch-specific
bridge normals.  The frozen endpoint completions are realized as

\[
E_b^{+}=E_a,\qquad E_b^0=Z,\qquad E_b^{-}=-E_a.
\]

Thus the three arms have \(\kappa=+1,0,-1\) exactly and share a lawful world
identifier only for paired reporting; every branch’s unconditional full-path
Gaussian forward law remains identical across arms.  The terminal best-route
comparison is never asserted to be a population mean effect.

## Registered mechanism gate

The code must first pass all deterministic checks: 2,000 rows for each
`(arm, branch)`, finite values, the frozen mesh lengths and positions, native
shadowing disabled, exact target covariance matrices shared by all arms, and
conditional-Gaussian reconstruction error at most \(10^{-10}\).  Any failure
is `INVALID_MECHANISM_WITNESS`.

The registered empirical checks are diagnostic tests of the exact construction,
fixed before data generation.  For every arm/branch, endpoint mean zero and
endpoint variance 16 are assessed using two-sided 99% normal and chi-square
reference intervals respectively with \(n=2000\).  The endpoint covariances
\(16e^{-10/37}\), \(16e^{-60/37}\) and first-full-1-m-step covariance
\(16e^{-1/37}\) are assessed using their two-sided 99% Gaussian-product
normal-approximation intervals,

\[
\operatorname{SE}(\widehat{\operatorname{Cov}}(X,Y))
=\sqrt{(\sigma_X^2\sigma_Y^2+c^2)/2000}.
\]

The complete empirical covariance matrix and its maximum absolute/relative
error remain reported, but are not retrospectively thresholded.  An empirical
reference-band failure is an implementation/mathematical audit failure and
prevents operational interpretation.

## Frozen operational status rule

After (and only after) `MECHANISM_PASS`, `WITNESS_PASS` requires at least one
of these **primary terminal** completion-dependent consequences, evaluated only
for the three locked arm pairs:

1. the two-sided normal 95% paired interval for the mean change in absolute
   A-minus-B terminal delivered-bit contrast excludes zero **and** the absolute
   paired mean change is at least 1% of the mean of the two arm-specific
   absolute contrasts; or
2. the rate-based realized-best-route selection differs between the two arms
   in at least 1% of registered worlds and the Wilson 95% lower bound for that
   proportion is above zero.

Otherwise a mechanism-passing run is `WITNESS_WEAK`.  Secondary integrated
metrics cannot upgrade this status.  This is an execution interpretation rule,
not a claim about population marginal mean action values.

## Execution prohibitions

This lock does not modify and must not be used to tune: route geometry,
endpoint separation, carrier frequency, LOS condition, shadowing parameters,
evaluation mesh, completion class, master seed, registered world IDs, or the
canonical zero-shadowing diagnostic.  No other simulation, retry, or sample
size extension is authorized except an identically-input deterministic rerun
of a logged engineering crash.
