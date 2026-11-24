# Final B3′ R0 provenance and interpretation audit

**Audit mode:** read-only review of committed history and frozen R0 artifacts.
No simulation, analysis rerun, parameter change, or new metric was performed.

## Audit status

`PAPER_GATE_READY`

This is a provenance/interpretation gate only.  It is not a paper-go decision.

## Temporal record

| Role | Commit | Git timestamp (+0800) |
|---|---|---|
| R0 precommit | `51be4469e8c9fd6d5ebc18c6eee1e464b33bb513` | 2026-08-28 18:02:59 |
| implementation self-check | `57fb69178eee8812c1d1c8964d02d3933ffc1d64` | 2026-08-28 18:32:30 |
| execution lock | `f200c0cb8a17bfc0600f7381f423c9eeefada38d` | 2026-08-28 19:24:34 |
| frozen executor/analysis | `161d2f2004239db36cb97735cea940bfa04cf103` | 2026-08-28 19:31:04 |
| registered result | `b85281845cec43dbe57e43752fb7678b402c1828` | 2026-08-28 19:32:54 |

The committed `RUNLOG.md` records `161d2f2` as the start implementation
commit (lines 3–8) and records the exact registered command, IDs, arms, and
post-gate analysis (lines 10–26).  Thus every status-rule concept was frozen
in `f200c0c`, and its executable definition was frozen in `161d2f2`, before
the committed registered result.  Git history cannot independently attest a
wall-clock process start; this conclusion is the strongest committed-provenance
statement available and is consistent with the frozen execution log.

`PRECOMMIT.md` is byte-identical to its precommit version in the registered
result ancestry; its SHA-256 remains
`2ca23c64a3187f8a4540d51cd465dc5c255d916f443bdecc1a55755efdbe78a4`.

## Metric provenance

“First semantic freeze” identifies the first committed specification of the
metric.  “Exact executable definition” identifies the first committed source
line that constructs it.  Both precede the registered result commit.

| Metric | Class | Mathematical/operational definition | First semantic freeze | Exact executable definition | Pre-execution? |
|---|---|---|---|---|---|
| terminal signed A−B delivered bits | primary paired | \(B_A-B_B\), where \(B_j=1\,\mathrm{s}\,R(P_j)\) | `f200c0c`, `EXECUTION_LOCK.md:41–54` | `161d2f2`, `ns3_witness/scripts/analyze_registered.py:69` | yes |
| terminal absolute \(|A-B|\) delivered bits | primary paired; **status metric** | \(|B_A-B_B|\) | `f200c0c`, `EXECUTION_LOCK.md:52–54,110–117` | `161d2f2`, `…/analyze_registered.py:70,111–117` | yes |
| realized route selection | primary paired | select A iff \(R_A\ge R_B\); equality goes to A | `f200c0c`, `EXECUTION_LOCK.md:32,52–54` | `161d2f2`, `…/analyze_registered.py:71` | yes |
| realized route-selection disagreement | primary paired; **status metric** | \(1\{\mathrm{select}_{\kappa_1}\ne\mathrm{select}_{\kappa_2}\}\) | `f200c0c`, `EXECUTION_LOCK.md:63–64,118–120` | `161d2f2`, `…/analyze_registered.py:105–120` | yes |
| route-integrated delivered bits | secondary | \(\sum_{i\ge1}\Delta t_iR(P_i)\), \(\Delta t_i\) is incoming mesh displacement at 1 m/s | `f200c0c`, `EXECUTION_LOCK.md:29–30,55–57` | `161d2f2`, `ns3_witness/src/b3p_r0_witness.cc:244–248` | yes |
| route-average rate | secondary | route-integrated bits divided by route time | `f200c0c`, `EXECUTION_LOCK.md:55–57` | `161d2f2`, `…/b3p_r0_witness.cc:265` | yes |
| route outage fraction | secondary | route-time fraction with \(\mathrm{SNR}<0\) dB | `f200c0c`, `EXECUTION_LOCK.md:31,55–57` | `161d2f2`, `…/b3p_r0_witness.cc:249–252,266` | yes |
| canonical \(Y_j\) | diagnostic only | \(Y_j=1\{S_T^j\le0\}\); report \(P[Y_A\ne Y_B]\) | `51be446`, `PRECOMMIT.md:47–50` | `161d2f2`, `…/b3p_r0_witness.cc:359,390–393`; analysis `…/analyze_registered.py:41–50` | yes |

The rate/service definition underlying delivered bits was frozen in
`EXECUTION_LOCK.md:34–43` and implemented at
`b3p_r0_witness.cc:205–210,254–258` before the result.

## Status-decision audit

The frozen rule is `EXECUTION_LOCK.md:108–124`: after `MECHANISM_PASS`, either
(1) a locked terminal absolute-contrast paired 95% interval excluding zero
with at least 1% relative change, or (2) a locked route-selection disagreement
of at least 1% with Wilson lower bound above zero, is sufficient.  Secondary
integrated metrics cannot upgrade the status.

The direct positive system-level witness is the **primary terminal absolute
delivered-bit contrast**.  It meets condition (1) for all locked arm pairs;
for example PLUS−ZERO is −63,021,941 bits, 95% CI
[−68,038,921, −58,004,961], relative effect 0.6331.  Independently, the
primary realized-route-selection disagreement meets condition (2): PLUS−ZERO
is 0.449, Wilson 95% CI [0.4273, 0.4709].

Therefore `WITNESS_PASS` follows strictly from the frozen rule.  Removing all
post-execution derived or non-preregistered metrics leaves those two frozen
primary metrics and the mechanism gate intact; the status still follows.  No
reported metric used for the status decision is post hoc.  Endpoint SD and
canonical disagreement are diagnostic; signed and integrated quantities are
reported boundaries, not status substitutes.

## Marginal-law and signed-versus-absolute interpretation

The frozen cross-arm construction states that every branch’s unconditional
full-path Gaussian forward law is identical across arms
(`EXECUTION_LOCK.md:67–80`).  The self-check records identical target
covariance matrices across all arms in
`selfcheck/arm_marginal_validation.csv`, committed in `57fb691`.
The registered mechanism gate passed, with six arm/branch rows and no failures
in `mechanism_gate.json`.

No finite-sample raw A or B arm mean is used as evidence of a completion
effect.  The signed terminal A−B completion contrasts all have 95% intervals
crossing zero.  This is consistent with preserved action marginals and is not
evidence of completion-induced population mean action-value bias.

By contrast, \(|B_A-B_B|\) and cross-arm realized-selection disagreement are
dependence-sensitive joint/pathwise quantities.  They support the limited
claim: changing an admissible cross-branch completion can change realized
counterfactual route comparisons while preserving branchwise forward laws.

`analysis_summary.md` correctly says that a coupling contrast is not a
population marginal mean action-value bias (line 5).  No frozen result wording
claims average wireless-performance improvement.  Future paper text must avoid
such wording.

## Integrated-outcome boundary

All registered route-integrated signed A−B cross-arm contrasts have 95%
intervals crossing zero: PLUS−ZERO −18,353,138 bits
[−213,359,998, 176,653,722]; PLUS−MINUS −44,738,771 bits
[−324,885,216, 235,407,674]; ZERO−MINUS −26,385,633 bits
[−225,318,688, 172,547,423].  Route-average-rate and route-outage cross-arm
contrasts likewise do not supply the status decision.

The result is therefore limited to terminal counterfactual intervention
ranking/joint consequences under the locked link task, not large whole-route
aggregate performance changes.  This negative boundary is retained.

## Reproducibility and execution audit

- `registered_world_manifest.csv` contains exactly world IDs 1–2000, each once,
  `master_seed=2026082801`, and exactly PLUS/ZERO/MINUS in every row.
- Result row counts are 2,000 manifest, 6,000 endpoint, 12,000
  branch-operational, and 438,000 trajectory rows.  The latter equals
  \(2000\times3\times(11+62)\) explicit mesh evaluations.
- `RUNLOG.md:19–23` records successful completion, no crashes/retries, and
  `MECHANISM_PASS` before analysis.  It records no rescue tuning and no post-
  analysis simulation (`RUNLOG.md:15,22,26–27`).
- Native shadowing is disabled in the executor
  (`b3p_r0_witness.cc:348–352`) and forcibly disabled in the wrapper
  (`bridge_shadowing_path_loss.h:38–40`).  The wrapper subtracts/increments
  exactly once per evaluation (`…:47–52`); executor consumption equality is
  enforced at `b3p_r0_witness.cc:261–264`.

## Wording to avoid

Do not state that a completion arm improves average wireless performance,
changes a population marginal action value, exposes an ns-3 bug, establishes a
position-indexed field as physically correct, or demonstrates large whole-route
aggregate effects.  Do not present the canonical \(Y\) diagnostic alone as the
system-level witness.  The defensible claim is the narrower completed-process,
joint/pathwise counterfactual consequence stated above.
