# External validation closure (Simu5G)

## Formal freeze

`SIMU5G_EXTERNAL_VALIDATION_INCOMPLETE`

## Why

The radio bottleneck itself is real and reproducible: a saturated engineering
probe on the pinned, official Simu5G toolchain established `C_UL_ref =
887.301 Mbit/s`, and the pre-registered load sweep (0.50/0.90/1.30 x
reference) activated the target uplink path exactly as designed
(`P4_simu5g_external_validation/radio_operating_point_audit/`, decision
`P4B_R1_RADIO_ACTION_NOT_RECOVERABLE`). But the RADIO recovery action does
not recover the foreground path under strict paired evaluation: across 13
consistently adverse independent seeds, RADIO action made mean radio-stage
delay, end-to-end delay, and deadline misses *worse* than matched NO_ACTION,
even after removing the pre-registered virtual background interferer and its
~2.08 GB queued backlog at decision time (no MAC/PHY or other Simu5G core
code was changed to produce this). Without a working paired action
counterfactual, there is no legitimate external RADIO-vs-NO_ACTION action
contrast to report as evidence -- only a confirmed bottleneck with a null
recovery result under this specific virtual-background construction.

## What survives vs. what does not

| Stage | Status |
|---|---|
| P4A: Simu5G feasibility (toolchain builds, H3/H4 map to real radio/MEC resources, paired control is possible) | `P4A_SIMU5G_FEASIBILITY_SURVIVES` -- feasibility evidence only, explicitly not a matrix/model/replication claim |
| P4B: operating point search (original config) | `P4B_OPERATING_POINT_INVALID` |
| P4B-R1: radio operating point diagnosis + recovery audit | `P4B_R1_RADIO_ACTION_NOT_RECOVERABLE` -- bottleneck established, action not recoverable |
| P4B: external scientific replication of B1's H3/H4 action-separation or risk-direction claims | **not established** |

**P4A feasibility succeeded. P4B scientific external replication was not
established.** These are two different claims and must not be merged in the
paper: P4A says the toolchain and mechanism mapping are usable; P4B says
whether the actual action-recovery science replicates externally, and that
did not survive.

## What this freeze forbids for the remainder of B1 (including F0)

- Fixing the RADIO action in Simu5G.
- Adjusting the recovery window.
- Swapping background-traffic implementation (e.g., foreground-contention
  UEs instead of virtual background).
- Installing ns-3.
- Installing 5G-LENA.
- Using any other external simulator.
- Wrapping the P4A smoke test as a paper external-validation claim.

F0 (this directory) validates B1's H1/H2/H3/H4 mechanism pairs at final
scale using only the two already-frozen synthetic mechanism simulators
(`cf0_simulator.py` for H1/H2 with its upstream F2 instrumented-replay
telemetry layer for T0/T4/R4; `P2_radio_compute_pair_pilot/src/pilot.py` for
H3/H4). No Simu5G, ns-3, or other external simulator run is part of F0.
