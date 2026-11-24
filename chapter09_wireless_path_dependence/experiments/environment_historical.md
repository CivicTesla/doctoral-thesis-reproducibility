# R0 environment audit

Candidate toolchain is `/tmp/b2-ns3-47/ns-3.47`, an existing local official
ns-3.47 source/build. `src/propagation/model/three-gpp-propagation-loss-model.cc`
was inspected read-only: `GetShadowing` caches per-link state, uses
\(\exp(-\Delta/d)\) for displacement correlation, and initializes independently
on `notFound || newCondition`. `ThreeGppUmaPropagationLossModel` returns 4 dB
and 37 m for LOS above 6 GHz. GCC is 11.4.0 and CMake is 3.22.1.

No B3′ code was compiled or run during this audit.
