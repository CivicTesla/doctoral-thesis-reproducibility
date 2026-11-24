# R3 sharp decision-band protocol

Theory and deterministic analysis only: no ns-3, worlds, parameter tuning, or
manuscript edits. R0/R1/R1b/R2 remain frozen.

The frozen convention is retained: `P_A=P0-S_A+Delta/2` and
`P_B=P0-S_B-Delta/2`, so the strictly increasing terminal rate selector picks
A exactly when `Delta+S_B-S_A >= 0`. The requested abstract notation is used
with `V_a=-S_A`, `V_b=-S_B`, hence `Delta+V_a-V_b>=0`; no scientific sign is
flipped.

Question: under one shared factual prefix and complete conditional branch
process laws, determine the sharp all-admissible-completion range of this
terminal ranking probability, audit process-tight lifting, and compute the
frozen 10/60 m deterministic band and robust certificate.
