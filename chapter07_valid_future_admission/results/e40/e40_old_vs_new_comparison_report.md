# E40 old-vs-new qualitative comparison

- VFA improves over AoIOnly: PASS
- VFA improves over RGSA: PASS
- VFA improves over Best FQ: PASS
- VFA remains route-cost competitive with Route-ref.: PASS
- VFA/Crit-TopK reduce same-agent concentration: PASS
- Crit-TopK reduces W10 starvation under fixed scarcity: PASS
- Runtime story remains valid: PASS

## Route-cost deltas

```text
                        index  old_route_cost  e40_route_cost  delta  delta_percent
                          aoi        0.503507        0.503507    0.0            0.0
                      best_fq        0.503135        0.503135    0.0            0.0
                    crit_topk             NaN        0.495585    NaN            NaN
        fixed_quota_aoi_heavy        0.505647             NaN    NaN            NaN
         fixed_quota_balanced        0.504475             NaN    NaN            NaN
   fixed_quota_coverage_heavy        0.504475             NaN    NaN            NaN
   fixed_quota_critical_heavy        0.508802             NaN    NaN            NaN
                pd_no_rollout             NaN        0.495039    NaN            NaN
                       random        0.507169        0.507169    0.0            0.0
                         rgsa        0.500941        0.500941    0.0            0.0
             rgsa_prefix_risk        0.497345             NaN    NaN            NaN
rgsa_prefix_risk_diverse_soft        0.493148             NaN    NaN            NaN
                    route_ref        0.495248        0.495248    0.0            0.0
                          vfa             NaN        0.495070    NaN            NaN
```
