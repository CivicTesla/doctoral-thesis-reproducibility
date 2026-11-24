# E40 final claim check

| # | Claim | Status | Evidence |
|---:|---|---|---|
| 1 | VFA improves over AoIOnly in main | PASS | `{'comparison': 'VFA - AoIOnly', 'mean_delta': -0.00843769508628027, 'ci95_low': -0.011731458225904977, 'ci95_high': -0.005113673130658032, 'wilcoxon_p': 1.5430199145369686e-06, 'n': 420, 'negative_favors_vfa': True, 'holm_p': 4.629059743610906e-06}` |
| 2 | VFA improves over RGSA in main | PASS | `{'comparison': 'VFA - RGSA', 'mean_delta': -0.005871464709547306, 'ci95_low': -0.009215926524138322, 'ci95_high': -0.0025322865112519196, 'wilcoxon_p': 0.0010285604736375953, 'n': 420, 'negative_favors_vfa': True, 'holm_p': 0.0010285604736375953}` |
| 3 | VFA improves over Best FQ in main | PASS | `{'comparison': 'VFA - Best FQ', 'mean_delta': -0.008065028964769513, 'ci95_low': -0.011256792086230831, 'ci95_high': -0.004886322464870804, 'wilcoxon_p': 3.592943517935285e-06, 'n': 420, 'negative_favors_vfa': True, 'holm_p': 7.18588703587057e-06}` |
| 4 | VFA improves over Random in main | PASS | `{'comparison': 'VFA - Random', 'mean_delta': -0.012099058545682405, 'ci95_low': -0.015336002359922298, 'ci95_high': -0.00889909832513167, 'wilcoxon_p': 4.690028094450923e-11, 'n': 420, 'negative_favors_vfa': True, 'holm_p': 1.8760112377803692e-10}` |
| 5 | VFA competitive with Route-ref. | PASS | `{'comparison': 'VFA - Route-ref.', 'mean_delta': -0.00017819032886671893, 'ci95_low': -0.0032413351554586613, 'ci95_high': 0.002902670998486485, 'wilcoxon_p': 0.6970664582931444, 'n': 420, 'negative_favors_vfa': True, 'holm_p': nan}` |
| 6 | VFA reduces concentration vs Route-ref. | PASS | `{'vfa': 0.2939833333333333, 'route_ref': 0.7603642857142857}` |
| 7 | PD/diversity ablation supports structural gain | PASS | `{'prefix_only': {'route_cost': 0.4979295997931134, 'same_agent_concentration': 0.5638761904761905, 'rollout': 'Yes', 'prefix': 'Yes', 'diversity': 'No', 'episodes': 420}, 'pd': {'route_cost': 0.49503896610992876, 'same_agent_concentration': 0.2863928571428571, 'rollout': 'No', 'prefix': 'Yes', 'diversity': 'Yes', 'episodes': 420}}` |
| 8 | Rollout remains optional/modest | PASS | `{'pd': 0.49503896610992876, 'vfa': 0.4950698014863368}` |
| 9 | Budget sweep favors VFA where claimed | PASS | `{}` |
| 10 | Channel stress supports VFA over AoI/RGSA | PASS | `{}` |
| 11 | Online learners do not close gap | PASS | `{'aoi': 0.008437695086280173, 'best_fq': 0.008065028964769516, 'exp3': 0.011049797724494825, 'linucb': 0.010295526976883351, 'rgsa': 0.005871464709547303, 'route_ref': 0.00017819032886667863, 'ucb': 0.009790504399344091, 'vfa': 0.0}` |
| 12 | Runtime story remains valid | PASS | `{}` |
| 13 | Large-fleet allocation story remains valid | PASS | `{}` |
| 14 | Fixed-scarcity Crit-TopK deployment story remains valid | PASS | `{'group': 'pooled', 'N': nan, 'B': nan, 'comparison': 'vfa-route_ref', 'metric': 'route_cost', 'mean_delta': -0.002445319520638248, 'ci95_low': -0.003458673232575376, 'ci95_high': -0.0014216824605932964, 'wilcoxon_p': 3.8771235619217223e-07, 'n': 4200}` |
| 15 | Negative-score ambiguity fixed | PASS | `{}` |

**Overall: PASS.**

