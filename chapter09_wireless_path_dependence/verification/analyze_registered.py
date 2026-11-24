#!/usr/bin/env python3
"""Frozen post-gate analysis for the single registered B3' R0 execution."""
import itertools
import json
import math
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import norm

ARMS = ['KAPPA_PLUS', 'KAPPA_ZERO', 'KAPPA_MINUS']
PAIR_NAMES = [('KAPPA_PLUS', 'KAPPA_ZERO'), ('KAPPA_PLUS', 'KAPPA_MINUS'), ('KAPPA_ZERO', 'KAPPA_MINUS')]

def ci_mean(series):
    x = series.astype(float)
    n = len(x)
    mean = x.mean()
    half = 1.959963984540054 * x.std(ddof=1) / math.sqrt(n)
    return float(mean), float(mean-half), float(mean+half)

def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    den = 1 + z*z/n
    centre = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return float(p), float(centre-half), float(centre+half)

def main():
    root = Path(sys.argv[1])
    gate = json.loads((root / 'mechanism_gate.json').read_text())
    if gate['status'] != 'MECHANISM_PASS':
        raise SystemExit('Refusing operational analysis: mechanism gate did not pass')
    endpoints = pd.read_csv(root / 'endpoint_results.csv')
    operations = pd.read_csv(root / 'operational_results.csv')
    assert len(endpoints) == 6000 and len(operations) == 12000
    assert sorted(endpoints.world_id.unique()) == list(range(1, 2001))
    assert set(endpoints.arm) == set(ARMS)

    endpoint_summary = []
    for arm in ARMS:
        data = endpoints[endpoints.arm == arm]
        sd = float(data.endpoint_difference_db.std(ddof=1))
        disagree = int(data.canonical_disagreement.sum())
        p, lo, hi = wilson(disagree, len(data))
        endpoint_summary.append({'arm': arm, 'n': len(data), 'endpoint_difference_sd_db': sd,
                                 'canonical_disagreement_count': disagree,
                                 'canonical_disagreement_probability': p,
                                 'canonical_disagreement_ci95': [lo, hi]})

    operational_summary = []
    by_arm_branch = operations.set_index(['world_id', 'arm', 'branch']).sort_index()
    for arm in ARMS:
        for branch in ['A', 'B']:
            row = by_arm_branch.xs((arm, branch), level=('arm', 'branch'))
            values = {'arm': arm, 'branch': branch, 'n': len(row)}
            for metric in ['terminal_rx_power_dbm', 'terminal_rate_bps', 'terminal_delivered_bits',
                           'route_delivered_bits', 'route_average_rate_bps', 'route_outage_fraction']:
                mean, lo, hi = ci_mean(row[metric])
                values[metric] = {'mean': mean, 'ci95': [lo, hi]}
            operational_summary.append(values)

    per_arm = {}
    for arm in ARMS:
        a = by_arm_branch.xs((arm, 'A'), level=('arm', 'branch')).sort_index()
        b = by_arm_branch.xs((arm, 'B'), level=('arm', 'branch')).sort_index()
        frame = pd.DataFrame(index=a.index)
        frame['terminal_delivered_bits_A_minus_B'] = a.terminal_delivered_bits - b.terminal_delivered_bits
        frame['terminal_abs_delivered_bits_difference'] = frame.terminal_delivered_bits_A_minus_B.abs()
        frame['selected_A_terminal_rate'] = (a.terminal_rate_bps >= b.terminal_rate_bps).astype(int)
        frame['route_delivered_bits_A_minus_B'] = a.route_delivered_bits - b.route_delivered_bits
        frame['route_average_rate_A_minus_B_bps'] = a.route_average_rate_bps - b.route_average_rate_bps
        frame['route_outage_fraction_A_minus_B'] = a.route_outage_fraction - b.route_outage_fraction
        frame['A_terminal_delivered_bits'] = a.terminal_delivered_bits
        frame['B_terminal_delivered_bits'] = b.terminal_delivered_bits
        frame['A_route_delivered_bits'] = a.route_delivered_bits
        frame['B_route_delivered_bits'] = b.route_delivered_bits
        per_arm[arm] = frame

    arm_summary = []
    for arm in ARMS:
        item = {'arm': arm, 'n': 2000}
        for metric in per_arm[arm].columns:
            if metric == 'selected_A_terminal_rate':
                count = int(per_arm[arm][metric].sum())
                p, lo, hi = wilson(count, 2000)
                item[metric] = {'count': count, 'probability': p, 'ci95': [lo, hi]}
            else:
                mean, lo, hi = ci_mean(per_arm[arm][metric])
                item[metric] = {'mean': mean, 'ci95': [lo, hi]}
        arm_summary.append(item)

    contrast_rows = []
    pass_evidence = []
    for left, right in PAIR_NAMES:
        for metric in [m for m in per_arm[left].columns if m != 'selected_A_terminal_rate']:
            difference = per_arm[left][metric] - per_arm[right][metric]
            mean, lo, hi = ci_mean(difference)
            reference = (per_arm[left][metric].abs().mean() + per_arm[right][metric].abs().mean()) / 2
            contrast_rows.append({'comparison': f'{left}_MINUS_{right}', 'metric': metric,
                                  'kind': 'mean_paired_difference', 'n': 2000, 'estimate': mean,
                                  'ci95_lower': lo, 'ci95_upper': hi,
                                  'relative_to_mean_absolute_metric': float(abs(mean)/reference) if reference else None})
        selection_disagreement = (per_arm[left].selected_A_terminal_rate != per_arm[right].selected_A_terminal_rate)
        count = int(selection_disagreement.sum())
        p, lo, hi = wilson(count, 2000)
        contrast_rows.append({'comparison': f'{left}_MINUS_{right}', 'metric': 'selected_route_disagreement',
                              'kind': 'paired_binary_disagreement', 'n': 2000, 'estimate': p,
                              'ci95_lower': lo, 'ci95_upper': hi, 'relative_to_mean_absolute_metric': None})
        absolute_change = per_arm[left].terminal_abs_delivered_bits_difference - per_arm[right].terminal_abs_delivered_bits_difference
        mean, lo_abs, hi_abs = ci_mean(absolute_change)
        baseline = (per_arm[left].terminal_abs_delivered_bits_difference.mean() +
                    per_arm[right].terminal_abs_delivered_bits_difference.mean()) / 2
        if not (lo_abs <= 0 <= hi_abs) and abs(mean) / baseline >= 0.01:
            pass_evidence.append({'comparison': f'{left}_MINUS_{right}', 'criterion': 'terminal_absolute_contrast',
                                  'estimate': mean, 'relative_effect': float(abs(mean)/baseline), 'ci95': [lo_abs, hi_abs]})
        if p >= 0.01 and lo > 0:
            pass_evidence.append({'comparison': f'{left}_MINUS_{right}', 'criterion': 'terminal_selection_disagreement',
                                  'estimate': p, 'ci95': [lo, hi]})
    pd.DataFrame(contrast_rows).to_csv(root / 'paired_contrasts.csv', index=False)
    status = 'WITNESS_PASS' if pass_evidence else 'WITNESS_WEAK'
    summary = {'status': status, 'mechanism_status': gate['status'], 'endpoint_diagnostic': endpoint_summary,
               'operational_by_arm_branch': operational_summary, 'primary_intervention_by_arm': arm_summary,
               'witness_pass_evidence': pass_evidence,
               'frozen_status_rule': 'EXECUTION_LOCK.md: primary terminal absolute contrast or primary terminal selection disagreement only'}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    lines = ['# Registered R0 analysis', '', f'**Status:** `{status}`.', '',
             'All intervals use the pre-locked methods in `EXECUTION_LOCK.md`.  This report does not interpret a coupling contrast as a population marginal mean action-value bias.', '',
             '## Endpoint diagnostic', '']
    for row in endpoint_summary:
        lines.append(f"- {row['arm']}: endpoint-difference SD {row['endpoint_difference_sd_db']:.6f} dB; canonical disagreement {row['canonical_disagreement_count']}/2000 = {row['canonical_disagreement_probability']:.4f} (95% CI {row['canonical_disagreement_ci95'][0]:.4f}, {row['canonical_disagreement_ci95'][1]:.4f}).")
    lines += ['', '## Primary status evidence', '']
    if pass_evidence:
        for item in pass_evidence:
            lines.append(f"- `{item['comparison']}`: `{item['criterion']}`; estimate {item['estimate']}; 95% CI {item['ci95']}.")
    else:
        lines.append('- No locked primary terminal status criterion was met.')
    lines += ['', 'Detailed arm results are in `summary.json`; all locked paired contrasts are in `paired_contrasts.csv`.']
    (root / 'analysis_summary.md').write_text('\n'.join(lines) + '\n')

    plots = root / 'plots'
    plots.mkdir(exist_ok=True)
    (plots / 'README.md').write_text('Plot directory reserved by the registered protocol. Numeric source data are endpoint_results.csv and paired_contrasts.csv.\n')

if __name__ == '__main__':
    main()
