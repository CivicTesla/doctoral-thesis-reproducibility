#!/usr/bin/env python3
"""Frozen R0 mechanism gate; it reads only mechanism_validation.csv."""
import csv
import json
import math
import sys
from pathlib import Path

from scipy.stats import chi2

N = 2000
SIGMA = 4.0
SIGMA2 = 16.0
Z99 = 2.5758293035489004
R_A = math.exp(-10.0 / 37.0)
R_B = math.exp(-60.0 / 37.0)
ADJ = math.exp(-1.0 / 37.0)

def covariance_band(target):
    return Z99 * math.sqrt((SIGMA2 * SIGMA2 + target * target) / N)

def main():
    root = Path(sys.argv[1])
    rows = list(csv.DictReader((root / 'mechanism_validation.csv').open()))
    failures = []
    if len(rows) != 6:
        failures.append(f'expected 6 arm/branch rows, found {len(rows)}')
    low_var = (N - 1) * SIGMA2 / chi2.ppf(0.995, N - 1)
    high_var = (N - 1) * SIGMA2 / chi2.ppf(0.005, N - 1)
    expected_cov = {'A': SIGMA2 * R_A, 'B': SIGMA2 * R_B}
    for row in rows:
        arm, branch = row['arm'], row['branch']
        f = {key: float(value) for key, value in row.items()
             if key not in {'arm', 'branch', 'native_shadowing_enabled'}}
        if int(f['n']) != N or row['native_shadowing_enabled'] != 'false':
            failures.append(f'{arm}/{branch}: count or native-shadowing invariant failed')
        if abs(f['path_length_m'] - (10.0 if branch == 'A' else 60.0)) > 1e-9:
            failures.append(f'{arm}/{branch}: frozen mesh length mismatch')
        if f['conditional_bridge_reconstruction_error'] > 1e-10:
            failures.append(f'{arm}/{branch}: bridge reconstruction identity failed')
        if abs(f['endpoint_mean']) > Z99 * SIGMA / math.sqrt(N):
            failures.append(f'{arm}/{branch}: endpoint mean outside frozen 99% reference band')
        if not low_var <= f['endpoint_variance'] <= high_var:
            failures.append(f'{arm}/{branch}: endpoint variance outside frozen 99% reference interval')
        target = expected_cov[branch]
        if abs(f['cov_s0_endpoint'] - target) > covariance_band(target):
            failures.append(f'{arm}/{branch}: S0-endpoint covariance outside frozen 99% reference band')
        target_adjacent = SIGMA2 * ADJ
        if abs(f['first_1m_adjacent_cov'] - target_adjacent) > covariance_band(target_adjacent):
            failures.append(f'{arm}/{branch}: first 1m covariance outside frozen 99% reference band')
    status = 'MECHANISM_PASS' if not failures else 'INVALID_MECHANISM_WITNESS'
    report = {'status': status, 'n_worlds': N, 'rows': len(rows), 'variance_99pct_interval': [low_var, high_var],
              'endpoint_mean_99pct_halfwidth': Z99 * SIGMA / math.sqrt(N), 'failures': failures}
    (root / 'mechanism_gate.json').write_text(json.dumps(report, indent=2) + '\n')
    print(status)
    for failure in failures:
        print(failure)
    return 0 if not failures else 1

if __name__ == '__main__':
    raise SystemExit(main())
