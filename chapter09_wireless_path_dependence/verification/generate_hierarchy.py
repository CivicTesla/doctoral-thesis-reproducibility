"""Deterministic R3b identification-hierarchy calculation; no world generation."""
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
R3 = ROOT.parent / "r3"
SIGMA, D, LA, LB = 4.0, 37.0, 10.0, 60.0
RA, RB = math.exp(-LA / D), math.exp(-LB / D)
QA, QB = math.sqrt(1 - RA**2), math.sqrt(1 - RB**2)
GMIN_SIGMA = SIGMA * math.sqrt((RA - RB)**2 + (QA - QB)**2)
SD = {
    "PLUS": GMIN_SIGMA,
    "ZERO": SIGMA * math.sqrt((RA - RB)**2 + QA**2 + QB**2),
    "MINUS": SIGMA * math.sqrt((RA - RB)**2 + (QA + QB)**2),
}
GRID = (0.0, 0.5, 1.0, 2.0, 3.0)


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def c0_lower(delta):
    # Sharp Makarov lower probability for Delta + S_b - S_a >= 0.
    return 2 * norm.cdf(delta / (2 * SIGMA)) - 1


def p_named(delta, arm):
    return norm.cdf(delta / SD[arm])


def read_r3():
    with (R3 / "decision_band.csv").open(newline="") as stream:
        return {float(r["eta"]): r for r in csv.DictReader(stream)}


def hierarchy_row(eta, c1):
    delta = eta * GMIN_SIGMA
    c0lo, c0hi = c0_lower(delta), 1.0
    c1lo, c1hi = float(c1["P_lower"]), float(c1["P_upper"])
    plus, zero, minus = (p_named(delta, arm) for arm in ("PLUS", "ZERO", "MINUS"))
    c2lo, c2hi = minus, plus
    return {
        "eta": eta, "delta_db": delta,
        "C0_lower": c0lo, "C0_upper": c0hi, "C0_width": c0hi - c0lo,
        "C1_lower": c1lo, "C1_upper": c1hi, "C1_width": c1hi - c1lo,
        "C2_lower": c2lo, "C2_upper": c2hi, "C2_width": c2hi - c2lo,
        "P_PLUS": plus, "P_ZERO": zero, "P_MINUS": minus,
    }


def main():
    frozen_c1 = read_r3()
    rows = [hierarchy_row(eta, frozen_c1[eta]) for eta in GRID]
    write_csv(ROOT / "identification_hierarchy.csv", rows)

    gains = []
    for row in rows:
        w0, w1, w2 = row["C0_width"], row["C1_width"], row["C2_width"]
        gains.append({
            "eta": row["eta"], "delta_db": row["delta_db"],
            "W0": w0, "W1": w1, "W2": w2,
            "C0_minus_C1": w0 - w1,
            "C0_to_C1_fraction_reduced": (w0 - w1) / w0 if w0 else "",
            "C1_minus_C2": w1 - w2,
            "C1_to_C2_fraction_reduced": (w1 - w2) / w1 if w1 else "",
        })
    write_csv(ROOT / "shared_prefix_identification_gain.csv", gains)

    certificates = []
    for q in (0.90, 0.95, 0.99):
        z = norm.ppf(q)
        plus, zero, minus = (SD[arm] * z for arm in ("PLUS", "ZERO", "MINUS"))
        c0 = 2 * SIGMA * norm.ppf((1 + q) / 2)
        with (R3 / "robust_margin_certificate.csv").open(newline="") as stream:
            c1 = next(float(r["delta_rob_db"]) for r in csv.DictReader(stream)
                      if abs(float(r["q"]) - q) < 1e-12)
        certificates.append({
            "q": q,
            "PLUS_margin_db": plus, "ZERO_margin_db": zero, "MINUS_margin_db": minus,
            "C2_gaussian_robust_margin_db": minus,
            "C1_all_completion_robust_margin_db": c1,
            "C0_unconditional_robust_margin_db": c0,
            "PLUS_eta": plus / GMIN_SIGMA, "ZERO_eta": zero / GMIN_SIGMA,
            "MINUS_eta": minus / GMIN_SIGMA, "C2_eta": minus / GMIN_SIGMA,
            "C1_eta": c1 / GMIN_SIGMA, "C0_eta": c0 / GMIN_SIGMA,
        })
    write_csv(ROOT / "robust_margin_hierarchy.csv", certificates)

    dense_c1 = []
    with (R3 / "decision_band_dense.csv").open(newline="") as stream:
        dense_c1 = list(csv.DictReader(stream))
    x = [float(r["eta"]) for r in dense_c1]
    delta = [eta * GMIN_SIGMA for eta in x]
    c0lo = [c0_lower(v) for v in delta]
    c1lo = [float(r["P_lower"]) for r in dense_c1]
    c1hi = [float(r["P_upper"]) for r in dense_c1]
    plus = [p_named(v, "PLUS") for v in delta]
    zero = [p_named(v, "ZERO") for v in delta]
    minus = [p_named(v, "MINUS") for v in delta]
    plt.figure(figsize=(6.8, 4.25))
    plt.fill_between(x, c0lo, 1.0, color="#d8d8d8", alpha=0.78,
                     label="C0: endpoint marginals")
    plt.fill_between(x, c1lo, c1hi, color="#8dbfe4", alpha=0.70,
                     label="C1: shared-prefix branch laws")
    plt.fill_between(x, minus, plus, color="#f5c276", alpha=0.82,
                     label="C2: conditional joint Gaussian")
    plt.plot(x, plus, color="#1261a0", lw=2.0, label="PLUS")
    plt.plot(x, zero, color="#404040", lw=2.0, label="ZERO")
    plt.plot(x, minus, color="#bb5a31", lw=2.0, label="MINUS")
    plt.xlabel(r"normalized deterministic margin $\eta$")
    plt.ylabel(r"$P(\mathrm{select\ A})$")
    plt.xlim(0.0, 5.5)
    plt.ylim(-0.02, 1.02)
    plt.legend(frameon=False, fontsize=8.7, ncol=2, loc="lower right")
    plt.tight_layout()
    plt.savefig(ROOT / "identification_hierarchy.pdf")


if __name__ == "__main__":
    main()
