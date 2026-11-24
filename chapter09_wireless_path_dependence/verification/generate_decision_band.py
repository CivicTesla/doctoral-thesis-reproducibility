"""Deterministic R3 Makarov band calculation; generates no stochastic worlds."""
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr

OUT = Path(__file__).resolve().parent
SIGMA, D_COR, LA, LB = 4.0, 37.0, 10.0, 60.0
RA, RB = math.exp(-LA / D_COR), math.exp(-LB / D_COR)
SA = SIGMA * math.sqrt(1.0 - RA * RA)
SB = SIGMA * math.sqrt(1.0 - RB * RB)
GMIN_SIGMA = SIGMA * math.sqrt(
    (RA - RB) ** 2
    + (math.sqrt(1.0 - RA * RA) - math.sqrt(1.0 - RB * RB)) ** 2
)
PHI_NORM = math.sqrt(2.0 * math.pi)


def normal_pdf(u):
    return math.exp(-0.5 * u * u) / PHI_NORM


def conditional_band(u, delta):
    """Exact conditional continuous Makarov extrema.

    Stationary points solve f_b(x+delta)=f_a(x).  Taking logarithms gives
    ((x-ma)/sa)^2 - ((x+delta-mb)/sb)^2 = 2 log(sb/sa).
    The two objectives vanish at both tails, so their global extrema occur
    among these roots and zero.
    """
    ma, mb = -SIGMA * RA * u, -SIGMA * RB * u
    a = 1.0 / SA**2 - 1.0 / SB**2
    b = -2.0 * ma / SA**2 - 2.0 * (delta - mb) / SB**2
    c = ma**2 / SA**2 - (delta - mb) ** 2 / SB**2 - 2.0 * math.log(SB / SA)
    roots = []
    if abs(a) < 1e-14:
        if abs(b) >= 1e-14:
            roots = [-c / b]
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            root = math.sqrt(discriminant)
            roots = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
    lower_candidates, complement_candidates = [0.0], [0.0]
    for x in roots:
        fa = ndtr((x - ma) / SA)
        fb = ndtr((x + delta - mb) / SB)
        lower_candidates.append(fb - fa)
        complement_candidates.append(fa - fb)
    return max(lower_candidates), 1.0 - max(complement_candidates)


def band(delta):
    lower = quad(
        lambda u: conditional_band(u, delta)[0] * normal_pdf(u),
        -10.0, 10.0, epsabs=2e-10, epsrel=2e-10, limit=400,
    )[0]
    upper = quad(
        lambda u: conditional_band(u, delta)[1] * normal_pdf(u),
        -10.0, 10.0, epsabs=2e-10, epsrel=2e-10, limit=400,
    )[0]
    return lower, upper


def convention_probability(delta, sd):
    return ndtr(delta / sd)


def row(eta):
    delta = eta * GMIN_SIGMA
    lower, upper = band(delta)
    return {
        "eta": eta,
        "delta_db": delta,
        "P_lower": lower,
        "P_upper": upper,
        "width": upper - lower,
        "P_PLUS": convention_probability(delta, GMIN_SIGMA),
        "P_ZERO": convention_probability(delta, 5.2129492075086086),
        "P_MINUS": convention_probability(delta, 6.8880636365970336),
    }


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    frozen = [row(eta) for eta in (0.0, 0.5, 1.0, 2.0, 3.0)]
    dense = [row(eta / 20.0) for eta in range(0, 181)]
    write_csv(OUT / "decision_band.csv", frozen)
    write_csv(OUT / "decision_band_dense.csv", dense)

    certificates = []
    for q in (0.90, 0.95, 0.99):
        delta = brentq(lambda x: band(x)[0] - q, 0.0, 30.0, xtol=3e-10)
        lower, upper = band(delta)
        certificates.append({"q": q, "delta_rob_db": delta,
                             "eta_rob": delta / GMIN_SIGMA,
                             "P_lower": lower, "P_upper": upper})
    write_csv(OUT / "robust_margin_certificate.csv", certificates)

    plt.figure(figsize=(6.4, 4.0))
    x = [r["eta"] for r in dense]
    lo = [r["P_lower"] for r in dense]
    hi = [r["P_upper"] for r in dense]
    plt.fill_between(x, lo, hi, color="#86bde8", alpha=0.45,
                     label="sharp all-completion band")
    plt.plot(x, [r["P_PLUS"] for r in dense], label="PLUS", color="#1b6ca8")
    plt.plot(x, [r["P_ZERO"] for r in dense], label="ZERO", color="#444444")
    plt.plot(x, [r["P_MINUS"] for r in dense], label="MINUS", color="#be5b32")
    plt.xlabel(r"normalized deterministic margin $\eta$")
    plt.ylabel(r"$P(\mathrm{select\ A})$")
    plt.ylim(-0.02, 1.02)
    plt.xlim(0, 9)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT / "decision_band.pdf")


if __name__ == "__main__":
    main()
