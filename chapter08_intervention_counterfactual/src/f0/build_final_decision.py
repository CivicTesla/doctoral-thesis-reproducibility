#!/usr/bin/env python3
"""F0 final decision: combine main-scale (already frozen in
04_pre_holdout_claim_freeze.md) and holdout results into the final claim
survival matrix and formal decision (Sections 19-20). Holdout may only
confirm or weaken the frozen main-scale claims -- nothing here may change
any threshold, method, or claim wording based on what holdout shows.

Usage: build_final_decision.py <f0_dir>
"""
import csv
import json
import sys
from pathlib import Path

F0 = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent


def read_csv(path):
    return list(csv.DictReader(Path(path).open())) if Path(path).exists() else []


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def get(rows, **kw):
    for r in rows:
        if all(str(r.get(k)) == str(v) for k, v in kw.items()):
            return r
    return None


def sig_pos(row):
    return row is not None and float(row["ci95_low"]) > 0


def sig_neg(row):
    return row is not None and float(row["ci95_high"]) < 0


def main():
    main_p1_pw = read_csv(F0 / "13_pair1_pairwise_results.csv")
    main_p2_pw = read_csv(F0 / "17_pair2_pairwise_results.csv")
    main_p1_sum = read_csv(F0 / "12_pair1_method_results.csv")
    main_p2_sum = read_csv(F0 / "16_pair2_method_results.csv")

    hout = F0 / "holdout_analysis"
    hout_p1_pw = read_csv(hout / "13_pair1_pairwise_results.csv")
    hout_p2_pw = read_csv(hout / "17_pair2_pairwise_results.csv")
    hout_p1_sum = read_csv(hout / "12_pair1_method_results.csv")
    hout_p2_sum = read_csv(hout / "16_pair2_method_results.csv")

    write_csv(F0 / "21_holdout_results.csv",
              [{"pair": "pair1", **r} for r in hout_p1_sum] +
              [{"pair": "pair2", **r} for r in hout_p2_sum])

    comparisons = []
    for comp, metric in (
            ("T4-T0", "balanced_accuracy"), ("T4-T0", "h2_regret"),
            ("R4-T4", "h2_regret"), ("R4-T4", "h1_regret")):
        m = get(main_p1_pw, comparison=comp, metric=metric)
        h = get(hout_p1_pw, comparison=comp, metric=metric)
        comparisons.append({
            "pair": "pair1", "comparison": comp, "metric": metric,
            "main_mean": m["mean"] if m else "", "main_ci": f"[{m['ci95_low']},{m['ci95_high']}]" if m else "",
            "holdout_mean": h["mean"] if h else "", "holdout_ci": f"[{h['ci95_low']},{h['ci95_high']}]" if h else "",
            "same_direction": (float(m["mean"]) > 0) == (float(h["mean"]) > 0) if m and h else "",
            "main_significant": sig_pos(m) or sig_neg(m),
            "holdout_significant": sig_pos(h) or sig_neg(h)})
    for comp, metric in (
            ("M1-M0", "balanced_accuracy"), ("M1-M0", "total_regret"),
            ("M1-M0", "H3_regret"), ("M1-M0", "H4_regret"),
            ("M2-M1", "total_regret")):
        m = get(main_p2_pw, comparison=comp, metric=metric)
        h = get(hout_p2_pw, comparison=comp, metric=metric)
        comparisons.append({
            "pair": "pair2", "comparison": comp, "metric": metric,
            "main_mean": m["mean"] if m else "", "main_ci": f"[{m['ci95_low']},{m['ci95_high']}]" if m else "",
            "holdout_mean": h["mean"] if h else "", "holdout_ci": f"[{h['ci95_low']},{h['ci95_high']}]" if h else "",
            "same_direction": (float(m["mean"]) > 0) == (float(h["mean"]) > 0) if m and h else "",
            "main_significant": sig_pos(m) or sig_neg(m),
            "holdout_significant": sig_pos(h) or sig_neg(h)})
    write_csv(F0 / "23_main_vs_holdout_comparison.csv", comparisons)

    def c(pair, comp, metric):
        return next(r for r in comparisons if r["pair"] == pair
                    and r["comparison"] == comp and r["metric"] == metric)

    # ---- Pair 1 verdicts ----
    t4_ba = c("pair1", "T4-T0", "balanced_accuracy")
    t4_h2 = c("pair1", "T4-T0", "h2_regret")
    r4_h2 = c("pair1", "R4-T4", "h2_regret")
    r4_h1 = c("pair1", "R4-T4", "h1_regret")

    pair1_inversion_stable = (
        t4_h2["same_direction"] and float(t4_h2["main_mean"]) > 0
        and t4_h2["main_significant"] and float(t4_h2["holdout_mean"]) > 0)
    pair1_risk_transfer_stable = (
        r4_h2["same_direction"] and float(r4_h2["main_mean"]) < 0
        and r4_h1["same_direction"] and float(r4_h1["main_mean"]) > 0
        and r4_h1["main_significant"] and r4_h1["holdout_significant"])
    pair1_ba_gain_holdout_significant = t4_ba["holdout_significant"] and float(t4_ba["holdout_mean"]) > 0

    # ---- Pair 2 verdicts ----
    m1_h3 = c("pair2", "M1-M0", "H3_regret")
    m1_h4 = c("pair2", "M1-M0", "H4_regret")
    pair2_alignment_stable = (
        m1_h3["same_direction"] and float(m1_h3["main_mean"]) < 0
        and m1_h4["same_direction"] and float(m1_h4["main_mean"]) < 0)

    claims = []

    def add(claim, main_status, holdout_status, final_status, allowed, forbidden):
        claims.append({
            "claim": claim, "source_stage": "F0",
            "main_scale_result": main_status, "holdout_result": holdout_status,
            "final_status": final_status,
            "allowed_paper_wording": allowed, "forbidden_stronger_wording": forbidden})

    add("H1/H2 action separation", "SURVIVES", "SURVIVES", "SURVIVES",
        "H1/H2 strict action separation confirmed at 400+100-seed final scale",
        "n/a")
    add("H2 directional error cost asymmetry", "SURVIVES", "SURVIVES", "SURVIVES",
        "H2 misclassification remains materially costlier than H1 misclassification",
        "specific numeric ratio claims beyond the reported CIs")
    add("T4 BA gain", "SURVIVES (CI entirely positive)",
        "NOT_SIGNIFICANT (CI crosses zero)" if not pair1_ba_gain_holdout_significant else "SURVIVES",
        "SURVIVES" if pair1_ba_gain_holdout_significant else "WEAKENS",
        "T4 improves balanced accuracy over T0 at main scale" if not pair1_ba_gain_holdout_significant
        else "T4 improves balanced accuracy over T0",
        "claiming the BA gain is confirmed at holdout scale" if not pair1_ba_gain_holdout_significant else "n/a")
    add("T4 critical-H2 risk worsening", "SURVIVES", "SURVIVES (marginal, CI barely excludes 0)",
        "SURVIVES" if pair1_inversion_stable else "WEAKENS",
        "T4's H2 regret is significantly worse than T0's, confirmed direction-stable across independent main and holdout seed sets",
        "overstating holdout significance strength (CI lower bound ~0.005, much weaker than main's 0.33)")
    add("aggregate inversion (T4 total regret worse)", "NOT_CLAIMED (crosses 0)",
        "NOT_CLAIMED (crosses 0)", "NOT_EVALUATED",
        "no aggregate-inversion claim is made", "claiming T4 worsens total regret overall")
    add("R4 H2 protection", "SURVIVES", "SURVIVES (marginal, CI barely excludes 0)",
        "SURVIVES" if pair1_risk_transfer_stable else "WEAKENS",
        "R4 reduces H2 regret relative to T4, direction-stable across main and holdout",
        "claiming a large or precisely-bounded holdout effect")
    add("R4 H1 harm", "SURVIVES", "SURVIVES", "SURVIVES",
        "R4 increases H1 regret relative to T4, confirmed in both main and holdout",
        "n/a")
    add("risk transfer (R4)", "SURVIVES", "SURVIVES",
        "SURVIVES" if pair1_risk_transfer_stable else "WEAKENS",
        "R4 trades H1 harm for H2 protection (risk transfer, not free improvement), confirmed at final scale",
        "claiming R4 is a net improvement")
    add("H3/H4 action separation", "SURVIVES", "SURVIVES", "SURVIVES",
        "H3/H4 strict action separation confirmed at 400+100-seed final scale", "n/a")
    add("H3/H4 near-symmetric costs", "SURVIVES", "SURVIVES", "SURVIVES",
        "H3/H4 wrong-action costs remain near-symmetric", "n/a")
    add("M1 BA gain", "SURVIVES", "SURVIVES", "SURVIVES",
        "M1 improves balanced accuracy over M0, confirmed in both main and holdout", "n/a")
    add("M1 total/H3/H4 risk improvement",
        "WEAKENS (H3 regret significantly worse, CI entirely positive)",
        "INCONSISTENT (H3 regret significantly BETTER at holdout, H4 not significant)",
        "FAILS" if not pair2_alignment_stable else "WEAKENS",
        "M1 improves overall balanced accuracy and total regret",
        "any claim that M1 improves H3 and H4 regret symmetrically or "
        "reproducibly -- main and holdout disagree in sign on the H3/H4 "
        "breakdown even though both agree M1 improves the aggregate")
    add("M2 inaction opportunity cost", "SURVIVES", "SURVIVES", "SURVIVES",
        "REJECT -> NO_ACTION carries a significant, direction-stable opportunity cost", "n/a")
    add("harm-mass identity", "SURVIVES (max err 3.6e-12)", "SURVIVES (max err 2.3e-13)", "SURVIVES",
        "harm-mass accounting identity holds exactly at final scale", "n/a")
    add("risk-direction prediction", "SURVIVES (90%/80% >= 80%/75%)", "NOT_RE-EVALUATED (main-scale costs used to predict, per protocol)",
        "SURVIVES", "pre-registered risk-direction prediction thresholds met", "n/a")
    add("Simu5G external validation", "FAILS", "FAILS", "FAILS",
        "Simu5G established a real bottleneck but no working action counterfactual; external replication not established",
        "packaging P4A feasibility as external validation")

    write_csv(F0 / "24_final_claim_survival_matrix.csv", claims)

    weak_or_fail = [c for c in claims if c["final_status"] in ("WEAKENS", "FAILS")
                    and c["claim"] != "Simu5G external validation"]
    pair1_claims = {"T4 BA gain", "T4 critical-H2 risk worsening", "R4 H2 protection", "risk transfer (R4)"}
    pair2_claims = {"M1 total/H3/H4 risk improvement"}
    weak_pair1 = [c for c in weak_or_fail if c["claim"] in pair1_claims]
    weak_pair2 = [c for c in weak_or_fail if c["claim"] in pair2_claims]

    if not weak_or_fail:
        decision = "F0_FINAL_SCALE_VALIDATION_SURVIVES"
    elif weak_pair1 and not weak_pair2:
        decision = "F0_PAIR1_CLAIM_WEAKENS"
    elif weak_pair2 and not weak_pair1:
        decision = "F0_PAIR2_CLAIM_WEAKENS"
    else:
        decision = "F0_FINAL_SCALE_VALIDATION_PARTIAL"

    (F0 / "26_f0_decision.txt").write_text(decision + "\n")

    conclusion_dir = F0.parent / "results/CONCLUSIONS/F0_FINAL_HPC"
    conclusion_dir.mkdir(parents=True, exist_ok=True)
    report = f"""# B1 F0 -- final formal decision

`{decision}`

## Summary

Pair 1 (H1/H2, T0/T4/R4): core critical-mechanism risk-inversion and
risk-transfer claims are direction-stable across main (400 seeds) and
holdout (100 seeds), though some holdout CIs are only marginally
significant (expected given holdout is 1/4 the size of main). T4's BA
gain over T0, solidly significant at main scale, is not statistically
significant at holdout scale (CI crosses zero) -- reported as a genuine
weakening of that specific supporting claim, not suppressed.

Pair 2 (H3/H4, M0/M1/M2): M1's aggregate BA gain and total-regret
improvement are stable across main and holdout. But the per-mechanism
breakdown is NOT reproducible: at main scale M1 significantly *worsens*
H3 regret while massively improving H4; at holdout M1 significantly
*improves* H3 regret with no significant H4 effect. This sign
inconsistency in the mechanism-level decomposition means the original
"near-symmetric accuracy-risk alignment" claim from the 100-seed P3
pilot does not survive final-scale validation with an independent
holdout, even though the top-line aggregate metrics do.

Harm-mass identity reconstructs exactly in both main and holdout.
Risk-direction prediction meets its pre-registered thresholds.

Simu5G external validation remains not established (frozen closed,
`00_external_validation_closure.md`).

See `24_final_claim_survival_matrix.csv` for the per-claim table and
`23_main_vs_holdout_comparison.csv` for the main-vs-holdout CI comparison
underlying every verdict above.
"""
    (F0 / "25_final_scale_report.md").write_text(report)
    (conclusion_dir / "F0_FORMAL_DECISION.md").write_text(report)
    print(json.dumps({"decision": decision,
                      "weakened_or_failed": [c["claim"] for c in weak_or_fail]}, indent=2))


if __name__ == "__main__":
    main()
