#!/usr/bin/env python3
"""Build 01_frozen_artifact_manifest.csv: hash every frozen artifact F0
depends on, recording its actual last-touched commit (via git log, not
guessed) so the manifest is verifiable rather than asserted."""
import csv
import hashlib
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve()
F0 = HERE.parent
B1 = F0.parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def last_commit(relpath):
    out = subprocess.run(
        ["git", "log", "-1", "--format=%h %s", "--", relpath],
        cwd=B1, text=True, capture_output=True, check=True).stdout.strip()
    return out or "NOT_IN_GIT_HISTORY"


ROWS = [
    # --- Pair 1: H1/H2 mechanism dynamics + action simulation ---
    ("R0_hidden_mechanism_audit/src/simulator.py", "H1 mechanism dynamics + ACTIONS tuple", "frozen; import only", "any edit to dynamics/action set"),
    ("R0_hidden_mechanism_audit/src/suppression_simulator.py", "H2 mechanism dynamics", "frozen; import only", "any edit to dynamics"),
    ("R0_hidden_mechanism_audit/src/schemas.py", "O0_FIELDS observation schema", "frozen; import only", "field changes"),
    ("R0_hidden_mechanism_audit/src/config.py", "config loader", "frozen; import only", "loader semantics change"),
    ("R0_hidden_mechanism_audit/results/phase3c_redesign/matched_config_h1.yaml", "H1 frozen config", "read-only", "any parameter edit"),
    ("R0_hidden_mechanism_audit/results/phase3c_redesign/matched_config_h2.yaml", "H2 frozen config", "read-only", "any parameter edit"),
    ("CF0_strict_counterfactual_grounding/src/cf0_simulator.py", "checkpoint-fork strict 3-action simulator (H1/H2)", "frozen; import only", "any change to exogenous/endogenous RNG split or action-cost formula"),
    # --- Pair 1: F2 communication scheme + O0 generation (T0/T4/R4 telemetry layer) ---
    ("R0_hidden_mechanism_audit/src/phase5_communication.py", "F2 communication scheme (transmit_run/central_summary)", "frozen; import only", "scheme/feature edits"),
    ("R0_hidden_mechanism_audit/src/phase5_event.py", "event-scheme transmission (unused by F2 but imported)", "frozen; import only", "edits"),
    ("R0_hidden_mechanism_audit/src/phase5_final.py", "frozen_scheme/predict_frozen/fit_frozen_family", "frozen; import only", "classifier family/threshold edits"),
    ("R0_hidden_mechanism_audit/scripts/run_phase5_final.py", "transmission() -- delay/packet-loss channel simulation", "frozen; import only (transmission function)", "delay/loss model edits"),
    ("R0_hidden_mechanism_audit/results/phase5_minimal_distributed_evidence/development_frozen_schemes.json", "frozen F0-F4 scheme definitions incl. F2", "read-only", "any scheme edit"),
    ("R0_hidden_mechanism_audit/results/phase5_minimal_distributed_evidence/frozen_scheme_manifest.json", "quantization ranges", "read-only", "range edits"),
    ("R1_age_conditioned_decision_recovery/src/gate0_config.py", "gate0/gate1 config loader", "frozen; import only", "loader semantics change"),
    ("R1_age_conditioned_decision_recovery/src/protocol.py", "assert_fixed_schemes", "frozen; import only", "assertion relaxation"),
    ("R1_age_conditioned_decision_recovery/src/receiver_features.py", "selected_receiver_features (age/mask/silence)", "frozen; import only", "feature edits"),
    ("R1_age_conditioned_decision_recovery/src/rejection_diagnosis.py", "recover_score", "frozen; import only", "scoring edits"),
    ("R1_age_conditioned_decision_recovery/scripts/run_gate0_pilot.py", "simulate_records: per-seed O0 trajectory generation", "frozen; import only", "trajectory/split-index edits"),
    ("R1_age_conditioned_decision_recovery/scripts/run_gate1_formal.py", "ABLATIONS, message_hash, parameter_matched, prediction_row", "frozen; import only", "prediction-row cost formula edits"),
    ("R1_age_conditioned_decision_recovery/configs/gate1_formal.yaml", "seed splits, primary/auxiliary conditions, gate thresholds", "read-only (seed ranges superseded by F0 seed provenance for new seeds)", "condition/threshold edits"),
    ("R1_age_conditioned_decision_recovery/results/gate1_formal_age_necessity/gate1_frozen_receiver_models.json", "T0 frozen model (age_aware_model) + cost-sensitive reject constants", "frozen; T0 predictions read directly, never refit", "coefficient edits"),
    ("R2_mechanism_conditioned_risk_rescue/scripts/run_g1a_instrumented_replay.py", "G1A instrumented replay driver (O0 -> F2 traces -> decision-state manifest)", "frozen; reused by new driver with new seed set substituted for train/validation/dev seeds", "feature/scheme/threshold edits"),
    ("R2_mechanism_conditioned_risk_rescue/src/query_value_audit.py", "assert_seed_safety, sha256", "frozen; import only", "safety-check relaxation"),
    ("R2_mechanism_conditioned_risk_rescue/G1A_instrumented_replay_data/01_replay_manifest.csv", "old-seed replay manifest (reproduction baseline)", "read-only reference for reproduction check", "n/a (not modified)"),
    ("R2_mechanism_conditioned_risk_rescue/G1A_instrumented_replay_data/03_decision_state_manifest.csv", "old-seed decision-state manifest (reproduction baseline)", "read-only reference for reproduction check", "n/a (not modified)"),
    # --- Pair 1: T4/R4 feature extraction + fitting protocol ---
    ("R3_temporal_mechanism_evidence/scripts/run_g0_temporal_audit.py", "static_features, convolution_features, visible_series, fit_classifier, probabilities, action_rows, metrics -- T4 protocol", "frozen; import only", "any feature/model/window-selection edit"),
    ("R3_temporal_mechanism_evidence/scripts/run_g0b_risk_constrained_convolution.py", "fit_classifier_costsensitive, probabilities_biased, candidate selection -- R4 protocol", "frozen; import only", "any cost-sensitivity/candidate-selection edit"),
    ("R3_temporal_mechanism_evidence/G0_temporal_consistency_and_convolution/02_window_manifest.csv", "T4 per-outer-fold selected window (old seeds)", "read-only reference for reproduction check", "n/a"),
    ("R3_temporal_mechanism_evidence/G0_temporal_consistency_and_convolution/07_method_summary.csv", "T0/T4 archived summary (old seeds)", "read-only reference for reproduction check", "n/a"),
    ("R3_temporal_mechanism_evidence/G0B_risk_constrained_convolution/05_selected_candidate_by_outer_fold.csv", "R4 per-outer-fold selected candidate (old seeds)", "read-only reference for reproduction check", "n/a"),
    ("R3_temporal_mechanism_evidence/G0B_risk_constrained_convolution/07_method_summary.csv", "R4 archived summary (old seeds)", "read-only reference for reproduction check", "n/a"),
    # --- Pair 1: CF1 regret substitution protocol ---
    ("CF1_strict_regret_regeneration/scripts/reproduce_predictions.py", "step1: frozen T0/T4/R4 prediction reproduction", "frozen; reused unchanged for new-seed extension", "any refit/threshold change"),
    ("CF1_strict_regret_regeneration/scripts/run_cf1_checkpoints.py", "step2: strict checkpoint-fork mapping for T0/T4/R4 sample", "frozen; reused unchanged for new-seed extension", "mapping rule change"),
    ("CF1_strict_regret_regeneration/scripts/run_cf1_regret.py", "step3: strict regret substitution + decision rule", "frozen; decide()/REJECT_COST reused unchanged", "cost/decision-rule change"),
    # --- Pair 2: H3/H4 ---
    ("P2_radio_compute_pair_pilot/src/pilot.py", "H3/H4 synthetic mechanism, M0/M1/M2 features+classifier, strict 3-action fork", "frozen; import only", "any mechanism/feature/model/threshold edit"),
    ("P2_radio_compute_pair_pilot/01_frozen_config.yaml", "H3/H4 frozen config (documentation copy of CFG dict)", "read-only", "parameter edit"),
    ("P3_independent_risk_validation/src/run_p3.py", "P2-reuse protocol: outer-CV M0/M1/M2 fit, harm decomposition, bootstrap", "frozen; import only (functions reused by new driver with new seed set)", "protocol/threshold edit"),
]


def main():
    rows = []
    for relpath, role, allowed, forbidden in ROWS:
        path = B1 / relpath
        if not path.exists():
            rows.append({
                "artifact": relpath, "source_commit": "MISSING",
                "path": relpath, "sha256": "MISSING",
                "scientific_role": role, "allowed_operation": allowed,
                "forbidden_modification": forbidden})
            continue
        rows.append({
            "artifact": relpath.rsplit("/", 1)[-1],
            "source_commit": last_commit(relpath),
            "path": relpath, "sha256": sha256(path),
            "scientific_role": role, "allowed_operation": allowed,
            "forbidden_modification": forbidden})

    out = F0 / "01_frozen_artifact_manifest.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    missing = [r for r in rows if r["sha256"] == "MISSING"]
    if missing:
        (F0 / "26_f0_decision.txt").write_text("F0_FROZEN_ARTIFACT_INVALID\n")
        print("F0_FROZEN_ARTIFACT_INVALID:", [r["artifact"] for r in missing])
        raise SystemExit(1)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
