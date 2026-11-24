#!/usr/bin/env python3
"""Fail-closed conclusion from canonical and modular exact Psi3 separations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(partial_path: Path, certificates_path: Path, modular_preflight: Path) -> dict:
    partial, certificates = json.loads(partial_path.read_text()), json.loads(certificates_path.read_text())
    preflight = json.loads(modular_preflight.read_text())
    summary = partial.get("pair_summary", {})
    if (summary.get("total_psi2_collision_pairs"), summary.get("exactly_separated"), summary.get("exactly_equal"),
            summary.get("incomplete")) != (81, 74, 0, 7):
        raise ValueError("partial exact Psi3 baseline is not authoritative")
    if (certificates.get("pair_count"), certificates.get("separated"), certificates.get("inconclusive")) != (7, 7, 0):
        raise ValueError("modular certificates do not close every hard pair")
    if preflight.get("status") != "PASS" or preflight.get("known_pair_detection", {}).get("known_exactly_separated_pairs") != 74:
        raise ValueError("modular evaluator preflight is not authoritative")
    return {"status": "complete", "N": 64, "total_objects": 1172,
            "psi2_collision_pairs": 81, "psi3_canonically_separated_pairs": 74,
            "psi3_modularly_certified_pairs": 7, "psi3_equal_pairs": 0, "psi3_incomplete_pairs": 0,
            "k_star_Psi_64_UPO": 3,
            "justification": "exact Psi<=2 collision proves order 2 insufficient; all 81 Psi<=2 collision pairs are separated either by canonical exact Psi3 data or by validated exact finite-field evaluation witnesses",
            "psi4_manifest_allowed": False,
            "provenance": {"partial_analysis_sha256": digest(partial_path),
                           "modular_certificates_sha256": digest(certificates_path),
                           "modular_preflight_sha256": digest(modular_preflight)}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--certificates", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.partial, args.certificates, args.preflight)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "k_star": result["k_star_Psi_64_UPO"]}, sort_keys=True))


if __name__ == "__main__":
    main()
