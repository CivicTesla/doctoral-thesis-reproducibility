#!/usr/bin/env python3
"""Fail-closed 1172-object N=64 Psi<=2 audit and exact reducer.

The two historical extended-tail timeouts are replaced only by their later
residual records: residual task 1 and the exact four-root-shard assembly for
residual task 0.  No raw attempt is overwritten.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from census_core import canonical_bytes
from psi2_census_worker import item_from_manifest


IDENTITY = ("object", "N", "K", "i", "J", "r", "branch_size")


def checked(path: Path, item: dict, *, root_assembled: bool = False) -> dict:
    raw = path.read_bytes()
    if not root_assembled:
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not sidecar.exists() or hashlib.sha256(raw).hexdigest() != sidecar.read_text().split()[0]:
            raise ValueError(f"invalid raw record checksum: {path}")
    row = json.loads(raw)
    if row.get("status") != "complete" or any(row.get(k) != item[k] for k in IDENTITY):
        raise ValueError(f"incomplete or identity-mismatched record: {path}")
    if row.get("manifest_task_index") != item["task_index"]:
        raise ValueError(f"task-index mismatch: {path}")
    if row.get("orders_computed") != [1, 2]:
        raise ValueError(f"order provenance mismatch: {path}")
    engine = "exact integer C1-style Polar parity row recursion"
    if root_assembled:
        if row.get("engine") != engine + " root-range assembly" or not row.get("root_shard_provenance"):
            raise ValueError("missing assembled-root provenance")
    elif row.get("engine") != engine:
        raise ValueError(f"engine provenance mismatch: {path}")
    canonical = row.get("sig_le2_canonical")
    if not isinstance(canonical, str) or canonical_bytes(json.loads(canonical)).decode("ascii") != canonical:
        raise ValueError(f"noncanonical Psi<=2 data: {path}")
    if hashlib.sha256(canonical.encode("ascii")).hexdigest() != row.get("sig_le2_sha256"):
        raise ValueError(f"canonical digest mismatch: {path}")
    if row.get("signature_bytes") != len(canonical):
        raise ValueError(f"canonical size mismatch: {path}")
    return row


def collect(normal_manifest: Path, normal_dir: Path, tail_manifest: Path, tail_dir: Path,
            residual_manifest: Path, residual_dir: Path, root_final: Path) -> tuple[dict, dict]:
    normal = json.loads(normal_manifest.read_text())["objects"]
    tail = json.loads(tail_manifest.read_text())["objects"]
    if len(normal) != 1168 or len(tail) != 4:
        raise ValueError("unexpected baseline manifest counts")
    expected = {row["object"]: row for row in normal + tail}
    if len(expected) != 1172:
        raise ValueError("baseline manifests do not give 1172 unique objects")
    residual_zero, residual_one = (item_from_manifest(residual_manifest, index) for index in (0, 1))
    tail_by_id = {row["object"]: row for row in tail}
    if residual_zero["object"] not in tail_by_id or residual_one["object"] not in tail_by_id:
        raise ValueError("residual objects are absent from authoritative tail manifest")
    records = {}
    for item in normal:
        row = checked(normal_dir / f"task-{item['task_index']}.json", item)
        records[row["object"]] = row
    # Extended task 0 and 3 completed first; the other two are superseded.
    for index in (0, 3):
        item = tail[index]
        row = checked(tail_dir / f"task-{index}.json", item)
        records[row["object"]] = row
    row = checked(residual_dir / "task-1.json", residual_one)
    records[row["object"]] = row
    row = checked(root_final, residual_zero, root_assembled=True)
    records[row["object"]] = row
    missing = sorted(set(expected) - set(records))
    unexpected = sorted(set(records) - set(expected))
    if missing or unexpected or len(records) != 1172:
        raise ValueError(f"coverage failure: missing={len(missing)} unexpected={len(unexpected)}")
    audit = {"universe": {"expected_nonempty_objects": 1172, "valid_complete_union_count": len(records),
                            "valid_complete_union_missing_count": 0,
                            "complete_exact_psi2_coverage": True},
             "authoritative_sources": {"normal": 1168, "extended_tail_original": 2,
                                        "residual_retry": 1, "root_shard_assembly": 1},
             "rejected_historical_attempts": ["extended-tail task-1 timeout", "extended-tail task-2 timeout",
                                               "residual-two task-0 timeout at 43200 seconds"],
             "integrity": {"timeouts": 0, "missing": 0, "duplicates": 0, "invalid": 0,
                           "checksum_failures": 0, "provenance_failures": 0}}
    return records, audit


def reduce(records: dict, audit: dict, audit_raw: bytes) -> dict:
    indexed: dict[int, dict[str, dict[str, list[dict]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in records.values():
        indexed[row["K"]][row["sig_le2_sha256"]][row["sig_le2_canonical"]].append(row)
    buckets, distinct = [], 0
    for k in sorted(indexed):
        for digest in sorted(indexed[k]):
            for canonical, group in sorted(indexed[k][digest].items()):
                distinct += 1
                if len(group) > 1:
                    buckets.append({"K": k, "sig_le2_sha256": digest,
                                    "exact_canonical_signature": canonical,
                                    "objects": [{key: row[key] for key in ("object", "i", "J", "r", "K", "branch_size")}
                                                for row in sorted(group, key=lambda row: row["object"])]})
    buckets.sort(key=lambda row: (row["K"], row["sig_le2_sha256"], row["objects"][0]["object"]))
    survivors = sum((row["objects"] for row in buckets), [])
    sizes = sorted({len(row["objects"]) for row in buckets})
    ks = sorted({row["K"] for row in buckets})
    return {"N": 64, "coverage": "complete exact Psi<=2 coverage", "object_count": 1172,
            "distinct_sig_le2": distinct, "singleton_sig_le2_classes": distinct - len(buckets),
            "psi_le2_collision_bucket_count": len(buckets), "psi_le2_collision_objects": len(survivors),
            "bucket_size_distribution": {str(size): sum(len(row["objects"]) == size for row in buckets) for size in sizes},
            "collision_K_distribution": {str(k): sum(row["K"] == k for row in buckets) for k in ks},
            "psi_le2_collision_buckets": buckets,
            "exactness": "K-partitioned full canonical Psi<=2 equality; SHA-256 is only an index.",
            "provenance": {"coverage_audit_sha256": hashlib.sha256(audit_raw).hexdigest(),
                           "coverage_audit": "n64_psi2_complete_audit.json"},
            "next_step": "Psi3 is permitted only for objects in the exact collision buckets."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal-manifest", type=Path, required=True)
    parser.add_argument("--normal-records", type=Path, required=True)
    parser.add_argument("--extended-manifest", type=Path, required=True)
    parser.add_argument("--extended-records", type=Path, required=True)
    parser.add_argument("--residual-manifest", type=Path, required=True)
    parser.add_argument("--residual-records", type=Path, required=True)
    parser.add_argument("--root-final-record", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--reducer-output", type=Path, required=True)
    args = parser.parse_args()
    records, audit = collect(args.normal_manifest, args.normal_records, args.extended_manifest,
                             args.extended_records, args.residual_manifest, args.residual_records,
                             args.root_final_record)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    reducer = reduce(records, audit, args.audit_output.read_bytes())
    args.reducer_output.write_text(json.dumps(reducer, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"coverage": "1172/1172", "collision_buckets": reducer["psi_le2_collision_bucket_count"],
                      "collision_objects": reducer["psi_le2_collision_objects"]}, sort_keys=True))


if __name__ == "__main__":
    main()
