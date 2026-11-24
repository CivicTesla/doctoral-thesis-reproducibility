# Chapter 08: Intervention counterfactual audit

The evidence chain is CF0 to CF1 to F0. The package preserves their selected source, configuration, raw tables, frozen manifests, and tests as separate branches. A self-contained copy of F0's 37 frozen input artifacts supports a release-local integrity check.

## Review entry points

- [CF0 branch hashes](results/cf0/07_branch_hash_validation.csv): saved paired branch checks
- [CF1 model manifest](results/cf1/02_frozen_model_manifest.csv): frozen selection and input hashes
- [F0 frozen manifest](verification/frozen_b1/F0_final_hpc_validation/01_frozen_artifact_manifest.csv): 37 input artifact hashes
- [F0 verifier](verification/frozen_b1/F0_final_hpc_validation/hpc/12_verify_all_hashes.py): release-local hash check

## Layout and use

Present evidence directories: `src/`, `experiments/`, `results/`, `tests/`, `verification/`. The `experiments/` directory holds historical configurations or plotting entry points where available. Missing directories mean no suitable material was identified for this V1 package. Historical scripts may assume their original directory layout and dependencies; commands below are limited to checks known to be safe in this package.

From the repository root, run `python3 chapter08_intervention_counterfactual/verification/frozen_b1/F0_final_hpc_validation/hpc/12_verify_all_hashes.py`. The staged copy returned `all 37 frozen artifacts verified unchanged`. This is an integrity check, not a full simulation rerun.

See [provenance](PROVENANCE.md) and [limitations](LIMITATIONS.md). The top-level [manifest](../GITHUB_FILE_MANIFEST.csv) gives a SHA-256 and source alias for every copied file.
