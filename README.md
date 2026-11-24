# Decision Relevant Information under Resource Constraints: Structure, Communication, and Counterfactual Evaluation

Private dissertation reproducibility review package, V1. The reference is the dissertation submitted on 2026-10-12 by Siyu Yuan. This repository links Chapters 3–9 to selected original code, configurations, saved results, and finite verification records. It is intended for supervisor and authorized examiner review.

## Research overview

The dissertation studies which structural, communicated, and counterfactual information can support decisions under finite resources. The package keeps original experiments separate from later audits and repairs. A file hash establishes provenance of a copied artifact; it does not, by itself, establish that an experiment has been independently reproduced.

## Chapter-to-code map

| Chapter | Topic | Primary evidence | Review status |
|---|---|---|---|
| [03](chapter03_polar_prefix_cosets/README.md) | Polar prefix cosets | N=32/N=64 frozen certificates | Table 3.1 frozen certificate was rebuilt from eight matched inputs; full enumeration not rerun. |
| [04](chapter04_subtree_reconstruction/README.md) | Subtree reconstruction | Nine-vertex verifier and proof records | Finite checker: 57/57 passed; general proof awaits expert review. |
| [05](chapter05_massive_and_future_intent/README.md) | Massive Twinning; later Future Intent | Figure 6 input arrays and source records; event pipeline | Prior audit matched the panels pixel for pixel; panel files await rights clearance, and the precise historical run chain remains incomplete. |
| [06](chapter06_digital_twin_obs/README.md) | Digital Twin OBS | Historical 03/04 code and selected P2P arrays | One object-type historical array is withheld for safety; most CSV values were traced, while the 100-agent block and stopping semantics remain limited. |
| [07](chapter07_valid_future_admission/README.md) | Valid-Future Semantic Admission | E40 source and original results | 15/20 printed values match; five differ; three file checks passed. |
| [08](chapter08_intervention_counterfactual/README.md) | Intervention counterfactuals | CF0, CF1, F0 frozen evidence | 37/37 frozen hashes matched; S1R and Per Stream raw audits missing. |
| [09](chapter09_wireless_path_dependence/README.md) | Wireless path dependence | ns-3.47 R0 and R3/R3B | Saved R0 counts and analytical rows checked; original ns-3 build unavailable. |

## Repository structure

Each chapter has its own `README.md`, `PROVENANCE.md`, and `LIMITATIONS.md`. Evidence appears under `src/`, `experiments/`, `results/`, `tests/`, or `verification/` when a suitable artifact exists. The [file manifest](GITHUB_FILE_MANIFEST.csv) records source aliases, byte counts, and SHA-256 for copied files. [Audit summary](docs/AUDIT_SUMMARY.md) and [claim index](docs/CLAIM_EVIDENCE_INDEX.csv) map the main dissertation results to these materials. [Sharing exclusions](SHARING_EXCLUSIONS.md) lists files withheld from V1.

## Quick verification

From the repository root:

```sh
python3 verification/check_manifest.py
python3 chapter08_intervention_counterfactual/verification/frozen_b1/F0_final_hpc_validation/hpc/12_verify_all_hashes.py
```

The first command checks each selected upload artifact against the release manifest. The second checks F0's 37 frozen inputs. Neither command runs a large simulation or changes historical data. The prior Ch04 57/57 finite check is documented in the audit summary; its release-local dependency layout has not been tested as a quick command.

## Reproducibility status and known limitations

The package supports file-level review and several finite or numerical checks, not a claim of complete end-to-end reproduction. The highest-priority open issues are OBS's untraced 100-agent fourth block and stopping-rule ambiguity; Ch08's absent S1R and Per Stream raw audit records; Massive Figure 6's missing execution log; Ch07's five E40 channel discrepancies; and the missing original Ch09 ns-3 build. Every chapter's `LIMITATIONS.md` gives its own scope. [Audit summary](docs/AUDIT_SUMMARY.md) explains the status of each key result.

## Code provenance and AI use

The author reports that the original Massive Twinning and OBS research code was written manually without AI code generation. A later Codex-assisted OBS replay is clearly labeled as later verification code. Other module-level development histories are recorded as `TO_BE_CONFIRMED` in [AI_USE_REGISTER.csv](AI_USE_REGISTER.csv); no chapter is categorically labeled AI-generated. See each chapter's `PROVENANCE.md` and the file manifest for origin details. Historical source comments are retained verbatim to preserve byte identity; all new repository explanations are in English.

## Access and confidentiality

This is a private review package. Access should be restricted to explicitly authorized reviewers. Redistribution, public release, or collaborator invitation requires a separate rights and privacy decision. No open-source license is granted or implied. The full submitted dissertation, unpublished manuscript bundles, two Figure 6 panels awaiting rights clearance, and one unsafe object-type OBS array are outside the selected upload set. The author confirmed private GitHub storage of the selected Chapters 3–9 research materials on 2026-11-24; see [release readiness](GITHUB_RELEASE_READINESS.md) and [sharing exclusions](SHARING_EXCLUSIONS.md).
