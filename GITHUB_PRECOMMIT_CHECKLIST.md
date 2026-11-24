# GitHub V1 precommit checklist — interrupted-session recovery

> Current release gate (2026-11-24): the recovery observations below are historical. GitHub CLI authenticates as `CivicTesla`; the existing remote is Private with ADMIN permission, and repository-local noreply identity is configured. The selected manifest has 1,940 artifacts after withholding two Massive PNGs and one object-type OBS `.npy`. The proposed Git set has 1,975 files. Read-only Git HTTPS access succeeded using the `gh` credential helper for that command. The author confirmed private-storage authorization for the selected materials and authorized the initial commit and push. See [release readiness](GITHUB_RELEASE_READINESS.md) and [sharing exclusions](SHARING_EXCLUSIONS.md).

Checked 2026-11-19 (Asia/Shanghai). This is a local release gate for the **private** GitHub repository. It records file integrity and upload safety, not a new scientific audit or permission to commit or push.

## Recovery and repository state

- [x] Repository root is this release directory; `.git` is a directory inside it, with no `core.worktree` override. It is separate from the original research work tree.
- [x] Branch `main`; zero commits; no tracked or staged files. All release files are currently untracked. No lock or typical editor/recovery file, nested `.git`, symlink, cache directory, or related active Git/Codex process was found.
- [x] Core documents and all Chapter 03–09 directories exist. This checklist did not exist at the start of recovery and was created during this check.
- [x] Manifest has 1,943 unique copied files: 146 source, 1,651 result/certificate, 117 verification, 29 experiment/configuration. All 1,943 exist and match both recorded byte size and SHA-256. Before this checklist there were 1,977 files outside `.git`: 1,943 registered copies and 34 packaging/documentation files. The checklist adds one documentation file.
- [x] No missing, changed, or moved *registered* file was detected. With zero commits and no earlier filesystem snapshot for the packaging files, the exact changes made during the interrupted session cannot be reconstructed from Git. The earlier baseline describes the 34 unregistered packaging files but does not provide their prior hashes.
- [x] `origin` was absent and is now `https://github.com/CivicTesla/doctoral-thesis-reproducibility.git` for fetch and push. The local branch was already `main`.
- [ ] At recovery time no repository-local identity was configured; an inherited non-noreply identity was present. Current status: repository-local GitHub noreply identity is configured and takes precedence.
- [ ] HTTPS authentication is not ready: a non-interactive, read-only `git ls-remote origin HEAD` could not obtain a GitHub username. `gh` is not installed. Verify access and the remote repository's Private setting after configuring authentication, without exposing credentials.

## File-group decisions

`CLEAR` means a local technical check found no current blocker for that group; it does **not** override the pending rights and privacy decisions below. `NEEDS_APPROVAL` must not be included in a first push until the named person reviews it. `PRIVATE_EXCLUDE` stays outside the proposed commit and push; files already absent from V1 remain absent.

| File or group | Decision | Reason / required action |
|---|---|---|
| Repository packaging prose, manifest, inventories, chapter `README.md` / `PROVENANCE.md` / `LIMITATIONS.md`, `docs/`, and `verification/check_manifest.py` | CLEAR | Present; text scan found no email, internal absolute path, or common credential marker. Readability and authorship assertions still require author review before publication. |
| Registered copied source, tests, verifiers, experiments, results, certificates, and derived data across Chapters 03–09 | NEEDS_APPROVAL | Confirm author, collaborator, institution, dataset, and publisher rights for storage on a private GitHub account and the intended supervisor/examiner access. Exact copied bytes are intact, but hashes do not grant sharing rights. |
| Massive Figure 6 PNG files (`chapter05_massive_and_future_intent/results/massive/*.png`) | NEEDS_APPROVAL | Verify figure/publisher/coauthor redistribution rights and visually inspect pixels; PNG metadata contains only a Matplotlib software tag in the inspected files. |
| All 1,481 `.npy` files, including historical OBS arrays and maps | NEEDS_APPROVAL | Confirm no confidential third-party data, personal information, or unexpected binary metadata. Header inspection found 1,480 numeric arrays and one object-typed array. A text/format scan cannot establish binary privacy clearance. |
| `chapter06_digital_twin_obs/results/p2p_03_raw/03P2P!!!!!!!!!!!!!!!!!!!!!!!.npy` | PRIVATE_EXCLUDE | Object-typed NumPy array, 35,891 bytes, potentially pickle serialized. Keep out of the initial upload until a trusted, non-executing content review and explicit release decision; do not load it with `allow_pickle=True` just to inspect it. Current status: it remains on disk but is Git ignored and absent from the selected release manifest. |
| Nine Ch07 `results/e40/*/by_episode.csv` files | NEEDS_APPROVAL | Header has `display_name`; verify that values are experiment labels rather than people. Large CSVs also merit manual privacy review. |
| Full dissertation PDF, final Massive paper PDF, unpublished manuscripts, double-blind submission bundles, Ch09 R0 run log, original `.git` trees, caches, and credentials | PRIVATE_EXCLUDE | Already withheld from V1 as documented in `SHARING_EXCLUSIONS.md` and `docs/BUILD_EXCLUSIONS.csv`. Do not add them to this repository. |

## Upload safety gates

- [x] Common textual private-key/token patterns, personal email strings, and `/home/` or `/Users/` paths were not found in the included text files by heuristic scans. This is not proof that no secret exists. No actual credential was opened or printed.
- [x] No nested Git repository, symlink, typical cache/recovery file, or file over GitHub's 100 MB single-file limit was found. Largest file is 60,175,128 bytes. The package is approximately 280 MB on disk; confirm repository/storage limits and upload plan before a push.
- [ ] Obtain explicit confirmation of coauthor, publisher, and third-party data permissions for private GitHub storage and each proposed reviewer. Private visibility does not substitute for redistribution rights.
- [ ] Manually review binary content, figures, and CSV label fields. Resolve the object-typed OBS array before staging anything; keep it excluded unless an explicit safe release decision is recorded.
- [ ] Confirm no unpublished or double-blind material is embedded in the selected code, result rows, or figures, even though the known full bundles were withheld.
- [ ] Obtain the exact GitHub noreply email; configure `user.name` and `user.email` **locally**. Verify GitHub authentication and Private repository access without revealing tokens or changing global Git settings.
- [ ] After approvals and any deliberate exclusions, update the manifest/package as needed, rerun all file and hash checks, and review the exact proposed initial staged diff before authorizing a first commit or push.

## Research interpretation to preserve

The original Massive and OBS research code was written by the author without AI code generation. Later AI-assisted verification is separate from original research code. Other development origins remain `TO_BE_CONFIRMED` unless supported by direct evidence; see `AI_USE_REGISTER.csv`. Keep all Chapter 03–09 limitations and the four-stage audit boundaries stated in `docs/AUDIT_SUMMARY.md`. File-level and finite checks do not establish complete reproduction of the dissertation.

**Historical recovery decision (2026-11-19): `BLOCKED_PENDING_USER_CONFIRMATION`.** That decision was superseded by the author's 2026-11-24 private-storage authorization and the current gate at the top of this file. No collaborator invitation or repository visibility change was authorized.
