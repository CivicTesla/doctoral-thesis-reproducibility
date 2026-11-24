# GitHub private repository V1 readiness — updated 2026-11-24

## Package state

The proposed initial upload contains 1,975 files. Its exact byte count should be recalculated immediately before staging. The release manifest lists 1,940 selected research artifacts: 146 source, 1,648 result or certificate, 117 verification, and 29 experiment/configuration files. Three locally retained artifacts are excluded from both the Git upload set and release manifest. Counts include duplicate frozen inputs intentionally preserved inside the self-contained Ch08 verifier bundle. Every selected research artifact has a manifest SHA-256 and source alias. No scientific result was generated for this release.

| Chapter | Supervisor review readiness | Qualification |
|---|---|---|
| 03 | Ready for qualified review | Frozen certificate and inputs available; full enumeration not rerun. |
| 04 | Ready for qualified review | 57/57 finite check recorded; human proof review remains. |
| 05 | Ready for qualified review | Figure 6 identity recorded in the prior audit; panel images withheld pending rights clearance; execution chain incomplete. |
| 06 | Ready with prominent limitation | Original code and selected data available; one object-type array withheld; 100-agent block and stopping rule unresolved. |
| 07 | Ready with prominent limitation | Original E40 data available; five printed channel values differ. |
| 08 | Ready with prominent limitation | 37/37 frozen inputs locally checkable; S1R/Per Stream original audits absent. |
| 09 | Ready with prominent limitation | R0 and R3/R3B evidence available; original ns-3.47 build absent. |

## Security and integrity gates

- The assembly screened selected text for likely private keys, API tokens, personal email addresses, and internal absolute paths. One run log with an internal path/contact was excluded. This heuristic scan does not replace human review of binary files or rights.
- File names in the release are ASCII. New repository documentation is English. Original source code retains historical comments without rewriting.
- No selected file exceeds GitHub's 100 MB hard per-file limit. The largest selected file is 60,175,128 bytes. The proposed upload is approximately 285 MB before Git history. Git LFS is not configured.
- Nested `.git`, symlinks, cache directories, temporary simulator builds, and unused paper bundles were excluded.
- The independent package does not rely on the `codexlab` parent Git tree. Historical commands that assume the original layout are described as such, not advertised as working quick checks.

## Connection and release decision

GitHub CLI is authenticated as `CivicTesla`. The existing remote `CivicTesla/doctoral-thesis-reproducibility` is Private and the account has `ADMIN` permission. The repository-local identity is `Siyu Yuan` / `23435949+CivicTesla@users.noreply.github.com`; `origin` uses the expected HTTPS URL. Read-only Git HTTPS access succeeded using the existing `gh` credential helper for that command, without changing global Git configuration. The remote has no refs before the initial push.

**Decision: AUTHORIZED_FOR_INITIAL_COMMIT_AND_PUSH on 2026-11-24.** The author confirmed that the selected Chapters 3–9 research materials may be stored in this Private GitHub repository, that no known collaborator, institutional, research-group, or project restriction prohibits this storage, and that the selected materials contain no confidential information. The original Massive Twinning and OBS research code was manually written by the author without AI code generation; other projects include author-written and AI/Codex-assisted development. The two Massive Figure 6 PNG files, one OBS object-type `.npy`, dissertation PDF, unpublished submission materials, and sensitive run log remain outside the selected set. This authorization does not extend to public release or reviewer invitations.
