# V1 sharing exclusions and review queue

The following items are withheld from the proposed initial Git upload or are not represented as runnable evidence. The original local files remain unchanged. The earlier assembly omissions are logged in [docs/BUILD_EXCLUSIONS.csv](docs/BUILD_EXCLUSIONS.csv).

| Item | Decision and reason |
|---|---|
| Full 2026-10-12 dissertation PDF | Withheld from the code repository; distributed through the authorized institutional channel. Its reference SHA-256 is `57c3f8cfb65043a2b8b1dfdae4b4c739e2e2cec65b3d569849cbb3a4864bd72e`. |
| Final Massive paper PDF | Withheld pending publisher and coauthor redistribution review. |
| `chapter05_massive_and_future_intent/results/massive/01-GridonRadius.png` and `02-GridonRandomNeighbor.png` | Locally retained and Git ignored pending publisher and coauthor permission to store the published panels on private GitHub. The prior pixel-match finding remains documented; the images are absent from the proposed upload. |
| `chapter06_digital_twin_obs/results/p2p_03_raw/03P2P!!!!!!!!!!!!!!!!!!!!!!!.npy` | Locally retained and Git ignored because its NumPy header declares object type and its content may use pickle serialization. Do not load it with `allow_pickle=True` for release screening. |
| Unpublished full manuscripts, double-blind review files, and submission bundles | Outside V1; no complete submission package was intentionally copied. |
| C1 Journal R3–R6, E41, and unrelated later studies | Excluded because they are not 2026-10-12 thesis evidence. Their absence does not erase their separate research history. |
| OBS `MapMatrix.npy` in the two checked 03/04 roots | Named in the assembly selection but absent there; the available PER maps were retained. A later audit may determine whether other map data are needed for execution. |
| Five historical Ch08 Markdown audit records with non-English content | Withheld to keep repository explanations in English; source hashes and their omission are documented. The raw CF0/CF1/F0 data and frozen bundle remain included. |
| Ch09 R0 run log | Withheld because it contains an internal absolute path or personal contact. Its data, manifest, and provenance audit remain included. |
| Original `.git` trees, symlinks, caches, notebooks, temporary outputs, credentials | Not copied. The release has its own Git repository only after initialization. |

## Author confirmation for the initial private push — 2026-11-24

The author explicitly authorized storage of the selected Chapters 3–9 research materials in the specified Private GitHub repository. The author reported no confidential information in the selected research materials and no known collaborator, institutional, research-group, or project prohibition on this private storage. This confirmation covers the selected original research code and project-generated evidence, including author-written and AI/Codex-assisted development. The original Massive Twinning and OBS research code was manually written without AI code generation. The exclusions above remain in force; no permission for public release or collaborator invitation is inferred.

The nine Ch07 `by_episode.csv` files use algorithm labels in `display_name`, not personal names. Historical source comments remain unchanged for byte-level provenance.
