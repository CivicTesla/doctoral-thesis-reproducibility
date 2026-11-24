# Chapter 05: Massive Twinning and Future Intent

The historical Massive Twinning program and the later Future Intent event pipeline are separate evidence families. The final Massive paper Figure 6 panels were matched pixel for pixel to two archived images from the 2022 candidate engineering project.

## Review entry points

- [Massive Figure 6 inputs](results/massive/01-SumArrayNeighborRadius.npy): 50-agent radius experiment array
- Massive Figure 6 panels were matched to archived images during the prior audit. The two image files are held locally pending publisher and coauthor sharing clearance; see [sharing exclusions](../SHARING_EXCLUSIONS.md).
- [Future Intent core](src/future_intent/event_coupled_field.py): later event-coupled method

## Layout and use

Present evidence directories: `src/`, `results/`. The `experiments/` directory holds historical configurations or plotting entry points where available. Missing directories mean no suitable material was identified for this V1 package. Historical scripts may assume their original directory layout and dependencies; commands below are limited to checks known to be safe in this package.

The source programs record 50 agents and 300 iterations for the Figure 6 candidate configurations. They are historical files, not ready-to-run release commands. Run `python3 verification/check_manifest.py` for byte integrity only.

See [provenance](PROVENANCE.md) and [limitations](LIMITATIONS.md). The top-level [manifest](../GITHUB_FILE_MANIFEST.csv) gives a SHA-256 and source alias for every selected upload file.
