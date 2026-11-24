# Chapter 07: Valid-Future Semantic Admission

The chapter's E40 results are preserved as historical outputs. Main, channel, pilot, and related episode-level CSVs remain distinct. Later E41 repairs are not used as E40 evidence.

## Review entry points

- [E40 main](results/e40/main/by_episode.csv): historical main episodes
- [E40 channel](results/e40/channel/by_episode.csv): historical channel episodes
- [E40 run program](src/original_project/scripts/run/run_e40_normalized_dc_diversity_fix.py): original E40 study entry point with its dependency layout
- [E40 final audit](results/e40/final_analysis/e40_final_claim_check.md): historical claim review

## Layout and use

Present evidence directories: `src/`, `results/`. The `experiments/` directory holds historical configurations or plotting entry points where available. Missing directories mean no suitable material was identified for this V1 package. Historical scripts may assume their original directory layout and dependencies; commands below are limited to checks known to be safe in this package.

The stage-three audit independently aggregated the frozen CSVs using Python standard-library CSV and mean calculations. Do not rerun the full simulation as part of the quick guide. Use `python3 verification/check_manifest.py` for the release-file hash check.

See [provenance](PROVENANCE.md) and [limitations](LIMITATIONS.md). The top-level [manifest](../GITHUB_FILE_MANIFEST.csv) gives a SHA-256 and source alias for every copied file.
