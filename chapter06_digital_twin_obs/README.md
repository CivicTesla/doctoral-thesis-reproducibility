# Chapter 06: Digital Twin OBS

The release preserves the author's original OBS 03 and 04 program snapshots and the P2P plot inputs. Their stopping rules and movement/communication logic differ; the 03 CSVs and historical arrays are presented without rewriting them.

## Review entry points

- [03 P2P code](src/obs_historical_03/003-ServerP2PMonteCarloSim.py): strictly more than 90 percent reached in active condition
- [04 P2P code](src/obs_historical_04/003-ServerP2PMonteCarloSim.py): first agent reached in active condition
- [P2P CSV](results/plot_inputs/P2PLoopCovMoves.csv): historical 10-group plotting input
- [Later replay](verification/later_obs_p2p_300_replay.py): later Codex-assisted program; never historical evidence

## Layout and use

Present evidence directories: `src/`, `experiments/`, `results/`, `verification/`. The `experiments/` directory holds historical configurations or plotting entry points where available. Missing directories mean no suitable material was identified for this V1 package. Historical scripts may assume their original directory layout and dependencies; commands below are limited to checks known to be safe in this package.

The original scripts use project-relative files and random state without an archived per-run seed lock. No 300-trial run is requested for V1. Use the top-level manifest verifier to check copied bytes; consult `docs/AUDIT_SUMMARY.md` for the previous numerical trace audit.

See [provenance](PROVENANCE.md) and [limitations](LIMITATIONS.md). The top-level [manifest](../GITHUB_FILE_MANIFEST.csv) gives a SHA-256 and source alias for every copied file.
