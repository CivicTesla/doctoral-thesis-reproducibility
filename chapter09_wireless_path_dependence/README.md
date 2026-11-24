# Chapter 09: Wireless path dependence

The release preserves the native ns-3.47 witness source, registered R0 data, and R3/R3B analytical results. Whole-route, final-10-metre, and later filtered populations are not merged.

## Review entry points

- [Native witness](src/ns3_witness/b3p_r0_witness.cc): original ns-3 source
- [R0 manifest](results/r0_registered_2000/registered_world_manifest.csv): 2,000 registered worlds
- [R3 decision band](results/r3/decision_band.csv): five saved sensitivity rows
- [R3B hierarchy](results/r3b/identification_hierarchy.csv): distinct analytical population

## Layout and use

Present evidence directories: `src/`, `experiments/`, `results/`, `verification/`. The `experiments/` directory holds historical configurations or plotting entry points where available. Missing directories mean no suitable material was identified for this V1 package. Historical scripts may assume their original directory layout and dependencies; commands below are limited to checks known to be safe in this package.

Run `python3 verification/check_manifest.py` for copied-file integrity. The stage-three audit checked R0 row counts and independently recalculated the five R3/R3B rows. Do not interpret the historical `environment_historical.md` path as a working local ns-3 installation.

See [provenance](PROVENANCE.md) and [limitations](LIMITATIONS.md). The top-level [manifest](../GITHUB_FILE_MANIFEST.csv) gives a SHA-256 and source alias for every copied file.
