# Chapter 04: Subtree reconstruction

The chapter studies intersection moments, trade bounds, and the seven nine-vertex shapes with identification order four. The release contains the finite checker and supporting proof and witness records.

## Review entry points

- [Finite verifier](tests/verify_c2.py): nine-vertex classification checker
- [Seven hosts](results/R3_SEVEN_HOSTS.md): witness summary
- [Human proof](results/R4_NINE_VERTEX_HUMAN_PROOF.md): general argument for expert review

## Layout and use

Present evidence directories: `src/`, `results/`, `tests/`. The `experiments/` directory holds historical configurations or plotting entry points where available. Missing directories mean no suitable material was identified for this V1 package. Historical scripts may assume their original directory layout and dependencies; commands below are limited to checks known to be safe in this package.

The stage-three audit ran `python3 verify_c2.py --quick` in an isolated directory and observed 57 PASS, 0 FAIL. The copied verifier is historical research material, but its imports may require the original C2_REAL layout; use the recorded result as an audit observation until a release-local invocation has been checked.

See [provenance](PROVENANCE.md) and [limitations](LIMITATIONS.md). The top-level [manifest](../GITHUB_FILE_MANIFEST.csv) gives a SHA-256 and source alias for every copied file.
