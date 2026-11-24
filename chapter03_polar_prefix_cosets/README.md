# Chapter 03: Polar prefix cosets

The dissertation develops finite structural identification order, exact N=32/N=64 certificates, and a scoped N=128 boundary. The primary archived evidence here is the frozen N=64 theorem object and its inputs; later C1 journal R3–R6 work is outside this release.

## Review entry points

- [Frozen theorem certificate](results/n64_theorem_freeze.json): Table 3.1 values and input hashes
- [Freeze builder](src/build_n64_theorem_freeze.py): original certificate construction code

## Layout and use

Present evidence directories: `src/`, `results/`, `tests/`. The `experiments/` directory holds historical configurations or plotting entry points where available. Missing directories mean no suitable material was identified for this V1 package. Historical scripts may assume their original directory layout and dependencies; commands below are limited to checks known to be safe in this package.

For a lightweight integrity check, run `python3 verification/check_manifest.py` from the repository root. The stage-three audit rebuilt the frozen theorem object from eight hash-matched inputs; this release does not claim to repeat the full enumeration.

See [provenance](PROVENANCE.md) and [limitations](LIMITATIONS.md). The top-level [manifest](../GITHUB_FILE_MANIFEST.csv) gives a SHA-256 and source alias for every copied file.
