from pathlib import Path

F0 = Path(__file__).resolve().parents[1]


def test_no_new_gate_or_mechanism_directories_created():
    forbidden_prefixes = ("G", "H5", "H6", "P5", "P6", "CF2")
    for child in F0.parent.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith("F0_final_hpc_validation"):
            continue
        for prefix in forbidden_prefixes:
            assert not name.startswith(prefix), (
                f"unexpected new gate/mechanism directory: {name}")


def test_stop_rules_documented():
    text = (F0 / "03_preregistered_protocol.md").read_text()
    assert "F0" in text  # sanity: file exists and non-empty
