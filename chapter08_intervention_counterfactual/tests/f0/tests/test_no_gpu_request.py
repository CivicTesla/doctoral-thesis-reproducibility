from pathlib import Path

HPC = Path(__file__).resolve().parents[1] / "hpc"


def test_no_gpu_flags_in_any_hpc_script():
    for script in HPC.glob("*.sh"):
        text = script.read_text().lower()
        assert "gpu" not in text and "--gres" not in text, script
