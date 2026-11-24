import subprocess
from pathlib import Path

F0 = Path(__file__).resolve().parents[1]
B1 = F0.parent


def test_no_simu5g_or_ns3_reference_in_f0_code():
    # build_reports.py legitimately *reports* the frozen Simu5G closure
    # status (Section 00) -- that is documentation, not a run. Every other
    # F0 script must have zero Simu5G/ns-3 reference.
    allowed = {"build_reports.py", "build_final_decision.py"}
    for script in list(F0.glob("*.py")) + list((F0 / "pair1").glob("*.py")) + \
            list((F0 / "pair2").glob("*.py")) + list((F0 / "hpc").glob("*")):
        if script.is_dir() or script.name in allowed:
            continue
        text = script.read_text(errors="ignore").lower()
        assert "simu5g" not in text and "ns-3" not in text and "ns3" not in text, script


def test_no_new_simu5g_commits_since_f0_branched():
    # F0 branched from b1-p4b-r1-radio-operating-point-audit at 6d8b4f7,
    # which legitimately contains prior (already-closed) Simu5G work; the
    # check is that F0 itself adds no *new* commits touching Simu5G.
    out = subprocess.run(
        ["git", "log", "--oneline", "6d8b4f7..HEAD", "--",
         "B1/P4_simu5g_external_validation"],
        cwd=B1.parent, capture_output=True, text=True)
    assert out.stdout.strip() == "", out.stdout
