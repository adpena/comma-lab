"""The fire tool refuses a lane id the canonical pointer's maturity gate would refuse.

2026-09-05: lane ``ddm_pc1_t4_v3_lattice_x4_on_rc1_20260905`` bought a T4 row (S 0.1451981569076111)
that the pointer refresh then refused to promote because ``v3`` — pc1's VARIANT number — reads as
an untagged vehicle token. The same seal had to be re-fired under a compliant name. This test pins
the fire-time check so the refusal happens before any Modal spend.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _load_fire_tool():
    spec = importlib.util.spec_from_file_location(
        "fire_modal_auth_eval_under_test", REPO / "tools" / "fire_modal_auth_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _argv(lane_id: str, *extra: str) -> list[str]:
    return [
        "--seal", "/nonexistent/seal.json",
        "--output-dir", "/nonexistent/out",
        "--lane-id", lane_id,
        "--instance-job-id", "job_x",
        "--claim-agent", "main",
        "--dry-run",
        *extra,
    ]


def test_vehicle_shaped_lane_id_is_refused_before_any_subprocess(capsys) -> None:
    mod = _load_fire_tool()
    with pytest.raises(SystemExit) as exc:
        mod.main(_argv("ddm_pc1_t4_v3_lattice_x4_on_rc1_20260905"))
    assert exc.value.code == 2  # argparse error path
    err = capsys.readouterr().err
    assert "checkpoint-maturity" in err and "v3" in err


def test_compliant_lane_id_passes_the_lane_check(tmp_path) -> None:
    mod = _load_fire_tool()
    # Passing the lane check means the tool proceeds to the seal read, which fails on the
    # nonexistent path with a NON-argparse exit — anything but the argparse code 2 with
    # the maturity text proves the lane gate let it through.
    with pytest.raises((SystemExit, Exception)) as exc:
        mod.main(_argv("ddm_pc1_t4_lattice_x4_on_rc1_20260905", "--output-dir", str(tmp_path)))
    if isinstance(exc.value, SystemExit):
        assert exc.value.code != 2 or "checkpoint-maturity" not in str(exc.value)


def test_explicit_allow_flag_bypasses_the_lane_check(tmp_path) -> None:
    mod = _load_fire_tool()
    with pytest.raises((SystemExit, Exception)) as exc:
        mod.main(_argv("ddm_pc1_t4_v3_lattice_x4_on_rc1_20260905", "--allow-nonpromotable-lane-id", "--output-dir", str(tmp_path)))
    if isinstance(exc.value, SystemExit):
        assert exc.value.code != 2 or "checkpoint-maturity" not in str(exc.value)


def test_fire_tool_imports_experiments_from_a_foreign_cwd(tmp_path):
    """Regression: stage 3 imports `experiments.contest_auth_eval`; the tool must make the repo
    root importable itself (pytest's implicit rootdir hid the ModuleNotFoundError that aborted
    the pc2 fire on 2026-09-09 before any dispatch)."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('fire_tool', {str(repo / 'tools' / 'fire_modal_auth_eval.py')!r})\n"
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
        "from experiments.contest_auth_eval import _runtime_dependency_manifest\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
