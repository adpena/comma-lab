from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_script_module(script_rel: str):
    path = REPO_ROOT / script_rel
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_contest_submission_packet_help_and_real_module() -> None:
    script = REPO_ROOT / "scripts/build_contest_submission_packet.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "--artifact-dir" in result.stdout
    module = _load_script_module("scripts/build_contest_submission_packet.py")
    assert getattr(module, "PacketError")
    assert callable(getattr(module, "build_packet"))
    assert "__recovery_status__" not in vars(module)


def test_q_faithful_snapshot_loop_help_and_runtime_sha_contract() -> None:
    script = REPO_ROOT / "scripts/q_faithful_snapshot_loop.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "--checkpoint-dir" in result.stdout
    module = _load_script_module("scripts/q_faithful_snapshot_loop.py")
    shas = module.source_runtime_shas(REPO_ROOT)
    assert shas["scripts/q_faithful_snapshot_loop.py"]
    assert shas["submissions/robust_current/inflate_renderer.py"]
    assert "__recovery_status__" not in vars(module)
