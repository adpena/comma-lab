"""Tests for the #254 self-protect coverage gate
``tac.preflight.check_heavy_witness_trainers_call_admission_guard``.

The gate scans ``experiments/train_*witness*realized_through_R*.py`` AND (since the 2026-07-06
memory-safety review) ``experiments/train_substrate_*.py``. The 2026-08-07 rr8-F3 class-fix
extends it to top-level ``experiments/*.py`` and ``src/tac/**/train*.py`` argparse surfaces that
expose a train mode and GPU/Metal/CUDA/MPS device choice. It warns for any heavy trainer that does
NOT call ``assert_governed_admission`` (the P0 machine-crash gate), so a future un-wired heavy
entrypoint cannot silently bypass the memory governor. Known historical families remain warn-only
VISIBLE BACKLOG: a tracked queue, not a carve-out."""
from __future__ import annotations

import pytest

from tac.preflight import (
    PreflightError,
)
from tac.preflight import (
    check_heavy_witness_trainers_call_admission_guard as _gate,
)

_WITNESS = "train_levelset_witness_realized_through_R_mlx.py"
_BODY_NO_GUARD = (
    "import argparse\n"
    "def main(argv=None):\n"
    "    ap = argparse.ArgumentParser()\n"
    "    args = ap.parse_args(argv)\n"
    "    return 0\n"
)
_BODY_WITH_GUARD = (
    "import argparse\n"
    "def main(argv=None):\n"
    "    ap = argparse.ArgumentParser()\n"
    "    args = ap.parse_args(argv)\n"
    "    from tac.admission_guard import assert_governed_admission\n"
    "    assert_governed_admission('train_levelset_witness_realized_through_R_mlx')\n"
    "    return 0\n"
)


def _mk(tmp_path, name, body):
    _mk_rel(tmp_path, f"experiments/{name}", body)


def _mk_rel(tmp_path, rel, body):
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_positive_missing_guard_flags(tmp_path):
    _mk(tmp_path, _WITNESS, _BODY_NO_GUARD)
    v = _gate(repo_root=tmp_path, strict=False, verbose=False)
    assert len(v) == 1
    assert "assert_governed_admission" in v[0]
    assert _WITNESS in v[0]


def test_negative_guard_present_clears(tmp_path):
    _mk(tmp_path, _WITNESS, _BODY_WITH_GUARD)
    v = _gate(repo_root=tmp_path, strict=False, verbose=False)
    assert v == []


def test_waiver_respected(tmp_path):
    body = _BODY_NO_GUARD + "# ADMISSION_GUARD_WAIVED: light entrypoint, never allocates heavy\n"
    _mk(tmp_path, _WITNESS, body)
    v = _gate(repo_root=tmp_path, strict=False, verbose=False)
    assert v == []


def test_substrate_trainer_now_in_scope_flagged(tmp_path):
    # 2026-07-06 review: train_substrate_*.py is IN scope — an un-guarded substrate trainer is a
    # visible warn (the backlog), no longer a silent carve-out.
    _mk(tmp_path, "train_substrate_foo.py", _BODY_NO_GUARD)
    v = _gate(repo_root=tmp_path, strict=False, verbose=False)
    assert len(v) == 1
    assert "train_substrate_foo.py" in v[0]
    assert "assert_governed_admission" in v[0]


def test_substrate_trainer_waiver_respected(tmp_path):
    body = _BODY_NO_GUARD + "# ADMISSION_GUARD_WAIVED: light entrypoint, never allocates heavy\n"
    _mk(tmp_path, "train_substrate_foo.py", body)
    v = _gate(repo_root=tmp_path, strict=False, verbose=False)
    assert v == []


def test_substrate_trainer_with_guard_clears(tmp_path):
    _mk(tmp_path, "train_substrate_foo.py", _BODY_WITH_GUARD)
    v = _gate(repo_root=tmp_path, strict=False, verbose=False)
    assert v == []


def test_experiment_argparse_gpu_train_mode_positive_control(tmp_path):
    body = (
        "import argparse\n"
        "def main(argv=None):\n"
        "    ap = argparse.ArgumentParser()\n"
        "    ap.add_argument('--mode', choices=['probe', 'mlx-train'])\n"
        "    ap.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')\n"
        "    args = ap.parse_args(argv)\n"
        "    return args\n"
    )
    _mk(tmp_path, "ddm_future_lifted_entrypoint.py", body)
    v = _gate(repo_root=tmp_path, strict=False, verbose=False)
    assert len(v) == 1
    assert "ddm_future_lifted_entrypoint.py" in v[0]
    assert "assert_governed_admission" in v[0]


def test_src_train_argparse_gpu_device_positive_control(tmp_path):
    body = (
        "import argparse\n"
        "def main(argv=None):\n"
        "    ap = argparse.ArgumentParser()\n"
        "    ap.add_argument('--device', default='cuda')\n"
        "    args = ap.parse_args(argv)\n"
        "    return args\n"
    )
    _mk_rel(tmp_path, "src/tac/example/train_pose_future.py", body)
    v = _gate(repo_root=tmp_path, strict=False, verbose=False)
    assert len(v) == 1
    assert "src/tac/example/train_pose_future.py" in v[0]


def test_argparse_gpu_train_waiver_respected(tmp_path):
    body = (
        "import argparse\n"
        "def main(argv=None):\n"
        "    ap = argparse.ArgumentParser()\n"
        "    ap.add_argument('--mode', choices=['probe', 'mlx-train'])\n"
        "    ap.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')\n"
        "    args = ap.parse_args(argv)\n"
        "    return args  # ADMISSION_GUARD_WAIVED: bounded metadata-only emitter\n"
    )
    _mk(tmp_path, "ddm_future_light_emitter.py", body)
    v = _gate(repo_root=tmp_path, strict=False, verbose=False)
    assert v == []


def test_strict_raises_on_missing(tmp_path):
    _mk(tmp_path, _WITNESS, _BODY_NO_GUARD)
    with pytest.raises(PreflightError):
        _gate(repo_root=tmp_path, strict=True, verbose=False)


def test_real_repo_lifted_target_entrypoints_wired():
    v = _gate(strict=False, verbose=False)
    joined = "\n".join(v)
    assert "experiments/ddm_mx1_pr130_semantic_renderer.py" not in joined
    assert "src/tac/pr130_lift/pose/train_pose_carrier_full_resumable.py" not in joined


def test_ah1_top_tier_backlog_slice_wired():
    v = _gate(strict=False, verbose=False)
    joined = "\n".join(v)
    for rel in (
        "experiments/train_anr_token_renderer.py",
        "experiments/train_balle_hyperprior.py",
        "experiments/train_blocknerv_as_renderer.py",
        "experiments/train_categorical_renderer.py",
        "experiments/train_charm_50k_toy_substrate.py",
    ):
        assert rel not in joined
