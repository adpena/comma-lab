"""Tests for tac.substrates._shared.smoke_auth_eval_gate.

Per CLAUDE.md "Auth eval EVERYWHERE" non-negotiable + HNeRV parity lesson L13
+ Catalog #127 (custody validator) + Catalog #167 (smoke-before-full pattern):
the canonical smoke-auth-eval gate must REFUSE auth eval at smoke / explicit
``--skip-auth-eval`` / non-CUDA device / full-CPU advisory paths, and MUST
run + validate the score claim on full-CUDA paths.

Lane: ``lane_canonicalize_inflate_and_smoke_auth_eval_20260514``
Memory: ``feedback_canonicalize_inflate_and_smoke_auth_eval_landed_20260514.md``
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from tac.substrates._shared.smoke_auth_eval_gate import (
    AuthEvalGateError,
    AuthEvalGateRefusal,
    CPU_REFUSAL_REASON,
    FULL_CPU_REFUSAL_REASON,
    SKIP_FLAG_REASON,
    SMOKE_REFUSAL_REASON,
    _detect_device_type,
    format_smoke_skip_banner,
    gate_auth_eval_call,
)


def _make_args(**kwargs) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.smoke = kwargs.pop("smoke", False)
    ns.skip_auth_eval = kwargs.pop("skip_auth_eval", False)
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


def _gate_kwargs(tmp_path: Path, **overrides):
    base = {
        "archive_zip": tmp_path / "archive.zip",
        "inflate_sh": tmp_path / "runtime" / "inflate.sh",
        "upstream_dir": tmp_path / "upstream",
        "output_json": tmp_path / "auth_eval.json",
        "contest_auth_eval_script": tmp_path / "contest_auth_eval.py",
        "substrate_tag": "testsub",
    }
    base.update(overrides)
    return base


class _FakeClaim:
    def __init__(self, score: float, lane_tag: str = "[contest-CUDA]") -> None:
        self.score = score
        self.lane_tag = lane_tag


# ---------------------------------------------------------------------------
# Refusal banners
# ---------------------------------------------------------------------------


def test_format_smoke_skip_banner_includes_substrate_tag() -> None:
    out = format_smoke_skip_banner("ananas")
    assert "[ananas]" in out
    assert "smoke" in out
    assert "training_artifact_v1" in out


# ---------------------------------------------------------------------------
# _detect_device_type
# ---------------------------------------------------------------------------


def test_detect_device_type_from_torch_device_like() -> None:
    class _Dev:
        type = "cuda"

    assert _detect_device_type(_Dev()) == "cuda"


def test_detect_device_type_from_string_cuda() -> None:
    assert _detect_device_type("cuda") == "cuda"
    assert _detect_device_type("cuda:0") == "cuda"


def test_detect_device_type_from_string_cpu() -> None:
    assert _detect_device_type("cpu") == "cpu"


def test_detect_device_type_from_string_mps() -> None:
    assert _detect_device_type("mps") == "mps"


def test_detect_device_type_none_defaults_cpu() -> None:
    assert _detect_device_type(None) == "cpu"


# ---------------------------------------------------------------------------
# Smoke path - ALWAYS refuses
# ---------------------------------------------------------------------------


def test_gate_smoke_refuses_with_cuda_device(tmp_path: Path) -> None:
    args = _make_args(smoke=True, skip_auth_eval=False)

    class _CudaDev:
        type = "cuda"

    out = gate_auth_eval_call(args=args, device=_CudaDev(), **_gate_kwargs(tmp_path))
    assert out is None
    assert getattr(args, "auth_eval_skipped_reason", None) == SMOKE_REFUSAL_REASON
    # Refusal sentinel string class
    assert isinstance(args.auth_eval_skipped_reason, AuthEvalGateRefusal)


def test_gate_smoke_refuses_even_with_skip_auth_eval_false(tmp_path: Path) -> None:
    """Path B: smoke-mode skip is INDEPENDENT of --skip-auth-eval flag."""
    args = _make_args(smoke=True, skip_auth_eval=False)
    out = gate_auth_eval_call(args=args, device="cuda", **_gate_kwargs(tmp_path))
    assert out is None
    assert SMOKE_REFUSAL_REASON in args.auth_eval_skipped_reason


def test_gate_smoke_does_not_subprocess(tmp_path: Path) -> None:
    args = _make_args(smoke=True)
    with mock.patch.object(subprocess, "run") as run_mock:
        out = gate_auth_eval_call(args=args, device="cuda", **_gate_kwargs(tmp_path))
    assert out is None
    assert run_mock.call_count == 0


# ---------------------------------------------------------------------------
# Full-CPU advisory path (time-traveler L5)
# ---------------------------------------------------------------------------


def test_gate_full_cpu_refuses(tmp_path: Path) -> None:
    args = _make_args(smoke=False, skip_auth_eval=False)
    out = gate_auth_eval_call(
        args=args, device="cuda", full_cpu_active=True, **_gate_kwargs(tmp_path)
    )
    assert out is None
    assert args.auth_eval_skipped_reason == FULL_CPU_REFUSAL_REASON


def test_gate_full_cpu_does_not_subprocess(tmp_path: Path) -> None:
    args = _make_args()
    with mock.patch.object(subprocess, "run") as run_mock:
        out = gate_auth_eval_call(
            args=args,
            device="cuda",
            full_cpu_active=True,
            **_gate_kwargs(tmp_path),
        )
    assert out is None
    assert run_mock.call_count == 0


# ---------------------------------------------------------------------------
# --skip-auth-eval explicit opt-out
# ---------------------------------------------------------------------------


def test_gate_skip_flag_refuses_full_path(tmp_path: Path) -> None:
    args = _make_args(smoke=False, skip_auth_eval=True)
    out = gate_auth_eval_call(args=args, device="cuda", **_gate_kwargs(tmp_path))
    assert out is None
    assert args.auth_eval_skipped_reason == SKIP_FLAG_REASON


# ---------------------------------------------------------------------------
# Non-CUDA device refusal
# ---------------------------------------------------------------------------


def test_gate_cpu_device_refuses(tmp_path: Path) -> None:
    args = _make_args(smoke=False, skip_auth_eval=False)
    out = gate_auth_eval_call(args=args, device="cpu", **_gate_kwargs(tmp_path))
    assert out is None
    assert args.auth_eval_skipped_reason == CPU_REFUSAL_REASON


def test_gate_mps_device_refuses(tmp_path: Path) -> None:
    args = _make_args()
    out = gate_auth_eval_call(args=args, device="mps", **_gate_kwargs(tmp_path))
    assert out is None
    assert "device_type=cpu" in args.auth_eval_skipped_reason or "device_type" in args.auth_eval_skipped_reason


# ---------------------------------------------------------------------------
# Full / CUDA success path
# ---------------------------------------------------------------------------


def test_gate_full_cuda_runs_subprocess(tmp_path: Path) -> None:
    args = _make_args(smoke=False, skip_auth_eval=False)
    out_json = tmp_path / "auth_eval.json"
    out_json.write_text(json.dumps({"any": "payload"}), encoding="utf-8")

    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    fake_claim = _FakeClaim(score=0.234, lane_tag="[contest-CUDA]")
    with mock.patch.object(subprocess, "run", return_value=fake_proc) as run_mock, mock.patch(
        "tac.auth_eval_result.parse_auth_eval_score_claim", return_value=fake_claim
    ):
        out = gate_auth_eval_call(
            args=args,
            device="cuda",
            **_gate_kwargs(tmp_path, output_json=out_json),
        )
    assert run_mock.call_count == 1
    assert out is not None
    assert out["auth_eval_cuda_score"] == pytest.approx(0.234)
    assert out["auth_eval_score_axis"] == "contest_cuda"
    assert out["auth_eval_score_claim_valid"] is True
    assert out["auth_eval_lane_tag"] == "[contest-CUDA]"


def test_gate_full_cuda_proc_nonzero_raises(tmp_path: Path) -> None:
    args = _make_args()
    fake_proc = subprocess.CompletedProcess(args=[], returncode=2, stdout="x", stderr="y")
    with mock.patch.object(subprocess, "run", return_value=fake_proc):
        with pytest.raises(RuntimeError, match="contest_auth_eval.py failed"):
            gate_auth_eval_call(args=args, device="cuda", **_gate_kwargs(tmp_path))


def test_gate_full_cuda_invalid_claim_raises(tmp_path: Path) -> None:
    args = _make_args()
    out_json = tmp_path / "auth_eval.json"
    out_json.write_text(json.dumps({"any": "payload"}), encoding="utf-8")
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with mock.patch.object(subprocess, "run", return_value=fake_proc), mock.patch(
        "tac.auth_eval_result.parse_auth_eval_score_claim", return_value=None
    ):
        with pytest.raises(AuthEvalGateError, match="did not produce a valid"):
            gate_auth_eval_call(
                args=args,
                device="cuda",
                **_gate_kwargs(tmp_path, output_json=out_json),
            )


# ---------------------------------------------------------------------------
# Ordering: smoke takes priority over CPU refusal
# ---------------------------------------------------------------------------


def test_gate_smoke_short_circuits_before_device_check(tmp_path: Path) -> None:
    args = _make_args(smoke=True)
    out = gate_auth_eval_call(args=args, device="cpu", **_gate_kwargs(tmp_path))
    assert out is None
    # Smoke reason wins over CPU reason because gate checks smoke FIRST
    assert SMOKE_REFUSAL_REASON in args.auth_eval_skipped_reason


def test_gate_full_cpu_takes_priority_over_skip_flag(tmp_path: Path) -> None:
    args = _make_args(smoke=False, skip_auth_eval=True)
    out = gate_auth_eval_call(
        args=args, device="cuda", full_cpu_active=True, **_gate_kwargs(tmp_path)
    )
    assert out is None
    assert args.auth_eval_skipped_reason == FULL_CPU_REFUSAL_REASON


# ---------------------------------------------------------------------------
# Subprocess command shape
# ---------------------------------------------------------------------------


def test_gate_subprocess_command_contains_required_flags(tmp_path: Path) -> None:
    args = _make_args()
    out_json = tmp_path / "auth_eval.json"
    out_json.write_text("{}", encoding="utf-8")
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    fake_claim = _FakeClaim(score=0.0)
    captured = {}

    def _capture(cmd, **kw):
        captured["cmd"] = cmd
        return fake_proc

    with mock.patch.object(subprocess, "run", side_effect=_capture), mock.patch(
        "tac.auth_eval_result.parse_auth_eval_score_claim", return_value=fake_claim
    ):
        gate_auth_eval_call(
            args=args,
            device="cuda",
            **_gate_kwargs(tmp_path, output_json=out_json),
        )
    cmd = captured["cmd"]
    assert "--archive" in cmd
    assert "--inflate-sh" in cmd
    assert "--upstream-dir" in cmd
    assert "--device" in cmd
    assert "cuda" in cmd
    assert "--json-out" in cmd


def test_gate_extra_argv_appended(tmp_path: Path) -> None:
    args = _make_args()
    out_json = tmp_path / "auth_eval.json"
    out_json.write_text("{}", encoding="utf-8")
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    fake_claim = _FakeClaim(score=0.0)
    captured = {}

    def _capture(cmd, **kw):
        captured["cmd"] = cmd
        return fake_proc

    with mock.patch.object(subprocess, "run", side_effect=_capture), mock.patch(
        "tac.auth_eval_result.parse_auth_eval_score_claim", return_value=fake_claim
    ):
        gate_auth_eval_call(
            args=args,
            device="cuda",
            extra_argv=("--operator-approved-exact-cuda",),
            **_gate_kwargs(tmp_path, output_json=out_json),
        )
    assert "--operator-approved-exact-cuda" in captured["cmd"]


# ---------------------------------------------------------------------------
# Read-only namespace tolerance
# ---------------------------------------------------------------------------


def test_gate_namespace_setattr_tolerant() -> None:
    """A frozen namespace must not raise; the gate is best-effort on stamping."""

    class _Frozen:
        __slots__ = ()
        smoke = True
        skip_auth_eval = False

    args = _Frozen()  # type: ignore[abstract]
    # Should not raise even though setattr is forbidden.
    out = gate_auth_eval_call(
        args=args,
        device="cuda",
        archive_zip=Path("/tmp/dummy/archive.zip"),
        inflate_sh=Path("/tmp/dummy/inflate.sh"),
        upstream_dir=Path("/tmp/dummy/upstream"),
        output_json=Path("/tmp/dummy/auth_eval.json"),
        contest_auth_eval_script=Path("/tmp/dummy/contest_auth_eval.py"),
        substrate_tag="frozentest",
    )
    assert out is None
