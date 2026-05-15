# SPDX-License-Identifier: MIT
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
    CPU_REFUSAL_REASON,
    EXPLICIT_NON_CUDA_AUTH_EVAL_RESULT_REASON,
    FULL_CPU_REFUSAL_REASON,
    SKIP_FLAG_REASON,
    SMOKE_REFUSAL_REASON,
    AuthEvalGateError,
    AuthEvalGateRefusal,
    _detect_device_type,
    format_smoke_skip_banner,
    gate_auth_eval_call,
)


@pytest.fixture(autouse=True)
def _clear_auth_eval_device_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests should not inherit provider wrapper auth-eval device settings."""

    monkeypatch.delenv("AUTH_EVAL_DEVICE", raising=False)


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


def _coherent_score_payload(*, score_axis: str = "contest_cpu") -> dict[str, object]:
    seg = 0.001
    pose = 0.001
    archive_bytes = 1000
    score = 100.0 * seg + (10.0 * pose) ** 0.5 + 25.0 * archive_bytes / 37_545_489.0
    return {
        "canonical_score": score,
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "archive_bytes": archive_bytes,
        "score_axis": score_axis,
        "lane_tag": "[contest-CPU]",
        "evidence_grade": "contest-CPU",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "exact_cuda_eval_complete": False,
        "cpu_leaderboard_reproduction_eligible": True,
    }


def _diagnostic_cpu_score_payload() -> dict[str, object]:
    payload = _coherent_score_payload(score_axis="diagnostic_cpu")
    payload.update(
        {
            "cpu_leaderboard_reproduction_eligible": False,
            "diagnostic_blockers": ["modal_training_wrapper_auth_eval_advisory_only"],
            "evidence_grade": "B",
            "lane_tag": "[diagnostic-auth-eval]",
        }
    )
    return payload


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


def test_gate_explicit_env_cpu_runs_cpu_auth_eval_without_cuda_claim_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Modal can train on CUDA while explicitly running CPU auth eval."""

    monkeypatch.setenv("AUTH_EVAL_DEVICE", "cpu")
    args = _make_args(smoke=False, skip_auth_eval=False)
    out_json = tmp_path / "auth_eval_cpu.json"
    out_json.write_text(json.dumps(_diagnostic_cpu_score_payload()), encoding="utf-8")
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    with mock.patch.object(subprocess, "run", return_value=fake_proc) as run_mock:
        out = gate_auth_eval_call(
            args=args,
            device="cuda",
            **_gate_kwargs(tmp_path, output_json=out_json),
        )

    assert out is None
    assert run_mock.call_count == 1
    cmd = run_mock.call_args.args[0]
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert args.auth_eval_skipped_reason == EXPLICIT_NON_CUDA_AUTH_EVAL_RESULT_REASON


def test_gate_explicit_env_cpu_can_return_axis_preserving_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU-aware callers can consume CPU auth eval without CUDA field aliases."""

    monkeypatch.setenv("AUTH_EVAL_DEVICE", "cpu")
    args = _make_args(smoke=False, skip_auth_eval=False)
    out_json = tmp_path / "auth_eval_cpu.json"
    out_json.write_text(json.dumps(_coherent_score_payload()), encoding="utf-8")
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    with mock.patch.object(subprocess, "run", return_value=fake_proc):
        out = gate_auth_eval_call(
            args=args,
            device="cuda",
            return_non_cuda_result=True,
            **_gate_kwargs(tmp_path, output_json=out_json),
        )

    assert out is not None
    assert out["auth_eval_device"] == "cpu"
    assert out["auth_eval_score_axis"] == "contest_cpu"
    assert out["auth_eval_cpu_score"] == pytest.approx(out["auth_eval_score"])
    assert "auth_eval_cuda_score" not in out
    assert out["auth_eval_cpu_leaderboard_reproduction_eligible"] is True


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
    with mock.patch.object(
        subprocess, "run", return_value=fake_proc
    ), pytest.raises(RuntimeError, match=r"contest_auth_eval\.py failed"):
        gate_auth_eval_call(args=args, device="cuda", **_gate_kwargs(tmp_path))


def test_gate_full_cuda_invalid_claim_raises(tmp_path: Path) -> None:
    args = _make_args()
    out_json = tmp_path / "auth_eval.json"
    out_json.write_text(json.dumps({"any": "payload"}), encoding="utf-8")
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with mock.patch.object(subprocess, "run", return_value=fake_proc), mock.patch(
        "tac.auth_eval_result.parse_auth_eval_score_claim", return_value=None
    ), pytest.raises(AuthEvalGateError, match="did not produce a valid"):
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
    assert "--keep-work-dir" in cmd
    assert "--work-dir" in cmd
    assert cmd[cmd.index("--work-dir") + 1] == str(
        out_json.parent / f"{out_json.stem}_work"
    )
    assert "--allow-temp-work-dir" not in cmd


def test_gate_modal_cpu_advisory_uses_explicit_temp_bypass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Modal train-lane CPU auth eval is advisory, not score custody."""

    monkeypatch.setenv("AUTH_EVAL_DEVICE", "cpu")
    monkeypatch.setenv("MODAL_AUTH_EVAL_ADVISORY_ONLY", "1")
    args = _make_args(smoke=False, skip_auth_eval=False)
    out_json = tmp_path / "contest_auth_eval_cpu.json"
    out_json.write_text(json.dumps(_diagnostic_cpu_score_payload()), encoding="utf-8")
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    captured = {}

    def _capture(cmd, **kw):
        captured["cmd"] = cmd
        return fake_proc

    with mock.patch.object(subprocess, "run", side_effect=_capture):
        out = gate_auth_eval_call(
            args=args,
            device="cuda",
            **_gate_kwargs(tmp_path, output_json=out_json),
        )

    assert out is None
    cmd = captured["cmd"]
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert "--keep-work-dir" in cmd
    assert "--work-dir" in cmd
    assert "--allow-temp-work-dir" in cmd
    assert args.auth_eval_skipped_reason == EXPLICIT_NON_CUDA_AUTH_EVAL_RESULT_REASON


def test_gate_modal_cpu_advisory_rejects_undemoted_contest_cpu_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Modal advisory eval must not leak contest-CPU authority through the gate."""

    monkeypatch.setenv("AUTH_EVAL_DEVICE", "cpu")
    monkeypatch.setenv("MODAL_AUTH_EVAL_ADVISORY_ONLY", "1")
    args = _make_args(smoke=False, skip_auth_eval=False)
    out_json = tmp_path / "contest_auth_eval_cpu.json"
    out_json.write_text(json.dumps(_coherent_score_payload()), encoding="utf-8")
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    with mock.patch.object(subprocess, "run", return_value=fake_proc), pytest.raises(
        AuthEvalGateError,
        match="contest-CPU-authority payload",
    ):
        gate_auth_eval_call(
            args=args,
            device="cuda",
            return_non_cuda_result=True,
            **_gate_kwargs(tmp_path, output_json=out_json),
        )


def test_gate_modal_cpu_advisory_can_return_demoted_diagnostic_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU-aware Modal train-lane callers must not see contest-CPU authority."""

    monkeypatch.setenv("AUTH_EVAL_DEVICE", "cpu")
    monkeypatch.setenv("MODAL_AUTH_EVAL_ADVISORY_ONLY", "1")
    args = _make_args(smoke=False, skip_auth_eval=False)
    out_json = tmp_path / "contest_auth_eval_cpu.json"
    out_json.write_text(json.dumps(_diagnostic_cpu_score_payload()), encoding="utf-8")
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    with mock.patch.object(subprocess, "run", return_value=fake_proc):
        out = gate_auth_eval_call(
            args=args,
            device="cuda",
            return_non_cuda_result=True,
            **_gate_kwargs(tmp_path, output_json=out_json),
        )

    assert out is not None
    assert out["auth_eval_device"] == "cpu"
    assert out["auth_eval_score_axis"] == "diagnostic_cpu"
    assert out["auth_eval_evidence_grade"] == "B"
    assert out["auth_eval_cpu_leaderboard_reproduction_eligible"] is False


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
