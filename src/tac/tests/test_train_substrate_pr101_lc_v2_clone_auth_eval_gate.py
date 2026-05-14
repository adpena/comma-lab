"""Path B regression tests for the PR95++ smoke auth-eval gate.

Per CLAUDE.md "Auth eval EVERYWHERE" non-negotiable + HNeRV parity lesson L13
+ recipe ``smoke_validation_contract: training_artifact_v1``: smoke ALWAYS
skips contest_auth_eval, regardless of whether ``--skip-auth-eval`` is passed.
The full path runs auth eval unless ``--skip-auth-eval`` is set.

These tests exercise the call-site gate at
``experiments/train_substrate_pr101_lc_v2_clone_enhanced_curriculum.py`` line
~916 by importing ``main`` and patching the heavy paths
(``_train_smoke_loop`` + ``_run_contest_auth_eval_cuda``).

Lane: ``lane_pr95plus_smoke_archive_completion_20260513``
Memory: ``feedback_pr95plus_smoke_archive_completion_landed_20260513.md``
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER_PATH = (
    REPO_ROOT
    / "experiments"
    / "train_substrate_pr101_lc_v2_clone_enhanced_curriculum.py"
)


def _load_trainer_module():
    """Import the trainer module by file path (not on sys.path by default)."""

    spec = importlib.util.spec_from_file_location(
        "_pr95plus_trainer_under_test", TRAINER_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_smoke_result(output_dir: Path) -> dict:
    """Mimic the ``_train_smoke_loop`` contract without spinning up training."""

    archive_dir = output_dir / "archive_dir"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_bytes = b"PRC1\x00\x00\x00\x00stub-bytes-for-test"
    (archive_dir / "0.bin").write_bytes(archive_bytes)
    archive_zip = output_dir / "archive.zip"
    import zipfile

    with zipfile.ZipFile(archive_zip, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", archive_bytes)
    runtime_dir = output_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    inflate_sh = runtime_dir / "inflate.sh"
    inflate_sh.write_text("#!/bin/bash\nset -euo pipefail\necho noop\n")
    inflate_sh.chmod(0o755)
    return {
        "archive_sha256": "0" * 64,
        "archive_bytes": len(archive_bytes),
        "archive_zip_sha256": "1" * 64,
        "archive_zip_bytes": archive_zip.stat().st_size,
        "archive_zip_path": str(archive_zip),
        "runtime_inflate_sh": str(inflate_sh),
        "smoke_archive_mode": "ema_state",
        "smoke_archive_score_claim": False,
        "last_loss_parts": {},
        "stage_log": [],
        "training_mode": "smoke",
        "evidence_grade": "training-only",
    }


def _build_argv(
    *,
    output_dir: Path,
    upstream_dir: Path,
    device: str,
    smoke: bool,
    skip_auth_eval: bool,
) -> list[str]:
    """Construct argv for the trainer's CLI under test."""

    argv = [
        "--video-path",
        str(upstream_dir / "videos" / "0.mkv"),
        "--output-dir",
        str(output_dir),
        "--upstream-dir",
        str(upstream_dir),
        "--device",
        device,
        "--curriculum",
        "pr95_enhanced",
    ]
    if smoke:
        argv.append("--smoke")
    if skip_auth_eval:
        argv.append("--skip-auth-eval")
    return argv


@pytest.fixture
def trainer():
    return _load_trainer_module()


@pytest.fixture
def fake_upstream(tmp_path: Path) -> Path:
    """Minimal upstream tree so the trainer's preflight passes."""

    upstream = tmp_path / "upstream"
    (upstream / "videos").mkdir(parents=True)
    (upstream / "videos" / "0.mkv").write_bytes(b"\x00stub-mkv-bytes")
    (upstream / "public_test_video_names.txt").write_text("0.mkv\n")
    return upstream


def _patch_trainer(
    trainer_module,
    *,
    device_type: str = "cuda",
    decode_real_returns_pairs: bool = False,
):
    """Patch the heavy paths; return the run_auth_eval mock for assertions.

    When ``decode_real_returns_pairs`` is False (default; smoke tests),
    calling ``_canon_decode_real_pairs`` is treated as a contract violation.
    The full-mode tests pass True so ``_canon_decode_real_pairs`` returns a
    deterministic zero tensor and ``main`` proceeds to the smoke/full branch.
    """

    import torch

    auth_mock = mock.MagicMock(
        return_value={
            "auth_eval_json_path": "/tmp/fake.json",
            "auth_eval_cuda_score": 0.42,
            "auth_eval_score_axis": "contest_cuda",
            "auth_eval_lane_tag": "[contest-CUDA]",
            "auth_eval_score_claim_valid": True,
        }
    )
    smoke_loop = mock.MagicMock(side_effect=lambda **kw: _fake_smoke_result(kw["output_dir"]))
    full_main = mock.MagicMock(return_value=0)
    fake_device = torch.device(device_type)
    device_or_die = mock.MagicMock(return_value=fake_device)
    audit_func = mock.MagicMock(
        return_value={"E1": {"L1": "PASS"}, "E2": {"L2": "PASS"}}
    )
    detect_hw = mock.MagicMock(
        return_value=f"linux_x86_64_{device_type}_unknown"
    )
    if decode_real_returns_pairs:
        decode_real = mock.MagicMock(return_value=torch.zeros(2, 2, 3, 16, 16))
    else:
        decode_real = mock.MagicMock(
            side_effect=AssertionError(
                "decode_real_pairs must not be called in smoke"
            )
        )
    pin_seeds = mock.MagicMock()

    def _build_substrate(_):
        sub = mock.MagicMock()
        sub.to.return_value = sub
        sub.cfg = mock.MagicMock(num_pairs=2, latent_dim=8)
        return sub

    pres = [
        mock.patch.object(trainer_module, "_train_smoke_loop", smoke_loop),
        mock.patch.object(trainer_module, "_run_contest_auth_eval_cuda", auth_mock),
        mock.patch.object(trainer_module, "_full_main", full_main),
        mock.patch.object(trainer_module, "_canon_device_or_die", device_or_die),
        mock.patch.object(
            trainer_module,
            "audit_enhanced_curriculum_against_hnerv_parity_lessons",
            audit_func,
        ),
        mock.patch.object(trainer_module, "_canon_detect_hardware_substrate", detect_hw),
        mock.patch.object(trainer_module, "_canon_decode_real_pairs", decode_real),
        mock.patch.object(trainer_module, "_canon_pin_seeds", pin_seeds),
        mock.patch.object(
            trainer_module, "Pr101LcV2CloneSubstrate", side_effect=_build_substrate
        ),
    ]
    for p in pres:
        p.start()
    return auth_mock, smoke_loop, full_main, pres


def _stop_all(pres):
    for p in pres:
        try:
            p.stop()
        except Exception:
            pass


# ---- POSITIVE tests: smoke must not invoke contest_auth_eval ----


def test_smoke_cuda_without_skip_flag_does_not_invoke_auth_eval(
    trainer, tmp_path: Path, fake_upstream: Path
) -> None:
    """Path B core contract: smoke + cuda + no skip flag -> auth eval skipped."""

    output_dir = tmp_path / "out"
    auth_mock, smoke_loop, full_main, pres = _patch_trainer(
        trainer, device_type="cuda"
    )
    try:
        rc = trainer.main(
            _build_argv(
                output_dir=output_dir,
                upstream_dir=fake_upstream,
                device="cuda",
                smoke=True,
                skip_auth_eval=False,  # explicitly NOT set
            )
        )
    finally:
        _stop_all(pres)

    assert rc == 0
    assert auth_mock.call_count == 0, (
        "smoke MUST NOT invoke contest_auth_eval even on CUDA when "
        "--skip-auth-eval is not passed (Path B council decision)"
    )
    assert smoke_loop.call_count == 1
    assert full_main.call_count == 0
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["score_claim"] is False
    assert manifest["score_claim_valid"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["research_only"] is True
    assert manifest["evidence_grade"] == "training-only"
    result = manifest["result"]
    assert "auth_eval_skipped_reason" in result
    assert "training_artifact_v1" in result["auth_eval_skipped_reason"]
    assert "auth_eval_cuda_score" not in result


def test_smoke_cuda_with_skip_flag_does_not_invoke_auth_eval(
    trainer, tmp_path: Path, fake_upstream: Path
) -> None:
    """Idempotent: passing --skip-auth-eval at smoke is a no-op (still skipped)."""

    output_dir = tmp_path / "out"
    auth_mock, smoke_loop, _full, pres = _patch_trainer(
        trainer, device_type="cuda"
    )
    try:
        rc = trainer.main(
            _build_argv(
                output_dir=output_dir,
                upstream_dir=fake_upstream,
                device="cuda",
                smoke=True,
                skip_auth_eval=True,
            )
        )
    finally:
        _stop_all(pres)

    assert rc == 0
    assert auth_mock.call_count == 0
    assert smoke_loop.call_count == 1
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["result"].get("auth_eval_skipped_reason"), (
        "auth_eval_skipped_reason must be present even when --skip-auth-eval "
        "is explicitly passed at smoke; the gate emits the same reason for "
        "audit-trail consistency"
    )


def test_smoke_cpu_does_not_invoke_auth_eval(
    trainer, tmp_path: Path, fake_upstream: Path
) -> None:
    """CPU smoke is the macOS-CPU advisory path; auth eval gated by Path B."""

    output_dir = tmp_path / "out"
    auth_mock, _smoke, _full, pres = _patch_trainer(trainer, device_type="cpu")
    try:
        rc = trainer.main(
            _build_argv(
                output_dir=output_dir,
                upstream_dir=fake_upstream,
                device="cpu",
                smoke=True,
                skip_auth_eval=False,
            )
        )
    finally:
        _stop_all(pres)

    assert rc == 0
    assert auth_mock.call_count == 0
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["result"].get("auth_eval_skipped_reason")


# ---- NEGATIVE tests: full mode must still run auth eval (when implementing path
# returns; today _full_main raises NotImplementedError so we patch it). ----


def test_full_mode_without_skip_invokes_full_main_path(
    trainer, tmp_path: Path, fake_upstream: Path
) -> None:
    """Full mode (no --smoke) must reach ``_full_main``; the smoke gate must
    not catch full mode. The current SCAFFOLD ``_full_main`` raises, so we
    patch it to a no-op return-0 and assert the auth eval gate is reached
    only AFTER ``_full_main`` returns (i.e. proves full path is structurally
    different from smoke)."""

    output_dir = tmp_path / "out"
    auth_mock, smoke_loop, full_main, pres = _patch_trainer(
        trainer, device_type="cuda", decode_real_returns_pairs=True
    )
    try:
        rc = trainer.main(
            _build_argv(
                output_dir=output_dir,
                upstream_dir=fake_upstream,
                device="cuda",
                smoke=False,
                skip_auth_eval=False,
            )
        )
    finally:
        _stop_all(pres)

    assert rc == 0
    # Full path: _full_main is hit BEFORE the smoke loop / auth eval gate.
    assert full_main.call_count == 1
    assert smoke_loop.call_count == 0
    # auth_mock at line 916 is gated on `args.smoke`; full mode does NOT enter
    # the smoke arm, so the elif arm runs IF main() reaches that line. But
    # `_full_main` returns 0 BEFORE reaching the smoke loop, so auth eval is
    # not invoked from this path either. The structural separation is the
    # point: smoke vs full take entirely different code paths.
    assert auth_mock.call_count == 0


def test_full_mode_unimplemented_in_scaffold(
    trainer, tmp_path: Path, fake_upstream: Path, monkeypatch
) -> None:
    """Without the patch, ``_full_main`` raises NotImplementedError in
    SCAFFOLD mode per HNeRV parity lesson L13. This test pins that contract
    so a future Path-A or Phase-2 landing has to deliberately drop the
    NotImplementedError."""

    # Re-import a fresh module (don't apply the smoke-loop patches)
    fresh = _load_trainer_module()
    output_dir = tmp_path / "out"

    with mock.patch.object(fresh, "_canon_pin_seeds"), mock.patch.object(
        fresh, "_canon_detect_hardware_substrate", return_value="linux_x86_64_cuda_unknown"
    ), mock.patch.object(
        fresh, "_canon_device_or_die", return_value=__import__("torch").device("cuda")
    ), mock.patch.object(
        fresh,
        "audit_enhanced_curriculum_against_hnerv_parity_lessons",
        return_value={},
    ), mock.patch.object(
        fresh, "_canon_decode_real_pairs", return_value=__import__("torch").zeros(2, 2, 3, 16, 16)
    ), mock.patch.object(
        fresh, "Pr101LcV2CloneSubstrate", return_value=mock.MagicMock(to=lambda d: mock.MagicMock())
    ):
        with pytest.raises(NotImplementedError):
            fresh.main(
                _build_argv(
                    output_dir=output_dir,
                    upstream_dir=fake_upstream,
                    device="cuda",
                    smoke=False,
                    skip_auth_eval=False,
                )
            )


# ---- Argparse contract tests ----


def test_skip_auth_eval_help_documents_smoke_always_skips(trainer) -> None:
    """The --skip-auth-eval help string must document that smoke ALWAYS skips
    auth eval regardless of this flag, so a future caller does not assume the
    flag is the only opt-out."""

    parser = trainer._build_parser()
    skip_action = next(
        a for a in parser._actions if "--skip-auth-eval" in a.option_strings
    )
    help_text = (skip_action.help or "").lower()
    assert "smoke" in help_text and "always" in help_text
    assert "training_artifact_v1" in help_text or "no score" in help_text


def test_call_site_gate_branches_distinct(trainer) -> None:
    """Inspect the trainer's ``main`` source to assert the smoke gate is
    syntactically structured as ``if args.smoke ... elif not args.skip_auth_eval
    ...`` so a refactor cannot accidentally drop the smoke arm."""

    src = TRAINER_PATH.read_text(encoding="utf-8")
    # Both required tokens must be present, in order, near each other.
    smoke_arm_idx = src.find("if args.smoke:")
    elif_arm_idx = src.find("elif not args.skip_auth_eval and device.type ==")
    assert smoke_arm_idx >= 0, "smoke gate `if args.smoke:` must be present"
    assert elif_arm_idx >= 0, (
        "elif arm `elif not args.skip_auth_eval and device.type == \"cuda\":` "
        "must follow the smoke gate"
    )
    assert smoke_arm_idx < elif_arm_idx, (
        "smoke gate must precede the elif arm so smoke short-circuits before "
        "the auth-eval invocation"
    )
    assert "auth_eval_skipped_reason" in src
    assert "training_artifact_v1" in src
