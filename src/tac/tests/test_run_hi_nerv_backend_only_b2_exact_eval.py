# SPDX-License-Identifier: MIT
"""Tests for the B2 HiNeRV backend-only exact-eval BRIDGE.

NO FAKE: every test builds a REAL tiny HiNeRV archive (member ``x``) via the
canonical archive serializer and exercises the bridge's REAL behavior:
- runtime emission vendors the real decode-only modules (NO scorer imports),
- the emitted inflate.py accepts member ``x``,
- the num_pairs probe reads the real config,
- the inflate-only validation runs the REAL inflate.sh end-to-end on REAL bytes
  and produces frame-aligned contest raw output,
- device->axis tagging marks local-CPU advisory / non-authoritative,
- the dual-axis authoritative recipe is emitted with both contest axes.

A test whose assertions would still pass if the bridge body were replaced by
``return canonical_markers`` is forbidden (Slot EEE Class 2). These assert on
real returncodes, real raw byte counts, and real vendored module contents.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tac.submission_archive import build_minimal_single_member_archive_bytes
from tac.substrates.hi_nerv.architecture import HinervConfig, HinervSubstrate
from tac.substrates.hi_nerv.archive import pack_archive

REPO_ROOT = Path(__file__).resolve().parents[3]
_BRIDGE_PATH = REPO_ROOT / "tools" / "run_hi_nerv_backend_only_b2_exact_eval.py"


def _load_bridge():
    """Import the bridge module by path (it lives under tools/, not a package)."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    spec = importlib.util.spec_from_file_location(
        "run_hi_nerv_backend_only_b2_exact_eval", _BRIDGE_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BRIDGE = _load_bridge()


def _tiny_backend_archive(tmp_path: Path, *, num_pairs: int = 2) -> Path:
    """Build a REAL tiny HiNeRV backend-only archive.zip (member ``x``)."""
    cfg = HinervConfig(
        latent_dim_coarse=2,
        latent_dim_mid=2,
        latent_dim_fine=2,
        embed_dim=2,
        initial_grid_h=1,
        initial_grid_w=1,
        decoder_channels=(2, 2, 2),
        sin_frequency=3.0,
        num_upsample_blocks=3,
        mid_injection_block_index=0,
        fine_injection_block_index=1,
        num_pairs=num_pairs,
        output_height=8,
        output_width=8,
    )
    torch.manual_seed(7)
    model = HinervSubstrate(cfg).eval()
    decoder_state = {
        k: v
        for k, v in dict(model.state_dict()).items()
        if k not in {"latents_coarse", "latents_mid", "latents_fine"}
    }
    meta = {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "mid_injection_block_index": cfg.mid_injection_block_index,
        "fine_injection_block_index": cfg.fine_injection_block_index,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }
    packet = pack_archive(
        decoder_state,
        model.latents_coarse.detach(),
        model.latents_mid.detach(),
        model.latents_fine.detach(),
        meta,
    )
    archive_bytes, _method = build_minimal_single_member_archive_bytes(packet)
    tmp_path.mkdir(parents=True, exist_ok=True)
    archive = tmp_path / "archive_backend_only.zip"
    archive.write_bytes(archive_bytes)
    return archive


def _single_video_list(tmp_path: Path) -> Path:
    p = tmp_path / "names.txt"
    p.write_text("0.mkv\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Runtime emission (the x/0.bin-accepting hermetic runtime).
# ---------------------------------------------------------------------------


def test_emit_runtime_dir_vendors_real_decode_modules_no_scorer(tmp_path: Path) -> None:
    info = BRIDGE._emit_runtime_dir(tmp_path / "rt")
    assert info["scorer_imports_vendored"] is False
    assert info["member_parser_accepts"] == ["x", "0.bin"]
    # Real vendored modules present on disk.
    rt = Path(info["runtime_dir"])
    assert (rt / "inflate.sh").is_file()
    assert (rt / "inflate.py").is_file()
    assert (rt / "src" / "tac" / "substrates" / "hi_nerv" / "inflate.py").is_file()
    assert (rt / "src" / "tac" / "substrates" / "hi_nerv" / "archive.py").is_file()
    assert (rt / "src" / "tac" / "substrates" / "_shared" / "inflate_runtime.py").is_file()
    assert "hi_nerv/inflate.py" in info["vendored_modules"]
    # NO scorer module vendored (strict-scorer-rule).
    for forbidden in ("scorer.py", "modules.py", "segnet", "posenet"):
        assert not any(forbidden in m for m in info["vendored_modules"])


def test_emitted_inflate_sh_passes_three_arg_contract(tmp_path: Path) -> None:
    info = BRIDGE._emit_runtime_dir(tmp_path / "rt")
    text = Path(info["inflate_sh"]).read_text(encoding="utf-8")
    assert "$1" in text and "$2" in text and "$3" in text
    assert "set -euo pipefail" in text
    # Honours ${PYTHON:-python3} so dep-closure can be satisfied.
    assert "${PYTHON:-python3}" in text


def test_emitted_inflate_py_accepts_member_x(tmp_path: Path) -> None:
    info = BRIDGE._emit_runtime_dir(tmp_path / "rt")
    text = Path(info["inflate_py"]).read_text(encoding="utf-8")
    assert "archive_dir / 'x'" in text
    assert "archive_dir / '0.bin'" in text
    # No scorer-network import in the emitted inflate.py.
    assert "segnet" not in text.lower() and "posenet" not in text.lower()


# ---------------------------------------------------------------------------
# Archive parsing + num_pairs probe (on a REAL archive).
# ---------------------------------------------------------------------------


def test_payload_member_bytes_reads_x(tmp_path: Path) -> None:
    archive = _tiny_backend_archive(tmp_path)
    member, payload = BRIDGE._payload_member_bytes(archive)
    assert member == "x"
    assert payload[:4] == b"HIV1"


def test_probe_num_pairs_reads_real_config(tmp_path: Path) -> None:
    archive = _tiny_backend_archive(tmp_path, num_pairs=2)
    assert BRIDGE._probe_num_pairs(archive) == 2
    archive5 = _tiny_backend_archive(tmp_path / "five", num_pairs=5)
    assert BRIDGE._probe_num_pairs(archive5) == 5


def test_probe_num_pairs_returns_none_on_garbage(tmp_path: Path) -> None:
    import zipfile

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("x", b"not a hinerv archive")
    assert BRIDGE._probe_num_pairs(bad) is None


# ---------------------------------------------------------------------------
# Device -> axis tagging (the false-authority contract).
# ---------------------------------------------------------------------------


def test_cpu_device_axis_is_non_authoritative_on_macos(monkeypatch) -> None:
    monkeypatch.setattr(BRIDGE.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(BRIDGE.platform, "machine", lambda: "arm64")
    axis = BRIDGE._device_axis_caveat("cpu")
    assert axis["axis_tag"] == "[macOS-CPU advisory]"
    assert axis["authoritative"] is False
    assert axis["score_claim"] is False
    assert axis["frontier_claim"] is False


def test_cpu_device_axis_is_contest_cpu_on_linux_x86(monkeypatch) -> None:
    monkeypatch.setattr(BRIDGE.platform, "system", lambda: "Linux")
    monkeypatch.setattr(BRIDGE.platform, "machine", lambda: "x86_64")
    axis = BRIDGE._device_axis_caveat("cpu")
    assert axis["axis_tag"] == "[contest-CPU]"
    assert axis["authoritative"] is True


def test_cuda_device_axis_is_authoritative_pending_hardware() -> None:
    axis = BRIDGE._device_axis_caveat("cuda")
    assert "contest-CUDA" in axis["axis_tag"]
    assert axis["authoritative"] is True


# ---------------------------------------------------------------------------
# Dual-axis authoritative recipe (both contest axes; not fired).
# ---------------------------------------------------------------------------


def test_dual_axis_recipe_has_both_contest_axes(tmp_path: Path) -> None:
    archive = _tiny_backend_archive(tmp_path)
    recipe = BRIDGE._dual_axis_authoritative_recipe(
        archive,
        inflate_sh="rt/inflate.sh",
        video_names_file=_single_video_list(tmp_path),
        out_row=None,
    )
    assert recipe["axis_1_contest_cpu_linux_x86_64"]["expected_axis_tag"] == "[contest-CPU]"
    assert recipe["axis_2_contest_cuda_t4"]["expected_axis_tag"] == "[contest-CUDA]"
    # Both reference the SAME archive bytes (apples-to-apples).
    assert archive.as_posix() in recipe["axis_1_contest_cpu_linux_x86_64"]["command"]
    assert archive.as_posix() in recipe["axis_2_contest_cuda_t4"]["command"]
    # The compliance gate is wired in.
    assert "pre_submission_compliance_check.py" in recipe["step_3_compliance_gate"]["command"]
    assert "--device cpu" in recipe["axis_1_contest_cpu_linux_x86_64"]["command"]
    assert "--device cuda" in recipe["axis_2_contest_cuda_t4"]["command"]


# ---------------------------------------------------------------------------
# END-TO-END inflate-only validation on a REAL archive (PR106 dep-closure half).
# ---------------------------------------------------------------------------


def _venv_python() -> str | None:
    cand = REPO_ROOT / ".venv" / "bin" / "python"
    if not cand.exists():
        return None
    try:
        out = subprocess.run(
            [cand.as_posix(), "-c", "import torch, brotli, numpy"],
            capture_output=True,
            timeout=60,
        )
    except Exception:  # pragma: no cover
        return None
    return cand.as_posix() if out.returncode == 0 else None


def test_inflate_only_validation_produces_frame_aligned_raw(tmp_path: Path) -> None:
    """The REAL inflate.sh runs end-to-end and writes contest-shaped raw bytes."""
    py = _venv_python()
    if py is None:
        pytest.skip("repo .venv with torch+brotli+numpy not available")
    archive = _tiny_backend_archive(tmp_path, num_pairs=2)
    info = BRIDGE._emit_runtime_dir(tmp_path / "rt")
    inflate_sh = Path(info["inflate_sh"])
    result = BRIDGE._inflate_only_validation(
        archive=archive,
        inflate_sh=inflate_sh,
        video_names_file=_single_video_list(tmp_path),
        inflated_dir=tmp_path / "inflated",
        timeout=600,
        inflate_python=py,
    )
    assert result["inflate_returncode"] == 0, result["inflate_stderr_tail"]
    assert result["inflate_ok"] is True
    # num_pairs=2 -> 4 frames at camera resolution, frame-aligned.
    bytes_per_frame = 874 * 1164 * 3
    out = result["raw_outputs"][0]
    assert out["raw_bytes"] == 4 * bytes_per_frame
    assert out["raw_bytes_frame_aligned"] is True
    assert out["raw_frames"] == 4
    assert result["inflate_elapsed_seconds"] >= 0.0


def test_inflate_only_validation_fails_closed_without_dep_closure(tmp_path: Path) -> None:
    """Bare python3 (no brotli/torch) makes the inflate fail-closed (PR106 class)."""
    archive = _tiny_backend_archive(tmp_path, num_pairs=2)
    info = BRIDGE._emit_runtime_dir(tmp_path / "rt")
    inflate_sh = Path(info["inflate_sh"])
    # Point PYTHON at an interpreter that lacks the deps: a brand-new venv-free
    # python whose sys.path we cannot guarantee. Use a tiny shim that raises.
    shim = tmp_path / "nodeps_python.sh"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        # Emulate a python that cannot import brotli by running a python -c that
        # forces ImportError on the inflate import chain.
        'exec python3 -c "import sys; sys.modules[\'brotli\']=None; '
        "import runpy; runpy.run_path(sys.argv[1], run_name=\\\"__main__\\\")\" \"$@\"\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    result = BRIDGE._inflate_only_validation(
        archive=archive,
        inflate_sh=inflate_sh,
        video_names_file=_single_video_list(tmp_path),
        inflated_dir=tmp_path / "inflated",
        timeout=300,
        inflate_python=shim.as_posix(),
    )
    # Must NOT silently succeed; the bridge surfaces the failure honestly.
    assert result["inflate_ok"] is False
    assert result["inflate_returncode"] != 0


def test_inflate_only_rejects_unsafe_archive_member(tmp_path: Path) -> None:
    import zipfile

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../escape", b"payload")
    info = BRIDGE._emit_runtime_dir(tmp_path / "rt")
    with pytest.raises(SystemExit, match="unsafe archive member"):
        BRIDGE._inflate_only_validation(
            archive=bad,
            inflate_sh=Path(info["inflate_sh"]),
            video_names_file=_single_video_list(tmp_path),
            inflated_dir=tmp_path / "inflated",
            timeout=60,
        )


# ---------------------------------------------------------------------------
# Disk hygiene (certify-or-block; rebuildable scratch cleaned).
# ---------------------------------------------------------------------------


def test_certify_and_clean_deletes_rebuildable_frames(tmp_path: Path) -> None:
    inflated = tmp_path / "inflated"
    inflated.mkdir()
    (inflated / "0.raw").write_bytes(b"x" * 1000)
    rec = BRIDGE._certify_and_clean(inflated, keep=False)
    assert rec["rebuildable"] is True
    assert rec["inflated_bytes"] == 1000
    assert rec["cleaned"] is True
    assert not inflated.exists()


def test_certify_and_clean_keeps_when_requested(tmp_path: Path) -> None:
    inflated = tmp_path / "inflated"
    inflated.mkdir()
    (inflated / "0.raw").write_bytes(b"x" * 10)
    rec = BRIDGE._certify_and_clean(inflated, keep=True)
    assert rec["cleaned"] is False
    assert inflated.exists()


# ---------------------------------------------------------------------------
# Full CLI run on a REAL smoke archive (the e2e advisory path).
# ---------------------------------------------------------------------------


def test_cli_smoke_archive_emits_v1_row_with_pipeline_works(tmp_path: Path) -> None:
    """End-to-end CLI on a REAL tiny archive: inflate runs, v1 row emitted, NO score claim."""
    py = _venv_python()
    if py is None:
        pytest.skip("repo .venv with torch+brotli+numpy not available")
    archive = _tiny_backend_archive(tmp_path, num_pairs=2)
    # pytest tmp_path resolves under /private/tmp on macOS; the bridge's
    # disk-hygiene guard refuses /tmp work roots. Use a repo-local .omx/tmp
    # scratch dir (cleaned at the end) so the legit smoke path is exercised.
    import shutil as _shutil
    import uuid as _uuid

    scratch = REPO_ROOT / ".omx" / "tmp" / f"b2_bridge_test_{_uuid.uuid4().hex[:8]}"
    scratch.mkdir(parents=True, exist_ok=True)
    out_row = scratch / "out" / "hi_nerv_backend_only_exact_eval.json"
    work_root = scratch / "work"
    try:
        proc = subprocess.run(
            [
                py,
                _BRIDGE_PATH.as_posix(),
                "--archive",
                archive.as_posix(),
                "--device",
                "cpu",
                "--work-root",
                work_root.as_posix(),
                "--out-row",
                out_row.as_posix(),
                "--inflate-python",
                py,
            ],
            cwd=REPO_ROOT.as_posix(),
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert proc.returncode == 0, proc.stderr[-3000:]
        assert out_row.is_file()
        row = json.loads(out_row.read_text(encoding="utf-8"))
    finally:
        _shutil.rmtree(scratch, ignore_errors=True)
    assert row["schema"] == "hi_nerv_backend_only_exact_eval.v1"
    assert row["family"] == "hinerv"
    assert row["archive_num_pairs"] == 2
    assert row["pipeline_mode"] == "inflate_only_validation_smoke_archive"
    # Pipeline-works claim is REAL; score claim is False (smoke axis).
    assert row["pipeline_works"] is True
    assert row["score_claim"] is False
    assert row["promotable"] is False
    assert row["verdict"] == "pipeline_inflate_ok_evaluate_requires_600_pairs"
    # False-authority + axis markers.
    assert row["axis_tag"] == "[macOS-CPU advisory]" or "advisory" in row["axis_tag"]
    assert row["frontier_claim"] is False
    # Dual-axis authoritative recipe is present (turnkey for a B1 600-pair archive).
    rec = row["dual_axis_authoritative_recipe"]
    assert rec["axis_1_contest_cpu_linux_x86_64"]["expected_axis_tag"] == "[contest-CPU]"
    assert rec["axis_2_contest_cuda_t4"]["expected_axis_tag"] == "[contest-CUDA]"
    # Disk hygiene cleaned the rebuildable frames.
    assert row["disk_hygiene"]["cleaned"] is True


def test_cli_print_dual_axis_recipe_no_run(tmp_path: Path) -> None:
    py = _venv_python() or sys.executable
    proc = subprocess.run(
        [py, _BRIDGE_PATH.as_posix(), "--print-dual-axis-recipe"],
        cwd=REPO_ROOT.as_posix(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = json.loads(proc.stdout)
    assert payload["mode"] == "dual_axis_recipe_only"
    assert (
        payload["dual_axis_authoritative_recipe"]["axis_2_contest_cuda_t4"][
            "expected_axis_tag"
        ]
        == "[contest-CUDA]"
    )


def test_cli_rejects_tmp_work_root(tmp_path: Path) -> None:
    py = _venv_python() or sys.executable
    archive = _tiny_backend_archive(tmp_path)
    proc = subprocess.run(
        [
            py,
            _BRIDGE_PATH.as_posix(),
            "--archive",
            archive.as_posix(),
            "--work-root",
            "/tmp/b2_should_be_refused",
        ],
        cwd=REPO_ROOT.as_posix(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0
    assert "tmp" in (proc.stderr + proc.stdout).lower()
