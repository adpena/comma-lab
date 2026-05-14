# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import struct
import sys
import zipfile
from pathlib import Path

import pytest

from tac.pr85_bundle import PR85_HEADERLESS_RANDMULTI_SPECS, pack_pr85_bundle


brotli = pytest.importorskip("brotli")

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_pr85_fixed_runtime_bridge_candidate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_pr85_fixed_runtime_bridge_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _p1d1_pose_raw() -> bytes:
    rows = b"\x00" * 600
    return b"P1D1" + bytes([1, 0]) + len(rows).to_bytes(2, "little") + rows


def _randmulti_zero_payload() -> bytes:
    return b"\x00" * sum(spec[3] for spec in PR85_HEADERLESS_RANDMULTI_SPECS)


def _write_pr85_archive(path: Path) -> None:
    segments = {
        "mask": b"QMA9" + (600).to_bytes(4, "little") + (512).to_bytes(4, "little")
        + (384).to_bytes(4, "little") + (8).to_bytes(4, "little") + b"maskdata",
        "model": brotli.compress(b"QH0" + b"renderer" * 128, quality=5),
        "pose": brotli.compress(_p1d1_pose_raw(), quality=5),
        "post": brotli.compress(bytes([0]) * 2400, quality=5),
        "shift": brotli.compress(b"SD4" + bytes([0]) * 600, quality=5),
        "frac": brotli.compress(b"FV1" + b"\x00\x00", quality=5),
        "frac2": brotli.compress(b"FH2" + bytes([4]) * 600, quality=5),
        "frac3": brotli.compress(b"FD3" + bytes([0]) * 600, quality=5),
        "bias": brotli.compress(b"BD1" + bytes([0]) * 600, quality=5),
        "region": brotli.compress(b"RH1" + bytes([0]) * 600, quality=5),
        "randmulti": brotli.compress(_randmulti_zero_payload(), quality=5),
    }
    raw = pack_pr85_bundle(segments, header_mode="explicit_30")
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("x", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, raw)


def _runtime_dir(path: Path, *, qh0: bool) -> Path:
    path.mkdir(parents=True)
    (path / "inflate.sh").write_text(
        "#!/usr/bin/env bash\nif [ -f \"$ARCHIVE_DIR/qpost.bin\" ]; then echo qpost.bin; fi\n",
        encoding="utf-8",
    )
    qh0_marker = "\n# QH0\n" if qh0 else "\n"
    (path / "inflate_renderer.py").write_text(
        "# masks.qma9\n"
        "def _load_masks_from_qma9(path):\n"
        "    return path\n"
        + qh0_marker,
        encoding="utf-8",
    )
    (path / "apply_qzs3_postprocess.py").write_text(
        "# QRM1 qpost.bin\n"
        "def read_qpost(path, device):\n"
        "    return path, device\n"
        "def _decode_qrm1_randmulti(raw, device):\n"
        "    return raw, device\n",
        encoding="utf-8",
    )
    return path


def test_bridge_candidate_writes_expanded_archive_and_blocks_only_on_qh0(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    _write_pr85_archive(source)
    runtime = _runtime_dir(tmp_path / "runtime", qh0=False)

    manifest = module.build_bridge_candidate(source, tmp_path / "out", robust_current_dir=runtime)

    assert manifest["score_claim"] is False
    assert manifest["dispatch_performed"] is False
    assert manifest["runtime_gate"]["remaining_blockers"] == ["qh0_model_loader_available"]
    assert manifest["dispatch_gate"] == "blocked_local_runtime_bridge"
    archive_path = REPO / manifest["candidate_archive"]["path"]
    with zipfile.ZipFile(archive_path, "r") as zf:
        assert zf.namelist() == ["masks.qma9", "renderer.bin", "optimized_poses.bin", "qpost.bin"]
        assert zf.read("renderer.bin").startswith(b"QH0")
        assert len(zf.read("optimized_poses.bin")) == 600 * 6 * 2
        qpost = zf.read("qpost.bin")
    lengths = struct.unpack_from("<" + "I" * 8, qpost, 4)
    randmulti = qpost[4 + 8 * 4 + sum(lengths[:-1]) :]
    assert brotli.decompress(randmulti).startswith(b"QRM1")
    assert (tmp_path / "out" / "expanded_qpost_qrm1_posefp16" / "manifest.json").is_file()


def test_bridge_candidate_dispatch_gate_passes_when_runtime_has_qh0(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    _write_pr85_archive(source)
    runtime = _runtime_dir(tmp_path / "runtime", qh0=True)

    manifest = module.build_bridge_candidate(source, tmp_path / "out", robust_current_dir=runtime)

    assert manifest["runtime_gate"]["ready_for_exact_eval_dispatch_claim"] is True
    assert manifest["runtime_gate"]["remaining_blockers"] == []
    assert manifest["dispatch_gate"] == "eligible_for_exact_eval_after_lane_claim"
