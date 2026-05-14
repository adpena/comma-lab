# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

from tac.qma9_range_mask_contract import encode_qma9_mask, sha256_bytes


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_qma9_context_run_escape_candidate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_qma9_context_run_escape_candidate_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_zip(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("p")
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def test_qmc1_context_run_escape_roundtrips_and_keeps_model_hot() -> None:
    script = _load_script()
    row_a = bytes([0, 0, 0, 0, 1, 1, 1, 1])
    row_b = bytes([2, 2, 2, 2, 3, 3, 3, 3])
    raw = row_a + row_a + row_b + row_b

    encoded, stats = script.encode_qma9_context_run_escape_mask(
        raw,
        frame_count=1,
        width=4,
        height=8,
        min_run_length=4,
        require_left_context=False,
    )

    assert encoded[:4] == b"QMC1"
    assert script.decode_qma9_context_run_escape_mask(encoded) == raw
    assert stats["copied_runs"] >= 2
    assert stats["escaped_model_update_pixels"] == stats["copied_pixels"]
    assert stats["model_update_policy"].startswith("copied runs update")


def test_context_run_escape_screen_emits_planning_manifest(tmp_path: Path) -> None:
    script = _load_script()
    frame = bytes([0, 0, 0, 0, 1, 1, 1, 1] * 4)
    raw = frame + frame
    qma9 = encode_qma9_mask(raw, frame_count=2, width=4, height=8)
    payload = qma9 + b"model" + b"pose" + b"router"
    archive = tmp_path / "archive.zip"
    _write_zip(archive, payload)
    constants = tmp_path / "inflate.py"
    constants.write_text(
        "\n".join(
            [
                f"RANGE_MASK_BYTES = {len(qma9)}",
                "SPLIT_MODEL_REORDERED_BYTES = 5",
                "POSE_STREAM_BYTES = 4",
                "ROUTER_ACTION_BYTES = 6",
                "PACKED_PAYLOAD_BYTES = RANGE_MASK_BYTES + SPLIT_MODEL_REORDERED_BYTES + POSE_STREAM_BYTES + ROUTER_ACTION_BYTES",
            ]
        )
    )

    manifest = script.build_context_run_escape_screen(
        archive_path=archive,
        split_constants_path=constants,
        output_dir=tmp_path / "out",
        candidate_id="qmc1_tiny",
        frames=2,
        min_run_length=4,
        require_left_context=False,
    )

    candidate_dir = tmp_path / "out" / "qmc1_tiny"
    assert manifest["score_claim"] is False
    assert manifest["dispatch_performed"] is False
    assert manifest["gpu_required"] is False
    assert manifest["decision"]["dispatchable"] is False
    assert manifest["decision"]["archive_relevant_state_change"] is True
    assert "no CUDA auth eval" in manifest["decision"]["non_dispatchable_reasons"]
    assert manifest["subset"]["decode_parity"] is True
    assert manifest["subset"]["raw_sha256"] == sha256_bytes(raw)
    assert manifest["context_run_escape"]["copied_runs"] > 0
    assert "linear projection" in manifest["full_stream_linear_projection"]["reason"]
    assert (candidate_dir / "manifest.json").exists()
    assert (candidate_dir / "candidate_subset.qmc1").stat().st_size == manifest["subset"]["candidate_qmc1_bytes"]
