# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

from tac.qma9_range_mask_contract import encode_qma9_mask, sha256_bytes


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_qma9_context_backoff_candidate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_qma9_context_backoff_candidate_test", SCRIPT)
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


def test_context_backoff_roundtrips_without_side_header() -> None:
    script = _load_script()
    raw = bytes(
        [
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            2,
            2,
            3,
            3,
            2,
            2,
            3,
            3,
        ]
    )

    payload, stats = script.encode_qma9_context_backoff_mask(
        raw,
        frame_count=1,
        width=4,
        height=4,
        mode_id="always_lu",
    )

    assert payload[:4] == b"QMA9"
    assert script.CONTEXT_BACKOFF_HEADER.size == 20
    assert script.decode_qma9_context_backoff_mask(payload, mode_id="always_lu") == raw
    assert stats["backoff_pixels"] == len(raw)
    assert stats["context_family_counts"] == {"lu": len(raw)}


def test_full9_mode_is_rejected_as_source_preserving_no_op() -> None:
    script = _load_script()
    raw = bytes([0, 1, 2, 3, 4, 0])
    baseline = encode_qma9_mask(raw, frame_count=1, width=2, height=3)
    candidate, stats = script.encode_qma9_context_backoff_mask(
        raw,
        frame_count=1,
        width=2,
        height=3,
        mode_id="always_full9",
    )

    record = script._candidate_record(
        mode_id="always_full9",
        candidate_payload=candidate,
        baseline_payload=baseline,
        raw=raw,
        decoded=script.decode_qma9_context_backoff_mask(candidate, mode_id="always_full9"),
        source_archive_bytes=123,
        stats=stats,
        path=None,
    )

    assert candidate == baseline
    assert record["raw_mask_parity"] is True
    assert record["archive_relevant_state_change"] is False
    assert record["selectable_for_local_followup"] is False
    assert "no_op_or_source_preserving_transform" in record["rejection_reasons"]


def test_context_backoff_screen_emits_planning_manifest_for_pr84_shape(tmp_path: Path) -> None:
    script = _load_script()
    raw = bytes([0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3])
    qma9 = encode_qma9_mask(raw, frame_count=1, width=4, height=4)
    payload = qma9 + b"model" + b"pose"
    archive = tmp_path / "archive.zip"
    _write_zip(archive, payload)
    constants = tmp_path / "inflate.py"
    constants.write_text(
        "\n".join(
            [
                f"RANGE_MASK_BYTES = {len(qma9)}",
                "SPLIT_MODEL_REORDERED_BYTES = 5",
                "POSE_STREAM_BYTES = 4",
                "ROUTER_ACTION_BYTES = 0",
                "PACKED_PAYLOAD_BYTES = RANGE_MASK_BYTES + SPLIT_MODEL_REORDERED_BYTES + POSE_STREAM_BYTES",
            ]
        ),
        encoding="utf-8",
    )

    manifest = script.build_context_backoff_screen(
        archive_path=archive,
        split_constants_path=constants,
        output_dir=tmp_path / "out",
        candidate_id="qma9cb_tiny",
        frames=1,
        modes=("always_lu", "always_full9"),
    )

    screen_dir = tmp_path / "out" / "qma9cb_tiny"
    assert manifest["planning_only"] is True
    assert manifest["score_claim"] is False
    assert manifest["dispatch_performed"] is False
    assert manifest["gpu_required"] is False
    assert manifest["baseline"]["payload_bytes"] == len(qma9)
    assert manifest["baseline"]["raw_mask_parity"] is True
    assert manifest["raw_mask"]["raw_sha256"] == sha256_bytes(raw)
    assert manifest["mode_matrix"]["candidate_count"] == 2
    assert {row["mode_id"] for row in manifest["candidates"]} == {"always_lu", "always_full9"}
    assert all(row["safe_for_remote_dispatch"] is False for row in manifest["candidates"])
    assert any(
        "no_op_or_source_preserving_transform" in row["rejection_reasons"]
        for row in manifest["candidates"]
        if row["mode_id"] == "always_full9"
    )
    assert (screen_dir / "manifest.json").exists()
    assert (screen_dir / "baseline.qma9").read_bytes() == qma9
