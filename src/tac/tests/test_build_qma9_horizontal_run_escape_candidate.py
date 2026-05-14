# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

from tac.qma9_range_mask_contract import encode_qma9_mask, sha256_bytes


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_qma9_horizontal_run_escape_candidate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_qma9_horizontal_run_escape_candidate_test", SCRIPT)
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


def test_qmh1_horizontal_run_escape_roundtrips_transition_tails() -> None:
    script = _load_script()
    row_a = bytes([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    row_b = bytes([2, 2, 2, 2, 3, 3, 3, 3, 4, 4])
    raw = row_a + row_b

    encoded, stats = script.encode_qma9_horizontal_run_escape_mask(
        raw,
        frame_count=1,
        width=2,
        height=10,
        min_run_length=3,
        require_up_disagreement=False,
    )

    assert encoded[:4] == b"QMH1"
    assert script.decode_qma9_horizontal_run_escape_mask(encoded) == raw
    assert stats["escaped_runs"] >= 3
    assert stats["copied_pixels"] >= 12
    assert stats["model_update_policy"].startswith("escaped horizontal tails update")


def test_horizontal_run_escape_screen_emits_planning_manifest_for_pr84_style_payload(tmp_path: Path) -> None:
    script = _load_script()
    frame = bytes([0, 0, 0, 0, 0, 1, 1, 1] * 3)
    raw = frame + frame
    qma9 = encode_qma9_mask(raw, frame_count=2, width=3, height=8)
    payload = qma9 + b"model" + b"pose"
    archive = tmp_path / "archive.zip"
    _write_zip(archive, payload)
    constants = tmp_path / "profile.json"
    constants.write_text(
        json.dumps(
            {
                "split_constants": {
                    "RANGE_MASK_BYTES": len(qma9),
                    "SPLIT_MODEL_REORDERED_BYTES": 5,
                    "POSE_STREAM_BYTES": 4,
                    "ROUTER_ACTION_BYTES": 6,
                    "PACKED_PAYLOAD_BYTES": len(qma9) + 15,
                }
            }
        )
    )

    manifest = script.build_horizontal_run_escape_screen(
        archive_path=archive,
        split_constants_path=constants,
        output_dir=tmp_path / "out",
        candidate_id="qmh1_tiny",
        frames=2,
        min_run_length=3,
        require_up_disagreement=False,
    )

    candidate_dir = tmp_path / "out" / "qmh1_tiny"
    assert manifest["score_claim"] is False
    assert manifest["dispatch_performed"] is False
    assert manifest["gpu_required"] is False
    assert manifest["decision"]["dispatchable"] is False
    assert manifest["decision"]["dispatch_gate"] == "planning_only/no_remote_dispatch"
    assert "worker scope forbids remote dispatch" in manifest["decision"]["non_dispatchable_reasons"]
    assert manifest["subset"]["decode_parity"] is True
    assert manifest["subset"]["raw_sha256"] == sha256_bytes(raw)
    assert manifest["candidate_qmh1_header"]["magic"] == "QMH1"
    assert manifest["horizontal_run_escape"]["escaped_runs"] > 0
    assert manifest["segments"][-1]["name"] == "router_actions.3bit"
    assert manifest["segments"][-1]["size_bytes"] == 0
    assert "full 600-frame QMH1" in manifest["decision"]["required_gates_before_dispatch"][0]
    assert (candidate_dir / "manifest.json").exists()
    assert (candidate_dir / "candidate_subset.qmh1").stat().st_size == manifest["subset"]["candidate_qmh1_bytes"]
