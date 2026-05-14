# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

from tac.qma9_range_mask_contract import encode_qma9_mask, sha256_bytes


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "profile_qma9_range_mask_bitstream.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("profile_qma9_range_mask_byte_search_test", SCRIPT)
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


def test_byte_search_accepts_pr84_style_no_router_profile_json(tmp_path: Path) -> None:
    script = _load_script()
    raw = bytes([0, 0, 1, 1, 0, 0, 1, 1] * 2)
    qma9 = encode_qma9_mask(raw, frame_count=2, width=2, height=4)
    payload = qma9 + b"model" + b"pose"
    archive = tmp_path / "archive.zip"
    _write_zip(archive, payload)
    profile_json = tmp_path / "profile.json"
    profile_json.write_text(
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

    manifest = script.build_byte_search_profile(
        archive_path=archive,
        split_constants_path=profile_json,
        output_dir=tmp_path / "out",
        frames=None,
        qmb1_block_widths=(2, 4),
    )

    assert manifest["score_claim"] is False
    assert manifest["dispatch_performed"] is False
    assert manifest["gpu_required"] is False
    assert manifest["segments"][-1]["name"] == "router_actions.3bit"
    assert manifest["segments"][-1]["size_bytes"] == 0
    assert manifest["raw_mask"]["raw_sha256"] == sha256_bytes(raw)
    assert manifest["mode_matrix"]["candidate_count"] == 6
    assert manifest["mode_matrix"]["qmf1_first_row_modes"] == [1, 2, 3]
    assert (tmp_path / "out" / "byte_search" / "qma9_range_mask_byte_search_profile.json").exists()

    by_id = {row["mode_id"]: row for row in manifest["candidates"]}
    baseline = by_id["qma9_reference_reencode"]
    assert baseline["role"] == "reference"
    assert baseline["raw_mask_parity"] is True
    assert baseline["accepted_for_exact_eval_candidate"] is False
    assert baseline["selectable"] is False
    assert baseline["no_op_status"] == "no_archive_relevant_state_change"
    assert "no_archive_relevant_state_change" in baseline["rejection_reasons"]
    assert "no_local_byte_screen_win" in baseline["rejection_reasons"]

    qmb1 = by_id["qmb1_vertical_block_escape_bw2"]
    assert qmb1["payload_bytes"] is not None
    assert qmb1["payload_sha256"]
    assert qmb1["raw_mask_parity"] is True
    assert "selectable" in qmb1
    assert "accepted_for_exact_eval_candidate" in qmb1
    assert "required_before_dispatch" in manifest["selection_guard"]

    qmf1 = by_id["qmf1_first_row_1_skip_static_up_gate_full_context"]
    assert qmf1["mode_family"] == "qmf1_first_row_specialization"
    assert qmf1["payload_bytes"] is not None
    assert qmf1["payload_sha256"]
    assert qmf1["raw_mask_parity"] is True
    assert qmf1["header"]["mode_name"] == "skip_static_up_gate_full_context"
    assert qmf1["specialization"]["first_row_pixels"] == 8


def test_build_profile_can_embed_prefix_byte_search_without_cpp(tmp_path: Path) -> None:
    script = _load_script()
    raw = bytes([0, 1, 2, 3, 4, 0, 1, 2])
    qma9 = encode_qma9_mask(raw, frame_count=1, width=2, height=4)
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

    profile = script.build_profile(
        archive_path=archive,
        split_constants_path=constants,
        output_dir=tmp_path / "out",
        cpp_profiler=SCRIPT,
        pure_python_max_pixels=len(raw),
        checkpoint_pixels=(0,),
        skip_cpp_full=True,
        cpp_timeout_seconds=1,
        run_byte_search=True,
        byte_search_frames=1,
        qmb1_block_widths=(2,),
        qmf1_first_row_modes=(1,),
        raw_mask_path=None,
    )

    assert profile["byte_search"]["raw_mask"]["frames"] == 1
    assert profile["byte_search"]["raw_mask"]["raw_sha256"] == sha256_bytes(raw)
    assert profile["byte_search"]["candidates"][0]["mode_id"] == "qma9_reference_reencode"
    assert profile["byte_search"]["mode_matrix"]["qmf1_first_row_modes"] == [1]
    assert profile["byte_search"]["selection_guard"]["safe_to_choose_exact_eval_candidate_after_local_screen_win"] in {
        True,
        False,
    }
