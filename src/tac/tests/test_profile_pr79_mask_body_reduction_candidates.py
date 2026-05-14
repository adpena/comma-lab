# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "profile_pr79_mask_body_reduction_candidates.py"


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("pr79_mask_body_profile_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_p3_payload_records_slice_lengths() -> None:
    script = _load_script()

    payload = script._build_p3_payload(  # noqa: SLF001
        mask_wire=b"mask-wire",
        renderer_wire=b"renderer",
        actions_wire=b"S2actions",
        pose_wire=b"pose",
    )

    assert payload.startswith(b"P3")
    mask_len, renderer_len, actions_len = struct.unpack_from("<IHH", payload, 2)
    assert (mask_len, renderer_len, actions_len) == (9, 8, 9)
    assert payload[10:] == b"mask-wirerendererS2actionspose"


def test_candidate_stream_rows_filters_by_source_mask_sha(tmp_path: Path) -> None:
    script = _load_script()
    matching_archive = tmp_path / "matching_archive.zip"
    stale_archive = tmp_path / "stale_archive.zip"
    with zipfile.ZipFile(matching_archive, "w") as zf:
        zf.writestr("masks.mkv", b"candidate-mask")
    with zipfile.ZipFile(stale_archive, "w") as zf:
        zf.writestr("masks.mkv", b"stale-mask")

    source_mask = b"source-mask"
    matching_manifest = tmp_path / "matching" / "protected_mask_reencode_manifest.json"
    matching_manifest.parent.mkdir()
    matching_manifest.write_text(
        json.dumps(
            {
                "archive": {"path": str(matching_archive)},
                "candidate_mask_stream": {"member": "masks.mkv"},
                "source_mask_stream": {"sha256": script._sha256_bytes(source_mask)},  # noqa: SLF001
            }
        )
    )
    stale_manifest = tmp_path / "stale" / "protected_mask_reencode_manifest.json"
    stale_manifest.parent.mkdir()
    stale_manifest.write_text(
        json.dumps(
            {
                "archive": {"path": str(stale_archive)},
                "candidate_mask_stream": {"member": "masks.mkv"},
                "source_mask_stream": {"sha256": "0" * 64},
            }
        )
    )

    source = type("Source", (), {"decoded": {"masks.mkv": source_mask}})()
    rows = script._candidate_stream_rows(  # noqa: SLF001
        source=source,
        manifest_paths=[matching_manifest, stale_manifest],
    )

    assert [row["status"] for row in rows] == ["stream_loaded"]
    assert rows[0]["candidate_mask_bytes"] == len(b"candidate-mask")
    assert rows[0]["candidate_mask_sha256"] == script._sha256_bytes(b"candidate-mask")  # noqa: SLF001


def test_select_recommendation_requires_three_kb_and_preflight() -> None:
    script = _load_script()

    recommendation = script._select_recommendation(  # noqa: SLF001
        [
            {
                "archive_bytes": 100_000,
                "candidate_id": "small",
                "delta_bytes_vs_pr79": -2_999,
                "exact_eval_ready_after_lane_claim": False,
            },
            {
                "archive_bytes": 99_000,
                "candidate_id": "ready",
                "delta_bytes_vs_pr79": -3_001,
                "exact_eval_ready_after_lane_claim": True,
            },
        ]
    )

    assert recommendation["decision"] == "recommend_exact_cuda_eval_after_lane_claim"
    assert recommendation["candidate"]["candidate_id"] == "ready"
