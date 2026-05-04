from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import brotli
import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_pr77_tile_delta_transfer_candidates.py"
PR77_ARCHIVE = REPO / "experiments/results/top_submission_reverse_engineering_20260503_pr77/archive.zip"
PR79_S2_ARCHIVE = (
    REPO
    / "experiments/results/pr79_action_dictionary_repack_v2_20260503_codex/"
    "pr79_s2_fixed_adaptive_actions/archive.zip"
)


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("pr77_tile_delta_transfer_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stored_zip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        info = zipfile.ZipInfo("p", (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        info.create_system = 3
        zf.writestr(info, payload)


def _raw4(records: list[tuple[int, int, int]]) -> bytes:
    out = bytearray()
    for pair, tile, action in records:
        out += int(pair).to_bytes(2, "little") + bytes([tile, action])
    return bytes(out)


def _p3_payload(action_wire: bytes) -> bytes:
    mask_br = brotli.compress(b"\x12\x00\x0a\x0a" + b"m" * 32, quality=0)
    renderer_br = brotli.compress(b"QZS3" + b"r" * 32, quality=0)
    pose_br = brotli.compress(b"QP1" + (5120).to_bytes(2, "little") + b"p" * 8, quality=0)
    return (
        b"P3"
        + struct.pack("<IHH", len(mask_br), len(renderer_br), len(action_wire))
        + mask_br
        + renderer_br
        + action_wire
        + pose_br
    )


def test_transfer_builder_writes_only_archive_closed_non_noop_candidates(tmp_path: Path) -> None:
    script = _load_script()
    pr77_raw = _raw4([(3, 82, 2), (9, 140, 7)])
    pr79_raw = _raw4([(1, 82, 2), (2, 83, 3), (5, 84, 4)])
    pr77_s2 = script.S2.encode_s2_adaptive_actions(pr77_raw)["wire"]
    pr79_s2 = script.S2.encode_s2_adaptive_actions(pr79_raw)["wire"]
    pr77_archive = tmp_path / "pr77.zip"
    pr79_archive = tmp_path / "pr79_s2.zip"
    _stored_zip(pr77_archive, _p3_payload(pr77_s2))
    _stored_zip(pr79_archive, _p3_payload(pr79_s2))

    script.PR77_EXPECTED_ACTION_WIRE_BYTES = len(pr77_s2)
    script.PR77_EXPECTED_RECORD_COUNT = 2
    matrix = script.build_candidates(
        pr77_archive=pr77_archive,
        pr79_s2_archive=pr79_archive,
        pr79_s2_eval=None,
        raw_output_parity=None,
        output_dir=tmp_path / "out",
        policies=["replace_pr79_with_pr77_all"],
    )

    assert matrix["score_claim"] is False
    assert matrix["candidate_count"] == 1
    row = matrix["candidates"][0]
    assert row["changed_members"] == ["seg_tile_actions.bin"]
    assert row["dispatch_recommendation"]["dispatch_ready_now"] is False
    assert row["break_even_vs_pr79_s2"]["baseline_score"] == script.PR79_S2_SCORE
    manifest = json.loads(Path(row["manifest_path"]).read_text())
    assert manifest["runtime_parse_validation"]["non_action_streams_preserved"] is True
    assert (
        manifest["runtime_parse_validation"]["no_op_status"]
        == "changes_pr79_s2_action_record_set"
    )
    assert manifest["runtime_parse_validation"]["action_record_multiset_equal_to_pr77"] is True
    assert manifest["raw_output_proof"]["candidate_raw_output_proof_available"] is False
    with zipfile.ZipFile(row["archive_path"]) as zf:
        infos = zf.infolist()
    assert [info.filename for info in infos] == ["p"]
    assert infos[0].date_time == (1980, 1, 1, 0, 0, 0)
    assert infos[0].compress_type == zipfile.ZIP_STORED


def test_transfer_builder_skips_pr79_noop_action_sets(tmp_path: Path) -> None:
    script = _load_script()
    raw = _raw4([(3, 82, 2), (9, 140, 7)])
    s2 = script.S2.encode_s2_adaptive_actions(raw)["wire"]
    pr77_archive = tmp_path / "pr77.zip"
    pr79_archive = tmp_path / "pr79_s2.zip"
    _stored_zip(pr77_archive, _p3_payload(s2))
    _stored_zip(pr79_archive, _p3_payload(s2))

    script.PR77_EXPECTED_ACTION_WIRE_BYTES = len(s2)
    script.PR77_EXPECTED_RECORD_COUNT = 2
    matrix = script.build_candidates(
        pr77_archive=pr77_archive,
        pr79_s2_archive=pr79_archive,
        pr79_s2_eval=None,
        raw_output_parity=None,
        output_dir=tmp_path / "out",
        policies=["replace_pr79_with_pr77_all"],
    )

    assert matrix["candidate_count"] == 0
    assert matrix["skipped_policies"][0]["status"] == "skipped_not_closed_or_noop"
    assert "PR79 no-op" in matrix["skipped_policies"][0]["reason"]


@pytest.mark.skipif(
    not (PR77_ARCHIVE.exists() and PR79_S2_ARCHIVE.exists()),
    reason="public PR77/PR79 S2 cached archives missing",
)
def test_transfer_builder_real_cached_archives_emit_p3_candidate(tmp_path: Path) -> None:
    script = _load_script()
    matrix = script.build_candidates(
        pr77_archive=PR77_ARCHIVE,
        pr79_s2_archive=PR79_S2_ARCHIVE,
        pr79_s2_eval=None,
        raw_output_parity=None,
        output_dir=tmp_path,
        policies=["replace_pr79_with_pr77_all"],
    )

    assert matrix["candidate_count"] == 1
    best = matrix["candidates"][0]
    assert best["selected_record_count"] == 147
    assert best["s2_action_wire_bytes"] < 325
    assert best["archive_bytes"] < 277_321
