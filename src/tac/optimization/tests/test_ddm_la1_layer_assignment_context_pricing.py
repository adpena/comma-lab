from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from tac.optimization.ddm_la1_layer_assignment_context_pricing import (
    LA1PricingError,
    _decode_explicit,
    _encode_explicit,
    _extract_streams,
    materialize,
    race_stream,
)

REPO = Path(__file__).resolve().parents[4]
C1 = (
    REPO
    / ".omx/research/ddm_v15_scorer_solved_templates_n600_20260723T013000Z"
    / "ddm_v15_solved_templates_n600.not_a_candidate.zip.receipt-bytes"
)


@pytest.mark.parametrize("codec", ["RAW_EXPLICIT", "BROTLI_Q11", "RAW_LZMA1"])
def test_explicit_uniform_frames_are_exact_and_canonical(codec: str) -> None:
    raw = (b"lane:road:movable:" * 19) + bytes(range(32))
    frame = _encode_explicit(raw, codec)
    assert len(frame) >= 46
    assert _decode_explicit(frame) == raw
    assert _encode_explicit(_decode_explicit(frame), codec) == frame


def test_context_race_uses_uniform_framing_and_scoped_winner() -> None:
    row = race_stream(
        "receiver_realization_profile",
        b"DDRP1\x01\x00\x99\x00\x0b\x03\x09\x33\xff\xcc\x6b\x00\x72\x3f\x48\x01\x02\x03",
        current_home_bytes=85,
    )
    assert row["schema"] == "ddm_la1_residual_context_race.v1"
    assert row["parseback_exact_all_arms"] is True
    assert {arm["ownership"] for arm in row["arms"]} == {"RESIDUAL", "CONTEXT"}
    assert all(arm["header_bytes"] == 46 for arm in row["arms"])
    assert row["verdict_scope"].startswith("INSTANCE:")


def test_extract_exact_c1_homes_and_separate_lane_payload() -> None:
    base = C1.read_bytes()
    lane_payload = bytes(range(90))
    rows = _extract_streams(base, lane_payload)
    assert {name: len(payload) for name, payload in rows.items()} == {
        "manifest": 3302,
        "v15_predictor_zip_outer": 100056,
        "g1_movable_worldsheet_outer": 29810,
        "receiver_realization_profile": 23,
        "solved_template_outer": 86,
        "central_directory": 383,
        "lane_seed": 90,
    }
    with zipfile.ZipFile(io.BytesIO(base)) as archive:
        assert rows["central_directory"] == base[archive.start_dir :]


def test_extract_rejects_wrong_lane_payload_size() -> None:
    with pytest.raises(LA1PricingError, match="lane_seed"):
        _extract_streams(C1.read_bytes(), b"wrong")


def test_materialize_refuses_transient_output_outside_repository(
    tmp_path: Path,
) -> None:
    config = REPO / ".omx/research/configs" / "ddm_la1_layer_assignment_context_pricing_20260725.json"
    with pytest.raises(LA1PricingError, match="output directory"):
        materialize(config, tmp_path)
