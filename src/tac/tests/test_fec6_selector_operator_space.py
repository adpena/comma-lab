# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from tac.fec6_selector_operator_space import (
    FEC6_FIXED_K16_MODE_IDS,
    build_fec6_selector_operator_space,
    decode_fec6_fixed_huffman_codes,
    encode_fec6_fixed_huffman_codes,
    parse_fec6_selector_archive,
)


def _write_zip(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("x")
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _fp11_fec6_payload(codes: tuple[int, ...]) -> bytes:
    source = b"source-payload"
    encoded, _ = encode_fec6_fixed_huffman_codes(codes)
    selector = b"FEC6" + len(codes).to_bytes(2, "little") + encoded
    return (
        b"FP11"
        + len(source).to_bytes(4, "little")
        + source
        + len(selector).to_bytes(2, "little")
        + selector
    )


def _write_pair_rows(path: Path) -> None:
    rows = [
        {
            "pair": 0,
            "mode_id": "none",
            "component_score_no_rate": 0.100,
            "posenet_dist": 0.000030,
            "segnet_dist": 0.00060,
        },
        {
            "pair": 0,
            "mode_id": "frame0_rgb_bias_p2_m1_m1",
            "component_score_no_rate": 0.090,
            "posenet_dist": 0.000025,
            "segnet_dist": 0.00060,
        },
        {
            "pair": 1,
            "mode_id": "frame0_blue_chroma_amp_3",
            "component_score_no_rate": 0.120,
            "posenet_dist": 0.000035,
            "segnet_dist": 0.00070,
        },
        {
            "pair": 1,
            "mode_id": "none",
            "component_score_no_rate": 0.125,
            "posenet_dist": 0.000037,
            "segnet_dist": 0.00070,
        },
    ]
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_fec6_fixed_huffman_roundtrip() -> None:
    codes = (0, 2, 13, 4, 7, 15)
    encoded, bit_count = encode_fec6_fixed_huffman_codes(codes)
    decoded, decoded_bits = decode_fec6_fixed_huffman_codes(encoded, n_pairs=len(codes))

    assert decoded == codes
    assert decoded_bits == bit_count


def test_parse_fec6_selector_archive(tmp_path: Path) -> None:
    archive = tmp_path / "fec6.zip"
    codes = (0, 2, 13, 4)
    _write_zip(archive, _fp11_fec6_payload(codes))

    parsed = parse_fec6_selector_archive(archive, repo_root=tmp_path)

    assert parsed.archive_path == "fec6.zip"
    assert parsed.member_name == "x"
    assert parsed.n_pairs == 4
    assert parsed.codes == codes
    assert parsed.histogram == {"0": 1, "2": 1, "4": 1, "13": 1}
    assert parsed.to_manifest()["mode_histogram"]["none"] == 1


def test_selector_operator_space_is_fail_closed_and_grammar_level(tmp_path: Path) -> None:
    archive = tmp_path / "fec6.zip"
    pair_rows = tmp_path / "pair_rows.jsonl"
    none = FEC6_FIXED_K16_MODE_IDS.index("none")
    blue3 = FEC6_FIXED_K16_MODE_IDS.index("frame0_blue_chroma_amp_3")
    _write_zip(archive, _fp11_fec6_payload((none, blue3)))
    _write_pair_rows(pair_rows)

    manifest = build_fec6_selector_operator_space(
        fec6_archive=archive,
        pair_component_rows_paths=(pair_rows,),
        repo_root=tmp_path,
        current_cpu_score=0.1920513168811056,
        target_cpu_score=0.192,
    )

    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["operator_row_count"] == 2
    assert manifest["raw_archive_byte_rows_emitted"] == 0
    row = manifest["top_proxy_improving_rows"][0]
    assert row["mutation_grain"] == "grammar_aware_selector_symbol"
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert "packet_candidate_not_materialized" in row["blockers"]
    assert (
        manifest["score_threshold"][
            "required_rate_bytes_to_strictly_cross_target_if_components_unchanged"
        ]
        == 78
    )
