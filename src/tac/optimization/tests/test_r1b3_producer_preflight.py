from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.r1b3_producer_preflight import (
    R1B2_EXTENSION_NAMES,
    R1B3ProducerError,
    audit_full_kernel_inputs,
    audit_production_receiver_binding,
    audit_rank4_strata,
    build_xi0_bundle,
    decode_xi0_payload,
    encode_xi0_payload,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_xi0_payload_roundtrip_is_canonical_and_corruption_refuses() -> None:
    source = np.linspace(-3.0, 3.0, 600, dtype=np.float32)
    payload = encode_xi0_payload(source)
    decoded = decode_xi0_payload(payload)
    assert decoded.dtype == np.dtype("float16")
    np.testing.assert_array_equal(decoded, source.astype(np.float16))
    assert encode_xi0_payload(decoded) == payload

    corrupted = bytearray(payload)
    corrupted[-5] ^= 1
    with pytest.raises(R1B3ProducerError, match=r"SHA|CRC|header"):
        decode_xi0_payload(bytes(corrupted))
    with pytest.raises(R1B3ProducerError, match="trailing bytes"):
        decode_xi0_payload(payload + b"x")


def test_build_xi0_bundle_uses_coordinate_zero_only(tmp_path: Path) -> None:
    poses = np.arange(600 * 6, dtype=np.float32).reshape(600, 6) / 1000.0
    cache = tmp_path / "gt.npz"
    np.savez(cache, gt_poses=poses)
    payload_path = tmp_path / "xi0.xi0"
    result = build_xi0_bundle(cache, payload_path=payload_path)
    assert result["manifest"]["coordinate_indices"] == [0]
    assert result["manifest"]["other_coordinates_counted"] == 0
    assert result["manifest"]["payload_path"] == str(payload_path.resolve())
    assert result["payload"]["bytes"] == 1500
    assert result["receiver_actuation_status"] == "ABSENT"
    np.testing.assert_array_equal(decode_xi0_payload(result["payload_bytes"]), poses[:, 0].astype(np.float16))


def _hard_batch(path: Path, pair: int, margin: float) -> None:
    _write_json(
        path,
        {
            "schema": "r2b_hard_oracle_batch.v1",
            "pair_start": pair,
            "pair_stop": pair + 1,
            "flip_count": 1,
            "flips": [[pair, 0, 0, 1, 2, margin]],
        },
    )


def test_rank4_strata_rederived_from_exact_batch_rows(tmp_path: Path) -> None:
    _hard_batch(tmp_path / "batch-0000.json", 0, 0.5)
    _hard_batch(tmp_path / "batch-0001.json", 1, 0.0005)
    result = audit_rank4_strata(
        tmp_path,
        pair_count=2,
        batch_size=1,
        expected_moderate=1,
        expected_tie=1,
    )
    assert result["moderate_margin_1e_3_to_1"] == 1
    assert result["tie_tight_lt_1e_3"] == 1
    assert result["other"] == 0
    assert [row["pair_index"] for row in result["per_pair"]] == [0, 1]


def test_rank4_strata_refuses_unregistered_other_margin(tmp_path: Path) -> None:
    _hard_batch(tmp_path / "batch-0000.json", 0, 1.0)
    with pytest.raises(R1B3ProducerError, match="stratum totals drifted"):
        audit_rank4_strata(
            tmp_path,
            pair_count=1,
            batch_size=1,
            expected_moderate=0,
            expected_tie=0,
        )


def test_rank4_strata_refuses_out_of_geometry_flip(tmp_path: Path) -> None:
    _hard_batch(tmp_path / "batch-0000.json", 0, 0.5)
    row = json.loads((tmp_path / "batch-0000.json").read_text())
    row["flips"][0][2] = 512
    _write_json(tmp_path / "batch-0000.json", row)
    with pytest.raises(R1B3ProducerError, match="value custody is malformed"):
        audit_rank4_strata(
            tmp_path,
            pair_count=1,
            batch_size=1,
            expected_moderate=1,
            expected_tie=0,
        )


def test_full_kernel_preflight_keeps_two_realization_definitions_separate(
    tmp_path: Path,
) -> None:
    full = tmp_path / "full.json"
    _write_json(
        full,
        {
            "schema": "resize_null_preimage_full_kernel_measurement.v1",
            "minimum_description": {"selected_full_kernel_frames": 0},
            "frame_rows": [{}],
        },
    )
    stream = tmp_path / "stream.r2b"
    stream.write_bytes(b"real compact stream")
    r2b = tmp_path / "r2b.json"
    _write_json(
        r2b,
        {
            "schema": "r2b_sparse_target_selection_receipt.v1",
            "kkt_stop_decisions": 0,
            "candidate_evaluation_decisions": 16_751,
            "candidate_recovered_score": 0.0012332316583976016,
            "baseline": {"flip_count": 17_926},
            "candidate": {"flip_count": 16_341},
            "curve": [{"scheduled_recovered_seg_score_upper_bound": 0.014199998643663194}],
            "stream": {
                "path": str(stream),
                "bytes": stream.stat().st_size,
                "sha256": hashlib.sha256(stream.read_bytes()).hexdigest(),
            },
        },
    )
    result = audit_full_kernel_inputs(full, r2b)
    assert result["measured_full_kernel_frames"] == 1
    assert result["selected_full_kernel_frames"] == 0
    assert result["r2b_hard_fixed_flips"] == 1_585
    assert result["r2b_decision_realization_fraction"] == pytest.approx(1_585 / 16_751)
    assert result["r2b_score_realization_fraction"] == pytest.approx(0.0012332316583976016 / 0.014199998643663194)
    assert "R1B3_P2_N600_FULL_KERNEL_MDL_SELECTION_ABSENT" in result["blockers"]


def test_production_receiver_binding_is_literal_and_fail_closed(tmp_path: Path) -> None:
    decoder = tmp_path / "inflate.py"
    parser_source = tmp_path / "parser.py"
    decoder.write_text("print('base decoder')\n", encoding="utf-8")
    parser_source.write_text("ARCHIVE_SCHEMA='c2.v1'\n", encoding="utf-8")
    refused = audit_production_receiver_binding(decoder, parser_source)
    assert refused["receiver_bound"] is False
    assert refused["blockers"] == [
        "R1B3_APPENDED_SECTIONS_ABSENT_FROM_PRODUCTION_DECODER",
        "R1B3_APPENDED_SECTIONS_REFUSED_BY_PRODUCTION_C2_PARSER",
    ]

    member_literals = "\n".join(repr(name) for name in R1B2_EXTENSION_NAMES)
    decoder.write_text(member_literals, encoding="utf-8")
    parser_source.write_text(
        "ARCHIVE_SCHEMA='r1b2_counted_archive.v1'\n" + member_literals,
        encoding="utf-8",
    )
    still_unproved = audit_production_receiver_binding(decoder, parser_source)
    assert still_unproved["literal_binding_present"] is True
    assert still_unproved["receiver_bound"] is False
    assert still_unproved["blockers"] == ["R1B3_PRODUCTION_RECEIVER_BEHAVIORAL_PROOF_ABSENT"]
