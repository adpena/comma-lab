"""Mechanics tests for the dense-free 600-pair H0 solution index."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.ddm_lattice_teacher_solution_index import (
    MS2R_RECEIVER_CONTRACT,
    LatticeTeacherIndexError,
    TeacherAssetSpec,
    build_solution_index,
    scan_selected_packet,
)
from tac.witness_dsl.v10_production_receiver import PREDICTOR_RESIDUAL_Y_CODEC_ID, build_packet


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes, role: str) -> TeacherAssetSpec:
    path.write_bytes(payload)
    return TeacherAssetSpec(
        path=path.resolve(),
        sha256=_sha(payload),
        bytes=len(payload),
        role=role,
    )


def _json_asset(path: Path, value: object, role: str) -> TeacherAssetSpec:
    return _write(path, json.dumps(value, sort_keys=True).encode("utf-8"), role)


def _fixture(tmp_path: Path) -> dict[str, TeacherAssetSpec]:
    pair_count = 600
    frame0 = np.arange(pair_count * 3, dtype=np.uint16).astype(np.uint8).reshape(pair_count, 1, 1, 3)
    frame1 = np.ascontiguousarray(frame0 + 7)
    packet_bytes = build_packet(
        frame1,
        camera_height=2,
        camera_width=2,
        y_codec_id=PREDICTOR_RESIDUAL_Y_CODEC_ID,
        frame0_y_planes=frame0,
    )
    packet = _write(tmp_path / "0.bin", packet_bytes, "selected_packet")
    scan = scan_selected_packet(packet)
    steps = [4] * 208 + [8] * 392
    selected_rows = [
        {
            "pair_id": record.pair_id,
            "selected_step": steps[record.pair_id],
            "record_bytes": (148 + record.bootstrap_bytes + record.descriptor_bytes + record.residual_bytes),
        }
        for record in scan.records
    ]
    ms2r = {
        "schema": "ddm_ms2r_tolerance_capped_solve_r2_receipt.v1",
        "authority": {
            "research_only": True,
            "execution_allowed": False,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
        },
        "homotopy": {
            "solve": {
                "selected_steps": steps,
                "q4_pair_count": 208,
                "q8_pair_count": 392,
                "rows_sha256": "a" * 64,
            },
            "candidate": {
                "predictor": {
                    "bytes": packet.bytes,
                    "sha256": packet.sha256,
                },
                "receiver_contract": MS2R_RECEIVER_CONTRACT,
                "score_claim": False,
                "selected_record_rows": selected_rows,
            },
        },
    }
    sense_lines = []
    for pair_id in range(pair_count):
        sense_lines.append(
            json.dumps(
                {
                    "schema": "ddm_min_description_lattice_sense_pair.v1",
                    "pair_id": pair_id,
                    "research_only": True,
                    "execution_allowed": False,
                    "promotion_eligible": False,
                    "score_claim": False,
                    "origin_sha256": "1" * 64,
                    "selected_sha256": "2" * 64,
                    "residual_sha256": "3" * 64,
                    "rate": {
                        "canonical_member_bytes": 10,
                        "selected_residual_bytes": 9,
                        "delta_bytes": -1,
                    },
                },
                sort_keys=True,
            )
        )
    sense = _write(
        tmp_path / "sense.jsonl",
        ("\n".join(sense_lines) + "\n").encode("utf-8"),
        "sense",
    )
    factorization = _json_asset(
        tmp_path / "factorization.json",
        {
            "schema": "ddm_min_description_lattice_sense_factorization.v1",
            "pair_count": 600,
            "execution_allowed": False,
            "research_only": True,
            "score_claim": False,
            "matrix_sha256": "4" * 64,
            "admitted_factor_count": 6,
        },
        "factorization",
    )
    ms1 = {
        "schema": "ddm_min_description_lattice_solve_receipt.v1",
        "execution_allowed": False,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "sense": {
            "pair_jsonl": {"bytes": sense.bytes, "sha256": sense.sha256},
            "factorization": {
                "bytes": factorization.bytes,
                "sha256": factorization.sha256,
            },
        },
        "factorization_summary": {
            "pair_count": 600,
            "admitted_factor_count": 6,
            "matrix_sha256": "4" * 64,
        },
    }
    return {
        "selected_packet": packet,
        "ms2r_receipt": _json_asset(tmp_path / "ms2r.json", ms2r, "ms2r"),
        "ms1_receipt": _json_asset(tmp_path / "ms1.json", ms1, "ms1"),
        "ms1_sense_rows": sense,
        "ms1_factorization": factorization,
    }


def test_h0_indexes_all_600_pairs_without_retaining_teacher_bytes(tmp_path: Path) -> None:
    assets = _fixture(tmp_path)
    index = build_solution_index(**assets)
    assert index["pair_count"] == 600
    assert index["q4_pair_count"] == 208
    assert index["q8_pair_count"] == 392
    assert index["dense_teacher_bytes_persisted"] == 0
    assert index["candidate_payload_created"] is False
    assert len(index["pairs"]) == 600
    assert index["packet"]["decoder_peak_population_rows"] == 1


def test_h0_refuses_packet_custody_drift(tmp_path: Path) -> None:
    assets = _fixture(tmp_path)
    packet = assets["selected_packet"]
    packet.path.write_bytes(packet.path.read_bytes() + b"x")
    with pytest.raises(LatticeTeacherIndexError, match="byte custody drift"):
        build_solution_index(**assets)
