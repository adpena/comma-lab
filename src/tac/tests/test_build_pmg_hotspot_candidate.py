# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_pmg_hotspot_candidate.py"
CMG3_BUILDER_PATH = REPO / "experiments" / "build_cmg3_rowspan_candidate.py"
INFLATE_PATH = REPO / "submissions" / "robust_current" / "inflate_renderer.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_pmg_hotspot_payload_roundtrips_through_cmg3_runtime(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "_pmg_hotspot_builder_payload_test")
    cmg3 = _load(CMG3_BUILDER_PATH, "_pmg_hotspot_cmg3_payload_test")
    inflate = _load(INFLATE_PATH, "_pmg_hotspot_inflate_payload_test")

    source = np.zeros((2, 384, 512), dtype=np.uint8)
    source[:, 100:180, 160:300] = 2
    source[0, 140, 200:240] = 3
    spans = cmg3.row_spans(source, row_stride=4)
    policy = {
        "row_stride": 4,
        "default_class": 0,
        "row_fill": "nearest",
        "draw_order": [0, 1, 2, 3, 4],
    }
    base = cmg3.reconstruct_row_spans(
        spans,
        row_stride=4,
        default_class=0,
        row_fill="nearest",
        draw_order=(0, 1, 2, 3, 4),
    )
    residual_records = [(0, 140, 200, 240, 3), (1, 101, 160, 300, 4)]
    expected = builder.apply_residual_records(base, residual_records)

    payload, header = builder.encode_pmg_hotspot_payload(
        spans=spans,
        residual_records=residual_records,
        policy=policy,
        source_mask_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
        base_reconstructed_sha256=hashlib.sha256(base.tobytes()).hexdigest(),
        final_reconstructed_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
        plan_sha256="0" * 64,
        candidate_id="unit_test",
        compressor="raw",
    )
    path = tmp_path / "masks.cmg3"
    path.write_bytes(payload)

    decoded = inflate._load_masks_from_cmg3(path, expected_frames=2)

    np.testing.assert_array_equal(decoded.numpy(), expected)
    assert header["mode"] == builder.MODE
    assert header["residual_record_count"] == 2
    assert header["residual_record_bytes"] == 2 * builder.RESIDUAL_RECORD_STRUCT.size
    assert header["score_claim"] is False


def test_pmg_hotspot_rejects_overlapping_records() -> None:
    builder = _load(BUILDER_PATH, "_pmg_hotspot_builder_overlap_test")
    candidate = {
        "protected_atoms": {
            "selected_atoms": [
                {
                    "identity": {
                        "frame_index": 0,
                        "y": 10,
                        "x0": 5,
                        "x1_exclusive": 20,
                        "source_class": 2,
                    }
                },
                {
                    "identity": {
                        "frame_index": 0,
                        "y": 10,
                        "x0": 10,
                        "x1_exclusive": 30,
                        "source_class": 3,
                    }
                },
            ]
        }
    }

    try:
        builder.residual_records_from_candidate(candidate)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("overlapping PMG-HOTSPOT residual records were accepted")


def test_pmg_hotspot_builder_defaults_to_lzma_xz() -> None:
    builder = _load(BUILDER_PATH, "_pmg_hotspot_builder_defaults_test")
    args = builder.parse_args(
        [
            "--plan-json",
            "plan.json",
            "--frontier-archive",
            "archive.zip",
            "--output-dir",
            "out",
        ]
    )

    assert args.compressor == "lzma_xz"


def test_pmg_hotspot_pair_protection_repairs_half_frame_pair() -> None:
    builder = _load(BUILDER_PATH, "_pmg_hotspot_builder_pair_protection_test")

    source = np.zeros((4, 384, 512), dtype=np.uint8)
    source[2, 8, 10:14] = 1
    source[2, 8, 16:20] = 2
    source[2, 9, 100:103] = 3
    current = source.copy()
    current[2, 8, 10:20] = 0
    current[2, 9, 100:103] = 0
    current[3, 8, 10:20] = 0

    records = builder.residual_records_for_protected_pairs(
        source=source,
        current=current,
        pair_indices=(2,),
    )
    repaired = builder.apply_residual_records(current, records)

    assert builder.parse_pair_indices("2, 2, 1") == (1, 2)
    assert records == [
        (2, 8, 10, 14, 1),
        (2, 8, 16, 20, 2),
        (2, 9, 100, 103, 3),
    ]
    np.testing.assert_array_equal(repaired[2], source[2])
    np.testing.assert_array_equal(repaired[3], current[3])


def test_pmg_hotspot_atom_ledger_selection_is_charged_and_verified(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "_pmg_hotspot_builder_atom_ledger_test")
    source_sha = "a" * 64
    ledger = {
        "schema": "cmg3_pixel_lagrangian_atom_ledger_v1",
        "score_claim": False,
        "evidence_grade": "planning_only",
        "atom_count": 3,
        "inputs": {
            "source_mask_array": {"tensor_sha256": source_sha},
            "candidate": {"tensor_sha256": "b" * 64},
        },
        "top_atoms": [
            {
                "atom_id": "row-a",
                "atom_family": "row_run",
                "identity": {
                    "frame_index": 2,
                    "y": 10,
                    "x0": 5,
                    "x1_exclusive": 9,
                    "class_id": 3,
                },
            },
            {
                "atom_id": "pair-skip",
                "atom_family": "pair",
                "identity": {"pair_index": 1},
            },
            {
                "atom_id": "row-b",
                "atom_family": "row_run",
                "identity": {
                    "frame_index": 3,
                    "y": 11,
                    "x0": 0,
                    "x1_exclusive": 2,
                    "source_class": 1,
                },
            },
        ],
    }
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger, sort_keys=True) + "\n")

    records, report = builder.residual_records_from_atom_ledger(
        ledger_path,
        atom_count=2,
        expected_source_mask_sha256=source_sha,
    )

    assert records == [(2, 10, 5, 9, 3), (3, 11, 0, 2, 1)]
    assert report["score_claim"] is False
    assert report["selected_row_run_atom_count"] == 2
    assert report["skipped_non_row_run_top_atoms"] == 1
    assert report["selected_residual_pixels_touched"] == 6
    assert report["selected_atom_id_prefix"] == ["row-a", "row-b"]


def test_pmg_hotspot_atom_ledger_source_sha_must_match(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "_pmg_hotspot_builder_atom_ledger_sha_test")
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "schema": "cmg3_pixel_lagrangian_atom_ledger_v1",
                "score_claim": False,
                "evidence_grade": "planning_only",
                "inputs": {"source_mask_array": {"tensor_sha256": "a" * 64}},
                "top_atoms": [],
            },
            sort_keys=True,
        )
        + "\n"
    )

    try:
        builder.residual_records_from_atom_ledger(
            ledger_path,
            atom_count=1,
            expected_source_mask_sha256="b" * 64,
        )
    except ValueError as exc:
        assert "source tensor SHA mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected source SHA guard to fail")
