# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from tac.optimization import predictor_r4_tailrace as r4
from tac.optimization.predictor_upgrade_xi_chart import CLASS_NAMES, STRATA


def _baseline_fixtures() -> tuple[dict, dict]:
    r3_rows = []
    r2_rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        for stratum_id, stratum in enumerate(STRATA):
            selected = class_id == 0 and stratum_id == 0
            r3_rows.append(
                {
                    "class_name": class_name,
                    "stratum": stratum,
                    "admitted_bytes": 7 if selected else 0,
                    "admitted_corrected_misses": 11 if selected else 0,
                    "eaten_bytes": 1234 if selected else 0,
                    "eaten_misses": r4.R3_EATEN_LEDGER_MISSES if selected else 0,
                }
            )
            r2_rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "stratum": stratum,
                    "kinds": {
                        "scattered_incoherent": {
                            "count": r4.R2_SCATTERED_MISSES if selected else 0,
                        }
                    },
                }
            )
    return (
        {"D4_composed_curve_v3": {"per_class_per_stratum": r3_rows}},
        {"D1_miss_structure": {"n600": {"rows": r2_rows}}},
    )


def test_d1_baseline_reconciles_complete_tail_and_marks_partial_literal() -> None:
    r3, r2 = _baseline_fixtures()
    rows = r4.build_d1_baseline_rows(r3, r2)
    assert len(rows) == len(CLASS_NAMES) * len(STRATA)
    assert sum(row["tail_sites"] for row in rows) == r4.R3_KNEE_MISSES
    road = rows[0]
    assert road["literal_equal_fidelity_for_full_tail"] is False
    assert road["generator_strict_entry_bar_bytes"] == road["eaten_lambda_equivalent_bytes"]


def test_tail_reconstruction_subtracts_admitted_component_across_strata(tmp_path) -> None:
    truth = np.zeros((64, 2, 2), dtype=np.uint8)
    cache = tmp_path / "cache.npz"
    np.savez(cache, lstars=truth)
    predicted = truth.copy()
    predicted[0, 0, 1] = 1
    strata = np.zeros_like(truth)
    strata[0, 0, 1] = 1
    chunk_dir = tmp_path / "r2" / "n64" / "chunks"
    chunk_dir.mkdir(parents=True)
    np.savez(chunk_dir / "chunk_000.npz", predicted=predicted, strata=strata)

    residual_raw = r4._RESIDUAL_RECORD.pack(0, 1, 1, 0)
    residual_blob = zlib.compress(residual_raw, 9)
    residual_path = tmp_path / "residual.zlib"
    residual_path.write_bytes(residual_blob)

    component_raw = r4._COMPONENT_HEADER.pack(0, 0, 0, 1, 1)
    compressed_component = zlib.compress(component_raw, 9)
    component_packet = struct.pack("<I", len(compressed_component)) + compressed_component
    component_path = tmp_path / "components.bin"
    component_path.write_bytes(component_packet)
    candidate = {
        "frame": 0,
        "class_id": 0,
        "stratum_id": 0,
        "pixels": 1,
        "packet_offset": 0,
        "bytes": len(component_packet),
        "packet_sha256": hashlib.sha256(component_packet).hexdigest(),
    }
    r3 = {
        "D1_causal_jitter": {
            "models": {
                "adaptive_prior_frames": {
                    "per_class_per_stratum": [
                        {
                            "class_id": 0,
                            "stratum_id": 1,
                            "residual_packet": {
                                "path": str(residual_path),
                                "sha256": hashlib.sha256(residual_blob).hexdigest(),
                                "best": {"coder": "zlib9"},
                                "raw_bytes": len(residual_raw),
                                "record_count": 1,
                            },
                        }
                    ]
                }
            }
        },
        "D2_surgical_components": {
            "all_component_packets": {
                "path": str(component_path),
                "sha256": hashlib.sha256(component_packet).hexdigest(),
            },
            "candidates": [candidate],
        },
        "D4_composed_curve_v3": {
            "admitted": [{"kind": "coherent_component_shape", "name": "component:0:test"}],
            "eaten": [],
        },
    }
    sites, custody = r4.reconstruct_tail_sites(
        r3=r3,
        r2_work_dir=tmp_path / "r2",
        cache=cache,
        pair_count=64,
    )
    assert sum(len(value) for value in sites.values()) == 0
    assert custody["admitted_component_overcredit_site_count"] == 0
    assert custody["per_stream"][1]["physical_tail_sites"] == 0


def test_rule_and_bitstreams_parse_back_exactly() -> None:
    weights = np.asarray([2.2, -0.2, -3.8], dtype=np.float32)
    payload = r4.serialize_rule(weights, iterations=4, seed_factor=64)
    parsed = r4.parse_rule(payload)
    assert parsed["weights"].tolist() == [2.0, 0.0, -4.0]
    bits = np.zeros((2, 8, 16), dtype=np.bool_)
    bits[:, 2:5, 7] = True
    encoded = r4._serialize_bits(r4.MASK_MAGIC, bits)
    assert np.array_equal(r4._parse_bits(encoded, r4.MASK_MAGIC), bits)


@pytest.mark.parametrize("class_id", range(5))
def test_rank4_scorer_rule_counts_all_four_coordinates_and_replays_support(class_id: int) -> None:
    weights = np.asarray([2.2, -0.2, -3.8], dtype=np.float32)
    payload = r4.serialize_scorer_rule(
        weights,
        class_id=class_id,
        iterations=4,
        seed_factor=64,
    )
    parsed = r4.parse_scorer_rule(payload)
    assert len(payload) == r4._SCORER_RULE.size
    assert parsed["weights_bar"].shape == (4, 3)
    assert parsed["support_weights"].tolist() == [2.0, 0.0, -4.0]


def test_uint8_cellular_receiver_is_deterministic_and_inert() -> None:
    seed = np.ones((1, 1, 1), dtype=np.bool_)
    emitted_a, telemetry_a = r4._generate_uint8(
        seed,
        64,
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        iterations=4,
    )
    emitted_b, telemetry_b = r4._generate_uint8(
        seed,
        64,
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        iterations=4,
    )
    assert np.array_equal(emitted_a, emitted_b)
    assert set(np.unique(emitted_a)) == {255}
    assert telemetry_a == telemetry_b
    assert telemetry_a["rc7_inertness_halt"] is True


def test_small_factor_training_is_resumable_and_equal_fidelity(tmp_path) -> None:
    target = np.zeros((1, 64, 64), dtype=np.bool_)
    target[0, 20:44, 20:44] = True
    first = r4._train_factor(
        target=target,
        factor=64,
        class_id=1,
        output_dir=tmp_path / "factor_64",
        implementation_sha256="1" * 64,
    )
    second = r4._train_factor(
        target=target,
        factor=64,
        class_id=1,
        output_dir=tmp_path / "factor_64",
        implementation_sha256="1" * 64,
    )
    assert first == second
    assert first["equal_fidelity_after_own_exceptions"] is True
    assert len(first["checkpoints"]) == 3
    assert first["exact_bytes"] == (
        first["counted_weight_bytes"] + first["instance_seed_bytes"] + first["own_exception_bytes"]
    )
    assert first["scorer_rank4_contract"]["eligible_for_admission"] is False
    assert first["scorer_rank4_exact_support_bytes"] == (
        first["scorer_rank4_counted_weight_bytes"] + first["instance_seed_bytes"] + first["own_exception_bytes"]
    )


def test_factor_resume_refuses_wrong_stage_metadata(tmp_path) -> None:
    target = np.zeros((1, 64, 64), dtype=np.bool_)
    target[0, 20:44, 20:44] = True
    output_dir = tmp_path / "factor_64"
    r4._train_factor(
        target=target,
        factor=64,
        class_id=3,
        output_dir=output_dir,
        implementation_sha256="2" * 64,
    )
    checkpoint_path = output_dir / "stage_00_warmup.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["stage_name"] = "wrong_stage"
    checkpoint_path.write_text(json.dumps(checkpoint))
    with pytest.raises(r4.PredictorR4Error, match="checkpoint refused"):
        r4._train_factor(
            target=target,
            factor=64,
            class_id=3,
            output_dir=output_dir,
            implementation_sha256="2" * 64,
        )


def test_lane_polytope_warm_start_consumes_separate_geometry_target(tmp_path) -> None:
    target = np.zeros((1, 64, 64), dtype=np.bool_)
    target[0, 20:44, 20:44] = True
    lane_base = np.zeros_like(target)
    lane_base[0, :, 30:34] = True
    row = r4._train_factor(
        target=target,
        factor=64,
        class_id=1,
        output_dir=tmp_path / "lane_factor",
        implementation_sha256="3" * 64,
        initialization_target=lane_base,
        initialization_method="task208_lane_openpilot_degree4_polytope_pretrain",
    )
    assert row["polytope_warm_start"]["openpilot_degree4_lane_prior_numerically_consumed"] is True
    assert row["polytope_warm_start"]["initialization_target_sha256"] != row["target_sha256"]


def test_final_refuses_stale_n64_source_config(tmp_path) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    work_dir = tmp_path / "work"
    (work_dir / "n64").mkdir(parents=True)
    n64 = {
        "schema": r4.N64_SCHEMA,
        "target_custody": {"pair_count": 64, "target_membership_sha256": "0" * 64},
        "source_config": {},
        "source_config_sha256": "0" * 64,
    }
    (work_dir / "n64" / "receipt.json").write_text(json.dumps(n64))
    r2_work_dir = tmp_path / "r2_work"
    (r2_work_dir / "n64").mkdir(parents=True)
    (r2_work_dir / "n64" / "receipt.json").write_text("{}")
    r2_receipt = tmp_path / "r2.json"
    r2_receipt.write_text(json.dumps({"schema": "predictor_r2_missdelta_task578.v1"}))
    r3_receipt = tmp_path / "r3.json"
    r3_receipt.write_text(json.dumps({"schema": "predictor_r3_causal_task578.v1"}))
    cache = tmp_path / "cache.bin"
    cache.write_bytes(b"cache")
    with pytest.raises(r4.PredictorR4Error, match="stale n64"):
        r4.build_final_receipt(
            repository_root=repository_root,
            cache=cache,
            r2_work_dir=r2_work_dir,
            r2_receipt_path=r2_receipt,
            r3_receipt_path=r3_receipt,
            work_dir=work_dir,
            output_path=tmp_path / "output.json",
        )
