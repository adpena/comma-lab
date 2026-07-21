# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "tools" / "audit_task_space_levelset_spine.py"
SPEC = importlib.util.spec_from_file_location("audit_task_space_levelset_spine", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> dict:
    upstream = tmp_path / "upstream"
    (upstream / "models").mkdir(parents=True)
    for path, content in (
        (tmp_path / "0.mkv", b"video"),
        (tmp_path / "gt.npz", b"cache"),
        (upstream / "modules.py", b"modules"),
        (upstream / "models" / "segnet.safetensors", b"segnet"),
        (upstream / "models" / "posenet.safetensors", b"posenet"),
    ):
        path.write_bytes(content)
    cache_sha = audit.sha256_file(tmp_path / "gt.npz")
    tie = _write_json(
        tmp_path / "tie.json",
        {
            "score_claim": False,
            "input_fidelity": {
                "n_pairs": 600,
                "all_fp32_exact": True,
                "total_nonzero_values": 0,
                "max_abs_over_pairs": 0.0,
                "total_values": 117_964_800,
            },
        },
    )
    m2 = _write_json(
        tmp_path / "m2.json",
        {
            "schema": "m2_live_target_selection_receipt.v1",
            "score_claim": False,
            "source_custody": {"gt_cache_sha256": cache_sha, "pair_count": 600},
            "candidate": {"archive_bytes": 1_717_172_741, "d_seg": 0.0, "d_pose": 0.0},
            "geometry_decomposition": {
                "resize_numerator_mismatches": 0,
                "full_linear_nullity_fraction": 0.8067423152232891,
                "implemented_integer_exact_null_mask_fraction": 0.22696926089315625,
            },
        },
    )
    lane = _write_json(
        tmp_path / "lane.json",
        {
            "n_pairs": 600,
            "gt_cache": str(tmp_path / "gt.npz"),
            "correspondence_lossless_vs_sort": True,
            "fit_stats": {"total_lines": 2967, "band_recall_mean": 0.5474557765871068},
            "rows": [
                {
                    "variant": "coherent_slot_none",
                    "brotli_bytes": 41303,
                    "rate_term": 25.0 * 41303 / audit.RATE_DENOMINATOR,
                }
            ],
        },
    )
    g1g3 = _write_json(
        tmp_path / "g1g3.json",
        {
            "schema": "g1_worldsheet_g3_cellcode_measurements.v1",
            "score_claim": False,
            "cache": {"sha256": cache_sha},
            "g1": {
                "verdict_scope": "single global ground-plane-homography test; not family",
                "aggregate": {
                    "by_transition": {
                        "within_pair": {"median_of_transition_medians_px": 0.2792, "event_fraction_gt_px": {"4": 0.08}},
                        "cross_pair": {"median_of_transition_medians_px": 0.2798, "event_fraction_gt_px": {"4": 0.09}},
                    }
                },
            },
            "g3": {
                "best_measured_prior": "spatial_temporal_laplace",
                "alphabet": {"headers_and_finite_coder_overhead_included": False, "site_location_cost_included": False},
                "all_flips": {"flip_count": 17926, "ideal_bytes": {"spatial_temporal_laplace": 2724.87}},
            },
        },
    )
    aa = _write_json(
        tmp_path / "aa.json",
        {
            "n_pairs": 600,
            "axis_tag": "[macOS-CPU advisory] NON-PROMOTABLE",
            "render_grid_curve": {"384": {"real": {"aa": {"d_seg": {"mean": 0.0008598582}}}}},
        },
    )
    r1b7 = _write_json(
        tmp_path / "r1b7.json",
        {
            "schema": "r1b7_uint8_survival_carrier_measurement.v1",
            "verdict": "MEASURED_N16_FIXED_NONPOSITIVE_INTEGER_PREFIX_NO_NEW_CROSSING",
            "verdict_scope": "n16 inherited-base instance only",
            "integer_aware": {"search": {"new_hard_crossing_site_count": 0}},
        },
    )
    finite_packet_path = tmp_path / "s2_seed.bin"
    finite_packet_path.write_bytes(b"finite-packet")
    finite_seed = _write_json(
        tmp_path / "s2_seed.json",
        {
            "schema": "s2_partition_seed_measurement.v1",
            "score_claim": False,
            "n_pairs": 600,
            "gt_cache": {"sha256": cache_sha},
            "content_lineage": {"inherited_bytes_in_candidate": 0},
            "finite_packet": {
                "event_count": 17926,
                "packet_bytes": finite_packet_path.stat().st_size,
                "packet_sha256": audit.sha256_file(finite_packet_path),
                "path": str(finite_packet_path),
                "double_encode_byte_identical": True,
                "parse_back_event_identity": True,
                "stored_plane_value_bytes": 0,
            },
            "semantic_detection": {
                "luma_consulted": False,
                "semantic_class_ids": [0, 1, 2, 3, 4],
                "method": "spatial_static_signature_v1",
            },
            "verdict_scope": "finite cell-event component only",
        },
    )
    genuine_proof_path = tmp_path / "genuine_proof.json"
    genuine_proof_path.write_text("{}", encoding="utf-8")
    lane_curve = _write_json(
        tmp_path / "lane_curve.json",
        {
            "schema": "s2_lane_true_mask_curve.v1",
            "score_claim": False,
            "gt_cache": {"sha256": cache_sha},
            "fit": {"n_pairs": 600},
            "semantic_detection": {
                "luma_consulted": False,
                "semantic_class_ids": [0, 1, 2, 3, 4],
            },
            "curve": [
                {"variant": "dash", "finite_brotli_bytes": 41303},
                {"variant": "continuous", "finite_brotli_bytes": 41298},
            ],
            "curvelet_shearlet_residual": {
                "status": "BLOCKED_TARGET_BOUNDARY_INVERSE_CUSTODY",
                "genuine_structural_proof_path": str(genuine_proof_path),
                "genuine_structural_proof_sha256": audit.sha256_file(genuine_proof_path),
            },
            "content_lineage": {"inherited_bytes_in_candidate": 0},
            "verdict_scope": "polynomial control only",
        },
    )
    return {
        "source_video": tmp_path / "0.mkv",
        "upstream_root": upstream,
        "gt_cache": tmp_path / "gt.npz",
        "tie_receipt_path": tie,
        "m2_receipt_path": m2,
        "lane_receipt_path": lane,
        "g1g3_receipt_path": g1g3,
        "aa_receipt_path": aa,
        "r1b7_receipt_path": r1b7,
        "s2_partition_seed_receipt_path": finite_seed,
        "s2_lane_curve_receipt_path": lane_curve,
        "rust_parity": {"status": "PASS", "scope": "test fixture"},
        "lane_id": "test_lane",
    }


def test_build_receipt_keeps_partial_components_out_of_s4(tmp_path: Path) -> None:
    receipt = audit.build_receipt(**_fixture(tmp_path))
    assert receipt["borrowed_substrate_accounting"]["borrowed_candidate_archives"] == []
    assert receipt["borrowed_substrate_accounting"]["inherited_bytes_in_candidate"] == 0
    assert receipt["stages"]["S2_task_space_level_set_witness"]["lane_chart"]["coherent_slot_brotli_bytes"] == 41303
    assert receipt["consumed_artifacts"][0]["path"].endswith("tie.json")
    assert receipt["stages"]["S4_strict_archive_n600_receiver"]["status"].startswith("NOT_BUILT")
    finite = receipt["stages"]["S2_task_space_level_set_witness"]["morse_smale_cell_prior"]["finite_seed"]
    assert finite["status"] == "MEASURED_FINITE_PARSEBACK_COMPLETE"
    assert finite["packet_bytes"] == len(b"finite-packet")
    assert receipt["admission"]["s2_finite_cell_event_seed"] is True
    assert receipt["admission"]["s2_lane_true_mask_curve_measured"] is True
    lane_curve = receipt["stages"]["S2_task_space_level_set_witness"]["lane_chart"]["true_mask_curve"]
    assert lane_curve["status"] == "MEASURED_POLYNOMIAL_CONTROL_GENUINE_RESIDUAL_OPEN"
    assert receipt["admission"]["score_or_pointer_authority"] is False


def test_build_receipt_rejects_cross_cache_g1g3(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    payload = json.loads(kwargs["g1g3_receipt_path"].read_text(encoding="utf-8"))
    payload["cache"]["sha256"] = "0" * 64
    _write_json(kwargs["g1g3_receipt_path"], payload)
    with pytest.raises(audit.SpineAuditError, match="G1/G3 cache hash mismatch"):
        audit.build_receipt(**kwargs)


def test_build_receipt_rejects_nonexact_support_fill(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    payload = json.loads(kwargs["tie_receipt_path"].read_text(encoding="utf-8"))
    payload["input_fidelity"]["total_nonzero_values"] = 1
    _write_json(kwargs["tie_receipt_path"], payload)
    with pytest.raises(audit.SpineAuditError, match="nonzero errors"):
        audit.build_receipt(**kwargs)


def test_build_receipt_rejects_finite_packet_custody_drift(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    payload = json.loads(kwargs["s2_partition_seed_receipt_path"].read_text(encoding="utf-8"))
    Path(payload["finite_packet"]["path"]).write_bytes(b"tampered")
    with pytest.raises(audit.SpineAuditError, match="byte count mismatch"):
        audit.build_receipt(**kwargs)


def test_build_receipt_rejects_semantic_disagreement(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    path = kwargs["s2_lane_curve_receipt_path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["semantic_detection"]["semantic_class_ids"] = [1, 0, 2, 3, 4]
    _write_json(path, payload)
    with pytest.raises(audit.SpineAuditError, match="semantic detection disagree"):
        audit.build_receipt(**kwargs)


def test_build_receipt_rejects_genuine_proof_drift(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    payload = json.loads(kwargs["s2_lane_curve_receipt_path"].read_text(encoding="utf-8"))
    Path(payload["curvelet_shearlet_residual"]["genuine_structural_proof_path"]).write_text(
        '{"tampered":true}', encoding="utf-8"
    )
    with pytest.raises(audit.SpineAuditError, match="structural proof SHA mismatch"):
        audit.build_receipt(**kwargs)
