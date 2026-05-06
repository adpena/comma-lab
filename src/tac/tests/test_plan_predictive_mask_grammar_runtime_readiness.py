from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "plan_predictive_mask_grammar_runtime_readiness.py"


def _load_planner():
    spec = importlib.util.spec_from_file_location("pmg_runtime_readiness_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _probe_manifest() -> dict:
    return {
        "baseline": {
            "bytes": 219472,
            "role": "charged_current_mask_stream",
            "source": "unit-test",
        },
        "best_candidate_by_compressed_size": {
            "candidate_id": "row_span_stride4_class_predictor",
            "family": "low_rank_row_column_spans",
            "evidence_grade": "empirical_byte_probe_only",
            "charged_payload_estimate_scope": (
                "compressed probe payload bytes only; excludes future inflate decoder"
            ),
            "exact_evaluable_now": False,
            "exact_evaluable_reason": "no archive member or exact CUDA artifact",
            "promotion_eligible": False,
            "score_claim": False,
            "best_compression": {
                "compressed_size_bytes": 63212,
                "compressor": "lzma6",
                "delta_bytes_vs_baseline": -156260,
            },
            "transform_stats": {
                "span_shape": [600, 5, 96, 2],
                "valid_span_records": 84980,
            },
        },
    }


def _write_runtime_files(root: Path) -> None:
    runtime = root / "submissions/robust_current/inflate_renderer.py"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(
        "\n".join(
            [
                "def _load_masks_from_cmg3(): pass",
                "row_span_stride_class_predictor_v1 = True",
                "row_span_stride_class_predictor_hotspot_residual_v1 = True",
                "def _decode_cmg3_row_span_hotspot_residual(): pass",
                "nonzero_row_runs_topk_v1 = True",
                "def _decode_cmg3_nonzero_row_runs(): pass",
            ]
        )
    )
    unpacker = root / "submissions/robust_current/unpack_renderer_payload.py"
    unpacker.write_text('"masks.cmg3"\n')
    packed_builder = root / "experiments/build_renderer_packed_payload_archive.py"
    packed_builder.parent.mkdir(parents=True, exist_ok=True)
    packed_builder.write_text('"masks.cmg3"\n')
    (root / "experiments/build_cmg3_rowspan_candidate.py").write_text("")
    (root / "experiments/build_pmg_hotspot_candidate.py").write_text("")


def test_runtime_readiness_blocks_dispatch_on_same_family_exact_negative(tmp_path: Path) -> None:
    planner = _load_planner()
    probe_path = tmp_path / "probe.json"
    _write_json(probe_path, _probe_manifest())
    _write_runtime_files(tmp_path)
    exact_path = tmp_path / "exact_negative.json"
    _write_json(
        exact_path,
        {
            "archive_size_bytes": 128028,
            "score_recomputed_from_components": 29.22356762140238,
            "avg_posenet_dist": 39.78416443,
            "avg_segnet_dist": 0.09192351,
            "n_samples": 600,
        },
    )

    manifest = planner.build_plan(
        probe_manifest=probe_path,
        output_json=tmp_path / "out" / planner.REPORT_NAME,
        exact_negative_specs=(
            planner.ExactNegativeSpec("cmg3_top1", "cmg3_nonzero_row_runs", exact_path),
        ),
        repo_root=tmp_path,
        command=["unit-test"],
    )

    assert manifest["schema"] == planner.SCHEMA
    assert manifest["score_claim"] is False
    assert manifest["gpu_required"] is False
    assert manifest["cuda_jobs_launched"] is False
    assert manifest["cloud_jobs_dispatched"] is False
    assert manifest["remote_dispatch_allowed"] is False
    assert manifest["dispatch_readiness"]["dispatchable_now"] is False
    assert manifest["probe"]["candidate_id"] == "row_span_stride4_class_predictor"
    assert manifest["byte_headroom"]["required_savings_at_unchanged_distortion_bytes"] == 113564
    assert manifest["byte_headroom"]["formula_only_headroom_before_decoder_packer_runtime_overhead_bytes"] == 42696
    assert manifest["runtime_decoder_packer_readiness"]["runtime_decoder_exists"] is True
    assert manifest["runtime_decoder_packer_readiness"]["packed_payload_builder_accepts_cmg3"] is True
    review = manifest["same_family_exact_negative_review"]
    assert review["same_family_blockers_present"] is True
    assert review["byte_sufficient_but_collapsed_count"] == 1
    blocker_ids = {item["blocker_id"] for item in manifest["implementation_blockers"]}
    assert "byte_probe_not_archive_member" in blocker_ids
    assert "same_family_exact_negatives_pose_seg_collapse" in blocker_ids
    assert "probe_compressor_label_not_runtime_enum" in blocker_ids
    assert manifest["ranked_next_actions"][0]["action_id"] == "local_decode_geometry_parity_gate"
    assert json.loads((tmp_path / "out" / planner.REPORT_NAME).read_text()) == manifest


def test_runtime_readiness_fails_closed_on_missing_probe(tmp_path: Path) -> None:
    planner = _load_planner()

    with pytest.raises(planner.PlannerError, match="required probe manifest is missing"):
        planner.build_plan(
            probe_manifest=tmp_path / "missing.json",
            output_json=tmp_path / "out.json",
            exact_negative_specs=(),
            repo_root=tmp_path,
        )


def test_runtime_readiness_fails_closed_on_malformed_probe(tmp_path: Path) -> None:
    planner = _load_planner()
    probe = _probe_manifest()
    probe["best_candidate_by_compressed_size"]["best_compression"]["compressed_size_bytes"] = "63212"
    probe_path = tmp_path / "probe.json"
    _write_json(probe_path, probe)

    with pytest.raises(planner.PlannerError, match="compressed_size_bytes"):
        planner.build_plan(
            probe_manifest=probe_path,
            output_json=tmp_path / "out.json",
            exact_negative_specs=(),
            repo_root=tmp_path,
        )


def test_runtime_readiness_missing_exact_negative_is_reported_not_fabricated(tmp_path: Path) -> None:
    planner = _load_planner()
    probe_path = tmp_path / "probe.json"
    _write_json(probe_path, _probe_manifest())
    _write_runtime_files(tmp_path)

    manifest = planner.build_plan(
        probe_manifest=probe_path,
        output_json=tmp_path / "out.json",
        exact_negative_specs=(
            planner.ExactNegativeSpec(
                "missing_negative",
                "cmg3_row_span",
                tmp_path / "does_not_exist.json",
            ),
        ),
        repo_root=tmp_path,
    )

    review = manifest["same_family_exact_negative_review"]
    assert review["present_count"] == 0
    assert review["missing_count"] == 1
    assert review["same_family_blockers_present"] is False
    assert review["records"][0]["present"] is False
