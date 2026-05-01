"""Synthetic tests for Alpha-Geo-0 NeRV geometry diagnostics.

These tests keep the diagnostic on CPU integer tensors. They do not import or
exercise scorer/PoseNet code.
"""
from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
DIAG_PATH = REPO_ROOT / "experiments" / "diagnose_nerv_geometry.py"


def _load_diag_module():
    spec = importlib.util.spec_from_file_location("_alpha_geo_diag", DIAG_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


diag = _load_diag_module()


def _two_class_boundary_masks() -> torch.Tensor:
    masks = torch.zeros(2, 8, 8, dtype=torch.long)
    masks[:, :, 4:] = 1
    masks[:, 2:4, 1:3] = 2
    return masks


def test_identical_masks_have_zero_disagreement_and_pass() -> None:
    baseline = _two_class_boundary_masks()

    result = diag.compute_nerv_geometry_diagnostics(
        baseline,
        baseline.clone(),
        num_classes=5,
        thresholds=diag.GeometryThresholds.from_preset("exploratory"),
        chunk_frames=1,
    )

    assert result["global"]["disagreement_pixels"] == 0
    assert result["global"]["global_disagreement"] == 0.0
    assert result["per_class"]["confusion_matrix"][1][1] == int((baseline == 1).sum().item())
    assert "2" in result["boundary_bands"]
    assert result["pass_fail"]["checks"]["boundary_band_2px_disagreement"]["passed"] is True
    assert result["temporal"]["stable_region"]["false_flip_rate"] == 0.0
    assert result["temporal"]["pair_transition"]["f1"] == 1.0
    assert result["components"]["speckle_rate"] == 0.0
    assert result["pass_fail"]["overall_pass"] is True


def test_global_confusion_and_boundary_band_disagreement_are_reported() -> None:
    baseline = _two_class_boundary_masks()
    candidate = baseline.clone()
    candidate[0, 3, 4] = 0

    result = diag.compute_nerv_geometry_diagnostics(
        baseline,
        candidate,
        num_classes=5,
        boundary_radii=(1, 3, 5),
        thresholds=diag.GeometryThresholds(
            global_disagreement_max=0.001,
            boundary_band_disagreement_max={1: 0.001},
            stable_region_false_flip_rate_max=None,
            pair_transition_disagreement_max=None,
            tiny_speckle_rate_max=None,
            max_component_centroid_jump_px=None,
            missing_component_rate_max=None,
        ),
        chunk_frames=1,
    )

    assert result["global"]["disagreement_pixels"] == 1
    assert result["global"]["global_disagreement"] == pytest.approx(1 / (2 * 8 * 8))
    assert result["per_class"]["confusion_matrix"][1][0] == 1
    assert result["boundary_bands"]["1"]["disagreement_pixels"] == 1
    assert result["boundary_bands"]["1"]["disagreement_rate"] > 0.0
    assert result["pass_fail"]["checks"]["global_disagreement"]["passed"] is False
    assert result["pass_fail"]["checks"]["boundary_band_1px_disagreement"]["passed"] is False


def test_stable_region_temporal_false_flip_and_transition_f1() -> None:
    baseline = torch.zeros(2, 6, 6, dtype=torch.long)
    candidate = baseline.clone()
    candidate[1, 2, 3] = 1

    result = diag.compute_nerv_geometry_diagnostics(
        baseline,
        candidate,
        num_classes=5,
        thresholds=diag.GeometryThresholds.from_preset("none"),
        chunk_frames=1,
        worst_pair_count=3,
    )

    stable = result["temporal"]["stable_region"]
    transition = result["temporal"]["pair_transition"]
    assert stable["stable_pixels"] == 36
    assert stable["false_flip_pixels"] == 1
    assert stable["false_flip_rate"] == pytest.approx(1 / 36)
    assert transition["fp_pixels"] == 1
    assert transition["fn_pixels"] == 0
    assert transition["f1"] == 0.0
    assert result["temporal"]["worst_frame_pairs"][0]["pair_index"] == 0
    assert result["temporal"]["worst_frame_pairs"][0]["stable_false_flip_pixels"] == 1


def test_pair_transition_miss_counts_false_negative() -> None:
    baseline = torch.zeros(2, 6, 6, dtype=torch.long)
    candidate = baseline.clone()
    baseline[1, 1, 1] = 2

    result = diag.compute_nerv_geometry_diagnostics(
        baseline,
        candidate,
        num_classes=5,
        thresholds=diag.GeometryThresholds.from_preset("none"),
        chunk_frames=1,
    )

    transition = result["temporal"]["pair_transition"]
    assert transition["tp_pixels"] == 0
    assert transition["fp_pixels"] == 0
    assert transition["fn_pixels"] == 1
    assert transition["disagreement_rate"] == pytest.approx(1 / 36)
    assert transition["f1"] == 0.0


def test_tiny_unsupported_component_is_counted_as_speckle() -> None:
    baseline = torch.zeros(1, 7, 7, dtype=torch.long)
    candidate = baseline.clone()
    candidate[0, 3, 3] = 2

    result = diag.compute_nerv_geometry_diagnostics(
        baseline,
        candidate,
        num_classes=5,
        thresholds=diag.GeometryThresholds.from_preset("none"),
        component_options=diag.ComponentOptions(tiny_component_max_area=2),
    )

    assert result["components"]["candidate_tiny_components"] >= 1
    assert result["components"]["speckle_components"] == 1
    assert result["components"]["speckle_pixels"] == 1
    assert result["components"]["speckle_rate"] == pytest.approx(1 / 49)
    assert result["components"]["per_class"]["2"]["speckle_components"] == 1


def test_component_centroid_jump_gate_fails_large_shift() -> None:
    baseline = torch.zeros(1, 8, 8, dtype=torch.long)
    candidate = torch.zeros_like(baseline)
    baseline[0, 2:5, 2:5] = 1
    candidate[0, 2:5, 4:7] = 1

    result = diag.compute_nerv_geometry_diagnostics(
        baseline,
        candidate,
        num_classes=5,
        thresholds=diag.GeometryThresholds(
            global_disagreement_max=None,
            boundary_band_disagreement_max={},
            stable_region_false_flip_rate_max=None,
            pair_transition_disagreement_max=None,
            tiny_speckle_rate_max=None,
            max_component_centroid_jump_px=1.0,
            missing_component_rate_max=0.0,
        ),
    )

    centroid = result["components"]["centroid"]
    assert centroid["matched_components"] >= 1
    assert centroid["max_matched_jump_px"] == pytest.approx(2.0)
    assert result["pass_fail"]["checks"]["max_component_centroid_jump_px"]["passed"] is False


def test_residual_region_ranking_prioritizes_lower_field_lane_errors() -> None:
    baseline = torch.zeros(2, 10, 10, dtype=torch.long)
    candidate = baseline.clone()
    baseline[0, 7:9, 1:3] = 1
    candidate[0, 1:3, 7:9] = 4
    candidate[1, 5, 5] = 3

    result = diag.compute_nerv_geometry_diagnostics(
        baseline,
        candidate,
        num_classes=5,
        thresholds=diag.GeometryThresholds.from_preset("none"),
        residual_ranking_options=diag.ResidualRankingOptions(
            max_regions=3,
            min_area=1,
            boundary_radius=2,
            near_field_y_fraction=0.60,
            critical_classes=(1, 2),
        ),
    )

    ranking = result["residual_region_ranking"]
    assert ranking["score_evidence_grade"] == "empirical"
    assert ranking["device"] == "cpu"
    assert ranking["promotion_eligible"] is False
    assert ranking["score_claim_eligible"] is False
    assert ranking["regions_returned"] == 3
    top = ranking["regions"][0]
    assert top["rank"] == 1
    assert top["priority_label"] == "lower_field_lane_marking"
    assert top["suggested_repair"] == "lane_lower_field_residual"
    assert top["box_xyxy"] == [1, 7, 3, 9]
    assert top["baseline_class_hist"] == {"1": 4}
    assert top["candidate_class_hist"] == {"0": 4}
    assert top["critical_class_pixels"] == 4
    assert top["boundary_band_pixels"] == 4
    assert top["estimated_uncompressed_pixels"] == 4


def test_residual_region_ranking_can_be_explicitly_skipped_without_score_claim() -> None:
    baseline = torch.zeros(2, 10, 10, dtype=torch.long)
    candidate = baseline.clone()
    candidate[:, 6:8, 2:4] = 1

    result = diag.compute_nerv_geometry_diagnostics(
        baseline,
        candidate,
        num_classes=5,
        thresholds=diag.GeometryThresholds.from_preset("none"),
        residual_ranking_options=diag.ResidualRankingOptions(max_regions=0),
    )

    ranking = result["residual_region_ranking"]
    assert ranking["scan_skipped"] is True
    assert ranking["skip_reason"] == "max_regions=0"
    assert ranking["regions_returned"] == 0
    assert ranking["promotion_eligible"] is False
    assert ranking["score_claim_eligible"] is False
    assert result["global"]["disagreement_pixels"] == 8


def test_visual_primitives_packet_is_cpu_non_promotable_and_summarizes_failures() -> None:
    baseline = torch.full((3, 12, 12), 4, dtype=torch.long)
    baseline[:, 6:, :] = 0
    baseline[0, 8:11, 2:5] = 1
    baseline[1, 8:11, 3:6] = 1
    baseline[2, 8:11, 4:7] = 1
    baseline[1, 7:10, 8:11] = 2
    candidate = baseline.clone()
    candidate[0, 8:11, 2:5] = 0
    candidate[1, 7:10, 8:11] = 0
    candidate[1, 7:10, 9:12] = 2

    result = diag.compute_nerv_geometry_diagnostics(
        baseline,
        candidate,
        num_classes=5,
        boundary_radii=(1, 2),
        thresholds=diag.GeometryThresholds.from_preset("none"),
        chunk_frames=1,
    )

    packet = result["visual_primitives"]
    assert packet["diagnostic"] == "alpha_geo_visual_primitives_v1"
    assert packet["score_evidence_grade"] == "empirical"
    assert packet["device"] == "cpu"
    assert packet["scorer_proxy"] is False
    assert packet["scorer_network_loaded"] is False
    assert packet["promotion_eligible"] is False
    assert packet["score_claim_eligible"] is False
    assert packet["exact_eval_claim"] is False
    assert packet["source"]["baseline_mask_sha256"] == diag._mask_tensor_sha256(baseline)
    assert packet["source"]["candidate_mask_sha256"] == diag._mask_tensor_sha256(candidate)
    assert packet["source"]["visual_frame_stride"] == 1
    assert packet["visual_shape"]["frames"] == 3

    components = packet["primitive_metrics"]["connected_components"]
    lane = components["per_class"]["1"]
    assert lane["evaluated_baseline_components"] == 3
    assert lane["missing_components"] == 1
    assert lane["missing_area_px"] == 9
    assert components["critical_summary"]["missing_components"] >= 1
    top_failure = components["critical_box_failures"][0]
    assert top_failure["failure_type"] == "missing_component"
    assert top_failure["class_id"] == 1
    assert top_failure["box_xyxy"] == [2, 8, 5, 11]
    assert top_failure["pose_sensitive"] is True

    boundary = packet["primitive_metrics"]["boundary_primitives"]["families"]["lane"]
    assert boundary["polyline_ordered"] is False
    assert boundary["baseline_boundary_pixels"] > 0
    assert boundary["baseline_coverage_at_2px"] < 1.0
    temporal = packet["primitive_metrics"]["temporal_primitives"]
    assert temporal["track_proxy_method"] == "same_class_iou_then_centroid_cpu"
    assert temporal["critical_track_fragmentation_rate_proxy"] > 0.0
    assert packet["pass_fail"]["exact_eval_spend_gate"]["promotion_eligible"] is False
    repair = packet["repair_retrain_spec"]
    assert repair["diagnostic"] == "alpha_geo_repair_retrain_spec_v1"
    assert repair["l2_clearance_created"] is False
    assert repair["promotion_eligible"] is False
    assert repair["score_claim_eligible"] is False
    assert "critical_missing_rate" in repair["blocker_to_spec_ids"]
    assert "critical_component_recall_retrain" in repair["blocker_to_spec_ids"]["critical_missing_rate"]
    assert "boundary_band_retrain" in {
        spec["spec_id"] for spec in repair["training_specs"]
    }
    assert repair["currently_admissible_commands"][0]["requires_l2_clearance"] is False
    assert repair["currently_admissible_commands"][0]["score_claim_eligible"] is False
    assert "--residual-region-count" in repair["currently_admissible_commands"][0]["command"]
    assert repair["blocked_commands"][0]["requires_l2_clearance"] is True
    assert packet["next_action"]["score_claim_eligible"] is False


def test_visual_primitives_can_stride_expensive_cpu_packet() -> None:
    baseline = torch.full((6, 12, 12), 4, dtype=torch.long)
    baseline[:, 6:, :] = 0
    baseline[:, 8:11, 2:5] = 1
    candidate = baseline.clone()
    candidate[0, 8:11, 2:5] = 0
    candidate[2, 8:11, 2:5] = 0
    candidate[4, 8:11, 2:5] = 0

    result = diag.compute_nerv_geometry_diagnostics(
        baseline,
        candidate,
        num_classes=5,
        boundary_radii=(1,),
        thresholds=diag.GeometryThresholds.from_preset("none"),
        visual_primitive_options=diag.VisualPrimitiveOptions(
            frame_stride=2,
            critical_component_min_area=4,
            boundary_distance_sample_cap=16,
            critical_classes=(1, 2),
        ),
    )

    packet = result["visual_primitives"]
    assert result["shape"]["frames"] == 6
    assert packet["shape"]["frames"] == 6
    assert packet["visual_shape"]["frames"] == 3
    assert packet["source"]["baseline_mask_sha256"] == diag._mask_tensor_sha256(baseline)
    assert packet["source"]["visual_baseline_mask_sha256"] == diag._mask_tensor_sha256(baseline[::2])
    assert packet["source"]["visual_frame_stride"] == 2
    assert packet["primitive_metrics"]["connected_components"]["critical_summary"]["missing_components"] == 3


def test_visual_boundary_distance_samples_are_globally_bounded() -> None:
    baseline = torch.full((4, 16, 16), 4, dtype=torch.long)
    baseline[:, 8:, :] = 0
    baseline[:, 10:14, 2:6] = 1
    candidate = baseline.clone()
    candidate[:, 10:14, 4:8] = 1
    candidate[:, 10:14, 2:4] = 0

    result = diag.compute_nerv_geometry_diagnostics(
        baseline,
        candidate,
        num_classes=5,
        boundary_radii=(1, 2),
        thresholds=diag.GeometryThresholds.from_preset("none"),
        visual_primitive_options=diag.VisualPrimitiveOptions(
            boundary_distance_sample_cap=64,
            boundary_distance_global_sample_cap=7,
            critical_component_min_area=4,
            critical_classes=(1, 2),
        ),
    )

    families = result["visual_primitives"]["primitive_metrics"]["boundary_primitives"]["families"]
    for row in families.values():
        assert row["distance_sample_count"] <= 7
        assert row["distance_sample_global_cap"] == 7
        assert row["distance_sample_method"] == "deterministic_hash_reservoir_from_frame_samples"
        assert row["distance_sample_population_seen"] >= row["distance_sample_count"]


def test_visual_temporal_track_proxies_can_be_disabled_without_disabling_scalar_temporal() -> None:
    baseline = torch.full((3, 12, 12), 4, dtype=torch.long)
    baseline[:, 6:, :] = 0
    baseline[:, 8:11, 2:5] = 1
    candidate = baseline.clone()
    candidate[1, 8:11, 3:6] = 1

    result = diag.compute_nerv_geometry_diagnostics(
        baseline,
        candidate,
        num_classes=5,
        thresholds=diag.GeometryThresholds.from_preset("none"),
        residual_ranking_options=diag.ResidualRankingOptions(max_regions=0),
        visual_primitive_options=diag.VisualPrimitiveOptions(
            critical_component_min_area=4,
            temporal_tracks_enabled=False,
        ),
    )

    temporal = result["visual_primitives"]["primitive_metrics"]["temporal_primitives"]
    assert temporal["track_proxy_method"] == "disabled_by_visual_primitive_options"
    assert temporal["critical_track_survival_recall_proxy"] is None
    assert temporal["score_claim_eligible"] is False
    assert temporal["existing_temporal_mask_metrics"]["pair_transition"]["total_pixels"] == 2 * 12 * 12


def test_predecoded_mask_cache_reuses_hash_keyed_tensor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    masks = _two_class_boundary_masks()
    payload = io.BytesIO()
    torch.save(masks, payload)
    archive = tmp_path / "nested.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("payloads/masks.pt", payload.getvalue())
    cache_dir = tmp_path / "mask_cache"

    loaded, first_metadata = diag._load_mask_stream_with_metadata(
        archive,
        archive_member="payloads/masks.pt",
        expected_frames=None,
        height=None,
        width=None,
        nrv_batch_size=16,
        mask_cache_dir=cache_dir,
    )

    assert torch.equal(loaded.long(), masks)
    first_cache = first_metadata["predecoded_cache"]
    assert first_cache["status"] == "miss_written"
    assert first_cache["promotion_eligible"] is False
    assert first_cache["score_claim_eligible"] is False
    assert Path(first_cache["tensor_path"]).exists()
    assert Path(first_cache["metadata_path"]).exists()

    def _boom(*args, **kwargs):
        raise AssertionError("cache hit should not decode the source again")

    monkeypatch.setattr(diag, "_load_mask_stream", _boom)
    cached, second_metadata = diag._load_mask_stream_with_metadata(
        archive,
        archive_member="payloads/masks.pt",
        expected_frames=None,
        height=None,
        width=None,
        nrv_batch_size=16,
        mask_cache_dir=cache_dir,
    )

    assert torch.equal(cached.long(), masks)
    assert second_metadata["predecoded_cache"]["status"] == "hit"
    assert second_metadata["predecoded_cache"]["cache_key"] == first_cache["cache_key"]
    assert second_metadata["decoded_mask_sha256"] == diag._mask_tensor_sha256(masks)


def test_streaming_nerv_render_matches_reference_small_grid() -> None:
    from tac.nerv_mask_codec import NeRVMaskCodec, render_mask_argmax

    codec = NeRVMaskCodec(num_freqs=2, hidden_dim=8, num_classes=5, depth=3, seed=123)

    expected = render_mask_argmax(
        codec,
        num_frames=3,
        height=5,
        width=7,
        batch_size=11,
    )
    actual = diag._render_nerv_mask_argmax_streaming(
        codec,
        num_frames=3,
        height=5,
        width=7,
        batch_size=13,
    )

    assert torch.equal(actual, expected)


def test_cli_writes_json_from_tensor_files(tmp_path: Path) -> None:
    baseline = _two_class_boundary_masks()
    candidate = baseline.clone()
    candidate[1, 0, 0] = 1
    baseline_path = tmp_path / "baseline.pt"
    candidate_path = tmp_path / "candidate.pt"
    out_json = tmp_path / "diag.json"
    torch.save(baseline, baseline_path)
    torch.save(candidate, candidate_path)

    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            str(DIAG_PATH),
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--output-json",
            str(out_json),
            "--threshold-preset",
            "none",
            "--residual-region-count",
            "2",
            "--visual-frame-stride",
            "2",
            "--visual-boundary-distance-sample-cap",
            "16",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[alpha-geo-0] wrote" in proc.stdout
    data = json.loads(out_json.read_text())
    assert data["inputs"]["baseline"] == str(baseline_path)
    assert data["inputs"]["baseline_source"]["source_sha256"] == hashlib.sha256(
        baseline_path.read_bytes()
    ).hexdigest()
    assert data["inputs"]["baseline_source"]["decoded_mask_sha256"] == diag._mask_tensor_sha256(
        baseline
    )
    assert data["diagnostic_config"]["boundary_radii"] == [1, 2, 3, 5]
    assert data["diagnostic_config"]["residual_ranking_options"]["max_regions"] == 2
    assert data["command"]["tool"] == "experiments/diagnose_nerv_geometry.py"
    assert data["global"]["disagreement_pixels"] == 1
    assert data["residual_region_ranking"]["diagnostic"] == "alpha_geo_residual_region_ranking"
    assert data["residual_region_ranking"]["regions_returned"] == 1
    assert data["residual_region_ranking"]["regions"][0]["frame"] == 1
    assert data["visual_primitives"]["diagnostic"] == "alpha_geo_visual_primitives_v1"
    assert data["visual_primitives"]["device"] == "cpu"
    assert data["visual_primitives"]["promotion_eligible"] is False
    assert data["visual_primitives"]["score_claim_eligible"] is False
    assert data["visual_primitives"]["source"]["visual_frame_stride"] == 2
    assert data["visual_primitives"]["source"]["diagnose_nerv_geometry_json"] == str(out_json)
    assert data["diagnostic_config"]["visual_primitive_options"]["critical_component_min_area"] == 9
    assert data["diagnostic_config"]["visual_primitive_options"]["frame_stride"] == 2


def test_zip_member_loader_records_member_and_decoded_hash(tmp_path: Path) -> None:
    masks = _two_class_boundary_masks()
    payload = io.BytesIO()
    torch.save(masks, payload)
    payload_bytes = payload.getvalue()
    archive = tmp_path / "nested.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("payloads/masks.pt", payload_bytes)

    loaded, metadata = diag._load_mask_stream_with_metadata(
        archive,
        archive_member="payloads/masks.pt",
        expected_frames=None,
        height=None,
        width=None,
        nrv_batch_size=16,
    )

    assert torch.equal(loaded.long(), masks)
    assert metadata["archive_member_resolved"] == "payloads/masks.pt"
    assert metadata["archive_member_sha256"] == hashlib.sha256(payload_bytes).hexdigest()
    assert metadata["decoded_mask_sha256"] == diag._mask_tensor_sha256(masks)


def test_zip_member_loader_rejects_unsafe_member_path(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../masks.pt", b"not a tensor")

    with pytest.raises(ValueError, match="unsafe archive member path"):
        diag._load_mask_stream(
            archive,
            archive_member="../masks.pt",
            expected_frames=None,
            height=None,
            width=None,
            nrv_batch_size=16,
        )


def test_zip_member_loader_rejects_duplicate_members(tmp_path: Path) -> None:
    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("masks.pt", b"first")
        with pytest.warns(UserWarning):
            zf.writestr("masks.pt", b"second")

    with pytest.raises(ValueError, match="duplicate archive member"):
        diag._load_mask_stream(
            archive,
            archive_member="masks.pt",
            expected_frames=None,
            height=None,
            width=None,
            nrv_batch_size=16,
        )
