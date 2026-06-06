# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.repo_io import sha256_file
from tac.submission_archive import MINIMAL_SINGLE_MEMBER_NAME
from tac.substrates.hi_nerv.architecture import HinervConfig, HinervSubstrate
from tac.substrates.hi_nerv.archive import pack_archive
from tac.substrates.hi_nerv.archive_candidate import (
    build_hi_nerv_archive_replay_components,
)
from tac.substrates.hi_nerv.receiver_cache_quality import (
    HI_NERV_RECEIVER_CACHE_DISTORTION_CRUX_SCHEMA,
    HI_NERV_RECEIVER_CACHE_MLX_SCORER_RESPONSE_PROBE_SCHEMA,
    HI_NERV_RECEIVER_CACHE_QUALITY_REPORT_SCHEMA,
    HI_NERV_RECEIVER_CACHE_SEGNET_ARGMAX_PROBE_SCHEMA,
    build_hi_nerv_receiver_cache_mlx_scorer_response_probe,
    build_hi_nerv_receiver_cache_scorer_input_distribution_gate,
    build_hi_nerv_receiver_cache_segnet_argmax_probe,
    write_hi_nerv_receiver_cache_quality_report,
)


def test_hi_nerv_receiver_cache_quality_writes_direct_cache_from_archive(
    tmp_path: Path,
) -> None:
    archive = _write_tiny_hiv1_archive(tmp_path / "archive.zip")

    report = write_hi_nerv_receiver_cache_quality_report(
        archive_zip_path=archive,
        output_dir=tmp_path / "quality",
        max_pairs=1,
        batch_pairs=1,
    )

    cache_dir = Path(report["candidate_cache_dir"])
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads(
        (cache_dir / "hi_nerv_direct_receiver_render_cache_identity_audit.json").read_text(
            encoding="utf-8"
        )
    )
    pair_indices = np.load(cache_dir / "pair_indices.npy")

    assert report["schema"] == HI_NERV_RECEIVER_CACHE_QUALITY_REPORT_SCHEMA
    assert report["archive_sha256"] == sha256_file(archive)
    assert report["quality_gate"] is None
    assert report["quality_gate_passed"] is False
    assert "hi_nerv_receiver_cache_quality_reference_gate_not_run" in report[
        "blockers"
    ]
    assert manifest["source_kind"] == "hi_nerv_direct_receiver_render"
    assert manifest["pair_count"] == 1
    assert pair_indices.tolist() == [[0, 1]]
    assert audit["source"]["archive_magic"] == "HIV1"
    assert audit["cache"]["raw_sha256"] == manifest["raw_sha256"]
    assert audit["score_claim"] is False


def test_hi_nerv_receiver_cache_quality_consumes_minimal_x_member(
    tmp_path: Path,
) -> None:
    archive = _write_tiny_hiv1_archive(
        tmp_path / "archive.zip",
        member_name=MINIMAL_SINGLE_MEMBER_NAME,
    )

    report = write_hi_nerv_receiver_cache_quality_report(
        archive_zip_path=archive,
        output_dir=tmp_path / "quality",
        max_pairs=1,
        batch_pairs=1,
    )

    assert report["zip_member"] == MINIMAL_SINGLE_MEMBER_NAME
    assert report["direct_receiver_cache_report"]["zip_member"] == (
        MINIMAL_SINGLE_MEMBER_NAME
    )
    audit = json.loads(
        (
            Path(report["candidate_cache_dir"])
            / "hi_nerv_direct_receiver_render_cache_identity_audit.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["source"]["zip_member"] == MINIMAL_SINGLE_MEMBER_NAME
    assert Path(report["candidate_cache_manifest_path"]).is_file()


def test_hi_nerv_archive_replay_components_parse_minimal_x_member(
    tmp_path: Path,
) -> None:
    archive = _write_tiny_hiv1_archive(
        tmp_path / "archive.zip",
        member_name=MINIMAL_SINGLE_MEMBER_NAME,
    )
    target0, target1 = _render_tiny_hiv1_targets()

    components = build_hi_nerv_archive_replay_components(
        archive,
        {"local_pair_indices": np.array([0, 1], dtype=np.int64)},
        target_rgb_0=target0,
        target_rgb_1=target1,
        candidate_kind="ema",
    )

    assert components["archive_replay_pair_count"] == pytest.approx(2.0)
    assert components["archive_replay_archive_bytes"] == pytest.approx(
        archive.stat().st_size
    )
    assert components["archive_replay_payload_bytes"] > 0
    assert components["archive_replay_candidate_is_ema"] == pytest.approx(1.0)
    assert components["parseback_rgb_pair_mse"] < 1.0e-3
    assert components["selection_health_parseback_rgb_dynamic_range"] > 0.0
    assert components["selection_health_parseback_rgb_temporal_delta_std"] >= 0.0


def test_hi_nerv_archive_replay_components_attach_segnet_teacher_metrics(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mlx.core")
    archive = _write_tiny_hiv1_archive(tmp_path / "archive.zip")
    target0, target1 = _render_tiny_hiv1_targets()
    labels = (target1[..., 0] > target1[..., 1]).astype(np.int32)

    class FakeSegnetTeacher:
        num_classes = 2

        def teacher_argmax_for_indices(self, idx):
            import mlx.core as mx

            return mx.array(labels[np.asarray(idx, dtype=np.int64)], dtype=mx.int32)

        def teacher_logits_for_frames_nhwc01(self, frames):
            import mlx.core as mx

            return mx.stack([frames[..., 1], frames[..., 0]], axis=-1)

    components = build_hi_nerv_archive_replay_components(
        archive,
        {"local_pair_indices": np.array([0, 1], dtype=np.int64)},
        target_rgb_0=target0,
        target_rgb_1=target1,
        scorer_teacher=FakeSegnetTeacher(),
    )

    assert components["parseback_segnet_argmax_disagreement_score_units"] < 1.0
    assert (
        components[
            "selection_health_segnet_direct_live_candidate_occupied_class_fraction"
        ]
        > 0.0
    )
    assert (
        components[
            "selection_health_segnet_direct_live_candidate_target_class_min_ratio"
        ]
        >= 0.0
    )


def test_hi_nerv_receiver_cache_quality_rejects_ambiguous_payload_members(
    tmp_path: Path,
) -> None:
    legacy = _write_tiny_hiv1_archive(tmp_path / "legacy.zip")
    with zipfile.ZipFile(legacy, "r") as zf:
        packet = zf.read("0.bin")
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", packet)
        zf.writestr(MINIMAL_SINGLE_MEMBER_NAME, packet)

    with pytest.raises(ValueError, match="exactly one receiver payload member"):
        write_hi_nerv_receiver_cache_quality_report(
            archive_zip_path=archive,
            output_dir=tmp_path / "quality",
            max_pairs=1,
            batch_pairs=1,
        )


def test_hi_nerv_receiver_cache_quality_uses_explicit_source_pair_indices(
    tmp_path: Path,
) -> None:
    archive = _write_tiny_hiv1_archive(tmp_path / "archive.zip")

    report = write_hi_nerv_receiver_cache_quality_report(
        archive_zip_path=archive,
        output_dir=tmp_path / "quality",
        max_pairs=1,
        batch_pairs=1,
        pair_indices=(1,),
    )

    cache_dir = Path(report["candidate_cache_dir"])
    pair_indices = np.load(cache_dir / "pair_indices.npy")
    direct = report["direct_receiver_cache_report"]
    audit = json.loads(
        (cache_dir / "hi_nerv_direct_receiver_render_cache_identity_audit.json").read_text(
            encoding="utf-8"
        )
    )

    assert pair_indices.tolist() == [[2, 3]]
    assert direct["selected_pair_indices"] == [1]
    assert direct["pair_index_scope"] == "explicit_source_pair_indices"
    assert audit["direct_render"]["selected_pair_indices"] == [1]


def test_hi_nerv_receiver_cache_quality_requires_argmax_probe_for_reference_gate(
    tmp_path: Path,
) -> None:
    archive = _write_tiny_hiv1_archive(tmp_path / "archive.zip")
    reference_report = write_hi_nerv_receiver_cache_quality_report(
        archive_zip_path=archive,
        output_dir=tmp_path / "reference",
        max_pairs=1,
        batch_pairs=1,
    )

    report = write_hi_nerv_receiver_cache_quality_report(
        archive_zip_path=archive,
        output_dir=tmp_path / "candidate",
        reference_cache_dir=Path(reference_report["candidate_cache_dir"]),
        max_pairs=1,
        batch_pairs=1,
        min_segnet_std=0.0,
        min_segnet_dynamic_range=0.0,
        max_segnet_mae_vs_reference_for_fit_gate=1.0,
        min_posenet_yuv6_std=0.0,
        min_posenet_yuv6_dynamic_range=0.0,
        max_posenet_yuv6_mae_vs_reference_for_fit_gate=1.0,
        require_mlx_scorer_response_probe=False,
    )

    assert report["quality_gate"] is not None
    assert report["quality_gate"]["schema"] == "mlx_cache_quality_gate.v1"
    assert report["quality_gate"]["verdict"] == "CACHE_INPUTS_NONDEGENERATE_LOCAL_ONLY"
    assert report["quality_gate_passed"] is False
    assert report["segnet_argmax_probe"]["fit_gate_passed"] is False
    assert "hi_nerv_receiver_cache_segnet_argmax_probe_not_run" in report["blockers"]
    assert report["distortion_crux_probe"]["schema"] == (
        HI_NERV_RECEIVER_CACHE_DISTORTION_CRUX_SCHEMA
    )
    assert report["distortion_crux_probe"]["fit_gate_passed"] is True
    assert report["distortion_crux_probe"]["hard_pair_rows"][0]["pair_index"] == 0
    assert report["hard_pair_coverage"]["score_axis_hard_pair_coverage"] is False
    assert Path(report["segnet_argmax_probe_path"]).is_file()
    assert Path(report["distortion_crux_probe_path"]).is_file()
    assert report["score_claim"] is False
    assert Path(report["quality_gate_path"]).is_file()


def test_hi_nerv_receiver_cache_quality_passes_with_argmax_probe(
    tmp_path: Path,
) -> None:
    archive = _write_tiny_hiv1_archive(tmp_path / "archive.zip")
    reference_report = write_hi_nerv_receiver_cache_quality_report(
        archive_zip_path=archive,
        output_dir=tmp_path / "reference",
        max_pairs=1,
        batch_pairs=1,
    )

    def fake_segnet_logits(x_nchw: np.ndarray) -> np.ndarray:
        b, _c, h, w = x_nchw.shape
        logits = np.zeros((b, 5, h, w), dtype=np.float32)
        logits[:, 0, :, :] = 1.0
        return logits

    report = write_hi_nerv_receiver_cache_quality_report(
        archive_zip_path=archive,
        output_dir=tmp_path / "candidate",
        reference_cache_dir=Path(reference_report["candidate_cache_dir"]),
        max_pairs=1,
        batch_pairs=1,
        min_segnet_std=0.0,
        min_segnet_dynamic_range=0.0,
        max_segnet_mae_vs_reference_for_fit_gate=1.0,
        min_posenet_yuv6_std=0.0,
        min_posenet_yuv6_dynamic_range=0.0,
        max_posenet_yuv6_mae_vs_reference_for_fit_gate=1.0,
        segnet_argmax_probe_logits_fn=fake_segnet_logits,
        require_mlx_scorer_response_probe=False,
    )

    assert report["quality_gate_passed"] is True
    assert report["segnet_argmax_probe"]["scorer_backend"] == "injected_segnet_logits_fn"
    assert report["segnet_argmax_probe"]["fit_gate_passed"] is True
    assert report["segnet_argmax_probe"]["segnet_argmax_disagreement_rate"] == 0.0
    assert "candidate_segnet_argmax_disagreement_too_high" not in report["blockers"]
    assert Path(report["segnet_argmax_probe_path"]).is_file()


def test_hi_nerv_receiver_cache_quality_requires_mlx_scorer_response_probe(
    tmp_path: Path,
) -> None:
    archive = _write_tiny_hiv1_archive(tmp_path / "archive.zip")
    reference_report = write_hi_nerv_receiver_cache_quality_report(
        archive_zip_path=archive,
        output_dir=tmp_path / "reference",
        max_pairs=1,
        batch_pairs=1,
    )

    def fake_segnet_logits(x_nchw: np.ndarray) -> np.ndarray:
        b, _c, h, w = x_nchw.shape
        logits = np.zeros((b, 5, h, w), dtype=np.float32)
        logits[:, 0, :, :] = 1.0
        return logits

    def fake_response_payload(**kwargs):
        return {
            "schema": "mlx_scorer_response.v1",
            "score_axis": "[macOS-MLX research-signal]",
            "hardware_substrate": "MLX cpu",
            "archive_size_bytes": kwargs["archive_size_bytes"],
            "archive_sha256": "a" * 64,
            "avg_posenet_dist": 0.0025,
            "avg_segnet_dist": 0.0,
            "canonical_score": 0.2,
            "score_rate_contribution": 0.001,
            "n_samples": 1,
            "cache_identity": {
                "candidate": {
                    "candidate_cache_identity_mode": "unaudited_debug_override"
                }
            },
            "components": {},
            "blockers": ["mlx_scorer_response_is_false_authority"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    report = write_hi_nerv_receiver_cache_quality_report(
        archive_zip_path=archive,
        output_dir=tmp_path / "candidate",
        reference_cache_dir=Path(reference_report["candidate_cache_dir"]),
        max_pairs=1,
        batch_pairs=1,
        min_segnet_std=0.0,
        min_segnet_dynamic_range=0.0,
        max_segnet_mae_vs_reference_for_fit_gate=1.0,
        min_posenet_yuv6_std=0.0,
        min_posenet_yuv6_dynamic_range=0.0,
        max_posenet_yuv6_mae_vs_reference_for_fit_gate=1.0,
        segnet_argmax_probe_logits_fn=fake_segnet_logits,
        require_mlx_scorer_response_probe=True,
        mlx_scorer_response_payload_fn=fake_response_payload,
    )

    assert report["quality_gate_passed"] is True
    assert Path(report["reference_cache_dir"]) == Path(
        reference_report["candidate_cache_dir"]
    )
    probe = report["mlx_scorer_response_probe"]
    assert probe["schema"] == HI_NERV_RECEIVER_CACHE_MLX_SCORER_RESPONSE_PROBE_SCHEMA
    assert probe["scorer_backend"] == "injected_mlx_scorer_response_payload_fn"
    assert probe["fit_gate_passed"] is True
    assert probe["avg_posenet_dist"] == pytest.approx(0.0025)
    assert Path(report["mlx_scorer_response_probe_path"]).is_file()


def test_hi_nerv_receiver_cache_mlx_scorer_response_probe_blocks_pose_mse(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate_cache"
    reference = tmp_path / "reference_cache"
    candidate.mkdir()
    reference.mkdir()

    def fake_response_payload(**kwargs):
        return {
            "schema": "mlx_scorer_response.v1",
            "score_axis": "[macOS-MLX research-signal]",
            "hardware_substrate": "MLX cpu",
            "archive_size_bytes": kwargs["archive_size_bytes"],
            "avg_posenet_dist": 0.25,
            "avg_segnet_dist": 0.01,
            "canonical_score": 1.59,
            "score_rate_contribution": 0.001,
            "n_samples": kwargs["max_pairs"],
            "cache_identity": {
                "candidate": {
                    "candidate_cache_identity_mode": "unaudited_debug_override"
                }
            },
            "components": {},
            "blockers": ["mlx_scorer_response_is_false_authority"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    report = build_hi_nerv_receiver_cache_mlx_scorer_response_probe(
        candidate_cache_dir=candidate,
        reference_cache_dir=reference,
        archive_size_bytes=123,
        output_json=tmp_path / "response_probe.json",
        sample_pairs=2,
        max_posenet_dist_for_fit_gate=0.01,
        max_segnet_dist_for_fit_gate=0.25,
        response_payload_fn=fake_response_payload,
    )

    assert report["fit_gate_passed"] is False
    assert "hi_nerv_receiver_cache_posenet_response_too_high" in report["blockers"]
    assert "hi_nerv_receiver_cache_segnet_response_too_high" not in report["blockers"]
    assert Path(report["report_path"]).is_file()


def test_hi_nerv_receiver_cache_segnet_argmax_probe_prices_real_flip_surface(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate_cache"
    reference = tmp_path / "reference_cache"
    candidate.mkdir()
    reference.mkdir()
    ref = np.zeros((1, 3, 4, 4), dtype=np.float32)
    cand = ref.copy()
    cand[0, 0, 0, 0] = 255.0
    np.save(reference / "segnet_last_rgb.npy", ref)
    np.save(candidate / "segnet_last_rgb.npy", cand)

    def fake_segnet_logits(x_nchw: np.ndarray) -> np.ndarray:
        b, _c, h, w = x_nchw.shape
        logits = np.zeros((b, 5, h, w), dtype=np.float32)
        logits[:, 0, :, :] = 1.0
        logits[:, 1, :, :] = (x_nchw[:, 0, :, :] > 128.0).astype(np.float32) * 3.0
        return logits

    report = build_hi_nerv_receiver_cache_segnet_argmax_probe(
        candidate_cache_dir=candidate,
        reference_cache_dir=reference,
        upstream_dir=tmp_path / "upstream",
        sample_pairs=1,
        batch_frames=1,
        max_segnet_argmax_disagreement_for_fit_gate=0.05,
        segnet_logits_fn=fake_segnet_logits,
    )

    assert report["schema"] == HI_NERV_RECEIVER_CACHE_SEGNET_ARGMAX_PROBE_SCHEMA
    assert report["scorer_backend"] == "injected_segnet_logits_fn"
    assert report["total_pixels"] == 16
    assert report["mismatch_pixels"] == 1
    assert report["segnet_argmax_disagreement_rate"] == pytest.approx(1.0 / 16.0)
    assert report["target_region_error_score_contribution"] == pytest.approx(6.25)
    assert report["target_region_error_worst_class"] == 0
    assert report["target_region_error_worst_score_contribution"] == pytest.approx(
        6.25
    )
    assert report["target_region_error_profile"][0]["mismatch_pixels"] == 1
    assert report["target_region_error_profile"][0][
        "segnet_score_contribution"
    ] == pytest.approx(6.25)
    assert report["fit_gate_passed"] is False
    assert "candidate_segnet_argmax_disagreement_too_high" in report["blockers"]


def test_hi_nerv_receiver_cache_segnet_argmax_probe_names_class_collapse(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate_cache"
    reference = tmp_path / "reference_cache"
    candidate.mkdir()
    reference.mkdir()
    cand = np.zeros((1, 3, 4, 4), dtype=np.float32)
    ref = np.zeros((1, 3, 4, 4), dtype=np.float32)
    ref[0, 0, :, :] = np.arange(16, dtype=np.float32).reshape(4, 4)
    np.save(candidate / "segnet_last_rgb.npy", cand)
    np.save(reference / "segnet_last_rgb.npy", ref)

    def fake_segnet_logits(x_nchw: np.ndarray) -> np.ndarray:
        b, _c, h, w = x_nchw.shape
        logits = np.zeros((b, 5, h, w), dtype=np.float32)
        cls = (x_nchw[:, 0, :, :].astype(np.int64) % 5).reshape(b, h, w)
        for class_index in range(5):
            logits[:, class_index, :, :] = np.where(cls == class_index, 2.0, 0.0)
        return logits

    report = build_hi_nerv_receiver_cache_segnet_argmax_probe(
        candidate_cache_dir=candidate,
        reference_cache_dir=reference,
        upstream_dir=None,
        sample_pairs=1,
        batch_frames=1,
        max_segnet_argmax_disagreement_for_fit_gate=0.05,
        segnet_logits_fn=fake_segnet_logits,
    )

    assert report["candidate_argmax_histogram"] == [16, 0, 0, 0, 0]
    assert report["reference_argmax_histogram"] == [4, 3, 3, 3, 3]
    assert report["candidate_occupied_class_fraction"] == pytest.approx(0.2)
    assert report["reference_occupied_class_fraction"] == pytest.approx(1.0)
    assert report["fit_gate_passed"] is False
    assert (
        "hi_nerv_receiver_cache_segnet_argmax_class_collapse"
        in report["blockers"]
    )


def test_hi_nerv_receiver_cache_segnet_argmax_probe_ignores_one_pixel_class_crumb(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate_cache"
    reference = tmp_path / "reference_cache"
    candidate.mkdir()
    reference.mkdir()
    cand = np.zeros((1, 3, 4, 4), dtype=np.float32)
    ref = np.zeros((1, 3, 4, 4), dtype=np.float32)
    cand[0, 0, 0, 0] = 1.0
    ref[0, 0, :, :] = np.arange(16, dtype=np.float32).reshape(4, 4)
    np.save(candidate / "segnet_last_rgb.npy", cand)
    np.save(reference / "segnet_last_rgb.npy", ref)

    def fake_segnet_logits(x_nchw: np.ndarray) -> np.ndarray:
        b, _c, h, w = x_nchw.shape
        logits = np.zeros((b, 5, h, w), dtype=np.float32)
        cls = (x_nchw[:, 0, :, :].astype(np.int64) % 5).reshape(b, h, w)
        for class_index in range(5):
            logits[:, class_index, :, :] = np.where(cls == class_index, 2.0, 0.0)
        return logits

    report = build_hi_nerv_receiver_cache_segnet_argmax_probe(
        candidate_cache_dir=candidate,
        reference_cache_dir=reference,
        upstream_dir=None,
        sample_pairs=1,
        batch_frames=1,
        max_segnet_argmax_disagreement_for_fit_gate=1.0,
        segnet_logits_fn=fake_segnet_logits,
    )

    assert report["candidate_argmax_histogram"] == [15, 1, 0, 0, 0]
    assert report["candidate_any_occupied_class_fraction"] == pytest.approx(0.4)
    assert report["candidate_occupied_class_fraction"] == pytest.approx(0.2)
    assert report["thresholds"]["min_class_pixel_count_for_occupancy"] == 2
    assert report["fit_gate_passed"] is False
    assert (
        "hi_nerv_receiver_cache_segnet_argmax_class_collapse"
        in report["blockers"]
    )


def test_hi_nerv_receiver_cache_segnet_argmax_probe_uses_configured_class_gate(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate_cache"
    reference = tmp_path / "reference_cache"
    candidate.mkdir()
    reference.mkdir()
    cand = np.zeros((1, 3, 4, 4), dtype=np.float32)
    ref = np.zeros((1, 3, 4, 4), dtype=np.float32)
    ref[0, 0, :, :] = np.arange(16, dtype=np.float32).reshape(4, 4)
    np.save(candidate / "segnet_last_rgb.npy", cand)
    np.save(reference / "segnet_last_rgb.npy", ref)

    def fake_segnet_logits(x_nchw: np.ndarray) -> np.ndarray:
        b, _c, h, w = x_nchw.shape
        logits = np.zeros((b, 5, h, w), dtype=np.float32)
        cls = (x_nchw[:, 0, :, :].astype(np.int64) % 5).reshape(b, h, w)
        for class_index in range(5):
            logits[:, class_index, :, :] = np.where(cls == class_index, 2.0, 0.0)
        return logits

    report = build_hi_nerv_receiver_cache_segnet_argmax_probe(
        candidate_cache_dir=candidate,
        reference_cache_dir=reference,
        upstream_dir=None,
        sample_pairs=1,
        batch_frames=1,
        max_segnet_argmax_disagreement_for_fit_gate=1.0,
        min_segnet_argmax_occupied_class_fraction_for_fit_gate=0.2,
        min_segnet_argmax_target_class_coverage_fraction_for_fit_gate=0.0,
        segnet_logits_fn=fake_segnet_logits,
    )

    assert report["candidate_occupied_class_fraction"] == pytest.approx(0.2)
    assert report["thresholds"]["min_candidate_occupied_class_fraction"] == (
        pytest.approx(0.2)
    )
    assert report["fit_gate_passed"] is True
    assert (
        "hi_nerv_receiver_cache_segnet_argmax_class_collapse"
        not in report["blockers"]
    )


def test_hi_nerv_receiver_cache_segnet_argmax_probe_blocks_target_class_loss(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate_cache"
    reference = tmp_path / "reference_cache"
    candidate.mkdir()
    reference.mkdir()
    cand = np.zeros((1, 3, 4, 4), dtype=np.float32)
    ref = np.zeros((1, 3, 4, 4), dtype=np.float32)
    cand[0, 0, :2, :] = 0.0
    cand[0, 0, 2:, :] = 2.0
    ref[0, 0, :2, :] = 0.0
    ref[0, 0, 2:, :] = 1.0
    np.save(candidate / "segnet_last_rgb.npy", cand)
    np.save(reference / "segnet_last_rgb.npy", ref)

    def fake_segnet_logits(x_nchw: np.ndarray) -> np.ndarray:
        b, _c, h, w = x_nchw.shape
        logits = np.zeros((b, 5, h, w), dtype=np.float32)
        cls = x_nchw[:, 0, :, :].astype(np.int64).reshape(b, h, w)
        for class_index in range(5):
            logits[:, class_index, :, :] = np.where(cls == class_index, 2.0, 0.0)
        return logits

    report = build_hi_nerv_receiver_cache_segnet_argmax_probe(
        candidate_cache_dir=candidate,
        reference_cache_dir=reference,
        upstream_dir=None,
        sample_pairs=1,
        batch_frames=1,
        max_segnet_argmax_disagreement_for_fit_gate=1.0,
        min_segnet_argmax_occupied_class_fraction_for_fit_gate=0.2,
        min_segnet_argmax_target_class_coverage_fraction_for_fit_gate=0.8,
        segnet_logits_fn=fake_segnet_logits,
    )

    assert report["candidate_occupied_class_fraction"] == pytest.approx(0.4)
    assert report["reference_occupied_class_fraction"] == pytest.approx(0.4)
    assert report["candidate_target_class_coverage_fraction"] == pytest.approx(0.5)
    assert report["candidate_target_class_covered"][:3] == [True, False, False]
    assert report["fit_gate_passed"] is False
    assert (
        "hi_nerv_receiver_cache_segnet_argmax_target_class_coverage_collapse"
        in report["blockers"]
    )


def test_hi_nerv_receiver_cache_segnet_argmax_probe_rejects_bad_class_gate(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate_cache"
    reference = tmp_path / "reference_cache"
    candidate.mkdir()
    reference.mkdir()
    np.save(
        candidate / "segnet_last_rgb.npy",
        np.zeros((1, 3, 2, 2), dtype=np.float32),
    )
    np.save(
        reference / "segnet_last_rgb.npy",
        np.zeros((1, 3, 2, 2), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="occupied_class_fraction"):
        build_hi_nerv_receiver_cache_segnet_argmax_probe(
            candidate_cache_dir=candidate,
            reference_cache_dir=reference,
            upstream_dir=None,
            sample_pairs=1,
            batch_frames=1,
            max_segnet_argmax_disagreement_for_fit_gate=1.0,
            min_segnet_argmax_occupied_class_fraction_for_fit_gate=1.01,
            segnet_logits_fn=lambda x: np.zeros(
                (x.shape[0], 5, x.shape[2], x.shape[3]), dtype=np.float32
            ),
        )


def test_hi_nerv_receiver_cache_scorer_input_distribution_gate_passes_motion(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate_cache"
    reference = tmp_path / "reference_cache"
    _write_scorer_input_cache(candidate, temporal_delta=3.0)
    _write_scorer_input_cache(reference, temporal_delta=2.5)

    report = build_hi_nerv_receiver_cache_scorer_input_distribution_gate(
        candidate_cache_dir=candidate,
        reference_cache_dir=reference,
        sample_pairs=2,
        min_segnet_last_rgb_std=0.1,
        min_segnet_last_rgb_dynamic_range=1.0,
        min_posenet_yuv6_pair_std=0.1,
        min_posenet_yuv6_pair_dynamic_range=1.0,
        min_posenet_yuv6_temporal_signal_std=0.1,
        min_posenet_yuv6_temporal_signal_mean_abs=0.5,
    )

    assert report["fit_gate_passed"] is True
    assert report["posenet_yuv6_temporal_signal"]["candidate_delta_mean_abs"] > 0.5
    assert (
        "candidate_posenet_yuv6_temporal_signal_std_too_low"
        not in report["blockers"]
    )
    assert report["score_claim"] is False


def test_hi_nerv_receiver_cache_scorer_input_distribution_gate_blocks_flat_pose(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate_cache"
    reference = tmp_path / "reference_cache"
    _write_scorer_input_cache(candidate, temporal_delta=0.0)
    _write_scorer_input_cache(reference, temporal_delta=2.5)

    report = build_hi_nerv_receiver_cache_scorer_input_distribution_gate(
        candidate_cache_dir=candidate,
        reference_cache_dir=reference,
        sample_pairs=2,
        min_segnet_last_rgb_std=0.1,
        min_segnet_last_rgb_dynamic_range=1.0,
        min_posenet_yuv6_pair_std=0.1,
        min_posenet_yuv6_pair_dynamic_range=1.0,
        min_posenet_yuv6_temporal_signal_std=0.1,
        min_posenet_yuv6_temporal_signal_mean_abs=0.5,
    )

    assert report["fit_gate_passed"] is False
    assert "candidate_posenet_yuv6_temporal_signal_std_too_low" in report["blockers"]
    assert (
        "candidate_posenet_yuv6_temporal_signal_mean_abs_too_low"
        in report["blockers"]
    )
    assert report["posenet_yuv6_temporal_signal"]["candidate_delta_mean_abs"] == 0.0


def _write_scorer_input_cache(root: Path, *, temporal_delta: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    grid = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 1, 4, 4)
    rgb = np.concatenate([grid, grid + 10.0, grid + 20.0], axis=1)
    first = np.concatenate([grid + channel for channel in range(6)], axis=1)
    second = first + float(temporal_delta) * np.linspace(
        0.5,
        1.5,
        6,
        dtype=np.float32,
    ).reshape(1, 6, 1, 1)
    np.save(root / "segnet_last_rgb.npy", rgb.astype(np.float32))
    np.save(
        root / "posenet_yuv6_pair.npy",
        np.concatenate([first, second], axis=1).astype(np.float32),
    )


def _write_tiny_hiv1_archive(path: Path, *, member_name: str = "0.bin") -> Path:
    cfg = HinervConfig(
        latent_dim_coarse=2,
        latent_dim_mid=2,
        latent_dim_fine=2,
        embed_dim=2,
        initial_grid_h=1,
        initial_grid_w=1,
        decoder_channels=(2, 2, 2),
        sin_frequency=3.0,
        num_upsample_blocks=3,
        mid_injection_block_index=0,
        fine_injection_block_index=1,
        num_pairs=2,
        output_height=8,
        output_width=8,
    )
    torch.manual_seed(7)
    model = HinervSubstrate(cfg).eval()
    decoder_state = {
        key: value
        for key, value in dict(model.state_dict()).items()
        if key not in {"latents_coarse", "latents_mid", "latents_fine"}
    }
    packet = pack_archive(
        decoder_state,
        model.latents_coarse.detach(),
        model.latents_mid.detach(),
        model.latents_fine.detach(),
        {
            "embed_dim": cfg.embed_dim,
            "initial_grid_h": cfg.initial_grid_h,
            "initial_grid_w": cfg.initial_grid_w,
            "decoder_channels": list(cfg.decoder_channels),
            "sin_frequency": cfg.sin_frequency,
            "num_upsample_blocks": cfg.num_upsample_blocks,
            "mid_injection_block_index": cfg.mid_injection_block_index,
            "fine_injection_block_index": cfg.fine_injection_block_index,
            "output_height": cfg.output_height,
            "output_width": cfg.output_width,
        },
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(member_name, packet)
    return path


def _render_tiny_hiv1_targets() -> tuple[np.ndarray, np.ndarray]:
    cfg = HinervConfig(
        latent_dim_coarse=2,
        latent_dim_mid=2,
        latent_dim_fine=2,
        embed_dim=2,
        initial_grid_h=1,
        initial_grid_w=1,
        decoder_channels=(2, 2, 2),
        sin_frequency=3.0,
        num_upsample_blocks=3,
        mid_injection_block_index=0,
        fine_injection_block_index=1,
        num_pairs=2,
        output_height=8,
        output_width=8,
    )
    torch.manual_seed(7)
    model = HinervSubstrate(cfg).eval()
    with torch.no_grad():
        rgb0, rgb1 = model(torch.tensor([0, 1], dtype=torch.long))
    return (
        rgb0.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32),
        rgb1.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32),
    )
