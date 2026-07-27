# SPDX-License-Identifier: MIT
"""Focused exact-public-wire tests for the G111 semantic stage selector."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.witness_dsl import g111_parsed_g105_stage_selector_v1 as subject
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    V9PolarFourierConfigV1,
    V9RuntimeConfigV1,
    Y1WireCodecV1,
)
from tac.witness_dsl.taskspace_g110_generic_two_layer_public_product_v1 import (
    G110OuterZipMethodV1,
    parse_g110_counted_archive_variant,
    parse_g110_public_archive,
    parse_g110_two_layer_v1,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _small_exact_state() -> tuple[
    V9RuntimeConfigV1,
    dict[str, np.ndarray],
    np.ndarray,
]:
    basis = V9PolarFourierConfigV1(
        n_scales=1,
        n_orient0=1,
        f0=1.0,
        base=2.0,
        n_iso=0,
        max_freq=None,
    )
    config = V9RuntimeConfigV1(
        input_dim=basis.input_dim,
        hidden_dim=1,
        hidden_layer_count=1,
        modulation_dim=1,
        softmax_temp=1.0,
        hosc_beta=1.0,
        hosc_omega=1.0,
        chroma=True,
        film_per_layer=False,
        film_concat_code=False,
        basis=basis,
    )
    params = {
        "in_proj.weight": np.zeros((1, 2), dtype=np.float32),
        "in_proj.bias": np.zeros((1,), dtype=np.float32),
        "film.weight": np.zeros((2, 1), dtype=np.float32),
        "film.bias": np.zeros((2,), dtype=np.float32),
        "hidden.0.weight": np.zeros((1, 1), dtype=np.float32),
        "hidden.0.bias": np.zeros((1,), dtype=np.float32),
        "out_sdf.weight": np.zeros((5, 1), dtype=np.float32),
        "out_sdf.bias": np.zeros((5,), dtype=np.float32),
        "out_tex.weight": np.zeros((3, 1), dtype=np.float32),
        "out_tex.bias": np.zeros((3,), dtype=np.float32),
        "palette": np.zeros((5, 3), dtype=np.float32),
    }
    odd_y1 = np.zeros((600, 1), dtype=np.float32)
    return config, params, odd_y1


def _alternative(
    *,
    codec: Y1WireCodecV1,
    method: G110OuterZipMethodV1,
    archive_bytes: int,
) -> subject.G111SemanticStageAlternativeV1:
    return subject.G111SemanticStageAlternativeV1(
        y1_wire_codec=codec,
        outer_zip_method=method,
        d_seg=0.0,
        disagreement_pixels=0,
        semantic_packet=b"semantic",
        product_packet=b"product",
        archive=b"a" * archive_bytes,
        g105_quantization_receipt_sha256=_sha("quantization"),
        scorer_y1_population_sha256=_sha("scorer-y1"),
        camera_y1_population_sha256=_sha("camera-y1"),
        predicted_labels_sha256=_sha("predicted"),
        batch_progress_key_sha256=_sha("progress-key"),
        batch_receipt_chain_sha256=_sha("progress-chain"),
    )


def test_pure_same_object_action_selection_has_stable_tie_break() -> None:
    longer = _alternative(
        codec=Y1WireCodecV1.RAW_I16_LE,
        method=G110OuterZipMethodV1.STORE,
        archive_bytes=101,
    )
    rice_deflate = _alternative(
        codec=Y1WireCodecV1.DELTA_RICE_BEST_K,
        method=G110OuterZipMethodV1.DEFLATE,
        archive_bytes=100,
    )
    raw_deflate = _alternative(
        codec=Y1WireCodecV1.RAW_I16_LE,
        method=G110OuterZipMethodV1.DEFLATE,
        archive_bytes=100,
    )

    selected = subject.select_semantic_stage_alternative((longer, rice_deflate, raw_deflate))
    assert selected is raw_deflate
    assert selected.semantic_action == subject.semantic_stage_action(
        d_seg=0.0,
        archive_bytes=100,
    )


def test_failure_scope_is_exact_to_the_semantic_lower_bound() -> None:
    distortion_blocked = subject.semantic_stage_conditional_observation(
        d_seg=0.01,
        archive_bytes=1,
        effective_frontier_target=1.0,
    )
    assert distortion_blocked["conditional_disposition"] == "SAME_OBJECT_DISTORTION_LOWER_BOUND_OBSTRUCTION"
    assert distortion_blocked["distortion_only_lower_bound_obstruction_conditional_observation"] is True
    assert distortion_blocked["strict_lower_bound_condition_conditional_observation"] is True
    assert distortion_blocked["strict_lower_bound_condition_scope"] == "this_parsed_G105_Y1_stage_pointer_identity_only"
    assert distortion_blocked["family_wide_claim"] is False
    assert distortion_blocked["semantic_archive_exhaustion_conditional_observation"] is False

    semantic_exhausted = subject.semantic_stage_conditional_observation(
        d_seg=0.0,
        archive_bytes=1_000_000,
        effective_frontier_target=0.5,
    )
    assert semantic_exhausted["conditional_disposition"] == "DEFER_POST_G105_POSE_REFIT"
    assert semantic_exhausted["distortion_only_frontier_clear_conditional_observation"] is True
    assert semantic_exhausted["semantic_action_frontier_clear_conditional_observation"] is False
    assert semantic_exhausted["semantic_archive_exhaustion_conditional_scope"] == "DEFER_POST_G105_POSE_REFIT"
    assert semantic_exhausted["strict_lower_bound_condition_conditional_observation"] is False

    path_open = subject.semantic_stage_conditional_observation(
        d_seg=0.0,
        archive_bytes=1,
        effective_frontier_target=0.5,
    )
    assert path_open["conditional_disposition"] == "POSE_REFIT_PATH_OPEN_IF_INPUT_CUSTODY_CLOSES"
    assert path_open["semantic_archive_exhaustion_conditional_scope"] is None


def test_pose_reserve_distinguishes_zero_from_measured_custody() -> None:
    assert subject._pose_reserve_record(
        measured_pose_reserve_bytes=None,
        measured_pose_reserve_receipt_sha256=None,
    ) == {
        "status": "zero_unmeasured_semantic_only",
        "bytes": 0,
        "receipt_sha256": None,
        "included_in_semantic_selection": False,
    }
    receipt_sha256 = _sha("measured-pose-reserve")
    assert subject._pose_reserve_record(
        measured_pose_reserve_bytes=7_200,
        measured_pose_reserve_receipt_sha256=receipt_sha256,
    ) == {
        "status": "measured_reserved_bytes",
        "bytes": 7_200,
        "receipt_sha256": receipt_sha256,
        "included_in_semantic_selection": False,
    }


def test_pointer_identity_namespaces_historical_stage_artifacts() -> None:
    first = subject._artifact_namespace(
        stage_tag="ce_stage_0001",
        pointer_snapshot_identity_sha256=_sha("pointer-a"),
    )
    second = subject._artifact_namespace(
        stage_tag="ce_stage_0001",
        pointer_snapshot_identity_sha256=_sha("pointer-b"),
    )
    assert first != second
    assert first.startswith("ce_stage_0001.ptr_")
    assert second.startswith("ce_stage_0001.ptr_")


def test_exact_stage_selector_uses_rank_zero_g110_and_full_batch16_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, params, odd_y1 = _small_exact_state()
    target_labels = np.zeros((600, 384, 512), dtype=np.uint8)
    scorer_y1 = np.zeros((384, 512, 3), dtype=np.uint8)
    rank_zero_pair = np.stack((scorer_y1, scorer_y1))
    camera_y1 = np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(
        subject,
        "render_scorer_y1",
        lambda _program, _pair_id: scorer_y1,
    )
    monkeypatch.setattr(
        subject,
        "render_g110_rank_zero_scorer_pair",
        lambda _packet, _pair_id: rank_zero_pair,
    )
    monkeypatch.setattr(
        subject,
        "realize_factor2_uint8_numpy",
        lambda _scorer_y1: camera_y1,
    )
    monkeypatch.setattr(subject, "CAMERA_HW", (2, 2))

    batch_sizes: list[int] = []

    def scorer(camera_batch: np.ndarray) -> np.ndarray:
        batch_sizes.append(camera_batch.shape[0])
        prediction = np.zeros(
            (camera_batch.shape[0], 384, 512),
            dtype=np.uint8,
        )
        prediction[0, 0, 0] = 1
        return prediction

    out_dir = tmp_path.resolve() / "durable"
    progress_dir = tmp_path.resolve() / "progress"
    selection = subject.compile_select_parsed_g105_stage_v1(
        config=config,
        semantic_params=params,
        odd_y1=odd_y1,
        target_labels=target_labels,
        seg_argmax_batch_scorer=scorer,
        injected_inputs_are_test_only=True,
        seg_scorer_identity_sha256=_sha("seg-scorer"),
        source_checkpoint_identity_sha256=_sha("source-checkpoint"),
        pose_initializer_identity_sha256=_sha("pose-initializer"),
        effective_frontier_target=0.15,
        pointer_snapshot_identity_sha256=_sha("pointer"),
        out_dir=out_dir,
        progress_dir=progress_dir,
        stage_tag="ce_stage_0001",
    )

    assert tuple(batch_sizes) == subject.VERDICT_BATCH_SIZES * 2
    assert selection.selected.disagreement_pixels == 38
    assert selection.selected.d_seg == 38 / (600 * 384 * 512)
    assert selection.packet_path.read_bytes() == selection.selected.product_packet
    assert selection.archive_path.read_bytes() == selection.selected.archive
    parsed_product = parse_g110_two_layer_v1(selection.selected.product_packet)
    assert parsed_product.basis_q.shape == (0, 0, 0, 3)
    assert parsed_product.combined_scales.shape == (0,)
    assert parsed_product.coefficients_q.shape == (600, 0)
    assert (
        parse_g110_counted_archive_variant(
            selection.selected.archive,
            selection.selected.outer_zip_method,
        )
        == selection.selected.product_packet
    )
    assert parse_g110_public_archive(selection.selected.archive) == selection.selected.product_packet

    receipt = json.loads(selection.receipt_path.read_text("ascii"))
    assert receipt == selection.receipt
    assert receipt["artifact_namespace"] in selection.receipt_path.name
    assert receipt["frontier_snapshot_scope"] == {
        "conditional_path_is_snapshot_scoped": True,
        "production_current_status_confirmed_after_screen": False,
        "post_screen_dynamic_frontier_reverify_required": True,
        "new_pointer_identity_requires_new_artifact_namespace": True,
    }
    assert receipt["execution_surface"] == subject.EXECUTION_SURFACE
    assert receipt["engine_only"] is True
    assert receipt["injected_inputs_are_test_only"] is True
    assert receipt["production_authority_closed"] is False
    assert receipt["production_wrapper_required"] is True
    assert receipt["production_verdict_emitted"] is False
    assert receipt["production_admission"] is False
    assert len(receipt["alternatives"]) == 4
    assert {(row["y1_wire_codec"], row["outer_zip_method"]) for row in receipt["alternatives"]} == {
        ("RAW_I16_LE", "STORE"),
        ("RAW_I16_LE", "DEFLATE"),
        ("DELTA_RICE_BEST_K", "STORE"),
        ("DELTA_RICE_BEST_K", "DEFLATE"),
    }
    selected_row = min(
        receipt["alternatives"],
        key=lambda row: (
            row["semantic_action"],
            row["d_seg"],
            row["archive_bytes"],
            int(Y1WireCodecV1[row["y1_wire_codec"]]),
            int(G110OuterZipMethodV1[row["outer_zip_method"]]),
            row["archive_sha256"],
        ),
    )
    assert receipt["selected"] == selected_row
    assert receipt["batch_geometry"] == {
        "maximum_batch_size": 16,
        "batch_count_per_wire_family": 38,
        "batch_sizes": [16] * 37 + [8],
        "wire_family_count": 2,
        "total_scorer_callback_calls": 76,
    }
    conditional = receipt["conditional_observation"]
    assert conditional["distortion_only_frontier_clear_conditional_observation"] is True
    assert conditional["semantic_action_frontier_clear_conditional_observation"] is True
    assert conditional["optimistic_pose_refit_path_open_conditional_observation"] is True
    assert conditional["conditional_disposition"] == "POSE_REFIT_PATH_OPEN_IF_INPUT_CUSTODY_CLOSES"
    assert receipt["pose_reserve"] == {
        "status": "zero_unmeasured_semantic_only",
        "bytes": 0,
        "receipt_sha256": None,
        "included_in_semantic_selection": False,
    }
    assert len(receipt["pareto_ledger"]) == 4
    assert all(
        row["wire_regret"] == 0.0
        and row["wire_regret_scope"] == "relative_to_best_parsed_G105_wire_family_same_source"
        and row["pose_initializer_identity_sha256"] == _sha("pose-initializer")
        for row in receipt["pareto_ledger"]
    )
    assert receipt["source_float_to_parsed_wire_regret"]["measured"] is False
    assert len(selection.alternatives) == 4
    assert selection.cross_stage_pareto_row.to_dict() == receipt["cross_stage_pareto_row"]
    cross_stage_row = receipt["cross_stage_pareto_row"]
    assert cross_stage_row["schema"] == subject.CROSS_STAGE_PARETO_ROW_SCHEMA
    assert cross_stage_row["lineage_identity"]["source_checkpoint_identity_sha256"] == _sha("source-checkpoint")
    assert len(cross_stage_row["stage_pareto_rows"]) == 4
    assert (
        cross_stage_row["paired_resume_identity"]["batch_progress_key_sha256"]
        == selection.selected.batch_progress_key_sha256
    )
    assert (
        cross_stage_row["selected_deploy_identity"]["archive_sha256"]
        == hashlib.sha256(selection.selected.archive).hexdigest()
    )
    assert (
        cross_stage_row["cross_stage_retention_policy"] == "retain_nondominated_and_semantically_second_best_stage_rows"
    )
    assert cross_stage_row["production_admission"] is False
    assert cross_stage_row["production_wrapper_required"] is True
    assert receipt["batch_progress"] == {
        "schema": subject.BATCH_PROGRESS_SCHEMA,
        "store_supplied_by_caller": True,
        "atomic_rows": True,
        "rows_preserved": True,
        "rows_per_wire_family": 38,
        "wire_family_count": 2,
        "completed_rows": 76,
        "reuse_requires_verified_identity_and_body_hash": True,
        "identity_binds_semantic_packet_and_archive_variants": True,
        "identity_binds_source_labels_pointer_and_scorer": True,
    }
    progress_rows = sorted(progress_dir.rglob("batch_*.json"))
    assert len(progress_rows) == 76
    assert all(json.loads(path.read_text("ascii"))["schema"] == subject.BATCH_PROGRESS_SCHEMA for path in progress_rows)
    assert receipt["production_lower_bound_verdict_emitted"] is False
    assert receipt["family_wide_claim"] is False
    assert receipt["incomplete_semantic_only"] is True
    assert receipt["candidate_claim"] is False
    assert receipt["score_claim"] is False
    assert receipt["pointer_moved"] is False
    assert receipt["scorer_weights_emitted"] is False
    assert receipt["ground_truth_emitted"] is False
    assert not any(out_dir.glob(".*.tmp.*"))

    resumed = subject.compile_select_parsed_g105_stage_v1(
        config=config,
        semantic_params=params,
        odd_y1=odd_y1,
        target_labels=target_labels,
        seg_argmax_batch_scorer=lambda _batch: pytest.fail("verified batch progress must suppress scorer replay"),
        injected_inputs_are_test_only=True,
        seg_scorer_identity_sha256=_sha("seg-scorer"),
        source_checkpoint_identity_sha256=_sha("source-checkpoint"),
        pose_initializer_identity_sha256=_sha("pose-initializer"),
        effective_frontier_target=0.15,
        pointer_snapshot_identity_sha256=_sha("pointer"),
        out_dir=out_dir,
        progress_dir=progress_dir,
        stage_tag="ce_stage_0001",
    )
    assert resumed.receipt == selection.receipt
    assert resumed.selected.archive == selection.selected.archive

    corrupt_path = progress_rows[0]
    corrupt_row = json.loads(corrupt_path.read_text("ascii"))
    corrupt_row["disagreement_pixels"] += 1
    corrupt_path.write_text(
        json.dumps(
            corrupt_row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n",
        encoding="ascii",
    )
    with pytest.raises(
        subject.G111ParsedG105StageSelectorError,
        match="identity or body hash differs",
    ):
        subject._load_verified_batch_row(
            path=corrupt_path,
            identity=corrupt_row["identity"],
            progress_key_sha256=corrupt_row["progress_key_sha256"],
            batch_index=corrupt_row["batch_index"],
            pair_start=corrupt_row["pair_start"],
            pair_stop=corrupt_row["pair_stop"],
            target_batch_sha256=corrupt_row["target_batch_sha256"],
        )


def test_exact_inputs_and_scorer_output_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, params, odd_y1 = _small_exact_state()
    target_labels = np.zeros((600, 384, 512), dtype=np.uint8)
    with pytest.raises(
        subject.G111ParsedG105StageSelectorError,
        match="production requires the physical authority wrapper",
    ):
        subject.compile_select_parsed_g105_stage_v1(
            config=config,
            semantic_params=params,
            odd_y1=odd_y1,
            target_labels=target_labels,
            seg_argmax_batch_scorer=lambda _batch: target_labels[:16],
            injected_inputs_are_test_only=False,
            seg_scorer_identity_sha256=_sha("seg-scorer"),
            source_checkpoint_identity_sha256=_sha("source-checkpoint"),
            pose_initializer_identity_sha256=_sha("pose-initializer"),
            effective_frontier_target=0.15,
            pointer_snapshot_identity_sha256=_sha("pointer"),
            out_dir=(tmp_path / "must-refuse-production").resolve(),
            progress_dir=(tmp_path / "must-refuse-production-progress").resolve(),
            stage_tag="stage",
        )

    extra = {**params, "unexpected": np.ones((1,), dtype=np.float32)}
    with pytest.raises(
        subject.G111ParsedG105StageSelectorError,
        match="parameter census",
    ):
        subject.compile_select_parsed_g105_stage_v1(
            config=config,
            semantic_params=extra,
            odd_y1=odd_y1,
            target_labels=target_labels,
            seg_argmax_batch_scorer=lambda _batch: target_labels[:16],
            injected_inputs_are_test_only=True,
            seg_scorer_identity_sha256=_sha("seg-scorer"),
            source_checkpoint_identity_sha256=_sha("source-checkpoint"),
            pose_initializer_identity_sha256=_sha("pose-initializer"),
            effective_frontier_target=0.15,
            pointer_snapshot_identity_sha256=_sha("pointer"),
            out_dir=(tmp_path / "unused").resolve(),
            progress_dir=(tmp_path / "unused-progress").resolve(),
            stage_tag="stage",
        )

    with pytest.raises(
        subject.G111ParsedG105StageSelectorError,
        match="temporary root",
    ):
        subject.compile_select_parsed_g105_stage_v1(
            config=config,
            semantic_params=params,
            odd_y1=odd_y1,
            target_labels=target_labels,
            seg_argmax_batch_scorer=lambda _batch: target_labels[:16],
            injected_inputs_are_test_only=True,
            seg_scorer_identity_sha256=_sha("seg-scorer"),
            source_checkpoint_identity_sha256=_sha("source-checkpoint"),
            pose_initializer_identity_sha256=_sha("pose-initializer"),
            effective_frontier_target=0.15,
            pointer_snapshot_identity_sha256=_sha("pointer"),
            out_dir=Path("/tmp/g117-must-refuse"),
            progress_dir=(tmp_path / "unused-progress").resolve(),
            stage_tag="stage",
        )

    scorer_y1 = np.zeros((384, 512, 3), dtype=np.uint8)
    monkeypatch.setattr(
        subject,
        "render_scorer_y1",
        lambda _program, _pair_id: scorer_y1,
    )
    monkeypatch.setattr(
        subject,
        "render_g110_rank_zero_scorer_pair",
        lambda _packet, _pair_id: np.stack((scorer_y1, scorer_y1)),
    )
    monkeypatch.setattr(
        subject,
        "realize_factor2_uint8_numpy",
        lambda _scorer_y1: np.zeros((2, 2, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(subject, "CAMERA_HW", (2, 2))
    with pytest.raises(
        subject.G111ParsedG105StageSelectorError,
        match="argmax labels",
    ):
        subject.compile_select_parsed_g105_stage_v1(
            config=config,
            semantic_params=params,
            odd_y1=odd_y1,
            target_labels=target_labels,
            seg_argmax_batch_scorer=lambda batch: np.zeros(
                (batch.shape[0], 384, 512),
                dtype=np.int64,
            ),
            injected_inputs_are_test_only=True,
            seg_scorer_identity_sha256=_sha("seg-scorer"),
            source_checkpoint_identity_sha256=_sha("source-checkpoint"),
            pose_initializer_identity_sha256=_sha("pose-initializer"),
            effective_frontier_target=0.15,
            pointer_snapshot_identity_sha256=_sha("pointer"),
            out_dir=(tmp_path / "malformed-scorer").resolve(),
            progress_dir=(tmp_path / "malformed-progress").resolve(),
            stage_tag="stage",
        )
