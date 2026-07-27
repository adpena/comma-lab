from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tac.witness_control import taskspace_post_g105_pose_refit_population_v1 as population
from tac.witness_control import taskspace_post_g105_pose_refit_v1 as subject
from tac.witness_dsl import taskspace_g110_generated_y1_pose_product_v1 as g110
from tac.witness_dsl.v10_factor2_selected_preimage_v1 import (
    realize_factor2_uint8_numpy,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _binding(path: Path) -> dict[str, object]:
    return subject._output_binding(path)


def test_fixed_extremum_gauge_is_scoped_and_immutable() -> None:
    q_levels = 127
    q = np.zeros((subject.PAIR_COUNT, subject.POSE_DIM), dtype=np.int16)
    for dimension in range(subject.POSE_DIM):
        q[dimension, dimension] = q_levels if dimension % 2 == 0 else -q_levels
    scales = np.asarray(
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        dtype=np.float32,
    )
    anchors = subject.deterministic_anchor_rows(q, q_levels=q_levels)
    assert np.array_equal(anchors, np.arange(subject.POSE_DIM, dtype=np.int64))
    subject._validate_fixed_scale_gauge(
        q,
        scales,
        anchors,
        q_levels=q_levels,
    )

    changed = q.copy()
    changed[0, 0] -= 1
    with pytest.raises(
        subject.PostG105PoseRefitError,
        match="fixed XIP2 scale anchor",
    ):
        subject._validate_fixed_scale_gauge(
            changed,
            scales,
            anchors,
            q_levels=q_levels,
        )

    blocker = subject.global_range_reactivation_blocker()
    assert blocker["optimizer_verdict_scope"] == subject.OPTIMIZER_VERDICT_SCOPE
    assert blocker["global_optimality_claim"] is False
    assert "range/reanchor" in str(blocker["required_reactivation"])


def test_gauss_newton_step_descends_linear_pose_residual() -> None:
    jacobian = np.eye(subject.POSE_DIM, dtype=np.float64)[None]
    residual = np.asarray([[3.0, -2.0, 1.0, -0.5, 0.25, -0.125]])
    delta = subject._gauss_newton_delta(
        jacobian,
        residual,
        damping=1e-9,
        trust_radius_q=8,
    )
    assert delta.shape == residual.shape
    assert np.max(np.abs(delta)) <= 8.0
    assert np.sum((residual + delta) ** 2) < np.sum(residual**2)


def test_score_native_key_can_reject_pose_only_local_minimum() -> None:
    lower_pose = np.full((subject.PAIR_COUNT,), 1.0e-4, dtype=np.float64)
    cheaper_wire = np.full((subject.PAIR_COUNT,), 1.21e-4, dtype=np.float64)
    pose_only_key = subject._pose_rate_key(
        pose_losses=lower_pose,
        archive_bytes=10_000,
        proposal_order=0,
    )
    joint_key = subject._pose_rate_key(
        pose_losses=cheaper_wire,
        archive_bytes=100,
        proposal_order=1,
    )
    assert np.mean(lower_pose) < np.mean(cheaper_wire)
    assert joint_key < pose_only_key


def test_checkpoint_npz_is_byte_deterministic_and_pickle_free() -> None:
    arrays = {
        "z": np.arange(12, dtype=np.int16).reshape(3, 4),
        "a": np.asarray("strict"),
    }
    first = subject._deterministic_npz(arrays)
    second = subject._deterministic_npz(dict(reversed(tuple(arrays.items()))))
    assert first == second
    path = Path(__file__).parent / "_not_written.npz"
    assert not path.exists()


def test_cached_v10_operator_is_exact_public_realization() -> None:
    rng = np.random.default_rng(119)
    scorer = rng.integers(
        0,
        256,
        size=(384, 512, 3),
        dtype=np.uint8,
    )
    expected = realize_factor2_uint8_numpy(scorer)
    observed = subject.realize_factor2_uint8_scorer_plane(
        subject._v10_factor2_operator(),
        scorer,
    )
    assert np.array_equal(observed, expected)
    assert subject._v10_factor2_operator() is subject._v10_factor2_operator()


def test_pose_reductions_match_upstream_float32_batch_order() -> None:
    torch = pytest.importorskip("torch")
    oracle = object.__new__(subject.ExactBatch16PoseOracleV1)
    oracle.torch = torch
    oracle.device = torch.device("cpu")
    rng = np.random.default_rng(119)
    outputs = rng.standard_normal((subject.PAIR_COUNT, subject.POSE_DIM)).astype(
        np.float32
    )
    targets = rng.standard_normal((subject.PAIR_COUNT, subject.POSE_DIM)).astype(
        np.float32
    )
    observed_losses = oracle.pose_losses(outputs, targets)
    expected_losses = (
        (torch.from_numpy(outputs) - torch.from_numpy(targets))
        .pow(2)
        .mean(dim=1)
        .to(torch.float64)
        .numpy()
    )
    assert np.array_equal(observed_losses, expected_losses)
    accumulator = torch.zeros([], dtype=torch.float32)
    for start in range(0, subject.PAIR_COUNT, subject.BATCH_PAIRS):
        accumulator += torch.from_numpy(
            expected_losses[start : start + subject.BATCH_PAIRS].astype(
                np.float32
            )
        ).sum()
    assert oracle.population_mse(observed_losses) == (
        accumulator / subject.PAIR_COUNT
    ).item()


def test_archive_oracle_enumerates_full_semantic_xip2_zip_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = np.zeros((subject.PAIR_COUNT, subject.POSE_DIM), dtype=np.int16)
    scales = np.ones((subject.POSE_DIM,), dtype=np.float32)
    packet_rows: dict[bytes, SimpleNamespace] = {}
    archive_rows: dict[bytes, tuple[bytes, object]] = {}

    def encode_packet(
        *,
        semantic_packet: bytes,
        final_y1_binding: str,
        xip2_payload: bytes,
        pitch: float,
    ) -> bytes:
        packet = (
            bytes([len(packet_rows)])
            + semantic_packet
            + xip2_payload
            + np.asarray(pitch, dtype="<f8").tobytes()
        )
        packet_rows[packet] = SimpleNamespace(
            semantic_packet=semantic_packet,
            final_y1_binding_sha256=final_y1_binding,
            q=q,
            scales=scales,
        )
        return packet

    def build_archive(packet: bytes, method: object) -> bytes:
        archive = bytes([len(archive_rows)]) + packet
        archive_rows[archive] = (packet, method)
        return archive

    monkeypatch.setattr(subject, "_encode_g110_packet", encode_packet)
    monkeypatch.setattr(
        subject,
        "parse_g110_generated_y1_pose_v1",
        lambda packet: packet_rows[packet],
    )
    monkeypatch.setattr(subject, "_build_g110_archive_for_method", build_archive)
    monkeypatch.setattr(
        subject,
        "_read_g110_archive_member",
        lambda archive: archive_rows[archive],
    )
    monkeypatch.setattr(
        subject,
        "serialize_xi_payload",
        lambda _q, _scales, *, coder: (
            b"XIP2" + bytes([0]) + b"r"
            if coder == "none"
            else b"XIP2" + bytes([1]) + b"delta-is-longer"
        ),
    )
    oracle = subject.ExactCompleteArchiveRateOracleV1(
        semantic_variants=(
            (
                subject.Y1WireCodecV1.RAW_I16_LE,
                b"raw-semantic",
                _sha("raw-binding"),
            ),
            (
                subject.Y1WireCodecV1.DELTA_RICE_BEST_K,
                b"rice-semantic",
                _sha("rice-binding"),
            ),
        ),
        pitch=0.0,
    )
    xip2, _packet, _archive, selected, alternatives = oracle.materialize(
        q=q,
        scales=scales,
    )
    assert len(alternatives) == 8
    assert {row.xip2_coder for row in alternatives} == {"none", "delta_ar"}
    assert selected.xip2_coder == "none"
    assert xip2[4] == 0


def _population_config(
    tmp_path: Path,
    *,
    manifest: Path,
    target_receipt: Path,
) -> population.PostG105PoseRefitPopulationConfigV1:
    config_path = tmp_path / "population_config.json"
    config_path.write_text("{}", encoding="ascii")
    return population.PostG105PoseRefitPopulationConfigV1(
        config_path=config_path.resolve(),
        config_sha256=_sha("config"),
        run_id="g119-test",
        seed=119,
        output_root=(tmp_path / "out").resolve(),
        g121_retained_prepose=_binding(manifest),
        target_capsule_receipt=_binding(target_receipt),
        q_levels_candidates=(32,),
        local_gauss_newton_stages=1,
        finite_difference_q_steps=1,
        damping=1.0,
        trust_radius_q=1,
        line_search_scales=(1.0,),
        device="cpu",
        torch_num_threads=1,
    )


def test_population_opener_preserves_every_g121_retained_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / population.G121_MANIFEST_BASENAME
    manifest.write_text("typed-g121", encoding="ascii")
    target = tmp_path / "g109.json"
    target.write_text("typed-g109", encoding="ascii")
    completion = tmp_path / "g121_completion_receipt.json"
    completion.write_text("complete", encoding="ascii")
    rows = []
    denominator = population.EXACT_SEG_PIXEL_DENOMINATOR
    target_numerator = 43
    target_denominator = 250
    pointer_sha = _sha("pointer-snapshot")
    postverified_pointer_sha = _sha("postverified-pointer")
    for index, disagreements in enumerate((100_000, 150_000)):
        stage = f"stage_{index}"
        g112 = tmp_path / f"{stage}.g112.json"
        g112.write_text(stage, encoding="ascii")
        pose_sha = _sha(f"{stage}-pose")
        lhs = 100 * disagreements * target_denominator
        rhs = target_numerator * denominator
        rows.append(
            {
                "stage_tag": stage,
                "row_identity_sha256": _sha(f"{stage}-row"),
                # Legacy telemetry is deliberately contradictory.  G119-v2
                # must never consult it for retention.
                "d_seg_wire": 1.0,
                "live_target_score": 0.0001,
                "retained_for_post_g105_pose": False,
                "public_wire_seg": {
                    "disagreement_pixels": disagreements,
                    "pixel_denominator": denominator,
                    "d_seg_rational": {
                        "numerator": disagreements,
                        "denominator": denominator,
                    },
                    "d_seg_display_float": disagreements / denominator,
                    "measurement_identity_sha256": _sha(f"{stage}-measurement"),
                },
                "live_target": {
                    "score_decimal": "0.172",
                    "score_rational": {
                        "numerator": target_numerator,
                        "denominator": target_denominator,
                    },
                    "pointer_snapshot_identity_sha256": pointer_sha,
                    "postverified_pointer_identity_sha256": (postverified_pointer_sha),
                },
                "prepose_obstruction": {
                    "rule": population._EXACT_OBSTRUCTION_RULE,
                    "lhs": str(lhs),
                    "rhs": str(rhs),
                    "strict_distortion_open": True,
                    "disposition": population.G121_RETAIN_DISPOSITION,
                },
                "pose_initializer_identity_sha256": pose_sha,
                "physical_stage_identity": {
                    "g112_partition_receipt": _binding(g112),
                    "g112_pose_initializer": {"sha256": pose_sha},
                },
                "physical_stage_identity_sha256": _sha(f"{stage}-physical"),
            }
        )
    opened = SimpleNamespace(
        schema=population.G121_REQUIRED_SCHEMA,
        rows=rows,
        exhaustive_enumeration_proven=True,
        completion_receipt=_binding(completion),
        manifest_path=manifest.resolve(),
        manifest_sha256=_binding(manifest)["sha256"],
        pointer_snapshot_identity_sha256=pointer_sha,
    )
    monkeypatch.setattr(
        population.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(open_g121_retained_prepose_v1=lambda _path, **_kwargs: opened)
            if name == population.G121_MODULE
            else None
        ),
    )
    result = population._open_g121_retained_population(
        _population_config(
            tmp_path,
            manifest=manifest,
            target_receipt=target,
        )
    )
    assert [row.stage_tag for row in result.stages] == ["stage_0", "stage_1"]
    assert [row.disagreement_pixels for row in result.stages] == [
        100_000,
        150_000,
    ]
    assert result.live_target_score_decimal == "0.172"
    assert result.completion_receipt == _binding(completion)


def test_defer_g115_wire_qat_is_not_misread_as_pose_eligible() -> None:
    denominator = population.EXACT_SEG_PIXEL_DENOMINATOR
    disagreements = 300_000
    target_numerator = 43
    target_denominator = 250
    lhs = 100 * disagreements * target_denominator
    rhs = target_numerator * denominator
    row = {
        "public_wire_seg": {
            "disagreement_pixels": disagreements,
            "pixel_denominator": denominator,
            "d_seg_rational": {
                "numerator": disagreements,
                "denominator": denominator,
            },
            "d_seg_display_float": disagreements / denominator,
            "measurement_identity_sha256": _sha("measurement"),
        },
        "live_target": {
            "score_decimal": "0.172",
            "score_rational": {
                "numerator": target_numerator,
                "denominator": target_denominator,
            },
            "pointer_snapshot_identity_sha256": _sha("pointer"),
            "postverified_pointer_identity_sha256": _sha("postverified"),
        },
        "prepose_obstruction": {
            "rule": population._EXACT_OBSTRUCTION_RULE,
            "lhs": str(lhs),
            "rhs": str(rhs),
            "strict_distortion_open": False,
            "disposition": "DEFER_G115_WIRE_QAT",
        },
    }
    with pytest.raises(
        population.PostG105PoseRefitPopulationError,
        match="fails exact distortion-open cross-product",
    ):
        population._exact_prepose_coordinates(row)


def test_terminal_g115_exact_open_child_is_pose_eligible(
    tmp_path: Path,
) -> None:
    denominator = population.EXACT_SEG_PIXEL_DENOMINATOR
    base_disagreements = 300_000
    terminal_disagreements = 100_000
    target_numerator = 43
    target_denominator = 250
    lhs = 100 * base_disagreements * target_denominator
    rhs = target_numerator * denominator
    physical_sha = _sha("terminal-physical-stage")
    terminal_receipt = tmp_path / "g115_terminal_receipt.json"
    terminal_receipt.write_text("terminal", encoding="ascii")
    row = {
        "public_wire_seg": {
            "disagreement_pixels": base_disagreements,
            "pixel_denominator": denominator,
            "d_seg_rational": {
                "numerator": base_disagreements,
                "denominator": denominator,
            },
            "d_seg_display_float": base_disagreements / denominator,
            "measurement_identity_sha256": _sha("measurement"),
        },
        "live_target": {
            "score_decimal": "0.172",
            "score_rational": {
                "numerator": target_numerator,
                "denominator": target_denominator,
            },
            "pointer_snapshot_identity_sha256": _sha("pointer"),
            "postverified_pointer_identity_sha256": _sha("postverified"),
        },
        "prepose_obstruction": {
            "rule": population._EXACT_OBSTRUCTION_RULE,
            "lhs": str(lhs),
            "rhs": str(rhs),
            "strict_distortion_open": False,
            "disposition": population.G121_RETAIN_DISPOSITION,
        },
        "physical_stage_identity_sha256": physical_sha,
        "g115_qat": {
            "status": "terminal_stage_measured",
            "terminal_stage_physical_identity_sha256": physical_sha,
            "disagreement_pixels": terminal_disagreements,
            "pixel_denominator": denominator,
            "receipt": _binding(terminal_receipt),
        },
    }
    coordinates = population._exact_prepose_coordinates(row)
    assert coordinates[0] == terminal_disagreements
    assert coordinates[2] == terminal_disagreements / denominator


def test_public_population_config_refuses_semantic_best(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(population, "MIN_FREE_BYTES", 0)
    best = tmp_path / "g105_public_wire_best.json"
    best.write_text("{}", encoding="ascii")
    target = tmp_path / "g109.json"
    target.write_text("{}", encoding="ascii")
    output = tmp_path / "out"
    body = {
        "schema": population.CONFIG_SCHEMA,
        "run_id": "refuse-best",
        "seed": 1,
        "output_root": str(output.resolve()),
        "g121_retained_prepose": _binding(best),
        "target_capsule_receipt": _binding(target),
        "q_levels_candidates": [32],
        "local_gauss_newton_stages": 1,
        "finite_difference_q_steps": 1,
        "damping": 1.0,
        "trust_radius_q": 1,
        "line_search_scales": [1.0],
        "device": "cpu",
        "torch_num_threads": 1,
        "research_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
    }
    config = population.seal_population_config(body)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="ascii")
    with pytest.raises(
        population.PostG105PoseRefitPopulationError,
        match=r"only g121_retained_prepose\.json",
    ):
        population.load_population_config(
            path,
            allowed_output_roots=(tmp_path,),
        )


def test_final_checkpoint_and_run_receipt_are_accepted_by_g110(
    tmp_path: Path,
) -> None:
    hashes = {
        name: _sha(name)
        for name in (
            "partition",
            "semantic-child",
            "pose-initializer",
            "deploy",
            "resume",
            "lineage",
            "checkpoint-id",
            "root",
            "projection",
            "capsule",
            "pose-targets",
        )
    }
    xi_initializer = np.zeros(
        (subject.PAIR_COUNT, subject.POSE_DIM),
        dtype=np.float64,
    )
    initializer = SimpleNamespace(xi_init=xi_initializer, pitch=0.03125)
    base = SimpleNamespace(
        partition_receipt_sha256=hashes["partition"],
        semantic_child_sha256=hashes["semantic-child"],
        pose_initializer_sha256=hashes["pose-initializer"],
        source_deploy_checkpoint_sha256=hashes["deploy"],
        source_resume_checkpoint_sha256=hashes["resume"],
        source_lineage_receipt_sha256=hashes["lineage"],
        source_checkpoint_id_sha256=hashes["checkpoint-id"],
        source_root_sha256=hashes["root"],
        target_projection_sha256=hashes["projection"],
        target_capsule_receipt_sha256=hashes["capsule"],
        pose_targets_sha256=hashes["pose-targets"],
        pose_initializer=initializer,
    )
    semantic_packet = b"exact-parsed-g105-semantic-packet"
    final_y1_binding = _sha("final-y1-binding")
    custody = SimpleNamespace(
        base=base,
        semantic_packet=semantic_packet,
        final_y1_binding_sha256=final_y1_binding,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="ascii")
    config = subject.PostG105PoseRefitConfigV1(
        config_path=config_path,
        config_sha256=_sha("config"),
        run_id="g119-schema-proof",
        seed=119,
        output_root=tmp_path,
        g112_partition_receipt={},
        target_capsule_receipt={},
        q_levels_candidates=(32,),
        local_gauss_newton_stages=1,
        finite_difference_q_steps=1,
        damping=1.0,
        trust_radius_q=1,
        line_search_scales=(1.0,),
        device="cpu",
        torch_num_threads=1,
    )
    xi_eff = np.zeros_like(xi_initializer)
    checkpoint = tmp_path / "post_g105_pose_refit_final.npz"
    subject._write_once(
        checkpoint,
        subject._deterministic_npz(
            subject._final_checkpoint_arrays(
                config=config,
                custody=custody,
                q_levels=32,
                xi_eff=xi_eff,
                selected_xip2_coder="delta_ar",
            )
        ),
    )
    final_binding = _binding(checkpoint)
    run_body = {
        "schema": subject.POST_G105_REFIT_RUN_SCHEMA,
        "run_id": config.run_id,
        "seed": config.seed,
        "source_git_sha": "a" * 40,
        "command": ["runner", "--resume-from", str(tmp_path)],
        "fresh_own_lineage": True,
        "source_contract": g110.SOURCE_DOMAIN,
        "render_order": g110.RENDER_ORDER,
        "y1_selected_preimage_schema": g110.V10_FACTOR2_SELECTED_PREIMAGE_SCHEMA,
        "source_g112_partition_receipt_sha256": hashes["partition"],
        "source_g112_semantic_child_sha256": hashes["semantic-child"],
        "source_g112_pose_initializer_sha256": hashes["pose-initializer"],
        "source_g111_deploy_checkpoint_sha256": hashes["deploy"],
        "source_g111_resume_checkpoint_sha256": hashes["resume"],
        "source_g111_lineage_receipt_sha256": hashes["lineage"],
        "source_g111_checkpoint_id_sha256": hashes["checkpoint-id"],
        "source_g111_root_sha256": hashes["root"],
        "semantic_packet_sha256": subject._sha256(semantic_packet),
        "final_y1_binding_sha256": final_y1_binding,
        "xi_initializer_sha256": g110._xi_digest(xi_initializer),
        "target_projection_sha256": hashes["projection"],
        "target_capsule_receipt_sha256": hashes["capsule"],
        "pose_targets_sha256": hashes["pose-targets"],
        "selected_xip2_coder": "delta_ar",
        "g110_selected_xip2_coder_abi_closed": True,
        "exact_public_receiver_in_loop": True,
        "resumable_from_disk": True,
        "stage_checkpoints_preserved": True,
        "stage_checkpoints": [final_binding],
        "final_checkpoint": final_binding,
        "research_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
    }
    run_receipt = subject._seal(run_body, field="receipt_sha256")
    run_path = tmp_path / "run.json"
    subject._write_json_once(run_path, run_receipt)

    reopened = g110._verify_post_g105_refit(
        checkpoint=checkpoint,
        expected_checkpoint_sha256=subject.sha256_file(checkpoint),
        run_receipt=run_path,
        expected_run_receipt_sha256=subject.sha256_file(run_path),
        base_custody=base,
        initializer=initializer,
        semantic_packet=semantic_packet,
        final_y1_binding=final_y1_binding,
    )
    assert reopened.q_levels == 32
    assert np.array_equal(reopened.xi_eff, xi_eff)
    assert reopened.checkpoint_sha256 == subject.sha256_file(checkpoint)
