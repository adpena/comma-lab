from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.ddm_dm4_j5_adapter import DM4J5ProposalV1
from tac.optimization.ddm_dm4_j5_counted_application import (
    CONFIG_SCHEMA,
    EVIDENCE_AXIS,
    HORIZON_GAP,
    POINTER,
    RANGE_GAUGE_POLICY,
    VALIDITY_GAP,
    BoundArtifactV1,
    DDMCountedApplicationConfigV1,
    DDMCountedApplicationError,
    MS4DPairMetricV1,
    SparseJ5CoordinateEffectV1,
    apply_coordinate_choices,
    descriptor_camera_delta,
    exact_joint_delta,
    load_ms4d_pair_metric,
    select_counted_application,
)
from tools.run_ddm_j8f_counted_application import (
    CHECKPOINT_SCHEMA,
    RUN_RECEIPT_SCHEMA,
    _compiled_state_binding,
    _load_completed_receipt,
    _load_inventory_checkpoint,
    _preflight_path,
    _resume_receipts,
    _write_inventory_checkpoint,
)


def _proposal(*, candidate_id: str = "scorer_recursive_erf0.5_target") -> DM4J5ProposalV1:
    indices = np.asarray([0], dtype="<u4")
    return DM4J5ProposalV1(
        proposal_id=f"dm4.row05.pair090.{candidate_id}",
        aimed_cell={
            "row_index": 5,
            "pair_id": 90,
            "bucket_id": "undrivable_movable__boundary__transient",
        },
        corrected_j_row={
            "metric": "rank4 target-vs-runner SegNet head margin on categorical Fisher base",
            "projected_input_adjoint": "exact sum over the canonical disjoint factor2 preimage taps",
        },
        support_footprint={
            "schema": "ddm_dm4_scorer_recursive_write_support.v1",
            "stem_stride": 2,
            "stem_block_indices": [0],
            "stem_block_indices_sha256_uint32le": hashlib.sha256(indices.tobytes()).hexdigest(),
            "support_rule": "scorer-recursive; no disks, global writes, or history",
        },
        proposal_type="joint",
        candidate={
            "candidate_id": candidate_id,
            "mechanism": "scorer_recursive_target",
        },
        source_receipt_sha256="a" * 64,
    )


def _metric() -> MS4DPairMetricV1:
    return MS4DPairMetricV1(
        pair_id=90,
        bucket_id="undrivable_movable__boundary__transient",
        hessian=np.diag([2.0, 0.0, 0.0, 0.0]),
        adjoint=np.asarray([-2.0, 0.0, 0.0, 0.0]),
        rank4_pair_normal=np.asarray([1.0, 0.0, 0.0, 0.0]),
        source_sha256="b" * 64,
        support_count=2,
    )


def _effect(
    *,
    coordinate_index: int,
    flat_index: int,
    value: int = 1,
    direction: int = 1,
) -> SparseJ5CoordinateEffectV1:
    return SparseJ5CoordinateEffectV1(
        coordinate_index=coordinate_index,
        coordinate_name=f"island.track{coordinate_index}.center_x",
        direction=direction,
        pair_id=90,
        flat_indices=np.asarray([flat_index], dtype=np.int64),
        values=np.asarray([value], dtype=np.int16),
        camera_shape=(2, 874, 1164, 3),
        archive_bytes=100 + coordinate_index,
        archive_sha256=f"{coordinate_index + 1:064x}",
        archive_byte_delta=coordinate_index,
        changed_channel_values=1,
    )


def test_ms4d_newton_step_uses_minimum_norm_pseudoinverse() -> None:
    metric = _metric()
    assert np.array_equal(metric.newton_step(), np.asarray([1.0, 0.0, 0.0, 0.0]))
    payload = metric.to_payload()
    assert payload["global_learning_rate_used"] is False
    assert payload["step_rule"] == "minimum_norm_-pinv(H)g_machine_epsilon_rank_cutoff"


def test_load_ms4d_pair_metric_requires_unique_pair_bucket(tmp_path: Path) -> None:
    row = {
        "pair_id": 90,
        "bucket_id": "undrivable_movable__boundary__transient",
        "metric_mode": "DIRECT_SCORER_INTRINSIC_NO_ACTUATOR_INPUT",
        "secant_status": "NOT_APPLICABLE_DIRECT_SCORER_INTRINSIC_NO_ACTUATOR",
        "support_status": "MEASURED_EXACT_PF2_EVENT_INDEX",
        "support_count": 2,
        "composite_r_model_hessian": np.diag([2.0, 0.0, 0.0, 0.0]).tolist(),
        "composite_r_adjoint_readback": [-2.0, 0.0, 0.0, 0.0],
        "rank4_pair_normal": [1.0, 0.0, 0.0, 0.0],
    }
    path = tmp_path / "metric.json"
    raw = json.dumps(
        {
            "schema": "ddm_seg_metric_custody.direct_scorer_intrinsic.v2",
            "direct_blocks": [row],
        }
    ).encode()
    path.write_bytes(raw)  # PAYLOAD_WRITE_ORDER_OK:no run product exists to strand. Both writes target the SAME `tmp_path` fixture file: this one lays down the unique-bucket case, and the later `write_text` deliberately overwrites it with a duplicated row to prove the loader refuses it. `row` is fixture input, not an irreplaceable record, and swapping the two would destroy the test.
    loaded = load_ms4d_pair_metric(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        pair_id=90,
        bucket_id="undrivable_movable__boundary__transient",
    )
    assert loaded.support_count == 2

    path.write_text(
        json.dumps(
            {
                "schema": "ddm_seg_metric_custody.direct_scorer_intrinsic.v2",
                "direct_blocks": [row, row],
            }
        )
    )
    with pytest.raises(DDMCountedApplicationError, match="join differs"):
        load_ms4d_pair_metric(
            path=path,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            pair_id=90,
            bucket_id="undrivable_movable__boundary__transient",
        )


def test_selection_uses_newton_model_then_range_gauge_integer_reprojection() -> None:
    target = np.zeros((2, 874, 1164, 3), dtype=np.float64)
    target.reshape(-1)[0] = 1.0
    target.reshape(-1)[1] = 1.0
    effects = (
        _effect(coordinate_index=0, flat_index=0),
        _effect(coordinate_index=1, flat_index=1),
        _effect(coordinate_index=2, flat_index=2),
    )

    def project_first_to_second(
        value: np.ndarray, *, out_dtype: object, compute_dtype: object
    ) -> np.ndarray:
        del out_dtype, compute_dtype
        output = np.zeros_like(value, dtype=np.float64)
        output.reshape(-1)[1] = value.reshape(-1)[0]
        return output

    receipt = select_counted_application(
        proposal=_proposal(),
        descriptor_delta=target,
        effects=effects,
        metric=_metric(),
        used_raw_coordinates=set(),
        used_projected_coordinates=set(),
        projector=project_first_to_second,
    )
    assert receipt["raw_application"]["coordinate_index"] == 0
    assert receipt["projected_application"]["coordinate_index"] == 1
    projection = receipt["range_gauge_projection"]
    assert projection["rejected_null_gauge_energy"] == pytest.approx(2.0)
    assert projection["parameter_gauge_representative"].startswith("minimum_integer")
    trust = receipt["trust_region"]
    assert trust["validity_gap"] == VALIDITY_GAP
    assert trust["coordinate_quantum"] == 1
    assert trust["shrink_factor"] is None
    assert trust["grow_factor"] is None
    assert trust["global_learning_rate"] is None


def test_selection_refuses_coordinate_reuse() -> None:
    target = np.zeros((2, 874, 1164, 3), dtype=np.float64)
    target.reshape(-1)[0] = 1.0
    with pytest.raises(DDMCountedApplicationError, match="no unused raw"):
        select_counted_application(
            proposal=_proposal(),
            descriptor_delta=target,
            effects=(_effect(coordinate_index=0, flat_index=0),),
            metric=_metric(),
            used_raw_coordinates={0},
            used_projected_coordinates=set(),
            projector=lambda value, **_: value,
        )


def test_apply_coordinate_choices_is_one_quantum_and_no_reuse() -> None:
    receipts = [
        {
            "raw_application": {"coordinate_index": 1, "direction": 1},
            "projected_application": {"coordinate_index": 2, "direction": -1},
        },
        {
            "raw_application": {"coordinate_index": 3, "direction": -1},
            "projected_application": {"coordinate_index": 4, "direction": 1},
        },
    ]
    theta = np.zeros(6, dtype=np.float32)
    assert np.array_equal(
        apply_coordinate_choices(theta, receipts, arm="raw_application"),
        np.asarray([0, 1, 0, -1, 0, 0], dtype=np.float32),
    )
    assert np.array_equal(
        apply_coordinate_choices(theta, receipts, arm="projected_application"),
        np.asarray([0, 0, -1, 0, 1, 0], dtype=np.float32),
    )
    receipts[1]["raw_application"]["coordinate_index"] = 1
    with pytest.raises(DDMCountedApplicationError, match="one-quantum/no-reuse"):
        apply_coordinate_choices(theta, receipts, arm="raw_application")


def test_pair_inventory_checkpoint_roundtrips_sparse_exact_effects(
    tmp_path: Path,
) -> None:
    bindings = {
        "operator_source": BoundArtifactV1(
            path="operator.py", sha256="a" * 64, bytes=1
        ),
        "step4_checkpoint": BoundArtifactV1(
            path="step4.npz", sha256="b" * 64, bytes=1
        ),
    }
    config = DDMCountedApplicationConfigV1(
        config_path=str(tmp_path / ".omx" / "research" / "configs" / "j8f.json"),
        lane_id="lane",
        run_id="run",
        output_root=str(tmp_path / "ssd"),
        torch_threads=4,
        smoke_horizon=12,
        source_bindings=bindings,
        execution_allowed=False,
    )
    effects = (
        _effect(coordinate_index=1, flat_index=3, value=-2, direction=-1),
        _effect(coordinate_index=2, flat_index=7, value=3, direction=1),
    )
    archive = b"step4-archive"
    theta = np.arange(4, dtype=np.float32)
    inventory = {
        "schema": "ddm_j5_pair_coordinate_effect_inventory.v1",
        "pair_id": 90,
        "inventory_manifest_sha256": "c" * 64,
    }
    stored_effects, stored_inventory, binding = _write_inventory_checkpoint(
        config=config,
        pair_id=90,
        base_archive=archive,
        theta=theta,
        parameter_names=("a", "b", "c", "d"),
        effects=effects,
        inventory_receipt=inventory,
    )
    assert [effect.to_payload() for effect in stored_effects] == [
        effect.to_payload() for effect in effects
    ]
    assert stored_inventory == inventory
    assert Path(binding["path"]).is_file()
    loaded = _load_inventory_checkpoint(
        config=config,
        pair_id=90,
        base_archive=archive,
        theta=theta,
        parameter_names=("a", "b", "c", "d"),
    )
    assert loaded is not None
    loaded_effects, loaded_inventory, loaded_binding = loaded
    assert [effect.to_payload() for effect in loaded_effects] == [
        effect.to_payload() for effect in effects
    ]
    assert loaded_inventory == inventory
    assert loaded_binding == binding
    assert (
        _load_inventory_checkpoint(
            config=config,
            pair_id=90,
            base_archive=archive,
            theta=theta + 1,
            parameter_names=("a", "b", "c", "d"),
        )
        is None
    )


def test_compiled_stage_binding_refuses_clipping_and_requires_parseback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    theta = np.asarray([1.0, 2.0], dtype=np.float32)

    class _Lift:
        def __init__(self, archive: bytes) -> None:
            self.archive = archive

        def exact_reemit(self) -> bytes:
            return self.archive

    monkeypatch.setattr(
        "tools.run_ddm_j8f_counted_application.lift_v15_archive",
        lambda archive: _Lift(archive),
    )
    binding = _compiled_state_binding(
        theta=theta,
        realized=theta.copy(),
        archive=b"exact",
    )
    assert binding["parseback_exact"] is True
    with pytest.raises(DDMCountedApplicationError, match="clipped or changed"):
        _compiled_state_binding(
            theta=theta,
            realized=np.asarray([1.0, 1.0], dtype=np.float32),
            archive=b"exact",
        )

    monkeypatch.setattr(
        "tools.run_ddm_j8f_counted_application.lift_v15_archive",
        lambda _archive: _Lift(b"different"),
    )
    with pytest.raises(DDMCountedApplicationError, match="parse-back"):
        _compiled_state_binding(
            theta=theta,
            realized=theta.copy(),
            archive=b"exact",
        )


def test_resume_receipts_refuse_divergent_cumulative_chain(tmp_path: Path) -> None:
    config = DDMCountedApplicationConfigV1(
        config_path=str(tmp_path / ".omx" / "research" / "configs" / "j8f.json"),
        lane_id="lane",
        run_id="run",
        output_root=str(tmp_path),
        torch_threads=4,
        smoke_horizon=12,
        source_bindings={},
        execution_allowed=True,
    )
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    common = {
        "schema": CHECKPOINT_SCHEMA,
        "typed_config_hash": config.typed_hash(),
        "status": "PRESERVED",
    }
    (checkpoint_root / "application_step_00.json").write_text(
        json.dumps(
            {
                **common,
                "step_index": 0,
                "application_receipts": [{"step": 0}],
            }
        )
    )
    (checkpoint_root / "application_step_01.json").write_text(
        json.dumps(
            {
                **common,
                "step_index": 1,
                "application_receipts": [{"step": 999}, {"step": 1}],
            }
        )
    )
    with pytest.raises(DDMCountedApplicationError, match="identity differs"):
        _resume_receipts(output_root=tmp_path, config=config)


def test_completed_receipt_revalidates_and_resumes_immutably(
    tmp_path: Path,
) -> None:
    config = DDMCountedApplicationConfigV1(
        config_path=str(tmp_path / ".omx" / "research" / "configs" / "j8f.json"),
        lane_id="lane",
        run_id="run",
        output_root=str(tmp_path),
        torch_threads=4,
        smoke_horizon=12,
        source_bindings={},
        execution_allowed=True,
    )
    preflight_path = _preflight_path(config)
    preflight_path.parent.mkdir(parents=True)
    preflight_path.write_text(json.dumps({"admission": True}))
    reference = {"d_seg": 0.1, "d_pose": 4.0, "archive_bytes": 100}

    def arm(
        name: str, d_seg: float, *, projected: bool = False
    ) -> dict[str, object]:
        payload = name.encode() * (100 // len(name))
        payload = payload.ljust(100, b"_")
        path = tmp_path / f"{name}.zip"
        path.write_bytes(payload)
        verdict = {
            "d_seg": d_seg,
            "d_pose": 4.0,
            "archive_bytes": len(payload),
            "archive_sha256": hashlib.sha256(payload).hexdigest(),
            "num_pairs": 600,
        }
        value: dict[str, object] = {
            "archive": {
                "path": str(path),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "parseback_exact": True,
            },
            "verdict": verdict,
            "delta_vs_step4": exact_joint_delta(
                reference=reference,
                candidate=verdict,
            ),
        }
        if projected:
            value[
                "realized_joint_delta_unchanged_or_better_than_raw"
            ] = True
        return value

    receipt = {
        "schema": RUN_RECEIPT_SCHEMA,
        "run_id": config.run_id,
        "lane_id": config.lane_id,
        "typed_config_hash": config.typed_hash(),
        "preflight": {
            "path": str(preflight_path),
            "sha256": hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
            "admission": True,
        },
        "step4": {"reference": reference},
        "raw_arm": arm("raw", 0.09),
        "range_gauge_projected_arm": arm(
            "projected", 0.08, projected=True
        ),
        "verdict": "READY_TO_FIRE_DDM_EVENT_CONTINUATION",
        "execution_allowed": False,
        "main_landing_review_required": True,
        "pointer": POINTER,
        "pointer_moved": False,
        "score_claim": False,
    }
    final_path = tmp_path / "ddm_j8f_counted_application_receipt.json"
    final_path.write_text(json.dumps(receipt))
    final_binding = {
        "path": str(final_path),
        "bytes": final_path.stat().st_size,
        "sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
    }
    (tmp_path / "READY_TO_FIRE.ticket.json").write_text(
        json.dumps(
            {
                "status": receipt["verdict"],
                "receipt": final_binding,
                "execution_allowed": False,
                "main_landing_review_required": True,
                "pointer_moved": False,
            }
        )
    )
    loaded = _load_completed_receipt(
        config=config,
        preflight={"admission": True},
    )
    assert loaded == receipt


@pytest.mark.parametrize(
    ("candidate_id", "expected_quantum"),
    [
        ("scorer_recursive_erf0.5_target", None),
        ("scorer_recursive_erf0.5_target_q32", 32),
    ],
)
def test_descriptor_camera_delta_preserves_stored_support_and_quantum(
    monkeypatch: pytest.MonkeyPatch,
    candidate_id: str,
    expected_quantum: int | None,
) -> None:
    def fake_realize(plane: np.ndarray, _kernel: object) -> np.ndarray:
        output = np.zeros((874, 1164, 3), dtype=np.uint8)
        output[0, 0] = plane[0, 0]
        return output

    monkeypatch.setattr(
        "tac.optimization.ddm_dm4_j5_counted_application.realize_solve_camera",
        fake_realize,
    )
    predictor = np.zeros((2, 384, 512, 3), dtype=np.uint8)
    target = predictor.copy()
    target[1, 0, 0] = 100
    delta, receipt = descriptor_camera_delta(
        proposal=_proposal(candidate_id=candidate_id),
        predictor_planes=predictor,
        target_planes=target,
        kernel=object(),  # fake realization does not inspect the kernel
    )
    expected = 100 if expected_quantum is None else expected_quantum
    assert delta[1, 0, 0].tolist() == [expected, expected, expected]
    assert receipt["quantum"] == expected_quantum
    assert receipt["changed_channel_values"] == 3


def test_exact_joint_delta_uses_all_three_contest_terms() -> None:
    delta = exact_joint_delta(
        reference={"d_seg": 0.1, "d_pose": 4.0, "archive_bytes": 100},
        candidate={"d_seg": 0.09, "d_pose": 1.0, "archive_bytes": 110},
    )
    assert delta["seg_term"] == pytest.approx(-1.0)
    assert delta["pose_term"] == pytest.approx(np.sqrt(10.0) - np.sqrt(40.0))
    assert delta["rate_term"] == pytest.approx(250.0 / 37_545_489.0)
    assert delta["joint_delta"] < 0.0


def test_counted_application_config_binds_every_authority_surface(tmp_path: Path) -> None:
    artifact = tmp_path / "bound.bin"
    artifact.write_bytes(b"authority")
    binding = {
        "path": str(artifact),
        "bytes": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    names = {
        "authority",
        "j8e_ticket",
        "dm4_config",
        "dm4_receipt",
        "ms4d_direct_metric",
        "step4_ticket",
        "step4_checkpoint",
        "step4_verdict",
        "v17_validity_law",
        "v17_validity_receipt",
        "ncde_observer",
        "ncde_event_wiring",
        "range_a_projector",
        "operator_source",
        "runner_source",
    }
    config_path = tmp_path / "config.json"
    payload = {
        "schema": CONFIG_SCHEMA,
        "lane_id": "lane_ddm_j8f_counted_application_20260724",
        "run_id": "ddm_j8f_test",
        "output_root": "/Volumes/VertigoDataTier/pact/test",
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "pointer": f"{POINTER} UNMOVED",
        "evidence_axis": EVIDENCE_AXIS,
        "main_landing_review_required": True,
        "torch_threads": 4,
        "smoke_horizon": 12,
        "horizon_derivation": HORIZON_GAP,
        "validity_policy": VALIDITY_GAP,
        "range_gauge_policy": RANGE_GAUGE_POLICY,
        "source_bindings": dict.fromkeys(names, binding),
    }
    config_path.write_text(json.dumps(payload))
    config = DDMCountedApplicationConfigV1.from_path(config_path)
    assert config.smoke_horizon == 12
    assert set(config.validate_all_bindings()) == names
    assert len(config.typed_hash()) == 64

    payload["source_bindings"].pop("range_a_projector")
    config_path.write_text(json.dumps(payload))
    with pytest.raises(DDMCountedApplicationError, match="binding set differs"):
        DDMCountedApplicationConfigV1.from_path(config_path)
