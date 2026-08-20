from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.witness_control import fresh_producer_lineage_v1 as lineage
from tac.witness_control.g111_schedule_control_state_v1 import (
    new_state as new_g111_schedule_control_state,
)
from tac.witness_control.g111_schedule_control_state_v1 import (
    state_arrays as g111_schedule_control_state_arrays,
)
from tac.witness_control.g111_verdict_barrier_v1 import ImmutableVerdictResult
from tac.witness_control.trajectory_transaction_v2 import (
    ATOMIC_OWNERS,
    CANONICAL_DOMAIN_COVERAGE,
    MANIFEST_KEY,
    build_manifest,
    manifest_array,
)
from tac.witness_control import taskspace_g112_exact_checkpoint_partition_v1 as subject
from tac.witness_dsl import (
    taskspace_g105_exact_v9_semantic_root_adapter_v1 as g105,
)


def _fake_projection() -> dict[str, object]:
    return {
        "aggregate_receipt": {
            "path": "/physical/g109/21_receipt.json",
            "bytes": 123,
            "sha256": "4" * 64,
        },
        "aggregate_receipt_sha256": "5" * 64,
        "batch_digest_chain_sha256": "6" * 64,
        "pair_count": 600,
        "scorer_pair_batch_size": 16,
        "same_forward_seg_margin_pose": True,
        "encoder_only": True,
        "candidate_payload_allowed": False,
    }


def _fake_target_arrays() -> dict[str, np.ndarray]:
    return {
        subject.CHECKPOINT_PROJECTION_KEY: np.asarray("{}"),
        subject.CHECKPOINT_PROJECTION_SHA_KEY: np.asarray("1" * 64),
        "__cfg_target_authority_sha256": np.asarray("2" * 64),
        "__cfg_g46_target_evidence_sha256": np.asarray("3" * 64),
        "__cfg_verdict_batch": np.asarray(16),
    }


def _native_arrays_for_pair(
    *,
    deploy: dict[str, np.ndarray],
    resume: dict[str, np.ndarray],
    typed_config_sha256: str,
) -> dict[str, np.ndarray]:
    direct_prefixes = (
        lineage.RESUME_LIVE_PREFIX,
        lineage.RESUME_EMA_PREFIX,
        lineage.RESUME_OPT_PREFIX,
        lineage.RESUME_SEED_LIVE_PREFIX,
        lineage.RESUME_SEED_OPT_PREFIX,
        lineage.RESUME_POLYAK_PREFIX,
    )
    direct_resume = {
        key: value
        for key, value in resume.items()
        if key.startswith(direct_prefixes)
    }
    lineage_payload = {
        key: np.atleast_1d(value)
        for key, value in resume.items()
        if key.startswith("__cfg_fresh_")
    }
    control = {
        key: np.atleast_1d(value)
        for key, value in resume.items()
        if key not in direct_resume and key not in lineage_payload
    }
    epoch = int(np.asarray(resume["__resume_epoch"]).item())
    o3 = new_g111_schedule_control_state(
        typed_config_sha256=typed_config_sha256,
        completed_epoch=epoch,
        next_epoch=epoch + 1,
        accepted_optimizer_steps=0,
        stop_latched=False,
        control_scalars={},
        resume_control_arrays=control,
    )
    o3_arrays = g111_schedule_control_state_arrays(
        o3,
        prefix="__g111_o3__",
    )
    encoded_o6 = ImmutableVerdictResult.capture(
        submission_seq=0,
        result_id="g111-o6-lineage-envelope",
        payload={"lineage": lineage_payload},
    )
    o6_payload = np.zeros(64 * 1024, dtype=np.uint8)
    o6_payload[: len(encoded_o6.payload_bytes)] = np.frombuffer(
        encoded_o6.payload_bytes,
        dtype=np.uint8,
    )
    by_owner = {
        ATOMIC_OWNERS[0]: {
            **{f"deploy.{key}": value for key, value in deploy.items()},
            **{
                f"resume.{key}": value
                for key, value in direct_resume.items()
            },
        },
        ATOMIC_OWNERS[1]: {
            "controller.__test_o2_state": np.asarray([1], dtype=np.int64),
        },
        ATOMIC_OWNERS[2]: {
            f"controller.{key}": value
            for key, value in o3_arrays.items()
        },
        ATOMIC_OWNERS[3]: {
            "barrier.__test_o4_state": np.asarray([1], dtype=np.int64),
        },
        ATOMIC_OWNERS[4]: {
            "controller.__test_o5_state": np.asarray([1], dtype=np.int64),
        },
        ATOMIC_OWNERS[5]: {
            "lineage.__g111_o6__schema": np.frombuffer(
                b"tac.g111_lineage_envelope_arrays.v1",
                dtype=np.uint8,
            ).copy(),
            "lineage.__g111_o6__payload": o6_payload,
            "lineage.__g111_o6__payload_length": np.asarray(
                len(encoded_o6.payload_bytes),
                dtype=np.int64,
            ),
            "lineage.__g111_o6__sha256": np.frombuffer(
                encoded_o6.result_sha256.encode("ascii"),
                dtype=np.uint8,
            ).copy(),
        },
    }
    arrays = {
        key: np.asarray(value)
        for state in by_owner.values()
        for key, value in state.items()
    }
    manifest = build_manifest(
        arrays,
        owner_claims={
            owner: tuple(sorted(state))
            for owner, state in by_owner.items()
        },
        activity=dict.fromkeys(ATOMIC_OWNERS, True),
        domain_coverage=dict(CANONICAL_DOMAIN_COVERAGE),
    )
    return {
        **arrays,
        MANIFEST_KEY: manifest_array(manifest),
    }


@pytest.fixture
def fake_target_custody(monkeypatch: pytest.MonkeyPatch) -> None:
    projection = _fake_projection()
    expected = _fake_target_arrays()

    def reopen(**_kwargs):
        return projection

    def expected_arrays(*_args, **_kwargs):
        return {key: value.copy() for key, value in expected.items()}

    monkeypatch.setattr(
        subject,
        "reopen_v9_training_target_projection",
        reopen,
    )
    monkeypatch.setattr(
        subject,
        "checkpoint_target_arrays_from_projection",
        expected_arrays,
    )


def _source_arrays() -> dict[str, np.ndarray]:
    basis = g105.V9PolarFourierConfigV1(
        n_scales=1,
        n_orient0=2,
        f0=2.0,
        base=2.0,
        n_iso=1,
        max_freq=None,
    )
    hidden = 4
    layers = 1
    modulation = 3
    input_dim = basis.input_dim
    params = {
        "code": np.linspace(
            -0.5,
            0.5,
            1200 * modulation,
            dtype=np.float32,
        ).reshape(1200, modulation),
        "in_proj.weight": np.linspace(
            -0.2,
            0.2,
            hidden * input_dim,
            dtype=np.float32,
        ).reshape(hidden, input_dim),
        "in_proj.bias": np.zeros(hidden, dtype=np.float32),
        "film.weight": np.linspace(
            -0.1,
            0.1,
            2 * hidden * layers * modulation,
            dtype=np.float32,
        ).reshape(2 * hidden * layers, modulation),
        "film.bias": np.zeros(2 * hidden * layers, dtype=np.float32),
        "hidden.0.weight": np.eye(hidden, dtype=np.float32),
        "hidden.0.bias": np.zeros(hidden, dtype=np.float32),
        "out_sdf.weight": np.linspace(
            -0.3,
            0.3,
            5 * hidden,
            dtype=np.float32,
        ).reshape(5, hidden),
        "out_sdf.bias": np.zeros(5, dtype=np.float32),
        "out_tex.weight": np.linspace(
            -0.25,
            0.25,
            3 * hidden,
            dtype=np.float32,
        ).reshape(3, hidden),
        "out_tex.bias": np.zeros(3, dtype=np.float32),
        "palette": np.linspace(
            0.0,
            1.0,
            15,
            dtype=np.float32,
        ).reshape(5, 3),
        "pose_carrier.xi_stored": np.linspace(
            -0.01,
            0.01,
            600 * 6,
            dtype=np.float32,
        ).reshape(600, 6),
        "pose_carrier.dxi": np.linspace(
            0.002,
            -0.002,
            600 * 6,
            dtype=np.float32,
        ).reshape(600, 6),
    }
    seed = 112
    initial_sha = hashlib.sha256(b"g111-initial-placeholder").hexdigest()
    root_dsl = hashlib.sha256(b"g111-root-dsl").hexdigest()
    launch_dsl = hashlib.sha256(b"g111-launch-dsl").hexdigest()
    target_projection_sha = "1" * 64
    root_sha = lineage.fresh_producer_root_sha256(
        seed=seed,
        dsl_compile_hash=root_dsl,
        target_projection_sha256=target_projection_sha,
        initial_state_sha256=initial_sha,
    )
    configs = {
        "__cfg_fresh_producer": np.asarray(1, dtype=np.int8),
        "__cfg_fresh_lineage_schema": np.asarray(
            lineage.FRESH_PRODUCER_LINEAGE_SCHEMA
        ),
        "__cfg_fresh_seed": np.asarray(seed, dtype=np.int64),
        "__cfg_fresh_lineage_root_sha256": np.asarray(root_sha),
        "__cfg_fresh_initial_state_sha256": np.asarray(initial_sha),
        "__cfg_fresh_dsl_compile_hash": np.asarray(root_dsl),
        "__cfg_fresh_target_projection_sha256": np.asarray(
            target_projection_sha
        ),
        "__cfg_fresh_current_launch_dsl_compile_hash": np.asarray(
            launch_dsl
        ),
        "__epoch": np.asarray(0),
        "__cfg_activation": np.asarray("hosc"),
        "__cfg_upstream_snapshot_schema": np.asarray(g105.UPSTREAM_SOURCE_CLOSURE_SCHEMA),
        "__cfg_upstream_snapshot_sha256": np.asarray(g105.UPSTREAM_SOURCE_CLOSURE_SHA256),
        "__cfg_git_sha": np.asarray("a" * 40),
        "__render_hw": np.asarray([384, 512], dtype=np.int64),
        "__cfg_self_orient": np.asarray(0),
        "__cfg_render_aa": np.asarray("none"),
        "__bank_n_scales": np.asarray(basis.n_scales),
        "__bank_n_orient0": np.asarray(basis.n_orient0),
        "__bank_f0": np.asarray(basis.f0),
        "__bank_base": np.asarray(basis.base),
        "__bank_n_iso": np.asarray(basis.n_iso),
        "__cfg_max_bank_freq": np.asarray(-1.0),
        "__cfg_softmax_temp": np.asarray(0.5),
        "__cfg_hosc_beta": np.asarray(1.25),
        "__cfg_hosc_omega": np.asarray(2.0),
        "__cfg_chroma": np.asarray(1),
        "__cfg_pose_carrier_contract_schema": np.asarray(subject.POSE_CHECKPOINT_CONTRACT_SCHEMA),
        "__cfg_pose_carrier": np.asarray(1, dtype=np.int8),
        "__cfg_pose_carrier_source": np.asarray("generated_y1"),
        "__cfg_pose_carrier_residual_mode": np.asarray("table"),
        "__cfg_pose_carrier_residual_scale": np.asarray(0.75),
        "__cfg_pose_carrier_s_t": np.asarray(0.044),
        "__cfg_pose_carrier_s_r": np.asarray(0.01),
        "__cfg_pose_carrier_pitch": np.asarray(0.02),
        "__cfg_pose_carrier_native_hw": np.asarray(
            [874, 1164],
            dtype=np.int64,
        ),
        "__cfg_pose_carrier_xi_formula": np.asarray("xi_stored+residual_scale*dxi"),
        "__cfg_pose_carrier_y1_selected_preimage_schema": np.asarray(subject.Y1_SELECTED_PREIMAGE_SCHEMA),
    }
    provisional = {**params, **configs, **_fake_target_arrays()}
    initial_sha = lineage.fresh_resume_semantic_state_sha256_from_flat(
        _resume_arrays(provisional)
    )
    root_sha = lineage.fresh_producer_root_sha256(
        seed=seed,
        dsl_compile_hash=root_dsl,
        target_projection_sha256=target_projection_sha,
        initial_state_sha256=initial_sha,
    )
    configs.update(
        {
            "__cfg_fresh_lineage_root_sha256": np.asarray(root_sha),
            "__cfg_fresh_initial_state_sha256": np.asarray(initial_sha),
        }
    )
    return {**params, **configs, **_fake_target_arrays()}


def _write_checkpoint(path: Path, arrays: dict[str, np.ndarray]) -> str:
    payload = subject._deterministic_npz_bytes(arrays)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _refresh_root_lineage(arrays: dict[str, np.ndarray]) -> None:
    initial_sha = lineage.fresh_resume_semantic_state_sha256_from_flat(
        _resume_arrays(arrays)
    )
    root_sha = lineage.fresh_producer_root_sha256(
        seed=int(np.asarray(arrays["__cfg_fresh_seed"]).item()),
        dsl_compile_hash=str(
            np.asarray(arrays["__cfg_fresh_dsl_compile_hash"]).item()
        ),
        target_projection_sha256=str(
            np.asarray(
                arrays["__cfg_fresh_target_projection_sha256"]
            ).item()
        ),
        initial_state_sha256=initial_sha,
    )
    arrays["__cfg_fresh_initial_state_sha256"] = np.asarray(initial_sha)
    arrays["__cfg_fresh_lineage_root_sha256"] = np.asarray(root_sha)


def _resume_arrays(
    deploy: dict[str, np.ndarray],
    *,
    parent_checkpoint_id: str = lineage.ROOT_PARENT_CHECKPOINT_ID,
) -> dict[str, np.ndarray]:
    resume: dict[str, np.ndarray] = {}
    params = {
        key: np.asarray(value)
        for key, value in deploy.items()
        if not key.startswith("__")
    }
    for key, value in params.items():
        resume[lineage.RESUME_LIVE_PREFIX + key] = np.asarray(value).copy()
        resume[lineage.RESUME_EMA_PREFIX + key] = np.asarray(value).copy()
    resume[lineage.RESUME_OPT_PREFIX + "step"] = np.asarray(
        0,
        dtype=np.int64,
    )
    resume[lineage.RESUME_OPT_PREFIX + "m"] = np.zeros(
        8,
        dtype=np.float32,
    )
    for key, value in deploy.items():
        if (
            key in lineage.FRESH_DEPLOY_KEYS
            or key.startswith("__cfg_pose_carrier")
            or key.startswith("__cfg_g109_")
            or key.startswith("__cfg_g46_")
            or key
            in {
                "__cfg_target_authority_sha256",
                "__cfg_verdict_batch",
            }
        ):
            resume[key] = np.asarray(value).copy()
    stage = "stageColdRoot"
    event_ledger = json.dumps(
        {
            "schema": lineage.RESUME_EVENT_LEDGER_SCHEMA,
            "stage": stage,
            "persisted_keys": [],
            "active_event_flags": [],
            "inactive_explicit": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    resume.update(
        {
            "__cfg_film_stiefel": np.asarray(0),
            "__resume_epoch": np.asarray(0),
            "__resume_has_opt": np.asarray(1),
            "__resume_semantic_schema": np.asarray(
                lineage.RESUME_SEMANTIC_SCHEMA
            ),
            "__cfg_seed_islands": np.asarray(0),
            "__resume_primary_optimizer_family": np.asarray("adamw"),
            "__resume_has_seed": np.asarray(0),
            "__resume_active_trainable_components_json": np.asarray(
                '["primary_model"]'
            ),
            "__resume_stage": np.asarray(stage),
            "__resume_event_ledger_json": np.asarray(event_ledger),
            "__rng_np_algo": np.asarray("MT19937"),
            "__rng_np_keys": np.arange(624, dtype=np.uint32),
            "__rng_np_pos": np.asarray(23),
            "__rng_np_has_gauss": np.asarray(0),
            "__rng_np_cached_gauss": np.asarray(0.0),
            "__recent_losses": np.asarray(
                [1.0, 0.5],
                dtype=np.float64,
            ),
        }
    )
    state_sha = lineage.fresh_resume_semantic_state_sha256_from_flat(
        resume
    )
    checkpoint_id = lineage.fresh_checkpoint_id_sha256(
        root_sha256=str(
            resume["__cfg_fresh_lineage_root_sha256"].item()
        ),
        parent_checkpoint_id_sha256=parent_checkpoint_id,
        state_sha256=state_sha,
        epoch=0,
        stage=stage,
    )
    resume.update(
        {
            "__cfg_fresh_lineage_parent_checkpoint_id_sha256": np.asarray(
                parent_checkpoint_id
            ),
            "__cfg_fresh_lineage_state_sha256": np.asarray(state_sha),
            "__cfg_fresh_lineage_checkpoint_id_sha256": np.asarray(
                checkpoint_id
            ),
            "__cfg_fresh_lineage_epoch": np.asarray(
                0,
                dtype=np.int64,
            ),
            "__cfg_fresh_lineage_stage": np.asarray(stage),
        }
    )
    return resume


def _publish_source_node(
    tmp_path: Path,
    *,
    name: str,
    arrays: dict[str, np.ndarray],
) -> lineage.FreshProducerPhysicalCheckpointNodeV1:
    _refresh_root_lineage(arrays)
    source_dir = tmp_path / f"{name}-sources"
    source_dir.mkdir()
    run_dir = tmp_path / f"{name}-run"
    run_dir.mkdir()
    deploy_path = source_dir / "deploy.npz"
    resume_path = source_dir / "resume.npz"
    native_path = source_dir / "native.npz"
    deploy_sha = _write_checkpoint(deploy_path, arrays)
    resume_arrays = _resume_arrays(arrays)
    resume_sha = _write_checkpoint(
        resume_path,
        resume_arrays,
    )
    native_sha = _write_checkpoint(
        native_path,
        _native_arrays_for_pair(
            deploy=arrays,
            resume=resume_arrays,
            typed_config_sha256=str(
                arrays[
                    "__cfg_fresh_current_launch_dsl_compile_hash"
                ].item()
            ),
        ),
    )
    return lineage.write_fresh_physical_checkpoint_node_v1(
        out_dir=run_dir,
        deploy_checkpoint=deploy_path,
        expected_deploy_sha256=deploy_sha,
        resume_checkpoint=resume_path,
        expected_resume_sha256=resume_sha,
        expected_current_launch_dsl_compile_hash=str(
            arrays[
                "__cfg_fresh_current_launch_dsl_compile_hash"
            ].item()
        ),
        native_checkpoint=native_path,
        expected_native_sha256=native_sha,
    )


def _materialize(
    *,
    node: lineage.FreshProducerPhysicalCheckpointNodeV1,
    output_root: Path,
    allowed_output_roots: tuple[Path, ...],
) -> subject.G112CheckpointPartitionResultV1:
    return subject.materialize_g112_checkpoint_partition(
        checkpoint=node.pair.deploy.path,
        expected_checkpoint_sha256=node.pair.deploy.sha256,
        resume_checkpoint=node.pair.resume.path,
        expected_resume_checkpoint_sha256=node.pair.resume.sha256,
        lineage_receipt=node.receipt_path,
        expected_lineage_receipt_sha256=node.receipt_sha256,
        expected_current_launch_dsl_compile_hash=(
            node.pair.current_launch_dsl_compile_hash
        ),
        output_root=output_root,
        allowed_output_roots=allowed_output_roots,
    )


def test_physical_checkpoint_path_sha_and_symlink_fail_closed(
    tmp_path: Path,
) -> None:
    assert len(subject.POSE_CONFIG_KEYS) == 11
    checkpoint = tmp_path / "source.npz"
    sha = _write_checkpoint(checkpoint, _source_arrays())
    arrays, identity = subject._open_physical_checkpoint(
        checkpoint,
        expected_sha256=sha,
    )
    assert set(arrays) == set(_source_arrays())
    assert identity["sha256"] == sha
    assert identity["reopened_unchanged"] is True

    with pytest.raises(subject.G112CheckpointPartitionError, match="absolute"):
        subject._open_physical_checkpoint(
            Path("source.npz"),
            expected_sha256=sha,
        )
    with pytest.raises(subject.G112CheckpointPartitionError, match="SHA-256"):
        subject._open_physical_checkpoint(
            checkpoint,
            expected_sha256="0" * 64,
        )
    symlink = tmp_path / "source-link.npz"
    symlink.symlink_to(checkpoint)
    with pytest.raises(subject.G112CheckpointPartitionError, match="non-symlink"):
        subject._open_physical_checkpoint(
            symlink,
            expected_sha256=sha,
        )


def test_odd_only_partition_is_deterministic_and_even_invariant(
    tmp_path: Path,
    fake_target_custody: None,
) -> None:
    first_arrays = _source_arrays()
    second_arrays = {key: value.copy() for key, value in first_arrays.items()}
    second_arrays["code"][0::2] = np.float32(91.0)
    first_node = _publish_source_node(
        tmp_path,
        name="first",
        arrays=first_arrays,
    )
    second_node = _publish_source_node(
        tmp_path,
        name="second",
        arrays=second_arrays,
    )

    first = _materialize(
        node=first_node,
        output_root=tmp_path / "out-a",
        allowed_output_roots=(tmp_path,),
    )
    second = _materialize(
        node=second_node,
        output_root=tmp_path / "out-b",
        allowed_output_roots=(tmp_path,),
    )
    assert (
        first_node.pair.deploy.sha256
        != second_node.pair.deploy.sha256
    )
    assert first.semantic_packet_sha256 == second.semantic_packet_sha256
    first_child = subject.open_g112_semantic_child(
        first.semantic_child_path,
        expected_sha256=first.semantic_child_sha256,
    )
    second_child = subject.open_g112_semantic_child(
        second.semantic_child_path,
        expected_sha256=second.semantic_child_sha256,
    )
    assert np.array_equal(first_child.code_y1, second_child.code_y1)
    assert (
        first_child.semantic_packet_sha256
        == second_child.semantic_packet_sha256
        == first.semantic_packet_sha256
    )
    assert first.initializer_sha256 == second.initializer_sha256

    child = first_child
    initializer = subject.open_g112_pose_initializer(
        first.initializer_path,
        expected_sha256=first.initializer_sha256,
    )
    assert child.code_y1.shape == (600, 3)
    assert np.array_equal(child.code_y1, first_arrays["code"][1::2])
    assert "code" not in child.shared_params
    assert not any(key.startswith("pose_carrier.") for key in child.shared_params)
    assert child.semantic_packet_sha256 == first.semantic_packet_sha256
    expected_xi = first_arrays["pose_carrier.xi_stored"].astype(np.float64) + 0.75 * first_arrays[
        "pose_carrier.dxi"
    ].astype(np.float64)
    assert np.array_equal(initializer.xi_init, expected_xi)
    assert initializer.requires_post_g105_refit is True
    assert initializer.candidate_payload_eligible is False
    assert initializer.selected_preimage_schema == subject.Y1_SELECTED_PREIMAGE_SCHEMA

    with np.load(first.semantic_child_path, allow_pickle=False) as archive:
        assert subject.SEMANTIC_ODD_CODE_KEY in archive.files
        assert "code" not in archive.files
        assert not any(key.startswith("pose_carrier.") for key in archive.files)
        assert str(archive["__g112_semantic_child_schema"]) == subject.SEMANTIC_CHILD_SCHEMA
    receipt = json.loads(first.receipt_path.read_text(encoding="ascii"))
    partition = receipt["partition"]
    assert partition["source_tensor_union_complete"] is True
    assert partition["source_tensor_owners_disjoint"] is True
    assert partition["source_atoms_orphaned"] == 0
    assert partition["source_atom_ownership"]["even_code_rows"]["semantic_child_storage"] == "absent"
    assert receipt["conditional_initializer"]["final_payload"] is False
    assert receipt["conditional_initializer"]["requires_real_post_g105_refit"] is True
    assert receipt["fresh_producer_lineage"][
        "complete_trajectory_proven"
    ] is True
    reopened_receipt = subject.open_g112_partition_receipt(
        first.receipt_path,
        expected_sha256=first.receipt_sha256,
    )
    assert reopened_receipt.source_chain.complete_trajectory_proven is True
    assert len(reopened_receipt.source_chain.nodes) == 1


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda arrays: arrays.pop("pose_carrier.dxi"),
            "pose learned tensor set",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "pose_carrier.extra",
                np.zeros((600, 6), dtype=np.float32),
            ),
            "pose learned tensor set",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "__cfg_pose_carrier_source",
                np.asarray("generated"),
            ),
            "differs at __cfg_pose_carrier_source",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "__cfg_pose_carrier_residual_mode",
                np.asarray("film"),
            ),
            "differs at __cfg_pose_carrier_residual_mode",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "__cfg_pose_carrier_extra",
                np.asarray(1),
            ),
            "pose config set is not exact",
        ),
        (
            lambda arrays: arrays.pop("__cfg_pose_carrier_s_t"),
            "pose config set is not exact",
        ),
        (
            lambda arrays: arrays.pop("__cfg_pose_carrier_y1_selected_preimage_schema"),
            "pose config set is not exact",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "__cfg_pose_carrier_y1_selected_preimage_schema",
                np.asarray("tac.v10_factor2_selected_preimage.wrong"),
            ),
            "differs at __cfg_pose_carrier_y1_selected_preimage_schema",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "out_sdf.weight",
                np.full((5, 4), np.nan, dtype=np.float32),
            ),
            "non-finite",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "orphan.weight",
                np.zeros((1,), dtype=np.float32),
            ),
            "not exact G105",
        ),
    ],
)
def test_extra_missing_wrong_and_nonfinite_fail_closed(
    tmp_path: Path,
    fake_target_custody: None,
    mutation,
    match: str,
) -> None:
    arrays = _source_arrays()
    mutation(arrays)
    with pytest.raises(
        (
            subject.G112CheckpointPartitionError,
            lineage.FreshProducerLineageV1Error,
        ),
        match=match,
    ):
        node = _publish_source_node(
            tmp_path,
            name="bad",
            arrays=arrays,
        )
        _materialize(
            node=node,
            output_root=tmp_path / "out",
            allowed_output_roots=(tmp_path,),
        )


def test_extra_target_config_fails_closed(
    tmp_path: Path,
    fake_target_custody: None,
) -> None:
    arrays = _source_arrays()
    arrays["__cfg_g46_orphan"] = np.asarray("forbidden")
    with pytest.raises(
        subject.G112CheckpointPartitionError,
        match="target config set is not exact",
    ):
        node = _publish_source_node(
            tmp_path,
            name="bad-target",
            arrays=arrays,
        )
        _materialize(
            node=node,
            output_root=tmp_path / "out",
            allowed_output_roots=(tmp_path,),
        )


def test_receipt_rejects_cross_partition_initializer_mix(
    tmp_path: Path,
    fake_target_custody: None,
) -> None:
    first_arrays = _source_arrays()
    second_arrays = {
        key: value.copy()
        for key, value in first_arrays.items()
    }
    second_arrays["pose_carrier.dxi"] += np.float32(0.125)
    first_node = _publish_source_node(
        tmp_path,
        name="mix-first",
        arrays=first_arrays,
    )
    second_node = _publish_source_node(
        tmp_path,
        name="mix-second",
        arrays=second_arrays,
    )
    first = _materialize(
        node=first_node,
        output_root=tmp_path / "mix-out-first",
        allowed_output_roots=(tmp_path,),
    )
    second = _materialize(
        node=second_node,
        output_root=tmp_path / "mix-out-second",
        allowed_output_roots=(tmp_path,),
    )
    first.initializer_path.write_bytes(
        second.initializer_path.read_bytes()
    )
    with pytest.raises(
        subject.G112CheckpointPartitionError,
        match="physical SHA-256 differs",
    ):
        subject.open_g112_partition_receipt(
            first.receipt_path,
            expected_sha256=first.receipt_sha256,
        )


def test_real_n600_g109_projection_reopens_when_available() -> None:
    receipt_path = Path(
        "/Volumes/VertigoDataTier/pact/"
        "taskspace_v9_training_target_capsule_n600_20260727/"
        "21_v9_training_target_capsule_receipt.json"
    )
    if not receipt_path.is_file():
        pytest.skip("durable real n600 G109 capsule is not mounted")
    from tac.witness_control.taskspace_v9_training_target_binding_v1 import (
        _projection,
    )
    from tac.witness_control.taskspace_v9_training_target_capsule_v1 import (
        V9TrainingTargetCapsuleLoaderV1,
        sha256_file,
    )

    loader = V9TrainingTargetCapsuleLoaderV1.open(
        receipt_path,
        expected_sha256=sha256_file(receipt_path),
    )
    projection = _projection(loader)
    authority_sha = "9" * 64
    arrays = subject.checkpoint_target_arrays_from_projection(
        projection,
        active_target_authority_sha256=authority_sha,
        verdict_batch=16,
    )
    reopened = subject._validate_target_custody(arrays)
    assert reopened["pair_count"] == 600
    assert reopened["scorer_pair_batch_size"] == 16
    assert reopened["same_forward_seg_margin_pose"] is True
