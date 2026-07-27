from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.witness_control import fresh_producer_lineage_v1 as subject


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    np.savez(path, **arrays)
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _source_pair_arrays(
    *,
    parent_id: str = subject.ROOT_PARENT_CHECKPOINT_ID,
    seed: int = 112,
    epoch: int = 0,
    stage: str = "stageColdRoot",
    semantic_schema: str = subject.LEGACY_RESUME_SEMANTIC_SCHEMA,
    include_complete_manifest: bool = False,
    seed_islands: bool = False,
    include_seed_config: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]:
    root_dsl = _sha("root-dsl")
    launch_dsl = _sha("launch-dsl")
    target_sha = _sha("g109")
    params = {
        "code": np.arange(24, dtype=np.float32).reshape(8, 3),
        "layer.weight": np.arange(12, dtype=np.float32).reshape(4, 3),
        "layer.bias": np.zeros(4, dtype=np.float32),
        "pose_carrier.xi_stored": np.zeros((4, 6), dtype=np.float32),
        "pose_carrier.dxi": np.ones((4, 6), dtype=np.float32),
    }
    pose = {
        "__cfg_pose_carrier": np.asarray(1, dtype=np.int8),
        "__cfg_pose_carrier_source": np.asarray("generated_y1"),
    }
    g109 = {
        subject.G109_TARGET_PROJECTION_SHA_KEY: np.asarray(target_sha),
        "__cfg_g109_target_projection_json": np.asarray("{}"),
        "__cfg_g46_target_evidence_sha256": np.asarray(_sha("g46")),
        "__cfg_target_authority_sha256": np.asarray(_sha("authority")),
        "__cfg_verdict_batch": np.asarray(16),
    }

    def _semantic_arrays(at_epoch: int, at_stage: str) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for key, value in params.items():
            arrays[subject.RESUME_LIVE_PREFIX + key] = np.asarray(value).copy()
            arrays[subject.RESUME_EMA_PREFIX + key] = np.asarray(value).copy()
        arrays[subject.RESUME_OPT_PREFIX + "step"] = np.asarray(
            17, dtype=np.int64
        )
        arrays[subject.RESUME_OPT_PREFIX + "m"] = np.zeros(5, dtype=np.float32)
        event_ledger = json.dumps(
            {
                "schema": subject.RESUME_EVENT_LEDGER_SCHEMA,
                "stage": at_stage,
                "persisted_keys": [],
                "active_event_flags": [],
                "inactive_explicit": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        arrays.update({
            **pose,
            **g109,
            "__cfg_film_stiefel": np.asarray(0),
            "__resume_epoch": np.asarray(at_epoch),
            "__resume_has_opt": np.asarray(1),
            "__resume_semantic_schema": np.asarray(
                semantic_schema
            ),
            "__resume_stage": np.asarray(at_stage),
            "__resume_event_ledger_json": np.asarray(event_ledger),
            "__rng_np_algo": np.asarray("MT19937"),
            "__rng_np_keys": np.arange(624, dtype=np.uint32),
            "__rng_np_pos": np.asarray(23),
            "__rng_np_has_gauss": np.asarray(0),
            "__rng_np_cached_gauss": np.asarray(0.0),
            "__recent_losses": np.asarray([1.0, 0.5], dtype=np.float64),
        })
        if include_seed_config:
            arrays["__cfg_seed_islands"] = np.asarray(int(seed_islands))
        if semantic_schema == subject.RESUME_SEMANTIC_SCHEMA:
            arrays.update(
                {
                    "__resume_primary_optimizer_family": np.asarray("adamw"),
                    "__resume_has_seed": np.asarray(0),
                    "__resume_active_trainable_components_json": np.asarray(
                        '["primary_model"]'
                    ),
                }
            )
            if include_complete_manifest:
                active_keys = {
                    "primary_model_ema_optimizer_family": ["liveP__code"],
                    "fresh_root_physical_lineage": [
                        "__cfg_fresh_producer"
                    ],
                    "rng_streams": ["__rng_np_keys"],
                    "best_and_stage_checkpoint_bookkeeping": [
                        "__resume_epoch"
                    ],
                }
                components = [
                    {
                        "name": name,
                        "active": name in active_keys,
                        "keys": active_keys.get(name, []),
                    }
                    for name in subject.G111_TRAJECTORY_COMPONENTS
                ]
                arrays[subject.G111_COMPLETE_TRAJECTORY_KEY] = np.asarray(
                    json.dumps(
                        {
                            "schema": (
                                subject.G111_COMPLETE_TRAJECTORY_SCHEMA
                            ),
                            "components": components,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
        return arrays

    initial_sha = subject.fresh_resume_semantic_state_sha256_from_flat(
        _semantic_arrays(0, "stageColdRoot")
    )
    root_sha = subject.fresh_producer_root_sha256(
        seed=seed,
        dsl_compile_hash=root_dsl,
        target_projection_sha256=target_sha,
        initial_state_sha256=initial_sha,
    )
    root = {
        "__cfg_fresh_producer": np.asarray(1, dtype=np.int8),
        "__cfg_fresh_lineage_schema": np.asarray(
            subject.FRESH_PRODUCER_LINEAGE_SCHEMA
        ),
        "__cfg_fresh_seed": np.asarray(seed, dtype=np.int64),
        "__cfg_fresh_lineage_root_sha256": np.asarray(root_sha),
        "__cfg_fresh_initial_state_sha256": np.asarray(initial_sha),
        "__cfg_fresh_dsl_compile_hash": np.asarray(root_dsl),
        "__cfg_fresh_target_projection_sha256": np.asarray(target_sha),
        "__cfg_fresh_current_launch_dsl_compile_hash": np.asarray(launch_dsl),
    }
    deploy = {
        **params,
        **root,
        **pose,
        **g109,
        "__epoch": np.asarray(epoch),
    }
    resume = _semantic_arrays(epoch, stage)
    resume.update(root)
    state_sha = subject.fresh_resume_semantic_state_sha256_from_flat(resume)
    checkpoint_id = subject.fresh_checkpoint_id_sha256(
        root_sha256=root_sha,
        parent_checkpoint_id_sha256=parent_id,
        state_sha256=state_sha,
        epoch=epoch,
        stage=stage,
    )
    resume.update(
        {
            "__cfg_fresh_lineage_parent_checkpoint_id_sha256": np.asarray(
                parent_id
            ),
            "__cfg_fresh_lineage_state_sha256": np.asarray(state_sha),
            "__cfg_fresh_lineage_checkpoint_id_sha256": np.asarray(
                checkpoint_id
            ),
            "__cfg_fresh_lineage_epoch": np.asarray(epoch, dtype=np.int64),
            "__cfg_fresh_lineage_stage": np.asarray(stage),
        }
    )
    return deploy, resume, launch_dsl


def _open(
    tmp_path: Path,
    deploy: dict[str, np.ndarray],
    resume: dict[str, np.ndarray],
    launch_dsl: str,
) -> subject.FreshProducerCheckpointPairV1:
    deploy_path = tmp_path / "deploy.npz"
    resume_path = tmp_path / "resume.npz"
    deploy_sha = _write_npz(deploy_path, deploy)
    resume_sha = _write_npz(resume_path, resume)
    return subject.open_fresh_producer_checkpoint_pair_v1(
        deploy_checkpoint=deploy_path,
        expected_deploy_sha256=deploy_sha,
        resume_checkpoint=resume_path,
        expected_resume_sha256=resume_sha,
        expected_current_launch_dsl_compile_hash=launch_dsl,
    )


def _publish_node(
    tmp_path: Path,
    *,
    name: str,
    deploy: dict[str, np.ndarray],
    resume: dict[str, np.ndarray],
    launch_dsl: str,
    parent: subject.FreshProducerPhysicalCheckpointNodeV1 | None = None,
    run_name: str = "run",
) -> subject.FreshProducerPhysicalCheckpointNodeV1:
    source_dir = tmp_path / "sources"
    source_dir.mkdir(exist_ok=True)
    output_dir = tmp_path / run_name
    output_dir.mkdir(exist_ok=True)
    deploy_path = source_dir / f"{name}.deploy.npz"
    resume_path = source_dir / f"{name}.resume.npz"
    deploy_sha = _write_npz(deploy_path, deploy)
    resume_sha = _write_npz(resume_path, resume)
    return subject.write_fresh_physical_checkpoint_node_v1(
        out_dir=output_dir,
        deploy_checkpoint=deploy_path,
        expected_deploy_sha256=deploy_sha,
        resume_checkpoint=resume_path,
        expected_resume_sha256=resume_sha,
        expected_current_launch_dsl_compile_hash=launch_dsl,
        parent_receipt_path=(
            None if parent is None else parent.receipt_path
        ),
        expected_parent_receipt_sha256=(
            None if parent is None else parent.receipt_sha256
        ),
    )


def test_pair_recomputes_current_node_and_binds_deploy_to_ema(
    tmp_path: Path,
) -> None:
    deploy, resume, launch_dsl = _source_pair_arrays()
    pair = _open(tmp_path, deploy, resume, launch_dsl)
    assert pair.seed == 112
    assert pair.epoch == 0
    assert pair.stage == "stageColdRoot"
    assert pair.live_tensor_count == pair.ema_tensor_count == 5
    assert pair.optimizer_tensor_count == 2
    assert pair.deploy_equals_ema is True
    assert pair.complete_trajectory_proven is False


def test_legacy_v2_seed_off_reopens_without_v3_fields_or_complete_proof(
    tmp_path: Path,
) -> None:
    deploy, resume, launch_dsl = _source_pair_arrays(
        semantic_schema=subject.LEGACY_RESUME_SEMANTIC_SCHEMA,
        include_complete_manifest=False,
        seed_islands=False,
        include_seed_config=False,
    )
    assert "__resume_primary_optimizer_family" not in resume
    assert "__resume_active_trainable_components_json" not in resume
    assert subject.G111_COMPLETE_TRAJECTORY_KEY not in resume
    pair = _open(tmp_path, deploy, resume, launch_dsl)
    assert pair.complete_state_manifest_proven is False
    assert pair.complete_trajectory_proven is False


def test_legacy_v2_active_seed_is_refused(tmp_path: Path) -> None:
    deploy, resume, launch_dsl = _source_pair_arrays(
        semantic_schema=subject.LEGACY_RESUME_SEMANTIC_SCHEMA,
        include_complete_manifest=False,
        seed_islands=True,
    )
    with pytest.raises(
        subject.FreshProducerLineageV1Error,
        match="v2 fresh chains cannot carry an active",
    ):
        _open(tmp_path, deploy, resume, launch_dsl)


def test_v3_without_total_trajectory_manifest_is_refused(
    tmp_path: Path,
) -> None:
    deploy, resume, launch_dsl = _source_pair_arrays(
        semantic_schema=subject.RESUME_SEMANTIC_SCHEMA,
        include_complete_manifest=False,
    )
    with pytest.raises(
        subject.FreshProducerLineageV1Error,
        match="lacks the complete trajectory-state manifest",
    ):
        _open(tmp_path, deploy, resume, launch_dsl)


def test_v3_skeleton_manifest_cannot_claim_complete_trajectory(
    tmp_path: Path,
) -> None:
    deploy, resume, launch_dsl = _source_pair_arrays(
        semantic_schema=subject.RESUME_SEMANTIC_SCHEMA,
        include_complete_manifest=True,
    )
    with pytest.raises(
        subject.FreshProducerLineageV1Error,
        match="skeleton-only",
    ):
        _open(tmp_path, deploy, resume, launch_dsl)


@pytest.mark.parametrize("direction", ["declared_without_heavy", "heavy_without_declared"])
def test_polyak_on_off_checkpoint_drift_is_refused(
    tmp_path: Path,
    direction: str,
) -> None:
    deploy, resume, launch_dsl = _source_pair_arrays()
    if direction == "declared_without_heavy":
        resume.update(
            {
                "__pta_arm": np.asarray(1),
                "__pta_count": np.asarray(3),
                "__pta_start": np.asarray(0),
            }
        )
        match = "without its heavy tensor state"
    else:
        for key in (
            "code",
            "layer.weight",
            "layer.bias",
            "pose_carrier.xi_stored",
            "pose_carrier.dxi",
        ):
            resume[subject.RESUME_POLYAK_PREFIX + key] = np.asarray(
                resume[subject.RESUME_LIVE_PREFIX + key],
                dtype=np.float64,
            )
        match = "without an active scalar controller"
    with pytest.raises(subject.FreshProducerLineageV1Error, match=match):
        _open(tmp_path, deploy, resume, launch_dsl)


def test_deploy_ema_crossmix_fails_closed(tmp_path: Path) -> None:
    deploy, resume, launch_dsl = _source_pair_arrays()
    deploy["layer.bias"] = np.ones(4, dtype=np.float32)
    with pytest.raises(
        subject.FreshProducerLineageV1Error,
        match="differs from companion EMA",
    ):
        _open(tmp_path, deploy, resume, launch_dsl)


def test_state_mutation_without_rehash_fails_closed(tmp_path: Path) -> None:
    deploy, resume, launch_dsl = _source_pair_arrays()
    resume["optP__m"][0] = np.float32(1.0)
    with pytest.raises(
        subject.FreshProducerLineageV1Error,
        match="state SHA-256 does not recompute",
    ):
        _open(tmp_path, deploy, resume, launch_dsl)


def test_cold_root_changes_with_trainable_optimizer_or_support_state():
    base = {
        "live_state": {"w": np.asarray([1.0], np.float32)},
        "ema_state": {"w": np.asarray([1.0], np.float32)},
        "optimizer_state": {"m": np.asarray([0.0], np.float32)},
        "seed_state": {"residual": np.asarray([[2.0]], np.float32)},
        "seed_optimizer_state": {"m": np.asarray([[0.0]], np.float32)},
        "polyak_state": {},
        "config_state": {
            "__resume_seed_support_geometry_sha256": np.asarray(_sha("support-a"))
        },
    }
    initial = subject.fresh_resume_semantic_state_sha256(**base)
    root = subject.fresh_producer_root_sha256(
        seed=0,
        dsl_compile_hash=_sha("dsl"),
        target_projection_sha256=_sha("target"),
        initial_state_sha256=initial,
    )
    mutations = (
        ("live_state", {"w": np.asarray([1.25], np.float32)}),
        ("optimizer_state", {"m": np.asarray([0.5], np.float32)}),
        ("seed_state", {"residual": np.asarray([[2.5]], np.float32)}),
        ("seed_optimizer_state", {"m": np.asarray([[0.25]], np.float32)}),
        (
            "config_state",
            {
                "__resume_seed_support_geometry_sha256": np.asarray(
                    _sha("support-b")
                )
            },
        ),
    )
    for domain, changed in mutations:
        candidate = dict(base)
        candidate[domain] = changed
        changed_initial = subject.fresh_resume_semantic_state_sha256(**candidate)
        changed_root = subject.fresh_producer_root_sha256(
            seed=0,
            dsl_compile_hash=_sha("dsl"),
            target_projection_sha256=_sha("target"),
            initial_state_sha256=changed_initial,
        )
        assert changed_initial != initial
        assert changed_root != root


def test_arbitrary_parent_is_only_a_current_node_assertion(
    tmp_path: Path,
) -> None:
    deploy, resume, launch_dsl = _source_pair_arrays(
        parent_id=_sha("opaque-unopened-parent"),
    )
    pair = _open(tmp_path, deploy, resume, launch_dsl)
    assert pair.parent_checkpoint_id_sha256 == _sha(
        "opaque-unopened-parent"
    )
    assert pair.complete_trajectory_proven is False


def test_external_current_launch_hash_is_required(tmp_path: Path) -> None:
    deploy, resume, _launch_dsl = _source_pair_arrays()
    with pytest.raises(
        subject.FreshProducerLineageV1Error,
        match="caller custody",
    ):
        _open(tmp_path, deploy, resume, _sha("other-launch"))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda arrays: arrays.pop("__rng_np_keys"),
            "lacks RNG state",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "__cfg_film_stiefel",
                np.asarray(1),
            ),
            "state SHA-256 does not recompute",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "__recent_losses",
                np.asarray([np.nan], dtype=np.float64),
            ),
            "non-finite",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "__resume_stage",
                np.asarray("wrong"),
            ),
            "event ledger stage differs",
        ),
    ],
)
def test_incomplete_resume_semantics_fail_closed(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    deploy, resume, launch_dsl = _source_pair_arrays()
    mutation(resume)
    with pytest.raises(subject.FreshProducerLineageV1Error, match=match):
        _open(tmp_path, deploy, resume, launch_dsl)


def test_physical_node_writer_and_opener_prove_root_to_current_chain(
    tmp_path: Path,
) -> None:
    deploy0, resume0, launch0 = _source_pair_arrays()
    root = _publish_node(
        tmp_path,
        name="root",
        deploy=deploy0,
        resume=resume0,
        launch_dsl=launch0,
    )
    deploy1, resume1, launch1 = _source_pair_arrays(
        parent_id=root.pair.checkpoint_id_sha256,
        epoch=18,
        stage="l7",
    )
    child = _publish_node(
        tmp_path,
        name="child",
        deploy=deploy1,
        resume=resume1,
        launch_dsl=launch1,
        parent=root,
        run_name="resumed-run",
    )
    chain = subject.open_fresh_physical_checkpoint_chain_v1(
        child.receipt_path,
        expected_receipt_sha256=child.receipt_sha256,
        expected_current_launch_dsl_compile_hash=launch1,
    )
    assert root.complete_trajectory_proven is False
    assert child.complete_trajectory_proven is False
    assert chain.complete_trajectory_proven is False
    assert len(chain.nodes) == 2
    assert [node.sequence_index for node in chain.nodes] == [0, 1]
    assert chain.nodes[0].pair.parent_checkpoint_id_sha256 == "0" * 64
    assert (
        chain.current.pair.parent_checkpoint_id_sha256
        == chain.nodes[0].pair.checkpoint_id_sha256
    )
    assert chain.current.pair.epoch == 18
    assert chain.current.pair.stage == "l7"
    assert chain.current.receipt_path.name == (
        f"{chain.current.pair.checkpoint_id_sha256}.receipt.json"
    )
    assert (
        chain.nodes[0].receipt_path.parent
        != chain.current.receipt_path.parent
    )


def test_nonroot_node_without_physical_parent_fails_closed(
    tmp_path: Path,
) -> None:
    deploy, resume, launch = _source_pair_arrays(
        parent_id=_sha("invented-parent"),
    )
    with pytest.raises(
        subject.FreshProducerLineageV1Error,
        match="requires a physical parent receipt",
    ):
        _publish_node(
            tmp_path,
            name="orphan",
            deploy=deploy,
            resume=resume,
            launch_dsl=launch,
        )


def test_forged_sequence_with_resealed_receipt_fails_closed(
    tmp_path: Path,
) -> None:
    deploy0, resume0, launch0 = _source_pair_arrays()
    root = _publish_node(
        tmp_path,
        name="root",
        deploy=deploy0,
        resume=resume0,
        launch_dsl=launch0,
    )
    deploy1, resume1, launch1 = _source_pair_arrays(
        parent_id=root.pair.checkpoint_id_sha256,
        epoch=18,
    )
    child = _publish_node(
        tmp_path,
        name="child",
        deploy=deploy1,
        resume=resume1,
        launch_dsl=launch1,
        parent=root,
    )
    receipt = json.loads(child.receipt_path.read_text(encoding="ascii"))
    receipt["sequence_index"] = 9
    body = dict(receipt)
    body.pop("receipt_sha256")
    receipt["receipt_sha256"] = hashlib.sha256(
        subject._canonical_json_bytes(body)
    ).hexdigest()
    payload = subject._canonical_json_bytes(receipt) + b"\n"
    child.receipt_path.write_bytes(payload)
    with pytest.raises(
        subject.FreshProducerLineageV1Error,
        match="sequence index is not contiguous",
    ):
        subject.open_fresh_physical_checkpoint_chain_v1(
            child.receipt_path,
            expected_receipt_sha256=hashlib.sha256(payload).hexdigest(),
            expected_current_launch_dsl_compile_hash=launch1,
        )


def test_foreign_root_cannot_attach_to_valid_parent(
    tmp_path: Path,
) -> None:
    deploy0, resume0, launch0 = _source_pair_arrays()
    root = _publish_node(
        tmp_path,
        name="root",
        deploy=deploy0,
        resume=resume0,
        launch_dsl=launch0,
    )
    foreign_deploy, foreign_resume, foreign_launch = _source_pair_arrays(
        parent_id=root.pair.checkpoint_id_sha256,
        seed=999,
        epoch=18,
    )
    with pytest.raises(
        subject.FreshProducerLineageV1Error,
        match="does not continue its reopened parent",
    ):
        _publish_node(
            tmp_path,
            name="foreign",
            deploy=foreign_deploy,
            resume=foreign_resume,
            launch_dsl=foreign_launch,
            parent=root,
        )
