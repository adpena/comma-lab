from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

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
    configs = {
        "__cfg_fresh_producer": np.asarray(1, dtype=np.int8),
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
    }
    return {**params, **configs, **_fake_target_arrays()}


def _write_checkpoint(path: Path, arrays: dict[str, np.ndarray]) -> str:
    payload = subject._deterministic_npz_bytes(arrays)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_physical_checkpoint_path_sha_and_symlink_fail_closed(
    tmp_path: Path,
) -> None:
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
    first_checkpoint = tmp_path / "source-a.npz"
    second_checkpoint = tmp_path / "source-b.npz"
    first_sha = _write_checkpoint(first_checkpoint, first_arrays)
    second_sha = _write_checkpoint(second_checkpoint, second_arrays)

    first = subject.materialize_g112_checkpoint_partition(
        checkpoint=first_checkpoint,
        expected_checkpoint_sha256=first_sha,
        output_root=tmp_path / "out-a",
        allowed_output_roots=(tmp_path,),
    )
    second = subject.materialize_g112_checkpoint_partition(
        checkpoint=second_checkpoint,
        expected_checkpoint_sha256=second_sha,
        output_root=tmp_path / "out-b",
        allowed_output_roots=(tmp_path,),
    )
    assert first_sha != second_sha
    assert first.semantic_child_sha256 == second.semantic_child_sha256
    assert first.initializer_sha256 == second.initializer_sha256
    assert first.semantic_packet_sha256 == second.semantic_packet_sha256

    child = subject.open_g112_semantic_child(
        first.semantic_child_path,
        expected_sha256=first.semantic_child_sha256,
    )
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
    checkpoint = tmp_path / "bad.npz"
    sha = _write_checkpoint(checkpoint, arrays)
    with pytest.raises(subject.G112CheckpointPartitionError, match=match):
        subject.materialize_g112_checkpoint_partition(
            checkpoint=checkpoint,
            expected_checkpoint_sha256=sha,
            output_root=tmp_path / "out",
            allowed_output_roots=(tmp_path,),
        )


def test_extra_target_config_fails_closed(
    tmp_path: Path,
    fake_target_custody: None,
) -> None:
    arrays = _source_arrays()
    arrays["__cfg_g46_orphan"] = np.asarray("forbidden")
    checkpoint = tmp_path / "bad-target.npz"
    sha = _write_checkpoint(checkpoint, arrays)
    with pytest.raises(
        subject.G112CheckpointPartitionError,
        match="target config set is not exact",
    ):
        subject.materialize_g112_checkpoint_partition(
            checkpoint=checkpoint,
            expected_checkpoint_sha256=sha,
            output_root=tmp_path / "out",
            allowed_output_roots=(tmp_path,),
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
