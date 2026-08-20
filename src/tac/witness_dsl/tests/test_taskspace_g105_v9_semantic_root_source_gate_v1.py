from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.witness_dsl.taskspace_g105_v9_semantic_root_source_gate_v1 import (
    EXPECTED_LATENT_ROWS,
    EXPECTED_PAIR_COUNT,
    ArtifactRefV1,
    G105BlockerCodeV1,
    G105SourceGateError,
    candidate_physical_blockers,
    canonical_g105_pair_population,
    inspect_v9_checkpoint,
    inspect_v9_resume_checkpoint,
)


def _ref(path: Path) -> ArtifactRefV1:
    payload = path.read_bytes()
    return ArtifactRefV1(
        path=path,
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _write_checkpoint(
    path: Path,
    *,
    rows: int = EXPECTED_LATENT_ROWS,
    git_sha: str = "a" * 40,
    upstream_sha256: str = "b" * 64,
    git_dirty: int = 0,
    fresh_init: int = 1,
    even_fill: float = 0.0,
) -> None:
    code = np.zeros((rows, 19), dtype=np.float32)
    code[0::2] = even_fill
    arrays = {
        "code": code,
        "in_proj.weight": np.zeros((8, 4), dtype=np.float32),
        "in_proj.bias": np.zeros((8,), dtype=np.float32),
        "film.weight": np.zeros((16, 19), dtype=np.float32),
        "film.bias": np.zeros((16,), dtype=np.float32),
        "hidden.0.weight": np.zeros((8, 8), dtype=np.float32),
        "hidden.0.bias": np.zeros((8,), dtype=np.float32),
        "out_sdf.weight": np.zeros((5, 8), dtype=np.float32),
        "out_sdf.bias": np.zeros((5,), dtype=np.float32),
        "out_tex.weight": np.zeros((3, 8), dtype=np.float32),
        "out_tex.bias": np.zeros((3,), dtype=np.float32),
        "palette": np.zeros((5, 3), dtype=np.float32),
        "__cfg_chroma": np.asarray(1, dtype=np.int64),
        "__cfg_w_pose": np.asarray(1.0, dtype=np.float64),
        "__cfg_git_sha": np.asarray(git_sha),
        "__cfg_git_dirty": np.asarray(git_dirty, dtype=np.int8),
        "__cfg_upstream_snapshot_sha256": np.asarray(upstream_sha256),
        "__cfg_fresh_init": np.asarray(fresh_init, dtype=np.int8),
        "__epoch": np.asarray(100, dtype=np.int64),
    }
    np.savez(path, **arrays)


def _write_resume(
    path: Path,
    *,
    git_sha: str = "a" * 40,
    upstream_sha256: str = "b" * 64,
    git_dirty: int = 0,
    fresh_init: int = 1,
    include_optimizer: bool = True,
) -> None:
    arrays = {
        "P__code": np.zeros((EXPECTED_LATENT_ROWS, 19), dtype=np.float32),
        "EMA__code": np.zeros((EXPECTED_LATENT_ROWS, 19), dtype=np.float32),
        "__resume_epoch": np.asarray(100, dtype=np.int64),
        "__resume_stage": np.asarray("stageMuon_ep100"),
        "__resume_has_opt": np.asarray(int(include_optimizer), dtype=np.int8),
        "__cfg_fresh_init": np.asarray(fresh_init, dtype=np.int8),
        "__cfg_git_sha": np.asarray(git_sha),
        "__cfg_git_dirty": np.asarray(git_dirty, dtype=np.int8),
        "__cfg_upstream_snapshot_sha256": np.asarray(upstream_sha256),
    }
    if include_optimizer:
        arrays["optP__state"] = np.zeros((1,), dtype=np.float32)
    np.savez(path, **arrays)


def test_population_is_selected_compiler_owned_full_n600() -> None:
    population = canonical_g105_pair_population()
    assert type(population).__module__.endswith("taskspace_selected_solution_compiler")
    assert population.global_pair_ids == tuple(range(EXPECTED_PAIR_COUNT))
    assert len(population.v10_local_coordinates) == EXPECTED_PAIR_COUNT
    assert len(population.binding_sha256) == 64


def test_checkpoint_inventory_reads_learned_generator_and_temporal_code(tmp_path: Path) -> None:
    path = tmp_path / "fresh.npz"
    _write_checkpoint(path)

    inventory = inspect_v9_checkpoint(_ref(path))

    assert inventory.full_n600
    assert inventory.learned_surface_complete
    assert inventory.pair_count == EXPECTED_PAIR_COUNT
    assert inventory.modulation_dim == 19
    assert inventory.hidden_layer_ids == (0,)
    assert inventory.chroma_enabled
    assert inventory.pose_weight == 1.0
    assert inventory.fresh_init
    assert inventory.git_dirty is False
    assert inventory.epoch == 100
    assert inventory.y1_latent_projection_sha256 is not None
    assert {row.name for row in inventory.tensors} >= {
        "code",
        "film.weight",
        "out_sdf.weight",
        "out_tex.weight",
        "palette",
    }


def test_subset_checkpoint_is_not_promoted_to_full_n600(tmp_path: Path) -> None:
    path = tmp_path / "subset.npz"
    _write_checkpoint(path, rows=48)

    inventory = inspect_v9_checkpoint(_ref(path))

    assert inventory.pair_count == 24
    assert not inventory.full_n600


def test_checkpoint_hash_drift_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "fresh.npz"
    _write_checkpoint(path)
    wrong = ArtifactRefV1(path=path, bytes=path.stat().st_size, sha256="0" * 64)

    with pytest.raises(G105SourceGateError, match="SHA-256 drift"):
        inspect_v9_checkpoint(wrong)


def test_checkpoint_symlink_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "fresh.npz"
    alias = tmp_path / "alias.npz"
    _write_checkpoint(path)
    alias.symlink_to(path)
    payload = path.read_bytes()
    ref = ArtifactRefV1(
        path=alias,
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )

    with pytest.raises(G105SourceGateError, match="non-symlink regular file"):
        inspect_v9_checkpoint(ref)


def test_arbitrary_nonunknown_provenance_cannot_pass(tmp_path: Path) -> None:
    checkpoint = tmp_path / "fresh.npz"
    resume = tmp_path / "resume.npz"
    _write_checkpoint(checkpoint, git_sha="c" * 40, upstream_sha256="d" * 64)
    _write_resume(resume, git_sha="c" * 40, upstream_sha256="d" * 64)

    blockers = candidate_physical_blockers(
        inspect_v9_checkpoint(_ref(checkpoint)),
        inspect_v9_resume_checkpoint(_ref(resume)),
        expected_git_sha="a" * 40,
        expected_upstream_sha256="b" * 64,
    )

    assert G105BlockerCodeV1.CHECKPOINT_SOURCE_AUTHORITY_DRIFT in blockers


@pytest.mark.parametrize(
    ("git_dirty", "fresh_init", "expected_blocker"),
    [
        (1, 1, G105BlockerCodeV1.CHECKPOINT_SOURCE_AUTHORITY_DRIFT),
        (0, 0, G105BlockerCodeV1.CHECKPOINT_NOT_FRESH_INIT),
    ],
)
def test_dirty_or_nonfresh_physical_state_is_blocked(
    tmp_path: Path,
    git_dirty: int,
    fresh_init: int,
    expected_blocker: G105BlockerCodeV1,
) -> None:
    checkpoint = tmp_path / "fresh.npz"
    resume = tmp_path / "resume.npz"
    _write_checkpoint(checkpoint, git_dirty=git_dirty, fresh_init=fresh_init)
    _write_resume(resume, git_dirty=git_dirty, fresh_init=fresh_init)

    blockers = candidate_physical_blockers(
        inspect_v9_checkpoint(_ref(checkpoint)),
        inspect_v9_resume_checkpoint(_ref(resume)),
        expected_git_sha="a" * 40,
        expected_upstream_sha256="b" * 64,
    )

    assert expected_blocker in blockers


def test_y1_projection_ignores_encoder_only_frame0_rows(tmp_path: Path) -> None:
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    _write_checkpoint(left, even_fill=0.0)
    _write_checkpoint(right, even_fill=7.0)

    left_inventory = inspect_v9_checkpoint(_ref(left))
    right_inventory = inspect_v9_checkpoint(_ref(right))

    assert left_inventory.artifact.sha256 != right_inventory.artifact.sha256
    assert left_inventory.y1_latent_projection_sha256 == right_inventory.y1_latent_projection_sha256


def test_resume_without_optimizer_state_is_not_stage_resumable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "fresh.npz"
    resume = tmp_path / "resume.npz"
    _write_checkpoint(checkpoint)
    _write_resume(resume, include_optimizer=False)

    blockers = candidate_physical_blockers(
        inspect_v9_checkpoint(_ref(checkpoint)),
        inspect_v9_resume_checkpoint(_ref(resume)),
        expected_git_sha="a" * 40,
        expected_upstream_sha256="b" * 64,
    )

    assert G105BlockerCodeV1.CHECKPOINT_NOT_STAGE_RESUMABLE in blockers


def test_blocker_names_are_candidate_claim_explicit() -> None:
    assert (
        G105BlockerCodeV1.FRESH_G46_SOURCE_BOUND_CHECKPOINT_OWED.value
        == "G105_FRESH_G46_SOURCE_BOUND_V9_V10_CHECKPOINT_OWED"
    )
    assert (
        G105BlockerCodeV1.SUPERSEDED_BY_PHYSICAL_G109_G111_G112.value
        == "G105_SUPERSEDED_BY_PHYSICAL_G109_G111_G112"
    )
    assert json.loads(json.dumps([item.value for item in G105BlockerCodeV1]))
