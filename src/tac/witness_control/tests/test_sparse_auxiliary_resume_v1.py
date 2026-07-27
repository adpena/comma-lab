# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

import numpy as np
import pytest

from tac.witness_control.sparse_auxiliary_resume_v1 import (
    SelectedStateLeaf,
    SparseSupportGeometry,
    pack_sparse_auxiliary_selected_state,
    pack_sparse_auxiliary_state,
    restore_sparse_auxiliary_state,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    mask = np.zeros((2, 3, 4, 1), np.float32)
    mask[0, 0, 1, 0] = 0.25
    mask[0, 2, 3, 0] = 1.0
    mask[1, 1, 2, 0] = 0.5
    residual = np.zeros((2, 3, 4, 3), np.float32)
    residual[mask[..., 0] != 0.0] = np.arange(9, dtype=np.float32).reshape(3, 3)
    optimizer = {
        "step": np.asarray(7, np.uint64),
        "learning_rate": np.asarray(1e-3, np.float32),
        "residual.m": residual * np.float32(0.1),
        "residual.v": residual * residual * np.float32(0.01),
    }
    return mask, residual, optimizer


def test_sparse_auxiliary_roundtrip_is_bit_exact_and_compact():
    mask, residual, optimizer = _fixture()
    packed = pack_sparse_auxiliary_state(
        component="island_seed",
        live_state={"residual": residual},
        optimizer_state=optimizer,
        support_mask=mask,
        dense_shape=residual.shape,
    )
    live2, opt2 = restore_sparse_auxiliary_state(
        component="island_seed",
        packed_live=packed.live,
        packed_optimizer=packed.optimizer,
        manifest_json=packed.manifest_json,
        support_mask=mask,
    )
    assert packed.support_count == 3
    assert packed.live["residual"].shape == (3, 3)
    assert packed.optimizer["residual.m"].shape == (3, 3)
    assert np.array_equal(live2["residual"], residual)
    assert opt2.keys() == optimizer.keys()
    assert all(np.array_equal(opt2[key], optimizer[key]) for key in optimizer)
    dense_bytes = residual.nbytes + sum(value.nbytes for value in optimizer.values())
    packed_bytes = sum(value.nbytes for value in packed.live.values()) + sum(
        value.nbytes for value in packed.optimizer.values()
    )
    assert packed_bytes < dense_bytes / 4


def test_sparse_auxiliary_refuses_signal_outside_deterministic_support():
    mask, residual, optimizer = _fixture()
    residual[0, 0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="outside deterministic support"):
        pack_sparse_auxiliary_state(
            component="island_seed",
            live_state={"residual": residual},
            optimizer_state=optimizer,
            support_mask=mask,
            dense_shape=residual.shape,
        )


def test_sparse_auxiliary_refuses_support_value_or_keyset_drift():
    mask, residual, optimizer = _fixture()
    packed = pack_sparse_auxiliary_state(
        component="island_seed",
        live_state={"residual": residual},
        optimizer_state=optimizer,
        support_mask=mask,
        dense_shape=residual.shape,
    )
    changed = mask.copy()
    changed[0, 0, 1, 0] = 0.5
    with pytest.raises(ValueError, match="support geometry changed"):
        restore_sparse_auxiliary_state(
            component="island_seed",
            packed_live=packed.live,
            packed_optimizer=packed.optimizer,
            manifest_json=packed.manifest_json,
            support_mask=changed,
        )
    with pytest.raises(ValueError, match="keyset mismatch"):
        restore_sparse_auxiliary_state(
            component="island_seed",
            packed_live={},
            packed_optimizer=packed.optimizer,
            manifest_json=packed.manifest_json,
            support_mask=mask,
        )


def test_sparse_auxiliary_refuses_dtype_shape_and_optimizer_family_drift():
    mask, residual, optimizer = _fixture()
    packed = pack_sparse_auxiliary_state(
        component="island_seed",
        live_state={"residual": residual},
        optimizer_state=optimizer,
        support_mask=mask,
        dense_shape=residual.shape,
        optimizer_family="adamw",
    )
    wrong_dtype = dict(packed.live)
    wrong_dtype["residual"] = wrong_dtype["residual"].astype(np.float64)
    with pytest.raises(ValueError, match="dtype mismatch"):
        restore_sparse_auxiliary_state(
            component="island_seed",
            packed_live=wrong_dtype,
            packed_optimizer=packed.optimizer,
            manifest_json=packed.manifest_json,
            support_mask=mask,
        )
    wrong_shape = dict(packed.live)
    wrong_shape["residual"] = wrong_shape["residual"][:-1]
    with pytest.raises(ValueError, match="shape mismatch"):
        restore_sparse_auxiliary_state(
            component="island_seed",
            packed_live=wrong_shape,
            packed_optimizer=packed.optimizer,
            manifest_json=packed.manifest_json,
            support_mask=mask,
        )
    with pytest.raises(ValueError, match="optimizer family changed"):
        restore_sparse_auxiliary_state(
            component="island_seed",
            packed_live=packed.live,
            packed_optimizer=packed.optimizer,
            manifest_json=packed.manifest_json,
            support_mask=mask,
            expected_optimizer_family="muon",
        )


def test_sparse_auxiliary_refuses_nonfinite_packed_leaf_after_tamper():
    mask, residual, optimizer = _fixture()
    packed = pack_sparse_auxiliary_state(
        component="island_seed",
        live_state={"residual": residual},
        optimizer_state=optimizer,
        support_mask=mask,
        dense_shape=residual.shape,
    )
    tampered = dict(packed.live)
    tampered["residual"] = tampered["residual"].copy()
    tampered["residual"][0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        restore_sparse_auxiliary_state(
            component="island_seed",
            packed_live=tampered,
            packed_optimizer=packed.optimizer,
            manifest_json=packed.manifest_json,
            support_mask=mask,
        )


@pytest.mark.parametrize("sparse_geometry", [False, True])
def test_sparse_auxiliary_refuses_empty_active_support(
    sparse_geometry: bool,
):
    mask, residual, optimizer = _fixture()
    mask.fill(0.0)
    support = (
        SparseSupportGeometry(
            indices=np.asarray([], dtype=np.uint32),
            values=np.asarray([], dtype=np.float32),
            spatial_shape=mask.shape[:-1],
        )
        if sparse_geometry
        else mask
    )
    with pytest.raises(ValueError, match="at least one active row"):
        pack_sparse_auxiliary_state(
            component="island_seed",
            live_state={"residual": residual},
            optimizer_state=optimizer,
            support_mask=support,
            dense_shape=residual.shape,
        )


def _adam_step(
    residual: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    *,
    step: int,
    grad: np.ndarray,
    lr: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta1, beta2 = 0.9, 0.999
    m2 = beta1 * m + (1.0 - beta1) * grad
    v2 = beta2 * v + (1.0 - beta2) * grad * grad
    mh = m2 / (1.0 - beta1**step)
    vh = v2 / (1.0 - beta2**step)
    return residual - lr * mh / (np.sqrt(vh) + 1e-8), m2, v2


def test_interrupted_sparse_adam_trajectory_equals_uninterrupted():
    mask, residual0, _ = _fixture()
    support = mask != 0.0
    grad1 = np.broadcast_to(support, residual0.shape).astype(np.float32) * np.float32(0.125)
    grad2 = np.broadcast_to(support, residual0.shape).astype(np.float32) * np.float32(-0.0625)
    zeros = np.zeros_like(residual0)

    residual1, m1, v1 = _adam_step(
        residual0, zeros, zeros, step=1, grad=grad1
    )
    expected_residual, expected_m, expected_v = _adam_step(
        residual1, m1, v1, step=2, grad=grad2
    )

    packed = pack_sparse_auxiliary_state(
        component="island_seed",
        live_state={"residual": residual1},
        optimizer_state={
            "step": np.asarray(1, np.uint64),
            "learning_rate": np.asarray(1e-3, np.float32),
            "residual.m": m1,
            "residual.v": v1,
        },
        support_mask=mask,
        dense_shape=residual0.shape,
    )
    restored_live, restored_opt = restore_sparse_auxiliary_state(
        component="island_seed",
        packed_live=packed.live,
        packed_optimizer=packed.optimizer,
        manifest_json=packed.manifest_json,
        support_mask=mask,
    )
    actual_residual, actual_m, actual_v = _adam_step(
        restored_live["residual"],
        restored_opt["residual.m"],
        restored_opt["residual.v"],
        step=int(restored_opt["step"]) + 1,
        grad=grad2,
    )
    assert np.array_equal(actual_residual, expected_residual)
    assert np.array_equal(actual_m, expected_m)
    assert np.array_equal(actual_v, expected_v)


def test_manifest_is_canonical_json():
    mask, residual, optimizer = _fixture()
    packed = pack_sparse_auxiliary_state(
        component="island_seed",
        live_state={"residual": residual},
        optimizer_state=optimizer,
        support_mask=mask,
        dense_shape=residual.shape,
    )
    parsed = json.loads(packed.manifest_json)
    assert packed.manifest_json == json.dumps(
        parsed, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def test_selected_row_api_never_requires_dense_host_tensor():
    mask, residual, optimizer = _fixture()
    support = np.flatnonzero(mask[..., 0].reshape(-1) != 0.0)

    def selected_leaf(value: np.ndarray) -> SelectedStateLeaf:
        rows = value.reshape(-1, value.shape[-1])[support]
        return SelectedStateLeaf(
            value=rows,
            logical_shape=value.shape,
            dtype=value.dtype.str,
            encoding="support_rows",
        )

    sparse_support = SparseSupportGeometry(
        indices=support,
        values=mask[..., 0].reshape(-1)[support],
        spatial_shape=mask.shape[:-1],
    )
    packed = pack_sparse_auxiliary_selected_state(
        component="island_seed",
        live_state={"residual": selected_leaf(residual)},
        optimizer_state={
            key: (
                selected_leaf(value)
                if value.shape == residual.shape
                else SelectedStateLeaf(
                    value=value,
                    logical_shape=value.shape,
                    dtype=value.dtype.str,
                    encoding="full",
                )
            )
            for key, value in optimizer.items()
        },
        support_mask=sparse_support,
        dense_shape=residual.shape,
    )
    live2, opt2 = restore_sparse_auxiliary_state(
        component="island_seed",
        packed_live=packed.live,
        packed_optimizer=packed.optimizer,
        manifest_json=packed.manifest_json,
        support_mask=sparse_support,
    )
    assert np.array_equal(live2["residual"], residual)
    assert all(np.array_equal(opt2[key], optimizer[key]) for key in optimizer)


def test_selected_row_api_refuses_dense_value_disguised_as_selected():
    mask, residual, optimizer = _fixture()
    with pytest.raises(ValueError, match="selected support-row shape mismatch"):
        pack_sparse_auxiliary_selected_state(
            component="island_seed",
            live_state={
                "residual": SelectedStateLeaf(
                    value=residual,
                    logical_shape=residual.shape,
                    dtype=residual.dtype.str,
                    encoding="support_rows",
                )
            },
            optimizer_state={
                "step": SelectedStateLeaf(
                    value=optimizer["step"],
                    logical_shape=optimizer["step"].shape,
                    dtype=optimizer["step"].dtype.str,
                    encoding="full",
                )
            },
            support_mask=mask,
            dense_shape=residual.shape,
        )
