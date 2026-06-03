# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

import pytest

from tac.substrates.snerv_inverse_steg_carrier.advisory import (
    resolve_snerv_modelsize_control,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import SnervCarrierError


def test_official_modelsize_control_solves_fc_dim_and_metadata() -> None:
    """NO-FAKE: the advisory uses the source-bound quadratic, not a marker."""

    resolved = resolve_snerv_modelsize_control(
        full_data_length=1200,
        final_size=384 * 512,
        snerv_fc_dim=9,
        snerv_emb_size=0,
        snerv_official_modelsize_mparams=0.05,
    )

    assert resolved.capacity_source == "official_snerv_modelsize"
    assert resolved.model_size.fc_dim == 11
    assert resolved.model_size.feature_count == 11
    solution = resolved.official_modelsize_solution
    assert solution is not None
    assert solution["schema"] == "official_snerv_modelsize_to_fc_dim.v1"
    assert solution["modelsize_mparams"] == 0.05
    assert solution["full_data_length"] == 1200
    assert solution["final_size"] == 384 * 512
    assert solution["fc_dim"] == 11
    assert solution["score_claim"] is False
    assert solution["ready_for_exact_eval_dispatch"] is False
    metadata = resolved.metadata()
    assert metadata["official_modelsize_solution"]["fc_dim"] == 11
    json.dumps(metadata)


def test_official_modelsize_control_rejects_explicit_conflicting_fc_dim() -> None:
    with pytest.raises(SnervCarrierError, match="conflicts"):
        resolve_snerv_modelsize_control(
            full_data_length=1200,
            final_size=384 * 512,
            snerv_fc_dim=9,
            snerv_fc_dim_explicit=True,
            snerv_official_modelsize_mparams=0.05,
        )


def test_official_modelsize_control_rejects_invalid_mparams_before_metadata() -> None:
    with pytest.raises(SnervCarrierError, match="modelsize_mparams must be positive"):
        resolve_snerv_modelsize_control(
            full_data_length=1200,
            final_size=384 * 512,
            snerv_official_modelsize_mparams=-0.01,
        )


def test_manual_fc_dim_remains_manual_when_no_official_modelsize() -> None:
    resolved = resolve_snerv_modelsize_control(
        full_data_length=1200,
        final_size=384 * 512,
        snerv_fc_dim=7,
    )

    assert resolved.capacity_source == "manual_fc_dim"
    assert resolved.model_size.fc_dim == 7
    assert resolved.official_modelsize_solution is None
    assert resolved.metadata()["official_modelsize_solution"] is None
