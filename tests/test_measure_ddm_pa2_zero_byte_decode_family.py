# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.ddm_pa2_zero_byte_decode_family import PA2Member
from tools.measure_ddm_pa2_zero_byte_decode_family import (
    CONFIG_SCHEMA,
    PA1_MEMBER,
    PA2MeasurementError,
    _advisory_objective,
    _load_config,
    _publish_bytes,
    _publish_json,
)

REPO = Path(__file__).resolve().parents[1]


def test_live_config_is_strict_and_orders_three_bases() -> None:
    config, digest = _load_config(
        REPO / ".omx/research/configs/ddm_pa2_zero_byte_decode_family_20260724.json"
    )
    assert config.schema_ == CONFIG_SCHEMA
    assert len(digest) == 64
    assert [row.base_id for row in config.bases] == [
        "IC1_W_joint_PA1",
        "IC2_W_seg_PA1",
        "MS2R_q4_q8",
    ]
    assert config.scorer_batch_size == 32
    assert config.main_review_required is True


def test_pa1_is_not_represented_as_counted_payload() -> None:
    assert PA1_MEMBER.startswith("ddm_pa1_")
    assert PA2Member.BLIND_ZERO_FILL.value.startswith("pa2_")


def test_immutable_json_checkpoint_refuses_drift(tmp_path: Path) -> None:
    path = tmp_path / "row.json"
    _publish_json(path, {"schema": "test.v1", "value": 1})
    first = path.read_bytes()
    _publish_json(path, {"schema": "test.v1", "value": 1})
    assert path.read_bytes() == first
    with pytest.raises(PA2MeasurementError, match="immutable"):
        _publish_json(path, {"schema": "test.v1", "value": 2})


def test_immutable_binary_stage_refuses_drift(tmp_path: Path) -> None:
    path = tmp_path / "batch.raw"
    payload = np.arange(256, dtype=np.uint8).tobytes()
    _publish_bytes(path, payload)
    _publish_bytes(path, payload)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(payload).hexdigest()
    with pytest.raises(PA2MeasurementError, match="immutable"):
        _publish_bytes(path, b"different")


def test_config_has_no_hidden_candidate_tables() -> None:
    value = json.loads(
        (
            REPO
            / ".omx/research/configs/ddm_pa2_zero_byte_decode_family_20260724.json"
        ).read_bytes()
    )
    forbidden = {"pair_table", "coefficients", "gamma_table", "gauge_table", "residual"}
    assert forbidden.isdisjoint(value)


def test_advisory_objective_is_exact_contest_formula() -> None:
    value = _advisory_objective(
        errors=0,
        sites=600 * 384 * 512,
        d_pose=0.0,
        bytes_=37_545_489,
    )
    assert value == 25.0
