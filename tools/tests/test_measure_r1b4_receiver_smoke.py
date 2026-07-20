from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.boundary_coordinate_joint_solve import decode_boundary_packet
from tools import measure_r1b4_receiver_smoke as smoke


def test_receiver_smoke_boundary_is_typed_n600_zero_effect() -> None:
    packet = decode_boundary_packet(smoke._zero_boundary_payload())
    assert packet.pair_count == 600
    assert (packet.scorer_height, packet.scorer_width) == (384, 512)
    assert packet.coefficients.dtype == np.dtype(np.int8)
    assert np.count_nonzero(packet.coefficients) == 0
    assert np.all(packet.scales == np.float16(1.0))


def test_storage_preflight_records_explicit_local_scope(tmp_path: Path) -> None:
    result = smoke._storage_preflight(tmp_path / "artifacts", pair_cap=2)
    assert result["tier"] == "local_explicit"
    assert result["ok"] is True
    assert result["raw_bytes_each"] == 2 * 2 * 874 * 1164 * 3
    assert result["required_free_bytes"] > result["raw_bytes_each"]


def test_atomic_receipt_is_parseable_and_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    smoke._atomic_json(path, {"schema": "fixture.v1", "score_claim": False})
    assert json.loads(path.read_text()) == {"schema": "fixture.v1", "score_claim": False}
    with pytest.raises(smoke.R1B4SmokeError, match="overwrite refused"):
        smoke._atomic_json(path, {"schema": "fixture.v2"})


def test_atomic_receipt_preserves_stale_temporary_for_review(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    partial = path.with_name(f".{path.name}.tmp.{smoke.os.getpid()}")
    partial.write_bytes(b"stale-evidence")
    with pytest.raises(smoke.R1B4SmokeError, match="stale receipt temporary"):
        smoke._atomic_json(path, {"schema": "fixture.v1"})
    assert partial.read_bytes() == b"stale-evidence"
    assert not path.exists()
