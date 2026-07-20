from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from tac.boundary_math.r1b4_section_receiver import encode_replay_payload
from tac.optimization.r1b2_mdl_xi0_compile import _write_zip
from tac.optimization.r1b3_producer_preflight import encode_xi0_payload
from tools.audit_r1b5_carrier_bytes import (
    _BOUNDARY_PREFIX,
    _measure_case,
    _raw_deflate_size,
    _section_envelope,
    _zero_boundary_payload,
)


def test_settled_typed_fixture_decomposes_to_exact_2114_bytes() -> None:
    with tempfile.TemporaryDirectory() as temp_name:
        root = Path(temp_name)
        control = root / "control.zip"
        _write_zip(control, [("0.bin", b"base" * 256), ("ipe_manifest.json", b"{}")])
        result = _measure_case(
            control=control,
            boundary=_zero_boundary_payload(),
            replay=encode_replay_payload(()),
            xi0=encode_xi0_payload(np.full(600, 31.0, dtype=np.float32)),
            source_manifest_hashes={"vjp": "a" * 64},
            output=root / "candidate.zip",
        )
    assert result["carrier_delta_bytes"] == 2_114
    assert result["excess_bytes"] == 262
    assert result["fixed_cost_excluding_xi0_compressed_bytes"] == 1_865
    assert result["minimum_non_xi0_reduction_even_if_xi0_free"] == 13


def test_direct_shift_body_is_intrinsic_and_highly_compressible() -> None:
    shifts = np.asarray([-11] * 440 + [-12] * 145 + [-10] * 13 + [-8, -13], dtype=np.int8)
    assert shifts.nbytes == 600
    assert _raw_deflate_size(shifts.tobytes()) < 120


def test_boundary_envelope_accounts_for_versioned_prefix() -> None:
    payload = _zero_boundary_payload()
    envelope = _section_envelope(payload, _BOUNDARY_PREFIX)
    assert envelope["body_bytes"] == 3_004
    assert envelope["total_bytes"] == len(payload)
