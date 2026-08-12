from __future__ import annotations

import math

import numpy as np

from experiments import ddm_pz4a_precision_preproof as pz4a
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def test_quantize_signed_depth12_is_identity_and_depths_are_real() -> None:
    values = np.array(
        [[-2048, -2047, -1025, -3, -2, -1, 0, 1, 2, 3, 1023, 2047]],
        dtype=np.int16,
    )
    values = np.repeat(values, pz4a.N, axis=0)
    assert np.array_equal(pz4a.quantize_signed(values, 12), values)
    depth4 = pz4a.quantize_signed(values, 4)
    assert np.count_nonzero(depth4 != values) > 0
    assert depth4.min() >= -2048
    assert depth4.max() <= 2047


def test_reverse_waterfill_removes_real_precision_bits_and_nests() -> None:
    codes = np.zeros((pz4a.N, pz4a.D), dtype=np.int16)
    codes[:, 0] = np.arange(pz4a.N, dtype=np.int16) % 127
    codes[:, 1] = -(np.arange(pz4a.N, dtype=np.int16) % 91)
    jacobian = np.zeros((pz4a.N, pz4a.POSE_DIMS, pz4a.D), dtype=np.float64)
    jacobian[:, 0, 0] = 0.02
    jacobian[:, 1, 1] = 0.01
    active = np.tile(np.array([[0, 1, 2]], dtype=np.int8), (pz4a.N, 1))

    rows, arrays = pz4a.reverse_waterfill(
        codes,
        jacobian,
        active,
        tolerances=(0.05, 0.10),
    )

    tight = arrays[f"depth_map__{rows[0]['candidate_id']}"]
    loose = arrays[f"depth_map__{rows[1]['candidate_id']}"]
    assert rows[0]["predicted_induced_pose_contribution"] <= 0.05
    assert rows[1]["predicted_induced_pose_contribution"] <= 0.10
    assert np.all(loose <= tight)
    assert np.count_nonzero(loose < tight) > 0
    assert np.count_nonzero(tight < pz4a.MAX_DEPTH) > 0
    assert rows[1]["precision_bits_removed_from_depth12"] >= rows[0]["precision_bits_removed_from_depth12"]


def test_depth_metadata_real_coders_and_wire_roundtrip() -> None:
    depths = np.full((pz4a.N, pz4a.D), 12, dtype=np.uint8)
    depths[::3, 1::2] = 7
    depths[::5, 0::3] = 3
    payloads = pz4a.depth_coder_payloads(depths)

    assert pz4a.unpack_depth_nibbles(payloads["raw"]).shape == (pz4a.N, pz4a.D)
    assert np.array_equal(pz4a.unpack_depth_nibbles(payloads["raw"]), depths)
    assert payloads["selected_name"] in {"raw", "brotli_q11", "lzma1"}
    assert len(payloads["wire"]) > pz4a.DEPTH_HEADER.size


def test_real_shipped_cpr1_baseline_and_joint_cell_reproduce() -> None:
    pz4a.verify_inputs()
    codec = pz4a.import_codec()
    carrier = pz4a.CARRIER.read_bytes()
    state = codec.decode_compact_carrier(
        carrier,
        basis_count=pz4a.D * 3 * 24 * 32,
        frames=pz4a.N,
        dimensions=pz4a.D,
    )
    assert codec.encode_compact_carrier(*state) == carrier
    codes = pz4a.signed_codes_from_encoded(state[3])
    assert np.array_equal(codes, np.load(pz4a.COEFFICIENTS, allow_pickle=False))

    component = pz4a.split_coefficient_component(carrier, codec)
    coder = pz4a.coefficient_coder_stats(carrier, codec)
    assert len(component) == 4 + pz4a.D * 4 + pz4a.D + math.ceil(coder["rice_bits"] / 8)
    assert sum(coder["rice_bits_by_dimension"]) == coder["rice_bits"]
    assert all(0 <= value < pz4a.MAX_DEPTH for value in coder["rice_k_by_dimension"])

    shipped, decoded = pz4a.parse_archive_carrier_stream(pz4a.ARCHIVE)
    assert decoded == carrier
    assert pz4a.brotli.compress(carrier, quality=9) == shipped


def test_error_contribution_uses_full_six_by_twelve_jacobian() -> None:
    jacobian = np.zeros((pz4a.N, pz4a.POSE_DIMS, pz4a.D), dtype=np.float64)
    errors = np.zeros((pz4a.N, pz4a.D), dtype=np.int64)
    jacobian[0, :, 0] = 2.0
    errors[0, 0] = 3
    expected_d_pose = 6 * 36 / (pz4a.N * pz4a.POSE_DIMS)
    assert math.isclose(
        pz4a.contribution(jacobian, errors),
        math.sqrt(10 * expected_d_pose),
        rel_tol=0,
        abs_tol=1e-15,
    )


def test_runner_passes_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=pz4a.REPO,
        strict=False,
        roots=("experiments/ddm_pz4a_precision_preproof.py",),
    )
    assert findings == []
