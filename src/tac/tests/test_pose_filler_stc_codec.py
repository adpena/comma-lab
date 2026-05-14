# SPDX-License-Identifier: MIT
"""Filler-STC pose codec regression tests.

Covers (per the Decision-4 council-alternative spec):

  (a) Round-trip on synthetic pose-delta sequences across multiple seeds.
  (b) Edge cases: zero-delta sequence, single-pose-pair, all-max-delta,
      alternating sign.
  (c) Rate vs PD-V2 — STC is *expected* to be within ±25% on smooth random
      walks; on idle-dominant traces PD-V2 wins on rate and FSTC trades
      that for channel-noise robustness. Both numbers are surfaced via
      ``[empirical:...]`` tags.
  (d) Corruption: decoder fails gracefully on bad magic, truncation,
      header-shape mismatch, payload bit-flip (parity-syndrome miss).
  (e) Determinism: same input → identical bytes across two encode calls.

Memory cross-ref:
    `.omx/research/grand_council_extreme_rigor_track_1_20260508.md` —
    Decision 4 ENDORSE-low-priority; Phase A4-alt Filler STC pose is the
    canonical alternative council recorded for the residual-coded pose
    head. lane_pd_v2 is the rate-optimal sibling; FSTC is the
    channel-noise-robust sibling.
"""
from __future__ import annotations

import struct

import pytest
import torch

from tac.codec.pose_filler_stc_codec import (
    FSTC_DEFAULT_CODE_LENGTH,
    FSTC_DEFAULT_CONSTRAINT_HEIGHT,
    FSTC_DEFAULT_PARITY_SEED,
    FSTC_MAGIC,
    FSTC_VERSION,
    FillerSTCPoseDecoder,
    FillerSTCPoseEncoder,
    FSTCParams,
    FSTCParityMismatchError,
)
from tac.pose_delta_codec_v2 import encode_pose_delta_v2


def _smooth_poses(n: int = 12, pose_dim: int = 6) -> torch.Tensor:
    """Deterministic closed-form smooth pose trajectory."""
    t = torch.linspace(0.0, 1.0, steps=n).unsqueeze(1)
    cols = []
    for idx in range(pose_dim):
        cols.append((idx + 1) * 0.01 * torch.sin((idx + 1) * t))
    return torch.cat(cols, dim=1).to(torch.float32)


def _smooth_random_walk(seed: int, n_pairs: int = 600, pose_dim: int = 6) -> torch.Tensor:
    """Random-walk smooth trajectory for the per-seed roundtrip suite and
    the rate-vs-PD-V2 fixture (mirrors the V2 test fixture)."""
    torch.manual_seed(seed)
    return (
        torch.cumsum(torch.randn(n_pairs, pose_dim) * 0.001, dim=0)
        + torch.randn(pose_dim) * 0.01
    )


def _idle_dominant_poses(seed: int = 42, n_pairs: int = 600, pose_dim: int = 6) -> torch.Tensor:
    """Idle-dominant trajectory (vehicle stationary ~70% of frames)."""
    torch.manual_seed(seed)
    deltas = torch.zeros(n_pairs - 1, pose_dim)
    move_mask = torch.rand(n_pairs - 1) < 0.3
    deltas[move_mask] = torch.randn(int(move_mask.sum()), pose_dim) * 0.01
    poses = torch.zeros(n_pairs, pose_dim)
    poses[0] = torch.randn(pose_dim) * 0.01
    poses[1:] = poses[0:1] + torch.cumsum(deltas, dim=0)
    return poses


def _plane_payload_offset(blob: bytes, pose_dim: int) -> tuple[int, int]:
    header_len = (
        4  # magic
        + 2  # version
        + 2  # pose_dim
        + 4  # n_pairs
        + 4  # sign bit count
        + 4  # plane bit count
        + 4  # parity bit count
        + 4  # parity seed
        + 1  # constraint height
        + 2  # code length
        + 1  # n_planes
        + pose_dim * 2  # anchor fp16
        + pose_dim * 2  # delta scale fp16
    )
    sign_len = struct.unpack("<I", blob[header_len : header_len + 4])[0]
    plane_len_offset = header_len + 4 + sign_len
    plane_len = struct.unpack("<I", blob[plane_len_offset : plane_len_offset + 4])[0]
    return plane_len_offset + 4, plane_len


# ---------------------------------------------------------------------------
# (a) Round-trip on canonical fixture + multiple seeds
# ---------------------------------------------------------------------------


def test_filler_stc_pose_codec_round_trips_smooth_pose_tensor() -> None:
    poses = _smooth_poses()
    blob = FillerSTCPoseEncoder().encode(poses)

    decoded = FillerSTCPoseDecoder().decode(blob, num_poses=poses.shape[0])

    assert blob.startswith(FSTC_MAGIC)
    assert decoded.shape == poses.shape
    assert float((poses - decoded).abs().max()) < 5e-2


@pytest.mark.parametrize("seed", list(range(10)))
def test_roundtrip_smooth_random_walk_10_seeds(seed: int) -> None:
    """FSTC must round-trip every smooth 600x6 random-walk trajectory
    within the quantizer's int8 floor (~1e-3 max-abs)."""
    poses = _smooth_random_walk(seed=seed)
    enc = FillerSTCPoseEncoder()
    dec = FillerSTCPoseDecoder()
    blob = enc.encode(poses)
    recovered = dec.decode(blob, num_poses=poses.shape[0])
    assert recovered.shape == poses.shape
    assert recovered.dtype == torch.float32
    err = (poses - recovered).abs().max().item()
    assert err < 1e-3, f"seed={seed}: max-abs round-trip err {err:.6e} > 1e-3 floor"


# ---------------------------------------------------------------------------
# (b) Edge cases
# ---------------------------------------------------------------------------


def test_edge_zero_delta_sequence_roundtrips() -> None:
    """Constant-pose stream: every delta is zero; encoder must produce a
    valid blob and the decoder must recover the constant within the fp16
    floor."""
    poses = torch.full((600, 6), 0.5, dtype=torch.float32)
    blob = FillerSTCPoseEncoder().encode(poses)
    recovered = FillerSTCPoseDecoder().decode(blob, num_poses=600)
    err = (poses - recovered).abs().max().item()
    # Anchor stored fp16 → ~5e-4 floor on the constant value.
    assert err < 1e-3, f"constant-pose round-trip err {err:.6e} > 1e-3"


def test_edge_single_pose_pair_roundtrips() -> None:
    """The minimum legal trajectory is N=2 (one delta)."""
    poses = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1, -0.05, 0.02, 0.03, -0.01, 0.04]],
        dtype=torch.float32,
    )
    blob = FillerSTCPoseEncoder().encode(poses)
    recovered = FillerSTCPoseDecoder().decode(blob, num_poses=2)
    err = (poses - recovered).abs().max().item()
    assert err < 1e-3, f"N=2 max-abs err {err:.6e} > 1e-3"


def test_edge_all_max_delta_roundtrips() -> None:
    """Saturated alternating-extreme delta stream: every quantized delta
    is +127 then -127. Tests that the magnitude bit-plane MSB is always 1
    while the sign alternates."""
    n = 200
    poses = torch.zeros(n, 6, dtype=torch.float32)
    for i in range(1, n):
        if i % 2 == 1:
            poses[i] = poses[i - 1] + 0.05
        else:
            poses[i] = poses[i - 1] - 0.05
    blob = FillerSTCPoseEncoder().encode(poses)
    recovered = FillerSTCPoseDecoder().decode(blob, num_poses=n)
    err = (poses - recovered).abs().max().item()
    assert err < 1e-3, f"alternating-extreme err {err:.6e} > 1e-3"


def test_edge_alternating_sign_recovers_signs_correctly() -> None:
    """Stream where every other delta has positive sign and the intervening
    deltas have negative sign; round-trip must preserve every sign — so
    sign-bit and magnitude bit-plane decoding must align."""
    n = 100
    base_signs = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(n - 1)])
    deltas = base_signs.unsqueeze(1) * 0.005 * torch.ones(n - 1, 6)
    poses = torch.zeros(n, 6, dtype=torch.float32)
    poses[1:] = torch.cumsum(deltas, dim=0)
    blob = FillerSTCPoseEncoder().encode(poses)
    recovered = FillerSTCPoseDecoder().decode(blob, num_poses=n)
    err = (poses - recovered).abs().max().item()
    assert err < 1e-3


# ---------------------------------------------------------------------------
# (c) Rate vs PD-V2 — empirical comparison surfaced for ops dashboards
# ---------------------------------------------------------------------------


def test_rate_vs_pd_v2_within_25_percent_smooth_random_walk() -> None:
    """[empirical:src/tac/tests/test_pose_filler_stc_codec.py]

    On a smooth random-walk pose trajectory FSTC's parity overhead is in
    rough parity with PD-V2's static-histogram freq table; in the smooth
    regime FSTC sometimes BEATS PD-V2 because PD-V2 cannot exploit the
    near-uniform symbol distribution (entropy ~7.2 bits/sym). We assert
    FSTC stays within ±25% of PD-V2 in the smooth regime.
    """
    poses = _smooth_random_walk(seed=42)
    fstc_blob = FillerSTCPoseEncoder().encode(poses)
    v2_blob = encode_pose_delta_v2(poses)
    fstc_bytes = len(fstc_blob)
    v2_bytes = len(v2_blob)
    ratio = fstc_bytes / v2_bytes
    print(
        f"\n[empirical:src/tac/tests/test_pose_filler_stc_codec.py] smooth "
        f"FSTC={fstc_bytes}B V2={v2_bytes}B ratio={ratio:.3f}"
    )
    assert 0.75 <= ratio <= 1.25, (
        f"FSTC bytes {fstc_bytes} vs PD-V2 {v2_bytes} ratio {ratio:.3f} outside ±25% — "
        "either PD-V2 changed (re-tune FSTC) or the parity overhead grew."
    )


def test_rate_vs_pd_v2_idle_dominant_documents_tradeoff() -> None:
    """[empirical:src/tac/tests/test_pose_filler_stc_codec.py]

    On idle-dominant traces PD-V2 wins (AC exploits qint=0 dominance) and
    FSTC is *expected* to be larger — its value is channel-noise robustness,
    not Shannon-bound rate. This test does NOT assert FSTC <= PD-V2; it
    asserts FSTC remains within 2× PD-V2 (graceful degradation, not a
    regression cliff) and surfaces the empirical ratio for the operator.
    """
    poses = _idle_dominant_poses(seed=42)
    fstc_blob = FillerSTCPoseEncoder().encode(poses)
    v2_blob = encode_pose_delta_v2(poses)
    fstc_bytes = len(fstc_blob)
    v2_bytes = len(v2_blob)
    ratio = fstc_bytes / v2_bytes
    print(
        f"\n[empirical:src/tac/tests/test_pose_filler_stc_codec.py] idle-dominant "
        f"FSTC={fstc_bytes}B V2={v2_bytes}B ratio={ratio:.3f} "
        f"(FSTC is expected to be larger — channel-noise robustness trade-off)"
    )
    assert ratio < 2.0, (
        f"FSTC {fstc_bytes}B is more than 2× PD-V2 {v2_bytes}B (ratio {ratio:.3f}) — "
        "graceful-degradation contract violated."
    )


# ---------------------------------------------------------------------------
# (d) Corruption / failure-graceful tests
# ---------------------------------------------------------------------------


def test_filler_stc_pose_decoder_rejects_num_pose_mismatch() -> None:
    poses = _smooth_poses()
    blob = FillerSTCPoseEncoder().encode(poses)

    with pytest.raises(ValueError, match="n_pairs"):
        FillerSTCPoseDecoder().decode(blob, num_poses=poses.shape[0] + 1)


def test_filler_stc_pose_decoder_detects_plane_payload_corruption() -> None:
    poses = _smooth_poses()
    blob = bytearray(FillerSTCPoseEncoder().encode(poses))
    plane_offset, plane_len = _plane_payload_offset(bytes(blob), poses.shape[1])
    assert plane_len > 0
    blob[plane_offset] ^= 0x01

    with pytest.raises(FSTCParityMismatchError):
        FillerSTCPoseDecoder().decode(bytes(blob), num_poses=poses.shape[0])


def test_filler_stc_pose_decoder_rejects_bad_magic_and_truncation() -> None:
    with pytest.raises(ValueError, match="bad magic"):
        FillerSTCPoseDecoder().decode(b"BAD!" + b"\x00" * 64, num_poses=2)
    with pytest.raises(ValueError, match="truncated version"):
        FillerSTCPoseDecoder().decode(FSTC_MAGIC + b"\x01", num_poses=2)


# ---------------------------------------------------------------------------
# (e) Determinism — same input → identical bytes across two encode calls
# ---------------------------------------------------------------------------


def test_determinism_same_input_same_bytes() -> None:
    poses = _smooth_random_walk(seed=42)
    enc = FillerSTCPoseEncoder()
    blob_a = enc.encode(poses)
    blob_b = enc.encode(poses)
    assert blob_a == blob_b, "FSTC encoder is non-deterministic on identical input"
    # Parameter object isolation: a fresh encoder must produce identical bytes.
    blob_c = FillerSTCPoseEncoder().encode(poses)
    assert blob_a == blob_c


# ---------------------------------------------------------------------------
# Wire-format & FSTCParams sanity
# ---------------------------------------------------------------------------


def test_wire_format_magic_and_version() -> None:
    poses = _smooth_random_walk(seed=4)
    blob = FillerSTCPoseEncoder().encode(poses)
    assert blob[:4] == FSTC_MAGIC
    version = int.from_bytes(blob[4:6], "little")
    assert version == FSTC_VERSION


def test_fstc_params_validates_constraint_height() -> None:
    with pytest.raises(ValueError, match="constraint_height"):
        FSTCParams(constraint_height=0)
    with pytest.raises(ValueError, match="constraint_height"):
        FSTCParams(constraint_height=9)


def test_fstc_params_validates_code_length_vs_height() -> None:
    with pytest.raises(ValueError, match="code_length"):
        FSTCParams(constraint_height=8, code_length=4)


def test_default_params_match_documented_constants() -> None:
    """If the defaults change without a memory entry, this test surfaces
    the drift so the wire-format contract stays explicit."""
    p = FSTCParams()
    assert p.constraint_height == FSTC_DEFAULT_CONSTRAINT_HEIGHT
    assert p.code_length == FSTC_DEFAULT_CODE_LENGTH
    assert p.parity_seed == FSTC_DEFAULT_PARITY_SEED
    assert p.n_planes == 7


def test_round_trip_with_non_default_params() -> None:
    """A larger constraint height and code-length combination (h=4, n=64)
    must still round-trip; this exercises the parity-matrix + bit-plane
    bookkeeping in the off-default regime."""
    poses = _smooth_random_walk(seed=5)
    params = FSTCParams(constraint_height=4, code_length=64, parity_seed=99)
    enc = FillerSTCPoseEncoder(params)
    dec = FillerSTCPoseDecoder()
    blob = enc.encode(poses)
    recovered = dec.decode(blob, num_poses=600)
    err = (poses - recovered).abs().max().item()
    assert err < 1e-3
