# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the SPD-cone / Hilbert-projective water-filled pose-section codec
(sister of the #140 low-rank codec). Every size + MSE is RE-DERIVED from the actual
encode/round-trip on REAL or explicitly-synthetic data — nothing is an asserted constant
(Catalog #304 empirical bit-spend). The load-bearing claims:

1. ROUND-TRIP: decode(encode) reconstructs to the fit MSE; the codec is deterministic.
2. THE MEASURED WIN (real pose): SPD is a Pareto rate cut vs the #140 low-rank codec's
   SHIPPED default — fewer bytes at ≤ the same MSE (so contest d_pose cannot worsen).
3. THE MECHANISM (positive control): on a strongly-ANISOTROPIC covariance the SPD
   frontier beats the low-rank frontier by a large margin (water-filling reallocates
   bits that constant-levels wastes).
4. THE HONEST BOUND (isotropic control): on a near-ISOTROPIC covariance the frontiers
   nearly COINCIDE — the SPD advantage is small (it is a rate-distortion allocation
   improvement, not free bytes from nothing).
5. ARCHIVE round-trip: an archive built with pose_codec='spd' parses back through the
   SAME public inflate path (magic auto-dispatch) → the codec choice is invisible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tac.torch_vehicle.pose_film import build_archive_with_pose, parse_pose_section
from tac.torch_vehicle.pose_film import (
    lowrank_pose_section_fidelity,
)
from tac.torch_vehicle.pose_spd_codec import (
    _SPD_MAX_LEVELS,
    decode_pose_section_spd,
    encode_pose_section_spd,
    hilbert_projective_distance,
    spd_fit_to_bytes,
    spd_fit_to_mse,
    spd_pose_section_fidelity,
)

_GT_CACHE = Path("experiments/results/capstone_gt_targets_cache/gt_targets_n600.pt")


def _real_gt_pose() -> torch.Tensor:
    if not _GT_CACHE.exists():
        pytest.skip(f"GT cache not present at {_GT_CACHE}")
    d = torch.load(_GT_CACHE, map_location="cpu", weights_only=False)
    return d["pose"].float()


def _synth(ratio: float, seed: int = 7, n: int = 600) -> torch.Tensor:
    """A ``(n, 6)`` pose whose covariance has geometric spectrum with condition ``ratio``
    (ratio=1 → isotropic; large → anisotropic). Smooth temporal modes + random
    orthonormal mixing (matches the real pose's temporal smoothness)."""
    g = torch.Generator().manual_seed(seed)
    tt = torch.linspace(0, 12.0, n)
    m = torch.stack([torch.sin(tt * (k + 1) * 0.5 + k) for k in range(6)], dim=1)
    m = m - m.mean(0, keepdim=True)
    m = m / m.std(0, keepdim=True)
    q, _ = torch.linalg.qr(torch.randn(6, 6, generator=g))
    stds = torch.tensor([ratio ** (k / 5.0) for k in range(6)][::-1], dtype=torch.float32)
    return (m * stds[None, :]) @ q


def _base_frontier(pose: torch.Tensor) -> list[tuple[int, float]]:
    pts = []
    for r in range(1, 7):
        for lv in [3, 7, 15, 31, 63, 127, 254, 511, 1023, 2047, 4095, 8191, 16383, 32767]:
            try:
                pts.append(lowrank_pose_section_fidelity(pose, rank=r, levels=lv))
            except Exception:
                pass
    return pts


def _spd_frontier(pose: torch.Tensor) -> list[tuple[int, float]]:
    pts = []
    for e in np.linspace(-9.0, 2.0, 80):
        try:
            pts.append(spd_pose_section_fidelity(pose, water_level=float(10.0**e)))
        except Exception:
            pass
    return pts


def _min_bytes_at(frontier, mse_budget: float):
    ok = [b for b, m in frontier if m <= mse_budget * 1.00001]
    return min(ok) if ok else None


# ---------------------------------------------------------------------------
# 1. round-trip fidelity + determinism
# ---------------------------------------------------------------------------
def test_roundtrip_matches_fit_mse_and_is_deterministic():
    pose = _synth(64.0)
    theta, nbytes, mse = spd_fit_to_mse(pose, 1e-3)
    sec1 = encode_pose_section_spd(pose, water_level=theta)
    sec2 = encode_pose_section_spd(pose, water_level=theta)
    assert sec1 == sec2, "encode must be deterministic (same bytes)"
    rec = decode_pose_section_spd(sec1)
    rt = float(((rec - pose) ** 2).mean())
    assert abs(rt - mse) < 1e-9, f"round-trip mse {rt} != fit mse {mse}"
    assert len(sec1) == nbytes


def test_all_modes_dropped_degenerate_reconstructs_mean():
    pose = _synth(1.0)
    # Huge water level → every mode below θ → rank 0 → reconstruct is the per-dim mean.
    sec = encode_pose_section_spd(pose, water_level=1e9)
    rec = decode_pose_section_spd(sec)
    mu = pose.mean(0, keepdim=True).expand_as(pose)
    assert torch.allclose(rec, mu, atol=1e-5)


def test_levels_guard_and_hilbert_distance_monotone():
    pose = _synth(64.0)
    # very small θ pushes levels toward the cap but must never overflow the uint16 zigzag
    sec = encode_pose_section_spd(pose, water_level=1e-9)
    rec = decode_pose_section_spd(sec)
    assert torch.isfinite(rec).all()
    # Hilbert distance grows with anisotropy
    def dH(P):
        x = (P - P.mean(0, keepdim=True)).numpy()
        e = np.sort(np.linalg.eigvalsh(x.T @ x / (x.shape[0] - 1)))[::-1]
        return hilbert_projective_distance(e)
    assert dH(_synth(64.0)) > dH(_synth(2.0)) > dH(_synth(1.0))
    assert _SPD_MAX_LEVELS == 32767


# ---------------------------------------------------------------------------
# 2. THE MEASURED WIN on the REAL pose (Pareto vs the #140 shipped default)
# ---------------------------------------------------------------------------
def test_spd_pareto_beats_lowrank_default_on_real_pose():
    pose = _real_gt_pose()
    base_bytes, base_mse = lowrank_pose_section_fidelity(pose, rank=4, levels=511)
    # matched-MSE: SPD must reach ≤ base_mse in FEWER bytes (a strict rate win at no
    # worse fidelity → contest d_pose cannot worsen).
    _theta, spd_bytes, spd_mse = spd_fit_to_mse(pose, base_mse)
    assert spd_mse <= base_mse * 1.0001, f"SPD did not meet the MSE budget ({spd_mse} > {base_mse})"
    assert spd_bytes < base_bytes, (
        f"SPD not smaller at matched MSE: spd={spd_bytes} base={base_bytes}"
    )
    # It should be a MEANINGFUL cut on this near-rank-1 pose (measured ~27%); assert a
    # conservative floor so a regression fails but the exact number is not frozen.
    assert (base_bytes - spd_bytes) / base_bytes > 0.15


def test_spd_matched_bytes_gives_lower_mse_on_real_pose():
    pose = _real_gt_pose()
    base_bytes, base_mse = lowrank_pose_section_fidelity(pose, rank=4, levels=511)
    _theta, spd_bytes, spd_mse = spd_fit_to_bytes(pose, base_bytes)
    assert spd_bytes <= base_bytes
    assert spd_mse < base_mse, f"SPD not lower-MSE at matched bytes: spd={spd_mse} base={base_mse}"


# ---------------------------------------------------------------------------
# 3 + 4. mechanism: frontier gap scales with anisotropy (positive + parity controls)
# ---------------------------------------------------------------------------
def test_frontier_gap_large_when_anisotropic_small_when_isotropic():
    ref_mse = 2.7e-5

    def gap(pose):
        b = _min_bytes_at(_base_frontier(pose), ref_mse)
        s = _min_bytes_at(_spd_frontier(pose), ref_mse)
        assert b is not None and s is not None
        return b - s

    aniso_gap = gap(_synth(64.0))   # d_H ~ 8
    iso_gap = gap(_synth(1.0))      # d_H ~ 0.26
    # Positive control: SPD frontier meaningfully dominates on anisotropic data.
    assert aniso_gap > 200, f"anisotropic frontier gap too small: {aniso_gap}"
    # Parity control: on near-isotropic data the frontiers nearly coincide (small gap).
    assert iso_gap < aniso_gap, f"isotropic gap {iso_gap} not < anisotropic gap {aniso_gap}"
    assert iso_gap < 400, f"isotropic frontier gap unexpectedly large: {iso_gap}"


def test_spd_frontier_never_badly_dominated_by_lowrank_on_real_pose():
    # Across the near-lossless MSE ladder, SPD's frontier is >= the baseline's (within a
    # few bytes) at every budget — i.e. SPD never loses materially.
    pose = _real_gt_pose()
    bf = _base_frontier(pose)
    sf = _spd_frontier(pose)
    for mb in [3e-6, 1e-5, 2.7e-5, 5e-5, 1e-4]:
        b = _min_bytes_at(bf, mb)
        s = _min_bytes_at(sf, mb)
        assert b is not None and s is not None
        assert s <= b + 8, f"SPD lost at mse<={mb}: spd={s} base={b}"


# ---------------------------------------------------------------------------
# 5. archive build/parse round-trip through the public inflate path
# ---------------------------------------------------------------------------
def _fake_vendored():
    """Minimal stand-in for the vendored 3-section archive grammar
    ([len][blob] x 3) so we can test the additive pose-section append/parse without
    the full PR95 codec. Matches what parse_pose_section walks."""
    import struct as _struct

    def build(decoder_state_dict, latents, meta_dict):
        out = b""
        for blob in (b"DEC", b"LAT", b"META"):
            out += _struct.pack("<I", len(blob)) + blob
        return out

    def parse(archive_bytes):
        return None  # unused by the pose-section test (magic auto-dispatch handles it)

    return build, parse


def test_archive_spd_pose_section_parses_back():
    pose = _synth(64.0)
    build, parse = _fake_vendored()
    arch = build_archive_with_pose(
        build, {}, torch.zeros(1), {}, pose, pose_codec="spd"
    )
    rec = parse_pose_section(arch, parse)
    assert rec is not None, "SPD pose section did not parse back (magic dispatch failed)"
    # default (auto-fit θ) is Pareto vs the legacy iid codec → MSE ≤ iid's own MSE.
    from tac.torch_vehicle.pose_film import decode_pose_section, encode_pose_section

    iid_mse = float(((decode_pose_section(encode_pose_section(pose)) - pose) ** 2).mean())
    mse = float(((rec - pose) ** 2).mean())
    assert mse <= iid_mse * 1.02, f"SPD default MSE {mse} not ≤ iid MSE {iid_mse}"


def test_archive_spd_smaller_than_iid_on_real_pose():
    pose = _real_gt_pose()
    build, parse = _fake_vendored()
    a_iid = build_archive_with_pose(build, {}, torch.zeros(1), {}, pose, pose_codec="iid")
    a_spd = build_archive_with_pose(build, {}, torch.zeros(1), {}, pose, pose_codec="spd")
    # both parse back
    assert parse_pose_section(a_iid, parse) is not None
    assert parse_pose_section(a_spd, parse) is not None
    # SPD default auto-fits to ≤ iid MSE → must be no larger (measured: smaller)
    assert len(a_spd) < len(a_iid), f"spd archive {len(a_spd)} not < iid {len(a_iid)}"


def test_unknown_pose_codec_raises():
    pose = _synth(1.0)
    build, _ = _fake_vendored()
    with pytest.raises(ValueError, match="unknown pose_codec"):
        build_archive_with_pose(build, {}, torch.zeros(1), {}, pose, pose_codec="nope")
