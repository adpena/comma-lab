# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the Z8 joint P18/P19 dead-zone rate attack.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS": every test verifies ACTUAL behavior —
coefficients are really zeroed, the rate really drops, the masks are
non-trivial (not all-zero / not all-pass), the splice round-trips byte-identical
under identity, the joint surface uses REAL scorer saliencies, and the keep
priority really respects reconstruction energy. The archive is built from REAL
video frames (``upstream/videos/0.mkv``) per Catalog #213, NOT synthetic
fixtures. If the function body were replaced by a no-op, these tests FAIL.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tac.substrates.z8_hierarchical_predictive_coding.archive import (
    parse_z8hpc1_archive_bytes,
)
from tac.substrates.z8_hierarchical_predictive_coding.joint_p18_p19_deadzone_rate_attack import (
    apply_deadzone_to_pair_details,
    joint_deadzone_mask_for_pair,
    joint_keep_priority_for_pair,
    magnitude_deadzone_mask_for_pair,
    pack_pair_pyramids_to_wavelet_blob,
    parse_pair_blobs_from_wavelet_blob,
    splice_wavelet_blob_into_archive,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
VIDEO = REPO_ROOT / "upstream" / "videos" / "0.mkv"


def _real_z8_archive(num_pairs: int = 4):
    """Build a real 4-pair Z8HPC1 archive from real video frames (Catalog #213).

    NOT a synthetic fixture: the wavelet detail coefficients ARE the real
    Mallat DWT of the real dashcam frames.
    """

    from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
        build_canonical_quadruple_binding_from_z8_config,
        build_z8hpc1_archive_bytes_from_canonical_quadruple,
        load_real_video_pair_targets_numpy,
    )

    cfg = SimpleNamespace(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        num_pairs=num_pairs,
        deterministic_state_dim=16,
        ego_motion_dim=6,
        eval_size=(96, 128),
    )
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    f0, f1 = load_real_video_pair_targets_numpy(
        str(VIDEO), num_pairs=num_pairs, output_height=96, output_width=128
    )
    archive_bytes = build_z8hpc1_archive_bytes_from_canonical_quadruple(binding, f0, f1)
    return binding, np.asarray(f0), np.asarray(f1), archive_bytes


@pytest.fixture(scope="module")
def real_archive():
    if not VIDEO.is_file():
        pytest.skip(f"real video missing: {VIDEO}")
    return _real_z8_archive(num_pairs=4)


def _wavelet_pyramids(archive_bytes: bytes):
    secs = parse_z8hpc1_archive_bytes(archive_bytes)
    ws, wl = secs["wavelet_blob"]
    return parse_pair_blobs_from_wavelet_blob(archive_bytes[ws : ws + wl])


def _flat(details):
    parts = []
    for d in details:
        for k in ("lh", "hl", "hh"):
            parts.append(np.asarray(getattr(d, k), dtype=np.float64).reshape(-1))
    return np.concatenate(parts) if parts else np.zeros((0,))


# ---------------------------------------------------------------------------
# Masks are non-trivial + actually zero coefficients
# ---------------------------------------------------------------------------


def test_magnitude_mask_keeps_exact_fraction(real_archive):
    """The magnitude dead-zone mask zeros exactly (1 - keep_fraction) of atoms."""

    _b, _f0, _f1, archive = real_archive
    pyrs = _wavelet_pyramids(archive)
    details = pyrs[0]["frame_0_details"]
    mask = magnitude_deadzone_mask_for_pair(details, keep_fraction=0.30)
    n = mask.shape[0]
    # zero 70% (within 1 atom of exact for tie-breaks)
    assert abs(int(mask.sum()) - round(0.70 * n)) <= 1
    assert mask.any() and not mask.all()  # non-trivial


def test_deadzone_actually_zeros_coefficients(real_archive):
    """apply_deadzone ACTUALLY sets the masked coefficients to 0.0 (not a marker)."""

    _b, _f0, _f1, archive = real_archive
    pyrs = _wavelet_pyramids(archive)
    details = pyrs[0]["frame_0_details"]
    before = _flat(details)
    n_zero_before = int((before == 0.0).sum())
    mask = magnitude_deadzone_mask_for_pair(details, keep_fraction=0.30)
    new_details, n_total, _already = apply_deadzone_to_pair_details(details, mask)
    after = _flat(new_details)
    n_zero_after = int((after == 0.0).sum())
    # The masked atoms must ALL be exactly zero in the result.
    assert np.all(after[mask] == 0.0)
    # And strictly more zeros than before (the attack really happened).
    assert n_zero_after > n_zero_before
    assert n_total == before.shape[0]


def test_deadzone_keeps_largest_magnitude_coefficients(real_archive):
    """The magnitude mask keeps the top-magnitude atoms (RD-optimal) — the kept
    atoms' min magnitude >= the zeroed atoms' max magnitude (up to ties)."""

    _b, _f0, _f1, archive = real_archive
    pyrs = _wavelet_pyramids(archive)
    details = pyrs[0]["frame_0_details"]
    flat = _flat(details)
    mask = magnitude_deadzone_mask_for_pair(details, keep_fraction=0.30)
    kept = np.abs(flat[~mask])
    zeroed = np.abs(flat[mask])
    if kept.size and zeroed.size:
        assert kept.min() >= zeroed.max() - 1e-9


# ---------------------------------------------------------------------------
# Splice round-trips + rate really changes
# ---------------------------------------------------------------------------


def test_identity_splice_is_byte_identical(real_archive):
    """Parsing then re-packing then splicing an UNMODIFIED wavelet blob yields a
    parseable archive with an identical wavelet section."""

    _b, _f0, _f1, archive = real_archive
    secs = parse_z8hpc1_archive_bytes(archive)
    ws, wl = secs["wavelet_blob"]
    blob = archive[ws : ws + wl]
    pyrs = parse_pair_blobs_from_wavelet_blob(blob)
    blob2 = pack_pair_pyramids_to_wavelet_blob(pyrs)
    arc2 = splice_wavelet_blob_into_archive(archive, blob2)
    secs2 = parse_z8hpc1_archive_bytes(arc2)  # must parse
    ws2, wl2 = secs2["wavelet_blob"]
    assert blob == blob2  # deterministic re-pack
    assert arc2[ws2 : ws2 + wl2] == blob  # spliced section identical


def test_splice_preserves_non_wavelet_sections(real_archive):
    """Splicing a modified wavelet blob leaves every other section byte-identical."""

    _b, _f0, _f1, archive = real_archive
    secs = parse_z8hpc1_archive_bytes(archive)
    ws, wl = secs["wavelet_blob"]
    pyrs = parse_pair_blobs_from_wavelet_blob(archive[ws : ws + wl])
    # zero half the frame_0 details of pair 0
    details = pyrs[0]["frame_0_details"]
    mask = magnitude_deadzone_mask_for_pair(details, keep_fraction=0.50)
    new_details, _n, _a = apply_deadzone_to_pair_details(details, mask)
    pyrs[0]["frame_0_details"] = new_details
    new_blob = pack_pair_pyramids_to_wavelet_blob(pyrs)
    arc2 = splice_wavelet_blob_into_archive(archive, new_blob)
    secs2 = parse_z8hpc1_archive_bytes(arc2)
    for name in ("decoder_blob", "indices_blob", "wyner_ziv_blob", "dreamer_state_blob", "meta_blob"):
        s1, l1 = secs[name]
        s2, l2 = secs2[name]
        assert archive[s1 : s1 + l1] == arc2[s2 : s2 + l2], f"{name} changed"


def test_zeroing_coefficients_reduces_brotli_rate(real_archive):
    """Zeroing detail coefficients ACTUALLY shrinks the brotli-coded wavelet blob
    (the rate-binding section). This is the real rate attack, not a marker."""

    _b, _f0, _f1, archive = real_archive
    secs = parse_z8hpc1_archive_bytes(archive)
    ws, wl = secs["wavelet_blob"]
    pyrs = parse_pair_blobs_from_wavelet_blob(archive[ws : ws + wl])
    baseline_blob = pack_pair_pyramids_to_wavelet_blob(pyrs)
    for pyramid in pyrs:
        for frame_key in ("frame_0_details", "frame_1_details"):
            details = pyramid[frame_key]
            mask = magnitude_deadzone_mask_for_pair(details, keep_fraction=0.10)
            nd, _n, _a = apply_deadzone_to_pair_details(details, mask)
            pyramid[frame_key] = nd
    attacked_blob = pack_pair_pyramids_to_wavelet_blob(pyrs)
    # The attacked (mostly-zero) blob must be strictly SMALLER (brotli loves zeros).
    assert len(attacked_blob) < len(baseline_blob)


# ---------------------------------------------------------------------------
# Joint keep priority: RD-energy-aware + P19 pose protection
# ---------------------------------------------------------------------------


def test_keep_priority_respects_reconstruction_energy():
    """At zero protection gains the keep priority equals the coefficient
    magnitude (pure RD baseline) — the magnitude-only special case."""

    mag = np.array([5.0, 1.0, 3.0, 0.0])
    seg = np.array([0.0, 0.0, 0.0, 0.0])
    pose = np.array([0.0, 0.0, 0.0, 0.0])
    pose_null = np.ones(4, dtype=bool)
    prio = joint_keep_priority_for_pair(
        mag, seg, pose, pose_null, seg_protect_gain=0.0, pose_protect_gain=0.0
    )
    assert np.allclose(prio, np.abs(mag))


def test_pose_sensitive_coefficient_gets_protection_bonus():
    """A pose-SENSITIVE coefficient (not in null mask, high pose term) gets a
    higher keep priority than an identical-magnitude pose-null coefficient."""

    mag = np.array([2.0, 2.0])
    seg = np.array([0.0, 0.0])
    pose = np.array([1.0, 1.0])  # both have pose energy
    pose_null = np.array([True, False])  # atom 0 is pose-null, atom 1 is sensitive
    prio = joint_keep_priority_for_pair(
        mag, seg, pose, pose_null, seg_protect_gain=0.0, pose_protect_gain=1.0
    )
    # atom 1 (pose-sensitive) MUST be protected (higher keep priority).
    assert prio[1] > prio[0]


def test_seg_boundary_coefficient_gets_protection_bonus():
    """A seg-boundary coefficient (high seg term) gets a higher keep priority
    than an identical-magnitude seg-flat coefficient."""

    mag = np.array([2.0, 2.0])
    seg = np.array([0.0, 1.0])  # atom 1 is seg-boundary
    pose = np.array([0.0, 0.0])
    pose_null = np.ones(2, dtype=bool)
    prio = joint_keep_priority_for_pair(
        mag, seg, pose, pose_null, seg_protect_gain=1.0, pose_protect_gain=0.0
    )
    assert prio[1] > prio[0]


def test_joint_deadzone_mask_keeps_high_priority(real_archive):
    """The joint dead-zone mask keeps the top keep_fraction by priority and is
    non-trivial."""

    _b, _f0, _f1, archive = real_archive
    pyrs = _wavelet_pyramids(archive)
    details = pyrs[0]["frame_0_details"]
    flat = _flat(details)
    n = flat.shape[0]
    seg = np.zeros(n)
    pose = np.zeros(n)
    pose_null = np.ones(n, dtype=bool)
    prio = joint_keep_priority_for_pair(np.abs(flat), seg, pose, pose_null)
    mask = joint_deadzone_mask_for_pair(prio, keep_fraction=0.25)
    assert mask.any() and not mask.all()
    assert abs(int(mask.sum()) - round(0.75 * n)) <= 1


# ---------------------------------------------------------------------------
# Real scorer saliencies (P18 + P19) — these require torch + upstream
# ---------------------------------------------------------------------------


def test_real_segnet_boundary_saliency_is_nonconstant(real_archive):
    """The P18 SegNet boundary saliency from a REAL SegNet forward varies across
    pixels (high at class boundaries) — NOT a constant, NOT a stub."""

    torch = pytest.importorskip("torch")
    if not (REPO_ROOT / "upstream" / "models" / "segnet.safetensors").is_file():
        pytest.skip("segnet weights missing")
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.z8_hierarchical_predictive_coding.joint_p18_p19_deadzone_rate_attack import (
        segnet_boundary_pixel_saliency,
    )

    _b, f0, f1, _arc = real_archive
    pose, seg = load_differentiable_scorers(str(REPO_ROOT / "upstream"), device="cpu")
    gt = torch.from_numpy(np.stack([f0[0], f1[0]], axis=0)[None]).float()
    sal = segnet_boundary_pixel_saliency(pose, seg, gt)
    assert sal.shape == (384, 512)
    assert float(sal.std()) > 0.0  # not constant
    assert float(sal.min()) >= 0.0 and float(sal.max()) <= 1.0  # exp(-margin) in (0,1]


def test_real_posenet_pixel_jacobian_is_real_and_sparse(real_archive):
    """The P19 PoseNet pixel-Jacobian from a REAL differentiable PoseNet backward
    is non-zero and has a meaningful pose-null subset (sparse sensitivity)."""

    torch = pytest.importorskip("torch")
    if not (REPO_ROOT / "upstream" / "models" / "posenet.safetensors").is_file():
        pytest.skip("posenet weights missing")
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.z8_hierarchical_predictive_coding.joint_p18_p19_deadzone_rate_attack import (
        posenet_pixel_jacobian_norm,
    )

    _b, f0, f1, _arc = real_archive
    pose, seg = load_differentiable_scorers(str(REPO_ROOT / "upstream"), device="cpu")
    gt = torch.from_numpy(np.stack([f0[0], f1[0]], axis=0)[None]).float()
    jac = posenet_pixel_jacobian_norm(pose, seg, gt)
    assert jac.shape == (2, 96, 128)
    assert float(jac.max()) > 0.0  # real gradient flows
    # A non-trivial fraction of pixels are pose-null (below 5% of max).
    null_frac = float((jac < 0.05 * jac.max()).mean())
    assert 0.0 < null_frac < 1.0


def test_adjoint_dwt_push_preserves_shape(real_archive):
    """Pushing a per-pixel saliency through the analysis DWT yields per-level
    detail-coeff saliencies whose shapes match the stored detail coefficients
    (the adjoint maps pixel-space saliency onto the coefficient grid)."""

    _b, _f0, _f1, archive = real_archive
    binding = _b
    from tac.substrates.z8_hierarchical_predictive_coding.joint_p18_p19_deadzone_rate_attack import (
        push_pixel_saliency_to_detail_coeffs,
    )

    pyrs = _wavelet_pyramids(archive)
    details = pyrs[0]["frame_0_details"]
    saliency = np.random.RandomState(0).rand(96, 128)
    pushed = push_pixel_saliency_to_detail_coeffs(binding, saliency, num_levels=3)
    assert len(pushed) == len(details)
    for level_idx, detail in enumerate(details):
        for k in ("lh", "hl", "hh"):
            stored = np.asarray(getattr(detail, k))
            assert pushed[level_idx][k].shape == stored.shape


def test_apply_deadzone_rejects_mismatched_mask(real_archive):
    """apply_deadzone fails closed on a mask of the wrong length (no silent
    truncation)."""

    _b, _f0, _f1, archive = real_archive
    pyrs = _wavelet_pyramids(archive)
    details = pyrs[0]["frame_0_details"]
    bad = np.zeros(3, dtype=bool)
    with pytest.raises(ValueError):
        apply_deadzone_to_pair_details(details, bad)
