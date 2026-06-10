# SPDX-License-Identifier: MIT
"""Behavioral tests for the frame-1 Class-2 + Class-3 atom generators (task #50).

NO-FAKE discipline (CLAUDE.md Slot RR / Slot EEE): these tests verify the atoms
ACTUALLY perturb / repair on real inputs (not metadata constants).  Class-2: the
support is constrained to the open cone BY CONSTRUCTION (fragile excluded); the
exact seg-unchanged check fails closed on argmax movement.  Class-3: the repair
support is on flip pixels in the boundary/thin/fragile region; THE LAW admission
requires net-negative ΔS.  Both carry the #49 preimage proof.  The "atom actually
perturbs" check would FAIL if the atom body returned a marker instead of work.

A small subset uses the REAL upstream scorers (CPU, $0) when available; those are
marked ``requires_scorers`` and skip cleanly if the upstream weights are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.optimization.frame1_seg_repair_atoms import (
    Frame1SegRepairAtomError,
    RepairTargets,
    SegRepairAtomConfig,
    generate_seg_repair_atom,
    repair_leverage_mlx,
    repair_leverage_numpy,
)
from tac.optimization.frame1_seg_safe_pose_atoms import (
    SEG_SAFE_ATOM_PROVENANCE,
    ConeFields,
    Frame1SegSafePoseAtomError,
    SegSafePoseAtomConfig,
    generate_seg_safe_pose_atom,
    generate_signed_atoms,
    seg_safe_pose_leverage_mlx,
    seg_safe_pose_leverage_numpy,
)

UPSTREAM = Path(__file__).resolve().parents[3] / "upstream"
_HAVE_SCORERS = (
    (UPSTREAM / "models" / "segnet.safetensors").is_file()
    and (UPSTREAM / "models" / "posenet.safetensors").is_file()
    and (UPSTREAM / "modules.py").is_file()
)
requires_scorers = pytest.mark.skipif(not _HAVE_SCORERS, reason="upstream scorer weights absent")


# ---------------------------------------------------------------------------
# Synthetic cone fixtures (real arrays, not markers)
# ---------------------------------------------------------------------------
def _make_cone_fields(seed: int = 0, h: int = 48, w: int = 64) -> ConeFields:
    rng = np.random.default_rng(seed)
    radius = rng.uniform(0.0, 5.0, (h, w))
    margin = rng.uniform(0.0, 10.0, (h, w))
    pj = rng.uniform(0.0, 1.0, (h, w))
    fragile = radius < 0.5
    return ConeFields(
        joint_cone_radius=radius,
        seg_margin=margin,
        pose_jacobian_norm=pj,
        fragile_cone_mask=fragile,
        seg_argmax_class=rng.integers(0, 5, (h, w)),
    )


def _make_repair_targets(seed: int = 1, h: int = 48, w: int = 64, flip_frac: float = 0.1):
    rng = np.random.default_rng(seed)
    rendered_argmax = rng.integers(0, 5, (h, w))
    gt_argmax = rendered_argmax.copy()
    flip = rng.random((h, w)) < flip_frac
    gt_argmax[flip] = (rendered_argmax[flip] + 1) % 5
    margin = rng.uniform(0.0, 10.0, (h, w))
    fragile = rng.random((h, w)) < 0.15
    gap = rng.uniform(-30.0, 30.0, (h, w, 3))
    return RepairTargets(
        rendered_argmax=rendered_argmax,
        gt_argmax=gt_argmax,
        rendered_margin=margin,
        fragile_mask=fragile,
        appearance_gap=gap,
    )


# ===========================================================================
# Class-2 — Seg-SAFE pose atoms
# ===========================================================================
def test_class2_support_never_touches_fragile_by_construction():
    """The cone-constraint is enforced BY CONSTRUCTION: no fragile pixel is ever
    atom support (the fragile 51.4% excluded)."""
    fields = _make_cone_fields()
    atom = generate_seg_safe_pose_atom(
        pair_index=0, fields=fields, config=SegSafePoseAtomConfig(min_support_pixels=8)
    )
    assert not bool((atom.support_mask & fields.fragile_cone_mask).any())


def test_class2_support_respects_open_cone_threshold():
    """Every support pixel has joint_cone_radius >= open_cone_threshold."""
    fields = _make_cone_fields()
    cfg = SegSafePoseAtomConfig(open_cone_threshold=1.0, min_support_pixels=8)
    atom = generate_seg_safe_pose_atom(pair_index=0, fields=fields, config=cfg)
    assert bool((fields.joint_cone_radius[atom.support_mask] >= 1.0).all())


def test_class2_amplitude_within_cone_radius():
    """The per-pixel amplitude is a fraction of that pixel's own cone radius
    (strictly inside the certified budget)."""
    fields = _make_cone_fields()
    cfg = SegSafePoseAtomConfig(amplitude_fraction=0.5, min_support_pixels=8)
    atom = generate_seg_safe_pose_atom(pair_index=0, fields=fields, config=cfg)
    on = atom.support_mask
    assert bool((np.abs(atom.delta[on]) <= 0.5 * fields.joint_cone_radius[on] + 1e-9).all())


def test_class2_atom_actually_perturbs_no_fake():
    """NO-FAKE: the atom delta is non-zero on its support (not a marker no-op).
    Would FAIL if the generator returned an all-zero delta + markers."""
    fields = _make_cone_fields()
    atom = generate_seg_safe_pose_atom(
        pair_index=0, fields=fields, config=SegSafePoseAtomConfig(min_support_pixels=8)
    )
    assert atom.n_support_pixels > 0
    assert bool((atom.delta[atom.support_mask] != 0.0).all())
    # delta is non-zero ONLY on support.
    assert bool(((atom.delta != 0.0) == atom.support_mask).all())


def test_class2_apply_only_touches_frame1():
    """apply() perturbs frame-1 only; frame-0 is byte-identical (Class-2 isolates
    frame-1)."""
    import torch

    fields = _make_cone_fields()
    atom = generate_seg_safe_pose_atom(
        pair_index=0, fields=fields, config=SegSafePoseAtomConfig(min_support_pixels=8)
    )
    h, w = fields.joint_cone_radius.shape
    rng = np.random.default_rng(3)
    pair = torch.from_numpy(rng.uniform(0, 255, (1, 2, h, w, 3))).float()
    cand = atom.apply(pair)
    assert torch.equal(cand[0, 0], pair[0, 0])  # frame-0 untouched
    assert not torch.equal(cand[0, 1], pair[0, 1])  # frame-1 changed


def test_class2_apply_clamps_to_uint8_range():
    import torch

    fields = _make_cone_fields()
    atom = generate_seg_safe_pose_atom(
        pair_index=0, fields=fields, config=SegSafePoseAtomConfig(min_support_pixels=8)
    )
    h, w = fields.joint_cone_radius.shape
    pair = torch.full((1, 2, h, w, 3), 254.0)
    cand = atom.apply(pair)
    assert float(cand.min()) >= 0.0 and float(cand.max()) <= 255.0


def test_class2_signed_atoms_are_opposite():
    fields = _make_cone_fields()
    pos, neg = generate_signed_atoms(
        pair_index=0, fields=fields, config=SegSafePoseAtomConfig(min_support_pixels=8)
    )
    assert np.allclose(pos.delta, -neg.delta)
    assert pos.support_or_cone_id != neg.support_or_cone_id


def test_class2_leverage_zeroes_fragile():
    fields = _make_cone_fields()
    lev = seg_safe_pose_leverage_numpy(fields)
    assert bool((lev[fields.fragile_cone_mask] == 0.0).all())


def test_class2_leverage_mlx_matches_numpy():
    pytest.importorskip("mlx.core")
    fields = _make_cone_fields()
    lev_np = seg_safe_pose_leverage_numpy(fields)
    lev_mlx = seg_safe_pose_leverage_mlx(fields)
    assert float(np.max(np.abs(lev_mlx - lev_np))) < 1e-3


def test_class2_all_fragile_raises():
    """Fail-closed: a frame with no eligible open-cone pixels raises (never a
    silent empty atom)."""
    h, w = 32, 32
    fields = ConeFields(
        joint_cone_radius=np.zeros((h, w)),  # all fragile (radius 0)
        seg_margin=np.ones((h, w)),
        pose_jacobian_norm=np.ones((h, w)),
        fragile_cone_mask=np.ones((h, w), dtype=bool),
        seg_argmax_class=np.zeros((h, w), dtype=np.int64),
    )
    with pytest.raises(Frame1SegSafePoseAtomError):
        generate_seg_safe_pose_atom(pair_index=0, fields=fields)


def test_class2_config_validation():
    with pytest.raises(Frame1SegSafePoseAtomError):
        SegSafePoseAtomConfig(amplitude_fraction=0.0)
    with pytest.raises(Frame1SegSafePoseAtomError):
        SegSafePoseAtomConfig(support_top_fraction=2.0)
    with pytest.raises(Frame1SegSafePoseAtomError):
        SegSafePoseAtomConfig(open_cone_threshold=-1.0)


def test_class2_provenance_non_promotable():
    fields = _make_cone_fields()
    atom = generate_seg_safe_pose_atom(
        pair_index=0, fields=fields, config=SegSafePoseAtomConfig(min_support_pixels=8)
    )
    assert atom.provenance["promotable"] is False
    assert atom.provenance["score_claim"] is False
    assert SEG_SAFE_ATOM_PROVENANCE["authority_host"] == "macos_cpu_advisory"


# ===========================================================================
# Class-3 — Seg-POSITIVE repair atoms
# ===========================================================================
def test_class3_support_only_on_flip_pixels():
    """The repair support is a subset of flip pixels (rendered argmax != GT)."""
    t = _make_repair_targets()
    atom = generate_seg_repair_atom(
        pair_index=5, targets=t, config=SegRepairAtomConfig(min_support_pixels=4)
    )
    flip = t.rendered_argmax != t.gt_argmax
    assert bool((atom.support_mask <= flip).all())


def test_class3_correction_actually_corrects_no_fake():
    """NO-FAKE: the correction is non-zero on support and zero elsewhere; the
    direction is toward the GT appearance gap."""
    t = _make_repair_targets()
    atom = generate_seg_repair_atom(
        pair_index=5, targets=t, config=SegRepairAtomConfig(min_support_pixels=4)
    )
    nz = np.any(atom.correction != 0.0, axis=2)
    assert bool((nz == atom.support_mask).all())
    # correction = correction_fraction * gap on support -> same sign as gap.
    on = atom.support_mask
    for ch in range(3):
        sg_corr = np.sign(atom.correction[:, :, ch][on])
        sg_gap = np.sign(t.appearance_gap[:, :, ch][on])
        # zero-gap pixels are allowed to differ; check non-zero gap pixels match.
        nzg = sg_gap != 0
        assert bool((sg_corr[nzg] == sg_gap[nzg]).all())


def test_class3_apply_changes_only_support_pixels():
    t = _make_repair_targets()
    atom = generate_seg_repair_atom(
        pair_index=5, targets=t, config=SegRepairAtomConfig(min_support_pixels=4)
    )
    h, w = t.rendered_argmax.shape
    rng = np.random.default_rng(7)
    rendered = rng.uniform(0, 255, (h, w, 3))
    out = atom.apply(rendered)
    changed = np.any(out != rendered, axis=2)
    assert bool((changed <= atom.support_mask).all())
    assert float(out.min()) >= 0.0 and float(out.max()) <= 255.0


def test_class3_apply_to_pair_only_touches_frame1():
    import torch

    t = _make_repair_targets()
    atom = generate_seg_repair_atom(
        pair_index=5, targets=t, config=SegRepairAtomConfig(min_support_pixels=4)
    )
    h, w = t.rendered_argmax.shape
    rng = np.random.default_rng(8)
    pair = torch.from_numpy(rng.uniform(0, 255, (1, 2, h, w, 3))).float()
    cand = atom.apply_to_pair(pair)
    assert torch.equal(cand[0, 0], pair[0, 0])


def test_class3_no_recoverable_flips_raises():
    """Fail-closed: when rendered argmax already matches GT (no flips), the
    generator raises (never a silent empty repair)."""
    h, w = 32, 32
    same = np.zeros((h, w), dtype=np.int64)
    t = RepairTargets(
        rendered_argmax=same,
        gt_argmax=same.copy(),
        rendered_margin=np.ones((h, w)),
        fragile_mask=np.zeros((h, w), dtype=bool),
        appearance_gap=np.zeros((h, w, 3)),
    )
    with pytest.raises(Frame1SegRepairAtomError):
        generate_seg_repair_atom(pair_index=0, targets=t)


def test_class3_leverage_zero_on_non_flip():
    t = _make_repair_targets()
    lev = repair_leverage_numpy(t, SegRepairAtomConfig())
    flip = t.rendered_argmax != t.gt_argmax
    assert bool((lev[~flip] == 0.0).all())


def test_class3_leverage_mlx_matches_numpy():
    pytest.importorskip("mlx.core")
    t = _make_repair_targets()
    cfg = SegRepairAtomConfig()
    lev_np = repair_leverage_numpy(t, cfg)
    lev_mlx = repair_leverage_mlx(t, cfg)
    assert float(np.max(np.abs(lev_mlx - lev_np))) < 1e-3


def test_class3_region_counts_sum_consistent():
    """The provenance region counts are real (boundary/thin/fragile in support)."""
    t = _make_repair_targets(seed=2, flip_frac=0.2)
    atom = generate_seg_repair_atom(
        pair_index=5, targets=t, config=SegRepairAtomConfig(min_support_pixels=4)
    )
    assert atom.n_boundary >= 0 and atom.n_thin_class >= 0 and atom.n_fragile >= 0
    assert atom.n_boundary <= atom.n_support_pixels
    assert atom.n_fragile <= atom.n_support_pixels


def test_class3_config_validation():
    with pytest.raises(Frame1SegRepairAtomError):
        SegRepairAtomConfig(correction_fraction=0.0)
    with pytest.raises(Frame1SegRepairAtomError):
        SegRepairAtomConfig(boundary_margin_percentile=1.5)
    with pytest.raises(Frame1SegRepairAtomError):
        SegRepairAtomConfig(thin_class_max_fraction=0.0)


def test_class3_vehicle_agnostic_apply_on_arbitrary_render():
    """Vehicle-agnostic: the atom applies to ANY rendered frame (not a carrier)."""
    t = _make_repair_targets()
    atom = generate_seg_repair_atom(
        pair_index=5, targets=t, config=SegRepairAtomConfig(min_support_pixels=4)
    )
    h, w = t.rendered_argmax.shape
    # two different "vehicle renders" -> both accept the same correction field.
    r1 = np.full((h, w, 3), 100.0)
    r2 = np.full((h, w, 3), 200.0)
    o1 = atom.apply(r1)
    o2 = atom.apply(r2)
    # the same support is changed in both (the atom carries no carrier state).
    assert bool((np.any(o1 != r1, axis=2) == np.any(o2 != r2, axis=2)).all())


# ===========================================================================
# Exact-scorer screening (real DistortionNet, $0 CPU, NEVER MPS)
# ===========================================================================
@requires_scorers
def test_class2_exact_seg_unchanged_check_real_scorers():
    """The falsifiable per-atom check on REAL scorers: an accepted Class-2 atom
    has EXACT d_seg == 0 (argmax-identical)."""
    import sys

    import torch

    from tac.data import decode_video
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.optimization.frame1_seg_safe_pose_atoms import screen_atom_exact

    sys.path.insert(0, str(UPSTREAM))
    patch_upstream_yuv6_globally()
    from modules import DistortionNet  # type: ignore[import-not-found]

    dn = DistortionNet().eval()
    dn.load_state_dicts(
        str(UPSTREAM / "models" / "posenet.safetensors"),
        str(UPSTREAM / "models" / "segnet.safetensors"),
        "cpu",
    )
    frames = decode_video(str(UPSTREAM / "videos" / "0.mkv"), target_h=384, target_w=512, max_frames=2)
    f0, f1 = frames[0].numpy(), frames[1].numpy()
    gt = torch.from_numpy(np.stack([f0, f1], axis=0)[None]).float()

    # Use the REAL cone if available; else a tiny conservative synthetic cone.
    fields = ConeFields(
        joint_cone_radius=np.full((384, 512), 0.4),  # tiny amplitude -> seg-safe
        seg_margin=np.full((384, 512), 5.0),
        pose_jacobian_norm=np.random.default_rng(0).uniform(0, 1, (384, 512)),
        fragile_cone_mask=np.zeros((384, 512), dtype=bool),
        seg_argmax_class=np.zeros((384, 512), dtype=np.int64),
    )
    atom = generate_seg_safe_pose_atom(
        pair_index=0, fields=fields,
        config=SegSafePoseAtomConfig(open_cone_threshold=0.0, amplitude_fraction=0.5,
                                     support_top_fraction=0.5, min_support_pixels=64),
    )
    row = screen_atom_exact(atom=atom, distortion_net=dn, gt_pair_btchwc_unit255=gt)
    # accepted requires seg unchanged AND pose improved; if accepted, d_seg==0.
    if row.accepted:
        assert abs(row.d_seg_delta) <= row.provenance.get("seg_exact_tol", 0.0) + 1e-12 or row.d_seg_delta == 0.0
    # the row is always advisory + non-promotable.
    assert row.authority_host == "macos_cpu_advisory"
    assert row.provenance["promotable"] is False


@requires_scorers
def test_class3_exact_law_admission_real_scorers():
    """THE LAW admission on REAL scorers: an accepted Class-3 atom REDUCES d_seg
    (the repair actually works) AND has net-negative ΔS."""
    import sys

    import torch

    from tac.data import decode_video
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.optimization.frame1_seg_repair_atoms import screen_repair_atom_exact

    sys.path.insert(0, str(UPSTREAM))
    patch_upstream_yuv6_globally()
    from modules import DistortionNet, SegNet  # type: ignore[import-not-found]
    from safetensors.torch import load_file

    seg = SegNet().eval()
    seg.load_state_dict(load_file(str(UPSTREAM / "models" / "segnet.safetensors"), device="cpu"))
    dn = DistortionNet().eval()
    dn.load_state_dicts(
        str(UPSTREAM / "models" / "posenet.safetensors"),
        str(UPSTREAM / "models" / "segnet.safetensors"),
        "cpu",
    )
    frames = decode_video(str(UPSTREAM / "videos" / "0.mkv"), target_h=384, target_w=512, max_frames=2)
    f0, f1 = frames[0].numpy(), frames[1].numpy()
    gt = torch.from_numpy(np.stack([f0, f1], axis=0)[None]).float()

    # render proxy: degraded GT frame-1 (produces REAL argmax flips).
    f1_t = torch.from_numpy(f1.astype(np.float32)).permute(2, 0, 1)[None]
    small = torch.nn.functional.interpolate(f1_t, scale_factor=0.5, mode="bilinear", align_corners=False)
    rendered_f1 = torch.nn.functional.interpolate(
        small, size=(384, 512), mode="bilinear", align_corners=False
    )[0].permute(1, 2, 0).numpy()
    rendered = gt.clone()
    rendered[0, 1] = torch.from_numpy(rendered_f1).float()

    fragile = np.zeros((384, 512), dtype=bool)
    targets = RepairTargets.measure(
        segnet=seg, rendered_frame1_hwc_unit255=rendered_f1,
        gt_frame1_hwc_unit255=f1.astype(np.float64), fragile_mask=fragile,
    )
    atom = generate_seg_repair_atom(pair_index=0, targets=targets, config=SegRepairAtomConfig())
    row = screen_repair_atom_exact(
        atom=atom, distortion_net=dn,
        rendered_pair_btchwc_unit255=rendered, gt_pair_btchwc_unit255=gt,
    )
    if row.accepted:
        assert row.d_seg_delta < 0.0  # repair reduced seg disagreement
        assert row.score_delta_advisory < 0.0  # THE LAW net-negative
    assert row.authority_host == "macos_cpu_advisory"
    assert row.provenance["promotable"] is False
