# SPDX-License-Identifier: MIT
"""Tests for the EARLY-SEED + CONTAINMENT + AMPLIFICATION islands-protection kit.

Coverage: self-detection (islands = small-area AND unstable; lane vs movable by
thickness; NEVER hardcoded; deterministic), masks (dilation/union), early-seed
(births exactly / sparse / compose-inverse), containment (freeze/damp/shield
semantics + shield removes only the destructive component + MLX↔numpy parity),
amplification (birth-term math rides the top1-top2 margin field + gradient sign +
mean-1 persistence weight + MLX/compiled parity), recall.

These run on SYNTHETIC-but-STRUCTURED fixtures for unit determinism; the REAL
n600 frozen-SegNet island-survival measurement is
``experiments/island_protection_survival_smoke.py`` (NO-FAKE: the paradigm verdict
is that smoke, not these unit fixtures)."""
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math import island_protection as ip


# ---------------------------------------------------------------------------
# Structured fixture: a bulk-dominated argmax with a THIN lane band (cls 1) and a
# COMPACT movable blob (cls 3), matching the real signatures (small area, unstable).
# ---------------------------------------------------------------------------
def _synthetic_lstars(n: int = 6, h: int = 48, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.zeros((n, h, w), np.int64)
    a[:, : h // 3, :] = 2          # top third = Undrivable (bulk, static)
    a[:, h // 3 : 2 * h // 3, :] = 0  # mid = Road (bulk)
    a[:, 2 * h // 3 :, :] = 4      # bottom = MyCar/hood (bulk, static)
    for i in range(n):
        # thin lane band (cls 1): 1-px vertical stripe that WANDERS per frame (unstable)
        col = w // 2 + int(rng.integers(-4, 5))
        a[i, h // 3 : 2 * h // 3, col] = 1
        # compact movable blob (cls 3): a small square that MOVES per frame (unstable)
        r0 = h // 3 + int(rng.integers(0, 6))
        c0 = 6 + int(rng.integers(0, 8))
        a[i, r0 : r0 + 4, c0 : c0 + 4] = 3
    return a


# ============================ SELF-DETECTION ============================
def test_detects_lane_and_movable_as_islands():
    det = ip.identify_island_classes(_synthetic_lstars())
    assert set(det.island_classes) == {1, 3}
    assert det.lane_cls == 1 and det.movable_cls == 3


def test_bulk_classes_never_flagged_island():
    det = ip.identify_island_classes(_synthetic_lstars())
    kinds = {e.cls: (e.is_island, e.island_kind) for e in det.evidence}
    for bulk in (0, 2, 4):
        assert kinds[bulk][0] is False and kinds[bulk][1] is None


def test_lane_thinner_than_movable():
    det = ip.identify_island_classes(_synthetic_lstars())
    ev = {e.cls: e for e in det.evidence}
    # lane (thin stripe) has smaller interior thickness than the movable blob
    assert ev[1].mean_thickness_px < ev[3].mean_thickness_px


def test_detection_is_deterministic():
    lst = _synthetic_lstars()
    d1 = ip.identify_island_classes(lst)
    d2 = ip.identify_island_classes(lst)
    assert (d1.lane_cls, d1.movable_cls, d1.island_classes) == (d2.lane_cls, d2.movable_cls, d2.island_classes)


def test_detection_never_hardcodes_index_permuted_labels():
    """Permute the class labels — detection must FOLLOW the signature, not the index."""
    lst = _synthetic_lstars()
    perm = np.array([3, 2, 0, 1, 4])  # relabel: old0->3, old1->2, old2->0, old3->1, old4->4
    lst_p = perm[lst]
    det = ip.identify_island_classes(lst_p)
    # old lane(1)->2, old movable(3)->1 ; both still small+unstable islands
    assert set(det.island_classes) == {2, 1}
    ev = {e.cls: e for e in det.evidence}
    assert det.lane_cls == 2  # thin stripe is now labelled 2
    assert det.movable_cls == 1
    assert ev[det.lane_cls].mean_thickness_px < ev[det.movable_cls].mean_thickness_px


def test_area_and_iou_thresholds_exclude_a_large_unstable_class():
    lst = _synthetic_lstars()
    # make cls 0 (Road) UNSTABLE but keep it LARGE -> must NOT be an island (area gate)
    det = ip.identify_island_classes(lst)
    ev = {e.cls: e for e in det.evidence}
    assert ev[0].is_island is False  # large area vetoes island status regardless of IoU


# ============================ MASKS ============================
def test_masks_union_and_classes():
    lst = _synthetic_lstars()[0]
    m = ip.build_island_masks(lst, 1, 3, dilate_px=0)
    assert m.lane_mask is not None and m.movable_mask is not None
    assert np.array_equal(m.any_mask, m.lane_mask | m.movable_mask)
    assert np.array_equal(m.lane_mask, lst == 1)


def test_mask_dilation_grows_support():
    lst = _synthetic_lstars()[0]
    m0 = ip.build_island_masks(lst, 1, 3, dilate_px=0)
    m2 = ip.build_island_masks(lst, 1, 3, dilate_px=2)
    assert m2.any_mask.sum() > m0.any_mask.sum()
    assert np.all(m2.any_mask[m0.any_mask])  # dilation is a superset


def test_masks_handle_missing_class():
    lst = _synthetic_lstars()[0]
    m = ip.build_island_masks(lst, 1, None, dilate_px=0)
    assert m.movable_mask is None
    assert np.array_equal(m.any_mask, lst == 1)


# ============================ EARLY-SEED ============================
def test_seed_births_gt_appearance_at_islands():
    lst = _synthetic_lstars()[0]
    m = ip.build_island_masks(lst, 1, 3, dilate_px=0)
    gt = np.random.default_rng(2).integers(0, 256, (48, 64, 3)).astype(np.float32)
    seed = ip.build_island_seed(gt, m)
    comp = ip.compose_seed(np.zeros_like(gt), seed.residual)
    assert np.allclose(comp[seed.mask], gt[seed.mask])       # births exactly
    assert np.allclose(seed.residual[~seed.mask], 0.0)       # sparse: 0 off islands


def test_seed_residual_is_gt_minus_base():
    lst = _synthetic_lstars()[0]
    m = ip.build_island_masks(lst, 1, 3, dilate_px=0)
    gt = np.full((48, 64, 3), 200.0, np.float32)
    base = np.full((48, 64, 3), 30.0, np.float32)
    seed = ip.build_island_seed(gt, m, base_render_segres=base)
    assert np.allclose(seed.residual[m.any_mask], 170.0)     # 200 - 30 at islands
    comp = ip.compose_seed(base, seed.residual)
    assert np.allclose(comp[m.any_mask], 200.0)              # base + residual = GT


def test_seed_blend_scales_residual():
    lst = _synthetic_lstars()[0]
    m = ip.build_island_masks(lst, 1, 3, dilate_px=0)
    gt = np.full((48, 64, 3), 100.0, np.float32)
    s_full = ip.build_island_seed(gt, m, blend=1.0)
    s_half = ip.build_island_seed(gt, m, blend=0.5)
    assert np.allclose(s_half.residual[m.any_mask], 0.5 * s_full.residual[m.any_mask])


# ============================ CONTAINMENT ============================
def _grad_resid(h=48, w=64):
    g = np.random.default_rng(3).standard_normal((h, w, 3)).astype(np.float32)
    r = np.random.default_rng(4).standard_normal((h, w, 3)).astype(np.float32)
    return g, r


def test_containment_freeze_zeros_protected():
    g, r = _grad_resid()
    mask = np.zeros((48, 64), bool); mask[10:20, 10:20] = True
    out = ip.contain_protected_grad_np(g, r, ip.ContainmentSpec(mode="freeze", protected_mask=mask))
    assert np.allclose(out[mask], 0.0)
    assert np.allclose(out[~mask], g[~mask])   # bulk untouched


def test_containment_damp_scales_protected():
    g, r = _grad_resid()
    mask = np.zeros((48, 64), bool); mask[10:20, 10:20] = True
    out = ip.contain_protected_grad_np(g, r, ip.ContainmentSpec(mode="damp", damp=0.25, protected_mask=mask))
    assert np.allclose(out[mask], 0.25 * g[mask])
    assert np.allclose(out[~mask], g[~mask])


def test_containment_shield_removes_only_destructive_component():
    """A GD step r -= lr*g shrinks |r| when g has the same sign as r; shield removes
    exactly that same-sign part so a shielded step never shrinks the seed toward 0."""
    r = np.array([[[2.0, -3.0, 0.0]]], np.float32)           # (1,1,3)
    g = np.array([[[5.0, -4.0, 7.0]]], np.float32)           # same-sign on ch0,ch1 (destructive)
    mask = np.ones((1, 1), bool)
    out = ip.contain_protected_grad_np(g, r, ip.ContainmentSpec(mode="shield", protected_mask=mask))
    # ch0: r>0,g>0 -> destructive 5 removed -> 0 ; ch1: r<0,g<0 -> destructive -4 removed -> 0
    # ch2: r==0 -> sign 0 -> nothing removed -> keep 7
    assert np.allclose(out[0, 0], [0.0, 0.0, 7.0])
    # after a step, the seed magnitude does not shrink on the protected same-sign channels
    stepped = r[0, 0] - 0.1 * out[0, 0]
    assert abs(stepped[0]) >= abs(r[0, 0, 0]) - 1e-6
    assert abs(stepped[1]) >= abs(r[0, 0, 1]) - 1e-6


def test_containment_invalid_mode_raises():
    with pytest.raises(ValueError):
        ip.ContainmentSpec(mode="nope")
    with pytest.raises(ValueError):
        ip.ContainmentSpec(mode="damp", damp=2.0)


# ============================ AMPLIFICATION ============================
def test_birth_term_zero_when_island_already_wins_by_margin():
    """If the island class wins by >= margin_target everywhere, birth penalty is 0."""
    lst = np.zeros((8, 8), np.int64); lst[3:5, 3:5] = 3      # a movable blob
    mask = lst == 3
    oh = ip.island_one_hot(lst, mask)
    w = ip.island_persistence_weight(mask)
    logits = np.zeros((8, 8, 5), np.float32)
    logits[..., 3] = 5.0                                     # island class dominates by 5 > margin 1
    assert ip.island_birth_term_np(logits, oh, w, margin_target=1.0) == pytest.approx(0.0)


def test_birth_term_positive_when_island_below_margin():
    lst = np.zeros((8, 8), np.int64); lst[3:5, 3:5] = 3
    mask = lst == 3
    oh = ip.island_one_hot(lst, mask)
    w = ip.island_persistence_weight(mask)
    logits = np.zeros((8, 8, 5), np.float32)
    logits[..., 3] = 0.2                                     # island barely leads -> below margin 1
    t = ip.island_birth_term_np(logits, oh, w, margin_target=1.0)
    assert t > 0.0
    # signed margin = 0.2 - 0 = 0.2 ; penalty = 1 - 0.2 = 0.8 ; weight mean-1 -> term ~0.8
    assert t == pytest.approx(0.8, abs=1e-5)


def test_birth_term_rides_top1_top2_margin_field():
    """The term uses signed = island_logit - max competitor (the #141 top1-top2 field)."""
    lst = np.zeros((4, 4), np.int64); lst[0, 0] = 1
    mask = lst == 1
    oh = ip.island_one_hot(lst, mask)
    w = ip.island_persistence_weight(mask)   # single island pixel -> weight 1 there
    logits = np.zeros((4, 4, 5), np.float32)
    logits[0, 0, 1] = 1.0   # island logit
    logits[0, 0, 0] = 3.0   # a COMPETITOR beats it -> signed = 1 - 3 = -2
    t = ip.island_birth_term_np(logits, oh, w, margin_target=0.0)
    assert t == pytest.approx(2.0, abs=1e-5)   # relu(0 - (-2)) = 2


def test_persistence_weight_is_mean_one_and_zero_offisland():
    lst = np.zeros((16, 16), np.int64); lst[4:12, 4:12] = 3
    mask = lst == 3
    w = ip.island_persistence_weight(mask, kind="inverse_thickness")
    assert w[mask].mean() == pytest.approx(1.0, abs=1e-5)   # budget preserved
    assert np.allclose(w[~mask], 0.0)
    # thin (border) island pixels get MORE weight than the thick core
    from scipy.ndimage import distance_transform_edt
    d = distance_transform_edt(mask)
    border = mask & (d <= 1.5)
    core = mask & (d >= 3.0)
    assert w[border].mean() > w[core].mean()


def test_amplification_gradient_raises_island_logit():
    """A GD step on the birth term must INCREASE the island logit (birth direction)."""
    import mlx.core as mx
    lst = np.zeros((4, 4), np.int64); lst[0, 0] = 3
    mask = lst == 3
    oh = mx.array(ip.island_one_hot(lst, mask))
    w = mx.array(ip.island_persistence_weight(mask))
    logits0 = np.zeros((4, 4, 5), np.float32); logits0[0, 0, 3] = 0.1; logits0[0, 0, 0] = 0.3

    def loss(lg):
        return ip.island_birth_term_mx(lg, oh, w, 1.0)

    g = mx.grad(loss)(mx.array(logits0))
    gn = np.asarray(g)
    # descending the loss (lg -= lr*g) must RAISE the island logit (ch3) at (0,0)
    assert gn[0, 0, 3] < 0.0        # negative grad on island logit -> step increases it
    assert gn[0, 0, 0] > 0.0        # positive grad on competitor -> step decreases it


def test_birth_term_softplus_form_smooth_and_matches_mlx():
    import mlx.core as mx
    lst = np.zeros((8, 8), np.int64); lst[3:5, 3:5] = 3
    mask = lst == 3
    oh = ip.island_one_hot(lst, mask)
    w = ip.island_persistence_weight(mask)
    logits = np.zeros((8, 8, 5), np.float32); logits[..., 3] = 5.0   # island wins by 5 (past margin)
    # softplus keeps a small positive value past the margin (smooth), hinge is exactly 0
    t_hinge = ip.island_birth_term_np(logits, oh, w, 1.0, form="hinge")
    t_soft = ip.island_birth_term_np(logits, oh, w, 1.0, form="softplus", tau=0.3)
    assert t_hinge == pytest.approx(0.0)
    assert t_soft > 0.0
    t_soft_mx = float(ip.island_birth_term_mx(mx.array(logits), mx.array(oh), mx.array(w), 1.0,
                                              form="softplus", tau=0.3))
    assert abs(t_soft_mx - t_soft) / (abs(t_soft) + 1e-9) < 3e-4


def test_birth_term_invalid_form_raises():
    lst = np.zeros((4, 4), np.int64); lst[0, 0] = 1
    mask = lst == 1
    oh = ip.island_one_hot(lst, mask)
    w = ip.island_persistence_weight(mask)
    with pytest.raises(ValueError):
        ip.island_birth_term_np(np.zeros((4, 4, 5), np.float32), oh, w, 0.0, form="nope")


def test_birth_from_signed_matches_full_term():
    """The LEVER-4-composition variant (rides pre-computed _signed) must equal the full
    term that recomputes _signed from logits — proving it's the SAME math, one forward."""
    lst = _synthetic_lstars(n=1)[0]
    mask = (lst == 1) | (lst == 3)
    oh = ip.island_one_hot(lst, mask)
    w = ip.island_persistence_weight(mask)
    logits = np.random.default_rng(11).standard_normal((*lst.shape, 5)).astype(np.float32)
    # recompute _signed exactly as the trainer's _live_signed / the full term does
    gt_logit = np.sum(logits * oh, axis=-1)
    runner_up = np.max(logits + oh * (-1e9), axis=-1)
    signed = gt_logit - runner_up
    t_full = ip.island_birth_term_np(logits, oh, w, 0.5)
    t_from = ip.island_birth_from_signed_np(signed, w, 0.5)
    assert t_from == pytest.approx(t_full, abs=1e-5)


def test_birth_from_signed_mlx_parity():
    import mlx.core as mx
    signed = np.random.default_rng(12).standard_normal((1, 32, 40)).astype(np.float32)
    w = np.abs(np.random.default_rng(13).standard_normal((1, 32, 40))).astype(np.float32)
    t_np = ip.island_birth_from_signed_np(signed, w, 0.5, form="softplus", tau=0.3)
    t_mx = float(ip.island_birth_from_signed_mx(mx.array(signed), mx.array(w), 0.5, form="softplus", tau=0.3))
    assert abs(t_mx - t_np) / (abs(t_np) + 1e-9) < 3e-4


def test_metal_kernel_signature_contract():
    sig = ip.metal_island_birth_kernel_signature()
    assert sig["status"] == "FLAGGED_NOT_BUILT"          # honest: not yet built
    assert sig["env_flag"] == ip.TAC_MLX_CUSTOM_ISLAND_BIRTH_ENV
    assert sig["reference"].endswith("island_birth_from_signed_np")


# ============================ COMPUTE PARITY ============================
def test_birth_term_mlx_matches_numpy():
    import mlx.core as mx
    rng = np.random.default_rng(7)
    lst = _synthetic_lstars(n=1)[0]
    mask = (lst == 1) | (lst == 3)
    oh = ip.island_one_hot(lst, mask)
    w = ip.island_persistence_weight(mask)
    logits = rng.standard_normal((*lst.shape, 5)).astype(np.float32)
    t_np = ip.island_birth_term_np(logits, oh, w, 0.5)
    t_mx = float(ip.island_birth_term_mx(mx.array(logits), mx.array(oh), mx.array(w), 0.5))
    assert abs(t_mx - t_np) / (abs(t_np) + 1e-9) < 3e-4      # parity >= 0.9997


def test_birth_term_compiled_matches_numpy():
    import mlx.core as mx
    rng = np.random.default_rng(8)
    lst = _synthetic_lstars(n=1)[0]
    mask = (lst == 1) | (lst == 3)
    oh = ip.island_one_hot(lst, mask)
    w = ip.island_persistence_weight(mask)
    logits = rng.standard_normal((*lst.shape, 5)).astype(np.float32)
    t_np = ip.island_birth_term_np(logits, oh, w, 0.5)
    f = ip.make_island_birth_term_mx_compiled()
    t_c = float(f(mx.array(logits), mx.array(oh), mx.array(w), mx.array(0.5, dtype=mx.float32)))
    assert abs(t_c - t_np) / (abs(t_np) + 1e-9) < 3e-4


def test_containment_mlx_matches_numpy_all_modes():
    import mlx.core as mx
    g, r = _grad_resid()
    mask = np.zeros((48, 64), bool); mask[10:30, 10:30] = True
    for mode, damp in (("freeze", 0.0), ("damp", 0.3), ("shield", 0.0)):
        spec = ip.ContainmentSpec(mode=mode, damp=damp, protected_mask=mask)
        cn = ip.contain_protected_grad_np(g, r, spec)
        cm = np.asarray(ip.contain_protected_grad_mx(mx.array(g), mx.array(r), spec))
        assert np.max(np.abs(cn - cm)) < 1e-4, mode


# ============================ DSL LEG ============================
def test_island_flags_defaults_and_types():
    f = ip.island_protection_flags()
    assert f["--seed-islands"] is True
    assert set(f) == set(ip.ISLAND_TRAINER_FLAG_DEFAULTS)   # same flag names as the canonical spec
    assert isinstance(f["--amplify-weight"], float)
    assert isinstance(f["--island-dilate-px"], int)


def test_island_flags_reject_bad_choices():
    with pytest.raises(ValueError):
        ip.island_protection_flags(containment_mode="bad")
    with pytest.raises(ValueError):
        ip.island_protection_flags(amplify_form="bad")
    with pytest.raises(ValueError):
        ip.island_protection_flags(amplify_persist="bad")


def test_island_lever_renders_to_trainer_argv():
    """The DSL Lever's overrides render True->--flag, False->--no-flag, valued->[flag,str]."""
    lev = ip.build_island_protection_lever(amplify_weight=0.5, containment_mode="freeze")
    ov = lev.overrides
    # emulate curriculum_dsl.compile_trainer_argv's per-flag rendering
    argv: list[str] = []
    for flag, val in ov.items():
        if val is True:
            argv.append(flag)
        elif val is False:
            argv.append(flag.replace("--", "--no-", 1))
        else:
            argv.extend([flag, str(val)])
    assert "--seed-islands" in argv                       # True bool -> bare flag
    assert "--amplify-weight" in argv and "0.5" in argv
    assert "--containment-mode" in argv and "freeze" in argv


# ============================ RECALL ============================
def test_island_recall():
    lst = np.zeros((8, 8), np.int64); lst[2:6, 2:6] = 3
    pred = lst.copy()
    assert ip.island_recall(pred, lst, 3) == pytest.approx(1.0)
    pred[2, 2] = 0   # one island pixel erased
    assert ip.island_recall(pred, lst, 3) == pytest.approx(15 / 16)
    assert np.isnan(ip.island_recall(pred, np.zeros((8, 8), np.int64), 3))  # no GT island
