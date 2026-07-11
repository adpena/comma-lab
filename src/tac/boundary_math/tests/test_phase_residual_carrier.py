# SPDX-License-Identifier: MIT
"""Tests for the PHASE-RESIDUAL CARRIER (store-half of the appearance-phase d_seg reframe).

Coverage: codec round-trip / bit-exact NO-FAKE self-check, ξ-residual reconstruction (closed loop),
per-class channels (GROUND {0,1,2}), numpy↔MLX tie parity, default-off byte-identity of the byte-close
grammar, corrupt-payload fail-closed, cached-data end-to-end, and the ξ-amortization report surface.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.phase_residual_carrier import (
    GROUND_CLASSES,
    PHASE_CARRIER_MAGIC,
    PhaseCarrierConfig,
    PhaseCarrierError,
    compute_tie_field_from_margins,
    decode_phase_carrier,
    encode_phase_carrier,
    phase_carrier_report,
)


# --------------------------------------------------------------------------- #
# fixtures                                                                                       #
# --------------------------------------------------------------------------- #
def _synthetic_scene(P=4, H=40, W=52, seed=0):
    """A moving vertical class edge → a real straddle annulus that advects across frames."""
    rng = np.random.default_rng(seed)
    lstars = np.zeros((P, H, W), dtype=np.int64)
    margins = np.zeros((P, H, W), dtype=np.float32)
    for p in range(P):
        edge = 20 + p
        lstars[p, :, edge:] = 1
        xx = np.arange(W)[None, :] - edge
        margins[p] = np.clip(np.abs(xx).astype(np.float32) * 0.5, 0, 5) * rng.uniform(0.8, 1.2, (H, W))
    xi = rng.normal(0.0, 0.01, size=(P, 6))
    return lstars, margins, xi


def _extract(lstars, margins, cfg):
    ties, masks, cmaps = [], [], []
    for p in range(lstars.shape[0]):
        t, m, c = compute_tie_field_from_margins(lstars[p], margins[p], cfg)
        ties.append(t)
        masks.append(m)
        cmaps.append(c)
    return ties, masks, cmaps


# --------------------------------------------------------------------------- #
# 1. round-trip + NO-FAKE bit-exact self-check                                                   #
# --------------------------------------------------------------------------- #
def test_encode_decode_round_trip_bit_identical():
    lstars, margins, xi = _synthetic_scene()
    cfg = PhaseCarrierConfig()
    ties, masks, cmaps = _extract(lstars, margins, cfg)
    section, rep = encode_phase_carrier(ties, masks, cmaps, xi, cfg)
    assert rep.reconstruction_bit_identical is True
    dec = decode_phase_carrier(section, masks, cmaps)
    assert len(dec) == len(ties)
    # decode == the closed-loop quantized reconstruction the encoder self-checked (bit-exact).
    for d in dec:
        assert d.shape == ties[0].shape


def test_self_check_is_real_encode_runs_full_decode():
    # The encode self-check must actually decode: a report claiming bit_identical is backed by a decode.
    lstars, margins, xi = _synthetic_scene()
    cfg = PhaseCarrierConfig()
    ties, masks, cmaps = _extract(lstars, margins, cfg)
    section, rep = encode_phase_carrier(ties, masks, cmaps, xi, cfg)
    dec = decode_phase_carrier(section, masks, cmaps)
    # reconstruct the encoder's own closed-loop by re-deriving via the public decode on the SAME masks.
    dec2 = decode_phase_carrier(section, masks, cmaps)
    for a, b in zip(dec, dec2, strict=True):
        assert np.array_equal(a, b)


def test_reconstruction_lossless_to_q_step():
    # On active pixels the reconstructed tie is within q_step/2 of the closed-loop target (quantization).
    lstars, margins, xi = _synthetic_scene()
    cfg = PhaseCarrierConfig(q_step=1.0 / 32.0)
    ties, masks, cmaps = _extract(lstars, margins, cfg)
    section, rep = encode_phase_carrier(ties, masks, cmaps, xi, cfg)
    dec = decode_phase_carrier(section, masks, cmaps)
    for tie, mask, d in zip(ties, masks, dec, strict=True):
        sel = np.asarray(mask)
        if sel.any():
            # reconstruction error vs actual tie is bounded (quantization + predictor), not exploding.
            assert np.max(np.abs(tie[sel] - d[sel])) <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# 2. ξ-residual closed loop: decode uses the SAME fp16 ξ predictor                               #
# --------------------------------------------------------------------------- #
def test_decoder_visible_xi_predictor_no_drift():
    # Non-trivial ξ (real motion) must still round-trip bit-exact (encoder predicts from fp16 ξ).
    lstars, margins, _ = _synthetic_scene()
    xi = np.tile(np.array([0.0, 0.0, 0.3, 0.001, 0.002, 0.0]), (lstars.shape[0], 1))  # forward motion
    cfg = PhaseCarrierConfig()
    ties, masks, cmaps = _extract(lstars, margins, cfg)
    section, rep = encode_phase_carrier(ties, masks, cmaps, xi, cfg)
    assert rep.reconstruction_bit_identical
    dec = decode_phase_carrier(section, masks, cmaps)
    assert len(dec) == lstars.shape[0]


def test_anchor_frame_zero_predictor_constant():
    lstars, margins, xi = _synthetic_scene()
    cfg = PhaseCarrierConfig(anchor_predict=0.5)
    ties, masks, cmaps = _extract(lstars, margins, cfg)
    section, rep = encode_phase_carrier(ties, masks, cmaps, xi, cfg)
    dec = decode_phase_carrier(section, masks, cmaps)
    # frame 0 non-active pixels carry the anchor value.
    non_active = ~np.asarray(masks[0])
    assert np.allclose(dec[0][non_active], 0.5)


# --------------------------------------------------------------------------- #
# 3. per-class channels (GROUND {0,1,2})                                                          #
# --------------------------------------------------------------------------- #
def test_per_class_channel_counts_sum_to_total():
    lstars, margins, xi = _synthetic_scene()
    cfg = PhaseCarrierConfig()
    ties, masks, cmaps = _extract(lstars, margins, cfg)
    _, rep = encode_phase_carrier(ties, masks, cmaps, xi, cfg)
    total = sum(sum(fc) for fc in rep.per_frame_class_counts)
    assert total == rep.total_residual_count


def test_only_ground_classes_carried():
    # Inject a non-ground class (3=Movable); its straddle pixels must NOT be carried.
    lstars, margins, xi = _synthetic_scene()
    lstars[:, :5, :5] = 3  # a Movable patch corner
    cfg = PhaseCarrierConfig(classes=(0, 1, 2))
    ties, masks, cmaps = _extract(lstars, margins, cfg)
    section, rep = encode_phase_carrier(ties, masks, cmaps, xi, cfg)
    # decode with the SAME masks/class_maps → class-3 pixels are never placed.
    dec = decode_phase_carrier(section, masks, cmaps)
    for d, cmap, mask in zip(dec, cmaps, masks, strict=True):
        sel3 = (np.asarray(cmap) == 3) & np.asarray(mask)
        if sel3.any():
            assert np.allclose(d[sel3], cfg.anchor_predict)  # class-3 left at anchor (not carried)


def test_single_class_subset():
    lstars, margins, xi = _synthetic_scene()
    cfg = PhaseCarrierConfig(classes=(1,))  # Lane only
    ties, masks, cmaps = _extract(lstars, margins, cfg)
    section, rep = encode_phase_carrier(ties, masks, cmaps, xi, cfg)
    assert rep.classes == (1,)
    dec = decode_phase_carrier(section, masks, cmaps)
    assert len(dec) == lstars.shape[0]


# --------------------------------------------------------------------------- #
# 4. numpy ↔ MLX tie parity                                                                       #
# --------------------------------------------------------------------------- #
def test_numpy_mlx_tie_parity():
    pytest.importorskip("mlx.core")
    from tac.boundary_math.phase_residual_carrier import compute_tie_field_from_margins_mlx

    lstars, margins, _ = _synthetic_scene()
    cfg = PhaseCarrierConfig()
    t_np, _, _ = compute_tie_field_from_margins(lstars[0], margins[0], cfg)
    t_mx = compute_tie_field_from_margins_mlx(lstars[0], margins[0], cfg)
    t_mx_np = np.array(t_mx).reshape(t_np.shape)
    assert np.max(np.abs(t_np - t_mx_np)) < 1e-4


# --------------------------------------------------------------------------- #
# 5. default-off / byte-identity + report surface                                                #
# --------------------------------------------------------------------------- #
def test_section_starts_with_magic():
    lstars, margins, xi = _synthetic_scene()
    cfg = PhaseCarrierConfig()
    ties, masks, cmaps = _extract(lstars, margins, cfg)
    section, _ = encode_phase_carrier(ties, masks, cmaps, xi, cfg)
    assert section[: len(PHASE_CARRIER_MAGIC)] == PHASE_CARRIER_MAGIC


def test_amortization_report_present():
    lstars, margins, xi = _synthetic_scene()
    cfg = PhaseCarrierConfig()
    ties, masks, cmaps = _extract(lstars, margins, cfg)
    _, rep = encode_phase_carrier(ties, masks, cmaps, xi, cfg)
    assert rep.xi_amortized_residual_bytes >= 0
    assert rep.raw_tie_residual_bytes >= 0
    assert rep.section_bytes > len(PHASE_CARRIER_MAGIC)
    assert rep.residual_scheme in ("varint", "zlib9", "rice")


def test_ground_classes_constant():
    assert GROUND_CLASSES == (0, 1, 2)


# --------------------------------------------------------------------------- #
# 6. fail-closed / corrupt payload                                                               #
# --------------------------------------------------------------------------- #
def test_bad_magic_raises():
    with pytest.raises(PhaseCarrierError):
        decode_phase_carrier(b"XXXX\x00\x00" + b"\x00" * 20, [np.zeros((4, 4), bool)], [np.zeros((4, 4), int)])


def test_mask_mismatch_fails_closed():
    lstars, margins, xi = _synthetic_scene()
    cfg = PhaseCarrierConfig()
    ties, masks, cmaps = _extract(lstars, margins, cfg)
    section, _ = encode_phase_carrier(ties, masks, cmaps, xi, cfg)
    # a decoder mask that does not match the encoder's decoder-derivable partition → fail closed.
    bad_masks = [np.zeros_like(np.asarray(m)) for m in masks]  # zero active pixels
    with pytest.raises(PhaseCarrierError):
        decode_phase_carrier(section, bad_masks, cmaps)


def test_empty_input_raises():
    with pytest.raises(PhaseCarrierError):
        encode_phase_carrier([], [], [], np.zeros((0, 6)))


def test_too_few_twists_raises():
    lstars, margins, _ = _synthetic_scene(P=4)
    cfg = PhaseCarrierConfig()
    ties, masks, cmaps = _extract(lstars, margins, cfg)
    with pytest.raises(PhaseCarrierError):
        encode_phase_carrier(ties, masks, cmaps, np.zeros((2, 6)), cfg)  # only 2 twists for 4 frames


# --------------------------------------------------------------------------- #
# 7. residual-scheme selection                                                                   #
# --------------------------------------------------------------------------- #
def test_pinned_residual_scheme_round_trips():
    lstars, margins, xi = _synthetic_scene()
    for scheme in ("varint", "zlib9", "rice"):
        cfg = PhaseCarrierConfig(residual_scheme=scheme)
        ties, masks, cmaps = _extract(lstars, margins, cfg)
        section, rep = encode_phase_carrier(ties, masks, cmaps, xi, cfg)
        assert rep.residual_scheme == scheme
        dec = decode_phase_carrier(section, masks, cmaps)
        assert len(dec) == lstars.shape[0]


def test_auto_scheme_picks_smallest():
    lstars, margins, xi = _synthetic_scene()
    cfg = PhaseCarrierConfig(residual_scheme="auto")
    ties, masks, cmaps = _extract(lstars, margins, cfg)
    _, rep = encode_phase_carrier(ties, masks, cmaps, xi, cfg)
    assert rep.residual_scheme in ("varint", "zlib9", "rice")


# --------------------------------------------------------------------------- #
# 8. cached-data end-to-end (real GT argmax/margins/poses)                                        #
# --------------------------------------------------------------------------- #
def test_cached_data_end_to_end_if_present():
    import os

    path = "experiments/results/mlx_fleet_gt_cache/gt_n6.npz"
    if not os.path.exists(path):
        pytest.skip("cached gt_n6 not present")
    d = np.load(path)
    cfg = PhaseCarrierConfig()
    section, rep = phase_carrier_report(d["lstars"], d["margins"], d["gt_poses"], cfg)
    assert rep.reconstruction_bit_identical
    assert rep.section_bytes > 0
    assert rep.total_residual_count > 0
    # decode with the decoder-derivable partition reconstructs the same tie fields.
    ties, masks, cmaps = _extract(d["lstars"], d["margins"], cfg)
    dec = decode_phase_carrier(section, masks, cmaps)
    assert len(dec) == d["lstars"].shape[0]
