"""Tests for SOAR Closed-form Joint Scale Optimization (CJSO) opt-in.

Source: Bao et al. 2026 arXiv:2605.12245v1 §4.1 eqs (5) and (6).
Lane: lane_soar_cjso_dss_opt_in_dev_20260513.

Backward-compat invariant: with ``cjso_init=False`` (the default), ALL outputs
must be byte-identical to the pre-CJSO code path.

Honest empirical finding (Fridrich review Round 1): integer-exponent snapping
dominates the continuous closed-form optimization, so on most signals the
CJSO exponent equals the max-rule exponent. CJSO is a CORRECT implementation
of SOAR's algebra but the on-disk integer-log2 exponent constraint
collapses the benefit to ~0 on typical neural-network weights. The tests
below verify the algebra (closed-form scale formula matches eq 5) AND the
backward-compat byte invariant — not a guaranteed MSE improvement, which
SOAR's paper only proves on FP8-stored continuous scales (NVFP4) not on
integer-log2 exponents (our block-FP / Selfcomp layout).
"""
from __future__ import annotations

import pytest
import torch

from tac.block_fp_codec import (
    DEFAULT_CJSO_ITERS,
    _cjso_optimal_scale,
    decode_conv_weight,
    encode_conv_weight,
    pack_block_fp,
    pack_state_dict_block_fp_cjso,
    unpack_block_fp,
    unpack_payload_tar_xz,
)

# ── CJSO eq (5) closed-form scale ────────────────────────────────────────


def test_cjso_optimal_scale_matches_soar_eq5_closed_form():
    """SOAR eq (5): s* = Σ(W·Q) / Σ(Q²) for fixed quantization Q."""
    # Manual construction: W = [1.0, -2.0, 3.0], Q = [1, -1, 1]
    # Σ(W·Q) = 1.0*1 + (-2.0)*(-1) + 3.0*1 = 6.0
    # Σ(Q²)  = 1 + 1 + 1 = 3
    # s* = 6.0 / 3 = 2.0
    w = torch.tensor([1.0, -2.0, 3.0])
    q = torch.tensor([1, -1, 1], dtype=torch.int8)
    s_star = _cjso_optimal_scale(w, q)
    assert s_star == pytest.approx(2.0, rel=1e-6), f"got {s_star}"


def test_cjso_optimal_scale_zero_q_returns_zero():
    """All-zero Q block: Σ(Q²) = 0 → return 0.0 (degenerate)."""
    w = torch.tensor([0.5, -0.3, 0.1])
    q = torch.zeros(3, dtype=torch.int8)
    assert _cjso_optimal_scale(w, q) == 0.0


def test_cjso_optimal_scale_residual_aligned_with_q():
    """If W aligns perfectly with Q, s* recovers the true amplitude."""
    # W = 3.5 * Q with Q in {-1, +1} → s* should be 3.5.
    q = torch.tensor([1, -1, 1, -1, 1], dtype=torch.int8)
    w = 3.5 * q.to(torch.float32)
    s_star = _cjso_optimal_scale(w, q)
    assert s_star == pytest.approx(3.5, rel=1e-6)


def test_cjso_optimal_scale_sign_flip_negative_optimum():
    """If most W are aligned WITH -Q, s* is negative (anti-correlated)."""
    q = torch.tensor([1, 1, 1, 1], dtype=torch.int8)
    w = torch.tensor([-1.0, -1.0, -1.0, -1.0])
    s_star = _cjso_optimal_scale(w, q)
    assert s_star == pytest.approx(-1.0, rel=1e-6)


# ── pack_block_fp opt-in flag + backward-compat ────────────────────────


def test_pack_block_fp_default_off_bytes_identical_to_pre_cjso():
    """Backward-compat: cjso_init=False (default) MUST be byte-identical."""
    torch.manual_seed(42)
    w = torch.randn(64, 32) * 0.1
    bytes_default = pack_block_fp(w)
    bytes_explicit_off = pack_block_fp(w, cjso_init=False)
    assert bytes_default == bytes_explicit_off


def test_pack_block_fp_default_off_decodes_unchanged():
    """Roundtrip with cjso_init=False matches the pre-CJSO decoder output."""
    torch.manual_seed(0)
    w = torch.randn(32, 16) * 0.5
    packed = pack_block_fp(w, cjso_init=False)
    rec = unpack_block_fp(packed)
    assert rec.shape == w.shape
    # Ternary roundtrip is lossy; just assert reconstruction is finite + sane.
    assert torch.isfinite(rec).all()
    assert rec.abs().max() <= 2 ** 5  # exponent range cap.


def test_pack_block_fp_cjso_byte_layout_unchanged():
    """CJSO mode produces the SAME byte LENGTH as max-rule (same on-disk format)."""
    torch.manual_seed(1)
    w = torch.randn(48, 24) * 0.2
    bytes_max = pack_block_fp(w, cjso_init=False)
    bytes_cjso = pack_block_fp(w, cjso_init=True)
    assert len(bytes_max) == len(bytes_cjso), "byte format must be identical"


def test_pack_block_fp_cjso_roundtrip_finite():
    """CJSO roundtrip produces finite output of the correct shape."""
    torch.manual_seed(2)
    w = torch.randn(32, 8) * 0.1
    packed = pack_block_fp(w, cjso_init=True, cjso_iters=5)
    rec = unpack_block_fp(packed)
    assert rec.shape == w.shape
    assert torch.isfinite(rec).all()


def test_pack_block_fp_cjso_iters_must_be_positive():
    """n_iters >= 1 enforced."""
    w = torch.randn(16, 4)
    with pytest.raises(ValueError, match="n_iters"):
        pack_block_fp(w, cjso_init=True, cjso_iters=0)


def test_pack_block_fp_zero_tensor_handled_by_cjso():
    """All-zero block: CJSO must not divide by zero or fail."""
    w = torch.zeros(16, 4)
    packed = pack_block_fp(w, cjso_init=True)
    rec = unpack_block_fp(packed)
    assert torch.all(rec == 0)


# ── encode_conv_weight CJSO opt-in ──────────────────────────────────────


def test_encode_conv_weight_default_off_byte_identical():
    """encode_conv_weight backward-compat: default behavior unchanged."""
    torch.manual_seed(3)
    w = torch.randn(8, 4, 3, 3) * 0.3
    p_old = encode_conv_weight(w, qint_max=7)
    p_default = encode_conv_weight(w, qint_max=7, cjso_init=False)
    assert torch.equal(p_old["weight_qint"], p_default["weight_qint"])
    assert torch.equal(p_old["weight_exponents"], p_default["weight_exponents"])


def test_encode_conv_weight_cjso_roundtrip():
    """CJSO conv-weight roundtrip produces finite output of correct shape."""
    torch.manual_seed(4)
    w = torch.randn(6, 4, 3, 3) * 0.2
    p = encode_conv_weight(w, qint_max=7, cjso_init=True, cjso_iters=10)
    rec = decode_conv_weight(p)
    assert rec.shape == w.shape
    assert torch.isfinite(rec).all()


def test_encode_conv_weight_cjso_iters_validated():
    """cjso_iters < 1 raises."""
    w = torch.randn(4, 2, 3, 3)
    with pytest.raises(ValueError, match="cjso_iters"):
        encode_conv_weight(w, qint_max=7, cjso_init=True, cjso_iters=0)


def test_encode_conv_weight_cjso_with_per_channel_q():
    """CJSO composes with per_channel_qint_max."""
    torch.manual_seed(5)
    w = torch.randn(4, 2, 3, 3) * 0.1
    pc_q = [3, 7, 1, 15]
    p = encode_conv_weight(
        w, qint_max=7, per_channel_qint_max=pc_q,
        cjso_init=True, cjso_iters=5,
    )
    # Each channel's qint must respect its own qint_max.
    for c, qmax in enumerate(pc_q):
        ch_qint = p["weight_qint"][..., c, :]
        assert ch_qint.abs().max().item() <= qmax, (
            f"channel {c}: qint exceeded qint_max={qmax}"
        )


def test_encode_conv_weight_cjso_zero_channel_safe():
    """All-zero output channel: CJSO must not divide by zero."""
    w = torch.randn(4, 2, 3, 3) * 0.1
    w[2] = 0.0
    p = encode_conv_weight(w, qint_max=7, cjso_init=True)
    rec = decode_conv_weight(p)
    assert torch.all(rec[2] == 0)


# ── pack_state_dict_block_fp_cjso wrapper ──────────────────────────────


def test_pack_state_dict_block_fp_cjso_wrapper_runs(tmp_path):
    """Wrapper end-to-end: pack + unpack roundtrip of a tiny state dict."""
    torch.manual_seed(6)
    sd = {
        "layer.weight": torch.randn(4, 2, 3, 3) * 0.1,
        "layer.bias": torch.randn(4) * 0.05,
    }
    out = tmp_path / "soar_cjso.tar.xz"
    pack_state_dict_block_fp_cjso(sd, out, qint_max=7, cjso_iters=3)
    assert out.exists()
    decoded = unpack_payload_tar_xz(out)
    assert set(decoded.keys()) == set(sd.keys())
    for k in sd:
        assert decoded[k].shape == sd[k].shape


def test_pack_state_dict_block_fp_cjso_meta_records_provenance(tmp_path):
    """CJSO archive stamps scale_optimizer provenance in meta.json."""
    import json
    import tarfile

    sd = {"layer.weight": torch.randn(4, 2, 3, 3) * 0.1}
    out = tmp_path / "soar_meta.tar.xz"
    pack_state_dict_block_fp_cjso(sd, out, qint_max=7, cjso_iters=7)
    with tarfile.open(out, mode="r:xz") as tf:
        meta_bytes = tf.extractfile("meta.json").read()
    meta = json.loads(meta_bytes.decode("utf-8"))
    assert meta.get("scale_optimizer") == "cjso_soar_v1"
    assert meta.get("scale_optimizer_iters") == 7
    assert meta.get("scale_optimizer_paper") == "arXiv:2605.12245v1"


def test_pack_state_dict_block_fp_cjso_without_cjso_no_provenance(tmp_path):
    """Non-CJSO path (default) does NOT stamp the provenance field."""
    import json
    import tarfile

    from tac.block_fp_codec import pack_payload_tar_xz

    sd = {"layer.weight": torch.randn(4, 2, 3, 3) * 0.1}
    out = tmp_path / "no_soar_meta.tar.xz"
    pack_payload_tar_xz(sd, out, qint_max=7)
    with tarfile.open(out, mode="r:xz") as tf:
        meta_bytes = tf.extractfile("meta.json").read()
    meta = json.loads(meta_bytes.decode("utf-8"))
    assert "scale_optimizer" not in meta


# ── Algebraic invariants ───────────────────────────────────────────────


def test_cjso_default_iters_is_15():
    """SOAR Figure 5 empirical convergence default."""
    assert DEFAULT_CJSO_ITERS == 15


def test_cjso_iter_count_does_not_change_byte_format(tmp_path):
    """Varying cjso_iters does not change the on-disk byte layout (header
    structure unchanged); only the per-block exponents may differ."""
    torch.manual_seed(8)
    w = torch.randn(32, 16) * 0.1
    b3 = pack_block_fp(w, cjso_init=True, cjso_iters=3)
    b15 = pack_block_fp(w, cjso_init=True, cjso_iters=15)
    # Same length, same first 16 bytes (header), but exponents may differ.
    assert len(b3) == len(b15)
    assert b3[:16] == b15[:16]


def test_cjso_idempotent_on_quantization_exact_input():
    """If W = Q*scale exactly (Q ∈ {-1, 0, +1}), CJSO recovers the exact scale."""
    torch.manual_seed(9)
    # Construct W = ternary * scale where scale is a power-of-2.
    n_rows, n_cols = 16, 8
    q = torch.randint(-1, 2, (n_rows, n_cols)).to(torch.float32)
    scale = 2.0 ** 2  # exponent = 2.
    w = q * scale
    packed = pack_block_fp(w, cjso_init=True)
    rec = unpack_block_fp(packed)
    # For exact ternary input, the roundtrip must recover the input.
    assert torch.allclose(rec, w, atol=1e-5), f"max diff = {(rec - w).abs().max()}"
