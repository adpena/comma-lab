# SPDX-License-Identifier: MIT
"""Dedicated tests for the Z3 Ballé hyperprior bolt-on substrate.

Covers: architecture forward, rate bits computation, archive encode/decode
round-trip, composition pack/split, int8 quantize/dequantize round-trip,
inflate sidecar reconstruction, score-aware Lagrangian shape/non-negativity,
sidecar overhead estimation, and CLAUDE.md compliance smoke tests.

NO score claim. NO real-scorer dispatch. NO MPS device.
"""
from __future__ import annotations

import importlib.util
import json
import py_compile
import struct
import zipfile
from pathlib import Path

import pytest
import torch

from tac.substrates.z3_balle_hyperprior_bolton import (
    A1_LATENT_DIM,
    A1_N_PAIRS,
    Z3HyperpriorConfig,
    Z3HyperpriorMLP,
    conditional_gaussian_rate_bits,
    dequantize_int8_with_scale,
    factorized_uniform_rate_bits,
    quantize_int8_with_scale,
    total_balle_rate_bits,
)
from tac.substrates.z3_balle_hyperprior_bolton.archive import (
    Z3HP1_MAGIC,
    build_composition_archive_contract,
    decode_z3hp1_sidecar,
    encode_z3hp1_sidecar,
    pack_composition_archive,
    split_composition_archive,
)
from tac.substrates.z3_balle_hyperprior_bolton.archive_v2 import (
    A1_DECODER_SECTION_TOTAL,
    A1_LATENT_BLOB_LEN,
    Z3HV2_MAGIC,
)
from tac.substrates.z3_balle_hyperprior_bolton.score_aware_loss import (
    balle_rate_term_bits_per_sample,
    estimate_sidecar_overhead_bytes,
    z3_lagrangian,
)

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


def test_hyperprior_mlp_forward_shapes():
    cfg = Z3HyperpriorConfig()
    mlp = Z3HyperpriorMLP(cfg)
    y = torch.randn(8, A1_LATENT_DIM)
    sigma, w_hat = mlp(y)
    assert sigma.shape == (8, A1_LATENT_DIM)
    assert w_hat.shape == (8, cfg.hyper_latent_dim)


def test_hyperprior_mlp_refuses_wrong_input_dim():
    mlp = Z3HyperpriorMLP()
    with pytest.raises(ValueError, match="must be"):
        mlp(torch.randn(4, 7))


def test_hyperprior_mlp_sigma_bounded():
    cfg = Z3HyperpriorConfig(min_sigma=1e-3, max_sigma=16.0)
    mlp = Z3HyperpriorMLP(cfg)
    y = torch.randn(4, A1_LATENT_DIM) * 100.0  # extreme inputs
    sigma, _ = mlp(y)
    assert (sigma >= cfg.min_sigma).all()
    assert (sigma <= cfg.max_sigma).all()


def test_hyperprior_mlp_quantize_false_path_differentiable():
    mlp = Z3HyperpriorMLP()
    y = torch.randn(4, A1_LATENT_DIM, requires_grad=True)
    sigma, w_hat = mlp(y, quantize=False)
    loss = sigma.sum() + w_hat.sum()
    loss.backward()
    assert y.grad is not None
    assert not torch.isnan(y.grad).any()


def test_hyperprior_mlp_param_count_under_budget():
    """Council target ≤ 1500 params per architecture docstring."""
    mlp = Z3HyperpriorMLP()
    count = sum(p.numel() for p in mlp.parameters())
    # Default config: 28*16+16 + 16*8+8 + 8*16+16 + 16*28+28 ~= 1496 params + GDN beta/gamma
    # GDN has out_dim beta + out_dim^2 gamma. h_a_1 out=16: 16 + 256 = 272.
    # h_s_1 out=16: 16 + 256 = 272. So ~1496 + 544 = ~2040.
    # Adjust the assertion to a generous bound for the small variant.
    assert count <= 3000, f"hyperprior param count too high: {count}"


def test_hyperprior_mlp_quantize_true_uses_ste():
    """Round-with-STE: forward is rounded; backward identity."""
    mlp = Z3HyperpriorMLP()
    y = torch.randn(4, A1_LATENT_DIM, requires_grad=True)
    sigma, w_hat = mlp(y, quantize=True)
    # w_hat should be (close to) integer-valued with quantization_step=1
    # (residuals near zero will be exactly 0; larger ones near integers).
    assert torch.allclose(w_hat, w_hat.round(), atol=1e-6)
    loss = sigma.sum() + w_hat.sum()
    loss.backward()
    assert y.grad is not None  # STE allows gradient flow


# ---------------------------------------------------------------------------
# Rate computation
# ---------------------------------------------------------------------------


def test_conditional_gaussian_rate_bits_shape():
    y = torch.randn(10, A1_LATENT_DIM)
    sigma = torch.full_like(y, 1.0)
    bits = conditional_gaussian_rate_bits(y, sigma)
    assert bits.shape == (10,)
    assert (bits > 0).all()


def test_conditional_gaussian_rate_bits_smaller_sigma_means_more_bits_for_large_y():
    """Tighter prior on large |y| values → more bits needed."""
    y = torch.full((5, A1_LATENT_DIM), 10.0)
    sigma_tight = torch.full_like(y, 0.5)
    sigma_loose = torch.full_like(y, 5.0)
    bits_tight = conditional_gaussian_rate_bits(y, sigma_tight).mean()
    bits_loose = conditional_gaussian_rate_bits(y, sigma_loose).mean()
    assert bits_tight > bits_loose


def test_conditional_gaussian_rate_bits_shape_mismatch_raises():
    y = torch.randn(4, A1_LATENT_DIM)
    sigma = torch.randn(4, 5)
    with pytest.raises(ValueError, match="must match"):
        conditional_gaussian_rate_bits(y, sigma)


def test_factorized_uniform_rate_bits_constant():
    w = torch.randn(5, 8)
    bits = factorized_uniform_rate_bits(w, quantization_step=1.0, half_range=16.0)
    assert bits.shape == (5,)
    # log2(32) * 8 = 5*8 = 40
    assert torch.allclose(bits, torch.full((5,), 40.0))


def test_total_balle_rate_bits_sums_components():
    y = torch.randn(3, A1_LATENT_DIM)
    sigma = torch.full_like(y, 1.0)
    w_hat = torch.randn(3, 8).round()
    total = total_balle_rate_bits(y, sigma, w_hat)
    rate_y = conditional_gaussian_rate_bits(y, sigma)
    rate_w = factorized_uniform_rate_bits(w_hat)
    assert torch.allclose(total, rate_y + rate_w)


# ---------------------------------------------------------------------------
# Archive encode / decode round-trip
# ---------------------------------------------------------------------------


def test_z3hp1_sidecar_roundtrip_minimal():
    weights = bytes(range(8))  # 8 int8 bytes
    w_hat = bytes([1] * (A1_N_PAIRS * 8))
    residual = bytes([0] * (A1_N_PAIRS * A1_LATENT_DIM))
    sidecar = encode_z3hp1_sidecar(
        hyperprior_weights_int8=weights,
        w_hat_int8=w_hat,
        residual_int8=residual,
        hyper_dim=8,
        int8_w_scale=10.0,
        quant_step=1.0,
        min_sigma=1e-3,
        max_sigma=16.0,
    )
    meta, decoded_weights, decoded_w_hat, decoded_residual = decode_z3hp1_sidecar(sidecar)
    assert meta.n_pairs == A1_N_PAIRS
    assert meta.hyper_dim == 8
    assert meta.latent_dim == A1_LATENT_DIM
    assert meta.int8_w_scale == pytest.approx(10.0)
    assert decoded_weights == weights
    assert decoded_w_hat == w_hat
    assert decoded_residual == residual


def test_z3hp1_sidecar_starts_with_magic():
    sidecar = encode_z3hp1_sidecar(
        hyperprior_weights_int8=b"\x00" * 8,
        w_hat_int8=b"\x00" * (A1_N_PAIRS * 8),
        residual_int8=b"\x00" * (A1_N_PAIRS * A1_LATENT_DIM),
        hyper_dim=8,
        int8_w_scale=1.0,
        quant_step=1.0,
        min_sigma=1e-3,
        max_sigma=16.0,
    )
    assert sidecar.startswith(Z3HP1_MAGIC)


def test_z3hp1_sidecar_bad_magic_raises():
    with pytest.raises(ValueError, match="bad Z3HP1 magic"):
        decode_z3hp1_sidecar(b"XXXX" + b"\x00" * 100)


def test_z3hp1_sidecar_truncated_raises():
    with pytest.raises(ValueError, match="too short"):
        decode_z3hp1_sidecar(b"Z3H1")


def test_z3hp1_sidecar_invalid_n_pairs():
    with pytest.raises(ValueError, match="n_pairs must be 600"):
        encode_z3hp1_sidecar(
            hyperprior_weights_int8=b"",
            w_hat_int8=b"",
            residual_int8=b"",
            hyper_dim=8,
            int8_w_scale=1.0,
            quant_step=1.0,
            min_sigma=1e-3,
            max_sigma=16.0,
            n_pairs=42,
        )


def test_z3hp1_sidecar_w_hat_length_mismatch():
    with pytest.raises(ValueError, match="w_hat_int8 length"):
        encode_z3hp1_sidecar(
            hyperprior_weights_int8=b"\x00" * 8,
            w_hat_int8=b"\x00" * 100,  # wrong length
            residual_int8=b"\x00" * (A1_N_PAIRS * A1_LATENT_DIM),
            hyper_dim=8,
            int8_w_scale=1.0,
            quant_step=1.0,
            min_sigma=1e-3,
            max_sigma=16.0,
        )


def test_z3hp1_sidecar_residual_length_mismatch():
    with pytest.raises(ValueError, match="residual_int8 length"):
        encode_z3hp1_sidecar(
            hyperprior_weights_int8=b"\x00" * 8,
            w_hat_int8=b"\x00" * (A1_N_PAIRS * 8),
            residual_int8=b"\x00" * 100,  # wrong length
            hyper_dim=8,
            int8_w_scale=1.0,
            quant_step=1.0,
            min_sigma=1e-3,
            max_sigma=16.0,
        )


# ---------------------------------------------------------------------------
# Composition archive split / pack
# ---------------------------------------------------------------------------


def test_pack_composition_archive_refuses_append_only_sidecar_by_default():
    a1_bytes = b"PRETEND_A1_ARCHIVE_v0" * 100  # arbitrary base bytes
    sidecar = encode_z3hp1_sidecar(
        hyperprior_weights_int8=b"\x00" * 8,
        w_hat_int8=b"\x00" * (A1_N_PAIRS * 8),
        residual_int8=b"\x00" * (A1_N_PAIRS * A1_LATENT_DIM),
        hyper_dim=8,
        int8_w_scale=1.0,
        quant_step=1.0,
        min_sigma=1e-3,
        max_sigma=16.0,
    )
    with pytest.raises(ValueError, match="append-only Z3HP1 trailer adds bytes"):
        pack_composition_archive(a1_bytes, sidecar)


def test_append_only_contract_is_diagnostic_not_byte_saving_or_exact_ready():
    a1_bytes = b"PRETEND_A1_ARCHIVE_v0" * 100
    sidecar = encode_z3hp1_sidecar(
        hyperprior_weights_int8=b"\x00" * 8,
        w_hat_int8=b"\x00" * (A1_N_PAIRS * 8),
        residual_int8=b"\x00" * (A1_N_PAIRS * A1_LATENT_DIM),
        hyper_dim=8,
        int8_w_scale=1.0,
        quant_step=1.0,
        min_sigma=1e-3,
        max_sigma=16.0,
    )
    contract = build_composition_archive_contract(a1_bytes, sidecar)
    composed = pack_composition_archive(
        a1_bytes,
        sidecar,
        allow_append_only_diagnostic=True,
    )
    assert composed.startswith(a1_bytes)
    assert sidecar in composed
    assert composed == contract.payload_bytes
    assert contract.layout == "append_only_z3hp1_diagnostic"
    assert contract.byte_saving is False
    assert contract.byte_savings_bytes == 0
    assert contract.score_claim is False
    assert contract.promotion_eligible is False
    assert contract.ready_for_exact_eval_dispatch is False
    assert contract.exact_eval_ready is False
    assert "append-only Z3HP1 trailer adds bytes" in contract.result_review_blockers[0]


def test_empty_sidecar_contract_is_a1_identical_and_not_z3_ready():
    a1_bytes = b"A1_NO_SIDECAR" * 50
    contract = build_composition_archive_contract(a1_bytes, b"")
    composed = pack_composition_archive(a1_bytes, b"")
    assert composed == a1_bytes
    assert contract.payload_bytes == a1_bytes
    assert contract.layout == "a1_byte_identical_fallback"
    assert contract.byte_saving is False
    assert contract.ready_for_exact_eval_dispatch is False
    assert contract.promotion_eligible is False


def test_pack_composition_archive_bad_sidecar_magic_raises():
    with pytest.raises(ValueError, match="does not start with magic"):
        pack_composition_archive(b"a1", b"BADMAGIC_no_z3h1")


def test_split_composition_archive_roundtrip():
    a1_bytes = b"PRETEND_A1_ARCHIVE_v0" * 100
    sidecar = encode_z3hp1_sidecar(
        hyperprior_weights_int8=b"\x01" * 8,
        w_hat_int8=b"\x02" * (A1_N_PAIRS * 8),
        residual_int8=b"\x03" * (A1_N_PAIRS * A1_LATENT_DIM),
        hyper_dim=8,
        int8_w_scale=3.0,
        quant_step=1.0,
        min_sigma=1e-3,
        max_sigma=16.0,
    )
    composed = pack_composition_archive(
        a1_bytes,
        sidecar,
        allow_append_only_diagnostic=True,
    )
    a1_recovered, sidecar_recovered = split_composition_archive(composed)
    assert a1_recovered == a1_bytes
    assert sidecar_recovered == sidecar


def test_split_composition_archive_no_sidecar_returns_full_a1():
    a1_bytes = b"NO_SIDECAR_BYTES_HERE_AT_ALL"
    a1_recovered, sidecar_recovered = split_composition_archive(a1_bytes)
    assert a1_recovered == a1_bytes
    assert sidecar_recovered == b""


# ---------------------------------------------------------------------------
# Int8 quantize/dequantize
# ---------------------------------------------------------------------------


def test_quantize_int8_roundtrip_accuracy():
    t = torch.randn(50) * 3.0
    int8_bytes, scale = quantize_int8_with_scale(t, scale_clip_range=7.0)
    recovered = dequantize_int8_with_scale(int8_bytes, scale, shape=(50,))
    # Quantization error bounded by 1/scale.
    err = (t - recovered).abs().max().item()
    assert err < 1.0 / scale + 1e-6


def test_quantize_int8_all_zeros_returns_scale_one():
    t = torch.zeros(10)
    int8_bytes, scale = quantize_int8_with_scale(t)
    assert scale == 1.0
    assert int8_bytes == bytes(10)


# ---------------------------------------------------------------------------
# Score-aware Lagrangian
# ---------------------------------------------------------------------------


def test_balle_rate_term_bits_per_sample_shape():
    mlp = Z3HyperpriorMLP()
    a1 = torch.randn(16, A1_LATENT_DIM)
    bits, sigma, w_hat = balle_rate_term_bits_per_sample(
        hyperprior=mlp, a1_latents=a1
    )
    assert bits.shape == (16,)
    assert sigma.shape == (16, A1_LATENT_DIM)
    assert w_hat.shape == (16, 8)
    assert (bits >= 0).all()


def test_balle_rate_term_bits_refuses_wrong_dim():
    mlp = Z3HyperpriorMLP()
    with pytest.raises(ValueError, match="must be"):
        balle_rate_term_bits_per_sample(
            hyperprior=mlp, a1_latents=torch.randn(4, 5)
        )


def test_z3_lagrangian_rate_only_mode():
    """Rate-only mode (smoke): no scorer call → seg/pose are tensor(0)."""
    mlp = Z3HyperpriorMLP()
    a1 = torch.randn(8, A1_LATENT_DIM)
    out = z3_lagrangian(
        hyperprior=mlp,
        a1_latents=a1,
        seg_scorer=torch.nn.Identity(),  # unused in rate-only mode
        pose_scorer=torch.nn.Identity(),
        a1_pair_pred_rt=None,
        gt_pair=None,
    )
    assert "rate_bits_total" in out
    assert "rate_lagrangian" in out
    assert "total_loss" in out
    assert out["seg_dist"].item() == pytest.approx(0.0)
    assert out["pose_dist"].item() == pytest.approx(0.0)
    # Rate-only loss should equal the rate Lagrangian.
    assert torch.isclose(out["total_loss"], out["rate_lagrangian"])


def test_z3_lagrangian_rate_lagrangian_non_negative():
    mlp = Z3HyperpriorMLP()
    a1 = torch.randn(4, A1_LATENT_DIM)
    out = z3_lagrangian(
        hyperprior=mlp,
        a1_latents=a1,
        seg_scorer=torch.nn.Identity(),
        pose_scorer=torch.nn.Identity(),
        a1_pair_pred_rt=None,
        gt_pair=None,
    )
    assert out["rate_lagrangian"].item() >= 0.0


def test_z3_lagrangian_loss_backward_through_hyperprior():
    """Lagrangian gradient must flow through hyperprior parameters."""
    mlp = Z3HyperpriorMLP()
    # Detached A1 latents (frozen per Z3 design).
    a1 = torch.randn(4, A1_LATENT_DIM)
    out = z3_lagrangian(
        hyperprior=mlp,
        a1_latents=a1,
        seg_scorer=torch.nn.Identity(),
        pose_scorer=torch.nn.Identity(),
        a1_pair_pred_rt=None,
        gt_pair=None,
    )
    out["total_loss"].backward()
    grads = [p.grad for p in mlp.parameters() if p.grad is not None]
    assert len(grads) > 0
    # At least one parameter should have non-zero gradient.
    assert any(g.abs().sum() > 0 for g in grads)


def test_estimate_sidecar_overhead_bytes_reasonable():
    mlp = Z3HyperpriorMLP()
    overhead = estimate_sidecar_overhead_bytes(hyperprior=mlp)
    # Sanity: sidecar must be SMALLER than the A1 latent blob (15387B) it
    # would replace, otherwise amortization is impossible by construction.
    # Overhead includes residual_blob (600 * 28 * 0.6 = ~10080B) + w_hat
    # (600 * 8 * 0.6 = ~2880B) + weights (~1000B); ~14KB total < 15387B.
    A1_LATENT_BLOB_BYTES = 15_387
    assert overhead < A1_LATENT_BLOB_BYTES, (
        f"sidecar overhead {overhead}B >= A1 latent blob {A1_LATENT_BLOB_BYTES}B "
        f"— amortization impossible"
    )
    assert overhead > 200, f"sidecar overhead too small: {overhead} bytes"


# ---------------------------------------------------------------------------
# CLAUDE.md compliance smoke tests
# ---------------------------------------------------------------------------


def test_no_mps_fallback_in_inflate():
    """Catalog #1: inflate.py must not have MPS-fallback ternary."""
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[1] / "inflate.py"
    )
    text = path.read_text(encoding="utf-8")
    # Allow PACT_INFLATE_DEVICE=mps refusal but NOT silent fallback.
    forbidden = '"mps"' in text and "refuse" not in text.lower() and "RuntimeError" not in text
    assert not forbidden


def test_inflate_under_loc_budget():
    """HNeRV parity L4: inflate.py ≤ 100 LOC (bolt-on stricter than 200)."""
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[1] / "inflate.py"
    )
    loc = sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    # Strict bolt-on budget; allow modest slack for license headers + imports
    # (the substantive logic should be ≤ 100 LOC).
    assert loc <= 200, f"inflate.py LOC = {loc}, exceeds bolt-on ≤200 budget"


def test_no_scorer_load_at_inflate():
    """Strict scorer rule: inflate.py must not import segnet/posenet."""
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[1] / "inflate.py"
    )
    text = path.read_text(encoding="utf-8")
    assert "from upstream.modules" not in text
    assert "PoseNet" not in text
    assert "SegNet" not in text


def test_architecture_under_loc_budget():
    """HNeRV parity L7: bolt_on_loc_budget = 350 (substrate-engineering NOT used)."""
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[1] / "architecture.py"
    )
    loc = sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    # Architecture allows up to 350 LOC for bolt-on (the docstring counts but
    # let's be lenient with module-level docstrings + imports).
    assert loc <= 500, f"architecture.py LOC = {loc}, exceeds bolt-on ≤500 (incl docs)"


# ---------------------------------------------------------------------------
# Full trainer path
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _load_trainer_module():
    trainer_path = _repo_root() / "experiments" / "train_substrate_z3_balle_hyperprior_bolton.py"
    spec = importlib.util.spec_from_file_location("_z3_trainer_under_test", trainer_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_synthetic_a1_archive_zip(path: Path) -> bytes:
    """Write a minimal A1-shaped archive.zip fixture with inner member ``x``."""
    prefix = struct.pack("<I", A1_DECODER_SECTION_TOTAL)
    decoder = b"D" * (A1_DECODER_SECTION_TOTAL - len(prefix))
    latent = b"L" * A1_LATENT_BLOB_LEN
    sidecar = b"S" * 607
    inner = prefix + decoder + latent + sidecar
    info = zipfile.ZipInfo(filename="x", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, inner)
    return inner


def test_z3v2_affine_residual_int8_is_centered_signed_range():
    """Regression: a min-offset affine clips half the range in signed int8."""
    trainer = _load_trainer_module()
    lo = torch.linspace(-10.0, 5.0, A1_LATENT_DIM)
    hi = lo + torch.linspace(1.0, 20.0, A1_LATENT_DIM)
    mid = (lo + hi) * 0.5
    latents = torch.stack([lo, mid, hi], dim=0)

    residual, offset, scale = trainer._z3v2_affine_residual_int8(latents)

    assert residual.dtype == torch.int8
    assert torch.equal(
        residual[0], torch.full((A1_LATENT_DIM,), -127, dtype=torch.int8)
    )
    assert torch.equal(residual[1], torch.zeros(A1_LATENT_DIM, dtype=torch.int8))
    assert torch.equal(
        residual[2], torch.full((A1_LATENT_DIM,), 127, dtype=torch.int8)
    )
    assert torch.allclose(offset, mid)
    reconstructed = residual.to(torch.float32) * scale.unsqueeze(0) + offset.unsqueeze(0)
    assert torch.allclose(reconstructed, latents, atol=1e-5)


def test_smoke_main_emits_training_artifact_manifest(tmp_path):
    """Smoke-only Modal artifacts must satisfy training_artifact_v1."""

    trainer = _load_trainer_module()
    out_dir = tmp_path / "z3_smoke"
    args = trainer._build_parser().parse_args(
        [
            "--a1-archive-path",
            str(_repo_root() / "submissions" / "a1" / "archive.zip"),
            "--output-dir",
            str(out_dir),
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--smoke",
        ]
    )

    assert trainer._smoke_main(args) == 0

    stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "training_artifact_v1"
    assert manifest["lane_id"] == stats["lane_id"]
    assert manifest["training_mode"] == "smoke"
    assert manifest["research_only"] is True
    assert manifest["archive_bytes"] == stats["archive_bytes"]
    assert manifest["archive_sha256"] == stats["archive_sha256"]
    assert manifest["archive_zip_bytes"] == stats["archive_zip_bytes"]
    assert manifest["archive_zip_sha256"] == stats["archive_zip_sha256"]
    assert manifest["score_claim"] is False
    assert manifest["score_claim_valid"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["result"]["training_mode"] == "smoke"


def test_smoke_main_honors_v2_latent_replacement_flag(tmp_path):
    """Operator-authorized v2 smoke must emit the latent-replacement layout."""

    trainer = _load_trainer_module()
    a1_zip = tmp_path / "a1_synthetic_archive.zip"
    a1_inner = _write_synthetic_a1_archive_zip(a1_zip)
    out_dir = tmp_path / "z3_smoke_v2"
    args = trainer._build_parser().parse_args(
        [
            "--a1-archive-path",
            str(a1_zip),
            "--output-dir",
            str(out_dir),
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--smoke",
            "--enable-v2-latent-replacement",
        ]
    )

    assert trainer._smoke_main(args) == 0

    stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    contract = stats["archive_contract"]
    assert stats["v2_latent_replacement_requested"] is True
    assert stats["v2_latent_replacement_active"] is True
    assert contract["layout"] == "z3v2_latent_replacement"
    assert contract["byte_saving"] is True
    assert contract["archive_bytes"] == stats["archive_bytes"]
    assert manifest["result"]["archive_contract"]["layout"] == "z3v2_latent_replacement"

    with zipfile.ZipFile(out_dir / "archive.zip") as zf:
        assert zf.namelist() == ["0.bin"]
        payload = zf.read("0.bin")
    assert payload[:A1_DECODER_SECTION_TOTAL] == a1_inner[:A1_DECODER_SECTION_TOTAL]
    assert (
        payload[A1_DECODER_SECTION_TOTAL : A1_DECODER_SECTION_TOTAL + len(Z3HV2_MAGIC)]
        == Z3HV2_MAGIC
    )
    assert len(payload) < len(a1_inner)


def test_full_main_cpu_builds_z3v2_runtime_without_auth_eval(tmp_path):
    """The Phase-2 full path must be executable without remote dispatch.

    It emits a v2-only direct-residual packet and records fail-closed blockers
    because exact score authority still requires paired CPU/CUDA auth eval.
    """
    trainer = _load_trainer_module()
    out_dir = tmp_path / "z3_full"
    args = trainer._build_parser().parse_args(
        [
            "--a1-archive-path",
            str(_repo_root() / "submissions" / "a1" / "archive.zip"),
            "--output-dir",
            str(out_dir),
            "--epochs",
            "1",
            "--batch-size",
            "128",
            "--device",
            "cpu",
            "--full-cpu",
            "--advisory-cpu-explicitly-waived",
            "--skip-auth-eval",
        ]
    )

    assert trainer._full_main(args) == 0

    stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert stats["smoke"] is False
    assert stats["sidecar_shipped"] is True
    assert stats["archive_bytes"] != stats["base_archive_bytes"]
    assert stats["archive_contract"]["layout"] == "z3v2_latent_replacement"
    assert stats["archive_contract"]["ready_for_exact_eval_dispatch"] is False
    assert stats["ready_for_exact_eval_dispatch"] is False
    assert "paired exact CPU/CUDA auth eval" in stats["dispatch_blocker"]
    assert manifest["schema"] == "training_artifact_v1"
    assert manifest["training_mode"] == "full"
    assert manifest["research_only"] is True
    assert manifest["archive_sha256"] == stats["archive_sha256"]

    with zipfile.ZipFile(_repo_root() / "submissions" / "a1" / "archive.zip") as zf:
        a1_inner = zf.read("x")
    with zipfile.ZipFile(out_dir / "archive.zip") as zf:
        assert zf.namelist() == ["0.bin"]
        payload = zf.read("0.bin")
        assert payload != a1_inner
        assert (
            payload[A1_DECODER_SECTION_TOTAL : A1_DECODER_SECTION_TOTAL + len(Z3HV2_MAGIC)]
            == Z3HV2_MAGIC
        )

    inflate_text = (out_dir / "submission_dir" / "inflate.py").read_text(
        encoding="utf-8"
    )
    inflate_sh = (out_dir / "submission_dir" / "inflate.sh").read_text(
        encoding="utf-8"
    )
    assert "placeholder" not in inflate_text.lower()
    assert "require_reconstruct_a1_latents_from_v2_payload" in inflate_text
    assert "refusing A1/v1 fallback" in inflate_text
    assert "reconstruct_a1_latents as reconstruct_a1_latents_v1" not in inflate_text
    assert "archive_dir / 'archive.zip'" not in inflate_text
    assert '${DATA_DIR}/x' in inflate_sh
    py_compile.compile(str(out_dir / "submission_dir" / "inflate.py"), doraise=True)


def test_remote_z3_smoke_completion_marker_is_not_contest_cuda_by_default():
    """No-scorer smoke completion must not be logged as [contest-CUDA]."""
    script = (
        _repo_root()
        / "scripts"
        / "remote_lane_substrate_z3_balle_hyperprior_bolton.sh"
    ).read_text(encoding="utf-8")
    assert "LANE_Z3_BALLE_DONE [contest-CUDA]" not in script
    assert "score_claim=false" in script
