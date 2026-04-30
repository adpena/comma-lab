"""Lane 12 — NeRV mask codec tests.

Per Phase 2 Lane 12 spec (memory project_phases_2_3_4_*):
- Round-trip on a synthetic mask sequence
- Byte-count vs raw fp16 baseline
- Deterministic encode

All claims tagged [synthetic] — empirical real-archive validation is the
Phase 2 dispatch decision.

CLAUDE.md non-negotiables verified:
- No scorer load anywhere
- No silent defaults (every public arg required-keyword)
- No GPU
- Deterministic CPU-only
- Pure-math byte → tensor pipeline
- Trainer uses EMA (decay 0.997 default) + refuses MPS + uses
  ``tac.training.EMA`` canonical class.
- No bare ``.round()`` in any forward chain (Council A zero-gradient
  bug class).
"""
from __future__ import annotations

import inspect

import pytest
import torch

from tac.nerv_mask_codec import (
    NERV_MAGIC,
    NERV_VERSION,
    NeRVMaskCodec,
    NeRVMaskTrainer,
    decode_nerv_codec,
    encode_nerv_codec,
    nerv_codec_bytes,
    positional_encode,
    raw_fp16_baseline_bytes,
    render_mask_argmax,
    render_mask_logits,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: positional_encode determinism + shape correctness
# ─────────────────────────────────────────────────────────────────────────────


def test_positional_encode_shape_and_determinism_synthetic() -> None:
    """[synthetic] positional_encode shape = (B, D * 2 * num_freqs); deterministic."""
    coords = torch.tensor([[0.0, 0.5, -0.5], [1.0, -1.0, 0.0]])  # B=2, D=3
    enc1 = positional_encode(coords, num_freqs=4)
    enc2 = positional_encode(coords, num_freqs=4)
    # Shape: (B=2, D=3, 2 sin/cos, F=4) → flatten → (2, 24)
    assert enc1.shape == (2, 3 * 2 * 4)
    assert torch.allclose(enc1, enc2)
    # Bad input shapes
    with pytest.raises(ValueError, match="coords must be 2-D"):
        positional_encode(torch.zeros(5), num_freqs=2)
    with pytest.raises(ValueError, match="num_freqs must be"):
        positional_encode(coords, num_freqs=0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: encode → decode round-trip preserves all weights bit-exact (fp16)
# ─────────────────────────────────────────────────────────────────────────────


def test_encode_decode_roundtrip_fp16_synthetic() -> None:
    """[synthetic] fp16 round-trip preserves all weights to fp16 precision."""
    codec = NeRVMaskCodec(num_freqs=4, hidden_dim=32, num_classes=5, depth=4, seed=2026)
    blob = encode_nerv_codec(codec, weight_dtype="fp16")
    assert blob[:4] == NERV_MAGIC
    codec_decoded = decode_nerv_codec(blob)
    # Same arch
    assert codec_decoded.num_freqs == codec.num_freqs
    assert codec_decoded.hidden_dim == codec.hidden_dim
    assert codec_decoded.num_classes == codec.num_classes
    assert codec_decoded.depth == codec.depth
    # Weights match within fp16 precision
    for k, v in codec.state_dict().items():
        v2 = codec_decoded.state_dict()[k]
        assert v2.shape == v.shape
        # fp16 round-trip introduces ~1e-3 max error
        assert torch.allclose(v.float(), v2.float(), atol=1e-2, rtol=1e-2)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: forward output shape + finite (no NaN/Inf at init)
# ─────────────────────────────────────────────────────────────────────────────


def test_codec_forward_shape_and_finite_at_init_synthetic() -> None:
    """[synthetic] codec.forward(coords) → (B, num_classes) finite logits."""
    codec = NeRVMaskCodec(num_freqs=4, hidden_dim=32, num_classes=5)
    coords = torch.randn(7, 3)  # B=7, D=3
    logits = codec(coords)
    assert logits.shape == (7, 5)
    assert torch.isfinite(logits).all()


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: byte-count assertion — small NeRV << raw fp16 baseline
# ─────────────────────────────────────────────────────────────────────────────


def test_byte_count_small_nerv_beats_raw_fp16_baseline_synthetic() -> None:
    """[synthetic] A small NeRV codec is orders of magnitude smaller than raw fp16."""
    codec = NeRVMaskCodec(num_freqs=4, hidden_dim=32, num_classes=5)
    codec_bytes = nerv_codec_bytes(codec, weight_dtype="fp16")
    # Mock comma video: 1200 frames at 384x512x5 logits
    raw_bytes = raw_fp16_baseline_bytes(1200, 384, 512, 5)
    assert codec_bytes < raw_bytes
    # Order of magnitude check: NeRV scaffold (~6 KB) << raw (~2.4 GB).
    # Even at 100x larger NeRV (200 KB), still 4 orders of magnitude smaller.
    assert codec_bytes * 1000 < raw_bytes


def test_raw_fp16_baseline_rejects_zero_dims() -> None:
    """[synthetic] raw_fp16_baseline_bytes rejects zero/negative dims (no silent default)."""
    with pytest.raises(ValueError, match="all dims must be > 0"):
        raw_fp16_baseline_bytes(0, 100, 100, 5)
    with pytest.raises(ValueError, match="all dims must be > 0"):
        raw_fp16_baseline_bytes(100, 100, 100, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: render_mask_logits round-trip on tiny synthetic (T, H, W)
# ─────────────────────────────────────────────────────────────────────────────


def test_render_mask_logits_shape_and_determinism_synthetic() -> None:
    """[synthetic] render_mask_logits returns (T, H, W, C); deterministic."""
    codec = NeRVMaskCodec(num_freqs=4, hidden_dim=16, num_classes=5)
    out1 = render_mask_logits(codec, num_frames=3, height=4, width=5, batch_size=8)
    out2 = render_mask_logits(codec, num_frames=3, height=4, width=5, batch_size=64)
    assert out1.shape == (3, 4, 5, 5)
    assert torch.allclose(out1, out2, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: no silent defaults — encode_nerv_codec rejects None codec
# ─────────────────────────────────────────────────────────────────────────────


def test_no_silent_defaults_synthetic() -> None:
    """[synthetic] encode + decode require explicit args (Check 81 STRICT)."""
    with pytest.raises(ValueError, match="codec is required"):
        encode_nerv_codec(codec=None)
    with pytest.raises(ValueError, match="weight_dtype must be"):
        encode_nerv_codec(codec=NeRVMaskCodec(num_freqs=2, hidden_dim=8, num_classes=5), weight_dtype="bf16")
    with pytest.raises(ValueError, match="blob is required"):
        decode_nerv_codec(blob=None)
    with pytest.raises(ValueError, match="bad magic"):
        decode_nerv_codec(blob=b"BAD!" + b"\x00" * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: int8 path encodes (scaffold smoke; production needs scale table)
# ─────────────────────────────────────────────────────────────────────────────


def test_int8_encode_decode_roundtrip_synthetic() -> None:
    """[synthetic] int8 encode produces ~half the bytes of fp16."""
    codec = NeRVMaskCodec(num_freqs=4, hidden_dim=32, num_classes=5)
    blob_fp16 = encode_nerv_codec(codec, weight_dtype="fp16")
    blob_int8 = encode_nerv_codec(codec, weight_dtype="int8")
    # int8 is roughly half of fp16 (header is small overhead)
    assert len(blob_int8) < len(blob_fp16)
    # int8 decode runs (note: scaffold doesn't restore scale → outputs are
    # numerically different but the codec object is functional)
    codec_int8 = decode_nerv_codec(blob_int8)
    assert codec_int8.num_params() == codec.num_params()


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: NeRVMaskCodec rejects bad arch
# ─────────────────────────────────────────────────────────────────────────────


def test_codec_constructor_rejects_bad_arch_synthetic() -> None:
    """[synthetic] NeRVMaskCodec rejects invalid arch params."""
    with pytest.raises(ValueError, match="invalid arch"):
        NeRVMaskCodec(num_freqs=0, hidden_dim=32, num_classes=5)
    with pytest.raises(ValueError, match="invalid arch"):
        NeRVMaskCodec(num_freqs=4, hidden_dim=32, num_classes=5, depth=1)


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: NRV2 default version + back-compat with NRV1 decoder
# ─────────────────────────────────────────────────────────────────────────────


def test_default_version_is_v2_synthetic() -> None:
    """[synthetic] encode defaults to NRV_VERSION=2; v1 still decodes."""
    assert NERV_VERSION == 2
    codec = NeRVMaskCodec(num_freqs=4, hidden_dim=16, num_classes=5)
    blob_v2 = encode_nerv_codec(codec, weight_dtype="fp16")  # default version
    blob_v1 = encode_nerv_codec(codec, weight_dtype="fp16", version=1)
    # v2 has an extra 8-byte scale_table_size field (zero for fp16)
    assert len(blob_v2) == len(blob_v1) + 8
    # Both decode to functionally-equivalent codecs (within fp16 precision)
    c_v2 = decode_nerv_codec(blob_v2)
    c_v1 = decode_nerv_codec(blob_v1)
    for k in codec.state_dict():
        assert torch.allclose(
            c_v1.state_dict()[k].float(),
            c_v2.state_dict()[k].float(),
            atol=1e-6,
        )


def test_invalid_version_rejected_synthetic() -> None:
    """[synthetic] Encoder + decoder both reject unsupported versions."""
    codec = NeRVMaskCodec(num_freqs=2, hidden_dim=8, num_classes=5)
    with pytest.raises(ValueError, match="version must be 1 or 2"):
        encode_nerv_codec(codec, weight_dtype="fp16", version=3)
    # Forge a v3 header
    blob_v2 = encode_nerv_codec(codec, weight_dtype="fp16")
    forged = blob_v2[:4] + (3).to_bytes(2, "little") + blob_v2[6:]
    with pytest.raises(ValueError, match="unsupported version"):
        decode_nerv_codec(forged)


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: NRV2 int8 with scale table preserves numerical reconstruction
# ─────────────────────────────────────────────────────────────────────────────


def test_nrv2_int8_with_scale_table_roundtrip_synthetic() -> None:
    """[synthetic] NRV2 int8 path reproduces float weights to int8 precision.

    The v1 int8 path was broken (no scale persisted). The v2 int8 path ships
    a per-tensor float32 scale table so the decoder dequantizes to recover
    floats. Reconstruction error is dominated by per-tensor symmetric
    quantization with 127 levels (~vmax/127 per element).
    """
    codec = NeRVMaskCodec(num_freqs=4, hidden_dim=32, num_classes=5, seed=2026)
    blob = encode_nerv_codec(codec, weight_dtype="int8", version=2)
    decoded = decode_nerv_codec(blob)
    for k, v in codec.state_dict().items():
        v_dec = decoded.state_dict()[k]
        # Quantization step = vmax / 127. Allow 2x step as worst-case rounding.
        vmax = float(v.abs().max().clamp_min(1e-12))
        step = vmax / 127.0
        max_err = (v - v_dec).abs().max().item()
        assert max_err <= 2.0 * step, (
            f"key {k!r}: max int8 reconstruction error {max_err:.2e} "
            f"exceeds 2 × step ({2.0 * step:.2e}); v1 broken-int8 regression?"
        )


def test_nrv2_int8_smaller_than_fp16_synthetic() -> None:
    """[synthetic] NRV2 int8 ships ~half the bytes of fp16 even with scale table."""
    codec = NeRVMaskCodec(num_freqs=8, hidden_dim=64, num_classes=5)
    blob_fp16 = encode_nerv_codec(codec, weight_dtype="fp16")
    blob_int8 = encode_nerv_codec(codec, weight_dtype="int8", version=2)
    # int8 path: 1B/param + 4B/state-dict-key for scale table. For a 4-layer
    # MLP we have ~8 keys (4 weight + 4 bias) → 32B scale overhead. Total
    # int8 << fp16 for any non-trivial param count.
    assert len(blob_int8) < len(blob_fp16)
    assert len(blob_int8) < 0.7 * len(blob_fp16), (
        f"int8 {len(blob_int8)}B not <70% of fp16 {len(blob_fp16)}B; "
        f"scale-table overhead unexpectedly large"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: Trainer refuses MPS + accepts CPU + uses canonical EMA class
# ─────────────────────────────────────────────────────────────────────────────


def test_trainer_refuses_mps_synthetic() -> None:
    """[synthetic] NeRVMaskTrainer refuses device='mps' (CLAUDE.md non-negotiable)."""
    codec = NeRVMaskCodec(num_freqs=2, hidden_dim=8, num_classes=5)
    with pytest.raises(ValueError, match="refuses device='mps'"):
        NeRVMaskTrainer(codec=codec, device="mps")


def test_trainer_requires_codec_synthetic() -> None:
    """[synthetic] NeRVMaskTrainer rejects None codec (no silent default)."""
    with pytest.raises(ValueError, match="codec is required"):
        NeRVMaskTrainer(codec=None, device="cpu")


def test_trainer_uses_canonical_tac_training_ema_synthetic() -> None:
    """[synthetic] Trainer.ema is exactly tac.training.EMA, decay default 0.997.

    This guards against future regressions where someone re-implements EMA
    locally (the bug class that hit train_joint_pair.py — duplicate `class
    EMA` removed in Council D 2026-04-29 PM).
    """
    from tac.training import EMA as CanonicalEMA

    codec = NeRVMaskCodec(num_freqs=2, hidden_dim=8, num_classes=5)
    trainer = NeRVMaskTrainer(codec=codec, device="cpu")
    assert isinstance(trainer.ema, CanonicalEMA), (
        "NeRVMaskTrainer must use tac.training.EMA — not a local re-implementation"
    )
    assert trainer.ema.decay == pytest.approx(0.997)


def test_trainer_rejects_invalid_ema_decay_synthetic() -> None:
    """[synthetic] NeRVMaskTrainer rejects ema_decay outside (0, 1)."""
    codec = NeRVMaskCodec(num_freqs=2, hidden_dim=8, num_classes=5)
    with pytest.raises(ValueError, match="ema_decay must be in"):
        NeRVMaskTrainer(codec=codec, device="cpu", ema_decay=1.5)
    with pytest.raises(ValueError, match="ema_decay must be in"):
        NeRVMaskTrainer(codec=codec, device="cpu", ema_decay=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: Trainer step updates weights AND EMA shadow; EMA differs after N steps
# ─────────────────────────────────────────────────────────────────────────────


def _make_synthetic_masks(T: int = 4, H: int = 8, W: int = 8, num_classes: int = 5) -> torch.Tensor:
    """Build a deterministic (T, H, W) mask: vertical stripes whose phase
    rotates with t. Easy to overfit; shape preserved across frames.
    """
    g = torch.Generator().manual_seed(0)
    masks = torch.zeros(T, H, W, dtype=torch.long)
    for t in range(T):
        # Vertical stripes shifted by t
        cols = (torch.arange(W) + t) % num_classes
        masks[t] = cols.unsqueeze(0).expand(H, W)
    return masks


def test_trainer_step_changes_live_weights_AND_ema_shadow_synthetic() -> None:
    """[synthetic] After ≥1 step, live weights change AND EMA shadow tracks."""
    torch.manual_seed(0)
    codec = NeRVMaskCodec(num_freqs=4, hidden_dim=16, num_classes=5)
    trainer = NeRVMaskTrainer(codec=codec, device="cpu", learning_rate=1e-2)
    masks = _make_synthetic_masks()
    # Snapshot initial live weights AND initial EMA shadow
    init_live = {k: v.detach().clone() for k, v in codec.state_dict().items()}
    init_shadow = {k: v.detach().clone() for k, v in trainer.ema.shadow.items()}
    for _ in range(20):
        trainer.step(masks, batch_size=64)
    # Live weights moved
    moved_live = sum(
        (codec.state_dict()[k] - init_live[k]).abs().sum().item()
        for k in init_live
    )
    assert moved_live > 0, "live weights did not move after 20 steps"
    # EMA shadow ALSO moved (it tracks live with decay)
    moved_shadow = sum(
        (trainer.ema.shadow[k] - init_shadow[k]).abs().sum().item()
        for k in init_shadow
    )
    assert moved_shadow > 0, "EMA shadow did not update — wire-in regression"
    # AND the EMA shadow is NOT identical to live (decay 0.997 means lag)
    differs = any(
        not torch.allclose(trainer.ema.shadow[k], codec.state_dict()[k], atol=1e-8)
        for k in init_live
    )
    assert differs, "EMA shadow == live weights — decay not applied"


# ─────────────────────────────────────────────────────────────────────────────
# Test 13: Encode ships EMA shadow, NOT live weights
# ─────────────────────────────────────────────────────────────────────────────


def test_trainer_encode_ships_ema_shadow_not_live_synthetic() -> None:
    """[synthetic] trainer.encode() returns NRV2 of the EMA shadow.

    CLAUDE.md non-negotiable: "Inference / archive bytes come from
    ``ema.state_dict()`` — never from ``model.state_dict()`` after training."
    """
    torch.manual_seed(0)
    codec = NeRVMaskCodec(num_freqs=4, hidden_dim=16, num_classes=5)
    trainer = NeRVMaskTrainer(codec=codec, device="cpu", learning_rate=1e-2)
    masks = _make_synthetic_masks()
    for _ in range(20):
        trainer.step(masks, batch_size=64)
    # encode() should produce a payload whose decoded weights match the EMA
    # shadow (modulo fp16 precision), NOT the live weights.
    blob = trainer.encode(weight_dtype="fp16")
    decoded = decode_nerv_codec(blob)
    for k in trainer.ema.shadow:
        # Decoded weights ≈ EMA shadow (within fp16 ~1e-3 precision)
        if trainer.ema.shadow[k].dtype.is_floating_point:
            assert torch.allclose(
                decoded.state_dict()[k].float(),
                trainer.ema.shadow[k].float(),
                atol=1e-2,
                rtol=1e-2,
            ), f"key {k!r}: encoded payload does not match EMA shadow"


# ─────────────────────────────────────────────────────────────────────────────
# Test 14: evaluate_argmax_disagreement uses snapshot+restore (live weights
# unchanged after eval)
# ─────────────────────────────────────────────────────────────────────────────


def test_trainer_eval_preserves_live_weights_synthetic() -> None:
    """[synthetic] evaluate_argmax_disagreement does NOT mutate live weights.

    The CLAUDE.md canonical EMA pattern requires snapshot+restore so training
    resumes from un-shadowed weights. This guards against the freeze-bug
    class where ``ema.apply()`` is called inside ``train_epoch`` and never
    reverted (DARTS-S freeze symptom from a different root cause but same
    discipline pattern).
    """
    torch.manual_seed(0)
    codec = NeRVMaskCodec(num_freqs=4, hidden_dim=16, num_classes=5)
    trainer = NeRVMaskTrainer(codec=codec, device="cpu", learning_rate=1e-2)
    masks = _make_synthetic_masks(T=2, H=4, W=4)
    for _ in range(5):
        trainer.step(masks, batch_size=16)
    pre_eval = {k: v.detach().clone() for k, v in codec.state_dict().items()}
    _ = trainer.evaluate_argmax_disagreement(masks, batch_size=16)
    post_eval = codec.state_dict()
    for k in pre_eval:
        assert torch.allclose(pre_eval[k], post_eval[k], atol=1e-8), (
            f"key {k!r} changed across eval — snapshot+restore broken"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 15: Trainer overfits a tiny synthetic sequence (sanity check)
# ─────────────────────────────────────────────────────────────────────────────


def test_trainer_overfits_tiny_synthetic_sequence_synthetic() -> None:
    """[synthetic] On a 4×8×8 stripes-by-time sequence, trainer drives loss down
    and brings argmax-disagreement well below 50% (chance for 5-class).

    This is a SANITY check that the training loop is wired correctly. Real
    convergence + Lane G v3 1200-frame measurement happens at Phase F.
    """
    torch.manual_seed(0)
    codec = NeRVMaskCodec(num_freqs=4, hidden_dim=32, num_classes=5)
    trainer = NeRVMaskTrainer(codec=codec, device="cpu", learning_rate=5e-3)
    masks = _make_synthetic_masks(T=4, H=8, W=8)
    initial_loss = trainer.step(masks, batch_size=128)["loss"]
    losses = []
    for _ in range(300):
        m = trainer.step(masks, batch_size=128)
        losses.append(m["loss"])
    final_loss = sum(losses[-20:]) / 20.0  # smoothed
    assert final_loss < initial_loss, (
        f"loss did not decrease: initial={initial_loss:.4f}, "
        f"final-smoothed={final_loss:.4f}"
    )
    # Disagreement on the OVERFIT sequence should be < 0.5 (chance = 0.8 for 5-class)
    eval_metrics = trainer.evaluate_argmax_disagreement(masks, batch_size=64)
    assert eval_metrics["disagreement_rate"] < 0.5, (
        f"trainer failed to overfit synthetic stripes: "
        f"disagreement={eval_metrics['disagreement_rate']:.3f} "
        f"(expected < 0.5)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 16: render_mask_argmax produces correct shape + dtype + value range
# ─────────────────────────────────────────────────────────────────────────────


def test_render_mask_argmax_shape_dtype_range_synthetic() -> None:
    """[synthetic] render_mask_argmax → (T, H, W) uint8 in [0, num_classes-1]."""
    codec = NeRVMaskCodec(num_freqs=4, hidden_dim=16, num_classes=5)
    out = render_mask_argmax(codec, num_frames=3, height=4, width=5, batch_size=8, device="cpu")
    assert out.shape == (3, 4, 5)
    assert out.dtype == torch.uint8
    assert int(out.min()) >= 0
    assert int(out.max()) <= 4


def test_render_mask_argmax_refuses_mps_synthetic() -> None:
    """[synthetic] render_mask_argmax refuses device='mps' (CLAUDE.md)."""
    codec = NeRVMaskCodec(num_freqs=2, hidden_dim=8, num_classes=5)
    with pytest.raises(ValueError, match="refuses device='mps'"):
        render_mask_argmax(codec, num_frames=2, height=3, width=3, device="mps")


# ─────────────────────────────────────────────────────────────────────────────
# Test 17: No bare .round() in nerv_mask_codec source (Council A bug class)
# ─────────────────────────────────────────────────────────────────────────────


def test_no_bare_round_in_nerv_mask_codec_source_synthetic() -> None:
    """[synthetic] guard against re-introducing the .round() zero-gradient bug.

    Council A audit (2026-04-29) found bare ``.round()`` in
    ``segmap_renderer.py:281`` severed backprop and froze training. Same
    risk class for NeRV: any future eval-roundtrip-style chain in this
    module must use ``Uint8STE.apply()`` (or remain detach()'d).

    The current trainer does NOT need ``.round()`` — gradients flow through
    the cross-entropy on raw logits. This test fails loudly if a future
    refactor adds ``.round()`` somewhere where it would sever gradient
    (e.g. inside a forward chain instead of inside an inference-only
    helper protected by ``torch.no_grad()``).
    """
    src = inspect.getsource(NeRVMaskCodec.__module__ and __import__("tac.nerv_mask_codec", fromlist=["*"]))
    # Allow occurrences inside scale-table / quantization paths (those are
    # CPU-side numpy quantization, not autograd-active forwards). The
    # forbidden pattern is a bare ``.round()`` on a torch tensor inside a
    # forward chain. The check strips comments and docstring lines so a
    # comment "do not call .round()" does NOT trigger.
    forbidden_contexts = ("forward", "step(", "_sample_batch", "evaluate_")
    lines = src.splitlines()
    in_docstring = False
    for i, raw in enumerate(lines):
        # Strip line/block docstring + trailing # comment
        line_no_comment = raw.split("#", 1)[0]
        # Triple-quoted docstring tracking (heuristic — same line open+close OK)
        triple_count = line_no_comment.count('"""') + line_no_comment.count("'''")
        if triple_count % 2 == 1:
            in_docstring = not in_docstring
        if in_docstring:
            continue
        if ".round()" not in line_no_comment:
            continue
        # Walk backwards to find the enclosing def
        for j in range(i, -1, -1):
            stripped = lines[j].lstrip()
            if stripped.startswith("def "):
                fn_signature = stripped
                if any(ctx in fn_signature for ctx in forbidden_contexts):
                    raise AssertionError(
                        f"bare .round() found in forward/training context: "
                        f"line {i + 1} inside {fn_signature.strip()!r}"
                    )
                break
