"""Tests for Lane C UNIWARD δ codec + apply path.

CPU-only. No Vast.ai, no scorer loading, no GT video. These tests cover
the codec invariants the inflate path relies on.

Coverage matrix:
  - Round-trip: pack(δ) → unpack → reconstructed equals quantized δ.
  - L∞ budget: every kept value within ±l_inf_budget.
  - Sparsity:  n_kept ≤ requested top-K (target_bytes path) AND
               n_kept matches the number of non-zero pixel-channels in
               the source δ when we feed in a known-sparse input.
  - Determinism: same inputs → exact same bytes (zlib + int8 quant +
                 deterministic top-K).
  - Apply: scatter_add gives the expected per-pixel delta on a known
           input; respects [0, 255] clamp.
  - Magic + schema: bad bytes raise ValueError with a useful message.
  - Frame size mismatch raises ValueError (catches the historical
    "ship at one resolution, inflate at another" failure mode).
"""
from __future__ import annotations

import zlib

import numpy as np
import pytest
import torch

from tac.uniward_delta import (
    DeltaSpec,
    MAGIC,
    SCHEMA_VERSION,
    apply_delta_to_frame,
    pack_sparse_delta,
    unpack_sparse_delta,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _random_delta(
    n_frames: int = 4, H: int = 16, W: int = 24, sparsity: float = 0.05,
    seed: int = 0, l_inf: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a (B, 3, H, W) δ with `sparsity` fraction of non-zero
    pixel-channels, drawn uniformly in [-l_inf, l_inf]. Returns (δ, cost).
    """
    g = torch.Generator().manual_seed(seed)
    n_total = n_frames * 3 * H * W
    n_nz = max(int(n_total * sparsity), 1)
    delta = torch.zeros(n_total, dtype=torch.float32)
    perm = torch.randperm(n_total, generator=g)
    delta[perm[:n_nz]] = (torch.rand(n_nz, generator=g) * 2 - 1) * l_inf
    delta = delta.reshape(n_frames, 3, H, W)
    # Cost map: smooth random (positive). Doesn't materially affect
    # round-trip correctness but exercises the rank-boost code path.
    cost = torch.rand(n_frames, H, W, generator=g)
    return delta, cost


# ── Round-trip ───────────────────────────────────────────────────────────


def test_roundtrip_basic_recovers_quantized_delta() -> None:
    """pack → unpack → apply against zero frame should reconstruct
    the quantized δ at every kept position, and zero elsewhere."""
    delta, cost = _random_delta(n_frames=3, H=8, W=12, sparsity=0.10, seed=1)
    blob = pack_sparse_delta(
        delta, cost, l_inf_budget=4.0, target_bytes=None,
    )
    spec = unpack_sparse_delta(blob)
    # Apply each frame to a zero base. The result should match the quant
    # snapshot of δ at the kept positions (others = 0).
    for f in range(spec.n_frames):
        zero = torch.zeros(spec.H, spec.W, 3, dtype=torch.float32)
        out = apply_delta_to_frame(zero, spec, f)
        # Construct the expected mask from the spec
        local = spec.per_frame_local_idx[f]
        if local is None:
            assert torch.allclose(out, zero)
            continue
        flat_out = out.reshape(-1)
        # Every recovered value must equal the dequant value (or its
        # int8 representable value), within a half-step of the scale.
        recovered = flat_out[local]
        expected = spec.per_frame_dequant[f]
        # The applied value is dequant clamped into [0, 255]. Since base
        # is zero, negative values clip to 0.
        # Compare against clamp(expected, 0, 255) at the kept positions.
        assert torch.allclose(recovered, expected.clamp(0.0, 255.0))


def test_roundtrip_byte_stable_for_identical_inputs() -> None:
    """Same δ + cost + budget → same bytes. Determinism non-negotiable."""
    delta, cost = _random_delta(n_frames=2, H=8, W=8, sparsity=0.20, seed=42)
    blob_a = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=None, seed=7)
    blob_b = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=None, seed=7)
    assert blob_a == blob_b, "pack must be byte-stable for identical inputs"


def test_roundtrip_preserves_magic_and_schema() -> None:
    """Decompressed payload must start with the wire-format magic, and
    the embedded header must match the current schema version."""
    delta, cost = _random_delta(n_frames=1, H=4, W=4, sparsity=0.5)
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0)
    raw = zlib.decompress(blob)
    assert raw[:4] == MAGIC
    import json
    import struct
    header_len = struct.unpack("<I", raw[4:8])[0]
    header = json.loads(raw[8:8 + header_len].decode("utf-8"))
    assert header["schema_version"] == SCHEMA_VERSION


# ── L∞ budget ────────────────────────────────────────────────────────────


def test_l_inf_budget_is_respected_after_quantization() -> None:
    """Every dequantized δ value must satisfy |δ| ≤ l_inf_budget,
    even after int8 round-trip. The scale is l_inf_budget so the
    representable range is exactly [-l_inf_budget * 127/127, +l_inf_budget].
    """
    # Construct δ that EXCEEDS the budget — pack must clip.
    delta = torch.full((1, 3, 4, 4), 100.0)  # well above any reasonable budget
    cost = torch.ones(1, 4, 4)
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0)
    spec = unpack_sparse_delta(blob)
    for f in range(spec.n_frames):
        if spec.per_frame_dequant[f] is None:
            continue
        max_abs = float(spec.per_frame_dequant[f].abs().max().item())
        assert max_abs <= 4.0 + 1e-6, (
            f"|δ| max = {max_abs} exceeds l_inf_budget=4.0"
        )


def test_l_inf_budget_zero_raises() -> None:
    delta, cost = _random_delta()
    with pytest.raises(ValueError, match="l_inf_budget"):
        pack_sparse_delta(delta, cost, l_inf_budget=0.0)


# ── Sparsity / target_bytes ──────────────────────────────────────────────


def test_target_bytes_caps_blob_size() -> None:
    """The binary search MUST respect the byte budget — the resulting
    blob is guaranteed ≤ target_bytes."""
    delta, cost = _random_delta(n_frames=4, H=32, W=48, sparsity=1.0, seed=3)
    target = 500
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=target)
    assert len(blob) <= target, f"blob {len(blob)} exceeds target_bytes {target}"


def test_target_bytes_achieves_nonzero_sparsity_when_room() -> None:
    """With a generous budget we should keep SOME entries (otherwise
    the optimization is silently a no-op)."""
    delta, cost = _random_delta(n_frames=2, H=16, W=16, sparsity=0.5, seed=4)
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=10_000)
    spec = unpack_sparse_delta(blob)
    assert spec.any_delta
    assert spec.n_kept > 0


def test_target_bytes_zero_falls_back_to_empty() -> None:
    """An impossibly tight budget still produces a valid blob (n_kept=0)."""
    delta, cost = _random_delta()
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=1)
    # The all-zero blob should fit within ~50 bytes of overhead
    spec = unpack_sparse_delta(blob)
    assert spec.n_kept == 0
    assert not spec.any_delta


# ── Apply ────────────────────────────────────────────────────────────────


def test_apply_clamps_to_uint8_range() -> None:
    """δ added to a saturated base should clamp to 255, not overflow."""
    # Force one large positive δ
    delta = torch.zeros(1, 3, 4, 4)
    delta[0, 0, 1, 2] = 4.0  # within budget
    cost = torch.ones(1, 4, 4)
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=None)
    spec = unpack_sparse_delta(blob)

    base = torch.full((4, 4, 3), 254.0)
    out = apply_delta_to_frame(base, spec, frame_index=0)
    # All values clamped to [0, 255]
    assert (out >= 0).all() and (out <= 255).all()
    # The targeted pixel should be at 255 (was 254 + ~4 clip).
    assert out[1, 2, 0].item() == 255.0


def test_apply_no_delta_for_frame_returns_input_unchanged() -> None:
    """Frames absent from the spec must be returned untouched (zero-cost
    path the inflate loop relies on for the no-perturbation frames)."""
    delta = torch.zeros(2, 3, 4, 4)
    delta[0, 0, 0, 0] = 3.0  # only frame 0 has δ
    cost = torch.ones(2, 4, 4)
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=None)
    spec = unpack_sparse_delta(blob)

    base = torch.full((4, 4, 3), 100.0)
    out_unchanged = apply_delta_to_frame(base, spec, frame_index=1)
    assert torch.allclose(out_unchanged, base)


def test_apply_wrong_frame_size_raises() -> None:
    """Catches the historical (mask res != renderer res) failure mode
    in the spec dimension. δ is ALWAYS at renderer native resolution."""
    delta, cost = _random_delta(n_frames=1, H=8, W=8, sparsity=0.1)
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0)
    spec = unpack_sparse_delta(blob)

    wrong = torch.zeros(7, 8, 3)  # H mismatch
    with pytest.raises(ValueError, match="frame shape"):
        apply_delta_to_frame(wrong, spec, frame_index=0)


def test_apply_out_of_range_frame_index_raises() -> None:
    delta, cost = _random_delta(n_frames=2, H=4, W=4, sparsity=0.5)
    spec = unpack_sparse_delta(pack_sparse_delta(delta, cost, l_inf_budget=4.0))
    base = torch.zeros(4, 4, 3)
    with pytest.raises(IndexError):
        apply_delta_to_frame(base, spec, frame_index=99)


# ── Bad inputs ───────────────────────────────────────────────────────────


def test_unpack_bad_magic_raises() -> None:
    bad = zlib.compress(b"NOPE" + b"\x00" * 100)
    with pytest.raises(ValueError, match="bad UWD magic"):
        unpack_sparse_delta(bad)


def test_pack_shape_mismatch_raises() -> None:
    delta = torch.zeros(2, 3, 8, 8)
    cost = torch.zeros(2, 8, 9)  # W mismatch
    with pytest.raises(ValueError, match="spatial mismatch"):
        pack_sparse_delta(delta, cost, l_inf_budget=4.0)


def test_pack_frame_count_mismatch_raises() -> None:
    delta = torch.zeros(2, 3, 8, 8)
    cost = torch.zeros(3, 8, 8)  # B mismatch
    with pytest.raises(ValueError, match="frame count mismatch"):
        pack_sparse_delta(delta, cost, l_inf_budget=4.0)


def test_pack_bad_delta_shape_raises() -> None:
    delta = torch.zeros(2, 8, 8)  # not 4-D
    cost = torch.zeros(2, 8, 8)
    with pytest.raises(ValueError, match="delta_bchw"):
        pack_sparse_delta(delta, cost, l_inf_budget=4.0)


# ── Frame-index layout sanity ────────────────────────────────────────────


def test_apply_targets_correct_pixel_channel() -> None:
    """Ground-truth check that the (frame, y, x, c) wire layout matches
    the HWC apply layout. If the index encoding ever drifts, this test
    fails with a precise pinpoint of where δ landed."""
    H, W = 6, 8
    delta = torch.zeros(1, 3, H, W)
    # Put a known value at (y=2, x=5, c=1) → within HWC at flat index
    # y*W*3 + x*3 + c = 2*24 + 15 + 1 = 64
    delta[0, 1, 2, 5] = 4.0
    cost = torch.ones(1, H, W)
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=None)
    spec = unpack_sparse_delta(blob)
    base = torch.zeros(H, W, 3)
    out = apply_delta_to_frame(base, spec, 0)
    assert out[2, 5, 1].item() > 0, (
        "δ landed at the wrong (y, x, c). Layout drift between "
        "pack-side (b, c, y, x) and apply-side (H, W, 3) — see "
        "_build_bcyx_to_byxc_index in tac.uniward_delta."
    )


def test_apply_does_not_mutate_input_buffer() -> None:
    """The renderer output may be reused across frames; apply must not
    alias-mutate the input HWC tensor."""
    delta = torch.zeros(1, 3, 4, 4)
    delta[0, 0, 0, 0] = 4.0
    cost = torch.ones(1, 4, 4)
    spec = unpack_sparse_delta(pack_sparse_delta(delta, cost, l_inf_budget=4.0))
    base = torch.full((4, 4, 3), 100.0)
    base_before = base.clone()
    _ = apply_delta_to_frame(base, spec, 0)
    assert torch.equal(base, base_before)


# ── DeltaSpec ────────────────────────────────────────────────────────────


def test_delta_spec_fields_are_consistent_after_unpack() -> None:
    delta, cost = _random_delta(n_frames=3, H=8, W=8, sparsity=0.10, seed=99)
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=None)
    spec = unpack_sparse_delta(blob)
    assert isinstance(spec, DeltaSpec)
    assert spec.n_frames == 3
    assert spec.H == 8 and spec.W == 8
    assert len(spec.per_frame_local_idx) == 3
    assert len(spec.per_frame_dequant) == 3
    counted = sum(
        0 if t is None else int(t.numel())
        for t in spec.per_frame_local_idx
    )
    assert counted == spec.n_kept


# ── Council-review regression tests (Bugs C1, C2, C3, C4, M1, M2) ────────


def test_c1_optimizer_writeback_interleaves_pair_layout() -> None:
    """REGRESSION (Bug C1): the optimizer batched [t-frames, t+1-frames]
    in the inner loop but the inflate path consumes frames in
    pair-interleaved [p0_t, p0_t+1, p1_t, p1_t+1, ...] order. If the
    write-back ``delta_full[fs:fe] = delta_final`` runs linearly, pair 0's
    t+1-frame ends up where pair 1's t-frame should be. This test bakes in
    a per-pair sentinel and verifies the right global frame got it.

    If you ever re-introduce the linear write-back, this test fires with a
    precise ``(pair, frame_in_pair) → expected vs got`` mismatch.
    """
    H, W = 4, 4
    n_pairs = 4
    n_frames = 2 * n_pairs

    # SIMULATE the optimizer's batch loop layout:
    # base_pairs has shape (B, 2, H, W, 3); base_hwc =
    # cat([base_pairs[:, 0], base_pairs[:, 1]]) yields
    # [p0_t, p1_t, p2_t, p3_t, p0_t+1, p1_t+1, p2_t+1, p3_t+1].
    # We use a per-pair sentinel that varies across pairs so any cross-pair
    # leak is detectable. We ONLY perturb pixel (0, 0, 0) on each frame.
    delta_batch = torch.zeros(2 * n_pairs, 3, H, W)
    sentinels = torch.tensor([1.0, 2.0, 3.0, 4.0])  # one per pair
    for p in range(n_pairs):
        # t-frame of pair p is index p in the [t-frames-first] batch layout.
        delta_batch[p, 0, 0, 0] = sentinels[p]
        # t+1-frame of pair p is index n_pairs + p.
        delta_batch[n_pairs + p, 0, 0, 0] = sentinels[p]

    # Replicate the FIXED write-back from optimize_uniward_delta.py:
    #   delta_full[fs:fe:2]   = delta_batch[:n_pairs]      (t-frames)
    #   delta_full[fs+1:fe:2] = delta_batch[n_pairs:]      (t+1-frames)
    delta_full = torch.zeros(n_frames, 3, H, W)
    fs, fe = 0, 2 * n_pairs
    delta_full[fs:fe:2] = delta_batch[:n_pairs]
    delta_full[fs + 1:fe:2] = delta_batch[n_pairs:]

    cost_full = torch.ones(n_frames, H, W)
    # Use a generous target_bytes so all n_frames sentinel positions are
    # retained (not just top-K via the silent ~1% default).
    blob = pack_sparse_delta(
        delta_full, cost_full, l_inf_budget=4.0, target_bytes=10_000,
    )
    spec = unpack_sparse_delta(blob)
    # Sanity: every frame must have its sentinel kept.
    assert spec.n_kept >= n_frames, (
        f"test setup error: only {spec.n_kept} entries kept, need {n_frames}"
    )

    # Apply each global frame and verify the SENTINEL of the owning pair
    # ended up at (0, 0, 0). If the bug is reintroduced, frame 1 (which
    # SHOULD be pair 0's t+1-frame) would carry the sentinel of pair 1.
    for global_idx in range(n_frames):
        owning_pair = global_idx // 2
        expected_sentinel = float(sentinels[owning_pair].item())
        base = torch.zeros(H, W, 3)
        out = apply_delta_to_frame(base, spec, frame_index=global_idx)
        got = float(out[0, 0, 0].item())
        assert got == pytest.approx(expected_sentinel, abs=4.0 / 127.0 + 1e-5), (
            f"C1 layout drift at global frame {global_idx} "
            f"(owning pair {owning_pair}): expected sentinel "
            f"{expected_sentinel}, got {got}. The optimizer write-back "
            f"must interleave [t, t+1, t, t+1, ...]."
        )


def test_c1_buggy_writeback_would_corrupt() -> None:
    """REGRESSION GUARD (Bug C1, negative direction): demonstrate that the
    BUGGY linear write-back ``delta_full[fs:fe] = delta_batch`` produces
    wrong frame ownership. Locks the failure mode in tests so the bug
    cannot silently come back. Marked as a *positive* assertion on the
    detected mismatch, not a failure to compute.
    """
    H, W = 4, 4
    n_pairs = 4
    n_frames = 2 * n_pairs

    delta_batch = torch.zeros(2 * n_pairs, 3, H, W)
    sentinels = torch.tensor([1.0, 2.0, 3.0, 4.0])
    for p in range(n_pairs):
        delta_batch[p, 0, 0, 0] = sentinels[p]
        delta_batch[n_pairs + p, 0, 0, 0] = sentinels[p]

    # BUG path: linear copy.
    buggy = torch.zeros(n_frames, 3, H, W)
    buggy[0:2 * n_pairs] = delta_batch
    cost_full = torch.ones(n_frames, H, W)
    # Generous target_bytes so all n_frames sentinel positions are kept.
    blob = pack_sparse_delta(buggy, cost_full, l_inf_budget=4.0, target_bytes=10_000)
    spec = unpack_sparse_delta(blob)
    assert spec.n_kept >= n_frames, (
        f"test setup error: only {spec.n_kept} entries kept, need {n_frames}"
    )

    # Frame index 1 (global) is "pair 0, t+1-frame" per inflate convention.
    # In the buggy write-back, that slot was filled by delta_batch[1] which
    # is pair 1's t-frame → sentinel 2, NOT sentinel 1. The fix would put
    # sentinel 1 there. We assert the bug-version produces the WRONG value
    # (sentinel 2). If anyone "fixes" the bug to silently match here, this
    # test fails and the C1 fix is no longer locked in.
    base = torch.zeros(H, W, 3)
    out = apply_delta_to_frame(base, spec, frame_index=1)
    got = float(out[0, 0, 0].item())
    # Sentinel 2.0 expected from the BUG, not sentinel 1.0 (the correct one).
    assert got == pytest.approx(2.0, abs=4.0 / 127.0 + 1e-5), (
        "C1 buggy-path locked: linear write-back should put pair 1's "
        f"t-frame sentinel (=2.0) at global frame 1, got {got}. If this "
        "fails, the layout convention has changed — re-validate test_c1_*."
    )


def test_c2_segnet_shape_assertion_message_format() -> None:
    """REGRESSION (Bug C2): the optimizer's SegNet hinge-loss path must
    hard-fail when SegNet output spatial dims diverge from the GT mask
    dims. Without this, gradients silently broadcast over wrong dims and
    the optimizer learns garbage. We can't easily import the optimizer
    main() (it imports CUDA scorers), so we replicate the assertion
    contract here and verify the assert behaviour matches expectations.
    """
    # Simulate the assertion in isolation. If the assert in
    # optimize_uniward_delta.py is removed, this test still passes — but
    # the test_c2_assertion_present_in_optimizer_source test below catches
    # the source-level drift directly.
    seg_logits_good = torch.zeros(1, 5, 8, 8)
    gt_masks_good = torch.zeros(1, 8, 8, dtype=torch.long)
    assert seg_logits_good.shape[-2:] == gt_masks_good.shape[-2:]

    seg_logits_bad = torch.zeros(1, 5, 16, 16)
    gt_masks_bad = torch.zeros(1, 8, 8, dtype=torch.long)
    assert seg_logits_bad.shape[-2:] != gt_masks_bad.shape[-2:]


def test_c2_assertion_present_in_optimizer_source() -> None:
    """REGRESSION (Bug C2, source-level): grep the optimizer source to
    ensure the seg-shape assertion is present. This is a forensic guard:
    if anyone deletes the assertion, this test fires immediately (no
    GPU required to catch it).
    """
    from pathlib import Path
    src = Path(__file__).resolve().parents[3] / "experiments" / "optimize_uniward_delta.py"
    text = src.read_text()
    assert "seg_logits.shape[-2:] == gt_pair_masks.shape[-2:]" in text, (
        "Bug C2 fix removed: optimize_uniward_delta.py no longer asserts "
        "SegNet logits / GT mask spatial dims. Without this, broadcasting "
        "silently corrupts the hinge-loss gradient. Restore the assert."
    )


def test_c3_apply_does_not_mutate_non_perturbed_pixels() -> None:
    """REGRESSION (Bug C3): the in-apply ``flat.clamp_(0, 255)`` previously
    clamped the WHOLE frame, so a renderer that ever overshoots [0, 255]
    on non-perturbed pixels would see DIFFERENT outputs depending on
    whether δ was installed. Now we clamp only at the modified positions.
    """
    H, W = 4, 4
    delta = torch.zeros(1, 3, H, W)
    delta[0, 1, 2, 3] = 3.0  # one perturbed pixel
    cost = torch.ones(1, H, W)
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=None)
    spec = unpack_sparse_delta(blob)

    # Construct a base whose NON-perturbed pixels are out-of-range. A
    # buggy global clamp would mutate them.
    base = torch.zeros(H, W, 3)
    base.fill_(100.0)
    base[0, 0, 0] = -5.0   # negative outlier
    base[3, 3, 2] = 300.0  # positive outlier

    out = apply_delta_to_frame(base, spec, frame_index=0)
    # Non-perturbed outliers must be UNCHANGED (the apply path is no longer
    # responsible for the global uint8 clamp; the inflate path's
    # ``frame_up.round().clamp(0, 255)`` handles it).
    assert out[0, 0, 0].item() == -5.0, (
        "C3 regression: negative pixel at non-perturbed position was "
        "silently clamped by apply_delta_to_frame. The apply path must "
        "ONLY clamp the perturbed positions."
    )
    assert out[3, 3, 2].item() == 300.0, (
        "C3 regression: positive outlier at non-perturbed position was "
        "silently clamped by apply_delta_to_frame."
    )
    # Perturbed pixel still clamps to [0, 255]: 100 + ~3 well in range.
    assert 100.0 < out[2, 3, 1].item() <= 255.0


def test_c3_perturbed_pixel_still_clamps_to_uint8_range() -> None:
    """REGRESSION (Bug C3, positive direction): the clamp on the modified
    positions must still be active so saturated bases don't overflow uint8.
    """
    H, W = 4, 4
    delta = torch.zeros(1, 3, H, W)
    delta[0, 0, 1, 2] = 4.0  # within budget, will saturate at 255
    cost = torch.ones(1, H, W)
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=None)
    spec = unpack_sparse_delta(blob)

    base = torch.full((H, W, 3), 254.0)
    out = apply_delta_to_frame(base, spec, frame_index=0)
    # Perturbed pixel saturates to 255 (was 254 + ~4 → clip).
    assert out[1, 2, 0].item() == 255.0, (
        "C3 regression: perturbed pixel must still clamp to [0, 255]. "
        f"Got {out[1, 2, 0].item()}."
    )


def test_c4_compliance_warning_in_optimizer_source() -> None:
    """REGRESSION (Bug C4): the optimizer must emit a banner reminding
    operators that Lane C is PENDING council compliance ruling. If the
    banner is removed, scores from this pipeline could quietly land in
    the run-log without the [lane-c-pending-ruling] tag.
    """
    from pathlib import Path
    src = Path(__file__).resolve().parents[3] / "experiments" / "optimize_uniward_delta.py"
    text = src.read_text()
    assert "[lane-c-pending-ruling]" in text, (
        "C4 banner missing from optimize_uniward_delta.py. Operators must "
        "see the compliance reminder on every run."
    )
    assert "PENDING council ruling" in text, (
        "C4 banner missing 'PENDING council ruling' phrasing."
    )


def test_m1_target_bytes_none_emits_warning() -> None:
    """REGRESSION (Bug M1): pack_sparse_delta(target_bytes=None) must emit
    a UserWarning so operators are not surprised by a 100KB blob in
    production. The CLI script keeps an explicit default of 5000.
    """
    delta, cost = _random_delta(n_frames=2, H=8, W=8, sparsity=0.20, seed=11)
    with pytest.warns(UserWarning, match="target_bytes=None"):
        blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=None)
    # Sanity: blob is still well-formed (we did not break the legacy path).
    spec = unpack_sparse_delta(blob)
    assert spec.any_delta


def test_m1_target_bytes_explicit_does_not_warn() -> None:
    """REGRESSION (Bug M1, positive direction): with an explicit budget,
    the warning is suppressed (silence is golden when caller is explicit).
    """
    import warnings as _w
    delta, cost = _random_delta(n_frames=2, H=8, W=8, sparsity=0.20, seed=12)
    with _w.catch_warnings():
        _w.simplefilter("error")  # turn any warning into an error
        # Should not raise, since target_bytes is explicit.
        pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=2000)


def test_m2_no_eval_roundtrip_flag_removed_from_optimizer() -> None:
    """REGRESSION (Bug M2): the --no-eval-roundtrip flag MUST NOT exist in
    the CLI. The CLAUDE.md non-negotiable rule says every training path
    uses eval_roundtrip — silent disabling violates it. The escape hatch
    is now the TAC_ALLOW_NO_ROUNDTRIP env var.

    We grep for the *active* argparse registration ``add_argument("--no-eval-roundtrip"``
    (not any string mention — the comments still document why the flag was
    removed, which is desirable for future readers).
    """
    from pathlib import Path
    src = Path(__file__).resolve().parents[3] / "experiments" / "optimize_uniward_delta.py"
    text = src.read_text()
    # Look for any argparse registration variant — single or double quotes,
    # any whitespace within reason. Fail if a re-register slips back in.
    import re
    pattern = re.compile(r"""add_argument\(\s*['"]--no-eval-roundtrip['"]""")
    assert not pattern.search(text), (
        "M2 regression: --no-eval-roundtrip argparse flag is back. Remove "
        "it; if a diagnostic ablation is needed, use TAC_ALLOW_NO_ROUNDTRIP=1."
    )
    assert "TAC_ALLOW_NO_ROUNDTRIP" in text, (
        "M2 regression: TAC_ALLOW_NO_ROUNDTRIP escape hatch missing from "
        "optimize_uniward_delta.py."
    )


# ── Codex R5 review HIGH bugs (post-R4) ──────────────────────────────────


# ---- Issue 1: SegNet loss operates on wrong frame order ----
#
# In the optimizer, ``base_chw`` is built via
#   ``cat([base_pairs[:, 0], base_pairs[:, 1]], dim=0)``
# so its frames are in BATCHED layout
#   [p0_t, p1_t, ..., p(B-1)_t, p0_t+1, p1_t+1, ..., p(B-1)_t+1].
# But ``gt_pair_masks = gt_segnet_masks[fs:fe]`` slices the global
# per-frame mask tensor in PAIR-INTERLEAVED layout
#   [p0_t, p0_t+1, p1_t, p1_t+1, ..., p(B-1)_t, p(B-1)_t+1].
# For batch_pairs > 1 the two layouts diverge → SegNet loss compares the
# wrong frames. The fix: interleave perturbed_round to pair order BEFORE
# the SegNet forward (PoseNet path already re-pairs separately, so this
# does not affect it).
#
# The tests below verify both directions: the FIXED interleave puts the
# right-pair gradient on the right frame, and the BUGGY un-interleaved
# path puts it on the wrong frame.


def _layout_interleave_t_then_t1_to_pair_order(
    perturbed_round: torch.Tensor,
) -> torch.Tensor:
    """Mirror of the FIXED interleave inside optimize_uniward_delta.py.

    Take a tensor in BATCHED layout
        [p0_t, p1_t, ..., p(n-1)_t, p0_t+1, p1_t+1, ..., p(n-1)_t+1]
    and return one in PAIR-INTERLEAVED layout
        [p0_t, p0_t+1, p1_t, p1_t+1, ..., p(n-1)_t, p(n-1)_t+1].
    Preserves autograd lineage (uses views + transpose only).
    """
    n_half = perturbed_round.shape[0] // 2
    return (
        perturbed_round
        .reshape(2, n_half, *perturbed_round.shape[1:])
        .transpose(0, 1)
        .reshape(perturbed_round.shape)
        .contiguous()
    )


def test_codex_r5_segnet_pair_order_with_batch2_fixed_path() -> None:
    """REGRESSION (Codex R5 HIGH — Issue 1, positive direction): with
    batch_pairs=2 and a fake SegNet that produces logits proportional to
    its input, the FIXED interleave path lands the gradient w.r.t.
    perturbed[i] on the frame whose label is gt_pair_masks[i_pair_layout].

    The test puts a per-frame sentinel into the perturbed_round tensor
    (BATCHED layout), runs it through the FIXED interleave, computes
    a SegNet-shaped loss that depends on element [b, c=0, 0, 0] of the
    interleaved input, and inspects the gradient that flows back to the
    ORIGINAL perturbed_round. With the fix, the gradient at perturbed[i]
    corresponds to the gt slot for that frame in PAIR-INTERLEAVED order;
    with the bug it would correspond to a swapped neighbor.

    This locks the actual data-flow contract — not just a source grep —
    so reverting the interleave fails this test, even if the comments
    documenting it stay intact.
    """
    n_pairs = 2
    n_frames_in_batch = 2 * n_pairs  # 4 frames

    # Frame layout in perturbed_round (BATCHED): row0=p0_t, row1=p1_t,
    # row2=p0_t+1, row3=p1_t+1.
    # Sentinels: a unique non-zero magnitude per frame so we can detect
    # which input row a given output gradient came from.
    perturbed_round = torch.zeros(
        n_frames_in_batch, 3, 4, 4, requires_grad=True,
    )

    # Ordering after the FIX (PAIR-INTERLEAVED): row0=p0_t, row1=p0_t+1,
    # row2=p1_t, row3=p1_t+1. We design GT labels so each pair-interleaved
    # row has a UNIQUE per-pixel label, so the gradient on the SegNet
    # logits is uniquely attributable to a specific PAIR-INTERLEAVED row.
    gt_pair_masks = torch.zeros(n_frames_in_batch, 4, 4, dtype=torch.long)
    gt_pair_masks[0, 0, 0] = 1  # owns p0_t
    gt_pair_masks[1, 0, 1] = 1  # owns p0_t+1
    gt_pair_masks[2, 0, 2] = 1  # owns p1_t
    gt_pair_masks[3, 0, 3] = 1  # owns p1_t+1

    # FIXED interleave (matches optimizer source).
    pair_order = _layout_interleave_t_then_t1_to_pair_order(perturbed_round)
    # Trivial "SegNet": logits = pair_order broadcast to 5 classes via a
    # frozen 1-channel projection on channel 0. No learnable params; the
    # gradient w.r.t. pair_order[b, 0, y, x] is +1 wherever gt indexed
    # that pixel as class 1 (we just use a sum-of-targeted-pixels surrogate).
    targeted = (gt_pair_masks == 1).float().unsqueeze(1)  # (B, 1, H, W)
    # loss = sum(pair_order[:, 0:1] * targeted) — one labelled pixel per row.
    loss = (pair_order[:, 0:1] * targeted).sum()
    loss.backward()

    # Now check: gradient at perturbed_round[b, 0, y, x] should be 1.0 at
    # the pixel that PAIR-LAYOUT row "would have" labelled as class 1.
    # PAIR-INTERLEAVED row b corresponds to BATCHED row given by the
    # inverse permutation:
    #   pair_row b → (frame_in_pair = b % 2) and (pair_id = b // 2),
    #   batched_row = (frame_in_pair * n_pairs) + pair_id.
    # So:
    #   pair row 0 (p0_t)   → batched 0
    #   pair row 1 (p0_t+1) → batched 2
    #   pair row 2 (p1_t)   → batched 1
    #   pair row 3 (p1_t+1) → batched 3
    grad = perturbed_round.grad
    assert grad is not None
    # Fixed-direction expectations:
    assert grad[0, 0, 0, 0].item() == 1.0, (
        "FIXED path: pair row 0 (p0_t) GT pixel must show grad=1.0 on "
        f"batched row 0 (p0_t). Got grad={grad[0, 0, 0, 0].item()}. "
        "If 0.0, the interleave was a no-op (bug); if it landed elsewhere, "
        "the layout convention drifted."
    )
    assert grad[2, 0, 0, 1].item() == 1.0, (
        "FIXED path: pair row 1 (p0_t+1) GT pixel (at y=0, x=1) must show "
        "grad=1.0 on batched row 2 (p0_t+1). "
        f"Got grad={grad[2, 0, 0, 1].item()}."
    )
    assert grad[1, 0, 0, 2].item() == 1.0, (
        "FIXED path: pair row 2 (p1_t) GT pixel (at y=0, x=2) must show "
        "grad=1.0 on batched row 1 (p1_t). "
        f"Got grad={grad[1, 0, 0, 2].item()}."
    )
    assert grad[3, 0, 0, 3].item() == 1.0, (
        "FIXED path: pair row 3 (p1_t+1) GT pixel (at y=0, x=3) must show "
        "grad=1.0 on batched row 3 (p1_t+1). "
        f"Got grad={grad[3, 0, 0, 3].item()}."
    )
    # And no gradient should leak into other positions on row 1 (the row
    # the bug would have aliased pair row 1's signal into).
    assert grad[1, 0, 0, 1].item() == 0.0, (
        "FIXED path: no spurious gradient from pair row 1 should land on "
        "batched row 1 (the BUG signature). "
        f"Got grad={grad[1, 0, 0, 1].item()}."
    )


def test_codex_r5_segnet_buggy_layout_lands_gradient_on_wrong_row() -> None:
    """REGRESSION (Codex R5 HIGH — Issue 1, negative direction): the BUGGY
    path (SegNet input is the BATCHED-layout perturbed_round directly,
    without interleave) lands gradients from pair-row b on BATCHED row b.

    For batch_pairs=2 this means the GT slot for p0_t+1 (pair-row 1) gets
    matched against batched row 1 = p1_t — silent corruption. We assert
    the BUG signature explicitly so any future refactor that "happens to
    coincide" with the fix gets caught.
    """
    n_pairs = 2
    n_frames_in_batch = 2 * n_pairs

    perturbed_round = torch.zeros(
        n_frames_in_batch, 3, 4, 4, requires_grad=True,
    )
    gt_pair_masks = torch.zeros(n_frames_in_batch, 4, 4, dtype=torch.long)
    gt_pair_masks[0, 0, 0] = 1  # owns pair row 0 (p0_t)
    gt_pair_masks[1, 0, 1] = 1  # owns pair row 1 (p0_t+1)
    gt_pair_masks[2, 0, 2] = 1  # owns pair row 2 (p1_t)
    gt_pair_masks[3, 0, 3] = 1  # owns pair row 3 (p1_t+1)

    # BUG path: feed perturbed_round directly (batched layout) — no
    # interleave. Gradient at row b, channel 0 lands at the GT-targeted
    # pixel of GT row b, NOT at the pair-layout row b's slot.
    targeted = (gt_pair_masks == 1).float().unsqueeze(1)
    loss = (perturbed_round[:, 0:1] * targeted).sum()
    loss.backward()
    grad = perturbed_round.grad
    assert grad is not None
    # In the BUG, batched row 1 (p1_t) gets grad=1.0 at (y=0, x=1) — the
    # pixel that the GT mask for pair row 1 (p0_t+1) labelled. This is
    # the WRONG-FRAME-ORDER signature. Must hold so the locking test
    # below cannot silently regress to 0.
    assert grad[1, 0, 0, 1].item() == 1.0, (
        "BUG signature lock: in the un-interleaved path, batched row 1 "
        "(p1_t) absorbs the GT labels intended for pair row 1 (p0_t+1). "
        f"Got grad={grad[1, 0, 0, 1].item()}; expected 1.0 to lock the "
        "bug. If this fails, the layout assumptions changed — re-derive "
        "the C1 + Codex R5 fixes against the new layout."
    )


def test_codex_r5_segnet_pair_order_batch1_no_op() -> None:
    """SANITY (Codex R5 HIGH): with batch_pairs=1 the interleave is a
    no-op (the BATCHED and PAIR-INTERLEAVED layouts coincide for n=2
    frames). Confirms we did not regress single-pair behaviour.
    """
    n_pairs = 1
    perturbed_round = torch.zeros(
        2 * n_pairs, 3, 4, 4, requires_grad=True,
    )
    pair_order = _layout_interleave_t_then_t1_to_pair_order(perturbed_round)
    # Verify the contiguous storage matches element-by-element (i.e. for
    # n_pairs=1 the interleave is a permutation that fixes every position).
    # We do this by writing distinct values into both, then asserting
    # equality after the round-trip-through-interleave.
    for i in range(perturbed_round.shape[0]):
        for c in range(3):
            for y in range(4):
                for x in range(4):
                    perturbed_round.data[i, c, y, x] = float(
                        i * 1000 + c * 100 + y * 10 + x
                    )
    pair_order = _layout_interleave_t_then_t1_to_pair_order(perturbed_round)
    assert torch.equal(perturbed_round.detach(), pair_order.detach()), (
        "batch_pairs=1 interleave must be a no-op; got differences. The "
        "single-pair path is the most-tested code; do not break it."
    )


def test_codex_r5_segnet_interleave_preserves_autograd() -> None:
    """REGRESSION (Codex R5 HIGH): the layout interleave must be a pure
    view/transpose op so gradients flow back into the source tensor's
    .grad without a copy that breaks autograd lineage. This matters
    because the optimizer's ``optimizer.step()`` mutates ``delta`` via
    its .grad attribute.
    """
    src = torch.randn(8, 3, 4, 4, requires_grad=True)
    out = _layout_interleave_t_then_t1_to_pair_order(src)
    loss = out.sum()
    loss.backward()
    assert src.grad is not None and torch.allclose(
        src.grad, torch.ones_like(src),
    ), (
        "Interleave broke autograd lineage. The fix must use views + "
        "transpose, NOT torch.cat / torch.stack on cloned slices."
    )


def test_codex_r5_segnet_interleave_used_in_optimizer_source() -> None:
    """REGRESSION (Codex R5 HIGH, source-level): grep the optimizer for
    the interleave call. If anyone removes it (and reintroduces the
    bug), this fires immediately. Source-level guard for non-CUDA CI.
    """
    from pathlib import Path
    src = Path(__file__).resolve().parents[3] / "experiments" / "optimize_uniward_delta.py"
    text = src.read_text()
    # The fix code must:
    # 1. compute n_half from perturbed_round
    # 2. reshape (2, n_half, *) → transpose(0, 1) → reshape back
    # 3. feed the interleaved tensor (perturbed_round_pair_order) to SegNet
    assert "perturbed_round_pair_order" in text, (
        "Codex R5 HIGH regression: the interleaved-layout variable name "
        "is gone from optimize_uniward_delta.py. The SegNet loss is "
        "back to operating on BATCHED layout — wrong-frame gradients "
        "will silently corrupt scores when batch_pairs > 1."
    )
    assert ".transpose(0, 1)" in text, (
        "Codex R5 HIGH regression: the .transpose(0, 1) call that performs "
        "the BATCHED → PAIR-INTERLEAVED layout swap is gone. Restore it."
    )
    assert "preprocess_input(perturbed_round_pair_order" in text, (
        "Codex R5 HIGH regression: SegNet preprocess_input is being fed "
        "perturbed_round directly again instead of the pair-ordered "
        "view. Restore the interleaved input."
    )


# ---- Issue 2: silent contest-noncompliance gate ----
#
# The original C4 fix only emitted a stdout banner; it left no
# machine-readable state, so an archive could be built, zipped, copied,
# and submitted with no operator ever seeing the warning. The fix adds a
# compliance_status field to the wire format header (with backward-compat
# default ``pending_ruling`` for v1 blobs), threads it through pack/unpack/
# DeltaSpec, makes the inflate path print a banner whenever it loads a
# pending δ, and makes the archive builder fail closed on pending δ unless
# --allow-pending-compliance is explicitly passed.


def test_codex_r5_default_pack_carries_pending_ruling() -> None:
    """REGRESSION (Codex R5 HIGH — Issue 2): a freshly-built δ with no
    explicit compliance_status MUST decode as PENDING_RULING. This is
    the safe default; archives built around it will refuse to bundle.
    """
    from tac.uniward_delta import COMPLIANCE_PENDING
    delta, cost = _random_delta(n_frames=2, H=8, W=8, sparsity=0.1, seed=21)
    blob = pack_sparse_delta(delta, cost, l_inf_budget=4.0, target_bytes=2000)
    spec = unpack_sparse_delta(blob)
    assert spec.compliance_status == COMPLIANCE_PENDING, (
        f"Default compliance_status is {spec.compliance_status!r}, "
        f"expected {COMPLIANCE_PENDING!r}. Fail-closed default is part "
        "of the contest-compliance gate."
    )


def test_codex_r5_compliance_status_round_trips_for_all_values() -> None:
    """REGRESSION (Codex R5 HIGH — Issue 2): each valid compliance_status
    must round-trip exactly through pack → unpack. Regression-protects
    the wire format from silently dropping the field.
    """
    from tac.uniward_delta import (
        COMPLIANCE_PENDING, COMPLIANCE_APPROVED, COMPLIANCE_REJECTED,
    )
    delta, cost = _random_delta(n_frames=2, H=8, W=8, sparsity=0.1, seed=22)
    for status in (COMPLIANCE_PENDING, COMPLIANCE_APPROVED, COMPLIANCE_REJECTED):
        blob = pack_sparse_delta(
            delta, cost, l_inf_budget=4.0, target_bytes=2000,
            compliance_status=status,
        )
        spec = unpack_sparse_delta(blob)
        assert spec.compliance_status == status, (
            f"compliance_status round-trip failed: packed {status!r}, "
            f"got back {spec.compliance_status!r}."
        )


def test_codex_r5_compliance_status_round_trips_for_zero_kept() -> None:
    """The n_kept==0 (any_delta=False) early-return path must ALSO
    propagate compliance_status. Otherwise a pending δ that happens to
    pack to zero kept entries (e.g., very tight target_bytes) decodes
    as PENDING via the default but loses any explicit state.
    """
    from tac.uniward_delta import COMPLIANCE_REJECTED
    delta, cost = _random_delta(n_frames=2, H=8, W=8, sparsity=0.1, seed=23)
    # target_bytes=1 forces n_kept=0
    blob = pack_sparse_delta(
        delta, cost, l_inf_budget=4.0, target_bytes=1,
        compliance_status=COMPLIANCE_REJECTED,
    )
    spec = unpack_sparse_delta(blob)
    assert not spec.any_delta and spec.n_kept == 0
    assert spec.compliance_status == COMPLIANCE_REJECTED, (
        "Zero-kept early-return path lost the compliance_status. The "
        "fail-closed gate must work regardless of n_kept."
    )


def test_codex_r5_invalid_compliance_status_raises() -> None:
    """REGRESSION (Codex R5 HIGH — Issue 2): pack must reject an
    out-of-band compliance_status. Prevents typos like ``"PENDING"``
    or ``"yes"`` from silently bypassing the gate.
    """
    delta, cost = _random_delta(n_frames=1, H=4, W=4, sparsity=0.5, seed=24)
    with pytest.raises(ValueError, match="compliance_status"):
        pack_sparse_delta(
            delta, cost, l_inf_budget=4.0, target_bytes=2000,
            compliance_status="approved-by-vibes",
        )


def test_codex_r5_legacy_v1_blob_decodes_as_pending_ruling() -> None:
    """REGRESSION (Codex R5 HIGH — Issue 2, backward compat): a blob in
    the v1 format (no compliance_status field) MUST decode with
    compliance_status=PENDING_RULING (fail-safe default). Otherwise an
    older archive could silently bypass the gate.

    We synthesise a v1 blob by calling pack and then rewriting the
    header to drop the new field + downgrade schema_version to 1.
    """
    import json as _json
    import struct as _struct
    from tac.uniward_delta import COMPLIANCE_PENDING, MAGIC

    delta, cost = _random_delta(n_frames=2, H=8, W=8, sparsity=0.1, seed=25)
    v2_blob = pack_sparse_delta(
        delta, cost, l_inf_budget=4.0, target_bytes=2000,
    )
    raw = zlib.decompress(v2_blob)
    assert raw[:4] == MAGIC
    header_len = _struct.unpack("<I", raw[4:8])[0]
    header = _json.loads(raw[8:8 + header_len].decode("utf-8"))
    body = raw[8 + header_len:]
    # Synthesise v1: drop compliance_status, set schema_version=1.
    header.pop("compliance_status", None)
    header["schema_version"] = 1
    new_header = _json.dumps(header, sort_keys=True).encode("utf-8")
    legacy_raw = MAGIC + _struct.pack("<I", len(new_header)) + new_header + body
    legacy_blob = zlib.compress(legacy_raw, level=9)

    spec = unpack_sparse_delta(legacy_blob)
    assert spec.compliance_status == COMPLIANCE_PENDING, (
        f"v1 legacy blob decoded with compliance_status="
        f"{spec.compliance_status!r}, expected fail-safe default "
        f"{COMPLIANCE_PENDING!r}. Old archives would silently slip past "
        "the gate without this default."
    )


def test_codex_r5_unknown_compliance_status_in_blob_falls_back_to_pending() -> None:
    """REGRESSION (Codex R5 HIGH — Issue 2): an unrecognized
    compliance_status string in a foreign / corrupted blob must coerce
    to PENDING_RULING — never crash the loader, never silently approve.
    """
    import json as _json
    import struct as _struct
    from tac.uniward_delta import COMPLIANCE_PENDING, MAGIC

    delta, cost = _random_delta(n_frames=1, H=4, W=4, sparsity=0.5, seed=26)
    blob = pack_sparse_delta(
        delta, cost, l_inf_budget=4.0, target_bytes=2000,
    )
    raw = zlib.decompress(blob)
    header_len = _struct.unpack("<I", raw[4:8])[0]
    header = _json.loads(raw[8:8 + header_len].decode("utf-8"))
    body = raw[8 + header_len:]
    header["compliance_status"] = "frobnicated"  # garbage value
    new_header = _json.dumps(header, sort_keys=True).encode("utf-8")
    raw2 = MAGIC + _struct.pack("<I", len(new_header)) + new_header + body
    blob2 = zlib.compress(raw2, level=9)

    spec = unpack_sparse_delta(blob2)
    assert spec.compliance_status == COMPLIANCE_PENDING, (
        "Unknown compliance_status must coerce to PENDING (fail-safe). "
        f"Got {spec.compliance_status!r}."
    )


def test_codex_r5_inflate_renderer_emits_pending_banner() -> None:
    """REGRESSION (Codex R5 HIGH — Issue 2, source-level): the inflate
    path must emit the [lane-c-pending-ruling] banner whenever a δ with
    PENDING_RULING is loaded. No GPU required to verify — source grep.
    """
    from pathlib import Path
    src = Path(__file__).resolve().parents[3] / "submissions" / "robust_current" / "inflate_renderer.py"
    text = src.read_text()
    assert "[lane-c-pending-ruling]" in text, (
        "Codex R5 HIGH regression: inflate_renderer.py no longer prints "
        "the [lane-c-pending-ruling] banner. Eval logs would silently "
        "carry an unflagged Lane C score."
    )
    assert "compliance_status == _UWD_COMPLIANCE_PENDING" in text or \
           'compliance_status == "pending_ruling"' in text or \
           "compliance_status == COMPLIANCE_PENDING" in text, (
        "Codex R5 HIGH regression: the inflate banner is no longer gated "
        "on compliance_status — either always-on or never-on. Restore "
        "the conditional so APPROVED δ skip the banner."
    )


def test_codex_r5_archive_builder_refuses_pending_without_override() -> None:
    """REGRESSION (Codex R5 HIGH — Issue 2): the archive builder MUST
    raise SystemExit when asked to bundle a PENDING_RULING δ without
    --allow-pending-compliance. We invoke the script as a subprocess so
    the SystemExit / argparse interactions are tested end-to-end.

    This is the actual contest-compliance gate — the banner is only an
    audit signal. Without this test, a naive operator could ``cp
    delta.bin archive/`` + ``zip archive.zip ...`` and ship.
    """
    import subprocess
    import sys as _sys
    import tempfile
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    builder = repo_root / "experiments" / "build_baseline_archive.py"

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # Build a tiny δ with PENDING status (the default).
        delta, cost = _random_delta(n_frames=1, H=4, W=4, sparsity=0.5, seed=31)
        blob = pack_sparse_delta(
            delta, cost, l_inf_budget=4.0, target_bytes=2000,
        )
        delta_path = td_path / "delta.bin"
        delta_path.write_bytes(blob)

        # Synthesise placeholder paths (existence is enough — the gate
        # fires before any I/O). We use the δ file itself as the
        # placeholder so existence checks pass.
        cmd = [
            _sys.executable, str(builder),
            "--renderer", str(delta_path),
            "--poses", str(delta_path),
            "--gt-video", str(delta_path),
            "--output", str(td_path / "out.zip"),
            "--with-uniward-delta", str(delta_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode != 0, (
            "Archive builder accepted a PENDING_RULING δ without the "
            "override. The fail-closed gate is broken. "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
        combined = (proc.stdout + proc.stderr).lower()
        assert "pending_ruling" in combined or "pending-ruling" in combined, (
            f"Archive builder refused for the wrong reason. "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
        assert "allow-pending-compliance" in combined, (
            "Refusal message must point operators at the override flag. "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )


def test_codex_r5_archive_builder_refuses_rejected_even_with_override() -> None:
    """REGRESSION (Codex R5 HIGH — Issue 2): a δ marked REJECTED must
    NEVER be bundled, even with --allow-pending-compliance set. This is
    the absolute-floor refusal — REJECTED means the council has ruled
    against this specific artifact.
    """
    import subprocess
    import sys as _sys
    import tempfile
    from pathlib import Path
    from tac.uniward_delta import COMPLIANCE_REJECTED

    repo_root = Path(__file__).resolve().parents[3]
    builder = repo_root / "experiments" / "build_baseline_archive.py"

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        delta, cost = _random_delta(n_frames=1, H=4, W=4, sparsity=0.5, seed=32)
        blob = pack_sparse_delta(
            delta, cost, l_inf_budget=4.0, target_bytes=2000,
            compliance_status=COMPLIANCE_REJECTED,
        )
        delta_path = td_path / "delta.bin"
        delta_path.write_bytes(blob)

        cmd = [
            _sys.executable, str(builder),
            "--renderer", str(delta_path),
            "--poses", str(delta_path),
            "--gt-video", str(delta_path),
            "--output", str(td_path / "out.zip"),
            "--with-uniward-delta", str(delta_path),
            "--allow-pending-compliance",  # MUST NOT bypass REJECTED
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode != 0, (
            "Archive builder accepted a REJECTED δ even with the "
            "pending-compliance override. REJECTED is an absolute floor; "
            "it must never be bundled. "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
        combined = (proc.stdout + proc.stderr).lower()
        assert "rejected" in combined, (
            f"Refusal message did not mention 'rejected'. "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )


def test_codex_r5_optimizer_writes_compliance_status_into_header() -> None:
    """REGRESSION (Codex R5 HIGH — Issue 2, source-level): the optimizer
    must thread its --compliance-status arg into the pack call and into
    the provenance JSON. Source grep, no GPU required.
    """
    from pathlib import Path
    src = Path(__file__).resolve().parents[3] / "experiments" / "optimize_uniward_delta.py"
    text = src.read_text()
    assert "--compliance-status" in text, (
        "Codex R5 HIGH regression: --compliance-status flag missing from "
        "optimize_uniward_delta.py."
    )
    # CLI flag must reach pack_sparse_delta.
    assert "compliance_status=args.compliance_status" in text, (
        "Codex R5 HIGH regression: pack_sparse_delta is no longer being "
        "called with compliance_status=args.compliance_status. The wire "
        "header would silently default to PENDING for ALL builds, even "
        "if an operator passed --compliance-status approved."
    )
