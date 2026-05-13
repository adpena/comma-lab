"""Tests for Lane J-IMP iterative magnitude pruning module.

Per Round 26 finding: every test asserts SIGN/VALUE, not just shape, so
a no-op implementation cannot pass.
"""

from __future__ import annotations

import io

import pytest
import torch
import torch.nn as nn

from tac.iterative_magnitude_pruning import (
    IMPState,
    apply_mask_to_model,
    compute_actual_sparsity,
    fp4_pack_values,
    fp4_unpack_values,
    iter_prunable_parameters,
    prune_lowest_magnitude,
    rewind_weights_to_early_epoch,
    snapshot_state_dict,
    sparse_csr_decode,
    sparse_csr_export,
)


def _make_tiny_conv_net(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)

    class TinyConv(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Two Conv2d's; small enough to keep tests fast but big
            # enough that 20% pruning is unambiguous.
            self.c1 = nn.Conv2d(3, 16, 3, padding=1)  # 3*16*9 = 432 weights
            self.bn = nn.BatchNorm2d(16)
            self.c2 = nn.Conv2d(16, 8, 3, padding=1)  # 16*8*9 = 1152 weights
            # 432 + 1152 = 1584 prunable weights total

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.c2(self.bn(self.c1(x)))

    return TinyConv()


# ── Pruning correctness ─────────────────────────────────────────────────


def test_prune_lowest_magnitude_removes_target_fraction() -> None:
    """Cycle-0 prune must zero exactly the target fraction (within 1 weight)."""
    model = _make_tiny_conv_net(seed=42)
    total_prunable = sum(p.numel() for _, p in iter_prunable_parameters(model))
    target = 0.20

    mask = prune_lowest_magnitude(model, sparsity_increment=target)

    pruned = sum(int((~m).sum().item()) for m in mask.values())
    expected = int(round(total_prunable * target))
    # ±1 tolerance for kthvalue tie handling
    assert abs(pruned - expected) <= 1, (
        f"expected {expected} pruned, got {pruned} "
        f"(of {total_prunable} prunable)"
    )

    sparsity = compute_actual_sparsity(model, mask)
    assert abs(sparsity - target) < 0.01, (
        f"sparsity {sparsity:.3f} should be ~{target}"
    )


def test_prune_preserves_high_magnitude_weights() -> None:
    """The K largest-magnitude weights must NOT be pruned."""
    model = _make_tiny_conv_net(seed=7)
    target = 0.20

    # Find the top-K largest weights across all prunable layers BEFORE prune.
    abs_values: list[tuple[float, str, int]] = []
    for name, param in iter_prunable_parameters(model):
        flat = param.detach().abs().flatten().cpu()
        for i in range(int(flat.numel())):
            abs_values.append((float(flat[i].item()), name, i))
    abs_values.sort(key=lambda x: -x[0])

    total = len(abs_values)
    n_kept = total - int(round(total * target))
    # Top n_kept weights MUST survive (assert top 50 to keep test fast +
    # robust to tie-handling near the threshold).
    top_50 = abs_values[:50]

    mask = prune_lowest_magnitude(model, sparsity_increment=target)

    for _val, name, flat_idx in top_50:
        m = mask[name]
        flat_m = m.flatten()
        assert flat_m[flat_idx].item() is True or flat_m[flat_idx].item() == 1, (
            f"top-magnitude weight pruned: {name}[{flat_idx}]"
        )


# ── Mask application + rewinding ────────────────────────────────────────


def test_rewind_only_affects_unpruned_weights() -> None:
    """After rewind, pruned positions stay zero; survivors match snapshot."""
    model = _make_tiny_conv_net(seed=99)
    snap = snapshot_state_dict(model)

    # Take a prune
    mask = prune_lowest_magnitude(model, sparsity_increment=0.30)
    apply_mask_to_model(model, mask)

    # Mutate weights non-trivially to simulate "post-train"
    with torch.no_grad():
        for _name, p in iter_prunable_parameters(model):
            p.add_(torch.randn_like(p) * 0.5)

    # Rewind
    rewind_weights_to_early_epoch(model, snap, mask)

    for name, param in iter_prunable_parameters(model):
        m = mask[name].to(param.device)
        snap_t = snap[name].to(param.device).to(param.dtype)
        # Pruned positions must be exactly zero
        pruned_vals = param[~m]
        assert torch.all(pruned_vals == 0), (
            f"{name}: {(pruned_vals != 0).sum().item()} pruned positions are nonzero"
        )
        # Surviving positions must equal the snapshot
        kept_now = param[m]
        kept_snap = snap_t[m]
        assert torch.allclose(kept_now, kept_snap, atol=1e-6), (
            f"{name}: surviving weights diverge from snapshot, "
            f"max diff = {(kept_now - kept_snap).abs().max().item()}"
        )


# ── Sparsity reporting ──────────────────────────────────────────────────


def test_compute_actual_sparsity_matches_pruning_target() -> None:
    """compute_actual_sparsity must match the requested target within 1%."""
    model = _make_tiny_conv_net(seed=3)
    for target in (0.10, 0.30, 0.50, 0.80):
        m_test = _make_tiny_conv_net(seed=3)  # fresh independent model
        mask = prune_lowest_magnitude(m_test, sparsity_increment=target)
        apply_mask_to_model(m_test, mask)
        s_mask = compute_actual_sparsity(m_test, mask)
        s_model = compute_actual_sparsity(m_test)
        assert abs(s_mask - target) < 0.01, (
            f"sparsity-from-mask {s_mask:.3f} ≠ target {target}"
        )
        assert abs(s_model - target) < 0.01, (
            f"sparsity-from-model {s_model:.3f} ≠ target {target} "
            f"(apply_mask_to_model failed to zero)"
        )


# ── Sparse-CSR export ───────────────────────────────────────────────────


def test_sparse_csr_export_byte_size() -> None:
    """At 95% sparsity, sparse-CSR export must beat dense FP4 storage."""
    torch.manual_seed(0)
    # Use a tensor with numel < 65535 (uint16 indices)
    weights = torch.randn(8, 16, 3, 3)  # 1152 weights
    numel = int(weights.numel())
    # 95% sparsity: ~58 surviving weights
    threshold = weights.abs().flatten().kthvalue(int(numel * 0.95)).values.item()
    mask = weights.abs() > threshold
    actual_sparsity = 1.0 - float(mask.sum().item()) / numel
    assert actual_sparsity > 0.94, f"setup error: sparsity {actual_sparsity}"

    blob = sparse_csr_export(weights, mask)
    # Dense FP4 = numel × 4 bits / 8
    dense_fp4_bytes = (numel * 4 + 7) // 8

    assert len(blob) < dense_fp4_bytes, (
        f"at {actual_sparsity:.2%} sparsity, sparse blob {len(blob)} B "
        f"should beat dense FP4 {dense_fp4_bytes} B"
    )


def test_sparse_csr_roundtrip_preserves_values() -> None:
    """Encode → decode must preserve values exactly under FP4 quantization.

    Because FP4 is lossy, the test asserts:
      - mask is preserved exactly
      - surviving values dequantize to the nearest of the 16 FP4 levels
        scaled by per-tensor abs_max (within FP4 codebook step error)
    """
    torch.manual_seed(1)
    weights = torch.randn(4, 8, 3, 3) * 0.5  # 288 weights
    mask = weights.abs() > weights.abs().median().item()
    blob = sparse_csr_export(weights, mask)
    dense_dec, mask_dec = sparse_csr_decode(blob)

    assert dense_dec.shape == weights.shape
    assert mask_dec.shape == mask.shape
    # Mask MUST roundtrip exactly
    assert torch.equal(mask_dec, mask)

    # Surviving values: max FP4 step error is (2 / 15) * abs_max ≈ 0.133 * abs_max
    abs_max = float(weights[mask].abs().max().item())
    fp4_step = (2.0 / 15.0) * abs_max
    diff = (dense_dec[mask] - weights[mask]).abs()
    assert diff.max().item() <= fp4_step + 1e-5, (
        f"FP4 roundtrip error {diff.max().item():.4f} > expected step "
        f"{fp4_step:.4f}"
    )

    # Pruned positions must be zero
    pruned = dense_dec[~mask]
    assert torch.all(pruned == 0), (
        f"pruned positions non-zero after decode: {(pruned != 0).sum().item()}"
    )


# ── IMPState serialization ──────────────────────────────────────────────


def test_imp_state_serialization_roundtrip() -> None:
    """torch.save / torch.load must roundtrip an IMPState losslessly."""
    state = IMPState(
        cycle_count=3,
        sparsity_target=0.90,
        sparsity_increment=0.20,
        mask={
            "c1.weight": torch.tensor([[True, False], [True, True]]),
            "c2.weight": torch.tensor([False, True, False, True]),
        },
        early_epoch_weights={
            "c1.weight": torch.randn(2, 2),
            "c2.weight": torch.randn(4),
        },
    )

    buf = io.BytesIO()
    torch.save(state.to_dict(), buf)
    buf.seek(0)
    loaded = IMPState.from_dict(torch.load(buf, weights_only=False))

    assert loaded.cycle_count == 3
    assert loaded.sparsity_target == 0.90
    assert loaded.sparsity_increment == 0.20
    assert set(loaded.mask) == {"c1.weight", "c2.weight"}
    assert torch.equal(loaded.mask["c1.weight"], state.mask["c1.weight"])
    assert torch.equal(loaded.mask["c2.weight"], state.mask["c2.weight"])
    assert torch.allclose(
        loaded.early_epoch_weights["c1.weight"],
        state.early_epoch_weights["c1.weight"],
    )

    # Sparsity-after-cycle math
    assert state.expected_sparsity_after_cycle(0) == pytest.approx(0.20)
    assert state.expected_sparsity_after_cycle(9) == pytest.approx(
        1.0 - 0.8 ** 10, abs=1e-9
    )


# ── Cumulative-sparsity ratchet (multi-cycle integration) ────────────────


def test_multi_cycle_sparsity_grows_monotonically() -> None:
    """Each cycle must INCREASE sparsity and never restore a pruned weight."""
    model = _make_tiny_conv_net(seed=11)
    mask = None
    last_s = 0.0
    pruned_positions: dict[str, torch.Tensor] | None = None
    for cycle in range(3):
        new_mask = prune_lowest_magnitude(
            model, sparsity_increment=0.20, current_mask=mask
        )
        s = compute_actual_sparsity(model, new_mask)
        assert s > last_s, f"cycle {cycle}: sparsity did not grow ({last_s} → {s})"
        # Once-pruned weights must stay pruned.
        if pruned_positions is not None:
            for name, prev_pruned in pruned_positions.items():
                cur_mask = new_mask[name]
                # prev_pruned True = was pruned; cur_mask must be False there.
                violators = prev_pruned & cur_mask
                assert int(violators.sum().item()) == 0, (
                    f"cycle {cycle} {name}: {int(violators.sum().item())} "
                    "previously-pruned weights got resurrected"
                )
        pruned_positions = {n: ~m for n, m in new_mask.items()}
        last_s = s
        mask = new_mask
    # After 3 cycles: 1 - 0.8^3 = 48.8%
    assert abs(last_s - (1.0 - 0.8 ** 3)) < 0.01


# ── FP4 pack/unpack ──────────────────────────────────────────────────────


def test_fp4_pack_unpack_roundtrip() -> None:
    """Nibble pack/unpack must be exact for arbitrary length."""
    for n in (0, 1, 2, 3, 7, 16, 33):
        values = [(i * 7) % 16 for i in range(n)]
        data = fp4_pack_values(values)
        # Bytes used = ceil(n / 2)
        assert len(data) == (n + 1) // 2
        decoded = fp4_unpack_values(data, n)
        assert decoded == values, f"mismatch at n={n}: {decoded} vs {values}"
