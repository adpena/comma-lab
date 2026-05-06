"""Roundtrip tests for ``tac.semantic_quantization``.

Check 46 contract: ``unquantize(quantize(x)) ≈ x`` to a known tolerance.

``quantize_tensor(x, bits)`` returns the dequantized values directly
(it stores ``q * scale`` in the result), so the natural roundtrip is to
compare its output to the input within ``scale / 2`` per element.

For ``semantic_adaptive_quantize``, each tensor in the state dict gets
its own bit budget; we verify per-tensor reconstruction error is bounded
by the per-tensor scale.
"""

from __future__ import annotations

import torch

from tac.semantic_quantization import (
    CLASS_NAMES,
    DEFAULT_CLASS_BITS,
    quantize_tensor,
    semantic_adaptive_quantize,
)
from tac.semantic_label_contract import (
    CONTEST_SEGNET_CLASSES,
    CONTEST_SEGNET_CLASS_NAMES,
    SELFCOMP_CLASS_TO_GRAY,
    validate_contest_class_table,
)


# ── quantize_tensor (uniform symmetric) ───────────────────────────────────


def test_quantize_tensor_roundtrip_8bit_within_step() -> None:
    """8-bit symmetric quantizer: max error ≤ scale/2 per element."""
    torch.manual_seed(0)
    x = torch.randn(64, 32) * 2.0
    quantized, scale = quantize_tensor(x, bits=8)
    err = (quantized - x).abs().max().item()
    # 8-bit signed quantizer: qmax = 127. Step = amax / 127. Round-to-nearest
    # gives max error ≤ step / 2 ≈ amax / 254.
    amax = x.abs().max().item()
    expected = amax / 127.0 / 2.0
    # Allow a small margin for clamp behavior at the extremes.
    assert err <= expected * 2.0 + 1e-6, (
        f"8-bit quantize err {err:.6f} > {expected * 2.0:.6f}"
    )
    assert scale > 0


def test_quantize_tensor_roundtrip_4bit_within_step() -> None:
    """4-bit symmetric quantizer has a much larger step."""
    torch.manual_seed(1)
    x = torch.randn(32, 16)
    quantized, _ = quantize_tensor(x, bits=4)
    err = (quantized - x).abs().max().item()
    amax = x.abs().max().item()
    # 4-bit signed: qmax = 7, step = amax / 7. Max error ≤ step.
    assert err <= amax / 7.0 + 1e-5, (
        f"4-bit quantize err {err:.6f} > {amax / 7.0:.6f}"
    )


def test_quantize_tensor_higher_bits_smaller_error() -> None:
    """Monotonicity: more bits → less error."""
    torch.manual_seed(2)
    x = torch.randn(32, 16)
    err_by_bits = {}
    for bits in (2, 4, 6, 8):
        q, _ = quantize_tensor(x, bits=bits)
        err_by_bits[bits] = (q - x).abs().mean().item()
    assert err_by_bits[8] < err_by_bits[6] < err_by_bits[4] < err_by_bits[2], (
        f"non-monotonic: {err_by_bits}"
    )


def test_quantize_tensor_zero_input() -> None:
    """Edge: all-zero input must roundtrip exactly without divide-by-zero."""
    x = torch.zeros(8, 4)
    quantized, scale = quantize_tensor(x, bits=8)
    assert torch.allclose(quantized, x)
    assert scale == 1.0


# ── semantic_adaptive_quantize ────────────────────────────────────────────


def test_semantic_adaptive_quantize_per_tensor_roundtrip() -> None:
    """Per-tensor reconstruction error bounded by per-tensor max scale.

    Class-specific tensors use per-class bits (default road/lane=8).
    Backbone tensors use uniform max bits (8). The reconstructed tensor
    is what's stored, so we compare it to the float original within the
    bound implied by the smallest bit count used on each row.
    """
    torch.manual_seed(3)
    state = {
        "clade.class_gamma.weight": torch.randn(5, 32) * 0.5,
        "conv1.weight": torch.randn(32, 3, 3, 3) * 0.3,
    }
    result = semantic_adaptive_quantize(state)
    quantized = result["quantized_state_dict"]
    assert set(quantized.keys()) == set(state.keys())

    # Backbone (uniform 8-bit): per-element err ≤ amax / 127.
    bb_orig = state["conv1.weight"]
    bb_back = quantized["conv1.weight"]
    bb_err = (bb_back - bb_orig).abs().max().item()
    bb_amax = bb_orig.abs().max().item()
    assert bb_err <= bb_amax / 127.0 * 1.5 + 1e-6, (
        f"backbone roundtrip err {bb_err:.6f} > expected step"
    )

    # Class-specific (per-row bits): each row has its own scale; bound row
    # errors by that row's own scale so a 4-bit row gets a 4-bit budget.
    cs_orig = state["clade.class_gamma.weight"]
    cs_back = quantized["clade.class_gamma.weight"]
    bits_per_class = result["bits_used"]["clade.class_gamma.weight"]
    assert isinstance(bits_per_class, list) and len(bits_per_class) == 5
    for cls_idx, bits in enumerate(bits_per_class):
        row_orig = cs_orig[cls_idx]
        row_back = cs_back[cls_idx]
        amax = row_orig.abs().max().item()
        qmax = (1 << (bits - 1)) - 1
        step = amax / max(qmax, 1)
        err = (row_back - row_orig).abs().max().item()
        assert err <= step + 1e-5, (
            f"class {cls_idx} bits={bits}: roundtrip err {err:.6f} > "
            f"step {step:.6f}"
        )


def test_semantic_adaptive_quantize_savings_positive() -> None:
    """Mixed precision must report positive savings vs uniform 32-bit."""
    state = {
        "clade.class_gamma.weight": torch.randn(5, 8),
        "conv1.weight": torch.randn(8, 3, 3, 3),
    }
    result = semantic_adaptive_quantize(state)
    assert result["savings_estimate"] > 0.0


def test_default_class_bits_table_invariants() -> None:
    """Table sanity: follows canonical contest SegNet class IDs."""
    assert len(DEFAULT_CLASS_BITS) == 5
    assert CLASS_NAMES == {
        0: "road",
        1: "lane_markings",
        2: "undrivable",
        3: "movable",
        4: "my_car",
    }
    assert DEFAULT_CLASS_BITS[0] == 8
    assert DEFAULT_CLASS_BITS[1] == 8
    assert DEFAULT_CLASS_BITS[2] <= DEFAULT_CLASS_BITS[0]


def test_semantic_label_contract_pins_comma10k_zero_index_order() -> None:
    """comma10k is one-based; contest tensors are the same order zero-based."""
    validate_contest_class_table()
    assert [item.comma10k_id for item in CONTEST_SEGNET_CLASSES] == [1, 2, 3, 4, 5]
    assert CONTEST_SEGNET_CLASS_NAMES == {
        0: "road",
        1: "lane_markings",
        2: "undrivable",
        3: "movable",
        4: "my_car",
    }
    assert SELFCOMP_CLASS_TO_GRAY == {
        0: 0,
        1: 255,
        2: 64,
        3: 192,
        4: 128,
    }
