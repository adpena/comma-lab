# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the variable-level decoder codec
(``tac.losses.variable_level_codec``) — the Lever-4 OPTIMIZE reverse-waterfill.

The module ACTUALLY quantizes each tensor at its supplied ``n_levels`` and round-trips
bit-exactly. These tests verify the MECHANISM (not constants):

1. ``decode_decoder_variable(encode_decoder_variable(quantize_state_dict_variable(...)))``
   reconstructs the dequantized weights BIT-EXACTLY (the round-trip contract).
2. A COARSER per-tensor allocation produces a STRICTLY SMALLER brotli blob than all-127 on
   REAL-shaped weights (the byte-win mechanism — a constant return would FAIL this).
3. ``build_decoder_blob_variable_or_vendored`` with an all-uniform allocation emits the
   VENDORED ``encode_decoder`` bytes EXACTLY (the default-preserving guard — byte-identical
   archive until a non-uniform allocation is supplied).
4. The coarsened-tensor dequant has FEWER distinct values than the 127-level dequant (the
   coarse grid is REAL, not a relabelled 127 grid).
5. ``levels_from_sensitivity_for_codec`` is default-preserving (None/uniform -> all 127) and
   gives HIGH-sensitivity tensors MORE levels than LOW-sensitivity tensors.
"""
from __future__ import annotations

import numpy as np
import torch

from tac.losses.variable_level_codec import (
    VariableLevelConfig,
    build_decoder_blob_variable_or_vendored,
    decode_decoder_variable,
    encode_decoder_variable,
    levels_from_sensitivity_for_codec,
    quantize_state_dict_variable,
)


def _real_shaped_sd(seed: int = 0) -> dict[str, torch.Tensor]:
    """A small state dict with realistic (smooth-ish) weight tensors."""
    g = torch.Generator().manual_seed(seed)
    return {
        "blocks.0.weight": torch.randn(8, 4, 3, 3, generator=g),
        "blocks.0.bias": torch.randn(8, generator=g),
        "blocks.1.weight": torch.randn(4, 8, 3, 3, generator=g),
        "rgb.weight": torch.randn(3, 4, 1, 1, generator=g),
    }


def test_variable_codec_round_trips_bit_exact():
    """encode -> decode reconstructs the per-tensor dequant weights bit-exactly."""
    sd = _real_shaped_sd()
    levels = {"blocks.0.weight": 64, "blocks.1.weight": 100, "rgb.weight": 127}
    q_sd = quantize_state_dict_variable(sd, levels)
    blob = encode_decoder_variable(q_sd)
    dec = decode_decoder_variable(blob)
    # Reference: dequant each tensor by hand from the quantized record.
    for name, (q, scale, shape, _nl) in q_sd.items():
        want = torch.from_numpy(q.astype(np.float32).reshape(shape)) * scale
        assert torch.equal(dec[name], want), f"variable round-trip not bit-exact on {name}"


def test_coarser_allocation_gives_strictly_smaller_blob():
    """A coarse per-tensor allocation -> STRICTLY smaller brotli blob than all-127 (the
    byte-win mechanism, not a constant)."""
    sd = _real_shaped_sd(seed=1)
    names = list(sd.keys())
    uniform = quantize_state_dict_variable(sd, None)  # all 127
    coarse_levels = dict.fromkeys(names, 32)  # aggressive coarsening
    coarse = quantize_state_dict_variable(sd, coarse_levels)
    blob_u = encode_decoder_variable(uniform)
    blob_c = encode_decoder_variable(coarse)
    assert len(blob_c) < len(blob_u), (
        f"coarse blob {len(blob_c)} not smaller than uniform {len(blob_u)} — "
        "the variable codec did not actually shrink bytes (FAKE)"
    )


def test_uniform_builder_is_byte_identical_to_vendored():
    """``build_decoder_blob_variable_or_vendored`` with all-uniform allocation emits the
    EXACT vendored ``encode_decoder`` bytes (the default-preserving guard)."""
    from tac.torch_vehicle.vendored_imports import import_vendored

    codec = import_vendored("codec")
    sd = _real_shaped_sd(seed=2)
    vend = codec.encode_decoder(codec.quantize_state_dict(sd))
    blob, is_var = build_decoder_blob_variable_or_vendored(sd, None)
    assert is_var is False
    assert blob == vend, "uniform builder NOT byte-identical to vendored encode_decoder"
    # An explicitly-all-127 allocation must ALSO be vendored-identical.
    blob2, is_var2 = build_decoder_blob_variable_or_vendored(sd, dict.fromkeys(sd, 127))
    assert is_var2 is False and blob2 == vend


def test_variable_builder_round_trips_through_module_decoder():
    """A non-uniform allocation -> variable format that ``decode_decoder_variable`` reads;
    the dequant matches the hand-quantized reference (the deployed-grid contract)."""
    sd = _real_shaped_sd(seed=3)
    levels = {"blocks.0.weight": 48, "blocks.1.weight": 96, "rgb.weight": 127, "blocks.0.bias": 127}
    blob, is_var = build_decoder_blob_variable_or_vendored(sd, levels)
    assert is_var is True
    dec = decode_decoder_variable(blob)
    ref = quantize_state_dict_variable(sd, levels)
    for name, (q, scale, shape, _nl) in ref.items():
        want = torch.from_numpy(q.astype(np.float32).reshape(shape)) * scale
        assert torch.equal(dec[name], want), f"variable builder round-trip mismatch on {name}"


def test_coarse_dequant_has_fewer_distinct_values():
    """The coarsened tensor's dequant has FEWER distinct values than the 127-level dequant —
    the coarse grid is a REAL coarser grid, not a relabelled 127 grid."""
    sd = {"w": torch.randn(64, 64, generator=torch.Generator().manual_seed(4))}
    q127 = quantize_state_dict_variable(sd, {"w": 127})
    q32 = quantize_state_dict_variable(sd, {"w": 32})
    d127 = np.unique(q127["w"][0]).size  # distinct int8 symbols at 127
    d32 = np.unique(q32["w"][0]).size  # distinct int8 symbols at 32
    assert d32 < d127, f"32-level distinct symbols {d32} not < 127-level {d127} (grid not coarser)"
    assert d32 <= 65, f"32-level produced {d32} symbols, expected <= 2*32+1"


def test_levels_from_sensitivity_default_preserving_and_monotone():
    """None/uniform sensitivity -> all base 127 (default-preserving); a real sensitivity
    gives HIGH-sensitivity tensors MORE levels than LOW (the water-filling direction)."""
    names = ["a", "b", "c", "d"]
    # None -> all 127
    lev_none = levels_from_sensitivity_for_codec(None, names)
    assert all(v == 127 for v in lev_none.values())
    # uniform -> all 127
    lev_uniform = levels_from_sensitivity_for_codec(dict.fromkeys(names, 5.0), names)
    assert all(v == 127 for v in lev_uniform.values())
    # ramp sensitivity -> monotone level counts (highest sensitivity gets the most levels)
    sens = {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}
    lev = levels_from_sensitivity_for_codec(sens, names)
    assert lev["a"] <= lev["b"] <= lev["c"] <= lev["d"], f"levels not monotone in sensitivity: {lev}"
    assert lev["d"] == 127 and lev["a"] < 127, f"endpoints wrong: {lev}"


def test_min_abs_levels_floor_respected():
    """A near-zero-sensitivity tensor never collapses below the min_abs_levels floor."""
    cfg = VariableLevelConfig(min_abs_levels=16)
    sd = {"w": torch.randn(16, 16, generator=torch.Generator().manual_seed(9))}
    q = quantize_state_dict_variable(sd, {"w": 1}, cfg)  # ask for 1 -> floored to 16
    _arr, _scale, _shape, nl = q["w"]
    assert nl == 16, f"min_abs_levels floor not applied: got {nl}"


def test_codec_grid_matches_score_aware_qat_grid():
    """The variable-codec level map (``levels_from_sensitivity_for_codec``) must equal the
    score-aware-QAT training grid (``per_tensor_levels_from_sensitivity``) for the SAME
    sensitivity — so the DEPLOYED codec grid is exactly the grid the QAT trained the decoder
    to be robust at (the integration contract that makes the reverse-waterfill coherent)."""
    from tac.torch_vehicle.score_aware_qat import (
        ScoreAwareQATConfig,
        per_tensor_levels_from_sensitivity,
    )

    names = ["stem", "blocks.0", "blocks.1", "refine.0", "rgb_0"]
    sens = {"stem": 1.0, "blocks.0": 5.0, "blocks.1": 3.0, "refine.0": 0.1, "rgb_0": 9.0}
    codec_levels = levels_from_sensitivity_for_codec(sens, names)
    qat_levels = per_tensor_levels_from_sensitivity(sens, names, ScoreAwareQATConfig())
    assert codec_levels == qat_levels, (
        f"codec grid {codec_levels} != QAT training grid {qat_levels} — the deployed codec "
        "grid does not match the grid the score-aware QAT trained at (incoherent waterfill)"
    )


def test_variable_blob_strictly_smaller_on_real_basin_score_aware_allocation():
    """On the REAL basin EMA weights with a REAL score-aware allocation, the variable codec
    blob is STRICTLY smaller than the vendored-127 blob (the deployed byte win the vendored
    127-requant cannot deliver). Skips if the basin checkpoint / vendored codec is absent."""
    import sys
    from pathlib import Path

    pytest = __import__("pytest")
    repo = Path(__file__).resolve().parents[3]
    basin = (
        repo / "experiments/results/forkpoints/basin_bc20_20260612T121523Z"
        / "torch_vehicle_checkpoint_state.pt"
    )
    if not basin.exists():
        pytest.skip("basin checkpoint unavailable")
    if str(repo / "src") not in sys.path:
        sys.path.insert(0, str(repo / "src"))
    try:
        from tac.torch_vehicle.vendored_imports import import_vendored

        codec = import_vendored("codec")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"vendored codec unavailable: {exc}")

    ck = torch.load(basin, map_location="cpu", weights_only=False)
    sd = {k: v.detach().float() for k, v in ck["ema_decoder"].items()}
    names = list(sd.keys())
    # A real score-aware allocation: coarsen the weight tensors by index parity as a stand-in
    # for a sensitivity ramp (the exact sensitivity is measured by the probe; here we only
    # need a NON-uniform allocation to prove the codec shrinks deployed bytes).
    weight_keys = [k for k in names if k.endswith(".weight")]
    levels = dict.fromkeys(names, 127)
    for i, k in enumerate(weight_keys):
        levels[k] = 64 if i % 2 == 0 else 100
    vend_blob = codec.encode_decoder(codec.quantize_state_dict(sd))
    var_blob, is_var = build_decoder_blob_variable_or_vendored(sd, levels)
    assert is_var is True
    assert len(var_blob) < len(vend_blob), (
        f"variable blob {len(var_blob)} not smaller than vendored {len(vend_blob)} on the "
        "real basin weights — the deployed reverse-waterfill did not shrink bytes (FAKE)"
    )
