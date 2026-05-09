"""Tests for the A1 UNIWARD/Hessian byte-candidate builder."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "build_uniward_stc_hessian_a1_v1.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "build_uniward_stc_hessian_a1_v1_under_test",
        TOOL,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_allocate_bits_treats_source_decoder_size_as_eight_bit_baseline() -> None:
    mod = _load_tool()
    sd = {
        "low.weight": torch.ones(100),
        "high.weight": torch.ones(100) * 4,
    }
    fisher = {"low.weight": 1.0e-4, "high.weight": 1.0}

    no_cut = mod.allocate_bits_per_tensor(
        sd,
        fisher,
        target_decoder_bytes=1000,
        source_decoder_bytes=1000,
        floor_bits=4,
        ceiling_bits=8,
    )
    cut = mod.allocate_bits_per_tensor(
        sd,
        fisher,
        target_decoder_bytes=750,
        source_decoder_bytes=1000,
        floor_bits=4,
        ceiling_bits=8,
    )

    assert no_cut == {"low.weight": 8, "high.weight": 8}
    assert min(cut.values()) < 8
    assert cut["high.weight"] >= cut["low.weight"]


def test_allocate_bits_charges_large_tensors_by_numel() -> None:
    mod = _load_tool()
    sd = {
        "small_high.weight": torch.ones(10) * 4,
        "large_low.weight": torch.ones(1000),
    }
    fisher = {"small_high.weight": 1.0, "large_low.weight": 1.0e-4}

    bits = mod.allocate_bits_per_tensor(
        sd,
        fisher,
        target_decoder_bytes=650,
        source_decoder_bytes=1000,
        floor_bits=4,
        ceiling_bits=8,
    )
    charged_bits = sum(sd[name].numel() * bit for name, bit in bits.items())

    assert bits["small_high.weight"] >= bits["large_low.weight"]
    assert charged_bits <= sum(t.numel() for t in sd.values()) * 8
    assert charged_bits >= sum(t.numel() for t in sd.values()) * 4
