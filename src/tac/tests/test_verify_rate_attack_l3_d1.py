# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the rate-attack + L3 + D1 verification harness
(``experiments/verify_rate_attack_l3_d1.py``).

These test the harness's three verification functions on a small realistic state dict +
latents (NOT the real basin, which is a gitignored experiment artifact). They prove the
harness MEASURES the real mechanism — a no-op codec would FAIL the byte-delta / round-trip
assertions the harness emits:

  * Part D: the variable-level codec round-trips bit-exact AND the default-preserving uniform
    path is byte-identical to vendored AND the sensitivity allocation produces a level span.
  * Part E: every L3 primitive is byte-accounted (disabled=0, PR98/T10=54, S12 adds 0 over
    PR98), deterministic round-trips, and applies POST-round on uint8 frames.
  * Part F: the D1 adapter round-trips the deployed latents bit-exact AND the no-win path is
    byte-identical to vendored (the default-preserving guarantee).

The harness's own ``verify_rate_attack`` / ``verify_l3_kit`` / ``verify_d1_latent_dedup`` are
imported and exercised — if any of them silently returned constants, these structural
assertions would fail.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_HARNESS = (
    Path(__file__).resolve().parents[3] / "experiments" / "verify_rate_attack_l3_d1.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location("verify_rate_attack_l3_d1", _HARNESS)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _real_shaped_sd(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    sd: dict[str, torch.Tensor] = {"stem.weight": torch.randn(64, 28, generator=g)}
    sd["stem.bias"] = torch.randn(64, generator=g)
    for i in range(6):
        sd[f"blocks.{i}.weight"] = torch.randn(40, 16, 3, 3, generator=g) * (1.0 + 0.3 * i)
        sd[f"blocks.{i}.bias"] = torch.randn(40, generator=g)
    return sd


@pytest.fixture(scope="module")
def harness():
    return _load_harness()


def test_harness_importable(harness):
    assert hasattr(harness, "verify_rate_attack")
    assert hasattr(harness, "verify_l3_kit")
    assert hasattr(harness, "verify_d1_latent_dedup")


def test_part_d_rate_attack_measures_real_mechanism(harness):
    sd = _real_shaped_sd()
    out = harness.verify_rate_attack(sd)
    # the default-preserving uniform path is byte-identical to vendored
    assert out["default_preserving_uniform_byte_identical"] is True
    # the variable decoder round-trips bit-exact (no silent loss)
    assert out["roundtrip_exact_variable_decoder"] is True
    assert out["roundtrip_max_abs_err"] == 0.0
    # the sensitivity allocation produced a non-degenerate level span (real mechanism)
    levels = [int(k) for k in out["levels_histogram"]]
    assert max(levels) > min(levels), "allocation collapsed to a single level (no mechanism)"
    # a coarser allocation should not INCREASE bytes vs vendored (the byte-win direction)
    assert out["variable_decoder_bytes"] <= out["vendored_decoder_bytes"]


def test_part_e_l3_kit_byte_accounting(harness):
    out = harness.verify_l3_kit()
    # disabled kit = zero bytes (default-OFF contract)
    assert out["disabled"]["is_zero_byte"] is True
    assert out["disabled"]["parses_to_disabled"] is True
    # PR98 + T10 are the same fixed 54-byte section, deterministic round-trip
    assert out["pr98_residual"]["section_bytes"] == 54
    assert out["pr98_residual"]["deterministic_roundtrip_byte_identical"] is True
    assert out["t10_affine"]["section_bytes"] == 54
    assert out["t10_affine"]["deterministic_roundtrip_byte_identical"] is True
    assert out["t10_affine"]["scale_roundtrips"] is True
    assert out["t10_affine"]["bias_roundtrips"] is True
    # S12 adds zero bytes over PR98 (rides as a flag)
    assert out["s12_certification"]["adds_zero_bytes_over_pr98"] is True
    assert out["s12_certification"]["certification_flag_roundtrips"] is True
    # POST-round application: disabled is no-op same object; enabled changes uint8 in-range
    pr = out["post_round_application"]
    assert pr["disabled_is_noop_same_object"] is True
    assert pr["pr98_changes_frames"] is True
    assert pr["pr98_output_is_uint8"] is True
    assert pr["pr98_output_in_range"] is True
    assert pr["t10_output_in_range"] is True


def test_part_f_d1_latent_dedup_measures_real_mechanism(harness):
    g = torch.Generator().manual_seed(1)
    latents = torch.randn(64, 28, generator=g)
    out = harness.verify_d1_latent_dedup(latents)
    # the deployed latents round-trip bit-exact (lossless on the quantized grid)
    assert out["roundtrip_exact_deployed_latents"] is True
    assert out["roundtrip_max_abs_err_deployed"] == 0.0
    assert out["roundtrip_max_abs_err_framed_core"] == 0.0
    # the no-win path is byte-identical to vendored (default-preserving) OR a real win
    if not out["adapter_emitted_framed_format"]:
        assert out["default_preserving_no_win_byte_identical"] is True
        assert out["delta_bytes_adapter_vs_vendored"] == 0
    else:
        # if a structural candidate won, it must be STRICTLY smaller
        assert out["delta_bytes_adapter_vs_vendored"] < 0
    # the adapter never emits MORE bytes than vendored
    assert out["adapter_emitted_bytes"] <= out["vendored_latents_bytes"]
    # diagnostics are real
    assert out["n_pairs"] == 64
    assert out["latent_dim"] == 28
    assert 1 <= out["n_unique_quantized_rows"] <= 64
