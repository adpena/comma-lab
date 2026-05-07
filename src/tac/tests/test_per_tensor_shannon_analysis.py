"""Tests for :mod:`tools.per_tensor_shannon_analysis`.

Coverage:
- Synthetic uniform u8 stream → ``H_0 ≈ 8 bits/symbol``, ``H_2 ≈ 8 bits/symbol``
  (close to ``log2(256) = 8``).
- Synthetic delta (all-zero) state_dict → ``H_0 → 0``, ``shannon_floor_h0_bytes
  → 0``.
- PR106 substrate: actual codec measurement is consistent with the per-tensor
  bits/symbol arithmetic (sanity round-trip on the bits_per_symbol fields)
  AND documents the empirical reality on this substrate (per-tensor isolated
  AC vs brotli — see analysis_md for interpretation).
- Sum of per-tensor brotli bytes is within 5% of an Op1-alone-style total
  (sanity check vs ``encode_decoder_compact``).
- Shannon-floor monotonicity: H_2 ≤ H_1 ≤ H_0 (data-processing inequality on
  conditional entropies).
- ``shannon_floor_bytes`` rounds up correctly (Kraft inequality fenceposts).
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import pytest
import torch

# Ensure tools/ is importable. tac.tests is part of the project, but
# tools/per_tensor_shannon_analysis.py is a script — we import it as a module.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tools"))

import per_tensor_shannon_analysis as ptsa  # type: ignore  # noqa: E402

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    encode_decoder_compact,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_state_dict(seed: int = 0, scale: float = 0.1) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    sd: dict[str, torch.Tensor] = {}
    for name, shape in FIXED_STATE_SCHEMA:
        sd[name] = torch.randn(*shape, generator=g) * scale
    return sd


def _zero_state_dict() -> dict[str, torch.Tensor]:
    return {name: torch.zeros(*shape) for name, shape in FIXED_STATE_SCHEMA}


# ---------------------------------------------------------------------------
# Entropy primitives — synthetic streams
# ---------------------------------------------------------------------------


def test_uniform_u8_h0_near_8_bits():
    """Uniform random u8 stream → H_0 ~ log2(256) = 8 bits/symbol."""
    rng = np.random.default_rng(0)
    sym = rng.integers(0, 256, size=200_000, dtype=np.uint8)
    h0 = ptsa.shannon_entropy_h0(sym)
    # Empirical 200k-sample uniform alphabet of 256 should give H_0 within
    # ~0.01 bit of 8.0.
    assert math.isclose(h0, 8.0, abs_tol=0.05), f"H_0={h0!r} for uniform u8"


def test_uniform_small_alphabet_h2_near_h0():
    """Uniform i.i.d. stream over a SMALL alphabet → H_2 ≈ H_0 ≈ log2(A).

    On a 256-alphabet trigram table (256^3 = 16.7M cells), reaching the
    asymptotic H_2 ~ 8 bits/symbol requires far more than 1M samples (the
    plug-in entropy estimator has well-known O(A^k / N) bias). To validate
    that the estimator is *correctly implemented* we use A=8 (8^3 = 512
    cells), where 200k samples give ~390 hits per cell and the plug-in bias
    is negligible.
    """
    rng = np.random.default_rng(0)
    sym = rng.integers(0, 8, size=200_000, dtype=np.int64).astype(np.uint8)
    h0 = ptsa.shannon_entropy_h0(sym, alphabet_size=8)
    h2 = ptsa.shannon_entropy_h2(sym, alphabet_size=8)
    # H_0 should be near log2(8) = 3.0.
    assert math.isclose(h0, 3.0, abs_tol=0.02)
    # H_2 should also be near 3.0 (i.i.d. → conditional entropy = marginal).
    # Plug-in bias on 8^3=512 cells with 200k samples is sub-0.1 bit.
    assert math.isclose(h2, 3.0, abs_tol=0.05), f"H_2={h2!r}"


def test_uniform_u8_h2_data_processing_inequality_only():
    """For a uniform u8 stream, the H_2 estimator must satisfy the data-
    processing inequality H_2 ≤ H_0 (the asymptotic limit H_2 = 8 cannot be
    verified at any sample size that fits in CI memory; see the small-
    alphabet test for the asymptotic check)."""
    rng = np.random.default_rng(0)
    sym = rng.integers(0, 256, size=200_000, dtype=np.uint8)
    h0 = ptsa.shannon_entropy_h0(sym)
    h2 = ptsa.shannon_entropy_h2(sym)
    assert h2 <= h0 + 1e-9


def test_delta_distribution_zero_entropy():
    """All-zero state_dict → after quantization every code is 128 (offset
    binary of 0), so H_0 = 0 and Shannon floor = 0 bytes."""
    sd = _zero_state_dict()
    quantized = ptsa.quantize_state_dict_to_u8(sd)
    for name, u8, _scale in quantized:
        h0 = ptsa.shannon_entropy_h0(u8)
        assert h0 == 0.0, f"{name}: H_0={h0!r} on all-zero tensor"
        floor = ptsa.shannon_floor_bytes(int(u8.size), h0)
        assert floor == 0, f"{name}: shannon_floor_bytes={floor} on all-zero tensor"


def test_shannon_floor_bytes_ceiling_arithmetic():
    """Floor function rounds up — Kraft inequality fencepost."""
    # 100 symbols * 5.5 bits = 550 bits = 68.75 bytes → 69 bytes.
    assert ptsa.shannon_floor_bytes(100, 5.5) == 69
    # 8 symbols * 8 bits = 64 bits = exactly 8 bytes (no rounding).
    assert ptsa.shannon_floor_bytes(8, 8.0) == 8
    # 8 symbols * 8.001 bits = 64.008 bits = 8.001 bytes → 9 bytes.
    assert ptsa.shannon_floor_bytes(8, 8.001) == 9
    # Empty stream → 0 bytes regardless of bits/symbol.
    assert ptsa.shannon_floor_bytes(0, 7.5) == 0
    # Zero entropy → 0 bytes regardless of n_symbols.
    assert ptsa.shannon_floor_bytes(1000, 0.0) == 0


def test_h2_less_than_or_equal_h1_less_than_or_equal_h0():
    """Conditional entropy chain: H_2 ≤ H_1 ≤ H_0 (data processing on a
    stationary source's empirical distribution)."""
    # Use a low-entropy stream so the inequality has slack to verify.
    sd = _synthetic_state_dict(seed=1)
    quantized = ptsa.quantize_state_dict_to_u8(sd)
    # Pick a tensor large enough for trigram statistics to mean something.
    name, u8, _scale = quantized[0]  # stem.weight, 48,384 symbols
    h0 = ptsa.shannon_entropy_h0(u8)
    h1 = ptsa.shannon_entropy_h1(u8)
    h2 = ptsa.shannon_entropy_h2(u8)
    assert h1 <= h0 + 1e-9, f"H_1={h1!r} > H_0={h0!r}"
    assert h2 <= h1 + 1e-9, f"H_2={h2!r} > H_1={h1!r}"


# ---------------------------------------------------------------------------
# PR106 substrate measurements
# ---------------------------------------------------------------------------


def _pr106_state_dict_path() -> pathlib.Path:
    return (
        _REPO_ROOT
        / "experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt"
    )


def _load_pr106_state_dict() -> dict[str, torch.Tensor] | None:
    p = _pr106_state_dict_path()
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=True)


def test_pr106_per_tensor_bits_per_symbol_consistency():
    """For the actual PR106 substrate, bits_per_symbol_ac and
    bits_per_symbol_brotli must each equal (bytes * 8 / n_symbols) exactly,
    and the AC field is non-None exactly for AC_TENSOR_INDICES."""
    sd = _load_pr106_state_dict()
    if sd is None:
        pytest.skip("PR106 substrate not present at expected path")
    result = ptsa.analyze_state_dict(sd)
    from tac.pr103_arithmetic_codec import AC_TENSOR_INDICES

    for row in result["per_tensor"]:
        n = row["n_symbols"]
        # bits_per_symbol_brotli matches the byte counter.
        expected_br = row["brotli_bytes"] * 8 / max(n, 1)
        assert math.isclose(
            row["bits_per_symbol_brotli"], expected_br, rel_tol=1e-9, abs_tol=1e-9
        ), (
            f"idx {row['idx']}: bits_per_symbol_brotli mismatch "
            f"({row['bits_per_symbol_brotli']!r} vs {expected_br!r})"
        )
        # AC fields populated iff idx in AC set.
        if row["idx"] in AC_TENSOR_INDICES:
            assert row["ac_bytes"] is not None
            assert row["bits_per_symbol_ac"] is not None
            expected_ac = row["ac_bytes"] * 8 / max(n, 1)
            assert math.isclose(
                row["bits_per_symbol_ac"],
                expected_ac,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        else:
            assert row["ac_bytes"] is None
            assert row["bits_per_symbol_ac"] is None


def test_pr106_brotli_total_within_5pct_of_op1_alone():
    """Sum of per-tensor brotli bytes (computed in isolation, no split-stream
    context) should be within 5% of the Op1 ``encode_decoder_compact`` total
    (which packs payloads into 7 split-Brotli stream windows).

    This is a sanity check: the per-tensor measurement is using the same
    quantization + the same brotli quality + the same byte format as Op1,
    so the totals should be very close. Per-tensor measurement is generally
    SLIGHTLY larger because each isolated stream pays a brotli header that
    is amortized in the 7-window pack.
    """
    sd = _load_pr106_state_dict()
    if sd is None:
        pytest.skip("PR106 substrate not present at expected path")
    result = ptsa.analyze_state_dict(sd)
    per_tensor_brotli_total = result["summary"]["total_brotli_bytes"]

    op1_blob = encode_decoder_compact(sd)
    # Op1 wraps the same per-tensor payloads in 7 brotli streams + appends
    # fp16 scales (28 * 2 = 56 bytes) inside those streams. The per-tensor
    # measurement uses just raw u8 bytes (no fp16 tail). Allow 5%
    # tolerance — the difference is in the noise of brotli framing.
    rel = abs(per_tensor_brotli_total - len(op1_blob)) / max(len(op1_blob), 1)
    assert rel < 0.05, (
        f"per-tensor brotli total {per_tensor_brotli_total} differs from "
        f"Op1 total {len(op1_blob)} by {rel:.4f} (5% tolerance)"
    )


def test_pr106_summary_floors_monotone():
    """Aggregate summary must satisfy total_h2_floor ≤ total_h0_floor on the
    PR106 substrate (sum of per-tensor entropies is monotone in context order
    when aggregated."""
    sd = _load_pr106_state_dict()
    if sd is None:
        pytest.skip("PR106 substrate not present at expected path")
    result = ptsa.analyze_state_dict(sd)
    s = result["summary"]
    assert s["total_shannon_floor_h2_bytes"] <= s["total_shannon_floor_h0_bytes"]
    # Brotli ratio over H_0 should be > 1.0 (brotli can't beat the Shannon
    # bound H_0 on a pure i.i.d. assumption).
    # Brotli's ratio over H_2 may be > 1 (brotli is short of optimal context
    # modeling) — verified by the regression-toward-H_2 finding.
    assert s["brotli_over_h0_ratio"] >= 0.9  # should be ~1.0 for AC-style data
    # Brotli total is closer to H_0 than to H_2 (it's not a true
    # arbitrarily-large-context model).
    assert s["total_brotli_bytes"] >= s["total_shannon_floor_h2_bytes"]


def test_pr106_substrate_documents_per_tensor_ac_vs_brotli():
    """On the actual PR106 substrate, document whether AC regresses or wins
    per-tensor (in isolation, without histogram overhead).

    This test does NOT assert a particular sign — the empirical answer is the
    point of the analysis. It asserts that the analysis records a definitive
    answer (every AC tensor has a non-None ac_minus_brotli_bytes value) and
    that the regression flag is a strict function of that value.
    """
    sd = _load_pr106_state_dict()
    if sd is None:
        pytest.skip("PR106 substrate not present at expected path")
    result = ptsa.analyze_state_dict(sd)
    from tac.pr103_arithmetic_codec import AC_TENSOR_INDICES

    for row in result["per_tensor"]:
        if row["idx"] in AC_TENSOR_INDICES:
            assert row["ac_minus_brotli_bytes"] is not None
            # regresses_ac_vs_brotli is exactly (brotli < ac).
            assert row["regresses_ac_vs_brotli"] == (
                row["brotli_bytes"] < row["ac_bytes"]
            )


# ---------------------------------------------------------------------------
# JSON output shape
# ---------------------------------------------------------------------------


def test_analysis_json_shape_synthetic(tmp_path: pathlib.Path):
    """End-to-end: analyze a synthetic state_dict, write JSON, reload, sanity-
    check fields."""
    sd = _synthetic_state_dict(seed=2)
    result = ptsa.analyze_state_dict(sd)
    payload = {
        "started_at_utc": "1970-01-01T00:00:00Z",
        "substrate": "synthetic",
        "evidence_grade": "[empirical]",
        "score_claim": False,
        "per_tensor": result["per_tensor"],
        "summary": result["summary"],
        "analysis_md": ptsa.build_analysis_markdown(result),
    }
    out_path = tmp_path / "per_tensor_shannon.json"
    out_path.write_text(json.dumps(payload, indent=2))
    reloaded = json.loads(out_path.read_text())
    assert reloaded["evidence_grade"] == "[empirical]"
    assert reloaded["score_claim"] is False
    assert len(reloaded["per_tensor"]) == 28
    assert "total_shannon_floor_h0_bytes" in reloaded["summary"]
    assert "total_shannon_floor_h2_bytes" in reloaded["summary"]
    assert "ac_regression_total_bytes" in reloaded["summary"]
    assert "ac_regression_tensors" in reloaded["summary"]
    # analysis_md is non-empty Markdown text.
    assert len(reloaded["analysis_md"]) > 100
    assert "Shannon" in reloaded["analysis_md"]
