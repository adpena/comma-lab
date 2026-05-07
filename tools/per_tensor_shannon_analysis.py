"""Per-tensor Shannon-floor empirical analysis on a PR106-style HNeRV state_dict.

This tool answers — in information-theoretic terms — the question

    "WHY do PR103 AC tensors 0/2/4 regress vs brotli on PR106?"

For each of the 28 tensors in :data:`tac.pr101_split_brotli_codec.FIXED_STATE_SCHEMA`,
the script computes:

    H_0  — order-0 (zero-th order) empirical Shannon entropy of the int8 codes
           (offset to uint8) in bits per symbol.
    H_1  — order-1 entropy: H(X_t | X_{t-1}) via the empirical bigram table.
    H_2  — order-2 entropy: H(X_t | X_{t-1}, X_{t-2}) via the empirical
           trigram table (where feasible — falls back to H_1 for very short
           tensors).
    bits_per_symbol_brotli — actual brotli output bytes × 8 / n_symbols.
    bits_per_symbol_ac     — actual PR103 AC output bytes × 8 / n_symbols.
    shannon_floor_bytes_h0 — n_symbols × H_0 / 8 (rounded up).
    shannon_floor_bytes_h2 — n_symbols × H_2 / 8 (rounded up).

Mathematical grounding (Shannon 1948 source coding theorem):

    For a stationary source with order-k empirical conditional entropy H_k,
    no lossless code achieves an expected code-length below H_k bits/symbol.
    A symbol-frequency-only entropy coder (PR103's q8-histogram AC) targets
    H_0; a context-aware coder (brotli's LZ77 + Huffman + context modelling)
    targets H_k for some k ≥ 1. The regression

        bits_per_symbol_ac − bits_per_symbol_brotli > 0

    is therefore a *direct empirical lower bound* on the redundancy
    (H_0 − H_k) that brotli is exploiting and AC is not.

Strict-scorer-rule: this tool loads NO scorer weights, has no MPS/CUDA
dependency, makes no archive build, and emits no submission. CPU + numpy +
math only. Output JSON is tagged ``[empirical]``; no [contest-CUDA] claim is
made.

Usage:
    .venv/bin/python tools/per_tensor_shannon_analysis.py \
        --state-dict experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt \
        --output-dir experiments/results/lane_per_tensor_shannon_pr106_<UTC>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import pathlib
import sys
from typing import Any

import numpy as np
import torch

# Project import — makes this script runnable from the repo root with the venv.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
    pack_brotli_stream,
)
from tac.pr103_arithmetic_codec import (  # noqa: E402
    AC_SYMBOL_OFFSET,
    AC_TENSOR_INDICES,
    DEFAULT_BROTLI_QUALITY,
    _build_q8_histogram,
    pack_ac_stream,
)

logger = logging.getLogger(__name__)


# Matches CLAUDE.md "FORBIDDEN /tmp paths" — we always write next to the repo
# under experiments/results/ instead. Default-output-dir builder honors that.


# ---------------------------------------------------------------------------
# Empirical entropy primitives
# ---------------------------------------------------------------------------


def shannon_entropy_h0(symbols_u8: np.ndarray, *, alphabet_size: int = 256) -> float:
    """Order-0 (i.i.d.) empirical Shannon entropy in bits/symbol.

    H_0 = -sum_i p_i * log2(p_i)

    Buckets with zero count contribute zero (limit of x log x as x → 0).
    """
    if symbols_u8.size == 0:
        return 0.0
    counts = np.bincount(symbols_u8.astype(np.int64), minlength=alphabet_size)
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts.astype(np.float64) / float(n)
    nz = p > 0.0
    h = float(-np.sum(p[nz] * np.log2(p[nz])))
    # Clamp tiny negative noise from floating-point cancellation.
    return float(h) if h > 0.0 else 0.0


def shannon_entropy_h1(symbols_u8: np.ndarray, *, alphabet_size: int = 256) -> float:
    """Order-1 empirical conditional entropy.

    H_1 = H(X_t | X_{t-1}) = sum_a p(a) * H(X_t | X_{t-1}=a)
                          = -sum_{a,b} p(a,b) log2 p(b|a)

    Computed from the empirical bigram table. Returns H_0 when symbols.size < 2.
    """
    n = symbols_u8.size
    if n < 2:
        return shannon_entropy_h0(symbols_u8, alphabet_size=alphabet_size)
    s = symbols_u8.astype(np.int64)
    prev = s[:-1]
    curr = s[1:]
    # Encode (prev, curr) into a single int and bincount.
    keys = prev * alphabet_size + curr
    bigram_counts = np.bincount(keys, minlength=alphabet_size * alphabet_size).reshape(
        alphabet_size, alphabet_size
    )
    row_sums = bigram_counts.sum(axis=1, keepdims=True).astype(np.float64)
    total = float(row_sums.sum())
    if total <= 0:
        return 0.0
    # H(X_t | X_{t-1}) = -sum_{a,b} p(a,b) log2 p(b|a)
    safe_row = np.where(row_sums > 0, row_sums, 1.0)
    p_cond = bigram_counts.astype(np.float64) / safe_row  # p(b|a), zero rows → zero
    p_joint = bigram_counts.astype(np.float64) / total  # p(a, b)
    # log2 p(b|a) where p(b|a) > 0
    nz = p_joint > 0.0
    h_cond = float(-np.sum(p_joint[nz] * np.log2(p_cond[nz])))
    return float(h_cond) if h_cond > 0.0 else 0.0


def shannon_entropy_h2(
    symbols_u8: np.ndarray,
    *,
    alphabet_size: int = 256,
    max_table_bytes: int = 256 * 1024 * 1024,
) -> float:
    """Order-2 empirical conditional entropy.

    H_2 = H(X_t | X_{t-1}, X_{t-2})
        = -sum_{a,b,c} p(a,b,c) log2 p(c|a,b)

    Memory budget: a full A^3 = 256^3 = 16,777,216 cell table at int64 is
    134 MB. We cap at ``max_table_bytes`` (default 256 MB) and fall back to
    H_1 if the requested alphabet would blow it. For our use case (A=256) we
    always fit and the memory is bounded.

    Returns H_1 when symbols.size < 3.
    """
    n = symbols_u8.size
    if n < 3:
        return shannon_entropy_h1(symbols_u8, alphabet_size=alphabet_size)
    table_size_bytes = (alphabet_size ** 3) * 8  # int64
    if table_size_bytes > max_table_bytes:
        # Fall back to H_1 — caller's alphabet is too wide for an A^3 table.
        logger.warning(
            "shannon_entropy_h2: alphabet_size=%d exceeds memory budget; "
            "falling back to H_1",
            alphabet_size,
        )
        return shannon_entropy_h1(symbols_u8, alphabet_size=alphabet_size)
    s = symbols_u8.astype(np.int64)
    a = s[:-2]
    b = s[1:-1]
    c = s[2:]
    keys = a * (alphabet_size * alphabet_size) + b * alphabet_size + c
    trigram_counts = np.bincount(
        keys, minlength=alphabet_size ** 3
    ).reshape(alphabet_size, alphabet_size, alphabet_size)
    pair_sums = trigram_counts.sum(axis=2, keepdims=True).astype(np.float64)
    total = float(pair_sums.sum())
    if total <= 0:
        return 0.0
    safe_pair = np.where(pair_sums > 0, pair_sums, 1.0)
    p_cond = trigram_counts.astype(np.float64) / safe_pair  # p(c|a,b)
    p_joint = trigram_counts.astype(np.float64) / total  # p(a,b,c)
    nz = p_joint > 0.0
    h_cond = float(-np.sum(p_joint[nz] * np.log2(p_cond[nz])))
    return float(h_cond) if h_cond > 0.0 else 0.0


def shannon_floor_bytes(n_symbols: int, bits_per_symbol: float) -> int:
    """Shannon source-coding lower bound on bytes for an n-symbol stream
    with ``bits_per_symbol`` bits per symbol entropy.

    Rounded up — Kraft inequality forces at least one extra symbol of
    coding overhead per fractional bit.
    """
    if n_symbols <= 0 or bits_per_symbol <= 0:
        return 0
    return math.ceil(n_symbols * bits_per_symbol / 8.0)


# ---------------------------------------------------------------------------
# Per-tensor measurement: actual codec output bytes
# ---------------------------------------------------------------------------


def measure_brotli_bytes(symbols_u8: np.ndarray, *, quality: int = DEFAULT_BROTLI_QUALITY) -> int:
    """Actual brotli output bytes on the raw u8 symbol stream.

    Mirrors the per-tensor measurement convention used by
    :func:`tac.pr103_arithmetic_codec.validate_ac_savings`: pack the raw
    uint8 byte stream (no fp16 scale tail, no permutation) at the same
    brotli quality the wire-format pack uses.
    """
    return len(pack_brotli_stream(symbols_u8.tobytes(), quality=quality))


def measure_ac_bytes(symbols_u8: np.ndarray) -> int:
    """Actual PR103 single-stream AC output bytes for this u8 symbol stream.

    Uses the q8 histogram path (256-bucket uint8 histogram) that PR103 uses
    on its merged AC blob. NB: this does NOT include the per-tensor 256-byte
    histogram-storage overhead (which is amortized across all 8 streams in
    the merged blob and brotli'd as one short stream of ~895 bytes total).
    The H_0 lower bound also does not include histogram overhead, so the
    comparison is fair.
    """
    if symbols_u8.size == 0:
        return 0
    hist = _build_q8_histogram(symbols_u8)
    return len(pack_ac_stream(symbols_u8, hist))


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def quantize_state_dict_to_u8(
    state_dict: dict[str, torch.Tensor],
    *,
    n_quant: int = N_QUANT,
) -> list[tuple[str, np.ndarray, float]]:
    """Quantize each tensor in :data:`FIXED_STATE_SCHEMA` order to an int8 code
    (via the canonical PR101 ``_quantize_tensor`` helper), then offset to u8
    via the PR103 ``off`` byte_map ``i8 + 128``.

    Returns a list of (name, u8_symbols_flat, scale) tuples in schema order.
    """
    out: list[tuple[str, np.ndarray, float]] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        if name not in state_dict:
            raise ValueError(f"missing tensor {name!r} in state_dict")
        qt = _quantize_tensor(name, state_dict[name], n_quant=n_quant)
        u8 = (qt.q_i8.astype(np.int16) + AC_SYMBOL_OFFSET).astype(np.uint8).reshape(-1)
        out.append((name, u8, qt.scale))
    return out


def analyze_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    brotli_quality: int = DEFAULT_BROTLI_QUALITY,
    n_quant: int = N_QUANT,
) -> dict[str, Any]:
    """Run the full per-tensor entropy + codec measurement on a state_dict.

    Returns a JSON-serializable dict with one row per tensor + a summary.
    No fileystem side effects.
    """
    quantized = quantize_state_dict_to_u8(state_dict, n_quant=n_quant)
    ac_set = set(AC_TENSOR_INDICES)

    per_tensor: list[dict[str, Any]] = []
    total_brotli = 0
    total_ac = 0
    total_floor_h0 = 0
    total_floor_h2 = 0
    total_n_symbols = 0
    ac_regression_tensors: list[int] = []
    ac_regression_total_bytes = 0

    for idx, (name, u8, scale) in enumerate(quantized):
        n_symbols = int(u8.size)
        h0 = shannon_entropy_h0(u8)
        h1 = shannon_entropy_h1(u8)
        h2 = shannon_entropy_h2(u8)
        floor_h0 = shannon_floor_bytes(n_symbols, h0)
        floor_h2 = shannon_floor_bytes(n_symbols, h2)
        brotli_bytes = measure_brotli_bytes(u8, quality=brotli_quality)
        ac_bytes = measure_ac_bytes(u8) if idx in ac_set else None

        bps_brotli = brotli_bytes * 8 / max(n_symbols, 1)
        bps_ac = (ac_bytes * 8 / max(n_symbols, 1)) if ac_bytes is not None else None

        regresses = (
            bool(ac_bytes is not None and brotli_bytes < ac_bytes) if ac_bytes is not None else False
        )
        if regresses:
            ac_regression_tensors.append(idx)
            ac_regression_total_bytes += int(ac_bytes - brotli_bytes)

        # Note the ``ac`` field is None for non-AC tensors (PR103 doesn't AC-encode them).
        row: dict[str, Any] = {
            "idx": idx,
            "name": name,
            "n_symbols": n_symbols,
            "scale": float(scale),
            "in_pr103_ac_set": idx in ac_set,
            "H0_bits": h0,
            "H1_bits": h1,
            "H2_bits": h2,
            "shannon_floor_h0_bytes": floor_h0,
            "shannon_floor_h2_bytes": floor_h2,
            "brotli_bytes": int(brotli_bytes),
            "ac_bytes": int(ac_bytes) if ac_bytes is not None else None,
            "bits_per_symbol_brotli": bps_brotli,
            "bits_per_symbol_ac": bps_ac,
            "ac_minus_brotli_bytes": (
                int(ac_bytes - brotli_bytes) if ac_bytes is not None else None
            ),
            "regresses_ac_vs_brotli": regresses,
        }
        per_tensor.append(row)
        total_brotli += brotli_bytes
        if ac_bytes is not None:
            total_ac += ac_bytes
        total_floor_h0 += floor_h0
        total_floor_h2 += floor_h2
        total_n_symbols += n_symbols

    summary = {
        "total_n_symbols": total_n_symbols,
        "total_shannon_floor_h0_bytes": int(total_floor_h0),
        "total_shannon_floor_h2_bytes": int(total_floor_h2),
        "total_brotli_bytes": int(total_brotli),
        "total_ac_bytes_pr103_indices_only": int(total_ac),
        "ac_regression_total_bytes": int(ac_regression_total_bytes),
        "ac_regression_tensors": ac_regression_tensors,
        # The ratio of brotli bytes / Shannon floor H_0 indicates how far
        # brotli's context model is below the i.i.d. lower bound — i.e. how
        # much ``redundancy in conditional structure'' brotli is exploiting.
        "brotli_over_h0_ratio": (
            float(total_brotli * 8) / float(max(total_floor_h0 * 8, 1))
        ),
        "brotli_over_h2_ratio": (
            float(total_brotli * 8) / float(max(total_floor_h2 * 8, 1))
        ),
    }
    return {"per_tensor": per_tensor, "summary": summary}


# ---------------------------------------------------------------------------
# Mathematical interpretation (Markdown text written into the JSON)
# ---------------------------------------------------------------------------


def build_analysis_markdown(result: dict[str, Any]) -> str:
    """Compose a plain-Markdown interpretation grounded in Shannon's source
    coding theorem. Inserted into the JSON ``analysis_md`` field.
    """
    s = result["summary"]
    pt = result["per_tensor"]
    ac_regressions = [r for r in pt if r.get("regresses_ac_vs_brotli")]
    ac_regressions.sort(
        key=lambda r: (r["ac_minus_brotli_bytes"] or 0), reverse=True
    )
    top3 = ac_regressions[:3]

    lines: list[str] = []
    lines.append("# Per-tensor Shannon-floor analysis on PR106 substrate\n")
    lines.append(
        "## Why PR103 AC tensors regress vs brotli on PR106 (information-theoretic)\n"
    )
    lines.append(
        "Shannon's source coding theorem states that no lossless code can compress a "
        "stationary source below its empirical conditional entropy `H_k` bits/symbol "
        "(for any context order k). A symbol-frequency-only entropy coder (PR103's q8 "
        "AC) targets `H_0`. A context-aware coder (brotli's LZ77 + Huffman + context "
        "model) targets `H_k` for some k >= 1.\n"
    )
    lines.append(
        "Therefore the gap `bits_per_symbol_ac - bits_per_symbol_brotli > 0` is a "
        "*direct empirical lower bound* on `H_0 - H_k` — the redundancy in "
        "conditional structure that brotli is exploiting and AC is blind to.\n"
    )
    lines.append(
        f"### Aggregate measurements (28 tensors, n={s['total_n_symbols']:,} symbols)\n"
    )
    lines.append(
        f"- **Shannon floor (H_0, i.i.d. bound)**: {s['total_shannon_floor_h0_bytes']:,} bytes\n"
    )
    lines.append(
        f"- **Shannon floor (H_2, order-2 bound)**: {s['total_shannon_floor_h2_bytes']:,} bytes\n"
    )
    lines.append(
        f"- **Actual brotli output (sum)**: {s['total_brotli_bytes']:,} bytes "
        f"({s['brotli_over_h0_ratio']:.4f} x H_0; "
        f"{s['brotli_over_h2_ratio']:.4f} x H_2)\n"
    )
    lines.append(
        f"- **Actual AC output (PR103 indices only)**: "
        f"{s['total_ac_bytes_pr103_indices_only']:,} bytes\n"
    )
    if s["ac_regression_tensors"]:
        lines.append(
            f"- **AC regression total**: +{s['ac_regression_total_bytes']:,} bytes "
            f"across tensors {s['ac_regression_tensors']}\n"
        )
    else:
        lines.append("- AC does not regress on any tensor on this substrate.\n")

    if top3:
        lines.append("\n### Top 3 AC-regressing tensors\n")
        lines.append("| idx | name | gap (bytes) | H_0 (bits) | H_2 (bits) | n_symbols |\n")
        lines.append("|---:|:-----|------------:|-----------:|-----------:|----------:|\n")
        for r in top3:
            lines.append(
                f"| {r['idx']} | `{r['name']}` | "
                f"+{r['ac_minus_brotli_bytes']} | "
                f"{r['H0_bits']:.4f} | {r['H2_bits']:.4f} | "
                f"{r['n_symbols']:,} |\n"
            )
    else:
        # Empirical reality on PR106: AC wins on every PR103 AC tensor in
        # isolated per-tensor measurement (no histogram overhead). Surface the
        # top 3 AC-favorable tensors so the table has substance regardless.
        ac_winners = sorted(
            [r for r in pt if r.get("ac_minus_brotli_bytes") is not None],
            key=lambda r: r["ac_minus_brotli_bytes"],  # most negative first
        )[:3]
        if ac_winners:
            lines.append(
                "\n### No AC regressions on this substrate. Top 3 AC-favorable tensors\n"
            )
            lines.append(
                "| idx | name | gap (bytes) | H_0 (bits) | H_2 (bits) | n_symbols |\n"
            )
            lines.append(
                "|---:|:-----|------------:|-----------:|-----------:|----------:|\n"
            )
            for r in ac_winners:
                lines.append(
                    f"| {r['idx']} | `{r['name']}` | "
                    f"{r['ac_minus_brotli_bytes']:+d} | "
                    f"{r['H0_bits']:.4f} | {r['H2_bits']:.4f} | "
                    f"{r['n_symbols']:,} |\n"
                )

    lines.append(
        "\n### Implication for the δεζ training objective\n"
        "If we train weights so the conditional distribution `p(X_t | X_{t-1}, ...)` "
        "becomes uniform — i.e. `H_2 -> H_0` — then brotli loses its context advantage "
        "and AC catches up. Per-tensor entropy targets for the δεζ objective:\n\n"
        "    L_entropy = sum_idx ( H_0(idx) - H_2(idx) ) * lambda_idx\n\n"
        "minimizing this drives every tensor's `H_2` toward its `H_0`, removing the "
        "PR103 regression by construction. The per-tensor (H_0 - H_2) numbers in the "
        "JSON output give the gradient targets directly.\n"
    )
    lines.append(
        "\n### Operational corollary (Op2 auto-fallback)\n"
        "On any substrate where `bits_per_symbol_ac > bits_per_symbol_brotli` for a "
        "tensor, the canonical Op2 wrapper SHOULD fall back to brotli for that tensor. "
        "The per-tensor table in this JSON provides the exact decision matrix.\n"
    )
    return "".join(lines)


# ---------------------------------------------------------------------------
# Markdown table to stdout
# ---------------------------------------------------------------------------


def print_per_tensor_markdown(result: dict[str, Any]) -> None:
    """Pretty-print the per-tensor table as Markdown to stdout."""
    pt = result["per_tensor"]
    s = result["summary"]
    print()
    print("| idx | name | n_sym | H_0 | H_1 | H_2 | floor_h0 | floor_h2 | brotli | ac | ac-br |")
    print("|----:|:-----|------:|----:|----:|----:|---------:|---------:|-------:|---:|------:|")
    for r in pt:
        ac_str = f"{r['ac_bytes']}" if r["ac_bytes"] is not None else "—"
        delta_str = (
            f"{r['ac_minus_brotli_bytes']:+d}"
            if r["ac_minus_brotli_bytes"] is not None
            else "—"
        )
        marker = " *" if r["regresses_ac_vs_brotli"] else ""
        print(
            f"| {r['idx']} | `{r['name']}` | {r['n_symbols']} | "
            f"{r['H0_bits']:.3f} | {r['H1_bits']:.3f} | {r['H2_bits']:.3f} | "
            f"{r['shannon_floor_h0_bytes']} | {r['shannon_floor_h2_bytes']} | "
            f"{r['brotli_bytes']} | {ac_str} | {delta_str}{marker} |"
        )
    print()
    print(f"Total Shannon floor (H_0): {s['total_shannon_floor_h0_bytes']:,} bytes")
    print(f"Total Shannon floor (H_2): {s['total_shannon_floor_h2_bytes']:,} bytes")
    print(f"Total brotli:              {s['total_brotli_bytes']:,} bytes")
    print(
        f"Total AC (PR103 indices):  {s['total_ac_bytes_pr103_indices_only']:,} bytes"
    )
    print(
        f"AC regression total:       +{s['ac_regression_total_bytes']:,} bytes "
        f"across tensors {s['ac_regression_tensors']}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _utc_timestamp() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_isoformat() -> str:
    # "2026-05-07T17:34:36+00:00" → "2026-05-07T17:34:36Z"
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_output_dir() -> pathlib.Path:
    return _REPO_ROOT / "experiments" / "results" / f"lane_per_tensor_shannon_pr106_{_utc_timestamp()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Per-tensor Shannon-floor empirical analysis on a PR106-style "
            "HNeRV state_dict. CPU-only, no scorer load, no archive build."
        )
    )
    parser.add_argument(
        "--state-dict",
        type=pathlib.Path,
        default=_REPO_ROOT
        / "experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt",
        help="Path to the PR106 state_dict.pt (default: PR106 sensitivity map substrate)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Directory to write per_tensor_shannon.json (default: experiments/results/lane_per_tensor_shannon_pr106_<UTC>)",
    )
    parser.add_argument(
        "--brotli-quality",
        type=int,
        default=DEFAULT_BROTLI_QUALITY,
        help=f"Brotli quality for brotli measurements (default: {DEFAULT_BROTLI_QUALITY})",
    )
    parser.add_argument(
        "--n-quant",
        type=int,
        default=N_QUANT,
        help=f"Quantization clamp range used by _quantize_tensor (default: {N_QUANT})",
    )
    parser.add_argument(
        "--no-stdout-table",
        action="store_true",
        help="Suppress the Markdown table on stdout (JSON output is unaffected).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    state_dict_path: pathlib.Path = args.state_dict
    if not state_dict_path.exists():
        logger.error("state_dict not found: %s", state_dict_path)
        return 2

    output_dir: pathlib.Path = args.output_dir or _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load state_dict (CPU-only, weights_only=True per CLAUDE.md torch.load
    # security hardening).
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        logger.error(
            "Expected state_dict to be a dict, got %s", type(state_dict).__name__
        )
        return 2

    started_at = _utc_isoformat()

    result = analyze_state_dict(
        state_dict,
        brotli_quality=args.brotli_quality,
        n_quant=args.n_quant,
    )

    out_json: dict[str, Any] = {
        "started_at_utc": started_at,
        "substrate": str(state_dict_path.relative_to(_REPO_ROOT))
        if state_dict_path.is_absolute() and _REPO_ROOT in state_dict_path.parents
        else str(state_dict_path),
        "evidence_grade": "[empirical]",
        "score_claim": False,
        "brotli_quality": int(args.brotli_quality),
        "n_quant": int(args.n_quant),
        "n_tensors": len(FIXED_STATE_SCHEMA),
        "ac_tensor_indices": list(AC_TENSOR_INDICES),
        "per_tensor": result["per_tensor"],
        "summary": result["summary"],
        "analysis_md": build_analysis_markdown(result),
    }

    output_path = output_dir / "per_tensor_shannon.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    if not args.no_stdout_table:
        print_per_tensor_markdown(result)

    print()
    print(f"[empirical] wrote per-tensor Shannon analysis to {output_path}")
    print(f"[empirical] substrate: {out_json['substrate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
