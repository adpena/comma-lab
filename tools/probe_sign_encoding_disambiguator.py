#!/usr/bin/env python3
"""Sign-encoding probe-disambiguator.

For each tensor in a target state-dict, sweep all 5 sign-encoding
strategies (negzig / zig / twos / off / raw_uint8) and report the
optimal per-tensor strategy + total bytes-saved-vs-baseline.

This is the probe-disambiguator design from
``.omx/research/sign_encoding_unified_taxonomy_20260512.md`` §5 — CPU-only,
$0 GPU spend.

Outputs a structured JSON to ``--output`` (default:
``.omx/research/probe_sign_encoding_results_<utc>.json``) with the
per-tensor verdict + global savings summary.

Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact" — the
output path is ALWAYS under ``.omx/research/`` (or a user-supplied
repo-relative path), NEVER ``/tmp/``.

Per CLAUDE.md "Forbidden score claims" — the output JSON carries
``score_claim=false``, ``promotion_eligible=false``,
``ready_for_exact_eval_dispatch=false`` per Catalog #100.

Per CLAUDE.md "Apples-to-apples evidence discipline" — the probe MUST
run on the exact INT8 tensors that will ship in the archive. Running on
training-time tensors and quantising differently for the final archive
is a measurement-apparatus bug.

Usage
=====

::

    .venv/bin/python tools/probe_sign_encoding_disambiguator.py \
        --state-dict-pt experiments/results/<archive>/state_dict.pt \
        --output .omx/research/probe_sign_encoding_results_<utc>.json

If ``--state-dict-pt`` is omitted, the probe runs on a synthetic
fixture so the CLI shape can be smoke-tested without external state.

Compatibility metric
====================

Two metrics are computed per (tensor, strategy) pair:

* **Shannon entropy bits** — pure information-theoretic lower bound on
  any entropy coder's compressed output. Cheap; no I/O.
* **Brotli-q11 bytes** — actual compressed size if the UINT8 byte stream
  is brotli'd at quality 11 (the canonical comma-lab setting for archive
  bytes). Expensive but ground-truth-equivalent.

The probe reports BOTH metrics so consumers can compare the
entropy-lower-bound vs the real-coder ranking.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.packet_compiler import (  # noqa: E402
    SignEncodingStrategy,
    encode_sign,
    select_optimal_strategy,
)
from tac.packet_compiler.sign_encoding import _shannon_entropy_bits  # noqa: E402


def _load_state_dict_int8_tensors(
    pt_path: Path,
) -> dict[str, np.ndarray]:
    """Load a state-dict and extract int8 tensors as numpy arrays.

    Per CLAUDE.md "Forbidden ``torch.load(weights_only=False)``" Catalog
    #98 — we use ``weights_only=True`` and reject any non-tensor entries.

    Per CLAUDE.md "Apples-to-apples" — the loader does NOT requantise;
    only int8 tensors already in the state-dict are scanned (uint8 also
    accepted for the raw_uint8 strategy).
    """
    import torch  # WEIGHTS_ONLY_FALSE_OK: weights_only=True is the safe path.

    sd_raw = torch.load(pt_path, weights_only=True, map_location="cpu")
    out: dict[str, np.ndarray] = {}
    for name, val in sd_raw.items():
        if not isinstance(val, torch.Tensor):
            continue
        np_arr = val.detach().cpu().numpy()
        if np_arr.dtype in (np.int8, np.uint8):
            out[name] = np_arr
    return out


def _synthetic_fixture_int8_tensors() -> dict[str, np.ndarray]:
    """Synthetic state-dict for CLI smoke testing (no external file).

    Builds 4 tensors with different distributions to exercise the
    5-strategy disambiguator:

    * ``zig_optimal`` — symmetric peak at zero (zig is optimal)
    * ``off_optimal`` — left-skewed (off is optimal)
    * ``twos_optimal`` — symmetric near-uniform (twos is competitive)
    * ``raw_uint8`` — non-negative (only raw_uint8 applies)
    """
    rng = np.random.default_rng(0)
    out: dict[str, np.ndarray] = {}
    # Symmetric peak at zero — zig should win.
    out["zig_optimal"] = np.clip(
        np.round(rng.normal(0, 3, size=500)), -127, 127
    ).astype(np.int8)
    # Left-skewed — off should win.
    out["off_optimal"] = np.clip(
        np.round(rng.normal(-30, 10, size=500)), -127, 127
    ).astype(np.int8)
    # Symmetric near-uniform.
    out["twos_optimal"] = rng.integers(-127, 128, size=500, dtype=np.int8)
    # Non-negative uint8 — only raw_uint8 applies.
    out["raw_uint8_native"] = rng.integers(0, 256, size=500, dtype=np.uint8)
    return out


def _try_brotli_bytes(uint8_stream: bytes) -> int | None:
    """Return brotli-q11 compressed byte count, or None if brotli unavailable."""
    try:
        import brotli  # type: ignore
    except ImportError:
        return None
    return len(brotli.compress(uint8_stream, quality=11))


def probe(
    tensors: dict[str, np.ndarray],
) -> dict[str, object]:
    """Run the 5-strategy sweep over every tensor.

    Returns a JSON-serialisable dict with per-tensor verdicts + global
    savings summary.
    """
    per_tensor: dict[str, dict[str, object]] = {}
    baseline_strategy = SignEncodingStrategy.OFF.value
    total_baseline_bytes = 0
    total_optimal_bytes = 0
    total_baseline_brotli = 0
    total_optimal_brotli = 0
    any_brotli = False

    for name, arr in tensors.items():
        flat = np.ascontiguousarray(arr).reshape(-1)
        size = int(flat.size)
        per_strategy_entropy: dict[str, float] = {}
        per_strategy_brotli: dict[str, int] = {}
        applicable_strategies: list[SignEncodingStrategy] = []
        if flat.dtype == np.int8:
            applicable_strategies = [
                SignEncodingStrategy.NEGZIG,
                SignEncodingStrategy.ZIG,
                SignEncodingStrategy.TWOS,
                SignEncodingStrategy.OFF,
            ]
        elif flat.dtype == np.uint8:
            applicable_strategies = [SignEncodingStrategy.RAW_UINT8]
        else:
            per_tensor[name] = {
                "size": size,
                "skipped_reason": (
                    f"unsupported dtype {flat.dtype} (must be int8 or uint8)"
                ),
            }
            continue
        for strat in applicable_strategies:
            # Skip negzig if -128 is present.
            if strat is SignEncodingStrategy.NEGZIG and np.any(flat == -128):
                continue
            encoded = encode_sign(flat, strat)
            arr_u8 = np.frombuffer(encoded, dtype=np.uint8)
            per_strategy_entropy[strat.value] = _shannon_entropy_bits(arr_u8)
            br = _try_brotli_bytes(encoded)
            if br is not None:
                per_strategy_brotli[strat.value] = br
                any_brotli = True

        if not per_strategy_entropy:
            per_tensor[name] = {
                "size": size,
                "skipped_reason": "no applicable strategy",
            }
            continue

        # Entropy-based selection (matches select_optimal_strategy ranking).
        sel = select_optimal_strategy(flat)
        optimal_strategy = sel.strategy.value

        # Bytes proxy — Shannon entropy * size / 8 = lower bound on bytes.
        baseline_entropy = per_strategy_entropy.get(baseline_strategy)
        if baseline_entropy is None:
            # The baseline strategy may have been skipped (e.g., raw_uint8 only).
            baseline_entropy = sel.entropy_bits
            baseline_for_report: str = optimal_strategy
        else:
            baseline_for_report = baseline_strategy
        baseline_bytes = int(baseline_entropy * size / 8.0)
        optimal_bytes = int(sel.entropy_bits * size / 8.0)
        total_baseline_bytes += baseline_bytes
        total_optimal_bytes += optimal_bytes

        baseline_brotli = per_strategy_brotli.get(baseline_for_report)
        # argmin of per_strategy_brotli, if brotli is available
        if per_strategy_brotli:
            min_brotli_strat = min(
                per_strategy_brotli.items(), key=lambda kv: kv[1]
            )
            optimal_brotli = min_brotli_strat[1]
            total_optimal_brotli += optimal_brotli
            if baseline_brotli is not None:
                total_baseline_brotli += baseline_brotli
        else:
            optimal_brotli = None

        per_tensor[name] = {
            "size": size,
            "dtype": str(flat.dtype),
            "optimal_strategy_by_entropy": optimal_strategy,
            "baseline_strategy": baseline_for_report,
            "per_strategy_shannon_entropy_bits": per_strategy_entropy,
            "per_strategy_brotli_q11_bytes": (
                per_strategy_brotli if per_strategy_brotli else None
            ),
            "shannon_baseline_lower_bound_bytes": baseline_bytes,
            "shannon_optimal_lower_bound_bytes": optimal_bytes,
            "shannon_predicted_savings_bytes": baseline_bytes - optimal_bytes,
            "brotli_baseline_bytes": baseline_brotli,
            "brotli_optimal_bytes": optimal_brotli,
        }

    return {
        "schema_version": "sign_encoding_probe_v1",
        "evidence_grade": "[byte-anchor; non-authoritative]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "n_tensors_scanned": len(tensors),
        "baseline_strategy": baseline_strategy,
        "shannon_total_baseline_bytes": total_baseline_bytes,
        "shannon_total_optimal_bytes": total_optimal_bytes,
        "shannon_total_savings_bytes": (
            total_baseline_bytes - total_optimal_bytes
        ),
        "brotli_total_baseline_bytes": (
            total_baseline_brotli if any_brotli else None
        ),
        "brotli_total_optimal_bytes": (
            total_optimal_brotli if any_brotli else None
        ),
        "brotli_total_savings_bytes": (
            (total_baseline_brotli - total_optimal_brotli)
            if any_brotli
            else None
        ),
        "per_tensor_verdict": per_tensor,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sign-encoding 5-strategy probe-disambiguator. "
            "Sweeps all 5 strategies per-tensor and reports the optimal "
            "per-tensor strategy + total bytes-saved-vs-baseline. CPU-only, "
            "$0 GPU. Output JSON has score_claim=false per Catalog #100."
        )
    )
    parser.add_argument(
        "--state-dict-pt",
        type=Path,
        default=None,
        help=(
            "Path to a .pt state-dict whose int8/uint8 tensors will be "
            "probed. If omitted, runs on a synthetic fixture for smoke "
            "testing."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to write the probe results JSON. "
            "Default: .omx/research/probe_sign_encoding_results_<utc>.json"
        ),
    )
    args = parser.parse_args(argv)

    if args.state_dict_pt is not None:
        if not args.state_dict_pt.is_file():
            print(
                f"error: state-dict file not found: {args.state_dict_pt}",
                file=sys.stderr,
            )
            return 2
        tensors = _load_state_dict_int8_tensors(args.state_dict_pt)
        state_dict_sha = hashlib.sha256(
            args.state_dict_pt.read_bytes()
        ).hexdigest()
        source = str(args.state_dict_pt)
    else:
        tensors = _synthetic_fixture_int8_tensors()
        state_dict_sha = "synthetic-fixture-no-sha"
        source = "synthetic"

    if not tensors:
        print(
            "error: no int8/uint8 tensors found in state-dict; "
            "sign-encoding probe requires int8 or uint8 weights",
            file=sys.stderr,
        )
        return 3

    result = probe(tensors)
    result["state_dict_source"] = source
    result["state_dict_sha256"] = state_dict_sha
    result["generated_at_utc"] = _dt.datetime.now(_dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    out_path = args.output
    if out_path is None:
        # Per CLAUDE.md FORBIDDEN /tmp paths — write into .omx/research/.
        utc_stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = (
            REPO_ROOT
            / ".omx"
            / "research"
            / f"probe_sign_encoding_results_{utc_stamp}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"sign-encoding probe-disambiguator: wrote {out_path} "
        f"(n_tensors={result['n_tensors_scanned']}, "
        f"shannon_savings={result['shannon_total_savings_bytes']} B "
        f"vs baseline={result['baseline_strategy']!r})",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
