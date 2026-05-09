#!/usr/bin/env python3
"""Real-PR101 ChARM/range-coder probe over quantized HNeRV decoder weights.

This is a CPU-only byte probe, not a contest score claim. It answers the
Phase-A4 question that the toy ChARM substrate cannot answer by applying the
actual ChARM range-coder backend to the real PR101 quantized tensor stream.

The probe tests three deterministic, decoder-reproducible PMF families:

* ``tensor_gaussian``: one Gaussian PMF per tensor.
* ``previous_symbol_gaussian``: autoregressive PMF centered on the previously
  decoded symbol, with one per-tensor sigma.
* ``delta_zero_gaussian``: encode first differences under one zero-mean
  Gaussian PMF per tensor.

All rows round-trip the quantized symbols exactly. The manifest remains
fail-closed because no runtime decoder, archive substitution, or exact CUDA
auth eval is produced here.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.codec.charm_range_coder import (  # noqa: E402
    ChARMBitStreamHeader,
    HEADER_SIZE,
    gaussian_pmf_int8,
)
from tac.lossless.range_coder import (  # noqa: E402
    RangeDecoder,
    RangeEncoder,
    cumulative_frequencies,
    normalize_probabilities,
)
from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    LATENT_BLOB_LEN,
    N_QUANT,
    _quantize_tensor,
    encode_decoder_compact,
)

TOOL_NAME = "tools/pr101_charm_real_substrate_probe.py"
SCHEMA_VERSION = "pr101_charm_real_substrate_probe.v1"
EVIDENCE_GRADE = "empirical_planning"
EVIDENCE_SEMANTICS = "cpu_real_pr101_charm_range_coder_probe"
REFERENCE_BROTLI_OPTUNA_BYTES = 178_144
REFERENCE_PR101_DECODER_BLOB_BYTES = 162_164
DEFAULT_CHUNK_SYMBOLS = 16_384
ARCHIVE_OVERHEAD_BYTES = LATENT_BLOB_LEN + 607 + 100


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise SystemExit(f"loaded {path} is not a state_dict dict")
    return state_dict


def _quantized_symbol_rows(state_dict: dict[str, torch.Tensor]) -> list[tuple[str, np.ndarray]]:
    rows: list[tuple[str, np.ndarray]] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        if name not in state_dict:
            raise SystemExit(f"state_dict missing tensor {name!r}")
        qt = _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        rows.append((name, qt.q_i8.astype(np.int16).reshape(-1)))
    return rows


def _fp16_sideinfo_bytes(*values: float) -> int:
    # Actual packet would store fp16 params. For planning, charge the exact
    # number of bytes those fp16 scalars consume and clamp at finite values.
    arr = np.asarray(values, dtype=np.float16)
    if not np.all(np.isfinite(arr.astype(np.float32))):
        raise ValueError("side-info contains non-finite fp16 value")
    return int(arr.nbytes)


def _sigma(values: np.ndarray, *, floor: float = 0.5) -> float:
    if values.size == 0:
        return floor
    s = float(np.std(values.astype(np.float64)))
    if not math.isfinite(s):
        return floor
    return max(s, floor)


def _cum_cache_entry(pmf: np.ndarray, total_bits: int) -> tuple[list[int], int]:
    freqs = normalize_probabilities(np.asarray(pmf, dtype=np.float64).tolist(), total=1 << total_bits)
    return cumulative_frequencies(freqs)


def _encode_chunk_cached(
    symbols: Iterable[int],
    *,
    alphabet: tuple[int, int],
    pmf_total_bits: int,
    pmf_for_symbol: Callable[[int, int], np.ndarray],
    pmf_key_for_symbol: Callable[[int, int], object],
) -> bytes:
    lo, hi = alphabet
    enc = RangeEncoder()
    n = 0
    cache: dict[object, tuple[list[int], int]] = {}
    for idx, sym in enumerate(symbols):
        sym_int = int(sym)
        if not (lo <= sym_int <= hi):
            raise ValueError(f"symbol {sym_int} outside alphabet [{lo}, {hi}]")
        key = pmf_key_for_symbol(idx, sym_int)
        if key not in cache:
            cache[key] = _cum_cache_entry(pmf_for_symbol(idx, sym_int), pmf_total_bits)
        cumulative, total = cache[key]
        enc.encode(symbol=sym_int - lo, cumulative=cumulative, total=total)
        n += 1
    payload = b"" if n == 0 else enc.finish()
    if len(payload) > (1 << 16) - 1:
        raise ValueError(
            f"payload {len(payload)} exceeds ChARM uint16 payload field; "
            "lower --chunk-symbols"
        )
    header = ChARMBitStreamHeader(
        alphabet_lo=lo,
        alphabet_hi=hi,
        num_symbols=n,
        pmf_total_bits=pmf_total_bits,
        payload_len=len(payload),
    ).to_bytes()
    return header + payload


def _decode_chunk_cached(
    blob: bytes,
    *,
    pmf_for_index: Callable[[int], np.ndarray],
    pmf_key_for_index: Callable[[int], object],
) -> list[int]:
    header = ChARMBitStreamHeader.from_bytes(blob)
    expected_len = HEADER_SIZE + header.payload_len
    if len(blob) != expected_len:
        raise ValueError(
            f"bad chunk length: header expects {expected_len} bytes, got {len(blob)}"
        )
    payload = blob[HEADER_SIZE:expected_len]
    if header.num_symbols == 0:
        return []
    dec = RangeDecoder(payload)
    cache: dict[object, tuple[list[int], int]] = {}
    out: list[int] = []
    for idx in range(header.num_symbols):
        key = pmf_key_for_index(idx)
        if key not in cache:
            cache[key] = _cum_cache_entry(pmf_for_index(idx), header.pmf_total_bits)
        cumulative, total = cache[key]
        target = dec.target(total)
        symbol_index = bisect.bisect_right(cumulative, target) - 1
        if symbol_index < 0 or symbol_index >= header.alphabet_size:
            raise ValueError("decoded target outside frequency table")
        dec.update(
            low_count=cumulative[symbol_index],
            high_count=cumulative[symbol_index + 1],
            total=total,
        )
        out.append(symbol_index + header.alphabet_lo)
    return out


def _chunks(values: np.ndarray, chunk_symbols: int) -> Iterable[np.ndarray]:
    if chunk_symbols <= 0:
        raise ValueError("chunk_symbols must be positive")
    for start in range(0, int(values.size), chunk_symbols):
        yield values[start:start + chunk_symbols]


def _encode_tensor_gaussian(
    symbols: np.ndarray,
    *,
    chunk_symbols: int,
    pmf_total_bits: int,
) -> tuple[int, bool, dict[str, Any]]:
    mu = float(np.mean(symbols.astype(np.float64))) if symbols.size else 0.0
    sigma = _sigma(symbols)
    pmf = gaussian_pmf_int8(mu, sigma, alphabet_lo=-127, alphabet_hi=127)
    chunks: list[bytes] = []
    decoded: list[int] = []
    for chunk in _chunks(symbols, chunk_symbols):
        blob = _encode_chunk_cached(
            chunk,
            alphabet=(-127, 127),
            pmf_total_bits=pmf_total_bits,
            pmf_for_symbol=lambda _idx, _sym: pmf,
            pmf_key_for_symbol=lambda _idx, _sym: "tensor_gaussian",
        )
        chunks.append(blob)
        decoded.extend(
            _decode_chunk_cached(
                blob,
                pmf_for_index=lambda _idx: pmf,
                pmf_key_for_index=lambda _idx: "tensor_gaussian",
            )
        )
    sideinfo = 2 + _fp16_sideinfo_bytes(mu, sigma)
    return sum(len(c) for c in chunks) + sideinfo, decoded == symbols.astype(int).tolist(), {
        "mu": mu,
        "sigma": sigma,
        "side_info_bytes": sideinfo,
        "n_chunks": len(chunks),
        "coded_stream_bytes": sum(len(c) for c in chunks),
    }


def _encode_previous_symbol_gaussian(
    symbols: np.ndarray,
    *,
    chunk_symbols: int,
    pmf_total_bits: int,
) -> tuple[int, bool, dict[str, Any]]:
    if symbols.size <= 1:
        delta = np.array([0], dtype=np.int16)
    else:
        delta = np.diff(symbols.astype(np.int16))
    sigma = _sigma(delta)
    pmfs = {
        mu: gaussian_pmf_int8(float(mu), sigma, alphabet_lo=-127, alphabet_hi=127)
        for mu in range(-127, 128)
    }
    stream_bytes = 0
    decoded_all: list[int] = []
    prev_for_encode = 0
    for chunk in _chunks(symbols, chunk_symbols):
        prev_at_chunk_start = prev_for_encode

        def pmf_for_symbol(_idx: int, _sym: int, *, start_prev: int = prev_at_chunk_start) -> np.ndarray:
            del _sym
            if _idx == 0:
                return pmfs[int(start_prev)]
            # The encode-side key is supplied by pmf_key_for_symbol, so this
            # fallback is unused for cached hits after the first lookup.
            raise RuntimeError("previous-symbol PMF lookup requires explicit key")

        # Cache keys are absolute previous-symbol values. We cannot derive
        # them from idx alone during encode, so build a key list for the chunk.
        prev_keys: list[int] = []
        prev = prev_at_chunk_start
        for sym in chunk:
            prev_keys.append(int(prev))
            prev = int(sym)
        prev_for_encode = prev

        blob = _encode_chunk_cached(
            chunk,
            alphabet=(-127, 127),
            pmf_total_bits=pmf_total_bits,
            pmf_for_symbol=lambda idx, _sym, keys=prev_keys: pmfs[keys[idx]],
            pmf_key_for_symbol=lambda idx, _sym, keys=prev_keys: keys[idx],
        )
        stream_bytes += len(blob)

        header = ChARMBitStreamHeader.from_bytes(blob)
        payload = blob[HEADER_SIZE:HEADER_SIZE + header.payload_len]
        dec = RangeDecoder(payload)
        cache: dict[int, tuple[list[int], int]] = {}
        decoded_chunk: list[int] = []
        prev_for_decode = prev_at_chunk_start
        for _idx in range(header.num_symbols):
            key = int(prev_for_decode)
            if key not in cache:
                cache[key] = _cum_cache_entry(pmfs[key], header.pmf_total_bits)
            cumulative, total = cache[key]
            target = dec.target(total)
            symbol_index = bisect.bisect_right(cumulative, target) - 1
            if symbol_index < 0 or symbol_index >= header.alphabet_size:
                raise ValueError("decoded target outside previous-symbol PMF table")
            dec.update(
                low_count=cumulative[symbol_index],
                high_count=cumulative[symbol_index + 1],
                total=total,
            )
            value = symbol_index + header.alphabet_lo
            decoded_chunk.append(value)
            prev_for_decode = value
        decoded_all.extend(decoded_chunk)

    sideinfo = 2 + _fp16_sideinfo_bytes(sigma)
    return stream_bytes + sideinfo, decoded_all == symbols.astype(int).tolist(), {
        "sigma_delta": sigma,
        "side_info_bytes": sideinfo,
        "n_chunks": math.ceil(symbols.size / chunk_symbols) if symbols.size else 0,
        "coded_stream_bytes": stream_bytes,
    }


def _encode_delta_zero_gaussian(
    symbols: np.ndarray,
    *,
    chunk_symbols: int,
    pmf_total_bits: int,
) -> tuple[int, bool, dict[str, Any]]:
    if symbols.size == 0:
        deltas = np.array([], dtype=np.int16)
    else:
        prev = np.concatenate([np.array([0], dtype=np.int16), symbols[:-1].astype(np.int16)])
        deltas = symbols.astype(np.int16) - prev
    sigma = _sigma(deltas)
    pmf = gaussian_pmf_int8(0.0, sigma, alphabet_lo=-254, alphabet_hi=254)
    chunks: list[bytes] = []
    decoded_deltas: list[int] = []
    for chunk in _chunks(deltas, chunk_symbols):
        blob = _encode_chunk_cached(
            chunk,
            alphabet=(-254, 254),
            pmf_total_bits=pmf_total_bits,
            pmf_for_symbol=lambda _idx, _sym: pmf,
            pmf_key_for_symbol=lambda _idx, _sym: "delta_zero",
        )
        chunks.append(blob)
        decoded_deltas.extend(
            _decode_chunk_cached(
                blob,
                pmf_for_index=lambda _idx: pmf,
                pmf_key_for_index=lambda _idx: "delta_zero",
            )
        )
    reconstructed: list[int] = []
    prev_value = 0
    for delta in decoded_deltas:
        value = prev_value + int(delta)
        reconstructed.append(value)
        prev_value = value
    sideinfo = 2 + _fp16_sideinfo_bytes(sigma)
    return sum(len(c) for c in chunks) + sideinfo, reconstructed == symbols.astype(int).tolist(), {
        "sigma_delta": sigma,
        "side_info_bytes": sideinfo,
        "n_chunks": len(chunks),
        "coded_stream_bytes": sum(len(c) for c in chunks),
    }


def build_charm_real_substrate_report(
    state_dict_path: Path,
    *,
    chunk_symbols: int = DEFAULT_CHUNK_SYMBOLS,
    pmf_total_bits: int = 15,
) -> dict[str, Any]:
    t0 = time.time()
    state_dict = _load_state_dict(state_dict_path)
    quantized_rows = _quantized_symbol_rows(state_dict)
    try:
        split_brotli_decoder_blob_bytes = len(encode_decoder_compact(state_dict))
    except Exception as exc:  # pragma: no cover - defensive manifest field
        split_brotli_decoder_blob_bytes = None
        split_brotli_error = repr(exc)
    else:
        split_brotli_error = None

    model_fns = {
        "tensor_gaussian": _encode_tensor_gaussian,
        "previous_symbol_gaussian": _encode_previous_symbol_gaussian,
        "delta_zero_gaussian": _encode_delta_zero_gaussian,
    }
    model_totals: dict[str, dict[str, Any]] = {
        name: {
            "model": name,
            "decoder_payload_bytes": 0,
            "archive_estimate_bytes": 0,
            "roundtrip_exact": True,
            "per_tensor": [],
        }
        for name in model_fns
    }

    for tensor_name, symbols in quantized_rows:
        for model_name, fn in model_fns.items():
            encoded_bytes, roundtrip_exact, meta = fn(
                symbols,
                chunk_symbols=chunk_symbols,
                pmf_total_bits=pmf_total_bits,
            )
            model_totals[model_name]["decoder_payload_bytes"] += encoded_bytes
            model_totals[model_name]["roundtrip_exact"] = (
                model_totals[model_name]["roundtrip_exact"] and roundtrip_exact
            )
            model_totals[model_name]["per_tensor"].append({
                "name": tensor_name,
                "n_symbols": int(symbols.size),
                "encoded_total_bytes": encoded_bytes,
                "roundtrip_exact": roundtrip_exact,
                **meta,
            })

    models: list[dict[str, Any]] = []
    for row in model_totals.values():
        decoder_payload = int(row["decoder_payload_bytes"])
        archive_estimate = decoder_payload + ARCHIVE_OVERHEAD_BYTES
        row["archive_estimate_bytes"] = archive_estimate
        row["delta_vs_brotli_optuna_bytes"] = archive_estimate - REFERENCE_BROTLI_OPTUNA_BYTES
        row["decoder_delta_vs_pr101_decoder_blob_bytes"] = (
            decoder_payload - REFERENCE_PR101_DECODER_BLOB_BYTES
        )
        if split_brotli_decoder_blob_bytes is not None:
            row["decoder_delta_vs_current_split_brotli_bytes"] = (
                decoder_payload - split_brotli_decoder_blob_bytes
            )
        models.append(row)
    models.sort(key=lambda r: int(r["archive_estimate_bytes"]))
    best = models[0]

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "input_state_dict": str(state_dict_path),
        "input_state_dict_sha256": _sha256_path(state_dict_path),
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            "cpu_planning_probe_only",
            "range_coder_not_wired_into_inflate_runtime",
            "no_archive_substitution_performed",
            "missing_exact_cuda_auth_eval",
            "reactivation_required_before_new_dispatch",
        ],
        "reactivation_criteria": [
            "Implement runtime decoder consuming this exact ChARM packet grammar.",
            "Produce archive.zip whose inflate path consumes the changed bytes.",
            "Prove byte closure with archive SHA-256, member SHA-256s, runtime tree SHA, and no-op control.",
            "Run exact contest-CUDA and contest-CPU auth eval before promotion.",
        ],
        "family_falsified": False,
        "method_family_retired": False,
        "falsification_scope": "measured_real_pr101_hand_parametric_charm_probe_only",
        "n_tensors": len(quantized_rows),
        "n_symbols_total": int(sum(row[1].size for row in quantized_rows)),
        "chunk_symbols": chunk_symbols,
        "pmf_total_bits": pmf_total_bits,
        "archive_overhead_bytes": ARCHIVE_OVERHEAD_BYTES,
        "reference_brotli_optuna_bytes": REFERENCE_BROTLI_OPTUNA_BYTES,
        "reference_pr101_decoder_blob_bytes": REFERENCE_PR101_DECODER_BLOB_BYTES,
        "current_split_brotli_decoder_blob_bytes": split_brotli_decoder_blob_bytes,
        "current_split_brotli_error": split_brotli_error,
        "best_model": {
            "model": best["model"],
            "archive_estimate_bytes": best["archive_estimate_bytes"],
            "delta_vs_brotli_optuna_bytes": best["delta_vs_brotli_optuna_bytes"],
            "roundtrip_exact": best["roundtrip_exact"],
        },
        "models": models,
        "elapsed_seconds": time.time() - t0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-symbols", type=int, default=DEFAULT_CHUNK_SYMBOLS)
    parser.add_argument("--pmf-total-bits", type=int, default=15)
    args = parser.parse_args(argv)
    if not args.state_dict_path.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict_path}")
    report = build_charm_real_substrate_report(
        args.state_dict_path,
        chunk_symbols=args.chunk_symbols,
        pmf_total_bits=args.pmf_total_bits,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.output}")
    for row in report["models"]:
        print(
            f"{row['model']}: archive_estimate={row['archive_estimate_bytes']:,} "
            f"delta_vs_brotli={row['delta_vs_brotli_optuna_bytes']:+,} "
            f"roundtrip={row['roundtrip_exact']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
