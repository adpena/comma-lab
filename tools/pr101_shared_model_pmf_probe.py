#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Deterministic shared-model PMF probe for PR101 decoder weights.

This is a CPU-only research artifact builder. It fits a compact shared PMF
family over quantized PR101 decoder tensors, serializes the model, encodes the
symbols with the repo's deterministic range coder, and verifies exact
decode/reconstruction. It remains fail-closed: no PR101 archive substitution,
inflate runtime, scorer load, CUDA eval, score claim, promotion, ranking,
family falsification, or method kill happens here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)
from tac.shared_pmf_model import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_SEED,
    DEFAULT_TOTAL_FREQUENCY,
    SharedPMFConfig,
    TensorSymbolStream,
    build_shared_pmf_probe_result,
    compress_model,
    fit_shared_pmf_model,
    serialize_model,
)

TOOL_NAME = "tools/pr101_shared_model_pmf_probe.py"
SCHEMA_VERSION = "pr101_shared_model_pmf_probe.v1"
EVIDENCE_GRADE = "empirical"
EVIDENCE_MARKER = "[CPU-prep empirical]"
EVIDENCE_SEMANTICS = "cpu_shared_model_pmf_exact_roundtrip_research_artifact"

N_CATEGORIES = 255
ARCHIVE_OVERHEAD_BYTES = 16_094
REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES = 178_144
REFERENCE_PER_TENSOR_AAC_ARCHIVE_BYTES = 178_181
REFERENCE_IID_PER_TENSOR_FLOOR_ARCHIVE_BYTES = 175_916
REFERENCE_SHARED_POOLED_RANGE_ARCHIVE_BYTES = 203_196

DEFAULT_K_VALUES = (1, 2, 3, 4, 5, 6, 8, 12)
AUTODISCOVER_STATE_DICT_GLOBS = (
    "experiments/results/optuna_pr101_real_substrate_*/pr101_decoder_state_dict.pt",
    "experiments/results/cma_pr101_real_substrate_cmaes_*/pr101_decoder_state_dict.pt",
    "experiments/results/cma_pr101_real_substrate_*/pr101_decoder_state_dict.pt",
    "experiments/results/*pr101*/*decoder_state_dict*.pt",
)


def discover_default_state_dict(repo_root: Path = REPO_ROOT) -> Path | None:
    matches: list[Path] = []
    for pattern in AUTODISCOVER_STATE_DICT_GLOBS:
        matches.extend(repo_root.glob(pattern))
    files = sorted({path for path in matches if path.is_file()}, key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_pr101_symbol_rows(state_dict_path: Path) -> tuple[str, list[TensorSymbolStream], list[dict[str, Any]]]:
    input_sha256 = _sha256_file(state_dict_path)
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise SystemExit(f"loaded {state_dict_path} is not a dict")

    rows: list[TensorSymbolStream] = []
    summaries: list[dict[str, Any]] = []
    for idx, (name, expected_shape) in enumerate(FIXED_STATE_SCHEMA):
        if name not in state_dict:
            raise SystemExit(f"state_dict missing tensor {name!r}")
        qt = _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        shape = tuple(int(v) for v in getattr(qt, "shape", expected_shape))
        if shape != tuple(expected_shape):
            raise SystemExit(f"shape mismatch for {name!r}: schema {expected_shape}, quantized {shape}")
        q_i8 = np.asarray(qt.q_i8, dtype=np.int8)
        symbols = q_i8.astype(np.int64).reshape(-1) + 127
        if symbols.size and (int(symbols.min()) < 0 or int(symbols.max()) >= N_CATEGORIES):
            raise SystemExit(f"quantized symbols outside [0, {N_CATEGORIES - 1}] for {name!r}")
        counts = np.bincount(symbols, minlength=N_CATEGORIES)
        rows.append(TensorSymbolStream(name=name, symbols=symbols, shape=shape))
        summaries.append(
            {
                "idx": idx,
                "name": name,
                "shape": [int(v) for v in shape],
                "n_symbols": int(symbols.size),
                "n_unique_symbols": int(np.count_nonzero(counts)),
            }
        )
    return input_sha256, rows, summaries


def _parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit("expected at least one integer")
    return values


def _candidate_row(
    *,
    rows: list[TensorSymbolStream],
    n_models: int,
    seed: int,
    alpha: float,
    total_frequency: int,
    archive_overhead_bytes: int,
) -> dict[str, Any]:
    config = SharedPMFConfig(
        n_models=n_models,
        n_categories=N_CATEGORIES,
        total_frequency=total_frequency,
        alpha=alpha,
        seed=seed,
    )
    model = fit_shared_pmf_model(rows, config)
    raw_model = serialize_model(model)
    brotli_model = compress_model(model)
    payload_estimate_bytes = math.ceil(model.estimated_payload_bits / 8.0)
    archive_estimate = payload_estimate_bytes + len(brotli_model) + archive_overhead_bytes
    return {
        "n_models": n_models,
        "cluster_sizes": list(model.cluster_sizes),
        "init_strategy": model.init_strategy,
        "iterations": model.iterations,
        "payload_byte_source": "ceil_cross_entropy_bits_against_serialized_frequency_tables",
        "payload_estimate_bits": model.estimated_payload_bits,
        "payload_estimate_bytes": payload_estimate_bytes,
        "model_bytes": len(brotli_model),
        "model_raw_bytes": len(raw_model),
        "raw_frequency_table_bytes": model.table_bytes_raw,
        "raw_assignment_bytes": model.assignment_bytes_raw,
        "raw_tensor_length_bytes": model.tensor_length_bytes_raw,
        "model_raw_sha256": hashlib.sha256(raw_model).hexdigest(),
        "model_brotli_sha256": hashlib.sha256(brotli_model).hexdigest(),
        "archive_estimate_bytes": archive_estimate,
        "delta_vs_brotli_optuna_archive_bytes": archive_estimate - REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES,
        "delta_vs_per_tensor_aac_archive_bytes": archive_estimate - REFERENCE_PER_TENSOR_AAC_ARCHIVE_BYTES,
        "delta_vs_iid_per_tensor_floor_archive_bytes": archive_estimate - REFERENCE_IID_PER_TENSOR_FLOOR_ARCHIVE_BYTES,
    }


def _verdict(archive_estimate_bytes: int) -> str:
    loses_brotli = archive_estimate_bytes >= REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES
    loses_aac = archive_estimate_bytes >= REFERENCE_PER_TENSOR_AAC_ARCHIVE_BYTES
    if loses_brotli and loses_aac:
        return "negative_loses_to_brotli_and_per_tensor_aac_after_model_and_payload_bytes"
    if loses_brotli:
        return "negative_loses_to_brotli_after_model_and_payload_bytes"
    if loses_aac:
        return "mixed_beats_brotli_but_loses_to_per_tensor_aac_not_dispatchable"
    return "positive_research_artifact_beats_brotli_and_aac_estimates_not_dispatchable"


def build_probe_manifest(
    state_dict_path: Path,
    *,
    k_values: list[int] | None = None,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    total_frequency: int = DEFAULT_TOTAL_FREQUENCY,
    archive_overhead_bytes: int = ARCHIVE_OVERHEAD_BYTES,
    require_roundtrip: bool = True,
) -> dict[str, Any]:
    input_sha256, rows, per_tensor_summary = _load_pr101_symbol_rows(state_dict_path)
    requested_k = k_values if k_values is not None else list(DEFAULT_K_VALUES)
    valid_k = [int(k) for k in requested_k if 1 <= int(k) <= len(rows)]
    if not valid_k:
        raise SystemExit(f"no valid shared model counts for n_tensors={len(rows)}")

    candidates = [
        _candidate_row(
            rows=rows,
            n_models=k,
            seed=seed,
            alpha=alpha,
            total_frequency=total_frequency,
            archive_overhead_bytes=archive_overhead_bytes,
        )
        for k in valid_k
    ]
    best_candidate = min(candidates, key=lambda row: row["archive_estimate_bytes"])
    result = build_shared_pmf_probe_result(
        rows,
        SharedPMFConfig(
            n_models=int(best_candidate["n_models"]),
            n_categories=N_CATEGORIES,
            total_frequency=total_frequency,
            alpha=alpha,
            seed=seed,
        ),
        archive_overhead_bytes=archive_overhead_bytes,
        require_roundtrip=require_roundtrip,
    )
    exact = result.to_manifest_dict()
    exact["delta_vs_brotli_optuna_archive_bytes"] = (
        result.archive_estimate_bytes - REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES
    )
    exact["delta_vs_per_tensor_aac_archive_bytes"] = (
        result.archive_estimate_bytes - REFERENCE_PER_TENSOR_AAC_ARCHIVE_BYTES
    )
    exact["delta_vs_iid_per_tensor_floor_archive_bytes"] = (
        result.archive_estimate_bytes - REFERENCE_IID_PER_TENSOR_FLOOR_ARCHIVE_BYTES
    )
    exact["delta_vs_shared_pooled_range_archive_bytes"] = (
        result.archive_estimate_bytes - REFERENCE_SHARED_POOLED_RANGE_ARCHIVE_BYTES
    )
    exact["verdict"] = _verdict(result.archive_estimate_bytes)

    for row in candidates:
        row["verdict"] = _verdict(int(row["archive_estimate_bytes"]))
        if row["n_models"] == exact["n_models"]:
            row["exact_roundtrip_payload_bytes"] = exact["encoded_payload_bytes"]
            row["exact_roundtrip_archive_estimate_bytes"] = exact["archive_estimate_bytes"]

    dispatch_blockers = [
        "cpu_research_artifact_only",
        "no_pr101_archive_substitution_performed",
        "no_inflate_runtime_decoder_wired",
        "no_runtime_tree_manifest_or_submission_packet",
        "missing_exact_cuda_auth_eval",
        "score_claim_false",
        "promotion_requires_charged_archive_bytes_and_cuda_auth_eval",
    ]
    if not exact["roundtrip"]["exact_reconstruction_ok"]:
        dispatch_blockers.insert(0, "exact_reconstruction_roundtrip_failed")

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "input_state_dict": _repo_relative(state_dict_path),
        "input_state_dict_sha256": input_sha256,
        "source_state_dict_sha256": input_sha256,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_marker": EVIDENCE_MARKER,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "dispatch_blockers": dispatch_blockers,
        "disposition": exact["verdict"],
        "cpu_only": True,
        "deterministic_seed": seed,
        "alpha": alpha,
        "total_frequency": total_frequency,
        "n_categories": N_CATEGORIES,
        "n_tensors": len(rows),
        "n_symbols": int(sum(row.n_symbols for row in rows)),
        "archive_overhead_bytes": archive_overhead_bytes,
        "comparison_brotli_optuna_archive_bytes": REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES,
        "comparison_per_tensor_aac_archive_bytes": REFERENCE_PER_TENSOR_AAC_ARCHIVE_BYTES,
        "comparison_iid_per_tensor_floor_archive_bytes": REFERENCE_IID_PER_TENSOR_FLOOR_ARCHIVE_BYTES,
        "comparison_shared_pooled_range_archive_bytes": REFERENCE_SHARED_POOLED_RANGE_ARCHIVE_BYTES,
        "best_model_by_archive_estimate": exact,
        "candidate_models": candidates,
        "artifact_disposition_detail": {
            "beats_brotli_optuna": result.archive_estimate_bytes < REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES,
            "beats_per_tensor_aac": result.archive_estimate_bytes < REFERENCE_PER_TENSOR_AAC_ARCHIVE_BYTES,
            "beats_iid_per_tensor_floor": result.archive_estimate_bytes < REFERENCE_IID_PER_TENSOR_FLOOR_ARCHIVE_BYTES,
            "delta_vs_brotli_optuna_archive_bytes": exact["delta_vs_brotli_optuna_archive_bytes"],
            "delta_vs_per_tensor_aac_archive_bytes": exact["delta_vs_per_tensor_aac_archive_bytes"],
            "delta_vs_iid_per_tensor_floor_archive_bytes": exact["delta_vs_iid_per_tensor_floor_archive_bytes"],
            "negative_result_policy": (
                "If any delta is non-negative, record it as a charged-byte loss and do not promote, rank, "
                "kill, or dispatch from this CPU artifact."
            ),
        },
        "per_tensor_input_summary": per_tensor_summary,
        "model_storage_note": "model_bytes is brotli(serialized PMF tables + assignments + tensor lengths)",
        "payload_storage_note": "encoded_payload_bytes is an exact deterministic range-coded payload with header",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict-path", "--state-dict", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--k-values",
        type=str,
        default=",".join(str(k) for k in DEFAULT_K_VALUES),
        help="Comma-separated shared model counts to sweep before exact roundtrip of the best.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--total-frequency", type=int, default=DEFAULT_TOTAL_FREQUENCY)
    parser.add_argument("--archive-overhead-bytes", type=int, default=ARCHIVE_OVERHEAD_BYTES)
    parser.add_argument(
        "--allow-roundtrip-failure",
        action="store_true",
        help="Record failed roundtrip instead of raising. Intended only for debugging failing cases.",
    )
    args = parser.parse_args(argv)

    state_dict_path = args.state_dict_path
    autodiscovered = False
    if state_dict_path is None:
        state_dict_path = discover_default_state_dict()
        autodiscovered = True
    if state_dict_path is None or not state_dict_path.is_file():
        raise SystemExit(
            "state_dict not found; pass --state-dict-path or place pr101_decoder_state_dict.pt "
            "under an existing PR101 CMA/Optuna result directory"
        )

    manifest = build_probe_manifest(
        state_dict_path,
        k_values=_parse_int_list(args.k_values),
        seed=args.seed,
        alpha=args.alpha,
        total_frequency=args.total_frequency,
        archive_overhead_bytes=args.archive_overhead_bytes,
        require_roundtrip=not args.allow_roundtrip_failure,
    )
    manifest["autodiscovered_state_dict"] = autodiscovered
    manifest["tool_dependencies"] = {
        "brotli_available": True,
        "brotli_module": getattr(brotli, "__name__", "brotli"),
        "range_coder": "tac.lossless.range_coder.RangeEncoder/RangeDecoder",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    best = manifest["best_model_by_archive_estimate"]
    print(f"Wrote {args.output}")
    print(f"State dict:              {state_dict_path}")
    print(f"Source SHA-256:          {manifest['source_state_dict_sha256']}")
    print(f"Seed:                    {manifest['deterministic_seed']}")
    print(f"Tensors / symbols:       {manifest['n_tensors']} / {manifest['n_symbols']:,}")
    print(f"Best K:                  {best['n_models']}")
    print(f"Payload bytes:           {best['encoded_payload_bytes']:>10,}")
    print(f"Model bytes:             {best['model_bytes']:>10,}")
    print(f"Archive estimate:        {best['archive_estimate_bytes']:>10,}")
    print(f"Delta vs Brotli:         {best['delta_vs_brotli_optuna_archive_bytes']:>+10,}")
    print(f"Delta vs per-tensor AAC: {best['delta_vs_per_tensor_aac_archive_bytes']:>+10,}")
    print(f"Exact roundtrip:         {best['roundtrip']['exact_reconstruction_ok']}")
    print(f"Disposition:             {manifest['disposition']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
