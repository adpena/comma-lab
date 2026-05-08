#!/usr/bin/env python3
"""Tiny neural PMF/context smoke probe for PR101 quantized decoder weights.

This asks a narrow compression question: can a very small deterministic neural
predictor replace transmitted per-tensor PMF tables or large context tables?

The probe fits low-rank softmax models over quantized PR101 symbols:

    P(symbol_n | tensor_id, fixed tensor features[, previous_symbol])

The learned model parameters are charged as side information. The report is
planning evidence only: it computes ideal code lengths against the fitted PMF,
but it does not emit a range/ANS bitstream, serialize a runtime decoder, or
substitute an archive.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

TOOL_NAME = "tools/pr101_tiny_nn_predict_pmf.py"
SCHEMA_VERSION = "pr101_tiny_nn_predict_pmf.v1"
EVIDENCE_GRADE = "empirical"
EVIDENCE_SEMANTICS = "cpu_tiny_nn_pmf_planning_probe"
N_CATEGORIES = 255
START_SYMBOL = N_CATEGORIES
ARCHIVE_OVERHEAD_BYTES = 16_094
TENSOR_SCALE_BYTES_PER_TENSOR = 2
RANGE_STREAM_HEADER_BYTES_ESTIMATE = 16

REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES = 178_144
REFERENCE_IID_PER_TENSOR_FLOOR_ARCHIVE_BYTES = 175_916
REFERENCE_PER_TENSOR_AAC_ARCHIVE_BYTES = 178_181
REFERENCE_NAIVE_MARKOV1_AAC_ARCHIVE_BYTES = 199_238


@dataclass(frozen=True)
class TinyNnConfig:
    rank: int = 8
    epochs: int = 30
    learning_rate: float = 0.05
    batch_size: int = 16_384
    weight_decay: float = 0.0
    seed: int = 17
    max_train_symbols: int = 0
    torch_threads: int = 1
    include_position_features: bool = False


@dataclass(frozen=True)
class SymbolCorpus:
    tensor_ids: torch.Tensor
    prev_symbols: torch.Tensor
    continuous_features: torch.Tensor
    targets: torch.Tensor
    per_tensor_rows: list[dict[str, Any]]
    input_state_dict_sha256: str
    n_total_symbols: int


class TinyFactorizedContextPMF(nn.Module):
    """Low-rank neural categorical predictor.

    The model is intentionally small: tensor and optional previous-symbol
    embeddings are projected through a shared symbol basis. This tests whether a
    compact hyperprior/context model can recover the Markov/Pmf signal without
    transmitting explicit 28 x 255 or 28 x 255 x 255 PMF tables.
    """

    def __init__(
        self,
        *,
        n_tensors: int,
        n_categories: int,
        rank: int,
        n_continuous_features: int,
        use_prev_symbol: bool,
    ) -> None:
        super().__init__()
        self.n_tensors = n_tensors
        self.n_categories = n_categories
        self.rank = rank
        self.n_continuous_features = n_continuous_features
        self.use_prev_symbol = use_prev_symbol

        self.tensor_embedding = nn.Embedding(n_tensors, rank)
        self.continuous_projection = nn.Linear(n_continuous_features, rank, bias=False)
        self.prev_embedding = (
            nn.Embedding(n_categories + 1, rank) if use_prev_symbol else None
        )
        self.symbol_basis = nn.Parameter(torch.empty(rank, n_categories))
        self.global_logits = nn.Parameter(torch.zeros(n_categories))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.tensor_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.continuous_projection.weight, mean=0.0, std=0.02)
        if self.prev_embedding is not None:
            nn.init.normal_(self.prev_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.symbol_basis, mean=0.0, std=0.02)
        nn.init.zeros_(self.global_logits)

    def forward(
        self,
        tensor_ids: torch.Tensor,
        prev_symbols: torch.Tensor,
        continuous_features: torch.Tensor,
    ) -> torch.Tensor:
        context = self.tensor_embedding(tensor_ids)
        context = context + self.continuous_projection(continuous_features)
        if self.prev_embedding is not None:
            context = context + self.prev_embedding(prev_symbols)
        return self.global_logits + context @ self.symbol_basis


def _set_deterministic(seed: int, torch_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _as_int_symbols(q_i8: np.ndarray) -> np.ndarray:
    symbols = q_i8.astype(np.int32).reshape(-1) + 127
    if symbols.size and (int(symbols.min()) < 0 or int(symbols.max()) >= N_CATEGORIES):
        raise SystemExit("quantized tensor emitted symbols outside [0, 254]")
    return symbols.astype(np.int64)


def _tensor_continuous_features(
    *,
    idx: int,
    n_tensors: int,
    shape: tuple[int, ...],
    n_elements: int,
    scale: float,
) -> np.ndarray:
    denom = max(n_tensors - 1, 1)
    rank_norm = len(shape) / 4.0
    log_elements = math.log2(max(n_elements, 1)) / 20.0
    safe_scale = max(abs(float(scale)), 1e-12)
    log_scale = max(min(math.log2(safe_scale) / 16.0, 4.0), -4.0)
    return np.array([idx / denom, rank_norm, log_elements, log_scale], dtype=np.float32)


def _position_continuous_features(n_elements: int) -> np.ndarray:
    if n_elements <= 0:
        return np.empty((0, 2), dtype=np.float32)
    positions = np.arange(n_elements, dtype=np.float32)
    denom = float(max(n_elements - 1, 1))
    position_fraction = positions / denom
    log_position_fraction = np.log1p(positions) / math.log1p(float(max(n_elements - 1, 1)))
    return np.stack([position_fraction, log_position_fraction.astype(np.float32)], axis=1)


def _load_symbol_corpus(
    state_dict_path: Path,
    *,
    include_position_features: bool,
) -> SymbolCorpus:
    input_bytes = state_dict_path.read_bytes()
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise SystemExit(f"loaded {state_dict_path} is not a dict")

    tensor_ids: list[np.ndarray] = []
    prev_symbols: list[np.ndarray] = []
    continuous_features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    per_tensor_rows: list[dict[str, Any]] = []

    n_tensors = len(FIXED_STATE_SCHEMA)
    for idx, (name, expected_shape) in enumerate(FIXED_STATE_SCHEMA):
        if name not in state_dict:
            raise SystemExit(f"state_dict missing tensor {name!r}")
        qt = _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        symbols = _as_int_symbols(qt.q_i8)
        if tuple(qt.shape) != tuple(expected_shape):
            raise SystemExit(
                f"shape mismatch for {name!r}: schema {expected_shape}, quantized {qt.shape}"
            )
        prev = np.empty_like(symbols)
        if symbols.size:
            prev[0] = START_SYMBOL
            prev[1:] = symbols[:-1]
        tensor_feature = _tensor_continuous_features(
            idx=idx,
            n_tensors=n_tensors,
            shape=tuple(int(v) for v in expected_shape),
            n_elements=int(symbols.size),
            scale=float(getattr(qt, "scale", 1.0)),
        )
        per_symbol_features = np.repeat(tensor_feature[None, :], symbols.size, axis=0)
        if include_position_features:
            position_features = _position_continuous_features(int(symbols.size))
            per_symbol_features = np.concatenate(
                [per_symbol_features, position_features],
                axis=1,
            )
        tensor_ids.append(np.full(symbols.shape, idx, dtype=np.int64))
        prev_symbols.append(prev)
        continuous_features.append(per_symbol_features)
        targets.append(symbols)
        counts = np.bincount(symbols, minlength=N_CATEGORIES)
        nonzero = int(np.count_nonzero(counts))
        per_tensor_rows.append({
            "idx": idx,
            "name": name,
            "shape": [int(v) for v in expected_shape],
            "n_elements": int(symbols.size),
            "n_distinct_symbols": nonzero,
            "scale_fp16_side_info_bytes": TENSOR_SCALE_BYTES_PER_TENSOR,
        })

    if not targets:
        raise SystemExit("empty FIXED_STATE_SCHEMA")

    all_targets = np.concatenate(targets)
    return SymbolCorpus(
        tensor_ids=torch.from_numpy(np.concatenate(tensor_ids)).long(),
        prev_symbols=torch.from_numpy(np.concatenate(prev_symbols)).long(),
        continuous_features=torch.from_numpy(np.concatenate(continuous_features)).float(),
        targets=torch.from_numpy(all_targets).long(),
        per_tensor_rows=per_tensor_rows,
        input_state_dict_sha256=input_sha256,
        n_total_symbols=int(all_targets.size),
    )


def _choose_train_indices(n_total: int, max_train_symbols: int) -> torch.Tensor:
    if max_train_symbols <= 0 or max_train_symbols >= n_total:
        return torch.arange(n_total, dtype=torch.long)
    # Deterministic coverage over the whole stream, avoiding random sampling
    # as a hidden source of run-to-run drift.
    indices = np.linspace(0, n_total - 1, num=max_train_symbols, dtype=np.int64)
    return torch.from_numpy(indices).long()


def _init_global_logits_from_counts(model: TinyFactorizedContextPMF, targets: torch.Tensor) -> None:
    counts = torch.bincount(targets, minlength=N_CATEGORIES).float()
    probs = (counts + 1.0) / (counts.sum() + N_CATEGORIES)
    with torch.no_grad():
        model.global_logits.copy_(torch.log(probs))


def _train_model(
    corpus: SymbolCorpus,
    *,
    config: TinyNnConfig,
    variant_name: str,
    use_prev_symbol: bool,
) -> TinyFactorizedContextPMF:
    if config.rank <= 0:
        raise SystemExit("--rank must be > 0")
    if config.epochs < 0:
        raise SystemExit("--epochs must be >= 0")
    if config.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")

    model = TinyFactorizedContextPMF(
        n_tensors=len(FIXED_STATE_SCHEMA),
        n_categories=N_CATEGORIES,
        rank=config.rank,
        n_continuous_features=corpus.continuous_features.shape[1],
        use_prev_symbol=use_prev_symbol,
    )
    _init_global_logits_from_counts(model, corpus.targets)
    if config.epochs == 0:
        return model

    train_indices = _choose_train_indices(corpus.n_total_symbols, config.max_train_symbols)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed + (101 if use_prev_symbol else 0))
    model.train()
    for epoch in range(config.epochs):
        del epoch
        order = train_indices[torch.randperm(train_indices.numel(), generator=generator)]
        for start in range(0, order.numel(), config.batch_size):
            idx = order[start:start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                corpus.tensor_ids[idx],
                corpus.prev_symbols[idx],
                corpus.continuous_features[idx],
            )
            loss = F.cross_entropy(logits, corpus.targets[idx], reduction="mean")
            loss.backward()
            optimizer.step()
    model.eval()
    # Keep an unreferenced local named by variant so future profiler output can
    # distinguish models while preserving deterministic state_dict ordering.
    model.variant_name = variant_name  # type: ignore[attr-defined]
    return model


def _cross_entropy_bits(
    model: TinyFactorizedContextPMF,
    corpus: SymbolCorpus,
    *,
    batch_size: int,
) -> tuple[float, list[dict[str, Any]]]:
    model.eval()
    total_nll_nats = 0.0
    tensor_nll = [0.0 for _ in FIXED_STATE_SCHEMA]
    tensor_counts = [0 for _ in FIXED_STATE_SCHEMA]
    with torch.no_grad():
        for start in range(0, corpus.n_total_symbols, batch_size):
            end = min(start + batch_size, corpus.n_total_symbols)
            sl = slice(start, end)
            logits = model(
                corpus.tensor_ids[sl],
                corpus.prev_symbols[sl],
                corpus.continuous_features[sl],
            )
            losses = F.cross_entropy(logits, corpus.targets[sl], reduction="none")
            total_nll_nats += float(losses.sum().item())
            ids = corpus.tensor_ids[sl]
            for tensor_idx in torch.unique(ids):
                mask = ids == tensor_idx
                idx_int = int(tensor_idx.item())
                tensor_nll[idx_int] += float(losses[mask].sum().item())
                tensor_counts[idx_int] += int(mask.sum().item())

    inv_log2 = 1.0 / math.log(2.0)
    per_tensor: list[dict[str, Any]] = []
    for row, nll, count in zip(corpus.per_tensor_rows, tensor_nll, tensor_counts, strict=True):
        bits = nll * inv_log2
        per_tensor.append({
            "idx": row["idx"],
            "name": row["name"],
            "n_elements": count,
            "model_bits": bits,
            "model_bytes": math.ceil(bits / 8.0),
            "bits_per_symbol": bits / count if count else 0.0,
        })
    return total_nll_nats * inv_log2, per_tensor


def _parameter_blobs(model: TinyFactorizedContextPMF) -> dict[str, bytes]:
    fp16_parts: list[bytes] = []
    int8_parts: list[bytes] = []
    for _name, tensor in model.state_dict().items():
        arr = tensor.detach().cpu().float().numpy()
        fp16_parts.append(arr.astype(np.float16).tobytes())
        max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
        scale = max_abs / 127.0 if max_abs > 0 else 1.0
        q = np.round(arr / scale).clip(-127, 127).astype(np.int8)
        int8_parts.append(np.array([scale], dtype=np.float16).tobytes())
        int8_parts.append(q.tobytes())
    return {
        "fp16": b"".join(fp16_parts),
        "int8_symmetric": b"".join(int8_parts),
    }


def _parameter_estimates(model: TinyFactorizedContextPMF) -> dict[str, int]:
    param_count = sum(int(p.numel()) for p in model.parameters())
    blobs = _parameter_blobs(model)
    return {
        "parameter_count": param_count,
        "raw_fp32_bytes": param_count * 4,
        "raw_fp16_bytes": len(blobs["fp16"]),
        "brotli_fp16_bytes": len(brotli.compress(blobs["fp16"], quality=11)),
        "raw_int8_symmetric_bytes": len(blobs["int8_symmetric"]),
        "brotli_int8_symmetric_bytes": len(
            brotli.compress(blobs["int8_symmetric"], quality=11)
        ),
    }


def _int8_dequantized_clone(model: TinyFactorizedContextPMF) -> TinyFactorizedContextPMF:
    cloned = copy.deepcopy(model)
    with torch.no_grad():
        for param in cloned.parameters():
            max_abs = float(param.detach().abs().max().item()) if param.numel() else 0.0
            scale = max_abs / 127.0 if max_abs > 0 else 1.0
            q = torch.round(param / scale).clamp(-127, 127)
            param.copy_(q * scale)
    cloned.eval()
    return cloned


def _summarize_variant(
    corpus: SymbolCorpus,
    *,
    config: TinyNnConfig,
    variant_name: str,
    use_prev_symbol: bool,
) -> dict[str, Any]:
    model = _train_model(
        corpus,
        config=config,
        variant_name=variant_name,
        use_prev_symbol=use_prev_symbol,
    )
    float_bits, per_tensor = _cross_entropy_bits(
        model,
        corpus,
        batch_size=config.batch_size,
    )
    quantized_model = _int8_dequantized_clone(model)
    int8_bits, int8_per_tensor = _cross_entropy_bits(
        quantized_model,
        corpus,
        batch_size=config.batch_size,
    )
    estimates = _parameter_estimates(model)
    primary_param_bytes = estimates["brotli_int8_symmetric_bytes"]
    symbol_payload_bytes = math.ceil(int8_bits / 8.0)
    reference_accounting_total = (
        symbol_payload_bytes + primary_param_bytes + ARCHIVE_OVERHEAD_BYTES
    )
    conservative_total = (
        reference_accounting_total
        + len(FIXED_STATE_SCHEMA) * TENSOR_SCALE_BYTES_PER_TENSOR
        + RANGE_STREAM_HEADER_BYTES_ESTIMATE
    )
    return {
        "variant": variant_name,
        "uses_previous_symbol": use_prev_symbol,
        "rank": config.rank,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "max_train_symbols": config.max_train_symbols,
        "fit_on_same_symbols": True,
        "model_parameter_byte_estimate": estimates,
        "primary_model_parameter_byte_estimate_name": "brotli_int8_symmetric_bytes",
        "primary_model_parameter_bytes": primary_param_bytes,
        "float_model_bits": float_bits,
        "float_model_payload_bytes": math.ceil(float_bits / 8.0),
        "int8_dequantized_model_bits": int8_bits,
        "int8_dequantized_model_payload_bytes": symbol_payload_bytes,
        "estimated_archive_bytes_reference_accounting": reference_accounting_total,
        "estimated_archive_bytes_conservative_packet_accounting": conservative_total,
        "delta_vs_brotli_optuna_reference_accounting": (
            reference_accounting_total - REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES
        ),
        "delta_vs_iid_per_tensor_floor_reference_accounting": (
            reference_accounting_total - REFERENCE_IID_PER_TENSOR_FLOOR_ARCHIVE_BYTES
        ),
        "delta_vs_per_tensor_aac_reference_accounting": (
            reference_accounting_total - REFERENCE_PER_TENSOR_AAC_ARCHIVE_BYTES
        ),
        "per_tensor_results_float_model": per_tensor,
        "per_tensor_results_int8_dequantized_model": int8_per_tensor,
    }


def build_tiny_nn_pmf_report(
    state_dict_path: Path,
    *,
    config: TinyNnConfig,
    variants: list[str],
) -> dict[str, Any]:
    bad_variants = sorted(set(variants) - {"tensor_only", "tensor_prev_symbol"})
    if bad_variants:
        raise SystemExit(f"unknown variants: {bad_variants}")
    _set_deterministic(config.seed, config.torch_threads)
    corpus = _load_symbol_corpus(
        state_dict_path,
        include_position_features=config.include_position_features,
    )
    variant_rows = [
        _summarize_variant(
            corpus,
            config=config,
            variant_name=variant,
            use_prev_symbol=(variant == "tensor_prev_symbol"),
        )
        for variant in variants
    ]
    best = min(
        variant_rows,
        key=lambda row: row["estimated_archive_bytes_reference_accounting"],
    )
    if best["delta_vs_brotli_optuna_reference_accounting"] < 0:
        disposition = "positive_smoke_not_dispatchable_without_packet_compiler"
    else:
        disposition = "negative_smoke_loses_after_charging_model_parameters"
    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "input_state_dict": str(state_dict_path),
        "input_state_dict_sha256": corpus.input_state_dict_sha256,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "score_claim": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            "planning_probe_only",
            "no_actual_range_or_ans_bitstream",
            "no_runtime_model_serializer_or_decoder",
            "model_parameter_quantization_is_estimated_not_packet_verified",
            "no_archive_substitution_performed",
            "missing_exact_cuda_auth_eval",
        ],
        "disposition": disposition,
        "cpu_only": True,
        "fixed_seed": config.seed,
        "torch_threads": config.torch_threads,
        "include_position_features": config.include_position_features,
        "n_tensors": len(FIXED_STATE_SCHEMA),
        "n_categories": N_CATEGORIES,
        "n_total_symbols": corpus.n_total_symbols,
        "archive_overhead_bytes": ARCHIVE_OVERHEAD_BYTES,
        "tensor_scale_bytes_total": len(FIXED_STATE_SCHEMA) * TENSOR_SCALE_BYTES_PER_TENSOR,
        "range_stream_header_bytes_estimate": RANGE_STREAM_HEADER_BYTES_ESTIMATE,
        "comparison_brotli_optuna_archive_bytes": REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES,
        "comparison_iid_per_tensor_floor_archive_bytes": REFERENCE_IID_PER_TENSOR_FLOOR_ARCHIVE_BYTES,
        "comparison_per_tensor_aac_archive_bytes": REFERENCE_PER_TENSOR_AAC_ARCHIVE_BYTES,
        "comparison_naive_markov1_aac_archive_bytes": REFERENCE_NAIVE_MARKOV1_AAC_ARCHIVE_BYTES,
        "best_variant_by_reference_accounting": {
            "variant": best["variant"],
            "estimated_archive_bytes_reference_accounting": best[
                "estimated_archive_bytes_reference_accounting"
            ],
            "delta_vs_brotli_optuna_reference_accounting": best[
                "delta_vs_brotli_optuna_reference_accounting"
            ],
            "delta_vs_iid_per_tensor_floor_reference_accounting": best[
                "delta_vs_iid_per_tensor_floor_reference_accounting"
            ],
            "delta_vs_per_tensor_aac_reference_accounting": best[
                "delta_vs_per_tensor_aac_reference_accounting"
            ],
        },
        "variants": variant_rows,
        "per_tensor_input_summary": corpus.per_tensor_rows,
    }


def _parse_variants(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    if not variants:
        raise SystemExit("at least one variant is required")
    return variants


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants",
        type=str,
        default="tensor_only,tensor_prev_symbol",
        help="Comma-separated variants: tensor_only,tensor_prev_symbol",
    )
    parser.add_argument("--rank", type=int, default=TinyNnConfig.rank)
    parser.add_argument("--epochs", type=int, default=TinyNnConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TinyNnConfig.learning_rate)
    parser.add_argument("--batch-size", type=int, default=TinyNnConfig.batch_size)
    parser.add_argument("--weight-decay", type=float, default=TinyNnConfig.weight_decay)
    parser.add_argument("--seed", type=int, default=TinyNnConfig.seed)
    parser.add_argument(
        "--max-train-symbols",
        type=int,
        default=TinyNnConfig.max_train_symbols,
        help="0 means train on all quantized symbols",
    )
    parser.add_argument("--torch-threads", type=int, default=TinyNnConfig.torch_threads)
    parser.add_argument(
        "--include-position-features",
        action="store_true",
        help="Add decoder-known normalized position features; off by default after smoke regressions.",
    )
    args = parser.parse_args(argv)
    if not args.state_dict_path.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict_path}")

    config = TinyNnConfig(
        rank=args.rank,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        seed=args.seed,
        max_train_symbols=args.max_train_symbols,
        torch_threads=args.torch_threads,
        include_position_features=args.include_position_features,
    )
    manifest = build_tiny_nn_pmf_report(
        args.state_dict_path,
        config=config,
        variants=_parse_variants(args.variants),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(
        f"{'variant':<22} {'payload':>10} {'model':>8} {'archive':>10} "
        f"{'vs_178144':>10} {'vs_175916':>10}"
    )
    for row in manifest["variants"]:
        print(
            f"{row['variant']:<22} "
            f"{row['int8_dequantized_model_payload_bytes']:>10,} "
            f"{row['primary_model_parameter_bytes']:>8,} "
            f"{row['estimated_archive_bytes_reference_accounting']:>10,} "
            f"{row['delta_vs_brotli_optuna_reference_accounting']:>+10,} "
            f"{row['delta_vs_iid_per_tensor_floor_reference_accounting']:>+10,}"
        )
    best = manifest["best_variant_by_reference_accounting"]
    print(
        f"Best {best['variant']}: "
        f"{best['estimated_archive_bytes_reference_accounting']:,} bytes "
        f"(vs Brotli+Optuna {REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES:,}: "
        f"{best['delta_vs_brotli_optuna_reference_accounting']:+,})"
    )
    print("Planning-only: no bitstream, no archive substitution, no exact-eval dispatch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
