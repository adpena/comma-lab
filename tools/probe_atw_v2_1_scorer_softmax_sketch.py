#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ATW V2-1 scorer-softmax sketch conditioning probe.

The ATW V2-1 queue is blocked on the next substrate-native scorer-logit sketch
or trained ATW residual gate. Raw scorer logits are not preserved by the prior
Faiss-PQ probe, but it did preserve canonical SegNet per-region softmax arrays
from A1-rendered frame_1. This helper converts those scorer-derived softmax
arrays into small deterministic dictionary packets, recomputes MI against A1
latent bytes, and labels the result as diagnostic-only.

This never dispatches a provider job and never claims a contest score.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "tools"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from probe_atw_v2_1_byte_closed_side_info_channel import (  # noqa: E402
    CONTEST_NORMALIZER_BYTES,
    DEFAULT_SIDE_INFO_BUDGET_BYTES,
    decode_side_info_packet,
    encode_side_info_packet,
)
from probe_latent_conditional_entropy_h_latent_given_scorer_class import (  # noqa: E402
    INDEPENDENCE_TOLERANCE_BITS,
    _count_joint,
    _count_symbols,
    _plug_in_entropy_bits,
)

DEFAULT_MEANINGFUL_MI_THRESHOLD_BITS = 0.5
HIGH_CARDINALITY_UNIQUE_FRACTION = 0.25

DEFAULT_STATE_JSON = (
    REPO_ROOT / ".omx" / "state" / "atw_v2_1_scorer_softmax_sketch_probe.json"
)
DEFAULT_RESEARCH_JSON = (
    REPO_ROOT
    / ".omx"
    / "research"
    / "atw_v2_1_scorer_softmax_sketch_probe_20260518_codex.json"
)
DEFAULT_RESEARCH_MD = (
    REPO_ROOT
    / ".omx"
    / "research"
    / "atw_v2_1_scorer_softmax_sketch_probe_20260518_codex.md"
)


@dataclass(frozen=True)
class SketchVariant:
    variant_id: str
    source_regions: int
    description: str
    per_pair_symbols: list[int]


@dataclass(frozen=True)
class SketchMiVerdict:
    verdict: str
    h_latent_unconditional_bits_per_symbol: float
    h_latent_given_side_info_bits_per_symbol: float
    mutual_information_bits: float
    wyner_ziv_gain_ceiling_fraction: float
    num_latent_symbols: int
    num_side_info_symbols: int
    num_unique_side_info_symbols: int
    meaningful_mi_threshold_bits: float
    independence_tolerance_bits: float


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def default_output_dir() -> Path:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "experiments" / "results" / f"atw_v2_1_scorer_softmax_sketch_probe_{stamp}"


def replay_command() -> str:
    return " ".join([".venv/bin/python", repo_rel(Path(__file__)), *sys.argv[1:]])


def find_latest_softmax_input_dir() -> Path:
    candidates = sorted(
        (
            path.parent
            for path in (
                REPO_ROOT / "experiments" / "results"
            ).glob("atw_v2_1_faiss_pq_probe_*/segnet_region_softmax_16.npy")
            if (path.parent / "segnet_region_softmax_256.npy").is_file()
            and (path.parent / "a1_latents_u8.bin").is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "no cached ATW V2-1 Faiss-PQ softmax directory found under "
            "experiments/results/atw_v2_1_faiss_pq_probe_*"
        )
    return candidates[0]


def _parse_softmax_npy_args(items: list[str] | None) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--softmax-npy item must be REGIONS=PATH, got {item!r}")
        key, value = item.split("=", 1)
        out[int(key)] = Path(value)
    return out


def load_cached_inputs(
    *,
    input_dir: Path | None,
    latent_bytes_path: Path | None,
    softmax_npy: dict[int, Path],
    provenance_json: Path | None,
) -> tuple[bytes, dict[int, np.ndarray], dict[str, Any]]:
    root = input_dir or find_latest_softmax_input_dir()
    latent_path = latent_bytes_path or (root / "a1_latents_u8.bin")
    paths = {
        16: root / "segnet_region_softmax_16.npy",
        256: root / "segnet_region_softmax_256.npy",
    }
    paths.update(softmax_npy)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing softmax npy path(s): {missing}")
    if not latent_path.is_file():
        raise FileNotFoundError(f"missing latent bytes path: {latent_path}")
    provenance_path = provenance_json or (root / "softmax_collection_provenance.json")
    provenance: dict[str, Any] = {}
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.update(
        {
            "cached_input_dir": repo_rel(root),
            "latent_bytes_path": repo_rel(latent_path),
            "softmax_npy_paths": {str(k): repo_rel(v) for k, v in sorted(paths.items())},
            "scorer_signal_scope": (
                "SegNet per-region softmax arrays from A1-rendered frame_1; "
                "raw scorer logits were not available in the cached upstream artifact"
            ),
        }
    )
    arrays = {n_regions: np.load(path) for n_regions, path in paths.items()}
    return latent_path.read_bytes(), arrays, provenance


def _as_softmax_array(array: np.ndarray, *, n_regions: int) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"softmax array must be rank-3, got {arr.shape}")
    if int(arr.shape[1]) != n_regions:
        raise ValueError(f"expected {n_regions} regions, got {arr.shape[1]}")
    if int(arr.shape[2]) < 2:
        raise ValueError("softmax array must have at least two classes")
    if not np.isfinite(arr).all():
        raise ValueError("softmax array contains non-finite values")
    sums = arr.sum(axis=2, keepdims=True)
    sums = np.where(sums <= 0, 1.0, sums)
    return arr / sums


def _quantize_unit(value: float, levels: int) -> int:
    if levels <= 1:
        raise ValueError("levels must be > 1")
    clipped = min(max(float(value), 0.0), 1.0)
    return min(levels - 1, int(math.floor(clipped * levels)))


def _pack_digits(digits: Sequence[int], bases: Sequence[int]) -> int:
    if len(digits) != len(bases):
        raise ValueError("digits and bases length mismatch")
    value = 0
    for digit, base in zip(digits, bases, strict=True):
        if base <= 0:
            raise ValueError("bases must be positive")
        if digit < 0 or digit >= base:
            raise ValueError(f"digit {digit} outside base {base}")
        value = value * base + digit
    return int(value)


def _entropy_deficit(region_softmax: np.ndarray) -> np.ndarray:
    clipped = np.clip(region_softmax.astype(np.float64), 1e-12, 1.0)
    entropy = -(clipped * np.log2(clipped)).sum(axis=1)
    return math.log2(region_softmax.shape[1]) - entropy


def _global_mean_softmax_q3(softmax16: np.ndarray) -> list[int]:
    global_probs = softmax16.mean(axis=1)
    bases = [8] * int(global_probs.shape[1])
    return [
        _pack_digits([_quantize_unit(prob, 8) for prob in row], bases)
        for row in global_probs
    ]


def _global_top2_margin_q5(softmax16: np.ndarray) -> list[int]:
    global_probs = softmax16.mean(axis=1)
    n_classes = int(global_probs.shape[1])
    out: list[int] = []
    for row in global_probs:
        order = np.argsort(-row, kind="stable")
        top1 = int(order[0])
        top2 = int(order[1])
        conf = float(row[top1])
        margin = float(row[top1] - row[top2])
        out.append(
            _pack_digits(
                [top1, top2, _quantize_unit(conf, 32), _quantize_unit(margin, 32)],
                [n_classes, n_classes, 32, 32],
            )
        )
    return out


def _region16_entropy_anchor_q4(softmax16: np.ndarray) -> list[int]:
    n_classes = int(softmax16.shape[2])
    max_deficit = math.log2(n_classes)
    out: list[int] = []
    for row in softmax16:
        deficits = _entropy_deficit(row)
        region = int(np.argmax(deficits))
        probs = row[region]
        top = int(np.argmax(probs))
        out.append(
            _pack_digits(
                [
                    region,
                    top,
                    _quantize_unit(float(probs[top]), 16),
                    _quantize_unit(float(deficits[region] / max_deficit), 16),
                ],
                [16, n_classes, 16, 16],
            )
        )
    return out


def _region16_presence_confmask_q4(softmax16: np.ndarray) -> list[int]:
    n_classes = int(softmax16.shape[2])
    if n_classes > 30:
        raise ValueError("presence mask supports at most 30 classes")
    max_deficit = math.log2(n_classes)
    out: list[int] = []
    for row in softmax16:
        top_by_region = np.argmax(row, axis=1)
        mask = 0
        for cls in top_by_region:
            mask |= 1 << int(cls)
        max_conf = float(np.max(row))
        mean_conf = float(np.mean(np.max(row, axis=1)))
        max_def = float(np.max(_entropy_deficit(row)) / max_deficit)
        out.append(
            _pack_digits(
                [
                    mask,
                    _quantize_unit(max_conf, 16),
                    _quantize_unit(mean_conf, 16),
                    _quantize_unit(max_def, 16),
                ],
                [1 << n_classes, 16, 16, 16],
            )
        )
    return out


def _region256_coarse_entropy_anchor_q4(softmax256: np.ndarray) -> list[int]:
    n_classes = int(softmax256.shape[2])
    max_deficit = math.log2(n_classes)
    grid_side = int(round(math.sqrt(int(softmax256.shape[1]))))
    if grid_side * grid_side != int(softmax256.shape[1]):
        raise ValueError("256-region softmax input must form a square grid")
    if grid_side % 4:
        raise ValueError("region grid side must be divisible by 4 for coarse anchors")
    out: list[int] = []
    for row in softmax256:
        deficits = _entropy_deficit(row)
        region = int(np.argmax(deficits))
        y, x = divmod(region, grid_side)
        coarse_side = 4
        coarse = (y // (grid_side // coarse_side)) * coarse_side + (
            x // (grid_side // coarse_side)
        )
        probs = row[region]
        top = int(np.argmax(probs))
        out.append(
            _pack_digits(
                [
                    int(coarse),
                    top,
                    _quantize_unit(float(probs[top]), 16),
                    _quantize_unit(float(deficits[region] / max_deficit), 16),
                ],
                [16, n_classes, 16, 16],
            )
        )
    return out


def build_scorer_softmax_sketches(
    softmax_by_region_count: dict[int, np.ndarray],
) -> tuple[SketchVariant, ...]:
    softmax16 = _as_softmax_array(softmax_by_region_count[16], n_regions=16)
    softmax256 = _as_softmax_array(softmax_by_region_count[256], n_regions=256)
    if int(softmax16.shape[0]) != int(softmax256.shape[0]):
        raise ValueError("softmax16 and softmax256 pair counts disagree")
    return (
        SketchVariant(
            variant_id="global_mean_softmax_q3",
            source_regions=16,
            description="16-region mean SegNet softmax, 3-bit class buckets",
            per_pair_symbols=_global_mean_softmax_q3(softmax16),
        ),
        SketchVariant(
            variant_id="global_top2_margin_q5",
            source_regions=16,
            description="global top-2 classes plus 5-bit confidence and margin",
            per_pair_symbols=_global_top2_margin_q5(softmax16),
        ),
        SketchVariant(
            variant_id="region16_entropy_anchor_q4",
            source_regions=16,
            description="most non-uniform 16-region cell with class/confidence/entropy buckets",
            per_pair_symbols=_region16_entropy_anchor_q4(softmax16),
        ),
        SketchVariant(
            variant_id="region16_presence_confmask_q4",
            source_regions=16,
            description="top-class presence mask plus confidence and entropy buckets",
            per_pair_symbols=_region16_presence_confmask_q4(softmax16),
        ),
        SketchVariant(
            variant_id="region256_coarse_entropy_anchor_q4",
            source_regions=256,
            description="most non-uniform 256-region cell collapsed to a 4x4 anchor",
            per_pair_symbols=_region256_coarse_entropy_anchor_q4(softmax256),
        ),
    )


def compute_sketch_mi_verdict(
    *,
    latent_stream: bytes,
    per_pair_symbols: Sequence[int],
    symbols_per_pair: int,
    threshold: float = DEFAULT_MEANINGFUL_MI_THRESHOLD_BITS,
) -> SketchMiVerdict:
    if symbols_per_pair <= 0:
        raise ValueError("symbols_per_pair must be positive")
    if not per_pair_symbols:
        raise ValueError("per_pair_symbols must be non-empty")
    expanded: list[int] = []
    for symbol in per_pair_symbols:
        if int(symbol) < 0:
            raise ValueError("per_pair_symbols must be non-negative")
        expanded.extend([int(symbol)] * symbols_per_pair)
    if len(expanded) != len(latent_stream):
        raise ValueError(
            f"expanded side-info length {len(expanded)} != latent length {len(latent_stream)}"
        )
    latent_counts = _count_symbols(latent_stream)
    side_counts = _count_symbols(expanded)
    joint = _count_joint(latent_stream, expanded)
    h_latent = _plug_in_entropy_bits(latent_counts)
    total = sum(side_counts.values())
    h_cond = 0.0
    for symbol, per_symbol in joint.items():
        h_cond += (side_counts[symbol] / total) * _plug_in_entropy_bits(per_symbol)
    mi = max(h_latent - h_cond, 0.0)
    if mi <= INDEPENDENCE_TOLERANCE_BITS:
        verdict = "INDEPENDENT"
    elif mi >= threshold:
        verdict = "MEANINGFUL_CONDITIONING"
    else:
        verdict = "WEAK_CONDITIONING"
    return SketchMiVerdict(
        verdict=verdict,
        h_latent_unconditional_bits_per_symbol=float(h_latent),
        h_latent_given_side_info_bits_per_symbol=float(h_cond),
        mutual_information_bits=float(mi),
        wyner_ziv_gain_ceiling_fraction=float(mi / h_latent if h_latent > 0 else 0.0),
        num_latent_symbols=len(latent_stream),
        num_side_info_symbols=len(expanded),
        num_unique_side_info_symbols=len(set(int(v) for v in per_pair_symbols)),
        meaningful_mi_threshold_bits=threshold,
        independence_tolerance_bits=INDEPENDENCE_TOLERANCE_BITS,
    )


def variant_action(row: dict[str, Any]) -> str:
    if not row["byte_budget_ok"]:
        return "reject_or_recode_sketch_payload_before_mi_interpretation"
    if row["high_cardinality_bias_guard_triggered"]:
        return "reject_as_plugin_mi_upper_bound_until_lower_cardinality_or_heldout_probe"
    if row["verdict"]["verdict"] == "MEANINGFUL_CONDITIONING":
        return "run_new_d4_probe_on_selected_scorer_softmax_sketch_before_dispatch"
    if row["verdict"]["verdict"] == "WEAK_CONDITIONING":
        return "pivot_to_raw_logit_head_or_trained_atw_residual_probe"
    return "do_not_dispatch_from_this_sketch"


def build_probe_payload(
    *,
    latent_stream: bytes,
    softmax_by_region_count: dict[int, np.ndarray],
    output_dir: Path,
    softmax_provenance: dict[str, Any] | None = None,
    budget_bytes: int = DEFAULT_SIDE_INFO_BUDGET_BYTES,
    max_pairs: int | None = None,
) -> dict[str, Any]:
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive")
    if not latent_stream:
        raise ValueError("latent_stream must be non-empty")
    n_pairs_available = int(next(iter(softmax_by_region_count.values())).shape[0])
    if max_pairs is None:
        n_pairs = n_pairs_available
    else:
        if max_pairs <= 0:
            raise ValueError("max_pairs must be positive")
        n_pairs = min(max_pairs, n_pairs_available)
    if len(latent_stream) % n_pairs_available != 0:
        raise ValueError(
            f"latent stream length {len(latent_stream)} not divisible by "
            f"available pair count {n_pairs_available}"
        )
    symbols_per_pair = len(latent_stream) // n_pairs_available
    latent_slice = latent_stream[: n_pairs * symbols_per_pair]
    sliced_softmax = {
        n_regions: np.asarray(array[:n_pairs])
        for n_regions, array in softmax_by_region_count.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    variants: list[dict[str, Any]] = []
    for sketch in build_scorer_softmax_sketches(sliced_softmax):
        packet = encode_side_info_packet(
            sketch.per_pair_symbols,
            reducer_name=sketch.variant_id,
        )
        decoded_name, decoded_values = decode_side_info_packet(packet)
        roundtrip_ok = (
            decoded_name == sketch.variant_id and decoded_values == sketch.per_pair_symbols
        )
        if not roundtrip_ok:
            raise ValueError(f"sketch packet roundtrip failed for {sketch.variant_id}")
        packet_path = output_dir / f"atw_v2_1_scorer_softmax_sketch_{sketch.variant_id}.bin"
        packet_path.write_bytes(packet)
        verdict = compute_sketch_mi_verdict(
            latent_stream=latent_slice,
            per_pair_symbols=sketch.per_pair_symbols,
            symbols_per_pair=symbols_per_pair,
        )
        unique_fraction = verdict.num_unique_side_info_symbols / n_pairs
        high_cardinality = unique_fraction > HIGH_CARDINALITY_UNIQUE_FRACTION
        packet_bytes = len(packet)
        blockers: list[str] = []
        if high_cardinality:
            blockers.append("scorer_softmax_sketch_high_cardinality_plugin_mi_upper_bound_only")
        if packet_bytes > budget_bytes:
            blockers.append("scorer_softmax_sketch_packet_exceeds_side_info_budget")
        if verdict.verdict != "MEANINGFUL_CONDITIONING":
            blockers.append("scorer_softmax_sketch_did_not_reach_meaningful_mi_threshold")
        row: dict[str, Any] = {
            "variant_id": sketch.variant_id,
            "source_regions": sketch.source_regions,
            "description": sketch.description,
            "verdict": asdict(verdict),
            "unique_fraction": unique_fraction,
            "high_cardinality_bias_guard_triggered": high_cardinality,
            "packet_path": repo_rel(packet_path),
            "packet_sha256": sha256_bytes(packet),
            "packet_bytes": packet_bytes,
            "byte_budget": budget_bytes,
            "byte_budget_ok": packet_bytes <= budget_bytes,
            "side_info_rate_score_cost": 25.0 * packet_bytes / CONTEST_NORMALIZER_BYTES,
            "packet_roundtrip_ok": roundtrip_ok,
            "dispatch_blockers": blockers,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_paid_dispatch": False,
        }
        row["phase2_action"] = variant_action(row)
        variants.append(row)

    meaningful_without_bias = [
        row
        for row in variants
        if row["byte_budget_ok"]
        and not row["high_cardinality_bias_guard_triggered"]
        and row["verdict"]["verdict"] == "MEANINGFUL_CONDITIONING"
    ]
    weak_or_meaningful = [
        row
        for row in variants
        if row["byte_budget_ok"]
        and row["verdict"]["verdict"] in {"MEANINGFUL_CONDITIONING", "WEAK_CONDITIONING"}
    ]
    if meaningful_without_bias:
        phase2_status = (
            "scorer_softmax_sketch_meaningful_requires_new_d4_and_wave_n_plus_1_council"
        )
        recommended_next_gate = "run_new_d4_probe_on_selected_scorer_softmax_sketch_before_dispatch"
    elif weak_or_meaningful:
        phase2_status = "scorer_softmax_sketches_only_weak_or_biased_conditioning"
        recommended_next_gate = "trained_atw_residual_probe_or_raw_scorer_logit_head_design"
    else:
        phase2_status = "scorer_softmax_sketches_independent_or_over_budget"
        recommended_next_gate = "defer_atw_v2_1_softmax_sketch_and_train_residual_channel"

    best = max(
        variants,
        key=lambda row: (
            float(row["verdict"]["mutual_information_bits"]),
            -int(row["packet_bytes"]),
        ),
    )
    best_actionable = (
        max(
            meaningful_without_bias,
            key=lambda row: (
                float(row["verdict"]["mutual_information_bits"]),
                -int(row["packet_bytes"]),
            ),
        )
        if meaningful_without_bias
        else None
    )

    return {
        "schema": "atw_v2_1_scorer_softmax_sketch_probe_v1",
        "observed_at_utc": utc_now(),
        "command": replay_command(),
        "output_dir": repo_rel(output_dir),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "dispatch_attempted": False,
        "provider_spend_attempted": False,
        "evidence_grade": "diagnostic_cpu",
        "axis_label": "[diagnostic-CPU; ATW V2-1 scorer-softmax sketch MI probe]",
        "side_info_budget_bytes": budget_bytes,
        "meaningful_mi_threshold_bits": DEFAULT_MEANINGFUL_MI_THRESHOLD_BITS,
        "high_cardinality_unique_fraction_limit": HIGH_CARDINALITY_UNIQUE_FRACTION,
        "num_pairs": n_pairs,
        "symbols_per_pair": symbols_per_pair,
        "variants": variants,
        "best_variant": {
            "variant_id": best["variant_id"],
            "verdict": best["verdict"]["verdict"],
            "mutual_information_bits": best["verdict"]["mutual_information_bits"],
            "packet_bytes": best["packet_bytes"],
            "byte_budget_ok": best["byte_budget_ok"],
            "high_cardinality_bias_guard_triggered": best[
                "high_cardinality_bias_guard_triggered"
            ],
            "phase2_action": best["phase2_action"],
        },
        "best_actionable_variant": (
            {
                "variant_id": best_actionable["variant_id"],
                "mutual_information_bits": best_actionable["verdict"][
                    "mutual_information_bits"
                ],
                "packet_bytes": best_actionable["packet_bytes"],
                "phase2_action": best_actionable["phase2_action"],
            }
            if best_actionable is not None
            else None
        ),
        "phase2_status": phase2_status,
        "recommended_next_gate": recommended_next_gate,
        "result_review_blockers": [
            "diagnostic_probe_not_score_claim",
            "softmax_sketch_not_raw_scorer_logit_channel",
            "requires_new_d4_probe_before_dispatch_authority",
            "requires_paired_contest_cuda_cpu_harvest_before_promotion",
        ],
        "softmax_provenance": softmax_provenance or {},
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ATW V2-1 Scorer-Softmax Sketch Probe",
        "",
        f"- observed_at_utc: `{payload['observed_at_utc']}`",
        f"- axis_label: `{payload['axis_label']}`",
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        "- dispatch_attempted: `false`",
        "- provider_spend_attempted: `false`",
        f"- side_info_budget_bytes: `{payload['side_info_budget_bytes']}`",
        f"- phase2_status: `{payload['phase2_status']}`",
        f"- recommended_next_gate: `{payload['recommended_next_gate']}`",
        "",
        "## Variant Results",
        "",
        "| Variant | Packet bytes | Rate cost | Unique frac | MI bits/symbol | Verdict | Bias guard | Phase 2 action | Blockers |",
        "|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in payload["variants"]:
        verdict = row["verdict"]
        lines.append(
            "| {variant} | {bytes} | {rate:.8f} | {unique:.3f} | {mi:.12f} | {verdict} | {bias} | {action} | {blockers} |".format(
                variant=row["variant_id"],
                bytes=row["packet_bytes"],
                rate=row["side_info_rate_score_cost"],
                unique=row["unique_fraction"],
                mi=verdict["mutual_information_bits"],
                verdict=verdict["verdict"],
                bias=str(row["high_cardinality_bias_guard_triggered"]).lower(),
                action=row["phase2_action"],
                blockers=", ".join(row["dispatch_blockers"]) or "none",
            )
        )
    best = payload["best_variant"]
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            (
                f"Best sketch: `{best['variant_id']}` with verdict `{best['verdict']}`, "
                f"MI `{best['mutual_information_bits']:.12f}` bits/symbol, packet "
                f"bytes `{best['packet_bytes']}`, high-cardinality guard "
                f"`{str(best['high_cardinality_bias_guard_triggered']).lower()}`."
            ),
        ]
    )
    if payload["best_actionable_variant"] is None:
        lines.extend(
            [
                "",
                "No scorer-softmax sketch is dispatch authority. The probe is either",
                "weak, biased by high side-info cardinality, or over the configured",
                "side-info budget. This keeps ATW V2-1 on the trained residual or raw",
                "scorer-logit-head gate rather than paid dispatch.",
            ]
        )
    else:
        action = payload["best_actionable_variant"]
        lines.extend(
            [
                "",
                (
                    f"Actionable diagnostic sketch: `{action['variant_id']}` with MI "
                    f"`{action['mutual_information_bits']:.12f}` bits/symbol and "
                    f"`{action['packet_bytes']}` packet bytes."
                ),
                "This still requires a new D4 probe and Wave N+1 council before paid dispatch.",
            ]
        )
    lines.extend(
        [
            "",
            "## False-Authority Guard",
            "",
            "This diagnostic artifact uses cached SegNet softmax arrays, not raw scorer logits.",
            "It is not a contest score, not promotion evidence, and not provider-spend authority.",
            "",
            "## Reproduction",
            "",
            f"- command: `{payload['command']}`",
            f"- output_dir: `{payload['output_dir']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--latent-bytes", type=Path, default=None)
    parser.add_argument("--softmax-npy", action="append", default=[])
    parser.add_argument("--softmax-provenance-json", type=Path, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--budget-bytes", type=int, default=DEFAULT_SIDE_INFO_BUDGET_BYTES)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--research-json", type=Path, default=DEFAULT_RESEARCH_JSON)
    parser.add_argument("--research-md", type=Path, default=DEFAULT_RESEARCH_MD)
    args = parser.parse_args(argv)

    output_dir = args.output_dir or default_output_dir()
    latent_stream, softmax_by_region_count, provenance = load_cached_inputs(
        input_dir=args.input_dir,
        latent_bytes_path=args.latent_bytes,
        softmax_npy=_parse_softmax_npy_args(args.softmax_npy),
        provenance_json=args.softmax_provenance_json,
    )
    payload = build_probe_payload(
        latent_stream=latent_stream,
        softmax_by_region_count=softmax_by_region_count,
        output_dir=output_dir,
        softmax_provenance=provenance,
        budget_bytes=args.budget_bytes,
        max_pairs=args.max_pairs,
    )
    write_json(output_dir / "atw_v2_1_scorer_softmax_sketch_probe.json", payload)
    write_json(args.json_out, payload)
    write_json(args.research_json, payload)
    args.research_md.parent.mkdir(parents=True, exist_ok=True)
    args.research_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"[atw-softmax-sketch] wrote {args.json_out}")
    print(f"[atw-softmax-sketch] phase2_status={payload['phase2_status']}")
    print(f"[atw-softmax-sketch] recommended_next_gate={payload['recommended_next_gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
