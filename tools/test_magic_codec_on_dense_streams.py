"""Test the magic codec on TRAINER-FRESH dense streams (not entropy-saturated).

Per operator amplification 2026-05-11 ("push magic codec further") + council §B
Q3 verdict (#27 magic codec optimized variant on PR101 grammar archive +
deferred dense-streams investigation per Insight 3):

AA's landing memo (`feedback_magic_codec_auto_selector_landed_20260511.md`)
empirically established that the magic codec on PR106 r2's already-packet-
grammar-compressed ``0.bin`` saves only +1016 bytes (-0.5%). That archive is
*entropy-saturated* by construction — its bytes are already an AC-coded
PR101 grammar payload, so re-running AC over those bytes interpreted as int8
is a no-op (and the wrapper overhead loses).

The research-signal question this tool answers is:

    When the trainer produces DENSE un-coded streams (quantised int8
    decoder weights, dense FP4 codebook indices, dense pose deltas, dense
    quantized integer residuals), which packet_compiler primitive wins per
    stream class, and how much do we save vs naive int8 dense storage?

The tool DOES NOT touch a real trainer. It synthesises 4 representative
dense stream shapes that mimic trainer output statistics:

    1. ``dense_decoder_weights`` — int8 quantised decoder weights with
       peaked-Laplace residual distribution (~50% zeros, ~30% ±1).
    2. ``dense_fp4_codebook_indices`` — 4-bit codebook indices stored as
       int8, alphabet size 16, near-uniform distribution.
    3. ``dense_pose_deltas`` — 2D (n_frames, 6) float pose deltas with
       small-magnitude integer-rounding behaviour.
    4. ``dense_quantized_int_residuals`` — int8 residuals after quantisation
       with high sparsity (~90% zeros, ±1 tail).

For each stream class, the tool runs ``encode_magic_codec`` over the magic
codec's candidate inventory (the 6 primitives that map to the 0xF0-0xFF
reserved namespace) and records:

* selected primitive name + bytes
* baseline naive int8 dense storage byte count
* savings ratio (positive = magic codec wins; negative = magic codec loses
  to naive storage)
* selection log (per-primitive byte count + refusal reason)

The output is a typed JSON manifest:

    {
      "schema": "magic_codec_dense_streams_test.v1",
      "operator": "...",
      "generated_at_utc": "...",
      "results_by_stream": {
        "<class>": {
          "shape": [...],
          "dtype": "...",
          "naive_int8_bytes": ...,
          "magic_codec_bytes": ...,
          "savings_ratio_vs_naive": ...,
          "selected_primitive": "...",
          "selection_strategy": "...",
          "selection_log": [...]
        },
        ...
      },
      "aggregate": {
        "total_naive_int8_bytes": ...,
        "total_magic_codec_bytes": ...,
        "aggregate_savings_ratio": ...
      },
      "score_claim": false,
      "promotion_eligible": false,
      "ready_for_exact_eval_dispatch": false,
      "byte_proxy_only": true,
      "cuda_eval_worth_testing": false
    }

CLAUDE.md compliance:

* no scorer load (pure numpy + brotli + lzma + constriction);
* no MPS / torch import;
* no ``/tmp`` paths — refused at output-dir validation;
* deterministic-bytes: synthetic input is seeded; same args produce
  byte-identical output;
* no score claim ever;
* this is a RESEARCH-SIGNAL tool only — the predicted savings are
  byte-level proxies. A real trainer-fresh stream may have different
  statistics and the council should review any subsequent
  optimize-mode dispatch against this manifest's predictions.

Usage::

    .venv/bin/python tools/test_magic_codec_on_dense_streams.py \\
        --output-dir experiments/results/magic_codec_dense_streams_test_<ts>/

    .venv/bin/python tools/test_magic_codec_on_dense_streams.py \\
        --output-dir <dir> \\
        --selection-strategy entropy_estimate

    .venv/bin/python tools/test_magic_codec_on_dense_streams.py \\
        --dry-run  # prints manifest to stdout, writes nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from tac.packet_compiler.magic_codec import (
    MagicCodecError,
    MagicCodecResult,
    SelectionStrategy,
    StreamHint,
    StreamType,
    candidate_primitives_for,
    encode_magic_codec,
    recommendation_for,
)


_SELECTION_STRATEGIES: tuple[str, ...] = (
    "smallest_byte_count",
    "entropy_estimate",
    "stacked_optimal",
)


# ── Stream-class synthetic generators ───────────────────────────────────────


@dataclass(frozen=True)
class DenseStreamSpec:
    """Synthetic-stream specification for the test.

    Each spec describes ONE trainer-fresh dense stream class (its shape,
    dtype, generator function name, statistical signature, and the
    ``StreamType`` we pass to the magic codec).
    """

    name: str
    description: str
    stream_type: StreamType
    shape: tuple[int, ...]
    dtype: str
    statistical_signature: str


_DENSE_STREAM_SPECS: tuple[DenseStreamSpec, ...] = (
    DenseStreamSpec(
        name="dense_decoder_weights",
        description=(
            "Int8 quantised decoder weights with peaked-Laplace residual "
            "distribution. Models the post-QAT weight tensor before any "
            "AC packing."
        ),
        stream_type="weight_tensor",
        shape=(65536,),
        dtype="int8",
        statistical_signature="laplace_b=1.5_clip=127",
    ),
    DenseStreamSpec(
        name="dense_fp4_codebook_indices",
        description=(
            "FP4 codebook indices stored as int8 (alphabet size 16, "
            "near-uniform distribution). Models a categorical codebook "
            "stream before HPAC packing."
        ),
        stream_type="categorical",
        shape=(32768,),
        dtype="int32",
        statistical_signature="uniform_alphabet_16",
    ),
    DenseStreamSpec(
        name="dense_pose_deltas",
        description=(
            "2D float pose deltas (n_frames, 6) with small-magnitude "
            "integer-rounding behaviour. Models the PR93 delta-varint "
            "input before encoding."
        ),
        stream_type="pose",
        shape=(600, 6),
        dtype="float32",
        statistical_signature="gaussian_sigma=2.5_int_rounded",
    ),
    DenseStreamSpec(
        name="dense_quantized_int_residuals",
        description=(
            "Int8 residuals after quantisation with very high sparsity "
            "(~90% zeros, ±1 tail). Models the residual_basis stream "
            "PR101/PR103 grammars feed AC."
        ),
        stream_type="residual_basis",
        shape=(131072,),
        dtype="int8",
        statistical_signature="sparse_p_nonzero=0.10_tail_geometric",
    ),
)


def _synthesize_dense_decoder_weights(
    *, n_elements: int, rng: np.random.Generator
) -> np.ndarray:
    """Laplace-distributed int8 weights, clipped to [-127, 127]."""
    raw = rng.laplace(loc=0.0, scale=1.5, size=int(n_elements))
    return np.clip(np.round(raw), -127, 127).astype(np.int8)


def _synthesize_dense_fp4_codebook_indices(
    *, n_elements: int, rng: np.random.Generator
) -> np.ndarray:
    """Uniform [0, 16) categorical indices stored as int32."""
    return rng.integers(low=0, high=16, size=int(n_elements), dtype=np.int32)


def _synthesize_dense_pose_deltas(
    *, n_frames: int, rng: np.random.Generator
) -> np.ndarray:
    """2D (n_frames, 6) gaussian pose deltas rounded to integer-valued floats."""
    raw = rng.normal(loc=0.0, scale=2.5, size=(int(n_frames), 6))
    return np.round(raw).astype(np.float32)


def _synthesize_dense_quantized_int_residuals(
    *, n_elements: int, rng: np.random.Generator
) -> np.ndarray:
    """Sparse int8 residuals: 90% zeros, 10% small geometric tail."""
    base = np.zeros(int(n_elements), dtype=np.int8)
    nonzero_mask = rng.random(int(n_elements)) < 0.10
    n_nonzero = int(nonzero_mask.sum())
    if n_nonzero > 0:
        signs = rng.choice([-1, 1], size=n_nonzero).astype(np.int8)
        magnitudes = (rng.geometric(p=0.6, size=n_nonzero).astype(np.int16) - 1)
        magnitudes = np.clip(magnitudes, 0, 6).astype(np.int8)
        base[nonzero_mask] = (signs * (magnitudes + 1)).astype(np.int8)
    return base


def _synthesize_stream(
    spec: DenseStreamSpec, *, rng: np.random.Generator
) -> np.ndarray:
    """Dispatch to the per-class synthetic generator."""
    if spec.name == "dense_decoder_weights":
        return _synthesize_dense_decoder_weights(n_elements=spec.shape[0], rng=rng)
    if spec.name == "dense_fp4_codebook_indices":
        return _synthesize_dense_fp4_codebook_indices(
            n_elements=spec.shape[0], rng=rng
        )
    if spec.name == "dense_pose_deltas":
        return _synthesize_dense_pose_deltas(n_frames=spec.shape[0], rng=rng)
    if spec.name == "dense_quantized_int_residuals":
        return _synthesize_dense_quantized_int_residuals(
            n_elements=spec.shape[0], rng=rng
        )
    raise SystemExit(f"unknown stream-spec name {spec.name!r}")


# ── Tool entrypoints ────────────────────────────────────────────────────────


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="test_magic_codec_on_dense_streams",
        description=(
            "Synthesise 4 trainer-fresh dense stream classes and run the "
            "magic codec over each. Outputs a typed JSON manifest of per-"
            "class selection + savings vs naive int8 dense storage. "
            "score_claim=false; byte_proxy_only=true."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Must NOT be under /tmp (CLAUDE.md "
            "forbidden_/tmp_paths_in_any_persisted_artifact). Canonical "
            "location: experiments/results/<lane_id>_<timestamp>/. "
            "Required unless --dry-run is set."
        ),
    )
    parser.add_argument(
        "--selection-strategy",
        choices=_SELECTION_STRATEGIES,
        default="smallest_byte_count",
        help=(
            "Magic-codec selection strategy. Default: smallest_byte_count."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260511,
        help=(
            "RNG seed for synthetic stream generation. Default: 20260511. "
            "Same seed + same code produce byte-identical streams."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the manifest to stdout WITHOUT writing any files. Useful "
            "for research-signal-only ranking."
        ),
    )
    parser.add_argument(
        "--operator",
        default=None,
        help=(
            "Operator handle for manifest provenance. When omitted, the "
            "manifest records operator=unknown."
        ),
    )
    return parser.parse_args(argv)


def _validate_output_dir(output_dir: Path) -> None:
    """Refuse forbidden /tmp paths per CLAUDE.md non-negotiable."""
    as_str = str(output_dir.resolve())
    forbidden_anchors = ("/tmp/", "/var/tmp/", "/private/tmp/")
    for anchor in forbidden_anchors:
        if as_str.startswith(anchor):
            raise SystemExit(
                f"refusing to write to forbidden /tmp path {output_dir!s} "
                "per CLAUDE.md `forbidden_/tmp_paths_in_any_persisted_artifact`"
            )


def _naive_int8_bytes(arr: np.ndarray) -> int:
    """Naive baseline: dense int8 storage byte count.

    For float arrays we still treat the baseline as ``arr.size`` bytes
    (one int8 per scalar) because the trainer-fresh dense representation
    after FP4/int8 quantisation is exactly that — one byte per scalar in
    the dense form.
    """
    return int(np.prod(arr.shape))


def run_stream(
    spec: DenseStreamSpec,
    *,
    selection_strategy: SelectionStrategy,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Run the magic codec on one synthetic dense stream + return a row."""
    arr = _synthesize_stream(spec, rng=rng)

    naive_bytes = _naive_int8_bytes(arr)
    try:
        result = encode_magic_codec(
            arr,
            hint=StreamHint(stream_type=spec.stream_type),
            selection_strategy=selection_strategy,
        )
    except MagicCodecError as exc:
        return {
            "stream_class": spec.name,
            "description": spec.description,
            "stream_type": spec.stream_type,
            "shape": list(spec.shape),
            "dtype": str(arr.dtype),
            "statistical_signature": spec.statistical_signature,
            "naive_int8_bytes": naive_bytes,
            "magic_codec_refused": True,
            "refusal_reason": str(exc),
            "candidate_primitives": list(
                candidate_primitives_for(spec.stream_type)
            ),
            "recommendation": recommendation_for(spec.stream_type),
        }

    magic_bytes = len(result.payload)
    delta_bytes = magic_bytes - naive_bytes
    savings_ratio = (
        (naive_bytes - magic_bytes) / naive_bytes if naive_bytes > 0 else 0.0
    )
    return {
        "stream_class": spec.name,
        "description": spec.description,
        "stream_type": spec.stream_type,
        "shape": list(spec.shape),
        "dtype": str(arr.dtype),
        "statistical_signature": spec.statistical_signature,
        "naive_int8_bytes": naive_bytes,
        "magic_codec_bytes": magic_bytes,
        "magic_codec_inner_bytes": result.inner_primitive_byte_count,
        "byte_delta_vs_naive": delta_bytes,
        "savings_ratio_vs_naive": savings_ratio,
        "savings_pct_vs_naive": 100.0 * savings_ratio,
        "selected_primitive": result.selected_primitive,
        "selected_primitive_id": result.selected_primitive_id,
        "selection_strategy": result.selection_strategy,
        "candidate_primitives": list(
            candidate_primitives_for(spec.stream_type)
        ),
        "recommendation": recommendation_for(spec.stream_type),
        "selection_log": [
            {
                "primitive_name": c.primitive_name,
                "primitive_id": c.primitive_id,
                "encoded_bytes": len(c.encoded_bytes),
                "refused": c.refused,
                "refusal_reason": c.refusal_reason,
            }
            for c in result.selection_log
        ],
    }


def build_manifest(
    *,
    selection_strategy: SelectionStrategy,
    seed: int,
    operator: str | None,
) -> dict[str, object]:
    """Run every dense-stream spec + return the typed manifest dict."""
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, object]] = []
    for spec in _DENSE_STREAM_SPECS:
        # Use an independent sub-stream per class so adding a new class
        # doesn't perturb prior synthetic outputs (test stability).
        substream_rng = np.random.default_rng(int(seed) ^ hash(spec.name) & 0xFFFFFFFF)
        rows.append(
            run_stream(
                spec,
                selection_strategy=selection_strategy,
                rng=substream_rng,
            )
        )

    aggregate_naive_bytes = sum(
        int(r.get("naive_int8_bytes", 0)) for r in rows
    )
    aggregate_magic_bytes = sum(
        int(r.get("magic_codec_bytes", 0)) for r in rows
    )
    aggregate_savings_ratio = (
        (aggregate_naive_bytes - aggregate_magic_bytes) / aggregate_naive_bytes
        if aggregate_naive_bytes > 0
        else 0.0
    )

    # Per-class win/loss summary for at-a-glance research-signal reporting.
    win_count = sum(
        1
        for r in rows
        if int(r.get("byte_delta_vs_naive", 0)) < 0
    )
    loss_count = sum(
        1
        for r in rows
        if int(r.get("byte_delta_vs_naive", 0)) >= 0
        and not r.get("magic_codec_refused", False)
    )
    refused_count = sum(
        1 for r in rows if r.get("magic_codec_refused", False)
    )

    return {
        "schema": "magic_codec_dense_streams_test.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "operator": operator or "unknown",
        "seed": int(seed),
        "selection_strategy": selection_strategy,
        "stream_specs": [
            {
                "name": s.name,
                "description": s.description,
                "stream_type": s.stream_type,
                "shape": list(s.shape),
                "dtype": s.dtype,
                "statistical_signature": s.statistical_signature,
            }
            for s in _DENSE_STREAM_SPECS
        ],
        "results_by_stream": rows,
        "aggregate": {
            "total_naive_int8_bytes": aggregate_naive_bytes,
            "total_magic_codec_bytes": aggregate_magic_bytes,
            "aggregate_byte_delta": (
                aggregate_magic_bytes - aggregate_naive_bytes
            ),
            "aggregate_savings_ratio": aggregate_savings_ratio,
            "aggregate_savings_pct": 100.0 * aggregate_savings_ratio,
            "n_streams": len(rows),
            "n_win_vs_naive": win_count,
            "n_loss_vs_naive": loss_count,
            "n_refused": refused_count,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "byte_proxy_only": True,
        "cuda_eval_worth_testing": False,
        "blockers": [
            "no_real_trainer_stream_consumed",
            "synthetic_input_byte_proxy_only",
        ],
        "notes": (
            "Per CLAUDE.md `forbidden_premature_KILL_without_research_exhaustion`, "
            "negative-savings rows in this manifest are NOT a verdict that the "
            "magic codec is dominated for that stream class; they are a "
            "byte-proxy signal that the synthetic statistical signature does "
            "not amortise the envelope overhead at this stream-size. A real "
            "trainer-fresh stream may differ. Run on actual trainer output "
            "before any optimize-mode dispatch decision."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.dry_run and args.output_dir is None:
        raise SystemExit(
            "--output-dir is required unless --dry-run is set"
        )

    manifest = build_manifest(
        selection_strategy=args.selection_strategy,
        seed=args.seed,
        operator=args.operator,
    )

    if args.dry_run:
        json.dump(manifest, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    _validate_output_dir(args.output_dir)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "magic_codec_dense_streams_test_manifest.json"
    body = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(body)

    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    print(
        f"wrote {manifest_path} (sha={sha[:8]}, "
        f"streams={manifest['aggregate']['n_streams']}, "
        f"wins={manifest['aggregate']['n_win_vs_naive']}, "
        f"losses={manifest['aggregate']['n_loss_vs_naive']}, "
        f"refused={manifest['aggregate']['n_refused']}, "
        f"aggregate_pct={manifest['aggregate']['aggregate_savings_pct']:+.2f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
