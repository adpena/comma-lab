# SPDX-License-Identifier: MIT
"""Pair #4 local smoke: magic_codec x procedural-codebook seed bytes.

This is the low-cost boundary check from
``.omx/research/magic_codec_x_todays_cascade_stacking_analysis_20260520.md``
section 7, pair #4:

    magic_codec x procedural-codebook seed bytes (16-256B uniform-random)

The hypothesis is intentionally a NULL hypothesis. Procedural-codebook seed
bytes are the canonical representation of the replaced deterministic content.
If a seed is already high-entropy PRNG state, an additional magic-codec layer
should not reduce rate. The correct integration is therefore an external
raw-fallback selector: keep the raw seed whenever every wrapping codec costs
more bytes.

The operator called out ordering as a separate dimension, so this smoke tests
reversible byte-order permutations that a decoder can invert for free from the
known seed length. Value-dependent sorted orders are emitted only as non-free
negative controls because they require storing either the inverse permutation
or a different seed semantic.

This tool does not run the contest scorer and does not claim a score. It is a
byte-budget boundary check for dispatch routing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import brotli
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.packet_compiler.magic_codec import (  # noqa: E402
    MagicCodecError,
    StreamHint,
    encode_magic_codec,
    shannon_entropy_estimate_bits,
)
from tac.packet_compiler.magic_codec_dense_streams import (  # noqa: E402
    DenseStreamInput,
    MagicCodecDenseStreamsError,
    encode_magic_codec_dense_streams,
)


CANONICAL_RATE_MULTIPLIER = 25.0
CANONICAL_RATE_DENOM_BYTES = 37_545_489
PAIR_4_PREDICTED_DELTA_S = 0.0
PAIR_4_MEMO_PATH = ".omx/research/magic_codec_x_todays_cascade_stacking_analysis_20260520.md"

BROTLI_QUALITY = 11
BROTLI_LGWIN = 22
LZMA_PRESET = 9 | lzma.PRESET_EXTREME

SeedClass = Literal["canonical_uniform_seed", "structured_negative_control"]
OrderingCompliance = Literal["reversible_free", "non_free_control"]


@dataclass(frozen=True)
class OrderingSpec:
    name: str
    compliance: OrderingCompliance
    note: str


@dataclass(frozen=True)
class CodecCandidateRow:
    codec_name: str
    byte_count: int
    delta_vs_raw_ordered_bytes: int
    selected_primitive: str | None
    payload_sha256: str | None
    refused: bool
    refusal_reason: str | None
    compliant_for_verdict: bool


ORDERINGS: tuple[OrderingSpec, ...] = (
    OrderingSpec(
        "identity",
        "reversible_free",
        "original seed byte order",
    ),
    OrderingSpec(
        "reverse",
        "reversible_free",
        "fixed reverse order; decoder inverts from seed length",
    ),
    OrderingSpec(
        "even_then_odd",
        "reversible_free",
        "fixed even-index bytes followed by odd-index bytes",
    ),
    OrderingSpec(
        "odd_then_even",
        "reversible_free",
        "fixed odd-index bytes followed by even-index bytes",
    ),
    OrderingSpec(
        "adjacent_pair_swap",
        "reversible_free",
        "fixed swap of each adjacent byte pair",
    ),
    OrderingSpec(
        "rotate_left_half",
        "reversible_free",
        "fixed half-length rotation",
    ),
    OrderingSpec(
        "sorted_ascending",
        "non_free_control",
        "value-dependent sort; excluded because inverse permutation is not free",
    ),
    OrderingSpec(
        "sorted_descending",
        "non_free_control",
        "value-dependent sort; excluded because inverse permutation is not free",
    ),
)

MAGIC_CLASSIC_HINTS: tuple[str, ...] = (
    "latent_sidecar",
    "residual_basis",
    "categorical",
    "weight_tensor",
    "mask",
)


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _utc_now_filename() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def deterministic_seed_bytes(length: int, label: str) -> bytes:
    """Return deterministic high-entropy bytes for a seed-length test case."""
    if length <= 0:
        raise ValueError("length must be positive")
    return hashlib.shake_256(label.encode("utf-8")).digest(length)


def structured_control_seed_bytes(length: int, label: str) -> bytes:
    """Return deterministic low-entropy controls that are not valid seed priors."""
    if label == "all_zero":
        return bytes(length)
    if label == "alternating_00_ff":
        return bytes((0x00 if i % 2 == 0 else 0xFF) for i in range(length))
    if label == "ascending_mod_256":
        return bytes(i % 256 for i in range(length))
    raise ValueError(f"unknown structured control label {label!r}")


def apply_ordering(seed: bytes, ordering_name: str) -> bytes:
    """Apply a byte-ordering transform used by the smoke matrix."""
    if ordering_name == "identity":
        return seed
    if ordering_name == "reverse":
        return seed[::-1]
    if ordering_name == "even_then_odd":
        return seed[0::2] + seed[1::2]
    if ordering_name == "odd_then_even":
        return seed[1::2] + seed[0::2]
    if ordering_name == "adjacent_pair_swap":
        out = bytearray(seed)
        for i in range(0, len(out) - 1, 2):
            out[i], out[i + 1] = out[i + 1], out[i]
        return bytes(out)
    if ordering_name == "rotate_left_half":
        k = len(seed) // 2
        return seed[k:] + seed[:k]
    if ordering_name == "sorted_ascending":
        return bytes(sorted(seed))
    if ordering_name == "sorted_descending":
        return bytes(sorted(seed, reverse=True))
    raise ValueError(f"unknown ordering {ordering_name!r}")


def _candidate_raw(seed: bytes) -> CodecCandidateRow:
    return CodecCandidateRow(
        codec_name="raw_seed",
        byte_count=len(seed),
        delta_vs_raw_ordered_bytes=0,
        selected_primitive=None,
        payload_sha256=_sha256_bytes(seed),
        refused=False,
        refusal_reason=None,
        compliant_for_verdict=True,
    )


def _candidate_brotli(seed: bytes) -> CodecCandidateRow:
    encoded = brotli.compress(seed, quality=BROTLI_QUALITY, lgwin=BROTLI_LGWIN)
    return CodecCandidateRow(
        codec_name="brotli_q11_seed_bytes",
        byte_count=len(encoded),
        delta_vs_raw_ordered_bytes=len(encoded) - len(seed),
        selected_primitive="brotli",
        payload_sha256=_sha256_bytes(encoded),
        refused=False,
        refusal_reason=None,
        compliant_for_verdict=True,
    )


def _candidate_lzma(seed: bytes) -> CodecCandidateRow:
    encoded = lzma.compress(seed, preset=LZMA_PRESET)
    return CodecCandidateRow(
        codec_name="lzma_preset9_extreme_seed_bytes",
        byte_count=len(encoded),
        delta_vs_raw_ordered_bytes=len(encoded) - len(seed),
        selected_primitive="lzma",
        payload_sha256=_sha256_bytes(encoded),
        refused=False,
        refusal_reason=None,
        compliant_for_verdict=True,
    )


def _candidate_magic_classic(seed: bytes, stream_hint: str) -> CodecCandidateRow:
    arr = np.frombuffer(seed, dtype=np.uint8).copy()
    codec_name = f"magic_codec_classic_{stream_hint}"
    try:
        result = encode_magic_codec(arr, hint=StreamHint(stream_hint))  # type: ignore[arg-type]
    except MagicCodecError as exc:
        return CodecCandidateRow(
            codec_name=codec_name,
            byte_count=10**12,
            delta_vs_raw_ordered_bytes=10**12,
            selected_primitive=None,
            payload_sha256=None,
            refused=True,
            refusal_reason=str(exc),
            compliant_for_verdict=False,
        )
    return CodecCandidateRow(
        codec_name=codec_name,
        byte_count=len(result.payload),
        delta_vs_raw_ordered_bytes=len(result.payload) - len(seed),
        selected_primitive=result.selected_primitive,
        payload_sha256=_sha256_bytes(result.payload),
        refused=False,
        refusal_reason=None,
        compliant_for_verdict=True,
    )


def _candidate_dense(seed: bytes, strategy: str) -> CodecCandidateRow:
    arr = np.frombuffer(seed, dtype=np.uint8).copy()
    codec_name = f"magic_codec_dense_streams_{strategy}"
    try:
        result = encode_magic_codec_dense_streams(
            [DenseStreamInput("seed", arr, StreamHint("latent_sidecar"))],
            selection_strategy=strategy,  # type: ignore[arg-type]
        )
    except MagicCodecDenseStreamsError as exc:
        return CodecCandidateRow(
            codec_name=codec_name,
            byte_count=10**12,
            delta_vs_raw_ordered_bytes=10**12,
            selected_primitive=None,
            payload_sha256=None,
            refused=True,
            refusal_reason=str(exc),
            compliant_for_verdict=False,
        )
    selected = ",".join(s.selected_codec_name for s in result.selections)
    return CodecCandidateRow(
        codec_name=codec_name,
        byte_count=len(result.payload),
        delta_vs_raw_ordered_bytes=len(result.payload) - len(seed),
        selected_primitive=selected,
        payload_sha256=_sha256_bytes(result.payload),
        refused=False,
        refusal_reason=None,
        compliant_for_verdict=True,
    )


def evaluate_ordered_seed(seed: bytes) -> dict[str, Any]:
    """Evaluate all pair-4 codec candidates for one ordered seed byte string."""
    candidates: list[CodecCandidateRow] = [
        _candidate_raw(seed),
        _candidate_brotli(seed),
        _candidate_lzma(seed),
    ]
    for hint in MAGIC_CLASSIC_HINTS:
        candidates.append(_candidate_magic_classic(seed, hint))
    for strategy in (
        "smallest_byte_count",
        "brotli_only",
        "lzma_only",
        "magic_classic_only",
    ):
        candidates.append(_candidate_dense(seed, strategy))

    accepted = [c for c in candidates if not c.refused]
    compliant = [c for c in accepted if c.compliant_for_verdict]
    best = min(compliant, key=lambda c: c.byte_count)
    best_nonraw = min(
        (c for c in compliant if c.codec_name != "raw_seed"),
        key=lambda c: c.byte_count,
    )
    arr = np.frombuffer(seed, dtype=np.uint8)
    entropy_bits_per_byte = shannon_entropy_estimate_bits(arr)
    entropy_total_bytes = entropy_bits_per_byte * len(seed) / 8.0
    return {
        "raw_seed_len": len(seed),
        "raw_seed_sha256": _sha256_bytes(seed),
        "entropy_bits_per_byte": entropy_bits_per_byte,
        "entropy_total_bytes_estimate": entropy_total_bytes,
        "best_codec_name": best.codec_name,
        "best_byte_count": best.byte_count,
        "best_delta_vs_raw_ordered_bytes": best.delta_vs_raw_ordered_bytes,
        "best_nonraw_codec_name": best_nonraw.codec_name,
        "best_nonraw_byte_count": best_nonraw.byte_count,
        "best_nonraw_delta_vs_raw_ordered_bytes": (
            best_nonraw.delta_vs_raw_ordered_bytes
        ),
        "raw_seed_dominates": best.codec_name == "raw_seed",
        "candidates": [asdict(c) for c in candidates],
    }


def evaluate_seed_case(
    *,
    seed_bytes: bytes,
    seed_label: str,
    seed_class: SeedClass,
) -> dict[str, Any]:
    """Evaluate one seed across all ordering dimensions."""
    ordering_rows: list[dict[str, Any]] = []
    for ordering in ORDERINGS:
        ordered_seed = apply_ordering(seed_bytes, ordering.name)
        row = evaluate_ordered_seed(ordered_seed)
        row.update(
            {
                "ordering_name": ordering.name,
                "ordering_compliance": ordering.compliance,
                "ordering_note": ordering.note,
                "compliant_for_pair4_verdict": (
                    seed_class == "canonical_uniform_seed"
                    and ordering.compliance == "reversible_free"
                ),
            }
        )
        ordering_rows.append(row)

    verdict_rows = [r for r in ordering_rows if r["compliant_for_pair4_verdict"]]
    if verdict_rows:
        all_raw_dominates: bool | None = all(
            r["raw_seed_dominates"] for r in verdict_rows
        )
        min_nonraw_delta: int | None = min(
            int(r["best_nonraw_delta_vs_raw_ordered_bytes"]) for r in verdict_rows
        )
    else:
        all_raw_dominates = None
        min_nonraw_delta = None
    return {
        "seed_label": seed_label,
        "seed_class": seed_class,
        "seed_len": len(seed_bytes),
        "seed_sha256": _sha256_bytes(seed_bytes),
        "n_ordering_rows": len(ordering_rows),
        "n_reversible_free_orderings": sum(
            1 for r in ordering_rows if r["ordering_compliance"] == "reversible_free"
        ),
        "n_non_free_control_orderings": sum(
            1 for r in ordering_rows if r["ordering_compliance"] == "non_free_control"
        ),
        "all_reversible_free_orderings_raw_seed_dominates": all_raw_dominates,
        "min_reversible_free_best_nonraw_delta_vs_raw_bytes": min_nonraw_delta,
        "ordering_rows": ordering_rows,
    }


def run_smoke(
    *,
    seed_lengths: tuple[int, ...] = (16, 32, 64, 128, 256),
    include_structured_controls: bool = True,
) -> dict[str, Any]:
    """Run the full pair-4 ordering and codec matrix."""
    started = _utc_now_iso()
    cases: list[dict[str, Any]] = []
    for n in seed_lengths:
        seed = deterministic_seed_bytes(n, f"pair4_canonical_uniform_seed_len_{n}")
        cases.append(
            evaluate_seed_case(
                seed_bytes=seed,
                seed_label=f"canonical_uniform_seed_len_{n}",
                seed_class="canonical_uniform_seed",
            )
        )
    if include_structured_controls:
        for n in (32, 64):
            for label in (
                "all_zero",
                "alternating_00_ff",
                "ascending_mod_256",
            ):
                cases.append(
                    evaluate_seed_case(
                        seed_bytes=structured_control_seed_bytes(n, label),
                        seed_label=f"control_{label}_len_{n}",
                        seed_class="structured_negative_control",
                    )
                )

    canonical_cases = [
        c for c in cases if c["seed_class"] == "canonical_uniform_seed"
    ]
    all_canonical_raw_dominates = all(
        c["all_reversible_free_orderings_raw_seed_dominates"]
        for c in canonical_cases
    )
    min_nonraw_delta = min(
        int(c["min_reversible_free_best_nonraw_delta_vs_raw_bytes"])
        for c in canonical_cases
    )
    n_canonical_orderings = sum(
        int(c["n_reversible_free_orderings"]) for c in canonical_cases
    )
    n_canonical_orderings_raw_dominates = sum(
        sum(
            1
            for r in c["ordering_rows"]
            if r["compliant_for_pair4_verdict"] and r["raw_seed_dominates"]
        )
        for c in canonical_cases
    )
    if all_canonical_raw_dominates:
        cascade_verdict = "PAIR_4_BOUNDARY_VALIDATED_RAW_SEED_DOMINATES"
        recommendation = (
            "Keep procedural-codebook seed bytes raw; route magic_codec to residual "
            "streams, not to the seed itself."
        )
    else:
        cascade_verdict = "PAIR_4_COUNTEREXAMPLE_CODEC_SAVES_ON_CANONICAL_SEED"
        recommendation = (
            "Inspect the counterexample before routing; any saving on high-entropy "
            "seed bytes may indicate a non-uniform seed generator or accounting bug."
        )

    completed = _utc_now_iso()
    return {
        "smoke_label": "wave_3_magic_codec_pair_4_procedural_seed_orthogonality_smoke",
        "smoke_lane_id": "lane_wave_3_magic_codec_pair_4_procedural_seed_orthogonality_smoke_20260521",
        "smoke_pair_id": "pair_4_magic_codec_x_procedural_codebook_seed_bytes",
        "source_memo": PAIR_4_MEMO_PATH,
        "started_at_utc": started,
        "completed_at_utc": completed,
        "platform": platform.platform(),
        "seed_lengths": list(seed_lengths),
        "ordering_dimension": {
            "reversible_free_orderings": [
                o.name for o in ORDERINGS if o.compliance == "reversible_free"
            ],
            "non_free_control_orderings": [
                o.name for o in ORDERINGS if o.compliance == "non_free_control"
            ],
            "value_dependent_sort_controls_excluded_from_verdict": True,
        },
        "codec_dimensions": {
            "raw_seed": True,
            "brotli_q11_seed_bytes": True,
            "lzma_preset9_extreme_seed_bytes": True,
            "magic_codec_classic_hints": list(MAGIC_CLASSIC_HINTS),
            "magic_codec_dense_streams_strategies": [
                "smallest_byte_count",
                "brotli_only",
                "lzma_only",
                "magic_classic_only",
            ],
        },
        "n_seed_cases_total": len(cases),
        "n_canonical_seed_cases": len(canonical_cases),
        "n_canonical_reversible_ordering_rows": n_canonical_orderings,
        "n_canonical_reversible_ordering_rows_raw_seed_dominates": (
            n_canonical_orderings_raw_dominates
        ),
        "min_canonical_reversible_best_nonraw_delta_vs_raw_bytes": min_nonraw_delta,
        "all_canonical_reversible_orderings_raw_seed_dominates": (
            all_canonical_raw_dominates
        ),
        "predicted_delta_s_pair_4": PAIR_4_PREDICTED_DELTA_S,
        "empirical_delta_s_pair_4": 0.0 if all_canonical_raw_dominates else None,
        "cascade_verdict": cascade_verdict,
        "recommendation": recommendation,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "axis_tag": "[byte-budget local smoke only]",
        "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
        "evidence_grade": "local_byte_budget_smoke_advisory",
        "cases": cases,
    }


def emit_markdown_report(result: dict[str, Any], md_path: Path) -> None:
    """Emit a compact operator-readable report."""
    lines = [
        "<!-- SPDX-License-Identifier: MIT -->",
        "# WAVE-3 Pair #4: magic_codec x procedural-codebook seed orthogonality smoke",
        "",
        f"**Lane**: `{result['smoke_lane_id']}`  ",
        f"**Pair ID**: `{result['smoke_pair_id']}`  ",
        f"**Source memo**: `{result['source_memo']}`  ",
        f"**Started**: `{result['started_at_utc']}`  ",
        f"**Completed**: `{result['completed_at_utc']}`  ",
        f"**Platform**: `{result['platform']}`  ",
        "",
        "## Verdict",
        "",
        f"* `cascade_verdict`: `{result['cascade_verdict']}`",
        f"* `recommendation`: {result['recommendation']}",
        f"* `predicted_delta_s_pair_4`: `{result['predicted_delta_s_pair_4']}`",
        f"* `empirical_delta_s_pair_4`: `{result['empirical_delta_s_pair_4']}`",
        f"* `score_claim_valid`: `{result['score_claim_valid']}`",
        f"* `promotion_eligible`: `{result['promotion_eligible']}`",
        "",
        "## Dimensions",
        "",
        f"* Seed lengths: `{result['seed_lengths']}`",
        f"* Reversible free orderings: `{result['ordering_dimension']['reversible_free_orderings']}`",
        f"* Non-free controls: `{result['ordering_dimension']['non_free_control_orderings']}`",
        f"* Magic classic hints: `{result['codec_dimensions']['magic_codec_classic_hints']}`",
        f"* Dense-stream strategies: `{result['codec_dimensions']['magic_codec_dense_streams_strategies']}`",
        "",
        "## Canonical seed summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| canonical seed cases | {result['n_canonical_seed_cases']} |",
        f"| canonical reversible ordering rows | {result['n_canonical_reversible_ordering_rows']} |",
        f"| rows where raw seed dominates | {result['n_canonical_reversible_ordering_rows_raw_seed_dominates']} |",
        f"| min best-nonraw delta vs raw | {result['min_canonical_reversible_best_nonraw_delta_vs_raw_bytes']} bytes |",
        "",
        "## Per-case best codecs",
        "",
        "| Seed case | Class | Len | Ordering | Order compliance | Best codec | Best nonraw codec | Best nonraw delta vs raw |",
        "|---|---|---:|---|---|---|---|---:|",
    ]
    for case in result["cases"]:
        for row in case["ordering_rows"]:
            lines.append(
                "| "
                f"{case['seed_label']} | "
                f"{case['seed_class']} | "
                f"{case['seed_len']} | "
                f"{row['ordering_name']} | "
                f"{row['ordering_compliance']} | "
                f"{row['best_codec_name']} | "
                f"{row['best_nonraw_codec_name']} | "
                f"{row['best_nonraw_delta_vs_raw_ordered_bytes']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "For canonical high-entropy seed bytes, the raw seed is the rate floor in every reversible ordering tested. "
            "The magic-codec envelope is still useful for residual streams, but on the seed itself it should be "
            "declined by an external raw-fallback selector.",
            "",
            "Value-dependent sorted controls are excluded from the verdict because the inverse permutation is not free. "
            "Structured low-entropy controls are included only to prove the smoke can detect compressible non-seed data.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pair #4 procedural seed orthogonality smoke across "
            "seed length, ordering, codec, and control dimensions."
        )
    )
    parser.add_argument(
        "--seed-lengths",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256],
        help="Canonical uniform seed lengths to test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for smoke_result.json and smoke_result.md.",
    )
    parser.add_argument(
        "--no-structured-controls",
        action="store_true",
        help="Skip low-entropy negative controls.",
    )
    args = parser.parse_args(argv)

    if any(n <= 0 for n in args.seed_lengths):
        print("seed lengths must be positive", file=sys.stderr)
        return 2
    seed_lengths = tuple(int(n) for n in args.seed_lengths)
    result = run_smoke(
        seed_lengths=seed_lengths,
        include_structured_controls=not args.no_structured_controls,
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            REPO_ROOT
            / "experiments"
            / "results"
            / f"magic_codec_pair_4_procedural_seed_orthogonality_smoke_{_utc_now_filename()}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "smoke_result.json"
    md_path = output_dir / "smoke_result.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    emit_markdown_report(result, md_path)
    print(
        "pair4_seed_orthogonality_smoke "
        f"verdict={result['cascade_verdict']} "
        f"canonical_rows={result['n_canonical_reversible_ordering_rows']} "
        f"raw_dominates={result['n_canonical_reversible_ordering_rows_raw_seed_dominates']} "
        f"min_nonraw_delta={result['min_canonical_reversible_best_nonraw_delta_vs_raw_bytes']} "
        f"json={json_path} md={md_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
