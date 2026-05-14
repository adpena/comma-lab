# SPDX-License-Identifier: MIT
"""PR103 ``hnerv_lc_ac`` arithmetic-schema custody helpers.

This module profiles the public PR103 range-coded payload as a codec/custody
surface. It does not build a contest archive, dispatch work, or claim score.
The useful property is stricter than a byte profile: the merged constriction
queue stream can be decoded and re-encoded byte-identically against the public
histogram models, which gives later PR101/PR103 arithmetic-schema work a
fail-closed source manifest instead of prose.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import brotli
import numpy as np

from tac.hnerv_lowlevel_packer import read_strict_single_member_zip
from tac.repo_io import read_json, repo_relative, sha256_bytes, sha256_file

try:  # pragma: no cover - optional outside the contest venv
    import constriction
except ImportError:  # pragma: no cover
    constriction = None  # type: ignore[assignment]

SCHEMA_VERSION = 1
TOOL_NAME = "tac.hnerv_pr103_lc_ac_schema"
CONTEST_ORIGINAL_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / CONTEST_ORIGINAL_BYTES

N_PAIRS = 600
LATENT_DIM = 28
HI_SYMBOL_COUNT = N_PAIRS * LATENT_DIM

SCHEMA: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("stem.weight", (1728, 28)),
    ("stem.bias", (1728,)),
    ("blocks.0.weight", (144, 36, 3, 3)),
    ("blocks.0.bias", (144,)),
    ("blocks.1.weight", (144, 36, 3, 3)),
    ("blocks.1.bias", (144,)),
    ("blocks.2.weight", (108, 36, 3, 3)),
    ("blocks.2.bias", (108,)),
    ("blocks.3.weight", (80, 27, 3, 3)),
    ("blocks.3.bias", (80,)),
    ("blocks.4.weight", (72, 20, 3, 3)),
    ("blocks.4.bias", (72,)),
    ("blocks.5.weight", (72, 18, 3, 3)),
    ("blocks.5.bias", (72,)),
    ("skips.2.weight", (27, 36, 1, 1)),
    ("skips.2.bias", (27,)),
    ("skips.3.weight", (20, 27, 1, 1)),
    ("skips.3.bias", (20,)),
    ("skips.4.weight", (18, 20, 1, 1)),
    ("skips.4.bias", (18,)),
    ("refine.0.weight", (9, 18, 3, 3)),
    ("refine.0.bias", (9,)),
    ("refine.1.weight", (18, 9, 3, 3)),
    ("refine.1.bias", (18,)),
    ("rgb_0.weight", (3, 18, 3, 3)),
    ("rgb_0.bias", (3,)),
    ("rgb_1.weight", (3, 18, 3, 3)),
    ("rgb_1.bias", (3,)),
)
AC_INDICES = (0, 2, 4, 6, 8, 10, 12, 21)
AC_STREAM_SPECS = tuple(
    (SCHEMA[index][0], math.prod(SCHEMA[index][1]), index)
    for index in AC_INDICES
)


@dataclasses.dataclass(frozen=True)
class Pr103LcAcLayout:
    """Fixed section lengths from the public PR103 runtime."""

    scales_fp16: int = 56
    non_ac_weights_brotli: int = 7_097
    ac_histograms_brotli: int = 895
    merged_range_coded_weights_and_hi_latents: int = 153_856
    latent_min_scale_fp16: int = 112
    latent_low_bytes_brotli: int = 15_537
    latent_hi_histogram_brotli: int = 15

    def section_specs(self) -> tuple[tuple[str, int], ...]:
        return (
            ("scales_fp16", self.scales_fp16),
            ("non_ac_weights_brotli", self.non_ac_weights_brotli),
            ("ac_histograms_brotli", self.ac_histograms_brotli),
            (
                "merged_range_coded_weights_and_hi_latents",
                self.merged_range_coded_weights_and_hi_latents,
            ),
            ("latent_min_scale_fp16", self.latent_min_scale_fp16),
            ("latent_low_bytes_brotli", self.latent_low_bytes_brotli),
            ("latent_hi_histogram_brotli", self.latent_hi_histogram_brotli),
        )

    @property
    def fixed_bytes(self) -> int:
        return sum(size for _name, size in self.section_specs())


PUBLIC_PR103_LAYOUT = Pr103LcAcLayout()


class HnervPr103LcAcSchemaError(ValueError):
    """Raised when PR103 lc_ac custody input is malformed."""


@dataclasses.dataclass(frozen=True)
class Pr103LcAcSection:
    """One byte-addressed section inside the PR103 payload member."""

    name: str
    start: int
    end: int
    data: bytes

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "bytes": len(self.data),
            "sha256": sha256_bytes(self.data),
        }


@dataclasses.dataclass(frozen=True)
class Pr103LcAcPayload:
    """Parsed public PR103 payload sections."""

    sections: tuple[Pr103LcAcSection, ...]

    def section_bytes(self, name: str) -> bytes:
        for section in self.sections:
            if section.name == name:
                return section.data
        raise HnervPr103LcAcSchemaError(f"unknown PR103 lc_ac section: {name}")

    def section_records(self) -> list[dict[str, Any]]:
        return [section.record() for section in self.sections]


def parse_pr103_lc_ac_payload(
    payload: bytes,
    *,
    layout: Pr103LcAcLayout = PUBLIC_PR103_LAYOUT,
) -> Pr103LcAcPayload:
    """Parse the fixed PR103 lc_ac wire layout and fail closed on truncation."""

    if len(payload) < layout.fixed_bytes:
        raise HnervPr103LcAcSchemaError(
            f"PR103 lc_ac payload truncated: expected at least {layout.fixed_bytes}B, "
            f"got {len(payload)}B"
        )
    cursor = 0
    sections: list[Pr103LcAcSection] = []
    for name, size in layout.section_specs():
        if size < 0:
            raise HnervPr103LcAcSchemaError(f"negative section length for {name}: {size}")
        end = cursor + size
        if end > len(payload):
            raise HnervPr103LcAcSchemaError(
                f"section {name} exceeds payload length: end={end} len={len(payload)}"
            )
        sections.append(Pr103LcAcSection(name=name, start=cursor, end=end, data=payload[cursor:end]))
        cursor = end
    sections.append(
        Pr103LcAcSection(
            name="sidecar_corrections_brotli",
            start=cursor,
            end=len(payload),
            data=payload[cursor:],
        )
    )
    return Pr103LcAcPayload(sections=tuple(sections))


def decode_pr103_auxiliary_models(
    parsed: Pr103LcAcPayload,
    *,
    ac_stream_count: int = len(AC_STREAM_SPECS),
) -> dict[str, Any]:
    """Decode non-merged PR103 model sections into deterministic summaries."""

    non_ac_raw = _brotli_decompress_section(parsed, "non_ac_weights_brotli")
    hist_raw = _brotli_decompress_section(parsed, "ac_histograms_brotli")
    lo_raw = _brotli_decompress_section(parsed, "latent_low_bytes_brotli")
    hi_hist_raw = _brotli_decompress_section(parsed, "latent_hi_histogram_brotli")
    sidecar = parsed.section_bytes("sidecar_corrections_brotli")
    sidecar_raw = brotli.decompress(sidecar) if sidecar else b""

    if ac_stream_count <= 0:
        raise HnervPr103LcAcSchemaError("AC stream count must be positive")
    if len(hist_raw) % ac_stream_count != 0:
        raise HnervPr103LcAcSchemaError(
            "ac_histograms_brotli decoded bytes are not divisible by AC stream count"
        )
    symbol_count = len(hist_raw) // ac_stream_count
    histograms = np.frombuffer(hist_raw, dtype=np.uint8).reshape(ac_stream_count, symbol_count)
    if len(hi_hist_raw) % 2:
        raise HnervPr103LcAcSchemaError("latent_hi_histogram_brotli decoded odd byte count")
    hi_histogram = np.frombuffer(hi_hist_raw, dtype="<u2")

    return {
        "non_ac_weights_raw": _raw_record("non_ac_weights_raw", non_ac_raw),
        "ac_histograms": {
            "decoded_bytes": len(hist_raw),
            "sha256": sha256_bytes(hist_raw),
            "stream_count": int(histograms.shape[0]),
            "symbol_count": int(histograms.shape[1]),
            "row_weight_sums": [int(value) for value in histograms.sum(axis=1).tolist()],
        },
        "latent_low_bytes_raw": _raw_record("latent_low_bytes_raw", lo_raw),
        "latent_hi_histogram": {
            "decoded_bytes": len(hi_hist_raw),
            "sha256": sha256_bytes(hi_hist_raw),
            "symbol_count": int(hi_histogram.size),
            "weight_sum": int(hi_histogram.astype(np.uint64).sum()),
        },
        "sidecar_corrections_raw": _raw_record("sidecar_corrections_raw", sidecar_raw),
        "_histograms_array": histograms,
        "_hi_histogram_array": hi_histogram,
    }


def decode_pr103_merged_ac_stream(
    merged_ac: bytes,
    histograms: np.ndarray,
    hi_histogram: np.ndarray,
    *,
    stream_specs: Sequence[tuple[str, int, int | None]] = AC_STREAM_SPECS,
    hi_symbol_count: int = HI_SYMBOL_COUNT,
) -> dict[str, Any]:
    """Decode and byte-parity re-encode the merged constriction queue stream."""

    _require_constriction()
    if len(merged_ac) == 0 or len(merged_ac) % 4:
        raise HnervPr103LcAcSchemaError(
            "merged_range_coded_weights_and_hi_latents must be non-empty uint32 words"
        )
    hist = np.asarray(histograms)
    if hist.ndim != 2 or hist.shape[0] != len(stream_specs):
        raise HnervPr103LcAcSchemaError(
            "histogram rows must match AC stream specs: "
            f"rows={hist.shape[0] if hist.ndim == 2 else 'bad'} specs={len(stream_specs)}"
        )
    hi_hist = np.asarray(hi_histogram)
    if hi_hist.ndim != 1 or hi_hist.size == 0:
        raise HnervPr103LcAcSchemaError("latent hi histogram must be a non-empty 1D array")

    decoder = constriction.stream.queue.RangeDecoder(np.frombuffer(merged_ac, dtype="<u4"))
    stream_rows: list[dict[str, Any]] = []
    decoded_streams: list[np.ndarray] = []
    for row_index, (label, count, schema_index) in enumerate(stream_specs):
        symbols = _decode_constriction_symbols(decoder, hist[row_index], int(count))
        decoded_streams.append(symbols)
        stream_rows.append(
            _stream_entropy_record(
                label=label,
                role="ac_weight_tensor",
                schema_index=schema_index,
                symbols=symbols,
                weights=hist[row_index],
            )
        )
    hi_symbols = _decode_constriction_symbols(decoder, hi_hist, int(hi_symbol_count))
    stream_rows.append(
        _stream_entropy_record(
            label="latent_hi_bytes",
            role="latent_hi_stream",
            schema_index=None,
            symbols=hi_symbols,
            weights=hi_hist,
        )
    )
    maybe_exhausted = bool(decoder.maybe_exhausted())

    reencoded = encode_pr103_merged_ac_stream(
        [*decoded_streams, hi_symbols],
        [*list(hist), hi_hist],
    )
    byte_identical = reencoded == merged_ac
    return {
        "codec": "constriction.stream.queue.RangeEncoder_uint32_words",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "source_bytes": len(merged_ac),
        "source_sha256": sha256_bytes(merged_ac),
        "uint32_word_count": len(merged_ac) // 4,
        "decoded_symbol_count": int(sum(int(row["symbol_count"]) for row in stream_rows)),
        "decoder_maybe_exhausted": maybe_exhausted,
        "reencoded_bytes": len(reencoded),
        "reencoded_sha256": sha256_bytes(reencoded),
        "reencoded_byte_identical": byte_identical,
        "stream_rows": stream_rows,
        "stream_gap_ranking": sorted(
            stream_rows,
            key=lambda row: (-float(row["model_gap_bytes_estimate"]), str(row["label"])),
        ),
        "blockers": _merged_stream_blockers(maybe_exhausted, byte_identical),
    }


def encode_pr103_merged_ac_stream(
    symbol_streams: Sequence[np.ndarray],
    weights: Sequence[np.ndarray],
) -> bytes:
    """Encode streams with the PR103 constriction categorical contract."""

    _require_constriction()
    if len(symbol_streams) != len(weights):
        raise HnervPr103LcAcSchemaError("symbol stream count must match model weight count")
    encoder = constriction.stream.queue.RangeEncoder()
    for symbols, model_weights in zip(symbol_streams, weights, strict=True):
        cat = _categorical_from_weights(model_weights)
        for symbol in np.asarray(symbols, dtype=np.int64).reshape(-1).tolist():
            encoder.encode(int(symbol), cat)
    return encoder.get_compressed().astype(np.uint32).tobytes()


def build_pr103_lc_ac_schema_manifest(
    *,
    source_archive: str | Path,
    source_label: str,
    exact_adjudication_log: str | Path | None = None,
    replay_fidelity_json: str | Path | None = None,
    candidate_archive: str | Path | None = None,
    repo_root: str | Path | None = None,
    layout: Pr103LcAcLayout = PUBLIC_PR103_LAYOUT,
    stream_specs: Sequence[tuple[str, int, int | None]] = AC_STREAM_SPECS,
    hi_symbol_count: int = HI_SYMBOL_COUNT,
) -> dict[str, Any]:
    """Build a fail-closed PR103 arithmetic-schema source manifest."""

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    archive_path = Path(source_archive)
    source = read_strict_single_member_zip(archive_path)
    parsed = parse_pr103_lc_ac_payload(source.payload, layout=layout)
    auxiliary = decode_pr103_auxiliary_models(parsed, ac_stream_count=len(stream_specs))
    histograms = auxiliary.pop("_histograms_array")
    hi_histogram = auxiliary.pop("_hi_histogram_array")
    merged = decode_pr103_merged_ac_stream(
        parsed.section_bytes("merged_range_coded_weights_and_hi_latents"),
        histograms,
        hi_histogram,
        stream_specs=stream_specs,
        hi_symbol_count=hi_symbol_count,
    )
    adjudication = _load_adjudication_record(exact_adjudication_log, repo)
    replay_fidelity = _load_replay_fidelity(replay_fidelity_json, repo)
    candidate = _candidate_archive_record(
        candidate_archive,
        repo,
        source=source,
        layout=layout,
        ac_stream_count=len(stream_specs),
    )

    source_exact_blockers = _source_exact_eval_blockers(
        source_archive=source.archive_sha256,
        source_archive_bytes=source.archive_bytes,
        adjudication=adjudication,
    )
    replay_blockers = list(replay_fidelity.get("blockers") or [])
    archive_pair_closed = candidate.get("archive_sha256_differs_from_source") is True
    schema_blockers = list(merged["blockers"])
    runtime_adapter_classification = _runtime_adapter_classification(
        merged=merged,
        candidate=candidate,
    )
    readiness_blockers = _unique_ordered(
        [
            *schema_blockers,
            *source_exact_blockers,
            *(f"replay_fidelity:{item}" for item in replay_blockers),
            *([] if candidate.get("provided") else ["candidate_archive_missing"]),
            *([] if archive_pair_closed else ["old_new_archive_sha256_pair_missing"]),
            *runtime_adapter_classification["blockers"],
            "strict_pre_submission_compliance_json_missing",
            "lane_dispatch_claim_missing",
            "exact_cuda_auth_eval_missing",
        ]
    )
    ready_for_schema_review = not schema_blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "planning_only": True,
        "score_claim": False,
        "score_evidence_grade": "invalid",
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_schema_review": ready_for_schema_review,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "source_label": source_label,
        "source_archive": {
            "path": repo_relative(archive_path, repo),
            "bytes": source.archive_bytes,
            "sha256": source.archive_sha256,
            "member_name": source.member_name,
            "member_bytes": source.member_bytes,
            "member_sha256": sha256_bytes(source.payload),
            "zip_overhead_bytes": source.archive_bytes - source.member_bytes,
        },
        "payload_sections": parsed.section_records(),
        "auxiliary_models": auxiliary,
        "merged_arithmetic_stream": merged,
        "next_arithmetic_schema_targets": _next_targets(merged),
        "source_exact_adjudication": adjudication,
        "replay_fidelity": replay_fidelity,
        "candidate_archive": candidate,
        "runtime_adapter_classification": runtime_adapter_classification,
        "old_new_archive_sha256_pair": {
            "old": source.archive_sha256,
            "new": str(candidate.get("sha256") or ""),
            "closed": archive_pair_closed,
        },
        "rate_score_per_byte": RATE_SCORE_PER_BYTE,
        "readiness_blockers": readiness_blockers,
        "dispatch_blockers": [
            "pr103_lc_ac_schema_manifest_is_not_dispatch_authorization",
            *readiness_blockers,
        ],
    }


def render_markdown(manifest: Mapping[str, Any]) -> str:
    """Render a compact PR103 arithmetic-schema readiness note."""

    source = _as_mapping(manifest.get("source_archive"))
    merged = _as_mapping(manifest.get("merged_arithmetic_stream"))
    pair = _as_mapping(manifest.get("old_new_archive_sha256_pair"))
    lines = [
        "# PR103 lc_ac Arithmetic Schema Manifest",
        "",
        f"- planning_only: `{_bool(manifest.get('planning_only') is True)}`",
        f"- score_claim: `{_bool(manifest.get('score_claim') is True)}`",
        f"- dispatch_attempted: `{_bool(manifest.get('dispatch_attempted') is True)}`",
        f"- ready_for_exact_eval_dispatch: `{_bool(manifest.get('ready_for_exact_eval_dispatch') is True)}`",
        f"- ready_for_schema_review: `{_bool(manifest.get('ready_for_schema_review') is True)}`",
        "",
        "## Source Archive",
        "",
        f"- path: `{source.get('path')}`",
        f"- bytes: `{source.get('bytes')}`",
        f"- sha256: `{source.get('sha256')}`",
        f"- member: `{source.get('member_name')}` / `{source.get('member_bytes')}` bytes",
        "",
        "## Merged Arithmetic Stream",
        "",
        f"- bytes: `{merged.get('source_bytes')}`",
        f"- sha256: `{merged.get('source_sha256')}`",
        f"- decoded_symbols: `{merged.get('decoded_symbol_count')}`",
        f"- decoder_maybe_exhausted: `{_bool(merged.get('decoder_maybe_exhausted') is True)}`",
        f"- reencoded_byte_identical: `{_bool(merged.get('reencoded_byte_identical') is True)}`",
        "",
        "## Next Targets",
        "",
        "| rank | stream | role | symbols | model gap bytes | model floor bytes |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for index, row in enumerate(manifest.get("next_arithmetic_schema_targets") or [], start=1):
        item = _as_mapping(row)
        lines.append(
            "| {rank} | `{label}` | `{role}` | {symbols} | {gap} | {floor} |".format(
                rank=index,
                label=item.get("label"),
                role=item.get("role"),
                symbols=item.get("symbol_count"),
                gap=item.get("model_gap_bytes_estimate"),
                floor=item.get("model_cross_entropy_bytes_floor"),
            )
        )
    lines.extend(
        [
            "",
            "## Old/New Archive Pair",
            "",
            f"- old: `{pair.get('old')}`",
            f"- new: `{pair.get('new')}`",
            f"- closed: `{_bool(pair.get('closed') is True)}`",
            "",
            "## Blockers",
            "",
        ]
    )
    for blocker in manifest.get("readiness_blockers") or []:
        lines.append(f"- `{blocker}`")
    lines.append("")
    return "\n".join(lines)


def _decode_constriction_symbols(decoder: Any, weights: np.ndarray, count: int) -> np.ndarray:
    if count < 0:
        raise HnervPr103LcAcSchemaError(f"negative decode count: {count}")
    cat = _categorical_from_weights(weights)
    out = np.empty(int(count), dtype=np.int32)
    for index in range(int(count)):
        out[index] = decoder.decode(cat)
    return out


def _categorical_from_weights(weights: np.ndarray) -> Any:
    _require_constriction()
    probs = np.asarray(weights, dtype=np.float64).reshape(-1)
    if probs.size == 0:
        raise HnervPr103LcAcSchemaError("categorical weights must be non-empty")
    if np.any(probs < 0) or not np.all(np.isfinite(probs)):
        raise HnervPr103LcAcSchemaError("categorical weights must be finite and non-negative")
    probs = np.maximum(probs, 1e-10)
    probs = probs / probs.sum()
    return constriction.stream.model.Categorical(probs, perfect=False)


def _stream_entropy_record(
    *,
    label: str,
    role: str,
    schema_index: int | None,
    symbols: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    flat = np.asarray(symbols, dtype=np.int64).reshape(-1)
    if flat.size == 0:
        raise HnervPr103LcAcSchemaError(f"stream {label} decoded no symbols")
    model = np.asarray(weights, dtype=np.float64).reshape(-1)
    model = np.maximum(model, 1e-10)
    model = model / model.sum()
    if int(flat.min()) < 0 or int(flat.max()) >= model.size:
        raise HnervPr103LcAcSchemaError(
            f"stream {label} decoded symbol outside model alphabet"
        )
    counts = np.bincount(flat, minlength=model.size).astype(np.int64)
    entropy_bits = _entropy_bits_from_counts(counts)
    cross_entropy_bits = float(-np.sum(np.log2(model[flat])))
    model_gap_bits = max(0.0, cross_entropy_bits - entropy_bits)
    return {
        "label": label,
        "role": role,
        "schema_index": schema_index,
        "symbol_count": int(flat.size),
        "alphabet_size": int(model.size),
        "observed_symbol_count": int(np.count_nonzero(counts)),
        "model_weight_sum": int(np.asarray(weights, dtype=np.uint64).sum()),
        "decoded_symbols_sha256": sha256_bytes(flat.astype(np.uint16).tobytes()),
        "observed_entropy_bits_per_symbol": entropy_bits / float(flat.size),
        "model_cross_entropy_bits_per_symbol": cross_entropy_bits / float(flat.size),
        "observed_entropy_bytes_floor": math.ceil(entropy_bits / 8.0),
        "model_cross_entropy_bytes_floor": math.ceil(cross_entropy_bits / 8.0),
        "model_gap_bytes_estimate": math.ceil(model_gap_bits / 8.0),
        "score_claim": False,
        "dispatch_attempted": False,
    }


def _entropy_bits_from_counts(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total <= 0:
        raise HnervPr103LcAcSchemaError("cannot compute entropy of empty counts")
    probs = counts[counts > 0].astype(np.float64) / float(total)
    return float(-np.sum(probs * np.log2(probs)) * total)


def _raw_record(name: str, data: bytes) -> dict[str, Any]:
    return {"name": name, "bytes": len(data), "sha256": sha256_bytes(data)}


def _brotli_decompress_section(parsed: Pr103LcAcPayload, section_name: str) -> bytes:
    try:
        return brotli.decompress(parsed.section_bytes(section_name))
    except brotli.error as exc:
        raise HnervPr103LcAcSchemaError(
            f"PR103 lc_ac section {section_name} is not brotli-decompressible"
        ) from exc


def _load_adjudication_record(path: str | Path | None, repo: Path) -> dict[str, Any]:
    if path is None:
        return {
            "provided": False,
            "score_claim": False,
            "dispatch_attempted": False,
            "ready_for_exact_eval_dispatch": False,
        }
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    payload: dict[str, Any] | None = None
    for line in text.splitlines():
        if line.startswith("ADJUDICATION_JSON:"):
            payload = json.loads(line.split("ADJUDICATION_JSON:", 1)[1].strip())
            break
    if payload is None:
        raise HnervPr103LcAcSchemaError(f"adjudication log has no ADJUDICATION_JSON: {path}")
    return {
        "provided": True,
        "path": repo_relative(source, repo),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "archive_bytes": payload.get("archive_bytes"),
        "archive_sha256": payload.get("archive_sha256"),
        "score_recomputed": payload.get("score_recomputed"),
        "avg_posenet_dist": payload.get("avg_posenet_dist"),
        "avg_segnet_dist": payload.get("avg_segnet_dist"),
        "evidence_grade": payload.get("evidence_grade"),
        "gpu_model": payload.get("gpu_model"),
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _load_replay_fidelity(path: str | Path | None, repo: Path) -> dict[str, Any]:
    if path is None:
        return {
            "provided": False,
            "score_claim": False,
            "dispatch_attempted": False,
            "ready_for_exact_eval_dispatch": False,
            "blockers": ["replay_fidelity_json_missing"],
        }
    source = Path(path)
    payload = read_json(source)
    if not isinstance(payload, Mapping):
        raise HnervPr103LcAcSchemaError("replay fidelity JSON must be an object")
    return {
        "provided": True,
        "path": repo_relative(source, repo),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "replay_fidelity_closed": payload.get("replay_fidelity_closed"),
        "blockers": list(payload.get("blockers") or []),
        "summary": payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {},
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _candidate_archive_record(
    path: str | Path | None,
    repo: Path,
    *,
    source: Any,
    layout: Pr103LcAcLayout,
    ac_stream_count: int,
) -> dict[str, Any]:
    if path is None:
        return {
            "provided": False,
            "path": "",
            "bytes": None,
            "sha256": "",
            "score_claim": False,
            "dispatch_attempted": False,
            "ready_for_exact_eval_dispatch": False,
        }
    candidate_path = Path(path)
    archive = read_strict_single_member_zip(candidate_path)
    compatibility = _candidate_lc_ac_compatibility(
        archive.payload,
        layout=layout,
        ac_stream_count=ac_stream_count,
    )
    section_diffs: list[dict[str, Any]] = []
    if compatibility["payload_lc_ac_schema_compatible"]:
        candidate_sections = {
            row["name"]: row
            for row in parse_pr103_lc_ac_payload(archive.payload, layout=layout).section_records()
        }
        source_sections = {
            row["name"]: row
            for row in parse_pr103_lc_ac_payload(source.payload, layout=layout).section_records()
        }
        for name, candidate_row in candidate_sections.items():
            source_row = source_sections.get(name)
            changed = source_row is None or (
                source_row.get("sha256") != candidate_row.get("sha256")
                or source_row.get("bytes") != candidate_row.get("bytes")
            )
            if changed:
                section_diffs.append(
                    {
                        "name": name,
                        "source_bytes": source_row.get("bytes") if source_row else None,
                        "candidate_bytes": candidate_row.get("bytes"),
                        "source_sha256": source_row.get("sha256") if source_row else "",
                        "candidate_sha256": candidate_row.get("sha256"),
                    }
                )
        for name, source_row in source_sections.items():
            if name not in candidate_sections:
                section_diffs.append(
                    {
                        "name": name,
                        "source_bytes": source_row.get("bytes"),
                        "candidate_bytes": None,
                        "source_sha256": source_row.get("sha256"),
                        "candidate_sha256": "",
                    }
                )
    source_payload_sha = sha256_bytes(source.payload)
    source_archive_bytes = int(source.archive_bytes)
    source_member_bytes = int(source.member_bytes)
    archive_changed = archive.archive_sha256 != source.archive_sha256
    member_changed = sha256_bytes(archive.payload) != source_payload_sha
    return {
        "provided": True,
        "path": repo_relative(candidate_path, repo),
        "bytes": archive.archive_bytes,
        "sha256": archive.archive_sha256,
        "member_name": archive.member_name,
        "member_bytes": archive.member_bytes,
        "member_sha256": sha256_bytes(archive.payload),
        "zip_overhead_bytes": archive.archive_bytes - archive.member_bytes,
        "archive_sha256_differs_from_source": archive_changed,
        "member_sha256_differs_from_source": member_changed,
        "archive_byte_delta_vs_source": archive.archive_bytes - source_archive_bytes,
        "member_byte_delta_vs_source": archive.member_bytes - source_member_bytes,
        "section_diffs_vs_source": section_diffs,
        **compatibility,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _candidate_lc_ac_compatibility(
    payload: bytes,
    *,
    layout: Pr103LcAcLayout,
    ac_stream_count: int,
) -> dict[str, Any]:
    try:
        parsed = parse_pr103_lc_ac_payload(payload, layout=layout)
        auxiliary = decode_pr103_auxiliary_models(parsed, ac_stream_count=ac_stream_count)
    except (HnervPr103LcAcSchemaError, brotli.error) as exc:
        return {
            "payload_lc_ac_schema_compatible": False,
            "payload_lc_ac_schema_error": str(exc),
            "payload_sections": [],
            "auxiliary_model_decode": {},
        }
    auxiliary.pop("_histograms_array", None)
    auxiliary.pop("_hi_histogram_array", None)
    return {
        "payload_lc_ac_schema_compatible": True,
        "payload_lc_ac_schema_error": "",
        "payload_sections": parsed.section_records(),
        "auxiliary_model_decode": auxiliary,
    }


def _runtime_adapter_classification(
    *,
    merged: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_provided = candidate.get("provided") is True
    archive_changed = candidate.get("archive_sha256_differs_from_source") is True
    member_changed = candidate.get("member_sha256_differs_from_source") is True
    schema_compatible = candidate.get("payload_lc_ac_schema_compatible") is True
    source_schema_closed = not list(merged.get("blockers") or [])
    blockers = []
    if not candidate_provided:
        blockers.append("candidate_runtime_adapter_missing")
    else:
        if not archive_changed or not member_changed:
            blockers.append("candidate_byte_difference_missing")
        if not schema_compatible:
            blockers.append("candidate_lc_ac_schema_compatibility_missing")
        blockers.extend(
            [
                "candidate_runtime_adapter_missing",
                "candidate_inflate_output_parity_missing",
            ]
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": "pr103_lc_ac_runtime_adapter_classification_v1",
        "score_claim": False,
        "dispatch_attempted": False,
        "source_schema_contract_closed": source_schema_closed,
        "byte_different_candidate_provided": candidate_provided and archive_changed and member_changed,
        "candidate_lc_ac_schema_compatible": schema_compatible,
        "candidate_runtime_adapter_integrated": False,
        "candidate_inflate_output_parity_present": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
    }


def _source_exact_eval_blockers(
    *,
    source_archive: str,
    source_archive_bytes: int,
    adjudication: Mapping[str, Any],
) -> list[str]:
    if adjudication.get("provided") is not True:
        return ["source_exact_cuda_adjudication_missing"]
    blockers = []
    if adjudication.get("archive_sha256") != source_archive:
        blockers.append("source_exact_cuda_archive_sha256_mismatch")
    if adjudication.get("archive_bytes") != source_archive_bytes:
        blockers.append("source_exact_cuda_archive_bytes_mismatch")
    return blockers


def _next_targets(merged: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in merged.get("stream_gap_ranking") or []:
        item = dict(row)
        item["required_next_artifact"] = (
            "byte_different_archive_manifest_with_old_new_sha256_runtime_parity_and_exact_cuda"
        )
        item["dispatch_allowed"] = False
        rows.append(item)
    return rows[:5]


def _merged_stream_blockers(maybe_exhausted: bool, byte_identical: bool) -> list[str]:
    blockers = []
    if not maybe_exhausted:
        blockers.append("merged_range_decoder_not_exhausted")
    if not byte_identical:
        blockers.append("merged_range_reencode_not_byte_identical")
    return blockers


def _require_constriction() -> None:
    if constriction is None:  # pragma: no cover
        raise HnervPr103LcAcSchemaError("constriction_missing_for_pr103_lc_ac")


def _unique_ordered(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items if str(item)))


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bool(value: bool) -> str:
    return "true" if value else "false"


__all__ = [
    "AC_STREAM_SPECS",
    "PUBLIC_PR103_LAYOUT",
    "HnervPr103LcAcSchemaError",
    "Pr103LcAcLayout",
    "build_pr103_lc_ac_schema_manifest",
    "decode_pr103_auxiliary_models",
    "decode_pr103_merged_ac_stream",
    "encode_pr103_merged_ac_stream",
    "parse_pr103_lc_ac_payload",
    "render_markdown",
]
