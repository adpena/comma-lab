#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile tiny pair-index selector priors for the PR101/FEC6 near miss.

This is a deterministic, no-dispatch profiler. It does not build archives or
claim score movement. It asks whether the FEC6 selector can be replaced by a
small runtime-source rule that predicts selector modes from pair index alone.
Per contest compliance, the profiler accounts for source-embedded parameters
separately from archive bytes and flags selector-like source tables.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import repo_relative, write_json  # noqa: E402

DEFAULT_FEC6_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
)
DEFAULT_FEC6_MANIFEST = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/packet_manifest.json"
)
DEFAULT_CPU_EVAL = (
    REPO_ROOT / "experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/contest_auth_eval.json"
)
DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
)
DEFAULT_OUTPUT_JSON = REPO_ROOT / ".omx/research/pr101_selector_prior_profile_20260515_codex.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / ".omx/research/pr101_selector_prior_profile_20260515_codex.md"

RATE_DENOMINATOR_BYTES = 37_545_489
TARGET_CPU_SCORE = 0.192
N_PAIRS = 600
OUTER_MAGIC = b"FP11"
SELECTOR_MAGIC = b"FEC6"

FEC6_FIXED_K16_MODE_IDS = (
    "none",
    "frame0_blue_chroma_amp_1",
    "frame0_blue_chroma_amp_3",
    "frame0_luma_bias_+1",
    "frame0_luma_bias_-1",
    "frame0_luma_bias_-2",
    "frame0_luma_bias_-4",
    "frame0_rgb_bias_m2_p1_p1",
    "frame0_rgb_bias_m4_p2_p2",
    "frame0_rgb_bias_p0_m1_p1",
    "frame0_rgb_bias_p0_m2_p2",
    "frame0_rgb_bias_p0_p1_m1",
    "frame0_rgb_bias_p0_p2_m2",
    "frame0_rgb_bias_p2_m1_m1",
    "frame0_rgb_bias_p4_m2_m2",
    "frame0_roll_dx+0_dy+1",
)

FEC6_FIXED_K16_CODE_BITS = (
    "00",
    "1100",
    "01",
    "111010",
    "11010",
    "111011",
    "111100",
    "100",
    "111101",
    "11011",
    "1111110",
    "111110",
    "11111110",
    "101",
    "11100",
    "11111111",
)
FEC6_FIXED_K16_DECODE = {bits: code for code, bits in enumerate(FEC6_FIXED_K16_CODE_BITS)}


@dataclass(frozen=True)
class RuleCandidate:
    family: str
    name: str
    params: dict[str, Any]
    predictions: tuple[int, ...]
    source_literal_bytes_estimate: int
    learned_source_values: int


@dataclass(frozen=True)
class ComponentTables:
    component_score: tuple[tuple[float, ...], ...]
    pose: tuple[tuple[float, ...], ...]
    seg: tuple[tuple[float, ...], ...]
    missing_rows: tuple[tuple[bool, ...], ...]
    pair_rows_loaded: int
    pair_rows_paths: tuple[str, ...]


def sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_source_literal_bytes(payload: Mapping[str, Any]) -> int:
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def read_single_member_payload(path: Path) -> tuple[dict[str, Any], bytes]:
    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise ValueError(f"expected one archive member in {path}, found {len(infos)}")
        info = infos[0]
        if info.filename.startswith("/") or ".." in Path(info.filename).parts:
            raise ValueError(f"unsafe archive member name: {info.filename!r}")
        payload = zf.read(info.filename)
    return {
        "member_name": info.filename,
        "member_bytes": len(payload),
        "member_sha256": sha256_bytes(payload),
        "zip_compression": "stored" if info.compress_type == zipfile.ZIP_STORED else str(info.compress_type),
    }, payload


def decode_fec6_fixed_huffman_codes(payload: bytes, *, n_pairs: int) -> tuple[list[int], int]:
    codes: list[int] = []
    prefix = ""
    bit_pos = 0
    max_bits = len(payload) * 8
    while len(codes) < n_pairs:
        if bit_pos >= max_bits:
            raise ValueError("FEC6 selector bitstream truncated")
        bit = (payload[bit_pos // 8] >> (7 - (bit_pos % 8))) & 1
        bit_pos += 1
        prefix += "1" if bit else "0"
        code = FEC6_FIXED_K16_DECODE.get(prefix)
        if code is None:
            if len(prefix) > 8:
                raise ValueError("FEC6 selector has invalid prefix code")
            continue
        codes.append(int(code))
        prefix = ""
    if prefix:
        raise ValueError("FEC6 selector ended mid-symbol")
    for trailing in range(bit_pos, max_bits):
        if (payload[trailing // 8] >> (7 - (trailing % 8))) & 1:
            raise ValueError("FEC6 selector has non-zero padding bits")
    return codes, bit_pos


def parse_fec6_archive(path: Path) -> dict[str, Any]:
    member, wrapper = read_single_member_payload(path)
    if len(wrapper) < 10 or wrapper[:4] != OUTER_MAGIC:
        raise ValueError(f"expected FP11 wrapper in {path}")
    source_len = int.from_bytes(wrapper[4:8], "little")
    selector_len_offset = 8 + source_len
    if selector_len_offset + 2 > len(wrapper):
        raise ValueError("FP11 wrapper truncated before selector length")
    selector_len = int.from_bytes(wrapper[selector_len_offset : selector_len_offset + 2], "little")
    selector_start = selector_len_offset + 2
    selector_end = selector_start + selector_len
    if selector_end != len(wrapper):
        raise ValueError("FP11 wrapper selector length does not consume payload")
    selector_payload = wrapper[selector_start:selector_end]
    if len(selector_payload) < 6 or selector_payload[:4] != SELECTOR_MAGIC:
        raise ValueError(f"expected FEC6 selector payload, got {selector_payload[:4]!r}")
    n_pairs = int.from_bytes(selector_payload[4:6], "little")
    codes, used_bits = decode_fec6_fixed_huffman_codes(selector_payload[6:], n_pairs=n_pairs)
    return {
        "archive_path": repo_relative(path, REPO_ROOT),
        "archive_bytes": path.stat().st_size,
        "archive_sha256": sha256_bytes(path.read_bytes()),
        "member": member,
        "source_payload_bytes": source_len,
        "wrapper_overhead_bytes": 10,
        "selector_payload_bytes": len(selector_payload),
        "selector_index_bytes": len(selector_payload) - 6,
        "selector_code_bits_total": used_bits,
        "selector_payload_sha256": sha256_bytes(selector_payload),
        "n_pairs": n_pairs,
        "codes": codes,
        "histogram": {str(code): int(count) for code, count in sorted(Counter(codes).items())},
    }


def majority(values: Sequence[int]) -> int:
    if not values:
        return 0
    return min(Counter(values).items(), key=lambda item: (-item[1], item[0]))[0]


def source_embedding_risk(
    *, family: str, learned_source_values: int, n_pairs: int, exact_selector_match: bool
) -> dict[str, Any]:
    if learned_source_values >= n_pairs:
        level = "forbidden"
        reason = "per-pair selector-equivalent source table"
    elif exact_selector_match and learned_source_values > max(32, n_pairs // 16):
        level = "forbidden"
        reason = "exact selector reconstruction requires a large source table"
    elif family in {"periodic_table", "bucket_table"} and learned_source_values > 32:
        level = "high"
        reason = "source table is small versus 600 pairs but still selector-like"
    elif family in {"periodic_table", "bucket_table"} and learned_source_values > 8:
        level = "medium"
        reason = "compact source table; needs operator compliance review"
    else:
        level = "low"
        reason = "small general pair-index rule, no per-pair data"
    return {
        "level": level,
        "reason": reason,
        "learned_source_values": int(learned_source_values),
        "learned_values_per_pair": float(learned_source_values) / float(n_pairs or 1),
    }


def make_candidate(
    family: str,
    name: str,
    params: dict[str, Any],
    predictions: Iterable[int],
    learned_source_values: int,
) -> RuleCandidate:
    params_payload = {"family": family, "params": params}
    return RuleCandidate(
        family=family,
        name=name,
        params=params,
        predictions=tuple(int(value) for value in predictions),
        source_literal_bytes_estimate=canonical_source_literal_bytes(params_payload),
        learned_source_values=int(learned_source_values),
    )


def generate_rule_candidates(
    codes: Sequence[int], *, max_period: int = 64, max_buckets: int = 64
) -> list[RuleCandidate]:
    n = len(codes)
    candidates: list[RuleCandidate] = []

    for code in range(len(FEC6_FIXED_K16_MODE_IDS)):
        candidates.append(
            make_candidate("constant", f"constant_{code}", {"code": code}, [code] * n, 1)
        )

    for period in range(2, max_period + 1):
        table: list[int] = []
        predictions = [0] * n
        for residue in range(period):
            value = majority([codes[idx] for idx in range(residue, n, period)])
            table.append(value)
            for idx in range(residue, n, period):
                predictions[idx] = value
        candidates.append(
            make_candidate(
                "periodic_table",
                f"periodic_p{period}",
                {"period": period, "table": table},
                predictions,
                period,
            )
        )

    for buckets in range(2, max_buckets + 1):
        table = []
        predictions: list[int] = []
        for bucket in range(buckets):
            lo = bucket * n // buckets
            hi = (bucket + 1) * n // buckets
            value = majority(codes[lo:hi])
            table.append(value)
            predictions.extend([value] * (hi - lo))
        candidates.append(
            make_candidate(
                "bucket_table",
                f"equal_bucket_b{buckets}",
                {"buckets": buckets, "table": table},
                predictions,
                buckets,
            )
        )

    for threshold in range(1, n):
        left = majority(codes[:threshold])
        right = majority(codes[threshold:])
        candidates.append(
            make_candidate(
                "threshold_stump",
                f"threshold_t{threshold}",
                {"threshold": threshold, "left": left, "right": right},
                [left] * threshold + [right] * (n - threshold),
                3,
            )
        )

    for modulus in range(2, max_period + 1):
        for cutoff in range(1, modulus):
            left_values = [codes[idx] for idx in range(n) if idx % modulus < cutoff]
            right_values = [codes[idx] for idx in range(n) if idx % modulus >= cutoff]
            left = majority(left_values)
            right = majority(right_values)
            predictions = [left if idx % modulus < cutoff else right for idx in range(n)]
            candidates.append(
                make_candidate(
                    "modulo_stump",
                    f"mod{modulus}_lt{cutoff}",
                    {"modulus": modulus, "cutoff": cutoff, "left": left, "right": right},
                    predictions,
                    4,
                )
            )

    for a in range(16):
        for b in range(16):
            candidates.append(
                make_candidate(
                    "linear_mod16",
                    f"linear_a{a}_b{b}",
                    {"a": a, "b": b, "expr": "(a*i+b)&15"},
                    [((a * idx + b) & 15) for idx in range(n)],
                    2,
                )
            )

    for a in range(16):
        for b in range(16):
            for c in range(16):
                candidates.append(
                    make_candidate(
                        "quadratic_mod16",
                        f"quad_a{a}_b{b}_c{c}",
                        {"a": a, "b": b, "c": c, "expr": "(a*i*i+b*i+c)&15"},
                        [((a * idx * idx + b * idx + c) & 15) for idx in range(n)],
                        3,
                    )
                )

    for a in range(1, 64, 2):
        for b in range(64):
            for shift in range(1, 10):
                candidates.append(
                    make_candidate(
                        "hash_low_degree",
                        f"hash_a{a}_b{b}_s{shift}",
                        {"a": a, "b": b, "shift": shift, "expr": "((a*i+b)^(i>>shift))&15"},
                        [(((a * idx + b) ^ (idx >> shift)) & 15) for idx in range(n)],
                        3,
                    )
                )

    return candidates


def score_from_components(*, avg_pose: float, avg_seg: float, archive_bytes: int) -> float:
    return (
        100.0 * float(avg_seg)
        + math.sqrt(10.0 * max(0.0, float(avg_pose)))
        + 25.0 * int(archive_bytes) / RATE_DENOMINATOR_BYTES
    )


def load_pair_rows(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[(int(row["pair"]), str(row["mode_id"]))] = row
    return rows


def pair_rows_paths_from_manifest(manifest: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for item in manifest.get("source_transparency", {}).get("source_paths", []):
        if not isinstance(item, dict):
            continue
        path_text = str(item.get("path", ""))
        if path_text.endswith("pair_component_rows.jsonl"):
            paths.append(REPO_ROOT / path_text)
    for item in manifest.get("selector", {}).get("overlay_artifacts", []):
        if not isinstance(item, dict):
            continue
        path_text = str(item.get("pair_component_rows", ""))
        if path_text:
            path = REPO_ROOT / path_text
            if path not in paths:
                paths.append(path)
    return paths


def build_component_tables(
    *,
    n_pairs: int,
    mode_ids: Sequence[str],
    pair_rows_paths: Sequence[Path],
) -> ComponentTables:
    merged: dict[tuple[int, str], dict[str, Any]] = {}
    loaded_paths: list[str] = []
    for path in pair_rows_paths:
        if not path.is_file():
            continue
        merged.update(load_pair_rows(path))
        loaded_paths.append(repo_relative(path, REPO_ROOT))
    component: list[tuple[float, ...]] = []
    pose: list[tuple[float, ...]] = []
    seg: list[tuple[float, ...]] = []
    missing: list[tuple[bool, ...]] = []
    for pair in range(n_pairs):
        none = merged.get((pair, "none"))
        if none is None:
            raise ValueError(f"pair rows missing required baseline pair={pair} mode=none")
        comp_row: list[float] = []
        pose_row: list[float] = []
        seg_row: list[float] = []
        missing_row: list[bool] = []
        for mode_id in mode_ids:
            row = merged.get((pair, str(mode_id)))
            missing_row.append(row is None)
            row = row or none
            comp_row.append(float(row["component_score_no_rate"]))
            pose_row.append(float(row["posenet_dist"]))
            seg_row.append(float(row["segnet_dist"]))
        component.append(tuple(comp_row))
        pose.append(tuple(pose_row))
        seg.append(tuple(seg_row))
        missing.append(tuple(missing_row))
    return ComponentTables(
        component_score=tuple(component),
        pose=tuple(pose),
        seg=tuple(seg),
        missing_rows=tuple(missing),
        pair_rows_loaded=len(merged),
        pair_rows_paths=tuple(loaded_paths),
    )


def summarize_predictions(
    predictions: Sequence[int], tables: ComponentTables
) -> dict[str, float | int]:
    n = len(predictions)
    total_component = 0.0
    total_pose = 0.0
    total_seg = 0.0
    missing = 0
    for pair, code in enumerate(predictions):
        total_component += tables.component_score[pair][code]
        total_pose += tables.pose[pair][code]
        total_seg += tables.seg[pair][code]
        missing += int(tables.missing_rows[pair][code])
    return {
        "component_score_no_rate_proxy": total_component / n,
        "avg_posenet_dist_proxy": total_pose / n,
        "avg_segnet_dist_proxy": total_seg / n,
        "missing_pair_mode_rows": missing,
    }


def required_saving_bytes(score: float, threshold: float) -> int:
    if score < threshold:
        return 0
    raw = (score - threshold) * RATE_DENOMINATOR_BYTES / 25.0
    needed = max(0, math.floor(raw) + 1)
    while score - 25.0 * needed / RATE_DENOMINATOR_BYTES >= threshold:
        needed += 1
    return needed


def evaluate_candidate(
    candidate: RuleCandidate,
    *,
    target_codes: Sequence[int],
    target_component: float,
    tables: ComponentTables,
    cpu_score: float,
    threshold: float,
    fec6_archive_bytes: int,
    source_archive_bytes: int,
) -> dict[str, Any]:
    if len(candidate.predictions) != len(target_codes):
        raise ValueError("candidate prediction length mismatch")
    pred_summary = summarize_predictions(candidate.predictions, tables)
    missing_rows = int(pred_summary["missing_pair_mode_rows"])
    component_rows_complete = missing_rows == 0
    component_delta = (
        float(pred_summary["component_score_no_rate_proxy"]) - float(target_component)
    )
    archive_delta = int(source_archive_bytes) - int(fec6_archive_bytes)
    rate_only_score = float(cpu_score) + 25.0 * archive_delta / RATE_DENOMINATOR_BYTES
    estimated_score = rate_only_score + component_delta
    mismatches = sum(
        1
        for got, want in zip(candidate.predictions, target_codes, strict=True)
        if got != want
    )
    exact_match = mismatches == 0
    risk = source_embedding_risk(
        family=candidate.family,
        learned_source_values=candidate.learned_source_values,
        n_pairs=len(target_codes),
        exact_selector_match=exact_match,
    )
    return {
        "family": candidate.family,
        "name": candidate.name,
        "params": candidate.params,
        "selector_match": {
            "mismatches": mismatches,
            "accuracy": 1.0 - mismatches / float(len(target_codes) or 1),
            "exact_match": exact_match,
        },
        "source_param_accounting": {
            "runtime_source_literal_bytes_estimate": candidate.source_literal_bytes_estimate,
            "learned_source_values": candidate.learned_source_values,
            "source_bytes_are_not_archive_bytes": True,
            "risk": risk,
        },
        "component_risk": {
            **pred_summary,
            "component_score_delta_no_rate_proxy_vs_fec6": component_delta,
            "evidence_grade": "pair-row proxy/advisory; not exact CPU/CUDA score evidence",
        },
        "score_estimate": {
            "archive_bytes_estimated_if_selector_source_rule": int(source_archive_bytes),
            "archive_saved_bytes_vs_fec6": int(fec6_archive_bytes) - int(source_archive_bytes),
            "score_if_fec6_components_unchanged": rate_only_score,
            "allowable_component_delta_for_sub0192": float(threshold) - rate_only_score,
            "estimated_cpu_score_from_pair_rows": estimated_score,
            "component_rows_complete_for_prediction": component_rows_complete,
            "proxy_allows_sub0192_gate": (
                estimated_score < threshold and component_rows_complete
            ),
        },
        "verdict": (
            "proxy_allows_sub0192_if_compliance_review_passes"
            if estimated_score < threshold
            and component_rows_complete
            and risk["level"] not in {"forbidden", "high"}
            else "blocked_by_missing_pair_mode_rows"
            if not component_rows_complete
            else "blocked_by_component_risk"
            if estimated_score >= threshold
            else "blocked_by_source_embedding_risk"
        ),
    }


def build_profile(
    *,
    fec6_archive: Path = DEFAULT_FEC6_ARCHIVE,
    fec6_manifest: Path = DEFAULT_FEC6_MANIFEST,
    cpu_eval: Path = DEFAULT_CPU_EVAL,
    source_archive: Path = DEFAULT_SOURCE_ARCHIVE,
    threshold: float = TARGET_CPU_SCORE,
    max_period: int = 64,
    max_buckets: int = 64,
    top_k: int = 20,
) -> dict[str, Any]:
    selector = parse_fec6_archive(fec6_archive)
    manifest = load_json(fec6_manifest)
    cpu = load_json(cpu_eval)
    score_axis = cpu.get("score_axis")
    if score_axis != "contest_cpu":
        raise ValueError(
            f"--cpu-eval must be a [contest-CPU] artifact; got score_axis={score_axis!r}"
        )
    cpu_score = float(cpu.get("canonical_score", cpu.get("score_recomputed_from_components")))
    source_archive = source_archive if source_archive is not None else REPO_ROOT / manifest["archive"]["source"]
    if not source_archive.is_file():
        source_archive = REPO_ROOT / str(manifest["archive"]["source"])
    pair_paths = pair_rows_paths_from_manifest(manifest)
    tables = build_component_tables(
        n_pairs=int(selector["n_pairs"]),
        mode_ids=FEC6_FIXED_K16_MODE_IDS,
        pair_rows_paths=pair_paths,
    )
    target_codes = [int(code) for code in selector["codes"]]
    target_summary = summarize_predictions(target_codes, tables)
    target_component = float(target_summary["component_score_no_rate_proxy"])

    candidates = generate_rule_candidates(
        target_codes, max_period=int(max_period), max_buckets=int(max_buckets)
    )
    rows = [
        evaluate_candidate(
            candidate,
            target_codes=target_codes,
            target_component=target_component,
            tables=tables,
            cpu_score=cpu_score,
            threshold=threshold,
            fec6_archive_bytes=int(selector["archive_bytes"]),
            source_archive_bytes=source_archive.stat().st_size,
        )
        for candidate in candidates
    ]
    rows_by_score = sorted(
        rows,
        key=lambda row: (
            float(row["score_estimate"]["estimated_cpu_score_from_pair_rows"]),
            int(row["selector_match"]["mismatches"]),
            int(row["source_param_accounting"]["runtime_source_literal_bytes_estimate"]),
            row["name"],
        ),
    )
    rows_by_match = sorted(
        rows,
        key=lambda row: (
            int(row["selector_match"]["mismatches"]),
            float(row["score_estimate"]["estimated_cpu_score_from_pair_rows"]),
            int(row["source_param_accounting"]["runtime_source_literal_bytes_estimate"]),
            row["name"],
        ),
    )
    source_saved = int(selector["archive_bytes"]) - source_archive.stat().st_size
    rate_only_score = cpu_score - 25.0 * source_saved / RATE_DENOMINATOR_BYTES
    plausible = [
        row
        for row in rows
        if row["score_estimate"]["proxy_allows_sub0192_gate"]
        and row["source_param_accounting"]["risk"]["level"] not in {"forbidden", "high"}
    ]
    return {
        "schema": "pr101_selector_prior_profile.v1",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "research_only": True,
        "hypothesis": "source-free/tiny pair-index-only selector prior for PR101/FEC6",
        "inputs": {
            "fec6_archive": repo_relative(fec6_archive, REPO_ROOT),
            "fec6_manifest": repo_relative(fec6_manifest, REPO_ROOT),
            "cpu_eval": repo_relative(cpu_eval, REPO_ROOT),
            "source_archive": repo_relative(source_archive, REPO_ROOT),
            "pair_rows_paths": list(tables.pair_rows_paths),
            "pair_rows_loaded": tables.pair_rows_loaded,
        },
        "fec6_reference": {
            key: value
            for key, value in selector.items()
            if key not in {"codes"}
        },
        "exact_cpu_reference": {
            "score_axis": cpu.get("score_axis"),
            "score": cpu_score,
            "avg_segnet_dist": cpu.get("avg_segnet_dist"),
            "avg_posenet_dist": cpu.get("avg_posenet_dist"),
            "archive_size_bytes": cpu.get("archive_size_bytes"),
            "required_same_component_saving_bytes_for_sub0192": required_saving_bytes(
                cpu_score, threshold
            ),
        },
        "source_rule_byte_accounting": {
            "fec6_archive_bytes": int(selector["archive_bytes"]),
            "source_archive_bytes_if_selector_removed": source_archive.stat().st_size,
            "archive_saved_bytes_vs_fec6": source_saved,
            "removed_selector_payload_bytes": int(selector["selector_payload_bytes"]),
            "removed_wrapper_overhead_bytes": int(selector["wrapper_overhead_bytes"]),
            "score_if_fec6_components_unchanged": rate_only_score,
            "allowable_component_delta_for_sub0192": threshold - rate_only_score,
            "source_runtime_bytes_not_in_archive_score": True,
            "compliance_constraint": (
                "runtime source may contain only small general rules/models; no per-pair "
                "selector table or source-embedded 600-entry sequence"
            ),
        },
        "target_selector_component_proxy": target_summary,
        "rule_search": {
            "families": [
                "constant",
                "periodic_table",
                "equal-width pair-index buckets",
                "threshold decision stumps",
                "modulo decision stumps",
                "linear_mod16",
                "quadratic_mod16",
                "hash_low_degree",
            ],
            "candidate_count": len(rows),
            "max_period": max_period,
            "max_buckets": max_buckets,
            "top_by_estimated_cpu_score": rows_by_score[:top_k],
            "top_by_selector_match": rows_by_match[:top_k],
            "plausible_sub0192_rows": plausible[:top_k],
        },
        "conclusion": {
            "any_rule_plausibly_beats_sub0192": bool(plausible),
            "best_estimated_cpu_score": rows_by_score[0]["score_estimate"][
                "estimated_cpu_score_from_pair_rows"
            ],
            "best_rule": rows_by_score[0]["name"],
            "best_rule_family": rows_by_score[0]["family"],
            "best_rule_component_delta_vs_fec6": rows_by_score[0]["component_risk"][
                "component_score_delta_no_rate_proxy_vs_fec6"
            ],
            "best_rule_selector_mismatches": rows_by_score[0]["selector_match"]["mismatches"],
            "blocker": (
                "pair-index-only tiny rules do not preserve enough of the FEC6 selector's "
                "component gain; the rate saving is large enough, but component proxy loss "
                "is roughly an order of magnitude above the CPU allowance"
            )
            if not plausible
            else None,
        },
    }


def render_markdown(profile: Mapping[str, Any]) -> str:
    ref = profile["exact_cpu_reference"]
    byte = profile["source_rule_byte_accounting"]
    search = profile["rule_search"]
    conclusion = profile["conclusion"]
    lines = [
        "# PR101 Selector Prior Profile",
        "",
        "- score_claim: `false`",
        "- dispatch_attempted: `false`",
        "- ready_for_exact_eval_dispatch: `false`",
        "- evidence axis: pair-row proxy/advisory for component risk; exact CPU only for FEC6 anchor/rate math",
        f"- FEC6 exact CPU score: `{ref['score']}`",
        f"- same-component bytes needed for `<0.192`: `{ref['required_same_component_saving_bytes_for_sub0192']}`",
        "",
        "## Byte Accounting",
        "",
        f"- FEC6 archive bytes: `{byte['fec6_archive_bytes']}`",
        f"- source archive bytes if selector is removed and computed from source: `{byte['source_archive_bytes_if_selector_removed']}`",
        f"- archive bytes saved vs FEC6: `{byte['archive_saved_bytes_vs_fec6']}`",
        f"- removed selector payload bytes: `{byte['removed_selector_payload_bytes']}`",
        f"- removed wrapper overhead bytes: `{byte['removed_wrapper_overhead_bytes']}`",
        f"- score if FEC6 components were unchanged: `{byte['score_if_fec6_components_unchanged']}`",
        f"- allowable component delta for `<0.192`: `{byte['allowable_component_delta_for_sub0192']}`",
        "",
        "## Rule Search",
        "",
        f"- candidates tested: `{search['candidate_count']}`",
        f"- plausible `<0.192` rows under proxy + compliance filter: `{len(search['plausible_sub0192_rows'])}`",
        "",
        "| rank | family | rule | estimated CPU | component delta | mismatches | source bytes | risk | verdict |",
        "|---:|---|---|---:|---:|---:|---:|---|---|",
    ]
    for rank, row in enumerate(search["top_by_estimated_cpu_score"][:10], start=1):
        risk = row["source_param_accounting"]["risk"]
        lines.append(
            "| {rank} | {family} | `{name}` | {score:.12f} | {delta:.12f} | {mismatch} | {src_bytes} | {risk} | {verdict} |".format(
                rank=rank,
                family=row["family"],
                name=row["name"],
                score=float(row["score_estimate"]["estimated_cpu_score_from_pair_rows"]),
                delta=float(row["component_risk"]["component_score_delta_no_rate_proxy_vs_fec6"]),
                mismatch=int(row["selector_match"]["mismatches"]),
                src_bytes=int(row["source_param_accounting"]["runtime_source_literal_bytes_estimate"]),
                risk=risk["level"],
                verdict=row["verdict"],
            )
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- any rule plausibly beats `<0.192`: `{str(conclusion['any_rule_plausibly_beats_sub0192']).lower()}`",
            f"- best rule: `{conclusion['best_rule']}` (`{conclusion['best_rule_family']}`)",
            f"- best estimated CPU score: `{conclusion['best_estimated_cpu_score']}`",
            f"- best component delta vs FEC6 proxy: `{conclusion['best_rule_component_delta_vs_fec6']}`",
            f"- best selector mismatches: `{conclusion['best_rule_selector_mismatches']}`",
            f"- blocker: {conclusion['blocker']}",
            "",
            "## Compliance Risks",
            "",
            "- A true per-pair selector table in runtime source is forbidden and not profiled as a valid candidate.",
            "- Periodic/bucket source tables are flagged medium/high once they become selector-like rather than general rules.",
            "- Even a low-risk formula rule needs exact CPU/CUDA replay before any score or promotion claim.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fec6-archive", type=Path, default=DEFAULT_FEC6_ARCHIVE)
    parser.add_argument("--fec6-manifest", type=Path, default=DEFAULT_FEC6_MANIFEST)
    parser.add_argument("--cpu-eval", type=Path, default=DEFAULT_CPU_EVAL)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--threshold", type=float, default=TARGET_CPU_SCORE)
    parser.add_argument("--max-period", type=int, default=64)
    parser.add_argument("--max-buckets", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    profile = build_profile(
        fec6_archive=args.fec6_archive,
        fec6_manifest=args.fec6_manifest,
        cpu_eval=args.cpu_eval,
        source_archive=args.source_archive,
        threshold=args.threshold,
        max_period=args.max_period,
        max_buckets=args.max_buckets,
        top_k=args.top_k,
    )
    write_json(args.output_json, profile)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(profile) + "\n", encoding="utf-8")
    print(repo_relative(args.output_json, REPO_ROOT))
    print(repo_relative(args.output_md, REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
