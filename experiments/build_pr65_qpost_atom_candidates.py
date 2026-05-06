#!/usr/bin/env python3
"""Build tiny PR65 qpost atom-subset candidates on a C-089-style archive.

This is a local-only planner/builder.  It treats PR65/Henosis postprocess
streams as an atom dictionary, selects small pair-local subsets, charges the
resulting ``qpost.bin`` bytes inside the archive, and emits exact-eval command
templates without dispatching anything.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import brotli
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PRODUCER = "experiments/build_pr65_qpost_atom_candidates.py"
SCHEMA_VERSION = 1
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
BASELINE_SCORE = 0.3154707273953505  # [external: PR-65 contest-CUDA T4 frontier]
SUB314_TARGET = 0.314
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/pr65_qpost_atom_worker_20260503"
DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip"
)
DEFAULT_PR65_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/top_submission_delta_reverse_engineering_20260503/"
    "sources/pr65_henosis_archive.zip"
)
DEFAULT_ANATOMY_JSON = (
    REPO_ROOT
    / "experiments/results/top_submission_delta_reverse_engineering_20260503/"
    "archive_anatomy/public_vs_c089_anatomy.json"
)
DEFAULT_C089_TRACE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/component_trace.json"
)
DEFAULT_PR65_TRACE = (
    REPO_ROOT
    / "experiments/results/vast_harvest/public_external_component_trace_20260502T0642Z/"
    "pr65_torch25_compat_adapter/component_trace.json"
)
QPOST_BUILDER_PATH = REPO_ROOT / "experiments/build_qzs3_postprocess_candidate.py"
INFLATE_SH = REPO_ROOT / "submissions/robust_current/inflate.sh"
EXPECTED_C089_SHA256 = "0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8"
EXPECTED_PR65_SHA256 = "b331cb4f6df9d8929db966b943b8c73624cdf3b6db71acbde361570852e59e68"
EXPECTED_PR65_HEAD_SHA = "a8b53b5280ee8f05db65740cd48cf7c321a55497"
SUPPORTED_PAIR_FILTER_STREAMS = ("post", "shift", "frac", "frac2", "frac3", "bias", "region")
QPOST_STREAM_NAMES = (*SUPPORTED_PAIR_FILTER_STREAMS, "randmulti")


class QPostAtomPlannerError(ValueError):
    """Raised when local qpost atom planning fails a closed-world guard."""


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    include_streams: tuple[str, ...]
    top_pairs: int
    risk_family: str


DEFAULT_SPECS: tuple[CandidateSpec, ...] = (
    CandidateSpec("pr65_qpost_bias_poseadv_top008", ("bias",), 8, "low_byte_mild_global_negative"),
    CandidateSpec("pr65_qpost_bias_region_poseadv_top008", ("bias", "region"), 8, "medium_region_bias_global_negative"),
    CandidateSpec("pr65_qpost_post_bias_poseadv_top004", ("post", "bias"), 4, "high_post_global_negative"),
    CandidateSpec("pr65_qpost_post_region_bias_poseadv_top004", ("post", "region", "bias"), 4, "high_post_region_bias_global_negative"),
    CandidateSpec("pr65_qpost_shift_frac_poseadv_top004", ("shift", "frac", "frac2", "frac3"), 4, "very_high_motion_global_negative"),
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise QPostAtomPlannerError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise QPostAtomPlannerError(f"expected JSON object: {path}")
    return payload


def _load_qpost_builder():
    spec = importlib.util.spec_from_file_location("pact_qpost_builder_for_atoms", QPOST_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise QPostAtomPlannerError(f"cannot load qpost builder: {QPOST_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_runtime_hook(inflate_sh: Path) -> dict[str, Any]:
    text = inflate_sh.read_text()
    has_hook = "qpost.bin detected" in text and "apply_qzs3_postprocess.py" in text
    return {
        "inflate_sh": str(inflate_sh),
        "inflate_sh_sha256": _sha256_path(inflate_sh),
        "qpost_runtime_hook_present": has_hook,
    }


def _verify_source_custody(
    *,
    source_archive: Path,
    pr65_archive: Path,
    anatomy_json: Path | None,
    expected_source_sha256: str | None,
    expected_pr65_sha256: str | None,
    expected_pr65_head_sha: str | None,
) -> dict[str, Any]:
    source_sha = _sha256_path(source_archive)
    pr65_sha = _sha256_path(pr65_archive)
    if expected_source_sha256 and source_sha != expected_source_sha256:
        raise QPostAtomPlannerError(
            f"source archive SHA mismatch: expected {expected_source_sha256}, got {source_sha}"
        )
    if expected_pr65_sha256 and pr65_sha != expected_pr65_sha256:
        raise QPostAtomPlannerError(
            f"PR65 source SHA mismatch: expected {expected_pr65_sha256}, got {pr65_sha}"
        )
    anatomy: dict[str, Any] = {}
    if anatomy_json is not None and anatomy_json.exists():
        anatomy = _read_json(anatomy_json)
        pr65 = anatomy.get("archives", {}).get("pr65_henosis", {})
        if not isinstance(pr65, dict):
            raise QPostAtomPlannerError("anatomy JSON is missing archives.pr65_henosis")
        anatomy_sha = pr65.get("zip_sha256")
        head_sha = pr65.get("github_head_sha")
        if anatomy_sha and anatomy_sha != pr65_sha:
            raise QPostAtomPlannerError(
                f"PR65 archive does not match anatomy JSON: {pr65_sha} != {anatomy_sha}"
            )
        if expected_pr65_head_sha and head_sha != expected_pr65_head_sha:
            raise QPostAtomPlannerError(
                f"PR65 head SHA mismatch: expected {expected_pr65_head_sha}, got {head_sha}"
            )
    return {
        "source_archive": str(source_archive.resolve()),
        "source_archive_bytes": source_archive.stat().st_size,
        "source_archive_sha256": source_sha,
        "pr65_archive": str(pr65_archive.resolve()),
        "pr65_archive_bytes": pr65_archive.stat().st_size,
        "pr65_archive_sha256": pr65_sha,
        "pr65_head_sha256": expected_pr65_head_sha,
        "anatomy_json": str(anatomy_json.resolve()) if anatomy_json is not None else None,
    }


def _post_matrix(blob: bytes) -> np.ndarray:
    raw = brotli.decompress(blob)
    if raw[:4] == b"PCD1":
        pos = 4
        stage_count = raw[pos]
        pos += 1
        stages: list[np.ndarray] = []
        for _ in range(stage_count):
            pos += 1  # stage id
            n = int.from_bytes(raw[pos:pos + 2], "little")
            pos += 2
            if n != 600:
                raise QPostAtomPlannerError(f"PCD1 post stage has {n} choices, expected 600")
            stages.append(np.frombuffer(raw, dtype=np.uint8, count=n, offset=pos).copy())
            pos += n
        if pos != len(raw):
            raise QPostAtomPlannerError("PCD1 post stream has trailing bytes")
        return np.stack(stages, axis=0)
    if len(raw) % 600 != 0:
        raise QPostAtomPlannerError("headerless post stream length is not a multiple of 600")
    stage_count = len(raw) // 600
    if stage_count not in (3, 4):
        raise QPostAtomPlannerError(f"unsupported post stream stage count: {stage_count}")
    return np.frombuffer(raw, dtype=np.uint8).copy().reshape(stage_count, 600)


def _decode_stream_arrays(builder: Any, streams: Mapping[str, bytes]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "post": _post_matrix(streams["post"]),
        "shift": builder._dense_or_delta_to_array(streams["shift"], magic_full=b"SH4", magic_delta=b"SD4", default=40),
        "frac": builder._frac_to_array(streams["frac"]),
        "frac2": builder._dense_or_delta_to_array(streams["frac2"], magic_full=b"FH2", magic_delta=b"FD2", default=4),
        "frac3": builder._dense_or_delta_to_array(streams["frac3"], magic_full=b"FH3", magic_delta=b"FD3", default=4),
        "bias": builder._dense_or_delta_to_array(streams["bias"], magic_full=b"BH1", magic_delta=b"BD1", default=13, center=13),
        "region": builder._region_to_array(streams["region"]),
    }
    return arrays


def _nondefault_mask(name: str, arr: np.ndarray) -> np.ndarray:
    if name == "post":
        return (arr != 0).sum(axis=0).astype(np.int64)
    defaults = {
        "shift": 40,
        "frac": 4,
        "frac2": 4,
        "frac3": 4,
        "bias": 13,
        "region": 0,
    }
    if name not in defaults:
        raise QPostAtomPlannerError(f"unsupported atom stream: {name}")
    return (arr != defaults[name]).astype(np.int64)


def _load_trace(path: Path | None) -> dict[int, dict[str, float]]:
    if path is None or not path.exists():
        return {}
    payload = _read_json(path)
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise QPostAtomPlannerError(f"component trace missing samples list: {path}")
    out: dict[int, dict[str, float]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        pair = int(sample["pair_index"])
        seg = float(sample.get("score_seg_contribution_exact", 0.0))
        pose = float(sample.get("score_pose_contribution_first_order", 0.0))
        combined = float(sample.get("score_combined_contribution_first_order", seg + pose))
        out[pair] = {
            "combined": combined,
            "seg": seg,
            "pose": pose,
            "segnet_dist": float(sample.get("segnet_dist", 0.0)),
            "posenet_dist": float(sample.get("posenet_dist", 0.0)),
        }
    return out


def _rank_pairs(
    *,
    arrays: Mapping[str, np.ndarray],
    include_streams: Sequence[str],
    c089_trace: Mapping[int, Mapping[str, float]],
    pr65_trace: Mapping[int, Mapping[str, float]],
    positive_trace_only: bool,
) -> list[dict[str, Any]]:
    unknown = sorted(set(include_streams) - set(SUPPORTED_PAIR_FILTER_STREAMS))
    if unknown:
        raise QPostAtomPlannerError(f"unsupported pair-filter stream(s): {unknown}")
    active_by_stream = {name: _nondefault_mask(name, arrays[name]) for name in include_streams}
    records: list[dict[str, Any]] = []
    for pair in range(600):
        counts = {name: int(mask[pair]) for name, mask in active_by_stream.items()}
        active_count = sum(counts.values())
        if active_count <= 0:
            continue
        c089 = c089_trace.get(pair, {})
        pr65 = pr65_trace.get(pair, {})
        c089_combined = float(c089.get("combined", 0.0))
        pr65_combined = float(pr65.get("combined", 0.0))
        c089_pose = float(c089.get("pose", 0.0))
        pr65_pose = float(pr65.get("pose", 0.0))
        opportunity = c089_combined - pr65_combined if c089 and pr65 else 0.0
        pose_opportunity = c089_pose - pr65_pose if c089 and pr65 else 0.0
        if positive_trace_only and c089_trace and pr65_trace and opportunity <= 0.0:
            continue
        records.append(
            {
                "pair_index": pair,
                "active_atom_count": active_count,
                "active_by_stream": counts,
                "c089_combined_contribution": c089_combined,
                "pr65_combined_contribution": pr65_combined,
                "public_trace_opportunity": opportunity,
                "public_trace_pose_opportunity": pose_opportunity,
                "rank_score": opportunity * 1_000_000.0 + float(active_count) * 0.01,
            }
        )
    records.sort(key=lambda r: (-float(r["rank_score"]), int(r["pair_index"])))
    return records


def _summarize_original_streams(streams: Mapping[str, bytes], arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in SUPPORTED_PAIR_FILTER_STREAMS:
        decompressed = brotli.decompress(streams[name])
        active_counts = _nondefault_mask(name, arrays[name])
        summary[name] = {
            "encoded_bytes": len(streams[name]),
            "encoded_sha256": _sha256_bytes(streams[name]),
            "decoded_bytes": len(decompressed),
            "decoded_sha256": _sha256_bytes(decompressed),
            "active_pairs": int((active_counts > 0).sum()),
            "active_atoms": int(active_counts.sum()),
        }
    summary["randmulti"] = {
        "encoded_bytes": len(streams["randmulti"]),
        "encoded_sha256": _sha256_bytes(streams["randmulti"]),
        "pair_filter_supported": False,
        "omitted_reason": "randmulti has no reviewed sparse pair-filter encoder",
    }
    return summary


def _no_op_proof(
    *,
    arrays: Mapping[str, np.ndarray],
    include_streams: Sequence[str],
    selected_pairs: Sequence[int],
) -> dict[str, Any]:
    by_stream: dict[str, Any] = {}
    total = 0
    for name in include_streams:
        counts = _nondefault_mask(name, arrays[name])
        selected_active = int(counts[list(selected_pairs)].sum()) if selected_pairs else 0
        selected_pair_count = int((counts[list(selected_pairs)] > 0).sum()) if selected_pairs else 0
        total += selected_active
        by_stream[name] = {
            "selected_active_atoms": selected_active,
            "selected_active_pairs": selected_pair_count,
        }
    return {
        "is_noop": total == 0,
        "selected_active_atoms_total": total,
        "selected_pairs": list(selected_pairs),
        "by_stream": by_stream,
        "nonselected_pairs_default_to_identity": True,
    }


def _risk_classification(
    *,
    include_streams: Sequence[str],
    selected_pairs: Sequence[int],
    no_op_proof: Mapping[str, Any],
    formula_rate_score_delta: float,
    trace_opportunity_bound: float,
    risk_family: str,
) -> dict[str, Any]:
    needed = (BASELINE_SCORE - SUB314_TARGET) + formula_rate_score_delta
    reasons: list[str] = []
    inherited_shape = False
    if not selected_pairs:
        reasons.append("no selected pairs")
    if bool(no_op_proof.get("is_noop")):
        reasons.append("qpost subset is no-op after identity-default filtering")
    if "post" in include_streams:
        reasons.append("contains post stream; global exact ablation was strongly negative")
    if any(name in include_streams for name in ("shift", "frac", "frac2", "frac3")):
        reasons.append("contains motion/fractional shift stream; global exact ablations were pose-negative")
    if "region" in include_streams:
        reasons.append("contains region stream; global exact ablation was pose-negative")
    if len(selected_pairs) >= 600 or set(include_streams) == set(QPOST_STREAM_NAMES):
        inherited_shape = True
        reasons.append("full/global qpost shape is inherited exact-negative and blocked")
    if trace_opportunity_bound <= 0.0:
        reasons.append("no positive public-trace opportunity bound")
    if trace_opportunity_bound < needed:
        reasons.append(
            f"public-trace opportunity bound {trace_opportunity_bound:.9f} is below "
            f"sub-0.314 break-even {needed:.9f}"
        )
    dispatchable = not reasons and trace_opportunity_bound >= needed
    return {
        "risk_family": risk_family,
        "component_risk": "high" if reasons else "medium_unvalidated",
        "inherited_exact_negative_full_qpost": inherited_shape,
        "break_even_component_gain_for_sub314": needed,
        "public_trace_opportunity_bound": trace_opportunity_bound,
        "dispatch_recommendation": "do_not_dispatch" if not dispatchable else "exact_cuda_eval_candidate_after_lane_claim",
        "dispatch_blockers": reasons,
        "evidence_grade": "byte_trace_planning_only_until_exact_cuda",
    }


def _candidate_exact_eval_command(archive: Path, candidate_id: str) -> str:
    work_dir = DEFAULT_OUTPUT_DIR / "exact_eval_work" / candidate_id
    return (
        ".venv/bin/python -u experiments/contest_auth_eval.py "
        f"--archive {archive} "
        "--inflate-sh submissions/robust_current/inflate.sh "
        "--upstream-dir upstream --device cuda --keep-work-dir "
        f"--work-dir {work_dir}"
    )


def build_candidate_from_spec(
    *,
    builder: Any,
    source_archive: Path,
    pr65_archive: Path,
    output_dir: Path,
    spec: CandidateSpec,
    pair_rankings: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any] | None:
    selected = [int(row["pair_index"]) for row in pair_rankings[: spec.top_pairs]]
    no_op = _no_op_proof(arrays=arrays, include_streams=spec.include_streams, selected_pairs=selected)
    if bool(no_op["is_noop"]):
        return {
            "candidate_id": spec.candidate_id,
            "built": False,
            "skip_reason": "no_op_after_selected_pair_filter",
            "include_streams": list(spec.include_streams),
            "selected_pairs": selected,
            "no_op_proof": no_op,
        }

    candidate_dir = output_dir / spec.candidate_id
    archive_path = candidate_dir / "archive.zip"
    meta = builder.build_candidate(
        source_archive,
        pr65_archive,
        archive_path,
        include_streams=spec.include_streams,
        pair_indices=tuple(selected),
    )
    qpost_streams = meta.get("qpost_streams", {})
    original_streams_equal = {
        name: bool(qpost_streams.get(name, {}).get("sha256") == qpost_streams.get(name, {}).get("original_sha256"))
        for name in spec.include_streams
    }
    if selected and all(original_streams_equal.values()) and len(selected) >= 600:
        raise QPostAtomPlannerError(f"{spec.candidate_id} reproduces inherited full qpost stream(s)")

    selected_rank_records = [dict(row) for row in pair_rankings[: spec.top_pairs]]
    trace_opportunity = float(sum(float(row.get("public_trace_opportunity", 0.0)) for row in selected_rank_records))
    risk = _risk_classification(
        include_streams=spec.include_streams,
        selected_pairs=selected,
        no_op_proof=no_op,
        formula_rate_score_delta=float(meta["formula_rate_score_delta"]),
        trace_opportunity_bound=trace_opportunity,
        risk_family=spec.risk_family,
    )
    manifest = {
        **meta,
        "schema_version": SCHEMA_VERSION,
        "tool": PRODUCER,
        "candidate_id": spec.candidate_id,
        "score_claim": False,
        "selected_pair_rank_records": selected_rank_records,
        "no_op_proof": no_op,
        "original_streams_equal_after_filter": original_streams_equal,
        "component_risk_classification": risk,
        "exact_eval_command_template": _candidate_exact_eval_command(archive_path, spec.candidate_id),
        "remote_dispatch": {
            "dispatched": False,
            "requires_lane_claim": True,
            "note": "This worker is local-only; command template is not a dispatch.",
        },
    }
    _write_json(candidate_dir / "manifest.json", manifest)
    return {
        "candidate_id": spec.candidate_id,
        "built": True,
        "archive": str(archive_path),
        "archive_bytes": meta["output_archive_bytes"],
        "archive_sha256": meta["output_archive_sha256"],
        "archive_byte_delta": meta["archive_byte_delta"],
        "qpost_bytes": meta["members"]["qpost.bin"]["bytes"],
        "formula_rate_score_delta": meta["formula_rate_score_delta"],
        "include_streams": list(spec.include_streams),
        "selected_pairs": selected,
        "selected_active_atoms_total": no_op["selected_active_atoms_total"],
        "public_trace_opportunity_bound": trace_opportunity,
        "dispatch_recommendation": risk["dispatch_recommendation"],
        "dispatch_blockers": risk["dispatch_blockers"],
    }


def parse_specs(raw_specs: Sequence[str] | None) -> tuple[CandidateSpec, ...]:
    if not raw_specs:
        return DEFAULT_SPECS
    specs: list[CandidateSpec] = []
    for raw in raw_specs:
        parts = [part.strip() for part in raw.split(":")]
        if len(parts) != 3 or not all(parts):
            raise QPostAtomPlannerError(
                "custom specs must be candidate_id:stream1,stream2:top_pairs"
            )
        streams = tuple(part.strip() for part in parts[1].split(",") if part.strip())
        unknown = sorted(set(streams) - set(SUPPORTED_PAIR_FILTER_STREAMS))
        if unknown:
            raise QPostAtomPlannerError(f"unsupported custom qpost stream(s): {unknown}")
        top_pairs = int(parts[2])
        if top_pairs <= 0 or top_pairs > 600:
            raise QPostAtomPlannerError(f"invalid top_pairs for {parts[0]}: {top_pairs}")
        specs.append(CandidateSpec(parts[0], streams, top_pairs, "custom"))
    return tuple(specs)


def build_matrix(
    *,
    source_archive: Path = DEFAULT_SOURCE_ARCHIVE,
    pr65_archive: Path = DEFAULT_PR65_ARCHIVE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    anatomy_json: Path | None = DEFAULT_ANATOMY_JSON,
    c089_trace: Path | None = DEFAULT_C089_TRACE,
    pr65_trace: Path | None = DEFAULT_PR65_TRACE,
    specs: Sequence[CandidateSpec] = DEFAULT_SPECS,
    positive_trace_only: bool = True,
    expected_source_sha256: str | None = EXPECTED_C089_SHA256,
    expected_pr65_sha256: str | None = EXPECTED_PR65_SHA256,
    expected_pr65_head_sha: str | None = EXPECTED_PR65_HEAD_SHA,
) -> dict[str, Any]:
    source_archive = source_archive.resolve()
    pr65_archive = pr65_archive.resolve()
    output_dir = output_dir.resolve()
    runtime = _verify_runtime_hook(INFLATE_SH)
    if not runtime["qpost_runtime_hook_present"]:
        raise QPostAtomPlannerError("inflate runtime does not apply qpost.bin; refusing sidecar-only build")
    custody = _verify_source_custody(
        source_archive=source_archive,
        pr65_archive=pr65_archive,
        anatomy_json=anatomy_json,
        expected_source_sha256=expected_source_sha256,
        expected_pr65_sha256=expected_pr65_sha256,
        expected_pr65_head_sha=expected_pr65_head_sha,
    )
    builder = _load_qpost_builder()
    streams = builder.extract_pr65_qpost_streams(pr65_archive)
    arrays = _decode_stream_arrays(builder, streams)
    c089_trace_records = _load_trace(c089_trace)
    pr65_trace_records = _load_trace(pr65_trace)
    original_streams = _summarize_original_streams(streams, arrays)

    all_rankings: dict[str, list[dict[str, Any]]] = {}
    summary: list[dict[str, Any]] = []
    for spec in specs:
        ranked = _rank_pairs(
            arrays=arrays,
            include_streams=spec.include_streams,
            c089_trace=c089_trace_records,
            pr65_trace=pr65_trace_records,
            positive_trace_only=positive_trace_only,
        )
        all_rankings[spec.candidate_id] = ranked[: max(spec.top_pairs, 32)]
        if not ranked:
            summary.append(
                {
                    "candidate_id": spec.candidate_id,
                    "built": False,
                    "skip_reason": "no eligible active pairs after ranking filters",
                    "include_streams": list(spec.include_streams),
                }
            )
            continue
        summary.append(
            build_candidate_from_spec(
                builder=builder,
                source_archive=source_archive,
                pr65_archive=pr65_archive,
                output_dir=output_dir,
                spec=spec,
                pair_rankings=ranked,
                arrays=arrays,
            )
        )

    built = [row for row in summary if row and row.get("built")]
    dispatchable = [row for row in built if row.get("dispatch_recommendation") != "do_not_dispatch"]
    if dispatchable:
        dispatch_recommendation = "local_exact_eval_candidates_available_but_not_dispatched_by_worker"
        dispatch_reason = (
            "At least one bias-only qpost subset clears the public-trace "
            "sub-0.314 break-even bound. This is still planning evidence: "
            "the worker did not dispatch, and any exact eval must first claim "
            "a lane and run the canonical CUDA auth path on identical bytes."
        )
    else:
        dispatch_recommendation = "no_remote_dispatch_from_worker"
        dispatch_reason = (
            "All built qpost atom subsets are byte-additive and either inherit "
            "known global qpost negative risk or fail the public-trace "
            "break-even bound. Exact eval should wait for stronger component "
            "trace or a larger byte-saving base unless an operator chooses an "
            "explicit diagnostic lane claim."
        )
    matrix = {
        "schema_version": SCHEMA_VERSION,
        "tool": PRODUCER,
        "score_claim": False,
        "evidence_grade": "byte_trace_planning_only_until_exact_cuda",
        "baseline": {
            "label": "C-089 exact A++ frontier",
            "score": BASELINE_SCORE,
            "sub314_gap": BASELINE_SCORE - SUB314_TARGET,
            "bytes": 276342,
            "sha256": EXPECTED_C089_SHA256,
        },
        "source_custody": custody,
        "runtime_hook": runtime,
        "original_pr65_qpost_streams": original_streams,
        "ranking_inputs": {
            "c089_trace": str(c089_trace.resolve()) if c089_trace and c089_trace.exists() else None,
            "pr65_trace": str(pr65_trace.resolve()) if pr65_trace and pr65_trace.exists() else None,
            "positive_trace_only": positive_trace_only,
            "trace_pair_count": {
                "c089": len(c089_trace_records),
                "pr65": len(pr65_trace_records),
            },
        },
        "candidate_summary": summary,
        "rankings": all_rankings,
        "dispatch_summary": {
            "remote_dispatched": False,
            "dispatchable_candidates": dispatchable,
            "recommendation": dispatch_recommendation,
            "reason": dispatch_reason,
        },
    }
    _write_json(output_dir / "candidate_summary.json", matrix)
    return matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--pr65-archive", type=Path, default=DEFAULT_PR65_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anatomy-json", type=Path, default=DEFAULT_ANATOMY_JSON)
    parser.add_argument("--c089-trace", type=Path, default=DEFAULT_C089_TRACE)
    parser.add_argument("--pr65-trace", type=Path, default=DEFAULT_PR65_TRACE)
    parser.add_argument(
        "--spec",
        action="append",
        default=None,
        help="Custom candidate spec candidate_id:stream1,stream2:top_pairs. Can repeat.",
    )
    parser.add_argument(
        "--allow-negative-trace-pairs",
        action="store_true",
        help="Allow active qpost pairs even when PR65 public trace is not better on that pair.",
    )
    parser.add_argument(
        "--allow-source-sha-mismatch",
        action="store_true",
        help="Planning-only override for tests/forensics; default fails closed on C-089/PR65 SHA drift.",
    )
    args = parser.parse_args(argv)

    matrix = build_matrix(
        source_archive=args.source_archive,
        pr65_archive=args.pr65_archive,
        output_dir=args.output_dir,
        anatomy_json=args.anatomy_json,
        c089_trace=args.c089_trace,
        pr65_trace=args.pr65_trace,
        specs=parse_specs(args.spec),
        positive_trace_only=not args.allow_negative_trace_pairs,
        expected_source_sha256=None if args.allow_source_sha_mismatch else EXPECTED_C089_SHA256,
        expected_pr65_sha256=None if args.allow_source_sha_mismatch else EXPECTED_PR65_SHA256,
        expected_pr65_head_sha=None if args.allow_source_sha_mismatch else EXPECTED_PR65_HEAD_SHA,
    )
    print(json.dumps(matrix, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
