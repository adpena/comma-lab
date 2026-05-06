#!/usr/bin/env python3
"""Deterministic OWV3 byte-plan sweep.

This sweep is deliberately byte-only. It builds candidate archives from the
CUDA-authored OWV3 sensitivity map and records archive bytes/SHA/knobs, but it
does not run local CPU/MPS scorer paths or make score claims. Any selected
candidate is only "byte-feasible pending CUDA auth eval" until evaluated via
experiments/contest_auth_eval.py on the exact archive bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shlex
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_SENSITIVITY_MAP = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "lane_g_v3_owv3_fisher_lightning_20260430_codex_r2"
    / "owv3_sensitivity_map.pt"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments" / "results" / "lane_g_v3_owv3_byte_plan_sweep"
)
DEFAULT_PAIRED_PFP16_ARCHIVE = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "lane_g_v3_pfp16"
    / "archive_lane_g_v3_pfp16.zip"
)
PFP16_A_PLUS_PLUS_SCORE = 1.043987524793892  # [empirical:reports/raw/pfp16_a_plus_plus_deploy_baseline_freeze_20260501] PFP16 A++ deploy baseline contest-CUDA T4
PFP16_A_PLUS_PLUS_POSENET_DIST = 0.00346442
PFP16_A_PLUS_PLUS_SEGNET_DIST = 0.00400656
R5_COMPONENT_RELATIVE_CAP = 1.002
R5_REQUIRED_SAMPLES = 600
R5_FAILED_EXACT_CUDA_T4_REFERENCE = {
    "candidate_id": "owv3_0047_bbr0p67_protect0p00135_aggr1em05",
    "score_recomputed_from_components": 1.0373951773937642,
    "avg_posenet_dist": 0.0031739,
    "avg_segnet_dist": 0.0040215,
    "archive_size_bytes": 686468,
    "archive_sha256": "16ab95220c8add11b0bc40fb632bc8421f8bb8ad1cfba145f0b6058075237518",
    "paired_pfp16_score_recomputed_from_components": 1.037045485927815,
    "paired_pfp16_avg_posenet_dist": 0.00316404,
    "paired_pfp16_avg_segnet_dist": 0.00401966,
    "paired_pfp16_archive_size_bytes": 686635,
    "paired_pfp16_archive_sha256": "0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f",
    "lane_status": "COMPONENT_GATE_REVIEW_REQUIRED",
    "promotion_eligible": False,
}
R6_FAILED_EXACT_CUDA_T4_REFERENCE = {
    "candidate_id": "owv3_0076_bbr0p65_protect0p0013_aggr1em05",
    "score_recomputed_from_components": 1.0393166493980681,
    "avg_posenet_dist": 0.00323147,
    "avg_segnet_dist": 0.00402421,
    "archive_size_bytes": 686531,
    "archive_sha256": "9f7528bade11bf9cdf3df68f8073d11f196a6d5f48475a8680c21fb58c878c91",
    "paired_pfp16_score_recomputed_from_components": 1.037045485927815,
    "paired_pfp16_avg_posenet_dist": 0.00316404,
    "paired_pfp16_avg_segnet_dist": 0.00401966,
    "paired_pfp16_archive_size_bytes": 686635,
    "paired_pfp16_archive_sha256": "0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f",
    "posenet_relative_to_paired_pfp16": 1.0213113614240024,
    "segnet_relative_to_paired_pfp16": 1.0011319365319455,
    "lane_status": "REGRESSION_AND_COMPONENT_GATE_REVIEW_REQUIRED",
    "promotion_eligible": False,
}
LIGHTNING_REPO_DIR = "/teamspace/studios/this_studio/pact"
LIGHTNING_UPSTREAM_DIR = "/teamspace/studios/this_studio/upstream"
LIGHTNING_PYTHON_BIN = (
    "/teamspace/studios/this_studio/"
    "pact_pfp16_exact_20260430T1625Z/.venv/bin/python"
)

PRESETS: dict[str, dict[str, tuple[float, ...]]] = {
    "baseline": {
        "bit_budget_ratios": (0.7,),
        "protect_thresholds": (1e-3,),
        "aggressive_thresholds": (1e-5,),
    },
    "frontier": {
        "bit_budget_ratios": (0.70, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64, 0.60, 0.50),
        "protect_thresholds": (
            0.0010,
            0.0013,
            0.00135,
            0.0014,
            0.00145,
            0.0015,
            0.00155,
            0.0016,
            0.00165,
            0.0017,
            0.0018,
            0.0019,
            0.0020,
            0.0030,
            0.0050,
        ),
        "aggressive_thresholds": (1e-5,),
    },
    "broad": {
        "bit_budget_ratios": (0.70, 0.60, 0.50, 0.40, 0.30, 0.20),
        "protect_thresholds": (
            0.0010,
            0.0015,
            0.0020,
            0.0030,
            0.0050,
            0.0075,
            0.0100,
            0.0150,
            0.0200,
            0.0300,
            0.0500,
            0.0750,
            0.1000,
        ),
        "aggressive_thresholds": (1e-5,),
    },
}


@dataclass(frozen=True)
class CandidateKnobs:
    bit_budget_ratio: float
    protect_threshold: float
    aggressive_threshold: float
    fallback_action: str = "keep_asym"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_float_list(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    values: list[float] = []
    for part in raw.replace("\n", ",").split(","):
        token = part.strip()
        if not token:
            continue
        value = float(token)
        if not (value == value and abs(value) != float("inf")):
            raise argparse.ArgumentTypeError(f"non-finite float in list: {token!r}")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("float list must contain at least one value")
    return tuple(values)


def _dedupe_preserve(values: Iterable[float]) -> tuple[float, ...]:
    out: list[float] = []
    seen: set[str] = set()
    for value in values:
        key = f"{float(value):.17g}"
        if key in seen:
            continue
        seen.add(key)
        out.append(float(value))
    return tuple(out)


def resolve_grid(
    *,
    preset: str,
    bit_budget_ratios: tuple[float, ...] | None = None,
    protect_thresholds: tuple[float, ...] | None = None,
    aggressive_thresholds: tuple[float, ...] | None = None,
    fallback_action: str = "keep_asym",
) -> list[CandidateKnobs]:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; expected one of {sorted(PRESETS)}")
    base = PRESETS[preset]
    ratios = _dedupe_preserve(bit_budget_ratios or base["bit_budget_ratios"])
    protects = _dedupe_preserve(protect_thresholds or base["protect_thresholds"])
    aggressives = _dedupe_preserve(aggressive_thresholds or base["aggressive_thresholds"])
    grid: list[CandidateKnobs] = []
    for ratio in ratios:
        if ratio <= 0.0 or ratio >= 1.0:
            raise ValueError(f"bit_budget_ratio must be in (0, 1), got {ratio}")
        for protect in protects:
            if protect <= 0.0:
                raise ValueError(f"protect_threshold must be > 0, got {protect}")
            for aggressive in aggressives:
                if aggressive < 0.0:
                    raise ValueError(
                        f"aggressive_threshold must be >= 0, got {aggressive}"
                    )
                if aggressive >= protect:
                    raise ValueError(
                        "aggressive_threshold must be < protect_threshold; "
                        f"got aggressive={aggressive}, protect={protect}"
                    )
                grid.append(
                    CandidateKnobs(
                        bit_budget_ratio=ratio,
                        protect_threshold=protect,
                        aggressive_threshold=aggressive,
                        fallback_action=fallback_action,
                    )
                )
    return grid


def float_token(value: float) -> str:
    token = f"{float(value):.12g}".replace("+", "")
    return token.replace("-", "m").replace(".", "p")


def candidate_id(index: int, knobs: CandidateKnobs) -> str:
    return (
        f"owv3_{index:04d}"
        f"_bbr{float_token(knobs.bit_budget_ratio)}"
        f"_protect{float_token(knobs.protect_threshold)}"
        f"_aggr{float_token(knobs.aggressive_threshold)}"
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_custody(path: Path) -> dict[str, object]:
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
        "sha256": _sha256_path(path) if exists else "",
    }


def build_component_sensitivity_byte_plan_manifest(
    *,
    sensitivity_map: Path,
    preset: str,
    bit_budget_ratios: tuple[float, ...] | None = None,
    protect_thresholds: tuple[float, ...] | None = None,
    aggressive_thresholds: tuple[float, ...] | None = None,
    fallback_action: str = "keep_asym",
    frontier_comparator_bytes: int | None = None,
    frontier_comparator_sha256: str | None = None,
    frontier_comparator_label: str = "PFP16_A++",
    allow_non_authoritative: bool = False,
    limit: int | None = None,
) -> dict[str, object]:
    """Build a fast local byte-plan manifest without loading models or archives.

    The manifest is a DX and planning surface for the component-sensitivity
    allocator hidden gem. It records deterministic candidate knobs and fallback
    accounting, but it deliberately emits no archive bytes and cannot unlock
    exact-eval dispatch.
    """

    grid = resolve_grid(
        preset=preset,
        bit_budget_ratios=bit_budget_ratios,
        protect_thresholds=protect_thresholds,
        aggressive_thresholds=aggressive_thresholds,
        fallback_action=fallback_action,
    )
    if limit is not None:
        grid = grid[: int(limit)]
    candidates = [
        {
            "candidate_id": candidate_id(index, knobs),
            "candidate_index": index,
            "knobs": asdict(knobs),
            "score_status": "not_evaluated_cuda_auth_required",
            "archive_status": "not_built_manifest_only",
            "ready_for_exact_eval_dispatch": False,
        }
        for index, knobs in enumerate(grid)
    ]
    fallback_accounting = {
        "fallback_action": fallback_action,
        "keep_asym": {
            "charged_byte_policy": "preserve source asym bytes where sensitivity contract cannot lower safely",
            "promotion_eligible_before_eval": fallback_action == "keep_asym",
        },
        "diagnostic_fp16": {
            "charged_byte_policy": "diagnostic fallback bytes must remain non-promotable",
            "promotion_eligible_before_eval": False,
        },
        "error": {
            "charged_byte_policy": "fail closed on unsupported sensitivity coverage",
            "promotion_eligible_before_eval": False,
        },
    }
    blockers = [
        "manifest_only_no_archive_bytes",
        "cuda_auth_eval_required_for_score",
        "component_balanced_sensitivity_required_before_promotion",
    ]
    if not sensitivity_map.is_file():
        blockers.append("missing_sensitivity_map")
    if not allow_non_authoritative:
        blockers.append("authoritative_cuda_sensitivity_required")
    return {
        "format": "component_sensitivity_byte_plan_manifest_v1",
        "created_iso": utc_now_iso(),
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "evidence_grade": "planning_manifest_only",
        "hidden_gem_key": "component_sensitivity_byte_allocator",
        "sensitivity_map": _file_custody(sensitivity_map),
        "frontier_comparator": {
            "label": frontier_comparator_label,
            "archive_bytes": frontier_comparator_bytes,
            "archive_sha256": frontier_comparator_sha256 or "",
        },
        "grid": {
            "preset": preset,
            "candidate_count": len(grid),
            "bit_budget_ratios": sorted({k.bit_budget_ratio for k in grid}, reverse=True),
            "protect_thresholds": sorted({k.protect_threshold for k in grid}),
            "aggressive_thresholds": sorted({k.aggressive_threshold for k in grid}),
            "fallback_action": fallback_action,
        },
        "fallback_action_accounting": fallback_accounting,
        "candidates": candidates,
        "dispatch_blockers": blockers,
        "next_safe_actions": [
            "Run the full OWV3 sweep only after input sensitivity custody is authoritative.",
            "Treat manifest candidates as planning-only until archive bytes are built deterministically.",
            "Use exact CUDA auth eval and component gates before any score or promotion claim.",
        ],
    }


def classify_candidate(
    *,
    archive_bytes: int,
    frontier_bytes: int,
    lane_g_v3_bytes: int,
) -> dict[str, object]:
    frontier_delta = int(archive_bytes) - int(frontier_bytes)
    lane_delta = int(archive_bytes) - int(lane_g_v3_bytes)
    byte_feasible = frontier_delta <= 0
    return {
        "archive_bytes": int(archive_bytes),
        "frontier_bytes": int(frontier_bytes),
        "frontier_delta_bytes": frontier_delta,
        "lane_g_v3_bytes": int(lane_g_v3_bytes),
        "lane_g_v3_delta_bytes": lane_delta,
        "byte_feasible_vs_frontier": byte_feasible,
        "selection_status": (
            "byte_feasible_pending_cuda_auth_eval"
            if byte_feasible
            else "byte_infeasible_vs_frontier"
        ),
        "score_status": "not_evaluated_cuda_auth_required",
    }


def select_best_byte_candidate(rows: list[dict]) -> dict | None:
    feasible = [
        row for row in rows
        if row.get("build_status") == "ok"
        and row.get("byte_classification", {}).get("byte_feasible_vs_frontier")
    ]
    if not feasible:
        return None
    return sorted(
        feasible,
        key=lambda row: (
            int(row["byte_classification"]["frontier_bytes"])
            - int(row["archive"]["size_bytes"]),
            -float(row["knobs"]["bit_budget_ratio"]),
            float(row["knobs"]["protect_threshold"]),
            float(row["knobs"]["aggressive_threshold"]),
            str(row["candidate_id"]),
        ),
    )[0]


def _owv2_low_bit_channels(row: dict) -> int:
    return int(
        row.get("owv3_byte_plan", {})
        .get("action_counts", {})
        .get("owv2_low_bit_channels", 0)
        or 0
    )


def _frontier_margin(row: dict) -> int:
    return abs(int(row.get("byte_classification", {}).get("frontier_delta_bytes", 0)))


def _is_r5_promotion_eligible(row: dict) -> bool:
    plan = row.get("owv3_byte_plan", {})
    if plan.get("promotion_eligible") is not True:
        return False
    if plan.get("fallback_action") != "keep_asym":
        return False
    action_counts = plan.get("action_counts", {})
    if int(action_counts.get("diagnostic_fp16_layers", 0) or 0) != 0:
        return False
    return True


def select_r5_segnet_conservative_candidates(
    rows: list[dict],
    *,
    reference_candidate_id: str,
    limit: int = 5,
    max_bit_budget_ratio_drop: float = 0.05,
) -> list[dict]:
    """Rank byte-feasible R5 candidates less aggressive than a failed R4 row.

    R4 failed a predeclared SegNet relative gate despite a good total score.
    The next candidate should therefore be selected by a paired calibration
    rule: keep byte feasibility, reduce the number of OWV2-low-bit channels
    versus the exact-evaluated failed candidate, and prefer the smallest
    conservative move before spending another exact CUDA eval.
    """
    reference = next(
        (row for row in rows if row.get("candidate_id") == reference_candidate_id),
        None,
    )
    if reference is None:
        raise ValueError(f"reference candidate not found: {reference_candidate_id}")
    ref_low = _owv2_low_bit_channels(reference)
    ref_bbr = float(reference.get("knobs", {}).get("bit_budget_ratio", 0.0))
    min_bbr = max(0.0, ref_bbr - float(max_bit_budget_ratio_drop))
    out: list[dict] = []
    for row in rows:
        if row.get("build_status") != "ok":
            continue
        if not _is_r5_promotion_eligible(row):
            continue
        if not row.get("byte_classification", {}).get("byte_feasible_vs_frontier"):
            continue
        if row.get("candidate_id") == reference_candidate_id:
            continue
        low = _owv2_low_bit_channels(row)
        if low >= ref_low:
            continue
        candidate_bbr = float(row.get("knobs", {}).get("bit_budget_ratio", 0.0))
        if candidate_bbr < min_bbr:
            continue
        cloned = json.loads(json.dumps(row, sort_keys=True))
        cloned["r5_paired_calibration"] = {
            "reference_candidate_id": reference_candidate_id,
            "reference_owv2_low_bit_channels": ref_low,
            "candidate_owv2_low_bit_channels": low,
            "owv2_low_bit_channel_reduction": ref_low - low,
            "reference_bit_budget_ratio": ref_bbr,
            "candidate_bit_budget_ratio": candidate_bbr,
            "max_bit_budget_ratio_drop": float(max_bit_budget_ratio_drop),
            "min_allowed_bit_budget_ratio": min_bbr,
            "policy": (
                "byte-feasible SegNet-conservative neighbor; requires paired "
                "same-run PFP16 calibration and exact CUDA component gates"
            ),
            "score_status": "not_evaluated_cuda_auth_required",
        }
        out.append(cloned)

    return sorted(
        out,
        key=lambda row: (
            int(row["r5_paired_calibration"]["owv2_low_bit_channel_reduction"]),
            -float(row["r5_paired_calibration"]["candidate_bit_budget_ratio"]),
            _frontier_margin(row),
            str(row.get("candidate_id")),
        ),
    )[: max(0, int(limit))]


def select_r6_segnet_conservative_candidates(
    rows: list[dict],
    *,
    reference_candidate_id: str,
    limit: int = 5,
    max_bit_budget_ratio_drop: float = 0.05,
    failed_exact_reference: dict[str, object] | None = None,
) -> list[dict]:
    """Rank byte-feasible R6 candidates after an exact-evaluated R5 gate miss.

    R5 was worse than its same-run PFP16 calibration and tripped the SegNet
    component gate. R6 therefore uses the failed R5 exact-eval candidate as
    the reference, requires a strictly lower OWV2-low-bit channel count, and
    records the failed exact metrics so a future queue packet cannot drift into
    a promotion claim without paired CUDA adjudication.
    """
    reference = next(
        (row for row in rows if row.get("candidate_id") == reference_candidate_id),
        None,
    )
    if reference is None:
        raise ValueError(f"reference candidate not found: {reference_candidate_id}")
    ref_low = _owv2_low_bit_channels(reference)
    ref_bbr = float(reference.get("knobs", {}).get("bit_budget_ratio", 0.0))
    min_bbr = max(0.0, ref_bbr - float(max_bit_budget_ratio_drop))
    exact_reference = json.loads(json.dumps(
        failed_exact_reference or R5_FAILED_EXACT_CUDA_T4_REFERENCE,
        sort_keys=True,
    ))
    out: list[dict] = []
    for row in rows:
        if row.get("build_status") != "ok":
            continue
        if not _is_r5_promotion_eligible(row):
            continue
        if not row.get("byte_classification", {}).get("byte_feasible_vs_frontier"):
            continue
        if row.get("candidate_id") == reference_candidate_id:
            continue
        low = _owv2_low_bit_channels(row)
        if low >= ref_low:
            continue
        candidate_bbr = float(row.get("knobs", {}).get("bit_budget_ratio", 0.0))
        if candidate_bbr < min_bbr:
            continue
        cloned = json.loads(json.dumps(row, sort_keys=True))
        cloned["r6_paired_calibration"] = {
            "reference_candidate_id": reference_candidate_id,
            "reference_owv2_low_bit_channels": ref_low,
            "candidate_owv2_low_bit_channels": low,
            "owv2_low_bit_channel_reduction": ref_low - low,
            "reference_bit_budget_ratio": ref_bbr,
            "candidate_bit_budget_ratio": candidate_bbr,
            "max_bit_budget_ratio_drop": float(max_bit_budget_ratio_drop),
            "min_allowed_bit_budget_ratio": min_bbr,
            "failed_r5_exact_cuda_t4_reference": exact_reference,
            "policy": (
                "R6 after exact-evaluated R5 SegNet gate miss; requires lower "
                "OWV2-low-bit channel count than failed R5, same-run PFP16 "
                "calibration, exact CUDA/T4 component gates, and no promotion "
                "from byte-only evidence"
            ),
            "score_status": "not_evaluated_cuda_auth_required",
            "promotion_eligible_before_eval": False,
        }
        out.append(cloned)

    return sorted(
        out,
        key=lambda row: (
            int(row["r6_paired_calibration"]["owv2_low_bit_channel_reduction"]),
            -float(row["r6_paired_calibration"]["candidate_bit_budget_ratio"]),
            _frontier_margin(row),
            str(row.get("candidate_id")),
        ),
    )[: max(0, int(limit))]


def select_r7_pose_balanced_candidates(
    rows: list[dict],
    *,
    reference_candidate_id: str,
    limit: int = 5,
    min_bit_budget_ratio: float | None = None,
    max_owv2_low_bit_channels: int | None = None,
    failed_exact_reference: dict[str, object] | None = None,
    excluded_candidate_ids: Iterable[str] = (),
) -> list[dict]:
    """Rank scalar-threshold R7 candidates after an R6 PoseNet gate miss.

    R6 fixed the paired SegNet relative gate but failed PoseNet after lowering
    the bit budget on the remaining OWV2-coded channels. A scalar-threshold R7
    is only admissible if it keeps at least R6's protected-channel count and
    does not lower the remaining OWV2 bit budget further. If this returns no
    rows, the next step is component-balanced sensitivity evidence rather than
    another blind threshold spend.
    """
    reference = next(
        (row for row in rows if row.get("candidate_id") == reference_candidate_id),
        None,
    )
    if reference is None:
        raise ValueError(f"reference candidate not found: {reference_candidate_id}")
    ref_low = _owv2_low_bit_channels(reference)
    ref_bbr = float(reference.get("knobs", {}).get("bit_budget_ratio", 0.0))
    min_bbr = ref_bbr if min_bit_budget_ratio is None else float(min_bit_budget_ratio)
    max_low = ref_low if max_owv2_low_bit_channels is None else int(max_owv2_low_bit_channels)
    exact_reference = json.loads(json.dumps(
        failed_exact_reference or R6_FAILED_EXACT_CUDA_T4_REFERENCE,
        sort_keys=True,
    ))
    excluded = {
        str(reference_candidate_id),
        str(R5_FAILED_EXACT_CUDA_T4_REFERENCE["candidate_id"]),
        str(R6_FAILED_EXACT_CUDA_T4_REFERENCE["candidate_id"]),
        *(str(cid) for cid in excluded_candidate_ids),
    }

    out: list[dict] = []
    for row in rows:
        cid = str(row.get("candidate_id"))
        if row.get("build_status") != "ok":
            continue
        if cid in excluded:
            continue
        if not _is_r5_promotion_eligible(row):
            continue
        if not row.get("byte_classification", {}).get("byte_feasible_vs_frontier"):
            continue
        low = _owv2_low_bit_channels(row)
        if low > max_low:
            continue
        candidate_bbr = float(row.get("knobs", {}).get("bit_budget_ratio", 0.0))
        if candidate_bbr < min_bbr:
            continue
        cloned = json.loads(json.dumps(row, sort_keys=True))
        cloned["r7_pose_balanced_calibration"] = {
            "reference_candidate_id": reference_candidate_id,
            "reference_owv2_low_bit_channels": ref_low,
            "candidate_owv2_low_bit_channels": low,
            "max_owv2_low_bit_channels": max_low,
            "owv2_low_bit_channel_delta_vs_reference": low - ref_low,
            "reference_bit_budget_ratio": ref_bbr,
            "candidate_bit_budget_ratio": candidate_bbr,
            "min_bit_budget_ratio": min_bbr,
            "failed_r6_exact_cuda_t4_reference": exact_reference,
            "policy": (
                "R7 after exact-evaluated R6 PoseNet gate miss; scalar "
                "threshold candidates must preserve at least R6's SegNet-side "
                "protected-channel count and must not lower the remaining "
                "OWV2 bit budget below R6. Empty output means wait for "
                "component-balanced PoseNet/SegNet sensitivity rather than "
                "spending exact eval on another blind threshold."
            ),
            "score_status": "not_evaluated_cuda_auth_required",
            "promotion_eligible_before_eval": False,
        }
        out.append(cloned)

    return sorted(
        out,
        key=lambda row: (
            -float(row["r7_pose_balanced_calibration"]["candidate_bit_budget_ratio"]),
            abs(int(row["r7_pose_balanced_calibration"]["owv2_low_bit_channel_delta_vs_reference"])),
            _frontier_margin(row),
            str(row.get("candidate_id")),
        ),
    )[: max(0, int(limit))]


def _shell_join(tokens: list[str]) -> str:
    return " ".join(shlex.quote(str(token)) for token in tokens)


def _lightning_repo_path(path: str | Path) -> str:
    raw = str(path)
    if raw.startswith("<"):
        return raw
    candidate = Path(raw)
    if candidate.is_absolute():
        try:
            rel = candidate.resolve().relative_to(REPO_ROOT)
        except (OSError, ValueError):
            return f"<remote path for {raw}>"
    else:
        rel = candidate
    return str(PurePosixPath(LIGHTNING_REPO_DIR) / PurePosixPath(rel.as_posix()))


def r5_promotion_gate_policy() -> dict[str, object]:
    return {
        "score_status": "not_evaluated_cuda_auth_required",
        "promotion_eligible_before_eval": False,
        "required_device": "cuda",
        "required_gpu_t4_match": True,
        "required_samples": R5_REQUIRED_SAMPLES,
        "paired_pfp16_calibration_required": True,
        "paired_pfp16_policy": (
            "Run PFP16 A++ exact eval on the same Lightning/T4 runner/toolchain "
            "before adjudicating OWV3 R5. Use that JSON's component metrics as "
            "the relative-gate reference."
        ),
        "component_reference_label": "paired_pfp16_a_plus_plus_t4_same_runner",
        "max_posenet_relative_to_paired_pfp16": R5_COMPONENT_RELATIVE_CAP,
        "max_segnet_relative_to_paired_pfp16": R5_COMPONENT_RELATIVE_CAP,
        "adjudication_required": True,
    }


def _contest_auth_eval_command(*, archive_path: str, work_dir: str) -> list[str]:
    return [
        ".venv/bin/python",
        "experiments/contest_auth_eval.py",
        "--archive",
        archive_path,
        "--inflate-sh",
        "submissions/robust_current/inflate.sh",
        "--upstream-dir",
        "upstream",
        "--device",
        "cuda",
        "--keep-work-dir",
        "--work-dir",
        work_dir,
    ]


def _adjudication_command_template(
    *,
    archive_path: str,
    work_dir: str,
    baseline_archive_bytes: int,
) -> list[str]:
    return [
        ".venv/bin/python",
        "scripts/adjudicate_contest_auth_eval.py",
        "--contest-json",
        str(Path(work_dir) / "contest_auth_eval.json"),
        "--provenance",
        str(Path(work_dir) / "adjudication_provenance.json"),
        "--archive",
        archive_path,
        "--result-copy",
        str(Path(work_dir) / "contest_auth_eval.adjudicated.json"),
        "--baseline-score",
        "<paired_pfp16_score_recomputed_from_components>",
        "--predicted-band",
        "0.0",
        "<paired_pfp16_score_recomputed_from_components>",
        "--regression-threshold",
        "<paired_pfp16_score_recomputed_from_components>",
        "--delta-key",
        "score_delta_vs_paired_pfp16",
        "--component-reference-label",
        "paired_pfp16_a_plus_plus_t4_same_runner",
        "--required-device",
        "cuda",
        "--required-samples",
        str(R5_REQUIRED_SAMPLES),
        "--max-sane-score",
        "10.0",
        "--baseline-archive-bytes",
        str(int(baseline_archive_bytes)),
        "--baseline-posenet-dist",
        "<paired_pfp16_avg_posenet_dist>",
        "--baseline-segnet-dist",
        "<paired_pfp16_avg_segnet_dist>",
        "--max-posenet-relative",
        str(R5_COMPONENT_RELATIVE_CAP),
        "--max-segnet-relative",
        str(R5_COMPONENT_RELATIVE_CAP),
    ]


def _lightning_exact_eval_command_template(
    *,
    candidate_id: str,
    archive_path: str,
    archive_sha256: str,
    archive_size_bytes: int,
    baseline_archive_bytes: int,
    lane: str,
) -> list[str]:
    return [
        ".venv/bin/python",
        "scripts/launch_lightning_batch_job.py",
        "exact-eval",
        "--job-name",
        f"{candidate_id}_exact_cuda_<UTC>",
        "--archive",
        archive_path,
        "--repo-dir",
        LIGHTNING_REPO_DIR,
        "--upstream-dir",
        LIGHTNING_UPSTREAM_DIR,
        "--machine",
        "T4",
        "--studio",
        "lossy-compression-challenge",
        "--teamspace",
        "comma-lab",
        "--python-bin",
        LIGHTNING_PYTHON_BIN,
        "--expected-archive-sha256",
        archive_sha256,
        "--expected-archive-size-bytes",
        str(int(archive_size_bytes)),
        "--adjudicate",
        "--baseline-score",
        "<paired_pfp16_score_recomputed_from_components>",
        "--predicted-band",
        "0.0",
        "<paired_pfp16_score_recomputed_from_components>",
        "--regression-threshold",
        "<paired_pfp16_score_recomputed_from_components>",
        "--baseline-archive-bytes",
        str(int(baseline_archive_bytes)),
        "--baseline-posenet-dist",
        "<paired_pfp16_avg_posenet_dist>",
        "--baseline-segnet-dist",
        "<paired_pfp16_avg_segnet_dist>",
        "--max-posenet-relative",
        str(R5_COMPONENT_RELATIVE_CAP),
        "--max-segnet-relative",
        str(R5_COMPONENT_RELATIVE_CAP),
        "--component-reference-label",
        "paired_pfp16_a_plus_plus_t4_same_runner",
        "--queue-metadata",
        f"lane={lane}",
        "--queue-metadata",
        f"candidate_id={candidate_id}",
        "--queue-metadata",
        "paired_pfp16_calibration_required=true",
        "--queue-metadata",
        "required_gpu_t4_match=true",
    ]


def build_r5_paired_pfp16_calibration_queue(
    *,
    paired_pfp16_archive: Path,
    frontier_archive_sha256: str,
    frontier_archive_bytes: int,
) -> dict[str, object]:
    archive_path = str(paired_pfp16_archive)
    lightning_archive_path = _lightning_repo_path(paired_pfp16_archive)
    local_work_dir = (
        "experiments/results/lightning_batch/"
        "pfp16_a_plus_plus_r5_paired_calibration_<UTC>"
    )
    contest_cmd = _contest_auth_eval_command(
        archive_path=archive_path,
        work_dir=f"{local_work_dir}/eval_work",
    )
    lightning_cmd = [
        ".venv/bin/python",
        "scripts/launch_lightning_batch_job.py",
        "exact-eval",
        "--job-name",
        "pfp16_a_plus_plus_r5_paired_calibration_<UTC>",
        "--archive",
        lightning_archive_path,
        "--repo-dir",
        LIGHTNING_REPO_DIR,
        "--upstream-dir",
        LIGHTNING_UPSTREAM_DIR,
        "--machine",
        "T4",
        "--studio",
        "lossy-compression-challenge",
        "--teamspace",
        "comma-lab",
        "--python-bin",
        LIGHTNING_PYTHON_BIN,
        "--expected-archive-sha256",
        frontier_archive_sha256,
        "--expected-archive-size-bytes",
        str(int(frontier_archive_bytes)),
        "--adjudicate",
        "--baseline-score",
        str(PFP16_A_PLUS_PLUS_SCORE),
        "--predicted-band",
        "0.0",
        "10.0",
        "--regression-threshold",
        "10.0",
        "--baseline-archive-bytes",
        str(int(frontier_archive_bytes)),
        "--component-reference-label",
        "pfp16_a_plus_plus_self_r5_calibration",
        "--queue-metadata",
        "lane=pfp16_a_plus_plus_r5_paired_calibration",
        "--queue-metadata",
        "required_gpu_t4_match=true",
    ]
    return {
        "lane": "pfp16_a_plus_plus_r5_paired_calibration",
        "archive_path": archive_path,
        "lightning_archive_path": lightning_archive_path,
        "expected_archive_sha256": frontier_archive_sha256,
        "expected_archive_size_bytes": int(frontier_archive_bytes),
        "required_before_r5": True,
        "contest_auth_eval_command": contest_cmd,
        "contest_auth_eval_command_string": _shell_join(contest_cmd),
        "lightning_exact_eval_command_template": lightning_cmd,
        "lightning_exact_eval_command_template_string": _shell_join(lightning_cmd),
        "outputs_required_for_r5_adjudication": [
            "contest_auth_eval.json:score_recomputed_from_components",
            "contest_auth_eval.json:avg_posenet_dist",
            "contest_auth_eval.json:avg_segnet_dist",
            "contest_auth_eval.json:provenance.gpu_t4_match == true",
            "contest_auth_eval.json:provenance.device == cuda",
            f"contest_auth_eval.json:n_samples == {R5_REQUIRED_SAMPLES}",
        ],
    }


def build_r5_exact_eval_queue_entry(
    row: dict,
    *,
    archive_path: str,
    output_dir: Path,
    baseline_archive_bytes: int,
    lane: str = "owv3_r5_segnet_conservative",
    calibration_key: str = "r5_paired_calibration",
) -> dict[str, object]:
    if calibration_key not in row:
        raise ValueError(f"exact eval queue entries require {calibration_key} metadata")
    archive = row.get("archive", {})
    cid = str(row.get("candidate_id"))
    archive_sha = str(archive.get("sha256"))
    archive_bytes = int(archive.get("size_bytes"))
    work_dir = str(output_dir / "exact_eval_queue" / cid)
    lightning_archive_path = _lightning_repo_path(archive_path)
    contest_cmd = _contest_auth_eval_command(
        archive_path=archive_path,
        work_dir=work_dir,
    )
    adjudicate_cmd = _adjudication_command_template(
        archive_path=archive_path,
        work_dir=work_dir,
        baseline_archive_bytes=baseline_archive_bytes,
    )
    lightning_cmd = _lightning_exact_eval_command_template(
        candidate_id=cid,
        archive_path=lightning_archive_path,
        archive_sha256=archive_sha,
        archive_size_bytes=archive_bytes,
        baseline_archive_bytes=baseline_archive_bytes,
        lane=lane,
    )
    return {
        "candidate_id": cid,
        "archive_path": archive_path,
        "lightning_archive_path": lightning_archive_path,
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_bytes,
        calibration_key: row[calibration_key],
        "promotion_gates": r5_promotion_gate_policy(),
        "contest_auth_eval_command": contest_cmd,
        "contest_auth_eval_command_string": _shell_join(contest_cmd),
        "adjudication_command_template_after_paired_pfp16": adjudicate_cmd,
        "adjudication_command_template_after_paired_pfp16_string": _shell_join(adjudicate_cmd),
        "lightning_exact_eval_command_template_after_paired_pfp16": lightning_cmd,
        "lightning_exact_eval_command_template_after_paired_pfp16_string": _shell_join(lightning_cmd),
    }


def _load_builder_module():
    builder_path = REPO_ROOT / "experiments" / "build_lane_g_v3_owv3_stack.py"
    spec = importlib.util.spec_from_file_location("_owv3_stack_builder", builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load OWV3 builder from {builder_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _archive_target(archive_dir: Path, cid: str) -> Path:
    return archive_dir / f"{cid}.zip"


def run_sweep(args: argparse.Namespace) -> dict:
    builder = _load_builder_module()
    t0 = time.monotonic()
    if args.selection_mode in {"r5-segnet-conservative", "r6-segnet-conservative"} and not args.r5_reference_candidate_id:
        raise ValueError(
            f"--selection-mode {args.selection_mode} requires --r5-reference-candidate-id"
        )
    if args.selection_mode == "r7-pose-balanced" and not args.r6_reference_candidate_id:
        raise ValueError(
            "--selection-mode r7-pose-balanced requires --r6-reference-candidate-id"
        )
    if args.r5_write_candidate_archives and not args.r5_reference_candidate_id:
        raise ValueError("--r5-write-candidate-archives requires --r5-reference-candidate-id")
    paired_pfp16_archive = args.r5_paired_pfp16_archive.resolve()
    if (
        args.r5_reference_candidate_id
        or args.r6_reference_candidate_id
    ) and not paired_pfp16_archive.is_file():
        raise FileNotFoundError(f"paired PFP16 archive not found: {paired_pfp16_archive}")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty; pass --overwrite or choose a new --output-dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = output_dir / "archives"
    if args.archive_policy in {"all", "byte-feasible"}:
        archive_dir.mkdir(parents=True, exist_ok=True)

    grid = resolve_grid(
        preset=args.preset,
        bit_budget_ratios=parse_float_list(args.bit_budget_ratios),
        protect_thresholds=parse_float_list(args.protect_thresholds),
        aggressive_thresholds=parse_float_list(args.aggressive_thresholds),
        fallback_action=args.fallback_action,
    )
    if args.limit is not None:
        grid = grid[: int(args.limit)]

    print("=== OWV3 byte-plan sweep ===")
    print(f"  sensitivity map: {args.sensitivity_map}")
    print(f"  output dir: {output_dir}")
    print(f"  candidates: {len(grid)}")
    print(
        "  score policy: byte-only; CUDA auth eval required before any score claim"
    )

    from tac.owv3_sensitivity_weighted import (
        decode_owv3_archive,
        encode_owv3_archive,
        inspect_owv3_archive,
    )
    from tac.renderer_export import load_renderer_checkpoint
    from tac.sensitivity_map import (
        load_sensitivity_map,
        require_authoritative_device,
        validate_sensitivity_map_for_model,
    )

    anchor_meta = builder._verify_anchor()
    lane_g_v3_archive_bytes = int(anchor_meta["archive"]["size_bytes"])
    frontier_bytes = int(args.frontier_comparator_bytes)
    model = load_renderer_checkpoint(str(builder.LANE_G_V3_RENDERER))
    model.eval()
    sensitivities, sensitivity_metadata = load_sensitivity_map(args.sensitivity_map)
    source_device = (
        sensitivity_metadata.get("source_device")
        or sensitivity_metadata.get("device")
    )
    if not args.allow_non_authoritative:
        require_authoritative_device(source_device)
    sens_stats = validate_sensitivity_map_for_model(
        sensitivities,
        model,
        require_all_conv=True,
    )

    masks_bytes = builder.LANE_G_V3_MASKS.read_bytes()
    poses_bytes = builder.LANE_G_V3_POSES.read_bytes()
    asym_renderer_bytes = int(builder.LANE_G_V3_RENDERER.stat().st_size)

    rows: list[dict] = []
    selected_renderer_blob: bytes | None = None
    selected_archive_blob: bytes | None = None
    best_seen_row: dict | None = None
    best_seen_renderer_blob: bytes | None = None
    best_seen_archive_blob: bytes | None = None
    renderer_blobs_by_candidate_id: dict[str, bytes] = {}
    archive_blobs_by_candidate_id: dict[str, bytes] = {}

    for index, knobs in enumerate(grid):
        cid = candidate_id(index, knobs)
        row: dict = {
            "format": "owv3_byte_plan_candidate_v1",
            "candidate_id": cid,
            "candidate_index": index,
            "knobs": asdict(knobs),
            "build_status": "ok",
            "evidence_label": "byte-only-pending-cuda-auth-eval",
            "score_status": "not_evaluated_cuda_auth_required",
            "selection_basis": "archive_bytes_only_no_score_claim",
        }
        try:
            renderer_blob = encode_owv3_archive(
                model=model,
                sensitivities=sensitivities,
                bit_budget_ratio=knobs.bit_budget_ratio,
                protect_threshold=knobs.protect_threshold,
                aggressive_threshold=knobs.aggressive_threshold,
                require_all_conv_sensitivity=True,
                fallback_action=knobs.fallback_action,
            )
            inspection = inspect_owv3_archive(renderer_blob)
            archive_blob = builder._archive_bytes([
                ("renderer.bin", renderer_blob),
                ("masks.mkv", masks_bytes),
                ("optimized_poses.pt", poses_bytes),
            ])
            archive_rebuild = builder._archive_bytes([
                ("renderer.bin", renderer_blob),
                ("masks.mkv", masks_bytes),
                ("optimized_poses.pt", poses_bytes),
            ])
            deterministic_rebuild = archive_blob == archive_rebuild
            if not deterministic_rebuild:
                raise RuntimeError(f"{cid}: deterministic archive rebuild mismatch")

            archive_size = len(archive_blob)
            archive_sha = builder._sha256(archive_blob)
            renderer_blobs_by_candidate_id[cid] = renderer_blob
            archive_blobs_by_candidate_id[cid] = archive_blob
            byte_classification = classify_candidate(
                archive_bytes=archive_size,
                frontier_bytes=frontier_bytes,
                lane_g_v3_bytes=lane_g_v3_archive_bytes,
            )
            archive_path: str | None = None
            should_write_archive = (
                args.archive_policy == "all"
                or (
                    args.archive_policy == "byte-feasible"
                    and bool(byte_classification["byte_feasible_vs_frontier"])
                )
            )
            if should_write_archive:
                target = _archive_target(archive_dir, cid)
                target.write_bytes(archive_blob)
                archive_path = str(target)

            decode_verified = False
            if args.decode_verify == "all":
                decoded = decode_owv3_archive(data=renderer_blob, device="cpu")
                if set(decoded.state_dict()) != set(model.state_dict()):
                    raise RuntimeError(f"{cid}: decoded state_dict keys diverged")
                decode_verified = True

            row.update({
                "renderer": {
                    "size_bytes": len(renderer_blob),
                    "sha256": builder._sha256(renderer_blob),
                    "delta_vs_asym_renderer_bytes": len(renderer_blob) - asym_renderer_bytes,
                },
                "archive": {
                    "path": archive_path,
                    "size_bytes": archive_size,
                    "sha256": archive_sha,
                    "deterministic_rebuild": deterministic_rebuild,
                    "deterministic_rebuild_sha256": builder._sha256(archive_rebuild),
                },
                "byte_classification": byte_classification,
                "owv3_byte_plan": inspection.get("byte_plan", {}),
                "owv3_thresholds": inspection.get("header", {}).get("thresholds", {}),
                "decode_verified": decode_verified,
            })
            if not args.no_manifest:
                row["archive_manifest"] = builder._archive_manifest(archive_blob)

            if bool(byte_classification["byte_feasible_vs_frontier"]):
                candidate_margin = frontier_bytes - archive_size
                if best_seen_row is None:
                    best_seen_row = row
                    best_seen_renderer_blob = renderer_blob
                    best_seen_archive_blob = archive_blob
                else:
                    best_margin = frontier_bytes - int(best_seen_row["archive"]["size_bytes"])
                    if (
                        candidate_margin,
                        -knobs.bit_budget_ratio,
                        knobs.protect_threshold,
                        knobs.aggressive_threshold,
                        cid,
                    ) < (
                        best_margin,
                        -float(best_seen_row["knobs"]["bit_budget_ratio"]),
                        float(best_seen_row["knobs"]["protect_threshold"]),
                        float(best_seen_row["knobs"]["aggressive_threshold"]),
                        str(best_seen_row["candidate_id"]),
                    ):
                        best_seen_row = row
                        best_seen_renderer_blob = renderer_blob
                        best_seen_archive_blob = archive_blob
        except Exception as exc:
            row.update({
                "build_status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "score_status": "not_evaluated_cuda_auth_required",
            })
        rows.append(row)
        if (index + 1) % max(1, int(args.progress_every)) == 0 or index == len(grid) - 1:
            ok = sum(1 for r in rows if r.get("build_status") == "ok")
            feasible = sum(
                1 for r in rows
                if r.get("byte_classification", {}).get("byte_feasible_vs_frontier")
            )
            print(f"  swept {index + 1}/{len(grid)} candidates: ok={ok} byte_feasible={feasible}")

    row_by_id = {str(row.get("candidate_id")): row for row in rows}
    best_byte_feasible = select_best_byte_candidate(rows)
    if best_byte_feasible is not None and best_seen_row is not None:
        best_byte_feasible = best_seen_row
        best_seen_renderer_blob = renderer_blobs_by_candidate_id.get(
            str(best_seen_row["candidate_id"]),
            best_seen_renderer_blob,
        )
        best_seen_archive_blob = archive_blobs_by_candidate_id.get(
            str(best_seen_row["candidate_id"]),
            best_seen_archive_blob,
        )

    r5_candidates = (
        select_r5_segnet_conservative_candidates(
            rows,
            reference_candidate_id=args.r5_reference_candidate_id,
            limit=args.r5_candidate_limit,
            max_bit_budget_ratio_drop=args.r5_max_bit_budget_ratio_drop,
        )
        if args.r5_reference_candidate_id
        else []
    )
    r5_candidate_ids = [str(row["candidate_id"]) for row in r5_candidates]
    for candidate in r5_candidates:
        original = row_by_id.get(str(candidate["candidate_id"]))
        if original is not None:
            original["r5_paired_calibration"] = candidate["r5_paired_calibration"]

    r6_candidates = (
        select_r6_segnet_conservative_candidates(
            rows,
            reference_candidate_id=args.r5_reference_candidate_id,
            limit=args.r5_candidate_limit,
            max_bit_budget_ratio_drop=args.r5_max_bit_budget_ratio_drop,
        )
        if args.selection_mode == "r6-segnet-conservative"
        else []
    )
    r6_candidate_ids = [str(row["candidate_id"]) for row in r6_candidates]
    for candidate in r6_candidates:
        original = row_by_id.get(str(candidate["candidate_id"]))
        if original is not None:
            original["r6_paired_calibration"] = candidate["r6_paired_calibration"]

    r7_candidates = (
        select_r7_pose_balanced_candidates(
            rows,
            reference_candidate_id=args.r6_reference_candidate_id,
            limit=args.r7_candidate_limit,
            min_bit_budget_ratio=args.r7_min_bit_budget_ratio,
            max_owv2_low_bit_channels=args.r7_max_owv2_low_bit_channels,
        )
        if args.selection_mode == "r7-pose-balanced"
        else []
    )
    r7_candidate_ids = [str(row["candidate_id"]) for row in r7_candidates]
    for candidate in r7_candidates:
        original = row_by_id.get(str(candidate["candidate_id"]))
        if original is not None:
            original["r7_pose_balanced_calibration"] = candidate[
                "r7_pose_balanced_calibration"
            ]

    write_r5_archives = (
        args.archive_policy != "none"
        and bool(
            r7_candidates
            if args.selection_mode == "r7-pose-balanced"
            else r6_candidates
            if args.selection_mode == "r6-segnet-conservative"
            else r5_candidates
        )
        and (
            args.r5_write_candidate_archives
            or args.selection_mode in {
                "r5-segnet-conservative",
                "r6-segnet-conservative",
                "r7-pose-balanced",
            }
        )
    )
    if write_r5_archives:
        candidate_ids_to_write = (
            r7_candidate_ids
            if args.selection_mode == "r7-pose-balanced"
            else r6_candidate_ids
            if args.selection_mode == "r6-segnet-conservative"
            else r5_candidate_ids
        )
        r5_dir = output_dir / (
            "r7_pose_balanced"
            if args.selection_mode == "r7-pose-balanced"
            else "r6_segnet_conservative"
            if args.selection_mode == "r6-segnet-conservative"
            else "r5_segnet_conservative"
        )
        archive_path_key = (
            "r7_candidate_path"
            if args.selection_mode == "r7-pose-balanced"
            else "r6_candidate_path"
            if args.selection_mode == "r6-segnet-conservative"
            else "r5_candidate_path"
        )
        for candidate_id_value in candidate_ids_to_write:
            row = row_by_id[candidate_id_value]
            target = r5_dir / candidate_id_value / "archive_lane_g_v3_owv3.zip"
            target.parent.mkdir(parents=True, exist_ok=True)
            archive_blob = archive_blobs_by_candidate_id.get(candidate_id_value)
            if archive_blob is None:
                existing_path = row.get("archive", {}).get("path")
                if not existing_path:
                    raise RuntimeError(f"no archive blob/path available for {candidate_id_value}")
                shutil.copyfile(existing_path, target)
            else:
                target.write_bytes(archive_blob)
            row["archive"][archive_path_key] = str(target)

    selected_for_eval = best_byte_feasible
    if args.selection_mode == "r5-segnet-conservative":
        selected_for_eval = row_by_id[r5_candidate_ids[0]] if r5_candidate_ids else None
    if args.selection_mode == "r6-segnet-conservative":
        selected_for_eval = row_by_id[r6_candidate_ids[0]] if r6_candidate_ids else None
    if args.selection_mode == "r7-pose-balanced":
        selected_for_eval = row_by_id[r7_candidate_ids[0]] if r7_candidate_ids else None

    selected_archive_path: Path | None = None
    if selected_for_eval is not None and args.archive_policy != "none":
        selected_dir_name = (
            "r7_selected_for_eval"
            if args.selection_mode == "r7-pose-balanced"
            else "r6_selected_for_eval"
            if args.selection_mode == "r6-segnet-conservative"
            else "r5_selected_for_eval"
            if args.selection_mode == "r5-segnet-conservative"
            else "best_byte_feasible"
        )
        selected_dir = output_dir / selected_dir_name
        selected_dir.mkdir(parents=True, exist_ok=True)
        selected_archive_path = selected_dir / "archive_lane_g_v3_owv3.zip"
        selected_id = str(selected_for_eval["candidate_id"])
        selected_archive_blob = archive_blobs_by_candidate_id.get(selected_id)
        selected_renderer_blob = renderer_blobs_by_candidate_id.get(selected_id)
        if selected_archive_blob is None:
            existing_path = (
                selected_for_eval.get("archive", {}).get("r7_candidate_path")
                or selected_for_eval.get("archive", {}).get("r6_candidate_path")
                or selected_for_eval.get("archive", {}).get("r5_candidate_path")
                or selected_for_eval.get("archive", {}).get("path")
            )
            if existing_path:
                shutil.copyfile(existing_path, selected_archive_path)
                selected_archive_blob = selected_archive_path.read_bytes()
        else:
            selected_archive_path.write_bytes(selected_archive_blob)
        selected_for_eval["archive"]["selected_path"] = str(selected_archive_path)

    selected_decode_verified = False
    if (
        selected_for_eval is not None
        and args.decode_verify in {"selected", "all"}
        and selected_renderer_blob is not None
    ):
        decoded = decode_owv3_archive(data=selected_renderer_blob, device="cpu")
        if set(decoded.state_dict()) != set(model.state_dict()):
            raise RuntimeError("selected candidate decoded state_dict keys diverged")
        selected_decode_verified = True
        selected_for_eval["decode_verified"] = True

    for row in rows:
        if best_byte_feasible is not None and row["candidate_id"] == best_byte_feasible["candidate_id"]:
            row["selected_by_sweep"] = True
            row["selection_basis"] = (
                "closest_archive_bytes_not_exceeding_frontier_no_score_claim"
            )
        else:
            row["selected_by_sweep"] = False
        if selected_for_eval is not None and row["candidate_id"] == selected_for_eval["candidate_id"]:
            row["selected_for_eval"] = True
            row["selected_for_eval_mode"] = args.selection_mode
            if args.selection_mode == "r5-segnet-conservative":
                row["selection_basis"] = (
                    "r5_segnet_conservative_neighbor_pending_paired_pfp16_cuda_t4_eval"
                )
            if args.selection_mode == "r6-segnet-conservative":
                row["selection_basis"] = (
                    "r6_segnet_conservative_neighbor_after_failed_r5_pending_paired_pfp16_cuda_t4_eval"
                )
            if args.selection_mode == "r7-pose-balanced":
                row["selection_basis"] = (
                    "r7_pose_balanced_neighbor_after_failed_r6_pending_paired_pfp16_cuda_t4_eval"
                )
            if selected_archive_path is not None:
                row["archive"]["selected_path"] = str(selected_archive_path)
        else:
            row["selected_for_eval"] = False

    jsonl_path = output_dir / "byte_plan_candidates.jsonl"
    summary_path = output_dir / "byte_plan_summary.json"
    _write_jsonl(jsonl_path, rows)

    ok_rows = [row for row in rows if row.get("build_status") == "ok"]
    feasible_rows = [
        row for row in ok_rows
        if row.get("byte_classification", {}).get("byte_feasible_vs_frontier")
    ]
    best_by_bytes = min(ok_rows, key=lambda r: int(r["archive"]["size_bytes"])) if ok_rows else None
    r5_candidates_summary = [
        json.loads(json.dumps(row_by_id[cid], sort_keys=True))
        for cid in r5_candidate_ids
    ]
    r6_candidates_summary = [
        json.loads(json.dumps(row_by_id[cid], sort_keys=True))
        for cid in r6_candidate_ids
    ]
    r7_candidates_summary = [
        json.loads(json.dumps(row_by_id[cid], sort_keys=True))
        for cid in r7_candidate_ids
    ]
    r5_exact_eval_queue = []
    for candidate in r5_candidates_summary:
        archive_path = (
            candidate.get("archive", {}).get("r5_candidate_path")
            or candidate.get("archive", {}).get("selected_path")
            or candidate.get("archive", {}).get("path")
            or "<materialize candidate archive before queueing>"
        )
        r5_exact_eval_queue.append(
            build_r5_exact_eval_queue_entry(
                candidate,
                archive_path=str(archive_path),
                output_dir=output_dir,
                baseline_archive_bytes=frontier_bytes,
            )
        )
    r6_exact_eval_queue = []
    for candidate in r6_candidates_summary:
        archive_path = (
            candidate.get("archive", {}).get("r6_candidate_path")
            or candidate.get("archive", {}).get("selected_path")
            or candidate.get("archive", {}).get("path")
            or "<materialize candidate archive before queueing>"
        )
        r6_exact_eval_queue.append(
            build_r5_exact_eval_queue_entry(
                candidate,
                archive_path=str(archive_path),
                output_dir=output_dir,
                baseline_archive_bytes=frontier_bytes,
                lane="owv3_r6_segnet_conservative_after_failed_r5",
                calibration_key="r6_paired_calibration",
            )
        )
    r7_exact_eval_queue = []
    for candidate in r7_candidates_summary:
        archive_path = (
            candidate.get("archive", {}).get("r7_candidate_path")
            or candidate.get("archive", {}).get("selected_path")
            or candidate.get("archive", {}).get("path")
            or "<materialize candidate archive before queueing>"
        )
        r7_exact_eval_queue.append(
            build_r5_exact_eval_queue_entry(
                candidate,
                archive_path=str(archive_path),
                output_dir=output_dir,
                baseline_archive_bytes=frontier_bytes,
                lane="owv3_r7_pose_balanced_after_failed_r6",
                calibration_key="r7_pose_balanced_calibration",
            )
        )
    elapsed_s = time.monotonic() - t0
    summary = {
        "format": "owv3_byte_plan_sweep_v1",
        "created_iso": utc_now_iso(),
        "elapsed_s": elapsed_s,
        "repo_root": str(REPO_ROOT),
        "source_scripts": {
            "sweep": str(Path(__file__).resolve()),
            "builder": str(REPO_ROOT / "experiments" / "build_lane_g_v3_owv3_stack.py"),
        },
        "score_claim_policy": (
            "No score claims are made by this sweep. Candidate status is based "
            "only on archive bytes; CUDA auth eval is required for any score."
        ),
        "sensitivity_map": {
            "path": str(args.sensitivity_map),
            "sha256": builder._sha256(args.sensitivity_map),
            "metadata": sensitivity_metadata,
            "stats": {
                "n_layers": sens_stats.n_layers,
                "n_channels": sens_stats.n_channels,
                "min_value": sens_stats.min_value,
                "max_value": sens_stats.max_value,
            },
            "source_device": str(source_device),
            "allow_non_authoritative": bool(args.allow_non_authoritative),
        },
        "anchors": anchor_meta,
        "frontier_comparator": {
            "label": args.frontier_comparator_label,
            "archive_bytes": frontier_bytes,
            "archive_sha256": args.frontier_comparator_sha256,
        },
        "grid": {
            "preset": args.preset,
            "candidate_count": len(grid),
            "bit_budget_ratios": sorted({k.bit_budget_ratio for k in grid}, reverse=True),
            "protect_thresholds": sorted({k.protect_threshold for k in grid}),
            "aggressive_thresholds": sorted({k.aggressive_threshold for k in grid}),
            "fallback_action": args.fallback_action,
        },
        "result_counts": {
            "ok": len(ok_rows),
            "errors": len(rows) - len(ok_rows),
            "byte_feasible_vs_frontier": len(feasible_rows),
            "byte_infeasible_vs_frontier": len(ok_rows) - len(feasible_rows),
        },
        "best_byte_feasible": best_byte_feasible,
        "smallest_archive_by_bytes": best_by_bytes,
        "selection_mode": args.selection_mode,
        "selected_for_eval": selected_for_eval,
        "r5_promotion_gates": r5_promotion_gate_policy(),
        "r5_paired_pfp16_calibration_queue": (
            build_r5_paired_pfp16_calibration_queue(
                paired_pfp16_archive=paired_pfp16_archive,
                frontier_archive_sha256=str(args.frontier_comparator_sha256),
                frontier_archive_bytes=frontier_bytes,
            )
            if args.r5_reference_candidate_id or args.r6_reference_candidate_id
            else None
        ),
        "r5_segnet_conservative_candidates": r5_candidates_summary,
        "r5_exact_eval_queue": r5_exact_eval_queue,
        "r6_failed_r5_exact_cuda_t4_reference": R5_FAILED_EXACT_CUDA_T4_REFERENCE,
        "r6_segnet_conservative_candidates": r6_candidates_summary,
        "r6_exact_eval_queue": r6_exact_eval_queue,
        "r7_failed_r6_exact_cuda_t4_reference": R6_FAILED_EXACT_CUDA_T4_REFERENCE,
        "r7_pose_balanced_policy": (
            "After R6 PoseNet gate failure, exact-eval only scalar-threshold "
            "candidates that keep OWV2-low-bit channels <= the R6 reference "
            "and bit_budget_ratio >= the R6 reference. If none exist, require "
            "component-balanced PoseNet/SegNet sensitivity before another "
            "promotion-path OWV3 eval."
        ),
        "r7_pose_balanced_candidates": r7_candidates_summary,
        "r7_exact_eval_queue": r7_exact_eval_queue,
        "selected_decode_verified": selected_decode_verified,
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "archive_policy": args.archive_policy,
        "cuda_auth_eval_command_template": (
            _contest_auth_eval_command(
                archive_path=str(selected_archive_path) if selected_archive_path else "<candidate archive.zip>",
                work_dir=str(output_dir / "cuda_auth_eval_work"),
            )
            if selected_for_eval is not None
            else []
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  wrote JSONL: {jsonl_path}")
    print(f"  wrote summary: {summary_path}")
    if selected_for_eval is None:
        print("  no selected eval candidate found")
    else:
        delta = selected_for_eval["byte_classification"]["frontier_delta_bytes"]
        print(
            "  selected eval candidate: "
            f"{selected_for_eval['candidate_id']} "
            f"{selected_for_eval['archive']['size_bytes']}B "
            f"({delta:+d} vs frontier)"
        )
        if selected_archive_path is not None:
            print(f"  selected archive: {selected_archive_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sensitivity-map", type=Path, default=DEFAULT_SENSITIVITY_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="frontier")
    parser.add_argument(
        "--bit-budget-ratios",
        default=None,
        help="Comma-separated override, e.g. '0.70,0.69,0.68'.",
    )
    parser.add_argument(
        "--protect-thresholds",
        default=None,
        help="Comma-separated override, e.g. '0.0014,0.00145,0.0015'.",
    )
    parser.add_argument(
        "--aggressive-thresholds",
        default=None,
        help="Comma-separated override. Recorded in OWV3 metadata.",
    )
    parser.add_argument(
        "--fallback-action",
        choices=["keep_asym", "error", "diagnostic_fp16"],
        default="keep_asym",
    )
    parser.add_argument(
        "--archive-policy",
        choices=["selected", "byte-feasible", "all", "none"],
        default="selected",
        help="Which candidate archives to write. JSONL always records bytes/SHA.",
    )
    parser.add_argument(
        "--decode-verify",
        choices=["none", "selected", "all"],
        default="selected",
        help="CPU decode sanity only; never runs scorer/evaluate.py.",
    )
    parser.add_argument("--no-manifest", action="store_true")
    parser.add_argument("--allow-non-authoritative", action="store_true")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help=(
            "Write a fast planning manifest for the component-sensitivity byte "
            "allocator without loading models, building archives, or claiming score."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output path for --manifest-only JSON. Defaults inside --output-dir.",
    )
    parser.add_argument(
        "--frontier-comparator-bytes",
        type=int,
        default=None,
        help="Defaults to the PFP16 A++ byte frontier from the OWV3 builder.",
    )
    parser.add_argument(
        "--frontier-comparator-sha256",
        default=None,
        help="Defaults to the PFP16 A++ SHA from the OWV3 builder.",
    )
    parser.add_argument("--frontier-comparator-label", default="PFP16_A++")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--selection-mode",
        choices=[
            "byte-best",
            "r5-segnet-conservative",
            "r6-segnet-conservative",
            "r7-pose-balanced",
        ],
        default="byte-best",
        help=(
            "Which candidate should be materialized as selected_for_eval. "
            "R5/R6 modes require --r5-reference-candidate-id and never select "
            "the failed reference candidate. R6 additionally records the "
            "failed exact R5 CUDA/T4 evidence and requires lower OWV2-low-bit "
            "channel count than that reference. R7 requires "
            "--r6-reference-candidate-id and blocks scalar-threshold candidates "
            "that further reduce the OWV2 bit budget after the failed R6 "
            "PoseNet gate."
        ),
    )
    parser.add_argument(
        "--r5-reference-candidate-id",
        default=None,
        help=(
            "Optional exact-evaluated failed OWV3 candidate id. When set, the "
            "summary ranks byte-feasible SegNet-conservative R5 neighbors."
        ),
    )
    parser.add_argument("--r5-candidate-limit", type=int, default=5)
    parser.add_argument(
        "--r6-reference-candidate-id",
        default=None,
        help=(
            "Optional exact-evaluated failed R6 candidate id. Required for "
            "--selection-mode r7-pose-balanced."
        ),
    )
    parser.add_argument("--r7-candidate-limit", type=int, default=5)
    parser.add_argument(
        "--r7-min-bit-budget-ratio",
        type=float,
        default=None,
        help=(
            "For R7 pose-balanced ranking, require candidates to keep at least "
            "this bit_budget_ratio. Defaults to the failed R6 reference ratio."
        ),
    )
    parser.add_argument(
        "--r7-max-owv2-low-bit-channels",
        type=int,
        default=None,
        help=(
            "For R7 pose-balanced ranking, require OWV2-low-bit channels to be "
            "at most this count. Defaults to the failed R6 reference count."
        ),
    )
    parser.add_argument(
        "--r5-max-bit-budget-ratio-drop",
        type=float,
        default=0.05,
        help=(
            "For R5 conservative ranking, exclude candidates whose "
            "bit_budget_ratio drops more than this below the failed reference."
        ),
    )
    parser.add_argument(
        "--r5-write-candidate-archives",
        action="store_true",
        help="Materialize each ranked R5 candidate archive under r5_segnet_conservative/.",
    )
    parser.add_argument(
        "--r5-paired-pfp16-archive",
        type=Path,
        default=DEFAULT_PAIRED_PFP16_ARCHIVE,
        help="PFP16 A++ archive to exact-eval first for same-run R5 calibration.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.manifest_only:
        output_path = args.json_out or (args.output_dir / "byte_plan_manifest.json")
        try:
            manifest = build_component_sensitivity_byte_plan_manifest(
                sensitivity_map=args.sensitivity_map,
                preset=args.preset,
                bit_budget_ratios=parse_float_list(args.bit_budget_ratios),
                protect_thresholds=parse_float_list(args.protect_thresholds),
                aggressive_thresholds=parse_float_list(args.aggressive_thresholds),
                fallback_action=args.fallback_action,
                frontier_comparator_bytes=args.frontier_comparator_bytes,
                frontier_comparator_sha256=args.frontier_comparator_sha256,
                frontier_comparator_label=args.frontier_comparator_label,
                allow_non_authoritative=args.allow_non_authoritative,
                limit=args.limit,
            )
        except Exception as exc:
            print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    builder = _load_builder_module()
    if args.frontier_comparator_bytes is None:
        args.frontier_comparator_bytes = builder.PFP16_FRONTIER_ARCHIVE_BYTES
    if args.frontier_comparator_sha256 is None:
        args.frontier_comparator_sha256 = builder.PFP16_FRONTIER_ARCHIVE_SHA256
    try:
        run_sweep(args)
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
