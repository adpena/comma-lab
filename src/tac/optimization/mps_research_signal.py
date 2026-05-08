"""MPS research-signal manifests for cheap curve discovery.

Local Apple MPS is useful for discovering proxy curve shapes, smoke failures,
and code-correctness issues, but it is not score evidence. This module turns
MPS sweep observations into a typed manifest that can feed planning tools while
remaining fail-closed for promotion, ranking, and exact-eval dispatch.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

from tac.score_geometry import contest_score

SCHEMA_VERSION = "mps_research_signal_manifest.v1"
EVIDENCE_GRADE = "MPS-research-signal"
EVIDENCE_SEMANTICS = "mps_proxy_curve_shape_only"
DEFAULT_CONFIDENCE = 0.05
FORBIDDEN_USES = (
    "auth_eval",
    "score_claim",
    "promotion",
    "falsification",
    "method_retirement",
    "paper_empirical_claim",
)
ALLOWED_USES = (
    "proxy_curve_shape_discovery",
    "smoke_test",
    "code_correctness_check",
    "candidate_generation_prior",
)
DISPATCH_BLOCKERS = (
    "mps_proxy_signal_not_score_evidence",
    "not_cuda_auth_eval",
    "no_exact_archive_adjudication",
    "not_promotion_eligible",
    "requires_exact_cuda_auth_eval_before_any_score_use",
)


class MPSResearchSignalError(ValueError):
    """Raised when an MPS research-signal observation is malformed."""


def load_observations(path: Path) -> list[dict[str, Any]]:
    """Load MPS observations from JSON or JSONL."""

    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            raw_rows = payload.get("observations", payload.get("rows"))
            if not isinstance(raw_rows, list):
                raise MPSResearchSignalError(
                    f"{path}: JSON dict must contain observations[] or rows[]"
                )
            rows = raw_rows
        else:
            raise MPSResearchSignalError(f"{path}: expected JSON list or dict")
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise MPSResearchSignalError(f"{path}: row {index} is not an object")
        out.append(dict(row))
    return out


def build_mps_research_signal_manifest(
    observations: Iterable[Mapping[str, Any]],
    *,
    source: str,
    run_id: str,
    anchor_d_seg: float | None = None,
    anchor_d_pose: float | None = None,
    anchor_archive_bytes: int | None = None,
) -> dict[str, Any]:
    """Build a fail-closed MPS research-signal manifest.

    Args:
        observations: iterable of row-like mappings. Each row must contain
            ``family``, ``variant_id`` and ``archive_bytes``. Rows may include
            ``d_seg_proxy``/``d_pose_proxy`` or ``proxy_loss``.
        source: source label or path for custody.
        run_id: stable run id chosen by the caller.
        anchor_d_seg / anchor_d_pose / anchor_archive_bytes: optional exact-CUDA
            anchor values used only to compute proxy deltas. The output remains
            non-promotable even when these are supplied.
    """

    anchor = _normalize_anchor(
        d_seg=anchor_d_seg,
        d_pose=anchor_d_pose,
        archive_bytes=anchor_archive_bytes,
    )
    rows = [
        _normalize_observation(
            row,
            index=index,
            anchor=anchor,
        )
        for index, row in enumerate(observations)
    ]
    curves = _curve_summaries(rows)
    atoms = [_row_to_meta_lagrangian_atom(row) for row in rows]
    return {
        "schema": SCHEMA_VERSION,
        "source": source,
        "run_id": run_id,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "device_contract": {
            "device_family": "mps",
            "allowed_uses": list(ALLOWED_USES),
            "forbidden_uses": list(FORBIDDEN_USES),
            "cuda_auth_eval_required_for_score_use": True,
        },
        "anchor": anchor,
        "row_count": len(rows),
        "rows": rows,
        "curve_count": len(curves),
        "curves": curves,
        "meta_lagrangian_atoms": atoms,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
    }


def _normalize_anchor(
    *,
    d_seg: float | None,
    d_pose: float | None,
    archive_bytes: int | None,
) -> dict[str, Any]:
    supplied = [d_seg is not None, d_pose is not None, archive_bytes is not None]
    if any(supplied) and not all(supplied):
        raise MPSResearchSignalError(
            "anchor_d_seg, anchor_d_pose, and anchor_archive_bytes must be supplied together"
        )
    if not any(supplied):
        return {
            "available": False,
            "evidence_required": "exact_cuda_anchor_required_for_proxy_deltas",
        }
    if d_seg is None or d_pose is None or archive_bytes is None:
        raise AssertionError("unreachable anchor state")
    if d_seg < 0.0 or d_pose < 0.0 or archive_bytes <= 0:
        raise MPSResearchSignalError("anchor values must be non-negative and bytes must be positive")
    return {
        "available": True,
        "d_seg": float(d_seg),
        "d_pose": float(d_pose),
        "archive_bytes": int(archive_bytes),
        "score_formula_value": float(contest_score(d_seg, d_pose, int(archive_bytes))),
        "score_claim": False,
        "evidence_grade_required": "contest-CUDA",
    }


def _normalize_observation(
    row: Mapping[str, Any],
    *,
    index: int,
    anchor: Mapping[str, Any],
) -> dict[str, Any]:
    family = _required_text(row, "family", index=index)
    variant_id = _required_text(row, "variant_id", index=index)
    curve_id = str(row.get("curve_id") or family)
    device = str(row.get("device") or "mps").lower()
    if not (device == "mps" or device.startswith("mps:")):
        raise MPSResearchSignalError(f"row {index}: device must be mps, got {device!r}")
    archive_bytes = _required_positive_int(row, "archive_bytes", index=index)
    proxy_loss = _optional_float(row.get("proxy_loss"))
    d_seg = _optional_float(row.get("d_seg_proxy", row.get("proxy_d_seg")))
    d_pose = _optional_float(row.get("d_pose_proxy", row.get("proxy_d_pose")))
    if proxy_loss is None and d_seg is None and d_pose is None:
        raise MPSResearchSignalError(
            f"row {index}: supply proxy_loss or at least one proxy distortion component"
        )
    if d_seg is not None and d_seg < 0.0:
        raise MPSResearchSignalError(f"row {index}: d_seg_proxy must be non-negative")
    if d_pose is not None and d_pose < 0.0:
        raise MPSResearchSignalError(f"row {index}: d_pose_proxy must be non-negative")
    if proxy_loss is not None and not math.isfinite(proxy_loss):
        raise MPSResearchSignalError(f"row {index}: proxy_loss must be finite")

    proxy_formula = None
    if d_seg is not None and d_pose is not None:
        proxy_formula = float(contest_score(d_seg, d_pose, archive_bytes))

    byte_delta = None
    expected_seg_delta = None
    expected_pose_delta = None
    proxy_formula_delta = None
    if anchor.get("available"):
        byte_delta = archive_bytes - int(anchor["archive_bytes"])
        if d_seg is not None:
            expected_seg_delta = d_seg - float(anchor["d_seg"])
        if d_pose is not None:
            expected_pose_delta = d_pose - float(anchor["d_pose"])
        if proxy_formula is not None:
            proxy_formula_delta = proxy_formula - float(anchor["score_formula_value"])

    normalized = {
        "row_index": index,
        "family": family,
        "curve_id": curve_id,
        "variant_id": variant_id,
        "params": dict(row.get("params") or {}),
        "device": device,
        "archive_bytes": archive_bytes,
        "byte_delta_vs_anchor": byte_delta,
        "d_seg_proxy": d_seg,
        "d_pose_proxy": d_pose,
        "expected_seg_dist_delta_proxy": expected_seg_delta,
        "expected_pose_dist_delta_proxy": expected_pose_delta,
        "proxy_loss": proxy_loss,
        "proxy_formula_value": proxy_formula,
        "proxy_formula_delta_vs_anchor": proxy_formula_delta,
        "wall_clock_seconds": _optional_float(row.get("wall_clock_seconds")),
        "source_artifact": str(row.get("source_artifact") or ""),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatchable": False,
        "proxy_row": True,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
    }
    return normalized


def _curve_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["family"]), str(row["curve_id"]))].append(row)

    summaries: list[dict[str, Any]] = []
    for (family, curve_id), curve_rows in sorted(grouped.items()):
        ordered = sorted(curve_rows, key=lambda row: (int(row["archive_bytes"]), str(row["variant_id"])))
        metric_name = _curve_metric_name(ordered)
        slopes = _finite_difference_slopes(ordered, metric_name)
        lowest = _lowest_proxy_observation(ordered, metric_name)
        summaries.append(
            {
                "family": family,
                "curve_id": curve_id,
                "point_count": len(ordered),
                "metric": metric_name,
                "archive_byte_range": [
                    int(ordered[0]["archive_bytes"]),
                    int(ordered[-1]["archive_bytes"]),
                ],
                "finite_difference_slopes": slopes,
                "flattening_detected": _flattening_detected(slopes),
                "lowest_proxy_observation": lowest,
                "score_claim": False,
                "promotion_eligible": False,
                "interpretation": "candidate_generation_prior_only",
            }
        )
    return summaries


def _curve_metric_name(rows: list[dict[str, Any]]) -> str:
    if all(row.get("proxy_loss") is not None for row in rows):
        return "proxy_loss"
    if all(row.get("proxy_formula_value") is not None for row in rows):
        return "proxy_formula_value"
    if all(row.get("d_pose_proxy") is not None for row in rows):
        return "d_pose_proxy"
    return "d_seg_proxy"


def _finite_difference_slopes(rows: list[dict[str, Any]], metric_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for previous, current in pairwise(rows):
        prev_metric = previous.get(metric_name)
        cur_metric = current.get(metric_name)
        if prev_metric is None or cur_metric is None:
            continue
        byte_step = int(current["archive_bytes"]) - int(previous["archive_bytes"])
        if byte_step == 0:
            continue
        out.append(
            {
                "from_variant_id": previous["variant_id"],
                "to_variant_id": current["variant_id"],
                "byte_step": byte_step,
                "metric_delta": float(cur_metric) - float(prev_metric),
                "metric_delta_per_byte": (float(cur_metric) - float(prev_metric)) / float(byte_step),
            }
        )
    return out


def _flattening_detected(slopes: list[dict[str, Any]]) -> bool:
    if len(slopes) < 2:
        return False
    magnitudes = [abs(float(row["metric_delta_per_byte"])) for row in slopes]
    maximum = max(magnitudes)
    if maximum <= 0.0:
        return True
    return magnitudes[-1] <= 0.25 * maximum


def _lowest_proxy_observation(rows: list[dict[str, Any]], metric_name: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get(metric_name) is not None]
    if not candidates:
        return None
    row = min(candidates, key=lambda item: (float(item[metric_name]), int(item["archive_bytes"])))
    return {
        "variant_id": row["variant_id"],
        "archive_bytes": row["archive_bytes"],
        "metric": metric_name,
        "metric_value": row[metric_name],
        "candidate_generation_only": True,
    }


def _row_to_meta_lagrangian_atom(row: Mapping[str, Any]) -> dict[str, Any]:
    byte_delta = row.get("byte_delta_vs_anchor")
    return {
        "atom_id": f"mps_signal:{row['family']}:{row['variant_id']}",
        "family": str(row["family"]),
        "family_group": f"mps_research_signal:{row['family']}",
        "pareto_scope": f"mps_research_signal:{row['curve_id']}",
        "byte_delta": int(byte_delta) if byte_delta is not None else 0,
        "expected_seg_dist_delta": float(row.get("expected_seg_dist_delta_proxy") or 0.0),
        "expected_pose_dist_delta": float(row.get("expected_pose_dist_delta_proxy") or 0.0),
        "confidence": DEFAULT_CONFIDENCE,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "proxy_row": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "rankable": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatchable": False,
        "interaction_assumptions": [
            "mps_proxy_curve_shape_may_not_transfer_to_cuda_scorer",
        ],
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
        "source_artifact_bytes": int(row["archive_bytes"]),
    }


def _required_text(row: Mapping[str, Any], key: str, *, index: int) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise MPSResearchSignalError(f"row {index}: missing {key}")
    return value


def _required_positive_int(row: Mapping[str, Any], key: str, *, index: int) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MPSResearchSignalError(f"row {index}: {key} must be an integer")
    if value <= 0:
        raise MPSResearchSignalError(f"row {index}: {key} must be positive")
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise MPSResearchSignalError("boolean is not a valid float")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise MPSResearchSignalError(f"could not parse float from {value!r}") from exc
    if not math.isfinite(out):
        raise MPSResearchSignalError(f"non-finite float {value!r}")
    return out


def json_text(payload: Any) -> str:
    """Deterministic JSON text for manifest outputs."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


__all__ = [
    "EVIDENCE_GRADE",
    "EVIDENCE_SEMANTICS",
    "MPSResearchSignalError",
    "build_mps_research_signal_manifest",
    "json_text",
    "load_observations",
]
