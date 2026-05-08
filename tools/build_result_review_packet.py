#!/usr/bin/env python3
"""Build a result-review packet for returned exact/proxy evidence.

This tool is intentionally conservative. It does not promote, rank, or kill a
lane. It turns a returned structured eval artifact into a machine-readable
review packet with custody, score recomputation, failure classification, and
reactivation criteria so lane status changes cannot happen from prose alone.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

CONTEST_DENOMINATOR_BYTES = 37_545_489
SCHEMA = "tac_result_review_packet_v1"
DEFAULT_DISPATCH_CLAIMS_PATH = Path(".omx/state/active_lane_dispatch_claims.md")
TERMINAL_CLAIM_PREFIXES = (
    "completed_",
    "failed_",
    "preempted",
    "cancelled",
    "refused_dispatch",
    "stale_assumed_dead",
    "stale_superseded",
    "stopped_",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _nested_get(mapping: dict[str, Any], *keys: str) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _archive_bytes(payload: dict[str, Any]) -> int | None:
    return (
        _int_or_none(payload.get("archive_size_bytes"))
        or _int_or_none(payload.get("archive_bytes"))
        or _int_or_none(_nested_get(payload, "provenance", "archive_size_bytes"))
    )


def _archive_sha(payload: dict[str, Any]) -> str:
    return str(
        payload.get("archive_sha256")
        or _nested_get(payload, "provenance", "archive_sha256")
        or ""
    )


def _device(payload: dict[str, Any]) -> str:
    return str(payload.get("device") or _nested_get(payload, "provenance", "device") or "")


def _is_exact_cuda(payload: dict[str, Any]) -> bool:
    text = " ".join(
        str(x).lower()
        for x in (
            payload.get("evidence_grade"),
            _device(payload),
            _nested_get(payload, "provenance", "gpu_model"),
        )
        if x is not None
    )
    return "cuda" in text or "contest-cuda" in text or "tesla t4" in text


def _score_recompute(payload: dict[str, Any]) -> dict[str, Any]:
    seg = _float_or_none(payload.get("avg_segnet_dist"))
    pose = _float_or_none(payload.get("avg_posenet_dist"))
    archive_bytes = _archive_bytes(payload)
    reported = (
        _float_or_none(payload.get("score_recomputed_from_components"))
        or _float_or_none(payload.get("canonical_score"))
        or _float_or_none(payload.get("final_score"))
    )
    if seg is None or pose is None or archive_bytes is None:
        return {
            "available": False,
            "blockers": ["missing_seg_pose_or_archive_bytes"],
        }
    rate = 25.0 * float(archive_bytes) / float(CONTEST_DENOMINATOR_BYTES)
    recomputed = 100.0 * seg + math.sqrt(10.0 * pose) + rate
    delta = None if reported is None else abs(recomputed - reported)
    return {
        "available": True,
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "archive_bytes": archive_bytes,
        "rate_term": rate,
        "recomputed_score": recomputed,
        "reported_score": reported,
        "abs_delta_vs_reported": delta,
        "matches_reported": bool(delta is not None and delta <= 1e-8),
    }


def _runtime_custody(payload: dict[str, Any]) -> dict[str, Any]:
    manifest = _nested_get(payload, "provenance", "inflate_runtime_manifest")
    if not isinstance(manifest, dict):
        return {
            "runtime_manifest_present": False,
            "runtime_tree_sha256": "",
            "runtime_file_count": None,
            "payload_closure_fields_present": False,
        }
    files = manifest.get("files")
    return {
        "runtime_manifest_present": True,
        "runtime_tree_sha256": str(manifest.get("runtime_tree_sha256") or ""),
        "runtime_file_count": _int_or_none(manifest.get("runtime_file_count")),
        "runtime_files_listed": isinstance(files, list) and len(files) > 0,
        "external_dependency_roots": manifest.get("external_dependency_roots", []),
        "payload_closure_fields_present": bool(
            manifest.get("runtime_tree_sha256")
            and isinstance(files, list)
            and len(files) > 0
        ),
    }


def _is_terminal_claim_status(status: str) -> bool:
    return any(status.startswith(prefix) for prefix in TERMINAL_CLAIM_PREFIXES)


def _dispatch_claim_state(
    *,
    claims_path: Path | None,
    lane_id: str,
    job_id: str,
) -> dict[str, Any]:
    """Extract the dispatch-claim custody state for a reviewed result."""
    if claims_path is None:
        return {
            "claims_path": "",
            "claims_file_present": False,
            "matching_claim_count": 0,
            "latest_status": "",
            "latest_claim": None,
            "terminal_status_recorded": False,
        }
    if not claims_path.exists():
        return {
            "claims_path": str(claims_path),
            "claims_file_present": False,
            "matching_claim_count": 0,
            "latest_status": "",
            "latest_claim": None,
            "terminal_status_recorded": False,
        }
    matches: list[dict[str, str]] = []
    for line in claims_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        if "timestamp_utc" in line and "lane_id" in line:
            continue
        if line.replace("|", "").replace("-", "").strip() == "":
            continue
        cells = [cell.strip().replace("\\|", "|") for cell in line.strip("|").split("|")]
        if len(cells) < 8:
            continue
        claim = {
            "timestamp_utc": cells[0],
            "agent": cells[1],
            "lane_id": cells[2],
            "platform": cells[3],
            "instance_job_id": cells[4],
            "predicted_eta_utc": cells[5],
            "status": cells[6],
            "notes": cells[7],
        }
        if claim["lane_id"] == lane_id and claim["instance_job_id"] == job_id:
            matches.append(claim)
    latest = matches[0] if matches else None
    latest_status = latest["status"] if latest else ""
    return {
        "claims_path": str(claims_path),
        "claims_file_present": True,
        "matching_claim_count": len(matches),
        "latest_status": latest_status,
        "latest_claim": latest,
        "terminal_status_recorded": _is_terminal_claim_status(latest_status),
    }


def build_packet(
    *,
    auth_eval_json: Path,
    technique: str,
    lane_id: str,
    job_id: str,
    baseline_score: float | None,
    reactivation_criteria: list[str],
    reviewer: str,
    dispatch_claims_path: Path | None = DEFAULT_DISPATCH_CLAIMS_PATH,
) -> dict[str, Any]:
    payload = json.loads(auth_eval_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("auth eval JSON must contain an object")
    score = (
        _float_or_none(payload.get("canonical_score"))
        or _float_or_none(payload.get("score_recomputed_from_components"))
        or _float_or_none(payload.get("final_score"))
    )
    exact_cuda = _is_exact_cuda(payload)
    regression = bool(
        exact_cuda
        and baseline_score is not None
        and score is not None
        and score > baseline_score
    )
    if regression and not reactivation_criteria:
        raise ValueError(
            "reactivation criteria are required for negative exact-CUDA review packets"
        )
    status = "indeterminate"
    failure_class = "indeterminate"
    if regression:
        status = "measured_config_retired"
        failure_class = "legitimate_score_regression_or_component_collapse"
    elif exact_cuda and score is not None:
        status = "exact_cuda_result_reviewed"
        failure_class = "not_negative_against_supplied_baseline"
    elif not exact_cuda:
        status = "proxy_or_non_cuda_review_only"
        failure_class = "non_exact_cuda_evidence"
    return {
        "schema": SCHEMA,
        "tool": "tools/build_result_review_packet.py",
        "reviewer": reviewer,
        "technique": technique,
        "lane_id": lane_id,
        "job_id": job_id,
        "source_json_path": str(auth_eval_json),
        "source_json_sha256": _sha256_file(auth_eval_json),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "family_falsified": False,
        "method_family_retired": False,
        "measured_config_status": status,
        "failure_class": failure_class,
        "baseline_score": baseline_score,
        "canonical_score": score,
        "exact_cuda_evidence": exact_cuda,
        "custody": {
            "archive_bytes": _archive_bytes(payload),
            "archive_sha256": _archive_sha(payload),
            "device": _device(payload),
            "gpu_model": str(_nested_get(payload, "provenance", "gpu_model") or ""),
            "n_samples": _int_or_none(payload.get("n_samples")),
            "inflate_script": str(_nested_get(payload, "provenance", "inflate_script") or ""),
            "command": _nested_get(payload, "provenance", "sys_argv") or [],
        },
        "dispatch_claim_state": _dispatch_claim_state(
            claims_path=dispatch_claims_path,
            lane_id=lane_id,
            job_id=job_id,
        ),
        "runtime_custody": _runtime_custody(payload),
        "score_recomputation": _score_recompute(payload),
        "review_requirements": {
            "engineering_review_required": True,
            "math_review_required": True,
            "geometry_review_required": True,
            "optimization_review_required": True,
            "contest_compliance_review_required": True,
        },
        "reactivation_criteria": reactivation_criteria,
        "notes": [
            "This packet can retire only the measured config unless the broader kill discipline is separately satisfied.",
            "Exact CUDA evidence may update solver trust regions after adversarial review.",
        ],
    }


def evidence_row_from_packet(
    packet: dict[str, Any],
    *,
    review_packet_path: Path | None = None,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Derive a conservative planner evidence row from a review packet."""
    status = str(packet.get("measured_config_status") or "")
    exact_cuda = bool(packet.get("exact_cuda_evidence"))
    negative = exact_cuda and status == "measured_config_retired"
    custody = packet.get("custody") if isinstance(packet.get("custody"), dict) else {}
    score_recompute = (
        packet.get("score_recomputation")
        if isinstance(packet.get("score_recomputation"), dict) else {}
    )
    dispatch_claim_state = (
        packet.get("dispatch_claim_state")
        if isinstance(packet.get("dispatch_claim_state"), dict) else {}
    )
    archive_bytes = custody.get("archive_bytes")
    rate_term = score_recompute.get("rate_term")
    archive_rate_ratio = (
        (float(rate_term) / 25.0)
        if isinstance(rate_term, (int, float))
        else None
    )
    if timestamp_utc is None:
        timestamp_utc = dt.datetime.now(tz=dt.UTC).replace(
            microsecond=0
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

    blockers = ["review_packet_non_promotable_until_explicit_lane_promotion"]
    if negative:
        blockers = [
            "measured_config_retired_exact_cuda_negative",
            "reactivation_required_before_new_dispatch",
        ]
    elif not exact_cuda:
        blockers = ["non_exact_cuda_review_packet_not_promotable"]

    source_json_path = str(packet.get("source_json_path") or "")
    review_path = str(review_packet_path or "")
    return {
        "technique": str(packet.get("technique") or ""),
        "lane_id": str(packet.get("lane_id") or ""),
        "job_name": str(packet.get("job_id") or ""),
        "evidence_grade": (
            "[contest-CUDA A-negative]" if negative
            else "[contest-CUDA reviewed]" if exact_cuda
            else "[non-CUDA review]"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "contest_dispatch_verdict": (
            "measured_config_retired_exact_cuda_negative" if negative else status
        ),
        "measured_config_status": status,
        "score_contest_cuda": packet.get("canonical_score") if exact_cuda else None,
        "empirical_score": packet.get("canonical_score"),
        "empirical_archive_bytes": archive_bytes,
        "archive_sha256": custody.get("archive_sha256"),
        "segnet_distortion": score_recompute.get("avg_segnet_dist"),
        "posenet_distortion": score_recompute.get("avg_posenet_dist"),
        "rate": rate_term,
        "rate_term": rate_term,
        "archive_rate_ratio": archive_rate_ratio,
        "auth_eval_json": source_json_path,
        "exact_result_review_packet": review_path,
        "dispatch_claim_latest_status": dispatch_claim_state.get("latest_status"),
        "dispatch_claim_terminal_status_recorded": dispatch_claim_state.get(
            "terminal_status_recorded"
        ),
        "family_falsified": bool(packet.get("family_falsified") is True),
        "method_family_retired": bool(packet.get("method_family_retired") is True),
        "falsification_scope": (
            "measured_config_only" if negative else "none"
        ),
        "dispatch_blockers": blockers,
        "reactivation_criteria": packet.get("reactivation_criteria") or [],
        "source": (
            f"{source_json_path} reviewed_by={review_path}"
            if review_path else source_json_path
        ),
        "timestamp": timestamp_utc,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--auth-eval-json", type=Path, required=True)
    p.add_argument("--technique", required=True)
    p.add_argument("--lane-id", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--baseline-score", type=float)
    p.add_argument("--reactivation-criteria", action="append", default=[])
    p.add_argument("--reviewer", default="codex")
    p.add_argument("--dispatch-claims-md", type=Path, default=DEFAULT_DISPATCH_CLAIMS_PATH)
    p.add_argument("--evidence-row-output", type=Path, default=None)
    p.add_argument("--append-evidence-jsonl", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    packet = build_packet(
        auth_eval_json=args.auth_eval_json,
        technique=args.technique,
        lane_id=args.lane_id,
        job_id=args.job_id,
        baseline_score=args.baseline_score,
        reactivation_criteria=list(args.reactivation_criteria),
        reviewer=args.reviewer,
        dispatch_claims_path=args.dispatch_claims_md,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence_row = evidence_row_from_packet(packet, review_packet_path=args.output)
    if args.evidence_row_output is not None:
        args.evidence_row_output.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_row_output.write_text(
            json.dumps(evidence_row, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.append_evidence_jsonl is not None:
        args.append_evidence_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.append_evidence_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    if args.evidence_row_output is not None:
        print(f"wrote evidence row {args.evidence_row_output}")
    if args.append_evidence_jsonl is not None:
        print(f"appended evidence row {args.append_evidence_jsonl}")
    print(
        f"status={packet['measured_config_status']} "
        f"score={packet['canonical_score']} exact_cuda={packet['exact_cuda_evidence']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
