# SPDX-License-Identifier: MIT
"""Modal training dispatch-claim recovery helpers.

Training jobs are not score-claim surfaces, but their provider lifecycle still
needs the same custody closure as exact-eval jobs. These helpers are shared by
single-call recovery and bulk harvest so Modal result-cache recovery cannot
leave phantom active lane claims behind.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from tac.auth_eval_result import parse_auth_eval_score_claim
from tac.deploy.claims import DispatchClaimSpec, terminal_dispatch_claim
from tac.repo_io import json_text

SCHEMA = "modal_training_terminal_claim_v1"


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def recovered_inline_contest_cuda_auth_eval(out_dir: Path) -> dict[str, Any] | None:
    """Return recovered inline contest-CUDA auth-eval signal, if present.

    Modal training terminal claims are intentionally non-promotional, but some
    trainers run a byte-closed auth eval inline. This helper preserves that
    signal without turning provider lifecycle recovery into a submission-ready
    ranking surface.
    """

    root = Path(out_dir)
    candidates = (
        root / "harvested_artifacts" / "contest_auth_eval_cuda.json",
        root / "harvested_artifacts" / "contest_auth_eval.json",
        root / "contest_auth_eval_cuda.json",
        root / "contest_auth_eval.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        payload = _read_json_object(path)
        if payload is None:
            continue
        claim = parse_auth_eval_score_claim(
            payload,
            required_score_axis="contest_cuda",
        )
        if claim is None:
            continue
        return {
            "auth_eval_result_path": str(path),
            "auth_eval_score": claim.score,
            "auth_eval_score_axis": claim.score_axis,
            "auth_eval_score_claim_valid": True,
            "auth_eval_exact_cuda_complete": claim.exact_cuda_eval_complete,
            "auth_eval_evidence_grade": claim.evidence_grade,
            "auth_eval_lane_tag": claim.lane_tag,
            "auth_eval_score_source_key": claim.source_key,
            "auth_eval_recomputed_score": claim.recomputed_score,
        }
    return None


def modal_training_terminal_status(
    result: dict[str, Any] | None,
    *,
    recovered_auth_eval: dict[str, Any] | None = None,
) -> str:
    """Return the terminal lane-claim status for a recovered Modal training result."""

    if not isinstance(result, dict):
        return "failed_modal_training_missing_result"
    if result.get("timed_out") is True:
        return "failed_modal_training_timeout"
    rc = result.get("returncode", result.get("rc"))
    if isinstance(rc, bool):
        return "failed_modal_training_invalid_returncode"
    if isinstance(rc, int):
        if rc == 0:
            if recovered_auth_eval is not None:
                return "completed_modal_training_recovered_with_contest_cuda_auth_eval"
            return "completed_modal_training_recovered_no_score_claim"
        return f"failed_modal_training_rc_{rc}"
    return "failed_modal_training_unknown_returncode"


def append_modal_training_terminal_claim(
    *,
    repo_root: Path,
    out_dir: Path,
    metadata: dict[str, Any],
    result: dict[str, Any] | None,
    status: str | None = None,
    agent: str = "codex:modal_training_recovery",
) -> dict[str, Any]:
    """Append one terminal dispatch-claim row for a Modal training call.

    The helper is idempotent per ``out_dir``. An existing marker is returned
    without appending another terminal row. Failures are represented as a
    manifest with ``appended=false`` so artifact recovery never loses signal.
    """

    repo_root = Path(repo_root)
    out_dir = Path(out_dir)
    marker = out_dir / "modal_training_terminal_claim.json"

    def _write_marker(manifest: dict[str, Any]) -> dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(json_text(manifest), encoding="utf-8")
        return manifest

    if marker.is_file():
        try:
            existing = marker.read_text(encoding="utf-8")
            payload = json.loads(existing)
            if isinstance(payload, dict) and payload.get("appended") is True:
                return {**payload, "already_appended": True}
        except Exception:
            pass

    lane_id = str(metadata.get("lane_id") or "").strip()
    instance_job_id = str(metadata.get("label") or metadata.get("instance_job_id") or "").strip()
    platform = str(metadata.get("platform") or "modal").strip() or "modal"
    if not lane_id or not instance_job_id:
        return _write_marker({
            "schema": SCHEMA,
            "appended": False,
            "already_appended": False,
            "reason": "metadata_missing_lane_id_or_instance_job_id",
            "metadata_label": str(metadata.get("label") or ""),
            "metadata_call_id": str(metadata.get("call_id") or ""),
            "score_claim": False,
            "promotion_eligible": False,
        })

    if result is None:
        result = {}
    recovered_auth_eval = recovered_inline_contest_cuda_auth_eval(out_dir)
    terminal_status = status or modal_training_terminal_status(
        result,
        recovered_auth_eval=recovered_auth_eval,
    )
    rc = result.get("returncode", result.get("rc"))
    timed_out = bool(result.get("timed_out", False))
    elapsed = result.get("elapsed_seconds")
    recovered_score_note = ""
    if recovered_auth_eval is not None:
        recovered_score_note = (
            "; recovered_inline_contest_cuda_auth_eval_score="
            f"{recovered_auth_eval['auth_eval_score']}"
        )
    notes = (
        "Modal training terminal recovery; score_claim=false; "
        f"promotion_eligible=false; rc={rc}; timed_out={timed_out}; "
        f"elapsed_seconds={elapsed}; out_dir={out_dir}{recovered_score_note}"
    )
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "appended": False,
        "already_appended": False,
        "lane_id": lane_id,
        "instance_job_id": instance_job_id,
        "platform": platform,
        "status": terminal_status,
        "score_claim": False,
        "promotion_eligible": False,
        "notes": notes,
    }
    if recovered_auth_eval is not None:
        manifest["recovered_auth_eval"] = recovered_auth_eval
    try:
        terminal_dispatch_claim(
            repo_root=repo_root,
            spec=DispatchClaimSpec(
                lane_id=lane_id,
                instance_job_id=instance_job_id,
                platform=platform,
                agent=agent,
                force=True,
            ),
            status=terminal_status,
            notes=notes,
            python_executable=sys.executable,
        )
    except Exception as exc:
        manifest["reason"] = f"terminal_claim_failed:{type(exc).__name__}:{exc}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "modal_training_terminal_claim_error.json").write_text(
            json_text(manifest), encoding="utf-8"
        )
        return manifest

    manifest["appended"] = True
    return _write_marker(manifest)


__all__ = [
    "append_modal_training_terminal_claim",
    "modal_training_terminal_status",
    "recovered_inline_contest_cuda_auth_eval",
]
