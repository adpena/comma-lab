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

from tac.deploy.claims import DispatchClaimSpec, terminal_dispatch_claim
from tac.repo_io import json_text

SCHEMA = "modal_training_terminal_claim_v1"


def modal_training_terminal_status(result: dict[str, Any] | None) -> str:
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
            if isinstance(payload, dict):
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

    terminal_status = status or modal_training_terminal_status(result)
    if result is None:
        result = {}
    rc = result.get("returncode", result.get("rc"))
    timed_out = bool(result.get("timed_out", False))
    elapsed = result.get("elapsed_seconds")
    notes = (
        "Modal training terminal recovery; score_claim=false; "
        f"promotion_eligible=false; rc={rc}; timed_out={timed_out}; "
        f"elapsed_seconds={elapsed}; out_dir={out_dir}"
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
]
