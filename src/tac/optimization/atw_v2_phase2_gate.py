# SPDX-License-Identifier: MIT
"""Fail-closed ATW v2 Phase-2 gate status from the D4 verdict artifact."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ATW_V2_D4_VERDICT_ARTIFACT_PATH = (
    ".omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.json"
)
ATW_V2_PHASE2_GATE_STATUS_SCHEMA = "atw_codec_v2_phase2_gate_status_v1"
_EXPECTED_VERDICT_SCHEMA = "atw_codec_v2_d4_probe_verdict_v1"
_MEANINGFUL_VERDICT = "MEANINGFUL_CONDITIONING"


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def atw_v2_phase2_gate_status(
    *,
    repo_root: str | Path | None = None,
    verdict_artifact_path: str = ATW_V2_D4_VERDICT_ARTIFACT_PATH,
) -> dict[str, Any]:
    """Return the current ATW v2 Phase-2 lift status.

    The status is deliberately conservative. A missing or malformed verdict
    blocks Phase-2 authority, and a diagnostic ``INDEPENDENT`` verdict keeps
    ATW v2 research-only. This function never returns score or promotion
    authority; paired exact CPU/CUDA custody must remain a separate gate.
    """

    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    )
    artifact_path = resolved_repo_root / verdict_artifact_path
    blockers: list[str] = []
    payload: Mapping[str, Any] | None = None
    artifact_sha256 = ""
    artifact_present = artifact_path.is_file()
    if not artifact_present:
        blockers.append("atw_v2_d4_probe_verdict_missing")
    else:
        artifact_sha256 = _sha256_file(artifact_path)
        try:
            loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            blockers.append(f"atw_v2_d4_probe_verdict_json_invalid:{exc.msg}")
        else:
            if isinstance(loaded, Mapping):
                payload = loaded
            else:
                blockers.append("atw_v2_d4_probe_verdict_not_object")

    if payload is None:
        return {
            "schema": ATW_V2_PHASE2_GATE_STATUS_SCHEMA,
            "substrate_id": "atw_codec_v2",
            "verdict_artifact_path": verdict_artifact_path,
            "verdict_artifact_present": artifact_present,
            "verdict_artifact_sha256": artifact_sha256,
            "d4_verdict": "MISSING",
            "phase2_status": "blocked_missing_d4_probe_verdict",
            "recommended_variant": "none",
            "next_action": "run_atw_v2_d4_probe_before_phase2_lift",
            "axis_label": "",
            "evidence_grade": "",
            "mutual_information_bits": None,
            "wyner_ziv_gain_ceiling_fraction": None,
            "dispatch_allowed": False,
            "phase2_lift_allowed": False,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "blockers": blockers,
        }

    verdict = str(
        _nested(payload, "canonical_wz_sideinfo_verdict", "verdict") or ""
    ).strip()
    phase2 = payload.get("atw_v2_phase2_status")
    phase2_status = ""
    recommended_variant = ""
    next_action = ""
    if isinstance(phase2, Mapping):
        phase2_status = str(phase2.get("phase2_status") or "").strip()
        recommended_variant = str(phase2.get("recommended_variant") or "").strip()
        next_action = str(phase2.get("next_action") or "").strip()
    axis_label = str(payload.get("axis_label") or "").strip()
    evidence_grade = str(payload.get("evidence_grade") or "").strip()
    mutual_information_bits = _float_or_none(
        _nested(payload, "canonical_wz_sideinfo_verdict", "mutual_information_bits")
    )
    wz_gain_ceiling_fraction = _float_or_none(
        _nested(
            payload,
            "canonical_wz_sideinfo_verdict",
            "wyner_ziv_gain_ceiling_fraction",
        )
    )

    if payload.get("schema") != _EXPECTED_VERDICT_SCHEMA:
        blockers.append("atw_v2_d4_probe_verdict_schema_mismatch")
    if verdict != _MEANINGFUL_VERDICT:
        blockers.append(f"atw_v2_phase2_deferred_by_d4_verdict:{verdict or 'missing'}")
    if payload.get("score_claim") is not False:
        blockers.append("atw_v2_d4_probe_score_claim_not_false")
    if payload.get("promotion_eligible") is not False:
        blockers.append("atw_v2_d4_probe_promotion_eligible_not_false")
    if payload.get("ready_for_exact_eval_dispatch") is not False:
        blockers.append("atw_v2_d4_probe_ready_for_exact_eval_dispatch_not_false")
    if payload.get("rank_or_kill_eligible") is not False:
        blockers.append("atw_v2_d4_probe_rank_or_kill_eligible_not_false")
    if payload.get("dispatch_attempted") is not False:
        blockers.append("atw_v2_d4_probe_dispatch_attempted_not_false")
    if "[diagnostic-CPU" not in axis_label:
        blockers.append("atw_v2_d4_probe_axis_label_missing_diagnostic_cpu")
    if evidence_grade != "diagnostic_cpu":
        blockers.append("atw_v2_d4_probe_evidence_grade_not_diagnostic_cpu")
    if phase2_status != "defer_measured_a1_latent_class_conditioning_surface":
        blockers.append("atw_v2_phase2_status_not_defer_for_current_verdict")
    if recommended_variant != "none":
        blockers.append("atw_v2_phase2_recommended_variant_not_none")
    if next_action != "do_not_dispatch_atw_v2_phase2_from_this_signal":
        blockers.append("atw_v2_phase2_next_action_not_fail_closed")

    return {
        "schema": ATW_V2_PHASE2_GATE_STATUS_SCHEMA,
        "substrate_id": "atw_codec_v2",
        "verdict_artifact_path": verdict_artifact_path,
        "verdict_artifact_present": artifact_present,
        "verdict_artifact_sha256": artifact_sha256,
        "d4_verdict": verdict or "MISSING",
        "phase2_status": phase2_status or "missing",
        "recommended_variant": recommended_variant or "none",
        "next_action": next_action or "missing",
        "axis_label": axis_label,
        "evidence_grade": evidence_grade,
        "mutual_information_bits": mutual_information_bits,
        "wyner_ziv_gain_ceiling_fraction": wz_gain_ceiling_fraction,
        "dispatch_allowed": False,
        "phase2_lift_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "blockers": list(dict.fromkeys(blockers)),
    }


__all__ = [
    "ATW_V2_D4_VERDICT_ARTIFACT_PATH",
    "ATW_V2_PHASE2_GATE_STATUS_SCHEMA",
    "atw_v2_phase2_gate_status",
]
