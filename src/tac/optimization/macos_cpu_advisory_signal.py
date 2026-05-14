# SPDX-License-Identifier: MIT
"""macOS-CPU advisory-signal manifests for free first-class proxy ranking.

Operator routing 2026-05-13 ("training is the real roadblock; we can prepare
and run things on macos and cpu"). Per CLAUDE.md empirical calibration:
PR107 M5 Max ``0.19664189`` matched GHA Linux x86_64 ``0.1966358879`` within
``6e-6``. So macOS CPU is a high-fidelity advisory proxy for the contest-CPU
axis, BUT must NEVER be treated as a 1:1 contest-compliant authoritative
score per the "Submission auth eval — BOTH CPU AND CUDA" non-negotiable.

This module mirrors :mod:`tac.optimization.mps_research_signal` in spirit:
the manifest is structurally non-promotable. The CRITICAL difference is the
allowed_uses set — macOS-CPU may participate in **autopilot dispatch ranking**
(so cheap free-pre-GPU ranking actually orders the queue) whereas MPS may
only seed proxy-curve discovery.

Per CLAUDE.md "Apples-to-apples evidence discipline" every row carries:
    score_claim=False, promotion_eligible=False,
    ready_for_exact_eval_dispatch=False,
    evidence_grade="macOS-CPU-advisory",
    evidence_tag="[macOS-CPU advisory only]".

Per CLAUDE.md Catalog #127 (`check_authoritative_tag_requires_custody_metadata`)
the macOS-CPU tag is already routed to `refused_class="macos_substrate"` by
:meth:`tac.continual_learning.ContestResult.validate_custody_verdict`. This
module never re-asserts authority over that contract.

Per CLAUDE.md Catalog #192 (this session): every persisted manifest row also
carries an explicit ``ranking_only=true`` flag. The companion preflight gate
``check_macos_cpu_advisory_not_promoted_without_linux_verification`` refuses
any persisted artifact that combines ``evidence_grade="macOS-CPU-advisory"``
with ``score_claim=true``, ``promotion_eligible=true``, or
``ready_for_exact_eval_dispatch=true``.
"""

from __future__ import annotations

import json
import math
import platform
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.score_geometry import contest_score

SCHEMA_VERSION = "macos_cpu_advisory_signal_manifest.v1"
EVIDENCE_GRADE = "macOS-CPU-advisory"
EVIDENCE_TAG = "[macOS-CPU advisory only]"
EVIDENCE_SEMANTICS = "macos_cpu_first_class_advisory_proxy_for_contest_cpu_axis"

# Per CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval — BOTH CPU
# AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE": the macOS-CPU axis is
# advisory only. PR107 empirical calibration shows |Δ| <= 6e-6 vs GHA Linux
# x86_64 contest-CPU on the same exact archive. We expose that as the
# placeholder confidence interval below; the sister empirical-validation
# subagent's :func:`load_calibration_model` output supersedes this when it
# lands on disk.
PR107_PLACEHOLDER_CALIBRATION = {
    "schema": "macos_cpu_to_contest_cpu_drift_calibration.placeholder.v1",
    "source": "CLAUDE.md PR107 anchor 2026-05-08",
    "anchor_count": 1,
    "drift_p50_abs": 6.1e-6,
    "drift_p90_abs": 6.1e-6,
    "drift_p99_abs": 6.1e-6,
    "high_variance_multiplier": 100.0,
    "calibration_status": "PR107_placeholder_pending_subagent_a_full_drift_table",
    "evidence_grade": EVIDENCE_GRADE,
    "promotable": False,
}

EVIDENCE_AXIS_NOTE = (
    "macOS-CPU is NOT a 1:1 contest-compliant axis. Per CLAUDE.md every "
    "shippable archive must ALSO get an authoritative [contest-CPU GHA "
    "Linux x86_64] anchor AND a [contest-CUDA] anchor before any "
    "submission/frontier/medal-band claim."
)

FORBIDDEN_USES = (
    "auth_eval_authoritative_cpu_axis",
    "contest_cpu_score_claim",
    "promotion_to_contest_cpu",
    "method_retirement_on_macos_cpu_alone",
    "frontier_anchor_without_paired_linux_x86_64",
    "submission_packet_without_dual_axis_paired",
)
ALLOWED_USES = (
    "autopilot_dispatch_ranking_pre_gpu_spend",
    "candidate_generation_prior",
    "smoke_test",
    "code_correctness_check",
    "free_local_pre_dispatch_proxy_signal",
    "dual_axis_pre_pairing_discovery",
)
DISPATCH_BLOCKERS = (
    "macos_cpu_advisory_not_score_evidence",
    "not_a_11_contest_compliant_cpu_axis",
    "requires_paired_contest_cpu_gha_linux_x86_64_before_score_claim",
    "requires_paired_contest_cuda_before_dual_axis_submission",
    "not_promotion_eligible",
)


class MacOSCPUAdvisorySignalError(ValueError):
    """Raised when a macOS-CPU advisory-signal observation is malformed."""


def is_running_on_macos_arm64() -> bool:
    """Return True iff the local platform is Darwin ARM64.

    Sister tools (e.g. the macOS-CPU smoke ranker CLI) call this to fail
    closed before running any eval that would otherwise be tagged
    ``macOS-CPU-advisory``. Running the manifest builder on a non-Darwin
    host produces malformed evidence (the tag refers specifically to the
    PR107-calibrated M5 Max / Apple Silicon proxy axis).
    """
    return platform.system() == "Darwin" and platform.machine() in {
        "arm64",
        "aarch64",
    }


def detect_macos_cpu_hardware_substrate() -> str:
    """Return a canonical hardware-substrate string for the current macOS host.

    The substrate string follows the same convention as
    :data:`tac.continual_learning.TAG_HARDWARE_REQUIREMENT` keys
    (``linux_x86_64_t4``, ``linux_x86_64_gha_cpu``, ...). For macOS hosts
    we emit ``darwin_arm64_<chip>_cpu`` where ``<chip>`` is a short
    fingerprint of the local processor when available. Per CLAUDE.md
    Catalog #127 the validator refuses this substrate as
    ``refused_class="macos_substrate"`` — by design.
    """
    if not is_running_on_macos_arm64():
        return f"non_macos_arm64_{platform.system().lower()}_{platform.machine().lower()}"
    try:
        import subprocess

        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError, ValueError):  # type: ignore[name-defined]
        chip = ""
    if not chip:
        return "darwin_arm64_apple_silicon_cpu"
    # Normalize: "Apple M5 Max" -> "apple_m5_max"
    lowered = chip.lower().replace(" ", "_").replace("-", "_")
    safe = "".join(c for c in lowered if c.isalnum() or c == "_")
    return f"darwin_arm64_{safe}_cpu"


def load_calibration_model(
    search_root: Path | None = None,
) -> dict[str, Any]:
    """Discover the sister empirical-validation subagent's calibration model.

    The sister lane ``lane_macos_cpu_proxy_empirical_validation_20260513``
    emits a structured drift-table + calibration model at
    ``experiments/results/lane_macos_cpu_proxy_empirical_validation_20260513_<UTC>/calibration_model.json``.

    Returns the loaded JSON if present (highest mtime wins) else the
    PR107 placeholder. Per CLAUDE.md "Subagent coherence-by-default" the
    autopilot ranker calls this on every invocation so the sister's
    output lands without code changes.
    """
    root = Path(search_root) if search_root is not None else Path("experiments/results")
    if not root.is_dir():
        return dict(PR107_PLACEHOLDER_CALIBRATION)
    matches: list[Path] = []
    try:
        for p in root.glob("lane_macos_cpu_proxy_empirical_validation_*/calibration_model.json"):
            if p.is_file():
                matches.append(p)
    except OSError:
        return dict(PR107_PLACEHOLDER_CALIBRATION)
    if not matches:
        return dict(PR107_PLACEHOLDER_CALIBRATION)
    latest = max(matches, key=lambda p: p.stat().st_mtime)
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(PR107_PLACEHOLDER_CALIBRATION)
    if not isinstance(payload, dict):
        return dict(PR107_PLACEHOLDER_CALIBRATION)
    payload.setdefault("evidence_grade", EVIDENCE_GRADE)
    payload.setdefault("promotable", False)
    payload["calibration_source_path"] = str(latest)
    payload.setdefault("calibration_status", "loaded_from_sister_subagent_empirical_table")
    return payload


def load_observations(path: Path) -> list[dict[str, Any]]:
    """Load macOS-CPU observations from JSON or JSONL."""
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
                raise MacOSCPUAdvisorySignalError(
                    f"{path}: JSON dict must contain observations[] or rows[]"
                )
            rows = raw_rows
        else:
            raise MacOSCPUAdvisorySignalError(f"{path}: expected JSON list or dict")
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise MacOSCPUAdvisorySignalError(f"{path}: row {index} is not an object")
        out.append(dict(row))
    return out


def build_macos_cpu_advisory_signal_manifest(
    observations: Iterable[Mapping[str, Any]],
    *,
    source: str,
    run_id: str,
    hardware_substrate: str | None = None,
    calibration_model: Mapping[str, Any] | None = None,
    paired_contest_cpu_anchor: Mapping[str, Any] | None = None,
    paired_contest_cuda_anchor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed macOS-CPU advisory-signal manifest.

    Args:
        observations: iterable of macOS-CPU evaluation rows. Each row must
            contain ``family``, ``variant_id``, ``archive_bytes``, and at
            least one of ``score`` (preferred) / ``d_seg`` / ``d_pose``.
        source: source label or path for custody.
        run_id: stable run id chosen by the caller.
        hardware_substrate: canonical substrate string (defaults to
            :func:`detect_macos_cpu_hardware_substrate`).
        calibration_model: optional drift-calibration model from the sister
            empirical-validation lane. Defaults to
            :data:`PR107_PLACEHOLDER_CALIBRATION`.
        paired_contest_cpu_anchor: optional metadata for a paired
            ``[contest-CPU GHA Linux x86_64]`` result on the same archive,
            for dual-axis discipline. The presence of this field DOES NOT
            promote any row — it is recorded for cross-axis audit only.
        paired_contest_cuda_anchor: optional metadata for a paired
            ``[contest-CUDA]`` result on the same archive. Same
            non-promoting semantics.

    The output manifest's ``score_claim``, ``promotion_eligible``, and
    ``ready_for_exact_eval_dispatch`` are PERMANENTLY ``False``.
    """
    substrate = hardware_substrate or detect_macos_cpu_hardware_substrate()
    if calibration_model is None:
        calibration_model = dict(PR107_PLACEHOLDER_CALIBRATION)
    else:
        calibration_model = dict(calibration_model)
        calibration_model.setdefault("evidence_grade", EVIDENCE_GRADE)
        calibration_model.setdefault("promotable", False)
    rows = [
        _normalize_observation(
            row,
            index=index,
            hardware_substrate=substrate,
            calibration_model=calibration_model,
        )
        for index, row in enumerate(observations)
    ]
    ranking_atoms = [_row_to_ranking_atom(row) for row in rows]
    manifest: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "source": source,
        "run_id": run_id,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_tag": EVIDENCE_TAG,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "ranking_only": True,
        "hardware_substrate": substrate,
        "device_contract": {
            "device_family": "macos_cpu",
            "allowed_uses": list(ALLOWED_USES),
            "forbidden_uses": list(FORBIDDEN_USES),
            "promotion_requires_paired_linux_x86_64": True,
            "axis_note": EVIDENCE_AXIS_NOTE,
        },
        "calibration_model": calibration_model,
        "paired_contest_cpu_anchor": (
            dict(paired_contest_cpu_anchor) if paired_contest_cpu_anchor else None
        ),
        "paired_contest_cuda_anchor": (
            dict(paired_contest_cuda_anchor) if paired_contest_cuda_anchor else None
        ),
        "row_count": len(rows),
        "rows": rows,
        "ranking_atoms": ranking_atoms,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
    }
    return manifest


def _normalize_observation(
    row: Mapping[str, Any],
    *,
    index: int,
    hardware_substrate: str,
    calibration_model: Mapping[str, Any],
) -> dict[str, Any]:
    family = _required_text(row, "family", index=index)
    variant_id = _required_text(row, "variant_id", index=index)
    archive_bytes = _required_positive_int(row, "archive_bytes", index=index)
    score = _optional_float(row.get("score"))
    d_seg = _optional_float(row.get("d_seg", row.get("d_seg_macos_cpu")))
    d_pose = _optional_float(row.get("d_pose", row.get("d_pose_macos_cpu")))
    if score is None and d_seg is None and d_pose is None:
        raise MacOSCPUAdvisorySignalError(
            f"row {index}: must supply score, d_seg, or d_pose"
        )
    if d_seg is not None and d_seg < 0.0:
        raise MacOSCPUAdvisorySignalError(f"row {index}: d_seg must be non-negative")
    if d_pose is not None and d_pose < 0.0:
        raise MacOSCPUAdvisorySignalError(f"row {index}: d_pose must be non-negative")
    if score is not None and not math.isfinite(score):
        raise MacOSCPUAdvisorySignalError(f"row {index}: score must be finite")

    score_formula_macos_cpu: float | None = None
    if d_seg is not None and d_pose is not None:
        score_formula_macos_cpu = float(contest_score(d_seg, d_pose, archive_bytes))

    # Confidence interval on the projected contest-CPU score given the
    # macOS-CPU observation + the drift-calibration model. Per CLAUDE.md
    # "higher variance" rule for proxy evidence the band is wider than the
    # calibration's raw drift_p90 (we apply the high-variance multiplier).
    drift_p90 = float(calibration_model.get("drift_p90_abs", PR107_PLACEHOLDER_CALIBRATION["drift_p90_abs"]))
    high_var_mul = float(calibration_model.get("high_variance_multiplier", 100.0))
    advisory_band_half_width = drift_p90 * high_var_mul

    projected_contest_cpu_score_p50: float | None = None
    projected_contest_cpu_score_low: float | None = None
    projected_contest_cpu_score_high: float | None = None
    if score is not None:
        projected_contest_cpu_score_p50 = score
        projected_contest_cpu_score_low = score - advisory_band_half_width
        projected_contest_cpu_score_high = score + advisory_band_half_width

    return {
        "row_index": index,
        "family": family,
        "variant_id": variant_id,
        "params": dict(row.get("params") or {}),
        "device": "macos_cpu",
        "hardware_substrate": hardware_substrate,
        "archive_bytes": archive_bytes,
        "archive_sha256": str(row.get("archive_sha256") or ""),
        "score_macos_cpu": score,
        "d_seg_macos_cpu": d_seg,
        "d_pose_macos_cpu": d_pose,
        "score_formula_macos_cpu": score_formula_macos_cpu,
        "projected_contest_cpu_score_p50": projected_contest_cpu_score_p50,
        "projected_contest_cpu_score_low": projected_contest_cpu_score_low,
        "projected_contest_cpu_score_high": projected_contest_cpu_score_high,
        "advisory_band_half_width": advisory_band_half_width,
        "wall_clock_seconds": _optional_float(row.get("wall_clock_seconds")),
        "samples_evaluated": _optional_int(row.get("samples_evaluated")),
        "source_artifact": str(row.get("source_artifact") or ""),
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_tag": EVIDENCE_TAG,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "ranking_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "proxy_evidence": "macos_cpu_advisory",
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
    }


def _row_to_ranking_atom(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a normalized row to an autopilot-consumable ranking atom.

    Per CLAUDE.md operator routing 2026-05-13: macOS-CPU rows participate
    in RANKING (so cheap candidates can be ordered before GPU spend) but
    NEVER promote. The atom's ``rankable`` is True; the
    ``promotion_eligible`` / ``ready_for_exact_eval_dispatch`` flags stay
    False. The autopilot's dispatch journal will tag the entry with
    ``proxy_evidence="macos_cpu_advisory"``.
    """
    score = row.get("score_macos_cpu")
    return {
        "atom_id": f"macos_cpu_advisory:{row['family']}:{row['variant_id']}",
        "family": str(row["family"]),
        "family_group": f"macos_cpu_advisory:{row['family']}",
        "pareto_scope": f"macos_cpu_advisory:{row['family']}",
        "archive_bytes": int(row["archive_bytes"]),
        "archive_sha256": str(row.get("archive_sha256") or ""),
        "projected_contest_cpu_score_p50": row.get("projected_contest_cpu_score_p50"),
        "projected_contest_cpu_score_low": row.get("projected_contest_cpu_score_low"),
        "projected_contest_cpu_score_high": row.get("projected_contest_cpu_score_high"),
        "advisory_band_half_width": float(row.get("advisory_band_half_width") or 0.0),
        "score_macos_cpu": float(score) if score is not None else None,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_tag": EVIDENCE_TAG,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "ranking_only": True,
        "rankable": True,
        "proxy_evidence": "macos_cpu_advisory",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatchable": False,
        "interaction_assumptions": [
            "macos_cpu_pr107_calibration_within_6e-6_of_contest_cpu_gha_linux_x86_64",
            "macos_cpu_advisory_band_widened_by_calibration_high_variance_multiplier",
            "non_promotable_until_paired_linux_x86_64_lands",
        ],
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
    }


def _required_text(row: Mapping[str, Any], key: str, *, index: int) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise MacOSCPUAdvisorySignalError(f"row {index}: missing {key}")
    return value


def _required_positive_int(row: Mapping[str, Any], key: str, *, index: int) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MacOSCPUAdvisorySignalError(f"row {index}: {key} must be an integer")
    if value <= 0:
        raise MacOSCPUAdvisorySignalError(f"row {index}: {key} must be positive")
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise MacOSCPUAdvisorySignalError("boolean is not a valid int")
    if not isinstance(value, int):
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise MacOSCPUAdvisorySignalError(f"could not parse int from {value!r}") from exc
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise MacOSCPUAdvisorySignalError("boolean is not a valid float")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise MacOSCPUAdvisorySignalError(f"could not parse float from {value!r}") from exc
    if not math.isfinite(out):
        raise MacOSCPUAdvisorySignalError(f"non-finite float {value!r}")
    return out


def json_text(payload: Any) -> str:
    """Deterministic JSON text for manifest outputs."""
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def append_manifest_row_to_jsonl(
    row: Mapping[str, Any],
    *,
    output_path: Path,
) -> None:
    """Append a single manifest row to a JSONL aggregator.

    Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact": refuses
    ``/tmp``, ``/var/tmp``, ``/private/tmp`` prefixes.
    """
    output_str = str(output_path)
    if output_str.startswith("/tmp/") or "/private/tmp/" in output_str or "/var/tmp/" in output_str:
        raise ValueError(
            f"refusing to write macOS-CPU advisory manifest to forbidden /tmp path: {output_str!r}"
        )
    serializable = dict(row)
    # Per CLAUDE.md "Apples-to-apples evidence discipline" enforce flag set.
    serializable.setdefault("score_claim", False)
    serializable.setdefault("promotion_eligible", False)
    serializable.setdefault("ready_for_exact_eval_dispatch", False)
    serializable.setdefault("ranking_only", True)
    serializable.setdefault("evidence_grade", EVIDENCE_GRADE)
    serializable.setdefault("evidence_tag", EVIDENCE_TAG)
    if (
        serializable["score_claim"]
        or serializable["promotion_eligible"]
        or serializable["ready_for_exact_eval_dispatch"]
    ):
        raise MacOSCPUAdvisorySignalError(
            "macOS-CPU advisory manifest rows cannot carry score_claim=True, "
            "promotion_eligible=True, or ready_for_exact_eval_dispatch=True. "
            "Per CLAUDE.md Catalog #192 and the new STRICT preflight gate "
            "check_macos_cpu_advisory_not_promoted_without_linux_verification, "
            "promotion requires a paired [contest-CPU GHA Linux x86_64] anchor."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(serializable, sort_keys=True, allow_nan=False)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


__all__ = [
    "ALLOWED_USES",
    "DISPATCH_BLOCKERS",
    "EVIDENCE_GRADE",
    "EVIDENCE_SEMANTICS",
    "EVIDENCE_TAG",
    "FORBIDDEN_USES",
    "PR107_PLACEHOLDER_CALIBRATION",
    "SCHEMA_VERSION",
    "MacOSCPUAdvisorySignalError",
    "append_manifest_row_to_jsonl",
    "build_macos_cpu_advisory_signal_manifest",
    "detect_macos_cpu_hardware_substrate",
    "is_running_on_macos_arm64",
    "json_text",
    "load_calibration_model",
    "load_observations",
]
