#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Calibrate a non-HNeRV architecture-class CUDA/CPU drift profile.

Sister of the HNeRV-cluster bootstrap landed 2026-05-08 in
``cuda_cpu_axis_profile_registry.bootstrap_registry_from_hnerv_anchors``. The
HNeRV cluster (PR100/101/102/103/105) gave us the canonical ``R_pose=5.04 ±
0.10`` + ``R_seg=1.17 ± 0.01`` + ``ε=0.033`` posterior. The other 9 architecture
classes still ship with ``uncalibrated_default`` profiles that carry 4× wider
drift bands.

This tool ingests a paired (CUDA, CPU) eval result for one archive and
produces ONE row of calibration evidence in the canonical drift profile
registry format. It is the per-class equivalent of the HNeRV bootstrap loop —
each invocation lands one anchor and re-computes the running posterior.

Inputs
------

The tool accepts EITHER:

  - ``--paired-eval-json <path>`` — a single JSON with ``cuda`` + ``cpu``
    blocks (the format ``contest_auth_eval`` writes for adjudicated rows), OR
  - explicit flags: ``--archive-sha256``, ``--architecture-class``,
    ``--cuda-pose --cuda-seg --cuda-score`` and ``--cpu-pose --cpu-seg --cpu-score``

Outputs
-------

  - The architecture-class profile in the registry is updated in place via
    :func:`tac.optimization.cuda_cpu_axis_profile_registry.update_profile_from_anchor`.
  - A per-anchor calibration record is written to the audit log at
    ``.omx/research/cuda_cpu_axis_profile_updates.jsonl``.
  - Stdout prints a summary including ``observed_r_pose``, ``observed_r_seg``,
    ``score_gap``, the new posterior mean/std after update, and whether the
    anchor was promoted (vs flagged as outlier per
    ``forbidden_premature_kill_without_research_exhaustion``).

Per CLAUDE.md "killing as last resort" the tool NEVER drops or kills evidence.
Outlier-flagged anchors stay in the audit trail and can be promoted later by
operator review.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
CONTEST-COMPLIANT HARDWARE": both axes must come from 1:1 contest-compliant
substrates. macOS substrates are refused. Tag-mismatched anchors are refused.

Cross-references
----------------
- :mod:`tac.optimization.cuda_cpu_axis_profile_registry`
- ``feedback_cuda_cpu_axis_profile_learning_layer_20260508``
- ``feedback_dual_cpu_cuda_auth_eval_mandatory_20260508``
- ``feedback_5_beyond_phase4_modules_landed_20260509``

CLAUDE.md compliance tags
-------------------------
- ``custody_validator_required``
- ``no_macos_authoritative``
- ``no_kill_only_flag``
- ``no_tmp_paths``
- ``forbidden_score_claims_no_score_returned``
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.continual_learning import CONTEST_CPU_LINUX_X86_64_SUBSTRATES  # noqa: E402
from tac.optimization.cuda_cpu_axis_profile_registry import (  # noqa: E402
    ARCHITECTURE_CLASSES,
    DEFAULT_AUDIT_LOG_PATH,
    DEFAULT_REGISTRY_PATH,
    ArchitectureProfile,
    read_registry,
    update_profile_from_anchor,
    write_registry,
)

CALIBRATION_SCHEMA = "tac_non_hnerv_drift_calibration_v1"

# Canonical hardware substrates that satisfy CLAUDE.md "1:1 contest-compliant"
# (mirrored from tac.continual_learning).
ALLOWED_CUDA_SUBSTRATES = frozenset({
    "linux_x86_64_t4",
    "linux_x86_64_4090",
    "linux_x86_64_a100",
    "linux_x86_64_h100",
    "linux_x86_64_a10g",
    "linux_x86_64_l40s",
})
ALLOWED_CPU_SUBSTRATES = CONTEST_CPU_LINUX_X86_64_SUBSTRATES

NON_HNERV_CLASSES = tuple(
    c for c in ARCHITECTURE_CLASSES
    if c not in {"hnerv_ft_microcodec", "hnerv_lc_v2"}
)


class CalibrationRefusalError(RuntimeError):
    """Raised when a calibration request fails the custody / consistency gates."""


def validate_paired_eval(payload: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    """Extract (cuda, cpu) component dicts from a paired-eval payload.

    Required schema (the contest_auth_eval adjudicated format)::

        {
          "archive_sha256": "<hex>",
          "architecture_class": "<class>",
          "cuda": {"pose": float, "seg": float, "score": float,
                   "hardware_substrate": "linux_x86_64_t4"},
          "cpu":  {"pose": float, "seg": float, "score": float,
                   "hardware_substrate": "linux_x86_64_modal_cpu"},
        }

    Raises :class:`CalibrationRefusalError` on any missing field, blank value,
    or substrate that is not in the allowed contest-compliant set.
    """
    if not isinstance(payload, dict):
        raise CalibrationRefusalError(
            f"paired eval payload must be a dict; got {type(payload).__name__}"
        )

    for field_name in ("archive_sha256", "architecture_class"):
        v = payload.get(field_name)
        if v is None or (isinstance(v, str) and not v.strip()):
            raise CalibrationRefusalError(
                f"required field {field_name!r} missing or blank"
            )

    arch = payload["architecture_class"]
    if arch not in ARCHITECTURE_CLASSES:
        raise CalibrationRefusalError(
            f"architecture_class {arch!r} not in canonical set "
            f"{sorted(ARCHITECTURE_CLASSES)}"
        )

    cuda = payload.get("cuda")
    cpu = payload.get("cpu")
    if not isinstance(cuda, dict) or not isinstance(cpu, dict):
        raise CalibrationRefusalError(
            "paired eval payload must include cuda and cpu blocks"
        )

    for axis_name, block, allowed in (
        ("cuda", cuda, ALLOWED_CUDA_SUBSTRATES),
        ("cpu", cpu, ALLOWED_CPU_SUBSTRATES),
    ):
        for k in ("pose", "seg", "score", "hardware_substrate"):
            if k not in block:
                raise CalibrationRefusalError(
                    f"axis {axis_name!r} missing required field {k!r}"
                )
        sub = block["hardware_substrate"]
        if sub not in allowed:
            raise CalibrationRefusalError(
                f"axis {axis_name!r} hardware_substrate {sub!r} not in 1:1 "
                f"contest-compliant set {sorted(allowed)}"
            )
        for k in ("pose", "seg", "score"):
            v = block[k]
            if not isinstance(v, (int, float)):
                raise CalibrationRefusalError(
                    f"axis {axis_name!r} field {k!r} must be numeric; got "
                    f"{type(v).__name__}"
                )
            if v != v:  # NaN check
                raise CalibrationRefusalError(
                    f"axis {axis_name!r} field {k!r} is NaN"
                )

    if cpu["pose"] <= 0:
        raise CalibrationRefusalError(
            "cpu pose must be > 0 to compute observed_r_pose"
        )
    if cpu["seg"] <= 0:
        raise CalibrationRefusalError(
            "cpu seg must be > 0 to compute observed_r_seg"
        )

    return cuda, cpu


def derive_anchor(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert a validated paired-eval payload to the anchor format expected by
    :func:`update_profile_from_anchor`.

    Returns a fully-populated anchor dict including provenance.
    """
    cuda, cpu = validate_paired_eval(payload)
    observed_r_pose = float(cuda["pose"]) / float(cpu["pose"])
    observed_r_seg = float(cuda["seg"]) / float(cpu["seg"])
    score_gap = float(cuda["score"]) - float(cpu["score"])

    return {
        "archive_sha256": payload["archive_sha256"],
        "architecture_class": payload["architecture_class"],
        "observed_r_pose": observed_r_pose,
        "observed_r_seg": observed_r_seg,
        "score_gap": score_gap,
        "cuda_pose": cuda["pose"],
        "cuda_seg": cuda["seg"],
        "cuda_score": cuda["score"],
        "cuda_substrate": cuda["hardware_substrate"],
        "cpu_pose": cpu["pose"],
        "cpu_seg": cpu["seg"],
        "cpu_score": cpu["score"],
        "cpu_substrate": cpu["hardware_substrate"],
        "source": payload.get("source", "calibrate_non_hnerv_drift_class"),
        "ingested_utc": dt.datetime.now(dt.UTC).isoformat(),
        "calibration_schema": CALIBRATION_SCHEMA,
    }


def calibrate_class(
    payload: dict[str, Any],
    *,
    registry_path: Path | None = None,
    audit_log_path: Path | None = None,
) -> dict[str, Any]:
    """Run one calibration cycle for one archive.

    Loads the registry, derives the anchor, applies it to the matching
    architecture-class profile, persists the registry, and appends an audit
    log entry. Returns a calibration record with ``before`` / ``after`` deltas.
    """
    reg_path = registry_path or DEFAULT_REGISTRY_PATH
    audit_path = audit_log_path or DEFAULT_AUDIT_LOG_PATH

    anchor = derive_anchor(payload)
    arch = anchor["architecture_class"]

    registry = read_registry(reg_path)
    profile = registry.get(arch)
    if profile is None:
        # Cold-start: instantiate a default profile for this class.
        profile = ArchitectureProfile(architecture_class=arch)
        registry[arch] = profile

    update = update_profile_from_anchor(profile, anchor)

    write_registry(registry, reg_path)

    record = {
        "schema": CALIBRATION_SCHEMA,
        "calibrated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "architecture_class": arch,
        "anchor": anchor,
        "update": {
            "accepted": update.accepted,
            "outlier_candidate": update.outlier_candidate,
            "before": update.before,
            "after": update.after,
            "notes": update.notes,
        },
    }

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")

    return record


def _payload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build a paired-eval payload from either --paired-eval-json or explicit flags."""
    if args.paired_eval_json:
        path = Path(args.paired_eval_json)
        if not path.is_file():
            raise CalibrationRefusalError(f"paired-eval-json not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    required = (
        "archive_sha256", "architecture_class",
        "cuda_pose", "cuda_seg", "cuda_score", "cuda_substrate",
        "cpu_pose", "cpu_seg", "cpu_score", "cpu_substrate",
    )
    missing = [k for k in required if getattr(args, k.replace("-", "_"), None) is None]
    if missing:
        raise CalibrationRefusalError(
            f"missing required flags (or --paired-eval-json): {missing}"
        )

    return {
        "archive_sha256": args.archive_sha256,
        "architecture_class": args.architecture_class,
        "cuda": {
            "pose": float(args.cuda_pose),
            "seg": float(args.cuda_seg),
            "score": float(args.cuda_score),
            "hardware_substrate": args.cuda_substrate,
        },
        "cpu": {
            "pose": float(args.cpu_pose),
            "seg": float(args.cpu_seg),
            "score": float(args.cpu_score),
            "hardware_substrate": args.cpu_substrate,
        },
        "source": args.source or "calibrate_non_hnerv_drift_class",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paired-eval-json", type=str, default=None,
                        help="Path to a paired-eval JSON; mutually exclusive with explicit flags")
    parser.add_argument("--archive-sha256", type=str, default=None)
    parser.add_argument("--architecture-class", type=str, default=None,
                        choices=ARCHITECTURE_CLASSES)
    parser.add_argument("--cuda-pose", type=float, default=None)
    parser.add_argument("--cuda-seg", type=float, default=None)
    parser.add_argument("--cuda-score", type=float, default=None)
    parser.add_argument("--cuda-substrate", type=str, default=None)
    parser.add_argument("--cpu-pose", type=float, default=None)
    parser.add_argument("--cpu-seg", type=float, default=None)
    parser.add_argument("--cpu-score", type=float, default=None)
    parser.add_argument("--cpu-substrate", type=str, default=None)
    parser.add_argument("--source", type=str, default=None,
                        help="Free-form provenance label (e.g. 'PR101_paired_eval_20260508')")
    parser.add_argument("--registry-path", type=Path, default=None)
    parser.add_argument("--audit-log-path", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        payload = _payload_from_args(args)
        record = calibrate_class(
            payload,
            registry_path=args.registry_path,
            audit_log_path=args.audit_log_path,
        )
    except CalibrationRefusalError as exc:
        print(f"calibrate_non_hnerv_drift_class: REFUSED — {exc}", file=sys.stderr)
        return 2

    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
