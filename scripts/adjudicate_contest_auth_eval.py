#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Adjudicate a contest_auth_eval.json result without regex-log scraping.

Remote lane scripts should treat ``experiments/contest_auth_eval.py`` as the
source of truth and read its JSON artifact directly. This helper validates the
artifact against the exact archive bytes that were evaluated, updates the lane
provenance, and emits a few shell-safe KEY=VALUE lines for completion logs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    _bootstrap_path = Path(__file__).resolve().parent.parent / "tools" / "tool_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("tool_bootstrap", _bootstrap_path)
    if _spec is None or _spec.loader is None:
        raise
    _tool_bootstrap = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_tool_bootstrap)
    ensure_repo_imports = _tool_bootstrap.ensure_repo_imports
    repo_root_from_tool = _tool_bootstrap.repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, read_json, sha256_file  # noqa: E402

_COMPONENT_JSON_KEYS = {
    "posenet": "avg_posenet_dist",
    "segnet": "avg_segnet_dist",
}
_ACTIVE_DISTILLATION_FAMILIES = {
    "segnet_aux_kl",
    "primary_scorer_kl",
    "segnet_kl_legacy",
    "jbl",
}
_DISTILLATION_TEXT_MARKERS = (
    "kl_distill",
    "kl-distill",
    "kldistill",
    "distillation",
    "distill",
    "jbl",
)
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_RAW_PROMOTION_BLOCKER_FIELDS = (
    "promotion_blockers",
    "rank_or_kill_blockers",
)


def _expected_score_axis(device: str) -> str:
    if device == "cuda":
        return "contest_cuda"
    if device == "cpu":
        return "contest_cpu"
    raise SystemExit(f"FATAL: unsupported required device for adjudication: {device!r}")


def _score_tag_for_axis(score_axis: str) -> str:
    if score_axis == "contest_cuda":
        return "[contest-CUDA]"
    if score_axis == "contest_cpu":
        return "[contest-CPU]"
    raise SystemExit(f"FATAL: unsupported score axis for adjudication: {score_axis!r}")


def _require_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SystemExit(f"FATAL: contest_auth_eval.json missing numeric {key!r}")
    out = float(value)
    if not math.isfinite(out):
        raise SystemExit(f"FATAL: contest_auth_eval.json {key!r} is not finite: {value!r}")
    return out


def _optional_number(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _optional_arg_number(args: argparse.Namespace, key: str) -> float | None:
    value = getattr(args, key, None)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SystemExit(f"FATAL: {key} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise SystemExit(f"FATAL: {key} must be finite")
    return out


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _manifest_identity(manifest: dict[str, Any], *, manifest_path: Path) -> dict[str, Any]:
    declared = manifest.get("manifest_sha256")
    without_self = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    canonical = _canonical_json_sha256(without_self)
    if declared is not None:
        if not isinstance(declared, str) or not _SHA256_HEX_RE.fullmatch(declared):
            raise ValueError(f"source manifest has invalid manifest_sha256: {manifest_path}")
        if declared != canonical:
            raise ValueError(
                "source manifest manifest_sha256 is stale: "
                f"{manifest_path} expected={declared} actual={canonical}"
            )
    return {
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": declared or canonical,
        "source_manifest_file_sha256": sha256_file(manifest_path),
        "source_manifest_file_count": manifest.get("file_count"),
        "source_manifest_total_bytes": manifest.get("total_bytes"),
    }


def _manifest_entries_by_path(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError(f"source manifest missing files list: {manifest_path}")
    out: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            raise ValueError(f"source manifest files[{index}] must be an object: {manifest_path}")
        rel = item.get("path")
        if not isinstance(rel, str) or not rel or rel.startswith("/") or "\\" in rel:
            raise ValueError(f"source manifest files[{index}].path is not repo-relative: {rel!r}")
        parts = rel.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError(f"source manifest files[{index}].path is unstable: {rel!r}")
        if rel in out:
            raise ValueError(f"source manifest contains duplicate path: {rel}")
        out[rel] = item
    return out


def _manifest_file_meta(
    entries: dict[str, dict[str, Any]],
    rel: str,
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    entry = entries.get(rel)
    if entry is None:
        raise ValueError(f"source manifest missing runtime file: {rel}")
    byte_count = entry.get("bytes")
    sha256 = entry.get("sha256")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
        raise ValueError(f"source manifest runtime file lacks integer bytes: {rel}")
    if not isinstance(sha256, str) or not _SHA256_HEX_RE.fullmatch(sha256):
        raise ValueError(f"source manifest runtime file lacks sha256: {rel}")
    local = REPO_ROOT / rel
    if local.is_file():
        actual_bytes = local.stat().st_size
        actual_sha = sha256_file(local)
        if actual_bytes != byte_count or actual_sha != sha256:
            raise ValueError(
                "source manifest runtime file byte/SHA mismatch: "
                f"{rel} expected bytes={byte_count} sha256={sha256} "
                f"actual bytes={actual_bytes} sha256={actual_sha}"
            )
    return {"path": rel, "bytes": byte_count, "sha256": sha256}


def _load_sibling_json(path: Path, *, label: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _source_manifest_path_from_metadata(
    metadata: dict[str, Any],
    *,
    artifact_dir: Path,
) -> Path | None:
    queue_metadata = metadata.get("queue_metadata")
    if not isinstance(queue_metadata, dict):
        return None
    raw = queue_metadata.get("source_manifest")
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists():
        return repo_candidate
    return artifact_dir / path


def _iter_eval_runtime_files(runtime_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in runtime_manifest.get("files") or []:
        if isinstance(item, dict):
            out.append(item)
    for root in runtime_manifest.get("external_dependency_roots") or []:
        if not isinstance(root, dict):
            continue
        for item in root.get("files") or []:
            if isinstance(item, dict):
                out.append(item)
    repo_local_tac = runtime_manifest.get("repo_local_tac_import_manifest")
    if isinstance(repo_local_tac, dict):
        for item in repo_local_tac.get("files") or []:
            if isinstance(item, dict):
                out.append(item)
    return out


def _check_source_manifest_runtime_closure(
    *,
    eval_provenance: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    metadata_path = artifact_dir / "lightning_queue_metadata.json"
    eval_provenance_path = artifact_dir / "eval_provenance.json"
    metadata = _load_sibling_json(metadata_path, label="Lightning queue metadata") or {}
    queue_metadata = metadata.get("queue_metadata") if isinstance(metadata.get("queue_metadata"), dict) else {}
    source_manifest_path = _source_manifest_path_from_metadata(metadata, artifact_dir=artifact_dir)
    runtime_manifest = eval_provenance.get("inflate_runtime_manifest")
    available = source_manifest_path is not None and source_manifest_path.is_file() and isinstance(runtime_manifest, dict)
    result: dict[str, Any] = {
        "schema": "source_manifest_runtime_closure_adjudication_v1",
        "metadata_path": str(metadata_path) if metadata_path.is_file() else None,
        "eval_provenance_path": str(eval_provenance_path) if eval_provenance_path.is_file() else None,
        "source_manifest_path": str(source_manifest_path) if source_manifest_path is not None else None,
        "available": available,
        "verified": False,
        "violations": [],
        "checked_runtime_file_count": 0,
        "checked_external_dependency_root_count": 0,
    }
    if source_manifest_path is None and isinstance(queue_metadata, dict):
        expected_sha = queue_metadata.get("source_manifest_sha256")
        if isinstance(expected_sha, str):
            result["source_manifest_sha256"] = expected_sha
        return result
    if source_manifest_path is None:
        return result
    if not source_manifest_path.is_file():
        result["violations"].append(f"source_manifest_not_found:{source_manifest_path}")
        return result
    if not isinstance(runtime_manifest, dict):
        result["violations"].append("eval_provenance_missing_inflate_runtime_manifest")
        return result

    try:
        source_manifest = read_json(source_manifest_path)
        if not isinstance(source_manifest, dict):
            raise ValueError("source manifest must be a JSON object")
        identity = _manifest_identity(source_manifest, manifest_path=source_manifest_path)
        entries = _manifest_entries_by_path(source_manifest, manifest_path=source_manifest_path)
        result.update(identity)
        if isinstance(queue_metadata, dict):
            expected_manifest_sha = queue_metadata.get("source_manifest_sha256")
            if (
                isinstance(expected_manifest_sha, str)
                and expected_manifest_sha != identity["source_manifest_sha256"]
            ):
                result["violations"].append(
                    "queue_metadata_source_manifest_sha256_mismatch:"
                    f"expected={expected_manifest_sha} actual={identity['source_manifest_sha256']}"
                )
        checked_files = []
        for item in _iter_eval_runtime_files(runtime_manifest):
            rel = item.get("repo_relative_path") or item.get("relative_path")
            if not isinstance(rel, str) or not rel:
                result["violations"].append("runtime_manifest_file_missing_repo_relative_path")
                continue
            expected = _manifest_file_meta(entries, rel, manifest_path=source_manifest_path)
            actual = {
                "path": rel,
                "bytes": item.get("bytes"),
                "sha256": item.get("sha256"),
            }
            if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
                result["violations"].append(
                    "runtime_file_byte_sha_mismatch:"
                    f"{rel}: source_manifest={expected['bytes']}/{expected['sha256']} "
                    f"eval_runtime={actual['bytes']}/{actual['sha256']}"
                )
            checked_files.append(expected)
        external_roots = runtime_manifest.get("external_dependency_roots") or []
        checked_roots = []
        for root in external_roots:
            if not isinstance(root, dict):
                result["violations"].append("runtime_manifest_external_root_not_object")
                continue
            root_rel = root.get("repo_relative_root")
            if not isinstance(root_rel, str) or not root_rel:
                result["violations"].append("runtime_manifest_external_root_missing_repo_relative_root")
                continue
            if root.get("exists") is not True:
                result["violations"].append(f"runtime_manifest_external_root_missing:{root_rel}")
            checked_roots.append(root_rel)
        closure_basis = {
            "source_manifest_sha256": result.get("source_manifest_sha256"),
            "runtime_tree_sha256": runtime_manifest.get("runtime_tree_sha256"),
            "checked_files": sorted(checked_files, key=lambda row: row["path"]),
            "external_dependency_roots": sorted(checked_roots),
        }
        result.update(
            {
                "runtime_tree_sha256": runtime_manifest.get("runtime_tree_sha256"),
                "checked_runtime_file_count": len(checked_files),
                "checked_external_dependency_root_count": len(checked_roots),
                "checked_external_dependency_roots": sorted(checked_roots),
                "checked_runtime_total_bytes": sum(int(row["bytes"]) for row in checked_files),
                "closure_sha256": _canonical_json_sha256(closure_basis),
                "verified": not result["violations"],
            }
        )
    except Exception as exc:
        result["violations"].append(f"source_manifest_runtime_closure_error:{exc}")
    return result


def _distillation_policy_sha256(policy: dict[str, Any]) -> str:
    try:
        from tac.kl_config import distillation_policy_sha256

        return distillation_policy_sha256(policy)
    except Exception:
        return _canonical_json_sha256(policy)


def _is_positive_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return math.isfinite(float(value)) and float(value) > 0.0


def _policy_active(policy: Any) -> bool:
    if not isinstance(policy, dict):
        return False
    family = policy.get("family")
    return (
        family in _ACTIVE_DISTILLATION_FAMILIES
        and _is_positive_number(policy.get("weight"))
    )


def _text_has_distillation_marker(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return any(marker in lowered for marker in _DISTILLATION_TEXT_MARKERS)


def _distillation_active_from_provenance(provenance: dict[str, Any]) -> bool:
    policy = provenance.get("distillation_policy")
    if _policy_active(policy):
        return True
    for key in ("kl_distill_weight", "distill_weight", "jbl_weight"):
        if _is_positive_number(provenance.get(key)):
            return True
    for key in (
        "family",
        "loss_mode",
        "variant",
        "paradigm",
        "lane_name",
        "lane_script",
        "delta_from_lane_a",
    ):
        if _text_has_distillation_marker(provenance.get(key)):
            return True
    return False


def _check_distillation_promotion_gate(
    *,
    provenance: dict[str, Any],
    device: Any,
    archive_sha256: str,
    archive_bytes: int,
    component_gates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fail closed before promoting KL/JBL/distillation-active artifacts."""

    active = _distillation_active_from_provenance(provenance)
    policy = provenance.get("distillation_policy")
    policy_sha = (
        provenance.get("distillation_policy_sha256")
        or provenance.get("policy_sha256")
    )
    expected_policy_sha = None
    violations: list[dict[str, Any]] = []

    if not active:
        return {
            "active": False,
            "policy_sha256": policy_sha,
            "expected_policy_sha256": expected_policy_sha,
            "violations": violations,
        }

    if device != "cuda":
        violations.append(
            {
                "reason": "distillation_requires_exact_cuda",
                "observed_device": device,
                "expected_device": "cuda",
            }
        )
    if not archive_sha256 or len(archive_sha256) != 64:
        violations.append(
            {"reason": "missing_archive_sha256", "archive_sha256": archive_sha256}
        )
    if not isinstance(archive_bytes, int) or archive_bytes <= 0:
        violations.append(
            {"reason": "missing_archive_bytes", "archive_bytes": archive_bytes}
        )

    if not isinstance(policy, dict):
        violations.append({"reason": "missing_distillation_policy_v1"})
        if not isinstance(policy_sha, str) or len(policy_sha) != 64:
            violations.append({"reason": "missing_distillation_policy_sha256"})
    else:
        if policy.get("format") != "distillation_policy_v1":
            violations.append(
                {
                    "reason": "invalid_distillation_policy_format",
                    "format": policy.get("format"),
                }
            )
        if policy.get("schema_version") != 1:
            violations.append(
                {
                    "reason": "invalid_distillation_policy_schema_version",
                    "schema_version": policy.get("schema_version"),
                }
            )
        if not _policy_active(policy):
            violations.append(
                {
                    "reason": "inactive_distillation_policy",
                    "family": policy.get("family"),
                    "weight": policy.get("weight"),
                }
            )
        if policy.get("promotion_capable") is not True:
            violations.append(
                {
                    "reason": "distillation_policy_not_promotion_capable",
                    "family": policy.get("family"),
                    "promotion_capable": policy.get("promotion_capable"),
                    "promotion_blockers": policy.get("promotion_blockers"),
                }
            )
        expected_policy_sha = _distillation_policy_sha256(policy)
        if not isinstance(policy_sha, str) or len(policy_sha) != 64:
            violations.append({"reason": "missing_distillation_policy_sha256"})
        elif policy_sha != expected_policy_sha:
            violations.append(
                {
                    "reason": "distillation_policy_sha256_mismatch",
                    "distillation_policy_sha256": policy_sha,
                    "expected_distillation_policy_sha256": expected_policy_sha,
                }
            )

    gated_components = {
        str(gate.get("component"))
        for gate in component_gates
        if gate.get("passed") is True
    }
    missing_components = sorted({"posenet", "segnet"} - gated_components)
    if missing_components:
        violations.append(
            {
                "reason": "missing_component_noncollapse_gate",
                "missing_components": missing_components,
            }
        )

    return {
        "active": True,
        "policy_sha256": policy_sha,
        "expected_policy_sha256": expected_policy_sha,
        "violations": violations,
    }


def _as_blocker_list(value: Any) -> list[str]:
    """Normalize raw blocker fields without treating malformed data as clean."""

    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [value] if value else []
    return [f"malformed_blocker_field:{type(value).__name__}:{value!r}"]


def _check_raw_promotion_policy_gate(payload: dict[str, Any]) -> dict[str, Any]:
    """Fail closed when upstream/raw adjudication blockers are still present.

    This catches re-adjudication of partially blocked artifacts. A result may
    pass local component/regression/hardware checks but still carry raw
    `promotion_blockers` or `rank_or_kill_blockers` from earlier custody layers
    such as missing CPU pairing or missing strict pre-submission compliance.
    Those blockers must not be laundered into `promotion_eligible=true`.
    """

    blocker_fields: dict[str, list[str]] = {}
    violations: list[dict[str, Any]] = []
    for field in _RAW_PROMOTION_BLOCKER_FIELDS:
        blockers = _as_blocker_list(payload.get(field))
        if not blockers:
            continue
        blocker_fields[field] = blockers
        violations.append(
            {
                "field": field,
                "blockers": blockers,
                "reason": "raw_promotion_policy_blockers_present",
            }
        )
    return {
        "triggered": bool(violations),
        "blocker_fields": blocker_fields,
        "violations": violations,
    }


def _regression_threshold(args: argparse.Namespace) -> tuple[float, str]:
    threshold = getattr(args, "regression_threshold", None)
    if threshold is not None:
        return float(threshold), "delta_vs_baseline"
    # Backward-compatible direct-call support for older tests/tools. The
    # deprecated hard-kill flag was historically an absolute score ceiling.
    threshold = getattr(args, "hard_kill_above", None)
    if threshold is not None:
        return float(threshold), "absolute_score"
    raise SystemExit("FATAL: missing --regression-threshold")


def _status(score: float, predicted_low: float, predicted_high: float, regression_triggered: bool) -> str:
    if regression_triggered:
        return "REGRESSION_REVIEW_REQUIRED"
    if predicted_low <= score <= predicted_high:
        return "IN_PREDICTED_BAND"
    return "OUT_OF_PREDICTED_BAND"


def _add_review_status(lane_status: str, review_prefix: str) -> str:
    """Append a review class without hiding earlier non-promotion reasons."""

    if lane_status in {"IN_PREDICTED_BAND", "OUT_OF_PREDICTED_BAND"}:
        return f"{review_prefix}_REVIEW_REQUIRED"
    suffix = "_REVIEW_REQUIRED"
    if lane_status.endswith(suffix):
        base = lane_status[: -len(suffix)]
        if review_prefix not in base.split("_AND_"):
            return f"{base}_AND_{review_prefix}{suffix}"
    return lane_status


def _check_component_gates(
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, float | None], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate optional PoseNet/SegNet gates from contest_auth_eval.json only."""

    components = {
        json_key: _optional_number(payload, json_key)
        for json_key in _COMPONENT_JSON_KEYS.values()
    }
    gates: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    reference_label = getattr(args, "component_reference_label", "baseline") or "baseline"

    for component, json_key in _COMPONENT_JSON_KEYS.items():
        max_abs = _optional_arg_number(args, f"max_{component}_dist")
        reference = _optional_arg_number(args, f"baseline_{component}_dist")
        max_relative = _optional_arg_number(args, f"max_{component}_relative")

        if max_abs is not None and max_abs < 0.0:
            raise SystemExit(f"FATAL: max_{component}_dist must be non-negative")
        if reference is not None and reference <= 0.0:
            raise SystemExit(f"FATAL: baseline_{component}_dist must be positive")
        if max_relative is not None:
            if max_relative <= 0.0:
                raise SystemExit(f"FATAL: max_{component}_relative must be positive")
            if reference is None:
                raise SystemExit(
                    f"FATAL: --max-{component}-relative requires "
                    f"--baseline-{component}-dist or --reference-{component}-dist"
                )

        if max_abs is None and max_relative is None:
            continue

        observed = _require_number(payload, json_key)
        if observed < 0.0:
            raise SystemExit(f"FATAL: contest_auth_eval.json {json_key!r} is negative: {observed}")
        components[json_key] = observed
        gate: dict[str, Any] = {
            "component": component,
            "metric": json_key,
            "observed": observed,
            "max_absolute": max_abs,
            "reference": reference,
            "reference_label": reference_label,
            "max_relative": max_relative,
            "relative_to_reference": None,
            "passed": True,
        }
        if max_abs is not None and observed > max_abs:
            gate["passed"] = False
            violations.append(
                {
                    "component": component,
                    "metric": json_key,
                    "observed": observed,
                    "max_absolute": max_abs,
                    "reason": "absolute_component_gate",
                }
            )
        if max_relative is not None:
            assert reference is not None
            relative = observed / reference
            gate["relative_to_reference"] = relative
            if relative > max_relative:
                gate["passed"] = False
                violations.append(
                    {
                        "component": component,
                        "metric": json_key,
                        "observed": observed,
                        "reference": reference,
                        "reference_label": reference_label,
                        "relative_to_reference": relative,
                        "max_relative": max_relative,
                        "reason": "relative_component_gate",
                    }
                )
        gates.append(gate)

    return components, gates, violations


def adjudicate(args: argparse.Namespace) -> dict[str, Any]:
    contest_json = Path(args.contest_json)
    archive = Path(args.archive)
    provenance_path = Path(args.provenance)

    if not contest_json.is_file():
        raise SystemExit(f"FATAL: contest auth JSON not found: {contest_json}")
    if not archive.is_file():
        raise SystemExit(f"FATAL: evaluated archive not found: {archive}")

    payload = read_json(contest_json)
    score_recomputed = _require_number(payload, "score_recomputed_from_components")
    final_score = _require_number(payload, "final_score")
    sane_score_gate_triggered = not (0.0 < score_recomputed < args.max_sane_score)
    sane_score_gate_violation = (
        {
            "metric": "score_recomputed_from_components",
            "observed": score_recomputed,
            "exclusive_min": 0.0,
            "exclusive_max": args.max_sane_score,
            "reason": "sane_score_gate",
        }
        if sane_score_gate_triggered
        else None
    )

    n_samples = payload.get("n_samples")
    if n_samples != args.required_samples:
        raise SystemExit(
            f"FATAL: contest_auth_eval.json n_samples={n_samples!r}, "
            f"expected {args.required_samples}"
        )

    actual_archive_bytes = archive.stat().st_size
    actual_archive_sha256 = sha256_file(archive)
    payload_archive_bytes = payload.get("archive_size_bytes")
    if payload_archive_bytes != actual_archive_bytes:
        raise SystemExit(
            f"FATAL: contest_auth_eval archive_size_bytes={payload_archive_bytes!r}, "
            f"actual archive bytes={actual_archive_bytes}"
        )

    eval_provenance = payload.get("provenance")
    if not isinstance(eval_provenance, dict):
        raise SystemExit("FATAL: contest_auth_eval.json missing provenance object")
    required_device = str(args.required_device).strip().lower()
    device = eval_provenance.get("device")
    if device != required_device:
        raise SystemExit(
            f"FATAL: contest_auth_eval provenance.device={device!r}, "
            f"expected {required_device!r}"
        )
    score_axis = payload.get("score_axis")
    expected_score_axis = _expected_score_axis(str(device))
    if score_axis != expected_score_axis:
        raise SystemExit(
            f"FATAL: contest_auth_eval score_axis={score_axis!r}, "
            f"expected {expected_score_axis!r} for device={device!r}"
        )
    payload_archive_sha256 = eval_provenance.get("archive_sha256") or payload.get("archive_sha256")
    if payload_archive_sha256 != actual_archive_sha256:
        raise SystemExit(
            "FATAL: contest_auth_eval archive_sha256 does not match evaluated archive: "
            f"json={payload_archive_sha256!r} actual={actual_archive_sha256}"
        )

    provenance = read_json(provenance_path) if provenance_path.exists() else {}

    components, component_gates, component_gate_violations = _check_component_gates(payload, args)
    distillation_gate = _check_distillation_promotion_gate(
        provenance=provenance,
        device=device,
        archive_sha256=actual_archive_sha256,
        archive_bytes=actual_archive_bytes,
        component_gates=component_gates,
    )
    distillation_gate_violations = distillation_gate["violations"]
    distillation_gate_triggered = bool(distillation_gate_violations)
    source_manifest_closure = _check_source_manifest_runtime_closure(
        eval_provenance=eval_provenance,
        artifact_dir=provenance_path.parent,
    )
    source_manifest_closure_gate_triggered = bool(
        source_manifest_closure.get("violations")
    )
    raw_promotion_policy_gate = _check_raw_promotion_policy_gate(payload)
    raw_promotion_policy_gate_triggered = bool(
        raw_promotion_policy_gate["triggered"]
    )

    result_copy = Path(args.result_copy) if args.result_copy else contest_json

    predicted_low, predicted_high = args.predicted_band
    regression_threshold, regression_threshold_mode = _regression_threshold(args)
    score_delta_vs_baseline = score_recomputed - args.baseline_score
    if regression_threshold_mode == "absolute_score":
        regression_triggered = score_recomputed > regression_threshold
    else:
        regression_triggered = score_delta_vs_baseline > regression_threshold
    lane_status = _status(
        score_recomputed,
        predicted_low,
        predicted_high,
        regression_triggered,
    )
    component_gate_triggered = bool(component_gate_violations)
    if component_gate_triggered:
        lane_status = (
            "REGRESSION_AND_COMPONENT_GATE_REVIEW_REQUIRED"
            if lane_status == "REGRESSION_REVIEW_REQUIRED"
            else "COMPONENT_GATE_REVIEW_REQUIRED"
        )
    if sane_score_gate_triggered:
        lane_status = _add_review_status(lane_status, "SANE_SCORE")
    if distillation_gate_triggered:
        if lane_status == "REGRESSION_AND_COMPONENT_GATE_REVIEW_REQUIRED":
            lane_status = "REGRESSION_COMPONENT_AND_DISTILLATION_POLICY_REVIEW_REQUIRED"
        elif lane_status == "REGRESSION_REVIEW_REQUIRED":
            lane_status = "REGRESSION_AND_DISTILLATION_POLICY_REVIEW_REQUIRED"
        elif lane_status == "COMPONENT_GATE_REVIEW_REQUIRED":
            lane_status = "COMPONENT_AND_DISTILLATION_POLICY_REVIEW_REQUIRED"
        else:
            lane_status = "DISTILLATION_POLICY_REVIEW_REQUIRED"
    if source_manifest_closure_gate_triggered:
        lane_status = _add_review_status(lane_status, "SOURCE_MANIFEST_CLOSURE")
    if raw_promotion_policy_gate_triggered:
        lane_status = _add_review_status(lane_status, "RAW_PROMOTION_POLICY")
    scientific_score_eligible = (
        not regression_triggered
        and not component_gate_triggered
        and not sane_score_gate_triggered
        and not distillation_gate_triggered
        and not source_manifest_closure_gate_triggered
        and not raw_promotion_policy_gate_triggered
    )
    gpu_t4_match = eval_provenance.get("gpu_t4_match")
    is_cuda_axis = expected_score_axis == "contest_cuda"
    if is_cuda_axis:
        contest_equivalent_hardware = gpu_t4_match is True
        evidence_grade = (
            "A++ contest T4" if contest_equivalent_hardware else "A score-grade"
        )
        promotion_eligible = scientific_score_eligible and contest_equivalent_hardware
        score_claim_valid = promotion_eligible
        hardware_promotion_gate_triggered = scientific_score_eligible and not contest_equivalent_hardware
        paper_claim_grade = (
            evidence_grade
            if promotion_eligible
            else (
                "A score-grade; T4/equivalent promotion required"
                if scientific_score_eligible
                else "A-negative scoped forensic"
            )
        )
        if promotion_eligible:
            allowed_use = ["promotion_review", "rank_frontier_candidate"]
        elif scientific_score_eligible:
            allowed_use = ["diagnostic_score_screen", "requires_t4_confirmation", "no_promotion"]
        else:
            allowed_use = ["forensic", "no_rank_frontier", "no_promotion"]
    else:
        contest_equivalent_hardware = (
            payload.get("evidence_grade") == "contest-CPU"
            and payload.get("score_claim_valid") is True
            and payload.get("cpu_leaderboard_reproduction_eligible") is True
        )
        evidence_grade = "contest-CPU" if contest_equivalent_hardware else "CPU advisory"
        promotion_eligible = False
        score_claim_valid = scientific_score_eligible and contest_equivalent_hardware
        hardware_promotion_gate_triggered = False
        paper_claim_grade = (
            "contest-CPU public leaderboard reproduction"
            if score_claim_valid
            else (
                "contest-CPU replay requires Linux x86_64 full-sample custody"
                if scientific_score_eligible
                else "A-negative scoped CPU forensic"
            )
        )
        allowed_use = (
            [
                "cpu_axis_score_claim",
                "public_leaderboard_reproduction",
                "cpu_cuda_drift_diagnosis",
                "no_cuda_promotion",
            ]
            if score_claim_valid
            else ["forensic", "no_rank_frontier", "no_cuda_promotion"]
        )

    baseline_archive_bytes = args.baseline_archive_bytes
    archive_delta_bytes = None
    if baseline_archive_bytes is not None:
        archive_delta_bytes = actual_archive_bytes - baseline_archive_bytes
    axis_prefix = str(expected_score_axis)
    axis_score_tag = _score_tag_for_axis(axis_prefix)
    axis_provenance = {
        f"{axis_prefix}_score": score_recomputed,
        f"{axis_prefix}_score_recomputed": score_recomputed,
        f"{axis_prefix}_score_reported_rounded": final_score,
        f"{axis_prefix}_score_source": "contest_auth_eval.json:score_recomputed_from_components",
        f"{axis_prefix}_avg_posenet_dist": components["avg_posenet_dist"],
        f"{axis_prefix}_avg_segnet_dist": components["avg_segnet_dist"],
        f"{axis_prefix}_result_json": str(result_copy),
        f"{axis_prefix}_n_samples": n_samples,
        f"{axis_prefix}_archive_sha256": actual_archive_sha256,
        f"{axis_prefix}_archive_bytes": actual_archive_bytes,
        f"{axis_prefix}_device": device,
    }
    if is_cuda_axis:
        axis_provenance.update(
            {
                "contest_cuda_gpu_model": eval_provenance.get("gpu_model"),
                "contest_cuda_gpu_t4_match": gpu_t4_match,
            }
        )
    else:
        axis_provenance.update(
            {
                "contest_cpu_platform_system": eval_provenance.get("platform_system"),
                "contest_cpu_platform_machine": eval_provenance.get("platform_machine"),
                "contest_cpu_leaderboard_reproduction_eligible": payload.get(
                    "cpu_leaderboard_reproduction_eligible"
                ),
            }
        )

    provenance.update(
        {
            "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stacked_archive_bytes": actual_archive_bytes,
            "final_archive_bytes": actual_archive_bytes,
            "baseline_archive_bytes": baseline_archive_bytes,
            "archive_delta_bytes": archive_delta_bytes,
            **axis_provenance,
            "score_axis": axis_prefix,
            "contest_equivalent_hardware": contest_equivalent_hardware,
            "evidence_grade": evidence_grade,
            "promotion_eligible": promotion_eligible,
            "score_claim_valid": score_claim_valid,
            "scientific_score_eligible": scientific_score_eligible,
            "hardware_promotion_gate_triggered": hardware_promotion_gate_triggered,
            "paper_claim_grade": paper_claim_grade,
            "allowed_use": allowed_use,
            "score_tag": axis_score_tag,
            "result_tag": axis_score_tag,
            "score_delta_vs_baseline": score_delta_vs_baseline,
            args.delta_key: score_delta_vs_baseline,
            "regression_threshold": regression_threshold,
            "regression_threshold_mode": regression_threshold_mode,
            "regression_triggered": regression_triggered,
            "regression_scope": "measured_implementation_config_only_pending_review",
            "sane_score_gate_triggered": sane_score_gate_triggered,
            "sane_score_gate_violation": sane_score_gate_violation,
            "max_sane_score": args.max_sane_score,
            "component_gates": component_gates,
            "component_gate_violations": component_gate_violations,
            "component_gate_triggered": component_gate_triggered,
            "distillation_policy_active": distillation_gate["active"],
            "distillation_policy_gate_violations": distillation_gate_violations,
            "distillation_policy_gate_triggered": distillation_gate_triggered,
            "distillation_policy_sha256": distillation_gate["policy_sha256"],
            "distillation_policy_sha256_expected": distillation_gate["expected_policy_sha256"],
            "source_manifest_runtime_closure": source_manifest_closure,
            "source_manifest_runtime_closure_gate_triggered": source_manifest_closure_gate_triggered,
            "source_manifest_sha256": source_manifest_closure.get("source_manifest_sha256"),
            "source_manifest_file_sha256": source_manifest_closure.get("source_manifest_file_sha256"),
            "source_manifest_runtime_closure_sha256": source_manifest_closure.get("closure_sha256"),
            "raw_promotion_policy_gate": raw_promotion_policy_gate,
            "raw_promotion_policy_gate_triggered": raw_promotion_policy_gate_triggered,
            "raw_promotion_policy_gate_violations": raw_promotion_policy_gate["violations"],
            "lane_status": lane_status,
        }
    )
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = provenance_path.with_suffix(provenance_path.suffix + ".tmp")
    tmp_path.write_text(json_text(provenance))
    shutil.move(str(tmp_path), str(provenance_path))

    adjudication_summary = {
        "schema_version": 1,
        "completed_at_utc": provenance["completed_at_utc"],
        "lane_status": lane_status,
        "evidence_grade": evidence_grade,
        "paper_claim_grade": paper_claim_grade,
        "allowed_use": allowed_use,
        "promotion_eligible": promotion_eligible,
        "score_claim_valid": promotion_eligible,
        "scientific_score_eligible": scientific_score_eligible,
        "hardware_promotion_gate_triggered": hardware_promotion_gate_triggered,
        "contest_equivalent_hardware": contest_equivalent_hardware,
        "score_delta_vs_baseline": score_delta_vs_baseline,
        args.delta_key: score_delta_vs_baseline,
        "regression_threshold": regression_threshold,
        "regression_threshold_mode": regression_threshold_mode,
        "regression_triggered": regression_triggered,
        "regression_scope": "measured_implementation_config_only_pending_review",
        "sane_score_gate_triggered": sane_score_gate_triggered,
        "sane_score_gate_violation": sane_score_gate_violation,
        "component_gates": component_gates,
        "component_gate_violations": component_gate_violations,
        "component_gate_triggered": component_gate_triggered,
        "distillation_policy_active": distillation_gate["active"],
        "distillation_policy_gate_violations": distillation_gate_violations,
        "distillation_policy_gate_triggered": distillation_gate_triggered,
        "source_manifest_runtime_closure_gate_triggered": source_manifest_closure_gate_triggered,
        "source_manifest_sha256": source_manifest_closure.get("source_manifest_sha256"),
        "source_manifest_runtime_closure_sha256": source_manifest_closure.get("closure_sha256"),
        "raw_promotion_policy_gate_triggered": raw_promotion_policy_gate_triggered,
        "raw_promotion_policy_gate_violations": raw_promotion_policy_gate["violations"],
        "score_axis": axis_prefix,
        f"{axis_prefix}_score_source": "contest_auth_eval.json:score_recomputed_from_components",
        f"{axis_prefix}_archive_sha256": actual_archive_sha256,
        f"{axis_prefix}_archive_bytes": actual_archive_bytes,
        f"{axis_prefix}_device": device,
        f"{axis_prefix}_gpu_t4_match": gpu_t4_match if is_cuda_axis else None,
    }
    result_payload = dict(payload)
    result_payload.update(
        {
            "adjudication": adjudication_summary,
            "promotion_eligible": promotion_eligible,
            "score_claim_valid": score_claim_valid,
            "scientific_score_eligible": scientific_score_eligible,
            "hardware_promotion_gate_triggered": hardware_promotion_gate_triggered,
            "contest_equivalent_hardware": contest_equivalent_hardware,
            "paper_claim_grade": paper_claim_grade,
            "allowed_use": allowed_use,
            "allowed_uses": allowed_use,
            "lane_status": lane_status,
            "regression_triggered": regression_triggered,
            "component_gate_triggered": component_gate_triggered,
            "sane_score_gate_triggered": sane_score_gate_triggered,
            "distillation_policy_gate_triggered": distillation_gate_triggered,
            "source_manifest_runtime_closure_gate_triggered": source_manifest_closure_gate_triggered,
            "raw_promotion_policy_gate_triggered": raw_promotion_policy_gate_triggered,
            "raw_promotion_policy_gate_violations": raw_promotion_policy_gate["violations"],
            "score_delta_vs_baseline": score_delta_vs_baseline,
            args.delta_key: score_delta_vs_baseline,
            "adjudication_provenance": str(provenance_path),
        }
    )
    if promotion_eligible:
        result_payload["evidence_grade"] = "A++"
        result_payload["score_claim"] = True
        result_payload["rank_or_kill_eligible"] = True
    elif score_claim_valid:
        result_payload["evidence_grade"] = evidence_grade
        result_payload["score_claim"] = True
        result_payload["rank_or_kill_eligible"] = False
    else:
        result_payload["evidence_grade"] = paper_claim_grade
        result_payload["score_claim"] = False
        result_payload["rank_or_kill_eligible"] = False
    if args.result_copy:
        result_copy.parent.mkdir(parents=True, exist_ok=True)
        copy_tmp = result_copy.with_suffix(result_copy.suffix + ".tmp")
        copy_tmp.write_text(json_text(result_payload))
        shutil.move(str(copy_tmp), str(result_copy))

    return {
        "score_recomputed": score_recomputed,
        "score_reported_rounded": final_score,
        "lane_status": lane_status,
        "regression_triggered": regression_triggered,
        "regression_threshold": regression_threshold,
        "regression_threshold_mode": regression_threshold_mode,
        "sane_score_gate_triggered": sane_score_gate_triggered,
        "sane_score_gate_violation": sane_score_gate_violation,
        "avg_posenet_dist": components["avg_posenet_dist"],
        "avg_segnet_dist": components["avg_segnet_dist"],
        "component_gates": component_gates,
        "component_gate_violations": component_gate_violations,
        "component_gate_triggered": component_gate_triggered,
        "distillation_policy_active": distillation_gate["active"],
        "distillation_policy_gate_violations": distillation_gate_violations,
        "distillation_policy_gate_triggered": distillation_gate_triggered,
        "distillation_policy_sha256": distillation_gate["policy_sha256"],
        "distillation_policy_sha256_expected": distillation_gate["expected_policy_sha256"],
        "source_manifest_runtime_closure": source_manifest_closure,
        "source_manifest_runtime_closure_gate_triggered": source_manifest_closure_gate_triggered,
        "source_manifest_sha256": source_manifest_closure.get("source_manifest_sha256"),
        "source_manifest_runtime_closure_sha256": source_manifest_closure.get("closure_sha256"),
        "raw_promotion_policy_gate": raw_promotion_policy_gate,
        "raw_promotion_policy_gate_triggered": raw_promotion_policy_gate_triggered,
        "raw_promotion_policy_gate_violations": raw_promotion_policy_gate["violations"],
        "archive_sha256": actual_archive_sha256,
        "archive_bytes": actual_archive_bytes,
        "gpu_model": eval_provenance.get("gpu_model"),
        "gpu_t4_match": gpu_t4_match,
        "score_axis": axis_prefix,
        "contest_equivalent_hardware": contest_equivalent_hardware,
        "evidence_grade": evidence_grade,
        "promotion_eligible": promotion_eligible,
        "score_claim_valid": score_claim_valid,
        "scientific_score_eligible": scientific_score_eligible,
        "hardware_promotion_gate_triggered": hardware_promotion_gate_triggered,
        "paper_claim_grade": paper_claim_grade,
        "allowed_use": allowed_use,
        "result_json": str(result_copy),
        "provenance": str(provenance_path),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contest-json", required=True)
    p.add_argument("--provenance", required=True)
    p.add_argument("--archive", required=True)
    p.add_argument("--result-copy")
    p.add_argument("--baseline-score", type=float, required=True)
    p.add_argument("--baseline-archive-bytes", type=int)
    p.add_argument("--predicted-band", nargs=2, type=float, required=True, metavar=("LOW", "HIGH"))
    p.add_argument("--regression-threshold", type=float)
    p.add_argument(
        "--hard-kill-above",
        type=float,
        dest="hard_kill_above",
        help="Deprecated alias for --regression-threshold.",
    )
    p.add_argument("--delta-key", default="score_delta_vs_baseline")
    p.add_argument("--max-posenet-dist", type=float)
    p.add_argument("--max-segnet-dist", type=float)
    p.add_argument(
        "--baseline-posenet-dist",
        "--reference-posenet-dist",
        dest="baseline_posenet_dist",
        type=float,
        help="Reference PoseNet distortion for relative component gates.",
    )
    p.add_argument(
        "--baseline-segnet-dist",
        "--reference-segnet-dist",
        dest="baseline_segnet_dist",
        type=float,
        help="Reference SegNet distortion for relative component gates.",
    )
    p.add_argument(
        "--max-posenet-relative",
        type=float,
        help="Reject when avg_posenet_dist / reference_posenet_dist exceeds this ratio.",
    )
    p.add_argument(
        "--max-segnet-relative",
        type=float,
        help="Reject when avg_segnet_dist / reference_segnet_dist exceeds this ratio.",
    )
    p.add_argument("--component-reference-label", default="baseline")
    p.add_argument("--required-device", default="cuda")
    p.add_argument("--required-samples", type=int, default=600)
    p.add_argument("--max-sane-score", type=float, default=10.0)
    p.add_argument(
        "--allow-sane-score-forensic-success",
        action="store_true",
        help=(
            "Return 0 for finite exact-CUDA results outside --max-sane-score "
            "after writing custody artifacts. The result remains non-promotable "
            "via lane_status and sane_score_gate_triggered metadata."
        ),
    )
    p.add_argument(
        "--allow-component-gate-forensic-success",
        action="store_true",
        help=(
            "Return 0 for component-gate-only failures after writing custody "
            "artifacts. The result remains non-promotable via lane_status and "
            "component_gate_triggered metadata."
        ),
    )
    p.add_argument(
        "--allow-distillation-gate-forensic-success",
        action="store_true",
        help=(
            "Return 0 for KL/JBL/distillation policy-gate failures after "
            "writing custody artifacts. The result remains non-promotable via "
            "lane_status and distillation_policy_gate_triggered metadata."
        ),
    )
    args = p.parse_args()

    result = adjudicate(args)
    print(f"SCORE_RECOMPUTED={result['score_recomputed']:.15g}")
    print(f"SCORE_REPORTED_ROUNDED={result['score_reported_rounded']:.15g}")
    print(f"LANE_STATUS={result['lane_status']}")
    print(f"REGRESSION_TRIGGERED={int(result['regression_triggered'])}")
    print(f"REGRESSION_THRESHOLD={result['regression_threshold']:.15g}")
    print(f"SANE_SCORE_GATE_TRIGGERED={int(result['sane_score_gate_triggered'])}")
    print(f"COMPONENT_GATE_TRIGGERED={int(result['component_gate_triggered'])}")
    print(f"DISTILLATION_POLICY_ACTIVE={int(result['distillation_policy_active'])}")
    print(f"DISTILLATION_POLICY_GATE_TRIGGERED={int(result['distillation_policy_gate_triggered'])}")
    print(f"SOURCE_MANIFEST_CLOSURE_GATE_TRIGGERED={int(result['source_manifest_runtime_closure_gate_triggered'])}")
    print(f"RAW_PROMOTION_POLICY_GATE_TRIGGERED={int(result['raw_promotion_policy_gate_triggered'])}")
    print(f"ARCHIVE_SHA256={result['archive_sha256']}")
    print(f"ARCHIVE_BYTES={result['archive_bytes']}")
    print(f"GPU_T4_MATCH={json.dumps(result['gpu_t4_match'])}")
    print(f"EVIDENCE_GRADE={result['evidence_grade']}")
    print(f"PROMOTION_ELIGIBLE={int(result['promotion_eligible'])}")
    print(f"PAPER_CLAIM_GRADE={result['paper_claim_grade']}")
    print(f"RESULT_JSON={result['result_json']}")
    print(f"ADJUDICATION_JSON: {json.dumps(result, sort_keys=True)}")
    exit_code = 0
    if result["sane_score_gate_triggered"]:
        violation_json = json.dumps(result["sane_score_gate_violation"], sort_keys=True)
        if args.allow_sane_score_forensic_success:
            print(
                "SANE_SCORE_GATE_FORENSIC_SUCCESS=1 "
                "non_promotable exact-CUDA sane-score violation: "
                + violation_json
            )
        else:
            print("FATAL: exact-CUDA sane-score violation: " + violation_json)
            exit_code = 2
    if result["component_gate_triggered"]:
        violation_json = json.dumps(result["component_gate_violations"], sort_keys=True)
        if args.allow_component_gate_forensic_success:
            print(
                "COMPONENT_GATE_FORENSIC_SUCCESS=1 "
                "non_promotable component gate violation from contest_auth_eval.json: "
                + violation_json
            )
        else:
            print("FATAL: component gate violation from contest_auth_eval.json: " + violation_json)
            exit_code = 2
    if result["distillation_policy_gate_triggered"]:
        violation_json = json.dumps(result["distillation_policy_gate_violations"], sort_keys=True)
        if args.allow_distillation_gate_forensic_success:
            print(
                "DISTILLATION_POLICY_GATE_FORENSIC_SUCCESS=1 "
                "non_promotable KL/JBL/distillation gate violation: "
                + violation_json
            )
        else:
            print("FATAL: KL/JBL/distillation promotion gate violation: " + violation_json)
            exit_code = 2
    if result["source_manifest_runtime_closure_gate_triggered"]:
        violation_json = json.dumps(
            result["source_manifest_runtime_closure"].get("violations", []),
            sort_keys=True,
        )
        print("FATAL: source-manifest runtime byte/SHA closure violation: " + violation_json)
        exit_code = 2
    if result["raw_promotion_policy_gate_triggered"]:
        violation_json = json.dumps(
            result["raw_promotion_policy_gate_violations"],
            sort_keys=True,
        )
        print("FATAL: raw promotion/rank blockers still present: " + violation_json)
        exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
