"""Readiness custody for charged hyperbolic foveation parameter payloads."""

from __future__ import annotations

import ast
import zipfile
from pathlib import Path
from typing import Any

import torch

from tac.hyperbolic_foveation import EPSILON, load_foveation_params
from tac.repo_io import repo_relative, sha256_bytes, sha256_file

SCHEMA_VERSION = 1
FOVEATION_MEMBER = "foveation_params.bin"


def _stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().cpu().float()
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
    }


def audit_foveation_params(
    path: str | Path,
    *,
    repo_root: str | Path,
    expected_frames: int | None = None,
    expected_image_size: tuple[int, int] | None = None,
    source_archive_sha256: str | None = None,
    candidate_archive: str | Path | None = None,
    runtime_consumer: str | Path | None = None,
) -> dict[str, Any]:
    """Audit a charged foveation parameter payload without scoring it."""

    payload_path = Path(path)
    root = Path(repo_root)
    raw = payload_path.read_bytes()
    blockers = [
        "foveation_charged_member_not_proven",
        "foveation_runtime_consumer_not_proven",
        "exact_cuda_auth_eval_required_before_score_claim",
    ]
    warnings: list[str] = []
    wire_format = "HFV1" if raw[:4] == b"HFV1" else "legacy_raw_float32"
    if wire_format != "HFV1":
        blockers.append("foveation_params_not_hfv1_headered")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "hyperbolic_foveation_readiness",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical_payload_custody",
        "member": FOVEATION_MEMBER,
        "path": repo_relative(payload_path, root),
        "bytes": len(raw),
        "sha256": sha256_file(payload_path),
        "wire_format": wire_format,
        "source_archive_sha256": source_archive_sha256 or "",
        "runtime_contract": {
            "charged_member_required": FOVEATION_MEMBER,
            "scorer_loads_at_audit_time": False,
            "runtime_consumer_required": True,
            "geometry_preflight_required": True,
            "exact_cuda_auth_eval_required": True,
        },
        "archive_member": _audit_archive_member(
            candidate_archive,
            expected_raw=raw,
            repo_root=root,
            blockers=blockers,
        ),
        "runtime_consumer": _audit_runtime_consumer(
            runtime_consumer,
            repo_root=root,
            blockers=blockers,
        ),
    }

    try:
        module = load_foveation_params(payload_path, image_size=expected_image_size)
    except Exception as exc:
        manifest.update(
            {
                "ok": False,
                "load_error": f"{type(exc).__name__}: {exc}",
                "dispatch_blockers": [*blockers, "foveation_payload_load_failed"],
                "warnings": warnings,
            }
        )
        return manifest

    alpha = module.alpha.detach().cpu().float()
    radius = module.R.detach().cpu().float()
    power = module.p.detach().cpu().float()
    origin = module.o.detach().cpu().float()
    finite = (
        torch.isfinite(alpha).all()
        and torch.isfinite(radius).all()
        and torch.isfinite(power).all()
        and torch.isfinite(origin).all()
    )
    if not bool(finite):
        blockers.append("foveation_params_nonfinite")
    if bool(torch.any(radius <= EPSILON)):
        blockers.append("foveation_radius_nonpositive")
    if bool(torch.any(power < 0)):
        blockers.append("foveation_power_negative")

    h, w = module.image_size
    x_ok = (origin[:, 0] >= 0.0) & (origin[:, 0] <= float(max(w - 1, 0)))
    y_ok = (origin[:, 1] >= 0.0) & (origin[:, 1] <= float(max(h - 1, 0)))
    if not bool((x_ok & y_ok).all()):
        blockers.append("foveation_origin_outside_image")

    if expected_frames is not None and module.n_frames != int(expected_frames):
        blockers.append("foveation_frame_count_mismatch")
    if expected_image_size is not None and module.image_size != tuple(expected_image_size):
        blockers.append("foveation_image_size_mismatch")

    manifest.update(
        {
            "ok": not any(
                blocker
                not in {
                    "foveation_charged_member_not_proven",
                    "foveation_runtime_consumer_not_proven",
                    "exact_cuda_auth_eval_required_before_score_claim",
                }
                for blocker in blockers
            ),
            "n_frames": int(module.n_frames),
            "image_size": {"height": int(h), "width": int(w)},
            "geometry": {
                "alpha": _stats(alpha),
                "radius": _stats(radius),
                "power": _stats(power),
                "origin_x": _stats(origin[:, 0]),
                "origin_y": _stats(origin[:, 1]),
                "identity_like": bool(
                    torch.all(alpha.abs() <= EPSILON)
                    and torch.all(radius > EPSILON)
                    and torch.all(power >= 0)
                ),
                "all_finite": bool(finite),
                "all_origins_inside_image": bool((x_ok & y_ok).all()),
            },
            "dispatch_blockers": blockers,
            "warnings": warnings,
        }
    )
    return manifest


def _audit_archive_member(
    candidate_archive: str | Path | None,
    *,
    expected_raw: bytes,
    repo_root: Path,
    blockers: list[str],
) -> dict[str, Any]:
    if candidate_archive is None:
        return {
            "candidate_archive": "",
            "present": False,
            "bytes_match": False,
            "sha256_match": False,
        }
    archive_path = Path(candidate_archive)
    report: dict[str, Any] = {
        "candidate_archive": repo_relative(archive_path, repo_root),
        "present": False,
        "bytes_match": False,
        "sha256_match": False,
        "zip_read_error": "",
    }
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            duplicates = sorted({name for name in names if names.count(name) > 1})
            report["duplicate_member_names"] = duplicates
            if duplicates:
                blockers.append("foveation_candidate_archive_duplicate_members")
            if FOVEATION_MEMBER not in names:
                blockers.append("foveation_member_missing_from_candidate_archive")
                return report
            raw = archive.read(FOVEATION_MEMBER)
    except Exception as exc:
        report["zip_read_error"] = f"{type(exc).__name__}: {exc}"
        blockers.append("foveation_candidate_archive_not_readable")
        return report
    report.update(
        {
            "present": True,
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
            "bytes_match": len(raw) == len(expected_raw),
            "sha256_match": sha256_bytes(raw) == sha256_bytes(expected_raw),
        }
    )
    if not report["bytes_match"]:
        blockers.append("foveation_member_bytes_mismatch")
    if not report["sha256_match"]:
        blockers.append("foveation_member_sha256_mismatch")
    if report["bytes_match"] and report["sha256_match"]:
        while "foveation_charged_member_not_proven" in blockers:
            blockers.remove("foveation_charged_member_not_proven")
    return report


def _audit_runtime_consumer(
    runtime_consumer: str | Path | None,
    *,
    repo_root: Path,
    blockers: list[str],
) -> dict[str, Any]:
    if runtime_consumer is None:
        return {
            "path": "",
            "exists": False,
            "references_charged_member": False,
            "references_loader": False,
        }
    path = Path(runtime_consumer)
    report: dict[str, Any] = {
        "path": repo_relative(path, repo_root),
        "exists": path.is_file(),
        "references_charged_member": False,
        "references_loader": False,
        "read_error": "",
    }
    if not path.is_file():
        blockers.append("foveation_runtime_consumer_path_missing")
        return report
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        report["read_error"] = f"{type(exc).__name__}: {exc}"
        blockers.append("foveation_runtime_consumer_not_text")
        return report
    proof = _runtime_consumer_ast_proof(text)
    report.update(proof)
    report["references_charged_member"] = bool(proof["references_charged_member"])
    report["references_loader"] = bool(proof["references_loader"])
    if not report["references_charged_member"]:
        blockers.append("foveation_runtime_consumer_missing_member_reference")
    if not report["references_loader"]:
        blockers.append("foveation_runtime_consumer_missing_loader_reference")
    if report["references_charged_member"] and report["references_loader"]:
        while "foveation_runtime_consumer_not_proven" in blockers:
            blockers.remove("foveation_runtime_consumer_not_proven")
    return report


def _runtime_consumer_ast_proof(text: str) -> dict[str, Any]:
    """Return structural proof that runtime code references charged HFV1 bytes."""

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {
            "ast_parse_error": f"{type(exc).__name__}: {exc}",
            "references_charged_member": False,
            "references_loader": False,
            "load_foveation_params_call_count": 0,
        }

    assigned_strings: dict[str, set[str]] = {}
    direct_strings: set[str] = set()
    call_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            direct_strings.add(node.value)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned_strings.setdefault(target.id, set()).add(node.value.value)
        # Catalog #168 fix 2026-05-12: also handle annotated string constants
        # like `WAIVER_TOKEN: str = "..."`.
        elif (isinstance(node, ast.AnnAssign)
              and node.value is not None
              and isinstance(node.value, ast.Constant)
              and isinstance(node.value.value, str)
              and isinstance(node.target, ast.Name)):
            assigned_strings.setdefault(node.target.id, set()).add(node.value.value)
        if isinstance(node, ast.Call):
            func = node.func
            func_name = ""
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr
            if func_name == "load_foveation_params":
                call_count += 1
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        direct_strings.add(arg.value)
                    elif isinstance(arg, ast.Name):
                        direct_strings.update(assigned_strings.get(arg.id, set()))
                for keyword in node.keywords:
                    value = keyword.value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        direct_strings.add(value.value)
                    elif isinstance(value, ast.Name):
                        direct_strings.update(assigned_strings.get(value.id, set()))

    return {
        "ast_parse_error": "",
        "references_charged_member": FOVEATION_MEMBER in direct_strings,
        "references_loader": call_count > 0,
        "load_foveation_params_call_count": call_count,
    }


__all__ = ["FOVEATION_MEMBER", "SCHEMA_VERSION", "audit_foveation_params"]
