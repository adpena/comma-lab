"""Readiness custody for charged hyperbolic foveation parameter payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tac.hyperbolic_foveation import EPSILON, load_foveation_params
from tac.repo_io import repo_relative, sha256_file

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
) -> dict[str, Any]:
    """Audit a charged foveation parameter payload without scoring it."""

    payload_path = Path(path)
    root = Path(repo_root)
    raw = payload_path.read_bytes()
    blockers = [
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
            "ok": len(blockers) == 2,
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


__all__ = ["FOVEATION_MEMBER", "SCHEMA_VERSION", "audit_foveation_params"]
