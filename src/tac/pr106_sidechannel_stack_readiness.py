# SPDX-License-Identifier: MIT
"""Planning-only readiness for the PR106 sidechannel stack.

This module turns the local latent/yshift/LRL1/WR01 scaffolds into one
deterministic atom ledger. It does not dispatch, does not claim score, and does
not promote any row without exact CUDA evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.optimization.meta_lagrangian_allocator import build_atom_ledger
from tac.repo_io import read_json, repo_relative, sha256_file

SCHEMA = "pr106_sidechannel_stack_readiness.v1"
DEFAULT_BASE_POSE_DIST = 0.0


class PR106SidechannelStackReadinessError(ValueError):
    """Raised when a PR106 stack-readiness input is malformed."""


def build_pr106_sidechannel_stack_readiness_from_paths(
    *,
    repo_root: Path,
    baseline_json_path: Path,
    pr106_anchor_archive: Path,
    latent_metadata_path: Path | None = None,
    yshift_metadata_path: Path | None = None,
    lrl1_metadata_path: Path | None = None,
    three_sister_stacked_metadata_path: Path | None = None,
    wavelet_sidechannel_manifest_path: Path | None = None,
    wavelet_stacked_metadata_path: Path | None = None,
    wavelet_apply_gate_path: Path | None = None,
    wr01_exact_eval_packet_path: Path | None = None,
) -> dict[str, Any]:
    """Read local JSON artifacts and emit one fail-closed stack report."""

    repo_root = repo_root.resolve()
    baseline_path = _repo_path(repo_root, baseline_json_path)
    if not baseline_path.is_file():
        raise PR106SidechannelStackReadinessError(
            f"missing baseline JSON: {baseline_json_path}"
        )
    anchor_path = _repo_path(repo_root, pr106_anchor_archive)
    if not anchor_path.is_file():
        raise PR106SidechannelStackReadinessError(
            f"missing PR106 anchor archive: {pr106_anchor_archive}"
        )

    baseline = _read_mapping(baseline_path)
    anchor_identity = _file_identity(repo_root, anchor_path)
    artifact_specs = {
        "latent": latent_metadata_path,
        "yshift": yshift_metadata_path,
        "lrl1": lrl1_metadata_path,
        "three_sister_stack": three_sister_stacked_metadata_path,
        "wavelet_sidechannel": wavelet_sidechannel_manifest_path,
        "wavelet_noop_stack": wavelet_stacked_metadata_path,
        "wavelet_apply_gate": wavelet_apply_gate_path,
        "wr01_exact_eval_packet": wr01_exact_eval_packet_path,
    }
    artifacts = {
        name: _artifact_payload(repo_root, path)
        for name, path in artifact_specs.items()
    }
    return build_pr106_sidechannel_stack_readiness(
        repo_root=repo_root,
        baseline=baseline,
        baseline_path=baseline_path,
        pr106_anchor_identity=anchor_identity,
        artifacts=artifacts,
    )


def build_pr106_sidechannel_stack_readiness(
    *,
    repo_root: Path,
    baseline: Mapping[str, Any],
    baseline_path: Path | None,
    pr106_anchor_identity: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic planning-only stack readiness payload."""

    base_pose_dist, base_pose_blockers = _baseline_pose_dist(baseline)
    artifact_summaries = {
        name: _artifact_summary(name, artifact, repo_root, pr106_anchor_identity)
        for name, artifact in artifacts.items()
    }
    atoms = _stack_atoms(artifact_summaries)
    ledger = build_atom_ledger(
        atoms,
        base_pose_dist=base_pose_dist,
        source="pr106_sidechannel_stack_readiness",
    )
    unsafe_blockers = _unsafe_artifact_blockers(artifact_summaries)
    stack_blockers = _stack_blockers(artifact_summaries)
    blockers = _unique(
        [
            *base_pose_blockers,
            *unsafe_blockers,
            *stack_blockers,
            *ledger.get("dispatch_blockers", []),
        ]
    )
    ready_for_local_stack_planning = bool(atoms) and not unsafe_blockers
    return {
        "schema": SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "remote_gpu_run": False,
        "ready_for_local_stack_planning": ready_for_local_stack_planning,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
        "blocker_count": len(blockers),
        "no_score_claim_policy": (
            "all rows are planning-only until a candidate-specific exact CUDA "
            "auth-eval artifact exists"
        ),
        "baseline": {
            "path": _display(repo_root, baseline_path) if baseline_path else None,
            "archive_size_bytes": _safe_int(baseline.get("archive_size_bytes")),
            "score_recomputed_from_components": _safe_float(
                baseline.get("score_recomputed_from_components")
            ),
            "avg_posenet_dist": _safe_float(baseline.get("avg_posenet_dist")),
            "avg_segnet_dist": _safe_float(baseline.get("avg_segnet_dist")),
        },
        "pr106_anchor_archive": dict(pr106_anchor_identity),
        "artifact_summaries": artifact_summaries,
        "candidate_exact_cuda_artifacts": {
            name: summary["exact_cuda_artifact"]
            for name, summary in artifact_summaries.items()
            if summary["kind"] in {"sidechannel", "stack", "packet", "gate"}
        },
        "stack_sequence": [
            "latent",
            "yshift",
            "lrl1",
            "wr01_apply_transform",
            "three_sister_stack",
            "wavelet_noop_stack",
        ],
        "meta_lagrangian_atom_ledger": ledger,
    }


def _artifact_payload(repo_root: Path, path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "payload": {}}
    full = _repo_path(repo_root, path)
    if not full.is_file():
        return {"path": _display(repo_root, full), "exists": False, "payload": {}}
    payload = _read_mapping(full)
    return {
        "path": _display(repo_root, full),
        "exists": True,
        "sha256": sha256_file(full),
        "bytes": full.stat().st_size,
        "payload": payload,
    }


def _artifact_summary(
    name: str,
    artifact: Mapping[str, Any],
    repo_root: Path,
    pr106_anchor_identity: Mapping[str, Any],
) -> dict[str, Any]:
    payload = artifact.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    kind = _artifact_kind(name)
    archive_identity = _candidate_archive_identity(
        repo_root,
        payload,
        pr106_anchor_identity=pr106_anchor_identity,
        artifact_name=name,
    )
    byte_delta = _artifact_byte_delta(
        name=name,
        payload=payload,
        archive_identity=archive_identity,
        pr106_anchor_identity=pr106_anchor_identity,
    )
    exact_cuda_artifact = _exact_cuda_artifact(repo_root, payload)
    dispatch_blockers = _unique(
        [
            *_string_list(payload.get("dispatch_blockers")),
            *_string_list(payload.get("blockers")),
        ]
    )
    if exact_cuda_artifact is None and kind in {"sidechannel", "stack", "packet", "gate"}:
        dispatch_blockers.append(f"{name}_requires_exact_cuda_auth_eval")
    return {
        "name": name,
        "kind": kind,
        "artifact_path": artifact.get("path"),
        "artifact_exists": artifact.get("exists") is True,
        "artifact_sha256": artifact.get("sha256"),
        "score_claim": payload.get("score_claim"),
        "dispatch_attempted": payload.get("dispatch_attempted"),
        "remote_gpu_run": payload.get("remote_gpu_run"),
        "ready_for_archive_preflight": payload.get("ready_for_archive_preflight"),
        "ready_for_exact_eval_dispatch": payload.get("ready_for_exact_eval_dispatch"),
        "static_packet_ready": payload.get("static_packet_ready"),
        "candidate_static_preflight_ready": payload.get("candidate_static_preflight_ready"),
        "static_custody": _static_custody_summary(payload),
        "archive": archive_identity,
        "byte_delta_vs_stack_source": byte_delta,
        "evidence_grade": _evidence_grade(name, payload),
        "exact_cuda_artifact": exact_cuda_artifact,
        "dispatch_blockers": _unique(dispatch_blockers),
    }


def _stack_atoms(artifact_summaries: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    atoms = []
    for name in (
        "latent",
        "yshift",
        "lrl1",
        "three_sister_stack",
        "wavelet_noop_stack",
        "wr01_exact_eval_packet",
    ):
        summary = artifact_summaries.get(name)
        if not summary or summary.get("artifact_exists") is not True:
            continue
        byte_delta = summary.get("byte_delta_vs_stack_source")
        if not isinstance(byte_delta, int) or isinstance(byte_delta, bool):
            byte_delta = 0
        atoms.append(_atom_from_summary(name, summary, byte_delta=byte_delta))
    return atoms


def _atom_from_summary(
    name: str,
    summary: Mapping[str, Any],
    *,
    byte_delta: int,
) -> dict[str, Any]:
    archive = summary.get("archive")
    archive = archive if isinstance(archive, Mapping) else {}
    archive_manifest = archive.get("archive_manifest")
    archive_manifest = archive_manifest if isinstance(archive_manifest, Mapping) else {}
    exact_cuda_artifact = summary.get("exact_cuda_artifact")
    packet_ready = (
        name == "wr01_exact_eval_packet"
        and summary.get("static_packet_ready") is True
        and archive_manifest.get("exists") is True
    )
    return {
        "atom_id": f"pr106_sidechannel_stack:{name}",
        "family": _atom_family(name),
        "family_group": "pr106_sidechannel_stack",
        "pareto_scope": "pr106_sidechannel_stack",
        "byte_delta": byte_delta,
        "expected_seg_dist_delta": 0.0,
        "expected_pose_dist_delta": 0.0,
        "confidence": 1.0 if packet_ready else 0.0,
        "evidence_grade": summary.get("evidence_grade") or "planning_local_scaffold",
        "proxy_row": not packet_ready,
        "rankable": packet_ready,
        "raw_equal": True if packet_ready else None,
        "score_claim": False,
        "dispatchable": False,
        "ready_for_exact_eval_dispatch": False,
        "interaction_assumptions": _interaction_assumptions(name),
        "conflicts_with_families": [],
        "conflicts_with_atoms": [],
        "evidence_source_path": summary.get("artifact_path") or "",
        "evidence_source_sha256": summary.get("artifact_sha256") or "",
        "source_archive_sha256": archive.get("source_sha256") or "",
        "archive_manifest_path": archive_manifest.get("allocator_path") or "",
        "archive_manifest_sha256": archive_manifest.get("sha256") or "",
        "research_basis_ids": ["PR106-sidechannel-stack-local-readiness"],
        "kkt_proof": {
            "status": "blocked",
            "blockers": [
                "candidate_specific_exact_cuda_artifact_missing"
                if exact_cuda_artifact is None
                else "candidate_specific_exact_cuda_artifact_not_promoted",
            ],
        },
    }


def _candidate_archive_identity(
    repo_root: Path,
    payload: Mapping[str, Any],
    *,
    pr106_anchor_identity: Mapping[str, Any],
    artifact_name: str,
) -> dict[str, Any]:
    archive_path = _first_string(
        payload.get("archive_path"),
        payload.get("candidate_archive_path"),
    )
    if artifact_name == "wr01_exact_eval_packet":
        archive_path = _first_string(
            payload.get("archive_identity", {}).get("path")
            if isinstance(payload.get("archive_identity"), Mapping)
            else None,
            payload.get("archive_path"),
        )
    identity = {
        "path": archive_path,
        "exists": False,
        "sha256": _first_string(payload.get("archive_sha256"), payload.get("candidate_archive_sha256")),
        "bytes": _first_int(
            payload.get("archive_bytes"),
            payload.get("archive_zip_bytes"),
            payload.get("candidate_archive_bytes"),
        ),
        "source_sha256": _first_string(
            payload.get("source_archive_sha256"),
            pr106_anchor_identity.get("sha256"),
        ),
        "source_bytes": _first_int(
            payload.get("source_archive_bytes"),
            pr106_anchor_identity.get("bytes"),
        ),
        "archive_manifest": {},
    }
    if archive_path:
        full = _repo_path(repo_root, Path(archive_path))
        identity["exists"] = full.is_file()
        if full.is_file():
            identity["path"] = _display(repo_root, full)
            identity["sha256"] = sha256_file(full)
            identity["bytes"] = full.stat().st_size
    identity["archive_manifest"] = _archive_manifest_identity(repo_root, payload, artifact_name)
    return identity


def _archive_manifest_identity(
    repo_root: Path,
    payload: Mapping[str, Any],
    artifact_name: str,
) -> dict[str, Any]:
    manifest_path = ""
    if artifact_name == "wr01_exact_eval_packet":
        release_surface = payload.get("release_surface")
        if isinstance(release_surface, Mapping):
            files = release_surface.get("files")
            if isinstance(files, Mapping):
                archive_manifest = files.get("archive_manifest.json")
                if isinstance(archive_manifest, Mapping):
                    manifest_path = str(archive_manifest.get("path") or "")
    if not manifest_path:
        manifest_path = str(payload.get("archive_manifest_path") or "")
    if not manifest_path:
        return {"path": "", "allocator_path": "", "exists": False, "sha256": ""}
    full = _repo_path(repo_root, Path(manifest_path))
    exists = full.is_file()
    display = _display(repo_root, full)
    return {
        "path": display,
        "allocator_path": display if repo_root == Path.cwd().resolve() else full.as_posix(),
        "exists": exists,
        "sha256": sha256_file(full) if exists else "",
        "bytes": full.stat().st_size if exists else None,
    }


def _artifact_byte_delta(
    *,
    name: str,
    payload: Mapping[str, Any],
    archive_identity: Mapping[str, Any],
    pr106_anchor_identity: Mapping[str, Any],
) -> int | None:
    direct = _first_int(
        payload.get("byte_delta"),
        payload.get("archive_byte_delta"),
        payload.get("candidate_archive_byte_delta"),
        payload.get("candidate_archive_byte_delta_vs_source_estimate"),
        payload.get("delta_bytes_vs_pr106_zip"),
    )
    if direct is not None:
        return direct
    if name == "wr01_exact_eval_packet":
        packet_delta = _first_int(payload.get("byte_delta"))
        if packet_delta is not None:
            return packet_delta
    archive_bytes = _safe_int(archive_identity.get("bytes"))
    source_bytes = _safe_int(archive_identity.get("source_bytes"))
    if archive_bytes is not None and source_bytes is not None:
        return archive_bytes - source_bytes
    anchor_bytes = _safe_int(pr106_anchor_identity.get("bytes"))
    if archive_bytes is not None and anchor_bytes is not None:
        return archive_bytes - anchor_bytes
    return None


def _stack_blockers(artifact_summaries: Mapping[str, Mapping[str, Any]]) -> list[str]:
    blockers = [
        "stack_readiness_is_local_planning_only",
        "requires_exact_cuda_auth_eval_before_score_claim",
        "requires_no_dispatch_lane_claim_before_remote_submit",
    ]
    required = ("latent", "yshift", "lrl1", "wr01_exact_eval_packet")
    for name in required:
        summary = artifact_summaries.get(name)
        if not summary or summary.get("artifact_exists") is not True:
            blockers.append(f"{name}_artifact_missing")
            continue
        if summary.get("exact_cuda_artifact") is None:
            blockers.append(f"{name}_exact_cuda_artifact_missing")
    wr01 = artifact_summaries.get("wr01_exact_eval_packet", {})
    if wr01.get("static_packet_ready") is not True:
        blockers.append("wr01_static_packet_not_ready")
    static_custody = wr01.get("static_custody")
    if isinstance(static_custody, Mapping) and static_custody.get("ready") is False:
        blockers.append("wr01_static_custody_not_ready")
    wavelet_gate = artifact_summaries.get("wavelet_apply_gate", {})
    for blocker in wavelet_gate.get("dispatch_blockers", []):
        blockers.append(f"wavelet_gate:{blocker}")
    return _unique(blockers)


def _unsafe_artifact_blockers(artifact_summaries: Mapping[str, Mapping[str, Any]]) -> list[str]:
    blockers = []
    for name, summary in artifact_summaries.items():
        if summary.get("score_claim") is True:
            blockers.append(f"{name}_score_claim_true")
        if summary.get("dispatch_attempted") is True:
            blockers.append(f"{name}_dispatch_attempted_true")
        if summary.get("remote_gpu_run") is True:
            blockers.append(f"{name}_remote_gpu_run_true")
    return blockers


def _exact_cuda_artifact(repo_root: Path, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _first_string(
        payload.get("contest_auth_eval_json"),
        payload.get("exact_cuda_auth_eval_json"),
        payload.get("exact_cuda_artifact_path"),
    )
    if not path:
        return None
    full = _repo_path(repo_root, Path(path))
    if not full.is_file():
        return None
    return {
        "path": _display(repo_root, full),
        "sha256": sha256_file(full),
        "bytes": full.stat().st_size,
    }


def _static_custody_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    consistency = payload.get("release_surface_manifest_consistency")
    if isinstance(consistency, Mapping):
        return {
            "schema": consistency.get("schema"),
            "path": consistency.get("path"),
            "exists": consistency.get("exists"),
            "sha256": consistency.get("sha256"),
            "bytes": consistency.get("bytes"),
            "ready": consistency.get("ready"),
            "blockers": _string_list(consistency.get("blockers")),
        }
    return {
        "schema": "wr01_static_custody_summary_v1",
        "path": None,
        "exists": False,
        "ready": None,
        "blockers": [],
    }


def _baseline_pose_dist(payload: Mapping[str, Any]) -> tuple[float, list[str]]:
    value = _safe_float(
        payload.get("avg_posenet_dist"),
        payload.get("pose_dist"),
        payload.get("poseDist"),
    )
    if value is None:
        return DEFAULT_BASE_POSE_DIST, ["baseline_pose_dist_missing"]
    if value < 0.0:
        raise PR106SidechannelStackReadinessError("baseline pose distance is negative")
    return value, []


def _file_identity(repo_root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": _display(repo_root, path),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "bytes": path.stat().st_size if path.is_file() else None,
    }


def _read_mapping(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise PR106SidechannelStackReadinessError(f"{path} is not a JSON object")
    return payload


def _artifact_kind(name: str) -> str:
    if name in {"latent", "yshift", "lrl1", "wavelet_sidechannel"}:
        return "sidechannel"
    if name in {"three_sister_stack", "wavelet_noop_stack"}:
        return "stack"
    if name == "wr01_exact_eval_packet":
        return "packet"
    if name == "wavelet_apply_gate":
        return "gate"
    return "artifact"


def _atom_family(name: str) -> str:
    return {
        "latent": "pr106_latent_sidecar",
        "yshift": "pr106_yshift_sidechannel",
        "lrl1": "pr106_lrl1_sidechannel",
        "three_sister_stack": "pr106_three_sister_stack",
        "wavelet_noop_stack": "pr106_wavelet_noop_stack",
        "wr01_exact_eval_packet": "hnerv_wavelet_wr01_apply_transform",
    }.get(name, "pr106_sidechannel_stack")


def _evidence_grade(name: str, payload: Mapping[str, Any]) -> str:
    grade = _first_string(payload.get("evidence_grade"))
    if grade:
        return grade
    if name == "wr01_exact_eval_packet":
        return "empirical_archive_candidate_until_exact_cuda"
    if name in {"three_sister_stack", "wavelet_noop_stack"}:
        return "empirical_stack_scaffold_planning"
    return "empirical_local_scaffold_planning"


def _interaction_assumptions(name: str) -> list[str]:
    if name == "wr01_exact_eval_packet":
        return [
            "rate-only WR01 apply packet is stackable only after exact CUDA component response",
            "archive/runtime custody is necessary but insufficient for score promotion",
        ]
    return [
        "sidechannel atoms compose only after sister lanes have candidate-specific exact CUDA evidence",
        "local CPU smoke byte deltas are planning inputs, not score evidence",
    ]


def _repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _display(repo_root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    return repo_relative(path, repo_root)


def _safe_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _first_int(*values: Any) -> int | None:
    return _safe_int(*values)


def _safe_float(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return float(value)
    return None


def _first_string(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return ""


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _unique(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


__all__ = [
    "SCHEMA",
    "PR106SidechannelStackReadinessError",
    "build_pr106_sidechannel_stack_readiness",
    "build_pr106_sidechannel_stack_readiness_from_paths",
]
