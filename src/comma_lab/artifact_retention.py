# SPDX-License-Identifier: MIT
"""Retention planning for bulky rebuildable experiment artifacts.

The goal is to preserve signal, not raw scratch by default. Large raw inflation
outputs and tensor caches are safe to compact only when a small, durable
certificate proves how to rebuild or audit them later.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "comma_lab.artifact_retention_plan.v1"
EXECUTION_SCHEMA = "comma_lab.artifact_retention_execution.v1"
EXECUTION_JOURNAL_SCHEMA = "comma_lab.artifact_retention_execution_journal.v1"
UNKNOWN_RAW_KIND = "blocked_unknown_raw_surface"
INVERSE_SCORER_INFLATE_PARITY_RAW_KIND = "inverse_scorer_inflate_parity_raw_output"
INVERSE_SCORER_INFLATE_PARITY_PROBE_SCHEMA = "inverse_scorer_cell_inflate_parity_probe_v1"
INVERSE_SCORER_INFLATE_PARITY_SCOPE = "full_frame_inflate_output_tree"
INVERSE_SCORER_PARITY_BLOCKER = "candidate_inflate_output_parity_missing"
INVERSE_SCORER_PARITY_PROOF_NAMES = frozenset(
    {
        "inflate_parity_probe.json",
        "inverse_scorer_cell_inflate_parity_probe.json",
    }
)
KNOWN_RAW_WORKDIR_NAMES = frozenset(
    {
        "auth_eval_work",
        "eval_work",
        "contest_auth_eval_workdir",
        "contest_auth_eval_cpu_workdir",
        "contest_auth_eval_cuda_workdir",
        "local_macos_cpu_eval_work",
    }
)

DEFAULT_RETENTION_KINDS = frozenset(
    {
        "locality_inflated_raw",
        "local_cpu_advisory_inflated_raw",
        "local_cpu_advisory_extracted_scratch",
        INVERSE_SCORER_INFLATE_PARITY_RAW_KIND,
    }
)
KNOWN_MLX_SCORER_INPUT_CACHE_DIR_NAMES = frozenset(
    {
        "mlx_delta_cache",
        "mlx_scorer_input_cache",
        "reference_mlx_scorer_input_cache",
    }
)
AUTHORITY_FALSE_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "promotable",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
)
INVERSE_SCORER_AUTHORITY_FALSE_FIELDS = (
    *AUTHORITY_FALSE_FIELDS,
    "dispatch_attempted",
    "score_claim_eligible",
    "field_selection_ready_for_exact_eval_dispatch",
    "exact_cuda_auth_eval",
    "contest_cuda_auth_eval",
    "score_affecting_payload_changed",
    "charged_bits_changed",
)


class ArtifactRetentionError(ValueError):
    """Raised when a retention operation would risk signal loss."""


@dataclass(frozen=True)
class RetentionCandidate:
    path: str
    kind: str
    bytes: int
    certified_rebuildable: bool
    certificate: dict[str, Any]
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetentionPlan:
    schema: str
    generated_at_utc: str
    repo_root: str
    roots: list[str]
    include_kinds: list[str]
    min_bytes: int
    exclude_paths: list[str]
    candidates: list[RetentionCandidate]
    blocked_candidates: list[RetentionCandidate]

    @property
    def total_reclaimable_bytes(self) -> int:
        return sum(candidate.bytes for candidate in self.candidates)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["total_reclaimable_bytes"] = self.total_reclaimable_bytes
        payload["candidate_count"] = len(self.candidates)
        payload["blocked_candidate_count"] = len(self.blocked_candidates)
        payload["score_claim"] = False
        payload["promotion_eligible"] = False
        payload["ready_for_exact_eval_dispatch"] = False
        return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digest(path: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for file_path in sorted(p for p in path.rglob("*") if p.is_file() and not p.is_symlink()):
        rel = file_path.relative_to(path).as_posix()
        files.append(
            {
                "path": rel,
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    digest = hashlib.sha256()
    for item in files:
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(item["sha256"]).encode("ascii"))
        digest.update(b"\0")
    return {
        "file_count": len(files),
        "bytes": sum(int(item["bytes"]) for item in files),
        "sha256": digest.hexdigest(),
        "files": files,
    }


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactRetentionError(f"{path}: expected JSON object")
    return payload


def dumps_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def directory_size_bytes(path: Path) -> int:
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not (Path(dirpath) / dirname).is_symlink()
        ]
        for filename in filenames:
            candidate = Path(dirpath) / filename
            if candidate.is_symlink():
                continue
            try:
                total += candidate.stat().st_size
            except OSError:
                continue
    return total


def _rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_optional_json(path: Path, blockers: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        blockers.append(f"missing:{path.name}")
        return None
    try:
        return load_json_object(path)
    except (OSError, json.JSONDecodeError, ArtifactRetentionError) as exc:
        blockers.append(f"invalid_json:{path.name}:{type(exc).__name__}")
        return None


def _path_matches(actual: Any, expected: Path) -> bool:
    if not isinstance(actual, str) or not actual:
        return False
    try:
        return Path(actual).resolve() == expected.resolve()
    except OSError:
        return Path(actual) == expected


def _candidate_locality_manifests(candidate_root: Path) -> list[Path]:
    candidates = [candidate_root / "locality_controls.json"]
    candidates.extend(sorted(candidate_root.glob("*locality_controls*.json")))
    out: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            out.append(path)
    return out


def _load_matching_locality_manifest(
    *,
    candidate_root: Path,
    label: str,
    output_dir: Path,
    blockers: list[str],
) -> tuple[Path | None, dict[str, Any] | None]:
    rejected: list[str] = []
    for manifest_path in _candidate_locality_manifests(candidate_root):
        try:
            manifest = load_json_object(manifest_path)
        except (OSError, json.JSONDecodeError, ArtifactRetentionError) as exc:
            rejected.append(f"{manifest_path.name}:invalid:{type(exc).__name__}")
            continue
        targets = manifest.get("targets")
        target = targets.get(label) if isinstance(targets, dict) else None
        if isinstance(target, dict) and _path_matches(target.get("output_dir"), output_dir):
            return manifest_path, manifest
        rejected.append(f"{manifest_path.name}:output_dir_mismatch")
    blockers.append(
        "missing_matching_locality_controls_manifest"
        + (":" + ",".join(rejected[:8]) if rejected else "")
    )
    return None, None


def _certify_locality_inflated(path: Path, repo_root: Path) -> RetentionCandidate | None:
    if path.name != "inflated":
        return None
    target_root = path.parent
    label = target_root.name
    locality_root = target_root.parent
    if label not in {
        "parent",
        "selective",
        "global_mutated",
    }:
        return None
    if not (
        locality_root.name in {"locality_work", "locality_controls_work"}
        or locality_root.name.endswith("_locality_work")
        or locality_root.name.endswith("_locality_work_fullbatch")
    ):
        return None
    candidate_root = locality_root.parent
    blockers: list[str] = []
    manifest_path, manifest = _load_matching_locality_manifest(
        candidate_root=candidate_root,
        label=label,
        output_dir=path,
        blockers=blockers,
    )
    certificate: dict[str, Any] = {
        "kind": "locality_inflated_raw",
        "label": label,
        "locality_root": _rel(locality_root, repo_root),
    }
    if manifest_path is not None:
        certificate["manifest_path"] = _rel(manifest_path, repo_root)
    if manifest is not None:
        if manifest.get("schema") != "decoder_q_selective_runtime_locality_controls.v1":
            blockers.append("locality_schema_mismatch")
        if manifest.get("locality_controls_passed") is not True:
            blockers.append("locality_controls_not_passed")
        mismatches = manifest.get("mismatch_counts")
        if not isinstance(mismatches, dict) or any(
            int(value or 0) != 0 for value in mismatches.values()
        ):
            blockers.append("locality_mismatch_counts_not_zero")
        targets = manifest.get("targets")
        target = targets.get(label) if isinstance(targets, dict) else None
        if not isinstance(target, dict):
            blockers.append(f"locality_target_missing:{label}")
        else:
            if not _path_matches(target.get("output_dir"), path):
                blockers.append("locality_output_dir_mismatch")
            for key in ("archive_zip", "entrypoint_path"):
                raw = target.get(key)
                if not isinstance(raw, str) or not Path(raw).is_file():
                    blockers.append(f"locality_{key}_missing")
            returncode = target.get("returncode")
            if not isinstance(returncode, int) or returncode != 0:
                blockers.append("locality_inflate_returncode_nonzero")
            certificate["archive_sha256"] = target.get("archive_sha256")
            certificate["entrypoint_sha256"] = target.get("entrypoint_sha256")
        hashes = manifest.get("hashes")
        raw_files = (
            hashes.get("0.raw", {}).get("raw_files")
            if isinstance(hashes, dict)
            else None
        )
        if not isinstance(raw_files, dict) or not isinstance(raw_files.get(label), str):
            blockers.append(f"locality_raw_sha_missing:{label}")
        else:
            certificate["raw_sha256"] = raw_files[label]
            raw_path = path / "0.raw"
            if not raw_path.is_file():
                blockers.append("locality_raw_file_missing:0.raw")
            elif sha256_file(raw_path) != raw_files[label]:
                blockers.append("locality_raw_sha_mismatch:0.raw")
        if manifest_path is not None and manifest_path.is_file():
            certificate["manifest_sha256"] = sha256_file(manifest_path)
    return RetentionCandidate(
        path=_rel(path, repo_root),
        kind="locality_inflated_raw",
        bytes=directory_size_bytes(path),
        certified_rebuildable=not blockers,
        certificate=certificate,
        blockers=blockers,
    )


def _certify_local_cpu_advisory(path: Path, repo_root: Path) -> RetentionCandidate | None:
    if path.name not in {"inflated", "extracted"}:
        return None
    work_dir = path.parent
    if not (
        work_dir.name == "local_cpu_advisory_work"
        or work_dir.name.endswith("_cpu_advisory_work")
        or work_dir.name.endswith("_cpu_advisory_work_venv")
    ):
        return None
    kind = (
        "local_cpu_advisory_inflated_raw"
        if path.name == "inflated"
        else "local_cpu_advisory_extracted_scratch"
    )
    blockers: list[str] = []
    eval_path = work_dir / "contest_auth_eval.json"
    manifest_path = work_dir / "inflated_outputs_manifest.json"
    provenance_path = work_dir / "provenance.json"
    archive_path = work_dir / "archive.zip"
    eval_payload = _load_optional_json(eval_path, blockers)
    output_manifest = _load_optional_json(manifest_path, blockers)
    provenance = _load_optional_json(provenance_path, blockers)
    if not archive_path.is_file():
        blockers.append("missing:archive.zip")
    if eval_payload is not None:
        if eval_payload.get("score_claim") is not False:
            blockers.append("local_advisory_score_claim_not_false")
        if eval_payload.get("n_samples") != 600:
            blockers.append("local_advisory_n_samples_not_600")
    if output_manifest is not None:
        payload = output_manifest.get("payload")
        files = payload.get("files") if isinstance(payload, dict) else None
        if files is None:
            files = output_manifest.get("files")
        if not isinstance(files, list) or not files:
            blockers.append("inflated_manifest_files_missing")
        elif path.name == "inflated":
            _append_manifest_file_hash_blockers(
                blockers,
                root=path,
                files=files,
                prefix="inflated_manifest",
            )
    certificate = {
        "kind": kind,
        "eval_path": _rel(eval_path, repo_root),
        "inflated_outputs_manifest_path": _rel(manifest_path, repo_root),
        "provenance_path": _rel(provenance_path, repo_root),
        "archive_zip_path": _rel(archive_path, repo_root),
    }
    for label, cert_path in (
        ("eval_sha256", eval_path),
        ("inflated_outputs_manifest_sha256", manifest_path),
        ("provenance_sha256", provenance_path),
        ("archive_zip_sha256", archive_path),
    ):
        if cert_path.is_file():
            certificate[label] = sha256_file(cert_path)
    if isinstance(provenance, dict):
        certificate["command"] = provenance.get("command")
    return RetentionCandidate(
        path=_rel(path, repo_root),
        kind=kind,
        bytes=directory_size_bytes(path),
        certified_rebuildable=not blockers,
        certificate=certificate,
        blockers=blockers,
    )


def _certify_mlx_cache(path: Path, repo_root: Path) -> RetentionCandidate | None:
    blockers: list[str] = []
    manifest_path = path / "manifest.json"
    if path.name not in KNOWN_MLX_SCORER_INPUT_CACHE_DIR_NAMES:
        try:
            preview = load_json_object(manifest_path)
        except (OSError, json.JSONDecodeError, ArtifactRetentionError):
            return None
        if preview.get("schema_version") != "mlx_scorer_input_cache.v1":
            return None
    manifest = _load_optional_json(manifest_path, blockers)
    certificate = {
        "kind": "mlx_scorer_input_cache",
        "manifest_path": _rel(manifest_path, repo_root),
    }
    if manifest is not None:
        arrays = manifest.get("array_sha256")
        if not isinstance(arrays, dict) or not {
            "pair_indices",
            "posenet_yuv6_pair",
            "segnet_last_rgb",
        }.issubset(arrays):
            blockers.append("mlx_cache_array_hashes_missing")
        artifact_hashes = _mlx_cache_artifact_hashes(manifest, blockers)
        for key in (
            "archive_sha256",
            "raw_sha256",
            "inflated_outputs_aggregate_sha256",
        ):
            if not isinstance(manifest.get(key), str):
                blockers.append(f"mlx_cache_{key}_missing")
            else:
                certificate[key] = manifest[key]
        if manifest_path.is_file():
            certificate["manifest_sha256"] = sha256_file(manifest_path)
        _append_named_file_hash_blockers(
            blockers,
            root=path,
            hashes=artifact_hashes,
            names={
                "pair_indices": "pair_indices.npy",
                "posenet_yuv6_pair": "posenet_yuv6_pair.npy",
                "segnet_last_rgb": "segnet_last_rgb.npy",
            },
            prefix="mlx_cache",
        )
        identity_certificate = _mlx_cache_identity_certificate(
            manifest,
            cache_root=path,
            repo_root=repo_root,
            blockers=blockers,
        )
        if identity_certificate:
            certificate["identity_audit"] = identity_certificate
    return RetentionCandidate(
        path=_rel(path, repo_root),
        kind="mlx_scorer_input_cache",
        bytes=directory_size_bytes(path),
        certified_rebuildable=not blockers,
        certificate=certificate,
        blockers=blockers,
    )


def _certify_inverse_scorer_inflate_parity_raw(
    path: Path,
    repo_root: Path,
) -> RetentionCandidate | None:
    if not path.is_dir() or not _contains_raw_file(path):
        return None
    proof_paths = _candidate_inverse_scorer_parity_proofs(path, repo_root)
    if not proof_paths:
        return None

    blocked: list[RetentionCandidate] = []
    for proof_path in proof_paths:
        try:
            proof = load_json_object(proof_path)
        except (OSError, json.JSONDecodeError, ArtifactRetentionError):
            continue
        role = _inverse_scorer_parity_output_role(
            proof,
            output_path=path,
            repo_root=repo_root,
        )
        if role is None:
            continue
        candidate = _inverse_scorer_parity_candidate(
            output_path=path,
            repo_root=repo_root,
            proof_path=proof_path,
            proof=proof,
            role=role,
        )
        if candidate.certified_rebuildable:
            return candidate
        blocked.append(candidate)

    if blocked:
        blockers = _ordered_unique(
            blocker
            for candidate in blocked
            for blocker in candidate.blockers
        )
        certificate = dict(blocked[0].certificate)
        certificate["matching_proof_count"] = len(blocked)
        return RetentionCandidate(
            path=_rel(path, repo_root),
            kind=INVERSE_SCORER_INFLATE_PARITY_RAW_KIND,
            bytes=directory_size_bytes(path),
            certified_rebuildable=False,
            certificate=certificate,
            blockers=blockers,
        )
    return None


def _candidate_inverse_scorer_parity_proofs(path: Path, repo_root: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    ancestors = [path, *path.parents]
    for depth, ancestor in enumerate(ancestors):
        if depth > 8:
            break
        try:
            entries = list(ancestor.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_file():
                continue
            if entry.is_symlink():
                continue
            name = entry.name
            if name in INVERSE_SCORER_PARITY_PROOF_NAMES or (
                "inflate_parity" in name
                and "probe" in name
                and name.endswith(".json")
            ):
                resolved = entry.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                candidates.append(entry)
        try:
            ancestor.resolve().relative_to(repo_root.resolve())
        except ValueError:
            continue
        if ancestor.resolve() == repo_root.resolve():
            break
    return sorted(candidates)


def _inverse_scorer_parity_output_role(
    proof: dict[str, Any],
    *,
    output_path: Path,
    repo_root: Path,
) -> str | None:
    for role in ("source", "candidate"):
        tree = proof.get(f"{role}_output_tree")
        tree_path = tree.get("path") if isinstance(tree, dict) else None
        if _referenced_path_matches(tree_path, output_path, repo_root=repo_root):
            return role
        run = proof.get(f"{role}_inflate_run")
        run_path = run.get("output_dir") if isinstance(run, dict) else None
        if _referenced_path_matches(run_path, output_path, repo_root=repo_root):
            return role
    return None


def _inverse_scorer_parity_candidate(
    *,
    output_path: Path,
    repo_root: Path,
    proof_path: Path,
    proof: dict[str, Any],
    role: str,
) -> RetentionCandidate:
    blockers: list[str] = []
    proof_sha = sha256_file(proof_path)
    source_tree = proof.get("source_output_tree")
    candidate_tree = proof.get("candidate_output_tree")
    tree = proof.get(f"{role}_output_tree")
    certificate: dict[str, Any] = {
        "kind": INVERSE_SCORER_INFLATE_PARITY_RAW_KIND,
        "role": role,
        "proof_path": _rel(proof_path, repo_root),
        "proof_sha256": proof_sha,
        "proof_schema": proof.get("schema"),
    }

    _append_inverse_scorer_strict_proof_blockers(blockers, proof, proof_path=proof_path)
    _append_inverse_scorer_runtime_rebuild_blockers(
        blockers,
        proof,
        repo_root=repo_root,
        output_path=output_path,
    )
    _append_inverse_scorer_output_tree_blockers(
        blockers,
        tree,
        output_path=output_path,
        repo_root=repo_root,
        prefix=f"inverse_scorer_{role}_output_tree",
    )
    _append_inverse_scorer_tree_pair_blockers(blockers, source_tree, candidate_tree)
    _append_preserved_reference_blockers(
        blockers,
        proof_path,
        output_path=output_path,
        prefix="inverse_scorer_parity_probe",
    )

    if isinstance(tree, dict):
        certificate["output_tree_sha256"] = tree.get("tree_sha256")
        certificate["output_tree_total_bytes"] = tree.get("total_bytes")
        certificate["output_tree_file_count"] = tree.get("file_count")
    descriptor = proof.get("inverse_scorer_cell_descriptor")
    if isinstance(descriptor, dict):
        certificate["descriptor_packet_sha256"] = descriptor.get("packet_sha256")
        certificate["descriptor_packet_offset"] = descriptor.get("packet_offset")
        certificate["descriptor_packet_bytes"] = descriptor.get("packet_bytes")
    for key in ("source_archive_inflated", "candidate_archive_inflated", "inflate_runtime"):
        value = proof.get(key)
        if isinstance(value, dict):
            certificate[key] = {
                item_key: value.get(item_key)
                for item_key in ("path", "sha256", "inflate_sh", "inflate_sh_sha256")
                if item_key in value
            }
    return RetentionCandidate(
        path=_rel(output_path, repo_root),
        kind=INVERSE_SCORER_INFLATE_PARITY_RAW_KIND,
        bytes=directory_size_bytes(output_path),
        certified_rebuildable=not blockers,
        certificate=certificate,
        blockers=blockers,
    )


def _append_inverse_scorer_strict_proof_blockers(
    blockers: list[str],
    proof: dict[str, Any],
    *,
    proof_path: Path,
) -> None:
    if proof.get("schema") != INVERSE_SCORER_INFLATE_PARITY_PROBE_SCHEMA:
        blockers.append("inverse_scorer_parity_probe_schema_mismatch")
    if proof.get("proof_scope") != INVERSE_SCORER_INFLATE_PARITY_SCOPE:
        blockers.append("inverse_scorer_parity_scope_not_full_frame")
    if proof.get("full_frame_inflate_output_parity_claim") is not True:
        blockers.append("inverse_scorer_full_frame_parity_claim_not_true")
    if proof.get("output_contract_paths_match") is not True:
        blockers.append("inverse_scorer_output_contract_paths_not_matched")
    if proof.get("output_contract_nonempty") is not True:
        blockers.append("inverse_scorer_output_contract_empty")
    if proof.get("output_bytes_identical") is not True:
        blockers.append("inverse_scorer_output_bytes_not_identical")
    if INVERSE_SCORER_PARITY_BLOCKER not in set(proof.get("cleared_blockers") or []):
        blockers.append("inverse_scorer_parity_blocker_not_cleared")
    if proof.get("work_dir_retained") is not True:
        blockers.append("inverse_scorer_work_dir_retained_not_true")
    proof_blockers = proof.get("blockers")
    if not isinstance(proof_blockers, list):
        blockers.append("inverse_scorer_parity_probe_blockers_not_list")
    elif proof_blockers:
        blockers.append("inverse_scorer_parity_probe_has_blockers")
    dispatch_blockers = {str(item) for item in proof.get("dispatch_blockers") or []}
    for required in (
        "inverse_scorer_cell_inflate_parity_is_not_score_authority",
        "exact_auth_eval_required_before_score_claim",
    ):
        if required not in dispatch_blockers:
            blockers.append(f"inverse_scorer_parity_dispatch_blocker_missing:{required}")
    _append_required_false_blockers(
        blockers,
        proof,
        prefix="inverse_scorer_parity_probe",
        fields=INVERSE_SCORER_AUTHORITY_FALSE_FIELDS,
    )
    _append_recursive_truthy_authority_blockers(
        blockers,
        proof,
        prefix="inverse_scorer_parity_probe",
        fields=set(INVERSE_SCORER_AUTHORITY_FALSE_FIELDS),
    )
    if not proof_path.is_file():
        blockers.append("inverse_scorer_parity_probe_file_missing")
    if proof_path.is_symlink():
        blockers.append("inverse_scorer_parity_probe_is_symlink")


def _append_inverse_scorer_runtime_rebuild_blockers(
    blockers: list[str],
    proof: dict[str, Any],
    *,
    repo_root: Path,
    output_path: Path,
) -> None:
    for key in ("source_inflate_run", "candidate_inflate_run"):
        run = proof.get(key)
        if not isinstance(run, dict):
            blockers.append(f"inverse_scorer_{key}_missing")
            continue
        if run.get("returncode") != 0:
            blockers.append(f"inverse_scorer_{key}_returncode_nonzero")
        if run.get("full_frame_file_list_claim") is not True:
            blockers.append(f"inverse_scorer_{key}_full_frame_file_list_claim_not_true")
        entries = run.get("file_list_entries")
        if not isinstance(entries, list) or not entries:
            blockers.append(f"inverse_scorer_{key}_file_list_entries_missing")

    runtime = proof.get("inflate_runtime")
    if not isinstance(runtime, dict):
        blockers.append("inverse_scorer_inflate_runtime_missing")
    else:
        inflate_sh = _resolve_referenced_path_preserving_leaf(
            runtime.get("inflate_sh"),
            repo_root=repo_root,
            cache_root=output_path,
        )
        expected_sha = runtime.get("inflate_sh_sha256")
        if inflate_sh is None or not inflate_sh.is_file():
            blockers.append("inverse_scorer_inflate_runtime_inflate_sh_missing")
        elif inflate_sh.is_symlink():
            blockers.append("inverse_scorer_inflate_runtime_inflate_sh_path_is_symlink")
        elif not isinstance(expected_sha, str) or len(expected_sha) != 64:
            blockers.append("inverse_scorer_inflate_runtime_inflate_sh_sha256_missing")
        elif sha256_file(inflate_sh) != expected_sha:
            blockers.append("inverse_scorer_inflate_runtime_inflate_sh_sha256_mismatch")
        else:
            _append_preserved_reference_blockers(
                blockers,
                inflate_sh,
                output_path=output_path,
                prefix="inverse_scorer_inflate_runtime_inflate_sh",
            )
        if runtime.get("full_frame_file_list_claim") is not True:
            blockers.append("inverse_scorer_inflate_runtime_full_frame_file_list_claim_not_true")

    for key in ("source_archive_inflated", "candidate_archive_inflated"):
        _append_inverse_scorer_file_record_blockers(
            blockers,
            proof.get(key),
            repo_root=repo_root,
            output_path=output_path,
            prefix=f"inverse_scorer_{key}",
        )
    _append_inverse_scorer_file_record_blockers(
        blockers,
        proof.get("candidate_manifest"),
        repo_root=repo_root,
        output_path=output_path,
        prefix="inverse_scorer_candidate_manifest",
    )

    descriptor = proof.get("inverse_scorer_cell_descriptor")
    if not isinstance(descriptor, dict):
        blockers.append("inverse_scorer_descriptor_missing")
    else:
        for key in ("packet_sha256", "json_sha256"):
            value = descriptor.get(key)
            if not isinstance(value, str) or len(value) != 64:
                blockers.append(f"inverse_scorer_descriptor_{key}_missing")
        if not isinstance(descriptor.get("packet_offset"), int):
            blockers.append("inverse_scorer_descriptor_packet_offset_missing")
        if not isinstance(descriptor.get("packet_bytes"), int) or descriptor.get("packet_bytes") <= 0:
            blockers.append("inverse_scorer_descriptor_packet_bytes_missing")


def _append_inverse_scorer_file_record_blockers(
    blockers: list[str],
    record: Any,
    *,
    repo_root: Path,
    output_path: Path,
    prefix: str,
) -> None:
    if not isinstance(record, dict):
        blockers.append(f"{prefix}_missing")
        return
    path = _resolve_referenced_path_preserving_leaf(
        record.get("path"),
        repo_root=repo_root,
        cache_root=output_path,
    )
    expected_sha = record.get("sha256")
    if path is None or not path.is_file():
        blockers.append(f"{prefix}_path_missing_or_not_found")
        return
    if path.is_symlink():
        blockers.append(f"{prefix}_path_is_symlink")
        return
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        blockers.append(f"{prefix}_sha256_missing")
    elif sha256_file(path) != expected_sha:
        blockers.append(f"{prefix}_sha256_mismatch")
    _append_preserved_reference_blockers(
        blockers,
        path,
        output_path=output_path,
        prefix=prefix,
    )


def _append_inverse_scorer_output_tree_blockers(
    blockers: list[str],
    tree: Any,
    *,
    output_path: Path,
    repo_root: Path,
    prefix: str,
) -> None:
    if not isinstance(tree, dict):
        blockers.append(f"{prefix}_missing")
        return
    if not _referenced_path_matches(tree.get("path"), output_path, repo_root=repo_root):
        blockers.append(f"{prefix}_path_mismatch")
    if tree.get("exists") is not True:
        blockers.append(f"{prefix}_exists_not_true")
    tree_blockers = tree.get("blockers")
    if not isinstance(tree_blockers, list):
        blockers.append(f"{prefix}_blockers_not_list")
    elif tree_blockers:
        blockers.append(f"{prefix}_has_blockers")
    if not isinstance(tree.get("tree_sha256"), str) or len(tree.get("tree_sha256")) != 64:
        blockers.append(f"{prefix}_tree_sha256_missing")
    files = tree.get("files")
    if not isinstance(files, list) or not files:
        blockers.append(f"{prefix}_files_missing")
    else:
        _append_manifest_file_hash_blockers(
            blockers,
            root=output_path,
            files=files,
            prefix=prefix,
        )
    raw_count = _raw_file_count(output_path)
    if raw_count <= 0:
        blockers.append(f"{prefix}_raw_files_missing")
    if tree.get("file_count") != len(files or []):
        blockers.append(f"{prefix}_file_count_mismatch")
    if int(tree.get("total_bytes") or -1) != directory_size_bytes(output_path):
        blockers.append(f"{prefix}_total_bytes_mismatch")


def _append_inverse_scorer_tree_pair_blockers(
    blockers: list[str],
    source_tree: Any,
    candidate_tree: Any,
) -> None:
    if not isinstance(source_tree, dict) or not isinstance(candidate_tree, dict):
        blockers.append("inverse_scorer_output_tree_pair_missing")
        return
    for label, tree in (("source", source_tree), ("candidate", candidate_tree)):
        tree_blockers = tree.get("blockers")
        if not isinstance(tree_blockers, list):
            blockers.append(f"inverse_scorer_{label}_output_tree_blockers_not_list")
        elif tree_blockers:
            blockers.append(f"inverse_scorer_{label}_output_tree_has_blockers")
    if not isinstance(source_tree.get("files"), list) or not isinstance(candidate_tree.get("files"), list):
        blockers.append("inverse_scorer_output_tree_pair_files_missing")
        return
    if source_tree.get("file_count") != candidate_tree.get("file_count"):
        blockers.append("inverse_scorer_output_tree_pair_file_count_mismatch")
    if source_tree.get("total_bytes") != candidate_tree.get("total_bytes"):
        blockers.append("inverse_scorer_output_tree_pair_total_bytes_mismatch")
    if source_tree.get("tree_sha256") != candidate_tree.get("tree_sha256"):
        blockers.append("inverse_scorer_output_tree_pair_tree_sha256_mismatch")


def _append_preserved_reference_blockers(
    blockers: list[str],
    reference: Path,
    *,
    output_path: Path,
    prefix: str,
) -> None:
    try:
        reference.resolve().relative_to(output_path.resolve())
    except ValueError:
        return
    blockers.append(f"{prefix}_would_be_removed_with_raw_output")


def _contains_raw_file(path: Path) -> bool:
    return _raw_file_count(path) > 0


def _raw_file_count(path: Path) -> int:
    try:
        return sum(1 for item in path.rglob("*.raw") if item.is_file() and not item.is_symlink())
    except OSError:
        return 0


def _mlx_cache_artifact_hashes(
    manifest: dict[str, Any],
    blockers: list[str],
) -> dict[str, str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        blockers.append("mlx_cache_artifacts_missing")
        return {}
    out: dict[str, str] = {}
    for key in ("pair_indices", "posenet_yuv6_pair", "segnet_last_rgb"):
        artifact = artifacts.get(key)
        if not isinstance(artifact, dict):
            blockers.append(f"mlx_cache_artifact_{key}_missing")
            continue
        sha = artifact.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            blockers.append(f"mlx_cache_artifact_{key}_sha256_missing")
            continue
        out[key] = sha
    return out


def _mlx_cache_identity_certificate(
    manifest: dict[str, Any],
    *,
    cache_root: Path,
    repo_root: Path,
    blockers: list[str],
) -> dict[str, Any] | None:
    for stamp_key, expected_verdict, source_keys in (
        (
            "auth_eval_identity_audit",
            "PASS_CACHE_AUTH_EVAL_IDENTITY",
            ("auth_eval_path", "auth_eval_dir"),
        ),
        (
            "local_cpu_advisory_cache_identity_audit",
            "PASS_CACHE_LOCAL_CPU_ADVISORY_IDENTITY",
            ("local_cpu_advisory_path",),
        ),
    ):
        stamp = manifest.get(stamp_key)
        if stamp is None:
            continue
        if not isinstance(stamp, dict):
            blockers.append(f"{stamp_key}_not_object")
            return None
        return _validate_mlx_identity_stamp(
            stamp,
            stamp_key=stamp_key,
            expected_verdict=expected_verdict,
            source_keys=source_keys,
            manifest=manifest,
            cache_root=cache_root,
            repo_root=repo_root,
            blockers=blockers,
        )
    blockers.append("mlx_cache_identity_audit_stamp_missing")
    return None


def _validate_mlx_identity_stamp(
    stamp: dict[str, Any],
    *,
    stamp_key: str,
    expected_verdict: str,
    source_keys: tuple[str, ...],
    manifest: dict[str, Any],
    cache_root: Path,
    repo_root: Path,
    blockers: list[str],
) -> dict[str, Any] | None:
    certificate: dict[str, Any] = {
        "stamp_key": stamp_key,
        "expected_verdict": expected_verdict,
    }
    if stamp.get("verdict") != expected_verdict:
        blockers.append(f"{stamp_key}_verdict_not_{expected_verdict}")
    if stamp.get("passed") is not True:
        blockers.append(f"{stamp_key}_passed_not_true")
    _append_authority_false_blockers(blockers, stamp, prefix=stamp_key)

    audit_path = _resolve_referenced_path(
        stamp.get("path"),
        repo_root=repo_root,
        cache_root=cache_root,
    )
    expected_sha = stamp.get("sha256")
    if audit_path is None:
        blockers.append(f"{stamp_key}_path_missing_or_not_found")
    elif not isinstance(expected_sha, str) or len(expected_sha) != 64:
        blockers.append(f"{stamp_key}_sha256_missing")
    elif sha256_file(audit_path) != expected_sha:
        blockers.append(f"{stamp_key}_sha256_mismatch")
    else:
        certificate["path"] = _rel(audit_path, repo_root)
        certificate["sha256"] = expected_sha
        audit = _load_optional_json(audit_path, blockers)
        if audit is not None:
            _append_mlx_identity_audit_blockers(
                blockers,
                audit,
                manifest=manifest,
                stamp_key=stamp_key,
                expected_verdict=expected_verdict,
            )
            certificate["schema_version"] = audit.get("schema_version")

    source_certificate = _mlx_identity_source_certificate(
        stamp,
        source_keys=source_keys,
        repo_root=repo_root,
        cache_root=cache_root,
        blockers=blockers,
        prefix=stamp_key,
    )
    if source_certificate:
        certificate["source"] = source_certificate
    return certificate


def _append_mlx_identity_audit_blockers(
    blockers: list[str],
    audit: dict[str, Any],
    *,
    manifest: dict[str, Any],
    stamp_key: str,
    expected_verdict: str,
) -> None:
    if audit.get("verdict") != expected_verdict:
        blockers.append(f"{stamp_key}_audit_verdict_not_{expected_verdict}")
    if audit.get("passed") is not True:
        blockers.append(f"{stamp_key}_audit_passed_not_true")
    _append_authority_false_blockers(blockers, audit, prefix=f"{stamp_key}_audit")
    audit_cache = audit.get("cache")
    if not isinstance(audit_cache, dict):
        blockers.append(f"{stamp_key}_audit_cache_missing")
        return
    for key in (
        "archive_sha256",
        "inflated_outputs_aggregate_sha256",
        "raw_sha256",
        "pair_count",
        "array_sha256",
        "hash_domain",
    ):
        if audit_cache.get(key) != manifest.get(key):
            blockers.append(f"{stamp_key}_audit_cache_{key}_mismatch")


def _mlx_identity_source_certificate(
    stamp: dict[str, Any],
    *,
    source_keys: tuple[str, ...],
    repo_root: Path,
    cache_root: Path,
    blockers: list[str],
    prefix: str,
) -> dict[str, Any] | None:
    for key in source_keys:
        source_path = _resolve_referenced_path(
            stamp.get(key),
            repo_root=repo_root,
            cache_root=cache_root,
        )
        if source_path is None:
            continue
        return {
            "key": key,
            "path": _rel(source_path, repo_root),
            "sha256": sha256_file(source_path) if source_path.is_file() else None,
        }
    blockers.append(f"{prefix}_source_path_missing_or_not_found")
    return None


def _resolve_referenced_path(
    value: Any,
    *,
    repo_root: Path,
    cache_root: Path,
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [
        repo_root / raw,
        cache_root / raw,
        raw,
    ]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved.exists():
            return resolved
    return None


def _resolve_referenced_path_preserving_leaf(
    value: Any,
    *,
    repo_root: Path,
    cache_root: Path,
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [
        repo_root / raw,
        cache_root / raw,
        raw,
    ]
    for candidate in candidates:
        if candidate.exists() or candidate.is_symlink():
            return candidate
    return None


def _append_authority_false_blockers(
    blockers: list[str],
    payload: dict[str, Any],
    *,
    prefix: str,
) -> None:
    for name in AUTHORITY_FALSE_FIELDS:
        if payload.get(name) is not False:
            blockers.append(f"{prefix}_{name}_not_false")


def _append_required_false_blockers(
    blockers: list[str],
    payload: dict[str, Any],
    *,
    prefix: str,
    fields: Iterable[str],
) -> None:
    for name in fields:
        if payload.get(name) is not False:
            blockers.append(f"{prefix}_{name}_not_false")


def _append_recursive_truthy_authority_blockers(
    blockers: list[str],
    payload: Any,
    *,
    prefix: str,
    fields: set[str],
) -> None:
    def walk(value: Any, parts: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = str(key)
                next_parts = (*parts, key_text)
                if key_text in fields and item is not False:
                    blockers.append(f"{prefix}_authority_field_not_false:{'.'.join(next_parts)}")
                walk(item, next_parts)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, (*parts, str(index)))

    walk(payload, ())


def _ordered_unique(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _referenced_path_matches(
    actual: Any,
    expected: Path,
    *,
    repo_root: Path,
) -> bool:
    if not isinstance(actual, str) or not actual:
        return False
    raw = Path(actual)
    candidates = [raw] if raw.is_absolute() else [repo_root / raw, raw]
    for candidate in candidates:
        try:
            if candidate.resolve() == expected.resolve():
                return True
        except OSError:
            if candidate == expected:
                return True
    return False


def _append_manifest_file_hash_blockers(
    blockers: list[str],
    *,
    root: Path,
    files: list[Any],
    prefix: str,
) -> None:
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            blockers.append(f"{prefix}_file_{index}_not_object")
            continue
        raw_path = item.get("path") or item.get("name") or item.get("relative_path")
        raw_sha = item.get("sha256")
        if not isinstance(raw_path, str) or not raw_path:
            blockers.append(f"{prefix}_file_{index}_path_missing")
            continue
        if not isinstance(raw_sha, str) or len(raw_sha) != 64:
            blockers.append(f"{prefix}_file_{index}_sha256_missing")
            continue
        raw_file_path = root / raw_path
        if raw_file_path.is_symlink():
            blockers.append(f"{prefix}_file_{index}_is_symlink")
            continue
        file_path = raw_file_path.resolve()
        try:
            file_path.relative_to(root.resolve())
        except ValueError:
            blockers.append(f"{prefix}_file_{index}_path_escapes_root")
            continue
        if not file_path.is_file():
            blockers.append(f"{prefix}_file_{index}_missing")
            continue
        if sha256_file(file_path) != raw_sha:
            blockers.append(f"{prefix}_file_{index}_sha256_mismatch")


def _append_named_file_hash_blockers(
    blockers: list[str],
    *,
    root: Path,
    hashes: dict[str, Any],
    names: dict[str, str],
    prefix: str,
) -> None:
    for key, filename in names.items():
        expected = hashes.get(key)
        file_path = root / filename
        if not isinstance(expected, str) or len(expected) != 64:
            blockers.append(f"{prefix}_{key}_sha256_missing")
            continue
        if not file_path.is_file():
            blockers.append(f"{prefix}_{filename}_missing")
            continue
        if sha256_file(file_path) != expected:
            blockers.append(f"{prefix}_{filename}_sha256_mismatch")


def _certify_unknown_raw_surface(path: Path, repo_root: Path) -> RetentionCandidate | None:
    try:
        raw_files = [p for p in path.iterdir() if p.is_file() and p.suffix == ".raw"]
    except OSError:
        return None
    if not raw_files:
        return None
    return RetentionCandidate(
        path=_rel(path, repo_root),
        kind=UNKNOWN_RAW_KIND,
        bytes=directory_size_bytes(path),
        certified_rebuildable=False,
        certificate={
            "kind": UNKNOWN_RAW_KIND,
            "direct_raw_file_count": len(raw_files),
            "reason": "raw files detected but no known retention certificate matched",
        },
        blockers=["unknown_raw_surface_no_certifier"],
    )


def _certify_unknown_raw_workdir(path: Path, repo_root: Path) -> RetentionCandidate | None:
    if path.name not in KNOWN_RAW_WORKDIR_NAMES and not path.name.endswith(
        (
            "_auth_eval_work",
            "_eval_work",
            "_auth_eval_workdir",
            "_cpu_workdir",
            "_cuda_workdir",
        )
    ):
        return None
    try:
        raw_files = sorted(p for p in path.rglob("*.raw") if p.is_file() and not p.is_symlink())
    except OSError:
        return None
    if not raw_files:
        return None
    raw_bytes = 0
    for raw_file in raw_files:
        try:
            raw_bytes += raw_file.stat().st_size
        except OSError:
            continue
    return RetentionCandidate(
        path=_rel(path, repo_root),
        kind=UNKNOWN_RAW_KIND,
        bytes=directory_size_bytes(path),
        certified_rebuildable=False,
        certificate={
            "kind": UNKNOWN_RAW_KIND,
            "known_raw_workdir_name": path.name,
            "nested_raw_file_count": len(raw_files),
            "nested_raw_bytes": raw_bytes,
            "reason": (
                "known raw workdir contains raw outputs but no known retention "
                "certificate matched"
            ),
        },
        blockers=["unknown_raw_workdir_no_certifier"],
    )


def _iter_candidate_dirs(root: Path) -> Iterable[Path]:
    if root.is_dir():
        for dirpath, dirnames, _filenames in os.walk(root):
            current = Path(dirpath)
            yield current
            if current.name in {
                "inflated",
                "extracted",
                *KNOWN_MLX_SCORER_INPUT_CACHE_DIR_NAMES,
            }:
                dirnames[:] = []


def _is_excluded(path: Path, exclude_paths: set[Path]) -> bool:
    resolved = path.resolve()
    for excluded in exclude_paths:
        try:
            resolved.relative_to(excluded.resolve())
            return True
        except ValueError:
            continue
    return False


def build_retention_plan(
    roots: list[Path],
    *,
    repo_root: Path,
    include_kinds: set[str] | None = None,
    min_bytes: int = 1 << 30,
    exclude_paths: list[Path] | None = None,
) -> RetentionPlan:
    include = set(DEFAULT_RETENTION_KINDS if include_kinds is None else include_kinds)
    excludes = set(exclude_paths or [])
    candidates: list[RetentionCandidate] = []
    blocked: list[RetentionCandidate] = []
    seen: set[Path] = set()
    certifiers = (
        _certify_locality_inflated,
        _certify_local_cpu_advisory,
        _certify_mlx_cache,
        _certify_inverse_scorer_inflate_parity_raw,
    )
    for root in roots:
        for path in _iter_candidate_dirs(root):
            resolved = path.resolve()
            if resolved in seen or _is_excluded(path, excludes):
                continue
            seen.add(resolved)
            candidate = None
            for certifier in certifiers:
                candidate = certifier(path, repo_root)
                if candidate is not None:
                    break
            if candidate is None:
                unknown = _certify_unknown_raw_workdir(path, repo_root)
                if unknown is None:
                    unknown = _certify_unknown_raw_surface(path, repo_root)
                if unknown is not None and unknown.bytes >= min_bytes:
                    blocked.append(unknown)
                continue
            if candidate.kind not in include:
                continue
            if candidate.bytes < min_bytes:
                continue
            if candidate.certified_rebuildable:
                candidates.append(candidate)
            else:
                blocked.append(candidate)
    candidates.sort(key=lambda row: (-row.bytes, row.path))
    blocked.sort(key=lambda row: (-row.bytes, row.path))
    return RetentionPlan(
        schema=SCHEMA,
        generated_at_utc=datetime.now(UTC).isoformat(),
        repo_root=str(repo_root),
        roots=[str(root) for root in roots],
        include_kinds=sorted(include),
        min_bytes=int(min_bytes),
        exclude_paths=[str(path) for path in sorted(excludes)],
        candidates=candidates,
        blocked_candidates=blocked,
    )


def _normalize_move_roots(
    *,
    cold_store_root: Path | None,
    cold_store_roots: Iterable[Path] | None,
) -> list[Path]:
    roots: list[Path] = []
    if cold_store_roots is not None:
        roots.extend(Path(root) for root in cold_store_roots)
    if cold_store_root is not None:
        roots.insert(0, Path(cold_store_root))
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(root)
    return out


def _validate_move_roots(
    roots: list[Path],
    *,
    repo_root: Path,
    required_bytes: int,
    tiered: bool,
    reserve_bytes: int,
) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for index, root in enumerate(roots):
        contract = validate_cold_store_root(
            root,
            repo_root=repo_root,
            required_bytes=reserve_bytes if tiered else required_bytes + reserve_bytes,
        )
        contract["tier_index"] = index
        contract["reserve_bytes"] = int(reserve_bytes)
        contracts.append(contract)
    return contracts


def _allowed_source_roots(plan: RetentionPlan, *, repo_root: Path) -> list[Path]:
    roots = [repo_root]
    roots.extend(Path(root) for root in plan.roots)
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(root)
    return out


def _candidate_source_path(candidate: RetentionCandidate, *, repo_root: Path) -> Path:
    path = Path(candidate.path)
    return path if path.is_absolute() else repo_root / path


def _is_under_any(path: Path, roots: Iterable[Path]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.resolve(strict=False)
    for root in roots:
        try:
            resolved.relative_to(root.resolve(strict=False))
            return True
        except ValueError:
            continue
    return False


def _cold_store_relative_path(
    candidate_path: str,
    *,
    repo_root: Path,
    allowed_source_roots: Iterable[Path],
) -> Path:
    path = Path(candidate_path)
    if not path.is_absolute():
        return path
    resolved = path.resolve(strict=False)
    repo = repo_root.resolve(strict=False)
    try:
        return resolved.relative_to(repo)
    except ValueError:
        pass
    for root in allowed_source_roots:
        root_resolved = root.resolve(strict=False)
        try:
            rel = resolved.relative_to(root_resolved)
        except ValueError:
            continue
        root_slug = "_".join(part for part in root_resolved.parts if part and part != "/")
        return Path("__external__") / root_slug / rel
    return Path("__absolute__") / Path(*resolved.parts[1:])


def _select_cold_store_root(
    candidate: RetentionCandidate,
    *,
    roots: list[Path],
    planned_free_by_tier: dict[int, int],
    reserve_bytes: int,
    source_device_id: int | None = None,
) -> tuple[int, Path, int] | None:
    eligible: list[tuple[int, Path, int]] = []
    for index, root in enumerate(roots):
        free_before = int(planned_free_by_tier.get(index, 0))
        if free_before - int(candidate.bytes) >= int(reserve_bytes):
            eligible.append((index, root, free_before))
    if not eligible:
        return None
    if source_device_id is not None:
        for selected in eligible:
            _, root, _ = selected
            try:
                if _path_device_id(root) != source_device_id:
                    return selected
            except OSError:
                continue
    return eligible[0]


def _path_device_id(path: Path) -> int:
    return int(path.stat().st_dev)


def execute_retention_plan(
    plan: RetentionPlan,
    *,
    action: str,
    cold_store_root: Path | None = None,
    cold_store_roots: Iterable[Path] | None = None,
    cold_store_reserve_bytes: int = 0,
    journal_path: Path | None = None,
) -> dict[str, Any]:
    if action not in {"delete", "move"}:
        raise ArtifactRetentionError("action must be delete or move")
    move_roots = _normalize_move_roots(
        cold_store_root=cold_store_root,
        cold_store_roots=cold_store_roots,
    )
    tiered_cold_store = cold_store_roots is not None
    if action == "move" and not move_roots:
        raise ArtifactRetentionError("move action requires cold_store_root")
    if isinstance(cold_store_reserve_bytes, bool) or int(cold_store_reserve_bytes) < 0:
        raise ArtifactRetentionError("cold_store_reserve_bytes must be non-negative")
    repo_root = Path(plan.repo_root)
    allowed_source_roots = _allowed_source_roots(plan, repo_root=repo_root)
    cold_store_contracts = _validate_move_roots(
        move_roots,
        repo_root=repo_root,
        required_bytes=plan.total_reclaimable_bytes,
        tiered=tiered_cold_store,
        reserve_bytes=int(cold_store_reserve_bytes),
    )
    planned_free_by_tier = {
        index: int(contract["free_bytes"]) for index, contract in enumerate(cold_store_contracts)
    }
    cold_store_contract = (
        cold_store_contracts[0] if len(cold_store_contracts) == 1 and not tiered_cold_store else None
    )
    rows: list[dict[str, Any]] = []
    if journal_path is not None:
        _append_journal(
            journal_path,
            {
                "schema": EXECUTION_JOURNAL_SCHEMA,
                "event": "start",
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "action": action,
                "candidate_count": len(plan.candidates),
                "plan_schema": plan.schema,
                "cold_store_contract": cold_store_contract,
                "cold_store_contracts": cold_store_contracts,
                "tiered_cold_store": tiered_cold_store,
            },
        )
    for candidate in plan.candidates:
        source = _candidate_source_path(candidate, repo_root=repo_root)
        row = candidate.to_dict()
        row["action"] = action
        row["preflight_revalidated"] = False
        if journal_path is not None:
            _append_journal(journal_path, {"event": "candidate_start", "row": row})
        try:
            if not source.exists():
                row["status"] = "skipped_missing"
                rows.append(row)
                if journal_path is not None:
                    _append_journal(journal_path, {"event": "candidate_end", "row": row})
                continue
            revalidation_blockers = _execution_revalidation_blockers(
                source,
                candidate,
                repo_root,
                allowed_source_roots=allowed_source_roots,
            )
            if revalidation_blockers:
                row["status"] = "skipped_revalidation_failed"
                row["revalidation_blockers"] = revalidation_blockers
                rows.append(row)
                if journal_path is not None:
                    _append_journal(journal_path, {"event": "candidate_end", "row": row})
                continue
            row["preflight_revalidated"] = True
            if action == "delete":
                _delete_certified_tree(
                    source,
                    repo_root=repo_root,
                    bytes_estimate=candidate.bytes,
                    allowed_source_roots=allowed_source_roots,
                )
                row["status"] = "deleted"
            else:
                selected = _select_cold_store_root(
                    candidate,
                    roots=move_roots,
                    planned_free_by_tier=planned_free_by_tier,
                    reserve_bytes=int(cold_store_reserve_bytes),
                    source_device_id=_path_device_id(source),
                )
                if selected is None:
                    row["status"] = "skipped_no_cold_store_capacity"
                    rows.append(row)
                    if journal_path is not None:
                        _append_journal(journal_path, {"event": "candidate_end", "row": row})
                    continue
                tier_index, selected_root, free_before = selected
                destination = selected_root / _cold_store_relative_path(
                    candidate.path,
                    repo_root=repo_root,
                    allowed_source_roots=allowed_source_roots,
                )
                verification = _copy_verify_then_delete(
                    source,
                    destination,
                    repo_root=repo_root,
                    bytes_estimate=candidate.bytes,
                    allowed_source_roots=allowed_source_roots,
                )
                planned_free_by_tier[tier_index] = free_before - int(candidate.bytes)
                row["status"] = "moved"
                row["cold_store_tier_index"] = tier_index
                row["cold_store_root"] = str(selected_root)
                row["cold_store_path"] = str(destination)
                row["cold_store_free_bytes_before_planned"] = free_before
                row["cold_store_free_bytes_after_planned"] = planned_free_by_tier[tier_index]
                row["cold_store_reserve_bytes"] = int(cold_store_reserve_bytes)
                row["cold_store_verification"] = verification
                row["local_bytes_reclaimed"] = (
                    0
                    if verification.get("same_device_as_source") is True
                    else int(candidate.bytes)
                )
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            if journal_path is not None:
                _append_journal(journal_path, {"event": "candidate_error", "row": row})
                _append_journal(journal_path, {"event": "candidate_end", "row": row})
            raise
        rows.append(row)
        if journal_path is not None:
            _append_journal(journal_path, {"event": "candidate_end", "row": row})
    return {
        "schema": EXECUTION_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "action": action,
        "cold_store_root": None if cold_store_root is None else str(cold_store_root),
        "cold_store_roots": [str(root) for root in move_roots],
        "cold_store_contract": cold_store_contract,
        "cold_store_contracts": cold_store_contracts,
        "cold_store_reserve_bytes": int(cold_store_reserve_bytes),
        "tiered_cold_store": tiered_cold_store,
        "journal_path": None if journal_path is None else str(journal_path),
        "executed_count": sum(1 for row in rows if row.get("status") in {"deleted", "moved"}),
        "skipped_count": sum(1 for row in rows if str(row.get("status") or "").startswith("skipped")),
        "executed_bytes": sum(
            int(row.get("bytes") or 0)
            for row in rows
            if row.get("status") in {"deleted", "moved"}
        ),
        "local_bytes_reclaimed": sum(
            (
                int(row.get("bytes") or 0)
                if row.get("status") == "deleted"
                else int(row.get("local_bytes_reclaimed") or 0)
            )
            for row in rows
            if row.get("status") in {"deleted", "moved"}
        ),
        "rows": rows,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _append_journal(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def validate_cold_store_root(
    cold_store_root: Path | None,
    *,
    repo_root: Path,
    required_bytes: int,
) -> dict[str, Any]:
    """Validate a cold-store root before any source tree is moved."""

    if cold_store_root is None:
        raise ArtifactRetentionError("cold_store_root is required")
    root = cold_store_root.expanduser().resolve()
    repo = repo_root.expanduser().resolve()
    if not root.exists():
        raise ArtifactRetentionError(f"cold-store root does not exist: {root}")
    if root.is_symlink():
        raise ArtifactRetentionError(f"cold-store root must not be a symlink: {root}")
    if not root.is_dir():
        raise ArtifactRetentionError(f"cold-store root is not a directory: {root}")
    try:
        root.relative_to(repo)
    except ValueError:
        pass
    else:
        raise ArtifactRetentionError(
            f"cold-store root must be outside repo_root: root={root}:repo={repo}"
        )
    usage = shutil.disk_usage(root)
    if usage.free < required_bytes:
        raise ArtifactRetentionError(
            f"cold-store free space insufficient: free={usage.free}:required={required_bytes}"
        )
    root_stat = root.stat()
    repo_stat = repo.stat()
    probe_path = root / (
        f".artifact_retention_write_probe_{os.getpid()}_"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.tmp"
    )
    probe_payload = b"comma_lab_artifact_retention_cold_store_probe_v1\n"
    try:
        with probe_path.open("wb") as handle:
            handle.write(probe_payload)
            handle.flush()
            os.fsync(handle.fileno())
        if probe_path.read_bytes() != probe_payload:
            raise ArtifactRetentionError("cold-store write probe readback mismatch")
    finally:
        try:
            probe_path.unlink()
        except FileNotFoundError:
            pass
    return {
        "schema": "comma_lab.artifact_retention_cold_store_contract.v1",
        "path": str(root),
        "repo_root": str(repo),
        "required_bytes": int(required_bytes),
        "free_bytes": int(usage.free),
        "device_id": int(root_stat.st_dev),
        "repo_device_id": int(repo_stat.st_dev),
        "same_device_as_repo": root_stat.st_dev == repo_stat.st_dev,
        "write_probe_passed": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _execution_revalidation_blockers(
    source: Path,
    candidate: RetentionCandidate,
    repo_root: Path,
    *,
    allowed_source_roots: Iterable[Path],
) -> list[str]:
    blockers: list[str] = []
    if not _is_under_any(source, allowed_source_roots):
        blockers.append("source_outside_allowed_roots")
    if not candidate.certified_rebuildable:
        blockers.append("candidate_not_certified_rebuildable")
    if source.is_symlink():
        blockers.append("source_is_symlink")
    if not source.is_dir():
        blockers.append("source_not_directory")
    certifier = {
        "locality_inflated_raw": _certify_locality_inflated,
        "local_cpu_advisory_inflated_raw": _certify_local_cpu_advisory,
        "local_cpu_advisory_extracted_scratch": _certify_local_cpu_advisory,
        "mlx_scorer_input_cache": _certify_mlx_cache,
        INVERSE_SCORER_INFLATE_PARITY_RAW_KIND: _certify_inverse_scorer_inflate_parity_raw,
    }.get(candidate.kind)
    if certifier is None:
        blockers.append(f"unknown_candidate_kind:{candidate.kind}")
        return blockers
    refreshed = certifier(source, repo_root)
    if refreshed is None:
        blockers.append("candidate_no_longer_matches_certifier")
        return blockers
    if not refreshed.certified_rebuildable:
        blockers.extend(f"refreshed:{blocker}" for blocker in refreshed.blockers)
    if refreshed.bytes != candidate.bytes:
        blockers.append(f"bytes_changed:plan={candidate.bytes}:current={refreshed.bytes}")
    return blockers


def _delete_certified_tree(
    source: Path,
    *,
    repo_root: Path,
    bytes_estimate: int,
    allowed_source_roots: Iterable[Path] | None = None,
) -> None:
    allowed_roots = [repo_root]
    if allowed_source_roots is not None:
        allowed_roots.extend(allowed_source_roots)
    if not _is_under_any(source, allowed_roots):
        raise ArtifactRetentionError(f"refusing to delete outside allowed roots: {source}")
    experiments_results = (repo_root / "experiments/results").resolve()
    resolved = source.resolve()
    try:
        resolved.relative_to(experiments_results)
    except ValueError:
        shutil.rmtree(source)
        return
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools.gc_experiments_results import execute_plan

    rel = resolved.relative_to(repo_root.resolve()).as_posix()
    execute_plan(
        {
            "schema": "pact.experiments_results_gc_plan.v1",
            "would_delete": [
                {
                    "path": rel,
                    "bytes_estimate": int(bytes_estimate),
                    "rationale": "certified_rebuildable_artifact_retention",
                }
            ],
        },
        repo_root=repo_root,
        operator_approved="codex:certified_rebuildable_artifact_retention",
        verbose=False,
    )


def _copy_verify_then_delete(
    source: Path,
    destination: Path,
    *,
    repo_root: Path,
    bytes_estimate: int,
    allowed_source_roots: Iterable[Path] | None = None,
) -> dict[str, Any]:
    destination = destination.resolve()
    partial_destination = destination.with_name(
        f"{destination.name}.partial-{os.getpid()}-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    if destination.exists():
        raise ArtifactRetentionError(f"cold-store destination exists: {destination}")
    if partial_destination.exists():
        raise ArtifactRetentionError(
            f"cold-store partial destination exists: {partial_destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_device_id = _path_device_id(source)
    destination_device_id = _path_device_id(destination.parent)
    same_device_as_source = source_device_id == destination_device_id
    source_digest = directory_digest(source)
    if same_device_as_source:
        moved_to_partial = False
        finalized = False
        try:
            source.replace(partial_destination)
            moved_to_partial = True
            destination_digest = directory_digest(partial_destination)
            if source_digest["sha256"] != destination_digest["sha256"]:
                raise ArtifactRetentionError("cold-store copy verification failed")
            partial_destination.replace(destination)
            finalized = True
            final_digest = directory_digest(destination)
            if source_digest["sha256"] != final_digest["sha256"]:
                raise ArtifactRetentionError("cold-store copy verification failed")
        except Exception:
            if finalized and destination.exists() and not source.exists():
                try:
                    destination.replace(source)
                except OSError:
                    pass
            elif moved_to_partial and partial_destination.exists() and not source.exists():
                try:
                    partial_destination.replace(source)
                except OSError:
                    pass
            shutil.rmtree(partial_destination, ignore_errors=True)
            raise
        return {
            "schema": "comma_lab.artifact_retention_cold_store_copy.v1",
            "method": "same_device_rename",
            "same_device_as_source": True,
            "source_device_id": source_device_id,
            "destination_device_id": destination_device_id,
            "source_digest": source_digest,
            "destination_digest": final_digest,
            "bytes_estimate": int(bytes_estimate),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    try:
        shutil.copytree(source, partial_destination, symlinks=False)
        destination_digest = directory_digest(partial_destination)
        if source_digest["sha256"] != destination_digest["sha256"]:
            raise ArtifactRetentionError("cold-store copy verification failed")
        partial_destination.replace(destination)
    except Exception:
        shutil.rmtree(partial_destination, ignore_errors=True)
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        raise
    final_digest = directory_digest(destination)
    if source_digest["sha256"] != final_digest["sha256"]:
        shutil.rmtree(destination, ignore_errors=True)
        raise ArtifactRetentionError("cold-store copy verification failed")
    _delete_certified_tree(
        source,
        repo_root=repo_root,
        bytes_estimate=bytes_estimate,
        allowed_source_roots=allowed_source_roots,
    )
    return {
        "schema": "comma_lab.artifact_retention_cold_store_copy.v1",
        "method": "copy_verify_delete",
        "same_device_as_source": False,
        "source_device_id": source_device_id,
        "destination_device_id": destination_device_id,
        "source_digest": source_digest,
        "destination_digest": final_digest,
        "bytes_estimate": int(bytes_estimate),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
