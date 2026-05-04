"""Fail-closed local PR86 HPAC replay and byte-parity helpers.

This module is deliberately local-only: it never runs contest eval, never
dispatches remote work, and never makes a score claim. It validates the public
PR86 archive contract, decodes its torch/PPMd payloads, and attempts to prove
that submitted ``tokens.bin`` can be decoded and re-encoded byte-for-byte with
``constriction.stream.queue.RangeEncoder.get_compressed().tobytes()``.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import io
import json
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PR86_DIR = REPO_ROOT / "experiments/results/public_pr86_intake_20260504_codex"
DEFAULT_PR86_ARCHIVE = DEFAULT_PR86_DIR / "archive.zip"
DEFAULT_PR86_PROBABILITY_CONTRACT_DIR = (
    REPO_ROOT / "experiments/results/pr86_hpac_probability_contract_20260504_worker"
)
DEFAULT_PR86_PROBABILITY_CONTRACT_REPORT = (
    DEFAULT_PR86_PROBABILITY_CONTRACT_DIR / "pr86_hpac_probability_contract_variants.json"
)
DEFAULT_PR86_MERGED_SOURCE_DIR = (
    REPO_ROOT / "experiments/results/public_pr86_intake_20260504_merged_refresh"
)
DEFAULT_MERGED_INTAKE_SUMMARY = DEFAULT_PR86_MERGED_SOURCE_DIR / "intake_summary.json"
DEFAULT_MERGED_SOURCE_MANIFEST = DEFAULT_PR86_MERGED_SOURCE_DIR / "source_manifest.json"
DEFAULT_MERGED_PR_API = DEFAULT_PR86_MERGED_SOURCE_DIR / "pr86_api.json"
DEFAULT_FULL_REENCODE = (
    DEFAULT_PR86_DIR / "pr86_hpac_full_decode_reencode_gate_20260504_codex.json"
)
DEFAULT_TOKEN_ANATOMY = DEFAULT_PR86_DIR / "pr86_hpac_token_anatomy_forensics.json"
DEFAULT_PR85_PROBE = DEFAULT_PR86_DIR / "pr86_hpac_pr85_qma9_parity_probe.json"
DEFAULT_PR_VIEW = DEFAULT_PR86_DIR / "pr86_view.json"

EXPECTED_PR86_ARCHIVE_BYTES = 207_579
EXPECTED_PR86_ARCHIVE_SHA256 = "e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef"
EXPECTED_PR86_TOKENS_SHA256 = "14144bde496631f89a02646496bc2e66306bba6da149ddca37e21d85d175f225"
EXPECTED_PR86_MEMBERS = (
    "master.pt.gz",
    "slave.pt.gz",
    "hpac.pt.ppmd",
    "tokens.bin",
    "meta.pt",
)
EXPECTED_PR86_MEMBER_BYTES = {
    "master.pt.gz": 31_144,
    "slave.pt.gz": 32_287,
    "hpac.pt.ppmd": 28_243,
    "tokens.bin": 113_900,
    "meta.pt": 1_499,
}
RECORDED_PR86_DEPENDENCIES = {
    "python": "3.12.13",
    "torch": "2.11.0",
    "numpy": "2.4.4",
    "constriction": "0.4.2",
    "pyppmd": "1.3.1",
}

NUM_CLASSES = 5
SEGNET_IN_H = 384
SEGNET_IN_W = 512
PPMD_MAX_ORDER = 4
PPMD_MEM_SIZE = 16 << 20
PROB_EPS = 1e-7
DEFAULT_HPAC_PROBABILITY_VARIANT = "source_float64_perfect_false"


class Pr86HpacReplayError(RuntimeError):
    """Raised when PR86 replay violates a fail-closed contract."""

    def __init__(self, stage: str, reason: str, **context: Any) -> None:
        super().__init__(reason)
        self.stage = stage
        self.reason = reason
        self.context = context


@dataclass(frozen=True)
class Pr86ArchiveContract:
    """Static archive contract used by the replay shim."""

    required_members: tuple[str, ...] = EXPECTED_PR86_MEMBERS
    expected_archive_bytes: int | None = EXPECTED_PR86_ARCHIVE_BYTES
    expected_archive_sha256: str | None = EXPECTED_PR86_ARCHIVE_SHA256
    expected_member_bytes: Mapping[str, int] | None = None
    expected_tokens_sha256: str | None = EXPECTED_PR86_TOKENS_SHA256
    require_zip_stored: bool = True

    def member_bytes(self) -> Mapping[str, int]:
        if self.expected_member_bytes is None:
            return EXPECTED_PR86_MEMBER_BYTES
        return self.expected_member_bytes


@dataclass(frozen=True)
class Pr86ArchiveBundle:
    """Validated PR86 archive members plus custody facts."""

    path: Path
    members: Mapping[str, bytes]
    report: Mapping[str, Any]


@dataclass(frozen=True)
class HpacProbabilityVariant:
    """Named probability/categorical contract for PR86 HPAC range coding."""

    name: str
    probability_dtype: str
    categorical_perfect: bool
    source_contract: bool
    description: str


HPAC_PROBABILITY_VARIANTS: Mapping[str, HpacProbabilityVariant] = {
    "source_float64_perfect_false": HpacProbabilityVariant(
        name="source_float64_perfect_false",
        probability_dtype="float64",
        categorical_perfect=False,
        source_contract=True,
        description=(
            "Merged PR86 source contract: clipped/renormalized numpy float64 "
            "probabilities with Categorical(..., perfect=False)."
        ),
    ),
    "source_float32_perfect_false": HpacProbabilityVariant(
        name="source_float32_perfect_false",
        probability_dtype="float32",
        categorical_perfect=False,
        source_contract=False,
        description=(
            "Off-contract probe: pass clipped/renormalized numpy float32 "
            "probabilities to Categorical(..., perfect=False)."
        ),
    ),
    "source_float64_perfect_true": HpacProbabilityVariant(
        name="source_float64_perfect_true",
        probability_dtype="float64",
        categorical_perfect=True,
        source_contract=False,
        description=(
            "Off-contract probe: keep float64 probabilities but construct "
            "Categorical(..., perfect=True)."
        ),
    ),
    "source_float32_perfect_true": HpacProbabilityVariant(
        name="source_float32_perfect_true",
        probability_dtype="float32",
        categorical_perfect=True,
        source_contract=False,
        description=(
            "Off-contract combined probe: pass float32 probabilities to "
            "Categorical(..., perfect=True)."
        ),
    ),
}


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def supported_hpac_probability_variant_names() -> tuple[str, ...]:
    """Return stable CLI choices for HPAC probability-contract probes."""

    return tuple(HPAC_PROBABILITY_VARIANTS.keys())


def resolve_hpac_probability_variant(variant: str | HpacProbabilityVariant) -> HpacProbabilityVariant:
    """Resolve a named HPAC probability variant or fail closed on unknown input."""

    if isinstance(variant, HpacProbabilityVariant):
        return variant
    try:
        return HPAC_PROBABILITY_VARIANTS[str(variant)]
    except KeyError as exc:
        raise Pr86HpacReplayError(
            "probability_variant_contract",
            "unknown_hpac_probability_variant",
            requested_variant=str(variant),
            supported_variants=list(supported_hpac_probability_variant_names()),
        ) from exc


def default_source_artifact_paths() -> tuple[Path, ...]:
    return (
        DEFAULT_MERGED_INTAKE_SUMMARY,
        DEFAULT_MERGED_SOURCE_MANIFEST,
        DEFAULT_MERGED_PR_API,
        DEFAULT_FULL_REENCODE,
        DEFAULT_TOKEN_ANATOMY,
        DEFAULT_PR85_PROBE,
        DEFAULT_PR_VIEW,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return repo_rel(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": sha256_bytes(value.detach().cpu().numpy().tobytes())
            if value.device.type == "cpu" and value.numel() <= 16_384
            else None,
        }
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_json_file(path: Path) -> tuple[Any, dict[str, Any]]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise Pr86HpacReplayError(
            "source_artifacts",
            "source_artifact_json_decode_failed",
            artifact=repo_rel(path),
            error=str(exc),
        ) from exc
    return payload, {
        "path": repo_rel(path),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }


def _validate_safe_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if not name or name.endswith("/") or path.is_absolute() or ".." in path.parts:
        raise Pr86HpacReplayError(
            "archive_member_contract",
            "unsafe_zip_member_name",
            member_name=name,
        )


def read_pr86_archive(
    archive: Path,
    *,
    contract: Pr86ArchiveContract = Pr86ArchiveContract(),
) -> Pr86ArchiveBundle:
    """Read a PR86 archive after strict member, identity, and zip-slip checks."""

    archive = Path(archive)
    if not archive.is_file():
        raise Pr86HpacReplayError("archive_custody", "archive_missing", archive=repo_rel(archive))

    archive_bytes = archive.stat().st_size
    archive_sha = sha256_path(archive)
    if contract.expected_archive_bytes is not None and archive_bytes != contract.expected_archive_bytes:
        raise Pr86HpacReplayError(
            "archive_custody",
            "archive_size_mismatch",
            archive=repo_rel(archive),
            expected_archive_bytes=contract.expected_archive_bytes,
            actual_archive_bytes=archive_bytes,
            actual_archive_sha256=archive_sha,
        )
    if contract.expected_archive_sha256 is not None and archive_sha != contract.expected_archive_sha256:
        raise Pr86HpacReplayError(
            "archive_custody",
            "archive_sha256_mismatch",
            archive=repo_rel(archive),
            expected_archive_sha256=contract.expected_archive_sha256,
            actual_archive_sha256=archive_sha,
            actual_archive_bytes=archive_bytes,
        )

    members: dict[str, bytes] = {}
    member_rows: list[dict[str, Any]] = []
    expected_names = set(contract.required_members)
    expected_member_bytes = contract.member_bytes()
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            infos = zf.infolist()
            names = [info.filename for info in infos]
            duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
            if duplicates:
                raise Pr86HpacReplayError(
                    "archive_member_contract",
                    "duplicate_zip_members",
                    duplicate_member_names=duplicates,
                )
            for name in names:
                _validate_safe_member_name(name)
            missing = [name for name in contract.required_members if name not in names]
            unexpected = [name for name in names if name not in expected_names]
            if missing or unexpected:
                raise Pr86HpacReplayError(
                    "archive_member_contract",
                    "archive_member_set_mismatch",
                    required_members=list(contract.required_members),
                    missing_members=missing,
                    unexpected_members=unexpected,
                )
            for info in infos:
                if info.is_dir():
                    raise Pr86HpacReplayError(
                        "archive_member_contract",
                        "directory_member_not_allowed",
                        member_name=info.filename,
                    )
                if contract.require_zip_stored and info.compress_type != zipfile.ZIP_STORED:
                    raise Pr86HpacReplayError(
                        "archive_member_contract",
                        "zip_member_not_stored",
                        member_name=info.filename,
                        zip_compress_type=int(info.compress_type),
                    )
                with zf.open(info, "r") as handle:
                    data = handle.read()
                expected_size = expected_member_bytes.get(info.filename)
                if expected_size is not None and len(data) != expected_size:
                    raise Pr86HpacReplayError(
                        "archive_member_contract",
                        "member_size_mismatch",
                        member_name=info.filename,
                        expected_member_bytes=expected_size,
                        actual_member_bytes=len(data),
                    )
                members[info.filename] = data
                member_rows.append(
                    {
                        "name": info.filename,
                        "file_size": int(info.file_size),
                        "compress_size": int(info.compress_size),
                        "zip_compress_type": int(info.compress_type),
                        "crc32_hex": f"{info.CRC:08x}",
                        "sha256": sha256_bytes(data),
                    }
                )
    except zipfile.BadZipFile as exc:
        raise Pr86HpacReplayError(
            "archive_custody",
            "bad_zip_file",
            archive=repo_rel(archive),
            error=str(exc),
        ) from exc

    report = {
        "path": repo_rel(archive),
        "size_bytes": archive_bytes,
        "sha256": archive_sha,
        "expected_size_bytes": contract.expected_archive_bytes,
        "expected_sha256": contract.expected_archive_sha256,
        "identity_matches_expected": (
            archive_bytes == contract.expected_archive_bytes
            and archive_sha == contract.expected_archive_sha256
        )
        if contract.expected_archive_bytes is not None and contract.expected_archive_sha256 is not None
        else None,
        "member_contract_status": "passed",
        "member_count": len(member_rows),
        "members": member_rows,
    }
    return Pr86ArchiveBundle(path=archive, members=members, report=report)


def load_source_artifact_summaries(paths: tuple[Path, ...]) -> dict[str, Any]:
    """Load default PR86 intake JSONs and keep only replay-relevant fields."""

    summaries: dict[str, Any] = {}
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise Pr86HpacReplayError(
                "source_artifacts",
                "source_artifact_missing",
                artifact=repo_rel(path),
            )
        payload, summary = _load_json_file(path)
        name = path.name
        if path == DEFAULT_MERGED_INTAKE_SUMMARY or name == DEFAULT_MERGED_INTAKE_SUMMARY.name:
            summary["merged"] = payload.get("merged")
            summary["state"] = payload.get("state")
            summary["head_sha"] = payload.get("head_sha")
            summary["merge_commit_sha"] = payload.get("merge_commit_sha")
            summary["merged_at"] = payload.get("merged_at")
            summary["archive_bytes"] = payload.get("archive_bytes")
            summary["archive_sha256"] = payload.get("archive_sha256")
            summary["patch_sha256"] = payload.get("patch_sha256")
        elif path == DEFAULT_MERGED_SOURCE_MANIFEST or name == DEFAULT_MERGED_SOURCE_MANIFEST.name:
            entries = payload if isinstance(payload, list) else []
            wanted = {
                "inflate.py",
                "training/archive.py",
                "training/hpac.py",
                "training/README.md",
                "README.md",
                "pr86.patch",
            }
            summary["entry_count"] = len(entries)
            summary["selected_entries"] = [
                entry for entry in entries if isinstance(entry, dict) and entry.get("path") in wanted
            ]
        elif path == DEFAULT_MERGED_PR_API or name == DEFAULT_MERGED_PR_API.name:
            summary["number"] = payload.get("number")
            summary["state"] = payload.get("state")
            summary["merged_at"] = payload.get("merged_at")
            summary["merge_commit_sha"] = payload.get("merge_commit_sha")
            summary["head_sha"] = (payload.get("head") or {}).get("sha")
            summary["title"] = payload.get("title")
            summary["url"] = payload.get("html_url")
        elif name == DEFAULT_FULL_REENCODE.name:
            summary["dependencies"] = payload.get("dependencies")
            summary["previous_full_decode_reencode_gate"] = payload.get("full_decode_reencode_gate")
            summary["constriction_queue_contract"] = payload.get("constriction_queue_contract")
            summary["token_semantics"] = payload.get("token_semantics")
        elif name == DEFAULT_TOKEN_ANATOMY.name:
            summary["token_hpac_decode_contract"] = payload.get("token_hpac_decode_contract")
        elif name == DEFAULT_PR85_PROBE.name:
            summary["status"] = payload.get("status")
            summary["failure_class"] = payload.get("failure_class")
            summary["observed_error"] = payload.get("observed_error")
            summary["pr86_dependencies"] = payload.get("pr86_dependencies")
        elif name == DEFAULT_PR_VIEW.name:
            summary["url"] = payload.get("url")
            summary["number"] = payload.get("number")
            summary["title"] = payload.get("title")
        else:
            summary["top_level_keys"] = sorted(payload) if isinstance(payload, dict) else None
        summaries[repo_rel(path)] = summary
    return summaries


def analyze_pr86_current_source_context(source_dir: Path = DEFAULT_PR86_MERGED_SOURCE_DIR) -> dict[str, Any]:
    """Summarize the current merged PR86 source contract used for HPAC replay."""

    source_dir = Path(source_dir)
    report: dict[str, Any] = {
        "source_dir": repo_rel(source_dir),
        "status": "missing",
        "score_claim": False,
        "dispatch_performed": False,
    }
    if not source_dir.is_dir():
        report["missing_reason"] = "source_dir_missing"
        return report

    intake_path = source_dir / "intake_summary.json"
    manifest_path = source_dir / "source_manifest.json"
    pr_api_path = source_dir / "pr86_api.json"
    try:
        intake, intake_summary = _load_json_file(intake_path)
        manifest, manifest_summary = _load_json_file(manifest_path)
        pr_api, pr_api_summary = _load_json_file(pr_api_path)
    except FileNotFoundError as exc:
        report["missing_reason"] = "required_source_context_file_missing"
        report["missing_path"] = repo_rel(Path(exc.filename)) if exc.filename else None
        return report

    manifest_entries = manifest if isinstance(manifest, list) else []
    selected_paths = ("inflate.py", "training/archive.py", "training/hpac.py", "training/README.md")
    manifest_by_path = {
        str(entry.get("path")): entry
        for entry in manifest_entries
        if isinstance(entry, Mapping) and entry.get("path") in selected_paths
    }

    def _read_source(rel: str) -> str:
        path = source_dir / rel
        if not path.is_file():
            raise Pr86HpacReplayError(
                "source_context",
                "current_source_file_missing",
                source_file=repo_rel(path),
            )
        return path.read_text(encoding="utf-8")

    archive_text = _read_source("training/archive.py")
    hpac_text = _read_source("training/hpac.py")
    inflate_text = _read_source("inflate.py")
    readme_text = _read_source("training/README.md")

    raw_archive_call = "encode_frame(gen, tokens_t[f:f + 1]" in archive_text
    archive_mentions_residuals = "residual" in archive_text.lower()
    inflate_reconstructs_residuals = "residual" in inflate_text.lower() or "(decoded +" in inflate_text
    training_residual = (
        "def compute_residuals" in hpac_text
        and "(gt_tokens[1:] - gt_tokens[:-1]) % NUM_CLASSES" in hpac_text
    )
    submitted_encoding = "unknown"
    if raw_archive_call and not inflate_reconstructs_residuals:
        submitted_encoding = "raw_tokens"
    elif training_residual:
        submitted_encoding = "residual_tokens_unproven_for_archive"

    archive_probability_dtype = "float64" if ".astype(np.float64)" in hpac_text else "unknown"
    inflate_probability_dtype = "float64" if ".astype(np.float64)" in inflate_text else "unknown"
    archive_perfect_false = "Categorical(probabilities=probs_np[i], perfect=False)" in hpac_text
    inflate_perfect_false = "Categorical(probabilities=probs_np[i], perfect=False)" in inflate_text
    explicit_grid = "16384" in hpac_text or "16384" in inflate_text

    main_device_cuda = '"cuda" if torch.cuda.is_available() else "cpu"' in inflate_text
    hpac_call_passes_main_device = "hpac_path, P, delta, ch, device, use_spm, hpac_d_film" in inflate_text
    comment_claims_cpu = "Force HPAC decode onto CPU" in inflate_text

    identical_to_stale: dict[str, bool | None] = {}
    for rel in ("inflate.py", "training/archive.py", "training/hpac.py"):
        current = source_dir / rel
        stale = DEFAULT_PR86_DIR / rel
        identical_to_stale[rel] = (
            sha256_path(current) == sha256_path(stale) if current.is_file() and stale.is_file() else None
        )

    head_sha = intake.get("head_sha") if isinstance(intake, Mapping) else None
    merged = bool(intake.get("merged")) if isinstance(intake, Mapping) else False
    current_status = (
        "current_merged_source"
        if merged and head_sha == "0eabe354f09b7490fd1cbb2b05a9102ab528d4d4"
        else "source_context_needs_review"
    )
    report.update(
        {
            "status": current_status,
            "intake_summary": {
                "path": intake_summary["path"],
                "sha256": intake_summary["sha256"],
                "merged": merged,
                "state": intake.get("state") if isinstance(intake, Mapping) else None,
                "merged_at": intake.get("merged_at") if isinstance(intake, Mapping) else None,
                "head_sha": head_sha,
                "merge_commit_sha": intake.get("merge_commit_sha") if isinstance(intake, Mapping) else None,
                "archive_bytes": intake.get("archive_bytes") if isinstance(intake, Mapping) else None,
                "archive_sha256": intake.get("archive_sha256") if isinstance(intake, Mapping) else None,
                "archive_identity_matches_cached_bytes": (
                    intake.get("archive_bytes") == EXPECTED_PR86_ARCHIVE_BYTES
                    and intake.get("archive_sha256") == EXPECTED_PR86_ARCHIVE_SHA256
                )
                if isinstance(intake, Mapping)
                else None,
            },
            "pr_api": {
                "path": pr_api_summary["path"],
                "sha256": pr_api_summary["sha256"],
                "number": pr_api.get("number") if isinstance(pr_api, Mapping) else None,
                "state": pr_api.get("state") if isinstance(pr_api, Mapping) else None,
                "merged_at": pr_api.get("merged_at") if isinstance(pr_api, Mapping) else None,
                "head_sha": (pr_api.get("head") or {}).get("sha") if isinstance(pr_api, Mapping) else None,
                "merge_commit_sha": pr_api.get("merge_commit_sha") if isinstance(pr_api, Mapping) else None,
                "url": pr_api.get("html_url") if isinstance(pr_api, Mapping) else None,
            },
            "source_manifest": {
                "path": manifest_summary["path"],
                "sha256": manifest_summary["sha256"],
                "entry_count": len(manifest_entries),
                "selected_entries": [manifest_by_path.get(path) for path in selected_paths],
            },
            "source_files_identical_to_stale_cache": identical_to_stale,
            "token_semantics": {
                "training_objective": "residual_tokens" if training_residual else "unknown",
                "training_residual_definition_present": training_residual,
                "archive_encode_frame_raw_tokens_call_present": raw_archive_call,
                "archive_mentions_residuals": archive_mentions_residuals,
                "inflate_reconstructs_residuals": inflate_reconstructs_residuals,
                "submitted_archive_token_encoding": submitted_encoding,
                "frame0_raw_equals_residual_note": (
                    "Frame 0 cannot distinguish raw from residual semantics because residual[0] == token[0]."
                ),
            },
            "probability_model_contract": {
                "archive_probability_numpy_dtype": archive_probability_dtype,
                "inflate_probability_numpy_dtype": inflate_probability_dtype,
                "archive_categorical_perfect_false": archive_perfect_false,
                "inflate_categorical_perfect_false": inflate_perfect_false,
                "explicit_16384_grid_in_archive_or_inflate": explicit_grid,
                "readme_mentions_16384_grid": "1/16384" in readme_text,
                "probability_clip_eps": "1e-7",
                "contract_summary": (
                    "current merged source uses clipped/renormalized numpy float64 probabilities "
                    "with constriction Categorical(..., perfect=False)"
                ),
            },
            "inflate_device_contract": {
                "main_device_expression_selects_cuda_when_available": main_device_cuda,
                "comment_claims_hpac_cpu": comment_claims_cpu,
                "submitted_code_passes_main_device_to_hpac": hpac_call_passes_main_device,
                "comment_code_mismatch": bool(comment_claims_cpu and hpac_call_passes_main_device),
            },
        }
    )
    return report


def collect_dependency_report(
    *,
    expected_versions: Mapping[str, str] = RECORDED_PR86_DEPENDENCIES,
    require_behavior_self_test: bool = True,
) -> dict[str, Any]:
    """Record installed replay dependency versions and constriction behavior."""

    installed = {
        "python": sys.version.split()[0],
        "torch": _package_version("torch"),
        "numpy": _package_version("numpy"),
        "constriction": _package_version("constriction"),
        "pyppmd": _package_version("pyppmd"),
    }
    missing = [name for name, version in installed.items() if name != "python" and version is None]
    version_matches = {
        name: installed.get(name) == expected
        for name, expected in expected_versions.items()
        if name in installed
    }
    report: dict[str, Any] = {
        "expected_versions": dict(expected_versions),
        "installed_versions": installed,
        "version_matches_recorded": version_matches,
        "missing_required_packages": missing,
        "version_drift": [
            {
                "package": name,
                "expected": expected,
                "installed": installed.get(name),
            }
            for name, expected in expected_versions.items()
            if name in installed and installed.get(name) != expected
        ],
        "behavior_self_test": None,
        "status": "unknown",
    }
    if missing:
        report["status"] = "failed_closed_missing_dependency"
        return report

    try:
        import constriction

        symbols = np.array([(idx * 2 + 3) % NUM_CLASSES for idx in range(257)], dtype=np.int32)
        probs = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES, dtype=np.float64)
        model = constriction.stream.model.Categorical(probabilities=probs, perfect=False)
        encoder = constriction.stream.queue.RangeEncoder()
        for symbol in symbols:
            encoder.encode(int(symbol), model)
        compressed = encoder.get_compressed()
        compressed_bytes = compressed.tobytes()
        decoder = constriction.stream.queue.RangeDecoder(compressed)
        decoded = np.array([decoder.decode(model) for _ in range(len(symbols))], dtype=np.int32)
        report["behavior_self_test"] = {
            "queue_api": "constriction.stream.queue.RangeEncoder/RangeDecoder",
            "model_api": "constriction.stream.model.Categorical(..., perfect=False)",
            "get_compressed_tobytes_available": hasattr(compressed, "tobytes"),
            "compressed_dtype": str(compressed.dtype),
            "compressed_word_count": int(compressed.size),
            "compressed_sha256": sha256_bytes(compressed_bytes),
            "encoded_symbols_sha256": sha256_bytes(symbols.tobytes()),
            "decoded_symbols_sha256": sha256_bytes(decoded.tobytes()),
            "same_order_roundtrip_ok": bool(np.array_equal(symbols, decoded)),
        }
    except Exception as exc:  # pragma: no cover - exact exception is dependency-specific.
        report["behavior_self_test"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    self_test = report.get("behavior_self_test") or {}
    if require_behavior_self_test and not self_test.get("same_order_roundtrip_ok", False):
        report["status"] = "failed_closed_behavior_mismatch"
    else:
        report["status"] = (
            "passed_with_version_drift" if report["version_drift"] else "passed"
        )
    return report


def _torch_load_bytes(raw: bytes, *, map_location: str = "cpu") -> Any:
    return torch.load(io.BytesIO(raw), map_location=map_location, weights_only=False)


def decode_gzip_torch_member(data: bytes, *, member_name: str) -> tuple[Any, dict[str, Any]]:
    """Gzip-decompress and torch-load a PR86 state-dict member."""

    try:
        raw = gzip.decompress(data)
    except OSError as exc:
        raise Pr86HpacReplayError(
            f"decode_{member_name}",
            "gzip_decode_failed",
            member_name=member_name,
            error=str(exc),
        ) from exc
    try:
        payload = _torch_load_bytes(raw, map_location="cpu")
    except Exception as exc:
        raise Pr86HpacReplayError(
            f"decode_{member_name}",
            "torch_load_failed",
            member_name=member_name,
            error_type=type(exc).__name__,
            error=str(exc),
        ) from exc
    report = _torch_payload_report(payload)
    report.update(
        {
            "member_name": member_name,
            "compressed_bytes": len(data),
            "compressed_sha256": sha256_bytes(data),
            "decompressed_bytes": len(raw),
            "decompressed_sha256": sha256_bytes(raw),
            "status": "passed",
        }
    )
    return payload, report


def decode_meta_member(data: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        payload = _torch_load_bytes(data, map_location="cpu")
    except Exception as exc:
        raise Pr86HpacReplayError(
            "decode_meta_pt",
            "torch_load_failed",
            member_name="meta.pt",
            error_type=type(exc).__name__,
            error=str(exc),
        ) from exc
    if not isinstance(payload, dict):
        raise Pr86HpacReplayError(
            "decode_meta_pt",
            "meta_payload_not_dict",
            payload_type=type(payload).__name__,
        )
    required = ("N", "P", "delta", "ch")
    missing = [key for key in required if key not in payload]
    if missing:
        raise Pr86HpacReplayError("decode_meta_pt", "meta_missing_required_keys", missing_keys=missing)
    report = _torch_payload_report(payload)
    report.update(
        {
            "member_name": "meta.pt",
            "bytes": len(data),
            "sha256": sha256_bytes(data),
            "status": "passed",
            "selected_config": {
                "N": int(payload["N"]),
                "P": int(payload["P"]),
                "delta": int(payload["delta"]),
                "ch": int(payload["ch"]),
                "hpac_d_film": int(payload.get("hpac_d_film", 32)),
                "use_spm": bool(payload.get("use_spm", False)),
                "mode": payload.get("mode"),
                "tokens_bpp": float(payload["tokens_bpp"]) if "tokens_bpp" in payload else None,
            },
        }
    )
    return payload, report


def _torch_payload_report(payload: Any) -> dict[str, Any]:
    report: dict[str, Any] = {"payload_type": type(payload).__name__}
    if isinstance(payload, Mapping):
        keys = sorted(str(key) for key in payload.keys())
        report["key_count"] = len(keys)
        report["key_prefix"] = keys[:20]
    return report


def _patch_group_mask(k: int, delta: int, type_: str) -> torch.Tensor:
    mask = torch.zeros(k, k, dtype=torch.float32)
    center = (k - 1) // 2
    for dr_idx in range(k):
        for dc_idx in range(k):
            dr = dr_idx - center
            dc = dc_idx - center
            val = dc + delta * dr
            if type_ == "A":
                if val < 0:
                    mask[dr_idx, dc_idx] = 1.0
            elif val <= 0:
                mask[dr_idx, dc_idx] = 1.0
    return mask


class _MaskedConv2dPG(nn.Module):
    """Inflate-time masked conv for PR86 HPAC weights."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int,
        *,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        type_: str = "B",
        delta: int = 2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(c_out, c_in // groups, k, k))
        self.bias = nn.Parameter(torch.zeros(c_out)) if bias else None
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.register_buffer(
            "mask",
            _patch_group_mask(k, delta, type_).view(1, 1, k, k),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.weight * self.mask,
            self.bias,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class _ChannelNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_channels))
        self.shift = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)


class _CausalSPM(nn.Module):
    def __init__(self, ch: int, P: int = 32) -> None:
        super().__init__()
        self.P = P
        self.norm = _ChannelNorm2d(ch)
        self.dw = nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch)
        self.pw = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, h_past: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = h_past.shape
        patch = self.P
        n_row_patches, n_col_patches = height // patch, width // patch
        x_p = h_past.view(bsz, channels, n_row_patches, patch, n_col_patches, patch).mean(dim=(3, 5))
        x_p = self.norm(x_p)
        x_p = self.dw(x_p)
        x_p = F.gelu(x_p)
        x_p = self.pw(x_p)
        x_full = x_p.unsqueeze(3).unsqueeze(5).expand(
            bsz, channels, n_row_patches, patch, n_col_patches, patch
        )
        return x_full.contiguous().view(bsz, channels, n_row_patches * patch, n_col_patches * patch)


class HPACMini(nn.Module):
    """PR86 inflate-time HPAC entropy model."""

    def __init__(
        self,
        *,
        num_pairs: int = 600,
        num_classes: int = NUM_CLASSES,
        P: int = 32,
        delta: int = 2,
        d_film: int = 32,
        ch: int = 64,
        use_spm: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.P = P
        self.delta = delta
        self.ch = ch
        self.use_spm = use_spm
        self.frame_embed = nn.Embedding(num_pairs, d_film)
        self.film_gen = nn.Linear(d_film, ch * 2)
        self.conv_a = _MaskedConv2dPG(num_classes + 2, ch, k=7, padding=3, type_="A", delta=delta)
        self.gn_a = _ChannelNorm2d(ch)
        self.conv_b1 = _MaskedConv2dPG(ch, ch, k=5, padding=4, dilation=2, groups=ch, type_="B", delta=delta)
        self.gn_b1 = _ChannelNorm2d(ch)
        self.conv_b2 = _MaskedConv2dPG(ch, ch, k=3, padding=4, dilation=4, groups=ch, type_="B", delta=delta)
        self.gn_b2 = _ChannelNorm2d(ch)
        self.conv_past = nn.Conv2d(num_classes, ch, kernel_size=3, padding=1)
        self.spm = _CausalSPM(ch, P=P) if use_spm else None
        self.head = nn.Conv2d(ch, num_classes, kernel_size=1, padding=0)
        self.register_buffer("_coord_cache", torch.zeros(0), persistent=False)
        self._cached_P = -1

    def _patch_coord_grid(self, batch_patches: int, device: torch.device) -> torch.Tensor:
        if self._cached_P != self.P or self._coord_cache.numel() == 0:
            patch = self.P
            ys = torch.linspace(-1.0, 1.0, patch, device=device).view(1, 1, patch, 1).expand(
                1, 1, patch, patch
            )
            xs = torch.linspace(-1.0, 1.0, patch, device=device).view(1, 1, 1, patch).expand(
                1, 1, patch, patch
            )
            self._coord_cache = torch.cat([ys, xs], dim=1)
            self._cached_P = self.P
        return self._coord_cache.expand(batch_patches, -1, -1, -1)

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        patch = self.P
        n_row_patches, n_col_patches = height // patch, width // patch
        x = x.view(bsz, channels, n_row_patches, patch, n_col_patches, patch)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(bsz * n_row_patches * n_col_patches, channels, patch, patch)

    def _from_patches(self, x_p: torch.Tensor, batch: int, n_row_patches: int, n_col_patches: int) -> torch.Tensor:
        patch = self.P
        channels = x_p.shape[1]
        x_p = x_p.view(batch, n_row_patches, n_col_patches, channels, patch, patch)
        x_p = x_p.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x_p.view(batch, channels, n_row_patches * patch, n_col_patches * patch)

    def forward(self, tokens: torch.Tensor, idx: torch.Tensor, prev_tokens: torch.Tensor) -> torch.Tensor:
        batch, height, width = tokens.shape
        patch = self.P
        n_row_patches, n_col_patches = height // patch, width // patch
        n_patches = n_row_patches * n_col_patches

        x = F.one_hot(tokens, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        x_p = self._to_patches(x)
        coord_p = self._patch_coord_grid(batch * n_patches, x.device)
        h_p = self.gn_a(self.conv_a(torch.cat([x_p, coord_p], dim=1)))

        emb = self.frame_embed(idx)
        film = self.film_gen(emb)
        scale, shift = film.chunk(2, dim=1)
        scale_p = scale.view(batch, 1, self.ch, 1, 1).expand(batch, n_patches, self.ch, 1, 1)
        shift_p = shift.view(batch, 1, self.ch, 1, 1).expand(batch, n_patches, self.ch, 1, 1)
        h_p = h_p * (1.0 + scale_p.reshape(batch * n_patches, self.ch, 1, 1))
        h_p = h_p + shift_p.reshape(batch * n_patches, self.ch, 1, 1)
        h_p = F.gelu(h_p)

        x_prev = F.one_hot(prev_tokens, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        h_past_full = self.conv_past(x_prev)
        h_p = h_p + self._to_patches(h_past_full)
        if self.spm is not None:
            h_p = h_p + self._to_patches(self.spm(h_past_full))
        h_p = F.gelu(self.gn_b1(self.conv_b1(h_p)))
        h_p = F.gelu(self.gn_b2(self.conv_b2(h_p)))
        logits_p = self.head(h_p)
        return self._from_patches(logits_p, batch, n_row_patches, n_col_patches)


def reconstruct_hpac_state_dict(packed_sd: Mapping[str, Any], *, device: str = "cpu") -> dict[str, Any]:
    """Rehydrate PR86 INT8-packed HPAC weights to FP32 state-dict entries."""

    out: dict[str, Any] = {}
    bases = sorted(key[: -len(".weight_q")] for key in packed_sd if str(key).endswith(".weight_q"))
    for base in bases:
        q = packed_sd[base + ".weight_q"].to(device).float()
        scale = packed_sd[base + ".weight_scale"].to(device).float()
        shape = [1] * q.ndim
        shape[0] = -1
        out[base + ".weight"] = (q * scale.view(*shape)).to(torch.float32)
    skip = {base + suffix for base in bases for suffix in (".weight_q", ".weight_scale")}
    for key, value in packed_sd.items():
        if key in skip:
            continue
        if torch.is_tensor(value):
            out[str(key)] = value.to(device).float() if torch.is_floating_point(value) else value.to(device)
        else:
            out[str(key)] = value
    return out


def load_hpac_model_from_ppmd(
    data: bytes,
    *,
    config: Mapping[str, Any],
    device: str = "cpu",
) -> tuple[HPACMini, dict[str, Any]]:
    """Decode ``hpac.pt.ppmd`` and load a PR86 HPACMini entropy model."""

    if device != "cpu":
        raise Pr86HpacReplayError("device_contract", "pr86_hpac_replay_is_cpu_only", requested_device=device)
    try:
        import pyppmd
    except ImportError as exc:  # pragma: no cover - dependency gate catches this first.
        raise Pr86HpacReplayError("dependency_contract", "missing_pyppmd") from exc
    try:
        raw = pyppmd.decompress(data, max_order=PPMD_MAX_ORDER, mem_size=PPMD_MEM_SIZE)
    except Exception as exc:
        raise Pr86HpacReplayError(
            "decode_hpac_pt_ppmd",
            "ppmd_decompress_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        ) from exc
    try:
        packed_sd = _torch_load_bytes(raw, map_location="cpu")
    except Exception as exc:
        raise Pr86HpacReplayError(
            "decode_hpac_pt_ppmd",
            "torch_load_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        ) from exc
    if not isinstance(packed_sd, Mapping):
        raise Pr86HpacReplayError(
            "decode_hpac_pt_ppmd",
            "hpac_payload_not_state_dict",
            payload_type=type(packed_sd).__name__,
        )

    n_frames = int(config.get("N", 600))
    patch = int(config.get("P", 32))
    delta = int(config.get("delta", 2))
    ch = int(config.get("ch", 64))
    d_film = int(config.get("hpac_d_film", config.get("d_film", 32)))
    use_spm = bool(config.get("use_spm", False))
    sd = reconstruct_hpac_state_dict(packed_sd, device=device)
    gen = HPACMini(
        num_pairs=n_frames,
        num_classes=NUM_CLASSES,
        P=patch,
        delta=delta,
        ch=ch,
        d_film=d_film,
        use_spm=use_spm,
    ).to(device).eval()
    incompatible = gen.load_state_dict(sd, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise Pr86HpacReplayError(
            "decode_hpac_pt_ppmd",
            "hpac_state_dict_key_mismatch",
            missing_keys=missing,
            unexpected_keys=unexpected,
        )

    return gen, {
        "member_name": "hpac.pt.ppmd",
        "compressed_bytes": len(data),
        "compressed_sha256": sha256_bytes(data),
        "decompressed_bytes": len(raw),
        "decompressed_sha256": sha256_bytes(raw),
        "packed_state_key_count": len(packed_sd),
        "reconstructed_state_key_count": len(sd),
        "config": {
            "N": n_frames,
            "P": patch,
            "delta": delta,
            "ch": ch,
            "hpac_d_film": d_film,
            "use_spm": use_spm,
        },
        "ppmd_max_order": PPMD_MAX_ORDER,
        "ppmd_mem_size": PPMD_MEM_SIZE,
        "load_state_dict_strict": False,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "status": "passed",
    }


def _patch_group_grid(P: int, delta: int, device: torch.device) -> torch.Tensor:
    rows = torch.arange(P, device=device).view(P, 1).expand(P, P)
    cols = torch.arange(P, device=device).view(1, P).expand(P, P)
    return cols + delta * rows


def _full_mask_for_group(s_grid: torch.Tensor, group: int, n_row_patches: int, n_col_patches: int) -> torch.Tensor:
    patch = s_grid.shape[0]
    mask_p = s_grid == group
    full = mask_p.unsqueeze(0).unsqueeze(0).expand(n_row_patches, n_col_patches, patch, patch)
    return full.permute(0, 2, 1, 3).reshape(n_row_patches * patch, n_col_patches * patch)


def _group_masks(H: int, W: int, P: int, delta: int, device: torch.device) -> list[torch.Tensor | None]:
    if H % P != 0 or W % P != 0:
        raise Pr86HpacReplayError(
            "hpac_geometry_contract",
            "frame_dimensions_not_divisible_by_patch",
            height=H,
            width=W,
            P=P,
        )
    n_row_patches, n_col_patches = H // P, W // P
    s_grid = _patch_group_grid(P, delta, device)
    masks: list[torch.Tensor | None] = []
    for group in range(int((1 + delta) * P - delta)):
        mask = _full_mask_for_group(s_grid, group, n_row_patches, n_col_patches)
        masks.append(mask if bool(mask.any().item()) else None)
    return masks


def _normalize_probability_row(
    probs: np.ndarray,
    *,
    prob_eps: float,
    variant: HpacProbabilityVariant,
) -> np.ndarray:
    dtype = np.float32 if variant.probability_dtype == "float32" else np.float64
    clipped = np.clip(probs.astype(dtype, copy=False), dtype(prob_eps), dtype(1.0))
    clipped = clipped / clipped.sum()
    return clipped.astype(dtype, copy=False)


def _categorical_from_probs(
    probs: np.ndarray,
    *,
    prob_eps: float,
    variant: HpacProbabilityVariant,
):
    import constriction

    clipped = _normalize_probability_row(probs, prob_eps=prob_eps, variant=variant)
    return constriction.stream.model.Categorical(
        probabilities=clipped,
        perfect=variant.categorical_perfect,
    )


@torch.no_grad()
def decode_tokens_hpac(
    gen: HPACMini,
    token_blob: bytes,
    *,
    N: int,
    H: int,
    W: int,
    P: int,
    delta: int,
    device: str = "cpu",
    prob_eps: float = PROB_EPS,
    probability_variant: str | HpacProbabilityVariant = DEFAULT_HPAC_PROBABILITY_VARIANT,
    max_frames: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode PR86 HPAC tokens and report exact failure coordinates on error."""

    if device != "cpu":
        raise Pr86HpacReplayError("device_contract", "pr86_hpac_replay_is_cpu_only", requested_device=device)
    if len(token_blob) % 4 != 0:
        raise Pr86HpacReplayError(
            "tokens_bin_contract",
            "tokens_bin_not_uint32_aligned",
            tokens_bytes=len(token_blob),
        )
    try:
        import constriction
    except ImportError as exc:  # pragma: no cover - dependency gate catches this first.
        raise Pr86HpacReplayError("dependency_contract", "missing_constriction") from exc

    variant = resolve_hpac_probability_variant(probability_variant)
    frame_count = min(int(N), int(max_frames)) if max_frames is not None else int(N)
    dev = torch.device(device)
    gen = gen.to(dev).eval()
    masks = _group_masks(H, W, P, delta, dev)
    words = np.frombuffer(token_blob, dtype="<u4").astype(np.uint32, copy=False)
    decoder = constriction.stream.queue.RangeDecoder(words)
    tokens = np.empty((frame_count, H, W), dtype=np.uint8)
    decoded_prev = torch.zeros((1, H, W), dtype=torch.long, device=dev)
    decoded_symbols = 0
    started_at = time.time()
    for frame in range(frame_count):
        idx = torch.tensor([frame], dtype=torch.long, device=dev)
        cur = torch.zeros((1, H, W), dtype=torch.long, device=dev)
        frame_start_symbols = decoded_symbols
        for group, mask in enumerate(masks):
            if mask is None:
                continue
            group_start_symbols = decoded_symbols
            logits = gen(cur, idx, decoded_prev)
            probs = F.softmax(logits.float(), dim=1)
            probs_at_group = probs[0][:, mask].permute(1, 0).contiguous()
            probs_np = probs_at_group.cpu().numpy()
            decoded = np.empty(probs_np.shape[0], dtype=np.int64)
            for symbol_in_group, row in enumerate(probs_np):
                try:
                    decoded[symbol_in_group] = decoder.decode(
                        _categorical_from_probs(row, prob_eps=prob_eps, variant=variant)
                    )
                except Exception as exc:
                    raise Pr86HpacReplayError(
                        "submitted_tokens_decode",
                        "hpac_entropy_decode_contract_mismatch",
                        error_type=type(exc).__name__,
                        error=str(exc),
                        failed_at={
                            "frame": frame,
                            "group": group,
                            "symbol_in_group": symbol_in_group,
                        },
                        decoded_symbol_count_before_failure_group=group_start_symbols,
                        decoded_symbol_count_before_failure=decoded_symbols,
                        decoded_symbol_count_before_frame=frame_start_symbols,
                        source_tokens_bytes=len(token_blob),
                        source_tokens_sha256=sha256_bytes(token_blob),
                        probability_variant=_jsonable(variant.__dict__),
                    ) from exc
                decoded_symbols += 1
            cur[0, mask] = torch.from_numpy(decoded).to(dev)
        tokens[frame] = cur[0].cpu().numpy().astype(np.uint8)
        decoded_prev = cur.clone()
    report = {
        "status": "passed",
        "frames_decoded": frame_count,
        "requested_frames": int(N),
        "height": H,
        "width": W,
        "P": P,
        "delta": delta,
        "group_count": len(masks),
        "decoded_symbol_count": decoded_symbols,
        "elapsed_sec": round(time.time() - started_at, 3),
        "tokens_sha256": sha256_bytes(tokens.tobytes(order="C")),
        "probability_variant": _jsonable(variant.__dict__),
    }
    return tokens, report


@torch.no_grad()
def encode_tokens_hpac(
    gen: HPACMini,
    tokens: np.ndarray,
    *,
    P: int,
    delta: int,
    device: str = "cpu",
    prob_eps: float = PROB_EPS,
    probability_variant: str | HpacProbabilityVariant = DEFAULT_HPAC_PROBABILITY_VARIANT,
) -> tuple[bytes, dict[str, Any]]:
    """Encode decoded HPAC tokens using the public PR86 queue range-coder contract."""

    if device != "cpu":
        raise Pr86HpacReplayError("device_contract", "pr86_hpac_replay_is_cpu_only", requested_device=device)
    try:
        import constriction
    except ImportError as exc:  # pragma: no cover - dependency gate catches this first.
        raise Pr86HpacReplayError("dependency_contract", "missing_constriction") from exc
    if tokens.ndim != 3:
        raise Pr86HpacReplayError("reencode_tokens", "tokens_must_be_nhw", shape=list(tokens.shape))

    variant = resolve_hpac_probability_variant(probability_variant)
    dev = torch.device(device)
    gen = gen.to(dev).eval()
    frame_count, height, width = (int(v) for v in tokens.shape)
    masks = _group_masks(height, width, P, delta, dev)
    tokens_t = torch.from_numpy(tokens.astype(np.int64, copy=False)).to(dev)
    prev_all = torch.zeros_like(tokens_t)
    if frame_count > 1:
        prev_all[1:] = tokens_t[:-1]
    encoder = constriction.stream.queue.RangeEncoder()
    encoded_symbols = 0
    started_at = time.time()
    for frame in range(frame_count):
        idx = torch.tensor([frame], dtype=torch.long, device=dev)
        gt_tokens = tokens_t[frame : frame + 1]
        prev_tokens = prev_all[frame : frame + 1]
        current = torch.zeros_like(gt_tokens)
        for mask in masks:
            if mask is None:
                continue
            logits = gen(current, idx, prev_tokens)
            probs = F.softmax(logits.float(), dim=1)
            probs_at_group = probs[0][:, mask].permute(1, 0).contiguous()
            gt_at_group = gt_tokens[0][mask].cpu().numpy().astype(np.int32)
            probs_np = probs_at_group.cpu().numpy()
            for symbol, row in zip(gt_at_group, probs_np, strict=True):
                encoder.encode(
                    int(symbol),
                    _categorical_from_probs(row, prob_eps=prob_eps, variant=variant),
                )
                encoded_symbols += 1
            current[0, mask] = gt_tokens[0, mask]
    compressed = encoder.get_compressed()
    blob = compressed.tobytes()
    report = {
        "status": "passed",
        "frames_encoded": frame_count,
        "height": height,
        "width": width,
        "P": P,
        "delta": delta,
        "encoded_symbol_count": encoded_symbols,
        "compressed_dtype": str(compressed.dtype),
        "compressed_word_count": int(compressed.size),
        "compressed_bytes": len(blob),
        "compressed_sha256": sha256_bytes(blob),
        "elapsed_sec": round(time.time() - started_at, 3),
        "probability_variant": _jsonable(variant.__dict__),
    }
    return blob, report


@torch.no_grad()
def decode_symbols_hpac_with_prev_context(
    gen: HPACMini,
    token_blob: bytes,
    prev_context_tokens: np.ndarray,
    *,
    P: int,
    delta: int,
    device: str = "cpu",
    prob_eps: float = PROB_EPS,
    probability_variant: str | HpacProbabilityVariant = DEFAULT_HPAC_PROBABILITY_VARIANT,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode HPAC symbols while conditioning on a separate prev-token stream."""

    if device != "cpu":
        raise Pr86HpacReplayError("device_contract", "pr86_hpac_replay_is_cpu_only", requested_device=device)
    if prev_context_tokens.ndim != 3:
        raise Pr86HpacReplayError(
            "decode_symbols",
            "prev_context_must_be_nhw",
            shape=list(prev_context_tokens.shape),
        )
    if len(token_blob) % 4 != 0:
        raise Pr86HpacReplayError(
            "tokens_bin_contract",
            "tokens_bin_not_uint32_aligned",
            tokens_bytes=len(token_blob),
        )
    try:
        import constriction
    except ImportError as exc:  # pragma: no cover - dependency gate catches this first.
        raise Pr86HpacReplayError("dependency_contract", "missing_constriction") from exc

    variant = resolve_hpac_probability_variant(probability_variant)
    dev = torch.device(device)
    gen = gen.to(dev).eval()
    frame_count, height, width = (int(v) for v in prev_context_tokens.shape)
    masks = _group_masks(height, width, P, delta, dev)
    prev_t = torch.from_numpy(prev_context_tokens.astype(np.int64, copy=False)).to(dev)
    decoder = constriction.stream.queue.RangeDecoder(
        np.frombuffer(token_blob, dtype="<u4").astype(np.uint32, copy=False)
    )
    symbols = np.empty((frame_count, height, width), dtype=np.uint8)
    decoded_symbols = 0
    started_at = time.time()
    for frame in range(frame_count):
        idx = torch.tensor([frame], dtype=torch.long, device=dev)
        prev_tokens = prev_t[frame : frame + 1]
        current_symbols = torch.zeros((1, height, width), dtype=torch.long, device=dev)
        for mask in masks:
            if mask is None:
                continue
            logits = gen(current_symbols, idx, prev_tokens)
            probs = F.softmax(logits.float(), dim=1)
            probs_at_group = probs[0][:, mask].permute(1, 0).contiguous()
            probs_np = probs_at_group.cpu().numpy()
            decoded = np.empty(probs_np.shape[0], dtype=np.int64)
            for symbol_in_group, row in enumerate(probs_np):
                try:
                    decoded[symbol_in_group] = decoder.decode(
                        _categorical_from_probs(row, prob_eps=prob_eps, variant=variant)
                    )
                except Exception as exc:
                    raise Pr86HpacReplayError(
                        "submitted_symbols_decode",
                        "hpac_entropy_decode_contract_mismatch",
                        error_type=type(exc).__name__,
                        error=str(exc),
                        failed_at={"frame": frame, "symbol_in_group": symbol_in_group},
                        decoded_symbol_count_before_failure=decoded_symbols,
                        source_tokens_bytes=len(token_blob),
                        source_tokens_sha256=sha256_bytes(token_blob),
                        probability_variant=_jsonable(variant.__dict__),
                    ) from exc
                decoded_symbols += 1
            current_symbols[0, mask] = torch.from_numpy(decoded).to(dev)
        symbols[frame] = current_symbols[0].cpu().numpy().astype(np.uint8)
    return symbols, {
        "status": "passed",
        "frames_decoded": frame_count,
        "height": height,
        "width": width,
        "P": P,
        "delta": delta,
        "decoded_symbol_count": decoded_symbols,
        "elapsed_sec": round(time.time() - started_at, 3),
        "symbols_sha256": sha256_bytes(symbols.tobytes(order="C")),
        "probability_variant": _jsonable(variant.__dict__),
        "symbol_context_contract": "symbols_nhw_conditioned_on_separate_prev_raw_tokens_nhw",
    }


@torch.no_grad()
def encode_symbols_hpac_with_prev_context(
    gen: HPACMini,
    symbols: np.ndarray,
    prev_context_tokens: np.ndarray,
    *,
    P: int,
    delta: int,
    device: str = "cpu",
    prob_eps: float = PROB_EPS,
    probability_variant: str | HpacProbabilityVariant = DEFAULT_HPAC_PROBABILITY_VARIANT,
) -> tuple[bytes, dict[str, Any]]:
    """Encode HPAC symbols while conditioning on a separate previous-frame context.

    Public PR86's trainer describes a residual-symbol objective where
    ``symbols`` are ``(tokens[f] - tokens[f-1]) mod 5`` but ``prev_context`` is
    still the raw previous token map.  The older ``encode_tokens_hpac`` helper
    assumes symbols and previous context are the same stream; this helper keeps
    those contracts separate for contest-faithful residual experiments.
    """

    if device != "cpu":
        raise Pr86HpacReplayError("device_contract", "pr86_hpac_replay_is_cpu_only", requested_device=device)
    try:
        import constriction
    except ImportError as exc:  # pragma: no cover - dependency gate catches this first.
        raise Pr86HpacReplayError("dependency_contract", "missing_constriction") from exc
    if symbols.ndim != 3:
        raise Pr86HpacReplayError("reencode_symbols", "symbols_must_be_nhw", shape=list(symbols.shape))
    if prev_context_tokens.shape != symbols.shape:
        raise Pr86HpacReplayError(
            "reencode_symbols",
            "prev_context_shape_mismatch",
            symbols_shape=list(symbols.shape),
            prev_context_shape=list(prev_context_tokens.shape),
        )

    variant = resolve_hpac_probability_variant(probability_variant)
    dev = torch.device(device)
    gen = gen.to(dev).eval()
    frame_count, height, width = (int(v) for v in symbols.shape)
    masks = _group_masks(height, width, P, delta, dev)
    symbols_t = torch.from_numpy(symbols.astype(np.int64, copy=False)).to(dev)
    prev_t = torch.from_numpy(prev_context_tokens.astype(np.int64, copy=False)).to(dev)
    encoder = constriction.stream.queue.RangeEncoder()
    encoded_symbols = 0
    started_at = time.time()
    for frame in range(frame_count):
        idx = torch.tensor([frame], dtype=torch.long, device=dev)
        gt_symbols = symbols_t[frame : frame + 1]
        prev_tokens = prev_t[frame : frame + 1]
        current_symbols = torch.zeros_like(gt_symbols)
        for mask in masks:
            if mask is None:
                continue
            logits = gen(current_symbols, idx, prev_tokens)
            probs = F.softmax(logits.float(), dim=1)
            probs_at_group = probs[0][:, mask].permute(1, 0).contiguous()
            gt_at_group = gt_symbols[0][mask].cpu().numpy().astype(np.int32)
            probs_np = probs_at_group.cpu().numpy()
            for symbol, row in zip(gt_at_group, probs_np, strict=True):
                encoder.encode(
                    int(symbol),
                    _categorical_from_probs(row, prob_eps=prob_eps, variant=variant),
                )
                encoded_symbols += 1
            current_symbols[0, mask] = gt_symbols[0, mask]
    compressed = encoder.get_compressed()
    blob = compressed.tobytes()
    report = {
        "status": "passed",
        "frames_encoded": frame_count,
        "height": height,
        "width": width,
        "P": P,
        "delta": delta,
        "encoded_symbol_count": encoded_symbols,
        "compressed_dtype": str(compressed.dtype),
        "compressed_word_count": int(compressed.size),
        "compressed_bytes": len(blob),
        "compressed_sha256": sha256_bytes(blob),
        "elapsed_sec": round(time.time() - started_at, 3),
        "probability_variant": _jsonable(variant.__dict__),
        "symbol_context_contract": "symbols_nhw_conditioned_on_separate_prev_raw_tokens_nhw",
    }
    return blob, report


def _first_mismatch(left: bytes, right: bytes) -> int | None:
    for index, (left_byte, right_byte) in enumerate(zip(left, right, strict=False)):
        if left_byte != right_byte:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _validate_dependency_report(report: Mapping[str, Any]) -> None:
    if str(report.get("status", "")).startswith("failed_closed"):
        raise Pr86HpacReplayError(
            "dependency_contract",
            str(report.get("status")),
            dependency_contract=_jsonable(report),
        )


def _decode_required_members(bundle: Pr86ArchiveBundle, *, device: str) -> tuple[dict[str, Any], HPACMini, dict[str, Any]]:
    members = bundle.members
    _master_payload, master_report = decode_gzip_torch_member(members["master.pt.gz"], member_name="master.pt.gz")
    _slave_payload, slave_report = decode_gzip_torch_member(members["slave.pt.gz"], member_name="slave.pt.gz")
    meta_payload, meta_report = decode_meta_member(members["meta.pt"])
    hpac_model, hpac_report = load_hpac_model_from_ppmd(
        members["hpac.pt.ppmd"],
        config=meta_payload,
        device=device,
    )
    reports = {
        "master.pt.gz": master_report,
        "slave.pt.gz": slave_report,
        "meta.pt": meta_report,
        "hpac.pt.ppmd": hpac_report,
    }
    return reports, hpac_model, meta_payload


def run_pr86_hpac_replay(
    archive: Path = DEFAULT_PR86_ARCHIVE,
    *,
    contract: Pr86ArchiveContract = Pr86ArchiveContract(),
    source_dir: Path | None = DEFAULT_PR86_MERGED_SOURCE_DIR,
    source_artifacts: tuple[Path, ...] = (),
    device: str = "cpu",
    max_frames: int | None = None,
    attempt_reencode: bool = True,
    probability_variant: str | HpacProbabilityVariant = DEFAULT_HPAC_PROBABILITY_VARIANT,
) -> dict[str, Any]:
    """Run the fail-closed local replay gate and return a JSON-safe report."""

    started_at = time.time()
    try:
        variant = resolve_hpac_probability_variant(probability_variant)
        variant_report = _jsonable(variant.__dict__)
    except Pr86HpacReplayError as exc:
        return {
            "schema_version": 1,
            "tool": "tac.pr86_hpac_codec.run_pr86_hpac_replay",
            "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "status": "failed_closed",
            "score_claim": False,
            "dispatch_performed": False,
            "gpu_or_remote_work": False,
            "local_only": True,
            "device": device,
            "max_frames": max_frames,
            "probability_variant": str(probability_variant),
            "byte_parity_achieved": False,
            "dispatch_unlocked": False,
            "failure_stage": exc.stage,
            "failure_reason": exc.reason,
            "failure_context": _jsonable(exc.context),
            "elapsed_sec": round(time.time() - started_at, 3),
        }
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "tac.pr86_hpac_codec.run_pr86_hpac_replay",
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "running",
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "local_only": True,
        "device": device,
        "max_frames": max_frames,
        "probability_variant": variant_report,
        "byte_parity_achieved": False,
        "dispatch_unlocked": False,
        "archive": None,
        "current_source_context": {},
        "source_artifacts": {},
        "dependency_contract": None,
        "decoded_members": {},
        "tokens_bin": {},
        "hpac_decode": {},
        "hpac_reencode": {},
        "failure_stage": None,
        "failure_reason": None,
        "failure_context": {},
    }
    try:
        if source_dir is not None:
            report["current_source_context"] = analyze_pr86_current_source_context(source_dir)
        if source_artifacts:
            report["source_artifacts"] = load_source_artifact_summaries(source_artifacts)

        dependency_report = collect_dependency_report()
        report["dependency_contract"] = dependency_report
        _validate_dependency_report(dependency_report)

        bundle = read_pr86_archive(Path(archive), contract=contract)
        report["archive"] = bundle.report
        decoded_members, hpac_model, meta_payload = _decode_required_members(bundle, device=device)
        report["decoded_members"] = decoded_members

        token_blob = bundle.members["tokens.bin"]
        token_sha = sha256_bytes(token_blob)
        report["tokens_bin"] = {
            "bytes": len(token_blob),
            "sha256": token_sha,
            "expected_sha256": contract.expected_tokens_sha256,
            "sha256_matches_expected": token_sha == contract.expected_tokens_sha256
            if contract.expected_tokens_sha256 is not None
            else None,
            "uint32_word_count": len(token_blob) // 4 if len(token_blob) % 4 == 0 else None,
            "encoding": "little_endian_uint32_words_for_constriction_queue",
        }
        if contract.expected_tokens_sha256 is not None and token_sha != contract.expected_tokens_sha256:
            raise Pr86HpacReplayError(
                "tokens_bin_contract",
                "tokens_sha256_mismatch",
                expected_tokens_sha256=contract.expected_tokens_sha256,
                actual_tokens_sha256=token_sha,
                tokens_bytes=len(token_blob),
            )

        n_frames = int(meta_payload["N"])
        patch = int(meta_payload["P"])
        delta = int(meta_payload["delta"])
        height = int(meta_payload.get("H", SEGNET_IN_H))
        width = int(meta_payload.get("W", SEGNET_IN_W))
        tokens, decode_report = decode_tokens_hpac(
            hpac_model,
            token_blob,
            N=n_frames,
            H=height,
            W=width,
            P=patch,
            delta=delta,
            device=device,
            max_frames=max_frames,
            probability_variant=variant,
        )
        report["hpac_decode"] = decode_report

        if not attempt_reencode:
            raise Pr86HpacReplayError(
                "decode_then_reencode_byte_parity",
                "reencode_disabled",
                decoded_frames=int(tokens.shape[0]),
            )
        if tokens.shape[0] != n_frames:
            raise Pr86HpacReplayError(
                "decode_then_reencode_byte_parity",
                "partial_decode_not_dispatchable",
                decoded_frames=int(tokens.shape[0]),
                required_frames=n_frames,
            )

        reencoded_blob, reencode_report = encode_tokens_hpac(
            hpac_model,
            tokens,
            P=patch,
            delta=delta,
            device=device,
            probability_variant=variant,
        )
        mismatch = _first_mismatch(reencoded_blob, token_blob)
        reencode_report.update(
            {
                "byte_exact_reencode": reencoded_blob == token_blob,
                "source_tokens_bytes": len(token_blob),
                "source_tokens_sha256": token_sha,
                "first_mismatch_offset": mismatch,
            }
        )
        report["hpac_reencode"] = reencode_report
        if reencoded_blob != token_blob:
            raise Pr86HpacReplayError(
                "decode_then_reencode_byte_parity",
                "non_byte_exact_encode_parity",
                reencoded_tokens_bytes=len(reencoded_blob),
                reencoded_tokens_sha256=sha256_bytes(reencoded_blob),
                source_tokens_bytes=len(token_blob),
                source_tokens_sha256=token_sha,
                first_mismatch_offset=mismatch,
            )

        identity_matches = bool((report.get("archive") or {}).get("identity_matches_expected"))
        report["byte_parity_achieved"] = True
        report["dispatch_unlocked"] = identity_matches and variant.source_contract
        report["status"] = "passed"
    except Pr86HpacReplayError as exc:
        report["status"] = "failed_closed"
        report["failure_stage"] = exc.stage
        report["failure_reason"] = exc.reason
        report["failure_context"] = _jsonable(exc.context)
        if exc.stage == "submitted_tokens_decode":
            source_context = report.get("current_source_context") or {}
            token_semantics = (source_context.get("token_semantics") or {}) if isinstance(source_context, Mapping) else {}
            probability_contract = (
                source_context.get("probability_model_contract") or {}
                if isinstance(source_context, Mapping)
                else {}
            )
            report["hpac_decode"] = {
                "status": "failed_closed",
                "failure_class": exc.reason,
                **_jsonable(exc.context),
            }
            report["contract_mismatch_diagnostic"] = {
                "status": "failed_closed_current_source_contract",
                "source_context_status": source_context.get("status")
                if isinstance(source_context, Mapping)
                else None,
                "current_head_sha": (
                    (source_context.get("intake_summary") or {}).get("head_sha")
                    if isinstance(source_context, Mapping)
                    else None
                ),
                "archive_sha256": (
                    (report.get("archive") or {}).get("sha256")
                    if isinstance(report.get("archive"), Mapping)
                    else None
                ),
                "tokens_sha256": (
                    (report.get("tokens_bin") or {}).get("sha256")
                    if isinstance(report.get("tokens_bin"), Mapping)
                    else None
                ),
                "failed_at": _jsonable(exc.context.get("failed_at")),
                "decoded_symbol_count_before_failure": exc.context.get(
                    "decoded_symbol_count_before_failure"
                ),
                "raw_vs_residual_classification": token_semantics.get(
                    "submitted_archive_token_encoding"
                ),
                "training_objective": token_semantics.get("training_objective"),
                "frame0_raw_equals_residual_note": token_semantics.get(
                    "frame0_raw_equals_residual_note"
                ),
                "probability_contract_summary": probability_contract.get("contract_summary"),
                "explicit_16384_grid_in_archive_or_inflate": probability_contract.get(
                    "explicit_16384_grid_in_archive_or_inflate"
                ),
                "fail_closed_next_implementation": (
                    "Do not dispatch. First prove a full archive decode/re-encode gate under the "
                    "merged PR86 source contract, or add an explicitly named off-contract variant "
                    "probe for probability dtype/perfect-mode hypotheses and require byte-exact "
                    "tokens.bin parity before any transfer."
                ),
            }
        elif exc.stage == "decode_then_reencode_byte_parity":
            report["hpac_reencode"] = {
                "status": "failed_closed",
                "failure_class": exc.reason,
                **_jsonable(exc.context),
            }
    finally:
        report["elapsed_sec"] = round(time.time() - started_at, 3)
        report["dispatch_unlocked"] = bool(report.get("dispatch_unlocked")) and bool(
            report.get("byte_parity_achieved")
        )
    return _jsonable(report)


def run_pr86_hpac_probability_variant_matrix(
    archive: Path = DEFAULT_PR86_ARCHIVE,
    *,
    variants: tuple[str, ...] = supported_hpac_probability_variant_names(),
    contract: Pr86ArchiveContract = Pr86ArchiveContract(),
    source_dir: Path | None = DEFAULT_PR86_MERGED_SOURCE_DIR,
    source_artifacts: tuple[Path, ...] = (),
    device: str = "cpu",
    max_frames: int | None = None,
    attempt_reencode: bool = True,
) -> dict[str, Any]:
    """Run named HPAC probability variants and summarize the fail-closed gate."""

    started_at = time.time()
    variant_names = tuple(dict.fromkeys(str(name) for name in variants))
    results: list[dict[str, Any]] = []
    for name in variant_names:
        results.append(
            run_pr86_hpac_replay(
                archive=archive,
                contract=contract,
                source_dir=source_dir,
                source_artifacts=source_artifacts,
                device=device,
                max_frames=max_frames,
                attempt_reencode=attempt_reencode,
                probability_variant=name,
            )
        )

    dispatch_unlocked = any(bool(row.get("dispatch_unlocked")) for row in results)
    byte_parity_variants = [
        ((row.get("probability_variant") or {}).get("name") or str(row.get("probability_variant")))
        for row in results
        if row.get("byte_parity_achieved")
    ]
    source_contract_parity_variants = [
        ((row.get("probability_variant") or {}).get("name") or str(row.get("probability_variant")))
        for row in results
        if row.get("byte_parity_achieved")
        and (row.get("probability_variant") or {}).get("source_contract")
    ]
    status = "passed" if dispatch_unlocked else "failed_closed"
    failure_reason = None
    if not dispatch_unlocked:
        if max_frames is not None:
            failure_reason = "partial_variant_probe_not_dispatchable"
        elif not source_contract_parity_variants:
            failure_reason = "no_source_contract_variant_full_decode_byte_exact_reencode"

    report = {
        "schema_version": 1,
        "tool": "tac.pr86_hpac_codec.run_pr86_hpac_probability_variant_matrix",
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": status,
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "local_only": True,
        "device": device,
        "archive": repo_rel(Path(archive)),
        "variants_requested": list(variant_names),
        "full_submitted_tokens_required": max_frames is None,
        "attempt_reencode": attempt_reencode,
        "byte_exact_reencode_required": True,
        "byte_parity_variants": byte_parity_variants,
        "source_contract_byte_parity_variants": source_contract_parity_variants,
        "dispatch_unlocked": dispatch_unlocked,
        "failure_reason": failure_reason,
        "variant_results": results,
        "elapsed_sec": round(time.time() - started_at, 3),
    }
    return _jsonable(report)
