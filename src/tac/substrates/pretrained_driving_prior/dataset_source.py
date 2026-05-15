# SPDX-License-Identifier: MIT
"""Dataset-source custody contract for DP1/pretrained-driving-prior runs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    """Return SHA-256 for ``path`` using bounded memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_sha256_hex(value: object) -> bool:
    """Return True iff ``value`` is a lowercase 64-character SHA-256 hex."""
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def collect_local_video_manifest(
    root: Path,
    *,
    max_files: int | None = None,
) -> list[dict[str, Any]]:
    """Collect a deterministic local ``video.hevc`` manifest with SHA-256s.

    The function is intentionally narrow: DP1 Comma2k19 local-chunk mode consumes
    chunk trees whose decode entrypoints are ``video.hevc`` files. Hashing those
    files gives a reproducible source manifest without guessing dataset layout.
    """
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"DP1 dataset source root does not exist: {root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("video.hevc")):
        if max_files is not None and len(rows) >= int(max_files):
            break
        rows.append(
            {
                "relpath": path.relative_to(root).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return rows


def final_step_chunk_ids(schedule_log: list[Any]) -> list[str]:
    """Extract the final cumulative chunk-id window from a schedule log."""
    if not schedule_log:
        return []
    last = schedule_log[-1]
    provenance = (
        last.provenance
        if hasattr(last, "provenance")
        else dict(last.get("provenance", {}))
    )
    for key in ("cached_chunk_ids_used", "streamed_chunk_ids_used"):
        values = provenance.get(key)
        if isinstance(values, list):
            return [str(value) for value in values]
    return []


def final_step_chunk_sha256s(schedule_log: list[Any]) -> dict[str, str]:
    """Extract final-step chunk SHA-256 mapping when the schedule recorded it."""
    if not schedule_log:
        return {}
    last = schedule_log[-1]
    provenance = (
        last.provenance
        if hasattr(last, "provenance")
        else dict(last.get("provenance", {}))
    )
    values = provenance.get("chunk_sha256_manifest")
    if not isinstance(values, dict):
        return {}
    return {str(k): str(v) for k, v in values.items() if is_sha256_hex(v)}


@dataclass(frozen=True)
class DP1DatasetSource:
    """One DP1 dataset-source custody record.

    ``reproducibility_blockers`` is intentionally part of the data contract.
    Synthetic and prebuilt-codebook runs are allowed for structural tests, but
    real pretraining must expose chunk identity and SHA coverage before it can
    become score-bearing evidence.
    """

    dataset_name: str
    source_mode: str
    distillation_mode: str
    seed: int
    max_distillation_frames: int
    max_distillation_chunks: int
    synthetic: bool = False
    chunks_dir: str | None = None
    cache_dir: str | None = None
    stream_log_dir: str | None = None
    codebook_path: str | None = None
    chunk_ids: list[str] = field(default_factory=list)
    chunk_sha256_manifest: dict[str, str] = field(default_factory=dict)
    local_video_manifest: list[dict[str, Any]] = field(default_factory=list)
    license_tags: list[str] = field(default_factory=list)
    dataset_provenance: str = ""
    frames_used: int | None = None
    reproducibility_blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "dp1_dataset_source_manifest.v1",
            "dataset_name": self.dataset_name,
            "source_mode": self.source_mode,
            "distillation_mode": self.distillation_mode,
            "seed": int(self.seed),
            "max_distillation_frames": int(self.max_distillation_frames),
            "max_distillation_chunks": int(self.max_distillation_chunks),
            "synthetic": bool(self.synthetic),
            "chunks_dir": self.chunks_dir,
            "cache_dir": self.cache_dir,
            "stream_log_dir": self.stream_log_dir,
            "codebook_path": self.codebook_path,
            "chunk_ids": list(self.chunk_ids),
            "chunk_sha256_manifest": dict(self.chunk_sha256_manifest),
            "chunk_sha256_coverage": {
                "chunk_count": len(self.chunk_ids),
                "covered_count": sum(
                    1
                    for chunk_id in self.chunk_ids
                    if is_sha256_hex(self.chunk_sha256_manifest.get(chunk_id))
                ),
                "complete": all(
                    is_sha256_hex(self.chunk_sha256_manifest.get(chunk_id))
                    for chunk_id in self.chunk_ids
                )
                if self.chunk_ids
                else False,
            },
            "local_video_manifest": list(self.local_video_manifest),
            "license_tags": list(self.license_tags),
            "dataset_provenance": self.dataset_provenance,
            "frames_used": self.frames_used,
            "reproducibility_blockers": list(self.reproducibility_blockers),
            "score_claim_allowed": False,
        }


def build_dp1_dataset_source(
    *,
    dataset_name: str,
    source_mode: str,
    distillation_mode: str,
    seed: int,
    max_distillation_frames: int,
    max_distillation_chunks: int,
    codebook_metadata: dict[str, Any] | None = None,
    chunks_dir: Path | None = None,
    cache_dir: Path | None = None,
    stream_log_dir: Path | None = None,
    codebook_path: Path | None = None,
    schedule_log: list[Any] | None = None,
) -> DP1DatasetSource:
    """Build a fail-closed DP1 dataset-source record."""
    metadata = dict(codebook_metadata or {})
    schedule_log = list(schedule_log or [])
    chunk_ids = final_step_chunk_ids(schedule_log)
    chunk_sha256s = final_step_chunk_sha256s(schedule_log)
    local_manifest: list[dict[str, Any]] = []
    blockers: list[str] = []

    if chunks_dir is not None:
        local_manifest = collect_local_video_manifest(
            chunks_dir,
            max_files=max_distillation_chunks,
        )
        chunk_ids = [row["relpath"] for row in local_manifest]
        chunk_sha256s = {row["relpath"]: row["sha256"] for row in local_manifest}
        if not local_manifest:
            blockers.append("dp1_local_chunks_dir_has_no_video_hevc_files")

    if dataset_name == "comma2k19" and source_mode not in {
        "local_chunks",
        "local_cache",
        "stream_log",
        "prebuilt_codebook",
    }:
        blockers.append("dp1_comma2k19_requires_one_real_source_mode")
    if dataset_name in {"comma10k", "comma10k19"}:
        blockers.append("dp1_comma10k_is_segnet_image_prior_not_video_pretraining")
    if dataset_name in {"bdd100k", "waymo", "waymo_open_dataset"}:
        blockers.append(f"dp1_dataset_not_trainer_wired_{dataset_name}")
    if dataset_name == "comma2k19" and source_mode != "prebuilt_codebook":
        if not chunk_ids:
            blockers.append("dp1_real_dataset_source_has_no_chunk_ids")
        missing_sha = [
            chunk_id
            for chunk_id in chunk_ids
            if not is_sha256_hex(chunk_sha256s.get(chunk_id))
        ]
        if missing_sha:
            blockers.append("dp1_real_dataset_source_missing_pinned_sha256")

    return DP1DatasetSource(
        dataset_name=dataset_name,
        source_mode=source_mode,
        distillation_mode=distillation_mode,
        seed=seed,
        max_distillation_frames=max_distillation_frames,
        max_distillation_chunks=max_distillation_chunks,
        synthetic=dataset_name == "synthetic_test",
        chunks_dir=str(chunks_dir.expanduser().resolve()) if chunks_dir else None,
        cache_dir=str(cache_dir.expanduser().resolve()) if cache_dir else None,
        stream_log_dir=(
            str(stream_log_dir.expanduser().resolve()) if stream_log_dir else None
        ),
        codebook_path=str(codebook_path) if codebook_path else None,
        chunk_ids=chunk_ids,
        chunk_sha256_manifest=chunk_sha256s,
        local_video_manifest=local_manifest,
        license_tags=[str(x) for x in metadata.get("license_tags", [])],
        dataset_provenance=str(metadata.get("dataset_provenance", "")),
        frames_used=(
            int(metadata["num_frames_used"])
            if isinstance(metadata.get("num_frames_used"), int)
            else None
        ),
        reproducibility_blockers=blockers,
    )
