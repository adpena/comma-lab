# SPDX-License-Identifier: MIT
"""QA75 exact-solve frame targets — scorer-free materializer + typed loader.

Materializes the EXACT C1 (``ms2r_r3`` box-tolerance) solve archive's per-pair
camera-resolution RGB frames (``frame0`` + ``frame1``, 874x1164x3 ``uint8``) via
the v10 production receiver's *scorer-free* decode + factor-2 realization path,
so the burn-2 QA75 solve-frame distill stage can consume them as regression
targets. The teacher frames realize the EXACT solve THROUGH the real decode
path (feasible margins by construction), which GT frames cannot.

NO SegNet / PoseNet is run here — the logit/margin precompute on these frames is
the deliberately-deferred POST-BURN step (running the frozen scorer would take
the Metal slot the live burn holds). This surface only *materializes* the
integer frames the burn-2 distill stage will index by ``pair_id``.

Pointer honesty: ``0.1910828242 [contest-CPU]`` UNMOVED; every artifact this
module writes is ``research_only``, ``score_claim=false``, ``[macOS-CPU
advisory]``. Determinism is guaranteed by construction (the v10 realize path is
integer-only); :func:`determinism_spotcheck` proves it on a sample.

Ledger custody: QA75 in ``.omx/research/ddm_deferral_queue_ledger_20260729.md``;
design ph3 §10.1 (``ddm_ph3_realization_hybrid_adaptive_convocation_20260731.md``);
blocker source sg1 §5 (``ddm_sg1_segnet_typing_and_reburn_20260731.md``).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.witness_dsl import v10_production_receiver as _R

MANIFEST_SCHEMA = "qa75_solve_frame_targets.v1"
MANIFEST_NAME = "manifest.json"


class SolveFrameTargetsError(ValueError):
    """Raised on any materialize/load custody or geometry violation (fail-closed)."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    """Write a ``.npy`` atomically (tmp + ``os.replace``) so a crash never leaves a torn file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_manifest(out_dir: Path, manifest: dict[str, Any]) -> Path:
    path = out_dir / MANIFEST_NAME
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    file_descriptor, tmp_name = tempfile.mkstemp(prefix=f".{MANIFEST_NAME}.", suffix=".tmp", dir=out_dir)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(file_descriptor, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


@dataclass(frozen=True)
class MaterializeResult:
    """Typed receipt for a materialization run (advisory; no score authority)."""

    out_dir: Path
    manifest_path: Path
    pair_count: int
    frame_shape: tuple[int, int, int, int]  # (2, camera_h, camera_w, channels)
    frame0_described: bool
    source_archive_sha256: str
    packet_sha256: str


def _realize_pair(packet: Any, yp: Any, pair_index: int, *, described: bool) -> np.ndarray:
    """Realize (frame0, frame1) camera-resolution uint8 → stacked (2, H, W, C). Scorer-free."""

    frame1, _ = _R.realize_pair_frame1(packet, yp.frame1[pair_index])
    if described:
        frame0, _ = _R.realize_pair_frame1(packet, yp.frame0[pair_index])
    else:
        frame0 = frame1.copy()
    return np.stack([frame0, frame1], axis=0)


def materialize_solve_frames(
    archive_dir: Path | str,
    out_dir: Path | str,
    *,
    limit: int | None = None,
) -> MaterializeResult:
    """Decode the v10 solve archive to per-pair ``pair-NNNNNN.npy`` (2,H,W,C) uint8 + a manifest.

    ``limit`` caps the number of pairs (for a smoke); ``None`` materializes all.
    Each ``.npy`` is written atomically and the manifest carries its sha256 so the
    loader can re-verify custody. NO scorer is imported or run.
    """

    archive_dir = Path(archive_dir)
    out_dir = Path(out_dir)
    packet_bytes, archive_sha, archive_bytes = _R._read_archive_packet(archive_dir / "archive.zip")
    packet = _R.parse_packet(packet_bytes)
    if packet.header["residual_codec_id"] is not None:
        raise SolveFrameTargetsError(
            "this archive carries a quotient residual; this scorer-free materializer realizes the "
            "nullspace plane only — use v10_production_receiver.inflate_archive for residual archives"
        )
    yp = _R.decode_y_plane_pair(packet)
    camera_h, camera_w, _scorer_h, _scorer_w, channels = _R._validate_geometry(packet.header)
    pair_count = int(packet.header["pair_count"])
    described = packet.header["frame0_policy_id"] == _R.DESCRIPTION_FRAME0_POLICY_ID
    count = pair_count if limit is None else min(int(limit), pair_count)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[dict[str, Any]] = []
    for pair_index in range(count):
        stacked = _realize_pair(packet, yp, pair_index, described=described)
        name = f"pair-{pair_index:06d}.npy"
        path = out_dir / name
        _atomic_write_npy(path, stacked)
        pairs.append(
            {
                "pair_id": pair_index,
                "path": name,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )

    frame_shape = (2, int(camera_h), int(camera_w), int(channels))
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "score_claim": False,
        "research_only": True,
        "axis": "[macOS-CPU advisory]",
        "source_archive_dir": str(archive_dir),
        "source_archive_sha256": archive_sha,
        "source_archive_bytes": int(archive_bytes),
        "packet_sha256": packet.packet_sha256,
        "geometry": {
            "camera_h": int(camera_h),
            "camera_w": int(camera_w),
            "channels": int(channels),
            "frame_shape": list(frame_shape),
        },
        "frame0_described": bool(described),
        "frame0_policy_id": packet.header["frame0_policy_id"],
        "pair_count_total": pair_count,
        "pair_count_materialized": count,
        "pairs": pairs,
    }
    manifest_path = _write_manifest(out_dir, manifest)
    return MaterializeResult(
        out_dir=out_dir,
        manifest_path=manifest_path,
        pair_count=count,
        frame_shape=frame_shape,
        frame0_described=described,
        source_archive_sha256=archive_sha,
        packet_sha256=packet.packet_sha256,
    )


@dataclass(frozen=True)
class SolveFrameTargets:
    """Typed read-only loader for materialized QA75 solve-frame distill targets.

    The burn-2 distill stage consumes this: ``targets.frame1(pair_id)`` /
    ``targets.frame0(pair_id)`` return memmapped ``(H, W, C)`` uint8 arrays.
    """

    root: Path
    manifest: dict[str, Any]

    @classmethod
    def load(cls, root: Path | str) -> SolveFrameTargets:
        root = Path(root)
        manifest_path = root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise SolveFrameTargetsError(f"no {MANIFEST_NAME} under {root}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise SolveFrameTargetsError(f"unexpected manifest schema {manifest.get('schema')!r}")
        return cls(root=root, manifest=manifest)

    @property
    def pair_count(self) -> int:
        return int(self.manifest["pair_count_materialized"])

    @property
    def frame_shape(self) -> tuple[int, int, int, int]:
        return tuple(int(v) for v in self.manifest["geometry"]["frame_shape"])  # type: ignore[return-value]

    @property
    def frame0_described(self) -> bool:
        return bool(self.manifest["frame0_described"])

    def _row(self, pair_id: int) -> dict[str, Any]:
        pairs = self.manifest["pairs"]
        if not 0 <= pair_id < len(pairs):
            raise SolveFrameTargetsError(f"pair_id {pair_id} out of range [0,{len(pairs)})")
        row = pairs[pair_id]
        if row["pair_id"] != pair_id:
            raise SolveFrameTargetsError(f"manifest pair_id drift at index {pair_id}: {row['pair_id']}")
        return row

    def pair_path(self, pair_id: int) -> Path:
        return self.root / self._row(pair_id)["path"]

    def pair(self, pair_id: int, *, verify: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(frame0, frame1)`` as memmapped ``(H, W, C)`` uint8 arrays."""

        if verify and not self.verify_sha(pair_id):
            raise SolveFrameTargetsError(f"sha256 custody mismatch for pair {pair_id}")
        stacked = np.load(self.pair_path(pair_id), mmap_mode="r", allow_pickle=False)
        if stacked.shape[0] != 2:
            raise SolveFrameTargetsError(f"pair {pair_id} npy leading dim {stacked.shape[0]} != 2")
        return stacked[0], stacked[1]

    def frame0(self, pair_id: int, *, verify: bool = False) -> np.ndarray:
        return self.pair(pair_id, verify=verify)[0]

    def frame1(self, pair_id: int, *, verify: bool = False) -> np.ndarray:
        return self.pair(pair_id, verify=verify)[1]

    def verify_sha(self, pair_id: int) -> bool:
        row = self._row(pair_id)
        return _sha256_file(self.pair_path(pair_id)) == row["sha256"]


def determinism_spotcheck(
    archive_dir: Path | str,
    sample_ids: Sequence[int],
) -> dict[str, Any]:
    """Decode the packet twice and realize ``sample_ids`` twice; compare per-pair sha256.

    Proves the scorer-free materialization is bit-identical run-to-run (the v10
    realize path is integer-only, so this must be ``identical=True``).
    """

    archive_dir = Path(archive_dir)

    def _once() -> dict[int, str]:
        packet_bytes, _sha, _bytes = _R._read_archive_packet(archive_dir / "archive.zip")
        packet = _R.parse_packet(packet_bytes)
        if packet.header["residual_codec_id"] is not None:
            raise SolveFrameTargetsError("residual archives are unsupported by this scorer-free materializer")
        yp = _R.decode_y_plane_pair(packet)
        described = packet.header["frame0_policy_id"] == _R.DESCRIPTION_FRAME0_POLICY_ID
        out: dict[int, str] = {}
        for pair_index in sample_ids:
            stacked = _realize_pair(packet, yp, int(pair_index), described=described)
            out[int(pair_index)] = _sha256_bytes(stacked.tobytes(order="C"))
        return out

    run1 = _once()
    run2 = _once()
    per_pair = {
        int(i): {"run1": run1[int(i)], "run2": run2[int(i)], "match": run1[int(i)] == run2[int(i)]}
        for i in sample_ids
    }
    return {
        "schema": "qa75_solve_frame_determinism_spotcheck.v1",
        "sample_ids": [int(i) for i in sample_ids],
        "identical": run1 == run2,
        "per_pair": per_pair,
    }
