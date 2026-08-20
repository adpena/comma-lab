#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize the first real, causal ep725 n2 P/G/A3 counted object.

This is a structural receiver-closure artifact, not a contest candidate.  The
encoder reads the frozen ep725 source object and target cache once; the receiver
then consumes only the directory-owned counted P/G/A3 bytes plus the explicit
runtime bytes.  No scorer is invoked and no dense decoded frame is persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import stat
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tac.optimization.direct_description_carrier_compose as _ddm  # noqa: E402
from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.witness_dsl.bounded_target_g_encoder import (  # noqa: E402
    FrozenTargetSliceCustodyV1,
    compile_bounded_target_g_v2,
)
from tac.witness_dsl.dynamic_frontier_target import (  # noqa: E402
    DynamicFrontierTargetSnapshot,
    load_dynamic_frontier_target,
    verify_dynamic_frontier_target_snapshot,
)
from tac.witness_dsl.ep725_levelset_predictor_adapter import (  # noqa: E402
    EP725_ARCHIVE_BYTES,
    EP725_ARCHIVE_SHA256,
    EP725_MEMBER_BYTES,
    EP725_MEMBER_SHA256,
    EP725_RUNTIME_BYTES,
    EP725_RUNTIME_SHA256,
    EP725_SOURCE_DIRECTORY,
    decode_ep725_counted_member_ephemeral_surface,
    inspect_ep725_source,
)
from tac.witness_dsl.predictor_preserving_coupled_preimage import (  # noqa: E402
    PredictorCameraPairSurfaceV1,
    PredictorPreservingA3Mode,
    PredictorPreservingA3ProgramV1,
    compile_predictor_preserving_a3,
)
from tac.witness_dsl.predictor_preserving_taskspace_overlay import (  # noqa: E402
    overlay_g_on_predictor_camera_y1,
)
from tac.witness_dsl.taskspace_monolithic_pga_receiver import (  # noqa: E402
    build_taskspace_monolithic_pga_archive,
    receive_ep725_taskspace_monolithic_pga_archive,
)

SCHEMA: Final = "tac.ep725_n2_causal_taskspace_pga_materialization.v1"
LANE_ID: Final = "lane_codex_original_taskspace_inverse_codec_20260725"
PAIR_COUNT: Final = 2
TARGET_CACHE_SHA256: Final = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
TARGET_CACHE_BYTES: Final = 5_078_017_610
TARGET_N2_SHA256: Final = "6a9ee68a5d1ec8ec53653216d53b7406575530a2d1abf608d27547e779c6d474"
SPINE_SHA256: Final = "56b8039eba75426b3ee0ae4cb988fe1847a5f648b9f9d455cf5fc01911d19563"
SPINE_PATH: Final = REPO / ".omx/research/original_taskspace_inverse_witness_codec_20260725/spine_refresh.json"
V14_ARCHIVE_PATH: Final = (
    REPO / ".omx/research/ddm_v14_realization_fidelity_n600_20260722T215500Z/"
    "ddm_v14_islands_n600.not_a_candidate.zip.receipt-bytes"
)
V14_ARCHIVE_BYTES: Final = 133_247
V14_ARCHIVE_SHA256: Final = "2578d4909b99ce58e01575bafe8a8bb05e680019bd2157ca517aea02602c3e17"
V14_RECEIPT_PATH: Final = (
    REPO / ".omx/research/ddm_v14_realization_fidelity_n600_20260722T215500Z/"
    "ddm_v14_realization_fidelity_n600_receipt.json"
)
V14_RECEIPT_SHA256: Final = "82d3249908d42a86575c407ab3d7acdf9b3706b31225f2e46862b2472966e5a9"
PROFILE_SHA256: Final = "87c01e01c6431b6a6c96e13e368f5f6366044c21a421d35ade7970c42579cacf"
DEFAULT_ARCHIVE = (
    REPO / ".omx/research/original_taskspace_inverse_witness_codec_20260725/"
    "ep725_n2_causal_pga_control.not_a_candidate.zip"
)
DEFAULT_RECEIPT = (
    REPO / ".omx/research/original_taskspace_inverse_witness_codec_20260725/ep725_n2_causal_pga_control_receipt.json"
)
IMPLEMENTATION_PATHS: Final = (
    "src/tac/witness_dsl/ep725_levelset_predictor_adapter.py",
    "src/tac/witness_dsl/bounded_target_g_encoder.py",
    "src/tac/witness_dsl/predictor_preserving_taskspace_overlay.py",
    "src/tac/witness_dsl/predictor_preserving_coupled_preimage.py",
    "src/tac/witness_dsl/taskspace_monolithic_pga_receiver.py",
    "src/tac/witness_dsl/taskspace_outer_archive_codec.py",
    "tools/materialize_taskspace_pga_n2_receipt.py",
)


class TaskspacePGAMaterializationError(RuntimeError):
    """Source custody, exact replay, or durable-write closure failed."""


@dataclass(frozen=True, slots=True)
class MaterializedTaskspacePGAN2:
    archive_bytes: bytes
    receipt_bytes: bytes


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise TaskspacePGAMaterializationError("receipt is not finite canonical ASCII JSON") from exc


def _read_stable_regular(path: Path, *, expected_bytes: int | None = None) -> bytes:
    try:
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise TaskspacePGAMaterializationError(f"custody path is not a regular file: {path}")
        payload = path.read_bytes()
        after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise TaskspacePGAMaterializationError(f"cannot read custody path: {path}") from exc
    before_id = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if before_id != after_id or len(payload) != before.st_size:
        raise TaskspacePGAMaterializationError(f"custody path changed while reading: {path}")
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise TaskspacePGAMaterializationError(f"custody byte length changed: {path}")
    return payload


def _load_target_cache_path() -> tuple[Path, dict[str, Any]]:
    spine_bytes = _read_stable_regular(SPINE_PATH)
    if _sha256(spine_bytes) != SPINE_SHA256:
        raise TaskspacePGAMaterializationError("constructive-spine custody changed")
    try:
        spine = json.loads(spine_bytes)
        row = spine["source_custody"]["target_cache"]
        cache_path = REPO / row["path"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise TaskspacePGAMaterializationError("constructive spine lost target-cache custody") from exc
    expected = {
        "bytes": TARGET_CACHE_BYTES,
        "sha256": TARGET_CACHE_SHA256,
        "content_lineage": "source-video-derived our-build",
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise TaskspacePGAMaterializationError("constructive spine target-cache fields changed")
    try:
        metadata = cache_path.stat(follow_symlinks=False)
    except OSError as exc:
        raise TaskspacePGAMaterializationError("frozen target cache is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != TARGET_CACHE_BYTES:
        raise TaskspacePGAMaterializationError("frozen target cache identity/size changed")
    return cache_path, {
        "path": os.fspath(cache_path),
        "bytes": TARGET_CACHE_BYTES,
        "sha256": TARGET_CACHE_SHA256,
        "spine_path": os.fspath(SPINE_PATH),
        "spine_sha256": SPINE_SHA256,
        "fresh_full_cache_rehash_this_materialization": False,
        "hash_custody_inherited_from_verified_constructive_spine": True,
    }


def _load_realization_profile() -> tuple[_ddm.ReceiverRealizationProfileV1, dict[str, Any]]:
    archive_bytes = _read_stable_regular(V14_ARCHIVE_PATH, expected_bytes=V14_ARCHIVE_BYTES)
    receipt_bytes = _read_stable_regular(V14_RECEIPT_PATH)
    if _sha256(archive_bytes) != V14_ARCHIVE_SHA256 or _sha256(receipt_bytes) != V14_RECEIPT_SHA256:
        raise TaskspacePGAMaterializationError("V14 realization custody changed")
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            profile_bytes = archive.read(_ddm.REALIZATION_PROFILE_MEMBER)
            if archive.testzip() is not None:
                raise TaskspacePGAMaterializationError("V14 archive CRC verification failed")
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise TaskspacePGAMaterializationError("V14 realization member is unavailable") from exc
    if _sha256(profile_bytes) != PROFILE_SHA256:
        raise TaskspacePGAMaterializationError("V14 realization profile bytes changed")
    try:
        profile = _ddm._decode_realization_profile(profile_bytes)
    except _ddm.DirectDescriptionError as exc:
        raise TaskspacePGAMaterializationError("V14 realization profile does not parse exactly") from exc
    if profile is None:
        raise TaskspacePGAMaterializationError("V14 realization profile decoded as absent")
    return profile, {
        "archive_path": os.fspath(V14_ARCHIVE_PATH),
        "archive_bytes": len(archive_bytes),
        "archive_sha256": V14_ARCHIVE_SHA256,
        "receipt_path": os.fspath(V14_RECEIPT_PATH),
        "receipt_sha256": V14_RECEIPT_SHA256,
        "member_name": _ddm.REALIZATION_PROFILE_MEMBER,
        "member_bytes": len(profile_bytes),
        "member_sha256": PROFILE_SHA256,
        "semantic_paint_order": list(_ddm.REALIZATION_PAINT_ORDER),
        "role_rgb_u8": [list(row) for row in profile.role_rgb_u8],
        "coverage_radius": profile.coverage_radius,
        "amplitude_u8": profile.amplitude_u8,
    }


def _implementation_custody() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in IMPLEMENTATION_PATHS:
        payload = _read_stable_regular(REPO / relative)
        rows.append({"path": relative, "bytes": len(payload), "sha256": _sha256(payload)})
    return rows


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TaskspacePGAMaterializationError("git HEAD is unavailable") from exc


def materialize(*, timeout_seconds: float = 180.0) -> MaterializedTaskspacePGAN2:
    """Rebuild and double-receive the exact real n2 P/G/A3 object."""

    if (
        not isinstance(timeout_seconds, int | float)
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise TaskspacePGAMaterializationError("timeout_seconds must be finite and positive")
    frontier = load_dynamic_frontier_target(repo_root=REPO)
    if type(frontier) is not DynamicFrontierTargetSnapshot:
        raise TaskspacePGAMaterializationError("dynamic frontier loader changed exact return type")
    verify_dynamic_frontier_target_snapshot(frontier)

    source = inspect_ep725_source(EP725_SOURCE_DIRECTORY)
    if (
        len(source.archive) != EP725_ARCHIVE_BYTES
        or _sha256(source.archive) != EP725_ARCHIVE_SHA256
        or len(source.member) != EP725_MEMBER_BYTES
        or _sha256(source.member) != EP725_MEMBER_SHA256
        or len(source.runtime) != EP725_RUNTIME_BYTES
        or _sha256(source.runtime) != EP725_RUNTIME_SHA256
    ):
        raise TaskspacePGAMaterializationError("encoder-side ep725 source inspection changed exact custody")
    causal = decode_ep725_counted_member_ephemeral_surface(
        source.member,
        shipped_runtime=source.runtime,
        pair_count=PAIR_COUNT,
        timeout_seconds=timeout_seconds,
    )

    target_cache_path, target_cache_custody = _load_target_cache_path()
    lstars = open_stored_npy_memmap(target_cache_path, "lstars")
    target_labels = np.ascontiguousarray(lstars[:PAIR_COUNT], dtype=np.uint8)
    target_sha256 = _sha256(memoryview(target_labels).cast("B"))
    if target_sha256 != TARGET_N2_SHA256:
        raise TaskspacePGAMaterializationError("frozen n2 target slice changed")
    target_custody = FrozenTargetSliceCustodyV1(
        cache_sha256=TARGET_CACHE_SHA256,
        member_name="lstars",
        source_pair_ids=causal.predictor_state.source_pair_ids,
        target_labels_sha256=target_sha256,
    )
    profile, profile_custody = _load_realization_profile()
    compiled_g = compile_bounded_target_g_v2(
        causal.predictor_state,
        target_labels,
        target_custody=target_custody,
        realization_profile=profile,
    )
    if not np.array_equal(compiled_g.compiled.decoded.labels, target_labels):
        raise TaskspacePGAMaterializationError("compiled G did not close exact semantic n2 target")

    overlay = overlay_g_on_predictor_camera_y1(
        causal.frame1_camera,
        causal.predictor_state.labels,
        compiled_g.compiled.decoded,
    )
    predictor_surface = PredictorCameraPairSurfaceV1.from_ep725(causal.ephemeral_surface)
    compiled_a = compile_predictor_preserving_a3(
        PredictorPreservingA3ProgramV1(PredictorPreservingA3Mode.PASS_P0_V1),
        predictor_surface=predictor_surface,
        decoded_g=compiled_g.compiled.decoded,
        corrected_y1_overlay=overlay,
    )
    built = build_taskspace_monolithic_pga_archive(
        source.member,
        compiled_g.compiled.packet,
        compiled_a.packet,
    )
    archive_bytes = built.selected.archive_bytes
    receive_kwargs = {
        "predictor_runtime": source.runtime,
        "pair_count": PAIR_COUNT,
        "timeout_seconds": timeout_seconds,
        "expected_encoding": built.selected.encoding,
        "expected_archive_sha256": built.selected.archive_sha256,
        "expected_member_sha256": built.selected.member_sha256,
    }
    decoded_first = receive_ep725_taskspace_monolithic_pga_archive(archive_bytes, **receive_kwargs)
    decoded_second = receive_ep725_taskspace_monolithic_pga_archive(archive_bytes, **receive_kwargs)
    if decoded_first.receipt != decoded_second.receipt:
        raise TaskspacePGAMaterializationError("full receiver changed its exact receipt on replay")
    if not np.array_equal(
        decoded_first.chronological_camera_frames,
        decoded_second.chronological_camera_frames,
    ):
        raise TaskspacePGAMaterializationError("full receiver changed ephemeral chronological bytes on replay")
    if decoded_first.receipt.decoded_g_labels_sha256 != target_sha256:
        raise TaskspacePGAMaterializationError("receiver semantic output differs from frozen n2 target")

    verify_dynamic_frontier_target_snapshot(frontier)
    body = {
        "schema": SCHEMA,
        "lane_id": LANE_ID,
        "scope": "real ep725 prefix n2 structural receiver closure only",
        "competitive_target": asdict(frontier),
        "git_head_before_landing": _git_head(),
        "encoder_inputs": {
            "ep725_source_directory": os.fspath(EP725_SOURCE_DIRECTORY),
            "historical_source_archive_bytes": len(source.archive),
            "historical_source_archive_sha256": _sha256(source.archive),
            "counted_predictor_member_bytes": len(source.member),
            "counted_predictor_member_sha256": _sha256(source.member),
            "explicit_runtime_bytes": len(source.runtime),
            "explicit_runtime_sha256": _sha256(source.runtime),
            "source_directory_read_encoder_side_only": True,
            "source_directory_or_archive_read_receiver_side": False,
            "target_cache": target_cache_custody,
            "target_n2_sha256": target_sha256,
            "realization_profile": profile_custody,
        },
        "counted_sections": {
            "P": {"bytes": len(source.member), "sha256": _sha256(source.member)},
            "G": {
                "bytes": len(compiled_g.compiled.packet),
                "sha256": _sha256(compiled_g.compiled.packet),
                "encoder_receipt": compiled_g.receipt.as_dict(),
            },
            "A": {
                "bytes": len(compiled_a.packet),
                "sha256": _sha256(compiled_a.packet),
                "mode": compiled_a.program.mode.value,
                "receipt": compiled_a.receipt.as_dict(),
            },
        },
        "whole_object": {
            "selected_encoding": built.selected.encoding.value,
            "selected_archive_bytes": built.selected.archive_nbytes,
            "selected_archive_sha256": built.selected.archive_sha256,
            "selected_member_bytes": built.selected.member_nbytes,
            "selected_member_sha256": built.selected.member_sha256,
            "selected_compressed_member_bytes": built.selected.compressed_member_nbytes,
            "stored_archive_bytes": built.stored.archive_nbytes,
            "stored_archive_sha256": built.stored.archive_sha256,
            "deflated_archive_bytes": built.deflated.archive_nbytes,
            "deflated_archive_sha256": built.deflated.archive_sha256,
            "selection_reason": built.selection_reason,
            "tie_break_rule": built.tie_break_rule,
            "zlib_compile_version": built.zlib_compile_version,
            "zlib_runtime_version": built.zlib_runtime_version,
            "deflate_profile": built.deflate_profile,
        },
        "receiver": {
            "receipt": decoded_first.receipt.as_dict(),
            "receipt_sha256": decoded_first.receipt.receipt_sha256,
            "double_receive_exact": True,
            "directory_owned_counted_P_is_actual_decode_input": True,
            "explicit_runtime_is_actual_decode_input": True,
            "source_archive_read_for_decode": False,
            "dense_frames_persisted": False,
        },
        "measured_semantic_control": {
            "source_pair_ids": list(causal.predictor_state.source_pair_ids),
            "predictor_labels_sha256": causal.causal_receipt.labels_sha256,
            "target_labels_sha256": target_sha256,
            "semantic_debt_before_cells": compiled_g.receipt.debt_before_cells,
            "semantic_debt_after_cells": compiled_g.receipt.debt_after_cells,
            "topology_events": compiled_g.receipt.total_topology_events,
            "exact_semantic_target_reconstructed": True,
        },
        "open_blockers": [
            "same_class_realization_repair_not_composed_or_measured_through_R",
            "A3_SE3_XIP2_inverse_row_selection_not_implemented",
            "standalone_runtime_packaging_and_clean_contest_inflate_not_closed",
            "n600_materialization_and_authoritative_CPU_CUDA_eval_not_run",
        ],
        "truth": {
            "scorer_invoked": False,
            "through_r_target_realization_verified": False,
            "exact_score_claim": False,
            "candidate_archive_eligible": False,
            "standalone_runtime_closure": False,
            "originality_claim": False,
            "promotion_eligible": False,
            "research_only": True,
        },
        "implementation_custody": _implementation_custody(),
    }
    receipt_bytes = _canonical_json(body) + b"\n"
    return MaterializedTaskspacePGAN2(archive_bytes=archive_bytes, receipt_bytes=receipt_bytes)


def parse_materialization_receipt(payload: bytes) -> dict[str, Any]:
    """Strictly parse the newline-terminated materialization envelope."""

    if type(payload) is not bytes or not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise TaskspacePGAMaterializationError("receipt must have exactly one terminal newline")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TaskspacePGAMaterializationError(f"receipt repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload[:-1].decode("ascii"), object_pairs_hook=unique_pairs)
    except TaskspacePGAMaterializationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TaskspacePGAMaterializationError("receipt is not strict ASCII JSON") from exc
    expected_keys = {
        "schema",
        "lane_id",
        "scope",
        "competitive_target",
        "git_head_before_landing",
        "encoder_inputs",
        "counted_sections",
        "whole_object",
        "receiver",
        "measured_semantic_control",
        "open_blockers",
        "truth",
        "implementation_custody",
    }
    if type(value) is not dict or set(value) != expected_keys or value.get("schema") != SCHEMA:
        raise TaskspacePGAMaterializationError("receipt top-level schema is not closed V1")
    if _canonical_json(value) + b"\n" != payload:
        raise TaskspacePGAMaterializationError("receipt is not canonical on parse-back")
    truth = value.get("truth")
    if type(truth) is not dict or truth != {
        "candidate_archive_eligible": False,
        "exact_score_claim": False,
        "originality_claim": False,
        "promotion_eligible": False,
        "research_only": True,
        "scorer_invoked": False,
        "standalone_runtime_closure": False,
        "through_r_target_realization_verified": False,
    }:
        raise TaskspacePGAMaterializationError("receipt truth labels became permissive")
    return value


def write_once_or_equal(path: Path, payload: bytes) -> None:
    """Persist exact bytes without overwriting any different historical artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_stable_regular(path) != payload:
            raise TaskspacePGAMaterializationError(f"refusing to overwrite different artifact: {path}")
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                path.unlink()
            except OSError:
                pass
            raise
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError:
        if _read_stable_regular(path) != payload:
            raise TaskspacePGAMaterializationError(f"artifact race produced different bytes: {path}") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-output", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--receipt-output", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()
    artifact = materialize(timeout_seconds=args.timeout_seconds)
    parsed = parse_materialization_receipt(artifact.receipt_bytes)
    if parsed["whole_object"]["selected_archive_sha256"] != _sha256(artifact.archive_bytes):
        raise TaskspacePGAMaterializationError("receipt/archive binding changed before durable write")
    write_once_or_equal(args.archive_output, artifact.archive_bytes)
    write_once_or_equal(args.receipt_output, artifact.receipt_bytes)
    summary = {
        "archive_output": os.fspath(args.archive_output),
        "archive_bytes": len(artifact.archive_bytes),
        "archive_sha256": _sha256(artifact.archive_bytes),
        "receipt_output": os.fspath(args.receipt_output),
        "receipt_bytes": len(artifact.receipt_bytes),
        "receipt_sha256": _sha256(artifact.receipt_bytes),
        "candidate_archive_eligible": False,
        "score_claim": False,
    }
    print(_canonical_json(summary).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
