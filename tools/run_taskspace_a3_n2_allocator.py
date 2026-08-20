#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the real n2 A3 whole-object allocator after explicit root review.

The output is research-only macOS-CPU advisory evidence.  It is neither an
n600 evaluation nor a contest score, candidate, promotion, or frontier move.
Without ``--execute-reviewed`` the tool refuses to run the real decoder/scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shlex
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
UPSTREAM = REPO / "upstream"
TOOLS = REPO / "tools"
for _search_path in (SRC, UPSTREAM, TOOLS):
    if str(_search_path) not in sys.path:
        sys.path.insert(0, str(_search_path))

import materialize_taskspace_pga_n2_receipt as _materializer  # noqa: E402
import measure_taskspace_pga_n2_macos_cpu as _measurement  # noqa: E402

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
    EP725_RUNTIME_BYTES,
    EP725_RUNTIME_SHA256,
    decode_ep725_counted_member_ephemeral_surface,
    inspect_ep725_source,
)
from tac.witness_dsl.predictor_preserving_coupled_preimage import (  # noqa: E402
    PredictorCameraPairSurfaceV1,
)
from tac.witness_dsl.predictor_preserving_taskspace_overlay import (  # noqa: E402
    overlay_g_on_predictor_camera_y1,
)
from tac.witness_dsl.taskspace_chronological_a3_encoder import (  # noqa: E402
    A3PrefixPlanV1,
    ChronologicalA3AcquisitionV1,
    ChronologicalA3Interpretation,
    EncoderOnlyA3TargetV1,
    compile_chronological_a3_proposal_groups,
)
from tac.witness_dsl.taskspace_monolithic_pga_receiver import (  # noqa: E402
    TaskspaceMonolithicPGARole,
    parse_taskspace_monolithic_pga_member,
    receive_ep725_taskspace_monolithic_pga_archive,
)
from tac.witness_dsl.taskspace_outer_archive_codec import (  # noqa: E402
    parse_taskspace_outer_archive,
)
from tac.witness_dsl.taskspace_whole_archive_allocator import (  # noqa: E402
    TaskspaceMeasurementRequestV1,
    TaskspaceRealizedMeasurementReceiptV1,
    TaskspaceReceiverReceiptV1,
    TaskspaceReceiverRequestV1,
    TaskspaceSectionBundleV1,
    TaskspaceWholeArchiveAllocationV1,
    allocate_taskspace_whole_archive,
)

SCHEMA: Final = "tac.taskspace_a3_n2_whole_archive_allocation.v1"
LANE_ID: Final = "lane_g11_taskspace_a3_n2_allocator_runner_20260726"
AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
PAIR_COUNT: Final = 2
SEED: Final = 1234
DEFAULT_ROW_COUNTS: Final = (1, 4, 16)
DEFAULT_OUTPUT: Final = (
    REPO / ".omx/research/original_taskspace_inverse_witness_codec_20260725/"
    "ep725_n2_a3_whole_archive_allocation_20260726.json"
)
DEFAULT_ARCHIVE_OUTPUT: Final = (
    REPO / ".omx/research/original_taskspace_inverse_witness_codec_20260725/"
    "ep725_n2_a3_whole_archive_allocation_final.not_a_candidate.zip"
)
BASELINE_ARCHIVE: Final = _materializer.DEFAULT_ARCHIVE
BASELINE_RECEIPT: Final = _materializer.DEFAULT_RECEIPT
IMPLEMENTATION_PATHS: Final = (
    "src/tac/witness_dsl/taskspace_chronological_a3_encoder.py",
    "src/tac/witness_dsl/taskspace_whole_archive_allocator.py",
    "src/tac/witness_dsl/taskspace_monolithic_pga_receiver.py",
    "src/tac/witness_dsl/taskspace_outer_archive_codec.py",
    "tools/materialize_taskspace_pga_n2_receipt.py",
    "tools/measure_taskspace_pga_n2_macos_cpu.py",
    "tools/run_taskspace_a3_n2_allocator.py",
)
TRUTH: Final = {
    "authoritative_contest_cpu_evaluation": False,
    "authoritative_contest_cuda_evaluation": False,
    "candidate_archive_eligible": False,
    "exact_score_claim": False,
    "macos_cpu_advisory": True,
    "n2_only": True,
    "n600_evaluation": False,
    "originality_claim": False,
    "pointer_moved": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "research_only": True,
}


class TaskspaceA3N2AllocatorRunnerError(RuntimeError):
    """Real-input custody, callback binding, or durable-output failure."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(memoryview(np.ascontiguousarray(value)).cast("B"))


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
        raise TaskspaceA3N2AllocatorRunnerError("record is not finite canonical ASCII JSON") from exc


def _read_stable_regular(path: Path) -> bytes:
    try:
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise TaskspaceA3N2AllocatorRunnerError(f"custody path is not a regular file: {path}")
        payload = path.read_bytes()
        after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise TaskspaceA3N2AllocatorRunnerError(f"cannot read custody path: {path}") from exc
    before_id = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if before_id != after_id or len(payload) != before.st_size:
        raise TaskspaceA3N2AllocatorRunnerError(f"custody path changed while reading: {path}")
    return payload


def _git_head() -> str:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TaskspaceA3N2AllocatorRunnerError("git HEAD is unavailable") from exc
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise TaskspaceA3N2AllocatorRunnerError("git HEAD is not canonical lowercase SHA-1")
    return head


def _implementation_custody() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in IMPLEMENTATION_PATHS:
        payload = _read_stable_regular(REPO / relative)
        rows.append({"path": relative, "bytes": len(payload), "sha256": _sha256(payload)})
    return rows


def parse_row_counts(value: str) -> tuple[int, ...]:
    """Parse a bounded strictly geometric positive ladder."""

    try:
        rows = tuple(int(token) for token in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("row counts must be comma-separated integers") from exc
    if not rows or any(row < 1 for row in rows):
        raise argparse.ArgumentTypeError("row counts must be positive")
    if rows != tuple(sorted(set(rows))):
        raise argparse.ArgumentTypeError("row counts must be strictly increasing and unique")
    if len(rows) > 8 or rows[-1] > 4 * 384 * 512:
        raise argparse.ArgumentTypeError("row-count ladder exceeds the bounded A3 universe")
    if len(rows) > 2:
        ratios = tuple(rows[index + 1] / rows[index] for index in range(len(rows) - 1))
        if not all(math.isclose(ratio, ratios[0], rel_tol=0.0, abs_tol=0.0) for ratio in ratios[1:]):
            raise argparse.ArgumentTypeError("row counts must form an exact geometric ladder")
    return rows


def build_prefix_plans(row_counts: tuple[int, ...]) -> tuple[A3PrefixPlanV1, ...]:
    """Keep both executable interpretations in a deterministic paired order."""

    return (
        A3PrefixPlanV1(ChronologicalA3Interpretation.TARGET_CONSTANT_RGB_V1, row_counts),
        A3PrefixPlanV1(ChronologicalA3Interpretation.CORRECTED_Y1_SUPPORT_COPY_V1, row_counts),
    )


@dataclass(frozen=True, slots=True)
class RealN2Context:
    frontier: DynamicFrontierTargetSnapshot
    baseline_bundle: TaskspaceSectionBundleV1
    acquisition: ChronologicalA3AcquisitionV1
    predictor_runtime: bytes
    predictor_packet: bytes
    target_frames: np.ndarray
    target_labels: np.ndarray
    target_poses: np.ndarray
    custody: dict[str, Any]


def _build_real_n2_context(
    row_counts: tuple[int, ...],
    *,
    timeout_seconds: float,
) -> RealN2Context:
    frontier = load_dynamic_frontier_target(repo_root=REPO)
    verify_dynamic_frontier_target_snapshot(frontier)
    source = inspect_ep725_source()
    if len(source.runtime) != EP725_RUNTIME_BYTES or _sha256(source.runtime) != EP725_RUNTIME_SHA256:
        raise TaskspaceA3N2AllocatorRunnerError("explicit ep725 runtime custody changed")
    causal = decode_ep725_counted_member_ephemeral_surface(
        source.member,
        shipped_runtime=source.runtime,
        pair_count=PAIR_COUNT,
        timeout_seconds=timeout_seconds,
    )
    target_frames, target_labels, target_poses, target_custody = _measurement._load_target()
    target_cache_path, materializer_target_custody = _materializer._load_target_cache_path()
    labels_again = np.ascontiguousarray(
        open_stored_npy_memmap(target_cache_path, "lstars")[:PAIR_COUNT],
        dtype=np.uint8,
    )
    if not np.array_equal(labels_again, target_labels):
        raise TaskspaceA3N2AllocatorRunnerError("materializer and scorer target-label views differ")
    target_slice_custody = FrozenTargetSliceCustodyV1(
        cache_sha256=_materializer.TARGET_CACHE_SHA256,
        member_name="lstars",
        source_pair_ids=causal.predictor_state.source_pair_ids,
        target_labels_sha256=_array_sha256(target_labels),
    )
    realization_profile, realization_custody = _materializer._load_realization_profile()
    compiled_g = compile_bounded_target_g_v2(
        causal.predictor_state,
        target_labels,
        target_custody=target_slice_custody,
        realization_profile=realization_profile,
    )
    if not np.array_equal(compiled_g.compiled.decoded.labels, target_labels):
        raise TaskspaceA3N2AllocatorRunnerError("real G failed exact n2 semantic reconstruction")
    overlay = overlay_g_on_predictor_camera_y1(
        causal.frame1_camera,
        causal.predictor_state.labels,
        compiled_g.compiled.decoded,
    )
    predictor_surface = PredictorCameraPairSurfaceV1.from_ep725(causal.ephemeral_surface)
    target = EncoderOnlyA3TargetV1(
        source_pair_ids=causal.predictor_state.source_pair_ids,
        target_camera_y0=target_frames[:, 0],
        custody_sha256=_measurement.TARGET_F0_U8_SHA256,
        evidence_kind="frozen_gt_f0_n2_encoder_only.v1",
    )
    acquisition = compile_chronological_a3_proposal_groups(
        predictor_surface=predictor_surface,
        decoded_g=compiled_g.compiled.decoded,
        corrected_y1_overlay=overlay,
        target=target,
        plans=build_prefix_plans(row_counts),
        proposal_id_prefix="g11n2",
    )
    baseline_bundle = TaskspaceSectionBundleV1(
        predictor_packet=source.member,
        generative_correction_packet=compiled_g.compiled.packet,
        coupled_preimage_packet=acquisition.pass_p0_control.packet,
    )
    baseline_archive = _read_stable_regular(BASELINE_ARCHIVE)
    baseline_receipt_bytes = _read_stable_regular(BASELINE_RECEIPT)
    _materializer.parse_materialization_receipt(baseline_receipt_bytes)
    baseline_outer = parse_taskspace_outer_archive(baseline_archive)
    baseline_member = parse_taskspace_monolithic_pga_member(baseline_outer.member_bytes)
    observed = tuple((section.role, section.payload) for section in baseline_member.sections)
    if observed != baseline_bundle.role_payloads:
        raise TaskspaceA3N2AllocatorRunnerError("reconstructed P/G/PASS_P0 differs from frozen n2 baseline")
    return RealN2Context(
        frontier=frontier,
        baseline_bundle=baseline_bundle,
        acquisition=acquisition,
        predictor_runtime=source.runtime,
        predictor_packet=source.member,
        target_frames=target_frames,
        target_labels=target_labels,
        target_poses=target_poses,
        custody={
            "baseline_archive": {
                "path": os.fspath(BASELINE_ARCHIVE),
                "bytes": len(baseline_archive),
                "sha256": _sha256(baseline_archive),
                "receipt_path": os.fspath(BASELINE_RECEIPT),
                "receipt_sha256": _sha256(baseline_receipt_bytes),
            },
            "directory_owned_P": {"bytes": len(source.member), "sha256": _sha256(source.member)},
            "explicit_runtime": {"bytes": len(source.runtime), "sha256": _sha256(source.runtime)},
            "target": target_custody,
            "materializer_target": materializer_target_custody,
            "realization_profile": realization_custody,
            "dense_encoder_evidence_serialized": False,
        },
    )


@dataclass(slots=True)
class _CachedFrames:
    frames: np.ndarray
    output_sha256: str
    output_nbytes: int


class EphemeralFrameCache:
    """Bounded stage-local cache joining receiver output to measurement."""

    def __init__(self) -> None:
        self._archives: dict[tuple[str, str], str] = {}
        self._frames: dict[str, _CachedFrames] = {}

    def record(self, request: TaskspaceReceiverRequestV1, frames: np.ndarray) -> _CachedFrames:
        immutable = np.ascontiguousarray(frames, dtype=np.uint8).copy()
        immutable.setflags(write=False)
        output_hash = _array_sha256(immutable)
        cached = self._frames.get(output_hash)
        if cached is None:
            cached = _CachedFrames(immutable, output_hash, immutable.nbytes)
            self._frames[output_hash] = cached
        elif not np.array_equal(cached.frames, immutable):
            raise TaskspaceA3N2AllocatorRunnerError("decoded-output SHA collision")
        key = (request.stage_id, request.archive_sha256)
        prior = self._archives.get(key)
        if prior is not None and prior != output_hash:
            raise TaskspaceA3N2AllocatorRunnerError("receiver replay changed cached output")
        self._archives[key] = output_hash
        return cached

    def consume(self, request: TaskspaceMeasurementRequestV1) -> np.ndarray:
        key = (request.stage_id, request.archive_sha256)
        output_hash = self._archives.get(key)
        if output_hash != request.decoded_output_sha256:
            raise TaskspaceA3N2AllocatorRunnerError("measurement request is not cache-bound to receiver output")
        cached = self._frames.get(output_hash)
        if cached is None or cached.output_nbytes != request.decoded_output_nbytes:
            raise TaskspaceA3N2AllocatorRunnerError("measurement decoded-output byte identity changed")
        result = cached.frames
        stage_keys = [row for row in self._archives if row[0] == request.stage_id]
        for row in stage_keys:
            del self._archives[row]
        live_hashes = set(self._archives.values())
        for digest in tuple(self._frames):
            if digest not in live_hashes:
                del self._frames[digest]
        return result


class RealEp725Receiver:
    """Public receiver adapter with exact directory-owned P revalidation."""

    def __init__(self, *, predictor_runtime: bytes, predictor_packet: bytes, timeout_seconds: float) -> None:
        self.predictor_runtime = predictor_runtime
        self.predictor_packet = predictor_packet
        self.timeout_seconds = timeout_seconds
        self.cache = EphemeralFrameCache()
        self.calls = 0

    def __call__(self, request: TaskspaceReceiverRequestV1) -> TaskspaceReceiverReceiptV1:
        parsed = parse_taskspace_outer_archive(
            request.archive_bytes,
            expected_encoding=request.encoding,
            expected_archive_sha256=request.archive_sha256,
            expected_member_sha256=request.member_sha256,
        )
        member = parse_taskspace_monolithic_pga_member(parsed.member_bytes)
        counted_p = member.section(TaskspaceMonolithicPGARole.PREDICTOR).payload
        if counted_p != self.predictor_packet:
            raise TaskspaceA3N2AllocatorRunnerError("allocator archive changed directory-owned counted P")
        decoded = receive_ep725_taskspace_monolithic_pga_archive(
            request.archive_bytes,
            predictor_runtime=self.predictor_runtime,
            pair_count=PAIR_COUNT,
            timeout_seconds=self.timeout_seconds,
            expected_encoding=request.encoding,
            expected_archive_sha256=request.archive_sha256,
            expected_member_sha256=request.member_sha256,
        )
        cached = self.cache.record(request, decoded.chronological_camera_frames)
        self.calls += 1
        return TaskspaceReceiverReceiptV1(
            stage_id=request.stage_id,
            encoding=request.encoding,
            archive_sha256=request.archive_sha256,
            archive_nbytes=request.archive_nbytes,
            member_sha256=request.member_sha256,
            member_nbytes=request.member_nbytes,
            decoded_output_sha256=cached.output_sha256,
            decoded_output_nbytes=cached.output_nbytes,
            receiver_receipt_payload=decoded.receipt.to_receipt_bytes(),
        )


class FrozenN2ScorerSession:
    """Load frozen models and target outputs once for all G7 measurements."""

    def __init__(self, context: RealN2Context, receiver: RealEp725Receiver) -> None:
        import torch
        from modules import DistortionNet, posenet_sd_path, segnet_sd_path

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self._torch = torch
        self._receiver = receiver
        self._target_labels = context.target_labels
        self._target_poses = context.target_poses
        self._scorer_custody = _measurement._verify_small_scorer_custody()
        self._model = DistortionNet().eval().to(torch.device("cpu"))
        self._model.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
        target_tensor = torch.from_numpy(np.ascontiguousarray(context.target_frames).copy())
        with torch.inference_mode():
            target_pose, target_seg = self._model(target_tensor)
        self._target_pose6 = target_pose["pose"][..., :6]
        self._target_argmax = target_seg.argmax(dim=1)
        target_argmax_numpy = np.ascontiguousarray(self._target_argmax.cpu().numpy(), dtype=np.uint8)
        if not np.array_equal(target_argmax_numpy, context.target_labels):
            raise TaskspaceA3N2AllocatorRunnerError("fresh SegNet target differs from frozen labels")
        target_pose_numpy = np.ascontiguousarray(self._target_pose6.cpu().numpy())
        max_abs = float(np.max(np.abs(target_pose_numpy.astype(np.float64) - context.target_poses)))
        scale = max(1.0, float(np.max(np.abs(context.target_poses))))
        atol = 2.0 * float(np.finfo(np.float32).eps) * scale
        if max_abs > atol:
            raise TaskspaceA3N2AllocatorRunnerError("fresh PoseNet target differs beyond two scaled fp32 eps")
        self.target_receipt = {
            "labels_u8_sha256": _array_sha256(target_argmax_numpy),
            "pose6_f32_sha256": _array_sha256(target_pose_numpy),
            "pose_cache_max_abs": max_abs,
            "pose_cache_atol": atol,
            "scorer_custody": self._scorer_custody,
        }
        self.calls = 0

    def __call__(self, request: TaskspaceMeasurementRequestV1) -> TaskspaceRealizedMeasurementReceiptV1:
        torch = self._torch
        frames = self._receiver.cache.consume(request)
        candidate_tensor = torch.from_numpy(np.ascontiguousarray(frames).copy())
        with torch.inference_mode():
            pose_a, seg_a = self._model(candidate_tensor)
            pose_b, seg_b = self._model(candidate_tensor)
        if not torch.equal(pose_a["pose"], pose_b["pose"]) or not torch.equal(seg_a, seg_b):
            raise TaskspaceA3N2AllocatorRunnerError("frozen scorer changed on candidate replay")
        candidate_pose6 = pose_a["pose"][..., :6]
        candidate_argmax = seg_a.argmax(dim=1)
        per_pair_pose = (candidate_pose6 - self._target_pose6).pow(2).mean(dim=1)
        per_pair_seg = (candidate_argmax != self._target_argmax).float().mean(dim=(1, 2))
        d_pose = float(per_pair_pose.mean().item())
        d_seg = float(per_pair_seg.mean().item())
        if not math.isfinite(d_pose) or not 0.0 <= d_seg <= 1.0:
            raise TaskspaceA3N2AllocatorRunnerError("frozen scorer emitted invalid component distances")
        receipt_object = {
            "schema": "tac.taskspace_a3_n2_callback_measurement.v1",
            "axis": AXIS,
            "stage_id": request.stage_id,
            "archive_sha256": request.archive_sha256,
            "member_sha256": request.member_sha256,
            "decoded_output_sha256": request.decoded_output_sha256,
            "receiver_receipt_sha256": request.receiver_receipt_sha256,
            "d_seg": d_seg,
            "d_pose": d_pose,
            "per_pair_d_seg": [float(value) for value in per_pair_seg.cpu().tolist()],
            "per_pair_d_pose": [float(value) for value in per_pair_pose.cpu().tolist()],
            "candidate_labels_u8_sha256": _array_sha256(
                np.ascontiguousarray(candidate_argmax.cpu().numpy(), dtype=np.uint8)
            ),
            "candidate_pose6_f32_sha256": _array_sha256(np.ascontiguousarray(candidate_pose6.cpu().numpy())),
            "double_forward_exact": True,
            "research_only": True,
        }
        self.calls += 1
        return TaskspaceRealizedMeasurementReceiptV1(
            stage_id=request.stage_id,
            archive_sha256=request.archive_sha256,
            archive_nbytes=request.archive_nbytes,
            member_sha256=request.member_sha256,
            member_nbytes=request.member_nbytes,
            decoded_output_sha256=request.decoded_output_sha256,
            decoded_output_nbytes=request.decoded_output_nbytes,
            receiver_receipt_sha256=request.receiver_receipt_sha256,
            d_seg=d_seg,
            d_pose=d_pose,
            measurement_receipt_payload=_canonical_json(receipt_object),
        )


def _section_summary(bundle: TaskspaceSectionBundleV1) -> list[dict[str, Any]]:
    return [
        {"role": role.value, "bytes": len(payload), "sha256": _sha256(payload)}
        for role, payload in bundle.role_payloads
    ]


def _state_summary(state: Any) -> dict[str, Any]:
    return {
        "stage_id": state.stage_id,
        "bundle_sha256": state.bundle.bundle_sha256,
        "sections": _section_summary(state.bundle),
        "member_bytes": state.member_nbytes,
        "member_sha256": state.archive_build.selected.member_sha256,
        "stored_archive_bytes": state.archive_build.stored.archive_nbytes,
        "stored_archive_sha256": state.archive_build.stored.archive_sha256,
        "deflated_archive_bytes": state.archive_build.deflated.archive_nbytes,
        "deflated_archive_sha256": state.archive_build.deflated.archive_sha256,
        "selected_encoding": state.archive_build.selected.encoding.value,
        "selected_archive_bytes": state.selected_archive_nbytes,
        "selected_archive_sha256": state.archive_build.selected.archive_sha256,
        "decoded_output_bytes": state.measurement_receipt.decoded_output_nbytes,
        "decoded_output_sha256": state.measurement_receipt.decoded_output_sha256,
        "d_seg": state.measurement_receipt.d_seg,
        "d_pose": state.measurement_receipt.d_pose,
        "measurement_receipt_sha256": state.measurement_receipt.measurement_receipt_sha256,
    }


def _allocation_summary(allocation: TaskspaceWholeArchiveAllocationV1) -> dict[str, Any]:
    audits: list[dict[str, Any]] = []
    for audit in allocation.proposal_audits:
        audits.append(
            {
                "proposal_id": audit.proposal_id,
                "proposal_index": audit.proposal_index,
                "accepted": audit.accepted,
                "decision": audit.decision,
                "before_stage_id": audit.before_stage_id,
                "trial_state": _state_summary(audit.trial_state),
                "accepted_prefix_state": (
                    None if audit.accepted_prefix_state is None else _state_summary(audit.accepted_prefix_state)
                ),
                "section_deltas": [
                    {
                        "role": row.role.value,
                        "before_bytes": row.before_nbytes,
                        "after_bytes": row.after_nbytes,
                        "delta_bytes": row.delta_nbytes,
                    }
                    for row in audit.section_deltas
                ],
                "raw_section_bytes_delta": audit.raw_section_bytes_delta,
                "monolithic_member_bytes_delta": audit.monolithic_member_bytes_delta,
                "stored_archive_bytes_delta": audit.stored_archive_bytes_delta,
                "deflated_archive_bytes_delta": audit.deflated_archive_bytes_delta,
                "selected_archive_bytes_delta": audit.selected_archive_bytes_delta,
                "before_selected_encoding": audit.before_selected_encoding.value,
                "trial_selected_encoding": audit.trial_selected_encoding.value,
                "score_transition": asdict(audit.score_transition),
            }
        )
    return {
        "baseline_state": _state_summary(allocation.baseline_state),
        "final_state": _state_summary(allocation.final_state),
        "accepted_proposal_ids": list(allocation.accepted_proposal_ids),
        "rejected_proposal_ids": list(allocation.rejected_proposal_ids),
        "proposal_audits": audits,
        "allocator_truth": asdict(allocation.truth),
    }


def _acquisition_summary(acquisition: ChronologicalA3AcquisitionV1) -> dict[str, Any]:
    return {
        "source_binding_sha256": acquisition.source_binding.binding_sha256,
        "pass_p0": {
            "bytes": len(acquisition.pass_p0_control.packet),
            "sha256": _sha256(acquisition.pass_p0_control.packet),
        },
        "proposals": [
            {
                "proposal_id": proposal.allocator_proposal.proposal_id,
                "interpretation": proposal.interpretation.value,
                "row_count": len(proposal.ranked_prefix),
                "packet_bytes": len(proposal.compiled_a3.packet),
                "packet_sha256": _sha256(proposal.compiled_a3.packet),
                "encoder_receipt": proposal.receipt.as_dict(),
            }
            for proposal in acquisition.proposals
        ],
        "true_counted_xip2_receiver_mode_present": False,
    }


def build_allocation_receipt(
    *,
    context: RealN2Context,
    allocation: TaskspaceWholeArchiveAllocationV1,
    row_counts: tuple[int, ...],
    receiver_calls: int,
    measurement_calls: int,
    scorer_target_receipt: dict[str, Any],
    command: tuple[str, ...],
) -> bytes:
    verify_dynamic_frontier_target_snapshot(context.frontier)
    final_archive = allocation.final_state.archive_build.selected.archive_bytes
    payload = {
        "schema": SCHEMA,
        "lane_id": LANE_ID,
        "axis": AXIS,
        "scope": "real ep725 n2 whole-object A3 allocation; advisory research only",
        "git_head_before_landing": _git_head(),
        "implementation_custody": _implementation_custody(),
        "command": list(command),
        "competitive_target": asdict(context.frontier),
        "row_counts": list(row_counts),
        "encoder_custody": context.custody,
        "acquisition": _acquisition_summary(context.acquisition),
        "allocation": _allocation_summary(allocation),
        "final_archive": {
            "bytes": len(final_archive),
            "sha256": _sha256(final_archive),
            "encoding": allocation.final_state.archive_build.selected.encoding.value,
            "not_a_candidate": True,
        },
        "scorer_session": {
            "seed": SEED,
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "model_loaded_once": True,
            "target_forward_once": True,
            "candidate_double_forward_per_measurement": True,
            "receiver_calls": receiver_calls,
            "measurement_calls": measurement_calls,
            "target_receipt": scorer_target_receipt,
        },
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform()},
        "open_blockers": [
            "n2_is_not_n600_evidence_and_cannot_rank_or_kill_the_family",
            "current_A_wire_has_no_true_counted_XIP2_warp_domain_pitch_ABI",
            "G8_same_class_realization_composition_is_separately_owned",
            "standalone_runtime_packaging_and_clean_inflate_are_not_closed",
            "authoritative_n600_contest_CPU_CUDA_exact_eval_not_run",
        ],
        "truth": TRUTH,
    }
    return _canonical_json(payload) + b"\n"


def parse_allocation_receipt(payload: bytes) -> dict[str, Any]:
    """Strict closed parse/re-emit for the durable research receipt."""

    if type(payload) is not bytes or not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise TaskspaceA3N2AllocatorRunnerError("allocation receipt requires exactly one newline")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TaskspaceA3N2AllocatorRunnerError(f"allocation receipt repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload[:-1].decode("ascii"), object_pairs_hook=unique_pairs)
    except TaskspaceA3N2AllocatorRunnerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TaskspaceA3N2AllocatorRunnerError("allocation receipt is not strict ASCII JSON") from exc
    expected = {
        "schema",
        "lane_id",
        "axis",
        "scope",
        "git_head_before_landing",
        "implementation_custody",
        "command",
        "competitive_target",
        "row_counts",
        "encoder_custody",
        "acquisition",
        "allocation",
        "final_archive",
        "scorer_session",
        "runtime",
        "open_blockers",
        "truth",
    }
    if type(value) is not dict or set(value) != expected or value.get("schema") != SCHEMA:
        raise TaskspaceA3N2AllocatorRunnerError("allocation receipt top-level schema is not closed V1")
    if value.get("lane_id") != LANE_ID or value.get("axis") != AXIS or value.get("truth") != TRUTH:
        raise TaskspaceA3N2AllocatorRunnerError("allocation receipt authority labels became permissive")
    if _canonical_json(value) + b"\n" != payload:
        raise TaskspaceA3N2AllocatorRunnerError("allocation receipt is not canonical on parse-back")
    return value


def write_once_or_equal(path: Path, payload: bytes) -> None:
    """Create one durable artifact, accepting only exact replay bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_stable_regular(path) != payload:
            raise TaskspaceA3N2AllocatorRunnerError(f"refusing to overwrite different artifact: {path}")
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
    except FileExistsError:
        if _read_stable_regular(path) != payload:
            raise TaskspaceA3N2AllocatorRunnerError(f"artifact race produced different bytes: {path}") from None


class _ContextBuilder(Protocol):
    def __call__(self, row_counts: tuple[int, ...], *, timeout_seconds: float) -> RealN2Context: ...


def run_real_allocation(
    *,
    row_counts: tuple[int, ...],
    timeout_seconds: float,
    output: Path,
    archive_output: Path,
    command: tuple[str, ...],
    context_builder: _ContextBuilder = _build_real_n2_context,
) -> dict[str, Any]:
    """Run one reviewed bounded allocation and persist only terminal evidence."""

    context = context_builder(row_counts, timeout_seconds=timeout_seconds)
    receiver = RealEp725Receiver(
        predictor_runtime=context.predictor_runtime,
        predictor_packet=context.predictor_packet,
        timeout_seconds=timeout_seconds,
    )
    scorer = FrozenN2ScorerSession(context, receiver)
    allocation = allocate_taskspace_whole_archive(
        context.baseline_bundle,
        context.acquisition.allocator_proposals,
        frontier_snapshot=context.frontier,
        receiver_callback=receiver,
        measurement_callback=scorer,
    )
    receipt = build_allocation_receipt(
        context=context,
        allocation=allocation,
        row_counts=row_counts,
        receiver_calls=receiver.calls,
        measurement_calls=scorer.calls,
        scorer_target_receipt=scorer.target_receipt,
        command=command,
    )
    parsed = parse_allocation_receipt(receipt)
    final_archive = allocation.final_state.archive_build.selected.archive_bytes
    if parsed["final_archive"] != {
        "bytes": len(final_archive),
        "sha256": _sha256(final_archive),
        "encoding": allocation.final_state.archive_build.selected.encoding.value,
        "not_a_candidate": True,
    }:
        raise TaskspaceA3N2AllocatorRunnerError("receipt final archive identity drifted")
    for path, payload in ((archive_output, final_archive), (output, receipt)):
        if path.exists() and _read_stable_regular(path) != payload:
            raise TaskspaceA3N2AllocatorRunnerError(f"write-once output already differs: {path}")
    write_once_or_equal(archive_output, final_archive)
    write_once_or_equal(output, receipt)
    return {
        "output": os.fspath(output),
        "receipt_bytes": len(receipt),
        "receipt_sha256": _sha256(receipt),
        "archive_output": os.fspath(archive_output),
        "archive_bytes": len(final_archive),
        "archive_sha256": _sha256(final_archive),
        "accepted": list(allocation.accepted_proposal_ids),
        "rejected": list(allocation.rejected_proposal_ids),
        "axis": AXIS,
        "score_claim": False,
    }


def _reviewed_command(
    *,
    row_counts: tuple[int, ...],
    timeout_seconds: float,
    output: Path,
    archive_output: Path,
) -> tuple[str, ...]:
    return (
        os.fspath(REPO / ".venv/bin/python"),
        os.fspath(Path(__file__).resolve()),
        "--execute-reviewed",
        "--row-counts",
        ",".join(str(value) for value in row_counts),
        "--timeout-seconds",
        str(timeout_seconds),
        "--output",
        os.fspath(output),
        "--archive-output",
        os.fspath(archive_output),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-counts", type=parse_row_counts, default=DEFAULT_ROW_COUNTS)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive-output", type=Path, default=DEFAULT_ARCHIVE_OUTPUT)
    parser.add_argument("--print-authorized-command", action="store_true")
    parser.add_argument("--execute-reviewed", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0.0:
        parser.error("--timeout-seconds must be finite and positive")
    command = _reviewed_command(
        row_counts=args.row_counts,
        timeout_seconds=args.timeout_seconds,
        output=args.output,
        archive_output=args.archive_output,
    )
    if args.print_authorized_command:
        print(shlex.join(command))
        return 0
    if not args.execute_reviewed:
        parser.error("real allocation is locked; review --print-authorized-command, then pass --execute-reviewed")
    summary = run_real_allocation(
        row_counts=args.row_counts,
        timeout_seconds=args.timeout_seconds,
        output=args.output,
        archive_output=args.archive_output,
        command=command,
    )
    print(_canonical_json(summary).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
