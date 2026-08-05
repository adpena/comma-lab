#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""JB1 derived PoseNet-Jacobian pose basis producer and smoke race harness.

This arm consumes the MS4 pose metric custody bundle instead of re-measuring
the n600 PoseNet targets.  The only scorer work here is the charter-approved
strided smoke: finite-difference PoseNet responses and terminal GN solves on
at most eight pairs, all through the terminal receiver's uint8 frame-0 path.

Authority: [macOS-CPU frozen-PoseNet advisory], score_claim=false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "4")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.optimization.terminal_pose_gn import (  # noqa: E402
    STALE_POSE_REHEARSAL_AUTHORITY_MARKER,
    CandidateArtifactScope,
    PoseAuthorityMode,
    PoseJointEvaluation,
    TerminalPoseCandidateArtifact,
    TerminalPoseGNConfig,
    TerminalPosePacketV1,
    realize_terminal_pose_pair,
    serialize_terminal_pose_packet,
    solve_terminal_pose_gn,
)
from tac.optimization.trajectory_stopping import (  # noqa: E402
    TrajectoryStoppingError,
    adjudicate_tail_slope,
)

SCHEMA = "ddm_jb1_derived_pose_basis.v1"
ROW_SCHEMA = "ddm_jb1_smoke_row.v1"
EVIDENCE_AXIS = "[macOS-CPU frozen-PoseNet advisory]"
POINTER_LINE = "S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved"
PAIR_COUNT = 600
CHUNK_PAIRS = 32
CAMERA_H = 874
CAMERA_W = 1164
PAIR_SHAPE = (2, CAMERA_H, CAMERA_W, 3)
SEED = 20260728
AMPLITUDE_Q8 = 512
GENERIC_SELECTOR = "eg1_generic_low_frequency_six_v1"
DERIVED_SELECTOR = "jb1_posenet_jacobian_svd_v1"
EXPECTED_MS4_POSE_SHA256 = "5e06cc78711a6ca6984c907600a25816cdecc6239903f782d85bcf9473a8f1bc"

DEFAULT_FRAME_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_ct1_campaign_telemetry_encode_20260725/"
    "e5a_runtime/output_identity/"
    "2a2c0367150f8c8c0953dfb5c1485e238bbc9995c37385e149e52ae22f506241"
)
DEFAULT_METRIC = Path(
    "/Volumes/VertigoDataTier/pact/"
    "ddm_ms4_metric_producers_and_measurement_20260724T042005Z/"
    "pose_metric_n600_batch32.json"
)
DEFAULT_UPSTREAM = Path("/Volumes/VertigoDataTier/pact/molab_witness_machine_upstream_20260709")
DEFAULT_OUTPUT = REPO / ".omx/research/ddm_jb1_20260810"


class JB1Error(RuntimeError):
    """Fail-closed custody, receiver, basis, or smoke race error."""


@dataclass(frozen=True)
class PoseMetric:
    centers: np.ndarray
    factors: np.ndarray
    payload: dict[str, Any]
    sha256: str
    bytes: int


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def _publish_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise JB1Error(f"immutable output differs: {path}")
        return
    tmp = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with tmp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _publish_json(path: Path, payload: Any) -> None:
    _publish_bytes(path, _canonical_json(payload))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise JB1Error(f"cannot load JSON: {path}") from exc
    if not isinstance(value, dict):
        raise JB1Error(f"JSON root must be an object: {path}")
    return value


def _load_pose_metric(path: Path, expected_sha256: str) -> PoseMetric:
    if not path.is_file():
        raise JB1Error(f"MS4 pose metric is unavailable: {path}")
    size, digest = _sha256_file(path)
    if digest != expected_sha256:
        raise JB1Error("MS4 pose metric SHA-256 differs")
    payload = _load_json(path)
    rows = payload.get("rows")
    if (
        payload.get("schema") != "ddm_pose_metric_custody.v1"
        or payload.get("metric_surface") != "EXACT_POSENET_OUTPUT_MSE_QUADRATIC"
        or payload.get("pair_count") != PAIR_COUNT
        or payload.get("scorer_batch_size") != 32
        or payload.get("output_dimension") != 6
        or payload.get("score_claim") is not False
        or payload.get("research_only") is not True
        or not isinstance(rows, list)
        or len(rows) != PAIR_COUNT
    ):
        raise JB1Error("MS4 pose metric schema/header differs")
    if [row.get("pair_id") for row in rows if isinstance(row, dict)] != list(range(PAIR_COUNT)):
        raise JB1Error("MS4 pose metric rows are not exact pair IDs 0..599")
    centers = np.asarray([row["center"] for row in rows], dtype=np.float64)
    factors = np.asarray([row["low_rank_factors"] for row in rows], dtype=np.float64)
    ranks = [row.get("rank") for row in rows]
    if centers.shape != (PAIR_COUNT, 6) or factors.shape != (PAIR_COUNT, 6, 6):
        raise JB1Error("MS4 pose metric tensor geometry differs")
    if ranks != [6] * PAIR_COUNT:
        raise JB1Error("MS4 pose metric rank is not six for every pair")
    if not np.isfinite(centers).all() or not np.isfinite(factors).all():
        raise JB1Error("MS4 pose metric contains nonfinite values")
    for row in rows:
        if row.get("converged") is not True or row.get("convergence_status") != "CONVERGED":
            raise JB1Error("MS4 pose metric contains non-converged rows")
        if float(row.get("tube_radius", 0.0)) <= 0.0:
            raise JB1Error("MS4 pose metric tube radius is not positive")
    return PoseMetric(centers=centers, factors=factors, payload=payload, sha256=digest, bytes=size)


def _load_posenet(upstream_root: Path):
    import torch
    from safetensors.torch import load_file

    modules_path = upstream_root / "modules.py"
    weights_path = upstream_root / "models/posenet.safetensors"
    if not modules_path.is_file() or not weights_path.is_file():
        raise JB1Error(f"frozen PoseNet custody is unavailable under {upstream_root}")
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))
    import modules as upstream_modules

    if Path(upstream_modules.__file__).resolve() != modules_path.resolve():
        raise JB1Error("imported non-custodied upstream modules.py")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    posenet = upstream_modules.PoseNet().eval().to("cpu")
    posenet.load_state_dict(load_file(str(weights_path), device="cpu"))
    for parameter in posenet.parameters():
        parameter.requires_grad = False
    return posenet, {
        "modules_path": str(modules_path),
        "modules_sha256": _sha256_file(modules_path)[1],
        "pose_model_path": str(weights_path),
        "pose_model_sha256": _sha256_file(weights_path)[1],
        "threads": 4,
        "device": "cpu",
    }


def _pose6(posenet: Any, pair: np.ndarray) -> np.ndarray:
    import torch

    tensor = torch.from_numpy(np.asarray(pair)[None]).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        output = posenet(posenet.preprocess_input(tensor))
    pose = output["pose"] if isinstance(output, dict) else output
    result = pose[0, :6].detach().cpu().numpy().astype(np.float64)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise JB1Error("frozen PoseNet returned malformed pose6")
    return result


def _load_pair(frame_root: Path, pair_id: int, custody_cache: dict[tuple[int, int], dict[str, Any]]) -> np.ndarray:
    if not 0 <= pair_id < PAIR_COUNT:
        raise JB1Error("pair_id outside n600")
    start = (pair_id // CHUNK_PAIRS) * CHUNK_PAIRS
    stop = min(start + CHUNK_PAIRS, PAIR_COUNT)
    raw = frame_root / f"pairs_{start:04d}_{stop:04d}.raw"
    sidecar_path = frame_root / f"pairs_{start:04d}_{stop:04d}.json"
    if not raw.is_file() or not sidecar_path.is_file():
        raise JB1Error(f"composed-frame chunk unavailable for pair {pair_id}")
    key = (start, stop)
    if key not in custody_cache:
        sidecar = _load_json(sidecar_path)
        expected_bytes = (stop - start) * int(np.prod(PAIR_SHAPE))
        if (
            sidecar.get("bytes") != expected_bytes
            or sidecar.get("pair_start") != start
            or sidecar.get("pair_stop") != stop
            or not isinstance(sidecar.get("sha256"), str)
        ):
            raise JB1Error(f"composed-frame sidecar differs: {sidecar_path}")
        raw_bytes, raw_sha256 = _sha256_file(raw)
        if raw_bytes != expected_bytes or raw_sha256 != sidecar["sha256"]:
            raise JB1Error(f"composed-frame raw custody differs: {raw}")
        custody_cache[key] = {
            "bytes": raw_bytes,
            "pair_range": [start, stop],
            "raw_path": str(raw),
            "raw_sha256": raw_sha256,
            "sidecar_path": str(sidecar_path),
            "sidecar_sha256": _sha256_file(sidecar_path)[1],
            "state_sha256": sidecar.get("state_sha256"),
        }
    memmap = np.memmap(raw, mode="r", dtype=np.uint8, shape=(stop - start, *PAIR_SHAPE))
    return np.array(memmap[pair_id - start], dtype=np.uint8, copy=True, order="C")


def _generic_basis(shape: tuple[int, int, int]) -> np.ndarray:
    height, width, channels = shape
    if channels != 3:
        raise JB1Error("terminal pose basis expects RGB")
    x = np.cos(2.0 * np.pi * (np.arange(width, dtype=np.float64) + 0.5) / width)
    y = np.cos(2.0 * np.pi * (np.arange(height, dtype=np.float64) + 0.5) / height)
    fields = np.zeros((6, height, width, channels), dtype=np.float32)
    for channel in range(3):
        fields[channel, :, :, channel] = x[None, :]
        fields[channel + 3, :, :, channel] = y[:, None]
    return fields


def _canonicalize_vt(vt: np.ndarray) -> np.ndarray:
    out = np.asarray(vt, dtype=np.float64).copy()
    for row in out:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    return out


def _derive_pair_basis(
    *,
    posenet: Any,
    parent: np.ndarray,
    factor: np.ndarray,
    generic: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    rank = generic.shape[0]
    jacobian = np.empty((6, rank), dtype=np.float64)
    zeros = np.zeros(rank, dtype=np.int16)
    for idx in range(rank):
        plus = zeros.copy()
        minus = zeros.copy()
        plus[idx] = 1
        minus[idx] = -1
        plus_pose = _pose6(posenet, realize_terminal_pose_pair(parent, generic, plus, amplitude_q8=AMPLITUDE_Q8))
        minus_pose = _pose6(posenet, realize_terminal_pose_pair(parent, generic, minus, amplitude_q8=AMPLITUDE_Q8))
        jacobian[:, idx] = (plus_pose - minus_pose) / 2.0
    weighted = np.asarray(factor, dtype=np.float64) @ jacobian
    u, singular_values, vt = np.linalg.svd(weighted, full_matrices=True)
    del u
    vt = _canonicalize_vt(vt)
    derived = np.tensordot(vt, generic.astype(np.float64), axes=(1, 0)).astype(np.float32)
    payload = {
        "generic_basis_sha256": _sha256_bytes(np.ascontiguousarray(generic).tobytes()),
        "jacobian_sha256": _sha256_bytes(np.ascontiguousarray(jacobian).tobytes()),
        "metric_weighted_jacobian_sha256": _sha256_bytes(np.ascontiguousarray(weighted).tobytes()),
        "right_singular_vectors_sha256": _sha256_bytes(np.ascontiguousarray(vt).tobytes()),
        "derived_basis_sha256": _sha256_bytes(np.ascontiguousarray(derived).tobytes()),
        "singular_values": [float(value) for value in singular_values],
        "singular_value_energy": [
            float(np.sum(singular_values[:r] ** 2) / max(np.sum(singular_values**2), 1.0e-300))
            for r in range(1, 7)
        ],
        "rank": int(rank),
        "basis_policy": (
            "pair-specific right-singular-vector rotation of the generic terminal fields; "
            "MS4 metric factors are shared/identity-like, but real receiver PoseNet response is pair-local"
        ),
    }
    return derived, payload


def _artifact_for(selector: str, rank: int, codes: np.ndarray) -> TerminalPoseCandidateArtifact:
    packet = serialize_terminal_pose_packet(
        TerminalPosePacketV1(
            seed=SEED,
            basis_selector=selector,
            amplitude_q8=AMPLITUDE_Q8,
            coefficients=np.asarray(codes, dtype=np.int16).reshape(1, rank),
        )
    )
    return TerminalPoseCandidateArtifact(
        outer_archive=packet,
        terminal_packet=packet,
        scope=CandidateArtifactScope.TERMINAL_SECTION_ONLY,
    )


def _terminal_stop_payload(result_payload: dict[str, Any], safety_relins: int) -> dict[str, Any]:
    steps = result_payload["steps"]
    values = [float(result_payload["joint_action_initial"])]
    for step in steps:
        values.append(float(step["joint_action_after"]))
    coords = list(range(len(values)))
    last_rejected = bool(steps and steps[-1]["admitted"] is False)
    if last_rejected:
        return {
            "stop_reason": "converged_solver_rejection",
            "tail_adjudication": None,
            "safety_bound_relinearizations": safety_relins,
            "bound_reported": False,
        }
    tail = None
    if len(values) >= 3:
        try:
            tail = adjudicate_tail_slope(coords, values).to_payload()
        except TrajectoryStoppingError as exc:
            tail = {"error": str(exc)}
    if len(steps) >= safety_relins:
        reason = "safety_bound_REPORTED"
        if isinstance(tail, dict) and tail.get("verdict") == "censored_still_descending":
            reason = "censored_still_descending"
        elif isinstance(tail, dict) and tail.get("verdict") == "ascending_past_min":
            reason = "ascending_past_min"
        elif isinstance(tail, dict) and tail.get("verdict") == "converged_plateau":
            reason = "converged_plateau_at_safety_boundary"
        return {
            "stop_reason": reason,
            "tail_adjudication": tail,
            "safety_bound_relinearizations": safety_relins,
            "bound_reported": reason in {"safety_bound_REPORTED", "censored_still_descending"},
        }
    return {
        "stop_reason": "solver_stopped_before_safety_bound",
        "tail_adjudication": tail,
        "safety_bound_relinearizations": safety_relins,
        "bound_reported": False,
    }


def _solve_arm(
    *,
    posenet: Any,
    pair_id: int,
    parent: np.ndarray,
    target: np.ndarray,
    basis: np.ndarray,
    selector: str,
    arm: str,
    rank: int,
    safety_relins: int,
) -> dict[str, Any]:
    rendered = np.asarray(basis[:rank], dtype=np.float32)

    def render_basis(seed: int, basis_selector: str, shape: tuple[int, int, int]) -> np.ndarray:
        if seed != SEED or basis_selector != selector or tuple(shape) != tuple(parent.shape[1:]):
            raise JB1Error("terminal basis render key differs")
        return rendered

    def artifact_for(codes: np.ndarray) -> TerminalPoseCandidateArtifact:
        return _artifact_for(selector, rank, codes)

    def scorer(pair: np.ndarray, artifact: TerminalPoseCandidateArtifact) -> PoseJointEvaluation:
        if not np.array_equal(pair[1], parent[1]):
            raise JB1Error("terminal solve changed frozen frame 1")
        pose = _pose6(posenet, pair)
        d_pose = float(np.mean((pose - target) ** 2, dtype=np.float64))
        return PoseJointEvaluation(
            pose6=pose,
            d_seg=0.0,
            d_pose=d_pose,
            archive_bytes=artifact.archive_bytes,
            archive_sha256=artifact.archive_sha256,
            sample_count=1,
            authority_marker=STALE_POSE_REHEARSAL_AUTHORITY_MARKER,
            custody_digest=None,
            realized=True,
        )

    result = solve_terminal_pose_gn(
        parent,
        target,
        render_basis,
        artifact_for,
        scorer,
        seed=SEED,
        basis_selector=selector,
        config=TerminalPoseGNConfig(
            relinearizations=safety_relins,
            damping=1.0e-3,
            amplitude_q8=AMPLITUDE_Q8,
            authority_mode=PoseAuthorityMode.STALE_REHEARSAL,
        ),
        pair_index=0,
    )
    payload = result.to_payload()
    final_artifact = artifact_for(result.final_coefficients)
    pose_initial = float(payload["pose_mse_initial"])
    pose_final = float(payload["pose_mse_final"])
    row = {
        "schema": ROW_SCHEMA,
        "arm": arm,
        "basis_selector": selector,
        "basis_sha256": _sha256_bytes(np.ascontiguousarray(rendered).tobytes()),
        "rank": rank,
        "pair_id": pair_id,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "sample_count": 1,
        "frame1_frozen_all_realized_verdicts": True,
        "receiver_path": "terminal_pose_gn.realize_terminal_pose_pair uint8 frame0; frame1 byte-frozen",
        "candidate_artifact_scope": "terminal-section-only; outer artifact equals packet",
        "final_packet": final_artifact.to_payload(),
        "pose_mse_initial": pose_initial,
        "pose_mse_final": pose_final,
        "delta_d_pose": pose_final - pose_initial,
        "pose_score_term_initial": math.sqrt(10.0 * pose_initial),
        "pose_score_term_final": math.sqrt(10.0 * pose_final),
        "delta_pose_score_term": math.sqrt(10.0 * pose_final) - math.sqrt(10.0 * pose_initial),
        "terminal_bytes": final_artifact.archive_bytes,
        "terminal_byte_score_term": 25.0 * final_artifact.archive_bytes / 37_545_489.0,
        "joint_action_initial": float(payload["joint_action_initial"]),
        "joint_action_final": float(payload["joint_action_final"]),
        "delta_joint_action": float(payload["joint_action_final"]) - float(payload["joint_action_initial"]),
        "strict_realized_improvement": bool(payload["strict_realized_improvement"]),
        "stop": _terminal_stop_payload(payload, safety_relins),
        "solver_payload": payload,
    }
    return row


def _rank_knee_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_rank: dict[int, list[float]] = {}
    for row in rows:
        if row["arm"] == "derived_jacobian":
            by_rank.setdefault(int(row["rank"]), []).append(float(row["joint_action_final"]))
    if sorted(by_rank) != [1, 2, 3, 4, 5, 6]:
        return {"status": "NOT_AVAILABLE", "reason": "derived rank sweep incomplete"}
    ranks = sorted(by_rank)
    means = [float(np.mean(by_rank[r])) for r in ranks]
    try:
        verdict = adjudicate_tail_slope(ranks, means).to_payload()
    except TrajectoryStoppingError as exc:
        verdict = {"error": str(exc)}
    best_rank = int(min(ranks, key=lambda r: float(np.mean(by_rank[r]))))
    return {
        "status": "SMOKE_ONLY_NOT_A_FINDING",
        "ranks": ranks,
        "mean_final_joint_action": means,
        "best_smoke_rank": best_rank,
        "adjudicate_tail_slope": verdict,
        "authority_note": (
            "Rank-knee protocol is encoded for jd1/full race; n<=8 smoke rows bank no scientific finding"
        ),
    }


def _parse_ranks(value: str) -> list[int]:
    ranks = [int(part) for part in value.split(",") if part.strip()]
    if not ranks or any(rank < 1 or rank > 6 for rank in ranks) or len(set(ranks)) != len(ranks):
        raise argparse.ArgumentTypeError("ranks must be unique integers in 1..6")
    return ranks


def _strided_pairs(n: int) -> list[int]:
    if not 1 <= n <= 8:
        raise JB1Error("smoke pair count must be in 1..8")
    stride = PAIR_COUNT // n
    return [min(i * stride, PAIR_COUNT - 1) for i in range(n)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_dir = output_dir / "smoke_rows"
    rows_dir.mkdir(parents=True, exist_ok=True)

    metric = _load_pose_metric(args.pose_metric, args.pose_metric_sha256)
    posenet, posenet_custody = _load_posenet(args.upstream_root)
    pair_ids = args.pair_ids if args.pair_ids else _strided_pairs(args.smoke_pair_count)
    if len(pair_ids) > 8:
        raise JB1Error("JB1 smoke may not exceed eight pairs")
    if pair_ids != sorted(set(pair_ids)) or any(not 0 <= pair < PAIR_COUNT for pair in pair_ids):
        raise JB1Error("pair IDs must be unique, sorted, and inside n600")

    generic_metric_shared = bool(np.allclose(metric.factors, metric.factors[0], rtol=0.0, atol=0.0))
    recall = {
        "sources_searched": [
            "MEMORY.md: #899/#904/frontier/lane hooks; no JB1-specific prior hit",
            ".omx/research/ddm_naive_engineering_audit_20260805.md",
            ".omx/research/codex_findings_ddm_ms3_metric_custody_bundle_20260724_codex.md",
            ".omx/research/codex_findings_ddm_ms4_metric_producers_and_measurement_20260724_codex.md",
            ".omx/research/ddm_mhar1_crosswalk_20260805.md row 3",
            "canonical equations query for pose/trajectory/rank/Jacobian terms",
            ".omx/research/CANONICAL_RESEARCH_INDEX* and sub015_DAG_* targeted queries",
            "task ledgers .omx/state/canonical_task_status.jsonl and harness_tasklist_bridge_20260803.jsonl",
            "code search over terminal_pose_gn, trajectory_stopping, ms4 pose metric consumers",
        ],
        "found_beyond_charter_seeds": [
            "trajectory_derived_stopping_law_v1 is implemented in tac.optimization.trajectory_stopping",
            "terminal_pose_gn already emits marginal_value and stop-on-rejection proof; use it rather than a new GN solver",
            "ms4 pose factors are identical I/sqrt(6) across all 600 pairs, so metric-space basis is shared while receiver response is pair-local",
            "ddm_pc1 and rg4 consumers validate the same pose_metric_n600_batch32.json by SHA and schema",
        ],
        "plan_change": (
            "Use per-pair SVD of the real receiver finite-difference response F@J, with F from MS4 custody; "
            "encode the rank-knee protocol via adjudicate_tail_slope and label any smoke safety cap as reported."
        ),
    }

    charter_echo = {
        "schema": "ddm_jb1_charter_echo.v1",
        "charter": str(REPO / ".omx/tmp/codex_runs/jb1_prompt.md"),
        "common_contract": str(REPO / ".omx/tmp/codex_runs/_common_contract.md"),
        "boundaries": {
            "axis": EVIDENCE_AXIS,
            "score_claim": False,
            "launches": False,
            "full_race_fired": False,
            "live_w4_w4m_touched": False,
            "sealed_trainer_touched": False,
            "smoke_pair_ids": pair_ids,
        },
        "recall_evidence": recall,
    }
    _publish_json(output_dir / "charter_echo.json", charter_echo)

    frame_custody_cache: dict[tuple[int, int], dict[str, Any]] = {}
    producer_rows: list[dict[str, Any]] = []
    smoke_rows: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        parent = _load_pair(args.frame_root, pair_id, frame_custody_cache)
        target = metric.centers[pair_id]
        generic = _generic_basis(tuple(parent.shape[1:]))
        derived, producer = _derive_pair_basis(
            posenet=posenet,
            parent=parent,
            factor=metric.factors[pair_id],
            generic=generic,
        )
        producer.update(
            {
                "schema": "ddm_jb1_pair_basis_producer.v1",
                "pair_id": pair_id,
                "target_center_sha256": _sha256_bytes(np.ascontiguousarray(target).tobytes()),
                "ms4_factor_sha256": _sha256_bytes(np.ascontiguousarray(metric.factors[pair_id]).tobytes()),
                "metric_factor_shared_across_n600": generic_metric_shared,
                "source_pose_metric_sha256": metric.sha256,
                "evidence_axis": EVIDENCE_AXIS,
                "score_claim": False,
            }
        )
        producer_rows.append(producer)
        _publish_json(output_dir / "producer_rows" / f"pair_{pair_id:04d}.json", producer)

        arms: list[tuple[str, str, np.ndarray, int]] = [
            ("generic", GENERIC_SELECTOR, generic, 6),
        ]
        for rank in args.ranks:
            arms.append(("derived_jacobian", f"{DERIVED_SELECTOR}_rank{rank}", derived, rank))
        for arm, selector, basis, rank in arms:
            row_path = rows_dir / f"pair_{pair_id:04d}_{arm}_rank{rank}.json"
            if row_path.exists():
                row = _load_json(row_path)
                if row.get("schema") != ROW_SCHEMA or row.get("pair_id") != pair_id or row.get("rank") != rank:
                    raise JB1Error(f"existing smoke row differs: {row_path}")
            else:
                row = _solve_arm(
                    posenet=posenet,
                    pair_id=pair_id,
                    parent=parent,
                    target=target,
                    basis=basis,
                    selector=selector,
                    arm=arm,
                    rank=rank,
                    safety_relins=args.safety_relins,
                )
                _publish_json(row_path, row)
            smoke_rows.append(row)
            print(
                json.dumps(
                    {
                        "pair": pair_id,
                        "arm": arm,
                        "rank": rank,
                        "d_pose": row["pose_mse_final"],
                        "stop": row["stop"]["stop_reason"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    producer_receipt = {
        "schema": "ddm_jb1_basis_producer_receipt.v1",
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "research_only": True,
        "pose_metric": {
            "path": str(args.pose_metric),
            "bytes": metric.bytes,
            "sha256": metric.sha256,
            "schema": metric.payload["schema"],
            "metric_surface": metric.payload["metric_surface"],
            "quadratic_identity": metric.payload.get("quadratic_identity"),
            "metric_factor_shared_across_n600": generic_metric_shared,
        },
        "producer_rows": producer_rows,
        "basis_family": {
            "generic_control": GENERIC_SELECTOR,
            "derived": DERIVED_SELECTOR,
            "rank_sweep": args.ranks,
            "derivation": "top-r right singular vectors of F@J; F=MS4 low_rank_factors, J=real receiver finite-difference PoseNet6 response",
        },
    }
    _publish_json(output_dir / "producer_receipt.json", producer_receipt)

    rank_knee = _rank_knee_payload(smoke_rows)
    smoke_receipt = {
        "schema": "ddm_jb1_smoke_receipt.v1",
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "research_only": True,
        "promotion_eligible": False,
        "pointer_moved": False,
        "pointer": POINTER_LINE,
        "full_race_fired": False,
        "consumer": "jd1 / #366 ticket regeneration at w4/w4m ep1363 boundary",
        "sample_scope": {
            "pair_ids": pair_ids,
            "n": len(pair_ids),
            "selection": "strided over 0..599, not prefix",
            "finding_authority": "BUILD_VERIFICATION_ONLY; n<=8 banks no scientific finding",
        },
        "terminal_solver": {
            "module": "tac.optimization.terminal_pose_gn",
            "safety_relins": args.safety_relins,
            "stop_adjudicator": "tac.optimization.trajectory_stopping.adjudicate_tail_slope",
            "safety_bound_policy": "reported as safety_bound_REPORTED/censored, never convergence",
            "amplitude_q8": AMPLITUDE_Q8,
        },
        "posenet_custody": posenet_custody,
        "frame_custody": list(frame_custody_cache.values()),
        "rows": smoke_rows,
        "rank_knee_protocol": rank_knee,
        "wall_seconds": time.time() - started,
    }
    _publish_json(output_dir / "smoke_receipt.json", smoke_receipt)

    next_text = "\n".join(
        [
            "# NEXT_IF_RESUMED — JB1",
            "",
            "Boundary fire: at the w4/w4m ep1363 adjudication turn, regenerate jd1/#366 against the winner.",
            "",
            "Fire order:",
            "1. Run the full race on the w4/w4m winner, not this stale smoke parent.",
            "2. Use the `jb1_posenet_jacobian_svd_v1` rank sweep 1..6 against the eg1 generic rank-6 control.",
            "3. Stop by `tac.optimization.trajectory_stopping.adjudicate_tail_slope`; any safety cap is a typed `safety_bound_REPORTED`, not convergence.",
            "4. Preserve per-pair basis producer rows and terminal solve ledgers; n<=8 smoke rows here are build-verification only.",
            "",
            "Queued-with-fire-order: full race vs w4/w4m WINNER, consumer jd1.",
            "",
        ]
    )
    _publish_bytes(output_dir / "NEXT_IF_RESUMED.md", next_text.encode("utf-8"))

    done = {
        "schema": "ddm_jb1_done_receipt.v1",
        "done": True,
        "score_claim": False,
        "full_race_fired": False,
        "producer_receipt": str(output_dir / "producer_receipt.json"),
        "smoke_receipt": str(output_dir / "smoke_receipt.json"),
        "next_if_resumed": str(output_dir / "NEXT_IF_RESUMED.md"),
        "pointer": POINTER_LINE,
    }
    _publish_json(output_dir / ".done", done)
    return smoke_receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-root", type=Path, default=DEFAULT_FRAME_ROOT)
    parser.add_argument("--pose-metric", type=Path, default=DEFAULT_METRIC)
    parser.add_argument("--pose-metric-sha256", default=EXPECTED_MS4_POSE_SHA256)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--smoke-pair-count", type=int, default=8)
    parser.add_argument("--pair-ids", type=lambda s: [int(x) for x in s.split(",") if x], default=None)
    parser.add_argument("--ranks", type=_parse_ranks, default=_parse_ranks("1,2,3,4,5,6"))
    parser.add_argument("--safety-relins", type=int, default=2)
    args = parser.parse_args()
    if args.safety_relins < 2:
        raise SystemExit("--safety-relins must be >=2 (terminal_pose_gn contract)")
    if not str(args.output_dir.resolve()).startswith(str((REPO / ".omx/research").resolve())):
        raise SystemExit("--output-dir must stay under .omx/research for persisted evidence")
    receipt = run(args)
    print(
        json.dumps(
            {
                "receipt": str(args.output_dir / "smoke_receipt.json"),
                "pairs": receipt["sample_scope"]["pair_ids"],
                "rank_knee_status": receipt["rank_knee_protocol"]["status"],
                "score_claim": receipt["score_claim"],
                "full_race_fired": receipt["full_race_fired"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
