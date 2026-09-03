# SPDX-License-Identifier: MIT
"""Receiver-decoded n600 scorer realization for the retained QBZ1 archive.

The run is local macOS-CPU advisory evidence only.  It reads the exact retained
``archive.zip``, proves the archive receiver returns the retained QBF packet,
renders all 600 pairs through the camera round trip, runs both frozen scorers,
and retains every rendered/scorer payload in restartable 30-pair chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tarfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _root in (REPO, SRC):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments import ddm_qbflow_packet as qbf1
from experiments import ddm_qbt1_qbflow_trainer as qbt1
from experiments import ddm_qbz1_descent_rate_configuration as qbz1
from tac.gt_lineage import AUTHORITY_LINEAGE, assert_gt_lineage
from tac.scorer import load_differentiable_scorers

SCHEMA = "ddm_br2_born_object_scorer_realization.v1"
STAGE0_SCHEMA = "ddm_br2_receiver_custody.v1"
CHUNK_SCHEMA = "ddm_br2_scorer_chunk.v1"
N = 600
H, W = 384, 512
CHUNK_PAIRS = 30
RATE_DENOMINATOR = 37_545_489
ARCHIVE_BYTES = 106_832
ARCHIVE_SHA256 = "0e2ffdfaa5fe481d481dd70a9672a67f80b9aad7648f0c775fe2956dd3a4841d"
PACKET_SHA256 = "8c26684d33313ca44f3d4f02cf3c369f0f33d6de37eeba42ae4220faed3e6d38"
FIT_RESULT_SHA256 = "69b33e5d393deff7f1fcd76844cf524d7c19691f431aa399a876b2ad1ce227bf"
CONTAINER_SHA256 = "4c16e6c045768b2dee62f59ac9a2a27b7386280dfccff3dd5331a8d9509d95f7"
AFR1_SCORE = 0.14797617125559104
AFR1_CANONICAL_SHA256 = "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25"
NX1_RECALLED_AFR1_SHA256 = "cbb8e900b6d4accb2ec84506a247826502021698f37c13229f441385085c4b3d"
CLASS_NAMES = qbt1.PALETTE_CLASSES
OUTPUT_ROOT = Path("/Volumes/APDataStore/pact/ddm_br2")
INPUT_ROOT = Path("/Volumes/APDataStore/pact/ddm_qbz1_descent_rate_configuration")
FIT_RESULT = INPUT_ROOT / "FIT_RESULT.json"
CONTAINER = INPUT_ROOT / "final_reencode/reencode_payloads.tar"
ACTIVE_CLAIMS = REPO / ".omx/state/active_lane_dispatch_claims.md"
CANONICAL_POINTER = REPO / ".omx/state/canonical_frontier_pointer.json"
AFR1_MEMO = REPO / ".omx/research/ddm_afr1_pointer_move_and_no_toy_erratum_20260831.md"
MINIMUM_FREE_BYTES = 5_000_000_000


class BR2Error(RuntimeError):
    """Fail-closed refusal for BR2 custody, claim, or resume violations."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BR2Error(f"required file is absent: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def require_fact(path: Path, *, digest: str, size: int | None = None) -> dict[str, Any]:
    fact = file_fact(path)
    if fact["sha256"] != digest or (size is not None and fact["bytes"] != size):
        raise BR2Error(f"frozen input drifted: {path}")
    return fact


def storage_preflight(output: Path, *, required: int = MINIMUM_FREE_BYTES) -> dict[str, Any]:
    resolved = output.resolve()
    if resolved != OUTPUT_ROOT.resolve():
        raise BR2Error(f"output must be the chartered BR2 custody root: {resolved}")
    output.mkdir(parents=True, exist_ok=True)
    usage = os.statvfs(output)
    free = int(usage.f_bavail * usage.f_frsize)
    if free < required:
        raise BR2Error(f"AP storage preflight refused: free={free} required={required}")
    return {
        "root": str(resolved),
        "free_bytes": free,
        "required_free_bytes": required,
        "status": "PASS",
        "cleanup": "certify-or-block; no retained BR2 payload may be deleted by this runner",
    }


def tar_member_bytes(path: Path, name: str) -> bytes:
    with tarfile.open(path, mode="r") as archive:
        stream = archive.extractfile(name)
        if stream is None:
            raise BR2Error(f"retained container member is unreadable: {name}")
        return stream.read()


def retain_exact(path: Path, payload: bytes) -> dict[str, Any]:
    if path.exists():
        fact = file_fact(path)
        if fact["bytes"] != len(payload) or fact["sha256"] != sha256_bytes(payload):
            raise BR2Error(f"existing retained input differs; refusing overwrite: {path}")
        return fact
    return qbt1.atomic_bytes(path, payload)


def model_from_packet(packet: bytes) -> qbt1.QBFLOWTorch:
    decoded = qbf1.decode_packet(packet)
    expected = {
        qbf1.SECTION_CONFIG,
        qbf1.SECTION_MODEL,
        qbf1.SECTION_LATENT_META,
        qbf1.SECTION_LATENTS,
    }
    if set(decoded.sections) != expected:
        raise BR2Error("receiver-decoded QBF section set differs")
    params = qbf1.decode_model(decoded.sections[qbf1.SECTION_MODEL])
    meta = qbf1.decode_latent_meta(decoded.sections[qbf1.SECTION_LATENT_META])
    records = qbf1.decode_latent_table(decoded.sections[qbf1.SECTION_LATENTS])
    if set(records) != set(range(N)):
        raise BR2Error("receiver-decoded archive does not carry all 600 latent records")
    boundary = np.stack(
        [qbf1.dequantize(records[i][0], meta["boundary_scale"], (qbf1.BOUNDARY_LATENT_DIM,)) for i in range(N)]
    )
    interior = np.stack(
        [qbf1.dequantize(records[i][1], meta["interior_scale"], (qbf1.INTERIOR_LATENT_DIM,)) for i in range(N)]
    )
    return qbt1.QBFLOWTorch(params, boundary, interior)


def resolve_afr1_authority() -> dict[str, Any]:
    pointer = json.loads(CANONICAL_POINTER.read_text())
    pointer_sha = pointer["effective_frontier"]["archive_sha256"]
    memo_text = AFR1_MEMO.read_text()
    return {
        "nx1_memo_sha256": NX1_RECALLED_AFR1_SHA256,
        "canonical_pointer_sha256": pointer_sha,
        "afr1_authority_memo_contains_canonical_sha256": AFR1_CANONICAL_SHA256 in memo_text,
        "resolution": (
            "NX1 contains a non-authorizing SHA transcription error; the canonical pointer and "
            "AFR1 authority memo agree on cbb8d928...d405bf25. The discrepancy does not affect BR2 inputs."
        ),
        "status": "RESOLVED_NX1_TRANSCRIPTION_ERROR"
        if pointer_sha == AFR1_CANONICAL_SHA256 and AFR1_CANONICAL_SHA256 in memo_text
        else "UNRESOLVED_CURRENT_AUTHORITY_DRIFT",
        "pointer_fact": file_fact(CANONICAL_POINTER),
        "authority_memo_fact": file_fact(AFR1_MEMO),
    }


def stage0(output: Path, *, required_free_bytes: int = MINIMUM_FREE_BYTES) -> tuple[dict[str, Any], bytes]:
    storage = storage_preflight(output, required=required_free_bytes)
    fit_fact = require_fact(FIT_RESULT, digest=FIT_RESULT_SHA256)
    container_fact = require_fact(CONTAINER, digest=CONTAINER_SHA256)
    archive = tar_member_bytes(CONTAINER, "archive.zip")
    archive_repeat = tar_member_bytes(CONTAINER, "archive.repeat.zip")
    packet = tar_member_bytes(CONTAINER, "packet.qbf")
    packet_repeat = tar_member_bytes(CONTAINER, "packet.repeat.qbf")
    if len(archive) != ARCHIVE_BYTES or sha256_bytes(archive) != ARCHIVE_SHA256:
        raise BR2Error("chartered archive bytes or SHA-256 differ")
    if archive_repeat != archive or packet_repeat != packet:
        raise BR2Error("retained deterministic repeat differs")
    if sha256_bytes(packet) != PACKET_SHA256:
        raise BR2Error("retained packet SHA-256 differs")
    receiver_packet = qbf1.read_deterministic_archive(archive)
    if receiver_packet != packet:
        raise BR2Error("archive receiver output is not bit-identical to retained packet.qbf")
    if qbf1.deterministic_archive(receiver_packet) != archive:
        raise BR2Error("receiver decode/re-encode did not reconstruct archive.zip bit-identically")
    model_from_packet(receiver_packet)
    inputs = output / "inputs"
    retained = {
        "archive": retain_exact(inputs / "archive.zip", archive),
        "archive_repeat": retain_exact(inputs / "archive.repeat.zip", archive_repeat),
        "packet": retain_exact(inputs / "packet.qbf", packet),
        "packet_repeat": retain_exact(inputs / "packet.repeat.qbf", packet_repeat),
    }
    receipt = {
        "schema": STAGE0_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU advisory custody/receiver check]",
        "score_claim": False,
        "storage_preflight": storage,
        "fit_result": fit_fact,
        "container": container_fact,
        "retained_inputs": retained,
        "archive_repeat_bit_identity": True,
        "packet_repeat_bit_identity": True,
        "receiver_packet_bit_identity": True,
        "receiver_archive_roundtrip_bit_identity": True,
        "receiver_section_set_and_n600_latents_valid": True,
        "afr1_authority_resolution": resolve_afr1_authority(),
    }
    qbt1.atomic_json(output / "checkpoints/stage_00_receiver.json", receipt)
    return receipt, receiver_packet


def assert_active_scorer_claim(claim_id: str) -> dict[str, Any]:
    if not claim_id.startswith("ddm_br2_"):
        raise BR2Error("scorer claim must be a BR2-owned lane id")
    rows: list[dict[str, str]] = []
    for line in ACTIVE_CLAIMS.read_text().splitlines():
        if not line.startswith("|"):
            continue
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) == 8 and fields[0].startswith("20"):
            rows.append(
                {
                    "timestamp": fields[0],
                    "lane_id": fields[2],
                    "platform": fields[3],
                    "status": fields[6],
                    "raw": line,
                }
            )
    newest_by_lane: dict[str, dict[str, str]] = {}
    for row in rows:
        newest_by_lane.setdefault(row["lane_id"], row)
    own = newest_by_lane.get(claim_id)
    if own is None or own["platform"] != "local_macos_cpu" or not own["status"].startswith("active_"):
        raise BR2Error("newest BR2 row must be an active local_macos_cpu scorer claim")
    cutoff = datetime.now(UTC) - timedelta(hours=24)
    conflicts = []
    for lane_id, row in newest_by_lane.items():
        if lane_id == claim_id or "scorer" not in lane_id or not row["status"].startswith("active_"):
            continue
        timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        if timestamp >= cutoff:
            conflicts.append(row["raw"])
    if conflicts:
        raise BR2Error(f"another live scorer claim remains active: {conflicts}")
    return {"claim_id": claim_id, "registry": file_fact(ACTIVE_CLAIMS), "row": own["raw"]}


def chunk_rows(payload: Mapping[str, np.ndarray], expected_ids: Sequence[int]) -> list[dict[str, Any]]:
    ids = np.asarray(payload["pair_ids_i64"], dtype=np.int64)
    if ids.tolist() != list(expected_ids):
        raise BR2Error("retained chunk pair IDs differ from its stage boundary")
    predicted = np.asarray(payload["segnet_argmax_u8"], dtype=np.uint8)
    target = np.asarray(payload["target_argmax_u8"], dtype=np.uint8)
    pose = np.asarray(payload["posenet_pose6_f32"], dtype=np.float64)
    pose_target = np.asarray(payload["target_pose6_f32"], dtype=np.float64)
    rows = []
    for index, pair_id in enumerate(ids):
        per_class = []
        for class_id, class_name in enumerate(CLASS_NAMES):
            mask = target[index] == class_id
            per_class.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "target_pixels": int(mask.sum()),
                    "errors": int(((predicted[index] != target[index]) & mask).sum()),
                }
            )
        rows.append(
            {
                "pair_id": int(pair_id),
                "seg_errors": int((predicted[index] != target[index]).sum()),
                "seg_pixels": int(target[index].size),
                "pose_squared_error_sum": float(np.square(pose[index] - pose_target[index]).sum()),
                "pose_values": 6,
                "per_class": per_class,
            }
        )
    return rows


def load_chunk(path: Path, expected_ids: Sequence[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = {
        "pair_ids_i64",
        "camera_pair_u8",
        "segnet_logits_f16",
        "segnet_argmax_u8",
        "target_argmax_u8",
        "posenet_pose6_f32",
        "target_pose6_f32",
    }
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != required:
            raise BR2Error(f"retained chunk payload set differs: {path}")
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    return file_fact(path), chunk_rows(arrays, expected_ids)


def realize_chunk(
    output: Path,
    *,
    ids: Sequence[int],
    model: qbt1.QBFLOWTorch,
    gt: np.ndarray,
    pose_target: np.ndarray,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = output / "realized_n600" / f"scorer_pairs_{ids[0]:04d}_{ids[-1]:04d}.npz"
    checkpoint_path = output / "checkpoints" / f"stage_01_pairs_{ids[0]:04d}_{ids[-1]:04d}.json"
    if path.exists():
        fact, rows = load_chunk(path, ids)
        checkpoint = {"schema": CHUNK_SCHEMA, "resumed": True, "payload": fact, "pair_rows": rows}
        if not checkpoint_path.exists():
            qbt1.atomic_json(checkpoint_path, checkpoint)
        return checkpoint, rows
    camera_parts: list[np.ndarray] = []
    logits_parts: list[np.ndarray] = []
    argmax_parts: list[np.ndarray] = []
    pose_parts: list[np.ndarray] = []
    with torch.no_grad():
        for pair_id in ids:
            outputs = model(torch.tensor([pair_id], dtype=torch.long), height=H, width=W)
            camera = qbt1.roundtrip_to_camera_uint8_ste(outputs["rgb_pair_01"])
            pose6, logits = qbt1.scorer_forward(camera, posenet, segnet)
            camera_parts.append(camera[0].round().clamp(0, 255).to(torch.uint8).cpu().numpy())
            logits_parts.append(logits[0].cpu().numpy().astype("<f2"))
            argmax_parts.append(logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8))
            pose_parts.append(pose6[0].cpu().numpy().astype("<f4"))
    fact = qbt1.atomic_npz(
        path,
        pair_ids_i64=np.asarray(ids, dtype=np.int64),
        camera_pair_u8=np.stack(camera_parts),
        segnet_logits_f16=np.stack(logits_parts),
        segnet_argmax_u8=np.stack(argmax_parts),
        target_argmax_u8=np.asarray(gt[list(ids)], dtype=np.uint8),
        posenet_pose6_f32=np.stack(pose_parts),
        target_pose6_f32=np.asarray(pose_target[list(ids)], dtype="<f4"),
    )
    _, rows = load_chunk(path, ids)
    checkpoint = {"schema": CHUNK_SCHEMA, "resumed": False, "payload": fact, "pair_rows": rows}
    qbt1.atomic_json(checkpoint_path, checkpoint)
    return checkpoint, rows


def aggregate(pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seg_errors = sum(int(row["seg_errors"]) for row in pair_rows)
    seg_pixels = sum(int(row["seg_pixels"]) for row in pair_rows)
    pose_sse = sum(float(row["pose_squared_error_sum"]) for row in pair_rows)
    pose_values = sum(int(row["pose_values"]) for row in pair_rows)
    class_rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        target_pixels = sum(int(row["per_class"][class_id]["target_pixels"]) for row in pair_rows)
        errors = sum(int(row["per_class"][class_id]["errors"]) for row in pair_rows)
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "target_pixels": target_pixels,
                "errors": errors,
                "conditional_d_seg": None if target_pixels == 0 else errors / target_pixels,
                "contribution_to_global_d_seg": errors / seg_pixels,
            }
        )
    if seg_pixels != N * H * W or pose_values != N * 6:
        raise BR2Error("global scorer denominators differ from n600")
    if sum(row["target_pixels"] for row in class_rows) != seg_pixels:
        raise BR2Error("per-class target-pixel denominator does not partition global pixels")
    if sum(row["errors"] for row in class_rows) != seg_errors:
        raise BR2Error("per-class errors do not partition global SegNet errors")
    d_seg = seg_errors / seg_pixels
    d_pose = pose_sse / pose_values
    pose_term = math.sqrt(10.0 * d_pose)
    seg_term = 100.0 * d_seg
    rate = 25.0 * ARCHIVE_BYTES / RATE_DENOMINATOR
    distortion_budget = 0.12 - rate
    d_seg_budget_after_pose = (distortion_budget - pose_term) / 100.0
    score = seg_term + pose_term + rate
    return {
        "d_seg": d_seg,
        "d_pose": d_pose,
        "seg_term": seg_term,
        "pose_term": pose_term,
        "distortion": seg_term + pose_term,
        "rate": rate,
        "S": score,
        "sub_0_12_distortion_budget": distortion_budget,
        "sub_0_12_d_seg_budget_after_measured_pose": d_seg_budget_after_pose,
        "per_class": class_rows,
    }


def resume_storage_requirement(output: Path) -> int:
    paths = sorted((output / "realized_n600").glob("scorer_pairs_*.npz"))
    completed = len(paths)
    total_chunks = math.ceil(N / CHUNK_PAIRS)
    if completed == 0:
        return MINIMUM_FREE_BYTES
    average = sum(path.stat().st_size for path in paths) / completed
    remaining = max(0, total_chunks - completed)
    return max(500_000_000, math.ceil(1.25 * average * remaining + 100_000_000))


def run(output: Path, *, claim_id: str, launch_authorized: bool, resume_from: Path) -> dict[str, Any]:
    if not launch_authorized:
        raise BR2Error("scorer realization requires explicit launch authorization")
    if resume_from.resolve() != output.resolve():
        raise BR2Error("--resume-from must name the exact BR2 output root")
    claim = assert_active_scorer_claim(claim_id)
    required_free = resume_storage_requirement(output)
    stage0_receipt, packet = stage0(output, required_free_bytes=required_free)
    assert_gt_lineage(qbz1.GT_ARGMAX, required=AUTHORITY_LINEAGE, instrument="BR2 DALI partition")
    assert_gt_lineage(qbz1.GT_POSE6, required=AUTHORITY_LINEAGE, instrument="BR2 DALI pose")
    gt = np.load(qbz1.GT_ARGMAX, mmap_mode="r", allow_pickle=False)
    pose_target = np.load(qbz1.GT_POSE6, mmap_mode="r", allow_pickle=False)
    if gt.shape != (N, H, W) or gt.dtype != np.uint8 or pose_target.shape != (N, 6):
        raise BR2Error("registered scorer-target geometry differs")
    torch.manual_seed(qbz1.SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    model = model_from_packet(packet)
    model.eval()
    posenet, segnet = load_differentiable_scorers(REPO / "upstream", device=torch.device("cpu"))
    posenet.eval()
    segnet.eval()
    started = time.time()
    chunks = []
    pair_rows = []
    for start in range(0, N, CHUNK_PAIRS):
        ids = list(range(start, min(N, start + CHUNK_PAIRS)))
        checkpoint, rows = realize_chunk(
            output,
            ids=ids,
            model=model,
            gt=gt,
            pose_target=pose_target,
            posenet=posenet,
            segnet=segnet,
        )
        chunks.append(checkpoint["payload"])
        pair_rows.extend(rows)
        print(json.dumps({"realized_pairs": ids[-1] + 1, "n": N, "resumed": checkpoint["resumed"]}), flush=True)
    if len(pair_rows) != N or [row["pair_id"] for row in pair_rows] != list(range(N)):
        raise BR2Error("retained per-pair denominator is not exactly 600 ordered rows")
    pair_rows_fact = qbt1.atomic_json(output / "PAIR_ROWS.json", pair_rows)
    components = aggregate(pair_rows)
    verdict = "SUB-0.12-CANDIDATE" if components["S"] < 0.12 else "DISTORTION-REFUSED"
    result = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "verdict_scope": "INSTANCE (exact retained 106832-byte archive)",
        "verdict": verdict,
        "n": N,
        "pair_denominator": 600,
        "pixel_denominator": int(N * H * W),
        "pose_value_denominator": int(N * 6),
        "archive": file_fact(output / "inputs/archive.zip"),
        "source_facts": {
            "fit_result": require_fact(FIT_RESULT, digest=FIT_RESULT_SHA256),
            "container": require_fact(CONTAINER, digest=CONTAINER_SHA256),
            "gt_argmax": require_fact(qbz1.GT_ARGMAX, digest=qbz1.GT_ARGMAX_SHA256),
            "gt_pose6": require_fact(qbz1.GT_POSE6, digest=qbz1.GT_POSE6_SHA256),
            "runner": file_fact(Path(__file__).resolve()),
            "packet_module": file_fact(Path(qbf1.__file__).resolve()),
            "model_module": file_fact(Path(qbt1.__file__).resolve()),
        },
        "run_config": {
            "argv": list(sys.argv),
            "cwd": str(Path.cwd().resolve()),
            "seed": qbz1.SEED,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "platform": platform.platform(),
            "resume_from": str(resume_from.resolve()),
            "storage_required_at_resume_bytes": required_free,
        },
        "components": components,
        "delta_vs_0_12": components["S"] - 0.12,
        "delta_vs_afr1": components["S"] - AFR1_SCORE,
        "afr1_score": AFR1_SCORE,
        "gt_lineage": "DALI-aligned registered AUTHORITY_LINEAGE for partition and pose targets",
        "claim": claim,
        "stage0": stage0_receipt,
        "retained_chunks": chunks,
        "per_pair_rows": pair_rows_fact,
        "all_renders_logits_argmax_pose_and_targets_retained": True,
        "chunk_pairs": CHUNK_PAIRS,
        "elapsed_seconds": time.time() - started,
        "contest_eval_invocations": 0,
        "modal_invocations": 0,
        "pointer_moved": False,
        "boundaries": [
            "advisory macOS CPU row; not contest CPU/CUDA authority",
            "archive receiver is locally byte-closed, but no contest inflate runtime tree is sealed",
            "negative verdict, if any, is INSTANCE scope only",
        ],
    }
    qbt1.atomic_json(output / "checkpoints/stage_01_complete.json", result)
    qbt1.atomic_json(output / "REALIZED_RESULT.json", result)
    qbt1.atomic_json(
        output / "checkpoints/stage_02_verdict.json",
        {
            "schema": "ddm_br2_stage2_verdict.v1",
            "verdict": verdict,
            "components": components,
            "delta_vs_0_12": result["delta_vs_0_12"],
            "delta_vs_afr1": result["delta_vs_afr1"],
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    preflight = sub.add_parser("stage0", help="verify and retain receiver-decoded archive custody")
    preflight.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    realize = sub.add_parser("realize", help="run/resume the n600 frozen-scorer realization")
    realize.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    realize.add_argument("--resume-from", type=Path, required=True)
    realize.add_argument("--scorer-claim-id", required=True)
    realize.add_argument("--launch-authorized", action="store_true")
    args = parser.parse_args()
    if args.action == "stage0":
        result, _packet = stage0(args.output)
    else:
        result = run(
            args.output,
            claim_id=args.scorer_claim_id,
            launch_authorized=args.launch_authorized,
            resume_from=args.resume_from,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
