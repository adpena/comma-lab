#!/usr/bin/env python3
"""Realized JS5 acceptance over retained EC1 event-coordinate proposals.

This is a consume-only adapter.  It never invokes the EC1 producer and never
regenerates a semantic payload.  Each retained proposal camera is transported
onto the sealed CP135 SegNet/PoseNet custody planes, scored on CPU, and admitted
only when the JS5 strict pose gate passes and robust flips improve.  Every new
scorer output is retained under the JS6 run directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_js2b_edge_conditioning_relative_gauge as js2b
from experiments import ddm_js3_learned_implicit_conditioning as js3
from experiments import ddm_js5_projector_distilled_conditioning as js5

DEFAULT_PROPOSAL_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/"
    "follow_on/realized_acceptance_200"
)
DEFAULT_OUTPUT = Path("/Volumes/APDataStore/pact/ddm_js6_20260812")
EXPECTED_SOURCE_ARCHIVE_SHA256 = js2b.BASE_ARCHIVE_SHA256
EXPECTED_PROPOSALS = 200
EXPECTED_SAMPLE_N = 32
EXPECTED_JS5_SOURCE_SHA256 = "981240bef78e195595978241b383ea4b5ad4ac23ab321ffd4609f6c645dc5d80"
EXPECTED_CONSUMER_PAYLOADS = (
    "event.ec1p",
    "candidate_tokens.uint8.npy",
    "camera.uint8.npy",
    "scorer_input.float16.npy",
    "event.ec1p.br",
    "event.ec1p.xz",
)
EVENT_FAMILIES = {
    "boundary_offset": "boundary",
    "lane_program_delta": "lane",
    "island_birth": "island",
    "island_death": "island",
}
AXIS = js3.AXIS
POSE_GATE = js3.POSE_GUARD
ECONOMICS_BAR = 1.28
EXACT_ECONOMICS_BAR = 1.2731082153
DEFAULT_REQUIRED_FREE_BYTES = 2_000_000_000


class JS6Error(RuntimeError):
    """A custody, CPU, resume, payload, or acceptance invariant failed."""


@dataclasses.dataclass(frozen=True)
class Store:
    root: Path
    state: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    custody: dict[str, Any]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def require_record(record: dict[str, Any], *, beneath: Path | None = None) -> Path:
    """Re-derive a file record and optionally require proposal-store custody."""
    path = Path(str(record["path"])).resolve()
    if beneath is not None and not _is_relative_to(path, beneath.resolve()):
        raise JS6Error(f"payload escaped proposal-owned custody: {path}")
    if not path.is_file():
        raise JS6Error(f"missing retained payload: {path}")
    if js3.file_record(path) != record:
        raise JS6Error(f"retained payload differs from receipt: {path}")
    return path


def load_store(root: Path) -> Store:
    """Verify all 200 proposal IDs, hashes, receipts, and proposal-owned bytes."""
    root = root.resolve()
    state_path = root / "state.json"
    if not state_path.is_file():
        raise JS6Error(f"missing EC1 store state: {state_path}")
    state = json.loads(state_path.read_text())
    if (
        state.get("schema") != "ddm_js5_realized_acceptance_200_store.v1"
        or state.get("status") != "PRODUCED_NOT_ACCEPTANCE_TESTED"
        or bool(state.get("acceptance_tested"))
        or int(state.get("proposal_count", -1)) != EXPECTED_PROPOSALS
        or int(state.get("receiver_effective_count", -1)) != EXPECTED_PROPOSALS
        or state.get("source_archive_sha256") != EXPECTED_SOURCE_ARCHIVE_SHA256
    ):
        raise JS6Error("EC1 store state differs from the JS6 charter")
    sample = tuple(int(value) for value in state.get("sample", []))
    if len(sample) != EXPECTED_SAMPLE_N or len(set(sample)) != EXPECTED_SAMPLE_N:
        raise JS6Error("EC1 proposal sample is not the sealed unique n32 sample")
    index_path = require_record(state["proposal_index"], beneath=root)
    index_rows = tuple(json.loads(line) for line in index_path.read_text().splitlines() if line.strip())
    if len(index_rows) != EXPECTED_PROPOSALS:
        raise JS6Error(f"proposal index has {len(index_rows)} rows, expected {EXPECTED_PROPOSALS}")

    ids: set[str] = set()
    event_hashes: set[str] = set()
    family_counts: Counter[str] = Counter()
    verified_files = 0
    verified_bytes = 0
    for ordinal, row in enumerate(index_rows):
        proposal_id = str(row.get("proposal_id", ""))
        if not proposal_id or proposal_id in ids:
            raise JS6Error(f"duplicate or empty proposal ID at index {ordinal}")
        ids.add(proposal_id)
        if (
            row.get("source_archive_sha256") != EXPECTED_SOURCE_ARCHIVE_SHA256
            or int(row.get("pair", -1)) not in sample
            or not bool(row.get("receiver_effective"))
            or not bool(row.get("parse_back_exact"))
            or bool(row.get("acceptance_tested"))
        ):
            raise JS6Error(f"proposal {proposal_id} failed the producer-state contract")
        event_type = str(row.get("event_type", ""))
        if event_type not in EVENT_FAMILIES:
            raise JS6Error(f"proposal {proposal_id} has unknown event type {event_type!r}")
        family_counts[EVENT_FAMILIES[event_type]] += 1
        proposal_root = (root / "proposals" / proposal_id).resolve()
        receipt_path = require_record(row["proposal_receipt"], beneath=proposal_root)
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("proposal_id") != proposal_id:
            raise JS6Error(f"proposal receipt ID differs for {proposal_id}")
        indexed_receipt = {key: value for key, value in row.items() if key != "proposal_receipt"}
        if indexed_receipt != receipt:
            raise JS6Error(f"proposal index row differs from owned receipt for {proposal_id}")
        consumer = row.get("consumer_payloads", {})
        if tuple(sorted(consumer)) != tuple(sorted(EXPECTED_CONSUMER_PAYLOADS)):
            raise JS6Error(f"proposal {proposal_id} consumer payload set differs")
        for name in EXPECTED_CONSUMER_PAYLOADS:
            payload_path = require_record(consumer[name], beneath=proposal_root)
            if payload_path.name != name:
                raise JS6Error(f"proposal {proposal_id} payload name differs for {name}")
            verified_files += 1
            verified_bytes += int(consumer[name]["bytes"])
        event_sha = str(consumer["event.ec1p"]["sha256"])
        if event_sha in event_hashes or event_sha != str(row["payload"]["sha256"]):
            raise JS6Error(f"proposal {proposal_id} event hash is duplicate or mismatched")
        event_hashes.add(event_sha)

    custody = {
        "schema": "ddm_js6_input_custody.v1",
        "proposal_store_state": js3.file_record(state_path),
        "proposal_index": js3.file_record(index_path),
        "proposal_count": len(index_rows),
        "unique_proposal_ids": len(ids),
        "unique_event_payload_sha256": len(event_hashes),
        "family_counts": dict(sorted(family_counts.items())),
        "proposal_owned_payload_files_verified": verified_files,
        "proposal_owned_payload_bytes_verified": verified_bytes,
        "source_archive": js3.file_record(js2b.BASE_ARCHIVE),
        "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        "payload_regeneration": False,
        "sample": list(sample),
    }
    if custody["source_archive"]["sha256"] != EXPECTED_SOURCE_ARCHIVE_SHA256:
        raise JS6Error("CP135 source archive differs from the EC1 store pin")
    return Store(root=root, state=state, rows=index_rows, custody=custody)


def projected_pose_delta(candidate_error: float, base_error: float, weight: int) -> float:
    """Change in the sealed stratified-n32 pose mean for one changed pair."""
    values = (candidate_error, base_error)
    if any(not math.isfinite(value) for value in values) or weight <= 0:
        raise ValueError("pose errors must be finite and weight must be positive")
    return float((candidate_error - base_error) * weight / js2b.N)


def bytes_per_robust_flip(coded_bytes: int, projected_robust_delta: int) -> float | None:
    if coded_bytes < 0:
        raise ValueError("coded bytes must be non-negative")
    return coded_bytes / -projected_robust_delta if projected_robust_delta < 0 else None


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _runner_source_custody(output: Path) -> dict[str, Any]:
    source = Path(__file__).resolve()
    digest = js3.sha256_file(source)
    retained = output / "source_custody" / "by_sha256" / f"{digest}.py"
    if not retained.is_file():
        js3.atomic_bytes(retained, source.read_bytes())
    if js3.file_record(retained)["sha256"] != digest:
        raise JS6Error("JS6 immutable runner source custody differs")
    return js3.file_record(retained)


def preflight(output: Path, store: Store, required_free_bytes: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    js5_source = REPO / "experiments/ddm_js5_projector_distilled_conditioning.py"
    if js3.sha256_file(js5_source) != EXPECTED_JS5_SOURCE_SHA256:
        raise JS6Error("JS5 acceptance source differs from the measured custody pin")
    row = {
        "schema": "ddm_js6_preflight.v1",
        "axis": AXIS,
        "score_claim": False,
        "cpu_only": True,
        "mps_touched": False,
        "platform": platform.platform(),
        "required_free_bytes": required_free_bytes,
        "free_bytes": usage.free,
        "storage_pass": usage.free >= required_free_bytes,
        "proposal_store": str(store.root),
        "proposal_store_state": store.custody["proposal_store_state"],
        "js5_source": js3.file_record(js5_source),
        "js3_source": js3.file_record(REPO / "experiments/ddm_js3_learned_implicit_conditioning.py"),
        "js2b_source": js3.file_record(REPO / "experiments/ddm_js2b_edge_conditioning_relative_gauge.py"),
        "runner_source": _runner_source_custody(output),
        "git_head": _git_head(),
        "payload_regeneration": False,
    }
    js3.atomic_json(output / "preflight" / "PREFLIGHT.json", row)
    js3.atomic_json(output / "inputs" / "PROPOSAL_STORE_CUSTODY.json", store.custody)
    if not row["storage_pass"]:
        raise JS6Error("APDataStore free-space preflight failed")
    return row


def _run_config(args: argparse.Namespace, store: Store, preflight_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "ddm_js6_run_config.v1",
        "proposal_store": str(store.root),
        "proposal_index_sha256": store.custody["proposal_index"]["sha256"],
        "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        "sample": store.state["sample"],
        "axis": AXIS,
        "pose_gate_delta_lt": POSE_GATE,
        "robust_delta_required_lt": 0,
        "economics_bar_bytes_per_robust_flip_lte": ECONOMICS_BAR,
        "exact_cp135_marginal_bar": EXACT_ECONOMICS_BAR,
        "max_wall_seconds": args.max_wall_seconds,
        "required_free_bytes": args.required_free_bytes,
        "cpu_only": True,
        "payload_regeneration": False,
        "runner_source_sha256": preflight_row["runner_source"]["sha256"],
    }


def _initial_state(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "ddm_js6_state.v1",
        "status": "READY",
        "complete": False,
        "next_ordinal": 0,
        "pose_accepted": 0,
        "bare_admissions": 0,
        "config": config,
        "resumable": True,
        "score_claim": False,
    }


def load_or_start_state(output: Path, config: dict[str, Any], *, resume: bool) -> dict[str, Any]:
    state_path = output / "state.json"
    if state_path.is_file():
        if not resume:
            raise JS6Error(f"state already exists; pass --resume: {state_path}")
        state = json.loads(state_path.read_text())
        if state.get("config") != config:
            raise JS6Error("resume config differs from retained state")
        return state
    if resume:
        raise JS6Error("--resume requested but state.json does not exist")
    state = _initial_state(config)
    js3.atomic_json(state_path, state)
    return state


def _proposal_result_path(output: Path, ordinal: int, proposal_id: str) -> Path:
    return output / "proposals" / f"{ordinal:04d}_{proposal_id}" / "RESULT.json"


def _recover_completed_rows(output: Path, store: Store, state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ordinal, proposal in enumerate(store.rows):
        path = _proposal_result_path(output, ordinal, str(proposal["proposal_id"]))
        if not path.is_file():
            break
        row = json.loads(path.read_text())
        if int(row.get("ordinal", -1)) != ordinal or row.get("proposal_id") != proposal["proposal_id"]:
            raise JS6Error(f"retained proposal result order differs at {ordinal}")
        rows.append(row)
        if bool(row.get("bare_admission")):
            break
    if int(state.get("next_ordinal", -1)) > len(rows):
        raise JS6Error("state points beyond retained per-proposal results")
    state.update(
        next_ordinal=len(rows),
        pose_accepted=sum(bool(row["pose_gate_pass"]) for row in rows),
        bare_admissions=sum(bool(row["bare_admission"]) for row in rows),
    )
    js3.atomic_json(output / "state.json", state)
    return rows


def _assert_cpu_context(context: Any) -> None:
    for name, model in (("SegNet", context.segnet), ("PoseNet", context.posenet)):
        devices = {parameter.device.type for parameter in model.parameters()}
        if devices != {"cpu"}:
            raise JS6Error(f"{name} is not CPU-only: {sorted(devices)}")


def score_proposal(context: Any, store: Store, proposal: dict[str, Any], ordinal: int, output: Path) -> dict[str, Any]:
    """Score one retained camera without invoking the EC1 renderer."""
    proposal_id = str(proposal["proposal_id"])
    pair_id = int(proposal["pair"])
    sample = np.asarray(context.sample)
    matches = np.flatnonzero(sample == pair_id)
    if len(matches) != 1:
        raise JS6Error(f"proposal pair {pair_id} is not unique in the sealed sample")
    slot = int(matches[0])
    weight = int(context.sample_weights[slot])
    family = EVENT_FAMILIES[str(proposal["event_type"])]
    consumer = proposal["consumer_payloads"]
    camera_path = require_record(consumer["camera.uint8.npy"], beneath=store.root)
    scorer_path = require_record(consumer["scorer_input.float16.npy"], beneath=store.root)
    event_path = require_record(consumer["event.ec1p"], beneath=store.root)
    brotli_path = require_record(consumer["event.ec1p.br"], beneath=store.root)
    camera = np.load(camera_path, allow_pickle=False)
    retained_scorer = np.load(scorer_path, allow_pickle=False)
    if camera.shape != (js2b.CAMERA_H, js2b.CAMERA_W, 3) or camera.dtype != np.uint8:
        raise JS6Error(f"proposal {proposal_id} camera geometry differs")
    if retained_scorer.shape != (3, js2b.H, js2b.W) or retained_scorer.dtype != np.float16:
        raise JS6Error(f"proposal {proposal_id} scorer-input geometry differs")

    candidate_seg_input = js2b.preprocess_seg(context.modules.functional, camera[None])
    rederived_half = candidate_seg_input.half().cpu().numpy()[0]
    if not np.array_equal(rederived_half, retained_scorer):
        raise JS6Error(f"proposal {proposal_id} stored scorer input is not exact from retained uint8")
    custody_seg = context.modules.torch.from_numpy(np.asarray(context.custody_seg[pair_id : pair_id + 1]).copy()).float()
    transported_seg = custody_seg + (candidate_seg_input - context.base_seg_input[slot : slot + 1])
    logits = js2b.score_seg(context.segnet, transported_seg, batch=1)
    robust = js3.robust_metrics(
        logits,
        np.asarray(context.custody_argmax[pair_id : pair_id + 1]),
        np.asarray(context.gt_labels[pair_id : pair_id + 1]),
        np.asarray([weight]),
    )
    prediction = robust.pop("prediction")

    pair = np.stack((np.asarray(context.base_pairs[slot, 0]), camera), axis=0)[None]
    candidate_pose_input = js2b.preprocess_pairs(context.posenet, pair, batch=1)
    custody_pose = context.modules.torch.from_numpy(np.asarray(context.custody_pose[pair_id : pair_id + 1]).copy()).float()
    transported_pose = custody_pose + (candidate_pose_input - context.base_pose_input[slot : slot + 1])
    pose_output = js2b.score_pose(context.posenet, transported_pose, batch=1)[0]
    target_pose = np.asarray(context.gt_poses[pair_id], dtype=np.float64)
    candidate_pose_error = float(np.mean((pose_output.astype(np.float64) - target_pose) ** 2))
    base_pose_error = float(context.base_pose_errors[slot])
    pose_delta = projected_pose_delta(candidate_pose_error, base_pose_error, weight)
    pose_pass = js5.accept_realized_pose(pose_delta, POSE_GATE)
    projected_robust = int(robust["projected_n600_robust_delta_flips"])
    robust_pass = projected_robust < 0
    coded_bytes = int(consumer["event.ec1p.br"]["bytes"])
    bpf = bytes_per_robust_flip(coded_bytes, projected_robust)

    result_root = _proposal_result_path(output, ordinal, proposal_id).parent
    js3.atomic_npy(result_root / "retained" / "seg_logits.float32.npy", logits.astype(np.float32, copy=False))
    js3.atomic_npy(result_root / "retained" / "seg_argmax.uint8.npy", prediction)
    js3.atomic_npy(result_root / "retained" / "pose_output6.float32.npy", pose_output.astype(np.float32, copy=False))
    js3.atomic_npy(
        result_root / "retained" / "pose_error.float64.npy",
        np.asarray([candidate_pose_error], dtype=np.float64),
    )
    result = {
        "schema": "ddm_js6_proposal_acceptance.v1",
        "ordinal": ordinal,
        "proposal_id": proposal_id,
        "pair": pair_id,
        "sample_slot": slot,
        "stratum_weight": weight,
        "event_type": proposal["event_type"],
        "family": family,
        "site_count": int(proposal["site_count"]),
        "axis": AXIS,
        "score_claim": False,
        "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        "payload_regeneration": False,
        "realized_chain": "retained EC1 receiver camera uint8 -> exact CPU R recheck -> custody SegNet/PoseNet transport",
        "pose_base_pair_error": base_pose_error,
        "pose_candidate_pair_error": candidate_pose_error,
        "realized_pose_delta_stratified_n32": pose_delta,
        "pose_gate_delta_lt": POSE_GATE,
        "pose_gate_pass": pose_pass,
        **robust,
        "robust_movement_pass": robust_pass,
        "bare_admission": pose_pass and robust_pass,
        "bare_event_brotli_q11_bytes": coded_bytes,
        "bare_event_raw_bytes": int(consumer["event.ec1p"]["bytes"]),
        "bytes_per_projected_robust_flip": bpf,
        "economics_bar_bytes_per_robust_flip_lte": ECONOMICS_BAR,
        "economics_pass": bpf is not None and bpf <= ECONOMICS_BAR,
        "inputs": {
            "event": js3.file_record(event_path),
            "event_brotli_q11": js3.file_record(brotli_path),
            "camera_uint8": js3.file_record(camera_path),
            "scorer_input_float16": js3.file_record(scorer_path),
        },
        "retained_outputs": js5._payload_map(result_root / "retained"),
    }
    js3.atomic_json(result_root / "RESULT.json", result)
    return result


def _finite_bpf(rows: Iterable[dict[str, Any]]) -> list[float]:
    return [float(row["bytes_per_projected_robust_flip"]) for row in rows if row["bytes_per_projected_robust_flip"] is not None]


def summarize(rows: list[dict[str, Any]], total_available: int = EXPECTED_PROPOSALS) -> dict[str, Any]:
    measured = len(rows)
    pose_accepted = sum(bool(row["pose_gate_pass"]) for row in rows)
    bare = [row for row in rows if bool(row["bare_admission"])]
    completed_population = measured == total_available
    acceptance_rate = pose_accepted / measured if measured else 0.0
    family_rows = []
    for family in ("boundary", "lane", "island"):
        selected = [row for row in rows if row["family"] == family]
        accepted = sum(bool(row["pose_gate_pass"]) for row in selected)
        admissions = sum(bool(row["bare_admission"]) for row in selected)
        finite = _finite_bpf(selected)
        family_rows.append(
            {
                "family": family,
                "measured": len(selected),
                "pose_accepted": accepted,
                "pose_acceptance_rate": accepted / len(selected) if selected else None,
                "bare_admissions": admissions,
                "bare_yield": admissions / len(selected) if selected else None,
                "robust_improving": sum(int(row["projected_n600_robust_delta_flips"]) < 0 for row in selected),
                "best_bytes_per_projected_robust_flip": min(finite) if finite else None,
            }
        )
    finite = _finite_bpf(rows)
    f1_eligible = completed_population and not bare
    f1_fired = f1_eligible and acceptance_rate < 0.05
    return {
        "schema": "ddm_js6_acceptance_summary.v1",
        "axis": AXIS,
        "score_claim": False,
        "measured": measured,
        "available": total_available,
        "stop_reason": "FIRST_USEFUL_NONZERO_BARE_ADMISSION" if bare else ("ALL_200_MEASURED" if completed_population else "TIME_BOUND"),
        "pose_accepted": pose_accepted,
        "pose_acceptance_rate": acceptance_rate,
        "bare_admissions": len(bare),
        "first_bare_admission": None if not bare else bare[0]["proposal_id"],
        "per_family": family_rows,
        "economics": {
            "bar_bytes_per_robust_flip_lte": ECONOMICS_BAR,
            "exact_cp135_marginal_bar": EXACT_ECONOMICS_BAR,
            "finite_rows": len(finite),
            "best_bytes_per_projected_robust_flip": min(finite) if finite else None,
            "selected_bare_bytes_per_projected_robust_flip": None if not bare else bare[0]["bytes_per_projected_robust_flip"],
            "trend_verdict": "FIRST_POINT_ONLY" if bare else ("NO_FINITE_POINT" if not finite else "NO_POSE_PASSING_POINT"),
        },
        "falsifiers": {
            "F1": {
                "eligible": f1_eligible,
                "fired": f1_fired,
                "scope": "FAMILY at the EC1 representation-level event-coordinate endpoint only",
                "criterion": "all 200 measured, no useful bare admission, and pose acceptance below 5%",
            },
            "F2": {
                "eligible": False,
                "fired": False,
                "scope": "across-proposal B/robust-flip trend",
                "criterion": "pose acceptance at least 5% and a multi-point accepted curve never trends toward 1.28 B/robust-flip",
                "reason": "the mandatory first-bare-admission stop yields at most one pose-passing finite point",
            },
        },
    }


def queue_rows(summary: dict[str, Any], output: Path) -> list[dict[str, str]]:
    if summary["stop_reason"] == "TIME_BOUND":
        return [
            {
                "action": "resume JS6 event-proposal acceptance",
                "disposition": "QUEUED-WITH-A-FIRE-ORDER",
                "owner": "ddm_js6 successor",
                "consumer_store": str(output),
                "fire_trigger": "the local CPU scorer lane is free; pass --resume against the retained state.json without invoking EC1",
            }
        ]
    if summary["falsifiers"]["F1"]["fired"]:
        return [
            {
                "action": "route residual Seg value to the SE1 shipping-survival curve",
                "disposition": "QUEUED-WITH-A-FIRE-ORDER",
                "owner": "MAIN SE1 executor/harvester",
                "consumer_store": "/Volumes/APDataStore/pact/ddm_se1_20260812",
                "fire_trigger": "MAIN consumes the JS6 F1 receipt, confirms the existing SE1 run state is resumable and CPU/GPU lane-safe, and does not regenerate or retry the closed EC1 endpoint",
            }
        ]
    return [
        {
            "action": "compose the admitted EC1 event into the HY1 whole-container campaign",
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "HY1/js1 whole-container builder",
            "consumer_store": "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hy1_solved_carriage/",
            "fire_trigger": "the JS6 admitted proposal receipt is consumed on the exact CP135 source archive; count complete container growth, prove independent decode, and retain the composed archive before any n600 scorer request",
        },
        {
            "action": "seed the JS5 projector-distilled MAIN burn from a representation-level admission",
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "MAIN training-leg router",
            "consumer_store": "/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/main_burn",
            "fire_trigger": "a receiver-known adapter converts the admitted EC1 event into a nonzero bare JS5 checkpoint and proves byte-identical event replay; MAIN then owns the training leg and the sole n600 scorer slot",
        },
    ]


def _write_table(output: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    table_rows = [
        {
            "ordinal": row["ordinal"],
            "proposal_id": row["proposal_id"],
            "family": row["family"],
            "event_type": row["event_type"],
            "pair": row["pair"],
            "site_count": row["site_count"],
            "pose_delta": row["realized_pose_delta_stratified_n32"],
            "pose_gate_pass": row["pose_gate_pass"],
            "projected_robust_delta_flips": row["projected_n600_robust_delta_flips"],
            "bare_brotli_q11_bytes": row["bare_event_brotli_q11_bytes"],
            "bytes_per_projected_robust_flip": row["bytes_per_projected_robust_flip"],
            "bare_admission": row["bare_admission"],
        }
        for row in rows
    ]
    json_path = output / "ACCEPTANCE_TABLE.json"
    js3.atomic_json(json_path, {"schema": "ddm_js6_acceptance_table.v1", "rows": table_rows, "summary": summary})
    lines = [
        "# ddm_js6 acceptance table",
        "",
        "| ord | proposal | family | pair | sites | pose delta | pose pass | robust delta flips | B | B/robust flip | bare admission |",
        "|---:|---|---|---:|---:|---:|:---:|---:|---:|---:|:---:|",
    ]
    for row in table_rows:
        bpf = "undefined" if row["bytes_per_projected_robust_flip"] is None else f"{row['bytes_per_projected_robust_flip']:.9g}"
        lines.append(
            f"| {row['ordinal']} | `{row['proposal_id']}` | {row['family']} | {row['pair']} | "
            f"{row['site_count']} | {row['pose_delta']:.9g} | {'yes' if row['pose_gate_pass'] else 'no'} | "
            f"{row['projected_robust_delta_flips']} | {row['bare_brotli_q11_bytes']} | {bpf} | "
            f"{'yes' if row['bare_admission'] else 'no'} |"
        )
    lines.extend(["", "## Per-family yield", ""])
    for family in summary["per_family"]:
        lines.append(
            f"- {family['family']}: measured {family['measured']}; pose accepted {family['pose_accepted']}; "
            f"bare admissions {family['bare_admissions']}; best B/robust-flip {family['best_bytes_per_projected_robust_flip']}"
        )
    md_path = output / "ACCEPTANCE_TABLE.md"
    js3.atomic_bytes(md_path, ("\n".join(lines) + "\n").encode())
    return {"json": js3.file_record(json_path), "markdown": js3.file_record(md_path)}


def _write_queue_annex(output: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    md = "# ddm_js6 queue annex\n\n" + "\n".join(
        f"- **Action:** {row['action']}. **Disposition:** {row['disposition']}. **Owner:** {row['owner']}. "
        f"**Consumer store:** `{row['consumer_store']}`. **Fire trigger:** {row['fire_trigger']}."
        for row in rows
    ) + "\n"
    md_path = output / "QUEUE_ANNEX.md"
    json_path = output / "QUEUE_ANNEX.json"
    js3.atomic_bytes(md_path, md.encode())
    js3.atomic_json(json_path, {"schema": "ddm_js6_queue_annex.v1", "rows": rows})
    return {"markdown": js3.file_record(md_path), "json": js3.file_record(json_path)}


def finalize(output: Path, store: Store, preflight_row: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize(rows, len(store.rows))
    queued = queue_rows(summary, output)
    table = _write_table(output, rows, summary)
    annex = _write_queue_annex(output, queued)
    result = {
        "schema": "ddm_js6_final_result.v1",
        "axis": AXIS,
        "score_claim": False,
        "pointer_moved": False,
        "payload_regeneration": False,
        "input_custody": store.custody,
        "preflight": preflight_row,
        "acceptance": summary,
        "acceptance_table": table,
        "queued_actions": queued,
        "queue_annex": annex,
        "boundaries": {
            "proposal_population": "EC1 retained representation-level event proposals only",
            "sample": "seeded stratified n32 relative gauge; projected n600 flips are not an n600 scorer run",
            "axis": AXIS,
            "candidate_archive_built": False,
            "upstream_evaluate_ran": False,
            "contest_score_measured": False,
            "mps_touched": False,
            "complete_archive_delta_bytes_measured": False,
            "bare_bytes": "proposal-level Brotli q11 event payload only; container growth remains owed",
        },
    }
    js3.atomic_json(output / "FINAL_RESULT.json", result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    lock_path = args.output / ".run.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        store = load_store(args.proposal_store)
        preflight_row = preflight(args.output, store, args.required_free_bytes)
        config = _run_config(args, store, preflight_row)
        state = load_or_start_state(args.output, config, resume=args.resume)
        completed = _recover_completed_rows(args.output, store, state)
        if completed and bool(completed[-1]["bare_admission"]):
            result = finalize(args.output, store, preflight_row, completed)
            state.update(status="COMPLETE", complete=True, final_result=js3.file_record(args.output / "FINAL_RESULT.json"))
            js3.atomic_json(args.output / "state.json", state)
            return result
        if len(completed) == len(store.rows):
            result = finalize(args.output, store, preflight_row, completed)
            state.update(status="COMPLETE", complete=True, final_result=js3.file_record(args.output / "FINAL_RESULT.json"))
            js3.atomic_json(args.output / "state.json", state)
            return result

        state.update(status="SCORING", complete=False)
        js3.atomic_json(args.output / "state.json", state)
        context = js2b.build_context(args.output)
        _assert_cpu_context(context)
        if tuple(int(value) for value in context.sample) != tuple(int(value) for value in store.state["sample"]):
            raise JS6Error("live JS5 context sample differs from the EC1 store")
        for ordinal in range(len(completed), len(store.rows)):
            if time.monotonic() - started >= args.max_wall_seconds:
                break
            row = score_proposal(context, store, store.rows[ordinal], ordinal, args.output)
            completed.append(row)
            state.update(
                next_ordinal=len(completed),
                pose_accepted=sum(bool(item["pose_gate_pass"]) for item in completed),
                bare_admissions=sum(bool(item["bare_admission"]) for item in completed),
            )
            js3.atomic_json(args.output / "state.json", state)
            if bool(row["bare_admission"]):
                break

        summary = summarize(completed, len(store.rows))
        if summary["stop_reason"] == "TIME_BOUND":
            queued = queue_rows(summary, args.output)
            _write_table(args.output, completed, summary)
            _write_queue_annex(args.output, queued)
            state.update(status="TIME_BOUNDED", complete=False)
            js3.atomic_json(args.output / "state.json", state)
            return {"schema": "ddm_js6_partial_result.v1", "acceptance": summary, "queued_actions": queued}
        result = finalize(args.output, store, preflight_row, completed)
        state.update(status="COMPLETE", complete=True, final_result=js3.file_record(args.output / "FINAL_RESULT.json"))
        js3.atomic_json(args.output / "state.json", state)
        return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--proposal-store", type=Path, default=DEFAULT_PROPOSAL_ROOT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--max-wall-seconds", type=float, default=2_400.0)
    value.add_argument("--required-free-bytes", type=int, default=DEFAULT_REQUIRED_FREE_BYTES)
    value.add_argument("--resume", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.max_wall_seconds <= 0.0 or args.max_wall_seconds > 2_400.0:
        raise SystemExit("--max-wall-seconds must be in (0, 2400]")
    if args.required_free_bytes < 0:
        raise SystemExit("--required-free-bytes must be non-negative")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
