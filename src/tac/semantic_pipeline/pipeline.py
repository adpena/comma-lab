# SPDX-License-Identifier: MIT
"""The reusable stage graph behind the semantic-joint-ctxmix CLI."""

from __future__ import annotations

import dataclasses
import json
import math
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psutil

from tac.subset_selection import DEFAULT_STRATIFIED_BLOCKS, MODE_STRATIFIED

from .contracts import (
    PipelineBlocked,
    StageReceipt,
    TargetLineage,
    atomic_copy,
    atomic_json,
    file_fact,
    host_provenance,
    probe_clip,
    require_device,
    require_storage,
    run_payload_stage,
)
from .receiver import ReceiverRequest, subprocess_inflate
from .stages.compensation import CompensationRequest, restore_neutral_connective_support
from .stages.train import (
    DALI_CACHE,
    TOKEN_FIELD,
    TrainRequest,
    render_driver_prefix,
    run_train_stage,
)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_STORE = Path("/Volumes/VertigoDataTier/pact/ddm_fpc3_chunked_n600_trainer")
DEFAULT_VIDEO = REPO / "upstream" / "videos" / "0.mkv"
BASE_ARCHIVE = Path("/Volumes/APDataStore/pact/ddm_rc1/candidate_runtime_composed/archive.zip")
BASE_SHA256 = "df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080"
BASE_BYTES = 180_456
FINAL_SHA256 = "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25"
FINAL_BYTES = 180_002
REPLAY_PROOF = Path("/Volumes/VertigoDataTier/pact/ddm_g8s_single_run_reproof/store_v2")
FINAL_ARCHIVE = REPLAY_PROOF / "retained" / "run_1" / "05_afr1.zip"
PR130_REPRO = Path("/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo")
QS4_SUPPORT_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_qs4_20260813/retained/supports")
QS5_RESULT = Path("/Volumes/VertigoDataTier/pact/ddm_qs5_20260813/FINAL_RESULT.json")
QBR1_RESUME_RECEIPT = Path(
    "/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/"
    "resume_smoke/RESUME_SMOKE_RESULT.json"
)
QBR1_MEASURED_PEAK_RSS_BYTES = 45_475_168_256
QBR1_EXECUTED_UPDATES = 4
QBR1_ELAPSED_SECONDS = 112.13081141607836
MEASURED_CHECKPOINT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_fpc2_full_pipeline_ports/"
    "full/train/checkpoints/stage_train_step_0002.pt"
)
AP_RESERVE_BYTES = 8 * 1024**3
MEMORY_SAFETY_FRACTION = 0.70
ACTIVE_CLAIMS = REPO / ".omx" / "state" / "active_lane_dispatch_claims.md"
POPULATION_UPDATES = 6000

REPLAY_PAYLOADS = (
    ("fx5", "01_fx5.zip", "4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841", 180_386),
    ("dx2", "02_dx2.zip", "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674", 180_368),
    (
        "gb1_pointer",
        "03a_gb1_pointer.zip",
        "ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4",
        180_215,
    ),
    (
        "gb1_joint",
        "03b_gb1_joint.zip",
        "ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3",
        180_192,
    ),
    ("lb1", "04_lb1.zip", "5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9", 180_083),
    ("afr1", "05_afr1.zip", FINAL_SHA256, FINAL_BYTES),
)


def population_memory_preflight(
    *,
    chunk_pairs: int,
    verdict_batch: int = 32,
    total_ram_bytes: int | None = None,
    current_used_bytes: int | None = None,
) -> dict[str, Any]:
    """Project FPC3 peak RSS and fail closed above 70 percent of host RAM."""

    if chunk_pairs < 1 or chunk_pairs > 120:
        raise ValueError("chunk_pairs must be in [1, 120]")
    if verdict_batch < 1 or verdict_batch > 120:
        raise ValueError("verdict_batch must be in [1, 120]")
    memory = psutil.virtual_memory()
    total = int(memory.total if total_ram_bytes is None else total_ram_bytes)
    current_used = int(memory.used if current_used_bytes is None else current_used_bytes)
    measured_peak = QBR1_MEASURED_PEAK_RSS_BYTES
    projected_peak = math.ceil(measured_peak * max(1.0, chunk_pairs / 16.0))
    safety_limit = math.floor(MEMORY_SAFETY_FRACTION * total)
    static_safe = projected_peak <= safety_limit
    system_aware_peak = current_used + projected_peak
    system_safe = static_safe and system_aware_peak <= safety_limit
    return {
        "schema": "ddm_fpc3_population_memory_preflight.v1",
        "status": "PASS" if system_safe else "REFUSE",
        "axis": "[host-RSS system-aware admission; no training or scorer invocation]",
        "score_claim": False,
        "config": {
            "pair_count": 600,
            "chunk_pairs": chunk_pairs,
            "verdict_batch": verdict_batch,
        },
        "measured_basis": {
            "receipt": file_fact(QBR1_RESUME_RECEIPT),
            "chunk_pairs": 16,
            "peak_rss_bytes": measured_peak,
            "peak_rss_gb_decimal": measured_peak / 1e9,
            "peak_rss_gib": measured_peak / 1024**3,
            "note": "QBR1 real-scorer B=16 resume proof; 4/4 endpoint equal",
        },
        "projection": {
            "rule": "measured B16 floor; linear above B16",
            "projected_peak_rss_bytes": projected_peak,
            "projected_peak_rss_gib": projected_peak / 1024**3,
        },
        "host": {
            "total_ram_bytes": total,
            "current_used_bytes": current_used,
            "safety_fraction": MEMORY_SAFETY_FRACTION,
            "safety_limit_bytes": safety_limit,
            "projected_used_plus_peak_bytes": system_aware_peak,
        },
        "checks": {
            "projected_process_peak_within_limit": static_safe,
            "system_aware_used_plus_peak_within_limit": system_safe,
        },
        "refusal_reason": (
            None
            if system_safe
            else "projected current use plus trainer peak exceeds 0.70 x physical RAM"
        ),
    }


def population_storage_projection(*, updates: int, pair_count: int = 600) -> dict[str, Any]:
    """Price retained FPC3 payloads against both SSD tiers."""

    checkpoint_bytes = MEASURED_CHECKPOINT.stat().st_size
    av_cache_bytes = pair_count * 2 * 874 * 1164 * 3
    eval_camera_bytes = pair_count * 2 * 3 * 384 * 512
    semantic_target_bytes = pair_count * 384 * 512
    eval_seg_bytes = pair_count * 384 * 512
    eval_pose_bytes = pair_count * 6 * 4
    receiver_raw_bytes = av_cache_bytes
    subtotal = (
        checkpoint_bytes * updates
        + av_cache_bytes
        + semantic_target_bytes
        + eval_camera_bytes
        + eval_seg_bytes
        + eval_pose_bytes
        + receiver_raw_bytes
    )
    projected = math.ceil(subtotal * 1.25)
    vertigo = shutil.disk_usage(Path("/Volumes/VertigoDataTier/pact"))
    ap = shutil.disk_usage(Path("/Volumes/APDataStore/pact"))
    return {
        "schema": "ddm_fpc3_population_storage_projection.v1",
        "status": (
            "PASS"
            if vertigo.free >= projected and ap.free >= AP_RESERVE_BYTES
            else "REFUSE"
        ),
        "retention_tier": "/Volumes/VertigoDataTier/pact",
        "checkpoint_basis": file_fact(MEASURED_CHECKPOINT),
        "updates": updates,
        "components_bytes": {
            "all_per_chunk_checkpoints": checkpoint_bytes * updates,
            "av_rgb24_cache": av_cache_bytes,
            "semantic_target_cache": semantic_target_bytes,
            "final_eval_camera": eval_camera_bytes,
            "final_seg_argmax": eval_seg_bytes,
            "final_pose6": eval_pose_bytes,
            "receiver_raw": receiver_raw_bytes,
        },
        "subtotal_bytes": subtotal,
        "projection_with_25pct_margin_bytes": projected,
        "vertigo_free_bytes": vertigo.free,
        "ap_free_bytes": ap.free,
        "ap_required_reserve_bytes": AP_RESERVE_BYTES,
        "checks": {
            "vertigo_retention_fits": vertigo.free >= projected,
            "ap_reserve_preserved": ap.free >= AP_RESERVE_BYTES,
        },
    }


def governed_launch_ticket_payload(
    *,
    store: Path,
    video: Path,
    seed: int,
    chunk_pairs: int,
    verdict_batch: int,
    memory_receipt: dict[str, Any],
    storage_projection: dict[str, Any],
) -> dict[str, Any]:
    """Build the queued MAIN ticket without firing it."""

    updates = POPULATION_UPDATES
    seconds_per_update = QBR1_ELAPSED_SECONDS / QBR1_EXECUTED_UPDATES
    projected_seconds = seconds_per_update * updates
    walltime_cap_seconds = math.ceil(projected_seconds * 1.25)
    measured_gib = QBR1_MEASURED_PEAK_RSS_BYTES / 1024**3
    run_store = store.resolve() / "full"
    argv = [
        ".venv/bin/python",
        "tools/launch_detached_process.py",
        "--output-dir",
        str(store.resolve() / "launch" / "n600_cuda"),
        "--purpose",
        "FPC3 crash-resumable chunked n600 scorer-aware CUDA population run",
        "--authority",
        "candidate contest-CUDA; exact hardware and evaluator receipts decide promotion",
        "--derive-resource-budgets",
        "--measured-peak-rss-gib",
        f"{measured_gib:.9f}",
        "--measured-thread-need",
        "4",
        "--walltime-cap-s",
        str(walltime_cap_seconds),
        "--done-receipt",
        "ddm_fpc3_n600_cuda",
        "--",
        ".venv/bin/python",
        "experiments/semantic_joint_ctxmix_pipeline.py",
        "--mode",
        "full",
        "--device",
        "cuda",
        "--video",
        str(video.resolve()),
        "--store",
        str(store.resolve()),
        "--seed",
        str(seed),
        "--pairs",
        "600",
        "--updates",
        str(updates),
        "--chunk-pairs",
        str(chunk_pairs),
        "--selection-mode",
        MODE_STRATIFIED,
        "--stratified-blocks",
        str(DEFAULT_STRATIFIED_BLOCKS),
        "--verdict-batch",
        str(verdict_batch),
        "--scorer-claim-id",
        "MAIN_MUST_INSERT_UNIQUE_SCORER_CLAIM_ID",
        "--resume-from",
        "latest",
    ]
    return {
        "schema": "ddm_fpc3_governed_n600_launch_ticket.v1",
        "status": "QUEUED_WITH_FIRE_ORDER",
        "disposition": "QUEUED-WITH-FIRE-ORDER",
        "owner": "MAIN",
        "consumer_store": str(run_store),
        "score_claim": False,
        "scorer_lane_claim_id_placeholder": "MAIN_MUST_INSERT_UNIQUE_SCORER_CLAIM_ID",
        "argv": argv,
        "resume_contract": {
            "argument": "--resume-from latest",
            "meaning": "start fresh if no checkpoint exists; otherwise restore the latest sealed chunk",
        },
        "measured_timing_basis": {
            "receipt": file_fact(QBR1_RESUME_RECEIPT),
            "elapsed_seconds": QBR1_ELAPSED_SECONDS,
            "executed_updates": QBR1_EXECUTED_UPDATES,
            "seconds_per_update": seconds_per_update,
            "axis": "[macOS-CPU exact-scorer B16 resume mechanism smoke]",
        },
        "projected_wall_clock": {
            "updates": updates,
            "seconds": projected_seconds,
            "hours": projected_seconds / 3600.0,
            "walltime_cap_seconds": walltime_cap_seconds,
            "cap_rule": "1.25 x measured seconds/update projection",
        },
        "memory_preflight": memory_receipt,
        "storage_projection": storage_projection,
        "blocked_by": [
            {
                "code": "QBR1_METAL_SLOT_OWNED",
                "disposition": "QUEUED-WITH-FIRE-ORDER",
                "owner": "MAIN",
            },
            {
                "code": "SCORER_LANE_CLAIM_REQUIRED",
                "disposition": "QUEUED-WITH-FIRE-ORDER",
                "owner": "MAIN",
            },
            {
                "code": "SYSTEM_AWARE_MEMORY_PREFLIGHT_MUST_PASS_AT_FIRE",
                "disposition": "QUEUED-WITH-FIRE-ORDER",
                "owner": "MAIN",
            },
        ],
        "fire_trigger": (
            "Metal slot released by the QBR1 burn; MAIN inserts a unique scorer-lane claim id, "
            "reruns the system-aware memory/storage receipt with PASS, then executes argv exactly"
        ),
        "no_launch_from_arm": True,
    }


def retain_population_admission_attempt(
    *,
    store: Path,
    chunk_pairs: int,
    verdict_batch: int,
    updates: int,
) -> dict[str, Any]:
    """Retain every fire-time admission attempt, including refusals."""

    root = store.resolve() / "admission"
    root.mkdir(parents=True, exist_ok=True)
    ordinal = 1
    while (root / f"attempt_{ordinal:04d}.json").exists():
        ordinal += 1
    memory = population_memory_preflight(
        chunk_pairs=chunk_pairs,
        verdict_batch=verdict_batch,
    )
    storage = population_storage_projection(updates=updates)
    status = "PASS" if memory["status"] == storage["status"] == "PASS" else "REFUSE"
    payload = {
        "schema": "ddm_fpc3_population_fire_admission.v1",
        "status": status,
        "axis": "[fire-time host memory and SSD admission; no scorer invocation]",
        "score_claim": False,
        "memory": memory,
        "storage": storage,
    }
    path = root / f"attempt_{ordinal:04d}.json"
    atomic_json(path, payload)
    return {**payload, "receipt": file_fact(path)}


def require_unique_active_scorer_claim(claim_id: str | None) -> dict[str, Any]:
    """Require this launch's newest claim to be the only live scorer row."""

    if not claim_id or claim_id == "MAIN_MUST_INSERT_UNIQUE_SCORER_CLAIM_ID":
        raise PipelineBlocked("population fire requires a concrete scorer-lane claim id")
    rows: list[dict[str, str]] = []
    for line in ACTIVE_CLAIMS.read_text(encoding="utf-8").splitlines():
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
    if own is None or not own["status"].startswith("active_") or "scorer" not in claim_id:
        raise PipelineBlocked("newest population scorer claim row is absent or non-active")
    cutoff = datetime.now(UTC) - timedelta(hours=24)
    conflicts = []
    for lane_id, row in newest_by_lane.items():
        if lane_id == claim_id or "scorer" not in lane_id or not row["status"].startswith("active_"):
            continue
        timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        if timestamp >= cutoff:
            conflicts.append(row["raw"])
    if conflicts:
        raise PipelineBlocked(f"another live scorer claim remains active: {conflicts}")
    return {"claim_id": claim_id, "registry": file_fact(ACTIVE_CLAIMS), "row": own["raw"]}


@dataclasses.dataclass(frozen=True)
class FullPipelineConfig:
    mode: str
    device: str
    video: Path = DEFAULT_VIDEO
    store: Path = DEFAULT_STORE
    seed: int = 20260903
    smoke_pairs: int = 2
    smoke_steps: int = 2
    verdict_batch_size: int = 32
    chunk_pairs: int = 16
    selection_mode: str = MODE_STRATIFIED
    stratified_blocks: int = DEFAULT_STRATIFIED_BLOCKS
    resume: bool = False
    resume_from: Path | None = None
    scorer_claim_id: str | None = None
    from_scratch: bool = False
    smoke: bool = False
    target_lineage: TargetLineage = dataclasses.field(default_factory=TargetLineage)


class SemanticPipeline:
    def __init__(self, config: FullPipelineConfig) -> None:
        if config.mode not in {"replay", "full"}:
            raise ValueError(f"unknown mode: {config.mode}")
        if config.smoke_pairs < 1 or config.smoke_steps < 1 or config.verdict_batch_size < 1:
            raise ValueError("smoke pair, step, and verdict-batch counts must all be positive")
        if config.verdict_batch_size > 120:
            raise ValueError("verdict_batch_size must not exceed the fleet chunk ceiling of 120")
        if config.chunk_pairs < 1 or config.chunk_pairs > 120:
            raise ValueError("chunk_pairs must be in [1, 120]")
        if config.stratified_blocks < 1:
            raise ValueError("stratified_blocks must be positive")
        self.config = config
        self.run_store = config.store.resolve() / config.mode
        self.receipts: list[StageReceipt] = []

    def prepare_population_launch(self) -> dict[str, Any]:
        """Write numeric preflight and the governed ticket; never launch."""

        if self.config.mode != "full" or self.config.smoke_pairs != 600:
            raise PipelineBlocked("population launch preparation requires --mode full --pairs 600")
        if self.config.smoke_steps != POPULATION_UPDATES:
            raise PipelineBlocked(f"population ticket is sealed to {POPULATION_UPDATES} updates")
        root = self.config.store.resolve()
        require_storage(root)
        memory = population_memory_preflight(
            chunk_pairs=self.config.chunk_pairs,
            verdict_batch=self.config.verdict_batch_size,
        )
        storage = population_storage_projection(updates=POPULATION_UPDATES)
        memory_path = root / "MEMORY_PREFLIGHT.json"
        storage_path = root / "STORAGE_PROJECTION.json"
        atomic_json(memory_path, memory)
        atomic_json(storage_path, storage)
        ticket = governed_launch_ticket_payload(
            store=root,
            video=self.config.video,
            seed=self.config.seed,
            chunk_pairs=self.config.chunk_pairs,
            verdict_batch=self.config.verdict_batch_size,
            memory_receipt={**memory, "receipt": file_fact(memory_path)},
            storage_projection={**storage, "receipt": file_fact(storage_path)},
        )
        ticket_path = root / "governed_n600_launch_ticket.json"
        atomic_json(ticket_path, ticket)
        return {
            "schema": "ddm_fpc3_population_launch_bundle.v1",
            "status": "QUEUED_WITH_FIRE_ORDER",
            "score_claim": False,
            "memory_preflight": file_fact(memory_path),
            "storage_projection": file_fact(storage_path),
            "launch_ticket": file_fact(ticket_path),
            "fire_trigger": ticket["fire_trigger"],
            "no_launch_from_arm": True,
        }

    @staticmethod
    def _require_pin(path: Path, expected_sha256: str, expected_bytes: int) -> dict[str, Any]:
        fact = file_fact(path)
        if fact["sha256"] != expected_sha256 or fact["bytes"] != expected_bytes:
            raise PipelineBlocked(f"pinned payload mismatch for {path}: {fact['sha256']} @ {fact['bytes']} B")
        return fact

    def _run_manifest(self, *, status: str, extra: dict[str, Any]) -> dict[str, Any]:
        config = dataclasses.asdict(self.config)
        config["video"] = str(self.config.video.resolve())
        config["store"] = str(self.config.store.resolve())
        payload = {
            "schema": "semantic_joint_ctxmix_pipeline.v1",
            "status": status,
            "mode": self.config.mode,
            "device": self.config.device,
            "seed": self.config.seed,
            "config": config,
            "score_claim": False,
            "axis": "[macOS-CPU advisory / scorer-free byte measurement]",
            "store": str(self.run_store),
            "provenance": host_provenance(REPO),
            "pipeline_sources": {
                "cli": file_fact(REPO / "experiments" / "semantic_joint_ctxmix_pipeline.py"),
                "contracts": file_fact(Path(__file__).with_name("contracts.py")),
                "pipeline": file_fact(Path(__file__)),
                "receiver": file_fact(Path(__file__).with_name("receiver.py")),
                "archive": file_fact(Path(__file__).with_name("archive.py")),
                "train_stage": file_fact(Path(__file__).with_name("stages") / "train.py"),
                "compensation_stage": file_fact(Path(__file__).with_name("stages") / "compensation.py"),
            },
            "stages": [receipt.as_dict() for receipt in self.receipts],
            **extra,
        }
        atomic_json(self.run_store / "RESULT.json", payload)
        return payload

    def run(self) -> dict[str, Any]:
        storage = require_storage(self.run_store)
        if self.config.mode == "replay":
            return self._run_replay(storage)
        return self._run_full(storage)

    def _run_replay(self, storage: dict[str, Any]) -> dict[str, Any]:
        """Validate and retain the proven replay at each exact stage boundary.

        A fresh 4,140-second execution is deliberately not hidden inside a unit
        test. The public compressor remains the executable fresh-rebuild path;
        this mode accepts the G8S proof only after rechecking every byte pin and
        copying each payload into this arm's own durable stage store.
        """

        if self.config.device != "cpu":
            raise PipelineBlocked("retained replay verification is a CPU hash/copy stage")
        base = self._require_pin(BASE_ARCHIVE, BASE_SHA256, BASE_BYTES)
        proof_result_path = REPLAY_PROOF / "RESULT.json"
        proof_result = json.loads(proof_result_path.read_text(encoding="utf-8"))
        if proof_result.get("archive", {}).get("sha256") != FINAL_SHA256:
            raise PipelineBlocked("G8S replay proof does not bind the promoted AFR1 archive")
        stage_outputs: dict[str, Path] = {"base": BASE_ARCHIVE}
        stage_inputs = {
            "fx5": "base",
            "dx2": "fx5",
            "gb1_pointer": "dx2",
            "gb1_joint": "dx2",
            "lb1": "gb1_joint",
            "afr1": "lb1",
        }
        retained = self.run_store / "retained"
        for ordinal, (stage, filename, expected_sha, expected_bytes) in enumerate(REPLAY_PAYLOADS, 1):
            source = REPLAY_PROOF / "retained" / "run_1" / filename
            self._require_pin(source, expected_sha, expected_bytes)
            destination = retained / "stages" / filename

            def copy_stage(source: Path = source, destination: Path = destination) -> None:
                atomic_copy(source, destination)

            receipt = self._resume_or_copy_stage(
                ordinal=ordinal,
                stage=stage,
                source=source,
                destination=destination,
                previous=stage_outputs[stage_inputs[stage]],
                action=copy_stage,
            )
            self.receipts.append(receipt)
            if receipt.outputs[0]["sha256"] != expected_sha:
                raise PipelineBlocked(f"replay stage {stage} changed its pinned output")
            stage_outputs[stage] = destination
        final = retained / "archive.zip"
        if final.is_file():
            self._require_pin(final, FINAL_SHA256, FINAL_BYTES)
        else:
            atomic_copy(stage_outputs["afr1"], final)
        final_fact = self._require_pin(final, FINAL_SHA256, FINAL_BYTES)
        return self._run_manifest(
            status="PASS",
            extra={
                "storage_preflight": storage,
                "base_archive": base,
                "archive": final_fact,
                "replay_proof": file_fact(proof_result_path),
                "fresh_rebuild_executed": False,
                "fresh_rebuild_entrypoint": str(REPO / "submissions" / "semantic_joint_ctxmix" / "compress.py"),
                "stage_graph_note": (
                    "GB1 has two sibling outputs from DX2; LB1 consumes gb1_joint. "
                    "There are five sequential transforms and six retained stage rows."
                ),
            },
        )

    def _resume_or_copy_stage(
        self,
        *,
        ordinal: int,
        stage: str,
        source: Path,
        destination: Path,
        previous: Path,
        action,
    ) -> StageReceipt:
        from .contracts import run_payload_stage

        return run_payload_stage(
            store=self.run_store,
            ordinal=ordinal,
            stage=stage,
            device="cpu",
            seed=self.config.seed,
            inputs=[previous, source],
            outputs=[destination],
            config={
                "mode": "verified_retained_replay",
                "public_compressor": str(REPO / "submissions" / "semantic_joint_ctxmix" / "compress.py"),
            },
            non_negotiables={
                "payload_retained": True,
                "atomic_write": True,
                "score_claim": False,
                "mechanism_reexecution": False,
                "proof_source_reverified": True,
            },
            action=action,
            resume=self.config.resume,
        )

    def _run_full(self, storage: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        if self.config.from_scratch:
            raise PipelineBlocked(
                "--from-scratch is not implemented by this warm-start semantic port; "
                "refusing instead of silently consuming AFR1"
            )
        clip = probe_clip(self.config.video)
        video = file_fact(self.config.video)
        device = require_device(self.config.device)
        population = not self.config.smoke
        if self.config.smoke and self.config.smoke_pairs > 8:
            raise PipelineBlocked("bounded --smoke execution is limited to at most 8 pairs")
        if population and (self.config.smoke_pairs != clip.pair_count or clip.pair_count != 600):
            raise PipelineBlocked("population mode requires all 600 decoded pairs")
        if population and self.config.device != "cuda":
            raise PipelineBlocked("the retained population ticket is CUDA-only; CPU/MPS population fires are refused")
        if population and self.config.smoke_steps < math.ceil(600 / self.config.chunk_pairs):
            raise PipelineBlocked("population schedule must visit all 600 pairs before completion")
        if population and self.config.smoke_steps != POPULATION_UPDATES:
            raise PipelineBlocked(f"governed population mode is sealed to {POPULATION_UPDATES} updates")
        admission = None
        lane_claim = None
        if population:
            lane_claim = require_unique_active_scorer_claim(self.config.scorer_claim_id)
            admission = retain_population_admission_attempt(
                store=self.config.store,
                chunk_pairs=self.config.chunk_pairs,
                verdict_batch=self.config.verdict_batch_size,
                updates=self.config.smoke_steps,
            )
            if admission["status"] != "PASS":
                raise PipelineBlocked(
                    "population fire refused by the retained system-aware memory/storage admission"
                )
        self.config.target_lineage.__post_init__()
        source_archive = self._require_pin(FINAL_ARCHIVE, FINAL_SHA256, FINAL_BYTES)
        retained = self.run_store / "retained"
        train_root = self.run_store / "train"
        train_outputs = [
            train_root / "TRAIN_RESULT.json",
            train_root / "retained" / "semantic_ema_state.pt",
            train_root / "retained" / "archive.zip",
            train_root / "retained" / "archive.repeat.zip",
            train_root / "retained" / "semantic_quantized_state.pt",
        ]
        if population:
            train_outputs.extend(
                [
                    train_root / "retained" / "verdict" / "camera_eval_u8.raw",
                    train_root / "retained" / "verdict" / "seg_argmax_u8.raw",
                    train_root / "retained" / "verdict" / "pose6_f32.raw",
                    train_root / "retained" / "verdict" / "VERDICT_RESULT.json",
                ]
            )

        def train_action() -> None:
            run_train_stage(
                TrainRequest(
                    video=self.config.video,
                    source_archive=FINAL_ARCHIVE,
                    output_dir=train_root,
                    device=self.config.device,
                    pair_count=self.config.smoke_pairs,
                    steps=self.config.smoke_steps,
                    seed=self.config.seed,
                    lineage=self.config.target_lineage,
                    resume=self.config.resume,
                    chunk_pairs=self.config.chunk_pairs if population else None,
                    selection_mode=self.config.selection_mode,
                    stratified_blocks=self.config.stratified_blocks,
                    verdict_batch=self.config.verdict_batch_size,
                    resume_from=self.config.resume_from,
                )
            )

        train_stage_config: dict[str, Any] = {
            "pair_count": self.config.smoke_pairs,
            "steps": self.config.smoke_steps,
            "target_lineage": self.config.target_lineage.as_dict(),
        }
        train_non_negotiables: dict[str, Any] = {
            "ema_always": True,
            "ema_law": "ema_decay_run_geometry_v1",
            "eval_roundtrip_inside_loss": True,
            "differentiable_yuv6_before_scorers": True,
            "real_video": True,
        }
        if population:
            train_stage_config.update(
                {
                    "chunk_pairs": self.config.chunk_pairs,
                    "selection_mode": self.config.selection_mode,
                    "verdict_batch": self.config.verdict_batch_size,
                }
            )
            train_non_negotiables.update(
                {
                    "all_pair_population": True,
                    "per_chunk_checkpoint_with_ema_shadow": True,
                }
            )
        self.receipts.append(
            run_payload_stage(
                store=self.run_store,
                ordinal=1,
                stage="scorer_aware_train",
                device=self.config.device,
                seed=self.config.seed,
                inputs=[self.config.video, FINAL_ARCHIVE, TOKEN_FIELD, DALI_CACHE],
                outputs=train_outputs,
                config=train_stage_config,
                non_negotiables=train_non_negotiables,
                action=train_action,
                resume=self.config.resume,
            )
        )
        candidate_archive = train_outputs[2]
        quantized_state = train_outputs[4]
        port_archive = retained / "post_qs5_port_validation.zip"
        port_result = self.run_store / "qs5_kernel_regression.json"
        support_paths = [
            QS4_SUPPORT_ROOT / proposal / "site_attribution.jsonl"
            for proposal in (
                "js6_0000_9fbf75d81c43",
                "js6_0004_06fc74e20d9e",
                "js6_0001_da319a6b65d0",
            )
        ]

        def qs5_port_action() -> None:
            request_binding = CompensationRequest(
                archive=candidate_archive,
                archive_sha256=file_fact(candidate_archive)["sha256"],
                archive_bytes=file_fact(candidate_archive)["bytes"],
                clip=clip,
                device=self.config.device,
                pair_ids=tuple(range(self.config.smoke_pairs)),
            ).validate()
            rows = [
                json.loads(line)
                for path in support_paths
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            selected, counts = restore_neutral_connective_support(rows)
            from experiments.ddm_qs5_resolve_compensation import (
                restore_neutral_connective_support as legacy_kernel,
            )

            legacy_selected, legacy_counts = legacy_kernel(rows)
            if not (legacy_counts == counts and (legacy_selected == selected).all()):
                raise PipelineBlocked("extracted QS5 kernel differs on retained pinned inputs")
            atomic_copy(candidate_archive, port_archive)
            atomic_json(
                port_result,
                {
                    "schema": "ddm_fpc2_qs5_kernel_regression.v1",
                    "status": "PASS",
                    "request": request_binding,
                    "pinned_default_final_receipt": file_fact(QS5_RESULT),
                    "site_ledgers": [file_fact(path) for path in support_paths],
                    "selected_site_count": int(selected.size),
                    "selected_site_sha256": __import__("hashlib").sha256(selected.astype("<i8").tobytes()).hexdigest(),
                    "counts": counts,
                    "legacy_value_identity": True,
                    "candidate_archive_changed": False,
                    "reason": "port regression only; no FPC2 carrier compensation was admitted",
                    "score_claim": False,
                },
            )

        self.receipts.append(
            run_payload_stage(
                store=self.run_store,
                ordinal=2,
                stage="qs5_compensation_kernel_port",
                device=self.config.device,
                seed=self.config.seed,
                inputs=[candidate_archive, QS5_RESULT, *support_paths],
                outputs=[port_archive, port_result],
                config={
                    "pair_scope": list(range(self.config.smoke_pairs)),
                    "request_contract": "CompensationRequest.v1",
                },
                non_negotiables={"legacy_pinned_input_value_identity": True, "no_uncompiled_move_admitted": True},
                action=qs5_port_action,
                resume=self.config.resume,
            )
        )
        candidate_fact = file_fact(port_archive)
        if population:
            return self._finish_population(
                started=started,
                storage=storage,
                clip=clip.as_dict(),
                video=video,
                device=device.as_dict(),
                source_archive=source_archive,
                port_archive=port_archive,
                candidate_fact=candidate_fact,
                admission=admission,
                lane_claim=lane_claim,
            )
        driver_raw = retained / "direct_driver_render_n2.raw"
        receiver_raw = retained / "receiver_render_n2.raw"

        def receiver_action(destination: Path) -> None:
            root = self.run_store / "receiver"
            report = subprocess_inflate(
                ReceiverRequest(
                    archive=port_archive,
                    archive_sha256=candidate_fact["sha256"],
                    archive_bytes=candidate_fact["bytes"],
                    destination=destination,
                    runtime_root=root / "runtime_copy",
                    checkpoint_dir=root / "checkpoint",
                    device="cpu",
                    pair_count=self.config.smoke_pairs,
                )
            )
            if report["score_claim"] is not False:
                raise PipelineBlocked("receiver smoke attempted a score claim")

        driver_report = driver_raw.with_suffix(".driver.json")

        def driver_action() -> None:
            render_driver_prefix(
                source_archive=FINAL_ARCHIVE,
                quantized_state_path=quantized_state,
                destination=driver_raw,
                pair_count=self.config.smoke_pairs,
            )

        self.receipts.append(
            run_payload_stage(
                store=self.run_store,
                ordinal=3,
                stage="direct_driver_render",
                device="cpu",
                seed=self.config.seed,
                inputs=[FINAL_ARCHIVE, quantized_state, TOKEN_FIELD],
                outputs=[driver_raw, driver_report],
                config={"pair_count": self.config.smoke_pairs},
                non_negotiables={
                    "fresh_archive_parseback": False,
                    "compiled_quantized_state": True,
                },
                action=driver_action,
                resume=self.config.resume,
            )
        )

        self.receipts.append(
            run_payload_stage(
                store=self.run_store,
                ordinal=4,
                stage="receiver_receiver_render",
                device="cpu",
                seed=self.config.seed,
                inputs=[port_archive, TOKEN_FIELD],
                outputs=[receiver_raw, receiver_raw.with_suffix(".receiver.json")],
                config={"pair_count": self.config.smoke_pairs, "fresh_archive": candidate_fact},
                non_negotiables={
                    "copied_shipped_runtime": True,
                    "prefix_python_checkpoint": True,
                    "fresh_archive_parseback": True,
                },
                action=lambda: receiver_action(receiver_raw),
                resume=self.config.resume,
            )
        )
        if file_fact(driver_raw)["sha256"] != file_fact(receiver_raw)["sha256"]:
            raise PipelineBlocked("pipeline receiver differs from the driver's retained render")
        score_result = self.run_store / "advisory" / "EVALUATE_RESULT.json"
        score_report = self.run_store / "advisory" / "report.txt"
        score_stdout = self.run_store / "advisory" / "evaluate.stdout.txt"

        def score_action() -> None:
            self._run_advisory_score(port_archive, receiver_raw, score_result, score_report, score_stdout)

        self.receipts.append(
            run_payload_stage(
                store=self.run_store,
                ordinal=5,
                stage="upstream_evaluate_n2_advisory",
                device="cpu",
                seed=self.config.seed,
                inputs=[port_archive, receiver_raw, self.config.video, REPO / "upstream" / "evaluate.py"],
                outputs=[score_result, score_report, score_stdout],
                config={"pair_count": self.config.smoke_pairs, "verdict_batch_size": self.config.verdict_batch_size},
                non_negotiables={"axis": "[macOS-CPU advisory]", "score_claim": False, "upstream_evaluate": True},
                action=score_action,
                resume=self.config.resume,
            )
        )
        score = json.loads(score_result.read_text(encoding="utf-8"))
        return self._run_manifest(
            status="PASS",
            extra={
                "elapsed_seconds": time.monotonic() - started,
                "storage_preflight": storage,
                "clip": clip.as_dict(),
                "video": video,
                "device_binding": device.as_dict(),
                "target_lineage": self.config.target_lineage.as_dict(),
                "upstream_snapshot": {
                    "git_sha": host_provenance(REPO)["git_sha"],
                    "frame_utils": file_fact(REPO / "upstream" / "frame_utils.py"),
                    "evaluate": file_fact(REPO / "upstream" / "evaluate.py"),
                    "segnet_weights": file_fact(REPO / "upstream" / "models" / "segnet.safetensors"),
                    "posenet_weights": file_fact(REPO / "upstream" / "models" / "posenet.safetensors"),
                },
                "cleared_blockers": self._cleared_ports(),
                "launch_ticket": {
                    "status": "NOT_PREPARED_BY_BOUNDED_SMOKE",
                    "preparation_entrypoint": "--prepare-launch-ticket --mode full --pairs 600",
                },
                "source_archive": source_archive,
                "archive": candidate_fact,
                "fresh_archive": candidate_fact["sha256"] != FINAL_SHA256,
                "receiver_identity": {
                    "driver": file_fact(driver_raw),
                    "receiver": file_fact(receiver_raw),
                    "byte_identical": True,
                    "pair_count": self.config.smoke_pairs,
                },
                "advisory_score": score,
            },
        )

    def _finish_population(
        self,
        *,
        started: float,
        storage: dict[str, Any],
        clip: dict[str, Any],
        video: dict[str, Any],
        device: dict[str, Any],
        source_archive: dict[str, Any],
        port_archive: Path,
        candidate_fact: dict[str, Any],
        admission: dict[str, Any],
        lane_claim: dict[str, Any],
    ) -> dict[str, Any]:
        """Finish the retained n600 receiver and exact-evaluator candidate path."""

        retained = self.run_store / "retained"
        receiver_raw = retained / "receiver_render_n600.raw"
        receiver_report = receiver_raw.with_suffix(".receiver.json")

        def receiver_action() -> None:
            report = subprocess_inflate(
                ReceiverRequest(
                    archive=port_archive,
                    archive_sha256=candidate_fact["sha256"],
                    archive_bytes=candidate_fact["bytes"],
                    destination=receiver_raw,
                    runtime_root=self.run_store / "receiver" / "runtime_copy",
                    checkpoint_dir=self.run_store / "receiver" / "checkpoint",
                    device=self.config.device,
                    pair_count=600,
                )
            )
            if report["score_claim"] is not False:
                raise PipelineBlocked("receiver attempted an authority claim before evaluator custody")

        self.receipts.append(
            run_payload_stage(
                store=self.run_store,
                ordinal=3,
                stage="receiver_n600_render",
                device=self.config.device,
                seed=self.config.seed,
                inputs=[port_archive, TOKEN_FIELD],
                outputs=[receiver_raw, receiver_report],
                config={"pair_count": 600, "fresh_archive": candidate_fact},
                non_negotiables={
                    "copied_shipped_runtime": True,
                    "fresh_archive_parseback": True,
                    "all_payload_retained": True,
                },
                action=receiver_action,
                resume=self.config.resume,
            )
        )
        score_root = self.run_store / "evaluation"
        score_result = score_root / "EVALUATE_RESULT.json"
        score_report = score_root / "report.txt"
        score_stdout = score_root / "evaluate.stdout.txt"
        self.receipts.append(
            run_payload_stage(
                store=self.run_store,
                ordinal=4,
                stage="upstream_evaluate_n600_chunked",
                device=self.config.device,
                seed=self.config.seed,
                inputs=[port_archive, receiver_raw, self.config.video, REPO / "upstream" / "evaluate.py"],
                outputs=[score_result, score_report, score_stdout],
                config={"pair_count": 600, "verdict_batch": self.config.verdict_batch_size},
                non_negotiables={
                    "full_population": True,
                    "verdict_forward_chunked": True,
                    "score_claim": False,
                    "hardware_custody_required_for_promotion": True,
                },
                action=lambda: self._run_advisory_score(
                    port_archive,
                    receiver_raw,
                    score_result,
                    score_report,
                    score_stdout,
                ),
                resume=self.config.resume,
            )
        )
        score = json.loads(score_result.read_text(encoding="utf-8"))
        ticket_path = self.config.store.resolve() / "governed_n600_launch_ticket.json"
        if not ticket_path.is_file():
            raise PipelineBlocked("population execution requires its governed launch ticket")
        return self._run_manifest(
            status="PASS",
            extra={
                "elapsed_seconds": time.monotonic() - started,
                "storage_preflight": storage,
                "fire_admission": admission,
                "scorer_lane_claim": lane_claim,
                "clip": clip,
                "video": video,
                "device_binding": device,
                "target_lineage": self.config.target_lineage.as_dict(),
                "launch_ticket": file_fact(ticket_path),
                "source_archive": source_archive,
                "archive": candidate_fact,
                "fresh_archive": candidate_fact["sha256"] != FINAL_SHA256,
                "receiver_identity": {
                    "receiver": file_fact(receiver_raw),
                    "pair_count": 600,
                    "fresh_archive_parseback": True,
                },
                "population_score_candidate": score,
                "promotion_boundary": (
                    "exact archive and n600 evaluator ran, but contest-CUDA hardware/log custody "
                    "must be adjudicated before any pointer claim"
                ),
            },
        )

    @staticmethod
    def _cleared_ports() -> list[dict[str, Any]]:
        return [
            {"code": "QS5_INSTANCE_PINNED", "status": "CLEARED", "surface": "semantic_pipeline/stages/compensation.py"},
            {"code": "SOLVE_DEVICE_FLAGS_ABSENT", "status": "CLEARED", "surface": "FCD1/JG5/QS5/UP2 CLIs"},
            {"code": "SHIPPED_RECEIVER_FRESH_ARCHIVE_REFUSAL", "status": "CLEARED", "surface": "semantic_pipeline/receiver.py"},
            {"code": "PREFIX_RUNTIME_UNREACHABLE", "status": "CLEARED", "surface": "run-local F26 copy only"},
            {"code": "TRAINER_DEVICE_CONTRACT_MISMATCH", "status": "CLEARED", "surface": "semantic_pipeline/stages/train.py"},
            {"code": "TARGET_CACHE_LINEAGE_CONFOUND", "status": "CLEARED", "surface": "TargetLineage"},
        ]

    def _run_advisory_score(
        self,
        archive: Path,
        raw: Path,
        result_path: Path,
        report_path: Path,
        stdout_path: Path,
    ) -> None:
        root = result_path.parent / "submission"
        inflated = root / "inflated"
        inflated.mkdir(parents=True, exist_ok=True)
        atomic_copy(archive, root / "archive.zip")
        atomic_copy(raw, inflated / "0.raw")
        names = result_path.parent / "video_names.txt"
        names.write_text("0.mkv\n", encoding="utf-8")
        command = [
            sys.executable,
            str(REPO / "upstream" / "evaluate.py"),
            "--batch-size",
            str(min(self.config.smoke_pairs, self.config.verdict_batch_size)),
            "--device",
            self.config.device,
            "--submission-dir",
            str(root),
            "--uncompressed-dir",
            str(REPO / "upstream" / "videos"),
            "--video-names-file",
            str(names),
            "--report",
            str(report_path),
        ]
        completed = subprocess.run(command, cwd=REPO / "upstream", check=False, capture_output=True, text=True)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise PipelineBlocked(f"upstream evaluator failed: {completed.stderr[-2000:]}")
        report = report_path.read_text(encoding="utf-8")
        def extract(label: str) -> float:
            match = re.search(rf"{re.escape(label)}:\s+([0-9.]+)", report)
            if match is None:
                raise PipelineBlocked(f"evaluator report omitted {label}")
            return float(match.group(1))
        d_pose = extract("Average PoseNet Distortion")
        d_seg = extract("Average SegNet Distortion")
        rate = extract("Compression Rate")
        exact_from_report_components = 100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * rate
        atomic_json(
            result_path,
            {
                "schema": (
                    "ddm_fpc2_upstream_evaluate_smoke.v1"
                    if self.config.smoke
                    else "ddm_fpc3_upstream_evaluate_population_candidate.v1"
                ),
                "axis": (
                    "[macOS-CPU advisory]"
                    if self.config.smoke
                    else "[CUDA exact-evaluator candidate; hardware custody pending]"
                ),
                "score_claim": False,
                "sample_scope": {
                    "n": self.config.smoke_pairs,
                    "selection": (
                        "contiguous prefix plumbing smoke; no population verdict"
                        if self.config.smoke
                        else "full population"
                    ),
                },
                "d_pose": d_pose,
                "d_seg": d_seg,
                "rate": rate,
                "score_recomputed_from_report_components": exact_from_report_components,
                "report_precision_note": (
                    "components are upstream report's 8-decimal values; promotion requires "
                    "the retained report plus exact hardware/runtime custody"
                ),
                "archive": file_fact(archive),
                "raw": file_fact(raw),
                "argv": command,
                "report": file_fact(report_path),
                "stdout": file_fact(stdout_path),
            },
        )

def clean_success_scratch(root: Path) -> None:
    """Remove only driver-owned success scratch; retained payloads are never touched."""

    scratch = root / "scratch"
    if scratch.is_dir() and not any(scratch.iterdir()):
        shutil.rmtree(scratch)
