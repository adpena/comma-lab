# SPDX-License-Identifier: MIT
"""The reusable stage graph behind the semantic-joint-ctxmix CLI."""

from __future__ import annotations

import dataclasses
import json
import shutil
import time
from pathlib import Path
from typing import Any

from .contracts import (
    PipelineBlocked,
    StageReceipt,
    atomic_copy,
    atomic_json,
    file_fact,
    host_provenance,
    probe_clip,
    require_device,
    require_storage,
)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_STORE = Path("/Volumes/VertigoDataTier/pact/ddm_fpc1_full_pipeline_compress")
DEFAULT_VIDEO = REPO / "upstream" / "videos" / "0.mkv"
BASE_ARCHIVE = Path("/Volumes/APDataStore/pact/ddm_rc1/candidate_runtime_composed/archive.zip")
BASE_SHA256 = "df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080"
BASE_BYTES = 180_456
FINAL_SHA256 = "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25"
FINAL_BYTES = 180_002
REPLAY_PROOF = Path("/Volumes/VertigoDataTier/pact/ddm_g8s_single_run_reproof/store_v2")
PR130_REPRO = Path("/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo")

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
    resume: bool = False
    from_scratch: bool = False


class SemanticPipeline:
    def __init__(self, config: FullPipelineConfig) -> None:
        if config.mode not in {"replay", "full"}:
            raise ValueError(f"unknown mode: {config.mode}")
        if config.smoke_pairs < 1 or config.smoke_steps < 1 or config.verdict_batch_size < 1:
            raise ValueError("smoke pair, step, and verdict-batch counts must all be positive")
        self.config = config
        self.run_store = config.store.resolve() / config.mode
        self.receipts: list[StageReceipt] = []

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
        clip = probe_clip(self.config.video)
        video = file_fact(self.config.video)
        device = require_device(self.config.device)
        blockers = self._full_port_blockers()
        ticket = self._write_launch_ticket(clip.as_dict(), video, blockers)
        self._run_manifest(
            status="BLOCKED_PORT_REQUIRED",
            extra={
                "elapsed_seconds": time.monotonic() - started,
                "storage_preflight": storage,
                "clip": clip.as_dict(),
                "video": video,
                "device_binding": device.as_dict(),
                "upstream_snapshot": {
                    "git_sha": host_provenance(REPO)["git_sha"],
                    "frame_utils": file_fact(REPO / "upstream" / "frame_utils.py"),
                    "evaluate": file_fact(REPO / "upstream" / "evaluate.py"),
                    "segnet_weights": file_fact(REPO / "upstream" / "models" / "segnet.safetensors"),
                    "posenet_weights": file_fact(REPO / "upstream" / "models" / "posenet.safetensors"),
                },
                "blockers": blockers,
                "launch_ticket": file_fact(ticket),
                "archive": None,
            },
        )
        raise PipelineBlocked(
            "full mode refused before training: named solve/receiver stages are not "
            f"per-clip safe; blocker receipt: {self.run_store / 'RESULT.json'}"
        ) from None

    def _full_port_blockers(self) -> list[dict[str, Any]]:
        return [
            {
                "code": "QS5_INSTANCE_PINNED",
                "surface": str(REPO / "experiments" / "ddm_qs5_resolve_compensation.py"),
                "reason": (
                    "the only executable run path embeds QS4/CP135 paths and N=600; it has no "
                    "archive, clip-config, device, or pair-scope arguments"
                ),
            },
            {
                "code": "SOLVE_DEVICE_FLAGS_ABSENT",
                "surface": "FCD1/JG5/QS5 solve CLIs",
                "reason": (
                    "the named CLIs do not expose --device, so claiming CPU/MPS/CUDA routing "
                    "would invent flags forbidden by the charter"
                ),
            },
            {
                "code": "SHIPPED_RECEIVER_FRESH_ARCHIVE_REFUSAL",
                "surface": str(REPO / "submissions" / "semantic_joint_ctxmix" / "inflate.py"),
                "reason": (
                    "the shipped entrypoint pins the AFR1 sha/size and refuses CPU; a fresh "
                    "full-mode archive and the required CPU n=2 smoke cannot pass that entrypoint"
                ),
            },
            {
                "code": "PREFIX_RUNTIME_UNREACHABLE",
                "surface": str(REPO / "submissions" / "semantic_joint_ctxmix" / "runtime" / "f26_inflate.py"),
                "reason": (
                    "F26_ADVISORY_PAIR_LIMIT requires native-hpac, while the same runtime "
                    "unconditionally refuses every token decoder except python"
                ),
            },
            {
                "code": "TRAINER_DEVICE_CONTRACT_MISMATCH",
                "surface": str(REPO / "experiments" / "ddm_mx1_pr130_semantic_renderer.py"),
                "reason": (
                    "torch-smoke forces CPU and lacks an EMA shadow; mlx-train accepts MLX "
                    "cpu/gpu semantics rather than the promised torch mps/cpu/cuda solve contract, "
                    "and only constructs EMA when a controller policy is supplied"
                ),
            },
            {
                "code": "TARGET_CACHE_LINEAGE_CONFOUND",
                "surface": str(PR130_REPRO / "scripts" / "e2e.py"),
                "reason": (
                    "the strict raw-video graph feeds one fresh DALI cache to its stages, while "
                    "the retained selected lineage used AV-like semantic targets and DALI carrier, "
                    "HPAC, and token targets; the two full-population target fields differ"
                ),
            },
        ]

    def _write_launch_ticket(
        self,
        clip: dict[str, Any],
        video: dict[str, Any],
        blockers: list[dict[str, Any]],
    ) -> Path:
        ticket = self.run_store / "governed_n600_launch_ticket.json"
        argv = [
            ".venv/bin/python",
            "tools/launch_detached_process.py",
            "--output-dir",
            str(self.run_store / "detached_n600"),
            "--done-receipt",
            "ddm_fpc1_full_n600",
            "--",
            ".venv/bin/python",
            "experiments/semantic_joint_ctxmix_pipeline.py",
            "--mode",
            "full",
            "--device",
            "mps",
            "--video",
            str(self.config.video.resolve()),
            "--store",
            str(self.config.store.resolve()),
            "--seed",
            str(self.config.seed),
            "--smoke-pairs",
            str(clip["pair_count"]),
            "--smoke-steps",
            "6000",
            "--verdict-batch-size",
            str(self.config.verdict_batch_size),
            "--resume-from",
            str(self.run_store),
        ]
        payload = {
            "schema": "ddm_fpc1_governed_n600_launch_ticket.v1",
            "status": "QUEUED_WITH_FIRE_ORDER",
            "disposition": "QUEUED-WITH-FIRE-ORDER",
            "owner": "MAIN",
            "consumer_store": str(self.config.store.resolve()),
            "score_claim": False,
            "device_status": "UNTESTED-HERE; intended local Metal training-gradient lane",
            "clip": clip,
            "video": video,
            "argv": argv,
            "projected_wall_clock_seconds": {
                "status": "RECALLED_BRACKET_NOT_MEASURED_FPC1",
                "lower_bound": 14_400,
                "upper_bound": 259_200,
                "basis": "PR130 49-stage recipe plus measured 4,140.9 s lossless tail",
            },
            "projected_peak_rss_bytes": {
                "value": None,
                "status": "REQUIRES_FRESH_MEMORY_PREFLIGHT",
                "basis": (
                    "no config-matched memory receipt exists for the unported full graph; "
                    "the fire trigger requires tools/witness_memory_preflight.py first"
                ),
            },
            "fire_trigger": (
                "all listed ports land; the n=2 CPU receiver-identity test passes; a fresh "
                "memory-preflight receipt passes; MAIN owns the scorer/launch lane"
            ),
            "blocked_by": blockers,
        }
        atomic_json(ticket, payload)
        return ticket


def clean_success_scratch(root: Path) -> None:
    """Remove only driver-owned success scratch; retained payloads are never touched."""

    scratch = root / "scratch"
    if scratch.is_dir() and not any(scratch.iterdir()):
        shutil.rmtree(scratch)
