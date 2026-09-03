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
from pathlib import Path
from typing import Any

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
DEFAULT_STORE = Path("/Volumes/VertigoDataTier/pact/ddm_fpc2_full_pipeline_ports")
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
    smoke: bool = False
    target_lineage: TargetLineage = dataclasses.field(default_factory=TargetLineage)


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
        if not self.config.smoke or self.config.smoke_pairs > 8:
            raise PipelineBlocked(
                "this arm may execute only an explicit --smoke with at most 8 pairs; "
                "the retained n600 ticket is the sole full-population consumer"
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
                )
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
                config={
                    "pair_count": self.config.smoke_pairs,
                    "steps": self.config.smoke_steps,
                    "target_lineage": self.config.target_lineage.as_dict(),
                },
                non_negotiables={
                    "ema_always": True,
                    "ema_law": "ema_decay_run_geometry_v1",
                    "eval_roundtrip_inside_loss": True,
                    "differentiable_yuv6_before_scorers": True,
                    "real_video": True,
                },
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
        ticket = self._write_launch_ticket(
            clip.as_dict(),
            video,
            [
                {
                    "code": "N600_CHUNKED_TRAIN_CONSUMER_ABSENT",
                    "scope": "INSTANCE",
                    "owner": "MAIN",
                },
                {
                    "code": "N600_SCORER_LANE_UNCLAIMED",
                    "scope": "INSTANCE",
                    "owner": "MAIN",
                },
            ],
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
                "launch_ticket": file_fact(ticket),
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
            str(self.config.smoke_pairs),
            "--device",
            "cpu",
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
            raise PipelineBlocked(f"upstream advisory evaluator failed: {completed.stderr[-2000:]}")
        report = report_path.read_text(encoding="utf-8")
        def extract(label: str) -> float:
            match = re.search(rf"{re.escape(label)}:\s+([0-9.]+)", report)
            if match is None:
                raise PipelineBlocked(f"advisory report omitted {label}")
            return float(match.group(1))
        d_pose = extract("Average PoseNet Distortion")
        d_seg = extract("Average SegNet Distortion")
        rate = extract("Compression Rate")
        exact_from_report_components = 100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * rate
        atomic_json(
            result_path,
            {
                "schema": "ddm_fpc2_upstream_evaluate_smoke.v1",
                "axis": "[macOS-CPU advisory]",
                "score_claim": False,
                "sample_scope": {"n": self.config.smoke_pairs, "selection": "contiguous prefix plumbing smoke; no population verdict"},
                "d_pose": d_pose,
                "d_seg": d_seg,
                "rate": rate,
                "score_recomputed_from_report_components": exact_from_report_components,
                "report_precision_note": "components are upstream report's 8-decimal values; this is a smoke, never a score row",
                "archive": file_fact(archive),
                "raw": file_fact(raw),
                "argv": command,
                "report": file_fact(report_path),
                "stdout": file_fact(stdout_path),
            },
        )

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
            "ddm_fpc2_full_n600",
            "--",
            ".venv/bin/python",
            "experiments/semantic_joint_ctxmix_pipeline.py",
            "--mode",
            "full",
            "--device",
            "cuda",
            "--video",
            str(self.config.video.resolve()),
            "--store",
            str(self.config.store.resolve()),
            "--seed",
            str(self.config.seed),
            "--pairs",
            str(clip["pair_count"]),
            "--smoke-steps",
            "6000",
            "--verdict-batch-size",
            str(self.config.verdict_batch_size),
            "--resume-from",
            str(self.run_store),
        ]
        payload = {
            "schema": "ddm_fpc2_governed_n600_launch_ticket.v1",
            "status": "QUEUED_WITH_FIRE_ORDER",
            "disposition": "QUEUED-WITH-FIRE-ORDER",
            "owner": "MAIN",
            "consumer_store": str(self.config.store.resolve()),
            "score_claim": False,
            "device_status": "UNTESTED-HERE; intended contest-CUDA full receiver/training lane",
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
                "value": 32_502_165_012,
                "projected_peak_gib": 30.27,
                "status": "PROJECTED_FROM_MEASURED_GUARD_LAW_NOT_LIVE_ADMISSION",
                "basis": (
                    "tools/witness_memory_preflight.project_peak_rss_gib: n600, 384x512, "
                    "in_feat=96, self_orient=false, micro_batch_pairs=2, verdict_batch=32; "
                    "measured #205 constants; live system-aware admission still required"
                ),
            },
            "fire_trigger": (
                "MAIN lands a chunked n600 target materializer/trainer consumer for the same "
                "stage contract, reruns tools/witness_memory_preflight.py --system-aware, "
                "confirms no active scorer lane, and owns the launch lane"
            ),
            "blocked_by": blockers,
            "local_execution_boundary": (
                "the committed arm intentionally executes only --smoke n<=8; this ticket is "
                "not fireable until its named n600 chunking trigger is satisfied"
            ),
        }
        atomic_json(ticket, payload)
        return ticket


def clean_success_scratch(root: Path) -> None:
    """Remove only driver-owned success scratch; retained payloads are never touched."""

    scratch = root / "scratch"
    if scratch.is_dir() and not any(scratch.iterdir()):
        shutil.rmtree(scratch)
