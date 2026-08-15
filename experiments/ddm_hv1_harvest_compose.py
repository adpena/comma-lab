#!/usr/bin/env python3
"""HV1 harvest composition — byte-close a SELECTED MID-RUN HPAC checkpoint.

The e960 QAT burn was governed-early-stopped and its run root is read-only. The
selector (``tools/select_hpac_checkpoint.py``) minimizes
``(25/37545489)*estimated_joint_bytes + 100*top1_error`` over the retained
periodic checkpoints; over all 81 of them the argmin is epoch 634. This runner
composes ONE candidate archive from a selected checkpoint by DELEGATING to the
landed RX2 identity-race conventions
(``experiments/ddm_rx2_mc36_identity_race.py``) — it reinvents nothing.

Two deliberate, documented deltas versus the RX2 terminal run:

1. EPOCH-SCOPED STORE REDIRECTION. RX2's roots are module globals with no
   argument surface. Running RX2 unmodified would (a) overwrite the custodied
   e480b v2 terminal artifacts and (b) silently RESUME its 600 completed
   base-probability frames, pairing one epoch's weights with another epoch's
   logits — a silent wrong number. This module rebinds both roots to
   PER-EPOCH HV1 stores and asserts the rebinding took effect before any stage
   runs, which is also what makes an epoch-vs-epoch A/B safe. Bulk goes to
   APDataStore because the Vertigo tier is at capacity; the small payload set
   plus every receipt is mirrored to the charter-mandated Vertigo retention
   root by the ``manifest`` stage.

2. PREFLIGHT ADAPTATION. RX2's preflight demands a TERMINAL trainer report whose
   ``history[-1].epoch`` equals the packed epoch, plus the trainer artifact
   manifest. A mid-run periodic checkpoint from a LIVE run has neither: the
   report is written at run end. :func:`hv1_preflight` keeps every custody check
   RX2 makes on the objects that exist (base archive, expected spatial tokens,
   source event order, checkpoint schema/phase/profile/epoch, causal-state hash
   re-verification) and REPLACES the terminal-report check with three checks the
   live artifacts DO support: the pinned selection sha, the checkpoint's own
   embedded telemetry history, and the resume-lineage pin back to the e480b
   parent. It is a scope substitution on the report surface only; no custody
   invariant is dropped.

Axis: ``[macOS-CPU advisory, scorer-free lossless composition]``. No scorer runs
here. Distortion identity is proven by full raw-output byte identity against the
MC36 CPU decode, which makes d_seg/d_pose EXACTLY the base's; only the rate term
moves. No score is claimed and no dispatch is performed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

rx2 = importlib.import_module("experiments.ddm_rx2_mc36_identity_race")
rx1 = rx2.rx1
RX2_WORK_ROOT = rx2.WORK_ROOT
RX2_BULK_ROOT = rx2.BULK_ROOT

# ---------------------------------------------------------------- HV1 pins ---
HV1_WORK_BASE = Path("/Volumes/VertigoDataTier/pact/ddm_hv1_harvest_compose")
HV1_BULK_BASE = Path("/Volumes/APDataStore/pact/ddm_hv1_harvest_compose")
CHECKPOINT_ROOT = HV1_WORK_BASE / "retained"
# Every store is EPOCH-SCOPED. RX2's export stage resumes from existing frame
# receipts, so two checkpoints sharing one store would silently pair epoch A's
# weights with epoch B's logits. Scoping is what makes an A/B safe.
CANDIDATES: dict[int, dict[str, Any]] = {
    508: {
        "sha256": "68da5ee0135613ec9aebb1c323d26b475d1292e2006bfaad33bf1bd87659fa7a",
        "estimated_joint_bytes": 130_875,
        "top1_error": 0.0018965996636284722,
        "selection": "rfo2 partial-log argmin superseded by the governed early-stop selection",
    },
    634: {
        "sha256": "5007beae7af7789758092f12f49096e13692e2e59850c85eb4642cd6fad147ec",
        "estimated_joint_bytes": 130_393,
        "top1_error": 0.0018945397271050348,
        "selection": "governed early-stop argmin over all 81 retained periodic checkpoints",
    },
}
DEFAULT_EPOCH = 634
PARENT_CHECKPOINT_SHA256 = "cd89907b5330bd78f9c1477107504231792c235fa7637b8981698a10948a5a61"
PARENT_EPOCH = 480

# The e480b v2 incumbent this candidate must beat, with its exact CUDA components.
E480B_V2 = {
    "candidate_id": "e480b_v2_s1p25_c1p0_brotli_q10",
    "archive_bytes": 183_502,
    "archive_sha256": "e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3",
    "member_bytes": 183_402,
    "member_sha256": "30c0165ec56dd9327ca4dcda477c34c25f7664622ac37ec8ed171114267d1b58",
    "model_bytes": 70_557,
    "token_bytes": 112_749,
    "score_seg_contribution": 0.029611,
    "score_pose_contribution": 0.008294576541331089,
    "score_rate_contribution": 0.12218644961582469,
    "score_recomputed_from_components": 0.1600920261571558,
    "authority": "[contest-CUDA] Tesla T4 n600",
}
RATE_COEFFICIENT = 25.0 / rx1.RATE_DENOMINATOR
AXIS = rx2.AXIS
SCORE_CLAIM = False
STAGES = (
    "preflight",
    "prepare",
    "export-base",
    "fit",
    "materialize",
    "encode",
    "build",
    "cpu-decode",
    "finalize",
    "manifest",
    "fire-order",
)
# Stages after which the chain can stop UNAMBIGUOUSLY. The per-variant coding
# stages are excluded because the chain finishes every selected variant before
# it stops: "build" therefore means "after the last variant's build".
STOP_AFTER_CHOICES = ("prepare", "export-base", "fit", "build", "cpu-decode", "finalize")


class HV1Error(RuntimeError):
    """Fail-closed error for a broken HV1 custody or composition gate."""


def _bind_hv1_conventions(epoch: int) -> None:
    """Point the RX2 conventions at this epoch's HV1 stores and preflight, fail-closed.

    RX2 resolves both roots and its preflight through module globals at call
    time, so rebinding them redirects every stage. The assertions exist because
    a silent failure here would either destroy custodied e480b artifacts or
    resume another checkpoint's probability frames. ``rx2.prepare`` calls the
    module-global ``preflight``; binding it to :func:`hv1_preflight` is what
    lets a mid-run periodic checkpoint through the same prepare body.
    """
    if epoch not in CANDIDATES:
        raise HV1Error(f"epoch {epoch} is not a pinned HV1 candidate")
    work = HV1_WORK_BASE / f"ep{epoch:04d}"
    bulk = HV1_BULK_BASE / f"ep{epoch:04d}"
    if work == RX2_WORK_ROOT or bulk == RX2_BULK_ROOT:
        raise HV1Error("HV1 stores collide with the custodied RX2 terminal stores")
    if RX2_WORK_ROOT in work.parents or RX2_BULK_ROOT in bulk.parents:
        raise HV1Error("HV1 stores nest inside the custodied RX2 terminal stores")
    rx2.WORK_ROOT = work
    rx2.BULK_ROOT = bulk
    if rx2.WORK_ROOT != work or rx2.BULK_ROOT != bulk:
        raise HV1Error("HV1 root rebinding did not take effect")
    rx2.preflight = hv1_preflight
    if rx2.preflight is not hv1_preflight:
        raise HV1Error("HV1 preflight rebinding did not take effect")
    work.mkdir(parents=True, exist_ok=True)
    bulk.mkdir(parents=True, exist_ok=True)


def _checkpoint_path(epoch: int) -> Path:
    return CHECKPOINT_ROOT / f"epoch_{epoch:04d}.pt"


def _selected_history_row(checkpoint: dict[str, Any], epoch: int) -> dict[str, Any]:
    pins = CANDIDATES[epoch]
    history = checkpoint.get("history")
    if not isinstance(history, list) or not history:
        raise HV1Error("selected checkpoint carries no embedded telemetry history")
    row = history[-1]
    if not isinstance(row, dict) or row.get("epoch") != epoch:
        raise HV1Error("selected checkpoint history does not end at the selected epoch")
    if row.get("phase") != "discrete_qat" or row.get("evaluated_weights") != "ema_shadow":
        raise HV1Error("selected checkpoint history row is not the discrete-QAT EMA row")
    if int(row.get("estimated_joint_bytes", -1)) != pins["estimated_joint_bytes"]:
        raise HV1Error("selected checkpoint joint-byte telemetry does not match the selection")
    if float(row.get("top1_error", -1.0)) != pins["top1_error"]:
        raise HV1Error("selected checkpoint top-1 telemetry does not match the selection")
    return row


def hv1_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """RX2 preflight custody, adapted to a mid-run periodic checkpoint.

    Kept from RX2: base archive / expected spatial / source manifest pins, the
    source event-order digest, checkpoint schema-phase-profile-epoch, and the
    causal-state hash RE-VERIFICATION. Replaced: the terminal trainer report and
    artifact manifest (absent by construction for a live run) give way to the
    pinned selection sha, the checkpoint's own history row, and the resume
    lineage pin back to the e480b parent.
    """
    rx2._require(
        rx2.BASE_ARCHIVE, size=rx1.EXPECTED_ARCHIVE_BYTES, digest=rx1.EXPECTED_ARCHIVE_SHA256
    )
    rx2._require(
        rx2.EXPECTED_SPATIAL, size=rx2.TOKEN_COUNT, digest=rx1.EXPECTED_SPATIAL_SHA256
    )
    epoch = args.epoch
    rx2._require(args.checkpoint, digest=CANDIDATES[epoch]["sha256"])
    if not rx2.BASE_RUNTIME.is_dir() or not rx2.EXPERIMENT_BOOK.is_dir() or not rx2.INTAKE_CODE.is_dir():
        raise HV1Error("pinned runtime, ExperimentBook, or intake source is absent")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema") != "ddm_cl1_hpac_capacity_checkpoint.v2"
        or checkpoint.get("epoch") != epoch
        or checkpoint.get("phase") != "discrete_qat"
        or checkpoint.get("deployment_weights") != "ema_shadow"
        or checkpoint.get("run_identity", {}).get("training_config", {}).get("profile") != "rx2_mc36"
    ):
        raise HV1Error(f"selected checkpoint is not the RX2-profile EMA QAT checkpoint at epoch {epoch}")
    if checkpoint.get("causal_state_sha256") is None:
        raise HV1Error("selected checkpoint lacks its causal-state hash")
    trainer = importlib.import_module("tools.train_ddm_cl1_hpac_capacity")
    if trainer._causal_state_sha256(checkpoint) != checkpoint["causal_state_sha256"]:
        raise HV1Error("selected checkpoint causal-state hash does not verify")
    history_row = _selected_history_row(checkpoint, epoch)
    lineage = checkpoint.get("resume_lineage")
    if not isinstance(lineage, list) or len(lineage) != 1:
        raise HV1Error("selected checkpoint resume lineage is absent or ambiguous")
    parent = lineage[0]
    if parent.get("sha256") != PARENT_CHECKPOINT_SHA256 or parent.get("epoch") != PARENT_EPOCH:
        raise HV1Error("selected checkpoint does not descend from the custodied e480b parent")
    source_digest = rx2._source().digest()
    if source_digest != rx1.EXPECTED_EVENT_SHA256:
        raise HV1Error("MC36 event-order symbol source changed")
    storage = rx2._require_bulk_free(args.required_free_gib)
    return {
        "schema": "ddm_hv1_harvest_preflight.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "checkpoint": rx2.file_record(args.checkpoint),
        "checkpoint_causal_state_sha256": checkpoint["causal_state_sha256"],
        "terminal_epoch": epoch,
        "selection_basis": CANDIDATES[epoch]["selection"],
        "selection_telemetry": history_row,
        "resume_parent": parent,
        "preflight_adaptation": (
            "terminal trainer report and artifact manifest are absent for a live mid-run "
            "checkpoint; replaced by the pinned selection sha, the checkpoint-embedded "
            "history row, and the e480b resume-lineage pin"
        ),
        "runner": rx2.file_record(Path(__file__)),
        "conventions_donor": rx2.file_record(Path(rx2.__file__)),
        "software": {"python": sys.version, "torch": torch.__version__},
        "base_archive": rx2.file_record(rx2.BASE_ARCHIVE),
        "expected_spatial": rx2.file_record(rx2.EXPECTED_SPATIAL),
        "source_manifest": rx2.file_record(rx2.SOURCE_MANIFEST),
        "source_event_order_sha256": source_digest,
        "work_root": str(rx2.WORK_ROOT),
        "bulk_root": str(rx2.BULK_ROOT),
        "incumbent": E480B_V2,
        **storage,
        "all_long_stages_checkpointed": True,
    }


def _projection(archive_bytes: int) -> dict[str, Any]:
    """Exact S projection against the e480b v2 incumbent under raw-output identity."""
    delta_bytes = archive_bytes - E480B_V2["archive_bytes"]
    delta_score = RATE_COEFFICIENT * delta_bytes
    return {
        "archive_bytes": archive_bytes,
        "delta_bytes_vs_e480b_v2": delta_bytes,
        "delta_score_vs_e480b_v2": delta_score,
        "projected_score": E480B_V2["score_recomputed_from_components"] + delta_score,
        "projection_basis": (
            "full raw-output byte identity holds d_seg and d_pose EXACTLY at the incumbent's "
            "measured CUDA values, so only the rate term moves"
        ),
        "residual_assumption": (
            "the measured identity is on the CPU decode axis while the incumbent's "
            "d_seg/d_pose were measured on CUDA; carrying them across assumes a "
            "device-deterministic integer receiver — sound, but an assumption on this "
            "arm's evidence until the T4 row fires"
        ),
        "authority": "[macOS-CPU advisory projection; the exact row remains owed on T4]",
        "score_claim": False,
    }


def manifest(_args: argparse.Namespace) -> dict[str, Any]:
    """Mirror the winning payload set plus every receipt to the Vertigo retention root."""
    final_path = rx2.BULK_ROOT / "FINAL_RESULT.json"
    if not final_path.is_file():
        raise HV1Error("manifest requires a complete finalize receipt")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if not final.get("complete"):
        raise HV1Error("finalize receipt is incomplete")
    winner = final["winner"]
    retained_root = rx2.WORK_ROOT / "retained/candidate"
    retained_root.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for role in ("archive", "repeat_archive", "member", "model", "residual", "token"):
        source = Path(winner[role]["path"])
        if rx2.file_record(source) != winner[role]:
            raise HV1Error(f"winner {role} failed custody before retention")
        target = retained_root / f"{role}{''.join(source.suffixes)}"
        shutil.copy2(source, target)
        record = rx2.file_record(target, role=role)
        if record["sha256"] != winner[role]["sha256"]:
            raise HV1Error(f"retention copy of {role} is not byte-identical")
        copied.append({**record, "source_path": str(source)})
    for name in ("PREFLIGHT.json", "PREPARE_RESULT.json", "FIT_RESULT.json", "FINAL_RESULT.json"):
        source = rx2.BULK_ROOT / name
        if source.is_file():
            shutil.copy2(source, rx2.WORK_ROOT / name)
    projection = _projection(winner["archive"]["bytes"])
    lines = [f"{row['sha256']}  {row['path']}" for row in copied]
    rx2.atomic_bytes(rx2.WORK_ROOT / "SHA256SUMS", ("\n".join(lines) + "\n").encode())
    result = {
        "schema": "ddm_hv1_retention_manifest.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "selected_checkpoint": final["selected_checkpoint"],
        "winner_variant": winner["variant"],
        "winner_representation": winner["representation"],
        "retained_payloads": copied,
        "bulk_root": str(rx2.BULK_ROOT),
        "work_root": str(rx2.WORK_ROOT),
        "bulk_retention_inventory": final["retention_inventory"],
        "raw_decode_identity": final["local_rgb_raw_decode"]["raw_identity_vs_mc36_cpu"],
        "decoded_token_identity": final["decoded_token_identity"],
        "incumbent": E480B_V2,
        "projection_vs_incumbent": projection,
        "admitted_vs_incumbent": projection["delta_score_vs_e480b_v2"] < 0.0,
        "all_materialized_payloads_retained": True,
    }
    rx2.atomic_json(rx2.WORK_ROOT / "RETENTION_MANIFEST.json", result)
    rx2.atomic_json(rx2.BULK_ROOT / "RETENTION_MANIFEST.json", result)
    return result


def fire_order(_args: argparse.Namespace) -> dict[str, Any]:
    """Emit the SEALED T4 fire order from the receipts. MAIN fires; this never dispatches.

    Every hash is read back off disk at emit time rather than copied from prose,
    so a stale or edited payload cannot be sealed by accident. The CUDA runtime
    tree stays MAIN-owned: this order pins the runtime the local full-raw decode
    actually ran through and requires MAIN to stage/verify the CUDA tree through
    the landed pq1 swap procedure before firing.
    """
    manifest_path = rx2.WORK_ROOT / "RETENTION_MANIFEST.json"
    if not manifest_path.is_file():
        raise HV1Error("fire order requires a complete retention manifest")
    retention = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not retention.get("complete"):
        raise HV1Error("retention manifest is incomplete")
    payloads = {row["role"]: row for row in retention["retained_payloads"]}
    for role, row in payloads.items():
        if rx2.file_record(Path(row["path"])) != {
            "path": row["path"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }:
            raise HV1Error(f"retained {role} failed custody at seal time")
    if payloads["archive"]["sha256"] != payloads["repeat_archive"]["sha256"]:
        raise HV1Error("determinism repeat is not byte-identical at seal time")
    if not retention["raw_decode_identity"] or not retention["decoded_token_identity"]:
        raise HV1Error("fire order requires the full raw and decoded-token identity gates")
    archive = payloads["archive"]
    decode = json.loads(
        (rx2.BULK_ROOT / "retained/cpu_decode/best_rx2/receipts/CPU_DECODE_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    projection = retention["projection_vs_incumbent"]
    tag = f"ddm_hv1_ep{_args.epoch:04d}_harvest_exact_contest_cuda_20260815"
    lane_id = f"lane_{tag}"
    results_dir = f"experiments/results/ddm_hv1_ep{_args.epoch:04d}_exact_contest_cuda_20260815"
    result = {
        "schema": "ddm_hv1_t4_sealed_fire_order.v1",
        "disposition": "QUEUED_WITH_A_FIRE_ORDER" if retention["admitted_vs_incumbent"] else "FOLDED",
        "owner": "MAIN",
        "score_claim": False,
        "axis": "[contest-CUDA] Tesla T4 n600",
        "candidate": {
            "candidate_id": (
                f"hv1_ep{_args.epoch:04d}_{retention['winner_variant']}"
                f"_{retention['winner_representation']}"
            ),
            "archive_bytes": archive["bytes"],
            "archive_sha256": archive["sha256"],
            "archive_path": archive["path"],
            "member_bytes": payloads["member"]["bytes"],
            "member_sha256": payloads["member"]["sha256"],
            "model_bytes": payloads["model"]["bytes"],
            "model_sha256": payloads["model"]["sha256"],
            "token_bytes": payloads["token"]["bytes"],
            "token_sha256": payloads["token"]["sha256"],
            "residual_sha256": payloads["residual"]["sha256"],
            "repeat_archive_sha256": payloads["repeat_archive"]["sha256"],
            "selected_checkpoint": retention["selected_checkpoint"],
        },
        "incumbent": E480B_V2,
        "projection_vs_incumbent": projection,
        "local_identity_evidence": {
            "raw_output_bytes": decode["raw_output"]["bytes"],
            "raw_output_sha256": decode["raw_output"]["sha256"],
            "raw_identity_vs_mc36_cpu": decode["raw_identity_vs_mc36_cpu"],
            "decoded_token_sha256": decode["decoded_token_sha256"],
            "decoded_token_identity": decode["decoded_token_identity"],
            "decode_runtime_entrypoint": decode["adapted_runtime_entrypoint"],
            "decode_receiver": decode["adapted_runtime_receiver"],
            "axis": decode["axis"],
        },
        "why_only_the_rate_term_moves": (
            "the full raw decode is byte-identical to the MC36 CPU decode the incumbent "
            "shares, so SegNet and PoseNet consume identical frames and d_seg/d_pose are "
            "the incumbent's measured CUDA values by construction"
        ),
        "preconditions": [
            "MAIN stages and verifies the CUDA runtime tree for these exact archive bytes "
            "through .omx/research/ddm_pq1_submission_packet_prep_20260815/SWAP_PROCEDURE.md "
            "steps VERIFY_SOURCE and STAGE_NEW_GENERATION; runtime custody is MAIN-owned",
            "reconciliation reports zero live remote calls and zero active claims",
            f"the unique lane claim {lane_id} succeeds",
            "every hash in this order still matches the retained payloads on disk",
        ],
        "claim_argv": [
            ".venv/bin/python",
            "tools/claim_lane_dispatch.py",
            "claim",
            "--lane-id",
            lane_id,
            "--platform",
            "modal",
            "--instance-job-id",
            tag,
            "--agent",
            "MAIN",
            "--status",
            "dispatching",
            "--notes",
            f"hv1 ep{_args.epoch} harvest candidate {archive['sha256'][:16]} @ {archive['bytes']} B; "
            f"rate-only delta {projection['delta_bytes_vs_e480b_v2']} B vs e480b v2",
        ],
        "canonical_chain": "experiments/modal_auth_eval.py::main",
        "command_argv_template": [
            "env",
            "PYTHONPATH=src:upstream:$PWD",
            ".venv/bin/modal",
            "run",
            "--detach",
            "experiments/modal_auth_eval.py::main",
            "--archive",
            archive["path"],
            "--expected-archive-sha256",
            archive["sha256"],
            "--submission-dir",
            "MAIN_STAGED_CUDA_RUNTIME_DIR",
            "--inflate-sh",
            "MAIN_STAGED_CUDA_RUNTIME_DIR/inflate.sh",
            "--expected-runtime-tree-sha256",
            "MAIN_STAGED_CUDA_RUNTIME_TREE_SHA256",
            "--gpu",
            "T4",
            "--scorer-device",
            "cuda",
            "--output-dir",
            results_dir,
            "--lane-id",
            lane_id,
            "--instance-job-id",
            tag,
            "--claim-agent",
            "MAIN",
            "--detach",
            "--provider-detach-ack",
        ],
        "close": {
            "poller_argv_template": [
                ".venv/bin/python",
                "tools/modal_harvest_poller.py",
                "--call-id",
                "CALL_ID_FROM_MODAL_AUTH_EVAL_SPAWN_JSON",
                "--output-dir",
                results_dir,
                "--result-name",
                "MODAL_REMOTE_RESULT.json",
                "--deadline-s",
                "9000",
                "--poll-s",
                "20",
            ],
            "rule": (
                "Launch the poller through tools/launch_detached_process.py immediately after "
                "the spawn receipt yields a call id; the poller owns terminal claim closure."
            ),
        },
        "dual_repeat": {
            "requirement": "fire the identical archive twice or reuse the retained repeat archive",
            "local_repeat_byte_identical": True,
            "repeat_archive_path": payloads["repeat_archive"]["path"],
        },
        "cpu_axis_note": (
            "the pq1 CPU-axis sealed order targets e480b v2 at swap_generation 0; re-targeting "
            "it to these bytes is the landed SWAP_PROCEDURE.md path and needs its own CPU row — "
            "no e480b receipt transfers to changed bytes"
        ),
        "consumer_store": ".omx/state/main_hot_state.md plus the canonical frontier pointer",
        "fire_trigger": (
            "MAIN only, on operator GO: the candidate is strictly smaller than the incumbent, "
            "carries local full-raw identity, and no other full-n600 scorer owns the slot"
        ),
    }
    rx2.atomic_json(rx2.WORK_ROOT / "T4_SEALED_FIRE_ORDER.json", result)
    rx2.atomic_json(
        REPO / f".omx/research/ddm_hv1_t4_sealed_fire_order_ep{_args.epoch:04d}_20260815.json",
        result,
    )
    return result


def _run_stage(stage: str, args: argparse.Namespace) -> dict[str, Any]:
    if stage == "preflight":
        result = hv1_preflight(args)
        rx2.atomic_json(rx2.BULK_ROOT / "PREFLIGHT.json", result)
        return result
    if stage == "prepare":
        return rx2.prepare(args)
    if stage == "export-base":
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)
        return rx2.export_base(args)
    if stage == "fit":
        return rx2.fit_tables(args)
    if stage == "materialize":
        return rx2.materialize_probabilities(args)
    if stage == "encode":
        return rx2.encode_rc64(args)
    if stage == "build":
        return rx2.build(args)
    if stage == "cpu-decode":
        return rx2.cpu_decode(args)
    if stage == "finalize":
        return rx2.finalize(args)
    if stage == "manifest":
        return manifest(args)
    if stage == "fire-order":
        return fire_order(args)
    raise HV1Error(f"unknown stage: {stage}")


def _run_all(args: argparse.Namespace) -> dict[str, Any]:
    """Drive the resumable chain; each delegated stage is already idempotent.

    ``--stop-after`` exists because the stages have very different memory
    footprints and the system admission gate prices a launch by ONE projected
    peak. Splitting the chain lets the measured 1.6 GiB coding stages be
    admitted at a small projection instead of being priced at the full-raw
    decoder's much larger one.
    """
    started = time.time()
    timeline: list[dict[str, Any]] = []
    state: dict[str, Any] = {"result": {}}

    def run(stage: str, variant: str | None = None) -> bool:
        """Run one stage, record its timing, and report whether to stop after it."""
        if variant is not None:
            args.variant = variant
        stage_started = time.time()
        state["result"] = _run_stage(stage, args)
        row: dict[str, Any] = {"stage": stage, "wall_s": time.time() - stage_started}
        progress = {"hv1_stage_complete": stage, "elapsed_s": time.time() - started}
        if variant is not None:
            row["variant"] = variant
            progress["variant"] = variant
        timeline.append(row)
        print(json.dumps(progress), flush=True)
        return args.stop_after == stage

    variants: list[str] = []
    for stage in ("preflight", "prepare", "export-base", "fit"):
        if run(stage):
            break
    else:
        variants = list(rx2._fit_result()["selected_for_full_n600_real_rc64"])
        stop_after_coding = False
        for variant in variants:
            for stage in ("materialize", "encode", "build"):
                stop_after_coding = run(stage, variant) or stop_after_coding
        if not stop_after_coding:
            for stage in ("cpu-decode", "finalize", "manifest", "fire-order"):
                if run(stage):
                    break
    return {
        **state["result"],
        "variants": variants,
        "timeline": timeline,
        "total_wall_s": time.time() - started,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("stage", choices=(*STAGES, "all"))
    value.add_argument("--stop-after", choices=STOP_AFTER_CHOICES, default=None)
    value.add_argument("--epoch", type=int, choices=sorted(CANDIDATES), default=DEFAULT_EPOCH)
    value.add_argument("--checkpoint", type=Path, default=None)
    value.add_argument("--variant", default="neutral")
    value.add_argument("--start-frame", type=int, default=0)
    value.add_argument("--end-frame", type=int, default=600)
    value.add_argument("--torch-threads", type=int, default=4)
    value.add_argument("--required-free-gib", type=int, default=32)
    value.add_argument("--brotli", default=shutil.which("brotli") or "brotli")
    return value


def main() -> None:
    from tac.admission_guard import assert_governed_admission

    assert_governed_admission("ddm_hv1_harvest_compose")
    args = parser().parse_args()
    if args.checkpoint is None:
        args.checkpoint = _checkpoint_path(args.epoch)
    _bind_hv1_conventions(args.epoch)
    if not 0 <= args.start_frame < args.end_frame <= 600:
        raise SystemExit("invalid frame interval")
    result = _run_all(args) if args.stage == "all" else _run_stage(args.stage, args)
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, (list, dict))}, indent=2))


if __name__ == "__main__":
    main()
