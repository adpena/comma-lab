#!/usr/bin/env python3
"""Build and seal the byte-gated RB1 born-small renderer route without launch.

This runner is scorer-free and Metal-free.  It prices retained WD3 rungs with
the real Brotli-q11 pack path, retains complete RB1 stub archives, proves exact
BS3/WD3 parse-back through a patched copy of the shipped GB1 receiver, creates
scorer-free birth checkpoints, and seals launch-disabled aligned configs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np
import torch

REPO: Final = Path(__file__).resolve().parents[1]
SRC: Final = REPO / "src"
for root in (REPO, SRC):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments import ddm_rb1_born_small_receiver as rb1_receiver
from experiments import ddm_wd3_scorer_aware_width_distillation as wd3
from experiments import ddm_wd3_student_receiver as wd3_receiver

OUTPUT: Final = wd3.RB1_OUTPUT_ROOT
BODY_RESULT: Final = Path("/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/BODY_RESULT.json")
GB1_RUNTIME: Final = Path("/Volumes/APDataStore/pact/ddm_gb1_groupbin8_conditioning/runtime_fire_v1")
W96B_CONFIG: Final = Path(
    "/Volumes/APDataStore/pact/ddm_w96a_aligned_window/launch_requests/aligned_seed_20260815.json"
)
W96B_MEMO: Final = REPO / ".omx/research/ddm_w96b_aligned_loss_implementation_20260826.md"
COMPLETE_CEILING: Final = 137_986
MINIMUM_CREDIT: Final = 2_000
BODY_BYTES: Final = 101_150
PROJECTED_PER_CONFIG_RETENTION_BYTES: Final = 12_489_721_856
TEACHER_MASTER_BYTES: Final = 1_831_204_800
RESERVE_BYTES: Final = 8 << 30

RUNG_SOURCES: Final = {
    "D56": Path(
        "/Volumes/APDataStore/pact/ddm_wd3_scorer_aware_width_distillation/D56/D56/"
        "retained/evaluations/epoch_0065_n60/candidate"
    ),
    "F64": Path(
        "/Volumes/APDataStore/pact/ddm_wd3_scorer_aware_width_distillation/F64/F64/"
        "retained/evaluations/epoch_0065_n60/candidate"
    ),
}


class RB1BuildError(RuntimeError):
    """RB1 refused an unclosed byte, object, receiver, config, or custody claim."""


def file_record(path: Path) -> dict[str, Any]:
    return wd3.file_record(path)


def retain_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    expected = {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if path.is_file():
        if file_record(path) != expected:
            raise RB1BuildError(f"refusing to overwrite different retained bytes: {path}")
        return expected
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return file_record(path)


def retain_json(path: Path, value: object) -> dict[str, Any]:
    return retain_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n").encode(),
    )


def retain_sealed_json(path: Path, value: object) -> dict[str, Any]:
    """Preserve a pre-completion stale seal, then write the current exact seal."""

    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n").encode()
    expected_sha = hashlib.sha256(payload).hexdigest()
    if path.is_file() and file_record(path)["sha256"] != expected_sha:
        if (OUTPUT / "BUILD_RECEIPT.json").is_file():
            raise RB1BuildError(f"completed RB1 seal drifted: {path}")
        prior = file_record(path)
        superseded = OUTPUT / "configs/superseded_partial" / f"{path.stem}.{prior['sha256']}.json"
        superseded.parent.mkdir(parents=True, exist_ok=True)
        if superseded.exists():
            if file_record(superseded)["sha256"] != prior["sha256"]:
                raise RB1BuildError("superseded partial config identity collision")
            path.unlink()
        else:
            os.replace(path, superseded)
        retain_json(
            superseded.with_suffix(".recovery.json"),
            {
                "schema": "ddm_rb1_partial_seal_recovery.v1",
                "original_path": prior["path"],
                "preserved_payload": file_record(superseded),
                "reason": "builder source changed before any birth receipt or completed build existed",
                "recoverable": True,
            },
        )
    return retain_bytes(path, payload)


def _supersede_completed_build_if_builder_changed() -> dict[str, Any] | None:
    """Move a pre-launch completed seal aside when the sealed WD3 source changes."""

    build_path = OUTPUT / "BUILD_RECEIPT.json"
    if not build_path.is_file():
        return None
    prior = json.loads(build_path.read_text(encoding="utf-8"))
    prior_builder = prior.get("provenance", {}).get("wd3_builder", {}).get("sha256")
    current_builder = wd3.sha256_file(Path(wd3.__file__).resolve())
    if prior_builder == current_builder:
        return None
    if not isinstance(prior_builder, str) or len(prior_builder) != 64:
        raise RB1BuildError("completed RB1 build lacks a canonical WD3 source identity")
    superseded_root = OUTPUT / "superseded_builds" / prior_builder
    if superseded_root.exists():
        raise RB1BuildError("RB1 completed-build supersession destination already exists")
    superseded_root.mkdir(parents=True)
    moved = []
    for relative in ("configs", "training", "MAIN_FIRE_HANDOFF.json", "BUILD_RECEIPT.json"):
        source = OUTPUT / relative
        if not source.exists():
            raise RB1BuildError(f"completed RB1 build artifact is absent: {source}")
        destination = superseded_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        if destination.is_file():
            moved.append(
                {
                    "original_path": str(source),
                    "preserved_path": str(destination),
                    "payload": file_record(destination),
                }
            )
        else:
            files = [file_record(path) for path in sorted(destination.rglob("*")) if path.is_file()]
            moved.append(
                {
                    "original_path": str(source),
                    "preserved_path": str(destination),
                    "file_count": len(files),
                    "total_bytes": sum(record["bytes"] for record in files),
                    "manifest_sha256": wd3.canonical_sha256(files),
                    "files": files,
                }
            )
    manifest = {
        "schema": "ddm_rb1_completed_build_supersession.v1",
        "reason": "P0 post-atomic-rename teacher-master resume recovery changed the sealed WD3 builder",
        "prior_builder_sha256": prior_builder,
        "replacement_builder_sha256": current_builder,
        "prior_launch_now": prior.get("launch_now"),
        "prior_training_launched": prior.get("training_launched"),
        "prior_scorer_invocations": prior.get("scorer_invocations"),
        "prior_metal_invocations": prior.get("metal_invocations"),
        "moved": moved,
        "recoverable": True,
    }
    retain_json(superseded_root / "SUPERSESSION_MANIFEST.json", manifest)
    return manifest


def _body_binding() -> tuple[dict[str, Any], dict[str, Any]]:
    body_record = file_record(BODY_RESULT)
    if body_record["sha256"] != "ea3ce5b18ec88d1451c5cd90cd49afc97ee1e52b67cebfe1524aa7abf49f84f3":
        raise RB1BuildError("BS3 BODY_RESULT identity differs")
    body = json.loads(BODY_RESULT.read_text(encoding="utf-8"))
    archive = body["body"]["archive"]
    direct = body["body"]["direct_decode"]
    parsed = body["body"]["archive_parseback_decode"]
    semantic = body["retained_sources"]["semantic_renderer"]
    if (
        archive["bytes"] != BODY_BYTES
        or archive["sha256"] != "5743f0ac7e8881e970ef8ba53c4bee3fd2a7a6157d2a50d381fd609ae624fea6"
        or direct["bytes"] != 117_964_800
        or parsed["bytes"] != 117_964_800
        or direct["sha256"] != parsed["sha256"]
        or direct["sha256"] != "2884c5701dc2b2059df0e9f8e4ee3ed81809457b127a48ad3fd3fb6f7a17152b"
        or direct.get("corrections") != 0
        or parsed.get("corrections") != 0
        or semantic["bytes"] != 30_856
        or semantic["sha256"] != "39d1be52ba62933498395c48ce4d9482f37db097d504da76c2a321efe3e4a76f"
    ):
        raise RB1BuildError("BS3 exact body/parse-back binding differs")
    for record in (archive, direct, parsed, semantic):
        if file_record(Path(record["path"])) != {k: record[k] for k in ("path", "bytes", "sha256")}:
            raise RB1BuildError(f"BS3 retained payload drifted: {record['path']}")
    return body, body_record


def _runtime_receiver_records() -> dict[str, dict[str, Any]]:
    return {relative: file_record(GB1_RUNTIME / relative) for relative in wd3.RB1_RECEIVER_RELATIVE_FILES}


def _target_object(body: Mapping[str, Any], admitted: Sequence[str]) -> dict[str, Any]:
    return {
        "schema": "ddm_rb1_born_small_target_object.v1",
        "body_result": str(BODY_RESULT),
        "body_result_sha256": "ea3ce5b18ec88d1451c5cd90cd49afc97ee1e52b67cebfe1524aa7abf49f84f3",
        "body_archive": body["body"]["archive"]["path"],
        "body_archive_bytes": BODY_BYTES,
        "body_archive_sha256": body["body"]["archive"]["sha256"],
        "direct_tokens": body["body"]["direct_decode"]["path"],
        "parsed_tokens": body["body"]["archive_parseback_decode"]["path"],
        "tokens_bytes": body["body"]["archive_parseback_decode"]["bytes"],
        "tokens_sha256": body["body"]["archive_parseback_decode"]["sha256"],
        "semantic_renderer": body["retained_sources"]["semantic_renderer"]["path"],
        "semantic_renderer_bytes": body["retained_sources"]["semantic_renderer"]["bytes"],
        "semantic_renderer_sha256": body["retained_sources"]["semantic_renderer"]["sha256"],
        "gb1_runtime": str(GB1_RUNTIME),
        "gb1_receiver_files": _runtime_receiver_records(),
        "candidate_exporter": str(Path(rb1_receiver.__file__).resolve()),
        "candidate_exporter_sha256": wd3.sha256_file(Path(rb1_receiver.__file__).resolve()),
        "teacher_master": str(OUTPUT / "retained/teacher/teacher_master_camera.rgb.u8"),
        "teacher_master_receipt": str(OUTPUT / "retained/teacher/TEACHER_MASTER_RECEIPT.json"),
        "admitted_arms": list(admitted),
        "complete_archive_ceiling_bytes": COMPLETE_CEILING,
        "minimum_complete_archive_credit_bytes": MINIMUM_CREDIT,
    }


def _price_rungs(body: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    body_payload = Path(body["body"]["archive"]["path"]).read_bytes()
    rows = []
    admitted = []
    for arm, source in RUNG_SOURCES.items():
        packet_path = source / "semantic.wd3q"
        retained_stream_path = source / "semantic.br"
        packet = packet_path.read_bytes()
        semantic = brotli.compress(packet, mode=brotli.MODE_GENERIC, quality=11)
        repeated = brotli.compress(packet, mode=brotli.MODE_GENERIC, quality=11)
        if semantic != repeated or brotli.decompress(semantic) != packet:
            raise RB1BuildError(f"{arm} real coder repeat or parse-back differs")
        if retained_stream_path.is_file() and retained_stream_path.read_bytes() != semantic:
            raise RB1BuildError(f"{arm} retained WD3 pack-path stream differs")
        root = OUTPUT / "retained/byte_gate" / arm
        stream_record = retain_bytes(root / "semantic.br", semantic)
        repeat_record = retain_bytes(root / "semantic.repeat.br", repeated)
        archive = rb1_receiver.pack_archive_bytes(body_payload, semantic)
        archive_repeat = rb1_receiver.pack_archive_bytes(body_payload, repeated)
        if archive != archive_repeat:
            raise RB1BuildError(f"{arm} complete RB1 archive repeat differs")
        archive_record = retain_bytes(root / "archive.zip", archive)
        archive_repeat_record = retain_bytes(root / "archive.repeat.zip", archive_repeat)
        parsed_body, parsed_stream = rb1_receiver.parse_archive_bytes(archive)
        model = rb1_receiver.unpack_renderer(parsed_stream)
        if parsed_body != body_payload or wd3_receiver.pack_student(
            model, wd3_receiver.packet_allocation(packet)
        ) != packet:
            raise RB1BuildError(f"{arm} complete receiver parse-back differs")
        credit = COMPLETE_CEILING - archive_record["bytes"]
        passes = credit >= MINIMUM_CREDIT
        if passes:
            admitted.append(arm)
        rows.append(
            {
                "arm": arm,
                "source_packet": file_record(packet_path),
                "source_retained_real_coder": (
                    file_record(retained_stream_path) if retained_stream_path.is_file() else None
                ),
                "real_coder": "brotli.MODE_GENERIC quality=11",
                "renderer_payload": stream_record,
                "renderer_repeat": repeat_record,
                "renderer_bytes": stream_record["bytes"],
                "body_bytes": BODY_BYTES,
                "rb1_framing_bytes": archive_record["bytes"] - BODY_BYTES - stream_record["bytes"],
                "complete_archive": archive_record,
                "complete_archive_repeat": archive_repeat_record,
                "complete_archive_bytes": archive_record["bytes"],
                "complete_archive_ceiling_bytes": COMPLETE_CEILING,
                "complete_archive_credit_bytes": credit,
                "minimum_credit_bytes": MINIMUM_CREDIT,
                "gate": "ADMIT" if passes else "REJECT",
                "packet_parseback_exact": True,
                "archive_repeat_byte_identical": True,
            }
        )
    return rows, admitted


def _load_patched_receiver(path: Path) -> Any:
    module_name = "_ddm_rb1_patched_gb1_wd3_receiver"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RB1BuildError("patched shipped-GB1 WD3 receiver could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        sys.modules.pop(module_name, None)
    return module


def _receiver_stub_proof(
    body: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    patched_summary: dict[str, Any]
    with tempfile.TemporaryDirectory(prefix="ddm_rb1_runtime_patch_") as temporary:
        patched_root = Path(temporary) / "runtime"
        patch = rb1_receiver.patch_gb1_runtime_tree(GB1_RUNTIME, patched_root)
        patched = _load_patched_receiver(patched_root / "cpr1/wd3_receiver.py")
        patched_summary = {
            "schema": patch["schema"],
            "source_tree": patch["source_tree"],
            "student_magic": patch["student_magic"],
            "inactive_wans_sd1m_sm3r_branches_retained": patch[
                "inactive_wans_sd1m_sm3r_branches_retained"
            ],
            "file_count": len(patch["files"]),
            "total_file_bytes": sum(int(record["bytes"]) for record in patch["files"]),
            "manifest_sha256": wd3.canonical_sha256(patch["files"]),
            "modified_receiver_files": {
                relative: next(record for record in patch["files"] if record["path"] == relative)
                for relative in (
                    "cpr1/wd3_receiver.py",
                    "runtime/f26_inflate.py",
                    "runtime/residual_archive.py",
                )
            },
            "scratch_disposition": "context-managed code-build scratch removed after exact proof",
        }
        first = rows[0]
        packet = Path(first["source_packet"]["path"]).read_bytes()
        local_model = wd3_receiver.unpack_student(packet).eval()
        shipped_model = patched.unpack_student(packet).eval()
        tokens_map = np.memmap(
            Path(body["body"]["archive_parseback_decode"]["path"]),
            mode="r",
            dtype=np.uint8,
            shape=(wd3_receiver.N, wd3_receiver.EVAL_H, wd3_receiver.EVAL_W),
        )
        tokens = torch.from_numpy(np.asarray(tokens_map[:1]).copy())
        indices = torch.zeros(1, dtype=torch.long)
        local = rb1_receiver.render_camera_uint8(local_model, tokens, indices)
        shipped = rb1_receiver.render_camera_uint8(shipped_model, tokens, indices)
        del tokens_map
        if not torch.equal(local, shipped):
            raise RB1BuildError("patched shipped-GB1 WD3 forward differs on exact BS3 parsed tokens")
        output = local.cpu().numpy()
        output_record = retain_bytes(OUTPUT / "retained/receiver_stub/frame_0000.rgb.u8", output.tobytes())
        repeat_record = retain_bytes(
            OUTPUT / "retained/receiver_stub/frame_0000.repeat.rgb.u8",
            shipped.cpu().numpy().tobytes(),
        )
    return {
        "schema": "ddm_rb1_shipped_gb1_receiver_stub_proof.v1",
        "complete": True,
        "body_archive": file_record(Path(body["body"]["archive"]["path"])),
        "body_archive_parseback_tokens": file_record(
            Path(body["body"]["archive_parseback_decode"]["path"])
        ),
        "patched_shipped_runtime": patched_summary,
        "renderer_packet": rows[0]["source_packet"],
        "receiver_output": output_record,
        "receiver_output_repeat": repeat_record,
        "local_vs_patched_shipped_receiver_byte_identical": True,
        "bs3_direct_equals_archive_parseback": True,
        "training_launched": False,
        "scorer_invocations": 0,
        "metal_invocations": 0,
    }


def _aligned_objective() -> dict[str, Any]:
    return {
        "scoreaware": True,
        "seg_score_coefficient": 100.0,
        "pose_exact_nonlinear": True,
        "temperature": 2.0,
        "adaptive_duals": True,
        "decode_mse_ceiling": wd3.DECODE_MSE_CEILING,
        "packet_quantizer_in_loop": True,
        "seg_loss_law": wd3.SEG_LOSS_EXPECTED_FLIP_MARGIN,
        "expected_flip_tau_start": wd3.ALIGNED_TAU_START,
        "expected_flip_tau_end": wd3.ALIGNED_TAU_END,
        "full_window_epochs": wd3.ALIGNED_FULL_WINDOW_EPOCHS,
        "pose_start_step": 0,
    }


def _base_config(
    *,
    arm: str,
    seed: int,
    target: Mapping[str, Any],
    action: str,
    minimum_free_bytes: int,
) -> dict[str, Any]:
    inherited = json.loads(W96B_CONFIG.read_text(encoding="utf-8"))
    run_root = OUTPUT / "training" / f"seed_{seed}" / arm
    resume_root = run_root / "resume"
    birth_path = resume_root / f"{arm}_birth.pt"
    return {
        "schema": wd3.SCHEMA,
        "action": action,
        "output": str(run_root),
        "seed": seed,
        "device": "cpu" if action == "prepare_arm_birth" else "mps",
        "chunk_pairs": 60,
        "retain_all_payloads": True,
        "checkpoint_every_epochs": 5,
        "minimum_free_bytes": minimum_free_bytes,
        "base_receipt": str(wd3.BASE_RECEIPT),
        "teacher_cache_result": str(OUTPUT / "teacher_scorer_cache/TEACHER_SCORER_CACHE_RESULT.json"),
        "resume_from": str(birth_path),
        "resume_root": str(resume_root),
        "arm": arm,
        "completed_arms": [],
        "negative_confirmed_arms": [],
        "capacity_pressure_confirmed": False,
        "real_coder_override_dense_w96": False,
        "subsets": copy.deepcopy(inherited["subsets"]),
        "objective": _aligned_objective(),
        "optimizer": {
            "lr": 2.0e-5,
            "weight_decay": 1.0e-4,
            "grad_clip": 1.0,
            "dual_step": 1.0e-3,
            "reset_ramp_divisor": 3.16,
        },
        "epochs": 65,
        "batch_pairs": 1,
        "scorer_lane": {"claimed": False, "claim_id": None, "agent": "MAIN", "platform": "macos-cpu"},
        "metal_lane": {"claimed": False, "claim_id": None, "agent": "MAIN", "platform": "macos-mps"},
        "launch_authorized": False,
        "r5_exit_verified": False,
        "source_pins": wd3.source_pin_contract(),
        "expected_builder_sha256": wd3.sha256_file(Path(wd3.__file__).resolve()),
        "expected_receiver_sha256": wd3.sha256_file(Path(wd3_receiver.__file__).resolve()),
        "stage_a": None,
        "cheap_to_shrink": {
            "mode": "off",
            "allocation_family": "waterfill_ceiling",
            "uniform_bits": [],
            "rung_weights": [],
            "base_weight": 1.0,
            "sampler_seed": seed,
        },
        "evaluation_retention": {
            "schema": "ddm_w96b_evaluation_retention.v1",
            "mode": "content_addressed_chunks_v1",
            "cas_root": str(wd3.RB1_CAS_ROOT),
            "compact_after_verify": True,
        },
        "target_object": dict(target),
    }


def _storage_memory_projection(config_count: int) -> dict[str, Any]:
    retained_path = OUTPUT / "STORAGE_MEMORY_PREFLIGHT.json"
    if retained_path.is_file():
        retained = json.loads(retained_path.read_text(encoding="utf-8"))
        if (
            retained.get("schema") != "ddm_rb1_real_config_storage_memory_projection.v1"
            or retained.get("config_count") != config_count
            or retained.get("projected_per_config_retention_bytes")
            != PROJECTED_PER_CONFIG_RETENTION_BYTES
            or retained.get("target_teacher_master_bytes") != TEACHER_MASTER_BYTES
            or retained.get("reserve_bytes") != RESERVE_BYTES
        ):
            raise RB1BuildError("retained RB1 storage/memory projection differs")
        return retained
    retention = PROJECTED_PER_CONFIG_RETENTION_BYTES * config_count + TEACHER_MASTER_BYTES
    required = retention + RESERVE_BYTES
    OUTPUT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(OUTPUT).free
    physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    peak_anchor = 7_511 * (1 << 20)
    required_memory = 2 * peak_anchor
    return {
        "schema": "ddm_rb1_real_config_storage_memory_projection.v1",
        "config_count": config_count,
        "retention_source": file_record(W96B_MEMO),
        "retention_mode": "four aligned runs at the measured W96B per-config logical demand; no unmeasured CAS credit",
        "projected_per_config_retention_bytes": PROJECTED_PER_CONFIG_RETENTION_BYTES,
        "target_teacher_master_bytes": TEACHER_MASTER_BYTES,
        "projected_retention_bytes": retention,
        "reserve_bytes": RESERVE_BYTES,
        "required_free_bytes": required,
        "observed_free_bytes": free,
        "storage_status": "PASS" if free >= required else "BLOCKED_SR3_RECLAIM_REQUIRED",
        "memory_anchor": "measured F64 peak RSS 7511 MiB; doubled for fail-closed launch admission",
        "projected_peak_rss_bytes": peak_anchor,
        "required_physical_memory_bytes": required_memory,
        "observed_physical_memory_bytes": physical,
        "memory_status": "PASS" if physical >= required_memory else "BLOCKED_MEMORY",
        "cleanup_policy": "certify-or-block; CAS retains every evaluation payload and every stage checkpoint",
    }


def _seal_configs(
    body: Mapping[str, Any],
    admitted: Sequence[str],
    projection: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = _target_object(body, admitted)
    # This validates every large retained identity now, before a config is emitted.
    wd3.target_object_binding({"target_object": target})
    config_rows = []
    for arm in admitted:
        for seed in wd3.RB1_SEEDS:
            birth = _base_config(
                arm=arm,
                seed=seed,
                target=target,
                action="prepare_arm_birth",
                minimum_free_bytes=1 << 30,
            )
            birth_path = OUTPUT / "configs" / f"{arm.lower()}_seed_{seed}.birth.json"
            birth_record = retain_sealed_json(birth_path, birth)
            birth_receipt_path = Path(birth["output"]) / f"{arm}_BIRTH_RECEIPT.json"
            if birth_receipt_path.is_file():
                birth_receipt = json.loads(birth_receipt_path.read_text(encoding="utf-8"))
                if birth_receipt.get("config_sha256") != wd3.canonical_sha256(birth):
                    raise RB1BuildError("retained birth receipt config identity differs")
            else:
                birth_receipt = wd3.prepare_arm_birth(birth)
            train = _base_config(
                arm=arm,
                seed=seed,
                target=target,
                action="train",
                minimum_free_bytes=int(projection["required_free_bytes"]),
            )
            train_path = OUTPUT / "configs" / f"{arm.lower()}_seed_{seed}.train.json"
            train_record = retain_sealed_json(train_path, train)
            fire_order = wd3.compile_fire_order(train)
            fire_path = OUTPUT / "configs" / f"{arm.lower()}_seed_{seed}.fire_order.json"
            fire_record = retain_sealed_json(fire_path, fire_order)
            if fire_order["disposition"] != "BLOCKED_NOT_LAUNCHABLE":
                raise RB1BuildError("launch-disabled RB1 config unexpectedly compiled ready")
            config_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "birth_config": birth_record,
                    "birth_receipt": file_record(birth_receipt_path),
                    "birth_checkpoint": birth_receipt["checkpoint"],
                    "train_config": train_record,
                    "train_config_sha256": wd3.canonical_sha256(train),
                    "fire_order": fire_record,
                    "launch_now": False,
                    "resumable_from_disk": True,
                    "checkpoint_every_epochs": 5,
                    "terminal_epoch": 65,
                    "pose_start_step": 0,
                }
            )
    return config_rows, target


def build() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _supersede_completed_build_if_builder_changed()
    body, body_result_record = _body_binding()
    rows, admitted = _price_rungs(body)
    gate_receipt = {
        "schema": "ddm_rb1_complete_archive_byte_gate.v1",
        "body_result": body_result_record,
        "body_archive_bound_by_reference_not_copied": True,
        "complete_archive_ceiling_bytes": COMPLETE_CEILING,
        "minimum_complete_archive_credit_bytes": MINIMUM_CREDIT,
        "rows": rows,
        "admitted_arms": admitted,
        "status": "ADMIT" if admitted else "CLOSE_AT_BUILD_SCOPE",
        "real_coder_outputs_and_repeats_retained": True,
        "training_launched": False,
        "scorer_invocations": 0,
        "metal_invocations": 0,
    }
    retain_json(OUTPUT / "BYTE_GATE.json", gate_receipt)
    if not admitted:
        result = {
            "schema": "ddm_rb1_born_small_renderer_build.v1",
            "complete": True,
            "disposition": "CLOSED_BYTE_GATE",
            "byte_gate": gate_receipt,
            "configs": [],
            "frontier_moved": False,
        }
        retain_json(OUTPUT / "BUILD_RECEIPT.json", result)
        return result
    admitted_rows = [row for row in rows if row["arm"] in admitted]
    receiver_proof = _receiver_stub_proof(body, admitted_rows)
    retain_json(OUTPUT / "RECEIVER_STUB_PROOF.json", receiver_proof)
    projection = _storage_memory_projection(len(admitted) * len(wd3.RB1_SEEDS))
    retain_json(OUTPUT / "STORAGE_MEMORY_PREFLIGHT.json", projection)
    configs, target = _seal_configs(body, admitted, projection)
    handoff = {
        "schema": "ddm_rb1_typed_main_fire_handoff.v1",
        "disposition": "QUEUED_AFTER_SR3_BS4_W96B",
        "owner": "MAIN",
        "consumer_store": str(OUTPUT) + "/",
        "fire_trigger": (
            "ddm_sr3 is green with at least "
            f"{projection['required_free_bytes']} free bytes for this four-config retention projection; "
            "the #1304 BS4 then W96B sequence is complete; the Metal slot is free; MAIN claims distinct "
            "scorer and Metal lanes, materializes the shared target teacher/cache, then fires the sealed configs"
        ),
        "config_order": [
            f"{row['arm']} seed {row['seed']}" for row in configs
        ],
        "launch_now": False,
        "training_launched": False,
        "scorer_invocations": 0,
        "metal_invocations": 0,
        "storage_memory_projection": projection,
    }
    retain_json(OUTPUT / "MAIN_FIRE_HANDOFF.json", handoff)
    result = {
        "schema": "ddm_rb1_born_small_renderer_build.v1",
        "complete": True,
        "axis": "[scorer-free/Metal-free build and exact byte/receiver authority]",
        "disposition": "BUILT_SEALED_STORAGE_GATED_NO_LAUNCH",
        "byte_gate": gate_receipt,
        "target_object": target,
        "receiver_stub_proof": receiver_proof,
        "storage_memory_projection": projection,
        "configs": configs,
        "handoff": handoff,
        "provenance": {
            "runner": file_record(Path(__file__).resolve()),
            "rb1_receiver": file_record(Path(rb1_receiver.__file__).resolve()),
            "wd3_builder": file_record(Path(wd3.__file__).resolve()),
            "wd3_receiver": file_record(Path(wd3_receiver.__file__).resolve()),
            "w96b_reference_config": file_record(W96B_CONFIG),
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "seeds": list(wd3.RB1_SEEDS),
        },
        "all_materialized_payloads_retained": True,
        "body_archive_copied": False,
        "launch_now": False,
        "training_launched": False,
        "scorer_invocations": 0,
        "metal_invocations": 0,
        "score_claim": False,
        "promotion_eligible": False,
        "frontier_moved": False,
    }
    retain_json(OUTPUT / "BUILD_RECEIPT.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--output", type=Path, default=OUTPUT)
    root.add_argument("--launch-now", action="store_true")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.output.resolve() != OUTPUT.resolve():
        raise RB1BuildError(f"RB1 output must be exactly {OUTPUT}")
    if args.launch_now:
        raise RB1BuildError("RB1 charter forbids launch")
    result = build()
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "complete": result["complete"],
                "disposition": result["disposition"],
                "admitted_arms": result["byte_gate"]["admitted_arms"],
                "config_count": len(result.get("configs", [])),
                "launch_now": result.get("launch_now", False),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
