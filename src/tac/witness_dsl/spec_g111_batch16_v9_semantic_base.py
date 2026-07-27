# SPDX-License-Identifier: MIT
"""Typed cold V9 producer bound to the physical G109 batch-16 target capsule.

This is a descendant of the settled V9 ideal mod-32 geometry, not a parallel
hand-written argv.  G109 is reopened recursively at compile time and again by
the trainer against the active source-frame cache.  The target capsule remains
encoder-only evidence; only a later exact G105 packet may enter a candidate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tac.witness_control.taskspace_v9_training_target_capsule_v1 import (
    PRODUCTION_BATCH_PAIRS,
    PRODUCTION_PAIR_COUNT,
    V9TrainingTargetCapsuleLoaderV1,
    sha256_file,
)
from tac.witness_dsl.spec_v9_cgauge import (
    DSL_IDENTITY_EQUATION_ID,
    _merge_lever_constant_manifests,
    attach_flag_custody,
    compile_v9_cgauge_ideal_launch_config,
)
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    V9PolarFourierConfigV1,
    V9RuntimeConfigV1,
    Y1WireCodecV1,
    compile_from_y1_state,
    encode_packet_y1_variant,
)
from tac.witness_dsl.typed_config import TypedLever, build_launch_manifest
from tac.witness_dsl.v10_factor2_selected_preimage_v1 import (
    SCHEMA as V10_FACTOR2_SELECTED_PREIMAGE_SCHEMA,
)

PROGRAM_NAME = "g111_batch16_v9_semantic_base"
TARGET_LEVER_NAME = "g111_physical_batch16_target_custody"
TAIL_STOP_FLAG = "--tail-stop-marginal-s"
SKIP_BOOT_BASELINE_FLAG = "--skip-boot-baseline-verdict"
G111_TAIL_STOP_MARGINAL_S = 0.0
TARGET_CONTRACT_SCHEMA = "tac.g111_batch16_v9_semantic_base_target_contract.v2"
Y1_RATE_ARBITRATION_SCHEMA = "tac.g105_y1_outer_archive_rate_arbitration.v1"
SOURCE_VIDEO_BYTES = 37_545_489
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DRY_START_SEARCH_ROOTS = (
    _REPO_ROOT / "experiments" / "results",
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)


class G111Batch16V9SemanticBaseError(RuntimeError):
    """The physical target, typed composition, or cold-producer contract differs."""


def structural_semantic_rate_preflight() -> dict[str, Any]:
    """Derive the fixed G105 semantic packet load before any expensive training.

    The raw Y1 variant has value-independent section lengths, so zero tensors
    measure its exact structural byte load without pretending to predict
    learned entropy or the complete G110 archive.
    """

    basis = V9PolarFourierConfigV1(
        n_scales=4,
        n_orient0=6,
        f0=2.0,
        base=2.0,
        n_iso=4,
        max_freq=64.0,
    )
    config = V9RuntimeConfigV1(
        input_dim=basis.input_dim,
        hidden_dim=96,
        hidden_layer_count=4,
        modulation_dim=32,
        softmax_temp=0.31,
        hosc_beta=3.177,
        hosc_omega=1.0,
        chroma=True,
        film_per_layer=False,
        film_concat_code=False,
        basis=basis,
    )
    zeros = np.zeros
    params = {
        "in_proj.weight": zeros((96, basis.input_dim), dtype=np.float32),
        "in_proj.bias": zeros((96,), dtype=np.float32),
        "film.weight": zeros((2 * 96 * 4, 32), dtype=np.float32),
        "film.bias": zeros((2 * 96 * 4,), dtype=np.float32),
        "out_sdf.weight": zeros((5, 96), dtype=np.float32),
        "out_sdf.bias": zeros((5,), dtype=np.float32),
        "out_tex.weight": zeros((3, 96), dtype=np.float32),
        "out_tex.bias": zeros((3,), dtype=np.float32),
        "palette": zeros((5, 3), dtype=np.float32),
    }
    for layer in range(4):
        params[f"hidden.{layer}.weight"] = zeros((96, 96), dtype=np.float32)
        params[f"hidden.{layer}.bias"] = zeros((96,), dtype=np.float32)
    program = compile_from_y1_state(
        config=config,
        params=params,
        y1_code=zeros((PRODUCTION_PAIR_COUNT, 32), dtype=np.float32),
    )
    raw_packet = encode_packet_y1_variant(
        program,
        codec=Y1WireCodecV1.RAW_I16_LE,
    )
    model_data_bytes = sum(len(tensor.data) for tensor in program.tensors)
    return {
        "schema": "tac.g111_structural_semantic_rate_preflight.v1",
        "authority": "exact_G105_value_independent_raw_section_lengths",
        "input_dim": basis.input_dim,
        "counted_tensor_values": sum(int(np.prod(tensor.shape)) for tensor in program.tensors),
        "model_data_bytes": model_data_bytes,
        "raw_y1_data_bytes": PRODUCTION_PAIR_COUNT * 32 * 2,
        "raw_semantic_packet_bytes": len(raw_packet),
        "semantic_packet_rate_score_if_only_archive": (25.0 * len(raw_packet) / SOURCE_VIDEO_BYTES),
        "complete_archive_measured": False,
        "learned_entropy_predicted": False,
        "candidate_or_score_claim": False,
    }


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise G111Batch16V9SemanticBaseError(f"{label} must be a lowercase SHA-256")
    return value


def _open_bound_bytes(path: Path) -> tuple[bytes, dict[str, Any]]:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise G111Batch16V9SemanticBaseError(
            f"release artifact must not be a symlink: {candidate}"
        )
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise G111Batch16V9SemanticBaseError(
            f"release artifact is not a regular file: {resolved}"
        )
    before = resolved.stat()
    payload = resolved.read_bytes()
    after = resolved.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or len(payload) != before.st_size:
        raise G111Batch16V9SemanticBaseError(
            f"release artifact changed during reopen: {resolved}"
        )
    return payload, {
        "path": str(resolved),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _open_bound_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, binding = _open_bound_bytes(path)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G111Batch16V9SemanticBaseError(
            f"release artifact is not JSON: {binding['path']}"
        ) from exc
    if type(value) is not dict:
        raise G111Batch16V9SemanticBaseError(
            f"release artifact root is not an object: {binding['path']}"
        )
    return value, binding


def _argv_value(argv: object, flag: str) -> str | None:
    if (
        type(argv) is not list
        or any(type(token) is not str for token in argv)
        or argv.count(flag) != 1
    ):
        return None
    index = argv.index(flag)
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        return None
    return argv[index + 1]


def _parse_dry_start_log_evidence(
    payload: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute the boot/resume facts from trainer and safe-run telemetry."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise G111Batch16V9SemanticBaseError(
            "dry-start run.log is not UTF-8"
        ) from exc
    metrics: dict[str, Any] = {
        "epochs_completed": -1,
        "checkpoint_written": False,
        "last_ckpt_epoch": None,
        "resume_model_source": False,
        "resume_start_epoch": None,
        "resume_ckpt_epoch": None,
    }
    safe_rows: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("SAFE_RUN "):
            try:
                safe = json.loads(line.removeprefix("SAFE_RUN ").strip())
            except json.JSONDecodeError:
                continue
            if type(safe) is dict:
                safe_rows.append(safe)
            continue
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if type(row) is not dict:
            continue
        epoch = row.get("ep", row.get("epoch"))
        if isinstance(epoch, (int, float)) and not isinstance(epoch, bool):
            metrics["epochs_completed"] = max(
                int(metrics["epochs_completed"]),
                int(epoch),
            )
        stage = row.get("stage")
        if stage == "checkpoint" and row.get("resume_latest"):
            metrics["checkpoint_written"] = True
            checkpoint_epoch = row.get("epoch")
            if isinstance(checkpoint_epoch, (int, float)) and not isinstance(
                checkpoint_epoch,
                bool,
            ):
                metrics["last_ckpt_epoch"] = int(checkpoint_epoch)
        if stage == "resume_model_source":
            metrics["resume_model_source"] = True
        resume_start = row.get("resume_start_epoch")
        if isinstance(resume_start, (int, float)) and not isinstance(
            resume_start,
            bool,
        ):
            metrics["resume_start_epoch"] = int(resume_start)
        resume_checkpoint = row.get("resume_ckpt_epoch")
        if isinstance(resume_checkpoint, (int, float)) and not isinstance(
            resume_checkpoint,
            bool,
        ):
            metrics["resume_ckpt_epoch"] = int(resume_checkpoint)
    if len(safe_rows) != 1:
        raise G111Batch16V9SemanticBaseError(
            "dry-start run.log must contain exactly one SAFE_RUN terminal row"
        )
    safe = safe_rows[0]
    if (
        not isinstance(safe.get("exit"), int)
        or isinstance(safe.get("exit"), bool)
        or not isinstance(safe.get("peak_rss_mib"), (int, float))
        or isinstance(safe.get("peak_rss_mib"), bool)
        or float(safe["peak_rss_mib"]) <= 0.0
    ):
        raise G111Batch16V9SemanticBaseError(
            "dry-start SAFE_RUN row lacks an integer exit or positive peak RSS"
        )
    return metrics, safe


def _verify_dsl_launch_unit(
    unit_dir: Path,
    *,
    target_path: str,
    target_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reopen and independently verify one root/pass DSL launch unit."""

    launch_path = unit_dir / "launch.sh"
    provenance_path = unit_dir / "dsl_provenance.json"
    manifest_path = unit_dir / "launch_manifest.json"
    _, launch_binding = _open_bound_bytes(launch_path)
    _, provenance_binding = _open_bound_bytes(provenance_path)
    manifest, manifest_binding = _open_bound_json(manifest_path)
    dsl_hash = _require_sha256(
        manifest.get("dsl_compile_hash"),
        "G111 dry-start DSL compile hash",
    )
    from tac.v9_provenance_gates import verify_dsl_provenance_artifacts

    ok, detail = verify_dsl_provenance_artifacts(
        launch_path,
        provenance_path=provenance_path,
        launch_manifest_path=manifest_path,
        expected_hash=dsl_hash,
    )
    if not ok:
        raise G111Batch16V9SemanticBaseError(
            f"dry-start DSL artifact verification failed: {detail}"
        )
    argv = manifest.get("resolved_launch_argv")
    if (
        manifest.get("schema") != "witness_launch_manifest.v1"
        or manifest.get("config_family") != PROGRAM_NAME
        or manifest.get("spec_id") != PROGRAM_NAME
        or _argv_value(argv, "--training-target-capsule") != target_path
        or _argv_value(argv, "--training-target-capsule-sha256") != target_sha
        or _argv_value(argv, "--num-pairs") != str(PRODUCTION_PAIR_COUNT)
        or type(argv) is not list
        or argv.count("--fresh-producer") != 1
    ):
        raise G111Batch16V9SemanticBaseError(
            "dry-start DSL launch unit differs from G111/G109 custody"
        )
    return manifest, {
        "dsl_compile_hash": dsl_hash,
        "launch_sh": launch_binding,
        "dsl_provenance": provenance_binding,
        "launch_manifest": manifest_binding,
    }


def _verify_pass_log_matches_report(
    pass_report: object,
    *,
    pass_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if type(pass_report) is not dict:
        raise G111Batch16V9SemanticBaseError(
            "GREEN dry-start report lacks a pass object"
        )
    reported_dir = Path(str(pass_report.get("dir", ""))).expanduser()
    if (
        pass_dir.is_symlink()
        or not pass_dir.is_dir()
        or reported_dir.resolve() != pass_dir.resolve()
    ):
        raise G111Batch16V9SemanticBaseError(
            "dry-start pass directory differs from its report/run custody"
        )
    log_payload, log_binding = _open_bound_bytes(pass_dir / "run.log")
    metrics, safe = _parse_dry_start_log_evidence(log_payload)
    for key, value in metrics.items():
        if pass_report.get(key) != value:
            raise G111Batch16V9SemanticBaseError(
                f"dry-start report {key} differs from run.log recomputation"
            )
    peak_gib = round(float(safe["peak_rss_mib"]) / 1024.0, 3)
    if (
        pass_report.get("rc") != safe["exit"]
        or pass_report.get("outer_timeout") is not False
        or pass_report.get("peak_rss_gib") != peak_gib
    ):
        raise G111Batch16V9SemanticBaseError(
            "dry-start report rc/timeout/peak differs from SAFE_RUN telemetry"
        )
    return metrics, safe, log_binding


def _open_pass1_physical_checkpoint(
    pass_dir: Path,
    *,
    dsl_compile_hash: str,
    last_ckpt_epoch: int,
) -> dict[str, Any]:
    """Reopen the immutable deploy/resume pair and its recursive ancestry."""

    tip, tip_binding = _open_bound_json(pass_dir / "fresh_lineage_tip.json")
    if (
        tip.get("schema") != "tac.fresh_producer_lineage_tip.v1"
        or tip.get("complete_trajectory_proven") is not True
        or tip.get("epoch") != last_ckpt_epoch
    ):
        raise G111Batch16V9SemanticBaseError(
            "pass-1 fresh-lineage tip does not name its logged checkpoint"
        )
    receipt_path = Path(str(tip.get("receipt_path", "")))
    receipt_payload, receipt_binding = _open_bound_bytes(receipt_path)
    receipt_sha = _require_sha256(
        tip.get("receipt_sha256"),
        "pass-1 physical checkpoint receipt SHA-256",
    )
    checkpoint_id = _require_sha256(
        tip.get("checkpoint_id_sha256"),
        "pass-1 physical checkpoint id",
    )
    if (
        hashlib.sha256(receipt_payload).hexdigest() != receipt_sha
        or len(receipt_payload) != tip.get("receipt_bytes")
    ):
        raise G111Batch16V9SemanticBaseError(
            "pass-1 physical checkpoint receipt identity differs from its tip"
        )
    try:
        from tac.witness_control.fresh_producer_lineage_v1 import (
            open_fresh_physical_checkpoint_chain_v1,
        )

        chain = open_fresh_physical_checkpoint_chain_v1(
            receipt_path,
            expected_receipt_sha256=receipt_sha,
            expected_current_launch_dsl_compile_hash=dsl_compile_hash,
        )
    except Exception as exc:
        raise G111Batch16V9SemanticBaseError(
            f"pass-1 physical checkpoint chain failed recursive reopen: {exc}"
        ) from exc
    if (
        chain.current.pair.epoch != last_ckpt_epoch
        or chain.current.pair.checkpoint_id_sha256 != checkpoint_id
        or chain.current.receipt_path.resolve() != receipt_path.resolve()
        or chain.current.receipt_sha256 != receipt_sha
    ):
        raise G111Batch16V9SemanticBaseError(
            "pass-1 reopened physical checkpoint differs from tip/log"
        )
    return {
        "tip": tip_binding,
        "receipt": receipt_binding,
        "receipt_sha256": receipt_sha,
        "checkpoint_id_sha256": checkpoint_id,
        "epoch": last_ckpt_epoch,
        "root_sha256": chain.root_sha256,
        "sequence_index": chain.current.sequence_index,
    }


def _verify_green_physical_evidence(
    run_dir: Path,
    *,
    report: dict[str, Any],
    target_path: str,
    target_sha: str,
) -> dict[str, Any]:
    """Make GREEN a recomputed physical verdict, never a JSON marker."""

    root_manifest, root_dsl = _verify_dsl_launch_unit(
        run_dir,
        target_path=target_path,
        target_sha=target_sha,
    )
    root_argv = root_manifest["resolved_launch_argv"]
    if _argv_value(root_argv, "--resume-from") is not None:
        raise G111Batch16V9SemanticBaseError(
            "dry-start release root launch is not the cold G111 producer"
        )

    pass1_dir = run_dir / "dry_start"
    pass1_manifest, pass1_dsl = _verify_dsl_launch_unit(
        pass1_dir,
        target_path=target_path,
        target_sha=target_sha,
    )
    pass1_argv = pass1_manifest["resolved_launch_argv"]
    if (
        _argv_value(pass1_argv, "--ckpt-every") != "1"
        or pass1_argv.count("--no-mod-dim-ablation") != 1
        or _argv_value(pass1_argv, "--resume-from") is not None
    ):
        raise G111Batch16V9SemanticBaseError(
            "pass-1 launch lacks the typed crash-checkpoint bench lever"
        )
    pass1_metrics, pass1_safe, pass1_log = _verify_pass_log_matches_report(
        report.get("pass1"),
        pass_dir=pass1_dir,
    )
    if (
        pass1_metrics["epochs_completed"] < 1
        or pass1_metrics["checkpoint_written"] is not True
        or not isinstance(pass1_metrics["last_ckpt_epoch"], int)
    ):
        raise G111Batch16V9SemanticBaseError(
            "pass-1 physical log does not prove boot, step, and checkpoint"
        )
    checkpoint = _open_pass1_physical_checkpoint(
        pass1_dir,
        dsl_compile_hash=pass1_dsl["dsl_compile_hash"],
        last_ckpt_epoch=pass1_metrics["last_ckpt_epoch"],
    )

    pass2_dir = run_dir / "dry_start_resume"
    pass2_manifest, pass2_dsl = _verify_dsl_launch_unit(
        pass2_dir,
        target_path=target_path,
        target_sha=target_sha,
    )
    pass2_argv = pass2_manifest["resolved_launch_argv"]
    if (
        _argv_value(pass2_argv, "--ckpt-every") != "1"
        or pass2_argv.count("--no-mod-dim-ablation") != 1
        or _argv_value(pass2_argv, "--resume-from") != str(pass1_dir)
        or _argv_value(pass2_argv, "--fresh-lineage-parent-receipt")
        != checkpoint["receipt"]["path"]
        or _argv_value(
            pass2_argv,
            "--fresh-lineage-parent-receipt-sha256",
        )
        != checkpoint["receipt_sha256"]
    ):
        raise G111Batch16V9SemanticBaseError(
            "pass-2 launch is not bound to the exact pass-1 physical parent"
        )
    pass2_metrics, pass2_safe, pass2_log = _verify_pass_log_matches_report(
        report.get("pass2"),
        pass_dir=pass2_dir,
    )
    resume_start = pass2_metrics["resume_start_epoch"]
    resume_checkpoint = pass2_metrics["resume_ckpt_epoch"]
    if (
        pass2_metrics["resume_model_source"] is not True
        or not isinstance(resume_start, int)
        or not isinstance(resume_checkpoint, int)
        or resume_checkpoint != pass1_metrics["last_ckpt_epoch"]
        or resume_start != resume_checkpoint + 1
        or pass2_metrics["epochs_completed"] < resume_start
    ):
        raise G111Batch16V9SemanticBaseError(
            "pass-2 physical log does not prove exact-position resume and progress"
        )
    if (
        report.get("peak_rss_gib")
        != round(float(pass1_safe["peak_rss_mib"]) / 1024.0, 3)
    ):
        raise G111Batch16V9SemanticBaseError(
            "top-level GREEN peak differs from pass-1 physical telemetry"
        )
    return {
        "root": root_dsl,
        "pass1": {
            **pass1_dsl,
            "run_log": pass1_log,
            "safe_run_exit": pass1_safe["exit"],
            "checkpoint": checkpoint,
        },
        "pass2": {
            **pass2_dsl,
            "run_log": pass2_log,
            "safe_run_exit": pass2_safe["exit"],
            "resumed_checkpoint_epoch": resume_checkpoint,
            "resume_start_epoch": resume_start,
        },
    }


def _find_green_dry_start_release(
    *,
    typed_config_hash: str,
    target_contract: dict[str, Any],
    search_roots: Sequence[Path] = _DRY_START_SEARCH_ROOTS,
) -> dict[str, Any] | None:
    """Re-derive launch release only from a physical same-config dry-start."""

    expected_typed_hash = _require_sha256(
        typed_config_hash,
        "G111 typed config hash",
    )
    target_path = target_contract["physical_receipt"]["path"]
    target_sha = target_contract["external_receipt_sha256"]
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        for report_path in root.glob("*/dry_start_report.json"):
            if report_path.is_symlink() or not report_path.is_file():
                continue
            launch_path = report_path.parent / "launch_manifest.json"
            if launch_path.is_symlink() or not launch_path.is_file():
                continue
            try:
                report, report_binding = _open_bound_json(report_path)
                launch, launch_binding = _open_bound_json(launch_path)
            except (OSError, G111Batch16V9SemanticBaseError):
                continue
            argv = launch.get("resolved_launch_argv") if type(launch) is dict else None
            if (
                report.get("gate") != "full_config_dry_start"
                or report.get("config") != PROGRAM_NAME
                or report.get("typed_config_hash") != expected_typed_hash
                or report.get("num_pairs") != PRODUCTION_PAIR_COUNT
                or report.get("green") is not True
                or report.get("boot_ok") is not True
                or report.get("resume_round_trip_ok") is not True
                or type(report.get("ts")) is not str
                or not report["ts"]
                or launch.get("schema") != "witness_launch_manifest.v1"
                or launch.get("config_family") != PROGRAM_NAME
                or launch.get("spec_id") != PROGRAM_NAME
                or _argv_value(argv, "--training-target-capsule")
                != target_path
                or _argv_value(argv, "--training-target-capsule-sha256")
                != target_sha
                or _argv_value(argv, "--num-pairs")
                != str(PRODUCTION_PAIR_COUNT)
            ):
                continue
            dsl_hash = launch.get("dsl_compile_hash")
            try:
                _require_sha256(dsl_hash, "G111 dry-start DSL compile hash")
            except G111Batch16V9SemanticBaseError:
                continue
            try:
                physical_evidence = _verify_green_physical_evidence(
                    report_path.parent,
                    report=report,
                    target_path=target_path,
                    target_sha=target_sha,
                )
            except (OSError, G111Batch16V9SemanticBaseError):
                continue
            if physical_evidence["root"]["dsl_compile_hash"] != dsl_hash:
                continue
            receipt = {
                "schema": "tac.g111_green_dry_start_release.v2",
                "typed_config_hash": expected_typed_hash,
                "dsl_compile_hash": dsl_hash,
                "target_capsule_receipt_sha256": target_sha,
                "report": report_binding,
                "launch_manifest": launch_binding,
                "physical_evidence": physical_evidence,
                "boot_ok": True,
                "resume_round_trip_ok": True,
                "peak_rss_gib": report.get("peak_rss_gib"),
                "sec_per_ep_marginal": report.get("sec_per_ep_marginal"),
                "measured_at_utc": report["ts"],
            }
            candidates.append(
                (report["ts"], str(report_path.resolve()), receipt)
            )
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row[0], row[1]))[2]


def _open_production_target(
    path: str | Path,
    *,
    expected_sha256: str,
) -> V9TrainingTargetCapsuleLoaderV1:
    receipt_path = Path(path).expanduser()
    expected = _require_sha256(expected_sha256, "G109 receipt SHA-256")
    loader = V9TrainingTargetCapsuleLoaderV1.open(
        receipt_path,
        expected_sha256=expected,
    )
    if (
        loader.pair_count != PRODUCTION_PAIR_COUNT
        or loader.batch_pairs != PRODUCTION_BATCH_PAIRS
        or loader.preflight.get("test_only_small_fixture") is not False
    ):
        raise G111Batch16V9SemanticBaseError("G111 admits only the real full-n600 upstream-batch16 G109 capsule")
    return loader


def _target_contract(
    loader: V9TrainingTargetCapsuleLoaderV1,
    *,
    external_receipt_sha256: str,
) -> dict[str, Any]:
    receipt = loader.receipt
    return {
        "schema": TARGET_CONTRACT_SCHEMA,
        "physical_receipt": {
            "path": str(loader.receipt_path),
            "bytes": int(loader.receipt_path.stat().st_size),
            "sha256": sha256_file(loader.receipt_path),
        },
        "external_receipt_sha256": external_receipt_sha256,
        "aggregate_receipt_sha256": receipt["aggregate_receipt_sha256"],
        "preflight_sha256": receipt["preflight_sha256"],
        "batch_digest_chain_sha256": receipt["batch_digest_chain_sha256"],
        "pair_count": loader.pair_count,
        "scorer_pair_batch_size": loader.batch_pairs,
        "same_forward_seg_margin_pose": True,
        "source_cache_reverified_by_trainer": True,
        "cold_own_lineage_producer": True,
        "fresh_spectral_initializer_required": False,
        "pose_carrier_source": "generated_y1",
        "conditional_y0_source": "final_odd_code_y1_render",
        "conditional_y0_source_boundary": "scorer_grid_uint8",
        "conditional_y0_camera_realization": V10_FACTOR2_SELECTED_PREIMAGE_SCHEMA,
        "pose_gradient_public_camera_realization_identical": True,
        "semantic_training_loss_public_wire_identical": False,
        "semantic_stage_selection_public_wire_identical": False,
        "serialized_even_code_rows_required": False,
        "render_aa": "none",
        "boot_baseline_verdict_observability_only": True,
        "boot_baseline_verdict_skipped": True,
        "first_own_lineage_measurement": "first_immutable_in_loop_stage",
        "post_semantic_compile_xi_refit_required": True,
        "tail_stop_marginal_s": G111_TAIL_STOP_MARGINAL_S,
        "tail_stop_policy": "pareto_nonnegative_score_benefit_v1",
        "tail_stop_current_g111_law_fit": "owed",
        "tail_stop_ancestor_forfeit_law_authoritative": False,
        "y1_rate_arbitration": Y1_RATE_ARBITRATION_SCHEMA,
        "y1_rate_domain": "exact_complete_archive_zip_bytes",
        "y1_wire_families": ["raw_i16le", "delta_rice_best_k"],
        "outer_zip_methods": ["stored", "deflated"],
        "fresh_lineage_root_seed_persisted": True,
        "fresh_lineage_root_recomputed_by_consumer": True,
        "physical_cold_full_state_checkpoint_before_first_step": True,
        "full_state_companion_required_for_own_lineage_claim": True,
        "recursive_physical_checkpoint_chain_required": True,
        "fresh_lineage_tip_schema": "tac.fresh_producer_lineage_tip.v1",
        "resume_requires_external_parent_receipt_path_and_sha256": True,
        "semantic_verdict_surface": "parsed_G105_public_wire_v1",
        "semantic_checkpoint_selection_surface": "parsed_G105_public_wire_v1",
        "legacy_arbitrary_scale_int8_selection_allowed": False,
        "parsed_g105_wire_verdict_implemented": True,
        "external_exhaustive_stage_compiler": (
            "tac.g121_retained_prepose.v2"
        ),
        "post_g105_conditional_pose_compiler": (
            "tac.post_g105_generated_y1_pose_refit_run.v1"
        ),
        "selected_xip2_coder_archive_abi_closed": True,
        "frontier_launch_blocker": None,
        "structural_semantic_rate_preflight": structural_semantic_rate_preflight(),
        "self_orient": False,
        "mod_dim": 32,
        "encoder_only": True,
        "candidate_payload_allowed": False,
        "score_claim": False,
        "pointer_moved": False,
    }


def compile_g111_batch16_v9_semantic_base_launch_config(
    *,
    training_target_capsule: str | Path,
    training_target_capsule_sha256: str,
    gt_cache_path: str = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    num_pairs: int = PRODUCTION_PAIR_COUNT,
    epochs: int = 3000,
    out_dir: str = ("/Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base"),
):
    """Compile the first real cold V9 producer on the exact G109 scorer fiber."""

    if int(num_pairs) != PRODUCTION_PAIR_COUNT:
        raise G111Batch16V9SemanticBaseError(f"G111 is a full-n600 producer, got num_pairs={num_pairs}")
    expected_receipt_sha = _require_sha256(
        training_target_capsule_sha256,
        "G109 receipt SHA-256",
    )
    loader = _open_production_target(
        training_target_capsule,
        expected_sha256=expected_receipt_sha,
    )

    # The settled mod-32 branch is the distortion-feasible base.  This compile is
    # explicitly cold own-lineage, while FreSh spectral initialization remains an
    # optional algorithm and is not falsely claimed: repository FreSh currently
    # requires self-orient, whereas the public G105 decoder has no fixed-point
    # self-orient ABI.
    wrapped = compile_v9_cgauge_ideal_launch_config(
        gt_cache_path=gt_cache_path,
        num_pairs=PRODUCTION_PAIR_COUNT,
        epochs=int(epochs),
        out_dir=out_dir,
        mod_dim=32,
        program_name=PROGRAM_NAME,
        flag_custody=False,
    )
    typed = wrapped.typed
    # The inherited TAIL owner carries the historical global 1e-4 stop floor.
    # That value and the ancestor forfeit-matched LawRef have no current-G111
    # dynamics authority. Re-home this ONE flag in the G111 child target Lever
    # so last-wins composition also has exactly one semantic owner. The G111
    # floor is the Pareto-dominance boundary: stop only when net score benefit
    # is negative; nonnegative cycles remain eligible for k_max-bounded mining
    # and whole-object G121 stage harvesting.
    inherited_tail_owners = [
        lever for lever in typed.levers if TAIL_STOP_FLAG in lever.overrides
    ]
    if (
        len(inherited_tail_owners) != 1
        or inherited_tail_owners[0].name != "tail_k_warm_restart"
    ):
        raise G111Batch16V9SemanticBaseError(
            "G111 expected exactly one inherited tail_k_warm_restart owner for "
            f"{TAIL_STOP_FLAG}, got {[lever.name for lever in inherited_tail_owners]}"
        )
    parent_levers: list[TypedLever] = []
    for lever in typed.levers:
        if TAIL_STOP_FLAG not in lever.overrides:
            parent_levers.append(lever)
            continue
        updates: dict[str, object] = {}
        for field_name in (
            "overrides",
            "lawrefs",
            "lawref_declarations",
            "constant_manifest",
            "runtime_receipt_schemas",
        ):
            values = dict(getattr(lever, field_name))
            values.pop(TAIL_STOP_FLAG, None)
            updates[field_name] = values
        parent_levers.append(lever.model_copy(update=updates))
    target_lever = TypedLever(
        name=TARGET_LEVER_NAME,
        overrides={
            "--training-target-capsule": str(loader.receipt_path),
            "--training-target-capsule-sha256": expected_receipt_sha,
            "--fresh-producer": True,
            "--verdict-batch": PRODUCTION_BATCH_PAIRS,
            "--self-orient": False,
            "--render-aa": "none",
            "--pose-carrier-source": "generated_y1",
            # The pre-loop full-n600 scorer is observation-only and costs about
            # one complete verdict wall before the cold-root checkpoint.  G111
            # obtains its first own-lineage measurement from the first immutable
            # in-loop stage instead, so omitting the boot row changes no update,
            # controller decision, checkpoint tensor, or receiver output.
            SKIP_BOOT_BASELINE_FLAG: True,
            TAIL_STOP_FLAG: G111_TAIL_STOP_MARGINAL_S,
        },
        notes=(
            "Reopen physical G109 at compile and train time; bind labels, margins, "
            "and Pose6 from one upstream-batch16 forward; cold own-lineage producer; "
            "derive Y0 conditionally by warping the final odd-code Y1 render; "
            "skip the observation-only pre-loop full-population verdict so the "
            "cold root and first trainable stage are reached one scorer wall sooner; "
            "own the G111-only Pareto tail floor 0.0 so no nonnegative score-benefit "
            "cycle is truncated before k_max/G121 whole-object harvesting."
        ),
    )
    typed = typed.model_copy(
        update={
            "name": PROGRAM_NAME,
            "purpose": (
                "First real full-n600 scorer-native semantic-base producer: settled "
                "V9 mod-32 geometry on a recursively verified G109 batch-16 target "
                "capsule, cold own-lineage checkpoint, G105-compatible semantic "
                "gauge plus a final-Y1-conditioned pose carrier whose xi is "
                "refit after semantic packet quantization."
            ),
            "base": {
                **dict(typed.base),
                "--out-dir": str(out_dir),
            },
            "levers": (*tuple(parent_levers), target_lever),
        }
    )
    violations = typed.validate_program()
    if violations:
        raise G111Batch16V9SemanticBaseError(f"G111 typed DSL validation failed: {violations[:4]}")

    program = typed.to_program()
    argv = tuple(program.compile_trainer_argv())
    flags = program.flag_dict()
    required = {
        "--training-target-capsule": str(loader.receipt_path),
        "--training-target-capsule-sha256": expected_receipt_sha,
        "--fresh-producer": True,
        "--verdict-batch": PRODUCTION_BATCH_PAIRS,
        "--self-orient": False,
        "--render-aa": "none",
        "--pose-carrier-source": "generated_y1",
        SKIP_BOOT_BASELINE_FLAG: True,
        "--mod-dim": 32,
        TAIL_STOP_FLAG: G111_TAIL_STOP_MARGINAL_S,
    }
    mismatches = {flag: (flags.get(flag), value) for flag, value in required.items() if flags.get(flag) != value}
    forbidden = {
        flag: flags.get(flag)
        for flag in ("--resume-from", "--warm-start-weights-only", "--fresh-init")
        if flags.get(flag) not in (None, False)
    }
    if mismatches or forbidden:
        raise G111Batch16V9SemanticBaseError(
            f"G111 cold-producer argv differs: mismatches={mismatches}, forbidden={forbidden}"
        )

    expected_levers = (
        *tuple(wrapped.dsl_program_manifest["expected_active_levers"]),
        TARGET_LEVER_NAME,
    )
    observed_pre_custody_levers = tuple(lever.name for lever in program.levers)
    if sorted(observed_pre_custody_levers) != sorted(expected_levers):
        raise G111Batch16V9SemanticBaseError(
            "G111 pre-custody lever set differs from its V9 parent plus target binding"
        )
    from tac.witness_autoconfig import _crucible_v7_argv_pairs

    emitted_flag_names = sorted(dict(_crucible_v7_argv_pairs(argv)))
    manifest = dict(wrapped.dsl_program_manifest)
    manifest.update(
        build_launch_manifest(
            program_name=PROGRAM_NAME,
            emitted_flag_names=emitted_flag_names,
            typed_config_hash=typed.typed_config_hash(),
            typed_validated=True,
        )
    )
    manifest.update(
        {
            "expected_active_levers": list(expected_levers),
            "training_target_contract": _target_contract(
                loader,
                external_receipt_sha256=expected_receipt_sha,
            ),
            "held": True,
            "operator_go_required": False,
            "fire_after": (
                "parsed G105 wire-quantized semantic verdict and checkpoint "
                "selection are wired, then governed storage/memory/receiver "
                "readiness gates pass"
            ),
            "hold_reason": "current typed config still owes a green governed dry-start",
            "candidate_claim": False,
            "score_claim": False,
            "pointer_moved": False,
        }
    )
    # The V9 parent has already performed its one authoritative argv/constant
    # reconciliation (notably preserving the historical beta_end=10 input while
    # emitting 3.177).  Re-running that helper would overwrite the historical
    # value with 3.177 and make the equation self-recompile falsely compare
    # 10.0->3.177.  Preserve parent rows verbatim; only retire a row when this
    # child actually changes its flag, so attach_flag_custody honestly rebuilds
    # that changed flag from the child's emitted scalar. The G111 tail floor is
    # always retired even if an ancestor row happens to share the numeric value:
    # current G111 explicitly rejects ancestor derivational authority.
    constants = {
        key: (dict(value) if isinstance(value, dict) else value) for key, value in wrapped.constants_manifest.items()
    }
    for flag, child_value in target_lever.overrides.items():
        key = flag.removeprefix("--").replace("-", "_")
        row = constants.get(key)
        if flag == TAIL_STOP_FLAG or (
            isinstance(row, dict) and row.get("value") != child_value
        ):
            constants.pop(key, None)
    constants = _merge_lever_constant_manifests(constants, tuple(program.levers))
    typed, constants = attach_flag_custody(
        typed,
        constants,
        program_name=PROGRAM_NAME,
    )
    tail_key = TAIL_STOP_FLAG.removeprefix("--").replace("-", "_")
    tail_row = constants.get(tail_key)
    final_tail_owners = [
        lever
        for lever in typed.levers
        if TAIL_STOP_FLAG in lever.overrides
    ]
    if (
        len(final_tail_owners) != 1
        or final_tail_owners[0].name != TARGET_LEVER_NAME
        or final_tail_owners[0].overrides[TAIL_STOP_FLAG]
        != G111_TAIL_STOP_MARGINAL_S
        or final_tail_owners[0].lawref_declarations.get(
            TAIL_STOP_FLAG, {}
        ).get("equation_id")
        != DSL_IDENTITY_EQUATION_ID
        or not isinstance(tail_row, dict)
        or tail_row.get("value") != G111_TAIL_STOP_MARGINAL_S
        or tail_row.get("equation_id") != DSL_IDENTITY_EQUATION_ID
        or tail_row.get("ladder_class") != "hardcoded_waiver"
    ):
        raise G111Batch16V9SemanticBaseError(
            "G111 tail-stop custody must be one target-Lever-owned 0.0 scalar "
            "under non-derivational hardcoded-waiver identity custody"
        )
    tail_row["single_value_owner"] = f"dsl_lever:{TARGET_LEVER_NAME}"
    tail_row["emitted_flag"] = TAIL_STOP_FLAG
    manifest["typed_config_hash"] = typed.typed_config_hash()
    # The custody rollup is itself a real composed Lever.  The launcher compares
    # this manifest against the post-custody program, so its name belongs in the
    # final expected set after the pre-custody parent+delta equality above passed.
    manifest["expected_active_levers"] = [lever.name for lever in typed.to_program().levers]
    release = _find_green_dry_start_release(
        typed_config_hash=manifest["typed_config_hash"],
        target_contract=manifest["training_target_contract"],
    )
    blockers: list[dict[str, str]] = []
    if release is None:
        blockers.append(
            {
                "id": "G111_CURRENT_TYPED_DRY_START_NOT_GREEN",
                "detail": (
                    "no physical GREEN full_config_dry_start receipt matches "
                    "this exact G111 typed_config_hash and G109 target; run the "
                    "governed launcher with --dry-start before real spawn"
                ),
            }
        )
    else:
        manifest["green_dry_start_release"] = release
    manifest["launch_blockers"] = blockers
    manifest["held"] = bool(blockers)
    manifest["hold_reason"] = (
        blockers[0]["detail"] if blockers else None
    )

    from tac.witness_autoconfig import CrucibleV7LaunchConfig

    return CrucibleV7LaunchConfig(
        typed=typed,
        constants_manifest=constants,
        dsl_program_manifest=manifest,
        schedule_governance=dict(wrapped.schedule_governance),
    )


__all__ = [
    "G111_TAIL_STOP_MARGINAL_S",
    "PROGRAM_NAME",
    "SKIP_BOOT_BASELINE_FLAG",
    "TAIL_STOP_FLAG",
    "TARGET_CONTRACT_SCHEMA",
    "TARGET_LEVER_NAME",
    "G111Batch16V9SemanticBaseError",
    "compile_g111_batch16_v9_semantic_base_launch_config",
]
