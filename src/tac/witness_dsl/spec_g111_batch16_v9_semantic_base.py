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


def _open_bound_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
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
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G111Batch16V9SemanticBaseError(
            f"release artifact is not JSON: {resolved}"
        ) from exc
    if type(value) is not dict:
        raise G111Batch16V9SemanticBaseError(
            f"release artifact root is not an object: {resolved}"
        )
    return value, {
        "path": str(resolved),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


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
            receipt = {
                "schema": "tac.g111_green_dry_start_release.v1",
                "typed_config_hash": expected_typed_hash,
                "dsl_compile_hash": dsl_hash,
                "target_capsule_receipt_sha256": target_sha,
                "report": report_binding,
                "launch_manifest": launch_binding,
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
        "post_semantic_compile_xi_refit_required": True,
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
        },
        notes=(
            "Reopen physical G109 at compile and train time; bind labels, margins, "
            "and Pose6 from one upstream-batch16 forward; cold own-lineage producer; "
            "derive Y0 conditionally by warping the final odd-code Y1 render."
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
            "levers": (*tuple(typed.levers), target_lever),
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
        "--mod-dim": 32,
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
    # that changed flag from the child's emitted scalar.
    constants = {
        key: (dict(value) if isinstance(value, dict) else value) for key, value in wrapped.constants_manifest.items()
    }
    for flag, child_value in target_lever.overrides.items():
        key = flag.removeprefix("--").replace("-", "_")
        row = constants.get(key)
        if isinstance(row, dict) and row.get("value") != child_value:
            constants.pop(key)
    constants = _merge_lever_constant_manifests(constants, tuple(program.levers))
    typed, constants = attach_flag_custody(
        typed,
        constants,
        program_name=PROGRAM_NAME,
    )
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
    "PROGRAM_NAME",
    "TARGET_CONTRACT_SCHEMA",
    "TARGET_LEVER_NAME",
    "G111Batch16V9SemanticBaseError",
    "compile_g111_batch16_v9_semantic_base_launch_config",
]
