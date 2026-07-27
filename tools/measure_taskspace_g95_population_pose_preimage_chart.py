#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fit and exactly replay the bounded G95 population Pose preimage chart.

This is an encode-side research runner.  It reconstructs the exact committed
G94 fixture before scorer load, builds real frozen-PoseNet VJP costates, learns
one shared quantized basis, fits quantized per-pair coefficients by damped
natural-gradient/Levenberg-Marquardt steps, and admits improvements only after
strict packet parse-back plus deterministic NumPy receiver replay.

The default governed run is pair 0 only because the bound G94 state does not
contain final learned semantic Y1.  Every output remains research-only and
cannot move the competitive pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "src", REPO / "upstream"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.canonical_frontier_pointer import (  # noqa: E402
    POINTER_SCHEMA_VERSION,
    effective_frontier_score,
    load_canonical_frontier_pointer_strict,
    recompute_effective_frontier,
)
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    BoundaryShearletAtomV1,
    LanePeriodicProgramV1,
    MovableWorldsheetTrackV1,
)
from tac.witness_dsl.taskspace_g88_population_conditional_y0_pvsa_v1 import (  # noqa: E402
    ConditionalY0ControlV1,
    PopulationConditionalOperandV1,
)
from tac.witness_dsl.taskspace_g89_class_complete_semantic_compiler_v1 import (  # noqa: E402
    ClassCompleteSemanticProgramV1,
    SharedTopologyApplicationV1,
    SharedTopologyTemplateV1,
)
from tac.witness_dsl.taskspace_g94_sequential_typed_actuator_product_v1 import (  # noqa: E402
    SequentialTypedArchiveBuildV1,
    build_sequential_typed_archive,
)
from tac.witness_dsl.taskspace_g95_population_pose_preimage_chart_v1 import (  # noqa: E402
    BILINEAR_REFERENCE_ID,
    BOUND_G94_CONDITIONING_STATE_SHA256,
    BOUND_G94_PARENT_COMMIT,
    BOUND_G94_PRODUCT_MEMBER_SHA256,
    FORMULATION_SCOPE,
    MISS_VERDICT_SCOPE,
    MISSING_INTEGRATION,
    ONE_STATE_MISS_AXIS,
    OUTER_ZIP_SCORE_ADMISSION,
    POPULATION_TRANSFER_REQUEST,
    REACHABILITY_COORDINATE_SCOPE,
    REACHABILITY_THRESHOLD,
    ROUNDING_POLICY_ID,
    G95ControlModeV1,
    PopulationPosePreimageChartBatchResultV1,
    PopulationPosePreimageChartReceiverV1,
    PopulationPosePreimageChartWireSetV1,
    encode_population_pose_preimage_basis,
    encode_population_pose_preimage_coefficient_chunk,
    parse_population_pose_preimage_basis,
    parse_population_pose_preimage_coefficient_chunk,
    richer_control_request_for_miss,
)
from tac.witness_dsl.taskspace_outer_archive_codec import (  # noqa: E402
    parse_taskspace_outer_archive,
)

SCHEMA = "tac.g95_population_pose_preimage_chart_runner_config.v1"
PREFLIGHT_SCHEMA = "tac.g95_population_pose_preimage_chart_preflight.v1"
MEASUREMENT_SCHEMA = "tac.g95_population_pose_preimage_chart_measurement.v1"
DEFAULT_CONFIG = REPO / ".omx/research/configs/taskspace_g95_population_pose_preimage_chart_20260727.json"
DEFAULT_FRONTIER_POINTER = REPO / ".omx/state/canonical_frontier_pointer.json"
EXPECTED_FINAL_RECEIPT = (
    REPO
    / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
    / "g95_population_pose_preimage_chart_receipt_20260727.json"
)
PRIMARY_SSD_ROOT = Path("/Volumes/VertigoDataTier/pact")
EXPECTED_CONDITIONING_SHA256 = BOUND_G94_CONDITIONING_STATE_SHA256
EXPECTED_G94_PARENT_COMMIT = BOUND_G94_PARENT_COMMIT
EXPECTED_G94_PRODUCT_SHA256 = BOUND_G94_PRODUCT_MEMBER_SHA256
PAIR_SHAPE = (2, 874, 1164, 3)
MEASUREMENT_TOOL_PATH = Path(__file__).resolve()
RECEIVER_MODULE_PATH = (REPO / "src/tac/witness_dsl/taskspace_g95_population_pose_preimage_chart_v1.py").resolve()
RANK_STAGE_NAMES = {
    6: "stage_03_rank06_48x64",
    12: "stage_04_rank12_48x64",
    24: "stage_05_rank24_48x64",
}
PUBLIC_INFLATE_BLOCKER = "G95_PUBLIC_INFLATE_SH_RECURSIVE_RUNTIME_CLOSURE_OWED"
FULL_N600_BLOCKER = "G95_FULL_N600_FINAL_SEMANTIC_G94_SAME_ARCHIVE_REPLAY_OWED"
UPSTREAM_EVAL_BLOCKER = "G95_UPSTREAM_EVALUATE_PY_EXACT_ARCHIVE_AUTHORITY_OWED"
G83_BLOCKER = "G95_G83_COMPLETE_COMPONENT_SAME_ARCHIVE_ADMISSION_OWED"
OUTER_ZIP_BLOCKER = "G95_G88_G94_OUTER_ZIP_STORE_DEFLATE_RACE_OWED"


class G95RunnerError(RuntimeError):
    """A G95 launch, scorer, fit, checkpoint, or exact replay contract failed."""


def _sha256_bytes(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return _sha256_bytes(memoryview(np.ascontiguousarray(value)).cast("B"))


def _sha256_file(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256_text(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise G95RunnerError(f"{label} must be canonical lowercase SHA-256")
    return value


def _live_competitive_target_snapshot(
    path: Path = DEFAULT_FRONTIER_POINTER,
) -> dict[str, Any]:
    """Bind the current competitive target without making it experiment state."""

    resolved = path.resolve(strict=True)
    before = resolved.read_bytes()
    pointer = load_canonical_frontier_pointer_strict(
        repo_root=REPO,
        path=resolved,
    )
    after = resolved.read_bytes()
    if before != after:
        raise G95RunnerError("canonical frontier pointer changed during G95 custody read")
    if pointer.schema_version != POINTER_SCHEMA_VERSION:
        raise G95RunnerError("canonical frontier pointer schema differs")
    if pointer.is_stale():
        raise G95RunnerError("canonical frontier pointer is stale; refresh before G95 reporting")
    recomputed = recompute_effective_frontier(pointer)
    if not isinstance(recomputed, Mapping):
        raise G95RunnerError("canonical frontier pointer has no competitive target")
    if pointer.effective_frontier != recomputed:
        raise G95RunnerError("serialized frontier target differs from constituent minimum")
    score = effective_frontier_score(pointer)
    if score is None or score != float(recomputed["score"]):
        raise G95RunnerError("canonical frontier target score failed strict recomposition")
    return {
        "path": str(resolved),
        "bytes": len(before),
        "sha256": _sha256_bytes(before),
        "schema_version": pointer.schema_version,
        "last_refreshed_utc": pointer.last_refreshed_utc,
        "effective_frontier": dict(recomputed),
        "score_to_beat": score,
        "selection_rule": recomputed["selection_rule"],
        "role": "REPORTING_CUSTODY_ONLY_NOT_G95_FIT_OR_DECODE_STATE",
    }


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _resume_state_key(
    *,
    source_pair_ids: tuple[int, ...],
    preconditional_camera_sha256: str,
    whole_preconditional_camera_sha256: str,
    selected_target_sha256: str,
    selected_target_table_sha256: str,
    posenet_weights_sha256: str,
    g94_product_member_sha256: str,
    g94_conditioning_state_sha256: str,
    config_sha256: str,
    receiver_module_sha256: str,
    measurement_tool_sha256: str,
) -> str:
    if (
        type(source_pair_ids) is not tuple
        or not source_pair_ids
        or any(type(value) is not int or not 0 <= value < 600 for value in source_pair_ids)
    ):
        raise G95RunnerError("resume-state source_pair_ids must be exact n600 integers")
    payload = {
        "schema": "tac.g95_resume_state_key.v1",
        "source_pair_ids": list(source_pair_ids),
        "preconditional_camera_sha256": _require_sha256_text(
            preconditional_camera_sha256,
            label="preconditional_camera_sha256",
        ),
        "whole_preconditional_camera_sha256": _require_sha256_text(
            whole_preconditional_camera_sha256,
            label="whole_preconditional_camera_sha256",
        ),
        "selected_target_sha256": _require_sha256_text(
            selected_target_sha256,
            label="selected_target_sha256",
        ),
        "selected_target_table_sha256": _require_sha256_text(
            selected_target_table_sha256,
            label="selected_target_table_sha256",
        ),
        "posenet_weights_sha256": _require_sha256_text(
            posenet_weights_sha256,
            label="posenet_weights_sha256",
        ),
        "g94_product_member_sha256": _require_sha256_text(
            g94_product_member_sha256,
            label="g94_product_member_sha256",
        ),
        "g94_conditioning_state_sha256": _require_sha256_text(
            g94_conditioning_state_sha256,
            label="g94_conditioning_state_sha256",
        ),
        "config_sha256": _require_sha256_text(
            config_sha256,
            label="config_sha256",
        ),
        "receiver_module_sha256": _require_sha256_text(
            receiver_module_sha256,
            label="receiver_module_sha256",
        ),
        "measurement_tool_sha256": _require_sha256_text(
            measurement_tool_sha256,
            label="measurement_tool_sha256",
        ),
    }
    return _sha256_bytes(_canonical_json(payload))


def _resume_state_array(resume_state_key: str) -> np.ndarray:
    return np.asarray(
        _require_sha256_text(resume_state_key, label="resume_state_key").encode("ascii"),
        dtype="S64",
    )


def _verify_resume_state_array(
    value: np.ndarray,
    *,
    expected_resume_state_key: str,
    label: str,
) -> None:
    raw = np.asarray(value)
    expected = _require_sha256_text(
        expected_resume_state_key,
        label="expected_resume_state_key",
    )
    if raw.shape != () or raw.dtype != np.dtype("S64") or raw.item().decode("ascii") != expected:
        raise G95RunnerError(f"{label} resume-state key differs")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_once_or_equal(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise G95RunnerError(f"immutable artifact differs on resume: {path}")
        return
    _atomic_write(path, payload)


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    _write_once_or_equal(path, _canonical_json(payload))


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _write_npz_once(path: Path, **arrays: np.ndarray) -> None:
    _write_once_or_equal(path, _npz_bytes(**arrays))


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise G95RunnerError(f"{label} is unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise G95RunnerError(f"{label} must contain one JSON object")
    return value


def _exact_path_identity(spec: Mapping[str, Any], *, label: str) -> tuple[Path, dict[str, Any]]:
    try:
        path = Path(str(spec["path"])).resolve(strict=True)
        expected_bytes = int(spec["bytes"])
        expected_sha = str(spec["sha256"])
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise G95RunnerError(f"{label} custody mapping is malformed") from exc
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise G95RunnerError(f"{label} byte count differs from sealed custody")
    observed_sha = _sha256_file(path)
    if observed_sha != expected_sha:
        raise G95RunnerError(f"{label} SHA-256 differs from sealed custody")
    return path, {
        "path": str(path),
        "bytes": expected_bytes,
        "sha256": observed_sha,
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _refuse_tmp_class_path(path: Path, *, label: str) -> None:
    resolved = path.resolve(strict=False)
    forbidden = (
        Path("/tmp").resolve(strict=False),
        Path("/private/tmp").resolve(strict=False),
        Path("/private/var/folders").resolve(strict=False),
        Path("/var/folders").resolve(strict=False),
    )
    if any(_is_relative_to(resolved, root) for root in forbidden):
        raise G95RunnerError(f"{label} may not use a /tmp-class path")


def _exact_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise G95RunnerError(f"{label} must be an exact nonnegative integer")
    return value


def _finite_positive(value: object, *, label: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise G95RunnerError(f"{label} must be one finite positive scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise G95RunnerError(f"{label} must be one finite positive scalar")
    return result


def _load_config(path: Path) -> dict[str, Any]:
    config = _read_mapping(path, label="G95 config")
    if config.get("schema") != SCHEMA:
        raise G95RunnerError(f"config schema must be {SCHEMA}")
    for label, expected in (
        ("research_only", True),
        ("candidate_claim", False),
        ("score_claim", False),
        ("promotion_eligible", False),
        ("pointer_moved", False),
        ("non_final_semantic_y1", True),
    ):
        if config.get(label) is not expected:
            raise G95RunnerError(f"config {label} truth boundary differs")
    _exact_nonnegative_int(config.get("seed"), label="seed")
    threads = _exact_nonnegative_int(config.get("torch_num_threads"), label="torch_num_threads")
    if threads < 1:
        raise G95RunnerError("torch_num_threads must be positive")
    pair_start = _exact_nonnegative_int(config.get("pair_start"), label="pair_start")
    pair_count = _exact_nonnegative_int(config.get("pair_count"), label="pair_count")
    scorer_batch = _exact_nonnegative_int(
        config.get("scorer_batch_pairs"),
        label="scorer_batch_pairs",
    )
    if not 1 <= pair_count <= 16 or pair_start + pair_count > 600:
        raise G95RunnerError("bounded pair range must contain 1..16 contiguous exact n600 pairs")
    if not 1 <= scorer_batch <= 16:
        raise G95RunnerError("scorer_batch_pairs must be in [1,16]")
    _exact_nonnegative_int(
        config.get("safety_reserve_bytes"),
        label="safety_reserve_bytes",
    )
    if config.get("hardware_axis") != "[macOS-CPU advisory / CPU-torch research-signal] NON-PROMOTABLE":
        raise G95RunnerError("G95 refuses MPS, CUDA, and contest score authority")
    threshold = _finite_positive(
        config.get("reachability_threshold_d_pose"),
        label="reachability_threshold_d_pose",
    )
    if threshold != REACHABILITY_THRESHOLD:
        raise G95RunnerError("G95 reachability threshold differs from frozen exact 0.00047366")
    if config.get("grid_height") != 48 or config.get("grid_width") != 64:
        raise G95RunnerError("G95 V1 ladder grid must remain exact 48x64")
    if config.get("rank_ladder") != [6, 12, 24]:
        raise G95RunnerError("G95 V1 rank ladder must remain exact [6,12,24]")
    for section_name, positive_fields, integer_fields in (
        (
            "basis_fit",
            (
                "dense_teacher_initial_damping",
                "dense_teacher_damping_increase",
                "dense_teacher_damping_decrease",
                "svd_relative_threshold",
                "basis_scale_floor",
            ),
            (
                "dense_teacher_anchor_limit",
                "dense_teacher_line_search_steps",
            ),
        ),
        (
            "coefficient_fit",
            (
                "initial_damping",
                "damping_increase",
                "damping_decrease",
                "coefficient_scale_floor",
                "minimum_exact_improvement",
            ),
            (
                "maximum_iterations",
                "line_search_steps",
                "checkpoint_every_iterations",
            ),
        ),
    ):
        section = config.get(section_name)
        if not isinstance(section, dict):
            raise G95RunnerError(f"{section_name} must be one typed mapping")
        for field_name in positive_fields:
            _finite_positive(section.get(field_name), label=f"{section_name}.{field_name}")
        for field_name in integer_fields:
            if (
                _exact_nonnegative_int(
                    section.get(field_name),
                    label=f"{section_name}.{field_name}",
                )
                < 1
            ):
                raise G95RunnerError(f"{section_name}.{field_name} must be positive")
    richer = config.get("richer_control_request")
    if not isinstance(richer, dict):
        raise G95RunnerError("richer_control_request must be one typed mapping")
    for field_name in (
        "requested_minimum_rank",
        "requested_minimum_grid_height",
        "requested_minimum_grid_width",
    ):
        if _exact_nonnegative_int(richer.get(field_name), label=field_name) < 1:
            raise G95RunnerError(f"{field_name} must be positive")
    if (
        int(richer["requested_minimum_rank"]) <= 24
        and int(richer["requested_minimum_grid_height"]) <= 48
        and int(richer["requested_minimum_grid_width"]) <= 64
    ):
        raise G95RunnerError("richer-control request must increase rank or grid")
    if (
        richer.get("one_state_miss_axis") != ONE_STATE_MISS_AXIS
        or richer.get("population_transfer_failure_classification") != POPULATION_TRANSFER_REQUEST
        or richer.get("population_transfer_request") != POPULATION_TRANSFER_REQUEST
        or richer.get("pair0_population_viability_claim") is not False
    ):
        raise G95RunnerError("richer-control population-transfer request truth boundary differs")
    g94 = config.get("g94")
    if not isinstance(g94, dict):
        raise G95RunnerError("g94 must be one exact custody mapping")
    if (
        g94.get("parent_git_commit") != EXPECTED_G94_PARENT_COMMIT
        or g94.get("product_member_sha256") != EXPECTED_G94_PRODUCT_SHA256
        or g94.get("conditioning_state_sha256") != EXPECTED_CONDITIONING_SHA256
    ):
        raise G95RunnerError("G94 parent/product/conditioning identity drifted")
    output_root = Path(str(config.get("output_root", "")))
    final_receipt = Path(str(config.get("final_receipt_path", "")))
    if not output_root.is_absolute() or not final_receipt.is_absolute():
        raise G95RunnerError("output_root and final_receipt_path must be absolute")
    if final_receipt.resolve(strict=False) != EXPECTED_FINAL_RECEIPT.resolve(strict=False):
        raise G95RunnerError("final_receipt_path differs from the frozen G95 receipt surface")
    _refuse_tmp_class_path(output_root, label="output_root")
    _refuse_tmp_class_path(final_receipt, label="final_receipt_path")
    return config


def _configure_determinism(config: Mapping[str, Any]) -> None:
    import torch

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config["torch_num_threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        if "cannot set number of interop threads" not in str(exc):
            raise
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False


def _git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ("git", *args),
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise G95RunnerError(f"git custody command failed: git {' '.join(args)}") from exc


def _source_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _validate_stage00_reuse_state(
    preserved: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    field_name = "resume_state_sources_hashed_before_scorer_load"
    if preserved.get(field_name) != current.get(field_name):
        raise G95RunnerError("preserved stage_00 receiver/tool/config bytes or SHA differ from current launch")


def _storage_and_custody_preflight(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    primary = PRIMARY_SSD_ROOT.resolve(strict=True)
    resolved_root = run_root.resolve(strict=False)
    if not _is_relative_to(resolved_root, primary):
        raise G95RunnerError("G95 output_root must be under the first operator SSD tier")
    _refuse_tmp_class_path(resolved_root, label="output_root")
    run_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(run_root)
    reserve = int(config["safety_reserve_bytes"])
    if usage.free <= reserve:
        raise G95RunnerError("SSD storage preflight failed closed below configured reserve")
    identities: dict[str, dict[str, Any]] = {}
    for key in (
        "source_video",
        "base_archive",
        "base_member",
        "gt_cache",
        "posenet_weights",
    ):
        spec = config.get(key)
        if not isinstance(spec, dict):
            raise G95RunnerError(f"{key} custody mapping is missing")
        _path, identities[key] = _exact_path_identity(spec, label=key)
    parent_commit = str(config["g94"]["parent_git_commit"])
    observed_parent = _git_output("rev-parse", f"{parent_commit}^{{commit}}")
    if observed_parent != parent_commit:
        raise G95RunnerError("exact committed G94 parent object is unavailable")
    config_identity = _source_identity(config_path)
    resume_state_sources = {
        "config": config_identity,
        "receiver_module": _source_identity(RECEIVER_MODULE_PATH),
        "measurement_tool": _source_identity(MEASUREMENT_TOOL_PATH),
    }
    implementation_sources: dict[str, dict[str, Any]] = {}
    for source_path in (
        MEASUREMENT_TOOL_PATH.resolve(strict=True),
        RECEIVER_MODULE_PATH.resolve(strict=True),
        (REPO / "src/tac/witness_dsl/taskspace_g94_sequential_typed_actuator_product_v1.py").resolve(strict=True),
        (REPO / "src/tac/witness_dsl/taskspace_g89_class_complete_semantic_compiler_v1.py").resolve(strict=True),
        (REPO / "src/tac/witness_dsl/taskspace_g88_population_conditional_y0_pvsa_v1.py").resolve(strict=True),
        (REPO / "src/tac/scorer.py").resolve(strict=True),
        (REPO / "upstream/modules.py").resolve(strict=True),
        (REPO / "upstream/frame_utils.py").resolve(strict=True),
    ):
        implementation_sources[str(source_path.relative_to(REPO))] = {
            "bytes": source_path.stat().st_size,
            "sha256": _sha256_file(source_path),
        }
    payload = {
        "schema": PREFLIGHT_SCHEMA,
        "config": config_identity,
        "output_root": str(resolved_root),
        "selected_storage_tier": {
            "root": str(primary),
            "priority": 0,
        },
        "free_bytes_at_preflight": usage.free,
        "safety_reserve_bytes": reserve,
        "storage_status": "PASS",
        "input_identities_hashed_before_scorer_load": identities,
        "resume_state_sources_hashed_before_scorer_load": resume_state_sources,
        "implementation_sources_hashed_before_scorer_load": implementation_sources,
        "g94_parent_git_commit_verified": parent_commit,
        "git_head_observed_in_shared_dirty_tree": _git_output("rev-parse", "HEAD"),
        "dirty_tree_qualifier": bool(_git_output("status", "--short", "--untracked-files=all")),
        "determinism": {
            "seed": int(config["seed"]),
            "torch_num_threads": int(config["torch_num_threads"]),
            "torch_num_interop_threads": 1,
            "torch_deterministic_algorithms": True,
            "device": "cpu",
            "mps_authority": False,
            "cuda_authority": False,
        },
        "large_artifact_policy": {
            "new_raw_video_materialized": False,
            "maximum_pair_count": int(config["pair_count"]),
            "maximum_scorer_batch_pairs": int(config["scorer_batch_pairs"]),
            "atomic_temporary_files_success_cleaned": True,
            "immutable_stage_artifacts": True,
            "certify_or_block": True,
        },
        "false_authority": {
            "research_only": True,
            "candidate_claim": False,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
        },
    }
    receipt_path = run_root / "stage_00_preflight/receipt.json"
    if receipt_path.exists():
        preserved = _read_mapping(receipt_path, label="preserved stage_00 preflight")
        _validate_stage00_reuse_state(preserved, payload)
        payload["free_bytes_at_preflight"] = preserved.get("free_bytes_at_preflight")
    _write_json_once(receipt_path, payload)
    return payload


def _g89_program(semantic_p_sha256: str) -> ClassCompleteSemanticProgramV1:
    return ClassCompleteSemanticProgramV1(
        semantic_archive_sha256=semantic_p_sha256,
        topology_templates=(SharedTopologyTemplateV1(0, "birth", "box", 1, 3, 3),),
        topology_applications=(
            SharedTopologyApplicationV1(0, "UndrivableBoundary", 0, 0, 0),
            SharedTopologyApplicationV1(0, "Road", 0, 0, 5),
            SharedTopologyApplicationV1(0, "Lane", 0, 0, 10),
            SharedTopologyApplicationV1(0, "MyCar", 0, 0, 20),
            SharedTopologyApplicationV1(0, "Movable", 0, 0, 15),
        ),
        boundary_shearlets=(
            BoundaryShearletAtomV1(
                0,
                "UndrivableBoundary",
                178,
                437,
                4,
                8,
                0,
                64,
            ),
            BoundaryShearletAtomV1(0, "Road", 240, 494, 4, 8, 0, 64),
        ),
        island_shapes=(),
        worldsheet_tracks=(
            MovableWorldsheetTrackV1(
                object_id=0,
                birth_pair=0,
                death_pair_exclusive=1,
                center_y=30,
                center_x=30,
                radius_y=2,
                radius_x=2,
                angle_u8=0,
                skew_q6=0,
                taper_q6=0,
                curvelet_q6=0,
            ),
        ),
        worldsheet_knots=(),
        lane_programs=(
            LanePeriodicProgramV1(
                line_index=0,
                birth_pair=0,
                death_pair_exclusive=600,
                dash_phase_origin_delta_q8=0,
                dash_phase_xi_gain_q8=0,
                width_bias_q8=64,
                width_slope_q12=0,
            ),
        ),
        lane_knots=(),
    )


def _g88_operand(
    *,
    base_member: bytes,
    semantic_p_sha256: str,
) -> PopulationConditionalOperandV1:
    return PopulationConditionalOperandV1(
        base_pvsa_member_sha256=_sha256_bytes(base_member),
        semantic_p_sha256=semantic_p_sha256,
        controls=(ConditionalY0ControlV1.copy_conditional_y1(0),),
    )


def _reconstruct_exact_g94(
    *,
    config: Mapping[str, Any],
    source_pair_ids: tuple[int, ...],
) -> tuple[SequentialTypedArchiveBuildV1, np.ndarray, dict[str, Any]]:
    base_archive_path, _archive_identity = _exact_path_identity(
        config["base_archive"],
        label="base_archive",
    )
    base_member_path, _member_identity = _exact_path_identity(
        config["base_member"],
        label="base_member",
    )
    exact_base_archive = parse_taskspace_outer_archive(
        base_archive_path.read_bytes(),
        expected_archive_sha256=str(config["base_archive"]["sha256"]),
    )
    base_member = base_member_path.read_bytes()
    if exact_base_archive.member_bytes != base_member or exact_base_archive.member_sha256 != str(
        config["base_member"]["sha256"]
    ):
        raise G95RunnerError("base archive/member exact reopen differs")
    semantic_p_sha256 = str(config["semantic_p_sha256"])
    program = _g89_program(semantic_p_sha256)
    conditional = _g88_operand(
        base_member=base_member,
        semantic_p_sha256=semantic_p_sha256,
    )
    g94 = config["g94"]
    if (
        len(program.to_bytes()) != int(g94["g89_program_bytes"])
        or program.sha256 != str(g94["g89_program_sha256"])
        or len(conditional.to_bytes()) != int(g94["g88_operand_bytes"])
        or _sha256_bytes(conditional.to_bytes()) != str(g94["g88_operand_sha256"])
    ):
        raise G95RunnerError("reconstructed exact G89/G88 fixture sections drifted")
    build = build_sequential_typed_archive(
        base_pvsa_member_bytes=base_member,
        g89_program_bytes=program.to_bytes(),
        g88_conditional_operand_bytes=conditional.to_bytes(),
    )
    selected_outer = build.outer_build.selected
    if (
        len(build.selected.member_bytes) != int(g94["product_member_bytes"])
        or build.selected.member_sha256 != str(g94["product_member_sha256"])
        or selected_outer.archive_nbytes != int(g94["selected_outer_archive_bytes"])
        or selected_outer.archive_sha256 != str(g94["selected_outer_archive_sha256"])
        or build.conditioning_state_sha256 != str(g94["conditioning_state_sha256"])
        or build.conditioning_state_sha256 != EXPECTED_CONDITIONING_SHA256
    ):
        raise G95RunnerError("reconstructed G94 product/archive/conditioning identity drifted")
    receiver = build.selected.open_receiver(verify_member_effects=False)
    whole_preconditional_digest = hashlib.sha256()
    whole_pair_count = 0
    for population_batch in receiver.iter_camera_pair_batches(batch_pairs=16):
        population_preconditional = np.ascontiguousarray(population_batch.preconditional_camera_pairs)
        whole_preconditional_digest.update(memoryview(population_preconditional).cast("B"))
        whole_pair_count += population_preconditional.shape[0]
    if whole_pair_count != 600:
        raise G95RunnerError("exact G94 population stream did not cover all 600 pairs")
    whole_preconditional_camera_sha256 = whole_preconditional_digest.hexdigest()
    bounded = receiver.render_camera_pair_batch(source_pair_ids)
    preconditional = np.ascontiguousarray(bounded.preconditional_camera_pairs)
    if (
        bounded.product_member_sha256 != EXPECTED_G94_PRODUCT_SHA256
        or bounded.conditioning_state_sha256 != EXPECTED_CONDITIONING_SHA256
        or not np.array_equal(
            preconditional[:, 0],
            bounded.base_incumbent_camera_pairs[:, 0],
        )
    ):
        raise G95RunnerError("exact G94 preconditional chronological state differs")
    metadata = {
        "parent_git_commit": str(g94["parent_git_commit"]),
        "base_archive": {
            "bytes": exact_base_archive.archive_nbytes,
            "sha256": exact_base_archive.archive_sha256,
        },
        "base_member": {
            "bytes": len(base_member),
            "sha256": _sha256_bytes(base_member),
        },
        "g89_program": {
            "bytes": len(program.to_bytes()),
            "sha256": program.sha256,
        },
        "g88_operand": {
            "bytes": len(conditional.to_bytes()),
            "sha256": _sha256_bytes(conditional.to_bytes()),
        },
        "g94_product_member": {
            "bytes": len(build.selected.member_bytes),
            "sha256": build.selected.member_sha256,
        },
        "g94_selected_outer_archive": {
            "bytes": selected_outer.archive_nbytes,
            "sha256": selected_outer.archive_sha256,
        },
        "conditioning_state_sha256": build.conditioning_state_sha256,
        "whole_population_pair_count": whole_pair_count,
        "whole_preconditional_camera_sha256": whole_preconditional_camera_sha256,
        "source_pair_ids": list(source_pair_ids),
        "preconditional_camera_sha256": _sha256_array(preconditional),
        "exact_y1_sha256": _sha256_array(preconditional[:, 1]),
        "non_final_semantic_y1": True,
        "research_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    return build, preconditional, metadata


def _stage_exact_g94_state(
    *,
    config: Mapping[str, Any],
    run_root: Path,
    source_pair_ids: tuple[int, ...],
) -> tuple[SequentialTypedArchiveBuildV1, np.ndarray, dict[str, Any]]:
    build, preconditional, metadata = _reconstruct_exact_g94(
        config=config,
        source_pair_ids=source_pair_ids,
    )
    stage = run_root / "stage_01_exact_g94_state"
    _write_once_or_equal(stage / "g94_product_member.bin", build.selected.member_bytes)
    _write_once_or_equal(
        stage / "g94_selected_outer_archive.zip",
        build.outer_build.selected.archive_bytes,
    )
    _write_npz_once(
        stage / "preconditional_camera_pairs.npz",
        source_pair_ids=np.asarray(source_pair_ids, dtype=np.uint16),
        preconditional_camera_pairs=preconditional,
    )
    metadata["checkpoint_kind"] = "BYTE_CLOSE_LOADABLE_EXACT_G94_PRODUCT_AND_PRECONDITIONAL_BATCH"
    metadata["artifacts"] = {
        "g94_product_member.bin": {
            "bytes": len(build.selected.member_bytes),
            "sha256": _sha256_bytes(build.selected.member_bytes),
        },
        "g94_selected_outer_archive.zip": {
            "bytes": build.outer_build.selected.archive_nbytes,
            "sha256": build.outer_build.selected.archive_sha256,
        },
        "preconditional_camera_pairs.npz": {
            "bytes": (stage / "preconditional_camera_pairs.npz").stat().st_size,
            "sha256": _sha256_file(stage / "preconditional_camera_pairs.npz"),
        },
    }
    _write_json_once(stage / "receipt.json", metadata)
    return build, preconditional, metadata


def _load_pose_targets(
    *,
    config: Mapping[str, Any],
    source_pair_ids: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_path, cache_identity = _exact_path_identity(
        config["gt_cache"],
        label="gt_cache",
    )
    with np.load(cache_path, allow_pickle=False) as archive:
        if "gt_poses" not in archive.files:
            raise G95RunnerError("GT cache lacks exact gt_poses member")
        all_targets = np.asarray(archive["gt_poses"])
    if (
        all_targets.dtype != np.float64
        or all_targets.shape != (600, 6)
        or not np.all(np.isfinite(all_targets))
        or _sha256_array(all_targets) != str(config["gt_poses_member_sha256"])
    ):
        raise G95RunnerError("GT Pose target cache changed exact float64 [600,6] custody")
    targets = np.ascontiguousarray(all_targets[list(source_pair_ids)])
    return targets, {
        "cache": cache_identity,
        "member": "gt_poses",
        "full_member_shape": [600, 6],
        "full_member_dtype": "float64",
        "full_member_sha256": _sha256_array(all_targets),
        "selected_target_sha256": _sha256_array(targets),
        "source_pair_ids": list(source_pair_ids),
        "receiver_boundary": "ENCODER_ONLY_NEVER_PACKED_IN_G95_PACKET",
    }


def _load_frozen_posenet(weights_path: Path):
    import torch
    from modules import PoseNet
    from safetensors.torch import load_file

    model = PoseNet().eval()
    model.load_state_dict(load_file(str(weights_path), device="cpu"))
    model = model.to(torch.device("cpu"))
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _pose_tensor(output: Any):
    pose = output["pose"] if isinstance(output, dict) else output
    if pose.ndim != 2 or pose.shape[1] < 6:
        raise G95RunnerError("PoseNet output changed exact first-six ABI")
    return pose[:, :6]


def _torch_camera_pairs(pairs_uint8: np.ndarray):
    import torch

    pairs = np.asarray(pairs_uint8)
    if pairs.dtype != np.uint8 or pairs.ndim != 5 or pairs.shape[1:] != PAIR_SHAPE or not 1 <= pairs.shape[0] <= 16:
        raise G95RunnerError("PoseNet input must be exact uint8 [1..16,2,874,1164,3]")
    return torch.from_numpy(np.ascontiguousarray(pairs).copy()).permute(0, 1, 4, 2, 3).contiguous().float()


def _pose_predictions(posenet, pairs_uint8: np.ndarray) -> np.ndarray:
    import torch

    pairs = _torch_camera_pairs(pairs_uint8)
    with torch.inference_mode():
        prediction = _pose_tensor(posenet(posenet.preprocess_input(pairs)))
    result = prediction.detach().cpu().numpy().astype(np.float64)
    if result.shape != (pairs.shape[0], 6) or not np.all(np.isfinite(result)):
        raise G95RunnerError("PoseNet first-six prediction is nonfinite or changed shape")
    return np.ascontiguousarray(result)


def _patch_and_check_differentiable_yuv6(
    *,
    posenet,
    exact_pairs_uint8: np.ndarray,
) -> dict[str, Any]:
    import torch

    from tac.scorer import make_scorers_differentiable

    exact_pairs = _torch_camera_pairs(exact_pairs_uint8)
    with torch.inference_mode():
        official_preprocess = posenet.preprocess_input(exact_pairs).detach().clone()
        official_prediction = _pose_tensor(posenet(official_preprocess)).detach().clone()
    make_scorers_differentiable(posenet, torch.nn.Identity())
    with torch.inference_mode():
        patched_preprocess = posenet.preprocess_input(exact_pairs).detach()
        patched_prediction = _pose_tensor(posenet(patched_preprocess)).detach()
    preprocess_max_abs = float(torch.max(torch.abs(official_preprocess - patched_preprocess)).item())
    prediction_max_abs = float(torch.max(torch.abs(official_prediction - patched_prediction)).item())
    if preprocess_max_abs > 1e-5 or prediction_max_abs > 1e-5:
        raise G95RunnerError("differentiable evaluator YUV6 patch failed checked forward equivalence")
    return {
        "official_preprocess_sha256": _sha256_array(official_preprocess.cpu().numpy().astype(np.float32)),
        "patched_preprocess_sha256": _sha256_array(patched_preprocess.cpu().numpy().astype(np.float32)),
        "preprocess_max_abs_error": preprocess_max_abs,
        "official_prediction_sha256": _sha256_array(official_prediction.cpu().numpy().astype(np.float32)),
        "patched_prediction_sha256": _sha256_array(patched_prediction.cpu().numpy().astype(np.float32)),
        "prediction_max_abs_error": prediction_max_abs,
        "status": "PASS",
        "scope": ("ENCODER_SIDE_DIFFERENTIABLE_EQUIVALENT_ONLY_RECEIVER_REMAINS_NUMPY_AND_SCORER_FREE"),
    }


def _pose_replay_metrics(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 6:
        raise G95RunnerError("Pose replay prediction/target shapes differ")
    per_pair = np.mean((prediction - target) ** 2, axis=1).astype(np.float64)
    d_pose = float(np.mean(per_pair))
    return {
        "d_pose": d_pose,
        "pose_term": math.sqrt(10.0 * d_pose),
        "per_pair_d_pose": per_pair.tolist(),
        "per_pair_d_pose_sha256": _sha256_array(per_pair),
        "prediction_sha256": _sha256_array(prediction),
        "target_sha256": _sha256_array(target),
    }


def build_noop_control_rows(
    *,
    posenet,
    preconditional_camera_pairs: np.ndarray,
    targets: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    """Measure the two explicit pre-chart controls on actual uint8 pairs."""

    preconditional = np.asarray(preconditional_camera_pairs)
    pass_pairs = np.ascontiguousarray(preconditional).copy()
    copy_pairs = np.ascontiguousarray(preconditional).copy()
    copy_pairs[:, 0] = copy_pairs[:, 1]
    pass_prediction = _pose_predictions(posenet, pass_pairs)
    copy_prediction = _pose_predictions(posenet, copy_pairs)
    pass_metrics = _pose_replay_metrics(
        predictions=pass_prediction,
        targets=targets,
    )
    copy_metrics = _pose_replay_metrics(
        predictions=copy_prediction,
        targets=targets,
    )
    common = {
        "packet_bytes": 0,
        "basis_bytes": 0,
        "coefficient_bytes": 0,
        "gradient_space_proposal": False,
        "exact_receiver_replay": True,
        "research_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    pass_row = {
        **common,
        "treatment_id": "PASS_PRECONDITIONAL_Y0",
        "mode": G95ControlModeV1.PASS_PRECONDITIONAL_Y0.name,
        "camera_sha256": _sha256_array(pass_pairs),
        "exact_y1_sha256": _sha256_array(pass_pairs[:, 1]),
        **pass_metrics,
    }
    copy_row = {
        **common,
        "treatment_id": "COPY_EXACT_CONDITIONAL_Y1",
        "mode": G95ControlModeV1.COPY_EXACT_CONDITIONAL_Y1.name,
        "camera_sha256": _sha256_array(copy_pairs),
        "exact_y1_sha256": _sha256_array(copy_pairs[:, 1]),
        **copy_metrics,
    }
    return pass_row, copy_row, pass_prediction, copy_prediction


def _stage_noop_controls(
    *,
    run_root: Path,
    posenet,
    preconditional_camera_pairs: np.ndarray,
    targets: np.ndarray,
    source_pair_ids: tuple[int, ...],
    yuv6_equivalence: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    pass_row, copy_row, pass_prediction, copy_prediction = build_noop_control_rows(
        posenet=posenet,
        preconditional_camera_pairs=preconditional_camera_pairs,
        targets=targets,
    )
    stage = run_root / "stage_02_noop_controls"
    _write_npz_once(
        stage / "pose_rows.npz",
        source_pair_ids=np.asarray(source_pair_ids, dtype=np.uint16),
        targets=np.asarray(targets, dtype=np.float64),
        pass_predictions=pass_prediction,
        copy_predictions=copy_prediction,
    )
    receipt = {
        "schema": "tac.g95_noop_control_stage.v1",
        "g94_product_member_sha256": EXPECTED_G94_PRODUCT_SHA256,
        "g94_conditioning_state_sha256": EXPECTED_CONDITIONING_SHA256,
        "source_pair_ids": list(source_pair_ids),
        "controls": [pass_row, copy_row],
        "differentiable_yuv6_forward_equivalence": dict(yuv6_equivalence),
        "checkpoint_kind": "COMPLETE_EXACT_CPU_POSENET_NOOP_AND_COPY_ROWS",
        "artifacts": {
            "pose_rows.npz": {
                "bytes": (stage / "pose_rows.npz").stat().st_size,
                "sha256": _sha256_file(stage / "pose_rows.npz"),
            }
        },
        "research_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    _write_json_once(stage / "receipt.json", receipt)
    return pass_row, copy_row


def _torch_pose_from_residual_grid(
    *,
    posenet,
    exact_y1_chw,
    residual_grid_hwc,
):
    import torch
    import torch.nn.functional as torch_functional

    residual_chw = residual_grid_hwc.permute(2, 0, 1).unsqueeze(0)
    residual_camera = torch_functional.interpolate(
        residual_chw,
        size=(PAIR_SHAPE[1], PAIR_SHAPE[2]),
        mode="bilinear",
        align_corners=False,
    )[0]
    y0_unquantized = exact_y1_chw + residual_camera
    clamped = y0_unquantized.clamp(0.0, 255.0)
    y0_ste = clamped + (torch.round(clamped) - clamped).detach()
    pair = torch.stack((y0_ste, exact_y1_chw), dim=0).unsqueeze(0)
    return _pose_tensor(posenet(posenet.preprocess_input(pair)))[0]


def _pose_and_grid_jacobian(
    *,
    posenet,
    exact_y1_uint8: np.ndarray,
    residual_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    grid = torch.from_numpy(np.ascontiguousarray(residual_grid).copy()).float().requires_grad_(True)
    y1 = torch.from_numpy(np.ascontiguousarray(exact_y1_uint8).copy()).permute(2, 0, 1).contiguous().float()
    pose = _torch_pose_from_residual_grid(
        posenet=posenet,
        exact_y1_chw=y1,
        residual_grid_hwc=grid,
    )
    jacobian = np.empty((6, grid.numel()), dtype=np.float64)
    for dimension in range(6):
        gradient = torch.autograd.grad(
            pose[dimension],
            grid,
            retain_graph=dimension < 5,
            create_graph=False,
        )[0]
        jacobian[dimension] = gradient.detach().cpu().numpy().reshape(-1).astype(np.float64)
    prediction = pose.detach().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(prediction)):
        raise G95RunnerError("PoseNet grid VJP produced nonfinite values")
    return prediction, jacobian


def _pose_from_residual_grid(
    *,
    posenet,
    exact_y1_uint8: np.ndarray,
    residual_grid: np.ndarray,
) -> np.ndarray:
    import torch

    grid = torch.from_numpy(np.ascontiguousarray(residual_grid).copy()).float()
    y1 = torch.from_numpy(np.ascontiguousarray(exact_y1_uint8).copy()).permute(2, 0, 1).contiguous().float()
    with torch.inference_mode():
        pose = _torch_pose_from_residual_grid(
            posenet=posenet,
            exact_y1_chw=y1,
            residual_grid_hwc=grid,
        )
    prediction = pose.detach().cpu().numpy().astype(np.float64)
    if prediction.shape != (6,) or not np.all(np.isfinite(prediction)):
        raise G95RunnerError("dense teacher Pose prediction became nonfinite")
    return prediction


def _lm_minimum_norm_delta(
    *,
    jacobian: np.ndarray,
    residual: np.ndarray,
    damping: float,
) -> tuple[np.ndarray, float]:
    j = np.asarray(jacobian, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64)
    gram = j @ j.T
    maximum_eigenvalue = float(max(np.linalg.eigvalsh(gram)[-1], 1e-30))
    system = gram + damping * maximum_eigenvalue * np.eye(gram.shape[0], dtype=np.float64)
    try:
        dual = np.linalg.solve(system, r)
    except np.linalg.LinAlgError:
        dual = np.linalg.lstsq(system, r, rcond=None)[0]
    delta = -(j.T @ dual)
    if not np.all(np.isfinite(delta)):
        raise G95RunnerError("damped natural-gradient step became nonfinite")
    return delta, maximum_eigenvalue


def _collect_shared_costate_basis(
    *,
    posenet,
    exact_y1_uint8: np.ndarray,
    targets: np.ndarray,
    requested_rank: int,
    grid_height: int,
    grid_width: int,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Collect a real dense-teacher VJP union and deterministically SVD-prune it."""

    settings = config["basis_fit"]
    residual_grids = np.zeros(
        (len(exact_y1_uint8), grid_height, grid_width, 3),
        dtype=np.float32,
    )
    damping = float(settings["dense_teacher_initial_damping"])
    rows: list[np.ndarray] = []
    row_telemetry: list[dict[str, Any]] = []
    teacher_trace: list[dict[str, Any]] = []
    singular_values = np.empty(0, dtype=np.float64)
    right_vectors = np.empty((0, residual_grids[0].size), dtype=np.float64)
    retained = 0
    for anchor_index in range(int(settings["dense_teacher_anchor_limit"])):
        anchor_predictions: list[np.ndarray] = []
        anchor_jacobians: list[np.ndarray] = []
        for pair_local, source_pair_id in enumerate(
            range(int(config["pair_start"]), int(config["pair_start"]) + int(config["pair_count"]))
        ):
            prediction, jacobian = _pose_and_grid_jacobian(
                posenet=posenet,
                exact_y1_uint8=exact_y1_uint8[pair_local],
                residual_grid=residual_grids[pair_local],
            )
            anchor_predictions.append(prediction)
            anchor_jacobians.append(jacobian)
            for dimension in range(6):
                rows.append(jacobian[dimension].copy())
                row_telemetry.append(
                    {
                        "anchor_index": anchor_index,
                        "source_pair_id": source_pair_id,
                        "pose_dimension": dimension,
                        "costate_l2_norm": float(np.linalg.norm(jacobian[dimension])),
                        "costate_sha256": _sha256_array(jacobian[dimension]),
                    }
                )
        costate_matrix = np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float64)
        _left, singular_values, right_vectors = np.linalg.svd(
            costate_matrix,
            full_matrices=False,
        )
        if singular_values.size:
            retained = int(
                np.count_nonzero(singular_values >= singular_values[0] * float(settings["svd_relative_threshold"]))
            )
        anchor_mse = float(np.mean((np.stack(anchor_predictions, axis=0) - np.asarray(targets, dtype=np.float64)) ** 2))
        teacher_trace.append(
            {
                "anchor_index": anchor_index,
                "costate_rows": len(rows),
                "svd_retained_rank": retained,
                "gradient_space_d_pose": anchor_mse,
                "damping_before_step": damping,
            }
        )
        if retained >= requested_rank:
            break

        accepted = False
        for line_search_index in range(int(settings["dense_teacher_line_search_steps"])):
            candidate = residual_grids.copy()
            for pair_local, (prediction, jacobian) in enumerate(zip(anchor_predictions, anchor_jacobians, strict=True)):
                delta, maximum_eigenvalue = _lm_minimum_norm_delta(
                    jacobian=jacobian,
                    residual=prediction - targets[pair_local],
                    damping=damping,
                )
                candidate[pair_local] += delta.reshape(
                    grid_height,
                    grid_width,
                    3,
                ).astype(np.float32)
                teacher_trace[-1].setdefault("maximum_eigenvalues", []).append(maximum_eigenvalue)
            candidate_predictions = []
            for pair_local in range(len(exact_y1_uint8)):
                prediction = _pose_from_residual_grid(
                    posenet=posenet,
                    exact_y1_uint8=exact_y1_uint8[pair_local],
                    residual_grid=candidate[pair_local],
                )
                candidate_predictions.append(prediction)
            candidate_mse = float(
                np.mean((np.stack(candidate_predictions, axis=0) - np.asarray(targets, dtype=np.float64)) ** 2)
            )
            if candidate_mse < anchor_mse:
                residual_grids = candidate
                damping = max(
                    damping * float(settings["dense_teacher_damping_decrease"]),
                    1e-12,
                )
                teacher_trace[-1].update(
                    {
                        "step_accepted": True,
                        "line_search_index": line_search_index,
                        "candidate_gradient_space_d_pose": candidate_mse,
                    }
                )
                accepted = True
                break
            damping = min(
                damping * float(settings["dense_teacher_damping_increase"]),
                1e12,
            )
        if not accepted:
            teacher_trace[-1]["step_accepted"] = False
            break
    if retained < requested_rank:
        raise G95RunnerError(f"real costate union retained rank {retained} below requested {requested_rank}")
    selected = np.ascontiguousarray(right_vectors[:requested_rank], dtype=np.float64)
    for rank_index in range(requested_rank):
        pivot = int(np.argmax(np.abs(selected[rank_index])))
        if selected[rank_index, pivot] < 0.0:
            selected[rank_index] *= -1.0
    basis_float = selected.reshape(requested_rank, grid_height, grid_width, 3).astype(np.float32)
    scales = np.maximum(
        np.max(np.abs(basis_float), axis=(1, 2, 3)) / np.float32(127.0),
        np.float32(settings["basis_scale_floor"]),
    ).astype(np.float32)
    basis_q = np.rint(basis_float / scales.reshape(requested_rank, 1, 1, 1)).clip(-127, 127).astype(np.int8)
    if np.any(np.max(np.abs(basis_q.astype(np.int16)), axis=(1, 2, 3)) == 0):
        raise G95RunnerError("quantized retained costate basis contains a zero direction")
    basis_metadata = {
        "requested_rank": requested_rank,
        "retained_rank": retained,
        "costate_matrix_shape": list(costate_matrix.shape),
        "costate_matrix_sha256": _sha256_array(costate_matrix),
        "singular_values": singular_values.tolist(),
        "canonical_sign_rule": "largest_absolute_coordinate_positive_first_index_tie",
        "basis_q_sha256": _sha256_array(basis_q),
        "basis_scales_sha256": _sha256_array(scales),
        "dense_teacher_is_encoder_only_rate_dead": True,
        "dense_teacher_crosses_receiver_boundary": False,
        "teacher_trace": teacher_trace,
    }
    return (
        basis_q,
        scales,
        np.asarray(singular_values[:requested_rank], dtype=np.float64),
        row_telemetry,
        basis_metadata,
    )


def _load_or_build_basis(
    *,
    stage: Path,
    posenet,
    exact_y1_uint8: np.ndarray,
    targets: np.ndarray,
    rank: int,
    config: Mapping[str, Any],
    resume_state_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    checkpoint_path = stage / "basis_checkpoint.npz"
    receipt_path = stage / "basis_receipt.json"
    if checkpoint_path.exists() or receipt_path.exists():
        if not checkpoint_path.is_file() or not receipt_path.is_file():
            raise G95RunnerError("partial basis checkpoint cannot be resumed")
        receipt = _read_mapping(receipt_path, label="preserved G95 basis receipt")
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            required = {
                "basis_q",
                "basis_scales",
                "singular_values",
                "resume_state_key",
            }
            if set(archive.files) != required:
                raise G95RunnerError("preserved basis checkpoint member set differs")
            _verify_resume_state_array(
                archive["resume_state_key"],
                expected_resume_state_key=resume_state_key,
                label="preserved basis checkpoint",
            )
            basis_q = np.asarray(archive["basis_q"])
            basis_scales = np.asarray(archive["basis_scales"])
            singular_values = np.asarray(archive["singular_values"])
        if (
            basis_q.dtype != np.int8
            or basis_q.shape != (rank, 48, 64, 3)
            or basis_scales.dtype != np.float32
            or basis_scales.shape != (rank,)
            or singular_values.dtype != np.float64
            or singular_values.shape != (rank,)
            or _sha256_array(basis_q) != receipt.get("basis_q_sha256")
            or _sha256_array(basis_scales) != receipt.get("basis_scales_sha256")
            or _sha256_array(singular_values) != receipt.get("selected_singular_values_sha256")
            or receipt.get("g94_conditioning_state_sha256") != EXPECTED_CONDITIONING_SHA256
            or receipt.get("resume_state_key") != resume_state_key
        ):
            raise G95RunnerError("preserved basis checkpoint exact custody differs")
        telemetry = receipt.get("costate_rows")
        metadata = receipt.get("basis_metadata")
        if not isinstance(telemetry, list) or not isinstance(metadata, dict):
            raise G95RunnerError("preserved basis receipt telemetry differs")
        return basis_q, basis_scales, singular_values, telemetry, metadata
    basis_q, basis_scales, singular_values, telemetry, metadata = _collect_shared_costate_basis(
        posenet=posenet,
        exact_y1_uint8=exact_y1_uint8,
        targets=targets,
        requested_rank=rank,
        grid_height=48,
        grid_width=64,
        config=config,
    )
    _write_npz_once(
        checkpoint_path,
        basis_q=basis_q,
        basis_scales=basis_scales,
        singular_values=singular_values,
        resume_state_key=_resume_state_array(resume_state_key),
    )
    receipt = {
        "schema": "tac.g95_shared_costate_basis_checkpoint.v1",
        "rank": rank,
        "grid_height": 48,
        "grid_width": 64,
        "g94_product_member_sha256": EXPECTED_G94_PRODUCT_SHA256,
        "g94_conditioning_state_sha256": EXPECTED_CONDITIONING_SHA256,
        "resume_state_key": resume_state_key,
        "basis_q_sha256": _sha256_array(basis_q),
        "basis_scales_sha256": _sha256_array(basis_scales),
        "selected_singular_values_sha256": _sha256_array(singular_values),
        "costate_rows": telemetry,
        "basis_metadata": metadata,
        "checkpoint_kind": "IMMUTABLE_ENCODER_SIDE_VJP_SVD_QUANTIZED_BASIS",
        "research_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    _write_json_once(receipt_path, receipt)
    return basis_q, basis_scales, singular_values, telemetry, metadata


def _quantize_coefficients(
    coefficients: np.ndarray,
    *,
    scale_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(coefficients, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 1 or not np.all(np.isfinite(raw)):
        raise G95RunnerError("coefficient proposal must be finite [pairs,rank]")
    scales = np.maximum(
        np.max(np.abs(raw), axis=0) / 32767.0,
        scale_floor,
    ).astype(np.float32)
    coefficients_q = (
        np.rint(raw / scales.astype(np.float64))
        .clip(
            -32767,
            32767,
        )
        .astype(np.int16)
    )
    dequantized = coefficients_q.astype(np.float64) * scales.astype(np.float64)
    return coefficients_q, scales, np.ascontiguousarray(dequantized)


def _packet_and_exact_replay(
    *,
    posenet,
    preconditional_camera_pairs: np.ndarray,
    targets: np.ndarray,
    source_pair_ids: tuple[int, ...],
    basis_q: np.ndarray,
    basis_scales: np.ndarray,
    coefficients_q: np.ndarray,
    coefficient_scales: np.ndarray,
    wire_custody: Mapping[str, str],
) -> tuple[
    bytes,
    bytes,
    PopulationPosePreimageChartWireSetV1,
    PopulationPosePreimageChartBatchResultV1,
    np.ndarray,
    dict[str, Any],
]:
    preconditional_camera_sha256 = _sha256_array(preconditional_camera_pairs)
    basis_object = encode_population_pose_preimage_basis(
        g94_product_member_sha256=EXPECTED_G94_PRODUCT_SHA256,
        g94_conditioning_state_sha256=EXPECTED_CONDITIONING_SHA256,
        whole_preconditional_camera_sha256=str(wire_custody["whole_preconditional_camera_sha256"]),
        selected_target_table_sha256=str(wire_custody["selected_target_table_sha256"]),
        posenet_weights_sha256=str(wire_custody["posenet_weights_sha256"]),
        basis_q=basis_q,
        basis_scales=basis_scales,
    )
    parsed_basis = parse_population_pose_preimage_basis(
        basis_object,
        expected_object_sha256=_sha256_bytes(basis_object),
    )
    coefficient_chunk = encode_population_pose_preimage_coefficient_chunk(
        basis_object_sha256=parsed_basis.object_sha256,
        population_state_key_sha256=parsed_basis.population_state_key,
        preconditional_camera_sha256=preconditional_camera_sha256,
        selected_target_sha256=str(wire_custody["selected_target_sha256"]),
        source_pair_ids=source_pair_ids,
        rank=parsed_basis.rank,
        coefficients_q=coefficients_q,
        coefficient_scales=coefficient_scales,
    )
    parsed_chunk = parse_population_pose_preimage_coefficient_chunk(
        coefficient_chunk,
        expected_object_sha256=_sha256_bytes(coefficient_chunk),
    )
    wire_set = PopulationPosePreimageChartWireSetV1(
        basis=parsed_basis,
        chunk=parsed_chunk,
    )
    receiver = PopulationPosePreimageChartReceiverV1.open(
        parsed_basis,
        expected_g94_product_member_sha256=EXPECTED_G94_PRODUCT_SHA256,
        expected_g94_conditioning_state_sha256=EXPECTED_CONDITIONING_SHA256,
        expected_whole_preconditional_camera_sha256=str(wire_custody["whole_preconditional_camera_sha256"]),
        expected_selected_target_table_sha256=str(wire_custody["selected_target_table_sha256"]),
        expected_posenet_weights_sha256=str(wire_custody["posenet_weights_sha256"]),
    )
    first = receiver.decode_preconditional_chunk(
        parsed_chunk,
        preconditional_camera_pairs,
    )
    second = receiver.decode_preconditional_chunk(
        parsed_chunk,
        preconditional_camera_pairs,
    )
    if (
        first.camera_sha256 != second.camera_sha256
        or not np.array_equal(first.camera_pairs, second.camera_pairs)
        or first.basis_object_sha256 != parsed_basis.object_sha256
        or first.coefficient_chunk_sha256 != parsed_chunk.object_sha256
    ):
        raise G95RunnerError("G95 tool-level deterministic NumPy double replay differs")
    predictions = _pose_predictions(posenet, first.camera_pairs)
    metrics = _pose_replay_metrics(predictions=predictions, targets=targets)
    metrics.update(
        {
            "basis_object_bytes": len(basis_object),
            "basis_object_sha256": parsed_basis.object_sha256,
            "coefficient_chunk_bytes": len(coefficient_chunk),
            "coefficient_chunk_sha256": parsed_chunk.object_sha256,
            "wire_set_sha256": wire_set.wire_set_sha256,
            "total_counted_bytes": wire_set.total_counted_bytes,
            "receiver_camera_sha256": first.camera_sha256,
            "receiver_preconditional_camera_sha256": first.preconditional_camera_sha256,
            "exact_y1_sha256": first.exact_y1_sha256,
            "changed_y0_values": first.changed_y0_values,
            "changed_y0_pixels": first.changed_y0_pixels,
            "deterministic_numpy_double_replay": True,
            "basis_and_chunk_parse_reencode_identity": True,
        }
    )
    return basis_object, coefficient_chunk, wire_set, first, predictions, metrics


def _pose_and_coefficient_jacobian(
    *,
    posenet,
    exact_y1_uint8: np.ndarray,
    basis_q: np.ndarray,
    basis_scales: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    dequantized_basis = basis_q.astype(np.float32) * basis_scales.reshape(
        basis_q.shape[0],
        1,
        1,
        1,
    )
    basis = torch.from_numpy(np.ascontiguousarray(dequantized_basis).copy()).float()
    coefficient = (
        torch.from_numpy(np.ascontiguousarray(coefficients, dtype=np.float32).copy()).float().requires_grad_(True)
    )
    residual_grid = torch.sum(
        coefficient.reshape(-1, 1, 1, 1) * basis,
        dim=0,
    )
    y1 = torch.from_numpy(np.ascontiguousarray(exact_y1_uint8).copy()).permute(2, 0, 1).contiguous().float()
    pose = _torch_pose_from_residual_grid(
        posenet=posenet,
        exact_y1_chw=y1,
        residual_grid_hwc=residual_grid,
    )
    jacobian = np.empty((6, basis_q.shape[0]), dtype=np.float64)
    for dimension in range(6):
        gradient = torch.autograd.grad(
            pose[dimension],
            coefficient,
            retain_graph=dimension < 5,
            create_graph=False,
        )[0]
        jacobian[dimension] = gradient.detach().cpu().numpy().astype(np.float64)
    prediction = pose.detach().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(jacobian)):
        raise G95RunnerError("coefficient-space PoseNet Jacobian became nonfinite")
    return prediction, jacobian


def _write_iteration_checkpoint(
    *,
    stage: Path,
    iteration: int,
    basis_object: bytes,
    coefficient_chunk: bytes,
    wire_set: PopulationPosePreimageChartWireSetV1,
    result: PopulationPosePreimageChartBatchResultV1,
    predictions: np.ndarray,
    metrics: Mapping[str, Any],
    coefficients_dequantized: np.ndarray,
    damping: float,
    proposal_telemetry: Sequence[Mapping[str, Any]],
    resume_state_key: str,
) -> None:
    iteration_root = stage / "iterations" / f"iteration_{iteration:04d}"
    _write_once_or_equal(iteration_root / "population_basis.bin", basis_object)
    _write_once_or_equal(iteration_root / "coefficient_chunk.bin", coefficient_chunk)
    _write_npz_once(
        iteration_root / "checkpoint.npz",
        coefficients_q=wire_set.coefficients_q,
        coefficient_scales=wire_set.coefficient_scales,
        coefficients_dequantized=np.asarray(coefficients_dequantized, dtype=np.float64),
        predictions=np.asarray(predictions, dtype=np.float64),
        resume_state_key=_resume_state_array(resume_state_key),
    )
    receipt = {
        "schema": "tac.g95_coefficient_fit_iteration_checkpoint.v1",
        "iteration": iteration,
        "damping_for_next_iteration": damping,
        "g94_product_member_sha256": EXPECTED_G94_PRODUCT_SHA256,
        "g94_conditioning_state_sha256": EXPECTED_CONDITIONING_SHA256,
        "resume_state_key": resume_state_key,
        "population_basis": {
            "bytes": len(basis_object),
            "sha256": wire_set.basis.object_sha256,
        },
        "coefficient_chunk": {
            "bytes": len(coefficient_chunk),
            "sha256": wire_set.chunk.object_sha256,
        },
        "wire_set_sha256": wire_set.wire_set_sha256,
        "basis_section_sha256": _sha256_bytes(wire_set.basis.basis_bytes),
        "coefficient_section_sha256": _sha256_bytes(wire_set.chunk.coefficients_bytes),
        "receiver_camera_sha256": result.camera_sha256,
        "metrics": dict(metrics),
        "proposal_telemetry": [dict(row) for row in proposal_telemetry],
        "checkpoint_kind": ("IMMUTABLE_BYTE_CLOSE_LOADABLE_QUANTIZED_COEFFICIENT_ITERATION"),
        "research_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    _write_json_once(iteration_root / "receipt.json", receipt)


def _resume_latest_iteration(
    *,
    stage: Path,
    posenet,
    preconditional_camera_pairs: np.ndarray,
    targets: np.ndarray,
    source_pair_ids: tuple[int, ...],
    basis_q: np.ndarray,
    basis_scales: np.ndarray,
    resume_state_key: str,
    wire_custody: Mapping[str, str],
) -> (
    tuple[
        int,
        float,
        np.ndarray,
        bytes,
        bytes,
        PopulationPosePreimageChartWireSetV1,
        PopulationPosePreimageChartBatchResultV1,
        np.ndarray,
        dict[str, Any],
    ]
    | None
):
    iteration_root = stage / "iterations"
    if not iteration_root.is_dir():
        return None
    candidates = sorted(iteration_root.glob("iteration_[0-9][0-9][0-9][0-9]"))
    if not candidates:
        return None
    latest = candidates[-1]
    receipt_path = latest / "receipt.json"
    basis_path = latest / "population_basis.bin"
    chunk_path = latest / "coefficient_chunk.bin"
    checkpoint_path = latest / "checkpoint.npz"
    if (
        not receipt_path.is_file()
        or not basis_path.is_file()
        or not chunk_path.is_file()
        or not checkpoint_path.is_file()
    ):
        raise G95RunnerError("latest G95 iteration checkpoint is partial")
    receipt = _read_mapping(receipt_path, label="latest G95 iteration receipt")
    if receipt.get("resume_state_key") != resume_state_key:
        raise G95RunnerError("latest G95 iteration receipt resume-state key differs")
    try:
        iteration = int(latest.name.rsplit("_", 1)[1])
        damping = float(receipt["damping_for_next_iteration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise G95RunnerError("latest G95 iteration metadata is malformed") from exc
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "coefficients_q",
            "coefficient_scales",
            "coefficients_dequantized",
            "predictions",
            "resume_state_key",
        }:
            raise G95RunnerError("latest G95 iteration checkpoint member set differs")
        _verify_resume_state_array(
            archive["resume_state_key"],
            expected_resume_state_key=resume_state_key,
            label="latest G95 iteration checkpoint",
        )
        coefficients_q = np.asarray(archive["coefficients_q"])
        coefficient_scales = np.asarray(archive["coefficient_scales"])
        coefficients_dequantized = np.asarray(archive["coefficients_dequantized"])
        preserved_predictions = np.asarray(archive["predictions"])
    (
        basis_object,
        coefficient_chunk,
        wire_set,
        result,
        predictions,
        metrics,
    ) = _packet_and_exact_replay(
        posenet=posenet,
        preconditional_camera_pairs=preconditional_camera_pairs,
        targets=targets,
        source_pair_ids=source_pair_ids,
        basis_q=basis_q,
        basis_scales=basis_scales,
        coefficients_q=coefficients_q,
        coefficient_scales=coefficient_scales,
        wire_custody=wire_custody,
    )
    preserved_metrics = receipt.get("metrics")
    if (
        basis_object != basis_path.read_bytes()
        or coefficient_chunk != chunk_path.read_bytes()
        or wire_set.basis.object_sha256 != receipt.get("population_basis", {}).get("sha256")
        or wire_set.chunk.object_sha256 != receipt.get("coefficient_chunk", {}).get("sha256")
        or coefficients_dequantized.dtype != np.float64
        or coefficients_dequantized.shape != coefficients_q.shape
        or not np.array_equal(
            coefficients_dequantized,
            coefficients_q.astype(np.float64) * coefficient_scales.astype(np.float64).reshape(1, -1),
        )
        or not np.array_equal(predictions, preserved_predictions)
        or not isinstance(preserved_metrics, dict)
        or metrics["prediction_sha256"] != preserved_metrics.get("prediction_sha256")
        or metrics["receiver_camera_sha256"] != preserved_metrics.get("receiver_camera_sha256")
    ):
        raise G95RunnerError("latest G95 iteration failed exact resume verification")
    return (
        iteration,
        damping,
        coefficients_dequantized,
        basis_object,
        coefficient_chunk,
        wire_set,
        result,
        predictions,
        metrics,
    )


def _load_iteration_proposal_history(
    stage: Path,
    *,
    resume_state_key: str,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    iteration_root = stage / "iterations"
    if not iteration_root.is_dir():
        return history
    for path in sorted(iteration_root.glob("iteration_[0-9][0-9][0-9][0-9]")):
        receipt_path = path / "receipt.json"
        checkpoint_path = path / "checkpoint.npz"
        if not receipt_path.is_file() or not checkpoint_path.is_file():
            raise G95RunnerError("G95 iteration history contains a partial checkpoint")
        receipt = _read_mapping(receipt_path, label="G95 iteration history receipt")
        if receipt.get("resume_state_key") != resume_state_key:
            raise G95RunnerError("G95 iteration history resume-state key differs")
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            if "resume_state_key" not in archive.files:
                raise G95RunnerError("G95 iteration history checkpoint lacks resume-state key")
            _verify_resume_state_array(
                archive["resume_state_key"],
                expected_resume_state_key=resume_state_key,
                label="G95 iteration history checkpoint",
            )
        rows = receipt.get("proposal_telemetry")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise G95RunnerError("G95 iteration proposal telemetry changed type")
        history.extend(dict(row) for row in rows)
    return history


def _direction_sensitivity_rows(
    *,
    posenet,
    preconditional_camera_pairs: np.ndarray,
    targets: np.ndarray,
    source_pair_ids: tuple[int, ...],
    wire_set: PopulationPosePreimageChartWireSetV1,
    final_predictions: np.ndarray,
    final_d_pose: float,
    singular_values: np.ndarray,
    wire_custody: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank_index in range(wire_set.rank):
        ablated_coefficients = np.array(
            wire_set.coefficients_q,
            dtype=np.int16,
            copy=True,
            order="C",
        )
        ablated_coefficients[:, rank_index] = 0
        (
            _basis_object,
            _coefficient_chunk,
            _ablated_wire_set,
            _result,
            predictions,
            metrics,
        ) = _packet_and_exact_replay(
            posenet=posenet,
            preconditional_camera_pairs=preconditional_camera_pairs,
            targets=targets,
            source_pair_ids=source_pair_ids,
            basis_q=wire_set.basis_q,
            basis_scales=wire_set.basis_scales,
            coefficients_q=ablated_coefficients,
            coefficient_scales=wire_set.coefficient_scales,
            wire_custody=wire_custody,
        )
        basis_direction_wire = wire_set.basis_q[rank_index].tobytes(order="C") + wire_set.basis_scales[
            rank_index : rank_index + 1
        ].astype(">f4", copy=False).tobytes(order="C")
        pair_response = np.asarray(final_predictions, dtype=np.float64) - predictions
        rows.append(
            {
                "rank_index": rank_index,
                "singular_value": float(singular_values[rank_index]),
                "quantized_basis_sha256": _sha256_bytes(basis_direction_wire),
                "pair_response_pose6": pair_response.tolist(),
                "pair_response_sha256": _sha256_array(pair_response),
                "ablated_exact_d_pose": float(metrics["d_pose"]),
                "exact_replay_delta_d_pose_final_minus_ablated": (final_d_pose - float(metrics["d_pose"])),
                "ablated_wire_set_sha256": str(metrics["wire_set_sha256"]),
                "ablated_receiver_camera_sha256": str(metrics["receiver_camera_sha256"]),
                "authority": "EXACT_G95_PACKET_PARSE_NUMPY_RECEIVER_CPU_POSENET_REPLAY",
            }
        )
    return rows


def _treatment_row(
    *,
    rank: int,
    wire_set: PopulationPosePreimageChartWireSetV1,
    result: PopulationPosePreimageChartBatchResultV1,
    metrics: Mapping[str, Any],
    copy_control_d_pose: float,
    sensitivity: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sections = {name: (size, sha) for name, size, sha in wire_set.counted_sections}
    return {
        "treatment_id": f"RANK_{rank:02d}_48X64",
        "mode": G95ControlModeV1.POPULATION_SHARED_PREIMAGE_CHART.name,
        "rank": rank,
        "grid_height": wire_set.grid_height,
        "grid_width": wire_set.grid_width,
        "d_pose": float(metrics["d_pose"]),
        "pose_term": float(metrics["pose_term"]),
        "per_pair_d_pose": list(metrics["per_pair_d_pose"]),
        "exact_replay_delta_d_pose_vs_copy_control": (float(metrics["d_pose"]) - copy_control_d_pose),
        "population_basis_object_bytes_p_once": wire_set.basis.counted_bytes,
        "population_basis_object_sha256": wire_set.basis.object_sha256,
        "coefficient_chunk_bytes": wire_set.chunk.counted_bytes,
        "coefficient_chunk_sha256": wire_set.chunk.object_sha256,
        "total_counted_wire_bytes": wire_set.total_counted_bytes,
        "wire_set_sha256": wire_set.wire_set_sha256,
        "pair_selector_bytes": wire_set.pair_selector_bytes,
        "basis_bytes": sections["shared_basis_i8"][0],
        "basis_scales_bytes": sections["basis_scales_f32be"][0],
        "coefficient_bytes": sections["per_pair_coefficients_i16be"][0],
        "coefficient_scales_bytes": sections["coefficient_scales_f32be"][0],
        "learned_payload_bytes": wire_set.learned_payload_bytes,
        "counted_sections": [
            {"name": name, "bytes": size, "sha256": sha} for name, size, sha in wire_set.counted_sections
        ],
        "receiver_camera_sha256": result.camera_sha256,
        "receiver_preconditional_camera_sha256": result.preconditional_camera_sha256,
        "chunk_bound_preconditional_camera_sha256": wire_set.chunk.preconditional_camera_sha256,
        "whole_population_state_key": wire_set.basis.population_state_key,
        "exact_y1_sha256": result.exact_y1_sha256,
        "changed_y0_values": result.changed_y0_values,
        "changed_y0_pixels": result.changed_y0_pixels,
        "deterministic_numpy_double_replay": True,
        "basis_and_chunk_parse_reencode_identity": True,
        "gradient_space_proposal": False,
        "exact_receiver_replay": True,
        "sufficient_reachability_coordinate_d_pose": REACHABILITY_THRESHOLD,
        "sufficient_reachability_coordinate_scope": REACHABILITY_COORDINATE_SCOPE,
        "sufficient_reachability_coordinate_crossed": (float(metrics["d_pose"]) <= REACHABILITY_THRESHOLD),
        "joint_feasible_surface": {
            "d_seg": None,
            "d_pose": float(metrics["d_pose"]),
            "outer_archive_bytes": None,
            "exact_score": None,
            "admission": OUTER_ZIP_SCORE_ADMISSION,
        },
        "miss_verdict_scope": MISS_VERDICT_SCOPE,
        "population_transfer_viability_claim": False,
        "sensitivity": [dict(row) for row in sensitivity],
        "missing_integration": MISSING_INTEGRATION,
        "research_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }


def _verify_completed_rank_stage(
    *,
    stage: Path,
    posenet,
    preconditional_camera_pairs: np.ndarray,
    targets: np.ndarray,
    source_pair_ids: tuple[int, ...],
    basis_q: np.ndarray,
    basis_scales: np.ndarray,
    copy_control_d_pose: float,
    resume_state_key: str,
    wire_custody: Mapping[str, str],
) -> dict[str, Any] | None:
    receipt_path = stage / "receipt.json"
    if not receipt_path.exists():
        return None
    basis_path = stage / "population_basis.bin"
    chunk_path = stage / "coefficient_chunk.bin"
    coefficient_path = stage / "coefficients.npz"
    prediction_path = stage / "predictions.npz"
    if (
        not basis_path.is_file()
        or not chunk_path.is_file()
        or not coefficient_path.is_file()
        or not prediction_path.is_file()
    ):
        raise G95RunnerError("completed G95 rank stage is missing immutable artifacts")
    receipt = _read_mapping(receipt_path, label="completed G95 rank receipt")
    if receipt.get("resume_state_key") != resume_state_key:
        raise G95RunnerError("completed G95 rank receipt resume-state key differs")
    with np.load(coefficient_path, allow_pickle=False) as coefficient_archive:
        if set(coefficient_archive.files) != {
            "coefficients_q",
            "coefficient_scales",
            "resume_state_key",
        }:
            raise G95RunnerError("completed G95 coefficient checkpoint differs")
        _verify_resume_state_array(
            coefficient_archive["resume_state_key"],
            expected_resume_state_key=resume_state_key,
            label="completed G95 coefficient checkpoint",
        )
        preserved_coefficients_q = np.asarray(coefficient_archive["coefficients_q"])
        preserved_coefficient_scales = np.asarray(coefficient_archive["coefficient_scales"])
    with np.load(prediction_path, allow_pickle=False) as prediction_archive:
        if set(prediction_archive.files) != {"predictions", "resume_state_key"}:
            raise G95RunnerError("completed G95 prediction checkpoint differs")
        _verify_resume_state_array(
            prediction_archive["resume_state_key"],
            expected_resume_state_key=resume_state_key,
            label="completed G95 prediction checkpoint",
        )
        preserved_predictions = np.asarray(prediction_archive["predictions"])
    basis_object = basis_path.read_bytes()
    coefficient_chunk = chunk_path.read_bytes()
    parsed_basis = parse_population_pose_preimage_basis(
        basis_object,
        expected_object_sha256=str(receipt.get("population_basis_object_sha256")),
    )
    parsed_chunk = parse_population_pose_preimage_coefficient_chunk(
        coefficient_chunk,
        expected_object_sha256=str(receipt.get("coefficient_chunk_sha256")),
    )
    wire_set = PopulationPosePreimageChartWireSetV1(
        basis=parsed_basis,
        chunk=parsed_chunk,
    )
    if not np.array_equal(wire_set.basis_q, basis_q) or not np.array_equal(
        wire_set.basis_scales,
        basis_scales,
    ):
        raise G95RunnerError("completed G95 rank stage basis differs from checkpoint")
    (
        rebuilt_basis,
        rebuilt_chunk,
        _wire_set,
        result,
        predictions,
        metrics,
    ) = _packet_and_exact_replay(
        posenet=posenet,
        preconditional_camera_pairs=preconditional_camera_pairs,
        targets=targets,
        source_pair_ids=source_pair_ids,
        basis_q=basis_q,
        basis_scales=basis_scales,
        coefficients_q=wire_set.coefficients_q,
        coefficient_scales=wire_set.coefficient_scales,
        wire_custody=wire_custody,
    )
    treatment = receipt.get("treatment")
    if (
        rebuilt_basis != basis_object
        or rebuilt_chunk != coefficient_chunk
        or not isinstance(treatment, dict)
        or metrics["receiver_camera_sha256"] != treatment.get("receiver_camera_sha256")
        or metrics["prediction_sha256"] != receipt.get("prediction_sha256")
        or float(metrics["d_pose"]) != float(treatment.get("d_pose"))
    ):
        raise G95RunnerError("completed G95 rank stage failed exact replay verification")
    sensitivity = treatment.get("sensitivity")
    if not isinstance(sensitivity, list) or len(sensitivity) != wire_set.rank:
        raise G95RunnerError("completed G95 rank stage sensitivity surface differs")
    expected_treatment = _treatment_row(
        rank=wire_set.rank,
        wire_set=wire_set,
        result=result,
        metrics=metrics,
        copy_control_d_pose=copy_control_d_pose,
        sensitivity=sensitivity,
    )
    if expected_treatment != treatment:
        raise G95RunnerError("completed G95 rank treatment receipt is not self-consistent")
    if not np.array_equal(
        preserved_coefficients_q,
        wire_set.coefficients_q,
    ) or not np.array_equal(
        preserved_coefficient_scales,
        wire_set.coefficient_scales,
    ):
        raise G95RunnerError("completed G95 coefficient checkpoint differs")
    if not np.array_equal(preserved_predictions, predictions):
        raise G95RunnerError("completed G95 prediction checkpoint differs")
    return treatment


def _fit_rank_stage(
    *,
    run_root: Path,
    rank: int,
    posenet,
    preconditional_camera_pairs: np.ndarray,
    targets: np.ndarray,
    source_pair_ids: tuple[int, ...],
    config: Mapping[str, Any],
    copy_control_d_pose: float,
    resume_state_key: str,
    wire_custody: Mapping[str, str],
) -> dict[str, Any]:
    stage = run_root / RANK_STAGE_NAMES[rank]
    (
        basis_q,
        basis_scales,
        singular_values,
        costate_rows,
        basis_metadata,
    ) = _load_or_build_basis(
        stage=stage,
        posenet=posenet,
        exact_y1_uint8=preconditional_camera_pairs[:, 1],
        targets=targets,
        rank=rank,
        config=config,
        resume_state_key=resume_state_key,
    )
    completed = _verify_completed_rank_stage(
        stage=stage,
        posenet=posenet,
        preconditional_camera_pairs=preconditional_camera_pairs,
        targets=targets,
        source_pair_ids=source_pair_ids,
        basis_q=basis_q,
        basis_scales=basis_scales,
        copy_control_d_pose=copy_control_d_pose,
        resume_state_key=resume_state_key,
        wire_custody=wire_custody,
    )
    if completed is not None:
        return completed

    settings = config["coefficient_fit"]
    resumed = _resume_latest_iteration(
        stage=stage,
        posenet=posenet,
        preconditional_camera_pairs=preconditional_camera_pairs,
        targets=targets,
        source_pair_ids=source_pair_ids,
        basis_q=basis_q,
        basis_scales=basis_scales,
        resume_state_key=resume_state_key,
        wire_custody=wire_custody,
    )
    proposal_telemetry = _load_iteration_proposal_history(
        stage,
        resume_state_key=resume_state_key,
    )
    if resumed is None:
        current_iteration = 0
        damping = float(settings["initial_damping"])
        zero_coefficients = np.zeros((len(source_pair_ids), rank), dtype=np.float64)
        coefficients_q, coefficient_scales, coefficients = _quantize_coefficients(
            zero_coefficients,
            scale_floor=float(settings["coefficient_scale_floor"]),
        )
        (
            basis_object,
            coefficient_chunk,
            wire_set,
            result,
            predictions,
            metrics,
        ) = _packet_and_exact_replay(
            posenet=posenet,
            preconditional_camera_pairs=preconditional_camera_pairs,
            targets=targets,
            source_pair_ids=source_pair_ids,
            basis_q=basis_q,
            basis_scales=basis_scales,
            coefficients_q=coefficients_q,
            coefficient_scales=coefficient_scales,
            wire_custody=wire_custody,
        )
        _write_iteration_checkpoint(
            stage=stage,
            iteration=0,
            basis_object=basis_object,
            coefficient_chunk=coefficient_chunk,
            wire_set=wire_set,
            result=result,
            predictions=predictions,
            metrics=metrics,
            coefficients_dequantized=coefficients,
            damping=damping,
            proposal_telemetry=(),
            resume_state_key=resume_state_key,
        )
    else:
        (
            current_iteration,
            damping,
            coefficients,
            basis_object,
            coefficient_chunk,
            wire_set,
            result,
            predictions,
            metrics,
        ) = resumed

    maximum_iterations = int(settings["maximum_iterations"])
    for iteration in range(current_iteration + 1, maximum_iterations + 1):
        jacobians: list[np.ndarray] = []
        gradient_predictions: list[np.ndarray] = []
        for pair_local in range(len(source_pair_ids)):
            gradient_prediction, jacobian = _pose_and_coefficient_jacobian(
                posenet=posenet,
                exact_y1_uint8=preconditional_camera_pairs[pair_local, 1],
                basis_q=basis_q,
                basis_scales=basis_scales,
                coefficients=coefficients[pair_local],
            )
            gradient_predictions.append(gradient_prediction)
            jacobians.append(jacobian)
        gradient_d_pose = float(
            np.mean((np.stack(gradient_predictions, axis=0) - np.asarray(targets, dtype=np.float64)) ** 2)
        )
        iteration_proposals: list[dict[str, Any]] = []
        accepted = False
        for line_search_index in range(int(settings["line_search_steps"])):
            candidate_float = coefficients.copy()
            step_norms: list[float] = []
            maximum_eigenvalues: list[float] = []
            for pair_local, jacobian in enumerate(jacobians):
                delta, maximum_eigenvalue = _lm_minimum_norm_delta(
                    jacobian=jacobian,
                    residual=gradient_predictions[pair_local] - targets[pair_local],
                    damping=damping,
                )
                candidate_float[pair_local] += delta
                step_norms.append(float(np.linalg.norm(delta)))
                maximum_eigenvalues.append(maximum_eigenvalue)
            candidate_q, candidate_scales, candidate_dequantized = _quantize_coefficients(
                candidate_float,
                scale_floor=float(settings["coefficient_scale_floor"]),
            )
            (
                candidate_basis_object,
                candidate_coefficient_chunk,
                candidate_wire_set,
                candidate_result,
                candidate_predictions,
                candidate_metrics,
            ) = _packet_and_exact_replay(
                posenet=posenet,
                preconditional_camera_pairs=preconditional_camera_pairs,
                targets=targets,
                source_pair_ids=source_pair_ids,
                basis_q=basis_q,
                basis_scales=basis_scales,
                coefficients_q=candidate_q,
                coefficient_scales=candidate_scales,
                wire_custody=wire_custody,
            )
            proposal = {
                "iteration": iteration,
                "line_search_index": line_search_index,
                "damping": damping,
                "gradient_space_current_d_pose": gradient_d_pose,
                "gradient_space_step_l2_norms": step_norms,
                "gradient_space_maximum_eigenvalues": maximum_eigenvalues,
                "exact_receiver_replay_d_pose": float(candidate_metrics["d_pose"]),
                "exact_receiver_wire_set_sha256": candidate_wire_set.wire_set_sha256,
                "exact_receiver_camera_sha256": candidate_result.camera_sha256,
                "admitted": False,
                "authority_separation": ("GRADIENT_PROPOSAL_TELEMETRY_NOT_AUTHORITY_EXACT_NUMPY_REPLAY_IS_AUTHORITY"),
            }
            iteration_proposals.append(proposal)
            if float(candidate_metrics["d_pose"]) < float(metrics["d_pose"]) - float(
                settings["minimum_exact_improvement"]
            ):
                proposal["admitted"] = True
                basis_object = candidate_basis_object
                coefficient_chunk = candidate_coefficient_chunk
                wire_set = candidate_wire_set
                result = candidate_result
                predictions = candidate_predictions
                metrics = candidate_metrics
                coefficients = candidate_dequantized
                damping = max(
                    damping * float(settings["damping_decrease"]),
                    1e-12,
                )
                accepted = True
                break
            damping = min(
                damping * float(settings["damping_increase"]),
                1e12,
            )
        proposal_telemetry.extend(iteration_proposals)
        if not accepted:
            break
        if iteration % int(settings["checkpoint_every_iterations"]) == 0:
            _write_iteration_checkpoint(
                stage=stage,
                iteration=iteration,
                basis_object=basis_object,
                coefficient_chunk=coefficient_chunk,
                wire_set=wire_set,
                result=result,
                predictions=predictions,
                metrics=metrics,
                coefficients_dequantized=coefficients,
                damping=damping,
                proposal_telemetry=iteration_proposals,
                resume_state_key=resume_state_key,
            )
        current_iteration = iteration

    sensitivity = _direction_sensitivity_rows(
        posenet=posenet,
        preconditional_camera_pairs=preconditional_camera_pairs,
        targets=targets,
        source_pair_ids=source_pair_ids,
        wire_set=wire_set,
        final_predictions=predictions,
        final_d_pose=float(metrics["d_pose"]),
        singular_values=singular_values,
        wire_custody=wire_custody,
    )
    treatment = _treatment_row(
        rank=rank,
        wire_set=wire_set,
        result=result,
        metrics=metrics,
        copy_control_d_pose=copy_control_d_pose,
        sensitivity=sensitivity,
    )
    _write_once_or_equal(stage / "population_basis.bin", basis_object)
    _write_once_or_equal(stage / "coefficient_chunk.bin", coefficient_chunk)
    _write_npz_once(
        stage / "coefficients.npz",
        coefficients_q=wire_set.coefficients_q,
        coefficient_scales=wire_set.coefficient_scales,
        resume_state_key=_resume_state_array(resume_state_key),
    )
    _write_npz_once(
        stage / "predictions.npz",
        predictions=np.asarray(predictions, dtype=np.float64),
        resume_state_key=_resume_state_array(resume_state_key),
    )
    receipt = {
        "schema": "tac.g95_rank_treatment_stage.v1",
        "g94_product_member_sha256": EXPECTED_G94_PRODUCT_SHA256,
        "g94_conditioning_state_sha256": EXPECTED_CONDITIONING_SHA256,
        "resume_state_key": resume_state_key,
        "source_pair_ids": list(source_pair_ids),
        "rank": rank,
        "grid_height": 48,
        "grid_width": 64,
        "population_basis_object_sha256": wire_set.basis.object_sha256,
        "coefficient_chunk_sha256": wire_set.chunk.object_sha256,
        "wire_set_sha256": wire_set.wire_set_sha256,
        "basis_q_sha256": _sha256_array(basis_q),
        "basis_scales_sha256": _sha256_array(basis_scales),
        "coefficient_q_sha256": _sha256_array(wire_set.coefficients_q),
        "coefficient_scales_sha256": _sha256_array(wire_set.coefficient_scales),
        "prediction_sha256": _sha256_array(predictions),
        "receiver_camera_sha256": result.camera_sha256,
        "selected_singular_values": singular_values.tolist(),
        "basis_metadata": basis_metadata,
        "costate_rows": costate_rows,
        "gradient_space_proposal_telemetry": proposal_telemetry,
        "treatment": treatment,
        "checkpoint_kind": ("BYTE_CLOSE_LOADABLE_END_OF_RANK_STAGE_PACKET_BASIS_COEFFICIENTS_RECEIVER_HASHES"),
        "research_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    _write_json_once(stage / "receipt.json", receipt)
    return treatment


def _checkpoint_manifest(run_root: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("stage_0[0-5]*/**/*")):
        if path.is_file():
            manifest.append(
                {
                    "path": str(path.relative_to(run_root)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return manifest


def _final_measurement_receipt(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    run_root: Path,
    preflight: Mapping[str, Any],
    g94_metadata: Mapping[str, Any],
    target_custody: Mapping[str, Any],
    yuv6_equivalence: Mapping[str, Any],
    pass_control: Mapping[str, Any],
    copy_control: Mapping[str, Any],
    treatments: Sequence[Mapping[str, Any]],
    resume_state_key: str,
) -> dict[str, Any]:
    if not treatments:
        raise G95RunnerError("final G95 receipt requires at least the rank-6 exact treatment")
    final_treatment = treatments[-1]
    crossed = any(bool(row["sufficient_reachability_coordinate_crossed"]) for row in treatments)
    richer_request = None
    if not crossed:
        if int(final_treatment["rank"]) != 24:
            raise G95RunnerError("G95 may emit a richer request only after exact rank-24 miss")
        requested = config["richer_control_request"]
        richer_request = richer_control_request_for_miss(
            g94_product_member_sha256=EXPECTED_G94_PRODUCT_SHA256,
            g94_conditioning_state_sha256=EXPECTED_CONDITIONING_SHA256,
            source_pair_ids=tuple(int(value) for value in final_treatment["source_pair_ids"])
            if "source_pair_ids" in final_treatment
            else tuple(
                range(
                    int(config["pair_start"]),
                    int(config["pair_start"]) + int(config["pair_count"]),
                )
            ),
            attempted_rank=24,
            attempted_grid_height=48,
            attempted_grid_width=64,
            exact_d_pose=float(final_treatment["d_pose"]),
            reachability_threshold=REACHABILITY_THRESHOLD,
            requested_minimum_rank=int(requested["requested_minimum_rank"]),
            requested_minimum_grid_height=int(requested["requested_minimum_grid_height"]),
            requested_minimum_grid_width=int(requested["requested_minimum_grid_width"]),
        ).to_dict()
    exact_rows = [dict(pass_control), dict(copy_control), *(dict(row) for row in treatments)]
    checkpoint_manifest = _checkpoint_manifest(run_root)
    sensitivity = [
        {
            "treatment_id": row["treatment_id"],
            "rank": row["rank"],
            "directions": row["sensitivity"],
        }
        for row in treatments
    ]
    blockers = [
        MISSING_INTEGRATION,
        OUTER_ZIP_BLOCKER,
        PUBLIC_INFLATE_BLOCKER,
        FULL_N600_BLOCKER,
        UPSTREAM_EVAL_BLOCKER,
        G83_BLOCKER,
    ]
    receipt = {
        "schema": MEASUREMENT_SCHEMA,
        "lane": "lane_g95_population_pose_inverse_control_20260727",
        "config": {
            "path": str(config_path.resolve(strict=True)),
            "bytes": config_path.stat().st_size,
            "sha256": _sha256_file(config_path),
        },
        "run_root": str(run_root.resolve(strict=True)),
        "git_head_observed_in_shared_dirty_tree": preflight["git_head_observed_in_shared_dirty_tree"],
        "dirty_tree_qualifier": bool(preflight["dirty_tree_qualifier"]),
        "g94_exact_state": dict(g94_metadata),
        "g94_parent_commit": EXPECTED_G94_PARENT_COMMIT,
        "g94_product_member_sha256": EXPECTED_G94_PRODUCT_SHA256,
        "g94_conditioning_state_sha256": EXPECTED_CONDITIONING_SHA256,
        "resume_state_key": resume_state_key,
        "non_final_semantic_y1": True,
        "source_target_custody": dict(target_custody),
        "posenet_custody": preflight["input_identities_hashed_before_scorer_load"]["posenet_weights"],
        "source_video_custody": preflight["input_identities_hashed_before_scorer_load"]["source_video"],
        "seed": int(config["seed"]),
        "torch_num_threads": int(config["torch_num_threads"]),
        "hardware_axis": str(config["hardware_axis"]),
        "source_pair_ids": list(
            range(
                int(config["pair_start"]),
                int(config["pair_start"]) + int(config["pair_count"]),
            )
        ),
        "scorer_batch_pairs": int(config["scorer_batch_pairs"]),
        "differentiable_yuv6_forward_equivalence": dict(yuv6_equivalence),
        "receiver_numeric_contract": {
            "bilinear_reference_id": BILINEAR_REFERENCE_ID,
            "rounding_policy_id": ROUNDING_POLICY_ID,
            "receiver_imports_scorer": False,
            "learned_values_in_packet_only": True,
        },
        "exact_replay_rows": exact_rows,
        "sufficient_reachability_coordinate_d_pose": REACHABILITY_THRESHOLD,
        "sufficient_reachability_coordinate_scope": REACHABILITY_COORDINATE_SCOPE,
        "sufficient_reachability_coordinate_crossed": crossed,
        "joint_feasible_surface": [dict(row["joint_feasible_surface"]) for row in treatments],
        "outer_zip_score_admission": OUTER_ZIP_SCORE_ADMISSION,
        "miss_verdict_scope": MISS_VERDICT_SCOPE,
        "population_transfer_assessment": {
            "one_state_miss_axis": ONE_STATE_MISS_AXIS,
            "failure_classification": POPULATION_TRANSFER_REQUEST,
            "request": POPULATION_TRANSFER_REQUEST,
            "pair0_population_viability_claim": False,
        },
        "richer_control_request": richer_request,
        "checkpoint_manifest": checkpoint_manifest,
        "hooks": {
            "sensitivity": sensitivity,
            "pareto": {
                "authority": "EXACT_BASIS_CHUNK_PARSE_NUMPY_RECEIVER_CPU_POSENET_REPLAY_ONLY",
                "rows": exact_rows,
                "gradient_proposals_excluded": True,
            },
            "bit_allocator": {
                "p_once_basis_object_bytes_by_rank": {
                    str(row["rank"]): int(row["population_basis_object_bytes_p_once"]) for row in treatments
                },
                "coefficient_chunk_bytes_by_rank": {
                    str(row["rank"]): int(row["coefficient_chunk_bytes"]) for row in treatments
                },
                "total_counted_wire_bytes_by_rank": {
                    str(row["rank"]): int(row["total_counted_wire_bytes"]) for row in treatments
                },
                "basis_bytes_by_rank": {str(row["rank"]): int(row["basis_bytes"]) for row in treatments},
                "coefficient_bytes_by_rank": {str(row["rank"]): int(row["coefficient_bytes"]) for row in treatments},
                "outer_zip_delta": None,
                "outer_zip_delta_blocker": OUTER_ZIP_BLOCKER,
            },
            "autopilot": {
                "next_action": (
                    MISSING_INTEGRATION if crossed else "EXECUTE_TYPED_RICHER_CONTROL_REQUEST_WITHOUT_FAMILY_KILL"
                ),
                "request": richer_request,
            },
            "continual_learning": {
                "append_only_evidence_key": (f"{EXPECTED_CONDITIONING_SHA256}:{FORMULATION_SCOPE}"),
                "conditioning_specific_no_marginal_transfer": True,
            },
            "dynamic_frontier": {
                "admitted": False,
                "g83_ready": False,
                "reason": ("bounded non-final semantic Y1 research state lacks public full-n600 same-archive closure"),
            },
        },
        "missing_g88_g94_typed_mode_and_outer_archive_race": MISSING_INTEGRATION,
        "blockers": blockers,
        "competitive_target_snapshot": _live_competitive_target_snapshot(),
        "competitive_pointer_delta": 0.0,
        "research_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    if (
        receipt["research_only"] is not True
        or receipt["candidate_claim"] is not False
        or receipt["score_claim"] is not False
        or receipt["promotion_eligible"] is not False
        or receipt["pointer_moved"] is not False
        or receipt["competitive_pointer_delta"] != 0.0
        or receipt["g94_conditioning_state_sha256"] != EXPECTED_CONDITIONING_SHA256
    ):
        raise G95RunnerError("final G95 receipt internal truth boundary differs")
    return receipt


def run(
    *,
    config_path: Path,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _load_config(config_path)
    configured_root = Path(str(config["output_root"])).resolve(strict=False)
    if resume_from is not None:
        resumed_root = resume_from.resolve(strict=True)
        if resumed_root != configured_root:
            raise G95RunnerError("--resume-from must equal the config output_root")
        run_root = resumed_root
    else:
        run_root = configured_root
    preflight = _storage_and_custody_preflight(
        config_path=config_path,
        config=config,
        run_root=run_root,
    )
    _configure_determinism(config)
    source_pair_ids = tuple(
        range(
            int(config["pair_start"]),
            int(config["pair_start"]) + int(config["pair_count"]),
        )
    )

    _build, preconditional, g94_metadata = _stage_exact_g94_state(
        config=config,
        run_root=run_root,
        source_pair_ids=source_pair_ids,
    )
    targets, target_custody = _load_pose_targets(
        config=config,
        source_pair_ids=source_pair_ids,
    )
    resume_sources = preflight["resume_state_sources_hashed_before_scorer_load"]
    resume_state_key = _resume_state_key(
        source_pair_ids=source_pair_ids,
        preconditional_camera_sha256=_sha256_array(preconditional),
        whole_preconditional_camera_sha256=str(g94_metadata["whole_preconditional_camera_sha256"]),
        selected_target_sha256=str(target_custody["selected_target_sha256"]),
        selected_target_table_sha256=str(target_custody["full_member_sha256"]),
        posenet_weights_sha256=str(
            preflight["input_identities_hashed_before_scorer_load"]["posenet_weights"]["sha256"]
        ),
        g94_product_member_sha256=str(g94_metadata["g94_product_member"]["sha256"]),
        g94_conditioning_state_sha256=str(g94_metadata["conditioning_state_sha256"]),
        config_sha256=str(resume_sources["config"]["sha256"]),
        receiver_module_sha256=str(resume_sources["receiver_module"]["sha256"]),
        measurement_tool_sha256=str(resume_sources["measurement_tool"]["sha256"]),
    )
    wire_custody = {
        "whole_preconditional_camera_sha256": str(g94_metadata["whole_preconditional_camera_sha256"]),
        "selected_target_table_sha256": str(target_custody["full_member_sha256"]),
        "selected_target_sha256": str(target_custody["selected_target_sha256"]),
        "posenet_weights_sha256": str(
            preflight["input_identities_hashed_before_scorer_load"]["posenet_weights"]["sha256"]
        ),
    }

    weights_path, _weights_identity = _exact_path_identity(
        config["posenet_weights"],
        label="posenet_weights",
    )
    posenet = _load_frozen_posenet(weights_path)
    yuv6_equivalence = _patch_and_check_differentiable_yuv6(
        posenet=posenet,
        exact_pairs_uint8=preconditional,
    )
    pass_control, copy_control = _stage_noop_controls(
        run_root=run_root,
        posenet=posenet,
        preconditional_camera_pairs=preconditional,
        targets=targets,
        source_pair_ids=source_pair_ids,
        yuv6_equivalence=yuv6_equivalence,
    )

    treatments: list[dict[str, Any]] = []
    for rank in config["rank_ladder"]:
        treatment = _fit_rank_stage(
            run_root=run_root,
            rank=int(rank),
            posenet=posenet,
            preconditional_camera_pairs=preconditional,
            targets=targets,
            source_pair_ids=source_pair_ids,
            config=config,
            copy_control_d_pose=float(copy_control["d_pose"]),
            resume_state_key=resume_state_key,
            wire_custody=wire_custody,
        )
        treatment["source_pair_ids"] = list(source_pair_ids)
        treatments.append(treatment)

    receipt = _final_measurement_receipt(
        config_path=config_path,
        config=config,
        run_root=run_root,
        preflight=preflight,
        g94_metadata=g94_metadata,
        target_custody=target_custody,
        yuv6_equivalence=yuv6_equivalence,
        pass_control=pass_control,
        copy_control=copy_control,
        treatments=treatments,
        resume_state_key=resume_state_key,
    )
    stage_receipt = run_root / "stage_06_receipt/receipt.json"
    final_receipt = Path(str(config["final_receipt_path"]))
    _write_json_once(stage_receipt, receipt)
    _write_json_once(final_receipt, receipt)
    return receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the research-only G95 shared Pose preimage chart and admit only "
            "exact packet/NumPy-receiver/CPU-PoseNet replay rows."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="frozen G95 JSON config",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="existing immutable G95 run root; must equal config output_root",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    receipt = run(
        config_path=args.config,
        resume_from=args.resume_from,
    )
    print(_canonical_json(receipt).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
