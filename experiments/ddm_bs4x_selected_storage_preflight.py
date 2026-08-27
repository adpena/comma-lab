#!/usr/bin/env python3
"""Refine BS4's Stage-2 retention floor on the sealed selected DX2 codes.

Stage 0 deliberately used endpoint-safe universal candidate-count minima.  This
gate loads the pinned DX2 carrier without a scorer, projects the sealed random
n32 selection, and proves whether the full QS5 cube and first descent pass are
mandatory for this exact object.  Every decoded array is retained before the
gate writes its additive refusal/ready checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_bs4_born_small_stage0_preflight as stage0
from experiments import ddm_po1_t4_error_feedback_pose_compensation as po1
from experiments import ddm_qs1_frame0_schur_coupled_solve as qs1
from experiments import ddm_rj2_joint_renderer_object_change as rj2

OUTPUT: Final = Path("/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved")
SELECTION: Final = OUTPUT / "retained/selection/random_pair_ids.int32.npy"
RETAINED: Final = OUTPUT / "retained/stage_15_selected_storage_preflight"
CHECKPOINT: Final = OUTPUT / "checkpoints/stage_15_selected_storage_preflight.json"
AXIS: Final = (
    "[macOS-CPU advisory, seeded uniform random n=32 from n600] NON-PROMOTABLE"
)
SELECTION_BYTES: Final = 256
SELECTION_SHA256: Final = (
    "1d088e908e74de605128083bff80949ae7574f50f7f495be8a625e0cfc2a9a1f"
)
PAIR_COUNT: Final = 32
DIMENSIONS: Final = 12
CAMERA_BYTES: Final = rj2.receiver.CAMERA_H * rj2.receiver.CAMERA_W * 3
BYTES_PER_CANDIDATE: Final = 3 * CAMERA_BYTES
RESERVE_BYTES: Final = 8 * 1024**3


class BS4XStorageError(RuntimeError):
    """The selected-object gate or its retained payload identity failed."""


def atomic_npy_once(path: Path, value: np.ndarray) -> dict[str, Any]:
    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    payload = stream.getvalue()
    expected = {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        observed = stage0.file_fact(path)
        comparable = {key: observed[key] for key in ("path", "bytes", "sha256")}
        if comparable != expected:
            raise BS4XStorageError(f"refusing to replace different retained array: {path}")
        return expected
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return expected


def selected_storage_floor(codes: np.ndarray) -> dict[str, Any]:
    values = np.asarray(codes, dtype=np.int32)
    if values.shape != (PAIR_COUNT, DIMENSIONS):
        raise BS4XStorageError(f"selected DX2 code geometry differs: {values.shape}")
    if np.any(values < -2048) or np.any(values > 2047):
        raise BS4XStorageError("selected DX2 codes exceed the signed-int12 lattice")

    endpoint_margin = np.minimum(values + 2048, 2047 - values)
    # QS5 clips the proposed coordinate update at 32; the integer cube adds 2;
    # the mandatory first strict-descent pass adds 1 more.
    full_surface_margin = int(qs1.MAX_CODE_STEP) + qs1.NEIGHBOUR_RADIUS + 1
    full_rows = np.all(endpoint_margin >= full_surface_margin, axis=1)
    if not np.all(full_rows):
        raise BS4XStorageError(
            "selected object does not support the all-full-row lower-bound proof"
        )

    central_candidates = 1 + 2 * DIMENSIONS
    integer_cube_candidates = (2 * qs1.NEIGHBOUR_RADIUS + 1) ** qs1.NEIGHBOUR_DIMS
    first_descent_candidates = 1 + 2 * DIMENSIONS
    candidates_per_pair = (
        1 + 1 + central_candidates + integer_cube_candidates + first_descent_candidates
    )
    payload_bytes = PAIR_COUNT * candidates_per_pair * BYTES_PER_CANDIDATE
    return {
        "schema": "ddm_bs4x_selected_stage2_storage_floor.v1",
        "selected_pairs": PAIR_COUNT,
        "dimensions": DIMENSIONS,
        "selected_code_min": int(values.min()),
        "selected_code_max": int(values.max()),
        "minimum_endpoint_margin": int(endpoint_margin.min()),
        "required_full_surface_margin": full_surface_margin,
        "full_surface_rows": int(np.count_nonzero(full_rows)),
        "candidate_count_terms": {
            "baseline": 1,
            "edited_event": 1,
            "central_difference": central_candidates,
            "integer_cube_lower_bound": integer_cube_candidates,
            "first_strict_descent_pass": first_descent_candidates,
        },
        "minimum_candidate_evaluations_per_pair": candidates_per_pair,
        "bytes_per_candidate_lower_bound": BYTES_PER_CANDIDATE,
        "minimum_materialized_payload_bytes": payload_bytes,
        "reserve_bytes": RESERVE_BYTES,
        "required_free_bytes": payload_bytes + RESERVE_BYTES,
        "excluded_from_lower_bound": [
            "codes and pose vectors",
            "born-small masters and semantic fingerprints",
            "JSON/checkpoint framing",
            "all improving descent passes after the first mandatory pass",
            "resolved carrier/container payloads",
            "three-way SegNet/PoseNet measurement payloads",
        ],
    }


def latest_ready_stage0() -> dict[str, Any]:
    candidates = sorted(
        (OUTPUT / "checkpoints").glob("stage_00_source_preflight*.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in candidates:
        value = json.loads(path.read_text())
        if (
            value.get("schema") == "ddm_bs4_stage_00_source_preflight.v1"
            and value.get("status") == "READY_FOR_STAGE_1"
            and value.get("identity_controls", {}).get("passed") is True
            and value.get("dx2_runtime_pin_consistency", {}).get("ok") is True
            and value.get("harness_controls", {}).get("scorer_slot_free") is True
        ):
            return {"value": value, "record": stage0.file_fact(path)}
    raise BS4XStorageError("no retained identity-clean Stage-0 checkpoint exists")


def run(output: Path = OUTPUT) -> dict[str, Any]:
    if output.resolve() != OUTPUT.resolve():
        raise BS4XStorageError("selected-object gate may write only the BS4 AP root")

    stage0_receipt = latest_ready_stage0()
    selection_control = stage0.checked_file(
        "sealed_random_n32_selection", SELECTION, SELECTION_BYTES, SELECTION_SHA256
    )
    if not selection_control["passed"]:
        raise BS4XStorageError(f"sealed selection pin differs: {selection_control['problems']}")
    selection = np.load(SELECTION, allow_pickle=False)
    if (
        selection.dtype != np.int32
        or selection.shape != (PAIR_COUNT,)
        or np.any(np.diff(selection) <= 0)
    ):
        raise BS4XStorageError("sealed selection dtype, shape, or ordering differs")

    _parts, state = po1.load_carrier(rj2.DX2_ARCHIVE, rj2.DX2_RUNTIME)
    all_codes = np.asarray(state.codes, dtype=np.int16)
    selector_choices = np.asarray(state.selector_choices, dtype=np.uint8)
    if all_codes.shape != (600, DIMENSIONS) or selector_choices.shape != (600,):
        raise BS4XStorageError("decoded DX2 carrier geometry differs")
    selected_codes = all_codes[selection].copy()
    selected_choices = selector_choices[selection].copy()

    retained = {
        "all_dx2_codes": atomic_npy_once(RETAINED / "all_dx2_codes.int16.npy", all_codes),
        "all_selector_choices": atomic_npy_once(
            RETAINED / "all_selector_choices.uint8.npy", selector_choices
        ),
        "selected_dx2_codes": atomic_npy_once(
            RETAINED / "selected_dx2_codes.int16.npy", selected_codes
        ),
        "selected_selector_choices": atomic_npy_once(
            RETAINED / "selected_selector_choices.uint8.npy", selected_choices
        ),
    }
    floor = selected_storage_floor(selected_codes)
    usage = shutil.disk_usage(output)
    passed = usage.free >= int(floor["required_free_bytes"])
    status = "READY_FOR_STAGE_2" if passed else "REFUSED_SELECTED_OBJECT_STORAGE_PREFLIGHT"
    result = {
        "schema": "ddm_bs4x_selected_stage2_storage_preflight.v1",
        "stage": "1.5",
        "status": status,
        "axis": AXIS,
        "verdict_scope": "INSTANCE: sealed BS4 random-n32 DX2/QS5 retained-payload fire object",
        "blockers": []
        if passed
        else [
            "The charter-mandated APDataStore root cannot retain the selected object's mandatory QS5 payload floor plus reserve.",
            "Stages 1-4 were not started; no scorer was loaded and no retained custody was deleted or moved.",
        ],
        "stage0_receipt": stage0_receipt["record"],
        "selection_control": selection_control,
        "sources": {
            "dx2_archive": stage0.file_fact(rj2.DX2_ARCHIVE),
            "dx2_runtime": str(rj2.DX2_RUNTIME.resolve()),
            "joint_solver": stage0.file_fact(qs1.JOINT_SOLVER_SOURCE),
            "qs1_reference": stage0.file_fact(Path(qs1.__file__).resolve()),
        },
        "retained_payloads": retained,
        "selected_selector_choice_counts": {
            str(int(choice)): int(count)
            for choice, count in zip(*np.unique(selected_choices, return_counts=True), strict=True)
        },
        "storage_floor": floor,
        "storage_observation": {
            "tier": str(output.resolve()),
            "free_bytes": usage.free,
            "required_free_bytes": floor["required_free_bytes"],
            "shortfall_bytes": max(0, int(floor["required_free_bytes"]) - usage.free),
            "passed": passed,
            "cleanup_policy": "certify-or-block; do not delete or move BS3 custody; no local fallback",
        },
        "stage_1_through_4_fired": False,
        "stage_5_fired": False,
        "scorer_forwards": 0,
        "segnet_forwards": 0,
        "posenet_forwards": 0,
        "all_materialized_payloads_retained": True,
        "score_claim": False,
        "promotion_eligible": False,
        "provenance": {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "python": sys.version,
            "platform": platform.platform(),
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip(),
            "runner": stage0.file_fact(Path(__file__).resolve()),
        },
    }
    checkpoint_path = stage0.additive_checkpoint_path(CHECKPOINT, result)
    result["checkpoint"] = stage0.atomic_json_once(checkpoint_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> int:
    result = run(parse_args().output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "READY_FOR_STAGE_2" else 2


if __name__ == "__main__":
    raise SystemExit(main())
