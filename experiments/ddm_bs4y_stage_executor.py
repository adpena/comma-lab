#!/usr/bin/env python3
"""Execute the sealed BS3 FIRE_ORDER stages 1-4 on the exact born-small object.

Stage 0 (``ddm_bs4_born_small_stage0_preflight``) and the selected-object storage
gate (``ddm_bs4x_selected_storage_preflight``) already exist; this runner is the
missing stages 1--4 executor.  It consumes the sealed order verbatim:

* Stage 1 retains, for every sealed pair, the born-small frame-1 master produced
  by the exact BO2 receiver and proves its semantic field is BODY_RESULT's.
* Stage 2 decodes the exact DX2 600x12 signed-int12 carrier and runs the QS5
  reference solve against that born-small master: central-difference 6x12
  PoseNet Jacobian, damped least squares, radius-2 integer neighbourhood, and
  exact coordinate descent to one full non-improving pass.
* Stage 3 replays RJ2's identity-controlled CPR1->CAP1->DX2->RR5->Brotli q9/w16
  production chain and replaces only the carrier section.
* Stage 4 measures realized ``d_seg``/``d_pose`` for the DX2 base, the
  born-small object on the stale carrier, and the born-small object on the fresh
  solve, on the same sealed pairs and one instrument.

Stage 5 is a conditional gate and is never fired here.

Every scored row is ``[macOS-CPU advisory, seeded uniform random n=32 from
n600] NON-PROMOTABLE``.  Nothing here is a score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_bs4_born_small_stage0_preflight as stage0
from experiments import ddm_po1_t4_error_feedback_pose_compensation as po1
from experiments import ddm_qs1_frame0_schur_coupled_solve as qs1
from experiments import ddm_rj1_renderer_joint_move as rj1
from experiments import ddm_rj2_joint_renderer_object_change as rj2

OUTPUT: Final = Path("/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved")
RETAINED: Final = OUTPUT / "retained/bs4y"
CHECKPOINTS: Final = OUTPUT / "checkpoints"
SELECTION: Final = OUTPUT / "retained/selection/random_pair_ids.int32.npy"
SELECTION_BYTES: Final = 256
SELECTION_SHA256: Final = "1d088e908e74de605128083bff80949ae7574f50f7f495be8a625e0cfc2a9a1f"

BO2_RAW: Final = Path(
    "/Volumes/APDataStore/pact/ddm_bo2_born_small_distortion/"
    "rows/hg1_generator_field/work/inflated/0.raw"
)
BO2_RAW_BYTES: Final = 3_662_409_600
BO2_RAW_SHA256: Final = "43c359eadd7c6e263adf7a1e2732a2b34948b1db8681bcc1be8f7c493b2ac841"
BO2_TOKENS: Final = BO2_RAW.parent / ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
BO2_TOKENS_SHA256: Final = "2884c5701dc2b2059df0e9f8e4ee3ed81809457b127a48ad3fd3fb6f7a17152b"
BO2_TOKENS_BYTES: Final = 117_964_800

BODY_RESULT: Final = OUTPUT / "BODY_RESULT.json"
FIRE_ORDER: Final = OUTPUT / "FIRE_ORDER.json"
FIRE_ORDER_SHA256: Final = "d684c9bc859f825e5d5341c822dcd8c989f91d3a8e7aef1a44316ced3b333db5"

PAIR_COUNT: Final = 600
SAMPLE_PAIRS: Final = 32
DIMENSIONS: Final = 12
POSE_DIMENSIONS: Final = 6
CAMERA_H: Final = 874
CAMERA_W: Final = 1164
RATE_DENOMINATOR: Final = 37_545_489
GB1_ARCHIVE_BYTES: Final = 180_215
GB1_SCORE: Final = 0.14811799921260607

BYTES_PER_CANDIDATE: Final = 3 * CAMERA_H * CAMERA_W * 3
DESCENT_CANDIDATES: Final = 1 + 2 * DIMENSIONS
HARD_FREE_FLOOR_BYTES: Final = 4 * 1024**3

AXIS: Final = "[macOS-CPU advisory, seeded uniform random n=32 from n600] NON-PROMOTABLE"


class BS4YError(RuntimeError):
    """A sealed pin, retained payload, solve binding, or storage guard failed."""


# --------------------------------------------------------------------------- #
# retention primitives (additive, never clobbering)
# --------------------------------------------------------------------------- #


def file_fact(path: Path) -> dict[str, Any]:
    observed = stage0.file_fact(path)
    if not observed.get("present"):
        raise BS4YError(f"required file is absent: {path}")
    return {key: observed[key] for key in ("path", "bytes", "sha256")}


def atomic_bytes_once(path: Path, payload: bytes) -> dict[str, Any]:
    expected = {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if file_fact(path) != expected:
            raise BS4YError(f"refusing to replace different retained bytes: {path}")
        return expected
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return expected


def retain_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
    return atomic_bytes_once(path, stream.getvalue())


def retain_json(path: Path, value: Any) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return atomic_bytes_once(path, payload)


def finite_or_none(value: float) -> float | None:
    """Retained JSON is ``allow_nan=False``; a singular Jacobian yields inf."""
    number = float(value)
    return number if math.isfinite(number) else None


def require_free_bytes(need: int, label: str) -> dict[str, Any]:
    """Fail closed before materializing a payload the mandated root cannot hold."""
    free = shutil.disk_usage(OUTPUT).free
    required = int(need) + HARD_FREE_FLOOR_BYTES
    row = {
        "label": label,
        "tier": str(OUTPUT.resolve()),
        "projected_payload_bytes": int(need),
        "hard_free_floor_bytes": HARD_FREE_FLOOR_BYTES,
        "required_free_bytes": required,
        "free_bytes": free,
        "passed": free >= required,
    }
    if not row["passed"]:
        raise BS4YError(
            f"storage waterfall refuses {label}: free {free} < required {required}"
        )
    return row


# --------------------------------------------------------------------------- #
# in-compile compensation binding (the qs4 stale-compensation cure)
# --------------------------------------------------------------------------- #


def compensation_object_fingerprint(
    *,
    pair: int,
    semantic_field: dict[str, Any],
    master_camera: dict[str, Any],
    carrier_archive_sha256: str,
) -> str:
    """Bind one frame-0 carrier solve to one exact frame-1 object.

    Paths are excluded so the binding follows bytes across retained copies.  The
    carrier archive SHA-256 is the DX2 object's, re-derived on this object; no
    other object's archive identity may appear here.
    """
    if not 0 <= int(pair) < PAIR_COUNT:
        raise BS4YError("compensation pair is outside the n600 domain")
    payload = {
        "schema": "ddm_bs4y_compensation_object_fingerprint.v1",
        "pair": int(pair),
        "semantic_field": {
            "bytes": int(semantic_field["bytes"]),
            "sha256": str(semantic_field["sha256"]),
        },
        "master_camera": {
            "bytes": int(master_camera["bytes"]),
            "sha256": str(master_camera["sha256"]),
        },
        "carrier_archive_sha256": str(carrier_archive_sha256),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def assert_compensation_matches_compile_object(
    row: dict[str, Any],
    *,
    semantic_field: dict[str, Any],
    carrier_archive_sha256: str,
) -> dict[str, Any]:
    """Refuse a solve that was not derived on the object being compiled NOW.

    This re-derives the fingerprint from the master bytes that are on disk at
    compile time.  A binding is mandatory: there is no legacy escape branch, and
    a stale ``final_codes`` carried across a frame-1 object change is a compile
    error rather than a silent regression (the qs4 +2.4e-4 pose disaster).
    """
    pair = int(row["pair"])
    binding = row.get("compensation_object")
    if not isinstance(binding, dict):
        raise BS4YError(f"pair {pair} solve carries no compensation-object binding")
    if binding.get("schema") != "ddm_bs4y_compensation_object_binding.v1":
        raise BS4YError(f"pair {pair} compensation binding schema differs")
    if int(binding.get("pair", -1)) != pair:
        raise BS4YError(f"pair {pair} compensation binding pair differs")
    master_record = binding.get("master_camera")
    if not isinstance(master_record, dict):
        raise BS4YError(f"pair {pair} compensation binding has no master record")
    if file_fact(Path(str(master_record["path"]))) != master_record:
        raise BS4YError(f"pair {pair} bound master bytes differ at compile")
    if binding.get("semantic_field") != {
        "bytes": int(semantic_field["bytes"]),
        "sha256": str(semantic_field["sha256"]),
    }:
        raise BS4YError(f"pair {pair} bound semantic field differs at compile")
    expected = compensation_object_fingerprint(
        pair=pair,
        semantic_field=semantic_field,
        master_camera=master_record,
        carrier_archive_sha256=carrier_archive_sha256,
    )
    if binding.get("fingerprint_sha256") != expected:
        raise BS4YError(f"pair {pair} compensation fingerprint differs at compile")
    if row.get("solve", {}).get("compensation_object_fingerprint_sha256") != expected:
        raise BS4YError(f"pair {pair} frame-0 solve is stale for the compile object")
    return {
        "schema": "ddm_bs4y_compile_compensation_binding.v1",
        "pair": pair,
        "mode": "EXACT_OBJECT_BOUND_FRESH_SOLVE",
        "fingerprint_sha256": expected,
        "master_camera": master_record,
        "passed": True,
    }


# --------------------------------------------------------------------------- #
# the exact DX2 frame-0 carrier surface
# --------------------------------------------------------------------------- #


class DX2Surface:
    """The exact DX2 receiver's signed-int12 frame-0 actuator, batched.

    The per-candidate maths is byte-for-byte ``rj2.carrier_frame0``; batching
    only stacks independent rows, and ``verify_receiver`` proves row 0 equals the
    shipped ``0.raw`` frame-0 before any solve consumes the surface.
    """

    def __init__(self, state: Any, renderer_module: Any) -> None:
        import torch

        self.state = state
        self.renderer = renderer_module
        raw_basis = torch.from_numpy(
            state.basis_codes.reshape(DIMENSIONS, 3, 24, 32).astype(np.float32)
            * state.basis_scales[:, None, None, None]
        )
        self.basis = renderer_module.normalized_basis(raw_basis)
        self.coefficient_scales = torch.from_numpy(np.asarray(state.coefficient_scales))
        self.amplitude = float(renderer_module.CARRIER_AMPLITUDE)

    def render(self, codes: np.ndarray, pair: int) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        values = np.asarray(codes, dtype=np.int32)
        if values.ndim != 2 or values.shape[1] != DIMENSIONS:
            raise BS4YError(f"carrier code geometry differs: {values.shape}")
        if np.any(values < -2048) or np.any(values > 2047):
            raise BS4YError("carrier candidate exceeds the signed-int12 lattice")
        if int(self.state.selector_choices[pair]) != 0:
            raise BS4YError(
                f"pair {pair} has a non-identity F0E1 selector; the carrier "
                "derivative the receiver honours is undefined here"
            )
        with torch.inference_mode():
            coefficients = (
                torch.from_numpy(values.astype(np.float32)) * self.coefficient_scales[None]
            )
            carrier = torch.einsum("bk,kchw->bchw", coefficients, self.basis)
            carrier = carrier / math.sqrt(DIMENSIONS)
            low = (127.5 + self.amplitude * carrier).clamp(0.0, 255.0).round()
            high = functional.interpolate(
                low, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
            ).clamp(0.0, 255.0).round()
            return high.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    def verify_receiver(self, pair: int, shipped_frame0: np.ndarray) -> None:
        rendered = self.render(self.state.codes[pair][None].astype(np.int32), pair)[0]
        if not np.array_equal(rendered, np.asarray(shipped_frame0)):
            raise BS4YError(
                f"pair {pair} carrier surface does not reproduce the shipped frame-0"
            )


def pose_vectors(posenet: Any, pairs: np.ndarray) -> np.ndarray:
    import torch

    value = np.asarray(pairs)
    if value.ndim != 5 or value.shape[1:] != (2, CAMERA_H, CAMERA_W, 3):
        raise BS4YError(f"PoseNet input geometry differs: {value.shape}")
    with torch.inference_mode():
        tensor = torch.from_numpy(np.ascontiguousarray(value)).permute(0, 1, 4, 2, 3).float()
        output = posenet(posenet.preprocess_input(tensor))["pose"][..., :POSE_DIMENSIONS]
    return output.cpu().numpy().astype(np.float32, copy=False)


def evaluate_codes(
    *,
    surface: DX2Surface,
    posenet: Any,
    codes: Sequence[np.ndarray],
    master: np.ndarray,
    master_record: dict[str, Any],
    pair: int,
    stage_root: Path,
) -> np.ndarray:
    """Render and score candidates in retained, resumable, object-bound batches.

    Every batch stamps the master's SHA-256 and its own codes.  On resume both
    are verified, so a retained batch can never be replayed against a different
    frame-1 object -- the master-blind cache is the fourth route by which a stale
    pose number reaches a verdict.
    """
    code_array = np.stack([np.asarray(item, dtype=np.int32) for item in codes])
    master_sha256 = str(master_record["sha256"])
    all_vectors: list[np.ndarray] = []
    for first in range(0, len(code_array), qs1.POSE_BATCH):
        last = min(first + qs1.POSE_BATCH, len(code_array))
        batch_root = stage_root / f"batch_{first:04d}_{last:04d}"
        result_path = batch_root / "RESULT.json"
        batch_codes = code_array[first:last]
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            if result.get("master_sha256") != master_sha256:
                raise BS4YError(
                    f"retained batch {batch_root} is bound to a different frame-1 object"
                )
            if int(result.get("pair", -1)) != pair:
                raise BS4YError(f"retained batch {batch_root} is bound to another pair")
            retained_codes = np.load(
                Path(result["codes"]["path"]), allow_pickle=False
            )
            if not np.array_equal(retained_codes, batch_codes):
                raise BS4YError(f"retained batch {batch_root} holds different codes")
            vectors_path = Path(result["pose_vectors"]["path"])
            if file_fact(vectors_path) != result["pose_vectors"]:
                raise BS4YError(f"resumed pose vectors differ: {vectors_path}")
            all_vectors.append(np.load(vectors_path, allow_pickle=False))
            continue
        require_free_bytes(len(batch_codes) * BYTES_PER_CANDIDATE, str(batch_root))
        slaves = surface.render(batch_codes, pair)
        masters = np.repeat(np.asarray(master)[None], len(batch_codes), axis=0)
        inputs = np.stack((slaves, masters), axis=1)
        codes_record = retain_npy(batch_root / "codes.int32.npy", batch_codes)
        slave_record = retain_npy(batch_root / "slave_camera.uint8.npy", slaves)
        input_record = retain_npy(batch_root / "pose_input.uint8.npy", inputs)
        vectors = pose_vectors(posenet, inputs)
        vector_record = retain_npy(batch_root / "pose_vectors.float32.npy", vectors)
        retain_json(
            result_path,
            {
                "schema": "ddm_bs4y_pose_batch.v1",
                "pair": pair,
                "candidate_first": first,
                "candidate_last_exclusive": last,
                "master_sha256": master_sha256,
                "codes": codes_record,
                "slave_camera": slave_record,
                "pose_input": input_record,
                "pose_vectors": vector_record,
                "axis": AXIS,
                "score_claim": False,
                "promotion_eligible": False,
            },
        )
        all_vectors.append(vectors)
    vectors = np.concatenate(all_vectors, axis=0)
    if vectors.shape != (len(code_array), POSE_DIMENSIONS):
        raise BS4YError("retained PoseNet output census differs")
    retain_npy(stage_root / "ALL_CODES.int32.npy", code_array)
    retain_npy(stage_root / "ALL_POSE_VECTORS.float32.npy", vectors)
    return vectors


def guarded_strict_descent(
    current: np.ndarray,
    objective: float,
    evaluate: Callable[[tuple[np.ndarray, ...], int], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, float, int, np.ndarray]:
    """``qs1.strict_descent`` with a storage waterfall recheck before each pass.

    The retained-payload floor prices one mandatory pass.  Every further pass is
    real payload the mandated root may not be able to hold, so the guard runs at
    the pass boundary and fails closed instead of dying mid-write.
    """

    def guarded(candidates: tuple[np.ndarray, ...], pass_index: int) -> tuple[np.ndarray, np.ndarray]:
        require_free_bytes(
            len(candidates) * BYTES_PER_CANDIDATE, f"descent_pass_{pass_index:04d}"
        )
        return evaluate(candidates, pass_index)

    return qs1.strict_descent(current, objective, guarded)


# --------------------------------------------------------------------------- #
# Stage 1 -- exact born-small frame-1 masters
# --------------------------------------------------------------------------- #


def stage_10_masters(selection: np.ndarray) -> dict[str, Any]:
    checkpoint = CHECKPOINTS / "stage_10_exact_born_small_masters.json"
    body = json.loads(BODY_RESULT.read_text())
    semantic_field = {
        "bytes": BO2_TOKENS_BYTES,
        "sha256": BO2_TOKENS_SHA256,
    }
    tokens_control = stage0.checked_file(
        "bo2_decoded_semantic_field", BO2_TOKENS, BO2_TOKENS_BYTES, BO2_TOKENS_SHA256
    )
    raw_control = stage0.checked_file(
        "bo2_born_small_raw", BO2_RAW, BO2_RAW_BYTES, BO2_RAW_SHA256
    )
    if not tokens_control["passed"] or not raw_control["passed"]:
        raise BS4YError("BO2 receiver pins differ; refusing to build masters")

    body_field = body["retained_sources"]["generated_tokens"]
    field_matches_body = (
        str(body_field["sha256"]) == BO2_TOKENS_SHA256
        and int(body_field["bytes"]) == BO2_TOKENS_BYTES
    )
    if not field_matches_body:
        raise BS4YError(
            "BO2 receiver's decoded semantic field is not BODY_RESULT's born-small field"
        )

    require_free_bytes(
        3 * len(selection) * (CAMERA_H * CAMERA_W * 3), "stage_10_masters"
    )
    born = np.memmap(
        BO2_RAW, mode="r", dtype=np.uint8, shape=(2 * PAIR_COUNT, CAMERA_H, CAMERA_W, 3)
    )
    base = np.memmap(
        rj2.DX2_RAW, mode="r", dtype=np.uint8, shape=(2 * PAIR_COUNT, CAMERA_H, CAMERA_W, 3)
    )
    rows: list[dict[str, Any]] = []
    for pair in selection.tolist():
        pair = int(pair)
        born_master = np.asarray(born[2 * pair + 1]).copy()
        born_slave = np.asarray(born[2 * pair]).copy()
        base_master = np.asarray(base[2 * pair + 1]).copy()
        base_slave = np.asarray(base[2 * pair]).copy()
        root = RETAINED / f"stage_10/pair_{pair:04d}"
        rows.append(
            {
                "pair": pair,
                "born_small_master": retain_npy(root / "born_small_master.uint8.npy", born_master),
                "dx2_base_master": retain_npy(root / "dx2_base_master.uint8.npy", base_master),
                "dx2_base_slave": retain_npy(root / "dx2_base_slave.uint8.npy", base_slave),
                "born_small_slave_equals_dx2_base_slave": bool(
                    np.array_equal(born_slave, base_slave)
                ),
                "master_changed_from_dx2_base": bool(
                    not np.array_equal(born_master, base_master)
                ),
                "master_max_abs_delta_vs_dx2_base": int(
                    np.abs(born_master.astype(np.int16) - base_master.astype(np.int16)).max()
                ),
            }
        )
    del born, base

    carrier_only_rows = sum(1 for row in rows if row["born_small_slave_equals_dx2_base_slave"])
    result = {
        "schema": "ddm_bs4y_stage_10_exact_born_small_masters.v1",
        "stage": 1,
        "status": "RETAINED",
        "axis": AXIS,
        "selection_pairs": [int(value) for value in selection.tolist()],
        "semantic_field": semantic_field,
        "semantic_field_matches_body_result": field_matches_body,
        "body_result_generated_tokens": body_field,
        "controls": [tokens_control, raw_control],
        "rows": rows,
        "frame0_is_carrier_only_rows": carrier_only_rows,
        "frame0_is_carrier_only_all": carrier_only_rows == len(rows),
        "scorer_forwards": 0,
        "all_materialized_payloads_retained": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    result["checkpoint"] = stage0.atomic_json_once(
        stage0.additive_checkpoint_path(checkpoint, result), result
    )
    return result


# --------------------------------------------------------------------------- #
# Stage 2 -- the QS5 exact per-pair solve on the born-small object
# --------------------------------------------------------------------------- #


def solve_one_pair(
    *,
    pair: int,
    surface: DX2Surface,
    posenet: Any,
    solver: Any,
    master_row: dict[str, Any],
    semantic_field: dict[str, Any],
    carrier_archive_sha256: str,
) -> dict[str, Any]:
    root = RETAINED / f"stage_20/pair_{pair:04d}"
    completed = root / "SOLVE.json"
    if completed.is_file():
        prior = json.loads(completed.read_text())
        assert_compensation_matches_compile_object(
            prior, semantic_field=semantic_field, carrier_archive_sha256=carrier_archive_sha256
        )
        return prior

    base_codes = np.asarray(surface.state.codes[pair], dtype=np.int32)
    base_master_record = master_row["dx2_base_master"]
    born_master_record = master_row["born_small_master"]
    base_master = np.load(Path(base_master_record["path"]), allow_pickle=False)
    born_master = np.load(Path(born_master_record["path"]), allow_pickle=False)
    base_slave = np.load(Path(master_row["dx2_base_slave"]["path"]), allow_pickle=False)
    surface.verify_receiver(pair, base_slave)

    baseline = evaluate_codes(
        surface=surface, posenet=posenet, codes=(base_codes,), master=base_master,
        master_record=base_master_record, pair=pair, stage_root=root / "stage_10_baseline",
    )[0]
    event = evaluate_codes(
        surface=surface, posenet=posenet, codes=(base_codes,), master=born_master,
        master_record=born_master_record, pair=pair,
        stage_root=root / "stage_20_exact_object_leak",
    )[0]
    leak = event.astype(np.float64) - baseline.astype(np.float64)

    jacobian_codes = [base_codes.copy()]
    for dimension in range(DIMENSIONS):
        for delta in (-1, 1):
            candidate = base_codes.copy()
            candidate[dimension] += delta
            if not -2048 <= candidate[dimension] <= 2047:
                raise BS4YError(f"pair {pair} coefficient is at an int12 endpoint")
            jacobian_codes.append(candidate)
    jacobian_vectors = evaluate_codes(
        surface=surface, posenet=posenet, codes=tuple(jacobian_codes), master=born_master,
        master_record=born_master_record, pair=pair, stage_root=root / "stage_30_jacobian",
    )
    jacobian = np.empty((POSE_DIMENSIONS, DIMENSIONS), dtype=np.float64)
    for dimension in range(DIMENSIONS):
        minus = jacobian_vectors[1 + 2 * dimension].astype(np.float64)
        plus = jacobian_vectors[2 + 2 * dimension].astype(np.float64)
        jacobian[:, dimension] = (plus - minus) / 2.0
    retain_npy(root / "stage_30_jacobian/J_POSE0.float64.npy", jacobian)

    update = solver.solve_damped_least_squares(
        jacobian, -leak, damping=qs1.GN_DAMPING, max_code_step=qs1.MAX_CODE_STEP
    )
    centre = solver.quantize_int12_update(base_codes, update.update)
    active = solver.rank_neighbour_dimensions(jacobian, update.update, qs1.NEIGHBOUR_DIMS)
    neighbourhood = solver.nearby_int12_candidates(
        base_codes, centre, active_dimensions=active, radius=qs1.NEIGHBOUR_RADIUS
    )
    neighbourhood_vectors = evaluate_codes(
        surface=surface, posenet=posenet, codes=neighbourhood, master=born_master,
        master_record=born_master_record, pair=pair, stage_root=root / "stage_40_integer_cube",
    )
    objectives = np.mean(
        np.square(neighbourhood_vectors.astype(np.float64) - baseline[None].astype(np.float64)),
        axis=1,
    )
    best_index = min(range(len(neighbourhood)), key=lambda index: (float(objectives[index]), index))
    current_codes = np.asarray(neighbourhood[best_index], dtype=np.int32)
    current_objective = float(objectives[best_index])

    def evaluate_descent(
        candidates: tuple[np.ndarray, ...], pass_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        stage_root = root / f"stage_50_descent/pass_{pass_index:04d}"
        vectors = evaluate_codes(
            surface=surface, posenet=posenet, codes=candidates, master=born_master,
            master_record=born_master_record, pair=pair, stage_root=stage_root,
        )
        values = np.mean(
            np.square(vectors.astype(np.float64) - baseline[None].astype(np.float64)), axis=1
        )
        retain_npy(stage_root / "OBJECTIVES.float64.npy", values)
        return vectors, values

    final_codes, final_objective, passes, final_vector = guarded_strict_descent(
        current_codes, current_objective, evaluate_descent
    )
    residual = final_vector.astype(np.float64) - baseline.astype(np.float64)
    metrics = qs1.cancellation_metrics(leak, residual)
    fingerprint = compensation_object_fingerprint(
        pair=pair,
        semantic_field=semantic_field,
        master_camera=born_master_record,
        carrier_archive_sha256=carrier_archive_sha256,
    )
    row = {
        "schema": "ddm_bs4y_stage_20_pair_solve.v1",
        "pair": pair,
        "axis": AXIS,
        "compensation_object": {
            "schema": "ddm_bs4y_compensation_object_binding.v1",
            "pair": pair,
            "semantic_field": {
                "bytes": int(semantic_field["bytes"]),
                "sha256": str(semantic_field["sha256"]),
            },
            "master_camera": born_master_record,
            "carrier_archive_sha256": carrier_archive_sha256,
            "fingerprint_sha256": fingerprint,
            "master_taken_from_exact_bo2_receiver_output": True,
        },
        "solve": {
            "base_codes": base_codes.tolist(),
            "final_codes": np.asarray(final_codes, dtype=np.int32).tolist(),
            "final_code_delta": (
                np.asarray(final_codes, dtype=np.int32) - base_codes
            ).tolist(),
            "baseline_pose": baseline.astype(float).tolist(),
            "stale_carrier_pose": event.astype(float).tolist(),
            "final_pose": final_vector.astype(float).tolist(),
            "leak": leak.tolist(),
            "residual": residual.tolist(),
            "objective_stale_carrier": float(np.mean(np.square(leak))),
            "objective_final": float(final_objective),
            "descent_passes": int(passes),
            "damped_update": np.asarray(update.update, dtype=np.float64).tolist(),
            "damped_rank": int(update.rank),
            "damped_condition": finite_or_none(update.condition),
            "damped_ridge_lambda": finite_or_none(update.ridge_lambda),
            "active_dimensions": [int(value) for value in active],
            "neighbourhood_candidates": len(neighbourhood),
            "cancellation": metrics,
            "compensation_object_fingerprint_sha256": fingerprint,
        },
        "score_claim": False,
        "promotion_eligible": False,
    }
    assert_compensation_matches_compile_object(
        row, semantic_field=semantic_field, carrier_archive_sha256=carrier_archive_sha256
    )
    retain_json(completed, row)
    return row


def stage_20_solves(stage_10: dict[str, Any]) -> dict[str, Any]:
    checkpoint = CHECKPOINTS / "stage_20_qs5_exact_pair_solves.json"
    _parts, state = po1.load_carrier(rj2.DX2_ARCHIVE, rj2.DX2_RUNTIME)
    renderer_module = po1._load_renderer(rj2.DX2_RUNTIME)
    surface = DX2Surface(state, renderer_module)
    posenet = qs1.load_posenet()
    solver = qs1._load_module("ddm_bs4y_joint_pose_solve", qs1.JOINT_SOLVER_SOURCE)
    carrier_archive_sha256 = file_fact(rj2.DX2_ARCHIVE)["sha256"]
    semantic_field = stage_10["semantic_field"]
    by_pair = {int(row["pair"]): row for row in stage_10["rows"]}

    rows: list[dict[str, Any]] = []
    for pair in stage_10["selection_pairs"]:
        rows.append(
            solve_one_pair(
                pair=int(pair),
                surface=surface,
                posenet=posenet,
                solver=solver,
                master_row=by_pair[int(pair)],
                semantic_field=semantic_field,
                carrier_archive_sha256=carrier_archive_sha256,
            )
        )
    moved = sum(1 for row in rows if any(row["solve"]["final_code_delta"]))
    result = {
        "schema": "ddm_bs4y_stage_20_qs5_exact_pair_solves.v1",
        "stage": 2,
        "status": "SOLVED",
        "axis": AXIS,
        "carrier_archive": file_fact(rj2.DX2_ARCHIVE),
        "carrier_geometry": [int(value) for value in state.codes.shape],
        "joint_solver": file_fact(qs1.JOINT_SOLVER_SOURCE),
        "solver_form": (
            "central-difference 6x12 Jacobian (h=1), damped least squares "
            "(lambda=(0.01*sigma_max)^2, step clip 32), radius-2 integer cube on the "
            "top-3 ranked dimensions, exact coordinate descent to one full "
            "non-improving pass"
        ),
        "pairs_solved": len(rows),
        "pairs_with_moved_codes": moved,
        "rows": rows,
        "all_materialized_payloads_retained": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    result["checkpoint"] = stage0.atomic_json_once(
        stage0.additive_checkpoint_path(checkpoint, result), result
    )
    return result


# --------------------------------------------------------------------------- #
# Stage 3 -- RJ2 production carrier re-encode, carrier section only
# --------------------------------------------------------------------------- #


def stage_30_container(stage_20: dict[str, Any]) -> dict[str, Any]:
    checkpoint = CHECKPOINTS / "stage_30_resolved_carrier_container.json"
    root = RETAINED / "stage_30"
    parts, state = po1.load_carrier(rj2.DX2_ARCHIVE, rj2.DX2_RUNTIME)
    riders = rj2._runtime_riders(rj2.DX2_RUNTIME)
    source_cap1, _selector = riders.carrier_repack.split_frame0_selector_carrier(
        parts.carrier_blob
    )
    source_predictor_metadata = source_cap1[14:50]

    container = rj1.source_container()
    identity = rj2.encode_carrier_stream(
        state,
        state.codes.astype(np.int32),
        runtime=rj2.DX2_RUNTIME,
        retention_root=root / "identity_control",
        source_predictor_metadata=source_predictor_metadata,
    )
    identity_passed = identity["stream"] == container["carrier"]
    if not identity_passed:
        raise BS4YError("production carrier encoder identity control failed")

    final_codes = state.codes.astype(np.int32, copy=True)
    for row in stage_20["rows"]:
        final_codes[int(row["pair"])] = np.asarray(
            row["solve"]["final_codes"], dtype=np.int32
        )
    codes_record = retain_npy(root / "carrier_codes_final.int32.npy", final_codes)

    # The identity control above is fatal: it proves the encoder reproduces the
    # shipped object.  A refusal on the RESOLVED lattice is different in kind --
    # the SA2 metadata packer bounds Rice-k and predictor-factor spread, so a
    # moved lattice can legally exceed the shipped carrier's grammar.  That is a
    # measured property of this object, not a harness failure, and it must not
    # destroy Stage 4: realized d_seg/d_pose depend only on rendered frames.
    try:
        encoded = rj2.encode_carrier_stream(
            state,
            final_codes,
            runtime=rj2.DX2_RUNTIME,
            retention_root=root / "resolved_carrier",
            source_predictor_metadata=source_predictor_metadata,
        )
    except Exception as failure:
        result = {
            "schema": "ddm_bs4y_stage_30_resolved_carrier_container.v1",
            "stage": 3,
            "status": "REFUSED_RESOLVED_CARRIER_ENCODE",
            "axis": AXIS,
            "identity_control_passed": identity_passed,
            "carrier_codes_final": codes_record,
            "dx2_archive_bytes": len(container["archive"]),
            "resolved_archive_bytes": None,
            "refusal": {"type": type(failure).__name__, "message": str(failure)},
            "retained_identity_chain": identity["retained"],
            "all_materialized_payloads_retained": True,
            "score_claim": False,
            "promotion_eligible": False,
        }
        result["checkpoint"] = stage0.atomic_json_once(
            stage0.additive_checkpoint_path(checkpoint, result), result
        )
        return result
    # Carrier-only move: reuse the shipped semantic stream verbatim rather than
    # re-deriving it, so no semantic byte can drift into a carrier measurement.
    if max(
        len(container["hpac"]), len(container["semantic"]), len(encoded["stream"])
    ) > 0xFFFF:
        raise BS4YError("RX1 uint16 section ceiling exceeded by the resolved carrier")
    member = (
        rj1.RX1_HEADER.pack(
            container["magic"],
            container["version"],
            container["codec"],
            container["table_mode"],
            container["reserved"],
            len(container["hpac"]),
            len(container["semantic"]),
            len(encoded["stream"]),
        )
        + container["hpac"]
        + container["semantic"]
        + encoded["stream"]
        + container["tail"]
    )
    archive = rj1.deterministic_zip(member)
    repeat = rj1.deterministic_zip(member)
    if archive != repeat:
        raise BS4YError("primary and repeat resolved containers differ")
    archive_record = atomic_bytes_once(root / "archive.zip", archive)
    repeat_record = atomic_bytes_once(root / "archive.repeat.zip", repeat)

    parseback = rj2.fresh_process_parseback(
        runtime=rj2.DX2_RUNTIME,
        archive=Path(archive_record["path"]),
        expected_codes=Path(codes_record["path"]),
        semantic_sha256=hashlib.sha256(container["semantic"]).hexdigest(),
        output=root / "parseback_transcript.txt",
    )

    result = {
        "schema": "ddm_bs4y_stage_30_resolved_carrier_container.v1",
        "stage": 3,
        "status": "BYTE_CLOSED",
        "axis": AXIS,
        "identity_control_passed": identity_passed,
        "source_predictor_sha256": hashlib.sha256(source_predictor_metadata).hexdigest(),
        "carrier_codes_final": codes_record,
        "shipped_carrier_bytes": len(container["carrier"]),
        "resolved_carrier_bytes": len(encoded["stream"]),
        "carrier_delta_bytes": len(encoded["stream"]) - len(container["carrier"]),
        "dx2_archive_bytes": len(container["archive"]),
        "resolved_archive_bytes": len(archive),
        "archive_delta_bytes": len(archive) - len(container["archive"]),
        "archive": archive_record,
        "archive_repeat": repeat_record,
        "repeat_equal": archive_record["sha256"] == repeat_record["sha256"],
        "semantic_reused_verbatim": True,
        "retained_identity_chain": identity["retained"],
        "retained_resolved_chain": encoded["retained"],
        "rr5_basis_bits": encoded["rr5_basis_bits"],
        "dx2_cabac_bits": encoded["dx2_cabac_bits"],
        "receiver_parseback": parseback,
        "all_materialized_payloads_retained": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    result["checkpoint"] = stage0.atomic_json_once(
        stage0.additive_checkpoint_path(checkpoint, result), result
    )
    return result


# --------------------------------------------------------------------------- #
# Stage 4 -- the three-way realized measurement
# --------------------------------------------------------------------------- #


def stage_40_three_way(
    stage_10: dict[str, Any], stage_20: dict[str, Any], stage_30: dict[str, Any]
) -> dict[str, Any]:
    import torch

    from tac.scorer import load_differentiable_scorers

    checkpoint = CHECKPOINTS / "stage_40_three_way_measurement.json"
    root = RETAINED / "stage_40"
    _parts, state = po1.load_carrier(rj2.DX2_ARCHIVE, rj2.DX2_RUNTIME)
    renderer_module = po1._load_renderer(rj2.DX2_RUNTIME)
    surface = DX2Surface(state, renderer_module)
    posenet, segnet = load_differentiable_scorers(REPO / "upstream", device="cpu")
    posenet.eval()
    segnet.eval()

    solved_by_pair = {int(row["pair"]): row for row in stage_20["rows"]}
    masters_by_pair = {int(row["pair"]): row for row in stage_10["rows"]}
    legs = ("gb1_dx2_base", "born_small_stale_carrier", "born_small_fresh_solve")
    per_pair: list[dict[str, Any]] = []

    for pair in stage_10["selection_pairs"]:
        pair = int(pair)
        master_row = masters_by_pair[pair]
        solve_row = solved_by_pair[pair]
        subset = rj2.load_source_subset(pair)
        original_argmax = torch.from_numpy(np.asarray(subset["original_argmax"]))[None]
        # float64 throughout the pose difference: d_pose is O(1e-6) and a
        # float32 subtraction of two O(1) vectors loses the quantity being read.
        original_pose6 = torch.from_numpy(
            np.asarray(subset["original_pose6"], dtype=np.float64)
        )[None]
        base_master = np.load(Path(master_row["dx2_base_master"]["path"]), allow_pickle=False)
        born_master = np.load(Path(master_row["born_small_master"]["path"]), allow_pickle=False)
        base_codes = np.asarray(solve_row["solve"]["base_codes"], dtype=np.int32)
        final_codes = np.asarray(solve_row["solve"]["final_codes"], dtype=np.int32)
        base_slave = surface.render(base_codes[None], pair)[0]
        fresh_slave = surface.render(final_codes[None], pair)[0]

        row: dict[str, Any] = {"pair": pair, "legs": {}}
        pair_root = root / f"pair_{pair:04d}"
        for leg, slave, master in (
            (legs[0], base_slave, base_master),
            (legs[1], base_slave, born_master),
            (legs[2], fresh_slave, born_master),
        ):
            stacked = np.stack((slave, master), axis=0)[None]
            require_free_bytes(stacked.nbytes, f"stage_40_{leg}_pair_{pair}")
            payload = retain_npy(pair_root / f"{leg}.pose_input.uint8.npy", stacked)
            tensor = (
                torch.from_numpy(np.ascontiguousarray(stacked))
                .permute(0, 1, 4, 2, 3)
                .float()
            )
            metrics, pose6, logits = rj2.measured_metrics(
                tensor,
                posenet=posenet,
                segnet=segnet,
                original_argmax=original_argmax,
                original_pose6=original_pose6,
            )
            row["legs"][leg] = {
                "d_seg": float(metrics["d_seg"]),
                "d_pose": float(metrics["d_pose"]),
                "pose6": pose6[0].cpu().numpy().astype(float).tolist(),
                "pose_input": payload,
                "argmax": retain_npy(
                    pair_root / f"{leg}.argmax.uint8.npy",
                    logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8),
                ),
            }
        row["fresh_equals_stale_d_seg"] = bool(
            row["legs"][legs[2]]["d_seg"] == row["legs"][legs[1]]["d_seg"]
        )
        retain_json(pair_root / "PAIR_ROW.json", row)
        per_pair.append(row)

    aggregate: dict[str, Any] = {}
    dx2_bytes = int(stage_30["dx2_archive_bytes"])
    resolved_bytes = stage_30.get("resolved_archive_bytes")
    for leg in legs:
        d_seg = float(np.mean([row["legs"][leg]["d_seg"] for row in per_pair]))
        d_pose = float(np.mean([row["legs"][leg]["d_pose"] for row in per_pair]))
        archive_bytes = (
            resolved_bytes if leg == legs[2] else dx2_bytes
        )
        row_out: dict[str, Any] = {
            "measured_pairs": len(per_pair),
            "population_pairs": PAIR_COUNT,
            "d_seg": d_seg,
            "d_pose": d_pose,
            "seg_s": 100.0 * d_seg,
            "pose_s": math.sqrt(10.0 * d_pose),
            "distortion_s": 100.0 * d_seg + math.sqrt(10.0 * d_pose),
            "axis": AXIS,
            "score_claim": False,
            "promotion_eligible": False,
        }
        if archive_bytes is None:
            # Stage 3 refused this leg's container; the rate half is UNMEASURED
            # and no S may be recomputed for it.  Distortion still stands.
            row_out["archive_bytes"] = None
            row_out["rate_s"] = None
            row_out["s"] = None
        else:
            arithmetic = rj2.contest_arithmetic(
                d_seg=d_seg, d_pose=d_pose, archive_bytes=int(archive_bytes)
            )
            row_out["archive_bytes"] = int(archive_bytes)
            row_out["rate_s"] = arithmetic["rate_s"]
            row_out["s"] = arithmetic["s"]
        aggregate[leg] = row_out

    stale = aggregate[legs[1]]
    fresh = aggregate[legs[2]]
    base = aggregate[legs[0]]
    pose_recovered = stale["pose_s"] - fresh["pose_s"]
    pose_gap = stale["pose_s"] - base["pose_s"]
    seg_identical = all(row["fresh_equals_stale_d_seg"] for row in per_pair)
    perfect_pose_floor_s = fresh["seg_s"] + base["pose_s"]

    result = {
        "schema": "ddm_bs4y_stage_40_three_way_measurement.v1",
        "stage": 4,
        "status": "MEASURED",
        "axis": AXIS,
        "verdict_scope": (
            "INSTANCE: sealed BS3 random-n32 born-small object through the exact "
            "DX2 receiver, carrier and scorers"
        ),
        "selection_pairs": stage_10["selection_pairs"],
        "prefix": False,
        "legs": aggregate,
        "per_pair": per_pair,
        "adjudication": {
            "d_seg_identical_across_carrier_legs": seg_identical,
            "carrier_is_frame0_only": stage_10["frame0_is_carrier_only_all"],
            "pose_s_gap_stale_minus_base": pose_gap,
            "pose_s_recovered_by_fresh_solve": pose_recovered,
            "pose_gap_recovered_fraction": (
                pose_recovered / pose_gap if abs(pose_gap) > 0.0 else None
            ),
            "distortion_s_stale": stale["distortion_s"],
            "distortion_s_fresh": fresh["distortion_s"],
            "distortion_s_base": base["distortion_s"],
            "distortion_s_delta_fresh_minus_base": fresh["distortion_s"] - base["distortion_s"],
            "perfect_pose_floor_distortion_s": perfect_pose_floor_s,
            "perfect_pose_floor_delta_vs_base": perfect_pose_floor_s - base["distortion_s"],
        },
        "gb1_context_not_this_instrument": {
            "archive_bytes": GB1_ARCHIVE_BYTES,
            "s": GB1_SCORE,
            "axis": "[contest-CUDA T4 n600]",
            "note": "recalled contest row; different authority surface, not inserted into the n32 legs",
        },
        "scorer_forwards": 3 * len(per_pair),
        "all_materialized_payloads_retained": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    result["checkpoint"] = stage0.atomic_json_once(
        stage0.additive_checkpoint_path(checkpoint, result), result
    )
    return result


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def latest_ready_stage0() -> dict[str, Any]:
    candidates = sorted(
        CHECKPOINTS.glob("stage_00_source_preflight*.json"),
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
            return file_fact(path)
    raise BS4YError("no retained identity-clean READY Stage-0 checkpoint exists")


STAGES: Final = ("stage_10", "stage_20", "stage_30", "stage_40")


def run(*, output: Path = OUTPUT, resume_from: str = "stage_10") -> dict[str, Any]:
    if output.resolve() != OUTPUT.resolve():
        raise BS4YError("BS4Y may write only the charter-mandated APDataStore root")
    if resume_from not in STAGES:
        raise BS4YError(f"--resume-from must be one of {STAGES}")

    stage0_receipt = latest_ready_stage0()
    fire_order_control = stage0.checked_file(
        "fire_order", FIRE_ORDER, None, FIRE_ORDER_SHA256
    )
    selection_control = stage0.checked_file(
        "sealed_random_n32_selection", SELECTION, SELECTION_BYTES, SELECTION_SHA256
    )
    if not fire_order_control["passed"] or not selection_control["passed"]:
        raise BS4YError("sealed FIRE_ORDER or selection pin differs")
    rj2.require_pins()
    selection = np.load(SELECTION, allow_pickle=False)
    if (
        selection.dtype != np.int32
        or selection.shape != (SAMPLE_PAIRS,)
        or np.any(np.diff(selection) <= 0)
    ):
        raise BS4YError("sealed selection dtype, shape, or ordering differs")

    start = STAGES.index(resume_from)
    stage_10 = (
        stage_10_masters(selection)
        if start <= 0
        else _resume("stage_10_exact_born_small_masters")
    )
    stage_20 = (
        stage_20_solves(stage_10)
        if start <= 1
        else _resume("stage_20_qs5_exact_pair_solves")
    )
    stage_30 = (
        stage_30_container(stage_20)
        if start <= 2
        else _resume("stage_30_resolved_carrier_container")
    )
    stage_40 = stage_40_three_way(stage_10, stage_20, stage_30)

    return {
        "schema": "ddm_bs4y_stage_1_4_execution.v1",
        "status": "STAGES_1_THROUGH_4_COMPLETE",
        "axis": AXIS,
        "stage0_receipt": stage0_receipt,
        "fire_order": fire_order_control,
        "selection": selection_control,
        "stage_10": stage_10["checkpoint"],
        "stage_20": stage_20["checkpoint"],
        "stage_30": stage_30["checkpoint"],
        "stage_40": stage_40["checkpoint"],
        "legs": stage_40["legs"],
        "adjudication": stage_40["adjudication"],
        "stage_5_fired": False,
        "modal_invocations": 0,
        "upstream_mutated": False,
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
            "runner": file_fact(Path(__file__).resolve()),
        },
    }


def _resume(stem: str) -> dict[str, Any]:
    candidates = sorted(
        CHECKPOINTS.glob(f"{stem}*.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        raise BS4YError(f"cannot resume: no retained {stem} checkpoint")
    value = json.loads(candidates[0].read_text())
    # The retained checkpoint body deliberately excludes its own identity, so a
    # resumed stage must be re-stamped before a later stage dereferences it.
    value["checkpoint"] = file_fact(candidates[0])
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume-from", default="stage_10", choices=STAGES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(output=args.output.resolve(), resume_from=args.resume_from)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
