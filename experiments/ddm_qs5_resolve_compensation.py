#!/usr/bin/env python3
"""QS5 exact-object compensation repair and partial de-trim compile.

The runner never invokes SegNet, Modal, or ``upstream/evaluate.py``.  It binds
the QS4 worker receipt, emits the exact retained-field recovery rung when the
Modal volume is unreachable, restores only zero-priced connective token sites,
renders that exact frame-1 object, re-solves frame-0 compensation against it,
and byte-closes the result through the proven HP3/RC64 path.  Every materialized
payload is retained under the governed SSD store.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_cp135_rate_compose as cp135
from experiments import ddm_jo1_joint_probability_object as jo1
from experiments import ddm_js1_stage0_per_edge as js1
from experiments import ddm_js1b_modal_cuda_argmax_field_materializer as js1b
from experiments import ddm_js6_seg_representation_join as js6
from experiments import ddm_qs1_frame0_schur_coupled_solve as qs1
from experiments import ddm_qs2_compensation_overlay_runtime as overlay_codec
from experiments import ddm_qs2_compensation_rate_rung as qs2
from experiments import ddm_qs3_saturation_compose as qs3
from experiments import ddm_qs4_collateral_suppression as qs4
from tac.semantic_pipeline.contracts import ClipConfig, require_device
from tac.semantic_pipeline.stages.compensation import (
    CompensationRequest,
)
from tac.semantic_pipeline.stages.compensation import (
    restore_neutral_connective_support as _ported_restore_neutral_connective_support,
)

OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_qs5_20260813")
QS4_RESULT: Final = (
    qs4.OUTPUT
    / "dispatch/ddm_qs4_dual_axis_20260813_r1/QS4_T4_REMOTE_RESULT.json"
)
QS4_FIELD: Final = OUTPUT / "retained/inputs/qs4_candidate_argmax_n600.npy"
QS4_FIELD_SHA256: Final = (
    "34a8dad9cb02f2abcfdd3b0f5cae882ec7269fd448ed10aa7a488b11890acfcc"
)
REMOTE_VOLUME: Final = "comma-ddm-js1b-argmax-retained"
REMOTE_FIELD: Final = (
    "ddm_qs4_dual_axis_20260813_r1/retained/fields/"
    "candidate_argmax_n600.npy"
)
REMOTE_RECOVERY_ARGV: Final = (
    ".venv/bin/modal",
    "volume",
    "get",
    REMOTE_VOLUME,
    REMOTE_FIELD,
    str(QS4_FIELD.resolve()),
)
SELECTED_IDS: Final = (
    "js6_0000_9fbf75d81c43",
    "js6_0004_06fc74e20d9e",
    "js6_0001_da319a6b65d0",
)
PAIR_COUNT: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
DENOMINATOR_PIXELS: Final = PAIR_COUNT * HEIGHT * WIDTH
CP135_BYTES: Final = 186_252
CP135_BASE_DPOSE: Final = 6.885642960696714e-6
RATE_S_PER_BYTE: Final = 25.0 / 37_545_489
QS2_MEASURED_POSE_LEAKAGE_S: Final = 1.126177e-7
POSE_PROJECTION_BUDGET_S: Final = 2.0 * QS2_MEASURED_POSE_LEAKAGE_S
PROJECTED_FIRE_BAR: Final = -1.5e-5
SUPER_BAND_GATE: Final = 1e-5
AXIS: Final = (
    "[macOS-CPU advisory exact-object PoseNet solve + exact byte/container; "
    "scorer-free Seg projection] NON-PROMOTABLE"
)


class QS5Error(RuntimeError):
    """A retained input, object binding, solve, coder, or fire gate differed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retain_json(path: Path, value: Any) -> dict[str, Any]:
    qs1.atomic_json(path, value)
    return qs1.file_record(path)


def storage_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    retained = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    expected = 32 * 1024**3
    reserve = 8 * 1024**3
    required = max(0, expected - retained) + reserve
    free = shutil.disk_usage(output).free
    result = {
        "schema": "ddm_qs5_storage_preflight.v1",
        "tier": str(output.resolve()),
        "already_retained_bytes": retained,
        "expected_total_bytes": expected,
        "reserve_bytes": reserve,
        "required_free_bytes": required,
        "free_bytes": free,
        "passed": free >= required,
        "cleanup_policy": "certify-or-block; no generated payload deleted or moved",
    }
    retain_json(output / "STORAGE_PREFLIGHT.json", result)
    if not result["passed"]:
        raise QS5Error("SSD storage preflight failed")
    return result


def _embedded_worker_result(path: Path) -> dict[str, Any]:
    outer = json.loads(path.read_text())
    encoded = outer["artifacts"]["QS1_T4_REMOTE_RESULT.json"]
    payload = ast.literal_eval(encoded)
    if not isinstance(payload, bytes):
        raise QS5Error("QS4 embedded worker result is not a byte payload")
    result = json.loads(payload)
    if result.get("execution_status") != "MEASUREMENT_COMPLETE":
        raise QS5Error("QS4 worker measurement is incomplete")
    return result


def source_preflight(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    worker = _embedded_worker_result(QS4_RESULT)
    field = worker["retention"]["candidate_argmax_field"]
    if field["sha256"] != QS4_FIELD_SHA256 or int(field["bytes"]) != 117_964_928:
        raise QS5Error("QS4 worker field identity differs from the charter")
    measurement = worker["field_measurement"]
    pose = worker["pose_measurement"]
    if (
        int(measurement["candidate_changed_pixels_vs_cp135"]) != 100
        or int(measurement["candidate_minus_base_flips"]) != -17
        or not math.isclose(
            float(pose["d_pose_candidate_first"]),
            7.2889442890300415e-6,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or float(pose["d_pose_candidate_repeat"])
        != float(pose["d_pose_candidate_first"])
    ):
        raise QS5Error("QS4 refusal components differ from the charter")
    sources = {
        "qs4_modal_envelope": qs1.require_file(QS4_RESULT),
        "qs4_compile": qs1.require_file(qs4.OUTPUT / "candidate/COMPILE_RESULT.json"),
        "qs4_collateral_map": qs1.require_file(qs4.OUTPUT / "COLLATERAL_MAP.json"),
        "qs4_full_bank_screen": qs1.require_file(qs4.OUTPUT / "FULL_BANK_SCREEN.json"),
        "base_argmax": qs1.require_file(
            qs3.QS1_BASE_FIELD, expected_sha256=qs3.EXPECTED_BASE_SHA256
        ),
        "gt_argmax": qs1.require_file(
            qs3.QS1_GT_FIELD, expected_sha256=qs3.EXPECTED_GT_SHA256
        ),
        "cp135_archive": qs1.require_file(
            qs1.CP135_ARCHIVE,
            expected_bytes=CP135_BYTES,
            expected_sha256=qs1.CP135_ARCHIVE_SHA256,
        ),
        "cp135_raw": qs1.require_file(
            qs1.CP135_RAW,
            expected_bytes=qs1.RAW_BYTES,
            expected_sha256=qs1.CP135_RAW_SHA256,
        ),
        "cp135_pose": qs1.require_file(qs1.CP135_BASE_POSE),
        "gt_pose": qs1.require_file(qs1.GT_POSE),
        "qs1_engine": qs1.require_file(Path(qs1.__file__).resolve()),
        "qs2_engine": qs1.require_file(Path(qs2.__file__).resolve()),
        "qs4_engine": qs1.require_file(Path(qs4.__file__).resolve()),
        "qs5_engine": qs1.require_file(Path(__file__).resolve()),
        "joint_integer_solver": qs1.require_file(qs1.JOINT_SOLVER_SOURCE),
        "upstream_modules": qs1.require_file(qs1.UPSTREAM / "modules.py"),
        "upstream_pose_weights": qs1.require_file(
            qs1.UPSTREAM / "models/posenet.safetensors"
        ),
        "dispatcher": qs1.require_file(
            REPO / "experiments/ddm_qs1_modal_t4_dual_axis.py"
        ),
        "worker": qs1.require_file(REPO / "experiments/ddm_re1t_t4_sign_gate_worker.py"),
        "js1b_worker": qs1.require_file(
            REPO / "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py"
        ),
    }
    result = {
        "schema": "ddm_qs5_source_preflight.v1",
        "sources": sources,
        "qs4_worker_components": {
            "changed_pixels": 100,
            "net_flip_improvement": 17,
            "candidate_dpose": float(pose["d_pose_candidate_first"]),
            "deterministic_pose_repeat": True,
            "candidate_field": field,
        },
        "seed": 135,
        "deterministic_algorithms": True,
        "resume_from": str(output.resolve()),
        "scorer_slot_owned": False,
        "segnet_rerun": False,
        "modal_fired": False,
        "passed": True,
    }
    retain_json(output / "checkpoints/stage_00_source_preflight.json", result)
    return result, worker


def decompose_qs4_field(output: Path, worker: dict[str, Any]) -> dict[str, Any]:
    recovery = {
        "schema": "ddm_qs5_qs4_field_recovery.v1",
        "disposition": "FIRED_LOCALLY_BUT_BLOCKED_BY_MODAL_CONNECTIVITY",
        "owner": "MAIN if Modal connectivity remains unavailable to Codex",
        "consumer_store": str(output.resolve()),
        "fire_trigger": "Modal API connectivity is available",
        "exact_recovery_argv": list(REMOTE_RECOVERY_ARGV),
        "remote_volume": REMOTE_VOLUME,
        "remote_path": REMOTE_FIELD,
        "local_destination": str(QS4_FIELD.resolve()),
        "expected_bytes": 117_964_928,
        "expected_sha256": QS4_FIELD_SHA256,
        "observed_failure": "Could not connect to the Modal server",
        "no_scorer_rerun": True,
    }
    if not QS4_FIELD.is_file():
        result = {
            "schema": "ddm_qs5_qs4_bhw_decomposition.v1",
            "axis": "[contest-CUDA T4 worker scalars; dense retained field unavailable locally] COMPONENT-ONLY",
            "status": "BLOCKED_MISSING_LOCAL_QS4_FIELD",
            "selection_mode": "all 100 worker-reported changed pixels once field is recovered",
            "worker_measured": {
                "changed": 100,
                "B_minus_H": 17,
                "candidate_flips": int(worker["field_measurement"]["candidate_flips_vs_gt"]),
                "base_flips": int(worker["field_measurement"]["base_flips_vs_gt"]),
            },
            "B": None,
            "H": None,
            "W": None,
            "lost_modeled_flips": 40,
            "lost_modeled_flips_status": (
                "DERIVED 57 modeled minus 17 realized; trim-vs-new-collateral split unresolved"
            ),
            "recovery": recovery,
            "score_claim": False,
        }
        retain_json(output / "QS4_FIELD_RECOVERY.json", recovery)
        retain_json(output / "QS4_BHW_DECOMPOSITION.json", result)
        retain_json(output / "checkpoints/stage_05_qs4_bhw_decomposition.json", result)
        return result
    record = qs1.require_file(QS4_FIELD, expected_sha256=QS4_FIELD_SHA256)
    base = np.load(qs3.QS1_BASE_FIELD, mmap_mode="r", allow_pickle=False)
    candidate = np.load(QS4_FIELD, mmap_mode="r", allow_pickle=False)
    gt = np.load(qs3.QS1_GT_FIELD, mmap_mode="r", allow_pickle=False)
    changed, beneficial, harmful, wrong = qs4.classify_changes(base, candidate, gt)
    totals = {
        "changed": int(np.count_nonzero(changed)),
        "B": int(np.count_nonzero(beneficial)),
        "H": int(np.count_nonzero(harmful)),
        "W": int(np.count_nonzero(wrong)),
    }
    if (
        totals["changed"] != 100
        or totals["B"] + totals["H"] + totals["W"] != totals["changed"]
        or totals["B"] - totals["H"] != 17
    ):
        raise QS5Error(f"recovered QS4 B/H/W totals differ: {totals}")
    rows = []
    for pair in sorted(np.flatnonzero(np.any(changed, axis=(1, 2))).tolist()):
        rows.append(
            {
                "pair": pair,
                "changed": int(np.count_nonzero(changed[pair])),
                "B": int(np.count_nonzero(beneficial[pair])),
                "H": int(np.count_nonzero(harmful[pair])),
                "W": int(np.count_nonzero(wrong[pair])),
            }
        )
    payload = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode() for row in rows
    )
    row_record = qs1.retain_bytes(
        output / "retained/decomposition/qs4_bhw_per_pair.jsonl", payload
    )
    result = {
        "schema": "ddm_qs5_qs4_bhw_decomposition.v1",
        "axis": "[contest-CUDA T4 retained argmax fields, n600] COMPONENT-ONLY",
        "status": "MEASURED_COMPLETE",
        "selection_mode": "all 100 changed pixels; no sample or prefix",
        **totals,
        "B_minus_H": totals["B"] - totals["H"],
        "field": record,
        "per_pair": row_record,
        "recovery": {**recovery, "disposition": "FOLDED_FIELD_RECOVERED"},
        "score_claim": False,
    }
    retain_json(output / "QS4_FIELD_RECOVERY.json", result["recovery"])
    retain_json(output / "QS4_BHW_DECOMPOSITION.json", result)
    retain_json(output / "checkpoints/stage_05_qs4_bhw_decomposition.json", result)
    return result


def restore_neutral_connective_support(
    site_rows: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, int]]:
    """Keep strict-positive sites and restore only zero-net priced connectors."""
    return _ported_restore_neutral_connective_support(site_rows)


def build_partial_detrim_object(output: Path) -> dict[str, Any]:
    checkpoint = output / "checkpoints/stage_10_partial_detrim_object.json"
    if checkpoint.is_file():
        prior = json.loads(checkpoint.read_text())
        for row in prior["rows"]:
            for key in ("candidate_tokens", "support_mask"):
                record = row[key]
                qs1.require_file(
                    Path(record["path"]),
                    expected_bytes=int(record["bytes"]),
                    expected_sha256=str(record["sha256"]),
                )
        return prior
    base_spatial = np.memmap(
        jo1.BASE_SPATIAL,
        mode="r",
        dtype=np.uint8,
        shape=(PAIR_COUNT, HEIGHT, WIDTH),
    )
    map_result = json.loads((qs4.OUTPUT / "COLLATERAL_MAP.json").read_text())
    map_by_id = {row["proposal_id"]: row for row in map_result["per_proposal"]}
    rows = []
    totals = {
        "strict_sites": 0,
        "neutral_restored_sites": 0,
        "negative_sites_excluded": 0,
        "model_B": 0,
        "model_H": 0,
        "model_W": 0,
    }
    for proposal_id in SELECTED_IDS:
        source = map_by_id[proposal_id]
        pair = int(source["pair"])
        support_root = qs4.OUTPUT / "retained/supports" / proposal_id
        site_rows = [
            json.loads(line)
            for line in (support_root / "site_attribution.jsonl").read_text().splitlines()
        ]
        keep_sites, counts = restore_neutral_connective_support(site_rows)
        original_path = qs1.JS6_BANK / "proposals" / proposal_id / "candidate_tokens.uint8.npy"
        original = np.load(original_path, allow_pickle=False)
        candidate = np.asarray(base_spatial[pair]).copy()
        candidate.reshape(-1)[keep_sites] = original.reshape(-1)[keep_sites]
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
        mask.reshape(-1)[keep_sites] = True
        root = output / "retained/supports" / proposal_id
        token_record = qs1.retain_npy(root / "candidate_tokens.uint8.npy", candidate)
        mask_record = qs1.retain_npy(root / "partial_detrim_support.bool.npy", mask)
        if int(np.count_nonzero(candidate != np.asarray(base_spatial[pair]))) != len(keep_sites):
            raise QS5Error(f"partial de-trim token count differs: {proposal_id}")
        for key in totals:
            totals[key] += counts[key]
        rows.append(
            {
                "proposal_id": proposal_id,
                "pair": pair,
                "candidate_tokens": token_record,
                "support_mask": mask_record,
                "kept_sites": len(keep_sites),
                **counts,
                "support_rule": (
                    "retain QS4 strict-positive sites; restore B==H,W==0 connective sites; "
                    "exclude all modeled-negative sites"
                ),
            }
        )
    if totals["model_B"] - totals["model_H"] != 57:
        raise QS5Error(f"partial de-trim modeled net differs: {totals}")
    result = {
        "schema": "ddm_qs5_partial_detrim_object.v1",
        "selection_mode": "all three QS4 compile pairs; complete retained site ledgers",
        "rows": rows,
        "totals": {
            **totals,
            "model_net_flips": totals["model_B"] - totals["model_H"],
        },
        "seg_projection_status": (
            "TOY-BRACKET: preserves QS4's 57-flip nearest-site model while restoring "
            "zero-priced connective sites; unchanged worker is the verdict"
        ),
        "all_materialized_payloads_retained": True,
        "score_claim": False,
    }
    retain_json(output / "PARTIAL_DETRIM_OBJECT.json", result)
    retain_json(checkpoint, result)
    return result


def _render_exact_masters(
    output: Path, object_result: dict[str, Any]
) -> tuple[list[dict[str, Any]], Any]:
    proof_root = output / "retained/receiver_state"
    *_, semantic, _basis, _coefficients = js1.parse_receiver_state(
        js1.CANDIDATES["cp135_base"], proof_root
    )
    semantic = semantic.eval().cpu()
    rendered = []
    for row in object_result["rows"]:
        root = output / "retained/supports" / row["proposal_id"] / "exact_master"
        result_path = root / "RESULT.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            for key in ("pre_r", "master_camera", "scorer_input"):
                record = result[key]
                qs1.require_file(
                    Path(record["path"]),
                    expected_bytes=int(record["bytes"]),
                    expected_sha256=str(record["sha256"]),
                )
            rendered.append(result)
            continue
        tokens = np.load(row["candidate_tokens"]["path"], allow_pickle=False)
        pre_r, master, scorer_input = js6.render_candidate(
            semantic, tokens, int(row["pair"])
        )
        result = {
            "schema": "ddm_qs5_exact_master_render.v1",
            "proposal_id": row["proposal_id"],
            "pair": int(row["pair"]),
            "semantic_tokens": row["candidate_tokens"],
            "pre_r": qs1.retain_npy(root / "candidate_pre_r.float32.npy", pre_r),
            "master_camera": qs1.retain_npy(
                root / "candidate_camera.uint8.npy", master
            ),
            "scorer_input": qs1.retain_npy(
                root / "candidate_scorer_input.float32.npy", scorer_input
            ),
            "exact_master_rendered_from_semantic_tokens": True,
            "all_materialized_payloads_retained": True,
        }
        retain_json(result_path, result)
        rendered.append(result)
    return rendered, semantic


def solve_exact_object(
    *,
    output: Path,
    support: dict[str, Any],
    rendered: dict[str, Any],
    surface: qs1.CP135Surface,
    posenet: Any,
    raw: np.memmap,
    base_pose_all: np.ndarray,
    gt_pose_all: np.ndarray,
    solver: Any,
) -> dict[str, Any]:
    proposal_id = str(support["proposal_id"])
    pair = int(support["pair"])
    root = output / "retained/compensation" / proposal_id
    result_path = root / "RESULT.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text())
        qs1.assert_compensation_matches_compile_object(prior)
        return prior
    base_codes = surface.codes[pair].copy()
    base_master = np.asarray(raw[2 * pair + 1])
    exact_master = np.load(rendered["master_camera"]["path"], allow_pickle=False)
    baseline = qs1.evaluate_codes(
        surface=surface,
        posenet=posenet,
        codes=(base_codes,),
        master=base_master,
        pair=pair,
        stage_root=root / "stage_10_baseline",
    )[0]
    event = qs1.evaluate_codes(
        surface=surface,
        posenet=posenet,
        codes=(base_codes,),
        master=exact_master,
        pair=pair,
        stage_root=root / "stage_20_exact_object_leak",
    )[0]
    retained_base_drift = float(np.max(np.abs(baseline - base_pose_all[pair])))
    leak = event.astype(np.float64) - baseline.astype(np.float64)
    jacobian_codes = [base_codes.copy()]
    for dimension in range(qs1.DIMENSIONS):
        for delta in (-1, 1):
            candidate = base_codes.copy()
            candidate[dimension] += delta
            if not -2048 <= candidate[dimension] <= 2047:
                raise QS5Error("CP135 coefficient is at an int12 endpoint")
            jacobian_codes.append(candidate)
    jacobian_vectors = qs1.evaluate_codes(
        surface=surface,
        posenet=posenet,
        codes=tuple(jacobian_codes),
        master=exact_master,
        pair=pair,
        stage_root=root / "stage_30_jacobian",
    )
    jacobian = np.empty((qs1.POSE_DIMENSIONS, qs1.DIMENSIONS), dtype=np.float64)
    for dimension in range(qs1.DIMENSIONS):
        minus = jacobian_vectors[1 + 2 * dimension]
        plus = jacobian_vectors[2 + 2 * dimension]
        jacobian[:, dimension] = (
            plus.astype(np.float64) - minus.astype(np.float64)
        ) / 2.0
    qs1.retain_npy(root / "stage_30_jacobian/J_POSE0.float64.npy", jacobian)
    solve = solver.solve_damped_least_squares(
        jacobian,
        -leak,
        damping=qs1.GN_DAMPING,
        max_code_step=qs1.MAX_CODE_STEP,
    )
    centre = solver.quantize_int12_update(base_codes, solve.update)
    active = solver.rank_neighbour_dimensions(
        jacobian, solve.update, qs1.NEIGHBOUR_DIMS
    )
    neighbourhood = solver.nearby_int12_candidates(
        base_codes,
        centre,
        active_dimensions=active,
        radius=qs1.NEIGHBOUR_RADIUS,
    )
    neighbourhood_vectors = qs1.evaluate_codes(
        surface=surface,
        posenet=posenet,
        codes=neighbourhood,
        master=exact_master,
        pair=pair,
        stage_root=root / "stage_40_integer_cube",
    )
    objectives = np.mean(
        np.square(neighbourhood_vectors.astype(np.float64) - baseline[None]), axis=1
    )
    best_index = min(
        range(len(neighbourhood)), key=lambda index: (float(objectives[index]), index)
    )
    current_codes = np.asarray(neighbourhood[best_index], dtype=np.int32)
    current_objective = float(objectives[best_index])

    def evaluate_descent(
        candidates: tuple[np.ndarray, ...], pass_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        vectors = qs1.evaluate_codes(
            surface=surface,
            posenet=posenet,
            codes=candidates,
            master=exact_master,
            pair=pair,
            stage_root=root / f"stage_50_descent/pass_{pass_index:04d}",
        )
        values = np.mean(
            np.square(vectors.astype(np.float64) - baseline[None]), axis=1
        )
        qs1.retain_npy(
            root / f"stage_50_descent/pass_{pass_index:04d}/OBJECTIVES.float64.npy",
            values,
        )
        return vectors, values

    final_codes, final_objective, passes, final_vector = qs1.strict_descent(
        current_codes, current_objective, evaluate_descent
    )
    residual = final_vector.astype(np.float64) - baseline.astype(np.float64)
    metrics = qs1.cancellation_metrics(leak, residual)
    exact_delta_dpose = (
        float(np.sum(np.square(final_vector.astype(np.float64) - gt_pose_all[pair])))
        - float(np.sum(np.square(baseline.astype(np.float64) - gt_pose_all[pair])))
    ) / (PAIR_COUNT * qs1.POSE_DIMENSIONS)
    fingerprint = qs1.compensation_object_fingerprint(
        pair=pair,
        semantic_tokens=support["candidate_tokens"],
        master_camera=rendered["master_camera"],
    )
    binding = {
        "schema": "ddm_qs1_compensation_object_binding.v1",
        "pair": pair,
        "semantic_tokens": support["candidate_tokens"],
        "master_camera": rendered["master_camera"],
        "fingerprint_sha256": fingerprint,
        "exact_master_rendered_from_semantic_tokens": True,
    }
    result = {
        "schema": "ddm_qs5_exact_object_schur_pair_result.v1",
        "proposal_id": proposal_id,
        "pair": pair,
        "axis": "[macOS-CPU advisory frozen CPU-torch PoseNet; exact receiver objects]",
        "candidate_tokens_path": support["candidate_tokens"]["path"],
        "token_site_count": int(support["kept_sites"]),
        "compensation_object": binding,
        "solve": {
            "jacobian_rank": int(solve.rank),
            "jacobian_condition": float(solve.condition),
            "ridge_lambda": float(solve.ridge_lambda),
            "float_update": solve.update.tolist(),
            "quantized_centre": centre.tolist(),
            "active_dimensions": list(active),
            "integer_cube_candidates": len(neighbourhood),
            "coordinate_descent_full_passes": passes,
            "derived_stop": (
                "one complete signed-int12 singleton pass accepted zero strict improvements"
            ),
            "base_codes": base_codes.tolist(),
            "final_codes": final_codes.tolist(),
            "final_code_delta": (final_codes - base_codes).tolist(),
            "final_objective_mse_to_base_pose_vector": final_objective,
            "compensation_object_fingerprint_sha256": fingerprint,
        },
        "pose": {
            **metrics,
            "leak_vector": leak.tolist(),
            "residual_vector": residual.tolist(),
            "exact_local_delta_dpose_one_pair_over_n600": exact_delta_dpose,
            "retained_base_pose_vector_max_abs_drift": retained_base_drift,
        },
        "all_materialized_payloads_retained": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    qs1.retain_npy(root / "FINAL_CODES.int32.npy", final_codes)
    qs1.retain_npy(root / "FINAL_POSE_VECTOR.float32.npy", final_vector)
    retain_json(result_path, result)
    qs1.assert_compensation_matches_compile_object(result)
    return result


def _exact_overlay_candidate(
    output: Path,
    primary: dict[str, Any],
    solved: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    pairs, deltas = qs2.exact_deltas(solved)
    if np.any(deltas < overlay_codec.MIN_DELTA) or np.any(
        deltas > overlay_codec.MAX_DELTA
    ):
        return None
    sources = qs4._rate_sources_from_archive(Path(primary["archive"]["path"]))
    rows = [
        qs2.build_rate_candidate(
            output=output,
            label="qs5_exact_object_step_1",
            pair_indices=pairs,
            deltas=deltas,
            carrier_quality=quality,
            sources=sources,
        )
        for quality in range(12)
    ]
    return min(rows, key=lambda row: (row["archive"]["bytes"], row["carrier_quality"]))


def _compile_final_runtime(
    output: Path,
    primary: dict[str, Any],
    solved: Sequence[dict[str, Any]],
    overlay_winner: dict[str, Any] | None,
) -> dict[str, Any]:
    root = output / "candidate"
    root.mkdir(parents=True, exist_ok=True)
    if overlay_winner is None:
        source_archive = Path(primary["archive"]["path"])
        archive_payload = source_archive.read_bytes()
        archive = qs1.retain_bytes(root / "archive.zip", archive_payload)
        repeat = qs1.retain_bytes(
            root / "archive.repeat.zip", qs2.deterministic_zip(qs2._zip_member(source_archive))
        )
        if archive["sha256"] != repeat["sha256"]:
            raise QS5Error("direct exact-object archive repeat differs")
        runtime_root = Path(primary["runtime_root"])
        return {
            "chosen_carrier_form": "direct_variable_CAP1",
            "archive": archive,
            "archive_repeat": repeat,
            "runtime_root": str(runtime_root.resolve()),
            "runtime_tree": cp135.tree_record(runtime_root),
            "overlay": None,
            "receiver_code_lattice_exact": True,
        }
    source_archive = Path(overlay_winner["archive"]["path"])
    archive_payload = source_archive.read_bytes()
    archive = qs1.retain_bytes(root / "archive.zip", archive_payload)
    repeat = qs1.retain_bytes(
        root / "archive.repeat.zip", qs2.deterministic_zip(qs2._zip_member(source_archive))
    )
    if archive["sha256"] != repeat["sha256"]:
        raise QS5Error("overlay exact-object archive repeat differs")
    runtime_root = root / "adapted_runtime"
    runtime_copy = jo1.copy_runtime(runtime_root, archive_payload)
    runtime_patches = qs2.patch_runtime(runtime_root)
    parseback = qs2.runtime_parseback(
        runtime_root=runtime_root,
        archive=runtime_root / "archive.zip",
        expected_overlay=overlay_winner["overlay"],
    )
    base_codes = qs1._load_cp135_carrier_codes()
    pairs, deltas = qs2.exact_deltas(solved)
    payload = Path(overlay_winner["overlay"]["path"]).read_bytes()
    actual = overlay_codec.apply_compensation_overlay(base_codes, payload)
    expected = base_codes.copy()
    expected[pairs.astype(np.int64)] += deltas
    if not np.array_equal(actual, expected):
        raise QS5Error("QS5 overlay does not reproduce the fresh exact-object solve")
    codes = qs1.retain_npy(root / "candidate_codes.int32.npy", actual)
    return {
        "chosen_carrier_form": "Q2C1_exact_sparse_overlay",
        "archive": archive,
        "archive_repeat": repeat,
        "runtime_root": str(runtime_root.resolve()),
        "runtime_tree": cp135.tree_record(runtime_root),
        "runtime_copy": runtime_copy,
        "runtime_patches": runtime_patches,
        "runtime_parseback": parseback,
        "overlay": overlay_winner["overlay"],
        "candidate_codes": codes,
        "receiver_code_lattice_exact": True,
    }


def compile_candidate(output: Path, object_result: dict[str, Any]) -> dict[str, Any]:
    result_path = output / "candidate/COMPILE_RESULT.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text())
        for key in ("archive", "archive_repeat"):
            record = prior[key]
            qs1.require_file(
                Path(record["path"]),
                expected_bytes=int(record["bytes"]),
                expected_sha256=str(record["sha256"]),
            )
        for row in prior["fresh_compensation_rows"]:
            qs1.assert_compensation_matches_compile_object(row)
        return prior
    rendered, _semantic = _render_exact_masters(output, object_result)
    surface, _carrier = qs1.CP135Surface.load()
    posenet = qs1.load_posenet()
    solver = qs1._load_module("ddm_qs5_joint_pose_solve", qs1.JOINT_SOLVER_SOURCE)
    raw = np.memmap(
        qs1.CP135_RAW,
        mode="r",
        dtype=np.uint8,
        shape=(PAIR_COUNT * 2, qs1.CAMERA_H, qs1.CAMERA_W, 3),
    )
    base_pose = np.load(qs1.CP135_BASE_POSE, allow_pickle=False)
    gt_pose = np.load(qs1.GT_POSE, allow_pickle=False)
    by_id = {row["proposal_id"]: row for row in rendered}
    solved = [
        solve_exact_object(
            output=output,
            support=support,
            rendered=by_id[support["proposal_id"]],
            surface=surface,
            posenet=posenet,
            raw=raw,
            base_pose_all=base_pose,
            gt_pose_all=gt_pose,
            solver=solver,
        )
        for support in object_result["rows"]
    ]
    solved.sort(key=lambda row: int(row["pair"]))
    # The solve is deliberately inside this compile.  The QS1 compiler then
    # rechecks every content binding before it touches the carrier lattice.
    primary = qs1._compile_one(output=output, selected=solved, repeat=False)
    repeated = qs1._compile_one(output=output, selected=solved, repeat=True)
    if primary["archive"]["sha256"] != repeated["archive"]["sha256"]:
        raise QS5Error("fresh exact-object HP3/RC64 repeat differs")
    overlay_winner = _exact_overlay_candidate(output, primary, solved)
    final = _compile_final_runtime(output, primary, solved, overlay_winner)
    model_net_flips = int(object_result["totals"]["model_net_flips"])
    delta_bytes = int(final["archive"]["bytes"]) - CP135_BYTES
    seg_delta_s = -100.0 * model_net_flips / DENOMINATOR_PIXELS
    rate_delta_s = delta_bytes * RATE_S_PER_BYTE
    projected_delta_s = seg_delta_s + POSE_PROJECTION_BUDGET_S + rate_delta_s
    local_delta_dpose = sum(
        float(row["pose"]["exact_local_delta_dpose_one_pair_over_n600"])
        for row in solved
    )
    local_pose_delta_s = (
        math.sqrt(10.0 * (CP135_BASE_DPOSE + local_delta_dpose))
        - math.sqrt(10.0 * CP135_BASE_DPOSE)
    )
    result = {
        "schema": "ddm_qs5_compiled_candidate.v1",
        "axis": AXIS,
        "selected_proposal_ids": [row["proposal_id"] for row in solved],
        "selected_pairs": [int(row["pair"]) for row in solved],
        "fresh_compensation_rows": solved,
        "all_compensation_solved_inside_compile": True,
        "hp3_rc64_primary": primary,
        "hp3_rc64_repeat": repeated["archive"],
        "hp3_rc64_repeat_byte_identical": True,
        "exact_overlay_race": overlay_winner,
        **final,
        "archive_delta_bytes_vs_cp135": delta_bytes,
        "partial_detrim_model": object_result["totals"],
        "local_pose_advisory": {
            "delta_dpose": local_delta_dpose,
            "delta_pose_score_term": local_pose_delta_s,
            "axis": "[macOS-CPU advisory frozen CPU-torch PoseNet] NON-PROMOTABLE",
        },
        "preworker_projection": {
            "seg_model_net_flips": model_net_flips,
            "seg_delta_s": seg_delta_s,
            "pose_budget_s": POSE_PROJECTION_BUDGET_S,
            "pose_budget_derivation": "2x QS2 measured +1.126177e-7 S leakage",
            "rate_delta_s": rate_delta_s,
            "complete_delta_s": projected_delta_s,
            "required_delta_s_at_most": PROJECTED_FIRE_BAR,
            "target_cleared": projected_delta_s <= PROJECTED_FIRE_BAR,
            "authority": (
                "TOY-BRACKET Seg projection plus derived pose budget and exact bytes; "
                "unchanged T4 worker is the verdict"
            ),
        },
        "all_materialized_payloads_retained": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    retain_json(result_path, result)
    retain_json(output / "checkpoints/stage_40_candidate_compile.json", result)
    return result


def seal_order(output: Path, compiled: dict[str, Any]) -> dict[str, Any]:
    projection = compiled["preworker_projection"]
    if not projection["target_cleared"]:
        order = {
            "schema": "ddm_qs5_sealed_no_fire_order.v1",
            "sealed": True,
            "disposition": "FOLDED",
            "owner": "ddm_qs5",
            "consumer_store": str(output.resolve()),
            "fire_trigger": (
                "reopen only when the same byte-closed object has a projected complete delta "
                "at most -1.5e-5 under the 2x-QS2 pose budget"
            ),
            "reason": "byte-closed candidate missed the charter projection gate",
            "candidate_archive": compiled["archive"],
            "projection": projection,
            "modal_fired": False,
            "score_claim": False,
        }
        retain_json(output / "SEALED_NO_FIRE_ORDER.json", order)
        retain_json(output / "checkpoints/stage_50_sealed_order.json", order)
        return order
    run_id = "ddm_qs5_dual_axis_20260813_r1"
    fire_root = output / "fire_order"
    input_root = fire_root / "fire_inputs"
    archive_path = Path(compiled["archive"]["path"])
    runtime_root = Path(compiled["runtime_root"])
    runtime_bundle, runtime_manifest = js1b.build_runtime_bundle(
        runtime_root, label="ddm_qs5_exact_object_partial_detrim"
    )
    screen_payload = (
        json.dumps(
            {
                "preworker_projection": projection,
                "local_pose_advisory": compiled["local_pose_advisory"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    payloads = {
        "candidate_archive.zip": archive_path.read_bytes(),
        "candidate_runtime.zip": runtime_bundle,
        "POSE_SCREEN_RESULT.json": screen_payload,
    }
    for name, payload in payloads.items():
        qs1.retain_bytes(input_root / name, payload)
    git_status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO)
    request = {
        "schema": "ddm_qs1_t4_dual_axis_request.v1",
        "run_id": run_id,
        "resume_from": run_id,
        "lane_id": "ddm_qs5_resolve_compensation_n600_20260813",
        "instance_job_id": f"modal:{run_id}",
        "claim_agent": "MAIN",
        "seed": 1234,
        "batch_size": 16,
        "retain_pose_vectors": True,
        "candidate_archive": qs1.file_record(archive_path),
        "candidate_runtime": compiled["runtime_tree"],
        "runtime_manifest": runtime_manifest,
        "inputs": {name: js1b.payload_record(payload) for name, payload in payloads.items()},
        "local_pose_delta": compiled["local_pose_advisory"]["delta_dpose"],
        "pose_unmeasured": True,
        "source_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "source_git_dirty": bool(git_status),
        "source_git_status_sha256": hashlib.sha256(git_status).hexdigest(),
        "dispatcher_source_sha256": sha256_file(
            REPO / "experiments/ddm_qs1_modal_t4_dual_axis.py"
        ),
        "worker_source_sha256": sha256_file(
            REPO / "experiments/ddm_re1t_t4_sign_gate_worker.py"
        ),
        "js1b_worker_source_sha256": sha256_file(
            REPO / "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py"
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }
    request_record = retain_json(fire_root / "SEALED_REQUEST.json", request)
    from experiments import ddm_qs1_modal_t4_dual_axis as dispatcher

    dispatcher.load_sealed_inputs(
        Path(request_record["path"]), input_root, str(request_record["sha256"])
    )
    command = [
        ".venv/bin/modal",
        "run",
        "--detach",
        "experiments/ddm_qs1_modal_t4_dual_axis.py::main",
        "--sealed-request",
        request_record["path"],
        "--fire-input-dir",
        str(input_root.resolve()),
        "--expected-request-sha256",
        request_record["sha256"],
        "--output-dir",
        str((output / "dispatch" / run_id).resolve()),
        "--detach",
        "--provider-detach-ack",
    ]
    order = {
        "schema": "ddm_qs5_sealed_fire_order.v1",
        "sealed": True,
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN sole scorer-lane router",
        "consumer_store": str(output.resolve()),
        "fire_trigger": (
            "MAIN confirms no active n600 scorer lane, the worker self-claims "
            "ddm_qs5_resolve_compensation_n600_20260813, and every sealed SHA verifies"
        ),
        "fresh_run_id": run_id,
        "request": request_record,
        "fire_inputs": str(input_root.resolve()),
        "exact_command_argv": command,
        "estimated_cost_usd": 0.16,
        "remote_scope": (
            "one candidate; unchanged worker retains n600 T4 Seg field, official Pose "
            "first-six vectors, inputs, outputs, and deterministic repeat"
        ),
        "post_harvest_rule": (
            "admit only realized matched-instrument delta S < 0; canonical-row naming also "
            "requires abs(delta S) >= 1e-5"
        ),
        "canonical_evaluate_follow_on": "NOT_NAMED_UNTIL_NEGATIVE_SUPER_BAND_WORKER_RESULT",
        "dispatcher_validation_passed": True,
        "modal_fired": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    retain_json(output / "SEALED_FIRE_ORDER.json", order)
    retain_json(output / "checkpoints/stage_50_sealed_order.json", order)
    return order


def run(output: Path = OUTPUT) -> dict[str, Any]:
    if output.resolve() != OUTPUT.resolve():
        raise QS5Error(f"output must be the governed SSD store: {OUTPUT}")
    storage_preflight(output)
    preflight, worker = source_preflight(output)
    decomposition = decompose_qs4_field(output, worker)
    object_result = build_partial_detrim_object(output)
    try:
        compiled = compile_candidate(output, object_result)
    finally:
        js1.release_runtime()
    order = seal_order(output, compiled)
    result = {
        "schema": "ddm_qs5_final_result.v1",
        "axis": AXIS,
        "disposition": order["disposition"],
        "source_preflight": preflight,
        "qs4_bhw_decomposition": decomposition,
        "partial_detrim_object": object_result,
        "compiled_candidate": compiled,
        "sealed_order": order,
        "segnet_rerun": False,
        "modal_fired": False,
        "all_materialized_payloads_retained": True,
        "pointer_moved": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    retain_json(output / "FINAL_RESULT.json", result)
    retain_json(output / "checkpoints/stage_90_final.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume-from", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--clip-config", type=Path)
    parser.add_argument("--pair-scope", type=str)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require_device(args.device)
    port_args = (args.archive, args.clip_config, args.pair_scope)
    if any(value is not None for value in port_args):
        if not all(value is not None for value in port_args):
            raise QS5Error("--archive, --clip-config, and --pair-scope must be supplied together")
        clip_payload = json.loads(args.clip_config.read_text(encoding="utf-8"))
        clip_payload["video"] = Path(clip_payload["video"])
        clip = ClipConfig(**clip_payload)
        pair_ids = tuple(int(value) for value in args.pair_scope.split(",") if value.strip())
        archive_record = qs1.file_record(args.archive)
        CompensationRequest(
            archive=args.archive,
            archive_sha256=archive_record["sha256"],
            archive_bytes=archive_record["bytes"],
            clip=clip,
            device=args.device,
            pair_ids=pair_ids,
        ).validate()
    if args.resume_from.resolve() != args.output.resolve():
        raise QS5Error("--resume-from must equal --output")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "RUN.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise QS5Error("another QS5 process holds the governed run lock") from error
        result = run(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
