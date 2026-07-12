#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Measured local UGC A/B on the completed #399/#400 pair-local exact surface.

No arguments are accepted: this is the pinned 2026-07-12 measurement contract, not an
ad-hoc flag surface.  It reads the STOP-sealed ``click_polish_399_campaign`` fixture and
writes a separate resumable result directory.  The active import campaign is never read or
mutated.  Every scorer cell is frozen CPU-torch in canonical 16-pair layout; every mask uses
the real repacked archive bytes and the exact nonlinear contest objective.  Rows are local
macOS-CPU advisory measurements, never contest score claims, and no dispatch is performed.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Containment: the operator's live run owns the machine.  This measurement is local-only
# and deliberately uses the same two-thread cap as click_polish_local.py.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "2")

import torch  # noqa: E402

from tac import click_polish as cp  # noqa: E402
from tac.through_r.mc_finisher import (  # noqa: E402
    VALID_MASK_ESTIMATORS,
    DirectionPinnedMaskFinisher,
    DirectionPinnedPairLocalObjective,
    exact_bernoulli_estimator_moments,
    measure_estimator_variance,
    ugc_boundary_threshold,
)

AXIS = "[macOS-CPU advisory . frozen CPU-torch exact cells . NON-PROMOTABLE]"
FIXTURE = REPO / "experiments/results/click_polish_399_campaign"
ARCHIVE = FIXTURE / "candidate_archive.zip"
AUTHORITY = FIXTURE / "authority_perpair.npz"
FIXTURE_STATE = FIXTURE / "campaign_state.json"
STOP_SENTINEL = FIXTURE / "STOP"
OUT = REPO / "experiments/results/ugc_terminal_polish_ab_20260712"
SUBMISSION_DIR = REPO / cp.DEFAULT_SUBMISSION_DIR
GT_CACHE = REPO / cp.DEFAULT_GT_CACHE

# DERIVED: one canonical scorer chunk, six approximately equispaced rows.  K=6 makes exact
# enumeration cost 2**K=64, which pins the common variance and search budgets without a toy
# surrogate objective.  The next completed-campaign block starts at pair 144.
CANONICAL_CHUNK = tuple(range(144, 160))
CANDIDATE_PAIRS = (144, 147, 150, 153, 156, 159)
DIRECTION_DELTAS = (1, -1)  # existing click-polish terminal proposal grid
SEED = 396_400


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _score_from_cells(
    packet: cp.FrozenPacket,
    renderer: cp.Renderer,
    q: np.ndarray,
    dseg: np.ndarray,
    dpose: np.ndarray,
) -> tuple[float, int]:
    archive_bytes = len(packet.repack_archive_bytes(q, drop_sidecar=renderer.drop_sidecar))
    return cp.compute_contest_score(float(dseg.mean()), float(dpose.mean()), archive_bytes), archive_bytes


def _load_fixture() -> tuple[
    cp.FrozenPacket,
    cp.Renderer,
    cp.Scorer,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    if not STOP_SENTINEL.exists():
        raise RuntimeError("fixture is not STOP-sealed; refusing to read an in-flight campaign")
    for required in (ARCHIVE, AUTHORITY, FIXTURE_STATE):
        if not required.exists():
            raise FileNotFoundError(required)
    state = json.loads(FIXTURE_STATE.read_text(encoding="utf-8"))
    archive_sha = _sha256(ARCHIVE)
    if archive_sha != state.get("candidate_sha256"):
        raise RuntimeError("fixture archive SHA does not match its campaign state")
    packet = cp.FrozenPacket.parse(ARCHIVE, SUBMISSION_DIR)
    roundtrip = packet.verify_roundtrip()
    if not roundtrip["archive_byte_exact"]:
        raise RuntimeError("fixture archive failed byte-exact parse/repack")
    renderer = cp.Renderer(packet, device="cpu")
    scorer = cp.Scorer(device="cpu")
    locality = cp.verify_pair_locality(packet, renderer)
    if not locality["locality_holds"]:
        raise RuntimeError("pair-locality guard failed on the measurement fixture")
    gt_lstars, gt_poses, _source = cp.load_gt_targets(GT_CACHE, 600)
    with np.load(AUTHORITY) as authority:
        base_dseg = np.asarray(authority["d_seg"], dtype=np.float64)
        base_dpose = np.asarray(authority["d_pose"], dtype=np.float64)
    base_s, base_bytes = _score_from_cells(
        packet, renderer, packet.Q0, base_dseg, base_dpose
    )
    if abs(base_s - float(state["S_authority"])) > 1e-12:
        raise RuntimeError(
            f"fixture S drift: cells={base_s:.17g} state={state['S_authority']:.17g}"
        )
    state = {
        **state,
        "archive_sha256": archive_sha,
        "authority_sha256": _sha256(AUTHORITY),
        "gt_cache_sha256": _sha256(GT_CACHE / "gt_n600.npz"),
        "base_archive_bytes": base_bytes,
        "base_s_rederived": base_s,
        "locality": locality,
    }
    return packet, renderer, scorer, gt_lstars, gt_poses, base_dseg, base_dpose, state


def _direction_sweep(
    packet: cp.FrozenPacket,
    renderer: cp.Renderer,
    scorer: cp.Scorer,
    gt_lstars: np.ndarray,
    gt_poses: np.ndarray,
    base_dseg: np.ndarray,
    base_dpose: np.ndarray,
    fixture_sha: str,
) -> dict[str, np.ndarray | int | float]:
    """Resume-safe shared diagonal sweep that pins one direction per candidate pair."""

    checkpoint = OUT / "direction_sweep_checkpoint.npz"
    directions = tuple(
        (column, delta)
        for column in range(cp.LATENT_DIM)
        for delta in DIRECTION_DELTAS
    )
    k = len(CANDIDATE_PAIRS)
    if checkpoint.exists():
        with np.load(checkpoint) as saved:
            saved_sha = str(saved["fixture_sha"])
            if saved_sha != fixture_sha:
                raise RuntimeError("direction-sweep checkpoint fixture SHA drift")
            next_index = int(saved["next_index"])
            best_s = np.asarray(saved["best_s"], dtype=np.float64)
            best_columns = np.asarray(saved["best_columns"], dtype=np.int64)
            best_values = np.asarray(saved["best_values"], dtype=packet.Q0.dtype)
            edited_dseg = np.asarray(saved["edited_dseg"], dtype=np.float64)
            edited_dpose = np.asarray(saved["edited_dpose"], dtype=np.float64)
        _log(f"direction sweep resumed at {next_index}/{len(directions)}")
    else:
        next_index = 0
        best_s = np.full(k, np.inf, dtype=np.float64)
        best_columns = np.full(k, -1, dtype=np.int64)
        best_values = np.zeros(k, dtype=packet.Q0.dtype)
        edited_dseg = np.full(k, np.nan, dtype=np.float64)
        edited_dpose = np.full(k, np.nan, dtype=np.float64)

    chunk_index = {pair: index for index, pair in enumerate(CANONICAL_CHUNK)}
    for direction_index in range(next_index, len(directions)):
        column, delta = directions[direction_index]
        q_diagonal = packet.Q0.copy()
        for pair in CANDIDATE_PAIRS:
            q_diagonal[pair, column] = np.uint8(
                np.clip(int(q_diagonal[pair, column]) + delta, 0, 255)
            )
        t0 = time.monotonic()
        dseg_chunk, dpose_chunk = cp.render_and_score(
            renderer,
            scorer,
            q_diagonal,
            CANONICAL_CHUNK,
            gt_lstars,
            gt_poses,
            batch_pairs=16,
        )
        for candidate_index, pair in enumerate(CANDIDATE_PAIRS):
            new_value = int(q_diagonal[pair, column])
            if new_value == int(packet.Q0[pair, column]):
                continue
            q_single = packet.Q0.copy()
            q_single[pair, column] = np.uint8(new_value)
            candidate_dseg = base_dseg.copy()
            candidate_dpose = base_dpose.copy()
            cell_index = chunk_index[pair]
            candidate_dseg[pair] = dseg_chunk[cell_index]
            candidate_dpose[pair] = dpose_chunk[cell_index]
            candidate_s, _bytes = _score_from_cells(
                packet, renderer, q_single, candidate_dseg, candidate_dpose
            )
            if candidate_s < best_s[candidate_index]:
                best_s[candidate_index] = candidate_s
                best_columns[candidate_index] = column
                best_values[candidate_index] = np.uint8(new_value)
                edited_dseg[candidate_index] = dseg_chunk[cell_index]
                edited_dpose[candidate_index] = dpose_chunk[cell_index]
        _atomic_npz(
            checkpoint,
            fixture_sha=np.array(fixture_sha),
            next_index=np.int64(direction_index + 1),
            best_s=best_s,
            best_columns=best_columns,
            best_values=best_values,
            edited_dseg=edited_dseg,
            edited_dpose=edited_dpose,
        )
        _log(
            f"direction {direction_index + 1:02d}/{len(directions)} "
            f"col={column} delta={delta:+d} scorer_s={time.monotonic() - t0:.2f}"
        )
    if np.any(best_columns < 0) or not np.all(np.isfinite(best_s)):
        raise RuntimeError("direction sweep failed to pin every candidate")
    return {
        "next_index": len(directions),
        "best_s": best_s,
        "best_columns": best_columns,
        "best_values": best_values,
        "edited_dseg": edited_dseg,
        "edited_dpose": edited_dpose,
    }


def _verify_mask(
    objective: DirectionPinnedPairLocalObjective,
    mask: np.ndarray,
    renderer: cp.Renderer,
    scorer: cp.Scorer,
    gt_lstars: np.ndarray,
    gt_poses: np.ndarray,
) -> dict[str, Any]:
    """Independent canonical-chunk scorer verification of one arm's final mask."""

    table = objective.table_for_mask(mask)
    t0 = time.monotonic()
    dseg_chunk, dpose_chunk = cp.render_and_score(
        renderer,
        scorer,
        table,
        CANONICAL_CHUNK,
        gt_lstars,
        gt_poses,
        batch_pairs=16,
    )
    dseg = objective.base_dseg.copy()
    dpose = objective.base_dpose.copy()
    expected_dseg_chunk = objective.base_dseg[list(CANONICAL_CHUNK)].copy()
    expected_dpose_chunk = objective.base_dpose[list(CANONICAL_CHUNK)].copy()
    chunk_index = {pair: index for index, pair in enumerate(CANONICAL_CHUNK)}
    for bit, pair in enumerate(CANDIDATE_PAIRS):
        if int(mask[bit]) == 1:
            expected_dseg_chunk[chunk_index[pair]] = objective.edited_dseg[bit]
            expected_dpose_chunk[chunk_index[pair]] = objective.edited_dpose[bit]
    seg_cell_maxabs = float(np.max(np.abs(dseg_chunk - expected_dseg_chunk)))
    pose_cell_maxabs = float(np.max(np.abs(dpose_chunk - expected_dpose_chunk)))
    dseg[list(CANONICAL_CHUNK)] = dseg_chunk
    dpose[list(CANONICAL_CHUNK)] = dpose_chunk
    archive_bytes = int(objective.archive_bytes_fn(table))
    verified_s = float(
        cp.compute_contest_score(float(dseg.mean()), float(dpose.mean()), archive_bytes)
    )
    composed = objective.components(mask)
    residual = verified_s - composed.s
    if seg_cell_maxabs != 0.0 or pose_cell_maxabs > 1e-12 or abs(residual) > 1e-12:
        raise RuntimeError(
            "exact cell composition failed final scorer verification: "
            f"seg_cell_maxabs={seg_cell_maxabs:.3e} "
            f"pose_cell_maxabs={pose_cell_maxabs:.3e} residual={residual:.3e}"
        )
    return {
        "verified_s": verified_s,
        "cell_composed_s": composed.s,
        "residual": residual,
        "seg_cell_maxabs": seg_cell_maxabs,
        "pose_cell_maxabs": pose_cell_maxabs,
        "d_seg": float(dseg.mean()),
        "d_pose": float(dpose.mean()),
        "archive_bytes": archive_bytes,
        "wall_clock_s": time.monotonic() - t0,
        "canonical_chunk": list(CANONICAL_CHUNK),
        "authority": "fresh frozen CPU-torch canonical-16 scorer verification",
    }


def main() -> int:
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    OUT.mkdir(parents=True, exist_ok=True)
    _log(f"axis={AXIS}")
    _log(f"fixture={FIXTURE} (STOP-sealed, read-only)")
    packet, renderer, scorer, gt_lstars, gt_poses, base_dseg, base_dpose, fixture = (
        _load_fixture()
    )
    fixture_sha = str(fixture["archive_sha256"])
    _atomic_json(OUT / "fixture_manifest.json", fixture)
    sweep = _direction_sweep(
        packet,
        renderer,
        scorer,
        gt_lstars,
        gt_poses,
        base_dseg,
        base_dpose,
        fixture_sha,
    )
    objective = DirectionPinnedPairLocalObjective(
        base_dseg=base_dseg,
        base_dpose=base_dpose,
        base_table=packet.Q0,
        candidate_pairs=np.asarray(CANDIDATE_PAIRS),
        candidate_columns=np.asarray(sweep["best_columns"]),
        candidate_values=np.asarray(sweep["best_values"]),
        edited_dseg=np.asarray(sweep["edited_dseg"]),
        edited_dpose=np.asarray(sweep["edited_dpose"]),
        archive_bytes_fn=lambda table: len(
            packet.repack_archive_bytes(table, drop_sidecar=renderer.drop_sidecar)
        ),
        authority_label=AXIS,
    )
    k = objective.n_bits
    eval_budget = 1 << k
    tau = ugc_boundary_threshold(k)
    probabilities = np.array([tau / 2.0] * (k // 2) + [0.5] * (k - k // 2))
    zero = np.zeros(k, dtype=np.int8)
    initial = objective.components(zero)
    candidate_rows = []
    for index, pair in enumerate(CANDIDATE_PAIRS):
        candidate_rows.append(
            {
                "bit": index,
                "pair": pair,
                "column": int(objective.candidate_columns[index]),
                "base_value": int(packet.Q0[pair, objective.candidate_columns[index]]),
                "edited_value": int(objective.candidate_values[index]),
                "single_edit_s": float(np.asarray(sweep["best_s"])[index]),
                "single_edit_delta_s": float(np.asarray(sweep["best_s"])[index] - initial.s),
                "edited_dseg_cell": float(objective.edited_dseg[index]),
                "edited_dpose_cell": float(objective.edited_dpose[index]),
            }
        )
    _atomic_json(
        OUT / "candidate_manifest.json",
        {
            "fixture_sha256": fixture_sha,
            "candidate_pairs": list(CANDIDATE_PAIRS),
            "canonical_chunk": list(CANONICAL_CHUNK),
            "direction_grid": [
                [column, delta]
                for column in range(cp.LATENT_DIM)
                for delta in DIRECTION_DELTAS
            ],
            "rows": candidate_rows,
            "initial_components": asdict(initial),
            "authority": AXIS,
        },
    )

    arm_rows: list[dict[str, Any]] = []
    for estimator in VALID_MASK_ESTIMATORS:
        arm_receipt_path = OUT / f"receipt_{estimator}.json"
        force_fresh_search = False
        if arm_receipt_path.exists():
            sealed = json.loads(arm_receipt_path.read_text(encoding="utf-8"))
            # Contract v2 counts the ES reference value inside B. Refuse/resample the one
            # legacy receipt that reported B proposal draws while silently making B+1 calls.
            legacy_es_budget = estimator == "one_plus_one_es" and sealed.get(
                "variance_budget_contract"
            ) != "B_includes_all_objective_calls"
            if not legacy_es_budget:
                arm_rows.append(sealed)
                _log(f"arm={estimator} resumed from sealed receipt")
                continue
            _log("arm=one_plus_one_es legacy B+1 variance receipt invalidated; remeasuring")
            force_fresh_search = True
        _log(f"arm={estimator} variance budget={eval_budget}")
        variance = measure_estimator_variance(
            estimator,
            objective,
            probabilities,
            eval_budget=eval_budget,
            seed=SEED,
            ugc_tau=tau,
        )
        suffix = "_budget_v2" if force_fresh_search else ""
        snapshot = OUT / f"search_{estimator}{suffix}_stage_snapshot.json"
        log_path = OUT / f"search_{estimator}{suffix}_accepted_proposals.jsonl"
        if snapshot.exists() and not force_fresh_search:
            finisher = DirectionPinnedMaskFinisher.resume_from(snapshot, objective)
        else:
            finisher = DirectionPinnedMaskFinisher(
                objective,
                probabilities,
                estimator=estimator,
                initial_mask=zero,
                initial_value=initial.s,
                seed=SEED,
                ugc_tau=tau,
            )
        _log(f"arm={estimator} search budget={eval_budget}")
        result = finisher.run(
            eval_budget=eval_budget,
            snapshot_path=snapshot,
            log_path=log_path,
        )
        verification = _verify_mask(
            objective,
            result.best_mask,
            renderer,
            scorer,
            gt_lstars,
            gt_poses,
        )
        row = {
            "estimator": estimator,
            "function_eval_budget_variance": eval_budget,
            "function_evals_variance": variance.function_evals,
            "variance_budget_contract": "B_includes_all_objective_calls",
            "variance_samples": variance.n_samples,
            "gradient_trace_variance": variance.trace_variance,
            "proposal_gain_variance": variance.proposal_gain_variance,
            "variance_budget_padding_evals": variance.budget_padding_evals,
            "mean_gradient": (
                None if variance.mean_gradient is None else variance.mean_gradient.tolist()
            ),
            "variance_wall_clock_s": variance.wall_clock_s,
            "function_eval_budget_search": eval_budget,
            "function_evals_search": result.function_evals,
            "search_budget_padding_evals": result.budget_padding_evals,
            "n_accepted": result.n_accepted,
            "final_mask": result.best_mask.tolist(),
            "start_s": result.start_s,
            "best_s": result.best_s,
            "delta_s": result.delta_s,
            "improvement_per_search_eval": -result.delta_s / eval_budget,
            "search_wall_clock_s": result.wall_clock_s,
            "verification": verification,
        }
        arm_rows.append(row)
        _atomic_json(arm_receipt_path, row)
        _log(
            f"arm={estimator} delta_S={result.delta_s:+.9e} "
            f"trace_var={variance.trace_variance} mask={result.best_mask.tolist()}"
        )

    _log("deriving exhaustive finite-support estimator moments (64 exact objective states)")
    exact_moments = exact_bernoulli_estimator_moments(
        objective, probabilities, ugc_tau=tau
    )
    for row in arm_rows:
        estimator = str(row["estimator"])
        row.setdefault("variance_budget_contract", "B_includes_all_objective_calls")
        if estimator in exact_moments:
            moments = exact_moments[estimator]
            row["exact_distribution_trace_variance"] = moments.trace_variance
            row["exact_distribution_coordinate_variance"] = (
                moments.coordinate_variance.tolist()
            )
            row["exact_distribution_mean_gradient"] = moments.mean_gradient.tolist()
            row["exact_distribution_probability_mass"] = moments.probability_mass
            row["exact_distribution_objective_states"] = moments.objective_states
        elif estimator == "exact_enumeration":
            row["exact_distribution_trace_variance"] = 0.0
        else:
            row["exact_distribution_trace_variance"] = None
        _atomic_json(OUT / f"receipt_{estimator}.json", row)

    exact_row = next(row for row in arm_rows if row["estimator"] == "exact_enumeration")
    stochastic = [row for row in arm_rows if row["estimator"] != "exact_enumeration"]
    winner = min(stochastic, key=lambda row: (row["best_s"], row["estimator"]))
    ugc_row = next(row for row in arm_rows if row["estimator"] == "ugc")
    es_row = next(row for row in arm_rows if row["estimator"] == "one_plus_one_es")
    disarm_row = next(row for row in arm_rows if row["estimator"] == "disarm")
    ugc_default = bool(
        ugc_row["best_s"] < es_row["best_s"] and ugc_row["best_s"] < disarm_row["best_s"]
    )
    verdict = (
        "UGC_WINS_ROUTE_TO_396_400"
        if ugc_default
        else "UGC_LOSES_INSTANCE_FORMULATION_SCOPED"
    )
    ugc_exact_variance = exact_moments["ugc"].trace_variance
    disarm_exact_variance = exact_moments["disarm"].trace_variance
    ugc_exact_variance_reduction_vs_disarm = (
        (disarm_exact_variance - ugc_exact_variance) / disarm_exact_variance
    )
    payload = {
        "schema": "ugc_terminal_polish_ab_receipt.v2",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lane_id": "lane_ugc_terminal_polish_ab_396_400_20260712",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "paid_dispatch": False,
        "live_run_mutation": False,
        "fixture": fixture,
        "candidate_rows": candidate_rows,
        "n_pairs_authority": objective.n_pairs,
        "active_support_k": k,
        "probabilities": probabilities.tolist(),
        "ugc_tau": tau,
        "eval_budget_per_variance_arm": eval_budget,
        "eval_budget_per_search_arm": eval_budget,
        "function_eval_budget_contract": "B includes every objective call, including references and padding",
        "arms": arm_rows,
        "exact_distribution_variance": {
            estimator: {
                "trace_variance": moments.trace_variance,
                "coordinate_variance": moments.coordinate_variance.tolist(),
                "mean_gradient": moments.mean_gradient.tolist(),
                "probability_mass": moments.probability_mass,
                "objective_states": moments.objective_states,
                "label": "DERIVED exhaustive distribution on exact 64-state objective",
            }
            for estimator, moments in exact_moments.items()
        },
        "ugc_exact_variance_reduction_vs_disarm": (
            ugc_exact_variance_reduction_vs_disarm
        ),
        "measured_stochastic_winner": winner["estimator"],
        "exact_enumeration_best_s": exact_row["best_s"],
        "ugc_default_route": ugc_default,
        "verdict": verdict,
        "verdict_scope": {
            "archive_sha256": fixture_sha,
            "candidate_pairs": list(CANDIDATE_PAIRS),
            "direction_grid": "28 columns x {+1,-1}",
            "probability_geometry": "three tau/2 boundary + three p=0.5 interior",
            "seed": SEED,
            "function_eval_budget": eval_budget,
            "scope_level": "instance/formulation",
        },
    }
    _atomic_json(OUT / "measurement_receipt.json", payload)
    _log(f"DONE winner={winner['estimator']} verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
