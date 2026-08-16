#!/usr/bin/env python3
"""ddm_ps1u -- the UNCAPPED, realized-acceptance, OFFENSIVE pose solve on hv1.

WHAT IS NEW HERE (the single declared mechanism change)
-------------------------------------------------------
``ddm_qs1_frame0_schur_coupled_solve`` already owns the exact-object frame-0
actuator at optimal form: it renders CP135's real signed-int12 frame-0 carrier,
scores it through the frozen CPU-torch PoseNet, and descends on the int12
lattice with an ALREADY-UNCAPPED stop (``strict_descent``: ``while True``,
terminate on one full non-improving singleton pass).  Its objective, however, is

    ||pose6(candidate_frame0, EDITED frame_1) - pose6(base_frame0, base_frame_1)||^2

i.e. CANCELLATION of the pose leak a frame-1 seg edit introduced (qs1:718 uses
the GT only to REPORT, never to steer).  That is a defensive use.

This module changes exactly one thing -- the TARGET:

    ||pose6(candidate_frame0, BASE frame_1) - gt_pose6||^2

which is the pair's own contribution to ``d_pose``.  Same actuator, same
renderer, same frozen scorer, same lattice, same uncapped stop rule.  Nothing
else is reduced: no proxy, no surrogate, no fitted model, no held-out
generalization step (there is no fit -- every pair is solved on its own object
and accepted only on its own REALIZED score).

WHY THIS IS NOT ddm_pk4
-----------------------
pk4 FIT a low-knot linear frame-0 overlay across pairs and failed heldout at
every rate.  This is a per-pair EXACT SOLVE with realized acceptance: the
quantity that decides acceptance is the pair's realized d_pose after rendering
the int12 candidate, not a modeled projection.  pk4's ceiling therefore does not
bind here (its own verdict says so).

REALIZED ACCEPTANCE IS STRUCTURAL
---------------------------------
Every candidate is an int12 code vector, is RENDERED through the exact CP135
receiver surface, and is scored by the frozen CPU-torch PoseNet.  A pair can
never end worse than it started, and the reported endpoint is the shipped-lattice
value -- not an off-lattice optimum that rounding would discard (the ddm_pg1
lesson: the shipped off-lattice (a,b) solve could ship a pair worse than doing
nothing).

THE #850 QUESTION, ASKED IN THIS VEHICLE'S OWN TERMS
----------------------------------------------------
The corpus cap is real and cited: ``ddm_su2_qa43_tail_solver.py:148`` refuses any
``relinearizations`` outside ``(2, 3)``; ``:1072`` then runs ``range(cap)`` with
no convergence test.  This module runs relinearization SWEEPS to a genuine
convergence proof and records the objective after EVERY sweep, so the value a
2-sweep and a 3-sweep cap would have shipped is read directly off the same
trajectory as the converged endpoint.  The descent curve is the payload.

AXIS
----
``[macOS-CPU advisory frozen CPU-torch PoseNet, realized through the exact CP135
receiver render]``.  ``score_claim=false``, ``promotable=false``.  The local
instrument reproduces the hv1 base advisory row's d_pose (1.474653e-04 vs the
mp2 relay's 1.4747e-04) and is 21.42x the CUDA value 6.885642960696714e-06 --
a gap this module reports and never launders.  Only ``upstream/evaluate.py`` on
byte-closed archive bytes is a score.

RESUMABILITY (P0)
-----------------
One append-only, fsynced JSONL row per pair; re-run with the same ``--out`` to
continue.  Nothing is held only in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

AXIS = (
    "[macOS-CPU advisory frozen CPU-torch PoseNet; exact CP135 receiver render] "
    "NON-PROMOTABLE"
)
SCORE_CLAIM = False

PAIR_COUNT = 600
DIMENSIONS = 12
POSE_DIMENSIONS = 6
INT12_MIN = -2048
INT12_MAX = 2047

# Custody pins (verified by this arm, 2026-08-16).
HV1_ARCHIVE_SHA256 = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
HV1_ARCHIVE_BYTES = 182_759
CUDA_BASE_DPOSE = 6.885642960696714e-06
ADVISORY_BASE_DPOSE = 1.474653494795297e-04
RATE_S_PER_BYTE = 25.0 / 37_545_489


class PS1UError(RuntimeError):
    """A retained input, actuator binding, or solve invariant differed."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": _sha256_bytes(payload)}


def _qs1_paths() -> dict[str, Any]:
    from experiments import ddm_qs1_frame0_schur_coupled_solve as qs1

    return {"base_pose": qs1.CP135_BASE_POSE, "gt_pose": qs1.GT_POSE}


def top_mass_pairs(n: int, base_pose: np.ndarray, gt_pose: np.ndarray) -> np.ndarray:
    """The n pairs carrying the most d_pose mass, by retained base-vs-GT error.

    This is a DELIBERATE non-random selection: the candidate's byte budget is spent
    where the pose mass is, so the selection rule must be the mass itself. It is
    reproducible from retained vectors alone (no scorer call) and is recorded with
    the realized mass share it captures."""
    if not 1 <= n <= PAIR_COUNT:
        raise PS1UError("top-mass size must lie in [1, 600]")
    per_pair = np.mean(
        np.square(base_pose.astype(np.float64) - gt_pose.astype(np.float64)), axis=1
    )
    order = np.argsort(-per_pair, kind="stable")[:n]
    return np.sort(order).astype(np.int32)


def seeded_pair_subset(n: int, seed: int) -> np.ndarray:
    """A SEEDED RANDOM subset -- never a prefix (m88/m96: a prefix of a skewed
    per-pair quantity is a different population, and the bias is 2.5-4.2x
    ANTI-conservative on the pose axis specifically)."""
    if not 1 <= n <= PAIR_COUNT:
        raise PS1UError("subset size must lie in [1, 600]")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(PAIR_COUNT, size=n, replace=False)).astype(np.int32)


class Instrument:
    """The exact CP135 frame-0 actuator + frozen CPU-torch PoseNet."""

    def __init__(self, *, batch: int = 8) -> None:
        from experiments import ddm_qs1_frame0_schur_coupled_solve as qs1

        self._qs1 = qs1
        self.batch = int(batch)
        self.surface, self.surface_pins = qs1.CP135Surface.load()
        self.posenet = qs1.load_posenet()
        self.raw = np.memmap(
            qs1.CP135_RAW, dtype=np.uint8, mode="r"
        ).reshape(2 * PAIR_COUNT, qs1.CAMERA_H, qs1.CAMERA_W, 3)
        self.base_pose = np.load(qs1.CP135_BASE_POSE, allow_pickle=False)
        self.gt_pose = np.load(qs1.GT_POSE, allow_pickle=False)
        if self.base_pose.shape != (PAIR_COUNT, POSE_DIMENSIONS):
            raise PS1UError("retained base pose geometry differs")
        if self.gt_pose.shape != self.base_pose.shape:
            raise PS1UError("retained GT pose geometry differs")
        if self.surface.codes.shape != (PAIR_COUNT, DIMENSIONS):
            raise PS1UError("CP135 carrier code geometry differs")
        self.evals = 0

    def assert_receiver_identity(self, pair: int) -> int:
        """The rendered base frame-0 MUST equal the retained receiver raw."""
        rendered = self.surface.render(self.surface.codes[pair : pair + 1], pair)[0]
        mismatch = int(np.count_nonzero(rendered != np.asarray(self.raw[2 * pair])))
        if mismatch:
            raise PS1UError(
                f"pair {pair}: rendered frame-0 is not the retained receiver raw "
                f"({mismatch} values differ)"
            )
        return mismatch

    def objective(self, codes: np.ndarray, pair: int) -> tuple[np.ndarray, np.ndarray]:
        """Realized per-pair d_pose for each candidate code vector.

        Renders on the int12 lattice, pairs with the UNEDITED frame_1, and scores
        through the frozen PoseNet.  Returns (pose6 vectors, objectives)."""
        values = np.atleast_2d(np.asarray(codes, dtype=np.int32))
        if values.shape[1] != DIMENSIONS:
            raise PS1UError(f"candidate geometry differs: {values.shape}")
        if np.any(values < INT12_MIN) or np.any(values > INT12_MAX):
            raise PS1UError("candidate exceeds signed-int12")
        master = np.asarray(self.raw[2 * pair + 1])
        target = self.gt_pose[pair].astype(np.float64)
        vectors: list[np.ndarray] = []
        for first in range(0, len(values), self.batch):
            chunk = values[first : first + self.batch]
            slaves = self.surface.render(chunk, pair)
            inputs = np.stack(
                (slaves, np.repeat(master[None], len(chunk), axis=0)), axis=1
            )
            vectors.append(self._qs1.pose_vectors(self.posenet, inputs))
            self.evals += len(chunk)
        pose = np.concatenate(vectors, axis=0)
        objectives = np.mean(np.square(pose.astype(np.float64) - target[None]), axis=1)
        return pose, objectives


def _singleton_candidates(codes: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for dimension in range(DIMENSIONS):
        for delta in (-1, 1):
            candidate = codes.copy()
            candidate[dimension] += delta
            if INT12_MIN <= candidate[dimension] <= INT12_MAX:
                out.append(candidate)
    return out


def strict_descent(
    instrument: Instrument,
    pair: int,
    codes: np.ndarray,
    value: float,
    *,
    trace: list[dict[str, Any]],
    sweep: int,
    max_passes: int,
) -> tuple[np.ndarray, float, str, int]:
    """Uncapped integer coordinate descent; stop on one full non-improving pass.

    ``max_passes`` is a RUNAWAY GUARD, not a tuning cap: reaching it is recorded
    as ``pass_guard`` so a truncated pair can never be silently read as
    converged (the #850 defect this arm exists to not repeat)."""
    current = np.asarray(codes, dtype=np.int32).copy()
    best = float(value)
    passes = 0
    while passes < max_passes:
        candidates = _singleton_candidates(current)
        _, objectives = instrument.objective(np.stack(candidates), pair)
        index = int(min(range(len(candidates)), key=lambda i: (objectives[i], i)))
        if not float(objectives[index]) < best:
            return current, best, "descent_converged", passes
        current = candidates[index]
        best = float(objectives[index])
        passes += 1
        trace.append(
            {
                "stage": "descent",
                "sweep": sweep,
                "pass": passes,
                "objective": best,
                "evals": instrument.evals,
            }
        )
    return current, best, "pass_guard", passes


def solve_pair(
    instrument: Instrument,
    pair: int,
    *,
    max_sweeps: int,
    max_passes: int,
    rel_tol: float,
    damping: float,
    max_code_step: float,
    neighbour_dims: int,
    neighbour_radius: int,
) -> dict[str, Any]:
    """One pair, solved to a convergence PROOF with realized acceptance."""
    solver = instrument._qs1._load_module(
        "ps1u_joint_pose_solve", instrument._qs1.JOINT_SOLVER_SOURCE
    )
    mismatch = instrument.assert_receiver_identity(pair)
    started = time.time()
    base_codes = instrument.surface.codes[pair].copy()
    base_pose, base_objectives = instrument.objective(base_codes[None], pair)
    base_objective = float(base_objectives[0])
    retained_objective = float(
        np.mean(
            np.square(
                instrument.base_pose[pair].astype(np.float64)
                - instrument.gt_pose[pair].astype(np.float64)
            )
        )
    )
    trace: list[dict[str, Any]] = [
        {"stage": "base", "sweep": 0, "pass": 0, "objective": base_objective, "evals": instrument.evals}
    ]
    sweep_objectives: list[float] = [base_objective]

    current = base_codes.copy()
    value = base_objective
    stop = "sweep_cap"
    sweeps_used = 0
    for sweep in range(1, max_sweeps + 1):
        before = value
        # --- relinearize: central-difference Jacobian on the int12 lattice ---
        jac_codes = []
        for dimension in range(DIMENSIONS):
            for delta in (-1, 1):
                candidate = current.copy()
                candidate[dimension] += delta
                if not INT12_MIN <= candidate[dimension] <= INT12_MAX:
                    raise PS1UError(
                        f"pair {pair}: coefficient {dimension} is at an int12 endpoint"
                    )
                jac_codes.append(candidate)
        jac_pose, _ = instrument.objective(np.stack(jac_codes), pair)
        jacobian = np.empty((POSE_DIMENSIONS, DIMENSIONS), dtype=np.float64)
        for dimension in range(DIMENSIONS):
            minus = jac_pose[2 * dimension].astype(np.float64)
            plus = jac_pose[2 * dimension + 1].astype(np.float64)
            jacobian[:, dimension] = (plus - minus) / 2.0
        current_pose, _ = instrument.objective(current[None], pair)
        residual = current_pose[0].astype(np.float64) - instrument.gt_pose[pair].astype(
            np.float64
        )
        # --- damped LS step, quantized onto the shipped lattice ---
        solve = solver.solve_damped_least_squares(
            jacobian, -residual, damping=damping, max_code_step=max_code_step
        )
        centre = solver.quantize_int12_update(current, solve.update)
        active = solver.rank_neighbour_dimensions(jacobian, solve.update, neighbour_dims)
        neighbourhood = solver.nearby_int12_candidates(
            current, centre, active_dimensions=active, radius=neighbour_radius
        )
        pool = [current.copy(), *[np.asarray(c, dtype=np.int32) for c in neighbourhood]]
        pool = [np.clip(c, INT12_MIN, INT12_MAX).astype(np.int32) for c in pool]
        _, pool_objectives = instrument.objective(np.stack(pool), pair)
        index = int(min(range(len(pool)), key=lambda i: (pool_objectives[i], i)))
        if float(pool_objectives[index]) < value:
            current = pool[index]
            value = float(pool_objectives[index])
        trace.append(
            {"stage": "gn", "sweep": sweep, "pass": 0, "objective": value, "evals": instrument.evals}
        )
        # --- uncapped singleton descent to a non-improving pass ---
        current, value, descent_stop, passes = strict_descent(
            instrument, pair, current, value, trace=trace, sweep=sweep, max_passes=max_passes
        )
        sweep_objectives.append(value)
        sweeps_used = sweep
        if descent_stop == "pass_guard":
            stop = "pass_guard"
            break
        gain = (before - value) / before if before > 0 else 0.0
        if value >= before:
            stop = "sweep_no_improvement"
            break
        if gain <= rel_tol:
            stop = "sweep_relative_gain_below_tol"
            break
    final_pose, final_objectives = instrument.objective(current[None], pair)
    final_objective = float(final_objectives[0])
    if final_objective > base_objective:
        raise PS1UError(
            f"pair {pair}: realized acceptance violated ({final_objective} > {base_objective})"
        )
    delta = (current.astype(np.int64) - base_codes.astype(np.int64)).astype(np.int32)

    def at_cap(k: int) -> float:
        """What a k-sweep cap would have shipped, off this same trajectory."""
        return float(sweep_objectives[min(k, len(sweep_objectives) - 1)])

    return {
        "schema": "ddm_ps1u_uncapped_pose_solve.v1",
        "pair": int(pair),
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "promotion_eligible": False,
        "receiver_identity_mismatched_values": mismatch,
        "base_objective": base_objective,
        "retained_base_objective": retained_objective,
        "base_instrument_drift": abs(base_objective - retained_objective),
        "final_objective": final_objective,
        "relative_reduction": (base_objective - final_objective) / base_objective
        if base_objective > 0
        else 0.0,
        "sweep_objectives": sweep_objectives,
        "objective_at_cap_2": at_cap(2),
        "objective_at_cap_3": at_cap(3),
        "gain_past_cap_3": (at_cap(3) - final_objective) / at_cap(3)
        if at_cap(3) > 0
        else 0.0,
        "sweeps_used": sweeps_used,
        "stop_reason": stop,
        "base_codes": base_codes.tolist(),
        "final_codes": current.tolist(),
        "code_delta": delta.tolist(),
        "code_delta_absmax": int(np.abs(delta).max()),
        "code_delta_nonzero": int(np.count_nonzero(delta)),
        "base_pose6": base_pose[0].astype(float).tolist(),
        "final_pose6": final_pose[0].astype(float).tolist(),
        "gt_pose6": instrument.gt_pose[pair].astype(float).tolist(),
        "scorer_evals": int(instrument.evals),
        "wall_seconds": time.time() - started,
    }


def run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"rows_shard{args.shard:02d}of{args.shards:02d}.jsonl"
    done: set[int] = set()
    if rows_path.is_file():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                done.add(int(json.loads(line)["pair"]))
    if args.top_mass:
        base_pose = np.load(_qs1_paths()["base_pose"], allow_pickle=False)
        gt_pose = np.load(_qs1_paths()["gt_pose"], allow_pickle=False)
        pairs = top_mass_pairs(args.top_mass, base_pose, gt_pose)
    elif args.all_pairs:
        pairs = np.arange(PAIR_COUNT, dtype=np.int32)
    else:
        pairs = seeded_pair_subset(args.n_pairs, args.seed)
    pairs = pairs[args.shard :: args.shards]
    todo = [int(p) for p in pairs if int(p) not in done]
    print(
        f"[ps1u] {AXIS}\n[ps1u] shard {args.shard}/{args.shards} "
        f"pairs={len(pairs)} done={len(done)} todo={len(todo)}",
        flush=True,
    )
    if not todo:
        return 0
    instrument = Instrument(batch=args.batch)
    codes_dir = out / "retained/codes"
    with rows_path.open("a") as handle:
        for count, pair in enumerate(todo, start=1):
            row = solve_pair(
                instrument,
                pair,
                max_sweeps=args.max_sweeps,
                max_passes=args.max_passes,
                rel_tol=args.rel_tol,
                damping=args.damping,
                max_code_step=args.max_code_step,
                neighbour_dims=args.neighbour_dims,
                neighbour_radius=args.neighbour_radius,
            )
            row["final_codes_record"] = _atomic_write_npy(
                codes_dir / f"pair_{pair:04d}_final_codes.int32.npy",
                np.asarray(row["final_codes"], dtype=np.int32),
            )
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(
                f"[ps1u] pair {pair:3d} ({count}/{len(todo)}) "
                f"{row['base_objective']:.6e} -> {row['final_objective']:.6e} "
                f"({100 * row['relative_reduction']:+.2f}%) sweeps={row['sweeps_used']} "
                f"stop={row['stop_reason']} |dcode|max={row['code_delta_absmax']} "
                f"past_cap3={100 * row['gain_past_cap_3']:.2f}% {row['wall_seconds']:.1f}s",
                flush=True,
            )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-pairs", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--all-pairs", action="store_true")
    ap.add_argument("--top-mass", type=int, default=0,
                    help="solve the N pairs carrying the most d_pose mass (candidate mode)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-sweeps", type=int, default=24)
    ap.add_argument("--max-passes", type=int, default=200)
    ap.add_argument("--rel-tol", type=float, default=1e-3)
    ap.add_argument("--damping", type=float, default=0.01)
    ap.add_argument("--max-code-step", type=float, default=32.0)
    ap.add_argument("--neighbour-dims", type=int, default=3)
    ap.add_argument("--neighbour-radius", type=int, default=2)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
