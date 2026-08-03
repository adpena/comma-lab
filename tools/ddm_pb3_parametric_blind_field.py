#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_pb3 — can a PARAMETRIC blind-set perturbation beat bp2's per-coordinate price?

bp2 closed the blind-set pose actuator at FORMULATION scope: the channel is real,
exactly seg-free, 116x-154x reach, and 2.32x under water because every coordinate and
sign must be shipped.  Its named survivor was a parametric shape.  This tool tests it.

TWO MODES.

  ceiling    SCORER-FREE.  Re-derives, from bp2's own n600 receipts, (a) the
             LINEARIZED FLOOR on d_pose reachable by ANY 1-LSB blind perturbation,
             (b) the alignment of bp2's own pixel-space step with the 6-dim target
             direction -- the mechanism of its failure -- and (c) the break-even
             byte table for a parametric payload.  Includes a real falsification
             test: no measured arm may sit BELOW the floor.

  alignment  Measures ``eta``, the fraction of the maximum first-order pose descent
             that a 6-coefficient RECEIVER-COMPUTABLE field realizes, against the
             TRUE blind gradient of the real frozen PoseNet.  ``eta`` is what turns
             the universal floor into this family's floor, so this is the arm's
             decisive number.  Two controls: a random-sign step (must be ~0) and a
             random 6-dim smooth basis of the same rank (isolates whether the
             interaction-matrix STRUCTURE carries the signal, or merely its rank).

PRE-REGISTERED FALSIFIER.  The parametric family closes at FORMULATION scope if the
measured ``eta`` of the fitted receiver-computable field is below the break-even
capture at the cheapest credible payload (7 scalars x 8 bits = 7 B/pair over all 600
pairs), i.e. if it cannot buy back its own bytes.

AXIS: [macOS-CPU advisory] NON-PROMOTABLE.  Frozen CPU-torch scorers on decoded
camera rasters, not ``upstream/evaluate.py`` on an archive.  score_claim=false.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Both roots are wired HERE rather than inherited from PYTHONPATH: the scorer import
# is the authority surface, and an env-dependent import path is exactly the kind of
# implicit contract that fails after 35 s of decode instead of at line 1.
for _root in (REPO_ROOT / "src", REPO_ROOT / "upstream"):
    if not _root.is_dir():
        raise RuntimeError(f"required root missing: {_root}")
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from tac.optimization.ddm_pb3_parametric_blind_field import (  # noqa: E402
    alignment_efficiency,
    delta_s_rate,
    fit_basis_coefficients,
    fit_max_alignment,
    linearized_pose_floor,
    payload_bytes,
    pose_contribution,
    pullback_to_blind,
    random_polynomial_saliency_fields,
    subset_index_bytes,
    vo_saliency_fields,
)

GRAD_MASS_TARGETS = (0.002, 0.01, 0.05, 0.15, 0.35, 0.7, 1.0)
AXIS = "[macOS-CPU advisory] frozen CPU-torch scorers; NON-PROMOTABLE"


def _bp2_tool():
    """Import bp2's tool as a module so the decode/scorer substrate is REUSED.

    Re-implementing the staging or the scorer wrappers would fork the authority
    surface; the whole point of the bp2 receipts is that both arms measure the same
    object with the same code.
    """
    import importlib.util

    path = REPO_ROOT / "tools" / "ddm_bp2_blind_warp_reach.py"
    spec = importlib.util.spec_from_file_location("ddm_bp2_tool", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ddm_bp2_tool"] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------- ceiling
def mode_ceiling(jsonl: Path) -> dict:
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{jsonl} is empty")
    pairs = [r["pair"] for r in rows]
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate pairs in the bp2 receipt — refusing to average them")

    d0 = np.array([r["d_pose_base"] for r in rows])
    g1 = np.array([r["grad_blind_l1"] for r in rows])
    best = np.array([r["d_pose_best"] for r in rows])
    floor = linearized_pose_floor(d0, g1)

    # FALSIFICATION: the floor is a bound, so no measured arm may sit below it.
    arm_keys = [f"d_pose_t{t}" for t in GRAD_MASS_TARGETS] + [
        "d_pose_full_desc",
        "d_pose_full_asc",
        "d_pose_random_sign_same_k",
    ]
    arms = np.stack([np.array([r[k] for r in rows]) for k in arm_keys])
    violations = int((arms < floor[None, :] - 1e-12).sum())
    informative = int((g1 < 2.0 * d0).sum())

    # The mechanism: cos(e, J delta) for bp2's own pixel-space step, at each size.
    # d_pose(t)/d_pose(0) = 1 - t + rho^2 with rho = ||J delta|| / ||e||, and
    # <e, J delta> = -(t/2) ||e||^2, so cos = -(t/2)/rho.  Recovered independently at
    # every step size: a drifting cos would mean the quadratic model is wrong.
    alignment = []
    for t in GRAD_MASS_TARGETS:
        dt = np.array([r[f"d_pose_t{t}"] for r in rows])
        kk = np.array([r[f"k_t{t}"] for r in rows], dtype=float)
        rho2 = dt / d0 - 1.0 + t
        ok = rho2 > 0
        rho = np.sqrt(np.clip(rho2, 1e-300, None))
        cos = -(t / 2.0) / rho
        alignment.append(
            {
                "grad_mass_target": t,
                "mean_k": float(kk.mean()),
                "median_rho_norm_ratio": float(np.median(rho[ok])),
                "median_cos_e_Jdelta": float(np.median(cos[ok])),
                "n_pairs_model_consistent": int(ok.sum()),
            }
        )

    base_c = pose_contribution(float(d0.mean()))
    ceil_c = pose_contribution(float(floor.mean()))
    bp2_c = pose_contribution(float(best.mean()))

    breakeven = []
    for bpp in (4, 7, 8, 12, 16, 24, 48):
        rate = delta_s_rate(payload_bytes(len(rows), 1, bpp * 8))
        need_d = (base_c - rate) ** 2 / 10.0
        breakeven.append(
            {
                "bytes_per_pair": bpp,
                "total_bytes": float(payload_bytes(len(rows), 1, bpp * 8)),
                "delta_s_rate": rate,
                "d_pose_needed": float(need_d),
                "pct_d_pose_reduction_needed": float(100.0 * (1.0 - need_d / d0.mean())),
                "pct_of_ceiling_gain_needed": float(100.0 * rate / (base_c - ceil_c)),
            }
        )

    # Tail: the ceiling's gain is extremely concentrated, so a SUBSET payload with a
    # colex index is cheaper still.  Priced against the same floor.
    reduction = d0 - floor
    order = np.argsort(-reduction)
    subsets = []
    for n_sel in (6, 30, 60, 150, len(rows)):
        idx_b = subset_index_bytes(n_sel, len(rows))
        total = payload_bytes(n_sel, 7, 8, index_bytes=idx_b)
        mean_after = (d0.sum() - reduction[order[:n_sel]].sum()) / len(rows)
        ds_pose = pose_contribution(float(mean_after)) - base_c
        ds_rate = delta_s_rate(total)
        subsets.append(
            {
                "n_pairs_corrected": n_sel,
                "index_bytes": float(idx_b),
                "total_bytes": float(total),
                "pct_of_ceiling_gain_captured": float(
                    100.0 * reduction[order[:n_sel]].sum() / reduction.sum()
                ),
                "delta_s_rate": ds_rate,
                "delta_s_pose_at_ceiling": float(ds_pose),
                "net_delta_s_at_ceiling": float(ds_rate + ds_pose),
            }
        )

    return {
        "schema": "ddm_pb3_ceiling.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "mode": "ceiling",
        "source_receipt": str(jsonl),
        "n_pairs": len(rows),
        "falsification_no_measured_arm_below_floor": {
            "cells_checked": int(arms.size),
            "violations": violations,
            "pairs_with_informative_floor": informative,
            "PASSED": violations == 0,
        },
        "population": {
            "mean_d_pose_base": float(d0.mean()),
            "median_d_pose_base": float(np.median(d0)),
            "max_d_pose_base": float(d0.max()),
            "mean_d_pose_bp2_argmin": float(best.mean()),
            "mean_d_pose_linearized_floor": float(floor.mean()),
            "frac_pairs_full_cancellation_reachable": float((g1 >= 2.0 * d0).mean()),
            "pose_contribution_base": base_c,
            "pose_contribution_bp2_argmin": bp2_c,
            "pose_contribution_ceiling": ceil_c,
            "delta_s_pose_bp2_argmin": float(bp2_c - base_c),
            "delta_s_pose_ceiling": float(ceil_c - base_c),
            "median_pct_reduction_bp2_argmin": float(100.0 * np.median((d0 - best) / d0)),
        },
        "bp2_step_alignment_with_6d_target": alignment,
        "breakeven_all_pairs": breakeven,
        "subset_payload_at_ceiling": subsets,
    }


# ----------------------------------------------------------------- alignment
def stratified_pairs(jsonl: Path, n: int) -> list[int]:
    """Pairs spanning the d_pose distribution, NOT a video-order prefix.

    bp2's own n600 overturned its prefix conclusion because the prefix was a 5.1x
    HARDER population than the whole (median d_pose 8.2e-4, max 0.77).  Any per-pair
    verdict on this skewed quantity must be stratified or it inherits that defect.
    Deterministic: sort by d_pose, take n evenly-spaced quantiles.
    """
    rows = [json.loads(x) for x in jsonl.read_text().splitlines() if x.strip()]
    order = sorted(rows, key=lambda r: r["d_pose_base"])
    picks = np.unique(np.linspace(0, len(order) - 1, n).round().astype(int))
    return sorted(int(order[i]["pair"]) for i in picks)


def mode_alignment(archive: Path, template: Path, stage: Path, n_pairs: int,
                   threads: int, densities: tuple[float, ...], seed: int,
                   pair_list: list[int] | None = None,
                   apply_densities: tuple[float, ...] = ()) -> dict:
    bp2 = _bp2_tool()
    from tac.optimization.ddm_bp2_blind_pose_actuator import v4d_pair_taps
    from tac.optimization.ddm_ll1_window_solve import blind_mask

    arch_dir = bp2.stage_receiver(stage, archive, template)
    from inflate_runner_v4d import Decoder  # type: ignore

    decoder = Decoder(arch_dir)
    scorers = bp2.Scorers(threads)
    blind = blind_mask()
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    want = set(pair_list) if pair_list else None
    limit = (max(pair_list) + 1) if pair_list else n_pairs
    for i, gt0, gt1 in bp2.gt_pair_stream(REPO_ROOT / "upstream" / "videos" / "0.mkv", limit):
        if want is not None and i not in want:
            continue
        gt = np.stack([gt0, gt1])
        f1 = decoder.f1(i)
        f0 = decoder.f0(i, f1)
        dp0, ds0 = scorers.distortion(gt, np.stack([f0, f1]))
        gt_out, gt_seg = scorers.pose_out(gt), scorers.seg_out(gt)
        if not (
            scorers.dpose_from(gt_out, np.stack([f0, f1])) == dp0
            and scorers.dseg_from(gt_seg, np.stack([f0, f1])) == ds0
        ):
            raise RuntimeError(f"pair {i}: cached-GT fast path drifted from the authority")
        grad_f0, dp0_diff = scorers.dpose_grad_wrt_frame0(gt, f0, f1)
        idx, w, _ = v4d_pair_taps(decoder, i)
        a = float(decoder.ab[i][0])
        from tac.optimization.ddm_bp2_blind_pose_actuator import adjoint_taps

        g_blind = (a * adjoint_taps(idx, w, grad_f0))[blind].ravel()

        basis = pullback_to_blind(idx, w, a, vo_saliency_fields(f0, decoder.K), blind)
        _, r2 = fit_basis_coefficients(g_blind, basis)
        c, eta_full = fit_max_alignment(g_blind, basis, seed=seed + i)
        phi = basis.T @ c

        # CONTROL 1: same rank, same smoothness class, same grad-I factor, RANDOM
        # quadratics instead of the interaction matrix.  Isolates the geometry.
        ctrl_basis = pullback_to_blind(
            idx, w, a, random_polynomial_saliency_fields(f0, decoder.K, rng), blind
        )
        _, r2_ctrl = fit_basis_coefficients(g_blind, ctrl_basis)
        c_ctrl, eta_ctrl_full = fit_max_alignment(g_blind, ctrl_basis, seed=seed + i)
        phi_ctrl = ctrl_basis.T @ c_ctrl

        row = {
            "pair": i,
            "d_pose_base": dp0,
            "d_seg_base": ds0,
            "diff_path_matches_authority": bool(abs(dp0_diff - dp0) <= 1e-5 * max(dp0, 1e-12)),
            "grad_blind_l1": float(np.abs(g_blind).sum()),
            "r2_vo_basis": r2,
            "r2_random_basis_control": r2_ctrl,
            "eta_vo_d1.0": eta_full,
            "eta_random_basis_d1.0": eta_ctrl_full,
        }
        for dens in densities:
            if dens == 1.0:
                continue
            eta, _ = alignment_efficiency(g_blind, phi, density=dens)
            eta_c, _ = alignment_efficiency(g_blind, phi_ctrl, density=dens)
            row[f"eta_vo_d{dens}"] = eta
            row[f"eta_random_basis_d{dens}"] = eta_c
        # CONTROL 2: random signs, full density.  Must be ~0 or the metric is broken.
        eta_rand, _ = alignment_efficiency(
            g_blind, rng.standard_normal(g_blind.size), density=1.0
        )
        row["eta_random_sign_control"] = eta_rand
        # ORACLE: phi == g gives delta == -sign(g), the unconstrained optimum, so eta
        # MUST be exactly 1.  A metric that does not return 1 on its own optimum is
        # measuring nothing — this control caught a sign error on its first run.
        eta_true, _ = alignment_efficiency(g_blind, g_blind, density=1.0)
        row["eta_true_sign_oracle"] = eta_true
        # THE DECISIVE ARM.  eta is a FIRST-ORDER statistic; it does not say what the
        # step actually achieves, because the realized ``J delta`` also carries
        # components in the five pose dimensions that are already correct -- which is
        # precisely how bp2's own step wasted 99.6% of its reach.  So APPLY the
        # parametric field at a sweep of densities (the shipped scale knob),
        # re-render f0 through the real receiver, and re-score on the canonical
        # unpatched scorer.  No model, no linearization: the measured d_pose.
        best_dp, best_dens = dp0, 0.0
        for dens in apply_densities:
            _, delta = alignment_efficiency(g_blind, phi, density=dens)
            pert = bp2._apply_blind_step(f1, blind, delta.astype(np.int16))
            dp = scorers.dpose_from(gt_out, np.stack([decoder.f0(i, pert), pert]))
            row[f"d_pose_applied_d{dens}"] = dp
            if dp < best_dp:
                best_dp, best_dens = dp, dens
        # ASCENT control at the best density: a channel that only ever goes down is
        # measuring noise, not steering.
        _, delta = alignment_efficiency(g_blind, -phi, density=max(best_dens, apply_densities[0]))
        pert_up = bp2._apply_blind_step(f1, blind, delta.astype(np.int16))
        row["d_pose_applied_ascent_control"] = scorers.dpose_from(
            gt_out, np.stack([decoder.f0(i, pert_up), pert_up])
        )
        row["d_seg_under_ascent_control"] = scorers.dseg_from(
            gt_seg, np.stack([decoder.f0(i, pert_up), pert_up])
        )
        row["d_seg_identical_under_parametric_step"] = bool(
            row["d_seg_under_ascent_control"] == ds0
        )
        row["d_pose_applied_best"] = best_dp
        row["best_density"] = best_dens
        row["delta_d_pose_applied_best"] = best_dp - dp0
        rows.append(row)
        print(
            f"  pair {i:3d}  d_pose {dp0:.6f}  eta_vo {eta_full:+.4f}  "
            f"eta_ctrl {eta_ctrl_full:+.4f}  r2 {r2:.2e}  "
            f"APPLIED {best_dp:.6f} (dens {best_dens})  d{best_dp - dp0:+.3e}",
            flush=True,
        )

    def agg(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    d0 = np.array([r["d_pose_base"] for r in rows])
    g1 = np.array([r["grad_blind_l1"] for r in rows])
    applied = np.array([r["d_pose_applied_best"] for r in rows])
    base_c = pose_contribution(float(d0.mean()))
    per_density = []
    for dens in densities:
        eta = np.array([r[f"eta_vo_d{dens}"] for r in rows])
        fl = linearized_pose_floor(d0, g1, capture=np.clip(eta, 0.0, None))
        per_density.append(
            {
                "density": dens,
                "mean_eta_vo": float(eta.mean()),
                "median_eta_vo": float(np.median(eta)),
                "mean_eta_random_basis": agg(f"eta_random_basis_d{dens}"),
                "mean_d_pose_floor_at_this_eta": float(fl.mean()),
                "delta_s_pose_at_this_eta": float(pose_contribution(float(fl.mean())) - base_c),
            }
        )

    return {
        "schema": "ddm_pb3_alignment.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "mode": "alignment",
        "archive": str(archive),
        "archive_sha256": bp2._sha256(archive),
        "n_pairs": len(rows),
        "seed": seed,
        "pair_selection": "stratified_by_d_pose" if pair_list else "video_order_prefix",
        "pairs": [r["pair"] for r in rows],
        "controls": {
            "mean_eta_random_sign": agg("eta_random_sign_control"),
            "mean_eta_true_sign_oracle": agg("eta_true_sign_oracle"),
            "oracle_is_unity": bool(
                all(abs(r["eta_true_sign_oracle"] - 1.0) < 1e-9 for r in rows)
            ),
            "mean_r2_vo_basis": agg("r2_vo_basis"),
            "mean_r2_random_basis": agg("r2_random_basis_control"),
            "all_diff_path_matches_authority": all(
                r["diff_path_matches_authority"] for r in rows
            ),
        },
        "MEASURED_applied": {
            "note": (
                "d_pose re-scored on the canonical unpatched scorer through the REAL "
                "re-rendered f0. No linearization. This subset is stratified, so its "
                "mean is NOT the n600 mean; read the per-pair relative reduction."
            ),
            "mean_d_pose_base_subset": float(d0.mean()),
            "mean_d_pose_applied_subset": float(applied.mean()),
            "median_relative_reduction": float(np.median((d0 - applied) / d0)),
            "mean_relative_reduction": float(np.mean((d0 - applied) / d0)),
            "frac_pairs_improved": float((applied < d0).mean()),
            "mean_d_pose_ascent_control": float(
                np.mean([r["d_pose_applied_ascent_control"] for r in rows])
            ),
            "d_seg_identical_all_pairs": all(
                r["d_seg_identical_under_parametric_step"] for r in rows
            ),
        },
        "per_density": per_density,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("ceiling", "alignment"))
    ap.add_argument(
        "--bp2-jsonl",
        type=Path,
        default=REPO_ROOT / "reports" / "ddm_bp2" / "reach_n600.jsonl",
        help="ceiling mode: bp2's per-pair n600 receipt",
    )
    ap.add_argument("--archive", type=Path, help="alignment mode: v4d archive.zip")
    ap.add_argument(
        "--template",
        type=Path,
        default=Path(
            "/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/pfs1"
        ),
    )
    ap.add_argument("--stage", type=Path, help="alignment mode: scratch dir")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--densities", type=float, nargs="+", default=[1.0, 0.25, 0.05, 0.01])
    ap.add_argument(
        "--apply-densities",
        type=float,
        nargs="+",
        default=[0.002, 0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0],
        help="alignment: densities at which the parametric step is APPLIED and re-scored",
    )
    ap.add_argument(
        "--stratified",
        type=int,
        default=0,
        help="alignment: sample this many pairs across the bp2 d_pose distribution",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.mode == "ceiling":
        result = mode_ceiling(args.bp2_jsonl)
    else:
        if args.archive is None or args.stage is None:
            ap.error("alignment mode requires --archive and --stage")
        result = mode_alignment(
            args.archive,
            args.template,
            args.stage,
            args.pairs,
            args.threads,
            tuple(args.densities),
            args.seed,
            stratified_pairs(args.bp2_jsonl, args.stratified) if args.stratified else None,
            tuple(args.apply_densities),
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
