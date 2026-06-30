#!/usr/bin/env python3
"""RL ANNULUS-UNLOCK GATE — the $0, CPU-authority, GPU-free make-or-break test.

QUESTION (the only thing that justifies building the RL lab):
    Does optimizing the EXACT non-differentiable d_seg reward DIRECTLY move a *stalled* annulus
    residual that the smooth-surrogate gradient has plateaued on?

WHY THIS IS THE RIGHT TEST (deep-math, `DERIVED` from the convergence memo):
    d_seg is a 0-1 argmax-disagreement loss whose gradient is a Dirac on a measure-zero codim-1
    contour (the annulus). The trained smooth surrogate's gradient mass on the boundary VANISHES as
    tau anneals -> the long flat tail (the Muon arm's d_seg ~0.00425 plateau). The warm-start ckpt
    IS the gradient's converged point. So if a gradient-FREE search (CEM) finds a per-pair code that
    lowers the EXACT through-R d_seg from there, that is BY DEFINITION a move the plateaued gradient
    could not make -> RL/policy optimization of the exact reward is justified.

ACTION SPACE: the per-pair FiLM code (the manipulable level-set knob). d_seg is scored on frame1
    only (SegNet last-frame argmax), so we perturb code row ``2*pi+1`` and hold the shared decoder
    weights + frame0 FIXED. This isolates the d_seg lever and measures its d_pose cost as a
    constraint (pose rides the SOLVED stored sidecar; w_pose=0 in this witness).

NO-FAKE INVARIANTS (binding):
  * The REWARD is the EXACT through-R frozen **CPU-authority** SegNet argmax d_seg
    (``cpu_verdict_d_seg_batch`` on ``_torch_R_to_camera_uint8`` renders) — the SAME codepath the
    trainer's ``realized_verdict()`` uses. NO MLX, NO MPS, NO proxy. (CLAUDE.md "MPS never authority".)
  * The render is the numpy deploy-faithful ONE CODEPATH (``levelset_rgb_forward_numpy``) — the same
    forward byte-close/inflate use. The front-end (curvelet bank + self-orient dir feats) is rebuilt
    EXACTLY from the ckpt ``__cfg_*``/``__bank_*`` provenance.
  * ENV-FIDELITY GATE: a random-pair sample's base d_seg must reproduce the Muon arm's realized
    verdict (~0.00425). If it does not, the env is broken and the run REFUSES (a broken reward is a
    fake reward).
  * Read-only on the warm-start ckpt; does not touch the live run. CPU-only -> does not contend the
    GPU the live Muon arm owns. Resumable, seeded, --min-free-gb guarded.

MEANS != ENDS: this moves NO score. Output is an ADVISORY distortion go/no-go on n=few pairs (NOT a
    600-row contest score; rate not measured here). Pointer UNMOVED contest-CPU 0.19110.

Anchors: `.omx/research/rl_lab_levelset_scoping_20260630T160101Z.md` (design) ·
    `project_rl_lab_levelset_exact_dseg_reward_direction_20260630` (spec).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "experiments"), str(_REPO / "src"), str(_REPO / "upstream")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _utc() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _refuse_tmp(path: Path) -> None:
    s = str(path.resolve())
    if any(t in s + "/" for t in _FORBIDDEN_TMP):
        raise ValueError(f"refusing transient /tmp evidence path: {path} (CLAUDE.md durable-evidence rule)")


def _atomic_write_json(path: Path, obj: dict) -> None:
    _refuse_tmp(path)
    import os
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warm-start", type=Path,
                    default=_REPO / "experiments/results/levelset_thetastar_muon_arm/levelset_witness_ema_BEST.npz")
    ap.add_argument("--gt-cache", type=Path,
                    default=_REPO / "experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz")
    ap.add_argument("--num-pairs", type=int, default=200)
    ap.add_argument("--n-sanity", type=int, default=12, help="random pairs for the env-fidelity gate")
    ap.add_argument("--n-candidate", type=int, default=20, help="candidate stalled pairs to measure base d_seg on")
    ap.add_argument("--n-stalled", type=int, default=6, help="worst-residual pairs to run CEM on")
    ap.add_argument("--cem-pop", type=int, default=16)
    ap.add_argument("--cem-gens", type=int, default=8)
    ap.add_argument("--cem-sigma", type=float, default=0.08, help="initial code-perturbation std (relative to code std)")
    ap.add_argument("--elite-frac", type=float, default=0.25)
    ap.add_argument("--cem-sigma-floor-frac", type=float, default=0.5,
                    help="sigma never shrinks below this fraction of the initial (anti-collapse -> no false-RED)")
    ap.add_argument("--cem-explore-frac", type=float, default=0.34,
                    help="fraction of each generation drawn from the WIDE fixed init sigma around base (keeps exploring)")
    ap.add_argument("--reorient-iters", type=int, default=4, help="self-orient dir-feats fixed-point iterations")
    ap.add_argument("--success-rel", type=float, default=0.10, help="GREEN if mean relative d_seg drop >= this")
    ap.add_argument("--torch-threads", type=int, default=4, help="cap to not starve the live arm's CPU verdict")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    out = args.out or (_REPO / f"experiments/results/rl_annulus_unlock_gate_{_utc()}")
    out.mkdir(parents=True, exist_ok=True)
    _refuse_tmp(out)

    free_gb = shutil.disk_usage(out).free / 1e9
    if free_gb < args.min_free_gb:
        print(json.dumps({"stage": "preflight", "REFUSE": "low_disk", "free_gb": round(free_gb, 1),
                          "min_free_gb": args.min_free_gb}), flush=True)
        return 2

    import torch
    torch.set_num_threads(int(args.torch_threads))

    # ---- import the trainer's EXACT authority codepaths (reuse, do NOT reimplement) ----------------
    from train_witness_realized_through_R_mlx import (  # noqa: E402
        _build_render_coords, _torch_R_to_camera_uint8,
        cpu_verdict_d_seg_batch, cpu_verdict_d_pose_batch, load_gt_from_cache,
    )
    from tac.boundary_math.lever_b_generator import self_orientation_directional_feats  # noqa: E402
    from tac.boundary_math.lever_b_levelset_generator import (  # noqa: E402
        CurveletBankConfig, curvelet_directional_B, curvelet_feats, int8_dequant_params,
        levelset_rgb_forward_numpy,
    )

    # ---- load ckpt + config provenance (rebuild the EXACT front-end) ------------------------------
    # SNAPSHOT the warm-start ckpt into the out-dir (the live Muon arm keeps OVERWRITING BEST.npz as it
    # descends; a crash+resume must load the SAME ckpt the done-pairs used -> deterministic/consistent).
    snap_ckpt = out / "warm_start_snapshot.npz"
    if not snap_ckpt.exists():
        shutil.copy2(args.warm_start, snap_ckpt)
    z = np.load(snap_ckpt, allow_pickle=False)
    cfg = {k[len("__cfg_"):]: z[k] for k in z.files if k.startswith("__cfg_")}
    bnk = {k[len("__bank_"):]: z[k] for k in z.files if k.startswith("__bank_")}
    deploy = int8_dequant_params({k: np.asarray(z[k], np.float32) for k in z.files if not k.startswith("__")})
    render_h, render_w = (int(x) for x in z["__render_hw"])
    n_hidden = int(cfg["n_hidden"]); hidden_dim = int(cfg["hidden_dim"])
    softmax_temp = float(cfg["softmax_temp"]); activation = str(cfg["activation"])
    chroma = bool(int(cfg["chroma"])); in_feat = int(cfg["in_feat"])
    self_orient = bool(int(cfg.get("self_orient", 0)))
    n_dir_freqs = int(cfg.get("n_dir_freqs", 2))
    freq_across = float(cfg.get("freq_across", 32.0)); freq_along = float(cfg.get("freq_along", 4.0))
    akw = dict(wire_w0=float(cfg["wire_w0"]), wire_s0=float(cfg["wire_s0"]),
               hosc_beta=float(cfg["hosc_beta"]), hosc_omega=float(cfg["hosc_omega"]))
    fwd_kw = dict(n_hidden=n_hidden, hidden_dim=hidden_dim, n_classes=5, activation=activation,
                  softmax_temp=softmax_temp, chroma=chroma, **akw)

    coords_np = _build_render_coords(render_h, render_w)
    bank = CurveletBankConfig(n_scales=int(bnk["n_scales"]), n_orient0=int(bnk["n_orient0"]),
                              f0=float(bnk["f0"]), base=float(bnk["base"]), n_iso=int(bnk["n_iso"]))
    B = curvelet_directional_B(bank, max_freq=float(cfg["max_bank_freq"]))
    curv_feats_np = curvelet_feats(coords_np, B).astype(np.float32)
    print(json.dumps({"stage": "front_end", "curvelet_cols": int(curv_feats_np.shape[1]),
                      "self_orient": self_orient, "render_hw": [render_h, render_w],
                      "code_rows": int(deploy["code"].shape[0]), "muon_epoch": int(z["__epoch"])}), flush=True)

    gt, seg_cpu, posenet_cpu = load_gt_from_cache(args.gt_cache, args.num_pairs)
    P = min(args.num_pairs, int(deploy["code"].shape[0]) // 2)

    def _render_frame(pi: int, fk: int, code_row: np.ndarray, dir_feats: np.ndarray | None) -> np.ndarray:
        """ONE CODEPATH render -> R -> camera uint8 (the authority frame for the CPU scorer)."""
        feats = curv_feats_np if not self_orient else np.concatenate([curv_feats_np, dir_feats], axis=-1).astype(np.float32)
        if feats.shape[1] != in_feat:
            raise ValueError(f"feat width {feats.shape[1]} != ckpt in_feat {in_feat} (front-end mismatch)")
        rgb, _phi = levelset_rgb_forward_numpy(deploy, feats, code_row, **fwd_kw)
        return _torch_R_to_camera_uint8(rgb.reshape(render_h, render_w, 3))

    def _argmax_frame1(pi: int, dir_feats: np.ndarray | None) -> np.ndarray:
        feats = curv_feats_np if not self_orient else np.concatenate([curv_feats_np, dir_feats], axis=-1).astype(np.float32)
        _rgb, phi = levelset_rgb_forward_numpy(deploy, feats, deploy["code"][2 * pi + 1], **fwd_kw)
        return phi.argmax(-1).reshape(render_h, render_w).astype(np.int64)

    def _reconstruct_dir_feats(pi: int) -> np.ndarray | None:
        """Self-orient fixed-point: dir feats are NOT stored (O(GBs)); reconstruct from the deploy's
        OWN frame1 argmax (start zeros -> render -> argmax -> dir -> ... converge). This is exactly
        what the trainer's reorient does; iterating to a stable argmax reconstructs what the Muon arm
        was using. Held FIXED during the CEM = the trainer's within-reorient-window behavior (the
        code optimizes against fixed feats; the gradient that plateaued did too)."""
        if not self_orient:
            return None
        dw = in_feat - curv_feats_np.shape[1]
        df = np.zeros((coords_np.shape[0], dw), np.float32)
        prev_am = None
        for _ in range(max(1, args.reorient_iters)):
            am = _argmax_frame1(pi, df)
            df = self_orientation_directional_feats(
                coords_np, am, n_freqs=n_dir_freqs, freq_across=freq_across, freq_along=freq_along).astype(np.float32)
            if prev_am is not None and np.array_equal(am, prev_am):
                break
            prev_am = am
        return df

    rng = np.random.default_rng(args.seed)
    code_std = float(np.std(deploy["code"], axis=0).mean())

    # ---- resume support -------------------------------------------------------------------------
    prog_path = out / "progress.json"
    state: dict = json.loads(prog_path.read_text()) if prog_path.exists() else {}
    dir_cache: dict[int, np.ndarray | None] = {}

    def _dir(pi: int):
        if pi not in dir_cache:
            dir_cache[pi] = _reconstruct_dir_feats(pi)
        return dir_cache[pi]

    def _base_d_seg(pi: int) -> float:
        f1 = _render_frame(pi, 1, deploy["code"][2 * pi + 1], _dir(pi))
        return float(cpu_verdict_d_seg_batch(seg_cpu, [f1], [gt.lstars[pi]])[0])

    # ---- ENV-FIDELITY GATE: random sample base d_seg must reproduce the Muon verdict ~0.00425 ----
    if "sanity" not in state:
        t0 = time.time()
        spairs = sorted(int(x) for x in rng.choice(P, size=min(args.n_sanity, P), replace=False))
        sds = [_base_d_seg(pi) for pi in spairs]
        sanity = {"pairs": spairs, "base_d_seg_mean": float(np.mean(sds)),
                  "base_d_seg_per_pair": [round(x, 6) for x in sds],
                  "muon_realized_mean_ref": 0.004250, "secs": round(time.time() - t0, 1)}
        ratio = sanity["base_d_seg_mean"] / 0.004250
        sanity["fidelity_ratio_vs_muon"] = round(ratio, 3)
        sanity["env_fidelity_ok"] = bool(0.6 <= ratio <= 1.7)  # reproduce verdict within a tight band
        state["sanity"] = sanity
        _atomic_write_json(prog_path, state)
        print(json.dumps({"stage": "env_fidelity_gate", **sanity}), flush=True)
        if not sanity["env_fidelity_ok"]:
            print(json.dumps({"stage": "REFUSE", "reason": "env_fidelity_failed",
                              "note": "base d_seg does not reproduce the Muon realized verdict -> the "
                              "reconstructed env (likely dir-feats fixed-point) is wrong; a broken reward "
                              "is a fake reward. Fix before trusting the gate."}), flush=True)
            return 3

    # ---- pick the STALLED pairs: highest base-residual d_seg among candidates -----------------------
    if "stalled_pairs" not in state:
        # rank candidates by GT low-margin density (a $0 hardness proxy), then measure base d_seg on
        # them and keep the worst-residual = the most stalled (where the gradient plateaued hardest).
        lowmarg = np.array([int(np.count_nonzero(np.abs(np.asarray(gt.margins[pi])) < 0.30)) for pi in range(P)])
        cand = [int(x) for x in np.argsort(-lowmarg)[: max(args.n_candidate, args.n_stalled)]]
        cand_ds = {pi: _base_d_seg(pi) for pi in cand}
        stalled = sorted(cand, key=lambda pi: -cand_ds[pi])[: args.n_stalled]
        state["stalled_pairs"] = stalled
        state["candidate_base_d_seg"] = {str(pi): round(cand_ds[pi], 6) for pi in cand}
        _atomic_write_json(prog_path, state)
        print(json.dumps({"stage": "stalled_selection", "stalled_pairs": stalled,
                          "base_d_seg": {str(pi): round(cand_ds[pi], 6) for pi in stalled}}), flush=True)
    stalled = state["stalled_pairs"]

    # ---- CEM on the per-pair frame1 code vs the EXACT d_seg reward -------------------------------
    results = state.get("cem_results", {})
    n_elite = max(2, int(round(args.cem_pop * args.elite_frac)))
    for pi in stalled:
        if str(pi) in results:
            continue
        t0 = time.time()
        dir_pi = _dir(pi)
        c1 = np.asarray(deploy["code"][2 * pi + 1], np.float64)
        f0 = _render_frame(pi, 0, deploy["code"][2 * pi + 0], dir_pi)  # base frame0 (for d_pose constraint)
        base_ds = float(cpu_verdict_d_seg_batch(seg_cpu, [_render_frame(pi, 1, c1, dir_pi)], [gt.lstars[pi]])[0])
        base_dp = float(cpu_verdict_d_pose_batch(posenet_cpu, [f0], [_render_frame(pi, 1, c1, dir_pi)], [gt.gt_poses[pi]])[0])
        init_sigma = float(args.cem_sigma) * code_std
        sigma_floor = float(args.cem_sigma_floor_frac) * init_sigma  # ANTI-COLLAPSE: never search-stall to base
        n_explore = max(1, int(round(args.cem_pop * float(args.cem_explore_frac))))
        mu = np.zeros_like(c1); sigma = init_sigma * np.ones_like(c1)
        best_delta = np.zeros_like(c1); best_ds = base_ds
        crng = np.random.default_rng(args.seed * 100003 + pi)
        for g in range(args.cem_gens):
            # population = [delta=0 (base, so best never regresses)]
            #            + [n_explore WIDE candidates from N(0, init_sigma) around base — guarantees
            #               continued exploration even if the CEM mean/sigma collapse (anti-false-RED)]
            #            + [rest from the adapted N(mu, max(sigma, sigma_floor))].
            n_cem = args.cem_pop - 1 - n_explore
            sig_eff = np.maximum(sigma, sigma_floor)
            wide = init_sigma * crng.standard_normal((n_explore, c1.shape[0]))
            cem = mu[None] + sig_eff[None] * crng.standard_normal((max(n_cem, 0), c1.shape[0]))
            deltas = np.concatenate([np.zeros((1, c1.shape[0])), wide, cem], axis=0)
            frames = [_render_frame(pi, 1, c1 + d, dir_pi) for d in deltas]
            dss = np.asarray(cpu_verdict_d_seg_batch(seg_cpu, frames, [gt.lstars[pi]] * len(frames)))
            # update mu/sigma from the elites among the SAMPLED (non-base) candidates — exclude delta=0
            # so a deceptive "base is best" generation cannot drag the search mean to zero.
            sampled_idx = np.arange(1, len(deltas))
            elite_idx = sampled_idx[np.argsort(dss[sampled_idx])[:n_elite]]
            elite = deltas[elite_idx]
            mu = elite.mean(0); sigma = np.maximum(elite.std(0), sigma_floor)
            gbest = int(np.argmin(dss))
            if dss[gbest] < best_ds:
                best_ds = float(dss[gbest]); best_delta = deltas[gbest].copy()
        # d_pose at the best d_seg code (frame0 unchanged) — the constraint check
        f1_best = _render_frame(pi, 1, c1 + best_delta, dir_pi)
        best_dp = float(cpu_verdict_d_pose_batch(posenet_cpu, [f0], [f1_best], [gt.gt_poses[pi]])[0])
        rel = (base_ds - best_ds) / max(base_ds, 1e-9)
        results[str(pi)] = {"base_d_seg": round(base_ds, 6), "best_d_seg": round(best_ds, 6),
                            "rel_drop": round(rel, 4), "base_d_pose": round(base_dp, 7),
                            "best_d_pose": round(best_dp, 7), "d_pose_harmed": bool(best_dp > base_dp * 1.5 + 1e-6),
                            "delta_l2": round(float(np.linalg.norm(best_delta)), 4), "secs": round(time.time() - t0, 1)}
        state["cem_results"] = results
        _atomic_write_json(prog_path, state)
        print(json.dumps({"stage": "cem_pair", "pair": pi, **results[str(pi)]}), flush=True)

    # ---- VERDICT --------------------------------------------------------------------------------
    rels = [results[str(pi)]["rel_drop"] for pi in stalled]
    mean_rel = float(np.mean(rels)); n_improved = int(sum(1 for r in rels if r >= args.success_rel))
    any_pose_harm = bool(any(results[str(pi)]["d_pose_harmed"] for pi in stalled))
    verdict = "GREEN" if mean_rel >= args.success_rel else ("AMBER" if mean_rel >= 0.5 * args.success_rel else "RED")
    summary = {
        "tag": "[$0 CPU-authority advisory / distortion go-no-go / n=few pairs / NOT a contest score]",
        "utc": _utc(), "warm_start": str(args.warm_start), "muon_epoch": int(z["__epoch"]),
        "n_stalled": len(stalled), "stalled_pairs": stalled,
        "mean_rel_drop": round(mean_rel, 4), "n_pairs_improved_ge_threshold": n_improved,
        "success_rel_threshold": args.success_rel, "any_d_pose_harmed": any_pose_harm,
        "per_pair": {str(pi): results[str(pi)] for pi in stalled},
        "VERDICT": verdict,
        "verdict_meaning": {
            "GREEN": "gradient-free search moved a STALLED annulus residual the plateaued gradient could not -> RL/policy lab JUSTIFIED",
            "AMBER": "partial movement -> deeper search / different action space before deciding (do NOT conclude RED from first results)",
            "RED": "no movement under this budget/action -> SUSPECT, adversarially audit (budget? action space? wrong stalled set? small-n?) BEFORE accepting the aleatoric-floor conclusion",
        }[verdict],
        "MEANS_NOT_ENDS": "advisory only; pointer UNMOVED contest-CPU 0.19110; the END is a byte-closed n600 row < 0.19110",
        "caveats": [
            "d_pose is advisory-only here: this arm has w_pose=0 (d_seg-only witness) so the witness's PoseNet "
            "reading is garbage (~120); deployed pose rides the SEPARATE stored sidecar. d_pose_harmed compares "
            "garbage-to-garbage and is NOT the deployed pose — do not over-interpret.",
            "self-orient dir-feats are HELD FIXED during CEM (faithful to the trainer's within-reorient window, "
            "and the plateaued gradient operated under the same fixed feats). A win should be re-checked AFTER a "
            "reorient (further-testing) to confirm survival.",
            "the Muon arm was STILL DESCENDING at warm-start -> a RED is less damning (gradient not fully "
            "plateaued); re-run from the FINAL flattened ckpt before any aleatoric-floor conclusion.",
        ],
        "config": {"cem_pop": args.cem_pop, "cem_gens": args.cem_gens, "cem_sigma": args.cem_sigma,
                   "cem_sigma_floor_frac": args.cem_sigma_floor_frac, "cem_explore_frac": args.cem_explore_frac,
                   "elite_frac": args.elite_frac, "reorient_iters": args.reorient_iters, "seed": args.seed},
        "env_fidelity": state.get("sanity"),
    }
    _atomic_write_json(out / "summary.json", summary)
    print(json.dumps({"stage": "VERDICT", **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
