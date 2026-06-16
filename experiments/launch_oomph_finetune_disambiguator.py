#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""The CAPACITY-vs-LOSS-WALL disambiguator (resolves the #116 reflection's open crux).

The clean from-0 gate-3 A/B showed the seg-loss lever stack (soft_cosine + margin) is
~NEUTRAL vs CE (stack_ce_early best d_seg 0.00412 vs control 0.00359). The reflection
(``feedback_why_seg_loss_levers_neutral_capacity_bound``) attributed this to a
CAPACITY-bound d_seg plateau — but it left ONE claim unproven: gate-3 used GENTLE seg
settings (T_end=0.3, τ=1.0, no renorm), while l235's lower d_seg (0.00247) used SHARP
T_end=0.05. l235 was ALSO budget+pair-count confounded (29,650/600 vs 800/96), so its win
cannot be attributed to sharpness. This run ISOLATES the oomph effect from the budget
effect: all arms fine-tune the SAME converged control checkpoint for the SAME short window,
so the ONLY difference is the seg loss form/sharpness.

Three arms, BIT-IDENTICAL init (same ``--warm-start-dir`` converged checkpoint + ``--seed``):

  * ``ce_continue`` — continue with vendored CE (the fine-tune CONTROL: "what does +N
                      epochs of plain CE from the converged basin buy?").
  * ``gentle``      — soft_cosine @ T=0.3 static + Lever-5 margin τ=1.0, NO renorm
                      (≈ the gate-3 refinement config, now from the CONVERGED basin).
  * ``oomph``       — EVERYTHING cranked: soft_cosine annealed T 0.3→0.05 (SHARP →
                      ≈ exact argmax-flip d_seg) + tight margin τ=0.5 + margin RENORM +
                      seg_weight 1.5× (the "more oomph" the operator asked about).

Verdict (read off the per-arm BEST exact d_seg vs control's 0.00359):
  * GO  (loss-wall):     ``oomph`` (and/or ``gentle``) BEST d_seg < 0.00359 AND < ce_continue
                         → the plateau was an optimization/loss-wall, not capacity →
                         REINSTATE the seg lever (sharp soft_cosine) in the scaled stack.
  * NO-GO (capacity):    all three plateau at ~0.00359 (oomph ≈ gentle ≈ ce_continue)
                         → CAPACITY wall confirmed → commit to the STRUCTURAL levers
                         (d_seg-aware taper + FP4-QAT), drop the loss-lever.
The d_pose / total score are ALSO reported so an oomph d_seg win bought by pose drift
(seg_weight↑) is visible (still informative: it proves d_seg is loss-movable → reinstate
WITH pose protection).

WHY warm-start (not from-0): the question is "can the loss push d_seg BELOW the converged
floor", which is a refinement-regime property. Starting all arms from the converged basin
(via ``cfg.warm_start_dir``) controls budget + init + bottleneck simultaneously — the exact
apples-to-apples the reflection's META-LESSON demands. ``cfg.ema_warmup`` is ON so the EMA
shadow (which the BEST d_seg reads) TRACKS the short fine-tune instead of staying frozen at
the warm init (the EMA-shadow-lag fix).

AUTHORITY (CLAUDE.md): exact d_seg/d_pose run on the CPU authority (``--device cpu``; NO MPS
as a scorer). ``--train-device mps --split-by-head`` is the gradient-only speed path (SegNet
grad on MPS, PoseNet grad on the CPU authority). Every in-loop number is
``[contest-CPU advisory]`` NON-PROMOTABLE until a byte-closed archive runs through
``upstream/evaluate.py``.

Run ONE arm per invocation (own out-dir; resumable; DONE-marker on exit), as a DETACHED
nohup daemon (NOT tool-background — SIGURG-kills bg bash at ~3 min). Sequence AFTER the live
A/B frees the GPU (do not contend the MPS device). Example:
    nohup bash -c '.venv/bin/python experiments/launch_oomph_finetune_disambiguator.py \
        --arm oomph --warm-start-dir experiments/results/from0_ab_v2_n96/control/best \
        --out-dir experiments/results/oomph_ft_<stamp>/oomph \
        --train-device mps --split-by-head --epochs 150 --n-pairs 96 \
        --pose-grad-every-k 4 --pose-grad-resume-threshold 0.001 --go' \
        </dev/null >.../oomph.outer.log 2>&1 & disown
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

# The fine-tune CONTROL's converged d_seg (the apples-to-apples baseline this disambiguator
# must beat). Read from --warm-start-dir/best_meta.json at runtime; this is the documented
# expectation for the from0_ab_v2_n96/control checkpoint (advisory; the runtime value wins).
_CONTROL_DSEG_EXPECTED = 0.003590689778017501

_ARMS = ("ce_continue", "gentle", "oomph")

# gentle = the gate-3 refinement seg config from the CONVERGED basin (static T=0.3, τ=1.0).
_GENTLE = dict(seg_surrogate="soft_cosine", seg_temperature=0.3, seg_temperature_end=None,
               seg_temperature_hold_frac=None, margin_weight_tau=1.0,
               margin_weight_renorm=False, seg_weight_mult=1.0)
# oomph = EVERYTHING cranked (the operator's "soft cosine at the end with more oomph").
#   sharp anneal T 0.3→0.05 over the first half then hold (→ ≈ exact argmax-flip d_seg),
#   tight margin τ=0.5 (concentrate harder on the flip-prone boundary), margin RENORM (a
#   true weighted mean → no total-magnitude decay as margins grow), seg_weight 1.5× (safe
#   on a pose-solved basin; pose drift, if any, is reported).
_OOMPH = dict(seg_surrogate="soft_cosine", seg_temperature=0.3, seg_temperature_end=0.05,
              seg_temperature_hold_frac=0.5, margin_weight_tau=0.5,
              margin_weight_renorm=True, seg_weight_mult=1.5)


def _last_evaluated_dseg(out_dir: Path) -> float | None:
    """The d_seg of the FINAL evaluated epoch from the trajectory JSONL (NOT the min).
    Reads the last row with a non-null ``d_seg``. Robust to malformed/partial JSONL lines.
    Returns None if no eval row exists. (driver.run()'s summary carries no last_eval, so the
    durable telemetry is the source of truth for the sustained-win check.)"""
    traj = Path(out_dir) / "torch_vehicle_trajectory.jsonl"
    if not traj.exists():
        return None
    for line in reversed(traj.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("d_seg") is not None:
            try:
                return float(row["d_seg"])
            except (ValueError, TypeError):
                return None
    return None


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=_ARMS,
                   help="which arm (one per invocation; SAME --warm-start-dir + --seed across arms)")
    p.add_argument("--warm-start-dir", type=Path, required=True,
                   help="a prior run's best/ dir (best_ema_decoder.pt + best_ema_latents.pt) — "
                        "the CONVERGED checkpoint all arms fine-tune from (bit-identical init).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="arm run dir (resumes if a checkpoint is present). MUST be NEW.")
    p.add_argument("--base-channels", type=int, default=20)
    p.add_argument("--latent-dim", type=int, default=28)
    p.add_argument("--epochs", type=int, default=150,
                   help="refinement-only fine-tune epochs (single Muon+QAT stage).")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--eval-every", type=int, default=10,
                   help="exact d_seg/d_pose eval cadence (10 → ~15 reads over 150 ep).")
    p.add_argument("--checkpoint-every-epochs", type=int, default=1)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="AUTHORITY/eval device (CPU-TRUSTED). NO mps.")
    p.add_argument("--train-device", default="cpu", choices=["cpu", "cuda", "mps"],
                   help="gradient backend. mps requires --split-by-head.")
    p.add_argument("--split-by-head", action="store_true",
                   help="MPS SegNet grad + CPU PoseNet grad (the ~90x lever).")
    p.add_argument("--async-eval", action="store_true")
    p.add_argument("--n-pairs", type=int, default=96,
                   help="MUST match the warm-start checkpoint's basis (control = 96).")
    p.add_argument("--pose-grad-every-k", type=int, default=4,
                   help="pose-throttle (pose is solved on the converged basin → cadence it).")
    p.add_argument("--pose-grad-resume-threshold", type=float, default=0.001,
                   help="force pose every epoch while pose_mse exceeds this (self-protect).")
    p.add_argument("--seg-weight-base", type=float, default=100.0,
                   help="the stage seg_weight the arms scale (oomph ×--oomph-seg-weight-mult); "
                        "matches the vendored muon-finetune default.")
    p.add_argument("--oomph-seg-weight-mult", type=float, default=1.5,
                   help="oomph's seg_weight multiplier (default 1.5 = 'everything cranked'). "
                        "Set 1.0 for a CLEAN isolation run that varies ONLY T-sharpness + "
                        "margin + renorm (so an oomph win cannot be attributed to seg_weight).")
    p.add_argument("--go-rel-threshold", type=float, default=0.03,
                   help="GO requires best d_seg to beat the run's OWN warm baseline by MORE "
                        "than this relative margin (default 3%%) — above the EMA-shadow noise "
                        "floor, so a best-of-noise minimum cannot trip a false GO. l235's real "
                        "win was ~31%%, far above this bar.")
    p.add_argument("--seed", type=int, default=0,
                   help="SHARED across arms → with the same warm-start, bit-identical init.")
    p.add_argument("--targets-cache", type=Path,
                   default=Path("experiments/results/capstone_gt_targets_cache"))
    p.add_argument("--video-path", type=Path, default=None)
    p.add_argument("--go", action="store_true",
                   help="REQUIRED to launch the real-scorer arm (loads the heavy scorer).")
    p.add_argument("--print-plan", action="store_true",
                   help="print the fine-tune stage spec for this arm and exit ($0).")
    return p


def _finetune_spec(base_spec, arm: str, epochs: int, seg_weight_base: float,
                   oomph_seg_weight_mult: float = 1.5):
    """Build the single refinement fine-tune stage by overlaying ``arm`` onto the
    vendored Muon+QAT refinement stage (``base_spec``). ``init_latents_random=True`` is
    REQUIRED so the driver's warm-start branch fires (it replaces the random draw with the
    loaded converged latents). All non-seg hyperparameters (Muon, QAT, C1a λ, σ, LR) are
    inherited UNCHANGED from the faithful refinement stage → the ONLY cross-arm difference
    is the seg loss form/sharpness/weight."""
    common = dict(epochs=int(epochs), init_latents_random=True, seg_weight=float(seg_weight_base))
    if arm == "ce_continue":
        # Vendored CE: NO surrogate, NO margin-weight, NO renorm. The fine-tune control.
        return dataclasses.replace(
            base_spec, seg_surrogate=None, seg_temperature_end=None,
            seg_temperature_hold_frac=None, margin_weight_tau=None,
            margin_weight_renorm=False, **common,
        )
    cfg = _GENTLE if arm == "gentle" else _OOMPH
    # gentle keeps its 1.0 mult; oomph uses the CLI mult (1.5 default; set 1.0 to isolate
    # sharpness/margin/renorm from the seg_weight bump).
    mult = cfg["seg_weight_mult"] if arm == "gentle" else float(oomph_seg_weight_mult)
    return dataclasses.replace(
        base_spec,
        seg_surrogate=cfg["seg_surrogate"],
        seg_temperature=cfg["seg_temperature"],
        seg_temperature_end=cfg["seg_temperature_end"],
        seg_temperature_hold_frac=cfg["seg_temperature_hold_frac"],
        margin_weight_tau=cfg["margin_weight_tau"],
        margin_weight_renorm=cfg["margin_weight_renorm"],
        **{**common, "seg_weight": float(seg_weight_base) * mult},
    )


def _plan_row(spec) -> dict:
    return {
        "name": spec.name, "epochs": spec.epochs, "use_muon": spec.use_muon,
        "use_qat": spec.use_qat, "muon_lr": spec.muon_lr, "adamw_lr": spec.adamw_lr,
        "seg_surrogate": spec.seg_surrogate, "seg_temperature": spec.seg_temperature,
        "seg_temperature_end": spec.seg_temperature_end,
        "seg_temperature_hold_frac": spec.seg_temperature_hold_frac,
        "margin_weight_tau": spec.margin_weight_tau,
        "margin_weight_renorm": spec.margin_weight_renorm, "seg_weight": spec.seg_weight,
        "init_latents_random": spec.init_latents_random,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    from tac.torch_vehicle.curriculum import build_curriculum

    # The faithful refinement stage (Muon + QAT + C1a λ + σ) is the LAST PR95 stage; we
    # fine-tune a single instance of it from the converged checkpoint.
    base_refine = build_curriculum(total_epoch_budget=800, ema_decay=args.ema_decay,
                                   eval_every=args.eval_every)[-1]
    spec = _finetune_spec(base_refine, args.arm, args.epochs, args.seg_weight_base,
                          oomph_seg_weight_mult=args.oomph_seg_weight_mult)

    if args.print_plan:
        print(json.dumps({"arm": args.arm, "stage": _plan_row(spec),
                          "control_dseg_expected": _CONTROL_DSEG_EXPECTED}, indent=2))
        return 0

    out_dir = Path(args.out_dir)
    warm = Path(args.warm_start_dir)
    if not (warm / "best_ema_decoder.pt").exists():
        raise SystemExit(f"--warm-start-dir {warm} has no best_ema_decoder.pt (need a "
                         "converged best/ dir).")
    # Hygiene: never write the disambiguator INTO the warm/control checkpoint or the live
    # basin tree (would corrupt the source of truth + break apples-to-apples re-runs).
    try:
        out_res = out_dir.resolve()
        if out_res == warm.resolve() or warm.resolve() in out_res.parents:
            raise SystemExit(f"REFUSED: --out-dir {out_dir} is the warm-start dir (or inside "
                             "it). Use a NEW results dir for the disambiguator.")
        for live in (Path("experiments/results/forkpoints"),
                     Path("experiments/results/from0_ab_v2_n96")):
            try:
                lr = live.resolve()
                if out_res == lr or lr in out_res.parents:
                    raise SystemExit(f"REFUSED: --out-dir {out_dir} is inside the live basin "
                                     f"tree {live}. Use a NEW results dir.")
            except FileNotFoundError:
                pass
    except FileNotFoundError:
        pass
    if args.split_by_head and args.train_device != "mps":
        raise SystemExit("--split-by-head requires --train-device mps")
    if args.train_device == "mps" and not args.split_by_head:
        raise SystemExit("--train-device mps must be used WITH --split-by-head (PoseNet stays "
                         "on the CPU authority; MPS corrupts PoseNet 23x).")
    if not args.go:
        raise SystemExit("REFUSED without --go (loads the heavy SegNet/PoseNet). Use "
                         "--print-plan for the $0 stage-spec preview.")

    out_dir.mkdir(parents=True, exist_ok=True)
    from tac.torch_vehicle.driver import (
        TorchVehicleConfig, TorchVehicleDriver, import_vendored_bundle,
    )
    from tac.torch_vehicle.scorer_context import RealScorerContext

    video_path = args.video_path
    if video_path is None:
        from tac.torch_vehicle.vendored_imports import import_vendored

        video_path = import_vendored("data").get_default_video_path()

    # Read the control's converged d_seg for the verdict (advisory; runtime value wins).
    control_dseg = _CONTROL_DSEG_EXPECTED
    meta_path = warm / "best_meta.json"
    if meta_path.exists():
        try:
            control_dseg = float(json.loads(meta_path.read_text()).get("d_seg", control_dseg))
        except Exception:
            pass

    scorer = RealScorerContext(
        video_path, device=args.device, train_device=args.train_device,
        split_by_head=args.split_by_head, max_pairs=args.n_pairs,
        targets_cache=args.targets_cache,
    )

    # --- RUN'S-OWN WARM BASELINE (the adversarial-review fix) -------------------
    # Re-evaluate the warm checkpoint on THIS run's exact eval config (same scorer,
    # same n_pairs, same targets cache) BEFORE any training. This is the honest
    # apples-to-apples baseline: it removes any eval-config mismatch vs the external
    # control number AND captures the floor d_seg is fine-tuning DOWN FROM. The verdict
    # compares the run's best d_seg against THIS, not the external control_dseg. It is
    # ~identical across arms (same checkpoint) — an internal-consistency check too.
    import torch as _torch

    vbundle = import_vendored_bundle()
    _warm_dec = vbundle.HNeRVDecoder(
        latent_dim=args.latent_dim, base_channels=args.base_channels, eval_size=(384, 512)
    ).to(args.device).eval()
    _wsd = _torch.load(warm / "best_ema_decoder.pt", map_location=args.device, weights_only=False)
    if isinstance(_wsd, dict) and "state_dict" in _wsd:
        _wsd = _wsd["state_dict"]
    _warm_dec.load_state_dict({k: v.to(args.device) for k, v in _wsd.items()})
    _wlat = _torch.load(warm / "best_ema_latents.pt", map_location=args.device, weights_only=False)
    if isinstance(_wlat, dict):
        if not _wlat:
            raise SystemExit(f"warm latents dict at {warm / 'best_ema_latents.pt'} is empty")
        _wlat = _wlat.get("latents", next(iter(_wlat.values())))
    # archive_bytes for the warm baseline: the verdict uses d_seg ONLY (archive-independent),
    # but the printed score is honest only with the real byte count. Use the warm dir's
    # archive if present (the driver writes best_archive.bin), else best_meta's recorded
    # archive_bytes, else 0 (score's rate term then = 0 — clearly not a real composite).
    _warm_archive_bytes = 0
    _wab = warm / "best_archive.bin"
    if _wab.exists():
        _warm_archive_bytes = _wab.stat().st_size
    elif meta_path.exists():
        try:
            _warm_archive_bytes = int(json.loads(meta_path.read_text()).get("archive_bytes", 0))
        except Exception:
            _warm_archive_bytes = 0
    _warm_eval = scorer.exact_eval(_warm_dec, _wlat.to(args.device), _warm_archive_bytes)
    warm_dseg = float(_warm_eval.get("seg_distortion", _warm_eval.get("d_seg")))
    warm_dpose = float(_warm_eval.get("pose_distortion", _warm_eval.get("d_pose")))
    print(json.dumps({
        "warm_baseline_this_run": {"d_seg": warm_dseg, "d_pose": warm_dpose,
                                   "archive_bytes": _warm_archive_bytes},
        "control_dseg_recorded": control_dseg,
        "note": "verdict compares run best d_seg vs warm_baseline_this_run (eval-matched)",
        "authority": "[contest-CPU advisory] NON-PROMOTABLE",
    }, indent=2), flush=True)
    del _warm_dec, _wsd, _wlat  # free before the heavy run

    cfg = TorchVehicleConfig(
        base_channels=args.base_channels, latent_dim=args.latent_dim, out_dir=out_dir,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        total_epoch_budget=args.epochs, ema_decay=args.ema_decay,
        eval_every=args.eval_every, device=args.device, train_device=args.train_device,
        split_by_head=args.split_by_head, async_eval=args.async_eval,
        pose_film_enabled=False,  # warm-start loads a vendored (no-FiLM) state_dict
        seed=args.seed,
        pose_grad_every_k=args.pose_grad_every_k,
        pose_grad_resume_threshold=args.pose_grad_resume_threshold,
        warm_start_dir=warm,   # the disambiguator's defining hook
        ema_warmup=True,       # short fine-tune → shadow must track (EMA-shadow-lag fix)
    )
    print(json.dumps({
        "arm": args.arm, "warm_start_dir": str(warm), "control_dseg": control_dseg,
        "device": args.device, "train_device": args.train_device,
        "split_by_head": args.split_by_head, "n_pairs": args.n_pairs,
        "epochs": args.epochs, "ema_warmup": True, "seed": args.seed,
        "out_dir": str(out_dir), "stage": _plan_row(spec),
        "authority": "[contest-CPU advisory] NON-PROMOTABLE",
    }, indent=2), flush=True)

    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(), curriculum=[spec]
    )
    summary = driver.run()

    # best d_seg over the run (the BEST tracker's min over evals). Surface schema
    # mismatches LOUDLY (a missing 'd_seg' key must not collapse silently to an
    # unexplained INDETERMINATE — the operator needs the diagnosis).
    best_dseg = None
    bm = out_dir / "best" / "best_meta.json"
    if bm.exists():
        try:
            _bm = json.loads(bm.read_text())
            _raw = _bm.get("d_seg")
            if _raw is None:
                print(f"WARNING: {bm} has no 'd_seg' key; keys={list(_bm)}", flush=True)
            else:
                best_dseg = float(_raw)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"WARNING: could not parse best d_seg from {bm}: {e}", flush=True)
    else:
        print(f"WARNING: no {bm} — the run produced no BEST checkpoint (no eval beat +inf?)",
              flush=True)
    # last-eval d_seg = the d_seg of the FINAL evaluated epoch (NOT the min) — the SUSTAINED
    # signal that defeats the min-of-N extreme-value bias in `best`. Read from the durable
    # trajectory telemetry (driver.run()'s summary carries no last_eval).
    last_dseg = _last_evaluated_dseg(out_dir)

    thr = float(args.go_rel_threshold)
    verdict = "INDETERMINATE (no best d_seg)"
    rel = None
    if best_dseg is not None:
        # Compare against the run's OWN warm baseline (eval-config-matched). TWO guards
        # against false GO: (1) the BEST drop must clear the noise margin; (2) the drop must
        # be SUSTAINED — the LAST eval is also below warm. Guard (2) defeats the min-of-N
        # extreme-value bias: the minimum over ~15 noisy evals is biased low, so a lucky-noise
        # `best` 3% under warm with a `last` back at warm is NOT a real win.
        rel = (best_dseg - warm_dseg) / max(warm_dseg, 1e-9)
        best_clears = rel < -thr
        sustained = (last_dseg is not None) and (last_dseg < warm_dseg)
        if best_clears and sustained:
            verdict = (
                f"GO: arm {args.arm} best d_seg {best_dseg:.6f} beats warm {warm_dseg:.6f} by "
                f"{-rel*100:.1f}% (> {thr*100:.0f}% margin) AND last {last_dseg:.6f} < warm "
                f"(SUSTAINED) → d_seg is LOSS-MOVABLE from the converged basin → reinstate the lever"
            )
        elif best_clears and not sustained:
            verdict = (
                f"GO-WEAK (likely min-of-N noise): arm {args.arm} best d_seg {best_dseg:.6f} beats "
                f"warm {warm_dseg:.6f} by {-rel*100:.1f}% but last {last_dseg} NOT < warm (not "
                f"sustained) → treat as NO-GO pending a longer run / multi-eval mean"
            )
        else:
            verdict = (
                f"NO-GO: arm {args.arm} best d_seg {best_dseg:.6f} vs warm {warm_dseg:.6f} "
                f"(Δ{rel*100:+.1f}%, within ±{thr*100:.0f}% noise) → no loss-driven d_seg "
                f"improvement → consistent with the CAPACITY wall"
            )
    print(json.dumps({
        **summary, "arm": args.arm,
        "warm_dseg_this_run": warm_dseg, "warm_dpose_this_run": warm_dpose,
        "control_dseg_recorded": control_dseg,
        "best_dseg": best_dseg, "last_dseg": last_dseg,
        "rel_vs_warm": rel, "go_rel_threshold": thr, "verdict": verdict,
        "authority": "[contest-CPU advisory] NON-PROMOTABLE",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
