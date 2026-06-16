#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""From-0 A/B for the #1 STRUCTURAL lever: d_seg-AWARE TAPER vs the vendored HNeRV taper.

The reflection (``feedback_why_seg_loss_levers_neutral_capacity_bound``) concluded the
base_ch=20 d_seg plateau is CAPACITY-bound, and the gate-2 sensitivity map showed ~81% of
capacity is mis-allocated to the insensitive low-res early stages while the d_seg-critical
mid-late band (``skips.2-4`` / ``blocks.4`` / ``rgb_1``) is starved. This A/B tests the
direct fix: REALLOCATE channels from the early stages to the mid-late stages at a
~byte-matched param count (conv params are ``in·out·k²``, resolution-independent), via the
``ConfigurableTaperHNeRVDecoder`` (a faithful generalization of the vendored decoder).

  * ``baseline``    — the vendored HNeRV taper [C,C,C,.75C,.58C,.5C,.5C] (``taper_channels``
                      None → the byte-identical vendored decoder). The control.
  * ``dseg_aware``  — the late-weighted reallocation ``dseg_aware_taper(C)``, byte-matched
                      to baseline within ~0.1% (so the A/B isolates ALLOCATION from total
                      capacity).
  * ``--taper-channels "a,b,c,d,e,f,g"`` — an explicit 7-stage schedule for sweeps.

Both arms train from-0 with the PLAIN vendored curriculum (CE; NO loss levers — the
reflection dropped them as neutral on a capacity-bound plateau; the taper is the structural
lever tested on its OWN). Same ``--seed`` / budget / data; the two arms are DIFFERENT
architectures so the init is same-seed-same-distribution (a bit-identical init is impossible
across architectures — this is the cleanest apples-to-apples for an architecture A/B). The
VERDICT is cross-arm: dseg_aware best d_seg vs baseline best d_seg at matched bytes.

The vendored codec is schedule-agnostic (verified ``test_configurable_taper_decoder``) so
the taper byte-closes + parse-backs through ``build_archive``/``parse_archive`` unchanged →
the rate term is REAL. AUTHORITY: exact d_seg/d_pose on the CPU authority (NO MPS as a
scorer); ``--train-device mps --split-by-head`` is the gradient-only speed path. Every
in-loop number is ``[contest-CPU advisory]`` NON-PROMOTABLE until a byte-closed archive runs
through ``upstream/evaluate.py``.

Run ONE arm per invocation (own out-dir; resumable; DONE-marked), as a DETACHED nohup
daemon. The baseline arm MAY reuse an existing vendored-curriculum control run (e.g.
from0_ab_v2_n96/control, d_seg 0.00359) via ``--baseline-dir`` instead of re-training it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_ARMS = ("baseline", "dseg_aware", "custom")
_LIVE_BASIN_DIRS = (Path("experiments/results/forkpoints"),)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=_ARMS,
                   help="baseline (vendored taper) | dseg_aware (reallocated) | custom "
                        "(--taper-channels). Same --seed/budget across arms.")
    p.add_argument("--out-dir", type=Path, required=True, help="arm run dir (resumes; MUST be NEW).")
    p.add_argument("--taper-channels", type=str, default=None,
                   help="custom 7-stage schedule 'a,b,c,d,e,f,g' (required for --arm custom).")
    p.add_argument("--base-channels", type=int, default=20)
    p.add_argument("--latent-dim", type=int, default=28)
    p.add_argument("--total-epoch-budget", type=int, default=800,
                   help="match the control's budget (800/96) for comparability.")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--checkpoint-every-epochs", type=int, default=1)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--train-device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--split-by-head", action="store_true")
    p.add_argument("--async-eval", action="store_true")
    p.add_argument("--n-pairs", type=int, default=96)
    p.add_argument("--pose-grad-every-k", type=int, default=4)
    p.add_argument("--pose-grad-resume-threshold", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--targets-cache", type=Path,
                   default=Path("experiments/results/capstone_gt_targets_cache"))
    p.add_argument("--video-path", type=Path, default=None)
    p.add_argument("--baseline-dir", type=Path, default=None,
                   help="optional: a prior vendored-curriculum control best/ dir to read the "
                        "baseline d_seg from for the cross-arm verdict (e.g. "
                        "experiments/results/from0_ab_v2_n96/control/best) — avoids re-training it.")
    p.add_argument("--go", action="store_true",
                   help="REQUIRED to launch the real-scorer arm.")
    p.add_argument("--print-plan", action="store_true",
                   help="print the resolved taper + param count and exit ($0).")
    return p


def _resolve_taper(args):
    from tac.torch_vehicle.configurable_taper_decoder import (
        decoder_param_count, dseg_aware_taper, vendored_taper,
    )

    C = args.base_channels
    if args.arm == "baseline":
        return None, vendored_taper(C)  # None → the byte-identical vendored decoder
    if args.arm == "dseg_aware":
        ch = dseg_aware_taper(C)
        return ch, ch
    # custom
    if not args.taper_channels:
        raise SystemExit("--arm custom requires --taper-channels 'a,b,c,d,e,f,g'")
    ch = [int(x) for x in args.taper_channels.split(",")]
    if len(ch) != 7:
        raise SystemExit(f"--taper-channels must have 7 stages; got {len(ch)}")
    return ch, ch


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    from tac.torch_vehicle.configurable_taper_decoder import (
        decoder_param_count, vendored_taper,
    )

    taper_channels, resolved = _resolve_taper(args)
    base_params = decoder_param_count(vendored_taper(args.base_channels), args.latent_dim)
    arm_params = decoder_param_count(resolved, args.latent_dim)

    if args.print_plan:
        print(json.dumps({
            "arm": args.arm, "taper_channels": resolved,
            "decoder_params": arm_params, "baseline_decoder_params": base_params,
            "byte_delta_pct": round(100 * (arm_params - base_params) / base_params, 2),
            "taper_channels_active": taper_channels,  # None for the vendored baseline
        }, indent=2))
        return 0

    out_dir = Path(args.out_dir)
    for live in _LIVE_BASIN_DIRS:
        try:
            if live.resolve() in out_dir.resolve().parents or out_dir.resolve() == live.resolve():
                raise SystemExit(f"REFUSED: --out-dir {out_dir} inside the live basin tree {live}.")
        except FileNotFoundError:
            pass
    if args.split_by_head and args.train_device != "mps":
        raise SystemExit("--split-by-head requires --train-device mps")
    if args.train_device == "mps" and not args.split_by_head:
        raise SystemExit("--train-device mps must be used WITH --split-by-head (PoseNet stays CPU).")
    if not args.go:
        raise SystemExit("REFUSED without --go. Use --print-plan for the $0 taper preview.")

    out_dir.mkdir(parents=True, exist_ok=True)
    from tac.torch_vehicle.curriculum import build_curriculum
    from tac.torch_vehicle.driver import (
        TorchVehicleConfig, TorchVehicleDriver, import_vendored_bundle,
    )
    from tac.torch_vehicle.scorer_context import RealScorerContext

    video_path = args.video_path
    if video_path is None:
        from tac.torch_vehicle.vendored_imports import import_vendored

        video_path = import_vendored("data").get_default_video_path()

    # plain vendored curriculum (CE; NO lever overlay — the taper is the structural lever).
    curriculum = build_curriculum(total_epoch_budget=args.total_epoch_budget,
                                  ema_decay=args.ema_decay, eval_every=args.eval_every)

    scorer = RealScorerContext(
        video_path, device=args.device, train_device=args.train_device,
        split_by_head=args.split_by_head, max_pairs=args.n_pairs,
        targets_cache=args.targets_cache,
    )
    cfg = TorchVehicleConfig(
        base_channels=args.base_channels, latent_dim=args.latent_dim, out_dir=out_dir,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        total_epoch_budget=args.total_epoch_budget, ema_decay=args.ema_decay,
        eval_every=args.eval_every, device=args.device, train_device=args.train_device,
        split_by_head=args.split_by_head, async_eval=args.async_eval,
        pose_film_enabled=False, seed=args.seed,
        pose_grad_every_k=args.pose_grad_every_k,
        pose_grad_resume_threshold=args.pose_grad_resume_threshold,
        taper_channels=taper_channels,  # None for baseline → vendored decoder
    )
    print(json.dumps({
        "arm": args.arm, "taper_channels": resolved, "taper_active": taper_channels,
        "decoder_params": arm_params, "baseline_decoder_params": base_params,
        "byte_delta_pct": round(100 * (arm_params - base_params) / base_params, 2),
        "device": args.device, "train_device": args.train_device,
        "split_by_head": args.split_by_head, "n_pairs": args.n_pairs,
        "total_epoch_budget": args.total_epoch_budget, "seed": args.seed,
        "out_dir": str(out_dir), "authority": "[contest-CPU advisory] NON-PROMOTABLE",
    }, indent=2), flush=True)

    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(), curriculum=curriculum
    )
    summary = driver.run()

    best_dseg = None
    bm = out_dir / "best" / "best_meta.json"
    if bm.exists():
        try:
            _bm = json.loads(bm.read_text())
            if _bm.get("d_seg") is not None:
                best_dseg = float(_bm["d_seg"])
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"WARNING: could not parse best d_seg from {bm}: {e}", flush=True)

    # cross-arm verdict vs an optional baseline dir.
    verdict = None
    if args.baseline_dir is not None and best_dseg is not None:
        bdir = Path(args.baseline_dir)
        bmeta = bdir / "best_meta.json"
        if bmeta.exists():
            try:
                _braw = json.loads(bmeta.read_text()).get("d_seg")
                if _braw is None:
                    raise ValueError(f"{bmeta} has no 'd_seg' key")
                base_dseg = float(_braw)
                rel = (best_dseg - base_dseg) / max(base_dseg, 1e-9)
                byte_delta = round(100 * (arm_params - base_params) / base_params, 2)
                verdict = (
                    f"{'GO' if rel < 0 else 'NO-GO'}: {args.arm} best d_seg {best_dseg:.6f} vs "
                    f"baseline {base_dseg:.6f} (Δ{rel*100:+.1f}%) at {byte_delta:+.1f}% bytes"
                )
            except Exception as e:
                verdict = f"baseline compare failed: {e}"

    print(json.dumps({**summary, "arm": args.arm, "best_dseg": best_dseg,
                      "decoder_params": arm_params, "verdict": verdict,
                      "authority": "[contest-CPU advisory] NON-PROMOTABLE"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
