"""Canonical tac CLI.

Provides the top-level command router for the lossy and lossless namespaces.
The lossy path is the standard training route; the lossless path is a minimal
profile-based skeleton that keeps the namespace canonical without duplicating
training logic.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

from .lossless.arithmetic import GPTArithmeticEstimate, build_gpt_arithmetic_plan, estimate_gpt_arithmetic_workload
from .lossless.codecs import (
    compress_lossless_file,
    decompress_lossless_file,
    evaluate_lossless_baseline_submission,
)
from .lossless.evaluate import evaluate_lossless_archive
from .lossless.profiles import PROFILES as LOSSLESS_PROFILES
from .lossless.state import promote_lossless_result
from .lossless.submission import build_submission_zip
from .profiles import PROFILES as LOSSY_PROFILES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = PROJECT_ROOT / "workspace" / "upstream" / "comma_video_compression_challenge"

DEFAULTS = {
    "archive": str(PROJECT_ROOT / "submissions" / "robust_current" / "archive.zip"),
    "gt_video": str(UPSTREAM_ROOT / "videos" / "0.mkv"),
    "saliency": str(PROJECT_ROOT / "experiments" / "masks" / "posenet_saliency.npy"),
    "models_dir": str(UPSTREAM_ROOT / "models"),
    "upstream_dir": str(UPSTREAM_ROOT),
}


def _add_lossy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        default=None,
        help="Named profile from tac.profiles (e.g., council_v1, segnet_attack, smoke). "
        "CLI args override profile values.",
    )
    parser.add_argument("--archive", default=os.environ.get("TAC_ARCHIVE", DEFAULTS["archive"]))
    parser.add_argument("--gt-video", default=os.environ.get("TAC_GT_VIDEO", DEFAULTS["gt_video"]))
    parser.add_argument("--precomputed", default=os.environ.get("TAC_PRECOMPUTED", None))
    parser.add_argument("--saliency", default=os.environ.get("TAC_SALIENCY", DEFAULTS["saliency"]))
    parser.add_argument("--models-dir", default=os.environ.get("TAC_MODELS_DIR", DEFAULTS["models_dir"]))
    parser.add_argument("--upstream-dir", default=os.environ.get("TAC_UPSTREAM_DIR", DEFAULTS["upstream_dir"]))
    parser.add_argument("--variant", default="standard")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--sal-lambda", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--subsample", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--hard-frame-ratio", type=float, default=0.0)
    parser.add_argument("--error-replay-every", type=int, default=0)
    parser.add_argument(
        "--loss-mode",
        default="standard",
        choices=["standard", "temperature", "focal_ste", "kl_distill", "pcgrad"],
    )
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--segnet-loss-weight", type=float, default=100.0)
    parser.add_argument("--use-dual-saliency", action="store_true")
    parser.add_argument("--alpha-seg", type=float, default=200.0)
    parser.add_argument("--use-ste", action="store_true")
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", default="experiments/postfilter_weights")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tac", description="Canonical tac CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    lossy = subparsers.add_parser("lossy", help="Run the learned post-filter training lane.")
    _add_lossy_arguments(lossy)

    lossless = subparsers.add_parser("lossless", help="Run canonical lossless workflows.")
    lossless_sub = lossless.add_subparsers(dest="lossless_command", required=True)

    sp = lossless_sub.add_parser("profiles", help="List available lossless profiles")
    sp.set_defaults(lossless_handler="profiles")

    sp = lossless_sub.add_parser("plan", help="Build a non-measured lossless experiment plan")
    sp.add_argument("--profile", required=True, choices=sorted(LOSSLESS_PROFILES))
    sp.add_argument("--work-dir", default=None)
    sp.add_argument("--split", nargs="*", default=["challenge"])
    sp.set_defaults(lossless_handler="plan")

    sp = lossless_sub.add_parser("estimate", help="Estimate a non-measured lossless arithmetic workload")
    sp.add_argument("--profile", required=True, choices=sorted(LOSSLESS_PROFILES))
    sp.add_argument("--work-dir", default=None)
    sp.add_argument("--split", nargs="*", default=["challenge"])
    sp.set_defaults(lossless_handler="estimate")

    sp = lossless_sub.add_parser("baseline", help="Build a real dataset-backed lossless baseline submission")
    sp.add_argument("--profile", required=True, choices=sorted(LOSSLESS_PROFILES))
    sp.add_argument("--work-dir", required=True)
    sp.add_argument("--split", nargs="*", default=["0", "1"])
    sp.set_defaults(lossless_handler="baseline")

    sp = lossless_sub.add_parser("compress", help="Run a real lossless baseline compressor and exact round-trip check")
    sp.add_argument("--profile", required=True, choices=sorted(LOSSLESS_PROFILES))
    sp.add_argument("--input", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--decompressed-output", required=True)
    sp.set_defaults(lossless_handler="compress")

    sp = lossless_sub.add_parser("package", help="Build a commavq-style submission zip")
    sp.add_argument("--payload-dir", required=True)
    sp.add_argument("--decompress", required=True)
    sp.add_argument("--output", required=True)
    sp.set_defaults(lossless_handler="package")

    sp = lossless_sub.add_parser("evaluate", help="Evaluate an exact lossless archive result")
    sp.add_argument("--profile", required=True, choices=sorted(LOSSLESS_PROFILES))
    sp.add_argument("--method", required=True)
    sp.add_argument("--original", required=True)
    sp.add_argument("--decompressed", required=True)
    sp.add_argument("--archive", required=True)
    sp.set_defaults(lossless_handler="evaluate")

    sp = lossless_sub.add_parser("promote", help="Promote a measured lossless result into separate lossless state surfaces")
    sp.add_argument("--result-json", required=True)
    sp.add_argument("--repo-root", required=True)
    sp.set_defaults(lossless_handler="promote")

    return parser


def _select(profile_defaults: dict[str, Any], args: argparse.Namespace, name: str, default: Any) -> Any:
    value = getattr(args, name.replace("-", "_"))
    if value != default:
        return value
    return profile_defaults.get(name.replace("-", "_"), value)


def _run_lossy(args: argparse.Namespace) -> dict[str, Any]:
    from tac.architectures import build_postfilter
    from tac.data import load_frames, load_raw_saliency
    from tac.scorer import detect_device, load_scorers
    from tac.training import TrainConfig, Trainer

    device = detect_device()
    print(f"[tac] device: {device}")

    profile_defaults: dict[str, Any] = {}
    if args.profile:
        if args.profile not in LOSSY_PROFILES:
            raise SystemExit(f"ERROR: unknown profile '{args.profile}'")
        profile_defaults = LOSSY_PROFILES[args.profile]
        print(f"[tac] Using profile: {args.profile}")

    effective_variant = _select(profile_defaults, args, "variant", "standard")
    effective_hidden = int(_select(profile_defaults, args, "hidden", 64))
    effective_kernel = int(_select(profile_defaults, args, "kernel", 3))

    config = TrainConfig(
        hidden=effective_hidden,
        kernel=effective_kernel,
        variant=effective_variant,
        epochs=int(_select(profile_defaults, args, "epochs", 2500)),
        alpha=float(_select(profile_defaults, args, "alpha", 20.0)),
        sal_lambda=float(_select(profile_defaults, args, "sal-lambda", 1.0)),
        lr=float(_select(profile_defaults, args, "lr", 5e-4)),
        ema_decay=float(_select(profile_defaults, args, "ema-decay", 0.997)),
        accum_steps=int(_select(profile_defaults, args, "accum-steps", 4)),
        eval_every=int(_select(profile_defaults, args, "eval-every", 5)),
        hard_frame_ratio=float(_select(profile_defaults, args, "hard-frame-ratio", 0.0)),
        error_replay_every=int(_select(profile_defaults, args, "error-replay-every", 0)),
        loss_mode=str(_select(profile_defaults, args, "loss-mode", "standard")),
        temperature_start=float(_select(profile_defaults, args, "temperature-start", 1.0)),
        temperature_end=float(_select(profile_defaults, args, "temperature-end", 0.05)),
        focal_gamma=float(_select(profile_defaults, args, "focal-gamma", 2.0)),
        segnet_loss_weight=float(_select(profile_defaults, args, "segnet-loss-weight", 100.0)),
        use_dual_saliency=bool(args.use_dual_saliency or profile_defaults.get("use_dual_saliency", False)),
        alpha_seg=float(_select(profile_defaults, args, "alpha-seg", 200.0)),
        use_ste_segnet=bool(args.use_ste or profile_defaults.get("use_ste_segnet", False)),
        boundary_weight=float(_select(profile_defaults, args, "boundary-weight", 1.0)),
        boundary_anneal=bool(profile_defaults.get("boundary_anneal", False)),
        resume_from=args.resume_from,
        output_dir=args.output_dir,
        tag=args.tag,
    )

    print(
        f"[tac] config: h={config.hidden} {config.variant} epochs={config.epochs} "
        f"alpha={config.alpha} sal_lambda={config.sal_lambda} loss={config.loss_mode}"
    )

    comp_frames, gt_frames = load_frames(
        archive_path=args.archive,
        gt_video_path=args.gt_video,
        precomputed_dir=args.precomputed,
    )
    print(f"[tac] {len(comp_frames)} compressed + {len(gt_frames)} GT frames")

    raw_saliency = load_raw_saliency(args.saliency)
    print(f"[tac] Saliency shape: {tuple(raw_saliency.shape)}")

    models_dir = Path(args.models_dir)
    posenet, segnet = load_scorers(
        models_dir / "posenet.safetensors",
        models_dir / "segnet.safetensors",
        device=device,
        upstream_dir=args.upstream_dir,
    )

    model = build_postfilter(effective_variant, hidden=effective_hidden, kernel=effective_kernel)
    print(f"[tac] Model: {effective_variant} h={effective_hidden} ({sum(p.numel() for p in model.parameters())} params)")

    trainer = Trainer(model, config, device=device)
    best = trainer.fit_lazy(comp_frames, gt_frames, posenet, segnet, raw_saliency, subsample=args.subsample)
    print(f"[tac] Done. Best scorer: {best:.4f}")
    return {
        "command": "lossy",
        "best_scorer": best,
        "tag": args.tag,
        "variant": effective_variant,
        "hidden": effective_hidden,
        "kernel": effective_kernel,
    }


def _run_lossless(args: argparse.Namespace) -> dict[str, Any]:
    if args.lossless_handler == "profiles":
        payload = {
            "command": "lossless_profiles",
            "profiles": sorted(LOSSLESS_PROFILES),
        }
        print(json.dumps(payload, indent=2))
        return payload

    if args.lossless_handler == "plan":
        plan = build_gpt_arithmetic_plan(
            args.profile,
            split=args.split,
            work_dir=Path(args.work_dir) if args.work_dir else None,
        )
        plan_payload = asdict(plan)
        plan_payload["split"] = list(plan_payload["split"])
        payload = {
            "command": "lossless_plan",
            "plan": plan_payload,
        }
        print(json.dumps(payload, indent=2))
        return payload

    if args.lossless_handler == "estimate":
        estimate = estimate_gpt_arithmetic_workload(
            args.profile,
            split=args.split,
            work_dir=Path(args.work_dir) if args.work_dir else None,
        )
        estimate_payload = asdict(estimate)
        estimate_payload["split"] = list(estimate_payload["split"])
        payload = {
            "command": "lossless_estimate",
            "estimate": estimate_payload,
        }
        print(json.dumps(payload, indent=2))
        return payload

    if args.lossless_handler == "package":
        output = build_submission_zip(
            payload_dir=Path(args.payload_dir),
            decompress_path=Path(args.decompress),
            output_path=Path(args.output),
        )
        payload = {
            "command": "lossless_package",
            "output": str(output),
        }
        print(json.dumps(payload, indent=2))
        return payload

    if args.lossless_handler == "baseline":
        baseline = evaluate_lossless_baseline_submission(
            profile=args.profile,
            split=args.split,
            work_dir=Path(args.work_dir),
        )
        payload = {
            "command": "lossless_baseline",
            **baseline,
        }
        print(json.dumps(payload, indent=2))
        return payload

    if args.lossless_handler == "compress":
        archive_path = Path(args.output)
        decompressed_path = Path(args.decompressed_output)
        compression = compress_lossless_file(
            profile=args.profile,
            input_path=Path(args.input),
            output_path=archive_path,
        )
        decompressed = decompress_lossless_file(
            profile=args.profile,
            archive_path=archive_path,
            output_path=decompressed_path,
        )
        verification_compression, verification = evaluate_lossless_archive(
            profile=args.profile,
            original_tokens=Path(args.input).read_bytes(),
            decompressed_tokens=decompressed.read_bytes(),
            archive_path=archive_path,
            archive_bytes=archive_path.stat().st_size,
            method=compression.method,
        )
        payload = {
            "command": "lossless_compress",
            "compression": asdict(verification_compression),
            "verification": asdict(verification),
        }
        print(json.dumps(payload, indent=2))
        return payload

    if args.lossless_handler == "evaluate":
        original = Path(args.original).read_bytes()
        decompressed = Path(args.decompressed).read_bytes()
        archive = Path(args.archive)
        compression, verification = evaluate_lossless_archive(
            profile=args.profile,
            original_tokens=original,
            decompressed_tokens=decompressed,
            archive_path=archive,
            archive_bytes=archive.stat().st_size,
            method=args.method,
        )
        payload = {
            "command": "lossless_evaluate",
            "compression": asdict(compression),
            "verification": asdict(verification),
        }
        print(json.dumps(payload, indent=2))
        return payload

    if args.lossless_handler == "promote":
        payload = promote_lossless_result(repo_root=args.repo_root, result_path=args.result_json)
        print(json.dumps(payload, indent=2))
        return payload

    raise SystemExit(f"Unknown lossless subcommand: {args.lossless_handler}")


def main(argv: list[str] | None = None) -> Any:
    args = build_parser().parse_args(argv)
    if args.command == "lossy":
        return _run_lossy(args)
    if args.command == "lossless":
        return _run_lossless(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    result = main()
    raise SystemExit(result if isinstance(result, int) else 0)
