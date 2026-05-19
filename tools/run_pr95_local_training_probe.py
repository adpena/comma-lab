#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the public PR95 HNeRV trainer locally without mutating the intake tree.

This is an execution harness around the lifted PR95 source, not a rewrite of
that source. It exists so local Apple Silicon / MPS timing and gradient
portability can be measured with explicit authority labels before any full
campaign uses the results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto"
    / "source/submissions/hnerv_muon"
)
DEFAULT_PUBLIC_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto"
    / "archive.zip"
)
LANE_ID = "lane_pr95_local_mps_source_faithful_training_probe_20260519"

STAGE_MODULES: tuple[str, ...] = (
    "stage1_v328_ce",
    "stage2_v331_softplus",
    "stage3_v332_smooth",
    "stage4_v332_qat",
    "stage5_c1a_l7",
    "stage6_lambda_sweep",
    "stage7_sigma_sweep",
    "stage8_muon_finetune",
)

IGNORED_TREE_PARTS = {
    "__pycache__",
    "ckpts",
    ".git",
}


@dataclass(frozen=True)
class Pr95SourceLayout:
    source_dir: Path
    source_stack_dir: Path
    challenge_root: Path
    train_py: Path
    compress_sh: Path
    inflate_sh: Path
    public_archive: Path | None


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_layout(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    public_archive: Path = DEFAULT_PUBLIC_ARCHIVE,
) -> Pr95SourceLayout:
    source_dir = source_dir.resolve()
    source_stack_dir = source_dir / "src"
    challenge_root = source_dir.parent.parent
    train_py = source_stack_dir / "train.py"
    compress_sh = source_dir / "compress.sh"
    inflate_sh = source_dir / "inflate.sh"
    required = [
        source_stack_dir / "stages/common.py",
        train_py,
        compress_sh,
        inflate_sh,
        challenge_root / "frame_utils.py",
        challenge_root / "modules.py",
        challenge_root / "models/segnet.safetensors",
        challenge_root / "models/posenet.safetensors",
        challenge_root / "videos/0.mkv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = ", ".join(_rel(path) for path in missing)
        raise FileNotFoundError(f"PR95 source layout incomplete: {joined}")
    archive = public_archive.resolve() if public_archive.exists() else None
    return Pr95SourceLayout(
        source_dir=source_dir,
        source_stack_dir=source_stack_dir,
        challenge_root=challenge_root.resolve(),
        train_py=train_py,
        compress_sh=compress_sh,
        inflate_sh=inflate_sh,
        public_archive=archive,
    )


def source_tree_sha256(layout: Pr95SourceLayout) -> str:
    """Hash small source/config files, excluding checkpoints, videos, and weights."""

    roots = [
        layout.source_dir,
        layout.challenge_root / "pyproject.toml",
        layout.challenge_root / "uv.lock",
        layout.challenge_root / "README.md",
        layout.challenge_root / "public_test_video_names.txt",
    ]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in IGNORED_TREE_PARTS for part in path.parts):
                continue
            if path.suffix in {".pt", ".pth", ".bin", ".zip", ".mkv", ".safetensors"}:
                continue
            files.append(path)
    h = hashlib.sha256()
    for path in sorted(set(files), key=lambda p: str(p.relative_to(layout.challenge_root))):
        rel = str(path.relative_to(layout.challenge_root)).encode("utf-8")
        h.update(len(rel).to_bytes(4, "little"))
        h.update(rel)
        data = path.read_bytes()
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()


def parse_stage_epoch_overrides(raw: list[str]) -> dict[int, int]:
    overrides: dict[int, int] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"stage override must be N=EPOCHS, got {item!r}")
        left, right = item.split("=", 1)
        stage = int(left)
        epochs = int(right)
        if stage < 1 or stage > len(STAGE_MODULES):
            raise ValueError(f"stage index {stage} outside 1..{len(STAGE_MODULES)}")
        if epochs < 1:
            raise ValueError("stage epochs must be positive")
        overrides[stage] = epochs
    return overrides


def select_device(requested: str, *, allow_mps_fallback: bool) -> Any:
    import torch

    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested cuda but torch.cuda.is_available() is false")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("requested mps but torch.backends.mps.is_available() is false")
    if (
        requested == "mps"
        and _truthy(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
        and not allow_mps_fallback
    ):
        raise RuntimeError(
            "PYTORCH_ENABLE_MPS_FALLBACK is enabled; refusing MPS probe because "
            "silent CPU fallback would destroy portability evidence"
        )
    if requested not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"unsupported device {requested!r}")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _install_pr95_imports(layout: Pr95SourceLayout) -> None:
    os.environ["COMMA_CHALLENGE_ROOT"] = str(layout.challenge_root)
    for path in (layout.source_stack_dir, layout.challenge_root, layout.challenge_root.parent):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _stage_builder_modules() -> list[Any]:
    from stages import (  # type: ignore[import-not-found]
        stage1_v328_ce,
        stage2_v331_softplus,
        stage3_v332_smooth,
        stage4_v332_qat,
        stage5_c1a_l7,
        stage6_lambda_sweep,
        stage7_sigma_sweep,
        stage8_muon_finetune,
    )

    return [
        stage1_v328_ce,
        stage2_v331_softplus,
        stage3_v332_smooth,
        stage4_v332_qat,
        stage5_c1a_l7,
        stage6_lambda_sweep,
        stage7_sigma_sweep,
        stage8_muon_finetune,
    ]


def build_plan(
    *,
    layout: Pr95SourceLayout,
    output_dir: Path,
    device: str,
    full_curriculum: bool,
    stage_limit: int,
    stage_epoch_overrides: dict[int, int],
    eval_every: int | None,
    allow_mps_fallback: bool,
    seed: int,
) -> dict[str, Any]:
    selected_count = len(STAGE_MODULES) if full_curriculum else stage_limit
    selected_count = max(1, min(selected_count, len(STAGE_MODULES)))
    public_archive = (
        {
            "path": _rel(layout.public_archive),
            "sha256": _sha256_file(layout.public_archive),
            "bytes": layout.public_archive.stat().st_size,
        }
        if layout.public_archive is not None
        else None
    )
    stages = []
    for index, name in enumerate(STAGE_MODULES[:selected_count], start=1):
        stages.append({
            "index": index,
            "module": name,
            "epoch_override": stage_epoch_overrides.get(index),
            "eval_every_override": eval_every,
        })
    return {
        "schema": "pr95_local_training_probe_plan_v1",
        "lane_id": LANE_ID,
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_dir": _rel(layout.source_dir),
        "source_stack_dir": _rel(layout.source_stack_dir),
        "challenge_root": _rel(layout.challenge_root),
        "source_tree_sha256": source_tree_sha256(layout),
        "public_archive": public_archive,
        "output_dir": _rel(output_dir),
        "device_requested": device,
        "allow_mps_fallback": allow_mps_fallback,
        "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "seed": seed,
        "full_curriculum": full_curriculum,
        "stage_count": selected_count,
        "stages": stages,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "rank_or_kill_eligible": False,
        "evidence_grade": "local_training_portability_probe_advisory",
        "authority_contract": {
            "local_mps_or_cpu": "training_velocity_and_transfer_probe_only",
            "score_authority": "requires byte_closed_archive_replay_on_contest_CPU_and_contest_CUDA",
            "fallback_policy": "fail_closed_unless_allow_mps_fallback_for_debug",
        },
        "source_faithful_command": (
            f"cd {layout.challenge_root} && COMMA_CHALLENGE_ROOT=$PWD "
            "python -m submissions.hnerv_muon.src.train"
        ),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    layout = resolve_layout(args.source_dir, args.public_archive)
    output_dir = args.output_dir or (
        REPO_ROOT / "experiments/results" / f"pr95_local_training_probe_{_utc_stamp()}"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_overrides = parse_stage_epoch_overrides(args.stage_epochs)
    plan = build_plan(
        layout=layout,
        output_dir=output_dir,
        device=args.device,
        full_curriculum=args.full_curriculum,
        stage_limit=args.stage_limit,
        stage_epoch_overrides=stage_overrides,
        eval_every=args.eval_every,
        allow_mps_fallback=args.allow_mps_fallback,
        seed=args.seed,
    )
    (output_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.plan_only:
        return {"plan": plan, "manifest_path": str(output_dir / "plan.json")}

    import torch

    _install_pr95_imports(layout)
    device = select_device(args.device, allow_mps_fallback=args.allow_mps_fallback)
    _seed_everything(args.seed)

    from stages import codec_stage  # type: ignore[import-not-found]
    from stages.common import train_stage  # type: ignore[import-not-found]

    from data import get_default_video_path  # type: ignore[import-not-found]

    builders = _stage_builder_modules()
    selected_count = len(builders) if args.full_curriculum else args.stage_limit
    selected_count = max(1, min(selected_count, len(builders)))
    video_path = get_default_video_path()
    shared_state: dict[str, Any] = {}
    prev: Path | None = None
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    manifest = {
        **plan,
        "schema": "pr95_local_training_probe_manifest_v1",
        "device_selected": str(device),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "video_path": _rel(Path(video_path)),
        "started_utc": datetime.now(UTC).isoformat(),
        "ok": False,
        "results": results,
    }
    (output_dir / "manifest.started.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    try:
        for stage_index, module in enumerate(builders[:selected_count], start=1):
            stage_out = output_dir / f"stage{stage_index}_{STAGE_MODULES[stage_index - 1]}"
            if stage_index == 1:
                cfg = module.make_config(stage_out)
            else:
                assert prev is not None
                cfg = module.make_config(prev, stage_out)
            if stage_index in stage_overrides:
                cfg.epochs = stage_overrides[stage_index]
            elif not args.full_curriculum and not args.stage_epochs:
                cfg.epochs = 1
            if args.eval_every is not None:
                cfg.eval_every = args.eval_every
            stage_started = time.perf_counter()
            result = train_stage(cfg, device, video_path=video_path, shared_state=shared_state)
            stage_result = {
                **{
                    key: (str(value) if isinstance(value, Path) else value)
                    for key, value in result.items()
                },
                "stage_index": stage_index,
                "stage_module": STAGE_MODULES[stage_index - 1],
                "epochs_run": cfg.epochs,
                "eval_every": cfg.eval_every,
                "wall_seconds": time.perf_counter() - stage_started,
            }
            results.append(stage_result)
            prev = stage_out
            (output_dir / "manifest.partial.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
        if args.run_codec_stage:
            if prev is None:
                raise RuntimeError("codec stage requested before any training stage ran")
            codec_out = output_dir / "submission_archive"
            codec_started = time.perf_counter()
            codec_result = codec_stage.run_codec_stage(prev, codec_out, video_path)
            manifest["codec_stage"] = {
                **codec_result,
                "wall_seconds": time.perf_counter() - codec_started,
            }
    except Exception as exc:
        manifest.update({
            "ok": False,
            "failure_type": type(exc).__name__,
            "failure": str(exc),
            "wall_seconds": time.perf_counter() - started,
            "finished_utc": datetime.now(UTC).isoformat(),
        })
        (output_dir / "manifest.failed.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        raise
    manifest.update({
        "ok": True,
        "wall_seconds": time.perf_counter() - started,
        "finished_utc": datetime.now(UTC).isoformat(),
    })
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return {"manifest": manifest, "manifest_path": str(manifest_path)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--public-archive", type=Path, default=DEFAULT_PUBLIC_ARCHIVE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--allow-mps-fallback", action="store_true")
    parser.add_argument("--full-curriculum", action="store_true")
    parser.add_argument("--stage-limit", type=int, default=1)
    parser.add_argument(
        "--stage-epochs",
        action="append",
        default=[],
        metavar="N=EPOCHS",
        help="Override a stage epoch count, repeatable. Default smoke runs stage 1 for 1 epoch.",
    )
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--run-codec-stage", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_probe(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
