#!/usr/bin/env python3
"""Recover PR95 hnerv_muon curriculum facts and full-run campaign estimates.

This is a deterministic forensic helper. It reads local PR95 source files and
does not import torch, load scorers, dispatch providers, or spend GPU.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto"
    / "source/submissions/hnerv_muon"
)
DEFAULT_PR_METADATA = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto"
    / "pr_metadata.json"
)

MODAL_RATES_2026_05_13_USD_PER_GPU_HOUR = {
    "A100_40GB": 2.0988,
    "A100_80GB": 2.4984,
    "H100": 3.9492,
    "L40S": 1.9512,
    "T4": 0.5904,
}

PLANNING_GPU_HOURS = {
    # Planning bands only. The first smoke run must replace these with measured
    # seconds/epoch before full spend.
    "H100": [45.0, 70.0],
    "A100_40GB": [90.0, 140.0],
    "A100_80GB": [80.0, 125.0],
    "L40S": [110.0, 170.0],
    "T4": [250.0, 400.0],
}

STOP_GATES = [
    {
        "gate": "smoke",
        "trigger": "import, parser, yuv6 grad reachability, 1-2 batches, archive parse smoke",
        "artifact": "smoke_manifest.json with source_tree_sha256, gradcheck, seconds_per_epoch_estimate",
        "stop_condition": "any import/scorer/yuv6/roundtrip/codec failure or sec/epoch exceeds cap",
    },
    {
        "gate": "stage1_ce",
        "trigger": "Stage 1 completed",
        "artifact": "stage1/final_decoder.pt, final_latents.pt, best_meta.json, best_archive.bin",
        "stop_condition": "seg CE and pose proxy do not decrease over measured windows",
    },
    {
        "gate": "stage4_qat",
        "trigger": "Stage 4 completed after QAT joins",
        "artifact": "stage4 checkpoint pair plus parsed archive roundtrip and QAT weight-delta report",
        "stop_condition": "QAT causes component collapse or archive parse/eval roundtrip fails",
    },
    {
        "gate": "stage7_pre_muon",
        "trigger": "Stage 7 completed before Muon",
        "artifact": "stage7 checkpoint pair, best_archive.bin, C1a entropy trend, section-byte manifest",
        "stop_condition": "C1a entropy does not improve or bytes/components regress beyond operator threshold",
    },
    {
        "gate": "stage8_muon",
        "trigger": "Stage 8 completed",
        "artifact": "stage8 checkpoint pair, best_archive.bin, Muon/AdamW partition manifest, archive SHA",
        "stop_condition": "Muon step is unstable, NaN, or worse than stage7 under same proxy/eval contract",
    },
    {
        "gate": "exact_cuda_cpu_eval",
        "trigger": "candidate archive.zip and runtime are byte-closed",
        "artifact": "contest_auth_eval CUDA and CPU JSON, report.txt, inflated manifest, runtime tree SHA",
        "stop_condition": "missing claim, missing custody, axis mismatch, dependency closure failure, or score regression",
    },
]


@dataclass(frozen=True)
class StageRecord:
    order: int
    name: str
    source_file: str
    epochs: int
    loss_family: str
    adamw_lr: float | None
    muon_lr: float | None
    muon_weight_decay: float | None
    cat_lambda: float | None
    cat_sigma: float | None
    uses_qat: bool
    uses_muon: bool
    resume: str


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _literal_number(pattern: str, text: str) -> float | int | None:
    match = re.search(pattern, text)
    if not match:
        return None
    value = match.group(1)
    if "." in value or "e" in value.lower():
        return float(value)
    return int(value)


def _stage_name(path: Path, text: str) -> str:
    match = re.search(r'name="([^"]+)"', text)
    return match.group(1) if match else path.stem


def _loss_family(text: str) -> str:
    if "ce_seg_loss" in text:
        return "ce_seg_loss"
    if "tau_softplus_seg_loss" in text:
        return "tau_softplus_seg_loss"
    if "smooth_disagreement_seg_loss" in text:
        return "smooth_disagreement_seg_loss"
    if "l7_softplus_seg_loss" in text:
        return "l7_softplus_seg_loss"
    return "unknown"


def parse_stage_file(path: Path, order: int) -> StageRecord:
    text = path.read_text(encoding="utf-8")
    return StageRecord(
        order=order,
        name=_stage_name(path, text),
        source_file=_rel(path),
        epochs=int(_literal_number(r"epochs:\s*int\s*=\s*(\d+)", text) or 0),
        loss_family=_loss_family(text),
        adamw_lr=_literal_number(r"adamw_lr=([0-9.eE+-]+)", text),
        muon_lr=_literal_number(r"muon_lr=([0-9.eE+-]+)", text),
        muon_weight_decay=_literal_number(
            r"muon_weight_decay:\s*float\s*=\s*([0-9.eE+-]+)", text
        ),
        cat_lambda=_literal_number(r"cat_lambda=([0-9.eE+-]+)", text),
        cat_sigma=_literal_number(r"cat_sigma=([0-9.eE+-]+)", text),
        uses_qat="use_qat=True" in text,
        uses_muon="use_muon=True" in text,
        resume="random_init" if "init_latents_random=True" in text else "previous_stage_final",
    )


def recover_curriculum(source_dir: Path) -> dict[str, Any]:
    stage_dir = source_dir / "src/stages"
    stage_files = sorted(stage_dir.glob("stage[0-9]*.py"))
    stages = [parse_stage_file(path, index) for index, path in enumerate(stage_files, start=1)]
    total_epochs = sum(stage.epochs for stage in stages)
    return {
        "source_dir": _rel(source_dir),
        "stage_count": len(stages),
        "total_epochs": total_epochs,
        "stages": [asdict(stage) for stage in stages],
        "shared_training_loop": _rel(source_dir / "src/stages/common.py"),
        "orchestrator": _rel(source_dir / "src/train.py"),
        "codec": _rel(source_dir / "src/codec.py"),
        "inflate_py": _rel(source_dir / "inflate.py"),
        "inflate_sh": _rel(source_dir / "inflate.sh"),
    }


def estimate_campaign(total_epochs: int) -> dict[str, Any]:
    stage_rows = []
    for gpu, hours in PLANNING_GPU_HOURS.items():
        low_h, high_h = hours
        rate = MODAL_RATES_2026_05_13_USD_PER_GPU_HOUR[gpu]
        stage_rows.append(
            {
                "gpu": gpu,
                "provider_rate_assumption": "Modal GPU Tasks, fetched 2026-05-13; verify live before dispatch",
                "usd_per_gpu_hour": rate,
                "gpu_hours_low": low_h,
                "gpu_hours_high": high_h,
                "cost_low_usd": round(low_h * rate, 2),
                "cost_high_usd": round(high_h * rate, 2),
                "epochs": total_epochs,
            }
        )
    return {
        "rate_assumptions": {
            "modal_usd_per_gpu_hour_2026_05_13": MODAL_RATES_2026_05_13_USD_PER_GPU_HOUR,
            "source": "https://modal.com/pricing",
            "note": "GPU-hours are planning bands, not measurements. Recompute from smoke seconds/epoch before paid full run.",
        },
        "gpu_hour_estimates": stage_rows,
        "stop_gates": STOP_GATES,
    }


def build_payload(source_dir: Path, pr_metadata: Path | None) -> dict[str, Any]:
    curriculum = recover_curriculum(source_dir)
    metadata = None
    if pr_metadata and pr_metadata.is_file():
        metadata = json.loads(pr_metadata.read_text(encoding="utf-8"))
    return {
        "schema": "pr95_curriculum_recovery_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "dispatch_attempted": False,
        "source": curriculum,
        "pr_metadata": metadata,
        "campaign": estimate_campaign(int(curriculum["total_epochs"])),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    source = payload["source"]
    lines = [
        "# PR95 Curriculum Recovery Helper Output",
        "",
        f"- source: `{source['source_dir']}`",
        f"- stages: `{source['stage_count']}`",
        f"- total epochs: `{source['total_epochs']}`",
        "- score_claim: `false`",
        "- dispatch_attempted: `false`",
        "",
        "## Stages",
        "",
        "| # | name | epochs | loss | AdamW LR | Muon LR | C1a lambda | C1a sigma | QAT | Muon | resume |",
        "|---:|---|---:|---|---:|---:|---:|---:|---|---|---|",
    ]
    for stage in source["stages"]:
        lines.append(
            "| {order} | `{name}` | {epochs} | `{loss_family}` | {adamw_lr} | {muon_lr} | "
            "{cat_lambda} | {cat_sigma} | {uses_qat} | {uses_muon} | `{resume}` |".format(**stage)
        )
    lines.extend(
        [
            "",
            "## GPU-Hour Campaign Estimates",
            "",
            "| GPU | hours low | hours high | USD/hr | cost low | cost high |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["campaign"]["gpu_hour_estimates"]:
        lines.append(
            "| {gpu} | {gpu_hours_low:.1f} | {gpu_hours_high:.1f} | {usd_per_gpu_hour:.4f} | "
            "${cost_low_usd:.2f} | ${cost_high_usd:.2f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Rate assumptions are planning-only and must be verified live before dispatch.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--pr-metadata", type=Path, default=DEFAULT_PR_METADATA)
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = build_payload(args.source_dir, args.pr_metadata)
    if args.format == "markdown":
        text = render_markdown(payload) + "\n"
    else:
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
