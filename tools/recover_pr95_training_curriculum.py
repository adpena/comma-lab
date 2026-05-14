#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
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

CURRICULUM_MUTATIONS = [
    {
        "id": "control_pr95_exact_replay",
        "priority": "P0",
        "kind": "control",
        "target_stage": "stages_1_8",
        "hypothesis": "The public eight-stage schedule is the control arm; no mutation is interpretable until the source-faithful path has timing, manifests, and archive custody.",
        "implementation": "Port architecture, staged losses, QAT, C1a, differentiable YUV6, eval roundtrip, EMA selection, Muon partition, and archive parse/build without changing stage semantics.",
        "exact_gate": "Full source-faithful stage manifests plus byte-closed archive/runtime exact CUDA and CPU eval.",
        "dispatch_eligible": False,
        "why_non_arbitrary": "Defines the baseline distribution and prevents mutation wins from being harness, parser, or export drift.",
    },
    {
        "id": "timing_smoke_epoch_budget",
        "priority": "P0",
        "kind": "measurement",
        "target_stage": "smoke",
        "hypothesis": "Real seconds per epoch, scorer import cost, and archive parse cost determine whether the full burn should use H100, A100, L40S, or T4.",
        "implementation": "Run 1-2 batches with gradient-reachable scorer preprocess, eval roundtrip, archive parser smoke, and dependency/import probes.",
        "exact_gate": "smoke_manifest.json with hardware, seconds_per_epoch, source tree SHA, grad reachability, and no score claim.",
        "dispatch_eligible": False,
        "why_non_arbitrary": "Replaces stale cost priors with measured throughput before any long paid run.",
    },
    {
        "id": "score_domain_stage_boundary_controller",
        "priority": "P1",
        "kind": "curriculum_control",
        "target_stage": "stages_1_8",
        "hypothesis": "Fixed epoch counts are likely suboptimal; transitions should depend on exported-archive component deltas, entropy trend, and plateau detection.",
        "implementation": "Keep PR95 losses but allow CE->softplus->smooth->QAT->C1a->Muon transitions when component plateau and archive-byte entropy conditions are met.",
        "exact_gate": "For every transition, emit candidate archive bytes/SHA, seg/pose/rate components, plateau window, and the counterfactual fixed-stage checkpoint.",
        "dispatch_eligible": False,
        "why_non_arbitrary": "The contest objective, not calendar epoch count, chooses stage boundaries.",
    },
    {
        "id": "earlier_muon_partition_sweep",
        "priority": "P1",
        "kind": "optimizer",
        "target_stage": "stage5_to_stage8",
        "hypothesis": "Muon only in Stage 8 may leave curvature-aligned improvement unused during the long C1a phase.",
        "implementation": "Sweep Muon start at Stage 5, Stage 6, Stage 7, and Stage 8 while preserving the PR95 hidden-2D+ vs AdamW parameter partition as a controlled variable.",
        "exact_gate": "Per-arm Muon/AdamW partition manifest, optimizer state hash, Stage 7 and Stage 8 archive component deltas, NaN/instability checks.",
        "dispatch_eligible": False,
        "why_non_arbitrary": "Tests a single optimizer-timing variable against the same architecture, losses, and export path.",
    },
    {
        "id": "dual_ema_archive_selector",
        "priority": "P1",
        "kind": "selection",
        "target_stage": "all_eval_points",
        "hypothesis": "A single EMA decay can be wrong for rate, pose, and segmentation simultaneously; archive selection should compare raw, fast EMA, and slow EMA.",
        "implementation": "Maintain raw weights plus at least two EMA shadows and build scored archive candidates at the same eval cadence.",
        "exact_gate": "Candidate archive manifest for each shadow, byte term, seg/pose components, selected winner reason, and no hidden evaluator state.",
        "dispatch_eligible": False,
        "why_non_arbitrary": "Selection is based on byte-closed archive components instead of proxy loss smoothness.",
    },
    {
        "id": "hard_pair_waterfill_sampler",
        "priority": "P1",
        "kind": "data_schedule",
        "target_stage": "stages_2_8",
        "hypothesis": "Uniform pair sampling under-spends training on high-marginal-value pose/seg pairs near the frontier.",
        "implementation": "Use component-response and score-gradient maps to oversample hard pairs with a floor probability for all pairs; record pair weights per epoch.",
        "exact_gate": "Pair-weight manifest, before/after component deltas by pair/category, and exported archive score on the full fixed video.",
        "dispatch_eligible": False,
        "why_non_arbitrary": "The water-fill weights are derived from measured marginal score contribution, not handpicked frames.",
    },
    {
        "id": "c1a_rate_schedule_grid",
        "priority": "P1",
        "kind": "rate_regularization",
        "target_stage": "stages_5_7",
        "hypothesis": "PR95's lambda/sigma choices are sparse; a smoother schedule can reduce entropy without crossing pose/seg cliffs.",
        "implementation": "Grid or bandit-search C1a lambda and sigma schedules, with per-section entropy and decoded-component guardrails.",
        "exact_gate": "Section-byte manifest, entropy trend, component cliff report, and exact archive parse/build roundtrip.",
        "dispatch_eligible": False,
        "why_non_arbitrary": "Rate pressure is accepted only when exact byte savings exceed measured component loss.",
    },
    {
        "id": "quantization_native_training",
        "priority": "P1",
        "kind": "quantization",
        "target_stage": "stages_3_8",
        "hypothesis": "Late QAT may learn float features that are brittle under the PR101-style microcodec; quantization-native training can improve final bytes at the same distortion.",
        "implementation": "Introduce fake-quant earlier, sweep per-tensor/per-channel routes, and keep apply/restore tests for every QAT step.",
        "exact_gate": "QAT delta report, quantized export roundtrip, archive-byte comparison, and scorer component preservation.",
        "dispatch_eligible": False,
        "why_non_arbitrary": "Optimizes the representation actually consumed by the archive rather than a float proxy.",
    },
    {
        "id": "pr101_microcodec_export_over_pr95_weights",
        "priority": "P1",
        "kind": "archive_export",
        "target_stage": "post_stage8",
        "hypothesis": "PR101's score came from codec/runtime polish over PR95-family weights; improved PR95 weights should be exported through that byte discipline.",
        "implementation": "Emit PR101-style schema-driven sections, split Brotli streams, compact latent packing, and sidecar/no-op table selection over each improved checkpoint.",
        "exact_gate": "Byte-identical no-op control, member SHA manifest, parser consumption proof, inflate runtime SHA, and exact CUDA/CPU eval.",
        "dispatch_eligible": False,
        "why_non_arbitrary": "Separates representation gains from microcodec gains while preserving apples-to-apples archive custody.",
    },
    {
        "id": "score_aware_residual_atom_stack",
        "priority": "P2",
        "kind": "composition",
        "target_stage": "post_base_anchor",
        "hypothesis": "The best sub-0.17 route is likely PR95/PR101 as a base plus small typed residual atoms for scorer-sensitive errors.",
        "implementation": "Attach byte-closed SABOR, S2SBS, SIREN/FINER/WIRE, wavelet, LA-pose/telescope foveation, or scorer-inverse atoms only after a verified base archive anchor.",
        "exact_gate": "Each atom carries typed input/output contract, bytes/SHA, parser proof, consumed-byte proof, component deltas, and no proxy score authority.",
        "dispatch_eligible": False,
        "why_non_arbitrary": "Composition is gated on measured residual value per byte, not on paper novelty.",
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


def curriculum_mutation_matrix() -> list[dict[str, Any]]:
    return [dict(row) for row in CURRICULUM_MUTATIONS]


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
        "mutation_matrix": curriculum_mutation_matrix(),
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
            "## Curriculum Mutation Matrix",
            "",
            "| priority | id | kind | target stage | exact gate |",
            "|---|---|---|---|---|",
        ]
    )
    for row in payload["mutation_matrix"]:
        lines.append(
            "| {priority} | `{id}` | `{kind}` | `{target_stage}` | {exact_gate} |".format(
                **row
            )
        )
    lines.append("")
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
