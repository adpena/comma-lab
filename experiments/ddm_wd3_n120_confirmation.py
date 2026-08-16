#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_wd3 n120 seeded-stratified negative-confirmation harness (eval-only).

The sealed wd3 spec law forbids emitting a family negative from n60 evidence; the
trainer stamps ``n120_negative_confirmation_run: false`` and provides the verdict
compiler (``compile_n120_negative_confirmation``, trainer:2551) but NO eval-only
subcommand. This harness is that missing invocation path — it produces the two
retained receiver-closed ``ddm_wd3_retained_subset_evaluation.v1`` rows the compiler
demands (candidate + matched baseline, same seeded nonprefix n120, same cache
surface) by REUSING the trainer's own ``evaluate_subset_and_retain`` on retained
stage checkpoints. Same instrument by construction (batch-shape law honored: the
n120 eval chunks with the SAME ``chunk_pairs`` the n60 evals used).

Axis: [Darwin-mps or cpu / n120 subset advisory — NEVER a score]. research_only.
No training; no optimizer step ever runs (the optimizer/scheduler objects exist
only because ``load_checkpoint`` restores their state to verify checkpoint
integrity). All payloads retained per ALWAYS-KEEP-THE-PAYLOAD.

Usage:
  evaluate   --arm-config <COMPILED_CONFIG.json> --checkpoint <wd3 ckpt .pt>
             --output <dir>            # writes <dir>/N120_EVALUATION.json
  adjudicate --arm <name> --candidate <N120_EVALUATION.json>
             --baseline <N120_EVALUATION.json> --output <dir>
                                       # writes <dir>/N120_VERDICT_<arm>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from experiments import ddm_wd2_width_distillation_build as wd2_build
from experiments.ddm_wd3_scorer_aware_width_distillation import (
    ARM_SPECS,
    CHECKPOINT_SCHEMA,
    REPO,
    WD3Error,
    _device,
    _load_cache_result,
    _new_optimizer_scheduler,
    assert_governed_admission,
    atomic_json,
    compile_n120_negative_confirmation,
    evaluate_subset_and_retain,
    file_record,
    load_checkpoint,
    load_differentiable_scorers,
    receiver,
    seed_everything,
    validate_compiled_config,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WD3Error(f"{path} is not a JSON object")
    return value


def _negative_ids(config: Mapping[str, Any]) -> np.ndarray:
    ids = np.asarray(config["subsets"]["negative_n120"], dtype=np.int64)
    # Mirror the verdict compiler's fail-closed shape check at evaluate time so a
    # wrong subset is refused BEFORE any scorer forward, not at adjudication.
    if ids.size != 120 or np.unique(ids).size != 120 or ids.tolist() == list(range(120)):
        raise WD3Error("config negative subset is not seeded nonprefix n120")
    if config["subsets"].get("negative_kind") != "seeded_stratified_random" or config["subsets"].get("prefix"):
        raise WD3Error("config negative subset kind/prefix contract differs")
    return ids


def evaluate_arm(*, arm_config: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    config = _load_json(arm_config)
    validation = validate_compiled_config(config)
    assert_governed_admission("ddm_wd3_n120_confirmation_evaluate")
    seed_everything(int(config["seed"]))
    device = _device(str(config["device"]))
    cache_receipt, cache = _load_cache_result(Path(config["teacher_cache_result"]))
    tokens = wd2_build._load_tokens()
    posenet, segnet = load_differentiable_scorers(REPO / "upstream", device=device)
    posenet.eval()
    segnet.eval()
    negative = _negative_ids(config)
    arm = str(config["arm"])
    model = receiver.StudentSemanticRenderer(ARM_SPECS[arm]).to(device)
    optimizer, scheduler = _new_optimizer_scheduler(model, config)
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
    resume = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if resume.get("schema") != CHECKPOINT_SCHEMA:
        raise WD3Error("n120 evaluation requires a retained WD3 stage checkpoint")
    ema = wd2_build.DeploymentEMA(model, float(resume["ema"]["decay"]))
    payload, _, allocation = load_checkpoint(
        checkpoint,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        generator=generator,
        expected_config=config,
    )
    if list(map(int, payload["subset_ids"]["negative_n120"])) != negative.tolist():
        raise WD3Error("checkpoint negative subset differs from the compiled config")
    output.mkdir(parents=True, exist_ok=True)
    with wd2_build.ema_scope(model, ema):
        evaluation = evaluate_subset_and_retain(
            root=output / "retained" / f"{arm}_n120",
            model=model,
            allocation=allocation,
            pair_ids=negative,
            tokens=tokens,
            cache=cache,
            posenet=posenet,
            segnet=segnet,
            device=device,
            chunk_pairs=int(config["chunk_pairs"]),
        )
    receipt = {
        "schema": "ddm_wd3_n120_evaluation_receipt.v1",
        "arm": arm,
        "checkpoint": file_record(checkpoint),
        "checkpoint_stage": resume.get("stage"),
        "config_sha256": validation["config_sha256"],
        "cache_receipt": file_record(Path(config["teacher_cache_result"])),
        "deployment_weights": "ema_shadow",
        "population": "seeded_stratified_random_nonprefix_n120",
        "score_claim": False,
        "promotion_eligible": False,
        "research_only": True,
        "evaluation": evaluation,
    }
    atomic_json(output / "N120_EVALUATION.json", receipt)
    return receipt


def adjudicate(*, arm: str, candidate: Path, baseline: Path, output: Path) -> dict[str, Any]:
    candidate_receipt = _load_json(candidate)
    baseline_receipt = _load_json(baseline)
    for label, receipt in (("candidate", candidate_receipt), ("baseline", baseline_receipt)):
        if receipt.get("schema") != "ddm_wd3_n120_evaluation_receipt.v1":
            raise WD3Error(f"{label} receipt schema differs")
    expected = candidate_receipt["evaluation"]["pair_ids"]
    verdict = compile_n120_negative_confirmation(
        arm=arm,
        candidate=candidate_receipt["evaluation"],
        matched_baseline=baseline_receipt["evaluation"],
        expected_pair_ids=expected,
    )
    verdict["candidate_receipt"] = {
        "arm": candidate_receipt["arm"],
        "checkpoint": candidate_receipt["checkpoint"],
    }
    verdict["baseline_receipt"] = {
        "arm": baseline_receipt["arm"],
        "checkpoint": baseline_receipt["checkpoint"],
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / f"N120_VERDICT_{arm}.json", verdict)
    return verdict


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--arm-config", type=Path, required=True)
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ad = sub.add_parser("adjudicate")
    ad.add_argument("--arm", type=str, required=True)
    ad.add_argument("--candidate", type=Path, required=True)
    ad.add_argument("--baseline", type=Path, required=True)
    ad.add_argument("--output", type=Path, required=True)
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "evaluate":
        result = evaluate_arm(arm_config=args.arm_config, checkpoint=args.checkpoint, output=args.output)
        summary = {
            "arm": result["arm"],
            "hard_d_seg": result["evaluation"]["hard_d_seg"],
            "d_pose": result["evaluation"]["d_pose"],
            "n_pairs": result["evaluation"]["n_pairs"],
        }
    else:
        summary = adjudicate(
            arm=args.arm, candidate=args.candidate, baseline=args.baseline, output=args.output
        )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
