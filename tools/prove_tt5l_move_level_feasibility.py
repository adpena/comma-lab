#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit a TT5L move-level structural feasibility proof artifact.

This is a structural mechanism proof, not a score claim. It verifies that the
implemented TT5L packet exposes the five required move-level mechanisms and
binds the contest-axis constraints to the existing Dykstra score-axis sanity
artifact. The output can then be canonicalized by
``tools/build_tt5l_move_level_feasibility_artifact.py``.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from tac.codec.cooperative_receiver import (  # noqa: E402
    AtickRedlichWeights,
    PredictiveCodingWeights,
    cooperative_receiver_loss,
    predictive_coding_residual_term,
)
from tac.optimization.l5_staircase_v2 import (  # noqa: E402
    TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH,
    TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS,
    TT5L_DYKSTRA_SUBSTRATE_ID,
    TT5L_MOVE_LEVEL_FEASIBILITY_PREDICATE_ID,
)
from tac.substrates.time_traveler_l5_autonomy.architecture import (  # noqa: E402
    TimeTravelerConfig,
    TimeTravelerSubstrate,
)
from tac.substrates.time_traveler_l5_autonomy.archive import (  # noqa: E402
    pack_archive,
    parse_archive,
    parse_tt5l_archive_bytes,
)

PROOF_SCHEMA = "tt5l_move_level_structural_feasibility_proof_v1"
PROOF_TOOL_PATH = "tools/prove_tt5l_move_level_feasibility.py"
DEFAULT_PROOF_PATH = (
    ".omx/research/tt5l_move_level_structural_proof_20260516_codex.json"
)
RESIDUAL_TOLERANCE = 1e-12


class _StandinSegScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 5, kernel_size=1, bias=False)

    def preprocess_input(self, x_btchw: torch.Tensor) -> torch.Tensor:
        return x_btchw[:, -1]

    def forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
        return self.conv(x_bchw)


class _StandinPoseScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(12, 6, bias=False)

    def preprocess_input(self, x_btchw: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x_btchw.shape
        gray = x_btchw.reshape(b * t, c, h, w).mean(dim=1, keepdim=True)
        pose_input = gray.expand(-1, 6, -1, -1).reshape(b, t * 6, h, w)
        return pose_input[:, :, ::2, ::2]

    def forward(self, x_b12hw: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"pose": self.proj(x_b12hw.flatten(2).mean(dim=2))}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, *, repo_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(path)


def _resolve_repo_path(value: str | Path, *, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"path must be inside repo root: {resolved}") from exc
    return resolved


def _record(
    constraint_id: str,
    *,
    passed: bool,
    details: dict[str, Any],
    residual: float = 0.0,
) -> dict[str, Any]:
    return {
        "constraint_id": constraint_id,
        "passed": bool(passed),
        "residual": float(residual if passed else max(residual, 1.0)),
        "details": details,
    }


def _load_score_axis_sanity(
    path: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("score-axis sanity artifact must be a JSON object")
    return {
        "path": _display_path(path, repo_root=repo_root),
        "sha256": _sha256_file(path),
        "payload": payload,
    }


def _contest_axis_records(score_axis: dict[str, Any]) -> list[dict[str, Any]]:
    payload = score_axis["payload"]
    feasible = payload.get("verdict") == "FEASIBLE"
    exact_scope = payload.get("move_level_constraint_proof") is False
    details = {
        "score_axis_artifact_path": score_axis["path"],
        "score_axis_artifact_sha256": score_axis["sha256"],
        "dykstra_verdict": payload.get("verdict"),
        "dykstra_move_level_constraint_proof": payload.get(
            "move_level_constraint_proof"
        ),
        "authority_scope": "contest_axes_bound_to_dykstra_sanity_not_reproved_here",
    }
    return [
        _record("contest_rate_budget", passed=feasible and exact_scope, details=details),
        _record("contest_seg_dist_budget", passed=feasible and exact_scope, details=details),
        _record("contest_pose_dist_budget", passed=feasible and exact_scope, details=details),
    ]


def _renderer_gradient_record() -> dict[str, Any]:
    cfg = TimeTravelerConfig(
        hidden_dim=8,
        num_hidden_layers=1,
        num_pairs=2,
        output_height=8,
        output_width=12,
        foveation_grid_h=4,
        foveation_grid_w=6,
        coord_feature_freqs=2,
    )
    substrate = TimeTravelerSubstrate(cfg)
    rgb_0, rgb_1 = substrate.render_pair(1)
    loss = rgb_0.mean() + rgb_1.mean()
    loss.backward()
    grad_by_group = {
        "renderer": sum(
            float(p.grad.abs().sum() if p.grad is not None else torch.tensor(0.0))
            for p in substrate.renderer.parameters()
        ),
        "foveation": sum(
            float(p.grad.abs().sum() if p.grad is not None else torch.tensor(0.0))
            for p in substrate.foveation.parameters()
        ),
        "dynamics": sum(
            float(p.grad.abs().sum() if p.grad is not None else torch.tensor(0.0))
            for p in substrate.dynamics.parameters()
        ),
        "pose_codes": float(
            substrate.pose_codes.grad.abs().sum()
            if substrate.pose_codes.grad is not None
            else torch.tensor(0.0)
        ),
    }
    passed = all(value > 0.0 for value in grad_by_group.values())
    return _record(
        "tt5l_differentiable_world_model",
        passed=passed,
        details={
            "output_shapes": [list(rgb_0.shape), list(rgb_1.shape)],
            "grad_abs_sum_by_group": grad_by_group,
            "proof_scope": "tiny_cpu_structural_gradient_probe",
        },
    )


def _foveation_ego_motion_record() -> dict[str, Any]:
    cfg = TimeTravelerConfig(
        hidden_dim=8,
        num_hidden_layers=1,
        num_pairs=2,
        output_height=8,
        output_width=12,
        foveation_grid_h=4,
        foveation_grid_w=6,
        coord_feature_freqs=2,
    )
    substrate = TimeTravelerSubstrate(cfg)
    with torch.no_grad():
        base_0, base_1 = substrate.render_pair(1)
        substrate.pose_codes[1, 0] += 0.5
        substrate.foveation.grid_weights.add_(0.1)
        changed_0, changed_1 = substrate.render_pair(1)
    max_delta = float(
        max(
            (changed_0 - base_0).abs().max().item(),
            (changed_1 - base_1).abs().max().item(),
        )
    )
    return _record(
        "tt5l_ego_motion_foveation",
        passed=max_delta > 0.0,
        details={
            "max_output_delta_after_pose_and_foveation_perturb": max_delta,
            "pose_dim": cfg.pose_dim,
            "foveation_grid": [cfg.foveation_grid_h, cfg.foveation_grid_w],
        },
    )


def _predictive_coding_record() -> dict[str, Any]:
    residual = torch.randn(2, 45, requires_grad=True)
    out = predictive_coding_residual_term(
        residual,
        weights=PredictiveCodingWeights(delta_predict=0.1),
    )
    out.scaled_term.backward()
    grad_sum = float(residual.grad.abs().sum() if residual.grad is not None else 0.0)
    return _record(
        "tt5l_predictive_coding_hierarchy",
        passed=bool(out.scaled_term.item() > 0.0 and grad_sum > 0.0),
        details={
            "scaled_term": float(out.scaled_term.detach()),
            "unscaled_residual_l2": float(out.unscaled_residual_l2.detach()),
            "residual_grad_abs_sum": grad_sum,
        },
    )


def _cooperative_receiver_record() -> dict[str, Any]:
    rgb_0 = (torch.rand(1, 3, 16, 24) * 255.0).requires_grad_(True)
    rgb_1 = (torch.rand(1, 3, 16, 24) * 255.0).requires_grad_(True)
    gt_0 = torch.rand(1, 3, 16, 24) * 255.0
    gt_1 = torch.rand(1, 3, 16, 24) * 255.0
    out = cooperative_receiver_loss(
        rgb_0,
        rgb_1,
        gt_0,
        gt_1,
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        weights=AtickRedlichWeights(),
        eval_roundtrip_fn=lambda x: x,
    )
    out.cooperative_loss.backward()
    grad_sum = float(
        (rgb_0.grad.abs().sum() if rgb_0.grad is not None else torch.tensor(0.0))
        + (rgb_1.grad.abs().sum() if rgb_1.grad is not None else torch.tensor(0.0))
    )
    return _record(
        "tt5l_cooperative_receiver",
        passed=bool(torch.isfinite(out.cooperative_loss).item() and grad_sum > 0.0),
        details={
            "cooperative_loss": float(out.cooperative_loss.detach()),
            "seg_term": float(out.seg_term.detach()),
            "pose_term": float(out.pose_term.detach()),
            "rgb_grad_abs_sum": grad_sum,
            "proof_scope": "standin_scorer_gradient_contract",
        },
    )


def _archive_roundtrip_record() -> dict[str, Any]:
    cfg = TimeTravelerConfig(
        hidden_dim=8,
        num_hidden_layers=1,
        num_pairs=2,
        output_height=8,
        output_width=12,
        foveation_grid_h=4,
        foveation_grid_w=6,
        coord_feature_freqs=2,
    )
    substrate = TimeTravelerSubstrate(cfg)
    side_info = np.zeros((cfg.num_pairs, cfg.per_pair_side_info_bytes), dtype=np.int8)
    meta = {
        "coord_feature_freqs": cfg.coord_feature_freqs,
        "first_omega": cfg.first_omega,
        "hidden_omega": cfg.hidden_omega,
        "int8_scale": 64.0,
        "markov_transition_band": cfg.markov_transition_band,
    }
    archive_bytes = pack_archive(
        world_model_state_dict=substrate.state_dict(),
        per_pair_side_info=side_info,
        meta=meta,
        num_pairs=cfg.num_pairs,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
        foveation_grid_h=cfg.foveation_grid_h,
        foveation_grid_w=cfg.foveation_grid_w,
        pose_dim=cfg.pose_dim,
        per_pair_bytes=cfg.per_pair_side_info_bytes,
        ac_state=b"tt5l-ac-state",
    )
    parsed = parse_archive(archive_bytes)
    sections = parse_tt5l_archive_bytes(archive_bytes)
    expected_sections = {
        "tt5l_header",
        "world_model_blob",
        "per_pair_side_info_blob",
        "ac_state_blob",
        "meta_blob",
    }
    passed = (
        parsed.per_pair_side_info.shape == side_info.shape
        and expected_sections.issubset(sections)
        and all(length > 0 for _start, length in sections.values())
    )
    return _record(
        "tt5l_archive_move_roundtrip",
        passed=passed,
        details={
            "archive_bytes": len(archive_bytes),
            "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
            "sections": {
                key: {"start": start, "length": length}
                for key, (start, length) in sections.items()
            },
            "per_pair_side_info_shape": list(parsed.per_pair_side_info.shape),
        },
    )


def _tikhonov_rate_regularization_record(repo_root: Path) -> dict[str, Any]:
    trainer_path = repo_root / "experiments/train_substrate_time_traveler_l5_autonomy.py"
    source = trainer_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    has_weight_decay_flag = "--weight-decay" in source and "default=1e-5" in source
    has_adamw_weight_decay = "AdamW" in source and "weight_decay=args.weight_decay" in source
    has_argparse = any(
        isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--weight-decay"
        for node in ast.walk(tree)
    )
    return _record(
        "tt5l_tikhonov_rate_regularization",
        passed=has_weight_decay_flag and has_adamw_weight_decay and has_argparse,
        details={
            "trainer_path": _display_path(trainer_path, repo_root=repo_root),
            "weight_decay_flag_declared": has_weight_decay_flag,
            "argparse_flag_observed": has_argparse,
            "adamw_weight_decay_threaded": has_adamw_weight_decay,
            "implementation_kind": "AdamW_decoupled_weight_decay_regularization",
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--score-axis-sanity-artifact",
        default=TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH,
    )
    parser.add_argument("--output-json", default=DEFAULT_PROOF_PATH)
    return parser


def _build_payload(args: argparse.Namespace, *, repo_root: Path) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    score_axis_artifact = _resolve_repo_path(
        args.score_axis_sanity_artifact,
        repo_root=repo_root,
    )
    if not score_axis_artifact.is_file():
        raise FileNotFoundError(f"score-axis sanity artifact missing: {score_axis_artifact}")
    score_axis = _load_score_axis_sanity(score_axis_artifact, repo_root=repo_root)
    records = [
        *_contest_axis_records(score_axis),
        _predictive_coding_record(),
        _cooperative_receiver_record(),
        _foveation_ego_motion_record(),
        _renderer_gradient_record(),
        _tikhonov_rate_regularization_record(repo_root),
        _archive_roundtrip_record(),
    ]
    required_records = [
        record
        for record in records
        if record["constraint_id"] in TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS
    ]
    witness_variables = {
        str(record["constraint_id"]): record["details"]
        for record in required_records
        if record.get("passed") is True and isinstance(record.get("details"), dict)
    }
    residual_max = max(float(record["residual"]) for record in required_records)
    blockers = [
        str(record["constraint_id"])
        for record in required_records
        if record["passed"] is not True
    ]
    predicate_passed = residual_max <= RESIDUAL_TOLERANCE and not blockers
    return {
        "schema": PROOF_SCHEMA,
        "subject_id": TT5L_DYKSTRA_SUBSTRATE_ID,
        "predicate_id": TT5L_MOVE_LEVEL_FEASIBILITY_PREDICATE_ID,
        "predicate_passed": predicate_passed,
        "move_level_constraint_proof": predicate_passed,
        "proof_type": "structural_mechanism_feasibility",
        "authority_scope": (
            "structural_move_level_mechanism_proof_only_not_score_claim_not_promotion"
        ),
        "residual_max": residual_max,
        "residual_tolerance": RESIDUAL_TOLERANCE,
        "constraint_set_ids": sorted(TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS),
        "constraint_set_count": len(TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS),
        "mechanism_records": records,
        "witness_variables": witness_variables,
        "score_axis_sanity_artifact_path": score_axis["path"],
        "score_axis_sanity_artifact_sha256": score_axis["sha256"],
        "generated_by_tool": PROOF_TOOL_PATH,
        "tool_sha256": _sha256_file(repo_root / PROOF_TOOL_PATH),
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "command_argv": sys.argv if sys.argv else [PROOF_TOOL_PATH],
        "blockers": blockers,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_path = Path(args.output_json).expanduser()
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    try:
        payload = _build_payload(args, repo_root=repo_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        print(f"[tt5l-move-proof] FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"artifact_path": str(output_path), "predicate_passed": payload["predicate_passed"]}, sort_keys=True))
    return 0 if payload["predicate_passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
