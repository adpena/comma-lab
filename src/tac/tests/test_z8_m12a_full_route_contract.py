# SPDX-License-Identifier: MIT
"""Z8 M12a active-route contract tests.

These are fail-closed wiring tests for the post-Yousfi M12a route:
``Z8_TRAINER_MODE=full`` must be the active dispatch path, and that path must
bind the real SegNet + PoseNet teacher bundle rather than the older M9
canonical-quadruple diagnostic loop or a scorer-blind reconstruction proxy.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER = REPO_ROOT / "experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py"
REMOTE_DRIVER = REPO_ROOT / "scripts/remote_lane_substrate_z8_hierarchical_predictive_coding.sh"
RECIPE = (
    REPO_ROOT
    / ".omx/operator_authorize_recipes/substrate_z8_hierarchical_predictive_coding_modal_t4_dispatch.yaml"
)


def _module_ast() -> ast.Module:
    return ast.parse(TRAINER.read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {TRAINER}")


def _call_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _calls(fn: ast.FunctionDef) -> list[ast.Call]:
    return [node for node in ast.walk(fn) if isinstance(node, ast.Call)]


def _call_keyword_sets(fn: ast.FunctionDef, call_name: str) -> list[set[str]]:
    return [
        {kw.arg for kw in call.keywords if kw.arg is not None}
        for call in _calls(fn)
        if _call_name(call) == call_name
    ]


def test_z8_full_parser_exposes_real_pose_and_stabilizer_flags() -> None:
    parser_fn = _function(_module_ast(), "_build_parser")
    flag_names = {
        arg.value
        for call in _calls(parser_fn)
        if _call_name(call) == "add_argument"
        for arg in call.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    }

    assert "--pose-distillation-weight" in flag_names
    assert "--allow-mock-scorer-teacher" in flag_names
    assert "--grad-clip-max-norm" in flag_names
    assert "--warmup-epochs" in flag_names
    assert "--weight-decay" in flag_names
    assert "--optimizer-kind" in flag_names


def test_z8_full_main_wires_real_segnet_posenet_teachers() -> None:
    full_fn = _function(_module_ast(), "_full_main")
    call_names = {_call_name(call) for call in _calls(full_fn)}

    assert "build_mlx_segnet_pair_teacher" in call_names
    assert "build_mlx_posenet_pair_teacher" in call_names
    assert "build_learnable_student_head" in call_names
    assert "build_learnable_pose_student_head" in call_names

    bundle_kw_sets = _call_keyword_sets(full_fn, "RendererBundle")
    required_bundle_kwargs = {
        "scorer_teacher",
        "learnable_student_head",
        "pose_distillation_weight",
        "pose_scorer_teacher",
        "learnable_pose_student_head",
        "allow_mock_scorer_teacher",
        "substrate_artifact_metadata",
    }
    assert any(required_bundle_kwargs <= kws for kws in bundle_kw_sets)

    run_kw_sets = _call_keyword_sets(full_fn, "run_mlx_score_aware_full_main")
    required_run_kwargs = {
        "ema_decay",
        "grad_clip_max_norm",
        "warmup_epochs",
        "weight_decay",
        "optimizer_kind",
    }
    assert any(required_run_kwargs <= kws for kws in run_kw_sets)


def test_z8_remote_driver_defaults_to_full_route_and_harvests_training_artifact() -> None:
    text = REMOTE_DRIVER.read_text(encoding="utf-8")

    assert 'Z8_TRAINER_MODE="${Z8_TRAINER_MODE:-full}"' in text
    assert '--pose-distillation-weight "$Z8_POSE_DISTILLATION_WEIGHT"' in text
    assert '--grad-clip-max-norm "$Z8_GRAD_CLIP_MAX_NORM"' in text
    assert '--warmup-epochs "$Z8_WARMUP_EPOCHS"' in text
    assert 'training_artifact_path = output_dir / "training_artifact.json"' in text


def test_z8_recipe_active_contract_matches_full_score_aware_route() -> None:
    recipe = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))

    assert recipe["active_dispatch_contract"] == "z8_hpc_m12a_full_score_aware_long_training"
    active = recipe["dispatch_contracts"][0]
    assert active["id"] == recipe["active_dispatch_contract"]
    assert active["builder_mode_flag"] == "Z8_TRAINER_MODE=full"
    assert "real_posenet_pose_mse" in active["canonical_compose"]
    assert recipe["env_overrides"]["Z8_TRAINER_MODE"] == "full"
