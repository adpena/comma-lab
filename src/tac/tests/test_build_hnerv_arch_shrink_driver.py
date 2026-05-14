# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

from tac.hnerv_arch_schema import HNeRVArchConfig, generate_hnerv_state_schema

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "build_hnerv_arch_shrink_driver.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("build_hnerv_arch_shrink_driver", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _state_dict(config: HNeRVArchConfig) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, shape in generate_hnerv_state_schema(config):
        state[name] = torch.ones(*shape, dtype=torch.float32)
    return state


def test_build_driver_artifacts_emit_fail_closed_manifest(tmp_path: Path) -> None:
    tool = _load_tool()
    source_path = tmp_path / "source.pt"
    torch.save(
        _state_dict(HNeRVArchConfig(latent_dim=4, base_channels=8, eval_size=(64, 64))),
        source_path,
    )

    manifest = tool.build_driver_artifacts(
        source_state_dict_path=source_path,
        output_dir=tmp_path / "out",
        scenario_name="unit_shrink",
        target_config=HNeRVArchConfig(latent_dim=4, base_channels=5, eval_size=(64, 64)),
        started_at_utc="2026-05-07T00:00:00Z",
    )

    assert manifest["schema"] == "hnerv_arch_shrink_training_driver.v1"
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "exact_cuda_auth_eval_missing" in manifest["dispatch_blockers"]
    assert Path(manifest["outputs"]["initial_state_dict_path"]).is_file()
    assert Path(manifest["outputs"]["generated_schema_path"]).is_file()
    saved = torch.load(
        manifest["outputs"]["initial_state_dict_path"],
        map_location="cpu",
        weights_only=True,
    )
    assert tuple(saved["stem.weight"].shape) == (5 * 6 * 8, 4)


def test_cli_writes_manifest_and_generated_schema(tmp_path: Path) -> None:
    tool = _load_tool()
    source_path = tmp_path / "source.pt"
    torch.save(
        _state_dict(HNeRVArchConfig(latent_dim=4, base_channels=8, eval_size=(64, 64))),
        source_path,
    )
    output_dir = tmp_path / "out"

    assert tool.main([
        "--source-state-dict",
        str(source_path),
        "--base-channels",
        "5",
        "--latent-dim",
        "4",
        "--eval-height",
        "64",
        "--eval-width",
        "64",
        "--output-dir",
        str(output_dir),
        "--scenario-name",
        "unit_cli",
    ]) == 0

    manifest = json.loads((output_dir / "training_driver_manifest.json").read_text())
    schema = json.loads((output_dir / "generated_schema.json").read_text())
    assert manifest["scenario_name"] == "unit_cli"
    assert schema["target_config"]["base_channels"] == 5
