from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

from tac.sensitivity_map import save_sensitivity_map

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "sensitivity_weighted_lossy_coarsening.py"


def _load_tool():
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    spec = importlib.util.spec_from_file_location(
        "sensitivity_weighted_lossy_coarsening_under_test",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_tiny_inputs(tmp_path: Path) -> tuple[Path, Path]:
    state_dict = tmp_path / "state_dict.pt"
    sensitivity_map = tmp_path / "sensitivity.pt"
    torch.save(
        {
            "stem.weight": torch.linspace(-1.0, 1.0, steps=16).reshape(4, 4),
            "blocks.0.weight": torch.tensor(
                [[4.0, -4.0, 3.0, -3.0], [1.0, -1.0, 0.5, -0.5]],
                dtype=torch.float32,
            ),
        },
        state_dict,
    )
    save_sensitivity_map(
        sensitivity_map,
        {
            "stem.weight": torch.tensor([10.0, 10.0, 10.0, 10.0]),
            "blocks.0.weight": torch.tensor([1.0, 1.0]),
        },
        metadata={
            "device": "cuda",
            "source": "unit-certified-shape",
        },
    )
    return state_dict, sensitivity_map


def test_tool_builds_cpu_only_no_score_manifest(tmp_path: Path, monkeypatch) -> None:
    tool = _load_tool()
    monkeypatch.setattr(
        tool,
        "FIXED_STATE_SCHEMA",
        (("stem.weight", (4, 4)), ("blocks.0.weight", (2, 4))),
    )
    state_dict, sensitivity_map = _write_tiny_inputs(tmp_path)
    output_json = tmp_path / "manifest.json"

    rc = tool.main(
        [
            "--state-dict",
            str(state_dict),
            "--sensitivity-map",
            str(sensitivity_map),
            "--rms-budget",
            "0.25",
            "--max-K",
            "4",
            "--output",
            str(output_json),
        ]
    )

    assert rc == 0
    manifest = json.loads(output_json.read_text(encoding="utf-8"))
    assert manifest["schema"] == tool.SCHEMA_VERSION
    assert manifest["cpu_only"] is True
    assert manifest["remote_dispatch_allowed"] is False
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["weighted_k_allocations"][0]["selected_Ks"]
    assert manifest["implementation"]["allocator_module"] == (
        "tac.optimization.lagrangian_per_tensor_allocation"
    )
    assert manifest["implementation"]["byte_closed_packet_ladder_builder"] == (
        "tools/build_a2_sensitivity_weighted_pr101_packet.py"
    )
    assert manifest["packet_ladder_builder"]["selected_k_schedule_field"] == (
        "weighted_k_allocations[].selected_Ks"
    )


def test_tool_rejects_stub_sensitivity_unless_explicitly_allowed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = _load_tool()
    monkeypatch.setattr(
        tool,
        "FIXED_STATE_SCHEMA",
        (("stem.weight", (4, 4)), ("blocks.0.weight", (2, 4))),
    )
    state_dict, sensitivity_map = _write_tiny_inputs(tmp_path)
    save_sensitivity_map(
        sensitivity_map,
        {
            "stem.weight": torch.ones(4),
            "blocks.0.weight": torch.ones(2),
        },
        metadata={"device": "cpu", "is_stub": True},
    )
    output_json = tmp_path / "blocked.json"

    rc = tool.main(
        [
            "--state-dict",
            str(state_dict),
            "--sensitivity-map",
            str(sensitivity_map),
            "--rms-budget",
            "0.25",
            "--max-K",
            "2",
            "--output-json",
            str(output_json),
        ]
    )

    assert rc == 2
    manifest = json.loads(output_json.read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked_fail_closed"
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "diagnostic/stub sensitivity rejected" in manifest["reason"]


def test_tool_dry_run_writes_fail_closed_manifest(tmp_path: Path) -> None:
    tool = _load_tool()
    output_json = tmp_path / "dry.json"

    rc = tool.main(["--dry-run", "--output", str(output_json)])

    assert rc == 0
    manifest = json.loads(output_json.read_text(encoding="utf-8"))
    assert manifest["status"] == "dry_run_fail_closed"
    assert manifest["dispatch_attempted"] is False
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
