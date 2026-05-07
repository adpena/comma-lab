from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load_tool(module_name: str):
    path = REPO / "tools" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ready_lightning_candidate() -> dict:
    return {
        "candidate_id": "candidate42",
        "lane_id": "pr106_latent_sidecar",
        "archive_path": "experiments/results/candidate42/archive.zip",
        "predicted_band": [0.19, 0.24],
        "ready_for_exact_eval_dispatch": True,
        "evidence_semantics": "contest_cuda_exact_eval_positive",
        "score_claim": False,
    }


def _write_ranked_input(tmp_path: Path, candidates: list[dict]) -> Path:
    ranked = tmp_path / "ranked.json"
    ranked.write_text(json.dumps({"dispatch_ready": candidates}), encoding="utf-8")
    return ranked


def test_parallel_dispatch_lightning_command_uses_stack_dispatcher_flags() -> None:
    tool = _load_tool("parallel_dispatch_top_k")

    cmd = tool._build_dispatch_cmd(
        _ready_lightning_candidate(),
        provider="lightning",
        lane_script="scripts/legacy_should_be_ignored_for_lightning.sh",
        label_prefix="batch",
        estimated_cost=0.11,
        max_dph=0.50,
    )

    joined = " ".join(cmd)
    assert "tools/lightning_dispatch_pr106_stack.py" in joined
    assert "--lane pr106_latent_sidecar" in joined
    assert "--archive experiments/results/candidate42/archive.zip" in joined
    assert "--predicted-low 0.19" in joined
    assert "--predicted-high 0.24" in joined
    assert "--job-name batch_candidate42" in joined
    assert "--lane-script" not in cmd
    assert "--predicted-band" not in cmd


def test_parallel_dispatch_rejects_above_active_floor_archive_by_default(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = {
        **_ready_lightning_candidate(),
        "archive_size_bytes": tool.DEFAULT_ACTIVE_FLOOR_ARCHIVE_BYTES + 1,
    }
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "above_active_floor_archive_bytes" in message
    assert str(tool.DEFAULT_ACTIVE_FLOOR_ARCHIVE_BYTES + 1) in message


def test_parallel_dispatch_allows_above_active_floor_with_operator_reason(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = {
        **_ready_lightning_candidate(),
        "archive_size_bytes": tool.DEFAULT_ACTIVE_FLOOR_ARCHIVE_BYTES + 1,
    }
    ranked = _write_ranked_input(tmp_path, [candidate])

    loaded = tool._load_top_k(
        ranked,
        k=None,
        allow_above_active_floor_dispatch=True,
        operator_override_reason="calibration dispatch for non-rate-axis candidate",
    )

    assert loaded == [candidate]


def test_feedback_loop_lightning_command_uses_stack_dispatcher_flags() -> None:
    tool = _load_tool("feedback_loop_sweep")

    cmd = tool._build_dispatch_cmd(
        _ready_lightning_candidate(),
        provider="lightning",
        lane_script="scripts/legacy_should_be_ignored_for_lightning.sh",
        label_prefix="cycle",
        max_dph=0.50,
        estimated_cost=0.11,
    )

    joined = " ".join(cmd)
    assert "tools/lightning_dispatch_pr106_stack.py" in joined
    assert "--lane pr106_latent_sidecar" in joined
    assert "--archive experiments/results/candidate42/archive.zip" in joined
    assert "--predicted-low 0.19" in joined
    assert "--predicted-high 0.24" in joined
    assert "--job-name cycle_candidate42" in joined
    assert "--lane-script" not in cmd
    assert "--predicted-band" not in cmd
