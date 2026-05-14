# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load_tool():
    repo_root = Path(__file__).resolve().parents[3]
    tool_path = repo_root / "experiments" / "dump_scorer_activations.py"
    spec = importlib.util.spec_from_file_location("dump_scorer_activations", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {tool_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_real_video_cuda_path_rejects_avvideodataset_cuda(tmp_path: Path) -> None:
    mod = _load_tool()
    video_dir = tmp_path / "videos"
    video_dir.mkdir(parents=True)
    (video_dir / "0.mkv").write_bytes(b"not a real video; guard runs before decode")

    with pytest.raises(RuntimeError, match="AVVideoDataset"):
        mod._gt_video_to_pair_input(tmp_path, frame_pair_idx=0, device=torch.device("cuda"))


def _shared_input_payload(mod, tensor: torch.Tensor) -> dict[str, object]:
    return {
        "schema": mod.SHARED_INPUT_TENSOR_SCHEMA,
        "created_by": "tools/probe_eval_loader_drift.py",
        "cell_id": "cpu_av",
        "tensor_role": mod.SHARED_INPUT_TENSOR_ROLE,
        "tensor_custody": {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": mod._tensor_sha256(tensor),
        },
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "tensor": tensor,
    }


def test_shared_input_tensor_loader_accepts_probe_artifact(tmp_path: Path) -> None:
    mod = _load_tool()
    tensor = torch.arange(1 * 2 * 3 * 4 * 3, dtype=torch.uint8).reshape(1, 2, 3, 4, 3)
    path = tmp_path / "shared_input.pt"
    torch.save(_shared_input_payload(mod, tensor), path)

    loaded, metadata = mod._load_shared_input_tensor(path, torch.device("cpu"))

    assert loaded.dtype == torch.float32
    assert torch.equal(loaded.to(torch.uint8), tensor)
    assert metadata["schema"] == mod.SHARED_INPUT_TENSOR_SCHEMA
    assert metadata["cell_id"] == "cpu_av"
    assert metadata["shared_input_contract_valid"] is True
    assert metadata["score_claim"] is False
    assert metadata["promotion_eligible"] is False
    assert metadata["artifact_sha256"]


def test_shared_input_tensor_loader_rejects_promotable_artifact(tmp_path: Path) -> None:
    mod = _load_tool()
    tensor = torch.zeros(1, 2, 3, 4, 3, dtype=torch.uint8)
    payload = _shared_input_payload(mod, tensor)
    payload["score_claim"] = True
    path = tmp_path / "bad_shared_input.pt"
    torch.save(payload, path)

    with pytest.raises(ValueError, match="score_claim=false"):
        mod._load_shared_input_tensor(path, torch.device("cpu"))


def test_shared_input_tensor_loader_rejects_bad_shape(tmp_path: Path) -> None:
    mod = _load_tool()
    tensor = torch.zeros(1, 2, 3, 4, dtype=torch.uint8)
    path = tmp_path / "bad_shape.pt"
    torch.save(_shared_input_payload(mod, tensor), path)

    with pytest.raises(ValueError, match="shape"):
        mod._load_shared_input_tensor(path, torch.device("cpu"))
