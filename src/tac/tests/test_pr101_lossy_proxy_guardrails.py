# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def _load_script(relpath: str, name: str):
    path = REPO / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_mixed_precision_rel_err_ignores_zeros_and_accounts_wire_padding() -> None:
    tool = _load_script(
        "tools/pr101_lossy_mixed_precision_int4_int8.py",
        "pr101_lossy_mixed_precision_int4_int8",
    )

    stats = tool.per_tensor_rel_err_stats(
        np.array([0.0, 1.0, 2.0], dtype=np.float32),
        np.array([100.0, 2.0, 1.0], dtype=np.float32),
    )

    assert stats["n_nontrivial"] == 2
    assert stats["rel_err_pct_mean"] == 75.0
    assert tool.encoded_bytes_for_tensor(1, 1, 6) == (
        tool.TENSOR_HEADER_BYTES + 2 + 3
    )
    assert tool.encoded_bytes_for_tensor(5, 1, 6) == (
        tool.TENSOR_HEADER_BYTES + 2 + 6
    )


def test_mixed_precision_dominated_cpu_proxy_cannot_route_to_cuda() -> None:
    tool = _load_script(
        "tools/pr101_lossy_mixed_precision_int4_int8.py",
        "pr101_lossy_mixed_precision_int4_int8_dominated",
    )

    verdict, cuda_eval_worth_testing, blockers = tool.classify_cpu_proxy_candidate(
        weighted_avg_rel_err_pct=4.0,
        archive_bytes=tool.PR101_BROTLI_BASELINE_BYTES + 1,
        target_rel_err_pct=5.0,
    )

    assert verdict == "MEASURED_CONFIG_DOMINATED_BY_PR101_BROTLI_BASELINE"
    assert cuda_eval_worth_testing is False
    assert blockers[0] == "archive_bytes_not_below_pr101_brotli_baseline"
    assert "missing_exact_cuda_auth_eval" in blockers


def test_int4_roundtrip_cpu_proxy_never_sets_exact_dispatch_ready(tmp_path: Path) -> None:
    import torch

    tool = _load_script(
        "tools/pr101_lossy_int4_roundtrip_test.py",
        "pr101_lossy_int4_roundtrip_test_guard",
    )
    state_dict = {
        name: torch.ones(tuple(shape), dtype=torch.float32)
        for name, shape in tool.FIXED_STATE_SCHEMA
    }
    state_path = tmp_path / "state.pt"
    torch.save(state_dict, state_path)

    manifest = tool.measure_full_roundtrip(state_path, block_size=1024)

    assert manifest["cuda_eval_worth_testing"] is True
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["family_falsified"] is False
    assert "missing_exact_cuda_auth_eval" in manifest["dispatch_blockers"]


def test_lossy_coarsening_cpu_build_manifest_guard_fields() -> None:
    tool = _load_script(
        "experiments/lossy_coarsening_lightning_cuda_test.py",
        "lossy_coarsening_lightning_cuda_test_guard",
    )

    guard = tool.cpu_build_proxy_guard_fields()

    assert guard["score_claim"] is False
    assert guard["promotion_eligible"] is False
    assert guard["rank_or_kill_eligible"] is False
    assert guard["ready_for_exact_eval_dispatch"] is False
    assert guard["family_falsified"] is False
    assert guard["custody_status"] == "transient-allowed"
    assert "auth eval" in guard["custody_status_reason"]
    assert "exact_cuda_auth_eval_not_yet_harvested" in guard["dispatch_blockers"]
