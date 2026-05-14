# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load_tool_module():
    path = REPO / "tools" / "pr101_lossy_int4_qat.py"
    spec = importlib.util.spec_from_file_location("pr101_lossy_int4_qat", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_local_qat_candidate_never_sets_exact_dispatch_ready() -> None:
    tool = _load_tool_module()

    contract = tool.local_qat_dispatch_contract(cuda_eval_worth_testing=True)

    assert contract["cuda_eval_worth_testing"] is True
    assert contract["ready_for_exact_eval_dispatch"] is False
    assert contract["promotion_eligible"] is False
    assert contract["rank_or_kill_eligible"] is False
    assert contract["dispatch_attempted"] is False
    assert "missing_exact_cuda_auth_eval" in contract["dispatch_blockers"]
    assert "byte_closed_int4_candidate_packet_missing" in contract["dispatch_blockers"]
