from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load_tool_module():
    path = REPO / "tools" / "cathedral_autopilot.py"
    spec = importlib.util.spec_from_file_location("cathedral_autopilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_autopilot_rejects_spoofed_mps_promotability_booleans() -> None:
    tool = _load_tool_module()
    evidence = tool.TechniqueEvidence(
        technique="arch_shrink_mps",
        empirical_archive_bytes=100_000,
        evidence_grade="[MPS-research-signal]",
        evidence_marker="[MPS-research-signal]",
        evidence_semantics="mps_proxy_curve_shape_only",
        score_claim=True,
        promotion_eligible=True,
        rank_or_kill_eligible=True,
        ready_for_exact_eval_dispatch=True,
        dispatch_blockers=[],
        source="[MPS-research-signal] local proxy",
    )

    assert tool._is_explicitly_promotable_evidence(evidence) is False


def test_autopilot_requires_exact_cuda_for_promotable_evidence() -> None:
    tool = _load_tool_module()
    exact = tool.TechniqueEvidence(
        technique="exact_anchor",
        empirical_archive_bytes=100_000,
        evidence_grade="[contest-CUDA]",
        evidence_semantics="contest_cuda_exact_eval_positive",
        score_claim=True,
        promotion_eligible=True,
        rank_or_kill_eligible=True,
        ready_for_exact_eval_dispatch=True,
        dispatch_blockers=[],
        source="contest_auth_eval.json",
    )
    cpu = tool.TechniqueEvidence(
        technique="cpu_anchor",
        empirical_archive_bytes=100_000,
        evidence_grade="[CPU-prep]",
        evidence_semantics="cpu_substrate_predicted_band",
        score_claim=True,
        promotion_eligible=True,
        rank_or_kill_eligible=True,
        ready_for_exact_eval_dispatch=True,
        dispatch_blockers=[],
        source="local cpu prep",
    )

    assert tool._is_explicitly_promotable_evidence(exact) is True
    assert tool._is_explicitly_promotable_evidence(cpu) is False
