# SPDX-License-Identifier: MIT
"""Regression coverage for Venn reweighting + predicted dispatch risk composition."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _load_autopilot_module():
    spec = importlib.util.spec_from_file_location(
        "autopilot_loop_venn_risk",
        REPO_ROOT / "tools" / "cathedral_autopilot_autonomous_loop.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("autopilot_loop_venn_risk", mod)
    spec.loader.exec_module(mod)
    return mod


def test_venn_reweight_does_not_replace_predicted_dispatch_risk_refusal(
    tmp_path, monkeypatch
) -> None:
    """HIGH PAIR_INVARIANT Venn reward must not resurrect a risk-refused row."""
    mod = _load_autopilot_module()
    fake_root = tmp_path / "master_gradient_consumers"
    fake_root.mkdir(parents=True)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", fake_root)
    sha = "66" * 32
    sidecar = fake_root / f"venn_classification_{sha[:12]}_20260517T210000.json"
    sidecar.write_text(
        json.dumps(
            {
                "class_counts": {
                    "PAIR_SPECIFIC": 100,
                    "PAIR_INVARIANT": 9000,
                    "PAIR_NEUTRAL": 900,
                    "DEAD": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    candidate = mod.CandidateRow(
        candidate_id="risk_refused_venn_rewarded",
        family="fec6_like",
        predicted_score_delta=-0.012,
        expected_information_gain=1.0,
        estimated_dispatch_cost_usd=1.0,
        archive_sha256=sha,
        predicted_dispatch_risk=75.0,
    )
    rank_key = mod.apply_z1_empirical_revision_to_candidate_delta(candidate)
    assert rank_key == mod.PREDICTED_DISPATCH_RISK_REFUSAL_DELTA_FLOOR
    assert rank_key == 0.0
