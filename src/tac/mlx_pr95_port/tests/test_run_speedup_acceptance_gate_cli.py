# SPDX-License-Identifier: MIT
"""Tests for the thin speedup-acceptance-gate CLI (tools/run_speedup_acceptance_gate.py).

The CLI is the operator-facing surface of the BOTH-TERMS gate. The load-bearing
test here is ``test_cli_rejects_pose_divergent_ab_json`` — it proves the CLI
EXITS NONZERO on the exact n600 failure (perfect d_seg, diverging d_pose) fed
through the real ``measure_descent_equivalence`` A/B JSON shape, so an operator
or CI harness gating a dispatch on this exit code cannot admit a pose-divergent
speedup.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
CLI = REPO / "tools" / "run_speedup_acceptance_gate.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_speedup_acceptance_gate", CLI)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _ab_json(tmp_path: Path, torch_arm, mlx_arm, n) -> Path:
    blob = {
        "config": {"max_pairs": n},
        "arm_torch_cpu": torch_arm,
        "arm_mlx_gpu": mlx_arm,
    }
    p = tmp_path / "ab.json"
    p.write_text(json.dumps(blob))
    return p


def _clean_arm():
    return [
        {"epoch": 0, "exact_d_seg": 0.030, "mean_d_pose": 0.10},
        {"epoch": 10, "exact_d_seg": 0.017, "mean_d_pose": 0.050},
        {"epoch": 20, "exact_d_seg": 0.0041, "mean_d_pose": 3.1e-5},
    ]


def test_cli_rejects_pose_divergent_ab_json(tmp_path):
    """THE load-bearing CLI test: pose-divergent candidate => exit code 1."""
    mod = _load_cli()
    torch_arm = _clean_arm()
    mlx_arm = [
        {"epoch": 0, "exact_d_seg": 0.030, "mean_d_pose": 0.10},
        {"epoch": 10, "exact_d_seg": 0.028, "mean_d_pose": 0.835},
        {"epoch": 20, "exact_d_seg": 0.0042, "mean_d_pose": 36.46},  # seg fine, pose explodes
    ]
    ab = _ab_json(tmp_path, torch_arm, mlx_arm, n=600)
    out = tmp_path / "verdict.json"
    rc = mod.main(["--from-ab", str(ab), "--out-json", str(out)])
    assert rc == 1
    v = json.loads(out.read_text())
    assert v["passed"] is False
    assert v["seg"]["tracks"] is True  # the trap: seg passed
    assert v["pose"]["diverged"] is True


def test_cli_passes_clean_descent_equivalent_ab_at_n600(tmp_path):
    mod = _load_cli()
    torch_arm = _clean_arm()
    mlx_arm = [
        {"epoch": 0, "exact_d_seg": 0.030, "mean_d_pose": 0.10},
        {"epoch": 10, "exact_d_seg": 0.0171, "mean_d_pose": 0.050001},
        {"epoch": 20, "exact_d_seg": 0.0041, "mean_d_pose": 3.2e-5},
    ]
    ab = _ab_json(tmp_path, torch_arm, mlx_arm, n=600)
    rc = mod.main(["--from-ab", str(ab)])
    assert rc == 0


def test_cli_flags_n8_pass_provisional(tmp_path):
    """A PASS at n8 still exits 0 but the verdict JSON marks it provisional."""
    mod = _load_cli()
    ab = _ab_json(tmp_path, _clean_arm(), _clean_arm(), n=8)
    out = tmp_path / "v.json"
    rc = mod.main(["--from-ab", str(ab), "--out-json", str(out)])
    assert rc == 0
    v = json.loads(out.read_text())
    assert v["generalization_warning"] is True


def test_cli_separate_baseline_candidate_files(tmp_path):
    mod = _load_cli()
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    base.write_text(json.dumps(_clean_arm()))
    cand.write_text(json.dumps(_clean_arm()))
    rc = mod.main(["--baseline", str(base), "--candidate", str(cand), "--n-pairs", "600"])
    assert rc == 0


def test_cli_refuses_d_seg_only_trajectory(tmp_path):
    """A trajectory with no pose key is structurally refused (raises, not silent)."""
    import pytest

    mod = _load_cli()
    torch_arm = [{"epoch": 0, "exact_d_seg": 0.03}, {"epoch": 20, "exact_d_seg": 0.004}]  # NO pose
    ab = _ab_json(tmp_path, torch_arm, _clean_arm(), n=600)
    from tac.mlx_pr95_port.speedup_acceptance_gate import DSegOnlyGateMisuse

    with pytest.raises(DSegOnlyGateMisuse):
        mod.main(["--from-ab", str(ab)])
