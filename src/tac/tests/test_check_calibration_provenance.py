# SPDX-License-Identifier: MIT
"""Tests for tools/check_calibration_provenance.py (PCC10).

Per council Q4: every numeric literal bound to a prediction-logic identifier
must carry a provenance tag. This catches the "magic number" anti-pattern
the apogee_int4 8x miss surfaced.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PCC10 = REPO_ROOT / "tools" / "check_calibration_provenance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_pcc10_test", PCC10)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_repo(tmp_path: Path, glob_subdir: str, name: str, content: str) -> Path:
    """Create a synthetic file under one of PCC10's scanned globs."""
    target_dir = tmp_path / glob_subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / name).write_text(content)
    return tmp_path


def test_pcc10_catches_untagged_literal(tmp_path: Path) -> None:
    """A bare numeric literal bound to a prediction-named target is flagged."""
    mod = _load_module()
    repo = _make_repo(tmp_path, "src/tac/predictor", "score_band.py", """
PR106_SCORE = 0.20946
SOMETHING_ELSE = 42
""")
    findings = mod.scan(repo)
    flagged_targets = {f.target_name for f in findings}
    assert "PR106_SCORE" in flagged_targets


def test_pcc10_accepts_contest_defined_tag(tmp_path: Path) -> None:
    """Provenance tag on the same line passes."""
    mod = _load_module()
    repo = _make_repo(tmp_path, "src/tac/predictor", "score_band.py", """
RATE_COEF = 25.0  # [contest-defined] from upstream/evaluate.py
SCORE_THRESHOLD = 0.30  # [contest-defined]
""")
    findings = mod.scan(repo)
    assert findings == [], f"unexpected findings: {[f.target_name for f in findings]}"


def test_pcc10_accepts_calibration_tag(tmp_path: Path) -> None:
    """Calibration tag on the line above passes."""
    mod = _load_module()
    repo = _make_repo(tmp_path, "tools", "apogee_intN_pareto.py", """
# [calibration:.omx/calibration/anchors_apogee_intN.json]
PR106_BASELINE_SCORE = 0.20946
""")
    findings = mod.scan(repo)
    assert findings == [], f"unexpected findings: {[f.target_name for f in findings]}"


def test_pcc10_accepts_empirical_tag(tmp_path: Path) -> None:
    """Empirical tag passes."""
    mod = _load_module()
    repo = _make_repo(tmp_path, "experiments", "build_apogee_int4.py", """
APOGEE_INT4_DELTA = 0.0237  # [empirical:experiments/results/apogee_int4_repack_20260504_claude/]
""")
    findings = mod.scan(repo)
    assert findings == [], f"unexpected findings: {[f.target_name for f in findings]}"


def test_pcc10_accepts_heuristic_tag(tmp_path: Path) -> None:
    """Heuristic tag passes."""
    mod = _load_module()
    repo = _make_repo(tmp_path, "tools", "predispatch_sanity.py", """
HIGH_REL_ERR_THRESHOLD_PCT = 1.0  # [heuristic:Q1-Hotz "above 1%, run local proxy"]
""")
    findings = mod.scan(repo)
    assert findings == [], f"unexpected findings: {[f.target_name for f in findings]}"


def test_pcc10_whitelists_trivial_literals(tmp_path: Path) -> None:
    """0/1/-1/2 don't need provenance tags (loop indices, defaults)."""
    mod = _load_module()
    repo = _make_repo(tmp_path, "src/tac/predictor", "score_band.py", """
DEFAULT_SCORE_INDEX = 0
SCORE_OFFSET = 1
""")
    findings = mod.scan(repo)
    assert findings == [], f"unexpected findings: {[f.target_name for f in findings]}"


def test_pcc10_only_flags_prediction_named_targets(tmp_path: Path) -> None:
    """Non-prediction-named literals are out of scope."""
    mod = _load_module()
    repo = _make_repo(tmp_path, "src/tac/predictor", "score_band.py", """
RANDOM_HYPERPARAM = 0.42
""")
    findings = mod.scan(repo)
    assert findings == [], f"unexpected findings: {[f.target_name for f in findings]}"


def test_pcc10_strict_exits_nonzero(tmp_path: Path) -> None:
    """--strict exits 1 when violations exist."""
    mod = _load_module()
    repo = _make_repo(tmp_path, "src/tac/predictor", "score_band.py", """
SCORE_THRESHOLD = 0.42
""")
    rc = mod.main(["--repo-root", str(repo), "--strict"])
    assert rc == 1
