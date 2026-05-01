from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_launcher():
    path = REPO_ROOT / "scripts" / "launch_lane_with_retry.py"
    spec = importlib.util.spec_from_file_location("_launch_lane_with_retry_forensic", path)
    assert spec and spec.loader
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    return launcher


def test_current_lane19_script_satisfies_forensic_clearance_markers() -> None:
    launcher = _load_launcher()

    violations = launcher.lane_forensic_clearance_violations(launcher.LANE_19_HOLD_KEY)

    assert violations == []


def test_current_lane20_script_remains_on_hold_until_real_bhv1_integration() -> None:
    launcher = _load_launcher()

    violations = launcher.lane_forensic_clearance_violations(launcher.LANE_20_HOLD_KEY)

    assert any("real BHv1 archive build" in v for v in violations)
    assert any("inflate-side BHv1 decode integration" in v for v in violations)


def test_cleared_lane19_hold_still_blocks_without_required_repairs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    launcher = _load_launcher()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "scripts" / "remote_lane_19_logit_margin.sh").write_text(
        "#!/bin/bash\n# stale Lane 19 script without deterministic archive or adjudication\n"
    )
    (tmp_path / "src" / "tac" / "profiles.py").write_text(
        "PROFILES = {'lane_19_logit_margin': {}}\n"
    )
    holds = tmp_path / "dispatch_holds.json"
    holds.write_text(json.dumps({
        "holds": [
            {
                "logical_key": launcher.LANE_19_HOLD_KEY,
                "reason": "operator marked this cleared too early",
                "cleared": True,
            }
        ]
    }))
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "DISPATCH_HOLDS_PATH", holds)

    hold = launcher.dispatch_hold_for_label(
        "lane_19_logit_margin_2026-04-30_q1d_20260430T212704Z"
    )

    assert hold is not None
    assert "marked cleared" in hold["reason"]
    assert any("deterministic archive build" in v for v in hold["clearance_violations"])
    assert any("JSON adjudication" in v for v in hold["clearance_violations"])
    assert any("current frontier score gate" in v for v in hold["clearance_violations"])


def test_missing_lane20_hold_entry_still_blocks_without_bhv1_integration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    launcher = _load_launcher()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "submissions" / "robust_current").mkdir(parents=True)
    (tmp_path / "scripts" / "remote_lane_20_balle.sh").write_text(
        "#!/bin/bash\n"
        "echo FATAL_NON_STATIC_BYTE_PRECHECK_FAILED\n"
        "echo non_static_byte_precheck.json BALLE_BEATS_STATIC\n"
        "echo best_full_balle_bytes static_baseline_bytes\n"
    )
    (tmp_path / "submissions" / "robust_current" / "inflate_renderer.py").write_text(
        "# no BHv1 renderer.bhv1 decode path\n"
    )
    holds = tmp_path / "dispatch_holds.json"
    holds.write_text(json.dumps({"holds": []}))
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "DISPATCH_HOLDS_PATH", holds)

    hold = launcher.dispatch_hold_for_label("lane_20_balle_2026-04-30")

    assert hold is not None
    assert "entry is missing" in hold["reason"]
    assert any("real BHv1 archive build" in v for v in hold["clearance_violations"])
    assert any("inflate-side BHv1 decode integration" in v for v in hold["clearance_violations"])
