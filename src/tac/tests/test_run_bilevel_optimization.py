"""Tests for ``tools.run_bilevel_optimization`` — magic-autopilot driver."""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest


def _load_driver_module():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    path = repo_root / "tools" / "run_bilevel_optimization.py"
    spec = importlib.util.spec_from_file_location("run_bilevel_optimization", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_phase_trajectory_has_seven_phases() -> None:
    mod = _load_driver_module()
    assert set(mod.PHASE_TRAJECTORY.keys()) == {1, 2, 3, 4, 5, 6, 7}


def test_phase_trajectory_target_scores_strictly_decreasing() -> None:
    """Each phase should target a score strictly below the previous (we're descending the floor)."""
    mod = _load_driver_module()
    targets = [float(mod.PHASE_TRAJECTORY[ph]["target_score"]) for ph in range(1, 8)]
    for i in range(len(targets) - 1):
        assert targets[i + 1] < targets[i], (
            f"target_score not strictly decreasing: phase {i+1} {targets[i]} -> phase {i+2} {targets[i+1]}"
        )


def test_phase_trajectory_costs_strictly_increasing() -> None:
    """Each phase should cost more GPU $ than the previous (deeper search = more compute)."""
    mod = _load_driver_module()
    costs = [int(mod.PHASE_TRAJECTORY[ph]["estimated_gpu_usd"]) for ph in range(1, 8)]
    for i in range(len(costs) - 1):
        assert costs[i + 1] > costs[i]


def test_detect_substrates_returns_list_of_candidates(tmp_path: pathlib.Path) -> None:
    mod = _load_driver_module()
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    # Empty repo → no substrates
    assert mod.detect_substrates(repo) == []
    # Add a fake PR106 state_dict
    pr106_dir = repo / "experiments/results/sensitivity_map_pr106_20260504_claude"
    pr106_dir.mkdir(parents=True)
    (pr106_dir / "state_dict.pt").write_text("dummy")
    subs = mod.detect_substrates(repo)
    assert len(subs) == 1
    assert "PR106" in subs[0].label


def test_detect_substrates_real_repo_finds_pr100() -> None:
    """Real repo should have at least PR100 substrate present."""
    mod = _load_driver_module()
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    subs = mod.detect_substrates(repo_root)
    pr100_present = any("PR100" in s.label for s in subs)
    pr101_present = any("PR101" in s.label for s in subs)
    pr103_present = any("PR103" in s.label for s in subs)
    # At least one of the canonical-winner substrates must be on disk.
    assert pr100_present or pr101_present or pr103_present


def test_detect_current_phase_handles_missing_registry(tmp_path: pathlib.Path) -> None:
    mod = _load_driver_module()
    fake_repo = tmp_path / "fresh"
    fake_repo.mkdir()
    # No .omx/state/lane_registry.json → fresh checkout, return phase 1
    assert mod.detect_current_phase(fake_repo) == 1


def test_detect_current_phase_handles_corrupt_registry(tmp_path: pathlib.Path) -> None:
    mod = _load_driver_module()
    fake_repo = tmp_path / "corrupt"
    state_dir = fake_repo / ".omx/state"
    state_dir.mkdir(parents=True)
    (state_dir / "lane_registry.json").write_text("{not valid json")
    # Corrupt registry → fall back to phase 1
    assert mod.detect_current_phase(fake_repo) == 1


def test_detect_current_phase_dict_schema(tmp_path: pathlib.Path) -> None:
    mod = _load_driver_module()
    fake_repo = tmp_path / "dict_schema"
    state_dir = fake_repo / ".omx/state"
    state_dir.mkdir(parents=True)
    (state_dir / "lane_registry.json").write_text(json.dumps({
        "lanes": {
            "lane_pr100_anchor": {"level": 3},
            "lane_other": {"level": 1},
        }
    }))
    # PR100 lane at L3 → advance past phase 1
    assert mod.detect_current_phase(fake_repo) >= 2


def test_detect_current_phase_list_schema(tmp_path: pathlib.Path) -> None:
    """Registry uses list-of-lanes form per actual repo schema."""
    mod = _load_driver_module()
    fake_repo = tmp_path / "list_schema"
    state_dir = fake_repo / ".omx/state"
    state_dir.mkdir(parents=True)
    (state_dir / "lane_registry.json").write_text(json.dumps({
        "lanes": [
            {"id": "lane_pr100_canonical", "level": 3},
            {"id": "lane_other", "level": 1},
        ]
    }))
    assert mod.detect_current_phase(fake_repo) >= 2


def test_atom_ledger_appends_jsonl(tmp_path: pathlib.Path) -> None:
    mod = _load_driver_module()
    repo = tmp_path / "ledger_repo"
    repo.mkdir()
    candidate = mod.SubstrateCandidate(
        label="test", path="/tmp/fake", bytes=100, score_anchor=0.2, notes="test",
    )
    path = mod.append_to_atom_ledger(
        repo,
        phase=1,
        substrate=candidate,
        contest_cuda_score=0.18,
        archive_bytes=180_000,
        archive_sha256="deadbeef",
        cathedral_op="Op1+Op2",
        notes="test entry",
    )
    assert path.exists()
    line = path.read_text().strip().splitlines()[-1]
    record = json.loads(line)
    assert record["phase"] == 1
    assert record["contest_cuda_score"] == 0.18
    assert record["evidence_grade"] == "[contest-CUDA]"


def test_atom_ledger_marks_cpu_prep_when_no_score(tmp_path: pathlib.Path) -> None:
    mod = _load_driver_module()
    repo = tmp_path / "ledger_cpu"
    repo.mkdir()
    candidate = mod.SubstrateCandidate(
        label="cpu", path="/tmp/cpu", bytes=50, score_anchor=None, notes="",
    )
    path = mod.append_to_atom_ledger(
        repo, phase=2, substrate=candidate,
        contest_cuda_score=None, archive_bytes=None, archive_sha256=None,
        cathedral_op="prep", notes="prep-only",
    )
    line = path.read_text().strip().splitlines()[-1]
    record = json.loads(line)
    assert record["evidence_grade"] == "[CPU-prep]"


def test_run_phase_1_blocks_when_no_substrate(tmp_path: pathlib.Path) -> None:
    mod = _load_driver_module()
    fake = tmp_path / "no_substrate"
    fake.mkdir()
    result = mod.run_phase_1(fake)
    assert result["status"] == "BLOCKED"


def test_phase_trajectory_target_score_is_below_medal_band_for_phases_2_plus() -> None:
    """Per Grand Council, phase 2+ target should beat the public medal band (0.193)."""
    mod = _load_driver_module()
    for ph in range(2, 8):
        target = float(mod.PHASE_TRAJECTORY[ph]["target_score"])
        assert target < 0.193, f"phase {ph} target {target} not below medal band 0.193"
