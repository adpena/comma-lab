# SPDX-License-Identifier: MIT
# FAKE_LANE_OK_FILE: documentation tests for Check #125/#126 use synthetic
# lane_id fixtures (lane_registered, lane_missing) inside synthetic registries.
# Per Check #126 file-level waiver semantics.
"""Tests for subagent-coherence preflight checks (#125/#126)."""
from __future__ import annotations

import json
from pathlib import Path

from tac import preflight as pf


def _write_registry(repo: Path, lane_ids: list[str]) -> None:
    state = repo / ".omx" / "state"
    state.mkdir(parents=True, exist_ok=True)
    lanes = [
        {
            "id": lane_id,
            "name": lane_id,
            "phase": 1,
            "level": 0,
            "gates": {},
            "notes": "",
        }
        for lane_id in lane_ids
    ]
    (state / "lane_registry.json").write_text(
        json.dumps({"schema_version": 1, "lanes": lanes}),
        encoding="utf-8",
    )


def test_landing_wire_in_accepts_all_six_hooks(tmp_path: Path) -> None:
    memo = tmp_path / "feedback_good_landed_20260509.md"
    memo.write_text(
        "\n".join(
            [
                "Sensitivity-map contribution: none",
                "Pareto constraint: rate only",
                "Bit-allocator hook: N/A - documentation-only landing",
                "Cathedral autopilot dispatch: not dispatchable",
                "Continual-learning posterior update: record negative",
                "Probe-disambiguator: deterministic one interpretation",
            ]
        ),
        encoding="utf-8",
    )

    assert pf.check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path,
        strict=False,
        verbose=False,
    ) == []


def test_landing_wire_in_flags_missing_hooks(tmp_path: Path) -> None:
    memo = tmp_path / "feedback_bad_landed_20260509.md"
    memo.write_text("Sensitivity-map contribution: present\n", encoding="utf-8")

    violations = pf.check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "missing wire-in declaration" in violations[0]
    assert "pareto" in violations[0]


def test_landing_wire_in_accepts_research_only_optout(tmp_path: Path) -> None:
    memo = tmp_path / "feedback_research_landed_20260509.md"
    memo.write_text(
        "research_only=true\nrationale: author-source intake only\n",
        encoding="utf-8",
    )

    assert pf.check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path,
        strict=False,
        verbose=False,
    ) == []


def test_lane_pre_registered_accepts_registered_lane(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_registry(tmp_path, ["lane_registered"])
    source = tmp_path / "src" / "tac" / "candidate.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('_LANE_ID = "lane_registered"\n', encoding="utf-8")
    monkeypatch.setattr(
        pf,
        "_commit_introduced_files",
        lambda repo_root, n_commits, since_sha=None: {
            "a" * 40: ["src/tac/candidate.py"]
        },
    )

    assert pf.check_lane_pre_registered_before_work_starts(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    ) == []


def test_lane_pre_registered_flags_unregistered_lane(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_registry(tmp_path, ["lane_registered"])
    source = tmp_path / "tools" / "candidate.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('_LANE_ID = "lane_missing"\n', encoding="utf-8")
    monkeypatch.setattr(
        pf,
        "_commit_introduced_files",
        lambda repo_root, n_commits, since_sha=None: {
            "b" * 40: ["tools/candidate.py"]
        },
    )

    violations = pf.check_lane_pre_registered_before_work_starts(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "lane_missing" in violations[0]
    assert "Pre-register" in violations[0]


def test_lane_pre_registered_allows_fake_test_lane_with_marker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_registry(tmp_path, [])
    source = tmp_path / "src" / "tac" / "tests" / "test_candidate.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "# FAKE_LANE_OK: test fixture only\n_LANE_ID = \"lane_missing\"\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pf,
        "_commit_introduced_files",
        lambda repo_root, n_commits, since_sha=None: {
            "c" * 40: ["src/tac/tests/test_candidate.py"]
        },
    )

    assert pf.check_lane_pre_registered_before_work_starts(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    ) == []
