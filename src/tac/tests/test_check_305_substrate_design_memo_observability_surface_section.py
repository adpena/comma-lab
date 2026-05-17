# SPDX-License-Identifier: MIT
"""Tests for Catalog #305 ``check_substrate_design_memo_has_observability_surface_section``.

Per ``feedback_max_observability_into_behavior_xray_autopilot_tools_experiments_designs_standing_directive_20260516.md``
(operator standing directive 2026-05-16).

The gate refuses substrate design / landing / composition memos dated
>= 2026-05-16 that lack the literal section header ``## Observability surface``
(case-insensitive). Pre-cutoff memos are exempt. Same-line waiver
``# OBSERVABILITY_SURFACE_SECTION_WAIVED:<rationale>`` accepted (placeholder
``<rationale>`` / ``<reason>`` literals rejected).

Sister of:
- Catalog #290 (canonical-vs-unique decision per layer — Dimension 5)
- Catalog #294 (9-dim success checklist evidence section)
- Catalog #303 (cargo-cult audit section)
- Catalog #296 (predicted-band Dykstra feasibility)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_design_memo_has_observability_surface_section,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(memory_dir: Path, name: str, body: str) -> Path:
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / name
    path.write_text(body, encoding="utf-8")
    return path


def _frontmatter(name: str = "test-substrate-design") -> str:
    return (
        "---\n"
        f"name: {name}\n"
        "description: test fixture\n"
        "metadata:\n"
        "  node_type: memory\n"
        "  type: feedback\n"
        "---\n\n"
    )


# ---------------------------------------------------------------------------
# Live-repo regression guard
# ---------------------------------------------------------------------------


def test_305_live_repo_regression_guard() -> None:
    """The live repo should have a bounded number of violations.

    At landing, the 10 existing 2026-05-16 design memos predate the directive
    and are flagged. The OBSERVABILITY-ADDENDUM backfill drives the count
    down. This regression guard caps at 20 to allow some sister-subagent
    drift during the strict-flip transition.
    """
    violations = check_substrate_design_memo_has_observability_surface_section(
        strict=False, verbose=False
    )
    assert len(violations) <= 20, (
        f"Catalog #305 live count is {len(violations)}; should be <= 20 "
        "until strict-flip after OBSERVABILITY-ADDENDUM backfill completes."
    )


# ---------------------------------------------------------------------------
# Positive (flagged) cases
# ---------------------------------------------------------------------------


def test_305_design_memo_without_section_flagged(tmp_path: Path) -> None:
    research = tmp_path / "research"
    _write(
        research,
        "my_substrate_design_20260516.md",
        _frontmatter() + "# My Substrate Design\n\nSome content here.\n",
    )
    violations = check_substrate_design_memo_has_observability_surface_section(
        research_dir=research, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "my_substrate_design_20260516.md" in violations[0]
    assert "Observability surface" in violations[0]


def test_305_landing_memo_substrate_without_section_flagged(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    _write(
        memory,
        "feedback_xyz_substrate_landed_20260516.md",
        _frontmatter() + "# Body without the section\n",
    )
    violations = check_substrate_design_memo_has_observability_surface_section(
        memory_dir=memory, research_dir=tmp_path / "no_research", strict=False, verbose=False
    )
    assert len(violations) == 1


def test_305_nscs_memo_without_section_flagged(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    _write(
        memory,
        "feedback_nscs06_v8_landed_20260516.md",
        _frontmatter() + "# Body without the section\n",
    )
    violations = check_substrate_design_memo_has_observability_surface_section(
        memory_dir=memory, research_dir=tmp_path / "no_research", strict=False, verbose=False
    )
    assert len(violations) == 1


def test_305_stack_of_stacks_memo_without_section_flagged(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    _write(
        memory,
        "feedback_a_stack_of_stacks_landed_20260516.md",
        _frontmatter() + "# Body without the section\n",
    )
    violations = check_substrate_design_memo_has_observability_surface_section(
        memory_dir=memory, research_dir=tmp_path / "no_research", strict=False, verbose=False
    )
    assert len(violations) == 1


def test_305_composition_memo_without_section_flagged(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    _write(
        memory,
        "feedback_a_stack_composition_landed_20260516.md",
        _frontmatter() + "# Body without the section\n",
    )
    violations = check_substrate_design_memo_has_observability_surface_section(
        memory_dir=memory, research_dir=tmp_path / "no_research", strict=False, verbose=False
    )
    assert len(violations) == 1


def test_305_multiple_violations_aggregated(tmp_path: Path) -> None:
    research = tmp_path / "research"
    _write(research, "alpha_design_20260516.md", _frontmatter() + "no section\n")
    _write(research, "beta_design_20260516.md", _frontmatter() + "no section\n")
    _write(research, "gamma_design_20260516.md", _frontmatter() + "no section\n")
    violations = check_substrate_design_memo_has_observability_surface_section(
        research_dir=research, strict=False, verbose=False
    )
    assert len(violations) == 3


# ---------------------------------------------------------------------------
# Negative (accepted) cases
# ---------------------------------------------------------------------------


def test_305_design_memo_with_section_accepted(tmp_path: Path) -> None:
    research = tmp_path / "research"
    _write(
        research,
        "my_substrate_design_20260516.md",
        _frontmatter() + (
            "# My Substrate Design\n\n"
            "## Observability surface\n\n"
            "Per-layer inspection: torch hooks; "
            "per-signal decomposition: SegNet/PoseNet/rate; "
            "run-to-run diff: archive sha256 + scorer state; "
            "post-hoc query: experiments/results/<lane>/observability.json; "
            "cite-chain: commit + call_id + upstream_snapshot_sha256; "
            "counterfactual hooks: byte-mutation via Catalog #139.\n"
        ),
    )
    violations = check_substrate_design_memo_has_observability_surface_section(
        research_dir=research, strict=False, verbose=False
    )
    assert len(violations) == 0


def test_305_section_header_case_insensitive(tmp_path: Path) -> None:
    research = tmp_path / "research"
    _write(
        research,
        "my_substrate_design_20260516.md",
        _frontmatter() + "## OBSERVABILITY SURFACE\n\nBody.\n",
    )
    violations = check_substrate_design_memo_has_observability_surface_section(
        research_dir=research, strict=False, verbose=False
    )
    assert len(violations) == 0


def test_305_section_with_mixed_case_accepted(tmp_path: Path) -> None:
    research = tmp_path / "research"
    _write(
        research,
        "my_substrate_design_20260516.md",
        _frontmatter() + "## Observability Surface\n",
    )
    violations = check_substrate_design_memo_has_observability_surface_section(
        research_dir=research, strict=False, verbose=False
    )
    assert len(violations) == 0


# ---------------------------------------------------------------------------
# Cutoff (date) filter
# ---------------------------------------------------------------------------


def test_305_pre_cutoff_memo_exempt(tmp_path: Path) -> None:
    research = tmp_path / "research"
    _write(
        research,
        "old_substrate_design_20260515.md",
        _frontmatter() + "# Body without the section\n",
    )
    violations = check_substrate_design_memo_has_observability_surface_section(
        research_dir=research, strict=False, verbose=False
    )
    assert len(violations) == 0


def test_305_post_cutoff_2026_06_memo_in_scope(tmp_path: Path) -> None:
    research = tmp_path / "research"
    _write(
        research,
        "future_substrate_design_20260601.md",
        _frontmatter() + "# Body without the section\n",
    )
    violations = check_substrate_design_memo_has_observability_surface_section(
        research_dir=research, strict=False, verbose=False
    )
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# Waiver semantics
# ---------------------------------------------------------------------------


def test_305_same_line_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    research = tmp_path / "research"
    _write(
        research,
        "my_substrate_design_20260516.md",
        _frontmatter() + (
            "# Body\n\n"
            "# OBSERVABILITY_SURFACE_SECTION_WAIVED: substrate is a pure "
            "wire-format spec with no runtime behavior; observability surface "
            "lives in the consumer trainer's memo\n"
        ),
    )
    violations = check_substrate_design_memo_has_observability_surface_section(
        research_dir=research, strict=False, verbose=False
    )
    assert len(violations) == 0


def test_305_waiver_placeholder_rejected(tmp_path: Path) -> None:
    research = tmp_path / "research"
    _write(
        research,
        "my_substrate_design_20260516.md",
        _frontmatter() + (
            "# Body\n\n"
            "# OBSERVABILITY_SURFACE_SECTION_WAIVED:<rationale>\n"
        ),
    )
    violations = check_substrate_design_memo_has_observability_surface_section(
        research_dir=research, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_305_waiver_reason_placeholder_rejected(tmp_path: Path) -> None:
    research = tmp_path / "research"
    _write(
        research,
        "my_substrate_design_20260516.md",
        _frontmatter() + (
            "# Body\n\n"
            "# OBSERVABILITY_SURFACE_SECTION_WAIVED:<reason>\n"
        ),
    )
    violations = check_substrate_design_memo_has_observability_surface_section(
        research_dir=research, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_305_waiver_empty_rationale_rejected(tmp_path: Path) -> None:
    research = tmp_path / "research"
    _write(
        research,
        "my_substrate_design_20260516.md",
        _frontmatter() + (
            "# Body\n\n"
            "# OBSERVABILITY_SURFACE_SECTION_WAIVED:\n"
        ),
    )
    violations = check_substrate_design_memo_has_observability_surface_section(
        research_dir=research, strict=False, verbose=False
    )
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


def test_305_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    research = tmp_path / "research"
    _write(
        research,
        "my_substrate_design_20260516.md",
        _frontmatter() + "# Body without the section\n",
    )
    with pytest.raises(PreflightError) as excinfo:
        check_substrate_design_memo_has_observability_surface_section(
            research_dir=research, strict=True, verbose=False
        )
    msg = str(excinfo.value)
    assert "Catalog #305" in msg
    assert "Observability surface" in msg
    assert "max observability" in msg.lower() or "MAX-OBSERVABILITY" in msg


def test_305_strict_mode_silent_on_clean(tmp_path: Path) -> None:
    research = tmp_path / "research"
    _write(
        research,
        "my_substrate_design_20260516.md",
        _frontmatter() + "## Observability surface\n\nBody.\n",
    )
    # Should not raise.
    result = check_substrate_design_memo_has_observability_surface_section(
        research_dir=research, strict=True, verbose=False
    )
    assert result == []


# ---------------------------------------------------------------------------
# Scope / exemption edge cases
# ---------------------------------------------------------------------------


def test_305_no_research_dir_returns_empty(tmp_path: Path) -> None:
    # Pass a non-existent directory; should silently skip.
    violations = check_substrate_design_memo_has_observability_surface_section(
        research_dir=tmp_path / "does_not_exist", strict=False, verbose=False
    )
    assert violations == []


def test_305_external_memory_opt_in_only(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    research = tmp_path / "research"
    _write(
        memory,
        "feedback_xyz_substrate_landed_20260516.md",
        _frontmatter() + "# Body without the section\n",
    )
    # Without memory_dir explicit, the gate does NOT scan memory.
    violations_default = check_substrate_design_memo_has_observability_surface_section(
        research_dir=research, strict=False, verbose=False
    )
    assert violations_default == []
    # With memory_dir explicit, the gate DOES scan.
    violations_opt_in = check_substrate_design_memo_has_observability_surface_section(
        memory_dir=memory, research_dir=research, strict=False, verbose=False
    )
    assert len(violations_opt_in) == 1


def test_305_unrelated_filename_not_in_scope(tmp_path: Path) -> None:
    research = tmp_path / "research"
    _write(
        research,
        "random_notes.md",
        _frontmatter() + "# Body without the section\n",
    )
    violations = check_substrate_design_memo_has_observability_surface_section(
        research_dir=research, strict=False, verbose=False
    )
    assert violations == []


def test_305_string_repo_root_accepted(tmp_path: Path) -> None:
    # Pass repo_root as a str (sister gates accept both str and Path).
    research = tmp_path / "research"
    _write(
        research,
        "my_substrate_design_20260516.md",
        _frontmatter() + "## Observability surface\n",
    )
    violations = check_substrate_design_memo_has_observability_surface_section(
        research_dir=str(research), repo_root=str(tmp_path), strict=False, verbose=False
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Orchestrator wire-in regression guard
# ---------------------------------------------------------------------------


def test_305_orchestrator_callsite_strict_after_wave_1_backfill() -> None:
    """Verify Catalog #305 is strict after the WAVE-1 backfill drove live
    violations to 0."""
    import inspect
    from tac import preflight

    src = inspect.getsource(preflight.preflight_all)
    # Find the line invoking our gate.
    assert (
        "check_substrate_design_memo_has_observability_surface_section"
        in src
    ), "Catalog #305 gate not wired into preflight_all()"
    # Verify the call uses strict=True after WAVE-1 strict-flip.
    # The pattern in preflight_all is:
    # check_substrate_design_memo_has_observability_surface_section(
    #     strict=True, verbose=verbose,
    # )
    idx = src.find("check_substrate_design_memo_has_observability_surface_section(")
    assert idx > 0
    snippet = src[idx : idx + 200]
    assert "strict=True" in snippet, (
        "Catalog #305 should remain strict after WAVE-1 backfilled the "
        "remaining 2026-05-16 design memos and live count reached 0."
    )
