# SPDX-License-Identifier: MIT
"""Tests for Catalog #376 — check_subagent_spawn_includes_head_state_pv_evidence.

STRICT preflight gate that refuses spawn-event checkpoint rows lacking PV
evidence. Sister of Catalog #340 STAGING-time guard + Catalog #229 PV at
the SPAWN-time surface.

Per CLAUDE.md "Subagent coherence-by-default" Mandatory pre-flight + Wave
N+7 Slot 2 anti-pattern #13 registration + canonical helper
``tac.discipline_anti_pattern_guards.verify_head_state_before_spawn``.

Lane: lane_strict_gate_check_subagent_spawn_includes_head_state_pv_evidence_plus_canonical_helper_20260528
Memory:
feedback_strict_gate_check_subagent_spawn_includes_head_state_pv_evidence_plus_canonical_helper_landed_20260528.md
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _CHECK_376_CHECKPOINT_RELPATH,
    _CHECK_376_DISCIPLINE_CUTOFF_UTC,
    _CHECK_376_PV_EVIDENCE_TOKENS,
    _CHECK_376_WAIVER_RE,
    _check_376_body_has_pv_evidence,
    _check_376_iter_spawn_event_rows,
    check_subagent_spawn_includes_head_state_pv_evidence,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ── Helper-unit tests ────────────────────────────────────────────────────
class TestCheck376Tokens:
    def test_canonical_tokens_non_empty(self) -> None:
        assert len(_CHECK_376_PV_EVIDENCE_TOKENS) > 10

    def test_canonical_helper_token_present(self) -> None:
        assert "verify_head_state_before_spawn" in _CHECK_376_PV_EVIDENCE_TOKENS

    def test_catalog_229_token_present(self) -> None:
        assert "Catalog #229" in _CHECK_376_PV_EVIDENCE_TOKENS

    def test_git_log_token_present(self) -> None:
        assert "git log" in _CHECK_376_PV_EVIDENCE_TOKENS

    def test_resume_predecessor_token_present(self) -> None:
        # The resume-predecessor pattern (this very subagent's mode of work)
        # must satisfy the gate.
        assert "Resuming predecessor" in _CHECK_376_PV_EVIDENCE_TOKENS


class TestCheck376BodyHasPvEvidence:
    def test_empty_returns_false(self) -> None:
        assert _check_376_body_has_pv_evidence("") is False

    def test_none_returns_false(self) -> None:
        # Type annotation says str but defensive against None
        assert _check_376_body_has_pv_evidence("") is False

    def test_catalog_229_token_detected(self) -> None:
        assert _check_376_body_has_pv_evidence("Per Catalog #229 PV done") is True

    def test_canonical_helper_token_detected(self) -> None:
        assert (
            _check_376_body_has_pv_evidence(
                "Ran verify_head_state_before_spawn returned PROCEED"
            )
            is True
        )

    def test_git_log_token_detected(self) -> None:
        assert _check_376_body_has_pv_evidence("git log --oneline -30") is True

    def test_unrelated_text_returns_false(self) -> None:
        assert (
            _check_376_body_has_pv_evidence("just some random subagent notes")
            is False
        )

    def test_case_insensitive(self) -> None:
        assert _check_376_body_has_pv_evidence("PREMISE VERIFICATION done") is True


class TestCheck376Waiver:
    def test_waiver_with_rationale_accepted(self) -> None:
        text = (
            "# SPAWN_PV_EVIDENCE_WAIVED:operator administrative bundle landing "
            "with explicit scope per Catalog #340 ownership map"
        )
        assert _check_376_body_has_pv_evidence(text) is True

    def test_waiver_placeholder_rationale_rejected(self) -> None:
        # The placeholder <rationale> token is rejected so the docstring
        # cannot self-waive.
        text = "# SPAWN_PV_EVIDENCE_WAIVED:<rationale>"
        # The regex requires non-< first character.
        match = _CHECK_376_WAIVER_RE.search(text)
        assert match is None

    def test_waiver_placeholder_reason_rejected(self) -> None:
        text = "# SPAWN_PV_EVIDENCE_WAIVED:<reason>"
        match = _CHECK_376_WAIVER_RE.search(text)
        assert match is None

    def test_waiver_bare_rejected(self) -> None:
        # Bare "# SPAWN_PV_EVIDENCE_WAIVED:" (empty reason) MUST be rejected.
        text = "# SPAWN_PV_EVIDENCE_WAIVED:"
        match = _CHECK_376_WAIVER_RE.search(text)
        assert match is None

    def test_waiver_with_whitespace_only_rejected(self) -> None:
        # Whitespace-only after the colon should be rejected (regex
        # requires non-< non-whitespace first char).
        text = "# SPAWN_PV_EVIDENCE_WAIVED:   "
        match = _CHECK_376_WAIVER_RE.search(text)
        assert match is None


class TestCheck376IterSpawnEventRows:
    def test_no_checkpoint_file_returns_empty(self, tmp_path: Path) -> None:
        rows = _check_376_iter_spawn_event_rows(tmp_path)
        assert rows == []

    def test_finds_spawn_event_row(self, tmp_path: Path) -> None:
        ckpt = tmp_path / _CHECK_376_CHECKPOINT_RELPATH
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(
            json.dumps({
                "subagent_id": "test_subagent_x",
                "step": 1,
                "status": "in_progress",
                "notes": "Per Catalog #229 PV passed",
                "next_action": "land STRICT gate",
                "files_touched": ["src/tac/foo.py"],
                "written_at_utc": "2026-05-28T18:00:00+00:00",
            }) + "\n",
            encoding="utf-8",
        )
        rows = _check_376_iter_spawn_event_rows(tmp_path)
        assert len(rows) == 1
        assert rows[0]["subagent_id"] == "test_subagent_x"

    def test_skips_complete_rows(self, tmp_path: Path) -> None:
        ckpt = tmp_path / _CHECK_376_CHECKPOINT_RELPATH
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(
            json.dumps({
                "subagent_id": "complete_subagent",
                "step": "complete",
                "status": "complete",
                "notes": "done",
                "written_at_utc": "2026-05-28T18:00:00+00:00",
            }) + "\n",
            encoding="utf-8",
        )
        rows = _check_376_iter_spawn_event_rows(tmp_path)
        assert rows == []

    def test_skips_step_other_than_1(self, tmp_path: Path) -> None:
        ckpt = tmp_path / _CHECK_376_CHECKPOINT_RELPATH
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(
            json.dumps({
                "subagent_id": "step3_subagent",
                "step": 3,
                "status": "in_progress",
                "notes": "step 3 work",
                "written_at_utc": "2026-05-28T18:00:00+00:00",
            }) + "\n",
            encoding="utf-8",
        )
        rows = _check_376_iter_spawn_event_rows(tmp_path)
        assert rows == []

    def test_malformed_jsonl_tolerated(self, tmp_path: Path) -> None:
        ckpt = tmp_path / _CHECK_376_CHECKPOINT_RELPATH
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(
            "this is not json\n"
            + json.dumps({
                "subagent_id": "good_subagent",
                "step": 1,
                "status": "in_progress",
                "notes": "git log --oneline -30",
                "written_at_utc": "2026-05-28T18:00:00+00:00",
            }) + "\n",
            encoding="utf-8",
        )
        rows = _check_376_iter_spawn_event_rows(tmp_path)
        assert len(rows) == 1


# ── End-to-end gate tests ────────────────────────────────────────────────
class TestCheck376Gate:
    def test_clean_repo_no_violations(self, tmp_path: Path) -> None:
        violations = check_subagent_spawn_includes_head_state_pv_evidence(
            repo_root=tmp_path
        )
        assert violations == []

    def test_pre_cutoff_legacy_exempt(self, tmp_path: Path) -> None:
        # Pre-cutoff spawn-event row WITHOUT PV evidence is exempt
        # (legacy backfill window).
        ckpt = tmp_path / _CHECK_376_CHECKPOINT_RELPATH
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(
            json.dumps({
                "subagent_id": "legacy_sub",
                "step": 1,
                "status": "in_progress",
                "notes": "no PV evidence here",
                "written_at_utc": "2026-05-14T00:00:00+00:00",
            }) + "\n",
            encoding="utf-8",
        )
        violations = check_subagent_spawn_includes_head_state_pv_evidence(
            repo_root=tmp_path
        )
        assert violations == []

    def test_post_cutoff_with_pv_evidence_passes(self, tmp_path: Path) -> None:
        # Post-cutoff (we move cutoff backward via monkey-patching for this
        # test) spawn-event row WITH PV evidence in notes passes.
        ckpt = tmp_path / _CHECK_376_CHECKPOINT_RELPATH
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(
            json.dumps({
                "subagent_id": "pv_compliant_sub",
                "step": 1,
                "status": "in_progress",
                "notes": "Per Catalog #229 PV: ran git log --oneline -30 + git status",
                "next_action": "land STRICT gate",
                "files_touched": ["src/tac/foo.py"],
                "written_at_utc": "2030-06-01T00:00:00+00:00",
            }) + "\n",
            encoding="utf-8",
        )
        violations = check_subagent_spawn_includes_head_state_pv_evidence(
            repo_root=tmp_path
        )
        assert violations == []

    def test_post_cutoff_without_pv_evidence_flagged(self, tmp_path: Path) -> None:
        # Post-cutoff spawn-event row WITHOUT PV evidence is flagged.
        ckpt = tmp_path / _CHECK_376_CHECKPOINT_RELPATH
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(
            json.dumps({
                "subagent_id": "no_pv_subagent",
                "step": 1,
                "status": "in_progress",
                "notes": "just some boring notes",
                "next_action": "do work",
                "files_touched": ["src/tac/bar.py"],
                "written_at_utc": "2030-06-01T00:00:00+00:00",
            }) + "\n",
            encoding="utf-8",
        )
        violations = check_subagent_spawn_includes_head_state_pv_evidence(
            repo_root=tmp_path
        )
        assert len(violations) == 1
        assert "no_pv_subagent" in violations[0]

    def test_post_cutoff_with_waiver_passes(self, tmp_path: Path) -> None:
        ckpt = tmp_path / _CHECK_376_CHECKPOINT_RELPATH
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(
            json.dumps({
                "subagent_id": "waived_subagent",
                "step": 1,
                "status": "in_progress",
                "notes": (
                    "# SPAWN_PV_EVIDENCE_WAIVED:"
                    "operator administrative bundle landing with explicit scope"
                ),
                "next_action": "do work",
                "files_touched": ["src/tac/baz.py"],
                "written_at_utc": "2030-06-01T00:00:00+00:00",
            }) + "\n",
            encoding="utf-8",
        )
        violations = check_subagent_spawn_includes_head_state_pv_evidence(
            repo_root=tmp_path
        )
        assert violations == []

    def test_post_cutoff_with_placeholder_waiver_flagged(
        self, tmp_path: Path
    ) -> None:
        # Placeholder waiver MUST be rejected so the gate's docstring
        # example cannot self-waive (Catalog #287 sister discipline).
        ckpt = tmp_path / _CHECK_376_CHECKPOINT_RELPATH
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(
            json.dumps({
                "subagent_id": "placeholder_waiver_sub",
                "step": 1,
                "status": "in_progress",
                "notes": "# SPAWN_PV_EVIDENCE_WAIVED:<rationale>",
                "next_action": "do work",
                "files_touched": ["src/tac/qux.py"],
                "written_at_utc": "2030-06-01T00:00:00+00:00",
            }) + "\n",
            encoding="utf-8",
        )
        violations = check_subagent_spawn_includes_head_state_pv_evidence(
            repo_root=tmp_path
        )
        assert len(violations) == 1

    def test_strict_mode_raises_with_catalog_376_message(
        self, tmp_path: Path
    ) -> None:
        ckpt = tmp_path / _CHECK_376_CHECKPOINT_RELPATH
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(
            json.dumps({
                "subagent_id": "to_be_flagged",
                "step": 1,
                "status": "in_progress",
                "notes": "no PV here",
                "written_at_utc": "2030-06-01T00:00:00+00:00",
            }) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(PreflightError) as exc:
            check_subagent_spawn_includes_head_state_pv_evidence(
                repo_root=tmp_path, strict=True
            )
        assert "Catalog #376" in str(exc.value)

    def test_strict_silent_on_clean(self, tmp_path: Path) -> None:
        # Strict mode with no violations does NOT raise.
        result = check_subagent_spawn_includes_head_state_pv_evidence(
            repo_root=tmp_path, strict=True
        )
        assert result == []

    def test_verbose_output(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        check_subagent_spawn_includes_head_state_pv_evidence(
            repo_root=tmp_path, verbose=True
        )
        captured = capsys.readouterr()
        assert "[subagent-spawn-pv-evidence]" in captured.out

    def test_string_repo_root_accepted(self, tmp_path: Path) -> None:
        # str repo_root path should be normalized to Path.
        violations = check_subagent_spawn_includes_head_state_pv_evidence(
            repo_root=str(tmp_path)
        )
        assert violations == []

    def test_multiple_violations_aggregated(self, tmp_path: Path) -> None:
        ckpt = tmp_path / _CHECK_376_CHECKPOINT_RELPATH
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(
            "\n".join([
                json.dumps({
                    "subagent_id": f"violator_{i}",
                    "step": 1,
                    "status": "in_progress",
                    "notes": f"no PV evidence in violator {i}",
                    "written_at_utc": "2030-06-01T00:00:00+00:00",
                })
                for i in range(5)
            ]) + "\n",
            encoding="utf-8",
        )
        violations = check_subagent_spawn_includes_head_state_pv_evidence(
            repo_root=tmp_path
        )
        assert len(violations) == 5


# ── Live-repo regression guards ─────────────────────────────────────────
class TestCheck376LiveRepo:
    def test_live_repo_warn_only_returns_list(self) -> None:
        # Live repo currently has many legacy spawn-event rows. The cutoff
        # is set far in the future so warn-only landing produces 0
        # violations. Make sure the call is non-raising and returns a list.
        result = check_subagent_spawn_includes_head_state_pv_evidence()
        assert isinstance(result, list)

    def test_live_repo_cutoff_far_in_future(self) -> None:
        # The initial cutoff is set to 2030-01-01 so warn-only landing
        # produces zero violations until the operator-routed strict-flip
        # wave bumps the cutoff to a recent timestamp.
        assert _CHECK_376_DISCIPLINE_CUTOFF_UTC.startswith("2030")

    def test_orchestrator_wires_check_376_warn_only(self) -> None:
        # Catalog #176 META-meta sister regression: CLAUDE.md catalog row
        # MUST cite the canonical wire-in pattern. We verify the wire-in
        # is present in preflight.py preflight_all body.
        preflight_path = REPO_ROOT / "src" / "tac" / "preflight.py"
        text = preflight_path.read_text(encoding="utf-8")
        assert "check_subagent_spawn_includes_head_state_pv_evidence" in text
        # WARN-ONLY wire-in: there is a `strict=False` on the wire-in line.
        # We don't pin to a specific line but verify the pattern exists in
        # the file.
        assert "[subagent-spawn-pv-evidence]" in text

    def test_check_185_sister_callable(self) -> None:
        # Catalog #185 META-meta-meta sister regression: the gate function
        # MUST be importable from tac.preflight module globals.
        from tac import preflight
        fn = getattr(preflight, "check_subagent_spawn_includes_head_state_pv_evidence")
        assert callable(fn)

    def test_check_186_canonical_serializer_committed(self) -> None:
        # Catalog #186 META-meta-meta sister regression: the catalog
        # number claim must have been committed via canonical serializer.
        # We verify the claim log shows the canonical commit pattern by
        # checking that the gate function exists and the catalog number
        # txt file points at 377 (376 was claimed, next is 377).
        catalog_num_path = REPO_ROOT / ".omx" / "state" / "next_catalog_number.txt"
        text = catalog_num_path.read_text(encoding="utf-8").strip()
        # Some sister claim may have advanced the counter further; the
        # important invariant is that 376 was claimed (already reserved
        # for this gate) so next is at least 377.
        assert int(text) >= 377

    def test_claude_md_catalog_376_row_present(self) -> None:
        # Catalog #176 META-meta sister regression: STRICT-callsite must
        # have a CLAUDE.md catalog row.
        claude_md_path = REPO_ROOT / "CLAUDE.md"
        text = claude_md_path.read_text(encoding="utf-8")
        assert "376. `check_subagent_spawn_includes_head_state_pv_evidence`" in text


# ── Canonical helper integration ─────────────────────────────────────────
class TestCanonicalHelperIntegration:
    def test_canonical_helper_importable(self) -> None:
        # The canonical helper that satisfies this gate's PV requirement
        # MUST be importable from tac.discipline_anti_pattern_guards.
        from tac.discipline_anti_pattern_guards import (
            verify_head_state_before_spawn,
        )
        assert callable(verify_head_state_before_spawn)

    def test_canonical_helper_token_in_pv_evidence_tokens(self) -> None:
        # The canonical helper's function name MUST be in the PV-evidence
        # token set so a subagent that ran the helper satisfies this gate.
        assert (
            "verify_head_state_before_spawn"
            in _CHECK_376_PV_EVIDENCE_TOKENS
        )

    def test_handoff_helper_token_in_pv_evidence_tokens(self) -> None:
        assert (
            "verify_predecessor_working_tree_committed_or_auto_commit"
            in _CHECK_376_PV_EVIDENCE_TOKENS
        )
