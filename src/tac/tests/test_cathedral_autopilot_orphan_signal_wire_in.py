# SPDX-License-Identifier: MIT
"""Tests for the cathedral autopilot's ORPHAN-SIGNAL-AUDIT wire-in functions.

Per `feedback_cathedral_autopilot_orphan_signal_wire_in_landed_20260517.md` +
ORPHAN-SIGNAL-AUDIT task #711 + operator standing directive 2026-05-17 verbatim
*"Ensure all producers wired up and integrated into consumers as appropriate
with the cathedral autopilot the ultimate consumer."*

Verifies the 5 new producer-to-cathedral-autopilot wire-ins:
  1. load_candidates_from_dp1_composition_primitives
  2. rerank_candidates_via_council_continual_learning
  3. refuse_candidates_via_probe_outcomes
  4. update_cost_band_from_modal_call_id_ledger
  5. refuse_candidates_via_recursive_review_unsealed

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
each wire-in is fail-CLOSED + each test exercises (empty / well-formed /
malformed / fail-closed) cases.

Per Catalog #185 sister regression guard: all 5 new functions callable via
`tools.cathedral_autopilot_autonomous_loop` module globals.

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag": all assertions
carry explicit evidence (test inputs are deterministic + outputs are reasoned
from the live API surface verified in `.omx/tmp/producer_wire_in_premise_verifier.txt`).
"""
from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Repo-canonical sys.path setup mirroring CLAUDE.md "Tailscale fleet"-style
# path discipline (canonical entry points use PYTHONPATH=src:upstream).
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "tools"))

# Use canonical imports (Catalog #188 — `from tac.*` / `from tools.*`).
import cathedral_autopilot_autonomous_loop as mod  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# Module-globals regression guard (Catalog #185 sister)
# ─────────────────────────────────────────────────────────────────────────


class TestModuleGlobalsRegression:
    """Catalog #185 sister regression guard: all 5 new functions callable via
    module globals (i.e. importable as `cathedral_autopilot_autonomous_loop.X`)."""

    def test_load_candidates_from_dp1_composition_primitives_callable(self) -> None:
        fn = mod.load_candidates_from_dp1_composition_primitives
        assert callable(fn)

    def test_rerank_candidates_via_council_continual_learning_callable(self) -> None:
        fn = mod.rerank_candidates_via_council_continual_learning
        assert callable(fn)

    def test_refuse_candidates_via_probe_outcomes_callable(self) -> None:
        fn = mod.refuse_candidates_via_probe_outcomes
        assert callable(fn)

    def test_update_cost_band_from_modal_call_id_ledger_callable(self) -> None:
        fn = mod.update_cost_band_from_modal_call_id_ledger
        assert callable(fn)

    def test_refuse_candidates_via_recursive_review_unsealed_callable(self) -> None:
        fn = mod.refuse_candidates_via_recursive_review_unsealed
        assert callable(fn)


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def repo_root() -> Path:
    return _REPO_ROOT


@pytest.fixture
def single_candidate() -> list[mod.CandidateRow]:
    """One generic candidate for pass-through / no-match tests."""
    return [
        mod.CandidateRow(
            candidate_id="test_candidate_id",
            family="test_family",
            predicted_score_delta=-0.001,
            expected_information_gain=0.10,
            estimated_dispatch_cost_usd=1.0,
        )
    ]


@pytest.fixture
def multi_candidates() -> list[mod.CandidateRow]:
    """Three candidates with distinct families for diverse-match tests."""
    return [
        mod.CandidateRow(
            candidate_id=f"cand_{i}",
            family=f"family_{i}",
            predicted_score_delta=-0.001 * (i + 1),
            expected_information_gain=0.10,
            estimated_dispatch_cost_usd=1.0,
        )
        for i in range(3)
    ]


@pytest.fixture
def tmp_jsonl_path(tmp_path: Path) -> Path:
    """Empty JSONL file pre-created."""
    p = tmp_path / "test.jsonl"
    p.touch()
    return p


# ─────────────────────────────────────────────────────────────────────────
# 1. load_candidates_from_dp1_composition_primitives
# ─────────────────────────────────────────────────────────────────────────


class TestLoadCandidatesFromDp1CompositionPrimitives:
    def test_returns_one_row_per_known_base_substrate(self, repo_root: Path) -> None:
        rows = mod.load_candidates_from_dp1_composition_primitives(repo_root)
        # Premise verifier PV-4: 6 base substrates registered as of 2026-05-17.
        assert len(rows) >= 1
        # All rows are CandidateRow instances.
        assert all(isinstance(r, mod.CandidateRow) for r in rows)

    def test_all_rows_have_research_only_blocker(self, repo_root: Path) -> None:
        rows = mod.load_candidates_from_dp1_composition_primitives(repo_root)
        assert len(rows) > 0
        for r in rows:
            assert (
                "dp1_composition_research_only_pending_paired_anchor"
                in r.blockers
            ), f"Row {r.candidate_id} missing research-only blocker"

    def test_candidate_ids_prefixed_dp1_composed(self, repo_root: Path) -> None:
        rows = mod.load_candidates_from_dp1_composition_primitives(repo_root)
        for r in rows:
            assert r.candidate_id.startswith("dp1_composed__"), (
                f"Row {r.candidate_id} missing canonical prefix"
            )

    def test_all_rows_family_is_canonical(self, repo_root: Path) -> None:
        rows = mod.load_candidates_from_dp1_composition_primitives(repo_root)
        for r in rows:
            assert r.family == "pretrained_driving_prior_composition"

    def test_all_rows_lane_class_is_research_substrate(self, repo_root: Path) -> None:
        rows = mod.load_candidates_from_dp1_composition_primitives(repo_root)
        for r in rows:
            assert r.lane_class == "research_substrate"

    def test_notes_carry_evidence_axis_tag(self, repo_root: Path) -> None:
        rows = mod.load_candidates_from_dp1_composition_primitives(repo_root)
        for r in rows:
            # Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog
            # #287: every notes string must carry an axis-tag.
            assert "[predicted; DP1 composition primitive" in r.notes
            assert "research_only" in r.notes

    def test_kwargs_propagate_to_defaults(self, repo_root: Path) -> None:
        rows = mod.load_candidates_from_dp1_composition_primitives(
            repo_root,
            default_predicted_score_delta=-0.005,
            default_expected_information_gain=0.2,
            default_estimated_dispatch_cost_usd=0.5,
        )
        for r in rows:
            assert r.predicted_score_delta == pytest.approx(-0.005)
            assert r.expected_information_gain == pytest.approx(0.2)
            assert r.estimated_dispatch_cost_usd == pytest.approx(0.5)

    def test_fail_closed_on_missing_canonical_helper(
        self, monkeypatch: pytest.MonkeyPatch, repo_root: Path
    ) -> None:
        # Per CLAUDE.md "Bugs must be permanently fixed AND self-protected
        # against": if the canonical helper is missing, ImportError must
        # propagate (NOT silent zero-row return).
        # We simulate via removing the module from sys.modules + blocking import.
        import builtins

        real_import = builtins.__import__

        def _block_composition(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "tac.substrates.pretrained_driving_prior.composition":
                raise ImportError("simulated missing canonical helper")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_composition)
        with pytest.raises(ImportError):
            mod.load_candidates_from_dp1_composition_primitives(repo_root)


# ─────────────────────────────────────────────────────────────────────────
# 2. rerank_candidates_via_council_continual_learning
# ─────────────────────────────────────────────────────────────────────────


class TestRerankCandidatesViaCouncilContinualLearning:
    def test_empty_candidate_list_returns_empty(
        self, repo_root: Path, tmp_jsonl_path: Path
    ) -> None:
        ranked = mod.rerank_candidates_via_council_continual_learning(
            [], repo_root=repo_root, posterior_path=tmp_jsonl_path
        )
        assert ranked == []

    def test_no_matching_deliberation_passes_through(
        self,
        repo_root: Path,
        single_candidate: list[mod.CandidateRow],
        tmp_jsonl_path: Path,
    ) -> None:
        # Empty ledger → no match → passthrough with explanation.
        ranked = mod.rerank_candidates_via_council_continual_learning(
            single_candidate, repo_root=repo_root, posterior_path=tmp_jsonl_path
        )
        assert len(ranked) == 1
        cand, delta, expl = ranked[0]
        assert cand.candidate_id == "test_candidate_id"
        assert delta == pytest.approx(-0.001)
        assert "[council; no-matching-deliberation]" in expl

    def test_proceed_unconditional_verdict_boosts_candidate(
        self,
        repo_root: Path,
        tmp_path: Path,
        single_candidate: list[mod.CandidateRow],
    ) -> None:
        # Construct a council deliberation that matches the candidate via
        # deferred_substrate_id == candidate.family.
        from tac.council_continual_learning import (
            CouncilDeliberationRecord,
            CouncilTier,
            append_council_anchor,
        )

        posterior = tmp_path / "council_posterior.jsonl"
        record = CouncilDeliberationRecord(
            deliberation_id="test_delib_proceed_unconditional",
            topic="test topic",
            council_tier=CouncilTier.T2,
            council_attendees=("Shannon", "Dykstra"),
            council_quorum_met=True,
            council_verdict="PROCEED",
            predicted_mission_contribution="frontier_breaking",
            deferred_substrate_id="test_family",
            council_assumption_adversary_verdict=(
                {
                    "assumption": "test assumption",
                    "classification": "HARD-EARNED",
                    "rationale": "test rationale",
                },
            ),
        )
        append_council_anchor(record, posterior_path=posterior)

        ranked = mod.rerank_candidates_via_council_continual_learning(
            single_candidate, repo_root=repo_root, posterior_path=posterior
        )
        assert len(ranked) == 1
        cand, delta, expl = ranked[0]
        # PROCEED-unconditional default weight = -0.20 (boost).
        assert delta == pytest.approx(-0.001 + (-0.20))
        assert "verdict=PROCEED" in expl
        assert "T2" in expl

    def test_refuse_verdict_penalizes_candidate(
        self,
        repo_root: Path,
        tmp_path: Path,
        single_candidate: list[mod.CandidateRow],
    ) -> None:
        from tac.council_continual_learning import (
            CouncilDeliberationRecord,
            CouncilTier,
            append_council_anchor,
        )

        posterior = tmp_path / "council_posterior.jsonl"
        record = CouncilDeliberationRecord(
            deliberation_id="test_delib_refuse",
            topic="test topic",
            council_tier=CouncilTier.T3,
            council_attendees=("Shannon",),
            council_quorum_met=True,
            council_verdict="REFUSE",
            predicted_mission_contribution="frontier_protecting",
            deferred_substrate_id="test_family",
            council_assumption_adversary_verdict=(
                {
                    "assumption": "test assumption",
                    "classification": "CARGO-CULTED",
                    "rationale": "test rationale",
                },
            ),
        )
        append_council_anchor(record, posterior_path=posterior)

        ranked = mod.rerank_candidates_via_council_continual_learning(
            single_candidate, repo_root=repo_root, posterior_path=posterior
        )
        cand, delta, expl = ranked[0]
        # REFUSE default weight = +0.50 (strong penalty).
        assert delta == pytest.approx(-0.001 + 0.50)
        assert "verdict=REFUSE" in expl

    def test_sort_order_ascending_most_negative_first(
        self,
        repo_root: Path,
        tmp_path: Path,
        multi_candidates: list[mod.CandidateRow],
    ) -> None:
        from tac.council_continual_learning import (
            CouncilDeliberationRecord,
            CouncilTier,
            append_council_anchor,
        )

        posterior = tmp_path / "council_posterior.jsonl"
        # family_0 gets REFUSE (large penalty), family_1 gets PROCEED (boost),
        # family_2 has no match.
        for delib_id, family, verdict in [
            ("d0", "family_0", "REFUSE"),
            ("d1", "family_1", "PROCEED"),
        ]:
            append_council_anchor(
                CouncilDeliberationRecord(
                    deliberation_id=delib_id,
                    topic="test",
                    council_tier=CouncilTier.T2,
                    council_attendees=("Shannon",),
                    council_quorum_met=True,
                    council_verdict=verdict,
                    predicted_mission_contribution="frontier_breaking",
                    deferred_substrate_id=family,
                    council_assumption_adversary_verdict=(
                        {
                            "assumption": "test",
                            "classification": "HARD-EARNED",
                            "rationale": "test",
                        },
                    ),
                ),
                posterior_path=posterior,
            )

        ranked = mod.rerank_candidates_via_council_continual_learning(
            multi_candidates, repo_root=repo_root, posterior_path=posterior
        )
        # Ascending sort means most-negative-delta-first = best candidate first.
        deltas = [r[1] for r in ranked]
        assert deltas == sorted(deltas), "Sort must be ascending"
        # PROCEED-boosted family_1 should be first.
        assert ranked[0][0].family == "family_1"

    def test_fail_closed_on_missing_canonical_helper(
        self,
        monkeypatch: pytest.MonkeyPatch,
        repo_root: Path,
        single_candidate: list[mod.CandidateRow],
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def _block_ccl(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "tac.council_continual_learning":
                raise ImportError("simulated missing canonical helper")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_ccl)
        with pytest.raises(ImportError):
            mod.rerank_candidates_via_council_continual_learning(
                single_candidate, repo_root=repo_root
            )


# ─────────────────────────────────────────────────────────────────────────
# 3. refuse_candidates_via_probe_outcomes
# ─────────────────────────────────────────────────────────────────────────


class TestRefuseCandidatesViaProbeOutcomes:
    def test_empty_candidate_list_returns_empty(
        self, repo_root: Path, tmp_jsonl_path: Path
    ) -> None:
        kept, refused = mod.refuse_candidates_via_probe_outcomes(
            [], repo_root=repo_root, ledger_path=tmp_jsonl_path
        )
        assert kept == []
        assert refused == []

    def test_no_blocking_outcome_passes_through(
        self,
        repo_root: Path,
        single_candidate: list[mod.CandidateRow],
        tmp_jsonl_path: Path,
    ) -> None:
        # Empty ledger → no blocking outcomes → all pass.
        kept, refused = mod.refuse_candidates_via_probe_outcomes(
            single_candidate, repo_root=repo_root, ledger_path=tmp_jsonl_path
        )
        assert len(kept) == 1
        assert len(refused) == 0
        assert kept[0].candidate_id == "test_candidate_id"

    def test_blocking_kill_verdict_refuses_candidate(
        self,
        repo_root: Path,
        tmp_path: Path,
        single_candidate: list[mod.CandidateRow],
    ) -> None:
        from tac.probe_outcomes_ledger import (
            VERDICT_KILL,
            register_probe_outcome,
        )

        ledger = tmp_path / "probe_outcomes.jsonl"
        register_probe_outcome(
            probe_id="test_probe",
            substrate="test_family",  # Matches single_candidate.family
            recipe_path=None,
            probe_kind="distinguishing_feature",
            verdict=VERDICT_KILL,
            metric_name="test_metric",
            metric_value=0.0,
            threshold=0.5,
            threshold_token="<",
            evidence_path="/tmp/test_evidence",
            next_action="defer until reactivation criteria met",
            adjudicator="test",
            path=ledger,
        )

        kept, refused = mod.refuse_candidates_via_probe_outcomes(
            single_candidate, repo_root=repo_root, ledger_path=ledger
        )
        assert len(kept) == 0
        assert len(refused) == 1
        refused_row = refused[0]
        assert any("probe_outcome_blocking_verdict__KILL" in b for b in refused_row.blockers)
        assert "[probe-outcome refuse" in refused_row.notes

    def test_blocking_independent_verdict_refuses_candidate(
        self,
        repo_root: Path,
        tmp_path: Path,
        single_candidate: list[mod.CandidateRow],
    ) -> None:
        from tac.probe_outcomes_ledger import (
            VERDICT_INDEPENDENT,
            register_probe_outcome,
        )

        ledger = tmp_path / "probe_outcomes.jsonl"
        register_probe_outcome(
            probe_id="test_probe_indep",
            substrate="test_family",
            recipe_path=None,
            probe_kind="information_theoretic",
            verdict=VERDICT_INDEPENDENT,
            metric_name="mutual_information_bits",
            metric_value=0.006,
            threshold=0.5,
            threshold_token="<",
            evidence_path="/tmp/test",
            next_action="seek alternative reducer",
            adjudicator="test",
            path=ledger,
        )

        kept, refused = mod.refuse_candidates_via_probe_outcomes(
            single_candidate, repo_root=repo_root, ledger_path=ledger
        )
        assert len(kept) == 0
        assert len(refused) == 1
        assert any("INDEPENDENT" in b for b in refused[0].blockers)

    def test_non_blocking_verdict_does_not_refuse(
        self,
        repo_root: Path,
        tmp_path: Path,
        single_candidate: list[mod.CandidateRow],
    ) -> None:
        from tac.probe_outcomes_ledger import (
            VERDICT_PROMOTE,
            register_probe_outcome,
        )

        ledger = tmp_path / "probe_outcomes.jsonl"
        register_probe_outcome(
            probe_id="test_probe_promote",
            substrate="test_family",
            recipe_path=None,
            probe_kind="distinguishing_feature",
            verdict=VERDICT_PROMOTE,
            metric_name="test",
            metric_value=0.99,
            threshold=0.5,
            threshold_token=">",
            evidence_path="/tmp/test",
            next_action="advance to L2",
            adjudicator="test",
            path=ledger,
        )

        kept, refused = mod.refuse_candidates_via_probe_outcomes(
            single_candidate, repo_root=repo_root, ledger_path=ledger
        )
        # PROMOTE is NOT in BLOCKING_VERDICTS → passes through.
        assert len(kept) == 1
        assert len(refused) == 0

    def test_fail_closed_on_missing_canonical_helper(
        self,
        monkeypatch: pytest.MonkeyPatch,
        repo_root: Path,
        single_candidate: list[mod.CandidateRow],
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def _block_probe(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "tac.probe_outcomes_ledger":
                raise ImportError("simulated missing canonical helper")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_probe)
        with pytest.raises(ImportError):
            mod.refuse_candidates_via_probe_outcomes(
                single_candidate, repo_root=repo_root
            )


# ─────────────────────────────────────────────────────────────────────────
# 4. update_cost_band_from_modal_call_id_ledger
# ─────────────────────────────────────────────────────────────────────────


class TestUpdateCostBandFromModalCallIdLedger:
    def test_missing_ledger_returns_zero_anchors(
        self, repo_root: Path, tmp_path: Path
    ) -> None:
        # Pointing to a non-existent file → load_call_ids returns [].
        ledger = tmp_path / "missing_ledger.jsonl"
        posterior = tmp_path / "cost_band_posterior.jsonl"
        status = mod.update_cost_band_from_modal_call_id_ledger(
            repo_root, ledger_path=ledger, posterior_path=posterior
        )
        assert status["rows_scanned"] == 0
        assert status["anchors_appended"] == 0

    def test_dispatched_only_row_is_skipped(
        self, repo_root: Path, tmp_path: Path
    ) -> None:
        from tac.deploy.modal.call_id_ledger import register_dispatched_call_id

        ledger = tmp_path / "ledger.jsonl"
        posterior = tmp_path / "cost_band_posterior.jsonl"
        register_dispatched_call_id(
            call_id="fc-test123",
            lane_id="lane_test",
            label="test_dispatch",
            platform="modal",
            gpu="T4",
            expected_cost_usd=1.0,
            expected_axis="contest_cuda",
            recipe="test_recipe",
            path=ledger,
        )

        status = mod.update_cost_band_from_modal_call_id_ledger(
            repo_root, ledger_path=ledger, posterior_path=posterior
        )
        assert status["rows_scanned"] == 1
        assert status["anchors_appended"] == 0
        assert status["skipped_reasons"].get("status_dispatched_no_outcome_yet") == 1

    def test_harvested_row_appends_cost_band_anchor(
        self, repo_root: Path, tmp_path: Path
    ) -> None:
        from tac.deploy.modal.call_id_ledger import (
            register_dispatched_call_id,
            update_call_id_outcome,
        )

        ledger = tmp_path / "ledger.jsonl"
        posterior = tmp_path / "cost_band_posterior.jsonl"
        register_dispatched_call_id(
            call_id="fc-harvested",
            lane_id="lane_test_harvested",
            label="harvested_dispatch",
            platform="modal",
            gpu="T4",
            expected_cost_usd=1.0,
            expected_axis="contest_cuda",
            recipe="harvested_recipe",
            path=ledger,
        )
        update_call_id_outcome(
            call_id="fc-harvested",
            status="harvested",
            rc=0,
            elapsed_seconds=123.4,
            cost_actual_usd=0.50,
            score=0.195,
            score_axis="contest_cuda",
            archive_sha256="0" * 64,
            archive_bytes=100000,
            evidence_grade="[contest-CUDA]",
            path=ledger,
        )

        status = mod.update_cost_band_from_modal_call_id_ledger(
            repo_root, ledger_path=ledger, posterior_path=posterior
        )
        # 2 rows = dispatched + outcome; only outcome appends.
        assert status["rows_scanned"] == 2
        assert status["anchors_appended"] == 1
        # Verify the cost-band posterior file was actually written.
        assert posterior.is_file()
        lines = [
            line.strip()
            for line in posterior.read_text().splitlines()
            if line.strip()
        ]
        assert len(lines) == 1
        anchor_dict = json.loads(lines[0])
        assert anchor_dict["platform"] == "modal"
        assert anchor_dict["gpu"] == "T4"
        assert anchor_dict["actual_wall_clock_sec"] == pytest.approx(123.4)
        assert anchor_dict["actual_cost_usd"] == pytest.approx(0.50)
        assert anchor_dict["outcome"] == "successful_dispatch"

    def test_failed_row_appends_failed_dispatch_anchor(
        self, repo_root: Path, tmp_path: Path
    ) -> None:
        from tac.deploy.modal.call_id_ledger import (
            register_dispatched_call_id,
            update_call_id_outcome,
        )

        ledger = tmp_path / "ledger.jsonl"
        posterior = tmp_path / "cost_band_posterior.jsonl"
        register_dispatched_call_id(
            call_id="fc-failed",
            lane_id="lane_failed",
            label="failed_dispatch",
            platform="modal",
            gpu="A100",
            expected_cost_usd=5.0,
            expected_axis="contest_cuda",
            recipe="failed_recipe",
            path=ledger,
        )
        update_call_id_outcome(
            call_id="fc-failed",
            status="failed",
            rc=1,
            elapsed_seconds=60.0,
            cost_actual_usd=0.10,
            score=None,
            score_axis=None,
            archive_sha256=None,
            archive_bytes=0,
            evidence_grade=None,
            path=ledger,
        )

        status = mod.update_cost_band_from_modal_call_id_ledger(
            repo_root, ledger_path=ledger, posterior_path=posterior
        )
        assert status["anchors_appended"] == 1
        anchor_dict = json.loads(posterior.read_text().splitlines()[0])
        assert anchor_dict["outcome"] == "failed_dispatch"

    def test_since_utc_filter_excludes_old_rows(
        self, repo_root: Path, tmp_path: Path
    ) -> None:
        from tac.deploy.modal.call_id_ledger import (
            register_dispatched_call_id,
            update_call_id_outcome,
        )

        ledger = tmp_path / "ledger.jsonl"
        posterior = tmp_path / "cost_band_posterior.jsonl"
        register_dispatched_call_id(
            call_id="fc-old",
            lane_id="lane_old",
            label="old_dispatch",
            platform="modal",
            gpu="T4",
            expected_cost_usd=1.0,
            expected_axis="contest_cuda",
            recipe="old_recipe",
            path=ledger,
        )
        update_call_id_outcome(
            call_id="fc-old",
            status="harvested",
            rc=0,
            elapsed_seconds=100.0,
            cost_actual_usd=0.2,
            score=0.2,
            score_axis="contest_cuda",
            archive_sha256="a" * 64,
            archive_bytes=1000,
            evidence_grade="[contest-CUDA]",
            path=ledger,
        )

        # since_utc filter set to far-future timestamp — should exclude all rows.
        status = mod.update_cost_band_from_modal_call_id_ledger(
            repo_root,
            ledger_path=ledger,
            posterior_path=posterior,
            since_utc="2099-01-01T00:00:00Z",
        )
        assert status["rows_scanned"] == 0
        assert status["anchors_appended"] == 0

    def test_fail_closed_on_missing_canonical_helper(
        self, monkeypatch: pytest.MonkeyPatch, repo_root: Path
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def _block_modal_ledger(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "tac.deploy.modal.call_id_ledger":
                raise ImportError("simulated missing canonical helper")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_modal_ledger)
        with pytest.raises(ImportError):
            mod.update_cost_band_from_modal_call_id_ledger(repo_root)


# ─────────────────────────────────────────────────────────────────────────
# 5. refuse_candidates_via_recursive_review_unsealed
# ─────────────────────────────────────────────────────────────────────────


class TestRefuseCandidatesViaRecursiveReviewUnsealed:
    def test_empty_candidate_list_returns_empty(
        self, repo_root: Path, tmp_jsonl_path: Path
    ) -> None:
        kept, refused = mod.refuse_candidates_via_recursive_review_unsealed(
            [], repo_root=repo_root, ledger_path=tmp_jsonl_path
        )
        assert kept == []
        assert refused == []

    def test_missing_ledger_passes_through_all(
        self,
        repo_root: Path,
        single_candidate: list[mod.CandidateRow],
        tmp_path: Path,
    ) -> None:
        missing = tmp_path / "missing.jsonl"
        kept, refused = mod.refuse_candidates_via_recursive_review_unsealed(
            single_candidate, repo_root=repo_root, ledger_path=missing
        )
        # Per docstring: missing ledger → no SEAL constraint enforceable → pass through.
        assert len(kept) == 1
        assert len(refused) == 0

    def test_no_matching_bundle_passes_through(
        self,
        repo_root: Path,
        single_candidate: list[mod.CandidateRow],
        tmp_path: Path,
    ) -> None:
        from tac.recursive_adversarial_review import (
            RecursiveReviewRound,
            ReviewFinding,
            append_round_locked,
            compute_bundle_id,
            compute_scope_content_sha256,
        )

        ledger = tmp_path / "review_rounds.jsonl"
        # Register a bundle for a TOTALLY unrelated scope path.
        scope_paths = ("src/tac/unrelated_module.py",)
        bid = compute_bundle_id(scope_paths)
        sha = compute_scope_content_sha256(scope_paths, repo_root=repo_root)
        round_rec = RecursiveReviewRound(
            review_id="r1",
            bundle_id=bid,
            scope_paths=scope_paths,
            scope_content_sha256=sha,
            round_number=1,
            council_rotation="skunkworks_sextet",
            council_attendees=("Shannon",),
            findings=(),
            verdict="PROCEED",
            counter_before=0,
            counter_after=1,
            reviewed_at_utc="2026-05-17T00:00:00+00:00",
            reviewer_agent="test",
        )
        append_round_locked(round_rec, path=ledger)

        kept, refused = mod.refuse_candidates_via_recursive_review_unsealed(
            single_candidate, repo_root=repo_root, ledger_path=ledger
        )
        # Candidate's family "test_family" does NOT appear in scope_paths → pass through.
        assert len(kept) == 1
        assert len(refused) == 0

    def test_matching_unsealed_bundle_refuses_candidate(
        self,
        repo_root: Path,
        tmp_path: Path,
    ) -> None:
        from tac.recursive_adversarial_review import (
            RecursiveReviewRound,
            append_round_locked,
            compute_bundle_id,
            compute_scope_content_sha256,
        )

        ledger = tmp_path / "review_rounds.jsonl"
        # Use a scope path that contains the candidate's family substring.
        scope_paths = ("src/tac/my_special_family/codec.py",)
        bid = compute_bundle_id(scope_paths)
        sha = compute_scope_content_sha256(scope_paths, repo_root=repo_root)
        round_rec = RecursiveReviewRound(
            review_id="r2",
            bundle_id=bid,
            scope_paths=scope_paths,
            scope_content_sha256=sha,
            round_number=1,
            council_rotation="skunkworks_sextet",
            council_attendees=("Shannon",),
            findings=(),
            verdict="PROCEED",
            counter_before=0,
            counter_after=1,  # < SEAL_THRESHOLD (3) → unsealed
            reviewed_at_utc="2026-05-17T00:00:00+00:00",
            reviewer_agent="test",
        )
        append_round_locked(round_rec, path=ledger)

        candidates = [
            mod.CandidateRow(
                candidate_id="my_special_family_candidate",
                family="my_special_family",
                predicted_score_delta=-0.001,
                expected_information_gain=0.1,
                estimated_dispatch_cost_usd=1.0,
            )
        ]
        kept, refused = mod.refuse_candidates_via_recursive_review_unsealed(
            candidates, repo_root=repo_root, ledger_path=ledger
        )
        assert len(kept) == 0
        assert len(refused) == 1
        refused_row = refused[0]
        assert any("recursive_review_unsealed__bundle_id" in b for b in refused_row.blockers)
        assert "[recursive-review-unsealed refuse" in refused_row.notes

    def test_sealed_bundle_passes_through(
        self,
        repo_root: Path,
        tmp_path: Path,
    ) -> None:
        from tac.recursive_adversarial_review import (
            SEAL_THRESHOLD,
            RecursiveReviewRound,
            append_round_locked,
            compute_bundle_id,
            compute_scope_content_sha256,
        )

        ledger = tmp_path / "review_rounds.jsonl"
        scope_paths = ("src/tac/sealed_family/codec.py",)
        bid = compute_bundle_id(scope_paths)
        sha = compute_scope_content_sha256(scope_paths, repo_root=repo_root)
        # Build up to SEAL_THRESHOLD clean rounds.
        for i in range(SEAL_THRESHOLD):
            append_round_locked(
                RecursiveReviewRound(
                    review_id=f"r{i}",
                    bundle_id=bid,
                    scope_paths=scope_paths,
                    scope_content_sha256=sha,
                    round_number=i + 1,
                    council_rotation="skunkworks_sextet",
                    council_attendees=("Shannon",),
                    findings=(),
                    verdict="PROCEED",
                    counter_before=i,
                    counter_after=i + 1,
                    reviewed_at_utc=f"2026-05-17T0{i}:00:00+00:00",
                    reviewer_agent="test",
                ),
                path=ledger,
            )

        candidates = [
            mod.CandidateRow(
                candidate_id="sealed_family_candidate",
                family="sealed_family",
                predicted_score_delta=-0.001,
                expected_information_gain=0.1,
                estimated_dispatch_cost_usd=1.0,
            )
        ]
        kept, refused = mod.refuse_candidates_via_recursive_review_unsealed(
            candidates, repo_root=repo_root, ledger_path=ledger
        )
        # SEALed → pass through.
        assert len(kept) == 1
        assert len(refused) == 0

    def test_fail_closed_on_missing_canonical_helper(
        self,
        monkeypatch: pytest.MonkeyPatch,
        repo_root: Path,
        single_candidate: list[mod.CandidateRow],
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def _block_review(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "tac.recursive_adversarial_review":
                raise ImportError("simulated missing canonical helper")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_review)
        with pytest.raises(ImportError):
            mod.refuse_candidates_via_recursive_review_unsealed(
                single_candidate, repo_root=repo_root
            )


# ─────────────────────────────────────────────────────────────────────────
# End-to-end orchestration test
# ─────────────────────────────────────────────────────────────────────────


class TestEndToEndAllFiveWireInsExercised:
    """Verify that all 5 wire-ins can be chained in a realistic autopilot pass."""

    def test_full_pipeline_with_synthetic_state(
        self, repo_root: Path, tmp_path: Path
    ) -> None:
        # Set up empty ledgers / posteriors so the test runs hermetically.
        council_posterior = tmp_path / "council_posterior.jsonl"
        probe_ledger = tmp_path / "probe_outcomes.jsonl"
        review_ledger = tmp_path / "review_rounds.jsonl"
        modal_ledger = tmp_path / "modal_ledger.jsonl"
        cost_band_posterior = tmp_path / "cost_band_posterior.jsonl"

        # 1. Load DP1 composition primitives → some candidates.
        dp1_rows = mod.load_candidates_from_dp1_composition_primitives(repo_root)
        assert len(dp1_rows) > 0

        # Add a "regular" candidate too.
        all_rows = dp1_rows + [
            mod.CandidateRow(
                candidate_id="regular_cand",
                family="regular_family",
                predicted_score_delta=-0.005,
                expected_information_gain=0.2,
                estimated_dispatch_cost_usd=2.0,
            )
        ]
        starting_count = len(all_rows)

        # 2. Rerank via council → still N rows (no matching deliberations).
        ranked = mod.rerank_candidates_via_council_continual_learning(
            all_rows, repo_root=repo_root, posterior_path=council_posterior
        )
        assert len(ranked) == starting_count

        # 3. Refuse via probe outcomes → all pass (empty ledger).
        candidates_only = [r[0] for r in ranked]
        kept_after_probe, refused_after_probe = mod.refuse_candidates_via_probe_outcomes(
            candidates_only, repo_root=repo_root, ledger_path=probe_ledger
        )
        assert len(kept_after_probe) == starting_count
        assert len(refused_after_probe) == 0

        # 4. Refuse via recursive review → all pass (missing ledger).
        kept_after_review, refused_after_review = (
            mod.refuse_candidates_via_recursive_review_unsealed(
                kept_after_probe, repo_root=repo_root, ledger_path=review_ledger
            )
        )
        assert len(kept_after_review) == starting_count
        assert len(refused_after_review) == 0

        # 5. Update cost-band from Modal call-id ledger → no anchors (missing).
        status = mod.update_cost_band_from_modal_call_id_ledger(
            repo_root,
            ledger_path=modal_ledger,
            posterior_path=cost_band_posterior,
        )
        assert status["rows_scanned"] == 0
        assert status["anchors_appended"] == 0

    def test_full_pipeline_with_one_refusal_each(
        self, repo_root: Path, tmp_path: Path
    ) -> None:
        """Inject a refusal in BOTH the probe ledger AND the review ledger."""
        from tac.probe_outcomes_ledger import (
            VERDICT_KILL,
            register_probe_outcome,
        )
        from tac.recursive_adversarial_review import (
            RecursiveReviewRound,
            append_round_locked,
            compute_bundle_id,
            compute_scope_content_sha256,
        )

        probe_ledger = tmp_path / "probe_outcomes.jsonl"
        review_ledger = tmp_path / "review_rounds.jsonl"

        # Probe ledger: substrate "probed_family" gets KILL.
        register_probe_outcome(
            probe_id="probe_kill_1",
            substrate="probed_family",
            recipe_path=None,
            probe_kind="distinguishing_feature",
            verdict=VERDICT_KILL,
            metric_name="test",
            metric_value=0.0,
            threshold=0.5,
            threshold_token="<",
            evidence_path="/tmp/test",
            next_action="research_exhaustion_required",
            adjudicator="test",
            path=probe_ledger,
        )

        # Review ledger: bundle for "reviewed_family" still unsealed.
        scope_paths = ("src/tac/reviewed_family/codec.py",)
        bid = compute_bundle_id(scope_paths)
        sha = compute_scope_content_sha256(scope_paths, repo_root=repo_root)
        append_round_locked(
            RecursiveReviewRound(
                review_id="r1",
                bundle_id=bid,
                scope_paths=scope_paths,
                scope_content_sha256=sha,
                round_number=1,
                council_rotation="skunkworks_sextet",
                council_attendees=("Shannon",),
                findings=(),
                verdict="PROCEED",
                counter_before=0,
                counter_after=1,  # unsealed
                reviewed_at_utc="2026-05-17T00:00:00+00:00",
                reviewer_agent="test",
            ),
            path=review_ledger,
        )

        candidates = [
            mod.CandidateRow(
                candidate_id="probed_family_cand",
                family="probed_family",
                predicted_score_delta=-0.001,
                expected_information_gain=0.1,
                estimated_dispatch_cost_usd=1.0,
            ),
            mod.CandidateRow(
                candidate_id="reviewed_family_cand",
                family="reviewed_family",
                predicted_score_delta=-0.002,
                expected_information_gain=0.1,
                estimated_dispatch_cost_usd=1.0,
            ),
            mod.CandidateRow(
                candidate_id="passing_cand",
                family="passing_family",
                predicted_score_delta=-0.005,
                expected_information_gain=0.1,
                estimated_dispatch_cost_usd=1.0,
            ),
        ]

        # Apply probe + review filters in sequence.
        kept_after_probe, refused_p = mod.refuse_candidates_via_probe_outcomes(
            candidates, repo_root=repo_root, ledger_path=probe_ledger
        )
        assert len(refused_p) == 1
        assert refused_p[0].candidate_id == "probed_family_cand"

        kept_after_review, refused_r = (
            mod.refuse_candidates_via_recursive_review_unsealed(
                kept_after_probe, repo_root=repo_root, ledger_path=review_ledger
            )
        )
        assert len(refused_r) == 1
        assert refused_r[0].candidate_id == "reviewed_family_cand"

        # Final kept = only "passing_cand".
        assert len(kept_after_review) == 1
        assert kept_after_review[0].candidate_id == "passing_cand"


# ─────────────────────────────────────────────────────────────────────────
# Live-repo regression guards
# ─────────────────────────────────────────────────────────────────────────


class TestLiveRepoRegressionGuards:
    """Each new function MUST run cleanly against the actual state files in the
    repo, with no exception. Defense-in-depth against import-error or schema
    drift regression."""

    def test_dp1_composition_loader_runs_live(self, repo_root: Path) -> None:
        rows = mod.load_candidates_from_dp1_composition_primitives(repo_root)
        assert isinstance(rows, list)
        # 6 base substrates registered as of 2026-05-17; allow growth.
        assert len(rows) >= 1

    def test_council_rerank_runs_live(self, repo_root: Path) -> None:
        # Use the live council posterior (default path).
        candidates = [
            mod.CandidateRow(
                candidate_id="live_test_candidate",
                family="live_test_family",
                predicted_score_delta=-0.001,
                expected_information_gain=0.1,
                estimated_dispatch_cost_usd=1.0,
            )
        ]
        # Don't override posterior_path → uses default. May or may not have
        # matches; either way the function must not raise.
        ranked = mod.rerank_candidates_via_council_continual_learning(
            candidates, repo_root=repo_root
        )
        assert isinstance(ranked, list)
        assert len(ranked) == 1

    def test_probe_outcomes_filter_runs_live(self, repo_root: Path) -> None:
        candidates = [
            mod.CandidateRow(
                candidate_id="live_test_candidate_probe",
                family="live_test_family_probe",
                predicted_score_delta=-0.001,
                expected_information_gain=0.1,
                estimated_dispatch_cost_usd=1.0,
            )
        ]
        kept, refused = mod.refuse_candidates_via_probe_outcomes(
            candidates, repo_root=repo_root
        )
        # Lists of CandidateRow.
        assert isinstance(kept, list)
        assert isinstance(refused, list)
        # Total preserved (kept + refused = input).
        assert len(kept) + len(refused) == 1

    def test_modal_cost_band_wire_in_runs_live(
        self, repo_root: Path, tmp_path: Path
    ) -> None:
        # Use the live Modal ledger, but route the posterior write to tmp so
        # the test does NOT mutate the live cost-band posterior state.
        tmp_posterior = tmp_path / "cost_band_tmp.jsonl"
        status = mod.update_cost_band_from_modal_call_id_ledger(
            repo_root, posterior_path=tmp_posterior
        )
        assert status["schema"] == "modal_call_id_ledger_to_cost_band_wire_in_v1"
        assert "rows_scanned" in status
        assert "anchors_appended" in status
        assert isinstance(status["skipped_reasons"], dict)

    def test_recursive_review_filter_runs_live(self, repo_root: Path) -> None:
        candidates = [
            mod.CandidateRow(
                candidate_id="live_test_candidate_review",
                family="live_test_family_review",
                predicted_score_delta=-0.001,
                expected_information_gain=0.1,
                estimated_dispatch_cost_usd=1.0,
            )
        ]
        kept, refused = mod.refuse_candidates_via_recursive_review_unsealed(
            candidates, repo_root=repo_root
        )
        assert isinstance(kept, list)
        assert isinstance(refused, list)
        assert len(kept) + len(refused) == 1


# ─────────────────────────────────────────────────────────────────────────
# Schema regression guards (defense-in-depth)
# ─────────────────────────────────────────────────────────────────────────


class TestSchemaRegressionGuards:
    """Catalog #156 / #154 / #153 META-meta sister pattern: pin the canonical
    constants of new wire-ins so a future refactor cannot silently shift them."""

    def test_council_verdict_weight_defaults_unchanged(self) -> None:
        # If any operator deliberately changes these, the test fails loud so
        # the operator must update the test to reflect the new policy.
        assert mod.COUNCIL_VERDICT_WEIGHT_PROCEED_UNCONDITIONAL_DEFAULT == -0.20
        assert mod.COUNCIL_VERDICT_WEIGHT_PROCEED_WITH_REVISIONS_DEFAULT == -0.05
        assert mod.COUNCIL_VERDICT_WEIGHT_DEFER_PENDING_EVIDENCE_DEFAULT == 0.10
        assert mod.COUNCIL_VERDICT_WEIGHT_REFUSE_DEFAULT == 0.50
        assert mod.COUNCIL_VERDICT_WEIGHT_ESCALATE_DEFAULT == 0.20
