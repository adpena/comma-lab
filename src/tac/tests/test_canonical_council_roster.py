# SPDX-License-Identifier: MIT
"""Tests for tac.canonical_council_roster — anti-recurrence for under-rostering.

Covers:
- CouncilSeat dataclass invariants (frozen, validation)
- INNER_COUNCIL canonical set (11 seats: 6 sextet + 5 additional)
- GRAND_COUNCIL canonical set (20 seats: 12 existing + 8 new 2026-05-15)
- required_attendees_for_topic across T1/T2/T3/T4
- validate_council_dispatch_roster with complete / under-rostered fixtures
- Slot 20 anchor: under-rostered T3 with 18/29 attendees flagged
- Slot 20-supplemental anchor: 21/29 still under-rostered for PP topic
- Slot 21 anchor: complete T3 roster passes
"""
from __future__ import annotations

import pytest

from tac.canonical_council_roster import (
    CouncilSeat,
    GRAND_COUNCIL,
    INNER_COUNCIL,
    RosterValidationVerdict,
    required_attendees_for_topic,
    validate_council_dispatch_roster,
)


class TestCouncilSeat:
    def test_frozen(self) -> None:
        seat = INNER_COUNCIL[0]
        with pytest.raises((AttributeError, Exception)):
            seat.name = "Other"  # type: ignore[misc]

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(ValueError, match="role"):
            CouncilSeat(
                name="X", role="bogus",  # type: ignore[arg-type]
                canonical_position_summary="summary",
                relevance_tokens=("any",),
                canonical_reference_path="CLAUDE.md",
            )

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name"):
            CouncilSeat(
                name="", role="inner_council",
                canonical_position_summary="summary",
                relevance_tokens=("any",),
                canonical_reference_path="CLAUDE.md",
            )

    def test_empty_summary_rejected(self) -> None:
        with pytest.raises(ValueError, match="canonical_position_summary"):
            CouncilSeat(
                name="X", role="inner_council",
                canonical_position_summary="",
                relevance_tokens=("any",),
                canonical_reference_path="CLAUDE.md",
            )

    def test_relevance_tokens_must_be_tuple(self) -> None:
        with pytest.raises(ValueError, match="relevance_tokens"):
            CouncilSeat(
                name="X", role="inner_council",
                canonical_position_summary="summary",
                relevance_tokens=["any"],  # type: ignore[arg-type]
                canonical_reference_path="CLAUDE.md",
            )

    def test_canonical_reference_path_required(self) -> None:
        with pytest.raises(ValueError, match="canonical_reference_path"):
            CouncilSeat(
                name="X", role="inner_council",
                canonical_position_summary="summary",
                relevance_tokens=("any",),
                canonical_reference_path="",
            )


class TestInnerCouncilRoster:
    def test_inner_council_has_14_seats(self) -> None:
        # 6 sextet pact + 5 additional + PR95Author (2026-05-19) + Rudin +
        # Daubechies (2026-05-19 ROSTER-MAINTENANCE-V2 4-co-lead structure)
        # = 14 mandatory.
        assert len(INNER_COUNCIL) == 14

    def test_sextet_pact_complete(self) -> None:
        sextet_names = frozenset(
            s.name for s in INNER_COUNCIL
            if s.role == "inner_council_sextet"
        )
        assert sextet_names == frozenset(
            ["Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian",
             "Assumption-Adversary"]
        )

    def test_inner_additional_complete(self) -> None:
        additional = frozenset(
            s.name for s in INNER_COUNCIL
            if s.role == "inner_council"
        )
        # Per CLAUDE.md "Experiment design — non-negotiable" + operator 2026-05-19
        # PR95Author addition + 2026-05-19 ROSTER-MAINTENANCE-V2 (Rudin +
        # Daubechies co-leads): Quantizr / Hotz / Selfcomp / MacKay / Balle /
        # PR95Author / Rudin / Daubechies are mandatory inner-council.
        assert additional == frozenset(
            ["Quantizr", "Hotz", "Selfcomp", "MacKay", "Balle", "PR95Author",
             "Rudin", "Daubechies"]
        )

    def test_pr95_author_present(self) -> None:
        # Per operator 2026-05-19 verbatim "the PR 95 author has been added to
        # the inner council as well"
        names = frozenset(s.name for s in INNER_COUNCIL)
        assert "PR95Author" in names

    def test_shannon_is_lead(self) -> None:
        shannon = next(s for s in INNER_COUNCIL if s.name == "Shannon")
        assert "LEAD" in shannon.canonical_position_summary

    def test_dykstra_is_co_lead(self) -> None:
        dykstra = next(s for s in INNER_COUNCIL if s.name == "Dykstra")
        assert "CO-LEAD" in dykstra.canonical_position_summary

    def test_assumption_adversary_present(self) -> None:
        # Per CLAUDE.md "Council conduct" Fix-7 amendment.
        names = frozenset(s.name for s in INNER_COUNCIL)
        assert "Assumption-Adversary" in names

    def test_inner_seat_unique_names(self) -> None:
        names = [s.name for s in INNER_COUNCIL]
        assert len(names) == len(frozenset(names))

    def test_rudin_present_as_inner_co_lead(self) -> None:
        # Per operator 2026-05-19 verbatim "rudin and debauchies should still
        # be on the inner council, they co-lead with shannon and dykstra now"
        names = frozenset(s.name for s in INNER_COUNCIL)
        assert "Rudin" in names
        rudin = next(s for s in INNER_COUNCIL if s.name == "Rudin")
        assert rudin.is_co_lead is True

    def test_daubechies_present_as_inner_co_lead(self) -> None:
        # Per operator 2026-05-19 verbatim "rudin and debauchies should still
        # be on the inner council, they co-lead with shannon and dykstra now"
        names = frozenset(s.name for s in INNER_COUNCIL)
        assert "Daubechies" in names
        daubechies = next(s for s in INNER_COUNCIL if s.name == "Daubechies")
        assert daubechies.is_co_lead is True


class TestFourCoLeadStructure:
    """Per CLAUDE.md 'Council conduct' 2026-05-19 amendment: Shannon LEAD +
    Dykstra CO-LEAD + Rudin CO-LEAD + Daubechies CO-LEAD form the inner
    council shared-leadership core. ALL 4 are required at every T2+ deliberation.
    """

    def test_exactly_four_co_leads(self) -> None:
        co_leads = [s for s in INNER_COUNCIL if s.is_co_lead]
        assert len(co_leads) == 4

    def test_co_leads_are_shannon_dykstra_rudin_daubechies(self) -> None:
        co_lead_names = frozenset(
            s.name for s in INNER_COUNCIL if s.is_co_lead
        )
        assert co_lead_names == frozenset(
            ["Shannon", "Dykstra", "Rudin", "Daubechies"]
        )

    def test_sister_inner_members_not_co_leads(self) -> None:
        # All non-co-lead inner-council members provide domain-specific
        # perspectives, not shared-leadership-core decisions.
        non_co_lead_names = frozenset(
            s.name for s in INNER_COUNCIL if not s.is_co_lead
        )
        # Sister members: 10 voices (6 sextet minus Shannon+Dykstra = 4) +
        # (8 additional minus Rudin+Daubechies = 6) = 10 sister members.
        assert non_co_lead_names == frozenset([
            "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary",
            "Quantizr", "Hotz", "Selfcomp", "MacKay", "Balle", "PR95Author",
        ])

    def test_co_lead_field_default_false(self) -> None:
        # Backward-compat: CouncilSeat constructed without is_co_lead defaults
        # to False (not a co-lead).
        seat = CouncilSeat(
            name="X", role="inner_council",
            canonical_position_summary="test",
            relevance_tokens=("any",),
            canonical_reference_path="CLAUDE.md",
        )
        assert seat.is_co_lead is False

    def test_grand_council_seat_cannot_be_co_lead(self) -> None:
        # Per __post_init__: co-leads MUST be inner-council seats.
        with pytest.raises(ValueError, match="is_co_lead=True requires inner_council"):
            CouncilSeat(
                name="X", role="grand_council",
                is_co_lead=True,
                canonical_position_summary="test",
                relevance_tokens=("any",),
                canonical_reference_path="CLAUDE.md",
            )

    def test_is_co_lead_must_be_bool(self) -> None:
        with pytest.raises(ValueError, match="is_co_lead must be bool"):
            CouncilSeat(
                name="X", role="inner_council",
                is_co_lead="yes",  # type: ignore[arg-type]
                canonical_position_summary="test",
                relevance_tokens=("any",),
                canonical_reference_path="CLAUDE.md",
            )


class TestGrandCouncilRoster:
    def test_grand_council_has_22_seats(self) -> None:
        # 11 existing (since 2026-04-29; Schmidhuber listed once) +
        # 8 new (2026-05-15 expansion: Atick/Redlich/Rao/Ballard/Tishby/
        # Zaslavsky/Wyner/TimeTravelerProtege) +
        # 1 operator-initiated 2026-05-19 (TimeTraveler mentor reframe) +
        # 2 operator-initiated 2026-05-19 ROSTER-MAINTENANCE-V2 sister seats
        # (Rudin_Grand + Daubechies_Grand per Catalog #110 APPEND-ONLY
        # coexistence with inner-council co-leads) = 22.
        assert len(GRAND_COUNCIL) == 22

    def test_existing_11_seats_present(self) -> None:
        names = frozenset(s.name for s in GRAND_COUNCIL)
        for required in [
            "Boyd", "Tao", "Filler", "Mallat", "vdOord", "Carmack",
            "Hassabis", "Hinton", "Karpathy", "Schmidhuber",
            "JackFromSkunkworks",
        ]:
            assert required in names, f"{required} missing from grand council"

    def test_new_8_seats_present(self) -> None:
        names = frozenset(s.name for s in GRAND_COUNCIL)
        for required in [
            "Atick", "Redlich", "Rao", "Ballard", "Tishby", "Zaslavsky",
            "Wyner", "TimeTravelerProtege",
        ]:
            assert required in names, f"{required} missing from grand council"

    def test_time_traveler_mentor_seat_present(self) -> None:
        # Per operator 2026-05-19 verbatim "the time traveler is a mysterious
        # figure from the future whose identity has not been revealed yet..."
        names = frozenset(s.name for s in GRAND_COUNCIL)
        assert "TimeTraveler" in names

    def test_grand_seat_unique_names(self) -> None:
        names = [s.name for s in GRAND_COUNCIL]
        assert len(names) == len(frozenset(names))

    def test_rudin_grand_sister_seat_present(self) -> None:
        # Per Catalog #110 APPEND-ONLY: Rudin INNER_COUNCIL co-lead seat coexists
        # with Rudin_Grand sister seat (different name to avoid duplicate-name
        # collision; same canonical attendee in two rosters).
        names = frozenset(s.name for s in GRAND_COUNCIL)
        assert "Rudin_Grand" in names

    def test_daubechies_grand_sister_seat_present(self) -> None:
        # Per Catalog #110 APPEND-ONLY: Daubechies INNER_COUNCIL co-lead seat
        # coexists with Daubechies_Grand sister seat.
        names = frozenset(s.name for s in GRAND_COUNCIL)
        assert "Daubechies_Grand" in names


class TestRequiredAttendeesForTopic:
    def test_t1_returns_empty(self) -> None:
        # T1 working group has no mandatory roster requirement
        assert required_attendees_for_topic(["any"], "T1") == ()

    def test_t2_returns_all_inner(self) -> None:
        result = required_attendees_for_topic(["pp_integration"], "T2")
        # T2 requires all inner_council seats
        assert len(result) == len(INNER_COUNCIL)
        names = frozenset(s.name for s in result)
        assert names == frozenset(s.name for s in INNER_COUNCIL)

    def test_t3_returns_inner_plus_topical_grand(self) -> None:
        # PP integration topic should match Boyd / Tao / MacKay / Tishby / Hinton
        # / Karpathy / Hassabis / Schmidhuber / Atick / Wyner / Carmack / Balle
        # plus Filler / Mallat (info_gain + partition_discovery)
        result = required_attendees_for_topic(
            ["pp_integration", "info_gain", "partition_discovery"],
            "T3",
        )
        names = frozenset(s.name for s in result)
        # All inner
        for inner in INNER_COUNCIL:
            assert inner.name in names
        # Topically relevant grand
        for required_grand in ["Boyd", "Tao", "Filler", "Mallat", "Hassabis"]:
            assert required_grand in names, f"{required_grand} required for PP topic"

    def test_t4_returns_all(self) -> None:
        result = required_attendees_for_topic(["any"], "T4")
        assert len(result) == len(INNER_COUNCIL) + len(GRAND_COUNCIL)

    def test_invalid_tier_rejected(self) -> None:
        with pytest.raises(ValueError, match="council_tier"):
            required_attendees_for_topic(["any"], "T5")

    def test_t3_with_specialized_topic_narrows(self) -> None:
        # Only embodied_cognition + cooperative_receiver narrows to Atick/Redlich/Ballard/Rao
        result = required_attendees_for_topic(
            ["embodied_cognition", "cooperative_receiver"], "T3",
        )
        grand_names = frozenset(
            s.name for s in result if s.role == "grand_council"
        )
        assert "Ballard" in grand_names
        assert "Atick" in grand_names
        assert "Redlich" in grand_names
        # vdOord should NOT be there (VQ-VAE not related)
        assert "vdOord" not in grand_names


class TestValidateCouncilDispatchRoster:
    def test_complete_t2_passes(self) -> None:
        # Dispatch ALL inner-council members for a T2 deliberation
        attendees = [s.name for s in INNER_COUNCIL]
        verdict = validate_council_dispatch_roster(
            attendees, ["pp_integration"], "T2",
        )
        assert verdict.complete is True
        assert verdict.missing_inner_council == ()

    def test_missing_inner_council_flagged(self) -> None:
        # Drop Quantizr + Hotz from T2 dispatch — this is the slot 20 bug class
        attendees = [
            s.name for s in INNER_COUNCIL
            if s.name not in ("Quantizr", "Hotz")
        ]
        verdict = validate_council_dispatch_roster(
            attendees, ["pp_integration"], "T2",
        )
        assert verdict.complete is False
        assert "Quantizr" in verdict.missing_inner_council
        assert "Hotz" in verdict.missing_inner_council

    def test_slot_20_bug_class_anchor(self) -> None:
        """Empirical anchor: slot 20 dispatched 18/29 (missing 11).

        Updated for ROSTER-MAINTENANCE-V2 (2026-05-19): slot 20 also omitted
        the 2 new co-leads (Rudin + Daubechies) which are now mandatory at T2+.
        """
        # Slot 20 had: 6 sextet + 12 grand council = 18 (missing 4 inner + 7 grand
        # + 2 co-leads under new 4-co-lead structure)
        slot_20_attendees = [
            # 6 sextet
            "Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian",
            "Assumption-Adversary",
            # 1 of 5 additional inner (MacKay)
            "MacKay",
            # 11 of 12 existing grand (Filler / Mallat / Carmack / Karpathy /
            # JackFromSkunkworks omitted)
            "Boyd", "Tao", "vdOord", "Hassabis", "Hinton", "Schmidhuber",
            # 6 of 8 new grand (Atick + Redlich omitted)
            "Tishby", "Zaslavsky", "Wyner", "Rao", "Ballard",
        ]
        verdict = validate_council_dispatch_roster(
            slot_20_attendees,
            ["pp_integration", "info_gain", "partition_discovery"],
            "T3",
        )
        # Inner-council omissions: Quantizr, Hotz, Selfcomp, Balle, PR95Author,
        # Rudin, Daubechies. Plus 2 co-lead omissions (Rudin, Daubechies).
        assert verdict.complete is False
        assert "Quantizr" in verdict.missing_inner_council
        assert "Hotz" in verdict.missing_inner_council
        assert "Selfcomp" in verdict.missing_inner_council
        assert "Balle" in verdict.missing_inner_council
        # Co-lead omissions are STRUCTURALLY DISTINCT per 2026-05-19 amendment.
        assert "Rudin" in verdict.missing_co_leads
        assert "Daubechies" in verdict.missing_co_leads

    def test_missing_co_lead_blocks_t2_dispatch(self) -> None:
        """Per CLAUDE.md 'Council conduct' 2026-05-19 amendment: ALL 4 co-leads
        MUST be present at every T2+ deliberation. Missing ANY co-lead is
        structurally incomplete."""
        # Dispatch ALL inner-council EXCEPT Rudin (drop one co-lead)
        attendees = [
            s.name for s in INNER_COUNCIL if s.name != "Rudin"
        ]
        verdict = validate_council_dispatch_roster(
            attendees, ["pp_integration"], "T2",
        )
        assert verdict.complete is False
        assert "Rudin" in verdict.missing_co_leads
        assert "Rudin" in verdict.missing_inner_council

    def test_missing_daubechies_blocks_t2_dispatch(self) -> None:
        """Daubechies is the 4th co-lead per 2026-05-19 amendment."""
        attendees = [
            s.name for s in INNER_COUNCIL if s.name != "Daubechies"
        ]
        verdict = validate_council_dispatch_roster(
            attendees, ["pp_integration"], "T2",
        )
        assert verdict.complete is False
        assert "Daubechies" in verdict.missing_co_leads

    def test_all_four_co_leads_present_passes_t2(self) -> None:
        """Verify the 4-co-lead structure is satisfied when all 4 are present."""
        attendees = [s.name for s in INNER_COUNCIL]
        verdict = validate_council_dispatch_roster(
            attendees, ["pp_integration"], "T2",
        )
        assert verdict.complete is True
        assert verdict.missing_co_leads == ()

    def test_co_leads_missing_surfaced_in_render(self) -> None:
        """Verdict.render() surfaces missing co-leads as a distinct alert."""
        attendees = [
            s.name for s in INNER_COUNCIL
            if s.name not in ("Rudin", "Daubechies")
        ]
        verdict = validate_council_dispatch_roster(
            attendees, ["pp_integration"], "T2",
        )
        rendered = verdict.render()
        assert "MISSING_CO_LEADS" in rendered
        assert "Rudin" in rendered
        assert "Daubechies" in rendered
        # The render should also cite the 4-co-lead structure rationale
        assert "4-co-lead" in rendered or "Shannon" in rendered

    def test_slot_21_complete_t3_passes(self) -> None:
        """After this second supplemental: complete roster passes."""
        all_attendees = [s.name for s in INNER_COUNCIL] + [s.name for s in GRAND_COUNCIL]
        verdict = validate_council_dispatch_roster(
            all_attendees,
            ["pp_integration", "info_gain", "partition_discovery",
             "engineering_shortcuts", "neural_compression"],
            "T3",
        )
        assert verdict.complete is True

    def test_t3_grand_council_advisory_3_missing_ok(self) -> None:
        # T3 allows up to 4 missing grand-council seats per "advisory" rule
        attendees = (
            [s.name for s in INNER_COUNCIL]
            + [s.name for s in GRAND_COUNCIL[:-3]]  # omit last 3
        )
        verdict = validate_council_dispatch_roster(
            attendees, ["pp_integration"], "T3",
        )
        # Up to 4 missing grand acceptable; 3 missing is OK
        assert verdict.complete is True

    def test_t3_grand_council_5plus_missing_blocks(self) -> None:
        # T3 with 5+ missing topical grand seats is structural under-rostering.
        # Use a topic that maps to MANY grand-council seats so omitting many
        # produces a 5+ topical-grand miss.
        attendees = [s.name for s in INNER_COUNCIL] + ["Boyd"]  # only Boyd from grand
        verdict = validate_council_dispatch_roster(
            attendees,
            ["pp_integration", "info_gain", "partition_discovery",
             "neural_compression", "cooperative_receiver"],
            "T3",
        )
        assert verdict.complete is False
        assert len(verdict.missing_relevant_grand_council) >= 5

    def test_t4_requires_all_grand_council(self) -> None:
        attendees = [s.name for s in INNER_COUNCIL] + [s.name for s in GRAND_COUNCIL[:-1]]
        verdict = validate_council_dispatch_roster(
            attendees, ["any"], "T4",
        )
        # T4 needs ALL grand council; missing 1 is a violation
        assert verdict.complete is False

    def test_unknown_attendees_informational(self) -> None:
        all_inner = [s.name for s in INNER_COUNCIL]
        verdict = validate_council_dispatch_roster(
            all_inner + ["UnknownVoice"], ["pp_integration"], "T2",
        )
        # Complete (all inner present); unknown reported informational
        assert verdict.complete is True
        assert "UnknownVoice" in verdict.unknown_attendees

    def test_invalid_tier_rejected(self) -> None:
        with pytest.raises(ValueError, match="council_tier"):
            validate_council_dispatch_roster([], [], "T9")

    def test_render_human_readable(self) -> None:
        verdict = validate_council_dispatch_roster(
            ["Shannon"], ["pp_integration"], "T2",
        )
        rendered = verdict.render()
        assert "tier=T2" in rendered
        assert "MISSING_INNER_COUNCIL" in rendered
        assert "Quantizr" in rendered  # one of the missing names


class TestRosterValidationVerdictDataclass:
    def test_frozen(self) -> None:
        v = RosterValidationVerdict(
            complete=True,
            missing_inner_council=(),
            missing_relevant_grand_council=(),
            unknown_attendees=(),
            topic_tokens=("any",),
            council_tier="T2",
        )
        with pytest.raises((AttributeError, Exception)):
            v.complete = False  # type: ignore[misc]

    def test_missing_co_leads_field_default_empty(self) -> None:
        """Per 2026-05-19 amendment: missing_co_leads field default is ()."""
        v = RosterValidationVerdict(
            complete=True,
            missing_inner_council=(),
            missing_relevant_grand_council=(),
            unknown_attendees=(),
            topic_tokens=("any",),
            council_tier="T2",
        )
        assert v.missing_co_leads == ()

    def test_missing_co_leads_field_settable(self) -> None:
        """Verify the new missing_co_leads field is queryable."""
        v = RosterValidationVerdict(
            complete=False,
            missing_inner_council=("Rudin", "Daubechies"),
            missing_co_leads=("Rudin", "Daubechies"),
            missing_relevant_grand_council=(),
            unknown_attendees=(),
            topic_tokens=("pp_integration",),
            council_tier="T2",
        )
        assert v.missing_co_leads == ("Rudin", "Daubechies")
        assert v.complete is False


class TestEdgeCases:
    def test_empty_attendees_t2(self) -> None:
        verdict = validate_council_dispatch_roster([], ["pp_integration"], "T2")
        assert verdict.complete is False
        assert len(verdict.missing_inner_council) == len(INNER_COUNCIL)

    def test_whitespace_names_stripped(self) -> None:
        attendees = ["  Shannon  ", " Dykstra"]
        # Should be normalized; both should be recognized
        verdict = validate_council_dispatch_roster(
            attendees, ["pp_integration"], "T2",
        )
        # Inner council mostly missing, but Shannon + Dykstra recognized
        assert "Shannon" not in verdict.missing_inner_council
        assert "Dykstra" not in verdict.missing_inner_council

    def test_empty_topic_tokens_t3(self) -> None:
        # Empty topic = no topical grand match; inner still required
        attendees = [s.name for s in INNER_COUNCIL]
        verdict = validate_council_dispatch_roster(attendees, [], "T3")
        # No grand required (no topic match); complete
        assert verdict.complete is True

    def test_topic_tokens_case_insensitive(self) -> None:
        # Tokens normalized to lowercase
        result_upper = required_attendees_for_topic(["PP_INTEGRATION"], "T3")
        result_lower = required_attendees_for_topic(["pp_integration"], "T3")
        names_upper = frozenset(s.name for s in result_upper)
        names_lower = frozenset(s.name for s in result_lower)
        assert names_upper == names_lower
