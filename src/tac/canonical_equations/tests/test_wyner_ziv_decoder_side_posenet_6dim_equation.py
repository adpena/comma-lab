# SPDX-License-Identifier: MIT
"""Sister tests for the canonical Wyner-Ziv decoder-side PoseNet side-information equation.

These tests verify the predecessor-landed canonical equation
``wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1``
(registered 2026-05-30T16:04:07Z, Wave N+36, registry line 371) round-trips correctly
and carries canonical Provenance per Catalog #323. The completing landing
(`.omx/research/wyner_ziv_decoder_side_posenet_6dim_m6_side_info_wiring_gap_design_20260530.md`)
documents the M6 side-info wiring gap as a canonical operator-routable; this test suite
is the structural regression guard that the canonical equation row stays coherent
(Wyner-Ziv 1976 Theorem 1 + decoder-reproducibility + PoseNet pose side-info) and is NOT
silently mutated (Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE).

The tests read the canonical registry JSONL directly (stdlib only) so they do not depend
on a particular helper-module import surface — they verify the persisted canonical truth,
which is the contract every downstream consumer reads.

HONESTY NOTE (no fake / no over-fit): every assertion below was written against field
values VERIFIED from the actual persisted registry row at the time of authoring (not
assumed). The verified shape: `domain_of_validity` has top-level keys `excluded` +
`in_domain` (list-of-dicts each carrying `context_id`); the `in_domain[0]` entry carries
`side_info_kind = "posenet_output_per_pair_6_to_12_dim"` +
`decoder_side_posenet_reproducibility = True`; the equation already holds ONE
research-only synthetic-Gaussian paradigm anchor (evidence_grade=research_only,
measurement_axis=[predicted]) — it is NOT a fabricated contest-score anchor (Slot EEE
Class 3 distinguishes synthetic-fixture-with-honest-provenance from
synthetic-passed-off-as-real); the M6 wiring-gap distinction is encoded via the
`non_decoder_reproducible_substrates` excluded context + the
`side_info_kind=posenet_output...` in_domain context (the canonical Y is the PoseNet
output, NOT a generic spatial mean). The note carries the Wave N+36 paradigm-elevation
rationale.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

EQUATION_ID = (
    "wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1"
)


def _repo_root() -> Path:
    # src/tac/canonical_equations/tests/<this file> -> repo root is 4 parents up.
    return Path(__file__).resolve().parents[4]


def _registry_path() -> Path:
    return _repo_root() / ".omx" / "state" / "canonical_equations_registry.jsonl"


def _load_equation_events(equation_id: str) -> list[dict]:
    """Return all append-only registry events for an equation_id, in file order."""
    path = _registry_path()
    if not path.exists():
        pytest.skip(f"canonical equations registry not present at {path}")
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                # Skip an unrelated corrupt line rather than fail the whole suite.
                continue
            if row.get("equation_id") == equation_id:
                events.append(row)
    return events


def _latest_event(equation_id: str) -> dict:
    events = _load_equation_events(equation_id)
    assert events, (
        f"canonical equation {equation_id} not found in registry; expected the "
        "Wave N+36 predecessor landing at line 371 (2026-05-30T16:04:07Z)"
    )
    return events[-1]


def _latest_payload(equation_id: str) -> dict:
    return _latest_event(equation_id)["equation_payload"]


def _in_domain_contexts(dov: dict) -> list[dict]:
    return list(dov.get("in_domain", []))


def _excluded_context_ids(dov: dict) -> list[str]:
    return [e.get("context_id", "") for e in dov.get("excluded", [])]


def test_equation_exists_and_is_registered() -> None:
    events = _load_equation_events(EQUATION_ID)
    assert events, "predecessor-landed canonical equation row must exist"
    assert events[0].get("event_type") == "registered"


def test_equation_round_trips_canonical_payload_shape() -> None:
    payload = _latest_payload(EQUATION_ID)
    for key in (
        "equation_id",
        "name",
        "one_line_summary",
        "latex_form",
        "canonical_producers",
        "canonical_consumers",
        "domain_of_validity",
        "empirical_anchors",
        "provenance",
        "next_recalibration_trigger",
        "schema_version",
    ):
        assert key in payload, f"canonical equation payload missing key: {key}"
    assert payload["equation_id"] == EQUATION_ID


def test_equation_captures_wyner_ziv_theorem_1_conditional_rate() -> None:
    """Math-rigor (NO FAKE Class 1): the equation encodes R(D|Y) << R(D), not a placeholder.

    Verified against the actual persisted latex/name/summary:
    latex carries the R(D|Y) conditional rate-distortion infimum + Y=PoseNet(pair);
    name + summary reference the Wyner-Ziv decoder-side PoseNet conditional-entropy
    reduction.
    """
    payload = _latest_payload(EQUATION_ID)
    latex = payload["latex_form"]
    # Conditional rate-distortion with side-info Y at decoder: R(D|Y).
    assert "R(D|Y)" in latex or "R(D | Y)" in latex
    # Y is the PoseNet pair output (decoder-reproducible).
    assert "PoseNet" in latex or "posenet" in latex.lower()
    name = payload["name"].lower()
    summary = payload["one_line_summary"].lower()
    assert "wyner-ziv" in name or "wyner-ziv" in summary
    assert "posenet" in name or "posenet" in summary
    assert "decoder" in name or "decoder" in summary


def test_equation_in_domain_is_posenet_output_side_info() -> None:
    """The canonical Y is the PoseNet pose output, not a generic spatial statistic.

    Verified persisted shape: domain_of_validity.in_domain[0].side_info_kind ==
    'posenet_output_per_pair_6_to_12_dim' AND decoder_side_posenet_reproducibility True.
    This is the canonical claim the M6 wiring-gap operator-routable points at (the Z8 M6
    coder currently wires generic top-LL spatial mean instead of this canonical Y).
    """
    payload = _latest_payload(EQUATION_ID)
    dov = payload["domain_of_validity"]
    in_domain = _in_domain_contexts(dov)
    assert in_domain, "equation must declare at least one in_domain context"
    # The canonical Y is the per-pair PoseNet output (6-to-12 dim; first 6 scored).
    assert any(
        "posenet" in str(ctx.get("side_info_kind", "")).lower() for ctx in in_domain
    ), "in_domain side_info_kind must name the PoseNet output as the canonical Y"
    # PoseNet 6-dim is named in the canonical side-info kind taxonomy.
    assert any(
        "6" in str(ctx.get("side_info_kind", "")) for ctx in in_domain
    ), "canonical side_info_kind must reference the 6-dim PoseNet pose surface"
    # The FREE-side-info claim rests on decoder-side reproducibility.
    assert any(
        ctx.get("decoder_side_posenet_reproducibility") is True for ctx in in_domain
    ), "in_domain must assert decoder_side_posenet_reproducibility (the ZERO-byte Y claim)"


def test_equation_excludes_non_decoder_reproducible_and_degenerate_contexts() -> None:
    """The M6 wiring-gap distinction: non-decoder-reproducible Y is OUT of domain.

    Verified persisted excluded context_ids:
    posenet_as_source_degenerate, non_video_signals,
    non_decoder_reproducible_substrates, residual_hybrid_contexts_per_catalog_359.
    A generic top-LL spatial mean baked into the substrate (non-PoseNet) falls outside
    the canonical in_domain PoseNet-output Y; substrates whose decoder cannot reproduce
    the PoseNet Y are explicitly excluded (the canonical contract that protects the
    ZERO-byte FREE-side-info claim).
    """
    payload = _latest_payload(EQUATION_ID)
    excluded = _excluded_context_ids(payload["domain_of_validity"])
    # The Wyner-Ziv contract requires decoder-reproducible Y; substrates that cannot
    # reproduce it are excluded (this is the structural guard the M6 gap surfaces).
    assert "non_decoder_reproducible_substrates" in excluded
    # Degenerate case: X = PoseNet output itself (Y = X) is excluded (zero savings).
    assert "posenet_as_source_degenerate" in excluded


def test_equation_carries_canonical_provenance_catalog_323() -> None:
    payload = _latest_payload(EQUATION_ID)
    prov = payload["provenance"]
    # predicted_from_model: an equation predictor, not a contest-archive score claim.
    assert prov.get("artifact_kind") == "predicted_from_model"
    assert prov.get("score_claim_valid") is False
    assert prov.get("promotion_eligible") is False
    assert prov.get("measurement_axis") == "[predicted]"
    assert "build_provenance_for_predicted" in prov.get(
        "canonical_helper_invocation", ""
    )


def test_equation_anchor_is_research_only_predicted_not_contest_score_claim() -> None:
    """NO FAKE Class 3 + 4: the single anchor is honestly research-only/predicted.

    The equation holds ONE synthetic-Gaussian paradigm anchor (Wave N+36, Z8 M6 commit
    5d5634dd3). This is NOT a forbidden fake: its provenance is honest
    (evidence_grade=research_only, measurement_axis=[predicted], score_claim_valid=False)
    — Slot EEE Class 3 forbids synthetic-fixtures PASSED OFF AS real contest measurements,
    not honestly-tagged synthetic paradigm anchors. The real contest anchor that lifts
    this to a measured savings is the operator-routable WZ-M6-2 dispatch on actual
    upstream/videos/0.mkv pair frames. This test guards that the anchor stays honestly
    non-promotable and is not silently upgraded to a contest score claim.
    """
    payload = _latest_payload(EQUATION_ID)
    anchors = payload.get("empirical_anchors", [])
    assert anchors, "equation carries the Wave N+36 paradigm anchor"
    for anchor in anchors:
        prov = anchor.get("provenance", {})
        # Every anchor must be honestly non-promotable (no phantom contest score).
        assert prov.get("score_claim_valid") is False, (
            "anchor must NOT claim a contest score (research-only/predicted only)"
        )
        assert prov.get("promotion_eligible") is False
        assert prov.get("evidence_grade") in {
            "research_only",
            "predicted",
            "macos_cpu_advisory",
        }, f"anchor evidence_grade must be non-authoritative; got {prov.get('evidence_grade')!r}"
        assert prov.get("measurement_axis") in {
            "[predicted]",
            "[macOS-CPU advisory]",
            "MPS-research-signal",
        }, "anchor measurement_axis must be a non-contest axis"


def test_equation_note_records_wave_n36_paradigm_elevation_rationale() -> None:
    """The registration note must carry the canonical decoder-reproducibility rationale."""
    event = _latest_event(EQUATION_ID)
    notes = event.get("notes", "")
    assert "decoder-reproducible" in notes.lower() or "decoder reproducible" in notes.lower()
    assert "posenet" in notes.lower()


def test_equation_producers_name_canonical_wyner_ziv_codec_and_posenet() -> None:
    """Producers must include the canonical WZ codec module + the PoseNet scorer source.

    Verified persisted producers include 'tac.codec.wyner_ziv_layer:apply' (the canonical
    WZ codec) + 'upstream.modules.PoseNet' (the decoder-reproducible Y source) + the Z8
    M6 coder. This guards the producer chain that the M6 wiring-gap operator-routable
    (WZ-M6-1) rewires.
    """
    payload = _latest_payload(EQUATION_ID)
    producers = payload["canonical_producers"]
    assert any("wyner_ziv_layer" in p for p in producers), (
        "canonical producer must include the WZ codec module (tac.codec.wyner_ziv_layer)"
    )
    assert any("PoseNet" in p or "posenet" in p.lower() for p in producers), (
        "canonical producer must include the PoseNet scorer (the decoder-reproducible Y)"
    )


def test_equation_row_is_append_only_single_registered_event() -> None:
    """Catalog #110/#113: exactly one 'registered' event (no forbidden duplicate).

    A second 'registered' event for this equation_id would indicate a forbidden
    duplicate per CLAUDE.md anti-duplication primitive + Slot EEE Class 5. The completing
    landing this test ships alongside must NOT register a duplicate — it only adds the
    design memo + lane + probe + these tests around the existing APPEND-ONLY row.
    """
    events = _load_equation_events(EQUATION_ID)
    registered = [e for e in events if e.get("event_type") == "registered"]
    assert len(registered) == 1, (
        "exactly one 'registered' event expected; a second would be a forbidden "
        "duplicate per CLAUDE.md anti-duplication + Slot EEE Class 5"
    )
