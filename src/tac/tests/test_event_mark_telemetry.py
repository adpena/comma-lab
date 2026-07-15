from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tac.causal_manifest import (
    NON_PROMOTABLE_AXIS,
    ArtifactRef,
    CausalManifestConflictError,
    CausalManifestError,
    CausalManifestWriter,
    ClassEdgeMark,
    DigestRef,
    EventMarkRow,
    IncidenceMark,
    ReceiverStateMark,
    SpacetimeMark,
    StratumMark,
    freeze_fields,
    load_causal_manifest,
    row_from_dict,
)
from tac.witness_control.telemetry_producers import (
    MarkedEventTelemetryProducer,
    ProducerResumeState,
)

STAMP = "2026-07-15T09:00:00Z"
SHA_A = "a" * 64


def _artifact(role: str) -> ArtifactRef:
    return ArtifactRef(role, f"evidence/{role}", DigestRef("sha256", SHA_A))


def _event(
    family: str = "topology",
    kind: str = "component_birth",
    detectors: tuple[str, ...] = ("topology:birth_detector",),
) -> EventMarkRow:
    return EventMarkRow.build(
        run_id="run-a",
        stage_id="tau",
        checkpoint_id="ep100",
        pair_index=7,
        frame_from=14,
        frame_to=15,
        observed_at_utc=STAMP,
        family=family,
        kind=kind,
        detectors_matched=detectors,
        class_edge=ClassEdgeMark(1, 0, True),
        location=SpacetimeMark(
            "scorer_grid",
            3.0,
            4.0,
            1.0,
            2.0,
            3.0,
            5.0,
            6.0,
            "seg384",
            "round_half_even_q8",
        ),
        attachment=IncidenceMark(
            before_component_ids=("lane:0",),
            after_component_ids=("lane:0", "lane:1"),
            parent_child_edges=(("lane:0", "lane:1"),),
            attachment_rule_id="four_connected_components_v1",
        ),
        stratum_before=StratumMark("k0", "omega0", "hosc:4", "phase:0"),
        stratum_after=StratumMark("k1", "omega0", "hosc:4", "phase:0"),
        receiver_state=ReceiverStateMark(
            "canonical_R_v1",
            "uint8_half_even_v1",
            "xi_q2_v1",
            (1, 2),
            0,
            "segnet_argmax_v1",
        ),
        evidence=(_artifact("detector"),),
        receiver_derivable=False,
        public_derivation_ref=None,
        notes={"event_count_authority": False},
    )


@pytest.mark.parametrize(
    ("family", "kind", "detector"),
    [
        ("topology", "component_birth", "topology:birth_detector"),
        ("chart", "atlas_transition", "chart:atlas_detector"),
        (
            "receiver_lattice",
            "uint8_rounding_crossing",
            "receiver_lattice:uint8_detector",
        ),
    ],
)
def test_strict_round_trip_for_each_family(
    family: str, kind: str, detector: str
) -> None:
    row = _event(family, kind, (detector,))
    assert EventMarkRow.from_dict(row.to_dict()) == row
    assert row.authority_axis == NON_PROMOTABLE_AXIS


def test_reject_family_kind_mismatch() -> None:
    with pytest.raises(CausalManifestError, match="not valid for family"):
        _event("topology", "atlas_transition")


def test_reject_class_only_or_count_only_marks() -> None:
    with pytest.raises(CausalManifestError, match="distinct classes"):
        ClassEdgeMark(1, 1, True)
    with pytest.raises(CausalManifestError, match="count-only"):
        IncidenceMark(attachment_rule_id="event_count_only")


def test_priority_partition_records_overlapping_detectors_once(tmp_path: Path) -> None:
    producer = MarkedEventTelemetryProducer(
        CausalManifestWriter(tmp_path / "manifest.jsonl", "run-a"),
        ProducerResumeState(),
    )
    base = _event()
    row, appended = producer.record(
        stage_id=base.stage_id,
        checkpoint_id=base.checkpoint_id,
        pair_index=base.pair_index,
        frame_from=base.frame_from,
        frame_to=base.frame_to,
        observed_at_utc=base.observed_at_utc,
        detector_matches={
            "receiver_lattice": ("uint8_detector",),
            "chart": ("atlas_detector",),
            "topology": ("birth_detector",),
        },
        kind_by_family={
            "topology": "component_birth",
            "chart": "atlas_transition",
            "receiver_lattice": "uint8_rounding_crossing",
        },
        class_edge=base.class_edge,
        location=base.location,
        attachment=base.attachment,
        stratum_before=base.stratum_before,
        stratum_after=base.stratum_after,
        receiver_state=base.receiver_state,
        evidence=base.evidence,
        receiver_derivable=False,
        public_derivation_ref=None,
    )
    assert appended
    assert row.family == "topology"
    assert row.detectors_matched == (
        "chart:atlas_detector",
        "receiver_lattice:uint8_detector",
        "topology:birth_detector",
    )
    assert len(load_causal_manifest(producer.writer.path)) == 1


def test_stable_event_id_ignores_mapping_insertion_order() -> None:
    row = _event()
    reversed_notes = {"z": 1, "a": 2}
    forward_notes = {"a": 2, "z": 1}
    first = EventMarkRow.build(
        **{
            **{
                name: getattr(row, name)
                for name in (
                    "run_id",
                    "stage_id",
                    "checkpoint_id",
                    "pair_index",
                    "frame_from",
                    "frame_to",
                    "observed_at_utc",
                    "family",
                    "kind",
                    "detectors_matched",
                    "class_edge",
                    "location",
                    "attachment",
                    "stratum_before",
                    "stratum_after",
                    "receiver_state",
                    "evidence",
                    "receiver_derivable",
                    "public_derivation_ref",
                )
            },
            "notes": reversed_notes,
        }
    )
    second = EventMarkRow.build(
        **{
            **{
                name: getattr(row, name)
                for name in (
                    "run_id",
                    "stage_id",
                    "checkpoint_id",
                    "pair_index",
                    "frame_from",
                    "frame_to",
                    "observed_at_utc",
                    "family",
                    "kind",
                    "detectors_matched",
                    "class_edge",
                    "location",
                    "attachment",
                    "stratum_before",
                    "stratum_after",
                    "receiver_state",
                    "evidence",
                    "receiver_derivable",
                    "public_derivation_ref",
                )
            },
            "notes": forward_notes,
        }
    )
    assert first.event_id == second.event_id
    assert first.to_dict() == second.to_dict()


def test_identical_resume_append_is_idempotent_and_cursor_persists(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    row = _event()
    writer = CausalManifestWriter(path, "run-a")
    assert writer.record_event_mark(row)
    assert not CausalManifestWriter(path, "run-a").record_event_mark(row)

    state = ProducerResumeState(event_mark_resume_keys={row.resume_key})
    arrays = state.state_arrays("__dtp_")
    restored = ProducerResumeState()
    assert restored.restore_from_cfg("__dtp_", arrays)
    assert restored.event_mark_resume_keys == {row.resume_key}


def test_conflicting_immutable_event_id_fails_closed(tmp_path: Path) -> None:
    writer = CausalManifestWriter(tmp_path / "manifest.jsonl", "run-a")
    row = _event()
    assert writer.record_event_mark(row)
    changed = replace(row, notes=freeze_fields({"changed": True}))
    with pytest.raises(CausalManifestConflictError, match="different content"):
        writer.record_event_mark(changed)


def test_old_v1_row_still_parses_byte_for_byte() -> None:
    raw = {
        "schema_id": "pact.causal_manifest.v1",
        "row_kind": "coverage_receipt",
        "row_id": "coverage:old",
        "receipt_id": "coverage:old",
        "run_id": "old",
        "target_policy_id": "policy",
        "target_policy_sha256": SHA_A,
        "working_support_id": "support",
        "initial_state_covered": True,
        "one_step_target_covered": False,
        "action_support": [],
        "assessment_method": "historical",
        "evidence": ["receipt"],
        "verdict_scope": "INSTANCE",
        "emitted_at_utc": STAMP,
    }
    assert row_from_dict(raw).to_dict() == raw


def test_observability_axis_fixed_and_score_fields_rejected() -> None:
    row = _event()
    with pytest.raises(CausalManifestError, match="authority_axis"):
        replace(row, authority_axis="[contest-CUDA]")
    raw = row.to_dict()
    raw["score_claim"] = True
    with pytest.raises(CausalManifestError, match="score/promotion"):
        row_from_dict(raw)


def test_event_rejects_noncanonical_utc() -> None:
    row = _event()
    with pytest.raises(CausalManifestError, match="canonical UTC"):
        replace(row, observed_at_utc="2026-07-15T09:00:00+00:00")
