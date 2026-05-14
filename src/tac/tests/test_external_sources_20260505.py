# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.external_sources_20260505 import (
    FORBIDDEN_CONTEST_ACTIONS,
    SCHEMA_VERSION,
    all_lanes,
    all_sources,
    registry_payload,
    source_by_key,
    validate_lane,
)

EXPECTED_SOURCE_KEYS = [
    "la_pose",
    "ma_gig",
    "graph_lottery_ticket",
    "manifold_learning_survey",
    "flowm",
    "goodfire_vpd",
    "architecture_warmup",
    "multiplicative_gaussian_input",
    "cauchynet",
]

EXPECTED_LANE_KEYS = [
    "lapose_posenet_target_distillation",
    "lapose_hnerv_latent_conditioning",
    "lapose_motion_atom_sparsifier",
]


def test_sources_keep_lapose_as_dominant_focus() -> None:
    sources = all_sources()

    assert [source.key for source in sources] == EXPECTED_SOURCE_KEYS
    assert sources[0].key == "la_pose"
    assert sources[0].priority == "primary"
    assert "No public model weights or implementation" in sources[0].off_the_shelf_status


def test_lane_recommendations_are_contest_safe() -> None:
    for lane in all_lanes():
        validate_lane(lane)
        assert lane.score_claim is False
        assert lane.remote_dispatch_allowed is False
        assert set(lane.forbidden_actions) == FORBIDDEN_CONTEST_ACTIONS
        assert "exact_cuda_archive_eval" in {gate.name for gate in lane.evidence_gates}
        assert any("No scorer loads at inflate time" in note for note in lane.contest_safety_notes) or any(
            "planning signals only" in note for note in lane.contest_safety_notes
        ) or any("byte-accounted" in note for note in lane.contest_safety_notes)


def test_registry_payload_schema_and_stable_ordering() -> None:
    payload = registry_payload()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["registry"] == "external_sources_20260505"
    assert [source["key"] for source in payload["sources"]] == EXPECTED_SOURCE_KEYS
    assert [lane["key"] for lane in payload["lanes"]] == EXPECTED_LANE_KEYS


def test_lanes_reference_known_sources_and_prioritize_lapose() -> None:
    source_keys = {source.key for source in all_sources()}
    lanes = all_lanes()

    assert lanes[0].source_keys[0] == "la_pose"
    assert lanes[1].source_keys[0] == "la_pose"
    assert lanes[2].source_keys[0] == "la_pose"
    for lane in lanes:
        assert set(lane.source_keys) <= source_keys


def test_unknown_source_key_fails_closed() -> None:
    with pytest.raises(KeyError):
        source_by_key("missing")
