# SPDX-License-Identifier: MIT
"""Focused tests for the strict PREDICT-to-PROJECT seed schema."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

import tac.optimization.predict_project_schema as predict_project_schema
from tac.optimization.predict_project_schema import (
    DIGEST_BYTES,
    PREFIX,
    SECTION_PREFIX,
    PredictProjectSchemaError,
    build_minimal_constraint_seed,
    parse_constraint_seed,
    serialize_constraint_seed,
    validate_constraint_seed,
)


def seed() -> dict:
    return build_minimal_constraint_seed(
        bytes([0, 1, 2, 3, 4, 0]),
        scorer_height=2,
        scorer_width=3,
        camera_height=4,
        camera_width=6,
        constraint_seeds=[
            {
                "time": 3,
                "frame_index": 1,
                "y": 0,
                "x": 1,
                "cell_id": 2,
                "predictor_status": "violated",
                "stratum": "boundary_codim1",
                "pose_tube": None,
                "projector": None,
            }
        ],
    )


def test_schema_round_trip_is_byte_canonical_and_hash_bound():
    value = seed()
    encoded = serialize_constraint_seed(value)
    assert serialize_constraint_seed(parse_constraint_seed(encoded)) == encoded
    assert parse_constraint_seed(encoded)["authority"]["score_claim"] is False
    assert parse_constraint_seed(encoded)["ground_chart"]["representation"] == "morse_smale_graph_vineyard.v1"
    assert (
        parse_constraint_seed(encoded)["ground_chart"]["derived_raster_fixture"]["derivation_id"]
        == "nearest_ground_cell_site_then_min_cell_id.v1"
    )


def test_parse_back_validates_constraint_seed_exactly_once(monkeypatch):
    encoded = serialize_constraint_seed(seed())
    original_validate = predict_project_schema.validate_constraint_seed
    invocation_count = 0

    def counting_validate(value):
        nonlocal invocation_count
        invocation_count += 1
        return original_validate(value)

    monkeypatch.setattr(predict_project_schema, "validate_constraint_seed", counting_validate)
    parsed = predict_project_schema.parse_constraint_seed(encoded)
    assert parsed["schema"] == predict_project_schema.SCHEMA_ID
    assert invocation_count == 1


def test_canonical_wire_contains_no_raster_bytes_or_payload_field():
    raw_raster = bytes([0, 1, 2, 3, 4, 0])
    encoded = serialize_constraint_seed(seed())
    assert raw_raster.hex().encode("ascii") not in encoded
    assert b'"payload_hex"' not in encoded
    assert b'"payload"' not in encoded
    fixture = parse_constraint_seed(encoded)["ground_chart"]["derived_raster_fixture"]
    assert set(fixture) == {"derivation_id", "content_sha256"}


def test_named_section_container_is_versioned_length_delimited_and_complete():
    encoded = serialize_constraint_seed(seed())
    magic, version, section_count = PREFIX.unpack_from(encoded)
    assert magic == b"PPCS1\x00"
    assert version == 1
    assert section_count == 12
    cursor = PREFIX.size
    names = []
    for _ in range(section_count):
        name_length, payload_length = SECTION_PREFIX.unpack_from(encoded, cursor)
        cursor += SECTION_PREFIX.size
        names.append(encoded[cursor : cursor + name_length].decode("ascii"))
        cursor += name_length + payload_length + DIGEST_BYTES
    assert names == seed()["container"]["section_order"]
    assert cursor == len(encoded) - DIGEST_BYTES


def test_schema_rejects_unknown_fields_and_bool_as_integer():
    value = seed()
    value["unknown"] = 1
    with pytest.raises(PredictProjectSchemaError, match="fields mismatch"):
        validate_constraint_seed(value)
    value = seed()
    value["receiver"]["pair_count"] = True
    with pytest.raises(PredictProjectSchemaError, match="exact integer"):
        validate_constraint_seed(value)


def test_schema_rejects_noncanonical_json_hash_drift_and_trailing_bytes():
    encoded = serialize_constraint_seed(seed())
    with pytest.raises(PredictProjectSchemaError, match=r"hash drift|trailing"):
        parse_constraint_seed(encoded + b"x")
    drifted = bytearray(encoded)
    drifted[PREFIX.size] ^= 1
    with pytest.raises(PredictProjectSchemaError, match="hash drift"):
        parse_constraint_seed(bytes(drifted))

    _, _, _ = PREFIX.unpack_from(encoded)
    cursor = PREFIX.size
    name_length, payload_length = SECTION_PREFIX.unpack_from(encoded, cursor)
    cursor += SECTION_PREFIX.size
    name = encoded[cursor : cursor + name_length]
    cursor += name_length
    manifest = json.loads(encoded[cursor : cursor + payload_length])
    noncanonical = json.dumps(manifest, sort_keys=False, indent=2).encode("ascii")
    first_section_end = cursor + payload_length + DIGEST_BYTES
    body = bytearray(encoded[: PREFIX.size])
    body.extend(SECTION_PREFIX.pack(name_length, len(noncanonical)))
    body.extend(name)
    body.extend(noncanonical)
    body.extend(hashlib.sha256(noncanonical).digest())
    body.extend(encoded[first_section_end:-DIGEST_BYTES])
    body.extend(hashlib.sha256(body).digest())
    payload = bytes(body)
    with pytest.raises(PredictProjectSchemaError, match="not canonical"):
        parse_constraint_seed(payload)


def test_schema_rejects_unsorted_duplicates_hidden_per_frame_and_hash_drift():
    value = seed()
    value["constraint_seeds"].append(copy.deepcopy(value["constraint_seeds"][0]))
    with pytest.raises(PredictProjectSchemaError, match="sorted and unique"):
        validate_constraint_seed(value)

    value = seed()
    value["trajectory"]["controls"] = [{"time": index, "tx_q": 0, "ty_q": 0, "yaw_q": 0} for index in range(600)]
    with pytest.raises(PredictProjectSchemaError, match=r"2\.\.599"):
        validate_constraint_seed(value)

    value = seed()
    value["ground_chart"]["derived_raster_fixture"]["content_sha256"] = "0" * 64
    with pytest.raises(PredictProjectSchemaError, match=r"SHA-256|hash"):
        validate_constraint_seed(value)


def test_schema_requires_causal_jitter_rung_and_fail_closed_authority():
    value = seed()
    value["boundary_jitter"]["selected_rung"] = None
    with pytest.raises(PredictProjectSchemaError, match="jitter ladder"):
        validate_constraint_seed(value)
    value = seed()
    value["authority"]["n600_claim"] = True
    with pytest.raises(PredictProjectSchemaError, match="fail-closed"):
        validate_constraint_seed(value)


def test_native_grammar_is_structural_canonical_and_payload_free():
    value = seed()
    grammar = value["grammar"]
    assert grammar["start_nonterminal"] == "SceneChart"
    assert grammar["nonterminal_order"][-1] == "LearnedTailGenerator"
    assert grammar["learned_tail"]["status"] == "ABSENT_DEFAULT"
    assert all(declaration["wire_payload"] is False for declaration in grammar["nonterminals"])
    encoded = serialize_constraint_seed(value)
    assert b'"scorer_bytes_allowed":false' in encoded
    assert b'"raster_payload_bytes_allowed":false' in encoded
    assert b'"counted_weight_bytes"' not in encoded


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda grammar: grammar["nonterminal_order"].reverse(), "nonterminals"),
        (
            lambda grammar: grammar["productions"][0]["rhs_nonterminals"].append("UnknownSymbol"),
            "production|nonterminal",
        ),
        (lambda grammar: grammar["arithmetic_contexts"].pop(), "context coverage"),
        (
            lambda grammar: grammar["productions"][0].__setitem__("interpreter_procedure_id", "unknown.proc"),
            "production|procedure",
        ),
        (
            lambda grammar: grammar["learned_tail"].__setitem__("status", "ADMITTED_BY_STRICT_THREE_WAY_RACE"),
            "race receipt|SHA-256",
        ),
    ],
)
def test_native_grammar_rejects_unknown_order_reference_context_and_unproved_learning(mutation, message):
    value = seed()
    mutation(value["grammar"])
    with pytest.raises(PredictProjectSchemaError, match=message):
        validate_constraint_seed(value)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["ground_chart"]["critical_points"][0].__setitem__("persistence_q", 0), "persistence"),
        (
            lambda value: value["ground_chart"]["separatrix_arcs"][0].__setitem__("target_critical_id", 9999),
            "endpoints",
        ),
        (
            lambda value: value["ground_chart"]["cells"][0]["adjacent_cell_ids"].pop(),
            "symmetric|cover cell adjacency",
        ),
        (
            lambda value: value["ground_chart"]["critical_points"][0].__setitem__(
                "critical_id", value["ground_chart"]["critical_points"][1]["critical_id"]
            ),
            "unique|order",
        ),
    ],
)
def test_morse_smale_graph_rejects_malformed_topology(mutation, message):
    value = seed()
    mutation(value)
    with pytest.raises(PredictProjectSchemaError, match=message):
        validate_constraint_seed(value)


def test_raw_raster_cannot_become_normative_and_section_order_is_sealed():
    value = seed()
    value["ground_chart"]["payload_hex"] = "00"
    with pytest.raises(PredictProjectSchemaError, match="fields mismatch"):
        validate_constraint_seed(value)


def test_vineyard_lifecycle_symbols_are_typed_sorted_and_reference_topology():
    value = seed()
    value["ground_chart"]["vineyard_events"] = [
        {
            "time": 4,
            "kind": "birth",
            "persistence_pair_id": 0,
            "critical_ids": [0, 1],
            "cell_ids": [0, 1],
        }
    ]
    validate_constraint_seed(value)
    value["ground_chart"]["vineyard_events"][0]["critical_ids"] = [0, 999]
    with pytest.raises(PredictProjectSchemaError, match="lifecycle references"):
        validate_constraint_seed(value)
    value = seed()
    value["container"]["section_order"] = list(reversed(value["container"]["section_order"]))
    with pytest.raises(PredictProjectSchemaError, match="container policy"):
        validate_constraint_seed(value)


def test_frame0_cannot_carry_seg_obligation():
    value = seed()
    row = value["constraint_seeds"][0]
    row["frame_index"] = 0
    row["obligation"] = "seg_and_pose"
    with pytest.raises(PredictProjectSchemaError, match="frame0 is pose-only"):
        validate_constraint_seed(value)
