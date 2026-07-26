from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import measure_g1_teacher_atom_census as census_module
from tools.measure_g1_teacher_atom_census import (
    DEFAULT_N600_RECEIPT,
    DEFAULT_OUTPUT,
    DEFAULT_PBR2,
    DEFAULT_PBR2_RECEIPT,
    DEFAULT_PROGRAM,
    EXPECTED_PBR2_SHA256,
    EXPECTED_PROGRAM_SHA256,
    FAMILY_NAMES,
    SCHEMA,
    CensusError,
    canonical_json_bytes,
    derive_exclusive_ownership,
    make_receipt_envelope,
    measure_teacher_atoms,
    sha256_file,
    validate_frozen_teacher_contract,
    validate_receipt,
    write_once_receipt,
)


def _fixture() -> tuple[np.ndarray, tuple[np.ndarray, ...], list[dict[str, object]]]:
    predictor = np.zeros((3, 3, 4), dtype=np.uint8)
    target = predictor.copy()
    target[0, 1, 1:3] = 1
    target[0, 2, 3] = 2
    target[1] = target[0]

    temporal = predictor.copy()
    temporal[1] = target[1]
    islands = temporal.copy()
    islands[0, 1, 1:3] = 1
    tail = target.copy()
    counts = (3, 2, 1)
    rows: list[dict[str, object]] = []
    errors = sum(counts)
    for index, (name, count) in enumerate(zip(FAMILY_NAMES, counts, strict=True)):
        raw = f"raw-{name}".encode()
        payload = f"payload-{name}".encode()
        rows.append(
            {
                "name": name,
                "order": index + 1,
                "codec": "raw",
                "record_count": count,
                "span_count": count,
                "corrected_cells": count,
                "raw_bytes": len(raw),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "payload_bytes": len(payload),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "errors_before": errors,
                "errors_after": errors - count,
            }
        )
        errors -= count
    return predictor, (temporal, islands, tail), rows


def _valid_frozen_header() -> dict[str, object]:
    return {
        "target_semantic_lineage": "frozen_gt_argmax",
        "pbr2_reconstructs_exact_gt_argmax": True,
        "pbr2_is_target_derived": True,
        "target_derived_residual_promotion_admitted": False,
        "research_only": True,
        "candidate_archive_admissible": False,
        "exact_target_semantic_reconstruction": True,
        "score_claim": False,
        "promotion_eligible": False,
        "decode_scorer_dependency": False,
    }


def _valid_materialization_receipt() -> dict[str, object]:
    return {
        "candidate_payload_allowed": False,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "receiver_closure": {"candidate_payload_allowed": False},
    }


def test_synthetic_exhaustive_target_and_candidate_admission_are_rejected() -> None:
    synthetic = _valid_frozen_header()
    synthetic["target_semantic_lineage"] = "synthetic_fixture"
    synthetic["pbr2_reconstructs_exact_gt_argmax"] = False
    with pytest.raises(CensusError, match="frozen_gt_argmax"):
        validate_frozen_teacher_contract(synthetic, _valid_materialization_receipt())

    admitted = _valid_materialization_receipt()
    admitted["candidate_payload_allowed"] = True
    with pytest.raises(CensusError, match="candidate prohibition"):
        validate_frozen_teacher_contract(_valid_frozen_header(), admitted)

    body = {
        "schema": SCHEMA,
        "research_only": True,
        "candidate_payload_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "target_labels": [0, 1, 2],
    }
    with pytest.raises(CensusError, match="exhaustive teacher payload"):
        make_receipt_envelope(body)


def test_ownership_is_disjoint_complete_and_rejects_rewrites() -> None:
    predictor, stages, _rows = _fixture()
    target, masks = derive_exclusive_ownership(predictor, stages)
    correction = predictor != target
    assert np.array_equal(np.logical_or.reduce(masks), correction)
    assert sum(int(np.count_nonzero(mask)) for mask in masks) == int(np.count_nonzero(correction))
    for left in range(len(masks)):
        for right in range(left + 1, len(masks)):
            assert not np.any(masks[left] & masks[right])

    rewriting = [value.copy() for value in stages]
    rewriting[1][1, 1, 1] = 0
    with pytest.raises(CensusError, match="non-final teacher value"):
        derive_exclusive_ownership(predictor, tuple(rewriting))


def test_measurement_is_deterministic_and_closes_exact_family_debt() -> None:
    predictor, stages, rows = _fixture()
    first = measure_teacher_atoms(predictor, stages, rows, source_pair_start=448)
    second = measure_teacher_atoms(predictor, stages, rows, source_pair_start=448)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["predictor_debt_cells"] == 6
    assert first["exclusive_ownership_closed"] is True
    assert first["pairwise_owner_intersection_cells"] == 0
    assert [row["teacher_owned_cells"] for row in first["families"]] == [3, 2, 1]
    assert all(row["candidate_admissible_owned_cells"] == 0 for row in first["families"])
    assert all(row["score_value_per_byte"] == "unmeasured" for row in first["families"])
    assert first["temporal_support"]["motion_aligned_repeat_coverage"] == "unmeasured"
    assert first["palette_gauge_value_structure"]["rgb_palette_coverage"] == "unmeasured"
    assert first["candidate_admissible_remaining_debt_cells"] == 6


def test_receipt_body_hash_is_exact_and_detects_mutation() -> None:
    receipt = json.loads(DEFAULT_OUTPUT.read_text())
    assert receipt["body_sha256"] == hashlib.sha256(canonical_json_bytes(receipt["body"])).hexdigest()
    validate_receipt(receipt)

    mutated = copy.deepcopy(receipt)
    mutated["body"]["purpose"] = "mutated"
    with pytest.raises(CensusError, match="body SHA-256"):
        validate_receipt(mutated)


@pytest.mark.parametrize(
    "mutation",
    [
        "ownership",
        "authority",
        "candidate_atoms",
        "payload_alias",
        "n600_crosslink",
        "v14_row",
        "input_custody",
        "renderer_manifest",
    ],
)
def test_recomputed_envelope_cannot_relax_arithmetic_authority_or_lineage(mutation: str) -> None:
    receipt = json.loads(DEFAULT_OUTPUT.read_text())
    body = copy.deepcopy(receipt["body"])
    if mutation == "ownership":
        body["measurement"]["families"][0]["teacher_owned_cells"] += 1
    elif mutation == "authority":
        body["authority_axis"] = "[contest-CPU exact score]"
    elif mutation == "candidate_atoms":
        body["remaining_debt"]["candidate_admissible_atom_count"] = 1
    elif mutation == "payload_alias":
        body["teacher_payload_base64"] = "AA=="
    elif mutation == "n600_crosslink":
        body["full_n600_teacher_grammar_crosslink"]["evidence_rows"] = 0
    elif mutation == "v14_row":
        body["v14_exact_anchor_dispositions"]["rows"][0]["delta_d_seg"] = "999"
    elif mutation == "input_custody":
        body["inputs"]["n600_partition_grammar_receipt"]["sha256"] = "0" * 64
    else:
        body["inputs"]["predictor_renderer_source_manifest"]["schema"] = "evil"
    recomputed = {
        "schema": receipt["schema"],
        "body": body,
        "body_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }
    with pytest.raises(CensusError):
        validate_receipt(recomputed)


def test_write_once_receipt_does_not_clobber_a_concurrent_valid_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = json.loads(DEFAULT_OUTPUT.read_text())
    other_body = copy.deepcopy(receipt["body"])
    other_body["git_head"] = "d" * 40
    other = make_receipt_envelope(other_body)
    other_payload = canonical_json_bytes(other) + b"\n"
    target = tmp_path / "receipt.json"
    real_link = census_module.os.link

    def racing_link(source: str | Path, destination: str | Path) -> None:
        Path(destination).write_bytes(other_payload)
        real_link(source, destination)

    monkeypatch.setattr(census_module.os, "link", racing_link)
    with pytest.raises(CensusError, match="concurrent receipt differs"):
        write_once_receipt(target, receipt, reopen_sources=False)
    assert target.read_bytes() == other_payload


def test_existing_receipt_is_stable_reopened_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = json.loads(DEFAULT_OUTPUT.read_text())
    payload = canonical_json_bytes(receipt) + b"\n"
    target = tmp_path / "receipt.json"
    target.write_bytes(payload)
    real_read = census_module.read_stable_bytes
    reads = 0

    def racing_read(path: Path):
        nonlocal reads
        result = real_read(path)
        if Path(path) == target and reads == 0:
            target.write_bytes(b"peer-replaced-after-compare")
        reads += 1
        return result

    monkeypatch.setattr(census_module, "read_stable_bytes", racing_read)
    with pytest.raises(CensusError, match="changed during validation"):
        write_once_receipt(target, receipt, reopen_sources=False)


def test_sealed_program_pbr2_and_n600_receipt_hashes_are_exact() -> None:
    assert sha256_file(DEFAULT_PROGRAM) == EXPECTED_PROGRAM_SHA256
    assert sha256_file(DEFAULT_PBR2) == EXPECTED_PBR2_SHA256

    materialization = json.loads(DEFAULT_PBR2_RECEIPT.read_text())
    assert materialization["pbr2"]["packet_sha256"] == EXPECTED_PBR2_SHA256
    assert materialization["candidate_payload_allowed"] is False
    n600 = json.loads(DEFAULT_N600_RECEIPT.read_text())
    assert n600["body_sha256"] == hashlib.sha256(canonical_json_bytes(n600["body"])).hexdigest()
    assert n600["body"]["input_custody"]["sha256"] == materialization["inputs"]["gt_cache"]["sha256"]
