# SPDX-License-Identifier: MIT
"""Tests for the typed DOC_DIVERGENCE_RECEIPT writer (rv17 wave-2 F1).

The contract under test is an ORDERING claim as much as a validation claim: an invalid
record must leave the receipts directory **byte-identical**.  Every refusal test therefore
snapshots the directory before and asserts it after, rather than only asserting that an
exception was raised -- "it raised" and "it wrote nothing" are two different facts, and
only the second one is the cure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from comma_lab.packet_receipts import (
    RECEIPT_SCHEMA,
    DivergedFileEntry,
    DocDivergenceReceipt,
    FrozenOnlyDocEntry,
    ReceiptSchemaError,
    RepoOnlyDocEntry,
    check_publish_source_declared,
    main,
    next_receipt_name,
    receipt_rank,
    serialize_receipt,
    validate_receipt_mapping,
    write_receipt,
)

GEN6_RECEIPTS = Path(
    "/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen6_receipts"
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir()) if p.is_file()}


def _valid_receipt(**overrides) -> DocDivergenceReceipt:
    kwargs: dict = {
        "date_utc": "2026-08-21",
        "author": "MAIN",
        "reason": "test receipt: the packet-swap boundary append",
        "diverged_files": {
            "MANIFEST.sha256": DivergedFileEntry(
                repo_final_sha256=SHA_A,
                frozen_gen6_sha256=SHA_B,
                publish_source="prep",
                note="publish prep: header carries cures whose only copy is prep-side",
            )
        },
        "repo_only_docs": {
            "SWAP_PROCEDURE.md": RepoOnlyDocEntry(repo_final_sha256=SHA_A, note="prep only")
        },
        "frozen_only_docs": {
            "README.md": FrozenOnlyDocEntry(frozen_gen6_sha256=SHA_B, note="frozen only")
        },
        "supplements": "DOC_DIVERGENCE_RECEIPT_R16.json",
    }
    kwargs.update(overrides)
    return DocDivergenceReceipt(**kwargs)


# ---------------------------------------------------------------------------
# The happy path: a valid record writes, and writes stably.
# ---------------------------------------------------------------------------


def test_valid_receipt_writes_and_reparses(tmp_path):
    (tmp_path / "DOC_DIVERGENCE_RECEIPT_R16.json").write_text("{}")
    written = write_receipt(_valid_receipt(), tmp_path)
    assert written.name == "DOC_DIVERGENCE_RECEIPT_R17.json"
    payload = json.loads(written.read_text())
    assert payload["schema"] == RECEIPT_SCHEMA
    assert payload["diverged_files"]["MANIFEST.sha256"]["publish_source"] == "prep"
    round_tripped = validate_receipt_mapping(payload)
    assert round_tripped.to_dict() == payload


def test_serialization_is_byte_stable(tmp_path):
    receipt = _valid_receipt()
    first = serialize_receipt(receipt)
    assert serialize_receipt(receipt) == first
    a = write_receipt(receipt, tmp_path, name="DOC_DIVERGENCE_RECEIPT_R20.json")
    b_dir = tmp_path / "second"
    b_dir.mkdir()
    b = write_receipt(receipt, b_dir, name="DOC_DIVERGENCE_RECEIPT_R20.json")
    assert a.read_bytes() == b.read_bytes() == first.encode()


def test_optional_sections_are_omitted_not_nulled(tmp_path):
    receipt = DocDivergenceReceipt(
        date_utc="2026-08-21",
        author="MAIN",
        reason="minimal",
        diverged_files={
            "x.md": DivergedFileEntry(repo_final_sha256=SHA_A, frozen_gen6_sha256=SHA_A)
        },
    )
    payload = json.loads(serialize_receipt(receipt))
    for absent in ("repo_only_docs", "frozen_only_docs", "review_lineage", "supplements"):
        assert absent not in payload
    # An identical pair needs no publish_source (nothing to choose between).
    assert "publish_source" not in payload["diverged_files"]["x.md"]


def test_next_receipt_name_and_rank():
    assert receipt_rank("DOC_DIVERGENCE_RECEIPT.json") == 3
    assert receipt_rank("DOC_DIVERGENCE_RECEIPT_R16.json") == 16
    assert receipt_rank("MANIFEST.sha256") is None
    assert receipt_rank("DOC_DIVERGENCE_RECEIPT_RX.json") is None


def test_next_receipt_name_follows_the_chain_head(tmp_path):
    assert next_receipt_name(tmp_path) == "DOC_DIVERGENCE_RECEIPT_R4.json"
    (tmp_path / "DOC_DIVERGENCE_RECEIPT.json").write_text("{}")
    (tmp_path / "DOC_DIVERGENCE_RECEIPT_R9.json").write_text("{}")
    (tmp_path / "unrelated.txt").write_text("x")
    assert next_receipt_name(tmp_path) == "DOC_DIVERGENCE_RECEIPT_R10.json"


# ---------------------------------------------------------------------------
# The cure: an invalid record refuses and leaves NO bytes behind.
# ---------------------------------------------------------------------------


def test_the_r15_defect_is_unrepresentable():
    """A one-element list note -- the exact R15 trailing-comma slip -- cannot be built."""
    with pytest.raises(ReceiptSchemaError, match="trailing-comma"):
        RepoOnlyDocEntry(repo_final_sha256=SHA_A, note=["a note that slipped a comma"])
    with pytest.raises(ReceiptSchemaError, match="trailing-comma"):
        DivergedFileEntry(
            repo_final_sha256=SHA_A, frozen_gen6_sha256=SHA_A, note=["slipped"]
        )
    with pytest.raises(ReceiptSchemaError, match="trailing-comma"):
        FrozenOnlyDocEntry(frozen_gen6_sha256=SHA_A, note=["slipped"])


def test_r15_style_payload_is_refused_on_parse():
    payload = json.loads(serialize_receipt(_valid_receipt()))
    payload["repo_only_docs"]["SWAP_PROCEDURE.md"]["note"] = ["listed by mistake"]
    with pytest.raises(ReceiptSchemaError, match="expected a string, got list"):
        validate_receipt_mapping(payload)


def test_refusal_writes_nothing(tmp_path):
    """The ordering claim: validation happens BEFORE any disk contact."""
    (tmp_path / "DOC_DIVERGENCE_RECEIPT_R16.json").write_text("{}")
    before = _snapshot(tmp_path)

    bad = _valid_receipt(
        diverged_files={
            "MANIFEST.sha256": DivergedFileEntry(
                repo_final_sha256=SHA_A, frozen_gen6_sha256=SHA_B
            )
        }
    )
    with pytest.raises(ReceiptSchemaError, match="publish_source"):
        write_receipt(bad, tmp_path)

    assert _snapshot(tmp_path) == before
    assert not (tmp_path / "DOC_DIVERGENCE_RECEIPT_R17.json").exists()


def test_bad_date_and_bad_sha_are_refused():
    with pytest.raises(ReceiptSchemaError, match="YYYY-MM-DD"):
        _valid_receipt(date_utc="21-08-2026")
    with pytest.raises(ReceiptSchemaError, match="sha256"):
        DivergedFileEntry(repo_final_sha256="deadbeef", frozen_gen6_sha256=SHA_A)
    with pytest.raises(ReceiptSchemaError, match="sha256"):
        DivergedFileEntry(repo_final_sha256=SHA_A.upper(), frozen_gen6_sha256=SHA_A)


def test_empty_author_or_reason_is_refused():
    with pytest.raises(ReceiptSchemaError, match="author"):
        _valid_receipt(author="   ")
    with pytest.raises(ReceiptSchemaError, match="reason"):
        _valid_receipt(reason="")


def test_receipt_tracking_zero_documents_is_refused():
    with pytest.raises(ReceiptSchemaError, match="ZERO documents"):
        DocDivergenceReceipt(
            date_utc="2026-08-21", author="MAIN", reason="empty", diverged_files={}
        )


def test_unknown_fields_are_refused():
    payload = json.loads(serialize_receipt(_valid_receipt()))
    payload["smuggled_field"] = "x"
    with pytest.raises(ReceiptSchemaError, match="unknown top-level field"):
        validate_receipt_mapping(payload)

    payload = json.loads(serialize_receipt(_valid_receipt()))
    payload["diverged_files"]["MANIFEST.sha256"]["extra"] = "x"
    with pytest.raises(ReceiptSchemaError, match="unknown field"):
        validate_receipt_mapping(payload)


def test_bad_publish_source_enum_is_refused():
    with pytest.raises(ReceiptSchemaError, match="must be one of"):
        DivergedFileEntry(
            repo_final_sha256=SHA_A, frozen_gen6_sha256=SHA_B, publish_source="either"
        )


def test_a_section_given_as_a_list_is_refused():
    """A JSON list of pairs survives dict(); the container type is guarded before mapping."""
    payload = json.loads(serialize_receipt(_valid_receipt()))
    payload["repo_only_docs"] = [["SWAP_PROCEDURE.md", {"repo_final_sha256": SHA_A}]]
    with pytest.raises(ReceiptSchemaError, match="expected an object, got list"):
        validate_receipt_mapping(payload)


def test_a_bool_note_is_refused_like_a_list():
    with pytest.raises(ReceiptSchemaError, match="expected a string, got bool"):
        RepoOnlyDocEntry(repo_final_sha256=SHA_A, note=True)


def test_review_lineage_must_be_a_sequence_not_a_string():
    payload = json.loads(serialize_receipt(_valid_receipt(review_lineage=("r1", "r2"))))
    assert payload["review_lineage"] == ["r1", "r2"]
    payload["review_lineage"] = "r1"
    with pytest.raises(ReceiptSchemaError, match="got a string"):
        validate_receipt_mapping(payload)


@pytest.mark.parametrize("bad", [5, {"a": "b"}, None])
def test_sequence_fields_reject_non_sequences(bad):
    """A dict would tuple() into its keys and an int would raise TypeError, not our error."""
    payload = json.loads(serialize_receipt(_valid_receipt()))
    payload["corrections_applied"] = bad
    with pytest.raises(ReceiptSchemaError, match="expected a list of strings"):
        validate_receipt_mapping(payload)


def test_serialize_proves_full_round_trip_reconstruction():
    """Parsing is the weak claim; reconstructing the identical payload is the strong one."""
    receipt = _valid_receipt(
        review_lineage=("round 1", "round 2"),
        known_defect_in_predecessor="a recorded predecessor defect",
    )
    payload = json.loads(serialize_receipt(receipt))
    assert validate_receipt_mapping(payload).to_dict() == payload
    # Every optional field that was set survives the trip.
    for key in ("supplements", "review_lineage", "known_defect_in_predecessor"):
        assert key in payload


# ---------------------------------------------------------------------------
# Append-only.
# ---------------------------------------------------------------------------


def test_existing_receipt_is_never_overwritten(tmp_path):
    target = tmp_path / "DOC_DIVERGENCE_RECEIPT_R17.json"
    target.write_text('{"original": true}')
    before = _snapshot(tmp_path)
    with pytest.raises(ReceiptSchemaError, match="append-only"):
        write_receipt(_valid_receipt(), tmp_path, name=target.name)
    assert _snapshot(tmp_path) == before


def test_non_advancing_rank_is_refused(tmp_path):
    (tmp_path / "DOC_DIVERGENCE_RECEIPT_R16.json").write_text("{}")
    before = _snapshot(tmp_path)
    with pytest.raises(ReceiptSchemaError, match="append-only"):
        write_receipt(_valid_receipt(), tmp_path, name="DOC_DIVERGENCE_RECEIPT_R5.json")
    assert _snapshot(tmp_path) == before


def test_non_receipt_filename_is_refused(tmp_path):
    with pytest.raises(ReceiptSchemaError, match="not a receipt filename"):
        write_receipt(_valid_receipt(), tmp_path, name="receipt.json")


def test_missing_directory_is_refused(tmp_path):
    with pytest.raises(ReceiptSchemaError, match="not a directory"):
        write_receipt(_valid_receipt(), tmp_path / "nope")


# ---------------------------------------------------------------------------
# The R11-F2 rule is a WRITE-time policy, not a parse-time invariant.
# ---------------------------------------------------------------------------


def test_publish_source_rule_refuses_forward_but_parses_history():
    undeclared = DocDivergenceReceipt(
        date_utc="2026-08-19",
        author="MAIN",
        reason="a pre-R11 receipt, before the rule existed",
        diverged_files={
            "a.md": DivergedFileEntry(repo_final_sha256=SHA_A, frozen_gen6_sha256=SHA_B)
        },
    )
    # Constructing/parsing a historical row is fine ...
    assert undeclared.diverged_files["a.md"].copies_differ
    # ... but writing a NEW one is refused.
    with pytest.raises(ReceiptSchemaError, match="R11-F2"):
        check_publish_source_declared(undeclared)
    with pytest.raises(ReceiptSchemaError, match="R11-F2"):
        serialize_receipt(undeclared)


# ---------------------------------------------------------------------------
# The executed control against the 14 REAL receipts (schema derived, not invented).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not GEN6_RECEIPTS.is_dir(), reason="SSD tier not mounted")
def test_schema_parses_the_real_chain_and_catches_only_r15():
    results = {}
    for path in sorted(GEN6_RECEIPTS.iterdir()):
        if receipt_rank(path.name) is None:
            continue
        try:
            validate_receipt_mapping(json.loads(path.read_text()))
        except ReceiptSchemaError as exc:
            results[path.name] = str(exc)
        else:
            results[path.name] = None
    assert len(results) == 14, results.keys()
    failures = {name: err for name, err in results.items() if err}
    # R15 is the ONE refusal, and it is exactly the defect R16's own
    # known_defect_in_predecessor field records.
    assert set(failures) == {"DOC_DIVERGENCE_RECEIPT_R15.json"}, failures
    assert "expected a string, got list" in failures["DOC_DIVERGENCE_RECEIPT_R15.json"]


@pytest.mark.skipif(not GEN6_RECEIPTS.is_dir(), reason="SSD tier not mounted")
def test_cli_check_reports_the_real_chain(capsys):
    rc = main(["--check", str(GEN6_RECEIPTS)])
    out = capsys.readouterr()
    assert rc == 1  # R15 fails, by design
    assert "13/14 receipts parse" in out.out
    assert "DOC_DIVERGENCE_RECEIPT_R15.json" in out.err


def test_cli_refuses_an_empty_target(tmp_path, capsys):
    assert main(["--check", str(tmp_path)]) == 1
    assert "vacuity guard" in capsys.readouterr().err
