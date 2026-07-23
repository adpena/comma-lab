# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.optimization import ddm_dv2_sdwl1 as sdwl1
from tac.optimization.arith_selfcomp_rate_coders import (
    decode_spatial_context_arithmetic,
    encode_spatial_context_arithmetic,
)
from tools import measure_ddm_dv2_sdwl1 as cli


@pytest.fixture
def source_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.array(
        [
            [
                [0, 0, 0, 1, 1, 2, 2],
                [0, 0, 1, 1, 2, 2, 2],
                [0, 3, 3, 1, 2, 4, 4],
                [0, 3, 3, 1, 4, 4, 4],
                [0, 0, 0, 1, 1, 4, 4],
                [0, 0, 0, 1, 1, 4, 4],
            ],
            [
                [0, 0, 0, 1, 1, 2, 2],
                [0, 0, 1, 1, 2, 2, 2],
                [0, 3, 3, 1, 2, 4, 4],
                [0, 3, 3, 1, 4, 4, 4],
                [0, 0, 0, 1, 1, 4, 4],
                [0, 0, 0, 1, 1, 4, 4],
            ],
            [
                [0, 0, 1, 1, 1, 2, 2],
                [0, 0, 1, 3, 2, 2, 2],
                [0, 3, 3, 3, 2, 4, 4],
                [0, 3, 0, 1, 4, 4, 4],
                [0, 0, 0, 1, 1, 4, 4],
                [0, 0, 0, 1, 1, 4, 4],
            ],
        ],
        dtype=np.int8,
    )
    margins = np.linspace(0.0, 1.5, labels.size, dtype=np.float32).reshape(labels.shape)
    poses = np.array(
        [
            [0.0, -0.0, 1.0, -1.0, 2.0, -2.0],
            [0.0, -0.0, 1.0, -1.0, 2.0, -2.0],
            [0.125, -0.25, 1.5, -1.0, 2.0, -3.0],
        ],
        dtype=np.float64,
    )
    return labels, margins, poses


@pytest.fixture
def inventory(
    source_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> sdwl1.FactInventory:
    return sdwl1.extract_fact_inventory(*source_arrays)


def _replace_section(packet: bytes, tag: bytes, payload: bytes) -> bytes:
    sections = sdwl1._parse_sections(packet)
    replaced = [
        (section_tag, payload if section_tag == tag else section_payload) for section_tag, section_payload in sections
    ]
    return sdwl1._frame_sections(replaced)


def _frame_unchecked(sections: list[tuple[bytes, bytes]]) -> bytes:
    body = b"".join(
        sdwl1._SECTION_HEADER.pack(tag, len(payload), hashlib.sha256(payload).digest()) + payload
        for tag, payload in sections
    )
    return (
        sdwl1._PACKET_HEADER.pack(
            sdwl1._PACKET_MAGIC,
            sdwl1._PACKET_VERSION,
            len(sections),
            len(body),
            hashlib.sha256(body).digest(),
        )
        + body
    )


def _write_source_npz(
    path: Path,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    compressed: bool = False,
) -> None:
    writer = np.savez_compressed if compressed else np.savez
    writer(path, lstars=arrays[0], margins=arrays[1], gt_poses=arrays[2])


def test_all_layouts_modes_and_independent_parse_back(
    inventory: sdwl1.FactInventory,
) -> None:
    for layout in sdwl1.SentenceLayout:
        collection = sdwl1.serialize_independent_descriptions(inventory, layout)
        independent = sdwl1.decode_independent_descriptions(collection)
        assert independent.semantic_sha256 == inventory.semantic_sha256
        assert np.array_equal(independent.tensor, inventory.tensor)
        for mode in sdwl1.TemporalMode:
            options = sdwl1.SentenceOptions(layout=layout, temporal_mode=mode)
            packet = sdwl1.serialize_sentence(inventory, options)
            decoded = sdwl1.decode_sentence(packet)
            assert decoded.semantic_sha256 == inventory.semantic_sha256
            assert np.array_equal(decoded.tensor, inventory.tensor)


def test_record_scalar_accounting_and_measured_productions() -> None:
    tensor = np.zeros((3, sdwl1.SEMANTIC_ROWS, sdwl1.SEMANTIC_WIDTH), dtype="<i8")
    tensor[2, 0, 0] = 1
    tensor[2, 1, 7] = 1
    tensor[2, 10, 0] = 1
    inventory = sdwl1.FactInventory(
        tensor=tensor,
        source_height=6,
        source_width=7,
        semantic_sha256=sdwl1._semantic_sha256(tensor),
    )
    counts = sdwl1.measure_production_counts(tensor).as_dict()
    assert inventory.described_record_count == 33
    assert inventory.described_scalar_fact_count == 228
    assert counts["predicates"] == {
        "declare": 11,
        "deform": 1,
        "hold": 19,
        "omit_kernel": 0,
        "project_range": 0,
        "topology_delta": 1,
        "transport": 1,
    }
    assert counts["topology_births"] == 1
    assert counts["topology_deaths"] == 0
    measurement = sdwl1.measure_serialization(
        inventory,
        options=sdwl1.SentenceOptions(
            layout=sdwl1.SentenceLayout.MONOLITHIC,
            temporal_mode=sdwl1.TemporalMode.CAUSAL_DELTA,
        ),
    )
    assert measurement.described_record_count == 33
    assert measurement.described_scalar_fact_count == 228
    assert measurement.described_fact_count == 228
    assert measurement.bytes_per_described_fact == measurement.outer_deflate_bytes / 228


def test_nonzero_padding_and_noncanonical_schema_are_rejected(
    inventory: sdwl1.FactInventory,
) -> None:
    padded = inventory.tensor.copy()
    padded[0, 5, 7] = 1
    with pytest.raises(sdwl1.SDWL1Error, match="padding"):
        sdwl1.FactInventory(
            tensor=padded,
            source_height=inventory.source_height,
            source_width=inventory.source_width,
            semantic_sha256=sdwl1._semantic_sha256(padded),
        )

    options = sdwl1.SentenceOptions(
        layout=sdwl1.SentenceLayout.MONOLITHIC,
        temporal_mode=sdwl1.TemporalMode.ABSOLUTE,
    )
    packet = sdwl1.serialize_sentence(inventory, options)
    sections = sdwl1._parse_sections(packet)
    schema = json.loads(sections[1][1])
    schema["bbox_convention"] = "closed"
    malformed_sections = [
        (tag, sdwl1.canonical_json_bytes(schema) if tag == b"SCHJ" else payload) for tag, payload in sections
    ]
    with pytest.raises(sdwl1.SDWL1Error, match="canonical schema"):
        sdwl1.decode_sentence(_frame_unchecked(malformed_sections))


def test_hold_is_pruned_when_not_measured(inventory: sdwl1.FactInventory) -> None:
    tensor = inventory.tensor[:1]
    one = sdwl1.FactInventory(
        tensor=tensor,
        source_height=inventory.source_height,
        source_width=inventory.source_width,
        semantic_sha256=sdwl1._semantic_sha256(tensor),
    )
    options = sdwl1.SentenceOptions(
        layout=sdwl1.SentenceLayout.MONOLITHIC,
        temporal_mode=sdwl1.TemporalMode.ABSOLUTE,
    )
    predicates = {row["name"]: row["use_count"] for row in sdwl1.build_lexicon(one, options)["predicates"]}
    assert predicates == {"declare": 11}


def test_pose_causal_delta_roundtrip_is_exact_modulo_2_pow_64() -> None:
    patterns = np.array(
        [
            [0, 1, 2**63 - 1, 2**63, 2**64 - 2, 2**64 - 1],
            [2**64 - 1, 2**63, 0, 1, 0x7FF0000000000000, 0xFFF0000000000000],
            [0x7FF8000000000001, 0xFFF8000000000001, 17, 2**64 - 17, 9, 3],
        ],
        dtype="<u8",
    )
    tensor = np.zeros((3, sdwl1.SEMANTIC_ROWS, sdwl1.SEMANTIC_WIDTH), dtype="<i8")
    tensor[:, 10, :6] = patterns.view("<i8")
    inventory = sdwl1.FactInventory(
        tensor=tensor,
        source_height=1,
        source_width=1,
        semantic_sha256=sdwl1._semantic_sha256(tensor),
    )
    for layout in sdwl1.SentenceLayout:
        packet = sdwl1.serialize_sentence(
            inventory,
            sdwl1.SentenceOptions(
                layout=layout,
                temporal_mode=sdwl1.TemporalMode.CAUSAL_DELTA,
            ),
        )
        decoded = sdwl1.decode_sentence(packet)
        assert np.array_equal(decoded.tensor[:, 10, :6].view("<u8"), patterns)


def test_real_event_mask_is_derived_and_decoder_verified(
    inventory: sdwl1.FactInventory,
) -> None:
    options = sdwl1.SentenceOptions(
        layout=sdwl1.SentenceLayout.TYPED_SECTION,
        temporal_mode=sdwl1.TemporalMode.CAUSAL_DELTA,
        redundant_event_masks=True,
    )
    packet = sdwl1.serialize_sentence(inventory, options)
    payloads = dict(sdwl1._parse_sections(packet))
    event_mask = decode_spatial_context_arithmetic(payloads[b"EVNT"])
    assert event_mask.shape == (inventory.pair_count, sdwl1.PAIR_RECORD_COUNT, 1)
    assert np.array_equal(event_mask, sdwl1._causal_event_mask(inventory.tensor))
    assert np.any(event_mask)
    event_mask[1, 0, 0] ^= 1
    corrupted = _replace_section(
        packet,
        b"EVNT",
        encode_spatial_context_arithmetic(event_mask),
    )
    with pytest.raises(sdwl1.SDWL1Error, match="event mask"):
        sdwl1.decode_sentence(corrupted)


def test_repeated_provenance_is_per_pair_framed_and_verified(
    inventory: sdwl1.FactInventory,
) -> None:
    options = sdwl1.SentenceOptions(
        layout=sdwl1.SentenceLayout.STRATUM_SECTION,
        temporal_mode=sdwl1.TemporalMode.ABSOLUTE,
        repeated_provenance=True,
    )
    packet = sdwl1.serialize_sentence(inventory, options)
    sections = sdwl1._parse_sections(packet)
    assert [tag for tag, _payload in sections][:3] == [b"LEXJ", b"SCHJ", b"PROV"]
    provenance = dict(sections)[b"PROV"]
    digest = sdwl1.canonical_provenance_digest()
    assert provenance == digest * inventory.pair_count
    corrupted = bytearray(provenance)
    corrupted[-1] ^= 1
    with pytest.raises(sdwl1.SDWL1Error, match="provenance"):
        sdwl1.decode_sentence(_replace_section(packet, b"PROV", bytes(corrupted)))


def test_all_mdl_counterfactuals_parse_and_split_counts_are_measured(
    inventory: sdwl1.FactInventory,
) -> None:
    for field in cli.COUNTERFACTUAL_FIELDS:
        options = sdwl1.SentenceOptions(
            layout=sdwl1.SentenceLayout.MONOLITHIC,
            temporal_mode=sdwl1.TemporalMode.CAUSAL_DELTA,
            **{field: True},
        )
        decoded = sdwl1.decode_sentence(sdwl1.serialize_sentence(inventory, options))
        assert np.array_equal(decoded.tensor, inventory.tensor)
    split = sdwl1.build_lexicon(
        inventory,
        sdwl1.SentenceOptions(
            layout=sdwl1.SentenceLayout.MONOLITHIC,
            temporal_mode=sdwl1.TemporalMode.ABSOLUTE,
            split_topology_vocabulary=True,
        ),
    )
    predicates = {row["name"]: row["use_count"] for row in split["predicates"]}
    measured = sdwl1.measure_production_counts(inventory.tensor)
    assert "topology_delta" not in predicates
    assert predicates["topology_birth"] == measured.topology_births
    assert predicates["topology_death"] == measured.topology_deaths


def test_explicit_indices_and_split_vocabulary_are_decoder_verified(
    inventory: sdwl1.FactInventory,
) -> None:
    indexed_options = sdwl1.SentenceOptions(
        layout=sdwl1.SentenceLayout.MONOLITHIC,
        temporal_mode=sdwl1.TemporalMode.ABSOLUTE,
        explicit_frame_indices=True,
    )
    indexed = sdwl1.serialize_sentence(inventory, indexed_options)
    payloads = dict(sdwl1._parse_sections(indexed))
    indices = decode_spatial_context_arithmetic(payloads[b"FIDX"])
    indices[1, 0, 0] = 99
    with pytest.raises(sdwl1.SDWL1Error, match="frame indices"):
        sdwl1.decode_sentence(
            _replace_section(
                indexed,
                b"FIDX",
                encode_spatial_context_arithmetic(indices),
            )
        )

    split_options = sdwl1.SentenceOptions(
        layout=sdwl1.SentenceLayout.MONOLITHIC,
        temporal_mode=sdwl1.TemporalMode.ABSOLUTE,
        split_topology_vocabulary=True,
    )
    split = sdwl1.serialize_sentence(inventory, split_options)
    lexicon = json.loads(dict(sdwl1._parse_sections(split))[b"LEXJ"])
    lexicon["predicates"][-1]["use_count"] += 1
    malformed_lexicon = sdwl1.canonical_json_bytes(lexicon)
    sections = sdwl1._parse_sections(split)
    malformed_sections = [(tag, malformed_lexicon if tag == b"LEXJ" else payload) for tag, payload in sections]
    schema = json.loads(malformed_sections[1][1])
    schema["lexicon_sha256"] = hashlib.sha256(malformed_lexicon).hexdigest()
    malformed_sections[1] = (b"SCHJ", sdwl1.canonical_json_bytes(schema))
    with pytest.raises(sdwl1.SDWL1Error, match="lexicon drift"):
        sdwl1.decode_sentence(_frame_unchecked(malformed_sections))


def test_malformed_truncated_trailing_and_arithmetic_rejection(
    inventory: sdwl1.FactInventory,
) -> None:
    options = sdwl1.SentenceOptions(
        layout=sdwl1.SentenceLayout.MONOLITHIC,
        temporal_mode=sdwl1.TemporalMode.ABSOLUTE,
    )
    packet = sdwl1.serialize_sentence(inventory, options)
    for invalid in (packet[:-1], packet + b"x", b"not-sdwl1"):
        with pytest.raises(sdwl1.SDWL1Error):
            sdwl1.decode_sentence(invalid)
    hash_drift = bytearray(packet)
    hash_drift[-1] ^= 1
    with pytest.raises(sdwl1.SDWL1Error, match="hash drift"):
        sdwl1.decode_sentence(bytes(hash_drift))
    sections = sdwl1._parse_sections(packet)
    with pytest.raises(sdwl1.SDWL1Error, match="unknown SDWL1 section"):
        sdwl1.decode_sentence(_frame_unchecked([(b"NOPE", sections[0][1]), *sections[1:]]))
    lexicon = json.loads(sections[0][1])
    noncanonical = json.dumps(lexicon, indent=2).encode()
    with pytest.raises(sdwl1.SDWL1Error, match="not canonical JSON"):
        sdwl1.decode_sentence(_frame_unchecked([(b"LEXJ", noncanonical), *sections[1:]]))
    mono = dict(sdwl1._parse_sections(packet))[b"MONO"]
    with pytest.raises(sdwl1.SDWL1Error, match="arithmetic"):
        sdwl1.decode_sentence(_replace_section(packet, b"MONO", mono + b"x"))
    collection = sdwl1.serialize_independent_descriptions(inventory, sdwl1.SentenceLayout.MONOLITHIC)
    for invalid in (collection[:-1], collection + b"x"):
        with pytest.raises(sdwl1.SDWL1Error):
            sdwl1.decode_independent_descriptions(invalid)


def test_outer_deflate_is_complete_deterministic_and_strict(
    inventory: sdwl1.FactInventory,
) -> None:
    options = sdwl1.SentenceOptions(
        layout=sdwl1.SentenceLayout.TYPED_SECTION,
        temporal_mode=sdwl1.TemporalMode.CAUSAL_DELTA,
    )
    first = sdwl1.measure_serialization(inventory, options=options)
    second = sdwl1.measure_serialization(inventory, options=options)
    assert first.outer_payload == second.outer_payload
    assert first.outer_deflate_sha256 == hashlib.sha256(first.outer_payload).hexdigest()
    assert sdwl1.decompress_outer_payload(first.outer_payload).startswith(b"SDWL1PK")
    with pytest.raises(sdwl1.SDWL1Error, match="outer-zlib"):
        sdwl1.decompress_outer_payload(first.outer_payload + b"x")
    with pytest.raises(sdwl1.SDWL1Error, match="outer-zlib"):
        sdwl1.decompress_outer_payload(first.outer_payload[:-1])


def test_direct_zip_stored_memmap_and_compressed_rejection(
    tmp_path: Path,
    source_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = tmp_path / "stored.npz"
    _write_source_npz(stored, source_arrays)
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("np.load used"))
    mapped = cli.stored_npy_memmap(stored, "lstars")
    assert isinstance(mapped.array, np.memmap)
    assert mapped.array.mode == "r"
    assert mapped.custody["source_access"] == "direct_zip_stored_npy_read_only_memmap"
    assert np.array_equal(mapped.array, source_arrays[0])
    compressed = tmp_path / "compressed.npz"
    _write_source_npz(compressed, source_arrays, compressed=True)
    with pytest.raises(cli.MeasurementError, match="not ZIP_STORED"):
        cli.stored_npy_memmap(compressed, "lstars")


def test_measurement_cli_synthetic_resume_and_payload_persistence(
    tmp_path: Path,
    source_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    source = tmp_path / "source.npz"
    output = tmp_path / "out"
    _write_source_npz(source, source_arrays)
    source_sha = cli.sha256_file(source)
    argv = [
        "--source-cache",
        str(source),
        "--output-dir",
        str(output),
        "--n-pairs",
        "3",
        "--expected-source-bytes",
        str(source.stat().st_size),
        "--expected-source-sha256",
        source_sha,
        "--min-free-bytes",
        "1",
    ]
    assert cli.main(argv) == 0
    receipt_path = output / cli.FINAL_RECEIPT
    first_receipt = receipt_path.read_bytes()
    receipt = json.loads(first_receipt)
    assert receipt["coverage"]["counterfactual_fields"] == list(cli.COUNTERFACTUAL_FIELDS)
    assert receipt["coverage"]["layouts"] == [layout.value for layout in sdwl1.SentenceLayout]
    assert receipt["coverage"]["row_count"] == 33
    assert receipt["coverage"]["temporal_modes"] == [mode.value for mode in sdwl1.TemporalMode]
    assert "arithmetic state reset per pair" in receipt["coverage"]["independent_baseline_definition"]
    assert receipt["source_custody"]["bytes"] == source.stat().st_size
    assert receipt["source_custody"]["sha256"] == source_sha
    assert receipt["source_custody"]["mutated"] is False
    assert receipt["score_claim"] is False
    assert receipt["promotion_eligible"] is False
    assert receipt["candidate_archive"] is False
    assert receipt["grammar"]["name"] == "Scorer-Derived Worldsheet Language v1"
    assert receipt["main_landing_review_required"] is True
    assert receipt["syntax"]["strict_parseback"] is True
    assert len(receipt["mdl_pruning"]) == len(cli.COUNTERFACTUAL_FIELDS)
    assert set(receipt["dimension_selection"]["zero_use_vocabulary_pruned"]) == {
        "modifiers",
        "predicates",
        "subjects",
    }
    assert all(row["measurement"]["exact_parseback"] for row in receipt["rows"])
    assert all(row["measurement"]["described_record_count"] == 33 for row in receipt["rows"])
    assert all(row["measurement"]["described_scalar_fact_count"] == 228 for row in receipt["rows"])
    for row in receipt["rows"]:
        payload = output / row["outer_payload"]["path"]
        assert payload.stat().st_size == row["outer_payload"]["bytes"]
        assert cli.sha256_file(payload) == row["outer_payload"]["sha256"]
        sdwl1.decompress_outer_payload(payload.read_bytes())
    assert cli.main([*argv, "--resume"]) == 0
    assert receipt_path.read_bytes() == first_receipt
    interrupted_row = output / "rows/whole_monolithic_absolute.json"
    interrupted_row.unlink()
    assert cli.main([*argv, "--resume"]) == 0
    assert interrupted_row.is_file()
    assert receipt_path.read_bytes() == first_receipt
    assert cli.main(argv) == 2


def test_resume_can_recover_before_first_custody_receipt(
    tmp_path: Path,
    source_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    source = tmp_path / "source.npz"
    output = tmp_path / "early-interruption"
    _write_source_npz(source, source_arrays)
    output.mkdir()
    argv = [
        "--source-cache",
        str(source),
        "--output-dir",
        str(output),
        "--n-pairs",
        "1",
        "--expected-source-bytes",
        str(source.stat().st_size),
        "--expected-source-sha256",
        cli.sha256_file(source),
        "--min-free-bytes",
        "1",
        "--resume",
    ]
    assert cli.main(argv) == 0
    assert (output / cli.STAGE_CUSTODY).is_file()
    assert (output / cli.FINAL_RECEIPT).is_file()


def test_cli_storage_and_source_custody_fail_closed(
    tmp_path: Path,
    source_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    source = tmp_path / "source.npz"
    _write_source_npz(source, source_arrays)
    with pytest.raises(cli.MeasurementError, match="storage preflight"):
        cli.storage_preflight(tmp_path / "out", required_free_bytes=2**63)
    in_tree = cli.storage_preflight(
        cli.REPO / ".omx/research/sdwl1-test-not-created",
        required_free_bytes=1,
    )
    assert in_tree["output_dir"] == ".omx/research/sdwl1-test-not-created"
    assert in_tree["preflight_path"] == ".omx/research"
    assert (
        cli.main(
            [
                "--source-cache",
                str(source),
                "--output-dir",
                str(tmp_path / "out"),
                "--n-pairs",
                "3",
                "--expected-source-bytes",
                str(source.stat().st_size),
                "--expected-source-sha256",
                "0" * 64,
                "--min-free-bytes",
                "1",
            ]
        )
        == 2
    )


def test_derivation_coverage_and_canonical_parent_source_names() -> None:
    sdwl1.validate_derivation_coverage()
    registry_sources = {source for entry in sdwl1.DERIVATION_REGISTRY for source in entry.sources}
    assert "upstream/modules.py" in registry_sources
    assert "upstream/frame_utils.py" in registry_sources
    assert all(spec.derivation_refs for spec in sdwl1.SUBJECT_SPECS)
    assert all(spec.derivation_refs for spec in sdwl1.PREDICATE_SPECS)
    assert all(spec.derivation_refs for spec in sdwl1.MODIFIER_SPECS)


def test_cli_defaults_are_bounded_n600_and_do_not_execute_cache() -> None:
    args = cli.build_parser().parse_args([])
    assert args.n_pairs == 600
    assert args.source_cache == cli.DEFAULT_SOURCE_CACHE
    assert args.expected_source_bytes == cli.EXPECTED_SOURCE_BYTES
    assert args.expected_source_sha256 == cli.EXPECTED_SOURCE_SHA256
    assert args.resume is False
