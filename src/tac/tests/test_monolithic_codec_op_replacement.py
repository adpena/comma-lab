# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import brotli
import pytest

from tac.monolithic_codec_op_replacement import (
    MonolithicCodecOpReplacementError,
    build_monolithic_codec_op_replacement_manifest,
    sha256_bytes,
    sha256_file,
)
from tac.monolithic_packet_candidate import (
    ReplacementSection,
    build_monolithic_packet_candidate,
)


def _write_zip(path: Path, *, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _read_member(path: Path) -> bytes:
    with zipfile.ZipFile(path) as zf:
        info = zf.infolist()[0]
        return zf.read(info.filename)


def _pr106_payload(decoder: bytes, tail: bytes) -> bytes:
    return bytes([0xFF]) + len(decoder).to_bytes(3, "little") + decoder + tail


def test_pr106_adapter_emits_manifest_consumed_by_monolithic_builder(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zip"
    replacements_json = tmp_path / "replacements.json"
    candidate = tmp_path / "candidate.zip"
    old_decoder = brotli.compress(b"old-decoder")
    old_tail = brotli.compress(b"tail")
    new_decoder = brotli.compress(b"new-decoder-expanded")
    replacement = tmp_path / "decoder.br"
    replacement.write_bytes(new_decoder)
    _write_zip(source, name="0.bin", payload=_pr106_payload(old_decoder, old_tail))

    manifest = build_monolithic_codec_op_replacement_manifest(
        source_archive=source,
        target_section="decoder_packed_brotli",
        replacement_payload=replacement,
        output_replacement_manifest=replacements_json,
        candidate_id="unit-codecop-decoder",
        section_payload_contract="pr106_decoder_packed_brotli",
        expected_source_archive_sha256=sha256_file(source),
        expected_source_archive_bytes=source.stat().st_size,
    )

    assert manifest["schema"] == "tac_monolithic_codec_op_replacement_manifest_v1"
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["replacements"] == [
        {
            "section_name": "decoder_packed_brotli",
            "replacement_path": "decoder.br",
            "expected_old_sha256": sha256_bytes(old_decoder),
            "expected_old_bytes": len(old_decoder),
            "expected_new_sha256": sha256_bytes(new_decoder),
            "expected_new_bytes": len(new_decoder),
        }
    ]

    replacement_rows = json.loads(replacements_json.read_text(encoding="utf-8"))
    build_manifest = build_monolithic_packet_candidate(
        source_archive=source,
        output_archive=candidate,
        candidate_id="unit-codecop-built",
        replacements=[
            ReplacementSection(
                section_name=replacement_rows["replacements"][0]["section_name"],
                replacement_path=replacements_json.parent
                / replacement_rows["replacements"][0]["replacement_path"],
                expected_old_sha256=replacement_rows["replacements"][0]["expected_old_sha256"],
                expected_old_bytes=replacement_rows["replacements"][0]["expected_old_bytes"],
                expected_new_sha256=replacement_rows["replacements"][0]["expected_new_sha256"],
                expected_new_bytes=replacement_rows["replacements"][0]["expected_new_bytes"],
            )
        ],
    )
    roundtrip_payload = _read_member(candidate)
    assert roundtrip_payload == _pr106_payload(new_decoder, old_tail)
    assert build_manifest["ready_for_exact_eval_dispatch"] is False


@pytest.mark.parametrize(
    ("target_section", "match"),
    [
        ("ff_header", "ff_header"),
        ("missing_section", "unknown parser-proven section"),
    ],
)
def test_rejects_header_and_unknown_sections(
    tmp_path: Path,
    target_section: str,
    match: str,
) -> None:
    source = tmp_path / "source.zip"
    replacement = tmp_path / "replacement.bin"
    old_decoder = brotli.compress(b"old")
    old_tail = brotli.compress(b"tail")
    replacement.write_bytes(b"replacement")
    _write_zip(source, name="0.bin", payload=_pr106_payload(old_decoder, old_tail))

    with pytest.raises(MonolithicCodecOpReplacementError, match=match):
        build_monolithic_codec_op_replacement_manifest(
            source_archive=source,
            target_section=target_section,
            replacement_payload=replacement,
            output_replacement_manifest=tmp_path / "replacements.json",
            candidate_id="bad-section",
        )


def test_rejects_pr101_fixed_offset_length_change(tmp_path: Path) -> None:
    source = tmp_path / "pr101.zip"
    replacement = tmp_path / "replacement.bin"
    decoder = b"a" * 162_164
    latent = b"b" * 15_387
    replacement.write_bytes(b"shorter")
    _write_zip(source, name="x", payload=decoder + latent)

    with pytest.raises(MonolithicCodecOpReplacementError, match="fixed-offset"):
        build_monolithic_codec_op_replacement_manifest(
            source_archive=source,
            target_section="decoder_blob",
            replacement_payload=replacement,
            output_replacement_manifest=tmp_path / "replacements.json",
            candidate_id="bad-pr101",
        )


def test_rejects_planning_json_without_materialized_payload(tmp_path: Path) -> None:
    from tools.build_monolithic_codec_op_replacement_manifest import main

    source = tmp_path / "source.zip"
    evidence = tmp_path / "planning.json"
    old_decoder = brotli.compress(b"old")
    old_tail = brotli.compress(b"tail")
    _write_zip(source, name="0.bin", payload=_pr106_payload(old_decoder, old_tail))
    evidence.write_text(
        json.dumps(
            {
                "schema": "codec_op_admm_adapter_planning_row_v1",
                "bytes_out": 7,
                "dispatchable": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--source-archive",
                str(source),
                "--target-section",
                "decoder_packed_brotli",
                "--output-replacement-manifest",
                str(tmp_path / "replacements.json"),
                "--candidate-id",
                "no-payload",
                "--evidence-json",
                str(evidence),
            ]
        )
    assert excinfo.value.code != 0


def test_rejects_codecop_envelope_as_raw_runtime_section(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    replacement = tmp_path / "replacement.bin"
    old_decoder = brotli.compress(b"old")
    old_tail = brotli.compress(b"tail")
    replacement.write_bytes(b"COBM1not-a-runtime-section")
    _write_zip(source, name="0.bin", payload=_pr106_payload(old_decoder, old_tail))

    with pytest.raises(MonolithicCodecOpReplacementError, match="CodecOp/JCS envelope"):
        build_monolithic_codec_op_replacement_manifest(
            source_archive=source,
            target_section="decoder_packed_brotli",
            replacement_payload=replacement,
            output_replacement_manifest=tmp_path / "replacements.json",
            candidate_id="bad-envelope",
            section_payload_contract="raw_section_bytes",
        )


def test_evidence_json_must_match_materialized_payload(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    replacement = tmp_path / "decoder.br"
    evidence = tmp_path / "evidence.json"
    old_decoder = brotli.compress(b"old")
    old_tail = brotli.compress(b"tail")
    new_decoder = brotli.compress(b"new")
    replacement.write_bytes(new_decoder)
    _write_zip(source, name="0.bin", payload=_pr106_payload(old_decoder, old_tail))

    evidence.write_text(
        json.dumps(
            {
                "schema": "codec_op_admm_adapter_planning_row_v1",
                "materialized_payload_bytes": len(new_decoder),
                "materialized_payload_sha256": sha256_bytes(new_decoder),
                "dispatchable": False,
            }
        ),
        encoding="utf-8",
    )
    manifest = build_monolithic_codec_op_replacement_manifest(
        source_archive=source,
        target_section="decoder_packed_brotli",
        replacement_payload=replacement,
        output_replacement_manifest=tmp_path / "replacements.json",
        candidate_id="matched-evidence",
        section_payload_contract="pr106_brotli_section",
        evidence_json=evidence,
    )
    assert manifest["evidence_json"]["bytes_bound"] is True
    assert manifest["evidence_json"]["sha256_bound"] is True
    assert manifest["evidence_json"]["planning_only"] is True

    evidence.write_text(
        json.dumps(
            {
                "schema": "codec_op_admm_adapter_planning_row_v1",
                "materialized_payload_bytes": len(new_decoder) + 1,
                "materialized_payload_sha256": sha256_bytes(new_decoder),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MonolithicCodecOpReplacementError, match="evidence_bytes_out_mismatch"):
        build_monolithic_codec_op_replacement_manifest(
            source_archive=source,
            target_section="decoder_packed_brotli",
            replacement_payload=replacement,
            output_replacement_manifest=tmp_path / "bad-replacements.json",
            candidate_id="mismatched-evidence",
            evidence_json=evidence,
        )


def test_cli_infers_replacement_payload_from_materialized_evidence(
    tmp_path: Path,
) -> None:
    from tools.build_monolithic_codec_op_replacement_manifest import main

    source = tmp_path / "source.zip"
    replacement = tmp_path / "decoder.br"
    evidence = tmp_path / "evidence.json"
    output = tmp_path / "replacements.json"
    old_decoder = brotli.compress(b"old")
    old_tail = brotli.compress(b"tail")
    new_decoder = brotli.compress(b"new-from-evidence")
    replacement.write_bytes(new_decoder)
    _write_zip(source, name="0.bin", payload=_pr106_payload(old_decoder, old_tail))
    evidence.write_text(
        json.dumps(
            {
                "schema": "codec_op_cma_search_eval_v1",
                "materialized_payload_path": replacement.name,
                "materialized_payload_bytes": len(new_decoder),
                "materialized_payload_sha256": sha256_bytes(new_decoder),
                "score_claim": False,
            }
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "--source-archive",
            str(source),
            "--target-section",
            "decoder_packed_brotli",
            "--output-replacement-manifest",
            str(output),
            "--candidate-id",
            "evidence-inferred",
            "--section-payload-contract",
            "pr106_decoder_packed_brotli",
            "--evidence-json",
            str(evidence),
        ]
    )

    assert rc == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["replacements"][0]["replacement_path"] == "decoder.br"
    assert manifest["replacement_payload"]["sha256"] == sha256_bytes(new_decoder)
