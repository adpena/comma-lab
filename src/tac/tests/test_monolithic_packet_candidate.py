# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import brotli
import pytest

from tac.frontier_archive_layout import inspect_frontier_archive_layout
from tac.monolithic_packet_candidate import (
    MonolithicPacketCandidateError,
    ReplacementSection,
    build_monolithic_packet_candidate,
    sha256_bytes,
    sha256_file,
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
        infos = zf.infolist()
        assert len(infos) == 1
        return zf.read(infos[0].filename)


def _pr106_payload(decoder: bytes, tail: bytes) -> bytes:
    return bytes([0xFF]) + len(decoder).to_bytes(3, "little") + decoder + tail


def test_build_monolithic_pr106_section_candidate_updates_header_and_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    manifest_path = tmp_path / "manifest.json"
    replacement = tmp_path / "decoder.bin"
    old_decoder = brotli.compress(b"old-decoder")
    tail = brotli.compress(b"latent-sidecar")
    new_decoder = brotli.compress(b"packed-decoder")
    _write_zip(source, name="0.bin", payload=_pr106_payload(old_decoder, tail))
    replacement.write_bytes(new_decoder)

    manifest = build_monolithic_packet_candidate(
        source_archive=source,
        output_archive=output,
        candidate_id="unit-pr106-decoder",
        replacements=[
            ReplacementSection(
                section_name="decoder_packed_brotli",
                replacement_path=replacement,
                expected_old_sha256=hashlib.sha256(old_decoder).hexdigest(),
                expected_old_bytes=len(old_decoder),
                expected_new_sha256=hashlib.sha256(new_decoder).hexdigest(),
                expected_new_bytes=len(new_decoder),
            )
        ],
        expected_source_archive_sha256=sha256_file(source),
        expected_source_archive_bytes=source.stat().st_size,
        manifest_output=manifest_path,
    )

    candidate_payload = _read_member(output)
    assert candidate_payload[:4] == bytes([0xFF]) + len(new_decoder).to_bytes(3, "little")
    assert candidate_payload[4:4 + len(new_decoder)] == new_decoder
    assert candidate_payload[4 + len(new_decoder):] == tail
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "runtime_consumption_proof_missing" in manifest["dispatch_blockers"]
    assert "active_lane_claim_missing" in manifest["dispatch_blockers"]
    assert manifest["promotion_blockers"] == ["contest_cuda_auth_eval_missing"]
    assert manifest["candidate_archive"]["sha256"] == sha256_file(output)
    assert manifest["replacements"][0]["section_byte_delta"] == len(new_decoder) - len(old_decoder)
    assert manifest_path.is_file()


def test_frontier_layout_prefers_pr106_magic_even_when_member_is_x(tmp_path: Path) -> None:
    archive = tmp_path / "pr106x.zip"
    _write_zip(
        archive,
        name="x",
        payload=_pr106_payload(brotli.compress(b"decoder"), brotli.compress(b"tail")),
    )

    manifest = inspect_frontier_archive_layout(archive)

    assert manifest["logical_layout"]["grammar"] == "pr106_ff_packed_hnerv"
    assert manifest["logical_layout"]["single_member_name"] == "x"
    assert manifest["logical_layout"]["sections"][1]["name"] == "decoder_packed_brotli"


def test_frontier_layout_fails_closed_on_ambiguous_unvalidated_member_x(tmp_path: Path) -> None:
    archive = tmp_path / "ambiguous.zip"
    payload = _pr106_payload(b"not-brotli-decoder", b"x" * 180_000)
    _write_zip(archive, name="x", payload=payload)

    manifest = inspect_frontier_archive_layout(archive)

    assert manifest["logical_layout"] is None
    assert any("PR106-like" in caution and "ambiguous" in caution for caution in manifest["cautions"])


def test_frontier_layout_resolves_ambiguous_member_x_with_brotli_streams(tmp_path: Path) -> None:
    archive = tmp_path / "ambiguous_valid.zip"
    raw_tail = hashlib.shake_256(b"ambiguous-pr106-tail").digest(180_000)
    payload = _pr106_payload(brotli.compress(b"decoder"), brotli.compress(raw_tail))
    assert len(payload) > 177_551
    _write_zip(archive, name="x", payload=payload)

    manifest = inspect_frontier_archive_layout(archive)

    logical = manifest["logical_layout"]
    assert logical["grammar"] == "pr106_ff_packed_hnerv"
    assert logical["parser_ambiguous"] is False
    assert logical["validated_streams"]["decoder_packed_brotli"] is True
    assert logical["validated_streams"]["latents_and_sidecar_brotli"] is True


def test_monolithic_candidate_rejects_unproven_member_layout(tmp_path: Path) -> None:
    source = tmp_path / "multi.zip"
    output = tmp_path / "candidate.zip"
    repl = tmp_path / "replacement.bin"
    repl.write_bytes(b"x")
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("a", b"a")
        zf.writestr("b", b"b")

    with pytest.raises(MonolithicPacketCandidateError, match="single-member"):
        build_monolithic_packet_candidate(
            source_archive=source,
            output_archive=output,
            candidate_id="bad",
            replacements=[
                ReplacementSection(
                    section_name="decoder_packed_brotli",
                    replacement_path=repl,
                )
            ],
        )


def test_pr106_multi_section_replacement_is_atomic_and_forged_claim_stays_blocked(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    decoder_repl = tmp_path / "decoder.bin"
    latent_repl = tmp_path / "latent.bin"
    old_decoder = brotli.compress(b"decoder-old")
    old_latent = brotli.compress(b"latent-old")
    new_decoder = brotli.compress(b"decoder-new-longer")
    new_latent = brotli.compress(b"latent-new-longer")
    expected_member = _pr106_payload(new_decoder, new_latent)
    _write_zip(source, name="x", payload=_pr106_payload(old_decoder, old_latent))
    decoder_repl.write_bytes(new_decoder)
    latent_repl.write_bytes(new_latent)

    manifest = build_monolithic_packet_candidate(
        source_archive=source,
        output_archive=output,
        candidate_id="unit-pr106-multi",
        replacements=[
            ReplacementSection(
                section_name="decoder_packed_brotli",
                replacement_path=decoder_repl,
                expected_old_sha256=sha256_bytes(old_decoder),
                expected_new_sha256=sha256_bytes(new_decoder),
            ),
            ReplacementSection(
                section_name="latents_and_sidecar_brotli",
                replacement_path=latent_repl,
                expected_old_bytes=len(old_latent),
                expected_new_bytes=len(new_latent),
            ),
        ],
        runtime_parity={
            "schema": "tac_runtime_consumption_proof_v1",
            "ready_for_exact_eval_runtime": True,
            "candidate_archive_sha256": "a" * 64,
            "new_member_sha256": sha256_bytes(expected_member),
            "changed_sections": {
                "decoder_packed_brotli": sha256_bytes(new_decoder),
                "latents_and_sidecar_brotli": sha256_bytes(new_latent),
            },
            "command_sha256": "b" * 64,
            "log_sha256": "c" * 64,
        },
        lane_claim={
            "active": True,
            "lane_id": "unit_monolithic_candidate",
            "instance_job_id": "unit-job",
            "status": "active_dispatching",
            "claims_path": ".omx/state/active_lane_dispatch_claims.md",
            "claimed_with": ".venv/bin/python tools/claim_lane_dispatch.py claim --lane-id unit_monolithic_candidate",
        },
    )

    assert _read_member(output) == expected_member
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "active_lane_claim_schema_missing_or_invalid" in manifest["dispatch_blockers"]
    assert "runtime_consumption_candidate_archive_sha_mismatch_or_missing" in manifest["dispatch_blockers"]
    assert manifest["promotion_blockers"] == ["contest_cuda_auth_eval_missing"]
    assert [item["section_name"] for item in manifest["replacements"]] == [
        "decoder_packed_brotli",
        "latents_and_sidecar_brotli",
    ]
    sections = {section["name"]: section for section in manifest["monolithic_layout"]["sections"]}
    assert sections["decoder_packed_brotli"]["new_offset"] == 4
    assert sections["latents_and_sidecar_brotli"]["new_offset"] == 4 + len(new_decoder)


def test_pr106_dispatch_gate_opens_only_with_exported_claim_and_strict_runtime_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.export_active_lane_claim_json import build_active_lane_claim_json

    monkeypatch.chdir(tmp_path)
    claims_path = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims_path.parent.mkdir(parents=True)
    claim_row = (
        "| 2026-05-08T00:05:00Z | codex | unit_lane | lightning | unit_job |  | "
        "active_dispatching | unit |"
    )
    claims_path.write_text(
        "\n".join(
            [
                "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
                "|---|---|---|---|---|---|---|---|",
                claim_row,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    decoder_repl = tmp_path / "decoder.bin"
    old_decoder = brotli.compress(b"old-decoder")
    old_latent = brotli.compress(b"latent")
    new_decoder = brotli.compress(b"new-decoder")
    _write_zip(source, name="0.bin", payload=_pr106_payload(old_decoder, old_latent))
    decoder_repl.write_bytes(new_decoder)

    provisional = build_monolithic_packet_candidate(
        source_archive=source,
        output_archive=output,
        candidate_id="unit-provisional",
        replacements=[
            ReplacementSection(
                section_name="decoder_packed_brotli",
                replacement_path=decoder_repl,
            )
        ],
    )
    runtime_proof = {
        "schema": "tac_runtime_consumption_proof_v1",
        "ready_for_exact_eval_runtime": True,
        "candidate_archive_sha256": provisional["candidate_archive"]["sha256"],
        "new_member_sha256": provisional["monolithic_layout"]["new_member_sha256"],
        "changed_sections": {
            "decoder_packed_brotli": sha256_bytes(new_decoder),
        },
        "command_sha256": "b" * 64,
        "log_sha256": "c" * 64,
    }
    lane_claim = build_active_lane_claim_json(
        claims_path=Path(".omx/state/active_lane_dispatch_claims.md"),
        lane_id="unit_lane",
        instance_job_id="unit_job",
        now_utc="2026-05-08T00:06:00Z",
    )

    manifest = build_monolithic_packet_candidate(
        source_archive=source,
        output_archive=output,
        candidate_id="unit-ready-for-dispatch",
        replacements=[
            ReplacementSection(
                section_name="decoder_packed_brotli",
                replacement_path=decoder_repl,
            )
        ],
        runtime_parity=runtime_proof,
        lane_claim=lane_claim,
        dispatch_lane_id="unit_lane",
        dispatch_instance_job_id="unit_job",
    )

    assert manifest["ready_for_exact_eval_dispatch"] is True
    assert manifest["dispatch_blockers"] == []
    assert manifest["promotion_blockers"] == ["contest_cuda_auth_eval_missing"]


def test_pr106_header_replacement_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    repl = tmp_path / "header.bin"
    _write_zip(
        source,
        name="0.bin",
        payload=_pr106_payload(brotli.compress(b"decoder"), brotli.compress(b"tail")),
    )
    repl.write_bytes(b"bad")

    with pytest.raises(MonolithicPacketCandidateError, match="ff_header"):
        build_monolithic_packet_candidate(
            source_archive=source,
            output_archive=output,
            candidate_id="bad-header",
            replacements=[
                ReplacementSection(
                    section_name="ff_header",
                    replacement_path=repl,
                )
            ],
        )


def test_noop_replacement_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    repl = tmp_path / "decoder.bin"
    decoder = brotli.compress(b"same-decoder")
    _write_zip(source, name="0.bin", payload=_pr106_payload(decoder, brotli.compress(b"tail")))
    repl.write_bytes(decoder)

    with pytest.raises(MonolithicPacketCandidateError, match="no-op"):
        build_monolithic_packet_candidate(
            source_archive=source,
            output_archive=output,
            candidate_id="bad-noop",
            replacements=[
                ReplacementSection(
                    section_name="decoder_packed_brotli",
                    replacement_path=repl,
                )
            ],
        )


def test_pr106_replacement_must_remain_brotli_runtime_section(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    repl = tmp_path / "decoder.bin"
    _write_zip(
        source,
        name="0.bin",
        payload=_pr106_payload(brotli.compress(b"decoder"), brotli.compress(b"tail")),
    )
    repl.write_bytes(b"not-brotli-runtime-bytes")

    with pytest.raises(MonolithicPacketCandidateError, match="does not Brotli-decompress"):
        build_monolithic_packet_candidate(
            source_archive=source,
            output_archive=output,
            candidate_id="bad-pr106-brotli",
            replacements=[
                ReplacementSection(
                    section_name="decoder_packed_brotli",
                    replacement_path=repl,
                )
            ],
        )


def test_pr101_fixed_offset_decoder_replacement_must_keep_length(tmp_path: Path) -> None:
    source = tmp_path / "pr101.zip"
    output = tmp_path / "candidate.zip"
    repl = tmp_path / "replacement.bin"
    decoder = b"a" * 162_164
    latent = b"b" * 15_387
    sidecar = b"c"
    repl.write_bytes(b"short")
    _write_zip(source, name="x", payload=decoder + latent + sidecar)

    with pytest.raises(MonolithicPacketCandidateError, match="fixed-offset"):
        build_monolithic_packet_candidate(
            source_archive=source,
            output_archive=output,
            candidate_id="bad-pr101",
            replacements=[
                ReplacementSection(
                    section_name="decoder_blob",
                    replacement_path=repl,
                    expected_old_sha256=sha256_bytes(decoder),
                )
            ],
        )


def test_cli_accepts_replacement_manifest_for_multi_section_candidate(tmp_path: Path) -> None:
    from tools.build_monolithic_stack_candidate import main

    source = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    manifest_output = tmp_path / "manifest.json"
    replacement_manifest = tmp_path / "replacements.json"
    decoder_repl = tmp_path / "decoder.bin"
    latent_repl = tmp_path / "latent.bin"
    old_decoder = brotli.compress(b"old-decoder")
    old_latent = brotli.compress(b"old-latent")
    new_decoder = brotli.compress(b"new-decoder-longer")
    new_latent = brotli.compress(b"new-latent")
    _write_zip(source, name="0.bin", payload=_pr106_payload(old_decoder, old_latent))
    decoder_repl.write_bytes(new_decoder)
    latent_repl.write_bytes(new_latent)
    replacement_manifest.write_text(
        json.dumps(
            {
                "replacements": [
                    {
                        "section_name": "decoder_packed_brotli",
                        "replacement_path": decoder_repl.name,
                        "expected_old_bytes": len(old_decoder),
                        "expected_new_bytes": len(new_decoder),
                    },
                    {
                        "section_name": "latents_and_sidecar_brotli",
                        "replacement_path": latent_repl.name,
                        "expected_old_sha256": sha256_bytes(old_latent),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "--source-archive",
            str(source),
            "--output-archive",
            str(output),
            "--manifest-output",
            str(manifest_output),
            "--candidate-id",
            "unit-cli-multi",
            "--replacement-manifest",
            str(replacement_manifest),
        ]
    )

    assert rc == 0
    manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
    assert _read_member(output) == _pr106_payload(new_decoder, new_latent)
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert len(manifest["replacements"]) == 2


def test_cli_opens_dispatch_with_runtime_proof_and_claim_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.build_monolithic_runtime_consumption_proof import build_runtime_consumption_proof
    from tools.build_monolithic_stack_candidate import main

    monkeypatch.chdir(tmp_path)
    claims_path = Path(".omx/state/active_lane_dispatch_claims.md")
    claims_path.parent.mkdir(parents=True)
    claims_path.write_text(
        "\n".join(
            [
                "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
                "|---|---|---|---|---|---|---|---|",
                "| 2026-05-08T00:05:00Z | codex | unit_lane | lightning | unit_job |  | active_dispatching | unit |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    provisional_manifest_path = tmp_path / "provisional.json"
    final_manifest_path = tmp_path / "final.json"
    runtime_proof_path = tmp_path / "runtime_proof.json"
    runtime_log = tmp_path / "runtime.log"
    replacement = tmp_path / "decoder.bin"
    old_decoder = brotli.compress(b"old-decoder")
    tail = brotli.compress(b"tail")
    new_decoder = brotli.compress(b"new-decoder")
    _write_zip(source, name="0.bin", payload=_pr106_payload(old_decoder, tail))
    replacement.write_bytes(new_decoder)

    provisional = build_monolithic_packet_candidate(
        source_archive=source,
        output_archive=output,
        candidate_id="unit-cli-ready-provisional",
        replacements=[
            ReplacementSection(
                section_name="decoder_packed_brotli",
                replacement_path=replacement,
            )
        ],
        manifest_output=provisional_manifest_path,
    )
    runtime_log.write_text(
        " ".join(
            [
                provisional["candidate_archive"]["sha256"],
                provisional["monolithic_layout"]["new_member_sha256"],
                sha256_bytes(new_decoder),
            ]
        ),
        encoding="utf-8",
    )
    proof = build_runtime_consumption_proof(
        candidate_manifest_path=provisional_manifest_path,
        command_text="inflate.sh candidate.zip",
        runtime_log=runtime_log,
    )
    runtime_proof_path.write_text(json.dumps(proof), encoding="utf-8")

    rc = main(
        [
            "--source-archive",
            str(source),
            "--output-archive",
            str(output),
            "--manifest-output",
            str(final_manifest_path),
            "--candidate-id",
            "unit-cli-ready",
            "--target-section",
            "decoder_packed_brotli",
            "--replacement-section",
            str(replacement),
            "--runtime-parity-json",
            str(runtime_proof_path),
            "--claims-path",
            str(claims_path),
            "--dispatch-lane-id",
            "unit_lane",
            "--dispatch-instance-job-id",
            "unit_job",
            "--now-utc",
            "2026-05-08T00:06:00Z",
        ]
    )

    assert rc == 0
    manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    assert manifest["ready_for_exact_eval_dispatch"] is True
    assert manifest["dispatch_blockers"] == []
    assert manifest["promotion_blockers"] == ["contest_cuda_auth_eval_missing"]
