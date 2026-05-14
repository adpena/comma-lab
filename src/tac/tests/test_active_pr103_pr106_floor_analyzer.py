# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from tools.analyze_active_pr103_pr106_floor import (
    build_anatomy_report,
    main,
    sha256_bytes,
    synthetic_fixture_zip_blob,
)


def _write_fixture_zip(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("0.bin", date_time=(2026, 5, 7, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _payload_and_closure() -> tuple[bytes, dict]:
    decoder_sections = {
        "scales_fp16": b"s" * 56,
        "br": b"bbbb",
        "hists": b"hh",
        "merged_ac": b"range",
        "hi_hist": b"i",
        "ac_fallback": b"",
    }
    decoder = decoder_sections["scales_fp16"] + b"".join(
        decoder_sections[name] for name in ("br", "hists", "merged_ac", "hi_hist", "ac_fallback")
    )
    latents = b"latents"
    payload = b"\xff" + len(decoder).to_bytes(3, "little") + decoder + latents
    return payload, {
        "runtime_closure": {
            "schema_version": 1,
            "format": "pr103_ac_decoder_inside_pr106_ff_packed_v1",
            "section_lengths": {
                name: len(value)
                for name, value in decoder_sections.items()
                if name in ("br", "hists", "merged_ac", "hi_hist", "ac_fallback")
            },
            "ac_fallback_set": [],
            "n_latent_hi_symbols": 0,
            "decoder_section_bytes": len(decoder),
            "decoder_section_sha256": sha256_bytes(decoder),
            "latents_section_bytes": len(latents),
            "latents_section_sha256": sha256_bytes(latents),
        }
    }


def test_synthetic_fixture_report_records_zip_table_and_nested_decoder_sections(tmp_path: Path) -> None:
    public_root = tmp_path / "missing_public_prs"
    report = build_anatomy_report(
        archive_path=tmp_path / "missing_archive.zip",
        auth_eval_path=None,
        runtime_closure_path=None,
        runtime_packet_proof_path=None,
        pre_submission_path=None,
        public_intake_root=public_root,
        candidate_manifest_paths=(),
        repo_root=tmp_path,
        allow_synthetic_fixture=True,
    )

    archive = report["active_archive"]
    candidate = archive["packed_payload_candidates"][0]

    assert report["score_claim"] is False
    assert report["logs_parsed_for_score"] is False
    assert archive["synthetic_fixture"] is True
    assert archive["zip_summary"]["file_member_count"] == 1
    assert archive["zip_member_table"][0]["local_header_name"] == "0.bin"
    assert candidate["kind"] == "pr103_ac_decoder_inside_pr106_ff_packed_v1"
    assert [section["name"] for section in candidate["nested_decoder_sections"]] == [
        "decoder.scales_fp16",
        "decoder.br",
        "decoder.hists",
        "decoder.merged_ac",
        "decoder.hi_hist",
        "decoder.ac_fallback",
    ]
    assert "auth_eval_artifact_missing" in report["exact_eval_blockers"]


def test_auth_eval_score_fields_are_suppressed_while_identity_is_preserved(tmp_path: Path) -> None:
    payload, closure = _payload_and_closure()
    archive = tmp_path / "archive.zip"
    closure_path = tmp_path / "runtime_closure.json"
    auth_eval_path = tmp_path / "contest_auth_eval.adjudicated.json"
    _write_fixture_zip(archive, payload)
    closure_path.write_text(json.dumps(closure), encoding="utf-8")
    auth_eval_path.write_text(
        json.dumps(
            {
                "archive_size_bytes": archive.stat().st_size,
                "avg_posenet_dist": 1.0,
                "avg_segnet_dist": 2.0,
                "final_score": 3.0,
                "n_samples": 600,
                "score_recomputed_from_components": 4.0,
                "provenance": {
                    "archive_sha256": sha256_bytes(archive.read_bytes()),
                    "archive_size_bytes": archive.stat().st_size,
                    "cuda_available": True,
                    "cuda_device_count": 1,
                    "cuda_version": "13.0",
                    "device": "cuda",
                    "gpu_model": "Tesla T4",
                    "gpu_t4_match": True,
                    "inflate_runtime_manifest": {
                        "runtime_file_count": 1,
                        "runtime_tree_sha256": "a" * 64,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_anatomy_report(
        archive_path=archive,
        auth_eval_path=auth_eval_path,
        runtime_closure_path=closure_path,
        runtime_packet_proof_path=None,
        pre_submission_path=None,
        public_intake_root=tmp_path / "missing_public_prs",
        candidate_manifest_paths=(),
        repo_root=tmp_path,
        allow_synthetic_fixture=False,
    )

    encoded = json.dumps(report, sort_keys=True)
    assert report["exact_eval_blockers"] == []
    assert report["auth_eval_custody"]["score_values_suppressed"] is True
    assert report["auth_eval_custody"]["score_fields_suppressed"] == [
        "avg_posenet_dist",
        "avg_segnet_dist",
        "final_score",
        "score_recomputed_from_components",
    ]
    assert "4.0" not in encoded
    assert report["auth_eval_custody"]["runtime_tree_custody"]["runtime_tree_sha256"] == "a" * 64


def test_public_pr_inventory_preserves_pr100_107_metadata(tmp_path: Path) -> None:
    payload, closure = _payload_and_closure()
    archive = tmp_path / "archive.zip"
    closure_path = tmp_path / "runtime_closure.json"
    _write_fixture_zip(archive, payload)
    closure_path.write_text(json.dumps(closure), encoding="utf-8")

    public_root = tmp_path / "public_pr_intake_full"
    pr103 = public_root / "public_pr103_intake_20260505_auto"
    pr103.mkdir(parents=True)
    _write_fixture_zip(pr103 / "archive.zip", payload)
    (pr103 / "pr_metadata.json").write_text(
        json.dumps(
            {
                "author": "rem2",
                "head_repo": "rem2/comma_video_compression_challenge",
                "head_sha": "deadbeef",
                "leaderboard_name": "hnerv_lc_ac",
                "leaderboard_score": 0.195,
                "pr_number": 103,
                "title": "hnerv_lc_ac submission",
            }
        ),
        encoding="utf-8",
    )
    (pr103 / "archive_provenance.json").write_text(
        json.dumps({"archive_sha256": sha256_bytes((pr103 / "archive.zip").read_bytes()), "archive_size_bytes": 123}),
        encoding="utf-8",
    )

    report = build_anatomy_report(
        archive_path=archive,
        auth_eval_path=None,
        runtime_closure_path=closure_path,
        runtime_packet_proof_path=None,
        pre_submission_path=None,
        public_intake_root=public_root,
        candidate_manifest_paths=(),
        repo_root=tmp_path,
        allow_synthetic_fixture=False,
    )

    inventory = {item["pr_number"]: item for item in report["public_pr100_107_inventory"]}
    assert sorted(inventory) == list(range(100, 108))
    assert inventory[103]["archive_present"] is True
    assert inventory[103]["title"] == "hnerv_lc_ac submission"
    assert inventory[103]["url"].endswith("/pull/103")
    assert inventory[100]["archive_present"] is False


def test_cli_writes_json_and_ledger_with_synthetic_fixture(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    ledger = tmp_path / "ledger.md"
    exit_code = main(
        [
            "--archive",
            str(tmp_path / "missing_archive.zip"),
            "--auth-eval",
            str(tmp_path / "missing_auth.json"),
            "--runtime-closure",
            str(tmp_path / "missing_closure.json"),
            "--runtime-packet-proof",
            str(tmp_path / "missing_proof.json"),
            "--pre-submission",
            str(tmp_path / "missing_pre_submission.json"),
            "--public-intake-root",
            str(tmp_path / "missing_public_prs"),
            "--output",
            str(output),
            "--ledger-out",
            str(ledger),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["score_claim"] is False
    assert "does not claim score" in ledger.read_text(encoding="utf-8")
    assert synthetic_fixture_zip_blob()[0]
