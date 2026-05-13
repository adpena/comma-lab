from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import brotli

from tac.hnerv_brotli_saturation import build_hnerv_decoder_brotli_saturation_audit, render_markdown
from tac.hnerv_lowlevel_packer import sha256_bytes, write_stored_single_member_zip
from tac.repo_io import json_text, sha256_file

REPO = Path(__file__).resolve().parents[3]


def test_brotli_saturation_audit_proves_no_smaller_single_attempt(tmp_path: Path) -> None:
    raw = b"deterministic hnerv decoder raw" * 128
    decoder = brotli.compress(raw, quality=5)
    archive = _write_archive(tmp_path / "archive.zip", decoder)
    scorecard = _scorecard("fixture", archive, decoder)
    entropy_ranking = _entropy_ranking("fixture", archive, decoder)

    manifest = build_hnerv_decoder_brotli_saturation_audit(
        source_archive=archive,
        source_label="fixture",
        scorecard=scorecard,
        entropy_ranking=entropy_ranking,
        qualities=[5],
        lgwins=[None],
        lgblocks=[None],
        modes=["generic"],
        jobs=1,
    )

    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["scorecard_anchor"]["matched"] is True
    assert manifest["entropy_ranking_anchor"]["matched"] is True
    assert manifest["grid"]["attempt_count"] == 1
    assert manifest["rate_positive_attempt_count"] == 0
    assert manifest["proof_summary"]["verdict"] == "bounded_brotli_grid_saturated"
    assert manifest["proof_summary"]["bytes_short_of_rate_positive"] == 1
    assert manifest["best_attempt"]["candidate_section_sha256"] == sha256_bytes(decoder)
    assert "HNeRV Decoder Brotli Saturation Audit" in render_markdown(manifest)


def test_brotli_saturation_audit_supports_pr101_split_brotli_section(tmp_path: Path) -> None:
    raw_chunks = tuple((bytes([idx]) + b" split decoder raw ") * 64 for idx in range(7))
    decoder = b"".join(brotli.compress(chunk, quality=5) for chunk in raw_chunks)
    archive = _write_split_archive(tmp_path / "split.zip", decoder)
    scorecard = _scorecard("PR106-R2-lowlevel", archive, decoder, section_name="decoder_compact_brotli_streams")
    entropy_ranking = _entropy_ranking(
        "PR106-R2-lowlevel",
        archive,
        decoder,
        section_name="decoder_compact_brotli_streams",
    )

    manifest = build_hnerv_decoder_brotli_saturation_audit(
        source_archive=archive,
        source_label="PR106-R2-lowlevel",
        scorecard=scorecard,
        entropy_ranking=entropy_ranking,
        qualities=[5],
        lgwins=[None],
        lgblocks=[None],
        modes=["generic"],
        jobs=1,
    )

    assert manifest["source_decoder_section_name"] == "decoder_compact_brotli_streams"
    assert manifest["source_decoder_raw_bytes"] == sum(len(chunk) for chunk in raw_chunks)
    assert manifest["scorecard_anchor"]["matched"] is True
    assert manifest["entropy_ranking_anchor"]["matched"] is True
    assert manifest["best_attempt"]["raw_equal"] is True


def test_audit_hnerv_brotli_saturation_cli_writes_manifest(tmp_path: Path) -> None:
    raw = bytes(range(64)) * 64
    decoder = brotli.compress(raw, quality=4)
    archive = _write_archive(tmp_path / "archive.zip", decoder)
    scorecard_path = tmp_path / "scorecard.json"
    ranking_path = tmp_path / "ranking.json"
    json_out = tmp_path / "audit.json"
    md_out = tmp_path / "audit.md"
    scorecard_path.write_text(json_text(_scorecard("fixture", archive, decoder)), encoding="utf-8")
    ranking_path.write_text(json_text(_entropy_ranking("fixture", archive, decoder)), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_hnerv_brotli_saturation.py"),
            "--source-archive",
            str(archive),
            "--source-label",
            "fixture",
            "--scorecard",
            str(scorecard_path),
            "--entropy-ranking",
            str(ranking_path),
            "--quality",
            "4",
            "--lgwin",
            "default",
            "--lgblock",
            "default",
            "--mode",
            "generic",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO,
        check=True,
        text=True,
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["tool"] == "tac.hnerv_brotli_saturation.build_hnerv_decoder_brotli_saturation_audit"
    assert payload["tool_run_manifest"]["tool"] == "tools/audit_hnerv_brotli_saturation.py"
    assert payload["source_decoder_section_bytes"] == len(decoder)
    assert "Best Attempt" in md_out.read_text(encoding="utf-8")


def _write_archive(path: Path, decoder: bytes) -> Path:
    payload = b"\xff" + len(decoder).to_bytes(3, "little") + decoder + brotli.compress(
        b"latents", quality=3
    )
    write_stored_single_member_zip(path, member_name="x", payload=payload)
    return path


def _write_split_archive(path: Path, decoder: bytes) -> Path:
    section_total = 4 + len(decoder)
    payload = section_total.to_bytes(4, "little") + decoder + b"latent-sidecar"
    write_stored_single_member_zip(path, member_name="x", payload=payload)
    return path


def _scorecard(
    label: str,
    archive: Path,
    decoder: bytes,
    *,
    section_name: str = "decoder_packed_brotli",
) -> dict:
    payload = _payload(archive)
    return {
        "schema_version": 1,
        "rows": [
            {
                "label": label,
                "canonical_frontier_eligible": True,
                "score": 0.1,
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": sha256_file(archive),
                "payload_sha256": sha256_bytes(payload),
                "frontier_scope": "fixture",
                "evidence_grade": "empirical",
                "payload_sections": [
                    {
                        "name": section_name,
                        "bytes": len(decoder),
                        "sha256": sha256_bytes(decoder),
                        "entropy_bits_per_byte": 7.0,
                    }
                ],
            }
        ],
    }


def _entropy_ranking(
    label: str,
    archive: Path,
    decoder: bytes,
    *,
    section_name: str = "decoder_packed_brotli",
) -> dict:
    return {
        "schema_version": 1,
        "current_frontier": {
            "label": label,
            "archive_sha256": sha256_file(archive),
        },
        "next_entropy_research_action": {
            "action_id": "fixture_entropy_action",
            "target_section": section_name,
            "minimum_section_bytes_to_beat": len(decoder) - 1,
        },
        "frontier_byte_mass_ranking": [
            {
                "section": section_name,
                "section_bytes": len(decoder),
                "section_sha256": sha256_bytes(decoder),
                "entropy_bits_per_byte": 7.0,
            }
        ],
    }


def _payload(archive: Path) -> bytes:
    import zipfile

    with zipfile.ZipFile(archive) as zf:
        [name] = zf.namelist()
        return zf.read(name)
