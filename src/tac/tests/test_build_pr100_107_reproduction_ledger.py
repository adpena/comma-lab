# SPDX-License-Identifier: MIT
"""Tests for ``tools.build_pr100_107_reproduction_ledger``."""

from __future__ import annotations

import importlib.util
import json
import pathlib
from zipfile import ZipFile


def _load_tool_module():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    tool_path = repo_root / "tools" / "build_pr100_107_reproduction_ledger.py"
    spec = importlib.util.spec_from_file_location(
        "build_pr100_107_reproduction_ledger",
        tool_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {tool_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_pr_fixture(repo: pathlib.Path, pr: int, name: str = "hnerv_fixture") -> pathlib.Path:
    intake = repo / f"experiments/results/public_pr_intake_full/public_pr{pr}_intake_20260505_auto"
    source = intake / "source/submissions" / name
    source.mkdir(parents=True)
    (source / "inflate.py").write_text("print('inflate')\n", encoding="utf-8")
    (source / "inflate.sh").write_text("#!/usr/bin/env bash\npython inflate.py\n", encoding="utf-8")
    (source / "compress.sh").write_text("#!/usr/bin/env bash\nzip archive.zip 0.bin\n", encoding="utf-8")
    (source / "README.md").write_text("# fixture\n", encoding="utf-8")
    metadata = {
        "pr_number": pr,
        "title": "fixture",
        "author": "tester",
        "head_repo": "tester/repo",
        "head_sha": "abc123",
        "leaderboard_name": name,
        "leaderboard_score": 0.2,
        "created_at": "2026-05-04T00:00:00Z",
        "closed_at": "2026-05-04T01:00:00Z",
        "merged_at": None,
        "additions": 1,
        "deletions": 0,
        "changed_files": 1,
    }
    (intake / "pr_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (intake / "archive_provenance.json").write_text(
        json.dumps({"status": "downloaded"}),
        encoding="utf-8",
    )
    with ZipFile(intake / "archive.zip", "w") as zf:
        zf.writestr("0.bin", b"\xfffixture-payload")
    return intake


def test_build_row_records_archive_source_and_missing_proofs(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    _write_pr_fixture(tmp_path, 100)

    row = mod.build_row(100, tmp_path)

    assert row["pr"] == 100
    assert row["leaderboard_name"] == "hnerv_fixture"
    assert row["archive"]["present"] is True
    assert row["archive"]["is_zip"] is True
    assert row["archive"]["members"][0]["name"] == "0.bin"
    assert row["archive"]["members"][0]["prefix_hex"].startswith("ff")
    assert row["source"]["submission_dir_present"] is True
    assert row["source"]["key_files"]["inflate_py"] is not None
    assert row["source"]["key_files"]["compress_sh"] is not None
    assert row["ready_for_stack_atom"] is False
    assert "exact_cuda_replay_missing" in row["missing_proofs"]
    assert "decode_reencode_parity_proof_required" in row["missing_proofs"]
    assert "compress_to_archive_1to1_reproduction_required" in row["missing_proofs"]
    assert row["leaderboard_replay_drift"]["status"] == "no_same_archive_score_to_compare"


def test_pr102_prefers_corrected_zero_byte_archive(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    stale_intake = _write_pr_fixture(tmp_path, 102, name="hnerv_lc_v2_scale095_rplus1")
    canonical_intake = (
        tmp_path
        / "experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/"
        / "public_pr102_intake_20260507_auto"
    )
    canonical_source = canonical_intake / "source/submissions/hnerv_lc_v2_scale095_rplus1"
    canonical_source.mkdir(parents=True)
    (canonical_source / "inflate.py").write_text("print('correct runtime')\n", encoding="utf-8")
    (canonical_source / "compress.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    metadata = json.loads((stale_intake / "pr_metadata.json").read_text(encoding="utf-8"))
    (canonical_intake / "pr_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (canonical_intake / "archive_provenance.json").write_text(
        json.dumps({"status": "downloaded"}),
        encoding="utf-8",
    )
    corrected_dir = canonical_intake.parents[0]
    corrected_archive = canonical_intake / "archive.zip"
    with ZipFile(corrected_archive, "w") as zf:
        zf.writestr("0.bin", b"correct-pr102")
    manifest = {
        "pr": {
            "number": 102,
            "title": "hnerv_lc_v2_scale095_rplus1 submission",
            "author": "EthanYangTW",
            "head_repo": "EthanYangTW/repo",
            "head_sha": "abc",
        },
        "canonical_archive": {
            "local_path": (
                "experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/"
                "public_pr102_intake_20260507_auto/archive.zip"
            )
        }
    }
    (corrected_dir / "CUSTODY_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    row = mod.build_row(102, tmp_path)

    assert row["archive_basis"] == "pr102_zero_byte_tuning_corrected_archive"
    assert row["leaderboard_name"] == "hnerv_lc_v2_scale095_rplus1"
    assert row["leaderboard_score"] == 0.195
    assert row["archive"]["sha256"] == mod._sha256_file(corrected_archive)
    assert row["archive"]["bytes"] == corrected_archive.stat().st_size
    assert "public_pr102_hnerv_lc_v2_scale095_rplus1_custody" in row["intake"]["dir"]
    assert row["source"]["key_files"]["inflate_py"] is not None


def test_pr107_includes_pr98_apogee_exact_eval_alias(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    _write_pr_fixture(tmp_path, 107, name="apogee")
    eval_dir = tmp_path / "experiments/results/lightning_batch/exact_eval_public_pr98_hnerv_adapter_t4"
    eval_dir.mkdir(parents=True)
    with ZipFile(eval_dir / "archive.zip", "w") as zf:
        zf.writestr("0.bin", b"apogee")
    (eval_dir / "auth_eval.log").write_text("score 0.229\n", encoding="utf-8")

    row = mod.build_row(107, tmp_path)

    eval_dirs = [artifact["dir"] for artifact in row["exact_eval_artifacts"]]
    assert any("exact_eval_public_pr98_hnerv_adapter_t4" in path for path in eval_dirs)


def test_exact_eval_artifact_parses_result_json_from_legacy_log(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    _write_pr_fixture(tmp_path, 100)
    eval_dir = tmp_path / "experiments/results/lightning_batch/exact_eval_public_pr100_cuda"
    eval_dir.mkdir(parents=True)
    archive_src = tmp_path / "experiments/results/public_pr_intake_full/public_pr100_intake_20260505_auto/archive.zip"
    (eval_dir / "archive.zip").write_bytes(archive_src.read_bytes())
    payload = {
        "canonical_score": 0.22826947142244708,
        "avg_posenet_dist": 0.00017198,
        "avg_segnet_dist": 0.00067623,
        "archive_size_bytes": archive_src.stat().st_size,
        "provenance": {
            "device": "cuda",
            "archive_sha256": mod._sha256_file(archive_src),
            "inflate_runtime_manifest": {"runtime_tree_sha256": "runtime-sha"},
        },
    }
    (eval_dir / "auth_eval.log").write_text(
        "prefix\nRESULT_JSON: " + json.dumps(payload) + "\n",
        encoding="utf-8",
    )

    row = mod.build_row(100, tmp_path)

    artifact = row["exact_eval_artifacts"][0]
    assert artifact["score"] == 0.22826947142244708
    assert artifact["score_basis"] == "canonical_score"
    assert artifact["device"] == "cuda"
    assert artifact["structured_result_kind"] == "embedded_result_json_log"
    assert row["exact_eval_summary"]["same_archive_scored_eval_count"] == 1
    assert row["exact_eval_summary"]["same_archive_structured_json_eval_count"] == 0
    assert row["exact_eval_summary"]["same_archive_embedded_result_json_eval_count"] == 1
    assert "same_archive_structured_exact_eval_json_missing" in row["missing_proofs"]
    drift = row["leaderboard_replay_drift"]
    assert drift["status"] == "leaderboard_mismatches_same_archive_cuda_replay"
    assert drift["identity_basis"] == ["archive_sha256", "device", "runtime_tree_sha256"]
    assert drift["local_scores_by_device"]["cuda"][0]["score"] == 0.22826947142244708
    assert (
        drift["local_scores_by_device"]["cuda"][0]["replay_identity"]["runtime_tree_sha256"]
        == "runtime-sha"
    )


def test_structured_json_must_match_source_archive(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    intake = _write_pr_fixture(tmp_path, 103, name="hnerv_lc_ac")
    source_archive = intake / "archive.zip"
    eval_root = tmp_path / "experiments/results/lightning_batch"
    same_archive_dir = eval_root / "exact_eval_public_pr103_same_archive"
    same_archive_dir.mkdir(parents=True)
    (same_archive_dir / "archive.zip").write_bytes(source_archive.read_bytes())
    payload = {
        "canonical_score": 0.227765,
        "provenance": {
            "device": "cuda",
            "archive_sha256": mod._sha256_file(source_archive),
            "inflate_runtime_manifest": {"runtime_tree_sha256": "same-runtime"},
        },
    }
    (same_archive_dir / "auth_eval.log").write_text(
        "RESULT_JSON: " + json.dumps(payload) + "\n",
        encoding="utf-8",
    )
    repack_dir = eval_root / "pr103_repack_with_json"
    repack_dir.mkdir(parents=True)
    with ZipFile(repack_dir / "archive.zip", "w") as zf:
        zf.writestr("0.bin", b"different-repack")
    repack_payload = {
        "canonical_score": 0.208,
        "provenance": {
            "device": "cuda",
            "archive_sha256": mod._sha256_file(repack_dir / "archive.zip"),
            "inflate_runtime_manifest": {"runtime_tree_sha256": "repack-runtime"},
        },
    }
    (repack_dir / "contest_auth_eval.json").write_text(
        json.dumps(repack_payload),
        encoding="utf-8",
    )

    row = mod.build_row(103, tmp_path)

    assert row["exact_eval_summary"]["total_eval_dir_count"] == 2
    assert row["exact_eval_summary"]["same_archive_scored_eval_count"] == 1
    assert row["exact_eval_summary"]["same_archive_structured_json_eval_count"] == 0
    assert row["exact_eval_summary"]["same_archive_embedded_result_json_eval_count"] == 1
    assert "structured_exact_eval_json_missing" in row["missing_proofs"]
    assert "same_archive_structured_exact_eval_json_missing" in row["missing_proofs"]


def test_pr103_schema_manifest_closes_decode_reencode_but_not_compressor(
    tmp_path: pathlib.Path,
) -> None:
    mod = _load_tool_module()
    intake = _write_pr_fixture(tmp_path, 103, name="hnerv_lc_ac")
    archive = intake / "archive.zip"
    manifest_dir = tmp_path / "experiments/results/hnerv_pr103_lc_ac_schema_20260507_codex"
    manifest_dir.mkdir(parents=True)
    manifest = {
        "ready_for_schema_review": True,
        "source_archive": {"sha256": mod._sha256_file(archive), "bytes": archive.stat().st_size},
        "merged_arithmetic_stream": {
            "source_bytes": 153856,
            "source_sha256": "08f0",
            "reencoded_bytes": 153856,
            "reencoded_sha256": "08f0",
            "decoded_symbol_count": 237561,
            "reencoded_byte_identical": True,
        },
        "dispatch_blockers": ["candidate_archive_missing"],
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    row = mod.build_row(103, tmp_path)

    assert "decode_reencode_parity_proof_required" not in row["missing_proofs"]
    assert "compress_to_archive_1to1_reproduction_required" in row["missing_proofs"]
    assert row["binary_understanding"]["wire_grammar_status"] == "fixed_pr103_lc_ac_section_layout"
    assert (
        row["binary_understanding"]["decode_reencode_parity_status"]
        == "merged_arithmetic_stream_byte_identical"
    )
    assert row["binary_understanding"]["merged_arithmetic_stream"]["decoded_symbol_count"] == 237561
    assert "structured_exact_eval_json_missing" in row["missing_proofs"]


def test_build_ledger_and_markdown_are_non_scoring(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    _write_pr_fixture(tmp_path, 100)
    _write_pr_fixture(tmp_path, 101)

    ledger = mod.build_ledger(
        tmp_path,
        pr_numbers=(100, 101),
        created_at_utc="2026-05-07T20:00:00Z",
    )
    md = mod.render_markdown(ledger)

    assert ledger["score_claim"] is False
    assert ledger["summary"]["row_count"] == 2
    assert ledger["summary"]["missing_decode_reencode_count"] == 2
    assert ledger["summary"]["leaderboard_replay_drift_count"] == 0
    assert ledger["summary"]["missing_same_archive_structured_json_count"] == 2
    assert ledger["summary"]["same_archive_embedded_result_json_count"] == 0
    assert "PR100-PR107 reproduction and deconstruction ledger" in md
    assert "hnerv_fixture" in md


def test_write_outputs_is_deterministic_with_fixed_timestamp(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    _write_pr_fixture(tmp_path, 100)
    ledger = mod.build_ledger(
        tmp_path,
        pr_numbers=(100,),
        created_at_utc="2026-05-07T20:00:00Z",
    )
    out_a = tmp_path / "a" / "ledger.json"
    out_b = tmp_path / "b" / "ledger.json"
    md_a = tmp_path / "a" / "ledger.md"
    md_b = tmp_path / "b" / "ledger.md"

    mod.write_outputs(ledger, out_a, md_a)
    mod.write_outputs(ledger, out_b, md_b)

    assert out_a.read_text(encoding="utf-8") == out_b.read_text(encoding="utf-8")
    assert md_a.read_text(encoding="utf-8") == md_b.read_text(encoding="utf-8")
