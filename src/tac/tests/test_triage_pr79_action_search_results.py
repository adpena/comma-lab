# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "triage_pr79_action_search_results.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("triage_pr79_action_search_results", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_zip(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("p", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_candidate_manifest(*, archive: Path, sha: str, bytes_: int) -> dict[str, Any]:
    return {
        "action_record_accounting": {
            "duplicate_pair_tile_accounting": {
                "duplicate_pair_tile_record_count": 0,
                "total_records": 2,
                "unique_pair_tile_count": 2,
            },
            "raw_output_parity_requirement": {
                "required": False,
                "satisfied_by_runtime_parse_validation": True,
            },
            "record_order": {
                "encoder_reorders_records": False,
                "repacked_matches_source_order": True,
            },
        },
        "break_even_math": {
            "versus_pr79": {
                "archive_byte_delta": -10,
                "break_even_component_improvement": 0.0,
            },
            "versus_pr79_s2": {
                "archive_byte_delta": -5,
                "break_even_component_improvement": 0.0,
            },
        },
        "no_op_detection": {
            "archive_sha_equal_to_source": False,
            "payload_sha_equal_to_source": False,
            "status": "decoded_action_semantics_preserved_action_bytes_changed",
        },
        "output_archive": {
            "bytes": bytes_,
            "path": str(archive),
            "sha256": sha,
        },
        "score_claim": False,
        "stream_packing": {
            "action_codec": "S2_split_meta_delta_brotli_adaptive_arithmetic_actions",
        },
    }


def test_triage_marks_non_noop_action_archive_as_parity_gated(tmp_path: Path) -> None:
    module = _load_module()
    result_dir = tmp_path / "lane"
    archive = result_dir / "results" / "pr79" / "archive_optimized.zip"
    sha = _write_zip(archive, b"x" * 80)
    (archive.parent / "manifest.json").write_text(
        json.dumps(_strict_candidate_manifest(archive=archive, sha=sha, bytes_=archive.stat().st_size))
    )

    summary = module.triage_results(
        result_dir,
        output_json=tmp_path / "triage.json",
        frontier_score=0.32,
        frontier_bytes=1000,
        frontier_sha256="f" * 64,
        target_score=0.31,
    )

    row = summary["candidates"][0]
    assert row["archive_sha256"] == sha
    assert row["archive_name"] == "archive_optimized.zip"
    assert row["byte_delta_vs_frontier"] < 0
    assert row["score_claim"] is False
    assert row["preflight_guards"]["passed"] is True
    assert row["dispatchable_after_parity_gate"] is True
    assert summary["dispatchable_after_parity_gate_count"] == 1


def test_triage_fails_closed_without_strict_modal_harvest_guards(tmp_path: Path) -> None:
    module = _load_module()
    result_dir = tmp_path / "lane"
    archive = result_dir / "archive_optimized.zip"
    _write_zip(archive, b"candidate")
    (result_dir / "summary.json").write_text(
        json.dumps(
            {
                "score_claim": False,
                "archive_optimized.zip": {
                    "bytes": archive.stat().st_size,
                    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                },
            }
        )
    )

    summary = module.triage_results(
        result_dir,
        output_json=tmp_path / "triage.json",
        frontier_score=0.32,
        frontier_bytes=1000,
        frontier_sha256="f" * 64,
        target_score=0.31,
    )

    row = summary["candidates"][0]
    assert row["dispatchable_after_parity_gate"] is False
    assert row["preflight_guards"]["archive_identity"]["matches_actual"] is True
    assert row["preflight_guards"]["failed"] == [
        "s2_packing_or_explicit_reason",
        "no_op_detection",
        "duplicate_pair_tile_accounting",
        "break_even_math_vs_pr79_and_s2",
    ]


def test_triage_requires_raw_output_parity_requirement_when_duplicates_exist(
    tmp_path: Path,
) -> None:
    module = _load_module()
    result_dir = tmp_path / "lane"
    archive = result_dir / "probe_archive.zip"
    sha = _write_zip(archive, b"candidate")
    manifest = _strict_candidate_manifest(
        archive=archive,
        sha=sha,
        bytes_=archive.stat().st_size,
    )
    manifest["action_record_accounting"]["duplicate_pair_tile_accounting"][
        "duplicate_pair_tile_record_count"
    ] = 1
    manifest["action_record_accounting"].pop("raw_output_parity_requirement")
    (result_dir / "manifest.json").write_text(json.dumps(manifest))

    summary = module.triage_results(
        result_dir,
        output_json=tmp_path / "triage.json",
        frontier_score=0.32,
        frontier_bytes=1000,
        frontier_sha256="f" * 64,
        target_score=0.31,
    )

    row = summary["candidates"][0]
    assert row["dispatchable_after_parity_gate"] is False
    assert row["preflight_guards"]["action_record_accounting"][
        "raw_output_parity_requirement_required"
    ] is True
    assert "duplicate_pair_tile_accounting" in row["preflight_guards"]["failed"]


def test_default_frontier_tracks_pr79_s2_exact_t4() -> None:
    module = _load_module()

    assert module.DEFAULT_FRONTIER_SCORE == 0.31453355357318635
    assert module.DEFAULT_FRONTIER_BYTES == 277_321
    assert (
        module.DEFAULT_FRONTIER_SHA256
        == "5740aca7e255b00093154eb1823b5b6207d8795f8eb287d35758c4cda438ec68"
    )


def test_triage_rejects_base_and_score_claim_artifacts(tmp_path: Path) -> None:
    module = _load_module()
    result_dir = tmp_path / "lane"
    base = result_dir / "archive.zip"
    candidate = result_dir / "probe_archive.zip"
    frontier_sha = _write_zip(base, b"frontier")
    _write_zip(candidate, b"candidate")
    (result_dir / "probe_archive_manifest.json").write_text(
        json.dumps(
            {
                "score_claim": True,
                "source_archive_sha256": frontier_sha,
            }
        )
    )

    summary = module.triage_results(
        result_dir,
        output_json=tmp_path / "triage.json",
        frontier_score=0.32,
        frontier_bytes=base.stat().st_size,
        frontier_sha256=frontier_sha,
        target_score=0.31,
    )

    by_name = {row["archive_name"]: row for row in summary["candidates"]}
    assert by_name["archive.zip"]["dispatchable_after_parity_gate"] is False
    assert by_name["archive.zip"]["no_op_vs_frontier"] is True
    assert by_name["probe_archive.zip"]["dispatchable_after_parity_gate"] is False
    assert by_name["probe_archive.zip"]["manifest_score_claim"] is True
    assert summary["dispatchable_after_parity_gate_count"] == 0
