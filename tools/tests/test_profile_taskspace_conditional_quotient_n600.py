# SPDX-License-Identifier: MIT
"""CLI-boundary tests only; no subset result is scientific evidence."""

from __future__ import annotations

import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

TOOL_PATH = Path(__file__).resolve().parents[1] / "profile_taskspace_conditional_quotient_n600.py"
SPEC = importlib.util.spec_from_file_location("profile_taskspace_conditional_quotient_n600", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def _config(tmp_path: Path) -> Path:
    c1_root = tmp_path / "c1"
    c1_root.mkdir()
    prepare = c1_root / "prepare_receipt.json"
    prepare.write_bytes(b"fixture")
    digest = "a" * 64
    identity_checkpoints = []
    for index in range(38):
        start = index * 16
        stop = min(start + 16, 600)
        identity_checkpoints.append(
            {
                "path": f"/fixture/full_p_camera_identity/batch_{start:04d}_{stop:04d}.json",
                "expected_sha256": digest,
                "local_pair_range": [start, stop],
            }
        )
    value = {
        "schema": tool.CLI_CONFIG_SCHEMA,
        "profile": {
            "schema": tool.ConditionalQuotientProfileConfigV1(
                pair_count=600,
                chunk_pairs=12,
            ).as_mapping()["schema"],
            "pair_count": 600,
            "chunk_pairs": 12,
            "scorer_hw": [384, 512],
            "channels": 3,
            "resume": True,
            "test_only_small_fixture": False,
            "allow_local_storage": False,
        },
        "work_root": "/Volumes/VertigoDataTier/pact/fixture-not-launched",
        "fresh_v15_derivation": {
            "schema": tool.FRESH_V15_DERIVATION_SCHEMA,
            "expected_run_id": "fixture-fresh-v15-run",
            "compile_receipt": {
                "path": "/fixture/fresh/compile_receipt.json",
                "expected_sha256": digest,
                "expected_schema": tool.FRESH_V15_RECEIPT_SCHEMA,
            },
            "source_config": {
                "path": "/fixture/fresh/source_config.json",
                "expected_sha256": digest,
                "expected_rfc8785_sha256": digest,
            },
            "adjacent_archive": {
                "path": "/fixture/fresh/v15.zip",
                "expected_bytes": tool.FRESH_V15_ARCHIVE_BYTES,
                "expected_sha256": tool.FRESH_V15_ARCHIVE_SHA256,
            },
            "producer_sources": [
                {
                    "path": "fixture_producer.py",
                    "expected_bytes": 1,
                    "expected_sha256": digest,
                }
            ],
            "receiver_checkpoint": {
                "path": "/fixture/fresh/stage_checkpoints/02_receiver_closed_archive.json",
                "expected_sha256": digest,
            },
            "identity_checkpoints": identity_checkpoints,
            "identity_digest_chain_sha256": digest,
        },
        "c1_root": {
            "path": str(c1_root),
            "prepare_receipt_sha256": tool._sha256(b"fixture"),
        },
        "selected_plane_geometry_custody": {
            "path": "/fixture/custody.json",
            "expected_sha256": digest,
        },
        "canonical_batch16_debt_receipt": {
            "path": "/fixture/canonical_batch16.json",
            "expected_sha256": digest,
        },
        "independent_batch16_replay_receipt": {
            "path": "/fixture/g54.json",
            "expected_sha256": digest,
        },
        "fresh_teacher_receipt": {
            "path": "/fixture/fresh.json",
            "expected_sha256": digest,
        },
        "frontier_pointer": {
            "path": "/fixture/frontier.json",
            "expected_sha256": digest,
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fresh_derivation_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    monkeypatch.setattr(tool, "REPO_ROOT", tmp_path)
    run_root = tmp_path / "fresh-v15-run"
    checkpoint_root = run_root / "stage_checkpoints" / "full_p_camera_identity"
    checkpoint_root.mkdir(parents=True)
    producer_identities = []
    producer_records = []
    for index, payload in enumerate((b"producer-a", b"producer-b")):
        relative_path = f"producer_{index}.py"
        producer_path = tmp_path / relative_path
        producer_path.write_bytes(payload)
        digest = tool._sha256(payload)
        producer_identities.append(
            {
                "path": relative_path,
                "expected_bytes": len(payload),
                "expected_sha256": digest,
            }
        )
        producer_records.append(
            {
                "path": relative_path,
                "bytes": len(payload),
                "sha256": digest,
            }
        )

    run_id = "fixture-fresh-v15-run"
    source_config = {
        "schema": "FixtureFreshV15ConfigV1",
        "run_id": run_id,
        "initial_step": 32.0,
        "score_claim": False,
    }
    source_config_path = tmp_path / "source_config.json"
    source_config_path.write_text(json.dumps(source_config), encoding="utf-8")
    typed_config_sha = tool._sha256(tool.rfc8785_canonicalize(source_config))

    archive_path = run_root / "fresh.not_a_candidate.zip.receipt-bytes"
    archive_payload = b"x" * tool.FRESH_V15_ARCHIVE_BYTES
    archive_path.write_bytes(archive_payload)
    archive_sha = tool._sha256(archive_payload)
    monkeypatch.setattr(tool, "FRESH_V15_ARCHIVE_SHA256", archive_sha)

    identity_checkpoints = []
    digest_material = []
    for index in range(38):
        start = index * 16
        stop = min(start + 16, 600)
        camera_digest = tool._sha256(f"camera-{index}".encode())
        digest_material.append(camera_digest + camera_digest)
        row = {
            "schema": tool.FRESH_V15_IDENTITY_CHECKPOINT_SCHEMA,
            "typed_config_sha256": typed_config_sha,
            "local_pair_range": [start, stop],
            "base_camera_sha256": camera_digest,
            "final_camera_sha256": camera_digest,
            "byte_identical": True,
            "camera_bytes_released_after_compare": True,
            "score_claim": False,
        }
        checkpoint_path = checkpoint_root / f"batch_{start:04d}_{stop:04d}.json"
        checkpoint_path.write_bytes(tool._canonical_json(row))
        identity_checkpoints.append(
            {
                "path": str(checkpoint_path),
                "expected_sha256": tool.sha256_file(checkpoint_path),
                "local_pair_range": [start, stop],
            }
        )
    identity_chain = tool._sha256("".join(digest_material).encode("ascii"))

    receiver_checkpoint_path = run_root / "stage_checkpoints" / "02_receiver_closed_archive.json"
    receiver_checkpoint = {
        "schema": tool.FRESH_V15_RECEIVER_CHECKPOINT_SCHEMA,
        "typed_config_sha256": typed_config_sha,
        "archive": {
            "path": str(archive_path),
            "bytes": len(archive_payload),
            "sha256": archive_sha,
        },
        "score_claim": False,
    }
    receiver_checkpoint_path.write_bytes(tool._canonical_json(receiver_checkpoint))

    receipt_path = run_root / "compile_receipt.json"
    receipt = {
        "schema": tool.FRESH_V15_RECEIPT_SCHEMA,
        "run_id": run_id,
        "typed_config": source_config,
        "typed_config_sha256": typed_config_sha,
        "producer_custody": producer_records,
        "selected_candidate": "fixture-v15",
        "solved_template_ladder": [
            {
                "candidate": "fixture-v15",
                "archive_bytes": len(archive_payload),
                "archive_sha256": archive_sha,
                "receiver_custody": {
                    "archive_bytes": len(archive_payload),
                    "archive_sha256": archive_sha,
                    "score_claim": False,
                },
                "full_p_camera_identity": {
                    "pair_count": 600,
                    "batch_count": 38,
                    "batch_size": 16,
                    "all_camera_bytes_identical": True,
                    "digest_chain_sha256": identity_chain,
                },
                "score_claim": False,
            }
        ],
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    receipt_path.write_bytes(tool._canonical_json(receipt))
    return {
        "schema": tool.FRESH_V15_DERIVATION_SCHEMA,
        "expected_run_id": run_id,
        "compile_receipt": {
            "path": str(receipt_path),
            "expected_sha256": tool.sha256_file(receipt_path),
            "expected_schema": tool.FRESH_V15_RECEIPT_SCHEMA,
        },
        "source_config": {
            "path": str(source_config_path),
            "expected_sha256": tool.sha256_file(source_config_path),
            "expected_rfc8785_sha256": typed_config_sha,
        },
        "adjacent_archive": {
            "path": str(archive_path),
            "expected_bytes": len(archive_payload),
            "expected_sha256": archive_sha,
        },
        "producer_sources": producer_identities,
        "receiver_checkpoint": {
            "path": str(receiver_checkpoint_path),
            "expected_sha256": tool.sha256_file(receiver_checkpoint_path),
        },
        "identity_checkpoints": identity_checkpoints,
        "identity_digest_chain_sha256": identity_chain,
    }


def test_strict_archive_ceiling_arithmetic_matches_historical_ms1_coordinate() -> None:
    d_seg = Decimal("0.0001519690619574653")
    d_pose = Decimal("0.00010184327939026322")
    assert tool._largest_archive_below(target=Decimal("0.172"), d_seg=d_seg, d_pose=d_pose) == 187_562
    assert tool._largest_archive_below(target=Decimal("0.15"), d_seg=d_seg, d_pose=d_pose) == 154_522
    canonical_d_seg = Decimal("0.00015196058485243054")
    canonical_d_pose = Decimal("0.00010184347386600314")
    assert (
        tool._largest_archive_below(
            target=Decimal("0.172"),
            d_seg=canonical_d_seg,
            d_pose=canonical_d_pose,
        )
        == 187_563
    )
    assert (
        tool._largest_archive_below(
            target=Decimal("0.15"),
            d_seg=canonical_d_seg,
            d_pose=canonical_d_pose,
        )
        == 154_523
    )


def test_cli_config_is_closed_and_does_not_open_large_inputs(tmp_path: Path) -> None:
    path = _config(tmp_path)
    loaded = tool.load_cli_config(path)
    assert loaded["profile"]["pair_count"] == 600
    assert loaded["profile"]["test_only_small_fixture"] is False
    value = json.loads(path.read_text(encoding="utf-8"))
    value["invented"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(tool.ConditionalQuotientProfilerError, match="keys/schema"):
        tool.load_cli_config(path)


def test_frontier_pointer_requires_canonical_dynamic_selection_rule() -> None:
    pointer = {
        "effective_frontier": {
            "score": 0.172,
            "selection_rule": tool.EXPECTED_SELECTION_RULE,
        }
    }
    assert tool._frontier_score(pointer) == Decimal("0.172")
    pointer["effective_frontier"]["selection_rule"] = "hardcoded stale row"
    with pytest.raises(tool.ConditionalQuotientProfilerError, match="selection rule"):
        tool._frontier_score(pointer)


def test_fresh_v15_derivation_reopens_all_ordered_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    derivation = _fresh_derivation_fixture(tmp_path, monkeypatch)
    resolved = tool._resolve_fresh_v15_derivation(derivation)
    assert resolved.custody["derivation_proof_separate_from_archive_content_identity"] is True
    assert resolved.custody["historical_path_fallback_allowed"] is False
    full_p = resolved.custody["full_p_camera_identity"]
    assert full_p["batch_count"] == 38
    assert full_p["digest_chain_matches_receipt"] is True
    assert len(full_p["ordered_checkpoints"]) == 38
    assert all(row["score_claim"] is False for row in full_p["ordered_checkpoints"])
    assert all(row["live_rehashed"] is True for row in resolved.custody["producer_sources"])


def test_equal_archive_sha_does_not_replace_fresh_derivation_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    derivation = _fresh_derivation_fixture(tmp_path, monkeypatch)
    historical_root = tmp_path / "historical"
    historical_root.mkdir()
    historical_archive = historical_root / "same-content.zip"
    adjacent_archive = derivation["adjacent_archive"]
    historical_archive.write_bytes(Path(adjacent_archive["path"]).read_bytes())
    adjacent_archive["path"] = str(historical_archive)
    with pytest.raises(tool.ConditionalQuotientProfilerError, match="historical-path fallback"):
        tool._resolve_fresh_v15_derivation(derivation)


def test_fresh_v15_identity_digest_chain_refuses_checkpoint_value_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    derivation = _fresh_derivation_fixture(tmp_path, monkeypatch)
    identity = derivation["identity_checkpoints"][17]
    checkpoint_path = Path(identity["path"])
    row = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    row["final_camera_sha256"] = "f" * 64
    checkpoint_path.write_bytes(tool._canonical_json(row))
    identity["expected_sha256"] = tool.sha256_file(checkpoint_path)
    with pytest.raises(tool.ConditionalQuotientProfilerError, match="camera digests differ"):
        tool._resolve_fresh_v15_derivation(derivation)


def test_preflight_seals_zero_chunk_receipt_without_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_bytes(b"{}")
    config = tool.ConditionalQuotientProfileConfigV1(pair_count=600, chunk_pairs=12)
    sources = {
        "fixture": {
            "path": "fixture.py",
            "bytes": 1,
            "sha256": "a" * 64,
        }
    }
    prepared = tool._PreparedInputs(
        config_path=config_path,
        config=config,
        cli={},
        v15_path=tmp_path / "v15.zip",
        v15_payload=b"x",
        receiver=None,
        teacher=None,
        fresh={},
        binding={"fixture": "strictly prepared upstream"},
        implementation_sources=sources,
        config_sha256=tool.sha256_file(config_path),
        git_sha_start="b" * 40,
        work_root=tmp_path / "work",
    )
    monkeypatch.setattr(tool, "_prepare_inputs", lambda _path: prepared)
    monkeypatch.setattr(tool, "_implementation_sources", lambda: sources)
    monkeypatch.setattr(tool, "_git_head", lambda: "b" * 40)
    monkeypatch.setattr(
        tool,
        "storage_preflight",
        lambda *_args, **_kwargs: {
            "schema": "tac.c0b_semantic_quotient_storage_preflight.v1",
            "selected_tier": "test-local",
            "required_bytes": 1 << 30,
            "passed": True,
            "test_only_small_fixture": False,
            "allow_local_storage": False,
        },
    )
    receipt = tool.preflight(config_path)
    assert receipt["pair_rendering_started"] is False
    assert receipt["chunks_profiled"] == 0
    assert receipt["full_n600_launch_authorized_by_this_receipt"] is False
    assert receipt["pointer_mutation_performed"] is False
    assert receipt["launch_governance"]["status"] == "LAUNCH_NOT_PERFORMED"
    assert (prepared.work_root / "preflight_receipt.json").is_file()


def test_recursive_source_closure_includes_direct_and_transitive_dependencies() -> None:
    closure = tool._implementation_sources()
    assert "tools/profile_taskspace_conditional_quotient_n600.py" in closure
    assert "tools/build_c0b_semantic_quotient_archive.py" in closure
    assert "src/tac/optimization/direct_description_minimizer.py" in closure
    assert "src/tac/witness_dsl/v10_production_receiver.py" in closure
    assert tool.V15_RECEIVER_SOURCE_PATH in closure
    assert all(
        (tool.REPO_ROOT / row["path"]).is_file() and row["bytes"] > 0 and len(row["sha256"]) == 64
        for row in closure.values()
    )
