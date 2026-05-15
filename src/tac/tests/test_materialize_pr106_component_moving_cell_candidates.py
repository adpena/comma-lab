# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from tac.packet_compiler.pr106_sidecar_packet import (
    PR106_NO_OP_DIM,
    decode_pr106_sidecar_packet_dim_delta,
    parse_pr106_sidecar_packet,
    read_single_stored_member_archive,
)
from tac.repo_io import read_json, write_json
from tac.tests.tool_loader import load_repo_tool

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    return load_repo_tool(
        REPO_ROOT,
        "tools/materialize_pr106_component_moving_cell_candidates.py",
        "materialize_pr106_component_moving_cell_candidates_test",
    )


def _stored_zip(path: Path, payload: bytes = b"pr106-payload") -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        zf.writestr(info, payload)


def _plan(source_archive: Path) -> dict[str, object]:
    with zipfile.ZipFile(source_archive) as zf:
        payload = zf.read("0.bin")
    payload_sha = hashlib.sha256(payload).hexdigest()
    archive_sha = hashlib.sha256(source_archive.read_bytes()).hexdigest()
    cells = [
        {
            "rank": 1,
            "row_idx": 5,
            "pair_idx": 5,
            "frame_slot": None,
            "cell_id": "latent_sidecar:row5:candidate100",
            "candidate": {"dim": 24, "delta_q": 2},
            "component_score_delta_no_rate": -0.7,
            "net_score_delta_charged": -0.699,
            "false_authority": {"score_claim": False},
        },
        {
            "rank": 2,
            "row_idx": 7,
            "pair_idx": 7,
            "frame_slot": None,
            "cell_id": "latent_sidecar:row7:candidate21",
            "candidate": {"dim": 5, "delta_q": -2},
            "component_score_delta_no_rate": -0.6,
            "net_score_delta_charged": -0.599,
            "false_authority": {"score_claim": False},
        },
        {
            "rank": 3,
            "row_idx": 5,
            "pair_idx": 5,
            "frame_slot": None,
            "cell_id": "latent_sidecar:row5:candidate65",
            "candidate": {"dim": 16, "delta_q": -2},
            "component_score_delta_no_rate": -0.5,
            "net_score_delta_charged": -0.499,
            "false_authority": {"score_claim": False},
        },
    ]
    return {
        "schema": "pr106_component_moving_cell_plan_v1",
        "label": "unit_component_cells",
        "kind": "latent_sidecar",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "axis_labels": {"source_score_axis_label": "[provider-CUDA:test]"},
        "source_custody": {
            "local_source_archive": {
                "path": str(source_archive),
                "sha256": archive_sha,
                "member_name": "0.bin",
                "member_sha256": payload_sha,
            },
            "manifest_source_zero_bin_sha256": payload_sha,
        },
        "top_cells": cells,
        "dispatch_blockers": ["planning_artifact_only"],
    }


def test_materializes_single_and_prefix_component_cell_archives(tmp_path: Path) -> None:
    module = _load_tool()
    source_archive = tmp_path / "source.zip"
    _stored_zip(source_archive)
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, _plan(source_archive))

    summary = module.materialize_from_plan(
        plan_path=plan_path,
        source_archive=None,
        output_dir=tmp_path / "out",
        singles=2,
        prefixes=[2],
    )

    assert summary["score_claim"] is False
    assert summary["candidate_count"] == 3
    assert "requires_paired_contest_cuda_auth_eval_on_materialized_archive" in summary[
        "dispatch_blockers"
    ]
    single = read_json(
        tmp_path
        / "out"
        / "latent_sidecar_row5_candidate100"
        / "candidate_manifest.json"
    )
    assert single["archive"]["bytes"] > source_archive.stat().st_size
    member = read_single_stored_member_archive(
        (tmp_path / "out" / "latent_sidecar_row5_candidate100" / "archive.zip").read_bytes()
    )
    packet = parse_pr106_sidecar_packet(member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(packet)
    assert int(dims[5]) == 24
    assert int(deltas[5]) == 2
    assert int(dims[7]) == PR106_NO_OP_DIM
    assert "component_cells_are_proxy_score_table_evidence" in single["dispatch_blockers"]

    prefix = read_json(tmp_path / "out" / "prefix_top_2" / "candidate_manifest.json")
    assert [cell["cell_id"] for cell in prefix["applied_cells"]] == [
        "latent_sidecar:row5:candidate100",
        "latent_sidecar:row7:candidate21",
    ]
    member = read_single_stored_member_archive(
        (tmp_path / "out" / "prefix_top_2" / "archive.zip").read_bytes()
    )
    packet = parse_pr106_sidecar_packet(member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(packet)
    assert int(dims[5]) == 24
    assert int(deltas[5]) == 2
    assert int(dims[7]) == 5
    assert int(deltas[7]) == -2
    assert prefix["ready_for_exact_eval_dispatch"] is False


def test_prefix_materialization_skips_duplicate_pair_cells(tmp_path: Path) -> None:
    module = _load_tool()
    source_archive = tmp_path / "source.zip"
    _stored_zip(source_archive)
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, _plan(source_archive))

    summary = module.materialize_from_plan(
        plan_path=plan_path,
        source_archive=source_archive,
        output_dir=tmp_path / "out",
        singles=0,
        prefixes=[3],
    )

    assert summary["candidate_count"] == 1
    manifest = read_json(tmp_path / "out" / "prefix_top_3" / "candidate_manifest.json")
    assert manifest["skipped_duplicate_pair_cells"] == ["latent_sidecar:row5:candidate65"]
    assert [cell["cell_id"] for cell in manifest["applied_cells"]] == [
        "latent_sidecar:row5:candidate100",
        "latent_sidecar:row7:candidate21",
    ]
    assert [cell["cell_id"] for cell in manifest["requested_cells"]] == [
        "latent_sidecar:row5:candidate100",
        "latent_sidecar:row7:candidate21",
        "latent_sidecar:row5:candidate65",
    ]
    member = read_single_stored_member_archive(
        (tmp_path / "out" / "prefix_top_3" / "archive.zip").read_bytes()
    )
    packet = parse_pr106_sidecar_packet(member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(packet)
    assert int(dims[5]) == 24
    assert int(deltas[5]) == 2


def test_source_payload_mismatch_fails_closed(tmp_path: Path) -> None:
    module = _load_tool()
    source_archive = tmp_path / "source.zip"
    _stored_zip(source_archive, payload=b"expected")
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, _plan(source_archive))
    bad_archive = tmp_path / "bad.zip"
    _stored_zip(bad_archive, payload=b"changed")

    try:
        module.materialize_from_plan(
            plan_path=plan_path,
            source_archive=bad_archive,
            output_dir=tmp_path / "out",
            singles=1,
            prefixes=[],
        )
    except ValueError as exc:
        assert "source_payload_sha256_mismatch" in str(exc)
    else:  # pragma: no cover - easier failure message
        raise AssertionError("source mismatch should fail closed")
