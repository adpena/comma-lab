from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Any

import brotli


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_pr79_next_byte_candidates.py"


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("pr79_next_byte_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _raw4(records: list[tuple[int, int, int]]) -> bytes:
    out = bytearray()
    for pair, tile, action in records:
        out += int(pair).to_bytes(2, "little") + bytes([tile, action])
    return bytes(out)


def _p3_payload(*, mask: bytes, renderer: bytes, actions: bytes, pose: bytes) -> bytes:
    mask_br = brotli.compress(mask, quality=0, lgwin=10)
    renderer_br = brotli.compress(renderer, quality=0, lgwin=10)
    actions_br = brotli.compress(actions, quality=0, lgwin=10)
    pose_br = brotli.compress(pose, quality=0, lgwin=10)
    return (
        b"P3"
        + struct.pack("<IHH", len(mask_br), len(renderer_br), len(actions_br))
        + mask_br
        + renderer_br
        + actions_br
        + pose_br
    )


def _stored_p(path: Path, payload: bytes, script: Any) -> None:
    path.write_bytes(script._zip_bytes(payload))  # noqa: SLF001


def test_next_byte_builder_emits_deterministic_dispatchable_rows(tmp_path: Path) -> None:
    script = _load_script()
    pr79_archive = tmp_path / "pr79.zip"
    pr77_archive = tmp_path / "pr77.zip"
    mask_archive = tmp_path / "mask_candidate.zip"
    mask_matrix = tmp_path / "mask_matrix.json"
    missing_s2 = tmp_path / "missing_s2.json"
    renderer = b"QZS3" + b"r" * 48
    source_mask = b"\x12\x00\x0a\x0a" + b"m" * 96
    candidate_mask = b"\x1aE\xdf\xa3candidate-mkv-mask-body"
    pose = b"QP1" + (5120).to_bytes(2, "little") + b"p" * 16
    pr79_actions = _raw4([(1, 2, 1), (4, 5, 2)])
    pr77_actions = _raw4([(1, 2, 1)])

    _stored_p(
        pr79_archive,
        _p3_payload(mask=source_mask, renderer=renderer, actions=pr79_actions, pose=pose),
        script,
    )
    _stored_p(
        pr77_archive,
        _p3_payload(mask=source_mask, renderer=renderer, actions=pr77_actions, pose=pose),
        script,
    )
    mask_payload = script._build_rpk1_payload(  # noqa: SLF001
        source_archive_sha256=script._sha256_file(pr79_archive),  # noqa: SLF001
        ordered_members=[
            ("renderer.bin", renderer),
            ("masks.mkv", candidate_mask),
            ("seg_tile_actions.bin", pr79_actions),
            ("optimized_poses.qp1", pose),
        ],
    )
    _stored_p(mask_archive, mask_payload, script)
    mask_matrix.write_text(
        json.dumps(
            {
                "recommendation": {
                    "candidate": {
                        "archive_bytes": mask_archive.stat().st_size,
                        "archive_path": str(mask_archive),
                        "archive_sha256": script._sha256_file(mask_archive),  # noqa: SLF001
                        "candidate_id": "fixture_mask53",
                        "exact_eval_ready_after_lane_claim": True,
                        "manifest_path": None,
                    }
                }
            }
        )
    )

    summary = script.build_candidates(
        pr79_archive=pr79_archive,
        pr77_archive=pr77_archive,
        mask_matrix=mask_matrix,
        s2_matrix=missing_s2,
        output_dir=tmp_path / "out",
        force=True,
        s2_frontier={
            "archive_bytes": 1_000,
            "archive_sha256": "f" * 64,
            "score": 1.0,
            "score_source": "fixture",
        },
    )

    assert summary["score_claim"] is False
    assert summary["no_remote_dispatch_performed"] is True
    assert summary["candidate_count"] == 6

    source = next(row for row in summary["candidates"] if row["candidate_id"] == "source_pr79_noop_control")
    assert source["no_op_detection"]["status"] == "byte_identical_noop"

    pr79_row = next(
        row for row in summary["candidates"]
        if row["candidate_id"] == "fixture_mask53__pr79_public_raw4_action_wire_br__stored_rpk1"
    )
    assert pr79_row["changed_members"]["preserved_renderer_and_pose"] is True
    assert pr79_row["changed_members"]["semantic_equalities_vs_pr79"]["seg_tile_actions"] is True
    assert pr79_row["dispatchability"]["exact_screen_dispatchable_after_lane_claim"] is True
    assert pr79_row["score_estimate_vs_current_s2_frontier"]["score_claim"] is False

    manifest = json.loads(Path(pr79_row["manifest_path"]).read_text())
    assert manifest["zip_profile"]["deterministic_rebuild_equal"] is True
    assert manifest["runtime_parse_validation"]["action_loader"]["compatible"] is True
    assert manifest["body_byte_profile"]["archive_bytes"] == pr79_row["archive_bytes"]
    assert manifest["body_byte_profile"]["parsed_members"]["masks.mkv"]["bytes"] == len(
        candidate_mask
    )
    assert manifest["provenance"]["source_pr79"]["archive_sha256"] == script._sha256_file(  # noqa: SLF001
        pr79_archive
    )


def test_s2_action_wire_in_rpk1_is_marked_not_direct_loader_compatible(tmp_path: Path) -> None:
    script = _load_script()
    wire = brotli.compress(b"S2" + bytes([1]) + (0).to_bytes(4, "little") + b"\x00")
    action = script.ActionWire(
        action_id="s2_fixture",
        source_label="fixture",
        member_name="seg_tile_actions.br",
        wire_bytes=wire,
        runtime_raw=brotli.decompress(wire),
        decoded_semantics=b"",
        direct_inflate_loader_compatible=False,
        compatibility_reason="fixture S2 parser-only",
        source_archive_sha256=None,
        source_payload_format="fixture",
    )

    members = {
        "seg_tile_actions.br": action.wire_bytes,
    }
    raw, loader = script._direct_action_semantics(  # noqa: SLF001
        members,
        unpacker=script._load_unpacker(),  # noqa: SLF001
    )

    assert raw is not None and raw.startswith(b"S2")
    assert loader["compatible"] is False
    assert "public-slice unpacker" in loader["reason"]


def test_next_byte_builder_extends_to_crf52_and_present_crf51_crf50_rows(
    tmp_path: Path,
) -> None:
    script = _load_script()
    pr79_archive = tmp_path / "pr79.zip"
    pr77_archive = tmp_path / "pr77.zip"
    mask_matrix = tmp_path / "mask_matrix.json"
    missing_s2 = tmp_path / "missing_s2.json"
    renderer = b"QZS3" + b"r" * 48
    source_mask = b"\x12\x00\x0a\x0a" + b"m" * 96
    pose = b"QP1" + (5120).to_bytes(2, "little") + b"p" * 16
    pr79_actions = _raw4([(1, 2, 1), (4, 5, 2)])
    pr77_actions = _raw4([(1, 2, 1)])

    _stored_p(
        pr79_archive,
        _p3_payload(mask=source_mask, renderer=renderer, actions=pr79_actions, pose=pose),
        script,
    )
    _stored_p(
        pr77_archive,
        _p3_payload(mask=source_mask, renderer=renderer, actions=pr77_actions, pose=pose),
        script,
    )

    rows: list[dict[str, Any]] = []
    for crf, gate_passed in [(53, True), (52, True), (51, True), (50, False)]:
        candidate_id = f"fixture_trust_region_save05k_crf{crf}"
        mask_archive = tmp_path / f"{candidate_id}.zip"
        candidate_mask = f"mask-crf{crf}".encode()
        mask_payload = script._build_rpk1_payload(  # noqa: SLF001
            source_archive_sha256=script._sha256_file(pr79_archive),  # noqa: SLF001
            ordered_members=[
                ("renderer.bin", renderer),
                ("masks.mkv", candidate_mask),
                ("seg_tile_actions.bin", pr79_actions),
                ("optimized_poses.qp1", pose),
            ],
        )
        _stored_p(mask_archive, mask_payload, script)
        rows.append(
            {
                "archive_bytes": mask_archive.stat().st_size,
                "archive_path": str(mask_archive),
                "archive_sha256": script._sha256_file(mask_archive),  # noqa: SLF001
                "candidate_id": candidate_id,
                "exact_eval_ready_after_lane_claim": crf in {53, 52},
                "manifest_path": None,
                "mask_stream": {
                    "wire_sha256": script._sha256_bytes(candidate_mask),  # noqa: SLF001
                },
                "plausibility_gate": {"passed": gate_passed},
                "runtime_preflight": {
                    "archive_validation_errors": [],
                    "candidate_mask_sha_matches_expected": True,
                    "non_mask_runtime_members_preserved": True,
                    "runtime_parser_ok": True,
                },
            }
        )
    mask_matrix.write_text(
        json.dumps(
            {
                "recommendation": {"candidate": rows[0]},
                "candidates": rows,
            }
        )
    )

    summary = script.build_candidates(
        pr79_archive=pr79_archive,
        pr77_archive=pr77_archive,
        mask_matrix=mask_matrix,
        s2_matrix=missing_s2,
        output_dir=tmp_path / "out",
        force=True,
        s2_frontier={
            "archive_bytes": 1_000,
            "archive_sha256": "f" * 64,
            "score": 1.0,
            "score_source": "fixture",
        },
    )

    source_ids = {row["candidate_id"] for row in summary["mask_candidate_sources"]}
    assert "fixture_trust_region_save05k_crf53" in source_ids
    assert "fixture_trust_region_save05k_crf52" in source_ids
    assert "fixture_trust_region_save05k_crf51" in source_ids
    assert "fixture_trust_region_save05k_crf50" not in source_ids
    assert summary["candidate_count"] == 16

    crf52_row = next(
        row
        for row in summary["candidates"]
        if row["candidate_id"]
        == "fixture_trust_region_save05k_crf52__pr79_public_raw4_action_wire_br__stored_rpk1"
    )
    assert crf52_row["dispatchability"]["exact_screen_dispatchable_after_lane_claim"] is True
    manifest = json.loads(Path(crf52_row["manifest_path"]).read_text())
    assert manifest["mask_candidate"]["mask_crf"] == 52
    assert manifest["runtime_parse_validation"]["action_semantics_parity"][
        "equal_pr79_decoded_actions"
    ] is True


def test_next_byte_builder_keeps_crf51_50_wave_non_dispatchable_without_ready_mask(
    tmp_path: Path,
) -> None:
    script = _load_script()
    pr79_archive = tmp_path / "pr79.zip"
    pr77_archive = tmp_path / "pr77.zip"
    mask_matrix = tmp_path / "mask_matrix.json"
    missing_s2 = tmp_path / "missing_s2.json"
    renderer = b"QZS3" + b"r" * 48
    source_mask = b"\x12\x00\x0a\x0a" + b"m" * 96
    pose = b"QP1" + (5120).to_bytes(2, "little") + b"p" * 16
    pr79_actions = _raw4([(1, 2, 1), (4, 5, 2)])
    pr77_actions = _raw4([(9, 8, 1)])

    _stored_p(
        pr79_archive,
        _p3_payload(mask=source_mask, renderer=renderer, actions=pr79_actions, pose=pose),
        script,
    )
    _stored_p(
        pr77_archive,
        _p3_payload(mask=source_mask, renderer=renderer, actions=pr77_actions, pose=pose),
        script,
    )

    rows: list[dict[str, Any]] = []
    for crf in [51, 50]:
        candidate_id = f"fixture_save05k_crf{crf}"
        mask_archive = tmp_path / f"{candidate_id}.zip"
        candidate_mask = f"mask-crf{crf}".encode()
        mask_payload = script._build_rpk1_payload(  # noqa: SLF001
            source_archive_sha256=script._sha256_file(pr79_archive),  # noqa: SLF001
            ordered_members=[
                ("renderer.bin", renderer),
                ("masks.mkv", candidate_mask),
                ("seg_tile_actions.bin", pr79_actions),
                ("optimized_poses.qp1", pose),
            ],
        )
        _stored_p(mask_archive, mask_payload, script)
        rows.append(
            {
                "archive_bytes": mask_archive.stat().st_size,
                "archive_path": str(mask_archive),
                "archive_sha256": script._sha256_file(mask_archive),  # noqa: SLF001
                "candidate_id": candidate_id,
                "exact_eval_ready_after_lane_claim": False,
                "manifest_path": None,
                "mask_stream": {
                    "wire_sha256": script._sha256_bytes(candidate_mask),  # noqa: SLF001
                },
                "plausibility_gate": {"passed": True},
                "runtime_preflight": {
                    "archive_validation_errors": [],
                    "candidate_mask_sha_matches_expected": True,
                    "non_mask_runtime_members_preserved": True,
                    "runtime_parser_ok": True,
                },
            }
        )
    mask_matrix.write_text(
        json.dumps(
            {
                "recommendation": {
                    "candidate": {
                        **rows[0],
                        "candidate_id": "fixture_save05k_crf53",
                    }
                },
                "candidates": rows,
            }
        )
    )

    summary = script.build_candidates(
        pr79_archive=pr79_archive,
        pr77_archive=pr77_archive,
        mask_matrix=mask_matrix,
        s2_matrix=missing_s2,
        output_dir=tmp_path / "out",
        force=True,
        s2_frontier={
            "archive_bytes": 1_000,
            "archive_sha256": "f" * 64,
            "score": 1.0,
            "score_source": "fixture",
        },
        mask_crfs=(51, 50),
        action_sources=("pr79",),
        candidate_families=("brotli_rpk1_flatpack",),
        include_source_control=False,
        require_action_parity_for_dispatch=True,
    )

    assert summary["build_options"]["mask_crfs"] == [51, 50]
    assert summary["candidate_count"] == 2
    assert all("crf53" not in row["candidate_id"] for row in summary["candidates"])
    assert {row["candidate_family"] for row in summary["candidates"]} == {
        "brotli_rpk1_flatpack"
    }
    assert all("__pr79_public_raw4_action_wire_br__" in row["candidate_id"] for row in summary["candidates"])
    assert all(
        row["changed_members"]["semantic_equalities_vs_pr79"]["seg_tile_actions"]
        for row in summary["candidates"]
    )
    assert all(
        not row["dispatchability"]["exact_screen_dispatchable_after_lane_claim"]
        for row in summary["candidates"]
    )
    assert all(
        "not exact-eval-ready" in row["dispatchability"]["reason"]
        for row in summary["candidates"]
    )
    manifest = json.loads(Path(summary["candidates"][0]["manifest_path"]).read_text())
    assert manifest["provenance"]["build_options"]["require_action_parity_for_dispatch"] is True
    assert manifest["body_byte_profile"]["top_level_brotli_flatpack"] is True
