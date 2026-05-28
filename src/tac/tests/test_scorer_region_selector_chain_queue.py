from __future__ import annotations

import json
import zipfile
from pathlib import Path

from comma_lab.scheduler.scorer_region_selector_chain_queue import (
    SCORER_REGION_SELECTOR_CHAIN_CONTEXT_SCHEMA,
    SCORER_REGION_SELECTOR_CHAIN_REPORT_SCHEMA,
    build_scorer_region_selector_chain_context,
    build_scorer_region_selector_chain_queue,
    build_scorer_region_selector_chain_report,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.family_agnostic_materializers import ARCHIVE_ZIP_REPACK_SCHEMA
from tac.packet_compiler.feca_selector_reparameterize import (
    FECA_REPARAMETERIZATION_MANIFEST_SCHEMA,
)


def _write_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("0.bin", b"selector-stream")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _source_submission(tmp_path: Path) -> Path:
    submission = tmp_path / "submission"
    _write_zip(submission / "archive.zip")
    return submission


def _waterfill_work_order(tmp_path: Path) -> Path:
    path = tmp_path / "repair_budget_waterfill_work_order.json"
    _write_json(
        path,
        {
            "schema": "frontier_rate_attack_repair_budget_waterfill_work_order.v1",
            "repair_cascade_opportunity_rows": [
                {
                    "schema": "frontier_rate_attack_repair_cascade_opportunity_row.v1",
                    "cascade_id": "cascade_c_posenet_null_segnet_region_selector_codec",
                    "label": "Cascade C",
                    "pipeline_position": "P19+P18+P11",
                    "targeted_positions": [
                        {"position_id": "P19", "entropy_surface": "PoseNet"},
                        {"position_id": "P18", "entropy_surface": "SegNet"},
                        {"position_id": "P11", "entropy_surface": "selector"},
                    ],
                    "blockers": [],
                }
            ],
            **FALSE_AUTHORITY,
        },
    )
    return path


def test_chain_context_preserves_upstream_blockers_without_score_authority(
    tmp_path: Path,
) -> None:
    submission = _source_submission(tmp_path)
    work_order = _waterfill_work_order(tmp_path)
    parity = tmp_path / "parity.json"
    _write_json(parity, {"passed": True, **FALSE_AUTHORITY})

    context = build_scorer_region_selector_chain_context(
        repo_root=tmp_path,
        source_submission_dir=submission,
        source_waterfill_work_order=work_order,
        full_frame_inflate_parity_proof=parity,
    )

    assert context["schema"] == SCORER_REGION_SELECTOR_CHAIN_CONTEXT_SCHEMA
    assert context["p11_rate_anchor_can_run"] is True
    assert context["p18_p19_upstream_ready"] is False
    assert "p19_posenet_null_pairs_missing" in context["blockers"]
    assert "p18_segnet_region_masks_missing" in context["blockers"]
    assert context["score_claim"] is False
    assert context["ready_for_exact_eval_dispatch"] is False


def test_chain_queue_orders_context_selector_repack_report(
    tmp_path: Path,
) -> None:
    submission = _source_submission(tmp_path)
    work_order = _waterfill_work_order(tmp_path)
    parity = tmp_path / "parity.json"
    _write_json(parity, {"passed": True, **FALSE_AUTHORITY})

    queue = build_scorer_region_selector_chain_queue(
        repo_root=tmp_path,
        queue_id="chain_q",
        source_submission_dir=submission,
        output_root=tmp_path / "chain_out",
        source_waterfill_work_order=work_order,
        full_frame_inflate_parity_proof=parity,
        scales=(64,),
        alphas=(1,),
        codec_families=("fec10_adaptive_blend",),
    )

    steps = queue["experiments"][0]["steps"]
    assert [step["id"] for step in steps] == [
        "build_p18_p19_chain_context",
        "materialize_p11_selector_context_recode",
        "materialize_p15_archive_zip_repack",
        "emit_composed_chain_report",
    ]
    assert steps[1]["requires"] == ["build_p18_p19_chain_context"]
    assert steps[2]["requires"] == ["materialize_p11_selector_context_recode"]
    assert steps[3]["requires"] == ["materialize_p15_archive_zip_repack"]
    assert "--chain-parent-artifact" in steps[1]["command"]
    assert "archive_zip_repack_v1" in steps[2]["command"]
    assert queue["metadata"]["ready_for_exact_eval_dispatch"] is False


def test_chain_report_selects_repack_only_when_positive_and_receiver_closed(
    tmp_path: Path,
) -> None:
    context_path = tmp_path / "chain_context.json"
    selector_path = tmp_path / "selector.json"
    repack_path = tmp_path / "repack.json"
    context = {
        "schema": SCORER_REGION_SELECTOR_CHAIN_CONTEXT_SCHEMA,
        "chain_label": "cascade-c",
        "p18_p19_upstream_ready": True,
        "blockers": [],
        **FALSE_AUTHORITY,
    }
    selector = {
        "schema": FECA_REPARAMETERIZATION_MANIFEST_SCHEMA,
        "candidate_archive": {"path": "selector/archive.zip", "bytes": 90, "sha256": "a" * 64},
        "source_archive": {"path": "source/archive.zip", "bytes": 100, "sha256": "b" * 64},
        "selected_recode": {"saved_bytes": 10, "codec_family": "fec10_adaptive_blend"},
        "receiver_contract_satisfied": True,
        "readiness_blockers": [],
        **FALSE_AUTHORITY,
    }
    repack = {
        "schema": ARCHIVE_ZIP_REPACK_SCHEMA,
        "candidate_archive": {"path": "repack/archive.zip", "bytes": 86, "sha256": "c" * 64},
        "source_archive": {"path": "selector/archive.zip", "bytes": 90, "sha256": "a" * 64},
        "selected_repack": {"saved_bytes": 4, "strategy": "greedy", "plan_key": "deflated:9"},
        "receiver_contract_satisfied": True,
        "readiness_blockers": [],
        **FALSE_AUTHORITY,
    }
    _write_json(context_path, context)
    _write_json(selector_path, selector)
    _write_json(repack_path, repack)

    report = build_scorer_region_selector_chain_report(
        repo_root=tmp_path,
        chain_context=context,
        chain_context_path=context_path,
        selector_manifest=selector,
        selector_manifest_path=selector_path,
        repack_manifest=repack,
        repack_manifest_path=repack_path,
    )

    assert report["schema"] == SCORER_REGION_SELECTOR_CHAIN_REPORT_SCHEMA
    assert report["selected_local_survivor_stage"] == "P15_archive_zip_repack"
    assert report["cumulative_rate_saved_bytes_vs_source"] == 14
    assert report["selected_local_survivor_archive"]["sha256"] == "c" * 64
    assert report["blockers"] == report["readiness_blockers"]
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
