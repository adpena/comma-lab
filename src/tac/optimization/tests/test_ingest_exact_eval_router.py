"""Tests for the V3 closed-loop compiler STEP 1: exact-eval -> typed candidate
evidence + the substrate next-action router (operator's 5+1 case decision tree).

NO-FAKE discipline: the router tests assert BEHAVIOR (case mapping flips with the
inputs), not constants; the ingest tests assert the emitted rows carry the exact
contest-score arithmetic and the rejection of a garbage base.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from tac.optimization.harvest_evidence import (
    ROUTE_ADD_SEG_BOUNDARY_ATLAS,
    ROUTE_BUILD_AUTHORITY_TRACE,
    ROUTE_LAUNCH_DENSE_CARRIER,
    ROUTE_PATCH_BRIDGE_NEXT,
    ROUTE_STRENGTHEN_POSE_CARRIER,
    route_substrate_next_action,
)

# ---------------------------------------------------------------------------
# Import the thin CLI module (tools/ is not a package).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TOOL_PATH = _REPO_ROOT / "tools" / "ingest_exact_eval_to_candidate.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("ingest_exact_eval_to_candidate", _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ingest_exact_eval_to_candidate"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Pure router: the 6 cases (A/B/C/D/E/F) + the never-auto-kill invariant.
# ---------------------------------------------------------------------------
def test_route_case_A_seg_descended_pose_sane() -> None:
    r = route_substrate_next_action(d_seg=0.05, d_pose=0.001)
    assert r["case"] == "A"
    assert r["route"] == ROUTE_LAUNCH_DENSE_CARRIER
    assert r["seg_state"] == "descended" and r["pose_state"] == "sane"


def test_route_case_B_seg_flat_pose_improved() -> None:
    # seg flat (>= collapse floor) but pose IMPROVED vs a baseline.
    r = route_substrate_next_action(
        d_seg=0.50, d_pose=0.30, baseline_d_pose=1.0
    )
    assert r["case"] == "B"
    assert r["route"] == ROUTE_ADD_SEG_BOUNDARY_ATLAS
    assert r["pose_state"] == "improved"


def test_route_case_C_seg_descended_pose_exploded() -> None:
    r = route_substrate_next_action(d_seg=0.05, d_pose=200.0)
    assert r["case"] == "C"
    assert r["route"] == ROUTE_STRENGTHEN_POSE_CARRIER
    assert r["seg_state"] == "descended" and r["pose_state"] == "exploded"


def test_route_case_D_both_flat_builds_trace() -> None:
    # seg flat + pose flat (vs baseline) -> trace before any carrier decision.
    r = route_substrate_next_action(d_seg=0.50, d_pose=1.0, baseline_d_pose=1.0)
    assert r["case"] == "D"
    assert r["route"] == ROUTE_BUILD_AUTHORITY_TRACE


def test_route_case_E_bridge_failed() -> None:
    r = route_substrate_next_action(d_seg=0.0, d_pose=0.0, eval_bridge_ok=False)
    assert r["case"] == "E"
    assert r["route"] == ROUTE_PATCH_BRIDGE_NEXT


def test_route_case_F_ep250_seg_flat_pose_exploded_builds_trace() -> None:
    # The REAL ep250 numbers: seg flat at ~0.505, pose exploded at ~151.5.
    r = route_substrate_next_action(
        d_seg=0.50485, d_pose=151.45, proxy_total_diverged=True
    )
    assert r["case"] == "F"
    assert r["route"] == ROUTE_BUILD_AUTHORITY_TRACE
    assert r["seg_state"] == "flat" and r["pose_state"] == "exploded"
    assert "proxy_total_diverged" in r["reason"]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(d_seg=0.05, d_pose=0.001),
        dict(d_seg=0.50, d_pose=0.30, baseline_d_pose=1.0),
        dict(d_seg=0.05, d_pose=200.0),
        dict(d_seg=0.50, d_pose=1.0, baseline_d_pose=1.0),
        dict(d_seg=0.0, d_pose=0.0, eval_bridge_ok=False),
        dict(d_seg=0.50485, d_pose=151.45),
    ],
)
def test_route_never_auto_kills(kwargs) -> None:
    # Forbidden premature KILL: every case routes to a constructive action.
    r = route_substrate_next_action(**kwargs)
    assert r["auto_kill"] is False


def test_route_seg_descend_uses_baseline_when_given() -> None:
    # With a baseline of 0.30, d_seg=0.20 IS a descent (below 0.30 - margin);
    # without it, d_seg=0.20 is below the 0.50 collapse floor => also descended.
    r = route_substrate_next_action(d_seg=0.20, d_pose=0.001, baseline_d_seg=0.30)
    assert r["seg_state"] == "descended"
    r2 = route_substrate_next_action(d_seg=0.29, d_pose=0.001, baseline_d_seg=0.30)
    assert r2["seg_state"] == "flat"  # within margin of baseline => not a descent


# ---------------------------------------------------------------------------
# Ingest tool: end-to-end on synthetic fixtures (no SSD dependency).
# ---------------------------------------------------------------------------
def _write_frontier_pointer(tmp: Path, cpu_score: float, archive_sha: str = "frontier_base_sha") -> Path:
    p = tmp / "frontier.json"
    p.write_text(
        json.dumps(
            {
                "our_local_frontier_contest_cpu": {
                    "axis": "contest_cpu",
                    "score": cpu_score,
                    "evidence_grade": "[contest-CPU]",
                    "archive_sha256": archive_sha,
                }
            }
        )
    )
    return p


def _write_exact_eval(tmp: Path, *, d_seg: float, d_pose: float, bytes_: int, sha: str) -> Path:
    p = tmp / "exact_eval.json"
    p.write_text(
        json.dumps(
            {
                "schema": "hi_nerv_backend_only_exact_eval.v1",
                "avg_segnet_dist": d_seg,
                "avg_posenet_dist": d_pose,
                "axis_tag": "[macOS-CPU advisory]",
                "pipeline_works": True,
                "export": {"archive_bytes": bytes_, "archive_sha256": sha, "archive_path": str(tmp / "a.zip")},
                "checkpoint": {"global_epoch": 249, "ema_state_path": str(tmp / "c.npsd")},
                "b2": {"b2_returncode": 0, "b2_result": {"avg_segnet_dist": d_seg, "avg_posenet_dist": d_pose}},
            }
        )
    )
    return p


def test_ingest_emits_typed_rows_and_rejects_garbage_base(tmp_path: Path) -> None:
    mod = _load_tool()
    pointer = _write_frontier_pointer(tmp_path, 0.19199)
    ev = _write_exact_eval(tmp_path, d_seg=0.50485, d_pose=151.45, bytes_=257017, sha="abc123")
    out = mod.ingest_exact_eval(
        exact_eval_json=ev, tag="r3_ep250", output_dir=tmp_path, frontier_pointer=pointer
    )
    cae = out["candidate_action_evaluation"]
    dec = out["campaign_decision"]
    # The exact contest-score arithmetic (NOT proxy): 100*0.50485 + sqrt(10*151.45) + 25*257017/N.
    expected = 100.0 * 0.50485 + math.sqrt(10.0 * 151.45) + 25.0 * 257017 / 37_545_489.0
    assert cae["candidate_score"] == pytest.approx(expected, rel=1e-6)
    assert cae["pays_rent"] is False  # garbage base: ΔS >> 0
    assert cae["verdict"] == "above_frontier"
    assert cae["delta_score_total"] > 0.0
    # The UNIVERSAL verdict routes to INSPECT (never auto-kill) with binding named.
    assert dec["decision"] == "INSPECT_BINDING_CONSTRAINT"
    assert dec["auto_kill"] is False
    assert dec["binding_constraint"]["binding_constraint"] in ("seg", "pose")
    # The NEXT-ACTION routes to the authority trace (model-vs-bridge disambiguation).
    assert dec["next_action"]["case"] == "F"
    assert dec["next_action"]["route"] == ROUTE_BUILD_AUTHORITY_TRACE
    # Typed rows actually written.
    assert Path(out["candidate_action_evaluation_path"]).is_file()
    assert Path(out["campaign_decision_path"]).is_file()
    # Frontier read from the pointer, not hardcoded.
    assert dec["frontier"]["score"] == pytest.approx(0.19199)
    # AUTHORITY TIER + METRIC FAMILY firewall: a macOS advisory evaluate.py row is a real
    # exact_evaluate metric, but exact_cpu_advisory is NOT a contest axis -> it can inform the
    # next experiment (mechanism) but is structurally barred from the SCORE roadmap.
    assert cae["authority_tier"] == "exact_cpu_advisory"
    assert cae["metric_family"] == "exact_evaluate"  # ran evaluate.py + has d_seg/d_pose/bytes
    assert dec["authority_tier"] == "exact_cpu_advisory"
    assert dec["metric_family"] == "exact_evaluate"
    assert dec["score_roadmap_update_eligible"] is False  # not a contest axis
    assert dec["mechanism_update_eligible"] is True  # real measurement -> may direct next experiment
    assert dec["promotion_update_eligible"] is False  # requires paired CPU+CUDA
    assert dec["roadmap_update_eligible"] is False  # back-compat alias == score_roadmap


def test_ingest_contest_cpu_exact_is_score_roadmap_eligible(tmp_path: Path) -> None:
    # The firewall must PASS the right rows: a contest_cpu exact_evaluate row with full
    # fields IS score-roadmap-eligible (else the gate is uselessly strict).
    mod = _load_tool()
    pointer = _write_frontier_pointer(tmp_path, 0.19199)
    p = tmp_path / "exact_eval.json"
    p.write_text(
        json.dumps(
            {
                "schema": "hi_nerv_backend_only_exact_eval.v1",
                "avg_segnet_dist": 0.0012,
                "avg_posenet_dist": 0.00004,
                "axis_tag": "[contest-CPU]",  # the public-leaderboard authority axis
                "pipeline_works": True,
                "export": {"archive_bytes": 178000, "archive_sha256": "z"},
                "checkpoint": {"global_epoch": 3000},
                "b2": {"b2_returncode": 0, "b2_result": {"avg_segnet_dist": 0.0012, "avg_posenet_dist": 0.00004}},
            }
        )
    )
    out = mod.ingest_exact_eval(exact_eval_json=p, tag="t", output_dir=tmp_path, frontier_pointer=pointer)
    dec = out["campaign_decision"]
    assert dec["authority_tier"] == "contest_cpu"
    assert dec["metric_family"] == "exact_evaluate"
    assert dec["score_roadmap_update_eligible"] is True  # contest axis + exact metric + full fields
    assert dec["promotion_update_eligible"] is False  # still needs the PAIRED cuda axis too


def test_ingest_canonical_contest_auth_eval_json_schema(tmp_path: Path) -> None:
    """REGRESSION (pr110pp_r1, 2026-06-10): the canonical contest_auth_eval.json
    (the Modal CPU/CUDA auth-eval output) stores ``archive_size_bytes`` at TOP
    LEVEL (not nested under ``b2`` and not as ``archive_bytes``) and the archive
    sha under ``provenance.archive_sha256``. The ingest tool must read both so
    direct contest_auth_eval.json ingest works — previously it raised
    ``ValueError: ... archive_bytes (got ... bytes=None)``."""
    mod = _load_tool()
    pointer = _write_frontier_pointer(tmp_path, 0.19199)
    p = tmp_path / "contest_auth_eval.json"
    p.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "avg_segnet_dist": 0.00055979,
                "avg_posenet_dist": 0.00015129,
                "archive_size_bytes": 178520,  # top-level canonical field
                "final_score": 0.21,
                "score_recomputed_from_components": 0.2137441555314411,
                "evidence_grade": "contest-CPU",
                "score_axis": "contest_cpu",
                "lane_tag": "[contest-CPU]",
                # The REAL canonical contest_auth_eval.json has NO pipeline_works
                # and NO b2 block; its success signal is score_claim_valid + a
                # contest-axis evidence_grade.
                "score_claim": True,
                "score_claim_valid": True,
                "n_samples": 600,
                "provenance": {
                    "archive_sha256": "6c8059a75929e34eeae700093481c73b9569d32e58e07c85b218b0b8cd2d49c1",
                    "device": "cpu",
                },
            }
        )
    )
    out = mod.ingest_exact_eval(
        exact_eval_json=p, tag="contest_cpu_r1", output_dir=tmp_path, frontier_pointer=pointer
    )
    cae = out["candidate_action_evaluation"]
    # bytes were read from top-level archive_size_bytes (no ValueError).
    expected = 100.0 * 0.00055979 + math.sqrt(10.0 * 0.00015129) + 25.0 * 178520 / 37_545_489.0
    assert cae["candidate_score"] == pytest.approx(expected, rel=1e-6)
    # candidate sha read from provenance.archive_sha256.
    assert cae["candidate_archive_sha256"].startswith("6c8059a7")
    # contest-CPU exact metric is score-roadmap eligible (firewall passes the right row).
    assert out["campaign_decision"]["authority_tier"] == "contest_cpu"
    assert out["campaign_decision"]["metric_family"] == "exact_evaluate"


def test_ingest_stale_base_mismatch_flagged(tmp_path: Path) -> None:
    mod = _load_tool()
    pointer = _write_frontier_pointer(tmp_path, 0.19199)
    ev = _write_exact_eval(tmp_path, d_seg=0.50, d_pose=151.0, bytes_=257017, sha="cand_sha")
    out = mod.ingest_exact_eval(
        exact_eval_json=ev,
        tag="r3_ep250",
        output_dir=tmp_path,
        frontier_pointer=pointer,
        base_archive_sha256="a_different_base_sha",  # != frontier_base_sha
    )
    # Caller pinned an expected base that differs from the current frontier base
    # (frontier_base_sha) => the candidate's ΔS is against the wrong base => STALE.
    assert out["candidate_action_evaluation"]["stale_base_mismatch"] is True
    # And the matching-base case is NOT stale.
    out_ok = mod.ingest_exact_eval(
        exact_eval_json=ev,
        tag="r3_ep250_ok",
        output_dir=tmp_path,
        frontier_pointer=pointer,
        base_archive_sha256="frontier_base_sha",  # == frontier base
    )
    assert out_ok["candidate_action_evaluation"]["stale_base_mismatch"] is False


def test_ingest_detects_proxy_divergence_from_trajectory(tmp_path: Path) -> None:
    mod = _load_tool()
    pointer = _write_frontier_pointer(tmp_path, 0.19199)
    ev = _write_exact_eval(tmp_path, d_seg=0.50485, d_pose=151.45, bytes_=257017, sha="abc")
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    # best early (5.6), final much worse (65.1) => diverged.
    (ck / "best.meta.json").write_text(json.dumps({"global_epoch": 201, "checkpoint_selection_metric_value": 5.6, "is_final": False}))
    (ck / "final.meta.json").write_text(json.dumps({"global_epoch": 599, "checkpoint_selection_metric_value": 65.1, "is_final": True}))
    out = mod.ingest_exact_eval(
        exact_eval_json=ev, tag="r3_ep250", output_dir=tmp_path,
        frontier_pointer=pointer, checkpoint_trajectory_dir=ck,
    )
    div = out["campaign_decision"]["proxy_divergence"]
    assert div["scanned"] is True
    assert div["diverged"] is True
    assert div["best_epoch"] == 201 and div["final_epoch"] == 599


def test_ingest_bridge_failure_routes_to_patch(tmp_path: Path) -> None:
    mod = _load_tool()
    pointer = _write_frontier_pointer(tmp_path, 0.19199)
    p = tmp_path / "exact_eval.json"
    p.write_text(
        json.dumps(
            {
                "schema": "hi_nerv_backend_only_exact_eval.v1",
                "avg_segnet_dist": 0.5,
                "avg_posenet_dist": 1.0,
                "pipeline_works": False,
                "export": {"archive_bytes": 1000, "archive_sha256": "x"},
                "checkpoint": {"global_epoch": 1},
                "b2": {"b2_returncode": 2, "b2_result": {"avg_segnet_dist": 0.5, "avg_posenet_dist": 1.0}},
            }
        )
    )
    out = mod.ingest_exact_eval(exact_eval_json=p, tag="t", output_dir=tmp_path, frontier_pointer=pointer)
    # Bridge failed => the next-action router says patch the bridge (no model verdict).
    assert out["campaign_decision"]["next_action"]["case"] == "E"
    assert out["campaign_decision"]["next_action"]["route"] == ROUTE_PATCH_BRIDGE_NEXT
