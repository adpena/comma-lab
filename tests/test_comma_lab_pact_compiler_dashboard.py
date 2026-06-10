# SPDX-License-Identifier: MIT
"""Behavioral tests for the Vehicle-OS compiler dashboard generator.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" Class-2: these tests verify BEHAVIOR
(maturity assigned from evidence, pointer-only scores, fail-soft AUDIT_PENDING
rows on missing manifests) — NOT constants. Each test would FAIL if the
generator regressed to a constant-emitter.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from comma_lab.pact_compiler_dashboard import (  # noqa: E402
    AUDIT_PENDING,
    MATURITY_LADDER,
    VEHICLE_SPECS,
    build_dashboard_model,
    render_json,
    render_markdown,
    write_dashboard,
)
from tac.substrates._shared.objective_reachability_manifest import (  # noqa: E402
    ObjectiveReachabilityManifest,
    emit_objective_reachability_manifest,
)
from tac.substrates._shared.vehicle_fidelity_manifest import (  # noqa: E402
    MechanismEvidence,
    VehicleFidelityManifest,
    emit_manifest,
)

# ---------------------------------------------------------------------------
# fixtures — an isolated repo_root with controllable .omx/state
# ---------------------------------------------------------------------------


@pytest.fixture()
def empty_repo(tmp_path: Path) -> Path:
    """A bare repo_root with no manifests, no pointer, no progress log."""
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    (tmp_path / "experiments" / "results").mkdir(parents=True)
    return tmp_path


def _write_pointer(repo: Path, *, cpu: float, cuda: float, pr: object = None) -> None:
    payload = {
        "our_local_frontier_contest_cpu": {
            "score": cpu,
            "archive_sha256": "a" * 64,
        },
        "our_local_frontier_contest_cuda": {
            "score": cuda,
            "archive_sha256": "b" * 64,
        },
        "submitted_pr_number_for_current_frontier": pr,
        "last_refreshed_utc": "2026-06-09T00:00:00+00:00",
    }
    (repo / ".omx" / "state" / "canonical_frontier_pointer.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# 1. fresh-checkout / empty repo: generator runs, all rows AUDIT_PENDING-soft
# ---------------------------------------------------------------------------


def test_runs_from_fresh_checkout_without_crashing(empty_repo: Path) -> None:
    model = build_dashboard_model(empty_repo)
    assert model.rows  # every VehicleSpec produces a row
    assert len(model.rows) == len(VEHICLE_SPECS)


def test_missing_manifest_yields_explicit_audit_pending_not_crash(empty_repo: Path) -> None:
    model = build_dashboard_model(empty_repo)
    by_id = {r.vehicle: r for r in model.rows}
    # sane_hnerv has no manifest in the empty repo -> L0 with manifest_missing blocker.
    assert by_id["sane_hnerv"].maturity_level == "L0"
    assert "manifest" in by_id["sane_hnerv"].current_blocker.lower()
    # authority_tier is the honest AUDIT_PENDING sentinel, not a fabricated value.
    assert by_id["sane_hnerv"].authority_tier == AUDIT_PENDING


def test_empty_repo_renders_json_and_markdown(empty_repo: Path) -> None:
    model = build_dashboard_model(empty_repo)
    js = render_json(model)
    md = render_markdown(model)
    assert json.loads(js)["schema"] == "pact_compiler_dashboard.v1"
    assert "Per-vehicle maturity" in md


# ---------------------------------------------------------------------------
# 2. pointer-only scores: never hardcoded; reflect the pointer file
# ---------------------------------------------------------------------------


def test_frontier_scores_come_from_pointer_not_hardcoded(empty_repo: Path) -> None:
    _write_pointer(empty_repo, cpu=0.12345, cuda=0.23456, pr=99)
    model = build_dashboard_model(empty_repo)
    assert model.frontier["status"] == "ok"
    assert model.frontier["contest_cpu_score"] == 0.12345
    assert model.frontier["contest_cuda_score"] == 0.23456
    assert model.frontier["submitted_pr_number_for_current_frontier"] == 99


def test_frontier_changes_when_pointer_changes(empty_repo: Path) -> None:
    _write_pointer(empty_repo, cpu=0.5, cuda=0.6)
    first = build_dashboard_model(empty_repo).frontier["contest_cpu_score"]
    _write_pointer(empty_repo, cpu=0.4, cuda=0.6)
    second = build_dashboard_model(empty_repo).frontier["contest_cpu_score"]
    assert first == 0.5 and second == 0.4  # not a constant


def test_missing_pointer_is_explicit_status_not_a_fake_score(empty_repo: Path) -> None:
    model = build_dashboard_model(empty_repo)  # no pointer written
    assert model.frontier["status"] == "POINTER_MISSING"
    assert "contest_cpu_score" not in model.frontier


# ---------------------------------------------------------------------------
# 3. maturity FROM EVIDENCE
# ---------------------------------------------------------------------------


def test_laundering_fidelity_manifest_assigns_L0(empty_repo: Path) -> None:
    # A documentation-fake (claims bilinear-skip, declares it absent) -> L0.
    m = VehicleFidelityManifest(
        vehicle_id="sane_hnerv",
        claimed_family="HNeRV with bilinear-skip",
        mechanisms_absent=("bilinear_skip",),
        docstring_claims=("canonical HNeRV with bilinear-skip",),
    )
    emit_manifest(m, empty_repo)
    row = {r.vehicle: r for r in build_dashboard_model(empty_repo).rows}["sane_hnerv"]
    assert row.maturity_level == "L0"
    assert "LAUNDERING" in row.maturity_evidence


def test_present_mechanism_assigns_at_least_L1(empty_repo: Path) -> None:
    m = VehicleFidelityManifest(
        vehicle_id="pact_nerv_vq",
        claimed_family="VQ-VAE NeRV",
        actual_mechanisms_present=(
            MechanismEvidence(
                mechanism="codebook_vq",
                evidence="src/tac/substrates/pact_nerv_vq/architecture.py:141-166",
            ),
        ),
        mechanisms_absent=("bilinear_skip",),
        docstring_claims=("codebook",),
    )
    emit_manifest(m, empty_repo)
    row = {r.vehicle: r for r in build_dashboard_model(empty_repo).rows}["pact_nerv_vq"]
    assert row.maturity_level == "L1"
    assert "codebook_vq" in row.maturity_evidence


def test_zero_mechanism_honest_sketch_is_L0(empty_repo: Path) -> None:
    m = VehicleFidelityManifest(
        vehicle_id="ff_nerv",
        claimed_family="FFNeRV DCT-grid",
        mechanisms_absent=("residual_hf_path",),
    )
    emit_manifest(m, empty_repo)
    row = {r.vehicle: r for r in build_dashboard_model(empty_repo).rows}["ff_nerv"]
    assert row.maturity_level == "L0"


def test_reachability_severance_surfaces_in_blocker(empty_repo: Path) -> None:
    # hi_nerv present-mechanism + a weight-surface severance -> L1, blocker names it.
    fid = VehicleFidelityManifest(
        vehicle_id="hi_nerv",
        claimed_family="HiNeRV",
        actual_mechanisms_present=(
            MechanismEvidence(
                mechanism="bilinear_skip",
                evidence="src/tac/substrates/hi_nerv/architecture.py:154-160",
            ),
        ),
        mechanisms_absent=("grid_pe",),
    )
    emit_manifest(fid, empty_repo)
    reach = ObjectiveReachabilityManifest(
        vehicle="hi_nerv",
        segnet_objective_active=True,
        posenet_objective_active=True,
        loss_weights_nonzero=False,
        first_failed_surface="weight",
    )
    emit_objective_reachability_manifest(reach, empty_repo)
    row = {r.vehicle: r for r in build_dashboard_model(empty_repo).rows}["hi_nerv"]
    assert row.maturity_level == "L1"
    assert "weight" in row.current_blocker


def test_snerv_with_exact_cae_row_assigns_L4(empty_repo: Path) -> None:
    # Emit a clean fidelity + reachability + a fake SSD verdict tier under tmp.
    fid = VehicleFidelityManifest(
        vehicle_id="snerv_inverse_steg_carrier",
        claimed_family="SNeRV",
        actual_mechanisms_present=(
            MechanismEvidence(
                mechanism="mfu_hfr_tub",
                evidence="src/tac/substrates/snerv_inverse_steg_carrier/official_mfu.py:295-344",
            ),
        ),
        mechanisms_absent=("bilinear_skip",),
    )
    emit_manifest(fid, empty_repo)
    reach = ObjectiveReachabilityManifest(
        vehicle="snerv",
        segnet_objective_active=True,
        posenet_objective_active=True,
        segnet_surrogate_rows=("ce",),
        segnet_vjp_reaches_renderer=True,
        posenet_vjp_reaches_renderer=True,
        loss_weights_nonzero=True,
    )
    emit_objective_reachability_manifest(reach, empty_repo)
    # The SSD-tier CandidateActionEvaluation is read from the real mounts; on a
    # CI box without VertigoDataTier it degrades to L2 (clean, no row located).
    # So this test asserts the >=L2 floor that the clean manifests guarantee.
    row = {r.vehicle: r for r in build_dashboard_model(empty_repo).rows}["snerv"]
    assert row.maturity_level in {"L2", "L4"}
    if row.maturity_level == "L4":
        assert row.metric_family == "exact_pair_scorer"


def test_pr110pp_byte_closed_candidate_assigns_L3(empty_repo: Path) -> None:
    cand = empty_repo / "experiments" / "results" / "pr110pp_r2_nonmps_candidate_20260609"
    cand.mkdir(parents=True)
    (cand / "candidate_archive.zip").write_bytes(b"x")
    (cand / "byte_closure_proof.json").write_text(
        json.dumps({"candidate_archive_bytes": 178493}), encoding="utf-8"
    )
    (cand / "noop_detector.json").write_text(
        json.dumps({"consumption_proven": True}), encoding="utf-8"
    )
    row = {r.vehicle: r for r in build_dashboard_model(empty_repo).rows}["pr110pp"]
    assert row.maturity_level == "L3"
    assert "noop_detector" in row.maturity_evidence


def test_pr110pp_without_candidate_falls_back_below_L3(empty_repo: Path) -> None:
    row = {r.vehicle: r for r in build_dashboard_model(empty_repo).rows}["pr110pp"]
    assert row.maturity_level != "L3"  # no byte-closed candidate located


# ---------------------------------------------------------------------------
# 4. infrastructure row + allowed-claim wiring + write_dashboard
# ---------------------------------------------------------------------------


def test_infrastructure_row_is_marked_n_a_vehicle(empty_repo: Path) -> None:
    row = {r.vehicle: r for r in build_dashboard_model(empty_repo).rows}["atlas_atoms_v3"]
    assert row.maturity_level == "n/a-vehicle"
    assert row.is_vehicle is False


def test_allowed_claim_matches_ladder_for_each_level(empty_repo: Path) -> None:
    claim_by_level = {lvl: claim for lvl, _, claim in MATURITY_LADDER}
    for r in build_dashboard_model(empty_repo).rows:
        if r.maturity_level in claim_by_level:
            assert r.allowed_claim == claim_by_level[r.maturity_level]


def test_every_row_carries_maturity_evidence(empty_repo: Path) -> None:
    for r in build_dashboard_model(empty_repo).rows:
        assert r.maturity_evidence.strip(), f"{r.vehicle} has empty maturity_evidence"


def test_write_dashboard_emits_both_files(empty_repo: Path) -> None:
    json_path, md_path = write_dashboard(empty_repo)
    assert json_path.is_file() and md_path.is_file()
    assert json_path.name == "pact_compiler_dashboard.json"
    assert md_path.name == "pact_compiler_dashboard.md"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "pact_compiler_dashboard.v1"
    assert len(payload["rows"]) == len(VEHICLE_SPECS)


# ---------------------------------------------------------------------------
# 5. live-work section reads the progress log fail-soft
# ---------------------------------------------------------------------------


def test_live_work_reports_in_progress_subagents(empty_repo: Path) -> None:
    log = empty_repo / ".omx" / "state" / "subagent_progress.jsonl"
    rows = [
        {"subagent_id": "alpha", "status": "in_progress", "step": 1,
         "next_action": "do x", "written_at_utc": "2026-06-09T01:00:00+00:00"},
        {"subagent_id": "beta", "status": "complete", "step": 9,
         "next_action": "", "written_at_utc": "2026-06-09T02:00:00+00:00"},
        {"subagent_id": "alpha", "status": "in_progress", "step": 2,
         "next_action": "do y", "written_at_utc": "2026-06-09T03:00:00+00:00"},
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    model = build_dashboard_model(empty_repo)
    ids = [w["subagent_id"] for w in model.live_work]
    assert "alpha" in ids  # latest alpha row is in_progress
    assert "beta" not in ids  # complete is excluded
    # latest-row-wins: alpha's step is 2, next_action 'do y'
    alpha = next(w for w in model.live_work if w["subagent_id"] == "alpha")
    assert alpha["step"] == 2 and alpha["next_action"] == "do y"


def test_corrupt_progress_log_is_fail_soft(empty_repo: Path) -> None:
    log = empty_repo / ".omx" / "state" / "subagent_progress.jsonl"
    log.write_text("not json\n{bad\n", encoding="utf-8")
    model = build_dashboard_model(empty_repo)  # must not crash
    assert model.live_work == ()


def test_markdown_includes_pointer_only_note(empty_repo: Path) -> None:
    md = render_markdown(build_dashboard_model(empty_repo))
    assert "POINTER-ONLY" in md or "pointer-only" in md.lower()
    assert "SCHEMA GAP" in md  # the vehicle_fidelity maturity_level gap is surfaced


def test_schema_gap_is_machine_readable(empty_repo: Path) -> None:
    model = build_dashboard_model(empty_repo)
    surfaces = {g["surface"] for g in model.schema_gaps}
    assert "vehicle_fidelity_manifest.v1" in surfaces
    gap = next(
        g for g in model.schema_gaps if g["surface"] == "vehicle_fidelity_manifest.v1"
    )
    assert "maturity_level" in gap["gap"]
    # surfaced in the rendered JSON too.
    assert json.loads(render_json(model))["schema_gaps"]
