# SPDX-License-Identifier: MIT
"""Tests for Deliverable 1 — the objective-reachability manifest (Catalog #386).

NO FAKE Class-2 discipline: these verify BEHAVIOR (verify() raises on a severed
VJP; passes on a reaching carrier; rejects argmax-d_seg as a training row; the
SNeRV severance regression fixture). Replacing ObjectiveReachabilityManifest.
reachability_findings() with `return ()` would make the severance / mis-naming /
SNeRV regression tests FAIL.
"""

from __future__ import annotations

import json

import pytest

from tac.substrates._shared.objective_reachability_manifest import (
    AUDIT_PENDING,
    CANONICAL_GRADIENT_MECHANISMS,
    SEGNET_SURROGATE_ROWS,
    ObjectiveReachabilityManifest,
    ObjectiveReachabilityVerifyError,
    audit_objective_reachability_manifests,
    objective_reachability_path_for_vehicle,
)
from tac.substrates._shared.objective_reachability_manifests_canonical import (
    HI_NERV_REACHABILITY,
    PACT_NERV_VQ_REACHABILITY,
    SNERV_REACHABILITY,
    emit_all,
)


def _reaching() -> ObjectiveReachabilityManifest:
    """A faithful score-aware carrier whose objective reaches the renderer."""
    return ObjectiveReachabilityManifest(
        vehicle="faithful",
        segnet_objective_active=True,
        posenet_objective_active=True,
        segnet_surrogate_rows=("ce",),
        segnet_vjp_reaches_renderer=True,
        posenet_vjp_reaches_renderer=True,
        loss_weights_nonzero=True,
        gradient_norm_by_mechanism={"decoder_blocks": 0.42, "latents": 0.13},
        dseg_is_verification_metric_only=True,
    )


# ---------------------------------------------------------------------------
# Construction-time invariants
# ---------------------------------------------------------------------------


def test_empty_vehicle_rejected() -> None:
    with pytest.raises(ValueError, match="vehicle must be a non-empty string"):
        ObjectiveReachabilityManifest(vehicle="  ")


def test_argmax_dseg_as_surrogate_row_rejected_at_construction() -> None:
    # The official argmax d_seg is gradient-zero a.e. -> NOT a valid surrogate.
    with pytest.raises(ValueError, match="not a canonical surrogate"):
        ObjectiveReachabilityManifest(
            vehicle="x",
            segnet_objective_active=True,
            segnet_surrogate_rows=("d_seg",),  # forbidden
        )


def test_noncanonical_gradient_mechanism_key_rejected() -> None:
    with pytest.raises(ValueError, match="canonical mechanisms"):
        ObjectiveReachabilityManifest(
            vehicle="x",
            gradient_norm_by_mechanism={"not_a_mechanism": 1.0},
        )


def test_canonical_surrogate_rows_are_all_accepted() -> None:
    m = ObjectiveReachabilityManifest(
        vehicle="x",
        segnet_objective_active=True,
        segnet_surrogate_rows=SEGNET_SURROGATE_ROWS,
        segnet_vjp_reaches_renderer=True,
        posenet_vjp_reaches_renderer=True,
        loss_weights_nonzero=True,
    )
    assert set(m.segnet_surrogate_rows) == set(SEGNET_SURROGATE_ROWS)


# ---------------------------------------------------------------------------
# verify() — the fail-closed severance / mis-naming check
# ---------------------------------------------------------------------------


def test_reaching_carrier_verifies() -> None:
    _reaching().verify()  # must not raise


def test_weight_severance_fails() -> None:
    m = _reaching()
    m = ObjectiveReachabilityManifest(
        vehicle="w",
        segnet_objective_active=True,
        posenet_objective_active=True,
        segnet_surrogate_rows=("ce",),
        segnet_vjp_reaches_renderer=True,
        posenet_vjp_reaches_renderer=True,
        loss_weights_nonzero=False,  # the Catalog #384 precondition
    )
    with pytest.raises(ObjectiveReachabilityVerifyError, match="WEIGHT-SEVERANCE"):
        m.verify()


def test_pose_vjp_severance_fails_the_snerv_regression() -> None:
    # The canonical SNeRV pose-VJP severance the f5c66f43c uncrossing fixed:
    # weights nonzero, loss named score-aware, but the pose VJP does NOT reach.
    m = ObjectiveReachabilityManifest(
        vehicle="snerv_pre_uncross",
        segnet_objective_active=True,
        posenet_objective_active=True,
        segnet_surrogate_rows=("ce",),
        segnet_vjp_reaches_renderer=True,
        posenet_vjp_reaches_renderer=False,  # severed at 3 layers pre-f5c66f43c
        loss_weights_nonzero=True,
    )
    findings = m.reachability_findings()
    assert any("VJP-SEVERANCE" in f and "posenet" in f.lower() for f in findings)
    with pytest.raises(ObjectiveReachabilityVerifyError, match="VJP-SEVERANCE"):
        m.verify()


def test_grad_norm_zero_is_severance() -> None:
    m = ObjectiveReachabilityManifest(
        vehicle="g",
        segnet_objective_active=True,
        posenet_objective_active=True,
        segnet_surrogate_rows=("ce",),
        segnet_vjp_reaches_renderer=True,
        posenet_vjp_reaches_renderer=True,
        loss_weights_nonzero=True,
        gradient_norm_by_mechanism={"decoder_blocks": 0.0},  # measured ZERO
    )
    assert "decoder_blocks" in m.severed_mechanisms()
    with pytest.raises(ObjectiveReachabilityVerifyError, match="GRAD-NORM-SEVERANCE"):
        m.verify()


def test_audit_pending_grad_norm_is_not_severance() -> None:
    # AUDIT_PENDING is honest not-measured, NOT a severance -> verifies.
    m = ObjectiveReachabilityManifest(
        vehicle="p",
        segnet_objective_active=True,
        posenet_objective_active=True,
        segnet_surrogate_rows=("ce",),
        segnet_vjp_reaches_renderer=True,
        posenet_vjp_reaches_renderer=True,
        loss_weights_nonzero=True,
        gradient_norm_by_mechanism={"hfr": AUDIT_PENDING, "tub": AUDIT_PENDING},
    )
    m.verify()  # must not raise
    assert m.pending_mechanisms() == {"hfr", "tub"}
    assert not m.severed_mechanisms()


def test_surrogate_absence_fails() -> None:
    # SegNet active but no surrogate -> only gradient-zero argmax d_seg would
    # carry seg signal -> no seg learning.
    m = ObjectiveReachabilityManifest(
        vehicle="s",
        segnet_objective_active=True,
        posenet_objective_active=True,
        segnet_surrogate_rows=(),  # empty
        segnet_vjp_reaches_renderer=True,
        posenet_vjp_reaches_renderer=True,
        loss_weights_nonzero=True,
    )
    with pytest.raises(ObjectiveReachabilityVerifyError, match="SURROGATE-ABSENCE"):
        m.verify()


def test_dseg_not_verification_only_fails() -> None:
    m = ObjectiveReachabilityManifest(
        vehicle="d",
        segnet_objective_active=True,
        posenet_objective_active=True,
        segnet_surrogate_rows=("ce",),
        segnet_vjp_reaches_renderer=True,
        posenet_vjp_reaches_renderer=True,
        loss_weights_nonzero=True,
        dseg_is_verification_metric_only=False,  # the firewall trip
    )
    with pytest.raises(ObjectiveReachabilityVerifyError, match="DSEG-MIS-NAMING"):
        m.verify()


def test_declared_severance_fails() -> None:
    m = ObjectiveReachabilityManifest(
        vehicle="ds",
        segnet_objective_active=True,
        posenet_objective_active=True,
        segnet_surrogate_rows=("ce",),
        segnet_vjp_reaches_renderer=True,
        posenet_vjp_reaches_renderer=True,
        loss_weights_nonzero=True,
        severed_layers=("renderer_block_3 (stop_gradient)",),
    )
    with pytest.raises(ObjectiveReachabilityVerifyError, match="DECLARED-SEVERANCE"):
        m.verify()


def test_non_score_aware_carrier_makes_no_reachability_promise() -> None:
    # A recon-only carrier (neither objective active) makes no score-aware claim
    # -> no severance findings even with weights off + VJP not reaching.
    m = ObjectiveReachabilityManifest(
        vehicle="recon_only",
        segnet_objective_active=False,
        posenet_objective_active=False,
        loss_weights_nonzero=False,
    )
    assert m.reachability_findings() == ()
    m.verify()  # must not raise


def test_dseg_firewall_applies_even_to_non_score_aware() -> None:
    # The d_seg-naming firewall is always-on (mis-naming is always the bug).
    m = ObjectiveReachabilityManifest(
        vehicle="r",
        segnet_objective_active=False,
        posenet_objective_active=False,
        dseg_is_verification_metric_only=False,
    )
    with pytest.raises(ObjectiveReachabilityVerifyError, match="DSEG-MIS-NAMING"):
        m.verify()


def test_query_helpers_partition_mechanisms() -> None:
    m = ObjectiveReachabilityManifest(
        vehicle="q",
        gradient_norm_by_mechanism={
            "latents": 0.9,
            "decoder_blocks": 0.0,
            "hfr": AUDIT_PENDING,
        },
    )
    assert m.reaching_mechanisms() == {"latents"}
    assert m.severed_mechanisms() == {"decoder_blocks"}
    assert m.pending_mechanisms() == {"hfr"}


# ---------------------------------------------------------------------------
# Round-trip + canonical seed manifests
# ---------------------------------------------------------------------------


def test_round_trip_serialization() -> None:
    m = _reaching()
    rt = ObjectiveReachabilityManifest.from_dict(m.as_dict())
    assert rt == m


def test_from_dict_rejects_wrong_schema() -> None:
    bad = _reaching().as_dict()
    bad["schema"] = "not_the_schema"
    with pytest.raises(ValueError, match="unexpected schema"):
        ObjectiveReachabilityManifest.from_dict(bad)


def test_seed_snerv_reaches() -> None:
    # SNeRV post-f5c66f43c: both VJPs reach, weights 7.24/7.0 -> verifies.
    SNERV_REACHABILITY.verify()
    assert SNERV_REACHABILITY.segnet_vjp_reaches_renderer is True
    assert SNERV_REACHABILITY.posenet_vjp_reaches_renderer is True
    assert SNERV_REACHABILITY.loss_weights_nonzero is True


def test_seed_hi_nerv_fails_weight_severance() -> None:
    # HiNeRV: distill weights default 0.0 on the shared harness -> fails.
    assert HI_NERV_REACHABILITY.loss_weights_nonzero is False
    assert HI_NERV_REACHABILITY.first_failed_surface == "weight"
    with pytest.raises(ObjectiveReachabilityVerifyError):
        HI_NERV_REACHABILITY.verify()


def test_seed_pact_nerv_vq_fails_pending_mlx_route() -> None:
    assert PACT_NERV_VQ_REACHABILITY.loss_weights_nonzero is False
    with pytest.raises(ObjectiveReachabilityVerifyError):
        PACT_NERV_VQ_REACHABILITY.verify()


def test_emit_and_audit_round_trip(tmp_path) -> None:
    paths = emit_all(repo_root=tmp_path)
    assert len(paths) == 3
    # The emitted JSON is parseable + carries the schema tag.
    for p in paths:
        payload = json.loads(p.read_text())
        assert payload["schema"] == "objective_reachability_manifest.v1"
    # The audit surfaces hi_nerv + pact_nerv_vq findings, not snerv.
    findings = audit_objective_reachability_manifests(repo_root=tmp_path)
    flagged = {f.vehicle for f in findings}
    assert "hi_nerv" in flagged
    assert "pact_nerv_vq" in flagged
    assert "snerv" not in flagged


def test_audit_empty_state_dir_is_clean(tmp_path) -> None:
    assert audit_objective_reachability_manifests(repo_root=tmp_path) == []


def test_path_for_vehicle_under_state_dir(tmp_path) -> None:
    p = objective_reachability_path_for_vehicle("snerv", repo_root=tmp_path)
    assert p.name == "snerv.json"
    assert "objective_reachability" in str(p)


def test_corrupt_manifest_surfaced_as_finding(tmp_path) -> None:
    state = tmp_path / ".omx" / "state" / "objective_reachability"
    state.mkdir(parents=True)
    (state / "broken.json").write_text("{ not json", encoding="utf-8")
    findings = audit_objective_reachability_manifests(repo_root=tmp_path)
    assert any(f.vehicle == "broken" for f in findings)


def test_canonical_gradient_mechanisms_cover_nerv_family() -> None:
    # Sanity: the operator-spec mechanism vocabulary is present.
    for mech in ("latents", "decoder_blocks", "skip_path", "hf_residual",
                 "mfu", "hfr", "tub", "codebook", "selector"):
        assert mech in CANONICAL_GRADIENT_MECHANISMS
