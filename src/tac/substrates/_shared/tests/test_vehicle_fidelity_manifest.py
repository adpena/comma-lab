# SPDX-License-Identifier: MIT
"""Tests for the anti-name-laundering vehicle fidelity manifest (Deliverable 1).

NO FAKE Class-2 discipline: these tests verify BEHAVIOR (verify() raises on a
docstring-claims-X-but-X-absent case; passes on a faithful manifest), not just
constants. Replacing verify()'s body with `pass` would make
``test_verify_fails_closed_on_laundering`` FAIL.
"""

from __future__ import annotations

import json

import pytest

from tac.substrates._shared.vehicle_fidelity_manifest import (
    AUDIT_PENDING,
    CANONICAL_MECHANISM_VOCABULARY,
    MechanismEvidence,
    VehicleFidelityManifest,
    VehicleFidelityVerifyError,
    emit_manifest,
    manifest_path_for_vehicle,
)
from tac.substrates._shared.vehicle_fidelity_manifests_canonical import (
    CANONICAL_VEHICLE_MANIFESTS,
    emit_all,
)

# --------------------------------------------------------------------------
# MechanismEvidence schema
# --------------------------------------------------------------------------


def test_mechanism_evidence_round_trip() -> None:
    ev = MechanismEvidence(
        mechanism="bilinear_skip",
        evidence="src/tac/substrates/foo/architecture.py:10-12",
        test_id="tests/test_foo.py::test_skip_changes_output",
        notes="per-block residual",
    )
    payload = ev.as_dict()
    restored = MechanismEvidence.from_dict(payload)
    assert restored == ev
    assert restored.mechanism == "bilinear_skip"
    assert restored.test_id.endswith("test_skip_changes_output")


def test_mechanism_evidence_rejects_unknown_mechanism() -> None:
    with pytest.raises(ValueError, match="not in canonical"):
        MechanismEvidence(mechanism="warp_drive", evidence="a.py:1")


def test_mechanism_evidence_rejects_empty_evidence() -> None:
    with pytest.raises(ValueError, match="non-empty file:line"):
        MechanismEvidence(mechanism="grid_pe", evidence="   ")


def test_mechanism_evidence_rejects_placeholder_evidence() -> None:
    # Catalog #287: a placeholder literal masquerading as real evidence.
    for bad in ("TBD", "<value>", "placeholder", "pending_ratification"):
        with pytest.raises(ValueError, match="forbidden placeholder"):
            MechanismEvidence(mechanism="grid_pe", evidence=bad)


def test_mechanism_evidence_audit_pending_test_id_is_allowed() -> None:
    # AUDIT_PENDING is an HONEST status (not a fabricated value) -> allowed.
    ev = MechanismEvidence(
        mechanism="codebook_vq",
        evidence="src/tac/substrates/x/architecture.py:5",
        test_id=AUDIT_PENDING,
    )
    assert ev.test_id == AUDIT_PENDING


# --------------------------------------------------------------------------
# VehicleFidelityManifest schema + round-trip
# --------------------------------------------------------------------------


def test_manifest_round_trip_via_dict() -> None:
    m = VehicleFidelityManifest(
        vehicle_id="demo",
        claimed_family="DemoNeRV",
        actual_mechanisms_present=(
            MechanismEvidence(
                mechanism="codebook_vq",
                evidence="a.py:1",
                test_id="tests/t.py::test_vq",
            ),
        ),
        mechanisms_absent=("bilinear_skip", "grid_pe"),
        docstring_claims=("DemoNeRV with vector-quantized latents",),
        tests_proving_mechanism=("tests/t.py::test_vq",),
        summary="demo",
    )
    restored = VehicleFidelityManifest.from_dict(m.as_dict())
    assert restored == m
    # Schema tag is stamped.
    assert m.as_dict()["schema"] == "vehicle_fidelity_manifest.v1"


def test_manifest_round_trip_via_json_string() -> None:
    m = CANONICAL_VEHICLE_MANIFESTS[0]
    text = json.dumps(m.as_dict())
    restored = VehicleFidelityManifest.from_dict(json.loads(text))
    assert restored == m


def test_manifest_rejects_bad_schema() -> None:
    payload = CANONICAL_VEHICLE_MANIFESTS[0].as_dict()
    payload["schema"] = "some_other_schema.v9"
    with pytest.raises(ValueError, match="unexpected schema"):
        VehicleFidelityManifest.from_dict(payload)


def test_manifest_rejects_unknown_absent_mechanism() -> None:
    with pytest.raises(ValueError, match="not in canonical"):
        VehicleFidelityManifest(
            vehicle_id="x",
            claimed_family="X",
            mechanisms_absent=("teleporter",),
        )


def test_manifest_rejects_present_and_absent_overlap() -> None:
    with pytest.raises(ValueError, match="BOTH"):
        VehicleFidelityManifest(
            vehicle_id="x",
            claimed_family="X",
            actual_mechanisms_present=(
                MechanismEvidence(mechanism="grid_pe", evidence="a.py:1"),
            ),
            mechanisms_absent=("grid_pe",),
        )


def test_manifest_requires_nonempty_vehicle_id_and_family() -> None:
    with pytest.raises(ValueError):
        VehicleFidelityManifest(vehicle_id="", claimed_family="X")
    with pytest.raises(ValueError):
        VehicleFidelityManifest(vehicle_id="x", claimed_family="  ")


# --------------------------------------------------------------------------
# THE laundering check (the core fail-closed behavior)
# --------------------------------------------------------------------------


def test_verify_fails_closed_on_laundering_claimed_but_absent() -> None:
    """The sane_hnerv bug class: docstring claims a mechanism that is absent."""
    launderer = VehicleFidelityManifest(
        vehicle_id="fake_skip",
        claimed_family="HNeRV-with-bilinear-skip",
        actual_mechanisms_present=(),
        mechanisms_absent=("bilinear_skip",),
        docstring_claims=("canonical HNeRV with per-pair latent + bilinear-skip",),
    )
    with pytest.raises(VehicleFidelityVerifyError, match="NAME-LAUNDERING"):
        launderer.verify()
    findings = launderer.laundering_findings()
    assert len(findings) == 1
    assert "bilinear_skip" in findings[0]


def test_verify_fails_closed_on_laundering_claimed_but_silent() -> None:
    """A claimed mechanism that is neither present nor explicitly absent."""
    silent = VehicleFidelityManifest(
        vehicle_id="silent_skip",
        claimed_family="HNeRV",
        actual_mechanisms_present=(),
        mechanisms_absent=(),  # silence on the claimed mechanism
        docstring_claims=("decoder with bilinear skip from each prior block",),
    )
    with pytest.raises(VehicleFidelityVerifyError, match="silence is the laundering"):
        silent.verify()


def test_verify_passes_on_faithful_manifest_with_test() -> None:
    """A carrier that genuinely implements its claimed mechanism verifies."""
    faithful = VehicleFidelityManifest(
        vehicle_id="real_vq",
        claimed_family="VQ-VAE NeRV",
        actual_mechanisms_present=(
            MechanismEvidence(
                mechanism="codebook_vq",
                evidence="src/tac/substrates/pact_nerv_vq/architecture.py:141-166",
                test_id="tests/test_vq.py::test_codebook_consumed",
            ),
        ),
        mechanisms_absent=("bilinear_skip", "grid_pe"),
        docstring_claims=("renderer with vector-quantized per-pair latents",),
    )
    faithful.verify()  # must not raise
    assert faithful.laundering_findings() == ()
    assert faithful.unproven_claims() == ()


def test_verify_passes_when_claimed_present_but_test_pending() -> None:
    """A FAITHFUL carrier with an in-flight test verifies (AUDIT_PENDING ok).

    This is the SNeRV/pact_nerv_vq case: the mechanism is genuinely present, so
    it is NOT laundering; the missing test is a separate soft advisory.
    """
    faithful_pending = VehicleFidelityManifest(
        vehicle_id="real_mfu_pending_test",
        claimed_family="SNeRV",
        actual_mechanisms_present=(
            MechanismEvidence(
                mechanism="mfu_hfr_tub",
                evidence="src/tac/substrates/snerv_inverse_steg_carrier/official_mfu.py:295-344",
                test_id=AUDIT_PENDING,
            ),
        ),
        mechanisms_absent=("bilinear_skip",),
        docstring_claims=("SNeRV generates HF with MFU/HFR/TUB",),
    )
    faithful_pending.verify()  # must NOT raise (present == not laundering)
    assert faithful_pending.laundering_findings() == ()
    # ...but the soft advisory IS surfaced.
    advisories = faithful_pending.unproven_claims()
    assert len(advisories) == 1
    assert "UNPROVEN-CLAIM" in advisories[0]
    assert "mfu_hfr_tub" in advisories[0]


def test_no_docstring_claim_means_no_laundering() -> None:
    """A carrier honest about its absences (no false claim) verifies."""
    honest = VehicleFidelityManifest(
        vehicle_id="honest_nerv",
        claimed_family="vanilla NeRV",
        actual_mechanisms_present=(),
        mechanisms_absent=tuple(CANONICAL_MECHANISM_VOCABULARY),
        docstring_claims=("plain sin+PixelShuffle decoder, no residual path",),
    )
    honest.verify()  # must not raise
    assert honest.laundering_findings() == ()


def test_claims_mechanism_detects_trigger_substrings() -> None:
    m = VehicleFidelityManifest(
        vehicle_id="x",
        claimed_family="X",
        mechanisms_absent=("bilinear_skip",),
        docstring_claims=("uses a BILINEAR-SKIP per block",),  # case-insensitive
    )
    hits = m.claims_mechanism("bilinear_skip")
    assert len(hits) == 1
    # A mechanism whose triggers don't appear is not claimed.
    assert m.claims_mechanism("codebook_vq") == ()


# --------------------------------------------------------------------------
# Emit + the 5 real manifests
# --------------------------------------------------------------------------


def test_emit_manifest_writes_durable_json_not_tmp(tmp_path) -> None:
    m = CANONICAL_VEHICLE_MANIFESTS[0]
    out = emit_manifest(m, repo_root=tmp_path)
    assert out.exists()
    # Lands under <repo>/.omx/state/vehicle_fidelity/ (durable). The relative
    # path under the repo root must be the canonical durable location and must
    # NOT be a literal /tmp scratch path. (pytest's own tmp_path may itself sit
    # under /private/tmp, so assert on the repo-relative tail, not the abspath.)
    assert out.relative_to(tmp_path) == (
        type(out)(".omx") / "state" / "vehicle_fidelity" / "hi_nerv.json"
    )
    loaded = VehicleFidelityManifest.from_dict(json.loads(out.read_text()))
    assert loaded == m
    assert manifest_path_for_vehicle(m.vehicle_id, repo_root=tmp_path) == out


def test_emit_manifest_verify_true_raises_for_launderer(tmp_path) -> None:
    launderer = VehicleFidelityManifest(
        vehicle_id="fake",
        claimed_family="HNeRV",
        mechanisms_absent=("bilinear_skip",),
        docstring_claims=("HNeRV with bilinear-skip",),
    )
    with pytest.raises(VehicleFidelityVerifyError):
        emit_manifest(launderer, repo_root=tmp_path, verify=True)
    # ...and with verify=False it IS written (so the gate can surface it).
    out = emit_manifest(launderer, repo_root=tmp_path, verify=False)
    assert out.exists()


def test_emit_all_writes_five_manifests(tmp_path) -> None:
    paths = emit_all(repo_root=tmp_path)
    assert len(paths) == 5
    ids = {p.stem for p in paths}
    assert ids == {
        "hi_nerv",
        "snerv_inverse_steg_carrier",
        "pact_nerv_vq",
        "sane_hnerv",
        "ff_nerv",
    }
    for p in paths:
        assert p.exists()
        VehicleFidelityManifest.from_dict(json.loads(p.read_text()))  # parses


def test_five_real_manifests_load_and_sane_hnerv_is_the_only_launderer() -> None:
    """The 5 real manifests: exactly sane_hnerv fails verify() (the doc-fake)."""
    by_id = {m.vehicle_id: m for m in CANONICAL_VEHICLE_MANIFESTS}
    assert set(by_id) == {
        "hi_nerv",
        "snerv_inverse_steg_carrier",
        "pact_nerv_vq",
        "sane_hnerv",
        "ff_nerv",
    }
    # sane_hnerv is the canonical laundering FAIL.
    with pytest.raises(VehicleFidelityVerifyError, match="bilinear_skip"):
        by_id["sane_hnerv"].verify()
    # The other four are honest and verify.
    for vid in ("hi_nerv", "snerv_inverse_steg_carrier", "pact_nerv_vq", "ff_nerv"):
        by_id[vid].verify()


def test_real_manifests_have_evidence_with_file_line() -> None:
    """Every present mechanism in the real manifests cites file:line."""
    for m in CANONICAL_VEHICLE_MANIFESTS:
        for ev in m.actual_mechanisms_present:
            # evidence must look like path:line (contains a .py and a colon-number)
            assert ".py:" in ev.evidence, (m.vehicle_id, ev.mechanism, ev.evidence)


def test_real_manifests_cite_source_memos() -> None:
    """Every real manifest records which audit memo populated it."""
    fleet = ".omx/research/snerv_all_vehicles_fidelity_review_vs_evaluate_py_20260609.md"
    for m in CANONICAL_VEHICLE_MANIFESTS:
        assert m.source_memos, m.vehicle_id
        # all 5 trace to at least the fleet memo
        assert fleet in m.source_memos, m.vehicle_id


def test_snerv_is_faithful_but_objective_starved() -> None:
    """SNeRV: genuine MFU/HFR/TUB present, scorer-objective absent (recon-MSE)."""
    snerv = next(m for m in CANONICAL_VEHICLE_MANIFESTS if m.vehicle_id == "snerv_inverse_steg_carrier")
    assert "mfu_hfr_tub" in snerv.present_mechanism_names()
    # The objective-starvation finding: scorer weights are NOT nonzero.
    assert "scorer_objective_weights_nonzero" in snerv.mechanisms_absent


def test_pact_nerv_vq_has_genuine_codebook_but_no_skip() -> None:
    vq = next(m for m in CANONICAL_VEHICLE_MANIFESTS if m.vehicle_id == "pact_nerv_vq")
    assert "codebook_vq" in vq.present_mechanism_names()
    assert "bilinear_skip" in vq.mechanisms_absent
    assert "residual_hf_path" in vq.mechanisms_absent
