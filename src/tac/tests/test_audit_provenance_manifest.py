# SPDX-License-Identifier: MIT
"""Tests for Deliverable 2 — the audit-provenance manifest (Catalog #387).

NO FAKE Class-2 discipline: these verify BEHAVIOR (verify() raises on missing
surface / reproduce_command; recheck() detects wrong-candidate mismatch; the
three real 2026-06-09 lapse cases are regression fixtures). Replacing
AuditProvenanceRecord.provenance_findings() with `return ()` would make the
missing-surface / missing-reproduce / lapse-fixture tests FAIL.
"""

from __future__ import annotations

import json

import pytest

from tac.optimization.audit_provenance import (
    CANONICAL_AUDIT_SURFACES,
    AuditProvenanceRecord,
    AuditProvenanceVerifyError,
    audit_provenance_claim_records,
    audit_provenance_path_for_claim,
    emit_audit_provenance_record,
    recheck,
)


def _clean_record(**over) -> AuditProvenanceRecord:
    base = {
        "claim": "SNeRV-B live d_seg is 0.0023 at ep0",
        "file": ".omx/research/snerv_b_first_scorer_probe_verdict_20260609.md",
        "line_or_field": "telemetry.jsonl:ep0:live_argmax_d_seg",
        "candidate_id": "snerv_mistake_b_g1a_20260609T201221Z",
        "observed_value": "0.0023",
        "surface": "live",
        "reproduce_command": ".venv/bin/python tools/print_telemetry.py --ep 0",
        "expected_value": "",
    }
    base.update(over)
    return AuditProvenanceRecord(**base)


# ---------------------------------------------------------------------------
# Construction-time invariants
# ---------------------------------------------------------------------------


def test_empty_claim_rejected() -> None:
    with pytest.raises(ValueError, match="claim must be a non-empty string"):
        _clean_record(claim="   ")


def test_empty_candidate_id_rejected() -> None:
    # The wrong-candidate lapse was an unnamed/wrong candidate.
    with pytest.raises(ValueError, match="candidate_id must be a non-empty string"):
        _clean_record(candidate_id="")


# ---------------------------------------------------------------------------
# verify() — mandatory surface + reproduce_command
# ---------------------------------------------------------------------------


def test_clean_record_verifies() -> None:
    _clean_record().verify()  # must not raise


def test_missing_surface_fails() -> None:
    with pytest.raises(AuditProvenanceVerifyError, match="MISSING-SURFACE"):
        _clean_record(surface="").verify()


def test_invalid_surface_fails() -> None:
    with pytest.raises(AuditProvenanceVerifyError, match="INVALID-SURFACE"):
        _clean_record(surface="psnr").verify()


def test_all_canonical_surfaces_accepted() -> None:
    for s in CANONICAL_AUDIT_SURFACES:
        _clean_record(surface=s).verify()  # must not raise


def test_missing_reproduce_command_fails() -> None:
    with pytest.raises(AuditProvenanceVerifyError, match="MISSING-REPRODUCE-COMMAND"):
        _clean_record(reproduce_command="").verify()


def test_placeholder_reproduce_command_fails() -> None:
    for placeholder in ("TBD", "<command>", "placeholder", "n/a"):
        with pytest.raises(AuditProvenanceVerifyError, match="MISSING-REPRODUCE-COMMAND"):
            _clean_record(reproduce_command=placeholder).verify()


def test_unverified_but_reproducible_record_is_clean() -> None:
    # An honest unverified-but-reproducible record is NOT a hallucination.
    r = _clean_record(verified_by_main_agent=False)
    assert r.provenance_findings() == ()
    r.verify()


# ---------------------------------------------------------------------------
# The three real 2026-06-09 lapse cases as regression fixtures
# ---------------------------------------------------------------------------


def test_lapse_wrong_candidate_ep22399() -> None:
    # ep22399 0.7115 attributed to the WRONG candidate. The record names the
    # right candidate + export surface + a reproduce_command -> the cure is that
    # such a claim is now structurally required to carry the (correct) candidate
    # + surface, so the attribution mismatch is recheck-detectable.
    r = AuditProvenanceRecord(
        claim="ep22399 avg_segnet_dist=0.7115 belongs to the Haar score renderer",
        file=".omx/research/snerv_b_first_scorer_probe_verdict_20260609.md",
        line_or_field="ep22399:avg_segnet_dist",
        candidate_id="haar_score_renderer_ep22399",  # the CORRECT candidate
        observed_value="0.7115",
        surface="export",  # NOT live — the export/receiver-side surface
        reproduce_command=".venv/bin/python tools/print_export_eval.py --ep 22399",
    )
    # With surface + candidate + reproduce present, this is a well-formed record.
    assert r.provenance_findings() == ()
    # The bug it extincts: the SAME 0.7115 claimed as a LIVE SNeRV-B value would
    # be a DIFFERENT (conflated) record on a different surface.
    conflated = AuditProvenanceRecord(
        claim="SNeRV-B live d_seg is 0.7115",  # the hallucinated attribution
        file="x.md",
        line_or_field="ep22399:avg_segnet_dist",
        candidate_id="snerv_mistake_b_g1a_20260609T201221Z",
        observed_value="0.7115",
        surface="live",  # WRONG surface for an export value
        reproduce_command="",  # and unreproducible -> fails closed
    )
    with pytest.raises(AuditProvenanceVerifyError):
        conflated.verify()


def test_lapse_phantom_gates_pact() -> None:
    # pact phantom-gates claimed TRUE while the registry showed False.
    # A well-formed claim must point at the registry + carry a reproduce_command
    # so recheck can confirm the actual registry value.
    phantom = AuditProvenanceRecord(
        claim="pact gate G1b is TRUE",
        file="prose.md",
        line_or_field="lane_registry.json:pact:G1b",
        candidate_id="pact_nerv_vq",
        observed_value="true",
        surface="exact_archive",
        reproduce_command="",  # no way to confirm -> hallucination
    )
    with pytest.raises(AuditProvenanceVerifyError, match="MISSING-REPRODUCE-COMMAND"):
        phantom.verify()


def test_lapse_surface_conflation_071_vs_00023() -> None:
    # The 0.71-vs-0.0023 conflation: two values on DIFFERENT surfaces compared
    # as if the same. The mandatory surface field makes the conflation explicit.
    live = _clean_record(observed_value="0.0023", surface="live")
    export = AuditProvenanceRecord(
        claim="export-surface avg_segnet_dist",
        file="x.md",
        line_or_field="ep22399:avg_segnet_dist",
        candidate_id="haar_score_renderer_ep22399",
        observed_value="0.7115",
        surface="export",
        reproduce_command=".venv/bin/python tools/print_export_eval.py --ep 22399",
    )
    # Both are well-formed individually...
    live.verify()
    export.verify()
    # ...and the surface field proves they are NOT comparable (the cure).
    assert live.surface != export.surface


# ---------------------------------------------------------------------------
# recheck()
# ---------------------------------------------------------------------------


def test_recheck_unverifiable_when_findings_present() -> None:
    r = _clean_record(surface="")  # has a finding
    res = recheck(r)
    assert res.status == "unverifiable"


def test_recheck_unverifiable_for_unsafe_command() -> None:
    r = _clean_record(reproduce_command="rm -rf /")  # not in safe allowlist
    res = recheck(r)
    assert res.status == "unverifiable"
    assert not res.ok()


def _repo_root() -> str:
    # The recheck commands are repo-relative (.venv/bin/python); run from the
    # actual repo root so the interpreter path resolves.
    from pathlib import Path

    import tac.optimization.audit_provenance as ap

    return str(Path(ap.__file__).resolve().parents[3])


def test_recheck_verified_when_value_in_output() -> None:
    # A safe echo-like python command whose output contains observed_value.
    r = _clean_record(
        observed_value="0.0023",
        reproduce_command=".venv/bin/python -c \"print('d_seg=0.0023')\"",
    )
    res = recheck(r, repo_root=_repo_root())
    assert res.status == "verified"
    assert res.ok()


def test_recheck_mismatch_when_value_absent() -> None:
    # The wrong-candidate / phantom-gate signature: command runs but the claimed
    # value is NOT in the output.
    r = _clean_record(
        observed_value="0.0023",
        reproduce_command=".venv/bin/python -c \"print('d_seg=0.7115')\"",
    )
    res = recheck(r, repo_root=_repo_root())
    assert res.status == "mismatch"
    assert not res.ok()


def test_recheck_error_on_nonzero_rc() -> None:
    r = _clean_record(
        observed_value="0.0023",
        reproduce_command=".venv/bin/python -c \"import sys; sys.exit(3)\"",
    )
    res = recheck(r, repo_root=_repo_root())
    assert res.status == "error"


def test_recheck_unverifiable_without_observed_value() -> None:
    r = _clean_record(
        observed_value="",
        reproduce_command=".venv/bin/python -c \"print('x')\"",
    )
    res = recheck(r)
    assert res.status == "unverifiable"


# ---------------------------------------------------------------------------
# Round-trip + durable surface + gate helper
# ---------------------------------------------------------------------------


def test_round_trip_serialization() -> None:
    r = _clean_record()
    rt = AuditProvenanceRecord.from_dict(r.as_dict())
    assert rt == r


def test_from_dict_rejects_wrong_schema() -> None:
    bad = _clean_record().as_dict()
    bad["schema"] = "nope"
    with pytest.raises(ValueError, match="unexpected schema"):
        AuditProvenanceRecord.from_dict(bad)


def test_emit_and_audit_clean_record_no_findings(tmp_path) -> None:
    emit_audit_provenance_record(_clean_record(), repo_root=tmp_path, verify=True)
    findings = audit_provenance_claim_records(repo_root=tmp_path)
    assert findings == []


def test_emit_violating_record_surfaces_finding(tmp_path) -> None:
    # Emit a record with a missing reproduce_command (verify=False so it lands).
    bad = _clean_record(candidate_id="bad_cand", reproduce_command="")
    emit_audit_provenance_record(bad, repo_root=tmp_path, verify=False)
    findings = audit_provenance_claim_records(repo_root=tmp_path)
    assert any(f.candidate_id == "bad_cand" for f in findings)
    assert any("MISSING-REPRODUCE-COMMAND" in f.finding for f in findings)


def test_emit_with_verify_true_refuses_hallucination(tmp_path) -> None:
    with pytest.raises(AuditProvenanceVerifyError):
        emit_audit_provenance_record(
            _clean_record(surface=""), repo_root=tmp_path, verify=True
        )


def test_audit_empty_state_dir_is_clean(tmp_path) -> None:
    assert audit_provenance_claim_records(repo_root=tmp_path) == []


def test_corrupt_record_surfaced_as_finding(tmp_path) -> None:
    state = tmp_path / ".omx" / "state" / "audit_provenance"
    state.mkdir(parents=True)
    (state / "broken.json").write_text("{ not json", encoding="utf-8")
    findings = audit_provenance_claim_records(repo_root=tmp_path)
    assert any(f.candidate_id == "broken" for f in findings)


def test_path_sanitizes_candidate_id(tmp_path) -> None:
    p = audit_provenance_path_for_claim("a/b:c d", repo_root=tmp_path)
    assert "/" not in p.name and ":" not in p.name and " " not in p.name
    assert p.name.endswith(".json")


def test_emitted_record_carries_schema(tmp_path) -> None:
    p = emit_audit_provenance_record(_clean_record(), repo_root=tmp_path)
    payload = json.loads(p.read_text())
    assert payload["schema"] == "audit_provenance_manifest.v1"
    assert payload["surface"] == "live"
