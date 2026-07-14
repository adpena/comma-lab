"""Tests for the curriculum-candidate POOL — the P0 tracked costate class (task #403).

Covers the store (append + validated latest-wins), the NO-FAKE validation contract (status /
form-class / source-anchor / exactly-one-of DSL-leg / typed measured evidence / est_delta_s+axis),
the ranked report + duty queue, the idempotent seed, and the two DSL folds' completeness +
composability.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from tac.witness_dsl import curriculum_candidate_pool as ccp


def _production_receipt(
    candidate: str,
    *,
    axis: str = "cpu",
    evidence_tag: str = "[contest-CPU]",
    hardware_substrate: str = "linux_x86_64_cpu",
) -> dict:
    archive_bytes = 1234
    d_seg = 0.001
    d_pose = 0.0001
    score = 100.0 * d_seg + (10.0 * d_pose) ** 0.5 + 25.0 * archive_bytes / 37_545_489
    return {
        "schema": ccp.PRODUCTION_RECEIPT_SCHEMA,
        "receipt_type": ccp.PRODUCTION_RECEIPT_TYPE,
        "candidate": candidate,
        "verdict": {"status": "ADMITTED", "passed": True, "byte_closed": True},
        "authority": {
            "outcome": "ACCEPTED",
            "axis": axis,
            "evidence_tag": evidence_tag,
            "hardware_substrate": hardware_substrate,
            "research_only": False,
            "promotion_eligible": True,
        },
        "custody": {
            "outcome": "VALID",
            "archive_sha256": "a" * 64,
            "archive_bytes": archive_bytes,
            "runtime_tree_sha256": "b" * 64,
            "upstream_snapshot_sha256": "c" * 64,
        },
        "measurement": {"n_samples": 600, "d_seg": d_seg, "d_pose": d_pose, "score": score},
    }


# ── store round-trip + latest-wins ───────────────────────────────────────────────────────────────
def test_record_and_read_roundtrip(tmp_path):
    p = tmp_path / "pool.jsonl"
    row = ccp.record_candidate(
        "c_x",
        ccp.STATUS_BUILT_NEVER_FIRED,
        form_class="loss-geometry",
        source_anchor="commit abc",
        gate="fire when X",
        dsl_lever="SomeLever",
        path=p,
    )
    assert row["candidate"] == "c_x"
    st = ccp.candidate_status("c_x", path=p)
    assert st is not None
    assert st.status == ccp.STATUS_BUILT_NEVER_FIRED
    assert st.dsl_lever == "SomeLever"
    assert st.in_duty_queue is True


def test_latest_row_wins(tmp_path):
    p = tmp_path / "pool.jsonl"
    ccp.record_candidate(
        "c",
        ccp.STATUS_NEEDS_BUILD,
        form_class="averaging",
        source_anchor="a",
        gate="g",
        dsl_na_reason="unbuilt",
        path=p,
    )
    ccp.record_candidate(
        "c",
        ccp.STATUS_BUILT_NEVER_FIRED,
        form_class="averaging",
        source_anchor="a2",
        gate="g2",
        dsl_lever="Built",
        path=p,
    )
    st = ccp.candidate_status("c", path=p)
    assert st.status == ccp.STATUS_BUILT_NEVER_FIRED  # later row wins
    assert st.dsl_lever == "Built"


def test_read_is_lenient_to_corrupt_lines(tmp_path):
    p = tmp_path / "pool.jsonl"
    ccp.record_candidate(
        "ok", ccp.STATUS_ARMED, form_class="loss-geometry", source_anchor="a", gate="g", dsl_lever="L", path=p
    )
    with p.open("a") as f:
        f.write("not json at all\n")
        f.write('{"candidate": "", "status": "armed"}\n')  # empty candidate skipped
    rows = ccp.pool_report(path=p)
    assert [r["candidate"] for r in rows] == ["ok"]


def test_partial_latest_row_cannot_replace_complete_previous_row(tmp_path):
    p = tmp_path / "pool.jsonl"
    ccp.record_candidate(
        "stable",
        ccp.STATUS_NEEDS_BUILD,
        form_class="averaging",
        source_anchor="complete",
        gate="build first",
        dsl_na_reason="unbuilt",
        path=p,
    )
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"candidate": "stable", "status": ccp.STATUS_ARMED}) + "\n")
    status = ccp.candidate_status("stable", path=p)
    assert status is not None
    assert status.status == ccp.STATUS_NEEDS_BUILD
    assert status.source_anchor == "complete"


def test_research_only_row_cannot_be_declassified_by_missing_or_false_overlay(tmp_path):
    p = tmp_path / "pool.jsonl"
    ccp.record_candidate(
        "research",
        ccp.STATUS_BUILT_NEVER_FIRED,
        form_class="state-evolution",
        source_anchor="research-anchor",
        gate="diagnostic only",
        dsl_na_reason="no provider",
        research_only=True,
        path=p,
    )
    base_overlay = {
        "candidate": "research",
        "status": ccp.STATUS_ARMED,
        "form_class": "state-evolution",
        "source_anchor": "invalid-promotion",
        "gate": "must not win",
        "dsl_lever": "LiveLever",
        "dsl_na_reason": None,
        "blockers": [],
    }
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(base_overlay) + "\n")  # missing research_only
        fh.write(json.dumps({**base_overlay, "research_only": False}) + "\n")
    status = ccp.candidate_status("research", path=p)
    assert status is not None
    assert status.research_only is True
    assert status.status == ccp.STATUS_BUILT_NEVER_FIRED
    assert status.source_anchor == "research-anchor"


@pytest.mark.parametrize("malformed_blockers", [None, "not-a-sequence-of-blockers"])
def test_malformed_blockers_overlay_is_skipped_without_typed_read_crash(tmp_path, malformed_blockers):
    p = tmp_path / "pool.jsonl"
    ccp.record_candidate(
        "stable",
        ccp.STATUS_NEEDS_BUILD,
        form_class="averaging",
        source_anchor="complete",
        gate="build first",
        dsl_na_reason="unbuilt",
        path=p,
    )
    malformed = {
        "candidate": "stable",
        "status": ccp.STATUS_ARMED,
        "form_class": "averaging",
        "source_anchor": "malformed",
        "gate": "bad blockers",
        "dsl_lever": "LiveLever",
        "dsl_na_reason": None,
        "blockers": malformed_blockers,
    }
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(malformed) + "\n")
    status = ccp.candidate_status("stable", path=p)
    assert status is not None
    assert status.status == ccp.STATUS_NEEDS_BUILD
    assert status.blockers == ()


@pytest.mark.parametrize("evidence_kind", [None, ccp.EVIDENCE_RESEARCH_DIAGNOSTIC])
def test_malformed_measured_production_overlay_cannot_mint_authority(tmp_path, evidence_kind):
    p = tmp_path / "pool.jsonl"
    ccp.record_candidate(
        "stable",
        ccp.STATUS_NEEDS_BUILD,
        form_class="averaging",
        source_anchor="complete",
        gate="build first",
        dsl_na_reason="unbuilt",
        path=p,
    )
    malformed = {
        "candidate": "stable",
        "status": ccp.STATUS_MEASURED,
        "form_class": "averaging",
        "source_anchor": "arbitrary verdict path",
        "gate": "must not gain authority",
        "dsl_lever": "LiveLever",
        "dsl_na_reason": None,
        "verdict_ref": "arbitrary.json",
        "evidence_kind": evidence_kind,
        "research_only": False,
        "blockers": [],
    }
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(malformed) + "\n")
    status = ccp.candidate_status("stable", path=p)
    assert status is not None
    assert status.status == ccp.STATUS_NEEDS_BUILD
    assert status.source_anchor == "complete"


def test_missing_candidate_returns_none(tmp_path):
    assert ccp.candidate_status("nope", path=tmp_path / "empty.jsonl") is None


# ── NO-FAKE validation contract ──────────────────────────────────────────────────────────────────
def test_invalid_status_rejected(tmp_path):
    with pytest.raises(ValueError, match="invalid status"):
        ccp.record_candidate(
            "c", "flying", form_class="averaging", source_anchor="a", gate="g", dsl_lever="L", path=tmp_path / "p.jsonl"
        )


def test_invalid_form_class_rejected(tmp_path):
    with pytest.raises(ValueError, match="invalid form_class"):
        ccp.record_candidate(
            "c",
            ccp.STATUS_ARMED,
            form_class="magic",
            source_anchor="a",
            gate="g",
            dsl_lever="L",
            path=tmp_path / "p.jsonl",
        )


def test_source_anchor_required(tmp_path):
    with pytest.raises(ValueError, match="source_anchor is required"):
        ccp.record_candidate(
            "c",
            ccp.STATUS_ARMED,
            form_class="averaging",
            source_anchor="",
            gate="g",
            dsl_lever="L",
            path=tmp_path / "p.jsonl",
        )


def test_exactly_one_dsl_leg_required_both(tmp_path):
    with pytest.raises(ValueError, match="exactly one of dsl_lever"):
        ccp.record_candidate(
            "c",
            ccp.STATUS_ARMED,
            form_class="averaging",
            source_anchor="a",
            gate="g",
            dsl_lever="L",
            dsl_na_reason="also",
            path=tmp_path / "p.jsonl",
        )


def test_exactly_one_dsl_leg_required_neither(tmp_path):
    with pytest.raises(ValueError, match="exactly one of dsl_lever"):
        ccp.record_candidate(
            "c", ccp.STATUS_ARMED, form_class="averaging", source_anchor="a", gate="g", path=tmp_path / "p.jsonl"
        )


def test_measured_requires_verdict_ref(tmp_path):
    with pytest.raises(ValueError, match="requires a verdict_ref"):
        ccp.record_candidate(
            "c",
            ccp.STATUS_MEASURED,
            form_class="averaging",
            source_anchor="a",
            gate="g",
            dsl_lever="L",
            path=tmp_path / "p.jsonl",
        )


@pytest.mark.parametrize(
    ("axis", "evidence_tag", "hardware_substrate"),
    [
        ("cpu", "[contest-CPU]", "linux_x86_64_cpu"),
        ("cuda", "[contest-CUDA]", "linux_x86_64_t4"),
    ],
)
def test_measured_with_allowlisted_semantically_authoritative_receipt_ok(
    tmp_path, monkeypatch, axis, evidence_tag, hardware_substrate
):
    repo = tmp_path / "repo"
    receipt = repo / "byteclose" / "verdict.json"
    receipt.parent.mkdir(parents=True)
    receipt_bytes = (
        json.dumps(
            _production_receipt(
                "c",
                axis=axis,
                evidence_tag=evidence_tag,
                hardware_substrate=hardware_substrate,
            ),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    receipt.write_bytes(receipt_bytes)
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    monkeypatch.setattr(ccp, "_REPO_ROOT", repo)
    monkeypatch.setattr(
        ccp,
        "_TRUSTED_PRODUCTION_RECEIPTS",
        {("c", "byteclose/verdict.json"): receipt_sha256},
    )

    pool_path = tmp_path / "p.jsonl"
    row = ccp.record_candidate(
        "c",
        ccp.STATUS_MEASURED,
        form_class="averaging",
        source_anchor="a",
        gate="g",
        dsl_lever="L",
        verdict_ref="byteclose/verdict.json",
        evidence_kind=ccp.EVIDENCE_BYTE_CLOSED,
        trusted_receipt_sha256=receipt_sha256,
        path=pool_path,
    )
    assert row["status"] == ccp.STATUS_MEASURED
    assert row["verdict_ref"] == "byteclose/verdict.json"
    assert row["evidence_kind"] == ccp.EVIDENCE_BYTE_CLOSED
    assert row["trusted_receipt_sha256"] == receipt_sha256
    assert ccp.candidate_status("c", path=pool_path) is not None


def test_readme_hash_attack_cannot_mint_production_authority(tmp_path):
    readme = ccp._REPO_ROOT / "README.md"
    assert readme.is_file()
    with pytest.raises(ValueError, match="not in the code-reviewed trust registry"):
        ccp.record_candidate(
            "readme-attack",
            ccp.STATUS_MEASURED,
            form_class="averaging",
            source_anchor="caller-controlled",
            gate="must fail closed",
            dsl_lever="L",
            verdict_ref="README.md",
            evidence_kind=ccp.EVIDENCE_BYTE_CLOSED,
            trusted_receipt_sha256=hashlib.sha256(readme.read_bytes()).hexdigest(),
            path=tmp_path / "p.jsonl",
        )


def test_forged_supported_schema_and_matching_hash_still_need_trust_root(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    receipt = repo / "forged.json"
    receipt.write_text(json.dumps(_production_receipt("forged"), sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    monkeypatch.setattr(ccp, "_REPO_ROOT", repo)
    with pytest.raises(ValueError, match="not in the code-reviewed trust registry"):
        ccp.record_candidate(
            "forged",
            ccp.STATUS_MEASURED,
            form_class="averaging",
            source_anchor="caller-controlled",
            gate="must fail closed",
            dsl_lever="L",
            verdict_ref="forged.json",
            evidence_kind=ccp.EVIDENCE_BYTE_CLOSED,
            trusted_receipt_sha256=digest,
            path=tmp_path / "p.jsonl",
        )


def test_allowlisted_old_synthetic_archive_sha_fixture_is_semantically_rejected(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    receipt = repo / "synthetic.json"
    receipt.write_bytes(b'{"archive_sha256":"authority"}\n')
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    monkeypatch.setattr(ccp, "_REPO_ROOT", repo)
    monkeypatch.setattr(ccp, "_TRUSTED_PRODUCTION_RECEIPTS", {("synthetic", "synthetic.json"): digest})
    with pytest.raises(ValueError, match="unsupported production receipt schema"):
        ccp.record_candidate(
            "synthetic",
            ccp.STATUS_MEASURED,
            form_class="averaging",
            source_anchor="old fixture",
            gate="must fail closed",
            dsl_lever="L",
            verdict_ref="synthetic.json",
            evidence_kind=ccp.EVIDENCE_BYTE_CLOSED,
            trusted_receipt_sha256=digest,
            path=tmp_path / "p.jsonl",
        )


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("verdict", "passed", False, "status='ADMITTED' and passed=true"),
        ("authority", "outcome", "REFUSED", "authority.outcome"),
        ("custody", "outcome", "INVALID", "custody.outcome"),
        ("authority", "hardware_substrate", "macos_arm64", "canonical authority custody refused"),
    ],
)
def test_allowlisted_receipt_still_requires_admission_authority_and_custody_outcomes(
    tmp_path, monkeypatch, section, field, value, message
):
    repo = tmp_path / "repo"
    repo.mkdir()
    payload = _production_receipt("semantic-refuse")
    payload[section][field] = value
    receipt = repo / "receipt.json"
    receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    monkeypatch.setattr(ccp, "_REPO_ROOT", repo)
    monkeypatch.setattr(
        ccp,
        "_TRUSTED_PRODUCTION_RECEIPTS",
        {("semantic-refuse", "receipt.json"): digest},
    )
    with pytest.raises(ValueError, match=message):
        ccp.record_candidate(
            "semantic-refuse",
            ccp.STATUS_MEASURED,
            form_class="averaging",
            source_anchor="reviewed receipt",
            gate="must fail closed",
            dsl_lever="L",
            verdict_ref="receipt.json",
            evidence_kind=ccp.EVIDENCE_BYTE_CLOSED,
            trusted_receipt_sha256=digest,
            path=tmp_path / "p.jsonl",
        )


def test_measured_production_rejects_missing_receipt_sha(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    receipt = repo / "receipt.json"
    repo.mkdir()
    receipt.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(ccp, "_REPO_ROOT", repo)
    with pytest.raises(ValueError, match="trusted_receipt_sha256 is required"):
        ccp.record_candidate(
            "missing-sha",
            ccp.STATUS_MEASURED,
            form_class="averaging",
            source_anchor="receipt",
            gate="byte-close",
            dsl_lever="L",
            verdict_ref="receipt.json",
            evidence_kind=ccp.EVIDENCE_BYTE_CLOSED,
            path=tmp_path / "p.jsonl",
        )


def test_measured_production_rejects_nonexistent_receipt(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(ccp, "_REPO_ROOT", repo)
    monkeypatch.setattr(ccp, "_TRUSTED_PRODUCTION_RECEIPTS", {("missing-receipt", "missing.json"): "0" * 64})
    with pytest.raises(ValueError, match="receipt path is unavailable"):
        ccp.record_candidate(
            "missing-receipt",
            ccp.STATUS_MEASURED,
            form_class="averaging",
            source_anchor="receipt",
            gate="byte-close",
            dsl_lever="L",
            verdict_ref="missing.json",
            evidence_kind=ccp.EVIDENCE_BYTE_CLOSED,
            trusted_receipt_sha256="0" * 64,
            path=tmp_path / "p.jsonl",
        )


def test_measured_production_rejects_wrong_receipt_hash(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "receipt.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(ccp, "_REPO_ROOT", repo)
    monkeypatch.setattr(ccp, "_TRUSTED_PRODUCTION_RECEIPTS", {("wrong-sha", "receipt.json"): "0" * 64})
    with pytest.raises(ValueError, match="receipt SHA-256 mismatch"):
        ccp.record_candidate(
            "wrong-sha",
            ccp.STATUS_MEASURED,
            form_class="averaging",
            source_anchor="receipt",
            gate="byte-close",
            dsl_lever="L",
            verdict_ref="receipt.json",
            evidence_kind=ccp.EVIDENCE_BYTE_CLOSED,
            trusted_receipt_sha256="0" * 64,
            path=tmp_path / "p.jsonl",
        )


def test_measured_production_rejects_symlink_receipt(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    (repo / "receipt.json").symlink_to(target.name)
    monkeypatch.setattr(ccp, "_REPO_ROOT", repo)
    target_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    monkeypatch.setattr(ccp, "_TRUSTED_PRODUCTION_RECEIPTS", {("symlink", "receipt.json"): target_sha})
    with pytest.raises(ValueError, match="contains a symlink"):
        ccp.record_candidate(
            "symlink",
            ccp.STATUS_MEASURED,
            form_class="averaging",
            source_anchor="receipt",
            gate="byte-close",
            dsl_lever="L",
            verdict_ref="receipt.json",
            evidence_kind=ccp.EVIDENCE_BYTE_CLOSED,
            trusted_receipt_sha256=target_sha,
            path=tmp_path / "p.jsonl",
        )


def test_measured_production_tamper_invalidates_stored_authority(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    receipt = repo / "receipt.json"
    original = json.dumps(_production_receipt("tamper"), sort_keys=True).encode("utf-8") + b"\n"
    receipt.write_bytes(original)
    monkeypatch.setattr(ccp, "_REPO_ROOT", repo)
    original_sha = hashlib.sha256(original).hexdigest()
    monkeypatch.setattr(ccp, "_TRUSTED_PRODUCTION_RECEIPTS", {("tamper", "receipt.json"): original_sha})
    pool_path = tmp_path / "p.jsonl"
    ccp.record_candidate(
        "tamper",
        ccp.STATUS_MEASURED,
        form_class="averaging",
        source_anchor="receipt",
        gate="byte-close",
        dsl_lever="L",
        verdict_ref="receipt.json",
        evidence_kind=ccp.EVIDENCE_BYTE_CLOSED,
        trusted_receipt_sha256=original_sha,
        path=pool_path,
    )
    assert ccp.candidate_status("tamper", path=pool_path) is not None

    receipt.write_bytes(b'{"verdict":"tampered"}\n')
    assert ccp.candidate_status("tamper", path=pool_path) is None


def test_untrusted_production_seed_cannot_bypass_receipt_custody(tmp_path, monkeypatch):
    malicious_seed = {
        "candidate": "seed-bypass",
        "status": ccp.STATUS_MEASURED,
        "form_class": "averaging",
        "source_anchor": "untrusted seed",
        "gate": "must fail closed",
        "dsl_lever": "L",
        "verdict_ref": "missing.json",
        "evidence_kind": ccp.EVIDENCE_BYTE_CLOSED,
        "trusted_receipt_sha256": "0" * 64,
    }
    monkeypatch.setattr(ccp, "_SEED", (malicious_seed,))
    monkeypatch.setattr(ccp, "_REPO_ROOT", tmp_path)
    assert ccp._read_pool(path=tmp_path / "absent.jsonl", include_seed=True) == {}


def test_unmeasured_legacy_verdict_ref_remains_non_authority_compatible(tmp_path):
    p = tmp_path / "pool.jsonl"
    legacy = {
        "candidate": "legacy-unmeasured",
        "status": ccp.STATUS_NEEDS_BUILD,
        "form_class": "averaging",
        "source_anchor": "legacy source",
        "gate": "build first",
        "dsl_lever": None,
        "dsl_na_reason": "not built",
        "verdict_ref": "historical/nonexistent-receipt.json",
    }
    p.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    status = ccp.candidate_status("legacy-unmeasured", path=p)
    assert status is not None
    assert status.status == ccp.STATUS_NEEDS_BUILD
    assert status.evidence_kind is None


def test_measured_verdict_ref_alone_cannot_mint_authority(tmp_path):
    with pytest.raises(ValueError, match="requires evidence_kind='byte_closed'"):
        ccp.record_candidate(
            "untyped",
            ccp.STATUS_MEASURED,
            form_class="averaging",
            source_anchor="verdict path only",
            gate="must fail closed",
            dsl_lever="L",
            verdict_ref="arbitrary.json",
            path=tmp_path / "p.jsonl",
        )


def test_measured_nonresearch_rejects_proxy_evidence_kind(tmp_path):
    with pytest.raises(ValueError, match="requires evidence_kind='byte_closed'"):
        ccp.record_candidate(
            "proxy",
            ccp.STATUS_MEASURED,
            form_class="averaging",
            source_anchor="diagnostic",
            gate="not byte closed",
            dsl_lever="L",
            verdict_ref="diagnostic.json",
            evidence_kind=ccp.EVIDENCE_RESEARCH_DIAGNOSTIC,
            path=tmp_path / "p.jsonl",
        )


def test_measured_research_diagnostic_is_explicitly_accepted_and_nonfireable(tmp_path):
    p = tmp_path / "p.jsonl"
    row = ccp.record_candidate(
        "diagnostic",
        ccp.STATUS_MEASURED,
        form_class="preconditioning",
        source_anchor="n600 receipt",
        gate="research only",
        dsl_na_reason="no production activation",
        verdict_ref="measurement_receipt.json",
        evidence_kind=ccp.EVIDENCE_RESEARCH_DIAGNOSTIC,
        research_only=True,
        path=p,
    )
    assert row["evidence_kind"] == ccp.EVIDENCE_RESEARCH_DIAGNOSTIC
    assert ccp.candidate_status("diagnostic", path=p).in_duty_queue is False


def test_negative_est_delta_s_rejected(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        ccp.record_candidate(
            "c",
            ccp.STATUS_ARMED,
            form_class="averaging",
            source_anchor="a",
            gate="g",
            dsl_lever="L",
            est_delta_s=-0.1,
            axis="d_seg",
            path=tmp_path / "p.jsonl",
        )


def test_est_delta_s_requires_axis(tmp_path):
    with pytest.raises(ValueError, match="requires axis"):
        ccp.record_candidate(
            "c",
            ccp.STATUS_ARMED,
            form_class="averaging",
            source_anchor="a",
            gate="g",
            dsl_lever="L",
            est_delta_s=0.01,
            path=tmp_path / "p.jsonl",
        )


# ── ranking + duty queue ─────────────────────────────────────────────────────────────────────────
def test_report_ranks_built_never_fired_first(tmp_path):
    p = tmp_path / "p.jsonl"
    ccp.record_candidate(
        "armed1", ccp.STATUS_ARMED, form_class="loss-geometry", source_anchor="a", gate="g", dsl_lever="A", path=p
    )
    ccp.record_candidate(
        "needs1",
        ccp.STATUS_NEEDS_BUILD,
        form_class="averaging",
        source_anchor="a",
        gate="g",
        dsl_na_reason="unbuilt",
        path=p,
    )
    ccp.record_candidate(
        "bnf1",
        ccp.STATUS_BUILT_NEVER_FIRED,
        form_class="loss-geometry",
        source_anchor="a",
        gate="g",
        dsl_lever="B",
        path=p,
    )
    ccp.record_candidate(
        "ref1",
        ccp.STATUS_REFORMULATION_QUEUE,
        form_class="state-evolution",
        source_anchor="a",
        gate="g",
        dsl_na_reason="reform",
        path=p,
    )
    order = [r["candidate"] for r in ccp.pool_report(path=p)]
    assert order == ["bnf1", "needs1", "ref1", "armed1"]


def test_duty_queue_excludes_armed_measured_retired(tmp_path):
    p = tmp_path / "p.jsonl"
    ccp.record_candidate(
        "bnf",
        ccp.STATUS_BUILT_NEVER_FIRED,
        form_class="loss-geometry",
        source_anchor="a",
        gate="g",
        dsl_lever="B",
        path=p,
    )
    ccp.record_candidate(
        "armed", ccp.STATUS_ARMED, form_class="loss-geometry", source_anchor="a", gate="g", dsl_lever="A", path=p
    )
    ccp.record_candidate(
        "ret",
        ccp.STATUS_RETIRED,
        form_class="optimizer-stage",
        source_anchor="a",
        gate="g",
        dsl_na_reason="law",
        path=p,
    )
    ccp.record_candidate(
        "research",
        ccp.STATUS_BUILT_NEVER_FIRED,
        form_class="state-evolution",
        source_anchor="a",
        gate="g",
        dsl_na_reason="no live provider",
        research_only=True,
        path=p,
    )
    duty = {r["candidate"] for r in ccp.duty_to_measure_pool(path=p)}
    assert duty == {"bnf"}
    assert ccp.candidate_status("research", path=p).in_duty_queue is False


def test_est_delta_s_breaks_ties_within_status(tmp_path):
    p = tmp_path / "p.jsonl"
    ccp.record_candidate(
        "low",
        ccp.STATUS_BUILT_NEVER_FIRED,
        form_class="loss-geometry",
        source_anchor="a",
        gate="g",
        dsl_lever="L",
        est_delta_s=0.01,
        axis="d_seg",
        path=p,
    )
    ccp.record_candidate(
        "high",
        ccp.STATUS_BUILT_NEVER_FIRED,
        form_class="loss-geometry",
        source_anchor="a",
        gate="g",
        dsl_lever="H",
        est_delta_s=0.05,
        axis="d_seg",
        path=p,
    )
    ccp.record_candidate(
        "none",
        ccp.STATUS_BUILT_NEVER_FIRED,
        form_class="loss-geometry",
        source_anchor="a",
        gate="g",
        dsl_lever="N",
        path=p,
    )
    order = [r["candidate"] for r in ccp.pool_report(path=p)]
    assert order == ["high", "low", "none"]  # est desc, then None last


# ── seed (idempotent, honest statuses) ───────────────────────────────────────────────────────────
def test_seed_is_idempotent(tmp_path):
    p = tmp_path / "p.jsonl"
    n1 = ccp.seed_default_pool(path=p)
    assert n1 == len(ccp._SEED) > 0
    n2 = ccp.seed_default_pool(path=p)
    assert n2 == 0  # re-seed writes nothing


def test_clean_checkout_seed_reads_tracked_k2_and_sparse_receipts_without_runtime_jsonl(tmp_path):
    rows = ccp._read_pool(path=tmp_path / "absent.jsonl", include_seed=True)
    for candidate in ("p0_guarded_exact_costate_reuse_k2", "p0_sparse_adjoint_dense_fullrank"):
        assert candidate in rows
        assert rows[candidate]["status"] == ccp.STATUS_MEASURED
        assert rows[candidate]["research_only"] is True
        assert not any("RECEIPT_CUSTODY_BLOCKED" in blocker for blocker in rows[candidate]["blockers"])


def test_missing_pinned_research_receipt_surfaces_blocker_instead_of_losing_seed(tmp_path, monkeypatch):
    seed = next(row for row in ccp._SEED if row["candidate"] == "p0_guarded_exact_costate_reuse_k2")
    monkeypatch.setattr(ccp, "_SEED", (seed,))
    monkeypatch.setattr(ccp, "_REPO_ROOT", tmp_path)
    rows = ccp._read_pool(path=tmp_path / "absent.jsonl", include_seed=True)
    blocked = rows["p0_guarded_exact_costate_reuse_k2"]
    assert blocked["status"] == ccp.STATUS_REFORMULATION_QUEUE
    assert blocked["evidence_kind"] is None
    assert blocked["activation_status"] == "RECEIPT_CUSTODY_BLOCKED_NO_PRODUCTION_AUTHORITY"
    assert "UNVERIFIED_CUSTODY_RESEARCH_SIGNAL" in blocked["justification"]
    assert any("RECEIPT_CUSTODY_BLOCKED" in blocker for blocker in blocked["blockers"])


def test_seed_measured_rows_have_verdict_and_are_research_only(tmp_path):
    # NO-FAKE: committed measured research findings cite their verdict and cannot activate production.
    p = tmp_path / "p.jsonl"
    ccp.seed_default_pool(path=p)
    measured = [r for r in ccp.pool_report(path=p) if r["status"] == ccp.STATUS_MEASURED]
    assert measured
    for row in measured:
        assert row["verdict_ref"]
        assert row["evidence_kind"] in ccp.VALID_EVIDENCE_KINDS
    p0_rows = {row["candidate"]: row for row in measured if row["candidate"].startswith("p0_")}
    assert p0_rows
    for row in p0_rows.values():
        assert row["research_only"] is True
        assert "NO_LIVE" in row["activation_status"] or "NO_PRODUCTION" in row["activation_status"]


def test_seed_every_row_has_exactly_one_dsl_leg(tmp_path):
    p = tmp_path / "p.jsonl"
    ccp.seed_default_pool(path=p)
    for r in ccp.pool_report(path=p):
        assert bool(r.get("dsl_lever")) != bool(r.get("dsl_na_reason")), r["candidate"]


def test_seed_includes_the_two_folds(tmp_path):
    p = tmp_path / "p.jsonl"
    ccp.seed_default_pool(path=p)
    by_cand = {r["candidate"]: r for r in ccp.pool_report(path=p)}
    assert by_cand["hardness_oversample_lever5"]["dsl_lever"] == "HardnessOversample"
    assert by_cand["head_geometry_218_etf_am"]["dsl_lever"] == "HeadGeometry"


def test_pool_summary_shape(tmp_path):
    p = tmp_path / "p.jsonl"
    ccp.seed_default_pool(path=p)
    summ = ccp.pool_summary(path=p)
    assert summ["total"] == len(ccp._SEED)
    assert summ["owed"] == sum(
        1
        for r in ccp.pool_report(path=p)
        if r["status"] in ("built-never-fired", "needs-build", "reformulation-queue") and not r.get("research_only")
    )
    assert len(summ["top_fireable"]) <= 6
    assert all(not row.get("research_only") for row in summ["top_fireable"])
    assert all(row.get("research_only") for row in summ["research_signals"])


def test_p0_research_candidates_are_discoverable_with_honest_states():
    by_candidate = {row["candidate"]: row for row in ccp.pool_report()}
    summary = ccp.pool_summary()
    fireable_names = {row["candidate"] for row in summary["top_fireable"]}
    research_names = {row["candidate"] for row in summary["research_signals"]}

    reuse = by_candidate["p0_guarded_exact_costate_reuse_k2"]
    assert reuse["status"] == ccp.STATUS_MEASURED
    assert reuse["evidence_kind"] == ccp.EVIDENCE_RESEARCH_DIAGNOSTIC
    assert reuse["research_only"] is True
    assert reuse["realized_speedup_factor"] == 1.0
    assert reuse["derived_cost_reduction_fraction"] == 0.0
    assert reuse["trusted_receipt_sha256"] == ("30ce7e5e23b10cb15c52a89debc57b0bf5349be16ed9cb0e97c3974579465ff7")
    assert reuse["verdict_ref"] == (".omx/research/p0_costate_reuse_k2_corrected_adjudication_receipt_20260714.json")
    assert "NOT_ADMITTED corrected n600 gate" in reuse["gate"]
    assert "308/456" in reuse["gate"]
    assert "renderer-gradient relL2 < 1 passed 456/456" in reuse["gate"]
    assert "456/600 behavioral full-facet accepts (p=0.76)" in reuse["justification"]
    assert "NOT a global throughput win" in reuse["verdict_scope"]
    assert "HEAD e59f69a79c dominant 95%-kill forward-only frozen authority" in reuse["verdict_scope"]
    assert "pointer_moved=false" in reuse["verdict_scope"]
    assert "score_claim=false" in reuse["verdict_scope"]
    assert "FIDELITY_BLOCKED_PENDING_NEW_FORMULATION" in reuse["verdict_scope"]
    assert "UNKNOWN_IN_LOOP_TIMER_OWED" not in reuse["verdict_scope"]
    assert reuse["activation_status"] == ("NOT_ADMITTED_DEFAULT_OFF_NO_LIVE_PROVIDER_OR_RESUME_REGISTRATION")
    counterfactual = next(
        blocker
        for blocker in reuse["blockers"]
        if blocker.startswith("DERIVED_COUNTERFACTUAL_BEHIND_FAILED_FIDELITY_GATE")
    )
    assert "1.6129032258064517x" in counterfactual
    assert "reduction 0.38" in counterfactual
    assert "never achieved, admitted, global, or wall-clock" in counterfactual
    assert "canonical resume-registry integration absent" in reuse["blockers"]
    assert any("timer is owed only after a fresh formulation passes fidelity admission" in b for b in reuse["blockers"])
    assert reuse["candidate"] not in fireable_names
    assert reuse["candidate"] in research_names
    assert ccp.candidate_status(reuse["candidate"]).in_duty_queue is False

    sparse = by_candidate["p0_sparse_adjoint_dense_fullrank"]
    assert sparse["status"] == ccp.STATUS_MEASURED
    assert sparse["realized_speedup_factor"] == 1.0
    assert sparse["derived_cost_reduction_fraction"] is None
    assert sparse["trusted_receipt_sha256"] == ("bc3e68c139f8472cd43badeb6ce70d3270f2a30945c714a0b2c1d8da57eeb771")
    assert sparse["verdict_ref"] == ".omx/research/p0_sparse_adjoint_costate_vjp_20260713.md"
    assert "NO_GO_DENSE_FULLRANK" in sparse["activation_status"]
    assert "source-bound task455 n600 replay" in sparse["verdict_scope"]
    assert sparse["candidate"] not in fireable_names
    assert sparse["candidate"] in research_names

    terminal = by_candidate["p0_terminal_exact_metric_396_costate_skip"]
    assert terminal["status"] == ccp.STATUS_MEASURED
    assert terminal["realized_speedup_factor"] is None
    assert terminal["derived_cost_reduction_fraction"] == 1.0
    assert terminal["trusted_receipt_sha256"] == ("17574857da5ff862e520140977e988197962f009d6870d23fe3071c398112a9c")
    assert "route on the pinned n600 objective" in terminal["verdict_scope"]
    assert "SPSA/ES effective-dimension certificate unadmitted" in terminal["blockers"]
    assert terminal["activation_status"] == "ROUTE_LOCAL_ONLY_NO_LIVE_TRAINER_ACTIVATION"
    assert terminal["candidate"] not in fireable_names
    assert terminal["candidate"] in research_names


def test_reviewed_legacy_research_rows_remain_sense_only_with_untyped_overlay(tmp_path, monkeypatch):
    p = tmp_path / "pool.jsonl"
    legacy_overlay = {
        "agent": "codex",
        "axis": None,
        "candidate": "dig_s1_query_real_disagreement_audit_policy",
        "dsl_lever": ("witness_dsl.replace_round5_deeper_nonlinear_policy.ReplaceRound5DeeperNonlinearPolicy"),
        "dsl_na_reason": None,
        "est_delta_s": None,
        "form_class": "state-evolution",
        "gate": (
            "live admission requires passing localizer, on-policy transition custody, "
            "preserved 4% targeted plus 1% randomized-audit propensities, and explicit "
            "probability calibration"
        ),
        "justification": (
            "MEASURED research-only error-ranking gate passes (189.813x high/low error; "
            "Spearman 0.865610; positive audit propensity), but ensemble ECE 0.186204 and "
            "the localizer primary gate fails, so live remains REFUSE"
        ),
        "owner": "lane_replace_round5_deeper_nonlinear_20260713",
        "slot": "DIG-S1-QUERY-REAL-CALIBRATION",
        "source_anchor": ".omx/research/replace_round5_deeper_nonlinear_20260713.md",
        "status": ccp.STATUS_BUILT_NEVER_FIRED,
        "ts": "2026-07-13T21:06:59Z",
        "verdict_ref": ("experiments/results/replace_round5_deeper_nonlinear_20260713/receipt.json"),
    }
    p.write_text(json.dumps(legacy_overlay) + "\n", encoding="utf-8")
    monkeypatch.setattr(ccp, "POOL_PATH", p)

    duty_names = {row["candidate"] for row in ccp.duty_to_measure_pool()}
    summary = ccp.pool_summary()
    fireable_names = {row["candidate"] for row in summary["top_fireable"]}
    research_names = {row["candidate"] for row in summary["research_signals"]}
    reviewed = {
        "pose_inverse_carrier_distill",
        "dig_s1_query_real_disagreement_audit_policy",
    }
    assert reviewed.isdisjoint(duty_names)
    assert reviewed.isdisjoint(fireable_names)
    assert reviewed <= research_names

    dig = ccp.candidate_status("dig_s1_query_real_disagreement_audit_policy")
    assert dig is not None
    assert dig.status == ccp.STATUS_BUILT_NEVER_FIRED
    assert dig.research_only is True
    assert dig.activation_status == "REFUSE_LOCALIZER_AND_ECE_GATES_FAILED"
    assert dig.blockers == (
        "localizer primary gate fails",
        "ensemble ECE 0.186204 fails explicit probability calibration",
    )

    pose = ccp.candidate_status("pose_inverse_carrier_distill")
    assert pose is not None
    assert pose.status == ccp.STATUS_NEEDS_BUILD
    assert pose.research_only is True
    assert pose.activation_status == "RESEARCH_ONLY_ADVISORY_NO_PRODUCTION_ACTIVATION"


def test_research_metadata_roundtrips_through_typed_status(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    receipt = repo / "receipt.json"
    receipt_bytes = b'{"research_only":true}\n'
    receipt.write_bytes(receipt_bytes)
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    monkeypatch.setattr(ccp, "_REPO_ROOT", repo)
    p = tmp_path / "p.jsonl"
    ccp.record_candidate(
        "research",
        ccp.STATUS_MEASURED,
        form_class="state-evolution",
        source_anchor="receipt",
        gate="research-only",
        dsl_na_reason="no activation",
        verdict_ref="receipt.json",
        evidence_kind=ccp.EVIDENCE_RESEARCH_DIAGNOSTIC,
        research_only=True,
        authority_axis="[macOS-CPU advisory]",
        verdict_scope="one formulation",
        activation_status="NO_PRODUCTION_ACTIVATION",
        realized_speedup_factor=1.0,
        derived_cost_reduction_fraction=0.25,
        trusted_receipt_sha256=receipt_sha256,
        blockers=("live provider absent",),
        path=p,
    )
    status = ccp.candidate_status("research", path=p)
    assert status is not None
    assert status.research_only is True
    assert status.evidence_kind == ccp.EVIDENCE_RESEARCH_DIAGNOSTIC
    assert status.authority_axis == "[macOS-CPU advisory]"
    assert status.verdict_scope == "one formulation"
    assert status.realized_speedup_factor == 1.0
    assert status.derived_cost_reduction_fraction == 0.25
    assert status.trusted_receipt_sha256 == receipt_sha256
    assert status.blockers == ("live provider absent",)
    assert status.in_duty_queue is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"research_only": "yes"}, "research_only must be a bool"),
        ({"realized_speedup_factor": -0.1}, "must be non-negative"),
        ({"derived_cost_reduction_fraction": 1.1}, "must be in"),
        ({"trusted_receipt_sha256": "bad"}, "lowercase SHA-256"),
        ({"blockers": ("",)}, "non-empty strings"),
        ({"blockers": "oops"}, "non-string sequence"),
        ({"blockers": None}, "non-string sequence"),
    ],
)
def test_research_metadata_validation(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        ccp.record_candidate(
            "research",
            ccp.STATUS_BUILT_NEVER_FIRED,
            form_class="state-evolution",
            source_anchor="source",
            gate="gate",
            dsl_na_reason="research-only",
            path=tmp_path / "p.jsonl",
            **kwargs,
        )


def test_seed_dsl_lever_rows_reference_real_or_documented_factories(tmp_path):
    # Every seeded row that claims a dsl_lever must name a real held factory OR be one of the
    # documented-elsewhere factory names (owner-scoped). We assert the TWO folds this landing built
    # are actually held (the ones we are responsible for); others may live in sibling landings.
    from tac.witness_dsl.lever_registry import lever_factories

    held = set(lever_factories().keys())
    assert "HardnessOversample" in held
    assert "HeadGeometry" in held


# ── DSL folds: completeness shrink + composability ───────────────────────────────────────────────
def test_folds_hold_the_previously_unmapped_flags():
    from tac.witness_dsl.lever_registry import completeness, lever_factories

    lf = lever_factories()
    assert lf["HardnessOversample"] == frozenset(
        {"--hardness-oversample", "--hardness-weighted", "--hardness-source", "--hardness-power", "--hardness-band"}
    )
    assert lf["HeadGeometry"] == frozenset({"--head", "--additive-margin"})
    c = completeness()
    for flag in (
        "--hardness-oversample",
        "--hardness-weighted",
        "--hardness-source",
        "--hardness-power",
        "--hardness-band",
        "--head",
        "--additive-margin",
    ):
        assert flag not in c.unmapped, flag


def test_folds_are_composable_by_bare_name():
    from tac.witness_dsl.lever_registry import name_composable_levers, resolve_composable_lever

    comp = name_composable_levers()
    assert "HardnessOversample" in comp
    assert "HeadGeometry" in comp
    # armed defaults engage the mechanism (oversample>0, ETF head) — not a byte-identical no-op arm.
    assert resolve_composable_lever("HardnessOversample").overrides["--hardness-oversample"] == 0.5
    assert resolve_composable_lever("HeadGeometry").overrides["--head"] == "etf"


def test_fold_arg_validation():
    from tac.witness_dsl.curriculum_dsl import HardnessOversample, HeadGeometry

    with pytest.raises(ValueError, match="hardness-source"):
        HardnessOversample(source="bogus")
    with pytest.raises(ValueError, match="head must be"):
        HeadGeometry(head="bogus")


# ── digest surfacing (pure formatter + fail-open section) ────────────────────────────────────────
def test_digest_formatter_leads_with_counts_and_markers():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    from costate_digest import format_curriculum_pool_line

    summary = {
        "total": 3,
        "owed": 2,
        "counts": {"built-never-fired": 1, "needs-build": 1, "reformulation-queue": 0, "armed": 1},
        "top_fireable": [
            {"candidate": "bnf", "status": "built-never-fired", "dsl_lever": "L"},
            {"candidate": "nb", "status": "needs-build", "dsl_lever": None, "dsl_na_reason": "unbuilt"},
        ],
        "research_signals": [
            {"candidate": "research", "status": "built-never-fired", "research_only": True},
        ],
    }
    line = format_curriculum_pool_line(summary)
    assert "curriculum-pool (3 tracked; 2 owed a fire" in line
    assert "1 built-never-fired" in line
    assert "bnf[built·L]" in line  # held lever, no ~ marker
    assert "nb~[needs·N/A]" in line  # not-a-lever, ~ marker
    assert "research-only SENSE (non-fireable): research[built-never-fired]" in line


def test_digest_section_reads_real_seeded_store():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    from costate_digest import section_curriculum_pool

    # the store was seeded at landing; section must render a line + machine-readable data (fail-open).
    line, data = section_curriculum_pool()
    if data is not None:  # store present
        assert line.startswith("curriculum-pool (")
        assert data["total"] >= 1
        assert "counts" in data
        fireable_names = {row["candidate"] for row in data["top_fireable"]}
        research_names = {row["candidate"] for row in data["research_signals"]}
        assert "p0_guarded_exact_costate_reuse_k2" not in fireable_names
        assert {
            "p0_guarded_exact_costate_reuse_k2",
            "p0_sparse_adjoint_dense_fullrank",
            "p0_terminal_exact_metric_396_costate_skip",
        } <= research_names
        research = {row["candidate"]: row for row in data["research_signals"]}
        reuse = research["p0_guarded_exact_costate_reuse_k2"]
        assert reuse["evidence_kind"] == ccp.EVIDENCE_RESEARCH_DIAGNOSTIC
        assert reuse["realized_speedup_factor"] == 1.0
        assert reuse["derived_cost_reduction_fraction"] == 0.0
        assert "NOT a global throughput win" in reuse["verdict_scope"]
        assert any(
            blocker.startswith("DERIVED_COUNTERFACTUAL_BEHIND_FAILED_FIDELITY_GATE") for blocker in reuse["blockers"]
        )
        assert "research-only SENSE (non-fireable)" in line
