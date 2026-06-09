# SPDX-License-Identifier: MIT
"""Behavioral tests for the constants-provenance manifest + Catalog #385 gate (D2).

NO FAKE Class-2 discipline: these tests verify BEHAVIOR (the gate fires / doesn't
fire on the right inputs), NOT constants. Replacing ``blocks_at`` / ``verify`` /
``blocking_findings`` with a stub would make the fire tests FAIL:

* ``test_blocks_at_l2_arbitrary_score_relevant_no_replacement`` would FAIL if
  ``verify()`` were ``pass``.
* ``test_guardrail_harmless_constant_never_blocks`` would FAIL if the gate
  ignored the score_relevant/stability_critical guardrail (the operator-explicit
  anti-bureaucracy rule).
* ``test_replacement_path_unblocks`` would FAIL if a replacement plan didn't
  actually exempt the constant.
* ``test_audit_helper_fails_closed_on_malformed_manifest`` would FAIL if the
  scanner silently swallowed an unparseable manifest.
"""

from __future__ import annotations

import json

import pytest

from tac.substrates._shared.constants_provenance_audit import (
    audit_constants_provenance_manifests,
)
from tac.substrates._shared.constants_provenance_manifest import (
    CONSTANT_PROVENANCE_VALUES,
    ConstantProvenance,
    ConstantsProvenanceManifest,
    ConstantsProvenanceVerifyError,
    MeasurementScope,
    emit_constants_provenance_manifest,
    manifest_path_for_constants,
)
from tac.substrates._shared.constants_provenance_manifests_canonical import (
    CANONICAL_CONSTANTS_MANIFESTS,
    HI_NERV_CONSTANTS_MANIFEST,
)

# ---------------------------------------------------------------------------
# ConstantProvenance validation.
# ---------------------------------------------------------------------------


def test_constant_rejects_unknown_provenance() -> None:
    with pytest.raises(ValueError, match="provenance"):
        ConstantProvenance("w", 30.0, "GUESSED", score_relevant=True)


def test_constant_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="constant_name"):
        ConstantProvenance("", 1.0, "ARBITRARY", score_relevant=True)


def test_constant_rejects_unknown_blocking_maturity() -> None:
    with pytest.raises(ValueError, match="blocking_maturity_level"):
        ConstantProvenance(
            "w", 30.0, "ARBITRARY", score_relevant=True, blocking_maturity_level="L9"
        )


def test_constant_rejects_placeholder_replacement_path() -> None:
    """Catalog #287: a fake replacement plan must not be accepted (so it cannot
    later un-block an ARBITRARY score-relevant constant)."""
    for bad in ("TBD", "<value>", "pending", "todo"):
        with pytest.raises(ValueError, match="placeholder"):
            ConstantProvenance(
                "w", 30.0, "ARBITRARY", score_relevant=True, replacement_path=bad
            )


def test_provenance_values_are_the_canonical_four() -> None:
    assert set(CONSTANT_PROVENANCE_VALUES) == {"DERIVED", "MEASURED", "LEARNED", "ARBITRARY"}


# ---------------------------------------------------------------------------
# The blocking rule (the gate's core behavior).
# ---------------------------------------------------------------------------


def test_blocks_at_l2_arbitrary_score_relevant_no_replacement() -> None:
    """The canonical fire: L2 + ARBITRARY + score_relevant + no replacement."""
    m = ConstantsProvenanceManifest(
        "v",
        "L2",
        (ConstantProvenance("mystery_w", 30.0, "ARBITRARY", score_relevant=True),),
    )
    findings = m.blocking_findings()
    assert len(findings) == 1
    assert "mystery_w" in findings[0]
    with pytest.raises(ConstantsProvenanceVerifyError, match="mystery_w"):
        m.verify()


def test_does_not_block_below_blocking_maturity() -> None:
    """The SAME constant at L1 (below its L2 blocking level) does NOT block."""
    c = ConstantProvenance("w", 30.0, "ARBITRARY", score_relevant=True, blocking_maturity_level="L2")
    m_l1 = ConstantsProvenanceManifest("v", "L1", (c,))
    m_l2 = ConstantsProvenanceManifest("v", "L2", (c,))
    assert m_l1.blocking_findings() == ()
    m_l1.verify()  # must not raise
    assert len(m_l2.blocking_findings()) == 1


def test_blocks_at_l3_when_blocking_level_is_l2() -> None:
    """Maturity ABOVE the blocking level also blocks (>= semantics)."""
    c = ConstantProvenance("w", 30.0, "ARBITRARY", score_relevant=True, blocking_maturity_level="L2")
    assert c.blocks_at("L3") is True
    assert c.blocks_at("L7") is True
    assert c.blocks_at("L1") is False


def test_replacement_path_unblocks() -> None:
    """A real replacement_path exempts an ARBITRARY score-relevant constant even at L2."""
    c = ConstantProvenance(
        "w",
        30.0,
        "ARBITRARY",
        score_relevant=True,
        replacement_path="tools/measure_scorer_spectral_sensitivity.py v2",
    )
    m = ConstantsProvenanceManifest("v", "L2", (c,))
    assert m.blocking_findings() == ()
    m.verify()  # must not raise
    assert c.has_real_replacement() is True


def test_derived_measured_learned_never_block() -> None:
    """Only ARBITRARY blocks; DERIVED/MEASURED/LEARNED are resolved provenance."""
    for prov in ("DERIVED", "MEASURED", "LEARNED"):
        c = ConstantProvenance("w", 30.0, prov, score_relevant=True)
        m = ConstantsProvenanceManifest("v", "L7", (c,))
        assert m.blocking_findings() == (), f"{prov} should not block"


# ---------------------------------------------------------------------------
# The guardrail (operator-explicit anti-bureaucracy rule).
# ---------------------------------------------------------------------------


def test_guardrail_harmless_constant_never_blocks() -> None:
    """A constant that is neither score_relevant NOR stability_critical is EXEMPT
    even when ARBITRARY at L7 (the operator guardrail against bureaucratizing
    harmless engineering constants)."""
    harmless = ConstantProvenance(
        "log_every", 100, "ARBITRARY", score_relevant=False, stability_critical=False
    )
    m = ConstantsProvenanceManifest("v", "L7", (harmless,))
    assert harmless.is_gated is False
    assert m.blocking_findings() == ()
    m.verify()


def test_stability_critical_blocks_even_if_not_score_relevant() -> None:
    """A stability-critical ARBITRARY constant (e.g. grad-clip) blocks at L2 even
    when score_relevant=False — divergence-risk is a gated concern."""
    c = ConstantProvenance(
        "grad_clip", 1.0, "ARBITRARY", score_relevant=False, stability_critical=True
    )
    assert c.is_gated is True
    m = ConstantsProvenanceManifest("v", "L2", (c,))
    assert len(m.blocking_findings()) == 1


# ---------------------------------------------------------------------------
# Fragility advisory (the "measured can be cargo-cult too" guard).
# ---------------------------------------------------------------------------


def test_fragile_measured_empty_scope_advises_not_blocks() -> None:
    c = ConstantProvenance("w", 9.5, "MEASURED", score_relevant=True)  # empty scope
    m = ConstantsProvenanceManifest("v", "L2", (c,))
    # MEASURED -> not blocking
    assert m.blocking_findings() == ()
    # but the empty scope is a soft advisory
    adv = m.fragility_advisories()
    assert len(adv) == 1
    assert "FRAGILE-MEASURED" in adv[0]
    assert "w" in adv[0]


def test_measured_with_scope_has_no_fragility_advisory() -> None:
    scope = MeasurementScope(
        pairs=6,
        amplitude_range=(0.5, 2.0, 8.0),
        scorer_surfaces=("d_seg", "d_pose"),
        authority_tier="exact_pair_scorer",
        confidence_interval="+-0.02 (n=6x2)",
        artifact_path="/Volumes/VertigoDataTier/pact/atlas.json",
    )
    c = ConstantProvenance("w", 9.5, "MEASURED", score_relevant=True, measurement_scope=scope)
    m = ConstantsProvenanceManifest("v", "L2", (c,))
    assert m.fragility_advisories() == ()


def test_measurement_scope_rejects_tmp_artifact() -> None:
    with pytest.raises(ValueError, match="not /tmp"):
        MeasurementScope(artifact_path="/tmp/atlas.json")


# ---------------------------------------------------------------------------
# Manifest-level validation + serialization.
# ---------------------------------------------------------------------------


def test_manifest_rejects_duplicate_constant_names() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ConstantsProvenanceManifest(
            "v",
            "L1",
            (
                ConstantProvenance("w", 1.0, "ARBITRARY", score_relevant=True),
                ConstantProvenance("w", 2.0, "ARBITRARY", score_relevant=True),
            ),
        )


def test_manifest_rejects_bad_maturity() -> None:
    with pytest.raises(ValueError, match="declared_maturity_level"):
        ConstantsProvenanceManifest("v", "L99", ())


def test_manifest_round_trip_via_dict() -> None:
    m = ConstantsProvenanceManifest(
        "v",
        "L2",
        (
            ConstantProvenance(
                "w",
                30.0,
                "MEASURED",
                score_relevant=True,
                measurement_scope=MeasurementScope(pairs=6, scorer_surfaces=("d_seg",)),
            ),
        ),
    )
    back = ConstantsProvenanceManifest.from_dict(json.loads(json.dumps(m.as_dict())))
    assert back.vehicle_id == "v"
    assert back.declared_maturity_level == "L2"
    assert back.constants[0].constant_name == "w"
    assert back.constants[0].measurement_scope.pairs == 6


def test_provenance_histogram_counts_all_four() -> None:
    m = ConstantsProvenanceManifest(
        "v",
        "L1",
        (
            ConstantProvenance("a", 1, "DERIVED", score_relevant=False),
            ConstantProvenance("b", 2, "ARBITRARY", score_relevant=True),
            ConstantProvenance("c", 3, "ARBITRARY", score_relevant=True),
        ),
    )
    hist = m.provenance_histogram()
    assert hist == {"DERIVED": 1, "MEASURED": 0, "LEARNED": 0, "ARBITRARY": 2}


def test_emit_writes_durable_json_not_tmp(tmp_path) -> None:
    m = ConstantsProvenanceManifest("vx", "L1", ())
    out = emit_constants_provenance_manifest(m, repo_root=tmp_path)
    assert out.exists()
    assert out.relative_to(tmp_path) == (
        type(out)(".omx") / "state" / "constants_provenance" / "vx.json"
    )
    assert manifest_path_for_constants("vx", repo_root=tmp_path) == out
    loaded = ConstantsProvenanceManifest.from_dict(json.loads(out.read_text()))
    assert loaded.vehicle_id == "vx"


def test_emit_verify_true_raises_for_l2_launderer(tmp_path) -> None:
    bad = ConstantsProvenanceManifest(
        "bad", "L2", (ConstantProvenance("w", 30.0, "ARBITRARY", score_relevant=True),)
    )
    with pytest.raises(ConstantsProvenanceVerifyError):
        emit_constants_provenance_manifest(bad, repo_root=tmp_path, verify=True)


# ---------------------------------------------------------------------------
# The audit helper (the directory scan the gate delegates to).
# ---------------------------------------------------------------------------


def test_audit_helper_clean_dir_has_no_findings(tmp_path) -> None:
    clean = ConstantsProvenanceManifest(
        "clean", "L1", (ConstantProvenance("w", 30.0, "ARBITRARY", score_relevant=True),)
    )
    emit_constants_provenance_manifest(clean, repo_root=tmp_path)
    # include_canonical=False isolates the directory-scan from the committed seed.
    findings = audit_constants_provenance_manifests(repo_root=tmp_path, include_canonical=False)
    assert findings == []  # L1 below the L2 blocking level


def test_audit_helper_fires_on_l2_launderer(tmp_path) -> None:
    bad = ConstantsProvenanceManifest(
        "bad", "L2", (ConstantProvenance("mystery", 30.0, "ARBITRARY", score_relevant=True),)
    )
    emit_constants_provenance_manifest(bad, repo_root=tmp_path)
    findings = audit_constants_provenance_manifests(repo_root=tmp_path, include_canonical=False)
    assert len(findings) == 1
    assert findings[0].vehicle_id == "bad"
    assert findings[0].declared_maturity_level == "L2"


def test_audit_helper_includes_canonical_module_on_fresh_checkout() -> None:
    """The gate must work on a fresh checkout (no emitted JSON): the committed
    canonical module seed (hi_nerv @ L1) is scanned and contributes 0 blockers."""
    # point at a nonexistent dir so ONLY the canonical module is scanned.
    findings = audit_constants_provenance_manifests(
        manifest_dir="/nonexistent/constants_provenance", include_canonical=True
    )
    # hi_nerv is L1 -> its L2 blockers are recorded but not firing.
    assert findings == []


def test_audit_helper_json_overrides_canonical_for_same_vehicle(tmp_path) -> None:
    """A durable JSON for vehicle X OVERRIDES the canonical-module X (latest-emit-
    wins) — so an operator can advance hi_nerv to L2 and the gate then fires on its
    unresolved ARBITRARY constants without editing the seed module."""
    # emit a hi_nerv override that declares L2 with an unresolved ARBITRARY constant
    override = ConstantsProvenanceManifest(
        "hi_nerv",
        "L2",
        (ConstantProvenance("unresolved_knob", 7, "ARBITRARY", score_relevant=True),),
    )
    emit_constants_provenance_manifest(override, repo_root=tmp_path)
    findings = audit_constants_provenance_manifests(repo_root=tmp_path, include_canonical=True)
    hi_nerv_findings = [f for f in findings if f.vehicle_id == "hi_nerv"]
    # the JSON override (L2 + unresolved) wins over the canonical L1 seed -> fires.
    assert len(hi_nerv_findings) == 1
    assert "unresolved_knob" in hi_nerv_findings[0].finding


def test_audit_helper_fails_closed_on_malformed_manifest(tmp_path) -> None:
    """A manifest the gate cannot parse must be SURFACED (not silently passed)."""
    d = tmp_path / ".omx" / "state" / "constants_provenance"
    d.mkdir(parents=True)
    (d / "broken.json").write_text("{not valid json", encoding="utf-8")
    (d / "wrong_schema.json").write_text(
        json.dumps({"vehicle_id": "ws", "declared_maturity_level": "L99"}), encoding="utf-8"
    )
    findings = audit_constants_provenance_manifests(repo_root=tmp_path, include_canonical=False)
    msgs = [f.finding for f in findings]
    assert any("UNPARSEABLE" in m for m in msgs)
    assert any("MALFORMED" in m for m in msgs)


def test_audit_helper_missing_dir_is_empty(tmp_path) -> None:
    # no manifests emitted yet + no canonical -> nothing to gate
    assert (
        audit_constants_provenance_manifests(repo_root=tmp_path, include_canonical=False)
        == []
    )


# ---------------------------------------------------------------------------
# The canonical hi_nerv seed (the operator-required seed).
# ---------------------------------------------------------------------------


def test_hi_nerv_seed_declares_l1_and_verifies() -> None:
    """The seeded hi_nerv manifest declares L1 (mechanism-present) so its L2
    blockers are RECORDED but not firing — it verifies at L1."""
    m = HI_NERV_CONSTANTS_MANIFEST
    assert m.vehicle_id == "hi_nerv"
    assert m.declared_maturity_level == "L1"
    assert m.blocking_findings() == ()
    m.verify()  # must not raise at L1


def test_hi_nerv_seed_has_sin_frequency_arbitrary_with_replacement() -> None:
    """The worked symptom: sin_frequency=30 is ARBITRARY + score_relevant, with a
    real replacement_path (the v2 atlas) so it would NOT block even at L2."""
    m = HI_NERV_CONSTANTS_MANIFEST
    sin = next(c for c in m.constants if c.constant_name == "sin_frequency")
    assert sin.value == 30.0
    assert sin.provenance == "ARBITRARY"
    assert sin.score_relevant is True
    assert sin.has_real_replacement()
    assert "measure_scorer_spectral_sensitivity" in sin.replacement_path
    # would not block even at L2 (it has a replacement path -> records debt)
    assert sin.blocks_at("L2") is False


def test_hi_nerv_seed_has_mistake_b_distill_weights() -> None:
    """The Mistake-B anchor: both distill weights are ARBITRARY + score_relevant."""
    m = HI_NERV_CONSTANTS_MANIFEST
    names = {c.constant_name for c in m.constants}
    assert "segnet_distillation_weight" in names
    assert "pose_distillation_weight" in names
    for nm in ("segnet_distillation_weight", "pose_distillation_weight"):
        c = next(x for x in m.constants if x.constant_name == nm)
        assert c.value == 0.0
        assert c.provenance == "ARBITRARY"
        assert c.score_relevant is True


def test_hi_nerv_seed_includes_guardrail_exemption_and_a_derived_constant() -> None:
    """The seed demonstrates BOTH the guardrail (harmless constant) and a DERIVED
    (non-arbitrary) constant, proving the manifest is not 'everything ARBITRARY'."""
    m = HI_NERV_CONSTANTS_MANIFEST
    # guardrail exemption: checkpoint_cadence is ARBITRARY but not gated
    cadence = next(c for c in m.constants if c.constant_name == "checkpoint_cadence_epochs")
    assert cadence.is_gated is False
    # a DERIVED constant exists (contrast example)
    hist = m.provenance_histogram()
    assert hist["DERIVED"] >= 1


def test_canonical_manifests_tuple_contains_hi_nerv() -> None:
    ids = {m.vehicle_id for m in CANONICAL_CONSTANTS_MANIFESTS}
    assert "hi_nerv" in ids


# ---------------------------------------------------------------------------
# The preflight gate (Catalog #385) — fire / no-fire at the orchestrator surface.
# ---------------------------------------------------------------------------


def test_preflight_gate_no_fire_on_clean_repo_state() -> None:
    """Against the real repo state, the live count must be 0 (hi_nerv @ L1)."""
    from tac.preflight import check_no_arbitrary_score_relevant_constant_at_l2

    violations = check_no_arbitrary_score_relevant_constant_at_l2(strict=False)
    assert violations == []


def test_preflight_gate_raises_in_strict_when_manifest_dir_has_launderer(tmp_path) -> None:
    """The gate's STRICT branch must raise PreflightError when a manifest declares
    L2 with an unresolved ARBITRARY score-relevant constant. Points the gate's
    ``repo_root`` at an isolated dir so the strict-raise is proven end-to-end
    without mutating the repo state."""
    from tac.preflight import (
        PreflightError,
        check_no_arbitrary_score_relevant_constant_at_l2,
    )

    bad = ConstantsProvenanceManifest(
        "bad", "L2", (ConstantProvenance("w", 30.0, "ARBITRARY", score_relevant=True),)
    )
    emit_constants_provenance_manifest(bad, repo_root=tmp_path)

    # non-strict: returns the violation list (warn-only behavior). The committed
    # canonical hi_nerv seed (L1) is also merged but contributes 0 blockers, so the
    # only violation is the "bad"@L2 launderer.
    violations = check_no_arbitrary_score_relevant_constant_at_l2(
        strict=False, repo_root=tmp_path
    )
    assert any("bad" in v for v in violations)
    assert not any("hi_nerv" in v for v in violations)  # L1 seed does not block

    # strict: raises
    with pytest.raises(PreflightError, match="check_no_arbitrary_score_relevant_constant_at_l2"):
        check_no_arbitrary_score_relevant_constant_at_l2(strict=True, repo_root=tmp_path)
