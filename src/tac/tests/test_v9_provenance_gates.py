from __future__ import annotations

import ast
import inspect
from dataclasses import replace

import pytest

from tac.v9_provenance_gates import (
    ConfigBijectionSnapshot,
    EvidenceClaim,
    FakeClaim,
    FlagProvenanceBinding,
    V9ProvenanceGateError,
    audit_config_flag_provenance_bijection,
    audit_evidence_authority_claims,
    audit_v9_fake_claims,
    check_config_flag_provenance_bijection_complete,
    check_evidence_authority_claims_are_custodied,
    check_v9_fake_claim_guards,
    collect_live_v9_bijection_snapshots,
    collect_live_v9_evidence_claims,
    collect_live_v9_fake_claims,
)

GOOD_SHA = "a" * 64


def _clean_binding(**changes) -> FlagProvenanceBinding:
    binding = FlagProvenanceBinding(
        flag="--alpha",
        raw_value=3,
        raw_type="int",
        raw_tokens=("--alpha", "3"),
        runtime_dest="alpha",
        runtime_value=3,
        runtime_type="int",
        lever_owners=("alpha_lever",),
        lawref_equation_ids=("alpha_law_v1",),
        compiler_record_equation_ids=("alpha_law_v1",),
        provenance_rung="derived_at_config",
        consumer_locations=("experiments/train.py:main:10",),
        runtime_receipt_schemas=("alpha_runtime.v1",),
    )
    return replace(binding, **changes)


def _clean_snapshot(binding: FlagProvenanceBinding | None = None, **changes) -> ConfigBijectionSnapshot:
    binding = binding or _clean_binding()
    snapshot = ConfigBijectionSnapshot(
        program="clean_v9",
        bindings=(binding,),
        semantic_order=("--alpha",),
        emitted_order=("--alpha",),
        lawref_flags=("--alpha",),
        compiler_record_flags=("--alpha",),
        provenance_flags=("--alpha",),
    )
    return replace(snapshot, **changes)


def _clean_fake_claim(**changes) -> FakeClaim:
    claim = FakeClaim(
        claim_id="clean",
        vehicle="v9_cgauge",
        active_basis_label="polar_directional_fourier",
        basis_implementation="global_polar_directional_fourier_plane_waves",
        basis_is_spatially_localized=False,
        pose_selected=False,
        d_pose_source="live_posenet",
        numeric_d_pose=0.001,
        self_orient_claim=True,
        compiled_self_orient=True,
        receipt_self_orient=True,
        claimed_percent_reduction=2.0,
        receipt_vehicle="v9_cgauge",
        receipt_scope="full-n600-realized-through-R",
    )
    return replace(claim, **changes)


def _clean_evidence(**changes) -> EvidenceClaim:
    claim = EvidenceClaim(
        claim_id="clean",
        language="[contest-CPU] exact evaluation",
        evidence_axis="contest-CPU",
        archive_sha256=GOOD_SHA,
        archive_bytes=100,
        evaluator="upstream/evaluate.py",
        pairs=600,
        promotion_claim=True,
        requested_exact_authority=True,
        score_claim=True,
    )
    return replace(claim, **changes)


# ---------------------------------------------------------------------------
# #332 config-provenance bijection: 24 independently collected test cases.
# ---------------------------------------------------------------------------


def test_bijection_clean_synthesized_v9_snapshot_passes() -> None:
    assert audit_config_flag_provenance_bijection([_clean_snapshot()]) == []


def test_bijection_hash_is_deterministic_and_sensitive() -> None:
    first = _clean_snapshot()
    second = _clean_snapshot()
    changed = _clean_snapshot(_clean_binding(raw_value=4, runtime_value=4, raw_tokens=("--alpha", "4")))
    assert first.bijection_hash == second.bijection_hash
    assert first.bijection_hash != changed.bijection_hash


def test_bijection_preserves_but_accepts_argparse_numeric_normalization() -> None:
    binding = _clean_binding(
        raw_value="1e-3",
        raw_type="str",
        raw_tokens=("--alpha", "1e-3"),
        runtime_value=0.001,
        runtime_type="float",
    )
    assert audit_config_flag_provenance_bijection([_clean_snapshot(binding)]) == []


@pytest.mark.parametrize(
    ("changes", "needle"),
    [
        ({"raw_tokens": ()}, "raw DSL token"),
        ({"raw_type": ""}, "raw DSL type"),
        ({"runtime_dest": None, "runtime_type": None}, "argparse-normalized"),
        ({"runtime_value": 4}, "raw/runtime mismatch"),
        ({"lever_owners": ()}, "exactly one Lever owner"),
        ({"lever_owners": ("one", "two")}, "exactly one Lever owner"),
        ({"lawref_equation_ids": ()}, "Lever.constant_refs"),
        ({"lawref_equation_ids": ("one", "two")}, "Lever.constant_refs"),
        ({"compiler_record_equation_ids": ()}, "canonical compiler record"),
        ({"compiler_record_equation_ids": ("other",)}, "does not match owning LawRef"),
        ({"provenance_rung": None}, "value-provenance rung"),
        ({"provenance_rung": "invented"}, "value-provenance rung"),
        ({"consumer_locations": ()}, "trainer-consumer"),
        ({"runtime_receipt_schemas": ()}, "runtime receipt schema"),
        ({"runtime_receipt_schemas": ("one", "two")}, "runtime receipt schema"),
    ],
)
def test_bijection_catches_synthesized_binding_violation(changes, needle) -> None:
    violations = audit_config_flag_provenance_bijection([_clean_snapshot(_clean_binding(**changes))])
    assert any(needle in violation for violation in violations)


@pytest.mark.parametrize(
    ("changes", "needle"),
    [
        ({"emitted_order": ("--beta",)}, "reordered"),
        ({"lawref_flags": ("--alpha", "--stale")}, "LawRef coverage mismatch"),
        ({"compiler_record_flags": ()}, "compiler-record coverage mismatch"),
        ({"provenance_flags": ()}, "provenance-table coverage mismatch"),
    ],
)
def test_bijection_catches_snapshot_set_or_order_violation(changes, needle) -> None:
    violations = audit_config_flag_provenance_bijection([_clean_snapshot(**changes)])
    assert any(needle in violation for violation in violations)


def test_bijection_strict_mode_refuses_synthesized_violation() -> None:
    bad = _clean_snapshot(_clean_binding(lever_owners=()))
    with pytest.raises(V9ProvenanceGateError, match="bijection incomplete"):
        check_config_flag_provenance_bijection_complete(snapshots=[bad], strict=True, verbose=False)


def test_live_v9_bijection_collector_walks_four_real_factories_deterministically() -> None:
    first = collect_live_v9_bijection_snapshots()
    second = collect_live_v9_bijection_snapshots()
    assert [snapshot.program for snapshot in first] == [
        "v9_cgauge_432",
        "v9_cgauge_truly_optimal_core",
        "v9_cgauge_ideal_mod19",
        "v9_cgauge_ideal_mod32",
    ]
    assert [len(snapshot.bindings) for snapshot in first] == [199, 219, 219, 219]
    assert [snapshot.bijection_hash for snapshot in first] == [snapshot.bijection_hash for snapshot in second]
    assert all(not snapshot.lawref_flags for snapshot in first)
    assert [len(snapshot.compiler_record_flags) for snapshot in first] == [6, 7, 7, 7]


# ---------------------------------------------------------------------------
# #351 V9 fake-claim guard: 24 independently collected test cases.
# ---------------------------------------------------------------------------


def test_fake_claim_clean_synthesized_v9_claim_passes() -> None:
    assert audit_v9_fake_claims([_clean_fake_claim()]) == []


@pytest.mark.parametrize("label", ["curvelet", "ACTIVE CURVELET", "shearlet", "compact_shearlet"])
def test_fake_claim_refuses_unlocalized_fourier_with_frame_label(label) -> None:
    violations = audit_v9_fake_claims([_clean_fake_claim(active_basis_label=label)])
    assert any("unlocalized implementation" in violation for violation in violations)


@pytest.mark.parametrize(
    ("changes", "needle"),
    [
        (
            {
                "pose_selected": True,
                "pose_receiver_authenticated": False,
                "pose_parseback_byte_effect_sha256": GOOD_SHA,
            },
            "authenticated receiver",
        ),
        (
            {"pose_selected": True, "pose_receiver_authenticated": True, "pose_parseback_byte_effect_sha256": None},
            "byte-effect SHA-256",
        ),
        (
            {
                "pose_selected": True,
                "pose_receiver_authenticated": True,
                "pose_parseback_byte_effect_sha256": "not-a-sha",
            },
            "byte-effect SHA-256",
        ),
    ],
)
def test_fake_claim_refuses_selected_pose_without_receiver_byte_effect(changes, needle) -> None:
    violations = audit_v9_fake_claims([_clean_fake_claim(**changes)])
    assert any(needle in violation for violation in violations)


@pytest.mark.parametrize("source", [None, "banked", "proxy", "historical", "live"])
def test_fake_claim_refuses_numeric_nonlive_posenet_substitution(source) -> None:
    violations = audit_v9_fake_claims([_clean_fake_claim(d_pose_source=source)])
    assert any("numeric d_pose" in violation for violation in violations)


def test_fake_claim_respects_specific_historical_non_authorizing_waiver() -> None:
    claim = _clean_fake_claim(
        d_pose_source="historical",
        pose_selected=False,
        waiver="HISTORICAL_NON_AUTHORIZING:read-only migration fixture retained for ABI audit",
    )
    assert audit_v9_fake_claims([claim]) == []


@pytest.mark.parametrize(
    "waiver",
    ["HISTORICAL_NON_AUTHORIZING:TODO", "HISTORICAL_NON_AUTHORIZING:", "anything"],
)
def test_fake_claim_rejects_placeholder_or_unscoped_waiver(waiver) -> None:
    violations = audit_v9_fake_claims([_clean_fake_claim(waiver=waiver)])
    assert any("malformed or placeholder" in violation for violation in violations)


@pytest.mark.parametrize(
    ("changes", "needle"),
    [
        ({"compiled_self_orient": False}, "compiled vehicle flag"),
        ({"receipt_self_orient": False}, "scoped receipt"),
        ({"receipt_vehicle": "different_vehicle"}, "percentage claim vehicle"),
        ({"receipt_scope": "n10-proxy"}, "full-n600 realized-through-R"),
    ],
)
def test_fake_claim_refuses_self_orient_and_percentage_scope_disagreement(changes, needle) -> None:
    violations = audit_v9_fake_claims([_clean_fake_claim(**changes)])
    assert any(needle in violation for violation in violations)


def test_fake_claim_strict_mode_refuses_real_synthesized_violation() -> None:
    with pytest.raises(V9ProvenanceGateError, match="fake claim guard failed"):
        check_v9_fake_claim_guards(
            claims=[_clean_fake_claim(active_basis_label="curvelet")],
            strict=True,
            verbose=False,
        )


def test_live_v9_fake_claim_collector_derives_alias_from_source() -> None:
    claims = collect_live_v9_fake_claims()
    assert len(claims) == 4
    assert all(claim.active_basis_label == "curvelet" for claim in claims)
    assert all(claim.basis_implementation == "global_polar_directional_fourier_plane_waves" for claim in claims)
    assert len(audit_v9_fake_claims(claims)) == 4


# ---------------------------------------------------------------------------
# #351 evidence-authority custody: 25 independently collected test cases.
# ---------------------------------------------------------------------------


def test_evidence_authority_clean_contest_claim_passes() -> None:
    assert audit_evidence_authority_claims([_clean_evidence()]) == []


def test_evidence_authority_advisory_language_without_exact_claim_passes() -> None:
    claim = _clean_evidence(
        language="full-n600 macOS-CPU advisory; not a score",
        evidence_axis="macOS-CPU advisory",
        requested_exact_authority=False,
        score_claim=False,
        promotion_claim=False,
        evaluator=None,
    )
    assert audit_evidence_authority_claims([claim]) == []


@pytest.mark.parametrize(
    "language",
    [
        "exact authority",
        "authoritative result",
        "promotion-grade result",
        "promotion eligible",
        "pointer moved",
        "[contest-CUDA] score",
    ],
)
def test_evidence_authority_exact_language_is_semantic_not_marker_based(language) -> None:
    claim = _clean_evidence(
        language=language,
        evidence_axis="macOS-CPU advisory",
        promotion_claim=False,
    )
    assert audit_evidence_authority_claims([claim])


@pytest.mark.parametrize("axis", ["mps", "mlx", "macOS-CPU advisory", "proxy", "synthetic"])
def test_evidence_authority_refuses_advisory_axes(axis) -> None:
    violations = audit_evidence_authority_claims([_clean_evidence(evidence_axis=axis)])
    assert any("axis=" in violation for violation in violations)


@pytest.mark.parametrize(
    ("changes", "needle"),
    [
        ({"archive_sha256": None}, "archive SHA-256"),
        ({"archive_sha256": "bad"}, "archive SHA-256"),
        ({"archive_bytes": 0}, "positive archive bytes"),
        ({"evaluator": "custom_eval.py"}, "upstream/evaluate.py"),
        ({"pairs": 599}, "full n600"),
        ({"promotion_claim": False}, "promotion_claim=true"),
    ],
)
def test_evidence_authority_refuses_each_missing_custody_edge(changes, needle) -> None:
    violations = audit_evidence_authority_claims([_clean_evidence(**changes)])
    assert any(needle in violation for violation in violations)


def test_evidence_authority_waiver_cannot_grant_exact_authority() -> None:
    claim = _clean_evidence(
        evidence_axis="macOS-CPU advisory",
        waiver="ADVISORY_ONLY:historical receipt retained for non-authorizing comparison",
    )
    violations = audit_evidence_authority_claims([claim])
    assert any("waiver cannot grant exact authority" in violation for violation in violations)


@pytest.mark.parametrize("waiver", ["ADVISORY_ONLY:TODO", "ADVISORY_ONLY:", "blanket"])
def test_evidence_authority_rejects_placeholder_waiver(waiver) -> None:
    claim = _clean_evidence(
        language="advisory only",
        requested_exact_authority=False,
        score_claim=False,
        waiver=waiver,
    )
    assert any("malformed or placeholder" in v for v in audit_evidence_authority_claims([claim]))


def test_evidence_authority_strict_mode_refuses_synthesized_overclaim() -> None:
    with pytest.raises(V9ProvenanceGateError, match="lacks custody"):
        check_evidence_authority_claims_are_custodied(
            claims=[_clean_evidence(evidence_axis="macOS-CPU advisory")],
            strict=True,
            verbose=False,
        )


def test_live_v9_f6_receipt_is_reaudited_as_advisory_not_trusted_as_pass_marker() -> None:
    claims = collect_live_v9_evidence_claims()
    assert len(claims) == 1
    assert claims[0].evidence_axis == "macOS-CPU advisory"
    assert claims[0].requested_exact_authority is False
    assert audit_evidence_authority_claims(claims) == []


def test_preflight_all_wires_all_three_v9_gates_warn_only() -> None:
    from tac import preflight

    tree = ast.parse(inspect.getsource(preflight.preflight_all))
    calls = {
        node.func.id: {keyword.arg: getattr(keyword.value, "value", None) for keyword in node.keywords}
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    expected = {
        "check_config_flag_provenance_bijection_complete",
        "check_v9_fake_claim_guards",
        "check_evidence_authority_claims_are_custodied",
    }
    assert expected <= calls.keys()
    assert all(calls[name].get("strict") is False for name in expected)


def test_preflight_public_wrapper_translates_strict_gate_error(monkeypatch) -> None:
    from tac import preflight, v9_provenance_gates

    def refuse(**_kwargs):
        raise V9ProvenanceGateError("synthetic strict refusal")

    monkeypatch.setattr(v9_provenance_gates, "check_v9_fake_claim_guards", refuse)
    with pytest.raises(preflight.PreflightError, match="synthetic strict refusal"):
        preflight.check_v9_fake_claim_guards(strict=True, verbose=False)
