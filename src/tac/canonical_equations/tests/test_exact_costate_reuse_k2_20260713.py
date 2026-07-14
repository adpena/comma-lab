import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pytest

import tac.through_r.terminal_costate_skip as terminal_runtime
from tac.canonical_equations.exact_costate_reuse_k2_20260713 import (
    CORRECTED_RECEIPT,
    CORRECTED_RECEIPT_SHA256,
    CORRECTED_WRAPPER,
    CORRECTED_WRAPPER_SHA256,
    EQUATION_ID,
    FIDELITY_BLOCKED_STATUS,
    TIMING_ELIGIBILITY,
    amortized_cost_fraction,
    build_exact_costate_reuse_k2_guarded_v1,
    corrected_diagnostic_threshold,
    exact_backward_call_amortization,
    exact_backward_call_reduction,
    exact_costate_reuse_k2_laws,
    full_facet_guard,
    terminal_costate_skip_admitted,
)
from tac.through_r.terminal_costate_skip import (
    EffectiveDimensionCertificate,
    TerminalAction,
    TerminalMethod,
    TerminalReceiptIdentity,
    decide_terminal_costate_skip,
)

REPO = Path(__file__).resolve().parents[4]
TERMINAL_RECEIPT_PATH = REPO / ".omx/research/p0_terminal_costate_skip_handoff_20260713.json"
REVIEWED_TERMINAL_RECEIPT_SHA256 = "17574857da5ff862e520140977e988197962f009d6870d23fe3071c398112a9c"


@pytest.fixture
def durable_dir():
    path = Path.cwd() / ".pytest_artifacts" / f"equation-terminal-skip-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        parent = path.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_k2_cost_law_and_non_k2_refusal():
    assert amortized_cost_fraction(alpha=0.2) == pytest.approx(0.6)
    assert amortized_cost_fraction(alpha=0.2, charged_nonaccept_rate=0.25) == pytest.approx(0.725)
    assert amortized_cost_fraction(alpha=0.2, fallback_rate=0.25) == pytest.approx(0.725)
    with pytest.raises(ValueError, match="alias"):
        amortized_cost_fraction(alpha=0.2, charged_nonaccept_rate=0.25, fallback_rate=0.5)
    assert exact_backward_call_amortization(reuse_accept_fraction=1.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        amortized_cost_fraction(alpha=0.2, cadence=3)


def test_full_facet_guard():
    assert full_facet_guard(
        anchor_ce=1.0,
        candidate_ce=0.9,
        anchor_d_seg=0.2,
        candidate_d_seg=0.2,
        anchor_d_pose=0.3,
        candidate_d_pose=0.3,
    )
    assert not full_facet_guard(
        anchor_ce=1.0,
        candidate_ce=0.9,
        anchor_d_seg=0.2,
        candidate_d_seg=0.21,
        anchor_d_pose=0.3,
        candidate_d_pose=0.3,
    )


def test_terminal_exact_396_admits_only_from_committed_reviewed_receipt():
    assert sha256(TERMINAL_RECEIPT_PATH) == REVIEWED_TERMINAL_RECEIPT_SHA256
    equation_admitted = terminal_costate_skip_admitted(
        exact_metric_accept_reject=True,
        terminal_receipt_path=TERMINAL_RECEIPT_PATH,
        expected_receipt_sha256=REVIEWED_TERMINAL_RECEIPT_SHA256,
    )
    receipt = TerminalReceiptIdentity.from_path(
        TERMINAL_RECEIPT_PATH,
        expected_sha256=REVIEWED_TERMINAL_RECEIPT_SHA256,
    )
    expected_receipt = TerminalReceiptIdentity.from_path(
        TERMINAL_RECEIPT_PATH,
        expected_sha256=REVIEWED_TERMINAL_RECEIPT_SHA256,
    )
    runtime_decision = decide_terminal_costate_skip(
        method=TerminalMethod.EXACT_METRIC_MC_396,
        receipt=receipt,
        expected_receipt=expected_receipt,
        expected_receipt_sha256=REVIEWED_TERMINAL_RECEIPT_SHA256,
        expected_dimension_certificate_sha256=None,
    )
    assert equation_admitted is True
    assert runtime_decision.action is TerminalAction.SKIP_COSTATE_EXACT_METRIC_MC
    assert runtime_decision.costate_required is False


def test_prior_scalar_self_certification_is_refused_without_receipt_bytes():
    assert not terminal_costate_skip_admitted(
        exact_metric_accept_reject=True,
        expected_receipt_sha256=REVIEWED_TERMINAL_RECEIPT_SHA256,
        effective_dimension=None,
        deterministic_dimension_certificate=False,
        n_pairs=600,
        receipt_custody_valid=True,
        terminal_receipt_sha256=REVIEWED_TERMINAL_RECEIPT_SHA256,
    )
    assert not terminal_costate_skip_admitted(
        exact_metric_accept_reject=False,
        terminal_receipt_path=TERMINAL_RECEIPT_PATH,
        expected_receipt_sha256=REVIEWED_TERMINAL_RECEIPT_SHA256,
        effective_dimension=2,
        deterministic_dimension_certificate=True,
        n_pairs=600,
        receipt_custody_valid=True,
        terminal_receipt_sha256=REVIEWED_TERMINAL_RECEIPT_SHA256,
        effective_dimension_certificate_sha256="1" * 64,
    )
    laws = exact_costate_reuse_k2_laws(
        alpha=0.25,
        anchor_ce=1.0,
        candidate_ce=0.8,
        anchor_d_seg=0.2,
        candidate_d_seg=0.2,
        anchor_d_pose=0.3,
        candidate_d_pose=0.29,
        exact_metric_accept_reject=True,
        terminal_n_pairs=600,
        terminal_receipt_custody_valid=True,
        terminal_receipt_sha256=REVIEWED_TERMINAL_RECEIPT_SHA256,
    )
    assert laws["terminal_costate_skip_admitted"] is False


def test_terminal_receipt_missing_tampered_or_unknown_root_fails_closed(durable_dir: Path):
    missing = durable_dir / "missing.json"
    assert not terminal_costate_skip_admitted(
        exact_metric_accept_reject=True,
        terminal_receipt_path=missing,
        expected_receipt_sha256=REVIEWED_TERMINAL_RECEIPT_SHA256,
    )
    assert not terminal_costate_skip_admitted(
        exact_metric_accept_reject=True,
        terminal_receipt_path=TERMINAL_RECEIPT_PATH,
        expected_receipt_sha256="0" * 64,
    )
    tampered = durable_dir / "tampered.json"
    tampered.write_bytes(TERMINAL_RECEIPT_PATH.read_bytes() + b"\n")
    assert not terminal_costate_skip_admitted(
        exact_metric_accept_reject=True,
        terminal_receipt_path=tampered,
        expected_receipt_sha256=REVIEWED_TERMINAL_RECEIPT_SHA256,
    )


def test_terminal_spsa_stays_refused_while_certificate_registry_is_empty(
    durable_dir: Path,
):
    assert not terminal_runtime.TRUSTED_EFFECTIVE_DIMENSION_CERTIFICATE_SHA256S
    certificate_path = durable_dir / "dim2.json"
    certificate_path.write_text(
        json.dumps({"effective_dimension": 2, "deterministic": True}),
        encoding="utf-8",
    )
    assert not terminal_costate_skip_admitted(
        exact_metric_accept_reject=False,
        terminal_receipt_path=TERMINAL_RECEIPT_PATH,
        expected_receipt_sha256=REVIEWED_TERMINAL_RECEIPT_SHA256,
        dimension_certificate_path=certificate_path,
        expected_dimension_certificate_sha256=sha256(certificate_path),
    )


def test_terminal_dimension_certificate_missing_or_tampered_fails_closed(
    monkeypatch,
    durable_dir: Path,
):
    certificate_path = durable_dir / "dim2.json"
    certificate_path.write_text(
        json.dumps({"effective_dimension": 2, "deterministic": True}),
        encoding="utf-8",
    )
    certificate_sha256 = sha256(certificate_path)
    monkeypatch.setattr(
        terminal_runtime,
        "TRUSTED_EFFECTIVE_DIMENSION_CERTIFICATE_SHA256S",
        frozenset({certificate_sha256}),
    )
    assert not terminal_costate_skip_admitted(
        exact_metric_accept_reject=False,
        terminal_receipt_path=TERMINAL_RECEIPT_PATH,
        expected_receipt_sha256=REVIEWED_TERMINAL_RECEIPT_SHA256,
        dimension_certificate_path=durable_dir / "missing.json",
        expected_dimension_certificate_sha256=certificate_sha256,
    )
    certificate_path.write_bytes(certificate_path.read_bytes() + b"\n")
    assert not terminal_costate_skip_admitted(
        exact_metric_accept_reject=False,
        terminal_receipt_path=TERMINAL_RECEIPT_PATH,
        expected_receipt_sha256=REVIEWED_TERMINAL_RECEIPT_SHA256,
        dimension_certificate_path=certificate_path,
        expected_dimension_certificate_sha256=certificate_sha256,
    )


@pytest.mark.parametrize(
    ("effective_dimension", "deterministic", "admitted"),
    [(2, True, True), (3, True, False), (2, False, False)],
)
@pytest.mark.parametrize("method", [TerminalMethod.SPSA, TerminalMethod.ES])
def test_terminal_spsa_equation_matches_runtime_with_content_verified_certificate(
    monkeypatch,
    durable_dir: Path,
    method: TerminalMethod,
    effective_dimension: int,
    deterministic: bool,
    admitted: bool,
):
    certificate_path = durable_dir / "dimension-certificate.json"
    certificate_path.write_text(
        json.dumps(
            {
                "effective_dimension": effective_dimension,
                "deterministic": deterministic,
            }
        ),
        encoding="utf-8",
    )
    certificate_sha256 = sha256(certificate_path)
    monkeypatch.setattr(
        terminal_runtime,
        "TRUSTED_EFFECTIVE_DIMENSION_CERTIFICATE_SHA256S",
        frozenset({certificate_sha256}),
    )

    equation_admitted = terminal_costate_skip_admitted(
        exact_metric_accept_reject=False,
        terminal_receipt_path=TERMINAL_RECEIPT_PATH,
        expected_receipt_sha256=REVIEWED_TERMINAL_RECEIPT_SHA256,
        dimension_certificate_path=certificate_path,
        expected_dimension_certificate_sha256=certificate_sha256,
    )
    receipt = TerminalReceiptIdentity.from_path(
        TERMINAL_RECEIPT_PATH,
        expected_sha256=REVIEWED_TERMINAL_RECEIPT_SHA256,
    )
    expected_receipt = TerminalReceiptIdentity.from_path(
        TERMINAL_RECEIPT_PATH,
        expected_sha256=REVIEWED_TERMINAL_RECEIPT_SHA256,
    )
    certificate = EffectiveDimensionCertificate.from_path(
        certificate_path,
        expected_sha256=certificate_sha256,
    )
    runtime_decision = decide_terminal_costate_skip(
        method=method,
        receipt=receipt,
        expected_receipt=expected_receipt,
        expected_receipt_sha256=REVIEWED_TERMINAL_RECEIPT_SHA256,
        dimension_certificate=certificate,
        expected_dimension_certificate_sha256=certificate_sha256,
    )
    runtime_admitted = (
        runtime_decision.action is TerminalAction.SKIP_COSTATE_DIMENSION_CERTIFIED
        and runtime_decision.costate_required is False
    )
    assert equation_admitted is admitted
    assert runtime_admitted is admitted


def test_laws_inject_cost_guard_and_terminal_skip():
    laws = exact_costate_reuse_k2_laws(
        alpha=0.25,
        anchor_ce=1.0,
        candidate_ce=0.8,
        anchor_d_seg=0.2,
        candidate_d_seg=0.2,
        anchor_d_pose=0.3,
        candidate_d_pose=0.29,
        exact_metric_accept_reject=True,
        terminal_receipt_path=TERMINAL_RECEIPT_PATH,
        expected_receipt_sha256=REVIEWED_TERMINAL_RECEIPT_SHA256,
        charged_accept_fraction=1.0,
        actual_guard_fallback_fraction=0.0,
        terminal_or_blocked_fraction=0.0,
    )
    assert laws == {
        "cadence": 2,
        "n_pairs": 600,
        "charged_accept_fraction": 1.0,
        "actual_guard_fallback_fraction": 0.0,
        "terminal_or_blocked_fraction": 0.0,
        "charged_nonaccept_fraction": 0.0,
        "counterfactual_nonadmitted_amortized_cost_fraction": 0.625,
        "counterfactual_nonadmitted_teacher_slice_speedup": 1.6,
        "counterfactual_nonadmitted_exact_backward_call_amortization": 2.0,
        "counterfactual_nonadmitted_exact_backward_call_reduction": 0.5,
        "admitted_teacher_slice_speedup": 1.0,
        "admitted_exact_backward_call_reduction": 0.0,
        "reuse_accept_fraction": 1.0,
        "fallback_rate": 0.0,
        "amortized_cost_fraction": 0.625,
        "teacher_slice_speedup": 1.6,
        "exact_backward_call_amortization": 2.0,
        "exact_backward_call_reduction": 0.5,
        "diagnostic_accept_fraction_threshold_strict_gt": 0.75,
        "full_facet_guard_admitted": True,
        "terminal_costate_skip_admitted": True,
    }


def test_corrected_economics_and_strict_threshold_boundaries():
    alpha = 0.1784755863
    p = 0.76
    charged_nonaccept = 1.0 - p
    cost_fraction = amortized_cost_fraction(alpha=alpha, charged_nonaccept_rate=charged_nonaccept)
    assert cost_fraction == pytest.approx(1.4184755862999998 / 2.0)
    assert 1.0 / cost_fraction == pytest.approx(1.4099643443401577)
    assert exact_backward_call_amortization(reuse_accept_fraction=p) == pytest.approx(1.6129032258064517)
    assert exact_backward_call_reduction(reuse_accept_fraction=p) == pytest.approx(0.38)
    threshold = corrected_diagnostic_threshold(alpha=alpha)
    assert threshold == pytest.approx(0.5354267588999999)
    assert not (threshold > threshold)
    assert threshold < 1.0


def test_laws_distinguish_guard_fallback_from_terminal_charged_nonaccept():
    common = {
        "alpha": 0.2,
        "anchor_ce": 1.0,
        "candidate_ce": 0.9,
        "anchor_d_seg": 0.2,
        "candidate_d_seg": 0.2,
        "anchor_d_pose": 0.3,
        "candidate_d_pose": 0.3,
        "charged_accept_fraction": 0.25,
        "actual_guard_fallback_fraction": 0.5,
        "terminal_or_blocked_fraction": 0.25,
    }
    derived = exact_costate_reuse_k2_laws(**common)
    assert derived["actual_guard_fallback_fraction"] == pytest.approx(0.5)
    assert derived["terminal_or_blocked_fraction"] == pytest.approx(0.25)
    assert derived["charged_nonaccept_fraction"] == pytest.approx(0.75)
    assert derived["counterfactual_nonadmitted_amortized_cost_fraction"] == pytest.approx(0.975)
    assert derived["counterfactual_nonadmitted_teacher_slice_speedup"] == pytest.approx(1.0 / 0.975)
    assert derived["admitted_teacher_slice_speedup"] == 1.0
    assert derived["admitted_exact_backward_call_reduction"] == 0.0
    with pytest.raises(ValueError, match="must equal 1"):
        exact_costate_reuse_k2_laws(**(common | {"terminal_or_blocked_fraction": 0.0}))

    default_rates = exact_costate_reuse_k2_laws(
        alpha=0.2,
        anchor_ce=1.0,
        candidate_ce=0.9,
        anchor_d_seg=0.2,
        candidate_d_seg=0.2,
        anchor_d_pose=0.3,
        candidate_d_pose=0.3,
    )
    assert default_rates["charged_accept_fraction"] == 0.0
    assert default_rates["actual_guard_fallback_fraction"] is None
    assert default_rates["terminal_or_blocked_fraction"] is None
    assert default_rates["charged_nonaccept_fraction"] == 1.0
    assert default_rates["counterfactual_nonadmitted_teacher_slice_speedup"] == pytest.approx(2.0 / 2.2)


def test_v1_aliases_and_output_keys_remain_compatible_without_conflating_fallback():
    common = {
        "alpha": 0.2,
        "anchor_ce": 1.0,
        "candidate_ce": 0.9,
        "anchor_d_seg": 0.2,
        "candidate_d_seg": 0.2,
        "anchor_d_pose": 0.3,
        "candidate_d_pose": 0.3,
    }
    legacy = exact_costate_reuse_k2_laws(
        **common,
        reuse_accept_fraction=0.25,
        fallback_rate=0.75,
    )
    assert legacy["reuse_accept_fraction"] == 0.25
    assert legacy["fallback_rate"] == 0.75
    assert legacy["charged_nonaccept_fraction"] == 0.75
    assert legacy["actual_guard_fallback_fraction"] is None
    assert legacy["terminal_or_blocked_fraction"] is None
    assert legacy["amortized_cost_fraction"] == legacy["counterfactual_nonadmitted_amortized_cost_fraction"]
    assert legacy["teacher_slice_speedup"] == legacy["counterfactual_nonadmitted_teacher_slice_speedup"]
    assert (
        legacy["exact_backward_call_amortization"]
        == (legacy["counterfactual_nonadmitted_exact_backward_call_amortization"])
    )
    assert (
        legacy["exact_backward_call_reduction"] == (legacy["counterfactual_nonadmitted_exact_backward_call_reduction"])
    )
    with pytest.raises(ValueError, match="reuse_accept_fraction alias"):
        exact_costate_reuse_k2_laws(
            **common,
            charged_accept_fraction=0.5,
            reuse_accept_fraction=0.25,
        )
    with pytest.raises(ValueError, match="fallback_rate alias"):
        exact_costate_reuse_k2_laws(
            **common,
            charged_accept_fraction=0.25,
            fallback_rate=0.5,
        )


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: amortized_cost_fraction(alpha=True), "alpha"),
        (lambda: amortized_cost_fraction(alpha=0.2, charged_nonaccept_rate=False), "charged_nonaccept_rate"),
        (lambda: amortized_cost_fraction(alpha=0.2, cadence=True), "K=2"),
        (lambda: exact_backward_call_amortization(reuse_accept_fraction=True), "reuse_accept_fraction"),
        (lambda: exact_backward_call_reduction(reuse_accept_fraction=False), "reuse_accept_fraction"),
    ],
)
def test_economics_reject_bool_numerics(call, match):
    with pytest.raises(ValueError, match=match):
        call()


def test_composed_laws_and_full_facet_guard_reject_bool_numerics():
    common = {
        "alpha": 0.2,
        "anchor_ce": 1.0,
        "candidate_ce": 0.9,
        "anchor_d_seg": 0.2,
        "candidate_d_seg": 0.2,
        "anchor_d_pose": 0.3,
        "candidate_d_pose": 0.3,
        "charged_accept_fraction": 0.25,
        "actual_guard_fallback_fraction": 0.5,
        "terminal_or_blocked_fraction": 0.25,
    }
    for field in (
        "alpha",
        "charged_accept_fraction",
        "actual_guard_fallback_fraction",
        "terminal_or_blocked_fraction",
    ):
        with pytest.raises(ValueError, match=field):
            exact_costate_reuse_k2_laws(**(common | {field: True}))
    with pytest.raises(ValueError, match="finite numbers"):
        full_facet_guard(
            anchor_ce=True,
            candidate_ce=0.9,
            anchor_d_seg=0.2,
            candidate_d_seg=0.2,
            anchor_d_pose=0.3,
            candidate_d_pose=0.3,
        )


def test_equation_declares_measured_scoped_no_go_and_false_authority():
    equation = build_exact_costate_reuse_k2_guarded_v1()
    assert equation.equation_id == EQUATION_ID
    assert set(equation.canonical_consumers) == {
        "tac.witness_control.exact_costate_reuse",
        "tac.witness_dsl.exact_costate_reuse_policy",
        "tac.through_r.terminal_costate_skip",
    }
    assert equation.canonical_producers == (
        "tools.probe_p0_costate_reuse_k2",
        "tools.adjudicate_p0_costate_reuse_k2",
    )
    assert equation.domain_of_validity["provider_current"] is False
    assert equation.domain_of_validity["pointer_moved"] is False
    assert equation.domain_of_validity["verdict"] == "NO_GO_NOT_ADMITTED"
    assert equation.domain_of_validity["diagnostic_economics_authority"] == ("DERIVED_DIAGNOSTIC_NOT_IN_LOOP")
    assert equation.domain_of_validity["timing_status"] == FIDELITY_BLOCKED_STATUS
    assert equation.domain_of_validity["timing_eligibility"] == TIMING_ELIGIBILITY
    assert len(equation.empirical_anchors) == 1
    anchor = equation.empirical_anchors[0]
    tracked_receipt = REPO / CORRECTED_RECEIPT
    assert tracked_receipt.is_file()
    assert sha256(tracked_receipt) == CORRECTED_RECEIPT_SHA256
    assert anchor.source_artifact == CORRECTED_RECEIPT
    assert anchor.inputs["tracked_receipt_sha256"] == CORRECTED_RECEIPT_SHA256
    assert anchor.inputs["embedded_full_wrapper_path"] == CORRECTED_WRAPPER
    assert anchor.inputs["embedded_full_wrapper_sha256"] == CORRECTED_WRAPPER_SHA256
    assert anchor.inputs["actual_guard_accept"] == 456
    assert anchor.inputs["actual_guard_fallback"] == 67
    assert anchor.inputs["terminal_or_blocked"] == 77
    assert anchor.inputs["charged_nonaccept"] == 144
    assert "fallback" not in anchor.inputs
    assert anchor.predicted_output["admitted_teacher_slice_speedup_x"] == 1.0
    assert anchor.predicted_output["admitted_exact_backward_call_reduction_fraction"] == 0.0
    assert anchor.predicted_output["timing_status"] == FIDELITY_BLOCKED_STATUS
    assert anchor.predicted_output["timing_eligibility"] == TIMING_ELIGIBILITY
    assert "UNKNOWN_IN_LOOP_TIMER_OWED" not in repr(equation)
    assert anchor.empirical_output["corrected_gate_passed"] is False
    assert anchor.empirical_output["accepted_d_seg_regret_lte_zero"] == "308/456"
