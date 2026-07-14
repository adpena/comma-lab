import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pytest

from tac.witness_dsl.terminal_costate_skip_policy import (
    CANONICAL_RECEIPT_PATH,
    TerminalCostateSkipEvidence,
    TerminalCostateSkipPolicy,
    terminal_exact_metric_costate_skip_lever,
)

REPO = Path(__file__).resolve().parents[4]


@pytest.fixture
def durable_dir():
    path = REPO / ".pytest_artifacts" / f"terminal-costate-skip-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        parent = path.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_default_policy_and_dsl_lever_are_fail_closed_and_argv_inert():
    policy = TerminalCostateSkipPolicy()
    compiled = policy.compile_contract()
    lever = terminal_exact_metric_costate_skip_lever(policy)
    assert compiled["measurement_admitted"] is False
    assert compiled["terminal_route_activation_admitted"] is False
    assert compiled["live_trainer_argv"] == []
    assert compiled["bulk_training_teacher_cost_reduction"] == "UNQUANTIFIED_NOT_COMPOSABLE"
    assert compiled["candidate_bytes_revalidated_at_authorization"] is False
    assert lever.overrides == {}
    assert "SPSA/ES" in lever.notes


def test_canonical_n600_handoff_admits_measurement_but_stays_default_off():
    evidence = TerminalCostateSkipEvidence.canonical()
    policy = TerminalCostateSkipPolicy(evidence=evidence)
    compiled = policy.compile_contract()
    assert evidence.validation_errors() == ()
    assert compiled["measurement_admitted"] is True
    assert compiled["candidate_bytes_revalidated_at_authorization"] is False
    assert compiled["terminal_action"] == "skip_costate_exact_metric_mc"
    assert compiled["terminal_route_activation_admitted"] is False
    assert any(
        "default-off" in error
        for error in compiled["terminal_route_activation_errors"]
    )


def test_explicit_policy_enable_admits_only_the_existing_396_provider():
    policy = TerminalCostateSkipPolicy(
        enabled=True,
        evidence=TerminalCostateSkipEvidence.canonical(),
        provider_current=True,
    )
    compiled = policy.compile_contract()
    assert compiled["terminal_route_activation_admitted"] is True
    assert compiled["terminal_action"] == "skip_costate_exact_metric_mc"


def test_wrapper_tamper_after_evidence_construction_refuses(durable_dir: Path):
    source = REPO / CANONICAL_RECEIPT_PATH
    wrapper = durable_dir / "handoff.json"
    wrapper.write_bytes(source.read_bytes())
    evidence = TerminalCostateSkipEvidence(
        str(wrapper.relative_to(REPO)), _sha(wrapper)
    )
    assert evidence.validation_errors() == ()
    payload = json.loads(wrapper.read_text())
    payload["measured"]["delta_s"] = 1.0
    wrapper.write_text(json.dumps(payload, sort_keys=True) + "\n")
    assert "terminal handoff sha256 mismatch" in evidence.validation_errors()


def test_source_receipt_disappearance_after_valid_load_refuses(durable_dir: Path):
    canonical = json.loads((REPO / CANONICAL_RECEIPT_PATH).read_text())
    canonical_source = REPO / canonical["source_receipt_custody"]["path"]
    source_copy = durable_dir / "source_receipt.json"
    source_copy.write_bytes(canonical_source.read_bytes())
    canonical["source_receipt_custody"] = {
        "path": str(source_copy.relative_to(REPO)),
        "bytes": source_copy.stat().st_size,
        "sha256": _sha(source_copy),
    }
    wrapper = durable_dir / "handoff.json"
    wrapper.write_text(json.dumps(canonical, sort_keys=True, indent=2) + "\n")
    evidence = TerminalCostateSkipEvidence(str(wrapper.relative_to(REPO)), _sha(wrapper))
    assert evidence.validation_errors() == ()
    source_copy.unlink()
    assert "terminal source receipt bytes are unavailable" in evidence.validation_errors()


def test_untrusted_expected_wrapper_hash_is_refused():
    evidence = TerminalCostateSkipEvidence(CANONICAL_RECEIPT_PATH, "f" * 64)
    assert "terminal handoff sha256 mismatch" in evidence.validation_errors()
