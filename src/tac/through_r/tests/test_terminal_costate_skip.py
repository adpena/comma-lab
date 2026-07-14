import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

import pytest

from tac.through_r.terminal_costate_skip import (
    TRUSTED_EFFECTIVE_DIMENSION_CERTIFICATE_SHA256S,
    TRUSTED_TERMINAL_RECEIPT_SHA256S,
    EffectiveDimensionCertificate,
    TerminalAction,
    TerminalMethod,
    TerminalReceiptIdentity,
    decide_terminal_costate_skip,
)

REPO = Path(__file__).resolve().parents[4]
CANONICAL_RECEIPT = REPO / ".omx/research/p0_terminal_costate_skip_handoff_20260713.json"
CANONICAL_SHA256 = "17574857da5ff862e520140977e988197962f009d6870d23fe3071c398112a9c"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


@pytest.fixture
def durable_dir():
    path = Path.cwd() / ".pytest_artifacts" / f"terminal-costate-skip-{uuid.uuid4().hex}"
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


def canonical_receipt() -> TerminalReceiptIdentity:
    return TerminalReceiptIdentity.from_path(
        CANONICAL_RECEIPT, expected_sha256=CANONICAL_SHA256
    )


def constructed_receipt(**overrides) -> TerminalReceiptIdentity:
    values = {
        "path": "experiments/results/terminal/receipt.json",
        "sha256": CANONICAL_SHA256,
        "status": "completed",
        "n_pairs": 600,
        "objective_sha256": SHA_B,
        "scorer_sha256": SHA_C,
    }
    values.update(overrides)
    return TerminalReceiptIdentity(**values)


def test_code_reviewed_allowlists_are_sealed_to_one_receipt_and_no_certificate():
    assert TRUSTED_TERMINAL_RECEIPT_SHA256S == frozenset({CANONICAL_SHA256})
    assert TRUSTED_EFFECTIVE_DIMENSION_CERTIFICATE_SHA256S == frozenset()
    assert sha256(CANONICAL_RECEIPT) == CANONICAL_SHA256


def test_canonical_handoff_admits_396_without_a_gradient():
    expected = canonical_receipt()
    decision = decide_terminal_costate_skip(
        method=TerminalMethod.EXACT_METRIC_MC_396,
        receipt=expected,
        expected_receipt=expected,
        expected_receipt_sha256=CANONICAL_SHA256,
        expected_dimension_certificate_sha256=None,
    )
    assert decision.action is TerminalAction.SKIP_COSTATE_EXACT_METRIC_MC
    assert decision.costate_required is False


def test_arbitrary_self_hashed_receipt_is_not_a_trust_root(durable_dir: Path):
    path = durable_dir / "receipt.json"
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "n_pairs": 600,
                "objective_sha256": SHA_B,
                "scorer_sha256": SHA_C,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="code-reviewed trust root"):
        TerminalReceiptIdentity.from_path(path, expected_sha256=sha256(path))


def test_missing_or_wrong_receipt_root_refuses_even_canonical_bytes():
    with pytest.raises(ValueError, match="trust root"):
        TerminalReceiptIdentity.from_path(CANONICAL_RECEIPT, expected_sha256="0" * 64)
    expected = canonical_receipt()
    decision = decide_terminal_costate_skip(
        method=TerminalMethod.EXACT_METRIC_MC_396,
        receipt=expected,
        expected_receipt=expected,
        expected_receipt_sha256="0" * 64,
        expected_dimension_certificate_sha256=None,
    )
    assert decision.action is TerminalAction.EXACT_METRIC_MC_ORDINARY_ROUTE


def test_caller_constructed_metadata_is_never_admitted():
    expected = constructed_receipt()
    decision = decide_terminal_costate_skip(
        method=TerminalMethod.EXACT_METRIC_MC_396,
        receipt=expected,
        expected_receipt=expected,
        expected_receipt_sha256=CANONICAL_SHA256,
        expected_dimension_certificate_sha256=None,
    )
    assert decision.action is TerminalAction.EXACT_METRIC_MC_ORDINARY_ROUTE


@pytest.mark.parametrize("method", [TerminalMethod.SPSA, TerminalMethod.ES])
def test_spsa_es_remain_refused_even_for_a_self_hashed_dim2_certificate(
    method: TerminalMethod, durable_dir: Path
):
    expected = canonical_receipt()
    certificate_path = durable_dir / "dim2.json"
    certificate_path.write_text(
        json.dumps({"effective_dimension": 2, "deterministic": True}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="code-reviewed trust root"):
        EffectiveDimensionCertificate.from_path(
            certificate_path, expected_sha256=sha256(certificate_path)
        )
    unverified = EffectiveDimensionCertificate(
        effective_dimension=2,
        deterministic=True,
        artifact_path=str(certificate_path),
        artifact_sha256=sha256(certificate_path),
    )
    decision = decide_terminal_costate_skip(
        method=method,
        receipt=expected,
        expected_receipt=expected,
        expected_receipt_sha256=CANONICAL_SHA256,
        dimension_certificate=unverified,
        expected_dimension_certificate_sha256=unverified.artifact_sha256,
    )
    assert decision.action is TerminalAction.FULL_TEACHER_OR_396_ORDINARY_ROUTE
    assert decision.costate_required is True


def test_symlink_to_transient_is_refused_before_canonical_root_can_be_used(
    durable_dir: Path, tmp_path: Path
):
    target = tmp_path / "receipt.json"
    target.write_bytes(CANONICAL_RECEIPT.read_bytes())
    link = durable_dir / "transient-link.json"
    os.symlink(target, link)
    with pytest.raises(ValueError, match="durable"):
        TerminalReceiptIdentity.from_path(link, expected_sha256=CANONICAL_SHA256)
