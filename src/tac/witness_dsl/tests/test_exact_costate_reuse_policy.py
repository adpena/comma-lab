import json
import shutil
import uuid
from pathlib import Path

import pytest

import tac.witness_dsl.exact_costate_reuse_policy as policy_module
from tac.witness_dsl.exact_costate_reuse_policy import (
    CORRECTED_ADMISSION_VERDICT,
    ExactCostateReusePolicy,
    TemporalFidelityReceiptCustody,
    exact_costate_reuse_k2_lever,
)

REPO = Path(__file__).resolve().parents[4]
REAL_WRAPPER = REPO / "experiments/results/p0_costate_reuse_k2_n600_v3_20260713" / "corrected_adjudication_receipt.json"


@pytest.fixture
def durable_dir():
    path = Path.cwd() / ".pytest_artifacts" / f"costate-reuse-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        parent = path.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()


def real_policy(*, enabled: bool = True) -> ExactCostateReusePolicy:
    return ExactCostateReusePolicy(
        enabled=enabled,
        receipt=TemporalFidelityReceiptCustody.from_path(REAL_WRAPPER),
    )


def trusted_copy(monkeypatch, durable_dir: Path) -> Path:
    copied_root = durable_dir / "sealed"
    shutil.copytree(REAL_WRAPPER.parent, copied_root)
    wrapper = copied_root / REAL_WRAPPER.name
    monkeypatch.setattr(policy_module, "TRUSTED_CORRECTED_WRAPPER_PATH", wrapper)
    return wrapper


def test_default_off_and_lever_is_argv_inert():
    policy = ExactCostateReusePolicy()
    compiled = policy.compile_activation_contract()
    lever = exact_costate_reuse_k2_lever(policy)
    assert compiled["measurement_verified"] is False
    assert compiled["measurement_admitted"] is False
    assert compiled["trainer_activation_admitted"] is False
    assert compiled["live_trainer_argv"] == []
    assert compiled["provider_current"] is False
    assert lever.overrides == {}
    assert "trainer=REFUSED" in lever.notes


@pytest.mark.parametrize("field,value", [("k_max", 3), ("n_pairs", 599)])
def test_kmax_not_two_or_n_not_600_is_refused(field, value):
    with pytest.raises(ValueError):
        ExactCostateReusePolicy(**{field: value})


def test_real_corrected_wrapper_is_verified_offline_but_preserves_no_go():
    compiled = real_policy().compile_activation_contract()
    assert compiled["measurement_verified"] is True
    assert compiled["measurement_admitted"] is False
    assert compiled["corrected_admission_verdict"] == CORRECTED_ADMISSION_VERDICT
    assert compiled["measurement_authority"] == "OFFLINE_N600_TRAINING_SIGNAL_ONLY_NOT_ADMITTED"
    assert compiled["trainer_activation_admitted"] is False
    assert compiled["live_trainer_argv"] == []
    assert "current-costate provider is unavailable" in compiled["trainer_activation_errors"]


def test_original_receipt_and_arbitrary_paths_are_not_trusted():
    with pytest.raises(ValueError, match="not code-reviewed"):
        TemporalFidelityReceiptCustody.from_path(REAL_WRAPPER.parent / "measurement_receipt.json")
    with pytest.raises(ValueError, match="not code-reviewed"):
        TemporalFidelityReceiptCustody.from_path(Path.cwd() / "arbitrary.json")


def test_caller_selected_hash_api_does_not_exist():
    with pytest.raises(TypeError):
        ExactCostateReusePolicy(expected_receipt_sha256="0" * 64)


def test_direct_receipt_constructor_cannot_mint_verified_authority():
    trusted = TemporalFidelityReceiptCustody.from_path(REAL_WRAPPER)
    with pytest.raises(TypeError, match="_authority_token"):
        TemporalFidelityReceiptCustody(
            **trusted.public_custody(),
            _authority_token=object(),
        )

    direct_nominal = TemporalFidelityReceiptCustody(**trusted.public_custody())
    nominal_errors = ExactCostateReusePolicy(receipt=direct_nominal).measurement_errors()
    assert "corrected wrapper custody was not established by from_path" in nominal_errors
    assert "corrected wrapper from_path snapshot mismatch" in nominal_errors

    fake_fields = trusted.public_custody()
    fake_fields["objective_sha256"] = "0" * 64
    direct_fake = TemporalFidelityReceiptCustody(**fake_fields)
    fake_errors = ExactCostateReusePolicy(receipt=direct_fake).measurement_errors()
    assert "corrected wrapper custody was not established by from_path" in fake_errors
    assert "corrected wrapper instance objective_sha256 does not match bytes" in fake_errors
    assert "receipt objective sha256 mismatch" in fake_errors


def test_duck_receipt_is_rejected_before_any_method_dispatch():
    calls: list[str] = []

    class DuckReceipt:
        def validation_errors(self, **_kwargs):
            calls.append("validation_errors")
            raise AssertionError("duck validation method must not run")

        def public_custody(self):
            calls.append("public_custody")
            raise AssertionError("duck serialization method must not run")

    with pytest.raises(TypeError, match="subclasses and duck types are refused"):
        ExactCostateReusePolicy(receipt=DuckReceipt())
    assert calls == []


def test_malicious_receipt_subclass_is_rejected_before_overrides_run():
    calls: list[str] = []

    class MaliciousReceipt(TemporalFidelityReceiptCustody):
        def validation_errors(self, **_kwargs):
            calls.append("validation_errors")
            raise AssertionError("subclass validation override must not run")

        def public_custody(self):
            calls.append("public_custody")
            raise AssertionError("subclass serialization override must not run")

    malicious = MaliciousReceipt.from_path(REAL_WRAPPER)
    with pytest.raises(TypeError, match="subclasses and duck types are refused"):
        ExactCostateReusePolicy(receipt=malicious)
    assert calls == []


def test_symlink_wrapper_is_refused(monkeypatch, durable_dir):
    link = durable_dir / "wrapper.json"
    link.symlink_to(REAL_WRAPPER)
    monkeypatch.setattr(policy_module, "TRUSTED_CORRECTED_WRAPPER_PATH", link)
    with pytest.raises(ValueError, match="symlink"):
        TemporalFidelityReceiptCustody.from_path(link)


def test_wrapper_tamper_after_loading_fails_closed(monkeypatch, durable_dir):
    wrapper = trusted_copy(monkeypatch, durable_dir)
    custody = TemporalFidelityReceiptCustody.from_path(wrapper)
    payload = json.loads(wrapper.read_text())
    payload["authority"]["pointer_moved"] = True
    wrapper.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    errors = ExactCostateReusePolicy(receipt=custody).measurement_errors()
    assert "corrected wrapper bytes sha256 mismatch" in errors
    assert "corrected wrapper content sha256 mismatch" in errors
    assert "corrected wrapper carries false authority" in errors


def test_mutation_after_last_first_pass_read_fails_final_snapshot(monkeypatch, durable_dir):
    wrapper = trusted_copy(monkeypatch, durable_dir)
    custody = TemporalFidelityReceiptCustody.from_path(wrapper)
    pair_path = wrapper.parent / "pairs/pair_0599.json"

    def mutate_after_first_pass() -> None:
        pair = json.loads(pair_path.read_text())
        pair["status"] = "MUTATED_BETWEEN_PASSES"
        pair_path.write_text(json.dumps(pair, indent=2, sort_keys=True) + "\n")

    monkeypatch.setattr(policy_module, "_before_final_snapshot_verify", mutate_after_first_pass)
    errors = ExactCostateReusePolicy(receipt=custody).measurement_errors()
    assert "pairs/pair_0599.json changed between custody passes" in errors


def test_activation_contract_reuses_one_fail_closed_custody_transaction(monkeypatch, durable_dir):
    wrapper = trusted_copy(monkeypatch, durable_dir)
    custody = TemporalFidelityReceiptCustody.from_path(wrapper)
    pair_path = wrapper.parent / "pairs/pair_0599.json"
    hook_calls = 0

    def mutate_during_only_transaction() -> None:
        nonlocal hook_calls
        hook_calls += 1
        if hook_calls == 1:
            pair = json.loads(pair_path.read_text())
            pair["status"] = "MUTATED_DURING_COMPILED_TRANSACTION"
            pair_path.write_text(json.dumps(pair, indent=2, sort_keys=True) + "\n")

    monkeypatch.setattr(
        policy_module,
        "_before_final_snapshot_verify",
        mutate_during_only_transaction,
    )
    contract = ExactCostateReusePolicy(enabled=True, receipt=custody).compile_activation_contract()
    tamper_error = "pairs/pair_0599.json changed between custody passes"
    assert hook_calls == 1
    assert contract["measurement_verified"] is False
    assert tamper_error in contract["measurement_errors"]
    assert contract["trainer_activation_admitted"] is False
    assert (
        contract["trainer_activation_errors"][: len(contract["measurement_errors"])] == contract["measurement_errors"]
    )
    assert tamper_error in contract["trainer_activation_errors"]


def test_nested_pair_tamper_after_loading_fails_closed(monkeypatch, durable_dir):
    wrapper = trusted_copy(monkeypatch, durable_dir)
    custody = TemporalFidelityReceiptCustody.from_path(wrapper)
    pair_path = wrapper.parent / "pairs/pair_0000.json"
    pair = json.loads(pair_path.read_text())
    pair["status"] = "TAMPERED"
    pair_path.write_text(json.dumps(pair, indent=2, sort_keys=True) + "\n")
    errors = ExactCostateReusePolicy(receipt=custody).measurement_errors()
    assert "pairs/pair_0000.json sha256 mismatch" in errors
    assert "pair 0 sha256 mismatch" in errors
    assert "pair 0 semantic custody mismatch" in errors


def test_nested_stage_and_completion_tamper_fail_closed(monkeypatch, durable_dir):
    wrapper = trusted_copy(monkeypatch, durable_dir)
    custody = TemporalFidelityReceiptCustody.from_path(wrapper)
    stage_path = wrapper.parent / "stage_v9_ep150_ema_best_complete.json"
    stage = json.loads(stage_path.read_text())
    stage["tree_sha256"] = "0" * 64
    stage_path.write_text(json.dumps(stage, indent=2, sort_keys=True) + "\n")
    complete_path = wrapper.parent / "complete.json"
    complete = json.loads(complete_path.read_text())
    complete["receipt_sha256"] = "0" * 64
    complete_path.write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
    errors = ExactCostateReusePolicy(receipt=custody).measurement_errors()
    assert any("stage_v9_ep150_ema_best_complete.json sha256 mismatch" in e for e in errors)
    assert "completion seal does not bind the reviewed receipt" in errors


def test_nested_source_symlink_is_refused(monkeypatch, durable_dir):
    wrapper = trusted_copy(monkeypatch, durable_dir)
    custody = TemporalFidelityReceiptCustody.from_path(wrapper)
    pair_path = wrapper.parent / "pairs/pair_0000.json"
    target = durable_dir / "pair.json"
    target.write_bytes(pair_path.read_bytes())
    pair_path.unlink()
    pair_path.symlink_to(target)
    errors = ExactCostateReusePolicy(receipt=custody).measurement_errors()
    assert any("symlink" in error for error in errors)


def test_provider_current_cannot_be_claimed_before_integration():
    with pytest.raises(ValueError, match="not integrated"):
        ExactCostateReusePolicy(provider_current=True)
