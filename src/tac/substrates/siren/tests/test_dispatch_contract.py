from __future__ import annotations

import pytest

from tac.substrates.siren.dispatch_contract import (
    HYBRID_SIREN_DOMAIN_PRIOR,
    NAKED_SIREN_REPLACEMENT,
    SIREN_DISPATCH_CONTRACTS,
    SIREN_RESIDUAL_ON_HNERV_A1,
    normalize_siren_dispatch_contract,
    require_train_substrate_siren_contract,
    siren_dispatch_contract_manifest,
)


def test_all_three_siren_dispatch_contracts_are_named() -> None:
    assert set(SIREN_DISPATCH_CONTRACTS) == {
        NAKED_SIREN_REPLACEMENT,
        SIREN_RESIDUAL_ON_HNERV_A1,
        HYBRID_SIREN_DOMAIN_PRIOR,
    }


def test_naked_contract_explicitly_replaces_hnerv_a1_substrate() -> None:
    contract = require_train_substrate_siren_contract(NAKED_SIREN_REPLACEMENT)

    assert contract.archive_role == "replaces_hnerv_a1_substrate"
    assert contract.train_substrate_siren_supported is True
    assert "No HNeRV/A1 base archive is consumed" in contract.hnerv_a1_relationship


@pytest.mark.parametrize(
    "contract_id",
    [SIREN_RESIDUAL_ON_HNERV_A1, HYBRID_SIREN_DOMAIN_PRIOR],
)
def test_trainer_refuses_non_naked_contracts_to_prevent_wrong_archive(
    contract_id: str,
) -> None:
    with pytest.raises(NotImplementedError, match="wrong archive"):
        require_train_substrate_siren_contract(contract_id)


def test_contract_manifest_is_json_serializable_and_contains_builder_surfaces() -> None:
    manifest = siren_dispatch_contract_manifest()

    assert manifest[NAKED_SIREN_REPLACEMENT]["builder_surface"] == (
        "experiments/train_substrate_siren.py"
    )
    assert manifest[SIREN_RESIDUAL_ON_HNERV_A1]["archive_role"] == (
        "residual_sidecar_on_hnerv_a1_base"
    )
    assert manifest[HYBRID_SIREN_DOMAIN_PRIOR]["required_inputs"] == [
        "domain_prior_manifest.json"
    ]


def test_normalize_accepts_legacy_replacement_alias() -> None:
    assert normalize_siren_dispatch_contract("siren-replacement") == (
        NAKED_SIREN_REPLACEMENT
    )


def test_train_substrate_siren_threads_contract_flag() -> None:
    from experiments import train_substrate_siren as trainer

    parser = trainer._build_parser()
    args = parser.parse_args(["--output-dir", "out", "--epochs", "1"])

    assert args.dispatch_contract == NAKED_SIREN_REPLACEMENT
    assert (
        trainer.TIER_1_OPERATOR_REQUIRED_FLAGS["--dispatch-contract"]["env"]
        == "SIREN_DISPATCH_CONTRACT"
    )


def test_train_substrate_siren_threads_activation_family_flag() -> None:
    from experiments import train_substrate_siren as trainer

    parser = trainer._build_parser()
    args = parser.parse_args(
        ["--output-dir", "out", "--epochs", "1", "--activation-family", "wire"]
    )

    assert args.activation_family == "wire"
    assert (
        trainer.TIER_1_OPERATOR_REQUIRED_FLAGS["--activation-family"]["env"]
        == "SIREN_ACTIVATION_FAMILY"
    )


def test_train_substrate_siren_vendors_activation_family_runtime_module() -> None:
    from experiments import train_substrate_siren as trainer

    text = trainer.Path(trainer.__file__).read_text(encoding="utf-8")

    assert '"activation_family.py"' in text
