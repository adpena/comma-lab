from __future__ import annotations

from pathlib import Path

import pytest

from tac.deploy.gcp.gcp_dispatch import plan_gcp_dispatch
from tac.deploy.provider_contracts import (
    PROVIDER_NAMES,
    provider_contracts,
    validate_provider_contracts,
)
from tac.preflight import PreflightError, check_provider_deploy_contracts


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_provider_contract_registry_covers_required_providers() -> None:
    contracts = provider_contracts()
    assert tuple(contracts) == PROVIDER_NAMES
    assert set(contracts) == {"modal", "kaggle", "aws", "azure", "gcp"}


def test_provider_contracts_preserve_custody_and_proxy_rules() -> None:
    for contract in provider_contracts().values():
        assert contract.plan_only_default is True
        assert contract.requires_lane_claim_before_dispatch is True
        assert contract.terminal_claim_required is True
        assert contract.custody_manifest_required is True
        assert contract.proxy_score_claim_allowed is False
        assert contract.provider_proxy_promotes_score is False
        assert contract.mps_auth_eval_allowed is False


def test_only_implemented_cuda_surfaces_advertise_exact_eval_support() -> None:
    contracts = provider_contracts()

    assert {
        name for name, contract in contracts.items() if contract.exact_cuda_eval_supported
    } == {"modal", "azure"}
    for name in ("aws", "gcp"):
        assert contracts[name].scaffold_only is True
        assert contracts[name].exact_cuda_eval_supported is False


def test_validate_provider_contracts_passes_on_live_repo() -> None:
    assert validate_provider_contracts(repo_root=REPO_ROOT) == []


def test_preflight_provider_contract_guard_is_strict() -> None:
    assert check_provider_deploy_contracts(strict=True, verbose=False) == []


def test_preflight_provider_contract_guard_raises_on_registry_violation(monkeypatch) -> None:
    import tac.deploy.provider_contracts as contracts_mod

    original = contracts_mod.PROVIDER_CONTRACTS["kaggle"]
    bad = {**contracts_mod.PROVIDER_CONTRACTS}
    bad["kaggle"] = type(original)(
        **{
            **original.__dict__,
            "proxy_score_claim_allowed": True,
        }
    )
    monkeypatch.setattr(contracts_mod, "PROVIDER_CONTRACTS", bad)
    with pytest.raises(PreflightError, match="provider proxy score claims"):
        check_provider_deploy_contracts(strict=True, verbose=False)


def test_scaffold_contract_cannot_advertise_exact_cuda_support(monkeypatch) -> None:
    import tac.deploy.provider_contracts as contracts_mod

    original = contracts_mod.PROVIDER_CONTRACTS["gcp"]
    bad = {**contracts_mod.PROVIDER_CONTRACTS}
    bad["gcp"] = type(original)(
        **{
            **original.__dict__,
            "exact_cuda_eval_supported": True,
        }
    )
    monkeypatch.setattr(contracts_mod, "PROVIDER_CONTRACTS", bad)
    with pytest.raises(PreflightError, match="scaffold contracts must not advertise"):
        check_provider_deploy_contracts(strict=True, verbose=False)


def test_gcp_scaffold_is_dry_run_only_and_non_promotable() -> None:
    plan = plan_gcp_dispatch(lane_id="lane_test", project="pact-dev").to_dict()
    assert plan["provider"] == "gcp"
    assert plan["dry_run"] is True
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False
    assert plan["requires_lane_claim_before_dispatch"] is True
    assert plan["terminal_claim_required"] is True
    assert plan["custody_manifest_required"] is True


def test_gcp_scaffold_requires_lane_and_project() -> None:
    with pytest.raises(ValueError, match="lane_id"):
        plan_gcp_dispatch(lane_id="", project="pact-dev")
    with pytest.raises(ValueError, match="project"):
        plan_gcp_dispatch(lane_id="lane_test", project="")
