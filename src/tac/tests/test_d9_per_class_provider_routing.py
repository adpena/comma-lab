# SPDX-License-Identifier: MIT
"""Tests for D9 per-class provider routing API (council Decision 9 binding verdict).

Covers the canonical routing decision matrix, cost-band-posterior consumption,
dynamic re-routing trigger, recipe-side override pass-through, and end-to-end
:func:`select_provider_for_recipe`. Per CLAUDE.md "Subagent coherence-by-default"
mandatory wire-in tests cover all 5 dispatch classes (smoke, full, long_burn,
eval, cpu) declared in :data:`tac.cost_band_calibration.DISPATCH_CLASSES`.

Council Decision 9 anchors:

  * smoke (≤30 min, ≤$2): Modal T4 default; Modal A10G fallback
  * full  (1-12h, $2-$15): Vast.ai RTX 4090 default; Modal A100 fallback
  * long_burn (12h+, $50+): Lightning A100 default; Vast.ai H100 fallback
  * eval (auth eval): Modal T4 default; Modal A10G fallback
  * cpu (contest-CPU): GHA Linux x86_64 default

Time-Traveler amendment: cost-band posterior anchor shifts >25% trigger re-route.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.cost_band_calibration import (
    CANONICAL_PROVIDER_PER_CLASS,
    DISPATCH_CLASSES,
    FALLBACK_PROVIDERS_PER_CLASS,
    PER_CLASS_SOFT_COST_CEILING_USD,
    PER_CLASS_SOFT_WALLCLOCK_CEILING_HR,
    RE_ROUTING_TRIGGER_FRACTION,
    CostBandAnchor,
    ProviderRoutingDecision,
    SUCCESSFUL_DISPATCH,
    append_anchor,
    classify_dispatch,
    select_provider_for_class,
    select_provider_for_recipe,
)


# -- Decision-matrix invariants ---------------------------------------------


def test_council_decision_9_canonical_provider_matrix() -> None:
    """The canonical (provider, gpu) tuples MUST exactly match Decision 9.

    This pins the binding council verdict so a future edit cannot silently
    rename a class or swap a provider.
    """
    assert CANONICAL_PROVIDER_PER_CLASS == {
        "smoke": ("modal", "T4"),
        "full": ("vastai", "RTX_4090"),
        "long_burn": ("lightning", "A100"),
        "eval": ("modal", "T4"),
        "cpu": ("github", "ubuntu-latest"),
    }


def test_council_decision_9_fallback_provider_matrix() -> None:
    """The fallback list MUST match Decision 9 verdict ordering."""
    assert FALLBACK_PROVIDERS_PER_CLASS == {
        "smoke": [("modal", "A10G")],
        "full": [("modal", "A100")],
        "long_burn": [("vastai", "H100")],
        "eval": [("modal", "A10G")],
        "cpu": [],
    }


def test_dispatch_classes_enumeration() -> None:
    assert DISPATCH_CLASSES == ("smoke", "full", "long_burn", "eval", "cpu")


def test_re_routing_trigger_fraction_is_25_percent() -> None:
    """Time-Traveler amendment trigger MUST be 25% per the council ledger."""
    assert RE_ROUTING_TRIGGER_FRACTION == 0.25


# -- classify_dispatch -------------------------------------------------------


def test_classify_dispatch_explicit_override_wins() -> None:
    assert classify_dispatch(explicit_dispatch_class="long_burn") == "long_burn"
    assert classify_dispatch(explicit_dispatch_class="cpu") == "cpu"


def test_classify_dispatch_unknown_explicit_falls_through() -> None:
    """An unknown explicit_dispatch_class should fall through to inference,
    not raise — typos must not crash a routing call."""
    # Wrong explicit + smoke label -> infers smoke.
    assert classify_dispatch(
        explicit_dispatch_class="not_a_class",
        dispatch_label="substrate_x_modal_a100_dispatch_20260514T010000Z__smoke__100ep",
    ) == "smoke"


def test_classify_dispatch_smoke_label_token() -> None:
    label = "substrate_pr101_lc_v2_clone_enhanced_curriculum_modal_a100_dispatch_20260514T010436Z__smoke__100ep"
    assert classify_dispatch(dispatch_label=label) == "smoke"


def test_classify_dispatch_eval_label_token() -> None:
    assert classify_dispatch(dispatch_label="gha_cpu_eval_pr106_pr101grammar_20260514") == "eval"
    assert classify_dispatch(dispatch_label="modal_auth_eval_x_20260514") == "eval"


def test_classify_dispatch_long_burn_label_token() -> None:
    assert classify_dispatch(dispatch_label="lane_x_long_burn_20260514") == "long_burn"


def test_classify_dispatch_smoke_via_epochs_only() -> None:
    """Recipes pre-dating the smoke label convention: epochs<=200 -> smoke."""
    assert classify_dispatch(epochs=100) == "smoke"
    assert classify_dispatch(epochs=200) == "smoke"


def test_classify_dispatch_long_burn_via_wallclock() -> None:
    assert (
        classify_dispatch(estimated_wall_clock_sec=15 * 3600.0)
        == "long_burn"
    )


def test_classify_dispatch_full_default() -> None:
    """Ambiguous recipe (no label, no epochs, no wallclock) -> 'full' default."""
    assert classify_dispatch() == "full"
    assert classify_dispatch(dispatch_label="some_random_label") == "full"


def test_classify_dispatch_smoke_via_cost_ceiling() -> None:
    assert classify_dispatch(estimated_cost_usd=1.5) == "smoke"


def test_classify_dispatch_long_burn_via_cost_ceiling() -> None:
    assert classify_dispatch(estimated_cost_usd=120.0) == "long_burn"


# -- select_provider_for_class: canonical decisions --------------------------


def test_select_provider_smoke_canonical() -> None:
    decision = select_provider_for_class(
        "smoke",
        consult_posterior=False,
    )
    assert decision.provider == "modal"
    assert decision.gpu == "T4"
    assert decision.canonical_provider == "modal"
    assert decision.canonical_gpu == "T4"
    assert decision.re_routed is False
    assert "Decision 9 canonical" in decision.rationale


def test_select_provider_full_canonical() -> None:
    decision = select_provider_for_class(
        "full",
        consult_posterior=False,
    )
    assert decision.provider == "vastai"
    assert decision.gpu == "RTX_4090"
    assert decision.canonical_provider == "vastai"
    assert decision.canonical_gpu == "RTX_4090"
    assert decision.re_routed is False


def test_select_provider_long_burn_canonical() -> None:
    decision = select_provider_for_class(
        "long_burn",
        consult_posterior=False,
    )
    assert decision.provider == "lightning"
    assert decision.gpu == "A100"


def test_select_provider_eval_canonical() -> None:
    decision = select_provider_for_class(
        "eval",
        consult_posterior=False,
    )
    assert decision.provider == "modal"
    assert decision.gpu == "T4"


def test_select_provider_cpu_canonical() -> None:
    decision = select_provider_for_class(
        "cpu",
        consult_posterior=False,
    )
    assert decision.provider == "github"
    assert decision.gpu == "ubuntu-latest"


def test_select_provider_unknown_class_defaults_to_full() -> None:
    decision = select_provider_for_class(
        "unknown_class_typo",
        consult_posterior=False,
    )
    assert decision.dispatch_class == "full"
    assert decision.provider == "vastai"
    assert decision.gpu == "RTX_4090"
    assert "unknown" in decision.rationale.lower()


# -- Recipe-side override pass-through ---------------------------------------


def test_select_provider_recipe_explicit_provider_wins() -> None:
    """When the recipe sets `provider:` to a non-auto value, the helper passes
    through (operator's explicit choice always wins per CLAUDE.md)."""
    decision = select_provider_for_class(
        "full",
        recipe_meta={"provider": "modal", "gpu": "A100"},
        consult_posterior=False,
    )
    assert decision.provider == "modal"
    assert decision.gpu == "A100"
    # Canonical fields STILL record what Decision 9 would have chosen, so
    # forensics show the operator opted out of canonical routing.
    assert decision.canonical_provider == "vastai"
    assert decision.canonical_gpu == "RTX_4090"
    assert "explicitly set" in decision.rationale


def test_select_provider_recipe_provider_auto_uses_canonical() -> None:
    """`provider: auto` is the explicit opt-in to Decision 9 canonical."""
    decision = select_provider_for_class(
        "full",
        recipe_meta={"provider": "auto"},
        consult_posterior=False,
    )
    assert decision.provider == "vastai"
    assert decision.gpu == "RTX_4090"


def test_select_provider_recipe_platform_alias_for_provider() -> None:
    """The recipe key `platform:` is treated as an alias for `provider:`
    (operator_authorize.py reads `recipe.platform` from the same field)."""
    decision = select_provider_for_class(
        "smoke",
        recipe_meta={"platform": "lightning", "gpu": "T4"},
        consult_posterior=False,
    )
    assert decision.provider == "lightning"
    assert decision.gpu == "T4"


# -- Cost-band-posterior consumption ----------------------------------------


def _seed_posterior(
    tmp_path: Path,
    rows: list[dict],
) -> tuple[Path, Path]:
    """Helper: write a cost-band posterior at `tmp_path/cost_band.jsonl`.

    Returns ``(posterior_path, lock_path)`` for caller injection.
    """
    posterior = tmp_path / "cost_band.jsonl"
    lock = tmp_path / ".cost_band.lock"
    with posterior.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return posterior, lock


def _make_anchor_dict(
    *,
    platform: str,
    gpu: str,
    epochs: int,
    actual_cost_usd: float,
    actual_wall_clock_sec: float,
    outcome: str = SUCCESSFUL_DISPATCH,
    label: str | None = None,
) -> dict:
    return {
        "schema": "cost_band_posterior_v1",
        "schema_version": 1,
        "logged_at_utc": "2026-05-14T12:00:00+00:00",
        "dispatch_label": label or f"{platform}_{gpu}_test",
        "trainer": "experiments/train_test.py",
        "platform": platform,
        "gpu": gpu,
        "epochs": epochs,
        "batch_size": 16,
        "all_flags_on": True,
        "actual_wall_clock_sec": actual_wall_clock_sec,
        "actual_cost_usd": actual_cost_usd,
        "predicted_cost_usd_low": None,
        "predicted_cost_usd_high": None,
        "prediction_in_band": None,
        "outcome": outcome,
        "returncode": 0 if outcome == SUCCESSFUL_DISPATCH else 1,
        "notes": "test anchor",
    }


def test_select_provider_posterior_consulted_no_anchors(tmp_path: Path) -> None:
    """No anchors -> canonical decision returned, posterior_consulted=True,
    re_routed=False."""
    posterior = tmp_path / "empty.jsonl"
    posterior.touch()
    decision = select_provider_for_class(
        "full",
        posterior_path=posterior,
        consult_posterior=True,
    )
    assert decision.provider == "vastai"
    assert decision.gpu == "RTX_4090"
    assert decision.posterior_consulted is True
    assert decision.re_routed is False
    assert decision.canonical_cost_p50_usd is None
    assert decision.fallback_cost_p50_usd is None


def test_select_provider_posterior_re_routes_when_fallback_30pct_cheaper(
    tmp_path: Path,
) -> None:
    """If empirical fallback is >25% cheaper than canonical (matched
    confidence floor), recommend the fallback."""
    rows = []
    # 5 successful Vast.ai RTX_4090 anchors at $5.00 each (canonical for full)
    for i in range(5):
        rows.append(
            _make_anchor_dict(
                platform="vastai",
                gpu="RTX_4090",
                epochs=3000,
                actual_cost_usd=5.00,
                actual_wall_clock_sec=4 * 3600.0,
            )
        )
    # 5 successful Modal A100 anchors at $3.00 each (fallback for full;
    # 40% cheaper than canonical -> trigger re-route).
    for i in range(5):
        rows.append(
            _make_anchor_dict(
                platform="modal",
                gpu="A100",
                epochs=3000,
                actual_cost_usd=3.00,
                actual_wall_clock_sec=2 * 3600.0,
            )
        )
    posterior, _ = _seed_posterior(tmp_path, rows)

    decision = select_provider_for_class(
        "full",
        posterior_path=posterior,
        consult_posterior=True,
        epochs_for_posterior=3000,
    )
    assert decision.re_routed is True
    assert decision.provider == "modal"
    assert decision.gpu == "A100"
    assert decision.canonical_provider == "vastai"
    assert decision.canonical_gpu == "RTX_4090"
    assert decision.canonical_cost_p50_usd == pytest.approx(5.00)
    assert decision.fallback_cost_p50_usd == pytest.approx(3.00)
    assert "Time-Traveler amendment trigger" in decision.re_routing_rationale


def test_select_provider_posterior_does_NOT_re_route_when_fallback_only_20pct_cheaper(
    tmp_path: Path,
) -> None:
    """20% < 25% threshold -> stick with canonical."""
    rows = []
    for i in range(5):
        rows.append(
            _make_anchor_dict(
                platform="vastai",
                gpu="RTX_4090",
                epochs=3000,
                actual_cost_usd=5.00,
                actual_wall_clock_sec=4 * 3600.0,
            )
        )
    # 20% cheaper -> below trigger.
    for i in range(5):
        rows.append(
            _make_anchor_dict(
                platform="modal",
                gpu="A100",
                epochs=3000,
                actual_cost_usd=4.00,
                actual_wall_clock_sec=2 * 3600.0,
            )
        )
    posterior, _ = _seed_posterior(tmp_path, rows)

    decision = select_provider_for_class(
        "full",
        posterior_path=posterior,
        consult_posterior=True,
        epochs_for_posterior=3000,
    )
    assert decision.re_routed is False
    assert decision.provider == "vastai"
    assert decision.gpu == "RTX_4090"
    # Both p50s STILL recorded for forensics.
    assert decision.canonical_cost_p50_usd == pytest.approx(5.00)
    assert decision.fallback_cost_p50_usd == pytest.approx(4.00)


def test_select_provider_posterior_excludes_failed_anchors(tmp_path: Path) -> None:
    """Failed/timed-out anchors must NOT contribute to the routing decision
    per Catalog #175 + #177 cost-band-outcome discipline."""
    rows = []
    # Canonical: 5 SUCCESSFUL at $5
    for i in range(5):
        rows.append(
            _make_anchor_dict(
                platform="vastai",
                gpu="RTX_4090",
                epochs=3000,
                actual_cost_usd=5.00,
                actual_wall_clock_sec=4 * 3600.0,
            )
        )
    # Fallback: 5 FAILED at $0.10 each (would re-route to modal/A100 if folded in)
    for i in range(5):
        rows.append(
            _make_anchor_dict(
                platform="modal",
                gpu="A100",
                epochs=3000,
                actual_cost_usd=0.10,
                actual_wall_clock_sec=120.0,
                outcome="failed_dispatch",
            )
        )
    posterior, _ = _seed_posterior(tmp_path, rows)

    decision = select_provider_for_class(
        "full",
        posterior_path=posterior,
        consult_posterior=True,
        epochs_for_posterior=3000,
    )
    # No successful fallback anchors -> stick with canonical, no re-route.
    assert decision.re_routed is False
    assert decision.provider == "vastai"
    assert decision.fallback_cost_p50_usd is None


def test_select_provider_posterior_skipped_when_consult_false(tmp_path: Path) -> None:
    posterior, _ = _seed_posterior(tmp_path, [])
    decision = select_provider_for_class(
        "smoke",
        posterior_path=posterior,
        consult_posterior=False,
    )
    assert decision.posterior_consulted is False
    assert decision.canonical_cost_p50_usd is None
    assert decision.fallback_cost_p50_usd is None


# -- select_provider_for_recipe --------------------------------------------


def test_select_provider_for_recipe_classifies_smoke_via_label() -> None:
    """The end-to-end helper threads recipe meta through classify_dispatch
    and returns the correct routing decision."""
    decision = select_provider_for_recipe(
        {
            "dispatch_label": "substrate_x_modal_a100_dispatch_20260514T010000Z__smoke__100ep",
            "cost_band": {"epochs": 100},
        },
        consult_posterior=False,
    )
    assert decision.dispatch_class == "smoke"
    assert decision.provider == "modal"
    assert decision.gpu == "T4"


def test_select_provider_for_recipe_explicit_class_wins() -> None:
    decision = select_provider_for_recipe(
        {
            "dispatch_class": "long_burn",
            "cost_band": {"epochs": 50},  # would normally classify as smoke
        },
        consult_posterior=False,
    )
    assert decision.dispatch_class == "long_burn"
    assert decision.provider == "lightning"


def test_select_provider_for_recipe_recipe_provider_overrides_routing() -> None:
    """Recipe-side `provider:` always wins (operator's explicit choice)."""
    decision = select_provider_for_recipe(
        {
            "dispatch_label": "x__smoke__100ep",
            "provider": "vastai",
            "gpu": "H100",
        },
        consult_posterior=False,
    )
    assert decision.provider == "vastai"
    assert decision.gpu == "H100"
    # Classification still happened.
    assert decision.dispatch_class == "smoke"
    # And the canonical Decision 9 verdict for smoke is recorded.
    assert decision.canonical_provider == "modal"
    assert decision.canonical_gpu == "T4"


def test_select_provider_for_recipe_threads_epochs_to_posterior(
    tmp_path: Path,
) -> None:
    """The epochs from cost_band.epochs MUST be threaded to the posterior
    consultation so per-bucket anchors are queried correctly."""
    rows = [
        _make_anchor_dict(
            platform="modal",
            gpu="T4",
            epochs=100,  # smoke bucket
            actual_cost_usd=0.50,
            actual_wall_clock_sec=300.0,
        )
        for _ in range(5)
    ]
    posterior, _ = _seed_posterior(tmp_path, rows)
    decision = select_provider_for_recipe(
        {
            "dispatch_label": "x__smoke__100ep",
            "cost_band": {"epochs": 100},
        },
        posterior_path=posterior,
        consult_posterior=True,
    )
    assert decision.posterior_consulted is True
    assert decision.canonical_cost_p50_usd == pytest.approx(0.50)


# -- ProviderRoutingDecision dataclass surface ------------------------------


def test_provider_routing_decision_as_dict_round_trip() -> None:
    decision = select_provider_for_class("smoke", consult_posterior=False)
    d = decision.as_dict()
    assert d["dispatch_class"] == "smoke"
    assert d["provider"] == "modal"
    assert d["gpu"] == "T4"
    assert "rationale" in d


def test_provider_routing_decision_is_frozen() -> None:
    decision = select_provider_for_class("smoke", consult_posterior=False)
    with pytest.raises((AttributeError, Exception)):
        # frozen=True dataclass should refuse mutation
        decision.provider = "lightning"  # type: ignore[misc]


# -- Live-repo regression guards --------------------------------------------


def test_per_class_cost_ceiling_invariant() -> None:
    """Cost ceilings MUST be monotonic: smoke <= eval < full < long_burn.

    'cpu' is special (free) so it sits at $0.
    """
    assert PER_CLASS_SOFT_COST_CEILING_USD["smoke"] == 2.0
    assert PER_CLASS_SOFT_COST_CEILING_USD["eval"] == 2.0
    assert PER_CLASS_SOFT_COST_CEILING_USD["full"] == 15.0
    assert PER_CLASS_SOFT_COST_CEILING_USD["long_burn"] == 100.0
    assert PER_CLASS_SOFT_COST_CEILING_USD["cpu"] == 0.0


def test_per_class_wallclock_ceiling_invariant() -> None:
    """Wall-clock ceilings: smoke <= eval < full < long_burn."""
    assert PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["smoke"] == 0.5
    assert PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["eval"] == 0.5
    assert PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["full"] == 12.0
    assert PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["long_burn"] == 168.0
    assert PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["cpu"] == 12.0


def test_routing_consumes_real_posterior_without_crash() -> None:
    """Smoke test against the real `.omx/state/cost_band_posterior.jsonl`
    — must not crash regardless of posterior contents."""
    decision = select_provider_for_class("smoke", consult_posterior=True)
    assert decision.dispatch_class == "smoke"
    assert decision.provider in {"modal", "vastai", "lightning", "github"}
    assert decision.posterior_consulted is True
