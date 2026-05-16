# SPDX-License-Identifier: MIT
"""Registry/inventory visibility tests for the TT5L substrate."""

from __future__ import annotations

import importlib

from tac.optimization.autopilot_dispatch_ranking import rank_dispatches
from tac.optimization.substrate_composition_matrix import canonical_substrate_inventory
from tac.substrate_registry import (
    _clear_registry_for_tests,
    _REGISTERED_SUBSTRATES,
    get_registered_substrates,
)
from tac.substrates.time_traveler_l5_autonomy.archive import (
    TT5L_MAGIC,
    TT5L_SECTION_ROLES,
)


def test_time_traveler_l5_contract_registered_from_package() -> None:
    """Importing the package-level contract makes TT5L visible to auto-wire."""
    snapshot = dict(_REGISTERED_SUBSTRATES)
    try:
        _clear_registry_for_tests()
        module = importlib.import_module(
            "tac.substrates.time_traveler_l5_autonomy.registered_substrate"
        )
        importlib.reload(module)

        registered = get_registered_substrates()
        assert "time_traveler_l5_autonomy" in registered
        contract = registered["time_traveler_l5_autonomy"]
        assert TT5L_MAGIC.decode("ascii") in contract.archive_grammar
        assert "TTL5V1" not in contract.archive_grammar
        assert dict(contract.parser_section_manifest) == TT5L_SECTION_ROLES
        assert (
            contract.hook_autopilot_ranker_class_shift_token
            == "time_traveler_l5_autonomy"
        )
    finally:
        _clear_registry_for_tests()
        _REGISTERED_SUBSTRATES.update(snapshot)


def test_time_traveler_l5_visible_to_inventory_and_ranker() -> None:
    inventory_ids = {row.substrate_id for row in canonical_substrate_inventory()}
    assert "time_traveler_l5_autonomy" in inventory_ids

    ranking = rank_dispatches(drop_redundant_dominated=False, max_total=None)
    ranked_ids = {
        substrate_id
        for candidate in ranking.ranked_dispatches
        for substrate_id in candidate.substrate_ids
    }
    assert "time_traveler_l5_autonomy" in ranked_ids
