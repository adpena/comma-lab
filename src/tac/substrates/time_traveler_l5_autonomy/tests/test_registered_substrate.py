# SPDX-License-Identifier: MIT
"""Registry/inventory visibility tests for the TT5L substrate."""

from __future__ import annotations

import importlib
from pathlib import Path

from tac.optimization.autopilot_dispatch_ranking import rank_dispatches
from tac.optimization.l5_v2_probe_disambiguator import L5V2_PROBE_TOOL_PATH
from tac.optimization.substrate_composition_matrix import canonical_substrate_inventory
from tac.substrate_registry import (
    _REGISTERED_SUBSTRATES,
    _clear_registry_for_tests,
    get_registered_substrates,
    query_substrates_for_probe_disambiguators,
)
from tac.substrates.time_traveler_l5_autonomy.archive import (
    TT5L_MAGIC,
    TT5L_SECTION_ROLES,
)
from tac.substrates.time_traveler_l5_autonomy.archive import (
    __file__ as archive_module_file,
)
from tac.substrates.time_traveler_l5_autonomy.registered_substrate import (
    TIME_TRAVELER_L5_AUTONOMY_SUBSTRATE_CONTRACT,
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
        assert contract.hook_probe_disambiguator == L5V2_PROBE_TOOL_PATH
        assert Path(L5V2_PROBE_TOOL_PATH).is_file()
        assert contract.runtime_dep_closure == (
            "torch>=2.5,<2.7",
            "brotli",
            "numpy",
        )
        assert "av" not in contract.runtime_dep_closure
        assert contract.inflate_runtime_loc_budget >= 327
    finally:
        _clear_registry_for_tests()
        _REGISTERED_SUBSTRATES.update(snapshot)


def test_time_traveler_l5_visible_to_inventory_and_ranker() -> None:
    inventory_ids = {row.substrate_id for row in canonical_substrate_inventory()}
    assert "time_traveler_l5_autonomy" in inventory_ids
    tt5l = next(
        row for row in canonical_substrate_inventory()
        if row.substrate_id == "time_traveler_l5_autonomy"
    )
    assert tt5l.runtime_dep_closure == ("torch", "brotli", "numpy")
    assert "av" not in tt5l.runtime_dep_closure

    ranking = rank_dispatches(drop_redundant_dominated=False, max_total=None)
    ranked_ids = {
        substrate_id
        for candidate in ranking.ranked_dispatches
        for substrate_id in candidate.substrate_ids
    }
    assert "time_traveler_l5_autonomy" in ranked_ids


def test_time_traveler_l5_declares_archive_numpy_runtime_dependency() -> None:
    archive_source = Path(archive_module_file).read_text(encoding="utf-8")
    assert "import numpy as np" in archive_source

    registered = TIME_TRAVELER_L5_AUTONOMY_SUBSTRATE_CONTRACT.runtime_dep_closure
    assert "numpy" in registered

    tt5l = next(
        row for row in canonical_substrate_inventory()
        if row.substrate_id == "time_traveler_l5_autonomy"
    )
    assert "numpy" in tt5l.runtime_dep_closure


def test_time_traveler_l5_probe_disambiguator_is_auto_wire_visible() -> None:
    snapshot = dict(_REGISTERED_SUBSTRATES)
    try:
        _clear_registry_for_tests()
        module = importlib.import_module(
            "tac.substrates.time_traveler_l5_autonomy.registered_substrate"
        )
        importlib.reload(module)

        rows = query_substrates_for_probe_disambiguators()
        assert rows["time_traveler_l5_autonomy"] == L5V2_PROBE_TOOL_PATH
    finally:
        _clear_registry_for_tests()
        _REGISTERED_SUBSTRATES.update(snapshot)
