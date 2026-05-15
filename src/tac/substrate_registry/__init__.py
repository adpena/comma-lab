"""META layer for substrate contracts + auto-wire.

Operator directive 2026-05-15: "build a META layer that defines a
schema/contract substrates must respect, then makes wire-in AUTOMATIC across
the 6 canonical hooks per Catalog #125."

Public API (narrow per CLAUDE.md "Beauty, simplicity, and developer experience"):

  - ``register_substrate(contract)``: decorator that captures + registers a
    substrate's contract at import time.
  - ``SubstrateContract``: 36-field canonical schema (frozen dataclass).
  - ``SubstrateContractError``: raised at decoration time on contract drift.
  - ``get_registered_substrates()``: read API for the in-memory registry.
  - ``validate_all_registered()``: defensive re-validation helper.
  - 6 hook query helpers (``query_substrates_for_*``) for downstream
    consumer modules (sister WIRE-AND-INTEGRATE-ALL subagent owns the
    consumer-side integration).
  - 2 generator helpers (``generate_recipe_yaml``, ``generate_driver_shell``)
    for auto-emission of recipe + remote driver from a contract.

Cross-references:
  - Design memo: `.omx/research/substrate_meta_layer_design_20260515.md`
  - Premise verification (Catalog #229): `.omx/tmp/meta_layer_premise_verifier.py`
  - STRICT preflight Catalog #241 (decorator-or-legacy-tagged), #242
    (contract-fields-canonical).
"""

from __future__ import annotations

from tac.substrate_registry.auto_wire import (
    query_substrates_by_compliance_token,
    query_substrates_for_autopilot_ranker,
    query_substrates_for_bit_allocator_hook,
    query_substrates_for_continual_learning_anchor_kind,
    query_substrates_for_pareto_hook,
    query_substrates_for_probe_disambiguators,
    query_substrates_for_sensitivity_hook,
)
from tac.substrate_registry.contract import (
    KNOWN_CATALOG_COMPLIANCE_TOKENS,
    LEGAL_CANARY_STATUS,
    LEGAL_DEPLOYMENT_TARGETS,
    LEGAL_EXPORT_FORMATS,
    LEGAL_GPU_KEY,
    LEGAL_HOOK_BIT_ALLOCATOR,
    LEGAL_HOOK_CONTINUAL_LEARNING,
    LEGAL_HOOK_PARETO,
    LEGAL_HOOK_SENSITIVITY,
    LEGAL_MIN_SMOKE_GPU,
    LEGAL_OPERATIONAL_STATUS,
    LEGAL_PLATFORM_KEY,
    LEGAL_PYAV_DECODE_STRATEGY,
    LEGAL_SCORE_AWARE_LOSS,
    LEGAL_TARGET_MODES,
    LEGAL_VIDEO_INPUT_STRATEGY,
    NOT_APPLICABLE_WITH_RATIONALE,
    SubstrateContract,
    SubstrateContractError,
)
from tac.substrate_registry.decorator import (
    _clear_registry_for_tests,
    _REGISTERED_SUBSTRATES,
    get_registered_substrates,
    register_substrate,
    validate_all_registered,
)
from tac.substrate_registry.driver_generator import (
    default_driver_relpath,
    generate_driver_shell,
)
from tac.substrate_registry.recipe_generator import (
    default_recipe_relpath,
    generate_recipe_yaml,
)

__all__ = [
    # Decorator + registry
    "register_substrate",
    "get_registered_substrates",
    "validate_all_registered",
    "_REGISTERED_SUBSTRATES",
    "_clear_registry_for_tests",
    # Contract + errors + enums
    "SubstrateContract",
    "SubstrateContractError",
    "NOT_APPLICABLE_WITH_RATIONALE",
    "KNOWN_CATALOG_COMPLIANCE_TOKENS",
    "LEGAL_TARGET_MODES",
    "LEGAL_DEPLOYMENT_TARGETS",
    "LEGAL_EXPORT_FORMATS",
    "LEGAL_SCORE_AWARE_LOSS",
    "LEGAL_OPERATIONAL_STATUS",
    "LEGAL_MIN_SMOKE_GPU",
    "LEGAL_PYAV_DECODE_STRATEGY",
    "LEGAL_CANARY_STATUS",
    "LEGAL_VIDEO_INPUT_STRATEGY",
    "LEGAL_HOOK_SENSITIVITY",
    "LEGAL_HOOK_PARETO",
    "LEGAL_HOOK_BIT_ALLOCATOR",
    "LEGAL_HOOK_CONTINUAL_LEARNING",
    "LEGAL_GPU_KEY",
    "LEGAL_PLATFORM_KEY",
    # Auto-wire query helpers
    "query_substrates_for_sensitivity_hook",
    "query_substrates_for_pareto_hook",
    "query_substrates_for_bit_allocator_hook",
    "query_substrates_for_autopilot_ranker",
    "query_substrates_for_continual_learning_anchor_kind",
    "query_substrates_for_probe_disambiguators",
    "query_substrates_by_compliance_token",
    # Generators
    "generate_recipe_yaml",
    "default_recipe_relpath",
    "generate_driver_shell",
    "default_driver_relpath",
]
