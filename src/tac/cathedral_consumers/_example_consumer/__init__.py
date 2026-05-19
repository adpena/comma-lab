# SPDX-License-Identifier: MIT
"""Reference consumer demonstrating the canonical CathedralConsumerContract.

Per operator directive 2026-05-19 + Catalog #335 + tac.cathedral.consumer_contract.

This is a NO-OP reference implementation. Its purpose is to (a) serve as the
canonical migration template documented in
``src/tac/cathedral_consumers/README.md`` and (b) provide a permanent positive
fixture for the Catalog #335 STRICT preflight gate so the auto-discovery loop
always has at least one well-formed package to ingest.

The underscore prefix (``_example_consumer``) signals to the auto-discovery
loop that this is a reference / example, not a production consumer. The loop
discovers it (for contract-validation purposes) but reports it under the
canonical reference banner so operators can audit the example without
mistaking it for a production ranker contribution.
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "_example_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Reference implementation: NO-OP. A production consumer would refit a
    SLIM ranker / update a Rashomon ensemble member / recompute a
    sensitivity prior from the new empirical anchor.
    """
    _ = anchor  # explicit acknowledgment; no state to update


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Reference implementation: returns zero-adjustment + reference rationale
    + ``[predicted]`` axis tag. A production consumer would compute a
    bounded adjustment from the candidate's payload (e.g. apply a SLIM
    risk score, query a continual-learning posterior, run a Pareto
    feasibility check) and return the canonical contribution dict.

    Per CLAUDE.md "Apples-to-apples evidence discipline": the ``axis_tag``
    field is the canonical disambiguator; promoting ``[predicted]`` to
    ``[contest-CUDA]`` requires paired-axis empirical evidence per Catalog
    #127 + Catalog #323.
    """
    _ = candidate  # explicit acknowledgment; reference does not inspect
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": "Reference no-op consumer; demonstrates canonical contract",
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
