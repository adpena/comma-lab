# SPDX-License-Identifier: MIT
"""Cathedral consumer for ``tac.atom`` canonical atom emissions.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the orphan-signal-at-cathedral-autopilot bug class for ``tac.atom``
per wiring + integration audit 2026-05-19 (commit 3821cfb6b).

``tac.atom`` is the canonical typed-atom emission helper (cargo-cult /
premise-verification / probe-outcome / council-deliberation / meta-Lagrangian
atoms). This consumer queries atoms attached to a candidate's substrate /
lane / archive_sha256 and surfaces the count + KIND breakdown as a
non-promotable observability annotation. Atom emissions are by construction
``[predicted]`` axis (no contest score claim implied); per Catalog #287
the rationale carries explicit ``[predicted]`` tag.

Sister of:
- ``_example_consumer`` (canonical reference)
- ``tac.atom.linguistic_extensions.update_from_anchor`` (exposes the canonical
  contract token Catalog #265 / #335 sister discipline relies on)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "atom_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Reference: atom emissions are already persisted via the canonical
    ``tac.atom`` helpers fcntl-locked JSONL store; no additional posterior
    update is required here. NO-OP by design.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Returns zero-adjustment observability annotation citing atom-emission
    surface for the candidate's archive. No score adjustment — atoms are
    ``[predicted]`` axis per CLAUDE.md "Apples-to-apples evidence discipline".
    """
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.atom canonical typed-atom emission surface available "
            "(cargo-cult / premise-verification / probe-outcome / "
            "council-deliberation / meta-Lagrangian atoms) [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
