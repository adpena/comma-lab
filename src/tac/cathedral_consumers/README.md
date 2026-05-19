# Cathedral autopilot auto-discovered consumers

Per operator directive 2026-05-19 verbatim:
> "What if we change the paradigm by making cathedral autopilot ingest by default
> if within a certain directory and exposing/respecting a certain contract or
> schema. Fix permanently and self protect against"

## The paradigm shift

**Convention over configuration.** Every package in this directory is
auto-discovered by `tools/cathedral_autopilot_autonomous_loop.discover_and_register_consumers`
at loop start. No manual import-by-import wiring required.

This permanently extincts the **orphan-signal-at-cathedral-autopilot** bug
class. Per the wiring + integration audit (commit `3821cfb6b` 2026-05-19): 12
NEW `tac.*` namespaces landed in the recent window; **zero** consumed by
cathedral autopilot. Manual wire-in fixes CURRENT orphans but doesn't prevent
the 13th from landing tomorrow. The paradigm shift fixes the bug class
**structurally**.

## How to land a new consumer

1. Create directory: `src/tac/cathedral_consumers/<consumer_name>/`
2. Implement `__init__.py` exposing the canonical contract:

```python
# SPDX-License-Identifier: MIT
"""Brief one-line description.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
"""
from __future__ import annotations
from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber

CONSUMER_NAME = "my_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update."""
    # ... refit ranker / update prior / etc.


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution."""
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": "documented mechanism",
        "axis_tag": "[predicted]",
    }
```

3. Run preflight: `.venv/bin/python -m pytest src/tac/tests/test_check_335_cathedral_consumer_directory_contract.py`
4. Commit via canonical serializer per CLAUDE.md "Subagent commits MUST use serializer"

That's it. The next cathedral autopilot loop iteration auto-discovers + auto-registers your consumer.

## The canonical contract

See `src/tac/cathedral/consumer_contract.py` for the full Protocol + validator.

**Required module-level fields:**
- `CONSUMER_NAME: str` — human-readable consumer identifier
- `CONSUMER_VERSION: str` — semver-like (not strictly enforced)
- `CONSUMER_HOOK_NUMBERS: tuple[HookNumber, ...]` — Catalog #125 6-hook surfaces

**Required callable surfaces:**
- `update_from_anchor(anchor) -> None` — Catalog #125 hook #5
- `consume_candidate(candidate) -> Mapping[str, Any]` — Catalog #125 hook #4

**`consume_candidate` return contract:**
- `predicted_delta_adjustment: float` (bounded, non-NaN, additive to candidate's predicted delta)
- `rationale: str` (≥4 chars, human-readable)
- `axis_tag: str` (one of CLAUDE.md "Apples-to-apples" canonical tags)
- Optional: `promotable`, `provenance`, `confidence`

## Deferring contract compliance

If your consumer needs to land before the canonical contract is wired (e.g.
Phase 1 scaffold pending Phase 2 integration), add a same-line waiver in the
`__init__.py` first 30 lines:

```python
# CATHEDRAL_CONSUMER_DEFERRED_OK:Pending Phase 2 wire-in per <lane-id>
```

Per Catalog #287 sister discipline: placeholder rationales (`<rationale>` /
`<reason>` / empty / <4 chars) are rejected.

## See also

- `src/tac/cathedral/consumer_contract.py` — canonical Protocol + dataclasses
- `tools/cathedral_autopilot_autonomous_loop.py` — auto-discovery loop
- CLAUDE.md "Catalog #335" — STRICT preflight gate
- CLAUDE.md "Catalog #265" — sister canonical-contract pattern (symposium_impls)
- CLAUDE.md "Catalog #125" — 6-hook wire-in non-negotiable
- CLAUDE.md "Meta-Lagrangian/Pareto solver" — no-orphan-signals discipline
- `feedback_cathedral_auto_ingest_paradigm_shift_landed_20260519` — landing memo
