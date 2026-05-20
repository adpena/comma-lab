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

## The dual-tier architecture (Dim 6 + Catalog #357)

Per CATHEDRAL-SMARTER-DESIGN-MEMO Dimension 6 (operator blanket approval
2026-05-20), cathedral consumers belong to one of two tiers.

### Are these consumers "faking"?

**No.** The WIRE-IN-RIGOR audit surfaced an operator-mental-model gap: 44+
cathedral consumers fire per loop iteration but per Catalog #341 ALL return
`predicted_delta_adjustment=0.0`. From outside, this looked like "the
framework is fake — 44 things do nothing." From inside, this is by design:

- Most consumers are **routing recommenders** (e.g. `mps_viable_prescreen_consumer`),
  **annotators** (e.g. `canonical_equation_lookup_consumer`), or
  **diagnostics** (e.g. `substrate_fit_diagnostic_consumer`).
- Per CLAUDE.md "Forbidden score claims" + "MPS auth eval is NOISE": a
  consumer that mutated the score signal without empirical grounding
  would silently leak speculation into the autopilot ranker.
- The `predicted_delta_adjustment=0.0` invariant is the **structural
  protection** that prevents this leak (Catalog #341 STRICT gate).

The framework's *score-mutating* heart lives in the **10 in-main-line
adjusters** of `tools/cathedral_autopilot_autonomous_loop.py::main()` —
not in the cathedral_consumers/* tree. The 44+ consumers contribute
**observability** (routing recommendations, annotations, diagnostics,
continual-learning posterior updates) that the ranker honors *alongside*
the in-line adjusters.

### Tier A: observability-only (default; backward-compatible)

- **Contract**: Catalog #341 canonical routing markers
- **`predicted_delta_adjustment`**: MUST be `0.0`
- **Purpose**: Routing recommendation / annotation / diagnostic only
- **Safe-by-construction**: cannot leak into score signal or promotion
- **Migration burden**: zero (default for existing 44+ consumers; no
  `CONSUMER_TIER` attribute required)

Tier A is the canonical safe choice. The default for any new consumer
that doesn't have a paired empirical anchor for its predictions.

### Tier B: score-contributing (future extension)

- **Contract**: Catalog #357 canonical score-contributing contract
- **`predicted_delta_adjustment`**: MAY be non-zero (signed, finite)
- **REQUIRES** all of:
  1. `CONSUMER_TIER = ConsumerTier.TIER_B_SCORE_CONTRIBUTING` declared at
     module level + import of `ConsumerTier` from
     `tac.cathedral.consumer_contract`
  2. `provenance` field in every `consume_candidate` return value
     (canonical Provenance per Catalog #323)
  3. `promotable=False` preserved (Tier B contributes to RANKING but
     NEVER to PROMOTION; promotion requires paired contest-CPU +
     contest-CUDA empirical anchors per CLAUDE.md "Submission auth eval
     — BOTH CPU AND CUDA")
  4. `axis_tag` MUST be a canonical contest or diagnostic axis
     (NOT `[predicted]` / `[advisory only]` / `[mps-*]` — those re-collapse
     Tier B back into Tier A semantics)
  5. `predicted_axis_decomposition` per `AxisDecomposition` with non-empty
     `canonical_provenance` (Catalog #356 prerequisite; depends on Dim 3
     Step 3.1 landing)

Tier B is enforced structurally by Catalog #357 STRICT preflight gate at
landing time AND by `validate_tier_b_contribution(...)` at runtime.

### Tier B example (canonical template)

```python
# SPDX-License-Identifier: MIT
"""Hypothetical Tier B consumer demonstrating canonical contract."""
from __future__ import annotations
from typing import Any, Mapping
from tac.cathedral.consumer_contract import ConsumerTier, HookNumber

CONSUMER_NAME = "my_tier_b_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,)
CONSUMER_TIER = ConsumerTier.TIER_B_SCORE_CONTRIBUTING


def update_from_anchor(anchor: Any) -> None:
    """Refresh internal posterior from new empirical anchor."""
    # ... refit / update / etc.


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        # Signed bounded score-delta from empirically-grounded posterior
        "predicted_delta_adjustment": -0.0015,
        "rationale": "Bayesian posterior delta from 7-anchor cluster",
        # Empirically-grounded axis (NOT '[predicted]')
        "axis_tag": "[contest-CUDA]",
        # Tier B contributes to ranking, NEVER to promotion
        "promotable": False,
        # Canonical Provenance per Catalog #323
        "provenance": {
            "kind": "CONTEST_ARCHIVE_MEMBER",
            "grade": "contest_cuda",
            # ... per build_provenance_for_predicted(...)
        },
        "confidence": 0.85,
    }
```

### How to promote a consumer Tier A → Tier B

**Prerequisite**: Dim 3 Step 3.4 (per-axis decomposition wire-in) — until
that lands, Tier B promotion is scaffold-only (use the
`# CATHEDRAL_CONSUMER_TIER_B_DEFERRED_OK:<rationale>` waiver).

**Per Dim 6 Step 6.5**, the promotion path is:

1. Demonstrate the consumer's signal is empirically grounded (paired
   anchor on contest-compliant hardware).
2. Wire canonical Provenance per `tac.provenance.builders` into every
   `consume_candidate` return value.
3. Switch `axis_tag` from `[predicted]` to the canonical contest axis the
   anchor used.
4. Add `CONSUMER_TIER = ConsumerTier.TIER_B_SCORE_CONTRIBUTING` + import.
5. Run paired-comparison validation: rank-with-Tier-B vs rank-with-Tier-A
   for the same candidate set; verify the ranking changes correlate with
   empirical residuals.
6. Land with Catalog #357 gate green (no waiver active).

### 6-hook wire-in per consumer-tier

| Hook (#) | Tier A | Tier B |
|---|---|---|
| #1 Sensitivity map | OPTIONAL (annotation only) | OPTIONAL (annotation only) |
| #2 Pareto constraint | OPTIONAL | OPTIONAL |
| #3 Bit-allocator | OPTIONAL | OPTIONAL |
| #4 Cathedral autopilot dispatch | **ACTIVE** (routing/annotation) | **ACTIVE PRIMARY** (ranking contribution) |
| #5 Continual-learning posterior | OPTIONAL | **ACTIVE** (refresh on new anchor) |
| #6 Probe-disambiguator | OPTIONAL | OPTIONAL |

## See also

- `src/tac/cathedral/consumer_contract.py` — canonical Protocol + dataclasses
- `tools/cathedral_autopilot_autonomous_loop.py` — auto-discovery loop
- CLAUDE.md "Catalog #335" — STRICT preflight gate (canonical Protocol contract)
- CLAUDE.md "Catalog #341" — STRICT preflight gate (Tier A routing-markers)
- CLAUDE.md "Catalog #357" — STRICT preflight gate (Tier B canonical contract)
- CLAUDE.md "Catalog #265" — sister canonical-contract pattern (symposium_impls)
- CLAUDE.md "Catalog #125" — 6-hook wire-in non-negotiable
- CLAUDE.md "Meta-Lagrangian/Pareto solver" — no-orphan-signals discipline
- CLAUDE.md "Apples-to-apples evidence discipline" — canonical axis tokens
- CLAUDE.md "MPS auth eval is NOISE" — non-promotable structural protection
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" — promotion requires paired-axis
- `feedback_cathedral_auto_ingest_paradigm_shift_landed_20260519` — original landing memo
- `feedback_wave_1_dim_6_dual_tier_consumer_architecture_foundation_landed_20260520` — dual-tier landing memo
