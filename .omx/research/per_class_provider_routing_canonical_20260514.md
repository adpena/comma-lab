# Per-Class Provider Routing Canonical (D9)

**Council verdict.** Grand Council omnibus 2026-05-14, Decision 9, **PROCEED Option B (per-class canonical routing)**, vote tally 11/11. Time-Traveler amendment (dynamic re-routing on cost-band posterior shift >25%) **adopted**. Contrarian SUPER-VETO **not invoked**. Source ledger: `.omx/research/grand_council_omnibus_design_decisions_20260514.md` lines 392-441; commit `7872c9f4b`.

**Status.** L1 (impl_complete + memory_entry); strict-from-byte-one for the routing-decision matrix invariants per CLAUDE.md "Strict-flip atomicity rule". Live test count at landing: 50/50 dedicated + 121/121 sister regression.

**Scope.** This document is the operator-facing canonical for the per-class routing decision matrix. The `tac.cost_band_calibration.select_provider_for_class` + `select_provider_for_recipe` API consume this matrix; `tools/operator_authorize.py::_maybe_apply_auto_routing` wires it into the dispatch path.

---

## 1. Dispatch class taxonomy

Every dispatch is classified into ONE of five canonical classes. Operators may either declare `dispatch_class:` explicitly in the recipe YAML, or let the helper infer it from the recipe meta.

| class | wall-clock band | cost band ($) | typical recipes |
|---|---|---|---|
| **smoke** | ≤ 30 min | ≤ $2 | smoke-before-full pre-validation; integration probes; canary launches |
| **full** | 1–12 h | $2 – $15 | substrate trainer 100–3000 epochs; T1 Tier 1 envelope |
| **long_burn** | 12 h+ | $50+ | Phase 2/3 long-run; multi-day curriculum; 6000+ epochs |
| **eval** | ≤ 30 min | ≤ $2 | contest auth eval; CUDA scorer replay; archive parity |
| **cpu** | ≤ 12 h | $0 (free) | contest-CPU GHA Linux x86_64 replay; public-leaderboard pairing |

---

## 2. Canonical (provider, gpu) per class

| class | canonical provider | canonical GPU | rationale |
|---|---|---|---|
| **smoke** | `modal` | `T4` | $0.59/hr; 30-min ≤ $0.30; matches contest reference |
| **full** | `vastai` | `RTX_4090` | $0.25/hr; 4-5× faster than T4; no Modal 3600s hard-kill |
| **long_burn** | `lightning` | `A100` | operator subscription (sub/$0); long timeout window |
| **eval** | `modal` | `T4` | matches contest GitHub Actions CUDA reference |
| **cpu** | `github` | `ubuntu-latest` | free GHA minutes on public repo; matches contest CPU axis |

Per-class canonical is fixed by Decision 9; mutating the matrix requires a new council deliberation per CLAUDE.md "Design decisions — non-negotiable".

---

## 3. Fallback providers per class

Time-Traveler amendment: when the cost-band posterior anchors show the canonical provider is >25% more expensive than a fallback (matched confidence floor), the routing helper re-routes to the cheapest fallback.

| class | fallback provider | fallback GPU | trigger condition |
|---|---|---|---|
| **smoke** | `modal` | `A10G` | Modal T4 capacity contraction OR Modal A10G discount |
| **full** | `modal` | `A100` | Vast.ai RTX 4090 unavailable / >25% more expensive |
| **long_burn** | `vastai` | `H100` | race-mode urgency (Lightning A100 saturated) |
| **eval** | `modal` | `A10G` | Modal T4 capacity contraction |
| **cpu** | (none) | (none) | GHA is the only canonical free CPU surface |

---

## 4. Cost-per-dispatch (Decision 9 ledger reference table)

Mirrors the council ledger lines 426-440. Source: empirical anchors + provider rate cards as of 2026-05-14.

| Provider | GPU | $/hr | 4h full cost | wall-clock vs T4 |
|---|---|---:|---:|---|
| Modal | T4 | $0.59 | $2.36 | 1.0× |
| Vast.ai | RTX 4090 | $0.25 | $1.00 | 0.20–0.25× (4–5× faster) |
| Modal | A100 | $1.10 | $4.40 | 0.4× |
| Lightning | A100 | sub/$0 | $0 (subscription) | 0.4× |
| Vast.ai | H100 | $1.50–1.99 | $6–8 | 0.15–0.20× |

**Empirical posterior cross-check** (from `.omx/state/cost_band_posterior.jsonl` as of 2026-05-14):

| class | canonical | successful anchors | verdict |
|---|---|---:|---|
| smoke | modal/T4 | 5 | sufficient_anchors (HIGH posterior confidence) |
| full | vastai/RTX_4090 | 0 | no_anchors (first dispatch will seed) |
| long_burn | lightning/A100 | 0 | no_anchors |
| eval | modal/T4 | 0 | no_anchors (1 row matches but as `full` due to label) |
| cpu | github/ubuntu-latest | 0 | no_anchors (3 successful eval rows tagged `github/cpu`) |

The premise verifier `.omx/tmp/d9_provider_routing_premise_verifier.py` produces this table on demand; re-run it to refresh.

---

## 5. Recipe-side override priority (CLAUDE.md anti-fragmentation primitive)

The operator's explicit choice ALWAYS wins per CLAUDE.md "Subagent coherence-by-default" anti-fragmentation primitive. Auto-routing is opt-in:

```yaml
# Recipe with explicit operator choice (LEGACY DEFAULT, unchanged behavior).
# D9 routing is informational only; recipe.platform/recipe.gpu drive dispatch.
platform: modal
gpu: A100
dispatch_class: full

# Recipe opting INTO auto-routing (NEW with D9).
# The routing helper resolves to canonical Decision 9 provider+gpu for
# the declared dispatch_class.
platform: auto
dispatch_class: full        # -> resolves to vastai/RTX_4090

# Recipe with auto-routing AND class inference (NEW with D9).
# The routing helper classifies via dispatch_label OR cost_band.epochs OR
# wall-clock estimate.
platform: auto
dispatch_label: substrate_x_modal_a100_dispatch_20260514T010000Z__smoke__100ep
cost_band:
  epochs: 100               # -> classifies as smoke -> modal/T4
```

**The routing helper NEVER silently overrides an explicit `platform:` choice.** When the operator sets `platform: modal`, the routing decision is recorded under `recipe.raw["_d9_routing_decision"]` for forensic audit, but the dispatch goes to the explicit choice.

---

## 6. Time-Traveler amendment: dynamic re-routing

The routing helper consults the cost-band posterior at dispatch time. If the cheapest fallback is empirically >25% cheaper than the canonical (BOTH at "weak_posterior" or "empirical_posterior" confidence floor), the helper re-routes:

```python
from tac.cost_band_calibration import select_provider_for_class

decision = select_provider_for_class(
    "full",
    consult_posterior=True,
    epochs_for_posterior=3000,
)

if decision.re_routed:
    print(f"Re-routed: {decision.canonical_provider} -> {decision.provider}")
    print(f"Rationale: {decision.re_routing_rationale}")
```

The 25% threshold is the **`RE_ROUTING_TRIGGER_FRACTION`** constant in `tac.cost_band_calibration`. It is NOT user-tunable; mutation requires a new council decision per CLAUDE.md "Design decisions — non-negotiable".

The trigger only fires when BOTH canonical and fallback have empirical anchors at matched confidence floor. "hand_calibrated_fallback" stubs do NOT trigger re-routing — those are explicitly uncalibrated and not comparable across providers.

---

## 7. Reactivation criteria (Decision 9 ledger lines 435-440)

The Decision 9 verdict is reactivated for re-deliberation when:

- **Vast.ai RTX 4090 cost rises >$0.40/hr** (capacity contraction shifts the canonical for `full` class).
- **Modal T4 cost falls <$0.30/hr** (provider promo flips the canonical for `smoke` and `eval` classes).
- **Lightning subscription terminates** (zero-cost long_burn assumption invalidated).
- **A new provider class enters the council bench** (e.g. RunPod, Paperspace) with sustained empirical anchors.

---

## 8. Operator-routable checkpoints (Decision 9 ledger lines 437-440)

1. ✅ `tools/operator_authorize.py` honors per-class routing default — landed via `_maybe_apply_auto_routing` in same commit batch.
2. ✅ Autopilot v2 ranker can consult `select_provider_for_class` per dispatch — see hook #4 (cathedral autopilot dispatch hook) below.
3. ✅ Cost-band posterior updates after each dispatch — already canonical per Catalog #175 + #177.

---

## 9. Wire-in hooks (CLAUDE.md "Subagent coherence-by-default")

| hook | status | rationale |
|---|---|---|
| 1. Sensitivity-map contribution | N/A | routing is a cost/wall-clock optimization; does not change scorer output bytes |
| 2. Pareto constraint | YES | the routing helper IS the Pareto-frontier-on-{$/dispatch, wall-clock} solver per Dykstra co-lead's deliberation |
| 3. Bit-allocator hook | N/A | no per-tensor importance change |
| 4. Cathedral autopilot dispatch hook | YES | autopilot ranker calls `select_provider_for_class` directly to pre-compute the provider for every queued candidate |
| 5. Continual-learning posterior update | YES | every dispatch triggers `append_anchor` per Catalog #175 + #177; posterior feeds the next routing decision automatically |
| 6. Probe-disambiguator | N/A | no 2+ defensible interpretations; Decision 9 verdict is unanimous |

---

## 10. Live API surface (canonical Python imports)

```python
from tac.cost_band_calibration import (
    # Decision matrix
    DISPATCH_CLASSES,                       # ("smoke", "full", "long_burn", "eval", "cpu")
    CANONICAL_PROVIDER_PER_CLASS,           # dict[str, tuple[str, str]]
    FALLBACK_PROVIDERS_PER_CLASS,           # dict[str, list[tuple[str, str]]]
    PER_CLASS_SOFT_COST_CEILING_USD,        # dict[str, float]
    PER_CLASS_SOFT_WALLCLOCK_CEILING_HR,    # dict[str, float]
    RE_ROUTING_TRIGGER_FRACTION,            # 0.25

    # Routing helpers
    classify_dispatch,                      # (label, epochs, wallclock, cost) -> class
    select_provider_for_class,              # (class) -> ProviderRoutingDecision
    select_provider_for_recipe,             # (recipe_meta dict) -> ProviderRoutingDecision

    # Decision dataclass
    ProviderRoutingDecision,                # frozen dataclass (provider, gpu, rationale, ...)
)
```

---

## 11. Test coverage

- `src/tac/tests/test_d9_per_class_provider_routing.py` — 37 dedicated tests (decision matrix invariants, classify_dispatch, select_provider_for_class canonical decisions, recipe-side override pass-through, posterior consumption, Time-Traveler re-routing, ProviderRoutingDecision dataclass surface, live-repo regression guards).
- `src/tac/tests/test_d9_operator_authorize_routing_integration.py` — 13 integration tests (`_resolve_routing_decision`, `_maybe_apply_auto_routing`, backward compat for existing recipes, `_run_dispatch` wire-in, recipe explicit-override priority).

Total: **50 dedicated tests passing**; 121 sister regression tests passing (`test_cost_band_calibration.py`, `test_check_175`, `test_check_177`, `test_check_162`, `test_check_198`, `test_check_202`, `test_operator_authorize_canonical_tool`, `test_operator_authorize_dispatch_gates`).

---

## 12. Defensive reproducer

`.omx/tmp/d9_provider_routing_premise_verifier.py` audits the live cost-band posterior for per-class anchor density and emits the Decision 9 verdict-vs-empirical table. Run with `.venv/bin/python .omx/tmp/d9_provider_routing_premise_verifier.py`.

---

## 13. Cross-references

- CLAUDE.md "GPU budget and compute resources — non-negotiable" (the price/performance hierarchy that anchors Decision 9).
- CLAUDE.md "Subagent coherence-by-default" (anti-fragmentation: operator's explicit choice always wins).
- CLAUDE.md "Forbidden score claims" (routing decision is informational; never changes archive bytes).
- Catalog #175 (`check_cost_band_anchor_writers_declare_outcome`) — write-side discipline that feeds the posterior this routing helper consumes.
- Catalog #177 (`check_cost_band_posterior_rows_have_outcome_field`) — read-side discipline.
- Catalog #199 (`check_operator_authorize_bypass_requires_session_budget`) — operator-authorize bypass discipline; this routing helper does NOT bypass any gate.
- Catalog #202 (`check_catalog_202_bypass_requires_paired_env_attestation`) — sister bypass gate.
- `feedback_all_design_decisions_through_grand_council_standing_referral_20260514.md` — the standing referral that made the omnibus deliberation binding.
