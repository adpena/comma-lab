# Substrate META layer — design 2026-05-15

**Operator directive verbatim 2026-05-15:** *"build a META layer that defines a
schema/contract substrates must respect, then makes wire-in AUTOMATIC across
the 6 canonical hooks per Catalog #125."*

**Lane:** `lane_meta_layer_substrate_contract_auto_wire_20260515` (L1 at landing).

**Status:** DESIGN-AND-NEW-MODULES landing wave; 31 legacy substrate trainers
remain on the legacy pattern and will migrate sequentially in a follow-up wave.
Per CLAUDE.md "Forbidden premature KILL" they are TAGGED-PENDING-MIGRATION,
not killed.

**Premise verifier evidence:** `.omx/tmp/meta_layer_premise_verifier.txt`
(`.omx/tmp/meta_layer_premise_verifier.py`, all 6 premises CONFIRMED 2026-05-15).

**Catalog #s claimed:** #241 (`check_substrate_uses_register_decorator_or_explicitly_legacy_tagged`,
WARN-ONLY initially), #242 (`check_register_substrate_contract_fields_canonical`,
STRICT-from-byte-one for any file that uses the decorator).

---

## 1. Empirical anchor + bug class

The Z3 v2 finding 2026-05-15 surfaced a silent-drift across THREE independent
substrate surfaces:

| Surface | Stated state | Actual state |
| --- | --- | --- |
| Substrate code (`e54901d60`) | Saved 4842 B via unit test | TRUE |
| Recipe YAML | `smoke_only: true / research_only: true` | TRUE |
| Modal smoke verdict | V1 path (additive +838 B) | NOT V2 latent-replacement (-4842 B) |

The unit test passed. The recipe was honest (smoke_only). The Modal verdict
landed on the V1 codepath because the dispatch wiring routed to the
additive-bolt-on path, not the latent-replacement path. Three independent
surfaces drifted because **the surfaces do not share a single source of
truth**.

The 6 Catalog #125 mandatory wire-in hooks (sensitivity-map / Pareto / bit-
allocator / cathedral autopilot dispatch / continual-learning posterior /
probe-disambiguator) are currently a **process discipline** — every landing
memo declares them by hand. The META layer flips them to a **structural
mechanism**: the contract is the source of truth, every consumer reads from
the same substrate registry, decoration-time validation refuses to import a
substrate whose contract is internally inconsistent.

---

## 2. The `SubstrateContract` Pydantic-style schema

Single source of truth declared at the substrate's module top level, captured
by the `@register_substrate(...)` decorator at import time.

### 2.1 Identity & lifecycle (5 fields)

| Field | Type | Notes |
| --- | --- | --- |
| `id` | `str` | Canonical substrate id, e.g. `"c6_e4_mdl_ibps"`. Must match the trainer filename suffix. |
| `lane_id` | `str` | `lane_<id>_<YYYYMMDD>` per `tools/lane_maturity.py`. |
| `target_modes` | `list[Literal["contest_one_video_replay","contest_generalized","production_generalized","production_edge_adaptive","research_substrate"]]` | Per CLAUDE.md "Contest vs production target modes — non-negotiable". At least one. |
| `deployment_target` | `Literal["t4_contest_runtime","comma_ai_production","openpilot_edge","desktop_research","device_learning_optional"]` | Per the same section. |
| `council_verdict_provenance` | `str \| None` | Filename relative to repo root of the council deliberation memo (per CLAUDE.md "Design decisions — non-negotiable"); may be `None` for substrate-engineering scaffolds. |

### 2.2 Architecture & runtime contract (8 fields per Catalog #124)

| Field | Type | Notes |
| --- | --- | --- |
| `archive_grammar` | `str` | E.g. `"monolithic_ibps1_self_contained"`, `"hnerv_packet"`, `"d1poly1"`. |
| `parser_section_manifest` | `dict[str, str]` | Section id → role token consumed by `tac.analysis.hnerv_packet_sections`. |
| `inflate_runtime_loc_budget` | `int` | ≤100 default; ≤200 with explicit waiver per HNeRV parity discipline lesson 4. |
| `runtime_dep_closure` | `list[str]` | E.g. `["torch>=2.5,<2.7","brotli","constriction"]`. |
| `export_format` | `Literal["fp4_brotli","fp16_brotli","int8_arith","int4_lsq","fp4_packed_sorted_keys_ibps1","custom"]` | Per HNeRV parity lesson 2 + Catalog #102. |
| `score_aware_loss` | `Literal["scorer_loss_terms_btchw","kl_distill","custom"]` | Must be the canonical helper unless waived. |
| `bolt_on_loc_budget` | `int` | ≤350 default per HNeRV parity lesson 7; substrate_engineering may exceed. |
| `no_op_detector_planned` | `bool` | Per Catalog #105 / #139. |

### 2.3 Operational mechanism (Catalog #220) — 3 fields

| Field | Type | Notes |
| --- | --- | --- |
| `archive_bytes_added` | `str \| None` | Human-readable, e.g. `"~43 KB (full) or ~2.7 KB (shrunk)"`. May be `None` when net bytes are negative or zero. |
| `score_improvement_mechanism_status` | `Literal["OPERATIONAL","PRE_BUILD_SUBSTRATE_ENGINEERING","RESEARCH_ONLY","SCAFFOLD_DEFERRED_INTEGRATION"]` | Catalog #220 acceptance signal. |
| `runtime_overlay_consumed` | `bool` | Companion to status. `True` only when status is `OPERATIONAL`. |

### 2.4 Recipe schema (8 fields — auto-generated YAML)

Maps 1:1 to `.omx/operator_authorize_recipes/substrate_<id>_modal_<gpu>_dispatch.yaml`.

| Field | Type | Catalog ref |
| --- | --- | --- |
| `recipe_smoke_only` | `bool` | Truthful flag. |
| `recipe_research_only` | `bool` | Truthful flag. |
| `recipe_min_smoke_gpu` | `Literal["T4","L4","A10G","L40S","A100","H100"]` | Catalog #215. |
| `recipe_min_vram_gb` | `int` | Catalog #170. |
| `recipe_pyav_decode_strategy` | `Literal["cpu_thread_async_upload","cuda_nvdec","cpu_blocking_upload","not_applicable"]` | Catalog #181. |
| `recipe_canary_status` | `Literal["canary","post_canary_dependent","independent_substrate"]` | Catalog #173. |
| `recipe_video_input_strategy` | `Literal["per_dispatch_local_copy","readonly_mmap","shared_volume_no_contention_expected"]` | Catalog #171. |
| `recipe_canary_dependency` | `str \| None` | Required if `recipe_canary_status == "post_canary_dependent"`. |

### 2.5 Cost band & GPU envelope (4 fields)

| Field | Type | Notes |
| --- | --- | --- |
| `cost_band_epochs` | `int` | E.g. `200`. |
| `cost_band_gpu_key` | `Literal["T4","L4","A10G","L40S","A100","H100"]` | Per Catalog #175 / #177 cost-band posterior. |
| `cost_band_platform_key` | `Literal["modal","vastai","lightning","kaggle"]` | Per the same posterior. |
| `cost_band_p50_usd` | `float` | Hand-calibrated fallback; the cost-band posterior overrides at runtime. |

### 2.6 The 6 Catalog #125 wire-in hooks (6 declared fields)

| Field | Type | Notes |
| --- | --- | --- |
| `hook_sensitivity_contribution` | `Literal["scorer_conditional_entropy_map_v1","mdl_density_v1","custom","not_applicable_with_rationale"]` | Hook #1. If `not_applicable_with_rationale`, `hook_not_applicable_rationale` MUST be set. |
| `hook_pareto_constraint` | `Literal["rate_distortion_v1","cost_band_envelope_v1","custom","not_applicable_with_rationale"]` | Hook #2. |
| `hook_bit_allocator_class` | `Literal["per_tensor_uniform","per_channel_lsq","ibps_kkt","not_applicable_with_rationale"]` | Hook #3. |
| `hook_autopilot_ranker_class_shift_token` | `str \| None` | Hook #4. Token consumed by `tools/cathedral_autopilot_autonomous_loop.py::_CLASS_SHIFT_LITERATURE_TOKENS` (e.g. `"MDL-IBPS"`, `"Wyner-Ziv"`). `None` for within-class substrates. |
| `hook_continual_learning_anchor_kind` | `Literal["paired_axis","cuda_only","cpu_only","macos_cpu_advisory","not_applicable_with_rationale"]` | Hook #5. |
| `hook_probe_disambiguator` | `str \| None` | Hook #6. Path to `tools/probe_<track>_disambiguator.py` if applicable; else `None` with rationale field set. |

A separate `hook_not_applicable_rationale: dict[str, str]` field collects per-
hook rationales for any `not_applicable_with_rationale` value. Bare `not_applicable`
without a rationale is REJECTED at validation time.

### 2.7 Compliance declarations (1 list field)

| Field | Type | Notes |
| --- | --- | --- |
| `catalog_compliance_declarations` | `list[str]` | E.g. `["catalog_166_source_parity_honored","catalog_205_select_inflate_device_used","catalog_226_gate_auth_eval_call_used","catalog_146_3arg_archive_grammar_honored","catalog_164_scorer_preprocess_input_called"]`. |

The validator cross-references against a known set of catalog-compliance tokens
(stored at `_KNOWN_CATALOG_COMPLIANCE_TOKENS` in `contract.py`). Unknown tokens
warn but do not refuse, so substrates can declare new catalog hooks as they
land.

### 2.8 Field totals

5 (identity) + 8 (architecture) + 3 (operational) + 8 (recipe) + 4 (cost band)
+ 6 (hooks) + 1 (compliance) + 1 (rationale dict) = **36 canonical fields**.

---

## 3. The `@register_substrate(...)` decorator

```python
from tac.substrate_registry import register_substrate, SubstrateContract

@register_substrate(
    SubstrateContract(
        id="example_template",
        lane_id="lane_example_template_20260515",
        target_modes=["research_substrate"],
        deployment_target="desktop_research",
        # ... all 36 fields ...
    )
)
def main(argv: list[str] | None = None) -> int:
    ...
```

**Decoration-time semantics:**

1. **Validate** the contract against `SubstrateContract`'s Pydantic-style
   validators (raises `SubstrateContractError` on failure; the import then
   fails loud at module-load time per CLAUDE.md "fail-fast at every boundary").
2. **Refuse** internal inconsistencies:
   - `recipe_smoke_only=True` AND `cost_band_epochs > 100` → ambiguous.
   - `hook_*=not_applicable_with_rationale` without rationale → reject.
   - `recipe_canary_status="post_canary_dependent"` without
     `recipe_canary_dependency` → reject.
   - `score_improvement_mechanism_status="OPERATIONAL"` AND
     `runtime_overlay_consumed=False` → reject (the two MUST agree).
   - `archive_bytes_added` references >1 KB AND status is not `OPERATIONAL` /
     `RESEARCH_ONLY` / `PRE_BUILD_SUBSTRATE_ENGINEERING` → reject (mirror of
     Catalog #220).
3. **Register** the substrate into the in-memory `_REGISTERED_SUBSTRATES` dict
   (keyed by `id`); duplicate ids raise `SubstrateContractError`.
4. **Return** the original `main` callable unmodified (decorator is a pass-
   through; it never wraps the callable, only registers metadata).

The decoration-time validation IS the probe-disambiguator (hook #6) for the
META layer itself: any future surface that drifts will be caught at import
time — the substrate cannot be loaded until all four surfaces (code, recipe,
cost-band, runtime) agree on the contract.

---

## 4. Auto-wire mechanism

### 4.1 Reading consumers (one-way data flow)

The 6 hooks' canonical consumer modules are **READ FROM**, never WRITTEN TO,
by the META layer (per the sister-subagent ownership map; the
WIRE-AND-INTEGRATE-ALL subagent owns those write paths).

```
SubstrateContract (single source of truth)
        │
        ├──→ tac.sensitivity_map  (hook #1)        # READ
        ├──→ tac.cost_band_calibration  (hook #2)  # READ
        ├──→ tac.sensitivity_map  (hook #3)        # READ
        ├──→ cathedral_autopilot  (hook #4)        # READ
        ├──→ tac.continual_learning  (hook #5)     # READ
        └──→ tools/probe_*  (hook #6)              # READ
```

The auto-wire helpers in `auto_wire.py` expose:

- `query_substrates_for_sensitivity_hook() -> list[SubstrateContract]`
- `query_substrates_for_pareto_hook() -> list[SubstrateContract]`
- `query_substrates_for_bit_allocator_hook() -> list[SubstrateContract]`
- `query_substrates_for_autopilot_ranker() -> list[SubstrateContract]`
- `query_substrates_for_continual_learning_anchor_kind() -> dict[str, str]`
- `query_substrates_for_probe_disambiguators() -> dict[str, str]`

Existing consumer modules (sensitivity_map / cost_band / autopilot / continual
learning) can OPT IN to read from the META registry by importing these helpers.
The sister WIRE-AND-INTEGRATE-ALL subagent owns the actual integration; the
META layer just exposes the read API.

### 4.2 Generators (one-way artifact emission)

The contract drives auto-generation of per-substrate downstream artifacts:

```
SubstrateContract (single source of truth)
        │
        ├──→ recipe_generator  → .omx/operator_authorize_recipes/substrate_<id>_modal_<gpu>_dispatch.yaml
        ├──→ driver_generator  → scripts/remote_lane_substrate_<id>.sh
        └──→ (future) lane_registry_generator → .omx/state/lane_registry.json entry
        └──→ (future) e2e_pytest_fixture_generator → src/tac/tests/test_substrate_<id>_contract_fixture.py
```

The generators are PURE FUNCTIONS: input contract → output artifact bytes.
They do NOT touch in-flight substrates or sister-subagent surfaces; they emit
to NEW paths (or a `dry_run=True` mode that returns the bytes for diff
review).

The migration wave (follow-up subagent) will:

1. Run the legacy substrate trainer through a contract-extraction analysis.
2. Generate the canonical recipe + driver via the generators.
3. Diff against the existing recipe / driver — any drift surfaces as a
   migration finding.
4. Land the contract declaration + the regenerated artifacts in one commit.

---

## 5. Failure modes & STRICT preflight gates

### 5.1 Catalog #241 — `check_substrate_uses_register_decorator_or_explicitly_legacy_tagged`

**WARN-ONLY initially** (31 legacy substrates exist). Refuses
`experiments/train_substrate_*.py` files that neither use `@register_substrate`
NOR carry a same-line `# LEGACY_SUBSTRATE_PRE_META_LAYER:<rationale>` waiver
on the file's first 20 lines.

**Strict-flip plan:** after the migration wave completes (lands a contract
on every active substrate), flip to STRICT in the same commit batch per
CLAUDE.md "Strict-flip atomicity rule". The placeholder `<rationale>` literal
is rejected so the gate cannot self-waive.

### 5.2 Catalog #242 — `check_register_substrate_contract_fields_canonical`

**STRICT-from-byte-one** for any file that uses `@register_substrate`. If a
substrate file uses the decorator, the contract MUST validate against the
Pydantic schema (all required fields present, all enum values legal, all
internal-consistency rules satisfied). Live count at landing: 0 (only the
example_template uses the decorator and it's contract-clean). Strict-flip
atomicity: the gate ships strict because the ONLY users of the decorator
are net-new files; legacy substrates are exempted via Catalog #241.

### 5.3 Internal validators (raised at import time, not by preflight)

The following are decoration-time errors (raise `SubstrateContractError`):

- Duplicate `id` in `_REGISTERED_SUBSTRATES`.
- Required field missing.
- Enum value not in the legal set.
- `recipe_canary_status="post_canary_dependent"` with no `recipe_canary_dependency`.
- `score_improvement_mechanism_status="OPERATIONAL"` with `runtime_overlay_consumed=False`.
- `archive_bytes_added` declares >1 KB AND status is `SCAFFOLD_DEFERRED_INTEGRATION`.
- Hook value `not_applicable_with_rationale` without entry in `hook_not_applicable_rationale`.

---

## 6. Migration plan (NOT executed in this wave)

Follow-up subagent `MIGRATE-LEGACY-SUBSTRATES-TO-META-LAYER-SUBAGENT` (lane
`lane_migrate_legacy_substrates_to_meta_layer_<YYYYMMDD>`) sequentially
migrates the 31 legacy substrate trainers:

1. **Audit pass** (≤1h): for each trainer, extract the implied contract from
   the existing recipe + driver + TIER_N manifest + lane registry entry.
2. **Generate-and-diff pass** (≤2h): emit the canonical recipe + driver via
   the generators, diff against the existing artifacts; surface any drift
   as a per-substrate finding.
3. **Land pass** (per substrate, ≤30 min): add `@register_substrate(...)` at
   the trainer top level, regenerate the artifacts, land in one commit per
   substrate via the canonical serializer with `--expected-content-sha256`.
4. **Strict-flip Catalog #241** (final commit): once live count = 0, flip
   to STRICT in the same commit batch per the atomicity rule.

The migration is sequential, not parallel, to avoid Catalog #230 collisions
on the substrate trainer files.

---

## 7. Beauty + simplicity invariants (per CLAUDE.md non-negotiable)

- **Narrow API:** only `register_substrate`, `SubstrateContract`,
  `get_registered_substrates`, `validate_all_registered`, plus the 6 query
  helpers + 2 generators. No catch-all "substrate manager".
- **Reviewable in 30 seconds:** `example_template.py` declares all 36 fields
  in a single dataclass-ish block; the decorator call is a single
  expression.
- **Explicit failure modes:** `SubstrateContractError` with field-level error
  messages; no silent defaults beyond the documented enum carve-outs.
- **Composable contracts:** the decorator is a pass-through; substrates can
  still be tested in isolation (the registration is metadata-only, no
  runtime side effect on the trainer's behavior).
- **One-way data flow:** the contract is the single source of truth; consumers
  read from the registry; generators emit artifacts. No bidirectional
  surfaces.

---

## 8. 6-hook wire-in declaration for THIS landing (Catalog #125)

Per CLAUDE.md "Subagent coherence-by-default":

1. **Sensitivity-map contribution** — N/A this landing, the META layer is the
   STRUCTURAL ENFORCEMENT of all 6 hooks. Existing sensitivity-map consumer
   `tac.sensitivity_map` will read via `query_substrates_for_sensitivity_hook()`
   in the WIRE-AND-INTEGRATE-ALL follow-up.
2. **Pareto constraint** — N/A this landing, ditto via
   `query_substrates_for_pareto_hook()`.
3. **Bit-allocator hook** — N/A this landing, ditto via
   `query_substrates_for_bit_allocator_hook()`.
4. **Cathedral autopilot dispatch hook** — ACTIVE this landing. Future
   substrates registered via the decorator can declare
   `hook_autopilot_ranker_class_shift_token`; the autopilot ranker will
   pick this up via `query_substrates_for_autopilot_ranker()` (consumer wiring
   pending).
5. **Continual-learning posterior update** — N/A this landing, ditto via
   `query_substrates_for_continual_learning_anchor_kind()`.
6. **Probe-disambiguator** — ACTIVE this landing. Decoration-time validation
   IS a probe-disambiguator: any substrate whose contract is internally
   inconsistent fails to import. Per-substrate probes still go in
   `tools/probe_<track>_disambiguator.py` and are referenced via the
   `hook_probe_disambiguator` field.

---

## 9. Operator-routable decisions (5)

1. **CRITICAL** — Approve the migration wave (sequential subagent or operator-
   orchestrated). The META layer is structurally complete after this landing
   but provides ZERO empirical protection until at least one legacy substrate
   migrates.
2. **HIGH** — After migration, strict-flip Catalog #241 in the SAME commit
   batch per the atomicity rule. Currently warn-only.
3. **MEDIUM** — Decide whether to add a `lane_registry_generator` that emits
   the registry entry from the contract (would eliminate the
   manual `tools/lane_maturity.py add-lane / mark` step for new substrates).
4. **MEDIUM** — Decide whether the `hook_not_applicable_rationale` entries
   should be cross-checked against a list of canonical rationales (e.g., to
   prevent `"N/A"` placeholders from passing); currently free-text.
5. **LOW** — Consider extending the META layer to non-substrate dispatch
   surfaces (eval lanes, packet-compiler lanes, scoring lanes) once the
   substrate migration is proven.

---

## 10. Cross-references

- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" (8-field
  declaration, 13 inviolable lessons).
- CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" (Catalog #220).
- CLAUDE.md "Subagent coherence-by-default" (the 6-hook wire-in non-negotiable).
- Catalog #124 (`check_representation_lane_has_archive_grammar_at_design_time` —
  the design-time companion gate for representation lanes).
- Catalog #151 (`check_operator_wrapper_threads_trainer_tier_required_flags` —
  the env→CLI ladder gate; the META layer's recipe_generator emits compliant
  recipes).
- Catalog #166 (`check_modal_dispatch_verifies_worker_source_matches_head` —
  the source-parity gate; the META layer's driver_generator emits compliant
  drivers).
- Catalog #205 (`check_inflate_py_uses_canonical_select_inflate_device` — the
  inflate-device-fork gate; META compliance declarations declare honored).
- Catalog #220 (`check_substrate_l1_scaffold_no_byte_addition_without_operational_score_improvement_mechanism` —
  the runtime-effect gate; META operational-mechanism fields encode it).
- Catalog #226 (`check_trainer_auth_eval_uses_canonical_helper` — the auth-
  eval canonical helper gate; META compliance declarations declare honored).
- Catalog #229 (`check_subagent_landing_includes_premise_verification_evidence` —
  this landing satisfies via `.omx/tmp/meta_layer_premise_verifier.py`).
- Catalog #230 (`check_bulk_rewrite_respects_sister_subagent_ownership_map` —
  this landing honors the sister-subagent ownership map).
