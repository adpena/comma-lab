# One-arg local-MPS vs Modal dispatch switch — design memo

**Date:** 2026-05-17
**Lane:** `lane_one_arg_local_mps_vs_modal_dispatch_switch_20260517`
**Status:** L1 — implementation in flight
**horizon-class:** rigor_overhead (apparatus build; non-frontier; mission-alignment subsection per Catalog #291)

---

## Operator directive (verbatim)

> *"Deploying to local MPS versus modal should be super easy to configure, like one arg in a func"*
>
> *"Do everything possible you can to accelerate dev velocity and save money using local MPS"*

---

## Summary

Make local Apple-silicon **MPS** and **CPU** dispatch targets first-class peers
alongside `modal` / `lightning` / `vastai` in the `tools/operator_authorize.py`
flow, accessible via a SINGLE `--target` CLI argument (CLI > recipe precedence).
The legitimate non-authoritative uses (proxy curve discovery, smoke gates,
TTO sweeps, code-correctness checks) become 1-flag operations the operator can
fire without writing/editing recipes.

The build must NEVER let MPS or local-CPU results pollute the canonical
`[contest-CPU]` / `[contest-CUDA]` posterior. The contract — `evidence_grade`
auto-stamping, posterior isolation, MPS-availability check, loud banner — is
the structural protection.

---

## Architecture

```
operator-authorize --recipe <name> --target {auto|modal|vastai|local|local-mps|local-cpu}
   │
   ├─ recipe loaded; CLI --target overrides recipe `platform:` if provided
   ├─ Recipe.platform property resolves the effective platform
   │
   ├─ _maybe_apply_auto_routing(recipe)   ← Decision 9 routing (existing)
   ├─ _build_env_overrides(recipe, instance_job_id)
   │
   └─ _run_dispatch(recipe, ...)
         │
         platform-keyed fork:
         ┌───────────────────────────────────────────────────────────┐
         │ "modal"        → _dispatch_modal()        (EXISTING)       │
         │ "vastai"       → _dispatch_vastai()       (EXISTING)       │
         │ "local"        → _dispatch_local()        (EXISTING)       │
         │ "local_mps"    → _dispatch_local_mps()    (NEW)            │
         │ "local_cpu"    → _dispatch_local_cpu()    (NEW)            │
         │ "none/lightning/kaggle/gha/azure" → _dispatch_noop()       │
         └───────────────────────────────────────────────────────────┘
```

The `--target` CLI argument applies a simple ALIAS map BEFORE recipe property
resolution:

```
local-mps  → local_mps  (recipe.raw["platform"] = "local_mps")
local-cpu  → local_cpu  (recipe.raw["platform"] = "local_cpu")
modal      → modal      (overrides recipe `platform:`)
local      → local
vastai     → vastai
auto       → (no override; recipe.raw["platform"] = "auto"; _maybe_apply_auto_routing handles)
```

The override is recorded in a `cli_target_override` recipe field for forensic
clarity and surfaces in the dispatch banner.

---

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" (2026-05-15) the
default question is *"What's the OPTIMAL ENGINEERING for THIS method to achieve
the lowest score possible?"*. For this apparatus build the "lowest score" is
SHARP DEV-VELOCITY at $0 cost while STRUCTURALLY PRESERVING the canonical
score-discipline; we therefore weigh canonical-share heavily because the goal
IS coherent integration with the rest of the dispatch flow.

| Layer | Decision | Rationale |
|---|---|---|
| **Recipe schema (`platform:`)** | **ADOPT** canonical (string field; resolved via `_resolve_env_var`) | Single source of truth; adding new enum values does not fork the schema. |
| **CLI parser (`--target`)** | **FORK** (new flag; recipe-override semantics) | The "one-arg toggle" IS the operator's stated requirement; this is unique to local-MPS/local-CPU. |
| **Recipe.platform property** | **ADOPT** canonical lowercase normalization | Already case-insensitive; new values fit unchanged. |
| **`LEGAL_NATIVE_PLATFORMS` enum** | **EXTEND** to include `local_mps` + `local_cpu` | Single canonical set per `tac.deploy.dispatch_protocol`; sister Catalog #270 scope-fix already exercised this extension pattern. |
| **`_platform_has_native_dispatch`** | **EXTEND** to include `local_mps` + `local_cpu` | Yes — they ARE native (run on the operator's machine). |
| **`_run_dispatch` platform fork** | **EXTEND** with 2 new branches | The fork is the canonical dispatch surface; new platforms = new branch. |
| **`_dispatch_local_mps` / `_dispatch_local_cpu`** | **FORK** (new functions; cannot reuse `_dispatch_local`) | The contract differs: hardware-availability check + evidence-grade auto-stamp + manifest-write target are NEW responsibilities not in the generic local shell-out. |
| **Banner ("LOCAL-MPS RESEARCH-SIGNAL — NON-AUTHORITATIVE")** | **FORK** | Different evidence semantics → different operator-facing copy. Critical for non-confusability with paid Modal dispatch. |
| **MPS hardware-availability check** | **FORK** | `torch.backends.mps.is_available()` is MPS-specific; no canonical helper today. |
| **`PYTORCH_ENABLE_MPS_FALLBACK=0` env injection** | **FORK** | New default for local_mps; no canonical pattern. |
| **Posterior-write target** | **ADOPT** existing `mps_research_signal` + `macos_cpu_advisory_signal` canonical helpers | Catalog #192 + #131 sister discipline; do NOT fork the manifest schema. |
| **Append-row helper (`append_manifest_row_to_jsonl`)** | **EXTEND** mps_research_signal.py to add the sister helper (mirroring macos_cpu_advisory_signal L477) | One canonical helper per axis; symmetric API between MPS and CPU advisory. |
| **`local_pre_deploy_check` skip-set** | **EXTEND** existing `_TOOL_DISPATCH_SKIPPED_CHECKS` pattern + add `_LOCAL_RESEARCH_SIGNAL_SKIPPED_CHECKS` sibling frozenset | Sister of Catalog #270 scope clarification; same precedent, new kind. |
| **`dispatch_kind: tool` vs `dispatch_kind: local_research_signal`** | **EXTEND** `LEGAL_DISPATCH_KINDS` from {substrate, tool} to add `local_research_signal` | Principled fork: local_research_signal has its own contract (forbids auth_eval AND archive emit; requires MPS-availability check). |
| **Catalog #240 recipe-vs-trainer-state consistency** | **ADOPT** existing strict gate | local_mps recipes MUST declare `research_only: true` per the canonical discipline; the existing gate enforces it. |
| **Catalog #1 `check_no_mps_fallback_default`** | **ADOPT** unchanged | This gate scans for the silent-fallback ternary in source; the new local_mps dispatch is EXPLICIT opt-in via `--target local-mps`, NOT a silent default — fully compatible. |
| **Catalog #192 `check_macos_cpu_advisory_not_promoted_without_linux_verification`** | **ADOPT** unchanged | Catalog #192's read-side gate on persisted advisory artifacts already protects the canonical posterior. |
| **New STRICT preflight gate** | **CONSIDER** Catalog #317 `check_local_research_signal_dispatches_stamp_evidence_grade` | Source-text scan of `_dispatch_local_mps` + `_dispatch_local_cpu` requiring the canonical stamp tokens be present. Same META-meta defense-in-depth pattern as Catalog #279/#280/#283. |

---

## 9-dimension success checklist evidence

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" §"9-dim checklist"
+ Catalog #294 (this section is the structural evidence carrier):

1. **UNIQUENESS** — extension is the FIRST instance of `dispatch_kind=local_research_signal`. Class-shift from "substrate-trainer" / "tool-extraction" to "research-signal-generation"; not within-class refinement of existing dispatch types.

2. **BEAUTY + ELEGANCE** — Goal: PR101-style 30-sec-reviewable. Estimated LOC budget: ~300 LOC across `operator_authorize.py` (3 new functions + 1 CLI flag + 2 platform forks) + ~120 LOC `mps_research_signal.py` (1 sister helper) + ~60 LOC `local_pre_deploy_check.py` (1 sister frozenset + skip in main loop) + ~30 LOC `dispatch_protocol.py` (enum extension). Total ~510 LOC of code reviewable in <30 sec per file.

3. **DISTINCTNESS** — explicitly different from sister dispatch types:
   - vs `modal`: local (no provider cost) + non-authoritative
   - vs `local`: enforces evidence-grade contract + hardware-availability check + manifest write
   - vs `tool`: forbids archive emit + requires research_only=true

4. **RIGOR** — premise verification per Catalog #229 (5 PVs verified pre-edit; see `.omx/tmp/one_arg_local_mps_dispatch_switch_premise_verifier.txt`); adversarial review WITHIN this design memo via the cargo-cult audit section below; assumption classification HARD-EARNED-vs-CARGO-CULTED; empirical anchor = PR107 macOS-CPU calibration (|Δ| ≤ 6e-6 vs GHA Linux x86_64 per CLAUDE.md).

5. **OPTIMIZATION PER TECHNIQUE** — per the canonical-vs-unique table above: shared helpers where they serve (manifest schema / preflight pattern / Recipe class / CLI parser scaffold); forked where the substrate-optimal engineering differs (hardware-availability check, banner copy, env overrides, dispatch_kind contract).

6. **STACK-OF-STACKS-COMPOSABILITY** — orthogonal axes: this build composes with (a) Catalog #270 dispatch-protocol scope-fix (we extend the same `LEGAL_DISPATCH_KINDS` enum); (b) Catalog #192 macos_cpu_advisory posterior isolation (the local_cpu branch routes through `tac.optimization.macos_cpu_advisory_signal.append_manifest_row_to_jsonl`); (c) Catalog #1 MPS-fallback-default guard (we are explicit opt-in, not silent default).

7. **DETERMINISTIC REPRODUCIBILITY** — manifest writes use `sort_keys=True` per existing helper contracts; the `run_id` field is a stable caller-supplied identifier; `instance_job_id` follows the existing dispatch-claim convention. Pre-existing `cost_band_posterior` etc. are NOT touched.

8. **EXTREME OPTIMIZATION + PERFORMANCE** — `$0` provider cost (entire mission); single in-process subprocess.call for the trainer; no Modal mount upload + retry loop; no `git push` ceremony; no contest-CUDA waiting. Operator can iterate in seconds at $0 marginal cost.

9. **OPTIMAL MINIMAL CONTEST SCORE** — this build is `horizon_class: rigor_overhead` — it does NOT directly lower the contest score. Its value to score-lowering is INDIRECT: by making local research-signal dispatch a 1-flag operation, the operator can sweep curve shapes / smoke gates / TTO trajectories 10-100× faster, identifying which CUDA candidates are worth the $5-15 paid Modal dispatch BEFORE spending. Cost-band ROI from the existing canonical autopilot ranker is preserved (no posterior pollution).

---

## Cargo-cult audit per assumption

Per CLAUDE.md "Cargo-cult audit per assumption" non-negotiable (Catalog #303) +
the hard-earned-vs-cargo-culted addendum (Catalog #292 sister discipline).

| # | Assumption | Classification | Rationale / Unwind path |
|---|---|---|---|
| 1 | MPS dispatch CAN be made structurally indistinguishable from Modal dispatch at the wrapper layer (operator types one different flag and everything else is gated by metadata) | **HARD-EARNED** | The operator-authorize architecture is already platform-polymorphic at L1860-1880; adding 2 branches is the existing precedent (see L1872 noop list). Verified PV-1. |
| 2 | The MPS path can ALWAYS be made fail-closed against `[contest-CPU]`/`[contest-CUDA]` posterior writes by routing through `mps_research_signal.build_*_manifest` which permanently stamps `score_claim=False` | **HARD-EARNED** | Verified PV-2 in source — the manifest schema is structurally fail-closed; rows cannot carry `score_claim=True` per L513. The MPS module is the canonical fail-closed surface; rerouting writes through it is structurally safe. |
| 3 | `torch.backends.mps.is_available()` is a reliable runtime indicator that MPS dispatch will succeed | **PARTIALLY-CARGO-CULTED → mitigated** | The function returns True on Apple Silicon even when MPS ops are partial or fall back; we mitigate by ALSO setting `PYTORCH_ENABLE_MPS_FALLBACK=0` so MPS-unavailable ops raise (Catalog #1 sister discipline). |
| 4 | The 30-sec-reviewable budget (~510 LOC) is achievable with adequate test coverage | **TO-VERIFY at landing** | The estimate is based on counting existing `_dispatch_local` (~25 LOC) + `_dispatch_modal` (~115 LOC) and forecasting that local_mps + local_cpu sit between them in complexity. Test target ~40 tests should fit ~500 LOC. |
| 5 | Operators want a SINGLE `--target` CLI flag rather than separate `--platform-override` and `--mps` and `--local-cpu` flags | **HARD-EARNED** | Operator directive verbatim: *"like one arg in a func"*. One flag is the explicit ask. |
| 6 | The `local_research_signal` dispatch_kind cleanly extends the existing `{substrate, tool}` enum without breaking sister gates | **HARD-EARNED-after-test** | The 2-element enum is referenced by Catalogs #270 (scope clarification) and the dispatch-protocol's per-tier reports. Adding a third member must include a test that all existing sister tests still pass with the broader enum. PART 4 covers this. |
| 7 | A new STRICT preflight gate (Catalog #317) is necessary self-protection per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" | **HARD-EARNED** | Without the gate, a future refactor could silently remove the `score_claim=False` auto-stamp from `_dispatch_local_mps` and there is no structural detector. Same META-meta defense-in-depth pattern that Catalog #279/#280/#283 solved for the dispatch-optimization-protocol fail-open class. |
| 8 | Catalog #1 (`check_no_mps_fallback_default`) is NOT triggered by the new `--target local-mps` opt-in path | **HARD-EARNED** | Catalog #1 scans for the silent-fallback ternary `device = "cuda" if ... else "mps" else "cpu"` in source. The new code path is `if args.target == "local-mps": ...` which is EXPLICIT opt-in, not a silent fallback. Verified in part 1 of premise verification. |
| 9 | The empirical anchor for safety is PR107 M5 Max ≈ GHA Linux x86_64 to within 6e-6 — extending this to MPS-research-signal use is safe | **CARGO-CULTED at edge → mitigated** | PR107 calibration is for macOS-CPU, not MPS. MPS drift can be 23x for PoseNet per the existing CLAUDE.md anchor; the calibration does NOT transfer. The build mitigates by stamping `EVIDENCE_GRADE="MPS-research-signal"` (NOT `macOS-CPU-advisory`) and PERMANENTLY refusing promotion. The fact that MPS drifts more is INTENTIONALLY surfaced via the loud banner; this is correct apparatus, not blindspot. |
| 10 | The operator-authorize CLI is a stable enough surface to add `--target` without breaking sister tooling | **HARD-EARNED** | The CLI is purely additive (new flag, default None = no override = current behavior). Every existing `--recipe X` invocation continues to work unchanged. |

**Net assessment:** 8/10 hard-earned, 1/10 partially-cargo-culted-mitigated (assumption 3), 1/10 cargo-culted-at-edge-mitigated (assumption 9), 1/10 to-verify-at-landing (assumption 4). No critical cargo-cult that blocks the build.

---

## Predicted ΔS band

**Predicted ΔS band: [0.000, 0.000]** (no direct score impact).

**Dykstra-feasibility check / first-principles citation:**

This is an INFRASTRUCTURE / DEV-VELOCITY build with no archive bytes added or
removed. The operator never dispatches a contest-axis result through the
local_mps or local_cpu paths — the manifest writes are STRUCTURALLY isolated
from the canonical `[contest-CPU]` / `[contest-CUDA]` posterior. Per CLAUDE.md
"Apples-to-apples evidence discipline":

> Per CLAUDE.md "Apples-to-apples evidence discipline": Every numeric score
> must be tagged by axis (`[contest-CPU]` / `[contest-CUDA]` / `[macOS-CPU
> advisory]` / `[MPS-PROXY]` / `[advisory only]`). Missing axis label is a bug.

The MPS-research-signal manifest writes its results with
`evidence_grade="MPS-research-signal"` PERMANENTLY (the canonical helper enforces
it; see `src/tac/optimization/mps_research_signal.py` L513). The macOS-CPU
advisory manifest writes with `evidence_grade="macOS-CPU-advisory"`. Neither
can be silently promoted to a contest axis (Catalog #127 + #192 + #1).

Probe-disambiguator: not applicable. The build has a deterministic outcome — it
either works (operator can dispatch local MPS via 1 arg + manifest writes to
correct file + canonical posterior is untouched) or it does not. There is no
parametric uncertainty in the dev-velocity outcome.

**Indirect score impact:** by enabling 10-100× faster local iteration on
proxy curve discovery, the operator can identify which CUDA candidates are
worth paid dispatch BEFORE spending. The expected indirect ΔS band is
**[-0.005, 0.000]** over the next ~10 sweep cycles (operator hypothesis;
not committed to the canonical posterior).

---

## Observability surface

Per CLAUDE.md "Max observability — non-negotiable" + Catalog #305.

The 6-facet observability declaration:

1. **Inspectable per layer:**
   - CLI: `--target` flag is structurally visible via `--help`
   - Recipe: `cli_target_override` field is written into the resolved recipe object's `raw` dict for the dispatch run
   - Dispatcher: each `_dispatch_local_mps` / `_dispatch_local_cpu` call emits a loud banner with platform + run_id + manifest output path
   - Trainer: subprocess return code + stdout/stderr captured
   - Manifest: the canonical `mps_research_signal_manifest.v1` (or `macos_cpu_advisory_signal_manifest.v1`) schema is fully inspectable JSON

2. **Decomposable per signal:**
   - per-recipe `cli_target_override` field tells observability tooling whether the operator explicitly chose the local path
   - per-row manifest schema includes `family`, `variant_id`, `archive_bytes`, `proxy_loss`, `d_seg_proxy`, `d_pose_proxy` — every signal axis decomposable
   - `evidence_grade` + `evidence_semantics` + `forbidden_uses` per row carry the AXIS-LABEL non-negotiable structurally

3. **Diff-able across runs:**
   - Each dispatch writes a unique `instance_job_id`-keyed manifest row
   - Sort by `instance_job_id` chronologically to see run-to-run drift
   - `proxy_loss_delta_vs_anchor` field is computed per row when an anchor is provided

4. **Queryable post-hoc:**
   - JSONL append-only at `.omx/state/mps_research_signal_manifest.jsonl` (canonical helper writes here)
   - JSONL append-only at `.omx/state/macos_cpu_advisory_signal_manifest.jsonl` (canonical helper writes here)
   - Standard fcntl-locked write per Catalog #128/#131 sister discipline
   - Loadable via `tac.optimization.mps_research_signal.load_observations(path)` for arbitrary downstream analysis

5. **Cite-able:**
   - Every manifest row has `instance_job_id` + `source` + `run_id` + recipe path
   - Every dispatch logs to the existing `.omx/state/active_lane_dispatch_claims.md` with terminal status `completed_local_mps` / `failed_local_mps` so the operator-facing dispatch claim ledger has full provenance

6. **Counterfactual-able:**
   - The dispatcher caches the recipe contents + `cli_target_override` value alongside the manifest row so "what would the result be if we re-ran with `--target modal` instead" is answerable by re-loading the recipe + flipping the override (NOTE: re-running with `--target modal` incurs $$$; the operator decides)

---

## STRICT preflight gate (new Catalog #317)

**Name:** `check_local_research_signal_dispatches_stamp_evidence_grade`

**Bug class to prevent:** A future refactor silently removes the
`evidence_grade="MPS-research-signal"` (or `macOS-CPU-advisory`) auto-stamp from
`_dispatch_local_mps` (or `_dispatch_local_cpu`), allowing local-MPS or local-CPU
results to land in the canonical posterior without their non-authoritative
markers.

**Detection:** AST + text-scan of `tools/operator_authorize.py`. Refuses
absent any of the following from the bodies of `_dispatch_local_mps` /
`_dispatch_local_cpu`:

1. A call to `torch.backends.mps.is_available()` (only required in
   `_dispatch_local_mps`)
2. A reference to `EVIDENCE_GRADE` (token from the canonical helper module)
3. A reference to the loud banner string (or a constant containing the
   `NON-AUTHORITATIVE` token)
4. A reference to either `append_manifest_row_to_jsonl` (canonical helper)
   OR to the canonical jsonl path (`mps_research_signal_manifest.jsonl` or
   `macos_cpu_advisory_signal_manifest.jsonl`)

**Same-line waiver:** `# LOCAL_RESEARCH_SIGNAL_STAMP_WAIVED:<rationale>` on the
`def _dispatch_local_mps(` or `def _dispatch_local_cpu(` line. Placeholder
`<rationale>` / `<reason>` literals rejected.

**Strict-flip:** STRICT from byte one per "Strict-flip atomicity rule"; live
count at landing: 0 (the new functions carry every required token).

**Sister of:**
- Catalog #1 (`check_no_mps_fallback_default` — source-text scan for silent fallback)
- Catalog #127 (`check_authoritative_tag_requires_custody_metadata` — per-call-site custody routing)
- Catalog #192 (`check_macos_cpu_advisory_not_promoted_without_linux_verification` — per-artifact promotion guard)
- Catalog #279/#280/#283 (fail-closed source-text scan for dispatch guard helpers)

---

## Files touched (this build)

| File | Action | LOC est |
|---|---|---|
| `tools/operator_authorize.py` | EDIT — add `--target` CLI flag + `_dispatch_local_mps` + `_dispatch_local_cpu` + 2 fork branches in `_run_dispatch` + 2 fork branches in `_native_dispatch_preflight` + extend `_platform_has_native_dispatch` + banner helper | ~300 |
| `src/tac/optimization/mps_research_signal.py` | EDIT — add sister `append_manifest_row_to_jsonl` helper mirroring macos_cpu_advisory_signal L477 | ~80 |
| `src/tac/deploy/dispatch_protocol.py` | EDIT — extend `LEGAL_NATIVE_PLATFORMS` to include `local_mps` + `local_cpu`; extend `LEGAL_DISPATCH_KINDS` to include `local_research_signal`; extend `_is_tool_dispatch` to also detect `local_research_signal` for protocol-scope skip | ~40 |
| `tools/local_pre_deploy_check.py` | EDIT — add `_is_local_research_signal_dispatch_for_harness` helper + `_LOCAL_RESEARCH_SIGNAL_SKIPPED_CHECKS` frozenset + skip logic in main loop | ~80 |
| `tools/canonical_dispatch_optimization_protocol.py` | EDIT — extend the canonical helper's `_is_tool_dispatch` to ALSO detect `local_research_signal` (analogous scope-fix) | ~30 |
| `src/tac/preflight.py` | EDIT — add `check_local_research_signal_dispatches_stamp_evidence_grade` (Catalog #317) + wire into `preflight_all(strict=True)` | ~110 |
| `src/tac/tests/test_one_arg_local_mps_vs_modal_dispatch_switch.py` | NEW — 30-50 tests | ~700 |
| `.omx/research/one_arg_local_mps_vs_modal_dispatch_switch_design_20260517.md` | NEW (this memo) | — |
| `.omx/tmp/one_arg_local_mps_dispatch_switch_premise_verifier.txt` | NEW (premise verification) | — |
| `~/.claude/projects/.../feedback_one_arg_local_mps_vs_modal_dispatch_switch_landed_20260517.md` | NEW landing memo | ~400 |
| `CLAUDE.md` | EDIT — add Catalog #317 row | ~10 |
| `docs/pr_writeups/cpu_frontier_fec6_20260517.md` | EDIT — §4.5 only, 2-3 sentence note | ~5 |

**Estimated total: ~1755 LOC across 12 files.**

The "30-sec-reviewable" budget is comfortable per-file because each surface is
narrowly scoped and the canonical patterns (Recipe class, manifest helpers,
preflight gates) are reused verbatim.

---

## Sister-subagent coordination per Catalog #302

Active in-flight subagents (per `.omx/state/subagent_progress.jsonl`):

- `afcada2203ab5b774` DP1+fec6 dual stacking — owns `src/tac/substrates/pretrained_driving_prior/*`. NO OVERLAP.
- `a2ce5edd2d297d91f` producer→cathedral_autopilot wire-in — owns `tools/cathedral_autopilot_autonomous_loop.py`. NO OVERLAP.
- Scope-fix subagent COMPLETED earlier today; `tools/operator_authorize.py` + `tools/local_pre_deploy_check.py` + `src/tac/deploy/dispatch_protocol.py` + `tools/canonical_dispatch_optimization_protocol.py` are RELEASED. I own the local_mps/local_cpu extensions on these surfaces.

My files_touched declared in checkpoint discipline per Catalog #206; sister
subagents see the declaration and can coordinate around it.

---

## Risk + rollback

**Risk:** the new `--target` CLI flag could collide with a future flag name.

**Mitigation:** `--target` is reserved exclusively for platform-override in this
build's contract. Future flags must pick a different name.

**Rollback:** the build is purely additive. Reverting the commit restores
status quo without affecting any existing recipe.

---

## Cross-references

- CLAUDE.md "MPS auth eval is NOISE" non-negotiable
- CLAUDE.md "Forbidden device-selection defaults (the MPS-fallback trap)"
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"
- CLAUDE.md "Production-hardened dispatch optimization protocol" (Catalog #270 scope clarification 2026-05-17)
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode"
- Catalog #1 (`check_no_mps_fallback_default`) — silent MPS fallback detector
- Catalog #127 (`check_authoritative_tag_requires_custody_metadata`) — per-call-site routing
- Catalog #131 (`check_no_bare_writes_to_shared_state`) — JSONL write discipline
- Catalog #192 (`check_macos_cpu_advisory_not_promoted_without_linux_verification`) — per-artifact promotion guard
- Catalog #240 (`check_substrate_contest_cuda_chain_complete_or_research_only_tagged`) — recipe-vs-trainer-state consistency
- Catalog #270 (`check_dispatch_optimization_protocol_complete`) — umbrella protocol gate
- Catalog #279/#280/#283 — fail-closed dispatch guard pattern
- Catalog #292 (`check_grand_council_deliberation_has_explicit_assumption_statements`) — per-deliberation assumption surfacing
- Catalog #294 (`check_substrate_landing_memo_has_9_dim_checklist_evidence_section`)
- Catalog #296 (`check_substrate_predicted_band_has_dykstra_feasibility_check`)
- Catalog #303 (`check_substrate_design_memo_has_cargo_cult_audit_section`)
- Catalog #305 (`check_substrate_design_memo_has_observability_surface_section`)


# HORIZON_CLASS_DECLARATION_OK:design_memo_is_dispatch_switch_design_NOT_substrate_design_predicted_band_section_is_for_dispatch_mode_routing_not_substrate_score_band_filename_pattern_matches_gate_false_positive_per_comprehensive_bug_audit_cascade_20260526
