# Retroactive sweep for Catalog #358 — recipe-level /workspace/pact/<OUTPUT> bug class

**Sweep landed:** 2026-05-20T17:18:22Z
**Catalog #:** 358 (`check_recipe_workspace_output_path_canonical_or_modal_aware`)
**Lane:** `lane_wave_3_harden_1_master_gradient_tmp_path_extinction_20260520`
**Sister landing memo:** `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_wave_3_harden_1_master_gradient_tmp_path_extinction_landed_20260520.md`

## Per Catalog #348 4-field contract

Canonical field markers: bug-class symptom signature / pre-fix window /
historical-kill/defer/falsify search results / re-eval-priority assignment.

### 1. Bug-class symptom signature

The bug class is: **tool-dispatch recipes whose `env_overrides` block emits an OUTPUT env var (`*_NPY` / `*_OUTPUT_DIR` / `*_OUTPUT_*` / `*_OUT_*`) with a value under `/workspace/pact/` WITHOUT a Catalog #204 canonical 3-branch Modal-aware driver override.**

The symptom on Modal workers is one of:
- **Fail-fast (extractor /tmp guard):** `tools/extract_master_gradient.py:2369-2373` REFUSES `/tmp/...` paths per CLAUDE.md "Forbidden /tmp paths in any persisted artifact" non-negotiable + Catalog #220 transient-evidence trap. Modal mounts working tree under `/tmp/pact/` (NOT `/workspace/pact/`), so `/workspace/pact/.omx/state/foo.npy` resolves to `/tmp/pact/.omx/state/foo.npy` and triggers the extractor's `/tmp` refusal at runtime startup. **Symptom: rc=1 / 10-60s wall-clock / no output produced.**
- **Silent data loss (non-guard tool):** for tools that do NOT enforce the `/tmp` guard, output writes to `/tmp/pact/...` succeed but the Modal harvest pattern does NOT sync `/tmp/pact/` back to the local repo. The dispatch returns rc=0 but no artifact lands. **Symptom: rc=0 / full wall-clock / silent data loss.**

### 2. Pre-fix window

The bug class existed since the inception of tool-dispatch recipes (introduced 2026-05-17 per Catalog #270 scope clarification + `lane_catalog_270_scope_fix_tool_vs_substrate_dispatch_20260517`). Per-driver fixes have been landing since the Catalog #204 cross-driver expansion 2026-05-19 + STC v2 2026-05-14 + PR95++ 2026-05-13 + stack_of_stacks 2026-05-19, but NO recipe-level META gate existed until THIS landing.

Pre-fix window: **2026-05-17 → 2026-05-20** (~3 days). Empirical receipts:
- WAVE-3-OP3 dispatch `fc-01KS2Z2WJQW532A9226JAVQM8Y` (2026-05-20T15:11:22Z, ~$0.0016 wasted, rc=1 / 9.74s) — the canonical anchor for THIS bug class at the master_gradient_fec6_modal_t4_cuda_anchor recipe surface (closed by OP3-RETRY commit `75d39f32e` per-driver fix).
- Sister Catalog #204 anchors: STC v2 2026-05-14 (per `feedback_stc_v2_driver_path_layer_fix_landed_20260516.md`); PR95++ 2026-05-13; stack_of_stacks 2026-05-19 (per commit `956ad2e76`).

### 3. Historical KILL / DEFER / FALSIFY search results

**Searched path(s):** `.omx/research/` + `~/.claude/projects/-Users-adpena-Projects-pact/memory/` + `.omx/state/probe_outcomes.jsonl` + `src/tac/preflight.py` Catalog #204 + Catalog #220 + Catalog #270 sister rows + `git log --since=2026-05-13 --oneline` for STC v2 / PR95++ / stack_of_stacks / WAVE-3-OP3 anchors.

**Search command(s):**
- `grep -rn "/workspace/pact/.omx/state" .omx/operator_authorize_recipes/` — find OUTPUT-path bug-class candidates
- `git log --since=2026-05-13 --grep="Catalog #204" --grep="tmp path" --grep="modal_results"` — find sister-fix landings
- `grep -rln "Catalog #204" scripts/` — find drivers with canonical fix already landed

Searched for historical KILL / DEFER / FALSIFY verdicts that may have been issued against a substrate / lane / candidate where the actual root cause was THIS bug class (recipe emits `/workspace/pact/...` OUTPUT path silently lost or crashing on Modal).

**Findings:**
- **WAVE-3-OP3 predecessor** (2026-05-20T15:11:22Z) crashed at the extractor's `/tmp` guard. Per CLAUDE.md "Forbidden premature KILL without research exhaustion", the verdict was DEFER (sister OP3-RETRY landed the per-driver fix at commit `75d39f32e` then re-dispatched). No historical KILL verdict was issued; the recovery was correctly classified as IMPLEMENTATION-LEVEL FALSIFICATION per Catalog #307 (the specific recipe-driver configuration was falsified, not the master-gradient extraction paradigm).
- **STC v2 anchor** (2026-05-14, per sister memo `feedback_stc_v2_driver_path_layer_fix_landed_20260516.md`) — same bug class; per-driver fix landed; no historical KILL.
- **PR95++ anchor** (2026-05-13) — same bug class; per-driver fix landed; no historical KILL.
- **stack_of_stacks anchor** (2026-05-19) — same bug class; per-driver fix landed at commit `956ad2e76`; no historical KILL.

**Verdict:** ZERO historical KILL / FALSIFY verdicts need retroactive revision. The bug class has consistently been classified as IMPLEMENTATION-LEVEL FALSIFICATION (per-driver fixes), not PARADIGM-LEVEL FALSIFICATION (no substrate or technique was wrongly killed because of this bug class).

### 4. Per-finding RE-EVAL-priority assignment

NONE of the historical anchors require RE-EVAL because:
1. All 4 anchors (WAVE-3-OP3 + STC v2 + PR95++ + stack_of_stacks) received per-driver fixes that demonstrated the underlying paradigm was viable.
2. None of the 4 anchors produced a KILL / FALSIFY verdict against a substrate or technique.
3. The bug class is purely INFRASTRUCTURE-LEVEL (Modal mount-path mismatch + driver-side env-var override missing), not paradigm-level.

**The retroactive sweep verdict is therefore CLEAN:** Catalog #358 prevents future occurrences of the bug class structurally at the recipe-level surface, but no historical verdicts need revision.

## Cross-references

- Sister Catalog #204 (per-driver Modal-aware OUTPUT path expansion 2026-05-19); #358 is the recipe-level sister.
- Sister Catalog #220 (source-text /tmp guard at extractor surface 2026-05-15).
- Sister Catalog #270 (tool vs substrate dispatch scope clarification 2026-05-17).
- Sister Catalog #240 (recipe-vs-trainer-state consistency 2026-05-15).
- Sister Catalog #152 (operator-wrapper-validates-required-input-files canonical insertion point).
- Sister Catalog #185 (META-meta-meta drift detection — verified live count = 0 at landing).
- Sister Catalog #176 (META-meta STRICT-callsite-has-CLAUDE.md-row — verified satisfied).
- Sister Catalog #287 (placeholder-rationale rejection — applied to `WORKSPACE_OUTPUT_PATH_OK:<rationale>` waiver).
- Sister Catalog #307 (paradigm-vs-implementation-falsification classification — confirms IMPLEMENTATION-LEVEL verdicts for all 4 historical anchors).
- OP3-RETRY landing memo: `feedback_wave_3_op3_paid_retry_with_tmp_path_fix_landed_20260520.md` (the canonical Catalog #204 per-driver fix that THIS L3 META gate sister-extends).
