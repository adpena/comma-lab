<!-- SPDX-License-Identifier: MIT -->
<!-- canonical_equation_cross_ref: procedural_codebook_from_seed_compression_savings_v1 (Catalog #344 registry #26; 4 empirical anchors + 6 lifecycle events as of v1 + sister codex 5-substrate procedural surface matrix landing `06b69b8ed` superseding 5-substrate matrix design; FORMALIZATION_PENDING:wave_3_end_of_day_command_sheet_v2_respawn_synthesis_distills_state_post_dp1_paired_dispatch_landed_at_03_07Z_baseline_failed_03_10Z_no_new_empirical_finding_claim_just_ready_to_paste_canonical_harvest_and_re_rank_commands_per_carmack_mvp_first_phasing_20260520 -->
---
council_tier: T1
council_attendees: [Command-Sheet-Distiller-V2-Respawn]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_decisions_recorded:
  - "distill END-OF-DAY V1 command sheet (commit 4f35a8289) into V2-PIVOT reflecting MAJOR state shifts since v1: DP1 paired-smoke ALREADY DISPATCHED (baseline 03:07Z + procedural 03:09Z) + baseline harvest FAILED rc=1 at 03:10Z + 5-substrate matrix SUPERSEDED by procedural replacement surface matrix + ATW V2 BUILD DEFERRED with 3 failure modes + parser-safe null-byte subset smoke landed + scorer-response authority hardening landed"
  - "PRIMARY operator-routable shift: from BLOCKED-pending-3-prerequisites (v1) to HARVEST+RE-RANK guidance (v2) — DP1 paired-smoke is IN-FLIGHT not BLOCKED"
  - "5 operator-decision points reordered per actual current state: #1 was 'authorize DP1 paired-smoke' (BLOCKED), now becomes 'harvest DP1 procedural call + recover baseline Comma2k19 chunk_ids bug' (HARVEST mode)"
  - "5-priority cascade-continuation queue compressed: pair #2 FREE smoke ALREADY FALSIFIED (a986efa99 zscore=101.18); REMOVED from queue; replaced with 'parser-safe procedural surface matrix Rank #1 DP1 codebook_blob harvest closure'"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
horizon_class: frontier_pursuit
canonical_vs_unique_decision_per_layer: see §2 below
nine_dim_checklist_evidence: see §3 below
cargo_cult_audit_per_assumption: see §4 below
observability_surface: see §5 below
---

# WAVE-3 END-OF-DAY OPERATOR-ROUTABLE COMMAND SHEET v2 — 2026-05-20

**Lane**: `lane_wave_3_end_of_day_operator_routing_command_sheet_v2_20260520` L1
**Subagent**: `wave-3-end-of-day-command-sheet-v2-respawn-20260520` (respawn of crashed predecessor `adef5934`)
**Parent v1**: `.omx/research/wave_3_end_of_day_operator_routable_command_sheet_20260520.md` (commit `4f35a8289`, 33.4 KB / 436 lines)
**Parent cascade-mortality**: `.omx/research/wave_3_honest_cascade_mortality_assessment_20260520.md` (commit `d884dd6aa`, 46.8 KB)
**DP1 activation anchor**: `.omx/research/dp1_paired_smoke_operator_activation_20260521T030117Z_codex.md` (commit `9aab2a177`)
**Mission contribution per Catalog #300**: `apparatus_maintenance` — operator-routing distillation pivot from BLOCKED to HARVEST mode; immediate score-mutating value N/A; HIGH end-of-day-pivot decision-utility.
**Sister-DISJOINT** verified via PRE-FLIGHT `tools/check_sister_files_recently_landed.py` PROCEED (no sister commits touched the V2 memo path in 12-hour lookback). PARSER-SAFE METHODOLOGY EXTENSION sister is NOT currently respawned (not in checkpoint store); zero file-path collision.

## 1. Summary — what shifted since V1

V1 framed DP1 paired-smoke as **BLOCKED pending 3 prerequisites** (recipes + trainer extension + inflate runtime). The actual cascade landed those 3 prerequisites in 5 chronological commits between V1's distillation time (2026-05-21T00:17Z) and V2's distillation time (2026-05-21T03:12Z), then operator-activated AND dispatched both paired-smoke calls:

| Commit | UTC | Description |
|---|---|---|
| `b93c15afd` | 03:00Z | Wire DP1 procedural paired-smoke recipes (3 recipe YAMLs + procedural_codebook_inflate.py + trainer extension) |
| `d2229c49c` | 03:01Z | Wire DP1 procedural paired-harvest planner (`tools/plan_dp1_procedural_paired_harvest.py`) |
| `9aab2a177` | 03:02Z | Activate DP1 paired-smoke recipes (`dispatch_enabled: true` on baseline + procedural; null-exploit deferred) |
| `09ffe159e` | 03:05Z | Align research-smoke predeploy dispatch gate |
| `ef25aa20c` | 03:07Z | Record DP1 baseline Modal dispatch call `fc-01KS480WY6S90VFXX54SC7V209` |
| `4c90f2deb` | 03:09Z | Record DP1 procedural Modal dispatch call `fc-01KS484S3Z8YZBRVMCTQ6SX8MV` |

The **baseline call harvested rc=1 at 03:10Z** with a NEW bug class: `ValueError: Comma2k19LocalStreamer has no chunk ids; pass an explicit chunk_ids list or populate the streamer's dataset_sha256_manifest` (full traceback in `.omx/state/modal_call_id_ledger.jsonl` last 2 entries). The procedural call is still in-flight at V2 distillation time (3 min wall-clock; expected ~5-10 min to harvest).

Three additional parallel landings shifted the routing picture:

1. **5-substrate matrix SUPERSEDED** by `procedural_replacement_surface_matrix_20260521` (commit `06b69b8ed`); canonical ranking now lives at `experiments/results/procedural_replacement_surface_matrix_20260521T020000Z/surface_matrix.json` with DP1 ranked #1 (READY_TO_PAIR_SMOKE), ATW2 CDF table #2 (DESIGN_READY_DEFERRED), VQ-VAE #4 (ADAPTER_REQUIRED), GLV1 #5 (BLOCKED_NO_CURRENT_SURFACE), PR101 master-gradient null-bytes #6 (BLOCKED_PARSER_SAFE_SUBSET_EMPTY).
2. **ATW V2 procedural-variant BUILD DEFERRED** (`7ea78deaa`) with 3 failure modes documented (D4 predecessor verdict + Variant-C scoping gate + signal-preservation probe required for class-prior 19,168 B replacement).
3. **Parser-safe null-byte subset smoke landed** (`e3e198c9f`) — empirical receipt that **all null-gradient bytes are parser-essential** on PR101 FEC6 frontier (closes the master-gradient REMOVAL paradigm at the empirical surface, complementing Catalog #359 structural extinction).

**At V2 distillation time (2026-05-21T03:12Z UTC)**:
- 27 canonical equations registered (was 26 at v1; +1 net via wire-in audit)
- 56 cathedral consumer packages (unchanged)
- Catalog #185 META-meta drift = 0 violations
- Frontier pointer: [contest-CPU] **0.1920513168811056** fec6 sha `6bae0201` (unchanged — no paid GPU score change today)
- PR #110 OPEN with bot-ack only; no maintainer review

## 2. Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Canonical / Unique | Rationale |
|---|---|---|
| Canonical harvest invocation | ADOPT_CANONICAL `tools/parallel_harvest_actuator.py --recover-from-tmp --lookback-hours 24` per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" + Catalog #245 + Catalog #330 | Sister codex `tools/plan_dp1_procedural_paired_harvest.py` is the DP1-specific planner; the harvester is the canonical pickup |
| Canonical paired auth-eval | ADOPT_CANONICAL `tools/dispatch_modal_paired_auth_eval.py --skip-axis-if-promotable-anchor-exists` per Catalog #246 | Per codex DP1 activation memo "next action" instruction |
| Canonical procedural surface matrix | ADOPT_CANONICAL `experiments/results/procedural_replacement_surface_matrix_20260521T020000Z/surface_matrix.json` per sister landing `06b69b8ed` | Canonical ranked-routing matrix supersedes V1's ad-hoc 5-priority queue |
| Operator-direct work queue | NONE | The 3 paired-smoke recipe YAMLs that v1 §8 surfaced as operator-direct are NOW LANDED via codex sister; no operator-direct work remaining at distillation time |
| Memo composition | ADOPT_CANONICAL 6-section design-memo discipline per Catalog #290+#294+#296+#303+#305+#309 | Same shape as v1 + cascade-mortality memo |
| Frontier pointer citation | ADOPT_CANONICAL `tools/refresh_canonical_frontier.py --json` per Catalog #343 | Live pointer; no hardcoded score literals |

## 3. 9-dimension success checklist evidence (Catalog #294)

| Dim | Status | Evidence |
|---|---|---|
| 1. UNIQUENESS | YES | First V2 respawn distillation reflecting post-dispatch state; no prior memo at this path |
| 2. BEAUTY + ELEGANCE | YES | Compact section structure per task spec; ready-to-paste commands in `bash` code-fences; harvest-mode reorientation makes operator action clearer than v1 |
| 3. DISTINCTNESS | YES | Distinct from v1 (HARVEST mode vs BLOCKED-pending-prerequisites); distinct from cascade-mortality (decision sheet vs META-analysis); sister-DISJOINT |
| 4. RIGOR | YES | PV per Catalog #229: read v1 + cascade-mortality + DP1 activation + procedural matrix memos + queried Modal call_id ledger + canonical equation registry + frontier pointer; checkpointed every step per Catalog #206 |
| 5. OPTIMIZATION PER TECHNIQUE | N/A | Distillation memo; delegated to underlying recipes + sister landings |
| 6. STACK-OF-STACKS COMPOSABILITY | YES | V2 composes the operator-routable surfaces of v1 + cascade-mortality + DP1 activation + procedural surface matrix + ATW V2 DEFER + parser-safe smoke + 56 cathedral consumers into ONE end-of-day pivot sheet |
| 7. DETERMINISTIC REPRODUCIBILITY | YES | Every command cites canonical recipe filename + Catalog # disciplines + sister commit SHA |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | N/A | Distillation memo |
| 9. OPTIMAL MINIMAL CONTEST SCORE | N/A | Distillation does NOT mutate score; surfaces routable commands that DO |

## 4. Cargo-cult audit per assumption (Catalog #303)

| Assumption | Classification | Rationale |
|---|---|---|
| The baseline DP1 harvest failure (`ValueError: Comma2k19LocalStreamer has no chunk ids`) is a recoverable infrastructure bug, not a paradigm-level falsification | HARD-EARNED | Codex DP1 activation memo confirms the planner blocks "only on missing candidate output dirs, which are expected before Modal training/export runs complete"; the harvest failure is at training-time NOT eval-time; chunk_ids manifest is a known canonical state that the streamer must read from `dataset_sha256_manifest` per `tac.substrates.pretrained_driving_prior.local_chunk_cache.Comma2k19LocalCache` API |
| The procedural call (still in flight at distillation) WILL also fail the same Comma2k19LocalStreamer bug | HARD-EARNED-EMPIRICAL-PROJECTION | Same trainer entry point, same code path; high probability of same failure mode unless the procedural-recipe env_overrides explicitly route around it |
| The 5-substrate matrix supersession to procedural surface matrix is operator-routable | HARD-EARNED | Codex sister landing `06b69b8ed` ships the surface matrix as canonical Python module + CLI + tests; Decision #4 in v1 (NSCS06 v8 BUILD) is now subsumed by Rank #5 GLV1 BLOCKED + Rank #6 PR101 BLOCKED rows |
| PR #110 maintainer-review acceleration remains LOW-EV operator-only discretion | HARD-EARNED-PENDING-EMPIRICAL | Same as v1; no maintainer state shift |
| Parser-safe null-byte subset smoke EMPIRICALLY CONFIRMS the master-gradient REMOVAL paradigm extinction | HARD-EARNED | Commit `e3e198c9f` landing memo documents `PARSER_SAFE_SUBSET_EMPTY` verdict; complements Catalog #359 structural extinction; closes the loop on the H3_OPAQUE_TO_SCORER verdict from `3dfb877c0` |
| The cascade is in CHECKPOINT mode, not TERMINATION mode (per v1 §13 framing) | HARD-EARNED | Procedural DP1 call still in flight; harvest planner ready; surface matrix has 5 substrates with non-blocked status (3 READY/DESIGN_READY/ADAPTER_REQUIRED); tomorrow's cascade continues from DP1 harvest outcome |

## 5. Observability surface (Catalog #305)

1. **Inspectable per layer**: Each ready-to-paste command's canonical disciplines fire structurally per Catalog # — operator can inspect verbatim via `tools/operator_authorize.py --recipe <name> --dry-run` AND `tools/parallel_harvest_actuator.py --dry-run` AND `tools/plan_dp1_procedural_paired_harvest.py --json-out /tmp/<file>.json`
2. **Decomposable per signal**: 5 operator-decision points (post-pivot) × 4-priority cascade-continuation queue (pair #2 removed) = 9 distinct action surfaces; each independently inspectable
3. **Diff-able across runs**: Every command emits canonical artifacts (Modal call_id ledger / lane-claim ledger / canonical equations registry / commit-serializer log / procedural surface matrix JSON); diff at canonical surfaces
4. **Queryable post-hoc**: `tac.canonical_equations.query_equations()` + `ls -d src/tac/cathedral_consumers/*/` + `.omx/state/modal_call_id_ledger.jsonl` + `experiments/results/procedural_replacement_surface_matrix_*/surface_matrix.json` all queryable
5. **Cite-able**: Every command cites canonical recipe filename + Catalog # disciplines + sister commit SHA + parent design memo
6. **Counterfactual-able**: Catalog #272 byte-mutation smoke + Catalog #167 smoke-before-full both produce empirical counterfactual evidence; the DP1 paired-smoke IS the counterfactual surface for canonical equation #26's first IN-DOMAIN empirical anchor

## 6. 5 operator-decision points — READY-TO-PASTE canonical commands (v2-PIVOT)

### Decision #1 — Harvest DP1 procedural call (in-flight) AND recover baseline Comma2k19 chunk_ids bug

**Current state (v2 PIVOT)**: HARVEST MODE — baseline FAILED rc=1 at 03:10Z; procedural in-flight (~5-10 min remaining). The procedural call is the FIRST IN-DOMAIN empirical anchor candidate for canonical equation #26 PER recipe predicted band `[-0.005000, +0.000500]`.

**Step 1 — Harvest the procedural call once it completes** (canonical 24h harvest window per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE"):

```bash
.venv/bin/python tools/parallel_harvest_actuator.py \
    --recover-from-tmp \
    --lookback-hours 24
```

**Step 2 — Plan the paired CPU/CUDA auth-eval harvest** (per codex DP1 activation memo "next action" instruction; tools/plan_dp1_procedural_paired_harvest.py is the canonical planner):

```bash
.venv/bin/python tools/plan_dp1_procedural_paired_harvest.py \
    --baseline-output-dir experiments/results/lane_substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch_20260521T030640Z_modal/harvested_artifacts \
    --procedural-output-dir experiments/results/lane_substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch_20260521T030858Z_modal/harvested_artifacts \
    --json-out /tmp/dp1_paired_harvest_plan.json \
    --md-out /tmp/dp1_paired_harvest_plan.md
```

**Step 3 — Dispatch paired auth-eval through canonical actuator** (per Catalog #246 skip-axis-if-anchor-exists discipline; only after Step 2 plan is reviewed):

```bash
# Catalog #199 paired-env attestation
export OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1
export OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=0.10

.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
    --skip-axis-if-promotable-anchor-exists \
    --plan-from /tmp/dp1_paired_harvest_plan.json
```

**Step 4 — Diagnose + repair Comma2k19LocalStreamer chunk_ids bug for baseline re-dispatch**:

The baseline harvest failed at `src/tac/substrates/pretrained_driving_prior/log_incremental_feeder.py:514` with `Comma2k19LocalStreamer has no chunk ids; pass an explicit chunk_ids list or populate the streamer's dataset_sha256_manifest`. The fix path is operator-direct OR sister-subagent-routable (NOT auto-routable):

```bash
# Inspect the failing path empirically
grep -n "dataset_sha256_manifest\|chunk_ids" src/tac/substrates/pretrained_driving_prior/log_incremental_feeder.py | head -20

# Either: (a) populate manifest at module load OR (b) thread explicit chunk_ids
# through env_overrides in the baseline recipe; sister-subagent should investigate
```

**EV / cost**: HIGH (procedural-arm in-flight = first IN-DOMAIN empirical anchor for canonical equation #26; closed-form predicted ΔS −0.002706). **Sequencing**: Step 1 → Step 2 → Step 3 in 24h window per Catalog #245. Step 4 is independent + parallel.

### Decision #2 — Pivot from MAGIC CODEC pair cascade to canonical procedural surface matrix Rank #2 (ATW2 CDF table)

**Current state (v2 PIVOT)**: Pair #2 ALREADY FALSIFIED (commit `a986efa99` empirical ΔS +0.054055; zscore=101.18 per cascade-mortality §6 Falsification #3). The cascade-mortality verdict explicitly flagged "PIVOT to pair #4 (magic_codec orthogonality validation, FREE CPU) OR DP1-only paired smoke"; with DP1 now in-flight, the canonical next-cascade-target is ATW2 CDF table per procedural surface matrix Rank #2.

**ATW2 CDF table is BLOCKED** per sister memo `atw_v2_procedural_variant_build_design_20260520.md` (commit `7ea78deaa`) with 3 documented failure modes:
1. D4 predecessor verdict (need to resolve)
2. Variant-C scoping gate (need design memo)
3. Paid dispatch gate (need symposium per Catalog #325)

**Operator action — spawn sister subagent for ATW2 CDF table BUILD** (NOT ready-to-paste; sister-subagent-spawn skeleton):

```text
# Spawn sister subagent (via Task tool) with this prompt skeleton:
# - Subject: WAVE-3 ATW V2 CDF TABLE PROCEDURAL VARIANT BUILD per procedural surface matrix Rank #2
# - Parent design: .omx/research/atw_v2_procedural_variant_build_design_20260520.md
# - Resolve: D4 predecessor verdict + Variant-C scoping + signal-preservation probe
# - Deliverable: src/tac/substrates/atw_codec_v2/procedural_cdf_table.py + tests + lane registry L1
# - Discipline: full Catalog discipline checklist + #229 PV + #110/#113 APPEND-ONLY
```

**EV / cost**: MEDIUM ($0.30-1.00 paired smoke after BUILD lands + symposium; predicted ΔS −0.001683 per surface matrix). **Sequencing dependency**: ATW V2 BUILD + Catalog #325 symposium.

### Decision #3 — Decide DWT-HNeRV bind RE-SCOPE direction (Option A bilinearly-upsampled detail OR Option B brotli/STC-coded detail)

**Unchanged from v1**: Sister design memo `.omx/research/dwt_bind_rescope_intermediate_transform_path_design_20260520.md` (commit `37fea4aac`) carries both options with per-Dykstra-feasibility verdicts. T3 op-routable #1 ($1 paired smoke) DEFERRED pending RE-SCOPE direction.

**Operator action — sister symposium follow-up** (NOT ready-to-paste):

```text
# Spawn sister subagent (via Task tool) with this prompt skeleton:
# - Subject: WAVE-3 DWT-HNeRV BIND RE-SCOPE FOLLOW-UP T3 SYMPOSIUM
# - Read: .omx/research/dwt_bind_rescope_intermediate_transform_path_design_20260520.md
# - Verdict required: Option A vs Option B vs both-parallel per Catalog #292
# - Deliverable: council memo at .omx/research/council_T3_dwt_hnerv_bind_rescope_follow_up_<utc>.md
# - Discipline: Catalog #300 v2 frontmatter + #346 canonical roster + #292 per-deliberation assumption
```

**EV / cost**: MEDIUM ($1-$2 per option after RE-SCOPE; predicted band [-0.015, -0.005] would break 0.18 floor).

### Decision #4 — NSCS06 v8 substrate BUILD (DOWNGRADED per surface matrix Rank #5 BLOCKED)

**Current state (v2 PIVOT)**: NSCS06 lives at procedural surface matrix Rank #5 (`grayscale_lut_glv1` chroma_lut) with status `BLOCKED_NO_CURRENT_SURFACE` because "GLV2 explicit LUT grammar required". The v1 Decision #4 framing (BUILD + symposium) is structurally subsumed — the BUILD requires a GLV2 schema bump FIRST.

**Operator action — defer until GLV2 schema bump lands** (operator-direct decision; sister-subagent-routable after schema spec):

```text
# Spawn sister subagent (via Task tool) with this prompt skeleton:
# - Subject: WAVE-3 GLV2 SCHEMA BUMP DESIGN per procedural surface matrix Rank #5 unblock
# - Required: explicit chroma_lut grammar section so procedural codebook replacement is parser-visible
# - Deliverable: design memo at .omx/research/glv2_schema_bump_design_<utc>.md
# - After landing: NSCS06 v8 BUILD becomes operator-routable
```

**EV / cost**: LOW (only 224 B savings per surface matrix; predicted ΔS −0.000149). DOWNGRADED from v1 MEDIUM-priority because the canonical surface matrix ranks DP1 (4064 B / ΔS −0.002706) and ATW2 (2528 B + 19168 B / ΔS −0.001683 + −0.012763) much higher.

### Decision #5 — PR #110 maintainer-review acceleration

**Unchanged from v1**: PR OPEN; 0 maintainer reviews; bot acknowledgment 2026-05-20T03:29:04Z; last update 2026-05-20T14:46:27Z.

**Operator-routable surface**: NONE auto-routable per CLAUDE.md "Public Disclosure Hygiene" + "Executing actions with care". Operator-only discretion.

```bash
# Option A: passive monitoring only (recommended)
gh pr view 110 --repo commaai/comma_video_compression_challenge --json state,updatedAt,comments

# Option B: nothing — wait for natural maintainer cadence (recommended per PR108 rubric)
```

## 7. 4-priority cascade-continuation queue (v2-COMPRESSED) — READY-TO-PASTE commands

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" + Carmack MVP-first phasing. **Pair #2 removed (FALSIFIED)**. Queue compressed from 5 to 4 priorities.

### Priority #1 (PRIMARY) — DP1 procedural paired-smoke harvest (in-flight; harvest within 24h)

**Status**: IN-FLIGHT; procedural call `fc-01KS484S3Z8YZBRVMCTQ6SX8MV` dispatched 03:09Z; expected harvest within 5-10 min wall-clock.

**Canonical harvest command**: see Decision #1 Step 1 above.

**Predicted outcome**: canonical equation #26 first IN-DOMAIN empirical anchor; predicted ΔS −0.002706. If empirical ΔS within 2σ → HARD-EARNED + structural reinforcement of equation #26 INCLUDED contexts. If outside 2σ → cargo-cult-pending-investigation; cascade re-routes via Decision #2 ATW2 CDF table.

**EV / cost**: $0.30 Modal T4 already spent on dispatch; harvest is $0 + ~5 min wall-clock.

### Priority #2 — Baseline DP1 Comma2k19LocalStreamer bug repair + re-dispatch

**Status**: BLOCKED pending Step 4 of Decision #1 (chunk_ids manifest fix).

**Repair command** (operator-direct OR sister-subagent-routable; ~10-30 min):

See Decision #1 Step 4 above for the grep + investigation pattern. After fix lands + commit via canonical serializer:

```bash
# Catalog #199 paired-env (re-dispatch baseline only)
export OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1
export OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=0.30

.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch \
    --target modal \
    --yes
```

**EV / cost**: $0.30 Modal T4; baseline is the canonical comparator for procedural's first IN-DOMAIN anchor.

### Priority #3 — ATW2 CDF table BUILD (procedural surface matrix Rank #2)

**Status**: BLOCKED pending Decision #2 sister-subagent BUILD + Catalog #325 symposium.

**Sister-subagent spawn skeleton**: see Decision #2 above.

**EV / cost**: $0.30-1.00 after BUILD + symposium; predicted ΔS −0.001683 per surface matrix.

### Priority #4 — DWT-HNeRV bind RE-SCOPE (Decision #3 unchanged from v1)

**Status**: BLOCKED pending sister symposium follow-up OR operator-direct parallel-dispatch decision.

**Sister-spawn skeleton**: see Decision #3 above.

**EV / cost**: $1-$2 per option; predicted band [-0.015, -0.005] would break 0.18 floor.

**REMOVED from queue (FALSIFIED)**: V1 Priority #1 pair #2 FREE CPU smoke (commit `a986efa99` empirical ΔS +0.054055; zscore=101.18; closed by Catalog #359 structural extinction).

## 8. Operator-direct work queue (v2-EMPTY at distillation time)

V1 §8 surfaced 3 operator-direct work items (paired-smoke recipe YAMLs). **All 3 landed via codex sister `b93c15afd`** between v1 distillation and v2 distillation. No operator-direct work queued at v2 distillation time.

**Operator-frontier-override authorization per Catalog #300 Consequence 1** (if invoked): unchanged from v1 §8.2. Canonical pattern:

```yaml
# Add to ANY recipe under .omx/operator_authorize_recipes/:
council_override_invoked: true
council_override_rationale: "<verbatim operator quote ≥4 chars; non-placeholder>"
council_override_memo: .omx/research/operator_authorizations/operator_frontier_override_<lane_id>_<YYYYMMDD>.md
```

## 9. Discipline checklist

- Catalog #110 + #113 APPEND-ONLY HISTORICAL_PROVENANCE: NEW v2 command sheet + landing memo; ZERO mutation of v1 command sheet, cascade-mortality memo, DP1 activation memo, procedural surface matrix memo, ATW V2 DEFER memo, parser-safe smoke memo, or any cited sister artifact
- Catalog #117 + #157 + #174 commit serializer with POST-EDIT `--expected-content-sha256` (per landing commit below)
- Catalog #119 Co-Authored-By trailer (auto-appended by serializer)
- Catalog #125 6-hook wire-in declaration (§10 below)
- Catalog #167 smoke-before-full (cited; auto-gated inside canonical operator-authorize chain)
- Catalog #176 STRICT callsite has CLAUDE.md row (N/A — no new catalog claim)
- Catalog #185 META-meta drift sister regression (0 violations verified at distillation)
- Catalog #186 catalog # claim transactional (N/A — no new catalog # claimed)
- Catalog #199 paired-env discipline (canonical commands in §6 + §7 honor paired-env)
- Catalog #206 crash-resume discipline (4 in_progress checkpoints emitted: step 1+2+3 + final complete)
- Catalog #229 premise-verification (read v1 + cascade-mortality + DP1 activation + procedural surface matrix + ATW V2 DEFER + parser-safe smoke + queried Modal call_id ledger + canonical equation registry + frontier pointer + sister checkpoint state before write)
- Catalog #230 sister-subagent ownership map (DISJOINT verdict; sister-checkpoint guard PROCEED)
- Catalog #245 Modal call_id ledger (canonical 4-layer pattern cited for harvest discipline)
- Catalog #246 paired dispatch skip-axis-if-anchor-exists (cited in Decision #1 Step 3)
- Catalog #287 placeholder-rationale rejection (all rationales substantive ≥4 chars; zero `<rationale>` / `<reason>` literals)
- Catalog #290 canonical-vs-unique decision per layer (§2)
- Catalog #292 per-deliberation assumption surfacing (§4 cargo-cult audit)
- Catalog #294 9-dim success checklist (§3)
- Catalog #296 Dykstra-feasibility (N/A — distillation memo)
- Catalog #300 v2 council frontmatter (T1 distillation memo; apparatus_maintenance mission contribution)
- Catalog #303 cargo-cult audit (§4)
- Catalog #305 observability surface (§5)
- Catalog #309 horizon_class=`frontier_pursuit` (cascade-continuation lens)
- Catalog #313 probe-predecessor check (N/A — distillation memo, not dispatch)
- Catalog #314 + #340 sister-checkpoint absorption guard (PROCEED verdict at PRE-FLIGHT)
- Catalog #323 canonical Provenance umbrella (every empirical claim cites canonical commit SHA + canonical-state query)
- Catalog #325 per-substrate symposium recency (cited at Decision #2 + #4)
- Catalog #330 Modal harvester call_id outcome ledger (cited in Decision #1 Step 1 + Priority #1)
- Catalog #335 cathedral consumer Protocol contract (cited; 56 consumers verified at distillation)
- Catalog #339 silent-no-spawn structural extinction (canonical post-dispatch registration confirmed in ledger entries 03:07Z + 03:09Z)
- Catalog #343 frontier pointer (cited via `tools/refresh_canonical_frontier.py --json`)
- Catalog #344 canonical equation cross-reference (HTML comment after frontmatter cites equation #26 + FORMALIZATION_PENDING waiver per distillation-not-empirical-finding scope)
- Catalog #346 canonical roster (N/A — T1 distillation memo)
- Catalog #348 retroactive sweep (N/A — no new STRICT gate added)
- Catalog #359 residual-hybrid misapplication structural extinction (cited at Pair #2 cascade-removal rationale)

## 10. 6-hook wire-in declaration per Catalog #125

| Hook | Status | Rationale |
|---|---|---|
| #1 sensitivity-map | N/A | Distillation memo; no sensitivity-map signal contribution |
| #2 Pareto constraint | N/A | No Pareto-feasible-region constraint added |
| #3 bit-allocator | N/A | No bit-allocator signal contribution |
| #4 cathedral autopilot dispatch | ACTIVE INDIRECT | Distillation surfaces the 4-priority cascade-continuation queue + 5 operator-decision points; downstream operator-routed dispatchers consume the queue as routing input per Catalog #335 auto-discovery |
| #5 continual-learning posterior | ACTIVE | Distillation preserves canonical state (27 equations + 56 consumers + Catalog #185 0 violations + frontier pointer fec6 0.1920513 unchanged) + tracks the 2 new Modal call_ids (`fc-01KS480WY6S90VFXX54SC7V209` baseline + `fc-01KS484S3Z8YZBRVMCTQ6SX8MV` procedural) as v2-distillation-time posterior snapshot |
| #6 probe-disambiguator | ACTIVE | 4-priority sequencing per Carmack MVP-first IS the canonical disambiguator between competing cascade-continuation paths; the procedural surface matrix Rank ordering disambiguates Decision #2 vs Decision #4 directly |

## 11. Sister regression at v2 distillation time

- Canonical equations: **27 registered** (was 26 at v1; +1 net)
- Canonical equation #26 anchors: **4** (was 3 at v1; +1 via REMOVAL `domain_refined` event per cascade-mortality §8)
- Cathedral consumer packages: **56** (unchanged)
- Catalog #185 META-meta drift gate: **0 violations** (clean)
- Canonical frontier pointer (per `tools/refresh_canonical_frontier.py --json`):
  - [contest-CPU] **0.1920513168811056** fec6 frontier sha `6bae0201` (unchanged today)
  - [contest-CUDA] **0.20533** pr106_format0d sha `9cb989cef519` (unchanged)
- Modal call_id ledger: 2 NEW dispatched events at 03:07Z + 03:09Z; 1 NEW failed event at 03:10Z (baseline); procedural in-flight
- PR #110: **OPEN**; bot-ack only; no maintainer review; last update 2026-05-20T14:46:27Z
- HEAD at v2 distillation: `4c90f2deb`

## 12. Sister-collision verdict

**DISJOINT** at v2 distillation time. PRE-FLIGHT helper PROCEED verdict for V2 memo path. PARSER-SAFE METHODOLOGY EXTENSION sister NOT currently respawned in checkpoint store; zero file-path collision.

## 13. Blockers

NONE for THIS v2 distillation landing. The procedural DP1 call is in-flight (~5-10 min); harvest commands in §6 Decision #1 are ready-to-paste. The baseline failure is a known infrastructure bug class (Comma2k19LocalStreamer chunk_ids) with a clear repair path in §6 Decision #1 Step 4.

**End-of-day apparatus state v2 PIVOT**: the cascade's natural pause-point is a HARVEST CHECKPOINT, not a TERMINATION. Tomorrow's cascade continues from Priority #1 (DP1 procedural harvest — IN-FLIGHT) → Priority #2 (baseline repair + re-dispatch after Comma2k19 bug fix) → Priority #3 (ATW2 CDF table BUILD after symposium) → Priority #4 (DWT-HNeRV bind RE-SCOPE direction decision).

**End of v2 command sheet.**
