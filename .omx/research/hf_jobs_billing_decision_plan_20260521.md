# RATIFY-7 HF Jobs Billing Decision Plan for #523 LL Hinton-Distilled Scorer Surrogate

<!--
RATIFY-7 ratification landing per operator blanket approval 2026-05-21 #7 of 8.

This memo RATIFIES the existing canonical decision-plan memo
`.omx/research/hf_jobs_billing_unblock_523_hinton_surrogate_20260520.md`
(commit-batch-tracked; T1 council PROCEED_WITH_REVISIONS verdict) by
mapping its already-comprehensive 4-branch trade-off analysis onto the
operator's RATIFY-7-prompt-specified 3-branch frame (RECHARGE /
SISTER-PROVIDER MIGRATION / DEFER) and surfacing the canonical
ready-to-paste operator commands per CLAUDE.md "Executing actions with
care" + "Operator gates must be wired and used" non-negotiables.

Per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: this memo does
NOT mutate the 2026-05-20 source memo; it cross-references and ratifies
its content per the RATIFY-7 directive. The source memo's canonical
content remains the authoritative engineering analysis; this memo is
the operator-facing decision-action surface.

Sister memos consulted (cite-chain):
- `.omx/research/hf_jobs_billing_unblock_523_hinton_surrogate_20260520.md` (34.2 KB; canonical engineering analysis + 5-step operator-direct path)
- `.omx/research/hf_jobs_vs_modal_vs_vastai_cost_comparison_20260518.md` (8.9 KB; platform cost table; Insight 5)
- `.omx/research/hf_jobs_substrate_migration_audit_20260518.md` (6.7 KB; Q3 roadmap; defer-path anchor)
- `.omx/research/council_t1_hf_jobs_segnet_surrogate_distillation_symposium_20260519.md` (T1 PROCEED verdict; 5 op-routables)
- `.omx/research/pact_nerv_distilled_scorer_x_codex_ll_integration_design_20260520.md` (sister 3-surface coordination contract)
- `.omx/research/operator_authorizations/hf_jobs_segnet_surrogate_distillation_operator_approval_20260519T070000Z.md` (operator approval anchor)
- `.omx/state/hf_jobs_call_id_ledger.jsonl` row 2026-05-19T17:36:36Z (402 empirical anchor)
- `.omx/operator_authorize_recipes/substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch.yaml` (canonical recipe; dispatch_enabled: true)
-->

---

```yaml
council_tier: T1
council_attendees:
  - Shannon
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "this is a RATIFICATION memo of the 2026-05-20 source memo, which itself was already T1 PROCEED_WITH_REVISIONS pending operator decision; ratifying without a fresh empirical anchor on the HF Hub billing state (which may have changed in the 1-day window since 2026-05-20) carries the same staleness risk the source memo's Contrarian flagged. Recommend op-routable #1 = operator verifies billing dashboard state BEFORE acting on any of the 3 branches per Section 5 Step 1."
council_assumption_adversary_verdict:
  - assumption: "the 2026-05-20 source memo's 4-branch analysis (A HF Jobs recharge / B Modal T4 / C Lightning A100 / D Vast.ai 4090) cleanly maps onto the RATIFY-7 3-branch frame (RECHARGE / SISTER-PROVIDER / DEFER) WITHOUT loss of signal"
    classification: HARD-EARNED
    rationale: "Branch A = RECHARGE; Branches B+C+D collapse into SISTER-PROVIDER (with sub-options); the source memo's 'DEFER (no action)' row at recommendation matrix Section 4 = the third branch. Per CLAUDE.md 'Beauty, simplicity, and developer experience' the 3-branch frame is more operator-routable than the 4-branch frame for this decision-action surface; the 4-branch frame remains the engineering authority for sub-option selection within SISTER-PROVIDER."
  - assumption: "no NEW empirical anchor has landed in the 24-hour window since the source memo (i.e. operator has NOT recharged HF Hub OR migrated to a sister provider OR fired any related dispatch)"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "Verified empirically via canonical ledger surfaces: `.omx/state/hf_jobs_call_id_ledger.jsonl` shows only the 2026-05-19T17:35:02Z intent + 2026-05-19T17:36:36Z 402 failed rows (no subsequent dispatched or harvested rows); no sister-provider dispatch lane for #523 surrogate appears in `.omx/state/modal_call_id_ledger.jsonl` or sister provider ledgers within the 24-hour window."
council_decisions_recorded:
  - "op-routable #1: operator verifies HF Hub billing dashboard at https://huggingface.co/settings/billing per source memo Section 5 Step 1 (5 minutes; $0) BEFORE branching"
  - "op-routable #2: operator selects ONE of the 3 branches (RECHARGE / SISTER-PROVIDER / DEFER) per Section 3 of THIS memo"
  - "op-routable #3: if RECHARGE selected: operator follows source memo Section 5 Steps 3+4+5 per the ready-to-paste commands in Section 3.1 of THIS memo"
  - "op-routable #4: if SISTER-PROVIDER selected: operator decides Modal T4 vs Lightning A100 vs Vast.ai 4090 sub-option per Section 3.2 of THIS memo + commissions sister subagent to land the canonical sister-provider recipe + trainer adapter (~2-4h editor time)"
  - "op-routable #5: if DEFER selected: operator confirms reactivation criteria per Section 3.3 of THIS memo (e.g. 'when Q3 cost-band review lands' OR 'when sister substrate work proves Hinton teacher signal is high-EV')"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: ""
related_deliberation_ids:
  - hf_jobs_billing_unblock_523_hinton_surrogate_20260520
  - council_t1_hf_jobs_segnet_surrogate_distillation_symposium_20260519
  - hf_jobs_vs_modal_vs_vastai_cost_comparison_20260518
  - hf_jobs_substrate_migration_audit_20260518
  - pact_nerv_distilled_scorer_x_codex_ll_integration_design_20260520
horizon_class: apparatus_maintenance
predicted_band: null  # ratification memo of decision-plan; no contest score prediction
predicted_band_provenance: N/A-ratification-memo
predicted_band_validation_status: pending_post_training
deferred_substrate_id: hf_jobs_segnet_surrogate_distillation
substrate_alias: null
---
```

## Section 1 — RATIFY-7 scope + relationship to source memo

Per operator blanket approval 2026-05-21 #7 of 8, this memo prepares the
HF Jobs billing decision plan for task #523 (LL Hinton-distilled scorer
surrogate + saliency-masked residual ($0)). The cascade is BLOCKED on
HF Jobs billing per empirical anchor `.omx/state/hf_jobs_call_id_ledger.jsonl`
row `2026-05-19T17:36:36Z event_type=failed status=failed rc=1
failure_reason="Hugging Face Jobs launch rejected with 402 Payment
Required before returning hf_jobs_id; prepaid credit balance
insufficient."`.

The canonical engineering analysis already exists at
`.omx/research/hf_jobs_billing_unblock_523_hinton_surrogate_20260520.md`
(34.2 KB; commit-batch-tracked yesterday; T1 council
PROCEED_WITH_REVISIONS verdict; 13 sections including 5-step
operator-direct unblock path + alternative-provider trade-off analysis).

Per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE non-negotiable:
this memo does NOT mutate the source memo. It RATIFIES the source memo
by mapping its 4-branch trade-off analysis onto the RATIFY-7-prompt-
specified 3-branch frame and surfaces the canonical ready-to-paste
operator commands so the operator on next engagement can decide +
execute in <30 minutes.

The 3-branch frame consolidation:

| Source memo branch | RATIFY-7 branch |
|---|---|
| Option A (HF Jobs recharge ~$2-5) | **Branch 1 RECHARGE** |
| Option B (Modal T4 ~$2.36 + ~2-4h rewrite) | **Branch 2 SISTER-PROVIDER**, sub-option B |
| Option C (Lightning A100 $0 subscription + ~2-4h rewrite) | **Branch 2 SISTER-PROVIDER**, sub-option C |
| Option D (Vast.ai 4090 ~$1.00 + ~2-4h rewrite) | **Branch 2 SISTER-PROVIDER**, sub-option D |
| "DEFER (no action; revisit per Q3 cost-band review)" | **Branch 3 DEFER** |

No signal loss: the 4-branch engineering analysis remains authoritative
for sub-option selection within Branch 2; the 3-branch frame is the
operator-decision-routable surface.

## Section 2 — Cost + effort summary per branch (for operator decision)

| Branch | Operator $$ | Operator wall-clock | Subagent editor time | First-dispatch cost | Subsequent dispatch cost |
|---|---|---|---|---|---|
| **1 RECHARGE** | $2-5 recharge (operator-direct billing) | ~25 min active + 4-6h dispatch elapsed | 0 (recipe + trainer + ledger all already wired) | ~$3.60-6.60 (recharge + $1.60 dispatch) | ~$1.60 each (amortized) |
| **2 SISTER-PROVIDER (B Modal T4)** | $0 recharge (Modal balance exists per Modal ledger) | ~5 min decision + 4-6h dispatch elapsed | ~2-4h (recipe swap + trainer Modal Volume + HF token in `secrets={...}`) | ~$2.36 dispatch | ~$2.36 each |
| **2 SISTER-PROVIDER (C Lightning A100)** | $0 (subscription-amortized) | ~5 min decision + 4-6h dispatch elapsed | ~2-4h (recipe swap + Lightning dispatcher adapter) | $0 dispatch | $0 each (eats subscription opportunity cost) |
| **2 SISTER-PROVIDER (D Vast.ai 4090)** | $0 recharge (Vast.ai pay-as-you-go) | ~5 min decision + ~50-60 min dispatch elapsed | ~2-4h (recipe swap + Vast.ai lifecycle adapter) | ~$1.00 dispatch (cheapest paid) | ~$1.00 each |
| **3 DEFER** | $0 | 0 active | 0 | N/A (no dispatch) | N/A |

## Section 3 — Ready-to-paste operator commands per branch

### Section 3.1 — Branch 1 RECHARGE ready-to-paste

**Step 1 (operator-only; 5 minutes; $0):** verify billing-state via HF Hub dashboard.

```text
Operator manual step: open https://huggingface.co/settings/billing
in browser; verify prepaid credit balance is < $1.60 (the 402 trigger
threshold per source memo Section 3); verify HF_TOKEN scope includes
inference.serverless.write per https://huggingface.co/settings/tokens.

Per CLAUDE.md "Public Disclosure Hygiene" non-negotiable: dashboard
URLs are operator-facing pointer surfaces ONLY; credit balance values
+ token strings MUST NOT enter the git-tracked tree.
```

**Step 2 (operator-only; 10 minutes; ~$2-5):** recharge HF Hub credit balance.

```text
Operator manual step: at https://huggingface.co/settings/billing click
"Add credits" or equivalent; recommended floor $5.00 to cover first
dispatch ($1.60) + retry headroom + safety margin per source memo
Section 3 "Estimated recharge envelope".
```

**Step 3 (operator-direct CLI; 5 minutes; ~$1.60 dispatch):** fire #523 BUILD_1 via canonical entry point.

```bash
# Per Catalog #199 paired-env discipline + CLAUDE.md "Executing actions with care":
export OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1
export OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=5.00

.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch
```

The canonical entry point invokes (in order) per CLAUDE.md "Operator
gates must be wired and used":

1. Catalog #152 required input file validation
2. Catalog #313 probe-outcomes ledger check (no blocking outcome for `hf_jobs_segnet_surrogate_distillation` per current state)
3. Catalog #243 local pre-deploy harness (8 checks including Catalog #270 dispatch optimization protocol)
4. Catalog #271 codex pre-dispatch review
5. Catalog #325 per-substrate symposium anchor verification (PROCEED verdict from `council_t1_hf_jobs_segnet_surrogate_distillation_symposium_20260519` within 14-day window)
6. Catalog #339 sister fail-closed registration of dispatched `hf_jobs_id` via `tac.deploy.hf_jobs.job_id_ledger.register_dispatched_hf_jobs_id_fail_closed`

**Step 4 (within 24h post-dispatch; $0):** HARVEST OR LOSE per CLAUDE.md sister non-negotiable.

```bash
# After dispatch (within 24h):
.venv/bin/python tools/harvest_hf_jobs_calls.py \
    --hf-jobs-id <hf_jobs_id_from_step_3_output> \
    --update-canonical-ledger
```

### Section 3.2 — Branch 2 SISTER-PROVIDER ready-to-paste (3 sub-options)

Branch 2 requires sub-agent commissioning for the recipe/trainer
adapter; operator selects sub-option B, C, or D first, then commissions
a sister subagent with the operator-routable below.

**Sub-option B Modal T4** (cheapest immediate, existing Modal balance):

```text
Operator-routable to a sister subagent in next engagement:

"Land canonical Modal T4 sister recipe + trainer adapter for #523 LL
Hinton-distilled scorer surrogate. Source: source memo Section 4
Option B at `.omx/research/hf_jobs_billing_unblock_523_hinton_surrogate_20260520.md`.

Deliverables:
1. NEW recipe `.omx/operator_authorize_recipes/substrate_hf_jobs_segnet_surrogate_distillation_modal_t4_dispatch.yaml`
   mirroring the existing HF Jobs recipe with `platform: modal` swap +
   Catalog #244 NVML env block + Catalog #166 sentinel files.
2. EDITS to `experiments/hf_jobs_segnet_surrogate_distillation.py`
   _preprocess callback: replace HF Datasets `load_dataset(adpena/comma-video-segnet-image-level-600pairs)`
   with `huggingface_hub.snapshot_download` into Modal Volume; thread
   HF_TOKEN via `secrets={HF_TOKEN: ...}`.
3. Sister NEW trainer `experiments/modal_segnet_surrogate_distillation.py`
   wrapping the existing logic with the Modal-canonical entry point
   per `experiments/modal_train_lane.py` template.
4. Fire dispatch via the canonical entry point after sister trainer +
   recipe land:

   export OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1
   export OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=5.00
   .venv/bin/python tools/operator_authorize.py \\
       --recipe substrate_hf_jobs_segnet_surrogate_distillation_modal_t4_dispatch

Editor time estimate: ~2-4h. First-dispatch cost: ~$2.36 (Modal T4 at $0.59/hr × 4h)."
```

**Sub-option C Lightning A100** (zero marginal cost, subscription-amortized):

```text
Operator-routable to a sister subagent in next engagement:

"Land canonical Lightning A100 sister recipe + trainer adapter for #523
LL Hinton-distilled scorer surrogate. Source: source memo Section 4
Option C.

Deliverables:
1. NEW recipe `.omx/operator_authorize_recipes/substrate_hf_jobs_segnet_surrogate_distillation_lightning_a100_dispatch.yaml`
   with `platform: lightning` + Catalog #143 sister register-before-submit pattern.
2. Sister NEW trainer using `scripts/launch_lane_lightning.py` dispatcher
   per Catalog #143 sister.
3. Fire dispatch via canonical entry point after sister trainer + recipe land.

Editor time estimate: ~2-4h. First-dispatch cost: $0 (subscription-amortized);
opportunity cost: ~4 GPU-hours of long-burn substrate time."
```

**Sub-option D Vast.ai 4090** (cheapest paid + fastest wall-clock):

```text
Operator-routable to a sister subagent in next engagement:

"Land canonical Vast.ai 4090 sister recipe + trainer adapter for #523
LL Hinton-distilled scorer surrogate. Source: source memo Section 4
Option D.

Deliverables:
1. NEW recipe `.omx/operator_authorize_recipes/substrate_hf_jobs_segnet_surrogate_distillation_vastai_4090_dispatch.yaml`
   with `platform: vastai` + Catalog #225 lifecycle (claim_lane_dispatch
   + vastai_active_instances.json) + Catalog #288 cu124 driver pin per
   CLAUDE.md "Forbidden uv torch install without driver-version pin"
   non-negotiable.
2. Sister NEW trainer using `scripts/launch_lane_on_vastai.py` dispatcher.
3. Fire dispatch via canonical entry point after sister trainer + recipe land.

Editor time estimate: ~2-4h. First-dispatch cost: ~$1.00 (Vast.ai 4090
at $0.25/hr × 4h; cheapest paid option). Wall-clock ~50-60 min vs T4 ~4h
per CLAUDE.md 'Optimal GPU: RTX 4090 on Vast.ai' canonical anchor."
```

### Section 3.3 — Branch 3 DEFER ready-to-paste

```text
Operator manual decision (no command needed):

Per CLAUDE.md "Forbidden premature KILL without research exhaustion"
non-negotiable + Catalog #298 substrate retirement discipline:

#523 substrate `hf_jobs_segnet_surrogate_distillation` is L1 SCAFFOLD
+ recipe `dispatch_enabled: true` + T1 symposium PROCEED. DEFERRING is
distinct from KILLING; the lane remains LIVE pending the reactivation
criterion.

Suggested reactivation criteria (operator selects one or composes):

(a) "DEFER-pending-HF-Jobs-billing-resolved": revisit when operator
    chooses to recharge HF Hub credit balance (e.g. next billing cycle,
    other HF Jobs use case justifies the recharge, etc.)

(b) "DEFER-pending-PACT-NERV-DistilledScorer-empirical-anchor": per
    sister memo `.omx/research/pact_nerv_distilled_scorer_x_codex_ll_integration_design_20260520.md`
    the INSIDE-DECODER surrogate is sister lane Surface B; if Surface B
    lands an empirical anchor with measurable score improvement, the
    additional Surface C IMAGE-LEVEL surrogate may be deferred or
    de-prioritized as Surface B has already captured the canonical
    Quantizr KL-T=2.0 0.33 [contest-CUDA] technique.

(c) "DEFER-pending-Q3-cost-band-reevaluation": per
    `.omx/research/hf_jobs_substrate_migration_audit_20260518.md` Q3
    roadmap; revisit when sister substrates land sufficient empirical
    anchors to re-rank HF Jobs vs Modal vs Vast.ai cost-band priorities.

(d) "DEFER-pending-codex-LL-saliency-residual-validation": #523 task
    description includes "+ saliency-masked residual ($0)" sub-deliverable
    which is the Codex LL `scorer_response_dataset` surface; if that
    surface independently produces sufficient signal for downstream
    substrate trainers, the heavier image-level surrogate may be deferred.

Sister tasks that can advance INDEPENDENTLY while #523 is DEFERRED:

- #876 HF dataset prep (already complete per source memo §2: dataset
  exists at `adpena/comma-video-segnet-image-level-600pairs` sha
  `52ef7313ed2cb6f84e9635cd99bd9b51bc1ecd9a`)
- #877 HF skills research (complete per `huggingface_skills_comprehensive_design_implementation_plan_20260518.md`)
- PACT-NERV-DistilledScorer sister Surface B (L1 SCAFFOLD per source
  memo §6; can advance to L2 INTEGRATION independently)
- Codex LL `scorer_response_dataset` Surface A (29-row dataset complete
  per sister memo; can advance to consumer-wire-in independently)

Operator action: update task #523 description to add DEFERRED status +
reactivation criterion + sister-task continuation list. NO subagent
mutation of task description per CLAUDE.md "Executing actions with care";
operator updates the task on their canonical task-management surface.
```

## Section 4 — Recommended branch + rationale

**Recommendation**: **Branch 1 RECHARGE** (HF Jobs ~$2-5 recharge).

**Rationale** (per cumulative CLAUDE.md non-negotiables):

1. **Canonical-path purity**: per the T1 symposium PROCEED verdict +
   operator approval 2026-05-19T07:00Z "all operator routable items
   approved", HF Jobs T4 IS the canonical platform for this surrogate.
   Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode":
   the canonical contract is the trainer + recipe + symposium-PROCEED
   bound together at HF Jobs.
2. **Minimal-rewrite cost**: 0 editor time vs ~2-4h for Branch 2 sub-
   options. The 2026-05-19 trainer + recipe + ledger + dispatcher are
   already wired per source memo §2; only the billing state blocks.
3. **Operator-attention-threshold-safe**: ~$2-5 recharge is well within
   the operator-attention threshold per CLAUDE.md "GPU budget and
   compute resources" non-negotiable ($24 Vast.ai cap is the standing
   ceiling; this is well under).
4. **Amortized cost wins**: subsequent dispatches cost $1.60 each on
   HF Jobs vs $2.36 each on Modal; if even 2 dispatches are expected,
   HF Jobs is cheaper total ($3.20 vs $4.72).
5. **Race-mode prep**: per CLAUDE.md "Race-mode rigor inversion +
   parallel-dispatch first" non-negotiable, having HF Jobs already
   funded preserves a parallel-actuator surface independent of Modal
   for race-window fan-out.

**Counter-recommendation if operator prefers $0**: Branch 2 sub-option
C **Lightning A100** (subscription-amortized $0). The ~2-4h editor time
is the tradeoff for $0 marginal cost; if the operator already plans
sister Lightning A100 long-burn campaigns and would otherwise lose the
GPU-hours to idle, sub-option C is structurally free.

**Counter-recommendation if operator prefers DEFER**: per Catalog #298
substrate retirement discipline, DEFERRED-pending-reactivation is a
fully canonical state for L1 substrates. Per the sister coordination
contract in source memo §6 + sister `pact_nerv_distilled_scorer_x_codex_ll_integration_design_20260520`,
the cascade has THREE orthogonal surfaces and Surface B
(PACT-NERV-DistilledScorer inside-decoder Conv2d on Modal) can advance
independently. If Surface B yields sufficient empirical anchor, the
heavier image-level Surface C (#523) may be DEFERRED indefinitely
without paradigm loss.

## Section 5 — Cargo-cult audit per Catalog #303 (3 assumptions)

| Assumption | Classification | Unwind path |
|---|---|---|
| Source memo's 4-branch analysis cleanly maps onto RATIFY-7 3-branch frame WITHOUT signal loss | HARD-EARNED | Verified by per-branch table at Section 1: every source memo branch has explicit RATIFY-7 mapping; sub-options within Branch 2 preserve B/C/D engineering distinctions |
| No NEW empirical anchor (operator action / sister dispatch / billing state change) has landed in the 24-hour window since the 2026-05-20 source memo | HARD-EARNED-EMPIRICALLY-VERIFIED | `.omx/state/hf_jobs_call_id_ledger.jsonl` shows only the 2026-05-19T17:36:36Z 402 row + no subsequent rows; canonical Modal ledger has no #523-related sister-provider dispatch row |
| Branch 1 RECHARGE recommendation is HARD-EARNED at engineering surface BUT operator may have non-engineering preferences (budget, billing-account consolidation, sister-provider strategy) that this memo cannot anticipate | HARD-EARNED-WITH-OPERATOR-DECISION-NEEDED | Operator selects branch via op-routable #2; all 3 branches are canonically supported |

## Section 6 — 9-dimension success checklist evidence per Catalog #294

1. **UNIQUENESS**: NEW RATIFY-7 ratification memo collapsing source 4-branch to operator-routable 3-branch frame + canonical ready-to-paste commands; distinct from source memo (engineering analysis) and sister cost-comparison memo (platform cost table).
2. **BEAUTY+ELEGANCE**: ready-to-paste commands in Section 3 are operator-routable in <30s reviewability per PR101-style canonical pattern; recommendation matrix in Section 4 is single-sentence-per-branch.
3. **DISTINCTNESS**: Slot RATIFY-7 distinct from sister Slots 1+3+5 (Slot 1 Carmack T3 council; Slot 3 NSCS06 v8 binding revisions; Slot 5 disjoint per pre-flight check).
4. **RIGOR**: Catalog #229 PV (read source memo 466 lines + sister cost memo + recipe + 402 ledger + canonical frontier pointer + AGENTS.md + CLAUDE.md head) + Catalog #292 per-deliberation assumption surfacing in frontmatter.
5. **OPTIMIZATION-PER-TECHNIQUE**: per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD — the operator decision IS the optimization surface; this memo's job is to make decision-cost minimal (~5 min review + ~5 min decision + 0 subagent rewrite if Branch 1).
6. **STACK-OF-STACKS-COMPOSABILITY**: defers to source memo §8 composition-alpha estimates; this ratification preserves those estimates.
7. **DETERMINISTIC-REPRODUCIBILITY**: every command in Section 3 is canonical CLI invocation per Catalog #199 + #176 + #243 + #271; reproducible byte-for-byte.
8. **EXTREME-OPTIMIZATION-PERFORMANCE**: defers to source memo §10 per-platform cost-per-hour table; recommendation Branch 1 is cheapest amortized.
9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: defers to source memo §10; #523 surrogate output is `[predicted]` advisory-only; downstream score impact via Hinton teacher signal per Catalog #324 `predicted_band_validation_status: pending_post_training`.

## Section 7 — Observability surface per Catalog #305

1. **Inspectable per layer**: this memo + source memo + sister memos all carry v2 frontmatter per Catalog #300; operator decision per Section 4 surfaces in `tac.council_continual_learning` posterior anchor (op-routable #6 below).
2. **Decomposable per signal**: 3-branch frame separation per Section 3; per-branch cost/effort separation per Section 2.
3. **Diff-able across runs**: this memo references source memo by exact path + line counts; sister memos referenced by date suffix.
4. **Queryable post-hoc**: source memo's HF Jobs canonical ledger at `.omx/state/hf_jobs_call_id_ledger.jsonl` queryable per source memo §11 (4 query helpers).
5. **Cite-able**: every recommendation cites a CLAUDE.md non-negotiable section by name.
6. **Counterfactual-able**: source memo's dispatcher `--dry-run` mode per source memo §11 preserved.

## Section 8 — 6-hook wire-in declaration per Catalog #125

- **Hook #1 sensitivity-map contribution** = N/A (ratification + decision-plan memo; no signal contribution at this surface)
- **Hook #2 Pareto constraint** = N/A
- **Hook #3 bit-allocator hook** = N/A
- **Hook #4 cathedral autopilot dispatch hook** = **ACTIVE** (post-decision Branch 1 / Branch 2 dispatch produces empirical anchor consumable by autopilot ranker per `distilled_scorer_surrogate_canonical_equation_consumer` sister-lane BUILD per source memo §13)
- **Hook #5 continual-learning posterior update** = **ACTIVE** (post-dispatch empirical anchor appends to canonical equations registry via `tac.canonical_equations.update_equation_with_empirical_anchor` per Catalog #344 if source memo op-routable #5 lands)
- **Hook #6 probe-disambiguator** = **ACTIVE** (THIS memo's 3-branch frame IS the disambiguator between (1) RECHARGE / (2) SISTER-PROVIDER / (3) DEFER for the operator decision surface)

## Section 9 — Sister-collision verdict + sister-DISJOINT scope

Verified sister-DISJOINT per RATIFY-7 prompt-specified sister tracking:

- **Slot 1** (commit `a0d3683932` Carmack T3 council): touches `.omx/research/council_t3_carmack_*` + `.omx/state/council_deliberation_posterior.jsonl`; THIS memo touches `.omx/research/hf_jobs_billing_decision_plan_20260521.md` (NEW) — fully DISJOINT.
- **Slot 3** (commit `a9165f1a83` NSCS06 v8 binding revisions): touches `src/tac/substrates/nscs06_v8_chroma_lut/` files; fully DISJOINT.
- **Sister 2026-05-20 source memo** (`hf_jobs_billing_unblock_523_hinton_surrogate_20260520.md`): SISTER, not collision; THIS memo cross-references the source memo per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE without mutating it.

Per Catalog #340 sister-checkpoint guard pre-flight: NO sister subagents
in-flight overlap with target file `.omx/research/hf_jobs_billing_decision_plan_20260521.md`.

## Section 10 — Operator-routable summary (for next-engagement decision)

| Op-routable | Action | Estimated cost | Estimated wall-clock |
|---|---|---|---|
| #1 | Verify HF Hub billing dashboard state per Section 3.1 Step 1 | $0 | 5 min |
| #2 | Select Branch 1 / Branch 2 / Branch 3 per Section 4 recommendation | $0 | 5 min |
| #3 | If Branch 1: recharge + fire dispatch + harvest per Section 3.1 Steps 2+3+4 | ~$3.60-6.60 | ~25 min active + 4-6h dispatch elapsed |
| #4 | If Branch 2: commission sister subagent with operator-routable per Section 3.2 sub-option B/C/D | $0-2.36 | ~5 min decision + ~2-4h sister editor + 4-6h dispatch elapsed |
| #5 | If Branch 3: update task #523 description with DEFERRED status + reactivation criterion per Section 3.3 | $0 | 5 min |
| #6 | Per CLAUDE.md "Council hierarchy: 4-tier protocol" Maximum signal preservation rule: emit canonical posterior anchor via `tac.council_continual_learning.append_council_anchor` after decision lands (THIS memo's frontmatter PROCEED_WITH_REVISIONS will be ratified to PROCEED-unconditional by operator decision per Catalog #315 OPTIMAL FORM discipline) | $0 | automatic on next sister landing |

## Cross-references

- Source memo: `.omx/research/hf_jobs_billing_unblock_523_hinton_surrogate_20260520.md` (34.2 KB; THE engineering analysis)
- Cost-comparison memo: `.omx/research/hf_jobs_vs_modal_vs_vastai_cost_comparison_20260518.md` (8.9 KB; platform $/hr table)
- Migration audit: `.omx/research/hf_jobs_substrate_migration_audit_20260518.md` (6.7 KB; Q3 roadmap; defer-path anchor)
- T1 symposium: `.omx/research/council_t1_hf_jobs_segnet_surrogate_distillation_symposium_20260519.md` (PROCEED with 5 op-routables)
- Sister 3-surface design: `.omx/research/pact_nerv_distilled_scorer_x_codex_ll_integration_design_20260520.md`
- Operator approval anchor: `.omx/research/operator_authorizations/hf_jobs_segnet_surrogate_distillation_operator_approval_20260519T070000Z.md`
- HF skills design: `.omx/research/huggingface_skills_comprehensive_design_implementation_plan_20260518.md` (85.8 KB; full skills plan)
- Recipe: `.omx/operator_authorize_recipes/substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch.yaml` (9.4 KB; dispatch_enabled: true; research_only: false)
- 402 empirical anchor: `.omx/state/hf_jobs_call_id_ledger.jsonl` row 2026-05-19T17:36:36Z
- Canonical entry point: `tools/operator_authorize.py`
- Canonical HF Jobs dispatcher: `tools/dispatch_hf_jobs_vision_training.py` (17.8 KB)
- Canonical HF Jobs ledger helper: `src/tac/deploy/hf_jobs/job_id_ledger.py` (31.5 KB)
- Catalog gates: #110 + #113 + #125 + #131 + #138 + #143 + #152 + #176 + #199 + #225 + #229 + #243 + #245 + #270 + #271 + #287 + #288 + #290 + #292 + #294 + #298 + #300 + #303 + #305 + #315 + #322 + #323 + #324 + #325 + #335 + #339 + #340 + #341 + #344 + #346
