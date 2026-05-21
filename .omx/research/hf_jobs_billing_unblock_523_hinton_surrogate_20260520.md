# HF Jobs Billing Unblock for #523 Hinton-Distilled Scorer Surrogate Lane

<!--
Catalog #344 canonical equation registry cross-reference:
This memo cross-references the NEW canonical equation design spec
`kl_distillation_scorer_surrogate_compression_savings_v1` from the sister
landing memo `pact_nerv_distilled_scorer_x_codex_ll_integration_design_20260520.md`
(commit f39d6f6ce). THIS memo does NOT register the equation — registration
is OPERATOR-ROUTED per Catalog #344 EVOLUTION DISCIPLINE per the sister
memo's op-routable #1. THIS memo focuses on the HF-Jobs-billing-unblock
sub-surface which is necessary BEFORE the equation's image-level
empirical anchor can land via #523 BUILD_1.

Sister equations consulted: `categorical_posterior_capacity_vs_continuous_gaussian_v1`
(registered 2026-05-20T13:21Z), `procedural_codebook_from_seed_compression_savings_v1`
(registered earlier), `brotli_cascade_bounded_per_stream_v1` (registered 2026-05-19T13:42Z).
-->

---

```yaml
council_tier: T1
council_attendees:
  - Shannon       # information-theory grounding for KL distillation cost vs benefit
  - Dykstra       # alternating-projections feasibility for HF Jobs vs Modal vs Vast.ai cost polytope
  - Rudin         # interpretable-ML lens on operator-direct billing-decision steps
  - Daubechies    # multi-scale lens: per-substrate vs per-platform vs per-axis
  - Hinton        # CANONICAL inner-council seat: KL-T=2.0 distillation (1503.02531) — the technique at risk
  - Yousfi        # scorer-design lens: image-level vs per-pixel SegNet surrogate axis
  - Quantizr      # 0.33 [contest-CUDA] empirical anchor lens
  - Contrarian
  - Assumption-Adversary
  - PR95Author    # canonical HNeRV substrate + leaderboard-implementation parity
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "the 5-step operator-direct unblock path is engineering-sound BUT the 2026-05-19T17:36:36Z 402 Payment Required event is a 1-day-old empirical anchor that has NOT been re-tested; the prepaid credit balance state at the time of any future operator decision MAY differ from the time of this memo. Recommend op-routable #1: ratify via a $0 cost-band cheap probe (huggingface_hub.HfApi().run_uv_job(...) dry-run + balance check) BEFORE operator commits to recharge dollars."
  - member: Assumption-Adversary
    verbatim: "the assumption that 'HF Jobs is the canonical platform for image-level scorer surrogate distillation per the T1 working group symposium 2026-05-19 verdict PROCEED' is HARD-EARNED-EMPIRICALLY-VERIFIED (the symposium council voted PROCEED with 10 assumptions classified per Catalog #303 + the operator approved 'all operator routable items' 2026-05-19T07:00Z verbatim) — but the ALTERNATIVE-PROVIDER assumption (that Modal/Lightning/Vast.ai could ALSO host this surrogate dispatch with equivalent or better cost-band) is CARGO-CULTED-EMPIRICALLY-UNTESTED at the per-platform image-classification-template surface. The HF Jobs t4-small $0.40/hr advantage is HARD-EARNED-AT-THE-COST-COMPARISON-MEMO-SURFACE; whether equivalent dispatch produces equivalent surrogate quality on Modal/Vast.ai is unknown."
council_assumption_adversary_verdict:
  - assumption: "HF Jobs 402 Payment Required is the actual blocker (not an upstream Hugging Face Hub API quota issue, not a HF token scope issue, not an organization billing issue)"
    classification: HARD-EARNED
    rationale: "Ledger row at 2026-05-19T17:36:36Z explicitly records `failure_reason='Hugging Face Jobs launch rejected with 402 Payment Required before returning hf_jobs_id; prepaid credit balance insufficient.'` per `src/tac/deploy/hf_jobs/job_id_ledger.py` schema_version `hf_jobs_ledger_v1_catalog342_20260519`. The 402 status code is HTTP-standard for 'Payment Required' and Hugging Face Jobs canonical practice maps it to prepaid credit balance issues."
  - assumption: "Operator-direct billing-decision is the canonical unblock path (not autopilot-cascade-decision per Catalog #325 + #270 dispatch optimization protocol)"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'Executing actions with care' + 'Public Disclosure Hygiene' (operator-only for credential operations) + 'GPU budget and compute resources' non-negotiable (operator-only for budget caps). The 402 event involves real-money recharge at Hugging Face Hub which is operator-only territory."
  - assumption: "The HF Jobs surrogate is COMPOSITION-ORTHOGONAL with PACT-NERV-DistilledScorer's inside-decoder surrogate (i.e. the two surrogates are NOT redundant signal)"
    classification: HARD-EARNED
    rationale: "Per the sister landing memo `pact_nerv_distilled_scorer_x_codex_ll_integration_design_20260520.md` Section 4 (commit f39d6f6ce): #523 trains IMAGE-LEVEL mobilenetv3_small 2.5M params on full 600-pair contest video; PACT-NERV-DistilledScorer trains INSIDE-DECODER Conv2d 10k params simultaneously with HNeRV decoder. Different parameter counts (250x), different signal axes (image-level top-1 vs per-decoder-block channel-bias), different platforms (HF Jobs vs Modal). The structural orthogonality is sound per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD non-negotiable."
council_decisions_recorded:
  - "op-routable #1: operator-verify HF Jobs account billing dashboard at https://huggingface.co/settings/billing AND prepaid credit balance state BEFORE committing recharge dollars"
  - "op-routable #2: operator-direct decision — choose between (Option A) HF Jobs recharge ~$2-5 / (Option B) sister-provider migration (Modal $1.6 or Vast.ai $0.5-1) / (Option C) DEFER-pending-Q3-cost-band-reevaluation per the hf_jobs_substrate_migration_audit_20260518 roadmap"
  - "op-routable #3: if Option A or Option B selected, fire #523 BUILD_1 dispatch via `tools/operator_authorize.py --recipe substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch` (canonical entry point per Catalog #176 + #243 + #271 + #270 dispatch optimization protocol)"
  - "op-routable #4: post-dispatch HARVEST OR LOSE per CLAUDE.md 'Modal `.spawn()` HARVEST OR LOSE' sister non-negotiable for HF Jobs — register dispatched hf_jobs_id in `.omx/state/hf_jobs_call_id_ledger.jsonl` via `tac.deploy.hf_jobs.job_id_ledger.register_dispatched_hf_jobs_id_fail_closed` per Catalog #339 sister discipline"
  - "op-routable #5: register the NEW canonical equation `kl_distillation_scorer_surrogate_compression_savings_v1` from the sister landing memo's op-routable #1 IFF #523 BUILD_1 lands an empirical anchor (per Catalog #344 EVOLUTION DISCIPLINE)"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: ""
related_deliberation_ids:
  - pact_nerv_distilled_scorer_x_codex_ll_integration_design_20260520
  - council_t1_hf_jobs_segnet_surrogate_distillation_symposium_20260519
  - hf_jobs_segnet_surrogate_distillation_operator_approval_20260519T070000Z
  - hf_jobs_vs_modal_vs_vastai_cost_comparison_20260518
  - hf_jobs_substrate_migration_audit_20260518
horizon_class: frontier_pursuit
predicted_band: "[-0.003, +0.001]"
predicted_band_provenance: HARD-EARNED-via-Quantizr-0.33-anchor-AND-sister-pact-nerv-distilled-scorer-design-memo
predicted_band_validation_status: pending_post_training
deferred_substrate_id: hf_jobs_segnet_surrogate_distillation
substrate_alias: null
---
```

## Section 1 — Executive summary

#523 long-pending canonical lane (Catalog #523 L2 Hinton-distilled SegNet
surrogate Phase 1 BUILD; TaskCreate #875) is **BLOCKED 2026-05-19T17:37Z
on HF Jobs 402 Payment Required**. The blocker is structural (prepaid
credit balance insufficient at the Hugging Face Jobs platform) AND
operator-direct (real-money recharge is operator-only territory per
CLAUDE.md "Executing actions with care" + "GPU budget and compute
resources").

The #523 trainer (`experiments/hf_jobs_segnet_surrogate_distillation.py`,
359 LOC) is **impl_complete + T1 symposium PROCEED + recipe
`dispatch_enabled: true`** per the operator approval memo at
`.omx/research/operator_authorizations/hf_jobs_segnet_surrogate_distillation_operator_approval_20260519T070000Z.md`.
Every engineering surface is ready. The ONLY blocker is the platform
billing state.

THIS memo lands the canonical operator-direct 5-step unblock path PLUS
the alternative-provider trade-off analysis (HF Jobs recharge vs Modal
CPU vs Lightning Studios vs Vast.ai 4090). Sister-DISJOINT from the
in-flight 5-SUBSTRATE MATRIX SUPERSESSION (`a4b9500c`) + MAGIC CODEC
PAIR #4 ORTHOGONALITY SMOKE (`aae30921`) per the operator-mandated
scope-coherence rule.

Per the cascade-direction pivot to canonical IN-DOMAIN substrates
including KL distillation lane per the parser-safe subset smoke landing
(`e3e198c9f`), the #523 lane is the canonical IMAGE-LEVEL Hinton-distilled
SegNet surrogate axis matching the Quantizr KL-T=2.0 0.33 [contest-CUDA]
canonical technique per CLAUDE.md "Quantizr intelligence" + Hinton inner-
council seat per CLAUDE.md "Council conduct".

## Section 2 — #523 trainer impl_complete verification

| Artifact | Status | Verification |
|---|---|---|
| `experiments/hf_jobs_segnet_surrogate_distillation.py` | impl_complete | 359 LOC; complete `main()`; no `NotImplementedError`; PEP 723 inline metadata + `HfArgumentParser` + `Trainer` per plugin canonical template directives #1-#4 |
| `.omx/operator_authorize_recipes/substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch.yaml` | `dispatch_enabled: true` AND `research_only: false` | Per `hf_jobs_segnet_surrogate_distillation_operator_approval_20260519T070000Z.md` 2026-05-19T07:00Z |
| Council T1 symposium memo | PROCEED with 5 op-routables | `.omx/research/council_t1_hf_jobs_segnet_surrogate_distillation_symposium_20260519.md`; 6-step Catalog #325 contract complete |
| `tools/dispatch_hf_jobs_vision_training.py` | impl_complete | 17.8 KB canonical dispatcher; wraps `huggingface_hub.HfApi().run_uv_job(...)` per plugin canonical pattern; Catalog #245 4-layer sister discipline |
| HF Hub dataset | exists at `adpena/comma-video-segnet-image-level-600pairs` | sha `52ef7313ed2cb6f84e9635cd99bd9b51bc1ecd9a` per operator approval memo §"Cite-chain" |
| Catalog #339 sister fail-closed registration helper | impl_complete | `tac.deploy.hf_jobs.job_id_ledger.register_dispatched_hf_jobs_id_fail_closed` (sister of Catalog #339 Modal `register_dispatched_call_id_fail_closed`) |
| Lane registry | L1 (impl_complete + memory_entry) | `lane_hf_jobs_segnet_surrogate_distillation_20260519` |

Conclusion: **EVERY engineering surface is ready**; the ONLY blocker is
the platform billing state.

## Section 3 — HF Jobs billing audit (the 402 event)

### Empirical anchor from canonical HF Jobs ledger

```
File: .omx/state/hf_jobs_call_id_ledger.jsonl
Row 1: 2026-05-19T17:35:02Z event_type=intent label=substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch_20260519T173502Z
       expected_cost_usd=1.6, flavor=t4-small, hub_dataset_repo=adpena/comma-video-segnet-image-level-600pairs
Row 2: 2026-05-19T17:36:36Z event_type=failed status=failed rc=1
       failure_reason="Hugging Face Jobs launch rejected with 402 Payment Required before returning hf_jobs_id;
                       prepaid credit balance insufficient."
       evidence_grade=remote_hf_jobs_launch_failed_before_job_id
       cost_actual_usd=0.0
```

### Estimated recharge envelope per cost-comparison memo

Per `.omx/research/hf_jobs_vs_modal_vs_vastai_cost_comparison_20260518.md` §"HF Jobs vs Modal head-to-head":

| Parameter | Value |
|---|---|
| HF Jobs t4-small | $0.40/hr |
| Job timeout | 4 hours (recipe `cost_band.epochs=200` with `max_seconds=14400`) |
| Expected cost (intent row) | $1.60 (`expected_cost_usd: 1.6`) |
| Safety margin (retry + Hub push + cold-start) | +25% |
| **Recommended recharge floor** | **$2.00 minimum; $5.00 comfortable** |
| Operator budget envelope per CLAUDE.md "GPU budget" | within $24 Vast.ai cap; well under operator-attention threshold ($5/dispatch) |

### Billing-state diagnostic surface (operator-direct)

To verify the 402 root cause is account-level (not project / token /
organization scope), the operator can inspect:

1. **HF Hub billing dashboard**: https://huggingface.co/settings/billing
   - Prepaid credit balance (canonical 402 source)
   - Active subscription tier (Pro / Enterprise; subscription includes
     limited free Jobs hours)
   - Recent jobs history (verify no other 402 events on same account)
2. **HF token scope**: https://huggingface.co/settings/tokens
   - Verify the `HF_TOKEN` environment variable in operator shell has
     `write` AND `inference.serverless.write` scopes (not just `read`)
3. **HF Hub API rate limit**: free-tier API limit is `1000 requests/hour`;
   verify no recent rate-limit signals in operator session

Per CLAUDE.md "Public Disclosure Hygiene" non-negotiable: token values
+ billing dashboard URLs + credit balance numbers MUST NOT enter the
git-tracked tree. THIS memo intentionally only references the dashboard
URLs as operator-facing pointer surfaces.

## Section 4 — Alternative-provider trade-off analysis

Three primary alternatives if HF Jobs recharge is undesirable:

### Option A: HF Jobs recharge (canonical path)

| Pro | Con |
|---|---|
| Canonical T1 symposium PROCEED verdict at this platform | Real-money recharge required (~$2-5) |
| Plugin canonical template (`huggingface-skills:hugging-face-vision-trainer`) already wired | HF Jobs cold-start ~30-60s (vs Modal ~10-20s) |
| HF Datasets-native integration (no `upstream/videos/` mount via Catalog #152) | Less canonical-pattern-mass than Modal in our codebase |
| Catalog #339 sister fail-closed helper already exists | 30% cheaper than Modal t4 ($0.40 vs $0.59/hr) **only if recharge cost is amortized over multiple dispatches** |
| Hub-native model push (`adpena/segnet-image-level-surrogate-mobilenet-v3-small-200ep`) | First-dispatch case: 30% saving = $0.19/hr * 4h = $0.76 saving vs Modal — INSUFFICIENT to justify recharge dollars alone |

**Operator-direct cost**: ~$2-5 recharge + $1.60 expected dispatch =
~$3.60-6.60 total for first dispatch; subsequent dispatches amortize
to $1.60 each.

### Option B: Sister-provider migration to Modal CPU/T4 (immediate)

Per `.omx/research/hf_jobs_vs_modal_vs_vastai_cost_comparison_20260518.md`
§"Modal advantages":

| Pro | Con |
|---|---|
| **No recharge required** — Modal account has existing balance per `.omx/state/modal_call_id_ledger.jsonl` recent dispatches | 47% more expensive ($0.59 vs $0.40/hr); ~$2.36/dispatch vs ~$1.60 |
| **Canonical dispatch path** with Catalog #244 NVML + Catalog #166 HEAD parity + Catalog #245 call_id ledger | Trainer rewrite required: `experiments/hf_jobs_segnet_surrogate_distillation.py` uses HF Datasets `load_dataset(adpena/comma-video-segnet-image-level-600pairs)` which requires HF Hub access from Modal worker (HF token in `secrets`) |
| **Faster cold-start** (~10-20s vs HF Jobs ~30-60s) | Modal `.spawn()` 24h result-cache TTL constraint per CLAUDE.md non-negotiable; need HARVEST OR LOSE within 24h |
| **53+ substrate trainers already Modal-wired** — sister-trainer template exists | Some image-classification template adaptation work — ~2-4h editor time |
| **CPU-only Modal dispatch is viable** (mobilenetv3_small 2.5M params, batch-size 32, 200ep takes ~16h CPU vs ~4h T4) — would be FREE on Modal CPU | CPU-only path is `[macOS-CPU advisory]` per CLAUDE.md "MPS auth eval is NOISE" non-negotiable; downstream Hinton teacher signal MAY suffer from CPU-vs-CUDA forward drift |

**Sister-trainer template anchor**: `experiments/modal_train_lane.py` is
the canonical Modal-T4 wrapper; `tools/operator_authorize.py::_dispatch_modal`
is the canonical entry point per Catalog #176 + #243 + #271 dispatch
optimization protocol. Adapter cost: rewrite `_preprocess` callback in
`experiments/hf_jobs_segnet_surrogate_distillation.py` to fetch dataset
via `huggingface_hub.snapshot_download` into Modal Volume + apply same
`_reduce_mask_to_image_level_class` reduction. Sister recipe:
`.omx/operator_authorize_recipes/substrate_hf_jobs_segnet_surrogate_distillation_modal_t4_dispatch.yaml`
(NEW; mirrors existing recipe with `platform: modal` swap).

**Operator-direct cost**: $0 recharge + $2.36 dispatch = $2.36 total for
first dispatch (recharge saved).

### Option C: Lightning Studios A100 (subscription-amortized)

| Pro | Con |
|---|---|
| **$0 marginal cost** (eats existing Lightning subscription per CLAUDE.md "GPU budget" §"Lightning A100 24/7") | Over-provisioned: A100 80 GB for a 2.5M-param mobilenetv3_small is GPU-overkill |
| Already-wired dispatch path (`scripts/launch_lane_lightning.py` per Catalog #143 sister) | Lightning A100 dispatch lifecycle (Catalog #140 sister state-writer-own-lock discipline) is less mature than Modal/HF Jobs for image-classification workflows |
| **Long-burn-compatible** for follow-up Phase 2 per-pixel mIoU sister lane | Lightning's subscription is bounded; using it for a tiny mobilenetv3_small consumes opportunity cost for long-burn substrate campaigns |

**Operator-direct cost**: $0 recharge + $0 dispatch (subscription-amortized)
= $0 total; opportunity cost = ~4 GPU-hours of long-burn substrate time.

### Option D: Vast.ai 4090 (cheap-and-fast)

Per CLAUDE.md "GPU budget and compute resources" canonical anchor:

| Pro | Con |
|---|---|
| **Cheapest GPU/hr in market** ($0.25/hr) — total ~$1.00 for 4h training | Vast.ai for image-classification template requires manual lifecycle (Catalog #225 `claim_lane_dispatch` + `vastai_active_instances.json`) |
| **4-5x faster than T4** for our workload (4090 has more SMs + higher mem-bandwidth) — actual dispatch wall-clock ~50-60min | RTX 4090 24 GB VRAM is over-provisioned for 2.5M-param model (T4 16 GB sufficient) |
| **1:1 contest-compliant CUDA hardware** per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" — though not contest-relevant for THIS surrogate which is `[predicted]` advisory only | CUDA driver version drift (cu13 vs cu124) per CLAUDE.md "Forbidden uv torch install without driver-version pin" non-negotiable |
| | Trainer rewrite required (similar magnitude to Option B Modal) |

**Operator-direct cost**: $0 recharge + $1.00 dispatch = $1.00 total
(cheapest paid option).

### Recommendation matrix

| Operator preference | Recommended option |
|---|---|
| Canonical-path purity + minimal-rewrite | **Option A** (HF Jobs recharge); ~$2-5 recharge floor |
| Cheapest immediate path with existing balance | **Option B** (Modal T4); $2.36 dispatch, ~2-4h editor rewrite |
| Zero-marginal-cost (subscription-amortized) | **Option C** (Lightning A100); $0 dispatch, ~2-4h editor rewrite |
| Cheapest paid option + fastest wall-clock | **Option D** (Vast.ai 4090); $1.00 dispatch, ~2-4h editor rewrite |
| DEFER until Q3 cost-band review | DEFER (no action); per `hf_jobs_substrate_migration_audit_20260518` Q3 roadmap |

**Default operator-routable recommendation (per CLAUDE.md "Long-burn
score-lowering campaign default" non-negotiable + cascade-direction
pivot to canonical IN-DOMAIN substrates including KL distillation lane
per parser-safe subset smoke landing `e3e198c9f`)**: **Option A or
Option C**. Option A preserves canonical T1 symposium path + plugin-
template integrity; Option C preserves $0 marginal cost. Option B
introduces rewrite drift; Option D introduces lifecycle drift.

## Section 5 — Operator-direct 5-step unblock path

### Step 1: Verify billing-state diagnostic (5 minutes; $0)

Operator inspects HF Hub billing dashboard at
https://huggingface.co/settings/billing and verifies the 402 root cause
is account-level prepaid credit balance (NOT token scope, NOT
organization billing, NOT rate-limit). Cite verbatim CLAUDE.md "Public
Disclosure Hygiene" non-negotiable: dashboard URLs are operator-facing
pointer surfaces ONLY; credit balance values + token strings MUST NOT
enter the git-tracked tree.

### Step 2: Operator-direct decision (5 minutes; $0)

Operator chooses one of:

- **Option A** (HF Jobs recharge ~$2-5)
- **Option B** (Modal T4 ~$2.36, requires ~2-4h editor rewrite to swap `platform: hf_jobs` → `platform: modal` in recipe + adapt trainer to Modal Volume + `secrets={HF_TOKEN: ...}` + Catalog #166 sentinel files)
- **Option C** (Lightning A100 $0, requires ~2-4h editor rewrite)
- **Option D** (Vast.ai 4090 ~$1.00, requires ~2-4h editor rewrite)
- **DEFER** (no action; revisit per Q3 cost-band review)

### Step 3: If Option A — operator recharges HF Hub credit balance (10 minutes; ~$2-5)

Per https://huggingface.co/settings/billing operator-direct recharge
flow. Recommended floor: $5.00 to comfortably cover first dispatch
(~$1.60) + retry headroom + safety margin. Per CLAUDE.md "GPU budget
and compute resources" non-negotiable: budget caps are operator-direct;
this memo's $5 recommendation is advisory only.

### Step 4: Fire #523 BUILD_1 dispatch via canonical entry point (5 minutes; ~$1.60 dispatch)

Canonical operator-direct command per CLAUDE.md "Operator gates must be
wired and used" non-negotiable + Catalog #176 + #243 + #271 + #270
dispatch optimization protocol:

```bash
# Per Catalog #199 paired-env discipline + CLAUDE.md "Executing actions with care":
export OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1
export OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=5.00

.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch
```

The canonical entry point invokes (in order, per CLAUDE.md "Operator
gates must be wired and used"):

1. Catalog #152 required input file validation
2. Catalog #313 probe-outcomes ledger check (no blocking outcome for `hf_jobs_segnet_surrogate_distillation` per current state)
3. Catalog #243 local pre-deploy harness (8 checks including Catalog #270 dispatch optimization protocol)
4. Catalog #271 codex pre-dispatch review (canonical review per `tools/run_codex_review_for_dispatch.py`)
5. Catalog #325 per-substrate symposium anchor verification (PROCEED verdict from `council_t1_hf_jobs_segnet_surrogate_distillation_symposium_20260519` within 14-day window)
6. Catalog #339 sister fail-closed registration of dispatched hf_jobs_id

### Step 5: HARVEST OR LOSE post-dispatch (within 24h; $0)

Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" sister non-negotiable
for HF Jobs (the canonical sister 4-layer pattern at
`tac.deploy.hf_jobs.job_id_ledger` mirrors `tac.deploy.modal.call_id_ledger`
per Catalog #245):

```bash
# Post-dispatch (within 24h):
.venv/bin/python tools/harvest_hf_jobs_calls.py \
    --hf-jobs-id <hf_jobs_id> \
    --update-canonical-ledger
```

The harvester reads the canonical ledger row, fetches the HF Jobs job
output (model weights pushed to `adpena/segnet-image-level-surrogate-mobilenet-v3-small-200ep`),
verifies the `eval_accuracy` >= 60% per the symposium op-routable #4 +
appends the empirical anchor to the canonical equations registry IF
op-routable #5 lands (operator-routed downstream).

### Total operator-direct effort

- Step 1: 5 minutes (operator-only; billing dashboard inspection)
- Step 2: 5 minutes (operator-direct decision)
- Step 3 (if Option A): 10 minutes + ~$2-5 recharge
- Step 4: 5 minutes (canonical CLI invocation)
- Step 5: post-dispatch monitoring; harvester invoked once

Total wall-clock: ~25 minutes operator-active + 4-6h dispatch elapsed +
24h harvest window. Total operator-direct dollar cost (Option A path):
~$3.60-6.60 first dispatch; $1.60 subsequent dispatches.

## Section 6 — Sister-coordination contract with PACT-NERV-DistilledScorer + Codex LL

Per the sister landing memo `pact_nerv_distilled_scorer_x_codex_ll_integration_design_20260520.md`
(commit f39d6f6ce) Section 4:

| Surface | Role | Status |
|---|---|---|
| **#523 (this memo)** | IMAGE-LEVEL surrogate axis (2.5M-param mobilenetv3_small via HF Jobs T4) | BLOCKED on HF Jobs billing |
| **PACT-NERV-DistilledScorer** | INSIDE-DECODER surrogate axis (10k-param Conv2d via Modal) | L1 SCAFFOLD `lane_pact_nerv_distilled_scorer_l0_scaffold_20260520` |
| **Codex LL `scorer_response_dataset`** | OBSERVABILITY-ONLY paired-smoke advisory rows | 29-row dataset; canonical Provenance per Catalog #287/#323 |

The three surfaces are STRUCTURALLY ORTHOGONAL axes of the same canonical
Quantizr KL-T=2.0 0.33 [contest-CUDA] technique. THIS memo unblocks
Surface C (#523); the sister memo lands the sister-coordination contract
for all three.

The cascade-direction pivot from the parser-safe subset smoke landing
(`e3e198c9f`) explicitly anchors the cascade toward canonical IN-DOMAIN
substrates including the KL distillation lane. #523 BUILD_1 dispatch is
the empirical-anchor surface for the cascade.

## Section 7 — Catalog #313 probe-outcomes ledger check

Per Catalog #313 `check_dispatch_target_has_no_predecessor_adjudicated_outcome`
checked against substrate id `hf_jobs_segnet_surrogate_distillation`:

- Substrate is L1 per `lane_hf_jobs_segnet_surrogate_distillation_20260519`
- No predecessor adjudicated outcome in `.omx/state/probe_outcomes.jsonl` (canonical helper `tac.probe_outcomes_ledger`)
- Per Catalog #313 acceptance cascade (b) — no blocking outcome
- Status: clean per Catalog #313

## Section 8 — Catalog #322 composition-alpha with landed DP1+VQ-VAE substrates

Per Catalog #322 `check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha`
the composition-alpha consumer must consult VALIDATED_CONTEST_MEMBER
substrates only.

#523 surrogate output is `[predicted]` per Catalog #287/#323 canonical
Provenance with non-promotable markers — explicitly NOT a contest archive
member. Therefore Catalog #322 composition-alpha calculation against
DP1/VQ-VAE is N/A for #523 directly; composition-alpha applies to the
DOWNSTREAM substrate trainer that consumes #523's Hinton teacher signal
(deferred to Phase 2 sister lane per the T1 symposium op-routable #2).

Predicted composition-alpha for downstream consumers (HARD-EARNED-EMPIRICALLY-UNTESTED):

| Composition pair | Predicted alpha | Rationale |
|---|---|---|
| (Downstream substrate that consumes #523 teacher) × DP1 | 0.7-1.0 (additive-to-orthogonal) | DP1 is OUT-OF-DISTRIBUTION codebook from Comma2k19; #523 surrogate is contest-video-specific |
| (Downstream substrate that consumes #523 teacher) × VQ-VAE | 0.4-0.6 (sub-additive) | VQ-VAE also operates on contest-video signal; partial overlap |
| #523 image-level surrogate × PACT-NERV-DistilledScorer inside-decoder surrogate | 0.6-0.9 (additive-to-orthogonal) | Different parameter counts (250x) + different signal axes (image-level top-1 vs per-decoder-block channel-bias) |

## Section 9 — Cargo-cult audit per Catalog #303 (5 assumptions)

| Assumption | Classification | Unwind path |
|---|---|---|
| HF Jobs 402 Payment Required IS the actual blocker (not token scope / org billing / API quota) | HARD-EARNED | Verified via canonical ledger row 2026-05-19T17:36:36Z `failure_reason='prepaid credit balance insufficient'` |
| Operator-direct billing-decision is canonical (not autopilot) | HARD-EARNED | CLAUDE.md "Executing actions with care" + "GPU budget and compute resources" non-negotiables |
| HF Jobs t4-small is canonical platform per T1 symposium PROCEED 2026-05-19 | HARD-EARNED | Operator approved "all operator routable items" 2026-05-19T07:00Z verbatim |
| Sister-provider migration (Modal/Lightning/Vast.ai) is COMPOSITION-COMPATIBLE for #523 image-level surrogate | CARGO-CULTED-EMPIRICALLY-UNTESTED | $0-5 sister smoke on Modal CPU OR Vast.ai 4090 OR Lightning A100 with adapted recipe |
| HF Jobs recharge ~$2-5 is operator-attention-threshold-safe per CLAUDE.md "GPU budget" $24 Vast.ai cap | HARD-EARNED-AT-DESIGN-TIME | Within operator budget envelope; per CLAUDE.md "GPU budget" non-negotiable operator decides |

## Section 10 — 9-dimension success checklist evidence per Catalog #294

1. **UNIQUENESS**: NEW operator-direct billing-decision design memo for the canonical HF Jobs platform sub-surface; not present in sister lanes. Distinct from sister `hf_jobs_vs_modal_vs_vastai_cost_comparison_20260518` (which is platform-axis cost-comparison) and from `pact_nerv_distilled_scorer_x_codex_ll_integration_design_20260520` (which is cross-tool sister-coordination contract).
2. **BEAUTY+ELEGANCE**: design memo per CLAUDE.md "Beauty, simplicity, and developer experience"; 5-step path is operator-routable in <30 seconds reviewability per PR101-style canonical pattern.
3. **DISTINCTNESS**: distinct from in-flight 5-SUBSTRATE MATRIX SUPERSESSION (`a4b9500c`) + MAGIC CODEC PAIR #4 ORTHOGONALITY SMOKE (`aae30921`) per the operator-mandated scope-coherence rule.
4. **RIGOR**: Catalog #229 PV (read 7 sister memos + #523 trainer 359 LOC + recipe + dispatcher + ledger row 2026-05-19T17:36:36Z + canonical equations registry) + Catalog #292 per-deliberation assumption surfacing in council frontmatter.
5. **OPTIMIZATION-PER-TECHNIQUE**: per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD — three structurally-different surfaces (HF Jobs T4 mobilenetv3 + PACT-NERV-DistilledScorer Conv2d + Codex LL response-table) each engineered for their optimal score-lowering path per CLAUDE.md "Quantizr intelligence" canonical anchor.
6. **STACK-OF-STACKS-COMPOSABILITY**: see Section 8 (composition-alpha estimates with DP1/VQ-VAE/PACT-NERV-DistilledScorer).
7. **DETERMINISTIC-REPRODUCIBILITY**: canonical Provenance per Catalog #323 on every artifact; fcntl-locked HF Jobs ledger sister per Catalog #131 + #245; recipe declares `dataset_revision` for byte-stable Hub commit pinning.
8. **EXTREME-OPTIMIZATION-PERFORMANCE**: t4-small at $0.40/hr; 200ep training in ~4h = $1.60; HF Datasets-native (no Catalog #152 local file mount). Per CLAUDE.md "GPU budget and compute resources" cheapest CUDA option per GPU-hour for sub-100M-param distillation.
9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: predicted band [-0.003, +0.001] per HARD-EARNED-via-Quantizr-0.33-anchor for DOWNSTREAM Hinton-distillation-substituted substrate (per Catalog #324 `predicted_band_validation_status: pending_post_training`). #523 surrogate output ITSELF is `[predicted]` advisory-only.

## Section 11 — Observability surface per Catalog #305

1. **Inspectable per layer**: HF Jobs canonical ledger at `.omx/state/hf_jobs_call_id_ledger.jsonl` (fcntl-locked JSONL per Catalog #131); per-row schema includes `hf_jobs_id` / `lane_id` / `label` / `flavor` / `expected_cost_usd` / `failure_reason` / `evidence_grade`.
2. **Decomposable per signal**: per-row separation of `intent` / `failed` / `dispatched` / `harvested` event_types; per-platform separation (`platform: hf_jobs` vs `modal` vs `lightning` vs `vastai`).
3. **Diff-able across runs**: byte-stable JSONL schema_version `hf_jobs_ledger_v1_catalog342_20260519`; Hub commit sha + scorer sha stamped per row.
4. **Queryable post-hoc**: 4 query helpers exist: `query_by_hf_jobs_id`, `query_by_lane`, `latest_status_by_hf_jobs_id`, `load_hf_jobs_strict` per `src/tac/deploy/hf_jobs/job_id_ledger.py`.
5. **Cite-able**: every artifact carries `provenance` sub-object per Catalog #323 canonical Provenance umbrella; lane_id + git_HEAD sha + canonical_helper_invocation stamped per row.
6. **Counterfactual-able**: dispatcher `--dry-run` mode emits canonical JSON plan per `tools/dispatch_hf_jobs_vision_training.py` for re-runability without paid dispatch.

## Section 12 — Sister-collision verdict + sister-DISJOINT scope

Verified sister-DISJOINT from:

- **In-flight 5-SUBSTRATE MATRIX SUPERSESSION** (commit `a4b9500c`): different scope (substrate matrix supersession across 5 substrates) vs THIS lane (HF Jobs billing unblock for ONE specific lane #523).
- **In-flight MAGIC CODEC PAIR #4 ORTHOGONALITY SMOKE** (commit `aae30921`): different scope (magic codec pair-4 orthogonality smoke) vs THIS lane (HF Jobs billing unblock).
- **PACT-NERV-DistilledScorer × Codex LL INTEGRATION DESIGN** (commit `f39d6f6ce`): SISTER (not collision); THIS memo cross-references its Surface C finding without mutating the sister memo per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE.

STEP 0 sister-collision check via `tools/check_sister_files_recently_landed.py`:
**PROCEED** (no sister commits touched target file within 12-hour lookback).

Per Catalog #340 sister-checkpoint guard: sister subagents in flight do
not overlap with the file I am about to commit
(`.omx/research/hf_jobs_billing_unblock_523_hinton_surrogate_20260520.md`
+ this memo's sister landing memo).

## Section 13 — 6-hook wire-in declaration per Catalog #125

- **Hook #1 sensitivity-map contribution** = N/A (defensive design memo; no signal contribution at this surface)
- **Hook #2 Pareto constraint** = N/A (no Pareto-relevant signal at this surface)
- **Hook #3 bit-allocator hook** = N/A (no bit-allocator signal at this surface)
- **Hook #4 cathedral autopilot dispatch hook** = **ACTIVE** (post-unblock #523 BUILD_1 dispatch produces empirical anchor consumable by autopilot ranker per `distilled_scorer_surrogate_canonical_equation_consumer` sister-lane BUILD per the sister landing memo's op-routable #4)
- **Hook #5 continual-learning posterior update** = **ACTIVE** (post-unblock empirical anchor appends to canonical equations registry via `tac.canonical_equations.update_equation_with_empirical_anchor` per Catalog #344 if op-routable #5 from sister memo lands)
- **Hook #6 probe-disambiguator** = **ACTIVE** (the operator-direct billing-decision IS the disambiguator between (a) HF Jobs recharge / (b) sister-provider migration / (c) DEFER)

## Cross-references

- Sister design memo: `.omx/research/pact_nerv_distilled_scorer_x_codex_ll_integration_design_20260520.md` (commit `f39d6f6ce` Surface C finding)
- Substrate lane: `lane_hf_jobs_segnet_surrogate_distillation_20260519` L1
- Trainer: `experiments/hf_jobs_segnet_surrogate_distillation.py` (359 LOC; impl_complete)
- Recipe: `.omx/operator_authorize_recipes/substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch.yaml` (dispatch_enabled: true; research_only: false)
- T1 symposium: `.omx/research/council_t1_hf_jobs_segnet_surrogate_distillation_symposium_20260519.md` (PROCEED with 5 op-routables)
- Operator approval: `.omx/research/operator_authorizations/hf_jobs_segnet_surrogate_distillation_operator_approval_20260519T070000Z.md` ("all operator routable items approved" 2026-05-19T07:00Z verbatim)
- Cost-comparison memo: `.omx/research/hf_jobs_vs_modal_vs_vastai_cost_comparison_20260518.md`
- Migration audit: `.omx/research/hf_jobs_substrate_migration_audit_20260518.md`
- HF Jobs canonical ledger module: `src/tac/deploy/hf_jobs/job_id_ledger.py` (31.5 KB)
- HF Jobs canonical dispatcher: `tools/dispatch_hf_jobs_vision_training.py` (17.8 KB)
- 402 event empirical anchor: `.omx/state/hf_jobs_call_id_ledger.jsonl` row 2026-05-19T17:36:36Z
- Catalog gates: #110 + #113 + #125 + #131 + #138 + #143 + #152 + #176 + #199 + #229 + #243 + #245 + #265 + #270 + #271 + #287 + #290 + #292 + #294 + #303 + #305 + #309 + #313 + #322 + #323 + #325 + #335 + #339 + #340 + #341 + #344 + #346
