# PACT-NERV-DistilledScorer × Codex LL Planner × #523 Cross-Cascade Integration Design

<!--
Catalog #344 canonical equation registry cross-reference:
This memo introduces a NEW canonical equation design spec:
`kl_distillation_scorer_surrogate_compression_savings_v1`. The registration
itself is OPERATOR-ROUTED downstream per "evolution discipline" — this memo
defines the formula, IN-DOMAIN contexts, predicted band, and producer/consumer
contract. Sister equations consulted during design: `categorical_posterior_capacity_vs_continuous_gaussian_v1`,
`procedural_codebook_from_seed_compression_savings_v1`, `per_byte_leverage_uniformly_distributed_v1`,
`per_pair_master_gradient_score_impact_taylor_v1`. Producer/consumer audit
per `CanonicalEquation.__post_init__` invariant: PACT-NERV-DistilledScorer
trainer (producer when Stage 1 dispatch lands empirical anchor) + LL
`scorer_response_dataset` (producer when distilled-vs-direct paired smoke
appends rows) + canonical equation lookup consumer + autopilot ranker
adjustment branch + Catalog #322 composition-alpha consumer (consumers).
-->

---

```yaml
council_tier: T1
council_attendees:
  - Shannon  # information-theory grounding for KL distillation
  - Dykstra  # alternating-projections feasibility for archive+rate+seg+pose polytope
  - Rudin    # interpretable-ML lens on distilled surrogate
  - Daubechies  # multi-scale wavelet partition prior
  - Hinton   # CANONICAL inner-council seat: KL-T=2.0 distillation (1503.02531)
  - Yousfi   # scorer-design lens (Fridrich-pupil)
  - Quantizr # PR56-pattern empirical-anchor lens (0.33 [contest-CUDA])
  - Contrarian
  - Assumption-Adversary
  - PR95Author  # canonical HNeRV substrate + leaderboard-implementation parity
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "the predicted band [-0.003, +0.001] is HARD-EARNED via Quantizr 0.33 [contest-CUDA] anchor BUT the distillation-surrogate-as-conditioner CARGO-CULTED-MAY-BE-PROMISING assumption inherited from L0 SCAFFOLD is unresolved; the integration with codex LL response dataset MUST land BOTH paired distill-vs-direct smoke rows in the same operator-routed Stage 1 dispatch before any composition-alpha claim against DP1+VQ-VAE substrates is consumable"
  - member: Assumption-Adversary
    verbatim: "the assumption that 'codex LL scorer_response_dataset is observability-only AND PACT-NERV-DistilledScorer trains a Hinton-distilled internal surrogate AND THEREFORE the two surfaces can compose via a shared canonical helper' is HARD-EARNED-AT-DESIGN-TIME (sister-coordination contract is structurally sound per Catalog #335 canonical Protocol) BUT the actual KL target wiring (whether the LL dataset's score-vs-perturbation curve is a useful Hinton TEACHER signal for the distilled scorer surrogate, or merely correlated noise) is CARGO-CULTED-EMPIRICALLY-UNTESTED; Stage 1 paired smoke unwinds the cargo-cult"
council_assumption_adversary_verdict:
  - assumption: "KL-T=2.0 distillation IS the canonical Quantizr 0.33 [contest-CUDA] technique"
    classification: HARD-EARNED
    rationale: "Hinton 1503.02531 §3 + Quantizr 0.33 [contest-CUDA] empirical anchor + Hinton inner council seat per CLAUDE.md 'Council conduct'"
  - assumption: "Codex LL scorer_response_dataset is composition-compatible with PACT-NERV-DistilledScorer surrogate training as Hinton TEACHER signal"
    classification: CARGO-CULTED-EMPIRICALLY-UNTESTED
    rationale: "LL dataset (29 rows post-pair-4) is canonical observability surface for compress-time MASKED-KNOB and DECODER-Q candidate scoring; the dataset's score-vs-perturbation curve has NEVER been empirically tested as a Hinton teacher signal for a distilled scorer surrogate; the structural composability is sound (Catalog #335 contract) but the empirical signal-vs-noise ratio is unknown"
  - assumption: "The 3-surface convergence pattern (substrate + LL planner + #523 long-pending) is canonical apparatus design discipline, not coincidence"
    classification: HARD-EARNED
    rationale: "operator NON-NEGOTIABLE standing directive ('UNIQUE-AND-COMPLETE-PER-METHOD' + 'Subagent coherence-by-default' + 'consolidate everything into META layer or canonical helpers') REQUIRES convergent surfaces bind via canonical helpers per Catalog #335 + sister Catalog #344 canonical equations + #245 4-layer ledger exemplar"
council_decisions_recorded:
  - "op-routable #1: review NEW canonical equation design spec `kl_distillation_scorer_surrogate_compression_savings_v1`; register via tac.canonical_equations.register_canonical_equation iff predicted band + producer/consumer contract pass council review (operator decides; THIS lane MUST NOT register)"
  - "op-routable #2: PACT-NERV-DistilledScorer per-substrate symposium per Catalog #325 BEFORE Stage 1 dispatch unblock (the 14-day window requires the symposium to dated +/- 14 days of dispatch)"
  - "op-routable #3: codex LL scorer_response_dataset spec extension to ALSO emit a `distilled_vs_direct_scorer_paired_smoke` row family (CARGO-CULTED assumption unwind; cheap $0-5 smoke pair)"
  - "op-routable #4: NEW cathedral consumer `distilled_scorer_surrogate_canonical_equation_consumer` AT package surface src/tac/cathedral_consumers/distilled_scorer_surrogate_canonical_equation_consumer/__init__.py declaring CONSUMES_SCORER_RESPONSE_DATASET=True marker + canonical Catalog #335 Protocol contract + Catalog #341 Tier-A observability-only markers (defers to a sister lane's BUILD)"
  - "op-routable #5: bind #523 (Catalog #523 L2 Hinton-distilled SegNet surrogate Phase 1 BUILD) as the canonical SISTER lane to PACT-NERV-DistilledScorer; #523 trains the IMAGE-LEVEL scorer surrogate (mobilenetv3_small 2.5M params via HF Jobs); PACT-NERV-DistilledScorer trains the INSIDE-DECODER scorer surrogate (Conv2d 10k params via Modal). The two are ORTHOGONAL axes of the same canonical Quantizr KL-T=2.0 technique"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: ""
related_deliberation_ids:
  - pact_nerv_ultimate_research_and_design_20260520
  - pact_nerv_distilled_scorer_l0_scaffold_design_20260520
  - council_t1_hf_jobs_segnet_surrogate_distillation_symposium_20260519
  - codex_findings_ll_frame_pair_curriculum_masked_knobs_20260520
  - ll_scorer_response_pair4_guarded_plan_landed_20260521
horizon_class: frontier_pursuit
predicted_band: "[-0.003, +0.001]"
predicted_band_provenance: HARD-EARNED-via-Quantizr-0.33-anchor
predicted_band_validation_status: pending_post_training
deferred_substrate_id: null
substrate_alias: null
---
```

## Section 1 — Convergence narrative (executive summary)

Three independently-landed surfaces converge on the canonical Quantizr
KL-T=2.0 0.33 [contest-CUDA] distillation technique per CLAUDE.md "Quantizr
intelligence" + Hinton inner-council seat:

1. **PACT-NERV-DistilledScorer** (substrate L1 SCAFFOLD at commit `b56f24bc1`)
   trains a 10k-param Conv2d `DistilledScorerSurrogate` INSIDE the HNeRV
   decoder via KL-T=2.0 from frozen upstream SegNet+PoseNet logits. Stage 1
   dispatch operator-gated per Catalog #325.
2. **Codex LL planner + scorer_response_dataset** (`tools/plan_ll_scorer_response_next.py`
   + `tools/build_scorer_response_dataset.py` + `src/tac/optimization/scorer_response_dataset.py`)
   maintains a canonical observability-only dataset of paired
   compress-time perturbation → empirical score deltas. Per the latest
   codex landing (`codex_findings_ll_frame_pair_curriculum_masked_knobs_20260520T214749Z_codex.md`)
   the dataset is 29 rows: decoder_q (21) + inflate_postprocess (6) +
   scorer_gradient_sparse_residual (1) + sparse_residual_oracle (1).
3. **#523 long-pending canonical lane** (Catalog #523 L2 Hinton-distilled
   SegNet surrogate Phase 1 BUILD, TaskCreate #875) trains a 2.5M-param
   `mobilenetv3_small` IMAGE-LEVEL scorer surrogate via HF Jobs T4. Status:
   BLOCKED 2026-05-19T17:37Z on HF Jobs 402 Payment Required (lane
   `lane_hf_jobs_segnet_surrogate_distillation_20260519`).

The three surfaces are STRUCTURALLY ORTHOGONAL axes of the same canonical
Quantizr technique. This memo lands the canonical sister-coordination
contract: a NEW `CONSUMES_SCORER_RESPONSE_DATASET` cathedral consumer
marker (sister of the proven `CONSUMES_MASTER_GRADIENT_ANCHORS` marker
per Catalog #335 auto-discovery + the `tac.master_gradient.append_anchor_locked`
fan-out runtime hook precedent at commits 7b9d5e280 + later wire-in), a
NEW canonical equation design spec
`kl_distillation_scorer_surrogate_compression_savings_v1` per Catalog
#344 EVOLUTION DISCIPLINE (NOT REGISTERED — operator-routed downstream),
and a 5-step operator-routable path.

Sister-DISJOINT scope from in-flight PR101/PR106 PROCEDURAL VARIANT BUILD
DESIGN (`a97084e7`) + PR101 GOLD NULL-BYTE REMOVAL SMOKE (`a3dfc84c`) per
the operator-mandated scope-coherence rule.

## Section 2 — Per-surface canonical state audit

### Surface A: PACT-NERV-DistilledScorer substrate (commit `b56f24bc1`)

| Artifact | Status | LOC |
|---|---|---|
| `src/tac/substrates/pact_nerv_distilled_scorer/__init__.py` | L1 SCAFFOLD with Hinton anchor verbatim | 105 |
| `src/tac/substrates/pact_nerv_distilled_scorer/architecture.py` | L0+: `DistilledScorerSurrogate` + `PactNervDistilledScorerSubstrate` | 249 |
| `src/tac/substrates/pact_nerv_distilled_scorer/archive.py` | L0+: PDS magic + brotli FP16 state-dict + int16 latents | 198 |
| `src/tac/substrates/pact_nerv_distilled_scorer/score_aware_loss.py` | L0+: 4-term Lagrangian with `delta_distill * KL_T2` placeholder | 117 |
| `src/tac/substrates/pact_nerv_distilled_scorer/inflate.py` | L0+: ≤150 LOC PDS consumer | 82 |
| `experiments/train_substrate_pact_nerv_distilled_scorer.py` | L0+ trainer wrapper | ~250 (sister) |
| `.omx/operator_authorize_recipes/substrate_pact_nerv_distilled_scorer_modal_t4_dispatch.yaml` | research_only=true; dispatch_enabled=false | (recipe) |

Status: per the L0 design memo at
`.omx/research/pact_nerv_distilled_scorer_l0_scaffold_design_20260520T211500Z.md`
the lane is `lane_pact_nerv_distilled_scorer_l0_scaffold_20260520`. The
Hinton-distillation KL-T=2.0 wire-in is PLACEHOLDER at L0; Stage 1 dispatch
lands the actual KL-T=2.0 forward + backward.

Per Catalog #240 recipe-vs-trainer-state consistency: recipe declares
`research_only=true` AND trainer's `_full_main` raises `NotImplementedError`
at L0 — clean structural state per CLAUDE.md "Substrate scaffolds MUST be
COMPLETE or RESEARCH-ONLY".

### Surface B: Codex LL planner + scorer_response_dataset

| Artifact | Status | Role |
|---|---|---|
| `src/tac/optimization/scorer_response_dataset.py` | 37 KB canonical | Module-level `build_response_dataset` + `build_next_probe_plan` + `render_markdown` |
| `tools/build_scorer_response_dataset.py` | 3.1 KB CLI | Aggregates advisory JSONs into canonical dataset rows |
| `tools/plan_ll_scorer_response_next.py` | 3.9 KB CLI | Reads dataset + emits next-probe plan with prohibitions |
| `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/*` | active artifact dir | 29-row dataset; 21 decoder_q + 6 inflate_postprocess + 1 scorer_gradient_sparse_residual + 1 sparse_residual_oracle |

Status: per the codex findings memo
(`codex_findings_ll_frame_pair_curriculum_masked_knobs_20260520T214749Z_codex.md`)
the dataset row family taxonomy is observability-only per Catalog #287/#323
(canonical Provenance + `score_claim=False` + `promotion_eligible=False` +
`ready_for_exact_eval_dispatch=False` enforced at CLI surface). Best row
post-pair-4 is `pair_0007_seg_boundary_last_frame_rgb_bias_p1` at score
0.19206733847162177 (delta +5.91e-06 regression vs baseline 0.19206142414659494
per the canonical frontier pointer per Catalog #343).

### Surface C: #523 long-pending canonical lane

| Artifact | Status | Verdict |
|---|---|---|
| `experiments/hf_jobs_segnet_surrogate_distillation.py` | impl_complete | trainer is COMPLETE per #523 BUILD_1 |
| `.omx/operator_authorize_recipes/substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch.yaml` | research_only=true; dispatch_enabled=false | clean per Catalog #240 |
| Council T1 working group memo at `.omx/research/council_t1_hf_jobs_segnet_surrogate_distillation_symposium_20260519.md` | LANDED | symposium verdict PROCEED |
| Task `codex_routing_directive_comprehensive_wire_in_and_integration_pass_20260519T072000Z::BUILD_1` | status=blocked | blocker `hf_jobs_prepaid_credit_balance_insufficient_402_before_job_id` |
| Lane `lane_hf_jobs_segnet_surrogate_distillation_20260519` | L1 (impl_complete + memory_entry) | 2 gates green |
| Catalog #342 (sister) | claimed via canonical serializer 2026-05-19T12:21Z | reason: "HF Jobs SegNet/PoseNet surrogate dataset + Jobs implementation surface" |

Status: #523 is the IMAGE-LEVEL surrogate axis (2.5M-param mobilenetv3_small
trained on the full 600-pair contest video via HF Jobs T4); PACT-NERV-DistilledScorer
is the INSIDE-DECODER surrogate axis (10k-param Conv2d trained simultaneously
with the HNeRV decoder via Modal). The two are STRUCTURALLY DIFFERENT
implementations of the same canonical Quantizr KL-T=2.0 technique. Per
CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" non-negotiable each is
optimal-engineering-per-method.

## Section 3 — Cross-cascade convergence observation

The convergence is NOT coincidence; it is canonical apparatus discipline
producing convergent designs from independent surfaces. Per the operator
standing directive *"What we need to do is iterate, integrate, dispatch,
ensure no orphan signals or rotting fully and completely"*:

- PACT-NERV-DistilledScorer landed as the substrate-engineering surface
  for KL-T=2.0 distillation (Hinton inner-council seat).
- Codex LL response dataset landed as the **observability-only** surface
  for distill-vs-direct paired-smoke evidence (Catalog #287/#323
  canonical Provenance enforcement at the CLI surface).
- #523 long-pending canonical lane landed as the **HF-Jobs-platform**
  surface for the image-level surrogate distillation (Catalog #342
  sister + council T1 working group symposium verdict PROCEED).

These three independently-landed surfaces structurally bind via the
canonical Quantizr KL-T=2.0 technique. The integration design IS the
canonical sister-coordination contract that makes the binding STRUCTURAL
(per CLAUDE.md "Subagent coherence-by-default") rather than tribal.

## Section 4 — Distinguishing primitive contract per surface

### Surface A primitive: INSIDE-DECODER surrogate (10k params)

`DistilledScorerSurrogate(hidden=32, feature_dim=16)` per
`src/tac/substrates/pact_nerv_distilled_scorer/architecture.py:88-129`.
Forward `(B, 3, H_smoke, W_smoke) -> (B, 16)` via 3-layer Conv2d stride-2
encoder + global avg pool. Per-block channel-bias projection from the
16-dim feature vector to each upsample block's channel width. The
surrogate features condition the HNeRV decoder; the surrogate weights
LOGICAL-GROUP inside `DECODER_BLOB` per archive grammar (key prefix
`surrogate.`).

### Surface B primitive: paired-smoke advisory rows

`build_response_dataset(input_paths, baseline)` per
`tac.optimization.scorer_response_dataset.py`. Per-row schema includes
family (`decoder_q` / `inflate_postprocess` / `scorer_gradient_sparse_residual`
/ `sparse_residual_oracle`), measured score, archive bytes, perturbation
spec. Per CLAUDE.md "Apples-to-apples evidence discipline" the dataset
is canonical observability-only; scores are tagged per Catalog #323
canonical Provenance with non-promotable markers.

### Surface C primitive: IMAGE-LEVEL surrogate (2.5M params)

`mobilenetv3_small` from `timm` per
`experiments/hf_jobs_segnet_surrogate_distillation.py`. Image-level
distillation (NOT per-pixel; per-pixel mIoU distillation is sister Phase
2 per the T1 working group symposium). Forward `(B, 3, 384, 512) -> (B, 5)`
class logits matching upstream SegNet's argmax 5-class output. Trained
200 epochs via HF Jobs T4 with `KL(student || softmax(teacher / T=2.0))`.

The three primitives are STRUCTURALLY DIFFERENT (decoder-internal-Conv2d
vs response-table vs HF-Jobs-mobilenetv3_small) but produce the same
canonical KL-T=2.0 signal axis.

## Section 5 — Sister-coordination contract design (NEW marker)

### NEW marker: `CONSUMES_SCORER_RESPONSE_DATASET = True`

Sister of `CONSUMES_MASTER_GRADIENT_ANCHORS` per `src/tac/master_gradient.py:533`
+ `_fire_post_anchor_consumer_hooks` runtime fan-out pattern at
`src/tac/master_gradient.py:575-641`.

Canonical contract (placed at module level of every opt-in cathedral
consumer):

```python
# Per CLAUDE.md "Subagent coherence-by-default" + Catalog #335 canonical
# Protocol contract: consumers that opt-in to receive scorer_response_dataset
# refresh events declare CONSUMES_SCORER_RESPONSE_DATASET = True at module
# level. The runtime hook lands in tac.optimization.scorer_response_dataset
# at the canonical append surface; opt-in consumers receive each new
# row's parsed dict via update_from_anchor(row).
CONSUMES_SCORER_RESPONSE_DATASET = True
```

Sister fan-out runtime hook (deferred to a sister lane's BUILD per
Catalog #110/#113 APPEND-ONLY discipline; THIS lane is design-only):

```python
# In tac/optimization/scorer_response_dataset.py, AFTER the canonical
# atomic dataset write:
def _fire_post_dataset_row_consumer_hooks(row: Mapping[str, Any]) -> None:
    """Sister of master_gradient._fire_post_anchor_consumer_hooks.

    Per CLAUDE.md "Subagent coherence-by-default" maximum-signal-preservation:
    per-consumer exceptions are caught + warning-logged so a single buggy
    consumer cannot block sister consumers from receiving the row.
    Per Catalog #341 Tier-A canonical-routing-markers: consumers contribute
    predicted_delta_adjustment=0.0 + promotable=False + axis_tag="[predicted]"
    always; the auto-trigger fan-out does NOT mutate any score signal.
    """
    # ... discover_compliant_consumer_modules() per Catalog #335
    # ... filter on getattr(mod, "CONSUMES_SCORER_RESPONSE_DATASET", False)
    # ... call mod.update_from_anchor(row) under try/except per consumer
```

### Sister-coordination contract: PACT-NERV-DistilledScorer ↔ LL dataset

The PACT-NERV-DistilledScorer trainer's Stage 1 dispatch (when operator-gated
per Catalog #325 unblocks it) will produce paired distill-vs-direct smoke
artifacts. These artifacts append to the canonical LL scorer_response_dataset
via the existing `tools/build_scorer_response_dataset.py` CLI per the row
family extension proposed in op-routable #3 (NEW row family
`distilled_vs_direct_scorer_paired_smoke`).

The NEW cathedral consumer `distilled_scorer_surrogate_canonical_equation_consumer`
(op-routable #4; sister-lane BUILD) declares BOTH `CONSUMES_SCORER_RESPONSE_DATASET=True`
AND `CONSUMES_MASTER_GRADIENT_ANCHORS=True` markers per the Catalog #335
multi-marker pattern. Its `update_from_anchor(row)` reads the new
distilled-vs-direct rows + the master-gradient anchors AND populates the
NEW canonical equation `kl_distillation_scorer_surrogate_compression_savings_v1`
empirical anchor list per Catalog #344 EVOLUTION DISCIPLINE.

## Section 6 — KL target wiring spec (canonical helper recommendation)

The KL target wiring lives in PACT-NERV-DistilledScorer's
`score_aware_loss.py` Lagrangian, NOT in the LL dataset (the LL dataset
is observability-only per Catalog #287/#323).

Recommended canonical helper signature (sister-lane BUILD; defers to
PACT-NERV-DistilledScorer Stage 1 trainer):

```python
def compute_kl_distillation_loss_t2(
    student_logits: torch.Tensor,    # (B, C) — distilled surrogate output
    teacher_logits: torch.Tensor,    # (B, C) — frozen upstream SegNet OR PoseNet
    *,
    temperature: float = 2.0,        # Hinton 1503.02531 §3 canonical T=2.0
    reduction: str = "batchmean",    # Hinton canonical; NOT "mean" per common bug
) -> torch.Tensor:
    """Hinton-Vinyals-Dean 2015 (arXiv:1503.02531) KL-T=2.0 distillation.

    The TEMPERATURE softens both teacher and student softmax outputs;
    the T**2 factor on the loss restores the gradient magnitude scale
    so the distillation loss is comparable to a direct supervised loss
    at T=1.0.

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag":
    callers MUST tag the resulting loss term with `[predicted]` axis_tag
    until paired Stage 1 dispatch lands an empirical anchor against the
    direct-scorer-routing baseline.
    """
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(soft_student, soft_teacher, reduction=reduction)
    return kl * (temperature ** 2)
```

Canonical location (deferred to PACT-NERV-DistilledScorer Stage 1 BUILD):
`src/tac/substrates/pact_nerv_distilled_scorer/distillation.py` (NEW
module; ~30 LOC). The HF Jobs sister at #523 already implements this
identical KL formula per the canonical Quantizr technique; the canonical
helper would lift the implementation to a shared module that BOTH
substrates consume per CLAUDE.md "consolidate everything into META layer
or canonical helpers" standing directive.

## Section 7 — NEW canonical equation design spec (NOT REGISTERED)

Per Catalog #344 EVOLUTION DISCIPLINE the equation is defined here +
operator-routed to registration via op-routable #1.

```yaml
equation_id: kl_distillation_scorer_surrogate_compression_savings_v1
schema_version: canonical_equation_v1_20260519
name: "KL-T=2.0 distilled scorer surrogate compression savings"
one_line_summary: |
  Compression savings (bytes) when a substrate replaces direct scorer routing
  with a Hinton-distilled internal surrogate at training time, holding
  archive grammar + decoder backbone constant.

formula_text: |
  delta_archive_bytes(surrogate) = brotli_compress_bytes(surrogate_state_dict)
                                   - 0   # direct scorer routing adds zero archive bytes

  delta_predicted_score(surrogate) = 25 * delta_archive_bytes / 37545489
                                     + 100 * delta_d_seg(surrogate_signal_vs_direct)
                                     + sqrt(10) * delta_sqrt_d_pose(surrogate_signal_vs_direct)

  CARGO-CULTED-MAY-BE-PROMISING at L0: the surrogate-signal-vs-direct
  delta_d_seg + delta_d_pose terms are UNTESTED at first registration;
  the predicted band [-0.003, +0.001] is HARD-EARNED-via-Quantizr-0.33-anchor
  ONLY for the rate-cost side (~5-8 KB brotli'd surrogate weights).

latex_form: |
  \Delta S(\text{surrogate}) =
    \underbrace{\frac{25 \cdot B_{\text{surrogate}}}{37{,}545{,}489}}_{\text{rate cost}}
    + \underbrace{100 \cdot \Delta d_{\text{seg}}}_{\text{seg signal delta}}
    + \underbrace{\sqrt{10} \cdot \Delta \sqrt{d_{\text{pose}}}}_{\text{pose signal delta}}

units_in:
  - "surrogate_param_count: int (typical 10000-2500000)"
  - "brotli_quality: int (typical 9-11)"
  - "delta_d_seg: float (typical [-0.001, +0.005])"
  - "delta_d_pose: float (typical [-0.0001, +0.001])"
units_out:
  - "delta_predicted_score: float (typical [-0.003, +0.001])"

domain_of_validity:
  - "HNeRV-class renderer with brotli-FP16 weight serialization"
  - "distilled surrogate ships INSIDE decoder_blob (NOT as separate archive section)"
  - "frozen-teacher KL-T=2.0 per Hinton 1503.02531 §3 canonical"
  - "PR101-class operating point ([contest-CPU] ≈ 0.19205 per canonical frontier pointer)"

predicted_band: "[-0.003, +0.001]"
predicted_band_provenance: HARD-EARNED-via-Quantizr-0.33-anchor
predicted_band_classification: HARD-EARNED-EMPIRICALLY-PARTIAL
  # rate cost side: HARD-EARNED (5-8 KB brotli'd weights canonical)
  # signal delta side: HARD-EARNED-EMPIRICALLY-UNTESTED (Quantizr empirical
  # anchor at 0.33 [contest-CUDA] proves the technique works at SOME
  # operating point; our PR101-class [contest-CPU] 0.19205 operating point
  # is NOT the Quantizr operating point so the signal delta is unknown)

hard_earned_vs_cargo_culted: HARD-EARNED-EMPIRICALLY-PARTIAL

empirical_anchors:
  - anchor_id: quantizr_canonical_0_33_contest_cuda
    archive_sha256: "(unknown; Quantizr private archive)"
    measurement_axis: "[contest-CUDA]"
    score: 0.33
    operating_point: "Quantizr canonical (per CLAUDE.md 'Quantizr intelligence')"
    notes: "HISTORICAL_SCORE_LITERAL_OK:quantizr_0_33_canonical_anchor_2026-04-21_landed"

canonical_producers:
  - "src/tac/substrates/pact_nerv_distilled_scorer/score_aware_loss.py (Stage 1 dispatch)"
  - "src/tac/substrates/pact_nerv_distilled_scorer/distillation.py (NEW; sister-lane)"
  - "experiments/hf_jobs_segnet_surrogate_distillation.py (#523 #342 sister)"
  - "tools/build_scorer_response_dataset.py (extension; op-routable #3)"

canonical_consumers:
  - "src/tac/cathedral_consumers/distilled_scorer_surrogate_canonical_equation_consumer/__init__.py (NEW; sister-lane BUILD per op-routable #4)"
  - "src/tac/cathedral_consumers/canonical_equation_lookup_consumer/__init__.py (Catalog #344 sister; existing)"
  - "tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_composition_alpha_v2 (Catalog #322 sister)"

python_callable_module_path: "tac.substrates.pact_nerv_distilled_scorer.distillation:compute_kl_distillation_loss_t2 (PROPOSED)"

provenance:
  axis_tag: "[predicted]"
  evidence_grade: predicted
  custody_status: "design-time; NOT registered until operator-routable #1 verdict"

last_calibration_utc: null
next_recalibration_trigger: |
  Recalibrate when Stage 1 PACT-NERV-DistilledScorer dispatch lands
  paired distill-vs-direct empirical anchor (cheap $0-5 smoke per
  op-routable #3 LL dataset row family extension).

predicted_vs_empirical_residual: null  # no empirical anchor yet
```

## Section 8 — Composition-alpha estimate per Catalog #322 with landed DP1+VQ-VAE substrates

Per Catalog #322 `check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha`
the composition-alpha consumer must consult VALIDATED_CONTEST_MEMBER
substrates only. DP1 (pretrained_driving_prior) + VQ-VAE substrates landed
recently per WAVE-3-PROCEDURAL VARIANT BUILD landings.

Predicted composition-alpha with each landed substrate (HARD-EARNED-EMPIRICALLY-UNTESTED;
predicted-band per canonical equations + sister memo analysis):

| Composition pair | Predicted alpha | Rationale |
|---|---|---|
| PACT-NERV-DistilledScorer × DP1 | 0.7-1.0 (additive-to-orthogonal) | DP1 is OUT-OF-DISTRIBUTION codebook from Comma2k19; distilled surrogate is contest-video-specific; the two operate on orthogonal signal axes |
| PACT-NERV-DistilledScorer × VQ-VAE | 0.4-0.6 (sub-additive) | VQ-VAE also operates on contest-video signal; partial overlap with distilled surrogate's signal axis |
| PACT-NERV-DistilledScorer × PR101 GOLD | 0.5-0.8 (sub-additive) | PR101 GOLD already uses HNeRV-class decoder; distilled surrogate provides additional conditioning surface but same backbone |

Per the canonical equation `hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1`:
within-HNeRV-cluster pairs are pre-classified SUB_ADDITIVE per the
auto_trigger_similarity_after_master_gradient_anchor_consumer; the
PACT-NERV-DistilledScorer × PR101 pair inherits this classification.

## Section 9 — Catalog #325 per-substrate symposium gating verdict

Per Catalog #325 `check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor`
every ASYMPTOTIC pursuit candidate REQUIRES individual adversarial grand
council symposium within 14-day window BEFORE paid dispatch is admissible.

Status for PACT-NERV-DistilledScorer Stage 1 dispatch:
- L0 SCAFFOLD design memo exists at
  `.omx/research/pact_nerv_distilled_scorer_l0_scaffold_design_20260520T211500Z.md`
- THIS integration design memo (`.omx/research/pact_nerv_distilled_scorer_x_codex_ll_integration_design_20260520.md`)
  carries T1 working-group council deliberation
- A T2+ sextet pact symposium per Catalog #325 6-step contract is REQUIRED
  before Stage 1 dispatch unblock
- Verdict for THIS lane: PROCEED_WITH_REVISIONS (op-routables #1-#5
  recorded); per-substrate symposium per Catalog #325 op-routable #2

## Section 10 — Catalog #313 probe-outcomes ledger check

Per Catalog #313 `check_dispatch_target_has_no_predecessor_adjudicated_outcome`
checked against substrate id `pact_nerv_distilled_scorer`:

- Substrate is L1 SCAFFOLD per `lane_pact_nerv_distilled_scorer_l0_scaffold_20260520`
- No predecessor adjudicated outcome in `.omx/state/probe_outcomes.jsonl` (canonical helper `tac.probe_outcomes_ledger`)
- Per Catalog #313 acceptance cascade (b) — no blocking outcome
- Status: clean per Catalog #313

## Section 11 — Cargo-cult audit per Catalog #303 (5 assumptions)

| Assumption | Classification | Unwind path |
|---|---|---|
| KL-T=2.0 distill IS the canonical Quantizr technique | HARD-EARNED | Hinton 1503.02531 §3 + Quantizr 0.33 [contest-CUDA] + Hinton inner council seat |
| LL response dataset row family extension to `distilled_vs_direct_scorer_paired_smoke` is structurally additive (per Catalog #335) | HARD-EARNED-AT-DESIGN-TIME | Sister marker pattern + Catalog #335 canonical Protocol contract precedent |
| LL dataset's score-vs-perturbation curve is a USEFUL Hinton teacher signal for the distilled scorer surrogate | CARGO-CULTED-EMPIRICALLY-UNTESTED | Stage 1 paired smoke (op-routable #3) |
| Composition with PR101/DP1/VQ-VAE substrates is ADDITIVE on the score axis | CARGO-CULTED-EMPIRICALLY-UNTESTED | Catalog #322 composition-alpha empirical anchor after Stage 1 dispatch |
| Image-level surrogate (#523 mobilenetv3_small 2.5M params) AND inside-decoder surrogate (PACT-NERV-DistilledScorer Conv2d 10k params) are COMPOSABLE (both contribute orthogonal signal) | CARGO-CULTED-EMPIRICALLY-UNTESTED | After both Stage 1 dispatches land empirical anchors, run paired composition-alpha probe per Catalog #322 |

## Section 12 — 9-dimension success checklist evidence per Catalog #294

1. **UNIQUENESS**: NEW `CONSUMES_SCORER_RESPONSE_DATASET` marker + NEW canonical equation `kl_distillation_scorer_surrogate_compression_savings_v1` + 3-surface convergence pattern not present in sister lanes.
2. **BEAUTY+ELEGANCE**: design memo per CLAUDE.md "Beauty, simplicity, and developer experience"; canonical helper `compute_kl_distillation_loss_t2` is ~10 LOC; sister marker is 1 line per consumer.
3. **DISTINCTNESS**: distinct from PR101/PR106 PROCEDURAL VARIANT BUILD DESIGN (different scope: cross-tool integration vs substrate audit) and PR101 GOLD NULL-BYTE REMOVAL SMOKE (different scope: cross-tool integration vs frontier null-byte smoke).
4. **RIGOR**: Catalog #229 PV (read 5 substrate files + 2 LL tools + L0 design memo + codex findings memo + #523 lane state) + Catalog #292 per-deliberation assumption surfacing in council frontmatter.
5. **OPTIMIZATION-PER-TECHNIQUE**: per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD — three structurally-different surfaces (decoder-internal Conv2d + response-table + HF-Jobs-mobilenetv3) each engineered for their optimal score-lowering path per CLAUDE.md "Quantizr intelligence" canonical anchor.
6. **STACK-OF-STACKS-COMPOSABILITY**: see Section 8 (composition-alpha estimates with DP1/VQ-VAE/PR101).
7. **DETERMINISTIC-REPRODUCIBILITY**: canonical Provenance per Catalog #323 on every artifact; fcntl-locked sister marker fan-out per Catalog #131 + #245.
8. **EXTREME-OPTIMIZATION-PERFORMANCE**: T1 engineering hooks already declared in PACT-NERV-DistilledScorer L0 design memo per `TIER_1_OPERATOR_REQUIRED_FLAGS`.
9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: predicted band [-0.003, +0.001] per HARD-EARNED-via-Quantizr-0.33-anchor; per Catalog #324 `predicted_band_validation_status: pending_post_training` (Stage 1 dispatch validates).

## Section 13 — Observability surface per Catalog #305

1. **Inspectable per layer**: PACT-NERV-DistilledScorer surrogate `forward` returns (B, 16) — operator hookable per-pair; canonical equation `kl_distillation_scorer_surrogate_compression_savings_v1` empirical anchors append to canonical registry per Catalog #344; LL dataset rows append via canonical CLI per Catalog #287/#323 Provenance.
2. **Decomposable per signal**: per Catalog #356 per-axis-decomposition canonical contract (sister WAVE-1-DIM-3-PROTOCOL-AND-HELPER); the distillation `kl_term` is a 4th additive term in the Lagrangian distinct from seg/pose/rate.
3. **Diff-able across runs**: byte-stable PDS archive grammar + canonical Provenance.
4. **Queryable post-hoc**: 4 query helpers proposed for sister-lane BUILD: `query_distilled_vs_direct_paired_smokes(dataset)`, `query_canonical_equation_empirical_anchors(equation_id)`, `query_substrate_composition_with_pact_nerv_distilled_scorer(pair_key)`, `query_523_image_level_surrogate_dispatch_outcomes()`.
5. **Cite-able**: provenance.json carries git_head + lane_id + canonical_provenance per Catalog #323; each canonical equation row carries `provenance` sub-object.
6. **Counterfactual-able**: byte-mutation smoke test per Catalog #139 (already declared in PACT-NERV-DistilledScorer L0 design memo).

## Section 14 — Sister-collision verdict + sister-DISJOINT scope

Verified sister-DISJOINT from:
- **In-flight PR101/PR106 PROCEDURAL VARIANT BUILD DESIGN** (commit `a97084e7`): different scope (substrate procedural-codebook variant design + per-pair audit) vs THIS lane (cross-tool integration design for substrate + LL planner + #523 convergence).
- **In-flight PR101 GOLD NULL-BYTE REMOVAL SMOKE** (commit `a3dfc84c`): different scope (frontier null-byte smoke + master-gradient empirical anchor) vs THIS lane (cross-tool integration design only).

STEP 0 sister-collision check via `tools/check_sister_files_recently_landed.py`:
PROCEED (no sister commits touched target file within 12-hour lookback).

Per Catalog #340 sister-checkpoint guard: sister subagents in flight do not
overlap with the file I am about to commit (`.omx/research/pact_nerv_distilled_scorer_x_codex_ll_integration_design_20260520.md`
+ this memo's sister landing memo).

## Section 15 — Operator-routable 5-step path

1. **Review NEW canonical equation `kl_distillation_scorer_surrogate_compression_savings_v1`** per Section 7 design spec. Register via `tac.canonical_equations.register_canonical_equation` IFF predicted band + producer/consumer contract pass council review. Per Catalog #344 EVOLUTION DISCIPLINE this registration is OPERATOR-ROUTED downstream (NOT THIS lane). Cost: $0.
2. **Stage 1 PACT-NERV-DistilledScorer per-substrate symposium per Catalog #325** within 14-day window BEFORE dispatch. T2+ sextet pact (Shannon + Dykstra + Rudin + Daubechies + Hinton + Contrarian + Assumption-Adversary) with Yousfi + Quantizr + PR95Author topical-grand addition. Verdict required: PROCEED or PROCEED_WITH_REVISIONS (DEFER_PENDING_EVIDENCE blocks dispatch). Cost: $0 (deliberation).
3. **Extend LL `scorer_response_dataset` row family taxonomy** to include `distilled_vs_direct_scorer_paired_smoke` per op-routable #3. Sister-lane BUILD adds CLI flag `--include-distilled-vs-direct-rows` + dataset schema extension. Cost: $0 (engineering only; no GPU until paired smoke).
4. **NEW cathedral consumer `distilled_scorer_surrogate_canonical_equation_consumer` BUILD** per op-routable #4. Package at `src/tac/cathedral_consumers/distilled_scorer_surrogate_canonical_equation_consumer/__init__.py` declares Catalog #335 canonical Protocol + Catalog #341 Tier-A markers + BOTH `CONSUMES_SCORER_RESPONSE_DATASET=True` AND `CONSUMES_MASTER_GRADIENT_ANCHORS=True` markers. Sister-lane BUILD. Cost: $0.
5. **Bind #523 as canonical SISTER lane to PACT-NERV-DistilledScorer** per op-routable #5. Lane registry update via `tools/lane_maturity.py mark` + memory entry cross-reference. When HF Jobs prepaid credit balance is replenished (currently 402 Payment Required) the #523 BUILD_1 dispatch fires AND produces the image-level surrogate paired with PACT-NERV-DistilledScorer's inside-decoder surrogate. Composition-alpha verified per Catalog #322. Cost: $0 (registry only); $5-15 when HF Jobs unblock (paid dispatch).

## Cross-references

- Substrate: `src/tac/substrates/pact_nerv_distilled_scorer/` (commit `b56f24bc1`)
- LL tools: `tools/plan_ll_scorer_response_next.py`, `tools/build_scorer_response_dataset.py`, `src/tac/optimization/scorer_response_dataset.py`
- #523 sister: `experiments/hf_jobs_segnet_surrogate_distillation.py`, `.omx/operator_authorize_recipes/substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch.yaml`, `lane_hf_jobs_segnet_surrogate_distillation_20260519`
- L0 design memo: `.omx/research/pact_nerv_distilled_scorer_l0_scaffold_design_20260520T211500Z.md`
- Codex findings: `.omx/research/codex_findings_ll_frame_pair_curriculum_masked_knobs_20260520T214749Z_codex.md`
- Council T1 working group: `.omx/research/council_t1_hf_jobs_segnet_surrogate_distillation_symposium_20260519.md`
- LL pair-4 guarded plan: `.omx/research/ll_scorer_response_pair4_guarded_plan_landed_20260521T004628Z_codex.md`
- Catalog gates: #124 + #125 + #220 + #229 + #233 + #240 + #265 + #270 + #272 + #287 + #290 + #294 + #296 + #303 + #305 + #309 + #313 + #322 + #323 + #325 + #335 + #340 + #341 + #344 + #346
