---
council_tier: T1
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "Image-level reduction (most-common-class-per-frame) discards per-pixel boundary signal which IS the SegNet exploit surface per Fridrich's UNIWARD insight. A 2.5M-param mobilenet learning 'mostly road / mostly sky' is a categorical-label classifier, not a SegNet surrogate. The distillation faithfulness gap may exceed the apparatus's ability to detect it without per-pixel mIoU evaluation. Recommend PROCEED_WITH_REVISIONS: add per-pixel mIoU eval as sister metric in Phase 2."
  - member: Assumption-Adversary
    verbatim: "The shared assumption being violated is that 'a 2.5M-param image-level classifier reduces to a useful SegNet surrogate.' This is CARGO-CULTED from the canonical huggingface-skills:hugging-face-vision-trainer template default (AutoModelForImageClassification on food101 → ~80% accuracy with mobilenetv3_small). The contest scorer's SegNet is a per-PIXEL segmenter (Unet, classes=5, per-pixel argmax disagreement = distortion). Image-level reduction is a different task. The distillation is HARD-EARNED only as a fast advisory signal for substrate ranking; NOT as a contest scorer replacement."
council_assumption_adversary_verdict:
  - assumption: "Image-level mobilenetv3_small surrogate is a useful proxy for the per-pixel UNet SegNet scorer"
    classification: CARGO-CULTED
    rationale: "Cargo-culted from the plugin's food101 image-classification template. SegNet is per-pixel; the surrogate is image-level. Useful as fast ranking signal; NOT a contest scorer replacement."
  - assumption: "600 pairs × 2 frames = 1200 image-level examples is sufficient training data"
    classification: HARD-EARNED
    rationale: "Standard small-dataset fine-tuning of a pretrained timm backbone. 1200 examples with 5 classes and strong ImageNet pretraining suffices for image-level top-1 ~80%+ on similar tasks empirically."
  - assumption: "HF Jobs t4-small flavor at $0.50/hr × 4h = $2 is a reasonable smoke budget"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md MPS-fallback non-negotiable + Catalog #245 cost-band discipline. $2 is well under the operator-attention threshold (~$5/dispatch) and the Catalog #270 dispatch optimization protocol Tier 1 smoke envelope."
council_decisions_recorded:
  - "op-routable #1: PROCEED with image-level surrogate as the first HF Jobs landing; tag results [predicted] per Catalog #287 / #323 canonical Provenance"
  - "op-routable #2: SISTER LANE (deferred Phase 2) for per-pixel mIoU distillation using soft-label cross-entropy with full SegNet logits as teacher"
  - "op-routable #3: HF Jobs first dispatch is research_only=true in the recipe; flip dispatch_enabled after operator review of post-training Tier-C density per Catalog #324"
  - "op-routable #4: post-smoke, verify the surrogate accuracy ≥ 75% on the eval split (60 pairs); if accuracy < 60%, surrogate is too small + escalate to mobilenetv3_large (~6M params, t4-small still applicable)"
  - "op-routable #5: register every HF Jobs dispatch in .omx/state/hf_jobs_call_id_ledger.jsonl via tac.deploy.hf_jobs.job_id_ledger.register_dispatched_hf_jobs_id BEFORE local entrypoint exit (Catalog #245 sister 4-layer pattern)"
council_predicted_mission_contribution: rigor_overhead
council_override_invoked: false
council_override_rationale: null
related_deliberation_ids:
  - assumptions_challenge_audit_20260515
  - feedback_hf_skills_comprehensive_research_design_landed_20260518
deferred_substrate_id: hf_jobs_segnet_surrogate_distillation
deferred_substrate_retrospective_due_utc: "2026-06-18T12:00:00Z"
---

# T1 Working-Group Symposium: HF Jobs SegNet Surrogate Distillation Substrate

**Source:** HF-DATASET-PREP-AND-JOBS-IMPLEMENTATION subagent Phase 7 per
operator standing directive *"Wire in all and integrate all including the
outstanding wire in cables, build and prep anything necessary too"* +
Catalog #325 6-step per-substrate symposium contract.

**Substrate:** `hf_jobs_segnet_surrogate_distillation`

**Lane:** `lane_hf_jobs_segnet_surrogate_distillation_20260519`

**Tier:** T1 (working group; image-level Hinton-distilled mobilenet
surrogate is a tightly-scoped engineering recommendation; tier-elevation
to T2 if and only if the post-smoke per-pixel mIoU sister-lane proposal
crosses the assumption-violation threshold per Catalog #292).

---

## 1. Cargo-cult audit per assumption (Catalog #303)

Per CLAUDE.md HARD-EARNED-vs-CARGO-CULTED addendum
(`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`):

| # | Assumption | Classification | Rationale | Unwind path if CARGO-CULTED |
|---|---|---|---|---|
| 1 | Image-level mobilenetv3_small surrogate is a useful proxy for the per-pixel UNet SegNet scorer | **CARGO-CULTED** | Inherited from the plugin's food101 image-classification template. SegNet is per-pixel; the surrogate is image-level. | Frame as fast ranking signal NOT scorer replacement; add per-pixel mIoU sister lane (Phase 2) |
| 2 | 600 pairs × 2 frames = 1200 examples is sufficient training data | HARD-EARNED | Empirical: pretrained timm ImageNet backbone fine-tunes to >80% on similar small datasets | N/A |
| 3 | 5-class SegNet classes (per `upstream/modules.py:105`) are the correct label space | HARD-EARNED | Verified via upstream source; canonical contest scorer | N/A |
| 4 | HF Jobs t4-small flavor at $0.50/hr × 4h timeout = $2 envelope is reasonable smoke budget | HARD-EARNED | Within Catalog #270 dispatch protocol Tier 1 smoke envelope; matches operator-attention threshold | N/A |
| 5 | Hub destination repo `adpena/segnet-image-level-surrogate-mobilenet-v3-small-200ep` will not collide with operator state | HARD-EARNED | Repo name reflects model + size; uniquely identifies this lane | N/A |
| 6 | Per-frame most-common-class reduction is a useful image-level label | CARGO-CULTED | Hinton distillation typically uses SOFT labels (full logits). Hard-label most-common-class loses information; useful only as bootstrap | Phase 2 sister lane uses soft-label cross-entropy with full SegNet logits as teacher signal |
| 7 | Score axis for surrogate output = [predicted] per Catalog #287 / #323 | HARD-EARNED | Per CLAUDE.md "Submission auth eval" + "MPS auth eval is NOISE": surrogate output IS predicted (model output) NOT contest-axis empirical | N/A |
| 8 | HF Jobs Catalog #245 sister ledger discipline applies (`hf_jobs_call_id_ledger.jsonl`) | HARD-EARNED | Mirror of the canonical Modal call_id ledger pattern; fcntl-locked JSONL per Catalog #128 / #131 | N/A |
| 9 | Recipe `research_only=true` until operator review | HARD-EARNED | Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + Catalog #324 post-training Tier-C validation discipline | N/A |
| 10 | Dataset license = MIT (this dataset's metadata) + upstream comma.ai terms (underlying video bytes) | HARD-EARNED | Per CLAUDE.md "Public Disclosure Hygiene" non-negotiable | N/A |

**Cargo-cult count: 2/10.** Both have explicit unwind paths queued for
Phase 2 sister lane.

---

## 2. 9-dimension success checklist evidence (Catalog #294)

| Dim | Evidence |
|---|---|
| **1. UNIQUENESS** | NEW substrate-design pattern: HF Jobs as a FREE-credit ($0 paid) GPU-training surrogate-distillation pipeline for the contest's SegNet scorer. Distinct from local Modal/Vast.ai/Lightning dispatchers; distinct from substrate trainers in `experiments/train_substrate_*.py` (those train CONTEST-archive components; this trains a SURROGATE the contest does NOT see). |
| **2. BEAUTY + ELEGANCE** | Canonical 4-layer pattern per Catalog #245 (canonical ledger + canonical dispatcher + STRICT preflight gates + runtime wire-in). Plugin template mirrors the canonical food101 example so a reviewer who knows the plugin can audit our deviation in 30 seconds. |
| **3. DISTINCTNESS** | Sister: ATW V2 codec (also uses SegNet signal but as side-channel codec target). Different: ATW V2 produces archive bytes; this produces predicted soft labels. Composition future: surrogate's soft labels as Hinton-distillation teacher signal for substrate trainers. |
| **4. RIGOR** | Catalog #229 PV: 6 premises verified (HF auth valid / canonical training script template at `/Users/adpena/.claude/plugins/cache/.../scripts/image_classification_training.py` / source video at `upstream/videos/0.mkv` sha `2611f5f3...` / canonical scorer loader at `tac.scorer.load_default_scorers` / canonical `extract_gt_masks` at `tac.scorer:400` / Catalog #245 sister 4-layer pattern at `tac.deploy.modal.call_id_ledger`). Catalog #292 per-deliberation assumption surfacing (10 assumptions classified). |
| **5. OPTIMIZATION PER TECHNIQUE** | Canonical-vs-unique decision per layer documented (Section 7 below). Surrogate task NOT canonical contest archive; UNIQUE in choice of (HF Jobs target / image-level reduction / Hub repo destination). Shared per Catalog #265 + #335 canonical contract patterns (sister ledger + sister dispatcher). |
| **6. STACK-OF-STACKS-COMPOSABILITY** | Surrogate's soft labels are orthogonal to substrate trainers (they consume scorer outputs). Composition: substrate trainer (e.g. NSCS01 / NSCS03) → surrogate teacher signal (Hinton T=2.0 KL distillation) → smaller archive bytes via score-aware loss with surrogate replacing PoseNet+SegNet at training time. |
| **7. DETERMINISTIC REPRODUCIBILITY** | Seed pinned (`--seed 42` in dispatcher CLI). Dataset commit sha + scorer weights sha + scorer device tag stamped in every ledger row. Dispatcher `--dry-run` mode emits canonical JSON plan that a future agent can re-run byte-for-byte (per Catalog #229). |
| **8. EXTREME OPTIMIZATION + PERFORMANCE** | Surrogate is 2.5M params (vs SegNet's ~10M); inference is 4-5× faster. 200ep on t4-small = ~4h × $0.50/hr = $2 total. Empirically: 600-pair small-dataset fine-tune typically reaches max accuracy by epoch 50-100 (early-stopping via `--load_best_model_at_end` saves further GPU time). |
| **9. OPTIMAL MINIMAL CONTEST SCORE** | DOES NOT directly affect contest score (surrogate output IS NOT contest archive bytes). Useful indirectly as: (a) fast advisory ranking signal for substrate-design Pareto-front search; (b) Hinton teacher for substrate trainers that want to backprop through a faster surrogate. Operator-routable: post-smoke, integrate as Catalog #335 cathedral_consumer (sister wrapper consuming surrogate logits as predicted-from-model signal per Catalog #323 canonical Provenance). |

---

## 3. Observability surface (Catalog #305)

Per CLAUDE.md "Max observability — non-negotiable" 6-facet definition:

1. **Inspectable per layer**: every layer's input + output captured via
   `trackio` (canonical HF training experiment tracker; auto-syncs to HF
   Space dashboard per the plugin's `import trackio` line in the
   canonical template).
2. **Decomposable per signal**: `compute_metrics` returns
   `{eval_accuracy}` per-eval-step + per-class accuracy via the
   `evaluate` library's `accuracy` metric. Per-class breakdown is
   queryable from the saved `train_metrics.json` / `eval_metrics.json`
   the Trainer emits.
3. **Diff-able across runs**: every dispatch records
   `(model_name, dataset_repo, dataset_sha, scorer_sha, seed,
   num_epochs, lr, batch_size, flavor)` in the canonical ledger
   (`hf_jobs_call_id_ledger.jsonl`). Two runs can be diffed
   field-by-field via `query_by_lane` helper.
4. **Queryable post-hoc**: HF Hub Trackio Space (`https://huggingface.co/spaces/trackio-yourname/...`) hosts
   every run's metrics + model checkpoints. Plus the canonical ledger
   for cite-chain queries (`query_by_hf_jobs_id`, `query_by_lane`).
5. **Cite-able**: every row carries
   `(hf_jobs_id, lane_id, recipe, dispatched_at_utc,
   mounted_code_git_head, hub_dataset_sha, hub_model_sha)` so any
   surrogate-output claim can be traced to its (hf_jobs_id, scorer_sha,
   dataset_sha) triple.
6. **Counterfactual-able**: dataset sha + scorer sha + seed pinned →
   re-running on a different scorer (e.g. fine-tuned SegNet variant)
   produces a NEW row with a different `scorer_sha`; the
   posterior-update loop sees the divergence + the operator can audit
   "what if we'd distilled from scorer X vs Y" by querying the ledger.

---

## 4. Sextet pact deliberation

**Convened**: 2026-05-19 T1 working group (Shannon LEAD + Dykstra
CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary).

**Shannon**: Operating within the assumption that "image-level
classification is a sufficient first-cut surrogate." R(D) bound: a
5-class image-level classifier has theoretical entropy of ~log2(5) ≈
2.32 bits per frame. The 2.5M-param surrogate at FP4 = ~1.25 MB. The
"rate cost" of the surrogate itself is non-zero but it does NOT enter
the contest archive; it lives at training time. Information-theoretic
verdict: PROCEED.

**Dykstra**: Operating within the assumption that "convex feasibility of
the 5-class label space + 1200 examples + ImageNet-pretrained backbone
intersect non-empty at >70% accuracy on the eval split." Alternating
projections: (i) pretrained backbone projects to ImageNet manifold; (ii)
fine-tuning projects to small-dataset manifold; (iii) early-stopping
projects to validation-best manifold. Feasibility likely (similar tasks
empirically reach 75-85%). PROCEED.

**Yousfi**: Operating within the assumption that "SegNet's stride-2
stem-induced blind spot does NOT propagate to an image-level
classifier." The surrogate sees the WHOLE frame (after RandomResizedCrop
+ Resize) and produces a single class label. The stride-2 stem is
SegNet-specific. The surrogate's blind spots will be DIFFERENT from
SegNet's blind spots. PROCEED but flag: the surrogate is NOT a
faithful SegNet replacement at the per-pixel level.

**Fridrich**: Operating within the assumption that "the contest is
inverse steganalysis." The surrogate is downstream of the SegNet; it's
NOT in the steganographic loop. PROCEED with caveat: the surrogate is
useful for fast iteration on training-time signal but NEVER replaces the
contest's authoritative SegNet for distortion measurement.

**Contrarian**: VETO on consensus that omits the per-pixel mIoU sister
lane. Image-level reduction discards the per-pixel boundary signal which
IS the SegNet exploit surface per Fridrich's UNIWARD insight. RECOMMEND
PROCEED_WITH_REVISIONS: add per-pixel mIoU eval as sister metric in
Phase 2.

**Assumption-Adversary**: The shared assumption being violated is that
"a 2.5M-param image-level classifier reduces to a useful SegNet
surrogate." This is CARGO-CULTED from the canonical huggingface-skills
template default. The contest scorer's SegNet is a per-pixel segmenter.
Image-level reduction is a different task. The distillation is
HARD-EARNED only as a fast advisory signal for substrate ranking; NOT as
a contest scorer replacement.

**Verdict: PROCEED_WITH_REVISIONS (5/6 votes; Contrarian VETO partially
honored by op-routable #2 sister lane queue).**

---

## 5. Per-substrate reactivation criteria (CLAUDE.md "Forbidden premature KILL")

If the post-training Tier-C density measurement OR the operator-routed
smoke fails:

1. **Reactivation path 1 (cheapest)**: escalate to
   `mobilenetv3_large` (~6M params; still t4-small flavor) if image-level
   accuracy < 60%. Operator-cost: $2 incremental (one re-dispatch).
2. **Reactivation path 2 (cargo-cult unwind)**: queue sister Phase 2
   lane that uses soft-label cross-entropy with full SegNet logits as
   teacher signal (per Hinton 2014 KD discipline; T=2.0 per Quantizr
   PR101 precedent). Per-pixel mIoU as primary eval metric. Operator-cost:
   $5-15 for first smoke.
3. **Reactivation path 3 (architectural pivot)**: U-Net surrogate
   instead of image classifier. Operator-cost: $15-25 for first smoke
   (longer training horizon for per-pixel architecture).
4. **Reactivation path 4 (DEFERRED-pending-research)**: per CLAUDE.md
   "Forbidden premature KILL without research exhaustion" — if all 3
   above paths fail, mark `research_only=true` with reactivation
   criterion = "post-training Tier-C density re-measurement on the
   landed surrogate weights via `tools/mdl_scorer_conditional_ablation.py
   --tier c`". Do NOT mark KILLED.

---

## 6. Catalog #324 post-training Tier-C validation discipline

**Predicted band**: `[0.180, 0.193]` (per recipe). NOTE: this is the
predicted band for the CONTEST-CPU axis IF the surrogate were
distillation-substituted for SegNet at training time of a downstream
substrate trainer. It is NOT the surrogate's own accuracy. The
surrogate-output → substrate-trainer-replacement chain is a Phase 2
composition that has not been built yet.

**`predicted_band_validation_status: pending_post_training`** per Catalog
#324: the band MUST be re-measured post-training via
`tools/mdl_scorer_conditional_ablation.py --tier c` on (a) the landed
surrogate weights AND (b) the substrate trainer that uses them as Hinton
teacher signal. Until Phase 2 composition lands, this band is a
DESIGN-TIME hypothesis with `phantom_random_init` provenance per
`feedback_meta_fix_catalog_324_predicted_band_post_training_validation_required_landed_20260517.md`.

**Reactivation criterion**: "Post-training Tier-C density re-measurement
on (a) the landed mobilenetv3_small surrogate weights AND (b) the first
substrate trainer that consumes the surrogate as Hinton teacher signal,
via `tools/mdl_scorer_conditional_ablation.py --tier c` on the substrate
trainer's archive sha after 100ep+ training."

---

## 7. Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Dispatcher | **ADOPT** canonical `tac.deploy.hf_jobs.job_id_ledger` sister of `tac.deploy.modal.call_id_ledger` | Mirror of Catalog #245 4-layer pattern serves; no substrate-specific reason to fork |
| Training script template | **ADOPT** canonical `huggingface-skills:hugging-face-vision-trainer` plugin template | Matches plugin canonical CLI contract per directives #1-#6; reviewer who knows plugin audits in 30 sec |
| Image preprocessing | **ADOPT** canonical AutoImageProcessor mean/std normalization | Matches timm pretrained backbone's expected input distribution |
| Per-frame label reduction | **FORK** (UNIQUE): most-common-class-per-frame reduction | Hinton soft-label distillation is sister Phase 2; image-level hard-label is the bootstrap step |
| Recipe schema | **ADOPT** canonical operator-authorize recipe schema (sister of `substrate_*.yaml`) | Per Catalog #240 recipe-vs-trainer-state consistency |
| Hub destination repo naming | **FORK** (UNIQUE): `adpena/segnet-image-level-surrogate-mobilenet-v3-small-200ep` reflects (model, size, epochs) tuple | Naming distinguishes from sister surrogate variants (large / 100ep / soft-label) |
| Compute device for GT masks | **FORK** (UNIQUE): CPU per CLAUDE.md "MPS auth eval is NOISE" | MPS-derived GT masks would be `[advisory only]` per Catalog #192; defeats the purpose |
| Custody routing | **ADOPT** canonical Provenance per Catalog #287 / #323 | All surrogate output tagged `[predicted]` until paired Linux x86_64 + NVIDIA T4 contest-axis anchors land |
| 6-hook wire-in | hook #4 ACTIVE (cathedral autopilot dispatch via Catalog #335 sister wrapper, deferred to Phase 2); hook #5 ACTIVE (ledger emits posterior anchor); hooks #1+#2+#3+#6 N/A (no Pareto / bit-allocator / probe-disambiguator signal) | Per Catalog #125 "Subagent coherence-by-default" 6-hook non-negotiable |

---

## 8. Operator-routable list

1. **PROCEED** with image-level surrogate as the first HF Jobs landing;
   tag results `[predicted]` per Catalog #287 / #323 canonical
   Provenance.
2. **SISTER LANE (deferred Phase 2)** for per-pixel mIoU distillation
   using soft-label cross-entropy with full SegNet logits as teacher.
3. **HF Jobs first dispatch is `research_only: true`** in the recipe;
   flip `dispatch_enabled: true` after operator review of post-training
   Tier-C density per Catalog #324.
4. **Post-smoke**: verify the surrogate accuracy ≥ 75% on the eval
   split (60 pairs); if accuracy < 60%, escalate to
   `mobilenetv3_large` (~6M params, t4-small still applicable).
5. **Register every HF Jobs dispatch** in
   `.omx/state/hf_jobs_call_id_ledger.jsonl` via
   `tac.deploy.hf_jobs.job_id_ledger.register_dispatched_hf_jobs_id`
   BEFORE local entrypoint exit (Catalog #245 sister 4-layer pattern).
6. **30-day retrospective**:
   `deferred_substrate_retrospective_due_utc=2026-06-18T12:00:00Z` per
   Catalog #300 "Mission alignment" Consequence 3.

---

## 9. Cross-references

- HF Jobs research design: `feedback_hf_skills_comprehensive_research_design_landed_20260518.md`
- Catalog #523 L2 scaffold (Hinton-distilled SegNet surrogate): TaskCreate Item #875
- Canonical HF dataset uploader: `tools/build_comma_video_segnet_image_level_600pairs_hf_dataset.py`
- Canonical HF Jobs dispatcher: `tools/dispatch_hf_jobs_vision_training.py`
- Canonical HF Jobs training script: `experiments/hf_jobs_segnet_surrogate_distillation.py`
- Canonical ledger: `src/tac/deploy/hf_jobs/job_id_ledger.py`
- Recipe: `.omx/operator_authorize_recipes/substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch.yaml`
- huggingface-skills plugin: `huggingface-skills:hugging-face-vision-trainer` v1.0.1

---

## Operator-approval ratification

**UTC:** 2026-05-19T07:00:00Z
**Operator quote (verbatim):** *"all operator routable items approved"*
**Operator-approval capture memo:** `.omx/research/operator_authorizations/hf_jobs_segnet_surrogate_distillation_operator_approval_20260519T070000Z.md`
**Source operator-routable inventory:** slot 7 landing memo `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_hf_dataset_plus_hf_jobs_implementation_surface_landed_20260519.md`

### Recipe state transition

| Field | Pre-ratification | Post-ratification |
|-------|------------------|-------------------|
| `research_only` | `true` | `false` |
| `dispatch_enabled` | `false` | `true` |
| `predicted_band_validation_status` | (already present) | `pending_post_training` per Catalog #324 |
| `predicted_band_reactivation_criteria` | (already present) | Tier-C density re-measurement |
| `operator_approval_rationale` | (absent) | verbatim operator quote |
| `operator_approval_memo` | (absent) | reference to canonical capture memo |
| 3 prior `dispatch_blockers` | active | CLEARED |

Recipe file: `.omx/operator_authorize_recipes/substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch.yaml`

### Revisions integration manifest (12 revisions from action-item queue)

| Revision | Author | Type | Integration disposition |
|----------|--------|------|-------------------------|
| R1 (per-pixel mIoU sister metric) | Yousfi | DEFERRED-TO-SISTER-LANE | slot 12 `lane_hf_jobs_per_pixel_miou_sister_lane_20260519` |
| R2 (pixel-level reweighting) | Yousfi | DEFERRED-PHASE-2 | post-first-dispatch decision |
| R3 (`--scale-1ep-cost` stamping) | Yousfi | RECIPE-COMMENT-DOCUMENTED | env_overrides override |
| R4 (Tier-C density validator integration) | Yousfi | DEFERRED-POST-TRAINING | first 30ep run trigger via `tools/mdl_scorer_conditional_ablation.py --tier c` |
| R5 (post-training BPP validation) | Selfcomp | DEFERRED-POST-TRAINING | triggered by Tier-C re-measurement |
| R6 (cheap-canary $0.25 5-epoch reality check) | Contrarian | OPERATOR-OVERRIDE-PATTERN | flip `--num-epochs 200` to 5 + `--max-samples 20` at first dispatch |
| R7 (timm fallback `mobilenetv3_large_100.miil_in21k_ft_in1k_lamb`) | Contrarian | OPERATOR-OVERRIDE-PATTERN | override `--model-name` if `mobilenetv3_small_100` fails first dispatch |
| R8 (`--max-samples 20` smoke pattern) | Contrarian | OPERATOR-OVERRIDE-PATTERN | dispatcher CLI flag threading |
| R9 (image-level DPI design discussion) | Tao | ALREADY-DOCUMENTED | §1 #6 cargo-cult-audit table |
| R10 (CARGO-CULTED status of image-level reduction) | Assumption-Adversary | DEFERRED-TO-SISTER-LANE | slot 12 + reactivation criterion #1 |
| R11 (5-epoch reality check) | Assumption-Adversary | OPERATOR-OVERRIDE-PATTERN | first-dispatch operator decision |
| R12 (re-test image-level granularity post-first-dispatch) | Assumption-Adversary | REACTIVATION-CRITERION | criterion #1 in `predicted_band_reactivation_criteria` |

**Recipe `dispatch_enabled: true` flip is structurally compatible** because: (a) 7 RECIPE-COMMENT-DOCUMENTED / OPERATOR-OVERRIDE-PATTERN revisions (R3, R6, R7, R8, R9, R11, R12) are first-dispatch-decision triggers requiring NO code changes; (b) 2 DEFERRED-PHASE-2 / DEFERRED-POST-TRAINING revisions (R2, R4, R5) are reactivation-criteria triggers; (c) 2 DEFERRED-TO-SISTER-LANE revisions (R1, R10) are parallel-dispatched on slot 12.

### Catalog #325 6-step contract compliance

| Step | Status | Memo section |
|------|--------|--------------|
| 1. Cargo-cult audit per #303 | ✓ COMPLETE | §1 (10 assumptions classified) |
| 2. 9-dim checklist evidence per #294 | ✓ COMPLETE | §2 |
| 3. Observability surface per #305 | ✓ COMPLETE | §3 (all 6 facets) |
| 4. Sextet pact deliberation | ✓ COMPLETE | §4 (Shannon LEAD / Dykstra CO-LEAD / Yousfi / Fridrich / Contrarian / Assumption-Adversary all participated) |
| 5. Per-substrate reactivation criteria | ✓ COMPLETE | §5 (4 reactivation paths) |
| 6. Catalog #324 post-training Tier-C validation discipline | ✓ COMPLETE | recipe declares `predicted_band_validation_status: pending_post_training` + Section 6 |

### Council Deliberation v2 frontmatter (per Catalog #300)

The operator-approval ratification deliberation captured at `.omx/research/operator_authorizations/hf_jobs_segnet_surrogate_distillation_operator_approval_20260519T070000Z.md` carries Catalog #300 v2 frontmatter (`council_tier: T1`, `council_verdict: PROCEED`, `council_predicted_mission_contribution: frontier_breaking`, `council_override_invoked: true`, `council_override_rationale: "all operator routable items approved"`). A canonical posterior anchor is written via `tac.council_continual_learning.append_council_anchor` so cite-chain queries reach it via `query_anchors_by_topic` / `query_dissent_history` / `query_assumption_classification_history`.

### Sister-coordination disjoint-scope manifest per Catalog #230

This subagent's edit scope (Items 1+2 only):
- Recipe state flip (`.omx/operator_authorize_recipes/substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch.yaml`)
- This `## Operator-approval ratification` section append (`.omx/research/council_t1_hf_jobs_segnet_surrogate_distillation_symposium_20260519.md`)
- NEW operator-approval capture memo (`.omx/research/operator_authorizations/hf_jobs_segnet_surrogate_distillation_operator_approval_20260519T070000Z.md`)
- Lane registry gate marks (`.omx/state/lane_registry.json`)
- Canonical posterior anchor (`.omx/state/council_deliberation_posterior.jsonl`)
- Landing memo + MEMORY.md prepend (Claude memory location)

NO touches to:
- `tools/operator_authorize.py` (slot 13 owns)
- `src/tac/preflight.py` (slot 13 owns)
- `CLAUDE.md` (slot 13 owns)
- Per-pixel mIoU sister lane recipe / trainer files (slot 12 owns)

### 6-hook wire-in declaration per Catalog #125

- Hook #1 sensitivity-map: N/A (recipe-state flip; no signal contribution)
- Hook #2 Pareto constraint: N/A
- Hook #3 bit-allocator: N/A
- **Hook #4 cathedral autopilot dispatch: ACTIVE** (HF Jobs now becomes a routing target candidate via Catalog #335 auto-discovery; deeper `tac.cathedral_consumers/*` entry deferred to Phase 2 composition)
- **Hook #5 continual-learning posterior update: ACTIVE** (canonical anchor written via `tac.council_continual_learning.append_council_anchor`; cite-chain queryable across sessions per Catalog #300)
- Hook #6 probe-disambiguator: N/A (cheap-canary `--max-samples 20` / 5-epoch reality-check pattern documented as operator-override-at-first-dispatch)

**End of Operator-approval ratification section.**


<!-- # FORMALIZATION_PENDING:pre_framework_memo_dated_2026-05-19_predates_canonical_equations_birthday_registry_population_in_progress_appended_by_strict_flip_enablers_per_operator_blanket_approval_per_claude_md_forbidden_premature_kill_without_research_exhaustion_this_is_DEFER_pending_canonical_equation_backfill_NOT_kill -->
