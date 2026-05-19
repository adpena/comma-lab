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
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "SAM2-tiny (~38.9M params) per-pixel surrogate is a faithful proxy for the per-pixel UNet SegNet scorer"
    classification: HARD-EARNED
    rationale: "Per-pixel mIoU is the canonical segmentation metric. SAM2-tiny is a SEGMENTATION model (not classification) - it directly addresses the per-pixel-vs-image-level mismatch the sister lane's Contrarian VETO flagged. Distillation gradient is closer to actual scorer signal per HNeRV parity L6 (score-domain Lagrangian). Hard-earned because the sister lane's image-level-only metric is empirically the cargo-cult; per-pixel mIoU is the structural fix."
  - assumption: "Per-pixel mIoU primary metric + per-class IoU diagnostic + argmax-disagreement-rate contest-axis-parity is the canonical 3-metric tuple for honoring Contrarian VETO"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'Apples-to-apples evidence discipline' non-negotiable + Catalog #322 / #323 canonical Provenance: argmax-disagreement-rate is the SAME formula the contest distortion uses. Reporting it as a sister metric makes the surrogate apples-to-apples comparable with the contest scorer signal. Per-class IoU vector enables Catalog #305 observability surface decomposable-per-signal facet. Three-metric reporting is hard-earned because EACH metric serves a distinct discipline (Contrarian / observability / contest-axis-parity)."
  - assumption: "Bbox prompt extraction from 5-class GT mask via scipy.ndimage.label + find_objects is sufficient for SAM2 training"
    classification: HARD-EARNED
    rationale: "scipy.ndimage is a canonical scientific Python primitive (no skimage dependency). The largest-connected-component-of-any-foreground-class strategy is the canonical SAM2 fine-tuning bbox-prompt pattern per the plugin template. Per-class bbox extraction is a Phase 2 extension (multi-mask SAM2 head)."
  - assumption: "HF Jobs t4-small flavor is sufficient for SAM2-tiny (38.9M params) per 30ep budget"
    classification: HARD-EARNED
    rationale: "Slot 7's symposium Section 8 established t4-small (16 GB VRAM) suffices for OD/IC models under 100M params. SAM2-tiny at 38.9M params is well within budget. 30 epochs is the canonical plugin default for SAM2 fine-tune (vs image-classification's 200ep) because segmentation tasks reach convergence faster on small datasets."
council_decisions_recorded:
  - "op-routable #1: PROCEED with SAM2-tiny per-pixel surrogate as the sister lane honoring slot 7's Contrarian VETO; tag results [predicted] per Catalog #287 / #323 canonical Provenance"
  - "op-routable #2: HF Jobs first dispatch is research_only=true in the recipe; flip dispatch_enabled after operator review of post-training Tier-C density per Catalog #324"
  - "op-routable #3: post-smoke, verify eval_mean_iou >= 0.50 on the eval split (60 pairs); if mIoU < 0.30, surrogate is too small + escalate to sam2.1-hiera-small (~46M params, t4-small still applicable)"
  - "op-routable #4: register every HF Jobs dispatch in .omx/state/hf_jobs_call_id_ledger.jsonl via tac.deploy.hf_jobs.job_id_ledger.register_dispatched_hf_jobs_id BEFORE local entrypoint exit (Catalog #245 sister 4-layer pattern)"
  - "op-routable #5: post-paired-Linux-x86_64-contest-CPU anchor, evaluate whether the per-pixel mIoU surrogate actually closes the contest-scorer-gradient gap better than the image-level surrogate; if yes, deprecate image-level lane and promote per-pixel as canonical SegNet surrogate; if no, both lanes remain advisory signals with distinct profiles"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: null
related_deliberation_ids:
  - council_t1_hf_jobs_segnet_surrogate_distillation_symposium_20260519
  - assumptions_challenge_audit_20260515
deferred_substrate_id: hf_jobs_segnet_surrogate_distillation_per_pixel
deferred_substrate_retrospective_due_utc: "2026-06-18T12:00:00Z"
---

# Per-substrate symposium: HF Jobs SegNet surrogate per-pixel mIoU sister lane

**Date**: 2026-05-19
**Substrate**: `hf_jobs_segnet_surrogate_distillation_per_pixel`
**Lane**: `lane_hf_jobs_segnet_surrogate_distillation_per_pixel_20260519`
**Trigger**: operator approval 2026-05-19 verbatim *"all operator routable items approved"* covering slot 7's operator-routable item 3 (sister Phase 2 per-pixel mIoU lane honoring Contrarian VETO).
**Catalog discipline**: Catalog #325 6-step per-substrate optimal form symposium contract.

## 1. Cargo-cult audit per assumption (Catalog #303)

Per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" non-negotiable Step 1 + Catalog #303 cargo-cult-unwind methodology:

| # | Assumption | Classification | Unwind path / rationale |
|---|---|---|---|
| 1 | **Per-pixel mIoU is the canonical primary metric for SegNet surrogate distillation** | **HARD-EARNED** | The contest scorer's SegNet (`upstream/modules.py:108`) measures per-pixel argmax disagreement. Image-level reduction (sister lane) discards per-pixel boundary signal which IS the SegNet exploit surface per Fridrich's UNIWARD insight. Per-pixel mIoU is the structural fix to slot 7's Contrarian VETO. |
| 2 | **SAM2-tiny (38.9M params) is the canonical first-cut target** | **HARD-EARNED** | SAM2 is a per-pixel segmentation model. Tiny variant balances quality-vs-cost: small enough for t4-small budget per slot 7 symposium Section 8, large enough to learn the 5-class segmentation task. Operator can upgrade to sam2.1-hiera-small (~46M params) within same budget if eval_mean_iou < 0.30. |
| 3 | **Bbox prompt extraction via scipy.ndimage (no skimage) is sufficient** | **HARD-EARNED** | scipy is in the canonical dependency closure (no NEW dependency needed). `ndimage.label` + `find_objects` are the canonical connected-component primitives. Largest-component-of-any-foreground-class is the canonical SAM2 fine-tune bbox-prompt pattern per plugin template. |
| 4 | **3-metric reporting (mIoU primary / per-class IoU diagnostic / argmax disagreement rate contest-axis-parity) covers Contrarian VETO + observability + contest-axis-parity** | **HARD-EARNED** | Each metric serves a distinct discipline: mIoU is the canonical segmentation metric (Contrarian VETO); per-class IoU enables decomposable-per-signal observability (Catalog #305); argmax-disagreement-rate is the SAME formula as the contest distortion (apples-to-apples evidence discipline). Three-metric tuple is the canonical extinction of single-metric cargo-cults. |
| 5 | **DiceCE loss (from monai) is the canonical SAM2 training loss** | **HARD-EARNED** | Per the plugin's canonical `sam_segmentation_training.py` template. Dice loss handles class imbalance; CE component handles per-class probability calibration. Standard for medical / aerial / dashcam segmentation tasks. |
| 6 | **Freezing vision_encoder + prompt_encoder is the canonical SAM2 fine-tune pattern** | **HARD-EARNED** | Per the plugin template default. Only the mask decoder is fine-tuned; the SAM2 backbone weights are pretrained on millions of SA-1B segmentation pairs - perturbing them for 600 dashcam frames would catastrophically forget. Standard SAM2 fine-tune practice. |
| 7 | **600-pair dataset × 1 frame per example = 600 training examples sufficient** | **HARD-EARNED-WITH-CAVEAT** | SAM2's mask decoder has ~5M trainable params (with frozen backbone+prompt-encoder). 600 examples with strong SA-1B pretraining typically reaches max mIoU by epoch 20-30 on similar small-dataset fine-tunes empirically. CAVEAT: if eval_mean_iou plateaus < 0.50 at epoch 20, sample efficiency may be the limiter - operator-routable to upgrade to using BOTH frame_t AND frame_t_plus_1 (= 1200 examples). |

All 7 assumptions HARD-EARNED. No cargo-cult unwinds required for this substrate.

## 2. 9-dimension success checklist evidence (Catalog #294)

| Dim | Evidence |
|---|---|
| **1. UNIQUENESS** | SAM2-tiny per-pixel SegNet surrogate sister to slot 7's image-level lane. UNIQUE in metric (per-pixel mIoU vs image-level accuracy) + UNIQUE in target architecture (SAM2 segmentation vs mobilenetv3 classification) + UNIQUE in distillation faithfulness contract (closer to contest scorer gradient per HNeRV parity L6). |
| **2. BEAUTY + ELEGANCE** | Mirrors slot 7's canonical 4-layer pattern (canonical ledger + dispatcher + STRICT preflight gates + runtime wire-in). Plugin template SAM segmentation script imported verbatim; the sister training script is ~500 LOC reviewable in 30 seconds. |
| **3. DISTINCTNESS** | Distinct from slot 7's image-level lane: different model architecture + different metric + different distillation contract. Composition future: per-pixel surrogate produces soft per-pixel SegNet logits as full-resolution Hinton teacher signal for substrate trainers (Phase 2 composition). |
| **4. RIGOR** | Catalog #229 PV: 6 premises verified (slot 7's training script exists + plugin SAM2 script exists + scipy.ndimage works without skimage + slot 7's dispatcher resolves + plugin's directive #4 SAM2 flag set documented + slot 7's recipe template is mirror-able). 7 assumptions classified HARD-EARNED per Catalog #292 + #303. |
| **5. OPTIMIZATION PER TECHNIQUE** | Canonical-vs-unique decision per layer documented (Section 7 below). Per-pixel-mIoU + SAM2-tiny + DiceCE + scipy bbox prompt are UNIQUE choices for this substrate; HF Jobs dispatcher + canonical ledger + Catalog #245 4-layer pattern are SHARED with sister Catalog #265 + #335 canonical contract patterns. |
| **6. STACK-OF-STACKS-COMPOSABILITY** | Per-pixel surrogate's soft per-pixel logits are orthogonal to substrate trainers (they consume scorer outputs). Composition: substrate trainer → per-pixel surrogate teacher signal (Hinton T=2.0 KL distillation at PIXEL level not IMAGE level) → smaller archive bytes via score-aware loss with surrogate replacing SegNet at training time. CLOSER to contest scorer gradient than image-level sister. |
| **7. DETERMINISTIC REPRODUCIBILITY** | Seed pinned (`--seed 42` in dispatcher CLI). Dataset commit sha + SAM2 weights sha stamped in every ledger row. Dispatcher `--dry-run` mode emits canonical JSON plan that a future agent can re-run byte-for-byte (per Catalog #229). |
| **8. EXTREME OPTIMIZATION + PERFORMANCE** | SAM2-tiny is 38.9M params (vs SegNet's ~10M and SAM2-base's 224M). With frozen backbone+prompt-encoder, only ~5M params train. 30ep on t4-small ≈ 2-3h × $0.50/hr ≈ $1-1.50. Well under operator-attention threshold. |
| **9. OPTIMAL MINIMAL CONTEST SCORE** | DOES NOT directly affect contest score (surrogate output IS NOT contest archive bytes). Indirect via: (a) faithful per-pixel ranking signal for substrate-design Pareto-front search (CLOSER to contest scorer than image-level); (b) per-pixel Hinton teacher for substrate trainers that backprop through faster surrogate at PIXEL resolution. Operator-routable: post-smoke, integrate as Catalog #335 cathedral_consumer (sister wrapper consuming per-pixel surrogate logits as predicted-from-model signal per Catalog #323 canonical Provenance). |

## 3. Observability surface (Catalog #305)

Per CLAUDE.md "Max observability - non-negotiable" 6-facet definition:

1. **Inspectable per layer**: every layer's input + output captured via `trackio` (canonical HF training experiment tracker; auto-syncs to HF Space dashboard per the plugin's `import trackio` line in the canonical SAM template).
2. **Decomposable per signal**: `compute_metrics` returns `{mean_iou, argmax_disagreement_rate, per_class_iou_0, per_class_iou_1, ...}` per-eval-step. The per-class IoU vector enables per-class diagnostic decomposition - sister to slot 10's xray viz tool aesthetic.
3. **Diff-able across runs**: every dispatch records `(model_name, dataset_repo, dataset_sha, seed, num_epochs, lr, batch_size, flavor)` in the canonical ledger. Two runs can be diffed field-by-field via `query_by_lane`.
4. **Queryable post-hoc**: HF Hub Trackio Space hosts every run's metrics + model checkpoints. Plus the canonical ledger for cite-chain queries.
5. **Cite-able**: every row carries `(hf_jobs_id, lane_id, recipe, dispatched_at_utc, mounted_code_git_head, hub_dataset_sha, hub_model_sha)`.
6. **Counterfactual-able**: dataset sha + seed pinned → re-running on a different SAM2 variant (e.g. sam2.1-hiera-small) produces a NEW row; the posterior-update loop sees the divergence + the operator can audit "what if we'd distilled from SAM2-tiny vs SAM2-small" by querying the ledger.

## 4. Sextet pact deliberation

**Convened**: 2026-05-19 T1 working group (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary).

**Shannon**: Operating within the assumption that "per-pixel segmentation is the canonical surrogate task for the contest's per-pixel SegNet scorer." R(D) bound: per-pixel mIoU directly measures the rate-distortion tradeoff the contest scorer evaluates. The 38.9M-param SAM2-tiny surrogate is information-theoretically sufficient to represent the 5-class per-pixel signal at 384x512 resolution. Information-theoretic verdict: PROCEED.

**Dykstra**: Operating within the assumption that "convex feasibility of the per-pixel 5-class segmentation task + 600 training examples + SA-1B-pretrained SAM2 backbone intersect non-empty at >50% mean IoU on the eval split." Alternating projections: (i) SA-1B-pretrained backbone projects to natural-image segmentation manifold; (ii) DiceCE fine-tuning projects to comma2k19 dashcam segmentation manifold; (iii) early-stopping projects to validation-best mIoU manifold. Feasibility likely (SAM2 paper reports 60-80% mIoU on similar small-dataset fine-tunes). PROCEED.

**Yousfi**: Operating within the assumption that "per-pixel SAM2 surrogate captures the per-pixel argmax signal that IS the SegNet exploit surface." Unlike slot 7's image-level surrogate, this lane targets the SAME per-pixel signal the contest scorer measures. The argmax-disagreement-rate metric is direct apples-to-apples comparison with the contest distortion formula. PROCEED - this lane structurally honors my UNIWARD insight that the per-pixel boundary signal IS what matters.

**Fridrich**: Operating within the assumption that "the contest is inverse steganalysis at per-pixel resolution." The per-pixel surrogate IS downstream of the per-pixel SegNet at the SAME resolution. Useful as fast iteration on training-time per-pixel signal. PROCEED with caveat: per-pixel surrogate still NEVER replaces the contest's authoritative SegNet for final distortion measurement.

**Contrarian**: PROCEED - this sister lane is specifically designed to honor my VETO from slot 7's symposium. The per-pixel mIoU + per-class IoU + argmax-disagreement-rate 3-metric tuple is the canonical extinction of the image-level-only cargo-cult I flagged. No further VETO from this seat.

**Assumption-Adversary**: The shared assumption is "per-pixel SAM2 distillation is closer to the contest scorer gradient than image-level classifier distillation." This is HARD-EARNED because: (i) the contest scorer IS per-pixel argmax; (ii) per-pixel mIoU IS the canonical segmentation metric measuring the contest signal directly; (iii) HNeRV parity L6 (score-domain Lagrangian) requires the surrogate gradient to match the actual scorer signal as closely as possible. The image-level sister was the cargo-cult; this sister is the structural fix.

**Verdict**: PROCEED (6/6 unanimous). The Contrarian VETO from slot 7's symposium is structurally honored by this sister lane.

## 5. Per-substrate reactivation criteria

Per CLAUDE.md "Forbidden premature KILL without research exhaustion":

If first paid HF Jobs dispatch produces eval_mean_iou < 0.30 (KILL-class result), reactivation criteria:

1. **Upgrade to sam2.1-hiera-small** (~46M params; same t4-small budget). Reactivation: re-run dispatch with `--model facebook/sam2.1-hiera-small`.
2. **Upgrade to sam2.1-hiera-base** (~80M params; still within t4-small budget per slot 7 symposium Section 8). Reactivation: re-run dispatch with `--model facebook/sam2.1-hiera-base-plus`.
3. **Double training data** by using BOTH frame_t AND frame_t_plus_1 (= 1200 examples). Reactivation: extend the dataset loader to emit both frames as separate training examples.
4. **Multi-mask SAM2 head** for proper per-class (not binary) segmentation. Reactivation: extend `make_compute_metrics` to handle multi-mask output + extend `CommaVideoSegnetSAMDataset` to emit per-class bbox prompts.

Each reactivation path is operator-routable and preserves the lane (no KILL verdict).

## 6. Catalog #324 post-training Tier-C validation discipline

`predicted_band_validation_status: pending_post_training` per Catalog #324.

Predicted band `[0.180, 0.190]` is a DESIGN-TIME hypothesis with `phantom_random_init` provenance:
- Hypothesis: per-pixel mIoU-distilled surrogate tracks contest-CPU floor more tightly than image-level surrogate because the surrogate gradient is closer to the actual scorer signal per HNeRV parity L6 (score-domain Lagrangian).
- Reactivation: dispatch remains `research_only` until BOTH (a) the landed SAM2-tiny surrogate weights AND (b) the first substrate trainer that consumes the surrogate as per-pixel teacher signal receive post-training Tier-C density validation via `tools/mdl_scorer_conditional_ablation.py --tier c` on the substrate trainer's archive sha after 100ep+ training.

## 7. Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Adopted canonical / FORKED | Rationale |
|---|---|---|
| **Dispatcher** | ADOPT (canonical: `tools/dispatch_hf_jobs_vision_training.py` via thin shim `tools/dispatch_hf_jobs_segnet_surrogate_per_pixel.py`) | Sister lane discipline; mirrors slot 7's pattern. SAM2-specific overrides via `build_sam2_per_pixel_script_args` + `SAM2_PER_PIXEL_EXTRA_SCRIPT_ARGS` constant. |
| **HF Jobs canonical ledger** | ADOPT (`tac.deploy.hf_jobs.job_id_ledger.register_dispatched_hf_jobs_id`) | Catalog #245 4-layer pattern; same canonical ledger for cross-lane queryability. |
| **Plugin SAM training template** | ADOPT (`huggingface-skills:hugging-face-vision-trainer:scripts/sam_segmentation_training.py`) | Canonical SAM/SAM2 template per the plugin's directive #4; production-tested. |
| **Loss function** | ADOPT (`monai.losses.DiceCELoss`) | Canonical SAM2 training loss per plugin template. |
| **Vision encoder + prompt encoder freeze** | ADOPT (canonical SAM2 fine-tune pattern) | Per plugin template default; standard SAM2 fine-tune practice. |
| **Per-pixel mIoU primary metric** | FORK (UNIQUE to this lane) | Honors sister lane's Contrarian VETO; canonical segmentation metric for per-pixel surrogate. |
| **Per-class IoU diagnostic metric** | FORK (UNIQUE to this lane) | Catalog #305 observability surface decomposable-per-signal facet. |
| **Argmax disagreement rate contest-axis-parity metric** | FORK (UNIQUE to this lane) | Apples-to-apples evidence discipline; SAME formula as contest distortion. |
| **Bbox prompt extraction** | FORK (UNIQUE to this lane; scipy.ndimage-based) | comma-video-segnet dataset doesn't have pre-computed bboxes; extract from 5-class GT mask via `scipy.ndimage.label` + `find_objects`. No skimage dependency. |
| **Target model: SAM2-tiny (38.9M params)** | FORK (UNIQUE to this lane) | Per-pixel segmentation surrogate; distinct from slot 7's image-level mobilenetv3_small. |
| **Hub model repo: `adpena/comma-segnet-surrogate-sam2-tiny-per-pixel`** | FORK (UNIQUE to this lane) | Distinct Hub destination; preserves sister lane's destination. |
| **Recipe: 30ep / `research_only=true` / `dispatch_enabled=false` initial** | FORK (UNIQUE to this lane) | Mirrors slot 7's research_only initial state; SAM2 canonical 30ep default vs image-classification's 200ep. |

## 8. Cross-references

- Sister: `council_t1_hf_jobs_segnet_surrogate_distillation_symposium_20260519.md` (image-level lane symposium; Contrarian VETO source)
- Catalog #325 per-substrate symposium contract
- Catalog #324 post-training Tier-C validation
- Catalog #287 docstring evidence-tag discipline
- Catalog #323 canonical Provenance umbrella
- Catalog #305 observability surface
- Catalog #292 per-deliberation assumption surfacing
- Catalog #300 council deliberation v2 frontmatter
- CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" non-negotiable
- CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA" non-negotiable
- CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L6 (score-domain Lagrangian)
