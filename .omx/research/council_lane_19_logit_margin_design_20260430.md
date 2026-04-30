# Council — Lane 19 (SegNet logit-margin boundary loss) Design Review

**Date:** 2026-04-30
**Status:** Phase A audit + design (pre-implementation)
**Anchor:** Lane G v3 = 1.05 [contest-CUDA] (DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL)
**Author:** Lane 19 implementer subagent
**Maturity target:** Level 1 (scaffold + 8 tests) → Level 3 (production hardened)

---

## 1. Existing scaffold audit

### Files already present
- `src/tac/losses_logit_margin.py` (213 LOC) — `fragility_weights()` + `logit_margin_loss()` with no silent defaults, threshold + reduction REQUIRED keyword, 2D/N-D shape handling, top-2 margin via `torch.topk`.
- `src/tac/tests/test_losses_logit_margin.py` (8 tests) — covers extremes, spatial dims, zero on confident, positive on ambiguous, gradient magnitude (boundary > confident), gradient direction (matches CE), no-silent-default rejection, shape-mismatch rejection.

### What is NOT present
- NO integration into `train_renderer.py` (loss is module-only).
- NO profile entry — needs `LANE_19_LOGIT_MARGIN` in `profiles.py`.
- NO `remote_lane_logit_margin.sh` dispatch script.
- NO STRICT preflight check.

### Existing related implementations (do not duplicate)
- `src/tac/constrained_gen.py:447-485` — TTO/optimize-poses hinge loss `relu(margin - (target_logits - max_wrong))`. Same Fridrich principle, different signature.
- `src/tac/losses.py:901` — `kl_distill_segnet_only` is the canonical SegNet-only auxiliary.
- `src/tac/losses_jbl.py` — `combined_jbl_distill_loss` (Lane J-JBL Jaccard + Boundary Label Smoothing).

### Key argparse facts (verified — FORBIDDEN PATTERNS check)
- `train_renderer.py` HAS `--loss-mode {standard,kl,jbl}` BUT the runtime allowlist `_VALID_LOSS_MODES` is wider: `("standard","kl","jbl","temperature","focal_ste","kl_distill","pcgrad","feature_match","posenet_embedding","segnet_kl")`.
- The allowlist regex truncates at first `)` so paren-free.
- `train_renderer.py` does NOT have `--segnet-loss-mode` or `--logit-margin-threshold`. Both must be ADDED via add_argument before any subprocess call references them.
- The `hinge` mode lives in `constrained_gen.py` (TTO path), NOT in train_renderer.py.

---

## 2. Design tradeoffs (council deliberation)

### Insight summary
SegNet score = argmax disagreement rate. For each pixel, only the top-2 logit ordering matters — confident-correct pixels (margin >> threshold) contribute zero to the score; confident-wrong pixels also contribute zero gradient via standard CE if already saturated; **boundary pixels (margin < threshold) are the ENTIRE signal**.

Standard CE wastes capacity on the easy 95%; logit-margin loss spends every gradient step on the 5% that flip.

### Yousfi (contest design)
**Position:** Endorse. The contest scorer is `100 × (argmax(student) ≠ argmax(teacher)).mean()`. Anything that moves a NON-boundary pixel's logits is wasted — it cannot change argmax. Anything that pushes a boundary pixel past the threshold IS the score.

**Concern:** Make sure the loss is computed on the SAME logit space the contest scorer evaluates (i.e. the SegNet outputs at the same resolution as the upstream evaluator: 192×256 after stride-2 stem). If we operate on rendered-frame logits at 384×512, the upsample will dilute the boundary mask.

**Verdict:** GREEN — proceed with caveat that fragility weights are computed on the SegNet-resolution logits matching the eval pipeline.

### Fridrich (steganalysis)
**Position:** Strong endorse. This IS UNIWARD applied to segmentation. Errors in confident regions are detector-imperceptible (high margin = robust to perturbation = no score signal). Boundaries are where the CNN's decision surface is ambiguous = exactly where bits should be spent.

**Concern:** The square-root law says concentrate small errors, don't concentrate large ones. Logit-margin pushes gradient INTO boundaries, which is the right direction at training time. But at inflate time the rendered frames don't have a margin signal directly — this is a TRAINING-time loss, not an inflate-time technique.

**Verdict:** GREEN — explicitly tag as "training-only auxiliary; never invoked at inflate".

### Quantizr ("I shipped standard CE — what's the empirical win?")
**Position:** Skeptical. Quantizr ships at 0.33 with `kl_on_logits(T=2.0)` — distillation, not margin loss. The KL distill ALREADY teaches the student to match the teacher's full distribution including margin information. What does fragility weighting buy on top of T=2.0 KL distill?

**Counter (Yousfi):** KL distill matches the WHOLE distribution including the wide-margin (confident) tail. Logit-margin loss weights the SAME KL/CE signal by `(threshold - margin) / threshold`, effectively nulling out the confident tail. So margin-weighted KL = "match the teacher at boundaries; don't waste capacity matching the easy pixels."

**Action:** Implement as a SUPPLEMENT to KL distill, not a replacement. Loss path becomes: `scorer_loss + kl_distill_weight × KL_T2 + logit_margin_weight × margin_loss(student_logits, teacher_argmax, threshold)`.

**Verdict:** YELLOW pending empirical A/B. Hypothesis: adds 5-15bp on top of Lane G v3 KL distill anchor. If A/B shows zero delta, KILL.

### Hotz ("just upweight boundary pixels in CE; same effect, simpler")
**Position:** Why not just `loss = CE × boundary_mask`? The fragility-weight formula `(threshold - margin)/threshold` is doing the same thing.

**Counter (Fridrich):** The advantage of fragility weights over a hard boundary mask: the weight is **continuous** in margin, so the gradient updates push pixels toward (or away from) the boundary smoothly. A hard mask creates discontinuities that the optimizer fights. Also — fragility uses the STUDENT's own logits, so as the student's confidence grows, the weight decays automatically. A hard mask using teacher boundaries wouldn't adapt.

**Verdict:** GREEN — but instrument the fragility-weight histogram per epoch so we can verify it's actually decaying as confidence grows (concrete falsification: if at epoch 1980 the mean fragility weight has not decayed below 0.3, the student isn't learning the boundaries).

### Selfcomp ("my hinge loss already does margin; what's new?")
**Position:** `relu(margin - (target_logit - max_wrong))` in constrained_gen.py is the hinge variant. It uses GT class as the target and pushes `correct_logit - max_wrong_logit ≥ margin`. Lane 19's logit_margin_loss uses `top1 - top2` (irrespective of GT) and weights CE by `(threshold - margin)/threshold`. Different formulas.

**Differentiation:**
- **Hinge (Selfcomp):** "punish if correct class not winning by ≥ margin". Hard loss = max(0, ...).
- **Lane 19 fragility-CE:** "scale CE by ambiguity". Soft weight ∈ [0, 1].

The hinge zeroes out for confident correct pixels (good — same as Lane 19) AND for pixels where correct class loses but max_wrong is not within margin (different — these get 0 hinge but POSITIVE fragility-CE if the top-2 are close). Hinge can MISS some confident-wrong pixels; Lane 19 catches them.

**Verdict:** GREEN — distinct from existing hinge; complementary not redundant. Council asks: should the loss support both? Decision: NO. Lane 19 stays scoped to fragility-CE; hinge stays in TTO path. Two lanes, two losses, separate measurement.

### Contrarian ("when does margin loss UNDERSHOOT the wrong class — i.e., increase argmax disagreement?")
**Strong concern:** Imagine pixel P with logits `[2.0, 1.9, 0, 0, 0]` and GT = class 1. Margin = 0.1 (very fragile). Standard CE pushes class 1 up. Lane 19 scales the same gradient by `(1.0 - 0.1)/1.0 = 0.9`. Direction: same. Magnitude: scaled. So fragility weight **never reverses direction** — it only modulates magnitude.

**But:** what about the CONFIDENT WRONG case? Pixel P with logits `[5.0, 0, 0, 0, 0]` and GT = class 3. Margin = 5.0 (>> threshold). Fragility weight = 0. Loss = 0. **The student NEVER LEARNS this pixel.** Standard CE would have given strong gradient toward class 3. Lane 19 silently abandons it.

**This is a real bug.** The student can have a **confidently wrong** pixel that contributes maximally to seg distortion, and Lane 19 ignores it.

**Mitigation:** Add a complementary "confident-wrong" weight that fires when (margin >= threshold) AND (argmax(student) ≠ argmax(teacher)). Two-mode loss:
- Boundary pixels: fragility-CE weight ∈ (0, 1]
- Confident-wrong pixels: full CE weight = 1.0
- Confident-correct pixels: zero

**Verdict:** YELLOW with mitigation. The current scaffold only handles mode 1. Mode 2 must be added, OR we explicitly use Lane 19 ONLY as an auxiliary alongside standard CE (which catches confident-wrong by default). DESIGN CHOICE: keep Lane 19 module API simple (boundary-only); ADD it as auxiliary to scorer_loss in train_renderer.py so confident-wrong is caught by the underlying scorer_loss path. This is the safest integration and matches how kl_distill_segnet_only is wired.

### Council verdict
**5 of 6 GREEN with conditions:**
1. **Yousfi:** GREEN if computed at SegNet output resolution.
2. **Fridrich:** GREEN; tag training-only.
3. **Quantizr:** YELLOW pending A/B; predict +5-15bp on top of KL distill.
4. **Hotz:** GREEN; instrument fragility-weight histogram per epoch.
5. **Selfcomp:** GREEN; complementary to TTO hinge.
6. **Contrarian:** YELLOW; confident-wrong pixels handled via co-existing scorer_loss, NOT by Lane 19 alone.

**Final decision:** Lane 19 ships as an **auxiliary loss alongside scorer_loss** (NOT as a replacement). Wire as new auxiliary controlled by `--logit-margin-weight` (off by default, 1.0 in the LANE_19_LOGIT_MARGIN profile). The new `loss_mode="logit_margin"` triggers the auxiliary; standard scorer_loss still fires (catches confident-wrong).

Predicted band [prediction]:
- **Floor [prediction]:** -1e-3 SegNet distortion, ~0.10 score points improvement on Lane G v3 anchor.
- **Ceiling [prediction]:** -3e-3 SegNet distortion, ~0.30 score points (matches phase 2 spec).
- **Kill criterion:** if A/B local smoke shows margin loss not converging to lower SegNet proxy than standard CE on identical 100-epoch run, demote to Phase 3 deferred.

---

## 3. Integration plan (Phases B-G)

### Phase B — module already exists; small enhancement
- File: `src/tac/losses_logit_margin.py`
- ADD: `logit_margin_loss_with_teacher(student_logits, teacher_logits, threshold, reduction)` helper that computes `gt_argmax = teacher_logits.argmax(dim=1)` and calls `logit_margin_loss`. This matches the `kl_distill_segnet_only` pattern of teacher-derived GT.
- ADD: `compute_segnet_logit_margin_aux(rendered_pair, gt_pair, segnet, threshold)` — teacher/student forward through SegNet (mirrors `kl_distill_segnet_only`).

### Phase C — additional tests
- A/B test: same checkpoint init, two short training runs (margin vs CE) → margin lower SegNet proxy.
- Teacher helper test: round-trip GT through SegNet → use teacher argmax as gt → loss decreases on confident-wrong pixels too.
- Determinism test: same seed → same loss value (±1e-6).
- Confident-wrong handling: when STUDENT confidently wrong, our auxiliary loss is zero BUT the underlying scorer_loss path catches it (test uses scorer_loss + margin together).

### Phase D — train_renderer.py wiring
1. ADD `add_argument("--logit-margin-weight", type=float, default=None, help="Lane 19 ...")`.
2. ADD `add_argument("--logit-margin-threshold", type=float, default=None, help="Lane 19 ...")`.
3. ADD `_resolve(args.logit_margin_weight, "logit_margin_weight", 0.0)` and threshold = 1.0.
4. ADD `"logit_margin"` to `_VALID_LOSS_MODES` allowlist.
5. ADD aux-loss block alongside the existing kl_distill block:
   ```python
   if args.logit_margin_weight > 0:
       from tac.losses_logit_margin import compute_segnet_logit_margin_aux
       aux_lm = compute_segnet_logit_margin_aux(
           rendered_pair, gt_pair, segnet,
           threshold=args.logit_margin_threshold,
       )
       loss = loss + args.logit_margin_weight * aux_lm
   ```
6. ADD profile `LANE_19_LOGIT_MARGIN` inheriting Lane G v3 + `loss_mode="logit_margin"`, `logit_margin_weight=0.1`, `logit_margin_threshold=1.0`.

### Phase E — local smoke
- Train 100 epochs on subsampled data with `--profile LANE_19_LOGIT_MARGIN` on CPU.
- Verify: loss decreases, EMA shadow updates, fragility-weight histogram printed.
- TAG result as `[smoke-only]` — strategically inert per CLAUDE.md non-negotiables.

### Phase F — contest-CUDA dispatch
- Vast.ai 4090, ~5h training × $0.25 + 30min auth eval × $0.25 = ~$1.50.
- Pattern A nohup detach (CLAUDE.md non-negotiable).
- Modal harvest backup if Vast.ai NVDEC bad-host.
- Tag result `[contest-CUDA]` only after `inflate.sh` → `upstream/evaluate.py` returns score on archive bytes.

### Phase G — STRICT preflight check
- Add `check_logit_margin_loss_uses_boundary_mask` to `src/tac/preflight.py`.
- AST scan: any call to `logit_margin_loss(` MUST pass an explicit `threshold=` kwarg (no positional default).
- STRICT @ 0 violations (the module already raises ValueError on missing threshold; preflight enforces caller-side hygiene).

---

## 4. CLAUDE.md non-negotiables checklist

- [x] EMA: training path uses `tac.training.EMA`; Lane 19 adds an aux loss to the existing path; no new code path bypasses EMA.
- [x] eval_roundtrip: training path uses `eval_roundtrip=True` from profile; Lane 19 piggybacks on the existing roundtripped pair.
- [x] auth eval at end: Lane 19 dispatch script ends with auth eval (CUDA).
- [x] No MPS for strategy: local smoke labeled `[smoke-only]`; only `[contest-CUDA]` is decision-grade.
- [x] No invented CLI flags: `--logit-margin-weight` + `--logit-margin-threshold` are NEW additions to train_renderer.py argparse and `_VALID_LOSS_MODES` (this file documents the addition).
- [x] No silent defaults: both new args default to `None`, profile resolver handles default.
- [x] No scorer at inflate: aux loss is training-only.
- [x] subagent_commit_serializer: all commits via `python tools/subagent_commit_serializer.py`.

---

## 5. Cross-refs

- CLAUDE.md FORBIDDEN PATTERNS (CLI flag inventions, default-to-convenience, etc.)
- CLAUDE.md "Exact scorer architectures" — SegNet uses EfficientNet-B2 stride-2 stem (boundaries lose half resolution).
- CLAUDE.md "SegNet paradigm shift" — SegNet 77x more important than PoseNet at our operating point.
- memory/project_phases_2_3_4_design_implementation_math_provenance_20260429.md §"Lane 19".
- memory/feedback_production_hardened_standard_definition_20260430.md (Level 3 definition).
- src/tac/losses.py:901 (`kl_distill_segnet_only` — sibling auxiliary).
- src/tac/losses_jbl.py (`combined_jbl_distill_loss` — sibling auxiliary).
- src/tac/constrained_gen.py:447 (TTO hinge — sibling but different scope).

---

**Council verdict:** PROCEED to Phase B with all conditions noted.
