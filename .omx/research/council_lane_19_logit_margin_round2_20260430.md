# Council Round 2 — Lane 19 Logit-Margin Loss

**Date:** 2026-04-30 (~10min after Round 1)
**Round:** 2 of 3 clean-pass gate (Round 1 found 3 issues → counter reset to 0)
**Status:** REVIEW IN PROGRESS

## Round 1 fixes verified

**Profile changes (commit pending):**
- `LANE_19_LOGIT_MARGIN.logit_margin_weight`: 0.1 → 10.0 (Contrarian)
- `LANE_19_LOGIT_MARGIN.kl_distill_weight`: 0.002 → 0.0 (Quantizr)
- Profile comment expanded with threshold tuning guidance (Fridrich)

**Verification:** `python -c "from tac.profiles import PROFILES; p = PROFILES['lane_19_logit_margin']; ..."` confirms post-fix values.

---

## Rotating perspective: MacKay (information theory + Bayesian / MDL)

**Concern:** From an MDL standpoint, Lane 19 weight × CE weighted by fragility is encoding prior belief that boundaries deserve more bits. But the rate term R(D) is computed at compression time, not training time. The training loss is a SURROGATE for R(D). Is the surrogate aligned with the true rate-distortion curve?

**Counter:** SegNet score (the contest scorer) IS the relevant distortion D. Logit-margin loss is a re-weighting of CE (a smoother surrogate of argmax disagreement). The rate term R is independent of where Lane 19 spends gradient (R depends on archive bytes, not training-time loss weighting).

**Sub-concern:** Could Lane 19 boundary preference INCREASE archive bytes (e.g., by encouraging more-confident logits → more-encodable masks)? Looking at the inflate path: archive bytes come from `archive.zip` containing `renderer.bin` (FP4-quantized weights — not affected by Lane 19) + `masks.mkv` (AV1-encoded; affected by mask predictions which are NOT directly trained, but rather generated from the training pipeline) + `optimized_poses.pt`. Lane 19 trains the renderer, not the mask encoder. **No rate impact.**

**MacKay: GREEN.** Information-theoretically clean: surrogate aligned with distortion target, no rate-side effect.

## Rotating perspective: Ballé (neural compression SOTA)

**Concern:** Lane 19 doesn't use any hyperprior or scale prior. The 2018 entropy bottleneck literature shows that conditional rate prediction beats fixed factorized priors. Could Lane 19's boundary mask serve as a CONDITIONING signal for a future hyperprior?

**Counter:** Yes — but that's a Phase 3 extension (Lane 20 Ballé hyperprior). Lane 19 itself is just a training-time loss, not a codec change.

**Sub-concern:** The fragility weights ARE a learnable conditioning signal — the per-pixel weight `(threshold - margin) / threshold` could be a hyperprior input for a future entropy model on the mask AV1 stream. Worth flagging.

**Ballé: GREEN with Phase 3 extension flagged.**

## Rotating perspective: Selfcomp (architectural fit, working 0.38)

**Concern:** I noticed in my pipeline that adding too many auxiliary losses creates gradient interference. With Lane G v3 having KL distill + JBL + uncertainty + frequency-aware, adding Lane 19 brings the count higher. Is gradient interference a real risk?

**Counter (looking at the code):** In `LANE_19_LOGIT_MARGIN` profile, KL distill is now 0.0 (Round 1 Quantizr fix). JBL is loss_mode-gated (loss_mode='logit_margin' bypasses JBL via the `if args.loss_mode == "jbl":` branch in train_renderer.py L3083). Uncertainty loss is profile-controlled (`use_uncertainty_loss`) — let me check Lane G v3.

**Verifying:** Lane G v3 anchor profile DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL inherits from DILATED_H64_HALF_FRAME. Need to grep for `use_uncertainty_loss` in either.

```
grep "use_uncertainty_loss" src/tac/profiles.py
```
Result: not in DILATED_H64_HALF_FRAME family — defaults to False.

**Lane 19 active losses:** scorer_loss (× 100) + Lane 19 aux (× 10). Two losses. KL distill OFF, JBL OFF, uncertainty OFF. Clean signal.

**Selfcomp: GREEN.**

## Rotating perspective: Hotz (raw engineering)

**Concern:** The `compute_segnet_logit_margin_aux` does student forward (with grad) + teacher forward (no grad). The training loop ALSO calls `scorer_loss` which calls SegNet on the same rendered_pair. So we now have:
- scorer_loss: SegNet(rendered_pair[:, 1]) — student forward (with grad)
- Lane 19 aux: SegNet(rendered_pair[:, 1]) — student forward (with grad) + SegNet(gt_pair[:, 1]) — teacher forward (no grad)

**Optimization opportunity:** scorer_loss + Lane 19 share the student forward. Can we cache logits across them? Looking at scorer_loss in losses.py:210:
```python
def scorer_loss(rendered_pair, ..., segnet, ...) -> tensor:
```
Returns a SCALAR loss, not logits. So we can't directly reuse logits.

**Sub-concern:** Each step now does 3 SegNet forward passes (1 for scorer_loss student, 1 for Lane 19 student, 1 for Lane 19 teacher). The student pass for scorer_loss IS redundant with Lane 19's student pass. If we refactor `compute_segnet_logit_margin_aux` to ACCEPT pre-computed student_logits, we save one forward pass.

**Action item:** Add an optional `student_logits` kwarg to `compute_segnet_logit_margin_aux`. If passed, skip the student forward. Defer the train_renderer.py wiring to share until profiling shows it's a hotspot.

**Decision:** GREEN with deferred optimization (not a correctness issue).

**Hotz: GREEN.**

## Rotating perspective: Karpathy (engineering practitioner)

**Concern:** The remote_lane_19 script Stage 1 invokes train_renderer with NO explicit `--logit-margin-weight 10.0` or `--logit-margin-threshold 1.0` — relies entirely on the profile resolver. If the profile resolver had a bug, the run would fall through to defaults (0.0 weight = OFF). This is the silent-default class CLAUDE.md warns about.

**Counter:** The resolver was just added. The default is 0.0 (deliberate; matches Lane G v3 byte-identity rule). The profile sets 10.0 / 1.0. The resolver chain: CLI > profile > default. If profile doesn't have `logit_margin_weight`, default 0.0 wins (silent OFF).

**Sub-concern:** Need to verify the profile-resolver wiring was tested. Round 1 fix put `logit_margin_weight=10.0` in the profile dict. The resolver call:
```python
args.logit_margin_weight = _resolve(
    getattr(args, "logit_margin_weight", None),
    "logit_margin_weight", 0.0,
)
```
The second arg `"logit_margin_weight"` is the profile key — must MATCH the dict key exactly. Confirmed. Default 0.0 is the safe sentinel.

**Action item:** Add a smoke test that `parse_args(['--profile', 'lane_19_logit_margin', '--tag', 'smoke'])` yields `args.logit_margin_weight == 10.0` (not 0.0). This locks the profile→resolver→args pipe against future drift.

**Karpathy: YELLOW** — add resolver test before Round 3.

## Rotating perspective: Schmidhuber / Ballé / van den Oord (compression/MDL/VQ-VAE deep dive)

**Concern (van den Oord):** VQ-VAE codebooks have an EMA decay of 0.99 (faster than weight EMA at 0.997) because codebooks adapt faster than weights. Does Lane 19 have any codebook-style EMA need?

**Counter:** Lane 19 is a training-time LOSS, not a parametric module. No codebook, no EMA, no quantization. The loss is stateless.

**Concern (Schmidhuber):** Compression-as-intelligence: does Lane 19 reward the renderer for COMPRESSING boundaries better, or just for matching teacher boundaries? Standard CE rewards matching; logit-margin reweighting amplifies the boundary signal but doesn't directly reward compression.

**Counter:** Compression in this contest = archive bytes. Lane 19 doesn't change archive structure. Distortion improvement (boundary pixels matching teacher) IS a compression-equivalent signal because better boundary match → fewer "compressed errors" the contest scorer can detect. Indirect, but valid.

**Verdict:** GREEN.

## Round 2 verdict

**1 issue found. Counter resets to 0.**

**Issues:**
1. **Karpathy: YELLOW.** Need a test that confirms profile-resolver chain produces `args.logit_margin_weight=10.0` from the profile (not the 0.0 default fallback). Locks against silent-default drift.

**Actions before Round 3:**
- Add `test_lane_19_profile_resolver_pipe()` to test_losses_logit_margin.py.

**No CRITICAL findings; the Round 1 fixes addressed all real bugs. The Round 2 finding is a regression-prevention test, not a fix.**
