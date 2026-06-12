# CORRECTION: the param-space / inflate.py-runtime / 30-min / unlimited-compress lever is a LIVE $0 door — "every door is a retrain" was TOO PESSIMISTIC (2026-06-11)

**Authority:** advisory (torch-CPU, NO MPS, $0). Frontier UNMOVED 0.19109982 [contest-CPU], 177,169 B,
sha b46897267d. This is a NO-FAKE catch-and-fix of the verdict in
`post_hoc_levers_exhausted_every_door_routes_through_retrain_20260611.md` (commit 0e1cfacb6), prompted by
the operator re-opening repair/waterfill "through multiple passes / post-hoc analysis / a different lens to
adapt to procedural," then naming "all the unexploited inflate.py rate stuff and 30-minute eval time" +
"unlimited [compute] outside of that." The re-opening was VINDICATED.

## The 3-lens repair re-opening result (the operator's "Go ahead")

| Lens | Verdict | Key measurement |
|---|---|---|
| Multi-pass + collateral-aware **pixel** repair | incomplete (agent offloaded to a monitor, returned non-final) — but subsumed by the pixel-collateral floor below | — |
| Multi-scale / low-freq **pixel** repair | collateral hypothesis CONFIRMED, but still collateral-FLOORED | coherent low-freq = **~43× less collateral/fix** than per-pixel (96.8×→2.23×); but collat/fix 2.23 > 1 so net d_seg still rises. Advanced the seg-sidecar to **LAW 0.847 (1.2× miss, tightest ever)** by dissolving the position-entropy term. Points explicitly at the decoder/latent axis as "the only lever where the correction direction is free." |
| **Param-space / decoder-weight** repair | **WINS — clears the collateral floor** | gradient-targeted delta on `rgb_1.weight` (frame-1 G-head `[1,9,1,1]`) NET-REDUCES exact d_seg with negligible pose cost (pixel repair RAISED d_seg every step). Byte-closed advisory candidate: **−5.46e-5** at +41 B (`.omx/tmp/cand_rate_optimal/archive.zip`, sha e4fadee26a; borderline — contest sign uncertain via the +0.0166 local→contest drift → FLAG, do not fire yet). |

**The structural finding:** the frontier's own DQS1 tail is ALREADY this lens — a *single-element* decoder-weight
delta applied **in inflate.py**, found by sweep. The param-space win is not a new mechanism; it is the
**gradient-targeted + multi-element extension** of a lever the frontier already rides.

## Why the prior verdict was wrong (the catch-and-fix)

The "every door routes through a retrain" verdict treated repair as **pixel-domain byte-transforms** and,
finding those collateral-floored, concluded the $0 toolbox was empty. It **conflated two different things**:
1. "No more post-hoc PIXEL byte-transforms move the score" — TRUE (pixel repair is collateral-floored,
   re-confirmed by the multi-scale lens).
2. "No more COMPUTE-to-find-a-better-archive moves the score" — **FALSE.** The param-space / DQS1 lens is a
   COMPRESS-TIME-searched, INFLATE-TIME-applied procedural correction that clears the floor — and it has two
   large, un-exhausted headroom axes the verdict ignored.

## The two un-exploited levers (operator, 2026-06-11) — the live door

1. **Inflate.py runtime + 30-min eval budget (inflate-time).** `inflate.sh` may run up to ~30 min on T4
   (`upstream/README.md` budget). The archive stores a tiny delta/program; inflate.py COMPUTES the repair at
   decode. Richer procedural decode (multi-element DQS1, iterative refinement, computed-vs-stored, expanded
   legal witness-program grammar) → SMALLER archive for the same scored frames → cuts the 62%-of-score rate
   term. The budget is almost entirely unspent.
2. **Unlimited compute OUTSIDE that (compress-time).** Offline, our side, we may search/optimize the
   inflate-time program as hard as we want — exhaustive multi-element search, deep gradient optimization,
   program search — to find the archive+inflate that minimizes exact S. Agent 2 used ONE VJP pass; unlimited
   compress-time means a far deeper search of the same grammar.

This is the **"evaluator-equivalent witness compiler" / "Native eval-time runtime discipline"** paradigm
already in CLAUDE.md. Hard constraint (NO-FAKE): the inflate decoder stays **self-contained, ≤30 min, with NO
learned/video-derived constant outside archive.zip** (the constant lives IN the zip; inflate computes from it).

## Corrected disposition + the next doors

- The pointer is still UNMOVED, and the −5.46e-5 single-element candidate is too borderline to spend a paired
  eval on (MVP-first: extend for a bigger, clearer-sign margin first).
- **NOT every door is a retrain.** The live $0/cheap door is: **unlimited-compress search over a multi-element
  inflate-time procedural grammar (DQS1 and richer), applied within the 30-min budget.** A retrain remains a
  parallel (paid, larger) bet, but it is no longer the *only* door.
- Two forward investigations launched: (A) **unlimited-compress multi-element gradient-DQS1 search** — extend
  agent 2's single element to the multi-element optimum, byte-close + advisory S, find the maximal move;
  (B) **inflate.py + 30-min procedural-rate-lever inventory + top prototype** — enumerate what richer
  inflate-time decode (within budget) shrinks the archive, rank by $0/cheap frontier-move EV, prototype the top.
- Discipline unchanged: torch-CPU advisory is the GATE; recompute full S from components on the byte-closed
  archive (rate win must pay for sidecar bytes AND any d_seg/d_pose collateral); claimed ≠ measured; a clear
  advisory sub-0.19110 with margin > the local→contest drift → FLAG for operator-gated paired CPU+CUDA eval.

## Bottom line (non-sycophantic)

The operator's re-opening found the hole in my own verdict: pixel repair IS floored, but the
param-space/inflate-runtime axis — searched with unlimited compress-time compute and applied within the unspent
30-min inflate budget — is a genuinely live, un-exhausted rate door that does NOT require a retrain. The
frontier already rides its single-element version; the multi-element extension is the immediate next exact-row
target. Frontier UNMOVED 0.19110; this corrects the map and aims the next units at the live door.

## ⚠️⚠️ THIS CORRECTION IS ITSELF CORRECTED (2026-06-11, agent ac3751b8 — NO-FAKE catch) ⚠️⚠️

The "param-space −5.46e-5 win" this memo rests on was a **HARNESS ARTIFACT, not a win.** On the CALIBRATED
contest chain (local `upstream/evaluate.py --device cpu` on 0.mkv, 600 pairs, reproducing the frontier at
**0.19110976, Δ +9.9e-6**), the param-space candidate (idx247 `rgb_1.weight` delta, +41 B) is **+6.02e-5
WORSE, not −5.46e-5 better** — both d_seg AND rate move the wrong way. Root cause: the sister harness
**mis-decoded the SegNet GT** (384×512 bicubic + bilinear no-op) vs the contest chain (camera-res 874×1164 →
bilinear-resize to 384×512), inflating d_seg by **1.296×**; the 72-pair selection was optimized against that
wrong inflated-GT argmax pattern and does not transfer. The sister's entire gradient field
(`.omx/tmp/param_grad_full600_rows.jsonl`) is computed on the inflated GT and is **INVALID for selection**.
The frontier already embodies the single-element DQS1 optimum (its own element helps 8/8 scanned pairs;
idx247 broadly HURTS on the true GT).

**What this memo got WRONG and the corrected position:**
- **WITHDRAWN:** "param-space/inflate-runtime is a LIVE $0 door that does NOT require a retrain." It rested on
  a fake win. The frozen-frontier $0 byte-transform toolbox (rate [agent B], DQS1 single+multi [agent A],
  pixel-repair [multi-scale agent]) IS exhausted — the original "post-hoc exhausted, the door is a retrain"
  verdict (`post_hoc_levers_exhausted_..._20260611.md`) is RE-VINDICATED, not overturned.
- **STILL VALID (but reclassified):** the "compress-time trains the inflate-time decoder / unlimited-compress /
  30-min budget" framing is a real DESIGN AXIS — but it is a **RETRAIN-CLASS lever** (training a decoder/program),
  consistent with "the door is a retrain." It refines WHAT to train (correctly-recipe'd, possibly compute-heavy),
  not WHETHER to train. It is NOT a $0 byte-transform.
- **NEW system-intelligence (the durable win):** the canonical $0 advisory gate is **local `evaluate.py
  --device cpu`**, which reproduces the frontier within ~1e-5 (agent A's `experiments/results/dqs1_multielement_20260611/canonical_chain_scan.py`
  reuses the contest `DistortionNet` directly). Advisory harnesses that decode GT at 192×256 or pre-resize to
  384×512 are NON-CANONICAL and inflate/deflate absolute d_seg — so **some prior "walls" (0.014, 0.0025) may be
  partly harness-resolution artifacts**, which reinforces (does not replace) the live seg-convergence recipe
  + vendored-vs-port investigations: they MUST measure on the canonical chain.

Frontier UNMOVED 0.19110. Net: no $0 frozen-frontier door remains; the live door is a correctly-recipe'd
(seg-convergent) retrain, measured on the canonical chain — exactly what agents a28f8a9c / a173958b test.
