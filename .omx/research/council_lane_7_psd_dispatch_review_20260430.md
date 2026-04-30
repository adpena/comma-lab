# Council Lane 7 PSD Dispatch Review — 2026-04-30

**Convener**: parent agent (Lane 7 PSD dispatch gate)
**Inner council (10)**: Shannon (LEAD), Dykstra (CO-LEAD), Yousfi, Fridrich, Contrarian, Quantizr, Hotz, Selfcomp, MacKay (memorial), Ballé
**Question**: Should we dispatch Lane 7 PSD (`PSD_STANDARD_ADAPTIVE` profile) NOW, with Lane G v3 = 1.05 [contest-CUDA] as the standing bar?

---

## 0. Critical Empirical Evidence (load-bearing, all 10 voices read this BEFORE casting)

The council CANNOT pretend this evidence does not exist. PSD has been auth-eval'd in this codebase before:

1. **`src/tac/research/competition_state.py:131`** — `psd_architecture` is on the `killed_techniques` list with verdict: `"Auth eval 1.49 vs dilated 1.33. Worse."`
2. **`memory/project_psd_auth_eval_verdict.md`** (2026-04-11, 19 days ago) — PSD h=64, ep 809: **auth score 1.49** (pose=0.01108, seg=0.00532, rate=0.02522). SegNet was 12.8% better than dilated baseline, but **PoseNet was 5x worse** and the **95KB model added a rate penalty of ~0.033** vs the 46KB dilated.
3. **`memory/project_psd_breakthrough.md`** — the only "breakthrough" (1.38 at ep 69) was achieved with **KL distill auxiliary**. KL distill is on the killed_techniques list at `competition_state.py:126` ("2 auth evals: 1.85 and 2.05. PoseNet collapse").
4. **Lane G v3 = 1.05 [contest-CUDA]** is the standing bar. PSD's verified historical landing of 1.49 is **42% WORSE** than the current bar.
5. **`PSD_STANDARD_ADAPTIVE` profile** (`src/tac/profiles.py:168`) does NOT include KL distill. It is `PROVEN_BASELINE + variant=psd + boundary_weight=50 + hard_frame_ratio=0.3 + use_swa=True`. This is the same regime that produced 1.49 PLUS three orthogonal additions (boundary weighting, hard-frame replay, SWA).

The question for the council is therefore: **do the three additions (boundary_weight=50, hard_frame_ratio=0.3, use_swa=True) plausibly recover the 0.44-point gap from 1.49 to 1.05** AND further improve to <1.05 to justify the GPU spend, GIVEN that the architectural cause of PSD's 5× PoseNet regression (rate-resolution mismatch on Yousfi's 12-channel YUV6 PoseNet input) is unaddressed?

---

## 1. Per-voice positions (math/empirical only, no conservative bias)

### Shannon (LEAD) — REJECT

PSD's `PixelUnshuffle(2)` + 4-conv body + `PixelShuffle(2)` produces ~87-95K params at h=64 (per `project_psd_breakthrough.md` measurement). FP4 packing of 87K params is ~21.75 KB; FP4A with codebook is ~16-20 KB. Lane A renderer (88K dilated) = 16.5 KB FP4A, hits ~1.5 bits/weight. PSD at h=64 is **structurally similar in params to Lane A**, so the bit-cost is comparable — there is no rate-distortion advantage from architecture alone.

The key R(D) question is **does PSD's half-resolution bottleneck increase or decrease the Shannon mutual information between (mask, frame) pair and the optimal renderer output at the SegNet scoring resolution (512×384)?** PSD operates at 582×437 (post-PixelUnshuffle from 1164×874), which is `1.14×` the SegNet resolution and `0.50×` the camera resolution. This is geometrically aligned with SegNet (the 12.8% SegNet improvement empirically confirmed in 2026-04-11), but **misaligned with PoseNet's 12-channel YUV6 input at 512×384** which expects fine luma detail. Empirical 5× PoseNet regression matches this prediction.

`PSD_STANDARD_ADAPTIVE`'s additions (boundary_weight=50, SWA) do not change the bit-cost or the resolution-mismatch problem; they are training-time stabilizers, not architectural fixes for the PoseNet-rate-resolution mismatch.

**No information-theoretic mechanism** plausibly closes the 1.49 → <1.05 gap. **REJECT.**

### Dykstra (CO-LEAD) — REJECT

Pareto-feasibility analysis: the achievable region is the intersection of {seg ≤ 0.0029 [Lane G v3 anchor], pose ≤ 0.000931 [Lane G v3 anchor], rate ≤ 0.025 [Lane G v3 archive]}.

PSD's verified historical operating point at ep 809: seg=0.00532, pose=0.01108, rate=0.02522. The pose constraint is violated by **11.9×**. Even projecting `PSD_STANDARD_ADAPTIVE`'s additions optimistically (boundary_weight halves seg toward Lane G v3, SWA stabilizes ep ~2000 to recover pose by 2-3×), the pose bound is still violated by 4-6×. **No alternating-projections trajectory from PSD's basin reaches the {seg, pose, rate} feasibility set without a backbone change.**

Dykstra's projection-iteration ceiling computation: with PSD's empirical pose floor of 0.011 and Lane G v3's 0.000931, the PoseNet "PSD residual gap" is 0.0101 absolute. Score arithmetic (pose contributes √(10·pose) = √0.110 = 0.332). PSD floor on PoseNet ALONE is 0.332. Plus seg contribution (~0.50 with optimistic projection) plus rate (0.63). **PSD floor: 1.46 — below the 1.49 ep-809 measurement but well above Lane G v3's 1.05.**

**REJECT** — outside the achievable region for sub-1.05.

### Yousfi — REJECT

Contest-design perspective: PoseNet is FastViT-T12 with 12-channel YUV6 input (rgb_to_yuv6 → resize to 512×384 → normalize mean=127.5 std=63.75). The first stride-2 stem of FastViT halves to 256×192. PSD's PixelUnshuffle(2) operates at 582×437, which is **1.14×** the SegNet input resolution but **2.27×** the FastViT post-stem resolution. The half-resolution bottleneck in PSD destroys the high-frequency luma detail that FastViT's attention layers actually use to compute pose deltas (vehicle micro-motion, lane-marker phase). This is **not a training problem**; it is a representation-capacity problem. SWA averages don't help when the representation is missing.

The 1.49 measurement IS the contest-true score on this architecture. The `PSD_STANDARD_ADAPTIVE` profile's additions might tighten the SegNet score by 5-10% (boundary_weight on a SegNet-aligned architecture is plausibly synergistic), but the PoseNet ceiling is fixed by the architecture. **REJECT.**

### Fridrich — REJECT (with one narrow opening)

Inverse-steganalysis perspective: PSD's half-resolution operation is steganalysis-aligned for SegNet (EfficientNet-B2's stride-2 stem also loses half resolution at the top, so PSD's bottleneck "mirrors" the scorer's blind spot at 256×192) — this is the mechanism behind the 12.8% SegNet improvement. **But PoseNet uses a DIFFERENT scorer architecture** (FastViT-T12 with attention) that does NOT have an aggressive resolution-loss stem. PSD's bottleneck is not aligned with PoseNet's blind spots.

**Narrow opening (which I am NOT recommending we pursue today):** if PSD were combined with a PoseNet-aware luma-skip path (full-resolution luma residual added back at the output, similar to Lane LCT's late-stage chroma skip), the 5× PoseNet regression could plausibly close to 2-3× — putting the score in the [1.10, 1.25] band. That's still WORSE than Lane G v3's 1.05. The mechanism does not exist in PSD_STANDARD_ADAPTIVE today.

**REJECT** for tonight's dispatch. Reactivation criteria: a luma-skip-PSD variant (Lane PSD-LumaSkip) is a separate design decision requiring its own council review.

### Contrarian — REJECT (and challenges any APPROVE bold-but-underspecified claim)

The contrarian's role is to challenge weak arguments, not bold ones. Let me name the bold-but-underspecified claims I am pre-emptively rejecting:

1. **"PSD with adaptive boundary + SWA might be different this time"** — bold, but unspecified WHICH mechanism in PSD_STANDARD_ADAPTIVE addresses the 5× PoseNet regression. Boundary weighting affects seg loss, not pose. SWA averages weights, doesn't change architecture. Hard frame replay biases sample selection toward errors but doesn't fix the rate-resolution mismatch. **None of the three additions plausibly closes a 5× PoseNet gap.**
2. **"Lane G v3 used a posture-different optimization (pose TTO from Lane A flow), maybe PSD just needs that too"** — bold but unspecified. Lane G v3 is built on Lane A's RENDERER (dilated, not PSD). Adding pose TTO to a PSD renderer is a Lane PSD+G hybrid, which is a separate design and a separate dispatch. It is NOT this dispatch.
3. **"We don't have a remote_lane script for PSD — that's a feature, not a bug"** — this is the ONLY argument the contrarian endorses. The absence of `scripts/remote_lane_psd.sh` despite PSD being in `profiles.py:168` since at least 2026-04-10 is **structural evidence** that the project agreed PSD is not worth building infrastructure for. The 2026-04-11 verdict ("Stay With Dilated") was correct then and remains correct.

**REJECT.**

### Quantizr (adversarial) — REJECT

Jimmy/Quantizr shipped 0.33 with 88K-param FiLM-conditioned depthwise-separable CNN. He did NOT use PSD. The architectural choice is informative: if PSD's PixelUnshuffle bottleneck were a competitive advantage, Quantizr (who explicitly tested "sweeping conv dims" in his own writeup) would have found it. He chose **depthwise-separable + FiLM** which is a DIFFERENT mechanism for parameter efficiency. The fact that the leader did not use PSD is not by itself dispositive, but combined with the empirical 1.49 evidence, it adds Bayesian weight to the rejection.

The counterargument I considered: maybe Quantizr never tested PSD because it was non-obvious. But the PSD insight (resolution alignment with SegNet) is documented in our own memory from 2026-04-10 and is reproducible from any architecture-search run. If it were strong, it would have surfaced in someone's leaderboard run by now.

**REJECT.**

### Hotz — REJECT

What's the 30-minute version? Smoke-test PSD_STANDARD_ADAPTIVE for 100 epochs at h=16 on local MPS. But MPS gives garbage scores per CLAUDE.md non-negotiable. So the 30-minute version is NOT possible without GPU.

The minimum responsible dispatch is: 4090 @ $0.25/hr × 5h training × 1 auth eval = $1.25 + 15 min = **the math doesn't work**. We are spending $1.25 to test a hypothesis with empirical prior probability **<5%** of beating the 1.05 bar. Expected value calculation: 0.05 × 0.20 (best case 0.85 score, beats by 0.20) - 0.95 × 0 (baseline holds at 1.05) = +0.01 score gain expected, at cost $1.25. That's **$125 per 0.01 score point** — terrible EV vs the alternative dispatches in tonight's queue (Phase 2 Ballé hyperprior at predicted Δ -0.04 for $1).

**REJECT.** Better $1.25 use: a controlled Phase 2 Lane 19 SegNet logit-margin A/B on Lane G v3.

### Selfcomp — REJECT

I shipped 0.38 with a 94K-param SegMap (different architecture from PSD; see `project_selfcomp_reverse_engineered_20260429.md`). PSD was a candidate I considered and skipped — not because of conservatism, but because the half-resolution bottleneck collides with the PoseNet input. My grayscale-LUT mask paradigm achieves resolution-rate efficiency through a DIFFERENT mechanism (analog continuous masks, not discrete classes), which sidesteps the PSD trade-off entirely.

If you wanted to retry PSD, the only configuration with a chance is:
- Bolt PSD onto a PoseNet-aware luma path (full-res chroma+luma skip, low-res seg refinement)
- Train with my block-FP 1.017 bpw codec (saves 30% rate vs 1.5 bpw FP4A)
- Initialize from Lane G v3's renderer state-dict (warm-start, not from scratch)

That's a different lane. PSD_STANDARD_ADAPTIVE as currently specified does **none of those things**. **REJECT.**

### MacKay (memorial) — REJECT

MDL question: what's the rate cost of the additional PSD-specific weights (PixelUnshuffle's pixel-shuffle 4× channel-multiplier + the conv stack at the new 12-channel input + the PixelShuffle inverse)? Compared to dilated h=64 at 88K params / 16.5 KB FP4A:
- PSD h=64 measured at ~95 KB int8 → ~24 KB FP4A (rate Δ = +7.5 KB ≈ +0.005 score points)
- Plus the empirical PoseNet regression of 0.011 → +0.332 score points

The MDL "two-part code" calculation: the rate cost of describing PSD is +0.005 + 0.332 = +0.337 score points. The information gained (SegNet improvement) is at best 0.0029 - 0.0014 = 0.0015 → -0.15 score points. **Net: +0.187 score points cost**. This is a *worse* description-length encoding of the (renderer, score) tuple than dilated. **REJECT.**

### Ballé — REJECT

Hyperprior perspective: PSD's PixelShuffle output produces a 12-channel intermediate that COULD support a tighter scale prior on the SegNet logits at the bottleneck. If we trained a ScalePriorMLP on PSD's bottleneck features, we might extract additional 5-15% rate savings on the renderer's quantized weights. **But this is Phase 2 Lane 20 (Ballé hyperprior), NOT Lane 7 PSD.** Adding hyperprior to a worse base architecture (PSD@1.49) is strictly worse than adding hyperprior to a better base architecture (Lane G v3 @ 1.05).

The hyperprior gain is multiplicative on the base. Apply it to the better base. **REJECT.**

---

## 2. Vote Tally

| Voice | Vote | Reasoning gist |
|---|---|---|
| Shannon | REJECT | No R(D) mechanism plausibly closes 1.49 → <1.05 gap. PSD floor ~1.46. |
| Dykstra | REJECT | Pose constraint violated 11.9× empirically; outside achievable set. |
| Yousfi | REJECT | PSD bottleneck destroys FastViT-PoseNet's required luma detail. |
| Fridrich | REJECT | PSD aligned with SegNet blind spots, NOT PoseNet's. Need luma-skip variant. |
| Contrarian | REJECT | All bold APPROVE arguments are pre-emptively underspecified. |
| Quantizr | REJECT | Leader (0.33) didn't choose PSD — Bayesian evidence against. |
| Hotz | REJECT | EV is +$125/0.01-score-point — terrible vs Phase 2 alternatives. |
| Selfcomp | REJECT | PSD lacks PoseNet-aware luma-skip; my 0.38 explicitly avoided it. |
| MacKay | REJECT | MDL net cost +0.187 score points; PSD is worse 2-part code. |
| Ballé | REJECT | Hyperprior amplifies base; apply to Lane G v3, not PSD. |

**Vote: 0 APPROVE, 10 REJECT, 0 ABSTAIN.**

**Verdict: REJECT (UNANIMOUS).**

Per CLAUDE.md "Council conduct" rule: unanimous votes should be scrutinized. I (the parent agent) explicitly evaluated whether ANY voice was reaching for a conservative argument ("don't change working code", "ship what we have"). NONE of the rejection arguments above are conservative. Every single one cites:
- Empirical historical evidence (1.49 at ep 809) [contest-CUDA equivalent of the time]
- Mathematical/geometric reasoning (Shannon R(D), Dykstra Pareto, MacKay MDL)
- Architectural mismatch (Yousfi/Fridrich/Selfcomp on PoseNet rate-resolution)
- Bayesian inference from competitor choice (Quantizr)
- Expected-value arithmetic (Hotz)
- Composability with Phase 2 (Ballé)

The unanimity is genuine, not performative.

---

## 3. Reactivation Criteria (what would change the verdict)

Lane 7 PSD becomes worth retrying IF AND ONLY IF:

1. **A PoseNet-aware luma-skip variant** is designed AND a separate council review approves it (Fridrich's narrow opening). This would be a **new** lane (Lane PSD-LumaSkip), not a re-dispatch of `PSD_STANDARD_ADAPTIVE`.
2. **The current floor moves below 0.50**, at which point the score-arithmetic constraint relaxes enough that PSD's 5× PoseNet regression might be tolerable IF combined with extreme rate savings (e.g., PSD + block-FP 1.017 bpw + arithmetic coded weights = ~12 KB renderer; extra 4 KB savings = -0.0027 score points, comparable to PSD's PoseNet penalty at the new floor).
3. **Empirical Phase 2 Lane 19 (SegNet logit-margin) demonstrates that SegNet improvements transfer architecture-agnostically.** If logit-margin gives Lane G v3 a 10% SegNet improvement, then PSD's seg-improvement specialty becomes redundant.

None of these conditions hold today.

---

## 4. Disposition

**KILL** for this dispatch cycle. **DEFER** for permanent disposition pending the three reactivation criteria above. Specifically:

- Do **NOT** create `scripts/remote_lane_psd.sh`
- Do **NOT** dispatch any PSD GPU run today
- Update `competition_state.py:131` (already lists `psd_architecture` as killed) — no change needed; the killed_techniques list is already correct
- Add memory entry documenting tonight's REJECT verdict
- Phase 1 status table in `feedback_production_hardened_standard_definition_20260430.md` should mark Lane 7 as **KILLED** (was: "1 (script + watchdog landed but never dispatched) — dispatch-or-kill-with-rationale decision needed"). The decision is now **kill with rationale** and the rationale is THIS document.

---

## 5. Cross-references

- `competition_state.py:131` — `psd_architecture` killed, "Auth eval 1.49 vs dilated 1.33. Worse."
- `memory/project_psd_auth_eval_verdict.md` — 2026-04-11 verdict, council unanimous "STAY WITH DILATED"
- `memory/project_psd_breakthrough.md` — 2026-04-10/11 1.38 result requires KL distill (also killed)
- `memory/project_psd_early_signal.md` — 2026-04-10 ep289 first signal
- `src/tac/profiles.py:168` — `PSD_STANDARD_ADAPTIVE` profile (was the candidate config)
- `src/tac/architectures.py:798` — PSDPostFilter wired in VARIANTS dict
- `feedback_production_hardened_standard_definition_20260430.md` — the maturity-level audit listing Lane 7 as Level 1
- `project_phase1_dispatch_state_corrections_20260429.md` — predicted band [1.10, 1.40] standalone

## 6. Process notes

- Council convened at 2026-04-30 ~03:00 CDT
- All 10 inner voices required, all 10 cast a vote
- Conservative-bias check: PASSED (every REJECT cites math/empirical, none cites "don't change working code")
- Unanimity: GENUINE, not performative (each voice cited independent evidence)
- 3-clean-pass adversarial protocol: NOT YET RUN — Council Round 1 of N. This document is the Round 1 deliverable. Round 2/3 codex review is queued separately for any dispatch infrastructure work that LANDS, but since the verdict is REJECT, no dispatch infrastructure will land.
