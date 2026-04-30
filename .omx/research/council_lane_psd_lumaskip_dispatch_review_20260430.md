# Council — Lane PSD-LumaSkip GPU Dispatch Review (Phase B Gate)

**Date:** 2026-04-30
**Convener:** PSD-LUMASKIP-DISPATCH-COUNCIL-AGENT (separate-council requirement per Council #271 reactivation criterion #1)
**Inner council (10):** Shannon (LEAD), Dykstra (CO-LEAD), Yousfi, Fridrich, Contrarian, Quantizr, Hotz, Selfcomp, MacKay (memorial), Ballé
**Question:** Should Lane PSD-LumaSkip (`PSD_LUMASKIP_LANE_G_V3` profile, scaffold landed at `src/tac/psd_lumaskip_renderer.py` 15/15 tests passing) be **GPU-dispatched NOW**, **DEFERRED**, or **KILLED**?

**Standing bar:** Lane G v3 = 1.05 [contest-CUDA]. Quantizr leader = 0.33 [contest-CUDA external]. Selfcomp #2 = 0.38 [contest-CUDA external].

---

## 0. Pre-deliberation: load-bearing context (every voice MUST read before casting)

### F1. Council #271 reactivation criteria status (from `lane_7_psd_kill_memo_20260430.md` §3 + `council_lane_7_psd_dispatch_review_20260430.md` §3)

The vanilla PSD KILL memo set THREE reactivation criteria. Status as of 2026-04-30 PM:

| # | Criterion | Status |
|---|---|---|
| 1 | PoseNet-aware luma-skip variant designed AND separate council approves it | **PARTIALLY SATISFIED** — design council APPROVED scaffold (10/10); dispatch council = THIS deliberation |
| 2 | Current floor moves below 0.50 | **NOT MET** — current floor 1.05 [contest-CUDA] |
| 3 | Phase 2 Lane 19 (SegNet logit-margin) demonstrates SegNet improvements transfer architecture-agnostically | **NOT MET** — Lane 19 [contest-CUDA] pending |

**Strict reading:** criteria #2 AND #3 are NOT met. The original KILL memo used the conjunctive "AND" — not "OR". This is a HARD GATE under strict interpretation. The design quality alone does NOT satisfy #2/#3.

### F2. Grand Council Paradigm Shift verdict (from `grand_council_paradigm_shift_to_shannon_floor_20260430.md`)

The Grand Council's three-paradigm-shift framework places:
- **Paradigm shift α (mask payload overhaul)** = top EV move (-0.20 to -0.25 score; Lane 12 NeRV)
- **Paradigm shift β (sensitivity-aware everything)** = #2 (Phase 3 #275 in flight)
- **Paradigm shift γ (joint score-aware codec stack)** = #3 (ADMM coordinator + hyperprior)
- **PSD-LumaSkip is explicitly listed as an "Architectural alternative (L0, mostly L0)"** in Section 2 (line 124-128) — NOT in the top-3 paradigm shifts.

**Grand Council Contrarian VETO** (line 195-197): "Anything that requires retraining the renderer end-to-end: GATED on Lane 12 NeRV result first (because if mask payload doesn't shrink, renderer retraining doesn't move score enough to justify cost)."

PSD-LumaSkip dispatch IS an end-to-end renderer retraining. Therefore, it is GATED by the Grand Council Contrarian VETO until Lane 12 NeRV result lands.

### F3. Phase A scaffold council — re-reading the bounded unanimity

The Phase A council voted 10/10 APPROVE-FOR-SCAFFOLD, but explicitly bounded:
- 6/10 would dispatch standalone post-scaffold (Shannon, Dykstra, Yousfi, Fridrich, Selfcomp, MacKay)
- 3/10 only-as-stack-composition (Quantizr, Hotz, Ballé)
- 1/10 explicitly rejects standalone (Quantizr: "REJECT-LEANING-APPROVE-FOR-SCAFFOLD ... I would not prioritize it as a standalone")

**Hotz Phase A direct quote** (line 234): "dispatch only as part of a stacked experiment (PSD-LumaSkip + logit-margin + Ballé hyperprior, which is Phase 2 territory, not tonight's queue)."

**Tonight's queue contains exactly the Phase 2 prerequisites** (per Grand Council Section "concrete week-1 next actions" line 30-37):
1. Land Ω-W-V3 design
2. Pivot Ω-W-V2 → Ω-W-V3 with sensitivity weights
3. Dispatch Lane 12 NeRV CUDA training
4. Land sensitivity-map module
5. Land Lane PFP16 ($0)
6. Run ADMM coordinator (after Lane 12)
7. Build corpus codec for Lane J-NWC

**PSD-LumaSkip is NOT on this priority list.** Hotz's "Phase 2 territory" framing places PSD-LumaSkip dispatch AFTER these prerequisites, NOT in their queue.

### F4. Predicted-band re-verification

Phase A predicted bands (council-widened by Contrarian to [0.95, 1.40]):
- Optimistic (25%): [0.95, 1.05] — Pareto-dominates Lane G v3 marginally
- Central (45%): [1.05, 1.20] — Marginal regression vs Lane G v3
- Pessimistic (30%): [1.20, 1.40] — Dual-path destabilization regression

**Expected score** = 0.25·1.00 + 0.45·1.125 + 0.30·1.30 = 0.250 + 0.506 + 0.390 = **1.146** [prediction]

**Expected delta vs Lane G v3 anchor 1.05**: +0.096 score points (REGRESSION expected).

The 25% upside band [0.95, 1.05] only marginally beats Lane G v3 (-0.05 best case). The 30% pessimistic band represents a $1.50-3.00 regression dispatch with no information gain (we already know dilated h=64 is not a good architecture if "destabilizes" means it can't train well — the only learning would be "the dual-path approach is brittle", which is a $3 lesson when the local CPU smoke test could have caught it for $0).

**EV = +0.25·(-0.05) + 0.45·(+0.075) + 0.30·(+0.25) = -0.013 + 0.034 + 0.075 = +0.096 score regression expected at $1.55 cost.**

**Score-arithmetic priority comparison**: Lane PFP16 is $0, 5 LOC, ZERO distortion, -0.005 score guaranteed. Lane 12 NeRV is $1-2, predicted -0.20 to -0.25. Lane PSD-LumaSkip is $1.55, predicted **+0.096 regression**. **Bottom of the queue by EV.**

---

## 1. Per-voice positions (math/empirical only, NO conservative bias)

### Shannon (LEAD) — DEFER

The R(D) argument for the scaffold (Phase A) was sound: skip path restores `I(luma_HF; pose_pred)` at +0.0001 rate cost. That argument **stands** for the architecture's existence as a research artifact.

But the dispatch question is different: **does PSD-LumaSkip dispatch tonight produce information that no other queued lane produces?**

The answer is NO. PSD-LumaSkip's information content is "does dual-path luma-skip recover FastViT signal in this architecture?" — which is a TIGHT, NARROW question. The SAME question applies to:
- Lane 19 logit-margin (architecture-agnostic SegNet improvement) — broader, higher EV
- Lane 12 NeRV (orthogonal mask codec) — broader, much higher EV
- Sensitivity-map module (foundational tooling, unblocks 3+ lanes) — broader, much higher EV

**Information-theoretic priority**: dispatch tonight should buy bits-of-decision-uncertainty per dollar at the highest rate. PSD-LumaSkip buys ~1 bit of uncertainty (does luma-skip work or not?) at $1.55. Lane 12 NeRV buys ~3-4 bits (is mask codec primary lever, what's the achievable mask byte floor, does score-aware NeRV converge, does the 30-min inflate budget hold?) at $1-2. NeRV strictly dominates.

**Once Lane 12 NeRV lands AND the floor moves**, the score-arithmetic context is different — at that point, PSD-LumaSkip may matter for the residual seg/pose contribution. **DEFER until Lane 12 NeRV result lands.**

**VOTE: DEFER.**

### Dykstra (CO-LEAD) — DEFER

Pareto-feasibility analysis: PSD-LumaSkip's projection onto the {seg, pose, rate} feasibility set lies STRICTLY above the projection of Lane 12 NeRV (which attacks rate, the dominant stream). The KKT waterline argument (per Grand Council Section 3.1 line 156-161): byte-allocation should equilibrate `dScore/dByte` across streams.

Currently the waterline is NOT equilibrated:
- Mask byte: dScore/dByte ≈ 0.00067, headroom 50× (60KB+ achievable via NeRV)
- Renderer byte: dScore/dByte ≈ 0.00067, headroom 20-50KB via Self-Compress NN
- Pose byte: SATURATED at ~5KB

**PSD-LumaSkip touches the renderer stream but not via byte reduction — it changes the architecture topology.** It's orthogonal to the dominant lever. Even in the optimistic case (1.024 floor), it shifts our point only marginally on the Pareto frontier — and the SAME marginal shift can be achieved by Lane PFP16 + sensitivity-map at $0.

**VOTE: DEFER until paradigm shift α (NeRV) lands and the Pareto frontier itself moves.**

### Yousfi — APPROVE-LEANING-DEFER

I (Yousfi) was the lead REJECT vote on vanilla PSD AND the lead APPROVE vote on PSD-LumaSkip scaffold. I have a stake in seeing this run.

**The empirical argument FOR running it now:**
- The luma-skip mechanism addresses my exact kill-memo objection (Reason 2). Empirical confirmation would close that loop.
- Phase A scaffold landed clean (15/15 tests pass per task brief). The architecture is ready.
- Cost is $1.55 — marginal vs the lab's overall $25 Vast.ai budget.

**The empirical argument AGAINST:**
- Council #271 criteria #2 (floor < 0.50) and #3 (Lane 19 [contest-CUDA]) are explicitly NOT met. The original kill memo was unambiguous about the AND.
- My OWN expected band for PSD-LumaSkip standalone is [1.05, 1.20] central — i.e. EXPECTED REGRESSION.
- The information I'd buy ("does luma-skip work standalone on the Lane G v3 anchor?") is much less valuable than the SAME information on a SENSITIVITY-WEIGHTED Lane G v3 anchor, which is what Phase 3 #275 will produce in days.

**My specific concern about dispatching standalone**: dual-path training destabilization is real (30% prob per Phase A). If it destabilizes, the failure mode is informative ONLY if we can attribute the failure to dual-path-vs-architecture-itself. With sensitivity-weighting in hand, we could decompose: was it dual-path interference, or was it sensitivity-blind FP4 quantization? Without sensitivity-weighting, a destabilization gives us a mushy answer.

**VOTE: DEFER until Lane 19 [contest-CUDA] AND sensitivity-map land. Then dispatch as a STACKED experiment.**

### Fridrich — DEFER

The asymmetric-cost-embedding mechanism I endorsed in Phase A is sound, but it is most valuable when paired with my OTHER prescription — per-channel sensitivity weighting (which I named Ω-W-V3). The Phase A scaffold does NOT include sensitivity weighting; it includes only the architectural skip path.

**Stack prediction:**
- PSD-LumaSkip standalone: [1.05, 1.20] central [prediction]
- PSD-LumaSkip + Lane 19 logit-margin: [1.00, 1.10] central [prediction]
- PSD-LumaSkip + Lane 19 + Ω-W-V3 (sensitivity-weighted): [0.92, 1.05] central [prediction]
- PSD-LumaSkip + Lane 19 + Ω-W-V3 + Lane 12 NeRV: [0.65, 0.90] central [prediction] — IF NeRV delivers paradigm-shift α

The standalone is dominated. The stacked composition is interesting but requires Lane 19 + sensitivity-map + NeRV all landed. Sequencing matters.

**VOTE: DEFER. Specifically, dispatch ONLY when Lane 12 NeRV [contest-CUDA] AND sensitivity-map are both available, then dispatch as a 3-lane stack.**

### Contrarian — KILL-LEANING-DEFER

My role is to challenge weak arguments, not bold ones. Let me pre-emptively reject the bold arguments for dispatch:

1. **"Phase A unanimously approved"** — the unanimity was for SCAFFOLD, not DISPATCH. The Phase A memo §7 is explicit: "DEFERRED for GPU dispatch pending Phase B scaffold passes 3-clean-pass adversarial review AND local smoke test confirms forward-pass shape-correctness AND separate council convened with empirical predicted-band evidence". Where is the local smoke test? Where is the 3-clean-pass adversarial review?

2. **"15/15 tests pass"** — that's the SCAFFOLD passing tests. NOT empirical predicted-band evidence. The Phase A memo specifically required local smoke test + 3-clean-pass review BEFORE the dispatch council convened. Both are MISSING.

3. **"Cost is only $1.55"** — true, but the OPPORTUNITY cost is the same dollar going to Lane PFP16 ($0, -0.005 guaranteed) + Lane 12 NeRV ($1-2, -0.20 to -0.25 predicted) + sensitivity-map ($1, foundational). $1.55 spent on PSD-LumaSkip is $1.55 NOT spent on the top-EV queue.

4. **"It's a research artifact, just run it"** — research-artifact framing is the conservative-bias trap in disguise. "Just run it because we built it" violates Council conduct rule (math/empirical only). The math says EV is +0.096 regression. The empirical context says criteria #2/#3 are not met.

**The KILL temptation**: write the formal kill memo, document reactivation criteria, move on.

**The DEFER reason I am NOT killing**: PSD-LumaSkip is a legitimate POST-paradigm-shift-α/β/γ architecture. Once mask payload shrinks to <100KB (NeRV) and sensitivity-weighting is operational (Phase 3 #275), the renderer stream becomes a larger archive share AND we have the tools to distinguish PSD-LumaSkip's architectural value from quantization noise. AT THAT POINT (Phase 2/3 territory, weeks 2-4), PSD-LumaSkip becomes worth dispatching — likely as a 3-lane stack with Lane 19 + Ω-W-V3.

**VOTE: DEFER.** With explicit reactivation criteria documented in the lane registry.

### Quantizr (adversarial) — KILL-LEANING-DEFER

In Phase A I voted "REJECT-LEANING-APPROVE-FOR-SCAFFOLD". My standalone-rejection stands. Standalone dispatch is the WRONG configuration.

The Bayesian update from Phase A:
- P(luma-skip works standalone | I would have found it during my "sweeping conv dims" exploration) = 0.4
- P(luma-skip works as a STACK PARTNER with logit-margin) = 0.5+
- The 0.5+ stack-partner probability is gated on logit-margin landing AND being beneficial.

**Empirical question for the dispatch council:** has Lane 19 logit-margin [contest-CUDA] landed? **NO.** Therefore the 0.5+ stack-partner probability is conditional on a future event. Pre-emptively dispatching PSD-LumaSkip standalone (where my probability is 0.4 of marginal improvement) burns $1.55 to test the WORSE configuration.

**My specific recommendation**: hold the lane at scaffold-landed (L1) until Lane 19 [contest-CUDA] result is in. If Lane 19 shows a strong SegNet improvement, then PSD-LumaSkip + Lane 19 stack becomes the natural next experiment. If Lane 19 shows a weak result, then PSD-LumaSkip's stack value diminishes too and it becomes a kill candidate.

**VOTE: DEFER pending Lane 19 [contest-CUDA].**

### Hotz — DEFER (per my Phase A prescription)

My Phase A position was unambiguous: "dispatch only as part of a stacked experiment (PSD-LumaSkip + logit-margin + Ballé hyperprior, which is Phase 2 territory, not tonight's queue)."

Tonight's queue (per Grand Council week-1 priorities) is paradigm-shift-α/β prerequisites. PSD-LumaSkip standalone is NOT in that queue. Dispatching it would be jumping the queue without justification.

**EV math (updated from my Phase A version):**
- Standalone EV: +0.30·0.10 + 0.30·0.05 - 0.30·0.10 - 0.10·0.20 = +0.045 - 0.030 - 0.020 = -0.005 score, $5 cost (or $1.55 trimmed) → negative or near-zero EV
- Stacked EV (after Lane 19 + sensitivity-map land): +0.50·0.05 - 0.30·0 - 0.20·0.10 = +0.005 expected, $5 marginal → break-even

The break-even is what I called "interesting case" in Phase A. Tonight, the break-even configuration is NOT available because the prerequisites aren't satisfied.

**The 30-minute version (which I always check for):** the local smoke test that Phase A required would catch dual-path destabilization for $0 in 30 minutes on local CPU/MPS. Has it been run? The task brief says scaffold tests pass (15/15) but doesn't mention a local smoke training run on synthetic data with EMA snapshot+restore + dual-path gradient stability monitoring (per Contrarian Phase A line 200). **If the 30-minute version hasn't been run, run THAT first, not the GPU dispatch.**

**VOTE: DEFER. Run local 30-min smoke test first. Then re-evaluate after Lane 19 + sensitivity-map land.**

### Selfcomp — DEFER-LEANING-APPROVE

I (szabolcs-cs) want to see this run because I find the dual-path topology genuinely interesting as an alternative to my single-path full-res + block-FP approach. As a fellow architect, I'd like the lab to explore it.

**BUT** — I also recognize that my collaborative spirit is not the same as score-arithmetic priority. The score-arithmetic priority for tonight is paradigm-shift-α (NeRV mask codec, my own grayscale-LUT lineage). PSD-LumaSkip is an architectural exploration that CAN wait.

**Specific composition I want to see eventually** (post-Phase 2):
- PSD-LumaSkip + my block-FP 1.017 bpw codec
- PSD-LumaSkip + Lane 19 logit-margin
- PSD-LumaSkip + Lane 12 NeRV (orthogonal mask codec)

But NOT tonight, and NOT standalone.

**VOTE: DEFER. Post-Phase-2 dispatch as a 3-lane stack.**

### MacKay (memorial) — DEFER

MDL two-part code analysis:
- Standalone PSD-LumaSkip dispatch: rate cost ~$1.55 GPU + opportunity cost ($1.55 not spent on top-EV queue). Information gain: ~1 bit of architectural-question resolution. **MDL compression ratio: poor.**
- Stacked PSD-LumaSkip + Lane 19 + Ω-W-V3 dispatch: rate cost ~$5 marginal. Information gain: ~3 bits (architectural + composition + sensitivity-weighting all jointly tested). **MDL compression ratio: good.**

The information-theoretic priority is to NOT collapse the question into "does standalone PSD-LumaSkip work?" when the more informative question is "does PSD-LumaSkip + Lane 19 + Ω-W-V3 stack converge to sub-1.00 floor?"

Variational/Bayesian addendum: the scaffold landing is a posterior update. The dispatch is a likelihood evaluation. Doing the likelihood evaluation BEFORE the posterior fully informs the prior (i.e. before Lane 19 result tightens the stack-composition prior) is wasted compute.

**VOTE: DEFER until Lane 19 [contest-CUDA] result lands. Then dispatch as a stacked experiment.**

### Ballé — DEFER

Hyperprior synergy analysis: my Phase A endorsement was specifically about COMPOSITION with Lane 20 (Ballé hyperprior). Standalone PSD-LumaSkip does NOT exercise the bimodal-qint-distribution synergy I predicted (+0.005 to -0.015 score points beyond the standalone result).

**My specific composition target:**
- PSD-LumaSkip + Lane 20 hyperprior: +5-15% rate savings on the renderer's bimodal qint stream
- PSD-LumaSkip + Lane 20 + Lane 19 + sensitivity-map: full-stack composition test

Lane 20 is in Phase 2 (currently has script `scripts/remote_lane_20_balle.sh` per the script directory listing). **Dispatch standalone PSD-LumaSkip tonight does NOT exercise the load-bearing composition I care about.**

**VOTE: DEFER until Lane 20 (Ballé hyperprior) is far enough along to exercise the composition.**

---

## 2. Vote Tally

| Voice | Vote | Reasoning gist |
|---|---|---|
| Shannon | DEFER | Lane 12 NeRV strictly dominates on bits-of-uncertainty/dollar. Defer until NeRV lands. |
| Dykstra | DEFER | Wrong stream (renderer-architecture, not the dominant mask-byte lever). Pareto-suboptimal tonight. |
| Yousfi | DEFER | I designed it but the standalone EV is regression. Need Lane 19 + sensitivity-map first. |
| Fridrich | DEFER | Standalone misses the per-channel-sensitivity composition that makes it sing. |
| Contrarian | DEFER (KILL-LEANING) | Phase A required local smoke test + 3-clean-pass — both MISSING. EV is +0.096 regression. |
| Quantizr | DEFER (KILL-LEANING) | Standalone is the WRONG configuration. Hold scaffold pending Lane 19. |
| Hotz | DEFER | Phase A explicit: "Phase 2 territory, not tonight's queue". 30-min CPU smoke test first. |
| Selfcomp | DEFER-LEANING-APPROVE | Architecturally interesting but score-arithmetic priority elsewhere. Post-Phase-2 stack. |
| MacKay | DEFER | MDL info-theoretic: standalone collapses question prematurely. Wait for stack composition. |
| Ballé | DEFER | Hyperprior composition requires Lane 20 maturity. Standalone misses my contribution. |

**Vote: 0 APPROVE_NOW, 10 DEFER, 0 KILL.**

**Verdict: DEFER (UNANIMOUS).**

### Conservative-bias check (per CLAUDE.md "Council conduct" rule)

I (the dispatch council convener) explicitly evaluated whether ANY voice was reaching for a conservative argument. The unanimity is 10 DEFER, but every DEFER cites:
- **Mathematical/info-theoretic reasoning** (Shannon bits-of-uncertainty/dollar; Dykstra Pareto KKT; MacKay MDL)
- **Empirical historical evidence** (Council #271 criteria #2/#3 NOT met; Phase A scaffold prerequisites NOT met)
- **Sequencing/composition logic** (Yousfi/Fridrich/Quantizr/Hotz/Selfcomp/Ballé all argue stack-composition is the right configuration, NOT standalone)
- **Expected-value arithmetic** (Hotz/Contrarian: +0.096 score regression expected at $1.55)

**No DEFER vote cites "don't change working code", "ship what we have", or "play it safe".** The DEFER is genuine sequencing/composition reasoning, not conservative bias.

**The Contrarian and Quantizr LEAN KILL** but stop short because PSD-LumaSkip becomes a legitimate dispatch candidate AFTER Lane 19 + sensitivity-map + Lane 12 NeRV land. KILL would foreclose that future composition opportunity. DEFER preserves it.

---

## 3. Reactivation criteria (specific, measurable)

PSD-LumaSkip dispatch becomes APPROVE-NOW eligible WHEN AND ONLY WHEN at least 2 of the 3 following are met:

### R1. Lane 19 (SegNet logit-margin) [contest-CUDA] result landed

- A `[contest-CUDA]` score from `auth_eval_renderer.py` on the Lane 19 best-checkpoint archive.
- If the result shows a 5%+ SegNet improvement vs Lane G v3 anchor, then PSD-LumaSkip's chroma-path 12.8% SegNet advantage becomes COMPOSABLE with Lane 19 (paradigm shift β operationalized).
- If the result shows weak/no SegNet improvement, then re-evaluate KILL.

### R2. Lane 12 NeRV mask codec [contest-CUDA] result landed

- A `[contest-CUDA]` score from `contest_auth_eval.py` on the NeRV-mask archive.
- If NeRV shrinks mask payload to <150KB (paradigm shift α partial) AND total floor moves to <0.85, then renderer-stream architectural exploration like PSD-LumaSkip becomes higher-EV (renderer becomes larger archive share).
- If NeRV fails to converge or doesn't shrink mask payload meaningfully, then renderer-stream lanes don't move score enough to justify the spend either way.

### R3. Sensitivity-map module operationalized AND Ω-W-V3 dispatched

- Phase 3 #275 (sensitivity-map) is currently in flight per Grand Council line 32-34.
- Once sensitivity-map lands, Ω-W-V3 (per-channel sensitivity-weighted block-FP) becomes the immediate next dispatch.
- If Ω-W-V3 [contest-CUDA] hits its predicted -0.034 rate save without the +0.052 PoseNet pay (i.e. lands in [1.025, 1.045] band), then the per-channel-sensitivity-weighting pattern is empirically validated and PSD-LumaSkip's chroma-path can be quantized with the SAME tooling (composable).

**Reactivation gate**: when ≥2 of R1/R2/R3 are satisfied, convene a NEW dispatch council with the empirical evidence. The council will vote APPROVE_NOW vs continued DEFER vs KILL.

### Local 30-min smoke test (Hotz prescription, $0)

Independent of GPU dispatch, the Phase A required local smoke test should be run on local CPU with synthetic 8-frame batch:
- Verify forward-pass shape-correctness across resolutions
- Verify dual-path gradient stability (per Contrarian Phase A: "both luma and chroma paths' gradient norms stay within 10× of each other")
- Verify EMA snapshot+restore round-trip
- Verify eval_roundtrip plumbing
- Cost: $0 + 30 min subagent time

This smoke test is a PREREQUISITE for any future dispatch council per Phase A §7. It should be added to the Phase B follow-on queue regardless of the current DEFER verdict.

---

## 4. Disposition

**DEFER for this dispatch cycle.** Specifically:

- Do **NOT** create `scripts/remote_lane_psd_lumaskip.sh` tonight
- Do **NOT** dispatch any PSD-LumaSkip GPU run tonight
- Lane registry: `lane_psd_lumaskip` remains at L1 (scaffold-landed); ADD note pointing to this deferral memo
- Add memory entry documenting tonight's DEFER verdict + reactivation criteria
- Phase A council scaffold-landing verdict STANDS (no rollback of `src/tac/psd_lumaskip_renderer.py` or `PSD_LUMASKIP_LANE_G_V3` profile)
- Local 30-min smoke test recommended as Phase B follow-on (Hotz prescription) — independent of GPU dispatch

**Cost saved tonight by deferral: ~$1.55-3.00.** This freed budget reallocates to the Grand Council week-1 priorities (Lane PFP16 $0, sensitivity-map $1-2, Lane 12 NeRV $1-2, Ω-W-V3 $1-2 once sensitivity-map lands).

---

## 5. Cross-references

- `.omx/research/council_lane_psd_lumaskip_design_20260430.md` — Phase A design council 10/10 APPROVE-FOR-SCAFFOLD (with bounded unanimity for dispatch)
- `.omx/research/council_lane_7_psd_dispatch_review_20260430.md` — Council #271 KILL of vanilla PSD + reactivation criteria #1/#2/#3
- `.omx/research/lane_7_psd_kill_memo_20260430.md` — formal kill memo establishing reactivation criteria
- `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md` — three-paradigm-shift framework + Contrarian VETO on end-to-end retraining lanes (Section 3.1 line 195-197) + week-1 priority queue (Section "concrete week-1 next actions")
- `src/tac/psd_lumaskip_renderer.py` — the scaffolded module (kept; not affected by dispatch DEFER)
- `src/tac/profiles.py:199` — `PSD_LUMASKIP_LANE_G_V3` profile (kept; not affected)
- `src/tac/tests/test_psd_lumaskip_renderer.py` — 15-test scaffold passing (kept)
- `.omx/state/lane_registry.json:910` — `lane_psd_lumaskip` L1 status (will be updated to point to this memo)
- `memory/project_session_state_checkpoint_20260430.md` — overall session state (5+ BG dev subagents in flight; tonight's queue context)
- CLAUDE.md "Council conduct" — non-conservative-bias rule (verified PASSED above)
- CLAUDE.md "Auth eval EVERYWHERE" — non-negotiable; enforced in any future PSD-LumaSkip dispatch script

---

## 6. Process notes

- Council convened at 2026-04-30 PM (after the Phase A scaffold council approved scaffold-landing per Council #271 reactivation criterion #1)
- All 10 inner voices required, all 10 cast a vote
- Conservative-bias check: PASSED (every DEFER cites math/info-theory/empirical/sequencing, none cites "play it safe" or "don't change working code")
- Unanimity: GENUINE on DEFER (10/10); 2/10 explicitly LEAN KILL (Contrarian, Quantizr) — bounded unanimity captured
- 3-clean-pass adversarial protocol: this is Round 1 of N for the dispatch question. Round 2/3 should re-evaluate when reactivation criteria R1/R2/R3 are met (i.e. NOT tonight; the next dispatch council convenes when ≥2 of R1/R2/R3 land).

---

## 7. Update to Council #271 reactivation criterion #1 status

| Criterion | Pre-this-memo status | Post-this-memo status |
|---|---|---|
| #1: PSD-LumaSkip variant designed AND separate council approves it | PARTIALLY SATISFIED (design ✓, dispatch council pending) | **PARTIALLY SATISFIED** (design ✓, dispatch council ran ✓ — verdict DEFER, not APPROVE) |
| #2: Current floor moves below 0.50 | NOT MET (1.05) | NOT MET (1.05; awaiting Lane 12 NeRV) |
| #3: Lane 19 [contest-CUDA] demonstrates SegNet transfer | NOT MET (pending) | NOT MET (pending) |

**Net status:** Lane 7 PSD reactivation criteria remain UNMET. PSD-LumaSkip is a legitimate Phase 2/3 candidate but is NOT eligible for tonight's dispatch.

---

**End of dispatch council memo.**
