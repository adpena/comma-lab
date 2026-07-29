---
schema: ddm_xi1_carried_xi_inter_race.v1
date_utc: 2026-07-29
arm: ddm_xi1
axis: "[macOS-CPU advisory, rate-only] lossless byte measurement on frozen checkpoint arrays"
pointer: "0.1910828242 [contest-CPU] UNMOVED"
score_claim: false
promotable: false
pointer_moved: false
research_only: true
council_predicted_mission_contribution: frontier_protecting
verdict_scope: INSTANCE
consumes: [ddm_r7_token_coder_race_20260729, ddm_pfs1_posefield_and_recompose_20260729,
  ddm_deferral_queue_ledger_20260729, "gc8 §3(a)"]
consumers: [ddm_deferral_queue_ledger_20260729, QA06_wr1_waterfill, QA08_context_mixing,
  v10_SPEC_rate_axis]
---

# ddm_xi1 — carried-ξ token INTER-prediction race (QA39; HOTZ-constrained)

## §0 HEADLINE (pointer honesty first)

**Pointer `0.1910828242 [contest-CPU]` UNMOVED. This arm produced NO rate reduction.**
Everything below is `[macOS-CPU advisory, rate-only]` lossless byte accounting on the frozen
endpoint token array; no scorer, evaluator, dispatch, or pointer mutation ran ($0).

The **falsifier FIRED on BOTH named variants.** The carried-ξ warp INTER predictor — the warp
mechanism that just won the pose axis (194 B beat 7,295 B by 37×, pfs1 D1) — does NOT open the
rate axis on this token chart. Both a warp-context expert inside SMEVR and a warp innovation
alphabet code the complete token frame LARGER than the plain SMEVR baseline.

| complete token frame (bytes) | Δ vs SMEVR 557,253 | Δ vs charter 557,238 |
|---|---:|---:|
| **SMEVR baseline** (reproduced, roundtrip-exact) | **557,253** | — |
| Variant A — warp-context expert (backward warp) | **569,515** | +12,262 | +12,277 |
| Variant A — warp-context expert (forward warp) | 569,775 | +12,522 | +12,537 |
| Variant B — warp innovation alphabet (backward warp) | **723,124** | +165,871 | +165,886 |
| Variant B — warp innovation alphabet (forward warp) | 757,604 | +200,351 | +200,366 |

Best carried-ξ frame = 569,515 B ≥ 557,253 B ⇒ **QA39 → FIRED with the negative receipt**,
INSTANCE-scoped to this 24×32 token chart + L16 mode-delta alphabet + SMEVR context wiring
(scope named in §5). NOT a paradigm kill (verdict_scope INSTANCE).

## §1 What was built (HOTZ constraint honored)

Per the gc8 HOTZ dissent — *"Wire the carried-ξ warp in as ONE MORE CONTEXT/expert inside
SMEVR … and measure it in an afternoon. If the complete frame doesn't beat 557,238 B, stop
saying MPEG-4."* — this arm did **not** build a new codec. It EXTENDS the landed SMEVR coder
(`experiments/ddm_r7_token_coder.py`) by importing its exact arithmetic coder, KT rescale,
residual-rank map, and channel views, and wiring the carried-ξ warped previous token field in
as one additional context. Because sb1 owns the n600 slot and the r7 coder is shared, the
extension lives in a NEW module `experiments/ddm_xi1_carried_xi_coder.py` (imports r7; edits it
zero). Race harness: `experiments/ddm_xi1_carried_xi_race.py`.

**The mechanism (rule-118 boundary).** The receiver warps the previously-decoded token CODE
plane (pair i−1) by the CARRIED pose to predict pair i, then either (A) adds the warp prediction
as one more coding context, or (B) codes the innovation `(code − warp_pred) mod 16`. COUNTED: the
token frame only. FREE (rule 118): the warp is generic deterministic numpy — a token-grid
rescale of the vendored ground-homography engine in
`tools/pfs1_recompose_warp_base_and_eval.py` (EON intrinsics 910/582/437 at full frame
1164×874, camera height 1.22 m; rescaled aspect-preserving to the 24×32 token grid). The carried
pose `t_p` (600×6 float16 target) + per-pair `s_t` index are the archive's OWN pose-member
payload (pfs1 grammar v3), already counted for frame_0; using them as token context adds ZERO
token bytes. Decode is causal (SMEVR iterates channel-outer / pair-inner; pair i−1 is fully
reconstructed before pair i), and every frame decodes to the exact input and re-encodes
byte-identically before admission (canonical re-encode + semantic-SHA closure).

Both warp directions were raced (backward = predict later-from-earlier via H⁻¹; forward via H);
the chosen direction is a single counted header bit. Backward wins marginally on both variants.

## §2 The measured negative (the falsifier)

- **SMEVR baseline reproduced same-object:** 557,253 B (header 56 + mode base 1,361 + delta
  555,836), roundtrip-exact against the pb1 composed archive token codes
  (`p2c_aimed_archive.zip`, the shipping endpoint `33776302…` lineage). This is the pfs1 §2
  `tokens.dr7t` value (557,253) to the byte; the r7 memo's 557,238 is the same object with a
  15-byte-smaller base coding — I race against BOTH and lose to both.
- **Variant A (warp-context expert)** adds a 2-way warp-occupancy context to the SMEVR occupancy
  model and the warp-predicted delta to the SMEVR value model. Result: **+12,262 B** (best,
  backward). The delta stream grew 555,836 → 568,097.
- **Variant B (warp innovation alphabet)** recodes the field as `(code − warp_pred) mod 16` and
  runs the SMEVR occupancy/value split on the innovation. Result: **+165,871 B** (best,
  backward). The warped single previous frame is a far worse per-cell predictor than the temporal
  mode base, so replacing the mode with the warp destroys concentration.

## §3 Why it lost — the mode base already owns the temporal structure (measured)

The mode base + SMEVR renewal-age + spatial context already absorb the temporal structure the
warp could provide, exactly as the r7 memo warned. The plug-in conditional entropies (total
bytes over the 1.84 M-symbol delta field; **orientation only — not achievable coder bounds**)
make the mechanism explicit:

| plug-in conditional entropy of the mode-delta | bytes |
|---|---:|
| H(delta) | 662,644 |
| H(delta \| base) | 592,441 |
| H(delta \| base, prev_coloc)  ← ≈ what SMEVR conditions on | 560,509 |
| **SMEVR's REALIZED delta** (uses age+spatial too) | **555,836** |
| H(delta \| base, prev_coloc, warp_bwd)  ← "one more expert" IDEAL ceiling | 554,219 |

The warp's IDEAL marginal over the plug-in-without-warp is −6,290 B, but SMEVR's realized delta
(555,836) is ALREADY below the plug-in-without-warp — so the warp's ideal ceiling (554,219) is
only **1,617 B below SMEVR's realized delta**, and the adaptive KT coder cannot reach that ideal:
adding warp context multiplies the value-model context count ~16× and the occupancy context 2×,
diluting the adaptive counts. The learning cost of the new contexts (+12,261 B measured) swamps
the ~1.6 KB ideal gain. This is the "model over backend" lesson inverted: the extra context's
ideal information is smaller than its adaptive learning tax.

Two structural reasons the warp is weak AT THIS CHART, both measured:
1. **Resolution.** At 24×32 (16× downsample of the 384×512 seg grid) a single-frame ego step is
   sub-token-cell over most of the frame; the warp ≈ identity except in the lower rows. The
   identity control already lost in r7 (CAE identity-INTER 631,309 B > SMEVR).
2. **Predictor competition.** The per-cell temporal MODE over 600 pairs is a very strong static
   predictor (mode-delta occupancy only 0.596). A single warped previous frame agrees with the
   current code LESS often (innovation occupancy 0.77–0.81, measured), so it cannot beat the mode
   as a standalone predictor, and as an add-on context its information is below its dilution cost.

## §4 The Wyner conditional-entropy floor at the new conditioning (charter measurement 3)

Re-measuring r7's S4 floor at the carried-ξ conditioning. All **plug-in order-0** (total bytes;
NOT a theorem, score, or achievable bound — the same non-authoritative caveat as r7's S4):

| conditioning | H (bytes) |
|---|---:|
| **H(tokens_t \| ξ-warp(tokens_{t−1})), backward** (the charter's Wyner metric) | **728,435** |
| H(tokens_t \| ξ-warp(tokens_{t−1})), forward | 761,965 |
| H(tokens_t) unconditional | 822,349 |
| — for reference: H(delta \| base, prev_coloc) | 560,509 |

**The carried-ξ warp of the previous CODE is a WORSE conditioner than the per-cell mode base**
(728,435 B vs 560,509 B). It is far above the 130–200 KB target band (`190,334 B @0.172`,
`157,294 B @0.15`) — by ~3.8×. The only sub-baseline conditioning is the warp as an *extra*
expert on top of (base, prev_coloc), whose IDEAL plug-in floor (554,219 B) sits a mere ~1.6 KB
under SMEVR's realized delta and is unreachable by the adaptive coder (§3). Conclusion: the
carried-ξ conditioning does not lower the vehicle entropy floor on this chart.

## §5 Verdict scope (INSTANCE) + what a finer chart would need

**verdict_scope = INSTANCE.** The negative is bound to: the 24×32 token grid (grid_downsample
16), the L16 mode-delta token alphabet, the SMEVR occupancy/value context wiring, and the carried
pose used as the warp (intra-pair `t_p` f16 + `s_t` grid, an approximation of the inter-pair
motion). It does NOT falsify carried-ξ INTER prediction as a family. What a finer chart would
need for the warp to pay (named, not run — none authorized this arm):
1. **A finer token latent** where a single-frame ego step is ≥ ~1 cell over a meaningful frame
   fraction (e.g. 48×64+), so the warp carries real displacement — but that ~4× raw token count
   is a large deficit the warp must first overcome, and the mode base sharpens too.
2. **An inter-pair ego vector** (frame 2i−1→2i) rather than the intra-pair `t_p` (frame
   2i→2i+1). That is a NEW carried stream (more bytes) whose credit COMPETES with, not adds to,
   the pose member (§6).
3. **A low-cardinality, genuinely-predictive warp gate** (a 2-way "renewal here" signal that
   actually predicts occupancy). Measured here: the warp barely moves the occupancy entropy, so
   this does not exist at this chart.
The dominated innovation alphabet (Variant B) is INSTANCE-dead for the same reason at any chart
where the mode base out-predicts a single warped frame.

## §6 The wr1-pool overlap (honest, binding)

The carried-ξ token predictor rides on the SAME `t_p` (600×6 f16) + `s_t` (per-pair index) that
the pfs1 pose member already ships (**6,864 B**, counted for frame_0). Using them as token
context adds ZERO token bytes — and that is exactly why any token-stream credit CANNOT be summed
with a separate pose-carrier saving: both draw from the ONE `t_p`/`s_t` pool (the wr1
non-additive-pool law: same-pool levers COMPETE, never sum). In this arm the token stream got NO
benefit from the pool (it lost by ≥12 KB), so there is no double-count to reconcile: the
carried-ξ token predictor's realized value at this chart is ≤ 0. Had it won, the win would have
been the token member shrinking at zero marginal pose cost — genuine bytes, but capped by the
pool it shares with pose.

## §7 Routing (falsifier consequences)

1. **QA39 → FIRED** in the deferral ledger with this negative receipt (same commit).
2. **The named rate levers that remain** (per the ledger) are unaffected and still the live path:
   QA06/#766 wr1 sensitivity-weighted reverse-waterfill (the only named ~557 KB→130–200 KB lever)
   and QA08 context-mixing (nncp-class logistic over {prev1, spatial} — note the carried-ξ warp is
   NOT a useful additional expert for that mix at this chart, per §3/§4).
3. **The r7 verbatim owed item is now CLOSED at INSTANCE scope:** "nonidentity carried-ξ MPEG-4
   INTER-CAE remains owed" — measured here, it does not beat SMEVR on this token chart. The
   family reopens only on a finer chart / inter-pair vector per §5.

## §8 Wire-in (#125) + labels + custody

- sensitivity-map N/A · Pareto: the §2 rows are new advisory (bytes, variant) points, all
  dominated by SMEVR · bit-allocator N/A · cathedral N/A · continual-learning: this memo + DAG
  FEED + ledger QA39 flip · probe-disambiguator: the pre-registered ≥557,238 B falsifier IS the
  disambiguator (it fired).
- [no-triality] [p0-ledger-ok] — measurement arm; no DSL lever or canonical-equation surface
  changed; the coder is an experiments-tier prototype pending MAIN landing review.
- Receipts (SSD, certify-or-block): `/Volumes/VertigoDataTier/pact/ddm_xi1_20260729/`
  — `ddm_xi1_race_receipt.json` (sha `195f38039e6695dc29d27319abf6137f90760367a8fdc70395eb6816e7f66730`)
  + vendored `ddm_xi1_carried_xi_coder.py` + `ddm_xi1_carried_xi_race.py`.
- Counted object: endpoint token array `[600,24,32,4]` L16, seg archive
  `p2c_aimed_archive.zip` (SHA in receipt). Every frame roundtrip-exact + canonical.
- Tools: `experiments/ddm_xi1_carried_xi_coder.py`, `experiments/ddm_xi1_carried_xi_race.py`
  (ruff-clean, 2 review passes).
- Verdict INSTANCE; cure named (§5); no family/paradigm kill.

## §9 Pointer-delta honesty (last, as first)

Pointer `0.1910828242 [contest-CPU]` UNMOVED. Bars: 0.19108 / official ~0.172 / 0.15 — none
moved. This arm is a MEANS (a measured negative on a rate lever), not the END. It removes a
plausible-looking rate path from the queue with a real receipt, so the next unit aims at the
levers that remain (QA06 wr1 waterfill; QA08 context-mixing) rather than re-characterizing this
wall.
