---
title: "Adversarial review of ALL session results (2026-06-23) — confidence classification, retraction-completeness audit, the strategy-critical verdict, and the ranked re-validation queue"
authority: "[advisory / meta-review] — NON-PROMOTABLE. score_claim=false; promotion_eligible=false; pointer UNMOVED 0.19110. $0; CPU/source-read only; NO MPS; NO paid dispatch. This is a CRITIQUE memo — it moves no pointer and produces no exact row. It exists to keep the strategy from resting on under-powered/contaminated measurements."
date: 2026-06-23
subagent: adversarial-review-all-20260623
role: Contrarian + Assumption-Adversary + Recursive adversarial review protocol
score_claim: false
promotion_eligible: false
pointer_moved: false
reviewed_memos:
  - .omx/research/reverse_engineer_pr95_prune_capacity_rd_20260623.md
  - .omx/research/concentrated_saliency_taper_screen_20260623.md
  - .omx/research/qaxis_bitdepth_response_surface_20260623.md
  - .omx/research/dseg_384_achievability_floor_verdict_20260623.md
  - .omx/research/dseg_reducibility_gt_margin_verdict_20260623.md
  - .omx/research/math_optimal_joint_solver_20260623.md
  - .omx/research/nonrgb_capstone_reopen_verdict_20260623.md
  - .omx/research/anchor_hardening_gpu_resident_checkpoint_preserve_20260623.md
  - .omx/research/pr95_vs_ours_convergence_gap_and_capacity_rd_deepmath_20260623.md
binding_disciplines:
  - feedback_terminal_conclusion_needs_existence_proof_crosscheck_20260623   # existence-proof crosscheck
  - feedback_deepmath_joint_fullspace_review_each_finding_20260623           # 5-lens joint review
  - CLAUDE.md "Forbidden premature KILL without research exhaustion"
  - CLAUDE.md "Apples-to-apples evidence discipline"
  - CLAUDE.md NO-FAKE "surrogate-optimized-but-not-exact-authority-verified"
---

# Adversarial review of ALL session results (2026-06-23)

**Mandate (operator, 2026-06-23):** challenge EVERY result, classify its confidence, audit the
retractions, and produce the re-validation queue — so the narrowed strategy does not rest on
under-powered or contaminated measurements.

**Headline (read this first).**
- **Authority caveat that colors the WHOLE session:** every single result reviewed here is
  `[contest-CPU advisory]` / `[macOS-CPU advisory]` — **ZERO contest-CUDA rows, ZERO byte-closed
  `upstream/evaluate.py` rows on the actual submission bytes.** Per CLAUDE.md NO-FAKE class 8
  (surrogate-not-exact-authority) and "Submission auth eval — BOTH CPU AND CUDA," **none of these
  verdicts is a score authority.** They are gradient-rows + priors. The frontier pointer is UNMOVED
  at 0.19110 and the session produced no exact row. This is honest in every memo, but it means the
  strategy is currently steered entirely by advisory measurements.
- **The discipline held well in the FINDING-LEVEL framing** (every memo applies the existence-proof
  cross-check, distinguishes MEASURED vs DERIVED, refuses premature KILL) — but the **STRATEGY-LEVEL
  synthesis ("RGB rung capped at 0.191; sub-0.15 needs architecture-spectral or non-RGB") rests on a
  chain whose decisive link — "a small from-scratch generator can't hold d_seg at the rate-headroom
  budget" — has NEVER been measured.** It is inferred from (a) pruning [invalid for from-scratch], (b)
  an under-converged 2-point capacity power law, and (c) an under-converged taper screen. The single
  decisive measurement that would settle it is named in §4.

---

## §1 — Per-result classification table

Classification ∈ {SOLID = direct measurement, robust, confirmed; SUSPECT = under-converged /
under-tested / single-screen / proxy-not-confirmed; OVER-CLAIMED = projection/extrapolation framed
more strongly than the data; CONTAMINATED = rests on a known-invalid input}. A memo can carry
multiple findings at different grades; I grade the load-bearing claim and flag sub-claims.

| # | result (memo) | the load-bearing claim | evidence (data path + authority) | class | the adversarial challenge | re-validation that settles it |
|---|---|---|---|---|---|---|
| 1 | **reverse_engineer_pr95_prune_capacity_rd** | "0.191 = borrowed-substrate ceiling; NO smaller-via-pruning generator holds d_seg; capacity CLIFF below bc36" | `reveng_pr95_prune_20260623/kd_bc20.json` (d_seg 0.0239) + `kd_bc28.json` (0.0170), **600-pair / 60-epoch** KD-finetune; prune-only = **8-pair** smoke; score-aware KD = **n16/20ep** smoke. `[contest-CPU advisory]` | **CONTAMINATED** (for the strategy claim) / SOLID (for the narrow claim it actually measured) | The narrow measured fact is SOLID: *prune-then-KD-finetune-60ep of bc36 does not recover d_seg*. But the memo's headline "optimal size = bc36, capacity cliff" is then used to support "small generators can't hold d_seg" — and **pruning a co-adapted net ≠ training that capacity from scratch** (the memo SAYS this in hook #5, then the TL;DR partly elides it). 60ep KD vs PR95's 29,650ep from-scratch = ~500× less training. The bc20/bc28 KD rungs are *under-trained pruned subspaces*, not capacity floors. | A CLEAN **from-scratch** bc20/24/28/32 capacity sweep to convergence (the §4 #1 measurement). The prune path CANNOT answer the from-scratch question by construction. |
| 1b | (same) reverse-engineering faithfulness | "inflate→render→eval reproduces the converged decoder, d_seg 6.02e-4 ≈ published 5.6e-4" | 8-pair row, real `archive.zip` (178,417 B), in-process exact SegNet | **SOLID** | 8-pair subset, +7.5% above published — but the frozen-teacher KD render uses the same bc36 weights at 600 pairs and reproduces. Faithfulness does not depend on the 8-pair number. | Already adequate; the `curve_600_pruneonly.json` rung-0 (NOT yet on disk) would be the 600-pair exact anchor. |
| 2 | **concentrated_saliency_taper_screen** | "high-res-weighted taper is a d_seg NO-GO (+18% d_seg at matched budget); the last RGB-rung d_seg lever is closed" | `taper_screen_GENERIC_n100_b3000/best/best_meta.json` d_seg 0.004756 @ge300 vs `CONCB` 0.005614 @ge300 = **+18.1%** (matches memo exactly). `[contest-CPU advisory]` | **SUSPECT** | The +18% isolate is real but **STOPPED AT ge=300 of a 3000 budget (10%), both arms still in stage1-CE→stage2-softplus, 6 eval rows total.** This is a *mid-descent* comparison, not a basin. The converged disk anchors (ge2398, ge25748) reach d_seg 0.0026/0.0021 — and the converged-disk taper comparison flips SIGN (−8%, the OPPOSITE of +18%), which the memo correctly flags as confounded (hinge+train-length). So we have +18% (under-converged isolate) vs −8% (confounded converged) — **neither is a clean converged isolate.** The NO-GO-for-a-multi-day-burn verdict is defensible; the "+18% / d_seg lever is dead" framing is over-read from a 10%-budget screen. n=100 is also a memorization proxy (no n=600 isolate). | A **clean, matched-flags (hinge held identical), n=600, to-basin (ge≥1000)** GENERIC-vs-CONCB A/B — exactly the memo's OWN reactivation criterion. Until then "+18%" is an under-converged data point, not a converged verdict. |
| 3 | **qaxis_bitdepth_response_surface** | "bit-depth axis is RED — no PTQ Q<8 reaches sub-0.19 or sub-0.15; int8 is S-min; rate-lever-via-bits is REFUTED for PTQ" | `qaxis_..._20260623T232215Z.json`: n48 full 12-cell + n600 partial (int8/7/6 only). int8 n600 = 0.19646 local. `[contest-CPU advisory]` | **SOLID (for PTQ) / OVER-CLAIMED (as "bit axis dead")** | The PTQ RED verdict is SOLID — the monotone collapse is unambiguous and bit-count-robust (n48↔n600 agree on int8/7/6). BUT: (a) the load-bearing int5/int4/int3 cells are **n48-only** (n600 "left running," never landed) — the cells that matter most for the sub-0.15 question have NO 600-pair confirm; (b) the memo's OWN caveat says **QAT-finetune at int6/int7 is the un-run live lever** and only int5 QAT was tested (−9.5% d_seg). So "the bit axis only moves S up" is true for **PTQ + int5-QAT**, but int4/int6/int7 score-aware QAT-finetune is NOT exhausted. The per-channel-scale "not byte-closeable" claim is a real codec-grammar constraint but assumes the per-tensor-int8 grammar is fixed (a new section is a separate campaign, acknowledged). | (i) Land the int5/int4 **n600** cells (cheap, already detached). (ii) The int6/int7 **score-aware QAT-finetune** column the memo names as the only open follow-up — this is the actual "is the bit axis dead" test. |
| 4 | **dseg_384_achievability_floor** | "CAPACITY-LIMITED — a perfect-384 decoder floors d_seg at 0.019 S (11× below ours); the 384/uint8 pipeline does NOT block sub-0.15 on d_seg" | `dseg_384_achievability_floor_n600_20260623.json`: **n_pairs_scored=600**, floor_camres=0.0 (self-consistency PASSED), floor_384 d_seg 1.875e-4. `[contest-CPU advisory]` | **SOLID** | The construction (GT→bilinear↓384→bicubic↑→uint8→SegNet) is faithful, the self-consistency gate (GT-vs-GT=0) passed at n600, and the floor is a real existence proof (a perfect 384 reconstruction). The N=2→48→600 sequence is stable. The only soft spot: the "perfect 384" is the GT itself downsampled — it proves the *pipeline* floor, NOT that any *trainable byte-closed* decoder reaches it (the memo says exactly this in "the binding question is whether a decoder can be TRAINED+byte-closed to approach it"). Honest. | None needed for the floor itself. The downstream question (can training realize 0.019 S at small bytes) = §4 #1. |
| 5 | **dseg_reducibility_gt_margin** | "IRREDUCIBLE — our decoder's residual d_seg is low-GT-margin label-noise; a horizon decoder recovers ≤ ΔS 0.012; bank rate+pose, do NOT chase d_seg" | `dseg_reducibility_gt_margin_n600_20260623.json`: **n_pairs_scored=600**, sanity PASSED (measured 0.002124 vs live 0.002109), flip-margin median 0.122 vs non-flip 5.89 (~48×). `[contest-CPU advisory]` | **SOLID (measurement) / OVER-CLAIMED (the word "IRREDUCIBLE")** | The cross-tab is SOLID and confirmed at n600. BUT the verdict word **"IRREDUCIBLE" directly contradicts the sister floor memo's "CAPACITY-LIMITED (11× headroom)"** — and the contradiction is only resolved by a careful read (IRREDUCIBLE = *OUR decoder's specific flip set* is label-noise; CAPACITY-LIMITED = *a better decoder* has 0.21→0.016 S of headroom). A strategist scanning verdict words sees "IRREDUCIBLE → pivot off d_seg" and "CAPACITY-LIMITED → d_seg has 11× headroom" as a flat contradiction. The honest framing is: **our CURRENT decoder's flips are near-floor, but the d_seg AXIS is capacity-reachable to 13× lower.** "IRREDUCIBLE" is the wrong headline word for that. | No new measurement — a FRAMING reconciliation: rename to "OUR-CURRENT-DECODER-NEAR-ITS-FLIP-FLOOR (the d_seg AXIS has capacity headroom, sister floor memo)." The two memos must lead with the SAME composite verdict. |
| 6 | **math_optimal_joint_solver** | "two-layer floor: surface-model lower bound S≈0.179 (bc36+int4); physical floor S≈0.059; gap = capacity-realization; recommend train bc20 from scratch" | `reports/math_optimal_joint_solve.json`: optimum d_seg_evidence = **"power-law A=7.84e4 gamma=1.520 on bc20+frontier endpoints"**; physical_floor_S 0.0591. `[contest-CPU advisory]` | **OVER-CLAIMED (precision) / SOLID (the qualitative structure + the existence-proof self-catch)** | (a) **False precision:** the headline "0.179" is computed with **gamma=1.52** — which the SISTER deepmath memo (PART C.1) explicitly labels the CONTAMINATED cross-recipe fit ("conflates recipe+convergence+borrowed-substrate — DERIVED, not clean"), while the CLEAN matched-recipe fit is α=0.91. The solver uses the contaminated α for its headline. Both α give S*≈0.18, so the *conclusion* is robust, but the specific number 0.179 inherits a contaminated input. (b) **The physical floor S=0.059 is a perfect-384-d_seg construction gated on an unestablished d_seg-holding generator** — the memo's existence-proof cross-check CORRECTLY downgrades it from "floor" to "capacity-realization-limited model artifact" (the discipline ran on its own headline — credit). (c) The α is a **2-point fit** either way (bc20+bc36, or bc20+bc24). The "32.5 / 0.111" chained construction the prompt references does NOT appear in this memo (it appears to be a paraphrase of the C.3 deepmath table 2×-bpp→0.137 / 3×→0.116 chain, which is itself a hypothesis chain: −59% bpp NEEDS a d_seg-holding generator). | Pin α with a **multi-point from-scratch capacity sweep** (§4 #1 gives ≥3 clean points → a real exponent, not a 2-point lower bound). The physical floor stays a projection until a trained byte-closed decoder approaches 0.019-S d_seg at small bytes. |
| 7 | **nonrgb_capstone_reopen** | "GATED-GO — non-RGB witness viable as hybrid capstone; device LEGAL, rate −59% byte-closed, survival-wall avoided; gate = generator d_seg" | L13 byte-close 72,217 B (`score_native_candidate_20260610`, prior work, 8-pair parity); README L114 device quote; survival argument DERIVED. `[contest-CPU advisory]` | **OVER-CLAIMED (the GO) / SOLID (the rate byte-close + device reading)** | The −59% rate byte-close is real and prior-measured (SOLID). The device-legality reading is a defensible source read (SOLID-ish — but the L13 byte-close parity is **8-pair**, not 600, and "device legal" rests on the submitter self-routing to the T4, a real operational risk the memo flags). The **GATED-GO is conditional on TWO unestablished things stacked:** (i) a generator trained to d_seg < 3.2e-4 at ~65KB [UNMEASURED — same open question as everything else], AND (ii) pose carried as Wyner-Ziv FiLM side-info keeping d_pose at frontier 3e-5 [the pose-carrier-as-RGB-frame measured d_pose 0.006 → √(0.06)=0.245 ALONE blows the budget; side-info is DERIVED/projected, not byte-closed]. The sub-0.15 projection (0.110) multiplies two unestablished factors. "GATED-GO" is appropriate ONLY because the gate is honestly named as unmet; the risk is a reader treating "VIABLE" as "established." | (i) The generator-d_seg measurement (§4 #1, shared with everything). (ii) A **byte-closed** Wyner-Ziv FiLM pose-side-info parity test at 600 pairs (the 0.110 projection's pose term is currently a projection). Until both, this is a DESIGN, not a candidate. |
| 8 | **anchor_hardening** | "GPU-residency lever is a NO-OP (already resident); the throughput ceiling is the frozen scorer fwd/bwd (97%); checkpoint/snapshot/manifest added" | `probe_epoch_cost_breakdown.py` (CPU n=2 smoke: scorer fwd+bwd 96.8%); batch-invariance (bs64 13.43 s/ep = bs8); 553 torch_vehicle tests pass; `--defer-batch-sync` bit-identical | **SOLID** | The residency-is-already-done finding is a clean honest-negative backed by source line cites + a batch-invariance measurement + a component breakdown. The component-breakdown ratio is a CPU n=2 smoke (the memo says "confirms the SHAPE," not the MPS magnitude) — minor: the 97% is shape-correct, the exact MPS split is unmeasured, but the QUALITATIVE conclusion (transfer is not the bottleneck; scorer fwd/bwd is) is sound and the bit-identical defer-sync lever is test-proven. No confound found. This is the cleanest result of the session. | None. (Optional: an MPS-native component breakdown to pin the 97% magnitude, but the conclusion doesn't need it.) |
| 9 | **pr95_vs_ours_convergence_gap** (deepmath) | "ours plateaued from RECIPE bug (muon_lr 2e-4 = 150× too small, froze d_seg) + under-convergence (stage 1 of 8) + capacity; frontier decoder is at order-0 entropy floor (recode dead); fire the never-fired run" | `frontier_decoder_weight_entropy_20260623.json` (5.629 vs 5.586 b/param, ratio 1.0077); BUG-A A/B (0.507 frozen vs 0.066, 15 epochs matched); bc20_p48 0.00376 vs bc24_p48 0.00285. `[contest-CPU advisory]` | **SOLID (entropy floor + BUG-A) / SUSPECT (the muon-flat-on-d_seg robustness)** | The entropy-floor measurement (recode is dead, −0.00082 S) is SOLID and decisive — and it usefully KILLS the retracted "2×→0.132 recode" path (see §2). The BUG-A A/B (frozen-at-init under muon_lr 2e-4 vs 7.6× descent under 0.03) is a clean controlled isolate — SOLID. **The SUSPECT part:** "muon flat on d_seg" is from a **15-epoch, n=8, single-config** A/B. The mechanism (Muon Newton-Schulz makes step magnitude ∝ muon_lr not grad-norm, so 150× too small does ~0 work) is sound calculus, but whether it generalizes across LRs/stages is a 1-config finding. The capacity fit (α=0.91) is a clean 2-point but under-converged (120 CE epochs). | The BUG-A fix needs the **converged re-run at scale** to confirm the plateau actually lifts (the never-fired run, §4 #1 — which is ALSO the capacity sweep). The entropy floor needs no further work. |

**Tally:** SOLID **3** (anchor_hardening #8; dseg_384_floor #4; reverse-engineering-faithfulness #1b — and the entropy-floor + BUG-A sub-findings of #9). SUSPECT **2** (taper screen #2; pr95-convergence muon-flat robustness #9). OVER-CLAIMED **3** (qaxis "bit axis dead" #3; dseg_reducibility "IRREDUCIBLE" word #5; math-solver precision #6; nonrgb "GATED-GO" #7 — #6 and #7 are over-claimed-precision/over-claimed-GO respectively). CONTAMINATED **1** (prune-capacity #1, *as a from-scratch-capacity claim*).

(Counts overlap because several memos carry a SOLID narrow finding AND an OVER-CLAIMED/CONTAMINATED
strategy extrapolation. The honest per-MEMO headline grade: #1 CONTAMINATED-for-strategy, #2 SUSPECT,
#3 SOLID-for-PTQ-but-incomplete, #4 SOLID, #5 SOLID-measurement/OVER-CLAIMED-word, #6 OVER-CLAIMED-precision,
#7 OVER-CLAIMED-GO, #8 SOLID, #9 SOLID-core/SUSPECT-robustness.)

---

## §2 — Retraction-completeness audit (the previously-retracted results)

The prompt lists 5 retracted results. Audit: are the retractions complete, and does anything
downstream still depend on the wrong version?

| retracted result | what it claimed (wrong) | the refutation | retraction complete? | downstream dependency check |
|---|---|---|---|---|
| **E2 horizon-reducibility** | "horizon d_seg is reducible → launch a horizon decoder for a big win" | `horizon_band_dseg_lever_20260623.md` (sidecar oracle 0.70× of break-even, NO-GO) + `dseg_reducibility` (93.9% flips at GT-margin<0.5, label-noise) | **COMPLETE** — superseded by #5 (this session's IRREDUCIBLE) + the horizon-band NO-GO. The retraction is the correct direction. | ✅ No live memo recommends a horizon-decoder-for-big-d_seg. #5's "if any d_seg chased, near-zero-byte shared lever only, caps ΔS 0.024" is the surviving (correct) residue. CLEAN. |
| **recode "2× → 0.132"** | "a better entropy coder on the frontier weights → 2× bytes/param → sub-0.15" | `frontier_decoder_weight_entropy_20260623.json` (frontier already at order-0 floor, ratio 1.0077; recode buys −0.00082 S only) | **COMPLETE** — #9 PART C.4 explicitly: "refutes the 2× rate recode path AS A RECODE." | ✅ The math-solver (#6) and #9 both now route the rate axis to **retrain/QAT** (lower bits at held d_seg), NOT recode. The 2×-bpp→0.137 number SURVIVES but ONLY as an RD *target* gated on a d_seg-holding generator — correctly relabeled "unreachable by a coder." No memo still treats recode as live. CLEAN. |
| **IRREDUCIBLE (the earlier grand-unification §8 version)** | "d_seg IRREDUCIBLE, sub-0.15 NOT reachable" (from ONE n=48 probe) | the existence-proof crosscheck: PR95 reaches d_seg 5.6e-4 (3.75× below the "floor"); the 0.191 frontier IS a PR95-class decoder at that basin | **COMPLETE at the discipline level** (the crosscheck memo exists and is cited everywhere) — **BUT** the WORD "IRREDUCIBLE" was REUSED this session in #5 with a NARROWER (correct) meaning. ⚠️ **Naming collision risk:** the retracted sweeping "IRREDUCIBLE" and the new narrow "IRREDUCIBLE (our flip set)" share a word. | ⚠️ **PARTIAL** — see §1 #5. The new #5 is correct on its own terms but the verdict WORD invites re-conflation with the retracted version. The dseg_384 floor memo (#4) is the antidote (CAPACITY-LIMITED, 11× headroom) and the two are reconciled IN the floor memo — but a careless reader could resurrect the retracted sweeping claim. RECOMMEND: the #5 memo lead with "OUR-DECODER-NEAR-FLOOR; AXIS-HAS-HEADROOM." |
| **INT4-rate-lever (PTQ)** | "lower-bit PTQ weights = the rate lever to sub-0.15" | `qaxis` #3 (PTQ int4 = S 0.91–2.16; every Q<8 raises S) | **COMPLETE for PTQ** — #3 is the formal refutation. | ✅ But note the LIVE successor: "int4/int6/int7 score-aware QAT-finetune" is NOT retracted (correctly — only PTQ is dead). The math-solver still lists int4-QAT as a gated lever. The retraction is scoped correctly (PTQ dead, QAT open). CLEAN, with the caveat that QAT-int4 is still an open (unmeasured) lever, not a closed one. |
| **residency-gain** | "GPU-residency flag will speed the run" | `anchor_hardening` #8 (residency already in place; flag = NO-OP; ceiling is scorer fwd/bwd) | **COMPLETE** — and exemplary: the memo REFUSED to add the no-op flag (NO-FAKE class 1). | ✅ The hardened restart command relies on `--defer-batch-sync` (real, bit-identical) not a fake residency flag. CLEAN. |

**Retraction audit verdict:** **4 of 5 retractions are COMPLETE and clean with no live downstream
dependency on the wrong version.** The 1 PARTIAL is the **"IRREDUCIBLE" word-reuse** (E2/grand-unification
→ #5): the new use is technically correct but the shared word is a re-conflation hazard. The fix is a
rename, not a re-measurement. **Positive note:** the recode-2× and INT4-PTQ retractions are *backed by new
this-session measurements* (entropy floor; qaxis surface) — those are the strongest kind of retraction
(measurement-superseded, not just re-interpreted).

---

## §3 — THE STRATEGY-CRITICAL VERDICT: is the "0.191 RGB ceiling" narrative ESTABLISHED or UNDER-POWERED?

**The narrowed strategy:** *"the RGB rung is capped at ~0.191; sub-0.15 requires architecture-spectral
(concentrated-saliency own vehicle) or non-RGB (task-space witness)."*

**Decompose the chain that supports it:**

1. *Pure entropy recode → sub-0.15* — **REFUTED, SOLID** (entropy floor, #9 C.4). Correctly closed.
2. *Capacity scaling (bigger decoder) → sub-0.15* — **REFUTED, SOLID-ish** (capacity-RD optimum sits AT
   the frontier ~0.18-0.19 across all α; #6/#9 C.2). The author's own 2× teacher dead-end corroborates.
   This is the most defensible link.
3. *PTQ bit-shrink → sub-0.15* — **REFUTED, SOLID for PTQ** (#3), but int4/6/7 score-aware QAT is OPEN.
4. *Taper reallocation → d_seg* — **NO-GO, SUSPECT** (#2, under-converged 10%-budget screen).
5. **THE DECISIVE LINK — *a SMALL from-scratch generator CAN'T hold d_seg at the rate-headroom byte
   budget*** — this is what makes "RGB rung capped" terminal. **It is NOT ESTABLISHED.** It rests on:
   - **pruning** (#1) — *invalid* for the from-scratch question by construction (the memo's own hook #5
     says the solver must NOT treat prune-then-finetune as a valid way to instantiate a smaller-C config);
   - an **under-converged 2-point α power law** (#6/#9 — α∈[0.9,1.5], 120-CE-epoch fit, "lower bound");
   - an **under-converged taper screen** (#2 — ge=300 of 3000);
   - and the **never-fired** bc20 8-stage from-scratch run (acknowledged in EVERY memo as the open variable).

**VERDICT: the "0.191 RGB ceiling" narrative is UNDER-POWERED, not established.** Links 1–3 are solid
(recode/capacity-scaling/PTQ are genuinely closed). But the *terminal* claim that pivots the whole
strategy off the RGB rung — that a small own-trained decoder cannot reach sub-0.15 d_seg at the rate
budget — has **never been measured**. Worse, the existence-proof discipline cuts the OTHER way here:
**PR95 measures d_seg 5.6e-4, and the break-even for sub-0.15 at the bc20 byte budget is d_seg < 7.35e-4
(int8) — i.e. a converged PR95-class decoder at the SMALL byte budget is, on the solver's OWN
arithmetic (#6 C.3), a sub-0.15 candidate.** The strategy memos already CONTAIN this counter-finding
(it is literally the math-solver's "single recommended next config" and #9's E.1 "highest EV"), but the
*narrative* ("RGB capped → pivot to non-RGB/spectral") is being told as if the RGB-rung verdict were in,
when the decisive measurement is still pending and the arithmetic points the opposite way.

This is a textbook MP2 setup (apparatus/under-convergence artifact about to be mistaken for a physics
wall) — the exact failure the operator's 2026-06-23 rebuke and the existence-proof discipline were
created to prevent. The memos individually applied the discipline; the SESSION-LEVEL synthesis has not.

**Do NOT pivot off the RGB rung until the from-scratch capacity measurement is in.** The non-RGB capstone
and concentrated-saliency vehicle are legitimate PARALLEL R&D bets (and the non-RGB rate −59% is a real
asset), but they should not be justified by a "RGB is capped" premise that is currently under-powered.

---

## §4 — The ranked re-validation queue (highest-stakes first)

Each entry: the SUSPECT/OVER-CLAIMED/CONTAMINATED verdict it settles, the clean test, the falsifiable
threshold, and cost. **All must end in a byte-closed exact row to count as ground truth (the means/ends
firewall) — advisory CPU is the gate, contest-CUDA/contest-CPU on byte-closed bytes is the verdict.**

### #1 — THE DECISIVE MISSING MEASUREMENT: a CLEAN from-scratch d_seg(capacity) sweep to convergence
**Settles:** #1 (CONTAMINATED prune claim), #6 (OVER-CLAIMED α precision / physical-floor gating), #9
(SUSPECT muon-flat robustness, BUG-A-at-scale), AND the §3 strategy-critical verdict. This single sweep
is load-bearing for FIVE of the eight results and the entire strategy.

**The test:** train bc20 / bc24 / bc28 / bc32 **FROM SCRATCH** (NOT pruned) via the BUG-A-corrected full
8-stage PR95 curriculum (`muon_lr 0.03`, `--muon-lr-floor-fix`), n600, to convergence, byte-close each,
exact-eval. This is the "never-fired run" that every memo names as the open variable — it has been *armed*
(the anchor_hardening #8 hardened restart command is ready, the prune-path tooling is built) but **never
fired** (per MEMORY, armed-not-fired since 2026-06-11).

**Falsifiable thresholds (the existence-proof cross-check, both directions):**
- GREEN (strategy REFUTED, RGB rung lives): converged bc20 or bc24 byte-closed S < 0.15, OR < 0.19110.
  Break-even: bc20 needs d_seg < 7.35e-4 (int8); PR95 measures 5.6e-4 → this is a live shot.
- RED (strategy EARNED on solid ground): the FULLY-converged 8-stage from-scratch bc20/24 caps d_seg ≥
  the int8 break-even AND a clean ≥3-point α fit confirms the capacity exponent doesn't bend it down.
  ONLY THEN is "RGB rung capped" established (and the prune/taper/2-point-α artifacts are vindicated).

**Cost:** the run is ~4.6 days local MPS (anchor_hardening estimate) OR a paid GPU campaign ($12–49 for
the decisive exact row, $0.30 step-time smoke first per Carmack MVP-first). This is the run to FIRE, not
to characterize further. **It is the #1 priority by a wide margin** — it converts the entire
under-powered strategy chain into a measured verdict.

### #2 — int6/int7 score-aware QAT-finetune column (the only open bit-axis lever)
**Settles:** #3 (OVER-CLAIMED "bit axis dead" — PTQ is dead, QAT-int6/7 is unmeasured).
**The test:** the QAT-finetune column the qaxis memo names as "the only open follow-up" — int6/int7
score-aware QAT warm-started from a converged #1 basin (NOT from scratch). Falsifiable: int6/int7 QAT
holds d_seg within finetune-recoverable range AND the bytes saved net a sub-0.19/sub-0.15 S.
**Cost:** a training run (separate campaign); gated on #1 producing a converged basin to warm-start from.
**Stakes:** MEDIUM — if #1 lands sub-0.15 at int8, this is moot; if #1 lands [0.15,0.191], this is the
rate lever that could close the gap.

### #3 — A clean converged taper A/B (matched flags, n600, to basin)
**Settles:** #2 (SUSPECT taper screen — +18% at ge=300 vs −8% confounded converged).
**The test:** the memo's OWN reactivation criterion — GENERIC vs CONCB, margin-hinge held IDENTICAL,
n600, to ge≥1000 basin. Falsifiable: concentrated d_seg < generic d_seg at the basin → re-open;
else the +18% under-converged result stands.
**Cost:** can be folded as arms into the #1 sweep (the taper is byte-neutral; run it as a bc20 arm).
**Stakes:** LOW-MEDIUM — byte-neutral lever; settles a sign-flip ambiguity cheaply.

### #4 — Byte-closed Wyner-Ziv FiLM pose-side-info parity (the non-RGB capstone's pose gate)
**Settles:** #7 (OVER-CLAIMED GATED-GO — the 0.110 projection's pose term is a projection).
**The test:** byte-close the AmortizedLumaCarrier + FiLM pose-side-info at 600 pairs, measure d_pose on
the shipped bytes (the projection assumes d_pose stays at frontier 3e-5; the carrier-as-frame measured
0.006). Falsifiable: side-info holds d_pose < 1e-4 at < 2KB → the hybrid pose term is real; else the
0.110 projection is too optimistic.
**Cost:** $0 byte-close + CPU eval (no training; the carrier is prior work).
**Stakes:** MEDIUM — gates whether the non-RGB capstone is a real candidate or a design; but the WHOLE
capstone also gates on the #1 generator-d_seg, so this is secondary to #1.

### #5 — (FRAMING, not measurement) Reconcile the IRREDUCIBLE↔CAPACITY-LIMITED verdict words
**Settles:** #5 (OVER-CLAIMED "IRREDUCIBLE" word) + the §2 PARTIAL retraction hazard.
**The test:** none — a memo-framing pass so #4-floor and #5-reducibility lead with the SAME composite
verdict ("OUR-CURRENT-DECODER-NEAR-ITS-FLIP-FLOOR; the d_seg AXIS has capacity headroom to 13× lower").
**Cost:** $0. **Stakes:** LOW for the score, HIGH for not re-resurrecting the retracted sweeping
"IRREDUCIBLE → sub-0.15 unreachable" claim that the operator already rebuked once.

---

## §5 — 5-lens joint review of THIS review (per the deepmath standing mode)

- **Math/algebra:** the strategy chain's terminal link (small-generator d_seg) is the ONE unmeasured
  term in S = 100·d_seg + √(10·d_pose) + 25·bytes/N; every other term is closed. The algebra (break-even
  d_seg < 7.35e-4 vs PR95's measured 5.6e-4) says the link is plausibly FALSE — RGB may not be capped.
- **Geometry:** prune = a coordinate-subspace projection of a co-adapted net (impoverished); from-scratch
  = optimizing in the full space. They are geometrically different objects — the prune result cannot
  bound the from-scratch d_seg. The contamination is structural, not statistical.
- **Calculus/convergence:** the taper screen (ge300/3000) and the α fit (120 CE epochs) are both read
  off the *transient* of the convergence ODE, not its asymptote. An under-converged derivative is not a
  floor. The BUG-A muon mechanism (step ∝ muon_lr) is sound but measured at one point on the LR axis.
- **Physics/free-energy:** the entropy-floor measurement (recode dead) and the capacity-RD optimum
  (S*≈0.18 at the current bpp "temperature") ARE solid free-energy-minimum statements — those links of
  the chain are real. The under-powered link is purely the convergence/from-scratch one.
- **JOINT:** the results are individually disciplined but the SESSION synthesis isolated the wrong knob —
  it concluded "RGB capped" from the (prune × under-converged-α × under-converged-taper) cluster while
  the joint view (break-even arithmetic + PR95 existence proof + never-fired run) says the decisive
  measurement is still pending and points the other way. **Fire #1; don't narrate the wall yet.**

## NO-FAKE ledger
- **What I MEASURED (re-verified from the JSONs):** dseg_384 n600 has n_pairs_scored=600 + floor_camres
  self-consistency=0.0 (#4 SOLID); dseg_reducibility n600 n_pairs_scored=600 + sanity passed (#5
  measurement SOLID); taper GENERIC/CONCB best_meta d_seg 0.004756/0.005614 @ge300, both stopped ge≈300
  of 3000 in stage1/2 (#2 SUSPECT confirmed); qaxis n600 = int8/7/6 ONLY (int5/4/3 n48-only) (#3
  incompleteness confirmed); prune KD bc20/bc28 = 600-pair/60-epoch (#1 under-trained-vs-29650 confirmed),
  curve_600_pruneonly.json NOT on disk; math-solver optimum uses gamma=1.520 = the contaminated
  cross-recipe α (#6 precision flag confirmed).
- **What I did NOT do:** run any new training/eval; move any pointer; produce any exact row. This is a
  read-only critique. The frontier is UNMOVED at 0.19110.
- **Confidence on my own verdicts:** the data-completeness flags (n600-vs-n48, ge300-vs-3000, prune-vs-
  from-scratch, gamma-1.52-vs-0.91) are MEASURED from the artifacts (high confidence). The "strategy is
  under-powered" verdict is an interpretation — its refutation condition is exactly re-validation #1
  landing GREEN or RED on solid ground.

## 6-hook wire-in
1. **Sensitivity-map** — N/A (meta-review, no per-axis byte sensitivity).
2. **Pareto constraint** — ACTIVE: this review marks the "RGB-rung-capped" Pareto edge as PROVISIONAL
   (gated on re-validation #1), preventing the planner from treating it as a hard constraint.
3. **Bit-allocator** — N/A.
4. **Cathedral autopilot dispatch** — ACTIVE (op-routable): re-validation #1 is the named decisive
   exact-row-feeding dispatch (the never-fired from-scratch capacity sweep).
5. **Continual-learning posterior** — ACTIVE: this memo records the confidence grades + the
   re-validation queue for the next campaign to inherit (the strategy is under-powered, not established).
6. **Probe-disambiguator** — ACTIVE: re-validation #1 IS the disambiguator between "RGB rung capped at
   0.191 (capacity wall)" and "small from-scratch decoder reaches sub-0.15 (under-convergence artifact)".
