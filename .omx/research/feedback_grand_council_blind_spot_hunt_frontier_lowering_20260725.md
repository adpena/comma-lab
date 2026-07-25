---
council_tier: T3
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Schmidhuber, Hotz, Carmack, MacKay, Balle, Selfcomp, Quantizr, PR95Author, Wyner, Tishby-memorial, Mallat, Boyd]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_override_invoked: false
council_predicted_mission_contribution: frontier_protecting
council_dissent:
  - member: Hotz
    verbatim: "Don't halt descent while you re-derive targets — a 10-step probe from W_seg costs ~45 minutes and gives the only decay number that matters. Argue after the probe, not before."
  - member: Carmack
    verbatim: "This memo will be re-litigated forever; the cheapest REAL pointer move is ~1,627 bytes off the incumbent at held distortion. Ship that row first so the pointer moves while the cathedral argues."
  - member: Quantizr
    verbatim: "PR128's 0.187946 is closed and unmerged; the only bar that binds is our own pointer. Flag the public claim, don't chase a ghost row."
council_assumption_adversary_verdict:
  - assumption: "S composes additively: 100*d_seg + sqrt(10*d_pose) + 25*bytes/37,545,489"
    classification: HARD-EARNED
    rationale: "VERIFIED_VIA_SOURCE_INSPECTION upstream/evaluate.py:92; cross-checked to 11 decimals against attempt-5 receipt advisory_action=26.28022354503111."
  - assumption: "The #613 box (136,839 errors, d_seg 0.00116) is a seg target that composes below the pointer"
    classification: CARGO-CULTED
    rationale: "VERIFIED_VIA_EMPIRICAL_ANCHOR (this audit): box + ancestor-class pose + 130,789B = 0.22024 > 0.19108. Box only composes at B<=87,002 with pose<=2.94e-5. Inherited across memos without re-summation."
  - assumption: "Pose is solved/banked: R1 dxi d_pose 0.00161 (contribution 0.127) is an acceptable pose plan-of-record"
    classification: CARGO-CULTED
    rationale: "Arithmetic: 0.127 pose + 0.0871 rate = 0.2140 > pointer BEFORE any seg term. Inadmissible at any d_seg. Ancestor-framing rot; PC1 on this vehicle has descent_was_run=false and POSITIVE conditional deltas (+2.17/+16.65)."
  - assumption: "Correction streams can close the box"
    classification: HARD-EARNED-NEGATIVE
    rationale: "VERIFIED_VIA_EMPIRICAL_ANCHOR: v19c asymptote d_seg 0.0248 (21x box) at +2B; MS2R cheapest realized price 405.5 B/corrected-error vs affordable ~0.024 B/error (16,700x gap). Corrections are free-finishers only (07-23 fork resolution) — settled, stop re-testing."
  - assumption: "Advisory batch32 macOS-CPU d_seg/d_pose ~= contest-CPU for the DDM vehicle"
    classification: UNCLEAR
    rationale: "ASSUMED_AWAITING_VERIFICATION: zero contest-axis rows exist for any DDM composed candidate; freeze memo already prescribes an incumbent-referenced calibration row (~2.6 expected)."
  - assumption: "Warm-starting #366 from W_joint (d_seg 0.0705) preserves pose optionality worth paying 2.9x the seg distance vs W_seg (0.0241)"
    classification: CARGO-CULTED
    rationale: "CC3 fresh composition already measures d_seg 0.024732 with PC1 active-zero; the campaign is descending seg debt the composed object does not have, for a pose coupling that does not compose anyway."
council_decisions_recorded:
  - "op-routable 1: build tools S-composition preflight (recompute S for any (d_seg,d_pose,bytes) triple; REFUSE campaign seals whose 100%-success composite >= pointer); wire warn-only into the governed launcher seal path"
  - "op-routable 2: re-derive + reseal #366 stage targets from the composing triple (d_seg <= 8.684e-4 i.e. <=102,445 errors at B<=130,789 with pose <=2.94e-5-class; or a joint KKT waterfill over all three axes) BEFORE any resume"
  - "op-routable 3: PC1 pose-leg descent smoke — measure reachable d_pose per byte on the solved-plane target; the 0.127 bank is fallback-inadmissible, retire it as a plan-of-record"
  - "op-routable 4 ($0 batch): v19c 104-admission tail-exponent fit; margin-mass N(delta) from sn1/at1 atlases; predictor-stream deletion ablation scored batch32; LP1's owed G4 same-object context price; gc1's preregistered CONNECTION conditional-codelength probe"
  - "op-routable 5: 10-step descent probe from W_seg parent with per-step exact verdicts (~45 min) — direct decay measurement at the 0.024 operating point; fix telemetry so exact verdicts land every step (attempt-5 had 3 in 50)"
  - "op-routable 6 (operator): ruling on an original-work rate-polish lane on the incumbent (~1,627B = sub-0.191 pointer move) and on the competitive bar vs the 0.18804 bank / PR128 0.187946 public claim"
  - "op-routable 7: one contest-CPU calibration row for the current composed candidate (freeze-compliant; incumbent=reference)"
related_deliberation_ids: [ddm_gc1_schmidhuber_symposium_20260724, ddm_v19c_correction_saturation_20260723, ddm_ms2r_r3_box_tolerance_solve_20260725]
research_only: true
score_claim: false
pointer_before: "0.1910828242 [contest-CPU]"
pointer_after: "0.1910828242 [contest-CPU]"
---

# T3 Grand Council — blind-spot hunt for frontier score lowering (2026-07-25)

**HEADLINE (honesty first): the live #366/c1 composed plan does NOT compose below the pointer, even at 100% success of its own sealed targets.** The 100%-success composite is **S = 0.9007** vs pointer **0.1910828242** — a **+0.7096 deficit**. Worse: the pose plan-of-record (banked R1, contribution 0.127) plus the composed rate (0.0871) already sum to **0.2140 > pointer with the seg term set to ZERO**. The plan died at the adder, and no artifact in the chain ever wrote the sum down. Every leg (box, pose bank, byte budget) is individually receipted and individually honest; the composition was never audited. That is THE blind spot.

## 1. The arithmetic audit (SEED 1) — exact contest units

Formula (VERIFIED upstream/evaluate.py:92): `S = 100*d_seg + sqrt(10*d_pose) + 25*rate`, `rate = archive_bytes / 37,545,489` (evaluate.py:63 — archive.zip stat only). d_seg is the argmax disagreement **fraction** over 600 samples x 196,608 px (512x384) = **117,964,800 sites** (sum of attempt-5 per_class sites = 117,964,800, confirming).

Formula application cross-check: attempt-5 receipt `advisory_action = 26.28022354503111`; recomputed `100*0.07051923116 + sqrt(10*36.618184751) + 25*138801/37545489 = 26.28022354503111`. Exact match to 11 decimals — the audit's unit conversions are sound.

**Errors <-> fraction conversions (explicit):**
- baseline: 8,318,787 / 117,964,800 = **0.07051923116** (= W_joint receipt d_seg 0.070519231 at 138,801B — exact)
- box allowance: 136,839 / 117,964,800 = **0.001159998576** (= MS2R BOX solve d_seg 0.001159998575846354 — exact)
- required allowance at plan bytes (derived below): 102,445 errors = 8.684e-4

### (a) Pointer decomposition — 0.1910828242 [contest-CPU], archive ad02b012…, 177,169 B

| term | value | source |
|---|---|---|
| 100*d_seg | 0.055972 (d_seg 5.5972e-4 = 66,027 errors) | click_polish_399 n600 row on the same sha |
| sqrt(10*d_pose) | 0.017152 (d_pose 2.9418e-5) | same |
| 25*rate | 0.117970 (177,169 B) | exact from bytes |
| **sum** | **0.191094** | component row 0.19109312; official pointer row 0.19108282 (component-consistent within 1.1e-5, separate measurement events) |

### (b) The c1/e-chain composed candidate at each sealed stage target (bytes 130,789 = LP1 DERIVED_COORDINATED_BUDGET post-cc3; pose = admitted-PC1-at-banked-R1 contribution sqrt(10*0.00161) = 0.126886; rate = 25*130,789/37,545,489 = 0.087087)

| stage target d_seg | seg term | composite S | vs pointer |
|---|---|---|---|
| stage 1: 0.020602722168 | 2.060272 | **2.27425** | +2.083 |
| stage 2: 0.013735148112 | 1.373515 | **1.58749** | +1.396 |
| stage 3: 0.006867574056 | 0.686757 | **0.90073** | **+0.70965** |
| the #613 box itself: 0.001159998576 | 0.116000 | **0.32997** | +0.13889 |
| box + ancestor-class pose (2.9418e-5) | 0.116000 | **0.22024** | +0.02916 |

Even the BOX with ancestor-class pose at plan bytes is **above the pointer by 0.029**. The box composes below the pointer only when **bytes <= 87,002** (at pose 2.94e-5) — or when d_seg goes well below the box.

Current measured composed reality (CC3 fresh n600 batch32, advisory): d_seg 0.024732, rendered d_pose 163.049, 136,116 B -> **S = 42.94** as rendered; **2.691** if the pose stream hypothetically delivered the (inadmissible) 0.127.

### (c) Required triples

To land **strictly below 0.1910828242**:
- At B = 130,789 (rate 0.087087): distortion budget 0.103996. With pose 2.9418e-5 (0.017152): **d_seg <= 8.684e-4 (<=102,445 errors)**. With the banked-R1 pose 0.00161 (0.126886): **INFEASIBLE — requires 100*d_seg < −0.02289**.
- At box seg (0.00116): pose+rate < 0.075083 -> with pose 0.017152, **B <= 87,002**. With pose 0.127: infeasible at any bytes.
- Pose axis requirement in general: contribution must be <= ~0.017–0.05, i.e. **d_pose <= 2.9e-5 … 2.5e-4** — the banked 1.61e-3 is 6.4–55x too large.

To land **below 0.15**:
- At B = 130,789, pose 2.9418e-5: **d_seg <= 4.576e-4 (<=53,982 errors)** — i.e., BEAT the incumbent's distortion (5.60e-4) at 74% of its bytes.
- At box-level seg (0.116 term): infeasible even at pose=0 unless B <= 51,062.

### (d) VERDICT

**NO.** The live plan's 100%-success case (0.9007) does not beat the pointer; it misses by 0.7096. The deficit decomposes: **seg 76%** (0.687 vs required <=0.087 of the budget — stage-3 target is 7.9x above the composing requirement 8.68e-4, which was ALREADY IN CLAUDE.md as "need ~0.00087" and drifted 8x without re-summation), **pose 18%** (0.127 booked where <=0.017–0.05 composes; on this vehicle PC1 has not even realized 0.00161 — descent never run, conditional deltas positive, composition d_pose 163), **rate 6-and-only-conditionally-fine** (0.0871 is acceptable ONLY if both other axes hit; at box-seg it must fall to <=87,002B). All three axes must move together; no single-axis success composes.

Corollary from the price ladder: closing from composed 0.024732 to 102,445 errors means removing **2,815,055 errors with NEGATIVE byte headroom** (136,116 -> <=130,789). The cheapest measured realization price is **405.5 B/corrected-error** (MS2R q4/q8) vs an affordable ~**0.024 B/error** — a ~**16,700x** gap. Correction streams are arithmetically dead as closers (consistent with the 07-23 v19c free-finisher verdict, asymptote 0.0248 = 21x box). Only description/regeneration — errors and bytes falling together — can close it.

## 2. Council positions (operating-within assumption stated per member; Catalog #292)

**Shannon (LEAD)** — *operating within: additive budget accounting is exact.* "This is pure accounting: 0.127 + 0.087 = 0.214 > 0.191 before seg exists. Separately: RAW_COMPACT winning 50/50 coder races (cc2/cc3) means the predictor stream is near its marginal entropy — remaining redundancy is CONDITIONAL (model-level), not symbol-level. Both facts point the same way: change the description, not the coder, and gate every seal on the written sum."

**Dykstra (CO-LEAD)** — *operating within: feasibility = intersection of constraint sets in one coordinate system.* "The box is a projection onto the seg constraint alone, computed under stale byte/pose assumptions. The feasible set at plan values is EMPTY. Re-derive the allowance jointly: 102,445 errors at (130,789B, 2.94e-5) — or run the KKT waterfill across all three axes at once. Alternating projections onto stale sets converges to a point outside the true intersection."

**Rudin (CO-LEAD)** — *operating within: explanations are contracts.* "Every receipt in this chain honestly says score_claim=false — and that discipline created the gap: no artifact was ever OBLIGED to state the composed claim, so the false composite was never written where a gate could refuse it. Revision: every campaign seal carries a one-page falling-rule 'does-it-compose' statement with the three terms and the sum."

**Daubechies (CO-LEAD)** — *operating within: pick the basis where the target is sparse.* "405 B/error says the correction family's coefficients are not sparse where the errors live. The error mass is coarse-scale scene structure (Road/MyCar trunk buckets: 6.85M excess errors); the plan is spending fine-scale coefficients on it. Route capacity to the coarse generative chart (family-d), keep corrections for the final <=100K errors where they are provably free-finishers."

**Yousfi** — *operating within: only the scorer's argmax partition is real.* "The role_correction bucket (Lane/Movable) sits at 1,329,531 errors vs a 726,416 ceiling — 603K errors above ceiling in the two classes with the highest per-error S leverage per site. The incumbent full-RGB HNeRV reaches 5.6e-4 because everything it spends is scorer-visible. Any vehicle that wants the pointer must beat THAT number, not the box."

**Fridrich** — *operating within: the instrument must be able to measure the quantity the decision depends on.* "Attempt-5 recorded 3 exact verdicts in 50 steps; its own engage detector returned INSUFFICIENT_EXACT_VERDICTS; final state bit-identical to step 0 after ~3.75h. The campaign cannot observe the descent rate its go/no-go depends on. Also: `target_d_pose = 163.06` in a SEALED schedule is a placeholder promoted to a target — a confound-class artifact (garbage-as-sealed-constant)."

**Contrarian (double weight)** — *operating within: the quietest fake is target drift.* "Two facts on one table: (1) we hold a MEASURED 0.18804 at 176,564B in the bank, walled off as borrowed; (2) our original-work plan composes to 0.90 at 100% success. Either the operator explicitly authorizes deriving our own latent-polish-equivalent as original work, or the plan gets re-derived — but continuing #366 as-sealed spends ~39h certifying a number that cannot beat the pointer. And the 8.7e-4 requirement was in CLAUDE.md all along; the campaign relaxed it 8x with no written justification. That is the failure mode NO-FAKE exists for, wearing rigor as camouflage."

**Assumption-Adversary (double weight)** — classifications in frontmatter. Meta-verdict: "The shared backdrop assumption across ALL recent DDM memos is *'per-leg honesty composes into plan honesty.'* It does not. Legs were verified; the SUM was assumed. Add the composition inequality to the apparatus (gate), not to good intentions."

**Schmidhuber** — *operating within: the shortest program wins.* "gc1 stands, strengthened: coder saturation + the 405 B/error wall are exactly what you see when the program class is wrong. Family-(d) event-continuation is where errors and bytes fall together. Run the preregistered $0 CONNECTION conditional-codelength probe NOW — it was op-routable 4 of gc1 and is still unexecuted."

**Hotz** — *operating within: measure first, argue second.* "Delete the 100,099B predictor stream and score what breaks — one batch32 ablation, ~10 minutes. And the inner loop at ~270s/accepted-step is an engineering choice: batch the proposal verdicts. Dissent recorded on halting descent."

**Carmack** — *operating within: smallest credible increment.* "Sub-0.191 pointer move = −1,627 bytes off the incumbent at held distortion (matches the runbook's −1,651.74). While the grand plan is 0.71 away, a 1.6KB rate polish is a REAL row. Dissent recorded on ordering."

**MacKay** — *operating within: the margin distribution is the sufficient statistic for descent decay.* "SEED-2 answer: don't extrapolate one point. N(delta) = #pixels within delta of a flip (from the sn1/at1 atlases) IS the integrated decay curve; the descent rate at operating point d is bounded by margin mass near zero. Plus the v19c receipt already holds a 104-point measured admitted curve — fit its tail exponent for $0. Both bound terminal d_seg before burning 39h."

**Ballé** — *operating within: side-information should be modeled, not stored.* "The 100,099B stream is side information the receiver could largely regenerate. LP1's G4 context row is the existence proof: 490,794 explicit bytes beaten by 401,633 context bytes with ZERO counted context params on its own stream (+89,161B gain). The same-object price for the v15 stream is OWED and is a $0 local encode. That is the highest-EV rate measurement on the table."

**Selfcomp** — *operating within: store only what the receiver cannot infer.* "RAW winning every race means those records look uniform WITHOUT context — that's the signature of storing what conditioning would remove. Condition the records on decoded frame0 + masks before coding; my whole 0.38 archive was built on that principle."

**Quantizr** — *operating within: the leaderboard rewards what it measures.* "Public best claim is 0.187946 (PR128, closed/unmerged); our non-submission bank measured 0.18804. Beating our own pointer by 1.6KB still trails the best public claim. Name the bar explicitly. Dissent recorded: the binding bar remains our pointer."

**PR95-author** — *operating within: I know what these numbers look like from the inside.* "My family's submissions reach d_seg ~5.6e-4 by making every parameter scorer-visible. An 8x-relaxed seg target would have been caught by simply diffing against my numbers — do that diff at every seal. The task-space vehicle must beat 5.6e-4-at-fewer-bytes; that is the whole game."

**Wyner** — *operating within: conditional entropy given decoder side information is the bound.* "The receiver holds decoded frame0, masks, and the worldsheet — code frame1 structures GIVEN that. Records priced without conditioning pay the marginal-entropy price; the measured RAW dominance is that overpayment made visible."

**Tishby (memorial)** — *operating within: bytes should carry only scorer-relevant bits.* "Compute I(stream; scorer-visible partition) per stream. The null-subspace finding (ker(A) ~52% rate-neutral) says half the representation can be information the scorer never reads — LP1's typed homes are the scaffolding for dropping it."

**Mallat** — *(consulted)* "Concur with Daubechies: coarse chart first; the 2 compile-infeasible worldsheet moves in v19c hint the fine grammar is also brittle at its edges."

**Boyd** — *(consulted)* "Concur with Dykstra: state all three constraints in one program and solve once; per-axis targets derived at different times ARE the bug class."

## 3. Tally

Sextet+co-leads: 8/8 PROCEED_WITH_REVISIONS (Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary; no veto invoked — the assumption-violation hypotheses were engaged). Grand council: 12 voted — 10 PROCEED_WITH_REVISIONS, 2 PROCEED_WITH_REVISIONS-with-ordering-dissent (Hotz, Carmack), 1 scope-dissent recorded (Quantizr, on the competitive bar). 0 REFUSE, 0 abstain, 0 recusals. Quorum: 6-of-6 sextet + 12-of-20 grand — met for T3.

**Council verdict: PROCEED_WITH_REVISIONS** — the DDM program continues, but #366 may NOT resume on the sealed targets; revisions 1–3 (S-composition gate, target re-derivation, pose-leg re-scope) are blocking.

## 4. RANKED BLIND-SPOT TABLE

| # | blind spot | evidence | flagged by | cheapest resolving measurement | route |
|---|---|---|---|---|---|
| 1 | **No S-composition audit anywhere in the chain** — 100%-success case = 0.9007 > pointer; pose 0.127 + rate 0.0871 = 0.214 > pointer at seg=0 | this audit; every receipt says score_claim=false so no artifact owned the sum | Shannon, Dykstra, Rudin, Contrarian, Assumption-Adversary | $0: 20-line composition tool run on the plan's own numbers (done in this memo) | NEW: seal-gate `check_campaign_seal_composes_below_pointer` (warn-only first) |
| 2 | **Pose plan-of-record is arithmetically inadmissible** — banked R1 0.127 contribution vs <=0.017–0.05 required; PC1 descent never run, conditional deltas POSITIVE, composed d_pose 163 | PC1 FEED; ms2r blockers (inactive pose tube); this audit | Dykstra, Fridrich, Assumption-Adversary | PC1 solved-plane descent smoke: reachable d_pose per byte | #366 pose leg re-scope; retire 0.127 as fallback |
| 3 | **Target drift 8x** — sealed stage-3 6.87e-3 vs composing requirement 8.68e-4 (which CLAUDE.md carried as "~0.00087" all along); box itself needs B<=87,002 to compose | attempt-5 schedule; CLAUDE.md trilemma section; this audit | Contrarian, PR95-author, Dykstra | $0: re-derive allowance table at (bytes, pose) jointly | reseal #366 schedule before resume |
| 4 | **Correction-price wall treated as open** — 405.5 B/err measured vs 0.024 B/err affordable (16,700x); corrections = free-finishers only | ms2r; v19c asymptote 0.0248 (21x box) | Daubechies, Schmidhuber, Ballé | none — settled; stop aiming correction streams at the box | re-aim #366 at describe/regeneration (gc1 family-d) |
| 5 | **Descent instrument cannot measure decay** — 3 exact verdicts in 50 steps; detector says INSUFFICIENT; warm-started from W_joint (0.0705) when the composed object already measures 0.0247 | attempt-5 receipt; cc3 premise falsification | Fridrich, MacKay, Hotz | $0: v19c 104-pt tail fit + margin-mass N(delta) from atlases; ~45min: 10-step W_seg-parent probe with per-step exact verdicts | fix #344 detector config + reseal warm-start parent |
| 6 | **The 100,099B predictor stream** — coder axis exhausted (RAW 50/50; −3,422B total), structural replacement unmeasured; G4 context gain (+89,161B) exists but same-object price OWED | cc2/cc3; LP1 | Ballé, Selfcomp, Wyner, Hotz | $0: LP1's owed G4 same-object context encode; ~10min: predictor-deletion ablation scored batch32 | la1 + LP1 owed row |
| 7 | **Competitive bar ambiguity** — public best claim 0.187946 (PR128, closed) and our 0.18804 bank both sit below our pointer; mission says "frontier lowering" but the plan targets only 0.19108 | pointer file leaderboard snapshot; bank memo | Quantizr, Contrarian | none — operator decision, not measurement | operator ruling (bar + original-work latent-polish stance) |
| 8 | **Cheapest real pointer move unowned** — −1,627B off incumbent at held distortion = sub-0.19108; nobody owns it while DDM matures | click-polish runbook (−1,651.74B ≡ sub-0.19) | Carmack | existing la1/rate-attack machinery | existing rate-polish tasks on incumbent |
| 9 | **Export-chain cross-stream coupling** — activating PC1-zero changed d_seg 0.0702->0.0247 (CC2 reuse premise FALSIFIED); pose plateau fallback emits the inadmissible 0.127 bank | cc3 | Fridrich, Yousfi | already measured (cc3); enforce fresh-composition scoring on every composed claim | R6 consumer guidance -> gate |
| 10 | **Advisory-axis drift unmeasured for DDM vehicle** — all DDM rows are [macOS-CPU advisory]; zero contest-axis rows exist | all DDM receipts | Assumption-Adversary, Quantizr | <$1: one Modal contest-CPU calibration row of the composed candidate (freeze-compliant, incumbent=reference) | freeze-memo calibration row |

## 5. SEED-2 answer (descent decay), condensed

One point (−2.1e-4/step at step 1, then blocked regression) supports NO extrapolation — the run's own detector says so. Before any long burn: (i) $0 fit of the v19c measured 104-admission curve tail (same coordinate families as #366 proposals); (ii) $0 margin-mass N(delta) from the atlases — the mass of pixels within delta of a flip bounds the achievable errors-removed at any price and directly predicts terminal d_seg; (iii) one 10-step probe from the RIGHT parent (W_seg / the composed 0.0247 object) with exact verdicts every step. But note: per Section 1, decay measurement is subordinate — even zero decay to the box does not compose below the pointer at plan bytes/pose. Re-derive targets first.

## 6. SEED-3 answer (the 100,099B elephant), ranked by measured prices we hold

1. **Rule-118 distillation into receiver context (Ballé/Wyner/Selfcomp):** generic conditional-regeneration code is FREE; only video-derived params counted. Measured existence proof: LP1 G4 context beat explicit bytes by 89,161B on its stream with 0 counted context params. Owed: the SAME-OBJECT price for v15 (LP1 explicitly refuses transfer without it). $0 local encode — do first.
2. **xi-conditional regeneration / family-(d) event continuation (Schmidhuber/gc1):** ranked first by gc1; supporting price: 4,124->1,569B joint coding (2.6x); $0 CONNECTION probe preregistered and unexecuted.
3. **Coder axis: CLOSED.** RAW_COMPACT 50/50; total coder headroom measured at −3,422B (−2.5%). Do not spend more here.
4. **Worldsheet grammar replacement:** weakest measured support (v19c worldsheet moves saturating + 2 compile-infeasible). Hold.
0. **(Hotz precursor)** deletion ablation to price what the stream actually buys — 10 minutes, do alongside (1).

## 7. STORES CONSULTED

canonical_frontier_pointer.json (pointer 0.19108282419209976, ad02b012…, 177,169B; bank 0.18804 @176,564B; CUDA 0.20533) · attempt-5 full_run_receipt.json + run.log (verdict BLOCKED_REALIZED_DSEG_REGRESSION; baseline d_seg 0.070519231/8,318,787 errors; per_class; c1_debt_buckets incl. integer_target_error_allowance 136,839; stage targets 0.020603/0.013735/0.006868; target_d_pose 163.06; step_seconds; exact_d_seg [0.070519,0.070309,0.070519] @ steps [0,1,50]) · codex_findings_ddm_lp1_layer_pricing (134,211B typed allocation; 65,789 headroom NOT a reserve; G4 +89,161 stream-scoped; same-object price owed) · codex_findings_ddm_ms2r_r3_box_tolerance_solve (BOX 136,839 errors = d_seg 0.001159998575846354; 405.497 B/corrected-error; family realization 291MB -> S 194.4) · codex_findings_ddm_cc3 (136,116B receiver-closed; −3,422B; fresh composition d_seg 0.024732 / d_pose 163.049; CC2 premise falsified; LP1 budget 130,789 DERIVED) · FEED_ddm_pc1 (+2.17/+16.65 conditional deltas; descent_was_run=false; 40B home) · codex_findings_ddm_rd1 (knee custody; q1 control 17,927 errors) · codex_findings_ddm_v19c (asymptote d_seg 0.0248, ΔS −0.1807 @ +2B; 104 admissions) · codex_findings_ddm_gc1 (family-d PROVISIONAL; two-part charge boundary; CONNECTION probe preregistered) · click_polish_399 + clickpolish runbook (incumbent components 0.00055972/2.9418e-5/177,169; sub-0.19 ≡ −1,651.74B) · upstream/evaluate.py:63,92 · ws2 W_joint receipt via attempt-5 source_archive (138,801B, sha 5aa45850…).

*Pointer delta: NONE — 0.1910828242 [contest-CPU] unmoved. This memo is apparatus (frontier_protecting): it prevents a ~39h burn toward a non-composing target and re-aims the campaign at the composing triple.*
