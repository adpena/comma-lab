---
doc_type: t5_crucible_p5_second_redteam_verdict
role: P5 SECOND RED-TEAM (verify pass — operator-convened T5 crucible)
date: 2026-07-07
target: DRAFT_OPTIMAL_STACK_v2_20260707.md (305d884ce)
method: every §0.1 disposition checked for REALIZATION in its claimed section; load-bearing
  arithmetic + the §2.2b backtest INDEPENDENTLY RE-EXECUTED this session against the on-disk
  artifacts (not re-read — re-run); plus three v2-only angles (regression / interaction /
  launch-readiness) and the two operator pins (lane-anisotropy scope; apples-to-apples +
  per-class comparison hygiene, reqs H+I).
axis: all numbers [macOS-CPU/MLX advisory] unless tagged; pointer contest-CPU 0.19110 UNMOVED —
  this verdict is MEANS.
review_status: fresh-eyes second-pass (this doc), written by a verifier that did not author v1,
  v2, or P3; every independent re-computation below is tagged [re-executed]; readings of v2/P3
  text are tagged [verified-by-inspection].
---

STORES CONSULTED: DRAFT_OPTIMAL_STACK_v2_20260707.md (full) · P3_redteam_verdict_20260707.md
(full, F1-F17 + PASS-3 + counter-frame) · DRAFT_OPTIMAL_STACK_20260707.md v1 (full, diff
mentality) · pursuit_chainA_spectrum_solve_20260707.md (full, current on-disk state) ·
ORCHESTRATION_LEDGER (reqs A-I incl. the 2 operator addenda, counter-frame, queue, log) ·
islands_composed_ceiling_arithmetic_20260707.md (full; arithmetic re-verified) · mod32cap run dir
`levelset_n600_witness_mod32cap_20260706T115554Z/` (levelset_train_result.json 41-row history —
**backtest RE-EXECUTED**; launch.sh; per-stage ckpts confirmed on disk) · chain-A artifact dirs
(`t5_pursuit_chainA_20260707/` + `t5_s3_hvp_lanczos_20260707/` — spectrum jsons K8/K32 present,
K128 npz-only confirmed = branch honestly carried) · trainer argparse
(`experiments/train_levelset_witness_realized_through_R_mlx.py` — **18 law-carrying v2 flags
re-verified incl. 2 multi-line declarations**) · `curriculum_dsl.py` L2179 (AACoverageRender
factory confirmed) · `tools/witness_memory_preflight.py` (full header + projection formula —
**AA/supersample term confirmed ABSENT**) · `tools/launch_witness_run.py` (throughput gate
structure) · canonical_equations_registry
(`gn_hessian_spectrum_indefinite_at_ema_best_v1` row dumped; **AWAITING recount re-executed: 22
rows / 21 unique ids**) · DAG sub015 FEED-07g (L8991 — compose-after-downsample BUILT,
a2f4acee7, confirmed) · CLAUDE.md non-negotiables · operating manual. NOT consulted:
durable-state files (stale per sweep); the parallel meat-hunt lens's synergy matrix (owned
elsewhere per operator addendum — not duplicated here).

# P5 SECOND RED-TEAM VERDICT — SEAL-TO-RECESS (with a NAMED amendment list; no FAILS)

**One-line:** v2's dispositions are REAL, not paper — the crossing arithmetic re-verifies to the
digit, the §2.2b backtest reproduces bit-for-bit from the on-disk trace, the islands fold is
faithful, all 18 spot-checked flags exist, and no v1 strength was lost. Six findings are PARTIAL
on amendment-grade gaps (the sharpest: the P11 memory gate is AA-BLIND in exactly the #205/C4
false-green class it exists to extinct, and the 600-denominator introduces an un-named
anneal-speed confound). Nothing is FAILS-grade; nothing requires another full revision cycle.

---

## §1 — PER-FINDING VERIFICATION (F1-F17 + PASS-3)

| finding | verdict | evidence (what I re-did, not just re-read) |
|---|---|---|
| F1 crossing arithmetic | **HOLDS** | [re-executed] every §0.2 row by hand: 0.110+0.017320+0.0620=0.189320 ✓ (margin 0.00178); knife-edge 0.105+0.024083+0.0620=0.191083 ✓; S6 triple 0.092+0.038859+0.0602=0.191059 ✓; v1's false tail 0.220730 ✓ printed as the negative example; central 0.260730 ✓ does-not-cross stated. Budget legs: 0.19110−{0.0620,0.0573,0.0469,0.0768,0.0855}={0.1291,0.1338,0.1442,0.1143,0.1056} ✓ all five. §9.1 S-band lower 0.10+0.0173+0.0469=0.1642 ✓ component-consistent (the F1 corollary fixed). Nit: §9.1 rung-1 upper 0.0028 vs §0.3's design-sum upper 0.0023 — the widening is conservative-direction but underived; one sentence owed. Crossing triple correctly carries rate ≤0.062 (at rate-upper 0.0689 the same d_seg/pose triple sums 0.19622 — does NOT cross; v2's conditioning is right). |
| F2 #342 inventory | **HOLDS** | [verified-by-inspection] §11 exists, 15 blocks, each SOLVED/TRAINED-with-reason/NOT-SOLVABLE-with-proof; #11-13 proofs cite the actual chain-A receipts (gradstep worse at every η; negcurv rescreen flip; #341 +5.1%) — all cross-checked against the chain log + `gradstep_cross_subset.json` / `rescreen_negcurvK32_p02.json` (present on disk). |
| F3 vacuous TAU exit | **PARTIAL** | Event now REAL and the law arithmetic checks (625−25=600; earliest admissible fire = ep625 on the control trace, 101 ep pre-cap ✓). BUT three gaps: (a) **anneal-speed confound un-named** — `--anneal-epochs 600` makes the τ/β geometric paths ~1.67× faster than the control trace (denominator 1000) the backtest ran on, and 1.15× faster than v1's 726; the co-pred fire epochs were measured under the SLOW anneal, and the M2 "completion recovery" (already the named binding constraint, UNMEASURED) is now targeted at a never-run anneal speed. v2 states only the "arm levers shift exhaustion LATER" direction; the faster-anneal direction (earlier exhaustion / transient instability) is unstated. One paragraph + optionally upgrading B9 from optional to preferred resolves it. (b) §8's event path (21.6-26.7 h, "worth ~5-27%") assumes the TAU→FIN trigger is ARMED, which requires B1 (~80 LOC, owed) + injection test — §2.2 states B1 owed but §8 does not carry the B1-contingency tag; without B1 the realized path is cap (29.7 h). (c) the trigger fires one cadence BEFORE the control's measured best (ep625 vs ep650) — i.e. mild EARLY fire (~0.6% d_seg of TAU descent foregone, bounded by warm-Muon + B4 restore-best); v2 states the fact but frames it only positively. |
| F3/F6 backtest claim | **HOLDS — REPRODUCED EXACTLY** | [re-executed] ran the co-predicate over the on-disk 41-row `levelset_train_result.json` (cadence 25 confirmed). Matching definition: trailing V=4 VERDICT POINTS (3 intervals), rel slope per 25 ep = (d[i]−d[i−3])/d[i−3]/3. Results: eps 5e-3 → first fire **ep625, slope −1.3694e-3, n=8** (v2: ep625/−1.37e-3/8 ✓); 1e-2 → ep575/10 ✓; 2e-2 → ep275/23 ✓; V=5,5e-3 → ep625/9 ✓; slope at ep275 = −1.888e-2 ≈ the quoted −1.9%/25ep ✓. CE-side: no fire < ep300 at 5e-3 ✓ (cap-fired CE stated, accept-and-state honored). The 2e-2 rejection is sound (would fire ep275 mid-descent). |
| F4 AA admission | **PARTIAL** | AA IN from ep0 ✓; BA decode = same blocker class as B6, both LB-at-byte-close ✓; FEED-07g compose-after-downsample confirmed in the DAG (L8991, a2f4acee7) ✓; AA byte-cost 0 ✓; byte-close-selectable repair ✓; attribution via stage-boundary paired verdicts at fixed θ = apples-to-apples by construction ✓ (req H). **BUT P11 is under-specified in exactly the class it guards:** [re-executed] `tools/witness_memory_preflight.py` contains ZERO AA/supersample terms — its `pix_ratio` uses `--render-hw` only, so an ss=2 config (≈4× render px on the render-linked resident terms, incl. fine-mode self-orient features) gets the C4 false-SAFE (the in_feat-blind precedent the tool's own header narrates). And the 5-ep governed smoke NEVER fires an n600 verdict (cadence 25 ep), so the empirical half misses the +5.6 GiB (chunked) verdict transient. Each half covers the other's blind spot only if combined explicitly; v2 does not say so. **Amendment (small, pre-GO):** add an ss²·pix_ratio factor to the preflight's render-linked terms (~5-15 LOC) AND either force one n600 verdict inside the smoke or add the known chunked-verdict delta to the smoke-measured RSS before the SAFE call. |
| F5 composed ceiling | **HOLDS** | [re-executed] memo arithmetic: island share 0.4396+0.1226=0.5622 ✓; ceiling rows 100·0.5622·0.004571=0.257 ✓, ·0.003636=0.204 ✓; lower edges 0.022/0.072/0.065 ✓ all ≥4× the 0.005 gate ✓. §0.3 per-class transfer on 0.0034: lane 0.001495, movable 0.000417, big-3 0.001489, sum 0.0034 ✓; islands-only floor 0.0034·0.4378=0.001489≈0.0015 ✓. Shares are ep300/16-pair (+4.7% subset) transferred to ep650-best — ASSUMED, stated, P6 rides ✓. Per-class physics respected (req I): lane=dash-birth/placement levers, movable=SDF-dilation (5.3% within-flip, largely solved), big-3=anneal-completion — three different objects, three different lever rows ✓. |
| F6 telemetry promotion | **HOLDS** | F3/F4 rows PROMOTED to LB whenever any trigger ships armed ✓ (§4.1, LB set F1-F4,F9,F10,F11). |
| F7 attribution | **HOLDS (one condition owed)** | Twin=λ0, pose ON = clean single-dim comparator ✓; Class-D×B recess de-confounded (waterfill BOTH twins' tau-boundary ckpts) ✓; pose-interference attribution routed to run-2 with confound NAMED ✓; per-stage kills restated STACK-level and REALIZED in the §7 RUN row ✓. **Req-H caveat:** "matched epochs" across two EVENT-scheduled runs can compare different stage-states (λ=15 vs λ=0 will fire the co-pred at different epochs). Amendment: pin the twin to the PRIMARY's realized stage boundaries (mirror-schedule the twin) or compare at stage-relative epochs — one clause. |
| F8 chain-A fold + SOLVE | **PARTIAL (scope, per operator pin)** | Measured-acceptance HARD on SOLVE ✓; basin predicate annotated advisory-on-smoothed-operator ✓; folds faithful to the chain log (K-ladder 2.65→1.30→0.28; transfer test; isotropy 1.00-1.05; no-descent at every η; +5.2% int8 gap — all cross-checked against the artifacts; K=128 honestly carried: npz present, eigen json ABSENT, confirmed). v2 does NOT demote any lane-anisotropy lever on the isotropy negative — AA/band/comb/along=8/Rebalance-A/B all unchanged or PROMOTED ✓ (no category error; pin item (a) clean). **BUT the negative's scope is overstated as written:** "the hoped Lanczos shortcut is dead" — dead AT A LANE-DILUTE CHECKPOINT. mod32cap's training loss is area-weighted with NO lane up-weighting levers (lane ≈0.58% of pixels; the composed-ceiling memo shows lane at 44% of d_seg but that is the VERDICT surface, not the loss the HVP differentiates), so the ep650 Hessian is structurally near-blind to the lane-anisotropy axis; at a lane-bearing run-1 checkpoint (islands born + logit-adjust + LengthSigma up-weighting lane) the spectrum is a NEW measurement and anisotropy may appear in the lane-coupled blocks. [re-executed] registry row `gn_hessian_spectrum_indefinite_at_ema_best_v1` `domain_of_validity` carries checkpoint/axis/PROVISIONAL-K8 but NOT the lane-blind caveat — **flag for integration tranche 2**, and I-6's planned `hessian_negative_curvature_subset_artifact_v1` must carry it at birth; the F5 checkpoint-cadence Lanczos telemetry row should note that lane-bearing checkpoints re-open the question. |
| F9 ep0 gate | **HOLDS** | Gate re-predicated ✓; per-class hygiene GOOD (req I): movable gets the ABORT (its transfer mechanism is the proven one), lane gets an ep150 ALARM milestone calibrated from the memo's measured ep300 value (0.003 ≈ 50% of GT mass 0.00577 ✓) — a movable-calibrated check no longer silently governs lane. The paintseedON counterexample now PASSES the abort class (painted-px>0 ∧ −36% ≤0.8× ∧ movable part_frac — the last never measured at ep0, which is exactly what P12 measures pre-GO, and P12 is LB ✓). ep150 milestone is an interpolation from one ep300 point — acceptable as ALARM-only. |
| F10 q-law | **HOLDS** | smallest q ✓ (§1.2). |
| F11 byte bands | **PARTIAL (one label)** | [re-executed] all component sums + rates to 4 decimals: central 60,000−3,108+30,892+4,500+800=93,084 → 0.06198 ✓; independent band 70,392/103,513 → 0.04687/0.06893 ✓; waterfill-fail 115,277 → 0.07676 ✓. **Worst joint tail: printed 128,376 = 82,193−3,108+41,562+6,929+800 — it silently includes the pose-ξ UPPER tail (6,929) while the row label names only "waterfill-fail ∧ B6-slip"** (the two-leg version P3 asked for = 125,947 → rate 0.0839). Arithmetic right, label wrong — rename the row "waterfill-fail ∧ B6-slip ∧ pose-byte-upper" or print both. |
| F12 chroma label | **HOLDS** | Regrouped under score-affecting loss levers in the §1.1 sketch ✓. |
| F13 throughput | **HOLDS** | Folded into P11; §8 tagged pending-P11 ✓ (subject to the F4 P11 amendments). |
| F14 launch surfaces | **HOLDS** | §7 RUN row names `tools/launch_witness_run.py` (raw-python FORBIDDEN) + `witness_memory_preflight` at the REAL config ✓ — with the F4 caveat that the preflight must gain the AA term to be honest at THIS real config. |
| F15 ordering | **PARTIAL (letter)** | Substance realized: P7 now precedes P4/P3/P8/P9/P10 and every run (v1 had it behind the hour-scale probes). Letter not: §0.1 claims "P7 decode-integrity second" but P7 prints 8th, behind P6 (2-3 h) and P1 (~1 h). Either reorder P7 ahead of P6/P1 or state the concurrency plan (they are independent $0/CPU probes — one sentence). |
| F16 provenance | **HOLDS** | τ_e=305 re-tagged INFERRED with B4 carrying the weight ✓; LBND2 36 B routed to P5 ✓; [re-executed] AWAITING recount: **22 rows / 21 unique ids — matches v2 exactly** (v1's "15" corrected ✓). |
| F17 dual model | **HOLDS** | Both models printed; Model A arithmetic checks ((0.10-0.20)·(0.2-0.4)·0.8 = 1.6-6.4% ≈ "2-6%") ✓; per-lever repairs named incl. the two NEW ones (AA byte-close-selectable; λ-twin promotion) ✓; non-repairables named ✓; central adjudicated between the models, not asserted ✓. Model B's 8-15% is a printed judgment, labeled as such. |
| PASS-3 #149 | **HOLDS** | §9.4 DEFER-with-build-spec, partially-represented-by-AA stated ✓. |
| PASS-3 comb law | **HOLDS (one nit)** | Complete conditional law: P1-PASS ⇒ in-training only (render-composite +0.0038 receipt), engage band-fire+25, F8 paired row, 2-window kill + restore, P1-FAIL branch removes the run-2 A/B ✓. Req-I nit: comb's ARCHIVE byte cost is unstated in §5.1 — presumably 0 (phase derived from the rule-118 band geometry), but the crossing arithmetic deserves the one-line receipt. |

## §2 — THE THREE v2-ONLY ANGLES

**(1) REGRESSION (v1 → v2): CLEAN.** Checked law-by-law: every v1 §1.2 row inherited or
explicitly superseded; completion guarantees survive (600 ≤ cap 726 makes the anneal-complete
guarantee STRONGER, not weaker); req-E FOLD/DEFER/DEAD table, PR95 cargo audit, DECIDE
prohibitions, ACT boundary, B1-B8/W1/W2/I-1..5/T-1..3 all inherited ✓. No telemetry row
orphaned (F5 Lanczos row LB via I-5; F8 attribution rows still consumed by comb/AA). Two
deliberate trades, both STATED: the seg-only twin's pose-interference co-predicate (M5 via twin)
→ run-2 with confound named; v1's confounded Class-D×B vs-mod32cap → the cleaner twin-pair
version. One dropped sentence (v1 §8 "GPU-reorient only if its parity probe passes") — restore
it or confirm it lives in the inherited §1.2 determinism row; cosmetic.

**(2) INTERACTION (newly-promoted levers): guards declared; budget closes.** AA×band/seed seam:
resolved-by-build (FEED-07g byte-identity proofs — verified in DAG) ✓. comb×band: engage
band-fire+25 = one-homotopy-per-neighborhood ✓. AA×throughput/memory: P11 (with the F4
amendments) ✓. AA and comb both add 0 archive bytes (AA verified render-side; comb asserted —
see F11/comb nits), so the worst-joint byte case with ALL promoted levers = the printed 128,376
→ rate 0.0855, budget 0.1056 — closes ✓. One interaction v2 does not name: the TAU→FIN
co-predicate is a POOLED d_seg slope — on the arm, per-class cancellation (lane descending while
big-3 regresses) could fire it while the binding class still pays; the promoted-LB per-class F3
meat rows give the observability, but a one-clause per-class veto (do not fire while any class's
own slope < −eps with stable shares) would close it structurally. Flag for the meat-hunt lens
(law derivation is its lane, not mine).

**(3) LAUNCH-READINESS: launchable-as-written except the named LB builds, honestly labeled.**
[re-executed] 18/18 law-carrying flags exist in the levelset trainer argparse (incl.
`--muon-warm-start-momentum` / `--muon-lr-final-frac`, multi-line declarations at L8169/L8180);
`AACoverageRender` factory at curriculum_dsl L2179 ✓; I-4/I-5 (ChromaBoundarySharpen,
GNSpectrumProbe) correctly declared NEW+LB. Governed launcher + memory preflight named (F14) ✓;
resumability + per-stage EMA checkpoints inherited AND evidenced (mod32cap run dir carries
per-stage ckpt+resume npz files on disk) ✓. Req-B per-trigger honesty: CE→TAU backtested ✓ +
T-1 owed (stated); TAU→FIN backtested ✓ + B1 in-trainer trigger owed (stated in §2.2 — but §8's
event-path hours need the B1-contingency tag, F3(b)); FIN→END B4+T-2 owed ✓; run-end B5+T-3
owed ✓. P11/P12 are LB pre-GO probes ✓ — P11 needs the F4 amendment to be a real gate.

## §3 — OVERALL VERDICT: **SEAL-TO-RECESS** (no FAILS; amendment list binds at recess close)

No finding FAILS. Six PARTIALs are all amendment-grade (one paragraph, one label, ~15 LOC, one
clause each) — none changes the architecture, the lever set, or what the recess must measure.
Requiring a full P3c revision cycle for these would be rigor-overhead; requiring them folded
before GO is non-negotiable. **Binding amendment list (fold into the recess-exit revision):**
1. P11 spec: AA-aware preflight term (ss²·pix_ratio on render-linked components, ~5-15 LOC) +
   ≥1 forced n600 verdict in the 5-ep smoke (or add the known chunked-verdict delta) — F4.
2. F3 paragraph: name the 600-denominator anneal-SPEED confound (1.67× vs control) on both the
   co-pred transfer and the M2-recovery premise; consider upgrading B9 (re-anchor law) from
   optional to preferred; add the B1-contingency tag to §8's event-path hours.
3. F11 label: worst-joint-tail row names its THIRD leg (pose-byte upper) or prints the two-leg
   125,947/0.0839 alongside.
4. F7 clause: twin compared at the primary's REALIZED stage boundaries (mirror-schedule), not
   raw matched epochs.
5. F8/pin: lane-blind scope sentence on the isotropy negative in §2.3(4); domain_of_validity
   caveat on `gn_hessian_spectrum_indefinite_at_ema_best_v1` + on I-6's new row (integration
   tranche 2); F5 telemetry note that lane-bearing checkpoints are a NEW spectrum measurement.
6. Nits: §9.1 rung-1 upper 0.0028 derivation sentence · comb archive-byte receipt (0, rule-118)
   in §5.1 · P7-vs-P6/P1 order or concurrency sentence · restore the GPU-reorient parity caution.

## §4 — THE FINALIZED DECISIVE RECESS LIST (P4; cost · pre-registered band · kill · unblocks)

| # | item | cost | pre-registered band | kill threshold | unblocks |
|---|---|---|---|---|---|
| R1 | P5 LBND4-on-smoothed source | $0, ~1 min | 18-22 KB | ≥24,149 B ⇒ no gain | band-coder min + the 36-B LBND2 discrepancy (F16ii) |
| R2 | P12 ep0 init probe, EXACT ARM-PRIMARY seed config (1 epoch) | $0 | painted-px>0 both classes; init d_seg ≤0.8× control; part_frac[movable] measured (first-ever at ep0) | any abort-class miss ⇒ recalibrate before arming | the §3.1 gate constants (F9); LAUNCH-BLOCKING |
| R3 | P11′ AA memory+throughput gate AS AMENDED (§3 item 1) | $0, ~15 min + ~15 LOC | peak RSS SAFE at REAL config incl. verdict spike; s/ep ≤ 1.5×107 | preflight REFUSE or >1.5× ⇒ AA → run-2 with measured cost written | F4 AA-in decision + §8's s/ep base; LAUNCH-BLOCKING |
| R4 | P1 comb-REGISTRATION audit | $0, ~1 h | comb separates marks/gaps ≥ GT-sep floor | FAIL ⇒ comb OFF + run-2 A/B removed | the §1.2 conditional inclusion law |
| R5 | P2 FEED-08l fresh-eyes review | $0, reading | verdict survives its 2-rung/oracle-form limits | FAIL ⇒ lane_carried demotion reverts OPEN | S1 regime premise; P1's interpretive frame |
| R6 | P7 n600 realized-parity row on ep650 | ~30-60 min | realized d_seg 0.0034±3e-4; inflate ≤20 min | Δ>+5e-4 ⇒ decode defect — FIX before ANY run | decode integrity for EVERY later row (per-stage byte-closes, AA selection, twin) |
| R7 | P4 #336 waterfill on ep650 | $0, 30-90 min | base+code ∈ [52,68] KB @ Δd_seg ≤+5e-5 | >+2e-4 ⇒ fallback rate row (crossing then needs the 0.0768 budget leg) | the rate leg of the crossing triple |
| R8 | P3 K=128 eigen-extraction finish (state on disk) | $0, ~min | ratio ∈ [0.04, 0.14] (pre-registered) | >0.5 ⇒ §2.3 reverts toward v1 framing | closes the carried chain-A branch; kill bands carry the LINK-0 ~35% instrument gap |
| R9 | P6 flip-share stability + γ recal + composed-share stability | $0, ~2-3 h | island share 55-70% stable | <35% ⇒ big-3 levers re-rank first | rung-1 shares assumption (§0.3) |
| R10 | T-1/2/3 injection tests + B1 build decision | $0, ~2 h (+~80 LOC if B1) | fires-when-should + silent-when-shouldn't | any failing trigger ⇒ that trigger cap-only | req-B arming; realizing the ep625 event saving at all (F3b) |
| R11 | P8 band ROI + P9 pose chain/q-sweep (smallest-q) + P10 exact-eval dry-run | ~1-2 h each | as drafted §7 | as drafted | byte-close legs; the pointer-row plan |

Pointer 0.19110 UNMOVED — this verdict is MEANS; the run and its §7 ROW remain the only success
definition.
