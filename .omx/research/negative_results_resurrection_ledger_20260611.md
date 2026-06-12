# Negative-results resurrection sweep — the corpus-wide ledger (2026-06-11)

**Authority:** `$0` read+classify audit (subagent `negative_results_resurrection_20260611`).
NO training, NO dispatch, NO MPS. Frontier **UNMOVED 0.19109982 [contest-CPU], 177,169 B, sha `b46897267…`
— ABOVE T_1, GOAL UNSATISFIED.** This is a classification ledger, not a pointer move. Every resurrection
below is a candidate to RE-TEST on a fixed/top-AIML impl, NOT a claim the old conclusion was wrong.

**Precedent (the operator's "negatives may be bugs" thesis):** THREE measurement-artifact false-negatives
surfaced THIS session — the EMA-shadow-lag ("d_seg frozen 0.505"), the GT-decode miscalibration, the
recipe-throttle "capacity wall". The training-loop subset is already re-classified in
`recipe_bug_lens_findings_reaudit_ledger_20260611.md` (agent `ae07ebd`) — **CONSUMED, not re-derived here.**
This sweep is EVERYTHING ELSE: codec/carrier/byte negatives, pose/seg negatives outside that loop, the
NeRV/HiNeRV/SNeRV fleet, the deferral/orphan ledgers, and the killed/FALSIFIED lane-registry rows.

## The 4-class taxonomy (the precedent classes)
- **(a) MEASUREMENT-ARTIFACT** — a bug (EMA-lag / GT-decode / recipe-throttle / short-run shadow) produced the negative. RE-TEST on the fixed harness.
- **(b) JANKY-PROTOTYPE** — verdict on a sketch/prototype-grade impl. RE-OPEN with top-AIML (ANTI-SIGNAL-LOSS).
- **(c) WRONG-OPERATING-POINT** — measured at wrong scale/config/resolution/epochs. RE-TEST at the right point.
- **(d) GENUINE-PARADIGM** — research-exhausted, recipe-independent, real. **STAYS CLOSED.**

---

## ★ TOP 5 HIGHEST-EV RESURRECTIONS (lead) ★

### R1 — Lever-C "score-native cheap-frame1 carrier FALSIFIED" (tasks #57/#61/#62) — **MEASUREMENT-ARTIFACT (a)**
**The negative:** `lever_c_viability_smoke_20260610T144739Z.md` + lane `lane_c_conv_pair_decoder_20260610`:
"conv per-pair frame1 decoder moves d_pose 114× but moves **d_seg by ZERO** — trained-decoder exact d_seg
**0.50732 == constant-frame floor 0.50692**". Verdict: cheap-frame1 carrier FALSIFIED across coordinate-INR
(#57/#61) AND conv (#62) families; the score-native crux DEFERRED.
**Specific suspect-reason (DECISIVE):** the smoke is "**8 pairs, 180 epochs, M5 Max CPU-torch, EMA-shadow
inference checkpoint**" (memo §1 verbatim). 8 pairs ÷ bs → ~6 steps/epoch; the canonical EMA shadow with
**constant decay, no warmup** stays bit-frozen near init → exact d_seg reads the **near-init 0.507 value**,
which IS the constant-frame floor. This is the EXACT signature the capstone EMA-shadow-lag memo
(`capstone_ema_shadow_lag_reverses_seg_wall_verdict_20260611.md`) proved is an artifact: LIVE d_seg
descends 0.507→0.041 while the **shadow freezes at 0.507**. The memo's own "What this REVERSES" section names
the ~0.505 floor as the artifact. The pose path DID move 114× — consistent with the memo's note that the
**slow pose path tracks the shadow but the fast seg path does not.** The d_seg=0.50732≈0.50692 "moved by
zero" is almost certainly the frozen shadow, not a real seg-blindness of the conv carrier.
**Caveat (honest):** config B (250k) read 0.695 (ABOVE the floor) — pure lag freezes at ~0.507, so config B
has a SECOND effect (overfit/instability), not pure lag. So R1 is "re-measure with LIVE d_seg + EMA warmup",
not "the carrier definitely works." The 114× pose win is real and survives.
**Resurrection EV: VERY HIGH.** This negative DEFERRED the entire score-native cheap-frame1 crux (the
sub-0.15 class-shift path the GOAL names). If d_seg actually descends live, the rate-cheap frame1 carrier
re-opens. **$0 first-step:** re-run the lever-C smoke driver with `use_ema_for_eval=False` (LIVE weights)
OR the warmup-EMA fix (`f771e6e00`), 8 pairs ≥120 epochs, report LIVE exact d_seg per epoch. Falsifiable
threshold: if LIVE d_seg descends below ~0.05 (as the capstone did), the FALSIFICATION is reversed.

### R2 — torch-EMA short-run negatives, fleet-wide (the EMA-lag bug class is NOT confined to MLX) — **MEASUREMENT-ARTIFACT (a)**
**The negative:** a class, not one lane — any torch trainer that (i) used `tac.training.EMA`, (ii) ran a
SHORT run (few pairs/few epochs → few optimizer steps), (iii) evaluated/exported the EMA shadow, and (iv)
produced a "d_seg ≈ 0.5 mean-field / plateau" or "seg-walled" negative.
**Specific suspect-reason:** the adversarial review `adversarial_review_post_ema_fix_picture_20260611.md`
claim #1b CONFIRMED (and I verified at source): `tac.training.EMA.update()` (`src/tac/training.py:528`)
**STILL uses constant `self.decay` with NO warmup** — the warmup fix landed ONLY in the MLX
`_CapstoneWeightEMA`, not the canonical torch EMA. `decay_from_total_steps` (line 538) is OPT-IN and most
callers do not set it. So `self_compress`, `segmap_renderer`, `psd_lumaskip_renderer`,
`joint_scorer_aware_training`, `nerv_mask_codec`, and any short torch smoke that exported/eval'd the shadow
carry the SAME freeze. The B1 "clean PR95 mean-fields ≈0.505" (read at 33% epochs) and any "d_seg plateaus
~0.5" torch smoke are suspects.
**Resurrection EV: HIGH (broad).** This is the highest-leverage SINGLE fix: it potentially un-falsifies a
whole tier of short-run torch d_seg negatives at once. **$0 first-step:** (1) land the warmup in
`tac.training.EMA` (mirror `_CapstoneWeightEMA.effective_decay`) + the 2 NO-FAKE guard tests; (2) grep the
torch-trainer negatives for `use_ema`/short-run + re-read LIVE d_seg. Do NOT blanket-resurrect — re-test
each on the fixed EMA. (Sister of the recipe-bug ledger's TIER-1, but at the ORTHOGONAL torch surface the
recipe-fix did not touch.)

### R3 — apogee int4/int7 + lossy-int4 family (QAT/LSQ NEVER exact-CUDA-tested) — **WRONG-OPERATING-POINT (c) / proxy-only**
**The negative:** `lane_apogee_int4` (predicted [0.155,0.180], never CUDA), `lane_apogee_int7`
DEFERRED-pending-byte-aligned-packer, and the `lossy_falsification_scope_audit_20260508` family
(naive-int4 37% proxy rel_err, int4-QAT 28% MPS, per-channel 30%, GPTQ 46%, AWQ 37%).
**Specific suspect-reason:** the scope audit ITSELF says the honest verdict is "**measured config not
dispatchable, NOT a family kill**" — every row is CPU/MPS/PROXY rel_err, ZERO exact-CUDA. QAT/LSQ/per-channel
were measured as PROXY rel_err, never as exact d_seg through the frozen scorer. Per the CLAUDE.md QAT
pipeline non-negotiable, the canonical low-bit recipe (QAT + per-channel + LSQ step-size + outlier handling)
was never run to exact eval. This is the prior TOP-5 `top_5_apogee_int4_qat` resurrection probe (STILL
PARTIAL). It is a rate-axis lever, and rate IS 62% of the score.
**Resurrection EV: MEDIUM-HIGH** but gated by the d_seg-binding finding (see DO-NOT below): a smaller
quantized decoder only wins if it HOLDS the d_seg basin. **$0 first-step:** run an int4-QAT-on-the-frontier
exact d_seg smoke (LIVE weights, post-EMA-fix) — does QAT of the 177KB frontier decoder hold d_seg ≈5.6e-4
at ~9× fewer bytes? If yes → ONE paired exact eval. If d_seg blows up (as the `frontier_decoder_qat_recovery`
+0.056 kill-gate suggests it might), it stays deferred. NOTE: `frontier_decoder_qat_recovery_20260610`
already fired a +0.056 local kill-gate — so this resurrection must use the CANONICAL QAT recipe (per-channel
+ LSQ + outlier), not naive PTQ, to be a fair re-test.

### R4 — rate/entropy recode on the CURRENT 177,169 B frontier ("rate EXHAUSTED" claim — VERIFY currency) — **partly (c), partly (d)**
**The negative:** the GOAL scoreboard says frozen-byte rate is "EXHAUSTED at lossless / 7.999 bits/byte";
the `inflate_time_procedural_rate_lever_inventory_20260611` ledger confirms decoder section "at lossless
floor", latent "<300 B headroom".
**Specific suspect-reason:** this is MOSTLY genuine (the 7.999 bits/byte entropy floor is recipe-independent
— see DO-NOT R-DN1). BUT the adversarial review (lever #2) flags it as worth a CURRENCY re-check: the
frontier is only **0.0011 above T_1**, and 25/D = 6.66e-7 ΔS/byte → ~1,650 B at constant distortion crosses
T_1. The "exhausted" claim was made on a PRIOR frontier member; verify it on the CURRENT `b46897267` member-x.
**Resurrection EV: MEDIUM** (small, but cheapest sub-0.19 if ANY headroom remains). **$0 first-step:** re-run
the latent/decoder entropy probe on the current member-x; if any coder beats the current section by >~1,650 B,
byte-close + ONE paired exact eval. (This is largely a currency re-confirmation, not a true resurrection —
listed because the margin to T_1 is tiny.)

### R5 — pre-PR95 "compute-bound, not wrong" representation paths (Plate HRR / modern-Hopfield attractor latents) — **JANKY-PROTOTYPE / never-built (b)**
**The negative:** `ancient_elder_era_6_abandoned_paths_20260513.md` — Hopfield/HRR/RAAM/Smolensky filed as
"great ideas, no compute" (1972–1995); never built as contest substrates.
**Specific suspect-reason:** these were NEVER implemented at all (not even prototype-grade) — they are
abandoned-by-history, not falsified-by-measurement. AP-1 (Plate HRR for structured `ego⊛pose + scene⊛mask`
per-pair latents) and AP-2 (modern-Hopfield attractor decoder for noise-tolerant aggressive bit reduction)
are HNeRV-family-ORTHOGONAL and target the latent-noise-tolerance axis (the rate lever). Ramsauer 2020 proved
modern Hopfield = transformer attention, so these are now buildable with standard MLX/torch primitives.
**Resurrection EV: MEDIUM (speculative, high-variance)** — a genuine class-shift bet for the sub-0.15 horizon,
not a near-term pointer-mover. **$0 first-step:** a 1-pair MLX overfit smoke of an HRR-bound per-pair latent
vs the current 28-d latent — does HRR binding reach the same render fidelity at fewer effective bits? Cheap
falsifiable design probe before any build.

---

## FULL LEDGER (all classified negatives outside the training-loop subset)

| # | Negative | Class | Specific suspect-reason | Resurrection EV / $0 first-step |
|---|---|---|---|---|
| R1 | Lever-C cheap-frame1 carrier FALSIFIED (#57/#61/#62, `lane_c_conv_pair_decoder`) | **(a) ARTIFACT** | EMA-shadow eval on 8-pair/180ep short run → d_seg frozen at 0.507 ≈ constant floor (the proven lag signature) | VERY HIGH / re-run with LIVE d_seg or warmup-EMA |
| R2 | torch-EMA short-run "d_seg≈0.5 / seg-walled" negatives (fleet) | **(a) ARTIFACT** | `tac.training.EMA:528` still constant-decay no-warmup (the MLX fix didn't reach torch) | HIGH / land torch warmup-EMA, re-read LIVE d_seg per lane |
| R3 | apogee int4/int7 + lossy-int4 QAT/LSQ family | **(c) WRONG-OP** | scope audit says "config not dispatchable, NOT family kill"; QAT/LSQ never exact-CUDA, proxy rel_err only | MED-HIGH (d_seg-gated) / canonical-QAT exact d_seg smoke on frontier decoder |
| R4 | "rate EXHAUSTED at lossless" on current frontier | mostly **(d)**, currency-(c) | claim made on prior member; 0.0011 to T_1 makes a small recode decisive | MED / entropy probe on current member-x |
| R5 | Plate-HRR / modern-Hopfield latent paths | **(b) never-built** | abandoned-by-history (no compute in 1995), never implemented; Ramsauer 2020 makes buildable | MED-speculative / 1-pair HRR-latent fidelity smoke |
| R6 | `lane_gp_v4` polynomial pose-basis-fit KILLED | **(d) GENUINE / (b)-residual** | Council 4-round unanimous KILL; reactivation already pinned = non-poly fit (DCT/B-spline/wavelet) RMSE<0.5. The poly fit is genuinely infeasible; a non-poly fit is a DIFFERENT lane, not a resurrection of the poly one | LOW-as-poly / the non-poly successor is a NEW lane, not this kill. NOTE: pose is now "stored GT + FiLM" (cheaper) → pose-basis-fit is largely OBSOLETED, not resurrected |
| R7 | `lane_7_psd_killed` / `lane_psd` PSD renderer | **(b) JANKY pre-spend KILL** | KILL-DEFER before ANY GPU spend (10/10 reject); $0 spent. A pre-empirical council reject, never measured. Reactivation pinned = PSD-LumaSkip variant | LOW-MED / only if a measured PSD smoke beats baseline; pre-spend kills are weakest evidence but PSD is an old arch, low priority vs HNeRV-class |
| R8 | `lane_overfit_known_video` (5 approaches "all duplicate") | **(d) duplicate, not negative** | not a paradigm negative — the 5 approaches were folded into live lanes. No resurrection (the work continued under other names) | N/A — already absorbed |
| R9 | `lane_owv3_0120_arith_masks` AMRC DEFERRED | **(d) GENUINE** | AMRC=1,082,649 B vs AV1=421,483 B (2.6× WORSE) — measured byte loss, recipe-independent. Mask-only slot is also dominated per HNeRV-parity L5 | LOW / genuine byte negative; stays deferred |
| R10 | STC clean-source mask-delta (`lane_stc_clean_source_filler_delta`) | **(d) GENUINE (freshly re-tested)** | RE-TESTED this session: uniform-cost STC self-syndrome 2.4–2.6× LARGER than brotli(mask-delta) at contest sparsity ρ≤0.10. Symposium ≥5% bar NOT met. This was a janky-prototype before (dense-argmax), now properly re-tested on sparse delta → still loses | LOW / DEFER stands; reactivation = detector-informed cost-map (CC#2) only |
| R11 | `lane_nscs06_carmack_hotz_strip_everything` (105.15 vs [0.10,0.20]) | **(a/b) ARTIFACT+PROTOTYPE** | 700× band-miss = SegNet collapse on lifted-trainer form; NSCS06 v6→v7 already showed 44% recovery via cargo-cult-unwind (the canonical Catalog #315 anchor). The negative is implementation-level | MED / but this is squarely the recipe-bug ledger's domain (lifted-trainer); defer to that campaign |
| R12 | HiNeRV parse-back collapse (12k birth → 2 wins) | **NOT a kill — LIVE investigation** | `hinerv_collapse_axis_falsification_chain` is actively localizing the cause (export build-path / sidecar wrap, Case B confirmed). This is a mechanism-in-progress, not a closed negative to resurrect | N/A — already active; do not duplicate |
| R13 | `lane_frontier_seg_repair_pool_20260610` (3 carriers falsified) | **(c) WRONG-CARRIER, paradigm-intact** | DEFER-pending-NEW-carrier; the 3 carriers (appearance-gap, gradient-sign, shared-field) each measured-failed but the seg-repair PARADIGM is intact. Reactivation = new carrier | MED / the live frontier seg-axis lane; new-carrier search is the reactivation (sister to R1) |
| R14 | `lane_t1_s12_lossless_stack` (cross-pair MI=0, k-means LOSS) | **(d) GENUINE** | measured MI=0 cross-pair (no exploitable redundancy), k-means NET byte loss — recipe-independent information-theoretic negatives | LOW / genuine; stays closed |
| R15 | int4 lossy-coarsening 0.3517 [contest-CUDA] | **(d) GENUINE** | CUDA-CONFIRMED negative (not proxy); lossy weight-coarsening loses to lossless. Recipe-independent | LOW / stays closed (see DO-NOT) |

---

## DO-NOT-RESURRECT (genuine negatives — over-reactivation = signal loss)

These are research-exhausted, recipe-independent, and STAY CLOSED per the symmetric NO-FAKE discipline:

- **R-DN1 — the rate entropy-floor (7.999 bits/byte, 593/600 selector argmin exhausted, frozen-byte $0-transform exhaustion).** Information-theoretic, measured on the frozen archive bytes. R4 is only a currency re-check, not a reopening of the floor itself.
- **R-DN2 — the lossy-coarsening 0.3517 [contest-CUDA negative]** (R15) + the decoder-axis c1/c2/c3 lossy verdicts. CUDA-CONFIRMED, no recipe dependency. Do NOT redispatch.
- **R-DN3 — MPS-as-noise** (95.5% argmax-corruption receipt). Structural, not a measurement to retry.
- **R-DN4 — STC clean-source uniform-cost** (R10): freshly RE-TESTED this session on the CORRECT sparse-delta source; still 2.4–2.6× worse than brotli. The janky-prototype was already re-opened and re-closed properly.
- **R-DN5 — cross-pair MI=0 / k-means net-loss** (R14, T1/T8 KILLs): measured information-theoretic negatives.
- **R-DN6 — AMRC arith-mask 2.6× byte loss** (R9): measured byte negative.
- **R-DN7 — the per-subband range-coder DEFER (Z8)**: per the 2026-05-31 grand council, a GENUINE local optimum (global coder −3.9%/−0.5% WORSE than per-subband split; v2 within 5–10% of Shannon floor). The rate lever moved UPSTREAM to the per-subband delta water-fill, not to a better coder.
- **R-DN8 — UNIWARD 5th/6th/7th-order nulls on vanilla cover-modification**: PARADIGM-NULL on the plain surface (IMPLEMENTATION-level only for entropy-coded sidecar — already pinned as that narrow reactivation by the 2026-05-29 council). Do not re-litigate the vanilla surface.
- **R-DN9 — custom forward conv kernel (any language)**: `custom_conv_kernel_measured_dead_end_20260611` — bandwidth-bound (3.07 FLOP/byte), MLX already ~3× off the 450 GB/s floor; language is MOOT. Genuine-paradigm negative.
- **R-DN10 — smaller-basis-by-rate (Cool-Chic capacity sweep)**: REFUTED this session at the d_seg axis (L2 latent-heavy saturates ~0.014, ~25× the frontier; L3 with MORE latent got WORSE = capacity signature). BUT NOTE: this REFUTAL is itself entangled with R1/R2 (was its d_seg read post-EMA-fix? the memo says EMA vs live agree ≤2e-4, so it IS clean) — the smaller-basis REFUTAL is the ONE capacity negative that was re-measured AFTER the EMA fix, so it holds. Do not resurrect the Cool-Chic capacity knob; a *qualitatively different* smaller synth (R1's conv carrier, R5's HRR) is the open residual, not bigger Cool-Chic grids.

---

## Honest bottom line

- **Genuinely deserve a second life (re-test on fixed harness):** R1 (lever-C cheap-frame1 — the strongest; an
  EMA-shadow artifact deferred a sub-0.15 crux), R2 (the torch-EMA-lag fleet class — broadest single fix), R3
  (apogee/lossy-int4 QAT — never exact-tested, but d_seg-gated), R5 (HRR/Hopfield — never-built class-shift bet).
- **Fold into the n600 campaign?** No — these are $0 LOCAL re-tests that must run BEFORE any paid spend (they
  could un-falsify the cheaper carrier and make a frontier-class n600 unnecessary). R1+R2 are the gate: re-read
  LIVE d_seg on the cheap-frame1 carrier under the warmup-EMA before sizing any paid train.
- **Run as $0 smokes NOW:** R1 (re-run lever-C with LIVE/warmup eval), R2 (land torch warmup-EMA + re-read),
  R4 (frontier member-x entropy currency check).
- **Real negatives that stay closed:** R-DN1…R-DN10 — the rate floor, the CUDA-confirmed lossy negatives,
  MPS-noise, the freshly-re-tested STC, the MI=0/k-means/AMRC byte negatives, the range-coder local-optimum,
  the vanilla-UNIWARD null, the bandwidth-bound conv kernel, and the post-EMA-fix smaller-basis REFUTAL.

**The single highest-EV action:** land the torch-EMA warmup (R2) + re-run the lever-C cheap-frame1 smoke with
LIVE/warmup d_seg eval (R1). If the conv frame1 carrier's d_seg actually descends (as the capstone's did once
the shadow-lag was fixed), the score-native cheap-frame1 crux — DEFERRED on an EMA-shadow artifact — re-opens
as a live sub-0.15 path, and it is a $0 re-test, not a paid spend.
