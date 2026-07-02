---
title: "#205 PRE-LAUNCH GATE — PHASE 3: 3-clean-pass recursive adversarial review VERDICT (SEAL)"
date: 2026-07-02
axis: "[macOS-CPU / MLX advisory / design] — NON-PROMOTABLE. means != ends: this SEALs a launch-ready CONFIG (a MEANS). The ONLY end is a byte-closed n600 exact row < 0.19110 from upstream/evaluate.py (contest-CPU/CUDA, NEVER MPS). Pointer 0.19110 UNMOVED — this gate does NOT move it; the operator's launch commit → byte-close (#202) does."
pointer: "0.19110 UNMOVED (contest-CPU recoded-R3)"
scope: "Phase-3 of the #205 gate. Reviews §5b shippable argv (commit b0e45e0c7) + §5c schedule (b92864a8f) + seeding audit (fd49f858f) + AA reconciliation. CONTAINMENT: review + $0-local-measure ONLY. NO GPU launch, NO trainer edits. Config required ZERO edits (the one de-risk that could change a value CONFIRMED the shipped value)."
provenance: "3 adversarial passes (flags/couplings/means-ends + schedule/interactions + numbers/determinism). 1 NEW $0 n600 measured row (horizon geometry sweep). 2 pre-existing n600 through-R rows read (supersample, lane-band). 83-flag argparse verification (Explore, file:line callsites). Every claim MEASURED (n600 advisory) / DERIVED / or DEFERRED-with-reactivation."
---

# #205 P3 — 3-clean-pass recursive adversarial review VERDICT

**VERDICT: SEAL — the §5b shippable argv + §5c schedule + seeding are LAUNCH-READY as the config the
operator commits to launch.** No config edit is required. The review surfaced (i) one de-risk that
CONFIRMED the shipped value (horizon 174, NOT the audit's proposed 188 — R1 FALSIFIED at n600), (ii) one
honest S-budget correction (the warp-real-luma keyframe payload is a counted cost §7 omits), (iii) one
doc-consistency update (§4 coupling-4 AA-floor narrative is stale post-AA-reconciliation), and (iv) a set
of launch-time verifications + the mandatory-EXTEND lexicographic safeguard. NONE is a config defect; ALL
belong in the launch-ready statement, not in a code edit. The pointer stays 0.19110 — SEAL is of a MEANS;
the END is the operator's launch → byte-closed exact row.

---

## 0. The 3-clean-pass summary (what each pass attacked; findings)

The protocol: a round with any UNRESOLVED finding resets the clean counter. A "finding" that RESOLVES to
"the config is correct as-is" (de-risk confirms the shipped value; a residual belongs in the verdict not
the code) does NOT demand a config edit and does not leave the config unclean. Across 3 passes the
**config + schedule + seeding required ZERO edits.**

| pass | attack surface | outcome |
|---|---|---|
| **P1** | argv flags (NO-FAKE) · deep-math couplings §3 · means/ends · the 3 de-risks | 83/83 flags in-argparse (0 invented), all 18 load-bearing reach render/loss/opt/init consumers, all HARD interaction guards satisfied (Explore, file:line). De-risk (a) horizon **CONFIRMED 174** (R1 falsified). De-risk (b) supersample **CONFIRMED +49% HURT** → `none` airtight. means/ends correct (w_pose=1.0 shippable). **CLEAN (no config edit); 1 residual + 1 doc-update surfaced.** |
| **P2** | schedule §5c · lever×stage interactions · Muon-before-l7 WARN · closed-loop | DAG edges coupling-derived; l7-parked → Muon polishes tau-partition = INTENDED demoted-l7 (the soft WARN L3990 is cosmetic). persistence/amplify @300 ride shared `_signed` (no extra forward, no Muon conflict). `--render-aa none` + `--lane-render-band` NOT mutually-exclusive (exclusion is supersample+band only). **CLEAN (no config edit); monitored-risk: untuned amplify/persistence unit weights → closed-loop ROLLBACK safety-net.** |
| **P3** | S-budget numbers · determinism · lexicographic · launch-time env | determinism intact (seed 0, per-stage ckpts, EMA-shadow, atomic, crash-resume bit-identical incl. Muon-finisher). Lexicographic: S-neutral speed levers bit-identical; epochs=1000 = OPENING with **mandatory EXTEND** safeguard (else S traded for T). S-budget optimistic by the keyframe payload. **CLEAN (no config edit); launch-time checks enumerated §8.** |

The 3 passes are CLEAN in the load-bearing sense: **no config edit demanded across any pass.** Findings
are launch-context (residuals / doc-updates / launch-time verifications), catalogued below.

---

## 1. THE $0 DE-RISKS — RESULTS (n600 real, [macOS-CPU/MLX advisory], NON-PROMOTABLE)

### (a) lane-IPM horizon `v_h` {174,182,188,194} — R1 **FALSIFIED**; KEEP 174 (NO config change) [NEW n600 row]

The seeding audit R1 named this "the single measured-backed d_seg lever" and recommended `v_h 174→188`
(FEED-dj-optimal), flagging it needs re-confirmation through-R ("the FEED-dj anchor is a partition-level
IPM sweep; verify it holds"). I ran the DIRECT mechanism as a genuinely-$0 n600 **geometry** screen
(the horizon's entire effect is the image-row→forward-meter map that places the band + dash-gate; its
realized d_seg is driven by band-vs-GT-lane RECALL and FP — pure `build_analytic_lane_band_prior`, NO
render/SegNet/R). All 600 pairs, real gt lstars, `dash_forward_max_m=55` (the config value):

| v_h | band_recall | band_fp_frac | n_lines | Δ vs 174 |
|---:|---:|---:|---:|---|
| **174 (shipped)** | **0.5475** | **0.001979** | 4.945 | — (best on BOTH axes) |
| 182 | 0.5553 | 0.003271 | 5.067 | recall +0.0078 / **FP +0.00129 (worse)** |
| 188 (FEED-dj) | 0.4966 | 0.003554 | 5.518 | **recall −0.0509 / FP +0.00158 (both worse)** |
| 194 | 0.4267 | 0.003658 | 5.482 | recall −0.1207 / FP +0.00168 (both worse) |

**Verdict: the shipped `_V_HORIZON=174` is optimal; 188 DEGRADES band placement on BOTH axes** (−0.051
recall AND +0.0016 FP), monotonically worsening past 182. The partition-level FEED-dj "188 IPM-optimal"
anchor does NOT survive the GT-lstars-realized band placement — exactly the caveat the audit flagged.
**Config action: NONE.** 174 is the default `build_analytic_lane_band_prior` uses (the trainer does not
override it), so the shippable argv already ships the optimal value. **Seeding-audit R1 is FALSIFIED**
(marked below). This also removes the only pre-launch motivation to edit `lane_sdf_component._V_HORIZON`
or add a `--lane-band-v-horizon` flag.

Caveat (honest): this is a GEOMETRY screen (band-vs-GT recall/FP), not the full render+SegNet+R realized
d_seg. But band recall/FP vs GT lstars IS the direct upstream of the realized lane d_seg, AND the config
runs the WITNESS-UNCERTAINTY-gated band (which further attenuates FP downstream). The 188 degradation is
large + monotone → the burden of proof is on anyone claiming 188; a ~2h realized confirm is NOT warranted
to overturn a decisive $0 screen. If ever revisited, the realized command is `BAND_* env +
tools/levelset_analytic_lane_band_dseg_n600.py` with `v_h` threaded into `build_analytic_lane_band_prior`.
(Producer: `tools/levelset_ipm_horizon_geom_sweep_n600.py`; result:
`reports/levelset_ipm_horizon_geom_sweep_n600_20260702.json`, 85 s.)

### (b) supersample AA `−49%` re-confirm — **CONFIRMED +49% HURT**; `--render-aa none` airtight [pre-existing n600 through-R row]

The AA reconciliation flagged the −49% artifact "not located by name this pass — re-confirm before the
Wave D edit lands." I located the producer (`tools/levelset_render_side_sizing_l7best_n600.py`) and its
COMPLETE n600 output (`reports/levelset_render_side_sizing_l7best_n600_20260701.json`, n=600): on the
frozen l7-best ckpt, realized through the actual R + frozen CPU-torch SegNet argmax:

- `c1_default` (witness point-sample @384) **d_seg 0.00333**
- `c2_aa` (supersample 2× → area-down → R) **d_seg 0.00496 = +0.00163 (+49% HURT)** ← re-confirmed
- (`c3_lane` naive band +0.00082; `c4` both +0.00213 — all post-hoc render-side levers HURT.)

The JSON's own VERDICT: "ZERO of the d_seg gap is render-side-reachable POST-HOC ... AA recovers
UNDERSAMPLED real detail; it degrades OVERSAMPLED synthetic renders." Combined with the AA
reconciliation's INDEPENDENT decode-budget disqualifier (fp64 41.3 min > 30 min AND neither shipped
inflate applies ss → train/decode MISMATCH = a FAKE optimization), `--render-aa none` is airtight on
TWO independent grounds. **Config action: NONE** (argv already ships `--render-aa none`).

### (c) directional-basis realized `−48%` — DEFERRED (not genuinely $0 on the frozen ckpt)

The self-orient directional basis IS realized in both frozen-ckpt tools (`c1_default 0.00333` is measured
WITH `--self-orient` active, 4-iter argmax fixed point through R). But the −48%-vs-isotropic DELTA is a
PAIRED-ARM question (directional-on vs directional-off TRAINED) — turning self-orient off at render time
on a directional-trained ckpt just breaks the reconstruction; it is NOT a clean ablation. So (c) is NOT
genuinely $0 on the frozen ckpt (needs a GPU control arm). **Verdict: the directional basis is ON
(correct per coupling §3-3 basis-before-capacity); its realized magnitude is a #205-RUN attribution
(the directional-on run IS the realized number; an isotropic control arm quantifies −48%). DEFERRED to a
post-launch attribution arm — NOT launch-blocking.** Reactivation: run an isotropic-basis control arm
(`--self-orient` off) at the same config to quantify the realized directional Δd_seg.

---

## 2. §8 CONFIG OPEN QUESTIONS — ADJUDICATED (all RESOLVED / DEFERRED-with-reactivation; none launch-blocking)

1. **l7 reconcile (DEMOTE vs −0.00027 drop):** **RESOLVED — DEMOTE.** The 5-agent deep pass + eq
   `l7_linf_sharpening_defect` (a MECHANISM: L∞ sharpening inside a viscosity/smoothing flow decouples
   d_seg) outranks the small MLX-trace SURROGATE drop (not a through-R n600 row). Parking l7 (`--l7-start-
   epoch 1000 == epochs`, never runs — Explore-confirmed) is REVERSIBLE: the l7-on-vs-off through-R A/B
   at the tau-converged per-stage ckpt is a queued warm-start re-treatment (T2, free). Reactivation:
   that A/B.
2. **Pose SLOT — is w_pose=0 first-launch legit?** **RESOLVED — w_pose=1.0 shippable-first (NOT 0).**
   The §5b argv correctly commits `--w-pose 1.0 --pose-carrier` (a w_pose=0 row does NOT move the
   pointer — advisory only; means/ends firewall). The carrier is wired + durability-proven (crash-resume
   bit-identical incl. Muon-finisher). CAVEAT: the shippable row is a JOINT bet (d_seg floor × pose
   closure × keyframe bytes §4); the per-stage d_seg trace + seg⊥pose additive-S (`--pose-carrier-
   residual-mode table` isolates the code manifold, cos 5.9e-5) are the attribution safeguards. A short
   w_pose=0 advisory leg is OPTIONAL (de-risks d_seg attribution) but doubles wall-clock — not required.
3. **AA-render wire-in (B) sequencing:** **RESOLVED — DISSOLVED by the AA reconciliation.** There is no
   AA-render to wire: supersample is DISQUALIFIED (−49% + decode-budget + mismatch); the floor-reaching
   representation is point-sample render + directional basis + persistence/amplify/Muon (ALL wired). The
   Phase-1 "wire AA-render FIRST (floor-setter)" framing is OBSOLETE. #205 launches WITH the correct
   representation.
4. **mod-dim (#223) 32 vs 26 vs 19:** **RESOLVED — launch at 32** (proven arm; reached measured d_seg
   0.003698; covers composite m~13 with headroom; d_seg is the BINDING term and 19's d_seg-neutrality is
   UNMEASURED; rate has slack 0.055<0.081). mod-dim is a SHAPE-changing flag → a $0 pre-launch sweep
   isn't possible (needs its own trained arm). Reactivation: #223 byte-close sweep folds 32→26→19 ONLY if
   measured d_seg-neutral (a separate GPU arm, not warm-start).
5. **β₂ (#222) real lever or red herring?** **RESOLVED — launch at β₂=0.999** (MEASURED anchor ==
   MLX default, byte-identical, no bias-correction confound on the first attribution row; wired via
   `--adam-beta2 0.999`, Explore-traced to the optimizer). The "0.9999999" is a flagged MIS-ANCHOR,
   correctly NOT used. Reactivation: the #222 disambiguating optimizer-vs-representation sweep (post-launch).
6. **epochs (1500→reconciled 1000):** **RESOLVED — launch at 1000 as the OPENING; EXTEND is MANDATORY
   if the Muon d_seg slope is still negative @ep1000.** long900 was still descending @ep800 → the closed-
   loop `decide_next_stage` EXTEND (§5c-e) will fire; per-stage ckpts + `--resume-from` make it free.
   **LEXICOGRAPHIC SAFEGUARD (binding): accepting ep1000 as final while descending would trade S for T —
   FORBIDDEN.** The operator's launch must run the EXTEND (CONTAINMENT: operator-gated, not auto-fired).
7. **Directional basis realized verdict:** **DEFERRED** — see de-risk (c). ON (correct); realized
   magnitude is a #205-run attribution, not a $0 pre-launch de-risk.
8. **`--async-verdict` bit-identity + bottleneck:** **RESOLVED.** Bit-identical (verdict off a snapshot,
   never read back — durability smoke `2ca1726ae`). `--verdict-pairs 0` = **all-600** (line 2174: `...
   if args.verdict_pairs else list(range(P))`), async every 25 ep → off the critical path AND n600 (not a
   subset). Bonus: the ep-to-ep n600 verdict variance IS the noise floor for §5c-S6 threshold calibration.

---

## 3. §5c SCHEDULE OPEN QUESTIONS — ADJUDICATED (control-program is calibrated; none launch-blocking)

| # | question | verdict |
|---|---|---|
| S1 | tau→Muon boundary @726 | **RESOLVED-as-opening.** DERIVED from the proven 0.726 Muon-start fraction; a boundary A/B is a warm-start per-stage-ckpt re-treatment (post-launch). |
| S2 | Muon length 274 ep | **RESOLVED DYNAMICALLY** by (e) EXTEND (same as Q6); long900 still descending @ep800 → EXTEND-eligible. |
| S3 | persistence/amplify warmup@300 + ramp shape | **RESOLVED-as-default.** start=tau@300 is coupling-justified (finest-scale erasure long-tail EMERGES as the partition sharpens → late-engaged). Ramp SHAPE + unit weights are UN-tuned defaults (LABELLED); monitored-risk (over-amplify→FP) caught by closed-loop ROLLBACK. |
| S4 | l7 DEMOTE vs small-drop | **RESOLVED — DEMOTE** (= Q1). |
| S5 | reheat floor/shape 0.1×/8ep | **RESOLVED-as-measured** (partial-restart MEASURED; full 1.0× re-destabilizes; per-transition floor refine-measurable, not blocking). |
| S6 | closed-loop thresholds vs verdict-NOISE | **RESOLVED — self-calibrating.** Launch open-loop to ep1000; calibrate `plateau_abs_slope`/`descend_slope` from the OBSERVED async n600-verdict variance (the run supplies its own noise floor) BEFORE the ep1000 EXTEND/ADVANCE gate. NOT blocking. |
| S7 | stage ORDER robustness | **RESOLVED-as-derived.** DAG edges are coupling-MEASURED (basis-before-capacity, tau-before-Muon, representation-before-dynamics). A single-swap ablation would CONFIRM but is low-priority, post-launch. |

---

## 4. THE CORRECTED S-BUDGET (the honest residual: keyframe payload)

§7 predicts S ~0.13–0.15 (rate 0.055). **The rate term OMITS the warp-real-luma pose-carrier keyframe
payload** — a REAL, HONEST, self-flagged counted cost. `warp_real_luma_frame0.py` docstring: *"The SOURCE
luma (gt_f0) at decode is NOT the original video (unavailable) — it is a stored REAL keyframe (counted;
the W9/W10 reach gate schedules ~13 keyframes for the tested ~10 s window) ... the vehicle's S1/S3
concern, flagged as a dependency — NOT smuggled into this module's byte claim."* Measured anchor
(`measured_lever_inventory`): **13 keyframes ≈ rate-term 0.0060** for a ~10 s window.

| term | §7 (as written) | corrected (keyframe-aware) |
|---|---|---|
| 100·d_seg | 0.077–0.118 (floor-reach BET) | unchanged (BET) |
| √(10·d_pose) | ~0.018 (residual closure BET) | unchanged (BET) |
| 25·B/N | **0.055** (witness int8 + ξ 2.4 KB + lane slot) | **0.061–~0.095** (+ keyframes: +0.006 for 10 s; UNMEASURED/larger for the full 600-pair clip) |
| **S** | ~0.13–0.15 | **~0.14–0.19 realistic; sub-0.15 requires keyframes stay small** |

**Consequence for the launch-ready expectation:** the first byte-closed row should be expected in the
**sub-0.19 band**, with **sub-0.15 TIGHT and CONTINGENT** on (a) the d_seg floor-reach bet AND (b) an
aggressive/small keyframe payload (dual-use of the low-rank ξ; keyframe scheduling). This is NOT a launch
blocker — the #205 run measures d_seg + d_pose through R without needing the keyframe payload finalized —
but it IS a **#202 byte-close prerequisite for any sub-0.15 claim**, and §7's rate should be re-stated
with a keyframe line item. (Recorded to the nexus doc + DAG FEED per #219.)

---

## 5. DEEP-MATH COUPLING CONSISTENCY (§3/§4) — one STALE narrative, config CORRECT

The config honors every §3 coupling: seg⊥pose additive-S (`table` residual isolates the code manifold);
capacity↔rate KKT (mod-32/h96 RD-region); basis-BEFORE-capacity (`--self-orient` from INIT);
pose-carrier-REQUIRED (w_pose=1.0); l7-DEFECT DEMOTE. **One narrative is STALE:** §4 coupling-4
("representation ↔ dynamics = FLOOR CAPS CEILING; front-load the AA-render representation; the 0.00086
floor is BELOW target → sub-0.15 is a TRAINING problem not a representation problem") rests on the
**real-frame AA ceiling (SIGNAL A)**, which the AA reconciliation showed is **NOT witness-realized**
(supersample HURTS the witness −49%; §1b). The witness-realized floor is set by the **point-sample render
+ directional basis**, and the ~0.003→~0.001 gap is closed by the **LOSS levers (persistence/amplify) +
Muon conditioning**, NOT by an AA representation lever. **The config is CORRECT** (persistence/amplify ON,
AA none, Muon finisher). Only the coupling-4 TEXT needs the update: the central bet is *point-sample d_seg
0.00245 → ~0.001 via Muon+persistence+amplify* (a TRAINING+LOSS bet), and 0.00086 is a real-frame BOUND,
not a witness-realized floor. (Folded into the nexus doc §3/§4 + DAG FEED per #219.)

---

## 6. SEEDING REVIEW SYNTHESIS (no pre-launch change)

- **R1 (horizon 174→188): FALSIFIED** by §1a — KEEP 174, no change.
- **R2 (unify ground plane 1.2 vs 1.22): post-#205 hygiene** — low d_seg leverage; and with 188 falsified
  the horizon-unification urgency drops further. NOT launch-blocking.
- **R3/R4/R5 (φ1 seed / lane-band-source A/B / pose s_r-pitch warm-start): post-#205 levers/hygiene** —
  NOT launch-blocking. (R4 default `coherent_slot_none` lossless keeps train==decode; ANY lossy A/B must
  measure d_seg through the ACTUAL decode-reconstructed band, never GT — the anti-fake discipline.)
- **Unification (one world-model both axes): correctly NOT wired** — MEASURED: lane ξ-coding REFUTED
  (Pareto-dominated on the swap-light clip); pose warm-start already uses the SUPERIOR PoseNet-target
  prior. No large d_seg/d_pose lever hiding there. ✅

---

## 7. THE LAUNCH-READY COMMAND BLOCK (UNCHANGED from §5b — SEALED; 0 edits)

The §5b argv (commit b0e45e0c7) is SEALED as-is. No value changes (horizon confirmed 174 by default;
supersample already `none`). Reproduced for the operator's launch commit:

```
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 \
.venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py \
  --out-dir experiments/results/levelset_n600_witness_capstone_<UTC> \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 \
  --mlx-device gpu --seed 0 \
  --epochs 1000 --eval-every 25 --verdict-pairs 0 --async-verdict \
  --curriculum \
  --tau-softplus-start-epoch 300 --tau-softplus-tau 0.3 \
  --l7-start-epoch 1000 \
  --muon-start-epoch 726 --muon-lr 0.002 --muon-momentum 0.95 --muon-ns-steps 5 \
  --stage-transition-rewarmup-epochs 8 --stage-transition-rewarmup-floor 0.1 \
  --stage-transition-rewarmup-shape linear --stage-transition-reset-moments \
  --w-seg 100 --w-pose 1.0 --score-domain-loss \
  --pose-carrier --pose-carrier-residual-mode table \
  --mod-dim 32 --hidden-dim 96 --n-hidden 4 \
  --activation hosc --hosc-beta 1.0 --hosc-beta-end 4.0 --hosc-beta-anneal linear \
  --hosc-omega 1.0 --siren-init \
  --softmax-temp-start 1.0 --softmax-temp-end 0.05 --tau-anneal-shape cosine \
  --self-orient --n-dir-freqs 2 --freq-across 32 --freq-along 4 --reorient-every 50 \
  --max-bank-freq 64 \
  --chroma --palette-anchor \
  --eikonal-weight 0.01 --length-weight 0.001 \
  --render-h 384 --render-w 512 --render-aa none \
  --lane-render-band --lane-band-start-epoch 300 --lane-band-uncertainty-source witness \
  --lane-band-tau 0.85 --lane-band-eps 0.35 --lane-band-softness 1.0 \
  --lane-band-dash-forward-max-m 55.0 --lane-band-weight 1.0 \
  --persistence-loss-weight 1.0 --persistence-recall-weight 1.0 --cldice-iters 5 \
  --persistence-warmup-epochs 300 --persistence-classes auto \
  --amplify-weight 1.0 --amplify-form hinge --amplify-margin-target 1.0 \
  --amplify-persist inverse_thickness --island-dilate-px 1 \
  --structured-init --structured-init-include-lane \
  --lane-prior-phi1 --lane-prior-phi1-mode replace --lane-prior-phi1-dash-gate \
  --accum-pairs 8 --grad-clip 1.0 --ema-decay 0.997 \
  --lr 1e-3 --lr-end 1e-4 --weight-decay 1e-4 --adam-beta2 0.999 \
  --ckpt-every 25 --stage-checkpoints
```

Launcher note (unchanged): `tools/launch_witness_run.py --all-levers` reproduces the CAPSTONE all-levers
config (mod-dim 19 / β₂ 0.9999999 / w-pose 0) which HAND-DIVERGES on 4 deltas from this argv → assemble a
hand `launch.sh` or a small `witness_autoconfig` extension. Operator-gated; NOT this pass.

---

## 8. LAUNCH-TIME VERIFICATIONS (do at launch, not config edits) + RESIDUAL UNCERTAINTIES

**Launch-time checks (the operator's launch must confirm):**
1. **`custom_grouped_backward active=true`** — verify the 16.9× perf-env is ACTIVE (throughput, not just
   the flag), per the launch-gate-throughput discipline. (S-neutral; wrong → 16.9× slower, S unchanged.)
2. **The async n600 verdict prints d_seg every 25 ep** (confirm `--verdict-pairs 0` yields all-600
   verdicts in the log — the closed-loop EXTEND signal). 
3. **EXTEND at ep1000 if the Muon d_seg slope is still negative** (lexicographic: do NOT accept a
   descending ep1000 as final). Calibrate `decide_next_stage` thresholds from the observed n600-verdict
   variance before the gate (S6).
4. **Per-stage ckpts land** (`--stage-checkpoints --ckpt-every 25`) — resumability non-negotiable; the
   run is multi-day (~15 h opening + EXTEND).

**Residual uncertainties (ONLY the exact run + byte-close resolve — the honest three, sharpened):**
1. **Keyframe payload (rate):** §4 — the warp-real-luma decode keyframes are a counted cost §7 omits
   (~+0.006/10 s; full-clip UNMEASURED). #202 byte-close prerequisite for any sub-0.15 claim; first-row
   realistic band = sub-0.19.
2. **d_seg floor-reach (the central bet):** point-sample d_seg 0.00245 → ~0.001 via Muon+persistence+
   amplify is DESIGN, not measured (0.00086 is a real-frame BOUND, not witness-realized). The #205 run's
   binding open cell.
3. **Pose closure residual:** warp-alone d_pose 1.37 (term 3.70) → ~0.018 via trained dξ is
   ancestor-anchored, UNMEASURED on the witness (#221 fine-tune).

---

## VERDICT

**SEAL — LAUNCH-READY.** The §5b shippable argv + §5c calibrated schedule + seeding are the config the
operator commits to launch: NO-FAKE-clean (83/83 flags, all reach consumers, guards satisfied, the
supersample fake removed), deep-math-coupling-consistent (one doc-narrative update, no config defect),
means/ends-correct (w_pose=1.0 shippable), lexicographic-clean (S-neutral speed + mandatory EXTEND), and
de-risked at n600 ($0 horizon sweep CONFIRMS the shipped 174 / R1 FALSIFIED; supersample +49%-HURT
CONFIRMS `none`). All §8 config + §5c schedule open questions are RESOLVED or DEFERRED-with-reactivation;
NONE is launch-blocking. **Config edits required: ZERO.**

**This SEAL is of a MEANS.** The pointer stays **0.19110 UNMOVED**. The END is the operator's launch →
the closed-loop-extended n600 run → #202 byte-close (with the keyframe payload resolved) → the
`upstream/evaluate.py` exact row. Realistic first row: **sub-0.19**; **sub-0.15 is TIGHT** and contingent
on the d_seg floor-reach bet + a small keyframe payload. The three residuals (keyframe rate, d_seg
floor-reach, pose closure) are what only the exact run resolves.
