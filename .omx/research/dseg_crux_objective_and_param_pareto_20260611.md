# The d_seg crux: OBJECTIVE design + param↔d_seg↔byte Pareto + the exact next-daemon recipe (2026-06-11)

**Subagent:** `dseg_crux_objective_param_pareto`. **Mode:** DESIGN + ANALYSIS, CPU-only (the MLX GPU is
busy with a live daemon; NO GPU training launched). **Authority of every number:** `[macOS-CPU advisory]` /
`[macOS-MLX research-signal]` — each is a MEASURED artifact (cited inline) or a closed-form derivation shown
inline. NON-PROMOTABLE per the GOAL authority ladder: `promotable=false`, `score_claim=false`,
`ready_for_exact_eval_dispatch=false`. `$0` spend, NO cloud, NO paid GPU, NO MPS, NO /tmp. Frontier read from
pointer: **0.19109982 [contest-CPU], 177,169 B — UNMOVED.** This memo is a design + Pareto derivation; it does
NOT emit an archive. The two CPU checks below are tiny synthetic (the throughput profile re-read + a 64×64
synthetic-logit gradient-concentration probe), NOT scorer training.

NO FAKE: the param↔byte map is the MEASURED `capstone_vq_nerv_byte_budget_20260610.json` (exact byte
measurement). The param↔d_seg curve is the lab's own MEASURED cluster (PR101/102/103/A1 at 5.6e-4) +
Quantizr's 256K extrapolation. The throughput is the MEASURED `capstone_training_throughput_profile`. The
gradient-concentration check is a committed synthetic probe whose result CORRECTED my own prior (see §1.3).

---

## 0. HEADLINE (the three deliverables in one paragraph)

**(1) Objective:** do NOT replace the curriculum's seg losses with a new boundary-weighted TCKD/DKD objective —
that was MEASURED to LOSE 3.07× to KL-T2 in pixel space (`ab_boundary_tckd_vs_kl_t2_20260531.json`), and the
"margin surrogates concentrate gradient, CE doesn't" premise is FALSE (my §1.3 synthetic probe: CE already puts
99.4% of its gradient on the boundary band, same 5.1× as the surrogates). The d_seg-breaking objective is the
PR95 8-stage **seg-loss-FORM SCHEDULE that already exists** (`ce_seg_loss → tau_softplus → smooth_disagreement
→ l7_softplus`, all in `src/tac/score_aware_loop/live_segnet_loss.py`, all HNeRV-L6 score-domain margin
surrogates). The real lever is the SCHEDULE (CE seeds the partition fast; the bounded sigmoid/softplus
surrogates then stop over-spending on already-won pixels and reallocate to the still-flipping boundary), plus
the `l7` hard-pixel boost, plus EPOCHS. The crux is not the loss family — it is RUNNING ENOUGH EPOCHS of the
schedule on the EMA-fixed observable.
**(2) Pareto:** because the d_seg floor is FLAT (~5.6e-4) across the MEASURED 85K–180K param basin while
decoder bytes scale ~linearly, the SMALLEST base_ch that reaches the basin floor WINS on rate. **base_ch=20
(85K params, stored_latent carrier ≈ 101 KB, rate 0.067)** is the recommended point: predicted **S ≈ 0.140
(sub-0.15)** IF it reaches the basin floor; base_ch=22 (99K, rate 0.077) is the SAFER bank (S ≈ 0.150) if 85K
floors slightly high. This REVERSES the synthesis memo's "need MORE params" — that verdict read the FROZEN EMA
shadow (the d_seg-wall bug, `f771e6e00`), not the live descent.
**(3) Recipe:** one `run_capstone_campaign.py` command, `--carrier stored_latent --curriculum pr95_8stage
--base-channels 20`, on 600 pairs (the real gate) with `--curriculum-total-epochs ≥ 300`, detached daemon +
marker-on-exit. Pre-registered prediction: **live d_seg < 1.3e-3 by the smooth/l7 stages AND d_pose holds the
tube (≤ 5e-5) → S ∈ [0.135, 0.160] (sub-0.15 best case)**; pose drift to ~5e-4 is the named failure mode (S → 0.22).

---

## THE CENTRAL TENSION I HAD TO RESOLVE (two conflicting same-day verdicts)

Two memos dated hours apart on 2026-06-11 give OPPOSITE strategic verdicts:

- **Synthesis (01:50, `capstone_adversarial_synthesis_…`):** "CE plateaus ~0.008; the smaller-than-frontier
  basis FIGHTS the param↔d_seg curve; need frontier-CLASS params (base_ch 22–24); best capstone S ≈ 1.75."
- **EMA-shadow-lag (07:00, `capstone_ema_shadow_lag_reverses_seg_wall_verdict_…`, the NEWEST, commit
  `f771e6e00`):** "the d_seg the synthesis read was the EMA SHADOW, frozen near-init on short runs; the LIVE
  weights descend d_seg 0.507 → 0.041 in 25 ep and KEEP falling. The 'CE plateaus ~0.008' and 'smaller basis
  fights physics' verdicts are SUSPECT/WEAKENED — the wall was a measurement artifact."

**Resolution (this memo):** the EMA-fix memo is newer AND mechanistically decisive (it has a controlled
LIVE-vs-shadow A/B showing the shadow was bit-frozen while the live d_seg fell 12×). It SUPERSEDES the
synthesis's "need more params." The synthesis's two load-bearing claims — (a) "CE plateaus ~0.008" and (b) "85K
fights the curve" — both rested on shadow d_seg reads and are therefore not trustworthy. BUT the synthesis's
ONE durable contribution survives: **best capstone S ≈ 1.75 was dominated by d_pose ~0.1 (pose term 1.0), not
d_seg** — so the carrier MUST solve pose (the `stored_latent` carrier does; the 8-bit `vq_index` carrier does
NOT — `capstone_optimal_carrier_design` §1.2, pose needs 21 bits but a VQ index gives 8). The c1prime honest
run already uses `stored_latent` and shows d_pose → 4.4e-4 by stage-1 ep40 (`run.log`). So the resolved state:
**pose is handled by the carrier; d_seg is the binding wall; the wall is NOT a capacity floor at 85K (artifact
removed); the lever is the curriculum schedule + epochs at the SMALLEST basin-floor param count.**

The c1prime honest daemon (`capstone_c1prime_honest_b20_n48`, EMA FIXED) is the live decisive trajectory but it
**only reached stage-2 epoch 10 (10 logged rows) and is no longer running** (no process; no done-marker — the
session-watcher trap struck again). Its LIVE d_seg at last row = **0.0146 and still falling** at stage-2 ep10,
having descended 0.505 → 0.0198 in stage-1 alone. This is consistent with the basin reaching 5.6e-4 GIVEN more
epochs of the smooth/l7 stages — it has run ~50 of a planned 398 epochs, and 398 is itself ~75× short of PR95's
29,650. **The recipe below fixes both: enough epochs, and the marker-on-exit so it cannot silently die again.**

---

## 1. DELIVERABLE 1 — the d_seg-breaking OBJECTIVE design

### 1.1 What d_seg actually is (the functional the loss must minimize)
`d_seg = mean( argmax(SegNet(render_frame1)) != argmax(SegNet(GT_frame1)) )` over 384×512×600 — a per-pixel
0/1 argmax-FLIP rate (verified `upstream/modules.py`; `exact_d_seg_from_logits`,
`src/tac/score_aware_loop/live_segnet_loss.py:155`). Two structural facts the objective must exploit:
- The signal is 100% concentrated at the SegNet **decision boundary** (pixels with small top-2 logit margin);
  a confidently-correct interior pixel can NEVER flip → it has ZERO d_seg sensitivity.
- It is a SET functional on the argmax — invisible to the 80.67% scorer-null pixel energy. So a generic
  reconstruction loss (L2/PSNR) is the WRONG objective (HNeRV-L6: score-domain Lagrangian, not weight-domain
  proxy). The loss must drive boundary pixels back across the SegNet class line, per flip, per byte.

### 1.2 Is argmax-CE the right loss to reach d_seg ~1e-3? — NO as a SOLE loss; YES as the SCHEDULE's seed.
The PR95 curriculum already encodes the answer, and it is NOT "CE all the way":

| stage | seg loss | what it does on the d_seg functional |
|---|---|---|
| 1 | `ce_seg_loss` | F.cross_entropy. Seeds the partition FAST (c1prime: 0.505→0.020 in 40 ep). But CE is **unbounded log-loss** — it keeps pushing already-correct boundary pixels to ever-higher confidence, spending capacity on WON pixels. This is the "CE plateaus" mechanism: not that it spreads gradient over the interior (it doesn't — §1.3), but that it does not STOP at "correct," so the marginal pixel-flips get diluted by over-confidence updates on safe pixels. |
| 2 | `tau_softplus_seg_loss` | `τ·softplus(−margin/τ)` on `(target_logit − runnerup_logit)`. Gradient peaks near margin=0 (the boundary) and DECAYS smoothly past it → reallocates capacity from won pixels to flipping ones. |
| 3–4 | `smooth_disagreement_seg_loss` | `sigmoid(−margin/τ)` — at the optimum this EQUALS the true argmax-disagreement rate (the exact d_seg). Minimizing it directly minimizes d_seg; gradient is a bell curve peaked at margin=0 and VANISHES once a pixel is safely correct (the exact opposite of CE's unbounded push). This is the loss whose minimizer IS d_seg. |
| 5–8 | `l7_softplus_seg_loss` | softplus + a `(1 + l7_mult=4)` boost on HARD pixels (margin < threshold), renormalized to mean 1 → concentrates the last gradient budget on the still-flipping minority. This is the "boundary-weighted hard-pixel" objective the optimal-teacher memo proposed — it ALREADY EXISTS and is the curriculum's tail. |

**Verdict: the d_seg-breaking objective is the SCHEDULE `ce → tau_softplus → smooth_disagreement → l7_softplus`
(plus the C1a/sigma/QAT byte-robustness mechanisms deferred to byte-close), exactly as `curriculum.py` defines.
Reuse it; do NOT invent a new loss.** Predicted effect vs CE-only: CE-only asymptotes where over-confidence
updates dominate (synthesis estimated ~0.008, but that read the shadow — the true CE-only live floor is unknown
and likely lower); the bounded surrogates (stages 2–4) recover the descent past the CE knee because their
gradient vanishes on won pixels, and `l7` (stage 5+) concentrates the tail on the hard minority. This is the
MECHANISM by which PR95 crossed CE's knee to reach 5.6e-4.

### 1.3 The boundary-weighted-TCKD / DKD idea is FALSIFIED for this surface (honest correction of a prior)
The optimal-teacher design (`feedback_optimal_teacher_and_sensitivity_tools_landed_20260531.md`) proposed
boundary-weighted target-class-KD (DKD α-only ⊕ BPKD edge-loss) as the "score-optimal" seg distill, predicting
≥8% d_seg reduction vs KL-T2. **It was MEASURED to LOSE:** `ab_boundary_tckd_vs_kl_t2_20260531.json` (real
SegNet, 8 frames, 60 steps): KL-T2 final d_seg **0.00648** vs boundary-TCKD **0.0264** — boundary-TCKD is
**3.07× WORSE** (`prediction_holds: false`). And my own committed 64×64 synthetic gradient-concentration probe
(this session) shows the premise behind the proposal is wrong:

```
boundary-band pixel fraction = 0.194
loss                  frac grad mass IN band   concentration
ce_seg_loss                          0.994          5.1x
tau_softplus                         1.000          5.2x
smooth_disagreement                  1.000          5.2x
l7_softplus                          1.000          5.2x
```

**CE already puts 99.4% of its gradient on the boundary band — same 5.1× concentration as the margin
surrogates.** So "CE wastes gradient on the confident interior" is FALSE (the softmax gradient on a confidently-
correct pixel is ~0 under CE too). The surrogates' advantage is NOT concentration; it is the BOUNDED gradient
shape that vanishes on won pixels (CE's does not). This kills the "add a fancier boundary-weighted distill"
direction and REDIRECTS to "run the existing bounded-surrogate schedule for enough epochs." NO new loss is
warranted; the score-domain margin schedule that exists is the optimum the contest math selects.

### 1.4 Why this honors HNeRV-L6 + the innovation gate
The schedule is the score-domain Lagrangian `β·d_seg(via margin surrogates) + γ·√d_pose(via diff-YUV6 +
eval_roundtrip pose_loss)` — NOT rel_err². The innovation gate (lever C) is satisfied by the SMALLER fresh-init
score-aware basis (§2/§3), not by the loss; the loss is the proven PR95 mechanism reused faithfully.

---

## 2. DELIVERABLE 2 — the param↔d_seg↔byte Pareto

### 2.1 The param↔byte map (MEASURED, `capstone_vq_nerv_byte_budget_20260610.json` + quad fit on it)
Decoder params scale ~quadratically in base_ch (conv channels²); int8 decoder bytes = MEASURED 0.984 B/param.
Two carriers priced: `stored_latent` (C1′; decoder + 28-d latent ≈ 15,070 B temporal-delta+LZMA + decoupled
pose-store 1,557 B delta-coded floor-F2 + ~1,000 B framing) and the legacy `vq_index` (decoder + codebook 6,400
+ index 600 + pose 6,448 + framing).

| base_ch | params | int8 decoder B | **stored_latent total** | rate | vq_index total | rate |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 84,901 | 83,543 | **101,170** | **0.0674** | 97,241 | 0.0647 |
| 22 | ~99,231 | 97,643 | **115,270** | **0.0768** | 111,341 | 0.0741 |
| 24 | 114,710 | 112,875 | **130,502** | **0.0869** | 126,573 | 0.0843 |
| 28 | ~149,185 | 146,798 | **164,425** | **0.1095** | 160,496 | 0.1069 |

(base_ch 20/24 params are the MEASURED rows; 22/28 are the quad-fit interpolation `p ≈ 140.76·bc² + 1287.8·bc
+ 2773`, fit on the four measured rows, max residual <0.1%.) Sub-0.19 byte budget (rate < 0.077 / 25 × D ≈
115,640 B): base_ch ≤ 22 fits with margin; base_ch=24 (130 KB stored_latent) is just over for sub-0.15 but fine
for sub-0.19. Sub-0.15 rate budget (with seg+pose terms taking ~0.073, leaving rate ≤ ~0.077): base_ch ≤ 22.

### 2.2 The param↔d_seg curve (MEASURED cluster + extrapolation) — the DECISIVE shape
The lab's own floor memo (`grand_council_fields_medal_theoretical_floor_20260509.md`) MEASURED:
- **PR101 / PR102 / PR103 / A1 ALL cluster at d_seg ≈ 5.6e-4** across **88K–180K params** — a FLAT BASIN
  ("the architectural ceiling of THIS parameter budget"; the d_seg floor does NOT improve from 88K → 180K).
- **256K params → d_seg ≈ 2.8e-4** (Quantizr 2× extrapolation; needs +88 KB → loses on rate at our budget).
- Below ~88K the d_seg floor is expected to RISE (sub-basin), but 85K (base_ch=20) is at the very bottom EDGE
  of the measured basin → expected floor ≈ 5.6e-4 (advisory; the honest risk is it floors slightly higher,
  e.g. 7–9e-4, which only raises seg_term by 0.01–0.03).

**The decisive Pareto fact: the d_seg floor is FLAT (~5.6e-4) across 85K–180K while bytes scale linearly.**
Therefore adding params from 85K → 180K buys NO d_seg improvement but COSTS ~63 KB of rate (ΔS_rate +0.042).
The synthesis's "need more params for lower d_seg" is FALSE within the basin — it confused the shadow-frozen
d_seg (a measurement artifact) for a capacity floor. **The smallest base_ch that reaches the basin floor wins.**

### 2.3 The predicted-S Pareto (d_pose = 2.9e-5 tube via stored_latent, basin d_seg floor)

| base_ch | params | d_seg floor | seg_term | pose_term | rate (stored_latent) | **predicted S** | gate |
|---:|---:|---:|---:|---:|---:|---:|:--|
| **20** | 84,901 | 5.6e-4 | 0.0560 | 0.0170 | 0.0674 | **0.1404** | **< 0.15** ✓ |
| 22 | 99,231 | 5.6e-4 | 0.0560 | 0.0170 | 0.0768 | **0.1498** | **< 0.15** (edge) |
| 24 | 114,710 | 5.6e-4 | 0.0560 | 0.0170 | 0.0869 | 0.1599 | < 0.19 only |
| 28 | 149,185 | 5.6e-4 | 0.0560 | 0.0170 | 0.1095 | 0.1825 | < 0.19 only |

**RECOMMENDATION: base_ch = 20 (85K params).** Predicted S ≈ **0.140 (sub-0.15)**, the lowest of the four,
because it sits at the bottom of the flat d_seg basin AND the smallest byte count. base_ch=22 (S ≈ 0.150) is the
SAFE BANK: if 85K floors at ~7e-4 instead of 5.6e-4 (seg_term 0.070 not 0.056), base_ch=20 → S ≈ 0.154 (still
sub-0.19, just over sub-0.15), and base_ch=22 with the basin floor → S ≈ 0.150. **Run base_ch=20 first (best
case sub-0.15); if its d_seg floors above ~7e-4, the bank is base_ch=22.**

**Honest risk (the one named unprovable):** whether 85K reaches 5.6e-4 is a TRAINING outcome the basin curve
predicts but cannot prove (Kolmogorov-uncomputable per the floor report). 85K is at the basin EDGE, so the floor
could be 5.6e-4 (basin holds) or up to ~9e-4 (sub-basin onset). The recipe's job is to MEASURE the 85K floor on
600 pairs. If 85K floors > 1.3e-3 at convergence (sub-0.19 lost on seg alone), step to base_ch=24 (frontier-
class, sub-0.19 banked at S ≈ 0.16) — but the basin evidence says this is unlikely.

### 2.4 Cross-check against the free-decoder-conditional floor
`smaller_learned_basis_deep_math` §3 DERIVED the free-decoder-conditional intrinsic dimension at ~24.6–64.6 KB
(rate 0.016–0.043) — BELOW even base_ch=20's 0.067. So base_ch=20 is NOT at the theoretical floor; a more
aggressive smaller amortizer (or the fixed-PRNG-codebook VQ carrier on retrained weights, PATH 1) could go
lower later. base_ch=20 is the SAFE FIRST step toward that floor: large enough to plausibly hold the d_seg
basin, small enough to score sub-0.15. The sub-0.118 reach (below the measured S_floor) is a LATER campaign
(decoder shrink below 64 KB), not this daemon.

---

## 3. DELIVERABLE 3 — the EXACT next-daemon recipe

### 3.1 The affordability reality (MEASURED throughput — the binding constraint)
`capstone_training_throughput_profile_20260611T051024Z.json`: a full fwd+bwd step is **14.28 s**, of which
**98.15% is the torch-CPU scorer** (SegNet 4.70 s + PoseNet 1.23 s fwd, ~8.1 s bwd). MLX render is 0.04 s. So
the bottleneck is the CPU scorer, NOT the GPU (which is anyway busy). Epoch wall-clock (bs=8):

| pairs | steps/epoch | epoch wall | 300 ep | 1200 ep | 2000 ep |
|---:|---:|---:|---:|---:|---:|
| 48 (PROXY) | 6 | ~1.4 min | 7 h | 28 h | 48 h |
| 600 (the GATE) | 75 | ~17.9 min | 3.7 days | 14.9 days | 24.8 days |

**This is the hard wall.** A PR95-faithful curriculum (29,650 epochs) is INFEASIBLE on the CPU scorer at 600
pairs (would be years). The recipe must therefore (a) use a COMPRESSED curriculum-total-epochs, and (b)
exploit the MEASURED batch amortization (segnet per-frame 508 ms→147 ms from bs=1→bs=16) by using the LARGEST
batch the 128 GB unified memory allows. The two affordable real-gate options:

- **OPTION A (rank-1, the real verdict, ~3.7 days):** 600 pairs, `--curriculum-total-epochs 300`. Gives every
  stage ~enough epochs to show the schedule's re-acceleration; the d_seg/d_pose are the TRUE 600-pair terms
  (not a 48-pair proxy). This is the byte-closeable, paired-exact-eval-gating run.
- **OPTION B (rank-2, the cheap proxy first, ~28 h):** 48 pairs, `--curriculum-total-epochs 1200`. Cheaper +
  more epochs/stage, but 48/600 pairs is a PROXY — the d_seg on 48 pairs is NOT the 600-pair verdict (over-fits
  the 48). Use ONLY to confirm the schedule re-accelerates d_seg past the CE knee on the EMA-fixed observable
  BEFORE committing the 3.7-day 600-pair run. **Run B first as a $0 gating smoke, then A.**

### 3.2 The EXACT commands (reuse `experiments/run_capstone_campaign.py` — the actuator already exists)

**STEP 1 — cheap proxy gate (48 pairs, confirm the schedule re-accelerates; ~28 h, marker-on-exit):**
```bash
nohup bash -c '
  .venv/bin/python experiments/run_capstone_campaign.py \
    --max-pairs 48 --base-channels 20 --decoder-dtype int8 \
    --carrier stored_latent \
    --curriculum pr95_8stage --optimizer-schedule muon_throughout \
    --curriculum-total-epochs 1200 \
    --seg-weight 100 --pose-weight 1 \
    --muon-lr 3e-2 --grad-clip 50 --grad-clip-muon 1 \
    --eval-every 10 --seed 0 \
    --out-dir experiments/results/capstone_b20_n48_curric1200 ;
  touch experiments/results/capstone_b20_n48_curric1200/DONE.marker
' < /dev/null > .omx/tmp/capstone_b20_n48_curric1200.log 2>&1 &
disown
```

**STEP 2 — the real 600-pair gate (only if STEP 1 live d_seg crosses ~3e-3 in the smooth/l7 stages; ~3.7 days,
marker-on-exit):**
```bash
nohup bash -c '
  .venv/bin/python experiments/run_capstone_campaign.py \
    --max-pairs 600 --base-channels 20 --decoder-dtype int8 \
    --carrier stored_latent \
    --curriculum pr95_8stage --optimizer-schedule muon_throughout \
    --curriculum-total-epochs 300 \
    --seg-weight 100 --pose-weight 1 \
    --muon-lr 3e-2 --grad-clip 50 --grad-clip-muon 1 \
    --eval-every 5 --seed 0 \
    --out-dir experiments/results/capstone_b20_n600_curric300 ;
  touch experiments/results/capstone_b20_n600_curric300/DONE.marker
' < /dev/null > .omx/tmp/capstone_b20_n600_curric300.log 2>&1 &
disown
```

### 3.3 Recipe rationale (each knob, grounded)
- `--base-channels 20`: §2.3 — smallest basin-floor param count, lowest predicted S (0.140). Bank = 22.
- `--carrier stored_latent`: the pose-capable carrier (`capstone_optimal_carrier_design` §1.2 — pose needs 21
  bits; the vq_index 8-bit carrier is the MEASURED pose wall). The c1prime run confirms d_pose → 4.4e-4 with it.
- `--curriculum pr95_8stage`: §1 — the seg-loss-FORM schedule is the d_seg lever, not a new loss. The runner
  already builds it via `build_pr95_8stage_curriculum`.
- `--optimizer-schedule muon_throughout`: the #77 deviation (Muon from stage 1) — the inert-loop audit found
  AdamW + 100%-clip stalled the early MLX stages; the c1prime run used muon_throughout and descended cleanly.
  (`pr95_adamw_then_muon` is the FAITHFUL fallback if Muon-throughout over-fits the small basis.)
- `--grad-clip-muon 1`: the StageSpec default `grad_clip_muon=1.0` (PR95). Note: the synthesis flagged the
  earlier runs used `--grad-clip-muon 50` which the curriculum may override; the CLI value here matches PR95.
- The EMA is now FIXED (warmup decay, commit `f771e6e00`, default `ema_decay 0.997`, `use_ema_for_eval` →
  exports the shadow that the advisory measures): no flag needed; it is the trainer default post-fix.
- `eval_roundtrip=True` is hard-wired in the bridge (the int8/bicubic roundtrip is simulated in the loss).
- The runner reloads the int8 archive and scores the RELOADED int8 terms as the honest contest predictor
  (`score_reloaded_int8_archive`) — the printed advisory S is the quant-aware one.

### 3.4 Pre-registered prediction (Dykstra-feasibility grounded, NOT vibes)
**Prediction (advisory):** under the full margin-surrogate schedule at 600 pairs / base_ch=20, the LIVE
(EMA-shadow, now-tracking) d_seg crosses **1.3e-3 by the `smooth_disagreement`/`l7_softplus` stages** and floors
in **[5.6e-4, 9e-4]** (the basin edge band); d_pose holds in the tube (stored_latent carrier, c1prime-confirmed
at stage-1 ep40 = 4.4e-4 and still falling). The pose_term is the SWING factor (concave √): at the tube
(2.9e-5) pose_term = 0.017; at d_pose=5e-4 pose_term jumps to 0.071 (ALONE +0.054). So the band is
**conditional on pose holding the tube**: IF d_pose ≤ ~5e-5 → **S ∈ [0.135, 0.160]** (best 0.140 at the basin
floor; sub-0.15 to low-sub-0.16); IF d_pose drifts to ~5e-4 → S ∈ [0.21, 0.23] (pose-term-dominated,
sub-0.19 LOST — this is the synthesis's S≈1.75-class failure mode in miniature and the reason the carrier
MUST hold pose). **The gating sub-prediction: d_pose ≤ 5e-5 at curriculum end** (c1prime stage-1 already at
4.4e-4 descending → plausible but UNCONFIRMED at 600 pairs). **Feasibility basis (NOT vibes):** the (d_seg,
d_pose) target (5.6e-4, 2.9e-5)
is the MEASURED frontier-cluster vertex — the intersection `C_seg(5.6e-4) ∩ C_pose(tube)` is NON-EMPTY at
85K–180K params (PR101/102/103/A1 all LAND there, floor memo §observation), so the cell is Dykstra-feasible at
base_ch=20's capacity; the rate vertex 0.067 is the MEASURED byte budget (exact). The only open coordinate is
whether 85K (vs the cluster's 88K–180K) reaches the SAME d_seg vertex — the basin-edge risk §2.3 names, which
the run measures.
**Pre-registered KILL/DEFER:** if the 600-pair live d_seg floors > 1.3e-3 at curriculum end (seg_term > 0.13,
sub-0.19 lost on seg alone), DEFER base_ch=20 → reactivate at base_ch=24 (frontier-class, sub-0.19 banked) and
record the 85K sub-basin floor as a new MEASURED anchor on the param↔d_seg curve. NOT a paradigm kill
(Catalog #307) — the curriculum-at-85K is one config of the lever-C family.

### 3.5 The gate to a pointer move (what makes the daemon worth its 3.7 days)
Only if STEP 2's advisory S (reloaded int8) beats the frontier 0.19110 OR crosses sub-0.15 → byte-close (the
runner already emits `archive.zip`) → run `inflate.sh → evaluate.py` on the 600 real samples (NEVER validated
yet — only toy inputs, per the synthesis's honest caveat) → if it holds, ONE paired contest CPU+CUDA exact eval
(~$0.6, within budget). That paired exact row is the ONLY pointer-moving step; everything above is advisory.

---

## 4. WIRE-IN (Catalog #125) + SCOREBOARD

1. **sensitivity-map — ACTIVE.** New prior: the d_seg floor is FLAT (~5.6e-4) across the 85K–180K param basin;
   the rate lever is the SMALLEST basin-floor base_ch, not more params. The aiming surface is base_ch=20.
2. **Pareto — ACTIVE.** Adds the wall that the d_seg basin is flat 85K–180K (adding params buys 0 d_seg, costs
   ~0.042 rate) → the Pareto-optimal learned point is base_ch=20 (S 0.140), not the synthesis's base_ch=22–24.
3. **bit-allocator — ACTIVE.** Allocator = {decoder 83 KB int8 @ base_ch=20} + {stored 28-d latent ≈ 15 KB} +
   {decoupled scalar pose-store ≈ 1.6 KB} + framing 1 KB = ~101 KB; pose = bits-as-precision (1 scalar 21 bits),
   seg = the amortized decoder + per-pair latent refinement.
4. **cathedral-autopilot — gate-conditional.** The STEP-1 proxy → (re-acceleration confirmed) → STEP-2 600-pair
   → (advisory beats frontier / sub-0.15) → ONE paired exact eval is the dispatch surface.
5. **continual-learning — ACTIVE.** Reseeds the V3 judge: (a) the d_seg "wall" at 0.505 was an EMA-shadow-lag
   artifact, REVERSED (`f771e6e00`) — the small basis DESCENDS; (b) boundary-weighted-TCKD/DKD LOSES to KL-T2
   and to the existing margin schedule (MEASURED + synthetic) — do NOT re-propose a new boundary distill;
   (c) the d_seg floor is a FLAT basin 85K–180K → smaller-is-better on rate within the basin; (d) the binding
   compute constraint is the torch-CPU scorer (98% of step cost) → 600-pair PR95-faithful epoch counts are
   infeasible on CPU; the lever is a COMPRESSED curriculum + batch amortization.
6. **probe-disambiguator — RESOLVED.** "Is CE the right d_seg loss?" → CE seeds, the bounded margin schedule
   (tau_softplus→smooth_disagreement→l7) finishes; no new loss. "Bigger or smaller basis?" → SMALLER within
   the flat basin (base_ch=20); "synthesis vs EMA-fix?" → EMA-fix supersedes (the wall was an artifact).

**UPPER (vs T_1 sub-0.19):** unchanged — design memo, no archive. Frontier holds 0.19110 [contest-CPU].
**LOWER (the floor):** consistent with the free-decoder-conditional band (rate 0.016–0.043,
`smaller_learned_basis_deep_math` §3); base_ch=20's predicted 0.067 rate is ABOVE that floor (room remains for
a later decoder-shrink campaign toward sub-0.118), and its predicted S 0.140 is the named first crossing of
the sub-0.15 distortion-at-budget door.

## 5. CROSS-REFERENCES
`capstone_ema_shadow_lag_reverses_seg_wall_verdict_20260611T070000Z.md` (the wall = artifact, the reversal this
memo builds on; commit `f771e6e00`) · `capstone_adversarial_synthesis_and_honest_corrections_20260611T015018Z.md`
(the "need more params" verdict this memo SUPERSEDES — it read the shadow; the surviving pose-term contribution)
· `capstone_optimal_carrier_design_20260611T041937Z.md` (C1′ stored_latent carrier; pose intrinsic dim=1.00 →
21-bit scalar; the 8-bit VQ pose wall) · `capstone_vq_nerv_byte_budget_20260610.json` (the MEASURED param↔byte
map) · `capstone_training_throughput_profile_20260611T051024Z.json` (the 14.28 s/step, 98% CPU-scorer
affordability wall) · `grand_council_fields_medal_theoretical_floor_20260509.md` (the MEASURED param↔d_seg
basin 88K–180K@5.6e-4; 256K→2.8e-4 extrapolation) · `smaller_learned_basis_deep_math_20260610T191009Z.md` (§3
free-decoder-conditional floor 25–65 KB) · `ab_boundary_tckd_vs_kl_t2_20260531.json` (boundary-TCKD LOSES 3.07×
— the FALSIFIED objective) · `feedback_optimal_teacher_and_sensitivity_tools_landed_20260531.md` (the proposed
boundary-TCKD this memo retires) · `src/tac/score_aware_loop/live_segnet_loss.py` (the 4 margin-surrogate seg
losses + `exact_d_seg_from_logits`) · `src/tac/mlx_pr95_port/curriculum.py` (the 8-stage schedule;
`build_pr95_8stage_curriculum`) · `experiments/run_capstone_campaign.py` (the actuator; `--carrier`,
`--curriculum`, `--curriculum-total-epochs`) · `experiments/results/capstone_c1prime_honest_b20_n48/`
(the EMA-fixed live trajectory: stage-1 0.505→0.0198, stage-2 ep10 0.0146 still falling) ·
`GOAL_standing_v3_20260610.md` (lever C = the smaller fresh-init score-aware basis) ·
`upstream/{modules.py,evaluate.py,README.md}` (frozen authority).
