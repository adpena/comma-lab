# T5 CRUCIBLE-2 — POSITION S4 (RATE / BYTE-CLOSE / POSE-CARRIER / MODAL) — 2026-07-09

**Seat:** S4 (Shannon — rate axis + byte-close + pose composition + the Modal exact-eval envelope).
**Scope:** OPEN Q5 (pose composition + terminal head solve + #314 rate basis) · OPEN Q6 (torch-parity
twin) · the v7.5.2 RATE half · byte-close cadence · measurement-plan skeleton. INDEPENDENT — did NOT
read position_S1/S2/S3. `[no-triality]` (apparatus/derivation, not a lever build). `$0`. #205 untouched
(STOPPED; box free). Pointer **0.19110 UNMOVED** — everything here is `[macOS-CPU advisory]`
NON-PROMOTABLE MEANS; the END is a byte-closed `upstream/evaluate.py` n600 row < 0.19110.

**STORES CONSULTED:** DELTA_GROUNDING (§B L-8 · §C P-1/P-5/P-7 · §D · §F-2 · §M M-1/M-2 · Q5/Q6) ·
CONVENING · ORCHESTRATION_LEDGER (incl. the mid-P1 **POSE ENGAGEMENT GATE** operator constraint) ·
`tools/levelset_byte_close_and_eval.py` (header rule-118 FREE/COUNTED · `serialize_pose_carrier` vs
`serialize_pose_carrier_store_nothing` v2 derive-H · `select_best_weights_arm` ema/live/polyak ·
`_axis_and_authority` · `--run-exact-eval`/`--eval-device`) · DAG FEED-snx (store-nothing byte-close
MEASURED 1049 B vs 697941 B) · FEED-db (RATE-NOT-BINDING) · FEED-08d/08g (#341 subset-overfit NO-GO,
full-P-GPU-only, LM ρ CONFIRMED) · FEED-238resolved (R1 dxi BANKED) · DAG:3827 (l7 = implicit MDL) ·
`docs/operating_manual_craft_handoff.md` (means≠ends; label MEASURED/DERIVED/ASSUMED; attack own conclusion).

---

## 0. HEADLINE (answer-first)

1. **Pose-carrier source = `generated` store-nothing, MANDATORY, at any fresh v7.5.2 arm** — NOT an
   open optimization. The counted-keyframe alternative (v1→v5 lineage) is **rate 0.46473** (MEASURED
   697,941 B) = a submission-killer alone. Store-nothing is **rate 0.00068–0.00479** (MEASURED 1049 B →
   7,195 B). The operator's decision is a **1-bit confirmation + a warm-start-lineage guard**, not a tuning.
2. **Do NOT build a torch-parity twin of the trainer.** verdict_scope FORMULATION: dominated. It is a
   ~15,200-LOC re-implementation gated on Metal kernels (fused-R VJP, grouped-backward) that have NO
   CUDA twin, and — decisively — a torch-trained arm ≠ an MLX-trained arm, so the fan-out A/B would
   measure the ENGINE, not the lever (a fake A/B, NO-FAKE #8). Modal's proper role is **exact-eval of
   byte-closed ARCHIVES** (parity-safe by construction), not training.
3. **Byte-close cadence = often-and-local ($0 CPU advisory), exact-eval rare-and-paid (Modal).** Byte-close
   ALL arms (ema/live/polyak) at every per-stage checkpoint boundary + intra-stage on the eval cadence;
   spend the ≤$20 Modal envelope ONLY on the byte-closed WINNER archive(s), paired CPU+CUDA.
4. **Pose engagement = a d_seg-CONDITIONING-GATED event (σ_min basin), never epoch 726** — obeys the
   operator binding constraint. #341 terminal head solve composes BEFORE pose-descent (a d_seg
   conditioner), IN-TRAINER GPU (~11 min local MLX), fired IFF LM ρ re-verified.

---

## 1. Q5 — POSE-CARRIER SOURCE DECISION (the rate case, the risk, the operator's call)

### 1.1 The rate case is not close (MEASURED)

The scorer reads d_pose off the FRAMES, so the pose carrier is whatever makes frame0 a domain-coherent
ego-warp of frame1. Two carrier lineages exist, and their RATE differs by **~700× / ~0.46 score**:

| carrier (byte-close mode) | pose section bytes | rate = 25·B/37,545,489 | provenance |
|---|---|---|---|
| `warp_real_luma` (stores real keyframe luma + H) — **v1→v5 fresh_seeded lineage** | **697,941 B** | **0.46473** | MEASURED FEED-snx (t1/n6); P-7 |
| `store_nothing` v2 (coded ξ only, derive-H FREE rule-118) — **crucible_v6/v7 + store_nothing_205** | **1,049 B** floor | **0.00068** | MEASURED FEED-snx; `serialize_pose_carrier_store_nothing` |
| `store_nothing` at R1's converged ξ_eff | **7,195 B** | **0.00479** | MEASURED FEED-238resolved (row P-1) |

Per FEED-db (DAG:1794, **RATE-NOT-BINDING**): at the witness operating point d_seg DOMINATES rate ~4:1;
the whole payload is a few KB vs frontier ~177 KB. That conclusion **holds ONLY under store-nothing.**
The keyframe carrier is the ONE way the witness could become rate-doomed (C1: store-nothing walls →
forced to the keyframe table 0.51 = rate-DOOMED). So this is not a knob — it is a **fail-closed rate gate.**

### 1.2 The risk (honest — the coupling to pose efficacy)

The store-nothing rate case is contingent on the JOINT-trained render+dxi actually CONVERGING to a low
d_pose. **This is not hand-waved: R1 BANKED it at n600 authority — d_pose 0.001610 via store-nothing ξ
(row P-1).** So both the rate case AND a convergence precedent are MEASURED. The residual risk is P-5's
HONEST FLAG: whether v7.5.2's OWN terminal finish reproduces an R1-class dxi from its OWN basin is
**UNVALIDATED** (mechanism correct + byte-identical-when-off; efficacy OWED). If v7.5.2's finish walls,
the fallback is the **already-proven R1 dxi (7.2 KB, rate 0.00479)** — NOT the keyframe codec. The
keyframe carrier is retired as a rate lever (rate-doomed); it survives only as a diagnostic A/B ceiling.

### 1.3 The one real trap: warm-start lineage laundering the keyframe rate

Q2 (warm-start vs fresh) couples directly to rate. **If v7.5.2 warm-starts from a v1→v5 `real_keyframe`
checkpoint, the byte-close would charge 697,941 B (rate 0.46) even though the run "looks" store-nothing.**
The byte-close tool is lineage-tagged (P-7) but the LAUNCH config must PIN `--pose-carrier-source
generated`. Candidate warm-start basins named in Q2 (mod32cap 0.003366 / v2_attrclean 0.004024) are
`generated`-lineage — SAFE; a v1→v5 basin is NOT. **Guard: any warm-start source's pose-carrier lineage
is verified `generated` before launch; the byte-close asserts `pcar_store_nothing_v==2`.**

### 1.4 The decision the operator must make (crisp)

**CONFIRM `--pose-carrier-source generated` (store-nothing) as the v7.5.2 default** — this is already
crucible_v7's default, so the decision is: (a) do NOT flip it to `real_keyframe` under any "richer pose"
argument (rate-doomed); (b) if warm-starting (Q2), restrict the source basin to a `generated`-lineage
checkpoint. verdict_scope on the keyframe carrier: **FORMULATION NO-GO as a rate lever** (0.46 rate);
still valid as an A/B ceiling only. This is not re-opening N-2 (lane-ξ) or N-1 (OT) — those stay closed.

---

## 2. Q5 — TERMINAL POSE-FINISH + #341 HEAD SOLVE COMPOSITION (under the operator pose-gate)

**Operator binding constraint (ORCHESTRATION_LEDGER §POSE ENGAGEMENT GATE, verbatim):** *"pose must not
be fired for joint descent until optimal it needs dseg to be sufficiently conditioned first."* Pose-finish
is a d_seg-CONDITIONING-GATED EVENT, never an epoch. My rate/pose-seat position:

- **SHIP the R1 dxi (0.127 contribution, 7.2 KB) via the terminal store-nothing pose-finish (D.9),
  gated on the conditioning event — NOT epoch 726.** The D.9 wiring already exists (P-4); the change is
  the TRIGGER.
- **Conditioning quantity = σ_min basin sensor** (P-6: `median_sigma_min` / `basin_frac` from
  J_ξ=∂(PoseNet∘R)/∂ξ). Provenance: DERIVED, not hand-set — coherence↔conditioning is the measured
  Jacobian mechanism (converging d_seg → richer boundary normals → σ_min↑). The R1 precedent IS the
  threshold anchor: R1 converged d_pose 0.0011 from a CONVERGED trunk (σ_min high); the birth-arm's
  ill-conditioned trunk (σ_min low) gave pose 1.8–4.35. So **threshold = σ_min at/above the R1-converged
  basin level** (to be pinned from the jacobian_basin telemetry as a DERIVED-AT-CONFIG quantile, not a
  literal — a value-provenance-ladder obligation for the synthesis).
- **Hysteresis + never-reached fallback:** require σ_min ≥ threshold for K consecutive verdict rows
  (hysteresis kills a one-row spike); if never reached, pose-finish NEVER fires → the run stays
  pose-BLIND = byte-identical, and the row is **d_seg-only with pose UNSHIPPED** (do NOT assume 0.127).
  The d_pose is always MEASURED through byte-close, never asserted (P-5).
- **#341 terminal head GN/CG solve (L-8) composes BEFORE pose-descent, as a d_seg conditioner.** The
  ~791-param affine head solve (LM ρ 0.847/0.868 CONFIRMED) sharpens the argmax → richer boundary
  normals → higher σ_min → a better pose basin. Sequence: **converge d_seg → #341 head solve → σ_min
  conditioning gate → store-nothing pose-finish.** SOLVE-then-descend, NOT joint (joint head+pose
  confounds attribution and the head-solve IS the conditioner the pose gate wants). This satisfies
  SYNTHESIS REQ A (solve-where-solvable) AND the operator gate (pose strictly after conditioning).
- **#341 is IN-TRAINER LOCAL GPU (~11 min grouped-backward on the M5 Max), NOT Modal, NOT $0** —
  MEASURED FEED-08d (full-P CG iter ≈ 3.2 h CPU / ~11 min GPU). Fire IFF LM ρ ∈ ~[0.8,1.2] RE-VERIFIED
  on the CURRENT terminal ckpt, all 600 pairs, exact tau-stage loss, `--fused-r-kernel` bit-identity.
  verdict_scope on the K=8 subset variant: **FORMULATION NO-GO** (+5.1% overfit, N-3 / FEED-08d) — do
  NOT re-open the subset tool; full-P in-trainer only.

---

## 3. THE RATE HALF — WEIGHTS-RATE LEVERS (firing conditions)

**Framing (do not forget):** RATE IS NOT THE BINDING CONSTRAINT for this vehicle (FEED-db). d_seg is the
entire remaining fight; rate ~0.06 vs d_seg contribution ~0.45 at the current base. So every weights-rate
lever is **lexicographic-secondary** — near-goal ANY real byte cut at zero d_seg cost is pure S and worth
banking, but NONE is a launch-gating d_seg lever. Relative-significance note (per the anti-orphan rule):
these are small in ΔS but a genuine 0.003 free cut is ~7% of the remaining gap to sub-0.19 from an
advisory 0.64 — bankable at byte-close, not dismissible as "noise," but NOT sequenced ahead of d_seg.

- **#242 flat-minima / MDL is achieved IN-TRAINING, not post-hoc.** l7 (the worst-pixel high-p
  objective) is an implicit MDL regularizer (DAG:3827 — pushes toward the simplest piecewise-constant
  state → lower-entropy weights → smaller int8+brotli blob). Firing condition: it is ALREADY the l7
  curriculum stage; no separate post-hoc recode. Do NOT add a post-hoc weight-entropy pass — the counted
  payload minimality is a training-curriculum property, and every post-hoc sidecar/recode class is
  measured-CLOSED (DAG:182).
- **#336 sensitivity bit-alloc "compress-half" = TERMINAL-only, conditionally-fired.** Firing condition:
  fire at byte-close IFF (a) the archive is within a threshold of a submission boundary where a byte cut
  changes the rounded S, AND (b) the per-tensor sensitivity map shows int-precision headroom at zero
  realized-d_seg cost. Since rate is not binding, this is a terminal polish knob, not a launch flag.
  Default OFF; a duty-to-measure tail item, ranked BELOW every d_seg lever.
- **Pose-sidecar per-column bit-alloc recode = FREE −0.0030 S at byte-close (order-exploit, FEED-db
  DAG:1785).** The ξ/pose payload recoded per-column fixed-point ([11,5,5,4,4,5]+Δ+brotli) drops
  6.8 KB→2.3 KB with added MSE 1.8e-6 < fp16's 6.18e-6 = **d_pose-NEUTRAL, −0.0030 S, zero risk.** This
  is already the store-nothing `delta_ar` coder path (`xi_pose_coder`). Firing condition: ON at byte-close
  by default (`--pose-carrier-mode store_nothing` uses the coded payload). No launch decision.

---

## 4. BYTE-CLOSE CHECKPOINT CADENCE (feeds the Modal exact-row earmark EARLY)

**The principle:** byte-close is $0 local CPU-advisory; exact-eval is the only promotion authority and
the only paid step. So byte-close OFTEN and LOCAL; exact-eval RARE and on the WINNER. The per-stage
checkpoint is BOTH crash-insurance AND a measurement artifact (resumability non-negotiable: each stage's
EMA is independently byte-closeable → N early exact-row candidates from ONE run).

- **Byte-close ALL arms at every per-stage checkpoint boundary** (CE→tau, tau→l7, Muon boundary,
  terminal) via `select_best_weights_arm` (ema/live/polyak — records the N-way winner + margins). The
  EMA-lag escape (live vs ema) is exactly why all three arms are ranked, not just ema (R-2 EMA-lag caveat).
- **Byte-close intra-stage on the eval cadence** (~every 50–100 ep) at $0 — produces a rolling advisory
  realized-d_seg/d_pose/rate curve so an EARLY sub-0.19-advisory candidate surfaces before terminal
  convergence and can be earmarked for a Modal exact row.
- **Every byte-close records the full-facet row** (F-2 owed): per-class d_seg · **d_pose (decisive)** ·
  rate · peak-RSS (the #205-OOM lesson — a config test is a surrogate; the REAL n600 byte-close is the
  authority) · arm-winner · pose-carrier lineage assertion (`pcar_store_nothing_v==2`).
- **Governed:** the execute-at-n600 that PRODUCES the checkpoints rides the full `launch_witness_run.py`
  gate chain (operator-GO). The byte-close of an EXISTING ckpt is $0 CPU and needs no GO.

---

## 5. Q6 — TORCH-PARITY TWIN VERDICT: DO NOT BUILD (verdict_scope FORMULATION, dominated)

**The question mis-frames where the bottleneck is.** Modal is for exact-eval + the CPU-torch n600
verdict; witness TRAINING is MLX-local (M-2). To "fan out ON/OFF lever A/Bs in parallel on Modal," you
must TRAIN each arm on Modal — which requires a torch twin of the TRAINER. Counting the real surfaces:

| surface | cost | provenance |
|---|---|---|
| trainer LOC to re-implement | **~15,218 lines** (levelset 12,005 + base 3,213) | MEASURED `wc -l` |
| Metal kernels with NO CUDA twin | **fused-R VJP + grouped-backward (~17×) + mx.compile/mx.fast** | L70 / L45; each needs a CUDA/torch rewrite + numpy-fp32 parity gate |
| numeric-parity risk | **MLX-GPU bit-identity is per-{chip,os,mlx,device}** — a torch trainer follows a DIFFERENT optimizer trajectory | S-6 / L70 / #348 |
| scorer-forward batch dependence | bit-identity-at-speedup **IMPOSSIBLE**; bounded n600 A/B is the ONLY admission path | S-6 / #313 |

**Decisive kill (NO-FAKE #8):** a torch-trained arm is NOT the same run as an MLX-trained arm. So a
"lever-ON (torch, Modal) vs lever-OFF (MLX, local)" A/B measures the **ENGINE difference**, not the lever.
Even torch-ON vs torch-OFF on Modal would be a torch-vehicle finding that does not transfer to the MLX
vehicle we actually ship. The fan-out A/B is a **fake A/B** — the exact class the surrogate≠authority /
apples-to-apples disciplines forbid. Building it would burn a large multi-week engineering effort to
manufacture confounded rows.

**The parity-safe boundary is the ARCHIVE, not the trainer.** The byte-closed `archive.zip` + inflate.py
(numpy + brotli + torch-for-R — ALREADY not MLX) is the shared invariant; `upstream/evaluate.py` on it is
parity-safe by construction (DELTA Q6). So Modal already has its correct, safe job: exact-eval of
byte-closed archives, no twin required.

**RECOMMENDATION (with the envelope math):** Train A/B arms LOCALLY on the free M5 Max box (#205 STOPPED,
~120 GiB free) — serial, or memory-safe light-parallel per the memory policy (measure per-arm n600 RSS
first; NEVER blind). Byte-close each locally at $0. Spend the **≤$20 Modal envelope ONLY on paired
exact-eval of byte-closed WINNER archives.** Cost model: a paired n600 row ≈ CPU container (~60–120 min,
~$0.06/hr → ~$0.12) + T4 CUDA (~minutes, ~$0.59/hr → <$0.10) ≈ **~$0.25/paired-row → ~80 rows within
$20.** The throughput ceiling is TRAINING (MLX-local, un-helped by a twin); Modal cannot lift it without
the confounded twin, so the $20 is best spent BUYING exact rows, not compute. This is the "spend the
budget to BUY exact rows" mandate, exactly.

---

## 6. CONFIG-SHAPED BLOCK (rate flags · byte-close cadence · Modal exact-eval trigger)

```yaml
# ---- v7.5.2 RATE-HALF flags (launch config; DSL WitnessProgram) ----
pose_carrier:
  source: generated                 # MANDATORY store-nothing; rate 0.005 not 0.46. NEVER real_keyframe.
  warm_start_lineage_guard: true    # if warm-start (Q2): assert source basin is generated-lineage
  byte_close_mode: store_nothing    # v2 derive-H (rule-118 FREE); coded ξ (delta_ar) = the sidecar recode
  xi_coder: delta_ar                # per-column fixed-point recode -> -0.0030 S, d_pose-NEUTRAL, ON by default
pose_finish:                        # D.9 terminal finish — CONDITIONING-GATED EVENT, never epoch
  trigger: sigma_min_basin_conditioned   # NOT epoch 726 (that is the never-reached fail-safe cap)
  conditioning_quantity: median_sigma_min (jacobian_basin telemetry)
  threshold: DERIVED-AT-CONFIG from R1-converged basin quantile   # provenance owed to synthesis, not a literal
  hysteresis: K consecutive verdict rows >= threshold
  never_reached_fallback: pose UNSHIPPED, run stays byte-identical, row is d_seg-only (never assume 0.127)
terminal_head_solve:                # #341 — d_seg conditioner BEFORE pose-descent
  enabled_iff: LM_rho in [0.8,1.2] re-verified on CURRENT ckpt, full-P (P=600), exact tau-loss, --fused-r-kernel
  where: IN-TRAINER LOCAL GPU (~11 min grouped-backward)   # NOT Modal, NOT the K=8 subset (NO-GO N-3)
  sequence: converge d_seg -> #341 head solve -> sigma_min gate -> store_nothing pose-finish
weights_rate:
  flat_minima_mdl: via l7 curriculum stage (in-training)   # NOT a post-hoc recode
  sensitivity_bit_alloc_compress_half: TERMINAL-ONLY, default OFF
    fire_iff: archive within submission-boundary threshold AND per-tensor headroom at zero realized-d_seg cost

# ---- BYTE-CLOSE CADENCE ($0 local CPU advisory; all arms) ----
byte_close:
  tool: tools/levelset_byte_close_and_eval.py --select-arms --pose-carrier --pose-carrier-mode store_nothing --verify-bit-exact
  at_stage_boundary: [CE->tau, tau->l7, muon_boundary, terminal]   # each per-stage EMA independently byte-closeable
  intra_stage: every ~50-100 ep on eval cadence                    # rolling advisory curve -> EARLY candidates
  arms: [ema, live, polyak]           # select_best_weights_arm N-way winner+margins (EMA-lag escape)
  row_facets: [per_class_d_seg, d_pose(decisive), rate, peak_RSS, arm_winner, pcar_store_nothing_v==2]

# ---- MODAL EXACT-EVAL TRIGGER (<=$20 HARD CAP; the ONLY promotion authority) ----
modal_exact_eval:
  budget_cap_usd: 20                  # M-1
  cost_per_paired_row_est_usd: ~0.25  # CPU container + T4 CUDA -> ~80 rows headroom
  fire_when:                          # a byte-closed archive is worth a PAID row
    - local advisory realized-d_seg beats current best-candidate by a margin, AND
    - it is a stage-terminal OR a sub-0.19-advisory candidate
  run: paired  ->  --run-exact-eval --eval-device cpu   AND   --run-exact-eval --eval-device cuda
  authority: contest-CPU (leaderboard axis) + contest-CUDA, on the SAME byte-closed archive bytes
  NOT_modal: witness training (MLX-local), the torch-parity twin (do-not-build), #341 head solve (local GPU)
```

---

## 7. Adversarial self-check (attack my own conclusion)

- **"Store-nothing might not converge d_pose → forced to the 0.46 keyframe."** Countered by MEASURED
  precedent: R1 BANKED d_pose 0.001610 via store-nothing ξ at n600 (P-1). The residual risk is v7.5.2's
  OWN finish (P-5 UNVALIDATED) — handled by: d_pose always MEASURED through byte-close (never assumed),
  and the fallback is the PROVEN R1 dxi (7.2 KB), NOT the keyframe. The keyframe stays retired.
- **"A twin could still give torch-ON vs torch-OFF rows cheaply."** No — that is a torch-vehicle finding
  that does not transfer to the MLX ship vehicle (per-{chip,os,mlx,device} bit-identity); it manufactures
  confounded rows at large build cost. The archive is the only parity-safe boundary. Verdict holds.
- **"Rate levers dismissed as secondary — near-goal orphaning?"** Explicitly NOT orphaned: the pose-sidecar
  recode (−0.0030 free) is banked ON by default, and compress-half is a tracked terminal duty-to-measure
  item — just not sequenced ahead of the d_seg blocker. Relative-significance applied (§3).
- **"σ_min threshold is a hand-set literal."** Flagged as DERIVED-AT-CONFIG owed to the synthesis (R1-basin
  quantile), NOT a literal — a value-provenance-ladder obligation, not a claim.
- **Provenance honesty:** every number tagged. MEASURED: 697,941 B / 1,049 B / 7,195 B / rates / R1 d_pose
  0.001610 / LM ρ 0.847-0.868 / #341 timing / trainer LOC. DERIVED: σ_min threshold mechanism. ESTIMATED:
  Modal per-row cost ~$0.25 (provider-rate estimate, not a measured invoice). Pointer 0.19110 UNMOVED.
