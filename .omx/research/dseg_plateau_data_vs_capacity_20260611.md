# The d_seg plateau: DATA-limit vs CAPACITY-limit — the verdict that decides the 600-pair capstone bet (2026-06-11)

**Subagent:** `dseg_plateau_data_vs_capacity`. **Mode:** ANALYSIS — read-only + reasoning + two TINY bounded
numeric checks (a geometric-asymptote fit on the live trajectories + a params/frame capacity arithmetic). **NO
GPU training launched; NO MPS; the live daemons + `mlx_scorer_adapters.py` + `capstone_trainer.py` untouched.**
**Authority of every number:** `[macOS advisory]` / `[macOS-MLX research-signal]` — each is a MEASURED
trajectory row (cited inline) or a closed-form arithmetic shown inline. NON-PROMOTABLE per the GOAL authority
ladder: `promotable=false`, `score_claim=false`, `ready_for_exact_eval_dispatch=false`. `$0` spend, NO cloud, NO
paid GPU, NO /tmp. Frontier pointer UNMOVED: **0.19109982 [contest-CPU], 177,169 B.** This memo emits NO archive
— it is a verdict + a recipe-recommendation + a pre-registered confirming ablation.

NO FAKE: I do NOT fabricate a multi-pair-count d_seg curve. **The decisive cross-pair-count comparison the prompt
asked for DOES NOT EXIST as a clean measured anchor** (the `b16_n100` / `b20_n100_perframe` daemons died at 0
logged rows — the session-watcher trap; the `*_n8` smokes are EMA-shadow-frozen at 0.507 = init, useless). I say
so plainly and reason from the anchors that DO exist + first-principles single-video-NeRV capacity scaling. The
verdict is therefore a REASONED verdict with a named confounder and a cheap ablation to close it — not a measured
certainty.

---

## 0. HEADLINE

**VERDICT: predominantly CAPACITY-limited, and the data axis works AGAINST base_ch=20 at 600 pairs.** The
48-pair plateau is NOT the "small basis is data-starved, more pairs will rescue it" story the crux memo bet on.
The decisive arithmetic: at 48 pairs base_ch=20 already carries **884 params/frame — 6× the frontier's per-frame
capacity** — and STILL floors at d_seg ≈ 0.0073–0.0097 (13–17× above the 5.6e-4 frontier). Going to 600 pairs at
fixed 85K params DROPS per-frame capacity to **71 params/frame = 0.48× the frontier's**, on the SAME 1200 frames
the frontier needed ~178K params to reach 5.6e-4. More pairs at fixed 85K makes the single-video fit HARDER, not
easier. There is **no measured anchor showing 85K reaches the frontier d_seg**, and the crux memo's load-bearing
"flat 88K–180K basin" is a MIS-READ of the floor memo (whose cluster is 5 PRs ALL at ~178K params — one param
band, not a param sweep).

**IMPLICATION:** base_ch=20 @ 600 pairs is the WRONG default first run — its predicted S spans 0.14 (only if the
unmeasured frontier-floor miracle happens) to 0.28–0.93 (if the 48-pair plateau is the per-frame ceiling, which
the evidence favors). The CAPACITY-limited recommendation: **base_ch=24 (114,710 params = frontier-class, the
LARGEST that byte-fits sub-0.19 at int8) @ 600 pairs**, predicted S ≈ 0.158 IF it reaches the frontier floor;
base_ch=22 (99K, S ≈ 0.150 at floor) is the sub-0.15 edge bank. base_ch=20 is a research probe, not the bet.

**THE CHEAP DECISIVE TEST (one afternoon on the GPU-fast loop):** a 2×2 ablation that ISOLATES data from capacity
— `{base_ch=20, base_ch=24} × {48 pairs, 192 pairs}`, CE-only, equal epochs-per-pair, LIVE (use_ema=False)
d_seg. Pre-registered: if the plateau d_seg DROPS when pairs rise at fixed base_ch → data-limited (crux memo
right, run base_ch=20@600). If the plateau d_seg RISES or holds when pairs rise at fixed base_ch, AND drops when
base_ch rises at fixed pairs → capacity-limited (this memo right, scale to base_ch=24).

---

## 1. THE LIVE EVIDENCE (what is actually measured, EMA-fix-validated)

All clean LIVE-descent anchors are **base_ch=20, 48 pairs** (the only configs that produced real descending rows):

| run | carrier / FiLM | curriculum | epochs | LIVE d_seg reached | shape |
|---|---|---|---:|---:|---|
| `capstone_c1prime_honest_b20_n48` | stored_latent / shared | pr95 8-stage (CE→softplus→smooth→qat→c1a) | ~150 (st1-5) | **0.00968** (stage-2 conv) | stage2 geometric-converges to 0.00968 (q≈0.49); **stages 3-5 FLAT/REVERSE 0.0097–0.0102** |
| `capstone_daemon_b20_n48_LONG` | per-frame | CE-only | 50 | **0.00801** (still falling) | geometric asymptote **≈ 0.0073** (q≈0.65) |
| `capstone_daemon_b20_n48_stream` | per-frame | CE | 14 | 0.01058 (falling) | early descent |

Three load-bearing facts from the c1prime run.log + trajectory.jsonl:

1. **The c1prime plateau is a TRUSTWORTHY LIVE value, not EMA-shadow-lag.** By the end of stage-2 the EMA has had
   ~700 updates; the warmup decay `min(0.997,(1+t)/(10+t))` gives effective_decay ≈ 0.987 → time-constant ≈ 79
   steps; stage-2 ran ~456 steps = >5 time-constants. The shadow is well-tracked. (The 0.505-frozen
   `curriculum_b20_n48` run is the PRE-fix shadow-lag artifact, commit `f771e6e00` — correctly excluded.)
2. **The PR95 margin-surrogate schedule DID re-accelerate past the CE knee — but only ONCE, then STALLED.**
   Stage-1 CE: 0.505→0.0198. Stage-2 tau_softplus: 0.0198→**0.00968** (the bounded surrogate's reallocation,
   exactly the crux-memo mechanism — CONFIRMED). **But stage-3 `smooth_disagreement` (the loss whose minimizer
   IS d_seg) REVERSED to 0.0102, and stage-4 QAT + stage-5 c1a held flat.** The crux memo PREDICTED stages 3–5
   would keep driving toward 5.6e-4; they did NOT. The schedule's descent EXHAUSTED at ~0.0097 on 48 pairs.
3. **More conditioning CAPACITY lowered the plateau.** The per-frame-FiLM LONG run (CE-only, more per-frame
   conditioning params) asymptotes ≈ 0.0073 — LOWER than the shared-FiLM full-curriculum 0.0097. Adding capacity
   (per-frame vs shared FiLM) moved d_seg DOWN. That is a capacity-sensitivity signal in the data we have.

---

## 2. THE DATA-vs-CAPACITY ARITHMETIC (the decisive reasoning)

`d_seg = mean(argmax SegNet(render f1) != argmax SegNet(GT f1))` over 384×512 × N_frames — a per-pixel argmax-
flip rate on a SINGLE video. A NeRV/HNeRV decoder of P params memorizes the GT argmax field. The relevant
capacity unit is **params-per-frame** (how much representational budget the decoder can spend per frame it must
reproduce). At FIXED P, MORE frames = LESS budget/frame = a HARDER single-video overfit (the inverse of the
"more data helps generalization" intuition — single-video memorization has no held-out set; more frames is
strictly more to memorize at fixed capacity).

| config | params P | frames N | **params/frame** | d_seg |
|---|---:|---:|---:|---:|
| **frontier (PR101/A1)** | 178,262 | 1,200 | **148.6** | **5.6e-4** (MEASURED) |
| capstone @ 48 pairs | 84,901 | 96 | **884.4** | ~0.0073–0.0097 (MEASURED, LIVE) |
| capstone @ 600 pairs | 84,901 | 1,200 | **70.8** | UNMEASURED |
| base_ch=24 @ 600 pairs | 114,710 | 1,200 | 95.6 | UNMEASURED |

Two facts kill the naive data-limited thesis:

- **The 48-pair plateau is NOT a per-frame-capacity starvation.** At 48 pairs base_ch=20 has 884 params/frame =
  **6× the frontier's per-frame budget**, yet d_seg is **17× WORSE** than the frontier. If the architecture COULD
  represent the argmax field, 6× the frontier's per-frame budget on only 96 frames should drive d_seg toward
  zero. It floors at ~0.0073–0.0097 instead. The ceiling at 48 pairs is an ARCHITECTURE/representational ceiling
  of the base_ch=20 decoder family at this conditioning, not a data shortage. (A pure data-limit would predict
  near-zero d_seg at 884 params/frame and 96 frames — the opposite of what we see.)
- **600 pairs makes base_ch=20 WORSE on the capacity axis, not better.** At 600 pairs base_ch=20 drops to 71
  params/frame = **0.48× the frontier's** — for the SAME 1200 frames the frontier needed ~178K params (2.1×
  more) to reach 5.6e-4. The crux memo's hope is that base_ch=20 reaches the frontier d_seg with HALF the
  per-frame capacity the frontier required on identical data. That is the "85K reaches what 178K needed" leap the
  synthesis memo correctly flagged as motivated optimism — and the EMA-fix did NOT reverse this part (the EMA-fix
  reversed the FROZEN-0.505 artifact, not the param↔d_seg physics).

### 2.1 Why the crux memo's "flat 88K–180K basin" is a mis-read (the central error to correct)
The crux memo's Pareto rests on: *"PR101/PR102/PR103/A1 ALL cluster at d_seg ≈ 5.6e-4 across 88K–180K params — a
FLAT BASIN; the d_seg floor does NOT improve from 88K→180K."* **The floor memo
(`grand_council_fields_medal_theoretical_floor_20260509.md`) §observation table does NOT say this.** Its 5-PR
cluster (PR100/101/102/107/A1) is **ALL at B ≈ 178K params** (177–179K) — one param band, NOT a param sweep. The
floor memo's own council reading is the OPPOSITE of "flat basin": Dykstra: *"the convex feasibility region is
empty at S<0.155 with the current parameter cardinality (88K–180K)... we have hit the expressivity ceiling of
THIS parameter budget... to open the region requires adding coordinates."* Quantizr: *"ADD params. 88K→256K
shrinks d_seg 2× to 2.8e-4."* The "88K" anchor is Quantizr's 0.33-scoring archive — a DIFFERENT architecture
(FiLM-DSConv + AV1 masks) that never measured frontier d_seg=5.6e-4; it scored 0.33, not 0.19. **So there is NO
measured d_seg anchor at 85–100K params reaching the frontier floor.** The crux memo conflated "88K-param
architectures exist in the literature" with "88K params reach d_seg 5.6e-4." The floor memo's actual claim is a
capacity-CEILING claim, and it points UP (add params), not down.

---

## 3. THE CONFOUNDER I MUST NAME (honest — why this is a reasoned, not proven, verdict)

The 48-pair plateau has TWO candidate causes I cannot fully separate with the data on disk:
- **(A) capacity ceiling** (this memo's verdict): base_ch=20's decoder family cannot represent the SegNet argmax
  field below ~0.0073 regardless of pairs/epochs. Supported by: 6× frontier per-frame budget still 17× worse;
  per-frame-FiLM (more capacity) reaching lower; the floor memo's "expressivity ceiling, add params."
- **(B) epoch starvation of the late stages** (the residual data/optimization-limit story): the LONG CE run was
  *still descending* at ep50 (asymptote ~0.0073 by geometric fit, but UNCONVERGED), and the c1prime stages 3–5
  ran only 7–20 epochs each — possibly too few for the bounded surrogates to keep biting. Under (B), MANY more
  epochs at base_ch=20 might still creep lower.

**Why (A) dominates even under (B)'s most generous reading:** the LONG geometric asymptote is ~0.0073 — even if
the late stages converge perfectly to that, 0.0073 is **13× above 5.6e-4** → seg_term 0.73 → S ≈ 0.80 at base_ch=20's
rate. (B) would have to be wrong by 13× — i.e. the asymptote estimate would have to be catastrophically
pessimistic AND the late-stage schedule would have to find a second re-acceleration that 150 epochs of c1prime
(including the exact `smooth_disagreement` loss whose minimizer IS d_seg) did NOT find. The c1prime stage-3
REVERSAL (d_seg went UP under the d_seg-exact surrogate) is the strongest single datum against (B): when you
optimize the EXACT d_seg surrogate and it gets WORSE, you are at a representational limit, not an epoch limit.
The 2×2 ablation (§5) is designed to falsify this verdict cheaply if (B) is in fact right.

---

## 4. THE IMPLICATION FOR THE NEXT RUN (the bet)

Score model `S = 100·d_seg + √(10·d_pose) + rate`; d_pose held in the tube (≈3e-5) by the stored_latent carrier
(c1prime-confirmed d_pose→1.4e-4 by stage-2 and still falling — pose is NOT the binding axis). Byte budget
(MEASURED int8, `capstone_vq_nerv_byte_budget_20260610.json`): base_ch=20 → rate 0.067; base_ch=24 →
126,410 B / rate 0.0842; sub-0.19 ceiling rate<0.077 ⇒ total<115,640 B ⇒ **base_ch≤22 byte-fits sub-0.19;
base_ch=24 byte-fits sub-0.19 only if d_seg/d_pose are excellent** (126,410 B alone is rate 0.084).

Predicted S under the floor hypotheses (pose tube 3e-5):

| config | rate | IF d_seg=5.6e-4 (frontier floor) | IF d_seg=2e-3 (capacity-scaled likely) | IF d_seg=0.0085 (48-pair plateau holds) |
|---|---:|---:|---:|---:|
| base_ch=20 (85K) | 0.067 | **0.140** ✓sub-0.15 | 0.284 | 0.934 |
| base_ch=22 (99K) | 0.077 | **0.150** edge | 0.294 | 0.944 |
| base_ch=24 (114K, frontier-class) | 0.084 | 0.158 (sub-0.19) | 0.302 | 0.942 |

**The verdict's recommendation: do NOT run base_ch=20 @ 600 pairs as the bet.** Its sub-0.15 outcome requires the
single unmeasured miracle (85K reaching the frontier floor that needed 178K). The CAPACITY-limited play:

- **PRIMARY: base_ch=24 @ 600 pairs** (114,710 params = frontier-class, the most per-frame capacity that still
  byte-fits sub-0.19 at int8). It is the config with the BEST chance of actually reaching d_seg≈5.6e-4 because it
  is closest to the frontier's measured per-frame capacity (95.6 vs 148.6 params/frame — still below, but 2× more
  than base_ch=20). Predicted S ≈ 0.158 at the frontier floor; this BANKS sub-0.19 if d_seg lands ≤ ~1.1e-3.
- **BANK: base_ch=22 @ 600 pairs** if 126KB feels too tight — 99K params, rate 0.077, S ≈ 0.150 at the frontier
  floor (the sub-0.15 edge). Higher per-frame capacity than 20, fits sub-0.19 with margin.
- **base_ch=20 is a RESEARCH PROBE**, run ONLY inside the §5 ablation to MEASURE its 600-pair floor — not as the
  pointer bet. If the ablation shows base_ch=20 surprisingly reaches ≤1e-3 (data-limited after all), it becomes
  the rate-optimal point and the crux memo is vindicated; until then it is the high-variance option.

Note this REVERSES the crux memo (base_ch=20, S 0.140) back toward the synthesis memo's "need frontier-CLASS
params (≈100–115K)" — but for a CLEANER reason than the synthesis gave: the synthesis read the shadow-frozen
d_seg, this memo reads the EMA-FIXED live plateau + the params/frame physics. The synthesis's strategic
conclusion ("frontier-class params, not a smaller basis") survives the EMA-fix; only its EVIDENCE (shadow d_seg)
was wrong. The crux memo's strategic conclusion (smaller-is-better) does NOT survive the params/frame arithmetic.

---

## 5. THE CHEAP DECISIVE ABLATION (run on the GPU-fast loop once the wire-in lands; pre-registered)

A 2×2 that ISOLATES the two axes the prompt named, holding everything else fixed (carrier=stored_latent or
per-frame CE — pick ONE; seed=0; LIVE `use_ema_for_eval=False` so no shadow confound; equal **epochs-per-pair**
so total optimizer-steps-per-frame is matched across pair-counts):

| arm | base_ch | pairs | what it tests |
|---|---:|---:|---|
| A | 20 | 48 | baseline (reproduces 0.0097) |
| B | 20 | 192 | **DATA axis** at fixed capacity (4× pairs, same params) |
| C | 24 | 48 | **CAPACITY axis** at fixed data (frontier-class params, same pairs) |
| D | 24 | 192 | both |

Run CE-only to ~convergence per arm (geometric-delta ratio < 0.2), report mean LIVE d_seg over the last 5 evals.
Affordability (MEASURED `capstone_training_throughput_profile`: 14.28s/step, 98% torch-CPU scorer; 48p=6
steps/ep, 192p=24 steps/ep): at ~80 epochs each, A≈1.9h, B≈7.6h, C≈1.9h, D≈7.6h ⇒ ~19h total on the CPU scorer
(or far less on the GPU-fast loop the prompt anticipates). Use marker-on-exit per daemon (the session-watcher
trap killed the last 3).

**Pre-registered predictions (the verdict's falsifiable commitment):**
- **CAPACITY-limited (this memo, expected):** plateau(B) ≈ plateau(A) or HIGHER (more frames, same params, harder
  fit); plateau(C) < plateau(A) by a clear margin (frontier-class params lower the floor); plateau(D) ≈
  plateau(C) (capacity dominates). Decision: **scale to base_ch=24 @ 600**; base_ch=20 deprecated as the bet.
- **DATA-limited (crux memo):** plateau(B) << plateau(A) (more pairs sharply lower d_seg at fixed 85K) AND
  plateau(C) ≈ plateau(A) (params don't matter). Decision: **run base_ch=20 @ 600** (cheapest, sub-0.15).
- **Both (mixed):** plateau(B) modestly lower AND plateau(C) modestly lower. Decision: base_ch=22 @ 600 as the
  balance (some of each axis), re-fit the param↔d_seg slope from the 2 base_ch points and pick the byte-optimal.

The single number that decides it: **the sign of `plateau(B) − plateau(A)`** (the data axis at fixed capacity).
If negative and large → data-limited. If ≥0 → capacity-limited (more pairs did not help at fixed 85K, so 600
pairs at base_ch=20 will not reach the floor). This is the one measurement worth the GPU time before committing
the multi-day 600-pair run.

---

## 6. WIRE-IN (Catalog #125) + SCOREBOARD

1. **sensitivity-map — ACTIVE.** New prior REVERSING the crux memo: the d_seg floor is NOT flat across 88–180K;
   the only measured frontier-d_seg anchor is at ~178K, and base_ch=20 @ 600 pairs sits at 0.48× the frontier's
   per-frame capacity. The rate lever is the SMALLEST base_ch that *actually reaches* the d_seg floor — which the
   capacity arithmetic says is ≥ base_ch=22–24, not base_ch=20. Aiming surface: base_ch=24.
2. **Pareto — ACTIVE.** Adds the constraint that d_seg(base_ch) is a CAPACITY curve (decreasing in params toward
   the frontier floor), NOT a flat basin — so the byte-vs-d_seg trade is real, not free. base_ch=20's rate
   advantage (0.067 vs 0.084) is only worth taking IF its d_seg floor ≤ ~1.7e-3 (else seg_term swamps the
   rate saving); the §5 ablation measures whether it does.
3. **bit-allocator — ACTIVE.** Allocator should NOT shrink the decoder to base_ch=20 on the assumption d_seg is
   capacity-free; reserve frontier-class decoder bytes (base_ch=24 ≈ 113KB int8) until the ablation proves a
   smaller decoder holds the floor.
4. **cathedral-autopilot — gate-conditional.** The §5 2×2 ablation → (verdict) → ONE 600-pair daemon at the
   verdict's base_ch → (advisory beats frontier / sub-0.15) → ONE paired exact CPU+CUDA eval is the dispatch
   surface. Do NOT dispatch a 600-pair base_ch=20 run before the ablation.
5. **continual-learning — ACTIVE.** Reseeds the V3 judge: (a) the c1prime EMA-fixed plateau (~0.0097, stage-3
   REVERSAL) is a TRUSTWORTHY live value, not shadow-lag; (b) the "flat 88–180K d_seg basin" is a MIS-READ of the
   floor memo (one 178K param band, not a sweep) — there is NO measured 85K→5.6e-4 anchor; (c) single-video NeRV
   capacity scales as params/frame; 600 pairs at fixed 85K HURTS the fit; (d) the synthesis memo's "frontier-class
   params" conclusion survives the EMA-fix; the crux memo's "smaller-is-better" does not.
6. **probe-disambiguator — PARTIALLY RESOLVED + ONE OPEN.** "Is the 48-pair plateau data or capacity?" → REASONED
   capacity (with named epoch-starvation confounder); the OPEN probe is the §5 2×2 ablation whose `sign(plateau(B)
   − plateau(A))` is the empirical arbiter. "base_ch=20 or scale?" → scale to 24 (bank 22) pending the ablation.

**UPPER (vs T_1 sub-0.19):** unchanged — analysis memo, no archive. Frontier holds 0.19110 [contest-CPU].
**LOWER (the floor):** the capacity arithmetic says base_ch=24 (frontier-class) is the right vehicle to actually
*reach* d_seg≈5.6e-4 and thus the sub-0.15 distortion-at-budget door; the sub-0.118 reach remains a later
decoder-shrink campaign and is NOT contradicted, only deferred until a smaller decoder is PROVEN to hold the
floor (not assumed to).

## 7. CROSS-REFERENCES
`dseg_crux_objective_and_param_pareto_20260611.md` (the memo this PARTIALLY REVERSES — its objective/schedule
analysis stands; its "flat basin → base_ch=20" Pareto does not survive the params/frame physics + the floor-memo
re-read) · `capstone_adversarial_synthesis_and_honest_corrections_20260611T015018Z.md` (the "frontier-class
params, not a smaller basis" conclusion this memo REVIVES on cleaner EMA-fixed evidence) ·
`capstone_ema_shadow_lag_reverses_seg_wall_verdict_20260611T070000Z.md` (the EMA-fix; validates the c1prime
plateau as live; correctly reopens the small-basis question — which this memo answers as "still capacity-bound,
just not via the 0.505 artifact") · `grand_council_fields_medal_theoretical_floor_20260509.md` (the 5-PR cluster
ALL at ~178K params — the source the "flat basin" claim mis-read; its council reading is "expressivity ceiling,
ADD params") · `capstone_vq_nerv_byte_budget_20260610.json` (MEASURED param↔byte: base_ch 20/24 int8 rows) ·
`experiments/results/capstone_c1prime_honest_b20_n48/{trajectory.jsonl,run.log}` (the LIVE EMA-fixed plateau:
st1 0.505→0.0198, st2→0.00968, st3 REVERSAL→0.0102, st4-5 flat) ·
`experiments/results/capstone_daemon_b20_n48_LONG/trajectory.jsonl` (per-frame CE, asymptote ~0.0073) ·
`capstone_training_throughput_profile_20260611T051024Z.json` (14.28s/step, 98% CPU scorer — the ablation
affordability) · `GOAL_standing_v3_20260610.md` (lever C; the sub-0.15 ladder) ·
`upstream/{modules.py,evaluate.py}` (frozen authority; `d_seg` = per-pixel argmax-flip).
