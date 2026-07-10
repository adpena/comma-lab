# NEG-AUDIT RE-TESTS — C1 (directional −48% transfer) + E5 (fire sR) — 2026-07-09

**Task:** #390 follow-up — audit the two cheapest HIGH-reactivation negatives in
`.omx/research/negative_findings_register_20260709/auditor_A_dag_research.md` (items C1 + E5).
**Axis:** `[macOS-CPU advisory . REALIZED-through-R CPU-SegNet . NON-PROMOTABLE]`, 16 strided pairs
(stride 37, idx 0..555), subset (non-authority). **$0, CPU, cached-authority artifacts only.**
**Pointer contest-CPU 0.19110 UNMOVED — this is MEANS.** Result JSON:
`.omx/research/negaudit_retests_c1_e5_result_20260709.json` (sha256 0eb56daf…), reproducer
`negaudit_retests_c1_e5_measure_20260709.py`.

**STORES CONSULTED:** `docs/operating_manual_craft_handoff.md` (§4 re-derive-from-primary · §5 label ·
§8.5 no-borrowed-number) · the register (C1+E5 rows + TOP-10) · DAG FEED-25t/bh/bk/bl/ah/cap1/cs/cv ·
memory `msal_uni_texture_proxy_inert_build_exact_sR_reachability_weight_20260703` · the mod32cap run
dir (read-only) · `tac.through_r.measure_through_r` (canonical harness) · gt_n600.npz + gt_n600_sR.npz.

---

## ITEM 2 (primary) — E5: FIRE sR vs a REAL checkpoint's realized flips — FIRST MEASURED ROW

**The lever (#268):** `--margin-saliency-reachability` multiplies the margin-saliency loss weight by
the through-R fragility-Jacobian `sR = |d(Σ w·margin)/dx|`, `w = exp(−margin/τ)` — the EXACT signal the
inert texture proxy (`1/(1+β·tex)`, Pearson −0.033 vs reachability) was meant to be. `gt_n600_sR.npz`
was BUILT and READY but the lever had NEVER fired (never-fired queue). The prior anchor only measured
sR vs the θ-independent margin geometry — never against a **real checkpoint's actual realized flips.**

**Checkpoint:** mod32cap `BEST_ep300` realized-through-R verdict argmax (self-orient ON, chroma,
w_pose=0), 16 strided pairs. Flip field = `(realized_argmax ≠ L*)`; validated: subset realized
d_seg = **0.004785** ≈ the checkpoint's reported 0.004783 (pair-alignment + realized-field confirmed).

| signal vs REAL flips | Pearson | AUC | top-5% Jaccard (chance 0.0044) | flip-mass@top-5% | @top-10% |
|---|---:|---:|---:|---:|---:|
| **sR** (the never-fired reachability weight) | **+0.076** | **0.767** | 0.0197 (4.5× chance) | 0.221 | 0.355 |
| **w = exp(−margin/τ)** (PRIMARY factor, ALREADY in the loss) | +0.474 | **0.991** | 0.093 (21× chance) | **0.973** | — |
| −margin (simplest) | — | 0.991 | — | 0.973 | — |
| texprox (anchor baseline) | −0.033 vs sR | (at chance) | 0.024 | — | — |

**sR geometry cross-check (reproduces the anchor):** sR·margin −0.348 (anchor −0.323), sR·|∇margin|
+0.302 (anchor +0.272), sR·w +0.217. Method validated.

**VERDICT (E5 — first-fire characterization; verdict_scope INSTANCE: 16 pairs, ONE checkpoint):**
- **sR is NOT inert.** Unlike texprox (Pearson −0.033 = chance), sR is an **above-chance predictor of
  the real checkpoint's flips** (AUC 0.767, Pearson +0.076, Jaccard 4.5× chance). Building sR over the
  inert texture proxy is **vindicated** — sR carries genuine flip-predictive reachability signal.
- **BUT sR is strongly DOMINATED by the primary factor w it multiplies onto.** `sal = w · sR`, and w
  alone has AUC 0.991 and captures **97% of flip-mass in its top-5%** vs sR's 22%. Since w already
  near-perfectly localizes the flip band, sR's marginal reweighting value is **modest** — exactly the
  memory anchor's pre-registered honest scope ("SECONDARY multiplier → MODEST refinement, not a step
  change; the primary w already carries the fragility alignment"). CONFIRMED, now MEASURED.
- **Owed:** the training lever `--margin-saliency-reachability` is still **NEVER-FIRED in a run** — the
  byte-closed n600 A/B (does sR·w LOWER exact d_seg vs w-alone?) remains an operator-GO #205-class arm.
  This $0 field-characterization sharpens the PRIOR (modest, above-chance) but does not replace it.

---

## ITEM 1 — C1: directional all-class −48% basis — realized-transfer + transfer ceiling

**The claim under audit:** the −48% all-class directional Fourier basis is credited as the #1 d_seg
lever + "next-run PRIMARY" (MEMORY L25), but rests on a DIRECT-partition proxy; realized-through-R
transfer never cleanly measured (register C1: proxy-mirage ~170–350× off; HIGH-reactivation).

**Primary-artifact findings (verified, $0):**
1. **Direct-partition Δ CONFIRMED −50.8%** (n600, existing smoke_result.json): iso control 0.007476 →
   directional 0.003679. The credited "−48%" is real ON THE DIRECT-PARTITION (argmax-of-generator-
   logits vs GT argmax) axis.
2. **The clean realized-axis dir-vs-iso A/B (FEED-bk's named "$0 directional-axis-reality check")
   STILL never completed.** `experiments/results/dir_iso_axis_check_20260626T164308Z/`: the ISO arm
   finished (`arm_iso_v2`), the DIRECTIONAL arm **CRASHED** — `directional_fourier_feats` broadcast
   error (render-grid 98304 vs seg-grid 196608 px) — AND the daemon flagged the GT-tangent directional
   basis **NOT byte-closeable / research-only** (tangent needs GT SegNet argmax at decode). ⇒ **the
   through-R transfer ratio is UN-computable from existing artifacts** (no realized directional row
   exists). Register C1 "realized transfer UNVERIFIED" **CONFIRMED with the crash receipt.**
3. **Byte-closeability EVOLVED (register partly stale):** self-orientation (iso→own-argmax→tangent,
   cos 0.89–0.91 vs GT, FEED-cs) resolved the decode blocker; the self-orient directional IS wired into
   the levelset trainer and RAN on the realized axis (mod32cap, self-orient ON → realized d_seg 0.0048).
   **BUT** it never ran a matched self-orient-OFF control at n600 → **the realized DELTA of the
   directional lever is STILL unmeasured**; and the measured form had aliasing (only freq octaves
   {32,64} clean) + freq-along starvation (3.2× along-tangent deficit, MEMORY L25).

**Transfer CEILING (my new measured row, canonical `measure_through_r`, 16 strided pairs):**
Realize the PERFECT direct partition (GT L*) as palette RGB (per-class mean RGB) → R → frozen CPU-torch
SegNet → realized d_seg **F = 0.0337** (per-class: Movable 0.396, Lane 0.152, MyCar 0.090, Road 0.018,
Undrivable 0.002). Reproduces FEED-ah's ~0.005–0.008 order-of-magnitude claim with the canonical harness
(the class-3/1 thin-class palette collapse pushes the mean up).

- **F = 0.0337 is ~4.5× the iso direct (0.0075) and ~9× the directional direct (0.0037).** Under NAIVE
  palette realization the entire −50.8% direct improvement lives **BELOW** the realized floor → the
  direct-partition axis (where the −48% was measured) is a **MIRAGE for naive direct→palette realization**
  (FEED-ah, re-confirmed via the authority harness).
- **CAVEAT (load-bearing, no over-claim):** the palette ceiling is NOT a floor for a TRAINED-through-R
  witness. The mod32cap trained witness reaches realized d_seg **0.0048** — 7× BELOW the 0.0337 naive-
  palette ceiling (chroma-slack). So the ceiling kills the NAIVE direct→palette route, NOT the directional
  basis on the trained realized axis. The directional lever's realized value survives ONLY via a
  trained-through-R self-orient witness — which is exactly the still-owed clean A/B.

**VERDICT (C1 — verdict_scope FORMULATION/INSTANCE, NO new kill):** C1's grade STANDS and is SHARPENED:
the −48% is a CONFIRMED **direct-partition** number whose **realized-axis transfer remains UNVERIFIED**
(the clean dir-vs-iso realized A/B crashed and was never re-run to a matched pair). The byte-closeability
objection is RESOLVED by self-orient (register partly stale on that leg), but the realized Δ is still
owed. The naive-palette ceiling (0.0337 ≫ 0.0037) confirms direct-partition quality is a mirage for
naive realization; the trained-through-R self-orient witness (0.0048) is the only regime where the lever
can matter, and there its isolated contribution has NEVER been measured. **The decisive re-test remains a
self-orient-ON-vs-OFF trained-through-R n600 A/B (heavy; operator-GO), not a $0 pass.** "Next-run PRIMARY"
(MEMORY L25) is over-credited: the lever is proxy-confirmed, realized-unverified, and aliasing/freq-along
-caveated.

---

## Triality legs
- **DAG:** FEED-negaudit-retests (this landing).
- **Equations:** EmpiricalAnchor appended to `margin_saliency_reachability_replaces_texture_proxy_v1`
  (E5 first-fire) + `curvelet_directional_basis_dseg_reduction_v1` (C1 transfer ceiling).
- **DSL/activation ledger:** `margin_saliency_reachability` gets its first `measured` characterization
  row (honest reason: $0 field-analysis, above-chance-but-w-dominated; training-lever still NEVER-FIRED,
  byte-closed A/B owed).

**Pointer 0.19110 UNMOVED (means).**

---

## C1 UPDATE (2026-07-10, appended): the decisive re-test RAN — realized Δ MEASURED ≈ 0

The self-orient-ON-vs-OFF trained-through-R n600 A/B (operator-GO, bounded warm-start formulation)
completed. Matched clean cells: ep650 Δ(OFF−ON) −0.000089 (−1.39% of ON) · ep675 −0.000015 (−0.35%);
OFF ep700 0.004181 (OFF BEST); ON ep700 cell BLOCKED (resume projects 93.8 GiB > idle headroom; 6×
governed REFUSE, never bypassed). **Realized directional contribution ≈ ZERO (OFF marginally better,
INSTANCE-noise).** C1's grade is now CLOSED-AT-THIS-FORMULATION: the −48% is direct-partition-real but
realized-through-R transfer is NOT OBSERVED in a 50-ep warm-start fine-tune; the proxy-mirage reading
(~170–350× off) is CONFIRMED by direct measurement. verdict_scope FORMULATION (from-scratch arm remains
the one uncovered reformulation; freq-along starvation 3.2× is a live confound of the tested
formulation). Verdict artifact: `.omx/research/owed16_verdict_20260710.json`. MEMORY L25 "next-run
PRIMARY" is now measured-DOWN: the lever costs ~47–57 GiB RAM for no measured realized d_seg at this
formulation. Pointer 0.19110 UNMOVED.
