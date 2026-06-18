# Closure RE-AUDIT — fresh math/geometry/janky-prototype lens on the historical kill/defer/orphan ledger — 2026-06-18

**Subagent:** adversarial RE-AUDIT (READ-ONLY + light bash). **Evidence grade:** `[macOS-CPU advisory]` /
structural audit. NO score claims, NO dispatch, `promotable=false`, $0 spend. Frontier UNMOVED (pointer
`tools/refresh_canonical_frontier.py`: contest-CPU **0.19109982** sha `b46897267d…` 177,169 B; contest-CUDA
0.20533 sha `9cb989ce…`). Goal `S<0.15` UNSATISFIED.

**Trigger (operator, 2026-06-17):** this session a "falsification" (PoseNet low-rank pose coding) was proven
MATH-WRONG — closed at the wrong operating point, reopened as a ~2.7× real win. Re-audit ALL
negatives/deferred/killed for the same fallibility, through a fresh **math/algebra/geometry/calculus +
janky-prototype** lens, biased to the closures that — if wrongly closed — most help the CURRENT **bc20
HNeRV small-basis vehicle** toward sub-0.15.

**Scope:** the HISTORICAL kill/defer/orphan ledger (this session's fresh probes are a sister subagent's).
**No-duplicative-work:** this UNIFIES + RE-AUDITS five prior closure maps through the FRESH lens — it does
NOT rebuild them: `pre_rigor_kill_defer_falsified_inventory_20260517.md` (34 verdicts, 28-32 fail today's
rigor), `meta_bug_retroactive_defer_kill_falsify_audit_20260519T044057Z.md` (15 META-bug taint map),
`deferral_recovery_ledger_20260610T130200Z.md` (Catalog #307 audit CLEAN, recovery=sequencing), the
`evaluator_inverse_orphan_inventory_20260609.md` (103-surface map), and the 2026-06-16 witness-seg-boundary
TOP-AIML reopen (NO_GO_SURVIVAL_WALL). The net-new value is the **per-closure FRESH-LENS classification
keyed to the bc20 vehicle's actual binding axis (d_seg) + its two free-byte axes (rate, pose-null).**

---

## 0. THE VEHICLE-ANCHORED FRAME (why these rankings differ from the month-old audits)

The bc20 small-basis vehicle's score structure (operator reframe + master-gradient ledger):

- **d_seg is the BINDING constraint** — ~90.8% of byte sensitivity is seg-flip protection
  (`master_gradient_anchors.jsonl`: 161,779/178,158 threshold-dominant seg bytes; marginal seg=100, pose≈37.8,
  rate≈6.66e-7). Sub-0.15 needs d_seg < ~0.000322 (≈7-8× from the bc20 floor 0.00225-0.00260).
- **rate + pose-null are where FREE bytes live.** The bc20 small basis has rate HEADROOM (the asset);
  capacity-scaling FORFEITS it.
- **THE pose-low-rank canary** (`project_posenet_rank1_discovery.md`): PoseNet Jacobian rank ≈ 1.008, dim-0
  = 99.8% variance; optimal pose conditioning is a **scalar radial zoom from FoE (256,174)** — 600 FP16
  scalars (1.2 KB) capturing 99.8% of the pose signal vs 50K-param dense MotionPredictor. THIS is the math
  that reverses the pose-coding closures: every pose codec killed for "overparameterized / pose-blind" was
  closed against a rank-6 assumption that the geometry falsifies.

**Operating-point fallibility test applied to every closure below:** *"Was this closed against an operating
point / fidelity / substrate / objective that the bc20 small-basis vehicle does NOT share?"* If yes →
REOPEN (the canary pattern).

---

## 1. RANKED REOPEN LEDGER (score-EV-per-effort)

Verdict legend per CLAUDE.md Catalog #307: **REOPEN-MATH** (closing math is wrong at the vehicle's operating
point — the canary class) · **REOPEN-IMPLEMENTATION** (paradigm intact; closed on janky prototype / wrong
config / single un-swept point — top-AIML re-attempt warranted) · **SOUND-KILL** (math/empirical genuinely
exhausts it AT this vehicle's operating point too).

### TIER A — directly attacks the bc20 binding axis or its free-byte axes; cheap $0 re-validation

| # | closure (memo) | original verdict + cited reason | FRESH-LENS re-derivation | classify | score-EV | cheapest $0 re-validate | top-AIML the prototype lacked |
|---|---|---|---|---|---|---|---|
| **R-1** | **PoseNet low-rank / radial-zoom pose codec** (`project_posenet_rank1_discovery.md` + `feedback_preprocessing_dead_end.md` "ALL pose conditioning overparameterized") | pose codecs deferred/abandoned as "overparameterized" / "preprocessing kills PoseNet"; the rank-1 finding marked **"Theory"** (never byte-closed) | **REOPEN-MATH — this is the canary itself.** PoseNet rank≈1.008 (dim-0=99.8% var) is a GEOMETRIC fact: pose lives on a 1-D radial-zoom manifold from the FoE. Every "pose codec overparameterized" kill assumed a 6-DOF target. The 2.7× win the operator cites IS this. For bc20: the joint frame1 carrier (lever C, the LIVE blocker) only needs to carry **1 pose DOF**, not 6 → the pose half of lever C is far cheaper than assumed. | **HIGH** (unblocks lever C = the live frontier-breaking blocker; pose term is 86% of d_seg's marginal at the operating point) | render radial-zoom warp `grid=FoE+exp(s_i)(coord-FoE)` for 600 pairs, measure d_pose vs dense MotionPredictor on the frozen bc20 basin, CPU, $0 | byte-close the 600 FP16 scalars into the bc20 `0.bin`; couple s_i to the pose-MSE teacher (it was only ever a forward-warp theory) |
| **R-2** | **apogee_int4 / int-quant rate lever** (`project_apogee_int4_FALSIFIED_score_1_43…`) | FALSIFIED at 1.43 [contest-CUDA] — but explicitly **"NAIVE-PTQ config falsified, NOT killed"**; QAT/LSQ/per-channel/smaller-block/outlier-handling all **untested** | **REOPEN-IMPLEMENTATION (already self-flagged).** The 1.43 was naive post-training int4 PTQ at pose_avg 700× the frontier — a cliff crossing (see R-9), not a paradigm result. The bc20 vehicle's sub-0.15 note explicitly cites **"FP4 rate (−0.022, not wired)"** as a known headroom asset. QAT/LSQ are the canonical winning recipe (Quantizr FP4). The kill operating point (memorized-frontier PTQ) ≠ bc20 small-basis fresh-init basin (has slack). | **HIGH** (−0.022 rate is the single biggest named byte-neutral-adjacent lever on the rate asset; rate headroom is the bc20 sub-0.15 thesis) | run LSQ/QAT fake-quant fp4 on the bc20 decoder weights for a short smoke, measure d_seg/d_pose drift vs fp16 + the brotli'd byte delta, CPU/MLX, $0 | QAT inner loop (fake-quant STE + learned step size, `tac.quantization` LSQ exists) — the kill was PTQ-only, never trained |
| **R-3** | **R1/R2/R3 lossless entropy recode of decoder+latent** (`deferral_recovery_ledger` §B; `orphan_harvest…` R1-R3) | BLOCKED as "planning-coordinates only / requires adapter" — the adapter primitives (`pr101_split_brotli_codec`, `pr103_arithmetic_codec`, `shared_pmf_model`, `constriction`) were all judged missing | **REOPEN-IMPLEMENTATION — blocker DISSOLVED.** PR#112 cashed exactly this (−1,060 B byte-identical). All primitives are now in-tree. **NUANCE the AC kills miss (see R-7):** our latent verdict falsified 2nd-order *re-prediction*; it never tested PR#112's 1st-order AR + cross-dim LS + range-coder replacing LZMA (PR#112 measured −317 B). | **TOP value×readiness** (READY_NOW ~90 LOC; R1+R2 → ~177,114 B → S≈0.191117, beats frontier −0.00092, ZERO fidelity risk; orthogonal to the saturated distortion vertex) | extend `byte_range_entropy_recode_materializer.py`, recode the 7 brotli streams + latent, byte-close, ONE paired replay (~$0.3 — the only sub-$ cost in TIER A) | PR#112's adaptive geometric-primed per-tensor (ρ,M,inc,ε) model on `shared_pmf_model` (~50 LOC) |
| **R-4** | **STC / syndrome coding on mask-DELTA, detector-informed cost map** (`project_lane_stc_clean_source_FALSIFIED`; `feedback_stc_clean_source_mask_delta_disambiguator_probe_landed_20260530`) | original FALSIFICATION was **MPS-derived (INVALID per CLAUDE.md)**; 2026-05-30 $0 probe confirmed **DEFER** — uniform-cost STC self-syndrome 2.4-2.6× LARGER than brotli at every sparsity | **REOPEN-IMPLEMENTATION (narrow).** SOUND-KILL for *uniform-cost* STC (self-syndrome stores h bits/block regardless of sparsity — algebraic). But the symposium reactivation criterion (CC#2: inverse-SegNet-boundary **cost map** so STC near-optimality-at-fixed-rate moves the operating point) was NEVER built. For bc20 this is the lever-D contour-coder path: a margin-aware (low-cost = flip-prone) STC could hit ≲170-250 B/frame. | **MED** (lever D seg-axis rate lever; composes with the d_seg attack) | build the inverse-SegNet-boundary cost map, feed `ternary_stc_encode_stream` with non-uniform per-symbol costs, measure vs brotli(mask-delta) on the frozen basin, $0 | detector-informed (UNIWARD-style) cost map — the prototype used uniform cost, which provably cannot exploit sparsity |
| **R-5** | **Margin-aware boundary/contour coder (lever D)** (`deferral_recovery_ledger` A3; `closed_spec_boundary_solver_v1`) | DEFER — storing SegNet argmax partition directly is d_seg=0 by construction but costs 524.8 KB under LZMA-over-labels (2.96× the archive); boundary entropy too high | **REOPEN-IMPLEMENTATION.** SOUND-KILL only for the *LZMA-over-labels* coder. The d_seg=0-by-construction property is geometrically exact and unique to this route — it directly attacks the BINDING axis. The kill is purely a coder-efficiency problem (524.8 KB → need ≲170-250 B/frame, 3.6-5.3×), i.e. a margin-aware STC/UNIWARD boundary entropy coder (= R-4's coder) run on the bc20 base. | **MED-HIGH** (only known route that drives d_seg toward 0 structurally; the binding axis) | run `boundary_math_seg_core` on the bc20 lever-B/contiguous-residual base, measure the partition's conditional entropy with the margin-conditional position+class coder (reuse the witness-reopen ChARM coder, already bit-exact) | margin-conditional STC boundary coder at <1.27 B/flip (the 2026-06-16 witness probe proved the coder hits 0.75 B/flip — the coder EXISTS now) |

### TIER B — paradigm-intact substrate families closed on prototypes; need real training (NEEDS-CAMPAIGN)

| # | closure | original verdict | FRESH-LENS | classify | score-EV | $0 re-validate |
|---|---|---|---|---|---|---|
| **R-6** | **Cool-Chic / C3** (`lane_substrate_cool_chic_20260512` research_only; export-gated) | DEFERRED-pending-export-design; "C3 hit the FP4A export gate AFTER training, not before"; abandoned partly on training SLOWNESS | **REOPEN-IMPLEMENTATION.** Paradigm never falsified — it was export-gated + slow. **Enabler changed:** there is now a live `--train-device mps` Cool-Chic path (**28.8× faster/epoch** than CPU on the Cool-Chic gradient; registry note). The two original blockers (export-first + speed) are both addressable now. Cool-Chic's per-pixel latent + AR-prior is a legitimate alt frontier substrate to the HNeRV basin. | **MED** (alt-substrate hedge; only pursue if it beats bc20 on the d_seg axis — same caveat as #40 HiNeRV) | export contract FIRST (declare 0.bin grammar), then a short MPS-train smoke to confirm descent; do NOT train before export per HNeRV L2 | top-AIML = export-first design + the MPS-fast gradient (prototype was export-last + CPU-slow) |
| **R-7** | **Arithmetic/range/ANS bolt-on on HNeRV-class streams** (`feedback_ac_bolt_on_real_encoder_smoke_falsified`; `markov1_aac_falsified`) | retired: Brotli q11 dominates AC by 17.6% on the K-coarsened raw-int8 weight stream (Brotli≈Huffman-optimal there) | **REOPEN-IMPLEMENTATION (substrate-scoped).** SOUND-KILL for *raw-int8-weights at K-coarsening with a constant model* (the algebra: Brotli static-Huffman beats AC's modeling overhead on a 247-symbol skewed stream). But the self-flagged reactivation configs (#1 pre-coarsening 256-sym, #2 rms<0.04, #3 rANS/FSE-tANS/context-mix, #4 joint-header amortization) are UNTESTED, and PR#103 SILVER won with AC on HNeRV *quantized features* (a different distribution). bc20 latents/features ≠ raw int8 weights. | **MED** (this is R-3's open lever for the latent stream specifically) | run constriction rANS/context-mix on the bc20 latent stream (full 256 alphabet, joint header) vs brotli, $0 | context-mixing / learned-PMF AC (prototype used a constant static model = worst case for AC) |
| **R-8** | **NeRV-fleet L0/L1 scaffolds** (lane_substrate_{tc,block,ff,ds,hi,boost}_nerv, e_nerv, nervdc, ego_nerv, mnerv, vqvae, siren, sabor) | research_only=true; most `_full_main raises NotImplementedError`; several "TERMINATED-API-CRASH" | **NOT a kill — never-trained scaffolds (paradigm 100% intact).** These were never falsified; they were never RUN. **But low marginal EV for THIS vehicle:** bc20 IS already HNeRV-class, so a wholesale NeRV-fleet retrain re-litigates the same architecture family. Exception: **HiNeRV** — the bilinear-skip-residual kernel (F1, ~15 LOC, landed) is the named fix for "Shared Mistake A" (skip-free decoder → mean-field → d_seg≈0.5); it directly attacks the bc20 d_seg axis and is a parallel hedge to lever C. | **LOW** fleet-wide; **MED** for HiNeRV-skip specifically | wire `bilinear_skip_residual_canonical` into the bc20/pact_nerv_vq decoder, descent-proof 16-pair smoke, $0 | the bilinear-skip HF carrier the skip-free scaffolds all lacked |

### TIER C — SOUND-KILL at this vehicle's operating point (do NOT reopen; guard rails)

| # | closure | why it stays closed (FRESH-LENS confirms) |
|---|---|---|
| **R-9** | **`rel_err²` Lagrangian objective** (`feedback_three_lossy_anchors…falsified`) | **SOUND-KILL of the OBJECTIVE** (algebraically: S has no rel_err² term; ≥0.04 rel_err = discontinuous d_seg/d_pose cliff). NOTE: this kill is the *cause* of R-2/lossy-coarsening kills — those substrates are REOPEN (their technique), the objective is not. Replace with Fisher/component-sensitivity weighting (already in-tree, never wired). |
| **R-10** | **Pixel-domain preprocessing (blur/chroma/gentle)** (`feedback_preprocessing_dead_end`) | **SOUND-KILL.** +90-105% PoseNet on every variant — PoseNet uses the whole frame (geometric+color cues). Confirmed by the 2026-06-16 witness reopen: per-pixel RGB perturbations don't survive the bicubic^874→bilinear_384→uint8 chain (36.9% survival wall). d_seg/pose must move via the DECODER, not pixel post-processing. |
| **R-11** | **Per-pixel RGB seg-correction SIDECAR** (`witness_seg_boundary_topaiml_reopen_survival_wall_20260616`) | **SOUND-KILL (freshly top-AIML re-tested).** The CODING half is solved (47.5 KB in band, 0.75 B/flip < waterline, witness 138 KB < 177 KB frontier) but **survival is a structural wall (36.9%)** — only ~37% of boundary corrections survive eval resampling; sidecar ΔS = +0.152 (WORSE). Reactivates ONLY if a resampling-surviving (camera-grid / receptive-field-aware) correction raises survival >70%. The coder is reusable (feeds R-4/R-5). |
| **R-12** | **Pure-symbol arithmetic/RLE/Huffman of the AV1 mask stream** (`owv3_0120_arith_masks_FALSIFIED`) | **SOUND-KILL.** AMRC 2.57× / LZMA 2.55× larger than the AV1 monochrome baseline (421 KB). AV1's spatial+temporal model on the 5-class argmax is information-theoretically better than symbol coding. (Distinct from R-4/R-5 which code mask-DELTAs with a detector-informed cost map — a different source.) |

---

## 2. FLAGGED — KILL where CLAUDE.md required DEFER (consensus + research-exhaustion)

The two month-old audits already established the systemic finding: **28-32 of 34 historical verdicts fail
today's 5-gate rigor**; the pre-rigor apparatus systematically over-killed. The `deferral_recovery_ledger`
(2026-06-10) re-ran a Catalog #307 audit and found it **CLEAN** (every currently-KILL-flagged lane is
implementation-scoped with pinned reactivation criteria — no live paradigm kill). FRESH-LENS confirms: the
remaining fallibility is NOT mislabeled kills sitting in the registry; it is **un-executed reactivations**
(R-1 marked "Theory", R-2 self-flagged "NOT killed", R-3/R-4/R-5 blockers dissolved, R-7 configs untested).
Specific items whose verdict label was harsher than the evidence warranted:

- **`project_posenet_rank1_discovery`** — a MAJOR geometric discovery left at "Theory / never byte-closed"
  for ~6 weeks. This is the canary. Highest signal-recovery item.
- **STC clean-source** — original verdict literally "FALSIFIED" while the evidence was MPS-derived (invalid).
  Already softened to DEFER, but the detector-informed cost-map reactivation never ran.
- **apogee_int4** — title says FALSIFIED; body says "NOT killed, QAT/LSQ untested." The harsh title risks
  the lane being treated as dead.

No new KILL is proposed (per Forbidden-premature-KILL). No lane is auto-resurrected — all are CANDIDATES.

---

## 3. RECOMMENDED EXECUTION ORDER (score-EV-per-effort, biased to bc20 sub-0.15)

1. **R-3** (lossless recode) — TOP value×readiness, ~90 LOC, the only ready-now exact-axis win (−0.00092),
   zero fidelity risk. Bank it.
2. **R-1** (radial-zoom pose, 1 DOF) — $0 to validate the canary; unblocks lever C (the LIVE frontier-breaking
   blocker) by collapsing its pose half from 6-DOF to 1-DOF.
3. **R-2** (QAT/LSQ fp4 on bc20 weights) — $0 smoke for the −0.022 rate asset that IS the sub-0.15 thesis.
4. **R-5 + R-4** (margin-aware boundary/contour coder on the bc20 base) — the only structural d_seg→0 route;
   the coder already exists (witness reopen, 0.75 B/flip bit-exact).
5. **R-7** (context-mix AC on bc20 latents) — folds into R-3's latent lever.
6. **R-8-HiNeRV-skip / R-6 Cool-Chic-MPS** — parallel architecture hedges; only if they beat bc20 on d_seg.

DO NOT reopen R-9 through R-12 (sound kills at this operating point; R-9's objective replacement + R-11's
coder are already harvested as inputs to the live levers).

---

## 4. 6-hook wire-in declaration (Catalog #125)

- Sensitivity-map: ACTIVE — re-affirms seg=90.8% byte-sensitivity binding-axis prior for the bit-allocator.
- Pareto: ACTIVE — R-1/R-2/R-3 are orthogonal-axis (pose / rate / rate) moves off the saturated distortion vertex.
- Bit-allocator: ACTIVE — R-4/R-5 are margin-conditional boundary-bit allocators; R-2 is a per-tensor quant allocator.
- Cathedral autopilot: ACTIVE — this ledger is operator-routable into the per-substrate symposium / DAG queue
  as REOPEN candidates (sister of `deferred_items_must_feed_canonical_work_queue_and_dag` directive).
- Continual-learning posterior: ACTIVE — this memo is the canonical RE-AUDIT anchor; future audits diff against it.
- Probe-disambiguator: ACTIVE — each REOPEN names its falsifiable $0 re-validation threshold.

## 5. Cross-references

- `project_posenet_rank1_discovery.md` (R-1 canary source) · `project_apogee_int4_FALSIFIED…` (R-2) ·
  `deferral_recovery_ledger_20260610T130200Z.md` §B (R-3) · `feedback_stc_clean_source_mask_delta_disambiguator_probe_landed_20260530` (R-4) ·
  `evaluator_inverse_orphan_inventory_20260609.md` (R-5 surfaces + seg-dominance) ·
  `feedback_witness_seg_boundary_topaiml_reopen_survival_wall_20260616.md` (R-11 SOUND-KILL + reusable coder) ·
  `feedback_ac_bolt_on_real_encoder_smoke_falsified` (R-7) · `feedback_three_lossy_anchors…falsified` (R-9) ·
  `pre_rigor_kill_defer_falsified_inventory_20260517.md` + `meta_bug_retroactive_defer_kill_falsify_audit` (the over-kill anchors).
- CLAUDE.md: NO-FAKE · THE GOAL sub-0.15 + means/ends firewall · Forbidden premature KILL · ANTI-SIGNAL-LOSS
  (janky-prototype → top-AIML REOPEN) · Catalog #307 (paradigm-vs-implementation).
