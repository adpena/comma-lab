# Generator d_seg power-law to frontier — does the score-native witness generator reach 5.6e-4? (2026-06-21)

**Operator ask (gap #1 of the witness campaign):** fit the score-native GENERATOR's d_seg power-law from
EXISTING data and extrapolate to frontier-level (5.6e-4). The byte-closed L13 witness generator sits at
d_seg=0.0068 — 12× the frontier; the 72 KB / −59% rate win only becomes a winning S if the generator's d_seg
reaches frontier-level. Does it, within a feasible capacity/byte budget?

**Authority:** every number `[macOS-CPU advisory]` / `[macOS-MLX research-signal]`, **NON-PROMOTABLE**
(`promotable=false`, `score_claim=false`, `ready_for_exact_eval_dispatch=false`). `$0`, READ-ONLY harvest +
light CPU power-law fit (OMP_NUM_THREADS=2). **NO training launched; the live MPS run (pid 49375,
`launch_split_by_head_basin` bc20/600p/taper) was NOT touched.** Exact frontier pointer **UNMOVED: 0.19110
[contest-CPU]**. This memo emits NO archive and moves NO score — it is a feasibility verdict + a gated campaign spec.

---

## 0. HEADLINE — verdict: **WALL on the capacity axis; NARROW GO on the training+structure axis.**

The capacity power-law is REAL for real HNeRV decoders (`b≈2.6`, R²=0.91 incl the frontier anchor; bc24 0.00285 <
bc20 0.00376 — more capacity LOWERS d_seg, confirming the retracted-memory correction). **But reaching frontier
d_seg by ADDING CAPACITY is RD-INFEASIBLE for the witness: it needs ~189 K params ≈ 172 KB int8 → rate 0.117 ≈
the RGB-HNeRV frontier rate 0.118 — it ERASES the −59% witness rate win.** The witness win and frontier-via-capacity
are mutually exclusive: a generator big enough to floor d_seg low IS the dense decoder (relocates the coupling,
the factored-LF-gate finding generalizes).

**The decisive reframe — the witness does NOT need the full 5.6e-4.** With the seg-carrier at 65 KB and pose carried
separately, the witness BEATS the frontier S=0.191 when the **generator d_seg < 9.2e-4** (only **7.4× below the
current 0.0068**, not 12×) and reaches **sub-0.15 when generator d_seg < ~3.2e-4**. And the binding lever for that
gap is **TRAINING + SHARED STRUCTURE, not capacity:** the live bc20/600-pair run already trained d_seg from
0.0166 (ep25) → **0.00222 (ep6025)** at FIXED ~83 K params — i.e. the same-capacity decoder is already at **2.4× the
win line** purely from epochs. The capacity floor is not the wall; the d_seg floor at fixed capacity is descending.

---

## 1. The (params/epochs, d_seg) point table — all sources

### 1a. CAPACITY axis (d_seg vs params, at matched/converged training)

| family | config | params | d_seg | source | notes |
|---|---|---:|---:|---|---|
| **A: tiny/factored** | LF-only core bc8 (100% CE on d_seg, 1200ep conv.) | 20,078 | 0.02584 | `factored_lf_core_capacity_gate_20260618T233940Z.md` | LF-only, no recon/pose |
| **A** | LF-only core bc12 (1200ep conv.) | 36,540 | 0.01689 | same | |
| **A** | NCA amortized c8h32 (shared rule, 4 frames, conv.) | 10,299 | 0.012975 | `nca_amortized_capacity_break_main/gate_state.json` | best converged restart |
| **A** | NCA amortized c12h64 (larger rule) | 16,999 | 0.13929 | same | **INVERTS** (10× worse — optim-difficulty) |
| **B: real HNeRV decoder** | bc20 @48p, 120ep, stored_latent, CE | 83,422 | 0.0037602 | `capstone_capacity_ablation_2x2_20260611/bc20_p48/capstone_result.json` | matched-budget arm |
| **B** | bc24 @48p, 120ep, stored_latent, CE | 114,933 | 0.0028546 | `…/bc24_p48/capstone_result.json` | **bc24 < bc20 ⇒ more cap LOWERS d_seg** |
| **B** | bc20 @192p, 120ep | 83,422 | 0.019203 | `…/bc20_p192/` | 192p HARDER at fixed cap (data axis ↓) |
| **B (anchor)** | frontier PR101/A1 (1200 frames) | 178,262 | 0.00056 | `dseg_plateau_data_vs_capacity_20260611.md` §2 | the TARGET |
| **L13 witness gen** | score-native palette-painter generator, ~200ep | ~65 K (int8 65,305 B) | 0.006845 | `score_native_first_candidate_20260610T112433Z.md` §1 | the candidate to lift |

### 1b. TRAINING axis (d_seg vs epoch, fixed bc20/600-pair, the live generator-class run)

| global_epoch | d_seg | d_pose | stage | source |
|---:|---:|---:|---|---|
| 25 | 0.0166875 | 0.034879 | stage1 CE | `yousfi_r3_taper_marginhinge_e5_20260620/torch_vehicle_trajectory.jsonl` (evaluated rows) |
| 625 | 0.00306886 | 0.000708 | stage1 CE | same |
| 1225 | 0.00265723 | 0.000576 | stage1 CE | same |
| 2050 | 0.00235806 | 0.000392 | stage1 CE | same |
| 3025 | 0.00232796 | 0.000289 | stage2 softplus | same |
| 6025 (best) | **0.00222311** | 0.000438 | stage2 softplus | same |

(bc20/600-pair, taper `16,16,17,19,19,14,10`, margin-hinge, KD-warm-start, fresh-init — the same generator class.
Note this is at **600 pairs / 1200 frames** = the contest-faithful data load, unlike the 48-pair ablation arms.)

Sister anchors (CE-only, 96-pair, fresh-init, `from0_ab_v2_n96`): control reaches 0.00359 @ ep789; stack_ce_early
0.00406 @ ep789 — consistent flattening to ~0.0035–0.0040 by ~800ep at 96 pairs.

---

## 2. The power-law fit(s) + R² + the confound assessment

### 2a. Capacity law  `d_seg = A · params^(−b)`

| fit set | A | b | R² | verdict |
|---|---:|---:|---:|---|
| Family A (tiny LF + NCA, 3 pts) | 1.95e-3 | **−0.224** | 0.166 | **NO clean law** — scatter; b<0 (the c12h64 inversion) |
| canonical factored-LF (bc8/bc12 only) | 29.3 | 0.710 | (2 pts) | the memory-retracted fit; **Family-A-only** |
| **Family B (real bc20/bc24, 2 pts)** | 64.1 | **0.860** | 1.0 (2pts) | b>0: more cap LOWERS d_seg ✓ |
| **Family B + frontier (3 pts incl PR101)** | 2.28e10 | **2.58** | **0.913** | the best real-decoder law to the target |

**The two families are on DIFFERENT curves and the cross-family fit is meaningless.** Family A (tiny/factored,
10–37 K params) floors d_seg at 0.013–0.026; Family B (real HNeRV, 83–178 K params) floors at 0.0038–0.00056 —
**~2.5× better per param**. The factored-LF law `29.3·p^−0.71` predicts d_seg=0.0094 at 83 K params; the REAL bc20
measures 0.0038 (2.5× better). So the `29.3·p^−0.71` law is a **Family-A artifact** (factored/NCA tiny arches) and
does NOT govern the real RGB-emitting generator class — exactly the false-foundation the retracted memory warned of.

### 2b. Training law  `d_seg = A · ep^(−c)` (live bc20/600-pair)

| fit window | A | c | R² | epochs → 9.2e-4 (win) | epochs → 5.6e-4 (frontier) |
|---|---:|---:|---:|---:|---:|
| post-knee (ep≥625, mixed stages) | 7.46e-3 | **0.144** | 0.906 | ~3.0e6 | ~6.6e7 |
| CE-only sub-fit (ep625–2050) | 1.28e-2 | **0.221** | 0.9995 | ~6.6e5 | ~1.4e6 |

**The training law is SHALLOW (c≈0.14–0.22) — a stretched-exponential / glassy plateau, NOT a fast power-law.**
A pure-epochs path to the win line needs ~10⁵–10⁶ epochs (infeasible by epochs alone). The training axis HELPS
(0.0166→0.00222 over 6025ep, already 2.4× the win line at fixed capacity) but ASYMPTOTES well above the win line on
epochs alone. This matches every prior finding (factored-LF "CE keeps dropping while d_seg nearly stops"; NCA glassy
dynamics; capstone stage-3 reversal).

### 2c. Confound assessment — is the capacity law real or an artifact? (HONEST, per the retraction)

- **The Family-B law IS real** (b>0; bc24 0.00285 < bc20 0.00376 at MATCHED 48p/120ep budget; the frontier anchor
  lies on the same descending curve, R²=0.91). More REAL-decoder capacity genuinely lowers d_seg. The
  retracted-memory claim "power-law false-foundationed" applies to the **Family-A (factored/NCA tiny)** fit, NOT to
  Family B. **I do NOT repeat the retraction error: I split the families and only extrapolate within Family B.**
- **But two confounds bound the Family-B extrapolation:**
  1. **Under-training mixed with capacity.** The 2x2 arms are 120ep (NOT converged — the live 600-pair run is still
     descending at 6025ep). So bc20/bc24's 48p/120ep d_segs are upper-ish, and the b≈0.86 (2-pt) vs b≈2.58 (3-pt
     w/ frontier, fully-trained 1200-frame) divergence is partly the frontier being a 1200-frame fully-converged
     point vs the arms being 96-frame 120ep points. The cross-data-load comparison inflates the apparent slope.
  2. **Data load entangled.** bc20@48p (0.0038) vs bc20@192p (0.0192) shows 4× more pairs at fixed capacity makes
     d_seg ~5× WORSE (the params/frame physics — single-video memorization). The frontier's 0.00056 is at 1200
     frames; the ablation arms at 96 frames. So "params" and "frames" are confounded across the table; the clean
     capacity slope is only the 48p bc20-vs-bc24 pair (b≈0.86).
- **Net:** the capacity law is real and monotone (more real-decoder params → lower d_seg) but its EXPONENT is
  uncertain (0.86 same-budget vs 2.58 to-the-frontier-anchor). I report the RANGE and key the verdict on the
  RD-feasibility, which is robust across the range (see §3).

---

## 3. Extrapolation + the decisive RD-feasibility check → **WALL on capacity / GO on training+structure**

**The two crossover thresholds (witness, NOT the full 5.6e-4):** with the seg-carrier at its measured 65 KB rate
(0.0435) and pose carried separately (~frontier-level pose term 0.055):
- **WIN line (beat frontier 0.191):** generator d_seg < **9.2e-4** → only **7.4×** below current 0.0068.
- **sub-0.15 line:** generator d_seg < **~3.2e-4**.

### 3a. Capacity path to frontier d_seg — RD-INFEASIBLE (the WALL)

Using measured int8+brotli = **0.934 B/param** (bc20 79,211 B / 83,422 p; bc24 105,513 B / 114,933 p):

| capacity law used | params for d_seg=5.6e-4 | int8 bytes | rate `25·B/N` | witness total (+16.5 KB pose) | S IF d_seg=5.6e-4, pose=3e-4 |
|---|---:|---:|---:|---:|---:|
| real-decoder (3-pt, b=2.58) | 188,742 | 172.1 KB | **0.1174** | 188 KB → 0.1283 | **0.239** ✗ |
| real-decoder (2-pt, b=0.86) | 763,941 | 696.6 KB | 0.4750 | 713 KB → 0.486 | 0.597 ✗ |
| factored-LF (29.3·p^−0.71) | 10,679,777 | 9.7 MB | 6.64 | — | 6.76 ✗ |

**EVEN THE MOST OPTIMISTIC capacity law (b=2.58) needs ~189 K params ≈ 172 KB int8 → rate 0.117 ≈ the RGB-HNeRV
frontier rate 0.118.** Reaching frontier d_seg by capacity makes the witness rate equal the RGB-HNeRV it was meant
to beat — **the −59% rate win is exactly erased.** The generator big enough to floor d_seg low IS the dense decoder
(the factored-LF-gate "the factorization does not decouple, it relocates" result, generalized to the witness). The
capacity axis is a WALL: across the whole exponent range, frontier-d_seg-via-capacity costs ≥ the frontier's own rate.

### 3b. Training+structure path to the WIN line (d_seg < 9.2e-4) — NARROW GO

- The live bc20/600-pair run is at **0.00222 at 6025ep at FIXED ~83 K params** — already **2.4× the win line**,
  reached purely by training the same-capacity generator. The training law (c≈0.22 CE) says pure epochs to the win
  line is ~10⁵–10⁶ epochs (infeasible alone) — so epochs ALONE wall above the win line.
- **The gap-closing levers are the ones the 2026-06-21 sweep already identified as the ONLY real d_seg movers**
  (`structural_rate_axis…`, `dseg_boundary_hessian_conditioning…`): **(a) the Muon κ-buster** (d_seg = a
  conditioning problem on the shallow boundary; training-time, FREE); **(b) byte-neutral d_seg-aware taper /
  shared-structure** (re-allocate the SAME params to the 192×256 band where flips live; 0 bytes); **(c) recon-light
  score-aware loss** (the decoder is "full-rank for RGB, over-capacity for the score" — shrink into the
  score-relevant subspace). These move d_seg at FIXED or LOWER bytes — they do NOT pay the capacity rate.
- **The honest GO is NARROW:** the win line (9.2e-4) is 2.4× below the live 0.00222, and the training law is
  flattening hard (0.00236 → 0.00222 over ep2050→6025 = only −6% for 3× more epochs). Closing 2.4× requires the
  structure/conditioning levers to bend the floor, not just more epochs. It is plausible (the levers are real and
  free) but NOT measured. Sub-0.15 (3.2e-4 = 7× below 0.00222) is a much harder reach on the same axis.

---

## 4. Verdict + the gated campaign spec (INSUFFICIENT-DATA for the exact GO/WALL line; the wall is firm)

**VERDICT: WALL on the capacity axis (firm, RD-proven). INSUFFICIENT-DATA on whether training+structure crosses
the witness WIN line (9.2e-4) — but the existing live data places it within 2.4× at fixed capacity, making it the
ONLY feasible path and worth the gated probe.** The full-frontier-5.6e-4 target is a RED HERRING for the witness;
the binding number is 9.2e-4 (win) / 3.2e-4 (sub-0.15), and the binding lever is training+structure, not capacity.

**Why INSUFFICIENT-DATA, not GO:** the three feasibility-deciding measurements do NOT exist as clean anchors:
1. **No converged fixed-capacity d_seg floor with the κ-buster + d_seg-aware-taper levers ACTIVE.** The live run
   has margin-hinge + taper but the Muon κ-buster + the structure-realloc are not isolated. We do not know the
   fixed-83K floor WITH the free d_seg levers.
2. **The capacity exponent is confounded** (b 0.86 same-budget vs 2.58 to-the-frontier) by data-load + training-length
   mixing across the table — no clean same-frames, same-epochs, multi-capacity converged sweep exists.
3. **The L13 score-native generator itself has NO d_seg-vs-epoch trajectory** (only the single 200ep/0.0068 point);
   its own training curve and floor are unmeasured.

**Minimal new training points to settle it (GATED on MPS-free; do NOT run now — the live run owns the GPU):**

| probe | config | what it settles | cost (MPS-free) |
|---|---|---|---|
| **P1 (decisive)** | bc20, 600p, fresh-init, **κ-buster Muon + d_seg-aware-taper ON**, train to convergence (geometric-Δ<1%), LIVE d_seg | the fixed-83K d_seg floor WITH the free levers — does it cross 9.2e-4 at ZERO extra bytes? | 1 long daemon (~days) |
| **P2 (capacity slope)** | {bc20, bc24, bc28} × 600p, MATCHED epochs-to-convergence, CE-only, LIVE d_seg | the CLEAN same-data capacity exponent b (deconfound P1's table) → exact params-for-9.2e-4 | 3 daemons |
| **P3 (L13 own curve)** | the score-native palette-painter generator, d_seg-vs-epoch to convergence | the L13 generator's own floor (vs the HNeRV-class proxy used here) | 1 daemon |

P1 is the single decisive gate: if a fixed-83K (rate-winning) generator with the free d_seg levers crosses 9.2e-4,
the witness class-shift is GO at the −59% rate; if it floors above 9.2e-4 and P2 confirms the only way down is
capacity (which erases the rate win), the witness seg-side has a structural wall and the class-shift is dead.

---

## 5. Wire-in (Catalog #125)

1. **sensitivity-map — ACTIVE.** New prior: the witness needs generator d_seg < 9.2e-4 (win) / 3.2e-4 (sub-0.15),
   NOT the full frontier 5.6e-4; the binding lever is training+structure (κ-buster + d_seg-aware taper), capacity is
   a WALL (frontier-d_seg-via-capacity costs ≥ frontier rate).
2. **Pareto — ACTIVE.** Adds the witness (rate, d_seg) point (0.0435, 0.0068) and the WIN line d_seg<9.2e-4 at fixed
   65 KB; the capacity move (→189 K, rate 0.117) is dominated (it lands on the frontier's own rate).
3. **bit-allocator — ACTIVE.** Do NOT allocate the generator to ≥189 K params chasing d_seg (it erases the witness
   win); reserve the seg-carrier near 65 KB and spend the d_seg budget on TRAINING/structure (0 bytes).
4. **cathedral-autopilot — gate-conditional.** P1 (MPS-free) → (LIVE d_seg < 9.2e-4 at fixed 65 KB) → compose L13 +
   Wyner-Ziv pose-FiLM end-to-end → ONE paired exact CPU+CUDA eval. Do NOT dispatch before P1 crosses.
5. **continual-learning — ACTIVE.** Reseeds: (a) the capacity law is REAL for Family B (b>0; bc24<bc20) but a WALL
   for the witness (frontier-d_seg-via-capacity erases the rate win); (b) the `29.3·p^−0.71` law is Family-A-only
   (tiny/factored), does NOT govern the real generator (the retraction was right about Family A, wrong if applied to
   Family B); (c) the witness win line is 9.2e-4 not 5.6e-4; (d) training law is glassy (c≈0.22), epochs-alone walls.
6. **probe-disambiguator — PARTIALLY RESOLVED + ONE OPEN.** "Does the generator reach frontier d_seg via capacity?"
   → NO (RD-infeasible, WALL). "Via training+structure to the WIN line (9.2e-4)?" → OPEN (within 2.4× at fixed
   capacity; the P1 gate decides) — `sign(P1_converged_floor − 9.2e-4)` is the empirical arbiter.

## NO-FAKE ledger
- MEASURED: all (params, d_seg) and (epoch, d_seg) points are real argmax-flip-rates from the cited result JSONs /
  live trajectory (exact frozen SegNet, CPU/MLX-research authority). int8 B/param from measured decoder blobs.
- FITTED ($0, CPU): the Family-A / Family-B / training power-laws (A, b/c, R² inline); the RD byte/rate cost.
- REASONED: the family split (A vs B different curves); the win-line 9.2e-4 derivation; the capacity-WALL (frontier-
  d_seg-via-capacity ≈ frontier rate); the confound bounds on b (data-load + training-length entanglement).
- NOT claimed: NO score moved (pointer UNMOVED 0.19110); NO training launched; the training+structure GO is NARROW
  and UNMEASURED (P1 gate, MPS-free); the witness class-shift remains DERIVED-not-built. The exponent b is uncertain
  (0.86–2.58) and the verdict is keyed on the RD-feasibility, which holds across the range.

## Cross-references
- Witness formulation: `CAPSTONE_witness_taskspace_roundtrip_byte_floor_formulation_20260621.md` (§8 open gap #1 =
  THIS memo) · the 2026-06-21 sweep: `structural_rate_axis_and_sweep_conclusion_20260621.md` (rate = param_count;
  the only levers are designed-in training/structure) · `dseg_boundary_hessian_conditioning_20260621.md` (κ-buster;
  shallow boundary) · `decoder_weight_rate_axis_and_shallow_boundary_synthesis_20260621.md`.
- Capacity anchors: `factored_lf_core_capacity_gate_20260618T233940Z.md` (Family-A 29.3·p^−0.71 law + bc8/bc12) ·
  `generative_axis_nca_amortized_capacity_break_RED_20260619.md` (NCA c8h32/c12h64; the inversion) ·
  `dseg_plateau_data_vs_capacity_20260611.md` (params/frame physics; the frontier 178K/5.6e-4 anchor) ·
  `capstone_build_and_capacity_ablation_20260611.md` + `capstone_capacity_ablation_2x2_20260611/` (bc20/bc24 arms).
- Witness candidate: `score_native_first_candidate_20260610T112433Z.md` (L13: 72,217 B, d_seg 0.0068).
- Live training axis: `experiments/results/yousfi_r3_taper_marginhinge_e5_20260620/torch_vehicle_trajectory.jsonl`.
- Retraction context: MEMORY `project_representation_axis_sub015_exhausted…` (RETRACTED — power-law false-foundationed
  on the WRONG tiny arches; this memo honors that by splitting Family A from Family B and extrapolating only within B).
