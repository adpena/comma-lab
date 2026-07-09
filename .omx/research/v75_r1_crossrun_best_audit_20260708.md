# v7.5 vs R1 + CROSS-RUN MEASURED-BEST AUDIT — 2026-07-08

**Task (operator max-effort, "no stone unturned"):** are the best R1 + cross-run techniques folded into and
turned ON in v7.5? Read-only, $0. Every number `[macOS-CPU advisory]/[macOS-MLX research-signal]
NON-PROMOTABLE` unless noted. **Pointer contest-CPU 0.19110 UNMOVED.** `[no-triality]` — no eq/DSL leg lands.

STORES CONSULTED: R1 run dir (`levelset_n600_R1_storenothing_descent_ev1_20260703T004906Z`
launch.sh/run.log/levelset_best.json) · `r1_0011_custody_revalidation_20260708.md` ·
`witness_205_store_nothing_pose_config_and_review_20260703.md` · v2_attrclean + mod32cap + crucible_v6_run1
run dirs · `src/tac/witness_autoconfig.py` (`_build_crucible_v7`, `derive_crucible_v6_config`,
`derive_store_nothing_205_config`, `mod_dim_generator`, `whitney_mod_dim`, `_sealed_205_deltas`) ·
`SPEC_v75_optimal_single_trunk_20260708.md` · `probe_PA_paintfloor_perclass_20260708.md` ·
`clean_canonical_warp_budget_gate_20260629T203717Z.md`.

---

## PART 1 — R1's winning techniques (what reached d_seg ~0.0045 + d_pose 0.0011)

**R1 = task #245, a JOINT pose-descent run** (`launch.sh` verified). It **warm-started the converged
d_seg witness** `v2_attrclean` (`--resume-from levelset_n600_v2_attrclean_...`, mod-dim 26) **at ep1001
inside the Muon finisher**, then ran ~100 DEDICATED pose epochs with `--w-pose 1.0`.

R1's EXACT config (launch.sh): mod-dim **26** · hidden 96 · n-hidden 4 · hosc β4 ω1 siren-init ·
self-orient n-dir-freqs 2 freq-across 32 freq-along 4 · chroma + palette-anchor · eikonal 0.01 +
length 0.001 · structured-init +lane · lane-prior-phi1 replace dash-gate · ema 0.997 · accum 8 ·
Muon(start 726, lr 0.002, mom 0.95, ns 5) · **w-pose 1.0 · pose-carrier source=generated
residual-mode=table** (STORE-NOTHING: frame0 = warp(witness's OWN render, ξ); ships ONLY ξ/H).

### The two numbers, cleanly separated (this is the crux of the whole audit)

| quantity | R1 value | who actually produced it |
|---|---|---|
| **d_seg** | HELD ~0.00450–0.00453 (best 0.004502 @ep1001) | **INHERITED from v2_attrclean — NOT an R1 win.** R1 warm-started v2's converged d_seg (0.004024) and merely held it (slightly worse) during pose descent. Every R1 d_seg lever (mod-26, hosc, self-orient, chroma, structured-init, lane-prior) is **v2_attrclean's**, not R1's. |
| **d_pose** | 62.44 → **0.0011** (plateau ep1074–1108, custody-VALID) | **R1's genuine contribution** — joint pose-descent where the render co-adapts, carried entirely by a trained per-pair `dxi` (600×6) residual table. |

**So R1's ONE original winning technique is the joint pose-descent recipe, and it has TWO ingredients:**
1. **warm-start from a converged-d_seg basin** (the render is already good ⇒ pose can co-adapt cheaply); and
2. **a dedicated pose-finish phase** (~100 Muon epochs at w_pose 1.0 after d_seg converged).

**The load-bearing caveat (`r1_0011_custody_revalidation_20260708.md`, MEASURED):** the 0.0011 is a VALID
frozen-CPU-torch PoseNet, contest-definition, n600, through-R, EMA-conservative measurement (prime-suspect
ξ-MSE conflation REFUTED) — **BUT it is NOT byte-closed.** It lives entirely in the trained `dxi` table
which `build_pose_carrier_section` does **not** ship (it recomputes the deterministic calibration ξ →
the ~1.99 no-dxi floor). #238 = the un-done serialize-ξ_eff-and-re-measure test. So R1's pose win is
REAL training-side but **UNPROVEN as a shipped archive row.**

### d_seg-relevant vs pose-only classification of R1's levers
- **d_seg-relevant (all inherited from v2_attrclean):** mod-dim, hosc/self-orient/freq basis, chroma,
  structured-init+lane, lane-prior-phi1, Muon, eikonal/length.
- **pose-only (R1's own):** warm-start-then-dedicated-pose-phase sequencing + the `dxi` store-nothing carrier.

---

## PART 2 — cross-run BEST per class (mine every run)

### (a) Best COMPOSITE witness d_seg measured (the actual n600 verdict authority)

| run | best d_seg | epoch | mod-dim | notes |
|---|---:|---:|---:|---|
| **mod32cap** (`...mod32cap_20260706`) | **0.003366** | 650 | **32** | **BEST measured witness composite.** islands-unborn BY DESIGN (memory L2/L3, council-designed clean baseline). |
| v2_attrclean (`...v2_attrclean_20260630`) | 0.004024 | 650 | 26 | R1's warm-start source |
| R1 (`...R1_storenothing_descent`) | 0.004502 | 1001 | 26 | v2's d_seg held during pose descent |
| #205 CE-floor (memory L67) | ~0.00496 | 225 | 32 | CE-residual = temporal flicker |
| crucible_v6 run-1 (pid 63069, birth-arm) | 0.1198 | 275 | 32 | birth-arm ONLY — Road floored ~0.40 BY DESIGN-GAP; NEVER compare to bulk floors (SPEC §3) |

**Measured fact: mod-32 (0.003366) beats mod-26 (0.004024) by ~16% on composite d_seg.**

### (b) Per-class — measured floors + the one witness per-class block that exists

Caveat (attack-own-conclusion): the best composite runs (mod32cap, v2_attrclean) log **composite only**,
not per-class within-flip. The per-class data that DOES exist is (i) the birth-arm witness block, (ii) two
achievable-FLOOR probes. These are DIFFERENT measurement families — do not read a floor as a witness result.

**(i) run-1 crucible_v6 birth-arm witness within-flip @ep300 (MEASURED, n600):**

| class | part_frac | GT area | within_flip | reading |
|---|---:|---:|---:|---|
| Road0 | 0.153 | 0.232 | **0.345** | UNDER-painted → high flip (the birth imbalance) |
| Lane1 | 0.064 | 0.006 | 0.019 | ~10× OVER-painted (recall-no-precision) |
| Undriv2 | 0.466 | 0.495 | 0.082 | slightly under |
| Movable3 | 0.062 | 0.012 | 0.001 | ~5× OVER-painted |
| MyCar4 | 0.254 | 0.254 | 0.004 | balanced |

This is exactly the run-1 lesson driving v7.5's Chan-Vese counter-force (mass conserved: Lane/Movable
over-paint = Road/Undriv deficit). It is a birth-arm imbalance snapshot, NOT a best-per-class.

**(ii) probe_PA oracle-R paint FLOOR @384 (achievable through-R, NOT witness; composite 0.000910, n600):**
within-class flip Road **0.17%** · Lane **2.5%** · Undriv **0.03%** · Movable **0.76%** · MyCar **0.04%**.
Verdict: interiors paint near-perfectly; 100% of residual is codim-1 separatrix placement (Road = adjacency
hub). This is the irreducible per-class floor, not a witness achievement.

**(iii) clean_canonical warp-carrier per-class FLOOR (best column, warp regime, n600):**
Road(ground) **0.0051** · Lane(ground) **0.39** (vote ERODES to 0.66 — must be STORED/TRAINED, never warped) ·
**Undriv(sky, rotonly) 0.0016** · Movable(ground) **0.035** · MyCar(hood, identity) **0.0028**.
The 0.0016 Undriv floor is a clean-canonical sky-rotation warp-carrier floor; bulk (Road+Undriv+hood) →
clean-canonical carrier, Lane+Movable → must-store. Temporal-jitter/source-jitter is the fraction the
clean-canonical carrier removes (Road −67% pre-R denoise).

**Synthesis of the per-class picture:** bulk classes (Road/Undriv/MyCar) are near-free at the oracle floor
and the clean-canonical warp floor; the binding residual is Lane + Movable (must-store) + the all-class
separatrix placement — consistent with the campaign crux (islands = lane ~8-dim; annulus = boundary-jitter).

---

## PART 3 — the mod-dim verdict

- **Theory (`whitney_mod_dim`):** clamp(2m+1, 19, 26). composite m~13 (lane-orbit ~8 + screw ~6) → 27 →
  clamp **26** (overfit ceiling); aggressive theta*-floor **19** (m~9). Memory L: mod-16 under-embeds,
  17–19 proper floor.
- **MEASURED:** mod-**32** (0.003366) beats mod-**26** (0.004024) on composite d_seg by ~16%. mod-32 is
  ABOVE the Whitney ceiling (26) — SEALED as #205 delta Q4: *"proven arm reached d_seg 0.003698; d_seg is
  BINDING; 19's neutrality UNMEASURED"* — over the Whitney-floor 19.
- **VERDICT: v7.5's mod-32 is the empirically-best measured mod-dim and is INHERITED.** The theory ceiling
  (26) is superseded by the measurement; the aggressive rate-saving floor 19 is UNMEASURED (an open A/B),
  but since d_seg is the binding term and 32 wins d_seg, 32 is the correct default. **No gap here.**

---

## PART 4 — v7.5 inheritance verdict (per technique)

v7.5 = `crucible_v7` = `derive_crucible_v6_config` → `derive_store_nothing_205_config` base (VERIFIED in
`witness_autoconfig.py`): **mod-32 · store-nothing generated pose carrier (w_pose 1.0, residual=table) ·
FROM-SCRATCH 3000ep** + counter-force levers (Chan-Vese area + birth-completion event) ON + 3 P0 forces
default-OFF. It anchors softmax_temp_end 0.31 to "mod32cap ep650-best τ=0.3098" — i.e. **v7.5 is grounded
on the mod32cap arm.**

| technique | best source | in v7.5? | state |
|---|---|---|---|
| mod-32 (best composite d_seg) | mod32cap 0.003366 | YES | **ON** ✅ (best measured; inherited) |
| store-nothing generated pose carrier (w_pose 1.0, residual=table) | R1 / #205 | YES | **ON** ✅ (same carrier as R1) |
| hosc/self-orient/freq-basis/chroma/palette-anchor | v2/mod32cap | YES | **ON** ✅ |
| structured-init +lane / lane-prior-phi1 | v2/R1 | YES | **ON** ✅ |
| Muon finisher (+ Polyak tail, warm-start-momentum) | v2/R1 → v7.5 adds Polyak | YES | **ON** ✅ (v7.5 richer) |
| Chan-Vese area counter-force + birth-completion event | v7.5 NEW (answer to run-1 Road-floor imbalance) | YES | **ON** ✅ (net-new; NOT in R1/v2/mod32cap) |
| 3 P0 forces (temporal-screw / satisficing / tie-locus) | derived #360 | BUILT | **OFF** (registered, duty-to-measure; one-per-increment per SPEC §9) |
| **WARM-START from a converged-d_seg basin** | R1 / v2_attrclean | **NO** | **GAP** ❌ — v7.5 is from-scratch |
| **dedicated pose-finish phase after d_seg converges** | R1 (0.0011 training-side) | **NO** | **GAP** ❌ — v7.5 co-trains pose from ep0 |

### The GAP (R1/cross-run best techniques NOT in v7.5 that arguably should be)

**The single real gap = R1's two-phase pose recipe: converge d_seg first (warm-start a converged basin),
THEN run a dedicated joint pose-descent finish.** v7.5 uses the SAME store-nothing carrier and w_pose 1.0,
but co-trains pose+seg from-scratch across one 3000ep run. R1's evidence is that pose descends to
training-side 0.0011 **specifically when it finishes from an already-converged render** (the render
co-adapts cheaply). v7.5's SPEC §1 POSE LAUNCH GATE independently MEASURES that its as-configured pose
sits at ~1.79 (√(10·1.79)≈4.24 of S) and **"CANNOT reach sub-0.19 as-configured"** — i.e. the from-scratch
single-run pose is the launch blocker, and R1 is the one run that demonstrated a much lower training-side
pose via the two-phase sequence.

**Honest scoping of the gap (do NOT overstate):**
- R1's 0.0011 is **NOT byte-closed** (#238) — it lives in the un-shipped `dxi` table; a serializer that
  ships ξ_eff = ξ_stored + dxi (~7.2 KB, ~0.0005 rate) has never been built+measured through inflate.
  So this is a **JUSTIFIED-but-UNPROVEN** gap: a demonstrated training-side path, not a proven shippable row.
- Therefore the correct action is NOT "bolt warm-start onto v7.5 blindly." It is the SPEC's own pose
  decision (#238/#248): either (a) launch v7.5 explicitly as a **d_seg-only MEANS row** (pose not a
  pointer-mover), or (b) FIRST close #238 (serialize+re-measure R1's ξ_eff, or run a dedicated two-phase
  joint pose-descent to convergence and byte-close it) and re-derive the pose break-even with MEASURED
  d_pose before treating v7.5 as a sub-0.19 attempt.

**mod-dim + all d_seg levers: v7.5 is on the BEST base** (mod-32 = best measured; every d_seg lever from
the best arms inherited; PLUS the net-new Chan-Vese counter-force for the run-1 Road-floor imbalance that
NO prior run had). The ONLY thing the best-of-R1 has that v7.5 lacks is the **pose sequencing** — and that
is exactly the open pose launch-gate, not a forgotten lever.

---

## BOTTOM LINE

1. **R1's only original win is the two-phase pose recipe (warm-start converged basin + dedicated pose-finish)
   → d_pose 0.0011 training-side, NOT byte-closed (#238 pending).** Its d_seg was v2_attrclean's, inherited.
2. **Best measured witness composite d_seg = mod32cap 0.003366 @ep650 (mod-32).** v7.5 inherits mod-32 and
   every best d_seg lever, and ADDS the Chan-Vese counter-force. Per-class binding residual = Lane+Movable
   (must-store) + all-class separatrix; bulk classes near-free at the oracle/warp floors.
3. **mod-dim verdict: mod-32 is the empirically-best measured value (beats mod-26 by ~16%); v7.5 uses it.**
   Theory ceiling 26 superseded by measurement; floor-19 rate-saving A/B is UNMEASURED (open, non-blocking).
4. **The GAP: R1's two-phase pose sequencing is NOT in v7.5 (from-scratch, co-trained pose).** This is the
   SPEC's own pose launch-blocker — resolve via #238 (byte-close R1's ξ_eff) / #248, not a blind bolt-on.
   Every d_seg-side best technique IS in v7.5 and ON.

**Pointer 0.19110 UNMOVED.** All local numbers NON-PROMOTABLE.
