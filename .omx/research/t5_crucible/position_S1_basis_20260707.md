# T5 CRUCIBLE — SEAT S1 POSITION — BASIS (Daubechies/Mallat charter)

**Seat:** S1 (vehicle/basis — Arm A `DirectionalBasisRebalance` as PRIMARY d_seg mover)
**Date:** 2026-07-07 · anti-anchoring honored (no other position_S*.md read)
**Authority discipline:** every number below is advisory `[macOS-MLX/CPU research-signal]`
unless stated; pointer contest-CPU **0.19110 UNMOVED** — this file is MEANS.

---

## 1. Position

**Headline: the basis's job has NARROWED.** With the lane class offloaded to the rule-118
analytic band + dash-comb (the FEED-08l-favored dash carrier), the witness's basis carries
only the **C²-cartoon all-class separatrix remainder** — and for that job the optimal
configuration is the *registered two-regime law in its `lane_offloaded` branch, made
Nyquist-clean*. Do **not** chase dashes with linear along-tangent frequency: the n600 ladder
probe measured that hypothesis FLAT/indeterminate-leaning-negative and the comb strictly
better (FEED-08l). The basis is the −48% all-class-directional prior; the comb is the dash
carrier; the band is the lane authority. Three layers, one seam discipline.

### 1.1 Exact recommended configuration (primary run arm)

| Element | Value | Status |
|---|---|---|
| Generic curvelet bank | `--bank-n-scales 4 --bank-n-orient0 6 --bank-f0 2.0 --bank-base 2.0 --bank-n-iso 4` (defaults) + `--max-bank-freq 64` | KEEP (control-proven; max bank f=16 ≈ 4× below stem-Nyquist, cap is a no-op guard) |
| Self-orient directional | `--self-orient` ON, `--reorient-every 50`, `--gpu-reorient` OFF (parity-gated) | KEEP (byte-closeable −48% carrier; tangent transfer measured \|cos\| 0.893–0.909 ≥ 0.85 bar) |
| **Rebalance (THE lever)** | `DirectionalBasisRebalance(freq_across=32, regime="lane_offloaded")` → emits `--n-dir-freqs 4 --freq-across 32.0 --freq-along 6.0` | PRIMARY ARM (b); the registered law's owed A/B anchor |
| **Nyquist-clean variant** | `DirectionalBasisRebalance(freq_across=8, regime="lane_offloaded")` → `--freq-along 4.0`; across ladder {8,16,32,64} all ≤ stem-Nyquist 64 | ARM (c); the source-derived config (see §2-D3) |
| Activation | `--activation hosc --siren-init --hosc-beta 1.0 --hosc-beta-end 4.0`, anneal `geometric` if the ~10 LOC build lands else `linear`; β(Muon-fire)=4.00 (#302 derived endpoint) | KEEP (control-proven; fixed-β diverges) |
| `FinerBiasInit` | NOT in the primary arm; dedicated cheap A/B (RECESS R4) | never-fired; init-confound if bundled |
| `StepNativeActivation` (β 4→8) | NOT in the primary arm; finisher-stage resume-A/B only (RECESS R6) | honors dossier §5 DEFER; unswept |
| Chroma | `--chroma` ON + `--palette-anchor` ON (control-proven) **+** LEVER-4c `--seg-chroma-boundary-weight w --seg-chroma-boundary-margin-band 1.0 --seg-chroma-boundary-start-epoch <tau-stage>` as a treatment lever (RECESS R5); fold as a new DSL `ChromaBoundarySharpen` factory (flags exist, DSL does NOT hold them) | chroma is a measured boundary SHARPENER, not a basis change |
| Capacity | `--mod-dim 32 --hidden-dim 96` primary; mod-48 = the single SECONDARY point, fired ONLY after the basis A/B picks a winner; byte growth routed through Arm E (train-big-compress-small) | basis is PRIOR to capacity (capacity-alone +6% HARMFUL, measured) |
| Lane/dash seam | band ON (`AnalyticLaneRenderBand`, start 350) + `--lane-band-dash-comb` gated on the §21 GT-conditioned comb-registration audit; the basis does NOT budget for dashes | FEED-08l: comb 0.00695 < every freq_along rung |

### 1.2 The allocation derivation (along-vs-across), stated honestly

- **What the measured 3.2× deficit is:** the root cause of dash erasure (4-lens memo) — the
  witness at along=8 was frequency-starved on the along-tangent axis *for the dash texture*.
- **What it is NOT (new, FEED-08l):** a license to raise `freq_along`. The n600
  oracle-capacity ladder (along 0/8/16/25/32) is FLAT (0.00731→0.00714; rungs ≥16
  INDETERMINATE below the GT-separation floor), and the comb (carrier×envelope, 2nd-order
  scattering term) beats every rung (0.00695). Mallat's theorem-level reason: dashes are
  along-ridge amplitude MODULATION, which no first-order oriented basis carries after
  averaging (FEED-08f). So the `lane_carried` regime (along=26) is **demoted to a fallback**
  — admissible only if the band NO-GOs at net-S AND the form-a (retrain) discriminator
  overturns FEED-08l.
- **The `lane_offloaded` branch is the derivation that stands:** with lane out of the carried
  set, remaining edges are C²-cartoon → Candès–Donoho parabolic scaling → along ≈ √across at
  the base (32→6). Caveat (honest, §2-D4): the implementation applies √ at the BASE frequency
  only; the shared 2^k octave ladder then keeps a FIXED anisotropy ratio across scales
  (shearlet-like), not per-rung parabolic. Given FEED-08l this approximation is acceptable —
  do not fund a per-rung-parabolic build for run-1.
- **Nyquist cleanliness is the second derived correction:** the control's across ladder
  (32·2^k, k<4) = {32,64,128,256} puts HALF its across rungs above the SegNet stem-Nyquist
  (64 cycles/unit) — detail the scorer structurally cannot see, which under R (uint8@camera)
  aliases into off-boundary flips (the source module's own derivation). The −48% anchor was
  measured WITH those wasted rungs included, so removing them is upside-or-neutral by that
  anchor — but it is DERIVED, not yet measured; arm (c) + RECESS R1 close it.

### 1.3 What "capacity after basis-match" means concretely

Measured facts: capacity alone on the isotropic basis **+6% (HURTS)**; all-class directional
alone **−48%**; directional + modest capacity (h128/nh5) **−64% at n96** (bytes 50→113 KB).
Therefore: (1) no width/mod-dim change ships in the primary arm; (2) the ONLY capacity
question worth a run is the FEED-07a-binding 2-point {32, 48} secondary axis, fired after the
basis A/B, tracking counted bytes AND d_seg per point (answers §5A/Q9-Q10); (3) any capacity
the residual proves it needs is bought bytes-free first via Arm E (weight-entropy λ∈{5,15,30}
now MLX-ported, #157 bit-alloc, flat-minima) before any counted-byte width increase. The n96
−64% arm is an existence proof, NOT a transferable n600 number (allergic-to-toys).

### 1.4 Chroma routing

SegNet reads RGB; measured (n96 DOF probe a3e9f0bd GREEN): removing chroma flips 7.54%
Lane→Road + 4.38% Movable→Undrivable, with **93.4% of chroma-flips inside the margin<1
annulus** — chroma is a boundary SHARPENER orthogonal to the geometry levers. The basis
already routes capacity there (the per-pixel RGB head has chroma capacity; the annulus is
where the directional features have support); what is missing is SUPERVISION — the rendered
chroma collapses to a near-constant per-class palette because seg-CE only rewards argmax.
The fix is the built LEVER-4c annulus chroma-match term (realized-through-R, luma-invariant,
rides the shared render — no 2nd SegNet forward), NOT a chroma-specific basis. It is
default-off, never fired at n600, and its flags are NOT DSL-held → the fold plan + A/B is
RECESS R5. No witness d_seg verdict in the stack is final with this lever unmeasured
(CLAUDE.md: any witness verdict that ignored chroma is provisional).

---

## 2. Derivations + assumption tags (#363)

- **D1. −48% all-class directional / lane-only −8% / capacity-alone +6% / dir+cap −64%(n96).**
  VERIFIED-VIA-ANCHOR (`.omx/research/witness_capstone_deepmath_levers_20260625.md` table:
  dir_allcls 0.003416 vs baseline 0.008257-class; CLAUDE.md §capstone lever ranking;
  grounding packet "shared facts"). Config of the −48% arm: n_dir_freqs=6, across=32, along=4
  defaults (the "+n_dir_freqs=8, across=48 → −51%" sharp row implies the base row used
  defaults). NOTE: n96 for the −64% composite → hypothesis at n600.
- **D2. FEED-08l ladder verdict (along-freq NOT a confirmed win; comb best).**
  VERIFIED-VIA-ANCHOR (`.omx/research/freq_along_ladder_probe_verdict_20260707.md`, n600,
  GT-validity-controlled: cDC 0.00731 / cF8 0.00756 / cF16+ indeterminate (gt_sep<0.05) /
  cCOMB 0.00695; form-a retrain untested). This is the load-bearing demotion of
  `lane_carried`.
- **D3. Stem-Nyquist cap f_max = scorer_w/(4·stem_stride) = 64 cycles/unit; the over-Nyquist
  waste lives in the self-orient dir feats (across up to 32·2⁵ = 1024 at n=6; 256 at the
  control's n=4); Nyquist-implied configs "n_dir_freqs≤2 @ across=32" or "across=8,
  n_dir_freqs=4".** VERIFIED-VIA-SOURCE
  (`src/tac/boundary_math/lever_b_levelset_generator.py:106-135` — derivation + the
  measured-2026-06-27 note that the default curvelet bank is already sub-Nyquist;
  `directional_fourier_feats` at `src/tac/boundary_math/lever_b_generator.py:159-167` — the
  2^k ladder on both axes, no cap; the trainer consumes it uncapped at
  `experiments/train_levelset_witness_realized_through_R_mlx.py:4722,5286`). The Δd_seg of
  dropping over-Nyquist rungs is UNMEASURED → ASSUMED-upside-or-neutral (they encode
  scorer-invisible detail; the −48% anchor already includes them) → RECESS R1 + arm (c).
- **D4. The two-regime law as implemented: `lane_offloaded` → along = max(4, round(√across));
  `lane_carried` → along = min(across, round(8·3.2)) = 26; lever emits `--n-dir-freqs 4`.**
  VERIFIED-VIA-SOURCE (`src/tac/witness_dsl/curriculum_dsl.py:2133-2176`; registered
  equation `anisotropic_basis_two_regime_allocation_v1` in
  `.omx/state/canonical_equations_registry.jsonl`; √-optimum is
  ASSUMED_AWAITING_VERIFICATION in the registry row — the A/B in §4 is the owed anchor).
  Honest defect noted: √ applies at the BASE only; the shared octave ladder keeps a fixed
  anisotropy ratio per rung (INFERRED from the source construct) — acceptable per D2.
- **D5. Control config ground truth (mod32cap):** `--self-orient --n-dir-freqs 4
  --freq-across 32 --freq-along 8 --max-bank-freq 64 --activation hosc --hosc-beta 1.0
  --hosc-beta-end 4.0 --hosc-beta-anneal linear --chroma --mod-dim 32`, bank flags at
  defaults. VERIFIED-VIA-SOURCE
  (`experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/launch.sh`). The
  ledger's "never-fired" rows are NOT run ground truth (grounding-packet caveat honored).
- **D6. Activation facts:** fixed-β hosc diverges / annealed stable / hosc beat wire 0.221 vs
  0.265 / l7 = measured defect / β(Muon-fire)=4.00 derived / geometric shape derived but a
  ~10 LOC build (linear fallback). VERIFIED-VIA-ANCHOR (CLAUDE.md capstone-trainer section;
  dossier §3-5, §3 knobs table). `LearnableStepBasis` (the torch "step_basis survivor") is
  NOT MLX-ported — the base trainer fail-closes it (VERIFIED-VIA-SOURCE
  `experiments/train_witness_realized_through_R_mlx.py:2789-2802`); on the MLX capstone the
  step-native route IS the β-anneal, i.e. `StepNativeActivation` β 4→8
  (`curriculum_dsl.py:2209-2231`).
- **D7. FinerBiasInit:** flags exist (`--finer-bias-init/--finer-bias-k`, dedicated RNG
  stream, from-scratch-only, fail-closed on relu), published fix for fixed-β saturation
  death, DEFAULT OFF, genuinely never-fired. VERIFIED-VIA-SOURCE
  (`train_levelset_witness_realized_through_R_mlx.py:7641-7654`;
  `curriculum_dsl.py:2354-2380`). Its Δ is INFERRED-from-literature (FINER++, arXiv
  2407.19434) → must be measured before it rides the primary arm.
- **D8. Chroma LEVER-4c:** measured n96 DOF-probe numbers + mechanism (annulus chroma-match,
  luma-invariant, shared-render, fails closed with micro-batch), flags
  `--seg-chroma-boundary-*` exist default-off. VERIFIED-VIA-SOURCE
  (`train_levelset_witness_realized_through_R_mlx.py:3568-3599,7909-7921`). NOT DSL-held
  (grep: no chroma Lever factory in `curriculum_dsl.py`) → unmapped-flag fold owed. n96 →
  n600 transfer is ASSUMED until R5 fires.
- **D9. Seam constraints:** ground-frame-chart v0 FAIL-CLOSES with `--self-orient` and with
  `--render-aa != none` (VERIFIED-VIA-SOURCE trainer help, `--ground-frame-chart` block);
  AA×seed/band/residual incompatible until compose-after-downsample (dossier §6-Q3); the
  ep300 3-way collision measured 3.4× → band start 350 (dossier Arm B). These bound what can
  compose with my basis in run-1.
- **D10. Comb facts:** `--lane-band-dash-comb` exists (render-only, rides the band;
  VERIFIED-VIA-SOURCE trainer lines 3261-3277); comb-registration audit OWED before it fires
  (§21: GT-H shows the ego-phase comb separates GT marks/gaps only weakly — mis-phase risks
  suppressing real lane). VERIFIED-VIA-ANCHOR (dossier §21).
- **D11a. mod32cap is the COUNCIL-DESIGNED CLEAN BASELINE (operator correction honored).**
  VERIFIED-VIA-ANCHOR (`.omx/research/council_symposium_clean_config_20260705.md`, T3
  PROCEED_WITH_REVISIONS): the control's exclusions — no seeding, no lane prior, no analytic
  band, no island birth, `--eikonal-weight 0`, `--verdict-pairs 0` (all-600),
  fresh-from-scratch — are DELIBERATE control design, not gaps; the islands-ON treatment arm
  is separately designed (`council_t3_symposium_islands_treatment_arm_20260706.md`:
  margin-GATED support; uniform amplification measured net-negative; net-positive iff
  n_isl > n_big3). My §1.1 stack + seams (band, comb, seed islands, chroma) are the
  TREATMENT composition built ON TOP of that clean control, A/B'd against it at matched
  epochs. Basis-relevant fact from the same memo: the control's `--freq-along 8` was itself
  the T3 council's deliberate partial rebalance (`--n-dir-freqs 2→4`, `--freq-along 4→8`,
  citing the 3.2× deficit) — designed on then-current evidence, since BOUNDED by FEED-08l
  (raising along further is not a confirmed win); "backwards vs the deficit" framings should
  read as superseded-by-newer-measurement, not as design error.
- **D11. Break-even band:** rate 0.05499 MEASURED (byte-closed archive stat, FEED-07a; quote
  the lane-band counted cost at the LBND4 number 30,892 B ≈ 0.0206 if the band fires, §22);
  pose term p=0.018 ASSUMED/borrowed (witness d_pose OPEN) → d_seg break-even ≈ 0.00118,
  sub-0.15 ≈ 0.00077-0.00092 band. The basis+band+comb+islands composition must be sized
  against THAT, with the composed-surface ceiling arithmetic still OWED (dossier Q1).

---

## 3. PR95 cargo-cult audit (my face)

| Element | Verdict | Basis |
|---|---|---|
| Isotropic random Gaussian Fourier features (Tancik reflex; PR95-family generic-INR front-end) | **DROP/REPLACED** | replaced by the deterministic curvelet bank + self-orient directional basis, DERIVED from the codim-1 separatrix anisotropy (Candès–Donoho) and MEASURED −48% (D1). Only `--bank-n-iso 4` low bands survive, justified as the coarse chart the oriented atoms miss (source docstring). |
| Geometric 2^k octave ladder on dir feats, uncapped | **DROP the over-Nyquist rungs / REPLACE with a Nyquist-respecting ladder** | octaves themselves are derivable (multiscale); the uncapped top rungs are an inherited convenience that violates the DERIVED stem-Nyquist cap (D3). Arm (c) + R1. |
| Fixed anisotropy ratio per rung (√ at base only) | **JUSTIFIED-KEPT (for run-1)** | true per-rung parabolic is a build; FEED-08l says the along-axis is not where the meat is (D2). Named honestly, not laundered as "parabolic". |
| `freq_along=8 = √64` as a "ceiling" | **DROP the ceiling claim** | FEED-08l: config observation, NOT a demonstrated ceiling; the ladder is flat and the comb wins (D2). |
| Control's `along=8` itself | **JUSTIFIED-KEPT as the T3-designed control point** | not cargo: the clean-config T3 deliberately raised 4→8 on the 3.2× anchor (D11a); the A/B (R2) tests it as arm (a), it is not silently inherited. |
| hosc β=4 constant (the activation default) | **DROP/REPLACED** | measured saturation-death; replaced by the β 1→4 anneal, endpoint DERIVED at Muon-fire (D6). |
| siren-init | **JUSTIFIED-KEPT** | from-scratch trainability, literature + control-proven (D5/D6). |
| hosc over wire | **JUSTIFIED-KEPT** | measured A/B 0.221 vs 0.265 (D6). |
| mod-dim ladder as the capacity axis | **DROP as primary** | PR95 capacity reflex (FEED-07a binding); basis-match is PRIOR; 2-point secondary only, after the basis A/B (§1.3). |
| 384×512 render grid | **JUSTIFIED-KEPT** | measured mask-resolution catastrophe class (CLAUDE.md). |
| Chroma as full-RGB reconstruction | **DROP/REPLACED** | the witness is task-space; chroma enters ONLY as the annulus boundary-sharpener (measured mechanism, D8), luma-invariant, not an RGB-fidelity term. |

---

## 4. RECESS measurement proposals

**R1 — $0 over-Nyquist rung inertness probe (forward-only, no GO).**
What: on the frozen mod32cap ep650 + ep1000 checkpoints, zero the dir-feat input columns for
across rungs {128, 256} (k∈{2,3} at across=32, n=4) and re-score n600 through the exact R +
frozen CPU SegNet (the FEED-08c/08l apparatus; chunked resumable foreground).
Command sketch: a probe script reusing `freq_along_ladder` machinery
(`experiments/results/freq_along_ladder_probe_20260707/probe_state.ckpt.npz` pattern), n600,
verdict-batch chunked. Cost: hours-class CPU, <8 GiB chunked — crucible-run, not inline.
Predicted band (grounded in D3 sampling theory: those rungs encode scorer-invisible detail):
|Δd_seg| ≤ 3% relative → rungs inert → arm (c) is free param/alias reduction. Kill/proceed:
Δd_seg > +10% → the trained net uses them as (possibly aliased) carriers → keep across=32
base in run-1 and let the training A/B (R2) decide; do NOT hand-truncate a trained net.

**R2 — the basis A/B (training; operator-GO; THE owed anchor of the registered equation).**
What: 3-arm matched-epoch n600, all with band ON (start 350) + comb gated on R3, identical
seed/schedule: (a) control basis across=32/along=8/n=4; (b)
`DirectionalBasisRebalance(32,"lane_offloaded")` → along=6; (c)
`DirectionalBasisRebalance(8,"lane_offloaded")` → along=4 (Nyquist-clean ladder {8,16,32,64}).
Cost: 3 governed launches (the crucible's schedule seat sizes epochs; matched-epoch
checkpoints suffice — full 1000 ep not required for the pick, per-stage checkpoints
non-negotiable). Predicted band: (b),(c) ≤ (a) on d_seg at matched epochs (−48% anchor says
directional allocation matters; FEED-08l says the along cut costs nothing); (c) additionally
shrinks in_proj counted params. Kill/proceed: if (b) AND (c) are >5% WORSE than (a) at
matched epochs, the rebalance law is implementation-falsified → revert to control basis,
append the negative anchor to `anisotropic_basis_two_regime_allocation_v1`, re-derive.

**R3 — comb-registration audit (already OWED, $0; I ride it).** The §21 GT-conditioned
mark/gap audit gates `--lane-band-dash-comb`. My basis position ASSUMES comb+band carry
lane/dash; if the audit fails and the comb stays off, `lane_carried` (along=26) re-enters as
the fallback arm — but only WITH the form-a retrain discriminator, since FEED-08l's
oracle-capacity form already leans negative.

**R4 — FinerBiasInit cheap A/B (training; short).** What: from-scratch CE-stage-only pair
(~250 ep, n600 verdict at matched epochs), hosc β 1→4, ±`--finer-bias-init --finer-bias-k
10.0`. Grounding: FINER++ (arXiv 2407.19434) — phase-spread first-layer bias = capacity-free
spectral coverage; composes with the directional bank in principle. Predicted: CE-floor
reached earlier or lower; kill if no measurable CE-stage delta → retire-with-reason in the
activation ledger (it stays the fixed-β rescue, unneeded under anneal).

**R5 — LEVER-4c chroma-boundary arm (training; rides any R2 winner).** What: winner-config +
`--seg-chroma-boundary-weight {0.1, 0.3} --seg-chroma-boundary-margin-band 1.0
--seg-chroma-boundary-start-epoch <tau-fire>` (boundary must exist before chroma-matching
pays; start at the CE→tau hand-off, event-anchored if event mode ships). Grounding: measured
annulus chroma mechanism (D8); margin-gradient energy 21.2% chroma. Predicted: d_seg ↓ on
the annulus-jitter component (the ~97%-annulus residual, #333). Kill: d_seg worse at matched
epochs at both weights → retire; the palette stays the carrier. Pre-req: fold the
`ChromaBoundarySharpen` DSL factory (flags exist; never-invent-flags satisfied).

**R6 — StepNativeActivation finisher (resume-based; cheapest possible).** What: at the
primary run's Muon-fire checkpoint, fork: continue β=4.0 frozen vs `StepNativeActivation`
β 4→8 (+ FinerBiasInit is N/A — resume overwrites init, stamped applied:false). Grounding:
β→∞ = step-native (L∞-at-edge, no Gibbs) but τ_end already pins partition sharpness and R
low-passes the render — meat above β=4 is UNKNOWN, exactly why this is a resume-A/B and not
a primary-arm rider. Kill: no d_seg delta at matched finisher epochs → β=4 endpoint stands.

---

## 5. Interfaces

**I provide:**
- The exact primary basis flag set (§1.1) + the two lever calls, both compiling through the
  existing `DirectionalBasisRebalance` factory (no new flags invented; arm (c) is just
  `freq_across=8`).
- The seam contract for the lane/dash stack: band=lane authority (quote counted cost at
  LBND4 30,892 B), comb=dash carrier (gated on R3), basis=cartoon remainder. The basis
  spends ZERO budget on dashes.
- The `ChromaBoundarySharpen` stub-lever spec (D8) for the DSL seat: overrides
  `--seg-chroma-boundary-weight/-margin-band/-start-epoch`, default-off, fails closed with
  micro-batch (already trainer-enforced).

**I need:**
- From SCHEDULE/CURRICULUM: band start 350 confirmed (collision 3.4×); if event-triggered
  mode ships, the chroma start (R5) and any β-anneal endpoint must be boundary-relative
  (β(Muon-fire)=4.00 recomputed if Muon becomes event-fired — dossier §3-5).
- From COSTATE: the basis A/B rows (R2) land in the activation ledger with real activation
  events (the ledger-not-wired-to-runs apparatus gap must be fixed or my levers stay
  falsely "never-fired"); SENSE should carry the per-class annulus flip shares so the
  lane_offloaded assumption is monitored live (if lane share does NOT collapse after band
  start, the regime choice is wrong in-flight).
- From DSL: hold my two lever calls + the chroma stub; note `DirectionalBasisRebalance`
  hardcodes `--n-dir-freqs 4` — correct for arms (b)/(c), but it silently overrides any
  base n-dir-freqs; the WitnessProgram must not double-set it.
- **Declared conflicts (must be adjudicated, not discovered at launch):** (1)
  ground-frame-chart v0 FAIL-CLOSES with `--self-orient` — if another seat wants GFC in
  run-1, it cannot compose with the measured −48% basis; my position: the measured lever
  wins run-1, GFC defers or lands its compose build. (2) `AACoverageRender` incompatible
  with band/seed until compose-after-downsample — AA is a separate arm, not a rider on mine.

**Wall-clock note (lexicographic, score-equal only):** arm (c) shrinks in_feat (fewer dir
columns) → smaller in_proj matmul per step; `CacheGtSkeleton` and `--gpu-reorient` (only if
its parity probe passes) are the score-neutral speed riders on this face.
