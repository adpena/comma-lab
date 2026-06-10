# SCORE-NATIVE first candidate — solver-on-lever_b + legal-frame bridge (tasks #55-react + #56)

**Subagent:** `score_native_first_candidate_55_56`. **Authority of every number below:**
`[local CPU-torch advisory]` — exact upstream SegNet+PoseNet (`DistortionNet`) on CPU, GT decoded
via `upstream/frame_utils.yuv420_to_rgb` ONLY (PyAV rgb24 == ~100× phantom pose). **NOT** the
contest 600-sample harness → non-promotable per the GOAL authority ladder. `$0` spend, no GPU, no
paid dispatch, **NO MPS**. `[macOS-MLX research-signal]` for the generator forward (argmax-parity
portable, MLX→numpy 0.9997). `promotable=false`, `score_claim=false`,
`ready_for_exact_eval_dispatch=false`.

**Spec executed:** tasks #55-reactivation (run `closed_spec_boundary_solver.v1` on the
`lever_b_argmax_generator` base — its DEFER reactivation criterion) + #56 (the legal-frame
variational bridge — the campaign's first build). Source memos:
`closed_spec_boundary_solver_v1_20260610T105830Z.md`, `lever_b_score_native_argmax_smoke_verdict_20260610.md`,
`closed_spec_boundary_math_system_of_equations_20260610.md` (§4, §10),
`information_theoretic_floor_report_v1_20260610T102335Z.md` (S_floor=0.118).

---

## 0. PRE-REGISTERED prediction + kill criterion (written BEFORE the measurement)

**THE THESIS:** Lever B's amortized generator (70KB, d_seg=0.00826) breaks the rate floor but its
seg residual must be driven down. Its residual is structurally CONTIGUOUS patches (smooth-net
under-fit), unlike the frontier's unrepairable salt-and-pepper — so the boundary solver may repair
it net-positively.

**PRE-REGISTERED PREDICTION:** solver-on-lever_b nets d_seg down meaningfully; the byte-closed
score-native candidate scores S < 0.15 (advisory).

**PRE-REGISTERED KILL CRITERION:** if lever B's residual is ALSO net-unrepairable
(new_bad ≥ repaired) AND the raw generator byte-closes worse than the frontier, record the finding
(amortizer too lossy) + reactivate via lever D (STC/UNIWARD code the residual < 1.27 B/flip) or
lever C (better-trained smaller amortizer).

**RESULT against the pre-registration:** the prediction is **HALF-CONFIRMED, HALF-REFUTED**, and the
KILL criterion is **NOT triggered** (it is a DEFER, not a kill):
- **CONFIRMED:** the residual IS contiguous + the solver DOES net d_seg down (the #55 reactivation
  criterion is MET — see §1/§2).
- **REFUTED (the honest miss):** S < 0.15 did NOT land. The advisory S = **13.58** because the
  legal-frame bridge (palette) collapses POSE (d_pose 12.66 vs a measured GT-frame1 floor of 0.0).
  The seg+rate class shift is real; the POSE-carrying appearance section is the unsolved piece
  (exactly the verdict memo's predicted "appearance section" cost, now quantified).

---

## 1. STEP 1a — the generator residual structure (the DECISIVE measurement)

The lever-B generator's argmax residual (`A_g` vs `L*`) on the exact SegNet, 8-pair sample
(`experiments/results/score_native_candidate_20260610/score_native_first_candidate.json`):

| quantity | lever_b generator (THIS) | frontier base (the #55 DEFER) |
|---|---:|---:|
| generator d_seg vs L* | 0.006845 | 5.4e-4 |
| **contiguous flip fraction (≥4px)** | **0.744** | **~0** |
| single-pixel fraction | 0.460 | **0.95** |
| largest flip component | 149 px | 4 px |
| size histogram | {1px:1126, 2px:437, 3px:252, 4-9px:446, 10-49px:159, 50+px:27} | {1px:99, 2-4px:5} |

**THE THESIS IS CONFIRMED.** The generator's residual is **structurally repairable** — 74% of flip
pixels live in contiguous components of ≥4px (a smooth INR's under-fit regions), the EXACT OPPOSITE
of the frontier base's salt-and-pepper (95% single-pixel) that the boundary solver could not repair.
This is the precise reactivation condition the #55 DEFER named.

## 2. STEP 1b + STEP 2 — solver on the legal-frame + bridge cell-landing

The legal-frame bridge synthesizes a SCORER-FREE palette frame from the generator argmax (per-class
GT-region-mean colors, 15-byte palette); its ACTUAL SegNet argmax is measured, then the boundary
solver's `Gα≥b` solve repairs the residual. The three deterministic candidates (8-pair, palette
d_seg → corrected):

| candidate | net Δd_seg | repaired | new_bad | bytes |
|---|---:|---:|---:|---:|
| `contour_normal` | **−0.02501** | 47010 | 7677 | 0 |
| `graph_cut`      | −0.02374 | 44645 | 7309 | 2548 |
| `mdl_contour`    | **−0.02718** | 45841 | **3096** | 4786 |

**ALL THREE net d_seg DOWN** (repaired ≫ new_bad) — the OPPOSITE of the frontier base where all
three were net-WORSE (the #55 finding: GT-snap upper bound was net −536). `mdl_contour` is best
(fewest collateral new_bad). **This is the class-shift finding: the boundary SOLVE pays rent on the
lever_b base.**

**BUT two bridge problems (the honest blockers):**
1. **Imperfect cell-landing.** The palette frame's actual SegNet argmax has d_seg = **0.050**
   (10× the generator's own 0.0068) — a piecewise-constant palette does NOT reproduce the SegNet
   argmax over the boundary bands. The solver pulls it 0.050 → 0.025, still 3× the generator floor.
2. **POSE COLLAPSE.** d_pose = **12.66** with the palette frame1 (sqrt(10·12.66) = 11.25 score
   ALONE). The diagnostic measured the pose FLOOR with GT frame1 = **0.0** (GT frame0 + GT frame1
   reproduces the GT pose exactly), proving the palette IS the entire pose problem — a flat-color
   frame has no luma texture for PoseNet.

## 3. STEP 3 — byte-closed advisory S row + the appearance-section curve

**The byte-closed candidate** (`experiments/results/score_native_candidate_20260610/`,
archive sha `7dc512b5…`, **lossless parity all_match=True over 8 pairs**, scorer-free `inflate.py`
runs standalone):

| section | bytes |
|---|---:|
| generator (seg carrier, int8+brotli) | 65,305 |
| palette (appearance) | 15 |
| pose trajectory (fp16+brotli) | 6,650 |
| **archive.zip total** | **72,217** |
| **rate term (25·B/D)** | **0.0481** |

vs frontier 177,169 B / rate 0.1180 — **−59% bytes** (the rate class shift is REAL and byte-closed).

**Advisory S (8-pair, mdl_contour):** d_seg 0.0228, d_pose 12.66, **S = 13.58** — does NOT beat the
frontier (0.19110). The seg term (100·0.0228 = 2.28) + rate (0.048) are fine; the **sqrt(10·12.66) =
11.25 pose term dominates and kills it.**

**The reactivation evidence — the appearance-section rate-vs-(seg,pose) curve** (a GT-derived low-res
RGB frame1 diagnostic; `lowres_appearance_probe.json`):

| factor | lowres | d_seg | d_pose | bytes/pair (raw) | rate (600pair) |
|---:|---|---:|---:|---:|---:|
| 2 | 437×582 | 0.0006 | 0.0007 | 424 KB | 169.5 |
| 4 | 218×291 | 0.0023 | 0.0084 | 112 KB | 44.7 |
| 8 | 109×145 | 0.0088 | 0.033 | 28 KB | 11.3 |
| 16 | 54×72 | 0.023 | 0.59 | 7.2 KB | 2.9 |

**The decisive diagnosis:** pose IS cheaply recoverable in LUMA structure (factor-8 d_pose 0.033,
factor-4 0.0084 — vs palette's 12.66), BUT a RAW per-pair RGB appearance is **catastrophically
expensive** (factor-8 = 28 KB/pair × 600 = 17 MB, rate 11.3) because it is NOT AMORTIZED. The seg
term was made cheap by AMORTIZING the argmax into the 65 KB shared generator; the pose-carrying
appearance must be amortized the SAME WAY (a frame/luma generator or temporal-residual carrier).
The frontier's 177 KB HNeRV decoder amortizes BOTH seg AND appearance; the score-native carrier has
so far amortized only seg.

## 4. VERDICT

**VERDICT: DEFER-to-amortized-pose-carrying-appearance (NOT kill; the #55 reactivation criterion is
MET, the bridge is the open piece).** Per CLAUDE.md "Forbidden premature KILL" + Catalog #307
IMPLEMENTATION-LEVEL: the paradigm (score-native seg generator + boundary SOLVE) is PROVEN exact and
class-shifting on rate AND on seg-repairability; the SPECIFIC bridge (palette appearance) is
falsified on pose. The KILL criterion is NOT triggered (residual repairable; generator byte-closes
−59% smaller).

**Is sub-0.15 in reach?** NOT with the palette bridge (S=13.58, pose-dominated). It IS plausibly in
reach with an amortized pose-carrying appearance: the seg term is at ~2.28 (100·0.0228) and
descending with training-length; rate is 0.048; the only blocker is pose, which the curve shows is
recoverable at low rate IF amortized. A factor-8-equivalent appearance amortized to the generator's
~65 KB scale (not 17 MB raw) would put d_pose ~0.03 (sqrt = 0.55 score) — the candidate would then
sit near S ~ 2.3 + 0.55 + 0.10 ≈ 2.9, still far above frontier, because the **seg term itself
(2.28) is the next binding constraint** once pose is carried: the generator's d_seg=0.0068 → 100·
0.0068 = 0.68 if the bridge landed it perfectly, but the palette+solver only reach 0.0228. So BOTH
levers must improve: (a) amortized pose appearance, (b) a bridge that lands the generator's own
0.0068 d_seg (not the palette's 0.025). Sub-0.15 requires the seg term ≤ ~0.05 (d_seg ≤ 5e-4, the
frontier's level) AND pose ≤ ~0.01 — i.e. the generator must train to the frontier's d_seg AND the
appearance must be amortized. The class shift (rate −59%) is the headroom that makes this worth
pursuing; the two bridges are the work.

**NO paired exact eval is pre-registered/launched.** The eval gate ("advisory S beats frontier or
hits sub-0.15") is NOT met (S=13.58). Correct fail-closed: do not spend ~$0.6 to confirm a
non-improvement. $0 spent. The lane stays `[local CPU-torch advisory]`.

**Reactivation criteria (the next builds, priority-ordered):**
1. **Amortized pose-carrying appearance** (the dominant blocker): a luma/RGB frame generator or
   temporal-residual carrier amortized like the seg generator (target: pose-appearance < ~30 KB
   total, d_pose < 0.05). The lowres curve (§3) is its rate-vs-pose prior. THIS is the door to a
   real S.
2. **A bridge that lands the generator's own d_seg** (not the palette's 0.025): rasterize from the
   generator's logits/argmax with a margin-aware fill that reproduces the SegNet boundary, OR train
   the generator jointly with the rasterizer so its argmax is the rasterized frame's argmax.
3. **Lever D contour coder** on the (now contiguous) residual: the `mdl_contour` admission already
   prices it; an STC/UNIWARD boundary coder < 1.27 B/flip would let the solver's correction cross
   the water level (the residual IS contiguous now, so chain-coding is efficient — unlike frontier).
4. **Longer/larger generator** (lever C): the generator d_seg 0.0068 (200ep) is ~12× the frontier's
   5.6e-4; a capacity/training-length campaign closes the seg term.

## 5. Wire-in (Catalog #125)
1. **sensitivity-map** — ACTIVE: the residual contiguity histogram (§1) + the appearance rate-vs-pose
   curve (§3) are the new seg/pose sensitivity inputs the waterfiller (#54) consumes; the palette's
   per-class margin is the seg-logit-null prior.
2. **Pareto** — ACTIVE: this build maps the score-native carrier's {d_seg, d_pose, B} surface — it
   establishes that the carrier is OFF the frontier's vertex on rate (−59%) but ON a pose cliff
   (palette appearance). The Pareto-feasible move is the amortized appearance.
3. **bit-allocator** — ACTIVE: the byte breakdown (generator 65K + palette 15 + pose 6.65K) is the
   literal allocator; the next term to allocate is the amortized-appearance section.
4. **cathedral-autopilot** — the smoke → (conditional) paired-eval dispatch surface; gate NOT met
   (no advisory beat).
5. **continual-learning** — this verdict reseeds the planner: (a) the boundary SOLVE pays rent on a
   CONTIGUOUS-residual base (the #55 DEFER is reactivated — solver works here), (b) the score-native
   carrier's binding term flips from rate (lever B's win) to POSE once seg is amortized, (c) the
   pose-appearance must be amortized, not stored raw per-pair (the lowres curve quantifies it).
6. **probe-disambiguator** — RESOLVED: "is the lever_b residual repairable by the boundary solver?"
   → YES (net Δd_seg −0.027, repaired 45841 ≫ new_bad 3096; vs frontier net-negative). "Does the
   palette bridge land in the cell + hold pose?" → NO (cell-landing d_seg 0.050; pose 12.66 vs floor
   0.0). The next probe: an amortized pose-carrying appearance generator.

## 6. Deliverables + cross-references
- **Byte-closed candidate:** `experiments/results/score_native_candidate_20260610/`
  (archive.zip 72,217 B sha `7dc512b5…`, scorer-free inflate.py, decoded_frames/, manifest.json with
  lossless-parity proof all_match=True).
- **Reusable code (NO-FAKE, tested):** `src/tac/boundary_math/lever_b_generator.py` (checkpointed
  generator + residual stats) + `legal_frame_bridge.py` (palette + lowres-appearance carrier) +
  26 behavior tests (`tests/test_lever_b_generator.py` 11, `test_legal_frame_bridge.py` 8 +
  lowres-carrier coverage); 49 boundary_math tests green; ruff clean.
- **Tools:** `tools/lever_b_train_generator_checkpoint.py` (the missing artifact — trains + saves the
  generator; the smoke never did), `tools/score_native_first_candidate_smoke.py` (Steps 1-3),
  `tools/score_native_lowres_appearance_probe.py` (the appearance-section curve),
  `tools/score_native_build_byte_closed_candidate.py` (the byte-closed archive + parity proof).
- **Cross-refs:** `closed_spec_boundary_solver_v1_20260610T105830Z.md` (the #55 DEFER this
  reactivates) · `lever_b_score_native_argmax_smoke_verdict_20260610.md` (the generator) ·
  `closed_spec_boundary_math_system_of_equations_20260610.md` (§4 polytope, §10 water level) ·
  `information_theoretic_floor_report_v1_20260610T102335Z.md` · `upstream/{modules.py,frame_utils.py}`.
