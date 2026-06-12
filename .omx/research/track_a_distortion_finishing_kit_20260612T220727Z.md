# Track-A DISTORTION finishing-kit — built, re-fit, wired, integrated, tested (2026-06-12)

**Subagent:** `track-a-distortion-finishing-kit-20260612`. **Type:** build + measure (the inflate-side,
zero/near-zero-byte distortion bolt-ons for the base_ch=20 HNeRV pipeline).
**Evidence grade:** `[contest-CPU advisory] NON-PROMOTABLE` — every number is a frozen-CPU advisory
measurement on a MID-basin fork-point (`best_score=0.529`, ep340/426); no byte-closed `upstream/evaluate.py`
row. **Frontier UNMOVED** (`.omx/state/canonical_frontier_pointer.json` → contest-CPU 0.19109982, 177,169 B).
This is the FINISHING KIT — a post-convergence harvester that is READY to apply to the converged
distortion-arm decoder; it moves no row by itself.

**Commits (code+tests):** `de6cf6597` (kit + driver wiring + 20 tests + probe, +1532) + `c1a331376`
(eval applies the kit POST-round, production-inflate-faithful). **27 tests green** (20 kit + 7 export-faithful);
driver-resume 12 green (default-OFF byte-identity holds). **Probe JSON:**
`.omx/research/track_a_distortion_finishing_kit_probe_n24.json` (32-min run, n=24, real frozen scorer).

**Headline (n=24, advisory):** the full distortion-kit lowers the DISTORTION-score (rate excluded)
**0.47479 → 0.41677 = −0.058**, almost ENTIRELY on the POSE axis (d_pose 0.001478 → 0.000401; d_seg flat
~0.0035). The re-fit PR98 on base_ch=20 is a **pose-axis brightness/temporal-offset lever, NOT a seg lever**,
and the **canonical PR101 constants do NOT transfer** (they are WORSE than zero). S12 is a certified-safe
zero-distortion mask (no byte lever on a render base). LeverD is a measured **NO-GO at convergence** (the GO at
this mid-basin checkpoint is an artifact of its un-converged 0.35 seg term). NONE of this moves the frontier
while base_ch=20 sits at 0.75 — the kit is the post-convergence harvester, READY to apply.

---

## What was built / wired / tested

| Artifact | Path | Role |
|---|---|---|
| Kit module | `src/tac/torch_vehicle/distortion_finishing_kit.py` | `DistortionKitConfig` (PR98 bias + T10 affine + S12 flag) · 54-B section (de)serialize · **pure-numpy** raw-frame postproc + torch camera-float postproc · `LeverDVerdict` |
| Driver wiring | `src/tac/torch_vehicle/driver.py` | `cfg.distortion_kit=None` (default-OFF / byte-identical) · `kit_aware_exact_eval` (faithful eval with the kit hook, mirrors vendored `evaluate_decoder`, NO vendored edit) · `finish_checkpoint_with_distortion_kit` / `split_finished_archive` (append/recover the section) |
| Tests | `src/tac/torch_vehicle/tests/test_distortion_finishing_kit.py` | **19 NO-FAKE test functions (20 cases)** (default-OFF byte-identity · postproc actually changes the targeted slot · section round-trip + fail-closed · numpy↔torch parity · **full vendored-inflate chain** enabled+disabled) — all green; +27 with the export-faithful suite |
| Measurement probe | `experiments/probe_track_a_distortion_finishing_kit.py` | real-scorer A/B/C/D fit + measure on the basin fork-point |

**Integration contract (the operator ask "wired default-OFF + tested"):**
- `cfg.distortion_kit` defaults `None` → NO section, NO postproc, **byte-identical** (the live basin/distortion
  arm is unperturbed if it resumes onto this code — proved by `test_disabled_kit_*` + the full-chain test).
- The postproc is **pure numpy** on the flat raw uint8 `(N,874,1164,3)` the vendored `inflate.py` emits,
  applied AFTER it via the substrate's OWN `inflate.sh` — the pristine PR95 `inflate.py` is NOT edited.
- A single switch (`DistortionKitConfig(enabled=True, ...)`) applies the full distortion finishing-kit to a
  converged checkpoint; `finish_checkpoint_with_distortion_kit` appends the 54-B section; `split_finished_archive`
  recovers `(base 0.bin, kit)` for the inflate path. Inflate stays numpy-portable + ≪ the LOC budget.

---

## A. PR98 channel-bias RE-FIT (n=24, real scorer) — measured

The canonical PR101 constants (frame_0 R−1 / frame_0 B−1 / frame_1 G−1) are substrate-specific; re-derived on
the base_ch=20 render-vs-GT. **Baseline (n=24): d_seg=0.003532, d_pose=0.001478, distortion-score=0.47479.**

- **Re-fit bias (frame,channel):** `[[+1,+1,+1],[−1,−1,−1]]` — subtract 1 from ALL frame_0 channels, ADD 1
  to ALL frame_1 channels (a uniform per-frame brightness/temporal-offset correction).
- **Fit: d_seg=0.003540 (≈flat), d_pose=0.000530 (−64%), distortion-score=0.42681.**
- **Measured gain: distortion-score Δ = −0.047981** (0 archive bytes). `[contest-CPU advisory]`.
- **KEY FINDING — the re-fit PR98 on base_ch=20 is a POSE-axis lever, not a seg lever.** The gain is
  almost entirely d_pose (PoseNet reads BOTH frames; the per-frame offset is exactly what it responds to);
  d_seg (frame_1 argmax) is unmoved. **The canonical PR101 constants are WORSE than zero** here
  (`canonical_pr98`=0.47657 vs `zero`=0.47479) — they do NOT transfer; the re-fit is mandatory.
- The **closed-form mean/std-align OVER-corrects catastrophically** (analytic score 13.5 ≫ base 0.47):
  the render already matches the scorer's argmax (d_seg~0.0035), so a full-stat shift moves the input AWAY
  from the decision boundary. The score lever is the SMALL integer bias (PR98 regime), found by the
  measured ±1 refine. **CAVEAT:** n=24 subset; the pose term `√(10·d_pose)` is sensitive; re-validate via
  `kit_aware_exact_eval` on the converged 600-pair checkpoint before any score claim.

## B. T10 affine beyond PR98 (n=24) — measured

Per-(frame,channel) SMALL local affine around the PR98 operating point. The closed-form mean/std-align
OVER-corrects catastrophically (analytic affine score **92.74** ≫ base 0.47 — reported as the honest comparison
row, NOT selected), confirming the score lever is the small local affine, not the global stat-match.

- **Best scale (frame,channel):** `[[1.01,0.98,1.01],[1.01,1.01,1.00]]` (gentle ±1-2% per channel).
- **Affine: d_seg=0.003534 (flat), d_pose=0.000401 (further −24% vs PR98), distortion-score=0.41677.**
- **Gain BEYOND PR98: Δ = −0.010032** (again pose-axis; CPU-axis fit — CUDA needs its own). `[contest-CPU advisory]`.
- **Full-kit (PR98+T10) distortion-score: 0.47479 → 0.41677 = −0.058 total** (n=24-subset, advisory; the
  `√(10·d_pose)` term makes the pose-axis gains the dominant lever). The kit's exact gain on the converged
  600-pair checkpoint must be re-measured via `kit_aware_exact_eval` before any score claim.

## C. S12 resize-null preimage — measured

- **Certified-invisible fraction of the camera frame:** 0.2270 (22.7% — matches inventory exactly).
- **Per-frame brotli byte reduction (would-be):** 4,307,841 B on the 8 proof frames (22,533,379 → 18,225,538
  = **−19.12%** of coded frame bytes).
- **Zero-distortion certification HELD through the REAL eval round-trip:** max|R x̃ − R x| = **0.0**
  (certified-exact); real-scorer d_seg (0.0036799 → 0.0036799) AND d_pose (0.002204 → 0.002204) are
  **bit-identical** before/after the fill — the fill is provably scorer-invisible.
- **HONEST SCOPE (no fake):** this HNeRV substrate stores decoder+latents, NOT frames — so the −19% byte
  reduction does NOT materialize in the archive (there is no stored-frame section to compress). On this
  render-based substrate **S12 is the CERTIFICATION, not a byte lever**: it proves the invisible-region fill
  is safe on ANY frame-carrying section (a future residual sidecar) and gives the in-frame levers a certified
  safe-perturb mask. The kit carries the certification flag (`s12_invisibility_certified`) at 0 bytes.

## D. LeverD margin-conditional seg-repair — GO/NO-GO (n=24) — measured, with the convergence caveat

- flips/pair = **694** ; mean boundary fraction = 0.54% (thin ∂) ; conditional B/flip = **0.985 < 1.273**
  break-even (CLEARS) ; scaled residual over 600 pairs = **410,462 B**.
- seg drop if all flips fixed (isolated) = **0.353** ; rate cost of the 410-KB residual = 0.273 ;
  net ΔS vs the 177,169-B frontier = **−0.080**.
- **PROBE VERDICT: GO — but this is a MID-BASIN ARTIFACT, the HONEST verdict is NO-GO at convergence.**
  The probe's GO is mechanically correct for THIS fork-point ONLY because its seg term is **0.353** (far from
  converged — the frontier's is 0.056). The −0.080 net credits that full 0.353 seg drop against the residual.
  **At convergence the seg term descends to ~0.056**, so the available seg drop is ~6× smaller (~0.056) while the
  residual stays ~410 KB+ → net **POSITIVE** (the residual section RAISES S). This is exactly the witness-probe
  flip-count crux (`witness_seg_boundary_decisive_probe_20260612`: 884 flips/pair → 543 KB → sidecar dominates).
  The flip count here (694 → 416 K over 600 pairs) is the same order; the per-flip break-even is necessary but
  NOT sufficient — the FLIP COUNT priced against the frontier is the binding term.
- **DECISION: NO-GO for the seg-repair SIDECAR; the d_seg win belongs IN TRAINING** (margin-weighted seg loss,
  Lever 5 — already in the base_ch=20 curriculum) at ZERO added bytes + ZERO round-trip risk (the decoder
  renders the corrected frame). `LeverDVerdict` is carried as a NON-byte record so the autopilot does NOT
  allocate a per-flip seg sidecar on this base. The kit does NOT ship a seg-repair section.

---

## 6-hook wire-in (Catalog #125) + mission

#1 sensitivity-map ACTIVE (the per-(frame,channel) fit = the color-bias sensitivity; S12 mask = per-pixel
invisibility prior). #2 Pareto ACTIVE (PR98/T10 = distortion vertex at 0 rate). #3 bit-allocator N/A (0-byte).
#4 cathedral N/A (export-time finishing pass). #5 continual-learning ACTIVE (re-fit constants reseed the
color-bias-axis judge). #6 probe-disambiguator ACTIVE (this probe disambiguates PR101-canonical vs base_ch=20
re-fit). **Mission contribution:** `frontier_breaking_enabler` (the finishing kit that harvests the last
distortion fraction once the base converges into the frontier neighborhood). **Frontier UNMOVED.**
