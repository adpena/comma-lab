# Scorer spectral-sensitivity atlas v2 + constants-provenance gate — LANDED 2026-06-09

UTC 2026-06-09 · claude · `[macOS-CPU advisory]` / mechanism + apparatus (NOT a score claim).
Lane `lane_scorer_spectral_atlas_v2_20260609` L1. Source: operator hardening packet 2026-06-09
(the ARBITRARINESS CLASS crux). Sister design memo:
`.omx/research/principled_frequency_basis_synthesis_20260609.md`. Vehicle OS:
`docs/vehicle_operating_system.md`.

## What landed (3 deliverables, 3 commits)

1. **D1 — analyzer v2** (`dc3cd5851`): `tac.analysis.scorer_spectral_sensitivity_v2` (reusable
   physics) + `tools/measure_scorer_spectral_sensitivity.py v2` subcommand (the v1 flat tool became
   the `v1` subcommand; preserved) + `src/tac/tests/test_scorer_spectral_sensitivity_v2.py` (33
   behavioral tests). The hardened transfer-function ATLAS over
   {pair, band, orientation, amplitude_lsb, channel_basis∈{rgb,yuv}, channel, frame_incidence} with:
   amplitude sweep (H is a SURFACE not a curve), THREE response levels (Level-1 SegNet
   source-class logit margin BEFORE argmax flips — we reach the logits via `net.segnet(seg_in)`;
   Level-2 argmax d_seg + boundary/interior flip split by the MEASURED p25 source-margin threshold;
   Level-3 exact d_seg/d_pose/score_nonrate), frame incidence {frame0_only/frame1_only/both_same/
   both_opposite}, RGB AND full-res BT.601 Y/U/V channel basis, oriented bands
   {isotropic/horizontal/vertical/diag±} + random-phase CI, energy audit (pre_clip / post_clip /
   clip_fraction / post_resize_l2 / per-channel — H normalized by ACTUAL injected energy not nominal
   Fourier amplitude), and the **validity-critical coordinate conversion** (per band: camera cyc/px,
   scorer-input cyc/px via the real 874×1164→384×512 resize, normalized ω, SIREN-w-equivalent,
   aliases_at_scorer flag). `band_spacing=log` resolves the w=1..30 regime.

2. **D2 — constants-provenance manifest + Catalog #385 gate** (`bce596cc9`):
   `tac.substrates._shared.constants_provenance_manifest` (typed `ConstantsProvenanceManifest` /
   `ConstantProvenance` / `MeasurementScope`) + `constants_provenance_manifests_canonical`
   (the hi_nerv seed with 13 REAL constants) + `constants_provenance_audit` (the directory+canonical
   scanner) + `check_no_arbitrary_score_relevant_constant_at_l2` (Catalog #385, WARN-ONLY, live
   count 0) + 36 behavioral tests. Per-constant provenance ∈ {DERIVED, MEASURED, LEARNED, ARBITRARY};
   a vehicle declaring L2+ with a (score_relevant OR stability_critical) ARBITRARY constant with no
   `replacement_path` fails closed. **Adversarial guard**: a MEASURED constant is only valid within
   its `measurement_scope` (pairs/frames/amplitudes/surface/CI/artifact) — empty scope → fragility
   advisory ("measured can be cargo-cult too"). **Guardrail (operator-explicit)**: only
   score_relevant OR stability_critical constants block; harmless engineering constants exempt.

3. **D3 — forward-evidence regression** (`96fae8ccc`): named test
   `docstring_mechanism_claim_requires_forward_evidence` in the vehicle_fidelity suite, pinning the
   {function name (file:line evidence), activation test (test_id), export status} contract via the
   `sane_hnerv` FAIL case. Confirmed the existing `verify()` enforces it (claimed-but-absent →
   raises) + the `unproven_claims()` advisory covers the test-coverage leg.

## The headline measurement (the answer to "is w=1 or w=30 sane?")

The validity-critical coordinate conversion (empirically validated against synthesized-band FFT
power to <4% for bands 1-5) yields the decisive finding for the carrier's `sin_frequency`:

- A SIREN `sin(w·x)`, `x∈[−1,1]`, completes `w/π` cycles across the image. So **w=30 ≈ 9.5 cycles
  across the image**; **w=1 ≈ 0.32 cycles** (less than one full cycle).
- The radial bands map to `w_equivalent = π · (r/√2) · camera_extent_px`. Even the LOWEST linear band
  (r_center≈0.083) is already ≈ w_equivalent **162** (~68 cycles across the height); higher bands run
  to w≈1780. Bands above r≈0.4 **alias at the scorer** (scorer_cyc/px > 0.5 Nyquist after the
  874→384 downsample) — the literal w=30 alias mechanism the design memo named.

**Verdict: NEITHER w=1 nor w=30 is "sane" as a hand-tune** — both sit BELOW the frequency band where
the scorer is even resolvable on the radial grid (w=30 ≈ 9.5 cyc/image vs band0 ≈ 68 cyc/image). The
honest reading: `sin_frequency` should be a **per-scale Nyquist-capped + LEARNED ω** initialized from
the MEASURED scorer peak (the atlas's `headline.seg_peak.siren_w_equivalent` / `pose_peak`), not a
global scalar. The atlas (run via the CLI; `log` spacing) places bands across the low-frequency regime
so the measured peak's w-equivalent is the DERIVED replacement value. [Full-grid + fast-grid atlas
JSONs under `/Volumes/VertigoDataTier/pact/scorer_spectral_atlas*_20260609/` — `[macOS-CPU advisory]`,
mechanism_update_eligible only.]

This MEASURED peak is exactly the `replacement_path` the hi_nerv constants-provenance manifest cites
for `sin_frequency` — closing the loop: D1 measures the value, D2 records that `sin_frequency=30` is
ARBITRARY-with-a-measurement-replacement (records debt at L1, would not block at L2 because the
replacement path exists).

## Authority (the metric-laundering firewall)

Everything here is `[macOS-CPU advisory]` / `exact_pair_scorer` → `mechanism_update_eligible` ONLY.
The atlas runs the EXACT frozen `DistortionNet` (read-only; never edits upstream/), but it measures
the scorer's SENSITIVITY (a mechanism fact directing frequency-basis design) — it is NOT a candidate
score, NOT promotable, and does NOT update the score roadmap. The artifact carries
`score_roadmap_update_eligible=False`, `promotable=False`, `mechanism_update_eligible=True`.

## 6-hook wire-in (per Catalog #125)

- #1 sensitivity-map = ACTIVE for D1 (the per-band H_seg/H_pose IS a scorer spectral sensitivity map);
  N/A for D2/D3 (defensive validators).
- #2 Pareto constraint = N/A.
- #3 bit-allocator = ACTIVE (the measured per-band/per-amplitude sensitivity + the water-fill §7 of
  the design memo feed the spectral coordinate of the bit-allocator; the v2 atlas is the measurement).
- #4 cathedral autopilot dispatch = ACTIVE for the Catalog #385 gate (prevents vehicles claiming
  intrinsic optimization while score-relevant constants are guesses).
- #5 continual-learning posterior = N/A (the design memo is the anchor; the MEASURED ω values feed
  the constants-provenance manifest, not a new canonical equation — when a measured ω is adopted as a
  carrier constant it becomes a MEASURED-provenance row with its measurement_scope).
- #6 probe-disambiguator = ACTIVE (the provenance tag DERIVED/MEASURED/LEARNED vs ARBITRARY IS the
  disambiguator; the v2 scorer transfer function is the probe that resolves the SIREN-w instance).

## Scope discipline (what was deliberately NOT touched)

- No carrier forward / renderer edited (hi_nerv/snerv/pact own those; the F1 ablation + SNeRV-B are
  running). The atlas is a frozen-scorer measurement; the constants manifest is observational.
- Catalog #384 NOT flipped to strict (the 3 starved carriers aren't fixed yet); #385 lands WARN-ONLY.
- The `.omx/state/constants_provenance/*.json` seed is gitignored + regenerable (the canonical source
  of truth is the committed `constants_provenance_manifests_canonical` module, which the gate scans
  directly so it works on a fresh checkout).

## Cross-refs

`b1_f1_bilinear_skip_canonical_primitive_landed_20260609.md` (the skip + the w=30 trap) ·
`tac.substrates._shared.vehicle_fidelity_manifest` (Deliverable-1 sister: name-laundering) ·
Catalog #384 (objective-starvation sister) · Catalog #303 (cargo-cult audit; #385 applies it at the
constant level) · `tools/measure_scorer_spectral_sensitivity.py v2` (the measurement actuator).
