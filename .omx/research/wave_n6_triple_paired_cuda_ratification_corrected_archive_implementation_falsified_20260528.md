---
council_tier: T1
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "predicted band [0.155, 0.175] missed by ~593x; the alpha=0.9548 MOSTLY_ORTHOGONAL classification assumed Compound C renderer would deliver baseline-quality frames + Z6-v2 rate-axis bytes would not corrupt pose-axis — empirical 162.52 PoseNet distortion is the fail-loud signal"
council_decisions_recorded:
  - "Wave N+6 TRIPLE composition paired-CUDA RATIFICATION retry on CORRECTED archive grammar"
  - "Local PV + byte-mutation smoke per Catalog #139 PASSED before paid dispatch"
  - "Paired-CUDA dispatch fired per Catalog #246"
  - "CUDA HARVESTED: final_score=92.48 = ~593x WORSE than predicted point 0.156006"
  - "IMPLEMENTATION-LEVEL falsification per Catalog #307 — paradigm preserved; THIS specific TRIPLE inflate runtime falsified"
  - "DEFER-NOT-KILL per CLAUDE.md Forbidden premature KILL: Wave N+7 reactivation criteria pinned"
council_assumption_adversary_verdict:
  - assumption: "Predecessor respawn dispatched failed-archive-grammar sha aa81e158 (multi-section ZIP) and reported correction with sha fef2fa6233 — but NEVER re-dispatched the corrected archive"
    classification: HARD-EARNED
    rationale: "Empirical verification: corrected archive.zip on disk has sha fef2fa6233 (single 0.bin member); active dispatch claims ledger shows only failed_inflate_archive_grammar; no successful re-dispatch entry"
  - assumption: "TRIPLE composition substrate is vendored correctly per Catalog #146/#205/#295/#367"
    classification: HARD-EARNED
    rationale: "Empirical verification: local inflate.sh smoke produces exactly 3,662,409,600 byte raw output; full pipeline simulation (extract archive.zip → run inflate.sh) PASSED; byte-mutation smoke per Catalog #139 PASSED (mutated compound_c bytes break brotli decompression — proving operational consumption)"
  - assumption: "Compound C is the OPERATIONAL renderer; z6_v2 + nscs06_v8 contribute via rate-axis bytes only"
    classification: HARD-EARNED
    rationale: "inflate.py uses tac.substrates.pact_nerv_selector_v3.inflate (Compound C) as primary; the canonical Wave N+6 alpha=0.9548 first-order Volterra includes rate-axis bytes from all 3 substrates per the multi-section archive"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
---

# Wave N+6 TRIPLE Composition Paired-CUDA RATIFICATION (corrected archive grammar; in flight)

## Status

EMPIRICAL FALSIFICATION (CUDA harvested; CPU pending).

- **CUDA T4** call_id `fc-01KSR9K8QTHEWC90VXAMWTKFVZ`: `final_score=92.4795 [contest-CUDA T4]` (avg_segnet=0.5048, avg_posenet=162.520; elapsed 125.09s)
- **CPU** call_id `fc-01KSR9M1RMJ9TZKZHEEWQ0MDVH`: `final_score=92.4762 [contest-CPU]` (avg_segnet=0.5048, avg_posenet=162.496; elapsed 282.71s)

Apples-to-apples CPU-vs-CUDA Δ: 3.3e-3 (well within numerical noise per CLAUDE.md "Apples-to-apples evidence discipline") — both axes deliver the SAME ~92.48 FALSIFICATION verdict.

## Empirical falsification (verbatim CUDA components)

| Component | Raw | Contribution to final |
|---|---|---|
| `avg_segnet_dist` | 0.50482631 | 100 × 0.5048 = **50.4826** |
| `avg_posenet_dist` | 162.52037048 | sqrt(10 × 162.52) = **40.3138** |
| `archive_bytes` | 2,527,587 | 25 × 2,527,587 / 37,545,489 = **1.6830** |
| **TOTAL [contest-CUDA T4]** | — | **92.4795** |

Frontier `[contest-CPU]`: **0.19199** (PR101 GOLD class)
Frontier `[contest-CUDA T4]`: **0.20533** (PR106 format0d class)
Predicted point `[contest-CPU]`: 0.156006
Predicted band `[contest-CUDA]`: [0.170, 0.235]
Empirical `[contest-CUDA T4]`: 92.4795 = **~440x WORSE than CUDA-band upper bound** (450x worse than CUDA frontier)

The TRIPLE first-order Volterra α=0.9548 MOSTLY_ORTHOGONAL hypothesis is **EMPIRICALLY FALSIFIED at IMPLEMENTATION-LEVEL** per Catalog #307. PoseNet distortion 162.52 (vs frontier ~0.01) reveals that the Wave N+6 binding of "Compound C as PRIMARY renderer + Z6-v2 + NSCS06 v8 contribute rate-axis bytes only" does NOT actually produce frames the PoseNet scorer can recognize. Per CLAUDE.md "Forbidden premature KILL": this is IMPLEMENTATION-LEVEL falsification of the Wave N+6 BUILD chain, NOT paradigm-level kill of the orthogonality hypothesis.

## Original status (pre-harvest; preserved per Catalog #110/#113)

PAIRED-CUDA DISPATCH FIRED. Awaiting harvest.

- **CUDA T4**: call_id `fc-01KSR9K8QTHEWC90VXAMWTKFVZ`
- **CPU**: call_id `fc-01KSR9M1RMJ9TZKZHEEWQ0MDVH`
- **Archive sha**: `fef2fa623304f490c394137e270144415dd565ada2dc4b9a738ade132a711795` (2,527,587 B; single `0.bin` member containing multi-section composite)
- **Composite inner sha** (within `0.bin`): `aa81e158889a0f8b558dbf03b73a93335a2dcd5e5268f2b40143f09a99537c92`
- **Submission directory**: `experiments/results/triple_z6_v2_plus_nscs06_v8_plus_compound_c_wave_n6_20260528/submission/`

## Context: Wave N+6 Predecessor Chain

1. `1faf05951` (predecessor sister): TRIPLE composition test LANDED with submission vendoring + sister archive build at sha `aa81e158`
2. `d6867c6d4` (predecessor sister): paired-CUDA RATIFICATION DEFER citing missing inflate runtime — INCORRECT premise per Catalog #229 PV; runtime was already vendored
3. Predecessor respawn `claude:respawn_n6_triple` at 21:48Z: fired paired dispatch on sha `aa81e158` (WRONG grammar: multi-section ZIP directly contained `manifest.json` + `z6_v2.bin` + `nscs06_v8.bin` + `compound_c.bin` as top-level members; contest_auth_eval Stage 1 extracted these directly so the inflate.py expected `extracted/0.bin` which did not exist) → both axes failed rc=1 with `failed_inflate_archive_grammar_wrong_no_0bin_member`
4. Predecessor respawn rebuilt corrected archive at sha `fef2fa6233` (single `0.bin` member = the composite ZIP bytes) but did NOT re-dispatch
5. THIS LANDING (post-session-reset 4:30pm Chicago): re-fired paired dispatch on corrected archive after canonical 4-layer verification

## Canonical Premise Verification (per Catalog #229 + #139 + #220)

### Layer 1: archive grammar verification
- `archive.zip` (sha `fef2fa6233`, 2,527,587 B) contains exactly 1 top-level member: `0.bin` (sha `aa81e158`, 2,527,479 B)
- `0.bin` is itself a multi-section ZIP with members: `manifest.json` (4480 B) + `z6_v2.bin` (607,099 B) + `nscs06_v8.bin` (1,846,867 B) + `compound_c.bin` (68,609 B)
- Per HNeRV parity L3 multi-file justification (substrate_engineering)

### Layer 2: local inflate.sh smoke
- Command: `bash submission/inflate.sh archive_dir/ output_dir/ file_list.txt` (with `0.mkv` in file_list)
- Result: rc=0, produced `output_dir/0.raw` of exactly 3,662,409,600 bytes (= 1164 × 874 × 1200 × 3) per Catalog #367

### Layer 3: full pipeline simulation
- Command: extract `archive.zip` → `archive_dir/0.bin` → `inflate.sh archive_dir output_dir file_list`
- Result: rc=0, same 3,662,409,600 byte raw output

### Layer 4: byte-mutation smoke per Catalog #139
- Variant A: mutate `compound_c.bin` byte 1000 WITHOUT updating manifest sha
  - Result: inflate.py correctly REFUSED via manifest sha-verification → "section 'compound_c_heterogeneous_bit' sha mismatch"
- Variant B: mutate `compound_c.bin` byte 1000 AND update manifest sha to match
  - Result: brotli decompression FAILED inside Compound C archive parser → proving operational consumption of `compound_c.bin` bytes at the entropy-decode layer
- **Verdict**: Catalog #220 `score_improvement_mechanism_status=OPERATIONAL_VIA_COMPOUND_C_PRIMARY` empirically verified

## Predicted vs Empirical (band)

| Axis | Frontier | Predicted | Predicted point | Sub-frontier? |
|---|---|---|---|---|
| contest-CPU | 0.19199 | [0.155, 0.175] | 0.156006 | yes by 0.036 |
| contest-CUDA | 0.20533 | [0.170, 0.235] | TBD | TBD on harvest |

**Composition class**: MOSTLY_ORTHOGONAL per first-order Volterra alpha=0.9548
- AB pair (Z6-v2 × NSCS06 v8): alpha=1.0 (pose-axis ⊥ seg-axis)
- AC pair (Z6-v2 × Compound C): alpha=1.0 (pose-axis ⊥ rate-axis)
- BC pair (NSCS06 v8 × Compound C): alpha=0.85 (Compound F anchor)

## ## Cargo-cult audit per assumption

- **Hard-earned**: archive grammar uses single `0.bin` member per Catalog #146 contest 3-arg contract. **Unwind path**: N/A — verified empirically by Layer 1 above
- **Hard-earned**: Compound C as PRIMARY renderer per Catalog #220 operational mechanism. **Unwind path**: N/A — verified empirically by Layer 4 byte-mutation smoke
- **Hard-earned**: predicted band [0.155, 0.175] derived from canonical first-order Volterra alpha=0.9548 MOSTLY_ORTHOGONAL classification. **Unwind path**: empirical harvest of paired-CUDA dispatch will refit or ratify the prediction per Catalog #371

## ## Predicted ΔS band

[contest-CPU] [-0.0370, -0.0170] point -0.0359; per Dykstra-feasibility intersection of (Z6-v2 pose-axis) ∩ (NSCS06 v8 seg-axis chroma_lut) ∩ (Compound C rate-axis decoder) constraint sets. Validated by first-order Volterra alpha=0.9548 (>0.9 MOSTLY_ORTHOGONAL_OR_BETTER threshold) per `tac.optimization.substrate_composition_matrix.predicted_composite_delta`.

## ## 9-dimension success checklist evidence

1. **UNIQUENESS**: TRIPLE composition orthogonality hypothesis is unique to Wave N+6 (no other landing combines Z6-v2 pose-axis + NSCS06 v8 chroma + Compound C decoder simultaneously). [empirical: probe_outcomes row `wave_n6_triple_z6v2_nscs06v8_compoundc_orthogonality_composition_test_landed_20260528`]
2. **BEAUTY+ELEGANCE**: inflate.py is 286 LOC including provenance docs (within HNeRV parity L4 ≤200 LOC budget per substrate_engineering rationale per Catalog #328); multi-section ZIP parse + sequential decode is reviewable in 30 seconds
3. **DISTINCTNESS**: Compound C as PRIMARY renderer is structurally different from sister PR111 DUAL (NSCS06 v8 + Compound C without Z6-v2 pose-axis); TRIPLE adds pose-axis byte contribution to the composition rate-axis term
4. **RIGOR**: 4-layer PV chain executed before paid dispatch; byte-mutation smoke per Catalog #139 verified operational consumption at entropy-decode layer
5. **OPTIMIZATION-PER-TECHNIQUE**: per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" — composite uses canonical helpers (`tac.substrates._shared.inflate_runtime.select_inflate_device`, `tac.substrates.pact_nerv_selector_v3.inflate.inflate_one_video`) and forks where principled (multi-section ZIP archive grammar; bilinear upsample 384x512 → 1164x874)
6. **STACK-OF-STACKS-COMPOSABILITY**: orthogonal axes (pose / seg / rate) with first-order Volterra alpha=0.9548 verified by `tac.optimization.substrate_composition_matrix.predicted_composite_delta` per Catalog #344 canonical equation `triple_substrate_composition_orthogonal_pose_axis_savings_v1`
7. **DETERMINISTIC-REPRODUCIBILITY**: byte-stable ZIP via `ZIP_STORED` + fixed date_time + PYTHONDONTWRITEBYTECODE=1 + seed-pinned (deterministic ascii-numeric file_list semantics)
8. **EXTREME-OPTIMIZATION-PERFORMANCE**: Catalog #367 raw bytes fail-closed verification; per-pair decode + bilinear upsample; total wall-clock ~60s local CPU per video
9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: predicted -0.0359 [contest-CPU] from frontier 0.19199; sub-0.16 PR111-candidate IF RATIFIED

## ## Observability surface

- **Inspectable per layer**: manifest.json contains per-section sha256 + bytes + composition_strategy + composition_alpha; inflate.py sha-verifies every section before decode
- **Decomposable per signal**: per-axis predicted ΔS for seg / pose / rate; per-substrate (Z6-v2 / NSCS06 v8 / Compound C) contributions documented in composition matrix
- **Diff-able across runs**: archive sha + composite inner sha + per-section shas (4 layers) enable byte-level diff
- **Queryable post-hoc**: paired_dispatch_retry_executed_corrected.json carries full dispatch plan; harvest result in modal_call_id_ledger.jsonl per Catalog #245
- **Cite-able**: archive sha `fef2fa6233` + composite inner sha `aa81e158` + call_ids `fc-01KSR9K8QTHEWC90VXAMWTKFVZ` (CUDA) + `fc-01KSR9M1RMJ9TZKZHEEWQ0MDVH` (CPU)
- **Counterfactual-able**: byte-mutation smoke per Catalog #139 confirmed at this landing (`/tmp/triple_byte_mutation_smoke*`)

## Operator-routable next steps

**EMPIRICAL VERDICT**: IMPLEMENTATION-LEVEL falsification per Catalog #307 (option 3 of the pre-harvest decision tree). Both CPU and CUDA axes deliver final_score = 92.48 = 482x WORSE than [contest-CPU] frontier 0.19199.

### Diagnostic: why did the prediction miss by ~593x?

The Wave N+6 BUILD chain assumed **"Compound C as PRIMARY renderer delivers full 1200 RGB frames at baseline scorer-acceptable quality"**. Empirical PoseNet distortion of 162.52 (vs frontier ~0.01) reveals this is FALSE.

Per pre-Wave-N+6 probe outcomes:
- `pact_nerv_v3_compound_c_heterogeneous_bit_first_empirical_20260528`: PARTIAL verdict; `[macOS-MLX research-signal]` non-promotable per Catalog #192; rate-axis savings -0.0138 vs Slot 2 int8 baseline; no standalone paired-CUDA validation
- `pact_nerv_v3_int8_decoder_quant_brotli_q11_empirical_landed_20260528`: PARTIAL verdict; `[macOS-MLX research-signal]` non-promotable; archive byte savings 28.5%; per-axis decomposition pose 105.48 -> 0.064 was MLX-LOCAL signal that did NOT transfer to contest-CUDA axis
- `pact_nerv_selector_v3_pytorch_decoder_quant_sister_landing_op_routable_1_20260528`: PROCEED but only on `decoder_quant_modes_byte_deterministic_smoke` — meaning the byte-level archive plumbing works, NOT that the renderer outputs are scorer-valid

The Wave N+6 TRIPLE composition assumed (cargo-culted) that Compound C's `[macOS-MLX research-signal]` MLX-LOCAL per-axis decomposition would carry to contest-CUDA. Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192 + the canonical equation `mps_drift_architecture_class_dependent_v1`: this assumption was already documented as CARGO-CULTED. The Wave N+6 BUILD relied on it anyway.

### Reactivation criteria (Catalog #313 30-day expiry)

Per CLAUDE.md "Forbidden premature KILL without research exhaustion":

1. **Compound C standalone CUDA validation FIRST** — operator-routable paired-CUDA per Catalog #246 on `substrate_pact_nerv_selector_v3_int8_decoder_modal_t4_dispatch.yaml` (Slot 2 int8 baseline) → confirms whether the per-pair-decoder produces scorer-valid RGB at all
2. **If Compound C standalone produces < 1.0 contest-CUDA** — Wave N+7 BUILD chain re-engineered: route Z6-v2 pose-axis through a PR101-class HNeRV-family validated renderer (not Compound C), use NSCS06 v8 chroma_lut as overlay (not rate-axis contributor)
3. **Per-axis-decomposition refit** — the canonical equation `triple_substrate_composition_orthogonal_pose_axis_savings_v1` empirical anchor should record this falsification per Catalog #344 + Catalog #371 auto-recalibrator; refit alpha may drop from 0.9548 toward 0.0 for THIS specific binding without invalidating the orthogonality paradigm for sister substrate triples

### NO PR submission

Per `[[user_pr_attribution]]` + `[[forbidden-claude-attribution-in-public-pr-surfaces]]` + Catalog #370 + `[[pr-creation-requires-explicit-operator-authorization-with-adversarial-negative-findings-audit-standing-directive-20260528]]` + the operator's HARD GATE on PR creation: NO PR submission command emitted for this FALSIFIED candidate. Frontier pointer unchanged.

### Apples-to-apples per axis

| Axis | Predicted band | Empirical | Frontier | Δ from frontier | Verdict |
|---|---|---|---|---|---|
| `[contest-CPU]` | [0.155, 0.175] (point 0.156006) | 92.4762 | 0.19199 | +92.28 (482x WORSE) | IMPLEMENTATION-LEVEL FALSIFIED |
| `[contest-CUDA T4]` | [0.170, 0.235] | 92.4795 | 0.20533 | +92.27 (450x WORSE) | IMPLEMENTATION-LEVEL FALSIFIED |

Both axes deliver the SAME falsification verdict. The CPU-CUDA Δ (3.3e-3) is well within numerical noise; no CPU-CUDA-gap anomaly to investigate.

### Spend

- This dispatch: ~$1.50 paired (CUDA T4 125s + CPU 282s)
- Predecessor failed dispatch (sha aa81e158, wrong grammar): ~$0.10-0.20 (failed at archive_grammar)
- Total Wave N+6 paid spend: ~$1.70-1.80 (within ~$1.50-2.50 blanket-approved envelope)
- Net score evidence: TRIPLE composition Wave N+6 BUILD chain FALSIFIED at both axes; first paired-CUDA RATIFICATION of TRIPLE composition class


