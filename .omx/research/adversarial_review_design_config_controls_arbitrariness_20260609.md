# Adversarial review: all design / config / controls / arbitrariness (2026-06-09)

Operator directive: "need adversarial review of all design and config and controls and
arbitrariness." The non-arbitrariness law: **every config/control/threshold must be
DERIVED (from math, a known-good control, or measurement) or explicitly flagged
ARBITRARY_PENDING_MEASUREMENT.** A config chosen by convention/taste is a latent bug.

Live proof this review matters (this session): the recon-fit probe was built with
`lr=2e-3 + grad-clip=1.0` — values chosen by *convention*, NOT derived. The N=1 control
came back FLAT at 7.6 dB and would have produced a FALSE "CAPACITY_CRUX" verdict. The
known-good `one-pair-overfit` control uses `lr=1e-2 + NO clip` (reaches 21 dB). Matching
it (deriving the config from the control) flipped N=1 to **21.32 dB**. An arbitrary config
nearly became a false scientific conclusion. That is the failure mode this review extincts.

## Provenance tags
- **D-MATH**: derived from the contest law / first principles.
- **D-CTRL**: derived from a known-good control (e.g. one-pair-overfit 21 dB).
- **D-MEAS**: derived from a real measurement/anchor.
- **ARB**: arbitrary, chosen by convention — MUST be flagged + calibrated or it can fake a result.

## 1. Capacity-probe configs (`tools/run_hi_nerv_recon_fit_capacity.py`)
| control | value | provenance | verdict |
|---|---|---|---|
| lr | 1e-2 | D-CTRL (one-pair-overfit; validated N=1→21.32 dB) | FIXED (was 2e-3 ARB → false flat) |
| grad-clip | 0.0 / OFF | D-CTRL (one-pair-overfit does not clip) | FIXED (was 1.0 ARB → throttled gradient) |
| verdict thr "fits" | 18 dB | D-CTRL (one-pair-overfit "learned" threshold) | acceptable; state explicitly |
| verdict thr "plateaus" | 8 dB | D-MATH (≈ mean-field PSNR for [0,1] frames) | acceptable |
| batch-pairs | 16 | D-CTRL (matches production trainer batch) | acceptable; note |
| epochs | 2000 | ARB (BUDGET, not a ceiling) | flag: if still climbing at 2000, extend — NOT a "can't fit" |
| eval-sample-pairs | 32 | ARB (spread sample) | acceptable for a readout |
| no-clip @ batch-16/N=600 | OFF | D-CTRL but UNTESTED at N>1 | **OPEN RISK**: control was N=1; multi-pair no-clip may diverge. If Arm A diverges, that is a *finding*, add a clipped Arm A'. |

## 2. Authority-trace configs (`tools/run_hi_nerv_authority_trace.py`)
| control | value | provenance | verdict |
|---|---|---|---|
| fp16 codec | fp16_brotli_legacy | D-CTRL (the lossless-weight control vs int8) | sound; the controlled variable was correct |
| `--out-row` flag | (bridge arg) | D-CTRL (verified against argparse) | FIXED (was invented `--json-out` ARB → bridge-fail false signal) |
| `_SEG_GOOD` | 0.30 | ARB | did not bite (fp16 0.5048 clearly bad) but arbitrary; calibrate before a near-boundary verdict |
| `_POSE_SANE` | 1.0 | ARB | a frontier pose is ~1e-4; 1.0 is a generous guess; 151 clearly exceeds so it didn't bite |

## 3. Next-action router thresholds (`harvest_evidence.route_substrate_next_action`)
| threshold | value | provenance | verdict |
|---|---|---|---|
| seg_flat_reference | 0.50 | D-MEAS (R2 empirical collapse floor) | grounded |
| seg_descend_margin | 0.02 | ARB | flag; calibrate from the noise floor of d_seg |
| pose_sane_ceiling | 1.0 | ARB | flag; should be derived from the frontier pose-band, not a round number |
| _POSE_DELTA_MARGIN | 0.05 | ARB | flag |
These worked for ep250 (unambiguous Case F: d_seg 0.50 / d_pose 151). They are **NOT
trustworthy near the boundaries** until calibrated against measured d_seg/d_pose noise.

## 4. Production HiNeRV training configs (R1/R2/R3) — the BIGGEST arbitrariness surface
The diverge → dead → diverge sequence is itself the signature of unprincipled config:
| run | config | result | the arbitrariness |
|---|---|---|---|
| R1 | amplification ON, no grad-clip | diverged (CE 18→400) | grad-clip ABSENCE was an arbitrary omission |
| R2 | grad-clip ON, amplification OFF | dead latents (3000 ep flat) | amplification toggle was arbitrary |
| R3 | amplification ON, grad-clip 1.0 | diverged after ep201 | grad-clip VALUE (1.0), amplification MAGNITUDE, 600-ep budget all ARB |
**The structural arbitrariness: score-aware loss applied from epoch 0**, vs PR95's
recon-first 8-stage curriculum (CE-Seg → τ-softplus → smooth → QAT → C1a → λ → σ → **Muon
last**). We toggled knobs (amplification, clip, weights) instead of *deriving the
curriculum*. Every value — warmup=10, cosine, distillation-weight=1.0, curriculum stage
epochs, 600 vs PR95's 29,650 — is ARB relative to PR95's measured schedule.

## 5. Optimizer arbitrariness (the relayed correction — and it's correct)
"AdamW vs Muon" is an ARB binary framing. PR95 is **AdamW/Adam formation + staged
scorer/QAT/rate curriculum + FINAL Muon continuation** (Muon on matrix-like decoder
weights, last stage only — NOT from epoch 0, NOT on latents). The production
`pact_muon_adamw` grouping must be audited: is Muon applied final-stage-only on square
matrices (D-CTRL, PR95-faithful) or from-the-start-on-everything (ARB)?

The DERIVED optimizer stack (each arm a measurement, not a taste call):
- **Arm A** AdamW-all — capacity baseline (RUNNING; D-CTRL config). *Does it fit at all?*
- **Arm B** AdamW formation → final Muon on matrix-like decoder weights only (PR95 geometry).
- **Arm C** AdamW latents + **Aurora on rectangular projections** (`latent_embed`,
  `mid_injector.proj`, `fine_injector.proj`) — tests Aurora's dead-row hypothesis directly.
- **Arm D** AdamW latents + Aurora rect-proj + final Muon on square matrices (the aligned stack).
Aurora is D-MEAS-*pending*: it's a grounded hypothesis (`tilde-research/aurora-release`:
row-uniform Muon for non-square matrices; reduces to Muon for square) that BECOMES
non-arbitrary only when the dead-row telemetry confirms the pathology. comp-Muon stays
out of HiNeRV (no QK/OV attention) — it's for PR110++ selector×menu / pact_nerv_vq codebook.

## 6. Required telemetry to make the optimizer choice D-MEAS (non-negotiable)
Per arm: frame0/frame1 PSNR; d_seg/d_pose proxy; latents update-norm; latents cross-pair
variance; render_hash_unique_count; grad+update norms by group; **rectangular-projection
row diagnostics** (row_update_norm p10/p50/p90, row_weight_norm p10/p50/p90,
**dead_row_fraction**, activation_variance_by_row, **effective_rank / singular spectrum**,
row-leverage entropy). Without these, any "optimizer X seemed better" is ARB. Aurora helps
iff: lower dead_row_fraction + higher effective_rank + higher pair-diversity + better PSNR.

## 7. The remedy (binding going forward)
1. Every new probe/trainer config carries a provenance tag (D-MATH/D-CTRL/D-MEAS/ARB) in
   its docstring or emitted artifact. Silent convention values are forbidden.
2. ARB controls near a decision boundary MUST be calibrated (router/trace thresholds) or
   the verdict is downgraded to "unreliable near boundary."
3. The production training crux fix is a **derived recon-first curriculum** (not knob
   toggling): establish RGB fit (Arm-A-proven config) → anneal seg/pose in → QAT → final
   Muon — every value traced to PR95 or measured, not guessed.
4. The optimizer is a STACK chosen by the dead-row/effective-rank telemetry, not a name.

## Labels
authority = false_authority_research_capacity_probe; promotion_eligible = false;
score_claim = false. This is Vehicle-1 carrier+optimizer DIAGNOSIS feeding V3; V3 remains
the judge (candidate archive → exact d_seg/d_pose/bytes → CandidateActionEvaluation → ΔS).

## Cross-refs
- `evaluator_optimal_adaptive_waterfilling_non_arbitrariness_synthesis_20260609.md` (the non-arbitrariness law).
- `pact_nerv_vq_maturity_audit_for_codebook_investment_20260609.md` (sibling carrier; same 4.5 dB failure).
- `b1_r3_descent_smoke_refined_diagnosis_20260609.md` (the R1/R2/R3 sequence).
- `src/tac/optimization/composition_carrier_registry.py` (the carrier triage surface).
