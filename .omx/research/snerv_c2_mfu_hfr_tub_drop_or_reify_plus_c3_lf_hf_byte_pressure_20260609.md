# SNeRV C2 (MFU/HFR/TUB DROP_OR_REIFY binding) + C3 (LF/HF byte-pressure curve)

- **Agent**: swarm V2-SNERV-C2
- **Lane**: `lane_swarm_v2_snerv_c2_mfu_hfr_lfhf_20260609`
- **Date**: 2026-06-09
- **Authority**: research-only / false-authority. NO score claim, NO promotion,
  NO rank/kill, NOT ready for exact-eval dispatch. All advisory rows carry
  `score_claim=False`, `promotion_eligible=False`, `promotable=False`,
  `ready_for_exact_eval_dispatch=False`.
- **Spend**: $0 (CPU-portable, no GPU, no paid dispatch).

This closes Phases C2 and C3 of the SNeRV vehicle, building on C1 (the TUB
`DROP_OR_REIFY` source-forward proof, commit `06f3dc580`,
`build_snerv_official_tub_drop_or_reify_source_forward_proof`). It reuses the
**C1 causality criterion** verbatim: a source-state facet is *receiver-causal*
iff a smallest-step flip changes the **float receiver RGB** (re-run the REAL
receiver-RGB primitive); uint8/scorer survival is reported **separately**.

---

## C2 — MFU/HFR/TUB source-state DROP_OR_REIFY binding

**New API**:
`tac.analysis.snerv_official_source_forward_harness.build_snerv_official_mfu_hfr_tub_drop_or_reify_source_forward_proof`
(schema `snerv_mfu_hfr_tub_drop_or_reify_source_forward_proof.v1`).

**Mechanism** (CPU-portable, deterministic):
1. Run the pinned upstream `model.snerv.SNeRV` source graph once via the
   sparse-dyadic fixture (`_build_official_fixture`) — the SAME real source graph
   C1 uses — and render the base receiver RGB through the REAL receiver primitive
   `OfficialMfuHfrTubReceiverPayload.decode_frames` (which re-runs the portable
   MFU forward `low/skip_mid/skip_high -> pyr_out`, the portable HFR heads
   `pyr_out -> yh_out`, and the stored `output_2` temporal-fusion frame residual).
2. For each of 8 canonical source-state facets, flip the lowest mantissa bit and
   re-render the receiver RGB through the same primitive. Classify at the float
   receiver-RGB boundary.
3. REIFY iff receiver RGB changes AND survives uint8 AND a real `scorer_fn` is
   supplied (emit a base-bound `CandidateActionEvaluation`); DROP iff the receiver
   never consumes the facet; else REIFY_PENDING_SCORER.

### Key finding: the conservative minimal flip is arithmetic-annihilated

The lowest-mantissa-bit flip (~1e-17) on every MFU/HFR/TUB facet propagates to
**exactly 0.0** receiver-RGB change. The reason is real and important: the
sparse **dyadic** fixture weights are exact float64 fractions, and the conv /
transpose-conv / upsample arithmetic **rounds the sub-1e-17 perturbation back to
the identical float64 result** through the decoder stack. This is the same class
of phenomenon C1 saw with `yl_norm` ("sub-uint8 at the fixture") but stronger:
here it is sub-*float* at the receiver.

To answer the *structural* causality question honestly (does the facet reach the
receiver RGB **at all**), a second probe uses a minimal **representable dyadic
step** (`2**-6`) — the smallest source-faithful step that survives the
dyadic-fixture arithmetic. A facet is receiver-causal iff EITHER probe changes
the float receiver RGB. Both probes re-run the SAME real receiver primitive; no
fabrication.

### C2 verdicts (research-only, real source graph)

| Facet | Family | Receiver key | Verdict | float linf (2^-6 probe) |
|---|---|---|---|---|
| `mfu_skip_high` | mfu | `inputs.mfu.skip_high` | **REIFY_PENDING_SCORER** | 0.125 |
| `mfu_rb_high_residual` | mfu | `mfu.rb_high.input_conv.weight` | **REIFY_PENDING_SCORER** | 0.801 |
| `hfr_lh_head` | hfr | `hfr.lh.conv1.weight` | **REIFY_PENDING_SCORER** | 0.0068 |
| `hfr_hh_head` | hfr | `hfr.hh.conv2.weight` | **REIFY_PENDING_SCORER** | 0.052 |
| `tub_output_2` | tub | `tub.output2_raw` | **REIFY_PENDING_SCORER** | 0.0156 |
| `mfu_low` | mfu | `inputs.mfu.low` | **DROP** | 0.0 |
| `mfu_skip_mid` | mfu | `inputs.mfu.skip_mid` | **DROP** | 0.0 |
| `tub_temporal_encoder` | tub | `tub.temporal_encoder_concat` | **DROP** | 0.0 |

- **Headline**: `REIFY_PENDING_SCORER`. **Family verdicts**: mfu / hfr / tub all
  `REIFY_PENDING_SCORER` (each family has ≥1 receiver-causal facet).
- **DROP rationale**: `mfu_low` / `mfu_skip_mid` do not reach the final receiver
  frame planes (the low/mid-resolution contribution is annihilated by the
  frame-plane mapping that consumes only the final pyramid resolution + HF skip);
  `tub_temporal_encoder` is the output_2-decoder input that the *frame* decode
  does not consume (only `output2_raw` is added as a frame residual). These are
  metadata-only at the receiver → 0 bytes earned.
- **NOTE for C1 reconciliation**: C1 found `output_2` is DROP in the *bare-frame*
  fixture (`official_tub_frame_reconstruction_numpy`, no output_2 parameter).
  Here `output_2` is **REIFY-causal** because the MFU/HFR/TUB receiver payload was
  built with `store_tub_output2_for_receiver_proof=True`, so
  `_apply_official_tub_output2_frame_residual` adds it to the frame. Both are
  correct: causality is **receiver-path-specific**, and the proof flips the facet
  the receiver path actually consumes.

### CandidateActionEvaluation binding (the shared waterfilling currency)

A source-state flip **adds 0 bytes** (`with_action_archive_sha256 ==
base_archive_sha256`, `bytes_with_action == bytes_base`), so `delta_bytes == 0`
and admission is **pure distortion-ΔS** (`delta_score_total ==
delta_score_nonrate`). Verified both directions:
- A scorer that makes the candidate distortion **increase** → `delta_score_total
  > 0`, `pays_rent=False`, `verdict=reject` (correct: a perturbation that worsens
  RGB does not pay rent).
- A scorer where the candidate **fixes** base distortion → `delta_score_total =
  -1.3162`, `pays_rent=True`, `verdict=admit`, `delta_bytes=0`.

This is exactly the binding the CLAUDE.md SNeRV hard blocker asks for: each
receiver-causal MFU/HFR/TUB facet now flows through the shared
`evaluator_action_waterfill.CandidateActionEvaluation` law, base-bound to its
receiver-payload sha (`base_archive_sha256`) so it goes stale on base change.

### Gate wiring

`build_snerv_official_tub_lf_hf_replacement_authority_gate` gained an optional
`mfu_hfr_tub_drop_or_reify_proofs` parameter. When supplied it adds a
`mfu_hfr_tub_drop_or_reify_binding` gate row (depends on
`full_tub_source_forward_replay`). `binding_ready=True` when the source graph ran
AND ≥1 facet is receiver-causal — i.e. the source-forward binding question is
answered. The row stays **blocked** on
`snerv_official_mfu_hfr_tub_drop_or_reify_scorer_delta_pending_real_scorer` until
a real scorer lands (binding proven, score authority pending C4). Omitting the
parameter preserves the prior gate DAG unchanged (default empty).

---

## C3 — LF/HF carrier byte-pressure ↔ receiver-RGB-collapse curve

**New API**:
`tac.analysis.snerv_lf_hf_replacement_queue.build_snerv_lf_hf_byte_pressure_curve`
(schema `snerv_lf_hf_byte_pressure_curve.v1`).

The existing `snerv_lf_payload_codec_sweep` is **rate-only** (bytes, no visual
metric, no scorer replay). C3 adds the missing **distortion axis**: it drives the
REAL LF/HF wavelet carrier (`encode_frame_lf` pyramid → `quantize_lf` at a sweep
of quantization granularities `n_levels` → `encode_lf_quant_payload_v2` entropy
code → `dequantize_lf` → `decode_frame` inverse-DWT + HF restorer) and measures
the receiver-RGB collapse at each byte operating point: frame float linf, uint8
linf, and the **argmax-disagreement RATE** (the functional the SegNet `d_seg`
term is defined over).

### Finding 1: graceful LF-coarsening collapse (smooth content, default)

With a smooth LF frame (faithful finest-quant baseline), the receiver RGB
collapses monotonically as bytes are squeezed:

| n_levels | LF payload bytes | uint8 linf | argmax-disagree | collapsed |
|---:|---:|---:|---:|:--:|
| 256 | 108 | 0 | 0.000 | no (baseline) |
| 128 | 108 | 2 | 0.393 | no |
| 64 | 108 | 3 | 0.620 | **yes** |
| 32 | 108 | 6 | 0.773 | yes |
| 16 | 96 | 14 | 0.859 | yes |
| 8 | 96 | 31 | 0.962 | yes |
| 4 | 90 | 67 | 0.981 | yes |
| 2 | 90 | 114 | 0.990 | yes |

Collapse onset (argmax-disagreement ≥ 0.5) at `n_levels=64`. The collapse is
**steep and early**: even modest coarsening flips a majority of pixels' class.
This empirically grounds the CLAUDE.md SNeRV hard blocker "LF/HF representation
collapse under real byte pressure".

### Finding 2 (the bigger blocker): HF-restorer instability on ANY HF content

The single-frame least-squares HF restorer (`fit_hf_decoder_least_squares`)
**diverges catastrophically on any nonzero high-frequency content** — even at the
finest quantization the receiver frame leaves `[0,255]` by orders of magnitude
(e.g. amplitude=2 → base recon range `[-63057, 63439]`; amplitude=12 → `[-5176,
5519]`). This is a structural failure independent of byte pressure. The C3 curve
flags it via `finest_baseline_faithful=False` / `hf_restorer_diverges=True` and
the blocker
`snerv_lf_hf_byte_pressure_hf_restorer_diverges_finest_baseline_unfaithful`.

**Implication for the LF/HF replacement family**: the LF carrier degrades
gracefully but is rate-limited; the HF carrier's least-squares restorer is the
weak link — it does not generalize across LF quantization drift and is unstable
on real HF content. A learned/robust HF restorer (or an explicitly
regularized/clamped one) is the lever the LF/HF replacement queue should target.

### CandidateActionEvaluation binding (rate-distortion frontier)

With a real `scorer_fn`, each byte-pressure point emits a base-bound
`CandidateActionEvaluation` against the finest baseline. A byte-pressure step
**removes** bytes (`delta_bytes < 0`), so under a scorer indifferent to the pixel
change the step is unconditionally rent-paying (the rate term drops, distortion
unchanged) — verified. The curve thus doubles as the LF-carrier rate-distortion
frontier the waterfilling law selects an operating point from.

---

## Tests (NO-FAKE)

16 new tests, all behavioral (real source graph / real receiver RGB / real
carrier; never asserting constants):
- `test_snerv_official_source_forward_harness.py`: 7 C2 tests (real source graph
  ran; facets classified by **measured** receiver causality; DROP facets really
  leave receiver RGB unchanged for both probes; CAEs have `delta_bytes==0`;
  rent paid iff scorer lowers score; missing-checkout fail-closed).
- `test_snerv_lf_hf_replacement_queue.py`: 5 C3 tests (real carrier ran; collapse
  **monotone** in pressure and crosses the threshold; HF-restorer divergence
  flagged with the baseline really out of `[0,255]`; rate-distortion CAEs;
  empty/invalid sweep fail-closed).
- `test_snerv_official_tub_lf_hf_replacement_authority_gate.py`: 4 gate tests
  (binding row omitted without proof; real proof drives binding_ready + scorer
  blocker; no-causal-facet blocks; wrong-schema ignored).

Full owned-module suite: **99 passed** (83 prior + 16 new). With key downstream
consumers (`nerv_witness_readiness_dag`, `nerv_long_training_campaign_plan`):
**244 passed**, 0 regressions. Ruff clean on all 6 files.

---

## Blockers for the next agent (C4 — exact eval → waterfiller currency)

1. **C2 REIFY_PENDING_SCORER → REIFY requires a real SegNet/PoseNet** on contest
   hardware (Linux x86_64 CPU / NVIDIA CUDA). The 5 receiver-causal MFU/HFR/TUB
   facets are proven to reach the receiver RGB and survive uint8; the missing
   piece is the real `scorer_fn` `(base_rgb_uint8, candidate_rgb_uint8) ->
   {"d_seg","d_pose"}` so the CandidateActionEvaluation `delta_score_total` is
   authoritative (not the non-authority reference scorer). Blocker token:
   `snerv_official_mfu_hfr_tub_drop_or_reify_scorer_delta_pending_real_scorer`.
2. **C3 byte-pressure curve uses a non-authority reference scorer** — the real
   `d_seg`/`d_pose` at each LF operating point must be measured on contest
   hardware. Blocker token:
   `snerv_lf_hf_byte_pressure_real_scorer_terms_pending_contest_hardware`.
3. **HF-restorer instability** (`hf_restorer_diverges`) is the dominant LF/HF
   collapse blocker; it is structural (not byte-pressure) and needs a robust HF
   restorer before the LF/HF carrier is a viable replacement backend.
4. **Fixture, not trained checkpoint**: C2 runs the real source *graph* with
   sparse-dyadic fixture weights (CPU-portable), not the trained official
   checkpoint. The facet causal map (which facets the receiver consumes) is
   structural and should hold for trained weights, but the scorer-ΔS magnitudes
   are fixture-specific until C4 runs the trained receiver under the real scorer.

## Files

- `src/tac/analysis/snerv_official_source_forward_harness.py` (C2 proof)
- `src/tac/analysis/snerv_lf_hf_replacement_queue.py` (C3 byte-pressure curve)
- `src/tac/analysis/snerv_official_tub_lf_hf_replacement_authority_gate.py` (C2 gate wiring)
- `src/tac/tests/test_snerv_official_source_forward_harness.py` (C2 tests)
- `src/tac/tests/test_snerv_lf_hf_replacement_queue.py` (C3 tests)
- `src/tac/tests/test_snerv_official_tub_lf_hf_replacement_authority_gate.py` (gate tests)
