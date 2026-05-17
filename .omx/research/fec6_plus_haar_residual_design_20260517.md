# Design memo — fec6 + Haar 1-level wavelet residual (Ext 5, REFORMULATED)

**Date:** 2026-05-17
**Lane:** `lane_fec6_stacking_wave_5_grammar_extensions_20260517`
**horizon_class:** plateau_adjacent
**Status:** design + scaffold build tool + tests landed; **not dispatch-ready**
until Phase 2 wires generated `inflate.py` to canonical fec6 base inflate,
applies the decoded residual to rendered frames, and proves runtime byte
consumption.
**Frontier baseline:** fec6 `6bae0201` at `0.19205 [contest-CPU GHA Linux x86_64]` / `0.22621 [contest-CUDA T4]`
**Predicted ΔS band:** `[-0.0005, -0.0020] [predicted, theoretical]` on contest-CPU axis (small effect; ~2-5 KB rate cost; minor distortion savings on high-residual regions)
**Source-supports:** wavelet decomposition is a classical orthonormal transform; PR106 LANES r2 demonstrates that small per-frame additive corrections move the score. CompressAI / Ballé hyperprior literature establishes wavelet-domain rate-distortion theory.
**paper_claim_scope:** orthonormal wavelet decomposition preserves L2 energy; per-band scaling allows differential precision; per-band integer mantissa quantization with brotli outer achieves near-Shannon-bound rates for low-energy bands.
**pact_must_prove:** a 1-level Haar wavelet residual on the fec6-rendered RGB frames computes deterministically, the encoded residual stream is byte-deterministic, `inflate.py` decodes + applies the residual additively to fec6's rendered frames, the residual stream is entropy-coded or otherwise inside the marginal byte budget, and the predicted ΔS band stands up to paired-axis dispatch.
**decode_complexity_evidence:** 1-level Haar wavelet decode is ~10 LOC (inverse Haar = average + difference); the additive correction is per-pixel scalar add.

## Premise (verified per Catalog #229)

PV-1 (1-level Haar wavelet is implementable in numpy with no dependencies): Haar's forward transform = `((a+b)/√2, (a-b)/√2)` over 2-element windows; inverse is symmetric. Pure-numpy. ~10 LOC each.

PV-2 (fec6's rendered RGB output is available at inflate time): VERIFIED via `PR101_SELECTOR_INFLATE_PY_TEMPLATE` in `tools/build_pr101_frame_exploit_selector_packet.py:144` — the inflate.py invokes `HNeRVDecoder(...)` then `apply_selector_to_frames(...)` to produce the final RGB frames. The wavelet residual correction would apply AFTER this stage.

PV-3 (additive correction at inflate time is sound): the PR106 format0d primitive (Ext 4) demonstrates additive correction at the LATENT layer. Adding a SECOND additive correction at the RGB layer is well-formed (different layer, different tensor); the order of operations is: PR101 decode → fec6 FES1 selector → fec6+haar wavelet residual additive correction → output.

PV-4 (byte-deterministic build is feasible): the Haar transform of a fixed deterministic per-pair residual is deterministic; per-band fp16 scales + int8 quantized mantissas + brotli compression are all byte-deterministic per Catalog #158.

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Wavelet transform | FORK (numpy-only 1-level Haar) | Existing tac.wavelet_* modules (variance, mask_codec, hnerv_wavelet_*) are domain-specific (mask compression, HNeRV-specific). The Haar 1-level forward+inverse is so simple it's cheaper to inline than to extract a canonical helper. |
| Quantization | ADOPT canonical per-band fp16 scale + int8 mantissa pattern | Same pattern as PR106 LANES, fec6 selector encoder; well-known byte-deterministic. |
| Brotli compression | DEFERRED | Current scaffold emits raw int8 Haar payload bytes. Brotli/ANS/range coding must be measured in Phase 2 before any byte-budget or dispatch claim. |
| Outer wrapper extension grammar | FORK (NEW magic `FE6W`) | Sister of Ext 4's `FE6E` magic; new slot. The outer FE6E and FE6W slots are independent and can both be present in the same fec6+ext archive. |
| Inflate-time decoder | FORK (numpy-only inverse Haar + additive correction) | ~30 LOC; reuses select_inflate_device canonical helper. |
| Tests | ADOPT canonical pattern (round-trip byte-identity + sanity) | Same surface as Ext 4. |
| Recipe | ADOPT canonical schema `dispatch_kind: tool` | Paired-axis Modal dispatch; no GPU training. |

## 9-dimension success checklist evidence

(All evidence `[predicted, theoretical]` until paired-axis dispatch lands.)

1. **UNIQUENESS** (class-shift not within-class): WITHIN-CLASS. RGB-layer additive correction is bolt-on to fec6's rendered output.
2. **BEAUTY + ELEGANCE** (PR101-style 30-sec-reviewable): PARTIAL. Build tool and encoder are reviewable, but the generated inflate-side wrapper is Phase 1 scaffold-only and explicitly raises `NotImplementedError` until Phase 2 wires canonical fec6 base inflate.
3. **DISTINCTNESS**: YES. Ext 4 corrects LATENTS; Ext 5 corrects RGB. Different layer, different scope.
4. **RIGOR**:
   - Premise verification: PV-1 through PV-4 above. Pre-edit verified.
   - Adversarial review: the 1-level Haar restriction is honest scoping per HNeRV parity L7 (bolt-on size budget ≤350 LOC); Daubechies-4 multi-level is substrate-engineering scope and deferred.
   - Empirical anchor: PR106 r2 format0d showed additive corrections work on PR101 grammar empirically.
   - Assumption classification: HARD-EARNED (wavelet R(D) is classical; the additivity is structurally sound).
5. **OPTIMIZATION PER TECHNIQUE**: per the canonical-vs-unique decision per layer table.
6. **STACK-OF-STACKS-COMPOSABILITY**: orthogonal to Ext 4 (LATENT vs RGB layer). Stacks cleanly with Ext 4.
7. **DETERMINISTIC REPRODUCIBILITY**: YES for scaffold byte emission. Current code uses deterministic fp16 scales + int8 mantissas but does **not** Brotli-compress the payload yet. Promotion reproducibility is blocked until runtime consumption and entropy coding are real.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: offline residual computation (~1 hour wall-clock on M5 Max for 600 pairs × 384×512 frames); inflate-time cost ~50ms per frame for inverse-Haar.
9. **OPTIMAL MINIMAL CONTEST SCORE**: predicted ΔS `[-0.0005, -0.0020]`; small but positive.

## Cargo-cult audit per assumption

| Assumption | Classification | Rationale | Unwind path |
|---|---|---|---|
| Haar 1-level is sufficient for the residual | CARGO-CULTED | Inherits the "wavelets are good for natural images" prior without testing whether the fec6 RGB residual has the right band-energy decay | Test 2-level Haar variant if 1-level shows promise; that puts us at the bolt-on/substrate-engineering boundary |
| Per-band fp16 scale + int8 mantissa is the right quantization | CARGO-CULTED | Inherits standard image-coding quantization; could be overkill for low-energy bands (fp8 might suffice) | Test per-band variable bit-width if 1-level shows promise |
| Brotli is the right outer compressor for the integer mantissa stream | UNTESTED | Current implementation does not Brotli-compress the Haar payload; the claim was premature | Measure Brotli/ANS/range/lzma on emitted band streams before byte-budget claims |
| The residual computed on macOS-CPU is byte-identical to the residual on contest-CPU | HARD-EARNED | Haar transform + fp16 + int8 quantization are all bit-deterministic on CPU; verified by select_inflate_device + per-pixel rounding | N/A |

## Observability surface

1. **Inspectable per layer**:
   - Build emits per-band L2-energy + bytes-after-brotli + per-band scale + per-band integer-mantissa quantization error.
   - Inflate-time observability: `PACT_INFLATE_DEBUG_HAAR_RESIDUAL=1` prints per-frame inverse-Haar reconstruction L2 error.
2. **Decomposable per signal**: per-band rate × distortion attribution.
3. **Diff-able across runs**: byte-deterministic per Catalog #158.
4. **Queryable post-hoc**: build_manifest.json with per-band stats.
5. **Cite-able**: build artifact + commit sha + recipe hash in manifest.
6. **Counterfactual-able**: per-band mutation smoke (mutate one band's mantissa byte, verify inflate output changes) per Catalog #139 / #272.

## Predicted ΔS band

`[-0.0005, -0.0020] [predicted, theoretical]` on contest-CPU axis.

**Dykstra-feasibility intersection check**: the additive RGB residual increases R (archive bytes ↑ ~2-5 KB after brotli), decreases d_pose + d_seg (the residual corrects high-error regions). Rate-distortion polytope: feasibility holds when ΔR / Δd is below the marginal trade rate at the operating point. At fec6's operating point (frontier `0.19205 [contest-CPU]`), the marginal rate-to-distortion trade is ~25 (per the contest scoring formula `100 * d_seg + sqrt(10 * d_pose) + 25 * bytes / 37.5M`). 2-5 KB rate cost = 0.0013-0.0033 score contribution; need Δd reduction of at least that magnitude × marginal-d-per-byte-saved to be net-positive.

**First-principles citation**: Shannon R(D) for the per-band residual signal — the wavelet energy compaction property suggests that natural-image residual energy may concentrate in low-frequency bands, but this is a hypothesis for fec6 residuals, not a proven byte budget. Current Phase 1 code emits raw int8 bands with fp16 scales and no entropy coder. Any 3000-5000 byte claim requires a measured Brotli/ANS/range/lzma pass on the emitted band streams.

## Dispatch Authority

This lane is **scaffold-only** as of this memo revision.

The build manifest must carry:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_provider_dispatch=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_attempted=false`
- `runtime_scaffold_only=true`
- `runtime_consumption_proof=false`
- `byte_consumption_proof=false`
- `payload_compression=none_raw_int8_phase1_scaffold`

The current generated `inflate.py` raises `NotImplementedError`. Any Modal,
Lightning, GHA, or local exact-eval dispatch from this scaffold is a bug until
the Phase 2 unblocker lands:

1. wire generated `inflate.py` to the canonical fec6 base inflate path;
2. apply decoded Haar residuals to the rendered RGB frames;
3. prove Haar bytes are consumed by mutation/no-op tests;
4. measure and land entropy coding if raw residual bytes exceed the marginal
   score budget;
5. prove full-frame inflate success and axis-labelled component movement.

## Build tool API

`tools/build_fec6_plus_haar_residual_packet.py`:

```bash
.venv/bin/python tools/build_fec6_plus_haar_residual_packet.py \
    --fec6-archive experiments/results/.../archive.zip \
    --fec6-source-runtime experiments/results/.../source/submissions/hnerv_ft_microcodec \
    --target-video upstream/videos/0.mkv \
    --output-dir experiments/results/fec6_plus_haar_residual_20260517_codex/ \
    --haar-levels 1 \
    --per-band-bits 8 \
    --residual-downsample 4
```

The `--residual-downsample 4` flag indicates that the residual is computed at 96×128 (4× downsampled from 384×512) to keep the byte cost manageable. Per the inventory section §C "Mask resolution / per-class adaptive resolution" precedent, downsampled corrections are byte-efficient when the high-frequency error is concentrated.

## Inflate-side wrapper

The inflate.py reads OUTER_MAGIC FP11; if a trailing slot with magic `FE6W` is present, decode the per-band Haar coefficients, apply inverse Haar, upsample back to 384×512, and additively correct the fec6-rendered RGB frame. Per CLAUDE.md "Forbidden device-selection defaults", uses canonical `select_inflate_device` per Catalog #205.

## Reactivation criteria for the predicted ΔS band

- ΔS ≥ 0 (regression): refute the assumption that residual correction reduces total score at this rate cost. Document; defer to operator review for next stack iteration.
- ΔS ≤ -0.0005 and ≥ -0.0020: ratify; consider Daubechies-4 multi-level escalation (substrate-engineering scope).
- ΔS < -0.0020 (over-performance): document for sister composition work.

## Future work (DEFERRED-pending-research)

- Daubechies-4 multi-level wavelet (substrate-engineering scope per HNeRV parity L7).
- Per-band variable bit-width.
- Wavelet-domain hyperprior (CompressAI / Ballé 2018 pattern).
- All deferred with reactivation criterion: Haar 1-level baseline lands net-positive ΔS.

## Cross-references

- `src/tac/wavelet_variance.py`, `src/tac/wavelet_mask_codec.py`, `src/tac/hnerv_wavelet_*.py` — sister tac.wavelet primitives
- fec6 builder: `tools/build_pr101_frame_exploit_selector_packet.py`
- PR106 format0d additive correction precedent: `submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py:549-575`
- Catalog #205 inflate device-fork canonical helper
- Catalog #158 byte-determinism
- Catalog #295 self-contained inflate.py
- Catalog #270 dispatch_kind: tool scope fix
- Catalog #229 premise verification
- Catalog #287 evidence-tag discipline
- Catalog #290 canonical-vs-unique per-layer decision
- Catalog #294 9-dim checklist
- Catalog #296 Dykstra-feasibility predicted-band
- Catalog #303 cargo-cult audit per assumption
- Catalog #305 observability surface
- Catalog #309 horizon_class
