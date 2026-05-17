# Design memo — fec6 + format0d-EXTRA additive correction (Ext 4)

**Date:** 2026-05-17
**Lane:** `lane_fec6_stacking_wave_5_grammar_extensions_20260517`
**horizon_class:** plateau_adjacent
**Status:** design + scaffold build tool + tests landed; **not dispatch-ready**
until Phase 2 wires the generated `inflate.py` to the canonical fec6 base
inflate path and proves runtime byte consumption.
**Frontier baseline:** fec6 `6bae0201` at `0.19205 [contest-CPU GHA Linux x86_64]` / `0.22621 [contest-CUDA T4]`
**Predicted ΔS band:** `[-0.0005, -0.0030] [predicted, council-consensus]` on contest-CPU axis vs fec6 baseline, conditional on Dykstra-feasibility check (see below).
**Source-supports:** PR106 format0d 2-pass additive correction stream (verified at `submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py:549-575`). PR106 LANES r2 family demonstrates the additive-latent-correction pattern operationally on PR101 grammar.
**paper_claim_scope:** the format0d primitive empirically works on PR101 grammar in the PR106 lane family.
**pact_must_prove:** the same additive-latent-correction pattern stacks cleanly on top of fec6's outer FES1 wrapper (no double-correction conflict, no inflate.py determinism violation, byte-deterministic build, runtime byte-consumption proof).
**decode_complexity_evidence:** PR106 format0d decode is ~30 LOC inside `apply_sidecar_corrections`; trivial per-pair O(1) latent[p, dim_arr[p]] += delta_q[p] * scale.

## Premise (verified per Catalog #229)

PV-1 (PR106 format0d primitive applies to PR101 grammar): VERIFIED at `submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py:549` — `decode_format0d_sidecar(payload) -> (base_dim, base_delta_q, extra_dim, extra_delta_q)` decodes a 2-pass sidecar.

PV-2 (fec6 wraps PR101 payload as opaque bytes): VERIFIED at `tools/build_pr101_frame_exploit_selector_packet.py:1676` — `build_archive_blob(source_archive, selector_payload)` reads PR101's single ZIP member and emits `OUTER_MAGIC FP11 | u32 source_len | source_payload | u16 selector_len | selector_payload`.

PV-3 (the selector_payload slot is extensible): the fec6 outer grammar reserves a `u16 selector_len` so the FES1 payload is variable-length. By extending the outer wrapper with a NEW slot AFTER the selector, the format0d-EXTRA correction can ride alongside the FES1 selector without breaking PR101 source-payload parsing.

PV-4 (no double-correction conflict): the FES1 selector applies to FRAME_0 RGB at inflate time (per `apply_selector_to_frames` in `frame_selector.py`). The format0d-EXTRA correction applies to LATENTS at inflate time (per `apply_sidecar_corrections`). Different stages, different tensors, no conflict.

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Outer archive wrapper | ADOPT canonical (fec6 FP11 outer) | fec6's outer is already byte-deterministic + works with the PR101 source-payload contract. |
| Inflate device selection | ADOPT canonical `select_inflate_device` per Catalog #205 | Universal canonical helper; CPU/CUDA reproducibility. |
| Outer wrapper extension grammar | FORK (unique to fec6+format0d-EXTRA) | The outer wrapper has no existing extension slot. Define a NEW magic `FE6E` (fec6-extra) appended after the FES1 selector with `u32 extra_len | extra_payload` where `extra_payload` is a format0d-EXTRA-encoded blob. |
| Format0d-EXTRA encoder | ADOPT canonical | Reuse PR106's `decode_format0d_sidecar` decode path verbatim; the ENCODER counterpart is to be added at `tac.codec.pr106_format0d_extra.encode_format0d_extra(dim_arr, delta_q_arr) -> bytes`. |
| Per-pair correction discovery | UNIQUE coordinate search on macOS-CPU axis | Offline; no GPU; mirror the `learned_mps_selector` workflow from `build_frame_exploit_selector_packet.py`. |
| Tests | ADOPT canonical patterns from `src/tac/tests/test_pose_delta_codec_v2.py` (round-trip byte-identity + sanity) | Same test surface as sister tac.codec primitives. |
| Recipe | ADOPT canonical schema with `dispatch_kind: tool` per Catalog #270 scope fix | Paired-axis Modal dispatch; no GPU training. |

## 9-dimension success checklist evidence

(All evidence below is `[predicted, council-consensus]` until paired-axis dispatch lands.)

1. **UNIQUENESS** (class-shift not within-class): WITHIN-CLASS. This extension is a bolt-on inside fec6's outer wrapper that adds a corrective stream on existing PR101 latents. Class-shift work is the parallel `tac.boosting` namespace subagent's responsibility.
2. **BEAUTY + ELEGANCE** (PR101-style 30-sec-reviewable): PARTIAL. Build tool and encoder are reviewable, but the generated inflate-side wrapper is Phase 1 scaffold-only and explicitly raises `NotImplementedError` until Phase 2 wires canonical fec6 base inflate.
3. **DISTINCTNESS** (explicitly different from sisters): YES. fec6 is FES1 selector (RGB correction at frame-0). fec6+format0d-EXTRA adds LATENT correction (different tensor, different stage). The two are orthogonal axes.
4. **RIGOR**:
   - Premise verification: see PV-1 through PV-4 above. Verified pre-edit.
   - Adversarial review: design pre-empts the obvious risks (double-correction conflict, payload format conflict, byte-determinism violation).
   - Empirical anchor: PR106 r2 demonstrates format0d works on PR101 grammar empirically at contest-CUDA `0.20533` baseline — close to but not the same as fec6's `0.19205 [contest-CPU]`.
   - Assumption classification: HARD-EARNED (the format0d primitive is empirically anchored in PR106 r2).
5. **OPTIMIZATION PER TECHNIQUE** (substrate-optimal engineering): YES. Per the operator's standing directive, the canonical-vs-unique decision per layer is documented above; we ADOPT where canonical serves and FORK only the outer-wrapper extension grammar (where there is no canonical because the slot is new).
6. **STACK-OF-STACKS-COMPOSABILITY** (orthogonal axes + additive ΔS): YES. The FES1 selector + format0d-EXTRA correction are on DIFFERENT tensors at DIFFERENT inflate stages. Per Dykstra-feasibility (see below), their ΔS contributions should be approximately additive in the small-correction limit.
7. **DETERMINISTIC REPRODUCIBILITY** (byte-stable + seed-pinned): YES for scaffold byte emission. Promotion reproducibility is blocked until the runtime consumes the extra slot and produces full-frame inflate output.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: per-pair coordinate search at ~600 pairs × ~16 dim choices × ~21 delta_q values ≈ 200k macOS-CPU forward passes through the inflate path. Each forward is ~300 ms on M5 Max; total ~17 hours wall-clock for a full sweep. Mitigated by warm-start from PR106 r2's discovered (dim, delta_q) per pair as the seed.
9. **OPTIMAL MINIMAL CONTEST SCORE**: extension target band [-0.0005, -0.0030] on contest-CPU. Gets us to ~0.189-0.191 if the band lands, which is in PR101 GOLD territory.

## Cargo-cult audit per assumption

| Assumption | Classification | Rationale | Unwind path |
|---|---|---|---|
| The PR106 format0d primitive composes additively with fec6 FES1 selector | HARD-EARNED | Different tensors, different stages; PR106 r2 stacks atop PR101 grammar empirically | N/A |
| The optimal (dim, delta_q) per pair on PR106 r2 is also optimal under fec6 FES1 | CARGO-CULTED | fec6 FES1 already corrects frame-0 RGB; the residual latent error pattern may differ | Per-pair coordinate search from scratch on fec6+selector residuals, not from PR106's discovered values |
| 1 pair of (dim_arr, delta_q_arr) per pair is the right correction granularity | CARGO-CULTED | PR106 format0d uses (dim, delta_q) per pair because PR101 latents are (N_PAIRS, LATENT_DIM); we inherit this | Test 2-correction-per-pair variant if 1-correction stalls |
| Byte cost of format0d-EXTRA is ~600 × 2 = 1200 bytes | HARD-EARNED | At 1 byte per dim_arr entry + 1 byte per delta_q entry, the math is exact pre-compression | Brotli the (dim_arr, delta_q_arr) for additional rate gain if the predicted ΔR is too expensive |

## Observability surface

1. **Inspectable per layer**:
   - Build tool emits `experiments/results/fec6_plus_format0d_extra_20260517_codex/build_manifest.json` with per-pair `(dim, delta_q)` and per-pair predicted ΔS contribution.
   - Inflate-time observability: `inflate.py` can print per-pair correction application count when `PACT_INFLATE_DEBUG_FORMAT0D_EXTRA=1`.
2. **Decomposable per signal**: predicted ΔS decomposed into Δ_seg + Δ_pose contributions per pair from the discovery loop.
3. **Diff-able across runs**: byte-deterministic build ensures identical inputs → identical archive (Catalog #158).
4. **Queryable post-hoc**: build_manifest.json is machine-readable; per-pair correction tables exposed.
5. **Cite-able**: build artifact path + commit sha + recipe hash recorded in manifest.
6. **Counterfactual-able**: per-pair (dim, delta_q) MUTATION smoke at the byte-mutation surface (Catalog #139 / #272 / #297).

## Predicted ΔS band

`[-0.0005, -0.0030] [predicted, council-consensus]` on contest-CPU axis.

**Lower bound (-0.0005)**: conservative — assumes only ~3-5 pairs out of 600 admit a non-trivial correction, each with small magnitude. Rate cost ~50-100 extra bytes.

**Upper bound (-0.0030)**: optimistic — assumes ~30-50 pairs admit corrections, each contributing ~5e-5 to Δ_pose. Total rate cost ~600-1000 bytes (still net-positive on the score). Roughly matches PR106 r2's measured ΔS = -0.0033 vs PR101 baseline on contest-CPU axis (empirical lower-bound prior).

**Dykstra-feasibility intersection check**: the rate constraint `R + ΔR ≤ R_budget` + the SegNet distortion constraint `d_seg + Δd_seg ≤ d_seg_budget` + the PoseNet distortion constraint `d_pose + Δd_pose ≤ d_pose_budget` form a 3-constraint polytope. The format0d-EXTRA additive correction acts on per-pair latents which dominantly affect PoseNet output through the HNeRV decoder. Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent": at fec6's operating point (pose_avg ≈ 1e-3, similar to PR106 r2 frontier), pose marginal sensitivity exceeds SegNet's. Therefore latent corrections that reduce d_pose without significantly perturbing d_seg are feasible inside the polytope intersection. Reference: Boyd inner-quintet pact + Dykstra co-lead per CLAUDE.md "Council conduct".

**First-principles citation**: Shannon R(D) lower bound — the per-pair latent-correction stream encodes log2(LATENT_DIM × 2 × max_delta + 1) ≈ 9-10 bits per non-zero correction; ~30-50 non-zero corrections × ~10 bits = 300-500 bits = 38-63 bytes of pure information content. Brotli compresses this to ~30-50 bytes. The bound is well-below the predicted rate cost of 600-1000 bytes (which includes per-pair fixed-width int8 storage uncompressed for byte-deterministic-build simplicity).

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

The current generated `inflate.py` raises `NotImplementedError`. Any Modal,
Lightning, GHA, or local exact-eval dispatch from this scaffold is a bug until
the Phase 2 unblocker lands:

1. wire generated `inflate.py` to the canonical fec6 base inflate path;
2. apply format0d-EXTRA latent correction before the FES1 frame-0 RGB
   selector correction;
3. prove the extra bytes are consumed by mutation/no-op tests;
4. prove full-frame inflate success and axis-labelled component movement.

## Build tool API

`tools/build_fec6_plus_format0d_extra_packet.py`:

```bash
.venv/bin/python tools/build_fec6_plus_format0d_extra_packet.py \
    --fec6-archive experiments/results/.../archive.zip \
    --fec6-source-runtime experiments/results/.../source/submissions/hnerv_ft_microcodec \
    --extra-corrections-json experiments/results/.../format0d_extra_corrections.json \
    --output-dir experiments/results/fec6_plus_format0d_extra_20260517_codex/
```

## Inflate-side wrapper

The inflate.py reads OUTER_MAGIC FP11; if a trailing slot with magic `FE6E` is present after the FES1 selector, decode `(dim_arr, delta_q_arr)` and apply via `apply_sidecar_corrections` from the PR106 r2 module BEFORE the FES1 selector applies its RGB corrections to frame_0.

## Reactivation criteria for the predicted-ΔS band

If contest-CPU paired dispatch shows ΔS outside `[-0.0005, -0.0030]` band by > 0.001:
- ΔS > 0 (regression): refute the additive-stacking hypothesis. Per the cargo-cult audit, re-run discovery from scratch (drop the PR106 r2 warm-start).
- ΔS < -0.0030 (over-performance): document for the next stack iteration; the band's lower bound was over-conservative.
- ΔS strictly within: ratify the prediction and queue Phase 2 (Brotli compression on the (dim, delta_q) pair stream for additional rate savings).

## Cross-references

- PR106 format0d primitive: `submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py:549-575`
- fec6 builder: `tools/build_pr101_frame_exploit_selector_packet.py`
- Inventory: `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
- Premise verifier: `.omx/tmp/fec6_stacking_wave_premise_verifier.txt`
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
