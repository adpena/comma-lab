# ddm_gdc4 — run-native endpoint generator governed launch — charter, QUEUED-WITH-A-FIRE-ORDER, MAIN 2026-09-10

## Authority and stop state

This is a governed launch specification, not launch authority. `ddm_gdc3` did not train, dispatch, call Modal, or call a scorer. MAIN must explicitly GO, claim a unique lane, and satisfy the fire order below before any training begins. Until then the disposition is **QUEUED-WITH-A-FIRE-ORDER** and the owner is **MAIN**.

The target remains exact description of the move-44 token field in at most 94,010 B as `counted program packet + real exact residual`. The proposed vehicle is a categorical generator whose learned output language is horizontal run endpoints and class labels. It does not predict a dense cell field and then bolt on explicit escape flags. Generic endpoint traversal and rasterization are free receiver code; every video-derived weight, latent, run prior, class prior, and correction is counted.

## Evidence-bound design point

The gdc3 n600 geometry table measured long boundary-adjacent errors as the cheapest class in both predecessor fields: 0.494987 B/mismatch for gdc1 K=8 and 0.246234 B/mismatch for gdc2, after selecting the better of `tile64_time` and `frame_raster` for each source. Use the conservative 0.494987 B/mismatch for admission.

The pre-registered design point is:

- counted packet `K <= 60,000 B`;
- mismatch count `M <= 65,000` on the full n600 move-44 field;
- at least 90% of mismatches by count in long, boundary-adjacent horizontal runs and no more than 1% interior;
- projected `R_exact <= 65,000 * (2,814 / 5,685) = 32,174.1425 B` only as a DERIVED screen, never as the result;
- real `K + R_exact <= 94,010 B`, with `R_exact` coded both `tile64_time` and `frame_raster` orders and selected only after exact decode-to-target proof;
- deterministic receiver wall time `< 1,260 s` for n600 on the declared host.

At K=60,000 B, the conservative geometry rate permits at most 68,708 mismatches; 65,000 preserves 1,835.86 B of design headroom. These are launch thresholds, not measured predictions of training success.

## Vehicle and originality boundary

Use a small shared integer endpoint decoder conditioned on frame index and row, with a counted per-frame latent. It emits a bounded sequence of `(x_stop, class)` symbols per row, rasterized left-to-right. A class-protected occlusion rule must prevent the Movable regression seen in gc1. Lane receives a separately declared loss weight; the endpoint grammar must be able to represent Lane edges at at least the settled 25-pixel round-trip displacement scale.

This is structurally different from:

- gdc1 scanlines, which explicitly stored capped segments for every row and paid a 306 KB-class packet;
- gdc2, whose integer receiver emitted dense categorical cells and left scattered residuals;
- qma9 horizontal-run escape flags, which appended a separate exception family after a base codec; here endpoints are the native learned symbol space and the exact residual remains independently priced;
- gc1 class-blind square paints, which introduced a measured Movable regression.

Literature/OSS anchors are the official [AV1 partition vocabulary](https://github.com/AOMediaCodec/av1-spec/blob/master/07.bitstream.semantics.md) for anisotropic horizontal/vertical partitions, [ITU-T T.82](https://www.itu.int/rec/T-REC-T.82) for progressive bi-level spatial context, and the [COCO mask API](https://github.com/cocodataset/cocoapi) for run-native mask serialization. These are mechanism anchors only; the model, packet ABI, loss, and receiver must be original to this vehicle.

## Deterministic launch contract

- Full population: all 600 token frames, shape `(600, 384, 512)`, source `/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/pass6.u8`, 117,964,800 B, sha256 `78e57545439515eb29f806cc5a5f7d8b14acf658955cdd7561debb4edf3b7db6`.
- Seed: exactly `20260910`, recorded once and routed into Python, NumPy, Torch/MLX, data ordering, initialization, and any entropy-model sampling. Deterministic algorithms are mandatory. NumPy-fp32 or integer reference decode is verdict authority; MPS is advisory only.
- Storage root: `/Volumes/VertigoDataTier/pact/ddm_gdc4_run_native_endpoint_generator/`. Preflight the storage waterfall and require 20 GiB free before launch. Every produced packet, model, latent, render, residual raw stream, coded stream, exact closure field, checkpoint, log, and manifest is retained with bytes and sha256.
- Resume: the launch command must require `--resume-from`. Checkpoints are atomic tmp+rename, content-addressed, and never overwrite an earlier stage. Each carries model, EMA shadow, optimizer, scheduler, global step, stage, all RNG states, config, source hashes, packet ABI, and code commit. Preserve periodic checkpoints at most every 500 optimizer steps and a distinct checkpoint at every stage boundary.
- Decode: a generic, deterministic integer/reference receiver must parse the exact counted packet. Receiver code contains no video-derived constants. A two-repeat render must be byte-identical before any row is admissible.

## Pre-registered stages

1. `stage00_receiver_identity`: no training; packet parse/re-encode identity and reference/integer receiver parity on all n600 row coordinates.
2. `stage10_uniform_endpoint_fit`: train the run-endpoint model with uniform per-cell categorical objective plus explicit endpoint/order validity loss.
3. `stage20_lane25_weight`: resume from the preserved stage10 checkpoint and activate the declared Lane importance weight of 25; no other hyperparameter may change in this contrast.
4. `stage30_rate_constrained`: resume from the better retained stage checkpoint and activate the differentiable counted-packet proxy. Export both the live and EMA weights, but only EMA is eligible for packetization.
5. `stage40_byte_close`: packetize every preserved stage independently, render n600 twice, measure mismatch geometry, code the real exact residual in both orders, and prove exact closure to the target field.

Stage10 versus stage20 is the required Lane-weight A/B. No result may attribute a gain to Lane weighting if any other treatment changed.

## Fire order

1. **MAIN lane claim:** claim a unique `ddm_gdc4_run_native_endpoint_generator` local-training lane and verify no active duplicate in `main_hot_state.md`, the lane registry, or the task ledger.
2. **Implementation landing:** land the trainer, integer/reference receiver, packetizer, resumable checkpoint ABI, retained-payload manifest, and tests through the serializer with two review passes per Python file.
3. **Readiness preflight:** on the exact source hash above, pass storage, deterministic repeat, resume round-trip, stage-boundary checkpoint, packet mutation, and receiver wall-time preflights. Do not launch from a partial readiness result.
4. **MAIN GO:** MAIN explicitly authorizes the full-n600 local training command after inspecting the compiled config and all hashes. No Modal and no scorer are authorized by this charter.
5. **Harvest and byte-close:** retain all stage outputs; measure each full-n600 `K + R_exact` with its own residual geometry. Stop the family if every stage exceeds 94,010 B or if receiver time reaches 1,260 s.
6. **Conditional follow-on:** only if a byte-closed stage is at or below 94,010 B, queue a separate composition/scorer charter with the exact packet/render hashes. Do not infer a score here.

## Fail-closed conditions

Refuse launch or harvest on missing checkpoint state, overwritten stage checkpoints, missing EMA, non-deterministic packet/render bytes, unretained payloads, a video-derived receiver constant, an unpriced packet section, source hash drift, interior-error leakage above the declared threshold, or an active duplicate lane. A packet below 60,000 B without real exact residual closure is not success. A mismatch count below 65,000 in the wrong geometry is not success.

## Consumer and fire trigger

Consumer store: `/Volumes/VertigoDataTier/pact/ddm_gdc4_run_native_endpoint_generator/` plus the eventual compact harvest memo under `.omx/research/`. Fire trigger: MAIN issues explicit GO after the unique lane, implementation landing, compiled configuration, storage preflight, deterministic receiver, and resumable stage-checkpoint gates are all green.


<!-- # FORMALIZATION_PENDING: research memo; the geometry law lands in the equations leg with the first passing construction -->
