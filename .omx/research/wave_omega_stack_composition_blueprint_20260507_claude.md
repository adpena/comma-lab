---
title: Wave-Ω stack composition blueprint
date: 2026-05-07
author: Track 2 architecture subagent (claude-sonnet-4-6) + Track-1 coordinator persistence pass
status: DESIGN — implementer takes over from here
sources_read:
  - .omx/research/council_22_22_GO_wave_omega_FIELDS_MEDAL_20260501.md (council vote)
  - reference_pr56_selfcomp_blob_byte_layout_proper_reverse_engineering_20260501.md
  - reference_pr67_line_search_R_D_joint_coordinate_descent_20260501.md
  - feedback_sjkl_c067_lane_arity_drift_20260505.md (claim INVALIDATED)
  - feedback_may_4_hnerv_race_postmortem_20260505.md
  - src/tac/sjkl_basis.py (718 LOC, codec complete; fisher_matvec/lanczos_topk/effective_rank stubs)
  - experiments/build_sjkl_residual.py (NotImplementedError at :179)
  - experiments/prepare_sjkl_pair_tensors.py + experiments/build_sjkl_c067_archive.py (CLI verified)
  - scripts/remote_lane_sjkl_c067.sh (verified — arity drift FIXED in commits 8452f26a + d2aa0589)
  - submissions/robust_current/inflate_renderer.py (SJKL_PAYLOAD_FILENAME wired; NeRV mask branch missing; SZv1 dispatch present)
  - src/tac/nerv_mask_codec.py (codec complete; inflate wiring missing)
  - experiments/results/apogee_int6_repack_20260504_claude/repack_metadata.json (dispatch_blockers present)
---

# Wave-Ω stack composition blueprint

## Coordinator rigor note

This ledger is a design/control-plane artifact, not score evidence. All score
rows below are `prediction` until the exact archive bytes pass
`archive.zip -> inflate.sh -> upstream/evaluate.py` on CUDA with a structured
`contest_auth_eval.json`. The numerical targets are retained only as routing
priors for build order, dispatch selection, and adversarial review.

## Critical discovery (re-baselines this whole work)

Reading `scripts/remote_lane_sjkl_c067.sh` lines 323-350 on disk shows **all CLI flags
match the live argparse of every Python target** (`prepare_sjkl_pair_tensors.py`,
`build_sjkl_residual.py`, `build_sjkl_c067_archive.py`). The Bug Class A arity drift
described in `feedback_sjkl_c067_lane_arity_drift_20260505.md` was **already
remediated** in recovery commits 8452f26a + d2aa0589. **The 2-3h structural rewrite
budget allocated for SJKL plumbing is unnecessary.**

The remaining Omega-1 blocker is **3 NotImplementedError stubs** in
`src/tac/sjkl_basis.py` (lines 676 / 685 / 693) plus the CUDA scorer wiring at
`experiments/build_sjkl_residual.py:179` — **~150-200 LOC total, 3-4h dev**.

## Architecture decision

Compose the Wave-Ω stack on top of **PR106 frontier (186,080 bytes, 0.20935
[contest-CUDA])** — NOT on apogee_int6 (170,450 bytes, dispatch blockers
unresolved). PR106 is the cleanest base because:

- It carries a verified [contest-CUDA] baseline
- Its archive format is well-understood (OWV3 weight codec + masks.mkv + poses)
- `build_sjkl_c067_archive.py` adds `sjkl.bin` as a sibling member without
  mutating any existing member byte-for-byte
- The inflate side is already wired (`SJKL_PAYLOAD_FILENAME = "sjkl.bin"` at
  `submissions/robust_current/inflate_renderer.py:92`)

## Component design

### Omega-1: SJ-KL Fisher residual (compress-time scorer feedback)

**Status of code on disk:**
- COMPLETE: `encode_full_sjkl_payload`, `decode_full_sjkl_payload`,
  `encode_sjkl_basis`, `compute_sjkl_basis_lanczos` (CPU Lanczos with HVP),
  `encode_sjkl_alpha_block_v2_sparse`
- COMPLETE: inflate-side dispatch in `inflate_renderer.py` (line 92)
- COMPLETE: shell wrapper at `scripts/remote_lane_sjkl_c067.sh` (arity drift
  fixed)
- MISSING: `src/tac/sjkl_basis.py:676` `fisher_matvec(seg, pose, frames, v)` —
  Fisher information matrix-vector product via Hutchinson estimator or paired
  forward+backward through SegNet+PoseNet
- MISSING: `src/tac/sjkl_basis.py:685` `lanczos_topk(matvec, n, k, ...)` — thin
  wrapper around the existing `compute_sjkl_basis_lanczos` Lanczos iteration
- MISSING: `src/tac/sjkl_basis.py:693` `effective_rank` — participation ratio
  utility
- MISSING: real scorer wiring at `experiments/build_sjkl_residual.py:179` — load
  JointFrameGenerator + SegNet + PoseNet, build differentiable
  `score_fn = lambda frames: 100*seg_dist(frames) + sqrt(10*pose_dist(frames))`,
  pass to `compute_sjkl_basis_lanczos`

**Predicted byte cost / score delta:**
- sjkl.bin: ~10.4KB (`12 * 600 + 0.4*8*1024 = 10477 bytes` per council §5.3)
- Rate delta: `+25 * 10477 / 37545489 = +0.0070` [predicted]
- Distortion delta: -0.020 to -0.080 score on combined seg+pose [predicted]
- Net: -0.013 to -0.073 score from PR106 baseline 0.20935 [predicted]

### Omega-2: NeRV mask codec (mask stream replacement)

**Status of code on disk:**
- COMPLETE: `src/tac/nerv_mask_codec.py` NeRVMaskCodec + NRV1/NRV2 wire format
- MISSING: NeRV mask **inflate branch** in `inflate_renderer.py` mask-loading
  section. Current dispatch handles AV1, CMG1/CMG2/CMG3, CDO1 — needs new
  branch for `b"NRV1"`/`b"NRV2"` magic
- MISSING: `experiments/train_nerv_mask_codec.py` (EMA 0.997 + eval_roundtrip
  per CLAUDE.md non-negotiables)
- MISSING: `experiments/encode_nerv_mask.py` CLI to write masks.nerv
- MISSING: `scripts/remote_lane_nerv_mask_cuda.sh` dispatch script

**Predicted byte cost / score delta:**
- Byte savings: ~196KB (219KB AV1 → ~23.6KB NRV2 per Phase F empirical)
- Rate delta: `-25 * 196000 / 37545489 = -0.131` [predicted]
- Distortion risk: +0.025 if argmax disagreement rises from PR106's 0.0006 to
  1% at NeRV boundary artifacts
- Kill criterion: argmax_disagreement_rate > 1.0% at end of CUDA training

### Omega-3: Block-FP JointFrameGenerator (renderer weight compression)

**Status of code on disk:**
- COMPLETE: `b"SZv1"` dispatch in `inflate_renderer.py:3824` (Lane SZ no-masks
  SegMap path)
- MISSING: `src/tac/block_fp_jfg.py` — block-FP quantizer for JFG (Conv2d +
  FiLM layers, HWOI permute, int8+xz compression). Magic: `b"BFJ1"` (NEW —
  do not overload existing SZv1 dispatch which assumes SegMap arch)
- MISSING: `BFJ1` dispatch in `inflate_renderer.py` (~30 LOC inline
  deserializer)
- MISSING: `experiments/build_omega3_block_fp_archive.py` CLI

**Critical design caveat (Contrarian's note in council review):** PR #56's
Selfcomp 0.387 bpw was a SegMap (no-masks) architecture. JointFrameGenerator
has FiLM-conditioning layers whose weight distributions differ
significantly. The 0.387 bpw transfer to JFG is **unsupported by direct
empirical evidence**. Validate on a single FiLM layer first.

**Predicted byte cost / score delta:**
- Byte savings: ~52KB (56KB renderer.bin → ~4.2KB at 0.387 bpw) [predicted,
  CONDITIONAL on FiLM layer compressibility]
- Rate delta: `-25 * 52000 / 37545489 = -0.0347` [predicted]
- Kill criterion: any FiLM layer effective bpw > 2.5

## Composition contract

| Member | Written by | Read by | Constraint |
|---|---|---|---|
| `p` (QZS3 container) | PR106 packer or stack rebuild | inflate stage 1 | Replaced by Ω-2/Ω-3 stack from scratch; preserved verbatim for Ω-1 alone |
| `sjkl.bin` | Ω-1 `build_sjkl_c067_archive.py` | inflate `SJKL_PAYLOAD_FILENAME` | Sibling addition; must not pre-exist in source archive |
| `masks.nerv` | Ω-2 `encode_nerv_mask.py` | NEW NRV2 inflate branch (TBD) | Replaces masks.mkv inside `p` rebuild |
| `renderer.bfj1` | Ω-3 `build_omega3_block_fp_archive.py` | NEW BFJ1 inflate branch (TBD) | Replaces renderer.bin inside `p` rebuild |

**Composition reality:**

- **Omega-1 alone composes additively** — adds sjkl.bin sibling to PR106
- **Omega-2 and Omega-3 both require rebuilding the `p` container** from
  scratch because they replace renderer.bin / masks within the QZS3 blob
- **Ordering for full stack:** train JFG → block-FP export → renderer.bfj1;
  train NeRV → encode masks.nerv; encode poses; rebuild `p`; add sjkl.bin
  sibling; final archive.zip

## Predicted score arithmetic

```
PR106 baseline:                  0.20935 [contest-CUDA, 186080 bytes]

Ω-1 alone (additive sjkl.bin):
  rate +0.0070 from +10.4KB
  distortion -0.020 to -0.080
  Net:                           0.136 to 0.196 [predicted]

Full stack (Ω-1 + Ω-2 + Ω-3, ~40KB total archive):
  rate ~ 25 * 40000 / 37545489 = 0.027
  distortion ~ baseline+0.005 to baseline+0.025 (cross-layer coupling)
  Net (optimistic):              ~0.150 [prediction, all components clean]
  Net (central):                 ~0.180 [prediction, council §3 consensus]
  Net (pessimistic):             >=0.20935 [prediction, if distortion coupling
                                  overwhelms byte savings]
```

The prior `~0.029` "optimistic" row is not promoted as a roadmap target here:
it requires near-zero scorer-visible distortion after replacing both mask and
renderer streams, which is an extraordinary claim without exact stacked CUDA
evidence. Treat sub-0.15 as an upside research question, not a dispatch
promise.

## Adversarial council review (5/5 ENDORSE)

- **Shannon LEAD** — ENDORSE: independent entropy channels (residual subspace,
  mask rate, weight entropy) → additivity holds. Caveat: SJ-KL Fisher at
  D=589,824 may need k=100 Hutchinson probes for spectrum estimation, not the
  council §5.3 k=8 Lanczos coefficient count.
- **Dykstra CO-LEAD** — ENDORSE: convex-projection feasibility holds; intersect
  region non-empty. Correction: the rate delta on Ω-2 must offset BOTH mask
  shrinkage AND sjkl.bin overhead.
- **Contrarian** — CONDITIONAL: validates the arity-drift correction (saves
  2-3h). DEFERRED on Ω-3 until single-FiLM-layer block-FP validation lands.
- **Quantizr** — PARTIAL ENDORSE: verify whether apogee_int6 archive ships a
  float JFG checkpoint (would skip retraining for Ω-3). Read namelist.
- **Hotz** — ENDORSE with priority correction: ship Ω-1 alone FIRST, smallest
  credible bolt-on (~$3, predicts -0.013 to -0.073 score). Race rule: each
  component generates a public checkpoint, not one monolithic stacked
  submission.

VERDICT: 5/5 ENDORSE; Ω-3 conditional on FiLM validation; implementation
order: Ω-1 first, Ω-2 in parallel, Ω-3 after checkpoint availability.

## Implementation order (Hotz prioritization)

### Phase 1 — Omega-1 (3-4h dev + ~$3 GPU)

1. Implement `fisher_matvec` / `lanczos_topk` / `effective_rank` in
   `src/tac/sjkl_basis.py` (~120 LOC)
2. Wire real scorer in `experiments/build_sjkl_residual.py:179` (~80 LOC)
3. CPU smoke: `--allow-cpu-stub --device cpu --rank 4 --n-pairs 8`
4. Archive integrity test: sjkl.bin roundtrip + magic + member preservation
5. Lane registry: `tools/lane_maturity.py add-lane lane_omega_1_sjkl_pr106`
6. Dispatch claim: `tools/claim_lane_dispatch.py claim lane_omega_1_sjkl_pr106`
7. Lightning T4 dispatch (or Vast.ai 4090 ~$3, 60-disk, cuda_vers>=12.4)
8. Harvest contest_auth_eval.json
9. Mark gates: impl_complete → real_archive_empirical → contest_cuda

### Phase 2 — Omega-2 (parallel to Phase 1; 3-4h GPU + ~$0.85)

1. Add NRV2 inflate branch to `inflate_renderer.py` (~30 LOC)
2. Write `experiments/train_nerv_mask_codec.py` (EMA + eval_roundtrip)
3. Write `experiments/encode_nerv_mask.py`
4. Write `scripts/remote_lane_nerv_mask_cuda.sh`
5. Dispatch on Vast.ai 4090
6. Kill-condition watch: argmax_disagreement_rate > 1%

### Phase 3 — Omega-3 (after Phase 1 or 2 yields a usable checkpoint; ~$5)

1. Inspect apogee_int6 archive namelist for float JFG checkpoint
2. If absent, dequantize int6 → float (verify MSE < 1e-3) or retrain JFG
3. Implement `src/tac/block_fp_jfg.py` (BFJ1 magic, HWOI, int8+xz)
4. Add BFJ1 dispatch to `inflate_renderer.py`
5. Single-FiLM-layer validation (Contrarian's gate)
6. Write `experiments/build_omega3_block_fp_archive.py`
7. Dispatch with kill-condition: any FiLM bpw > 2.5

### Phase 4 — Stack composition (after Phases 1+2+3 land green)

1. Write `experiments/build_omega_stack_p_container.py`
2. Compose: BFJ1 renderer + masks.nerv + poses → `p`; outer ZIP; add sjkl.bin
   sibling
3. Run sanity ladder gates 1-5 on composed archive
4. Dispatch contest-CUDA eval
5. Mark `lane_omega_stack_composed` impl/empirical/contest_cuda

## Sanity ladder gates (per `predispatch_sanity.py`)

For each Phase's first dispatch:

- **Gate 1 (smoke CPU):** `--allow-cpu-stub --device cpu` exits 0; manifest
  `score_claim=false`
- **Gate 2 (lossy<lossless coherence):** predicted_high < lossless baseline
  for the relevant baseline
- **Gate 3 (distortion proxy):** local proxy run + non-proxy parity evidence
- **Gate 4 (hazard scan):** 0 dispatch_local_path_leak hazards
- **Gate 5 (lane registry):** `tools/lane_maturity.py validate` returns 0
- **Gate 6 (apogee evidence semantics):** N/A for Ω lanes (apogee-only gate)

## What this blueprint replaces

- The "Bug Class A SJKL arity drift" task in the post-deadline backlog (CLAIM
  INVALIDATED — already fixed in 8452f26a + d2aa0589)
- The vague "Wave-Ω is launch-ready" task #356 (CONCRETE: 3 NotImplementedError
  stubs + 1 scorer-wiring stub block all Ω-1 dispatch)
- The dispatch order ambiguity for the Wave-Ω 22/22 GO council vote (Hotz:
  ship Ω-1 FIRST, not the monolithic stack)

## Cross-references

- Council 22/22 GO: `project_grand_council_FIELDS_MEDAL_shannon_floor_obsession_20260501.md`
- Selfcomp byte layout: `reference_pr56_selfcomp_blob_byte_layout_proper_reverse_engineering_20260501.md`
- PR67 R(D)-joint: `reference_pr67_line_search_R_D_joint_coordinate_descent_20260501.md`
- May 4 race postmortem: `feedback_may_4_hnerv_race_postmortem_20260505.md`
- Predispatch ladder: `feedback_grand_council_predictor_calibration_no_arbitrariness_20260505.md`
- Lane Ω-W-V3 sensitivity gate: `omega_w_v3_real_sensitivity_gate_20260506_codex.md`
