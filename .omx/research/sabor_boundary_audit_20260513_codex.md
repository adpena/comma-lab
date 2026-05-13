# SABOR Boundary Audit - phi1 Codex Memo - 2026-05-13

Lane: `lane_sabor_boundary_audit_20260513`  
Evidence axis: `[macOS-CPU advisory]` only  
Verdict: `GO-FOR-PROTOTYPE` for a small byte-closed research prototype, not for
score promotion or dispatch.

## Scope And Preflight

- Repo: `/Users/adpena/Projects/pact`, branch `main`, HEAD observed
  `ce8fdcc7`.
- No GPU spend, no provider dispatch, no score claim.
- Lane registry contains `lane_sabor_boundary_audit_20260513` at L0.
- Active claims search found no SABOR claim conflict; existing active/conflict
  surfaces are on SIREN/HNeRV/T1/S2SBS-adjacent work and were not touched.
- No `.omx/research/*_directive_*` files dated within the last 24 hours were
  present.
- Partner/in-flight untracked files outside phi1 scope were left untouched:
  `.omx/research/s2sbs_blindspot_audit_20260513.md` and
  `tools/measure_scorer_hf_blindspot_capacity.py`.

## Tool Hardening Landed

File: `tools/measure_segnet_argmax_stable_interior.py`

- Preserves the upstream CPU scorer path: PyAV frame decode through
  `frame_utils.yuv420_to_rgb`, non-overlapping `seq_len=2` pairs, HWC uint8
  camera-size validation, TCHW float conversion, then
  `SegNet.preprocess_input(pair_5d)` before `SegNet.forward`.
- Replaced the risky direct-resized-frame scorer shortcut with a guarded
  `_segnet_logits_from_pair(...)` contract check for `(1,2,3,H,W) -> (1,3,384,512)`.
- Empirical perturbation accounting now records both
  `stable_fraction_mean_per_perturbation` and stricter
  `stable_fraction_all_samples`; free-byte capacity estimates use the stricter
  all-samples fraction.
- Aggregate per-class empirical stability is weighted by class pixel count,
  not by an unweighted per-frame average that could hide absent/rare classes.
- Output artifacts include `research_only=true`, `score_claim=false`,
  `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`, and
  `[macOS-CPU advisory]` evidence labels.
- Output artifacts now hash the input video and SegNet safetensors checkpoint.

## Focused Guards

File: `src/tac/tests/test_measure_segnet_argmax_stable_interior.py`

The tests cover the highest-risk bug classes without running a real scorer:

- HWC uint8 decode contract and non-overlapping pair preservation.
- Rejection of non-uint8 decoded RGB tensors.
- Mandatory 5D `SegNet.preprocess_input` path before forward.
- Separation of mean-per-perturbation stability from all-samples stability.

Commands:

```bash
.venv/bin/python -m py_compile tools/measure_segnet_argmax_stable_interior.py src/tac/tests/test_measure_segnet_argmax_stable_interior.py
.venv/bin/python -m pytest src/tac/tests/test_measure_segnet_argmax_stable_interior.py -q
```

Result: `4 passed in 1.24s`.

## Advisory Smoke

Command:

```bash
.venv/bin/python tools/measure_segnet_argmax_stable_interior.py \
  --n-frames 2 \
  --n-perturbation-samples 1 \
  --perturbation-subset-stride 1 \
  --epsilon-list 1,4 \
  --margin-thresholds 0.5,2.0 \
  --num-threads 2 \
  --save-spot-check-frames 1
```

Artifact directory:

`experiments/results/lane_sabor_boundary_audit_20260513_20260513T175742Z/`

Hashes:

- `stable_pixel_capacity.json`:
  `c5bc6453653da1a68c99e6430c06d88a55296fd82c20d9b6961820e4a4eb216d`
- `build_manifest.json`:
  `5d0ca92132da797ad827242f7965b8454a3739ab961155ab5fd06f5d95e6c4f7`
- `per_frame_records.json`:
  `56d85df7cadac254dec4d071eae6138c62bdcf69ed7b53ca50d0f5391adbac08`
- Input video SHA-256:
  `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`
- SegNet safetensors SHA-256:
  `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`

Observed `[macOS-CPU advisory]` smoke metrics:

- Frames measured: `2` non-overlapping last-pair frames.
- Margin proxy stable fraction: `margin > 0.5 = 0.9867324829`;
  `margin > 2.0 = 0.9542922974`.
- Empirical `epsilon=1`: all-samples stable fraction `0.9998474121`.
- Empirical `epsilon=4`: all-samples stable fraction `0.9993998210`.
- Conservative capacity estimate at `epsilon=4`: `24561.25` bytes/frame.

## Critical Caveats

- This is not a contest score, not `[contest-CUDA]`, not `[contest-CPU]`, and
  not promotion evidence.
- Smoke is only `n=2`, `K=1`; all-samples stability equals one sampled
  perturbation and must not be treated as a robust 600-frame conclusion.
- The CPU decode path intentionally matches `AVVideoDataset`/PyAV semantics,
  not CUDA/DALI semantics; no CUDA-axis inference is allowed.
- IID RGB perturbation stability is a necessary signal for SABOR, not a proof
  that a constructive boundary-only renderer can carry useful compressed bytes.
- Capacity estimates are gross per-frame carriers, not net archive savings.
  A byte-closed prototype still needs archive grammar, no-op proof,
  inflate-runtime closure, and PoseNet/SegNet component review.

## Next Action

Proceed to a small byte-closed `research_only=true` SABOR prototype only after
a fuller local advisory audit keeps the same qualitative signal:

```bash
.venv/bin/python tools/measure_segnet_argmax_stable_interior.py \
  --n-frames 600 \
  --n-perturbation-samples 2 \
  --perturbation-subset-stride 10 \
  --epsilon-list 1,2,4,8,16,32 \
  --margin-thresholds 0.5,1.0,2.0,4.0,8.0,16.0 \
  --num-threads 4
```

Prototype gate: use the stable-map artifact as a sensitivity mask for a
minimal boundary-only payload transform, prove targeted bytes are consumed by
inflate, keep `research_only=true`, and make no score/promotion claim until a
byte-closed archive/runtime packet exists and is evaluated on the correct axis.
