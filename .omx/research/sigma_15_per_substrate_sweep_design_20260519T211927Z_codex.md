# Sigma=15 Per-Substrate Sweep Design

Timestamp: 2026-05-19T21:19:27Z  
Owner: codex  
Task: `codex_routing_directive_session_20260519_max_score_lowering_batch_BCEF_20260519T051028Z::CLUSTER_F1`  
Lane: `lane_sigma15_per_substrate_sweep_design_codex_20260519`  
Evidence grade: design + premise verification; no score claim

## Verdict

Proceed with a corrected two-family sweep plan, not the literal "5 consumers x same grid" framing.

1. Grayscale-LUT bandwidth sweep:
   `sigma in {0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0}`.
2. SCPP block-FP cutoff sweep:
   separate integer-safe sweep only; SCPP `sigma` is not the grayscale-LUT bandwidth.

Predicted aggregate mission value remains small bolt-on grade: delta-S `[-0.002, -0.0003]`. It should not displace higher-EV frontier work, but the existing Lane FR-MM script had stale absolute-score predictions and a stale grid, so I hardened that operator surface now.

## Premise Corrections

The directive and source memo correctly reject the NSCS06 sigma resurrection premise. One additional correction is required:

| Surface | Status | Reason |
|---|---:|---|
| `src/tac/mask_grayscale_lut.py` | IN grayscale-LUT sweep | `create_gaussian_softmax_lut(sigma=...)` and `grayscale_to_probability_map(..., sigma=...)` expose the exact bandwidth knob. |
| `experiments/build_lane_mm_archive.py` + `submissions/robust_current/inflate_renderer_grayscale.py` | IN grayscale-LUT sweep | Archive build records sigma; decode-time `LANE_MM_SIGMA` changes the LUT without rebuilding bytes. |
| `src/tac/optimize_grayscale_canvas.py` + `experiments/optimize_grayscale_canvas.py` | IN grayscale-LUT sweep | `OptimizeConfig.sigma` and CLI `--sigma` expose train/optimize-time bandwidth. |
| `src/tac/segmap_renderer.py` LCT path | NEEDS SMALL PARAM-INJECTION PATCH | `train_epoch` hardcodes `sigma = 15.0` when 4-D grayscale masks are projected through learnable class targets. |
| `experiments/train_segmap.py` and `experiments/train_segmap_film_canvas.py` | NEEDS SMALL PARAM-INJECTION PATCH | Fixed-soft training calls `grayscale_to_probability_map(..., sigma=15.0)`. |
| `src/tac/contrib/szabolcs_renderer.py` | PARTIAL | `create_gaussian_softmax_lut(..., sigma=...)` supports the grid; `encode_luma_to_probability_map(lut=None)` and builder paths default to 15 unless passed a prebuilt LUT or patched. |
| `src/tac/scpp_substrate.py` | EXCLUDE from grayscale-LUT sweep | `SCPPSubstrateConfig.sigma` is a block-FP weight-scale cutoff in `scale = sigma * 2**exponent`; deserialization casts to `int`, so `0.5` is not archive-safe. |

The class-target order must stay channel-ordered as `[0, 255, 64, 192, 128]`. Treating it as sorted gray values changes the semantic channel binding.

## Runnable Sweep Surfaces

### A. Lane MM Decode-Time Sweep

Use the existing FR-MM remote script after this patch:

```bash
bash scripts/remote_lane_fr_mm_sigma_sweep.sh
```

The script now uses the corrected grid `{0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0}`, records predicted delta-S rather than stale absolute-score bands, and keeps results axis-tagged through `contest_auth_eval.py --device cuda`.

This is the cheapest exact CUDA path because it builds one shared archive and varies only `CONFIG_ENV_PATH` / `LANE_MM_SIGMA`.

### B. Lane AL Train/Optimize-Time Sweep

Run one optimize job per sigma:

```bash
.venv/bin/python experiments/optimize_grayscale_canvas.py \
  --anchor-archive <archive.zip> \
  --gt-video upstream/videos/0.mkv \
  --upstream-dir upstream \
  --output-dir experiments/results/lane_al_sigma_<sigma>_<utc> \
  --sigma <sigma> \
  --device cuda
```

This is not equivalent to Lane MM post-hoc decode-time sweep. The optimized grayscale tensor changes per sigma, so compare per-sigma archive bytes, output aggregate hash, and component deltas.

### C. SegMap Training Sweep

Do not dispatch yet. First add a small parameter injection so:

- `SegMapTrainer` receives `lut_sigma: float = LUT_DEFAULT_SIGMA`.
- LCT grayscale projection uses that field instead of hardcoded `15.0`.
- `experiments/train_segmap.py` and `experiments/train_segmap_film_canvas.py` expose matching CLI/config fields.

Then sweep the same grid with identical seed, training schedule, `eval_roundtrip=True`, and baseline checkpoint.

### D. Szabolcs Reference Sweep

Utility-level sweep is already possible:

```python
lut = create_gaussian_softmax_lut(sigma=<sigma>)
prob = encode_luma_to_probability_map(luma, lut=lut)
```

Production builder/inflate parity still needs either a `sigma` config field or an explicit prebuilt-LUT path. Do not treat utility support as full substrate support.

### E. SCPP Block-FP Sweep

Route separately:

```python
SCPPSubstrateConfig(sigma=<positive int>, qint_max=7, block_size=16)
```

Use integer grid only, e.g. `{1, 2, 5, 10, 15, 20, 30}`. Measure block-FP payload bytes, reconstruction error, exact archive bytes, and scorer components. This is a quantizer-scale sweep, not a class-probability LUT sweep.

## Measurement Contract

For every candidate row:

- axis tag: `[contest-CUDA]`, `[contest-CPU]`, or advisory, never implicit;
- archive path, archive bytes, archive SHA-256;
- runtime tree SHA or runtime content manifest;
- git commit + dirty-status summary;
- inflate device and eval device;
- output aggregate hash if auth-eval keeps inflated outputs;
- component metrics: final score, score recomputed from components, `seg_dist`, `pose_dist`, rate term;
- LUT diagnostics: target-row top-1 probability, midpoint entropy, argmax-flip rate after decode if masks/luma are present;
- multiple-comparison rule: promote only by paired baseline on the same axis, not best-of-grid optimism.

## Per-Substrate Symposium Drafts

| Substrate | Draft verdict | Required next action |
|---|---|---|
| Lane MM / FR-MM | PROCEED_AFTER_HIGHER_EV | Existing exact-CUDA script is now aligned with the corrected grid and custody language. |
| Lane AL | PROCEED_TO_TIMING_SMOKE | Run one low-step timing smoke before full grid because each sigma changes the optimized grayscale artifact. |
| SegMap fixed-soft + LCT | REQUIRES_PATCH | Add explicit `lut_sigma` plumbing before any sweep. |
| Szabolcs reference | REQUIRES_PATCH_OR_PREBUILT_LUT_DISCIPLINE | Utility supports sigma; builder/inflate parity does not yet. |
| SCPP | SPLIT_TO_BLOCKFP_SWEEP | Exclude from grayscale-LUT symposium; create separate integer quantizer-cutoff design if pursued. |

## 6-Hook Wire-In

1. Sensitivity map: record sigma rows as LUT-bandwidth or block-FP-cutoff, not a shared scalar.
2. Pareto constraint: compare paired score delta against archive byte/rate delta; Lane MM decode-time rows should have constant archive bytes.
3. Bit allocator: SCPP sweep affects weight payload bit allocation; LUT sweep affects scorer distortion at fixed payload in FR-MM.
4. Cathedral autopilot: canonical consumer should treat this as a small bolt-on behind higher-EV exact-eval lanes unless a sweep row breaks the predicted band.
5. Continual learning: append probe outcome only after empirical rows exist; this memo is design evidence, not posterior score evidence.
6. Probe disambiguator: the disambiguator is the corrected split between LUT bandwidth, block-FP cutoff, and boundary-curvature MAD sigma.

## Verification

- Spawned two xhigh read-only sidecars; both converged on the SCPP sigma split and F.2 completion status.
- Hardened `scripts/remote_lane_fr_mm_sigma_sweep.sh` to the corrected grid and removed stale absolute-score authority.
- No scorer, provider, or paid dispatch was invoked in this design pass.
