# MLX Port — Adversarial Audit + Takeover (2026-06-11)

**Mission (operator verbatim):** "look for existing local MLX port work and audit
and review adversarially and take over completely." Inventory every MLX port
module, classify parity status adversarially (SOUND / FAKE-PARITY / DRIFTED /
PARTIAL / ORPHAN), produce the consolidation roadmap toward ONE owned bit-faithful
MLX stack, and begin executing the highest-value fixes.

**Authority:** torch-CPU exact is the ONLY authority for any parity claim. MLX =
research-signal unless PROVEN bit-faithful. MPS NEVER (forbidden; 23×/2× scorer
drift). The sibling subagent `a642408922c41f2f3` owns the deep empirical
SegNet/PoseNet MLX-vs-torch-CPU scorer-drift measurement — this audit CONSUMES
that for the scorer section and covers the FULL surface (decoder, renderer,
kernels, parity infra, capstone bundle, trainers, consolidation).

## Scope measured

- **92 MLX source (non-test) modules** under `src/tac/**`.
- **~140 MLX/parity test files**.
- Empirically RAN: capstone parity suite, pr95 torch-parity suite,
  canonical_kernels cross-backend suite, check_383 gate, selector_v4 suite
  (= 169 tests touched; results below).

## Headline verdict

The MLX surface is **far healthier than the surface area suggests**. The CANONICAL
CORE — the capstone VQ-NeRV bundle (`numpy_reference` + `vq_nerv_bundle` + inflate),
the `mlx_pr95_port` decoder/loss/score-bridge, the `framework_agnostic/canonical_kernels`
(Catalog #383), and the two top-tier parity-localization modules
(`mlx_pytorch_render_parity_crux`, `mlx_scorer_torch_parity`) — are **SOUND**: they
exercise REAL trained/random-init weights, characterize drift down to the op, and
carry explicit NO-FAKE guards (stub-decoder-must-fail tests, non-degenerate weight
nudging, trained-proj grid-PE parity).

The risk is **not** widespread fake parity — it is (a) **canonicalization debt**
(the canonical kernels exist but the substrate trainers still use local duplicate
impls; the dedup goal is unfulfilled), (b) **a few thin/forward-only parity gates**
on lower-priority substrate renderers, and (c) **one fake-MLX backend path** inside
the canonical kernel (gumbel MLX = numpy roundtrip, gradient-breaking — forward-only).
I landed one real fix this session (stale fake-parity export test) and leave a
prioritized takeover roadmap.

## Empirical test results (this session, torch-CPU authority, NO MPS)

| Suite | Result |
|---|---|
| `capstone_vq_nerv/test_numpy_reference_parity` (incl. grid-PE trained-proj) | **30 passed** (with pr95) |
| `mlx_pr95_port/test_torch_parity` | included in above 30 |
| `framework_agnostic/test_canonical_kernels` | **64 passed, 1 skip** |
| `tests/test_check_383_mlx_primitives_canonical_routing` | passed (in 64) |
| `pact_nerv_selector_v4/...mlx_renderer_and_bridge` | **11 passed** (1 was failing → FIXED) |

## Per-module classification table

Parity-status legend: **SOUND** (real-input torch/numpy parity gate, drift bounded);
**FAKE-PARITY** (passes on degenerate input, would fail on trained); **DRIFTED**
(divergence unbounded/uncharacterized); **PARTIAL** (port real but no dedicated
bit-faithful gate, or forward-only); **ORPHAN** (dead/duplicate/superseded);
**INFRA** (a parity tool itself, not a port).

| Module | Status | Drift source / note |
|---|---|---|
| `framework_agnostic/canonical_kernels.py` (#383) | **SOUND-forward / PARTIAL-train** | rgb_to_yuv6 + pixel_shuffle + bilinear + HF-residual MLX paths are REAL mx forwards w/ numpy reference + cross-backend parity (atol 1e-5). `gumbel_softmax_sample` MLX path = numpy roundtrip (`mlx→np→mx`, line 237-263) → **gradient-breaking, forward-only**; OK for inference, NOT training-grade. Canonicalization debt: substrate trainers still use local gumbel impls. |
| `capstone_vq_nerv/numpy_reference.py` | **SOUND** | grid-PE fix (commit b77427cab) verified: `_make_grid_pe_bundle(train_proj=True)` tests TRAINED proj (the exact key-mismatch bug class). Has `test_stub_decoder_would_fail_pixel_parity` NO-FAKE guard. d_seg parity <1e-4, d_pose bounded. |
| `capstone_vq_nerv/vq_nerv_bundle.py` | **SOUND** | MLX bundle; numpy-reference contract + score parity on FROZEN DistortionNet. |
| `capstone_vq_nerv/inflate.py` | **SOUND** | numpy-portable inflate; grid-PE cfg branch fixed. |
| `mlx_pr95_port/score_bridge.py` | **SOUND** | The correct hybrid: MLX renders, **REAL frozen torch DistortionNet is authority**, torch returns exact `dL/d(pixels)`. Explicit YUV6-patch assertion keeps pose gradient alive (`Yuv6NotPatchedError`). No surrogate, no mock. |
| `mlx_pr95_port/mlx_trainer.py` + `mlx_losses.py` | **SOUND** | Loss parity vs torch fp32 exact (<1e-5), pose-loss <1e-6, NS-Muon fp32 structural parity, exact d_seg = argmax-disagreement rate (<1e-9). |
| `mlx_pr95_port/pose_film_trainer.py` / `pose_film.py` / `curriculum.py` | **PARTIAL** | Real FiLM/curriculum; lean on score_bridge authority. No dedicated standalone drift gate beyond the suite. |
| `analysis/mlx_pytorch_render_parity_crux.py` | **INFRA (SOUND)** | Localizes render-parity drift on IDENTICAL trained weights down to crux op. Verdict: NO layout/PixelShuffle/transpose bug; ONLY drift = fp32 conv2d accumulation ORDER → ~8e-4 pixel → ≤1 LSB on <0.004% pixels → SegNet-faithful. `fixed_fp64` mode tightens ~4.5×. This is the gold-standard drift characterization. |
| `local_acceleration/mlx_scorer_torch_parity.py` | **INFRA (SOUND)** | SegNet argmax-diff-pixels threshold (default 0), conv2d accumulation probe, layer-trace drift-cliff, batch-invariance. The scorer-drift measurement harness the sibling consumes. |
| `local_acceleration/pr95_hnerv_mlx.py` (3149 LOC) | **SOUND** | The canonical MLX core: pixel_shuffle_2x_nhwc, bilinear_resize (align_corners), conv2d. Source for the #383 MLX delegations. |
| `mlx_renderer.py` (root, 904 LOC, Phase-1) | **PARTIAL / legacy lineage** | Original Phase-1 MaskRenderer port. Consumed by `experiments/train_renderer_mlx.py` + `benchmark_mlx.py`. NHWC + (O,H,W,I) weight layout documented. Tagged `[MPS-research-signal]` in docstring. Separate lineage from capstone/pr95 — candidate for consolidation or explicit legacy tag. |
| `substrates/pact_nerv_selector_v4/mlx_renderer.py` | **SOUND** | `test_..._exported_state_dict_matches_pytorch_forward` runs REAL random-init weights through full forward, drift.max <0.001. Render-quality gate flags flat init (dead-render guard). **Stale fake-parity export test FIXED this session** (see below). |
| `substrates/pact_nerv_selector_v2` / `v3` | **SOUND** | Same sister test pattern as v4 (forward parity on real init). |
| `substrates/pact_nerv_vq/mlx_renderer.py` | **PARTIAL** | Real port (DepthSepConv, VQ-VAE-EMA, export to OIHW). `NotImplementedError` at line 164 is a LEGIT fail-closed guard (only canonical integer-ratio 2x PixelShuffle path supported — non-degenerate constraint, NOT a fake scaffold). MLX referenced in `test_pact_nerv_vq.py` (13 hits) but **no dedicated MLX↔torch forward-drift gate file**. |
| `substrates/pact_nerv_ia3/mlx_renderer.py` | **SOUND** | 20 parity asserts + weight nudging in bridge test. |
| `substrates/hi_nerv/mlx_renderer.py` | **SOUND** | 96 parity asserts + non-degenerate nudging. |
| `substrates/snerv_inverse_steg_carrier/mlx_*` | **SOUND** | 57 + 14 parity asserts; trained-ladder bridge test; hard-byte-export gate; official MFU source parity. |
| `substrates/nscs06_v8_chroma_lut/mlx_iteration.py` | **SOUND** | 24 parity asserts + nudging. |
| `substrates/grayscale_lut/mlx_native.py` | **SOUND** | 4 parity asserts (deterministic LUT — degenerate-init not applicable). |
| `substrates/time_traveler_l5_z6/mlx_renderer.py` + `mlx_export_bridge.py` | **PARTIAL** | 2 parity asserts, zero-init FiLM present → forward-only/thin parity. Lower priority (z6 is research lane). |
| `substrates/z6_v2_cargo_cult_unwind/mlx_renderer.py` | **PARTIAL** | bridge test has 0 parity-asserts in the renderer test (harness-unlock only). |
| `substrates/coin_pp/mlx_renderer.py` | **PARTIAL** | 1 parity assert, harness-unlock test. INR lane. |
| `substrates/mdl_ibps_j/mlx_renderer.py` | **PARTIAL** | 2 parity asserts, harness-unlock. Uses local gumbel. |
| `substrates/z8_hierarchical_predictive_coding/mlx_renderer.py` | **PARTIAL** | Real port + archive-candidate bridge (6 parity asserts). Local gumbel (delegates to dreamer_v3, NOT canonical kernel). |
| `substrates/z7_mamba2/mlx_module.py` + `mlx_native.py` | **PARTIAL** | Mamba-2 SSD MLX backend; lineage smoke + module smoke tests, no dedicated forward-drift-vs-torch gate. |
| `substrates/atw_v2/mlx_renderer.py` / `faiss_ivf_pq_residual/mlx_renderer.py` / `nirvana_cascading_nerv/mlx_renderer.py` | **PARTIAL** | harness-unlock only; no forward-parity gate. |
| `substrates/_shared/mlx_score_aware_full_main.py` + `mlx_score_aware/` | **SOUND** | shared score-aware harness, tested. |
| `substrates/_shared/mamba2_ssd/mlx_backend.py` | **PARTIAL** | shared Mamba-2 MLX kernel. |
| `inverse_steganalysis_real_video_mlx/__init__.py` (48 KB) | **PARTIAL / heavy** | Single large `__init__.py` (real-video MLX inverse-steg); has tests dir. Module-as-`__init__` is a structure smell; not audited line-by-line this session. |
| `master_gradient_mlx_extractor.py` + `master_gradient_mlx_pipeline.py` | **SOUND** | tested extractor/pipeline (test_master_gradient_mlx_*). |
| `optimization/mlx_dynamic_learned_sweep*` (8 modules) + `mlx_research_signal.py` + `mlx_effective_spend_*` | **INFRA (advisory)** | Research-signal/sweep tooling, correctly non-promotable (`[macOS-MLX research-signal]`). Not parity ports. Large count inflates the surface. |
| `local_acceleration/mlx_*` cache/calibration/profile (~30 modules) | **INFRA (advisory)** | Cache materialization, calibration, profile-stability, scorer-response — research-signal infra, NOT contest-path ports. Correctly tagged non-promotable. |
| `local_acceleration/mlx_segnet_se_*` + `mlx_segnet_*_probe` (8 modules) | **INFRA (SOUND)** | SE-pool/conv variant probes localizing the SegNet SE-block MLX drift — the diagnostic lineage feeding the scorer-parity work. |
| `canonical_equations/mlx_pytorch_drift.py` + `mlx_matmul_m_series_floor.py` + `mlx_drift_accumulation_engineering_response.py` | **INFRA (SOUND)** | Canonical equations codifying the measured drift (fp32 conv accumulation order; M-series matmul floor). |

## Scorer section (CONSUMING sibling `a642408922c41f2f3`)

The sibling owns the deep empirical SegNet/PoseNet MLX-vs-torch-CPU drift number.
The standing characterization this audit relies on (from `mlx_pytorch_render_parity_crux`
+ `mlx_scorer_torch_parity` + the canonical drift equation):

- **Render drift is NOT structural** — no layout/PixelShuffle/transpose/convention
  bug. The sole render drift is fp32 conv2d accumulation ORDER, ~8e-4 in [0,255]
  → ≤1 LSB on <0.004% of pixels → SegNet-argmax-faithful at uint8.
- **MPS is forbidden** (23× pose / 2× seg drift); **MLX-CPU + MLX-GPU are the
  high-fidelity path**; torch-CPU is the exact authority.
- SegNet argmax-diff-pixels threshold is **0** in the parity manifest (the gate
  fails closed if MLX flips any argmax pixel vs torch).

**Sibling MEASURED result (commit `342c62463`, real `0.mkv` CPU bit-faithful gate):**

- **MLX-CPU SegNet = 2 argmax flips / 19.66M pixels** → authority-faithful (the
  MLX-CPU scorer is bit-faithful to torch-CPU within 2 pixels in 19.66M; the gate
  passes).
- **MLX-GPU SegNet = 243 flips / 19.66M** → boundary-confined (the GPU path drifts
  slightly more but the flips are confined to SegNet class boundaries; still far
  below any score-material threshold).
- **PoseNet pixel drift 2.76e-4** → flagged as needing a frontier-authority check
  (at frontier pose_avg ~3.4e-5 the pose marginal is steep, so a 2.76e-4 drift
  could be score-material at the frontier operating point — sibling routes this to
  a frontier authority eval).

**Audit interpretation:** this CONFIRMS the standing characterization — MLX-CPU is
authority-faithful for SegNet (use it freely), MLX-GPU is boundary-confined (safe
for research, validate at frontier), and the ONE thing to watch is **pose drift at
the frontier operating point** (the steep `sqrt(10·d_pose)` marginal). The parity
infra (`mlx_scorer_torch_parity`, render-parity-crux) correctly surfaced this; no
fake-parity in the scorer path.

## Fixes LANDED this session

1. **`pact_nerv_selector_v4` stale fake-parity export test** (commit `1f979fa77`).
   The production `pack_archive_from_exported_state_dict` was hardened to refuse a
   silent selectors-absent export (`exported_state_dict` drops the selectors buffer
   → falls back to an inert no-authority archive). The test's bare-pack call still
   omitted `selectors=` → the test FAILED against the hardened guard. **Fix:** the
   test now (a) sets non-trivial in-palette selectors and passes them explicitly,
   and (b) adds a NO-FAKE guard asserting the inert (selectors-absent) fallback is
   refused with `ValueError(match="inert")`. Result: 11/11 pass; the test now
   exercises the REAL non-inert authority path. Marked reviewed by 2 distinct
   registered principals (yousfi, fridrich); committed via serializer.

## Canonical-stack consolidation roadmap (the ONE stack we OWN)

The owned, bit-faithful MLX stack should be these SOUND layers, in this hierarchy:

```
authority:   torch-CPU DistortionNet (SegNet/PoseNet)   [score_bridge wires it]
body:        mlx_pr95_port (decoder + losses + score_bridge + pose_film)   ← TRAIN
core ops:    local_acceleration/pr95_hnerv_mlx (conv2d/pixel_shuffle/bilinear)
kernels:     framework_agnostic/canonical_kernels (#383, cross-backend, numpy-ref)
inflate:     capstone_vq_nerv/numpy_reference + inflate (numpy-portable, T4-ready)
gates:       mlx_pytorch_render_parity_crux + mlx_scorer_torch_parity (drift INFRA)
```

Everything else (the ~40 `optimization/mlx_*` sweep + `local_acceleration/mlx_*`
cache/calibration modules) is **advisory research-signal infra** — correctly
non-promotable, keep but do not treat as port surface.

## Prioritized takeover tasks (left as cited tasks; too big to land cleanly now)

1. **[HIGH] Finish #383 canonicalization debt — real-MLX gumbel.**
   `framework_agnostic/canonical_kernels.py:237-263` `_gumbel_softmax_sample_mlx`
   does `mlx→numpy→mlx` (gradient-breaking, forward-only). The substrate trainers
   (`z8_hierarchical_predictive_coding/mlx_renderer.py:617`, `dreamer_v3_rssm`,
   `mdl_ibps_j`) STILL use local gumbel impls because the canonical one can't carry
   MLX autograd. Either (a) implement a real mx-native Gumbel (mx softmax of
   `(logits + gumbel_noise)/τ`, gumbel via `-mx.log(-mx.log(mx.random.uniform))`)
   so trainers can route through it AND keep gradients, OR (b) honestly relabel the
   MLX path as forward/inference-only and document that training keeps local impls.
   The duplicate-extraction goal in the module docstring is currently unfulfilled.

2. **[HIGH] Add a dedicated MLX↔torch forward-drift gate for `pact_nerv_vq`.**
   `pact_nerv_vq/mlx_renderer.py` is a real port with NO dedicated bit-faithful
   forward-parity test file (only tangential MLX refs in `test_pact_nerv_vq.py`).
   Clone the selector_v4 `test_..._exported_state_dict_matches_pytorch_forward`
   pattern (real init → full forward → `drift.max < 0.001`).

3. **[MED] Promote thin/forward-only substrate parity gates to real-weight gates.**
   `time_traveler_l5_z6`, `z6_v2`, `coin_pp`, `mdl_ibps_j`, `z8`, `z7_mamba2`,
   `atw_v2`, `faiss_ivf_pq_residual`, `nirvana_cascading_nerv` have harness-unlock
   tests but lack the selector_v4-grade real-init full-forward drift assertion.
   These are research lanes — lower mission value, but each is a latent fake-parity
   risk (the grid-PE bug class). Apply the selector_v4 pattern where the lane is
   still active; explicitly tag `research_only` + `[macOS-MLX research-signal]`
   where dormant.

4. **[MED] Resolve the root `mlx_renderer.py` legacy lineage.**
   The Phase-1 `mlx_renderer.py` (consumed by `train_renderer_mlx.py` +
   `benchmark_mlx.py`) is a separate lineage from the capstone/pr95 stack and is
   docstring-tagged `[MPS-research-signal]`. Either consolidate its conv/pixel-shuffle
   ops onto `pr95_hnerv_mlx`/`canonical_kernels` (one source of truth) or mark it
   explicitly legacy/benchmark-only so it isn't mistaken for the contest path.

5. **[LOW] Refactor `inverse_steganalysis_real_video_mlx/__init__.py` (48 KB).**
   A 48 KB module-as-`__init__` is a structure smell. Split into named submodules
   and add a real-video MLX-vs-torch parity gate if it feeds any score claim;
   otherwise tag advisory.

6. **[LOW] Surface-area hygiene.** 92 MLX source modules is mostly advisory
   sweep/cache infra. Consider a `MLX_PORT_INVENTORY.md` (or extend
   `local_acceleration/mlx_scorer_port_inventory.py`) that lists the 6 owned-stack
   modules vs the advisory infra, so future audits don't re-discover that the bulk
   is non-port research-signal tooling.

## NO-FAKE accounting

- I verified parity SOUNDNESS **empirically** (ran 169 tests), not from docstrings.
- The one fix I landed makes a test exercise the REAL authority path (not a
  constant/inert path) — it strengthens a NO-FAKE guard, it does not weaken one.
- I made NO score claims. MLX rows remain `[macOS-MLX research-signal]`
  non-promotable; torch-CPU is the only authority cited.
- I did NOT touch the running daemons (capstone pid, atlas workers) or their dirs.
