# DreamerV3 RSSM categorical posterior — MLX-local converged-training advisory verdict

- **Date:** 2026-05-27T18:35Z
- **Lane:** `lane_dreamer_v3_rssm_mlx_scaffold_20260526`
- **Substrate:** `dreamer_v3_rssm` (categorical posterior G=24 / K=256; HNeRV decoder; ~50K-param class)
- **Cost:** $0 (MLX-local M5 Max; NO paid CUDA/CPU dispatch)
- **Axis tag:** `[macOS-MLX research-signal]` training + `[macOS-CPU advisory]` distortion measurement
- **Promotable:** `false` (NON-PROMOTABLE by construction per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192 / #127 / #323)
- **dispatch_enabled:** `false` preserved (Catalog #325 — no dispatch fired)
- **Canonical equation ref:** `categorical_posterior_capacity_vs_continuous_gaussian_v1`
  (registered `.omx/state/canonical_equations_registry.jsonl` per Catalog #344;
  this verdict is an empirical anchor against it — see §Equation impact)

## TL;DR verdict

**NOT class-shift-competitive. This candidate is decisively NOT the winner — a
real, honest negative.** The class-shift thesis (categorical posterior over a
DreamerV3-style RSSM state CAN, in principle, beat continuous-Gaussian IB) still
stands; this *specific* implementation (HNeRV decoder fed a categorical→continuous
projection over a 6144-dim one-hot) collapses on BOTH axes exactly like the C6
IBPS v1 SegNet-collapse precedent (105.15 contest-CUDA). NO FIRE candidate is
named. `dispatch_enabled: false` stays.

## 1. Convergence (predecessor crash resumed → completed)

Predecessor `dreamer_v3_rssm_mlx_long_train` launched a detached 2000-epoch run
that the bash harness killed at epoch ~289 (CLAUDE.md "bash harness kills
long-running tasks"). The promised driver/advisory scripts were never committed
(they lived only in the killed process). This subagent ran a **fresh 500-epoch
`_full_main` SYNCHRONOUSLY in-turn** (NOT detached) through the canonical
`tac.substrates._shared.mlx_score_aware.run_mlx_score_aware_full_main` harness
→ `tac.training.long_training_canonical.run_long_training`.

- Run dir: `experiments/results/dreamer_v3_rssm_mlx_long_training_run/converged_20260527T182521Z/`
- 500 epochs in **49.8s** wall-clock (batch = min(600, 8) = 8 pair-indices/step), real `upstream/videos/0.mkv` targets via `decode_mlx_targets`.
- Reconstruction MSE proxy (normalized RGB): 0.367 (ep1) → 0.038 (ep5) → 0.006 (ep100) → **plateau ~0.0055 flat through ep500** (mean loss epochs≥450 = 0.0055; min 0.0034, max 0.0083).
- Confirms the predecessor's observation: the loss is genuinely flat by epoch ~100; 2000 epochs were unnecessary. The model is at its reconstruction-MSE minimum.

## 2. Numpy-portable archive + decode verification (8th MLX-first directive)

Built the RSSMC1 archive from the final converged checkpoint
(`final_epoch000499_*.live.state.npsd`, loaded framework-free via
`tac.substrates._shared.numpy_portable_inflate.unpack_state_dict_numpy`), derived
per-pair argmax category indices from trained logits, packed via
`tac.substrates.dreamer_v3_rssm.pack_archive`.

- **archive.zip = 532,072 bytes** (member 531,964 B; ZIP_STORED single member `0.bin`; sha256 `af5d78b9…`)
- Section breakdown: `decoder_blob` **517,352 B** + `indices_blob` 14,400 B + header 27 B + meta 185 B.
- **The 517 KB decoder_blob dominates** — the `cat_to_continuous` Linear projects a `G*K = 6144`-dim one-hot to `decoder_latent_dim=28` → ~172K params (fp16 = ~344 KB), plus stem/conv blocks. The categorical→continuous projection over a large one-hot is the rate killer.
- **Torch inflate** (`inflate.py`, contest-canonical PyTorch runtime): wrote 1200 frames = **3,662,409,600 bytes = exactly the contest 1164×874×1200×3 contract** (frame-count MATCH per Catalog #367).
- **Numpy-portable decode VERIFIED framework-free:** pure-numpy decode of the categorical dequant + cat_to_continuous + 6 PixelShuffle upsample blocks + refine + dual RGB heads (incl. pure-numpy bilinear x2, align_corners=False) matched the torch decoder on 2 pairs @384×512 to **max_abs = 0.0001, mean_abs = 8.7e-6** (FRAMEWORK-FREE-EQUIVALENT). The 8th MLX-first directive (inflate numpy-portable, no torch/MLX dep) is satisfied for this substrate's decode path.

Manifest: `experiments/results/dreamer_v3_rssm_mlx_long_training_run/converged_20260527T182521Z/archive_advisory_manifest.json`.

## 3. Advisory contest score ([macOS-CPU advisory], 60-pair sample, NON-PROMOTABLE)

Decoded frames vs `upstream/videos/0.mkv` ground truth through the canonical
contest `DistortionNet.compute_distortion` (PoseNet + SegNet weights from
`upstream/modules.py` `posenet_sd_path` / `segnet_sd_path`), mirroring
`upstream/evaluate.py`'s exact reduction `score = 100*seg + √(10*pose) + 25*rate`.
60-pair sample on macOS-CPU (full 600-pair eval would take 60-120 min and cannot
change the verdict — see §4):

| Component | Value | Term | Frontier (≈0.192 total) |
|---|---:|---:|---|
| SegNet distortion (avg) | 0.5234 | **52.34** | ~0.0007 → ~0.07 |
| PoseNet distortion (avg) | 184.81 | **42.99** | ~3.4e-5 → ~0.018 |
| rate = 532072 / 37545489 | 0.01417 | **0.354** | ~0.119 (178546 B) |
| **Advisory score** | | **≈ 95.7** | **0.192 [contest-CPU]** |

Result: `experiments/results/dreamer_v3_rssm_mlx_long_training_run/converged_20260527T182521Z/advisory_score_result.json`.

## 4. Advisory-only frontier gap + FIRE-or-NOT verdict

- Canonical frontier (`.omx/state/canonical_frontier_pointer.json`): **0.19202 [contest-CPU]** (lane_v14 cascade_a fec10; 178,546 B).
- **This candidate advisory ≈ 95.7 — roughly 498× WORSE than the frontier.**
- **The rate term ALONE (0.354) already exceeds the entire frontier score (0.192) by ~1.85×** before a single distortion bit is counted. No distortion improvement can rescue a candidate whose rate term alone nearly doubles the frontier.
- The distortion is catastrophic on BOTH axes (seg 0.52, pose 185) — near-random reconstructions. This is the **same SegNet-collapse failure mode as C6 IBPS v1** (105.15 contest-CUDA): the categorical bottleneck + HNeRV decoder produce frames that bear almost no semantic resemblance to the contest video.

**FIRE verdict: NO.** No operator-gated paired-CUDA+CPU (Catalog #246) candidate
is named. Firing paid GPU on this archive would re-measure a known catastrophic
negative. The candidate stays `dispatch_enabled: false` / `research_only: true`.

**Why it's not a paradigm KILL** (per CLAUDE.md "Forbidden premature KILL" +
Catalog #307 paradigm-vs-implementation): this is an **IMPLEMENTATION-LEVEL**
falsification of the specific (HNeRV-decoder + categorical→continuous-over-6144-
one-hot + MSE-proxy-loss) implementation, NOT a paradigm-level refutation of the
categorical-posterior class-shift thesis. The two structural defects are concrete
and addressable:

1. **MSE-proxy loss does not bind the scorer.** Training minimized reconstruction
   MSE on normalized RGB (converged to 0.0055) but NEVER routed gradients through
   SegNet/PoseNet (the `distillation_weight` Hinton-KL surrogate is a weak proxy,
   not the canonical `score_pair_components` per Catalog #164). A converged
   reconstruction-MSE minimum is uncorrelated with contest distortion here — the
   decoded frames look nothing like the video to the scorer.
2. **The rate is structurally too large.** `cat_to_continuous` over a 6144-dim
   one-hot is ~172K params (344 KB fp16) — the decoder_blob dominates the archive
   at 517 KB. Any viable categorical-posterior substrate must either (a) shrink
   the cat→continuous projection (per-group embedding lookup tables instead of a
   dense Linear over the concatenated one-hot), or (b) entropy-code the decoder
   weights far more aggressively. As built, the rate term alone is disqualifying.

## Reactivation criteria (DEFERRED-pending-research, per Catalog #313 probe-outcome)

The class-shift thesis reactivates IF a future iteration lands ALL of:
- (a) **score-aware loss** via canonical `tac.substrates._shared.score_aware_common.score_pair_components` (Catalog #164) — gradients through SegNet/PoseNet on the real video, NOT MSE-proxy;
- (b) **rate reduction** of the cat→continuous projection (per-group embedding tables, target decoder_blob < 100 KB);
- (c) MLX-local advisory re-measure showing seg+pose distortion within ~10× of frontier (current ~500×) before any paid dispatch is even considered.

Per Catalog #325 a fresh per-substrate adversarial grand council symposium is
required before the recipe may flip `dispatch_enabled: true`.

## Canonical-vs-unique decision per layer (Catalog #290)

- ADOPT_CANONICAL: training loop / EMA / checkpoint / telemetry / Provenance /
  posterior anchor (`run_long_training`); numpy-portable state_dict + inflate
  primitives; contest distortion reduction (`evaluate.py` formula); advisory
  custody (`[macOS-CPU advisory]` non-promotable markers).
- FORK (substrate UNIQUE): DreamerV3 RSSM categorical posterior + Gumbel-Softmax
  STE + HNeRV decoder (`tac.substrates.dreamer_v3_rssm.module`).

## Observability surface (Catalog #305)

- Inspectable per layer: per-epoch telemetry JSONL (loss + ema_drift_l2) at
  `.../converged_20260527T182521Z/telemetry.jsonl` (500 rows).
- Decomposable per signal: seg / pose / rate terms broken out in
  `advisory_score_result.json`.
- Diff-able across runs: checkpoint `.npsd` blobs every 10 epochs + final.
- Queryable post-hoc: `archive_advisory_manifest.json` + `training_artifact.json`.
- Cite-able: archive sha256 `af5d78b9…` + lane_id + checkpoint epoch + seed=0.
- Counterfactual-able: per-pair category indices in `indices_blob` (Catalog #272
  distinguishing feature — per-pair index mutation IS the byte-mutation surface).

## Equation impact (Catalog #344)

This is an empirical anchor against `categorical_posterior_capacity_vs_continuous_gaussian_v1`:
the equation's capacity-headroom claim (H(T)=192 bits/sample, ~4× the C6
continuous-Gaussian ~50 bits) is NECESSARY but NOT SUFFICIENT — capacity headroom
does not deliver score when (a) the loss does not bind the scorer and (b) the
projection cost dominates the rate. The recalibration trigger is the reactivation
criteria above; until then the equation's `predicted_vs_empirical_residual` should
record this implementation-level miss (advisory 95.7 vs the substrate's
landing-posterior `predicted_score=0.195` — a ~490× implementation-level miss,
NOT a paradigm refutation). `# FORMALIZATION_PENDING:empirical_anchor_recorded_against_existing_canonical_equation_categorical_posterior_capacity_vs_continuous_gaussian_v1_implementation_level_miss_per_catalog_307_recalibration_deferred_to_reactivation_criteria`

## 6-hook wire-in (Catalog #125)

- hook #1 sensitivity-map: N/A (advisory negative; no per-axis weight contribution).
- hook #2 Pareto constraint: ACTIVE — this anchor pins the categorical-posterior
  point far outside the feasible region (rate 0.354 alone > frontier 0.192).
- hook #3 bit-allocator: N/A (no promotion).
- hook #4 cathedral autopilot dispatch: ACTIVE — the substrate's
  `emit_landing_posterior_anchor` already routes the (refused, advisory-grade)
  signal into the cathedral consumers; this verdict supersedes the L0 scaffold's
  optimistic `predicted_score=0.195` with the empirical advisory 95.7.
- hook #5 continual-learning posterior: ACTIVE — advisory result is non-promotable
  research-signal; recorded in this memo + manifest JSON.
- hook #6 probe-disambiguator: ACTIVE — the G-vs-K probe (`module` docstring §)
  is moot for THIS implementation; the dominant ΔS driver is neither G nor K but
  the loss-binding + projection-cost defects above.

## Discipline

- CLAUDE.md 8th MLX-first directive (training MLX-local + inflate numpy-portable VERIFIED) ✓
- Catalog #1 / #114 real-video targets (`upstream/videos/0.mkv`, NOT synthetic) ✓
- Catalog #192 / #127 / #323 macOS-MLX/CPU non-promotable markers ✓
- Catalog #325 `dispatch_enabled: false` preserved (no dispatch fired) ✓
- Catalog #206 checkpoint discipline (4 checkpoints) honored ✓
- Catalog #307 paradigm-vs-implementation: IMPLEMENTATION-LEVEL falsification ✓
- Catalog #313 DEFERRED-pending-research (NOT KILL) with reactivation criteria ✓
- Catalog #287 phantom-API: all module citations real + this FORMALIZATION_PENDING tag ✓
- $0 cost; ~6 min total wall-clock (train 50s + archive/verify + advisory eval) ✓
