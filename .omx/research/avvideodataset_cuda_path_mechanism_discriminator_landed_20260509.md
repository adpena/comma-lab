# AVVideoDataset CUDA-CPU drift mechanism discriminator — design + landing ledger

**Date**: 2026-05-09
**Lane**: `lane_avvideodataset_cuda_path_mechanism_discriminator` (L1, impl_complete)
**Catalog atom**: domain catalog #4 EIG/$ ranking (`feedback_domain_exploitation_catalog_landed_20260509`)
**Operator approval**: 2026-05-09 ("proceed with all" + ~$1.20 dual-eval budget)
**Commit**: `0c2faf0a`
**Evidence grade**: `[predicted; cuda-cpu-drift-discriminator on A1 substrate; pre-dispatch]`

## TL;DR

The CUDA-CPU drift on HNeRV-cluster archives is empirically calibrated at
**R_pose = 5.04 ± 0.10** and **R_seg = 1.17 ± 0.01** across PR100/101/102/103/105
(commit 697bfe01). Three competing hypotheses explain the mechanism:

1. **Loader-byte drift** — DALI (CUDA decoder, GT side) vs PyAV (CPU decoder)
2. **Conv-kernel accumulation drift** — PoseNet/SegNet FP32 conv ops accumulate
   differently CUDA-vs-CPU
3. **Hydra/head numerical sensitivity** — PoseNet's 12-dim Hydra head has high
   condition number; tiny upstream perturbations are amplified

This lane builds a **3-variant isolation discriminator** plus a control on
A1's substrate (the 0.19284-CPU anchor, 178,262 B archive, sha
`87ec7ca5...492b5`). Archive bytes are bit-identical across variants — only
`inflate.py` differs. Resolves the mechanism question with **one paired CPU+CUDA
sweep at ~$0.40-$1.20 total cost**; cluster-wide payoff (every future archive
benefits from the corrected drift predictor).

## The 4 variants

| Variant | Mechanism hypothesis | Mutator | Inflate.py change |
|---|---|---|---|
| `v_baseline` | control | `_mutate_baseline` | unchanged (sanity) |
| `v_loader_isolated` | loader_byte_drift | `_mutate_loader_isolated` | `device = torch.device("cpu")` regardless of CUDA availability |
| `v_conv_isolated` | conv_kernel_accumulation_drift | `_mutate_conv_isolated` | `torch.use_deterministic_algorithms(True)` + cudnn determinism |
| `v_hydra_isolated` | hydra_head_numerical_sensitivity | `_mutate_hydra_isolated` | replace `.round()` with `.div(2.0).round().mul(2.0).clamp(0, 255)` |

### Why each isolation works

- **`v_loader_isolated`**: forces inflate-side decoder onto CPU regardless of
  available device. Inflated uint8 frames become bit-identical between CUDA-eval
  and CPU-eval. If the canonical CUDA-CPU score gap (~0.033) **shrinks**, the
  residual gap is dominated by GT-loader drift (DALI vs PyAV) + scorer-forward
  CUDA-CPU drift. If the gap **stays at ~0.033**, inflate-side conv-kernel
  asymmetry is NOT the dominant mechanism. *Note: cannot directly probe
  `DefaultDatasetClass` selection in upstream/evaluate.py — that's pinned. This
  variant probes the inflate-side conv asymmetry component, which is half of the
  loader-byte-drift hypothesis.*
- **`v_conv_isolated`**: forces deterministic conv kernels and disables cuDNN
  benchmark in the inflate-time decoder. Doesn't reach upstream PoseNet/SegNet
  conv kernels (those are pinned), but tightens the inflate-time decoder's CUDA
  numerics. Discriminates between inflate-side and downstream-side conv noise.
- **`v_hydra_isolated`**: pre-quantizes inflate output to nearest multiple of 2
  (instead of nearest integer). Washes out tiny upstream perturbations of
  magnitude < 1 LSB before they reach the high-condition Hydra head in PoseNet.
  If R_pose drops from 5.04 to <2.0 here, head-amplification of small inputs is
  the dominant pose-drift mechanism. (Score will likely be WORSE than baseline
  on absolute terms — the discriminator signal is the CUDA/CPU **ratio**, not
  the absolute score.)

## Discriminator decision rules

| Outcome | Verdict label | Action |
|---|---|---|
| 1 isolation drops R_pose < 2.0, others don't | `PRIMARY_MECHANISM_IDENTIFIED` | Set the corresponding registry field (`loader_drift_correction` / `conv_kernel_determinism_required` / `head_quantize_post_inference_dtype`) on the `hnerv_ft_microcodec` profile |
| 2+ isolations drop R_pose < 2.0 | `MULTI_MECHANISM_PRIMARY` | Mechanisms multiplicative; set both fields |
| 1 primary + 1+ contributing (≥30% drop) | `MULTI_MECHANISM_PRIMARY_PLUS_CONTRIBUTING` | Set primary + record contributing |
| Contributing only (no primary) | `MULTI_MECHANISM_CONTRIBUTING_ONLY` | Set per contributor |
| **None** of the 3 narrow drift | `FOURTH_MECHANISM_HYPOTHESIS` | Per CLAUDE.md `forbidden_premature_kill_without_research_exhaustion`: surface as operator decision; do NOT kill the discriminator family. Stamp verdict label on profile but do NOT toggle per-mechanism flags |
| `v_baseline` missing | `INCONCLUSIVE_NO_BASELINE` | Re-dispatch baseline before computing verdict |
| Any isolation variant missing | `INCONCLUSIVE_VARIANTS_MISSING` | Re-dispatch missing variant |

## Local M5 Max smoke verification (pre-dispatch)

Ran each variant's inflate.py against `archive.zip` on local M5 Max CPU:

| Variant | Inflated frames md5 |
|---|---|
| baseline | `d11e83b3acfa39f0232aa9e60497b206` |
| loader | `d11e83b3acfa39f0232aa9e60497b206` (= baseline) |
| conv | `d11e83b3acfa39f0232aa9e60497b206` (= baseline) |
| hydra | `6e93df53a883384c76219603f97be21b` (DISTINCT) |

This confirms: (a) baseline/loader/conv produce identical bytes on local CPU
because no CUDA path is active; (b) hydra-variant correctly applies the
coarse-quantize mutation. The discriminator signal will only emerge under
paired CPU+CUDA exact eval.

## Deliverables landed

1. `tools/build_a1_cuda_cpu_drift_discriminator_variants.py` — 4-variant builder.
   - 4 mutators registered in `_MUTATORS` dict.
   - `write_variant()` enforces: archive SHA byte-identical to A1; baseline
     mutator MUST produce SHA-identical inflate.py; isolation mutators MUST
     produce SHA-distinct inflate.py.
   - Per-variant `discriminator_manifest.json` carries `dispatch_blockers` =
     `["dispatch BOTH CPU (GHA) and CUDA (T4/4090/A100) per dual-eval mandate"]`
     plus the standard claim-lane / preflight blockers.
   - Per `forbidden_premature_kill_without_research_exhaustion`: rationale for
     `FOURTH_MECHANISM_HYPOTHESIS` outcome explicitly states "do NOT kill" the
     discriminator family.
2. `tools/analyze_a1_cuda_cpu_drift_discriminator_verdict.py` — verdict computer.
   - Refuses macOS-CPU / MPS / advisory tags. Refused-tag check fires BEFORE
     accepted-substring check (so `[contest-CPU advisory]` is correctly refused
     even though "contest-CPU" is a substring of an accepted tag).
   - Emits typed verdict + registry-update spec.
3. `src/tac/optimization/cuda_cpu_axis_profile_registry.py` extended with 5
   optional `ArchitectureProfile` fields + `apply_discriminator_verdict_to_registry()`.
   Serialisation is compact: unset fields stay unserialised.
4. `scripts/remote_lane_avvideodataset_cuda_path_mechanism_discriminator.sh` —
   parallel-dispatch actuator. Default behavior: CPU GHA enabled (free), CUDA
   opt-in (avoids burning GPU budget without explicit operator approval).
5. 87 tests pass (45 NEW for this lane + 42 backward-compat).

## Built artifacts (reproducible from tool; not committed since under `experiments/results/`)

- `experiments/results/a1_cuda_cpu_drift_discriminator_{v_baseline,v_loader_isolated,v_conv_isolated,v_hydra_isolated}_20260509T110211Z/`
- `experiments/results/a1_cuda_cpu_drift_discriminator_rollup_20260509T110211Z.json`

To regenerate:
```bash
.venv/bin/python tools/build_a1_cuda_cpu_drift_discriminator_variants.py \
  --timestamp 20260509T110211Z \
  --rollup-output experiments/results/a1_cuda_cpu_drift_discriminator_rollup_20260509T110211Z.json
```

## Dispatch deliverable status

Per CLAUDE.md "parallel-dispatch first" rule + operator approval:

- **CPU GHA dispatches (4 variants × free)**: actuator runbook ready
  (`scripts/remote_lane_*.sh`). Operator can run via:
  ```bash
  DISCRIMINATOR_TIMESTAMP_SUFFIX=20260509T110211Z \
    bash scripts/remote_lane_avvideodataset_cuda_path_mechanism_discriminator.sh
  ```
- **CUDA dispatches (4 variants × ~$0.20)**: deferred-pending operator decision
  on dispatch wrapper (Lightning T4 vs Vast.ai 4090 vs Modal A100). Per CLAUDE.md
  "NEVER invent CLI flags" — subagent context cannot select a wrapper without
  operator approval. Runbook surfaces 3 candidate paths with cost estimates.
  Operator can enable CUDA dispatch via:
  ```bash
  DISCRIMINATOR_TIMESTAMP_SUFFIX=20260509T110211Z \
    SKIP_CUDA=0 CUDA_PROVIDER=lightning \
    bash scripts/remote_lane_avvideodataset_cuda_path_mechanism_discriminator.sh
  ```

The dispatch step is **NOT a blocker** for the lane's L2 promotion — that
requires `real_archive_empirical` evidence which is exactly what the dispatch
will produce.

## Downstream implications

The discriminator's verdict will reshape:

1. **Drift predictor accuracy** — `tools/xray_cpu_cuda_drift_per_arch_class.py`
   currently uses a single class-level R_pose. With per-mechanism field set,
   the predictor can compute a tighter band per-archive.
2. **Per-architecture-class bit allocation** — if loader-byte drift dominates,
   bytes spent on stabilising frame-decode produce more value at CPU substrate.
   If conv-kernel dominates, bytes spent on numerically-stable kernels matter.
   If Hydra dominates, bytes spent on Hydra-head conditioning matter.
3. **Cross-paradigm composition** — extending the same 3-mechanism split to
   non-HNeRV families (Ballé, MNeRV, raw_av1_yuv) is straightforward once the
   HNeRV mechanism is known. The discriminator-builder pattern is reusable.
4. **Cathedral autopilot CPU-prediction step** — every CUDA dispatch can now
   compute a corrected expected-CPU-score using the per-mechanism profile
   instead of the global R_pose=5.04 prior.

## 6-hook wire-in design notes (per directive deliverable §7)

1. **Sensitivity-map** (`tac.sensitivity_map`): drift mechanism is now part of
   the score-axis prediction. The `mechanism_discriminator_verdict` field on
   `ArchitectureProfile` exposes which mechanism to prioritize when computing
   per-tensor / per-channel sensitivity.
2. **Pareto solver** (`tac.optimization.meta_lagrangian_allocator`): the
   per-class drift correction tightens the achievable region. When a class has
   `loader_drift_correction != None`, the achievable score band's lower bound
   tightens by approximately `loader_drift_correction × pose_marginal`.
3. **Bit-allocator** (`tac.codec.*`): per-tensor importance for CPU vs CUDA
   targets becomes per-class. Loader-dominated classes weight tensors that
   affect frame-decode-stability; conv-dominated classes weight tensors that
   affect numerically-stable kernels.
4. **Cathedral autopilot** (`tools/cathedral_autopilot.py`): every dispatch can
   compute a corrected expected-CPU-score from the CUDA result via
   `confidence_aware_score_band(architecture_class, cuda_score)` — this already
   exists; the discriminator verdict makes it more accurate.
5. **Continual-learning** (`src/tac/continual_learning.py`): posterior update on
   the drift mechanism class is straightforward: extend the existing
   `posterior_update` to also stamp the mechanism verdict on accepted anchors.
   (Not landed in this commit; future work.)
6. **Probe-disambiguator**: future variants can use the discriminator pattern
   to pick the right inflate path. The `_MUTATORS` dict is the canonical
   extension point.

## Reactivation criteria

If the dispatch returns `FOURTH_MECHANISM_HYPOTHESIS`:

1. Hypothesise a 4th mechanism (e.g. SegNet boundary-pixel sensitivity, raster
   ordering, batch-size effects, BatchNorm running-stats drift).
2. Design a 4th variant with an isolation that probes it.
3. Re-dispatch the SAME 4 variants + the new 4th variant.
4. Recompute verdict.

If the dispatch returns `INCONCLUSIVE_NO_BASELINE` or `INCONCLUSIVE_VARIANTS_MISSING`:

1. Re-dispatch the missing variant(s).
2. The discriminator-builder is idempotent; rebuilding is cheap.

## Cross-references

- Source memo: `~/.claude/projects/.../feedback_domain_exploitation_catalog_landed_20260509.md`
- Mechanism deep-dive: `~/.claude/projects/.../feedback_cuda_cpu_pose_drift_mechanism_deep_dive_20260508.md`
- Decoder-third-axis design: `~/.claude/projects/.../feedback_decoder_drift_third_axis_20260508.md`
- Sister discriminator (loader probe): `.omx/research/loader_drift_discriminator_hardening_20260508_worker_b.md`
- Per-architecture-class registry: `src/tac/optimization/cuda_cpu_axis_profile_registry.py` (commit 697bfe01)
- A1 canonical CPU anchor: `experiments/results/a1_latentalign_importpathfix_cpu_eval_gha_20260509/contest_auth_eval.adjudicated.json` (0.19284757 [contest-CPU GHA Linux x86_64])
- HIGH 1 fix (GHA dispatcher submission-name matching): `tools/dispatch_cpu_eval_via_github_actions.py` already has the AmbiguousSubmissionMatchError fix; no blocker.
- Catalog claim number: 129 (`tools/claim_catalog_number.py claim`)
