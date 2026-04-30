# Component Sensitivity OWV3/NWCS Execution Plan - 2026-04-30

Owner: Codex

Scope: CUDA component-sensitivity workstream for OWV3 and NWCS. This is an
execution/readiness report, not a score ledger. No GPU work was run for this
review.

## Reviewed Surfaces

- `experiments/profile_component_sensitivity.py`
- `experiments/build_component_sensitivity_manifest.py`
- `src/tac/component_sensitivity_artifact.py`
- `src/tac/sensitivity_map.py`
- `experiments/convert_fisher_to_owv3_sensitivity_map.py`
- `experiments/build_lane_g_v3_owv3_stack.py`
- `experiments/sweep_owv3_byte_plan.py`
- `src/tac/owv3_sensitivity_weighted.py`
- `src/tac/neural_weight_codec_sensitivity.py`
- `scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh`
- `scripts/remote_lane_j_nwcs_ec_stack.sh`
- Focused tests under `src/tac/tests/`.

## Current Artifact State

No `component_sensitivity_v1` JSON artifact exists locally yet.

Authoritative score/frontier references:

- Lane G v3 anchor:
  - archive: `experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip`
  - bytes: `694074`
  - sha256: `9b20bdfca246d8e32cc19da966c84cdae7e34f6b247161d107ec43cb9ef6870b`
  - CUDA score-grade eval: `experiments/results/lane_g_v3_landed/contest_auth_eval.json`
  - recomputed score: `1.048866504774971`
  - PoseNet: `0.00345458`
  - SegNet: `0.00400846`
  - hardware: RTX 4090, not T4/equivalent.

- PFP16 A++ frontier:
  - archive: `experiments/results/lane_g_v3_pfp16/archive_lane_g_v3_pfp16.zip`
  - bytes: `686635`
  - sha256: `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`
  - T4 exact eval: `experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/contest_auth_eval.json`
  - recomputed score: `1.043987524793892`
  - PoseNet: `0.00346442`
  - SegNet: `0.00400656`
  - hardware: Tesla T4, `gpu_t4_match=true`.

- Existing OWV3 CUDA-Fisher source:
  - map: `experiments/results/lane_g_v3_owv3_fisher_lightning_20260430_codex_r2/owv3_sensitivity_map.pt`
  - map sha256: `ed69bec3c9c530e4d574d82d3b6764399a6feca0289f2114899fa09689fabeba`
  - source device: CUDA
  - source scope: 30 pairs, Fisher proxy, not `component_sensitivity_v1`.
  - build archive: `experiments/results/lane_g_v3_owv3_fisher_lightning_20260430_codex_r2/archive_lane_g_v3_owv3.zip`
  - build bytes: `689342`, sha256 `29a02b2af2c37371eec80ca3e278c4ce368703ba0a0a2121e2b32f570106a84c`
  - status: byte-infeasible vs PFP16 frontier by `+2707` bytes; no exact eval.

- Existing OWV3 byte-feasible candidate:
  - archive: `experiments/results/lane_g_v3_owv3_byte_plan_sweep_worker_b/best_byte_feasible/archive_lane_g_v3_owv3.zip`
  - bytes: `686557`
  - sha256: `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`
  - candidate: `owv3_0018_bbr0p69_protect0p0014_aggr1em05`
  - knobs: `bit_budget_ratio=0.69`, `protect_threshold=0.0014`, `aggressive_threshold=1e-5`, `fallback_action=keep_asym`
  - byte status: `-78` bytes vs PFP16 A++ frontier, `-7517` bytes vs Lane G v3.
  - source map: same 30-pair Fisher proxy above.
  - status: byte-only pending exact CUDA auth eval. Not sensitivity-promotion evidence.

## Current Code Readiness

`src/tac/component_sensitivity_artifact.py` is a useful fail-closed custody
validator. Promotion mode requires CUDA device fields, no debug/proxy/random
markers, checkpoint/video/upstream custody, sample plan, PoseNet/SegNet/combined
maps, stability, response curves, exact eval custody, and `n_samples == 600`.

`experiments/build_component_sensitivity_manifest.py` deterministically assembles
the manifest from existing map/curve/stability/eval artifacts. It rejects
non-CUDA eval JSON, wrong sample counts, missing map tensors, and response
curves without a holdout-error field.

`experiments/profile_component_sensitivity.py` now provides the missing producer
shape: CUDA-only by default, absolute pair IDs, calibration/holdout split,
PoseNet/SegNet/combined map files, response JSON, stability JSON, and optional
manifest handoff.

Important limitation: the producer is still a Fisher-proxy producer, not yet a
promotion-grade official-component finite-difference producer. See blockers.

## Exact CUDA Command Plan

Use a CUDA-visible, supply-chain-clean runner. Lightning T4/equivalent is
preferred for A++ wording.

### 0. Runner Preflight

```bash
RUN_ID=component_sensitivity_owv3_nwcs_20260430_r1
.venv/bin/python scripts/scan_lightning_supply_chain.py --strict \
  --json-out ".omx/state/lightning_supply_chain_scan_${RUN_ID}.json"
nvidia-smi
.venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
PY
```

### 1. Anchor Exact Eval

For A++ component-sensitivity custody, rerun exact eval for the Lane G v3 anchor
on T4/equivalent. The existing Lane G v3 eval is CUDA A-grade but not T4.

```bash
EVID=experiments/results/component_sensitivity_owv3_nwcs_20260430
mkdir -p "$EVID/anchor_eval"
.venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir upstream \
  --device cuda \
  --keep-work-dir \
  --work-dir "$EVID/anchor_eval"
```

If only A-grade sensitivity custody is needed, the current local path can use:

```text
experiments/results/lane_g_v3_landed/contest_auth_eval.json
```

### 2. Component Sensitivity Producer

Promotion-intended run must use all 600 pairs. Top-k pair mode is diagnostic
unless explicitly scoped non-promotable.

After the response-curve gaps below are fixed, run:

```bash
EVID=experiments/results/component_sensitivity_owv3_nwcs_20260430
.venv/bin/python experiments/profile_component_sensitivity.py \
  --checkpoint experiments/results/lane_g_v3_landed/iter_0/renderer.bin \
  --video upstream/videos/0.mkv \
  --masks-mkv experiments/results/lane_g_v3_landed/iter_0/masks.mkv \
  --poses experiments/results/lane_g_v3_landed/iter_0/optimized_poses.pt \
  --upstream upstream \
  --output-dir "$EVID/profile" \
  --all-pairs \
  --pair-batch 2 \
  --response-top-k 32 \
  --response-epsilons=-0.002,-0.001,-0.0005,0,0.0005,0.001,0.002 \
  --split-seed 20260430 \
  --holdout-fraction 0.2 \
  --aggregate sum \
  --device cuda \
  --archive experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \
  --contest-auth-eval-json "$EVID/anchor_eval/contest_auth_eval.json" \
  --manifest-output "$EVID/component_sensitivity_v1.json" \
  --evidence-grade A
```

If reusing the existing non-T4 Lane G eval, replace the eval JSON path with
`experiments/results/lane_g_v3_landed/contest_auth_eval.json` and keep
`--evidence-grade A`.

Expected producer outputs:

- `$EVID/profile/posenet_sensitivity_map.pt`
- `$EVID/profile/segnet_sensitivity_map.pt`
- `$EVID/profile/combined_sensitivity_map.pt`
- `$EVID/profile/posenet_response_curve.json`
- `$EVID/profile/segnet_response_curve.json`
- `$EVID/profile/combined_response_curve.json`
- `$EVID/profile/stability.json`
- `$EVID/profile/sample_plan.json`
- `$EVID/profile/component_sensitivity_profile_summary.json`
- `$EVID/component_sensitivity_v1.json`

### 3. Explicit Manifest Assembly Alternative

If the producer is run without `--manifest-output`, assemble explicitly:

```bash
EVID=experiments/results/component_sensitivity_owv3_nwcs_20260430
.venv/bin/python experiments/build_component_sensitivity_manifest.py \
  --checkpoint experiments/results/lane_g_v3_landed/iter_0/renderer.bin \
  --video upstream/videos/0.mkv \
  --upstream upstream \
  --archive experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \
  --contest-auth-eval-json "$EVID/anchor_eval/contest_auth_eval.json" \
  --posenet-map "$EVID/profile/posenet_sensitivity_map.pt" \
  --segnet-map "$EVID/profile/segnet_sensitivity_map.pt" \
  --combined-map "$EVID/profile/combined_sensitivity_map.pt" \
  --posenet-response-curve "$EVID/profile/posenet_response_curve.json" \
  --segnet-response-curve "$EVID/profile/segnet_response_curve.json" \
  --combined-response-curve "$EVID/profile/combined_response_curve.json" \
  --stability-json "$EVID/profile/stability.json" \
  --sample-plan-json "$EVID/profile/sample_plan.json" \
  --output "$EVID/component_sensitivity_v1.json" \
  --device cuda \
  --evidence-grade A \
  --n-samples 600 \
  --n-pairs 600 \
  --split-seed 20260430
```

Validate:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from tac.component_sensitivity_artifact import validate_component_sensitivity_manifest
p = Path("experiments/results/component_sensitivity_owv3_nwcs_20260430/component_sensitivity_v1.json")
validate_component_sensitivity_manifest(json.loads(p.read_text()), promotion=True)
print("component_sensitivity_v1 validation passed")
PY
```

### 4. OWV3 Consumer Commands

Build/sweep OWV3 from the validated combined component map. This is still
byte-only until exact eval.

```bash
EVID=experiments/results/component_sensitivity_owv3_nwcs_20260430
.venv/bin/python experiments/sweep_owv3_byte_plan.py \
  --sensitivity-map "$EVID/profile/combined_sensitivity_map.pt" \
  --output-dir "$EVID/owv3_byte_plan_sweep" \
  --overwrite \
  --preset frontier \
  --fallback-action keep_asym \
  --archive-policy selected \
  --decode-verify selected \
  --frontier-comparator-bytes 686635 \
  --frontier-comparator-sha256 0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f \
  --frontier-comparator-label PFP16_A++
```

Then exact eval only the selected byte-plausible archive:

```bash
EVID=experiments/results/component_sensitivity_owv3_nwcs_20260430
.venv/bin/python experiments/contest_auth_eval.py \
  --archive "$EVID/owv3_byte_plan_sweep/best_byte_feasible/archive_lane_g_v3_owv3.zip" \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir upstream \
  --device cuda \
  --keep-work-dir \
  --work-dir "$EVID/owv3_exact_eval"
```

### 5. NWCS Consumer Commands

NWCS cannot consume the current component maps directly. It needs two derived
artifacts that do not yet have a checked-in builder:

- `anchor_sensitivity.pt`: per-parameter per-block sensitivities for the Lane G
  v3 anchor renderer, with `nwcs_anchor_parameter_sensitivity_v1` metadata.
- `corpus_sensitivity.pt`: corpus-manifest-ordered per-block sensitivities, with
  `nwcs_corpus_block_sensitivity_v1` metadata.

Required metadata link for both:

```text
component_sensitivity_manifest_sha256 = sha256(component_sensitivity_v1.json)
```

After those builders exist, the NWCS run shape is:

```bash
EVID=experiments/results/component_sensitivity_owv3_nwcs_20260430
AUTH_EVAL_DEVICE=cuda \
NWCS_ALLOW_DEBUG_SENSITIVITY=0 \
ANCHOR_LANE_G_V3_ARCHIVE=experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \
ANCHOR_CORPUS_DIR=experiments/results \
ANCHOR_SENSITIVITY_PT="$EVID/nwcs/anchor_sensitivity.pt" \
CORPUS_SENSITIVITY_PT="$EVID/nwcs/corpus_sensitivity.pt" \
LOG_DIR="$EVID/nwcs_run" \
bash scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh
```

Do not spend exact-eval time on NWCS until a build-only stop exists or the
resulting deterministic archive has been inspected before Stage 7.

## Required Inputs

Common:

- CUDA-visible runner; T4/equivalent for A++ wording.
- Strict Lightning supply-chain scan JSON.
- Source/staged-tree manifest for the exact code used.
- `upstream/videos/0.mkv`
- `upstream/models/posenet.safetensors`
- `upstream/models/segnet.safetensors`
- `submissions/robust_current/inflate.sh`
- `upstream/evaluate.py` unmodified.

Lane G v3 anchor:

- `experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip`
- `experiments/results/lane_g_v3_landed/iter_0/renderer.bin`
- `experiments/results/lane_g_v3_landed/iter_0/masks.mkv`
- `experiments/results/lane_g_v3_landed/iter_0/optimized_poses.pt`
- exact CUDA `contest_auth_eval.json` for the same archive.

OWV3:

- validated `component_sensitivity_v1.json`
- `combined_sensitivity_map.pt` with all Conv2d channel keys and CUDA metadata
- PFP16 comparator bytes/SHA for byte gating.

NWCS:

- validated `component_sensitivity_v1.json`
- deterministic corpus manifest from `experiments/train_neural_weight_codec.py`
- `anchor_sensitivity.pt` with per-parameter block counts, shapes, anchor archive
  SHA, anchor renderer SHA, block size, and component-manifest SHA.
- `corpus_sensitivity.pt` with corpus manifest SHA, total selected blocks, block
  size, ordering, transfer rule, and component-manifest SHA.

## Promotion Blockers

1. `profile_component_sensitivity.py` uses SegNet cross-entropy proxy for the
   SegNet map and response curves. Official promotion needs CUDA argmax
   disagreement from the upstream SegNet value path for the validation curves,
   and either an official finite-difference SegNet map or an explicitly
   validated proxy-to-argmax response gate.

2. Current response curves are one-sided positive perturbations. Promotion
   needs `eps=0` plus symmetric `-eps/+eps` and normally `-2eps/+2eps`, or a
   directional-action declaration for every atom. `apply_channel_perturbation`
   currently rejects negative epsilon, so this is a code gap.

3. Current `holdout_error` is the maximum observed absolute delta, not the
   prediction error between map-derived sensitivity and observed holdout damage.
   Response JSON needs threshold, `passed=true`, predicted delta, observed
   delta, and error fields per component.

4. Current combined response is a local proxy term. Promotion combined deltas
   must be recomputed from measured component means:

   ```text
   100 * (seg_eps - seg_0) + sqrt(10 * pose_eps) - sqrt(10 * pose_0)
   ```

5. Stability JSON is too thin. It records CV and one top-k overlap, but not
   Pearson/Spearman rank correlation, top 1%/5%/10% overlap, thresholds, pass
   booleans, per-layer summaries, or zero/nonfinite/negative/clipped counts.

6. Manifest validation checks metadata presence and custody, not scientific
   truth. Add tests that reject proxy-only SegNet promotion, missing symmetric
   curves, absent pass thresholds, and response curves with no official argmax
   readout.

7. NWCS has no checked-in builders or validators for
   `nwcs_anchor_parameter_sensitivity_v1` and
   `nwcs_corpus_block_sensitivity_v1`. The remote scripts validate some shape
   and hash metadata, but they do not enforce `component_sensitivity_v1` SHA
   linkage.

8. `scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh` still proceeds from
   archive build into exact eval. Add `NWCS_BUILD_ONLY=1` or `STOP_AFTER_STAGE=6`
   before using it as a pre-eval build harness.

9. Existing OWV3 byte-feasible archive uses a 30-pair Fisher proxy map. It may
   be worth exact eval as a byte-plausible diagnostic, but it does not satisfy
   the component-sensitivity promotion contract.

## Minimum Code/Test Work Before Promotion

1. Patch `profile_component_sensitivity.py` to compute official CUDA component
   values for response curves: PoseNet MSE, SegNet argmax disagreement, and
   combined formula from mean component distances.

2. Add symmetric perturbation support and record perturbation basis metadata:
   atom ordering, epsilon units, sign convention, normalization, clamp/domain
   rules, split seed, pair IDs, and response-curve thresholds.

3. Expand `stability.json` to include Spearman, Pearson, top 1%/5%/10% overlap,
   per-layer summaries, counts, thresholds, and pass booleans.

4. Harden `build_component_sensitivity_manifest.py` or
   `component_sensitivity_artifact.py` so promotion mode rejects proxy-only
   SegNet curves, missing `passed=true`, missing thresholds, and asymmetric
   curves without directional metadata.

5. Add NWCS sensitivity builders:
   - `experiments/build_nwcs_anchor_sensitivity.py`
   - `experiments/build_nwcs_corpus_sensitivity.py`

6. Add NWCS validators/tests for anchor/corpus sensitivity schema, manifest SHA
   linkage, anchor archive/renderer SHA, corpus manifest SHA, shape/block count,
   nonnegative finite values, and positive scorer signal.

7. Add NWCS build-only stop before exact eval and a focused test that Stage 6
   can emit a deterministic archive without entering Stage 7.

## Verification Performed Locally

No GPU work was run.

Passed:

```bash
.venv/bin/python -m py_compile \
  experiments/profile_component_sensitivity.py \
  experiments/build_component_sensitivity_manifest.py \
  src/tac/component_sensitivity_artifact.py \
  src/tac/sensitivity_map.py \
  src/tac/owv3_sensitivity_weighted.py \
  src/tac/neural_weight_codec_sensitivity.py \
  experiments/sweep_owv3_byte_plan.py \
  experiments/build_lane_g_v3_owv3_stack.py

.venv/bin/python -m pytest \
  src/tac/tests/test_profile_component_sensitivity.py \
  src/tac/tests/test_build_component_sensitivity_manifest.py \
  src/tac/tests/test_component_sensitivity_artifact.py \
  src/tac/tests/test_sensitivity_map.py \
  src/tac/tests/test_owv3_sensitivity_conversion.py \
  src/tac/tests/test_owv3_sensitivity_weighted.py \
  src/tac/tests/test_sweep_owv3_byte_plan.py \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py \
  src/tac/tests/test_neural_weight_codec_sensitivity_renderer_format.py \
  -q

bash -n \
  scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh \
  scripts/remote_lane_j_nwcs_ec_stack.sh \
  scripts/remote_lane_g_v3_owv3_fisher_stack.sh
```

Result:

```text
82 passed in 1.97s
```

## Decision

Do not produce or label a promotable `component_sensitivity_v1` yet. The current
path is ready for a diagnostic CUDA producer run, but promotion should wait for
the official SegNet argmax response gate, symmetric/directional response-curve
metadata, stronger stability thresholds, and NWCS per-block sensitivity builders.
