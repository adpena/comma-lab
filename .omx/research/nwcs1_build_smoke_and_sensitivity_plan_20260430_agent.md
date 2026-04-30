# NWCS1 Build-Only Smoke And Sensitivity Provenance Plan - 2026-04-30

Evidence grade: empirical/build-only for local checks below. No score claim,
ranking claim, promotion claim, or method retirement claim is made here.

This note covers Lane J-NWC/J-NWCS only. Exact score truth remains CUDA auth
eval on exact archive bytes through `archive.zip -> inflate.sh ->
upstream/evaluate.py`; none of the local smoke outputs below used that path.

## Inspected Surfaces

- `src/tac/neural_weight_codec.py`
  - Base `NWC1` VQ-style weight codec.
  - `WeightCodecConfig`, `WeightCodec`, `train_codec`, tensor block packing.
  - `export_neural_compressed_checkpoint` embeds the codec checkpoint in the
    renderer payload.
- `src/tac/neural_weight_corpus.py`
  - Deterministic corpus manifest and replay.
  - Records path, relative path, byte count, SHA-256, shape, dtype, block
    count, selected block ranges, caps, inclusion/exclusion reasons.
- `src/tac/neural_weight_codec_sensitivity.py`
  - `NWCS_RENDERER_MAGIC = b"NWCS1\0\0\0"`.
  - Deterministic `NWCS1` container: sorted tensor entries, JSON header with
    sorted keys, embedded codec checkpoint, signed int64 length fields,
    duplicate/truncation/negative-length/trailing-byte rejection.
  - Variable-codebook per-block encoding and decoding.
- `src/tac/renderer_export.py`
  - `detect_checkpoint_type` recognizes `NWCS1`.
  - `load_nwcs_sensitivity_compressed_checkpoint` decodes per-tensor blobs and
    loads into an inferred renderer when real architecture config is present.
  - If metadata config is `{"tensor_only": true}`, the loader only stashes
    decoded tensors on `_nwcs_state_dict`; that is useful for tests but is not
    sufficient for contest inflate rendering.
- `submissions/robust_current/inflate_renderer.py`
  - `_load_renderer` dispatches `raw_bytes[:8] == b"NWCS1\0\0\0"` to
    `tac.renderer_export.load_nwcs_sensitivity_compressed_checkpoint`.
- Remote scripts:
  - `scripts/remote_lane_j_nwc_neural_weight_compression.sh`
  - `scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh`
  - `scripts/remote_lane_j_nwcs_ec_stack.sh`
  - Good gates: `AUTH_EVAL_DEVICE` must be `cuda`, archive extraction is
    zip-slip safe, anchor members are allowlisted, no `extractall`, provenance
    records artifact custody, NWCS promotion requires anchor and corpus
    sensitivity metadata unless explicitly debug/non-promotable.
- Component sensitivity:
  - `src/tac/component_sensitivity_artifact.py` validates
    `component_sensitivity_v1`.
  - `experiments/build_component_sensitivity_manifest.py` assembles
    promotion-grade manifests from existing CUDA maps, response curves,
    stability JSON, exact archive/eval custody. It does not compute maps.

## Local Commands Run

```bash
test -x .venv/bin/python && .venv/bin/python --version || python3 --version
bash -n scripts/remote_lane_j_nwc_neural_weight_compression.sh
bash -n scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh
bash -n scripts/remote_lane_j_nwcs_ec_stack.sh
.venv/bin/python -m py_compile \
  src/tac/neural_weight_codec.py \
  src/tac/neural_weight_codec_sensitivity.py \
  src/tac/neural_weight_corpus.py \
  src/tac/component_sensitivity_artifact.py \
  experiments/train_neural_weight_codec.py \
  experiments/build_component_sensitivity_manifest.py
.venv/bin/python -m pytest \
  src/tac/tests/test_neural_weight_codec_sensitivity_renderer_format.py \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py \
  src/tac/tests/test_neural_weight_codec_corpus.py \
  src/tac/tests/test_component_sensitivity_artifact.py \
  src/tac/tests/test_build_component_sensitivity_manifest.py \
  -q
```

Observed output:

```text
Python 3.12.13
bash -n: passed for all three scripts
py_compile: passed
pytest: 48 passed in 1.54s
```

I also ran two local non-score NWCS1 build-only smokes under `/tmp`.

Tiny synthetic smoke:

```text
checkpoint_type: neural_weight_compression_sensitivity_v1
is_nwcs_renderer_container: true
container_tensor_count: 1
loaded_state_keys: ["layer.weight"]
renderer_bytes: 6592
archive_bytes: 3134
archive_members: ["renderer.bin", "masks.mkv", "optimized_poses.pt"]
promotion_eligible: false
score_claim: false
```

Real Lane G v3 anchor build-only smoke:

```text
anchor_archive: experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip
anchor_archive_bytes: 694074
anchor_archive_sha256: 9b20bdfca246d8e32cc19da966c84cdae7e34f6b247161d107ec43cb9ef6870b
anchor_renderer_bytes: 296776
anchor_renderer_sha256: 08f12d722dd33f9061deee72f49d782035597f78cd65ed1463a241ab430a7529
checkpoint_type: neural_weight_compression_sensitivity_v1
is_nwcs_renderer_container: true
encoded_parameter_tensors: 73
loaded_model_type: AsymmetricPairGenerator
loaded_param_count: 288363
nwcs_renderer_bytes: 103281
nwcs_renderer_sha256: 8a82dd1057271633ab50ff9a5b426a1f84c223ec91118138469f680f3296ce3f
archive_bytes: 468735
archive_sha256: 01e31ffca938e12c961094ba08d08bb747493629f3276b6ff2f469fde3319eb2
archive_members: ["renderer.bin", "masks.mkv", "optimized_poses.pt"]
promotion_eligible: false
score_claim: false
```

The real-anchor smoke used synthetic uniform sensitivities and an untrained
tiny codec. It proves only that the `NWCS1` container can be emitted, detected,
loaded through the real architecture path, and packed into a deterministic
contest-shaped ZIP. It says nothing about score or distortion.

## Immediate Blockers

1. **NWCS remote Stage 5 arch-config import bug.**
   `scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh` and
   `scripts/remote_lane_j_nwcs_ec_stack.sh` call `_infer_asymmetric_config(model)`
   in Stage 5 but import only `load_any_renderer_checkpoint`. The broad
   `except Exception` catches the resulting `NameError` and silently writes
   `{"tensor_only": true}` into the `NWCS1` metadata. In contest inflate, that
   can return a default `AsymmetricPairGenerator` with decoded tensors stashed
   on `_nwcs_state_dict`, not a renderer loaded with decoded weights.
   Fix before any real build-only or exact run:

   ```python
   from tac.renderer_export import _infer_asymmetric_config, load_any_renderer_checkpoint
   ```

   Add a static or functional test that Stage 5 imports `_infer_asymmetric_config`
   in the same Python snippet and that real-anchor `NWCS1` load does not fall
   back to `_nwcs_state_dict`.

2. **No CUDA component-sensitivity producer is present.**
   The validator and assembler exist, but there is no checked-in producer for
   PoseNet-only, SegNet-only, and combined CUDA maps plus response curves.

3. **No promotable `ANCHOR_SENSITIVITY_PT` / `CORPUS_SENSITIVITY_PT` is present.**
   Remote scripts correctly reject missing true sensitivity unless
   `NWCS_ALLOW_DEBUG_SENSITIVITY=1`; debug mode is non-promotable.

4. **Remote J-NWCS script has no build-only stop.**
   It proceeds from archive build into `contest_auth_eval.py`. Add an explicit
   `NWCS_BUILD_ONLY=1` or `STOP_AFTER_STAGE=6` gate for pre-eval build smoke.

5. **NWCS per-parameter sensitivity metadata is not yet unified with
   `component_sensitivity_v1`.**
   Remote scripts check anchor/corpus SHA, block size, shapes, and block counts,
   but the `.pt` sensitivity artifacts should also cite the validated
   `component_sensitivity_v1` manifest SHA and component map/response-curve
   custody.

## Sensitivity Provenance Contract

Promotion-capable NWCS must have three linked artifacts before exact eval spend:

1. `component_sensitivity_v1.json`
   - CUDA device only.
   - PoseNet, SegNet, and combined maps.
   - Calibration and holdout sample split with `split_hash`.
   - Stability metrics for cv/rank/top-k.
   - Response curves with holdout error for all three components.
   - Exact archive and `contest_auth_eval.json` custody.

2. `anchor_sensitivity.pt`
   - Suggested format:

   ```python
   {
       "format": "nwcs_anchor_parameter_sensitivity_v1",
       "sensitivities": {param_name: torch.Tensor[n_blocks]},
       "metadata": {
           "device": "cuda",
           "component_sensitivity_manifest_sha256": "<sha256>",
           "anchor_archive_sha256": "<sha256>",
           "anchor_renderer_sha256": "<sha256>",
           "block_size": 16,
           "reduction": "combined_score_gradient_energy_with_posenet_segnet_breakdown",
           "parameters": {
               param_name: {
                   "shape": [...],
                   "dtype": "torch.float32",
                   "numel": int,
                   "block_count": int,
                   "posenet_source": "<map key or file>",
                   "segnet_source": "<map key or file>",
                   "combined_source": "<map key or file>"
               }
           }
       }
   }
   ```

3. `corpus_sensitivity.pt`
   - Suggested format:

   ```python
   {
       "format": "nwcs_corpus_block_sensitivity_v1",
       "values": torch.Tensor[manifest_totals.selected_blocks],
       "metadata": {
           "device": "cuda",
           "component_sensitivity_manifest_sha256": "<sha256>",
           "corpus_manifest_sha256": "<sha256>",
           "block_size": 16,
           "num_blocks": int,
           "ordering": "corpus_manifest selected tensor order and block ranges",
           "transfer_rule": "per-checkpoint CUDA map or reviewed anchor-transfer rule"
       }
   }
   ```

For promotion, `values` must be finite, nonnegative, nonempty, and contain
positive scorer signal. Uniform, random, fake, CPU, MPS, proxy, or smoke values
must remain explicitly non-promotable.

## Exact Command Plan

Runner trust preflight:

```bash
RUNNER_ID=lightning_nwcs1_r1
.venv/bin/python scripts/scan_lightning_supply_chain.py --strict \
  --json-out ".omx/state/lightning_supply_chain_scan_${RUNNER_ID}_20260430.json"
nvidia-smi
.venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
PY
```

Build or replay the deterministic corpus manifest:

```bash
.venv/bin/python experiments/train_neural_weight_codec.py \
  --corpus-dir experiments/results \
  --output experiments/results/nwcs1_promote_20260430/base_codec.pt \
  --manifest-out experiments/results/nwcs1_promote_20260430/corpus_manifest.json \
  --num-steps 2000 \
  --batch-size 256 \
  --lr 1e-3 \
  --device cpu \
  --block-size 16 \
  --codebook-size 64 \
  --latent-dim 16 \
  --hidden 64 \
  --max-corpus-files 200 \
  --max-blocks-per-ckpt 50000 \
  --seed 1234
```

Assemble `component_sensitivity_v1` after CUDA maps/curves exist:

```bash
CUDA_ANCHOR_EVAL_DIR=experiments/results/nwcs1_component_cuda_20260430/anchor_eval
CUDA_SENS_DIR=experiments/results/nwcs1_component_cuda_20260430/sensitivity
.venv/bin/python experiments/build_component_sensitivity_manifest.py \
  --checkpoint experiments/results/nwcs1_promote_20260430/anchor/renderer.bin \
  --video upstream/videos/0.mkv \
  --upstream upstream \
  --archive experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \
  --contest-auth-eval-json "${CUDA_ANCHOR_EVAL_DIR}/contest_auth_eval.json" \
  --posenet-map "${CUDA_SENS_DIR}/posenet_map.pt" \
  --segnet-map "${CUDA_SENS_DIR}/segnet_map.pt" \
  --combined-map "${CUDA_SENS_DIR}/combined_map.pt" \
  --posenet-response-curve "${CUDA_SENS_DIR}/posenet_response_curve.json" \
  --segnet-response-curve "${CUDA_SENS_DIR}/segnet_response_curve.json" \
  --combined-response-curve "${CUDA_SENS_DIR}/combined_response_curve.json" \
  --stability-json "${CUDA_SENS_DIR}/stability.json" \
  --output experiments/results/nwcs1_promote_20260430/component_sensitivity_v1.json \
  --device cuda \
  --evidence-grade A \
  --n-samples 600 \
  --n-pairs 600 \
  --split-seed 20260430
```

Validate the assembled manifest explicitly:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from tac.component_sensitivity_artifact import validate_component_sensitivity_manifest
p = Path("experiments/results/nwcs1_promote_20260430/component_sensitivity_v1.json")
validate_component_sensitivity_manifest(json.loads(p.read_text()), promotion=True)
print("component_sensitivity_v1 promotion validation passed")
PY
```

Run NWCS build-only smoke after adding a build-only stop:

```bash
AUTH_EVAL_DEVICE=cuda \
NWCS_ALLOW_DEBUG_SENSITIVITY=0 \
NWCS_BUILD_ONLY=1 \
ANCHOR_LANE_G_V3_ARCHIVE=experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \
ANCHOR_CORPUS_DIR=experiments/results \
ANCHOR_SENSITIVITY_PT=experiments/results/nwcs1_promote_20260430/anchor_sensitivity.pt \
CORPUS_SENSITIVITY_PT=experiments/results/nwcs1_promote_20260430/corpus_sensitivity.pt \
LOG_DIR=experiments/results/nwcs1_build_only_20260430 \
bash scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh
```

Required build-only acceptance checks:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib
import zipfile
from tac.renderer_export import detect_checkpoint_type, load_nwcs_sensitivity_compressed_checkpoint

archive = Path("experiments/results/nwcs1_build_only_20260430/archive_lane_j_nwcs.zip")
renderer = Path("experiments/results/nwcs1_build_only_20260430/renderer_nwcs.bin")
assert renderer.read_bytes()[:8] == b"NWCS1\0\0\0"
assert detect_checkpoint_type(renderer) == "neural_weight_compression_sensitivity_v1"
model = load_nwcs_sensitivity_compressed_checkpoint(renderer, device="cpu")
assert not hasattr(model, "_nwcs_state_dict"), "real archive config fell back to tensor_only"
with zipfile.ZipFile(archive) as zf:
    names = [info.filename for info in zf.infolist()]
assert names == ["renderer.bin", "masks.mkv", "optimized_poses.pt"]
h = hashlib.sha256()
h.update(archive.read_bytes())
print({"archive_bytes": archive.stat().st_size, "archive_sha256": h.hexdigest()})
PY
```

Only after the build-only packet passes and provenance is reviewed, run exact
CUDA auth eval:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/nwcs1_build_only_20260430/archive_lane_j_nwcs.zip \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir upstream \
  --device cuda \
  --keep-work-dir \
  --work-dir experiments/results/nwcs1_exact_cuda_20260430/eval_work
```

Post-eval, parse only `contest_auth_eval.json`, recompute score from
components, verify sample count is 600, verify archive SHA/bytes match the
build-only packet, and apply PoseNet/SegNet component gates before making any
result claim.

## Next Work Order

1. Patch the Stage 5 `_infer_asymmetric_config` import in both NWCS scripts and
   add a regression test that prevents tensor-only fallback for real anchor
   containers.
2. Add `NWCS_BUILD_ONLY=1` or equivalent stop after deterministic archive
   construction, before auth eval.
3. Implement the CUDA component sensitivity producer, or wire an existing CUDA
   producer into `component_sensitivity_v1` maps/curves/stability.
4. Add explicit validators/tests for `nwcs_anchor_parameter_sensitivity_v1` and
   `nwcs_corpus_block_sensitivity_v1`, including manifest SHA linkage.
5. Run build-only smoke with true sensitivity artifacts.
6. Run exact CUDA auth eval only after the build-only and provenance packets are
   clean.
