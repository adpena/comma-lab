# J-NWC/J-NWCS Manifest And Fake-Sensitivity Hardening - 2026-04-30

Evidence grade: engineering hardening / empirical tests only. No score claim,
ranking claim, lane promotion, or method retirement claim is made here.

## Scope

Owned stream: J-NWC/J-NWCS corpus manifest replay and fake/debug sensitivity
promotion hardening.

Reviewed surfaces:

- `scripts/remote_lane_j_nwc_neural_weight_compression.sh`
- `scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh`
- `scripts/remote_lane_j_nwcs_ec_stack.sh`
- `src/tac/neural_weight_codec.py`
- `src/tac/neural_weight_codec_sensitivity.py`
- `src/tac/neural_weight_corpus.py`
- `src/tac/tests/test_neural_weight_codec_corpus.py`
- `src/tac/tests/test_remote_lane_j_nwc_hardening.py`
- `src/tac/tests/test_neural_weight_codec_sensitivity_renderer_format.py`
- `.omx/research/nwcs1_build_smoke_and_sensitivity_plan_20260430_agent.md`

## Changes Landed

1. Corpus manifest replay now rejects unsafe relocated paths.
   - `build_corpus_from_manifest(..., replay_root=...)` rejects absolute
     `relative_path`, backslashes, parent traversal, empty/current-dir parts,
     hidden path parts, and `__MACOSX`.
   - Replayed checkpoint paths are resolved and verified to remain under the
     replay root before loading.
   - Direct in-memory manifest dicts now reject unsupported `schema_version`
     the same way JSON-loaded manifests do.
   - Manifest generation excludes checkpoint paths outside the declared
     `corpus_dir` instead of silently recording selected files with absolute
     non-relocatable `relative_path` values.

2. Debug/fake J-NWCS sensitivity can no longer produce auth-eval-shaped score
   JSON by default.
   - Added `NWCS_BUILD_ONLY="${NWCS_BUILD_ONLY:-0}"` to both J-NWCS scripts.
   - If `NWCS_ALLOW_DEBUG_SENSITIVITY=1` or `NWCS_BUILD_ONLY=1`, the scripts
     stop after archive construction and before `experiments/contest_auth_eval.py`.
   - The build-only exit writes `provenance.json` and `final_record.json` with
     `build_only=true`, `score_claim=false`, `promotion_eligible=false`,
     `auth_eval_skipped=true`, and `result_json=null`.
   - This preserves smoke/build artifacts while preventing synthetic uniform
     sensitivity from entering the normal score-harvest path.

3. Tests now pin the hardened behavior.
   - Added corpus replay tests for traversal/absolute path rejection, direct
     dict schema rejection, and outside-`corpus_dir` exclusion.
   - Added remote-script static guards requiring the debug/build-only stop to
     appear before the auth eval command and to mark no score claim.

## Verification

Commands run:

```bash
.venv/bin/python -m py_compile src/tac/neural_weight_corpus.py
bash -n scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh
bash -n scripts/remote_lane_j_nwcs_ec_stack.sh
bash -n scripts/remote_lane_j_nwc_neural_weight_compression.sh
.venv/bin/python -m pytest \
  src/tac/tests/test_neural_weight_codec_corpus.py \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py \
  -q
.venv/bin/python -m py_compile \
  src/tac/neural_weight_corpus.py \
  src/tac/tests/test_neural_weight_codec_corpus.py \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py
.venv/bin/python -m pytest \
  src/tac/tests/test_neural_weight_codec_sensitivity_renderer_format.py \
  src/tac/tests/test_neural_weight_codec_corpus.py \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py \
  -q
git diff --check -- \
  scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh \
  scripts/remote_lane_j_nwcs_ec_stack.sh \
  src/tac/neural_weight_corpus.py \
  src/tac/tests/test_neural_weight_codec_corpus.py \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py
git diff --no-index --check /dev/null src/tac/neural_weight_corpus.py
git diff --no-index --check /dev/null src/tac/tests/test_neural_weight_codec_corpus.py
git diff --no-index --check /dev/null src/tac/tests/test_remote_lane_j_nwc_hardening.py
git diff --no-index --check /dev/null \
  .omx/research/j_nwc_j_nwcs_manifest_fake_sensitivity_hardening_20260430_codex.md
```

Observed:

```text
18 passed in 0.49s
25 passed in 0.49s
```

All listed syntax, compile, pytest, path-limited diff, and no-index diff
checks passed.

## Changed Files

- `scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh`
- `scripts/remote_lane_j_nwcs_ec_stack.sh`
- `src/tac/neural_weight_corpus.py`
- `src/tac/tests/test_neural_weight_codec_corpus.py`
- `src/tac/tests/test_remote_lane_j_nwc_hardening.py`
- `.omx/research/j_nwc_j_nwcs_manifest_fake_sensitivity_hardening_20260430_codex.md`

## Residual Findings

- No promotable `ANCHOR_SENSITIVITY_PT` or `CORPUS_SENSITIVITY_PT` was present
  in this pass. Exact J-NWCS promotion still requires real CUDA scorer-derived
  sensitivity with anchor/corpus hashes, parameter metadata, positive finite
  signal, and component sensitivity provenance.
- The scripts still rely on the existing inline sensitivity validators. A
  follow-up cleanup could move those validators into a shared Python module so
  the J-NWCS and J-NWCS-EC scripts cannot drift.
- No CUDA auth eval was run in this hardening pass because no candidate
  promotable archive was produced.

## Worker 4 Addendum - 2026-04-30T22:12Z

Scope: J-NWC/J-NWCS exact-eval provenance hardening only. No training dispatch,
CUDA auth eval, score claim, promotion, ranking, or method-retirement claim is
made here.

Patch landed:

- `scripts/remote_lane_j_nwc_neural_weight_compression.sh`
- `scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh`
- `scripts/remote_lane_j_nwcs_ec_stack.sh`
- `src/tac/tests/test_remote_lane_j_nwc_hardening.py`

The three exact-eval paths already ran `scripts/adjudicate_contest_auth_eval.py`
with CUDA/sample/component gates. This addendum closes the remaining custody
gap: final `provenance.json` and `final_record.json` now also surface
`contest_auth_eval.adjudicated.json`, `adjudication_provenance.json`,
`score_source`, and explicit `adjudication_required=true` /
`component_gates_required=true` fields. Artifact custody now hashes both
adjudication files.

Focused verification:

```bash
.venv/bin/python -m py_compile src/tac/tests/test_remote_lane_j_nwc_hardening.py
bash -n scripts/remote_lane_j_nwc_neural_weight_compression.sh \
  scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh \
  scripts/remote_lane_j_nwcs_ec_stack.sh
.venv/bin/python -m pytest \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py \
  src/tac/tests/test_neural_weight_codec_corpus.py \
  src/tac/tests/test_neural_weight_codec_sensitivity_renderer_format.py \
  -q
.venv/bin/python -m pytest src/tac/tests/test_remote_auth_eval_hardening.py -q
git diff --check -- \
  scripts/remote_lane_j_nwc_neural_weight_compression.sh \
  scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh \
  scripts/remote_lane_j_nwcs_ec_stack.sh \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py
```

Observed:

```text
31 passed in 0.52s
26 passed in 1.48s
```

Exact eval command template for any future promotable candidate remains:

```bash
AUTH_EVAL_DEVICE=cuda \
COMPONENT_SENSITIVITY_MANIFEST=<component_sensitivity_v1.json> \
ANCHOR_SENSITIVITY_PT=<anchor_sensitivity.pt> \
CORPUS_SENSITIVITY_PT=<corpus_sensitivity.pt> \
bash scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh
```

The script must produce and preserve `RESULT_JSON`,
`contest_auth_eval.adjudicated.json`, `adjudication_provenance.json`,
`provenance.json`, `final_record.json`, archive SHA/bytes, manifest hashes, and
logs before any score claim is considered.

## Worker D Verification Addendum - 2026-04-30T23:55Z

Scope: read-only verification of the Bernoulli/J-NWC corpus-manifest and
fake/debug-sensitivity hardening, plus this ledger update. No source code,
remote launcher, archive, or test file was edited in this pass. No CUDA auth
eval, score claim, promotion, ranking, or method-retirement claim is made.

Verdict:

- **Patch-level hardening is ready for non-score/build-only use.** The current
  code records deterministic corpus manifests, rejects unsafe/out-of-root
  replay paths, emits explicit build-only/non-promotable records, and prevents
  debug uniform sensitivity from flowing into an auth-eval-shaped result.
- **Promotion remains blocked.** The hardening is a gate, not evidence that
  J-NWCS is score-ready.

Verified code paths:

- `src/tac/neural_weight_corpus.py`:
  - `_manifest_safe_relative_path` rejects absolute paths, backslashes,
    traversal, hidden parts, and `__MACOSX`.
  - Manifest generation excludes checkpoints outside `corpus_dir`.
  - Manifest replay rechecks size, SHA-256, tensor shape, dtype, block count,
    used-block count, and corpus block ordering.
- `scripts/remote_lane_j_nwc_neural_weight_compression.sh`:
  - `NWC_BUILD_ONLY=1` stops after deterministic archive construction and
    writes `score_claim=false`, `promotion_eligible=false`,
    `auth_eval_skipped=true`, `result_json=null`, and artifact custody.
- `scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh` and
  `scripts/remote_lane_j_nwcs_ec_stack.sh`:
  - Promotion-eligible paths require `COMPONENT_SENSITIVITY_MANIFEST`.
  - Promotable `ANCHOR_SENSITIVITY_PT` must be metadata-wrapped, linked to the
    component manifest SHA, anchor archive SHA, anchor renderer SHA, block size,
    parameter shapes, and block counts.
  - Promotable `CORPUS_SENSITIVITY_PT` must be metadata-wrapped, linked to the
    component manifest SHA and corpus manifest SHA, and match corpus block
    count/block size.
  - Debug sensitivity sets `promotion_eligible=false`; debug/build-only paths
    stop before `contest_auth_eval.py` and emit no result JSON.
- `src/tac/neural_weight_codec_sensitivity.py`:
  - `NWCS1` container format is deterministic, magic-tagged, length-prefixed,
    duplicate-name checked, and rejects malformed lengths/trailing bytes.
  - Variable-codebook encode/decode rejects negative, non-finite, mismatched,
    malformed, or trailing sensitivity/codec payload data.

Exact blockers before any J-NWCS promotion:

1. No local promotable `component_sensitivity_v1.json` was found for NWCS.
2. No local promotable `ANCHOR_SENSITIVITY_PT` was found.
3. No local promotable `CORPUS_SENSITIVITY_PT` was found.
4. No checked-in NWCS builders exist yet for
   `nwcs_anchor_parameter_sensitivity_v1` or
   `nwcs_corpus_block_sensitivity_v1`; current remote scripts only consume and
   inline-validate those artifact shapes/hashes.
5. Current component-sensitivity producer state remains diagnostic/proxy unless
   separately upgraded to official CUDA component response evidence with
   promotion-grade response curves and stability gates.
6. No J-NWC/J-NWCS build-only or CUDA exact-eval dispatch was run in this
   verification pass, so there is no new archive custody packet from Worker D.
7. `J-NWCS-EC` build-only still performs CUDA EC correction search before its
   build-only stop; use plain `J-NWCS` for the cheapest sensitivity/archive
   build-only custody check.

Focused verification run by Worker D:

```bash
.venv/bin/python -m py_compile \
  src/tac/neural_weight_corpus.py \
  src/tac/neural_weight_codec_sensitivity.py \
  experiments/train_neural_weight_codec.py \
  src/tac/tests/test_neural_weight_codec_corpus.py \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py \
  src/tac/tests/test_neural_weight_codec_sensitivity.py \
  src/tac/tests/test_neural_weight_codec_sensitivity_renderer_format.py

bash -n \
  scripts/remote_lane_j_nwc_neural_weight_compression.sh \
  scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh \
  scripts/remote_lane_j_nwcs_ec_stack.sh

git diff --check -- \
  src/tac/neural_weight_corpus.py \
  src/tac/neural_weight_codec_sensitivity.py \
  experiments/train_neural_weight_codec.py \
  scripts/remote_lane_j_nwc_neural_weight_compression.sh \
  scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh \
  scripts/remote_lane_j_nwcs_ec_stack.sh \
  src/tac/tests/test_neural_weight_codec_corpus.py \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py \
  src/tac/tests/test_neural_weight_codec_sensitivity.py \
  src/tac/tests/test_neural_weight_codec_sensitivity_renderer_format.py

.venv/bin/python -m pytest \
  src/tac/tests/test_neural_weight_codec_corpus.py \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py \
  src/tac/tests/test_neural_weight_codec_sensitivity.py \
  src/tac/tests/test_neural_weight_codec_sensitivity_renderer_format.py \
  -q

.venv/bin/python -m pytest \
  src/tac/tests/test_neural_weight_codec.py \
  src/tac/tests/test_lane_j_nwc.py \
  -q
```

Observed:

```text
py_compile: passed
bash -n: passed
git diff --check: passed
42 passed in 1.31s
18 passed in 2.59s
```

Files changed by Worker D:

- `.omx/research/j_nwc_j_nwcs_manifest_fake_sensitivity_hardening_20260430_codex.md`

## Worker C Addendum - NWCS Sensitivity Builders - 2026-04-30

Scope: read-only inspection of `src/tac/neural_weight_corpus.py`,
`src/tac/neural_weight_codec_sensitivity.py`,
`scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh`, and
`scripts/remote_lane_j_nwcs_ec_stack.sh`, plus this ledger update. No source
code, launcher, archive, CUDA eval, score claim, promotion, ranking, or
method-retirement claim is made here. R5 has failed SegNet component gates and
no local promotable `component_sensitivity_v1` exists, so this is an unblock
spec only.

### Smallest Builder Surface

One checked-in builder is sufficient:

- `experiments/build_nwcs_sensitivity_inputs.py`

It should emit both remote-script inputs in one pass:

- `ANCHOR_SENSITIVITY_PT`
- `CORPUS_SENSITIVITY_PT`

Two separate builders are not required by the current consumers. Both outputs
share the same required custody: validated `component_sensitivity_v1` SHA,
anchor archive SHA, extracted anchor renderer SHA, block size 16, and the
exact corpus manifest hash. A single builder also prevents anchor/corpus
sensitivity transfer rules from drifting.

The builder must not run a scorer, train a codec, build an archive, or call
`contest_auth_eval.py`. Its job is only to validate existing CUDA component
sensitivity custody, project it into the NWCS block layout, and fail closed
when coverage is incomplete.

### Remote Consumer Contract

Both J-NWCS launchers already consume the same artifact shapes:

1. Promotion-eligible runs require `COMPONENT_SENSITIVITY_MANIFEST`.
2. The manifest is loaded and passed through
   `validate_component_sensitivity_manifest(..., promotion=True)`.
3. `ANCHOR_SENSITIVITY_PT` must load as:

   ```python
   {
       "format": "nwcs_anchor_parameter_sensitivity_v1",
       "sensitivities": {
           "<parameter_name>": torch.Tensor[n_blocks]
       },
       "metadata": {
           "device": "cuda",
           "component_sensitivity_manifest_sha256": "<sha256>",
           "anchor_archive_sha256": "<sha256>",
           "anchor_renderer_sha256": "<sha256>",
           "block_size": 16,
           "parameters": {
               "<parameter_name>": {
                   "shape": [int, ...],
                   "dtype": "torch.float32",
                   "numel": int,
                   "block_count": int,
                   "source": "component_sensitivity_v1:component_maps.combined",
                   "source_kind": "direct_block|per_element_block_mean",
                   "posenet_source": "component_sensitivity_v1:component_maps.posenet",
                   "segnet_source": "component_sensitivity_v1:component_maps.segnet",
                   "combined_source": "component_sensitivity_v1:component_maps.combined"
               }
           },
           "coverage": {
               "required_parameter_count": int,
               "covered_parameter_count": int,
               "fallback_parameter_count": 0,
               "required_block_count": int,
               "covered_block_count": int
           },
           "reduction": "combined_component_per_parameter_block_mean"
       }
   }
   ```

   Current launchers ignore the top-level `format`, `device`, `coverage`, and
   source fields, but they should be present for custody. They do enforce:
   nonempty tensor dict, finite nonnegative values, positive scorer signal,
   matching `component_sensitivity_manifest_sha256`, matching anchor archive
   SHA, matching extracted renderer SHA, `block_size == 16`, model parameter
   membership, exact parameter shape, exact block count, and exact tensor
   length.

4. `CORPUS_SENSITIVITY_PT` must load as:

   ```python
   {
       "format": "nwcs_corpus_block_sensitivity_v1",
       "values": torch.Tensor[corpus_manifest["totals"]["selected_blocks"]],
       "metadata": {
           "device": "cuda",
           "component_sensitivity_manifest_sha256": "<sha256>",
           "corpus_manifest_sha256": "<sha256>",
           "block_size": 16,
           "num_blocks": int,
           "ordering": "tac.neural_weight_corpus.v1 selected tensor order",
           "transfer_rule": "exact_parameter_name_shape_block_transfer_from_anchor",
           "coverage": {
               "selected_files": int,
               "selected_tensors": int,
               "selected_blocks": int,
               "matched_tensors": int,
               "unmatched_tensors": 0,
               "unmatched_blocks": 0
           },
           "tensor_sources": [
               {
                   "relative_path": str,
                   "tensor_name": str,
                   "corpus_block_start": int,
                   "corpus_block_end": int,
                   "source_parameter": str,
                   "source_block_start": int,
                   "source_block_end": int
               }
           ]
       }
   }
   ```

   Current launchers accept `values`, `sensitivities`, or
   `corpus_sensitivities`; use `values` for the new builder. They enforce:
   1-D tensor after flattening, metadata in promotion mode, matching component
   manifest SHA, matching corpus manifest SHA, `block_size == 16`,
   `num_blocks == replayed corpus blocks`, finite nonnegative values, and
   positive scorer signal.

### Required Builder Behavior

The builder should perform these exact checks and transformations:

1. Load `component_sensitivity_v1.json`; validate with
   `validate_component_sensitivity_manifest(manifest, promotion=True)`; compute
   its SHA-256 over the exact JSON bytes.
2. Zip-slip-safely extract or read `renderer.bin` from
   `ANCHOR_LANE_G_V3_ARCHIVE`; compute archive SHA-256 and extracted renderer
   SHA-256; load the model through `load_any_renderer_checkpoint`.
3. Load the component map source from the validated manifest. For promotion,
   the NWCS source must provide direct per-parameter block or per-element
   parameter sensitivity for every floating model parameter with
   `numel // 16 > 0`. A Conv2d-only per-channel OWV3 map is insufficient for
   the current anchor because Stage 5 encodes embeddings, GroupNorm weights and
   biases, Conv2d biases, ConvTranspose2d, Linear FiLM weights/biases, and
   motion-network parameters too.
4. Convert each covered anchor parameter to a `float32` CPU vector of length
   `param.numel() // 16` by mean aggregation over each contiguous 16-value
   block. Reject missing, non-finite, negative, empty, or all-zero promotion
   signal.
5. Build or load the deterministic corpus manifest using the same selection
   settings as the remote script: `block_size=16`, `max_files=200`,
   `max_blocks_per_ckpt=50000`, `min_checkpoint_bytes=1024`, stable path sort,
   and stable tensor-name sort.
6. For every selected manifest tensor, copy sensitivity from the exact matching
   anchor parameter name and shape, respecting `used_block_count`,
   `corpus_block_start`, and `corpus_block_end`. Default promotion behavior must
   fail on any unmatched tensor or block. A heterogeneous-corpus transfer rule
   can be added later, but it needs separate review and cannot be hidden behind
   a uniform fallback.
7. Write both `.pt` files with `torch.save` and immediately reload them through
   the same validation logic used by the launchers.

### Command Shape

Build the sensitivity inputs after a promotable component manifest exists:

```bash
EVID=experiments/results/nwcs_validated_20260430
ANCHOR_ARCHIVE=experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip
CORPUS_DIR=experiments/results/nwcs_anchor_compatible_corpus

.venv/bin/python experiments/build_nwcs_sensitivity_inputs.py \
  --component-sensitivity-manifest "$EVID/component_sensitivity_v1.json" \
  --anchor-archive "$ANCHOR_ARCHIVE" \
  --corpus-dir "$CORPUS_DIR" \
  --corpus-manifest-out "$EVID/nwcs/corpus_manifest.json" \
  --anchor-output "$EVID/nwcs/anchor_sensitivity.pt" \
  --corpus-output "$EVID/nwcs/corpus_sensitivity.pt" \
  --block-size 16 \
  --max-corpus-files 200 \
  --max-blocks-per-ckpt 50000 \
  --min-checkpoint-bytes 1024 \
  --transfer-rule exact_parameter_name_shape_block_transfer_from_anchor \
  --fail-on-unmatched-corpus-block
```

Validate the outputs before spending a build-only run:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import json
import torch

from tac.component_sensitivity_artifact import validate_component_sensitivity_manifest

evid = Path("experiments/results/nwcs_validated_20260430")
manifest = evid / "component_sensitivity_v1.json"
anchor = evid / "nwcs" / "anchor_sensitivity.pt"
corpus = evid / "nwcs" / "corpus_sensitivity.pt"

validate_component_sensitivity_manifest(json.loads(manifest.read_text()), promotion=True)
a = torch.load(anchor, map_location="cpu", weights_only=False)
c = torch.load(corpus, map_location="cpu", weights_only=False)
assert a["format"] == "nwcs_anchor_parameter_sensitivity_v1"
assert c["format"] == "nwcs_corpus_block_sensitivity_v1"
assert a["metadata"]["component_sensitivity_manifest_sha256"] == c["metadata"]["component_sensitivity_manifest_sha256"]
assert c["values"].detach().cpu().float().reshape(-1).numel() == c["metadata"]["num_blocks"]
assert float(c["values"].detach().cpu().float().clamp_min(0).sum()) > 0.0
print("NWCS sensitivity inputs validate structurally")
PY
```

Run plain J-NWCS build-only first. Do not use J-NWCS-EC for the first unblock,
because it still performs CUDA EC correction search before its build-only stop:

```bash
AUTH_EVAL_DEVICE=cuda \
NWCS_ALLOW_DEBUG_SENSITIVITY=0 \
NWCS_BUILD_ONLY=1 \
COMPONENT_SENSITIVITY_MANIFEST="$EVID/component_sensitivity_v1.json" \
ANCHOR_LANE_G_V3_ARCHIVE="$ANCHOR_ARCHIVE" \
ANCHOR_CORPUS_DIR="$CORPUS_DIR" \
ANCHOR_SENSITIVITY_PT="$EVID/nwcs/anchor_sensitivity.pt" \
CORPUS_SENSITIVITY_PT="$EVID/nwcs/corpus_sensitivity.pt" \
LOG_DIR="$EVID/nwcs_build_only" \
bash scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh
```

Exact CUDA eval is a later step only after build-only custody proves the
archive is deterministic, `renderer.bin` is `NWCS1`, no tensor-only fallback
occurred, the regenerated remote `corpus_manifest.json` SHA matches
`CORPUS_SENSITIVITY_PT`, and component gates are still justified:

```bash
AUTH_EVAL_DEVICE=cuda \
NWCS_ALLOW_DEBUG_SENSITIVITY=0 \
NWCS_BUILD_ONLY=0 \
COMPONENT_SENSITIVITY_MANIFEST="$EVID/component_sensitivity_v1.json" \
ANCHOR_LANE_G_V3_ARCHIVE="$ANCHOR_ARCHIVE" \
ANCHOR_CORPUS_DIR="$CORPUS_DIR" \
ANCHOR_SENSITIVITY_PT="$EVID/nwcs/anchor_sensitivity.pt" \
CORPUS_SENSITIVITY_PT="$EVID/nwcs/corpus_sensitivity.pt" \
LOG_DIR="$EVID/nwcs_exact_cuda" \
bash scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh
```

### Current Blockers

1. No promotable `component_sensitivity_v1.json` exists locally.
2. OWV3 R5 failed SegNet component gates, so R5 artifacts must not be used as
   NWCS promotion evidence.
3. The currently planned component maps are Conv2d per-channel maps. That is
   not enough for `ANCHOR_SENSITIVITY_PT` because the NWCS exporter requires
   sensitivity for every floating anchor parameter with at least one 16-value
   block.
4. No checked-in builder currently emits
   `nwcs_anchor_parameter_sensitivity_v1` or
   `nwcs_corpus_block_sensitivity_v1`.
5. The remote scripts regenerate `corpus_manifest.json` internally and do not
   accept a prebuilt manifest path. The new builder can still work by using the
   exact same corpus directory and selection settings, but any corpus drift will
   make the remote `corpus_manifest_sha256` check fail. A future source patch to
   accept a read-only prebuilt corpus manifest would reduce this risk.
6. `ANCHOR_CORPUS_DIR=experiments/results` is likely heterogeneous. Promotion
   should use a curated anchor-compatible corpus directory or fail on unmatched
   tensor names/shapes; uniform or random corpus sensitivity is debug-only.
7. No CUDA build-only or exact-eval run was performed in this Worker C pass.
