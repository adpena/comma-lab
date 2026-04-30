# NWCS Readiness Round 2 - 2026-04-30 Worker

Evidence grade: engineering hardening / non-score readiness review. No score
claim, ranking claim, promotion claim, or method-retirement claim is made here.
No `experiments/contest_auth_eval.py` command was run.

## Scope

Owned surface: J-NWC/J-NWCS artifacts, provenance, and focused tests.

Reviewed context:

- `.omx/research/j_nwc_j_nwcs_manifest_fake_sensitivity_hardening_20260430_codex.md`
- `.omx/research/nwcs_build_only_smoke_readiness_20260430_worker.md`
- `.omx/research/nwcs1_build_smoke_and_sensitivity_plan_20260430_agent.md`
- `.omx/research/component_sensitivity_owv3_nwcs_execution_plan_20260430_codex.md`
- `scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh`
- `scripts/remote_lane_j_nwcs_ec_stack.sh`
- `src/tac/neural_weight_codec_sensitivity.py`
- focused NWCS/component-sensitivity tests under `src/tac/tests/`

## Changes Landed

1. Promotion-eligible J-NWCS/J-NWCS-EC dispatch now requires a validated
   `component_sensitivity_v1` manifest before any auth-eval-capable path.
   - Added `COMPONENT_SENSITIVITY_MANIFEST`.
   - Promotion-eligible runs fail closed when the manifest path is missing.
   - The scripts call `validate_component_sensitivity_manifest(..., promotion=True)`.
   - `ANCHOR_SENSITIVITY_PT` and `CORPUS_SENSITIVITY_PT` must both cite the
     validated manifest via `component_sensitivity_manifest_sha256`.
   - Build-only and exact-run provenance now records the component sensitivity
     manifest in `artifact_custody`.

2. NWCS per-tensor blob decoding now rejects malformed payloads instead of
   silently accepting them.
   - Added bounds checks for scalar fields and payload slices.
   - Added trailing-byte rejection.
   - Added invalid bucket-id and code-index rejection.

3. Tests now pin the new fail-closed behavior.
   - Static remote-script guards require the component-manifest gate before
     `contest_auth_eval.py`.
   - Codec tests reject trailing bytes and invalid bucket IDs.

## Non-Score CUDA Dispatch Decision

A non-score CUDA dispatch plan is now mechanically feasible for build-only
J-NWCS readiness, but only after real sensitivity inputs exist:

- validated `component_sensitivity_v1.json`,
- `anchor_sensitivity.pt` with matching
  `component_sensitivity_manifest_sha256`, anchor archive SHA, renderer SHA,
  block size, parameter shapes, and block counts,
- `corpus_sensitivity.pt` with the same component manifest SHA, corpus manifest
  SHA, block size, `num_blocks`, finite nonnegative values, and positive scorer
  signal.

The current checked-in `experiments/profile_component_sensitivity.py` remains a
diagnostic Fisher-proxy producer with `promotion_eligible=false` and disabled
`--manifest-output`; it is not enough for validated NWCS provenance. Therefore
the next CUDA dispatch should be either:

- diagnostic-only component profiling, explicitly non-promotable, or
- build-only NWCS with externally produced/reviewed official component
  sensitivity artifacts.

Build-only dispatch shape, with auth eval skipped:

```bash
RUN_ID=nwcs_validated_build_only_20260430_r1
.venv/bin/python scripts/scan_lightning_supply_chain.py --strict \
  --json-out ".omx/state/lightning_supply_chain_scan_${RUN_ID}.json"

AUTH_EVAL_DEVICE=cuda \
NWCS_ALLOW_DEBUG_SENSITIVITY=0 \
NWCS_BUILD_ONLY=1 \
COMPONENT_SENSITIVITY_MANIFEST=experiments/results/nwcs_validated/component_sensitivity_v1.json \
ANCHOR_LANE_G_V3_ARCHIVE=experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \
ANCHOR_CORPUS_DIR=experiments/results \
ANCHOR_SENSITIVITY_PT=experiments/results/nwcs_validated/anchor_sensitivity.pt \
CORPUS_SENSITIVITY_PT=experiments/results/nwcs_validated/corpus_sensitivity.pt \
LOG_DIR=experiments/results/nwcs_validated_build_only_20260430_r1 \
bash scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh
```

Acceptance checks for that dispatch:

- `final_record.json`: `build_only=true`, `score_claim=false`,
  `promotion_eligible=false`, `result_json=null`.
- `provenance.json`: `auth_eval_skipped=true` and custody for archive,
  renderer, corpus manifest, component manifest, anchor sensitivity, corpus
  sensitivity, and codecs.
- Archive members are exactly `renderer.bin`, `masks.mkv`, `optimized_poses.pt`.
- `renderer.bin` has `NWCS1` magic, loads through the real architecture config,
  and does not fall back to `_nwcs_state_dict`.

Exact CUDA eval remains a separate gated step after build-only custody review.
This turn did not run it.

## Verification

Commands run:

```bash
.venv/bin/python -m py_compile \
  src/tac/neural_weight_codec_sensitivity.py \
  src/tac/tests/test_neural_weight_codec_sensitivity.py \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py

bash -n scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh
bash -n scripts/remote_lane_j_nwcs_ec_stack.sh
bash -n scripts/remote_lane_j_nwc_neural_weight_compression.sh

.venv/bin/python -m pytest \
  src/tac/tests/test_neural_weight_codec_sensitivity.py \
  src/tac/tests/test_neural_weight_codec_sensitivity_renderer_format.py \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py \
  src/tac/tests/test_neural_weight_codec_corpus.py \
  -q

.venv/bin/python -m pytest \
  src/tac/tests/test_component_sensitivity_artifact.py \
  src/tac/tests/test_build_component_sensitivity_manifest.py \
  -q

.venv/bin/python -m pytest \
  src/tac/tests/test_profile_component_sensitivity.py \
  -q

git diff --check -- \
  scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh \
  scripts/remote_lane_j_nwcs_ec_stack.sh \
  src/tac/neural_weight_codec_sensitivity.py \
  src/tac/tests/test_neural_weight_codec_sensitivity.py

awk '/[ \t]$/ { print FILENAME ":" FNR ": trailing whitespace"; bad=1 } END { exit bad }' \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py \
  .omx/research/nwcs_readiness_round2_20260430_worker.md
```

Observed:

```text
39 passed in 1.31s
36 passed in 1.32s
13 passed in 0.40s
diff/whitespace checks: passed with no output
```

## Changed Files

- `scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh`
- `scripts/remote_lane_j_nwcs_ec_stack.sh`
- `src/tac/neural_weight_codec_sensitivity.py`
- `src/tac/tests/test_neural_weight_codec_sensitivity.py`
- `src/tac/tests/test_remote_lane_j_nwc_hardening.py`
- `.omx/research/nwcs_readiness_round2_20260430_worker.md`

## Residual Gaps

- No promotable component-sensitivity artifact exists locally.
- No checked-in builder yet emits `nwcs_anchor_parameter_sensitivity_v1` or
  `nwcs_corpus_block_sensitivity_v1`.
- The current component profiler remains diagnostic Fisher-proxy evidence, not
  official finite-difference component response evidence.
- J-NWCS-EC build-only still performs CUDA EC correction search before its
  build-only stop; use J-NWCS alone for the cheapest provenance-only smoke.
