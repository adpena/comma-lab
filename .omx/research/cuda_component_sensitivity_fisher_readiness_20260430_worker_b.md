# CUDA Component Sensitivity/Fisher Readiness - Worker B - 2026-04-30

Scope: CUDA authoritative component sensitivity/Fisher map readiness. This is
a progress/readiness note, not a score ledger. No remote or spendful job was
launched and no score claim is made here.

## Inspected

- `experiments/profile_component_sensitivity.py`
- `experiments/build_component_sensitivity_manifest.py`
- `src/tac/component_sensitivity_artifact.py`
- `scripts/remote_lane_g_v3_owv3_fisher_stack.sh`
- Lightning exact-eval tooling:
  - `scripts/lightning_exact_eval_repro.py`
  - `scripts/launch_lightning_batch_job.py`
  - `src/tac/deploy/lightning/batch_jobs.py`
- Focused tests for profile, manifest, OWV3 Fisher stack, and Lightning batch
  custody.

## Readiness Decision

Promotable `component_sensitivity_v1` remains blocked. The current producer is
CUDA-ready for a clearly diagnostic map, but its maps are empirical Fisher
proxies and are intentionally written with `promotion_eligible=false`,
diagnostic evidence grades, and promotion blockers. The CLI also rejects
`--manifest-output` for these outputs.

Shortest contest-compliant path today is therefore:

1. Produce a CUDA diagnostic component/Fisher map packet with full 600 absolute
   pair IDs and no manifest promotion.
2. Use that map only for byte-plausible archive construction or design.
3. Promote only after a future official finite-difference component-map
   producer emits CUDA PoseNet, SegNet, and combined maps whose response gates
   pass and whose exact archive custody is attached.

## Patch Landed

Changed files:

- `src/tac/component_sensitivity_artifact.py`
- `src/tac/tests/test_component_sensitivity_artifact.py`
- `src/tac/tests/test_build_component_sensitivity_manifest.py`

Narrow hardening:

- Promotion validation now requires the sample plan to cover exactly 600 unique
  absolute contest `pair_index` values, `0..599`.
- Promotion validation now rejects subset-relative or partial plans by checking
  calibration plus holdout coverage, duplicate pair IDs, and `t/t1` consistency
  with absolute pair IDs.
- Manifest builder tests now exercise the default 600-pair generated plan
  instead of a 10-pair promotable fixture.

This closes a promotion-readiness gap where a top-k or partial sample plan could
previously satisfy the schema if all other custody fields were present.

## Ready Diagnostic CUDA Command

Run only on a CUDA-visible, supply-chain-clean runner:

```bash
RUN_ID=component_sensitivity_diag_20260430_worker_b_r1
EVID="experiments/results/${RUN_ID}"

.venv/bin/python scripts/scan_lightning_supply_chain.py --strict \
  --json-out ".omx/state/lightning_supply_chain_scan_${RUN_ID}.json"

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
  --device cuda
```

Expected evidence class: diagnostic CUDA Fisher/component-response packet, not
promotable `component_sensitivity_v1`.

## Ready Existing OWV3 R5 Exact-Eval Command

If the intent is to exact-eval the current byte-plausible OWV3 R5 diagnostic
archive, the local direct CUDA command is:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/lane_g_v3_owv3_r5_candidate_sweep_20260430_worker2/r5_selected_for_eval/archive_lane_g_v3_owv3.zip \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir upstream \
  --device cuda \
  --keep-work-dir \
  --work-dir experiments/results/lane_g_v3_owv3_r5_candidate_sweep_20260430_worker2/cuda_auth_eval_work
```

Archive identity to preserve:

```text
bytes = 686468
sha256 = 16ab95220c8add11b0bc40fb632bc8421f8bb8ad1cfba145f0b6058075237518
```

Promotion still requires paired same-run PFP16 component reference and
adjudication gates, as recorded in
`docs/runbooks/owv3_r5_exact_eval_queue.md`.

## Verification

Passed:

```bash
.venv/bin/python -m py_compile \
  src/tac/component_sensitivity_artifact.py \
  experiments/build_component_sensitivity_manifest.py \
  experiments/profile_component_sensitivity.py \
  src/tac/tests/test_component_sensitivity_artifact.py \
  src/tac/tests/test_build_component_sensitivity_manifest.py

.venv/bin/python -m pytest \
  src/tac/tests/test_component_sensitivity_artifact.py \
  src/tac/tests/test_build_component_sensitivity_manifest.py \
  -q

.venv/bin/python -m pytest \
  src/tac/tests/test_profile_component_sensitivity.py \
  src/tac/tests/test_remote_lane_g_v3_owv3_fisher_stack_script.py \
  src/tac/tests/test_lightning_batch_jobs.py \
  src/tac/tests/test_lightning_exact_eval_repro.py \
  -q

bash -n scripts/remote_lane_g_v3_owv3_fisher_stack.sh
git diff --check
```

Results:

```text
37 passed in 1.48s
61 passed in 1.10s
```

## Worker B Current Review Addendum

Scope: reviewed the current component-sensitivity validator, manifest
assembler, CUDA/Fisher profiler, component-sensitivity tests, and adjacent
progress ledgers. No code files were edited in this review.

Current conclusion: there is no single correct command in the checked-in code
that creates an authoritative promotion-grade `component_sensitivity_v1`
artifact. The manifest validator and assembler are ready for such an artifact,
but `experiments/profile_component_sensitivity.py` still emits diagnostic
Fisher-proxy maps with `promotion_eligible=false` and promotion blockers, and
its CLI intentionally rejects `--manifest-output`.

The shortest correct route is:

1. Run the current profiler only as a CUDA diagnostic Fisher/response packet.
2. Patch or add a promotion-mode producer that writes PoseNet, SegNet, and
   combined maps with no diagnostic/proxy markers only after finite-difference
   response gates pass.
3. Attach exact CUDA anchor custody from `contest_auth_eval.py`.
4. Assemble the manifest with
   `experiments/build_component_sensitivity_manifest.py`.
5. Validate the manifest, then use it for OWV3 or NWCS/J-NWC-derived
   sensitivity consumers. Any archive built from it still needs its own exact
   CUDA auth eval.

### Required CUDA-Only Gates

- CUDA-visible runner, `torch.cuda.is_available() == True`, and no CPU/MPS
  override. All device fields in maps, curves, exact eval, and manifest must be
  `cuda` or `cuda:<index>`.
- Strict Lightning supply-chain scan is preserved under `.omx/state/`.
- Repo-owned MCP configs remain disabled and no live MCP helper processes are
  present before trusting the runner.
- Anchor archive custody uses exact
  `archive.zip -> inflate.sh -> upstream/evaluate.py` via
  `experiments/contest_auth_eval.py --device cuda`.
- `contest_auth_eval.json` has `provenance.device == "cuda"` and
  `n_samples == 600`.
- Sample plan covers exactly the absolute contest pair IDs `0..599`, with
  unique calibration plus holdout records and correct `t=2*pair_index`,
  `t1=2*pair_index+1`.
- Component maps include `posenet`, `segnet`, and `combined`, matching
  `scorer_target`, tensor metadata, finite nonnegative values, and no debug,
  smoke, proxy, random, or diagnostic markers.
- Response curves have `official_component_response=true`, `passed=true`,
  finite `gate_spec`, finite `holdout_error`, official readouts
  (`official_pose_mse`, `official_argmax_disagreement`,
  `official_component_formula`), and either symmetric eps coverage with
  `eps=0` plus matched `-eps/+eps` or explicit directional-action metadata.
- Stability has `passed=true`, finite thresholds, CV/rank/top-k fields for all
  three components, and thresholds at least as strict as the current profiler
  constants: `cv_max <= 0.35`, `spearman_min >= 0.30`,
  `pearson_min >= 0.0`, `top_decile_overlap_min >= 0.50`.

### Finite-Difference Response Validation

Promotion-mode producer requirements:

- Materialize `perturbation_basis_v1` before curves: atom IDs, deterministic
  ordering, pair split, epsilon units, sign convention, normalization, and
  input custody.
- For each validation atom/component, evaluate `eps=0`, matched symmetric
  points such as `-0.002,+0.002`, and preferably a second magnitude such as
  `-0.001,+0.001`; for one-sided codec actions, record explicit directional
  action metadata.
- Evaluate PoseNet and SegNet on CUDA using official component readouts.
  SegNet validation must use argmax disagreement, not CE-only proxy.
- Compute combined deltas from measured mean components:

```text
DeltaCombined(eps)
  = 100 * (seg_eps - seg_0)
  + sqrt(10 * pose_eps) - sqrt(10 * pose_0)
```

- Record predicted delta from the map, observed delta, absolute/relative error,
  rank correlation, top-k overlap, zero-repro error, pass thresholds, and
  `passed=true` only if every gate passes.
- If any gate fails, preserve the packet as diagnostic with nonempty
  `promotion_blockers`; do not assemble a promotion manifest from it.

### Exact Commands

Runner preflight:

```bash
RUN_ID=cuda_component_sensitivity_owv3_jnwc_20260430_r1
EVID="experiments/results/${RUN_ID}"
mkdir -p "$EVID/anchor_eval" "$EVID/profile"

.venv/bin/python scripts/scan_lightning_supply_chain.py --strict --quiet \
  --json-out ".omx/state/lightning_supply_chain_scan_${RUN_ID}.json"

.venv/bin/python - <<'PY'
from tac.preflight import check_no_active_mcp_server_config, check_no_live_mcp_processes
check_no_active_mcp_server_config(strict=True, verbose=True)
check_no_live_mcp_processes(strict=True, verbose=True)
PY

nvidia-smi | tee "$EVID/nvidia-smi.txt"
.venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
PY
```

Anchor exact CUDA custody:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir upstream \
  --device cuda \
  --keep-work-dir \
  --work-dir "$EVID/anchor_eval"
```

Current diagnostic producer run, not promotable and intentionally no
`--manifest-output`:

```bash
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
  --device cuda
```

After a promotion-mode finite-difference producer replaces the diagnostic map
and curve payloads at the same paths, assemble:

```bash
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

p = Path("experiments/results/cuda_component_sensitivity_owv3_jnwc_20260430_r1/component_sensitivity_v1.json")
validate_component_sensitivity_manifest(json.loads(p.read_text()), promotion=True)
print("component_sensitivity_v1 validation passed")
PY
```

OWV3 consumer command after validation:

```bash
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

NWCS/J-NWC-family consumer shape after validated per-block sensitivity builders
exist:

```bash
AUTH_EVAL_DEVICE=cuda \
NWCS_ALLOW_DEBUG_SENSITIVITY=0 \
NWCS_BUILD_ONLY=1 \
COMPONENT_SENSITIVITY_MANIFEST="$EVID/component_sensitivity_v1.json" \
ANCHOR_LANE_G_V3_ARCHIVE=experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \
ANCHOR_CORPUS_DIR=experiments/results \
ANCHOR_SENSITIVITY_PT="$EVID/nwcs/anchor_sensitivity.pt" \
CORPUS_SENSITIVITY_PT="$EVID/nwcs/corpus_sensitivity.pt" \
LOG_DIR="$EVID/nwcs_build_only" \
bash scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh
```

### Blockers And Proposed Code Changes

No code changes were made in this review. Proposed changes before an
authoritative artifact can exist:

- Add a promotion-mode producer or new script that emits finite-difference or
  explicitly validated proxy-to-official component maps without Fisher-proxy
  promotion blockers.
- In that producer, write `passed=true` only when response gate results pass;
  current diagnostic curve writer hardcodes `passed=false`.
- Ensure map files themselves do not contain `promotion_eligible=false`,
  `official_component_response=false`, `diagnostic_cuda_fisher_proxy`, or
  nonempty `promotion_blockers`; the manifest assembler correctly rejects those
  markers.
- Add or finish NWCS derived sensitivity builders for
  `ANCHOR_SENSITIVITY_PT` and `CORPUS_SENSITIVITY_PT` with
  `component_sensitivity_manifest_sha256`, anchor archive SHA, renderer SHA,
  corpus manifest SHA, block size, shapes, block counts, finite nonnegative
  values, and positive scorer signal.

### Verification For This Review

Passed locally:

```bash
.venv/bin/python -m py_compile \
  src/tac/component_sensitivity_artifact.py \
  experiments/build_component_sensitivity_manifest.py \
  experiments/profile_component_sensitivity.py

.venv/bin/python -m pytest \
  src/tac/tests/test_component_sensitivity_artifact.py \
  src/tac/tests/test_build_component_sensitivity_manifest.py \
  src/tac/tests/test_profile_component_sensitivity.py \
  -q
```

Result:

```text
58 passed in 1.97s
```

## 2026-04-30T23:10Z Codex Implementation Delta

Implemented the first promotion-mode producer step:

- `experiments/profile_component_sensitivity.py` now accepts
  `--promotion-finite-difference`.
- The mode requires `--device cuda`, `--all-pairs`, and a full 600-pair contest
  video before it will run.
- It computes per-Conv2d-channel official component response by symmetric
  signed-RMS finite differences, separately on calibration and holdout splits.
- Response curves can now be promotion-eligible only when CUDA official
  response, zero-reproduction, and prediction gates pass.
- Default Fisher-proxy mode remains diagnostic and still blocks
  `--manifest-output`.

Verification added by Codex:

- `src/tac/tests/test_profile_component_sensitivity.py`: finite-difference map
  unit coverage and promotion response-curve coverage.
- Combined sensitivity/Lightning focused suite: `101 passed in 2.24s`.

Remaining blocker:

- A real CUDA run still needs to be dispatched with the documented commands and
  then assembled through `experiments/build_component_sensitivity_manifest.py`.
  No `component_sensitivity_v1` promotion artifact exists yet.

## 2026-04-30 Worker B Promotion-Finite-Difference Audit Addendum

Scope: audited `experiments/profile_component_sensitivity.py`,
`src/tac/tests/test_profile_component_sensitivity.py`,
`experiments/build_component_sensitivity_manifest.py`, and
`src/tac/component_sensitivity_artifact.py`. No production code and no tests
were edited for this addendum.

Current decision: do not launch a real promotion CUDA run yet. The new
`--promotion-finite-difference` path is directionally closer than the old
Fisher-proxy packet, but it can still produce or assemble artifacts that look
promotion-grade without enough mathematical and custody guards.

### Mathematical Coherence

Coherent pieces:

- PoseNet and SegNet component readouts in
  `_official_component_distortions_from_pairs()` use PoseNet MSE and SegNet
  argmax disagreement, not CE-only proxy.
- `component_formula_from_mean_distortions()` uses the contest-level
  `100 * mean_seg + sqrt(10 * mean_pose)` form rather than averaging per-pair
  square-root pose terms.
- Finite-difference maps are measured per Conv2d output channel on calibration
  and holdout splits, and response curves are measured on holdout channels
  selected from calibration maps.

Blocking gaps:

- The measured path is still a direct in-memory renderer path, not the canonical
  archive/inflate/evaluate output path. `_render_official_scorer_pairs()`
  scores float renderer output after direct resize to `(384, 512)`. The
  contest path upscales renderer output to camera size, rounds/clamps to uint8,
  writes raw/video output, then `upstream/evaluate.py` decodes and scorer-resizes.
  Small finite-difference perturbations can be visible in the direct float path
  while disappearing after uint8/camera-size roundtrip. A promotion producer
  needs either this exact roundtrip in the response measurement or a recorded
  proof/equivalence test that the shortcut is harmless at the chosen epsilon.
- Zero-signal curves can pass promotion gates. A throwaway repro wrote a CUDA
  promotion response curve with `selected_channels=[("conv.weight", 0, 0.0)]`
  and all observed deltas equal to zero. The writer emitted
  `promotion_eligible=true`, `passed=true`, `promotion_blockers=[]`,
  `holdout_error=0.0`. This would allow an all-zero sensitivity map to look
  successful instead of fail closed for missing scorer signal.
- Map payloads are stamped `promotion_eligible=true` and `evidence_grade="A"`
  before stability and response gates have proven pass. If a later gate fails,
  the response curve blocks manifest assembly, but the standalone map file can
  still be consumed by OWV3/NWCS paths that load only the map.

### Custody And Schema Blockers

- `build_component_sensitivity_manifest.py` does not validate tensor values
  inside map files. A throwaway manifest assembled successfully from three
  `.pt` maps whose only sensitivity tensor was `NaN`; the manifest retained
  only tensor metadata (`dtype`, `shape`, `numel`).
- The manifest builder does not compare the passed archive file SHA-256 against
  `contest_auth_eval.json["provenance"]["archive_sha256"]`. A throwaway
  manifest assembled successfully with a contest JSON claiming archive SHA
  `aaaaaaaa...` while the manifest archive SHA was
  `1e5f4f8c31d43d38613bfa5f200278be550370d1e5e4a0503112d22f7370d59d`.
- The builder rewrites a profile producer `sample_plan.split_hash` using a
  different hash payload than `profile_component_sensitivity.py`. Repro for
  `n_pairs=600`, `split_seed=20260430`, `holdout_fraction=0.2`:
  profile hash `f2b34c9657b49a8f45cc30d9d45b7ed06853ba9fad60348e3cd4ec45b00b57bc`;
  builder rewrite `2050558bb3fc181c11a1453769561c475615204a829b3f3df579c67eb5972ee3`.
  This breaks custody linkage with the producer's perturbation-basis split hash.
- Promotion mode checks `n_pairs == 600` via `n_frames // 2`, but does not
  require exactly `1200` frames. A 1201-frame video would still pass the
  600-pair guard while silently ignoring one frame.
- `--promotion-finite-difference` does not require response epsilons to include
  `0` and a matched `-eps/+eps` pair. The manifest validator can accept
  directional response metadata, but this producer's advertised mode is central
  finite difference; directional promotion should require an explicit separate
  mode and action contract.
- A++ is selectable in the manifest builder without enforcing T4/equivalent
  hardware, inflate budget proof, or adversarial review fields. Use `A` only
  until A++ evidence checks exist.

### Required Guards Before CUDA Spend

- Add producer-side fail-closed checks requiring positive finite scorer signal
  per component: nonzero map mass, nonzero selected-basis strength, and
  nontrivial observed holdout response above a reviewed floor. All-zero maps
  should remain diagnostic with promotion blockers.
- Apply the canonical uint8/camera-size roundtrip in finite-difference response
  measurement, or run a recorded equivalence gate comparing direct response to
  `inflate.sh -> upstream/evaluate.py` component deltas for representative
  perturbations.
- Validate loaded map tensors in the manifest builder: all tensors must be
  finite, nonnegative, nonempty, and have positive component signal before
  `component_sensitivity_v1` materialization.
- Cross-check contest eval custody: archive bytes/SHA in the JSON provenance
  must match the archive argument and manifest custody.
- Preserve the producer sample-plan hash or make both producer and builder use
  the same canonical split-hash payload; do not silently rewrite it.
- Promote maps only after response and stability gates pass, or write failed
  maps with nonempty promotion blockers.
- Require exact `n_frames == 1200`, `masks_frames == 1200`, and
  `poses.shape[0] == 600` for promotion mode.
- Require symmetric response epsilon coverage for this central-difference mode.

### Commands Run

```bash
.venv/bin/python -m py_compile \
  experiments/profile_component_sensitivity.py \
  src/tac/tests/test_profile_component_sensitivity.py \
  experiments/build_component_sensitivity_manifest.py \
  src/tac/component_sensitivity_artifact.py
```

Result: passed, no output.

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_profile_component_sensitivity.py \
  src/tac/tests/test_build_component_sensitivity_manifest.py \
  src/tac/tests/test_component_sensitivity_artifact.py \
  -q
```

Result:

```text
60 passed in 2.12s
```

Throwaway repros run with `.venv/bin/python - <<'PY'`:

- Zero-signal response-curve repro: emitted `promotion_eligible=true`,
  `passed=true`, `promotion_blockers=[]`.
- Split-hash mismatch repro: profile and builder hashes differed for the same
  600-pair split.
- NaN-map/mismatched-archive custody repro: manifest assembly did not reject
  NaN sensitivity tensors and did not reject contest JSON archive SHA mismatch.
