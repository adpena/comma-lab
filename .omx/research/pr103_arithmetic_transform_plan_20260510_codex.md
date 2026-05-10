# PR103 arithmetic transform planner (2026-05-10)

Generated: `2026-05-10`

`score_claim=false`; `dispatch_attempted=false`; `ready_for_archive_preflight=false`;
`ready_for_exact_eval_dispatch=false`.

## Landing

- Added `tac.pr103_arithmetic_transform_plan`.
- Added `tools/plan_pr103_arithmetic_transform.py`.
- Added focused module and CLI tests.

This is a local score-lowering planning layer only. It converts the PR103
`hnerv_lc_ac` schema refresh into one explicit arithmetic-stream transform
proposal while preserving the hard block between byte analysis and exact
dispatch.

## Artifact

```text
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/stem_weight_plan.json
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/stem_weight_plan.md
```

Command:

```bash
.venv/bin/python tools/plan_pr103_arithmetic_transform.py \
  --schema-manifest experiments/results/hnerv_pr103_lc_ac_schema_refresh_20260510_codex/manifest.json \
  --target-label stem.weight \
  --json-out .omx/research/pr103_arithmetic_transform_plans_20260510_codex/stem_weight_plan.json \
  --md-out .omx/research/pr103_arithmetic_transform_plans_20260510_codex/stem_weight_plan.md
```

## Result

- proposal: `pr103_ac_plan_a057fd21261e08dc`
- target stream: `stem.weight`
- decoded symbols: `48384`
- decoded symbol SHA-256:
  `a0c5e83fa8837d96bdb05b41129a3ae4baf869b47261be98200dc8d1e8996e88`
- model-gap byte upper bound: `46`
- rate-score upper-bound delta: `-3.0629511843619884e-05`

## Apples-to-apples correction: baseline retarget versus coordinate search

Artifact:

```text
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/stem_weight_retarget_probe.json
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/stem_weight_retarget_probe.md
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/*_coordinate_probe.json
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/*_coordinate_probe.md
```

Command:

```bash
.venv/bin/python tools/probe_pr103_arithmetic_retarget.py \
  --schema-manifest experiments/results/hnerv_pr103_lc_ac_schema_refresh_20260510_codex/manifest.json \
  --target-label stem.weight \
  --json-out .omx/research/pr103_arithmetic_transform_plans_20260510_codex/stem_weight_retarget_probe.json \
  --md-out .omx/research/pr103_arithmetic_transform_plans_20260510_codex/stem_weight_retarget_probe.md
```

That command was only a baseline reconstruction check. It rebuilds the q8 model
from the decoded symbols using the same rule PR103 already used, so a no-op is
not evidence that the local histogram neighborhood is exhausted. Treating it as
such would be an apples-to-oranges conclusion.

Corrected coordinate-search probes:

```bash
.venv/bin/python tools/probe_pr103_arithmetic_retarget.py \
  --schema-manifest experiments/results/hnerv_pr103_lc_ac_schema_refresh_20260510_codex/manifest.json \
  --target-label stem.weight \
  --probe-mode coordinate-search \
  --top-symbols 32 \
  --deltas=-2,-1,1,2 \
  --json-out .omx/research/pr103_arithmetic_transform_plans_20260510_codex/stem_weight_coordinate_probe.json \
  --md-out .omx/research/pr103_arithmetic_transform_plans_20260510_codex/stem_weight_coordinate_probe.md
```

Coordinate-search results across the top-5 targets:

| target | merged delta | histogram Brotli delta | total member delta | blocker |
|---|---:|---:|---:|---|
| `stem.weight` | `0` | `-2` | `-2` | `candidate_runtime_adapter_missing` |
| `blocks.1.weight` | `0` | `-1` | `-1` | `candidate_runtime_adapter_missing` |
| `blocks.0.weight` | `0` | `-2` | `-2` | `candidate_runtime_adapter_missing` |
| `blocks.2.weight` | `0` | `-2` | `-2` | `candidate_runtime_adapter_missing` |
| `blocks.3.weight` | `0` | `-2` | `-2` | `candidate_runtime_adapter_missing` |

Interpretation: the simple reconstruction path is a no-op, but actual q8
histogram coordinate perturbations do expose tiny byte-positive local moves.
They are far too small to justify exact CUDA alone and are not archive-ready
because PR103 has fixed section lengths and no runtime adapter for changed
sections. The valid next PR103 arithmetic work is a runtime-adapter prototype
or a larger multi-coordinate/multi-stream search that can amortize adapter
overhead and produce a material byte delta.

## Beam-search follow-up: composed coordinate moves

Artifact:

```text
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/*_beam_probe.json
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/*_beam_probe.md
```

Command shape:

```bash
.venv/bin/python tools/probe_pr103_arithmetic_retarget.py \
  --schema-manifest experiments/results/hnerv_pr103_lc_ac_schema_refresh_20260510_codex/manifest.json \
  --target-label stem.weight \
  --probe-mode beam-search \
  --top-symbols 16 \
  --deltas=-2,-1,1,2 \
  --rounds 3 \
  --beam-width 8 \
  --json-out .omx/research/pr103_arithmetic_transform_plans_20260510_codex/stem_weight_beam_probe.json \
  --md-out .omx/research/pr103_arithmetic_transform_plans_20260510_codex/stem_weight_beam_probe.md
```

Beam-search results across the same top-5 targets:

| target | evaluated candidates | merged delta | histogram Brotli delta | total member delta | changes | blocker |
|---|---:|---:|---:|---:|---:|---|
| `blocks.0.weight` | `968` | `0` | `-9` | `-9` | `3` | `candidate_runtime_adapter_missing` |
| `blocks.3.weight` | `979` | `0` | `-6` | `-6` | `3` | `candidate_runtime_adapter_missing` |
| `blocks.2.weight` | `954` | `0` | `-4` | `-4` | `2` | `candidate_runtime_adapter_missing` |
| `stem.weight` | `960` | `0` | `-3` | `-3` | `3` | `candidate_runtime_adapter_missing` |
| `blocks.1.weight` | `971` | `0` | `-2` | `-2` | `2` | `candidate_runtime_adapter_missing` |

Interpretation: composed q8 histogram moves improve the local byte signal from
single-digit `1-2` byte wins to a best observed `9` byte win on
`blocks.0.weight`, still entirely in the compressed histogram sideband. The
merged arithmetic stream length did not move in this small beam. This is
useful evidence that the previous no-op conclusion was too strong, but it is
not yet enough to justify exact CUDA or a contest packet without runtime
adapter overhead accounting. The next useful solver step is multi-stream beam
composition with adapter-overhead accounting, not another one-stream micro
probe.

## Materialized combined-beam candidate

Artifact:

```text
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/archive.zip
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/manifest.json
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/manifest.md
```

Command:

```bash
.venv/bin/python tools/materialize_pr103_arithmetic_histogram_candidate.py \
  --schema-manifest experiments/results/hnerv_pr103_lc_ac_schema_refresh_20260510_codex/manifest.json \
  --beam-probe-report .omx/research/pr103_arithmetic_transform_plans_20260510_codex/blocks_0_weight_beam_probe.json \
  --beam-probe-report .omx/research/pr103_arithmetic_transform_plans_20260510_codex/blocks_1_weight_beam_probe.json \
  --beam-probe-report .omx/research/pr103_arithmetic_transform_plans_20260510_codex/blocks_2_weight_beam_probe.json \
  --beam-probe-report .omx/research/pr103_arithmetic_transform_plans_20260510_codex/blocks_3_weight_beam_probe.json \
  --beam-probe-report .omx/research/pr103_arithmetic_transform_plans_20260510_codex/stem_weight_beam_probe.json \
  --output-archive .omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/archive.zip \
  --json-out .omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/manifest.json \
  --md-out .omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/manifest.md
```

Result:

| field | value |
|---|---:|
| source archive bytes | `178223` |
| candidate archive bytes | `178215` |
| archive byte delta | `-8` |
| payload byte delta | `-8` |
| changed section | `ac_histograms_brotli: 895 -> 887` |
| merged AC section | `153856 -> 153856` |

Custody:

- source archive SHA-256:
  `31881b2d23d027e6619f2d8df2fe35d4d207d08882ec673d6c1b7ff119f18c30`
- candidate archive SHA-256:
  `2427cbb7f68e8e3bcf1e989eee0cf511bff5994a5e856b500bfca3c95ca181d8`
- candidate merged AC SHA-256:
  `bd10827909bf34d746c21cc856f14e24e1a5a8130dde0438ceda213e1fb3c2bf`
- candidate roundtrip: `decoder_maybe_exhausted=true`,
  `reencoded_byte_identical=true`, `decoded_symbol_count=237561`
- semantic stream parity: `all_stream_symbol_sha_match=true` across `9`
  decoded streams

Interpretation: this is a real, reproducible, byte-different archive candidate,
not a no-op. It is still not score-promotable because public PR103
`inflate.py` hard-codes `HIST_LEN=895`; the candidate requires a runtime
adapter with `HIST_LEN=887` and the same section order. Current blockers:
`candidate_runtime_adapter_missing`, `candidate_inflate_output_parity_missing`,
`strict_pre_submission_compliance_json_missing`, `lane_dispatch_claim_missing`,
and `exact_cuda_auth_eval_missing`.

## Runtime adapter prototype

Artifact:

```text
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/runtime_adapter/
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/runtime_adapter_manifest.json
```

Command:

```bash
.venv/bin/python tools/build_pr103_lc_ac_runtime_adapter.py \
  --candidate-manifest .omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/manifest.json \
  --source-runtime-dir experiments/results/public_pr_intake_full/public_pr103_intake_20260505_auto/source/submissions/hnerv_lc_ac \
  --output-runtime-dir .omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/runtime_adapter \
  --json-out .omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/runtime_adapter_manifest.json \
  --force
```

Result:

| field | value |
|---|---:|
| runtime tree SHA-256 | `00d9e4550aa34a865414bff73029f0005f7684008fc5994c3cf5abaf951f775e` |
| changed runtime constant | `HIST_LEN: 895 -> 887` |
| shell closure patch | `python` -> `${PYTHON:-python}` |
| parsed candidate histogram bytes | `887` |
| parsed merged AC bytes | `153856` |
| state_dict tensors | `28` |
| state_dict params | `228958` |
| latents shape | `[600, 28]` |
| semantic stream parity | `true across 9 decoded streams` |
| decoder-state parity | `true; state_dict and latent SHA-256s match source runtime` |

Interpretation: the runtime-adapter blocker is cleared for local
decode-consumption. This still does **not** run full frame inflate, does not run
the scorer, and does not authorize dispatch. Decoder-state and latent parity
are necessary parser-consumption evidence only; they do **not** prove identical
HNeRV frame generation, scorer components, or CUDA numerics. Frame/eval parity
requires source-vs-adapter `inflate.sh` output byte parity or exact same-runtime
evals on both packets. Exact CUDA remains the only CUDA-axis score authority.

## Compliance packet

Artifact:

```text
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/packet/
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/packet_manifest.json
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/packet/pre_submission_compliance.json
```

Commands:

```bash
.venv/bin/python tools/build_pr103_lc_ac_candidate_packet.py \
  --runtime-adapter-manifest .omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/runtime_adapter_manifest.json \
  --packet-dir .omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/packet \
  --json-out .omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/packet_manifest.json \
  --force

.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir .omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/packet \
  --archive .omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/packet/archive.zip \
  --archive-manifest-json .omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/packet/archive_manifest.json \
  --expect-single-member x \
  --expected-archive-sha256 2427cbb7f68e8e3bcf1e989eee0cf511bff5994a5e856b500bfca3c95ca181d8 \
  --expected-archive-size-bytes 178215 \
  --json-out .omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/packet/pre_submission_compliance.json \
  --strict
```

Result:

| field | value |
|---|---:|
| packet archive SHA-256 | `2427cbb7f68e8e3bcf1e989eee0cf511bff5994a5e856b500bfca3c95ca181d8` |
| packet archive bytes | `178215` |
| packet runtime tree SHA-256 | `fec00c1db354fdde2929c3512906ea0a5ba7f81461704e8dba2804a36471dab3` |
| strict pre-submission compliance | `passed=true` |

Remaining blockers after local packaging/compliance:
`full_frame_inflate_output_parity_missing`, `lane_dispatch_claim_missing`, and
`exact_cuda_auth_eval_missing`.

## Exact CUDA dispatch and result

Lane claim:

- lane_id: `pr103_histogram_8b_packet_exact_cuda`
- instance/job_id: `pr103_histogram_8b_packet_exact_cuda_modal_20260510T221146Z`
- terminal status: `completed_contest_cuda_auth_eval_negative`

Command:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run experiments/modal_auth_eval.py \
  --archive .omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/packet/archive.zip \
  --submission-dir .omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/packet \
  --inflate-sh inflate.sh \
  --output-dir experiments/results/modal_auth_eval/pr103_histogram_8b_packet_exact_cuda_modal_20260510T221146Z \
  --gpu T4
```

Result:

| field | value |
|---|---:|
| evidence | `[contest-CUDA] Modal T4` |
| score_recomputed_from_components | `0.22777267708207616` |
| avg_posenet_dist | `0.00017199` |
| avg_segnet_dist | `0.00067635` |
| archive bytes | `178215` |
| source PR103 replay score | `0.2277649714224471` |
| score delta vs source PR103 | `+0.00000770565962906` |

Classification after follow-up apples-to-apples review:
`packet-specific exact-CUDA regression; method verdict indeterminate`. The
transform is byte-real and runtime-consumed by the rebuilt adapter packet, but
decoded state/latent parity is not full-frame output parity and the source
PR103 comparison did not run through the same rebuilt runtime contract. Do not
promote this packet. Preserve the runtime-adapter and packet-builder
infrastructure for stronger globally recomputed PR103 histogram optimizers.

## Adversarial classification

This is not a score candidate. It is the next byte-closed planning artifact
after the PR103 schema refresh and PacketIR certifier. It explicitly refuses
archive preflight and exact dispatch until full-frame inflate parity or
component-delta evidence exists, strict submission compliance passes, a lane
dispatch claim is opened, and exact CUDA returns.

Current blockers for interpreting this as a method verdict or dispatching a
successor:

- `full_frame_inflate_output_parity_missing`
- same-runtime source replay missing
- normal lane claim and exact CUDA gates for any successor packet

## Global-combo byte artifact

The next score-lowering artifact is materialized, but remains dispatch-blocked
pending the parity gate above:

| field | value |
|---|---:|
| selection mode | `global_combo_best` |
| source archive bytes | `178223` |
| candidate archive bytes | `178211` |
| archive byte delta | `-12` |
| payload byte delta | `-12` |
| source probe delta sum | `-13` |
| non-additivity delta | `1` |
| candidate archive SHA-256 | `578c8f4e86eafc9dc04eefe61cc0e7f3f3f43e134ef4447cf9ef26fd23a23551` |
| packet runtime tree SHA-256 | `bf43663559e88b89f1bc0a1fa14b5093b7195da64f5aa7ed1cac696cb60caa02` |
| full frame parity | `true` for all `600` pairs / `3,662,409,600` rendered bytes |

Artifacts:

```text
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_probe.json
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_candidate/archive.zip
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_candidate/packet/
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_candidate/frame_parity_probe_sampled.json
.omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_candidate/frame_parity_probe_full_cpu.json
```

The global-combo search is the first non-greedy recomputation over the full
merged AC stream plus the Brotli-compressed histogram sideband. It beats the
greedy materialized packet on rate (`-12B` vs `-8B`) while preserving decoded
state/latent parity. A full same-runtime CPU frame-digest probe also matches
source output bytes for all 600 pairs / 1,200 frames
(`074f834f14ba4611f9358bb0a3f8e729bb43e4ea673be23e2acf85e7448dd1e5`
over `3,662,409,600` rendered bytes). It is **not** a score claim and not
exact-eval dispatch authorization; the remaining blockers are a fresh lane
claim and exact CUDA on the packet.

The fresh exact CUDA pair has now landed and is positive under same-runtime
comparison:

- source PR103 Modal T4 same-runtime score: `0.22777817708207615`
- global-combo candidate Modal T4 score: `0.22777017708207614`
- delta: `-0.000008000000000008`, exactly the `-12B` rate movement with
  identical SegNet/PoseNet components

See
`.omx/research/pr103_global_combo_12b_same_runtime_cuda_positive_20260510_codex.md`.

## Next implementation target

The greedy `-8B` candidate completed the runtime-adapter, packet, compliance,
dispatch claim, and exact CUDA path, but its method verdict is indeterminate
until same-runtime source replay or full-frame output parity exists. Do not keep
promoting or widening the greedy per-stream-best packet.

The next implementation target is a non-arbitrary DP/Lagrangian optimizer over
histogram/range-code/Brotli sideband tradeoffs, with global recomputation of
the whole merged AC stream and full histogram sideband before any new CUDA
spend. The same-runtime source/candidate scorer harness comes first because it
determines whether tiny rate gains are real score-lowering moves or
runtime-comparison artifacts.

No GPU dispatch is warranted until that byte-different archive exists and
passes runtime-consumption and strict compliance gates.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_pr103_arithmetic_transform_plan.py \
  tests/test_plan_pr103_arithmetic_transform_cli.py -q
# 14 passed
```
