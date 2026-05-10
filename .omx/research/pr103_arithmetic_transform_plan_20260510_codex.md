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

Interpretation: this is a real, reproducible, byte-different archive candidate,
not a no-op. It is still not score-promotable because public PR103
`inflate.py` hard-codes `HIST_LEN=895`; the candidate requires a runtime
adapter with `HIST_LEN=887` and the same section order. Current blockers:
`candidate_runtime_adapter_missing`, `candidate_inflate_output_parity_missing`,
`strict_pre_submission_compliance_json_missing`, `lane_dispatch_claim_missing`,
and `exact_cuda_auth_eval_missing`.

## Adversarial classification

This is not a score candidate. It is the next byte-closed planning artifact
after the PR103 schema refresh and PacketIR certifier. It explicitly refuses
archive preflight and exact dispatch until a future runtime adapter consumes a
changed `ac_histograms_brotli` plus changed
`merged_range_coded_weights_and_hi_latents`, proves symbol roundtrip, records
old/new archive SHA-256s, and survives strict submission compliance plus exact
CUDA.

Current blockers:

- `candidate_archive_missing`
- `candidate_runtime_adapter_missing`
- `candidate_symbol_roundtrip_proof_missing`
- `candidate_inflate_output_parity_missing`
- `strict_pre_submission_compliance_json_missing`
- `lane_dispatch_claim_missing`
- `exact_cuda_auth_eval_missing`

## Next implementation target

Implement a runtime-adapter prototype for PR103 AC-section changes:

1. decode the source merged AC stream;
2. accept explicit q8 histogram coordinate changes, not just source-rule
   reconstruction;
3. re-encode the merged stream and compressed histogram section;
4. reject no-op or wrong-stream proposals by symbol SHA and stream count;
5. account for runtime-adapter overhead before archive materialization;
6. emit a byte-different candidate archive only if the runtime adapter can
   parse the new section lengths and consume the changed stream.

No GPU dispatch is warranted until that byte-different archive exists and
passes runtime-consumption and strict compliance gates.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_pr103_arithmetic_transform_plan.py \
  tests/test_plan_pr103_arithmetic_transform_cli.py -q
# 14 passed
```
