# PR91 HPM1 Probability Contract Worker - 2026-05-04

Scope: PR86/PR91 HPM1 probability-model and entropy-contract parity only.
No remote dispatch, no training, no scorer load, and no score claim were
performed.

## Inputs

- PR91 archive:
  `experiments/results/public_pr91_intake_20260504_codex/archive.zip`
- PR91 archive bytes: `222404`
- PR91 archive SHA-256:
  `4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f`
- PR91 HPM1 mask bytes: `145087`
- PR91 HPM1 mask SHA-256:
  `a4ed57ff0af1d8c914f004de165aeead50ec8dd61e99b0afdfbfa2d1e7fd9fcc`
- PR91 HPM1 token stream bytes: `116796`
- PR91 HPM1 token stream SHA-256:
  `541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b`
- PR91 HPM1 HPAC model bytes: `28243`
- PR91 HPM1 HPAC model SHA-256:
  `de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd`

## PR86 Relationship

The submitted PR91 `pr86_hpac.py` runtime is byte-identical to the merged PR86
`inflate.py` HPAC runtime:

- PR86 runtime:
  `experiments/results/public_pr86_intake_20260504_merged_refresh/inflate.py`
- PR91 runtime:
  `experiments/results/public_pr91_intake_20260504_codex/replay_submission/hpac_coder_hybrid/pr86_hpac.py`
- Shared bytes: `19657`
- Shared SHA-256:
  `f86f3067386928478d983817c9f9ee095ce6eb02aee8c0fbb7987cd0af1f9b01`

Static PR91-vs-PR86 HPAC payload comparison from
`compare_hpm1_to_pr86_hpac_contract(...)`:

- Relationship:
  `pr91_reuses_pr86_hpac_model_with_distinct_hpm1_token_stream`
- HPAC model equals PR86 `hpac.pt.ppmd`: `true`
- PR86 `tokens.bin` bytes: `113900`
- PR86 `tokens.bin` SHA-256:
  `14144bde496631f89a02646496bc2e66306bba6da149ddca37e21d85d175f225`
- PR91 token bytes minus PR86 token bytes: `+2896`
- PR91/PR86 token common prefix: `164` bytes / `41` uint32 words
- First token-stream mismatch: byte `164`, uint32 word `41`

This rules out PR91 bundle slicing, HPM1 header parsing, PPMd model extraction,
or a local PR91 runtime transcription error as the primary blocker. PR91 embeds
the same HPAC probability model and same decoder runtime class as PR86, but the
submitted HPM1 token stream is distinct.

## Probability Variant Matrix

Command:

```text
.venv/bin/python - <<'PY'
from tac.pr91_hpm1_codec import run_pr91_hpm1_preflight, DEFAULT_PR91_ARCHIVE
from tac.pr86_hpac_codec import supported_hpac_probability_variant_names
for variant in supported_hpac_probability_variant_names():
    report = run_pr91_hpm1_preflight(
        DEFAULT_PR91_ARCHIVE,
        max_frames=1,
        probability_variant=variant,
    )
    ctx = report.get("failure_context", {})
    print(
        variant,
        report["status"],
        report.get("failure_stage"),
        report.get("failure_reason"),
        ctx.get("failed_at"),
        ctx.get("decoded_symbol_count_before_failure"),
        report.get("elapsed_sec"),
    )
PY
```

Result: every variant failed closed before completing frame 0.

| Variant | Status | First failure | Decoded symbols before failure |
| --- | --- | --- | ---: |
| `source_float64_perfect_false` | `failed_closed` | frame 0 / group 10 / symbol 191 | `5951` |
| `source_float32_perfect_false` | `failed_closed` | frame 0 / group 24 / symbol 561 | `30513` |
| `source_float64_perfect_true` | `failed_closed` | frame 0 / group 15 / symbol 1534 | `13822` |
| `source_float32_perfect_true` | `failed_closed` | frame 0 / group 15 / symbol 191 | `12479` |

The source PR86 contract remains:
`constriction.stream.queue.RangeDecoder`, `Categorical(probabilities=..., perfect=False)`,
softmax probabilities converted to numpy `float64`, clip to `1e-7`, and
renormalize each row. Float32 and `perfect=True` are off-contract probes; they
extend the prefix in some cases but do not decode the submitted stream.

## Root Cause Classification

Fail-closed root cause:
`inherited_pr86_hpac_entropy_contract_unrecovered`.

Evidence:

- PR91's HPM1 header, token length, model length, and SHA fields parse cleanly.
- PR91's HPAC model PPMd blob is byte-identical to PR86's `hpac.pt.ppmd`.
- PR91 ships the PR86 HPAC decoder runtime byte-for-byte.
- The corrected `N=600` local prefix decode fails deterministically at frame 0,
  group 10, symbol 191 under the source PR86 contract.
- The same source contract is already known to fail closed on PR86's own
  submitted `tokens.bin` at the same frame/group/symbol coordinate.
- No tested precision, normalization, or constriction `perfect` mode variant
  decodes PR91's frame-0 stream.

Therefore PR91 HPM1 is not locally replayable at the probability-model or
entropy-contract level. The public PR91 score remains external/forensic until
a byte-exact PR86/PR91 constriction contract is recovered or a new token stream
is built with a locally proven encoder/decoder pair.

## Code Guard

Added PR91-side static custody comparison:

- `src/tac/pr91_hpm1_codec.py::compare_hpm1_to_pr86_hpac_contract`
- `run_pr91_hpm1_preflight(...)` now includes `pr86_hpac_relationship`
  before attempting entropy decode.

New PR91 test coverage:

- PR91 prefix decode still fails closed at the deterministic entropy mismatch.
- The preflight records that PR91 reuses the PR86 HPAC model with a distinct
  HPM1 token stream.
- The static comparison asserts the PR91-vs-PR86 token/model relationship:
  same HPAC model, distinct tokens, `+2896` token bytes, `164` common-prefix
  bytes, first mismatch at uint32 word `41`.

## Verification

```text
.venv/bin/python -m py_compile src/tac/pr91_hpm1_codec.py experiments/replay_pr91_hpm1_mask.py
.venv/bin/python -m pytest src/tac/tests/test_pr91_hpm1_codec.py -q
```

Results:

- Py compile: passed.
- PR91 focused pytest: `7 passed in 14.33s`.

## Replay Decision

PR91 replay must remain disabled. Re-enable only after one of these local,
fail-closed gates passes:

1. Full PR86/PR91 HPM1 decode under a recovered probability/entropy contract.
2. Byte-exact decode-to-reencode parity for the full submitted token stream.
3. A replacement HPM1 token stream built by a locally verified encoder and
   decoded by the submitted inflate runtime without scorer-side fallback.

Until then, PR91-derived archives are non-dispatchable and non-promotable.
