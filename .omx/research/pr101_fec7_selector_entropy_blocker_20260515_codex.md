# PR101 FEC7 selector entropy blocker - 2026-05-15

Tag: `research_only=true`

## Hypothesis

A byte-closed FEC7 selector encoding using PR103 range coding and PR84-style
adaptive contexts might replace the PR101 FEC6 fixed-Huffman selector and save
at least 79 charged archive bytes without changing decoded selector codes.

## Math

Contest rate term movement from pure byte savings is:

```text
delta_score_rate = -25 * saved_bytes / 37,545,489
```

For this task the dispatch gate is not score movement; it is the stricter byte
predicate:

```text
saved_bytes = fec6_selector_payload_bytes - fec7_selector_payload_bytes
target_saved_bytes >= 79
```

The real FEC6 selector payload is 249 bytes. Its global empirical entropy floor
from the 600-code histogram is 241 bytes, so a model-free global coder can save
at most 8 bytes before range-coder overhead:

```text
global_floor_saving = 249 - 241 = 8 bytes
```

## Provenance

| Element | Value |
|---|---|
| HEAD at run | `77a88f9f7694be09faa619eb3287fd5a76e327ac` |
| Source archive | `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip` |
| Source archive bytes | `178517` |
| Source archive sha256 | `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf` |
| FEC6 selector payload bytes | `249` |
| FEC6 selector payload sha256 | `fc5c431b5d793c33e2f320076fe6f0dd76c2d91e3826ae4b05abfb4f86f453ca` |
| Best FEC7 payload sha256 | `05d1a6102662e6dbdab93c9a08f6fca63a1d64cb295130d649a7054ddc501610` |
| Axis | no score axis; selector-byte prototype only |

## Reproduction

```bash
.venv/bin/python -m pytest src/tac/tests/test_pr101_fec7_selector_entropy.py -q

.venv/bin/python tools/profile_pr101_fec7_selector_entropy.py \
  --json-out /tmp/pr101_fec7_selector_entropy_profile.json \
  --md-out /tmp/pr101_fec7_selector_entropy_profile.md
```

Observed focused test result: `3 passed in 0.21s`.

## Results

All rows preserve `score_claim=false`, `dispatch_attempted=false`, and
`ready_for_exact_eval_dispatch=false`.

| Candidate | Payload bytes | Saving vs FEC6 | Charged model bytes | Verdict |
|---|---:|---:|---:|---|
| `fec7_global_pr103_range_u8_hist` | `268` | `-19` | `16` | best charged candidate; regresses bytes |
| `fec7_split_none_pr103_range` | `274` | `-25` | `19` | regresses bytes |
| `fec7_pairmod2_pr84_context_range` | `281` | `-32` | `32` | regresses bytes |
| `fec7_pairmod4_pr84_context_range` | `313` | `-64` | `64` | regresses bytes |
| `fec7_pairmod8_pr84_context_range` | `369` | `-120` | `128` | regresses bytes |
| `fec7_pairmod16_pr84_context_range` | `489` | `-240` | `256` | regresses bytes |
| `fec7_pairmod25_pr84_context_range` | `621` | `-372` | `400` | regresses bytes |
| `fec7_pairmod50_pr84_context_range` | `997` | `-748` | `800` | regresses bytes |
| `fec7_pairmod100_pr84_context_range` | `1757` | `-1508` | `1600` | regresses bytes |

The only lower bound that crosses 79 bytes is the non-byte-closed
`pairmod100` zero-model entropy floor: 148 bytes, or 101 bytes below FEC6. The
charged model for that context table is 1600 bytes, so the byte-closed payload
is 1757 bytes. Hardcoding that table in runtime would be source-embedded
selector payload data, not a compliant charged archive encoding.

## 6-Hook Wire-In

1. Sensitivity-map contribution: `N/A - selector-byte profiler only; no empirical score anchor`.
2. Pareto constraint: `N/A - no score claim and no promotion eligibility`.
3. Bit-allocator hook: candidate rows expose charged bytes and target-saving predicate.
4. Cathedral autopilot dispatch hook: `ready_for_exact_eval_dispatch=false`; no dispatch row.
5. Continual-learning posterior update: blocker row is durable here; no score posterior update.
6. Probe-disambiguator: `tools/profile_pr101_fec7_selector_entropy.py` compares global range, split range, and pairmod context range.

## Stop/Continue

Stop this selector-only byte path unless a new candidate payload is at least
79 bytes smaller than FEC6 after charging all selector model bytes. Continue
only with a component-changing selector/waterfill mechanism or a
compliance-reviewed runtime prior that is not source-embedded selector data.

## Explicit Blocker

`blocked=true`: FEC6 selector bytes are already near the global entropy floor;
tested byte-closed FEC7 range/adaptive prototypes charge their model bytes and
do not approach the required saving.

Reactivation criteria: reopen only with a selector model whose charged model
plus range stream is at least 79 bytes smaller than FEC6, or with a
compliance-reviewed runtime prior that is not source-embedded selector data.
