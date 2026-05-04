# PR91 HPM1 Context Window Probe - 2026-05-04

Scope: PR91/HPM1 local entropy diagnostics only. No remote job, scorer load,
exact eval, or score claim was performed.

## Code And Artifact

- Added `run_pr91_hpm1_context_window_probe(...)` in
  `src/tac/pr91_hpm1_codec.py`.
- Added CLI path:
  `experiments/replay_pr91_hpm1_mask.py --context-window-probe`.
- Added focused tests in `src/tac/tests/test_pr91_hpm1_codec.py`.
- Diagnostic JSON:
  `experiments/results/public_pr91_intake_20260504_codex/diagnostics/pr91_hpm1_context_window_probe_20260504_codex.json`
- Diagnostic JSON SHA-256:
  `bb6a51412d31dcb19b03fdcf450e8b076b309a2546848f4640b30e257e377a72`

Command:

```text
.venv/bin/python experiments/replay_pr91_hpm1_mask.py \
  --archive experiments/results/public_pr91_intake_20260504_codex/archive.zip \
  --context-window-probe \
  --probability-variants source_float64_perfect_false,source_float32_perfect_false,source_float64_perfect_true,source_float32_perfect_true \
  --symbol-windows 33:8,5948:8 \
  --json-out experiments/results/public_pr91_intake_20260504_codex/diagnostics/pr91_hpm1_context_window_probe_20260504_codex.json
```

## Result

Status remains `failed_closed`; `dispatch=false`;
`pr91_ready_for_exact_eval=false`; `score_claim=false`.

Source contract (`source_float64_perfect_false`):

- decoded-context replay first diverges from corrected PR85 reference at
  `global_symbol=33`, `frame=0`, `group=0`, `symbol_in_group=33`,
  `pixel=(y=64,x=32)`, decoded `4`, reference `2`.
- decoded-context replay then fails at `global_symbol=5951`, `frame=0`,
  `group=10`, `symbol_in_group=191`, `pixel=(y=37,x=480)`.
- PR85 reference-context teacher forcing does not rescue the stream; it fails
  earlier at `global_symbol=4114`, `frame=0`, `group=8`,
  `symbol_in_group=274`.

Off-contract window probes:

- `source_float32_perfect_false`, `source_float64_perfect_true`, and
  `source_float32_perfect_true` complete the requested decoded-context windows.
- `source_float32_perfect_true` also completes the requested reference-context
  windows.
- These are diagnostic-only window results. They do not prove full-frame/full
  stream decode, byte-exact re-encode, or PR91 readiness.

Reference-context probability-only rows were recorded for both requested
windows, including `5948..5955`, because `RangeDecoder` cannot seek past an
already-diverged prefix. At `5948..5955`, the PR85 reference symbol remains
class `2` with near-unit probability under the teacher-forced reference state.
This is probability-state evidence only, not entropy-byte validity.

## Narrowed Blocker

The blocker is now:

```text
range_probability_numeric_contract_at_first_divergence
```

The next mutation should instrument or reproduce the encoder-side
RangeDecoder/Categorical state at `global_symbol=33`, then replay the submitted
stream with that exact numeric contract before extending back to the later
`5951` failure. PR85-reference teacher forcing alone is not sufficient because
the arithmetic state has already diverged by the first mismatch.

No PR91/HPM1 exact-eval dispatch is allowed from this artifact.
