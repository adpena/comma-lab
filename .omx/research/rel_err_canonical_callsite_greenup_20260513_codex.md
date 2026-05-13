# rel_err canonical callsite greenup (2026-05-13)

## Summary

Fixed the four live `bugclass-b9-rel-err-canonical` violations reported by
release preflight. These were score-lowering build/probe tools that recomputed
relative error inline instead of routing through `tac.codec.rel_err`.

## Callsite decisions

- `tools/pr107_lossy_coarsening_apogee.py`: preserved the historical global
  L1 ratio semantics with `compute_rel_err(..., mode=RelErrForm.L1_RATIO)` and
  emitted `rel_err_form="l1_ratio"`.
- `tools/pr107_lossy_coarsening_brotli_optuna_stack.py`: same PR107 L1-ratio
  semantics, now routed through the canonical helper.
- `tools/build_a2_sensitivity_weighted_pr101_packet.py`: preserved the
  quantized-proxy L1 ratio over the concatenated decoder symbols and tagged
  the emitted form.
- `tools/probe_frame_conditional_quantization_disambiguator.py`: converted the
  frame-conditional probe to explicit per-frame `MAX_RATIO` via the canonical
  helper, because the probe arbitrates against hard-frame cliffs rather than a
  global RMS average.

## Verification

- `check_rel_err_definition_canonical(strict=True)`: OK.
- Python compile of all four modified tools: OK.
- CLI import smoke for all four modified tools via `--help`: OK.
- Developer preflight: `PREFLIGHT PASSED` in 7.849s wall.

## Score-lowering relevance

This does not create a score claim. It hardens the distortion math used by
PR107 lossy coarsening, A2 packet materialization, and frame-conditional
quantization probes so future byte candidates do not mix incompatible
relative-error definitions under one scalar.
