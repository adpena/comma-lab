# Proxy Signal And Entropy Oracle Guard - 2026-05-08

Scope: hardening after recursive adversarial review of MPS/CPU proxy rows,
PR101/PR106 monolithic archive layout, and entropy/AAC tooling. Evidence grade:
engineering guard + focused tests. Score claim: false.

## Findings

- PR101 local custody is a single ZIP member `x`; PR106 local custody is a
  single ZIP member `0.bin`. The later PR106x repack uses `x`. Preflight now
  checks the exact family-to-member and grammar mapping, not only the set of
  names.
- PR101 has no separate pose or mask ZIP member. PR101 archive-backed
  CodecOp sweeps must name parser-proven sections: `decoder_blob`,
  `latent_blob`, or `sidecar_blob`.
- MPS/CPU QAT and curve sweeps are useful candidate-generation priors only.
  Local rel-error thresholds can mark `cuda_eval_worth_testing=true`, but they
  must not set `ready_for_exact_eval_dispatch=true`.
- Markov-1 entropy floors are oracle/planning lower bounds. The implemented
  Markov-1 AAC round-trip is byte-negative versus the PR101 brotli anchor in
  the current report, so the tool now records implemented bytes separately from
  the oracle floor.

## Guardrails Added

- `tools/all_lanes_preflight.py` validates PR101=`x` with
  `pr101_fixed_offset_hnerv_microcodec` and PR106=`0.bin` with
  `pr106_ff_packed_hnerv`.
- `tools/codec_op_param_sweep_manifest.py` rejects PR101 archive-backed
  sweeps unless `--baseline-substream-role` is a parser-proven PR101 section.
- `tools/pr101_lossy_int4_qat.py`, `tools/cathedral_autopilot.py`,
  `tools/parallel_dispatch_top_k.py`, and
  `src/tac/optimization/meta_lagrangian_ledger_adapter.py` now fail closed on
  MPS/CPU/proxy/research-signal evidence for promotion or dispatch.
- `tools/per_tensor_shannon_analysis.py` and
  `tools/pr101_markov1_aac_codec.py` emit explicit non-promotable dispatch
  blockers.

## Verification

```bash
.venv/bin/python -m ruff check \
  tools/pr101_lossy_int4_qat.py tools/cathedral_autopilot.py \
  tools/parallel_dispatch_top_k.py tools/codec_op_param_sweep_manifest.py \
  tools/pr101_markov1_aac_codec.py tools/per_tensor_shannon_analysis.py \
  src/tac/optimization/meta_lagrangian_ledger_adapter.py \
  src/tac/optimization/mps_research_signal.py \
  src/tac/tests/test_all_lanes_geometry_feedback_gate.py \
  src/tac/tests/test_codec_op_param_sweep_manifest.py \
  src/tac/tests/test_cathedral_autopilot_proxy_guards.py \
  src/tac/tests/test_dispatch_command_builder_shapes.py \
  src/tac/tests/test_meta_lagrangian_atom_ledger_adapter.py \
  src/tac/tests/test_mps_research_signal.py \
  src/tac/tests/test_pr101_entropy_floor_tools.py \
  src/tac/tests/test_pr101_lossy_int4_qat_dispatch_contract.py
# All checks passed.

.venv/bin/python -m pytest \
  src/tac/tests/test_all_lanes_geometry_feedback_gate.py \
  src/tac/tests/test_codec_op_param_sweep_manifest.py \
  src/tac/tests/test_cathedral_autopilot_proxy_guards.py \
  src/tac/tests/test_dispatch_command_builder_shapes.py \
  src/tac/tests/test_meta_lagrangian_atom_ledger_adapter.py \
  src/tac/tests/test_mps_research_signal.py \
  src/tac/tests/test_pr101_entropy_floor_tools.py \
  src/tac/tests/test_pr101_lossy_int4_qat_dispatch_contract.py -q
# 62 passed.
```

## Remaining Work

- Update older strategy memos that still phrase PR101/PR106 interventions as
  ZIP-member pose/mask substitutions.
- Add the same full non-promotable row contract to older CPU/MPS generator
  tools as they are touched.
- Build the next byte-closed candidate against the PR103-on-PR106 anchor using
  the monolithic nested decoder section and old/new SHA/byte proof before any
  Lightning dispatch.
