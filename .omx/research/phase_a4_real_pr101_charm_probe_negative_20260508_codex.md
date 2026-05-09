# Phase A4 real-PR101 ChARM probe — hand-parametric configs retired

## Scope

This closes the missing A4 real-substrate probe that the 50K toy ChARM run
could not answer. The probe applies the committed ChARM/range-coder backend to
the actual PR101 quantized HNeRV decoder state dict and keeps the result
planning-only: no runtime decoder, no archive substitution, no scorer load, no
CUDA/CPU score claim.

## Artifact

- Tool: `tools/pr101_charm_real_substrate_probe.py`
- Report: `reports/pr101_charm_real_substrate_probe.json`
- Input state dict:
  `experiments/results/cma_pr101_real_substrate_20260507T222605Z/pr101_decoder_state_dict.pt`
- Input SHA-256:
  `b863362aaba1b9cae9b944f5e5b1a43a53ca824b7899ed7b80a2e2146d66f053`
- Evidence grade: `empirical_planning`
- Evidence semantics: `cpu_real_pr101_charm_range_coder_probe`

## Result

PR101 brotli baseline: 178,144 B total archive reference.

| Model | Archive estimate | Delta vs brotli | Roundtrip |
|---|---:|---:|---|
| tensor Gaussian | 206,745 B | +28,601 B | exact |
| previous-symbol Gaussian | 219,555 B | +41,411 B | exact |
| zero-mean delta Gaussian | 220,224 B | +42,080 B | exact |

Best measured model is the static tensor Gaussian at 206,745 B. This closely
matches the earlier constriction QuantizedGaussian/Laplace real-substrate
negative (205,938 B), which means the result is not an artifact of the range
coder implementation. The PR101 weight stream is too close to brotli's static
Huffman/iid basin for these hand-parametric PMFs to win.

## Classification

Measured configuration retired, not family killed.

Retired configs:

- one-Gaussian-per-tensor ChARM range-coded PR101 symbols;
- previous-symbol Gaussian autoregressive PMF with one sigma per tensor;
- zero-mean first-difference Gaussian PMF with one sigma per tensor.

Still live:

- learned ChARM/hyperprior trained with the substrate from epoch 0;
- score-gradient co-training that changes the weight distribution itself;
- grammar/context coders whose model cost is amortized or generated from the
  runtime;
- PR106 or future non-PR101 substrates whose symbol distributions differ from
  the saturated PR101 basin.

## Reactivation criteria

1. Implement a runtime decoder consuming the exact ChARM packet grammar.
2. Produce an `archive.zip` whose `inflate.sh` consumes the changed bytes.
3. Prove byte closure with archive SHA-256, member SHA-256s, runtime tree SHA,
   and no-op control.
4. Run exact contest-CUDA and contest-CPU auth eval before promotion.

## Solver integration

The result is also threaded into:

- `reports/cathedral_autopilot_evidence.jsonl` as
  `phase_a4_charm_real_pr101_hand_parametric_probe`, with
  `score_claim=false`, `rank_or_kill_eligible=false`, and
  `ready_for_exact_eval_dispatch=false`.
- `reports/cathedral_autopilot_catalog_updated_20260508.json`
- `reports/cathedral_autopilot_plan_pr103_pr106_to_019_reconciled_20260508.json`
- `reports/cathedral_autopilot_plan_pr103_pr106_to_0155_reconciled_20260508.json`
- `reports/cathedral_meta_lagrangian_ranking_pr103_pr106_to_019_reconciled_20260508.json`
- `reports/cathedral_meta_lagrangian_ranking_pr103_pr106_to_0155_reconciled_20260508.json`
- `reports/phase_a_pareto_20260508.md`

This keeps hand-parametric ChARM in the validation/learning queue and blocks
accidental score promotion or dispatch reuse until a runtime-consumed packet
exists.

## Verification

```bash
.venv/bin/python -m py_compile tools/pr101_charm_real_substrate_probe.py
.venv/bin/python -m pytest src/tac/tests/test_pr101_entropy_floor_tools.py -q
.venv/bin/python tools/pr101_charm_real_substrate_probe.py \
  --state-dict-path experiments/results/cma_pr101_real_substrate_20260507T222605Z/pr101_decoder_state_dict.pt \
  --output reports/pr101_charm_real_substrate_probe.json
.venv/bin/python tools/cathedral_autopilot.py evidence-update \
  --prior-evidence reports/cathedral_autopilot_evidence.jsonl \
  --output reports/cathedral_autopilot_catalog_updated_20260508.json
.venv/bin/python tools/cathedral_autopilot.py plan \
  --label pr103_pr106_A++_reconciled \
  --d-seg 0.00067082 \
  --d-pose 3.36e-05 \
  --archive-bytes 185578 \
  --target-score 0.190 \
  --output reports/cathedral_autopilot_plan_pr103_pr106_to_019_reconciled_20260508.json \
  --prior-evidence reports/cathedral_autopilot_evidence.jsonl
.venv/bin/python tools/cathedral_autopilot.py plan \
  --label pr103_pr106_A++_reconciled \
  --d-seg 0.00067082 \
  --d-pose 3.36e-05 \
  --archive-bytes 185578 \
  --target-score 0.155 \
  --output reports/cathedral_autopilot_plan_pr103_pr106_to_0155_reconciled_20260508.json \
  --prior-evidence reports/cathedral_autopilot_evidence.jsonl
.venv/bin/python tools/cathedral_autopilot_meta_lagrangian_bridge.py \
  --plan-json reports/cathedral_autopilot_plan_pr103_pr106_to_019_reconciled_20260508.json \
  --output reports/cathedral_meta_lagrangian_ranking_pr103_pr106_to_019_reconciled_20260508.json
.venv/bin/python tools/cathedral_autopilot_meta_lagrangian_bridge.py \
  --plan-json reports/cathedral_autopilot_plan_pr103_pr106_to_0155_reconciled_20260508.json \
  --output reports/cathedral_meta_lagrangian_ranking_pr103_pr106_to_0155_reconciled_20260508.json
.venv/bin/python tools/phase_a_pareto_summary.py \
  --output reports/phase_a_pareto_20260508.md
```
