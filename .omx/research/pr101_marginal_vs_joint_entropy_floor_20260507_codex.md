# PR101 marginal-vs-joint entropy floor finding - 2026-05-07

## Scope

This note preserves the PR101 entropy-routing finding without promoting a
CPU-only bound into a score claim. All numbers below are byte/planning evidence
only. They do not imply an exact-evaluable score-affecting archive exists.

## Direct Empirical Evidence

Artifact: `reports/pr101_per_tensor_brotli_sweep.json`

- Tool: `tools/pr101_per_tensor_brotli_sweep.py`
- Input state dict:
  `experiments/results/cma_pr101_real_substrate_20260507T222605Z/pr101_decoder_state_dict.pt`
- Sweep: 28 tensors x 504 configs/tensor = 14,112 evaluations
  (`quality=4..11`, `lgwin=10..18`, `lgblock=16..22`)
- Wall clock: 12.037 seconds
- Raw quantized bytes: 228,958
- Sum of independently optimized per-tensor Brotli streams: 162,227 bytes
- Sum of independently encoded per-tensor default q11/lgwin22/lgblock18 streams:
  162,264 bytes
- Best per-tensor independent-stream gain vs that per-tensor default sum:
  37 bytes

Global PR101 decoder-byte references:

- Authored PR101 decoder blob: 162,164 bytes
  (`src/tac/pr101_split_brotli_codec.py`, public deconstruction ledgers)
- Best CodecOp setting found by the true CMA-ES run this tranche:
  162,154 bytes, `quality=11`, `lgwin=13`, `lgblock=18`
  (`experiments/results/cma_pr101_real_substrate_cmaes_20260507T223229Z/cma_pr101_search_report.json`)
- Best CodecOp setting reproduced by hardened Optuna known-best probe:
  162,150 bytes, `quality=11`, `lgwin=16`, `lgblock=19`
  (`experiments/results/optuna_pr101_known_best_probe_20260507_codex/optuna_search_report.json`)
- Hardened 60-trial Optuna sweep:
  60/60 valid tensor-complete roundtrips, best found in that seeded run:
  162,151 bytes, `quality=10`, `lgwin=16`, `lgblock=18`
  (`experiments/results/optuna_pr101_real_substrate_hardened_20260507_codex/optuna_search_report.json`)

Interpretation:

- Independent per-tensor Brotli is not a hidden win here.
- The per-tensor optimum is 63 bytes larger than the authored PR101 decoder blob
  and 77 bytes larger than the current Optuna known-best probe.
- Therefore, the remaining rate leverage is not better independent Brotli
  parameterization. It must come from one of:
  - lower-entropy weights,
  - smaller architecture,
  - mixed precision with distortion validation,
  - sparsity/training-side entropy collapse,
  - a real joint entropy coder that beats the global Brotli stream while
    preserving decode and contest compliance.

## Marginal Shannon Floor

Artifact: `reports/pr101_per_tensor_entropy.json`

- Empirical marginal entropy floor: 175,916 bytes
- Uniform floor: 216,109 bytes
- Weighted empirical bits/element: 5.584317153649183
- Weighted entropy ratio vs uniform: 0.7990512383852286
- Zero-distortion score at empirical marginal floor: 0.11713524359743989

This marginal floor is not contradicted by a 162,164-byte Brotli stream because
the marginal floor includes the report's fixed overhead model and is not the
same byte contract as the isolated PR101 decoder blob. Cross-contract floor
comparisons must explicitly state which bytes are included.

## Joint Entropy Claim Status

Subagent claim reported in chat:

- Joint entropy floor estimate: 148-162 KB
- Claimed gap vs marginal floor: 14-32 KB

Current evidence status:

- Evidence grade: `prediction/subagent-estimate` until the estimator artifact is
  located or regenerated.
- Local search in this tranche found the per-tensor Brotli sweep artifact but no
  canonical `joint_entropy` estimator artifact with the 148-162 KB band.
- The claim is plausible in direction because joint entropy can be below the sum
  of marginal entropies, but the magnitude must be tied to an estimator,
  context model, stream contract, and reproducible command before it can drive
  dispatch.

## Optuna Artifact Disposition

`experiments/results/optuna_pr101_real_substrate_20260507T230716Z` is
superseded planning smoke: it was successful, but it predates the hardened
report/ledger schema and tensor-coverage fields. Use the two hardened artifacts
above for routing and ledger promotion. Both remain CPU-prep only:

- `score_claim=false`
- `score_affecting_payload_changed=false`
- `charged_bits_changed=false`
- blocker includes `missing_exact_cuda_auth_eval`
- all valid rows reconstruct all 28 expected tensor keys

## Routing Decision

Use the direct artifact to deprioritize:

- more independent per-tensor Brotli sweeps over this already-covered grid,
- pure encoder-parameter searches on the unchanged PR101 decoder substrate,
- claims that the authored PR101 decoder has large generic Brotli slack.

Prioritize:

- joint/context entropy model with deterministic decode,
- mixed-precision floor and per-tensor sensitivity validation,
- architecture shrink and sparsity training loops,
- dispatch-advisor integration that keeps all floor rows marked planning-only
  until an archive candidate, distortion validation, and exact CUDA eval exist.
