# Beta/Jacobian-Fisher Path-B CPU Candidate - 2026-05-08

Scope: CPU-only composition of beta-Fisher/Jacobian-Fisher importance
allocation with the Path-B no-dead-K lossy-coarsening builder. No GPU dispatch
was launched. This is byte-closed archive custody plus proxy rel_err only, not
score evidence.

## Inputs

- Result root:
  `experiments/results/beta_jacobian_fisher_path_b_cpu_20260508T115156Z/`
- State dict:
  `experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt`
- Importance source:
  `reports/raw/beta_fisher_lossy_coarsening_weights_20260508_codex_smoke/manifest.json`
- Generated curves:
  `experiments/results/beta_jacobian_fisher_path_b_cpu_20260508T115156Z/pr101_lossy_k_curves.json`
- Generated importance input:
  `experiments/results/beta_jacobian_fisher_path_b_cpu_20260508T115156Z/beta_fisher_importance_input.json`
- Generic allocator manifest:
  `experiments/results/beta_jacobian_fisher_path_b_cpu_20260508T115156Z/jacobian_fisher_allocation_manifest.json`

The first allocator handoff failed closed because a sorted JSON mapping emitted
`allocation.selected_by_tensor` in alphabetical tensor order while the builder
requires `FIXED_STATE_SCHEMA` order. The generated curves were rewritten as an
ordered list (`tensor_name` + `candidates`) and rerun. No repo code patch was
needed.

## Allocator Result

Command:

```text
.venv/bin/python tools/jacobian_fisher_importance_allocator.py \
  --curves-json experiments/results/beta_jacobian_fisher_path_b_cpu_20260508T115156Z/pr101_lossy_k_curves.json \
  --importance-json experiments/results/beta_jacobian_fisher_path_b_cpu_20260508T115156Z/beta_fisher_importance_input.json \
  --target-distortion 0.0386 \
  --output-json experiments/results/beta_jacobian_fisher_path_b_cpu_20260508T115156Z/jacobian_fisher_allocation_manifest.json
```

Output summary:

- Objective: `target_distortion`
- Allocator byte proxy: `130,900`
- Weighted RMS error: `0.036643`
- Selected Ks:
  `[10,24,3,10,3,10,5,10,2,1,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]`

The target sweep is saved at
`experiments/results/beta_jacobian_fisher_path_b_cpu_20260508T115156Z/allocator_target_sweep.json`.
The least aggressive byte-lowering row was selected because more aggressive
rows quickly raise aggregate rel_err.

## Byte-Closed Candidate

Builder command:

```text
.venv/bin/python tools/build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py \
  --selected-Ks-json /Users/adpena/Projects/pact/experiments/results/beta_jacobian_fisher_path_b_cpu_20260508T115156Z/jacobian_fisher_allocation_manifest.json \
  --selected-Ks-rms-target 0.0386 \
  --output-root /Users/adpena/Projects/pact/experiments/results/beta_jacobian_fisher_path_b_cpu_20260508T115156Z
```

Candidate artifact:

- Build dir:
  `experiments/results/beta_jacobian_fisher_path_b_cpu_20260508T115156Z/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T115513Z/`
- Archive:
  `experiments/results/beta_jacobian_fisher_path_b_cpu_20260508T115156Z/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T115513Z/archive.zip`
- Manifest:
  `experiments/results/beta_jacobian_fisher_path_b_cpu_20260508T115156Z/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T115513Z/build_manifest.json`
- Archive bytes: `147,285`
- Archive SHA-256:
  `55cbce00a67eb7334e31ddca6286c70b22bdb8a38e06408d4d00df912125f936`
- ZIP member: `x`, stored, `147,185` bytes
- Decoder section bytes: `131,191`
- Decoder brotli payload bytes: `131,131`
- K bytes in wire format: `0`
- CPU int8-symbol rel_err: `0.0580904313705469`
- CPU fp32 smoke rel_err vs quantized reference: `0.08739940096338861`
- Max per-tensor fp32 smoke rel_err: `0.1929084482955193`
- Latent pairs decoded in smoke: `600`

Git disposition: the ZIP archive itself is an ignored local custody artifact
under `experiments/results/**/*.zip`. The committed durable state is this
ledger plus the structured manifests, allocation inputs, and runtime source
needed to identify and rebuild the candidate. The archive SHA-256 and byte
count above are the custody identifiers for the ignored local ZIP.

Byte comparison:

- Versus default no-dead-K Path-B artifact
  `admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T064711Z`
  (`153,671` bytes): `-6,386` archive bytes, byte-term delta
  `-0.004252175274638186`.
- Versus prior beta-selected no-dead-K artifact (`159,576` bytes): `-12,291`
  archive bytes, byte-term delta `-0.008184072392824608`.

## Evidence Grade And Verdict

Evidence grade: `[CPU-build]`.

This artifact is byte-closed and score-affecting bytes changed, but it is not a
CUDA dispatch candidate as-is.

Reasons:

- The importance source ultimately comes from a diagnostic/stub beta-Fisher
  sensitivity input, so it is not score authority.
- The generic allocator's weighted RMS target does not bound the builder's
  aggregate quantized-weight rel_err; the selected row is byte-lower but has
  materially higher CPU rel_err than the default no-dead-K Path-B artifact.
- Manifest custody remains `score_claim=false`,
  `promotion_eligible=false`, `rank_or_kill_eligible=false`, and
  `ready_for_exact_eval_dispatch=false`.
- Active blocker classes include CPU/proxy-only importance, missing CUDA pixel
  Jacobian/Fisher pullback, missing static pre-submission compliance, missing
  exact CUDA auth eval, and the inherited
  `apogee_int6_contest_cuda_anchor_required_first`.

Interpretation: this is a useful byte-closed negative/edge candidate. It proves
the allocator-to-builder path changes charged bytes and can lower archive
bytes, but the rel_err profile and proxy-only sensitivity source make it a
fail-closed blocker for dispatch until a non-diagnostic importance artifact or
explicit operator-approved CUDA experiment rationale exists.

## Verification

Commands:

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_jacobian_fisher_importance_allocator.py \
  src/tac/tests/test_build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py -q
```

Result: `25 passed in 0.50s`.

Artifact sanity check:

- manifest archive bytes matched filesystem size (`147,285`)
- manifest SHA-256 matched recomputed SHA-256
- ZIP contained exactly one stored member, `x`
- smoke decoded `600` latent pairs
- staged `submission_dir/` had no remaining `__pycache__`
- `score_claim=false`
- `ready_for_exact_eval_dispatch=false`
