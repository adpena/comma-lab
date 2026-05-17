# Rule #6 Z3HV2 Direct-Residual Unwind - Codex 2026-05-17

parent_id_or_session: current_codex_l5_rule6_frontier_unwind
status: classified_and_blocked_from_frontier_dispatch_as_is
research_only: true
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
rank_or_kill_eligible: false
gpu_spend: false
axis_labels: `[contest-CPU]`, `[contest-CUDA]`, `[byte-layout]`

## Finding

The existing Z3 v2 full artifact is not a true Ballé entropy-coded residual
implementation. It is a Z3HV2 direct-residual control:

- `hyperprior_weights_int8` slot: empty
- `w_hat_int8` slot: empty
- residual coding: `brotli_direct_int8_residual`
- active Ballé entropy residual decoder in inflate path: `false`

The artifact must not be dispatched again as a frontier candidate without an
implementation change. The next valid Z3/Ballé action is either a real entropy
decoder that consumes side-info to decode residual symbols, or a different
Rule #6 latent/sidecar coder.

## Byte Evidence

Profile artifact:

- JSON: `.omx/research/z3v2_full_payload_authority_profile_20260517_codex.json`
- Markdown: `.omx/research/z3v2_full_payload_authority_profile_20260517_codex.md`

Profile command:

```bash
.venv/bin/python tools/profile_z3v2_payload_contract.py \
  --archive experiments/results/lane_substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch_20260515T114115Z__full__1000ep_modal/lane_substrate_z3_balle_hyperprior_bolton_results/output/submission_dir_reconstructed/archive.zip \
  --output-json .omx/research/z3v2_full_payload_authority_profile_20260517_codex.json \
  --output-md .omx/research/z3v2_full_payload_authority_profile_20260517_codex.md
```

Key fields:

| Field | Value |
|---|---:|
| Archive bytes | `179130` |
| Archive SHA-256 | `b6c4a6f1f1f4bb29695e8ee095ca3862690b2c4833fba31579406179aaf35a4b` |
| Inner member | `0.bin` |
| Inner payload bytes | `179022` |
| Z3HV2 section bytes | `16247` |
| A1 latent bytes replaced | `15387` |
| Signed byte savings | `-860` |
| Classification | `direct_residual_control` |

This is a byte regression before scorer effects. The rate term alone is worse
than A1 because the replacement section is larger than the original A1 latent
blob.

## Paired Exact-Eval Evidence

Same archive SHA-256:
`b6c4a6f1f1f4bb29695e8ee095ca3862690b2c4833fba31579406179aaf35a4b`.

| Axis | JSON | Canonical score | Components |
|---|---|---:|---|
| `[contest-CPU]` | `experiments/results/modal_auth_eval_cpu/z3_v2_full_paired_cpu_20260515_paired_modal_auth_z3_v2_full_paired_20260515T142723Z_cpu/contest_auth_eval.json` | `0.1986956456779881` | pose `0.018008331405213532`, seg `0.061412`, rate `0.11927531427277456` |
| `[contest-CUDA]` | `experiments/results/modal_auth_eval/z3_v2_full_paired_cpu_20260515_paired_modal_auth_20260515T142723Z_cuda/contest_auth_eval.json` | `0.23170948072940661` | pose `0.04132916645663205`, seg `0.071105`, rate `0.11927531427277456` |

Against the A1 local frontier (`0.19285 [contest-CPU]` per `PROGRAM.md`), this
is not sub-A1 on CPU and is much worse on CUDA. It is therefore not a promotion
result and not a dispatch-priority candidate as currently engineered.

## Root Cause

The 2026-05-14 Z3 fail-closed patch correctly killed append-only Z3HP1, but
the replacement Z3HV2 path still did not become a Ballé entropy-coded archive.
It became a safer direct-residual replacement grammar:

```text
[A1 decoder section]
[Z3HV2 header + brotli-compressed int8 residual + affine]
[A1 sidecar]
```

The current `inflate_v2.py` path may compute sigma if side-info exists, but the
residual has already been decompressed by brotli before sigma is available.
That sigma is diagnostic-only for this grammar, not an entropy decoding
conditioner. Therefore the "Ballé hyperprior" label is not sufficient authority
for score-lowering; the shipped decoder contract is what matters.

## Decision

Do not spend more GPU or exact-eval budget on this exact Z3HV2 artifact or
equivalent direct-residual Z3HV2 exports.

Valid reactivation requires at least one of:

1. A true entropy-coded residual decoder where side-info changes residual
   symbol probabilities at inflate time, with byte-mutation proof that
   side-info changes the decoded residual.
2. A Rule #6 latent coder that beats the A1 15,387 B latent slot on bytes before
   exact eval, then survives paired `[contest-CPU]` and `[contest-CUDA]`.
3. A sidecar-only surgical coder that targets the 607 B A1 sidecar if it can
   prove score-neutral or score-improving output changes.

## Six-Hook Disposition

- Sensitivity map: no scorer sensitivity update; this is an archive-contract
  falsification of the measured implementation.
- Pareto constraint: direct-residual Z3HV2 is constrained to
  `ready_for_exact_eval_dispatch=false` when section bytes are not smaller than
  the replaced A1 latent slot.
- Bit allocator: latent Rule #6 remains open, but allocation must target a
  true conditional/context coder, not direct int8 residual brotli.
- Cathedral autopilot: do not route existing Z3HV2 direct-residual artifacts as
  score-lowering candidates.
- Continual learning: paired exact-eval evidence updates the posterior against
  direct-residual Z3HV2, not against Ballé-style entropy coding generally.
- Probe-disambiguator: `tools/profile_z3v2_payload_contract.py` is the concrete
  byte-consumption/profiling probe for this failure class.

## Verification

```bash
.venv/bin/python -m py_compile \
  src/tac/analysis/z3v2_payload_profile.py \
  tools/profile_z3v2_payload_contract.py \
  src/tac/tests/test_z3v2_payload_profile.py
# clean

.venv/bin/python -m pytest src/tac/tests/test_z3v2_payload_profile.py -q
# 5 passed
```

## Next Frontier Actions

1. Run the same payload authority profile on any future Z3HV2/Z3G2 candidate
   before exact-eval dispatch.
2. Prioritize the Rule #6 A1 latent slot with a real context/entropy coder; the
   A1 byte profile shows first-order coding is saturated but order-1 latent
   headroom remains large enough to justify a context model probe.
3. Treat the existing Z3 v2 paired scores as a measured negative for the
   direct-residual implementation only, not a falsification of Ballé 2018 or
   learned hyperprior coding.
