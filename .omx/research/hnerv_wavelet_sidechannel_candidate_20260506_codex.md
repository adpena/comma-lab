# HNeRV Wavelet Sidechannel Candidate - 2026-05-06

This records the first byte-different HNeRV wavelet residual candidate. It is
not a score claim, not an archive-preflight pass, and not exact-eval dispatch
clearance.

Command:

```bash
.venv/bin/python tools/build_hnerv_wavelet_sidechannel_candidate.py \
  --source-archive experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/archive.zip \
  --scorecard experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/scorecard.json \
  --source-label PR106x \
  --output-dir experiments/results/hnerv_wavelet_sidechannel_pr106x_20260506_codex \
  --target-section latents_and_sidecar_brotli \
  --top-k 32 \
  --block-size 64 \
  --json-out experiments/results/hnerv_wavelet_sidechannel_pr106x_20260506_codex/manifest.json \
  --fail-if-blocked
```

Observed candidate output:

- `ready_for_wavelet_sidechannel_candidate=true`
- `ready_for_archive_preflight=false`
- `ready_for_exact_eval_dispatch=false`
- source archive bytes: `186231`
- candidate archive bytes: `186619`
- candidate archive byte delta: `+388`
- source payload bytes: `186131`
- candidate payload bytes: `186519`
- candidate payload byte delta: `+388`
- wavelet sidechannel bytes: `379`
- decoded atom count: `32`
- runtime consumed sidechannel: `true`
- runtime atom-coordinate SHA-256:
  `23f7615b6cc5b011363b5c1822cbd81ef2b9e05305d380e85a747af5ca90eff8`
- source plan SHA-256:
  `df131f012367c2a9bd9976f51e8d5907969613e4556aedbdcc6ee45923a69038`
- candidate archive SHA-256:
  `4e9846a98bf7b5d543552dabe34fd067d5b039ee1d3a75eebb83ff67bb88a993`
- wavelet sidechannel SHA-256:
  `d927bea797c257a4e99e2f5c9f10c1bc5a70bd57d2f3bd4adf3cc22fb19477a5`

Tracked metadata:

- `experiments/results/hnerv_wavelet_sidechannel_pr106x_20260506_codex/manifest.json`
- `experiments/results/hnerv_wavelet_sidechannel_pr106x_20260506_codex/hnerv_wavelet_sidechannel_candidate.json`

Ignored local payload artifact:

- `experiments/results/hnerv_wavelet_sidechannel_pr106x_20260506_codex/hnerv_wavelet_sidechannel_candidate.zip`

Interpretation:

- This is a charged sidechannel candidate over real PR106x HNeRV bytes.
- The original HNeRV payload is preserved byte-for-byte inside the wrapper.
- The WR01 sidechannel is brotli-compressed, decoded by runtime code, and
  validated by a deterministic atom-coordinate digest.
- The current candidate does not apply the atoms to output pixels. Therefore
  it is useful engineering evidence for stackability and no-op resistance, but
  it is not promotable score evidence.

Dispatch blockers:

- `candidate_sidechannel_not_applied_by_inflate_runtime`
- `requires_inflate_runtime_integration`
- `requires_archive_manifest_preflight`
- `requires_exact_cuda_auth_eval`

Relation to PARADIGM-alpha:

The original PARADIGM-alpha audit targeted full mask payload replacement:
NeRV, wavelet mask codec, VQ-VAE mask codec, and grayscale-LUT. This candidate
does not complete the α2 mask-codec lane. It is a smaller, stackable wavelet
residual mechanism for HNeRV latent/sidecar bytes and should be treated as an
adjacent residual/sidechannel atom path.

Next implementation step:

Add an inflate-runtime integration mode that consumes WR01 atoms in a
deterministic transform, or explicitly proves a byte-closed no-op mode for
stack-composition testing. Only after that should archive preflight decide
whether exact CUDA auth eval is eligible.

## Stack-Composition Runtime Follow-Up

The WR01 payload is now accepted by the canonical PR106 stack-composition
runtime as section `0x04`.

Command:

```bash
.venv/bin/python experiments/build_pr106_stacked.py \
  --pr106-archive experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip \
  --wavelet experiments/results/hnerv_wavelet_sidechannel_pr106x_20260506_codex/hnerv_wavelet_sidechannel_candidate.zip \
  --output-dir experiments/results/pr106_stacked_wavelet_wr01_noop_20260506_codex
```

Observed stacked output:

- stack archive bytes: `186626`
- PR106 anchor archive bytes: `186239`
- byte delta vs PR106 anchor archive: `+387`
- rate-only score delta vs PR106 anchor: `+0.0002576874148582803`
- parsed section ids: `[4]`
- WR01 section bytes: `379`
- WR01 decoded atoms: `32`
- WR01 runtime mode: `explicit_noop_consume_only`
- WR01 runtime consumed: `true`
- WR01 atom-coordinate SHA-256:
  `23f7615b6cc5b011363b5c1822cbd81ef2b9e05305d380e85a747af5ca90eff8`
- stacked archive SHA-256:
  `e1d22ae5e712944d5c5dd55a4a048eca17f98461ab643c74b6d7ec56b8bd64f1`
- build metadata SHA-256:
  `475e73a0fbd083cdbe28625c9e8617685694cb19c701bd0fd9012461c06a5051`

Tracked metadata:

- `experiments/results/pr106_stacked_wavelet_wr01_noop_20260506_codex/build_metadata.json`

Ignored local payload artifact:

- `experiments/results/pr106_stacked_wavelet_wr01_noop_20260506_codex/pr106_stacked_archive.zip`

Machine-readable blockers:

- `compose_time_scaffold_only`
- `requires_sister_sidechannels_to_win_exact_cuda_before_stack_dispatch`
- `requires_archive_manifest_preflight`
- `requires_exact_cuda_auth_eval`
- `wavelet_wr01_runtime_mode_is_explicit_noop`
- `wavelet_wr01_rate_regression_without_distortion_benefit`
- `requires_reviewed_wavelet_apply_transform`

Updated conclusion:

The stack parser and builder now close the WR01 runtime-consumption loop, but
the stacked artifact is intentionally worse on rate and has no distortion path.
It is engineering evidence for composability and no-op resistance, not a
candidate for exact eval. The next real score step is a reviewed wavelet apply
transform that can plausibly buy at least `0.000258` score through SegNet or
PoseNet improvement before exact CUDA dispatch is considered.

## WR01 Apply-Gate Break-Even

The no-op stack now has a nonarbitrary readiness gate for the next score-relevant
WR01 transform. The gate computes the exact contest-rate penalty and converts it
into the minimum component benefit that a future transform must justify before
archive preflight or exact CUDA dispatch.

Command:

```bash
.venv/bin/python tools/audit_hnerv_wavelet_apply_gate.py \
  --sidechannel-manifest experiments/results/hnerv_wavelet_sidechannel_pr106x_20260506_codex/manifest.json \
  --stacked-metadata experiments/results/pr106_stacked_wavelet_wr01_noop_20260506_codex/build_metadata.json \
  --json-out experiments/results/hnerv_wavelet_apply_gate_pr106x_20260506_codex.json
```

Observed gate output:

- archive byte delta: `+387`
- rate-only score delta: `+0.0002576874148582803`
- minimum SegNet-distance reduction to break even:
  `0.0000025768741485828032`
- runtime mode: `explicit_noop_consume_only`
- decoded WR01 atoms: `32`
- ready for archive preflight: `false`
- ready for exact CUDA dispatch: `false`
- gate JSON SHA-256:
  `8cf77a6c59f1d3421701de27f2bfd688d2cca3b812e40c0f90fcc90465af6940`

Dispatch blockers:

- `requires_reviewed_wr01_apply_transform`
- `requires_component_benefit_evidence_over_break_even`
- `requires_archive_manifest_preflight`
- `requires_exact_cuda_auth_eval`
- `wr01_rate_penalty_must_be_recovered_by_distortion`
- `wr01_runtime_mode_is_explicit_noop`

Updated conclusion:

The WR01 path is byte-closed, runtime-consumed, and now break-even-gated. It
must not dispatch as a no-op rate regression. The next implementation has to
replace `explicit_noop_consume_only` with a reviewed deterministic apply
transform and show component-response evidence above the `0.000258` score bar
before archive preflight or CUDA auth eval is eligible.

## Offline WR01 Apply Transform

The next tranche converts WR01 from a runtime no-op sidechannel into an offline
plain-HNeRV transform candidate. The tool decodes the WR01 atom coordinates,
verifies the referenced HNeRV section SHA-256, attenuates the selected Haar
detail coefficients in the decompressed latent/sidecar bytes, recompresses the
section, and emits a normal `0xff` PR106-style archive. No sidecar or runtime
dependency is required in the candidate archive.

Command family:

```bash
.venv/bin/python tools/build_hnerv_wavelet_apply_transform_candidate.py \
  --wavelet-archive experiments/results/hnerv_wavelet_sidechannel_pr106x_20260506_codex/hnerv_wavelet_sidechannel_candidate.zip \
  --output-dir experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex \
  --source-label PR106x \
  --strength-numerator 1 \
  --strength-denominator 2 \
  --json-out experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/manifest.json \
  --fail-if-not-archive-preflight-ready
```

Strength sweep:

| strength | candidate bytes | byte delta vs source estimate | section delta | changed raw positions | total abs raw delta |
|---:|---:|---:|---:|---:|---:|
| `1/4` | `186230` | `-1` | `-1` | `64` | `1820` |
| `1/2` | `186222` | `-9` | `-9` | `64` | `3610` |
| `3/4` | `186225` | `-6` | `-6` | `64` | `5424` |
| `1/1` | `186224` | `-7` | `-7` | `64` | `7210` |

Best rate candidate in this sweep:

- strength: `1/2`
- archive bytes: `186222`
- archive byte delta vs source estimate: `-9`
- rate-only score delta vs source estimate: `-0.0000059927305780995425`
- applied atoms: `32`
- candidate archive SHA-256:
  `d2208ffa41297c40ce5f0bdbbe4767a9831301e382522afd2f6acf455a6b1628`
- manifest SHA-256:
  `de79e2a5fb3efc70cabec70e6f7404c028f3ce3a8cbf0500cafe824cb9890fff`

Tracked manifests:

- `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_4_20260506_codex/manifest.json`
- `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/manifest.json`
- `experiments/results/hnerv_wavelet_apply_transform_pr106x_3_4_20260506_codex/manifest.json`
- `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_1_20260506_codex/manifest.json`

Ignored local payload artifacts:

- `experiments/results/hnerv_wavelet_apply_transform_pr106x_*/hnerv_wavelet_apply_transform_candidate.zip`

Updated conclusion:

WR01 is no longer only a no-op stackability proof. The half-strength offline
candidate is byte-negative and plain-HNeRV-format, so the next scientifically
correct step is archive preflight plus component-response or exact CUDA auth
eval. It remains `score_claim=false` until that happens; the current evidence
is a real byte-transform/rate artifact, not score evidence.

## WR01 Apply Transform Static Preflight And Dispatch Plan

The first archive-preflight attempt exposed a harness bug: single-member `x`
archives were always interpreted as the PR85 bundle family. Public HNeRV
archives also use member `x`, but their wire format starts with `0xff` and
contains two 24-bit-length brotli sections. The public replay preflight now
content-detects `0xff` HNeRV payloads before the PR85 parser and fails closed
on malformed HNeRV payloads.

Regression coverage:

```bash
.venv/bin/python -m pytest src/tac/tests/test_preflight_public_replay_intake.py -q
```

Static replay preflight command:

```bash
.venv/bin/python experiments/preflight_public_replay_intake.py \
  --archive experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/hnerv_wavelet_apply_transform_candidate.zip \
  --inflate-sh experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/inflate.sh \
  --upstream-dir experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source \
  --expected-archive-sha256 d2208ffa41297c40ce5f0bdbbe4767a9831301e382522afd2f6acf455a6b1628 \
  --expected-archive-size-bytes 186222 \
  --json-out experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/public_replay_preflight.json \
  --fail-if-not-ready
```

Observed static replay preflight:

- `ready_for_exact_eval_dispatch=true`
- blockers: `[]`
- archive bytes: `186222`
- archive SHA-256:
  `d2208ffa41297c40ce5f0bdbbe4767a9831301e382522afd2f6acf455a6b1628`
- member: `x`
- member SHA-256:
  `803a0940f92ec1cb1b70e9815fb6c666650b5371625e17103104baa21e96b4e7`
- detected member format: `hnerv_ff_len24_brotli_sections`
- decoder brotli bytes: `170278`
- decoder decoded bytes: `229070`
- latents/sidecar brotli bytes: `15840`
- latents/sidecar decoded bytes: `33712`
- inflate runtime tree SHA-256:
  `4a8445803e232159be84f84f726311c4e6846a0a9e7a1825e1be38cf9e165242`

Pre-submission compliance command:

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders \
  --archive experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/hnerv_wavelet_apply_transform_candidate.zip \
  --archive-manifest-json experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/manifest.json \
  --expect-single-member x \
  --expected-archive-sha256 d2208ffa41297c40ce5f0bdbbe4767a9831301e382522afd2f6acf455a6b1628 \
  --expected-archive-size-bytes 186222 \
  --json-out experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/pre_submission_compliance.json
```

Observed compliance result:

- failing checks: `[]`
- archive member: `x`
- archive bytes: `186222`
- archive SHA-256:
  `d2208ffa41297c40ce5f0bdbbe4767a9831301e382522afd2f6acf455a6b1628`

Lightning exact-eval dry-run:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py exact-eval \
  --job-name exact_eval_wr01_apply_pr106x_half_20260506 \
  --archive experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/hnerv_wavelet_apply_transform_candidate.zip \
  --repo-dir /Users/adpena/Projects/pact \
  --upstream-dir experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source \
  --inflate-sh experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/inflate.sh \
  --output-dir experiments/results/lightning_batch/exact_eval_wr01_apply_pr106x_half_20260506 \
  --local-artifact-dir experiments/results/lightning_batch/exact_eval_wr01_apply_pr106x_half_20260506 \
  --infer-expected-archive \
  --adjudicate \
  --baseline-score 0.20945123680571204 \
  --baseline-archive-bytes 186231 \
  --baseline-posenet-dist 0.00003351 \
  --baseline-segnet-dist 0.00067142 \
  --regression-threshold 0.02 \
  --max-posenet-relative 1.25 \
  --max-segnet-relative 1.10 \
  --queue-metadata lane=wr01_apply_pr106x_half \
  --queue-metadata archive_manifest=experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/manifest.json \
  --queue-metadata public_preflight=experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/public_replay_preflight.json \
  --predicted-band 0.18 0.25 \
  --max-sane-score 1.0 \
  --component-trace \
  --dry-run
```

Observed Lightning dry-run:

- status: `DRY_RUN`
- submit readiness: `ok=true`
- blockers: `[]`
- target machine: `T4`
- expected archive bytes: `186222`
- expected archive SHA-256:
  `d2208ffa41297c40ce5f0bdbbe4767a9831301e382522afd2f6acf455a6b1628`
- command SHA-256:
  `6da65a57962908529a6780454894d52574c57a6f2d9e1c8a06745ab6fccb49f2`
- predicted SDK artifact path:
  `/teamspace/jobs/exact-eval-wr01-apply-pr106x-half-20260506/artifacts`

Tracked artifacts:

- `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/public_replay_preflight.json`
- `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/pre_submission_compliance.json`
- `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/lightning_exact_eval_dry_run.json`

Current conclusion:

The half-strength WR01 apply transform is now preflight-clean for exact eval,
but it is still not a score claim. Local CUDA is unavailable on this machine,
so the next promotable evidence step is a claimed non-dry-run Lightning exact
CUDA auth eval, followed by harvest, JSON adjudication, component trace, and a
score-delta ledger update.

## WR01 Payload Section Diff

The HNeRV payload comparator now provides a standalone section-diff proof for
source/candidate archive pairs. This closes the no-op-control gap for packed
HNeRV archives without relying on scorecard prose or candidate-specific
manifest fields.

Command:

```bash
.venv/bin/python tools/compare_hnerv_payload_sections.py \
  --source-archive experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/archive.zip \
  --candidate-archive experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/hnerv_wavelet_apply_transform_candidate.zip \
  --source-label PR106x \
  --candidate-label WR01_half \
  --source-manifest-json experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/scorecard.json \
  --json-out experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/payload_section_diff_vs_pr106x.json \
  --fail-if-no-section-change
```

Observed section diff:

- `ready_for_archive_preflight=true`
- blockers: `[]`
- changed section count: `1`
- archive byte delta vs PR106x: `-9`
- payload byte delta vs PR106x: `-9`
- unchanged section: `packed_header_ff_len24`
- unchanged section: `decoder_packed_brotli`
- changed section: `latents_and_sidecar_brotli`
- changed section byte delta: `-9`
- decoder brotli raw equality: `true`
- latents/sidecar brotli raw equality: `false`
- latents/sidecar raw changed positions: `64`
- latents/sidecar raw absolute delta sum: `3610`

Tracked artifact:

- `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/payload_section_diff_vs_pr106x.json`

Updated conclusion:

The WR01 half-strength candidate is now byte-different and no-op-resistant at
the HNeRV section level. The transform does not perturb decoder bytes; all
charged changes are isolated to the latent/sidecar brotli section. This is
still byte-forensic evidence, not score evidence, until exact CUDA auth eval
measures SegNet/PoseNet.

## WR01 Strength Section-Diff Sweep

The section-diff guard was applied to all four offline WR01 strength candidates
so the exact-eval target is selected from deterministic byte evidence rather
than intuition.

Command family:

```bash
.venv/bin/python tools/compare_hnerv_payload_sections.py \
  --source-archive experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/archive.zip \
  --candidate-archive experiments/results/hnerv_wavelet_apply_transform_pr106x_<strength>_20260506_codex/hnerv_wavelet_apply_transform_candidate.zip \
  --source-label PR106x \
  --candidate-label WR01_<strength> \
  --source-manifest-json experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/scorecard.json \
  --json-out experiments/results/hnerv_wavelet_apply_transform_pr106x_<strength>_20260506_codex/payload_section_diff_vs_pr106x.json \
  --fail-if-no-section-change
```

Observed sweep:

| strength | archive byte delta | payload byte delta | changed raw positions | raw abs delta sum |
|---:|---:|---:|---:|---:|
| `1/4` | `-1` | `-1` | `64` | `1820` |
| `1/2` | `-9` | `-9` | `64` | `3610` |
| `3/4` | `-6` | `-6` | `64` | `5424` |
| `1/1` | `-7` | `-7` | `64` | `7210` |

Tracked artifacts:

- `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_4_20260506_codex/payload_section_diff_vs_pr106x.json`
- `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/payload_section_diff_vs_pr106x.json`
- `experiments/results/hnerv_wavelet_apply_transform_pr106x_3_4_20260506_codex/payload_section_diff_vs_pr106x.json`
- `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_1_20260506_codex/payload_section_diff_vs_pr106x.json`
- `experiments/results/hnerv_wavelet_apply_transform_wr01_strength_summary_20260506_codex.json`

Selection rationale:

The `1/2` candidate remains the exact-eval target because it has the largest
rate improvement (`-9` archive bytes) while staying in the middle of the raw
perturbation ladder. The `1/4` candidate is the conservative distortion probe;
the `1/1` candidate is the maximum perturbation probe. None of these rows is a
score claim until CUDA auth eval measures the actual component response.

## WR01 Exact-Eval Operator Packet

The WR01 exact-eval handoff is now a deterministic packet rather than a chat
command. It validates the static artifacts, records missing Lightning runtime
identity, and emits the three operator commands: lane claim, staged submit, and
harvest/adjudication.

Command:

```bash
.venv/bin/python tools/build_wr01_exact_eval_packet.py \
  --json-out experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/wr01_exact_eval_packet.json
```

Observed packet:

- `ready_for_submit=false`
- blocker: `missing_lightning_environment`
- missing env:
  `LIGHTNING_SSH_TARGET`, `LIGHTNING_REMOTE_PACT`, `LIGHTNING_UPSTREAM_DIR`,
  `LIGHTNING_TEAMSPACE`, `LIGHTNING_STUDIO`, `LIGHTNING_SDK_USER`
- `preflight_ready=true`
- `compliance_ok=true`
- `payload_diff_ready=true`
- `dry_run_ready=true`
- dispatch attempted: `false`

Tracked artifact:

- `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/wr01_exact_eval_packet.json`

Updated conclusion:

Once Lightning env is loaded, the packet's `commands.claim` must run before
`commands.submit`. Until then the candidate remains locally prepared but not
submitted; the refusal is intentional and contest-compliance preserving.
