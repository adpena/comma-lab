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
