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
