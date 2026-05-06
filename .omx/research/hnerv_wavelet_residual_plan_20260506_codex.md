# HNeRV Wavelet Residual Plan - 2026-05-06

This records the first promoted wavelet hidden-gem surface as a planning-only
artifact. It is not a score claim and not an exact-eval dispatch clearance.

Command:

```bash
.venv/bin/python tools/plan_hnerv_wavelet_residual.py \
  --source-archive experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/archive.zip \
  --scorecard experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/scorecard.json \
  --source-label PR106x \
  --target-section latents_and_sidecar_brotli \
  --top-k 32 \
  --block-size 64 \
  --json-out /tmp/pr106x_wavelet_residual_plan.json \
  --fail-if-blocked
```

Observed planning output:

- `ready_for_wavelet_candidate_build=true`
- `ready_for_archive_preflight=false`
- `ready_for_exact_eval_dispatch=false`
- `plan_sha256=df131f012367c2a9bd9976f51e8d5907969613e4556aedbdcc6ee45923a69038`
- source archive SHA-256: `d25bca80057e8b533197895b4c56370678feb4e05fea0312c405bd12f29bec8e`
- target section: `latents_and_sidecar_brotli`
- source section wire bytes: `15849`
- brotli-decompressed transform-domain bytes: `33712`
- raw transform-domain SHA-256 prefix: `a38778c6304bacba`
- selected Haar atoms: `32`
- estimated atom wire bytes: `192`

Top three atoms from the real PR106x latent/sidecar transform domain:

| raw_offset | raw_end | level | coeff_index | coeff_q | est_wire_bytes |
|---:|---:|---:|---:|---:|---:|
| 14688 | 14690 | 0 | 16 | -122 | 6 |
| 15730 | 15732 | 0 | 25 | 120 | 6 |
| 8372 | 8374 | 0 | 26 | 118 | 6 |

Interpretation:

- This is a concrete atom-selection surface over real HNeRV payload bytes,
  not a replacement codec claim.
- The selected atoms are optimizer feedback for a future charged sidechannel
  or section recode.
- A future candidate must record old/new archive SHA-256, old/new payload
  SHA-256, old/new section SHA-256, old/new bytes, and runtime consumption of
  the wavelet atoms before archive preflight can pass.
- Exact CUDA auth eval remains mandatory before ranking or score claims.

Next implementation step:

Build a byte-different candidate that consumes these atoms in a charged
runtime-visible sidechannel, then run `audit_candidate_section_diff` or an
equivalent section-proof gate before any exact eval dispatch.
