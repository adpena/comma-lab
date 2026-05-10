# PR103 global-combo mid32 + latent-hi -16B packet (2026-05-10)

## Verdict

This is a byte-closed PR103 rate candidate, not a score claim yet. It extends
the verified `-12B` same-runtime CUDA path by adding the latent-hi histogram
sideband to the exact global-combo objective and by widening the q8 histogram
frontiers to mid32 probes.

The candidate has full local CPU inflate-output parity with PR103 source over
all 600 pairs. Exact Modal CUDA remains required before any score or submission
readiness claim.

## Candidate

- archive:
  `.omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_mid32_plus_latent_hi_candidate/packet/archive.zip`
- archive SHA-256:
  `8460014d70855ce9226285f80513d6d743ed23723870a6a38b009cfca40f423e`
- archive bytes: `178207`
- source archive bytes: `178223`
- archive byte delta: `-16`
- expected rate-only score delta if components are unchanged:
  `-0.000010653743249954742`
- packet runtime tree SHA-256:
  `127ef03c27581f67b624e93ddd181701e32b259c5d046bf1a1ed3f35d714eb7e`
- adapter runtime tree SHA-256:
  `db3a2ac26718e2b797aca0eee5f026b5f9eb807bd83b3115519c84b6fd28b6fa`

## Local parity

- full frame parity report:
  `.omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_mid32_plus_latent_hi_candidate/frame_parity_probe_full_cpu.json`
- scope: full 600 pairs on local CPU
- output bytes: `3,662,409,600`
- source output SHA-256:
  `074f834f14ba4611f9358bb0a3f8e729bb43e4ea673be23e2acf85e7448dd1e5`
- candidate output SHA-256:
  `074f834f14ba4611f9358bb0a3f8e729bb43e4ea673be23e2acf85e7448dd1e5`
- pair hashes match: `true`

## Optimizer facts

The global optimizer minimizes the recomputed full-state objective:

```text
delta = merged_ac_delta
      + ac_histograms_brotli_delta
      + latent_hi_histogram_brotli_delta
```

The best local candidate selected:

- `blocks.0.weight:candidate2`
- `blocks.1.weight:candidate7`
- `blocks.2.weight:source`
- `blocks.3.weight:candidate8`
- `stem.weight:source`
- `latent_hi_bytes:candidate1`

The final objective components were:

- merged AC delta: `0`
- q8 AC histogram Brotli delta: `-15`
- latent-hi histogram Brotli delta: `-1`
- total member/archive delta: `-16`

## Dispatch blockers

- lane dispatch claim missing
- exact CUDA auth eval missing

Do not compare this packet to an older CUDA replay. The required next evidence
is a same-runtime Modal T4 candidate run, with PR103 source baseline reused only
when the runtime and command surfaces are demonstrably comparable.
