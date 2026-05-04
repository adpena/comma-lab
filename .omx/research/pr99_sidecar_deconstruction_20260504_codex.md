# PR99 Sidecar Deconstruction - 2026-05-04

Scope: local-only deconstruction of PR99's one-dimensional per-pair latent
correction sidecar. No remote jobs were dispatched, no dispatch claims were
created or modified, and no score claim is made here. Owned artifacts live under
`experiments/results/pr99_sidecar_deconstruction_20260504_codex/`.

## Inputs

- Source archive:
  `experiments/results/leaderboard_intel_20260504_codex/pr99_archive.zip`
- Source archive bytes: `178546`
- Source archive SHA-256:
  `278b1c7a1bd6b03a5bceddafcb3489b2624c558ad22825d9211b701333b6eefb`
- Source runtime:
  `experiments/results/leaderboard_intel_20260504_codex/pr99_runtime`
- Sanitized runtime baseline:
  `experiments/results/final_packet_readiness_pr98_pr99_20260504_codex/runtime_snapshots/pr99_runtime`

## Archive Anatomy

PR99 is a single stored ZIP member named `0.bin`. The member is a four-part
length-prefixed container:

| Part | Bytes | Interpretation |
| --- | ---: | --- |
| decoder brotli | 161883 | INT8 HNeRV decoder codes |
| scales fp16 | 56 | per-tensor decoder scales |
| latents brotli | 15868 | per-dim asym uint8 + temporal deltas |
| sidecar | 615 | per-pair one-dim latent corrections |
| length prefixes | 16 | four `u32` lengths |
| member total | 178438 | charged `0.bin` payload |
| outer ZIP overhead | 108 | stored single-member ZIP overhead |

The sidecar rate term alone is about `25 * 615 / 37545489 =
0.0004095033`, so deleting it is not attractive unless the sidecar is
score-neutral. It almost certainly is not score-neutral because it touches
`598 / 600` pairs.

## Sidecar Signal

Wire contract in `sidecar.py`:

```text
u16 n_pairs
per pair: u8 dim_idx, i8 delta_q
dim_idx=255 means no correction
real_delta = delta_q * 0.01
```

Measured distribution:

- Raw sidecar payload: `1202` bytes.
- Brotli-compressed sidecar: `615` bytes.
- Pairs: `600`.
- Corrected pairs: `598`.
- Uncorrected pairs: `412`, `414`.
- Corrected runs: `0-411`, `413-413`, `415-599`.
- Delta alphabet is tiny: `{-10, -5, -2, 2, 5, 10}`.
- Delta counts: `2:181`, `-2:183`, `5:116`, `-5:105`, `10:7`, `-10:6`.
- Mean absolute corrected delta: `3.282608695652174`; max absolute delta:
  `10`.
- Dimensions are broadly spread across all 28 latent dimensions, so a sparse
  `(pair, dim, delta)` list is worse than the dense pair-aligned stream.

Interpretation: PR99's sidecar is not a few exceptional fixes. It is a compact
one-dimensional learned correction field over almost every pair, with a very
small signed delta alphabet. That makes it a high-signal surface for future
water-fill work, but a poor target for deletion ablations under deadline.

## Local Candidates

All candidates are local-only and passed static
`preflight_public_replay_intake.py --fail-if-not-ready`. Exact eval is still
required before any score claim.

| Candidate | Bytes | Delta vs PR99 | Archive SHA-256 | Runtime tree SHA-256 | Notes |
| --- | ---: | ---: | --- | --- | --- |
| `baseline_normalized_store` | 178546 | 0 | `249e53bc9ffeacaea8b233cbcdcd13f134a841bd014a91d2620d0201432edec1` | `04583e3b64e7dc8b71a986e93b0ed685a5709355d5ab920bbde7b50b28fef0a3` | deterministic stored-zip control |
| `baseline_outer_deflate9` | 178601 | +55 | `cddf79a63917363746e8f271be86e29ff37b9504a9d4588f709bdd6da50fa9bf` | `d43e1398c9c156ab96dd3472016881ec3b0285c03483b889b8db2bfa1566c92c` | outer deflate loses; do not eval |
| `wrp_rebrotli_q10_w10` | 178546 | 0 | `adbb563802014276fff3efa311ce3be23e56006e8b5e26966d0616841ffafd41` | `851ebe864b5ef148a9a4928855fd883650dcc09654dcc1db9a76bb39ac506f0c` | sidecar rebrotli tie; no byte win |
| `split_stream` | 178509 | -37 | `a50913059816fec4bb7f7796865b1219a34be61a9f50d5768f1b42677705de9b` | `a7474576d61a621e5d511223af56bdedd56fa295e5c468207c7ad1289b4e8bc5` | lossless split dim/delta streams; recommended if PR99 baseline validates |

The `split_stream` variant changes the runtime sidecar decoder and the archive
sidecar wire format. It reconstructs the original PR99 `(dim, delta_q)` arrays
exactly in a local roundtrip check. The byte gain is only `37` bytes, worth
about `0.0000246368` score from rate alone, so this should not preempt PR98/PR99
baseline T4 harvest. It is a clean follow-up if PR99 itself validates.

## Codec Screens

Lossless sidecar recode screens:

- `split_stream`: `578` bytes, exact correction roundtrip, saves `37` bytes.
- `bitpack_stream`: `673` bytes, exact correction roundtrip, worse than source.
- `sparse_stream`: `1136` bytes, exact correction roundtrip, much worse because
  nearly every pair is corrected.

Brotli parameter sweep over the original raw sidecar found a tie at `615`
bytes; no archive-size win.

## Artifacts

- Builder:
  `experiments/results/pr99_sidecar_deconstruction_20260504_codex/build_pr99_sidecar_deconstruction.py`
- Analysis JSON:
  `experiments/results/pr99_sidecar_deconstruction_20260504_codex/analysis.json`
- Candidate archives:
  `experiments/results/pr99_sidecar_deconstruction_20260504_codex/candidates/`
- Runtime variants:
  `experiments/results/pr99_sidecar_deconstruction_20260504_codex/runtime_variants/`
- Static preflight JSONs:
  `experiments/results/pr99_sidecar_deconstruction_20260504_codex/preflight/`
- No-dispatch exact eval plans:
  `experiments/results/pr99_sidecar_deconstruction_20260504_codex/eval_command_plans/`

## Recommendation

1. First harvest the already-queued PR99 baseline T4 exact eval. That is the
   score-truth blocker.
2. If PR99 baseline validates and beats the current champion, run exactly one
   follow-up T4 eval for `split_stream`; it is lossless over the correction
   arrays and preflight-clean, but runtime-custody changes still require CUDA
   auth eval.
3. Do not dispatch `baseline_outer_deflate9`, `wrp_rebrotli_q10_w10`, or a
   deletion ablation before the baseline result. They do not have enough
   expected value under the remaining wall-clock constraints.

## Verification

- `AGENTS.md` read before work.
- No remote jobs dispatched.
- No dispatch claims created or modified.
- Static preflight passed for all emitted candidates.
- Runtime syntax checks passed without writing `__pycache__`.
- Sidecar `split_stream` local decode exactly matches original PR99 dimensions
  and deltas.
