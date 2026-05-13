# PR106/R2 sidecar recode emit manifest (2026-05-13)

Status: landed as ignored experiment artifacts plus tracked no-score summary.
Evidence axis: PacketIR/parser custody only; no contest score authority.

## Scope

This pass re-ran the canonical PR106/R2 latent-sidecar recode profiler with
runtime-candidate archive emission enabled. It does not dispatch a provider job,
does not evaluate a score, and does not promote any candidate.

Command:

```bash
.venv/bin/python tools/profile_pr106_latent_sidecar_recode.py \
  --sidecar-archive submissions/pr106_latent_sidecar_r2/archive.zip \
  --json-out experiments/results/pr106_latent_sidecar_recode_emit_r2_20260513_codex/profile.json \
  --md-out experiments/results/pr106_latent_sidecar_recode_emit_r2_20260513_codex/profile.md \
  --emit-runtime-candidates-dir experiments/results/pr106_latent_sidecar_recode_emit_r2_20260513_codex/runtime_candidates
```

## Emitted artifacts

Ignored artifact root:
`experiments/results/pr106_latent_sidecar_recode_emit_r2_20260513_codex/`

| candidate | archive bytes | archive SHA-256 | delta vs source bytes | score claim | ready for exact eval |
|---|---:|---|---:|---:|---:|
| `current_pr100_dim_delta_brotli_q11` | 186,822 | `7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f` | 0 | false | false |
| `pr101_ranked_no_op_sidecar_format_0x02` | 186,780 | `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383` | -42 | false | false |

The `pr101_ranked_no_op_sidecar_format_0x02` emitted archive is byte-identical
to the existing tracked PR106/R2 PR101-grammar archive:
`submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip`.

## Proof boundary

The profiler and per-candidate manifests prove:

- lossless semantic equivalence for the `(dim, delta_q)` sidecar arrays;
- PacketIR parse/re-emit identity for supported sidecar formats;
- all wrapper payload bytes are accounted by typed sections;
- no score claim, no promotion claim, and no exact-eval readiness claim.

The best runtime-decoded candidate remains the PR101-ranked 0x02 sidecar
grammar:

- charged sidecar bytes: 533
- source charged sidecar bytes: 575
- sidecar byte delta: -42
- rate-only formula delta if consumed with unchanged components:
  `-2.7966076031131196e-05`

This sidecar-only delta is smaller than the already landed low-level PR106/R2
PR101-grammar repack (`-151` archive bytes, exact CUDA delta
`-0.00010065943776230331`) and does not change the HNeRV local minimum by
itself. It is still useful because it keeps the candidate-emission path
byte-closed and machine-checkable for future arithmetic/range/ANS sidecar
experiments.

## Remaining blockers

Before any emitted sidecar candidate can enter a dispatch queue as a score
candidate:

1. Runtime decode/apply proof for the exact emitted archive/runtime pair.
2. Full-frame same-runtime parity or same-runtime auth eval if equivalence is
   claimed.
3. Claimed exact `[contest-CUDA]` auth eval with archive/runtime custody.
4. Contest adjudication and terminal dispatch-claim linkage.

The existing PR101-grammar candidate already has stronger evidence in
`.omx/research/pr106_r2_pr101_grammar_lowlevel_repack_20260513_codex.md`; this
memo preserves the regenerated candidate-emission result and prevents the
profile from becoming an orphaned ignored artifact.
