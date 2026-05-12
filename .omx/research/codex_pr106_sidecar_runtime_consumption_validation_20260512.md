# Codex validation: PR106 sidecar PacketIR runtime-consumption proof exists

Date: 2026-05-12
Agent: codex
Scope: validate fresh-eyes finding that PR106 PacketIR only had parser custody

## Verdict

The fresh-eyes finding was useful but stale for the current `main`: parser-only
PacketIR proof still exists, but it is no longer the strongest available proof.
`src/tac/packet_compiler/pr106_runtime_consumption.py` and
`tools/prove_pr106_sidecar_runtime_consumption.py` already provide a no-score
runtime decode/apply proof that imports the paired submission `inflate.py`,
decodes sidecar corrections through the runtime path, applies them to latents,
and proves a valid sidecar mutation changes runtime-visible correction and
corrected-latent digests.

## Verification

Commands run:

```bash
.venv/bin/python -m pytest src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py -q
.venv/bin/python tools/prove_pr106_sidecar_runtime_consumption.py \
  --archive submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip \
  --runtime-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --output-json experiments/results/pr106_sidecar_runtime_consumption_latest.json
```

Results:

- Tests: `6 passed in 0.76s`
- Proof schema: `pr106_sidecar_runtime_decode_consumption_proof_v1`
- Source archive SHA-256: `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`
- Runtime `inflate.py` SHA-256: `60055bced3ab608d0e93ba83e18fa5bc662746cfa273ad50d5960c34028d1fb3`
- `runtime_sidecar_decode_consumption_claim=true`
- `runtime_sidecar_apply_consumption_claim=true`
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

The generated JSON proof is intentionally ignored under `experiments/results/`;
this ledger preserves the validation outcome without tracking rebuildable
artifacts.

## Remaining boundary

This proof is not a full-frame or exact-eval proof. It closes the narrower
PacketIR-to-runtime sidecar decode/apply gap. Any promotion/ranking/submission
claim still requires same-runtime full-frame parity or exact contest auth eval
with archive SHA, runtime tree SHA, sample count, component recomputation, and
explicit `[contest-CUDA]` / `[contest-CPU]` axis labels.
