# PR106/R2 PacketIR runtime-sidecar consumption proof (no-score)

**Date:** 2026-05-12
**Author:** Codex
**Status:** LANDED as non-promotable runtime-decode proof

## Summary

The prior PR106/R2 PacketIR proof accounted for every emitted `0.bin` byte at
parser level but explicitly did **not** claim runtime consumption. This pass
adds a reusable no-score proof that imports the paired submission `inflate.py`,
decodes the sidecar through the runtime's own sidecar parser/decoder, applies a
valid one-correction PacketIR mutation, and verifies that the runtime-visible
correction-array digest changes.

This is **not** full-frame inflate parity and **not** a score claim.

## Durable code surfaces

- `src/tac/packet_compiler/pr106_sidecar_packet.py`
  - `mutate_pr106_sidecar_semantic_correction(...)`
  - `pr106_sidecar_mutation_manifest(...)`
- `src/tac/packet_compiler/pr106_runtime_consumption.py`
  - `prove_pr106_sidecar_runtime_decode_consumption(...)`
  - `runtime_sidecar_correction_digest(...)`
- `tools/prove_pr106_sidecar_runtime_consumption.py`
  - thin CLI writing a proof manifest with `score_claim=false`,
    `promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

## Proof artifacts

Raw manifests are intentionally under ignored `experiments/results/`:

- `experiments/results/pr106_r2_runtime_consumption_proof.json`
- `experiments/results/pr106_r2_pr101_grammar_runtime_consumption_proof.json`

Tracked custody summary:

| archive | format | archive SHA-256 | runtime `inflate.py` SHA-256 | source correction digest | mutated correction digest | runtime decode consumed? | score claim? |
|---|---:|---|---|---|---|---:|---:|
| `submissions/pr106_latent_sidecar_r2/archive.zip` | `0x01` | `7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f` | `093d20785ff29759ac835f01efc3caa2210d31b11b5b3874256be774bc22a6db` | `7d2ff1afa765a94966d73573e57429f58522a0d324979b796f3fb05f148e78d2` | `e1baa5d1be6980d23cdd12aac5454dc9fe33574feb81ae30c418a31043f909c7` | yes | no |
| `submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip` | `0x02` | `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383` | `60055bced3ab608d0e93ba83e18fa5bc662746cfa273ad50d5960c34028d1fb3` | `7d2ff1afa765a94966d73573e57429f58522a0d324979b796f3fb05f148e78d2` | `637441f9b0518e0649c7b630ef9607c4623ea1514468a56f95467b6c6ef721fa` | yes | no |

The shared source correction digest across `0x01` and `0x02` is a same-runtime
decode-equivalence signal only. It does not substitute for full-frame parity or
exact same-runtime auth eval.

## Adversarial constraints preserved

- `score_claim=false`.
- `promotion_eligible=false`.
- `ready_for_exact_eval_dispatch=false`.
- `full_frame_inflate_output_parity_claim=false`.
- No CPU/CUDA conversion or axis inference.
- The `format_id=0x02` proof is run only against
  `submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py`, not against
  the legacy `0x01`-only runtime.

## Tests and commands

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py -q

.venv/bin/python -m ruff check \
  src/tac/packet_compiler/pr106_sidecar_packet.py \
  src/tac/packet_compiler/pr106_runtime_consumption.py \
  src/tac/packet_compiler/__init__.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  tools/prove_pr106_sidecar_runtime_consumption.py

.venv/bin/python tools/prove_pr106_sidecar_runtime_consumption.py \
  --archive submissions/pr106_latent_sidecar_r2/archive.zip \
  --runtime-dir submissions/pr106_latent_sidecar_r2 \
  --output-json experiments/results/pr106_r2_runtime_consumption_proof.json

.venv/bin/python tools/prove_pr106_sidecar_runtime_consumption.py \
  --archive submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip \
  --runtime-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --output-json experiments/results/pr106_r2_pr101_grammar_runtime_consumption_proof.json
```

Focused tests: `15 passed`.

## Remaining blocker before score or promotion language

Need one of:

1. Full-frame source-vs-candidate `inflate.sh archive_dir output_dir file_list`
   raw-output parity under the paired runtime, with output aggregate SHA; or
2. Exact same-runtime auth eval with archive/runtime custody and explicit
   `[contest-CUDA]` / `[contest-CPU]` axis labels.

Until then, PR106/R2 PacketIR sidecar work is runtime-decode-consumed but not
promotion-ready.
