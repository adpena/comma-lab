# `tac.packet_compiler`

Reusable byte-grammar and entropy-coder primitives extracted from the public
PR101 (`hnerv_ft_microcodec`) and PR103 (`hnerv_lc_ac`) submissions and
generalised so they compose orthogonally over any sidecar / per-pair /
per-tensor stream — not just HNeRV.

## Why this exists

The 2026-05-11 score-lowering handoff (P4 "Deterministic packet compiler"
section) identifies `tac.packet_compiler` as the missing reusable layer: every
sub-0.20 PR rediscovers the same ranking/Huffman/arithmetic-coding tricks and
hardcodes them next to their substrate. Promoting the primitives here turns
each "stack" (PR101 ⊕ PR103 ⊕ PR106 ⊕ ...) into a composition of typed,
golden-vector-backed building blocks.

## Public surface

```python
from tac.packet_compiler import (
    # PR101 sidecar grammar
    RankedSidecarSchema,
    CenteredDeltaUint8Stream,
    SplitBrotliStream,
    encode_ranked_no_op_sidecar,
    decode_ranked_no_op_sidecar,
    encode_centered_delta_uint8,
    decode_centered_delta_uint8,
    split_brotli_self_delimiting,
    parse_split_brotli_self_delimiting,
    # PR103 arithmetic coding
    MergedRangeStream,
    WeightTensorACSpec,
    AdaptiveBrotliResult,
    encode_merged_range_stream,
    decode_merged_range_stream,
    encode_latent_hi_arithmetic,
    decode_latent_hi_arithmetic,
    adaptive_brotli_param_search,
)
```

Everything else is module-private (prefixed `_`).

## Quick examples

### Ranked Huffman/no-op sidecar (PR101)

A per-pair sparse correction stream where most pairs are *no-ops*. The
encoder ranks the no-op positions via a co-lex combination rank
(`ceil(log2(C(N,k)))` bits) and stores residual deltas under a canonical
Huffman code whose length vector is itself rank-encoded.

```python
import numpy as np
from tac.packet_compiler import RankedSidecarSchema, encode_ranked_no_op_sidecar

schema = RankedSidecarSchema(
    n_pairs=600,
    n_dims=28,
    deltas=(-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10),
)
dims = np.full(600, schema.no_op_sentinel, dtype=np.int64)
deltas = np.zeros(600, dtype=np.int64)
dims[42] = 5     # correct dim 5 of pair 42
deltas[42] = 7   # with delta index 7  (== schema.deltas[7] == -1)
payload = encode_ranked_no_op_sidecar(
    dims=dims, delta_indices=deltas, schema=schema
)
```

### Centered-delta uint8 latents under raw LZMA (PR101)

Quantise a per-column float stream to uint8, encode row 0 as absolute base
and rows 1..N-1 as centered temporal deltas, wrap in raw-LZMA bytes:

```python
import numpy as np
from tac.packet_compiler import encode_centered_delta_uint8

latents = np.random.default_rng(0).uniform(-1, 1, size=(600, 28)).astype(np.float32)
stream = encode_centered_delta_uint8(latents)
print(len(stream.lzma_bytes), "bytes after LZMA")
```

### Self-delimiting split Brotli (PR101)

Concatenate N independently-Brotli-compressed sub-streams without storing
per-stream lengths — Brotli's frame structure is enough for the parser:

```python
from tac.packet_compiler import split_brotli_self_delimiting, parse_split_brotli_self_delimiting

packed = split_brotli_self_delimiting([b"...weights...", b"...biases..."], quality=11, lgwin=22)
weights, biases = parse_split_brotli_self_delimiting(packed.payload, n_streams=2)
```

### Merged range stream (PR103)

Encode multiple weight tensors into a single range-coded byte string. Each
tensor uses its own categorical histogram but they share one constriction
range coder, removing per-stream tail overhead:

```python
import numpy as np
from tac.packet_compiler import WeightTensorACSpec, encode_merged_range_stream

shape0 = (144, 36, 3, 3)
hist0 = np.bincount(
    (np.random.randn(*shape0) * 8).astype(np.int8).flatten() + 128,
    minlength=256,
).astype(float) + 1.0
spec0 = WeightTensorACSpec(name="blocks.0.weight", shape=shape0, histogram=hist0)
# … repeat per tensor …
stream = encode_merged_range_stream(tensors, specs)
```

### Latent-hi arithmetic coding (PR103)

The 16-bit zigzag delta of a per-pair latent splits naturally into ``lo``
(uint8, Brotli) and ``hi`` (uint8, arithmetic). The hi-byte distribution is
sharply peaked at 0 so arithmetic beats LZMA:

```python
from tac.packet_compiler import encode_latent_hi_arithmetic
payload = encode_latent_hi_arithmetic(latents_uint16, histogram=hi_hist)
```

### Adaptive Brotli parameter search (PR103)

Sweep `(lgwin, quality)` under a time/eval budget, return the smallest
output:

```python
from tac.packet_compiler import adaptive_brotli_param_search
res = adaptive_brotli_param_search(raw_bytes, time_budget_s=30.0)
print(res.lgwin, res.quality, len(res.payload))
```

## Composition

All primitives are pure-function transducers over `bytes`/`np.ndarray`, so they
compose orthogonally:

```text
                                                      ┌─ split_brotli_self_delimiting ──┐
weights (int32) ─ encode_merged_range_stream ─ AC ─→ ─┤                                 │
                                                      └─ centered-delta uint8 LZMA ─────┘
                                                                                        │
                                            ranked-no-op sidecar ─────────────────────→ │
                                                                                        ↓
                                                                              archive.zip bytes
```

The order is the caller's choice; each primitive enforces only its own
contract.

## Design contract

Every primitive in this package satisfies:

* **No scorer load** — strict-scorer-rule applies on both archive and
  inflate sides. Nothing in this module imports `PoseNet` / `SegNet` /
  `FastViT` / `EfficientNet`.
* **No MPS / torch dependency** — pure numpy + brotli + lzma + constriction +
  stdlib.
* **No `/tmp` paths** — golden vectors live under
  `src/tac/packet_compiler/golden_vectors/` so a fresh checkout has access.
* **Deterministic + byte-stable** — `encode → decode` is bit-exact on
  golden vectors. Native ports (Rust/Zig/C) must match these byte-for-byte
  before promotion.
* **Frozen typed dataclasses** — every container is `@dataclass(frozen=True)`.
* **OSS-friendly** — public surface is narrow; everything else is `_`-prefixed.

## Golden vectors

Five golden vectors land alongside this README:

| File | What it pins | Constriction-sensitive? |
|---|---|---|
| `golden_vectors/centered_delta_uint8_v1.json` | LZMA bytes for a pinned 40×6 latent block | No |
| `golden_vectors/split_brotli_self_delim_v1.json` | Concatenated Brotli bytes for three pinned sub-streams | Brotli only |
| `golden_vectors/ranked_no_op_sidecar_v1.json` | Sidecar bytes for a pinned 24-pair, 8-dim pattern | No |
| `golden_vectors/merged_range_stream_v1.json` | Range-coded bytes for three pinned tensors | Yes |
| `golden_vectors/latent_hi_arithmetic_v1.json` | AC payload for 1000 pinned uint16 latents | Yes |

Native ports MUST reproduce these SHA-256 digests. Any regeneration of a
constriction-sensitive vector requires a paired version bump (`_v2`).

## CLAUDE.md compliance summary

| Non-negotiable | How this package complies |
|---|---|
| Deterministic packet compiler | Identity transducers + golden vectors + fail-closed contract checks |
| `tac` stays clean | All code in `src/tac/packet_compiler/`; thin re-exports in package `__init__` |
| Beauty, simplicity, DX | Frozen dataclasses + typed APIs + narrow surface + docstrings with PR101/PR103 references |
| No MPS authoritative | No torch / MPS imports anywhere |
| Bugs permanently fixed + self-protected | Each contract violation raises `ValueError` with a clear message; tests cover the failure modes |
| No `/tmp` paths in any persisted artifact | Golden vectors under `src/tac/packet_compiler/golden_vectors/` |

## Roadmap

This package is the **substrate** for the Phase 1 packet compiler's
`optimize` mode (see `src/tac/phase1_packet_compiler.py`). Future work:

1. Wire ranked-no-op sidecar into `phase1_packet_compiler::optimize` so that
   any sidecar caller can opt in via typed flags rather than rolling their
   own bit-packer.
2. Wire merged range stream + latent-hi AC into the same optimise pass for
   non-HNeRV weight families (NeRV/MNeRV/SIREN/Cool-Chic/C3).
3. Add a `_v2` adaptive Brotli search that uses Bayesian-EI on prior
   `(lgwin, quality, size)` triples rather than monotone sweep.
4. Promote Rust/Zig ports once the golden vectors stabilise across two
   consecutive contest releases.
