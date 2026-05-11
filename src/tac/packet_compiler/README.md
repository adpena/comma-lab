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
    # PR81 — Quantizr FP4 codebook + ROUTER_ACTION
    FP4Codebook,
    PR81_POS_LEVELS,
    encode_router_actions,
    decode_router_actions,
    pack_nibbles,
    unpack_nibbles,
    # PR84 — adaptive-context range coder
    AdaptiveContextSpec,
    encode_adaptive_context_stream,
    decode_adaptive_context_stream,
    # PR91 — universal AC wrapper + QMQH grammar
    MAGIC_QM0,
    MAGIC_QH0,
    QMQHHeader,
    encode_categorical_stream,
    decode_categorical_stream,
    emit_qmqh_header,
    parse_qmqh_header,
    pack_hi_lo_split,
    unpack_hi_lo_split,
    # PR92 — RMC1 / RSA1 / RSB1 joint-stream meta-codec
    MAGIC_RMC1,
    MAGIC_RSA1,
    MAGIC_RSB1,
    RMC1Composite,
    RSA1Side,
    RSB1Side,
    pack_rmc1_composite,
    unpack_rmc1_composite,
    pack_rsa1_side,
    unpack_rsa1_side,
    pack_rsb1_side,
    unpack_rsb1_side,
    # PR93 — delta-varint pose codec + QZMB1 grammar
    MAGIC_POSE_DV,
    MAGIC_MODEL_COMPACT,
    DeltaVarintPoseStream,
    QZMB1Block,
    encode_delta_varint_pose,
    decode_delta_varint_pose,
    pack_qzmb1_block,
    unpack_qzmb1_block,
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

### Delta-varint pose codec (PR93) — pose-axis #1 EV/byte

Encode a pose tensor as fp32 lo/scale plus signed-varint cumulative deltas
under the `QZPDV1` magic. Pose axis is 2.79x higher marginal value than
SegNet at the PR106 r2 operating point:

```python
import numpy as np
from tac.packet_compiler import encode_delta_varint_pose, decode_delta_varint_pose

poses = np.random.uniform(-0.5, 0.5, size=(600, 6)).astype(np.float32)
stream = encode_delta_varint_pose(poses)
recovered = decode_delta_varint_pose(stream.payload)
```

### PR81 FP4 codebook (asymmetric 8-level + sign)

Quantizr's `[0, 0.5, 1, 1.5, 2, 3, 4, 6]` non-negative codebook plus a sign
bit; 4-bit nibbles packed two per byte:

```python
import numpy as np
from tac.packet_compiler import FP4Codebook, pack_nibbles, unpack_nibbles

cb = FP4Codebook()  # uses PR81's asymmetric levels by default
values = np.random.uniform(-3, 3, size=64).astype(np.float32)
scales = np.array([1.5, 0.75], dtype=np.float32)  # one per 32-value block
nibbles = cb.quantize(values, scales=scales, block_size=32)
packed = pack_nibbles(nibbles)
recovered = cb.dequantize_from_nibbles(
    unpack_nibbles(packed, count=nibbles.size),
    scales=scales, block_size=32, n_values=values.size,
)
```

### PR81 ROUTER_ACTION packing (3-bit per frame)

Pack 600 frame-level small-integer router decisions into 225 bytes:

```python
import numpy as np
from tac.packet_compiler import encode_router_actions, decode_router_actions

actions = np.random.randint(0, 8, size=600).astype(np.uint8)
packed = encode_router_actions(actions, bits=3)  # 225 bytes
recovered = decode_router_actions(packed, count=600, bits=3)
```

### PR84 adaptive-context range coder

Per-symbol distribution selected by a context id; the smallest reusable
primitive that compresses correlated streams under a small finite set of
context distributions:

```python
import numpy as np
from tac.packet_compiler import (
    AdaptiveContextSpec, encode_adaptive_context_stream, decode_adaptive_context_stream
)

cdf = np.array([[0.85, 0.05, 0.05, 0.05],
                [0.05, 0.05, 0.85, 0.05]], dtype=np.float64)
spec = AdaptiveContextSpec(alphabet_size=4, cdf_table=cdf)
context_ids = np.tile(np.array([0, 1]), 100).astype(np.int32)
symbols = np.where(context_ids == 0, 0, 2).astype(np.int32)
payload = encode_adaptive_context_stream(symbols, context_ids, spec)
recovered = decode_adaptive_context_stream(payload, context_ids, spec)
```

### PR91 universal AC wrapper (per-symbol probabilities)

Generalises PR103's merged-range-stream to the case where each position has
its own distribution (HPACMini-style):

```python
import numpy as np
from tac.packet_compiler import encode_categorical_stream, decode_categorical_stream

probs = np.full((200, 8), 0.02)
probs[:, 0] = 0.86  # peaked at symbol 0
probs /= probs.sum(axis=1, keepdims=True)
symbols = np.zeros(200, dtype=np.int32)
payload = encode_categorical_stream(symbols, probs)
recovered = decode_categorical_stream(payload, probs)
```

### PR91 QM0 / QH0 grammar + hi-lo split

```python
from tac.packet_compiler import (
    emit_qmqh_header, parse_qmqh_header, pack_hi_lo_split, unpack_hi_lo_split,
)
header = emit_qmqh_header(hilo_split=True)  # b"QH0"
permuted = pack_hi_lo_split(b"\x00\x11" * 32)
recovered = unpack_hi_lo_split(permuted)
```

### PR92 RMC1 / RSA1 / RSB1 joint stream

Frame two correlated streams as ONE composite payload (vs separately):

```python
import numpy as np
from tac.packet_compiler import (
    pack_rmc1_composite, unpack_rmc1_composite,
    pack_rsa1_side, unpack_rsa1_side,
    pack_rsb1_side, unpack_rsb1_side,
    encode_router_actions,
)

# RMC1: joint mask + side-action
composite = pack_rmc1_composite(seg_bytes=b"...mask logits...", side_bytes=b"...actions...")
parsed = unpack_rmc1_composite(composite.payload)

# RSA1: range-coded side-action
body = encode_router_actions(np.array([0, 1, 2, 3], dtype=np.uint8), bits=3)
side = pack_rsa1_side(count=4, action_bits=3, table_id=0, body=body)

# RSB1: Brotli-fallback side-action
actions = np.random.randint(0, 256, size=300).astype(np.uint8)
side = pack_rsb1_side(actions=actions, table_id=7)
```

### QZMB1 compact-model framing (PR93)

```python
from tac.packet_compiler import pack_qzmb1_block, unpack_qzmb1_block
block = pack_qzmb1_block(
    block_size=32,
    arch_config_json=b'{"hidden": 64}',
    body=b"...tensor records...",
)
parsed = unpack_qzmb1_block(block.payload)
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

Thirteen golden vectors land alongside this README:

| File | What it pins | Constriction-sensitive? |
|---|---|---|
| `golden_vectors/centered_delta_uint8_v1.json` | LZMA bytes for a pinned 40×6 latent block | No |
| `golden_vectors/split_brotli_self_delim_v1.json` | Concatenated Brotli bytes for three pinned sub-streams | Brotli only |
| `golden_vectors/ranked_no_op_sidecar_v1.json` | Sidecar bytes for a pinned 24-pair, 8-dim pattern | No |
| `golden_vectors/merged_range_stream_v1.json` | Range-coded bytes for three pinned tensors | Yes |
| `golden_vectors/latent_hi_arithmetic_v1.json` | AC payload for 1000 pinned uint16 latents | Yes |
| `golden_vectors/pr81_fp4_codebook_v1.json` | 64-value FP4 codebook quantise+pack bytes | No |
| `golden_vectors/pr81_router_action_v1.json` | 225-byte 600×3-bit packed router actions | No |
| `golden_vectors/pr84_adaptive_mask_context_v1.json` | AC payload for 256 symbols × 4 raster contexts × 5 classes | Yes |
| `golden_vectors/pr91_arithmetic_coder_constriction_v1.json` | AC payload for 200 symbols under per-symbol probs | Yes |
| `golden_vectors/pr91_qmqh_grammar_v1.json` | QH0 magic + hi-lo-split 64-byte body | No |
| `golden_vectors/pr92_rmc_joint_stream_v1.json` | RMC1 framing of `seg + RSA1(120×3-bit actions)` | No |
| `golden_vectors/pr93_delta_varint_pose_v1.json` | QZPDV1 16×4 pose at 1/255 scale | No |
| `golden_vectors/pr93_qzmb1_v1.json` | QZMB1 framing with 42-byte arch-config JSON + 64-byte body | No |

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
3. Wire PR93 delta-varint pose codec into PR106 r2's pose sidechannel
   (pose-axis #1 marginal value at the current operating point per CLAUDE.md
   "SegNet vs PoseNet importance" rule).
4. Wire PR81 FP4 codebook + ROUTER_ACTION into the rate-axis optimise pass
   for Quantizr-family substrates.
5. Wire PR91 universal AC + QMQH grammar + PR84 adaptive-context coder into
   the per-axis mask-codec optimise pass.
6. Wire PR92 RMC1 joint-stream pattern into the cross-stream meta-codec when
   two correlated archive streams (pose-residual + flow-residual; sidecar
   latent + per-pair delta) appear together.
7. Add a `_v2` adaptive Brotli search that uses Bayesian-EI on prior
   `(lgwin, quality, size)` triples rather than monotone sweep.
8. Promote Rust/Zig ports once the golden vectors stabilise across two
   consecutive contest releases. The `runtime-rs/crates/tac-packet-compiler/`
   crate's coverage gate already enforces every golden vector has a paired
   Rust parity test stub.

## Composition example — full stack

Each primitive is pure-function over `bytes`/`np.ndarray`, so any subset
composes orthogonally. A typical Quantizr-family archive might bind:

```
PR81 FP4 codebook    ─→ weight nibbles ─→ PR91 QH0 hi-lo split ─→ Brotli ─→ archive member 1
PR93 delta-varint    ─→ pose payload                                ──────→ archive member 2
PR84 adaptive context ─→ mask AC payload                            ──────→ archive member 3
PR81 ROUTER_ACTION    ─→ 225-byte action stream                     ──────→ archive member 4
PR101 ranked sidecar  ─→ per-pair correction sidecar                ──────→ archive member 5
                        (or PR92 RMC1 wrapping multiple correlated sidecars)
```
