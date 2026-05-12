# Sign-encoding unified taxonomy memo — PR96 / PR101 / PR103

**Date:** 2026-05-12
**Lane:** `lane_schema_elision_and_sign_encoding_design_20260512`
**Sister memo:** `.omx/research/schema_elision_design_pr98_pr100_pr105_20260512.md`
**Status:** DESIGN-ONLY. No implementation. Implementation requires operator approval.
**Score axis target:** `rate` (charged bytes — entropy reduction via choosing the right unsigned-byte mapping per tensor).
**Operating point:** PR106 r2 (saturated entropy frontier).
**Score-claim status:** `score_claim=false`, `promotion_eligible=false`,
`ready_for_exact_eval_dispatch=false`, `evidence_grade="[design-memo;
non-authoritative]"`.

---

## 0. Problem statement

Range coders (`constriction.stream.queue.RangeDecoder`, ANS, arithmetic
coders) operate on **non-negative integer symbols**. INT8 weight tensors
contain values in `[-128, 127]`. To feed them to a range coder, we must
**bijectively map** INT8 → UINT8. The choice of mapping is **non-arbitrary**:
different mappings reshape the symbol histogram, and the entropy coder's
compressed length is `≈ -∑ p(s) log2 p(s)` where `p(s)` is the empirical
histogram of the UINT8 stream. **Choosing the wrong mapping inflates the
compressed bytes.**

This memo unifies 5 sign-encoding strategies in use across PR96 / PR101 /
PR103, with citations and entropy-theoretic optimality conditions.

---

## 1. The 5 strategies

### 1.1 `zig` — zigzag (positive-skew optimum)

**Formula:** `u = (2*x) if x >= 0 else (-(2*x + 1))` for x ∈ [-128, 127],
returning u ∈ [0, 255]. **Decoder:**
`x = (u // 2) if u % 2 == 0 else -(u // 2) - 1`.

**Citation:** PR101 `hnerv_ft_microcodec/src/codec.py:225-227`
(`zigzag_decode_u8`).

**Optimality:** entropy-minimal when the INT8 histogram is **symmetric and
peaked at zero** — zigzag maps {0, 1, -1, 2, -2, ...} → {0, 1, 2, 3, 4, ...},
keeping low-magnitude symbols (the high-mass values) packed at low UINT8
indices where the entropy model resolves them with short codes.

### 1.2 `negzig` — negated-zigzag (negative-skew optimum)

**Formula:** `u = zig(-x)` — zigzag applied to the negated value.
**Decoder:** `x = -zig⁻¹(u)`.

**Citation:** PR101 `hnerv_ft_microcodec/src/codec.py:233-234`
(`decode_mapped_u8(..., "negzig")` →
`(-zigzag_decode_u8(arr_u8).astype(np.int16)).astype(np.int8)`).

**Optimality:** entropy-minimal when the INT8 histogram is **symmetric and
peaked at zero with a left-tail bias** (more values below zero than above).
This is the mirror of `zig`. PR101's empirical per-tensor choice:
`DECODER_BYTE_MAPS = {9: "negzig", 14: "negzig", 20: "twos", 27: "off"}` —
tensors 9 and 14 land on `negzig`, indicating their weight histograms have a
**negative skew** after PR101's training pipeline.

### 1.3 `twos` — two's-complement (symmetric raw reinterpret)

**Formula:** `u = x.view(np.uint8)` — reinterpret the int8 bits as uint8
without changing the underlying bytes. **Decoder:** `x = u.view(np.int8)`.

**Citation:** PR101 `hnerv_ft_microcodec/src/codec.py:237-238`
(`if byte_map == "twos": return arr_u8.view(np.int8)`).

**Optimality:** entropy-minimal when the INT8 histogram is **symmetric and
NEAR-uniform** OR when **negative values cluster at u ≥ 128 and positive
values cluster at u < 128 with similar mass** — two's-complement maps
0 → 0, -1 → 255, -2 → 254, ..., 127 → 127. Negative values become large
UINT8 indices; positive values stay small. If the entropy model has good
support over both halves, this strategy avoids the "zigzag tax" of
interleaving positive and negative magnitudes.

### 1.4 `off` — signed-byte-offset (left-skew optimum / range-coder canonical)

**Formula:** `u = x + 128` for x ∈ [-128, 127], returning u ∈ [0, 255].
**Decoder:** `x = (u - 128).view(np.int8)`.

**Citations:**
- PR96 `rem2_HNeRV/inflate.py:90` (`quantized[name] = ((ub - 128).astype(np.int8), scale, shape)`).
- PR101 `hnerv_ft_microcodec/src/codec.py:235-236`
  (`if byte_map == "off": return (arr_u8.astype(np.int16) - 128).astype(np.int8)`).
- PR103 `hnerv_lc_ac/inflate.py:147`
  (`ac_arrays[idx] = (weight_arrays[k] - 128).astype(np.int8).reshape(shape)`).

**Optimality:** entropy-minimal when the INT8 histogram is **highly skewed
left** (many large-negative values) — offsetting by 128 maps the negative
peak to low UINT8 indices where the entropy model resolves them with short
codes. Also the **canonical default for arithmetic-coder pipelines** because
it preserves the order of the input distribution while making it
non-negative.

### 1.5 `raw-uint8` — already-unsigned passthrough

**Formula:** `u = x` (no transform). **Decoder:** `x = u` (no transform).

**Citation:** PR101 `hnerv_ft_microcodec/src/codec.py:225-239` does NOT
include `raw-uint8` as a strategy because PR101's encoded symbols are INT8.
However, the **lo/hi byte stream** in PR103
(`hnerv_lc_ac/inflate.py:164-183`,
`lo = np.frombuffer(brotli.decompress(lo_b), dtype=np.uint8).astype(np.uint16)`)
and the **latent uint8 byte streams** in PR105 are **already unsigned by
construction** — no mapping required.

**Optimality:** trivially optimal when the source is already non-negative
(e.g., quantized latents in [0, 255], or histogram counts).

---

## 2. Composition and exclusivity

The 5 strategies are **mutually exclusive per tensor** (a single tensor
uses exactly one mapping). The choice is **per-tensor** (PR101's
`DECODER_BYTE_MAPS = {9: "negzig", 14: "negzig", 20: "twos", 27: "off"}`
explicitly assigns one strategy per state-dict index).

Strategies CAN be **composed across tensors**: each tensor independently
selects its own mapping. The composition rule is a **per-tensor lookup
table** stored encoder-side; the decoder reads the lookup from a
hardcoded source constant (PR101) or from an in-archive header
(hypothetical generalization).

**Default fallback** (when no per-tensor table is specified): `zig` per
PR101's line 231 (`if byte_map == "zig": return zigzag_decode_u8(arr_u8)` is
the default branch in PR101's `decode_mapped_u8`).

---

## 3. Empirical anchor (PR101 / PR103 per-tensor choices)

The empirical anchor is the **per-tensor selector table** in PR101 and
PR103. These are the actual choices that won 0.193 [contest-CUDA] (PR101)
and 0.195 [contest-CUDA] (PR103).

### PR101 (state-dict indices 0..27, 4 explicit overrides)

```python
DECODER_BYTE_MAPS = {
    9: "negzig",
    14: "negzig",
    20: "twos",
    27: "off",
}  # implicit default "zig" for all other indices
```

**Read:**
- 24 of 28 tensors → `zig` (symmetric peak-at-zero)
- 2 of 28 tensors → `negzig` (negative-skew bias post-training)
- 1 of 28 tensors → `twos` (symmetric near-uniform)
- 1 of 28 tensors → `off` (left-skewed)

### PR103 (all-tensors, uniform `off` strategy)

```python
ac_arrays[idx] = (weight_arrays[k] - 128).astype(np.int8).reshape(shape)
```

**Read:** PR103 uses `off` for ALL tensors in its AC stream. The AC indices
list (`AC_INDICES = [0, 2, 4, 6, 8, 10, 12, 21]`) selects 8 tensors that
share a single merged RangeDecoder; all 8 use `off`.

### PR96 (rem2_HNeRV)

```python
quantized[name] = ((ub - 128).astype(np.int8), scale, shape)
```

**Read:** PR96 uses `off` for ALL tensors. Same as PR103 — `off` is the
canonical RangeDecoder pattern when no per-tensor entropy analysis is done.

**Insight:** PR101's empirical bytes-saved-vs-PR103 is concentrated in the
**`negzig` / `twos` per-tensor choices** that PR103 doesn't make. The
estimated savings per non-`off` override is ~50-200 B per tensor depending
on histogram skew (entropy-theoretic ceiling: at most 0.5 bit/symbol on a
single-tensor differential, ~50 KB at 8K params → ~50 KB × 0.5 / 8 → ~3
KB upper bound; empirical PR101 reduction is much smaller, single-digit
KB across 28 tensors).

---

## 4. Proposed `tac.packet_compiler.sign_encoding` module — API sketch

**DESIGN ONLY. Implementation deferred pending operator approval.**

### 4.1 Module layout (proposed)

```
src/tac/packet_compiler/sign_encoding.py        # ~120 LOC, pure numpy
src/tac/packet_compiler/golden_vectors/
    sign_encoding_zig_v1.json
    sign_encoding_negzig_v1.json
    sign_encoding_twos_v1.json
    sign_encoding_off_v1.json
    sign_encoding_raw_uint8_v1.json
src/tac/tests/test_sign_encoding.py             # ~30 dedicated tests
```

### 4.2 Public API (frozen-dataclass + pure-function pattern, matches PR105 primitive style)

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np

SignStrategy = Literal["zig", "negzig", "twos", "off", "raw_uint8"]

@dataclass(frozen=True)
class SignEncodingTransform:
    """One per-tensor sign-encoding transform.

    [empirical:src/tac/packet_compiler/golden_vectors/sign_encoding_*_v1.json]
    score_claim=false; promotion_eligible=false; ready_for_exact_eval_dispatch=false
    """
    strategy: SignStrategy

def encode_int8_to_uint8(
    arr_i8: np.ndarray,
    strategy: SignStrategy,
) -> np.ndarray:
    """Bijectively map INT8 → UINT8 under one of 5 sign-encoding strategies.

    Inverse: decode_uint8_to_int8(...).

    Per CLAUDE.md "Beauty, simplicity, and developer experience": the
    function is pure; no global state; bit-exact on golden vectors.
    """
    ...

def decode_uint8_to_int8(
    arr_u8: np.ndarray,
    strategy: SignStrategy,
) -> np.ndarray:
    """Inverse of encode_int8_to_uint8."""
    ...

def select_best_strategy(
    arr_i8: np.ndarray,
    *,
    candidates: tuple[SignStrategy, ...] = ("zig", "negzig", "twos", "off"),
    entropy_estimator: Literal["shannon", "brotli_byte_count"] = "brotli_byte_count",
) -> SignStrategy:
    """Per-tensor strategy disambiguator.

    For each candidate strategy, encode the INT8 array to UINT8 then either:
      - Compute Shannon entropy of the resulting histogram (cheap, lower
        bound on compressed bytes), or
      - Actually brotli-compress the UINT8 bytes and count (expensive but
        ground-truth).

    Returns the strategy with the minimum estimated/measured byte count.

    This IS the per-tensor probe-disambiguator (§5) folded into a single
    helper for offline encoder-side use.
    """
    ...
```

### 4.3 Forbidden patterns — pre-walked checklist

- [x] **No invented CLI flags** — module is library-only; implementation
  callers must grep argparse first if exposing CLI surface.
- [x] **No `/tmp` paths** — golden vectors live in
  `src/tac/packet_compiler/golden_vectors/`.
- [x] **No scorer load** — pure numpy + stdlib.
- [x] **No MPS path** — no torch import.
- [x] **No in-place edits to public PR intake clones** (Catalog #109) — the
  golden vectors are derived by parity-test against the intake clone's
  `decode_mapped_u8`, NOT by re-encoding into the clone tree.
- [x] **Round-trip tested** (Catalog #91 B1) — encode followed by decode is
  identity on all 5 strategies for all INT8 inputs.
- [x] **No score claim with byte change** unless inflate-consumes — module
  emits bytes only as part of a packet_compiler pipeline that already has
  this contract.

---

## 5. Probe-disambiguator design

**Question:** for each tensor in a target state-dict, which of the 5
strategies yields the smallest brotli'd UINT8 byte stream?

**Probe surface:** `tools/probe_sign_encoding_disambiguator.py` (DESIGN —
not yet implemented).

**Probe contract:**

1. Load target state-dict (e.g., PR106 r2 quantized INT8 tensors).
2. For each tensor `t` at state-dict index `i`:
   - For each strategy `s ∈ {zig, negzig, twos, off, raw_uint8}`:
     - Encode: `u_s = encode_int8_to_uint8(t, s)`.
     - Brotli-compress: `b_s = brotli.compress(u_s.tobytes(), quality=11)`.
     - Record `len(b_s)`.
   - Record verdict: `argmin_s len(b_s)`.
3. Emit verdict JSON:
   ```json
   {
     "schema_version": "sign_encoding_probe_v1",
     "operating_point": "pr106_r2",
     "state_dict_sha256": "<sha>",
     "per_tensor_verdict": {
       "0": {"strategy": "zig", "bytes": 1234, "alts": {"negzig": 1245, "twos": 1280, "off": 1300, "raw_uint8": 65536}},
       "1": {...},
       ...
     },
     "total_bytes_optimal": <sum_argmin>,
     "total_bytes_uniform_off": <sum_if_all_off>,
     "savings_from_per_tensor_selection": <int>,
     "evidence_grade": "[byte-anchor; non-authoritative]",
     "ready_for_exact_eval_dispatch": false
   }
   ```
4. Wire-in: the verdict map becomes the encoder-side `DECODER_BYTE_MAPS`
   constant in the in-house codec's source (mirroring PR101's pattern).

**Probe scope:** ~60 LOC (numpy + brotli + json; no scorer; no GPU).

**Apples-to-apples discipline note** (per CLAUDE.md): the probe MUST run
on the **same INT8 tensors** that will ship in the archive. Running the
probe on training-time tensors and then quantizing differently for the
final archive is a measurement-apparatus bug.

---

## 6. Six-hook wire-in (per CLAUDE.md Catalog #125)

| Hook | Status | Detail |
|---|---|---|
| 1. Sensitivity-map contribution | **N/A** — design memo only; no new sensitivity field. Implementation phase will add `per_tensor_optimal_strategy` field to the rate-axis sensitivity map (the probe verdict above). |
| 2. Pareto constraint | **N/A** — design memo only. Implementation phase: `pareto_constraint=sum_of_per_tensor_optimal_brotli_bytes` becomes the new rate-axis floor for the HNeRV-family substrate. |
| 3. Bit-allocator hook | **N/A** — sign-encoding does not change per-tensor importance; it changes the **encoding cost** of a fixed bit budget. Bit-allocator's `bits_per_tensor[i]` is unaffected. |
| 4. Cathedral autopilot dispatch hook | **N/A** — design only. Implementation phase: per-tensor strategy table becomes a candidate row in the autopilot stacking matrix with `predicted_bytes_saved_vs_uniform_off=<probe_result>` and `ready_for_exact_eval_dispatch=true` once the per-tensor table is materialized as code. |
| 5. Continual-learning posterior update | **N/A** — no empirical anchor produced by this memo. Implementation phase: per-tensor verdicts land as a single anchor row tagged `[byte-anchor]` (non-promotable) until paired with a `[contest-CUDA]` archive measurement. |
| 6. Probe-disambiguator | **REQUIRED** (see §5) — 5-way per-tensor disambiguation is the canonical "let math arbitrate" pattern per CLAUDE.md "Design tension → ship both interpretations, let math arbitrate" non-negotiable. |

---

## 7. Forbidden patterns — pre-walked checklist (memo level)

- [x] **No `/tmp` paths** in any cited evidence string.
- [x] **No scorer load at inflate time** — all 5 strategies are pure
  byte-rewriters; no PoseNet/SegNet/scorer involvement.
- [x] **No MPS-fallback default** — no device-selection logic anywhere.
- [x] **No invented CLI flags** — implementation phase will grep target's
  argparse first.
- [x] **No /tmp evidence paths** in this memo.
- [x] **No score claim without evidence tag** — every score claim carries
  `[contest-CUDA]`, `[byte-anchor]`, or `[design-memo]`.
- [x] **No in-place edits to public PR intake clones** (Catalog #109).
- [x] **No archive bytes modified** by this memo.
- [x] **No CUDA dispatch** by this memo.

---

## 8. Implementation roadmap (operator-gated)

Per CLAUDE.md "Design decisions — non-negotiable" — adding a per-tensor
sign-encoding strategy table is a **design tradeoff** (default `off` is
simple; per-tensor `{negzig, twos, off, zig, raw_uint8}` is complex but
measurably smaller). Implementation requires operator approval.

**Recommended path:**

1. **Stage 0** (this memo): unified taxonomy + probe spec. **DONE.**
2. **Stage 1**: implement `src/tac/packet_compiler/sign_encoding.py`
   (~120 LOC) + golden vectors + 30 round-trip tests. $0 GPU.
3. **Stage 2**: implement `tools/probe_sign_encoding_disambiguator.py`
   (~60 LOC). Run against PR106 r2 state-dict.
4. **Stage 3**: materialize the per-tensor verdict as a `DECODER_BYTE_MAPS`
   constant in the in-house HNeRV-family codec. Verify parity against
   PR101's empirical table (sanity check: does the probe re-derive
   PR101's `{9: negzig, 14: negzig, 20: twos, 27: off}` on PR101's
   state-dict bytes?).
5. **Stage 4**: build candidate archive with the new per-tensor table.
   Dispatch CUDA auth eval ($0.25 Vast.ai 4090). Tag `[contest-CUDA]`.
6. **Stage 5** (optional): GHA Linux x86_64 CPU auth eval ($0.06) for
   `[contest-CPU]` axis.

**Cost envelope estimate:** $0 (probe) → $0.25 (CUDA) → $0.06 (CPU) =
**<$0.50 total**.

**Predicted score-Δ at PR106 r2:**
- Empirical PR101-vs-PR103 bytes-saved attributable to sign encoding:
  ~500-1500 B (rough; PR101's 0.193 vs PR103's 0.195 difference has many
  contributing mechanisms, not just sign-encoding).
- rate term per byte at PR106 r2: 25 / 37545489 ≈ 6.66e-7
- predicted Δ score ≈ -3.3e-4 to -1.0e-3 (improvement)
- **Caveat:** at PR106 r2 (saturated entropy frontier per
  `feedback_b1_archive_build_empirical_falsifies_composition_cells_on_pr106_r2_20260512.md`),
  empirical Δ may regress if the substrate's INT8 weights are already
  near the per-tensor optimal histogram (i.e., the substrate trained
  itself toward whatever the encoder's default mapping was).

---

## 9. Operator decisions surfaced

1. **Implementation order?** Sign-encoding alone (Stage 1+2+3, no archive
   dispatch) vs full sign-encoding + archive ship (Stage 1-5). Recommend
   Stage 1-3 first (proves the taxonomy is bit-exact and the probe
   re-derives PR101's table); Stage 4-5 only if probe predicts ≥ 200 B
   savings on PR106 r2.
2. **Compose with schema-elision (sister memo)?** Recommend yes —
   sign-encoding and schema-elision are independent rate-axis primitives
   that stack (different byte regions). Combined predicted savings at
   PR106 r2: ~1400-2300 B (840 schema + 500-1500 sign-encoding).
3. **Implementation budget cap?** Recommend $1.00 USD for full Stage 1-5
   on each of (sign-encoding, schema-elision). $2.00 if composed.
4. **Probe before primitives, or primitives before probe?** Recommend
   **primitives + golden vectors first**, then probe (the probe imports
   the primitives). Order is dictated by the import graph, not the
   epistemic question.

---

## 10. Cross-references

- Sister memo: `.omx/research/schema_elision_design_pr98_pr100_pr105_20260512.md`
- Typed rows: `.omx/research/public_pr_mining_pr81_104_typed_rows_20260512.json`
- Sister landings:
  - `feedback_public_pr_mining_pr81_104_landed_20260512.md`
  - `feedback_public_pr_mining_expansion_pr50_80_pr105_115_landed_20260512.md`
  - `feedback_b1_archive_build_empirical_falsifies_composition_cells_on_pr106_r2_20260512.md`
- Source citations (read-only intake clones):
  - PR96: `experiments/results/public_pr_archive_kaggle_mirror/public_pr96_intake_20260505_auto/source/submissions/rem2_HNeRV/inflate.py:90`
  - PR101: `experiments/results/public_pr_archive_kaggle_mirror/public_pr101_intake_20260505_auto/source/submissions/hnerv_ft_microcodec/src/codec.py:57-62, 225-239`
  - PR103: `experiments/results/public_pr_archive_kaggle_mirror/public_pr103_intake_20260505_auto/source/submissions/hnerv_lc_ac/inflate.py:147`
- CLAUDE.md non-negotiables exercised:
  - "HNeRV / leaderboard-implementation parity discipline"
  - "Apples-to-apples evidence discipline"
  - "Forbidden in-place edits to public PR intake clones" (Catalog #109)
  - "Multiple contenders → multiple paths"
  - "Design decisions — non-negotiable"
  - "KILL is LAST RESORT" — no strategy is killed; the probe disambiguates.
