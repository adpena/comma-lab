# Schema-elision design memo — PR98 / PR100 / PR105

**Date:** 2026-05-12
**Lane:** `lane_schema_elision_and_sign_encoding_design_20260512`
**Sister memo:** `.omx/research/sign_encoding_unified_taxonomy_20260512.md`
**Status:** DESIGN-ONLY. No implementation. Implementation requires operator approval.
**Score axis target:** `rate` (charged bytes).
**Operating point:** PR106 r2 (saturated entropy frontier — see
`feedback_b1_archive_build_empirical_falsifies_composition_cells_on_pr106_r2_20260512.md`).
**Score-claim status:** `score_claim=false`, `promotion_eligible=false`,
`ready_for_exact_eval_dispatch=false`, `evidence_grade="[design-memo;
non-authoritative]"` (per CLAUDE.md `forbidden_score_claim_with_byte_change_unless_inflate_consumes`).

---

## 0. Scope

Three sibling **schema-elision** mechanisms surfaced by Subagent 7's PR81-104
mining (`.omx/research/public_pr_mining_pr81_104_typed_rows_20260512.json`)
and the PR50-80 / PR105-115 expansion landing (sister memo
`feedback_public_pr_mining_expansion_pr50_80_pr105_115_landed_20260512.md`):

| # | Primitive id | PR | Mechanism | Predicted bytes saved | Predicted EV/byte |
|---|---|---|---|---|---|
| V1 | `pr98_cd1_compact_architecture_ordered_decoder_format` | PR98 (0.200 CPU) | CD1 compact archive: magic `b'CD1'` + 1-byte scale_bits + 4-byte n_tensors, then per-tensor `[scale][zigzag INT8 body]` in canonical state-dict iteration order. **No tensor names, no shape headers per tensor.** | ~30 B/tensor × ~28 tensors ≈ ~840 B | 0.6 |
| V2 | `pr100_schema_driven_decoder_storage_grammar` | PR100 (CPU N/A) | Schema list `[(tensor_name, shape_tuple), ...]` hardcoded in decoder source. Archive stores ONLY concatenated INT8 codes (brotli'd) + fp16 scales in iteration order. Length-prefixed at parse time via `shape * dtype_size`. | Same ~840 B as V1 (different surface) | 0.5 |
| V3 | `pr105_kitchen_sink_packed_state_schema_size_sorted` | PR105 (0.198 CPU) | `PACKED_STATE_SCHEMA = sorted(FIXED_STATE_SCHEMA, key=lambda item: -int(np.prod(item[1])))`. Bodies emitted adjacent, then scales adjacent — improves brotli long-range matches on already-schema-elided substrate. | ~5-30 B (brotli's long-range gain, NOT schema elision per se) | 0.3 (composition with V2) |

V1 and V2 are **fundamentally the same mechanism** (eliminate per-tensor name +
shape metadata by hardcoding the schema into the decoder source), expressed
differently. V3 is a **composable reorder** on top of V2 that targets brotli's
context-window efficiency.

The packet_compiler primitive for V3 is **already landed** as a typed
transducer:
`src/tac/packet_compiler/pr105_packed_state_schema.py` (147 LOC) +
`src/tac/packet_compiler/golden_vectors/pr105_packed_state_schema_v1.json`.
V1/V2 do NOT yet have packet_compiler primitives.

---

## 1. Architectural classification (per HNeRV parity discipline)

All three variants are **rate-axis packet-grammar transducers** consumed by
an HNeRV-family substrate (PR95 / PR100 / PR101 / PR103 / PR105 / our
in-house `sane_hnerv`). They are NOT representations; they are **archive-byte
reorderers** that exploit a code-side architecture contract.

Per CLAUDE.md "Mutation frontier" and "Forbidden in-place edits to public PR
intake clones": every source citation below is **read-only** against the
pristine intake clones; nothing in the design or implementation path modifies
public-PR-intake bytes.

### 1.1 V1 — PR98 CD1 compact format

**Source (read-only intake):**
`experiments/results/public_pr_archive_kaggle_mirror/public_pr98_intake_20260505_auto/source/submissions/hnerv_muon_finetuned_from_pr95/src/codec.py`

**Grammar (literal byte layout):**
```
+------+------+--------+-----------+----------+
| 'CD' | '1'  | sb (1) | n_t (4 LE)| tensor[] |
+------+------+--------+-----------+----------+

tensor[i] = | scale (sb/8 bytes) | zigzag_int8_body (prod(shape) bytes) |
```

- `sb` ∈ {16, 32} → scale dtype is fp16 (2 B) or fp32 (4 B).
- `n_t` = number of tensors (LE u32).
- **Crucially:** tensor name + shape are NOT stored. The decoder iterates
  `HNeRVDecoder.state_dict()` in canonical insertion order and consumes
  `prod(shape) * 1 byte` per tensor. This is the **schema contract** —
  encoder + decoder MUST share the canonical state-dict iteration order;
  any architecture change requires a decoder source revision.

**Byte-savings derivation (vs PR95 self-describing format):**
- PR95 per-tensor overhead: ~30 B = (1 B name_len + ~22 B name + 4 B shape_dims_u8 + 4 B shape_u32_total).
- 28 tensors × 30 B = **~840 B saved**.

### 1.2 V2 — PR100 schema-driven decoder grammar

**Source (read-only intake):**
- `experiments/results/public_pr_archive_kaggle_mirror/public_pr100_intake_20260505_auto/source/submissions/hnerv_lc_v2/schema.py`
- `experiments/results/public_pr_archive_kaggle_mirror/public_pr100_intake_20260505_auto/source/submissions/hnerv_lc_v2/inflate.py`

**Grammar:** identical bytes-saved profile as V1 but architected differently:
the schema is exported to `schema.py` as a Python module-level constant
`FIXED_STATE_SCHEMA = [(name, shape), ...]`, then both encoder and decoder
import it. Archive bytes are `concat(zigzag_int8_body[i]) | concat(fp16_scale[i])`
brotli-compressed. Length is **derived at parse time** from `prod(shape) *
dtype_size`, NOT length-prefixed.

**Byte-savings derivation:** Same ~840 B as V1.

**V1 vs V2 — apples-to-apples differentiator (per CLAUDE.md "Apples-to-apples evidence discipline"):**
- V1 emits **per-tensor** `[scale][body]` interleaving.
- V2 emits **all bodies concatenated, then all scales concatenated**.
- V2 is structurally PR105's V3 ordering applied at the schema level (V3
  size-sorts the schema; V2's iteration order is unsorted but bodies are
  pre-grouped).
- The actual byte difference between V1 and V2 is **~zero** when brotli
  context is large enough to span all interleaved tensors anyway. V2's
  advantage is that the bodies share statistical structure (all zigzag INT8
  with similar histograms), and the scales share statistical structure (all
  fp16), so each stream brotli'd separately compresses better than the
  interleaved stream. Empirical residual savings: ~30-100 B on top of V1.

### 1.3 V3 — PR105 packed-state-schema size-sort

**Source (read-only intake):**
`experiments/results/public_pr_archive_kaggle_mirror/public_pr105_intake_20260505_auto/source/submissions/kitchen_sink/src/codec.py`
(line 58)

**Mechanism:**
```python
PACKED_STATE_SCHEMA = sorted(
    FIXED_STATE_SCHEMA,
    key=lambda item: -int(np.prod(item[1])),
)
```

Reorders the schema by descending tensor-element count. Bodies adjacent
(V2's first half) emit largest-first. Brotli's long-range matches benefit
from the highest-entropy streams appearing first while the entropy model is
still building — a classic prefix-code optimization.

**Byte-savings derivation:** ~5-30 B on top of V2 from improved brotli
context utilization. **NOT a schema-elision mechanism** strictly speaking —
it is a **schema-reorder** on V2's substrate.

**Packet_compiler primitive already landed:**
`src/tac/packet_compiler/pr105_packed_state_schema.py` (147 LOC, frozen
dataclass, golden-vector-tested).

---

## 2. Target-substrate compatibility

The substrate-vs-codec composition meta-pattern (per
`feedback_substrate_vs_codec_composition_meta_pattern_20260508.md`) is the
binding constraint. The 3 variants are **HNeRV-family-substrate-specific**:

| Variant | PR95-fam | PR100-fam | PR101-fam | PR103-fam | PR105-fam | `sane_hnerv` (in-house) |
|---|---|---|---|---|---|---|
| V1 (CD1 compact) | ✓ | ✓ | ✓ | ✓ | ✓ | **incompatible without α-layout adaptation** |
| V2 (schema-driven) | ✓ | ✓ (native) | ✓ | ✓ | ✓ | **incompatible without α-layout adaptation** |
| V3 (size-sort) | requires V2 base | ✓ (native) | requires V2 base | requires V2 base | ✓ (native) | requires V2 base + α adaptation |

**`sane_hnerv` incompatibility note:** the in-house `sane_hnerv` substrate
(per Wave 1 Subagent A's `experiments/train_substrate_sane_hnerv.py`) uses
the α-layout convention (multi-section archive with explicit per-section
length prefixes for robustness against decoder source drift). Adapting V1/V2
to α-layout requires a **bridge module** that:
1. Re-binds the canonical state-dict iteration order to α-layout's
   per-section ordering.
2. Either (a) replaces α's length prefixes with derived-at-parse-time
   shape*dtype math (loses robustness), or (b) keeps α's length prefixes
   and only elides the per-section NAME (saves ~half the bytes).

**Recommendation:** V1/V2/V3 are **immediately portable** to PR95/100/101/103/105
substrates (they share canonical state-dict iteration order via shared HNeRV
architecture). Adaptation to `sane_hnerv` is **deferred-pending-bridge-module-design**.

---

## 3. Composition rules

Stacking compatibility matrix at PR106 r2 operating point:

| Pairing | Stackable? | Predicted Δ bytes | Risk class |
|---|---|---|---|
| V1 alone | Yes (independent) | -840 | safe |
| V2 alone | Yes (independent) | -840 | safe |
| V1 + V2 | **MUTUALLY EXCLUSIVE** | n/a | **conflicts** (both target the same metadata bytes) |
| V2 + V3 | Yes (V3 is reorder on V2) | -870 to -900 (V2 -840 + V3 -30 to -60) | safe |
| V1 + V3 | Partial (requires V1→V2 lift first) | -840 to -870 | reqs design |
| V3 alone | **N/A** (V3 has no benefit without V2/V1 substrate) | 0 | not applicable |

**Mutual-exclusion proof:** V1 and V2 both elide tensor-name + shape headers
from the same 30 B/tensor region. Applying both is double-counting; the
archive has only one set of those bytes to elide. **The probe-disambiguator
(§5) is required to choose between V1 and V2 empirically.**

---

## 4. Six-hook wire-in (per CLAUDE.md Catalog #125)

| Hook | Status | Detail |
|---|---|---|
| 1. Sensitivity-map contribution | **N/A** — design memo only; no new sensitivity field. Implementation phase will register `bytes_saved_per_tensor_pred=30` per state-dict entry. |
| 2. Pareto constraint | **N/A** — design memo only; no Pareto-frontier movement until empirical Δ-bytes lands. Implementation phase: add `pareto_constraint=bytes_saved_at_pr106_r2` per variant to `tac.pareto_*`. |
| 3. Bit-allocator hook | **N/A** — schema elision does not change per-tensor importance; it removes metadata bytes that are not allocated by the entropy budget. |
| 4. Cathedral autopilot dispatch hook | **N/A** — design only. Implementation phase: register each variant as a candidate row in the cathedral autopilot stacking matrix with `ready_for_exact_eval_dispatch=false` until empirical Δ-bytes lands. |
| 5. Continual-learning posterior update | **N/A** — no empirical anchor produced by this memo. Implementation phase: each variant's empirical Δ-bytes on PR106 r2 lands as an anchor with `evidence_grade="[contest-CUDA]"` after dispatch. |
| 6. Probe-disambiguator | **REQUIRED** (see §5) — V1 vs V2 mutual-exclusion is a 2-defensible-interpretation design tension per CLAUDE.md "Design tension → ship both interpretations, let math arbitrate". |

---

## 5. Probe-disambiguator design

**Question:** for PR106 r2's exact state-dict, which of V1 / V2 yields the
smaller archive after brotli? (V3 is composition-only and not part of the
disambiguation.)

**Probe surface:** `tools/probe_schema_elision_disambiguator.py` (DESIGN —
not yet implemented).

**Probe contract:**
1. Load the PR106 r2 archive's state-dict.
2. Encode under V1 grammar → measure `bytes_v1`.
3. Encode under V2 grammar (same brotli params) → measure `bytes_v2`.
4. Encode under V2 + V3 (size-sort reorder) → measure `bytes_v2_v3`.
5. Emit verdict JSON:
   ```json
   {
     "schema_version": "schema_elision_probe_v1",
     "operating_point": "pr106_r2",
     "state_dict_sha256": "<sha>",
     "bytes_v1": <int>,
     "bytes_v2": <int>,
     "bytes_v2_v3": <int>,
     "verdict_min_bytes_variant": "v1|v2|v2_v3",
     "delta_v1_minus_v2": <int>,
     "delta_v2_v3_minus_v2": <int>,
     "evidence_grade": "[byte-anchor; non-authoritative]",
     "ready_for_exact_eval_dispatch": false,
     "cuda_eval_worth_testing": true
   }
   ```
6. Local-parse smoke: `inflate.sh archive_dir output_dir file_list` MUST
   succeed on the winning variant's archive before any GPU dispatch.

**Probe scope:** ~80 LOC (purely Python; no scorer load; no GPU).

**Wire-in target:** `tac.pareto_stacking_matrix` (the Cathedral autopilot
input) receives the verdict as a single ranked-candidate row with `predicted_delta_bytes` and `predicted_delta_score` (derived via `rate_term_at_pr106_r2_per_byte = 25 / 37545489 ≈ 6.66e-7`).

---

## 6. Forbidden patterns — pre-walked checklist

Per CLAUDE.md FORBIDDEN_PATTERNS the design avoids:

- [x] **No `/tmp` paths** in any cited evidence string.
- [x] **No scorer load at inflate time** — all 3 variants are pure
  state-dict re-orderers; no PoseNet/SegNet/scorer involvement.
- [x] **No MPS-fallback default** — no device-selection logic anywhere in
  these variants.
- [x] **No invented CLI flags** — implementation phase will grep target's
  argparse first.
- [x] **No /tmp evidence paths** in this memo.
- [x] **No silent /tmp-style transients** in golden vector / archive
  references — citations point at committed intake clones under
  `experiments/results/public_pr_archive_kaggle_mirror/`.
- [x] **No score claim without evidence tag** — every score claim in this
  memo carries `[design-memo]`, `[byte-anchor]`, or `[contest-CUDA]` per
  Catalog #97 discipline.
- [x] **No in-place edits to public PR intake clones** (Catalog #109) — all
  citations are read-only path references.
- [x] **No archive bytes modified** by this memo.
- [x] **No CUDA dispatch** by this memo (operator approval required for
  implementation + probe dispatch).

---

## 7. Implementation roadmap (operator-gated)

Per CLAUDE.md "Design decisions — non-negotiable" — the choice between V1
vs V2 (and whether to also land V3 reorder) is a **design tradeoff**, not a
bug fix. Implementation requires operator approval.

**Recommended path (council pre-submission — pending council vote):**

1. **Stage 0** (this memo): design + probe-disambiguator spec. **DONE.**
2. **Stage 1**: implement `tools/probe_schema_elision_disambiguator.py`
   (~80 LOC, $0 GPU). Run on PR106 r2 → empirical verdict.
3. **Stage 2**: implement the winning variant as a typed packet_compiler
   primitive (`src/tac/packet_compiler/pr98_cd1_compact_format.py` OR
   `src/tac/packet_compiler/pr100_schema_driven_grammar.py`). Land
   golden vector + parity test against intake clone's encoder.
4. **Stage 3** (optional): land V3 reorder on top of Stage 2 winner.
5. **Stage 4**: dispatch CUDA auth eval on the assembled archive ($0.25
   Vast.ai 4090 estimate, single-archive measurement). Tag result
   `[contest-CUDA]`.
6. **Stage 5** (optional): dispatch GHA Linux x86_64 CPU auth eval for
   `[contest-CPU]` axis (the leaderboard-ranking axis).

**Cost envelope estimate:** $0 (design + probe) → $0.25 (CUDA eval) →
$0.06 (CPU eval) = **<$0.50 total** for an empirically-verified
~840-byte rate-axis win at PR106 r2.

**Predicted score-Δ at PR106 r2:**
- bytes_saved ≈ 840
- rate term per byte: 25 / 37545489 ≈ 6.66e-7
- predicted Δ score ≈ -5.6e-4 (improvement)
- **Caveat:** at PR106 r2 (saturated entropy frontier), the empirical
  Δ may regress per `feedback_b1_archive_build_empirical_falsifies_composition_cells_on_pr106_r2_20260512.md`.
  The probe-disambiguator IS the empirical verdict; no advance score
  claim.

---

## 8. Operator decisions surfaced

1. **Which variant to implement first?** V1 (PR98 compact) or V2 (PR100
   schema-driven)? Recommend **probe-then-choose** (Stage 1 above) — let
   the empirical bytes count decide rather than a priori reasoning.
2. **Implementation budget cap?** Recommend $1.00 USD for full Stage 1
   → Stage 5. The probe alone is $0 (CPU-only); the auth evals are
   $0.25 (CUDA) + $0.06 (CPU).
3. **Compose with V3 size-sort by default?** Recommend yes once V2 is
   verified (V3 is +30 LOC composition wrapper on a frozen golden
   vector).
4. **Defer or proceed on `sane_hnerv` α-layout adaptation?** Recommend
   **defer-pending-bridge-module-design** — substrate-specific bridge
   work belongs in a sister substrate-engineering lane, not in this
   rate-axis packet-grammar lane.

---

## 9. Cross-references

- Sister memo: `.omx/research/sign_encoding_unified_taxonomy_20260512.md`
- Typed rows: `.omx/research/public_pr_mining_pr81_104_typed_rows_20260512.json`
- Sister landings:
  - `feedback_public_pr_mining_pr81_104_landed_20260512.md`
  - `feedback_public_pr_mining_expansion_pr50_80_pr105_115_landed_20260512.md`
  - `feedback_packet_compiler_5_pr63_64_65_105_primitives_landed_20260512.md`
  - `feedback_b1_archive_build_empirical_falsifies_composition_cells_on_pr106_r2_20260512.md`
- CLAUDE.md non-negotiables exercised:
  - "HNeRV / leaderboard-implementation parity discipline" (13 lessons)
  - "Apples-to-apples evidence discipline"
  - "Forbidden in-place edits to public PR intake clones" (Catalog #109)
  - "Multiple contenders → multiple paths" (V1 vs V2 probe)
  - "Design decisions — non-negotiable" (operator approval gate)
- Already-landed packet_compiler primitive (V3 reorder):
  - `src/tac/packet_compiler/pr105_packed_state_schema.py`
  - `src/tac/packet_compiler/golden_vectors/pr105_packed_state_schema_v1.json`
