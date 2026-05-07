# Four-Way Stack Composition Contract (canonical)

**Date**: 2026-05-07
**Author**: Claude (Opus 4.7 1M)
**Trigger**: user directive "do canonicalization and composability, you are exactly right"
**Companion implementations**:
- `src/tac/codec_pipeline.py` (canonical orchestrator, commit b4562092)
- `src/tac/tests/test_codec_pipeline.py` (12 tests passing)
- `scripts/lib_auto_resume.sh` (canonical bash helper, commit 2d19ab9b)
- `.omx/research/four_way_stack_cross_paradigm_composition_manifest_20260507_claude.md` (the matrix this contract makes operational)

---

## Why this memo exists

The four-way stack {Op 1 split-Brotli + Op 2 arithmetic coding + Op 2.5
inference tuning + Op 3 apogee_int6} and the cross-paradigm wave {α/β/γ/δεζ}
were originally documented as a **matrix of additive predictions**. Predictions
are not contracts. Without a shared interface, each lane re-invents:

- how it consumes the previous lane's output,
- how it reports its byte impact,
- how it preserves bit-faithful roundtrip,
- how its decoder reconstructs the input.

The result is what May 4 race postmortem
(`feedback_may_4_hnerv_race_postmortem_20260505.md`) calls "ranked plans no
one executes": every component works in isolation, but composing them
requires hand-coded glue that gets invented per-lane.

This memo locks the canonical interface. Every codec lane lands as a
`tac.codec_pipeline.CodecOp` and composes through the
`tac.codec_pipeline.CodecPipeline` orchestrator.

---

## The canonical interface

### `CodecOp` Protocol

```python
class CodecOp(Protocol):
    name: str

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> EncodeResult: ...

    def decode(
        self,
        blob: bytes,
        *,
        op_state: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]: ...

    def validate(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> ValidationReport: ...
```

### Three rules

1. **Validate first.** `validate(state_dict)` runs before any encode. If it
   returns `passed=False`, the pipeline aborts. This is the Contrarian gate
   — every op must declare what it can and cannot encode.
2. **Self-describe.** `encode` returns an `EncodeResult` with `bytes_in`,
   `bytes_out`, `op_state` (the dict the decoder needs to invert this op),
   plus the encoded `blob`. The pipeline records all of this in the manifest.
3. **Bit-faithful roundtrip.** For any op whose `validate(sd).passed = True`,
   `decode(encode(sd))` must reproduce `sd` exactly within the op's quantization
   grid (lossy ops document their grid; lossless ops are exact).

### Wire format (`CPL1`)

```
magic   : 4 bytes  = b"CPL1"
n_ops   : u32_LE
for each op:
    name_len : u16_LE
    name     : utf-8 bytes
    state_json_len : u32_LE
    state_json     : utf-8 bytes (json-encoded op_state, sort_keys=True)
    blob_len : u32_LE
    blob     : raw bytes
```

Deterministic by construction: identical ops over identical state_dict produce
identical bytes. Test:
`test_pipeline_explicit_overrides_byte_deterministic`.

---

## Composition matrix (operational)

Each row is an op landed (or in flight). Each column is what THIS op delivers
to the NEXT op in the pipeline. Compose by chaining rows.

| Op | name | input | output blob | op_state delivered | landed |
|---|---|---|---|---|---|
| Op 1 | `pr101_split_brotli` | 28-tensor HNeRV state_dict | ~200KB split-Brotli wrapper | `effective_byte_maps: {tensor_idx: "zig"\|"negzig"\|"twos"\|"off"}` | ✅ b4562092 |
| Op 2 | `pr103_arithmetic_codec` (in flight subagent a96c7aff938701fc0) | post-Op-1 quantized weights | AC-coded blob (PR103 silver 0.195 path) | `prob_table_sha256, ac_streams[]` | ⏳ in flight |
| Op 2.5 | `pr102_inference_tuning` | inflate-time only (zero-byte) | n/a (decoder-side scale + frame-0 nudges) | `scale: 0.0095, frame0_nudges: list[float]` | ✅ adapter at b6bda794 |
| Op 3 | `apogee_intN_substrate` | quantized HNeRV trained at intN bits | int-N packed weights | `bits: 6, scale_per_tensor: list[float]` | 🔜 queued |
| Op 4 | `full_stack_orchestrator` | composes Ops 1+2+2.5+3 | final archive bytes | manifest of all child ops | 🔜 queued post-Op-2 |

### Cross-paradigm slots

| Paradigm | role | acts where in pipeline | landed |
|---|---|---|---|
| α (mask-encoder portfolio: NeRV/wavelet/VQ-VAE/grayscale-LUT) | masks.mkv replacement | parallel slot (separate from decoder weights) | 🔜 4-way bakeoff |
| β (sensitivity-aware Ω-W-V3) | weights pre-quantization | runs BEFORE Op 1 (modifies input state_dict) | ✅ landed Vast 4090 |
| γ (joint score-aware codec ADMM+Ballé+arithmetic) | replaces Op 2 with score-coupled AC | drop-in substitute for Op 2 | ⏳ design |
| δεζ (joint training + self-compress + MDL) | trains a state_dict that's natively Op-1-friendly | upstream of pipeline (training-time) | 🔜 task #307 |

### Composition rule (lockable)

For any pipeline assembly:

1. **β/δεζ run upstream of the pipeline** — they produce the state_dict that
   gets fed in.
2. **Pipeline runs ops in order** — Op 1 → Op 2 → Op 3 → ... Each op sees
   the same state_dict by default (linear-over-state-dict mode); a future v2
   pipeline may chain reconstructions for ops that re-quantize.
3. **α (mask-encoder) is a separate pipeline** — masks live in `masks.mkv`,
   not in the decoder weights. They get their own `CodecPipeline` instance
   when the bakeoff lands.
4. **Op 2.5 (PR102 tuning) is decode-side only** — does not appear in the
   encode pipeline. Lives in the inflate adapter's runtime config.

### Multiplicative not additive

Per `feedback_op1_substrate_mismatch_codec_engineering_reframe_20260507.md`,
Op 1's empirical -241 bytes on PR106 is a substrate mismatch (PR101 byte_maps
were tuned for PR101's fine-tuned weights). The four-way stack therefore
multiplies on δεζ-trained weights, not on PR106 weights:

- **PR106 substrate**: Op 1 → -241 bytes (small additive)
- **δεζ-trained substrate** (training optimizes for split-Brotli + AC + intN):
  Op 1+2+3 → predicted -7 to -10 KB (multiplicative — training and codec
  co-adapted)

The contract makes this auditable: when a δεζ training lane lands a
checkpoint, encoding it through the pipeline produces a manifest with
per-op byte savings. The savings are empirical, tagged
`[empirical:<manifest path>]`, and any score derived from them must come
from contest-CUDA replay on the resulting archive.

---

## How to add a new op

1. Implement `MyOp` with three methods (`encode`, `decode`, `validate`).
2. Make `name` unique (e.g. `"pr103_arithmetic_codec"`,
   `"apogee_int6_substrate"`).
3. Add roundtrip + validation + idempotency tests under
   `src/tac/tests/test_my_op.py`.
4. Add the op to `src/tac/codec_pipeline.py`'s `__all__` and document its
   `op_state` schema here in this memo.
5. Land via `subagent_commit_serializer.py` per CLAUDE.md non-negotiable.
6. Run a 3-clean-pass adversarial review.
7. Empirical byte impact: encode + decode a representative state_dict, record
   the manifest in `experiments/results/<op_name>_<date>/manifest.json`,
   tag the JSON `[empirical:<path>]`.

If the op proposes a score improvement, the score MUST come from
contest-CUDA replay on a pipeline-produced archive — no proxy MSE, no MPS,
no extrapolation. Any score in any artifact must carry `[contest-CUDA]` or
`[predicted-band only]` per CLAUDE.md.

---

## What this contract canonicalizes

**Before**: each lane re-invented the wrapper format, the manifest schema,
the validation gate, the byte-impact reporting, the idempotency test.

**After**: every codec lane is a `CodecOp` plus three tests. Wrapper format
is fixed (`CPL1`), manifest schema is fixed (`PipelineManifest.to_dict`),
validation gate is the Contrarian protocol method, byte-impact reporting
is the `EncodeResult` dataclass.

Failure mode this prevents: the May 4 "ranked plans no one executes"
postmortem. With the canonical pipeline, a ranked plan of ops produces an
encoded archive **as a single function call**, no per-lane glue needed.

---

## References

- Canonical orchestrator: `src/tac/codec_pipeline.py` (commit b4562092)
- Tests: `src/tac/tests/test_codec_pipeline.py` (12 tests passing)
- Composition matrix (predictions): `.omx/research/four_way_stack_cross_paradigm_composition_manifest_20260507_claude.md`
- Substrate-mismatch reframe: `feedback_op1_substrate_mismatch_codec_engineering_reframe_20260507.md`
- May 4 race postmortem: `feedback_may_4_hnerv_race_postmortem_20260505.md`
- Universal auto-resume canonical helper: `scripts/lib_auto_resume.sh` (commit 2d19ab9b)
- Forbidden re-implementing inline: CLAUDE.md "Forbidden re-implementing remote bootstrap inline"
- Forbidden score claims: CLAUDE.md "Forbidden score claims"

---

## Status

- ✅ Op 1 wired
- ⏳ Op 2 (subagent a96c7aff938701fc0)
- ⏳ Derivers (subagent a00452e1ead175a32)
- 🔜 Op 3 apogee_int6
- 🔜 Op 4 full-stack orchestrator
- 🔜 α 4-way bakeoff
- 🔜 γ AC drop-in
- 🔜 δεζ training-time
