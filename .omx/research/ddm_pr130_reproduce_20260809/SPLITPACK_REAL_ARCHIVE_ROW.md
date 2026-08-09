# Fire-queue rank 1 MATERIALIZED: real archives at −903 B / −234 B, parse-back byte-identical

`score_claim=false` · scorer-free · byte measurements from REAL built archives, not projections.

## The row

Base: PR130 reproduction `archive.zip` 191,052 B sha `0491d5df84fc70b6…`, S = 0.172141297491896447
`[contest-CUDA, DALI GT, n600]`.

| variant | archive bytes | Δ | ΔS | S (derived) |
|---|---:|---:|---:|---:|
| shipped (joint LZMA over models_raw) | 191,052 | — | — | 0.172141297 |
| lzma2-xtreme per-section (**dep-free**) | 190,818 | **−234** | −0.0001558 | **0.171985486** |
| brotli-q11 per-section | **190,149** | **−903** | −0.0006013 | **0.171540027** |

Artifacts: `/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/splitpack/archive_{brotli_q11,lzma2_free}.zip`

## Anatomy, from the real archive (not a memo)

```
archive.zip 191,052 = ZIP overhead 100 + member `p` 190,952
p           = [u32 models_bytes][ LZMA(models_raw) 73,968 ][ HPAC tokens 116,980 ]
models_raw  = 83,493 = 8 + semantic 40,252 + pose_carrier 23,054 + hpac 20,179
```

Per-section under the split:

| section | raw | brotli-q11 | lzma2-xtreme |
|---|---:|---:|---:|
| semantic | 40,252 | **35,033** (−12.9%) | 35,640 |
| pose carrier | 23,054 | 23,058 (**+4**) | 23,058 (**+4**) |
| hpac weights | 20,179 | 14,962 | 15,024 |
| +12 B framing | | **73,065** | **73,734** |

**The mechanism is de-blending, not better compression.** Semantic carries essentially the whole win;
the pose carrier is INCOMPRESSIBLE — every coder loses 4 B on it. Joint LZMA was fitting one model
across a highly-compressible section and an incompressible one and paying for the blend. Splitting
stops the drag. Corollary: pose bytes are at their entropy AS REPRESENTED; only a different
representation shrinks them, never a better coder.

## Why distortion is unchanged BY CONSTRUCTION

The parse-back was executed, not asserted: each variant's 3 streams were decoded and re-assembled
into `models_raw`, and the result is **byte-identical to the original** (sha match) with the framed
length fully consumed (`o == len(new_models)`). Token stream untouched. The decoder therefore sees
identical weights and identical tokens; d_seg and d_pose cannot move.

## OWED before this is a submission candidate

1. **Receiver change.** The shipped `inflate.py` parses ONE LZMA blob; the split format needs a
   3-stream parser. That is rule-118 FREE (generic decoder algorithm, zero counted bytes) but it is
   REAL WORK and is NOT DONE. The intake copy is read-only; the change belongs in our own receiver.
2. **No decode was run** (`inflate.py:665` needs CUDA). Bit-identity of the model bundle is
   demonstrated at the byte level; the end-to-end decode is not.
3. **The brotli variant needs brotli in the runtime tree** — a REQUIRED self-install, not a
   fallback, because the archive is already brotli-encoded (no decode-time fallback exists). The
   dep-free lzma2 row (−234 B) carries no such risk. PR100/101 precedent exists for declaring brotli;
   the trade is −669 B against one added install dependency.
4. No `evaluate.py` row. `score_claim=false`.
