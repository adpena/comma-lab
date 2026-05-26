# NSCS06 v8 trainer `cls_bytes` routing — pre-execution gate report

- **subagent_id**: `nscs06-v8-trainer-v3-wire-in-cls-bytes-routing-20260526`
- **lane_id**: `lane_nscs06_v8_trainer_v3_wire_in_cls_bytes_routing_20260526` (L0 at registration; L1 at landing)
- **measurement_utc**: 2026-05-26T18:44:00Z
- **scope**: code wire-in only; LOCAL macOS M5 MAX execution; **NO PAID DISPATCH** per operator standing "Remember all on MLX" 2026-05-26
- **horizon_class**: `plateau_adjacent` (Catalog #309; trainer plumbing serves Catalog #233 4-gate REFRESH for T3 #1335 WINNER #1 chroma_lut path)
- **predicted ΔS band**: N/A (no empirical training in this lane; downstream paired-Modal verifies)

## Premise verification (Catalog #229)

Read in full:

| Path | LOC | Key finding |
|---|---|---|
| `experiments/train_substrate_nscs06_v8_chroma_lut.py` | 1003 | `_full_main` calls `pack_archive(...)` at lines 740 (v2_procedural_seed branch) + 754 (v1_inline_lut branch); cls_full computed full-resolution at Stage 4 lines 653-668 (`cls_full = np.concatenate(cls_full_chunks, axis=0)`); grayscale lowres at Stage 5 lines 686-693 via area-mean pool with `args.grayscale_downsample` factor (default 8 → 48×64); BOTH callsites omit `cls_bytes=` kwarg → trainer EMITS v2, NOT v3 |
| `src/tac/substrates/nscs06_v8_chroma_lut/archive.py` | 660 | `pack_archive(... cls_bytes: bytes \| None = None)` (line 253); v3 schema_version branch (line 347); cls length invariant `num_pairs * grayscale_h * grayscale_w` uint8 (lines 358-369); accept-cascade enforces shape match |
| `src/tac/substrates/nscs06_v8_chroma_lut/inflate.py` | 244 | `if arc.cls_lowres is not None: cls_full = upsample(Image.NEAREST) else: cls_full = np.zeros_like(gray_full)` (lines 197-205); v3 vs v2 differs ONLY at this branch |
| `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_cls_stream_wire_in.py` | 343 | 17 tests covering pack/parse roundtrip, schema versioning, byte stability, inflate consumption, cls=0 boundary parity; canonical fixture: `num_pairs * gh * gw` uint8 array values in `range(0, NUM_SEGNET_CLASSES)` |

## The exact wire-in

Both callsites at lines 740 + 754 require `cls_bytes=<bytes>` derived from `cls_full` (full-res ground-truth SegNet argmax, already computed at Stage 4). The canonical downsample sister to inflate's `Image.NEAREST` upsample is **point-sampling** at cell centers (NEAREST downsample); this is the lossless round-trip for the "uniform-cls" boundary test the existing wire-in test suite enforces (`test_inflate_v3_with_uniform_class_matches_v2`).

Canonical derivation (sister of Stage 5 grayscale lowres pattern):

```python
# Stage 5b: per-cell SegNet class label NEAREST downsample (sister of
# inflate.py Image.NEAREST upsample; canonical round-trip pair).
# cls_full: (n_pairs, H, W) uint8 from Stage 4.
# Take the top-left pixel of each cell (point-sample at the canonical NEAREST
# anchor); the inflate-side Image.NEAREST upsample replicates per cell.
cls_lowres = cls_full[
    :,
    :h_g * args.grayscale_downsample:args.grayscale_downsample,
    :w_g * args.grayscale_downsample:args.grayscale_downsample,
]
# Shape MUST be (n_pairs, h_g, w_g) — strict invariant for v3 cls_stream.
assert cls_lowres.shape == (n_pairs, h_g, w_g), (
    f"cls_lowres shape {cls_lowres.shape} != ({n_pairs}, {h_g}, {w_g})"
)
cls_bytes = cls_lowres.tobytes()
```

Both `pack_archive(...)` callsites receive `cls_bytes=cls_bytes` (v2 branch produces v3 archive; v1 branch is REFUSED per `pack_archive`'s explicit `ValueError: cls_bytes supplied but schema_version resolved to v1/v2`).

**Decision for v1 branch (`v1_inline_lut`)**: v1 carries the full 4096-byte LUT inline → cargo-cult #5 inflate site already uses cls=0 uniform for v1 archives per the inflate's `if arc.cls_lowres is not None:` branch. v3 stacking does NOT apply to v1 by codec design (archive.py line 353 raises on v1+cls_bytes). Therefore the v1 callsite at line 754 must NOT pass `cls_bytes` — the only callsite that needs the kwarg is line 740 (v2 → v3 promotion).

## The 2-line trainer change

1. Insert lines 728-735 (before the variant if-branch) computing `cls_lowres` + `cls_bytes`
2. Add `cls_bytes=cls_bytes` to the `pack_archive` callsite at line 740 (v2 path → v3)
3. v1 callsite at line 754 is UNTOUCHED (v1 codec does not support v3 stacking)

## Verification protocol (LOCAL MACOS M5 MAX; NO PAID DISPATCH)

1. **Static**: `ruff check experiments/train_substrate_nscs06_v8_chroma_lut.py` clean
2. **Existing suite**: re-run 17 tests in `test_cls_stream_wire_in.py` — no regression expected (codec surface untouched)
3. **NEW dedicated test** at `tests/test_trainer_v3_wire_in.py`:
   - mock `_full_main` minimal-fixture call (or canonical end-to-end at very small `--max-pairs=4 --grayscale-downsample=8`)
   - assert v3 archive emitted (`int(bin_bytes[4]) == CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM`)
   - assert `Nscs06V8Archive.cls_lowres is not None` post-parse
   - assert `cls_lowres.shape == (n_pairs, h_g, w_g)` matches Stage 5 grayscale_lowres shape (sister-shape invariant)
4. **Inflate round-trip**: existing `test_inflate_v3_vs_v2_produces_different_frames_proves_cls_consumption` already proves v3 inflate path; trainer wire-in is purely upstream of the codec → inflate contract preserved

## Drift surface declaration (per NEW MLX↔CUDA bidirectional standing directive 2026-05-26)

- **CPU vs CUDA**: cls_full is computed via `torch.argmax(seg_logits, dim=1).to(torch.uint8).cpu().numpy()` at compress-time on `device=args.device`. Argmax is **deterministic** under bit-identical scorer weights + bit-identical RGB inputs. There is no MLX path in this trainer (MLX iteration lives in sister `mlx_iteration.py` module; not exercised by `_full_main`).
- **NEAREST downsample**: pure numpy slicing — deterministic, byte-stable across CPU/CUDA/MLX
- **Pillow NEAREST upsample**: pure-CPU operation in `inflate.py`; identical across substrates
- **No drift surface**: this wire-in is uint8 byte-stream plumbing only

## Canonical-vs-frontier-push decision (per NEW pushing-the-frontier-of-research-on-optimization-algorithms standing directive 2026-05-26)

- **CANON-APPLICATION**: trainer plumbing routes existing canonical surfaces (cls_full at Stage 4 already computed for LUT derivation Stage 6) through the just-landed canonical `pack_archive(cls_bytes=)` kwarg. No new algorithm; no new optimization theory; no MLX iteration depth change. The frontier work is the EMPIRICAL question (paired Modal T4 4-arm) NOT this code change.

## Sister coordination (Catalog #230)

- **Slot 2** (Z7-Mamba-2 L1 EMPIRICAL fair-shake) — disjoint substrate scope (`src/tac/substrates/z7_*`); no file overlap
- **Slot 3** (BoostNeRV BPR1 Variant B codec redesign) — disjoint substrate scope (`src/tac/substrates/boost_nerv*`); no file overlap
- **My scope**: `experiments/train_substrate_nscs06_v8_chroma_lut.py` (1 file) + `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_trainer_v3_wire_in.py` (NEW) + 2 research memos

## Catalog #340 sister-checkpoint guard

Pre-edit check: `tools/check_sister_checkpoint_before_git_add.py` PROCEED expected (no in-flight sibling on my files within 60-min window).

## Verdict

**PROCEED**. Scope is bounded to 1 trainer file (Stage 5b insert + 1 kwarg add) + 1 new test file + 2 memos. No code outside `experiments/train_substrate_nscs06_v8_chroma_lut.py::_full_main` Stage 5/6 boundary; codec + inflate surfaces UNTOUCHED (already wired by sister commit `581b7b129` + `545beb35c`).

[predicted; canonical-equation-N/A; per-substrate-symposium-pending paired Modal T4 dispatch]
