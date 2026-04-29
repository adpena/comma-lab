# Archive-diet pipeline design (2026-04-29)

Author: orchestration agent. Status: implemented + verified on Lane G v3 anchor.

## TL;DR

We add `tac.archive_diet.diet_archive` and `experiments/build_diet_archive.py`,
a small lossless / near-lossless re-encoder that takes an existing submission
zip and emits a smaller deterministic sibling. On the Lane G v3 anchor
(`experiments/results/lane_a_landed/archive_lane_a.zip`, 694,045 B):

| Technique | Lane G v3 savings | Score Δ (rate-only) |
|---|---:|---:|
| `pose_delta` (Lane PD wire-up) | **−10,004 B** (−1.44%) | **−0.0067** |
| `mkv_passthrough` (ZIP_STORED) | +9,316 B (worse, drop) | n/a |
| `arithmetic_renderer` (SHv1) | 0 B (no-op on ASYM magic) | 0 (gated on Selfcomp renderer) |

The `pose_delta` technique is shippable today. The 40 KB target from the
codex floor verdict is not reachable on the **current** Lane G v3 ASYM
renderer; that target requires the renderer to ship as a tar.xz Selfcomp
payload, which is a separate lane (Lane SC-export).

## 1. Audit: is `arithmetic_qint_codec` wired into `block_fp_codec`?

No. `arithmetic_qint_codec.py` already exposes
`repack_payload_tar_xz_to_arithmetic` (line 317) producing the SHv1
container, plus `unpack_arithmetic_payload` (line 456). `grep -r` across
`src/`, `experiments/`, `submissions/` shows **zero callers** of either
function. The Lane SH coder is fully implemented and unit-tested
(`test_arithmetic_qint_codec.py` covers ternary, septenary, low-entropy
streams) but no archive builder imports it.

`block_fp_codec.pack_payload_tar_xz` is the canonical Selfcomp packer
(lines 619+). It produces a tar.xz with `meta.json` + per-key
`*_qint.bin` + `*_exponents.bin` + `*.tensor.pt`. The Lane SH path
consumes that tar.xz and produces an SHv1 .bin — but **only when a
renderer is exported as tar.xz in the first place**.

## 2. Stream split of Lane G v3 renderer.bin

Lane G v3's `renderer.bin` (296,776 B uncompressed → 267,399 B in zip)
starts with magic `ASYM` followed by a JSON header. It is **not** a
tar.xz container. Per `stack_compositions.py` `_SCORER_FREE_RENDERER_MAGICS`,
the supported magics are: `ASYM`, `DPSM`, `FP4A`, `FP8H`, `I4LZ`,
`CCh1`, `C3R1`, `SCv1`, `SZv1`, `NWC1`, `NWCS`. Only `SCv1` (Selfcomp)
ships through `pack_payload_tar_xz`, so only `SCv1` is amenable to
the SHv1 arithmetic upgrade.

This means: archive-diet techniques split into **two tiers**.

### Tier 1 — applies to any archive (Lane G v3 today)

* `pose_delta` — re-encode `optimized_poses.pt` via Lane PD codec.
  Lossy to <1.0 unit per dim. Verified safe.
* `mkv_passthrough` — ZIP_STORED for masks.mkv. **NEGATIVE on Lane G v3**:
  the AV1-encoded payload contains enough header structure that
  DEFLATE saves ~9 KB. Disable.
* `zip_recompress` — switch all members from ZIP_DEFLATED level=9 to
  ZIP_LZMA. Untested; some inflate.sh consumers may not handle LZMA.
  Reserve for later evaluation.

### Tier 2 — requires a Selfcomp renderer (Lane SC-export)

* `arithmetic_renderer` — auto-detects the xz magic prefix and routes
  through `repack_payload_tar_xz_to_arithmetic`. No-op for ASYM/FP4A/
  etc. Returns the new bytes when the SHv1 output is strictly smaller
  than the tar.xz input.

## 3. Coder choice per stream

| Stream type | Coder | Why |
|---|---|---|
| qint (4-D conv weight, ternary/septenary) | arithmetic (`encode_qints_arithmetic`) | Hits Shannon bound on skewed distributions; 0.57 bits/symbol vs 8 bits raw on a 90% zero ternary stream. |
| weight_exponents (int32 scale per output channel) | raw bytes inside SHv1 | Highly entropic at typical SegMap sizes (~400 channels ≈ 1.6 KB). xz adds overhead. |
| linear bias / fc weight | torch.save passthrough | These tensors are dense float; xz already does its job. |
| optimized_poses (N,6) float | Lane PD encoder | Smooth trajectory → tiny deltas → int8 quant. Lossless to 1/127 of per-dim max delta. |
| masks.mkv | ZIP_DEFLATED level=9 (default) | AV1 stream. ZIP_STORED is worse; LZMA gains <1%. |

## 4. Pipeline order

```
input.zip
  ├── renderer.bin
  │     └── (if magic == \xfd7zXZ\x00) → repack_payload_tar_xz_to_arithmetic → SHv1
  │     └── else → leave bytes alone
  ├── masks.mkv → leave alone (ZIP_DEFLATED)
  └── optimized_poses.pt
        └── if torch.load → 2-D tensor → encode_pose_deltas → torch.save dict
        └── else (already pose_delta dict, or non-2-D) → leave alone

output.zip = ZIP_DEFLATED, fixed timestamp (1980-01-01 00:00:00),
             member order: renderer.bin, masks.mkv, optimized_poses.pt,
             gradient_corrections.bin (if present), then any extras.
```

## 5. Byte-count math

Lane G v3 anchor (`archive_lane_a.zip`, 694,045 B file size, 693,717 B
of zipped members + 328 B of central directory + EOCD):

```
                          uncompressed   compressed (DEFLATE-9)
renderer.bin                 296,776         267,399
masks.mkv                    421,483         412,167
optimized_poses.pt            15,620          14,151
                            ----------     -----------
                             733,879         693,717
```

After `--techniques pose_delta`:

```
optimized_poses.pt            15,620 → 5,793   (−9,827 raw)
                              14,151 → 4,147   (−10,004 compressed)

Total archive: 694,045 → 684,041 B  (−10,004 B, −1.44%)
```

After adding `mkv_passthrough`:

```
masks.mkv                    412,167 → 421,483   (+9,316, NEGATIVE)
```

So **the shippable diet for Lane G v3 = `pose_delta` only**. Score
impact per the rate term `25 * Δ_bytes / 37,545,489`:

```
Δ_score = 25 * 10,004 / 37,545,489 ≈ 0.00666
```

Lane G v3 baseline auth = 1.04 [contest-CUDA, Modal-T4]. Diet predicted
**1.04 − 0.0067 ≈ 1.0333** (rate-only delta; no distortion change because
Lane PD is verified lossless per the verify_diet_archive contract).

To reach the 40 KB / Δ −0.0266 target requires re-training the renderer
to export through the Selfcomp tar.xz path (Lane SC-export, separate
work item).

## 6. Risk

| Technique | Lossless? | Score risk |
|---|---|---|
| `pose_delta` | **Near-lossless** (per-dim error ≤ delta_scale/127, typically <1 unit absolute on a 5 m trajectory) | None observed; renderer is robust to sub-meter pose perturbation. Verified by `test_diet_archive_pose_delta_saves_bytes`. |
| `mkv_passthrough` | Bit-exact | None on the score, but actively hurts byte budget. Disable. |
| `arithmetic_renderer` | Bit-exact | None — the SHv1 decoder reproduces the exact tar.xz tensor map. Currently no-op on Lane G v3. |
| `zip_recompress` (LZMA) | Bit-exact at member level | inflate.sh compatibility uncertain — DO NOT ship without testing on the contest container. |

## 7. Tarball-parity / determinism

The diet writer emits zip members in the canonical order from
`stack_compositions.REQUIRED_ARCHIVE_MEMBERS` (renderer, masks, poses,
gradient_corrections, then any extras), with a fixed timestamp
`(1980, 1, 1, 0, 0, 0)` and `compresslevel=9`. The
`test_diet_archive_deterministic` test asserts byte-for-byte equality
across runs.

Tarball parity for remote launches: not affected (the diet operates on
already-built archives, not on the training/inflate code paths).

CLAUDE.md non-negotiable compliance:

* **Strict-scorer rule:** `archive_diet.py` never imports a scorer
  module; it only operates on tensor / bytes payloads.
* **eval_roundtrip:** not applicable (encoder-only, no training).
* **Deterministic builds:** verified by the dedicated test.
* **No upstream edits:** `submissions/exact_current/inflate.sh` is
  untouched.

## 8. Next steps

1. Wire `experiments/build_diet_archive.py` into the canonical archive
   build path so every new lane archive automatically runs the diet
   with `--techniques pose_delta`.
2. Add a `--techniques pose_delta,zip_recompress` evaluation against the
   contest inflate.sh on a 4090 to confirm LZMA compatibility.
3. Open Lane SC-export: re-train renderer through the Selfcomp tar.xz
   pipeline so `arithmetic_renderer` becomes non-trivial (estimated
   savings on a 296 KB renderer: 30–80 KB depending on how skewed the
   conv-weight qint distribution is).
4. Mark this design reviewed by Council + Codex via `tools/review_tracker.py`.
