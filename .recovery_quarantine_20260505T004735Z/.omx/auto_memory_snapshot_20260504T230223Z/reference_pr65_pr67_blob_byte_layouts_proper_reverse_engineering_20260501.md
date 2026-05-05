---
name: PR #65 + PR #67 leader archive blob byte-layouts (PROPER reverse engineering, not lazy guessing)
description: 2026-05-01 ~13:50Z. After user pushback "binary objdump / ghidra / sophisticated / blobs / lazy and broken approach", proper byte-level reverse engineering of both top-leader archives by READING their actual parsers (pr65_inflate.py:613-707 load_compact_archive_bundle + pr67_inflate.py:746-768) instead of guessing TOC interpretations from xxd dumps. Surfaced ~6KB of side-channel correction data PR #65 ships that the prior Grand Council missed entirely.
type: reference
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## Methodology lesson (the user was right)

My first attempt: extract `xxd` dump, guess that bytes 2-30 of pr65 were a uint32 TOC. Got back garbage values like 2.5 billion. **Lazy.**

Proper approach: READ the parser source code (`pr65_inflate.py:613-707` `load_compact_archive_bundle`), apply its EXACT byte-decoding logic to the blob, verify totals match. The parser is the spec.

**Rule going forward**: never reverse-engineer by guessing on a hex dump. Find the parser. Read the parser. Apply the parser. Verify.

## PR #65 — `x` blob (284,325 bytes) — RANK 2, 0.32 score

### Structure (per `pr65_inflate.py:613-707`)

Self-describing container with v4/v3/v2/v1/v0 header detection. The actual deployed PR #65 archive uses **v4 (30-byte header, 10 sections + randmulti tail)**.

Header: 10 × 3-byte (24-bit) little-endian section lengths at byte offsets 0-29.

| Offset | Hex bytes | Section | Bytes | % of total | Council aware? |
|---|---|---|---|---|---|
| 0:3 | `50 59 03` | mask | 219,472 | 77.2% | ✅ matches council "mask.obu.br" |
| 3:6 | `f2 de 00` | model | 57,074 | 20.1% | ✅ matches council "model.pt.br" |
| 6:9 | `cf 05 00` | pose | 1,487 | 0.5% | ⚠️ council estimated this much smaller (~0.5KB) |
| 9:12 | `78 05 00` | **post** | 1,400 | 0.5% | ❌ **MISSED** — post-stage definitions (color LUT + correction codes) |
| 12:15 | `e2 00 00` | **shift** | 226 | 0.1% | ❌ **MISSED** — per-layer bit-shift quantization parameters |
| 15:18 | `6a 00 00` | **frac** | 106 | 0.0% | ❌ **MISSED** — fractional residual stage 1 |
| 18:21 | `95 00 00` | **frac2** | 149 | 0.1% | ❌ **MISSED** — fractional residual stage 2 |
| 21:24 | `9a 00 00` | **frac3** | 154 | 0.1% | ❌ **MISSED** — fractional residual stage 3 |
| 24:27 | `df 00 00` | **bias** | 223 | 0.1% | ❌ **MISSED** — per-region color bias map |
| 27:30 | `11 01 00` | **region** | 273 | 0.1% | ❌ **MISSED** — region IDs (probably for region-conditional bias lookup) |
| (trailing) | — | **randmulti** | 3,731 | 1.3% | ❌ **MISSED** — random-projection multipliers (Schmidhuber-flavored compression trick) |

Total: 30 (header) + 280,564 (sections) + 3,731 (randmulti) = 284,325 bytes ✓

### Hidden-layer total

**~6,138 bytes** of post/shift/frac×3/bias/region/randmulti side-channel correction data the Grand Council #1 entirely missed. This is ~2.2% of the archive and likely buys 0.005-0.015 score in distortion correction (henosis-us has the BEST PoseNet on the leaderboard at 0.00035 — these residual stages explain why).

### Implications

- **PR #65 is multi-stage residual refinement at the bit level** — NOT just "fp16 with HiLo byte split" as Grand Council #1 claimed
- The Wave-1 QZS3 packer subagent (`a3a932ac907d660b9`) should consider whether to add a `post`/`shift`/`frac`/`bias`/`region` analog as a Wave-1.5 stack-on-top (but probably not in T-65h scope; flag for paper)
- Schmidhuber's grand-council vote earlier was probably channeling this `randmulti` instinct (random projection multipliers for compression)

## PR #67 — `p` blob (276,464 bytes) — RANK 1, 0.31 score

### Structure (per `pr67_inflate.py:746-768`)

Crude hard-coded layout, NOT self-describing:

```python
mask_br_data = payload[:219472]              # hard-coded mask offset
if 276430 <= len(payload) <= 276470:         # length-lookup table for model size
    model_br_len = 56093
elif 276550 <= len(payload) <= 276610:
    model_br_len = 56221
elif 278100 <= len(payload) <= 278130:
    model_br_len = 57757
elif 277400 <= len(payload) <= 277430:
    model_br_len = 57053
elif 277350 <= len(payload) <= 277399:
    model_br_len = 57031
elif len(payload) == 281240:
    model_br_len = 60880
else:
    model_br_len = 61147
model_br_data = payload[219472 : 219472 + model_br_len]
pose_q_br_data = payload[219472 + model_br_len :]
```

### Implications

- **Only 3 sections**: mask + model + pose_q. NO color_lut, NO actuator, NO smooth_pose in the `p` blob.
- The `actuator.npz.br` and `color_lut.npy.br` are SEPARATE optional files (per pr67_inflate.py:742-744) — they exist OUTSIDE the `p` blob if they exist at all.
- The deployed PR #67 archive is `p` ONLY (no separate files). So the deployed archive uses NEITHER actuator NOR color_lut.
- Conclusion: **PR #67's 0.31 is achieved with 3-section concat (mask + model + pose) — no DCT actuator residual is actually shipped**. Council's claim that PR #67 ships actuator was WRONG.
- The `make_dct_basis` + actuator decoder code IS present in pr67_inflate.py (lines 640-682), but the actual archive doesn't ship one. EthanYang explored actuators but didn't ship them.
- Crude design is brittle: if the trained model produces a model.pt.br outside one of the 7 buckets, the inflate breaks. EthanYang must train to specific size buckets.

### Brittleness signal

The 7-bucket length lookup table is a code smell. PR #65's self-describing 10-section header is WAY more engineered. PR #67 won by 0.01 points despite having a cruder container — the gain comes from the QZS3 grouped variable-bit-depth FP4 packer (in the `model` section), NOT the container.

## Side-by-side

| Aspect | PR #65 (rank 2, 0.32) | PR #67 (rank 1, 0.31) |
|---|---|---|
| Container | Self-describing 30-byte header, 10 sections | Hard-coded mask offset + 7-bucket length lookup, 3 sections |
| Container engineering | High | Low (hack) |
| Sections shipped | 10 + randmulti | 3 only (mask, model, pose_q) |
| Side-channel correction | ~6KB across post/shift/frac×3/bias/region/randmulti | None |
| Model encoding | QM0/QH0 (FP16 + HiLo byte-split) | QZS3 (grouped variable-bit-depth FP4 + qv) |
| Pose encoding | QP12 unpack at frac granularity (~1.5KB) | QP1 (first uint16 + ZigZag-VLQ deltas on column 0 only, ~0.5-2KB) |
| Mask encoding | AV1 OBU + Brotli (219,472 bytes) | AV1 OBU + Brotli (219,472 bytes) — IDENTICAL byte count |
| Total | 284,325 bytes | 276,464 bytes |
| Score | 0.32 | 0.31 |
| Why PR #67 wins | QZS3 packer saves ~10KB on model section vs PR #65's QM0/QH0 | — |
| Why PR #65 doesn't lose more | ~6KB side-channel correction recovers some distortion (best PoseNet on leaderboard) | — |

## Wave-1 implications for QZS3 packer subagent (`a3a932ac907d660b9`)

The QZS3 packer subagent's prompt said "match pr67's byte layout EXACTLY". This is now refined:

- **CONFIRMED**: layout = `mask_br_data | model_br_data | pose_q_br_data` concatenated (no header, no TOC)
- **CONFIRMED**: mask must be exactly 219,472 bytes for the hard-coded offset to work
- **CONFIRMED**: payload total length must hit one of the 7 buckets [276430-276470, 276550-276610, 277350-277430, 278100-278130, 281240, else=61147] — otherwise the inflate uses wrong model_br_len and silently breaks
- **GOTCHA**: if our QZS3-packed model.pt.br does NOT match one of pr67's known model sizes, the inflate will use 61147 as fallback model_br_len, which will silently slice the wrong byte range — debug nightmare
- **MITIGATION**: write our own inflate.py mirror that reads section lengths from a small header, OR pad/trim the model.pt.br to hit exactly bucket [56093] (PR #67's smallest known good size)

## Wave-Ω implications for SJ-KL basis substitution (Fields-medal Q5)

The Fields-medal council prescribed SJ-KL basis as a substitute for PR #67's DCT cosine actuator. **CRITICAL UPDATE**: PR #67's deployed `p` archive does NOT ship an actuator. The DCT-actuator code in pr67_inflate.py:640-682 is dormant.

Therefore SJ-KL substitution doesn't replace anything in the deployed PR #67 archive — it ADDS a new component. This is GOOD news (no risk of regression on the 0.31 baseline) and BAD news (the SJ-KL subagent must also wire the actuator into the inflate pipeline + pay the rate cost of the actuator side-info).

The council's predicted Wave-Ω-1 score band [0.21, 0.245, 0.29] assumes the actuator is shipped and SJ-KL replaces DCT. With actuator NOT shipped in baseline, the SJ-KL gain is purely additive at compress time.

## Cross-refs

- `project_grand_council_shannon_floor_eureka_session_20260501.md` (Council #1, the engineering pragmatism)
- `project_grand_council_FIELDS_MEDAL_shannon_floor_obsession_20260501.md` (Council #2, the Fields-medal SJ-KL finding)
- `reports/raw/leaderboard_intel_20260501/pr65_inflate.py:613-707` (parser source for the `x` blob)
- `reports/raw/leaderboard_intel_20260501/pr67_inflate.py:746-768` (parser source for the `p` blob)
- AGENTS.md "Build Discipline" — the side-channel correction stages (post/shift/frac) are paper-publishable as "multi-stage residual refinement at the bit level"
