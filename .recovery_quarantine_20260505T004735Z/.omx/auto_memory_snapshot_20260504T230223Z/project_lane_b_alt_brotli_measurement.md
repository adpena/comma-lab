---
name: Lane B-alt brotli renderer.bin saves 35KB → -0.023 score (LOCAL projection)
description: 2026-04-27 LOCAL measurement: brotli q=11 on renderer.bin (296KB → 262KB) saves 35KB which translates to -0.023 score on the 2.29 baseline. Inflate-side already supports .br auto-decompression (submissions/robust_current/inflate_renderer.py:256-273). Just needs build_baseline_archive --use-brotli flag wired through.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Verified locally 2026-04-27 (no Vast.ai cost).**

| Metric | Original renderer | Brotli q=11 renderer | Delta |
|---|---|---|---|
| renderer.bin bytes | 296,776 | 261,806 | -34,970 (-11.8%) |
| Full-res-mask archive (proj) | ~734KB | ~699KB | -35KB |
| Rate score contribution (proj) | 0.488 | 0.465 | **-0.023** |
| Total score proj from 2.29 | 2.290 | **~2.267** | -0.023 |

**Inflate-side: already supports `.br` extension** via `_decompress_brotli_in_archive(archive_dir)` at `submissions/robust_current/inflate_renderer.py:258`. The convention: name the file `renderer.bin.br` in the archive, inflate auto-decompresses to `renderer.bin` before loading.

**Compress-side wiring needed:** `experiments/build_baseline_archive.py` does NOT currently pass `--use-brotli` through to `build_submission_archive`. The submission_archive function HAS a `use_brotli=False` flag (per memory `project_quantizr_full_intel_20260421` notes). Just need to wire the flag through.

**Estimated session gain:** -0.023 score for ~10 min of code work + one $0.30 Vast.ai eval to verify. Worth shipping.

**Sequencing:** defer to AFTER Lane A lands, then stack Lane A pose-TTO gains + brotli renderer rate cut in the SAME re-run of the eval. Two single-variable changes against the 2.29 baseline (poses, renderer encoding) — both clean.

**Why this is a "free" win:** purely rate-side, no distortion change, contest-compliant (inflate-side decompression is allowed under PR #35 per `feedback_strict_scorer_rule`).

**TODO when Lane A lands:**
1. Wire `--use-brotli` flag through `build_baseline_archive.py`
2. Build archive with brotli renderer + Lane A's optimized poses
3. Run contest_auth_eval on Vast.ai 4090
4. Expect: 2.29 → ~2.27 from brotli alone OR ~0.85 if Lane A's poses also land
