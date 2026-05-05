---
name: AMRC mask codec — empirical NEGATIVE on rate (Yousfi #8)
description: Lossless RLE+Huffman+delta mask codec ships at 1.03MB on real masks — 2.4× larger than AV1 CRF50 (421KB). Yousfi council estimate of +0.05–0.10 score did NOT pan out.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Headline:** Yousfi council #8 (RLE+Huffman+delta lossless mask codec) was projected at +0.05-0.10 score. **Empirical result on real comma SegNet masks: AMRC = 1.03MB, vs AV1 CRF50 = 421KB. AMRC loses by 2.4× on raw rate.**

**Why the estimate was wrong:**
- The estimate assumed mostly-road masks with 95%+ same-as-prev frames would compress trivially.
- Real SegNet output has 99.14% same-as-prev pixels BUT ~1.96M intra-frame class transitions per 1200-frame sequence. Each transition needs an RLE + Huffman code.
- AV1's DCT-style intra prediction handles these transitions for nearly free (lossy of course); strict losslessness on 5-class argmax can't beat it on bytes.
- The lossless `entropy_lossless` coder we already had is 990KB — AMRC is only 3.6% smaller than it. Both lose to AV1 lossy.

**Where AMRC is still useful (don't delete it):**
1. **No LZMA dependency at inflate** — AMRC decoder is pure Python+numpy, no extra deps to ship in the contest container.
2. **Byte-identical at train/test** — no AV1 dithering noise. If a renderer is trained against AMRC-faithful masks AND can absorb the +600KB rate hit through a smaller distortion term, the lane reopens. Unproven.
3. **Architectural reference** — proves the rate floor for any lossless 5-class argmax codec, useful as a Pareto baseline.

**How to apply:**
- DO NOT use AMRC as a drop-in replacement for AV1 CRF50 in any submission archive — score gets WORSE.
- DO consider AMRC if a future renderer is sensitive to mask noise and gains >0.4 distortion (compensating the 0.4 rate cost from 421KB → 1MB).
- Council Yousfi note: this estimate was off because it assumed compressibility from temporal redundancy alone; real signal density at class boundaries dominates.

**Code on disk (committed a94b8a1a):**
- `src/tac/lossless/argmax_codec.py` — pure-Python codec, magic bytes `b"AMRC"`.
- `src/tac/mask_codec.py::detect_mask_codec` — content sniffer for `.amrc` vs `.mkv`.
- `submissions/robust_current/inflate_renderer.py::_load_masks_from_amrc` — inflate-side decoder.
- `experiments/pipeline.py::step_extract_masks` — `--mask-codec argmax_rle` option.
- `src/tac/submission_archive.py` — `RENDERER_AMRC_MANIFEST` constant + mutex with `masks_mkv`.
- 22 property tests + preflight `_validate_amrc_artifacts`.
