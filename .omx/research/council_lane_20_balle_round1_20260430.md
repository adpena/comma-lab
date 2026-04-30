# Lane 20 Ballé — Adversarial Review Round 1 (per CLAUDE.md "Recursive
adversarial review protocol")

**Reviewer rotation:** Yousfi (steganalysis-detector blind spots) ·
Fridrich (UNIWARD distortion-cost) · Contrarian (kill-the-lane challenger)
· Quantizr (competitor-realism) · Hotz (engineering shortcuts).

**Scope under review:**
- `src/tac/balle_hyperprior_codec.py` (production codec, 17 tests passing)
- `src/tac/tests/test_balle_hyperprior_codec.py` (Phase C test suite)
- `experiments/measure_lane_20_balle_real_archive.py` (Phase E empirical)
- `experiments/train_balle_hyperprior.py` (trainer with EMA 0.997)
- `scripts/remote_lane_20_balle.sh` (Phase D + F dispatch)
- `src/tac/preflight.py:check_balle_hyperprior_includes_side_info_in_archive`
  (Phase G STRICT Check 91, 9 tests passing)
- `src/tac/profiles.py:lane_20_balle_lane_g_v3` (Phase D registry stub)
- `.omx/research/council_lane_20_balle_design_20260430.md` (Phase A design)

**Empirical anchor:** `reports/lane_20_balle_real_archive.json` (Lane G v3
qint stream — 141 KB total, 20 conv layers; trained codec verdict
``STATIC_WINS_FALLBACK`` at 2000 steps).

---

## Yousfi — steganalysis-detector blind-spot perspective

**Question:** Can the BHv1 wire format be steganographically signed in a
way that defeats EfficientNet-B2 SegNet at inflate time? (i.e. does
re-encoding the qint stream change the bitstream of the inflated archive
in any way that the SegNet detects?)

**Verdict:** GREEN. The decoded qint values are bit-identical to the
input (verified by 17 roundtrip tests). The ONLY thing that changes is
the ARCHIVE BYTE LAYOUT — which the inflate side decompresses to the
same in-memory tensors. The SegNet input frames are downstream of the
fully-decoded renderer and depend only on `model(masks, masks_t1)`
output, which is bit-identical pre and post Lane 20. No detector signal.

**Finding 1 (LOW):** The empirical script's `_unpack_fp4_nibbles`
implementation uses a Python loop over packed bytes (~140K iterations).
Slow (~3s on Lane G v3) but not a correctness issue. Could vectorise
with `np.frombuffer + np.bitwise_and` for ~100x speedup. NOT BLOCKING.

## Fridrich — UNIWARD distortion-cost perspective

**Question:** Does the Ballé hyperprior introduce any quantisation /
rounding distortion beyond what FP4A already imposes?

**Verdict:** GREEN. Lane 20 is a LOSSLESS codec on the FP4 nibble stream.
The ENCODE step takes int8 in {-7..+7} and writes them to the BHv1
container; the DECODE step reads BHv1 bytes and returns the SAME int8
values. There is NO additional distortion injected. The FP4 quantisation
itself happens UPSTREAM (in `_quantize_block`).

**Finding 2 (MEDIUM):** The trainer (`train_balle_hyperprior.py`)
uses `_quantize_block` from `tac.fp4_quantize` to convert FP16 weights
to signed indices. This implicitly assumes `DEFAULT_CODEBOOK` is the
right codebook. For Lane G v3 anchor (which uses `--fp4-codebook=residual`
per Lane V profile audit per ``feedback_silent_default_bug_class_findings_20260429.md``),
the WRONG codebook means the qint distribution doesn't match what would
ship in production. **Action:** trainer should accept `--fp4-codebook`
flag and forward to `_fp4_quantize_to_signed_indices`. PARTIALLY-LANDED:
empirical script + trainer both currently default to DEFAULT_CODEBOOK.

## Contrarian — kill-the-lane challenger perspective

**Question:** Why are we landing Lane 20 if the empirical verdict on the
Lane G v3 anchor is `STATIC_WINS_FALLBACK`? Doesn't that mean Lane 20
is dead?

**Verdict:** YELLOW with rebuttal:
1. The CODEC infrastructure has independent value: any future heteroscedastic
   anchor (Selfcomp block-FP, IMP-pruned, NeRV mask streams) can use Lane 20.
2. The auto-fallback path is the kill criterion ENFORCED IN CODE. The fact
   that it correctly engages on the FP4 anchor is PROOF that the kill
   criterion works.
3. The STRICT Check 91 prevents the bug class where "future devs add
   side-info" but forget to ship it (the same bug debugged in Phase B at
   commit time — FP16 round-trip mismatch).
4. The 5000-step CUDA training in remote_lane Stage 2 may yet flip the
   verdict (the local 2000-step CPU run was severely undertrained).

**Finding 3 (HIGH):** The remote_lane script's Stage 3+4 (build modified
archive + auth eval) is currently a **stub for the heteroscedastic-anchor
case**. For the Lane G v3 case, Stage 2's `STATIC_WINS_FALLBACK` exits
the script BEFORE Stage 3 executes — meaning **no contest-CUDA score
exists on a Lane 20 archive yet**. This means Lane 20 is at Level 2.5
(empirical [empirical] tag on bytes; no [contest-CUDA] tag yet). To
graduate to Level 3, EITHER (a) flip the verdict on a heteroscedastic
anchor + run auth eval, OR (b) explicitly accept that Lane 20 = static
on the Lane G v3 anchor and the [contest-CUDA] tag for that case = Lane G
v3 1.05 unchanged.

## Quantizr — competitor-realism perspective

**Question:** Does Lane 20 give us ANY edge against the 0.33 leader who
uses FP4 + Brotli outer compression?

**Verdict:** YELLOW. Brotli on the FP4 nibble stream IS effectively a
generic adaptive entropy coder — it captures the same heteroscedasticity
that Lane 20's hyperprior tries to capture, with the additional advantage
that Brotli's dictionary adapts to the actual byte distribution. Lane 20
is competitive with Brotli only when:
- Brotli's dictionary overhead exceeds Lane 20's side-info, OR
- The qint distribution has structure Brotli misses (e.g. block-level
  sigma variation that's invisible to byte-level LZ matching).

**Finding 4 (MEDIUM):** No A/B vs Brotli yet. The empirical report has
raw FP4 (4 bits/elem) and static-arithmetic (~3.86 bits/elem) but NOT
Brotli (which is what Quantizr ships). **Action:** add Brotli baseline
to the empirical script. PARTIALLY ADDRESSABLE in next round.

## Hotz — engineering-shortcut perspective

**Question:** Is there a simpler way to get the same byte savings?

**Verdict:** GREEN with refinement. Hotz's own proposal (chunked-static
prior, mode 0) IS in the codec. The Phase E result shows that even
chunked-static doesn't beat single-frequency-table-static on the
Lane G v3 FP4 stream — because the Lane G v3 weights are nearly
homoscedastic (uniform-alphabet FP4 is by design uniform).

**Finding 5 (LOW):** The 256×4 z_freq table in side_info (1024 bytes) is
overhead even when z is sparse. Could be replaced by a runtime-fitted
single-byte-Laplacian prior (~10 bytes side-info). NOT BLOCKING but a
~1KB improvement worth pursuing if Lane 20 ever wins on a heteroscedastic
anchor.

---

## Round 1 summary

| Finding | Severity | Status |
|---|---|---|
| 1. Slow `_unpack_fp4_nibbles` Python loop | LOW | DEFERRED (perf only) |
| 2. Trainer ignores `--fp4-codebook=residual` | MEDIUM | **OPEN** |
| 3. Remote_lane Stage 3+4 stub for heteroscedastic anchor | HIGH | **OPEN** |
| 4. No Brotli baseline in empirical | MEDIUM | DEFERRED to next round |
| 5. z_freq table 1024B overhead | LOW | DEFERRED |

**Round 1 verdict:** 2 OPEN findings (1 MEDIUM + 1 HIGH). Round 1 is
**NOT CLEAN**. Counter resets to 0/3.

**Required action before Round 2:**
- Address Finding 2: trainer + empirical script accept `--fp4-codebook`
  flag and forward correctly.
- Address Finding 3: explicitly tag the Lane G v3 verdict as
  `[empirical:reports/lane_20_balle_real_archive.json][STATIC_WINS_FALLBACK
  @ Lane G v3 anchor; not contest-CUDA-validated]` AND document that
  Level 3 graduation for Lane 20 awaits a heteroscedastic-anchor lane.
