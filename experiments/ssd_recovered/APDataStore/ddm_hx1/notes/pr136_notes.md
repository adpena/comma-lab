# PR136 `hnerv_rc` (JPL11 / Jacky Li) — intake notes, ddm_hx1

Fetched 2026-08-17. Detached custody: `/Volumes/APDataStore/pact/ddm_hx1/intake/`.
Read + static analysis only. No launches, no scorer runs, no dispatch.

## Status: CLOSED by maintainer

`state = CLOSED`, created 2026-08-07T21:37:17Z, head `95d1b49b21c4d0a596bcd47c6ca2edd8c15b5b48`.
Two comments only: the bot ack, then YassineYousfi 2026-08-07T22:27:31Z pointing at
the repo's *coding-agents-and-LLMs policy*. **No eval-bot score comment exists.**
The 0.19258 is the author's own pasted `report.txt`, never maintainer-verified.
Treat every number below as SELF-REPORTED.

## Custody

| file | bytes | sha256[:12] |
|---|---:|---|
| pr136_meta.json | 6154 | 34844c48545e |
| pr136_full.diff | 77329 | d4c8db96b0fe |
| README.md | 1484 | aa156f00374f |
| compress.sh | 991 | c4c94c43d368 |
| inflate.py | 2158 | fdc1e9af70a8 |
| inflate.sh | 811 | 0cb9c3df78ad |
| src/codec_rc.py | 4881 | 06dd2ffba5a3 |
| src/codec.py | 8223 | 981e07291313 |
| src/data.py | 6721 | db0016ba8e85 |
| src/losses.py | 6892 | 757725dc6468 |
| src/model.py | 2197 | e63b04ad3df4 |
| src/optim.py | 3785 | 6c79f7559e4f |
| src/score.py | 4661 | c8c4421fd26f |
| src/stages/codec_stage.py | 2870 | 1ddd7ccac942 |
| src/stages/common.py | 11658 | ad766d063926 |
| src/stages/stage1_v328_ce.py | 1456 | a43a44fa4cdc |
| src/stages/stage2_v331_softplus.py | 1216 | 1203339dc531 |
| src/stages/stage3_v332_smooth.py | 1206 | 119fe4e17185 |
| src/stages/stage4_v332_qat.py | 1233 | aaf7f704f0fc |
| src/stages/stage5_c1a_l7.py | 1329 | 6e1c4f8a1abd |
| src/stages/stage6_lambda_sweep.py | 1208 | fe3893d388d7 |
| src/stages/stage7_sigma_sweep.py | 1255 | 8b8b812bb4bc |
| src/stages/stage8_muon_finetune.py | 1866 | 9e036c99b7fc |
| src/train.py | 2488 | d88610929db7 |

(`._*` AppleDouble files are ExFAT metadata from the write, not payload.)

Archive.zip itself is a release asset on the author's fork; NOT fetched
(would be 178 KB; not needed for source analysis, and the PR is closed).

## Context model — order-0, reset per tensor. That is the whole model.

`codec_rc.py:28-36` `_encode_stream` allocates a fresh 256-slot count table per
call. `encode_decoder_rc` calls it once per tensor (`:68-69`, comment literally
"one stream per tensor, reset counts"). So the only conditioning is **tensor
identity**. Within a tensor: nothing. No previous-symbol context, no bit-plane,
no spatial neighbour, no layer-type class, no sign/magnitude split. Symbols are
the zigzagged INT8 weights (`codec.py:42-44`), i.e. a single flat uint8 stream
per tensor. Latents get exactly two streams: the lo byte-plane and the hi
byte-plane, each with its own fresh table (`codec_rc.py:108-109`).

## Adaptation law — literal

```python
INC = 8.0; PRIOR = 1.0; ALPHABET = 256          # codec_rc.py:23-25
counts = np.full(256, 1.0, dtype=np.float64)    # :30 / :40
p = counts / counts.sum()                       # :32 / :43
enc.encode([s], constriction.stream.model.Categorical(p, perfect=False))  # :33-34
counts[s] += 8.0                                # :35 / :46
```

Plain cumulative-count Laplace/Dirichlet estimator. Prior α = PRIOR/INC = 0.125
pseudo-observations per symbol (KT would be 0.5); total prior mass 256, growing
to 256 + 8n. **No decay, no windowing, no halving/rescale on overflow, no
shift-based update.** It is *not* CABAC-style `p += (target-p)>>rate` — that
would be exponential-forgetting and would track non-stationarity; this one's
adaptation rate falls as 1/n and converges to the stream's global empirical
distribution. For a weight tensor (roughly stationary) that is the right call;
it would be the wrong call for a non-stationary stream.

## State size and sync

Adaptive state = 256 × float64 = **2048 B per stream, transient only** — never
serialized, never transmitted. Number of contexts = number of tensors (~28 for
the 229 K-param HNeRV) + 2 latent planes; only one is live at a time.
Transmitted side-information is names/shapes/scales/sizes, brotli'd
(`codec_rc.py:71`) — no frequency table. That is the genuine win of going
adaptive.

Sync: decoder rebuilds the identical table from the identical constants and the
symbols it has already decoded (`:38-47` mirrors `:28-36` exactly). Both sides
consume streams in the same order driven by the brotli'd header sizes.

**Float-in-probability-path — examined, and it is safe here by accident, not by
design.** The docstring (`:6-8`) claims "identical **integer** count tables";
the code uses float64. It still round-trips bit-exactly because every count is
exactly `1.0 + 8k`, an exact integer well under 2^53, so `counts.sum()` is exact
under any summation order (numpy pairwise, SIMD, FMA — all identical), and the
subsequent division is IEEE-754 correctly-rounded and therefore deterministic
cross-platform. The *pattern* is fragile, not the instance: any non-integral
PRIOR/INC, any multiplicative decay, or counts past 2^53 makes `sum()`
order-dependent and silently desyncs the decoder. Anyone reimplementing should
use integer counts and an integer-domain CDF, which is what the docstring
already promises.

## Claimed delta vs static coding — and an arithmetic wrinkle

Author's numbers (`codec_rc.py:10-11`, README:16-19): decoder tensors brotli
**163,237 B** → rc **161,736 B** = **−1,501 B**; per-tensor order-0 entropy
floor quoted at **160,387 B** (so they land 1,349 B / +0.84 % above the bound —
the adaptive-learning cost, plausible for α=0.125 over ~28 streams).
Total archive 177,998 B. Prior PR-family precedent is cited: `codec.py:4-8`
records that an *earlier* hybrid per-tensor categorical AC was only ~217 B
better than brotli and was removed for simplicity — the gain jumped to 1.5 KB
only after the better-converged retrain, i.e. the coder win is a function of the
weight distribution, not a constant.

**Wrinkle: the latent stream's delta is never reported, and the arithmetic
suggests it is negative.** Decoder alone: 25 × 1501 / 37,545,489 = **0.00100**
of score. Author claims total "~1.1 KB" ≈ **0.00073**. If both figures are
taken at face value the latents got ~400 B *worse* under the range coder — which
is exactly what theory predicts, because `encode_latents` deliberately makes the
hi byte-plane "mostly zero" (`codec.py:12,114`) and an order-0 model cannot
exploit runs while brotli's LZ77 can. v2 applies rc to both streams
(`codec.py:166-169`) with no per-stream min(brotli, rc) choice. This is an
INFERENCE from their two stated numbers, not a measurement; "~1.1 KB" may
simply be a loose restatement of 1.5 KB. Either way the per-stream delta is
unreported, and a min() selector is free.

## CPU-axis engineering — nothing there

Zero mentions of wall-clock, timing, the 30-min budget, C extensions, or
vectorization anywhere in the submission (grepped). No `requirements.txt`;
`inflate.sh` installs nothing and assumes `torch`, `brotli` and `constriction`
are already importable in the eval environment. `codec.py:56` explicitly
celebrates having *removed* the constriction dependency, and `codec_rc.py`
puts it straight back — a self-documented runtime-closure regression, the same
class as the PR106 missing-brotli replay failure.

`inflate.py:32` is `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
— the auto-fallback pattern. The author's own body reports ~0.23 on the GPU/DALI
axis vs 0.19258 on CPU, and attributes it to "the known cross-decoder gap".

Decode cost, ESTIMATED (not measured — no code was run): ~229 K weight symbols
+ 33.6 K latent symbols ≈ **263 K decode iterations**, each rebuilding a fresh
256-entry `Categorical` through the FFI plus a numpy divide+sum. At a plausible
10–40 µs/symbol that is **~3–11 s**, i.e. irrelevant against 30 min even though
it is ~100× more work than an incremental-CDF implementation would need.
Error bars are wide; treat as order-of-magnitude only.

## Is it off-the-shelf?

Substantially yes, and it says so. The arithmetic engine is stock
`constriction.stream.queue.RangeEncoder/RangeDecoder` with stock
`stream.model.Categorical` — a published Rust/Python library, not a hand-rolled
coder. The novel surface is ~40 lines: the adaptive count loop, the per-tensor
reset, and the v2 archive framing. The model itself (uniform prior + fixed
increment, order-0) is the textbook adaptive arithmetic coder from
Witten-Neal-Cleary 1987. The honest framing is "correct, minimal, well-scoped
application of a standard coder", not a new coder.

## Also noted (outside the coder brief)

- `losses.py:1-14`: the 8-stage loss ladder, with `smooth_disagreement_seg_loss`
  (`:37-45`) — sigmoid of negative margin, gradient peaking exactly at margin=0
  — and `l7_softplus_seg_loss` (`:47-60`) boosting pixels with margin < 1.0 by
  4×. Aggregation `100*seg + 1*pose + λ*c1a_entropy`, EMA 0.999. This is
  hnerv_muon's ladder, not new here.
- `optim.py:1-10`: Muon + explicit decoupled weight decay, citing
  arXiv:2506.15054 for why WD must be active for Muon's spectral-norm KKT story.
- `compress.sh:10` runs `submissions.hnerv_muon.src.train` and `inflate.py:9`
  documents `submissions.hnerv_muon.inflate` — both point at the parent
  submission, not `hnerv_rc`. Copy-paste leftovers; the reproduce path as
  written does not run this PR's own code.
- Author reports a NEGATIVE in the body: an asymmetric-f0 pose-carrier variant
  costs ~8 KB on a rate-dominated score and lands ~0.196, "does not transfer to
  the HNeRV lineage — the shared trunk already makes f0 nearly free."
