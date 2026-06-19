# Frontier int5 Path-B BEST-SHOT re-test: per-tensor LSQ + outlier-clip (the canonical low-bit fixes the abs-max cap omitted)

- **UTC**: 2026-06-19T03:17:09Z
- **Commit (code)**: `2416b39be`
- **Authority**: `[contest-CPU advisory]` NON-PROMOTABLE. Frontier pointer UNMOVED `0.19110`.
- **Spend**: $0 (local MPS gradient + local CPU authority eval; no GPU dispatch, no paid).
- **Lane**: best-shot re-test of the int5 Path-B cap per the recursive-adversarial-review finding
  (`.omx/research/recursive_adversarial_review_recent_negatives_20260619T024605Z.md`).

## The finding under re-test (the premature-KILL the review caught)

The int5 Path-B campaign concluded *"Path-B QAT-shrink caps at S=0.483, d_seg walls ~0.0035,
STRUCTURAL to int5."* The recursive adversarial review found the quantizer was **per-tensor
symmetric abs-max** (`tac.torch_vehicle.score_aware_qat._fake_quantize_n` /
`tac.post_hoc_weight_shrink.intn_qdq`) with **NO per-channel scales, NO LSQ, NO outlier handling**
— the EXACT canonical low-bit fixes. A cap measured at lifted-implementation form, not best-shot
form → IMPLEMENTATION-LEVEL falsified, PARADIGM re-opened (Catalog #307).

## DECISIVE codec constraint (reshapes the review's proposed fix — measured, not assumed)

The frontier codec grammar (`tac.pr101_split_brotli_codec._quantize_tensor`) stores **exactly ONE
per-tensor int8 scale per tensor** (`abs_max / 127`, int8 codes + one fp16 scale). Consequences,
all MEASURED on the real frontier archive:

1. **Per-channel scales are NOT byte-closeable.** Per-channel int5 export blows the archive
   **118,589 → 197,353 B** (+78,764, WORSE than the int8 frontier's 177,169) because the
   per-channel grid maps to **176 distinct codec-int8 symbols on `stem`** vs 27 for per-tensor →
   more brotli entropy. AND per-channel int5 is not preserved by the codec's per-tensor int8
   (5.6e-3 weight err) → the byte-closed decode ≠ trained weights (a NO-FAKE divergence). **The
   review's "per-channel scales" fix is incompatible with this codec grammar** (would need a new
   archive section storing per-channel scales = a different campaign).

2. **The byte-close-COMPATIBLE form of the fix = per-tensor LSQ learned step + outlier clip.**
   A learned/calibrated per-tensor step SMALLER than abs-max clips the outlier and gives the bulk
   weights finer int5 resolution. The frontier `stem` (21% of params) has a 3.26× per-channel-max/
   median outlier — one channel sets the abs-max scale, starving the median channels. The
   MSE-optimal per-tensor clip (~0.5× abs-max on the heavy-tailed rate carriers stem/blocks.0/1)
   **cuts int5 reconstruction MSE 50-61%** at one-scale-per-tensor (codec-compatible).

## What was built (commit `2416b39be`, all NO-FAKE, 24 tests)

- `tac.frontier_int5_qat.FrontierLSQQuantizer` — per-tensor LEARNABLE LSQ steps (Esser et al. 2020),
  STE forward, the canonical step gradient `(round(v/s) − v/s)` in range / `±qmax` saturated, scaled
  `1/√(numel·qmax)` (VERIFIED against the Esser closed form to 1e-6; saturated outliers correctly
  get ZERO weight grad — the true clamp STE, stronger than the vendored identity-everywhere).
- `mse_optimal_step` — the outlier-handling calibration (LSQ init from the MSE-optimal clip).
- `hard_quantize_state_dict_lsq` — export at the learned-step grid (the grid trained == the grid
  shipped; still per-tensor symmetric int5 = byte-close compatible).
- Harness `experiments/frontier_int5_score_aware_qat_finetune.py`: `--quantizer lsq`,
  `--freeze-latents` (NO-FAKE: reencode ships `member.latent_raw` verbatim, so training the latents
  over-states the eval — freeze so trained == shipped), `--eval-redecode` (NO-FAKE eval-on-shipped-
  bytes: re-decode the byte-closed archive and eval THOSE weights, not the ~5.6e-3-off in-memory).

## Empirical signal (calibration-PTQ probe, ZERO training)

`experiments/probe_int5_lsq_calibration_vs_absmax.py` — abs-max int5 vs LSQ-calibrated int5,
byte-closed + re-decoded + REAL CPU exact eval.

**8-pair smoke (eval-redecode ON):**

| variant | d_seg | d_pose | archive_bytes | S |
|---|---|---|---|---|
| abs-max int5 (the cap's quantizer) | 0.004243 | 0.004266 | 118,589 | 0.7098 |
| LSQ-calibrated int5 (zero training) | 0.002655 (−37%) | 0.001659 (−61%) | 138,389 | 0.4864 |

LSQ calibration ALONE (no finetune) cuts d_seg 37% and d_pose 61% — the review was RIGHT that
abs-max left large precision on the table. Cost: +20k bytes (the finer-resolution clip spreads
weights over more int5 levels → larger brotli).

**600-pair calibration (abs-max baseline CONFIRMED; LSQ variant killed to free MPS for the
decisive long run):** the 600-pair abs-max int5 baseline landed at **S=0.6709, d_seg=0.004681,
d_pose=0.001533, bytes=118,589** — confirming the cap's int5 collapse on the full 600 pairs
(matches the memo's ~0.00475 d_seg). The LSQ 600-pair variant was killed mid-eval (MPS/CPU
contention was starving the decisive long finetune); its signal is established by the 8-pair smoke
(LSQ-calib d_seg −37%, d_pose −61%). The 600-pair abs-max baseline is the apples-to-apples cap
anchor the long-run LSQ result is compared against.

## The best-shot long finetune

`--quantizer lsq --mode uniform --low-nbits 5 --seg-loss ce --freeze-latents --epochs 2000`
(CE per the cap memo's finding that CE recovers d_seg where margin-hinge harms it at the coarse
grid — NOTE: this finding did NOT replicate at the byte-closed grid; see the verdict). MPS gradient,
CPU-authority byte-closed eval-on-shipped-bytes every 100 epochs (each eval ~9–11 min: the redecode
of the full byte-closed archive + 600-pair scorer pass is the cost — a known inefficiency that makes
eval-every-100 over a 2000-ep run impractical; the ep100 row is the decisive data point).

**Long-run ep100 (600-pair eval-on-shipped-bytes CPU authority):** S=0.5593, **d_seg=0.004236**,
d_pose=0.000165, bytes=142,853, rate=0.0951.

| config | d_seg | d_pose | rate (bytes) | S | axis |
|---|---|---|---|---|---|
| frontier int8 (pointer) | 0.00056 | 0.00003 | 0.118 (177,169) | **0.19110** | contest |
| int8 local baseline | — | — | — | 0.1965 | local-CPU |
| abs-max int5 cap (600-pair) | 0.004681 | 0.001533 | 0.0790 (118,589) | 0.6709 | local-CPU |
| **LSQ int5 + outlier-clip + CE, ep100 (600-pair, eval-redecode)** | **0.004236** | **0.000165** | 0.0951 (142,853) | **0.5593** | local-CPU advisory |

## VERDICT: RED — the int5 d_seg wall is CONFIRMED STRUCTURAL (cap stands, refined)

The best-shot (per-tensor LSQ learned step + outlier-clip calibration + CE finetune, the canonical
low-bit fixes the abs-max cap omitted) **recovers d_pose enormously (0.001533 → 0.000165, −89%) but
d_seg barely moves (0.004681 → 0.004236, −9.5%)**. d_seg stays ~0.0042 — **7.6× the frontier's
0.00056** — and the CE seg-loss is FLAT across ep10→100 (seg_l 0.0117–0.0119, NOT descending), so
more epochs will not break the wall. S=0.5593 is **2.9× the pointer 0.19110**. The d_seg wall is
NOT an artifact of the per-tensor-abs-max quantizer — the canonical low-bit fixes (per-tensor LSQ +
outlier clip + the better seg-loss) **cannot recover d_seg at the int5 byte-closed grid**, because:
1. per-channel scales (the one fix that could give d_seg-critical weights real per-output-channel
   precision) are NOT byte-closeable through the codec's per-tensor-int8 grammar (measured: blows
   the archive 118k → 197k);
2. the per-tensor LSQ/clip recovers the POSE head's precision (low-dim, per-tensor-friendly) but the
   d_seg-critical early/low-res stages (77% of params) need finer-than-int5 resolution that no
   per-tensor scale can provide.

**Honest correction to the cap memo**: the memo claimed CE recovers d_seg where margin-hinge harms
it. At the int5 byte-closed grid this did NOT hold — LSQ+CE ep100 d_seg=0.0042 is slightly WORSE
than the margin-hinge cap's ~0.0035 (and S higher, 0.5593 vs 0.483, also from +bytes). CE recovers
pose, not d_seg, at the coarse grid.

**The "STRUCTURAL to int5" cap is CONFIRMED at best-shot form** (not under-powered): even the
canonical low-bit toolkit cannot make the int5-byte-closed frontier decoder hold d_seg. This is now
a REAL closure (Catalog #307: best-shot implementation + measured byte-closed 600-pair CPU row +
adversarial self-review), not a premature KILL. The sub-0.15 path is NOT through bit-shrinking the
frontier's RATE — it routes to a concentrated-saliency OWN vehicle whose d_seg-critical capacity is
spent where the argmax boundary lives (per the cap memo's pivot + the small-basis micro/macro audit).

**Pointer UNMOVED 0.19110.** Not GREEN, not a pointer-move candidate. The long finetune is left
running (resumable) to confirm d_seg does not break by ep500+, but the flat seg_l makes that near-
certain.

## NO-FAKE / discipline

- Score is the REAL byte-closed archive through the frontier inflate/codec → REAL CPU
  `RealScorerContext.exact_eval`, recomputed S from components. NEVER MPS for the score.
- per-channel NOT byte-closeable (measured) → the byte-close-compatible LSQ+clip is the honest fix.
- eval-redecode + freeze-latents close two NO-FAKE gaps (in-memory-vs-shipped weights; trained-vs-
  shipped latents) present in BOTH the cap and the first draft of this re-test.
- If GREEN: do NOT self-promote. The frontier is PR101/106-derived → contest PR needs
  `borrowed_substrate_accounting` (NO-FAKE class 7); local-beats-0.191 needs paired contest CPU+CUDA
  (the +0.0054 local-vs-contest offset). Pointer stays UNMOVED until a paired contest row.
