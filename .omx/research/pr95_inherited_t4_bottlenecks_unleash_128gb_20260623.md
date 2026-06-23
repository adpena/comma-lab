# PR95-inherited T4-VRAM bottlenecks — UNLEASH on the 128GB M5-Max

**Date:** 2026-06-23 · **Subagent:** `t4-bottleneck-audit-20260623` · **READ-ONLY audit**
**Axis:** `[macOS-CPU advisory]` / throughput-only — NON-PROMOTABLE. Pointer UNMOVED (0.19110).
**Scope:** enumerate every memory-bounding choice PR95 made to fit the T4's 16GB, classify whether
unleashing it on the 128GB M5-Max unified memory actually moves s/ep / headroom / quality, while
preserving the training MATH (bit-identical) and EVAL-FAITHFULNESS. RESIDENCY is OWNED by sister
subagent `a816a1b` (GT-pairs-on-CPU + per-batch transfer) — referenced, NOT duplicated here.

> Discipline (from the measured batch finding): NOT every T4 concession binds throughput.
> `batch_size=8→64` gave IDENTICAL s/ep (compute-bound on the frozen-scorer fwd/bwd, not
> step-overhead-bound) — lifting batch did NOTHING for wall-clock. Each candidate below is
> classified by whether unleashing it ACTUALLY helps.

---

## Source map (PR95, cited file:line)

PR95 vendored: `experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src/`
Our inherited path: `src/tac/torch_vehicle/driver.py` + `scorer_context.py` + `curriculum.py`,
launched by `experiments/launch_split_by_head_basin.py`.

---

## The enumerated table

| # | Bottleneck | Source file:line | Why T4-bounded | Classification | EV | Risk |
|---|---|---|---|---|---|---|
| B1 | **Eval streams GT in `batch_pairs=8`** | scorer_context.py:251 (`exact_eval`→`evaluate_decoder`); PR95 score.py:34-99, common.py:237 | The eval holds GT pairs + decoded camera-res frames (1164×874×3) in VRAM; 8 pairs bounds it under 16GB | **UNLEASH** | HIGH — the eval is ~13min (the reason `async_eval` exists); a bigger `batch_pairs` cuts per-eval wall-clock → more eval rows / faster BEST-tracking, and on the **CPU authority device** it is the dominant non-train cost. Frees the async-eval throttle. | LOW — pure batching of an `@inference_mode` stream; d_seg/d_pose are SUMS over per-pair distortions (`score.py:96-98`), order-independent → **bit-identical metric**. Authority math unchanged. |
| B2 | **`cat_entropy_v2(sample_size=2000)`** subsamples each weight tensor for the C1a entropy regularizer | driver.py:2193-2194; PR95 losses.py:79,103-105 | Building the full `(numel × 255)` soft-assignment histogram for every Conv/Linear was a memory/compute cap; 2000 weights/tensor bounds it | **UNLEASH (small)** | LOW-MED — a 229K-param decoder's largest tensor fits the full soft-histogram trivially on 128GB; full population = a less-noisy C1a gradient (no random subsample variance) in stages 5-8. Cheap to lift (raise/remove `sample_size`). | **CHANGES THE MATH** — the entropy term's value/gradient differs (full vs subsampled). NOT bit-identical; it's a *quality* change, needs an A/B that the C1a-driven entropy is no worse. Treat as opt-in, not free. |
| B3 | **`batch_size=8`** per stage (75 serial steps/epoch on 600 pairs) | PR95 common.py:50,168; curriculum.py:168; launcher `--batch-size` already exists | PR95's T4-era default; 8 pairs × 2 frames through the frozen SegNet+PoseNet bwd fit 16GB | **INERT** (measured) | ~0 — **measured**: bs 8→64 identical s/ep (compute-bound on the scorer fwd/bwd, which dominates; the per-step Python/launch overhead is negligible vs the FastViT+UNet passes). Larger batch is a *more exact* gradient (less SGD noise) for single-video memorization, but that is a **gradient-quality** lever, NOT a throughput one. | If lifted: must co-scale LR (`--batch-lr-scale`, already wired) or it under-drives. Already-available; don't expect s/ep wins. |
| B4 | **half-res GT `gt_pairs_half` (192×256), kept on CPU, per-batch transfer** | PR95 data.py:8,97-123 (verbatim comment "kept on CPU to bound memory") | PR95's multi-res-L1 GT was downsampled to 192×256 AND CPU-resident to bound the 600× full-res GT tensor under 16GB | **KEEP / MOOT** | 0 — **our torch_vehicle path NEVER consumes the half-res GT or any multi-res L1 term** (verified: zero refs to `gt_pairs_half`/`multires`/L1 in driver+scorer; our loss is pure score-domain SegNet/PoseNet). The concession is inherited only in the unused PR95 `data.py`; our scorer_context drops `_gt_half` (scorer_context.py:106). | n/a — nothing to unleash; do not "train full-res GT" — that would be a faithfulness trap (the scorer sees the eval-roundtripped 384×512 frame, B5). |
| B5 | **training-target / decoder native size = 384×512** (eval-roundtrip 384→874→384) | driver.py:1979-1986; PR95 common.py:178-185; score.py:24,89-92 | Coupled to the **eval size** — SegNet resizes to (512,384), the scorer never sees more than the roundtripped 384×512 frame | **KEEP (faithfulness-coupled)** | 0 — "train at higher native res" is a **faithfulness trap**: the contest scorer argmax/pose is computed on the downsampled 384×512 (eval-roundtrip bicubic↑→bilinear↓→uint8), so extra native resolution is discarded before scoring AND diverges train from eval. NOT a free headroom win. | HIGH if changed — breaks eval-faithfulness + the eval-roundtrip non-negotiable. Do NOT touch. |
| B6 | **`torch.cuda.empty_cache()` after eval / after precompute** | PR95 common.py:242, data.py:126 | T4 fragmentation hygiene to avoid OOM across the 8-stage run | **ALREADY-UNLEASHED / N/A** | 0 — our driver+scorer have **zero** `cuda.empty_cache()` / `mps.empty_cache()` / `gc.collect()` in the train loop (verified). No per-step cache thrashing exists to remove. On 128GB the hygiene calls are unnecessary and we already omit them. | n/a |
| B7 | **`torch.compile` of the frozen scorers** | driver.py:985 (`compile_scorers` cfg) + 1329-1332; scorer_context.py:155-184 (`maybe_compile_scorers`) | PR95 ran eager scorers (compile memory + warmup cost was a T4-era avoid) | **ALREADY-UNLEASHED (opt-in)** | MED — the lever EXISTS (`compile_scorers=True`) and would cut the FastViT/UNet fwd cost (the s/ep dominator per B3), but is **default-OFF** and **not exposed in the launcher** (no `--compile-scorers` flag). Wiring the flag + running the d_seg/d_pose neutrality check = a real s/ep win on the compute-bound path. | LOW-MED — frozen-fwd compile is score-neutral by construction modulo inductor numerics; needs the paired ≤1e-4 neutrality check the helper docstring calls for before a real run trusts it. NOT bit-identical (inductor), but the metric is re-run eager on the CPU authority so the SCORE is safe. |
| B8 | **GT scorer targets precomputed once + cached resident** | scorer_context.py:101-153 (`_build_capped_targets`, `gt_targets_n<N>.pt`) | PR95 recomputed targets per stage unless `shared_state` carried them (common.py:92-107) | **ALREADY-UNLEASHED** | 0 — we already cache the full-600 targets to disk + hold them resident on `train_device` (scorer_context.py:115-116), skipping the ~2.5h re-precompute. This concession is already removed. | n/a |

---

## Ranked UNLEASH list (real EV, excluding residency which sister `a816a1b` owns)

### #1 (highest EV) — B1: lift `batch_pairs` in the authority eval
- **Action:** thread a `batch_pairs` knob (default keep 8) from the config → `exact_eval` →
  `evaluate_decoder`; set it to the full `n_pairs` (or e.g. 64-128) on the CPU-authority eval.
  Sister edit lives in `scorer_context.py:236-253` + the launcher.
- **EV:** the eval is the single biggest non-train wall-clock (~13min; the entire reason
  `async_eval` exists). Cutting it shortens the BEST-tracking latency and lets the async eval
  cadence keep up instead of self-throttling (skip+log). On a long curriculum this compounds.
- **Risk:** LOW. d_seg/d_pose are per-pair SUMS (`score.py:96-98`) → **bit-identical** regardless
  of `batch_pairs`. `@inference_mode`. Authority math untouched. Only watch peak RAM on the CPU
  authority (full-res camera frames ×600 ≈ a few GB — fine on 128GB).

### #2 — B7: wire `--compile-scorers` into the launcher + run the neutrality check
- **Action:** the cfg field + driver call already exist (driver.py:985,1329); add a
  `--compile-scorers` / `--compile-mode` flag to `launch_split_by_head_basin.py` and run the
  paired d_seg/d_pose ≤1e-4 neutrality smoke once, then enable for the real basin.
- **EV:** MED — the frozen FastViT(pose)+UNet(seg) fwd is the s/ep dominator (B3 proves the loop
  is compute-bound, not step-bound), so compiling the scorer fwd is the lever that *can* move s/ep
  (unlike batch). MPS/CUDA inductor speedup on a frozen fwd is real.
- **Risk:** LOW-MED — score-neutral by construction; inductor numerics need the one-time neutrality
  check; the SCORE is re-run eager on the CPU authority so it can never be corrupted.

### #3 (small, opt-in) — B2: full-population `cat_entropy_v2`
- **Action:** raise/remove `sample_size=2000` (driver.py:2194) so the C1a entropy uses the full
  weight population per tensor.
- **EV:** LOW-MED — a less-noisy C1a gradient in stages 5-8 (the rate-shaping stages) on 128GB at
  trivial cost; *quality*, not throughput.
- **Risk:** **CHANGES THE MATH** (not bit-identical) — opt-in + A/B that the entropy term is no
  worse, never a silent default flip.

---

## KEEP list (do NOT unleash — faithfulness / math reason)

- **B5 — 384×512 training-target resolution.** Coupled to the eval size; the scorer only ever sees
  the eval-roundtripped 384×512 frame. "Train full-res" discards the extra res before scoring AND
  diverges train from eval (eval-roundtrip non-negotiable). Faithfulness trap, not a headroom win.
- **B4 — half-res GT / multi-res L1.** Our path doesn't use it at all; lifting it is moot, and
  "full-res GT L1" would (a) add an unused term and (b) re-introduce B5's faithfulness coupling.

## INERT list (measured / derived not-helpful)

- **B3 — `batch_size=8`.** MEASURED bs 8→64 identical s/ep (compute-bound). Already launcher-exposed.
  A larger batch is a *gradient-exactness* knob (less SGD noise for single-video memorization), not
  a throughput one — do not expect s/ep wins from it.
- **B6 — `cuda.empty_cache()` hygiene.** Our loop has none; nothing to remove.
- **B8 — per-stage target recompute.** Already cached resident; concession already removed.

---

## Headline

The single highest-EV unleash **beyond residency** is **B1: lift the authority eval's
`batch_pairs=8`** — it directly attacks the ~13-min eval that gates BEST-tracking and the async-eval
cadence, is **bit-identical** to the current metric, and is trivially safe on 128GB. Then **B7**
(wire the already-built `--compile-scorers` + neutrality check) is the only lever that can actually
move **train** s/ep (the loop is compute-bound on the frozen scorer fwd, per the measured batch
finding). **B2** (full C1a entropy) is a small opt-in quality lever. Everything else is INERT
(batch, cache hygiene, target recompute) or KEEP (train-res + half-res GT are eval-faithfulness /
unused-term traps). Hand B1 + B7 to the hardening subagent / parent.

— `[macOS-CPU advisory]` NON-PROMOTABLE; $0; pointer-only; no code edited.
