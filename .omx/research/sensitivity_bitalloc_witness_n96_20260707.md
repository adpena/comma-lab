# #336 Sensitivity bit-alloc APPLY on witness weights — n96 MEASURED row (FEED-07b row 5)

**Status: COMPLETE (2026-07-07).** The #157 reverse-water-fill (`tac.frontier_exact_bitalloc.
waterfill_bit_allocation` + `lam_for_target_mean_bits`, imported UNCHANGED) pointed at OUR witness
checkpoint for the first time — the compress-half of train-big-compress-small.

Run: `tools/apply_sensitivity_bitalloc_witness.py` on the mod32cap live-run snapshot
(`snapshot_ema_BEST.npz`, ep425 best d_seg 0.0036364, sha256
`9f123bac950af1ec8eecc938c042f6d0e0662d0a427a16d5ba147e3f1fc62d93`; run
`levelset_n600_witness_mod32cap_20260706T115554Z`), probe n16 / eval n96 evenly-spaced pairs,
`--mean-bits 6 5`. Authority: **[macOS-CPU advisory] NON-PROMOTABLE** — bounded-subset
measurement, never a score. Blob-grammar parity vs the canonical `build_levelset_blob` (#202) was
MEASURED byte-identical 2026-07-07 (base int8 72,695 B sha-match; code int8 38,400 B sha-match;
brotli-11 streams byte-equal), pinned by `src/tac/tests/test_bitalloc_witness_blob_parity.py`.
Pointer 0.19110 UNMOVED (MEANS). Artifacts:
`experiments/results/sensitivity_bitalloc_witness_20260707/` (result JSON + per-unit resumable
state + chunk logs + ckpt provenance/sha).

## Measured sensitivity (per-tensor int8→int5 d_seg response, n16 probe; NOT a gradient proxy)

Baseline int8: weights_total = **83,098 B** (base brotli 61,953 + code brotli 21,145); d_seg
n16 probe 0.003620 / n96 eval **0.003699** (consistent with the live n600 verdict 0.003636).

| tensor | Δd_seg (int5, others int8) | c_t = Δd/(2⁻⁵−2⁻⁸) |
|---|---|---|
| in_proj.weight | **+0.003585** | 0.13109 |
| code | +0.001689 | 0.06177 |
| hidden.0.weight | +0.001336 | 0.04886 |
| hidden.1.weight | +0.000813 | 0.02974 |
| film.weight | +0.000504 | 0.01844 |
| hidden.2.weight | +0.000481 | 0.01758 |
| hidden.3.weight | +0.000274 | 0.01003 |
| out_tex.weight | +0.000147 | 0.00537 |
| palette | +0.000112 | 0.00409 |
| out_sdf.weight | +0.000042 | 0.00152 |

Depth-ordered sensitivity (in_proj ≫ … ≫ out heads) + the per-pair `code` — EVERY tensor's probe
response is positive (no free sub-int8 slack anywhere; out_sdf ≈ n16 noise floor).

## Waterfill vs uniform at matched budget (n96 eval, MEASURED through R + frozen CPU SegNet)

| operating point | bytes (real brotli) | d_seg | Δd_seg vs int8 | Δrate_S | Δseg_S | net ΔS (advisory) |
|---|---|---|---|---|---|---|
| int8 baseline | 83,098 | 0.003699 | — | — | — | — |
| **WF mean-6** (nbits: in_proj 8 / code 6 / h0 7 / h1 7 / h2 6 / h3 5 / film 5 / heads 7-8) | 55,528 | 0.005017 | +0.001319 | **−0.01836** | **+0.13189** | **+0.11353 (LOSES)** |
| uniform int6 control | 54,862 | 0.005917 | +0.002218 | | | (worse than WF) |
| WF mean-5 | 41,466 | 0.014907 | +0.011209 | −0.02772 | +1.12087 | +1.09315 (LOSES) |
| uniform int5 control | 41,872 | 0.011296 | +0.007597 | | | (BEATS WF at mb5) |

## Verdict (both branches signal)

1. **The apparatus TRANSFERS:** at mean-6 / matched bytes, the measured-sensitivity waterfill
   beats the uniform control (d_seg 0.005017 vs 0.005917 at ~equal bytes, −15% d_seg at +1.2%
   bytes) — the #157 KKT allocator + measured c_t works on the witness.
2. **The operating point does NOT pay:** at this witness the seg term (×100) dwarfs the weights
   rate term (83 KB → rate_S 0.055 total). Even the FULL weights section deleted buys only
   −0.055 S, while the cheapest measured sub-int8 step costs +0.132 S of d_seg. **int8 is already
   past the RD knee for this checkpoint — sub-int8 bit-alloc on the witness weights is a measured
   NO at the current d_seg sensitivity level.** Reactivation criterion: revisit iff the witness's
   quantization response drops ~50× (e.g. flat-minima #242 / larger-capacity witness whose weights
   dominate rate), or for a rate-starved composition where every KB is marginal.
3. **Model-validity caveat:** at mean-5 the first-order `D = Σ c_t·2^{−b_t}` model breaks
   (waterfill WORSE than uniform — film.weight at int3 is outside the linear-response regime the
   n16 int5 probe calibrated). The mb6 row (all tensors ≥ int5) is inside the calibrated regime.

## What n600/promotable needs

A full #202 byte-close of a chosen allocation (grammar unchanged, reader unchanged) + n600
verdict through `tools/levelset_byte_close_and_eval.py`, then contest-CPU/CUDA exact eval — but
per verdict (2), do not spend that until the reactivation criterion is met.

## Process note (the daemon-kill incident, measured)

The run was driven in bounded FOREGROUND chunks (`--chunk-seconds` + per-unit/per-8-render
resumable state) after detached-daemon launches were silently SIGKILLed ~5–6 min in across FOUR
generations (nohup/disown, sandboxed spawn_durable_daemon ×2, unsandboxed spawn_durable_daemon)
regardless of rss caps 10/12/16/20 GiB. Instrumented RSS across ~2,000 renders: **flat 2.4–2.7 GiB
(peak 3.9 GiB during the lane-band fit)** — the tools have NO memory spike; the kills correlate
with the agent-harness long-call sweep, not with tool memory. Chunked-foreground + per-unit state
is the robust drive mode; the durable note lives in `tools/spawn_durable_daemon.py`'s docstring.

## Checkpoint custody addendum (2026-07-07, clean-pass-#2 F1 remediation — APPEND-ONLY)

The evidence JSON's `ckpt.dir` cites a session-scratchpad path (transient). The exact ep425
EMA-BEST snapshot the measurement ran against (sha256
`9f123bac950af1ec8eecc938c042f6d0e0662d0a427a16d5ba147e3f1fc62d93`) has been copied into this
memo's durable results directory as `snapshot_ema_BEST.npz` alongside a `CHECKPOINT_CUSTODY.md`
note (sha verified equal post-copy). The sha-pinned evidence JSON is untouched; bit-exact
re-runnability is restored independent of scratchpad GC. Source run:
levelset_n600_witness_mod32cap_20260706T115554Z (snapshot 2026-07-07T02:37Z).
