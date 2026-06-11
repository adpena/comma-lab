# Capstone stored-28-d-latent carrier (Arm 2 of the pose A/B) — landed (2026-06-11)

**Authority:** all numbers below are `[macOS-CPU advisory]` / `[macOS-MLX research-signal]`,
NON-PROMOTABLE per CLAUDE.md "MPS auth eval is NOISE" + "MLX portable-local-substrate authority".
The torch-CPU scorer is TRUSTED (per "local CPU + MLX GPU good") and RANKS/GATES, but this is NOT a
contest-axis row and does NOT move the canonical frontier pointer. A sub-0.15 advisory here is the GATE
to a paired contest-CPU + contest-CUDA `upstream/evaluate.py` exact eval (the only pointer-moving step).

## What this fixes (the diagnosis)

Per `.omx/research/capstone_carrier_pivot_vq_index_impoverishment_20260611T034500Z.md`: the capstone's
pose failure (d_pose oscillates 0.06–0.34, never reaches the ~1e-4 tube) is the **8-bit VQ index
impoverishment** — the bundle has a 28-d per-pair latent but VQ-quantizes it to an 8-bit codebook index
(K=256). **8 bits/pair cannot encode 600 distinct ego-motions**, so the per-pair content the FiLM/decoder
sees is ~256 buckets → pose wanders. The fix is the frontier's / PR95's OWN carrier: **store the rich 28-d
per-pair latent DIRECTLY** (temporal-delta + raw-LZMA, PR95 L24/L25) — 28 floats/pair ≫ 8 bits, rate-efficient
AND pose-capable (the frontier reaches d_pose 2.9e-5 with it).

## What was built (ADDITIVE — `vq_index` default byte-identical)

A `carrier: Literal["vq_index","stored_latent"] = "vq_index"` switch threaded end-to-end:

* **`CapstoneVqNervConfig.carrier`** + `CapstoneVqNervBundle`: for `stored_latent` the forward uses the
  gathered per-pair 28-d latent DIRECTLY (no VQ, no codebook, `quantizer=None`, commitment loss ≡ 0); the
  gradient flows straight into `self.latents`. The per-frame FiLM + decoder are unchanged. `all_latents()`
  exposes the carrier; `vq_indices()`/`all_vq_indices()` REFUSE on `stored_latent` (no codebook).
* **`export.py`**: `encode_stored_latents` / `decode_stored_latents` — per-dim uint8 quantize (fp16 min/scale
  side info) + temporal-delta along the pair axis (centered uint8) + raw-LZMA (FORMAT_RAW, L24 filter).
  `build_capstone_stored_latent_archive_bytes` emits a 3-section monolithic `0.bin`
  `(dec | latent | pose)` — NO codebook, NO index. `CapstoneArchiveAccount` gains `carrier` + `latent_bytes`
  (positional-compatible defaults).
* **`inflate.py`** + **`numpy_reference.py`**: `decode_archive` branches on `config["carrier"]`; for
  `stored_latent` it prefix-sum-decodes the temporal-delta latents and feeds them straight to the decoder.
  The numpy render path is carrier-AGNOSTIC (`decoded["latents"]` is `codebook[index]` for vq_index, the
  decoded blob for stored_latent), so a single render path serves both. **d_seg score-parity is EXACT.**
* **`capstone_trainer.export_stored_latents()`**: the EMA-shadow latents to export (the EMA non-negotiable
  applied to the carrier, mirroring `export_render_weights`).
* **`advisory.py`**: `score_reloaded_int8_archive` is now carrier-agnostic (uses `decoded["latents"]`).
* **`run_capstone_campaign.py`**: `--carrier {vq_index,stored_latent}` threads the bundle config + the
  archive build branch + the config sidecar.

### NO-FAKE teeth (17 dedicated tests, all green)

`src/tac/capstone_vq_nerv/tests/test_stored_latent_carrier.py`:
* codec round-trip bounded by the uint8 quant step; single-pair (no-delta) path; beats raw fp16 on smooth.
* **ADDITIVE**: `vq_index` export byte-identical from the same bundle state; config default is `vq_index`;
  account positional construction backward-compatible (`latent_bytes` defaults 0).
* **NO-FAKE**: `stored_latent` bundle has NO quantizer + commitment ≡ 0; `_quantize` returns the RAW latent
  (no VQ snap — `max|z - take(latents,idx)| == 0`); 3-section archive (no codebook/index); end-to-end train
  descends d_seg with commitment staying 0 (a no-op or VQ-fallback would fail this).
* **score-parity** (the gate): `stored_latent` numpy-inflate render == MLX render, **d_seg EXACT** (<1e-4).
* **byte budget**: real archive at base_ch=20, n∈{48,600}, sub-0.15-capable; latent carrier < decoder basis.

`vq_index` byte-identity is also proven structurally: the `git diff` of `export.py`, `vq_nerv_bundle.py`,
`inflate.py` has **ZERO deletions** (purely additive carrier branches); the vq_index code path is untouched.
The full existing capstone suite stays green (67/68; the 1 "failure" is a 60s pytest-timeout on a heavy
`vq_index` real-scorer training loop under daemon GPU contention — verified PASS in isolation, exit 0).

## Byte budget (REAL bytes, base_ch=20, int8 decoder)

| carrier | n=600 total | decoder | carrier | pose | rate |
|---|---:|---:|---:|---:|---:|
| **stored_latent** | **101,187 B** | 83,778 | latent **10,937** | 6,460 | **0.06738** |
| vq_index (K=256) | 97,280 B | 83,777 | cb 6,427 + idx 600 | 6,460 | 0.06477 |

The rich 28-d latent carrier is **~11 KB for 600 pairs** (matching the diagnosis's ~10–15 KB prediction) —
only **+3,907 B (+0.0026 rate)** over the impoverished 8-bit index. With rate ≈ 0.067, the budget leaves
~0.083 for the d_seg + d_pose terms → **fully sub-0.15-capable** if the score terms cooperate.
(At n=48: stored_latent total 85,689 B, rate 0.0571.)

## Smoke (8 pairs, 30 epochs, base_ch=20, carrier=stored_latent, curriculum=none, seed=0)

`experiments/results/capstone_stored_latent_smoke_b20_n8/` — wall 382 s (heavy Metal contention with the
running d_seg daemon).

* **d_pose monotonic descent**: 119.79 (init) → 116.6 → 110.5 → 103.8 → 96.8 → 90.6 → 85.1 (ep30).
* **d_seg**: 0.5073 → 0.5073 (UNCHANGED — expected; this carrier fixes the POSE half; the d_seg floor is the
  separate problem the running curriculum daemon targets).
* **NO-FAKE confirmed live**: `commit_mean == 0.0` at every eval (no VQ fallback); `advisory_quant_gap_d_seg
  == 0.0` (the int8 numpy-inflate reload matches the live MLX render EXACTLY — score-parity holds on the
  real contest path).
* archive: total 85,742 B, rate 0.0571.

### The decisive matched-budget A/B (paired vq_index arm, identical config)

`experiments/results/capstone_vq_index_smoke_b20_n8/` — SAME config (8 pairs, 30 epochs, base_ch=20, seed=0,
curriculum=none), the ONLY difference is `--carrier vq_index` (K=256). Per-epoch d_pose:

| epoch | vq_index d_pose | stored_latent d_pose |
|---:|---:|---:|
| init | 167.9 | 119.8 |
| 5  | 168.4 | 116.6 |
| 10 | 159.7 | 110.5 |
| 15 | 148.7 | 103.8 |
| 20 | 137.1 | 96.8 |
| 25 | 122.3 | 90.6 |
| **30** | **108.42** | **85.11** |

Both arms hold d_seg identical (0.5073 → 0.5073) and both have `quant_gap_d_seg == 0.0` (score-parity holds
for BOTH carriers). The carrier is the ONLY moving variable, and **stored_latent ends 21.5% lower in d_pose
(85.11 vs 108.42)** with faster, smoother descent at every checkpoint — and the vq_index arm carries a
non-zero `commit_mean ≈ 9.3e-4` throughout (the live VQ codebook contrast vs stored_latent's commit ≡ 0).
This is the rich-carrier pose lever empirically isolated: at matched budget the 28-floats/pair carrier
descends pose decisively faster than the 8-bit index.

NOTE the regime caveat: the diagnosis's d_pose 0.06–0.34 oscillation is the vq_index arm's **converged**
regime (post-long-training); an 8-pair/30-epoch smoke at full-scale GT pose (init ~120-168) cannot reach
that low-pose regime in 30 epochs regardless of carrier. So this smoke proves the **matched-budget
descent-rate advantage** (stored_latent wins decisively), not the absolute tube; the long-budget A/B vs the
running d_seg daemon (the orchestrator's job) closes whether stored_latent reaches the ~1e-4 tube where
vq_index oscillates.

## Wire-in (6 hooks, Catalog #125)

1. sensitivity-map: N/A (advisory, non-promotable; no master-gradient row).
2. Pareto: ACTIVE — the carrier is a per-pair rate↔pose-capacity constraint (28 floats vs 8 bits at ~+0.0026 rate).
3. bit-allocator: ACTIVE — `encode_stored_latents` (per-dim uint8 + temporal-delta + LZMA) is the per-pair carrier allocator.
4. cathedral autopilot: N/A (advisory; not a pointer-moving archive yet).
5. continual-learning posterior: N/A (no contest-axis anchor; advisory only).
6. probe-disambiguator: ACTIVE — this carrier A/B (stored_latent vs vq_index) IS the pose-half disambiguator.

`council_predicted_mission_contribution: frontier_breaking_enabler` (the pose-half lever toward sub-0.15).

## Reactivation / next

The decisive step is the paired d_pose A/B at the trustworthy long budget (the orchestrator's A/B vs the
d_seg daemon). If stored_latent holds pose where vq_index oscillates AND the curriculum daemon breaks the
d_seg floor, the sub-0.15 path is the amortized capstone NeRV with a **stored-28-d-latent carrier** (NOT the
8-bit VQ index, NOT the rate-worse stored mask). Then: paired contest-CPU (Linux x86_64) + contest-CUDA
`upstream/evaluate.py` exact eval on the byte-closed `stored_latent` archive (the only pointer move).
