# #205 store-nothing-pose — axis-9 measured-runnability + measured-scored-quantity SMOKE RESULT — 2026-07-03

**Task:** run the axis-9 pre-burn GATE that gates the #205 launch (memo
`.omx/research/witness_205_store_nothing_pose_config_and_review_20260703.md` §5). BOUNDED 2-epoch
measurement at the REAL n600 config — NOT the multi-hour burn (operator-GO-gated). Pointer **0.19110
UNMOVED**; everything here is `[macOS-MLX advisory]` for the render/train + `[contest-CPU authority]`
for the byte-close verdict — NEVER MPS.

## VERDICT: **GO-READY** (memory + measurability gates PASS) — with ONE flagged rate-budget discrepancy

Both axis-9 gates pass: (a) measured peak RSS **54.78 GiB < 90 GiB cap**; (b) **all three scored
quantities (d_seg, d_pose, rate) reproduce through the REAL byte-closed decode + contest inflate**.
The store-nothing pose section byte-closes + inflates + yields a real, MEASURED d_pose (the exact
thing a prior byte-close "could not even reproduce"). **One honest finding flagged below:** the
store-nothing pose section measured **52135 B (~52 KB)**, NOT the memo's asserted ~1049 B / ~0.0007
rate — the store-nothing *property* holds (0 stored keyframe) but the per-pair xi/H section costs
~half the 110 KB archive. This does not block the memory/runnability gate but materially contradicts
the "~0 pose rate" design claim and must be resolved before the rate-budget half of the sub-0.15 story
is trusted.

## Provenance
- git_sha `9a10cf062fbf6f33523e8ed7721b862740b4a4db` (dirty), seed **0**, deterministic.
- upstream_snapshot_sha256 `d46d89155dbf0848e357858c8f62e12ef450a2914ef65814a4359ef6768d2d41`.
- `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` active (`custom_grouped_backward active=true`, the ~17× fast path).
- machine free (single job), 128 GiB RAM, governor ADMIT (projected system-used 89.8 ≤ ceiling 117.8).
- run dir `experiments/results/sn205_axis9_smoke`; probe log `reports/sn205_axis9_memprobe.log`;
  byte-close log `reports/sn205_axis9_byteclose.log`; byte-close JSON `reports/sn205_axis9_byteclose.json`.

## (a) Peak-RSS measurement — PASS (54.78 GiB < 90 GiB)
`TAC_MEM_PROBE=1 TAC_MEM_PROBE_EPOCHS=2` on the REAL n600 store_nothing_205 config (all self-orient /
basis / pose-carrier / chroma / lane-band flags identical to the emitted `launch.sh`; only the bounded-
smoke deltas applied, see NOTE-1). Both OOM drivers hit at startup: resident `cf_mx_cache` (mlx_active
41.91 GiB) + the v0 verdict spike.

| phase | rss_gib | note |
|---|---|---|
| after_cf_mx_cache_build | **55.69** | resident cache built (mlx_active 41.91) |
| before_v0_verdict | **55.69** | verdict_batch=32 |
| **after_v0_verdict** | **54.78** | **GATE: < 90 ✓** — d_seg 0.745683 d_pose 1.886958 (untrained baseline, advisory) |
| mid-verdict transient (ps) | **~65.2** | high-water during the chunked verdict; still < 90 |
| training peak (mlx_peak_gib) | **48.27** | epochs 1-2, rock-stable rss 54.55 |

The `--verdict-batch 32` chunk fix held memory FLAT through the entire v0 verdict — **no +66 GiB
un-chunked spike** (the exact failure that killed the last n600 launch). Projected 67.61 GiB was
conservative; measured after_v0_verdict is 54.78 GiB with a ~65 GiB transient. **PASS.**

## (b) Scored quantities through the REAL byte-closed decode — PASS (all three reproduce)
Tool: `tools/levelset_byte_close_and_eval.py` (the #202 levelset tool — see NOTE-2, the memo cited the
wrong tool). Decode tier `decode_cpu_16gb` (contest=True, bit_exact=True, eval_device=cpu, 1-thread
BLAS) = **[contest-CPU authority]**, on the epoch-2 EMA checkpoint `levelset_witness_ema_mlx.npz`
(n_pairs=600, params=117527, self_orient=True), `--pose-carrier --pose-carrier-mode store_nothing
--pc-s-t 0.044 --pc-s-r 0.0 --pc-pitch 0.0` (matching the trainer's self-fit `s_t_fit → 0.044`).

- **inflate**: `(1200, 874, 1164, 3) uint8 [f0,f1 per pair]  full_output_ok=True  raw_bytes=3662409600`
  — the store-nothing archive byte-closes AND the REAL contest `inflate.py` decodes it. RUNNABLE.
- **rate** = **0.002936**  (archive.zip = **110248 B**, 0.bin = 130952 B, rate_term = 0.0734).
- **d_seg** = **0.081390**  (realized on inflated frames, 600 pairs).
- **d_pose** = **1.095279**  (realized on inflated frames) — the store-nothing pose decode REPRODUCES
  and d_pose is MEASURED. v0 (untrained) d_pose 1.886958 → ep2 1.095279 = the store-nothing pose
  carrier IS training (descending). S_advisory = 11.5219.
- store-nothing pose section = **52135 B, keyframe_bytes=0** (frame0 = warp of the witness's OWN
  render by per-pair xi/H; ZERO stored keyframe image → the store-nothing property holds).

Values are UNCONVERGED at 2 epochs — **EXPECTED**; the axis-9 (b) gate is runnability + all-three-
scored-quantities-MEASURABLE through the real decode, NOT the score level. **PASS.**

## Flagged findings (honest; do not fake a pass)
- **FINDING-1 (rate-budget, material): store-nothing pose section = 52135 B (~52 KB), NOT the memo's
  ~1049 B.** The store-nothing *property* is real (keyframe_bytes=0, no stored image), but the per-pair
  xi/H section is ~52 KB — ~47% of the 110248 B archive, contributing ~25·52135/37.5M ≈ **0.0347 to the
  rate term**, ~50× the memo §2(A)/§5 claim of "~0.0007 (~0 pose rate)". This does NOT block the
  memory/runnability GO-READY gate, but the "~0 pose rate" design claim is **not reproduced** and must
  be re-derived before the rate half of the sub-0.15 story is trusted (is the ~1049 B a global-xi vs
  per-pair-xi encoding, a different n, or an over-optimistic memo number?).
- **NOTE-1 (smoke-plan refinement, not a config defect):** the memo §5(a) "swap `--epochs 1000→2`"
  collides with two epoch-keyed fail-closed guards — the curriculum guard (`l7_start_epoch ≤ epochs`,
  trainer :4106) and the muon guard (`1 ≤ muon_start_epoch ≤ epochs`, :4165). The faithful bounded-
  smoke scales the three curriculum stage-starts to fit (`--tau-softplus-start-epoch 1 --l7-start-epoch
  2 --muon-start-epoch 2`); stage TIMING is memory-irrelevant (peak = cf_mx_cache + v0-verdict at
  startup), so every memory-relevant flag stayed identical → the peak path is the REAL one. The LAUNCH
  config (epochs 1000) is unaffected.
- **NOTE-2 (memo spec bug): §5(b) cited `tools/witness_byte_close_and_eval.py` — WRONG tool.** That
  tool is for a simpler isotropic witness and explicitly REFUSES non-`isotropic` basis (and looks for
  `witness_ema_mlx.npz`, not `levelset_witness_ema_mlx.npz`). The correct byte-close for the self-orient
  + pose-carrier + store-nothing levelset config is **`tools/levelset_byte_close_and_eval.py`** (#202,
  MEMORY.md). The memo §5(b) command should be corrected.
- **NOTE-3 ([POSE-BLIND] heuristic):** the tool printed `[POSE-BLIND] realized d_pose=1.095 >> 0 → …
  w_pose=0`. This is a static heuristic (`d_pose > 1.0`) — a FALSE positive here: the config has
  `--w-pose 1.0` and d_pose is descending (1.887→1.095) at only 2 epochs. Not a real pose-blind config.
- **NOTE-4 (harness):** a foreground n600 MLX run is killed by the bash-harness SIGURG at ~3 min, and the
  n600 startup alone exceeds one 3-min window, so a literal "block foreground for tens of minutes" is
  impossible under this harness. Both jobs were run detached-to-file (python survives SIGURG) and
  actively polled in-turn against the REAL trainer/byte-close logs (real JSON progress markers — not a
  fire-and-forget `.output`-size monitor). The synchronous n600 verdict / byte-close parity are ~15-20
  min each on single-threaded CPU (the real run hides the verdict behind `--async-verdict`).

## Bottom line
- **(a) peak RSS 54.78 GiB < 90 → PASS.**  **(b) d_seg 0.081390 / d_pose 1.095279 / rate 0.002936
  (archive 110248 B) all reproduce through the real byte-closed contest decode → PASS.**
- **VERDICT: GO-READY** for the #205 launch on the axis-9 memory + measurability criteria.
- **BUT flag FINDING-1** (store-nothing pose section 52 KB, not ~1 KB → ~0.035 rate, not ~0.0007) for
  operator decision before trusting the "~0 pose rate" claim in the rate budget. Pointer 0.19110 UNMOVED.
