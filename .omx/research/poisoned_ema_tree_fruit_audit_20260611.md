# Fruit-of-the-poisoned-tree audit: EMA-shadow-lag exposure across the capstone + MLX-port work (2026-06-11)

**Authority:** `[macOS advisory]` — this is a read-only forensic audit, NO code modified, NO training run.
Every claim is file/line/mtime-cited. The exact frontier pointer (`0.19109982 [contest-CPU]`) is UNMOVED and
unaffected (exact `evaluate.py` rows do NOT use our EMA). `score_claim=false`, `promotion_eligible=false`.

**Scope:** the bug is `_CapstoneWeightEMA` constant-decay-no-warmup (0.999/0.997, time constant 333–1000
STEPS) frozen at init on SHORT MLX runs (≈6 steps/epoch). It poisons any number read from the EMA SHADOW
(`exact_d_seg`/`mean_d_pose` with `use_ema_for_eval=True`, OR an archive EXPORTED via
`export_render_weights`/`export_stored_latents`, which byte the shadow). Fixed in `f771e6e00` (warmup decay).

## THE DECISIVE TIMING DISCRIMINATOR (the spine of this audit)

The buggy weight-EMA did NOT exist for most of the capstone work. Git timeline (local, `git log`):

| commit | local time | what it did to EMA/eval |
|---|---|---|
| `cdeff20f3` (curriculum wire-in) and ALL prior | ≤ 2026-06-10 21:17 | **`use_ema_for_eval` default = FALSE**; `exact_d_seg` renders **LIVE** (verified: `git show cdeff20f3:.../capstone_trainer.py` → `use_ema_for_eval: bool = False  # eval LIVE weights (the 0.999-lag landmine)`; the function installs NO shadow) |
| `814680de8` (A1 weight-EMA add) | **2026-06-10 22:31** | introduces `_CapstoneWeightEMA` (constant decay, NO warmup) AND **flips the eval default to `use_ema_for_eval=True`** → eval reads the (frozen) shadow. THE POISON WINDOW OPENS. |
| `f771e6e00` (warmup fix) | **2026-06-11 00:56** | warmup decay `min(decay,(1+t)/(10+t))`; shadow tracks live from step 1. POISON WINDOW CLOSES. |

**Therefore the poison window is a NARROW 2h25m band (22:31 → 00:56 on 06-10/11).** Any number produced by a
run whose trajectory/result mtime is OUTSIDE that band — and that used `use_ema=False` — is **NOT poisoned**.
This is far narrower than the bug memo's worst-case framing ("any capstone archive exported pre-`f771e6e00`
on a short run is suspect"): in practice almost all of the *cited* capstone descent numbers predate the buggy
EMA and were measured LIVE.

## 1. THE TABLE (claim | source | verdict | re-validation)

| # | Claim | Source (file:line / memo / result-dir mtime) | Verdict | Re-validation action |
|---|---|---|---|---|
| 1 | **"d_seg frozen 0.505 / seg-capacity wall" (curriculum)** | `experiments/results/capstone_curriculum_b20_n48/trajectory.jsonl` (mtime **00:39**, INSIDE window); bit-identical `exact_d_seg=0.5053805311520895` across stage1 while `seg_loss_mean` drops 1.32→0.14 | **POISONED** (already reversed by `f771e6e00` verdict memo) | DONE in spirit — re-run is `capstone_c1prime_honest_b20_n48` (post-fix). Re-measure LIVE d_seg over the full curriculum. |
| 2 | Curriculum stage2 "d_seg descends 0.504→0.498" | same file, ep55–65 rows | **POISONED** (shadow slowly catching up under the lagging decay, NOT a live capacity readout) | Re-measure LIVE; the 0.498 is a lower bound on shadow lag, not a real d_seg. |
| 3 | **"CE plateaus ~0.008 / seg-walled" (the LONG run); deltas decay geometrically 0.49/0.33** | `capstone_adversarial_synthesis_…20260611T015018Z.md` §2; `capstone_campaign_launch_…20260610.md` correction banner; source data = `experiments/results/capstone_daemon_b20_n48_LONG/trajectory.jsonl` (mtime **20:35**, PRE-EMA-add) | **NOT poisoned** | None for the *measurement* (it is LIVE: that run had NO EMA; `use_ema_for_eval` defaulted False; d_seg = 0.0117→0.0080, visibly descending, NOT frozen). The *interpretation* "plateaus ~0.008, cannot cross 15×" stands on live data — but it's a CE-only-recipe extrapolation, separately re-openable by the curriculum, not by the EMA fix. |
| 4 | **"best capstone S ≈ 1.75 (9× from frontier)"** (d_seg 0.00838, d_pose 0.0724, 97,025 B at LONG ep40) | `capstone_adversarial_synthesis_…` §3; `capstone_campaign_launch_…` banner; LONG ep40 row | **NOT poisoned** (its d_seg/d_pose are LIVE, pre-EMA) | None — S≈1.75 is a faithful recompute from live components. (Caveat unrelated to EMA: d_pose used the pre-#4-fix clamp-only path on the LONG run; that *understates* d_pose, a separate apples-to-apples concern, not EMA.) |
| 5 | **"the small basis fights the physics / needs MORE params for lower d_seg"** | `capstone_adversarial_synthesis_…` "STRATEGIC CORRECTION"; `capstone_pr95_fullstack_…` Conclusion-1 | **UNCERTAIN → REOPENED** | The argument rested PARTLY on the frozen-0.505 curriculum (POISONED, claim #1) AND partly on the live LONG-run 0.008 plateau (NOT poisoned). The poisoned half is gone; the live half (CE plateaus ~0.008) remains but is a *recipe* plateau, not a *capacity* wall. **#1 re-measurement target: does the FIXED-EMA curriculum (loss-form schedule) drive LIVE d_seg below 0.003 at base_ch=20?** This is the single claim the fix most reopens. |
| 6 | Original recipe-validation: **"d_seg 0.507→0.0103 (49×), d_pose 140→0.03"** (12-pair, base_ch=36) | `capstone_original_small_vq_basis_20260610T214151Z.md` §3; result `capstone_recipe_validation_real_scorer_aggr_20260610.json` (run **21:41Z**, PRE-EMA-add) | **NOT poisoned** | None — ran under `use_ema_for_eval=False` default (`ema=0.95` is the *codebook* EMA, not the weight-shadow eval). Live measurement. |
| 7 | "12-pair smoke: d_seg 0.5073→0.0117 (43×) on the EXACT scorer" | `capstone_campaign_launch_…` line 44; `.omx/tmp/capstone_real_recipe.log` | **NOT poisoned** | None (pre-EMA, live). |
| 8 | **stored_latent vs vq_index pose A/B: d_seg held 0.5073→0.5073 for BOTH carriers** | `capstone_stored_latent_carrier_…md` §smoke; result dirs `capstone_{stored_latent,vq_index}_smoke_b20_n8` (mtime **23:25/23:32**, INSIDE window) | **d_seg numbers POISONED but IMMATERIAL** | The d_seg=0.5073 is the frozen shadow — BUT these runs are `curriculum=none`, 8 pairs, 30 ep, so LIVE d_seg would also be ~0.5 (too few epochs to descend). The memo already disclaims "this carrier fixes POSE not seg." No d_seg conclusion was drawn from them. **The d_pose A/B (the actual claim) is NOT poisoned** (see #9). |
| 9 | **stored_latent ends 21.5% lower d_pose than vq_index (85.11 vs 108.42); pose monotonic descent** | `capstone_stored_latent_carrier_…md` A/B table; same smoke dirs | **UNCERTAIN — VERIFY which use_ema path** | These d_pose values came from `mean_d_pose` during runs INSIDE the poison window with `use_ema_for_eval=True` by default. BUT the **slow pose path tracks the shadow** (its timescale ≈ the EMA time constant — the bug memo's own observation), so a *relative* A/B at matched config is likely directionally valid. Still: re-measure the pose A/B with `use_ema=False` (LIVE) to confirm the 21.5% gap is real and not a shadow-lag artifact differing between carriers. **Medium priority.** |
| 10 | **Carrier-pivot diagnosis: "8-bit VQ index can't encode 600 ego-motions → d_pose oscillates 0.06–0.34"** | `capstone_carrier_pivot_…md`; `capstone_optimal_carrier_design_…md` §1.2 (pose intrinsic dim=1.00, 21 bits) | **NOT poisoned** | None — grounded in the `carrier_intrinsic_dim_probe` ($0, intrinsic-dimension math on the GT pose store) + the #57 exact-CPU RD sweep, neither of which touches our EMA. The "0.06–0.34" oscillation is from the *converged* regime cited from the Quantizr-pose audit, not from a shadow read. |
| 11 | Byte-budget table (base_ch=16 int8 = 71,968 B rate 0.0479; ~0.983 B/param; int8 perturbation 0.2–0.4%) | `capstone_original_small_vq_basis_…214151Z.md` §2; `capstone_vq_nerv_byte_budget_20260610.json` | **NOT poisoned** | None — exact `len(brotli(...))` on bundle weights; dtype/codec measurement, no scorer, no EMA. |
| 12 | Projected S 0.1355 / 0.1523 at "Quantizr-class target operating point (d_seg=5.6e-4, d_pose=1e-4)" | `capstone_original_small_vq_basis_…` §2; `capstone_optimal_carrier_design_…` §3 | **NOT poisoned (but PROJECTION)** | None for poison; these are explicitly TARGET-operating-point projections, not measured d_seg/d_pose. They assume the descent the FIXED-EMA curriculum must now demonstrate (ties to #5). |
| 13 | numpy-inflate score-parity: **d_seg |Δ|=0.0 (EXACT)**, reloaded-int8 == live render | `capstone_numpy_inflate_portability_…md`; `advisory_quant_gap_d_seg=0.0` in result.json | **NOT poisoned** | None — a render-PARITY gate (numpy vs MLX on the SAME weights). Independent of whether those weights are shadow or live; it proves the codec is exact, not that any d_seg value is good. |
| 14 | Throughput profile (scorer fwd+bwd = 97% of step; batch amortization 3.5×; fused==separate bit-identical) | `capstone_training_throughput_…md`; `…json` | **NOT poisoned** | None — pure timing + numerics-invariance; no EMA-dependent score claim. |
| 15 | **validation_micro EXPORTED archive: d_seg 0.507, reloaded-int8 d_seg 0.507, S 87.4** | `experiments/results/capstone_validation_micro/capstone_result.json` (mtime **22:29**, window edge; `use_ema_for_eval: true`) | **POISONED export — but IMMATERIAL (toy run, live also ~0.507)** | This is the concrete export-poisoning signature (`d_seg_final == d_seg_init == reloaded_int8_d_seg == 0.5073`). On THIS run live d_seg was also ~0.507 (8 pairs/30 ep curriculum=none), so the archive is not *worse* than reality — but it is a worked example that the export DID byte the frozen shadow. No good-d_seg claim attached. |
| 16 | pr95-port `mlx_trainer.py` verdicts (d_seg 0.505→0.0106 in 14 ep, etc.) | `src/tac/mlx_pr95_port/mlx_trainer.py:101` `use_ema_for_eval=False`; `:395/:473` `exact_d_seg(use_ema=False)`; `pose_film_trainer.py:141` `use_ema_for_eval=False` | **NOT poisoned (CONFIRMED)** | None — the pr95 port deliberately eval'd LIVE (its docstring `:17` names "the lagging EMA-0.999 shadow" as the thing it avoids). As the prompt stated, these verdicts are clean. |

## 2. TOP RE-MEASUREMENTS, ranked by strategic impact

1. **(HIGHEST — already in flight) Is the small basis seg-walled? Re-run the curriculum with the FIXED EMA, reporting LIVE d_seg.** Claim #1 (frozen 0.505) is the poisoned linchpin of the whole "small basis fights the physics" thesis (#5). `experiments/results/capstone_c1prime_honest_b20_n48/` (post-fix, mtime 01:03) is the replacement run. The decisive number: does LIVE d_seg cross below ~0.003 under the loss-form curriculum at base_ch=20? If yes → the small-basis thesis is REOPENED and the strategic picture flips back toward the operator's instinct ("the smaller learned is not dead").
2. **Confirm the stored_latent-vs-vq_index pose A/B with `use_ema=False` (claim #9).** The 21.5% pose advantage gates the entire carrier decision (vq_index vs stored_latent vs mask). The slow-pose-path-tracks-shadow argument makes the *relative* result probably-robust, but a LIVE re-measure removes the only remaining doubt cheaply.
3. **Re-classify the "CE plateaus ~0.008" interpretation (claims #3/#5) as recipe-bound, not capacity-bound.** The *measurement* is live/clean, but it was being used (in the "fights the physics" argument) jointly with the poisoned curriculum number. Decouple: ~0.008 is a CE-ONLY-recipe floor on the LONG run; whether the curriculum re-accelerates is the open question, now testable on clean (post-fix) telemetry.

(Lower priority: nothing else materially changes strategy — the byte-budget, intrinsic-dim, throughput, and numpy-parity results are all EMA-independent.)

## 3. EXPORTED capstone archives that may carry near-init shadow weights

On-disk `archive.zip` inventory (mtime; bytes; poison verdict). The buggy-EMA window is **22:31→00:56**:

| archive | mtime | bytes | exported via shadow? | real-d_seg risk |
|---|---|---:|---|---|
| `capstone_smoke_b16/archive.zip` | 17:59 | 64,369 | **No** (pre-EMA-add; export path pre-A1) | none |
| `capstone_timing_probe/archive.zip` | 18:04 | 64,202 | No (pre-EMA) | none |
| `capstone_validation_micro/archive.zip` | 22:29 | 61,113 | **Yes** (`use_ema_for_eval:true`) | **shadow-bytes, but live≈shadow≈0.507 on this toy run → archive not worse than reality; NO good-d_seg claim** |
| `capstone_smoke_stored_latent_b20_n8/archive.zip` | 23:01 | 86,545 | Yes (window) | toy 8-pair carrier smoke; d_seg≈0.5 either way; pose-only claim |
| `capstone_smoke_vq_index_b20_n8/archive.zip` | 23:02 | 91,082 | Yes (window) | same — toy, no d_seg claim |
| `capstone_stored_latent_smoke_b20_n8/archive.zip` | 23:26 | 86,421 | Yes (window) | same |
| `capstone_vq_index_smoke_b20_n8/archive.zip` | 23:32 | 91,126 | Yes (window) | same |

**Material finding (NO-FAKE):** four+ archives WERE exported through the buggy shadow, BUT **none of them is a
trained-descent candidate** — every one is a toy 2–8-pair smoke whose LIVE d_seg was also ≈0.5 (too few epochs
to descend), so the shadow did not silently corrupt a *good* archive into a *bad* one. The two runs that
actually descended d_seg (the LONG daemon → 0.008; the 12-pair recipe-validation → 0.010) were **never
byte-closed/exported** (the LONG dir contains only `trajectory.jsonl`; no archive). **Conclusion: there is no
shipped capstone archive on disk that claims a good d_seg it does not actually carry.** The export-poison risk
is real and now structurally fixed, but it has NOT yet produced a corrupted promotion-candidate archive — the
fix landed before any descent-quality run was exported. Any FUTURE export must be on a post-`f771e6e00` run (or
a long run >~3000 steps where the shadow caught up).

## Cross-references
`capstone_ema_shadow_lag_reverses_seg_wall_verdict_20260611T070000Z.md` (the bug + proof) ·
`capstone_adversarial_synthesis_and_honest_corrections_…md` (S≈1.75, the ~0.008 plateau — LIVE/clean) ·
`capstone_pr95_fullstack_definitive_audit_synthesis_…md` (Conclusion-1, the half that was poisoned) ·
`src/tac/capstone_vq_nerv/capstone_trainer.py:173,377,407,662,686` (the shadow eval+export surfaces) ·
`src/tac/mlx_pr95_port/mlx_trainer.py:101` + `pose_film_trainer.py:141` (the CLEAN live-eval ports) ·
git `cdeff20f3`/`814680de8`/`f771e6e00` (the poison-window boundaries).
