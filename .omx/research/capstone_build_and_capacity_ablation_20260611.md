# Capstone build + 2x2 capacity-confirm ablation (2026-06-11)

**Subagent:** capstone-builder. **Operator ask:** implement the canonical capstone
vehicle spec's build sequence (`optimal_capstone_vehicle_spec_20260611.md` section 6
steps 1-3), then run the cheap de-risking 2x2 capacity ablation. **Did NOT touch** the
running 48-pair daemon (pid 72123) or the atlas workers. This memo records the BUILD
(3 items + commit + test/parity status) and the ablation (LAUNCH + verdict / partial
trend).

**Authority discipline (CLAUDE.md, binding).** Every number here is
`[macOS-CPU advisory]` / `[macOS-MLX research-signal]`, **NON-PROMOTABLE**
(`promotable=false`, `score_claim=false`, `ready_for_exact_eval_dispatch=false`). torch-CPU
contest `evaluate.py` (600-sample, Linux x86_64) is the ONLY leaderboard authority. NO MPS.
NO paid dispatch fired. The exact frontier pointer is **UNMOVED: 0.19109982 [contest-CPU]**
(ABOVE T_1 -> GOAL UNSATISFIED). This memo is a build + a gated ablation, NOT a pointer move.

---

## 1. THE BUILD (commit `3e082b1ed`, 7 files, +846 LOC; ruff clean; reviewed x2)

All three spec section-6 items landed NO-FAKE + tested + numpy-inflate-parity-proven.

### Item 1 — L1 weight-tie (`tie_depth`) — the rate lever

**Mechanism.** The PR95 taper `[bc,bc,bc,0.75bc,...]` makes ONLY the leading two upsample
blocks `base_ch->base_ch` (block 0/1 share the IDENTICAL conv shape `(bc*4, bc, 3, 3)` +
have no skip_conv; block 2 outputs `0.75*bc` so it is NOT shape-tieable). `tie_depth=N`
shares ONE conv across blocks `0..N-1` with a per-stage learnable FiLM (gamma/beta over
`bc`, identity at init) as the symmetry-breaker. **Max legal `tie_depth=2`** (a contiguous
leading prefix of equal-channel blocks); the canonical run uses `--tie-depth 2`. `tie_depth<=1`
= no tie = byte-identical to the untied decoder (proven: render delta exactly 0.0).

**NO-FAKE proof it REALLY shares weights (the grid-PE fake-parity lesson):**
- EXPORTED/archived render basis at base_ch=24: **114,933 -> 94,149 params (-20,784)** — the
  leading per-block convs are DROPPED from the export, replaced by the ONE `tied_conv` +
  the per-stage FiLMs. The TRAINABLE tree also prunes the dead convs (115,046 -> 94,262),
  so the optimizer + EMA do honest work, not dead-conv churn.
- **int8 decoder blob: 113,234 -> 92,546 bytes (-20,688 ~ -20.2 KB)** at base_ch=24
  (`_int8_brotli` measured) — matches/beats the spec's -10..-16 KB estimate; this is the
  rate lever that brings base_ch=24's ~126 KB int8 archive under the sub-0.19 budget.
- **The numpy inflate REPRODUCES the tied MLX render op-for-op:** parametrized
  base_ch={20,24} MLX<->numpy render drift < 0.05 on [0,255]; the int8-reloaded
  `advisory_quant_gap_d_seg == 0.0` end-to-end; **the slow REAL-scorer joint-loop test
  (EfficientNet-B2 SegNet + FastViT PoseNet, 134.7s) PASSES with the tie active.** A no-op
  tie (no real sharing) FAILS the param-reduction test; a numpy path that did not dispatch
  the tied stages to the shared conv FAILS the parity test.
- The shared conv is ACTUALLY trained (a step moves it) and is covered by the EMA shadow
  (so the exported shadow carries the trained tie).

### Item 2 — Cross-hardware-robust margin hinge (L7) — the numpy-portability guard

**Mechanism.** A NEW loss TERM `margin_hinge_weight * mean(relu(margin_floor - margin))`
over the boundary pixels, ADDED to the active PR95 stage seg-loss (capstone-owned
`cross_hw_margin_hinge.py`). It enforces a margin FLOOR (default 0.1 > the measured ~0.096
cross-hardware logit drift) so the LOCAL SegNet argmax SURVIVES macOS->numpy->Linux/CUDA —
DISTINCT from the existing weight-boost `l7_softplus_seg_loss` (which reweights, not floors).
A capstone-owned wrapper installs it on the torch-CPU bridge and re-points the wrapped base
loss on every curriculum stage transition (so the hinge always stacks on the CURRENT stage
surrogate). The reported authority `exact_d_seg`/`exact_d_pose` stay hinge-free (true argmax).

**NO-FAKE proof it is a REAL loss term (not a constant):**
- A small-positive-margin pixel (0.05 < floor 0.1) gets a POSITIVE hinge AND the gradient of
  the hinge w.r.t. the target logit is NEGATIVE (raising the target logit -> larger margin ->
  lower hinge). A clear-of-floor pixel (margin 5.0) gets EXACTLY 0 hinge.
- `hinge_weight=0` is byte-identical to the bare stage loss (default-off is provably inert).
- **Fail-closed scope (collision discipline):** the hinge wraps the torch-CPU bridge's single
  `seg_loss_fn` (the AUTHORITY + default gradient backend). The shared MLX-GPU bridge computes
  its seg loss by FORM-NAME inside its `value_and_grad` closure (no wrappable hook) and lives
  in `mlx_pr95_port` (a shared lane file) — so `margin_hinge_weight>0` + `scorer_backend=mlx_gpu`
  RAISES at trainer construction rather than silently NOT enforcing the floor in the gradient
  (NO-FAKE). The two are mutually exclusive until the MLX-native hinge is wired (a follow-up
  that edits the shared bridge — out of this lane's collision-safe scope).

### Item 3 — CLI passthrough + FP32-exact GPU scorer

`run_capstone_campaign.py` now exposes `--tie-depth`, `--hinerv-grid-pe`,
`--grid-pe-num-freqs` (default 4), plus `--margin-hinge-weight` / `--margin-hinge-floor`, all
threaded into the bundle config + train config + the `capstone_config_v1` inflate sidecar
(`decode_config_from_bundle` -> `asdict` carries `tie_depth`/`hinerv_grid_pe`/`grid_pe_num_freqs`;
the inflate reads them back). When `--scorer-backend mlx_gpu` is selected the CLI sets
`MLX_METAL_GPU_ARCH=applegpu_g15` BEFORE the first MLX import (the FP32-exact arch override:
243->0 d_seg flips, pose 2.76e-4->8.7e-11, zero throughput cost, decoder unaffected).

### Test status

- NEW `test_capstone_tie_and_margin_hinge.py`: **17 tests PASS** (tie param-reduction +
  base_ch={20,24} numpy parity + byte-identical-at-tie<=1 + trained+EMA-covered + hinge
  real-loss/real-gradient/no-op-at-0/stage-rebase + fail-closed-on-mlx_gpu + CLI flags +
  sidecar tie_depth).
- Existing suites: `test_numpy_reference_parity` + `test_stored_latent_carrier` +
  `test_carrier_independent_fixes` + `test_advisory_and_bicubic_fixes` + `test_film_adamw_routing`
  = **54 PASS, 0 regression**; `test_capstone_vq_nerv` = **23 PASS** incl the slow
  real-scorer joint-loop test (134.7s, --timeout=400). ruff clean on all 7 files.

---

## 2. THE 2x2 CAPACITY-CONFIRM ABLATION (the decisive de-risk; LAUNCHED)

**Design (spec section 6 step 6):** `{base_ch=20, base_ch=24} x {48 pairs, 192 pairs}`,
CE-only (`--curriculum none`), stored_latent carrier, int8 export, `--scorer-backend mlx_gpu`
(FP32-exact arch override), **equal epochs-per-pair (120 epochs all 4 arms)**, warmup-EMA eval
(the landed EMA-warmup fix `f771e6e00` -> the shadow tracks the live weights on short runs, so
the reported d_seg is the real plateau, not a frozen-init artifact). Marker-on-exit per arm +
a final `DONE.marker` (durable-daemon discipline; NOT session-bound).

**Driver:** `experiments/run_capstone_capacity_ablation_2x2.sh` ->
`experiments/results/capstone_capacity_ablation_2x2_20260611/{bc20_p48,bc20_p192,bc24_p48,bc24_p192}/`.
**Lane-claimed** `lane_capstone_capacity_ablation_2x2_20260611` (separate from the running
48-pair daemon pid 72123, which was NOT touched). Launched as a detached nohup daemon
2026-06-11T11:06Z; arm order = the two base_ch=20 arms FIRST (they answer the decisive sign).

**The single deciding number:**
`sign( plateau_d_seg(base_ch=20 @ 192) - plateau_d_seg(base_ch=20 @ 48) )`
- **negative + large -> DATA-LIMITED** (base_ch=20 @ 600 could reach the floor; the cheaper
  vehicle is viable -> revert the bet to base_ch=20).
- **>= 0 -> CAPACITY-LIMITED** (more pairs at fixed 85K params makes the single-video fit
  HARDER -> base_ch=24 is the right scale -> GREENLIT the base_ch=24 @ 600 bet).
ALSO reported: does base_ch=24 reach LOWER plateau d_seg than base_ch=20 at the SAME pairs
(the does-bigger-win check)?

### Ablation status: <!-- updated on completion -->

LAUNCHED + smoke-validated (mlx_gpu builds on the REAL scorer; arch override active;
numpy-parity quant-gap 0.0; ~26s for a 2ep/2pair warmup incl GT-cache + export). The
n=192 GT target cache builds once on the first 192-pair arm (slow precompute, shared).
**Verdict + per-arm plateau d_seg pending the daemon's completion** — read the per-arm
`capstone_result.json` (`d_seg_best` / `trajectory[-1].exact_d_seg`) + the `DONE.marker`.
Partial trend (if not done at hand-off): see the live `*/trajectory.jsonl`.

---

## 3. GREENLIT NEXT STEP

**Gated on the ablation verdict above:**
- IF **capacity-limited** (sign >= 0, AND/OR base_ch=24 < base_ch=20 at fixed pairs):
  proceed to the **base_ch=24 @ 600-pair pointer-moving run** — the spec section 7 command,
  now buildable since `--tie-depth`/`--hinerv-grid-pe`/`--grid-pe-num-freqs` exist:
  ```bash
  # lane-claim first, then (detached daemon, marker-on-exit; 600 GT targets precomputed first):
  OMP_NUM_THREADS=6 .venv/bin/python experiments/run_capstone_campaign.py \
      --max-pairs 600 --base-channels 24 --carrier stored_latent --decoder-dtype int8 \
      --hinerv-grid-pe --grid-pe-num-freqs 4 --tie-depth 2 \
      --curriculum pr95_8stage --optimizer-schedule pr95_adamw_then_muon \
      --curriculum-total-epochs 240 --seg-weight 100.0 --pose-weight 1.0 \
      --scorer-backend mlx_gpu --authority-recheck-every 50 --eval-every 10 --device cpu \
      --targets-cache experiments/results/capstone_gt_targets_cache \
      --out-dir experiments/results/capstone_base_ch24_600pair_mlxgpu_20260611
  ```
  NOTE: the 600-pair bet uses `--scorer-backend mlx_gpu`; the margin hinge is therefore OFF for
  the bet (mlx_gpu + hinge fail-closed). To get the L7 portability guard IN the gradient, either
  (a) run the bet on `--scorer-backend torch_cpu_bridge` with `--margin-hinge-weight`, or
  (b) wire the MLX-native hinge into the shared MLX-GPU bridge (a follow-up). Until then the
  hinge is available on the torch-CPU path only; the FP32-exact arch override already removes
  the GPU-vs-CPU drift that the hinge guards against, so the local advisory transfers under
  mlx_gpu even without the hinge — but the Linux x86_64 contest CPU drift remains the residual
  risk the hinge specifically addresses.
- IF **data-limited** (sign negative + large): reconsider — base_ch=20 @ 600 is the cheaper
  viable vehicle; re-aim the 600-pair run at base_ch=20 (and revisit the decoder-shrink class).

The pointer moves only when the byte-closed 600-pair advisory clears the drift-aware submit
rule (local <= 0.189987) AND a paired contest-CPU exact eval confirms sub-T_1. That run is a
SEPARATE later daemon — NOT fired here (this task ends at the ablation launch + verdict).

---

## 4. NO-FAKE / authority notes
- Every architectural number traces to a measured artifact (param counts, int8 blob bytes,
  the test suite) or an inline arithmetic. No predicted contest S is claimed here.
- All numbers `[macOS-CPU advisory]` / `[macOS-MLX research-signal]`, NON-PROMOTABLE. Only
  Linux x86_64 = `[contest-CPU]`; only NVIDIA T4 = `[contest-CUDA]`. NO MPS.
- The weight-tie REALLY shares weights (param + byte reduction proven) AND the numpy inflate
  reproduces it exactly (the grid-PE fake-parity lesson honored). The margin hinge is a REAL
  loss term with a REAL gradient (not a constant), default-off-inert, fail-closed on mlx_gpu.
- The exact frontier pointer is UNMOVED (0.19109982 [contest-CPU]). This is a build + a gated
  ablation; the means (this build) is not the end (a lower exact S) — the base_ch=24 @ 600 run,
  gated on the ablation, is the unit aimed at the exact CPU-axis row that crosses T_1.
