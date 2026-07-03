# #205 STORE-NOTHING-POSE launch config + extreme-iterated recursive-adversarial review — 2026-07-03

**Task:** design + review-only the launch-ready config for **#205 — the d_seg-convergence pointer-mover**,
now ALSO carrying **optimal store-nothing pose** (operator steer 2026-07-03, symposium §8). This is a
DESIGN + REVIEW deliverable. It does NOT launch the heavy run (operator-GO-gated). Pointer **0.19110
UNMOVED** — everything here is a MEANS `[macOS-MLX advisory / design]`; only a byte-closed `evaluate.py`
n600 row < 0.19110 moves it.

**Headline:** the store-nothing-pose #205 config is **already wired end-to-end** as the canonical
`derive_store_nothing_205_config` → `tools/launch_witness_run.py --config store_nothing_205`. I did NOT
need to hand-assemble an argv or wire a missing lever. The launcher **flag-validated 85/85 flags** against
the real trainer argparse and **memory-preflighted the REAL n600 config at 67.61 GiB projected peak ≤ 89.6
GiB safe ceiling (SAFE)**, system-admission **ADMIT**. **VERDICT: PROCEED** (launch-ready) — with the
axis-9 measured-runnability smoke as the pre-burn gate + one operator DECISION (three optional deferred
surgical levers, documented below).

---

## 1. The exact launch-ready invocation (flag-validated, memory-safe)

### One-command launcher (recommended — auto-derives + validates + mem-preflights + admission-gates)
```bash
.venv/bin/python tools/launch_witness_run.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 --epochs 1000 \
  --config store_nothing_205
# add --dry-run for the $0 CPU-only validation pass (what this review ran);
# drop --dry-run ONLY on operator GO, AFTER the axis-9 smoke (§5) passes.
```

`--config store_nothing_205` (`tools/launch_witness_run.py:281-284`) resolves to
`tac.witness_autoconfig.derive_store_nothing_205_config` (`src/tac/witness_autoconfig.py:998`) = the
Phase-3 **SEALED** capstone (`derive_sealed_205_config`, `witness_autoconfig.py:940`) **+** the single
store-nothing delta `--pose-carrier-source generated` (`witness_autoconfig.py:704-706`).

### The full expanded argv the launcher emits + validated (the `launch.sh` body)
Emitted + **`flag validation: 85/85 flags exist in the trainer argparse`** (dry-run, 2026-07-03):
```
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 \
  .venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py \
  --out-dir <RUN_DIR> --gt-cache .../gt_n600.npz --num-pairs 600 \
  --mlx-device gpu --seed 0 --epochs 1000 --eval-every 25 \
  --verdict-pairs 0 --async-verdict --verdict-batch 32 \
  --curriculum --tau-softplus-start-epoch 300 --tau-softplus-tau 0.3 \
  --l7-start-epoch 1000 --muon-start-epoch 726 --muon-lr 0.002 \
  --muon-momentum 0.95 --muon-ns-steps 5 \
  --stage-transition-rewarmup-epochs 8 --stage-transition-rewarmup-floor 0.1 \
  --stage-transition-rewarmup-shape linear --stage-transition-reset-moments \
  --w-seg 100 --w-pose 1.0 --score-domain-loss \
  --pose-carrier --pose-carrier-residual-mode table --pose-carrier-source generated \
  --mod-dim 32 --hidden-dim 96 --n-hidden 4 \
  --activation hosc --hosc-beta 1.0 --hosc-beta-end 4.0 --hosc-beta-anneal linear --hosc-omega 1.0 \
  --siren-init --softmax-temp-start 1.0 --softmax-temp-end 0.05 --tau-anneal-shape cosine \
  --self-orient --n-dir-freqs 2 --freq-across 32 --freq-along 4 --reorient-every 50 --max-bank-freq 64 \
  --chroma --palette-anchor --eikonal-weight 0.01 --length-weight 0.001 \
  --render-h 384 --render-w 512 \
  --render-aa none --lane-render-band --lane-band-start-epoch 300 \
  --lane-band-uncertainty-source witness --lane-band-tau 0.85 --lane-band-eps 0.35 \
  --lane-band-softness 1.0 --lane-band-dash-forward-max-m 55.0 --lane-band-weight 1.0 \
  --persistence-loss-weight 1.0 --persistence-recall-weight 1.0 --cldice-iters 5 \
  --persistence-warmup-epochs 300 --persistence-classes auto \
  --amplify-weight 1.0 --amplify-form hinge --amplify-margin-target 1.0 \
  --amplify-persist inverse_thickness --island-dilate-px 1 \
  --structured-init --structured-init-include-lane --lane-prior-phi1 \
  --lane-prior-phi1-mode replace --lane-prior-phi1-dash-gate \
  --accum-pairs 8 --grad-clip 1.0 --ema-decay 0.997 \
  --lr 1e-3 --lr-end 1e-4 --weight-decay 1e-4 --adam-beta2 0.999 \
  --ckpt-every 25 --stage-checkpoints
```
Flag-existence is enforced structurally: `tools/launch_witness_run.py:71-81` greps every real
`add_argument("--flag"...)` from the trainer and **REFUSES to write launch.sh** if any emitted flag is
not in that set (never-invent-a-flag, NO-FAKE). 85/85 passed.

---

## 2. Per-lever map — flag → routing call-site → deep-math → store-nothing/0-rate

### (A) The store-nothing-pose gauge (operator steer §8; 0 pose rate)
| lever | flag(s) | routes at | deep-math / 0-rate confirmation |
|---|---|---|---|
| pose loss ON | `--w-pose 1.0` | `total_loss_fn`→`base_loss` (`:2097-2098`) → d_pose through render_fn→R→frozen PoseNet | R1 floored d_pose at 0.0011 ONLY because `w_pose=0`; turning it on is THE primary store-nothing lever. `--w-pose` default 0.0 (`:3582`). |
| frame0 pose carrier | `--pose-carrier` | built `:1537`; **guard `:1543` raises if w_pose≤0**; **child-attach `:1587`** (dxi ∈ `model.trainable_parameters()` → EMA/AdamW/Muon-tracked) | SegNet reads **frame1 only** (`x[:,-1]`), so **frame0 is entirely in the SegNet null** → pose carrier at ~0 d_seg cost. This IS symposium §8's "paint pose-legibility into the SegNet-null," realized cleanly (frame0 is 100% null, not just within-cell). |
| **STORE-NOTHING source** | `--pose-carrier-source generated` | render_fn `:1815-1835`: frame1(odd)→witness render; **frame0(even)→warp(the witness's OWN up-to-camera render, ξ_eff)** | Stores **only ξ/H** — no keyframe image. Byte-close MEASURED BIT-EXACT: pose section **697941 B (real-keyframe table) → 1049 B (store-nothing)** ⇒ rate ≈ 25·1049/37.5M ≈ **0.0007 (~0 pose rate)**. `witness_autoconfig.py:1029-1035`. |
| ξ residual param | `--pose-carrier-residual-mode table` | `:1580-1587`; dxi = per-pair (P,6) table | SEAL Q4/Q2 chose `table` (byte-minimal (P,6); cos 5.9e-5 → seg⊥pose additive-S attribution safeguard). `film` (code-cond MLP) is the wired alt (`:1811-1813`) — NOT used (see review P1-a). |
| ξ from ego SCREW (canonicalize-to-ground) | `--pose-carrier-s-r 0.0 --pose-carrier-pitch 0.0`, `--pose-carrier-s-t` (None→**self-fit**) | `GroundHomographyGeom.eon` + `xi_from_pose_calibration` `:1554-1579`; s_t self-fit on the frozen CPU-torch PoseNet d_pose grid (`:1566-1576`) | #193 canonicalize-to-ground-frame is realized IN the carrier geom (road ≈ planar homography; the Chasles screw ξ dual-use — same ξ that would warp d_seg IS the pose target). Deterministic, GT-derived, **never MPS**. |
| seg⊥pose decoupling (#227) | *(structural — no flag)* | d_seg lever forward uses **c1 only** (`:2123 _render_R(model, cf, c1, ...)`); frame0∉SegNet | The pose gradient lives on frame0 (SegNet-invisible) → it **cannot fight d_seg by construction**. #227 is realized structurally, not as a soft penalty. |

### (B) The d_seg-convergence levers (#205 core — the pointer-mover)
| lever | flag(s) | routes at | deep-math / 0-rate |
|---|---|---|---|
| structured-init static-core partition (#FEED-ef) | `--structured-init --structured-init-include-lane` | `:3886` | rule-118 FREE generic same-rig geometry → 0 archive bytes (ships TRAINED weights). |
| openpilot deg-3 lane prior (#203/#213/FEED-fs) | `--lane-prior-phi1 --lane-prior-phi1-mode replace --lane-prior-phi1-dash-gate` | `:3954`; requires structured-init (guard `:4214`) | centerline = Road↔Lane separatrix (residual 1.9e-5); FREE generic geometry. |
| analytic lane render-band (#203/#213/#215) | `--lane-render-band --lane-band-*` (`--lane-band-start-epoch 300`) | `#224` unified render `:3999` | class-1 render-time authority; O(1)/pixel (decode-in-budget); witness-uncertainty FP gate. |
| AA — **coverage-integrated, NOT supersample** (#220) | `--render-aa none` (+ the analytic band above) | `:3976` | **SIGNAL-A/B lesson honored**: brute `supersample` HURTS the witness −49% + busts the 30-min decode; NONE + analytic band is the contest-feasible optimum. `witness_autoconfig.py:868-876`. |
| persistence/topology (#218) | `--persistence-loss-weight 1.0 --persistence-recall-weight 1.0 --cldice-iters 5 --persistence-warmup-epochs 300 --persistence-classes auto` | `:4047` | births the finest-scale erasure tail (lane dashes) the CE drops; ∝1/persistence. |
| island-birth amplification (#208) | `--amplify-weight 1.0 --amplify-form hinge --amplify-margin-target 1.0 --amplify-persist inverse_thickness --island-dilate-px 1` | `:4076`; rides the SHARED LEVER-4 `_signed` margin (`:2117`) | up-weights thin-island birth; 0 added bytes. **This is #205's #208 realization** (see review P2-a re: the `--seed-islands` early-seed variant). |
| directional all-class Fourier basis | `--self-orient --n-dir-freqs 2 --freq-across 32 --freq-along 4 --reorient-every 50 --max-bank-freq 64` | `:3637` | orient curvelet feats to the all-class boundary tangent (−48% d_seg exponent); byte-close deterministic (rule-118 FREE). |
| chroma (d_seg lever) | `--chroma` (+ `--palette-anchor`) | `:3620` | SegNet reads RGB → chroma flips the argmax at the boundary annulus (triple-use: also feeds PoseNet). |
| curriculum (l7 DEMOTED) | `--curriculum --tau-softplus-start-epoch 300 --l7-start-epoch 1000 --muon-start-epoch 726 --muon-lr 0.002` | `:3671-3676`, `:3908`; guard `:4105` | CE→tau@300→(l7 demoted to epochs=1000 per the measured L∞-defect)→Muon@726 (PR95 stage-8, "THE drop"). |
| annealed hosc (NOT fixed β=4) | `--activation hosc --hosc-beta 1.0 --hosc-beta-end 4.0 --hosc-beta-anneal linear` | `:3652-3666` | **CLAUDE.md divergence fix honored**: fixed β=4 DIVERGES (tanh saturation→vanishing grad); anneal 1.0→4.0 as the SDF pins → step-native L∞-optimal chart, no Gibbs. |
| stage-transition treatment | `--stage-transition-rewarmup-epochs 8 --stage-transition-rewarmup-floor 0.1 --stage-transition-reset-moments` | `:3931-3944` | "different stages need different treatment" — LR re-warmup + AdamW moment reset at ce→tau→l7 boundaries. |
| #222 small-n beta2 | `--adam-beta2 0.999` | `:3560`, bias-correction `:1617` | SEAL Q5: 0.999 == MLX default → byte-identical, no bias-correction confound on the first attribution row (over the all-levers 0.9999999 anchor). |

### (C) Resumability + determinism + the #205 OOM fix (non-negotiables)
| requirement | flag(s) | confirmation |
|---|---|---|
| per-stage PRESERVED checkpoints | `--stage-checkpoints` (ON) | `:3483` — stage-encoded, byte-close-loadable, EMA-shadow saved, atomic; loop-end-only FORBIDDEN. |
| rolling crash-window ckpt | `--ckpt-every 25` | `:3479` — bounds a crash/OOM to ≤25 epochs of loss + enables early byte-close. |
| resumable-from-disk | `--resume-from` available (default None) | `:3487` — restores decoder + per-pair codes + EMA shadow + optimizer + epoch. |
| seeded/deterministic | `--seed 0`; numpy-fp32 verdict authority; MLX-GPU = training-gradient ONLY | verdict is fp32 one-codepath (`--verdict-pairs 0` = ALL 600, n600-scale telemetry, never MPS). |
| **#205 OOM fix** | `--verdict-batch 32` (EXPLICIT) | `:3606`, routes to `_verdict_dseg_dpose_chunked(vbatch=…)` (`:2423/2484`) — chunks the CPU-scorer verdict spike (+66 GiB unchunked → +6 GiB floor) that killed the last n600 launch. Emitted EXPLICITLY to self-document + let the mem-preflight parse the real value. |

---

## 3. UNWIRED levers / blockers

**No hard blockers.** Every lever the store-nothing #205 config needs is a real, wired flag (85/85
argparse-verified) and routes to a real (non-fake) call site (traced in §2). The store-nothing-pose gauge
is fully wired and NO-FAKE (the generated-source render path at `:1815-1835` genuinely warps the witness's
own render; the byte-close section shrink 697941→1049 B is MEASURED bit-exact).

**One clarification (NOT a blocker) — "FiLM-condition the render on ξ (#206)":** there is **no flag that
FiLMs the WITNESS render itself on ξ**. The store-nothing mechanism does **not need one** — the frame0
pose-carrier warp (`frame0 = warp(render, ξ)`) IS the render consuming the twist (#206's intent). The
`film` value of `--pose-carrier-residual-mode` FiLMs only the small dxi RESIDUAL MLP (a different, optional
thing); the SEAL uses `table` (byte-minimal). If a future arm wants an explicit ξ-FiLM of the main render,
that is genuinely UNWIRED and would need a new lever — but it is out of scope for #205 store-nothing.

---

## 4. Recursive-adversarial review log (config surface)

**Pass 1 — flag existence + pose routing.** 85/85 flags exist (launcher grep-gate). Traced pose:
`--pose-carrier` guarded to require `--w-pose>0` (present, 1.0); dxi child-attached → EMA/opt-tracked;
generated-source render_fn dispatches frame1→witness / frame0→store-nothing warp; w_pose routes into
`base_loss` d_pose term. Findings P1-a (FiLM-on-ξ = warp, not a flag — correct), P1-b (canonicalize =
ground-homography geom + s_t self-fit — correct), P1-c (seg⊥pose = structural frame0∉SegNet — correct).
**CLEAN** (all clarifications confirm the config is correct).

**Pass 2 — d_seg levers + curriculum + stability.** Confirmed: annealed hosc 1→4 (NOT the divergent fixed
β=4), l7 demoted to epochs, render-aa none + analytic band (SIGNAL-A/B honored), curriculum guard passes
(0<300<1000≤1000). **muon@726 < l7@1000 → `muon_finisher_WARN` — EXPECTED** (l7 intentionally demoted; a
WARN not an error, `:4185`). **FINDING P2-a (surfaced for operator DECISION):** the task's lever list
names `--seed-islands` (#208 early-seed lane+movable at ep0) and implies margin-saliency (#218 LEVER-4) +
the margin-field head (#218 `--head`/`--margin-field-head-weight`). The SEALED config realizes #208 via
`--amplify-weight` (island-birth) and #218 via `--persistence-loss-weight`, but does **NOT** emit
`--seed-islands`, `--margin-saliency-weight`, or `--head`/`--margin-field-head-weight`. These are the
**deferred surgical levers** kept OFF for the attribution-clean FIRST run per
`witness_autoconfig.lever_priors` (they add no params → land as a shape-compatible warm-start
re-treatment). **This is a deliberate reviewed choice, not an omission bug — but it is a gap vs the task
prose and is an operator decision** (see §4.1). Counter reset.

**Pass 3 — resumability / determinism / memory / axis-9.** Per-stage + rolling checkpoints ON, EMA-shadow,
`--resume-from` available, seed 0, numpy-fp32 verdict (never MPS). Memory: projected 67.61 GiB ≤ 89.6 GiB
(SAFE); system-admission ADMIT (1 active job = R1). `--verdict-batch 32` chunks ALL verdicts incl. v0 (the
OOM driver). **FINDING P3-a/b (smoke-plan refinement, not a config bug):** the axis-9 memory MEASUREMENT
must EXECUTE the real config (the mem-preflight is a PROJECTION, not a measurement); the cleanest peak read
adds `--no-async-verdict` (deterministic phase markers) + a tiny epoch budget to the smoke (see §5). Config
itself unchanged/clean; counter reset for the smoke-plan.

**Pass 4 — re-review with P2-a documented + smoke plan refined.** Re-examined the store-nothing byte
accounting (ξ table ~1 KB counted, ~0.0007 rate — negligible), the s_t self-fit (deterministic GT-derived
startup calibration, 24 pairs × 8-pt grid, never MPS), the generated-source verdict path (`:2311` uses the
witness render, consistent with training). No new config bugs. **CLEAN.**

**Pass 5 — confirmatory.** 85/85 flags; memory-safe; resumable; deterministic; annealed-hosc; l7-demoted;
verdict-batch-32; store-nothing pose wired + routed + byte-close-measured; seg⊥pose structural. No new
bugs. **CLEAN.**

**3 consecutive clean passes on the launch CONFIG itself (P1, P4, P5).** The two counter-resets (P2, P3)
are (a) an operator DECISION on 3 optional deferred levers and (b) a smoke-PLAN refinement — neither is a
config defect.

### 4.1 Operator DECISION (the ONE open item) — the 3 deferred surgical levers
**Recommendation: launch the SEALED `store_nothing_205` as-is (attribution-clean FIRST row), and land the
3 levers as a shape-compatible warm-start re-treatment** — per `lever_priors` + the OPTIMAL-FORM discipline
(don't stack unproven levers on the same run as the arch+scale+pose changes = a 4-way confound with no
clean attribution). They add no params, so a `--resume-from` warm-start re-treats them cleanly later.

If you WANT the fuller stack in the first launch, the exact ready deltas (all real flags, correctly gated;
re-run the launcher `--dry-run` to re-validate + re-project memory after adding):
```
# #208 early-seed lane+movable islands at ep0 (needs --structured-init [ON]; ships 0 bytes):
  --seed-islands
# #218 LEVER-4 all-class margin-saliency (gate to the tau stage per the OPTIMAL-FORM start-epoch guard):
  --margin-saliency-weight 1.0 --margin-saliency-start-epoch 300   # (+ --margin-saliency-uniward optional)
# #218 margin-field head (rare-class logit adjustment; byte-free):
  --margin-field-head-weight 1.0 --logit-adjust-per-class --logit-adjust-tau 1.0
```
Note: these currently require invoking the trainer with the expanded argv + deltas (the
`store_nothing_205` config renderer does not emit them); the launcher's flag-gate + mem-preflight still
apply if you feed the edited launch.sh.

---

## 5. Axis-9 measured-runnability + measured-scored-quantity SMOKE PLAN (the pre-burn GATE — you run it)

The mem-preflight (67.61 GiB) is a **PROJECTION**. Axis-9 demands a **MEASUREMENT** at the REAL n600 config
BEFORE the multi-hour burn: (a) peak RSS, (b) d_seg/d_pose/rate through the real byte-closed decode. Both
tools are built-in.

### (a) Peak-RSS measurement (exercises the two OOM drivers: resident `cf_mx_cache` + v0-verdict spike)
`TAC_MEM_PROBE=1` makes the trainer self-report RSS at `after_cf_mx_cache_build`, `before_v0_verdict`,
`after_v0_verdict` (`experiments/train_levelset_witness_realized_through_R_mlx.py:2658/2830/2836`) with
`mx.reset_peak_memory()` per epoch — the two OOM drivers are BOTH hit at startup, so a tiny epoch budget
suffices. Take the emitted `launch.sh` argv (from the launcher, or `--dry-run`) and run it FOREGROUND with
these SMOKE-ONLY deltas (deterministic sync peak + a byte-closeable checkpoint by ep1):
```bash
# 1) emit + validate + write launch.sh to a scratch out-dir (CPU-only, safe):
.venv/bin/python tools/launch_witness_run.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 --epochs 1000 --config store_nothing_205 \
  --out-dir experiments/results/sn205_axis9_smoke --dry-run --no-dashboard

# 2) run the REAL n600 config FOREGROUND for a bounded smoke, with the peak-RSS probe on.
#    (edit the emitted launch.sh command: swap --epochs 1000 -> 2, --ckpt-every 25 -> 1,
#     --async-verdict -> --no-async-verdict; keep EVERYTHING else — same n600, same cf_mx_cache,
#     same --verdict-batch 32, same self-orient => same peak path.)
TAC_MEM_PROBE=1 TAC_MEM_PROBE_EPOCHS=2 TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 \
  .venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py \
  --out-dir experiments/results/sn205_axis9_smoke \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 \
  --epochs 2 --ckpt-every 1 --no-async-verdict --verdict-batch 32 \
  <...all other flags from the emitted launch.sh, UNCHANGED...>
```
**GATE:** the `mem_probe … "phase":"after_v0_verdict"` row's `rss_gib` must land near the projected
**67.6 GiB** and **< ~90 GiB** (safe_run's `--rss-cap-mb 90000` backstop). If it exceeds → do NOT launch;
raise `--verdict-batch` / reduce scope. (This is exactly the measurement the last launch skipped.)

### (b) Scored-quantity measurement through the REAL byte-closed decode
On the smoke checkpoint, run the byte-close → inflate → frozen-CPU-torch parity tool (measures
`archive.zip` st_size = rate, and d_seg/d_pose on the INFLATED frames — `tools/witness_byte_close_and_eval.py`,
argparse `:628-644`):
```bash
.venv/bin/python tools/witness_byte_close_and_eval.py \
  --ckpt-dir experiments/results/sn205_axis9_smoke \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --max-pairs 600 --keep-packet \
  --out reports/sn205_axis9_byteclose.json
```
**GATE:** the report must (i) produce an `archive.zip` and a MEASURED rate, (ii) reproduce the pose decode
and emit a real **d_pose** (the store-nothing symposium flagged that a prior byte-close "could not even
reproduce the pose decode to MEASURE d_pose" — proving the store-nothing pose section byte-closes + decodes
+ yields all three scored quantities IS the axis-9 gate), and (iii) show the store-nothing pose section is
~1 KB (not ~698 KB). d_seg/d_pose values will be UNCONVERGED on a 2-epoch checkpoint — that is EXPECTED;
axis-9 (b) proves the pipeline is RUNNABLE + all three scored quantities are MEASURABLE at the real config,
not that they are good yet.

**Only after (a) < ~90 GiB AND (b) reproduces all three scored quantities → operator GO → drop `--dry-run`
and launch `--config store_nothing_205` for the full 1000-epoch burn.**

---

## Verdict
**PROCEED** — `tools/launch_witness_run.py --config store_nothing_205` is launch-ready: 85/85 flags
validated, projected peak 67.61 GiB SAFE, system-admission ADMIT, store-nothing pose wired + routed +
byte-close-measured (~1 KB pose section, 0 pose rate), seg⊥pose structural, resumable + per-stage
checkpointed + EMA-shadow + seeded, annealed-hosc (not divergent fixed β=4), l7-demoted, verdict-batch-32
(the OOM fix). **Two gates before the burn:** (1) the operator DECISION on the 3 optional deferred levers
(recommendation: SEALED-as-is first, levers as warm-start), and (2) the axis-9 smoke (§5) — the measured
pre-burn gate the operator runs. Pointer 0.19110 UNMOVED until a byte-closed `evaluate.py` n600 row lands.
