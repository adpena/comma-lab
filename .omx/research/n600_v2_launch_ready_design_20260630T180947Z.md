# n600 v2 witness — LAUNCH-READY design + config (DERIVED, held-out-corrected)

**UTC** 20260630T180947Z · **tag** `[macOS-MLX advisory · design artifact · NON-PROMOTABLE]` · **pointer 0.19110 UNMOVED.**
**DO NOT LAUNCH** — this is the design artifact for a recursive adversarial review → operator go → GPU launch.
$0 CPU-only assembly, NO GPU touched, READ-ONLY on caches/ckpts, not committed.
**means≠ends:** every value below is a MEANS; the only END is a **byte-closed n600 exact row < 0.19110** (CPU/CUDA, never MPS).

## Dogfood principle (auto-config: derive, don't hardcode)
Every value is DERIVED from a generator (a measurement or a recalled proven config), not invented. The
strongest grounding: I **recalled the LIVE n200 Muon-arm config** from its preserved ckpt
(`levelset_thetastar_muon_arm/levelset_witness_ema_BEST.npz` `__cfg_*` + `levelset_train_result.json`),
which reached the **best measured realized d_seg 0.003698 @ ep1000**. That proven config supplies the
schedule + base arch; the **held-out n400 generalization probe** supplies the two arch corrections
(mod-dim, hidden); the surgical levers carry their FEED-eq/birth-death/screw-warp provenance. Every flag
emitted below was confirmed against the real argparse in
`experiments/train_levelset_witness_realized_through_R_mlx.py` (116 flags grepped; **no flag invented**).

> **NO-FAKE flag-validation catch:** the source recipe `post_muon_application_plan_optimal_form_*.md` §3
> wrote `--margin-saliency-target lane`. That is a **bug** — `--margin-saliency-target` is a **float**
> (the hinge decision margin), and LEVER-4 is **class-AGNOSTIC** (no "lane" target exists). It would
> crash argparse. This design does NOT propagate it: lane focus comes from `--lane-thin-*`/`--lane-edge-*`
> (which DO take a class index), not from margin-saliency.

---

## 1. The flag-validated trainer command (from-scratch n600)

Tier labels: **[P]** proven-lineage recall · **[H]** held-out-corrected · **[θ*]** θ*-pending weight (documented start) · **[F]** byte-free optimal-form.

```bash
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 \
.venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py \
  --out-dir experiments/results/levelset_n600_v2_20260630T180947Z \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 \
  --mlx-device gpu --seed 0 --async-verdict \
  --epochs 1000 --eval-every 25 \
  --curriculum --tau-softplus-start-epoch 300 --l7-start-epoch 600 \
  --muon-start-epoch 726 --muon-momentum 0.95 --muon-ns-steps 5 \
  --stage-transition-rewarmup-epochs 8 --stage-transition-rewarmup-floor 0.1 \
  --stage-transition-rewarmup-shape linear --stage-transition-reset-moments \
  --w-seg 100 --w-pose 0 --score-domain-loss \
  --mod-dim 26 --hidden-dim 120 --n-hidden 4 \
  --activation hosc --hosc-beta 4.0 --hosc-omega 1.0 --siren-init \
  --softmax-temp-start 1.0 --softmax-temp-end 0.05 \
  --self-orient --n-dir-freqs 2 --freq-across 32 --freq-along 4 --reorient-every 50 \
  --max-bank-freq 64 \
  --chroma --palette-anchor \
  --eikonal-weight 0.01 --length-weight 0.001 \
  --render-h 384 --render-w 512 \
  --structured-init --structured-init-include-lane \
  --lane-prior-phi1 --lane-prior-phi1-mode replace --lane-prior-phi1-dash-gate \
  --film-stiefel --code-spectral-entropy-weight 0.01 --dm1-telemetry \
  --margin-saliency-weight 1.0 --margin-saliency-tau 0.5 --margin-saliency-target 0.5 \
  --margin-saliency-start-epoch 300 --margin-saliency-uniward --margin-saliency-uniward-beta 4.0 \
  --lane-thin-weight 1.0 --lane-thin-radius 2 --lane-thin-target 0.5 --lane-thin-class 1 \
  --lane-thin-start-epoch 300 \
  --hardness-oversample 0.5 --hardness-weighted --hardness-source realized \
  --hardness-power 1.0 --hardness-band 0.5 \
  --ckpt-every 25
```

`--muon-lr` is **intentionally omitted** → trainer default `0.1·lr = 1e-4` (the value the proven n200 arm
most likely used; the result JSON did not expose it). The help documents `1e-3–5e-3` as the optimal-form
range — see Open Q6. `--lr 1e-3 / --lr-end 1e-4 / --weight-decay 1e-4 / --ema-decay 0.997 / --accum-pairs 8
/ --grad-clip 1.0 / --hinge-weight 4.0 / --spike-factor 5.0 / --warmup-epochs 1 / --lr-schedule /
--stage-checkpoints / --structured-init-include-lane / --lane-prior-phi1-dash-gate / --siren-init` are all
at their (proven-matching) defaults and shown only where load-bearing.

### Attribution-clean variant (the recommended FIRST launch — see Open Q3)
Drop the **[θ*]** block (the 4 surgical-lever groups: `--margin-saliency-*`, `--lane-thin-*`,
`--hardness-*`) AND `--film-stiefel --code-spectral-entropy-weight`. That leaves the **proven schedule +
base arch + the 2 held-out arch corrections** — a from-scratch n600 whose only deltas vs the measured
0.003698 lineage are `mod 32→26` and `hidden 96→120`. Cleanest attribution; the surgical levers + DM1
then land as a warm-start re-treatment (step 2/4) once #183 fills the θ* weights.

---

## 2. Per-value PROVENANCE table (the auto-config dogfood, made explicit)

| Flag / value | Tier | DERIVED FROM (the generator) |
|---|---|---|
| `--gt-cache gt_n600.npz --num-pairs 600` | [P]/[H] | full clip (600 pairs); held-out probe data-provenance table (verified disjoint subsets map to n600 idx) |
| `--mod-dim 26` | **[H]** | **n600 manifold eff-dim = 26.33** (held-out probe Tool-1 line 61: n96 5.96 → contig96 14.65 → heldout336 25.26 → **n600 26.33**). The corrected value — NOT the n200/n96-era 21. Capacity must ≥ the manifold dim. |
| `--hidden-dim 120` | **[H]** | **83%-off-pose-residual correction** (held-out line 37/116: irreducible learned residual is the DOMINANT manifold share, anchor said 49%) ⇒ generous trunk via the KKT waterfill. Proven lineage = **96** (= config-review #2 RD-optimum ~122KB). **RD-refined post-train** (Open Q1). |
| `--n-hidden 4` | [P] | proven `__cfg_n_hidden=4` |
| `--epochs 1000` | [P] | proven schedule TOTAL (recalled: `epochs=1000, anneal_epochs=1000`; CE300/Tau300/L7126/Muon274) |
| `--curriculum --tau-softplus-start-epoch 300 --l7-start-epoch 600` | [P] | proven `__cfg_tau_softplus_start_epoch=300, __cfg_l7_start_epoch=600` + curriculum ledger (generous CE+Tau bulk drop, short L7 knee) |
| `--muon-start-epoch 726` | [P] | proven `stage_checkpoints[0].epoch=726` (`stageMuonStart_ep726`); ≥ l7 (PR95 placement, no WARN); the measured d_seg finisher |
| `--muon-momentum 0.95 --muon-ns-steps 5` | [P] | PR95 stage-8 defaults (Keller-Jordan NS=5) |
| `--stage-transition-rewarmup-epochs 8 --…-floor 0.1 --…-shape linear --stage-transition-reset-moments` | [P] | proven `stage_transition_rewarmup_epochs=8, floor=0.1, linear, reset_moments=True` + "different stages need different treatment" (FEED-ft#3 tau-jump = stale momentum) |
| `--self-orient --n-dir-freqs 2 --freq-across 32 --freq-along 4 --reorient-every 50` | [P] | proven `__cfg` (self_orient=1, n_dir_freqs=2, freq_across=32, freq_along=4, reorient_every=50) + **directional basis −48% exponent (frontier lever #1)** |
| `--max-bank-freq 64` | [P] | proven `__cfg_max_bank_freq=64.0` + LEVER-2 stem-Nyquist `SEG_W/(4·stem_stride)=512/8=64` |
| `--activation hosc --hosc-beta 4.0 --hosc-omega 1.0` | [P] | proven `__cfg` (hosc, beta=4, omega=1) + config-review #3 (HOSC is the ONLY descent evidence: 0.221 hosc vs 0.265 wire) |
| `--softmax-temp-start 1.0 --softmax-temp-end 0.05` | [P] | trainer default; proven snapshot temp 0.2157 lies on this cosine; config-review #4 (anneal hi→lo, not fixed 0.1 = RGB Gibbs) |
| `--chroma` | [P] | proven `__cfg_chroma=1` + operator "Chroma too" (SegNet reads RGB ⇒ chroma is a d_seg actuator) |
| `--palette-anchor` | [P] | DIAGNOSED FIX (init palette to per-class mean GT RGB; breaks the ~0.51 luma-ramp plateau) |
| `--render-h 384 --render-w 512` | [P] | config-review #1: render-384 is the MEASURED R-survival floor (render-192 pre-caps at 0.00085 d_seg = +0.085 S) |
| `--eikonal-weight 0.01 --length-weight 0.001` | [P] | level-set variational regularizers (eikonal \|∇φ\|→1 topology bias; Chan-Vese boundary length) — defaults |
| `--w-seg 100 --w-pose 0 --score-domain-loss` | [P] | fix-g: pose SOLVED by the Quantizr stored sidecar (3.4e-5); witness's only binding job is d_seg |
| `--structured-init --structured-init-include-lane` | [P] | FEED-ef static-core partition from the **n600 cached L*** (auto-regen, majority-vote); rule-118 FREE train-time init, ships 0 bytes. **CAVEAT: no epoch-0 realized win, trajectory A/B only, hosc/SIREN-init-FRAGILE** (Open Q5) |
| `--lane-prior-phi1 --lane-prior-phi1-mode replace --lane-prior-phi1-dash-gate` | [P] | proven `lane_prior_phi1=True, mode=replace`; openpilot deg-3 centerline = the Road↔Lane separatrix (FEED-fs residual 1.9e-5), auto-fit from n600 GT pair-0; FREE generic geometry |
| `--film-stiefel --code-spectral-entropy-weight 0.01 --dm1-telemetry` | [F]/[θ*] | DM1 byte-free rank cure: PR(M) 1.19→4.57 so the mod-26 capacity is actually USED (else rank(codes)≤PR(M) caps it). **UNPROVEN in lineage** (proven arm had it OFF) — see Open Q2. weight 0.01 is θ*-pending. `--dm1-telemetry` forces the PR(M) row for the baseline. |
| `--margin-saliency-weight 1.0 --…-tau 0.5 --…-target 0.5 --…-start-epoch 300 --…-uniward --…-uniward-beta 4.0` | [θ*] | LEVER-4 (FEED-eq): all-class flip band Road47/Lane19/Undriv14/Movable9/MyCar11% ⇒ class-agnostic saliency defends 100% (vs LEVER-3's 19%). tau 0.5 ≈ p1-p5 of GT-margin; start 300 = tau margin stage; UNIWARD = Fridrich texture down-weight. **weight 1.0 = documented start, θ*-pending (#183)** |
| `--lane-thin-weight 1.0 --lane-thin-radius 2 --lane-thin-target 0.5 --lane-thin-class 1 --lane-thin-start-epoch 300` | [θ*] | LEVER-B (birth-death): thin dashes <3px flip ~92%; Lane PH⁰-dim 0.84 (highest, multi-scale); R-survival 85% (R-recoverable, not R-destroyed). radius 2 = the ~2px lane tube. **weight 1.0 θ*-pending; proven arm had it OFF** |
| `--hardness-oversample 0.5 --hardness-weighted --hardness-source realized --hardness-power 1.0 --hardness-band 0.5` | [θ*] | LEVER-5: waterfill per-pair code-fit budget to hard pairs; `realized` source is the SHARPER signal (margin spread only 1.31×). **oversample 0.5 θ*-pending** |
| `--ckpt-every 25 --stage-checkpoints` (default) `--async-verdict` | [P] | proven `ckpt_every=25`; resumability discipline; async-verdict = ~4.7% wall-clock reclaim, bit-identical training |
| `--seed 0 --mlx-device gpu` | [P] | deterministic reproducibility |

---

## 3. Pre-launch checklist (the binding disciplines)

- [ ] **Resumable** — `--resume-from <dir>` supported (restores decoder + per-pair codes + EMA shadow + optimizer + epoch). Verified.
- [ ] **Per-stage ckpts** — `--stage-checkpoints` default ON → PRESERVED stage-encoded ckpts at every curriculum boundary (CE→Tau→L7→MuonStart) + final, atomic (tmp+rename). `--ckpt-every 25` adds intra-stage rolling (bounds a crash to ≤25 ep).
- [ ] **EMA-shadow preserved (not live)** — saves `levelset_witness_ema_*.npz` (the shadow); best-preservation `_maybe_preserve_best` → `levelset_witness_ema_BEST.npz` + `levelset_best.json` on every strictly-better finite verdict, atomic (hardened in 3da9a6b10).
- [ ] **Perf env** — `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` (now the **default**: unset→"1"). **MUST verify** the one-time launch log line from `_log_custom_backward_decision_once` shows `active=True / reason="env_set_and_metal_backend_available"`. If it logs `reference`/`unavailable`/`disabled` → ~17× slow → **ABORT** (the launch-gate-throughput discipline; a SEAL'd run can still be 17× slow).
- [ ] **Whole-run archiver (sidecar daemon)** — `tools/archive_witness_checkpoints.py --run-dir experiments/results/levelset_n600_v2_20260630T180947Z --min-free-gb 10 --keep-window 0` (poll, spill/preserve every best+latest to SSD cold-store; certify-or-block).
- [ ] **Memory guard** — `tools/memory_guard.py --watch --min-free-gb 10` (allowlist the n600 training arm; **NEVER kill control plane** claude/codex; 10GB floor on the 128GB box; sheds the largest training arm only).
- [ ] **Contained out-dir** — `experiments/results/levelset_n600_v2_<utc>/` (durable, NOT /tmp).
- [ ] **One GPU** — the n200 Muon arm must be DONE/sequential (n600 owns the GPU). **No autonomous heavy launch — await operator go** (containment/protect/preservation).
- [ ] **Fail-closed config validators fire pre-GPU** — the trainer's own guards (curriculum order `0<300<600≤1000` ✓; muon placement `726∈[1,1000]` & `≥600` ✓; lane/saliency/muon silent-no-op traps) raise LOUD before any spend. Confirmed the current command passes all.
- [ ] **Determinism** — `--seed 0`; MLX/numpy/random seeded one path; numpy-fp32 EMA-shadow verdict authority (never MPS).

---

## 4. Open questions for the recursive review

1. **Exact `--hidden-dim`** — 96 (proven, RD-optimum ~122KB) vs **120** (generous, 83%-residual) vs sweep. The **archive-byte cost of mod-26/hidden-120 is UNMEASURED** (config-review #2: hidden-128 overshoots to 161KB = +0.026 S). This is the "map the RD curve" point — needs a byte-close. RATE is the binding sub-0.15 lever, so an overshoot here directly costs S.
2. **The mod-dim ↔ PR(M) ↔ manifold-eff-dim GAP (central capacity question).** The manifold needs **~26** eff-dims, but measured FiLM PR(M) = **1.19 uncured / 4.57 DM1-cured** — both ≪ 26. So per-pair FiLM modulation alone **cannot span the manifold**; the trunk (hidden) must carry the bulk (consistent with the 83%-residual correction → generous hidden). Is DM1 (unproven in lineage) worth turning on, or is the real fix all-in-the-trunk? Should mod-dim even be 26 if PR(M) caps at ~5?
3. **Surgical-lever θ* weights + stack-vs-attribution.** #183 per-lever A/B is NOT complete. Launch the **attribution-clean variant first** (proven base + 2 arch corrections only), then warm-start re-treat with the surgical levers once θ* is filled? Or full-stack now and accept un-attributable interactions? (Recommendation: attribution-clean first.)
4. **`--self-orient` from-scratch** — help says "finetune lever; needs a roughly-learned base." It changes `in_feat` so it CANNOT be added mid-run (must decide at launch). Early reorients (ep<300) orient against a forming partition — weak but likely not harmful; `--structured-init` may give it a base earlier. Confirm it doesn't hurt the CE stage.
5. **`--structured-init` MEASURED CAVEAT** — no epoch-0 realized win (texture-gated init), hosc/SIREN-init-FRAGILE (loud WARN if the pretrain stalls), trajectory A/B only / UNPROVEN. Worth the fragility from-scratch, or random/SIREN init?
6. **`--muon-lr`** — omitted → default `0.1·lr = 1e-4`. The help documents `1e-3–5e-3` as the optimal-form range. **Recall/confirm the live n200 arm's actual muon-lr** (not exposed in its result JSON) before reproducing the 0.003698 descent; tuning up is an optimal-form lever.
7. **Epoch budget** — 1000 (proven). The n200 Muon was **still descending at the end (critical slowing near an RD topological transition)** → consider extending Muon (epochs 1000→1200). The **root-tracking anneal scheduler** is the wall-clock-optimal cure (same d_seg, fewer epochs) — **NOT built** (follow-on, not a blocker).
8. **Screw-warp v2 vehicle wire-in status** — **NOT a trainer flag** (open build). Held-out: bulk-warp-through-R residual 0.0048 (~4× the 1.23e-3 budget) so "bulk needs no INR" is upper-bound-falsified for previous-frame-warp; lane is warp-UNEXPLAINABLE (0.39 flip) = the learned residual. So this INR witness IS the lane+movables learned residual; the screw-warp handles bulk deterministically as a SEPARATE v2 build (clean stored-canonical warp is the next $0 measurement). The n600 INR launch does not block on it.

---

## 5. Honest flags — measured vs estimated (means≠ends)

- **MEASURED (real provenance):** the entire schedule + base arch (recalled from the LIVE n200 Muon arm, best realized d_seg **0.003698** @ ep1000) + directional basis (−48%) + render-384 (R-survival floor) + curriculum boundaries + chroma + palette-anchor + max-bank-freq + stage-transition treatment + lane-prior-phi1. The data caches and disjoint splits are verified.
- **HELD-OUT-CORRECTED (measured on data 0%-overlapping the anchor):** `mod-dim 26` (n600 manifold eff-dim 26.33) and `hidden generous` (off-pose residual 83%). These are the two values the generalization probe changed vs the opening-segment anchor; the probe's scorer-geometry family HOLDS on held-out, but the pose↔manifold regression SHIFTS (window-conditioned) — so do NOT over-credit "pose is a near-free dual-use d_seg lever"; the learned residual is the dominant manifold share.
- **ESTIMATED / θ*-PENDING:** surgical-lever weights (margin-saliency / lane-thin / hardness), `--code-spectral-entropy-weight`, `--muon-lr` tuning, the exact `--hidden-dim` (RD-refined post-train), the DM1 benefit (unproven in lineage), the structured-init trajectory benefit (unproven, fragile). These carry documented starting values so the command is launch-ready, but they are NOT measured wins.
- **NOT a score:** `[macOS-MLX advisory · NON-PROMOTABLE]`. realized/deploy d_seg is the surrogate; the **pointer 0.19110 is UNMOVED** and stays UNMOVED until a **byte-closed n600 archive** is scored by `upstream/evaluate.py` (contest-CPU AND/OR CUDA, never MPS). A descending realized d_seg is a MEANS; the lower exact row is the only END.
- **Follow-on builds (noted, NOT launch blockers):** root-tracking anneal scheduler · auto-config-as-one-tool · K-planes spatial factor · the screw-warp v2 deterministic-bulk vehicle.
