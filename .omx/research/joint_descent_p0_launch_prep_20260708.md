# Joint pose-descent P0 — launch-ready prep (#366) — 2026-07-08

**Task:** make the R1-proven joint pose-descent recipe ONE-GO-from-launch on a **memory-SAFE** target,
in parallel with the live d_seg run (pid 63069), and MEASURE (not guess) the memory feasibility.
Read-only, $0, no heavy/paid launch. **Pointer 0.19110 UNMOVED.** All numbers `[macOS-CPU advisory] /
[macOS-MLX research-signal] NON-PROMOTABLE`.

**One-line verdict:** the R1 recipe is **MLX-Metal-only** (`import mlx.core` throughout, render on
`mx.gpu`) — so **neither bat00 (CUDA) nor Modal (CUDA) can run it** without a full torch/CUDA re-port +
re-validation (weeks, not launch-prep). The only hardware that runs it is Apple Silicon, and this M5 Max
is occupied by run-1 holding **~65 GiB** MLX unified memory. **There is NO genuinely-parallel MLX path**;
fp16-concurrent still REFUSES. The fastest memory-SAFE path is **(D) SEQUENCE** on this box — the exact
R1 launch.sh, warm-started from the converged mod-26 `v2_attrclean` witness, standalone preflight
**67.6 GiB SAFE** — launched when run-1 yields the box (governed stop / operator pause).

---

## 1. The R1-proven joint-descent config

**Source recipe (primary artifact):** `experiments/results/levelset_n600_R1_storenothing_descent_ev1_20260703T004906Z/launch.sh`
— the run that took d_pose **97 → 0.0011** in ~108 warm-started epochs holding d_seg ~0.0046
(custody re-validated 2026-07-08, `r1_0011_custody_revalidation_20260708.md`: VALID frozen-CPU-torch,
contest-definition, n600, through-R, EMA-conservative; NOT yet byte-closed — #238 owed).

**Warm-start source = the CONVERGED d_seg witness** `experiments/results/levelset_n600_v2_attrclean_20260630T194549Z/`
(mod-dim **26**; R1 resumed at ep1001 INSIDE the Muon finisher). Present + loadable: `levelset_witness_ema_mlx.npz`
+ `levelset_resume_state.npz` + per-stage ckpts (CE ep299 / Tau ep599 / MuonStart ep726 / L7 ep1000). This
is the PROVEN pose-descent substrate — do NOT substitute run-1's mod-32 checkpoint without re-validating the
carrier (see §5, option D-alt).

**Exact governed-launcher argv** (the joint-descent stage; all flags verified real against the trainer
argparse — never-invent-flags):

```
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 .venv/bin/python \
  experiments/train_levelset_witness_realized_through_R_mlx.py \
  --out-dir <NEW_RUN_DIR> \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 \
  --resume-from experiments/results/levelset_n600_v2_attrclean_20260630T194549Z \
  --resume-allow-lever-drift \
  --mlx-device gpu --seed 0 --async-verdict \
  --epochs 1400 --eval-every 1 --verdict-pairs 0 --verdict-batch 32 \
  --curriculum --tau-softplus-start-epoch 300 --tau-softplus-tau 0.3 \
  --l7-start-epoch 600 --muon-start-epoch 726 --muon-lr 0.002 --muon-momentum 0.95 --muon-ns-steps 5 \
  --stage-transition-rewarmup-epochs 8 --stage-transition-rewarmup-floor 0.1 \
  --stage-transition-rewarmup-shape linear --stage-transition-reset-moments \
  --w-seg 100 --w-pose 1.0 --score-domain-loss \
  --pose-carrier --pose-carrier-source generated --pose-carrier-residual-mode table \
  --mod-dim 26 --hidden-dim 96 --n-hidden 4 \
  --activation hosc --hosc-beta 4.0 --hosc-omega 1.0 --siren-init \
  --softmax-temp-start 1.0 --softmax-temp-end 0.05 \
  --self-orient --n-dir-freqs 2 --freq-across 32 --freq-along 4 --reorient-every 50 --max-bank-freq 64 \
  --chroma --palette-anchor --eikonal-weight 0.01 --length-weight 0.001 \
  --render-h 384 --render-w 512 --accum-pairs 8 --grad-clip 1.0 --ema-decay 0.997 \
  --structured-init --structured-init-include-lane \
  --lane-prior-phi1 --lane-prior-phi1-mode replace --lane-prior-phi1-dash-gate \
  --ckpt-every 1 --stage-checkpoints
```

**Resumability (MANDATORY — satisfied):** `--ckpt-every 1 --stage-checkpoints` + resumable
`levelset_resume_state.npz` per stage. EMA-shadow saved (`--ema-decay 0.997`). Launch via the durable
daemon under `safe_run.py --rss-mb 90000 --timeout 28800` (R1's wrapper).

**DSL leg (the pose-carrier orphan — CLOSED this task).** The DSL held only carrier **B**
(`WarpRealLumaFrame0`, stored-keyframe). R1's carrier **A** (store-nothing `--pose-carrier-source
generated`) was NOT a DSL Lever — the P0 recipe was DSL-orphaned. Added
`StoreNothingPoseCarrier(w_pose, residual_mode)` → `curriculum_dsl.py` (emits `--pose-carrier`,
`--pose-carrier-source generated`, `--pose-carrier-residual-mode table`, `--w-pose`; residual_mode
fail-closed to table|film) + 5 pure-argv tests. The rest of the argv is standard existing levers
(Muon finisher, self-orient directional bank, chroma/palette, eikonal/length, structured-init +
lane-prior). No new DSL gap remains for R1's recipe.

**Short or long?** SHORT in EPOCHS (~75–108 warm-started to the d_pose plateau) but **LONG in
wall-clock**: R1 ran `--eval-every 1 --verdict-pairs 0` (a FULL n600 verdict EVERY epoch, ~48 min/epoch
GPU-bound) → ~2–3 days to the plateau. The verdict is ADVISORY (not a training signal), so a LAUNCH can
set `--eval-every 5` (or 10) to cut wall-clock ~5–10× with zero effect on the trained descent — recommended
for the actual launch (R1's eval-every-1 was a per-epoch diagnostic choice, not a training requirement).

---

## 2. Memory preflight — MEASURED

`tools/witness_memory_preflight.py` on R1's launch.sh; ceiling = 0.70×128 = **89.6 GiB**. Live run-1's
true footprint is **`mlx_peak_gib` = 65.2 GiB** (verdict rows ep225–275; the `ps` RSS ~8 GiB is
misleading — MLX unified memory lives in the Metal pool, not RSS).

| scenario | projected peak | vs 89.6 ceiling | verdict |
|---|---|---|---|
| **R1 descent STANDALONE (fp32)** | **67.6 GiB** (15 fixed + 43.2 cf_mx_cache + 3.4 gt + 6 verdict) | −22 GiB | **SAFE** |
| R1 descent CONCURRENT with run-1 (fp32) | 65.2 + 67.6 = **132.8 GiB** unified | > 128 **total** | **CERTAIN OOM** |
| R1 descent CONCURRENT, fp16 cf-feats (#296) | 65.2 + ~46.0 = **111.2 GiB** unified | +21.6 over ceiling | **REFUSE** (jetsam risk; #296 rigor gate unmet) |
| `witness_memory_preflight --system-aware` (run-1 live) | — | — | **REFUSE** (system-used projection ≫ 64 GiB adaptive ceiling) |

**Confirmed:** fp32 concurrent REFUSES exactly as expected; the `--system-aware` governor also REFUSES
with the live run present. Even **fp16-concurrent REFUSES** (111 GiB unified = 87% of RAM leaves no
control-plane headroom; and #296's fp16 gate — measured-d_seg-impact + review — is unmet, so fp16 could
perturb the trained field).

---

## 3. Target feasibility table (go/no-go)

| target | genuinely parallel? | go/no-go | why / what it needs |
|---|---|---|---|
| **(A) bat00 RTX 3090 (CUDA/WSL2)** | yes (free) | **NO-GO for R1's recipe** | The trainer is **MLX-Metal-only** (`import mlx.core` throughout; render on `mx.gpu`). MLX does not run on CUDA/Windows. `src/tac/torch_vehicle/` is a SEPARATE CUDA vehicle (HNeRV-basin lineage) — NOT the store-nothing level-set witness, and MLX mod-26 weights don't transfer to torch. bat00 IS online (tailscale IP in gitignored fleet.local.toml, windows) but running R1 there needs a full torch/CUDA re-port + re-validation (weeks), not launch-prep. `scripts/bat00.py` also needs `BAT00_IP`/`BAT00_USER` operator env creds. |
| **(B) Modal T4/A10G (paid <$5)** | yes | **NO-GO for R1's recipe** | Same MLX≠CUDA blocker (Modal is x86+CUDA). No Modal wrapper for the MLX levelset trainer exists (`experiments/modal_*` are torch/CUDA jobs). Would need the same CUDA port + a new dispatch recipe + operator budget-GO + the P0-campaign clause. |
| **(C) fp16 cf-feats local, concurrent (#296)** | yes | **NO-GO (REFUSE)** | Projected concurrent **111.2 GiB** unified > 89.6 ceiling (§2). Plus #296's rigor gate (measured-d_seg impact + review) is UNMET → fp16 could perturb the field's d_seg. Not free to flip on. |
| **(D) SEQUENCE on this M5 Max** | no (serial) | **GO — memory-SAFE** | Standalone preflight **67.6 GiB SAFE**. MLX-native, zero new infra, byte-for-byte R1's launch.sh, warm-start from the converged mod-26 `v2_attrclean`. Needs: run-1 to yield the box (governed stop / operator pause) + operator-GO to launch. |
| (D-alt) SEQUENCE warm-started from **run-1** | no | GO **later, with a caveat** | run-1 is at ep275, d_seg **0.1198** — NOT converged (needs the Muon finisher ~ep726+ → ~0.0046). Days away, AND its checkpoint is **mod-32** (R1 was mod-26) → the carrier must be re-validated on mod-32 before trusting the descent. Prefer the proven mod-26 `v2_attrclean` source (D). |

---

## 4. RECOMMENDATION (ranked)

1. **(D) SEQUENCE the R1 recipe on this M5 Max, warm-started from `v2_attrclean` (mod-26), when run-1
   yields the box.** The ONLY memory-SAFE, MLX-native, launch-ready path (standalone 67.6 GiB SAFE,
   the config is R1-proven and now DSL-held). **Operator decision:** GO to pause/governed-stop run-1
   (preserving its stage ckpts) and launch — OR wait for run-1's own governed stop, then launch.
   Tuning for the launch: set `--eval-every 5` (advisory verdict; cuts wall-clock ~5× with zero effect
   on the trained descent). This lands the joint-descent arm on the proven substrate fastest.

2. **(A/B) CUDA port to bat00 or Modal — genuinely parallel but NOT near-term.** Requires a full
   torch/CUDA re-implementation of the level-set witness + store-nothing pose carrier + re-validation
   from scratch (weeks; MLX weights don't transfer). Worth queuing as a durable infra bet ONLY if
   sustained genuine parallelism is wanted; it is NOT a launch-prep deliverable. **Operator decision:**
   whether to fund/scope a CUDA port at all.

3. **(C) fp16 concurrent local — NO-GO.** REFUSES on memory (111 GiB unified) even before #296's unmet
   fp16 rigor gate. Do not pursue as a parallel path.

**Bottom line for the operator:** "parallel joint pose-descent" is blocked by a hardware fact — the
witness is MLX-only and this is the only MLX box, already ~65 GiB full. The realistic P0 action is the
SEQUENCE (D): one operator GO to yield run-1's box, then a byte-for-byte R1 launch (67.6 GiB SAFE) from
the converged mod-26 witness. Byte-close (#238) remains owed regardless of where the descent runs.

**Pointer 0.19110 UNMOVED.**
