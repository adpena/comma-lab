# Tertiary edge-tier memory-measurement sweep — SPEC (prep #297; sweep NOT launched)

**Date:** 2026-07-04 · **Task:** #297 prep (operator-activated) · **Axis:** `[macOS-M1-8GB measurement]`,
production_generalized METHOD — no score claim, pointer 0.19110 UNMOVED. · **Status:** PREP + $0 smoke
COMPLETE; sweep fires on the usual staggered GO after the primary fresh run's ep0 gate. This memo is the
launch card.

Methodology replicated from the #205 mine (`.omx/research/n205_memory_behavior_mine_20260704.md`):
blackbox 2s JSONL + per-config {steady band, peak, spike signature, leak slope} rows + projection
residual. Purpose (per #295/#294): grow the config→envelope ROW FAMILY across RAM tiers — the 8 GB
edge tier is the second point after the 128 GB primary row (the family IS the production sizing curve).

## 1. Machine state (probed 2026-07-04 over Tailscale; tertiary = M1 MacBook Pro)

- macOS 26.5.1 (25F80), Apple M1, **8.00 GiB RAM**, disk 228 Gi (47 Gi free after data ship).
- Idle profile (governor `--snapshot`, MEASURED): used **6.06 GiB** (wired 1.25 + compressor 2.41),
  available **1.94 GiB**, swap already 1.70 GiB, `pressure_level=1` (blackbox string: "critical").
  The 8 GB box idles HEAVY — the OS working set is compressible but real.
- Repo `~/Projects/pact` @ **55526e9be** = primary HEAD at sync time (parity gate PASSED; fetched via
  the new `primary` remote `ssh://adpena@100.81.85.28/Users/adpena/Projects/pact` — origin lacks ~280
  local commits and we do NOT push to origin; tertiary syncs FROM primary over Tailscale).
- Env: uv 0.11.26, venv Python 3.13 matching primary — torch 2.12.1 / mlx 0.31.2 / numpy 1.26.4 /
  scipy 1.17.1 / timm 1.0.27 / smp 0.5.0 / einops / safetensors / psutil; `.venv` = 859 MB.
- Data (sha256 VERIFIED identical both ends): `gt_n24.npz` (2c0fe204…, 193.7 MB), `gt_n96.npz`
  (6aad6600…, 774.8 MB), `upstream/models/segnet.safetensors` (68956e32…), `posenet.safetensors`
  (0f3a0874…). Also pre-existing `gt_heldout_n400.npz`.
- Telemetry: `memory_blackbox.py --daemon --no-govern` deployed via the canonical
  `tools/spawn_durable_daemon.py` (label `memory_blackbox`, registered + liveness-verified; JSONL at
  `.omx/state/memory_blackbox.jsonl`, 2 s cadence, 1 s fast-interval under pressure — observed live).

## 2. The 8 GB envelope (MEASURED, not assumed)

**Never-crash physics scaled** (per `operator_memory_policy_sole_workload_no_artificial_ceiling` legs):
control-plane+OS floor on THIS box ≈ 2.5–3.0 GiB residual under pressure (idle used is 6.06 but
compressor/cache yields ~3 GiB back under load — the smoke ran at 3.6 GiB RSS without swap-thrash);
spike margin ~0.5–1 GiB (verdict step at vb≤8 measured ≈ 1.3 GiB, §3); ⟹ **usable envelope ≈ 4.0–4.4
GiB ⟹ safe-frac ≈ 0.55** at this tier (vs 0.85 on the sole-workload 128 GB box — the fraction is
tier-dependent because the OS floor does not scale with RAM).

**Governor numbers on tertiary (run 2026-07-04, post-#294 units-fixed code):**

| Surface | Default (128 GB-calibrated) | 8 GB-scaled (what the sweep uses) |
|---|---|---|
| `compute_safety_margin_gib` | max(**8.0 floor**, 0.08×8=0.64) = **8.0 GiB** | — (floor not CLI-overridable; see gap below) |
| `--ceiling` adaptive ceiling | **0.0 GiB**, training_budget **−5.99 GiB** (REFUSES everything) | n/a at defaults |
| `--band-tick` envelope | 0.85 × 8 = **6.8 GiB** (unsafe: > box minus OS floor) | `--band-envelope-frac 0.55` → **4.4 GiB**; yellow @85% = **3.74**; red @90% = **3.96** (verified live: envelope_gib 4.4, band green at idle) |
| `witness_memory_preflight --safe-frac` | default 0.70 → 5.6 GiB (too high here) | `--safe-frac 0.55` → 4.4 GiB ceiling |
| `safe_run --rss-mb` | 2048 default | **4000 MiB** cap (proven in smoke) |

**⚠ SCALING GAP (flag for #294 follow-up, NOT edited here):** `DEFAULT_SAFETY_MARGIN_FLOOR_GIB = 8.0`
equals the ENTIRE box at this tier → `compute_adaptive_ceiling` = 0 and the admission gate refuses all
jobs; the pure function accepts `safety_margin_gib` but the CLI does not expose it. Consequence for the
sweep: the operative guards on tertiary are `safe_run --rss-mb 4000` (hard, 0.2 s poll) + band-tick at
`--band-envelope-frac 0.55` (cron/loop) + the blackbox recorder; the governor admission/throttle path
stays OFF (`--no-govern`) until the floor scales (proposed: `min(8.0, max(1.0, 0.25×total))` or a
`--safety-margin-gib` CLI flag — decision belongs to #294, not this prep).

## 3. $0 smoke row (the first tertiary calibration point; run twice, repeatable)

Config: n6 pairs (gt_n24 cache), mod-dim 8, hidden 32×2, bank 2×4(+2 iso), accum-pairs 2,
verdict-batch 6, MLX-**GPU** (M1 GPU worked; NO cpu fallback needed ⟹ #265
`TAC_MLX_CUSTOM_GROUPED_BACKWARD=0` guard not triggered), `TAC_MEM_PROBE=1`, safe_run cap 4000 MiB.

- **peak_rss 3639 / 3694 MiB** (two runs; safe_run 0.2 s-poll authority) · **wall 50.9 s** both runs
  (≈25 s startup: torch scorer load + cache; epochs+verdicts the rest) · exit 0 · no swap-thrash.
- mem_probe: before_v0_verdict rss 0.93 GiB → after_v0_verdict **2.19 GiB** ⟹ **verdict step +1.26 GiB
  at vb=6/n6** — the same verdict-(inference-)driven-peak signature as the #205 row, reproduced at
  edge scale. `mlx_active` 0.13 GiB, epoch-scoped `mlx_peak` 1.95 GiB.
- Envelope check: 3.69/4.4 = 84% of envelope = top of GREEN. **An n6 smoke already nearly fills the
  8 GB tier's envelope** — the frozen torch scorer pair (~0.9 GiB) + MLX runtime is the fixed floor;
  headroom for model/cache/verdict is ~2–3 GiB. Per-stage checkpoints landed
  (`levelset_ckpt_stageCE_ep3.npz` + resume state + EMA BEST) — resumability discipline holds at this tier.

## 4. Sweep grid (n24 base; measurement sweep — memory envelopes, not score)

All arms: `--gt-cache …/gt_n24.npz --num-pairs 24 --seed 0 --mlx-device gpu`, `TAC_MEM_PROBE=1`,
safe_run `--rss-mb 4000 --timeout 14400`, blackbox running, band-tick loop
(`--band-tick --band-envelope-frac 0.55` every 60 s). Grid = the #297 brief's axes:

| arm | mod-dim | hidden | self-orient (cf cache) | verdict-batch | epochs | expected peak (prior) |
|---|---|---|---|---|---|---|
| A1 | 8 | 32×2 | off (fixed bank = on-the-fly tier) | 4 | 200 | ~3.4 GiB |
| A2 | 8 | 32×2 | off | 8 | 200 | ~3.7 GiB |
| B1 | 16 | 64×3 | off | 4 | 200 | ~3.6 GiB |
| B2 | 16 | 64×3 | off | 8 | 200 | ~3.9 GiB |
| C1 | **19** (Whitney) | 64×3 | off | 4 | 200 | ~3.7 GiB |
| C2 | 19 | 64×3 | **on** (fp32 cf cache — "fits-if-fits" arm; n24 cache is small at these dims) | 4 | 200 | ~3.9 GiB; REFUSE→record if projected > envelope |
| D1 | 19 | 64×3 | off | **12** | 200 | verdict-batch spike curve point |

(No fp16-cache flag exists yet — the fp16-feats build is the named future unlock; arm C2 is the fp32
fits-if-fits probe. If any arm hits safe_run's 4000 MiB cap, that IS the measurement: record
`status=oom` + the trajectory as the tier's refusal boundary — the guard never touches the control plane.)

**Per-arm measurement protocol (#205-mine replication):** (1) safe_run exit line peak (authority);
(2) blackbox series → steady between-verdict band, in-verdict mean/p95/max, spike step vs climb split,
leak slope over the arm; (3) mem_probe rows → rss vs mlx_active/cache/peak split per phase (the split
the 128 GB row lacked); (4) one row per arm appended to the #294 ledger
(`.omx/state/memory_projection_ledger.jsonl`, `kind: measured_actual`, tier `m1_8gb`) with the
preflight's projected peak and the residual. Post-#294 units fix (51864adb1/3795339fe/de86ec060,
in-tree on tertiary via parity) tracked rows are true GiB — no ×0.9537 correction needed on NEW rows.

**Launch command template (staggered GO only — DO NOT run before the primary ep0 gate):**

```bash
ssh adpena@100.65.24.39 'cd ~/Projects/pact && OUT=experiments/results/tertiary_edge_sweep_<ARM>_$(date -u +%Y%m%dT%H%M%SZ) && mkdir -p $OUT && \
  TAC_MEM_PROBE=1 nohup .venv/bin/python tools/safe_run.py --rss-mb 4000 --timeout 14400 --label edge_<ARM> \
  .venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py \
  --out-dir $OUT --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n24.npz --num-pairs 24 \
  --mlx-device gpu --seed 0 --epochs 200 --eval-every 25 \
  --mod-dim <MD> --hidden-dim <HD> --n-hidden <NH> --verdict-batch <VB> [--self-orient] \
  --bank-n-scales 2 --bank-n-orient0 4 --bank-n-iso 2 --accum-pairs 4 \
  > $OUT/run.log 2>&1 & disown'
```

Arms run STRICTLY SEQUENTIALLY (one at a time — 8 GB admits exactly one job; concurrent arms would
re-create the machine-crash P0). Band-tick loop alongside:
`while :; do .venv/bin/python tools/system_memory_governor.py --band-tick --band-envelope-frac 0.55; sleep 60; done`.

**Wall-time expectation (ESTIMATE from the smoke, to be refined by each arm's first epochs):** smoke =
50.9 s for 3 ep @ n6 tiny (≈25 s fixed startup). n24 = 4× pairs, dims ~2–4× → ~3–10 s/epoch on M1-GPU
⟹ **~15–35 min/arm at 200 ep + verdicts**; 7 arms sequential ≈ **2.5–4.5 h** total. MLX-GPU numbers are
training-gradient only (`[macOS-MLX research-signal]`); every d_seg/d_pose printed is the frozen
CPU-torch advisory verdict — nothing here is a score.

## 5. Honest gaps

- Governor admission/throttle OFF on tertiary until the margin-floor scaling gap closes (§2); the
  hard cap is safe_run.
- Expected-peak priors in §4 are extrapolations from ONE smoke point — the sweep exists precisely to
  replace them with measured rows; no cross-config interpolation claimed.
- M1 thermal throttling untested beyond 51 s — arm wall-times may stretch; memory envelopes unaffected.
- `pressure` string field reads "critical" at idle on this box while `pressure_level=1` — field
  semantics need a look before alerting off the string (blackbox consumer note).

Pointer 0.19110 UNMOVED — this memo is MEANS (apparatus/measurement), no score claim.
