# M1 tertiary — held-out gen-gap gate FIRE runbook (provisioned 2026-06-27)

**Status:** M1 PROVISIONED + gate machinery VALIDATED end-to-end (smoke SMOKE_OK=true).
**Means≠ends:** infra for a FUTURE exact-eval row; pointer UNMOVED 0.19110. macOS=advisory;
the gate's authority is its **frozen CPU-torch SegNet argmax + PoseNet MSE** verdict (NEVER MPS/MLX).

## M1 readiness (verified)

| item | state |
|---|---|
| host | M1 tertiary, Tailscale `<tailscale-ip-redacted>`, user `<user>` |
| repo | `~/Projects/pact` — `git clone --depth 1` from M5, **HEAD `a534e2ee74fd96acafa0322b0d1b4cfcb1f5eadb` == M5 HEAD** (true git remote-code-parity; no marker hack) |
| venv | `~/Projects/pact/.venv` (uv, CPython 3.13.14 — matches M5 3.13.x) |
| deps (pinned to M5) | mlx 0.31.1, torch 2.11.0, timm 1.0.26, einops 0.8.2, safetensors 0.7.0, segmentation_models_pytorch 0.5.0, numpy 1.26.4, scipy 1.17.1, brotli 1.2.0 |
| import probe | gate's full import chain imports cleanly (`MISSING_TOPLEVEL: []`) |
| upstream (gitignored on M5 → manually synced) | `upstream/{modules.py, frame_utils.py, evaluate.py}` + `upstream/models/{segnet,posenet}.safetensors` |
| held-out GT cache | `experiments/results/mlx_fleet_gt_cache/gt_heldout_n400.npz` — **3385345610 bytes, size-verified == M5** (rsync checksum-verified) |
| smoke ckpt stand-in | `~/Projects/pact/.omx/tmp/smoke_decoder_ema.npz` (amort EMA, in_feat=88) |
| **SMOKE** | `rc=0 SMOKE_OK=true secs=369` — render(MLX-gpu)→R→**CPU-torch SegNet** verdict emits real d_seg (0.0363 @ 2pairs/3ep — NOT the verdict, just machinery proof). in_feat pre-check 88==88 (no width fail-close). |

## ⚠️ THE 8GB RAM CONSTRAINT (the one caveat — read before firing)

M1 has **8GB RAM**. `load_gt_from_cache` materializes the WHOLE n400 cache (~3.2GB resident)
regardless of `--num-pairs`, plus the CPU-torch scorers + MLX render. During the smoke this drove
the M1 into **~9.4GB swap** (macOS grew swap on the 31GB-free disk; no OOM-kill, but the 3.2GB
cache load dominated the 369s). Consequences:

1. The gate's default `--launch` requests `--projected-gb 15` → `spawn_durable_daemon`'s mem-preflight
   will **REFUSE on 8GB**. So **do NOT use `--launch`** on the M1 — use the `--dry-run`→daemon path below.
2. The n400 fire on the M1 WILL run but **slower than the M5** (swap-bound cache load + heavier per-epoch).
   The per-step code-fit footprint (decoder frozen, only codes train, accum-pairs=8) is modest; the
   one-time 3.2GB cache load is the swap cost.
3. **Recommended host for the real n400 fire = the M5 post-amort** (128GB, no swap). Use the M1 as a
   PARALLEL/BACKUP arm only if both fires are wanted at once (accepting the M1 is slower), or build a
   smaller `gt_heldout_n96` cache for the M1 (weaker but RAM-fitting held-out test).

## FIRE SEQUENCE (when the M5 amort reaches l7, `l7_start_epoch=900`)

Front-end note: the amort `levelset_amort_deconf_n200_taualone_20260627T194432Z` config
(activation hosc, hidden 96, mod 32, n_dir_freqs 2, freq_across 32, freq_along 4, max_bank_freq 64,
chroma, self_orient, in_feat **88**) **exactly matches the gate's SEALED defaults** — no front-end
flag overrides needed. (If a future decoder uses a different front-end, override the gate's
`--n-dir-freqs/--freq-across/--freq-along/--max-bank-freq` to reproduce its in_feat or the trainer
fail-closes on width mismatch.)

```bash
# === STEP 1 (on M5): locate the l7 stage EMA ckpt (preserved, stage-encoded) ===
RUN=experiments/results/levelset_amort_deconf_n200_taualone_20260627T194432Z
L7=$(ls -t "$RUN"/levelset_ckpt_stageL7_ep*.npz | head -1); echo "$L7"

# === STEP 2 (on M5): rsync it to the M1 (~350KB) ===
rsync -a "$L7" <user>@<tailscale-ip-redacted>:/Users/adpena/Projects/pact/.omx/tmp/l7_decoder_ema.npz

# === STEP 3 (on M1): generate the exact trainer cmd via the gate --dry-run ===
ssh <user>@<tailscale-ip-redacted> 'cd ~/Projects/pact && env TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 \
  .venv/bin/python tools/levelset_heldout_codefit_gate.py \
    --decoder-ckpt .omx/tmp/l7_decoder_ema.npz \
    --heldout-gt experiments/results/mlx_fleet_gt_cache/gt_heldout_n400.npz \
    --num-pairs 400 --epochs 600 --dry-run'
# -> prints {"stage":"dry_run","out_dir":".../heldout_gate_n400_<ts>","cmd_file":".omx/tmp/heldout_gate_cmd_<ts>.txt","cmd":"..."}

# === STEP 4 (on M1): run that trainer cmd under a durable daemon, 8GB-safe (skip mem-preflight) ===
ssh <user>@<tailscale-ip-redacted> 'cd ~/Projects/pact && \
  CMDFILE=$(ls -t .omx/tmp/heldout_gate_cmd_*.txt | head -1) && \
  .venv/bin/python tools/spawn_durable_daemon.py --label m1_gengap_fire \
    --log .omx/tmp/m1_gengap_fire.log --skip-mem-preflight --min-free-gb 5 \
    --rss-cap-mb 7000 --walltime-cap-s 172800 -- $(cat "$CMDFILE")'
# NOTE rss-cap 7000 leaves ~1GB for the OS; if it dies status=oom, re-run with a higher cap OR fire on the M5.

# === STEP 5 (poll): ===
ssh <user>@<tailscale-ip-redacted> '/Users/adpena/Projects/pact/.venv/bin/python \
  /Users/adpena/Projects/pact/tools/spawn_durable_daemon.py --status' | grep m1_gengap_fire

# === STEP 6 (HARVEST verdict back to M5 when DEAD) ===
OUT=$(ssh <user>@<tailscale-ip-redacted> 'ls -td ~/Projects/pact/experiments/results/heldout_gate_n400_* | head -1')
rsync -a "<user>@<tailscale-ip-redacted>:$OUT/levelset_train_result.json" reports/m1_gengap_verdict.json
# verdict d_seg = last history[].d_seg.  PASS if held-out d_seg <= 0.00186 ( = JOINT_REFERENCE_DSEG 0.00124 * 1.5 ).
# PASS => frozen decoder amortizes -> fit all 600 codes -> byte-close -> exact eval.
# FAIL => honest DEFER + resume the preserved n600 row.
```

## M5-fallback fire (recommended for full n400; 128GB, no swap)

Identical gate invocation ON THE M5 (just use `--launch` — the M5 has the RAM for projected-gb 15):
```bash
env TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 .venv/bin/python tools/levelset_heldout_codefit_gate.py \
  --decoder-ckpt "$L7" \
  --heldout-gt experiments/results/mlx_fleet_gt_cache/gt_heldout_n400.npz \
  --num-pairs 400 --epochs 600 --launch
```

## Provisioning notes / gotchas (for the next agent)
- macOS has **no `setsid`** — detach via `spawn_durable_daemon.py` (Python `start_new_session`), not bare setsid.
- M5 ships **openrsync 2.6.9** — `--info=progress2`/`--inplace` are REJECTED; use `-aP` (resume + progress).
- `spawn_durable_daemon --rss-cap-mb` is RESIDENT-RSS; set it ABOVE the working set or it OOM-kills
   (the 3.2GB cache transfer was killed at a 2000MB cap; 8000 worked, peak_rss 3328MiB).
- `upstream/` is **gitignored** in the main repo (it's the nested contest checkout) → not in the clone; sync the needed files manually (done).
