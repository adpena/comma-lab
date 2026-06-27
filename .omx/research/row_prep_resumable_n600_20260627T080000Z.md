# ROW PREP — RESUMABLE + PER-STAGE-CHECKPOINTING n600 trainer; w_pose=1.0 n600 launch ready

**UTC:** 2026-06-27T08:00:00Z · **Author:** row-prep subagent (DAG **FEED-eb**) ·
**Axis:** `[macOS-CPU advisory] / [macOS-MLX training-gradient]` NON-PROMOTABLE · **Pointer:** UNMOVED 0.19110.
**Constraints honored:** CPU-only `$0` (GPU pid 72602 = the n96 mod-32 self-orient baseline, UNTOUCHED + healthy
at ~ep615 throughout); additive/default-off edits; EMA-shadow save; atomic writes; no `/tmp` artifacts;
serializer commit with `--expected-content-sha256`; review-gate mark-file on the `.py`.

This is ROW-INFRA in direct service of an imminent exact row (justified per THE GOAL): the trainer saved its
EMA npz ONLY at loop-end, so the multi-day n600 row was non-resumable (crash = total loss) and had no early
byte-close. That blocker is now closed; the w_pose=1.0 n600 launch command is ready below.

---

## 1. WHAT LANDED (additive, default-safe, tested) — `experiments/train_levelset_witness_realized_through_R_mlx.py`

Per the operator binding rule 2026-06-27 ("never launch anything non-resumable; save + preserve a complete
checkpoint at the END OF EACH STAGE; loop-end-only is FORBIDDEN"):

- **`--resume-from <dir|npz>`** (default None = fresh = UNCHANGED behavior). Restores decoder + per-pair
  `code` (live weights) + EMA shadow + optimizer state (best-effort) + the epoch position, and continues
  `range(start_epoch, epochs+1)`. A run DIR prefers `levelset_resume_state.npz`, falls back to the deploy
  `levelset_witness_ema_mlx.npz` (uses the EMA shadow as live). Self-orient dir-feats are NOT stored (they
  are O(GBs) at n600) — they are regenerated deterministically from the restored EMA argmax fixed-point at
  resume (`recompute_self_orient`), so resume is exact-enough with zero bloat. Crash/OOM/operator-cut loses
  at most one `--ckpt-every` interval, never the whole run.
- **PER-STAGE PRESERVED checkpoints (default ON, `--stage-checkpoints`/`--no-stage-checkpoints`):** at every
  curriculum-stage TRANSITION (CE→tau→l7; the `_seg_form_for_epoch` boundary) AND at the final epoch, write a
  PRESERVED, stage-encoded, byte-close-loadable ckpt `levelset_ckpt_{stageCE,stageTau,stageL7}_ep{N}.npz`
  (+ matching `levelset_resume_{stage}_ep{N}.npz`). NOT overwritten → per-stage A/B of which stage moves
  d_seg + durability in one.
- **INTRA-STAGE rolling (`--ckpt-every N`, default 0 = off):** every N epochs overwrite the rolling
  `levelset_witness_ema_mlx.npz` + `levelset_resume_state.npz` (the crash-resume window for the long stages).
- **EMA-shadow save (non-negotiable):** every checkpoint writes the EMA SHADOW (deploy weights), never live.
- **Atomic writes:** `_atomic_savez` = tmp + `os.replace` (no partial/corrupt npz if the process dies
  mid-write); refuses `/tmp`-class paths.
- **Closed the byte-close cfg-persist gap (flagged in `tools/levelset_byte_close_and_eval.py`):** the deploy
  npz now also persists `__cfg_self_orient / __cfg_n_dir_freqs / __cfg_freq_across / __cfg_freq_along /
  __cfg_reorient_every / __cfg_w_pose / __cfg_in_feat / __cfg_curriculum / __cfg_tau_softplus_start_epoch /
  __cfg_l7_start_epoch / __epoch` (additive scalars; the byte-close tool reads cfg keys selectively → harmless
  + better provenance). The deploy npz keeps its canonical name + every historical key, so the byte-close
  tool consumes it unchanged.

**Default-safe proof:** with `--ckpt-every 0` + the existing single-stage default, the only behavioral delta
vs the prior code is that MORE checkpoint files are written (strictly additive on disk; training math —
loss/opt/order/spike-guard/EMA — is byte-identical). The running pid-72602 process loaded the old module into
memory and is unaffected.

### Validation (`$0` CPU, GPU untouched)
- `py_compile` OK; module imports clean.
- **18/18 unit tests** PASS (`experiments/tests/test_levelset_checkpoint_resume.py`, MLX-free): atomic write
  (loadable / no leftover tmp / overwrite / `/tmp`-refusal); EMA-dict reproduces ALL historical cfg keys +
  the new provenance keys + max_bank_freq None→−1 sentinel + softmax-temp passthrough; resume sidecar
  prefix/round-trip (with + without optimizer) + EMA-npz fallback; `_resolve_resume_path` dir-preference /
  ema-fallback / explicit-file / missing-raise; `_stage_tag` mapping; **INTEGRATION: the built EMA npz is
  loadable by `tools/levelset_byte_close_and_eval._load_levelset_ckpt`** (the exact-row consumer).
- **$0 end-to-end CPU smoke** (n6, render-48, curriculum tau@2/l7@4, `--ckpt-every 2`, `--self-orient`,
  `--w-pose 1.0`, `--mlx-device cpu`): stage-transition PRESERVED ckpts fired (`stageCE_ep1`, `stageTau_ep3`),
  intra-stage rolling fired (ep2), `has_opt:true`; the rolling + a PRESERVED stage npz both byte-close-load;
  `--resume-from run1` continued from the saved epoch (resume + resume_reorient + verdict + further
  checkpoints). Smoke scratch under `experiments/results/_feed_dz_ckpt_smoke/` (deleted post-validation).

---

## 2. THROUGHPUT / AMORTIZATION VERDICT

### Measured anchor (the running GPU baseline, READ-only)
- **n96 = 27.08 s/epoch** (MEASURED: the live pid-72602 n96 mod-32 self-orient run reached epoch 600 in
  16,247 s; verdict-pairs=96, eval-every-25, hosc, render-384, accum-pairs 8). d_seg descending
  0.00178 @ ep600, still in the tau stage.

### n600 realistic burn (linear-in-pairs extrapolation)
The per-epoch cost is dominated by P realized value_and_grad calls (render → R → frozen SegNet+PoseNet
fwd+bwd), LINEAR in P. n96→n600 = 6.25× pairs:
- **n600 ≈ 169 s/epoch ≈ 2.8 min/epoch → 1500 epochs ≈ 70.5 h ≈ ~3 days.** (Verdict cost is bounded by
  keeping `--verdict-pairs 96`, amortized over eval-every-25 → negligible per-epoch.) Early byte-close at
  the CE→tau preserved ckpt (~ep300, ~14 h) and the tau→l7 ckpt (~ep900, ~42 h) yields advisory rows long
  before the 3-day completion. **This ~3-day burn is exactly why the resume + per-stage checkpointing landed
  here is mandatory before launch.**

### Amortization (decoder-subset train + cheap 600-code fit) — DESIGNED, faster but UNVALIDATED
The architecture is NeRV-amortized: the decoder (`in_proj/film/hidden.*/out_sdf/out_tex/palette`) is SHARED
across all pairs; only `code` (1200×mod) is per-(pair,frame). So a faster path EXISTS in principle:
1. Train the shared decoder on a SUBSET (n96/n192) to a good d_seg (~hours, the running run is this);
2. FREEZE the decoder, fit ONLY the 1200 per-pair codes for all 600 pairs (a 32-dim per-pair optimization
   through the frozen render+R+SegNet/PoseNet; ~50-100 steps/pair ≈ ~75 "epochs"-equivalent ≈ hours, and
   embarrassingly parallel per pair).

**VERDICT: do NOT gate the row on amortization.** It is (a) UNVALIDATED — whether an n96-trained decoder
generalizes so frozen-decoder code-fit reaches comparable d_seg on the other 504 pairs is untested (NeRV
amortization holds when the subset spans scene diversity; novel per-frame geometry the shared basis can't
express would plateau higher), AND (b) requires a NEW freeze-decoder/code-only training mode the trainer does
not have (a separate landing, out of scope for this additive prep). With w_pose=1.0 the per-pair pose ALSO
rides the codes, which HELPS amortization (pose is inherently per-pair) but adds a wrinkle the code-fit must
satisfy. **Recommendation:** launch the full n600 row NOW (known-good, now resumable + early-byte-closeable);
prototype amortization as the speedup for FUTURE rows once the GPU frees.

### Fastest FEASIBLE path for THIS row
**Full n600 with per-stage + intra-stage checkpointing (the launch command below).** It is the only
known-good path to a real w_pose=1.0 exact row, the ~3-day burn is now crash-safe, and the per-stage
preserved ckpts turn the single multi-day run into N early byte-closeable advisory rows (per-stage A/B of
which curriculum stage moves d_seg + d_pose).

---

## 3. READY w_pose=1.0 n600 ROW — LAUNCH COMMAND (do NOT launch while pid 72602 holds the GPU)

Reviewed/SEALED Yousfi-levers optimal-form config (`.omx/research/yousfi_levers_optimal_form_review_20260627T063335Z.md`,
SEALED 3/3) with `--w-pose 1.0` (pose VERIFIED a1116e516f: realized store+supervise, parent existence-proof
d_pose 12.94→0.0009) + the new `--ckpt-every 100` + per-stage checkpoints (default on), wrapped in the durable
daemon with the scale-safeguard (RSS cap ~90 GB for n600's ~5 GB gt + witness; min-free 10 GB floor;
walltime cap ~80 h):

```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)
.venv/bin/python tools/spawn_durable_daemon.py \
  --log .omx/tmp/levelset_n600_wpose1_${TS}.log \
  --label levelset_n600_wpose1 \
  --rss-cap-mb 90000 --min-free-gb 10 --walltime-cap-s 288000 \
  -- env TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 \
  .venv/bin/python -u experiments/train_levelset_witness_realized_through_R_mlx.py \
    --out-dir experiments/results/levelset_n600_wpose1_${TS} \
    --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
    --num-pairs 600 --epochs 1500 --render-h 384 --render-w 512 \
    --hidden-dim 96 --mod-dim 32 --activation hosc --siren-init \
    --softmax-temp-start 1.0 --softmax-temp-end 0.05 \
    --curriculum --tau-softplus-start-epoch 300 --l7-start-epoch 900 \
    --palette-anchor --self-orient --reorient-every 50 \
    --freq-across 32 --n-dir-freqs 2 --freq-along 4 --max-bank-freq 64 \
    --chroma \
    --lane-edge-weight 30 --lane-edge-class 1 --lane-margin-target 0.5 --lane-edge-start-epoch 300 \
    --w-seg 100 --w-pose 1.0 --eikonal-weight 0.01 --length-weight 0.001 \
    --ema-decay 0.997 --accum-pairs 8 --grad-clip 1.0 --verdict-pairs 96 \
    --eval-every 25 --ckpt-every 100 --mlx-device gpu
```

**Crash/OOM resume:** `... --resume-from experiments/results/levelset_n600_wpose1_${TS} ...` (same command +
`--resume-from <that run dir>`) continues from the rolling/stage checkpoint.

**Early byte-close (advisory rows during the run):**
```bash
.venv/bin/python tools/levelset_byte_close_and_eval.py \
  --ckpt-dir experiments/results/levelset_n600_wpose1_${TS} \
  --npz-name levelset_ckpt_stageTau_ep900.npz \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --keep-packet
# -> packet archive.zip + inflate.py + realized d_seg/d_pose on inflated frames + staged contest-CPU cmd
```

### Monitor / abort plan (read-only verdict-log watch)
- **Watch:** `.omx/tmp/levelset_n600_wpose1_${TS}.log` verdict stream every ~25 ep.
- **GO signals:** d_seg tightening toward the < 0.00112 goal (mod-32 baseline floor was 0.0018 @ w_pose=0);
  d_pose DESCENDING toward ~0.0009 (the parent existence-proof); blob_bytes ~122-130 KB (RD-optimum B*).
- **NULL-CONTAINMENT (the decisive w_pose=1.0 test):** does adding pose HURT d_seg vs the w_pose=0 mod-32
  baseline (0.0018)? If d_seg rises materially while d_pose falls, pose is competing with seg in the
  SegNet-null space → ABORT/iterate (reduce w_pose or null-project). If d_seg holds/improves while d_pose
  descends, the row is healthy.
- **ABORT triggers:** spike_skip storm (loss instability), d_seg > baseline + 0.0005 persistent (pose
  antagonism), or RSS approaching the 90 GB cap (safe_run kills the arm cleanly; resume from last ckpt).

---

## NO-FAKE / compliance
All edits additive/default-off; EMA-shadow save; atomic writes; no `/tmp` in artifacts. No score / frontier /
promotion / kill claim — pointer UNMOVED 0.19110; axis `[macOS-CPU advisory]` NON-PROMOTABLE. The exact row
is produced ONLY when the byte-closed packet runs through `upstream/evaluate.py` on contest-CPU (Linux x86_64)
/ contest-CUDA. GPU pid 72602 (n96 baseline) UNTOUCHED throughout. Commit via serializer with
`--expected-content-sha256`.
