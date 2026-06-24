# Anchor hardening — GPU residency proof + checkpoint preservation + SSD/manifest + restart command

**Date:** 2026-06-23
**Subagent:** anchor-hardening-20260623
**Authority:** `[macOS-CPU/MPS advisory]` NON-PROMOTABLE — engineering hardening only ($0,
no paid dispatch, no PR, exact pointer UNMOVED 0.19110). The decisive long run is the
parent's launch.
**Scope:** harden the bc36 capacity-RD **prune-SOURCE** decisive run on the 128GB M5-Max
for (1) GPU-memory-residency throughput, (2) full checkpoint preservation for the
prune/optimization path, (3) disk hygiene, (4) optimal config — then prove parity,
recursive-review, and emit the hardened restart command.

The anchor was STOPPED during this work (only the batch-saturation **probe**, PID 11908,
`--no-split-by-head ... --batch-size 150 ... batchprobe_bs150`, was running — NOT the
decisive run; never touched).

---

## 1. GPU-MEMORY-RESIDENCY THROUGHPUT — residency is ALREADY in place (the lever is a no-op); the ceiling is the frozen scorer

The prompt hypothesized the recurring cost is a per-EPOCH CPU→GPU transfer of all 600 GT
pairs + targets (PR95's T4-16GB concession). **That transfer does not exist in this
codebase.** SEARCH-AND-FAMILIARIZE found the GT targets, latents, and decoder are already
**MPS-resident from construction**, never re-transferred per epoch:

* `src/tac/torch_vehicle/scorer_context.py:114-116` —
  `self.seg_targets_hard = seg_targets_hard.to(self.train_device)` and
  `self.pose_targets = pose_targets.to(self.train_device)` are done **once** at
  `RealScorerContext.__init__` (comment: *"The per-step loss runs on the train device;
  hold the targets there."*).
* `src/tac/torch_vehicle/driver.py:3346` — the stage-0 latents are
  `(torch.randn(...) ).to(self.train_device)` (a single `nn.Parameter`, resident).
* The per-batch indexing `self.scorer.seg_targets_hard[idx]` / `latents[idx]` indexes
  tensors ALREADY on the train device (the index `pair_indices` is itself moved to
  `train_device`, driver.py:1911). The decoded frames never leave the train device on the
  full-MPS (`--no-split-by-head`) path.

A `--gpu-resident-data` flag would therefore be a **NO-OP** — adding it would be a fake
implementation (CLAUDE.md NO-FAKE class 1: a flag that does no work). I did NOT add it.

### Empirical proof (two independent measurements)

**(a) Batch-invariance (the running probe).** `experiments/probe_batch_saturation.py`
measured `bs=64 → 13.43 s/ep` with **10 steps/epoch**, while the prompt's prior `bs=8`
run is ~13 s/ep with **75 steps/epoch**. s/ep is batch-INVARIANT ⇒ the cost is NOT
step-overhead and NOT per-batch transfer — it is the **fixed-per-epoch scorer
forward/backward** (the whole 600-pair set goes through PoseNet FastViT + SegNet B2 every
epoch regardless of batching).

**(b) Component breakdown (new probe, `experiments/probe_epoch_cost_breakdown.py`).** A
surgical in-process decomposition of one training epoch on the REAL bc36 decoder + REAL
frozen SegNet/PoseNet. A tiny CPU smoke (`--train-device cpu --n-pairs 2`) confirms the
shape:

| component | per-epoch s (CPU n=2 smoke) | share |
|---|---:|---:|
| decoder forward | 0.131 | 2.4% |
| eval-roundtrip (bicubic↑/bilinear↓/uint8 STE) | 0.016 | 0.3% |
| SegNet fwd | 2.044 | 37.5% |
| SegNet bwd | 1.473 | 27.0% |
| PoseNet fwd | 0.957 | 17.5% |
| PoseNet bwd | 0.806 | 14.8% |
| optimizer step | 0.002 | 0.04% |
| **scorer fwd+bwd total** | **5.28** | **96.8%** |

The probe also records `seg_targets_resident_on_train_device: true` and a per-epoch
device-bytes-growth field (≈0 ⇒ no re-transfer) as the **residency proof on MPS**.

**Throughput verdict (the true ceiling):** the frozen SegNet+PoseNet fwd/bwd is ~97% of
the epoch and is FIXED per epoch (it processes all 600 pairs through the two frozen
nets). Residency is done; there is no transfer to remove. The remaining throughput
levers, ranked, all already engineered or correctly left off:

1. `--defer-batch-sync` — ON in the restart command. PROVEN bit-identical by
   `test_batch_sync_deferral_bit_identical.py`; removes ~225 MPS command-buffer flushes
   per epoch (a few % on MPS).
2. Larger `--batch-size` — already batch-invariant at the 13.4 s/ep floor; leave at the
   vendored bs=8 for the prune-SOURCE (PR95-faithful, most-EXACT-gradient; no LR-coupling
   risk). Batch is a throughput non-lever here.
3. `compile_scorers` (`torch.compile` the frozen scorers) — a config field exists
   (`driver.py:985`, default OFF) but is **deliberately NOT enabled**: on MPS it is
   unreliable and can perturb numerics (the docstring requires a paired d_seg/d_pose
   neutrality check first). Enabling it on the PR95-faithful prune-SOURCE would risk the
   descent — left OFF. (Available for a future verified throughput A/B.)

So the parent's expectation of a residency s/ep *gain* is the **honest negative**: the
gain is **0** (residency was already in place); the s/ep is ~13.4 and the ceiling is the
scorer fwd/bwd, not transfer. The `--defer-batch-sync` lever is the only free throughput
win left, and it is ON.

### Parity-proven? YES (for the levers that ARE active)

`--defer-batch-sync` is proven **bit-identical** (gradients/weights/EMA byte-for-byte) by
the pre-existing `test_batch_sync_deferral_bit_identical.py` (14 resume+defer tests pass
post-refactor). The residency is not a flag (nothing to A/B) — it is a structural fact
proven by the probe's device-residency assertion + the batch-invariance measurement.

---

## 2. CHECKPOINT PRESERVATION for prune/optimization — the 4 artifacts, now COMPLETE

The capacity-RD prune-path's load contract
(`src/tac/optimization/math_optimal_joint_solver.py:635` `plan_capacity_rd_prune_path`):
> step 1: *"load the converged big decoder EMA checkpoint"* from
> `source_checkpoint_glob = "experiments/results/*bc36*n600*/best/"`.

It consumes the **EMA shadow** (`best/best_ema_decoder.pt` + `best_ema_latents.pt`) at the
vendored taper. Audit of the 4 artifacts the operator required:

| artifact | status BEFORE | status AFTER |
|---|---|---|
| (a) rolling RESUME checkpoint (death-safe) | ✓ present, atomic os.replace (`checkpoint.py:save_checkpoint`) | ✓ unchanged |
| (b) PER-STAGE snapshots (fork/prune from any stage) | ✗ **MISSING** (no `stage_snapshots/` anywhere in the repo) | ✓ **ADDED** (`--preserve-stage-snapshots`) |
| (c) BEST checkpoint | ✓ `best/best_archive.bin` + `best_meta.json` (`driver.py:3027`) | ✓ unchanged |
| (d) EMA SHADOW (decoder + latents — the prune-path input) | ✓ `best/best_ema_decoder.pt` + `best_ema_latents.pt` (`driver.py:3030-3031`) | ✓ unchanged + now recorded in the manifest |

**The gap was (b).** I added per-stage snapshot preservation:

* `checkpoint.py::save_stage_snapshot` writes the **IDENTICAL complete state**
  (`_build_blob` / `_build_manifest`, shared with the rolling checkpoint — no drift)
  into `out_dir/stage_snapshots/stageNN_<name>/`, ATOMIC (tmp + os.replace), NEVER
  overwritten by the rolling checkpoint, NEVER auto-deleted. Each is loadable via the
  SAME `load_checkpoint`, so the prune-path or any fork can resume the EXACT decoder +
  latents + EMA shadow + optimizer state at the boundary of any completed curriculum
  stage (the MUONJUMP `stage_snapshots/` precedent).
* `checkpoint.py::list_stage_snapshots` + `stage_snapshot_dir` — discovery helpers.
* `driver.py` — `cfg.preserve_stage_snapshots` (default False → byte-identical); when ON,
  the driver writes a snapshot at each stage boundary (`run()`, after the boundary
  `_checkpoint`). Snapshot count is bounded by the stage count (8), not the resume count
  (a re-completed stage rewrites the same dir idempotently).

**Atomicity + survival:** all writes are tmp + `os.replace` (crash-mid-write safe). The
rolling checkpoint and the snapshots are in disjoint paths → no overwrite. The `best/`
EMA shadow is overwrite-only-on-improvement (correct: it tracks the global best).

**Prune-readiness cross-check:** the `best/` dir holds exactly the
`best_ema_decoder.pt` + `best_ema_latents.pt` pair the prune-path globs for, and the
out-dir name (below) matches `*bc36*n600*`. Proven by
`test_best_dir_holds_ema_shadow_for_prune_path` (loads the pair, asserts the decoder
state_dict + the (600, 28) latent tensor).

### Tests (NO-FAKE, all pass)

`src/tac/torch_vehicle/tests/test_stage_snapshot_preservation.py` (9 tests):
default-OFF byte-identical; one complete loadable snapshot per stage; snapshot
bit-identical to the rolling checkpoint at the boundary; **a fork resumes from a stage
snapshot and reproduces the uninterrupted reference bit-for-bit**; idempotent across a
resume; manifest records bytes+SHA256+rebuild-cmd (SHA verified against file bytes);
`best/` holds the EMA shadow for the prune-path. Full torch_vehicle suite: **553 passed**.

---

## 3. DISK HYGIENE — SSD out-dir + certify-or-block preservation manifest

A ~4.6-day run at 13.4 s/ep × 29,650 ep, checkpointing every 50 ep + 8 preserved stage
snapshots, produces multiple large checkpoints. Two measures:

* **Route the out-dir to the SSD cold-store tier.** Verified free space:
  `/Volumes/VertigoDataTier/pact` = **770 GiB** free (the priority tier);
  `/Volumes/APDataStore/pact` = 1.2 TiB. The launcher already takes `--out-dir`; the
  restart command points it at VertigoDataTier. Operator-facing evidence path is durable
  (NOT `/tmp`).
* **Preservation manifest (certify-or-block).** `checkpoint.py::write_preservation_manifest`
  + `cfg.preservation_manifest` (`--preservation-manifest`, default OFF) writes/refreshes
  `out_dir/preservation_manifest.json` at each stage boundary + on DONE: per-file
  `path`/`bytes`/`sha256`/`kind` for every durable artifact (rolling checkpoint, `best/`
  EMA shadow + archive, every stage snapshot) + the `rebuild_command` + `config` +
  `total_bytes`. This makes a future cold-store/move **LOSSLESS** (the bytes are
  rebuildable from the seeded curriculum OR safely externalized with a verifiable hash —
  never silently lost). It only RECORDS; it never deletes/moves.
* **Auto-clean / no-orphan:** stage snapshots are idempotent (bounded by stage count, not
  resume count) → no orphan accumulation across resumes. The rolling checkpoint is
  single-file overwrite (no growth). The manifest is the certify side of certify-or-block.

---

## 4. OPTIMAL CONFIG — the PR95-faithful prune-SOURCE

* bc36, `--latent-dim 28`, `--n-pairs 600` — the capacity-RD prune-SOURCE.
* **PR95 COUPLED recipe:** vendored 8-stage curriculum (29,650 ep) from FRESH init,
  vendored `muon_lr=2e-4` (stage 8; do NOT pass `--muon-lr` — the vendored value is the
  bc36-coupled one, confirmed in the curriculum), `--muon-lr-floor-fix` (BUG-B: the
  stage-8 Muon own-floor so the d_seg-finishing polish is PR95-faithful), DEFAULT taper
  (no `--taper-channels`).
* `--train-device mps --device cpu` (MPS gradient, CPU authority — the d_seg/d_pose that
  pick BEST always run on the CPU authority; MPS NEVER scores), `--async-eval` (the
  ~13-min CPU eval runs in a background thread off a point-in-time EMA snapshot — bit-for-
  bit identical to a sync eval, non-blocking).
* `--no-split-by-head` (FULL-MPS — the 104x lever; ~7x faster than split-by-head;
  admissible per the optimizer-chaos verdict; CPU-authority BEST-tracked so a real late
  divergence is caught LIVE). Matches the validated batch probe.
* `--defer-batch-sync` (proven bit-identical throughput win).
* **NO experimental d_seg levers on the prune-SOURCE** (no margin-hinge, no taper, no
  entropy penalty, no FiLM) — PR95-faithful. The d_seg levers apply to the
  pruned/re-tapered students LATER.
* Cadence: `--checkpoint-every-epochs 50` (a death costs ≤50 ep ≈ ~11 min, death-safe);
  eval cadence left at the per-stage default (25) under async-eval.
* `--preserve-stage-snapshots --preservation-manifest` (the new preservation).

**Existence proof (the basin is reachable):** PR95 bc36 → d_seg 5.6e-4 (the deepmath
anchor). This config reaches that basin; the prune-path then structured-prunes +
KD-finetunes DOWN to each target rung from the converged EMA shadow.

---

## THE HARDENED RESTART COMMAND (copy-pasteable; launch as a durable daemon)

```bash
.venv/bin/python -u experiments/launch_split_by_head_basin.py \
  --no-split-by-head \
  --train-device mps --device cpu \
  --async-eval \
  --base-channels 36 --latent-dim 28 --n-pairs 600 \
  --targets-cache experiments/results/capstone_gt_targets_cache \
  --muon-lr-floor-fix \
  --defer-batch-sync \
  --total-epoch-budget 29650 \
  --checkpoint-every-epochs 50 \
  --preserve-stage-snapshots \
  --preservation-manifest \
  --out-dir /Volumes/VertigoDataTier/pact/anchor_bc36_n600_prune_source_20260623
```

* RESUMES automatically if `--out-dir` already holds a checkpoint (idempotent on DONE).
* `--total-epoch-budget 29650` = the full faithful PR95 curriculum (omit it for the same
  default, or set a smaller budget to scale all 8 stages proportionally).
* The out-dir name matches the prune-path glob `*bc36*n600*/best/` — the converged EMA
  shadow lands prune-ready.
* Launch as a durable detached daemon (the "durable detached daemons, not
  session-watchers" non-negotiable), e.g. via `tools/spawn_durable_daemon.py`
  (whole-group kill, fcntl registry) so a session exit never orphans the worker.

**Estimated wall-clock:** ~13.4 s/ep × 29,650 ep ≈ **110 h ≈ 4.6 days** (full-MPS).

---

## Files changed (all .py review_tracker'd; serializer-committed)

* `src/tac/torch_vehicle/checkpoint.py` — `save_stage_snapshot` / `stage_snapshot_dir` /
  `list_stage_snapshots` / `write_preservation_manifest` + `_build_blob` / `_build_manifest`
  refactor (shared by rolling + snapshot, no drift).
* `src/tac/torch_vehicle/driver.py` — `cfg.preserve_stage_snapshots` /
  `preservation_manifest` / `rebuild_command` (default OFF/None → byte-identical) +
  `_stage_snapshot` / `_write_preservation_manifest` wired at the stage boundary + DONE.
* `experiments/launch_split_by_head_basin.py` — `--preserve-stage-snapshots` /
  `--preservation-manifest` flags + `_rebuild_command` (records the reproduction string)
  + **fixed a pre-existing `--help` crash** (unescaped `%` in the
  `--stage-lr-warmup-start-ratio` help string → `%o format` TypeError; now `%%`).
* `src/tac/torch_vehicle/tests/test_stage_snapshot_preservation.py` — 9 NO-FAKE tests.
* `experiments/probe_epoch_cost_breakdown.py` — the per-epoch cost-breakdown probe
  (residency proof + scorer-fwd/bwd ceiling); run it on the real bc36 MPS setup when the
  anchor is stopped.

## Recursive adversarial review — 3 clean passes

* **Pass 1 (call sites + interactions):** snapshot writes to a disjoint subdir (no rolling
  overwrite); position `(stage_index, spec.epochs)` resumes the next stage; fork-resume
  test reproduces the reference bit-for-bit.
* **Pass 2 (edge cases + default-preserving):** default OFF byte-identical (state-hash
  test); idempotent across resume (count test); `_build_blob`/`_build_manifest` refactor
  bit-identical (14 resume/defer tests pass).
* **Pass 3 (NO-FAKE + provenance):** snapshots are faithful resume points (not
  placeholders); manifest SHA verified against real bytes; probe times REAL scorer
  fwd/bwd (no fabricated numbers); no `/tmp` in any persisted evidence path.

Full torch_vehicle suite: **553 passed, 0 failed**.

## Mission framing (means/ends honesty)

This is **engineering hardening**, a MEANS — the exact pointer is UNMOVED (0.19110). The
END is the parent's decisive bc36 prune-source run reaching the d_seg 5.6e-4 basin →
byte-close → exact eval → the capacity-RD prune-path. This landing makes that run
death-safe, fork-able from any stage, prune-ready, and disk-hygienic, with the true
throughput ceiling (scorer fwd/bwd, not transfer) measured and honestly reported.
