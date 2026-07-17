# Boundary-readiness rehearsal — 20260717

De-risking the post-v9c2 endgame pipeline so the pointer attempt hits ZERO first-time-integration
failures at the boundary. All three rehearsals ran READ-ONLY against the LIVE run
(`experiments/results/levelset_n600_witness_20260717T113932Z`, best d_seg=0.003458 @ ep725) with
copies into scratch. Memory authority = `tools/mem_basis.conservative_free_gib()` (18.5–28 GiB
free through the session; the 76-GiB trainer stayed up). 1-thread env, `nice -n 10`, live run dir
never written.

**Authority for every number below: `[macOS-CPU advisory] NON-PROMOTABLE`. Pointer UNMOVED.**

---

## R1 — BYTE-CLOSE PIPELINE on the live BEST checkpoint — **PASS** (after a fix)

Copied `levelset_witness_ema_BEST.npz` (self-contained; cfg lives in the npz `__cfg_*` keys — no
sidecar needed for the base path) to scratch; ran the FULL byte-close + inflate + 32-pair scorer
parity against `gt_n96.npz`.

**First run FAILED — the exact class the rehearsal exists to catch.** `parity_on_inflated` (and its
pose-carrier sister `pose_carrier_confirm`) were passed the full checkpoint `n_pairs` (600) as the
GT-load count, while they only score `P = min(eval_pairs, gt.n_pairs)` pairs. So a 32-pair
`--max-pairs` smoke demanded a ≥600-pair GT cache and raised
`ValueError: --gt-cache ... has only 96 pairs < requested --num-pairs 600`. A capped smoke could
never run against anything but `gt_n600`. **Fixed directly (tools-level):** pass
`inflate_info["eval_pairs"]` as the GT-load count — behaviour-identical for full eval
(`eval_pairs == n_pairs`), correct for capped smokes. Committed `42c2cfd2c7`.

**Post-fix run — end-to-end GREEN:**

| Stage | Result |
|---|---|
| byte-close | `0.bin=84536 B`, `archive.zip=83837 B`, rate=0.002233, **rate_term=0.0558**, bank=FREE (rule118) |
| inflate / receiver decode | `(64, 874, 1164, 3)` uint8 [f0,f1 per pair] **full_output_ok=True**, raw_bytes=195,328,512 → round-trips |
| 32-pair parity | **d_seg=0.003146** (vs run verdict 0.003458 — inside the ~0.0035 band ✓); d_pose=149.23 |
| wall-clock | 42.2 s real |
| peak RSS | **6.39 GiB** (32-pair; streamed .raw readback, no full-P batch) |

**d_pose=149 is EXPECTED, not a blocker.** This render is `w_pose=0` (POSE-BLIND by design); the
tool loudly flags it. Pose is banked separately (R1 dxi 7.2 KB / the `--pose-carrier` path). The
S_advisory=39 is garbage *because* pose is blind — as designed. For the real pointer attempt the
n600 byte-close must run with `--pose-carrier` composing the banked pose (see the run's pose bank).

**Post-run n600 planning (do NOT run next to the trainer):** the n600 leg loads `gt_n600.npz`
(4.8 GB npz → ~4.8 GiB resident) + the ~3.66 GB `.raw` on disk (streamed on readback, bounded RSS)
+ scorer. Budget **peak ~10–12 GiB**; gate on `conservative_free_gib() > ~15` OR run with the
trainer down. Storage preflight (`decode_cpu_16gb`, contest=True, bit_exact=True) passed.

**Reusable command (post-run, full n600):**
```
OMP_NUM_THREADS=1 .venv/bin/python tools/levelset_byte_close_and_eval.py \
  --ckpt-dir experiments/results/<final_run> --npz-name levelset_witness_ema_BEST.npz \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --pose-carrier --pose-carrier-xi-from-ckpt --keep-packet --out reports/<final>.json
```

---

## R2 — POSE-TARGET artifact for the apply-pass fire — **PASS** (adapter branch, no recompute)

`tools/witness_applypass_batch.py`'s `low_rank_pose_140` lever needs `--pose-target (600,6)` and
reported OWED without it. **The (600,6) GT PoseNet targets ALREADY EXIST** — `gt_n600.npz` carries
them under the `gt_poses` member (each row = `_cpu_pose_raw(posenet_cpu, f0, f1)` = frozen-CPU-torch
`PoseNet(GT_orig_pair)[:6]`, the exact `d_pose` target). No 600 forwards needed — this is the
adapter branch.

Extracted ONLY the tiny `gt_poses` member (lazy npz read — peak RSS **207 MB**, 0.5 s; no full-cache
spike) → `torch.save` as float32 (600,6):

- **Artifact:** `experiments/results/pose_targets/gt_pose_targets_n600.pt` (16 KB, sha `b34178eb31670dcb…`)
- **Manifest:** `experiments/results/pose_targets/gt_pose_targets_n600.manifest.json`
- stats: min −0.132 / max 35.05 / mean 5.21 / std 11.66 (raw PoseNet 6-vector scale — sane)

**Wired + confirmed:** `witness_applypass_batch.py --dry-run … --pose-target <path>` flips
`low_rank_pose_140` from OWED → `staged` (ready to delegate to `witness_apply_pass.py:_low_rank_pose`).

**Fire command (post-run, when the machine is free):**
```
OMP_NUM_THREADS=1 .venv/bin/python tools/witness_applypass_batch.py \
  --ckpt-dir experiments/results/<final_run> --npz-name levelset_witness_ema_BEST.npz \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --pairs 600 --compose-best \
  --pose-target experiments/results/pose_targets/gt_pose_targets_n600.pt \
  --out-dir experiments/results/applypass_batch_<utc>
```

---

## R3 — MERGE-QUEUE full rehearsal — **PASS** (2 ledger conflicts, resolved mechanically)

Scratch worktree from current `main` (42c2cfd2c7), sequential merge in the boundary order. Per-step:

| # | Branch | ahead/behind | Result |
|---|---|---|---|
| 1 | `claude/p0_518_resume_warmup_geometry` | 4 / 11 | **CLEAN** — auto-merged `tools/launch_witness_run.py`; 8 files |
| 2 | `claude/p0_328_408_merge_window_prep` | 3 / 8 | **CLEAN** — ort auto-merged BOTH the trainer AND `tools/levelset_byte_close_and_eval.py` (the warned proximity zones) |
| 3 | `claude/p0_521_spec_v10_capstone` | 3 / 8 | **CLEAN** — new files only |
| 4 | `codexwt/l7_default_failloud_budget_eventlaw` | 3 / 231 | **2 conflicts** (ledgers only); trainer + all tests auto-merged clean |

**Merge-4 conflicts — pure-additive, resolved by UNION (never wholesale-revert):**

1. `.omx/state/lane_maturity_audit.log` (append-only JSONL): kept ALL 23 HEAD rows + l7's 1 row
   (`l7_default_failloud_budget_eventlaw`, ts 2026-07-15T15:24:58Z) → 6548 rows, every line valid JSON.
2. `.omx/state/lane_registry.json` (lanes array): union BY id. base=1889, ours=1897 (+8 main lanes),
   theirs=1890 (+1 l7 lane), intersection=1889=base → **no lane modified on both sides**. Result =
   **1898 unique lanes**, kept HEAD top-level metadata. `tools/lane_maturity.py validate` → OK.

**Key tests on the final merged tree (main venv, PYTHONPATH → WT copy, WT-src precedence verified):**

- `test_p0_resume_warmup_geometry.py` + `test_clip_profile_rewire_byte_close.py` +
  `test_p0_408_rate_rolling_and_lever_engage_wirein.py` + `test_curriculum_epoch_budget_guard_20260713.py`
  + `test_scorer_throughput_gate.py` → **104 passed, 3 skipped** (21 s).
- Trainer import smoke (`train_levelset_witness_realized_through_R_mlx.py`) → **OK**.
- spec_v10: `compile_v10_capstone_launch_config()` → **fail-closed as designed** (8 blockers, gated
  on `v9c2_completion`); `spec_v10_status()` → clear=False, 8 blockers (usable pre-gate). Correct.

**Mechanical-merge artifacts** in `.omx/research/boundary_merge_rehearsal_20260717/`:
`RESOLVER.md` (union method), `l7_registry_lane_ADD.json`, `l7_audit_row_ADD.jsonl`,
`l7_ledger_resolution.patch`, `l7_merge_diffstat.txt`. Scratch worktree + branch removed; main clean.

**Real-merge note:** the real merge commits THROUGH the pre-commit hook (rehearsal used
`--no-verify` only because a scratch worktree has no `.venv`). Because both ledger conflicts are
pure-additive unions, the real merge is mechanical — re-run the same union; exact bytes drift as
main advances but the method is stable.

---

## Boundary-blocked residue (what remains truly at risk)

- **NONE new** from R1/R3 — the one integration bug found (R1 GT-load-count) is FIXED on main.
- **Pose composition at n600 is the real endgame dependency, not a bug:** the byte-close render is
  pose-blind by construction; the pointer attempt MUST invoke `--pose-carrier` composing the banked
  pose (R1 dxi 7.2 KB) — the tool supports it; verify the banked xi is in the FINAL run dir before
  the attempt.
- **Do the n600 byte-close + applypass fire ONLY with the trainer down** (or free > ~15 GiB) — both
  are post-run by design; peak ~10–12 GiB.
- spec_v10 stays fail-closed until `v9c2_completion` + 7 other gates clear — expected, not a blocker.
