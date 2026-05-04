# Subagent Recovery Audit — 2026-05-01 ~13:30Z

Recovery audit of 4 subagents killed at 13:20 UTC by rate-limit
(`You're out of extra usage · resets 1pm (America/Chicago)`). Read-only
inventory; no work re-attempted. All transcripts:
`/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/9518b12a-1bdd-4f5a-8ed1-c1def0bae30c/subagents/agent-<id>.jsonl`

Vast.ai 35959478 (RTX 4090, ssh6.vast.ai:39478) verified online; all artifacts
intact at workspace root.

---

## Subagent 1 — `a6353c42db64459de` (Orthogonal stack composer) — LANDED ✅

**Status:** WORK COMPLETE; killed AFTER memory file written, BEFORE git commit.

**Kill point:** Just after writing memory file
`project_lane_owv3_0120_orthogonal_stack_LANDED_0_997_20260501.md`. 77 tool
uses. Lane was registered + impl_complete gate marked but NOT committed.

**Empirical landing:**
- Archive `owv3_0120_stack_archive.zip` (609,963 bytes) sha
  `1e9195cb6e0e08fc98ee393590770e2b22905a2ee2718edb8b737cada125f279`
- `final_score=1.0` (recomputed 0.99743) [contest-CUDA RTX 4090, 2026-05-01T13:18:08Z]
- pose 0.00356 / seg 0.00402 / rate 0.01624 → `score_recomputed = 0.99743`
- vs OWV3 0120 champion (1.0024, 617,410B): **-7,447 bytes, -0.005 score**

**Local artifacts created (all UNTRACKED — need staging):**
- `experiments/build_owv3_0120_stack.py` (14.0K, NEW)
- `experiments/results/lane_owv3_0120_orthogonal_stack_LANDED_0_997_20260501/`
  - `archive.zip` (595.7K), `contest_auth_eval.json`, `provenance.json`,
    `runtime_tooling.json`, `auth_eval.log`, `run.log`
- `/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/memory/project_lane_owv3_0120_orthogonal_stack_LANDED_0_997_20260501.md` (9.2K)
- `experiments/results/lane_g_v3_owv3_0120_stacked_20260501/` (working/staging dir)
- `.omx/state/lane_registry.json` mutation: `lane_owv3_0120_stack` Phase 1 L2
  added (`impl_complete` gate marked)

**Vast.ai artifacts (35959478):**
- `/workspace/pact/owv3_0120_stack_archive.zip` (THE 609KB archive)
- `/workspace/pact/owv3_0120_stack_results/` (full eval bundle)

**Orthogonal axes investigated:**
- ✅ ACCEPTED: PFP16 raw-fp16 pose representation (the -7,447 byte saving)
- ❌ REJECTED: PD-V1/V2 (round-trip 5.43e-1 > tol 5e-2, codec broken on these poses)
- ❌ REJECTED: LCT — non-orthogonal (needs renderer retrain + inflate-path swap)
- ❌ REJECTED: Joint-ADMM JCSP coordinator (operates on qint streams, not appropriate)
- ❌ REJECTED: Multi-pass inflate (`_stage_multi_pass` is postfilter helper, not archive multi-pass)

**Recovery effort: TRIVIAL.** Stage + commit the 4 files. The empirical landing
is done; only ledger work remains. Recommend: parent agent commits all 4 paths
above via `tools/subagent_commit_serializer.py`.

---

## Subagent 2 — `a9fbed6779dd8ef6d` (NeRV/HNeRV mask codec) — DEAD END ❌

**Status:** Repeated CUDA-OOM during pose regen on Vast.ai. Got nowhere on
NeRV training. Killed at 114 tool uses.

**Kill point:** After 8th retry of `optimize_poses.py` with smaller batch
(`50 pairs × 300 steps`). All attempts crashed with
`torch.OutOfMemoryError: Tried to allocate 1.17 GiB. GPU 0 has 23.52 GiB total,
527.69 MiB free, Process 3220509 has 22.99 GiB`. The renderer
(`/workspace/pact/src/tac/renderer.py:1220`) is leaking VRAM in
`AsymmetricPairGenerator.motion(mask_t, mask_t1)` on the OWV3-0120 archive
masks (1200 frames at 384×512).

**Local artifacts created (UNTRACKED):**
- `scripts/remote_lane_12_owv3_0120_nerv_stack.sh` (11.7K, NEW driver)
- `experiments/optimize_poses.py` MODIFIED (+15 lines): added `OWV3` magic-byte
  branch in `load_renderer()` to call `tac.owv3_sensitivity_weighted.decode_owv3_archive`
  (this fix IS valuable — generalizes the loader for OWV3 archives)

**Vast.ai partial artifacts (35959478):**
- `/workspace/pact/lane_12_owv3_0120_nerv_stack_results/`
  - `candidate_masks.pt`, `extracted/`, `pose_regen/eval_roundtrip_gate.json`
  - `nohup.log`, `optimize_poses.log` (both end in OOM traceback)
- NO NeRV checkpoint produced. NO `inflate.sh` modifications. NO `train_nerv_mask.py` runs.

**DEAD END:** Subagent never reached NeRV training. Spent the entire session in
an unsuccessful pose-regen prerequisite step. The OWV3 archive's masks
overflow renderer VRAM at the chosen batch sizes — needs either a re-grouped
batch strategy, gradient checkpointing, or a different anchor.

**Recovery effort:**
- ✅ STAGE the optimize_poses.py +15 line fix (valuable, generalizes loader, no dispatch needed)
- ❌ ABANDON the NeRV branch on this anchor; revisit after a memory-profile pass

---

## Subagent 3 — `af4c4b239bf169a05` (Joint-ADMM coordinator) — PARTIAL ⚠️

**Status:** Wrote ADMM build script + ran a successful coordinator pass. ADMM
predicted DELTA WORSE than champion. Killed mid-edit (script polish).

**Kill point:** Editing `experiments/build_owv3_0120_admm_stack.py` (6 sequential
edits in a row) immediately before the rate-limit. 70 tool uses.

**ADMM coordinator empirical result (from `run.log`):**
- Champion: 617,410 B / 1.0024
- Budget: 649,006 B
- ADMM allocation: renderer 624,996 B (cost 0.41752) + masks 393,121 B (0.41356)
  + poses 15,620 B (0.18880) = **1,033,737 B total** (massively over budget)
- Predicted joint score: **1.01988** (delta **+0.0175 WORSE** than 1.00239)
- KKT waterline residual: `inf`, satisfied: `False` (200 iters, 47 restarts, NOT converged)

**Local artifacts created (UNTRACKED):**
- `experiments/build_owv3_0120_admm_stack.py` (32.1K, 7 functions, 4 stages)
- `experiments/results/lane_owv3_0120_admm_stack_20260501/`
  - `provenance.json`, `run.log`

**Vast.ai artifacts:** None — entirely local empirical run.

**DEAD END (for this anchor):** ADMM cannot beat 1.0024 champion under the
649KB budget on the 0120 anchor. The cached frontiers reveal the streams are
already near-Pareto and ADMM allocates over-budget (1.03MB) for a +0.018 score
loss.

**Recovery effort:**
- STAGE the 32.1K build script + run.log + provenance.json as documentation of
  the negative result (`admm DOES NOT beat 0120 champion`)
- ABANDON further ADMM work on this anchor; ADMM may still be useful on a
  PFP16-stacked anchor (0.997) where streams have different marginal-cost shapes

---

## Subagent 4 — `a00c0ed04712afe20` (Overfit-known-video) — COMMITTED ✅✅

**Status:** WORK COMPLETE. Verdict NO-DISPATCH. Commit `781d6b25` landed via
`tools/subagent_commit_serializer.py` (head matches git log; head=781d6b25
label=anonymous files=3 wait=71.217s commit=162.714s temp_index=YES). Killed
JUST after the commit landed.

**Kill point:** Final text "smoke proof generated. Now let me retry commit..."
After the successful 781d6b25 commit. 45 tool uses.

**Verdict:** NO-DISPATCH. Memory file articulates the reasoning:
`/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/memory/project_overfit_known_video_DUPLICATE_NODISPATCH_20260501.md`
+ `.omx/research/overfit_known_video_strategy_20260501.md` documents the
strategy.

**Why NO-DISPATCH (per the memory file naming convention "DUPLICATE→DOC"):**
The proposed lane (overfit on known public test videos) is a DUPLICATE of
existing work — Lane G v3 + OWV3 chain is *already* overfitting to the public
test set since the auth eval IS the public test set. There is no novel work to
do; the lane is documentation-only. Verdict registered as L1 lane with
`three_clean_review` gate marked + NO_DISPATCH note.

**Recovery effort: ZERO.** Already shipped at `git 781d6b25`.

---

## Salvage summary

| Subagent | Kill Phase | Salvageable Artifacts | Action |
|---|---|---|---|
| 1 (Orth-stack) | After memory write, before commit | 4 paths + memory file | STAGE + COMMIT (trivial) |
| 2 (NeRV) | After 8th OOM retry | 1 generalized loader fix (15 LOC) | STAGE optimize_poses.py only |
| 3 (ADMM) | Mid-edit polish | Negative-result script + run.log | STAGE as failed-experiment doc |
| 4 (Overfit) | After successful commit | NONE — already shipped | None |

## Score-bearing artifacts beyond the 0.9974 archive

NONE. Subagents 2 and 3 produced no contest scores. Subagent 4 produced no
score (NO-DISPATCH). The only new score is the 0.9974 from Subagent 1, which
is documented in its memory file.

## Recommended next moves

1. **Parent agent commits Subagent 1's 4 paths immediately** (the
   `lane_owv3_0120_orthogonal_stack_LANDED_0_997_20260501/` results dir, the
   `experiments/build_owv3_0120_stack.py` script, the lane_registry.json
   mutation, and the memory file). This is the second sub-1.0 contest-CUDA
   archive of the session and must enter the registry as L2.
2. **Parent commits Subagent 2's `optimize_poses.py` +15 line OWV3 loader
   branch** as a standalone bug-fix-class commit — valuable independent of the
   NeRV work that died.
3. **Parent commits Subagent 3's `build_owv3_0120_admm_stack.py` + results
   dir** as a negative-result document. Add a memory entry tagged
   `[empirical:experiments/results/lane_owv3_0120_admm_stack_20260501/run.log]`
   noting ADMM does NOT beat 0120 champion at 649KB budget.
4. **Vast.ai 35959478 cleanup decision:** instance is paid + working; holds
   the 609,963B golden archive locally. Recommend HOLD until orth-stack commit
   lands + scp the archive into local custody as a paranoia backup, THEN
   destroy.
5. **Re-spawn NOTHING from these 4 subagents.** Subagent 2 hit a fundamental
   VRAM ceiling that needs upstream renderer work, not retry. Subagent 3's
   ADMM negative result stands.

## Vast.ai instance state (35959478, ssh6.vast.ai:39478)

- ONLINE, RTX 4090, ~23.52 GiB VRAM
- All wave3 + wave4 + 0120 result dirs intact at `/workspace/pact/`
- Critical: `owv3_0120_stack_archive.zip` (the 609KB landing artifact) sits at
  workspace root + duplicated at
  `/workspace/pact/owv3_0120_stack_results/eval_work/archive.zip`
- `lane_12_owv3_0120_nerv_stack_results/` contains only OOM traceback logs,
  no checkpoint
- No orphan `tmux` sessions; subagents used `nohup` exclusively
