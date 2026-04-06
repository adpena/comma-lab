# PRD: cold-start dual-track verification and first measured progress

## Summary
Take the repo from zero promoted runs to grounded evidence with the smallest reversible execution loop: verify the pinned upstream snapshot, prove Track A (`exact_current`) is still wired under the current workflow, prove Track B (`robust_current`) can still package, and attempt the first scored run only when prerequisites are already present and the upstream copy is explicitly synced.

## RALPLAN-DR summary

### Principles
1. Ground assumptions before tuning.
2. Keep both tracks alive unless evidence forces demotion.
3. Prefer measured progress over elegant code changes.
4. Keep `current_workflow` and `rule_faithful` accounting explicit.
5. Leave the repo resumable from disk.

### Decision drivers
1. The repo currently has no promoted runs; the biggest risk is stale assumptions.
2. Track A can become dead weight quickly if the exploit path no longer works.
3. Track B is the honest optimization lane, but it should start from verified packaging + sync evidence before scoring.

### Viable options

#### Option A — two-stage verification-first cold-start loop (chosen)
- Stage 1: doctor, snapshot verification, explicit upstream sync, Track A single-pass smoke/eval, Track B package smoke, durable-state update.
- Stage 2: conditional first scored run, only if prerequisites are already satisfied and Stage 1 leaves a clean path.
- Pros: smallest reversible path to trustworthy state; matches the repo’s intended first checks; keeps both tracks explicit.
- Cons: may defer the first full score if prerequisites or upstream sync are not ready.

#### Option B — robust-first tuning loop
- Ignore Track A beyond file presence and start x265 sweeps immediately.
- Pros: earlier focus on the honest lane.
- Cons: risks missing a broken current-workflow track; tunes on unverified environment; violates repo guidance priority.

#### Option C — environment/tooling cleanup first
- Normalize scripts/CLI and add more diagnostics before running any evaluator.
- Pros: cleaner foundation.
- Cons: delays first measured progress and adds code churn before evidence.

### Decision
Choose Option A. The repo already includes scripts and starter submissions, and `reports/latest.md` explicitly calls for upstream snapshot verification, exact_current smoke, and robust_current package smoke first. Host prerequisites like `ffmpeg` or upstream Python deps are execution prerequisites, not the success definition of the repo-local cold-start loop. The first scored run should happen only when those prerequisites are already satisfied.

### Alternatives considered / invalidated
- Immediate x265 sweep: invalid until packaging and evaluation are proven live.
- Heavy learned/JAX/CUDA lane: invalid before the x265 floor is measured; only revisit if cheaper lanes stall.

## Scope

### In scope
- Verify live upstream commit and pinned-file digests against `workspace/upstream_snapshot.json`.
- Record environment prerequisites needed for first evaluation.
- Explicitly sync both submission tracks into the upstream checkout before any upstream `evaluate.sh` call.
- Run `exact_current` exactly once in the current cycle to confirm whether the current-workflow path is alive.
- Run `robust_current` packaging and record archive size/config.
- Attempt a first scored run (preferably Track B) only if prerequisites are already present and sync state is clean.
- Update durable state and queue the next 3 experiments with payoff/cost/risk.

### Out of scope
- Editing upstream or exact-current inflator files.
- Broad refactors or new dependencies in repo code.
- CUDA/JAX/Mojo lanes unless the cheap baseline loop stalls.
- Extended Track A debugging if Track B is still unverified.

## Concrete execution steps
1. **Ground the environment and snapshot**
   - Run `comma-lab doctor`.
   - Verify HEAD commit and all pinned file digests match `workspace/upstream_snapshot.json`.
   - Record missing external prerequisites, but do not make host mutation part of plan success.
2. **Sync both tracks into upstream explicitly**
   - Use `comma-lab install-submission exact_current` and `comma-lab install-submission robust_current`, or `scripts/sync_into_upstream.sh`.
   - Record that the upstream evaluator will read the synced copies, not just the repo-local source dirs.
3. **Confirm Track A wiring with one pass**
   - Run upstream `evaluate.sh --submission-dir ./submissions/exact_current --device cpu` once.
   - Record success/failure and archive bytes under `current_workflow`.
   - If it fails materially, demote it in reporting and shift optimization priority to Track B.
4. **Confirm Track B package smoke**
   - Run `submissions/robust_current/compress.sh`.
   - Re-sync `robust_current` into upstream after packaging so `archive.zip` is current.
   - Record archive size and config under both packaging views, clearly marking which bytes are only scored under `current_workflow`.
5. **Attempt the first scored baseline conditionally**
   - If Stage 1 left the evaluator runnable (dependencies present, upstream copy synced), run upstream `evaluate.sh --submission-dir ./submissions/robust_current --device cpu`.
   - If not, record the blocker and stop after updating durable state.
6. **Persist and queue**
   - Update required durable state files.
   - Queue exactly 3 next experiments with expected payoff, cost, and risk.
   - Add/update `docs/speculative_lanes.md` only if execution reveals a clearly attractive but currently unjustified lane.

## Acceptance criteria
1. Snapshot commit and pinned digests are verified and written to disk.
2. `exact_current` has a concrete single-pass run result (success or failure mode) recorded.
3. `robust_current` packaging has a concrete run result recorded.
4. A scored evaluator run is recorded **if and only if** prerequisites were already satisfied; otherwise the blocking prerequisite is recorded explicitly.
5. `.omx/state/current_focus.md`, `.omx/state/next_experiments.md`, `.omx/research/findings.md`, `.ralph/run_log.md`, and `reports/latest.md` are updated.

## Available-agent-types roster
- `executor`: run local implementation-safe execution steps and file updates.
- `architect`: final verification/sign-off on plan fidelity and risk posture.
- `verifier`: cross-check evidence files/commands after execution.
- `debugger`: only if evaluator/package failures require root-cause analysis.

## Staffing guidance
- **Ralph path**: single owner `executor` for steps 1-6, then `architect` sign-off, then optional `verifier` evidence pass.
- **Team path** (not preferred here): lane 1 snapshot/env, lane 2 Track A, lane 3 Track B, then shared verification.

## Reasoning by lane
- Environment/snapshot: medium
- Track A smoke/eval: medium
- Track B package/eval: medium
- Failure debugging: high
- Final verification/sign-off: high

## Verification commands
- `source .venv/bin/activate && comma-lab doctor`
- `python3 <snapshot-check-script>` or equivalent digest comparison
- `source .venv/bin/activate && comma-lab install-submission exact_current && comma-lab install-submission robust_current`
- `source workspace/upstream/comma_video_compression_challenge/.venv/bin/activate && bash scripts/eval_exact_current.sh`
- `bash scripts/package_robust_current.sh`
- `source .venv/bin/activate && comma-lab install-submission robust_current`
- `source workspace/upstream/comma_video_compression_challenge/.venv/bin/activate && bash scripts/eval_robust_current.sh`

## ADR
- **Decision:** Run a two-stage verification-first cold-start loop before tuning.
- **Drivers:** zero promoted runs, dual-track repo contract, explicit first checks in `reports/latest.md`, need to detect broken exploit path early, need explicit upstream sync for both tracks.
- **Alternatives considered:** robust-first tuning; tooling cleanup first; heavy learned/CUDA lane.
- **Why chosen:** smallest reversible path to trustworthy evidence while keeping the first scored run conditional on already-satisfied prerequisites.
- **Consequences:** the first cycle may end with package/smoke evidence but no score if prerequisites are missing; Track A gets only one pass in the cycle.
- **Follow-ups:** x265 floor sweep, resolution sweep, sparse residual side lane after the first honest baseline.
