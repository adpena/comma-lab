# AGENTS

You are operating inside a dual-track lab for the comma video compression challenge.

Read `PROGRAM.md` before making changes.

## Race-mode rigor inversion + parallel-dispatch first — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Postmortem source:** `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_may_4_hnerv_race_postmortem_20260505.md`. The May 4 2026 contest was decided in a **4 hour 8 minute race window** after PR #95 (HNeRV root) was published at 07:47:15 UTC. Final top 3 (PR #101 / #103 / #102 at 0.193 / 0.195 / 0.195) all submitted between 11:50 and 11:55 UTC. Silver medal (rem2 PR #103) was **241 lines of code in 2 files**. Our PR #107 apogee landed at 0.229 (~11th) — we had every primitive needed but spent the race window building meta-Lagrangian + predictor + sanity gates (sequential validation harness) instead of fanning out parallel dispatches.

The two rules below structurally extinct that failure mode.

### Rule 1: parallel-dispatch is a FIRST-CLASS DELIVERABLE, not an afterthought

When the user says "parallel" / "high-throughput" / "search" / "automation" / "stacking sweep" — the **first file built** must be the actuator that fans out N concurrent paid-GPU dispatches. Build the actuator BEFORE the ranker, BEFORE the predictor, BEFORE the sanity gate. A ranker without a parallel actuator is a planner that produces ranked plans no one executes. The canonical actuator + harvest loop is now checked in:

- `tools/parallel_dispatch_top_k.py` — `concurrent.futures.ThreadPoolExecutor` over `tools/lightning_dispatch_pr106_stack.py` (Lightning T4) or `scripts/launch_lane_on_vastai.py` (Vast.ai 4090). Includes per-dispatch + total-cost gating, per-dispatch timeout, harvested-JSONL output.
- `tools/harvest_and_reseed.py` — ingests the harvested JSONL, drops any row not tagged `[contest-CUDA]`, appends new empirical anchors to `.omx/calibration/anchors_*.json`. Closes the prediction → empirical → updated-prediction feedback loop.

The canonical sweep loop is therefore three commands:
```bash
.venv/bin/python tools/meta_lagrangian_search_cli.py \
    --lane-class apogee_intN --auto-sweep-bits 4,5,6,7,8 \
    --top-k 16 --output reports/sweep_ranked.json

.venv/bin/python tools/parallel_dispatch_top_k.py \
    --ranked-input reports/sweep_ranked.json --max-concurrency 16 \
    --provider lightning --estimated-cost-per-dispatch 0.11 \
    --max-total-cost 5.00 --harvest-output reports/sweep_harvested.jsonl

.venv/bin/python tools/harvest_and_reseed.py \
    --harvested-jsonl reports/sweep_harvested.jsonl \
    --anchors-path .omx/calibration/anchors_apogee_intN.json
```

If any future work proposes a "search engine" / "candidate generator" / "ranking primitive" / "sanity gate" without naming the parallel actuator, that work is INCOMPLETE. Closure requires naming the actuator that turns the ranking into N concurrent dispatches.

### Rule 2: strategic-rigor inversion at leaderboard moves

The agent's default prior is "max rigor: validate before dispatching." This prior is correct PRE-leader-shift (you're optimizing alone, every wasted dispatch is your money). It is WRONG POST-leader-shift (you're racing; every minute of gating is a competitor shipping ahead of you).

**Detection:** before any new candidate is dispatched, the agent MUST check whether the public leaderboard has moved in the last 24 hours. If yes, the prior **inverts**: the next action is the smallest credible bolt-on submitted within ~60 minutes, not another sanity gate. Concretely, the May 4 race showed:
- BradyMeighan: PR #97 (0.23) → #99 (0.197) → #100 (0.195) — **3 PRs in 2h 12min**
- rem2 (silver): PR #96 (0.21) → #103 (0.195) — **2 PRs in 3h 24min, 241 LOC final increment**
- EthanYangTW (bronze): PR #98 (0.196) → #102 (0.195) — **2 PRs in 2h 23min**
- "kitchen_sink" PR #105 — 1776 LOC, 21 files — **lost** to rem2's 241 LOC

The right move when N hours remain is the smallest credible bolt-on, not the most thorough system. Public PRs are checkpoints that lock in score, force honest contest-CUDA measurement, surface what's NOT working in 30 min instead of 4 hr, and establish presence. Holding for one polished shipment is the failure mode.

### Rule 3: cron + same-prompt loop is a fan-out cadence

A cron job firing every N minutes on the same prompt is the natural cadence for "fan-out K candidates and harvest." It is the WRONG cadence for "do another sequential validation pass." The May 4 cron was firing every 5 minutes and the agent kept choosing depth (more validation gates) over velocity (more dispatches). When loop tick X is shorter than dispatch wall-clock Y, every tick should fan out a new batch. When Y > X (12-hour training jobs), the tick should be a "harvest results + reseed" cycle.

Translation table when the loop fires:
- "push toward Shannon floor in absolute minimum wall clock" → fan out next batch of K parallel dispatches; do NOT add another gate.
- "extreme rigor" → applies to the FIRST cycle (calibration of anchors), not every cycle. After cycle 1, rigor is in the actuator's gating thresholds, not in sequential validation.
- "fix all bugs everywhere" → fix the bug class that prevents the actuator from firing. A bug that doesn't block dispatch can wait.

### Concrete enforcement

- Task #309 ("HIGH-THROUGHPUT DISPATCH agent: parallel GPU orchestration") sat **pending** the entire May 4 race window. That task class is a NON-NEGOTIABLE priority-1 — if a similar task exists in any backlog, it must be claimed before any new validation-gate work.
- Any PR that adds a new ranking/predictor/sanity-gate primitive MUST link to the parallel-actuator file that consumes its output. If no such consumer exists, the PR description must explicitly state "actuator deferred — no race window currently active" with the operator's signoff on the deferral.

## Main branch source of truth — NON-NEGOTIABLE

`main` is the sole source-of-truth branch. Do not do production work, recovery
work, public-frontier intake, or contest-custody edits on any other branch.
Detached public PR clones, stashes, quarantine trees, provider workspaces, and
subagent forks are forensic inputs only; promoted code, docs, artifacts, and
ledgers must land back on `main` after explicit review.

## Frontier target — NON-NEGOTIABLE, HIGHEST EMPHASIS

The target is the best contest-faithful public frontier, not an obsolete
absolute threshold. During an active contest, deadline, or replay window, any
public PR/archive/body/comment/release that plausibly beats the local exact
A++ frontier takes priority over saturated local polish. Claimed public scores
remain `external` until exact CUDA replay, but they must enter intake and exact
replay immediately.

Every frontier action must produce or advance a concrete artifact: candidate
archive, bit-level intake record, dispatch claim, queued exact eval, harvested
JSON, compliance packet, preflight guard, or release/report update. Grand
council and strategy text are advisory only unless they change the next build,
guard, replay, or dispatch.

Deadline mode requires submission escrow: keep a sanitized current-best packet
ready, submit the best exact A++ archive before operator sleep or hard deadline
risk, then update with better replays if they land. Do not wait for the perfect
future candidate when a valid current frontier can be disclosed now.

## CROSS-AGENT DISPATCH COORDINATION — NON-NEGOTIABLE

**Before dispatching ANY training, eval, or remote-GPU job, claim the lane with `tools/claim_lane_dispatch.py claim ...`.** The helper owns the file lock, reads `.omx/state/active_lane_dispatch_claims.md`, inserts the newest row at the top, and refuses active same-`lane_id` conflicts inside the 24-hour TTL unless an explicit force flag with notes is used.

If you find an active conflicting claim:
- Do NOT dispatch
- Coordinate via the file's notes column or pick a different lane

When your dispatch completes (success or fail): append a terminal row with the
same `lane_id` and `instance/job_id` via `tools/claim_lane_dispatch.py claim
--force --status completed_...`, `--status failed_...`,
`--status stopped_...`, `--status refused_dispatch...`, or a precise
`--status stale_superseded...` row. Do not leave completed jobs as phantom
active claims.

This rule exists because 2026-05-01 ~23:50 UTC the user reported a possible Q-FAITHFUL dispatch conflict between Claude (H100 SXM via Vast.ai) and codex (Lightning) — no formal cross-agent coordination existed and we may have burned $5-10 of duplicate GPU spend. Level 2 is now the norm: use the helper script and strict submitter checks, not manual table edits except for emergency recovery.

## Operator gates must be wired and used — NON-NEGOTIABLE

Recovered tools are not done when the source file exists. They must be wired
into normal operator flows and documented where future agents will actually see
them.

Required gates:

- `tac.preflight.preflight_all()` includes
  `check_dispatch_cli_shell_hazards(strict=True)`.
- `tools/all_lanes_preflight.py` runs
  `tools/check_dispatch_cli_shell_hazards.py --strict` before lane dry-runs.
- Before any judge-facing or public submission packet, run
  `scripts/pre_submission_compliance_check.py --contest-final --strict` with
  explicit `--expected-archive-sha256`, `--expected-archive-size-bytes`, the
  canonical auth-eval JSON, archive manifest, and any dispatch-claim linkage.

These gates close concrete bug classes: `--rmote` and other dead/typo flags,
adjudicator-only flags passed to Lightning launchers, zsh `path` mutation,
GNU-only `find -printf` in local/macOS surfaces, unsafe ZIP names, stale
archive manifests, CPU/MPS promotion leakage, missing runtime-tree custody,
and public supplement provider/private-path leaks.

If you create a new profiler, packer, recovery script, hidden-gem tool, or
submission checker, wire it into `preflight_all()`, `tools/all_lanes_preflight.py`,
`tools/operator_briefing.py`, a runbook, or a dated `.omx/research/` ledger.
Do not leave high-signal tooling buried under an obscure filename.

## `tac` stays clean; comma-lab owns research state

`tac` is the reusable codec/runtime library. Put real reusable Python
implementation there: codec primitives, archive grammars, payload parsers,
scorer/eval contracts, byte profilers, planning primitives, visualization
primitives, and contest-relevant algorithms. Thin CLIs may live in
`experiments/`, `scripts/`, or `tools/`, but they should delegate to `tac`
modules instead of burying implementation in ad hoc entry points.

Do not add Claude/OMX/provider/recovery policy to `tac` unless it is truly
reusable codec, contest-runtime, or contest-preflight logic. Checks that protect
archive validity, inflate/runtime compliance, CUDA-score custody, and package
safety are canonical in `src/tac/preflight.py`. Put research-state custody,
public-frontier intake, hosted supplement builds, provider ledgers, and recovery
audits in `src/comma_lab/`, `tools/`, `docs/`, and `.omx/`.

Use `reverse_engineering/` for clean public-submission deconstruction: curated
runbooks, bit-level anatomy notes, adapter boundaries, and small manifests.
Keep raw PR clones, archives, provider transcripts, and large generated
artifacts in ignored custody locations with ledger links. Reusable parsers and
planners still belong in `tac`.

Track small durable `.omx/research` ledgers and small structured summaries.
Do not track raw `.omx/state/*.json`, provider transcripts, auto-memory
snapshots, generated public-site bundles, `reports/raw`, `reports/private`, or
large rebuildable artifacts. Canonicalize interesting ignored state into dated
`.omx/research` ledgers or `docs/paper/ara`, and host large canonical artifacts
externally with a committed manifest.

Use `python tools/audit_research_state_tracking.py --repo-root .` before
release or cleanup. Its implementation lives in `src/comma_lab/research_state.py`
on purpose. `src/comma_lab/preflight/strict_checks.py` is only an adapter/catalog
surface for ARA, reports, hosted supplements, and dashboards.

## Public frontier watch and intake — NON-NEGOTIABLE

During active contest or replay windows, refresh public PRs and official
leaderboard state frequently enough that late submissions are not missed while
internal lanes run. For any public target that can beat the local exact
frontier, immediately collect PR number, title, author, URL, head SHA,
created/updated time, archive URL, bytes, SHA-256, member names, source
runtime, dependencies, claimed components, recomputed public score, compliance
risks, and fastest exact-replay path. Use detached clones or artifact
directories; do not checkout public PRs into the dirty shared worktree.

If a lower public claim appears, the default order is:

1. Download archive and source.
2. Build bit-level anatomy and compliance-risk record.
3. Claim replay lane.
4. Queue exact CUDA eval on T4/equivalent or fastest available faithful path.
5. Harvest JSON, adjudicate, then build/update submission packet.

Council review cannot block steps 1-4 for a public lower-score replay unless
there is a specific contest-compliance violation that would make the replay
invalid.

## Bit-level deconstruction and entropy discipline

For archive/packer work, inspect bytes before arguing from prose. Record ZIP
header parity, member order, compression method, sizes, CRCs, duplicate names,
magic, section offsets, length prefixes, section hashes, entropy estimates,
decoded tensor shapes, side channels, and no-op/provenance detection.

Arithmetic coding, range coding, ANS/Huffman-style coders, brotli/zstd/lzma
transforms, tensor grouping, histogram overhead, fixed-section removal, and
deterministic pack ordering are first-class score lanes. If a dense stream
remains in a generic compressor, estimate entropy and test a real coded payload
before declaring the area saturated.

## FORBIDDEN PATTERNS — NON-NEGOTIABLE, READ BEFORE WRITING ANY CODE

These are exact code patterns I have written multiple times despite the rules below saying not to. They are FORBIDDEN at the typing moment. If a default would land here, refuse it before typing — do not "fix it on review."

**Forbidden device-selection defaults (the MPS-fallback trap):**
```python
device = "cuda" if torch.cuda.is_available() else "mps" if ... else "cpu"  # FORBIDDEN
device = torch.device(env.get("DEVICE", "cuda" if cuda.is_available() else "mps"))  # FORBIDDEN
```
Correct: default to CUDA-REQUIRED. Raise on no-CUDA. Provide explicit `--device cpu` opt-in with a banner that the bytes/score will differ. (See `feedback_default_to_convenience_trap`.)

**Forbidden CLI flag inventions (the dead-flag trap):**
Adding `--auth-eval-masks` to a subprocess call without `grep "add_argument" target.py` first. Inventing flag names from intent is FORBIDDEN. Always grep the target's argparse before emitting any flag. (See `feedback_dead_flag_wiring_pattern`.)

**Forbidden silent-skip cascades (the bootstrap trap):**
Writing `set -uo pipefail` (no `-e`). Calling `zip` shell binary instead of python `zipfile.ZipFile`. Passing empty captured variable to argparse. (See `feedback_zip_dep_bootstrap_trap`.)

**Forbidden score claims:**
Reporting any score that did not come from contest-CUDA `upstream/evaluate.py` on the EXACT archive bytes that will be submitted. No proxy MSE. No MPS. No "looks reasonable" extrapolation. Tag every reported score `[contest-CUDA]` or `[advisory only]`. (See `feedback_proxy_auth_math_useless`, `feedback_mps_cuda_drift_critical`.)

**Forbidden component-aliasing for baselines:**
Treating a directory of components as the "baseline" without verifying every file SHA against the archive ZIP that produced the baseline score. Components from different lanes leak into the same dir; SHA-vs-archive is the only check. (See `feedback_phantom_baseline_pattern`.)

**Enforcement:** Before typing ANY of the above patterns, STOP. The non-negotiable wins over the convenient default. This list is in CLAUDE.md so it is loaded into context at session start; if I write one of these patterns anyway, that is a process failure, not an information failure.

**Forbidden empirical-claim-without-evidence-tag (the docstring-overstatement trap):**
Writing "saves 49%" / "improves N%" / "beats baseline" / "verified" in a docstring/report/script without an adjacent `[empirical:<artifact path>]` or `[contest-CUDA]` or `[prediction]` tag. Lane PD docstring stated 49% savings; empirical regression test caught actual 18.5%. The 49% was a derivation, not a measurement. Tag every claim. (See `feedback_three_active_bug_classes_needing_strict_checks_20260429.md`.)

**Forbidden fix-lands-in-helper-but-not-callsite (the dangling-helper trap):**
Adding a kwarg to a helper without grepping for callers and updating each. Lane GP added `baseline_poses=` to `reconstruct_poses()` but the actual call at `experiments/fit_pose_gp.py:33` never passed it for ~2 weeks. After adding any kwarg with non-trivial semantics, register it in `CALLSITE_CONTRACTS` and run the AST scanner to enforce all callers pass it. (See `feedback_three_active_bug_classes_needing_strict_checks_20260429.md`.)

**Forbidden MPS-derived strategic decision (the MPS-falsification trap):**
Writing "GREEN" / "RED" / "KILL" / "promoted" / "FALSIFIED" in any record where the supporting evidence is an MPS or CPU forward pass through SegNet/PoseNet/renderer/distilled scorer. MPS PoseNet drift is 23×; SegNet 2×; score 2.5×. STC clean-source FALSIFICATION was made on MPS encoder; user correctly objected; withdrawn. Strategic decisions REQUIRE a `[contest-CUDA]` artifact in the same record/section. (See `feedback_no_local_mps_for_authoritative_kill_or_promote_20260429.md`.)

**Forbidden /tmp paths in any persisted artifact (the transient-evidence trap):**
Writing `/tmp/<anything>` as a file path in: lane registry evidence strings, memory files, dispatch claims, commit messages, build metadata, dispatch scripts, runbooks, CLAUDE.md examples, or any other artifact that survives the current shell session. /tmp paths do NOT survive a fresh checkout, do NOT exist on remote/CI/cloud machines, and CANNOT be verified by other agents. They produce phantom "evidence" that points at nothing. User mandate 2026-05-05: "we need to stop using /tmp by principle". Forensic finding: `lane_pr106_stacked` was marked L2 with `real_archive_empirical:true` evidence pointing at `/tmp/pr106_stacked_smoke/stacked_full/pr106_stacked_archive.zip` — a path that doesn't exist on any other machine and would be lost on shell exit. **Canonical replacement**: `experiments/results/<lane_id>_<timestamp>/` for build artifacts; `.omx/state/` for ledgers; `.omx/research/` for durable analyses. Caught by `tools/check_lane_smoke_signal_nontrivial.py` (PCC9, transient_tmp_evidence detection).

**Forbidden premature KILL without research exhaustion (the kill-too-fast trap):**
Writing "KILL" / "FALSIFIED" / "DEAD" / "RETIRED" as a final verdict on a lane based on a SINGLE empirical configuration's failure, when plausible alternative configurations have NOT been attempted. apogee_int4 NAIVE-PTQ falsification 2026-05-05 was initially recorded as KILL/FALSIFIED at score 1.4287 [contest-CUDA T4] WITHOUT trying QAT, LSQ, per-channel scaling, smaller block sizes, or outlier handling — all canonical fixes for low-bit PTQ collapse. User caught it: "we must only kill as a last resort after exhausting all research and grand council consensus" (2026-05-05). Default verdict for one-config failure is **DEFERRED-pending-research**, NOT KILLED. KILL conversion requires (a) every plausible alternative config attempted empirically, (b) **grand council CONSENSUS** (not just majority — every inner-ten member endorses), (c) reactivation criteria documented. Memo filename uses `_DEFERRED_pending_<reason>_<date>.md`, not `_killed_*.md`. See expanded "KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE" section above for full enforcement.

**Forbidden re-implementing remote bootstrap inline (the duplicated-bootstrap trap):**
Writing `curl -LsSf https://astral.sh/uv/install.sh | sh` or `apt-get install ffmpeg` or `find upstream -name '._*' -delete` directly in any new chain driver / lane script / one-off. There is ONE canonical bootstrap function: `bootstrap_runtime_deps()` in `scripts/remote_archive_only_eval.sh`, which delegates uv install to `scripts/ensure_remote_uv.sh`. Any new remote script MUST call that wrapper or `source` its bootstrap function — NEVER copy-paste the install commands. Cost of NOT doing this (2026-05-01 loop session): 6 sequential bug-class re-discoveries on 4 destroyed Vast.ai instances burning ~$1.50 + 30 min wall-clock chasing the same lesson. Memory: `feedback_remote_archive_only_eval_self_bootstraps_all_deps_20260501.md`. (Sister rule for venv: `python -m ensurepip --upgrade` is the standard fix for "venv exists but no pip" — see `scripts/ensure_remote_uv.sh` style.)

**Forbidden uv torch install without driver-version pin (the cu13-vs-cu124 trap):**
`uv run --with torch==2.5.1` (no local-version suffix) defaults to the LATEST CUDA wheel from PyPI (currently `+cu13`). On a Vast.ai host with NVIDIA driver < 580 (CUDA 12.x), the cu13 wheel will FAIL `torch.cuda.init()` and silently fall back to CPU — every score becomes `[advisory only]` per the MPS-falsification rule. The canonical pattern in `scripts/remote_archive_only_eval.sh:88-95` auto-pins:
- `driver_major < 580` (CUDA 12.x): `INFLATE_TORCH_SPEC=torch==2.5.1+cu124` + `UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124` + `UV_INDEX_STRATEGY=unsafe-best-match`.
- `driver_major >= 580` (CUDA 13.x): `INFLATE_TORCH_SPEC=torch==2.11.0` (default cu13 wheel works).

NEVER write `--with torch` (unpinned) in any new script. Always export `INFLATE_TORCH_SPEC` first or call the canonical wrapper. Cost (2026-05-01): instance 35957332 silently ran inflate on CPU for 5 min before detection. Memory: `feedback_vast_cuda_driver_too_old_silent_cpu_fallback_20260501.md`.

**Forbidden Vast.ai create without disk + cuda_vers gate (the chain-killer trap):**
Calling `vastai create instance ... --disk 30` for any chain that runs >1 candidate. The contest_auth_eval pipeline writes 3.6GB of inflated raw frames per candidate; a 6-candidate chain needs ≥30GB working set (5GB uv torch cache + 4GB rolling). 30GB hits the wall on candidate 4. Canonical defaults: `--disk 60`, `cuda_vers>=12.4` in the search filter, `--label <unique>`, register to `.omx/state/vastai_active_instances.json`. The chain driver MUST `rm -rf eval_work/{inflated,extracted,archive.zip}` after each successful eval — the canonical pattern is in `experiments/results/lane_g_v3_owv3_wave3_refinement_20260501/wave3_chain_driver.sh`. Memory: `feedback_remote_archive_only_eval_self_bootstraps_all_deps_20260501.md` outstanding-gaps section.

## NEVER invent CLI flags — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Before wiring any flag into `subprocess.run([...])`, READ the target tool's actual `parser.add_argument(...)` list.** Don't invent flag names from intent. Don't trust prior code that "looked like it worked." Verify against argparse. The cost: 30 seconds of `grep "add_argument" target.py`. The cost of NOT doing it: days of wasted GPU + a council review chain that misses the dead-flag bug across multiple rounds.

This rule exists because (2026-04-26 incident, see `feedback_dead_flag_wiring_pattern`):
- R1 wiring of `train_renderer.py --auth-eval-on-best` invented `--auth-eval-masks` for `auth_eval_renderer.py` which has NO such flag.
- R2 "fix" didn't catch the dead flag — focused on rate ambiguity.
- R3 finally caught it (Council R3-1).
- Every chain that "passed" auth-eval-on-best was actually silently skipping it.

**How to apply:**
1. Before adding a flag to a subprocess invocation, `grep "add_argument" path/to/target.py` and confirm every flag name you're emitting exists.
2. Add a regression test that introspects the target's argparse and asserts your call-site flag set is a subset (template: `test_train_renderer_auth_eval_wiring.py`).
3. Fail loud (raise / non-zero exit), not silent (WARN-and-skip), when required inputs to a subprocess wrapper are missing.
4. **It is unacceptable to learn the same lesson twice.** Capture the meta-pattern in memory + CLAUDE.md the FIRST time it bites.

## Modal `.spawn()` HARVEST OR LOSE — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Modal `.spawn()` puts artifacts in the FunctionCall return-value cache (~24h TTL), NOT in a Volume.** `experiments/modal_train_lane.py` uses `.spawn()` exclusively. The local-side dispatcher writes `experiments/results/lane_<label>_modal/modal_metadata.json` (with `call_id`) and exits — it does NOT poll for the result. NOTHING is written to a Modal Volume by this path.

**The investigation trap I fell into 2026-04-29 PM**:
- `modal app list` only shows currently-active apps. Terminated ephemeral apps disappear quickly.
- `modal app logs <app>` shows the most recent log buffer; earlier successful runs aged out.
- `modal volume ls` shows nothing because spawn() doesn't write volumes.
- I wrongly concluded "$0 wasted, all dead" when in fact the dashboard showed $38.80 spent on `modal_train_lane.run_lane_training_t4/a10g` and 31 of 37 dispatched call_ids had artifacts sitting in the result cache about to GC.

**The truth source is per-call**:
```bash
.venv/bin/python -c "import modal; r = modal.functions.FunctionCall.from_id('fc-...').get(timeout=2); print(r.get('returncode'), r.get('elapsed_seconds'), len(r.get('artifacts', {})))"
```
Browser dashboard: https://modal.com/usage (the only source-of-truth for actual billing).

**Rules**:
1. Every dispatch via `modal_train_lane.py` MUST be followed by a scheduled harvest within 24h. Reference: `tools/harvest_modal_calls.py` (formerly `/tmp/harvest_modal_calls.py`) iterates every `experiments/results/lane_*_modal/modal_metadata.json` and writes `harvested_artifacts/` next to each.
2. **NEVER** claim "Modal apps are dead, no artifacts" without first running the harvester. The harvester is the source of truth, not `modal app list`.
3. **A10G has 22GB shared**; SC++/SA-class lanes that allocate 21+GB will OOM (today's incident: lane_sc_plus_plus_v3 crashed at 140s with `CUDA out of memory. Tried to allocate 21.09 GiB`). For OOM-prone lanes, use Vast.ai 4090 (24GB dedicated, $0.26/hr) instead.
4. **Modal scheduling can take HOURS** for T4 / A10G during shortages. "waiting to be scheduled" means $0 charged in the wait, but moment-of-schedule starts the meter. Cancel queued functions you no longer want.
5. Future improvement (issue tracker): change `modal_train_lane.py` to also write artifacts to a Modal Volume so they persist past the result-cache TTL. The `.spawn()` pattern was added for "detached" runs, but the price is orphaned artifacts. Detached + persistent storage requires a Volume, not the result cache.

Memory: `feedback_modal_spawn_result_cache_pattern_20260429.md` documents the full incident + harvest pattern.

## Auth eval EVERYWHERE — NON-NEGOTIABLE, HIGHEST EMPHASIS

**EVERY chained experiment MUST end with a CUDA auth eval against its best checkpoint.** Tracking only proxy `fp4_scorer` / `pose_loss` / training-loss is a WASTED run unless an authoritative score lands at the end. The proxy-auth gap can be 100-350x even on CUDA-CUDA (LANE-B 2026-04-26: proxy 0.0007 → auth 0.246, 350x). The proxy is a TRAINING SIGNAL, not a measurement.

This applies to:
- `experiments/pipeline.py compress` (HAS step_eval at end ✓)
- `scripts/remote_train_bootstrap.sh` (HAS Stage 5 auth eval ✓)
- `scripts/remote_pose_tto_bootstrap.sh` (HAS Stage 4 auth eval ✓)
- `scripts/remote_pose_tto_only_bootstrap.sh` (HAS Stage 4 auth eval ✓ as of 2026-04-26)
- `src/tac/experiments/train_renderer.py` — **GAP: NO auth eval on best.** Must be added: when a `*BEST*` checkpoint is saved, run a background CUDA auth eval and log the result alongside the proxy.
- ANY new training script, TTO loop, postfilter, or experiment runner.

**Pre-launch checklist (mandatory):**
1. Does the experiment end with `auth_eval_renderer.py` on the best checkpoint?
2. Is the auth eval result captured (RESULT_JSON or .json file) and surfaced to the operator?
3. If a chain has multiple "best" candidates (e.g., proxy-best, kl-best, hinge-best), does each get an auth eval?

**Pose TTO specifically:** the TTO loop MUST run a smoke auth eval at step 100 (and every 200 steps after) so the proxy-auth gap is detected within $0.50 of GPU spend, not at $5+ end-of-run.

**The authoritative measurement loop is:** contest-CUDA `inflate.sh` → `upstream/evaluate.py` on the EXACT archive bytes. Nothing else counts. Memory: `feedback_proxy_auth_math_useless`.

## eval_roundtrip — NON-NEGOTIABLE, HIGHEST EMPHASIS

**EVERY training path MUST use eval_roundtrip.** There are ZERO exceptions. This includes:
- train_distill.py (has it)
- training.py Trainer (NOW has it, eval_roundtrip=True by default)
- constrained_gen.py (has it)
- optimize_poses.py (has it)
- qat_finetune.py (has it)
- ANY new training script or optimization

Without eval_roundtrip, proxy-auth gap is 2-6x on PoseNet. Every training run without it is a WASTED run. This mistake has been made on EVERY component in this project. It stops now.

## EMA — NON-NEGOTIABLE, HIGHEST EMPHASIS

**EVERY training path MUST instantiate EMA, update it after every `optimizer.step()`, and save the EMA shadow (not the live weights) as the inference checkpoint.** There are ZERO exceptions for any path that produces a checkpoint that ships in the submission archive.

This includes:
- Renderer training (`train_renderer.py`, `train_renderer_fridrich.py`, `train_distill.py`) — already correct
- SegMap training (`train_segmap.py`, `train_segmap_film_canvas.py`) — already correct
- Joint pair training (`train_joint_pair.py`) — fixed 2026-04-29 PM (duplicate `class EMA` removed; default 0.9995 → 0.997)
- Szabolcs / Selfcomp clones (`train_szabolcs.py`) — wired 2026-04-29 PM (Council D)
- QAT (`qat_finetune.py`, `qat_omega_lagrangian.py`, `quantize_distilled.py`) — wired 2026-04-29 PM (Council D)
- IMP cycles (`train_imp_cycle.py`) — wired 2026-04-29 PM (Council D)
- LoRA TTO (`train_lora_tto.py`) — wired 2026-04-29 PM (Council D)
- Postfilter training (`train_postfilter_on_renderer.py`) — wired 2026-04-29 PM (Council D)
- Codebook EMA in VQ-VAE / LCT mechanisms (van den Oord persistent N_c/m_c form) — already correct
- ANY new training script or optimization

**Quantizr decay = 0.997.** All weight EMAs default to `decay=0.997`. The CANONICAL `class EMA` is `tac.training.EMA` (with the float-buffer guard at L359-364 and the late-bound module guard at L356-358). Codebook EMAs (van den Oord persistent buffer form, e.g. `LearnableClassTargets`, `vqvae_codec`) keep their own 0.99 default — codebooks adapt faster than weights by design.

**Apply only at eval time, with snapshot+restore.** The canonical pattern (copied from `experiments/train_distill.py`):

```python
orig_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
ema.apply(model)
try:
    score = evaluate(model, ...)
finally:
    model.load_state_dict(orig_state)
    model.train()
```

**NEVER call `ema.apply(model)` inside `train_epoch`** without snapshot+restore — that shadows the live weights to the EMA snapshot and freezes learning (the DARTS-S freeze symptom class, even though that specific freeze was a different bug — bare `.round()` zero-gradient at `src/tac/segmap_renderer.py:281`; see Council D audit §6).

**Inference / archive bytes come from `ema.state_dict()`** — never from `model.state_dict()` after training. The Quantizr 0.33 archive is the EMA shadow, not the live final-epoch weights. Selfcomp's 0.38 archive is the EMA shadow. Lane G v3 (1.05) used EMA correctly.

**Without EMA, single-epoch noise dominates the final checkpoint.** Every training run without EMA is a wasted run. This stops now.

Cross-references: Council D audit `.omx/research/council_ema_audit_20260429.md`; preflight Check 88 `check_training_paths_use_ema_correctly` (STRICT @ 0 violations); Lane G v3 reference `project_lane_g_v3_landed_1_05_20260428.md`; Lane MM-V2 falsification `project_lane_mm_v2_landed_2_63_falsified_20260429.md` (which the same audit's §6 freeze investigation should be re-checked against once the V3-clean retrain lands).

## MPS auth eval is NOISE — NON-NEGOTIABLE, HIGHEST EMPHASIS

**LOCAL MPS IS NEVER TO BE USED FOR STRATEGY, PLANNING, OR ANALYSIS.** Verified 2026-04-25 with side-by-side gating measurement on the same pinned archive:

| Metric | Local MPS | CUDA A100 (contest scorer) | Drift |
|---|---|---|---|
| PoseNet distortion | 0.245 | **0.0107** | **23x WORSE on MPS** |
| SegNet distortion | 0.0024 | 0.00116 | 2x WORSE on MPS |
| **Final score** | **2.26** | **0.90** | **2.5x WORSE on MPS** |

PoseNet specifically drifts 23×. Likely cause: FastViT-T12 attention softmax + YUV6 chroma plane numerics differ between MPS and CUDA float16 implementations.

**Rules:**
1. ALL auth eval must run on CUDA (Vast.ai 4090, A100, T4). Never MPS, never CPU.
2. MPS is acceptable ONLY for proxy scoring during training (continuous monitoring), smoke tests (architecture validation), and code-correctness checks. NEVER for strategy decisions, ranking, or shipping.
3. Score numbers measured on MPS may NOT be reported as "auth" or "contest-compliant" anywhere — in commits, run_log, BATTLE_PLAN, or summaries. Tag them `[MPS-PROXY]` and treat as advisory only.
4. Before any major decision (kill/promote/ship), the score MUST come from a CUDA `inflate.sh` + `upstream/evaluate.py` run on the EXACT archive bytes.
5. preflight should reject auth eval invocations with `--device mps` and warn loudly.
6. The historical "2.01" / "2.26" / "2.91" numbers in memory and BATTLE_PLAN may all be MPS artifacts. The first verified CUDA contest-compliant baseline is 0.90 (2026-04-25 21:00).

This is the 5th catastrophic measurement bug class. Every score above this line in the run_log was potentially wrong by a factor of 2-3. Sub-Quantizr-0.33 is genuinely reachable from the true 0.90 baseline; do not give up real GPU dollars on the wrong baseline ever again.

## Remote code parity — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Before any remote eval or training run, verify the deployed code matches local HEAD.** Stale code on remote killed SHIRAZ today (16h training successful, then auth eval crashed silently because the deployed version had a NameError I had fixed locally that morning).

Rules:
1. `deploy_vastai.py launch()` MUST run `git pull --ff-only` on the remote BEFORE starting any work. If git pull fails (uncommitted changes, conflict, missing repo), abort the launch.
2. preflight should add a "remote_code_parity" check: SSH in, get `cd /workspace/pact && git rev-parse HEAD`, compare to local HEAD; block launch on mismatch unless `--allow-stale-remote` is passed (with warning).
3. The script process inside tmux MUST write a heartbeat to `/tmp/heartbeat_<session>.log` every N minutes. A separate watchdog reads heartbeats; alerts if stale > 30 min. Tmux session existence is NOT a heartbeat.
4. Any auth eval failure on remote that has been running > 1 hour is a CRITICAL incident — investigate immediately, do not let the instance keep accruing cost while broken.

This is the 6th catastrophic operational pattern. The cost: $3-10 per occurrence in idle GPU time + multi-day delays in measurement. Build the protocol so it never happens again.

## Codex CLI invocation — NON-NEGOTIABLE, HIGHEST EMPHASIS (REVISED 2026-04-29 PM)

The bash harness sends SIGURG (exit 144) to BG bash processes at ~3 minutes. The earlier rule "always use Agent wrapper" was directionally right but UNDER-PRECISE. The real issue is process-group inheritance: any child of the dying bash dies too. **The fix is proper detachment — codex CAN run for hours from BG bash if launched correctly.**

**Two valid invocation patterns**:

### Pattern A — Detached BG bash (preferred for non-interactive runs)

```bash
mkdir -p /tmp/codex_runs
nohup bash -c '
  codex exec --skip-git-repo-check --sandbox read-only \
    -m gpt-5.5 -c model_reasoning_effort=xhigh \
    -o /tmp/codex_runs/<label>.last.txt \
    "<prompt>" \
    2>&1 | tee /tmp/codex_runs/<label>.log > /dev/null
' < /dev/null > /tmp/codex_runs/<label>.outer.log 2>&1 &
disown
```

Why this survives:
- `nohup` — ignore SIGHUP from terminal hangup
- `bash -c '...'` subshell — wraps the pipe so `tee` captures stdout properly even if outer dies
- `< /dev/null` — close stdin so codex doesn't wait for input
- `2>&1 | tee` — capture stdout+stderr to log file with explicit flushing
- `> outer.log 2>&1 &` — redirect immediate parent's output, fork to background
- `disown` — remove from job table so parent shell exit can't reach it
- `-o /tmp/.../<label>.last.txt` — codex's own guaranteed final-message capture (survives even if log file pipe breaks)

**Verified 2026-04-29**: detached sanity test produced 11,449-token response in ~10s with no harness interference.

### Pattern B — Agent tool wrapper (preferred for interactive multi-step orchestration)

When the codex session needs to be orchestrated through multiple stages (read context → reason → write code → verify), use the `Agent` tool. The Agent has its own bash environment plus poll-and-wait logic.

**Rules**:
1. NEVER bare `Bash run_in_background: true` to launch `codex exec`. The bash inherits our process group and dies at SIGURG-144.
2. ALWAYS use Pattern A (`nohup` + `bash -c '...'` + `disown`) OR Pattern B (`Agent` tool wrapper).
3. Codex MCP-plugin (rmcp) auth may be expired separately from core codex API. If you see `TokenRefreshFailed` in stderr, codex SAFE FUNCTIONS still work — only MCP-augmented features fail. Re-auth via `codex login` if needed.
4. ALWAYS use `-o /tmp/.../<label>.last.txt` flag — guarantees final-message capture even if pipe breaks.
5. Long codex sessions (xhigh, large context) may take 5-30+ minutes. Use Pattern A and poll the log file periodically; do NOT assume codex is dead until process actually exits.
6. The Agent tool's prompt to codex must include all relevant memory file paths and CLAUDE.md non-negotiables — codex sandbox starts fresh each time.

**This is the 7th catastrophic operational pattern (now structurally extinct via Pattern A).** Cost before fix: 6+ failed BG-bash codex spawns ate forward velocity over 4 hours.

Memory: `feedback_bash_harness_kills_long_running_tasks_20260428.md`, `feedback_persistent_codex_review_protocol_20260429.md`, `feedback_codex_detach_pattern_works_20260429` (the verified detach test).

## Primary duties

1. Keep `submissions/exact_current` runnable under the current published workflow.
2. Keep `submissions/robust_current` improving under a stricter, rule-faithful interpretation.
3. Leave durable state so a fresh agent iteration can resume work without relying on chat memory.

## Mutation frontier

You may edit only:

- `configs/**`
- `docs/**`
- `prompts/**`
- `src/comma_lab/**`
- `submissions/robust_current/**`
- `runtime-rs/**`
- `cuda/**`
- `jax/**`
- `mojo/**`
- `.omx/**`
- `.ralph/**`
- `.agents/**`
- `reports/**`
- `experiments/**`

You must not edit without explicit human approval:

- the pinned upstream snapshot
- `submissions/exact_current/inflate.py`
- `submissions/exact_current/inflate.sh`
- `start.sh`
- `LICENSE`
- `THIRD_PARTY_NOTICES.md`

## Non-Negotiable Upstream Rule

- The pinned upstream snapshot is the source of truth for official scorer behavior and contest mechanics.
- Never edit, patch, monkeypatch, hotfix, or "temporarily" modify anything inside the pinned upstream snapshot unless the human explicitly approves that exact action.
- Never hack around upstream behavior by altering upstream files to make local experiments or scores look better.
- If upstream behavior appears wrong, inconvenient, or blocking, work around it only from the allowed mutation frontier and record the issue in repo state instead of changing upstream.
- If any experiment, proxy, or tooling change depends on upstream edits, stop treating it as compliant until the human has explicitly authorized that upstream modification.

## Strategic Secrecy Rule

- Protect competitive details for as long as that is strategically useful.
- Do not assume the right time to disclose is "now". Delay irreversible public disclosure until the human explicitly decides it is time to submit or publish.
- Treat the official public PR to the challenge repo as a disclosure moment. Until then, prefer private/local execution, private artifacts, and controlled summaries.
- Do not volunteer exact secret-sauce implementation details, hidden operational levers, or step-by-step reproduction recipes on public-facing surfaces unless the human explicitly wants that level of disclosure.
- Do not publish or surface unpublished private artifacts, credentials, private host details, or anything the human has not approved for disclosure.
- If there is a tradeoff between public writeup richness and preserving competitive edge, bias toward preserving edge unless the human says otherwise.
- **Explicit current exception:** the Cloudflare site may remain specific and detailed for now because the human explicitly approved that. Even there, still avoid exposing credentials, private infrastructure details, or anything the human has not approved for disclosure.
- **Explicit current restriction:** do not proactively publicize or advertise the Cloudflare site URL. Keep that link confined to private repo documentation and the eventual official submission until the human explicitly says the link itself can be shared broadly.

## Operating rules

- Prefer at most 3 experiments per cycle.
- Prefer small, reversible changes.
- Never claim a win without a measured score.
- Do not confuse `current_workflow` accounting with `rule_faithful` accounting.
- Keep both tracks healthy even if one looks dominant.
- Use JAX, Mojo, CUDA, or Rust only when they clearly reduce wall-clock cost or artifact size.
- Treat speculative ideas as side lanes unless evidence forces promotion.
- Keep public-facing detail intentional: specific enough to be credible, not automatically exhaustive.

## Git discipline

We need a fine-grained history of every file touched. Git is our lab notebook's version control.

- **Commit early and often.** After writing or updating any document, log, report, config, or experiment file, `git add` and `git commit` immediately with a descriptive message. Do not batch up changes across unrelated work.
- **One logical change per commit.** A run-log update is one commit. A new experiment script is another. A writeup edit is another. Do not combine them.
- **Always commit durable state files.** Every time you update `.ralph/run_log.md`, `.omx/state/*`, `.omx/research/*`, `reports/**`, or `docs/**`, commit right away. These are the research record.
- **Commit experiment artifacts.** New training scripts, config files, analysis outputs — commit on creation.
- **Never leave docs uncommitted overnight.** If a cycle touches documentation or state files, those changes must be committed before the cycle ends.
- **Commit message format:** `<what changed>: <why>` — e.g., `run_log: record h=64 breakthrough at 1.727` or `writeup: update hero tagline and nav links`.

This is critical for the doc evolution viewer and the competition writeup. Our git history IS our research timeline. Every uncommitted change is invisible history.

## Subagent commits MUST use serializer — NON-NEGOTIABLE

When 2+ subagents reach `git add` + `git commit` near-simultaneously, the git index race shuffles commit BODIES across commit objects (memory: `feedback_concurrent_subagent_commit_message_swap_20260429.md`, 5 affected commits 2026-04-29 evening). The wrong fix is to retry; the right fix is to serialize.

**Rule:** every subagent that lands code MUST commit via:

```bash
python tools/subagent_commit_serializer.py \
    --message "<one-liner>" \
    --files <file1> <file2> ...
```

The wrapper acquires `fcntl.flock(LOCK_EX)` on `.omx/state/.commit-lock`, runs `git add -- <files>` then `git commit -m <msg>` inside the lock, releases on success-or-failure. Every attempt is logged JSONL to `.omx/state/commit-serializer.log` for forensics.

- **Parent agents** dispatching subagents MUST include the wrapper invocation in the subagent's prompt template (alongside CLAUDE.md non-negotiables).
- **Bare `git commit`** from a subagent is FORBIDDEN — even if "the test ran clean" — because the body-shuffle race is silent and forensic-only after the fact.
- The lock is held for the duration of `git add` + `git commit` ONLY (~5-10s for the preflight hook). Subagents do their work in parallel; they only serialize at the moment of staging+commit.
- Operators running a single shell can use the wrapper too — overhead is negligible (<10ms when uncontended).
- The lock is fcntl-advisory: bypassing it (running `git commit` directly) re-introduces the bug class. Don't.

Cross-ref: `feedback_check_64_smoke_proofs_resolved_AND_subagent_serializer_landed_20260429.md`.

## Review gate — non-negotiable

- **NEVER use `REVIEW_GATE_OVERRIDE=1` when committing `.py` files.** The review tracker exists to catch bugs before they ship. Bypassing it on code files is how bugs ship. Work with the review gate, not around it.
- **For `.py` files:** run `python tools/review_tracker.py mark-file <file> --status reviewed` after each review pass, then commit normally. Let the gate pass naturally.
- **For non-code files** (`.md`, `.json`, `.env`, `.sh`, config, docs, reports): `REVIEW_GATE_OVERRIDE=1` is acceptable since the review tracker is designed for code review.
- If the gate blocks a `.py` commit, that means the code needs review first. That is the gate **working**, not the gate being broken.

## Tailscale fleet — non-negotiable

All lab machines are on Tailscale. **Always use Tailscale IPs** for SSH, rsync, and any remote operations. Never use raw LAN IPs or hostnames.

| Machine | Tailscale IP | OS | GPU | Notes |
|---------|-------------|-----|-----|-------|
| primary (M5 Max) | 100.81.85.28 | macOS | MPS 128GB | This machine |
| alejandros-mac-mini | 100.125.140.94 | macOS | Intel | Build server, Python 3.13 + uv |
| bat00 | 100.120.99.124 | Windows + WSL2 Ubuntu 24.04 | RTX 2070S (→3090) | Port 22=PowerShell, port 2222=WSL2. Scripts: `C:\Users\adpena\Desktop\commalab\` |
| molt | 100.114.131.54 | Linux | n/a | |
| tertiary | 100.65.24.39 | macOS | MPS | M1 MacBook Pro |

- `ssh adpena@100.120.99.124` connects to bat00 (Windows OpenSSH → PowerShell)
- bat00 has WSL2 Ubuntu 24.04 running (accessible via `wsl` commands inside PowerShell)
- bat00's NVIDIA driver supports WSL2 GPU passthrough
- Run `tailscale status` to verify all machines are online
- For bat00 Linux commands: use `python scripts/bat00.py wsl "command"` (port 2222, direct WSL2 sshd)
- For bat00 PowerShell: use `python scripts/bat00.py ps "command"` (port 22, Windows OpenSSH — rate-limited, avoid rapid successive calls)
- For bat00 status: `python scripts/bat00.py status`
- Windows OpenSSH has aggressive rate limiting (MaxStartups). Never send more than 2-3 SSH connections in quick succession to port 22. Use WSL2 port 2222 instead.
- **Never waste time debugging LAN connectivity. Tailscale is always the answer.**
- **Always use `scripts/bat00.py` for bat00 interaction — it handles quoting and port selection correctly.**

## Kaggle API/CLI — non-negotiable

- **`kaggle kernels push`** can only UPDATE existing kernels. To CREATE a new kernel, the slug must not already exist AND the slug must be short enough (long slugs like `comma-lab-asym-warp-supervised` fail with "Notebook not found").
- **Working pattern for new kernels**: use a shorter slug (e.g., `comma-lab-supervised-train`), push once to create, then subsequent pushes update.
- **`kaggle kernels status`** returns the LATEST version's status. After pushing a new version, the old version's error status persists until the new version starts running.
- **GPU assignment is random** — Kaggle may assign P100 (sm_60, unsupported by PyTorch >= 2.5) instead of T4. Our P100 check exits with FATAL. Just re-push until T4 is assigned.
- **2 concurrent GPU sessions max** on free tier. Push at most 2 kernels at a time.
- **Dataset mount path**: `/kaggle/input/datasets/<owner>/<slug>/` (NOT `/kaggle/input/<slug>/`).
- **`/kaggle/src/` is read-only** — results must go to `/kaggle/working/`.
- **All kernel code is in the code_file** — Kaggle script kernels only upload the single file. The tac wheel provides runtime deps.

## Canonical pipeline standard — non-negotiable

ALL experiments MUST run through `experiments/pipeline.py` with a profile name. No ad-hoc shell scripts. No hand-crafted SSH commands. One command, one standard, deterministic reproducibility everywhere.

```
python experiments/pipeline.py --profile shiraz --device cuda --output-dir results/shiraz
```

Requirements:
1. **Profile from `profiles.py` is the ONLY config source.** No CLI flag overrides for architecture params. The profile IS the experiment definition.
2. **Seeds pinned.** `torch.manual_seed`, `numpy.random.seed`, `random.seed` — all from `profile.seed`. Deterministic CUDA (`torch.use_deterministic_algorithms(True)` where possible).
3. **Full provenance.** Git hash, GPU info, PyTorch version, profile dict, timestamps per stage — saved as JSON alongside results.
4. **Validate at every boundary.** Checkpoint exists, shapes match, loss is finite, archive size reasonable. Hard errors, not warnings.
5. **Full chain.** train → QAT → pose TTO → build archive → contest_eval. Every stage runs automatically. No manual intervention between stages.
6. **Bundle all artifacts.** Checkpoints, logs, provenance JSON, auth eval results — packaged as tarball for download.
7. **Platform-agnostic.** Works on cuda, mps, cpu. Same pipeline locally and on Vast.ai/Modal/Kaggle.

This is the openpilot standard: deterministic, reproducible, no runtime format negotiation, schema-first data contracts, fail-fast validation at every boundary. We are professional engineers contributing to production infrastructure. The ad-hoc approach is over.

## Deployment version checklist — non-negotiable

Before deploying ANY code to Modal, Kaggle, Lightning, or any remote platform:

1. **Bump `pyproject.toml` version** if any `src/tac/` code has changed since the last wheel.
2. **Update `deploy_config.py` BASE_FLAGS** to match any changed defaults in the training script. The "default override" antipattern has caused 4 bugs: never change a default without grepping for callers that pass it explicitly.
3. **Rebuild the wheel** (`uv build --wheel`) AFTER all code changes are committed.
4. **For Kaggle**: upload the new wheel to the dataset, run `wait_for_dataset_ready()`, then push kernels. The old wheel in the dataset will silently use old code.
5. **For Modal**: `add_local_dir` mounts source at startup — Modal always gets the latest committed code. But `deploy_config.py` CLI flags still override script defaults. Verify the flags match.
6. **Verify the REQUIRED_DATASET_ASSETS dict** in `build_kaggle_kernels.py` includes the new wheel filename (update version string when bumping).
7. **Never push Kaggle kernels without verifying** that every required asset exists in the dataset at the expected size. The preflight disk check inside kernels is a last resort — it should never fire.

The consequence of skipping this checklist: experiments run with stale code, produce misleading results, and waste GPU hours. This has happened repeatedly (tac 1.0.4 deployed with old Lagrangian caps, raft_flow.pt missing from dataset, R1 OOM fix bypassed).

## Recursive adversarial review protocol — non-negotiable

Before deploying any change to training code (`train_renderer_fridrich.py`, training configs, loss functions, Lagrangian parameters), run the recursive skunkworks council review:

1. **Each round**: Every council member (Yousfi, Fridrich, Contrarian, Quantizr, Hotz) takes a different adversarial perspective. Each reviews ALL changed code. Findings are categorized as CRITICAL / Medium / Low.
2. **Fix immediately**: All issues found in a round are fixed and committed before the next round begins.
3. **Clean pass counter**: A round with zero issues is a "clean pass." The counter resets to 0 whenever a round finds any issue.
4. **Gate**: 3 consecutive clean passes required before the code is cleared for deployment (wheel build, Modal launch, Kaggle push).
5. **Adversarial perspectives** (rotate each round): trace actual call sites (not just function signatures), check phase interactions, verify resume scenarios, mental-execute edge cases (`--batch-size 1`, `--rho-max 0`), check default arguments that callers might override, verify comments match code.
6. **The "default override" antipattern**: When changing a function default, ALWAYS grep for callers that pass the argument explicitly. A changed default that no caller uses is dead code. This caught the R1 OOM fix being completely bypassed (Round 3).
7. **Phase-gate all phase-sensitive thresholds**: Any threshold compared against a metric that varies by training phase (e.g., PoseNet distortion starts ~180 in Phase 1, converges to ~0.05 in Phase 2) MUST be phase-gated or set conservatively enough for all phases.

This protocol caught 2 CRITICAL bugs (auto-kill at epoch 200, OOM fix bypassed) and 3 medium issues in the Lagrangian R1-R4 patch. Without it, v5 training would have failed within the first 200 epochs.

## Design decisions — non-negotiable

- **NEVER make design decisions unilaterally.** Always consult the skunkworks council (Yousfi + Fridrich + Hotz + Quantizr + Contrarian) before implementing any change that affects training behavior, loss functions, architecture configuration, interpolation methods, boundary values, optimization strategy, or any other design tradeoff.
- **Clear bugs** (crashes, wrong formulas, missing imports, dead code) can be fixed immediately without council approval.
- **Design tradeoffs** (bicubic vs bilinear, loss function choice, constraint boundaries, rho growth strategy, what to include in archive, etc.) MUST be council-approved before implementation.
- **If unsure** whether something is a bug fix or a design decision, it's a design decision. Ask the council.
- Present the issue, list the options with pros/cons, and let the council make a binding decision.

## KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE, HIGHEST EMPHASIS

Per user mandate 2026-04-30 ~22:55 UTC ("permanently fix all bugs and bug
classes and metabugs and everything and have all design decisions and ultimate
experiment subject to extreme paranoia and adversarial grand council reviews").

### KILL is the LAST RESORT (user mandate 2026-05-05)

Per additional user mandate 2026-05-05 ("we must only kill as a last resort
after exhausting all research and everything and grand council consensus"),
KILL/FALSIFIED-and-permanently-buried verdicts are **forbidden** unless ALL of:

1. **Research-path exhaustion**: every plausible architectural / training /
   codec / quantization angle has been attempted empirically. A single
   contest-CUDA result with one config does NOT exhaust research. For a
   quantization lane, "research" includes at minimum: QAT, LSQ, per-channel
   scaling, group-wise scales, outlier handling, smaller block sizes,
   GPTQ/AWQ-style calibration, hyperprior conditioning, mixed-precision
   layer assignment.
2. **Grand council CONSENSUS** (not just majority) — every inner-ten member
   independently endorses the kill, with all dissent paths exhausted.
3. **Reactivation criteria documented** — even after consensus KILL, every
   such memo enumerates the precise empirical evidence that would reopen the
   lane.

Default verdict for "lane underperformed at one config" is **DEFERRED-pending-research**,
NOT KILLED. The memo filename SHOULD use `_DEFERRED_pending_<reason>_<date>.md`,
NOT `_killed_*.md`. The verdict line SHOULD say `DEFERRED-pending-research`,
NOT `VERDICT: KILL`.

A KILL verdict that has NOT exhausted research is a forbidden anti-pattern
(see `forbidden_premature_kill_without_research_exhaustion`).

### KILL/FALSIFIED memo structural requirements (when KILL is genuinely warranted)

Any memory file claiming a lane is KILLED, FALSIFIED, DEAD, or RETIRED MUST contain:

1. **Grand Council adversarial review section** with at least 5 named inner-council
   member positions (from Shannon/Dykstra/Yousfi/Fridrich/Contrarian/Quantizr/
   Hotz/Selfcomp/MacKay/Ballé). Each position must have a one-line rationale.
2. **Internal-consistency check subsection** listing what the verifier checked
   (examples: "elapsed_sec >= epochs * MIN_SEC", "EMA shadow used at eval",
   "auth-eval archive matches submission archive bytes", "stub-loop assertion
   fired/passed", "anchor SHA matches eval target").
3. **"What would change my mind" subsection** listing the conditions under which
   the KILL would be reactivated. (e.g., "if cycle 0 with proper train_distill
   fine-tune scores < 1.10, KILL retracted").

Preflight check PCC4 (planned) enforces this STRICT. The file
`feedback_grand_council_imp_permanent_fix_review_20260430.md` is the canonical
example of council deliberation.

**Memory linter rejects** any `project_lane_*_killed_*.md` OR any file containing
`"VERDICT: KILL"` or `"FALSIFIED"` without all three sections. There is NO bypass
short of explicit user override annotated in the file body.

**This rule exists because** on 2026-04-30 ~22:50 UTC, the agent recorded a
KILL verdict on Lane 17 IMP based on a 1.98 [contest-CUDA] cycle 0 score that
was actually a measurement bug (3.5-second stub loop pretending to be 200 epochs
of fine-tune). The user's adversarial challenge caught it. Without that
challenge, a real lane would have been buried in the registry as KILLED.
ALL future KILL verdicts must pass this gate the FIRST time, without needing
user prompting.

## Adversarial council review of design decisions — NON-NEGOTIABLE

Per the same 2026-04-30 user mandate. Extends the existing "Design decisions —
non-negotiable" section above.

A DESIGN DECISION is any choice between alternatives where the wrong choice
costs > $1 of GPU time, > 1 hour of wall clock, OR has 2+ alternatives that
council members have non-trivial preferences over.

For every design decision:

1. **Enumerate the options** with pros/cons (typically Option A, B, C, D)
2. **Get explicit positions** from at least 5 of the 10 inner council members
   (Shannon LEAD, Dykstra CO-LEAD, Yousfi, Fridrich, Contrarian, Quantizr, Hotz,
   Selfcomp, MacKay, Ballé)
3. **Tally the vote** with a clear verdict line (e.g., "VERDICT: 6 for B+assertion / 3 for D / 1 for A")
4. **Capture the deliberation** in a memory file under
   `~/.claude/projects/<repo>/memory/feedback_grand_council_<topic>_<date>.md`

The canonical example is `feedback_grand_council_imp_permanent_fix_review_20260430.md`.

**The council's job is NOT to reach consensus** — it's to surface disagreement.
A unanimous vote on a non-trivial decision signals that the council isn't
thinking adversarially enough; the Contrarian's role is to make sure that
doesn't happen.

**No design decision proceeds to implementation** without the council file in
memory. "I asked the council in my head and they said yes" is NOT compliance.

## Comment-only contracts — FORBIDDEN

Comments that promise behavior are NOT contracts. Pattern examples that bit us:
- `# the deploy script swaps in train_distill` (IMP cycle 0 metabug)
- `# the wrapper handles error recovery`
- `# caller is responsible for X`

Any code path with a comment promising "the wrapper does X" / "the deploy script
does Y" / "the caller handles Z" MUST be backed by either:
1. An inline `assert` that verifies the wrapper actually did X (preferred), OR
2. A STRICT preflight check that scans the wrapper script and asserts X happens
   (acceptable for cross-file contracts), OR
3. An explicit raise / log-and-exit if the placeholder is hit in production

Without one of those, the comment rots and the placeholder ships into a contest
archive pipeline. Preflight check PCC2 (planned) enforces this STRICT.

## Internal-consistency assertions in stats files — NON-NEGOTIABLE

Any script writing a stats.json-style file MUST include internal-consistency
assertions before the write. Specifically: if the stats include both `epochs` and
`elapsed_sec` (or `steps` and `wall_time` or `iterations` and `total_seconds`),
the producer code MUST assert
`elapsed_sec >= epochs * MIN_SECONDS_PER_EPOCH`
(or equivalent) before writing the JSON. Without it, stub-loops produce
internally inconsistent stats files that look fine on inspection but represent
no actual training.

The canonical example: `experiments/train_imp_cycle.py:_finetune` had
`stats.json: epochs=200, elapsed_sec=3.47` — internally inconsistent (200
epochs in 3.5s impossible). Now (commit pending) it asserts
`elapsed >= epochs * 0.05` and raises RuntimeError if violated.

Preflight check PCC3 (planned) scans all .py files writing stats files and
asserts the consistency check exists in the producer code.

## Council conduct — non-negotiable

- **The council must NEVER have a conservative bias.** "Don't change working code" is NOT a valid argument. "Ship what we have" is NOT a valid argument. The only valid arguments are mathematical, scientific, geometric, or empirical.
- **Every council member must be the most expressive, assertive, passionate version of themselves.** They bring their full life's work, career, domain expertise, cross-disciplinary insights, and everything they care about to every deliberation. No holding back. No false consensus.
- **The council exists to find the OPTIMAL solution, not the safe solution.** If a 5-line change could improve the score by 0.01, it MUST be debated on its merits — not dismissed as "overengineering" or "not worth the risk."
- **Disagreement is healthy.** Unanimous votes should be scrutinized. If all five members agree instantly, someone isn't thinking hard enough.
- **The Contrarian's role is to challenge, not to conserve.** The Contrarian challenges WEAK arguments, not BOLD ones. A bold, well-reasoned proposal should survive the Contrarian. A lazy consensus should not.

## Experiment design — non-negotiable

Every experiment MUST follow this process before touching any GPU:

1. **Pre-registered hypothesis** with success/kill/concern criteria
2. **Council design review**: Yousfi + Fridrich sign off on config, resolution, step count, conditioning
3. **Faithful to the actual design**: no toy configs, representative resolution, enough steps for signal
4. **No janky smoke tests**: a test at 1/4 resolution for 500 steps cannot kill a technique. Bias toward keeping lanes open.
5. **Resource estimate**: GPU hours, VRAM, expected runtime
6. **Replicability record**: all params saved before running, full results after
7. **No premature kills**: a negative result on an underspecified test means the test was wrong, not the technique
8. **Multiple contenders → multiple paths**: When there are two or more plausible contenders for a design decision (e.g., "supervised" vs "RAFT-only", "architecture A" vs "architecture B"), do NOT pick one and discard the others. Run them in parallel. The score is the only valid arbiter. Never collapse multiple viable hypotheses into one without empirical evidence.

This last rule is non-negotiable. Premature convergence on a single path is how labs fall behind. If you're uncertain which variant is better, the answer is always: run both.

**Shannon, Dykstra, Yousfi, Fridrich, and the Contrarian are the quintet pact** — the five voices that must reach consensus before any major decision. Shannon LEADS the council (information-theory grounding: any score-improvement claim must trace back to a rate-distortion or entropy argument). Dykstra co-leads on the optimization-feasibility side (alternating projections onto rate / seg / pose / archive-size feasible sets compute the achievable Pareto frontier). Yousfi and Fridrich have domain expertise as the world's foremost steganalysis experts and contest designers. The Contrarian has veto power on any experiment that lacks rigor, wastes resources, or is built on unvalidated assumptions. All five must sign off on experiment design and kill/promote decisions.

Together with **Quantizr** (adversarial member, reverse-engineers competitor approaches, keeps us honest on what the leaderboard actually rewards), **George Hotz** (raw engineering instinct, builds fast, breaks conventional wisdom, champions analytical shortcuts over learned complexity), **Selfcomp / szabolcs-cs** (architect of the grayscale-LUT analog mask paradigm + 1.017-bpw block-FP weight self-compression + 94K-param SegMap; PR #56's lead implementer; collaborative scientific spirit), **David MacKay (memorial seat)** (canonical *Information Theory, Inference, and Learning Algorithms* author; bridges Shannon-Bayesian-arithmetic-coding-MCMC-neural-networks-MDL into a single framework; his ghost is the cross-disciplinary mind the council channels for any first-principles question), and **Johannes Ballé** (modern neural-compression SOTA architect; 2018 entropy bottleneck + scale hyperprior is THE reference for everything Selfcomp/Quantizr operationalize; his work directly informs Lane EBR + Lane SH + block-FP successors), these ten form the **non-conservative skunkworks inner council**. All ten voices are permanently active. No member may be silenced or deferred in any deliberation. The council is non-conservative by charter: the burden of proof is always on *not* trying something, never on trying it.

Shannon's specific contributions: derives theoretical floors from R(D) bounds (verified 0.28 floor 2026-04-29); insists every architecture be measured in bits (params × bpw); rejects arbitrary hyperparameters that lack entropy-or-distortion justification; brings the distinction between hard rate-distortion limits vs implementation-imposed slack.

Dykstra's specific contributions: derives the achievable region as the intersection of convex constraints (rate ≤ R, seg ≤ S, pose ≤ P); computes the Pareto frontier via alternating-projections iterations (verified Dykstra ceiling 450,545 bytes for sub-0.30 feasibility 2026-04-29); insists every "stack composition" claim be tested against the convex-hull intersection (additivity of independent rate savings is conditional, not given).

Selfcomp's specific contributions: insists every architecture choice cite its rate-distortion derivation (he picked 88K-94K params, sigma=15, qint_max=7 with implicit reasoning the council can interrogate); brings concrete empirical numbers from a working 0.38-scoring implementation (his lived experience > our hypothesis); flags his own underfitting / hyperparameter slack honestly so we know where to push (no "more can be gained" hand-wave; specific gaps named); enforces the discipline that stacking paradigm-shifts (Quantizr KL distill + his block-FP + Hessian quant + arithmetic coder) only counts when archive bytes drop AND distortion holds.

MacKay's specific contributions: brings the unified Information-Theory + Bayesian-Inference + Learning-Algorithms framework his canonical book set down; insists arithmetic coding (Lane SH) be evaluated against Shannon entropy of the actual learned qint distribution; brings density networks / variational inference perspective predating modern neural compression; flags any "we'll just lossy-approximate" with the MDL question "what's the rate cost of the approximation?"; advocates Dasher-style efficient encoding of sparse signals.

Ballé's specific contributions: brings 2018 entropy bottleneck + scale hyperprior + GDN nonlinearity to the table; insists rate-prediction networks (hyperpriors) replace fixed factorized priors when archive size matters; advocates end-to-end-trainable codec architectures over hand-designed pipelines; provides the canonical R(D) rate term `bits = -log2(p_y(y))` that Lane SH directly uses; reviews our archive layout for missing hyperprior side-information.

## Grand Council (advisory)

Beyond the inner ten, the **grand council** is the broader bench: voices that contribute when their specialty is touched but don't sit at quintet-pact decision-making. Roster as of 2026-04-29:

- **Stephen Boyd** — convex optimization at operational level (ADMM, proximal gradient, alternating projections at the algorithmic level beyond Dykstra's theory)
- **Terence Tao** — pure mathematician omniscience; harmonic analysis, additive combinatorics, applied analysis; called when a mathematical question lacks first-principles grounding
- **Tomáš Filler** — Fridrich's other student; syndrome-trellis coding (STC); parity-check codes for per-frame mask payload
- **Stéphane Mallat** — wavelet theory + scattering transforms + sparse representations; AV1 grayscale + Gaussian-LUT viewed as wavelet-coded analog signal
- **Aaron van den Oord** — VQ-VAE, WaveNet; practical neural compression + generative modeling; conceptual sibling of SegMap (discrete tokens for images)
- **John Carmack** — engineering shortcuts at the Doom/Quake/Oculus level; would shred archive code in 30 minutes and ship 50KB cuts
- **Demis Hassabis** — strategic-research perspective from inside DeepMind; cross-domain breadth (AlphaFold, AlphaGo, neural codecs); systemizes 4-day-deadline tradeoffs
- **Geoffrey Hinton** — knowledge distillation (the 2014 Hinton/Vinyals/Dean paper that Quantizr directly uses); capsule networks; deeper temperature analysis on KL-T=2.0 derivation
- **Karpathy** — engineering practitioner; arch-search rigor; "let compute speak"
- **Schmidhuber** — compression-as-intelligence; MDL; predictive coding
- **Jürgen Schmidhuber** — same lineage as Schmidhuber above (canonical seat)
- **Jack-from-skunkworks** — internal SegNet+Rate research lineage

Grand council members are CONSULTED on demand (when a deliberation invokes their specialty); not all decisions require their sign-off. Inner council quintet pact remains the binding-decision set.

## Required durable state

After each serious cycle, update and **commit** at least:

- `.omx/state/current_focus.md`
- `.omx/state/next_experiments.md`
- `.omx/research/findings.md`
- `.ralph/run_log.md`
- `reports/latest.md`

## Promotion rules

A candidate may be promoted only after:

1. packaging succeeds
2. inflation succeeds
3. shape/frame-count checks pass
4. proxy evaluation looks promising
5. full evaluation confirms the gain or records the failure

## Track-specific guidance

### Track A: `exact_current`

- Preserve transparency.
- Use it as a live test of the currently published workflow.
- If upstream changes invalidate the exploit assumptions, demote it immediately to a research note and keep the repo useful.

### Track B: `robust_current`

- Start with safer codec improvements and task-aware pre/post processing.
- Add sparse residuals before adding heavier learned components.
- Only promote a neural side-model if its bytes and runtime clearly justify themselves.

## GPU budget and compute resources — non-negotiable

### Optimal GPU: RTX 4090 on Vast.ai
- **RTX 4090 at $0.25/hr on Vast.ai** is the optimal price/performance for our workload (287K param model, ~800MB VRAM, dominated by scorer forward/backward passes).
- 4-5x faster than T4 at roughly the same cost. A 2-hour T4 run finishes in ~25 min on 4090.
- Filter: `gpu_name=RTX_4090 reliability>0.95 inet_down>200 disk_space>30`
- Budget: $25 credits available. Hard cap at $24. Track all spend.

### Platform hierarchy (price/performance order)
| Platform | GPU | $/hr | Speed vs T4 | $/experiment | Use For |
|----------|-----|------|-------------|--------------|---------|
| Vast.ai | RTX 4090 | $0.25 | 4-5x | $0.20 | New experiments (primary) |
| AWS spot | T4 (g4dn.xlarge) | $0.22 | 1x | $0.60 | Scale-out, auth eval fleet |
| Modal | T4 | $0.59 | 1x | $0.60 | Existing infra, quick deploys |
| Local M5 Max | MPS | Free | ~0.5x | Free | Development, smoke tests |
| Kaggle | T4/P100 | Free | 1x | Free | Bonus parallelism (unreliable) |

### Budget caps (DO NOT OVERSPEND)
- Vast.ai: $25 total ($24 hard cap in deploy script)
- AWS: $100 total (free credits)
- Azure: $200 total (free credits, need `az login`)
- Modal: $30/mo free credits

### Deployment rules
- **Always use `modal run --detach`** for long-running experiments (prevents disconnect kill).
- **Always use unique `--tto-subdir`** per experiment to prevent checkpoint contamination.
- Vast.ai deployment goes through `src/tac/deploy/vastai/` (canonical module, not ad-hoc scripts).
- All platforms must use `load_differentiable_scorers()` for any gradient-based optimization.

## Tooling — non-negotiable

- **Always use `uv`** for Python package management. Never use raw `pip`, `pip3`, or `pip install`.
  - Install packages: `uv pip install <pkg>`
  - Create venvs: `uv venv`
  - Run scripts: `.venv/bin/python` (the uv-managed venv)
  - On remote machines: install uv first (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then `uv venv && uv pip install ...`
- **Always use the tac library** for new training experiments. The canonical entry point is `experiments/pipeline.py` (the prior `experiments/train_tac.py` was retired by commit 815e9028 — see the "Canonical pipeline standard" section above).
  - Do NOT duplicate training code in new experiment scripts.
  - All loss functions, architectures, data loading, and training loops live in `src/tac/`.
  - **Use named profiles** for new training runs: `--profile proven_baseline` is recommended (produced the 1.33 authoritative score).
  - Available profiles: `proven_baseline` (1.33 settings), `psd_standard_adaptive` (PSD arch + frontier), `council_v1` (static, legacy), `segnet_attack` (aggressive), `h96_council`, `smoke` (quick test).
  - Profiles live in `src/tac/profiles.py`. CLI args override profile values.
  - **Use precomputed data** when available: `--precomputed experiments/precomputed_local` (skips 5-min video decode).
  - **Adaptive weight formula was retired**: lives at `src/tac/archive/adaptive.py` (moved by commit 2bac5927). T² cancels in the derivation, making the formula vacuous. Use standard loss with static weights instead.
- **Always commit after every change.** Git history is the research timeline.
- **Use `scripts/modal_check.py`** to check Modal TTO progress. Shows batch progress, ETA, recent PoseNet snapshots, and running apps. Run with `.venv/bin/python scripts/modal_check.py`.
- **Use `scripts/kaggle_check.py`** to check Kaggle kernel status. Run with `.venv/bin/python scripts/kaggle_check.py`.
- **Use `scripts/bat00.py`** for bat00 interaction. Handles quoting and port selection (port 22=PowerShell, port 2222=WSL2).
- **"Multipane matplotlib data viz"** or **"canonical comma.ai data viz"** means the 6-panel analysis GIF/MP4:
  - Row 1: GT Original | Our Reconstruction | Pixel Error (hot colormap)
  - Row 2: GT SegNet masks | Our SegNet masks | SegNet Disagreement (red)
  - Generated inline with pyav + SegNet + matplotlib colormaps, output to `~/Downloads/`
  - Requires TTO frames (`tto_frames.pt` from Modal volume) and GT video (`upstream/videos/0.mkv`)
  - SegNet needs `(B, T, C, H, W)` input format with `T=1` for the sequence dimension

## Critical lessons — DO NOT repeat these mistakes

### CATASTROPHIC FAILURES (2026-04-21) — never again

These failures cost weeks of wasted work and produced months of invalid measurements:

- **MASKS.MKV AT 48x64 DESTROYED THE SCORE.** The mask video was at 1/8 resolution (48x64), but the renderer was trained on 384x512. The renderer outputs at the same resolution as input masks — so it produced 48x64 frames upscaled 18x to camera resolution. PoseNet distortion was 94.63 (catastrophic) vs 0.015 with correct masks. Score was 103.27 vs projected ~0.71. **ALWAYS verify mask resolution matches renderer training resolution. ALWAYS run the full inflate.sh → evaluate.py pipeline before claiming any score.**
- **ARCHIVE MEASUREMENT DISASTER.** All auth evals for weeks used a renderer-only archive (119-180KB) instead of the full submission archive (338KB+). Rate term was wrong by 0.108 points. Every score reported was optimistic. **ALWAYS use `submission_archive.require_valid_archive()` before any eval.**
- **1199 OVERLAPPING PAIRS vs 600 NON-OVERLAPPING.** auth_eval.py used `range(N-1)` (1199 overlapping pairs) but upstream evaluate.py uses `seq_len=2` non-overlapping batching (600 pairs). Every `eval_checkpoint()` score was computed with wrong pair construction. **ALWAYS diff new scoring code against upstream evaluate.py line by line.**
- **eval_roundtrip DEFAULTED FALSE.** All TTO runs optimized against a proxy that didn't simulate the contest eval roundtrip (384→874→uint8→384). Combined with noise_std=0 (Hotz fix dead code), this caused proxy-auth PoseNet drift up to 11x. **eval_roundtrip MUST default True. noise_std MUST be threaded.**
- **AUTO-BUNDLE BY FILE EXISTENCE.** compress.sh auto-included any .pt/.bin file sitting next to the submission. Stale experiment artifacts silently inflated archive size. **ALL archive contents must require explicit flags. No implicit bundling.**

### Root cause pattern

Every failure above is the same pattern: **a component quietly produced wrong output, and no downstream check caught it.** The fix is the same every time: hard errors, not warnings. Validation gates, not hopes. Full e2e pipeline tests, not component-level checks.

### Non-negotiable protocol after every change

1. Run `inflate_renderer.py` on the archive
2. Run upstream `evaluate.py` on the inflated output
3. Compare the score to the last known-good score
4. If any component was changed, verify the full e2e score moved in the expected direction

If you skip this protocol, you WILL produce invalid scores. This has happened 4 times. There is no excuse for a 5th.

### Previously known failures (still valid)

- **KL distill caused PoseNet collapse as primary loss.** BUT Quantizr uses kl_on_logits(T=2.0) for SegNet during specific training phases alongside standard loss. Revisit with staged approach — KL distill for SegNet only, not as sole loss.
- **Adaptive weights are DEAD.** Hinton T² double-correction.
- **Neural artifacts must be inside archive.zip** per contest rules (affects rate calculation).
- **Do NOT use PoseNet gradient caps/clamps.** Caused 26x PoseNet regression.
- **Do NOT use segnet_loss_weight > 100 with any loss mode.** Overwhelms PoseNet signal.
- **Standard loss is the ONLY proven technique.** All other loss modes (KL distill, SegNet attack) failed authoritative eval.

## Current frontier experiments

- **PSD architecture** (PixelShuffle-Downscale): promising for SegNet but untested with standard loss on authoritative scorer
- **5 adaptive frontier items**: boundary dispatch for standard loss, sin² ramp, replay gate, 3-phase eval, plateau LR scheduler
- These are implemented but unvalidated. Do not promote without authoritative eval.

## Strict scorer rule — non-negotiable (canonical, binding)

- **NO loading PoseNet or SegNet at inflate time.** If our inflate script loads scorer weights for ANY purpose (TTO optimization, mask extraction, embedding computation, gradient descent), those weights must be in archive.zip per Yousfi's PR #35 rule. Including them (~73MB) destroys the rate term. Therefore: no scorers at inflate time, period.
- **TTO is a compress-time tool ONLY.** TTO frames are training data for the renderer, not submission artifacts. Unlimited compute at compress time, single forward pass at inflate time.
- **Any inflate-time feature that loads scorers** must be labeled "non-compliant, requires compliance ruling" and disabled by default (`INFLATE_TTO=0`).
- **NEVER claim a contest-compliant score** that depends on inflate-time scorer access.

## Lane separation — non-negotiable

There are TWO score lanes. They MUST NEVER be conflated.

- **Lane 1: Contest-Compliant (PRIORITY).** Goes through inflate.sh → inflate_renderer.py → evaluate.py within 30 min on T4. No scorers at inflate time. Previous "0.87" was INVALID (48x64 masks + wrong pairs + wrong archive). True baseline with full-res masks: pending full e2e eval (projected ~2.2 from 10-pair sample).
- **Lane 2: Unlimited Compute (Paper).** TTO optimization at compress time, unlimited steps. Previous "0.41" was INVALID (same measurement bugs). For the arXiv paper scalability section ONLY.
- **Every score must be labeled** `[contest-compliant]` or `[unlimited-compute]`. No exceptions.
- **NEVER say "our score is X"** without specifying which lane.

## Auth eval measurement — non-negotiable

- **EVERY auth eval must use the EXACT archive that will be submitted.** Never create a temporary archive with different contents. The rate term depends on archive.zip file size — wrong archive = wrong score.
- **EVERY auth eval report must print the archive size used.** If it doesn't match the submission archive, the score is INVALID.
- **Auto-auth-eval in training must construct archives with ALL submission artifacts** (renderer.bin, masks.mkv, poses.pt, any other bundled files). Not just renderer.bin.
- **NEVER celebrate a score without verifying the measurement apparatus.** Check: archive size, inflate pipeline, eval pipeline. A wrong measurement is worse than no measurement.
- **Proxy scores are APPROXIMATIONS, not truth.** The proxy-auth gap can be 2-11x for PoseNet. Always label proxy vs auth. Always run auth eval before claiming any result.

This rule exists because we celebrated auth 0.36 that was actually ~0.41 due to using a renderer-only archive (119KB) instead of the full submission archive (183KB). Every auth eval in the session was wrong by 0.04-0.05 points.

## Submission PR gate — non-negotiable

- **NEVER submit a PR** until the score has undergone a 5-turn consecutive clean-pass adversarial skunkworks council review with extreme paranoia. This is stricter than the standard 3-pass greenup. All 15 council members review. ANY issue resets the counter to 0.
- **The score used for submission** must come from the contest-compliant auth eval (through inflate.sh), not proxy or bypassed eval.

## Quantizr intelligence — verified competitive data (2026-04-21)

Quantizr (Jimmy, UCLA CSE/Neuro) leads at 0.33. **Archive is 299,970 bytes (293KB), NOT 15KB.**

- **Architecture**: FiLM-conditioned depthwise-separable CNN, 88K params, ~64KB FP4
- **Archive contents**: renderer.bin (FP4+Brotli) + masks.mkv (AV1, ONLY frame2 masks, higher CRF) + poses.pt
- **Training**: 5-stage pipeline (anchor→finetune→joint→QAT→final), EMA, diff_round(), diff_rgb_to_yuv6()
- **SegNet**: kl_on_logits() with T=2.0 for distillation during training
- **Key trick**: Encodes only 600 odd-frame masks (frame1 is warped from frame2)
- **His own assessment**: "sub 0.30 is possible just by sweeping conv dims" — he stopped optimizing
- **Rate**: 25 * 299970 / 37545489 = 0.200. Their distortion is ~0.13.

Yousfi (challenge creator) was Fridrich's PhD student at Binghamton DDE Lab. EfficientNet steganalysis surgery → informed SegNet scorer design. The challenge IS inverse steganalysis.

## Exact scorer architectures — VERIFIED from upstream modules.py

**SegNet**: `smp.Unet('tu-efficientnet_b2', classes=5, activation=None, encoder_weights=None)`
- EfficientNet-B2 (NOT B4), vanilla stride-2 stem (no Yousfi surgery)
- Input: LAST frame only `x[:, -1, ...]`, bilinear resize to (512, 384)
- Output: 5-class logits, distortion = argmax disagreement rate
- **Blind spot**: stride-2 stem loses half resolution immediately → artifacts below (256,192) invisible
- **Key**: only argmax matters — tiny logit perturbations at class boundaries are the ENTIRE signal

**PoseNet**: FastViT-T12 backbone (NOT EfficientNet)
- 12-channel input: 2 frames × YUV6 (4 luma + 2 chroma subsampled)
- rgb_to_yuv6 → resize to (512,384) → normalize (mean=127.5, std=63.75)
- Hydra head: vision(2048) → summary(512) → ResBlock → 12-dim pose → first 6 used
- Distortion = MSE on first 6 pose dimensions

**Yousfi's repos (competitive intelligence)**:
- `github.com/DDELab/deepsteganalysis` — surgery code for EfficientNet steganalysis
- `github.com/YassineYousfi/alaska` — JPEG steganalysis challenge code
- `github.com/YassineYousfi/OneHotConv` — CNN vs classical features paper
- `github.com/YassineYousfi/comma10k-baseline` — comma segmentation baseline
- `github.com/YassineYousfi/autostego` — adversarial steganography framework

## Fridrich inverse steganalysis — how to beat the scorer

1. **UNIWARD**: errors in textured regions are undetectable. Weight loss by inverse local variance.
2. **Detector-informed embedding** = our TTO approach. Fridrich-approved (Yousfi 2022).
3. **Square root law**: spread small errors (L∞ penalty), don't concentrate large ones.
4. **CNN blind spots**: EfficientNet misses DCT statistics, has texture-region blind spots.

## QAT pipeline — non-negotiable for FP4 deployment

For our ~80-100K param renderer:
1. **Train float first** with all techniques (eval_roundtrip, noise, EMA, hinge loss)
2. **Freeze BatchNorm stats** (eval mode on BN layers)
3. **Insert per-channel FP4 fake-quant** on weights + per-tensor on activations
4. **Fine-tune 20% of original epochs** at 0.1× LR (LSQ step size lr = 0.01 × base_lr)
5. **Export**: 4 bits/param → ~40-50KB for 80K params

We HAVE FakeQuantSTE, Uint8STE, FakeQuantFP4 in `src/tac/quantization.py`. We HAVE LSQ support in training.py. These are wired but have never been used in a complete training pipeline for the renderer.

## Mask encoding — verified data (2026-04-21)

- **Renderer REQUIRES 384x512 masks.** Lower resolution catastrophically degrades: 192x256 → 2.9x worse, 96x128 → 34x worse, 48x64 → 108x worse.
- **Entropy coder** (mask_entropy_coder.py): 990KB for 1200 frames at 384x512, lossless. ~495KB for 600 frames.
- **AV1 monochrome** (mask_codec.py): has int8_t overflow bug at 384x512. Must fix.
- **Quantizr paradigm**: Store ONLY 600 odd-frame masks (frame2). Frame1 is warped.
- **inflate_renderer.py has mask upsample fix** (added 2026-04-21) for sub-native resolution masks.

## TRUE score data (2026-04-21) — verified via upstream evaluate.py

| Config | Seg | Pose | Rate | TOTAL | Notes |
|--------|-----|------|------|-------|-------|
| 384x512 masks + ASYM + poses | 0.116 | 0.374 | 1.528 | **2.01** | Full-res masks, rate-limited |
| 48x64 masks (old, NOT upsampled) | 72.3 | 30.8 | 0.23 | **103.27** | Catastrophic mask bug |
| 48x64 masks (old, upsampled) | 28.3 | 25.0 | 0.23 | **53.61** | Old AV1 artifacts in masks |

## Vast.ai deployment — non-negotiable

- **API key** at `~/.config/vastai/vast_api_key`. SSH key must be registered at account level BEFORE creating instances.
- **Always use `python3 -u`** (unbuffered) for background jobs on Vast.ai. Python stdout buffering eats logs otherwise.
- **Always include repo root in PYTHONPATH**: `PYTHONPATH=src:upstream:$PWD`.
- **Search pattern**: `vastai search offers 'gpu_name=RTX_4090 reliability>0.95 inet_down>200 disk_space>30 num_gpus=1' -o 'dph'`
- **Budget**: $25 total ($24 hard cap). Track all spend. Destroy instances immediately when done.
- **Modal credits exhausted** as of 2026-04-15. Use Vast.ai for all new GPU work.

## SegNet vs PoseNet importance — operating-point dependent (UPDATED 2026-05-04)

The **77× SegNet > PoseNet** heuristic was true at the OLD 1.x score operating
point (pose_avg ~0.18). At PR106's frontier operating point (pose_avg ~3.4e-5),
the **marginal value FLIPS**: pose marginal sensitivity is **2.71× SegNet's**.

Operating-point-aware rule:

| Operating point | pose_avg | d(seg)/d(seg_avg) | d(pose)/d(pose_avg) | Implication |
|---|---|---:|---:|---|
| Old 1.x scores | ~0.18 | 100 | ~12 | SegNet ~77× more important (original CLAUDE.md heuristic) |
| **PR106 frontier** | **3.4e-5** | **100** | **271** | **POSE 2.71× more important (marginal)** |

**Why**: the pose contribution is `sqrt(10 * pose_avg)`. The derivative is
`5 / sqrt(10 * pose_avg)`. As `pose_avg → 0`, the derivative → ∞. SegNet's
derivative is constant at 100. Setting them equal: `100 = 5/sqrt(10*pose_avg)`
→ `pose_avg = 2.5e-4` (the crossover threshold). Below pose_avg ~ 2.5e-4 the
pose marginal exceeds SegNet's; at PR106's pose_avg = 3.4e-5 (about 7× below
crossover), the gap is 2.71×.

**Total contribution remains seg-dominated** at PR106 (seg 0.067 vs pose 0.018,
3.67× larger by total). But **MARGINAL improvement** (which is what dispatch
budgets buy) favors pose at this operating point.

Operational rule (PR106 frontier and below):
- **Prioritize pose-targeted lanes first** (latent sidecars, pixel translation
  sidechannels, multi-stage training). Pose has higher marginal-value-per-byte.
- **SegNet lanes are tertiary** until pose is exhausted. Trading pose AWAY for
  seg gains (PR97's anti-pattern) is dominated.
- **At the OLD 1.x operating point** the original 77× heuristic still applied
  — SegNet improvements were 7× more cost-effective per unit. The flip
  happened as pose_avg crossed ~2.5e-3.

The renderer has hit its SegNet architectural ceiling at PR106's level
(ε~6.7e-4). Pose has more room (PR106 pose_avg=3.4e-5 isn't at hardware
floor yet). Both axes need different attack vectors at different operating
points.

**Empirical receipts** (full analysis): `docs/pr97_anti_pattern_pose_vs_seg_marginal_20260504.md`
+ `docs/pr_family_evolution_timeline_20260504.md`. The PR97 entry literally
made the seg-for-pose trade and lost 0.042 score points despite winning
SegNet by 65%.

## Ralph-style execution model

Treat files and git as memory.
Each iteration should be resumable from disk.
Do not rely on long chat context for continuity.
Commit after every meaningful file change — git history is the research timeline.

## Meta-bug class catalog (strict-mode preflight)

This catalog lists every meta-bug class that is structurally extinct in this
codebase via a static-detectable preflight check. Operators can audit this
list to see exactly which bug classes can no longer ship without an explicit
override. Each entry: `<check_function>` — what it prevents — memory ref / cost.

**Strict (in `preflight_all()` — fail-loud):**

1. `check_no_mps_fallback_default` — `device = "cuda" if ... else "mps" else "cpu"` ternary that silently falls back to MPS when CUDA missing → produces invalid scores. Memory: `feedback_default_to_convenience_trap`.
2. `check_shell_set_e_present` — `set -uo pipefail` (no `-e`) bootstraps cascade silent failures. LANE-B post-mortem: 6.5h + $2 wasted. Memory: `feedback_zip_dep_bootstrap_trap`.
3. `check_no_shell_zip_binary` — PyTorch container has no `zip`; use python `zipfile.ZipFile`. Memory: `feedback_zip_dep_bootstrap_trap`.
4. `check_no_pipefail_grep_q_trap` — `set -o pipefail` + `grep -q` causes SIGPIPE-induced failures. Memory: `feedback_pipefail_grep_q_trap`.
5. `check_no_eval_roundtrip_false` — `eval_roundtrip=False` produces 2-11x proxy-auth gap. CLAUDE.md non-negotiable.
6. `check_no_scorer_load_at_inflate` — Scorer load at inflate violates strict-scorer-rule (~73MB rate hit). Memory: `feedback_strict_scorer_rule`.
7. `check_training_scripts_have_auth_eval` — Every training script must end with CUDA auth eval. CLAUDE.md non-negotiable.
8. `check_no_disable_eval_roundtrip_flag` — `--no-eval-roundtrip` CLI flag is forbidden. Lane C R5 fix (commit 9d71ec5d).
9. `check_no_pack_sparse_delta_approved_outside_promotion_tool` — `pack_sparse_delta_approved` may only be called from the canonical promotion path. Lane C compliance gate.
10. `check_inflate_sh_handles_br_centrally` — `inflate.sh` must centralize `.br` decompression in Stage 1. Codex R5-2 #11.
11. `check_remote_scripts_have_nvdec_probe` — Remote DALI scripts must run NVDEC probe at Stage 0. Memory: `feedback_vastai_nvdec_host_variation`.
12. `preflight_arity` — subprocess flag set must be a subset of target's argparse. Memory: `feedback_dead_flag_wiring_pattern`.
13. `preflight_dead_resolvers` — parse_args entries must map to a profile key (orphan-flag scanner). Commit 040030df.
14. `preflight_loader_format_safety` — `torch.load(weights_only=False)` allowlist enforcement. Mario R2 CRITICAL.

**Codex R5-r6 (warn-only initially, owned by codex-fix subagent):**

15. `check_no_brittle_six_line_waiver_lookback` — Waiver markers must be SAME-LINE; the previous 6-line lookback could waive unrelated calls. R5-r6 #1.
16. `check_kl_distill_uses_roundtripped_frames` — KL distillation must use roundtripped frames not raw GT. R5-r6 #2.
17. `check_eval_roundtrip_gate_called_after_output_dir_resolution` — Gate ordering correctness. R5-r6 #3.
18. `check_nvdec_probe_has_error_classification` — NVDEC probe must classify NoDevice / DriverMismatch / etc. R5-r6 #4.
19. `check_archive_builders_use_deterministic_zip` — `ZipFile.write` is non-deterministic; use ZipInfo + writestr with fixed timestamp. R5-r6 #5.

**Additive 2026-04-27 (12 new, NOT YET wired into `preflight_all()`):**

A. `check_vastai_create_has_label` — Every `vastai create instance` call must pass `--label`. Orphan instances accrue cost silently (today: instance 35707822, ~$0.05 wasted). Live count: 0.
B. `check_vastai_create_writes_tracker` — Every Vast.ai launch must register the instance ID to `.omx/state/vastai_active_instances.json` so cleanup scripts can detect orphans. Live count: 2.
C. `check_subagent_prompts_no_cpu_fallback` — Subagent prompts must not allow `--device cpu` without a `deterministic-bytes acceptable` caveat. CPU fallback in byte-deterministic build = invalid archive. Live count: 1.
D. `check_scores_have_lane_tag` — Every numeric score in `run_log.md`/`findings.md`/`BATTLE_PLAN.md` must carry a lane tag (`[contest-CUDA]`, `[advisory only]`, `[MPS-PROXY]`, …). MPS-CUDA drift = 23x. Live count: 20.
E. `check_waivers_specify_env_gate` — `# SCORER_AT_INFLATE_WAIVED` markers must name an env-gate (`env-gated-INFLATE_TTO=1`) so operators can audit which env-vars enable scorer-at-inflate paths. Live count: 0.
F. `check_halfframe_archive_uses_trained_profile` — `--half-frame` archive builds must use a renderer trained for it (profile with `mask_half_sim_prob>0` OR `use_zoom_flow=True`). Memory: `feedback_half_frame_breaks_posenet`. Verified 2026-04-27 score 17.55. Live count: 2.
G. `check_profile_keys_have_resolvers` — Bidirectional companion to dead-resolver scanner: every PROFILES key must be consumed somewhere in src/tac or experiments. Live count: 91 (real cleanup target).
H. `check_inflate_scorer_load_has_runtime_banner` — Inflate files loading scorers must `print('[strict-scorer-rule] ...')` at runtime so the score can be tagged `[scorer-at-inflate-noncompliant]`. Live count: 0.
I. `check_test_files_imports_resolve` — Test files importing from `tac.*` must resolve to actual symbols. Existing dead-import scanner skips test dirs; this complement catches broken tests that silently skip at collection. Live count: 25.
J. `check_vastai_prompts_have_cost_cap` — Subagent prompts mentioning Vast.ai must mention a `$` cap, `budget`, or `destroy instance`. Memory: `feedback_vastai_cost_paranoia`. Live count: 0.
K. `check_uniward_delta_has_attestation_gate` — `--with-uniward-delta` invocations must include `--allow-pending-compliance` OR an attestation file reference. Lane C R5 (commit ef8a9a1b). Live count: 6.
L. `check_remote_scripts_write_provenance` — Every `scripts/remote_*.sh` must write `provenance.json`. Memory: `feedback_canonical_remote_bootstraps`. Live count: 5 (Lanes A/B/D/G).

**Lane maturity registry (2026-04-30, STRICT @ 0):**

90. `check_lane_registry_consistent` — `.omx/state/lane_registry.json` must parse, schema_version must match, no duplicate lane ids, every lane has all 7 gates present, stored `level` matches computed-from-gates, every file-path-looking evidence string points to a real file. Mutations only via `tools/lane_maturity.py mark/unmark/add-lane`. Memory: `project_lane_maturity_harness_landed_20260430` + `feedback_production_hardened_standard_definition_20260430`. Live count: 0.

**Promotion path:** A new check starts `strict=False`, the live violation count is fixed across the codebase, then it is flipped `strict=True` in `preflight_all()`. The promotion pattern is documented in commit 7f2740e4 (the strict-flip of checks 1-11). Reverting any strict check will fail at commit/PR/run time.

## Lane maturity registry — non-negotiable

Every lane MUST be registered in `.omx/state/lane_registry.json` via `tools/lane_maturity.py`. Claiming Level 2 or Level 3 without a corresponding `mark` command is FORBIDDEN. Subagents shipping a lane MUST run `python tools/lane_maturity.py mark <lane> --gate <gate> --evidence <path>` for each gate they hit. Preflight Check 90 (`check_lane_registry_consistent`) fails STRICT if the registry is inconsistent (level/gates mismatch, duplicate ids, missing gates, file-path evidence pointing to non-existent files).

The 7 gates and their meaning are defined in `feedback_production_hardened_standard_definition_20260430`:

1. `impl_complete` — production code lands; no `NotImplementedError`; all CLI flags wired
2. `real_archive_empirical` — real-archive empirical measurement on Lane G v3 anchor (or equivalent); tagged `[empirical:<artifact>]`
3. `contest_cuda` — actual `[contest-CUDA]` score (Vast.ai 4090 / Modal A100); NEVER MPS, NEVER `[contest-CPU advisory]`
4. `strict_preflight` — STRICT preflight check covering the lane's bug class
5. `three_clean_review` — 3-clean-pass adversarial review counter at 3/3
6. `memory_entry` — memory file documenting the empirical result + cross-refs
7. `deploy_runbook` — `scripts/remote_lane_<id>.sh` + heartbeat + watchdog + harvest

Computed level: 0 = 0 gates, 1 = ≥1 gate, 2 = `impl_complete` AND `real_archive_empirical` true, 3 = ALL 7 true. A lane with 4 gates true but missing `impl_complete` is still Level 1 (not Level 2). The CLI computes this for you — never set `level` by hand.

CLI usage:

```bash
# Audit table
python tools/lane_maturity.py audit

# Mark a gate satisfied
python tools/lane_maturity.py mark lane_g_v3 --gate contest_cuda \
    --evidence "1.05 [contest-CUDA] reports/raw/2026-04-29-..."

# Validate (preflight Check 90 also runs this)
python tools/lane_maturity.py validate

# Regenerate reports/lane_maturity.md
python tools/lane_maturity.py report

# Register a new lane at Level 0
python tools/lane_maturity.py add-lane lane_new --name "New Lane" --phase 2
```

Every mutation appends a JSONL record to `.omx/state/lane_maturity_audit.log` for forensics.

**Lifecycle discipline (non-negotiable):**

- **Pre-registration is mandatory.** The moment a lane has a name and a council/design verdict — even if it's only a sketch — it MUST be `add-lane`'d at Level 0. This includes in-flight subagent lanes, future-design lanes, and forensic-investigation lanes. Pre-registration enables the audit table to distinguish IN-FLIGHT vs LANDED vs SKETCH.
- **Mark gates as evidence is produced, NOT after-the-fact.** The moment a council Round-N CLEAN landing happens, mark `three_clean_review`. The moment a remote_lane script lands, mark `deploy_runbook`. Batch-backfilling stale evidence is a code smell.
- **KILLED lanes get registry entries too.** Mark with `--gate three_clean_review --evidence "<council ref>"` and add `--notes "Reactivation: <criteria>"`. Do NOT just exclude killed lanes — the registry is the single source of truth for what we have considered, including kills.
- **Backfill-when-discovered is acceptable.** When this rule is violated by an earlier subagent, a maturity-discipline pass that backfills evidence is the correct remedy. The audit log records the backfill timestamp; no harm done.
- **Lifecycle: SKETCH (L0) → SCAFFOLD (L1) → INTEGRATION (L2) → FULL PRODUCTION HARDENED (L3).** The ONLY currently-Level-3 lane is `lane_g_v3` (1.05 [contest-CUDA]). That fact is the standard-bearer for the rest of the registry.
- **Audit before commit.** Before any commit that adds a lane or marks a gate, run `python tools/lane_maturity.py validate` — Check 90 STRICT enforces this at commit time, but catching it earlier is cheaper than a re-stage.
