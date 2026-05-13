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

## HNeRV / leaderboard-implementation parity discipline — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** `feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md` (operator-mandated retrospective, 2026-05-09). Cross-refs `.omx/research/representation_integration_gap_audit_20260508_codex.md` (codex parallel finding) + `feedback_substrate_vs_codec_composition_meta_pattern_20260508.md` (claude-side framing) + `feedback_grand_council_fields_medal_phase2_floor_REBASELINE_with_integration_discipline_20260509.md` (Phase 2 floor rebaseline).

**The 2026-04-30 → 2026-05-04 retrospective:** we had HNeRV/MNeRV/NeRV/SIREN/Cool-Chic/C3 representation primitives in the repo BEFORE PR #95/#100/#101/#103 ever hit the contest. We never got sub-0.20 with them. The leaderboard HNeRV-family won not because of architectural novelty but because PR #100's hnerv_lc_v2 (268 LOC) bound architecture + score-aware training + archive grammar + inflate runtime + export contract simultaneously, and PR #101 (337 additional LOC of entropy bolt-ons) won gold at 0.193 by stacking on the verified substrate. Each layer was reviewable in 30 seconds.

Our internal NeRV/HNeRV/Cool-Chic/C3 work had every architectural ingredient but never bound them simultaneously. The integration loop was always 5-7 separate research artifacts that never converged into a single packet. Lane 12 NeRV mask codec targeted the WRONG slot (mask only, not full RGB renderer). Cool-Chic / C3 hit the FP4A export gate AFTER training, not before.

### The 13 inviolable lessons

Every representation/codec lane (NeRV / HNeRV / Cool-Chic / C3 / wavelet / VQ-VAE / grayscale-LUT / SIREN / coordinate-MLP / hyperprior / nonlinear-transform-coding / time-varying-FiLM / shared-codebook / etc.) MUST honor all 13 of these from byte zero:

1. **Substrate must be score-aware.** Train against the contest's actual `upstream/videos/0.mkv` with gradient-through-SegNet/PoseNet, not extracted masks, not L²/KL on raw frames, not synthetic data. Default loss `((mask_pred - mask_gt) ** 2).mean()` is FORBIDDEN as the primary training signal for any representation entering the archive.
2. **Export-first design.** Declare the archive grammar + parser-section manifest BEFORE writing the training script. If the variant cannot export into the contest packet format (e.g., Cool-Chic / C3 non-FP4A), the run is research-only by construction; tag `research_only=true` and ungate `--auth-eval-on-best` only after the export contract lands.
3. **Archive grammar = monolithic single-file `0.bin`** (or explicitly justified multi-file). Fixed offsets declared in `codec.py` source (e.g., `DECODER_BLOB_LEN = 162_164`, `LATENT_BLOB_LEN = 15_387`). ZIP-member-budget rows are invalid unless the packet really has separate ZIP members.
4. **Inflate.py ≤ 100 LOC** (default budget; explicit waiver for ≤ 200 with rationale). ≤ 2 external dependencies declared in the runtime tree. CUDA-or-CPU agnostic. Reviewable in 30 seconds.
5. **Architecture must be the FULL renderer** (RGB out), not a single-component slot (mask only / pose only). The contest scorer derives masks from frames; replacing the masks slot is dominated by replacing the frames the masks are derived from. Lane 12-style "mask codec only" lanes are DEFERRED-pending-research-with-renderer-rescope.
6. **Score-domain Lagrangian** (not weight-domain proxies like rel_err²). The Lagrangian must be `α·B(θ)/N + β·d_seg(θ) + γ·√d_pose(θ)` with `d_seg` and `d_pose` computed via the actual scorer (or a Hinton-distilled co-trained surrogate per Phase 2 / Phase 3). rel_err²-as-objective is FALSIFIED at rms ≥ 0.04 per `feedback_three_lossy_anchors_show_rel_err_squared_objective_falsified_20260508.md`.
7. **Bolt-on size ≤ 350 LOC** (substrate engineering may exceed; tag `lane_class=substrate_engineering` explicitly). Substrate engineering happens ONCE per architecture class; bolt-ons happen many times. PR101 was 605 total LOC = 268 substrate + 337 bolt-on. The kitchen_sink anti-pattern (PR105: 1776 LOC, 21 files, LOST to rem2's 241 LOC silver) is what happens when you violate this.
8. **Eval-roundtrip-aware and differentiable scorer-preprocess training.** The uint8 bottleneck (384 → 874 → uint8 → 384) MUST be simulated in the proxy loss. `eval_roundtrip=False` produces 2-11x proxy-auth gap and is FORBIDDEN per existing CLAUDE.md non-negotiable. The scorer preprocess must also be gradient-reachable: PR #95/#106 monkey-patched `rgb_to_yuv6` because the upstream challenge helper is `@torch.no_grad()` / in-place and otherwise severs PoseNet gradients. New NeRV/HNeRV/Cool-Chic/C3 renderer trainers need a PoseNet/SegNet gradient-reachability check before GPU dispatch.
9. **Runtime closure.** Run the exact contest `inflate.sh` signature in a clean environment BEFORE dispatch. Dependency closure failures (missing brotli, wrong wrapper signatures, hidden sidecars, local paths, CPU/CUDA mismatches) are runtime blockers, not method negatives. PR106 belt_and_suspenders FAILED its first replay due to missing `brotli` — exactly this bug class.
10. **Mask/pose coupling gate.** Any mask change requires pose regeneration + geometry diagnostics + decoded mask SHA-256s + mask disagreement record. Smaller mask bytes alone are insufficient.
11. **No-op detector.** Prove the targeted bytes changed AND were consumed by inflate. Reuse, decode/re-encode, provenance-only changes, and cosmetic ZIP repacks stay forensic until this proof exists.
12. **Single-LOC-per-LOC review discipline.** Every line in the bolt-on must be reviewable in 30 seconds. PR101's `codec.py` is 480 LOC of pure codec code (no training scaffold, no profile dispatch, no smoke/full mode flags). Our internal `nerv_mask_codec.py` is 1000+ LOC and includes coordinate sampling + training scaffolds + sample components + magic-byte versioning + ... — NOT a packetized codec.
13. **KILL/FALSIFIED is LAST RESORT.** Per the existing CLAUDE.md non-negotiable: if a representation lane returns negative, the default verdict is DEFERRED-pending-research-with-XYZ-applied with reactivation criteria, not KILLED. Lane 12 NeRV is DEFERRED-pending-renderer-rescope; Cool-Chic / C3 are DEFERRED-pending-export-design.

### The 8th forbidden pattern (named here)

**Forbidden representation-without-archive-grammar (the "research-substrate trap"):**

Building a representation (NeRV / Cool-Chic / C3 / wavelet / VQ-VAE / grayscale-LUT / SIREN / coordinate-MLP / hyperprior / etc.) WITHOUT simultaneously building (a) the `archive.zip` builder that emits scored bytes, (b) the `inflate.sh` runtime that reads them, (c) the parser-section manifest that locates them, (d) the export contract that converts trained weights → archive bytes, and (e) the score-aware training loop that backprops through SegNet/PoseNet on the contest video — is a research-only path by construction. The bytes never enter the contest packet; the score never moves.

This is the dominant representation-lane integration meta-bug from the 2026-04-30 → 2026-05-04 gap. It does not by itself explain the full miss: the postmortem also requires (a) failure to consume PR #95's open training stack during the race window, (b) failure to measure the CPU public-leaderboard axis early enough, and (c) missing differentiable scorer-preprocess training in our NeRV/HNeRV loops. STRICT preflight check #124 (`check_representation_lane_has_archive_grammar_at_design_time`) enforces the archive-grammar part; trainer-specific grad-reachability guards must cover the scorer-preprocess part.

### Five forbidden code patterns

1. **Forbidden NeRV-style coordinate MLP that targets the masks.mkv slot without rescope to the renderer.** Lane 12 mistake. If your representation's output shape is `(T, H, W, 5)` of mask logits and not `(T, 3, H, W)` of RGB frames, the lane is DEFERRED-pending-renderer-rescope.

2. **Forbidden `--auth-eval-on-best` gate bypass for non-FP4A export variants.** Cool-Chic / C3 mistake. `train_renderer.py:2099-2122` blocks `--auth-eval-on-best` for variants that lack full archive/export support — this is correct fail-closed behavior. NEVER add a workaround that runs auth eval against a non-exportable variant; instead, land the export contract first.

3. **Forbidden `make_synthetic_pair_batch` calls in any non-smoke training path.** Per `feedback_codex_finding_pr101_synthetic_targets_FIXED_20260508.md`. Train against `upstream/videos/0.mkv` decoded via pyav, not random Gaussian noise. Smoke-only mode does not generalize to non-smoke.

4. **Forbidden representation-lane Level 1+ promotion without `archive_grammar` / `parser_section_manifest` / `inflate_runtime_loc_budget` / `runtime_dep_closure` / `export_format` / `score_aware_loss` / `bolt_on_loc_budget` / `no_op_detector_planned` declared in lane-registry evidence.** STRICT preflight check #124.

5. **Forbidden cross-archive composition (HStack/VStack/cross-paradigm) without a single verified [contest-CUDA] substrate anchor.** Per substrate-vs-codec meta-pattern. T9 (cross-archive multi-substrate composition) is the kitchen_sink anti-pattern under a new name. DEFER until a verified composable substrate exists; or re-scope to single-axis branching from the ONE verified score-aware substrate (currently A1).

### Enforcement

- STRICT preflight check #124 `check_representation_lane_has_archive_grammar_at_design_time` lands warn-only initially; flip to STRICT after in-flight Phase 2 lanes (T1/T6/T10/T15/T17/T18) backfill the blueprint.
- `tools/lane_maturity.py` audit refuses to mark a representation lane as Level 1+ without the 8 declared fields.
- Council review of any new representation/codec lane MUST cite this section and walk through all 13 lessons.
- Memory file `feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md` is the canonical retrospective; future agents should re-read it before starting any new representation lane.

## Apples-to-apples evidence discipline — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** operator correction, 2026-05-10, after PR103 histogram packet review.

Never classify, promote, retire, or submit a HNeRV/public-frontier result from
an inferred equivalence. Every score conclusion needs an apples-to-apples
baseline on the same axis and runtime contract.

Hard rules:

1. **Decoded-state parity is not frame parity.** Identical decoded
   `state_dict` / latents / symbols proves parser consumption only. It does
   not prove same rendered RGB bytes, scorer components, CUDA numerics, or
   public-leaderboard behavior. A manifest may say
   `decoder_state_parity_passed=true`, but it MUST keep
   `full_frame_inflate_output_parity_missing` until source-vs-candidate
   `inflate.sh archive_dir output_dir file_list` outputs are compared
   byte-for-byte or exact same-runtime evals are available for both packets.
2. **CPU and CUDA are separate evidence spaces.** The HNeRV cluster often
   scores much better on `[contest-CPU]` than on `[contest-CUDA]`. CPU
   medal-band proximity is real public-axis evidence, but it is not CUDA
   readiness, CUDA frontier status, or a conversion shortcut. Never infer one
   axis from the other; run both when shipment/frontier language is used.
   Do not invert this into a universal "CPU is better" rule either: every
   packet must be measured per archive/runtime/inflate-device/evaluate-device,
   with inflated raw-output hashes and PoseNet/SegNet component deltas when
   diagnosing the mechanism.
3. **Source runtime must match the comparison.** For public PR clones, compare
   original archive + original `inflate.sh` against candidate archive +
   candidate runtime under the same evaluator path. If the candidate adapter
   changes `inflate.py`, `inflate.sh`, Python invocation, dependency closure, or
   section constants, the source replay used as baseline must be the matching
   source runtime, not a nearby repack or previous Modal/Lightning artifact.
4. **Negative exact evals need harness review before method verdicts.** If a
   byte transform preserves decoded tensors but exact eval changes, default to
   `indeterminate-harness-or-runtime-mismatch` until full-frame output parity,
   same-runtime source replay, and component recomputation agree. Do not call it
   a method negative just because a CUDA number returned.
5. **Generated reports must preserve the axis label.** Phrases like
   "rounds to gold", "medal-band", "submission-ready", "auto-promote", and
   "score gap" must include `[contest-CPU]`, `[contest-CUDA]`,
   `[macOS-CPU advisory]`, or `[proxy]` inline. Missing axis label is a bug.

When in doubt, downgrade the finding, write a supersession ledger, and run the
apples-to-apples proof before spending another dispatch.

## Bugs must be permanently fixed AND self-protected against — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** operator directive (2026-05-09): *"such bugs must be permanently fixed and self-protected against."*

Every adversarial-review finding (codex / grand council / sister subagent) that surfaces a real bug MUST be addressed with TWO landings, NOT ONE:

1. **The fix** — patches the immediate code surface
2. **A STRICT preflight check** — refuses any code surface in the repo that re-introduces the bug class, with a `check_<bug_class_name>` function in `src/tac/preflight.py`, wired into `preflight_all()`, with dedicated tests

Single-surface fixes are insufficient. Per the META-meta finding from a8bc7e79's proactive sweep: bug classes have **6-7× spread** across the repo. A fix at one surface leaves the same class active at 6 others.

### The codex-review fix-with-strict-preflight pattern (canonical)

For each codex review finding (HIGH or MEDIUM):

1. **Patch the cited file:line** with the recommended fix
2. **Claim a catalog #** via `tools/claim_catalog_number.py claim`
3. **Add a STRICT preflight check function** `check_<bug_class>` to `src/tac/preflight.py`:
   - Scans the targeted directories (`src/tac/`, `tools/`, `experiments/`, `scripts/`) for the bug-class signature
   - Allows opt-out via same-line `# <BUG_CLASS_OK>:<rationale>` waiver
   - Raises `PreflightError` in strict mode; warns in non-strict
4. **Wire into `preflight_all()`** — initially `strict=False` (warn-only)
5. **Write 15-25 dedicated tests** covering: positive (catches violation), negative (allows non-violations), waiver-respect, edge cases
6. **Verify live count = 0** by running the check strict against the current repo state
7. **Strict-flip the wire-in** to `strict=True` once live count = 0
8. **Add a row to the CLAUDE.md "Meta-bug class catalog (strict-mode preflight)" table** with catalog #, name, what it prevents, memory ref

### Strict-flip atomicity rule

If the fix subagent achieves live count = 0 in the same landing, the strict-flip should land in the SAME commit-batch (not a follow-up). This avoids the warn-only-purgatory failure mode where a check ships warn-only and the strict-flip never happens.

### Examples from this session (canonical pattern)

- Catalog #123 (`check_no_weight_domain_saliency_on_score_gradient_substrate`) extincts the Track 4 v1 Fisher-proxy-inversion bug class
- Catalog #124 + #125 + #126 + #127 + #128 + #130 + #131 — 7 META gates landed across the session, all following this pattern
- a00501f9's round-3 fix MUST land Catalog #132 (`check_locked_writes_preserve_deletions`) per the same pattern, plus #133/#134/#135 for HIGH 2 / MEDIUM 1 / MEDIUM 2 (one-strict-check-per-finding)

### Anti-pattern: fix-without-self-protection

Any commit that fixes a codex finding WITHOUT landing the corresponding STRICT preflight check is INCOMPLETE. The reviewer should reject the commit on the grounds that the bug class will re-emerge at a different surface.

## Subagent coherence-by-default — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** operator concern (2026-05-09): *"i am concerned we are building intelligent systems but they are not coherent and integrated and maybe duplicate ... the should just work and run in the background for us perhaps as skills or via mcp tools or something i'm not sure how to solve this problem ... or maybe just engineer correctly and then save related knowledge and instructions in claude and agents .md."*

**The answer is the latter**: don't add another orchestration layer (skills, MCP). Engineer the right primitives, save the discipline in CLAUDE.md + AGENTS.md, and EVERY future subagent honors it without an orchestrator. The non-negotiables in this file ARE the orchestration layer — they propagate via every subagent's mandatory pre-read.

### Mandatory pre-flight for every subagent (parent + nested)

Before starting any work, every subagent MUST:

1. **Read CLAUDE.md AND AGENTS.md** — both files. Honor every NON-NEGOTIABLE marker. The "I didn't read it" failure mode is a process bug, not an information gap.
2. **Check the lane registry** (`.omx/state/lane_registry.json`) for in-flight conflicts. If your lane shares a `lane_id` or `target_modes`/`deployment_target` with an active claim, coordinate via the file's notes column or pick a different lane.
3. **Check sibling subagents in the same conversation** — when the parent prompt says "running in parallel right now", read the listed sibling subagent IDs and their scopes. Do NOT duplicate their primary deliverable.
4. **Read latest top-of-MEMORY.md entries** — at least the top 10. Recent landings change the optimal next-step.
5. **Read all `.omx/research/*_directive_*` files** dated within the last 24 hours — they contain operator-routed inter-subagent directives that supersede the original prompt.

### Mandatory wire-in for every landing (no orphaned signals)

Every landing must wire its outputs into the unified solver stack OR explicitly tag `research_only=true`. Per `feedback_unified_lagrangian_action_principle_GR_style_20260509.md`:

1. **Sensitivity-map contribution** in `tac.sensitivity_map.*` (or sibling)
2. **Pareto constraint** added to `tac.pareto_*` (or explicitly reasoned why non-binding)
3. **Bit-allocator hook** registered if per-tensor importance changes
4. **Cathedral autopilot dispatch hook** registered if archive-deployable
5. **Continual-learning posterior update** triggered on every empirical anchor
6. **Probe-disambiguator** built if 2+ defensible interpretations exist (`tools/probe_<track>_disambiguator.py`)

If any of the 6 hooks is N/A, declare it explicitly in the landing memo with rationale. **Silent omission is the orphan-work failure mode.**

### Anti-duplication primitive: the lane registry IS the deduplication layer

Two subagents working on the same lane is a registry failure, not a coordination failure. The fix is:

1. Pre-register every lane (even SKETCH at L0) the moment a name + verdict exists, per CLAUDE.md "Lane maturity registry" non-negotiable lifecycle discipline.
2. Subagent prompts MUST cite the registered `lane_id` in the prompt body so collisions surface at parent-coordinator review time.
3. The `tools/lane_maturity.py audit` table is the single source of truth for "what's currently being worked on." Use it.

### Anti-fragmentation primitive: the unified-Lagrangian action

Per `feedback_unified_lagrangian_action_principle_GR_style_20260509.md`, the migration target is `tac.unified_action.S_total(theta, archive_bytes, hardware)` — ONE scalar action, all track-Lagrangians composed via δS/δθ = 0 (GR-style variational principle). Until that lands, individual track wire-ins must explicitly call all 6 integration hooks above.

When the unified action lands, every track plugs in by adding a term to `S_total` — no new orchestration layer. The coherence is structural.

### Anti-arbitrariness primitive: the probe-disambiguator pattern

Per `feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md`, when a design choice has 2+ defensible interpretations, ship BOTH modes via callable interface + build `tools/probe_<track>_disambiguator.py` that returns the regime-conditional verdict. The probe IS the arbitration; the trainer/codec/solver consumes the verdict.

### Background-execution clarification

The operator floated "skills or MCP tools" as the orchestration mechanism. **Do not pursue this path.** The CLAUDE.md + AGENTS.md non-negotiables ARE the always-on, zero-token orchestration layer. Every subagent loads them by default; every behavior is encoded in inviolable rules. Adding another layer would be the kitchen_sink anti-pattern at the meta level.

If a behavior should be automatic across all sessions, write it as a CLAUDE.md non-negotiable. The skill-vs-rule decision: **skills are user-invocable patterns; rules are agent-binding contracts.** The coherence problem is solved by RULES, not skills.

### Concrete enforcement

- New STRICT preflight check planned: `check_subagent_landing_has_solver_wire_in` — refuses any landing memo that doesn't declare all 6 wire-in hooks (or `research_only=true`).
- New STRICT preflight check planned: `check_lane_pre_registered_before_work_starts` — refuses subagent commits whose `lane_id` doesn't appear in the registry.
- The existing Check 90 `check_lane_registry_consistent` partially covers this; the two new checks extend it to subagent-discipline territory.

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

## Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS

The meta-Lagrangian, Pareto, field-equation, and cross-paradigm selector stack
is a living solver, not a one-off planning report. Any work on score lowering,
stacking, entropy coding, hidden gems, public PR deconstruction, categorical
labels, foveation, pose, sensitivity, or paradigm wiring must either improve
this solver or explicitly record why the new signal is not yet actionable.

Continuously improve the solver toward learnable and solvable theoretical-floor
discovery. No signal loss: keep exact CUDA outputs, archive bytes/SHA,
runtime-tree hashes, commands, assumptions, negatives, calibration residuals,
and cross-paradigm rows machine-readable so future agents can reseed the
planner without reverse-engineering prose.

Every stackable or substitutive idea should move toward a typed row consumed by
the planner: candidate id, family, pareto scope, charged bytes, predicted
SegNet/PoseNet/rate deltas, uncertainty, evidence grade, archive/runtime
custody, interaction assumptions, conflicts, Volterra or higher-order terms,
KKT/ADMM residuals, expected information gain, blockers, and next proof. If a
research artifact can affect score but is not visible to the selector, it is
orphaned work.

Prefer solvable math over arbitrary sweeps. New knobs must be grounded in
entropy/MDL, Fisher/Hessian/Jacobian or Frechet sensitivity, Dykstra/ADMM
feasibility, Bayesian experimental design, optimal transport/camera geometry,
component-response evidence, or a documented ablation. Heuristics stay tagged
`planning_only` until evidence closes the loop. Every exact CUDA result and
high-quality negative should reseed calibration, Pareto constraints, trust
regions, interaction terms, or strict guards.

Every returned result must receive adversarial custody review before it changes
lane status or solver routing. Record the archive bytes/SHA, runtime-tree SHA,
command, hardware, sample count, structured JSON/log path, dispatch-claim
state, recomputed score components, payload-consumption proof, failure class,
and reactivation criteria. A bad result retires only the measured config unless
research-path exhaustion plus consensus review supports a broader conclusion.
Proxy, MPS, non-`contest-CPU` CPU, byte-only, or stale results may seed priors
and TODOs, but they must not promote, rank, falsify, kill, or close a family.
`contest-CPU` ranks only the public leaderboard CPU axis; it does not replace
the CUDA axis or justify extrapolating a missing paired result.

The desired loop is: formulate objective and constraints -> emit typed atoms ->
Pareto/KKT/interaction prune -> select by score delta plus expected information
gain -> build deterministic archive -> exact CUDA eval and exact contest-CPU
eval when the archive is a frontier/submission candidate -> reseed the solver.
Keep this path simpler, faster, more deterministic, and more complete every
time it is touched.

Planner recipes and dispatch snippets must use the current tool surfaces. Grep
the real argparse/help contract before writing or invoking any flag, and record
a blocker or add a reviewed interface when the solver needs a capability the
tools do not expose. Never invent flags, schema keys, or evidence fields to
make a theoretical plan look executable.

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
Reporting any score that did not come from `upstream/evaluate.py` on the EXACT archive bytes that will be submitted. No proxy MSE. No MPS. No "looks reasonable" extrapolation. Tag every reported score by axis: `[contest-CUDA]` for CUDA promotion truth, `[contest-CPU]` for explicit public-leaderboard reproduction, or `[advisory only]` for everything else. (See `feedback_proxy_auth_math_useless`, `feedback_mps_cuda_drift_critical`.)

**Forbidden component-aliasing for baselines:**
Treating a directory of components as the "baseline" without verifying every file SHA against the archive ZIP that produced the baseline score. Components from different lanes leak into the same dir; SHA-vs-archive is the only check. (See `feedback_phantom_baseline_pattern`.)

**Enforcement:** Before typing ANY of the above patterns, STOP. The non-negotiable wins over the convenient default. This list is in CLAUDE.md so it is loaded into context at session start; if I write one of these patterns anyway, that is a process failure, not an information failure.

**Forbidden empirical-claim-without-evidence-tag (the docstring-overstatement trap):**
Writing "saves 49%" / "improves N%" / "beats baseline" / "verified" in a docstring/report/script without an adjacent `[empirical:<artifact path>]` or `[contest-CUDA]` or `[prediction]` tag. Lane PD docstring stated 49% savings; empirical regression test caught actual 18.5%. The 49% was a derivation, not a measurement. Tag every claim. (See `feedback_three_active_bug_classes_needing_strict_checks_20260429.md`.)

**Forbidden fix-lands-in-helper-but-not-callsite (the dangling-helper trap):**
Adding a kwarg to a helper without grepping for callers and updating each. Lane GP added `baseline_poses=` to `reconstruct_poses()` but the actual call at `experiments/fit_pose_gp.py:33` never passed it for ~2 weeks. After adding any kwarg with non-trivial semantics, register it in `CALLSITE_CONTRACTS` and run the AST scanner to enforce all callers pass it. (See `feedback_three_active_bug_classes_needing_strict_checks_20260429.md`.)

**Forbidden MPS-derived strategic decision (the MPS-falsification trap):**
Writing "GREEN" / "RED" / "KILL" / "promoted" / "FALSIFIED" in any record where the supporting evidence is an MPS or non-`contest-CPU` CPU forward pass through SegNet/PoseNet/renderer/distilled scorer. MPS PoseNet drift is 23×; SegNet 2×; score 2.5×. STC clean-source FALSIFICATION was made on MPS encoder; user correctly objected; withdrawn. Internal CUDA promotion/kill decisions REQUIRE a `[contest-CUDA]` artifact in the same record/section. Public-leaderboard CPU claims REQUIRE `[contest-CPU]` on exact archive/runtime custody, and still must record the missing paired CUDA/CPU axis instead of extrapolating it. (See `feedback_no_local_mps_for_authoritative_kill_or_promote_20260429.md`.)

**Forbidden /tmp paths in any persisted artifact (the transient-evidence trap):**
Writing `/tmp/<anything>` as a durable evidence path in: lane registry evidence strings, dispatch claims, commit messages, build metadata, score/rebuild manifests, runbooks, or any artifact that another agent may use to reproduce a result. /tmp paths do NOT survive a fresh checkout, do NOT exist on remote/CI/cloud machines, and CANNOT be verified by other agents. They produce phantom "evidence" that points at nothing. User mandate 2026-05-05: "we need to stop using /tmp by principle". Forensic finding: `lane_pr106_stacked` was marked L2 with `real_archive_empirical:true` evidence pointing at `/tmp/pr106_stacked_smoke/stacked_full/pr106_stacked_archive.zip` — a path that doesn't exist on any other machine and would be lost on shell exit. **Canonical replacement**: `experiments/results/<lane_id>_<timestamp>/` for build artifacts; `.omx/state/` for ledgers; `.omx/research/` for durable analyses; `.omx/tmp/` for explicitly ephemeral local scratch. Historical transcripts may mention `/tmp` only as scratch-only, non-evidence context; they must not be cited as reconstructable custody. Caught by `tools/check_lane_smoke_signal_nontrivial.py` (PCC9, transient_tmp_evidence detection).

**Forbidden artifact-lifecycle violations (the provenance-vs-state confusion meta-class — codex 2026-05-08, Catalog #113):**
Five surface findings (operator-approval-leak, public-PR-clone-dirty, status.json-stale-dirty, rebuild_command-baked-timestamps, recovery_metadata-mutated-in-place) all share ONE structural class: transient/global/upstream state being frozen or mutated into committed/forensic artifacts. Four-kind taxonomy enforced via `src/tac/artifact_lifecycle.py` + `.omx/state/artifact_kind_registry.yaml` + `check_artifact_lifecycle_compliance` umbrella gate. Specifically forbidden: (1) **committing transient state into LIVE_STATE files** — files matching `LIVE_STATE` patterns must be gitignored (e.g., locks, fs caches, vastai_active_instances.json); (2) **mutating HISTORICAL_PROVENANCE files** — `recovery_metadata.json`, `lane_maturity_audit.log`, dispatch claims, contest auth eval JSONs are append-only; field mutation outside registered `append_fields` is FORBIDDEN; (3) **baking transient values into LIVE_RECIPE files** — no hardcoded `--now-utc <ISO>`, `--operator-approved-*`, durable `/tmp/...` evidence paths, or hardcoded Vast.ai instance IDs in `rebuild_command.txt`/`scripts/*.sh`/`tools/*.sh`/`inflate.sh`; use `${PARAMETER}` placeholders OR add explicit `HISTORICAL_RECIPE_ONLY` header; (4) **stale session state in DERIVED_OUTPUT bodies without regeneration header** — `status.json`/`reports/latest.md`/dashboards must declare `generated_at: <utc>` + `from_state_hash: <sha>` within first 4 KB so consumers know it was regenerated, not snapshotted. Per CLAUDE.md "Operator gates must be wired and used" — `check_artifact_lifecycle_compliance` is wired into `preflight_all(strict=True)`. Reactivation criteria: every long-lived artifact in the repo explicitly classified in registry. Memory: `feedback_codex_findings_meta_pattern_artifact_lifecycle_FIXED_20260508.md`.

**Forbidden premature KILL without research exhaustion (the kill-too-fast trap):**
Writing "KILL" / "FALSIFIED" / "DEAD" / "RETIRED" as a final verdict on a lane based on a SINGLE empirical configuration's failure, when plausible alternative configurations have NOT been attempted. apogee_int4 NAIVE-PTQ falsification 2026-05-05 was initially recorded as KILL/FALSIFIED at score 1.4287 [contest-CUDA T4] WITHOUT trying QAT, LSQ, per-channel scaling, smaller block sizes, or outlier handling -- all canonical fixes for low-bit PTQ collapse. User caught it: "we must only kill as a last resort after exhausting all research and grand council consensus" (2026-05-05), reinforced 2026-05-08 as "always investigate all results that come back deeply and adversarially and rigorously and only falsify and kill as an absolutely last resort." Default verdict for one-config failure is **DEFERRED-pending-research** or **measured-config retired**, NOT KILLED. KILL conversion requires (a) every plausible alternative config attempted empirically, (b) exact custody/recomputation/failure classification for the returned result, (c) **grand council CONSENSUS** (not just majority — every inner-ten member endorses), (d) reactivation criteria documented. Memo filename uses `_DEFERRED_pending_<reason>_<date>.md`, not `_killed_*.md`. See expanded "KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE" section above for full enforcement.

**Forbidden re-implementing remote bootstrap inline (the duplicated-bootstrap trap):**
Writing `curl -LsSf https://astral.sh/uv/install.sh | sh` or `apt-get install ffmpeg` or `find upstream -name '._*' -delete` directly in any new chain driver / lane script / one-off. There is ONE canonical bootstrap function: `bootstrap_runtime_deps()` in `scripts/remote_archive_only_eval.sh`, which delegates uv install to `scripts/ensure_remote_uv.sh`. Any new remote script MUST call that wrapper or `source` its bootstrap function — NEVER copy-paste the install commands. Cost of NOT doing this (2026-05-01 loop session): 6 sequential bug-class re-discoveries on 4 destroyed Vast.ai instances burning ~$1.50 + 30 min wall-clock chasing the same lesson. Memory: `feedback_remote_archive_only_eval_self_bootstraps_all_deps_20260501.md`. (Sister rule for venv: `python -m ensurepip --upgrade` is the standard fix for "venv exists but no pip" — see `scripts/ensure_remote_uv.sh` style.)

**Forbidden uv torch install without driver-version pin (the cu13-vs-cu124 trap):**
`uv run --with torch==2.5.1` (no local-version suffix) defaults to the LATEST CUDA wheel from PyPI (currently `+cu13`). On a Vast.ai host with NVIDIA driver < 580 (CUDA 12.x), the cu13 wheel will FAIL `torch.cuda.init()` and silently fall back to CPU — every score becomes `[advisory only]` per the MPS-falsification rule. The canonical pattern in `scripts/remote_archive_only_eval.sh:88-95` auto-pins:
- `driver_major < 580` (CUDA 12.x): `INFLATE_TORCH_SPEC=torch==2.5.1+cu124` + `UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124` + `UV_INDEX_STRATEGY=unsafe-best-match`.
- `driver_major >= 580` (CUDA 13.x): `INFLATE_TORCH_SPEC=torch==2.11.0` (default cu13 wheel works).

NEVER write `--with torch` (unpinned) in any new script. Always export `INFLATE_TORCH_SPEC` first or call the canonical wrapper. Cost (2026-05-01): instance 35957332 silently ran inflate on CPU for 5 min before detection. Memory: `feedback_vast_cuda_driver_too_old_silent_cpu_fallback_20260501.md`.

**Forbidden Vast.ai create without disk + cuda_vers gate (the chain-killer trap):**
Calling `vastai create instance ... --disk 30` for any chain that runs >1 candidate. The contest_auth_eval pipeline writes 3.6GB of inflated raw frames per candidate; a 6-candidate chain needs ≥30GB working set (5GB uv torch cache + 4GB rolling). 30GB hits the wall on candidate 4. Canonical defaults: `--disk 60`, `cuda_vers>=12.4` in the search filter, `--label <unique>`, register to `.omx/state/vastai_active_instances.json`. The chain driver MUST `rm -rf eval_work/{inflated,extracted,archive.zip}` after each successful eval — the canonical pattern is in `experiments/results/lane_g_v3_owv3_wave3_refinement_20260501/wave3_chain_driver.sh`. Memory: `feedback_remote_archive_only_eval_self_bootstraps_all_deps_20260501.md` outstanding-gaps section.

**Forbidden in-place edits to public PR intake clones (the corrupt-source-provenance trap):**
Adding ANY waiver comment, fix, or annotation directly inside `experiments/results/public_pr*_intake_*/{source,repo,pr*_src/repo}/...py`. Even comment-only additions (e.g., `# KL_BATCHMEAN_OK:public-PR-intake-not-our-quality-debt`) corrupt source provenance — `git -C <clone> status` becomes dirty, replay/audit no longer matches the public PR head, and preflight cleanliness becomes dependent on local-only edits absent from upstream. Codex 2026-05-08 review HIGH finding. **Canonical replacement**: scanners that scan `experiments/` already exclude `_intake_` paths via `_VENDORED_PATH_MARKERS` (`src/tac/preflight.py:5113`). For the rare scanner that cannot honor a path-prefix exclusion, record waiver rationale in `reverse_engineering/public_pr_waiver_manifest.json` (committed metadata) — never inline. Caught STRICT by `check_public_pr_intake_clones_pristine` (Check 109). Memory: `feedback_codex_finding_2_public_intake_pristine_FIXED_20260508.md`.

**Forbidden timestamp-only mutation of recovery_metadata.json (the recovery-custody-evidence-corruption trap):**
Overwriting `started_at_utc` / `completed_at_utc` / `elapsed_seconds` on an existing `experiments/results/recovered_*/recovery_metadata.json` without adding new substantive evidence (artifacts, logs, archive, ssh-state changes). The file is a forensic audit record; in-place timestamp churn destroys the original recovery-attempt timeline so future audits can't distinguish the original failed attempt from a fresh probe. Codex 2026-05-08 finding caught April 30 → May 8 churn on `recovered_42_dead` + `recovered_99999_phantom` with sub-millisecond elapsed and zero substantive change. **Canonical replacement**: the `attempts[]` append-only schema (v2_attempts) — every probe appends a new attempt entry keyed by unique `started_at_utc`; closed attempts are immutable. The writer (`tools/recover_lane_artifacts.py::_write_report`) refuses same-`started_at_utc` mutation of closed attempts via `RecoveryMetadataAppendOnlyError`. Non-initial attempts MUST carry `command_log_path` OR `substantive_change_from_prior_attempt` provenance. Caught STRICT by `check_recovery_metadata_append_only` (Check 110). Memory: `feedback_codex_finding_5_recovery_metadata_appendonly_FIXED_20260508.md`.

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

**The authoritative measurement loop is:** contest `inflate.sh` → `upstream/evaluate.py` on the EXACT archive bytes. Nothing else counts. Memory: `feedback_proxy_auth_math_useless`.

## Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE — NON-NEGOTIABLE

**Every submission archive (anything that will be PR'd to the contest, or that we use to claim "medal-band score" / "frontier") MUST get authoritative auth eval scores on BOTH `--device cuda` AND `--device cpu`, AND BOTH must run on hardware that is 1:1 contest-compliant with the contest's GitHub Actions CI runner.** The contest leaderboard ranks by the CPU eval, not the CUDA eval. Verified 2026-05-08:

- PR102 (third prize) public CUDA comment: 0.22839 — matches our T4 replay within 3×10⁻⁶
- PR102 (third prize) public CPU comment: **0.19538** — this is the medal-band score the prize was awarded against
- Δ CUDA−CPU: +0.033 on PR102

**Our PR #107 (apogee submission) was scored publicly only on CUDA at 0.22936; the maintainer never triggered a CPU eval comment.** Lab replay closed that blind spot with a GHA Linux x86_64 `[contest-CPU]` score of `0.1966358879` on exact archive/runtime custody. This confirms the submission was near the public medal cluster on the CPU axis, and it proves future shippable archives must be evaluated on both axes.

**1:1 hardware-compliance rule (NON-NEGOTIABLE):**

- Local macOS (M-series ARM, Intel iMac, anywhere on Apple Silicon or otherwise) is NEVER a 1:1 axis for CPU auth eval. It is allowed as a high-throughput advisory/dev-loop signal because PR107 M5 Max `0.19664189` matched GHA Linux x86_64 `0.1966358879` within `6e-6`, but it must be tagged `[macOS-CPU advisory only]` until confirmed on Linux x86_64.
- Required CPU substrate: **Linux x86_64** (Ubuntu LTS, matching the contest's GitHub Actions `ubuntu-latest` runner family; AMD EPYC or Intel Xeon class). The contest CI runs on x86_64 Linux; our CPU eval must too.
- Required CUDA substrate: **NVIDIA T4 / A100 / 4090 / equivalent** (matching the contest's CUDA runner; T4 is the contest's reference for the bot's CUDA comments).
- Both eval paths must use IDENTICAL upstream `evaluate.py` SHA, IDENTICAL `public_test_video_names.txt`, IDENTICAL video payloads, IDENTICAL `inflate.sh` runtime tree, IDENTICAL archive bytes.

**Rules:**

1. **Dual-eval is mandatory for any submission packet.** Before declaring a candidate "ready to PR" or "frontier-anchored," produce BOTH a `[contest-CUDA]` artifact AND a `[contest-CPU]` artifact on the EXACT same archive bytes via `upstream/evaluate.py` on `--device cuda` AND `--device cpu` respectively.

2. **Both tags are authoritative for their axis IF AND ONLY IF the hardware is 1:1 contest-compliant.** `[contest-CUDA]` requires NVIDIA GPU on Linux. `[contest-CPU]` requires x86_64 Linux. Apple Silicon CPU eval is `[macOS-CPU advisory]` (NOT `[contest-CPU]`) and is non-promotable.

3. **The CUDA−CPU gap is empirical and per-archive.** Do NOT assume PR102's −0.033 gap generalizes to our archives without measurement. Pose component appears to be the dominant gap source (5× difference on PR102 pose between CUDA and CPU), but mechanism attribution remains open: DALI/NVDEC-vs-PyAV ground-truth decode, CPU/CUDA forward-kernel drift, and pose-head numerics must be separated by the 2x2 decoder/network diagnostic before we treat any explanation as fact. Earlier FastViT attention/TF32 compounding explanations are invalid for FastViT-T12 on T4.

4. **CPU eval execution (where to run):**
   - **Vast.ai CPU instance** (Linux x86_64; cheap; matches contest CI architecture)
   - **Modal CPU container** (Linux x86_64; ~$0.06/hr; matches contest CI architecture)
   - **Lightning CPU Studio** (Linux x86_64; matches contest CI architecture)
   - **GitHub Actions CI workflow** itself (the actual contest hardware)
   - **NOT** local M5 Max / Apple Silicon / any macOS as the authoritative axis. Use macOS CPU for free parallel sweeps, curve discovery, smoke, and dev-loop ranking only; tag it `[macOS-CPU advisory only]` and promote to `[contest-CPU]` only after Linux x86_64 replay.

5. **CPU eval discipline (regardless of where it runs):**
   - Use `--device cpu` on `upstream/evaluate.py` directly. Verify `torch.cuda.is_available() == False`.
   - Force NO MPS path (which doesn't exist on Linux x86_64 anyway, but be explicit).
   - Tag results `[contest-CPU]` distinctly ONLY when running on Linux x86_64. Apple Silicon CPU eval is `[macOS-CPU advisory only]`.
   - CPU eval on a small Vast.ai / Modal CPU instance takes 60-120 min for 600 samples (matching the contest GitHub Actions CPU runner). Budget accordingly.

6. **For non-submission empirical work (intermediate candidates, ablations, sweep arms), use the cheapest faithful signal that matches the question.** CUDA is still the GPU-axis truth; macOS CPU and MPS can accelerate research-signal sweeps when tagged non-authoritatively. The dual-eval mandate applies specifically to ARCHIVES THAT WILL SHIP or are used to make medal-band/frontier claims.

7. **Existing CUDA-only artifacts are NOT retroactively invalidated.** They remain `[contest-CUDA]` with their CUDA-axis truth value. The dual-eval mandate is forward-looking: from this rule's commit forward, every shippable archive gets both axes on 1:1 contest-compliant hardware.

8. **Lane Maturity registry must reflect both axes.** A lane reaching Level 2/3 with a `[contest-CUDA]` anchor but no `[contest-CPU]` anchor (on Linux x86_64) is incomplete for medal-band ranking purposes — record both or record the missing one as a known gap.

Tooling:
- `tools/plan_dual_device_auth_eval.py` emits paired CPU/CUDA commands for the exact same archive/runtime.
- `tools/plan_public_pr_cpu_auth_eval.py` plans or runs a public-PR CPU replay from the reproduction ledger.
- `tools/public_pr_eval_comment_scorecard.py` extracts host PR-comment eval rows and recomputes scores from rounded PoseNet/SegNet/bytes.
- `experiments/contest_auth_eval.py` stamps CPU full-sample results as `evidence_grade="contest-CPU"` with `promotion_eligible=false`, `score_claim_valid=false`, and `rank_or_kill_eligible=false`.

Memory: `feedback_dual_cpu_cuda_auth_eval_mandatory_20260508.md` (this rule's source memo). Cross-ref `feedback_cuda_cpu_auth_eval_drift_pr102_pr104_20260508` (codex's drift hypothesis matrix that established the empirical basis).

## eval_roundtrip — NON-NEGOTIABLE, HIGHEST EMPHASIS

**EVERY training path MUST use eval_roundtrip.** There are ZERO exceptions. This includes:
- train_distill.py (has it)
- training.py Trainer (NOW has it, eval_roundtrip=True by default)
- constrained_gen.py (has it)
- optimize_poses.py (has it)
- qat_finetune.py (has it)
- ANY new training script or optimization

Without eval_roundtrip, proxy-auth gap is 2-6x on PoseNet. Every training run without it is a WASTED run. This mistake has been made on EVERY component in this project. It stops now.

**NeRV/HNeRV renderer trainers must also keep scorer preprocess differentiable.** PR #95/#106 proved that eval-roundtrip belongs inside the training inner loop and that upstream `rgb_to_yuv6` severs PoseNet gradients because it is `@torch.no_grad()` / in-place. Any trainer that backprops through PoseNet/SegNet must either call `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` before scorer loss and patch/load differentiable YUV6 (`patch_upstream_yuv6_globally`, `load_differentiable_scorers`, or explicit `differentiable_rgb_to_yuv6`) before scorer construction, or carry a research-only ablation waiver with no score claim. Canonical implementation: `src/tac/differentiable_eval_roundtrip.py`; design memo: `.omx/research/CLAUDE_md_addition_eval_roundtrip_inner_loop_yuv6_20260509.md`.

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

**LOCAL MPS IS NEVER TO BE USED AS SCORE TRUTH OR AS THE AUTHORITY FOR STRATEGY, PLANNING, OR ANALYSIS.** Verified 2026-04-25 with side-by-side gating measurement on the same pinned archive:

| Metric | Local MPS | CUDA A100 (contest scorer) | Drift |
|---|---|---|---|
| PoseNet distortion | 0.245 | **0.0107** | **23x WORSE on MPS** |
| SegNet distortion | 0.0024 | 0.00116 | 2x WORSE on MPS |
| **Final score** | **2.26** | **0.90** | **2.5x WORSE on MPS** |

PoseNet specifically drifts 23× on MPS. Do not attribute this to FastViT-T12 attention: FastViT-T12 is RepMixer/convolutional on the contest path. Treat MPS drift as an empirical hardware/runtime mismatch until a layerwise diagnostic proves the mechanism.

**Rules:**
1. ALL **intermediate / non-submission** auth eval must run on CUDA (Vast.ai 4090, A100, T4). Never MPS as the authoritative axis. **EXCEPTION: submission packets (PR'd archives or "frontier-claimed" archives) require BOTH `[contest-CUDA]` AND `[contest-CPU]` per the new "Submission auth eval — BOTH CPU AND CUDA" section above.** The "never CPU" prohibition in the original rule was about local MPS-style CPU forward passes through SegNet/PoseNet scorers (which are noise like MPS), NOT about the contest's `upstream/evaluate.py --device cpu` path (which is the contest leaderboard's official scorer for ranking). The two are different: local-CPU-scorer-noise is invalid; contest-CPU-evaluator is authoritative.
2. MPS is acceptable for proxy scoring during training (continuous monitoring), smoke tests (architecture validation), code-correctness checks, and long cheap research-signal sweeps that only generate curve-shape priors. NEVER for strategy decisions, ranking, shipping, method retirement, or paper empirical claims.
3. Score numbers measured on MPS may NOT be reported as "auth" or "contest-compliant" anywhere — in commits, run_log, BATTLE_PLAN, or summaries. Tag them `[MPS-PROXY]` and treat as advisory only.
4. Before any major internal CUDA-axis decision (kill/promote) the score MUST come from a CUDA `inflate.sh` + `upstream/evaluate.py` run on the EXACT archive bytes. Before any ship/frontier/medal-band decision, the same archive must also have a `[contest-CPU]` Linux x86_64 result, and neither axis may be inferred from the other.
5. preflight should reject auth eval invocations with `--device mps` and warn loudly.
6. The historical "2.01" / "2.26" / "2.91" numbers in memory and BATTLE_PLAN may all be MPS artifacts. The first verified CUDA contest-compliant baseline is 0.90 (2026-04-25 21:00).

2026-05-07 refinement: use the local MPS GPU as a free signal generator, not as a judge. Any long MPS sweep that feeds autopilot, meta-Lagrangian, Pareto, or bilevel planning must be serialized through `tools/build_mps_research_signal_manifest.py` / `tac.optimization.mps_research_signal` and stamped `evidence_grade="MPS-research-signal"`, `score_claim=false`, `promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`. The output may seed candidates and curve-fit priors; exact CUDA auth eval on a byte-closed archive is still required before score use.

This is the 5th catastrophic measurement bug class. Every score above this line in the run_log was potentially wrong by a factor of 2-3. Sub-Quantizr-0.33 is genuinely reachable from the true 0.90 baseline; do not give up real GPU dollars on the wrong baseline ever again.

## Remote code parity — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Before any remote eval or training run, verify the deployed code matches local HEAD.** Stale code on remote killed SHIRAZ today (16h training successful, then auth eval crashed silently because the deployed version had a NameError I had fixed locally that morning).

Rules:
1. `deploy_vastai.py launch()` MUST run `git pull --ff-only` on the remote BEFORE starting any work. If git pull fails (uncommitted changes, conflict, missing repo), abort the launch.
2. preflight should add a "remote_code_parity" check: SSH in, get `cd /workspace/pact && git rev-parse HEAD`, compare to local HEAD; block launch on mismatch unless `--allow-stale-remote` is passed (with warning).
3. The script process inside tmux MUST write a heartbeat to `${WORKSPACE:-$PWD}/.omx/tmp/heartbeat_<session>.log` every N minutes. A separate watchdog reads heartbeats; alerts if stale > 30 min. Tmux session existence is NOT a heartbeat.
4. Any auth eval failure on remote that has been running > 1 hour is a CRITICAL incident — investigate immediately, do not let the instance keep accruing cost while broken.

This is the 6th catastrophic operational pattern. The cost: $3-10 per occurrence in idle GPU time + multi-day delays in measurement. Build the protocol so it never happens again.

## Codex CLI invocation — NON-NEGOTIABLE, HIGHEST EMPHASIS (REVISED 2026-04-29 PM)

The bash harness sends SIGURG (exit 144) to BG bash processes at ~3 minutes. The earlier rule "always use Agent wrapper" was directionally right but UNDER-PRECISE. The real issue is process-group inheritance: any child of the dying bash dies too. **The fix is proper detachment — codex CAN run for hours from BG bash if launched correctly.**

**Two valid invocation patterns**:

### Pattern A — Detached BG bash (preferred for non-interactive runs)

```bash
mkdir -p .omx/tmp/codex_runs
nohup bash -c '
  codex exec --skip-git-repo-check --sandbox read-only \
    -m gpt-5.5 -c model_reasoning_effort=xhigh \
    -o .omx/tmp/codex_runs/<label>.last.txt \
    "<prompt>" \
    2>&1 | tee .omx/tmp/codex_runs/<label>.log > /dev/null
' < /dev/null > .omx/tmp/codex_runs/<label>.outer.log 2>&1 &
disown
```

Why this survives:
- `nohup` — ignore SIGHUP from terminal hangup
- `bash -c '...'` subshell — wraps the pipe so `tee` captures stdout properly even if outer dies
- `< /dev/null` — close stdin so codex doesn't wait for input
- `2>&1 | tee` — capture stdout+stderr to log file with explicit flushing
- `> outer.log 2>&1 &` — redirect immediate parent's output, fork to background
- `disown` — remove from job table so parent shell exit can't reach it
- `-o .omx/tmp/.../<label>.last.txt` — codex's own guaranteed final-message capture (survives even if log file pipe breaks)

**Verified 2026-04-29**: detached sanity test produced 11,449-token response in ~10s with no harness interference.

### Pattern B — Agent tool wrapper (preferred for interactive multi-step orchestration)

When the codex session needs to be orchestrated through multiple stages (read context → reason → write code → verify), use the `Agent` tool. The Agent has its own bash environment plus poll-and-wait logic.

**Rules**:
1. NEVER bare `Bash run_in_background: true` to launch `codex exec`. The bash inherits our process group and dies at SIGURG-144.
2. ALWAYS use Pattern A (`nohup` + `bash -c '...'` + `disown`) OR Pattern B (`Agent` tool wrapper).
3. Codex MCP-plugin (rmcp) auth may be expired separately from core codex API. If you see `TokenRefreshFailed` in stderr, codex SAFE FUNCTIONS still work — only MCP-augmented features fail. Re-auth via `codex login` if needed.
4. ALWAYS use `-o .omx/tmp/.../<label>.last.txt` flag — guarantees final-message capture even if pipe breaks.
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

## Public Disclosure Hygiene

- Public release is intentional, not automatic. Keep credentials, private infrastructure URLs, local absolute paths, raw provider logs, unpublished operator state, and account metadata out of GitHub/docs/site/public supplement surfaces.
- Detailed OSS/paper writeups are allowed when they are deliberately promoted, but private `.omx/state`, raw experiment directories, and provider transcripts must be sanitized into release manifests or dated research ledgers first.
- Cloudflare/Lightning/public supplement links belong in sanitized release manifests or approved public docs, not incidental logs or generated state files.
- If a claim, archive recipe, or implementation detail is still marked private/pending approval in a ledger, preserve that disclosure label until the human explicitly changes it or a newer committed release manifest supersedes it.

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

**`--expected-content-sha256` discipline (NON-NEGOTIABLE, post-92aba3ca; docstring corrected 2026-05-13)**: Before calling the serializer, compute the sha256 of each file's **CURRENT WORKING-TREE content** (post-edit, the state you intend to commit). Pass it to the serializer via `--expected-content-sha256 <file>=<sha>` per Catalog #157 (`check_commit_serializer_pre_lock_hash_against_head`). The serializer hashes the working-tree content at lock-acquire time and refuses with rc=4 if it differs from the declared sha.

**CRITICAL — this is NOT the HEAD sha. This is the WORKING-TREE sha AFTER your edits.** The serializer's purpose is to detect concurrent sibling edits during the lock-wait window: if a sister subagent modifies the file between the moment you snapshot your post-edit working-tree content and the moment the serializer acquires the lock, your declared sha will no longer match the working-tree content the serializer hashes, the serializer refuses with rc=4, and YOU re-base on the sibling's landed work instead of silently swallowing it under your commit body.

The earlier wording ("compute the sha BEFORE editing") was misleading and led three subagents on 2026-05-13 (WAVE-6-FOLLOWUP-MULTI, NVIDIA-RIGOR-LOWS, TCNERV-BLOCKNERV-MIGRATE) to declare HEAD shas, all refused at rc=4; second attempts with post-edit working-tree shas succeeded. The canonical contract per `tools/subagent_commit_serializer.py::_expected_content_sha256_check` is: declare what the working-tree content SHOULD be at lock-acquire time, which is exactly the post-edit content you intend to commit.

This still extincts the 92aba3ca pre-pre-lock race (commit-swap bug class diagnosed 2026-05-12): if two subagents independently edit the same file and each declares their own post-edit sha, only ONE can match the working-tree at lock-acquire time. The losing subagent gets rc=4 and must re-base on the winner's landed work, instead of both edits silently colliding under one commit body.

Worked example:

```bash
# Step 1: make your edits freely (one or many edits to one or many files).
# ... your edits ...

# Step 2: AFTER all edits, capture the working-tree sha of every file you plan to commit.
PREFLIGHT_SHA=$(sha256sum src/tac/preflight.py | awk '{print $1}')
# (macOS: shasum -a 256 src/tac/preflight.py | awk '{print $1}')

# Step 3: commit through the canonical serializer with the captured post-edit sha:
.venv/bin/python tools/subagent_commit_serializer.py \
    --message "preflight: add new strict gate" \
    --files src/tac/preflight.py \
    --expected-content-sha256 "src/tac/preflight.py=${PREFLIGHT_SHA}"
```

Multiple files: repeat the flag (`--expected-content-sha256 a.py=<sha_a> --expected-content-sha256 b.py=<sha_b>`). Each declared sha must match its file's post-edit working-tree state at the moment of the serializer call.

**Empirical receipts:** three subagents on 2026-05-13 (WAVE-6-FOLLOWUP-MULTI's e_nerv driver commit, NVIDIA-RIGOR-LOWS first-pass, TCNERV-BLOCKNERV-MIGRATE) re-discovered this gotcha. First attempts with pre-edit HEAD shas refused with rc=4; second attempts with post-edit working-tree shas succeeded. The docstring now reflects the empirical canonical contract.

Without this discipline, Catalog #157's static gate still refuses bare `git commit` outside the serializer, but the pre-pre-lock race remains observable. WITH this discipline (declaring post-edit working-tree shas), both the static and dynamic surfaces of the commit-swap bug class are extincted.

Cross-ref: `feedback_check_64_smoke_proofs_resolved_AND_subagent_serializer_landed_20260429.md` (the canonical bug-class incident report) + Catalog #157 (the static gate + dynamic `--expected-content-sha256` check) + Catalog #117 (`check_subagent_commit_serializer_uses_lock`, the sister gate that enforces last-50-commit usage) + `feedback_concurrent_subagent_commit_message_swap_20260429.md` (the 2026-04-29 PM incident that originated the rule).

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

## Beauty, simplicity, and developer experience — non-negotiable

Beauty and elegance are engineering constraints here, not decoration. Prefer
small typed abstractions, clear names, deterministic schemas, and composable
contracts that make the next lane easier to build correctly. A powerful idea is
not done until a new engineer can find it, run it, inspect its artifacts,
understand its failure modes, and compose it with adjacent codec stages without
reverse-engineering hidden state.

When adding or hardening planner, codec, archive, native, or training code:

- choose the simplest abstraction that preserves the real invariants;
- keep public APIs expressive, documented, typed, and stable enough for OSS
  users and future native ports;
- make artifacts human-readable where possible and machine-checkable always;
- separate contest-only overfit paths from production/generalized paths with
  explicit metadata instead of runtime guesswork;
- delete dead fields, stale adapters, and duplicate one-offs once a canonical
  contract replaces them;
- add conformance vectors, examples, and focused tests so Rust/Zig/C/assembly
  ports can prove byte-for-byte behavior against the Python oracle.

Do not hide complexity behind vague helpers. If the domain is inherently hard,
make the interface narrow and explicit, preserve the proof artifacts, and keep
the implementation readable enough for adversarial review.

## Contest vs production target modes — non-negotiable

Exact-eval dispatch tools are contest-score actuators. They may consume only
candidates whose metadata targets contest exact eval, either implicitly or via
`target_modes=["contest_exact_eval"]`. Production-only work for comma-ai,
openpilot, edge devices, or optional on-device learning belongs in planning,
benchmarks, and production runbooks until it intentionally emits a contest
archive with full custody.

For mixed contest/production lanes, declare the split explicitly:

- `target_modes`: e.g. `["contest_exact_eval", "openpilot"]` for dual-purpose
  archive work, or `["openpilot_edge"]` for production-only exploration.
- `deployment_target`: e.g. `t4_contest_runtime`, `comma_ai_production`,
  `openpilot_edge`, `desktop_research`, or `device_learning_optional`.
- `score_affecting_payload_changed` or `charged_bits_changed`, plus old/new
  archive or payload SHA-256s, whenever a self/neural/codegen/binary lane
  claims it changes score-relevant bytes.

Self-compression, neural compression, on-device learning, edge-learning,
generated decoders, Rust/Zig/C kernels, and assembly kernels are first-class
optimization directions. They are contest-admissible only when charged bits
changed and exact archive custody exists. Outside contest mode they must be
optional, deterministic, reproducibly built, and paired with scalar or portable
fallbacks suitable for comma-ai/openpilot production review.

## Deterministic packet compiler — non-negotiable

Low-level native/codegen work should converge into a separate deterministic
submission-packet compiler. It must ingest a contest-compliant packet,
deconstruct archive/runtime/payload bytes into a typed manifest and golden
vectors, then emit either byte-identical output or an intentionally
byte-different packet with exact old/new SHA-256 and charged-byte proof.

The compiler must support separate target profiles:

- `contest_one_video_replay`: contest-only, one-video overfit replay. It may
  replace learned inference with deterministic generated code, fixed tables,
  distilled byte transducers, or per-frame/per-pair streams derived from the
  trained model's behavior on the scored video. It is admissible only when the
  archive remains self-contained and exact CUDA auth eval validates it.
- `contest_generalized`: contest-compliant but not one-video replay. It must
  preserve the runtime contract for unseen contest-shaped videos and must not
  rely on fixed per-frame lookup tables or replay data from the scored video.
- `production_generalized`: comma-ai/openpilot production target. It may reuse
  the same byte-deconstruction machinery, but must preserve cross-video
  behavior, portability, maintainability, and deterministic reproducible native
  builds.
- `production_edge_adaptive`: production-only edge target. Optional on-device
  learning is allowed only outside contest mode and only behind deterministic
  fallbacks, reproducible builds, and explicit capability gates.

Required modes:

- `identity`: re-emit the packet with byte-for-byte parity.
- `canonicalize`: normalize only compliance-approved metadata and report every
  changed byte.
- `optimize`: change score-affecting bytes only when the runtime consumes the
  new contract and all artifacts remain inside the contest packet.

The compiler must fail closed on hidden sidecars, scorer modifications,
external state, network dependencies, unsupported ZIP features, parser
divergence, non-deterministic native builds, missing golden vectors, or missing
runtime-tree custody. This tool is the bridge from Python deconstruction to
Rust/Zig/C/ASM ports: Python remains the oracle until native implementations
pass the same vectors byte-for-byte.

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

Every returned result also needs a composition review before retirement
language. Preserve whether the result is additive, antagonistic, orthogonal, or
redundant with current champion components. Check HStack/VStack/multi-pass
forms, residual rescue, per-tensor/per-channel routing, score-aware allocation,
hybrid fallback, and whether the result should become a sensitivity prior,
trust-region boundary, or side-info source. A standalone negative can still be
an engineering input. Do not mark a lane exhausted unless this
synergy/antagonism/stacking analysis is written into the ledger or review
packet.

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

**Codex R5-r6 (STRICT @ 0 — flipped 2026-05-12 per UUU audit + FFFF Bug 3 bulk text-fix; initial landing was warn-only):**

15. `check_no_brittle_six_line_waiver_lookback` — Waiver markers must be SAME-LINE; the previous 6-line lookback could waive unrelated calls. R5-r6 #1.
16. `check_kl_distill_uses_roundtripped_frames` — KL distillation must use roundtripped frames not raw GT. R5-r6 #2.
17. `check_eval_roundtrip_gate_called_after_output_dir_resolution` — Gate ordering correctness. R5-r6 #3.
18. `check_nvdec_probe_has_error_classification` — NVDEC probe must classify NoDevice / DriverMismatch / etc. R5-r6 #4.
19. `check_archive_builders_use_deterministic_zip` — `ZipFile.write` is non-deterministic; use ZipInfo + writestr with fixed timestamp. R5-r6 #5.

**Additive 2026-04-27 (12 new — wired into `preflight_all()` and STRICT @ 0; initial landing was warn-only-pending; STRICT-FLIPPED 2026-05-12 per UUU audit + FFFF Bug 3 bulk text-fix):**

A. `check_vastai_create_has_label` — Every `vastai create instance` call must pass `--label`. Orphan instances accrue cost silently (today: instance 35707822, ~$0.05 wasted). Live count: 0.
B. `check_vastai_create_writes_tracker` — Every Vast.ai launch must register the instance ID to `.omx/state/vastai_active_instances.json` so cleanup scripts can detect orphans. Live count: 0 → STRICT (initial landing was 2; cleared via tracker-write enforcement + WAVE-A-2 2026-05-12 bulk text-fix per UUU audit).
C. `check_subagent_prompts_no_cpu_fallback` — Subagent prompts must not allow `--device cpu` without a `deterministic-bytes acceptable` caveat. CPU fallback in byte-deterministic build = invalid archive. Live count: 0 → STRICT (initial landing was 1; cleared via prompt-template canonicalization + WAVE-A-2 2026-05-12 bulk text-fix per UUU audit).
D. `check_scores_have_lane_tag` — Every numeric score in `run_log.md`/`findings.md`/`BATTLE_PLAN.md` must carry a lane tag (`[contest-CUDA]`, `[advisory only]`, `[MPS-PROXY]`, …). MPS-CUDA drift = 23x. Live count: 0 → STRICT (initial landing was 20; cleared via bulk lane-tag pass + WAVE-A-2 2026-05-12 bulk text-fix per UUU audit).
E. `check_waivers_specify_env_gate` — `# SCORER_AT_INFLATE_WAIVED` markers must name an env-gate (`env-gated-INFLATE_TTO=1`) so operators can audit which env-vars enable scorer-at-inflate paths. Live count: 0.
F. `check_halfframe_archive_uses_trained_profile` — `--half-frame` archive builds must use a renderer trained for it (profile with `mask_half_sim_prob>0` OR `use_zoom_flow=True`). Memory: `feedback_half_frame_breaks_posenet`. Verified 2026-04-27 score 17.55. Live count: 0 → STRICT (initial landing was 2; cleared via profile-pinning + WAVE-A-2 2026-05-12 bulk text-fix per UUU audit).
G. `check_profile_keys_have_resolvers` — Bidirectional companion to dead-resolver scanner: every PROFILES key must be consumed somewhere in src/tac or experiments. Live count: 0 → STRICT (initial landing was 91; cleared via dead-profile-key cleanup + WAVE-A-2 2026-05-12 bulk text-fix per UUU audit).
H. `check_inflate_scorer_load_has_runtime_banner` — Inflate files loading scorers must `print('[strict-scorer-rule] ...')` at runtime so the score can be tagged `[scorer-at-inflate-noncompliant]`. Live count: 0.
I. `check_test_files_imports_resolve` — Test files importing from `tac.*` must resolve to actual symbols. Existing dead-import scanner skips test dirs; this complement catches broken tests that silently skip at collection. Live count: 0 → STRICT (initial landing was 25; cleared via test-import cleanup + WAVE-A-2 2026-05-12 bulk text-fix per UUU audit).
J. `check_vastai_prompts_have_cost_cap` — Subagent prompts mentioning Vast.ai must mention a `$` cap, `budget`, or `destroy instance`. Memory: `feedback_vastai_cost_paranoia`. Live count: 0.
K. `check_uniward_delta_has_attestation_gate` — `--with-uniward-delta` invocations must include `--allow-pending-compliance` OR an attestation file reference. Lane C R5 (commit ef8a9a1b). Live count: 0 → STRICT (initial landing was 6; cleared via attestation-gate backfill + WAVE-A-2 2026-05-12 bulk text-fix per UUU audit).
L. `check_remote_scripts_write_provenance` — Every `scripts/remote_*.sh` must write `provenance.json`. Memory: `feedback_canonical_remote_bootstraps`. Live count: 0 → STRICT (initial landing was 5; cleared via provenance.json emission in Lanes A/B/D/G + WAVE-A-2 2026-05-12 bulk text-fix per UUU audit).

**Session bug-classes (2026-05-08, BUGCLASSES subagent):** Eight new checks closing bug classes B1-B8 from the 2026-05-08 codex/review-engineering/review-math adversarial sessions. Memory: `feedback_session_bug_classes_to_preflight_20260508.md`.

91. `check_encoder_decoder_dequantization_roundtrip_tested` (B1) — Tools that quantize + emit an archive must have a paired roundtrip test (`# ROUNDTRIP_TESTED:<pytest>` or sibling `test_<basename>_roundtrip.py` with `ENCODE_INFLATE_ROUNDTRIP` token). Live count: 1 (`tools/build_admm_x_lossy_coarsening_path_b_step6.py`). Strict-flip pending sibling test landing.
92. `check_evidence_row_archive_bytes_has_provenance` (B2) — Every evidence row setting `empirical_archive_bytes` must satisfy ≥1 of: `archive_sha256`, `byte_proxy_only=true` AND `cuda_eval_worth_testing=false` AND `ready_for_exact_eval_dispatch=false`, proxy `measured_config_status`, or textual provenance tag in `source` (`[CPU-prep`/`[byte-anchor`/`[empirical:`/`[contest-CUDA`/etc). Live count: 5 (cathedral_autopilot phase4 orchestrator rows). Strict-flip pending row backfill.
93. `check_build_manifest_archive_custody_clean` (B3) — Every `experiments/results/**/build_manifest.json` referencing `archive_relpath` + `archive_sha256` must satisfy ≥1 of: archive committed in git, verifier script (`tools/verify_*archive*sha*.py`) references the relpath/SHA, OR `custody_status` ∈ `{published, committed-binary, ci-rebuildable, transient-allowed}`. Live count: 7 (lossy_coarsening + cross_paradigm dirty-disk archives). Strict-flip pending verifier scripts or custody annotations.
94. `check_admm_naming_matches_iterative_consensus_implementation` (B4) — Files/classes/functions named `admm`/`primal_dual` must contain real iterative consensus updates (rho/z/u) inside a loop OR be renamed `lagrangian_*`/`bisection_*` OR carry `# ADMM_WAIVED:<reason>`. Live count: 27 (Path B step 5/6 tools + cross-paradigm orchestrator + tac codec_op_admm_adapter). Strict-flip pending rename or waiver annotations. Memory: `feedback_review_math_council_4_landings_20260508.md`.
95. `check_inflate_wire_format_no_dead_bytes` (B5) — Variables read via `struct.unpack`/`read`/`frombuffer` in inflate.py must be loaded downstream OR carry `# DEAD_BYTES_AUDIT_OK:<reason>`. Vendored public-PR intakes excluded. Live count: 0 → STRICT.
96. `check_predispatch_retired_config_warning` (B6) — Every retired evidence row (`measured_config_retired_*`) must carry `dispatch_blockers=[…, "reactivation_required_before_new_dispatch"]` AND non-empty `reactivation_criteria`. Live count: 0 → STRICT.
97. `check_scores_have_lane_tag_paper_research` (B7) — Sister of Check D (`check_scores_have_lane_tag`); extends lane-tag discipline to `docs/paper/**/*.md` and `.omx/research/**/*.md`. Live count: 58. Strict-flip pending bulk re-tag pass.
98. `check_pr101_tools_torch_load_allowlist` (B8) — `tools/pr101_*.py` / `tools/build_admm_*.py` / `tools/build_cross_paradigm_*.py` calling `torch.load(..., weights_only=False)` must EITHER carry `# WEIGHTS_ONLY_FALSE_OK:<reason>` in 5-line window OR have a sha256/magic-byte preceding-30-line validation. Sister of `preflight_loader_format_safety` (Check 14). Live count at landing: 32; 31 at 2026-05-12 audit; STRICT @ 0 after T2-D cluster 4 retire-audit. The T2-D audit (61 candidate files; ~29.6K LOC) found ALL files STILL-ACTIVE infrastructure (36+ dedicated tests, 9 lane-registry entries, 2 preflight constants requiring file existence — including `_ADMM_BISECTION_TOOLS` consumed by `check_admm_lagrangian_bisection_convergent`). DELETE-OK count: 0. Catalog #14 (`preflight_loader_format_safety`) does NOT cover the `tools/` surface so #98 remains the dedicated cluster gate. All 31 violations annotated with canonical waiver `# WEIGHTS_ONLY_FALSE_OK:trusted-PR101-substrate-state-dict-local-artifact`. Memory: `feedback_catalog_98_retire_and_historical_tool_audit_landed_20260512.md`.

**Representation-integration gates (2026-05-08, codex audit):** Ten new prevent-recurrence gates closing the failure modes documented in `.omx/research/representation_integration_gap_audit_20260508_codex.md`. Memory: `feedback_representation_integration_gates_landed_20260508.md`.

99.  `check_gate1_representation_promotion_card` — Promotion-card manifests must carry all 12 fields: `representation_name`, `target_modes`, `source_artifact`, `archive_builder`, `inflate_consumer`, `runtime_manifest`, `changed_payload_paths`, `old_new_sha256s`, `component_risk_plan`, `exact_eval_command`, `owner`, `next_unblock_action`. Live count: 0 → STRICT.
100. `check_gate2_no_naked_bytes` — Rows claiming `score_claim=true` / `ready_for_exact_eval_dispatch=true` / positive `contest_dispatch_verdict` must include byte-closure proof (archive_path+sha256 / inflate_consumer / parser_section_manifest / `measured_config_status=contest_cuda*`). Sister of B2. Live count: 0 → STRICT.
101. `check_gate3_parser_section_manifest` — HNeRV-family monolithic-packet manifests must carry `offsets`, `lengths`, `section_names`, `section_sha256s`, `entropy_estimates`, `old_new_section_boundaries` OR `vendored_public_pr_intake=true`. Live count: 0 → STRICT.
102. `check_gate4_export_first` — Lane registry entries with learned-codec representation tokens at level >= 1 must declare `export_format` OR `research_only=true` OR a recognized format token in `notes`. Long-training scripts must carry `EXPORT_FORMAT="<format>"` declaration OR `# EXPORT_FORMAT_OK:<format>:<reason>` waiver. Live count: 2 (lane_12_nerv_mask_codec, lane_alpha_nerv_mask) — warn. Strict-flip pending `research_only=true` backfill.
103. `check_gate5_runtime_closure` — Build manifests claiming dispatch must record `runtime_manifest` / `runtime_closure_verified` / `inflate_smoke_log`. Public-PR replay rows marked negative/retired must classify `failure_class` (runtime_blocker_dependency_missing / wrapper_signature / hidden_sidecar / local_path / cpu_cuda_mismatch / method_negative). Live count: 0 → STRICT.
104. `check_gate6_mask_pose_coupling` — Mask representation replacements claiming dispatch must record `decoded_mask_sha256s`, `mask_disagreement`, `pose_regeneration_status`, `geometry_diagnostics`, `component_risk_plan`. Live count: 0 → STRICT.
105. `check_gate7_no_op_provenance` — Byte-level transforms (repack / brotli_param / codec_swap) claiming dispatch must record `old_archive_sha256` + `new_archive_sha256`, `payload_change_proof` (or `no_op_detector_passed=true`), and `runtime_consumption_proof` (or `runtime_closure_verified=true`). Sister of B5. Live count: 0 → STRICT.
106. `check_gate8_exact_evidence` — Frontier-promotion rows must carry archive_bytes, archive_sha256, runtime_manifest, exact_eval_command, hardware, sample_count >= 600, components (seg/pose/rate), recomputed_score, log_path, dispatch_claim_status. Live count: 0 → STRICT.
107. `check_gate9_blocker_ownership` — Blocked learned-codec lanes (NeRV/HNeRV/CoolChic/C3/Ballé/hyperprior) must carry one of: `(active_owner+unblock_experiment)`, `(exact_negative+reactivation_criteria)`, `compliance_impossibility_proof`, or `terminal_retirement_note`. Live count: 0 → STRICT.
108. `check_gate10_stack_promotion` — HStack/VStack/cross-paradigm rows claiming dispatch must record `archive_boundary`, `side_information`, `latent_streams`, `k_scale_tables`, `decoder_overhead_bytes`, `runtime_consumer`, `exact_eval_plan`. Proxy/byte-anchor rows must explicitly set `score_claim=false` AND `ready_for_exact_eval_dispatch=false`. Live count: 0 → STRICT.

**Public-PR-intake clone source-provenance (2026-05-08, codex finding 2, STRICT @ 0):**

109. `check_public_pr_intake_clones_pristine` — Public PR intake clones under `experiments/results/public_pr*_intake_*/{source,repo,pr*_src/repo}/` MUST be pristine bytes-identical to upstream PR head. The check discovers all clones (62 across `_codex`, `_worker`, `_auto`, `public_pr_archive_*` layouts), runs `git -C <clone> diff --numstat` to distinguish text edits (real source-provenance corruption) from git-LFS pointer-vs-content state mismatches (NOT corruption), and refuses any clone with text edits. Codex 2026-05-08 review caught 39 stale `KL_BATCHMEAN_OK` waiver comments across 8 dirty clones; same-day revert + STRICT gate landed. Replacement waiver-rationale location is `reverse_engineering/public_pr_waiver_manifest.json` (committed metadata). Live count: 0 → STRICT. Memory: `feedback_codex_finding_2_public_intake_pristine_FIXED_20260508.md`.

**Recovery-metadata custody-evidence-corruption (2026-05-08, codex finding 5, STRICT @ 0):**

110. `check_recovery_metadata_append_only` — `recovery_metadata.json` files under `experiments/results/recovered_*/` MUST use the append-only `attempts[]` schema with unique `started_at_utc` per attempt and provenance (`command_log_path` OR `substantive_change_from_prior_attempt`) for non-initial attempts. The check refuses duplicate timestamps (suggests in-place edit), missing required attempt fields (`attempt_kind`/`started_at_utc`/`completed_at_utc`/`elapsed_seconds`/`ssh_reachable`/`archive_zip`/`artifacts`), and provenance-less revisit/force-rerun entries. Legacy v1 single-record schema is grandfathered (warn-only) for backward compatibility during migration. Codex 2026-05-08 finding caught timestamp-only churn on `recovered_42_dead` + `recovered_99999_phantom` (April 30 → May 8 with no substantive change) destroying the original audit trail; same-day revert + schema migration to `v2_attempts` + writer rewrite (`tools/recover_lane_artifacts.py::_write_report` now keys on `started_at_utc`, refuses to mutate closed attempts via `RecoveryMetadataAppendOnlyError`) + STRICT gate landed. Live count: 0 → STRICT. Memory: `feedback_codex_finding_5_recovery_metadata_appendonly_FIXED_20260508.md`.

**Status-artifact stale-state freezing (2026-05-08, codex finding 3, STRICT @ 0):**

111. `check_status_artifacts_no_stale_dirty_paths` — Committed `*status*.json` files under `experiments/results/` MUST NOT carry a top-level `dirty_paths` field as a non-empty list. The path list is transient session noise and freezing it into a committed artifact poisons downstream operator review with stale paths. Only `dirty_path_count` (scalar) and per-row `dirty_path_blockers` (per-row intersections) are stable enough to persist. Codex 2026-05-08 review caught `frontier_roadmap_status_20260507_codex/status.json` listing 28 stale paths (vs. 7 actually dirty at scan time) emptying every candidate pool with `dirty_blocked_row_count: 13`; same-day fix removed the field from `tools/build_frontier_roadmap_status.py` output, regenerated 2 status JSONs, and stripped 2 additional historical roadmap status files. Live count: 0 → STRICT. Memory: `feedback_codex_findings_3_4_status_dirtypaths_rebuild_recipe_FIXED_20260508.md`.

**Rebuild-recipe baked runtime state (2026-05-08, codex finding 4, STRICT @ 0):**

112. `check_rebuild_commands_no_baked_runtime_state` — `rebuild_command.txt` recipe files under `experiments/results/` MUST NOT contain hardcoded `--now-utc <ISO_TIMESTAMP>` or `--operator-approved-*` flags inline UNLESS the file carries an explicit historical-only banner (one of: "HISTORICAL ARTIFACT", "DO NOT REPLAY", "frozen historical", "forensic reproduction") in the first 10 lines. Hardcoded `--now-utc` makes lane-claim TTL evaluate against an obsolete timestamp; hardcoded `--operator-approved-*` carries forward an approval context that may no longer apply. The check accepts the historical banner pattern as an explicit opt-out for forensic reproduction artifacts. Codex 2026-05-08 review caught `frontier_roadmap_status_20260507_codex/rebuild_command.txt` with `--now-utc 2026-05-07T18:37:16Z` + `--operator-approved-exact-cuda` baked in; same-day fix added historical banner to that recipe + `field_meta_dispatch_selection_20260507_codex/rebuild_command.txt`. Live count: 0 → STRICT. Memory: `feedback_codex_findings_3_4_status_dirtypaths_rebuild_recipe_FIXED_20260508.md`.

**Provenance-vs-state confusion META-CLASS (2026-05-08, codex META, STRICT):**

113. `check_artifact_lifecycle_compliance` — Umbrella META-CLASS gate over the unified provenance-vs-state-confusion bug class spanning findings 1+2+3+4+5 (sister gates `check_operator_approval_must_be_lane_scoped`, `check_public_pr_intake_clones_pristine`, `check_status_artifacts_no_stale_dirty_paths`, `check_rebuild_commands_no_baked_runtime_state`, `check_recovery_metadata_append_only`). Classifies every committed long-lived artifact in the repo into ONE of FOUR kinds — `LIVE_STATE` (transient session state, must be gitignored), `HISTORICAL_PROVENANCE` (append-only forensic record, mutation forbidden), `LIVE_RECIPE` (reusable instructions, baked transient values forbidden), or `DERIVED_OUTPUT` (computed from current state, regeneration header required) — using the committed registry at `.omx/state/artifact_kind_registry.yaml`. The four per-kind guards (`LiveStateGuard`/`ProvenanceGuard`/`RecipeGuard`/`DerivedOutputGuard` in `src/tac/artifact_lifecycle.py`) catch artifacts the per-finding sister gates miss because of broader path coverage. The unified taxonomy maps the meta-pattern: "transient/global/upstream state being frozen or mutated into committed/forensic artifacts in ways that destroy provenance or leak authorization." Forbidden patterns landed in CLAUDE.md FORBIDDEN_PATTERNS: (1) committing transient state into LIVE_STATE files; (2) mutating HISTORICAL_PROVENANCE files (must append); (3) baking transient values into LIVE_RECIPE files; (4) stale session state in DERIVED_OUTPUT bodies without regeneration header. The gate runs strict in `preflight_all()` (changed-scope path-filter) since sister gates 109-112 landed; FULL-STRICT mode (`ARTIFACT_LIFECYCLE_FULL_STRICT=1`) is also clean as of 2026-05-09 strict-flip cleanup (0 violations across 5494 tracked long-lived paths, down from 2061 at landing). 2026-05-09 cleanup work: (a) untracked 3 stale committed `.pid`/`.lock` LIVE_STATE files + scoped `.gitignore` patterns, (b) reclassified binary build outputs (`.bin`/`.zip`/`.pt`/`.mkv`/`.mp4`/`.png`/`.jpg`/`.gif`/`.xz`) and committed dashboard/figure snapshots from DERIVED_OUTPUT to HISTORICAL_PROVENANCE (binary files cannot embed regen headers; committed snapshots are immutable per-build forensic records), (c) added regen-header support to `tools/lane_maturity.py report` + `tools/build_frontier_roadmap_status.py` and backfilled 2 status.json files, (d) aligned `RecipeGuard` historical-marker accept list with sister gate #112 (`HISTORICAL ARTIFACT`/`DO NOT REPLAY`/`frozen historical`/`forensic reproduction` in addition to `HISTORICAL_RECIPE_ONLY`), (e) made `ProvenanceGuard` and `_git_show_json` binary-safe (`text=False` + explicit decode), (f) parameterized `/tmp` paths in `submissions/robust_current/{diagnose_scorer,eval}.py` and frozen `wave3_chain_driver.sh` with HISTORICAL_RECIPE_ONLY header. Memory: `feedback_codex_findings_meta_pattern_artifact_lifecycle_FIXED_20260508.md` + `feedback_meta_gate_113_strict_flipped_4016_artifacts_classified_20260509.md`. Recursive review: `.omx/research/artifact_lifecycle_meta_class_recursive_review_20260508.md` (3 clean passes per CLAUDE.md non-negotiable).
114. `check_training_scripts_use_real_data_in_nonsmoke_mode` — Training scripts under `experiments/train_*.py` that define BOTH `make_synthetic_*` AND `main()` MUST gate every call to `make_synthetic_*` behind `if smoke:` / `if args.smoke:` branches OR a smoke/synthetic-named function OR a same-line `# SYNTHETIC_NON_SMOKE_OK:<reason>` waiver. Codex Pattern A 2026-05-08 caught `experiments/train_score_gradient_pr101_finetune.py` calling `make_synthetic_pair_batch` unconditionally — non-smoke would have burned $8 of Lightning T4 optimizing PR101 weights against random noise (smoke gradient-path verification masked it). Same-day refactor to take an explicit `batch_source` callable + `RealPairBatchSource` (decodes `upstream/videos/0.mkv` via pyav) + non-smoke guard requiring `--pr101-archive` AND `--video-path`. Live count: 0 → STRICT (STRICT-FLIPPED 2026-05-12 per UUU audit + FFFF Bug 3 bulk text-fix; orchestrator callsite is `strict=True`). Memory: `feedback_codex_finding_pr101_synthetic_targets_FIXED_20260508.md`. Recursive review: `.omx/research/codex_finding_pr101_synthetic_targets_recursive_review_20260508.md` (3 clean passes — Hinton/Quantizr/Carmack → Yousfi/Contrarian/Boyd → Hotz/MacKay/Hassabis).

**Packet-clearance evidence-kind separation (2026-05-08, codex finding A2, STRICT @ 0):**

115. `check_packet_blocker_clearance_evidence_matches` — Packet-builder manifests (`build_manifest.json` / `candidate_manifest.json` / `a2_packet_ladder_manifest.json`) under `experiments/results/` MUST satisfy: (1) no blocker appears in BOTH `cleared_blockers` and `dispatch_blockers` (internal inconsistency); (2) every cleared blocker has a `cleared_blockers_by_evidence[blocker]` entry naming the evidence kind that justified the clearance; (3) the evidence label is in the blocker's accepted set — `no_byte_closed_runtime_packet_built` accepts `packet_local_parse_smoke` or `inflate_parity_log`; `packet_local_inflate_parity_not_run` accepts ONLY `inflate_parity_log`; (4) when `packet_local_inflate_parity_not_run` is cleared, an `inflate_parity_record` with `passed=true` MUST be present on the manifest. Codex 2026-05-08 review caught `tools/build_a2_sensitivity_weighted_pr101_packet.py` clearing the inflate-parity blocker via parse-smoke alone (4 misclear-rows across 2 manifests) while the same manifest's `next_required_actions[]` listed inflate parity as required — internally inconsistent. Same-day fix split `CLEARED_BY_PACKET_LADDER` into `CLEARED_BY_PARSE_SMOKE` + `CLEARED_BY_INFLATE_PARITY`, added `verify_inflate_parity()` helper that runs `inflate.sh` source-vs-candidate and compares outputs byte-for-byte, added `--run-inflate-parity` CLI flag, patched 2 historical manifests in place to drop the false clearance. Live count: 0 → STRICT. Memory: `feedback_codex_finding_a2_packet_inflate_parity_FIXED_20260508.md`. Recursive review: `.omx/research/codex_finding_a2_packet_inflate_parity_recursive_review_20260508.md` (3 clean passes per CLAUDE.md non-negotiable).

**META-META commit-machinery permanent-protection (2026-05-08, FIX-1/FIX-2/FIX-3/FIX-4):**

117. `check_subagent_commit_serializer_uses_lock` — Every subagent commit in the last 50 must appear in `.omx/state/commit-serializer.log` (i.e., went through `tools/subagent_commit_serializer.py`). Bypassed commits flagged unless allowlisted with `# NO_SERIALIZER_OK:<reason>` in the body. Bug class: META-FIX subagent's `src/tac/preflight.py` edits flowed into FIX-5 commit `89d6eba2` because the working tree was shared and the lock only serialized COMMITS, not EDITS. The companion FIX-1 `--no-concurrent-edit-check` (default off) on the serializer pre-and-post hashes working-tree files under LOCK_EX and refuses with rc=3 if a sister subagent edited our intended-to-commit files during the lock-wait window. THIS gate enforces that every subagent commit USES the serializer so the FIX-1 protection actually applies. Held warn-only initially; flip to STRICT after legacy-commit allowlist baseline is populated (~20 known legacy commits). Memory: `feedback_meta_meta_commit_machinery_protections_20260508.md`. Recursive review: `.omx/research/permanent_protection_meta_meta_recursive_review_20260508.md` (3 clean passes per CLAUDE.md non-negotiable).
118. `check_claude_md_catalog_no_duplicate_numbers` — CLAUDE.md catalog entries (`^[0-9]+\. \`check_*`` lines) must have unique numbers. Bug class: 2026-05-08 dual-#114 collision — FIX-A-CUSTODY (`fa604f72`) and FIX-A-SYNTH (`c80162e7`) both grabbed Catalog #114 because they read CLAUDE.md concurrently to find next-available without coordination, and a sister fork (`000089d1`) had to manually renumber after-the-fact. Now `tools/claim_catalog_number.py` (with `fcntl.flock(LOCK_EX)` on `.omx/state/next_catalog_number.txt`) serializes claims atomically — `python tools/claim_catalog_number.py claim` returns a monotonically-unique number per call. THIS gate refuses any CLAUDE.md state with duplicate `^[0-9]+\. \`check_*` lines so the lock can't be bypassed by the editor. Live count: 0 → STRICT. Memory: `feedback_meta_meta_commit_machinery_protections_20260508.md`.
119. `check_subagent_commits_have_co_author_trailer` — Every subagent commit in the last 50 must include the canonical Co-Authored-By trailer (`Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`). Allowlist via `# NO_CO_AUTHOR_OK:<reason>` for human-authored commits or commits intentionally without Claude attribution. Bug class: 3 subagents (FIX-1 `00896b43`, FIX-3+4 `c6d09bbb`, FIX-5 `89d6eba2`) flagged that the serializer wasn't auto-appending the trailer; the trailer was getting forgotten in commit messages. The companion FIX-3 makes the serializer auto-append the trailer (idempotent: skip if already present, two-newline separator between body and trailer per git convention). THIS gate enforces it on commits going forward. Held warn-only initially; flip to STRICT after legacy-commit allowlist baseline is populated (~16 known legacy commits). Memory: `feedback_meta_meta_commit_machinery_protections_20260508.md`. Recursive review: `.omx/research/permanent_protection_meta_meta_recursive_review_20260508.md` (3 clean passes per CLAUDE.md non-negotiable).

**Track 4 weight-domain saliency on score-aware substrate (2026-05-09, STRICT @ 0):**

123. `check_no_weight_domain_saliency_on_score_gradient_substrate` — Builders under `tools/build_*.py` and `experiments/build_*.py` MUST NOT compute pure-weight-domain saliency proxies (`mean(theta**2)`, `var(theta)`, `norm(theta)`, `(t * t).mean()`, `t.pow(2).mean()`, `abs(t).mean()`) on score-gradient-trained substrates (A1, anything tagged `score_gradient` / `train_score_gradient` / `track1_phase_a1_score_gradient` / `phase_a1_latent` / `a1_archive` / `a1_latent_aligned`) WITHOUT exposing a `--saliency-source score_gradient` opt-in (or wiring `tac.score_gradient_param_saliency`). Bug class: Track 4 v1 (`tools/build_uniward_stc_hessian_a1_v1.py`) used `mean(theta^2)` as a Fisher saliency proxy on the A1 archive. Empirical falsification 2026-05-09: best candidate `blocks4_7bit` (177,903 B, -359 B saved, rms ≈ 1.84e-3) lost **+0.0058** score on contest-CPU GHA Linux x86_64 — the proxy is **anti-correlated** with true score saliency on score-gradient-trained substrates because parameters with HIGH `mean(theta^2)` are exactly the ones the score-gradient pushed AWAY from zero (score-relevant), so coarsening "low-Fisher" tensors hits the directions the score-gradient training identified as score-relevant in the orthogonal sense. Same-line waiver: `# WEIGHT_SALIENCY_OK_ON_SCORE_AWARE:<reason>`. Function-context filter: only fires inside functions whose name contains a saliency-token (`fisher_proxy`, `hessian_diag`, `compute_saliency`, `weight_saliency`, `param_importance`, `uniward`, `stc_hessian`, etc.); forward-pass MSE / score expressions in non-saliency functions are correctly excluded via AST. Companion fix: `tools/build_uniward_stc_hessian_a1_v1.py --saliency-source score_gradient` + `tac.score_gradient_param_saliency.compute_score_gradient_param_saliency()` + cliff-zone gate (`bytes_saved < 1KB AND rms > 1e-3` refused unless `--allow-cliff-zone --cliff-zone-override-operator <handle>`). Live count: 0 → STRICT. Memory: `feedback_track4_bug_class_fix_self_protect_landed_20260509.md`. Cross-ref `feedback_track_4_uniward_stc_hessian_a1_landed_20260509.md` (the v1 falsification anchor) + `feedback_three_lossy_anchors_show_rel_err_squared_objective_falsified_20260508.md` (the broader cliff-class anchor).

**Representation lane archive grammar at design time (2026-05-09, warn-only initially):**

124. `check_representation_lane_has_archive_grammar_at_design_time` — Refuses Level 1+ promotion of representation/codec lanes (NeRV/HNeRV/Cool-Chic/C3/wavelet/VQ-VAE/grayscale-LUT/SIREN/coordinate-MLP/hyperprior/Ballé/nonlinear-transform/time-varying-FiLM/shared-codebook/etc.) without 8 declared evidence fields per HNeRV parity discipline forbidden pattern #4: `archive_grammar`, `parser_section_manifest`, `inflate_runtime_loc_budget`, `runtime_dep_closure`, `export_format`, `score_aware_loss`, `bolt_on_loc_budget`, `no_op_detector_planned`. Lane classification matches case-insensitive tokens against id/name (`nerv`, `hnerv`, `cool_chic`, `c3`, `wavelet`, `vqvae`, `grayscale_lut`, `siren`, `coordinate_mlp`, `hyperprior`, `balle`, etc.) — short tokens (≤ 5 chars) use word-boundary regex to avoid false positives like `c3` matching `c30`/`dc3` — OR description tokens (`representation`, `learned codec`, `neural compression`, `archive grammar`) on name/notes. Field discovery accepts (any one suffices per field): top-level lane keys, `lane["evidence"][field]`, `lane["design_evidence"][field]`, inline gate-evidence-string substrings (`<field>=` or `<field>:`), OR inline `lane["notes"]` substrings (same syntax). Two opt-outs (per HNeRV parity discipline lessons 2 + 7): `lane_class="substrate_engineering"` (substrate work that explicitly does not ship a packetized inflate) OR `research_only=true` (research-only path by construction). Bug class: leaderboard HNeRV won not by architectural novelty but by binding ARCHITECTURE + SCORE-AWARE TRAINING + ARCHIVE GRAMMAR + INFLATE RUNTIME + EXACT-EVAL CUSTODY simultaneously, in a single ~600-LOC artifact reviewable in 30 seconds. Our internal Lane 12 NeRV mask codec scaffold sat behind L2-clearance blockers for ~10 days while PR #95/#100/#101/#103 shipped end-to-end; this gate forces design-time declaration so the integration loop cannot age out. STRICT-FLIPPED 2026-05-12 per UUU audit + FFFF Bug 3 bulk text-fix; orchestrator callsite is `strict=True`. Live count: 0 → STRICT. Initial landing: warn-only per operator approval 2026-05-09; the 7 violations at that time (`lane_12_nerv_mask_codec` L2, `lane_20_balle_hyperprior` L2, `lane_pr106_latent_sidecar` L1, `lane_alpha_nerv_mask` L1, `lane_alpha_wavelet_mask` L1, `track1_phase_a3_alt_mallat_wavelet` L2, `track1_phase_a6_selfcomp_blockfp_hyperprior` L2) have since been driven to 0; subagent C's `lane_12_v2_nerv_as_renderer` PASSES with all 8 fields declared inline in `notes`. 51 dedicated tests in `src/tac/tests/test_check_representation_lane_has_archive_grammar.py`. Memory: `feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md` + `feedback_check_124_representation_archive_grammar_landed_20260509.md`. Cross-refs `.omx/research/representation_integration_gap_audit_20260508_codex.md` + `feedback_substrate_vs_codec_composition_meta_pattern_20260508.md`.

**Subagent coherence-by-default (2026-05-09, STRICT @ 0 — flipped 2026-05-12):**

125. `check_subagent_landing_has_solver_wire_in` — Refuses post-2026-05-09 landing memos (`feedback_*_landed_<YYYYMMDD>.md` with `YYYYMMDD >= 20260509`) missing the 6 mandatory unified-Lagrangian wire-in hooks per CLAUDE.md "Subagent coherence-by-default" non-negotiable: (1) Sensitivity-map contribution, (2) Pareto constraint, (3) Bit-allocator hook, (4) Cathedral autopilot dispatch hook, (5) Continual-learning posterior update, (6) Probe-disambiguator (if 2+ defensible interpretations). Two opt-outs: `research_only=true` declared anywhere in the memo body OR per-hook `<HookAlias>: N/A — <rationale>` (rationale required; bare `<Alias>: N/A` is rejected). Negative acknowledgments such as `not wired`, `missing`, `deferred`, or `TODO` are rejected and do not count as wire-in declarations. Missing memory dirs warn and strict mode raises instead of silently returning OK. Bug class: pre-rule, lateral-leap landings produced research artifacts the planner could not see or compose; the new track became orphan work with no path into the meta-Lagrangian/Pareto/autopilot/continual-learning loop. The wire-in declaration forces every subagent to acknowledge each integration surface explicitly so a missing hook is visible at landing-time, not weeks later when the solver diverges from reality. STRICT-FLIPPED 2026-05-12 per UUU audit + FFFF Bug 3 bulk text-fix; orchestrator callsite is `strict=True`. Initial landing: warn-only per operator approval 2026-05-09; legacy-memo backfill drove live count to 0. 50 dedicated tests in `src/tac/tests/test_check_subagent_landing_has_solver_wire_in.py`. Memory: `feedback_unified_lagrangian_action_principle_GR_style_20260509.md` + `feedback_check_125_126_coherence_by_default_strict_landed_20260509.md`. Cross-refs `feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md` + `.omx/research/operator_decisions_executed_20260509.md`.

126. `check_lane_pre_registered_before_work_starts` — Refuses subagent commits in the last 50 commits AND current staged/dirty/untracked WIP whose introduced files under `src/tac/` / `tools/` / `experiments/` / `scripts/` (.py / .sh extensions) reference a `lane_<NAME>` token (inside a single- or double-quoted Python/shell string literal) that does NOT appear in `.omx/state/lane_registry.json` per CLAUDE.md "Lane maturity registry" lifecycle discipline + "Subagent coherence-by-default" anti-duplication primitive. Acceptance/exemption rules: (a) Test-fixture per-line exemption — files under `*/tests/*` paths (or `test_*` filenames) that carry `# FAKE_LANE_OK:<reason>` on the same line OR within 5 lines above are exempt; (b) file-level `# FAKE_LANE_OK_FILE:<reason>` is limited to Check #126's own fixture-heavy self-test file, not arbitrary test files; (c) Helper-name blocklist — common identifiers like `lane_id`/`lane_class`/`lane_registry`/`lane_maturity`/`lane_script`/`lane_tag`/`lane_claim_opened`/`lane_summary`/etc. are never treated as lane_id references (~75 entries in `_LANE_ID_REFERENCE_BLOCKLIST`); (d) Alias support — registry lanes with `aliases` (or `alias`) field accept the alias as a known lane_id. Bug class: parallel subagents inventing the same fictional lane_id silently because nothing prevents two of them from naming the same lane. Pre-registration via `tools/lane_maturity.py add-lane <id> --name <...> --phase <N>` at SKETCH (Level 0) is the coherence primitive — once a lane is in the registry, the second subagent sees it and either contributes to it or coordinates with the operator. STRICT-FLIPPED 2026-05-12 per UUU audit + FFFF Bug 3 bulk text-fix; orchestrator callsite is `strict=True`. Initial landing: warn-only per operator approval 2026-05-09; legacy-commit backfill drove live count to 0 (across 121 scanned files / 51 sources, including dirty and untracked WIP). 118 dedicated tests in `src/tac/tests/test_check_lane_pre_registered_before_work_starts.py`. Memory: `feedback_unified_lagrangian_action_principle_GR_style_20260509.md` + `feedback_check_125_126_coherence_by_default_strict_landed_20260509.md`. Cross-refs CLAUDE.md "Lane maturity registry" lifecycle discipline section + Catalog #90 (`check_lane_registry_consistent`).

**Custody validator + locked posterior writes (2026-05-09, codex round-2 HIGH 2 + MEDIUM, STRICT @ 0 — flipped 2026-05-12):**

127. `check_authoritative_tag_requires_custody_metadata` — Refuses code sites under `src/tac/`, `tools/`, or `experiments/` that perform `<expr> in AUTHORITATIVE_TAGS` membership checks WITHOUT local line-window routing through `tac.continual_learning.ContestResult.validate_custody` / `validate_custody_verdict` / `posterior_update` / `posterior_update_locked`. Acceptance: (a) same-line or nearby custody-routing call; (b) same-line waiver `# CUSTODY_VALIDATOR_OK:<reason>` for intentionally tag-only inventory/filter sites; (c) the canonical implementation file (`src/tac/continual_learning.py`), tests, and generated `experiments/results/` artifacts are excluded. Whole-file validator-token accept is forbidden because it lets one valid helper mask later tag-only promotion sites. Bug class: codex round-2 HIGH 2 (2026-05-09). Tag-only predicate `is_authoritative()` accepted CPU tags on non-GHA Linux hosts AND CUDA tags with axis mismatch — both produce miscategorized empirical anchors that promote into the posterior. The new typed validator (`CustodyVerdict` with `refused_class` taxonomy: `tag_axis_mismatch` / `cpu_tag_non_gha_linux` / `cuda_tag_unknown_substrate` / `macos_substrate` / `missing_metadata` / `advisory_grade`) checks (tag, axis, hardware_substrate) jointly. STRICT-FLIPPED 2026-05-12 per UUU audit + FFFF Bug 3 bulk text-fix; orchestrator callsite is `strict=True`. Initial landing: warn-only per directive 2026-05-09; live count driven to 0. 21 focused #127/#128 tests in `src/tac/tests/test_preflight_custody_validator_and_locked_writes.py`. Memory: `feedback_codex_round2_custody_concurrency_fix_landed_20260509.md`.

128. `check_continual_learning_writes_use_lock` — Refuses code sites under `src/tac/`, `tools/`, or `experiments/` that call `save_posterior(...)` directly without a local `_posterior_lock` context visible near the direct write. Same-line waiver: `# SAVE_POSTERIOR_LOCKED_OK:<reason>` exempts single-writer / externally-serialized sites. Imports of `save_posterior` (without calling) and diagnostic/doc strings are accepted. The canonical implementation file, tests, and generated `experiments/results/` artifacts are excluded. File-level co-owner accept via any `posterior_update_locked` / `_posterior_lock` token is forbidden because it lets one safe writer mask later bare writes. Bug class: codex round-2 MEDIUM (2026-05-09). Bare `save_posterior(...)` under parallel harvesters silently drops concurrent updates because the load -> mutate -> write cycle is not atomic — two harvesters loading the same stale posterior + each updating their distinct anchor + each replacing -> ONE anchor's update silently dropped. The locked path uses `fcntl.flock(LOCK_EX)` on `.omx/state/.continual_learning.lock`, reloads inside the lock, re-runs the duplicate check, and writes to a UNIQUE temp file (`.tmp.<uuid12>`). True multiprocessing tests (4-proc spawn pool) prove distinct anchors all land + same anchor is idempotent across processes. STRICT-FLIPPED 2026-05-12 per UUU audit + FFFF Bug 3 bulk text-fix; orchestrator callsite is `strict=True`. Initial landing: warn-only per directive 2026-05-09; live count driven to 0. 21 focused #127/#128 tests in `src/tac/tests/test_preflight_custody_validator_and_locked_writes.py`. Memory: `feedback_codex_round2_custody_concurrency_fix_landed_20260509.md`.

130. `check_no_tag_only_custody_validation` — Extends #127 beyond `AUTHORITATIVE_TAGS` to broader tag/grade gates such as `evidence_grade in {...}` and `tag.startswith("[contest-...")`. The predicate must have local line-window custody context (`validate_custody`, `validate_custody_verdict`, `posterior_update`, `posterior_update_locked`, `is_promotable_exact_cuda_evidence`, `promotable_exact_cuda_evidence_blockers`, `archive_sha256`, or fail-closed blocker/error context) or a same-line `# CUSTODY_VALIDATOR_OK:<reason>` waiver. Broad whole-file validator-token accept is forbidden. Bug class: proactive custody sweep found the #127 META class can recur through grade strings even when `AUTHORITATIVE_TAGS` is absent. STRICT-FLIPPED 2026-05-12 per UUU audit + FFFF Bug 3 bulk text-fix; orchestrator callsite is `strict=True`. Live count: 0 → STRICT (across 8 scanned files). 29 focused #127/#128/#130/#131 tests in `src/tac/tests/test_preflight_custody_validator_and_locked_writes.py`. Memory: `feedback_proactive_custody_concurrency_audit_landed_20260509.md`; ledger `.omx/research/proactive_custody_concurrency_audit_20260509.md`.

131. `check_no_bare_writes_to_shared_state` — Refuses bare writes to known shared mutable state paths (`.omx/state/*`, `lightning_active_jobs`, `vastai_active_instances`, `active_lane_dispatch_claims`, `lane_registry.json`, etc.) unless the specific write has a local preceding-line lock/context token (`fcntl.flock`, `LOCK_EX`, `_posterior_lock`, `_lightning_state_lock`, `_active_jobs_lock`, `FileLock`) or a same-line `# BARE_WRITE_OK:<reason>` waiver. Canonical helper names such as `register_job` or `register_instance` do not waive an adjacent bare write; replace the bare write with the helper or put the write inside a real lock. Broad whole-file lock-token accept is forbidden. Bug class: proactive sweep found six additional shared-state surfaces with the same lost-update race as #128: `LIGHTNING_ACTIVE_JOBS_PATH` dispatch/harvest callers, `LIGHTNING_STATE`, a Vast.ai tracker bypass, and `instance_setup_first_seen`. STRICT-FLIPPED 2026-05-12 per UUU audit + FFFF Bug 3 bulk text-fix; orchestrator callsite is `strict=True`. Live count: 0 → STRICT (across 84 scanned files). New canonical helper: `src/tac/deploy/lightning/active_jobs_state.py`. Memory: `feedback_proactive_custody_concurrency_audit_landed_20260509.md`; ledger `.omx/research/proactive_custody_concurrency_audit_20260509.md`.
132. `check_locked_writes_preserve_deletions` — Refuses any "locked save" helper (function name matching `_save_*_first_seen` / `_save_*_state` / `update_*_locked` / `_save_*` patterns) whose body does `existing.update(data)` / `previous.update(data)` / `loaded.update(data)` / `prev.update(data)` / `current.update(data)` / `stored.update(data)` / `on_disk.update(data)` inside an fcntl-locked region (`fcntl.flock` / `LOCK_EX` / `_active_jobs_lock` / `_posterior_lock` token visible in the surrounding 30+5 line window). The deletion-merge anti-pattern silently re-introduces stale keys the caller deliberately pruned; the only safe contract for "caller passes the full post-prune map" helpers is TRANSACTIONAL REPLACE. Same-line waiver: `# DELETION_MERGE_OK:<reason>` for the rare genuinely-additive case (counter-bump only; never replace). Sister of #131 (which catches BARE/UNLOCKED writes — this one catches LOCKED but DELETION-LOSING writes) and #128 (which covers `continual_learning.save_posterior` specifically). Bug class: codex round-3 HIGH 1 (2026-05-09). `scripts/verify_vast_instances.py::_save_setup_first_seen` reloaded the on-disk file inside the fcntl lock and merged via `existing.update(data)`; pruned `first_seen` rows for instances no longer in the tracker silently resurfaced; `--auto-destroy-stale` then targeted FRESH instances that inherited an old age. The fix flipped to direct write of `data` inside the lock. Live count after the fix: 0. STRICT-FLIP per "fix all findings regardless of severity" operator approval 2026-05-09. 10 dedicated tests in `src/tac/tests/test_codex_round3_check_132_locked_writes_preserve_deletions.py`. Memory: `feedback_codex_round3_findings_fix_landed_20260509.md`.

**Codex round 4 META-meta gates (2026-05-09, STRICT @ 0):**

133. `check_no_excluded_writers_in_check_131_accept_list` — META-meta of #131. Iterates every entry in `_BARE_WRITE_CANONICAL_HELPERS` and verifies the file actually contains a canonical lock-pattern token (`fcntl.flock`, `LOCK_EX`, `_active_jobs_lock`, `_active_vms_lock`, `_lightning_state_lock`, `_posterior_lock`, `posterior_update_locked`, `update_active_jobs_locked`, `update_active_vms_locked`, `register_active_vm_record`, `register_job`, `register_instance`, `claim_lane_dispatch`, `threading.Lock`) OR is in the deferred-rationale dict `_CHECK_131_EXEMPT_AUDIT_DEFERRED_RATIONALE` (rationale + follow-up lane reference) OR has a file-wide `# CHECK_131_EXEMPT_AUDIT_OK:<reason>` waiver. Bug class: codex round-4 HIGH 1 (2026-05-09). The previous exempt list named `src/tac/deploy/azure/azure_dispatch.py` as "already locked", but the file did bare `write_text` to `.omx/state/azure_active_vms.json` with NO fcntl context. Strict #131 reported the bare-write META class extinct while concurrent Azure provisions could still drop VM rows. The fix landed `src/tac/deploy/azure/active_vms_state.py` (canonical fcntl-locked Azure helper sister to Lightning's `active_jobs_state.py`) + refactored `azure_dispatch.py` to delegate to `register_active_vm_record` + swapped the exempt-list entry. Live count after the fix: 0. STRICT-FLIP per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable. 12 dedicated tests in `src/tac/tests/test_codex_round4_check_133_check_131_exempt_audit.py`. Memory: `feedback_codex_round4_findings_fix_with_self_protection_landed_20260509.md`.
134. `check_phase3_dispatch_gate_fail_closed` — Refuses `Phase3DispatchGate(...)` constructor calls under `src/tac/`, `tools/`, `experiments/`, or `scripts/` (excluding the canonical scaffold module `src/tac/phase3/joint_scorer_renderer_codec.py` and tests) that do NOT pass `unsafe_test_only=True` AND do NOT supply the FULL production precondition kwarg set (`phase2_anchor_verified`, `phase2_anchor_score`, `phase2_anchor_evidence_path`, `distillation_gap_estimate`, `distillation_gap_evidence_path`, `operator_approved_gpu_budget_usd`, `aaf68f37_verdict_clean`, `aaf68f37_verdict_evidence_path`, `phase3_council_deliberation_path`). Same-line waiver: `# PHASE3_GATE_OK:<reason>` for rare integration callsites that wrap construction in additional validation. Bug class: codex round-4 MEDIUM 1 (2026-05-09). The previous `Phase3DispatchGate.__post_init__` was a no-op with the comment "tests need to construct with a permissive gate" — any future trainer/dispatcher could silently bypass every precondition by forgetting to call `gate.check()`. The fix made construction fail-closed with an explicit `unsafe_test_only=True` opt-out for tests; refactored `JointScorerRendererCodecScaffold.__post_init__` to enforce the gate as defence-in-depth. Live count after the fix: 0. STRICT-FLIP per the same self-protection mandate. 14 dedicated tests in `src/tac/tests/test_codex_round4_check_134_phase3_gate_fail_closed.py`. Memory: `feedback_codex_round4_findings_fix_with_self_protection_landed_20260509.md`.
135. `check_setup_first_seen_uses_transactional_update_inside_lock` — Refuses any module that defines BOTH a `_load_*_first_seen` and a `_save_*_first_seen` helper AND has a CALL SITE pattern of "load outside any lock, perform work, save under lock" — the lost-update anti-pattern for two overlapping invocations. Acceptance: (a) the enclosing function uses the canonical transactional helper (`update_*_first_seen_locked` / `update_setup_*_locked` / `update_active_jobs_locked` / `update_active_vms_locked`), (b) the load is INSIDE a `with <lock>` context manager at less indent than the load OR after an imperative `fcntl.flock(..., LOCK_EX)` call at <= load indent, (c) the enclosing function does NOT call any `_save_*_first_seen` (read-only consumer), or (d) same-line `# SETUP_FIRST_SEEN_LOST_UPDATE_OK:<reason>` waiver. Sister of #132 (which catches deletion-merge in the SAVE function itself) and #131 (which catches bare writes outside any lock). Bug class: codex round-4 MEDIUM 2 (2026-05-09). The round-3 #132 fix made `_save_setup_first_seen` write transactionally INSIDE the lock, but `scripts/verify_vast_instances.py::main` still loaded the on-disk state OUTSIDE the lock, ran per-instance verify (~minutes), then saved at end-of-run — a lost-update race for two overlapping verifier runs. The fix added `update_setup_first_seen_locked` (load + merge + prune + save inside ONE fcntl-locked window with KEEP-OLDER timestamp merge semantics) + `remove_setup_first_seen_locked` (locked transactional removal) + refactored `main()` to use them. Live count after the fix: 0. STRICT-FLIP per the same self-protection mandate. 11 dedicated tests in `src/tac/tests/test_codex_round4_check_135_setup_first_seen_transactional.py` including a multiprocessing regression test that proves KEEP-OLDER convergence under concurrent verifier runs. Memory: `feedback_codex_round4_findings_fix_with_self_protection_landed_20260509.md`.

**Defense-in-depth on codex round-3 fix (2026-05-09, STRICT @ 0):**

136. `check_custody_gate_accept_tokens_concrete_only` — Defense-in-depth on codex round-3 HIGH 2. Uses Python's `ast` module to walk every module-level `Assign` / `AnnAssign` node whose target name matches `_*VALIDATOR_TOKENS` / `_*VALIDATOR_PATTERNS` / `_*VALIDATOR_FNS` / `_*ACCEPT_TOKENS` / `_*CUSTODY_TOKENS` / `_*GATE_TOKENS` / `_*GUARD_TOKENS` and refuses any string-literal entry whose value is one of the forbidden generic identifiers (`blockers`, `errors`, `failures`, `validations`, `warnings`, `issues`, `problems`, `results`, `checks`, `messages`, `verdicts`, `reasons`). Concrete validator tokens — function-call patterns like `"validate_custody("`, attribute references like `"sha256_file("`, `"archive_sha256"` — are accepted; AST-based scanning correctly ignores comments / docstrings / string literals containing parentheses. Same-line waiver: `# ACCEPT_TOKENS_CONCRETE_OK:<reason>` on the entry line OR on the assignment-target line. Bug class: round-3 HIGH 2 patched ONE accept-list (`_TAG_GRADE_LOCAL_VALIDATOR_TOKENS` lost `"blockers"` / `"errors"`). This META gate refuses any future addition of a generic-token bypass to ANY accept-list at any of the 8 named-suffix patterns, in any of `src/tac/`, `tools/`, `experiments/`, `scripts/`. Live count at landing: 0. STRICT-FLIP per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable. 18 dedicated tests in `src/tac/tests/test_check_136_custody_gate_accept_tokens_concrete_only.py`. Memory: `feedback_production_hardening_polish_defense_in_depth_landed_20260509.md`. Cross-ref Catalog #130 (call-site missing-validator) + #133 (#131 exempt-list audit).
137. `check_remote_dispatch_runbooks_no_local_cuda_probe_default` — Defense-in-depth on codex round-3 MEDIUM 1. Scans `scripts/remote_lane_*.sh` (the local dispatch drivers; sister scripts like `remote_archive_only_eval.sh` are out-of-scope because they run ON the GPU) for invocations of `probe_nvdec.sh` / `nvidia-smi` / `nvcc --version` / `nvcc -V` / `torch.cuda.is_available()` and refuses each unless the same-line OR preceding-30-lines window contains a guard token: `LOCAL_CUDA_WORKER`, `DRY_RUN`, `vastai exec`, `lightning ssh`, `modal run`, `ssh `, `$REMOTE_HOST`, `${REMOTE_HOST`. Same-line waiver: `# LOCAL_CUDA_PROBE_OK:<reason>` for rare cases where the operator workstation legitimately has CUDA + DALI installed. Bug class: round-3 MEDIUM 1 patched ONE runbook (`remote_lane_avvideodataset_cuda_path_mechanism_discriminator.sh`). This META gate refuses any future `scripts/remote_lane_*.sh` from introducing an unguarded local CUDA/NVDEC probe — the probe belongs in the REMOTE provider bootstrap, not the local dispatch driver. Live count at landing: 0. STRICT-FLIP per the same self-protection mandate. 16 dedicated tests in `src/tac/tests/test_check_137_remote_dispatch_no_local_cuda_probe.py`. Memory: `feedback_production_hardening_polish_defense_in_depth_landed_20260509.md`. Cross-ref CLAUDE.md "Remote code parity" + "Vast.ai cost paranoia".
138. `check_state_writers_strict_load_for_mutating_path` — Defense-in-depth on codex round-3 MEDIUM 2. Scans `src/tac/**/*.py` and `tools/**/*.py` for function definitions whose name matches a state-writer pattern (`update_*_locked`, `_save_*`, `upsert_*`, `register_*`, `mark_*_terminal`) AND whose body performs a load + write (`_load_` / `read_text(` / `read_bytes(` / `json.load` AND `write_text(` / `write_bytes(` / `json.dump` / `atomic_write`). For each in-scope writer, requires either: (a) a strict-load helper token in the body (`_strict(`, `load_active_jobs_strict`, `load_active_vms_strict`, `load_first_seen_strict`, `load_setup_first_seen_strict`, `load_state_strict`, `ActiveJobsCorruptError`, `ActiveVmsCorruptError`, `CorruptStateError`, `raise PreflightError`); OR (b) a same-line `# STATE_WRITER_STRICT_LOAD_OK:<reason>` waiver on the `def` line. Bug class: round-3 MEDIUM 2 patched ONE writer (`active_jobs_state.py::update_active_jobs_locked` now uses `load_active_jobs_strict`). This META gate refuses any future writer from silently resetting corrupt state via a `load_*` that returns `[]`/`{}` on JSON parse failure — the consequence is dropping every active row when the dispatcher write happens after a corrupt read. Live count at landing: 0. STRICT-FLIP per the same self-protection mandate. 12 dedicated tests in `src/tac/tests/test_check_138_state_writers_strict_load.py`. Memory: `feedback_production_hardening_polish_defense_in_depth_landed_20260509.md`. Cross-ref Catalog #131/#132/#133/#135 (sister concurrency/locking gates).

**Codex round-5 findings fix + self-protection (2026-05-09, STRICT @ 0):**

139. `check_packet_compiler_no_op_proof_promotes_to_blocker` — Refuses any packet-compiler-style finalize function that calls `_build_no_op_proof(...)` AND mutates a `blockers` list but does NOT append at least one of the canonical promotion tags `inflate_does_not_consume_archive_bytes` / `no_op_detector_failed`. The check is signature-aware: only functions invoking the canonical `_build_no_op_proof` constructor are scanned; alternate-tag-scheme verifiers (e.g. `monolithic_packet_closure_gate._runtime_consumption_summary` using `runtime_proof_*` tags, `submission_archive._typed_sidechannel_contract_row` using typed-contract blockers) are out-of-scope. Same-line waiver on the `def` line: `# NO_OP_PROOF_ADVISORY_OK:<reason>` for probe-only packets that explicitly opt to keep the proof advisory. Bug class: codex round-5 HIGH 1 (2026-05-09). `tac.phase1_packet_compiler._finalize_packet_result` computed `no_op_proof.runtime_consumes_bytes` and `no_op_proof.no_op_detector_passed` but never converted `False` results into entries in `blockers`. The CLI exit gate (`tools/build_phase1_packet_compiler.py::main`) read only `result.blockers` so a no-op inflate.py exited 0 and burned eval spend. The fix promotes both failure modes to first-class blockers AND adds an executable byte-mutation smoke (`_verify_runtime_consumes_payload_bytes_executable`) that mutates a single archive byte and observes whether downstream inflate output changes. Live count at landing: 0. STRICT-FLIP per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable. 14 dedicated tests in `src/tac/tests/test_codex_round5_check_139_packet_no_op_proof_promotes_to_blocker.py`. Memory: `feedback_codex_round5_findings_fix_with_self_protection_landed_20260509.md`. Closes the round-5 HIGH 1 fail-open class.

140. `check_state_writers_own_their_lock_end_to_end` — Refuses any function whose name matches `_save_*` (or `save_*_state`) AND whose body contains a comment/docstring carrying one of the "MUST be called inside the lock" / "caller is responsible for the lock" / "called inside ... locked" contract phrases — UNLESS the body ALSO contains a runtime enforcement: a sister `_lock_held()` predicate, `_lock_depth > 0` check, `fcntl.flock` acquisition, OR a `raise RuntimeError` / `raise PreflightError` triggered by the predicate. Comment-only contracts are FORBIDDEN per CLAUDE.md. Same-line waiver on the `def` line: `# CALLER_LOCK_ENFORCED_OK:<reason>` for genuinely-single-process unittest scaffolds. Bug class: codex round-5 HIGH 2 (2026-05-09). `LightningDispatcher._save_state` documented "MUST be called inside `_lightning_state_lock`" but did NOT runtime-enforce; `scripts/launch_lane_lightning.py` violated the contract by doing `sessions = dispatcher.list_sessions(); ...; dispatcher._save_state(sessions)` outside any lock — concurrent register/remove rows landed between the two and were silently dropped. The fix paired the lock context manager with an in-process depth counter (`_lightning_state_lock_depth`) and a sister `_lightning_state_lock_held()` predicate; `_save_state` raises `RuntimeError` if not held. New canonical `update_session_locked` / `update_sessions_locked` API owns the full lock-load-mutate-save cycle; `scripts/launch_lane_lightning.py` was rewritten to use it. Sister fixes in `tac.deploy.lightning.active_jobs_state._save_active_jobs` and `tac.deploy.azure.active_vms_state._save_active_vms_atomic` apply the same lock-held assertion. Live count at landing: 0. STRICT-FLIP per the same self-protection mandate. 16 dedicated tests in `src/tac/tests/test_codex_round5_check_140_state_writers_own_their_lock.py`. Memory: `feedback_codex_round5_findings_fix_with_self_protection_landed_20260509.md`. Closes the round-5 HIGH 2 lost-session-row class.

141. `check_state_helper_paths_explicit` — Refuses any cross-module state-helper call where (1) the calling module defines a module-level tracker constant matching `*_PATH` / `*_LOCK` (with a `.omx/state` / `active_*` / `tracker` value), (2) the calling module imports a helper from `tac.deploy.azure.active_vms_state` / `tac.deploy.lightning.active_jobs_state` / `tac.continual_learning` whose name appears in `_STATE_HELPER_FN_NAMES_WITH_PATH_KW` (`register_active_vm_record`, `unregister_active_vm_by_name`, `load_active_vms`, `load_active_vms_strict`, `update_active_vms_locked`, `register_job`, `upsert_job`, `mark_job_terminal`, `load_active_jobs`, `load_active_jobs_strict`, `update_active_jobs_locked`), AND (3) the call does NOT thread `path=` (and `lock_path=` when accepted). Same-line waiver: `# STATE_HELPER_PATH_OK:<reason>` for the rare deliberate-canonical-path case. Bug class: codex round-5 HIGH 3 (2026-05-09). `azure_dispatch.py` defined its own `ACTIVE_VMS_PATH` constant AND imported `register_active_vm_record` / `unregister_active_vm_by_name` / `load_active_vms` from `tac.deploy.azure.active_vms_state` (which has its own `ACTIVE_VMS_PATH`) — without threading `path=`. Tests that monkeypatched `azd.ACTIVE_VMS_PATH` silently observed the helper's canonical path; recovery/cleanup tooling that constructed an alternative tracker location was equally affected. The fix adds `ACTIVE_VMS_LOCK = ACTIVE_VMS_PATH.with_suffix(...)` sibling constant, then threads `path=ACTIVE_VMS_PATH, lock_path=ACTIVE_VMS_LOCK` through every helper call in `azure_dispatch.py`. Live count at landing: 0. STRICT-FLIP per the same self-protection mandate. 11 dedicated tests in `src/tac/tests/test_codex_round5_check_141_state_helper_paths_explicit.py`. Memory: `feedback_codex_round5_findings_fix_with_self_protection_landed_20260509.md`. Closes the round-5 HIGH 3 path-override-bypass class.

**Codex round-6 findings fix + self-protection (2026-05-09, STRICT @ 0):**

142. `check_unsafe_test_only_restricted_to_test_paths` — Refuses any `Phase3DispatchGate(...)` constructor callsite outside a recognized test path (`*/tests/*`, `test_*.py`, `*_test.py`) that passes `unsafe_test_only=True` without ALSO passing `unsafe_test_only_path_audit_waived=True` OR carrying a same-line `# PHASE3_GATE_UNSAFE_PATH_WAIVED:<reason>` waiver. Bug class: codex round-6 HIGH 1 (2026-05-09). The round-4 fix (catalog #134) made `Phase3DispatchGate.__post_init__` fail-closed with `unsafe_test_only=True` as the test-fixture escape hatch. The structural check for #134 accepted the kwarg regardless of caller path — so any production caller under `tools/` / `experiments/` / `scripts/` / `src/tac/` could simply toggle the kwarg to bypass every Phase 3 dispatch precondition. The fix adds a path-audit guard inside `__post_init__`: `_caller_path_is_test()` walks the call stack out of the joint_scorer module (skipping dataclass-synthesized `<string>` frames AND any frame in the canonical module itself) and refuses `unsafe_test_only=True` from non-test paths. The double-opt-in `unsafe_test_only=True AND unsafe_test_only_path_audit_waived=True` is reserved for explicit operator-reviewed fixtures. STRICT-FLIP per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable. Live count at landing: 0. 15 dedicated tests in `src/tac/tests/test_codex_round6_high1_unsafe_test_only_path_audit.py`. Memory: `feedback_codex_round6_findings_fix_with_self_protection_landed_20260509.md`. Cross-ref Catalog #134 (the round-4 sister gate).
143. `check_paid_job_register_before_submit` — Refuses any function in a Lightning dispatcher file (`*lightning*.py`) that calls `Job.run(...)` (paid Lightning submit) WITHOUT calling `register_pending_job_locked(...)` first in the SAME function body. AST-aware: distinguishes real `Job.run(...)` Call nodes from docstrings/comments/error messages; restricts scan to filenames containing `lightning` so non-Lightning files with the literal substring are not affected. Same-line waiver on the `Job.run` line: `# JOB_RUN_BEFORE_REGISTER_OK:<reason>` reserved for the submit-helper itself (whose calling function owns the pending-row contract). Bug class: codex round-6 HIGH 2 (2026-05-09). With round-3's strict `register_job` (which raises on corrupt active-jobs state), the previous flow — submit-then-persist — meant a corrupt tracker file → paid job created but tracker write fails → invisible orphan paid job. The fix lands the canonical create-pending-row-before-submit pattern in `tac.deploy.lightning.active_jobs_state`: `register_pending_job_locked(metadata)` writes a `status=pending` row BEFORE submit (corrupt tracker → refuse submit, no orphan); `update_pending_to_active_locked(job_name, ...)` promotes pending → active POST-submit; `cancel_pending_job_locked(job_name)` drops the pending row when submit raises before any network call. Both `experiments/arch_shrink_x0.4_lightning_full.py` and `experiments/lossy_coarsening_lightning_cuda_test.py` were refactored to use the pending-row pattern. STRICT-FLIP per the same self-protection mandate. Live count at landing: 0. 15 dedicated tests in `src/tac/tests/test_codex_round6_high2_paid_job_register_before_submit.py`. Memory: `feedback_codex_round6_findings_fix_with_self_protection_landed_20260509.md`. Cross-ref Catalog #131/#132/#138 (sister concurrency/locking gates) + round-3 MEDIUM 2 fix (#138 — strict load that this fix relies on).
144. `check_setup_first_seen_no_split_transactions` — Refuses any function that calls BOTH a SETUP-first-seen observed-insertion helper (`update_setup_first_seen_locked` or `_save_setup_first_seen`) AND a left-removal helper (`remove_setup_first_seen_locked`) within the same function body. Same-line waiver on the function-def line: `# SETUP_FIRST_SEEN_SINGLE_TXN_OK:<reason>` for the canonical helper itself (which legitimately combines both inside one lock). Bug class: codex round-6 MEDIUM 1 (2026-05-09). The round-4 fix (catalog #135) introduced separate helpers for observed-insertion and left-removal, but `scripts/verify_vast_instances.py::main` called BOTH in sequence — two SEPARATE locked transactions. Two overlapping verifier runs that disagree on the same id (one observes SETUP, the other observes leaving SETUP) can drop first-seen timestamps the other inserted because the two transactions are not atomic together. The fix expands `update_setup_first_seen_locked` to accept BOTH `observed_setup_ids` AND `left_setup_ids` in a single transaction with explicit `monotonic_conflict_rule="prefer_observed"` (default); same-call same-id conflicts are resolved deterministically inside the lock (observed wins because it has direct SETUP evidence right now). `main()` was refactored to make ONE call. The deprecated `remove_setup_first_seen_locked` shim is kept as a thin self-locked wrapper for callers that only know about ids to remove (no observed/tracked sets). STRICT-FLIP per the same self-protection mandate. Live count at landing: 0. 8 dedicated tests in `src/tac/tests/test_codex_round6_medium1_setup_first_seen_single_txn.py` including a multiprocessing regression that proves coherent state under concurrent observe/leave races. Memory: `feedback_codex_round6_findings_fix_with_self_protection_landed_20260509.md`. Cross-ref Catalog #135 (the round-4 sister gate).
145. `check_preflight_cli_default_scope_is_bounded_dev` — Refuses any AST-detectable change to `parser.add_argument("--scope", default=...)` in `src/tac/preflight.py` that would silently re-introduce the regressed DX-budget-violating default. Same-line waiver on the `add_argument` line OR within 5 lines after (multi-line call): `# PREFLIGHT_CLI_SCOPE_DEFAULT_OK:<reason>` for intentional override with explicit operator review. Bug class: codex round-6 MEDIUM 2 (2026-05-09). The canonical default is the bounded-dev variant per codex round-6 supersession — `--scope dev` is the routine fast gate; `--scope all|release` is the explicit exhaustive sweep behind `--allow-slow-preflight`. **FFFF Bug 5 cleanup (2026-05-12 per UUU audit):** the historical alias `check_preflight_cli_default_scope_is_all` was a DEFINED_BUT_NOT_INVOKED orphan that forwarded to the canonical `check_preflight_cli_default_scope_is_bounded_dev`; the alias was deleted to extinct the orphan-function class. The canonical name `check_preflight_cli_default_scope_is_bounded_dev` is wired into `preflight_all()` strict=True. STRICT @ 0 — live count: 0. 9 dedicated tests in `src/tac/tests/test_codex_round6_medium2_preflight_cli_default_scope.py`. Memory: `feedback_codex_round6_findings_fix_with_self_protection_landed_20260509.md`.

**Phase 1 trainer contest-compliant runtime emission (2026-05-09, operator decision B, STRICT @ 0):**

146. `check_phase1_trainer_runtime_emits_contest_compliant_inflate` — Operator decision B 2026-05-09. Refuses any change to `experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py::_write_runtime` that would re-introduce the broken research-only-no-export scaffold pattern: (a) inflate.sh template missing any of `$1` (archive_dir) / `$2` (output_dir) / `$3` (file_list) — accepts braced `${1}` etc. and the helper-variable form `$DATA_DIR / $OUTPUT_DIR / $FILE_LIST`; (b) inflate.sh template missing `set -euo pipefail` / `set -e` per `check_shell_set_e_present`; (c) inflate.sh template using single-arg passthrough `"$HERE/inflate.py" "$@"` instead of explicit positional handoff; (d) inflate.py template containing FORBIDDEN_INFLATE_TOKENS (`PoseNet`, `SegNet`, `from upstream.modules`, `import upstream.modules`, `rgb_to_yuv6`, `EfficientNet`, `FastViT`) per CLAUDE.md strict-scorer-rule; (e) inflate.py template missing per-video loop pattern (`for line in file_list`, `splitlines()`, `while IFS= read`, `for base in`). Same-line `# CHECK146_WAIVED:<reason>` waiver honored for explicit research-only branches. Bug class: pre-fix the trainer emitted `exec uv ... "$HERE/inflate.py" "$@"` (one-arg passthrough) and `inflate(src_bin, dst_raw)` single-video signature; the Phase 1 packet compiler `tac.phase1_packet_compiler` correctly REFUSED the scaffold with "missing required positional args $1 (archive_dir) $2 (output_dir) $3 (file_list); contest contract requires all 3"; Phase 1 GPU dispatch was blocked until the trainer was rewritten in this same commit-batch. Live count at landing: 0. STRICT @ 0 per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable. Tests: `tests/paradigm_delta_epsilon_zeta/test_phase1_trainer_write_runtime_fix.py`. Memory: `feedback_phase1_trainer_write_runtime_fix_landed_20260509.md`. Cross-ref `feedback_phase1_packet_compiler_landed_20260509.md` (the consumer that emitted the rejection blocker), HNeRV parity discipline lessons 2/3/4 (export-first, monolithic single-file substrate-engineering exception, ≤100 LOC inflate budget).

**Codex round 7+8 findings fix + self-protection (2026-05-09, STRICT @ 0):**

147. `check_lightning_submit_cancel_only_before_network` — Refuses any function in a Lightning dispatcher file (`*lightning*.py`) where a `try:` block invokes `submit_lightning_job(...)` or `Job.run(...)` AND an attached `except` clause both (1) catches `BaseException` (or is a bare `except:`) AND (2) invokes `cancel_pending_job_locked(...)` in its body. AST-aware: `Try` nodes are walked top-down; only handlers attached to a try-block whose body really contains an ambiguous-submit Call are flagged. Same-line waiver `# CANCEL_PENDING_PRE_NETWORK_OK:<reason>` honored on the `try:` / `except` / `cancel` line. Pre-network narrow handlers (`except _PreNetworkSubmitError`) are exempt by construction. Bug class: codex round 7+8 HIGH 1 (2026-05-09). The round-6 fix (#143) made dispatchers register a pending row BEFORE submit, but the cancel-on-exception logic was too aggressive — `except BaseException` around `submit_lightning_job(...)` (which includes `Job.run(...)`) could silently delete the only harvester-visible row for a real paid Lightning job after an SDK timeout / post-create API error / KeyboardInterrupt mid-Job.run. The fix splits the dispatcher exception routing: `_PreNetworkSubmitError` (raised by the new `_prepare_lightning_submit_environment` step) → safe `cancel_pending_job_locked`; residual `BaseException` (anything escaping `Job.run`) → new `mark_pending_failed_unknown_billing_locked` API which preserves the row with `status=failed_unknown_billing` for forensic recovery against the Lightning dashboard. STRICT-FLIP per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable. Live count at landing: 0. 19 dedicated tests in `src/tac/tests/test_codex_round78_check_147_lightning_submit_cancel_pre_network.py` + 9 helper-API tests in `src/tac/tests/test_codex_round78_mark_pending_failed_unknown_billing.py`. Memory: `feedback_codex_round78_findings_fix_with_self_protection_landed_20260509.md`. Cross-ref Catalog #143 (the round-6 sister gate this layer extends).

148. `check_vastai_tracker_strict_load` — Refuses any function in `src/tac/vastai_tracker.py` whose body calls `_write_records(...)` (mutation) inside a body that also invokes the LOSSY `_load_records(...)` loader, UNLESS (1) the body also routes through `load_active_instances_strict(...)`, OR (2) the function carries a same-line waiver `# VASTAI_TRACKER_STRICT_LOAD_OK:<reason>` on its `def` line. AST-aware Call collection avoids docstring/comment false positives. Bug class: codex round 7 HIGH 2 (2026-05-09). The previous `_load_records` returned `[]` on malformed JSON; `register_instance` then silently overwrote the corrupt file with a single new record, dropping every previously-tracked active instance (and making `tools/vastai_orphan_cleanup.py` unable to find them). The fix lands a new `tac.vastai_tracker.load_active_instances_strict` raising `VastaiTrackerCorruptError` on JSON parse failure (mirrors `tac.deploy.lightning.active_jobs_state.load_active_jobs_strict`); `register_instance` and `remove_instance` use it inside the fcntl lock and quarantine corrupt files to `<path>.corrupt.<utc>`. `scripts/launch_lane_on_vastai.py::register_in_tracker` now catches `VastaiTrackerCorruptError` with a loud orphan-recovery banner so the operator manually reconciles the live Vast.ai instance set against the quarantined file. Sister of Catalog #138 (`check_state_writers_strict_load_for_mutating_path`) scoped to the vastai_tracker module specifically because the loader name there diverges from the `_load_*` prefix #138 detects. STRICT-FLIP per the same self-protection mandate. Live count at landing: 0. 21 dedicated tests in `src/tac/tests/test_codex_round78_check_148_vastai_tracker_strict_load.py`. Memory: `feedback_codex_round78_findings_fix_with_self_protection_landed_20260509.md`. Cross-ref Catalog #138 (the sister strict-load gate).

**Phase B Option C operator decision (2026-05-09, STRICT @ 0):**

150. `check_phase_b_auth_memo_in_repo` — Refuses any caller of `phase_b_preconditions_status(...)` under `src/tac/`, `tools/`, `experiments/`, or `scripts/` that passes `auth_memo_path=` pointing to a literal anchored at `~/.claude`, `/tmp/`, `/var/tmp/`, `/private/tmp/`, OR an inline `Path.home() / ".claude/..."` BinOp expression. AST-aware: the kwarg value is extracted from `Constant`, `JoinedStr` (f-string literals), `Call(Path(...))`, and `BinOp(Path.home() / ".claude/...")` shapes; tests / vendored intake / OSS export mirror paths are excluded; the implementation files (`src/tac/lane_12_v2_nerv_as_renderer.py` + `src/tac/preflight.py`) are exempt by construction. Same-line waiver `# PHASE_B_AUTH_MEMO_OK:<reason>` on the kwarg / call line for the rare `consult_session_state=True` legacy fallback. Bug class: codex round 8 HIGH 2 (2026-05-09). a6535b1ed (commit `c3ab229e`) intentionally LANDED `phase_b_preconditions_status(consult_session_state=True)` as default, which scans `~/.claude/projects/.../memory` for any `feedback_*.md` body carrying `operator_phase_b_authorization=true`. Codex round 8 flagged this as non-hermetic (machine-dependent), spoofable (any memo body could match), and machine-state-dependent. Per CLAUDE.md "Design decisions — non-negotiable", this is a council-grade tradeoff (NOT a clear bug). Operator decision 2026-05-09 via AskUserQuestion: **Option C compromise** — keep `True` default (preserves a6535b1ed's "real" gate behavior + back-compat) PLUS accept explicit `phase_b_preconditions_status(auth_memo_path=...)` argument that MUST resolve to a path under the git repo root. Forbidden anchors raise `ValueError` at runtime (`_assert_auth_memo_path_repo_relative`); STRICT preflight Catalog #150 enforces the same constraint at call-site time so reviewers see violations BEFORE dispatch. Canonical authorization location: `.omx/research/operator_authorizations/phase_b_auth_<lane_id>_<YYYYMMDD>.md` (committed audit trail; the file MUST contain the explicit unquoted line-level `operator_phase_b_authorization=true` token outside any code-fence/blockquote). CLI flag: `python -m tac.lane_12_v2_nerv_as_renderer --phase-b-auth-memo <committed_repo_path>`. STRICT-FLIP per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable — Catalog #150 lands at 0 live violations in same commit batch. 34 dedicated tests in `src/tac/tests/test_check_150_phase_b_auth_memo_in_repo.py`. Memory: `feedback_phase_b_option_c_landed_20260509.md`. Cross-ref `.omx/research/codex_review_round7_round8_findings_directive_20260509.md` (the original 3-options enumeration), `feedback_codex_round78_findings_fix_with_self_protection_landed_20260509.md` (where HIGH 2 round 8 was deferred to operator), `project_top_priority_revisit_NOT_YET_operator_decisions_20260509.md` ITEM 4 (the NOT YET pin this resolves).

**Trainer-flag-manifest wire-up self-protection (2026-05-12, STRICT @ 0):**

151. `check_operator_wrapper_threads_trainer_tier_required_flags` — Refuses operator-authorize / remote-lane / Modal / Lightning wrappers under `scripts/`, `tools/`, or `experiments/` (excluding `experiments/results/**` build artifacts + `_intake_` vendored clones + `.omx/oss_export/**` mirrors) that subprocess-invoke an `experiments/train_*.py` trainer without threading the env→CLI ladder for each flag declared in the trainer's `TIER_<N>_OPERATOR_REQUIRED_FLAGS` module-level dict. Per the grand council adversarial design review 2026-05-12 (9/10 PROCEED, Ballé abstain, R1-R7 binding stipulations): the check (R1) unions ALL `TIER_N_OPERATOR_REQUIRED_FLAGS` Assigns per trainer, (R2) unions required flags across every trainer a wrapper invokes, (R3) fail-opens on trainers with no manifest, (R4) accepts `--profile X` if X is in `meta["satisfied_by_profile"]`, (R5) scopes by INVOCATION evidence (subprocess token within ±3 lines of the trainer path — `subprocess.`, `Popen(`, `$PYBIN`, `python -u`, `uv run`, `modal run`, etc.) NOT by filename, (R6) excludes vendored intake / DERIVED_OUTPUT trees, (R7) strict-from-byte-one per the Strict-flip atomicity rule (NF1 wire-fix commit d37c6b20 already drove the 4-instance live count to 0). Acceptance: literal flag OR env-var-gated block referencing meta["env"] OR `--profile X` where X is in satisfied_by_profile OR same-line `# TIER_REQUIRED_FLAG_WAIVED_OK:<flag>:<reason>` waiver. Per-flag waivers only (file-level forbidden per Contrarian stipulation). Bug class: the recurring "landed but not wired" failure mode — four 2026-05-12 instances on the T1 Ballé trainer alone (`--enable-autocast-fp16`, `--enable-mp4-codec-sim`, `--segmentation-surrogate` default flip, `--batch-size` default flip) all silently nullified by stale wrappers. Sister of Catalog #12 `preflight_arity` (the dead-flag detector): #12 catches CALLER flags not in target argparse; #151 catches TARGET-required flags not threaded by caller. Together they close the bidirectional wire-up gap. 25 dedicated tests in `src/tac/tests/test_check_151_operator_wrapper_threads_trainer_tier_required_flags.py` covering each council refinement (R1-R7 + same-line waiver + AST extractor + invocation gating). STRICT-FLIP per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable. Memory: `feedback_adversarial_review_fixup_pass_2_nf1_landed_design_memo_for_meta_protection_20260512.md` + `feedback_catalog_151_landed_20260512.md` (the landing memo). Design memo: `.omx/research/design_trainer_flag_manifest_for_wireup_and_composition_20260512.md` (4-option design; Option D hybrid approved). Cross-ref the CLAUDE.md "Subagent coherence-by-default" non-negotiable — the manifest IS the trainer-side coherence primitive that future subagents inherit automatically.

**Required-input-file pre-dispatch validation (2026-05-12, STRICT @ 0):**

152. `check_operator_wrapper_validates_required_input_files_pre_dispatch` — Sister of Catalog #151. Refuses dispatch wrappers under `scripts/`, `tools/`, or `experiments/` (excluding `experiments/results/**` build artifacts + `_intake_` vendored clones + `.omx/oss_export/**` mirrors) that contain a dispatch token (`modal run`, `vastai create instance`, `lightning run`, `launch_lane_on_vastai`, `launch_lane_lightning`, `kaggle kernels push`, `modal_train_lane`) AND invoke (directly OR indirectly via `--lane-script scripts/X.sh`) a trainer with one or more `required_input_file=True` entries in its `TIER_<N>_OPERATOR_REQUIRED_FLAGS` manifest WITHOUT validating each required-input flag's value points to an existing file BEFORE the GPU dispatch fires. Acceptance: (a) literal `[ -f "$ENV" ]` / `[ -f "${ENV}" ]` / `[ ! -f "$ENV" ]` / `test -f "$ENV"` shell test referencing the env-var OR Python `Path(os.environ["ENV"]).is_file()` / `.exists()` equivalent; (b) `tools/validate_dispatch_required_inputs.py --trainer <path>` invocation (the canonical validator landed in the same commit batch — reads `_check_151_extract_tier_manifests` so the validator + gate stay in lock-step); (c) same-line `# REQUIRED_INPUT_VALIDATION_OK:<flag-or-env-or-ALL>:<reason>` waiver where the token MUST explicitly name the flag / env-var / `ALL` sentinel (reason-only tokens do NOT silently waive per CLAUDE.md "Comment-only contracts are FORBIDDEN"). Bug-class anchor: a 2026-05-12T17:12 Modal A100 dispatch (call_id redacted; private custody artifact only) burned $0.016 in 15s before crashing with rc=1 because `--pr95-parity-profile` pointed to a non-existent file on the Modal worker. A local 100ms `[ -f ... ]` validation would have caught it before the GPU meter started. STRICT-FLIP per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable — the T1 Ballé operator-authorize wrapper invokes the canonical validator tool in the same commit batch, driving live count to 0. 41 dedicated tests in `src/tac/tests/test_check_152_operator_wrapper_validates_required_inputs.py` covering (i) manifest extraction (3 tests), (ii) dispatch-token scope (4 tests), (iii) shell existence test acceptance (4 tests), (iv) validator-tool invocation acceptance (2 tests), (v) waiver collection + per-flag granularity (4 tests), (vi) indirect lane-script detection (3 tests), (vii) end-to-end (7 tests), (viii) R6 vendored-intake exclusion, (ix) strict-flip + live-repo invariant, (x) validator-tool CLI rc=0/1/2 behavior (7 tests). Memory: `feedback_permanent_fix_required_input_validation_20260512.md`. Cross-ref Catalog #151 (the env→CLI wire-up sister gate); Catalog #146 (Phase 1 trainer runtime contract); Catalog #12 `preflight_arity` (the bidirectional flag-wire-up gap closer). Companion tool: `tools/validate_dispatch_required_inputs.py` reads the manifest, resolves each `required_input_file=True` flag value from env-var (highest precedence) → default, exits 1 with actionable error (including the `generator_command` field if declared) when any required input file is missing.

**Modal mount manifest canonical builder (2026-05-12, STRICT @ 0):**

153. `check_modal_dispatcher_uses_canonical_mount_builder` — Sister of Catalog #151 + #152. Refuses any `experiments/modal_*.py` file that calls `Image.debian_slim(...).add_local_dir(...)` or `.add_local_file(...)` directly without routing through `tac.deploy.modal.mount_manifest.build_training_image`. Acceptance: (a) the file imports + uses `build_training_image`; OR (b) every `.add_local_dir(` / `.add_local_file(` callsite carries a same-line `# MODAL_MANUAL_MOUNT_OK:<reason>` waiver (per-callsite escape hatch; file-level waiver forbidden); OR (c) the file is on the exempt list (`src/tac/deploy/modal/mount_manifest.py` + its tests). Bug class: hand-curated mount lists are an "N+1 entry" failure mode — every new trainer-default path the operator adds is one place the mount list can be stale. A 2026-05-12T17:12 Modal A100 dispatch (call_id redacted; private custody artifact only) burned $0.016 in 15s because the operator added a new `required_input_file=True` flag to the trainer manifest without updating the 11 Modal dispatcher mount lists. Adding entries to hand-curated lists IS the bug class. The canonical builder `tac.deploy.modal.mount_manifest.build_training_image` resolves the same problem by INTROSPECTING the trainer's `TIER_<N>_OPERATOR_REQUIRED_FLAGS` at image-build time — adding a new `required_input_file=True` flag automatically propagates to the mount list. Sister of Catalog #151 (operator-wrapper-threads-trainer-flags) + Catalog #152 (operator-wrapper-validates-required-input-files): together they close the trainer-manifest → wrapper → image-build wire-up loop. T1-A landing 2026-05-12: `modal_train_lane.py` refactored to use the canonical builder; the other 10 narrow file-by-file dispatchers (auth_eval, alpha_geo0_pose_regen, phase_a1/a4, component_sensitivity, loader_drift, scorer_introspection, constrained_gen_test, t1_balle_endtoend) carry per-line `# MODAL_MANUAL_MOUNT_OK:<reason>` waivers because they are intentionally narrow (mount specific upstream files, not full trees). STRICT-FLIP per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable — live count: 0 at landing. 14 dedicated tests in `src/tac/tests/test_check_153_modal_mount_builder.py` covering canonical-builder acceptance, per-line waiver, mixed canonical+manual rejection without waiver, comment-line tolerance, non-modal-file out-of-scope, strict-raise behavior, multi-file aggregation, multi-line call waiver, and live-repo regression guard. 30 dedicated tests in `src/tac/tests/test_mount_manifest.py` covering structural-minimum-always-mounted, missing-structural-minimum fail-closed, trainer-introspection of `required_input_file=True` flags, missing-default fail-closed, `TIER_1_EXTRA_MOUNT_PATHS` + `MODAL_EXTRA_MOUNT_PATHS` alias, optional_files/optional_dirs conditional mount, multi-tier flag union, non-dict manifest tolerance, file-path vs dotted-module import, sys.modules pollution avoidance, absolute-default-path resolution. Memory: `feedback_modal_mount_manifest_consolidation_landed_20260512.md` (the landing memo). Cross-ref the T1-B simplification: `tac.deploy.modal.training_cost` was simultaneously inlined into `tac.cost_band_calibration` (`PLATFORM_RATES_USD_PER_HOUR` dict + `normalize_gpu(platform, gpu)` + `estimate_cost_usd(platform, gpu, sec)` + `append_platform_training_anchor(platform, ...)`) so the platform-keyed rate table sits next to the posterior it feeds; the old Modal-specific shim now delegates to the canonical functions and remains for backward compatibility.

**T1-D state-hygiene wave (2026-05-12, STRICT @ 0):**

154. `check_experiments_results_gc_helper_is_canonical` — Refuses any tool/script under `tools/`, `scripts/`, `experiments/`, or `src/tac/` that bulk-deletes (`shutil.rmtree`, `os.remove`/`removedirs`/`unlink`, `Path(...).unlink()`/`.rmdir()`, shell `rm -rf <experiments/results/...>`, `find experiments/results ... -delete`) under `experiments/results/` WITHOUT routing through the canonical helper `tools/gc_experiments_results.py`. The canonical helper enforces (a) `--dry-run` is the default mode (no deletion without explicit `--apply`); (b) `--apply` REQUIRES `--operator-approved '<handle>:<UTC_timestamp>'`; (c) git-tracked paths are NEVER auto-deleted; (d) `build_manifest.json::custody_status in {committed-binary, published, ci-rebuildable}` paths are PINNED; (e) `recovered_*/recovery_metadata.json` is HISTORICAL_PROVENANCE (Catalog #110) and the helper surfaces these for explicit operator review (PRESERVE-METADATA-DELETE-BODIES verdict). Bug class: ad-hoc cleanup scripts that crawl `experiments/results/` and delete what "looks unused" — silently destroying HISTORICAL_PROVENANCE artifacts per Catalog #110 / #113, deleting git-tracked paths the operator wanted preserved, OR bypassing the operator-consent gate. Acceptance: (1) routing through the canonical helper; OR (2) same-line `# GC_EXPERIMENTS_RESULTS_BYPASS_OK:<reason>` waiver on the destructive-call line for the rare operator-reviewed forensic-cleanup case; (3) test files (`test_*.py`) are exempt (they exercise rmtree on tmp_path fixtures); (4) vendored public-PR intake clones (`experiments/results/public_pr*_intake_*`) and OSS export mirrors are out of scope. Self-exempt list (file-level): `tools/gc_experiments_results.py` itself (the helper performs the licensed destructive call) + `src/tac/preflight.py` (the check defines regex patterns that contain the destructive-call signature literals). STRICT-FLIP per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable — live count: 0 at landing. 24 dedicated tests in `src/tac/tests/test_gc_experiments_results.py` covering (i) per-classification verdicts (smoke-old-delete, smoke-recent-keep, recovered-with-metadata-preserve, recovered-without-metadata-fallthrough, tracked-always-keep, custody-pinned-keep, very-recent-always-keep, pytest_tmp_outputs-smoke-token), (ii) build_gc_plan partitioning + rationale-per-path, (iii) CLI refuses `--apply` without `--operator-approved`, validates handle format, `--dry-run` does not delete, `--apply` deletes only DELETE-NOW entries, `--apply` never deletes recovery_metadata, (iv) the STRICT preflight check detects shutil.rmtree / os.removedirs / shell rm -rf / find -delete; accepts same-line waiver; skips test files / canonical helper / intake clones / unrelated rmtree calls. Companion T1-E sister landing: `tools/claim_lane_dispatch.py prune` archives terminal rows older than 7 days into monthly `.omx/state/dispatch_claims_archive/dispatch_claims_<YYYY-MM>.md` files (HISTORICAL_PROVENANCE, append-only) and rewrites the live ledger keeping only active + recent terminal rows. 13 dedicated tests in `src/tac/tests/test_claim_lane_dispatch_prune.py`. Memory: `feedback_state_hygiene_gc_and_prune_landed_20260512.md`. Cross-ref Catalog #110 (`check_recovery_metadata_append_only`) + Catalog #113 (`check_artifact_lifecycle_compliance`, the META-class umbrella) — together they extinct the "ad-hoc cleanup script destroys HISTORICAL_PROVENANCE" failure mode.

**GC tracked-path defense + commit-swap permanent fix (2026-05-12 subagent F Wave 1, STRICT @ 0):**

156. `check_gc_helper_refuses_delete_on_tracked_paths` — Sister of Catalog #154. Where #154 refuses NEW destructive calls under `experiments/results/` outside the canonical helper, #156 refuses external callers of the canonical helper that strip the Part-2 `TrackedDeleteRefusedError` defense-in-depth. Scans `tools/`, `scripts/`, `experiments/`, `src/tac/` for `.py` files that call `execute_plan(...)` / `classify_results_dirs(...)` / `build_gc_plan(...)` / `_classify_dir(...)` from outside the canonical helper itself. Each callsite MUST satisfy one of: (a) be in the canonical helper or its own tests; (b) carry a same-line `# GC_TRACKED_DELETE_OK:<reason>` waiver; (c) be in a file that imports + catches `TrackedDeleteRefusedError` somewhere (file-level exception-handler proxy). Companion fix: `tools/gc_experiments_results.py::_classify_dir` had a smoke-vs-tracked precedence bug where the smoke-token branch fired BEFORE the tracked-check (Part 1 of subagent F fix), and `execute_plan` lacked runtime `git ls-files` re-verification (Part 2 — now `_git_ls_files_batch` runs against every `would_delete` path BEFORE any `shutil.rmtree`, raising `TrackedDeleteRefusedError` if any path is tracked; CLI exits rc=4). Bug class: the T1-D wave's classifier could (with a smoke-named-but-committed dir) emit a plan that misclassified a tracked dir as DELETE-NOW, and a future external consumer of `execute_plan` could strip the Part-2 defense by silencing the exception. The two gates together extinct the entire bug class. STRICT-FLIP per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable — live count: 0 at landing (no external callers exist; helper is used only via its CLI surface). 12 dedicated tests in `src/tac/tests/test_check_156_gc_helper_refuses_delete_on_tracked.py` covering live count zero, external call without exception handler, exception-handler acceptance, same-line waiver, canonical helper self-exemption, test files exemption, intake clones exemption, all 4 public API names (execute_plan / classify_results_dirs / build_gc_plan / _classify_dir), unrelated callsites pass, and strict-raise. 7 new tests in `src/tac/tests/test_gc_experiments_results.py` covering Part 1 (smoke-named tracked dir → KEEP) + Part 2 (forged stale plan refused via TrackedDeleteRefusedError + rc=4 CLI surface + _git_ls_files_batch behavior). Memory: `feedback_gc_fix_and_commit_swap_class_protect_landed_20260512.md`. Cross-ref Catalog #154 (the canonical-helper-canonical gate this layer extends).

157. `check_commit_serializer_pre_lock_hash_against_head` — Commit-swap class permanent fix. The 2026-04-29 PM `feedback_concurrent_subagent_commit_message_swap_20260429.md` documented the bug class. 2026-05-08 META-META hardening added a pre-lock vs post-lock content-hash check (`FIX-1`) inside `tools/subagent_commit_serializer.py` to detect concurrent edits during the LOCK-WAIT WINDOW. The 2026-05-12 92aba3ca incident showed an EARLIER window FIX-1 cannot catch: when TWO subagents have ALREADY edited the same file in the working tree BEFORE either takes its pre-lock snapshot, BOTH subagents observe the merged content; the winning subagent's `git add <file>` packages BOTH edits under its commit body; the losing subagent's `git add <file>` returns "no changes" because HEAD already contains everything. Companion fix lands a new `--expected-content-sha256 <relpath>=<sha>` flag (caller declares the working-tree hash AT WORK-START); the serializer hashes the current content and refuses with rc=4 if it differs — catching the pre-pre-lock race. The static gate refuses ANY direct `git commit` invocation outside the canonical serializer file itself. Scans `tools/`, `scripts/`, `experiments/`, `src/tac/` for `.py` and `.sh` files invoking `subprocess.run(["git", "commit", ...])` / `subprocess.run(...shell=True...git commit...)` / `os.system("git commit ...")` / `^\s*git commit` (shell). Acceptance: (a) canonical serializer + preflight.py self-exempt; (b) test files exempt; (c) intake clones / OSS export mirrors exempt; (d) same-line `# COMMIT_SERIALIZER_BYPASS_OK:<reason>` waiver; (e) file-level `# COMMIT_SERIALIZER_BYPASS_OK_FILE:<reason>` for the rare operator-side housekeeping helper (`tools/auto_commit.sh` carries this). Bug class: the 92aba3ca commit-swap incident proved that even with the file lock, two subagents that edited the same file pre-snapshot can corrupt commit attribution. The structural fix forces every subagent commit through the canonical wrapper, and the new `--expected-content-sha256` flag catches the pre-pre-lock race. STRICT-FLIP per the same self-protection mandate — live count: 0 at landing. 20 dedicated tests in `src/tac/tests/test_check_157_commit_serializer_pre_lock_hash.py` covering live count zero, Python subprocess detection (run/Popen/check_call/shell=True), os.system detection, shell `git commit` detection, same-line + file-level waivers, canonical/preflight/test/intake exemptions, unrelated `git` commands pass, strict-raise, `_parse_expected_content_sha256` (well-formed, no-equals, short-sha, non-hex, empty), `_expected_content_sha256_check` (mismatch, match, empty). Plus 2 multiprocessing stress tests in `src/tac/tests/test_commit_serializer_4proc_stress.py` firing 4 concurrent serializer invocations on (a) disjoint files (all 4 commit cleanly; HEAD advances 4 times; each diff matches intended file set) and (b) overlapping file with shared expected-sha (proves the new flag refuses the race; no commit lands at seed content). Memory: `feedback_gc_fix_and_commit_swap_class_protect_landed_20260512.md`. Cross-ref Catalog #117 (`check_subagent_commit_serializer_uses_lock`, the sister gate that enforces last-50-commit usage) + `feedback_concurrent_subagent_commit_message_swap_20260429.md` (the canonical bug-class incident report).

158. `check_deterministic_compiler_canonical_use` — Refuses new packet-compilation surfaces that bypass `tac.packet_compiler.deterministic_compiler` or the canonical `tools/build_deterministic_packet.py` CLI. This prevents one-off PacketIR/byte-builder drift from silently producing non-reproducible archive transforms. STRICT-FLIPPED 2026-05-12; orchestrator callsite is `strict=True`; live count: 0. Dedicated tests: `src/tac/tests/test_check_158_deterministic_compiler_canonical_use.py`. Memory: `feedback_deterministic_packet_compiler_landed_20260512.md`.

159. `check_claude_md_catalog_text_matches_preflight_strict_value` — Refuses CLAUDE.md catalog entries whose strictness text contradicts the `strict=` value wired in `src/tac/preflight.py`. This keeps the operator-facing strict-mode ledger aligned with executable preflight behavior and is the sister of Catalog #118 duplicate-number protection. STRICT-FLIPPED 2026-05-12; orchestrator callsite is `strict=True`; live count: 0. Dedicated tests: `src/tac/tests/test_check_159_catalog_text_matches_preflight_strict.py`. Memory: `feedback_ffff_bug_sweep_landed_20260512.md`.

161. `check_quantize_degenerate_range_clamped_correctly` — Refuses substrate archive `_quantize_intN` implementations whose degenerate-range branch fills quantized tensors with zero instead of `-(MAX_LEVELS // 2)`, which breaks exact dequantization back to `lo`. The gate scans substrate archive quantizers and accepts only the mathematically correct sentinel or a same-line `QUANTIZE_DEGENERATE_OK:<reason>` waiver. STRICT-FLIPPED 2026-05-12; orchestrator callsite is `strict=True`; live count: 0. Dedicated tests: `src/tac/tests/test_check_160_quantize_degenerate_range.py` (filename retained for git-blame continuity after FIX-A renumber from #160 to #161 on 2026-05-12 per ZZZZZ collision audit). Memory: `feedback_ffff_bug_sweep_landed_20260512.md` + `feedback_fix_a_catalog_158_124_landed_20260512.md`.

**FIX-H WWW4 self-protection (2026-05-12):**

163. `check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap` — Refuses `scripts/remote_lane_*.sh` files that ``source`` the canonical `scripts/remote_archive_only_eval.sh` without prepending the `REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1` sentinel. Without the sentinel the sourced main flow runs to completion inside the calling shell, exits "FATAL: archive missing" before the calling lane's stages can start. Bug class: WWW4 dispatch (Modal A100 fc-01KREXK209TRX7ED5ZRVXHY1VT, 2026-05-12) crashed at this exact pattern; WWW4's same-day fix at `scripts/remote_lane_substrate_sane_hnerv.sh:81` (commit `02d7fc3f`) extincted the only known instance. Acceptance: same-line sentinel (`VAR=1 source ...`) OR export within 5 preceding lines OR same-line `# REMOTE_LANE_FULL_PIPELINE_OK:<reason>` waiver. Process-substitution forms (`source <(grep ...)` / `source <(awk ...)`) and child-shell invocations (`bash script.sh`) are explicitly skipped — they don't trigger main-flow side-effects in the caller. STRICT-FLIPPED 2026-05-12; orchestrator callsite is `strict=True`; live count: 0 across 124 scanned files. Dedicated tests: `src/tac/tests/test_check_163_remote_lane_sentinel.py` (12 tests). Memory: `feedback_fix_h_trainer_shape_fix_landed_20260512.md`.

164. `check_substrate_score_aware_loss_calls_preprocess_input_before_scorer` — Refuses `src/tac/substrates/*/score_aware*.py` whose function bodies call `self.<scorer>(...)` BARE forward without also calling `self.<scorer>.preprocess_input(...)` in the same body. Upstream `SegNet` is `smp.Unet` expecting 4D `(B, C, H, W)`; `preprocess_input` slices the last frame from 5D and interpolates to (384, 512). Upstream `PoseNet` expects 4D 12-channel; `preprocess_input` runs differentiable rgb_to_yuv6 and rearranges (B, T, C, H, W) → (B, T*6, H/2, W/2). Bug class: WWW4 dispatch crashed at `smp.Unet` stem because `sane_hnerv/score_aware_loss.py:129` passed 5D directly to `self.seg_scorer(...)`; sibling latent bug fed 4D 6-channel RGB to `self.pose_scorer(...)` instead of 4D 12-channel. FIX-H Part 1 (commit `6048d690`) remediated sane_hnerv. AST-aware: walks every `FunctionDef`, collects (a) bare `self.<scorer>(...)` forward calls and (b) `self.<scorer>.preprocess_input(...)` calls; refuses any forward whose scorer attr is not in the preprocess set. Same-line waiver: `# SCORER_PREPROCESS_HANDLED_OK:<reason>`. Known scorer attribute aliases: `seg_scorer` / `pose_scorer` / `scorer` / `segnet` / `posenet` / `distortion_net`. STRICT-FLIPPED 2026-05-12 after the substrate-wide canonicalization pass replaced all 15 lane-local scorer forwards with `tac.substrates.score_aware_common.score_pair_components`, and the helper now delegates to `tac.losses.scorer_loss_terms_btchw` so preprocess_input, target no-grad, SegNet surrogate semantics, and `sqrt(10)` pose weighting stay aligned with the contest formula. Dedicated tests: `src/tac/tests/test_check_164_scorer_preprocess_before_forward.py` (21 tests). Memory: `feedback_fix_h_trainer_shape_fix_landed_20260512.md`.

**PHASE-B1-PIVOT Modal HEAD-parity ledger + smoke-before-full pattern (2026-05-12, STRICT @ 0):**

166. `check_modal_dispatch_verifies_worker_source_matches_head` — Refuses any state of `experiments/modal_train_lane.py` that drops the dispatch-time HEAD parity ledger or the worker-side source-parity ledger. Required surfaces: `def _git_dirty_tree_summary` helper; metadata schema marker `modal_train_lane_dispatch_metadata_v2_catalog166`; metadata keys `mounted_code_git_head` / `working_tree_dirty` / `working_tree_dirty_summary` / `sentinel_files_local_sha256`; worker ledger keys `worker_sentinel_sha256` / `sentinel_mismatches`; the worker-side `modal_worker_head_ledger.json` write inside `_run_lane_inner`; the `CATALOG_166` worker-side warn token; the `require_clean_head` kwarg on `main()`. There is no waiver — the surfaces ARE the contract. Bug class: PHASE-B1-PIVOT 2026-05-12. Two consecutive Modal A100 dispatches of `experiments/train_substrate_sane_hnerv.py` crashed rc=1 (`fc-01KREXK209TRX7ED5ZRVXHY1VT` 14.77s + `fc-01KREXXSKGTDCF61QXQNBF6SX3` 72.03s). The 72-sec traceback hit `src/tac/substrates/sane_hnerv/score_aware_loss.py:129 unsqueeze(1)` — a bug that did NOT exist at HEAD (commit `6048d690` removed it). The canary subagent hypothesized "Modal worker mounted stale source" but the actual root cause was **chronological**: the dispatch was fired at 20:26:47Z, the fix landed at 20:44:00Z (17 minutes LATER). The local source on disk at dispatch time WAS still broken; the worker faithfully ran what it was given. Post-mortem could not distinguish "Modal worker mounted stale code" from "operator dispatched before fix landed" because `modal_metadata.json` did not record the dispatch-time HEAD SHA OR the working-tree-dirty state. The fix records the dispatch-time HEAD SHA + working-tree-dirty summary + sentinel-file sha256 ledger into `modal_metadata.json`, writes and harvests `modal_worker_head_ledger.json` on the worker, and adds a `--require-clean-head` opt-in fail-closed gate before lane-claim creation so refused dispatches do not leave phantom active claims. Sister of Catalog #153 (canonical mount builder) + Catalog #165 (mtime stability). The runtime sister of this STRICT gate is `tools/diagnose_modal_worker_source_staleness.py` (a tiny Modal probe operators can fire any time to verify mount discipline via the H1-H5 verdict taxonomy: H1=image cache, H2=eval-timing, H3=deploy stale, H4=manifest gap, H5=PYTHONPATH ordering, HOK=parity). STRICT-FLIPPED 2026-05-12 per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable; live count at landing: 0. Dedicated tests: `src/tac/tests/test_check_166_modal_dispatch_head_parity.py` (15 tests) + `src/tac/tests/test_diagnose_modal_worker_source_staleness.py` (15 tests). Memory: `feedback_phase_b1_pivot_modal_source_staleness_fix_LANDED_20260512.md`.

**META-CATALOG-152-FIX AnnAssign META-CLASS gate (2026-05-12, STRICT @ 0):**

**WAVE-7-MED-FIX REVIEW-OMNI Medium-severity gates (2026-05-12):** Five new prevent-recurrence gates closing 5 of 6 REVIEW-OMNI Medium findings (`.omx/research/review_omni_fields_medal_nvidia_grade_20260512.md`). M6 = C5 sensitivity-map artifact reader DEFERRED-pending-council per CLAUDE.md "Design decisions — non-negotiable" (~50 LOC ranking change is a council-grade tradeoff requiring inner-quintet sign-off).

169. `check_compressai_primitives_registered_in_canonical_inventory` — REVIEW-OMNI Medium A2-1 (Ballé) self-protection. Ensures the 3 CompressAI codec primitives (`compressai_factorized_prior`, `compressai_balle_hyperprior`, `compressai_cheng2020`) stay registered in `tac.composition.registry.canonical_primitive_inventory()`. FIX-J landed the rows on 2026-05-12; this gate prevents regression. Direct importer-side check (no AST walk needed since the source-of-truth is the function's return value); refuses any inventory state where one or more of the 3 required IDs is absent. Strict-from-byte-one per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" — live count at landing: 0 (3 rows already registered). 12 dedicated tests in `src/tac/tests/test_check_169_compressai_primitives_registered.py`. Memory: `feedback_wave7_review_omni_6_medium_fixes_LANDED_20260512.md`.

170. `check_substrate_recipes_declare_min_vram_gb_floor` — REVIEW-OMNI Medium NV5 (Carmack). Refuses substrate operator-authorize recipes (`.omx/operator_authorize_recipes/substrate_*_modal_*_dispatch.yaml`) that don't declare a top-level `min_vram_gb: <int>` field (>= 1 GB). Bug class: Council C OOM fix not validated across substrate trainers; T4 (16GB) and A10G (22GB shared) may OOM at 384x512 + batch_size=32. Without per-recipe declared minimum, the dispatch path cannot refuse instances below the declared floor. Same-line waiver: `# MIN_VRAM_GB_OK:<reason>` for substrates with no minimum (e.g. proxy / smoke variants); placeholder `<reason>` literal rejected. Initial landing: warn-only (22 legacy recipes; backfill-or-waive sweep is a separate wave per CLAUDE.md "Strict-flip atomicity rule"). STRICT-FLIPPED 2026-05-13 per FIX-WAVE-1 (live count = 0). 15 dedicated tests in `src/tac/tests/test_check_170_substrate_recipe_min_vram_gb.py`. Memory: same.

171. `check_substrate_recipes_declare_video_input_strategy` — REVIEW-OMNI Medium NV9 (Carmack). Refuses substrate operator-authorize recipes that don't declare `video_input_strategy` with a recognized value (`per_dispatch_local_copy` / `readonly_mmap` / `shared_volume_no_contention_expected`). Bug class: at 4-6 simultaneous Modal/Lightning dispatches sharing `upstream/videos/0.mkv` from a shared volume, FS-read contention degrades each dispatch's pyav decode throughput. Same-line waiver: `# VIDEO_INPUT_STRATEGY_OK:<reason>`; placeholder rejected. Initial landing: warn-only (22 legacy recipes; backfill sweep separate). STRICT-FLIPPED 2026-05-13 per FIX-WAVE-1 (live count = 0). 15 dedicated tests in `src/tac/tests/test_check_171_substrate_recipe_video_input_strategy.py`. Memory: same.

172. `check_substrate_trainers_declare_autocast_fp16_support` — REVIEW-OMNI Medium NV2 (Hotz). Refuses `experiments/train_substrate_*.py` trainers that neither declare `--enable-autocast-fp16` argparse flag (canonical T1 Balle pattern @ commit `b0ef91a3`) NOR carry a file-level `# AUTOCAST_FP16_WAIVED:<reason>` waiver. Bug class: 4-6× engineering-speed gap vs the T1 Balle pattern; without the autocast wrapper substrate trainers leave significant Modal/Vast.ai $/wall-clock on the table. Grep-based detection (intent: the flag name is present somewhere in the file). Initial landing: warn-only (14 legacy substrate trainers; backport-or-waive sweep is a separate engineering wave per CLAUDE.md "Strict-flip atomicity rule"). STRICT-FLIPPED 2026-05-13 per FIX-WAVE-1 (live count = 0). 15 dedicated tests in `src/tac/tests/test_check_172_substrate_trainer_autocast_fp16.py`. Memory: same.

173. `check_substrate_dispatch_honors_canary_first_ordering` — REVIEW-OMNI Medium C4 (Quantizr/Fridrich). Refuses substrate operator-authorize recipes that don't declare `canary_status` (one of `canary` / `post_canary_dependent` / `independent_substrate`). A `post_canary_dependent` recipe MUST also declare `canary_dependency: <substrate_id>` so the dispatch path can verify the canary has at least one successful contest-CUDA anchor before firing. Bug class: sane_hnerv had 5 failed first-anchor attempts on 2026-05-12; treating sane_hnerv as just-one-of-N in parallel fan-out is a Race-mode rigor inversion at the wrong moment (no contest race active). Same-line waiver: `# CANARY_FIRST_OK:<reason>`; placeholder rejected. Initial landing: warn-only (22 legacy recipes; backfill sweep separate). STRICT-FLIPPED 2026-05-13 per FIX-WAVE-1 (live count = 0). 18 dedicated tests in `src/tac/tests/test_check_173_substrate_recipe_canary_status.py`. Memory: same.

168. `check_ast_walker_handles_both_assign_and_annassign` — META-CLASS gate. Refuses any AST extractor function that filters by `ast.Assign` ONLY without also handling `ast.AnnAssign`. The bug class silently bypasses static analysis when the target code uses annotated assignment syntax (`name: type = value`). Bug-class anchor: 2026-05-12 META-CATALOG-152-FIX. The `_check_151_extract_tier_manifests` AST walker filtered `ast.Assign` only; substrate trainers (sane_hnerv, balle, SIREN, Cool-Chic, VQ-VAE, hybrid_renderer_residual, self_compress_nn, TCNeRV, BlockNeRV, FFNeRV, DSNeRV, HiNeRV) declare the manifest as `TIER_<N>_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {...}` (`ast.AnnAssign`), so 12/12 substrate trainers silently returned EMPTY manifests — making Catalog #151 + #152 STRICT modes structurally false-OK across the entire substrate canvas. WAVE-1-A (VQ-VAE subagent) confirmed the bug empirically; PHASE-B2-BUILD subagent caught it as the architecture gap. Companion fix: `_check_151_extract_tier_manifests` rewritten to walk both `ast.Assign` AND `ast.AnnAssign` (single-target normalization, value-None guard); `experiments/kaggle_t1_balle_sweep.py::_resolve_tier1_required_flags` patched with the same normalization (sister surface that also raised KeyError on substrate trainers); 9 additional non-Catalog-151 surfaces (preflight.py:5091/5721/6398/11745/28413/29379/31588 + preflight_charm_class_check.py + pr103_lc_ac_runtime_adapter.py + foveation_readiness.py + 2 tools/ + 5 experiments/ PR-replay validators) batch-fixed to handle both forms. AST-aware: walks every `isinstance(_, ast.Assign)` Call node and refuses unless (a) the type-tuple ALSO references `ast.AnnAssign`, OR (b) the enclosing FunctionDef/AsyncFunctionDef/ClassDef contains a sister `isinstance(_, ast.AnnAssign)` call (the canonical if/elif chain pattern), OR (c) the same-line OR preceding-line carries `# ASSIGN_ONLY_OK:<reason>` waiver (placeholder `<reason>` literal rejected to prevent self-waiver), OR (d) file-level `# CHECK_168_FILE_LEVEL_WAIVED:<reason>` waiver (same placeholder rejection). Excluded path markers: `experiments/results/` (DERIVED_OUTPUT per Catalog #113), `_intake_` (vendored PR clones per Catalog #109), `.omx/oss_export/` (mirror), `vendored`. Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable + "META-meta finding from a8bc7e79 sweep ... bug classes have 6-7x spread across the repo" — the fix at one surface left the same class active at 11 others; this gate extincts the bug class structurally. STRICT-FLIP per the "Strict-flip atomicity rule" — live count at landing: 0 (12 candidate sites driven to 0 in same commit batch; 2827 files scanned). 26 dedicated tests in `src/tac/tests/test_check_168_ast_walker_handles_assign_and_annassign.py` covering positive (bare Assign-only / AugAssign-sister / attribute-target / single-element-tuple), negative (dual-handling-tuple in either order / if-elif-chain), waiver (same-line / preceding-line / file-level + placeholder rejection), exempt-path (results/intake/oss_export), non-isinstance-comparisons-ignored, isinstance-with-other-ast-kinds-ignored, syntax-error-skipped, tools+experiments scanned, scope-aware-per-function-not-per-module, strict-mode-raises-PreflightError, and live-repo-regression-zero-violations. Sister of Catalog #151 (operator-wrapper-threads-trainer-flags, the empirical-bug consumer) + Catalog #152 (operator-wrapper-validates-required-input-files, the same-extractor consumer). Memory: `feedback_meta_catalog_152_annassign_fix_LANDED_20260512.md`. Cross-ref WAVE-1-A VQ-VAE subagent + PHASE-B2-BUILD subagent (the architecture gap discoverers).

**FIX-WAVE-1 META-meta commit-machinery + cost-band defense-in-depth (2026-05-13, STRICT @ 0):**

174. `check_subagent_commit_serializer_always_uses_expected_content_sha256` — FIX-WAVE-1 R1 Medium #1 (2026-05-13). The 2026-05-12 `8c9a5e7f` commit-swap incident proved Catalog #157's protection is asymmetric: a subagent that DECLARES `--expected-content-sha256` catches sibling edits to ITS declared files, but a sibling subagent that does NOT declare can freely absorb the work because the sibling's commit lock window is never gated by the pre-pre-lock content check. Refuses subprocess-style invocations of `tools/subagent_commit_serializer.py` that omit `--expected-content-sha256`. Same-line waiver: `# COMMIT_SERIALIZER_NO_SHA_OK:<reason>` for legitimate operator housekeeping (e.g. `tools/claim_catalog_number.py` which commits a single trivial state file). Self-exempt: serializer + preflight + claim_catalog_number. Test files excluded. STRICT-from-byte-one per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable AND "Strict-flip atomicity rule" — live count at landing: 0. 13 dedicated tests in `src/tac/tests/test_check_174_serializer_sha_required.py`. Memory: `feedback_fix_wave_1_r1_findings_LANDED_20260513.md`. Sister of Catalog #117 (`check_subagent_commit_serializer_uses_lock`, last-50-commit usage gate) + Catalog #157 (pre-pre-lock hash check; this gate makes the `--expected-content-sha256` flag MANDATORY so Catalog #157's protection is symmetric across all subagents).

175. `check_cost_band_anchor_writers_declare_outcome` — FIX-WAVE-1 R1 Medium #6 (2026-05-13). Defense-in-depth over the `cost_band_calibration.py:333` ambient-default fallback. Refuses direct writes to `.omx/state/cost_band_posterior.jsonl` that don't route through `tac.cost_band_calibration.append_anchor(outcome=...)` helper OR explicitly declare `outcome=` near the write. Scans `tools/`, `scripts/`, `src/tac/`, `experiments/` for `.py` files referencing the posterior path tokens; conservative path-assignment scoping (5-line backward lookback for `Path(` / `open(`) to avoid docstring + log false positives. Same-line waiver: `# COST_BAND_ANCHOR_OUTCOME_OK:<reason>`. Self-exempt: canonical helper module + preflight. Test files excluded. Bug class: the read-side default at `cost_band_calibration.py:333` silently coerces missing `outcome` to `successful_dispatch`; without this gate, a future writer that bypasses the canonical helper (or a CI/dashboard reader of the JSONL) inherits the silent coercion. STRICT-from-byte-one per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable — live count at landing: 0. 12 dedicated tests in `src/tac/tests/test_check_175_cost_band_anchor_outcome.py`. Memory: `feedback_fix_wave_1_r1_findings_LANDED_20260513.md`. Sister of Catalog #131 (`check_no_bare_writes_to_shared_state`, the META gate for `.omx/state/` write discipline) + Catalog #138 (`check_state_writers_strict_load_for_mutating_path`, the strict-load defense).

**FIX-WAVE-2 META-meta ledger consistency + cost-band read-side hardening (2026-05-13, STRICT @ 0):**

176. `check_strict_preflight_callsites_have_claude_md_catalog_row` — FIX-WAVE-2 R2-1 (2026-05-13). META-meta gate: refuses any state where `preflight_all()` wires a `strict=True` callsite for a `check_<name>` function that does NOT have a matching `^N. \`check_<name>\`` numbered row in the CLAUDE.md "Meta-bug class catalog" table. R1 missed this for #174 and #175; R2 caught it. Sister of Catalog #118 (`check_claude_md_catalog_no_duplicate_numbers`) and Catalog #159 (`check_claude_md_catalog_text_matches_preflight_strict_value`) — together they keep the CLAUDE.md ledger the canonical operator-facing strictness manifest. Scans `src/tac/preflight.py` for every `check_<name>(strict=True, ...)` call in `preflight_all()` body and verifies `^[0-9]+\. \`check_<name>\`` regex hits in CLAUDE.md. Same-line waiver on the orchestrator-callsite line: `# CLAUDE_MD_ENTRY_OK:<reason>` for the rare deliberate not-cataloged case. STRICT-from-byte-one per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable — live count at landing: 0 (both #174 and #175 entries added in same commit batch). Dedicated tests in `src/tac/tests/test_check_176_strict_callsite_has_claude_md_row.py`. Memory: `feedback_fix_wave_2_r2_findings_LANDED_20260513.md`.

177. `check_cost_band_posterior_rows_have_outcome_field` — FIX-WAVE-2 R2-3 (2026-05-13). Read-side defense-in-depth sister of Catalog #175 (which closed only the WRITE surface). Companion fix landed in `src/tac/cost_band_calibration.py::load_anchors`: missing `outcome` field is now tagged `outcome="legacy_pre_nv7"` (a new explicit token in `VALID_OUTCOMES`) rather than silently coerced to `successful_dispatch`. Sister gate scans `.omx/state/cost_band_posterior.jsonl` (if present) at preflight time and refuses any row missing `outcome`. Bug class: even with #175 closing the write side, the read-side default at `cost_band_calibration.py:333` corrupts the side-information channel (per Ballé's review) for any pre-NV7 row, future bypass writer, or external JSONL consumer. STRICT-from-byte-one — live count at landing: 0 (live posterior file has all 6 anchors carrying `outcome`). Dedicated tests in `src/tac/tests/test_check_177_cost_band_posterior_outcome_field.py`. Memory: `feedback_fix_wave_2_r2_findings_LANDED_20260513.md`. Sister of Catalog #175 (write-side gate).

**Codex round 8 MEDIUM in-place harden of #131 (2026-05-09, STRICT @ 0):**

131-v2. `check_no_bare_writes_to_shared_state` (in-place harden, no new catalog #) — Two in-place enhancements per codex round 8 MEDIUM (2026-05-09): (a) **AST-based path binding** (`_bare_write_collect_shared_vars_ast`) adds lowercase variables (e.g. `state_path = Path('.omx/state/foo.json')`), attribute paths (e.g. `self.foo_path = ...`), Path-joined RHS (`base / '.omx/state/foo.json'`), `AnnAssign` annotated bindings, and f-string bindings — every shape that the original all-caps regex collector silently missed; (b) **transactional-pattern requirement** — the previous lookback waiver accepted any lock token in the 20-line preceding window; the harden now requires EITHER a canonical helper invocation in the lookback (e.g. `update_active_jobs_locked(`, `register_instance(`, `posterior_update_locked(`) OR an explicit transactional pattern (write to `<path>.tmp` + `os.replace`) visible in the wider 30-line window. Lock-token alone is insufficient because partial writes are visible to readers even under the same lock and a process crash mid-write leaves a corrupt file. Canonical-helper detection skips `import ` / `from ` lines so an import statement bringing a helper name into the module does NOT falsely waive an unrelated bare write below. Same-line `# BARE_WRITE_OK:<reason>` waiver respected. Sister test files updated to reflect the new (correct) behavior: `test_131_locked_path_in_file_accepts` and `test_131_window_lock_token_accepts` (both in `src/tac/tests/test_preflight_proactive_custody_concurrency_audit.py`) now use the canonical transactional pattern; `test_check131_local_lock_context_accepted` (in `src/tac/tests/test_preflight_custody_validator_and_locked_writes.py`) updated similarly. New negative tests pin the hardened behavior: `test_131_lock_alone_no_atomic_replace_now_violation` + `test_check131_lock_alone_no_atomic_replace_is_violation_round8`. STRICT @ 0 (already strict in `preflight_all()`; no wire-in change). Live count after harden: 0. 15 dedicated tests in `src/tac/tests/test_codex_round78_check_131_harden_lowercase_and_atomic_replace.py`. Memory: `feedback_codex_round78_findings_fix_with_self_protection_landed_20260509.md`. The harden is a META-meta-meta layer over #131 → #128 (proactive sweep → continual-learning helper). Cross-ref Catalog #128 (`check_continual_learning_writes_use_lock`) and #131 (the original META gate).

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
