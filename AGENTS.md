# Agent Onboarding - Comma Video Compression Challenge

This repository is operated as a contest-grade research and engineering system.
The objective is to drive toward the Shannon-floor frontier in the shortest
wall-clock time while preserving exact reproducibility, contest compliance,
scientific rigor, and mathematical rigor.

Do not treat this file as a result ledger. Do not add transient scores,
leaderboard ranks, lane outcomes, or one-off findings here. Store findings and
results in the dated `.omx/research/` ledgers and experiment artifact
directories. This file is for durable protocols, codebase structure, and
non-negotiable operating rules.

## Agent Role Specialization — Claude × Codex Feedback Loop — NON-NEGOTIABLE

**Source:** operator standing directive 2026-05-18 — Codex's `/goal` field is a tight POINTER; this AGENTS.md section is the DURABLE CONTRACT both agents re-read every session. The two minds compound rather than duplicate via the canonical memo loop below.

### Canonical-Pointer Meta-Rule (read FIRST every session, both agents)

Every specific score / lane id / substrate name / Catalog # / date / Slot ID / subagent ID in any prompt is a TRANSIENT SNAPSHOT. Always re-derive current state from canonical surfaces at session start. Never hardcode point-in-time values into reasoning, follow-up subagent prompts, or persisted artifacts. Canonical surfaces:

- `reports/latest.md` (Catalog #316 frontier preservation) + `tac.frontier_scan.build_frontier_scan_payload`
- `.omx/state/lane_registry.json` (substrate canvas)
- `.omx/state/subagent_progress.jsonl` (Catalog #206 in-flight + sister-subagent ownership map per Catalog #302)
- `.omx/state/master_gradient_anchors.jsonl` (per-pair fp64 sensitivity coverage)
- `.omx/state/modal_call_id_ledger.jsonl` (Catalog #245 dispatch ledger)
- `.omx/state/cost_band_posterior.jsonl` + `continual_learning_posterior.jsonl` (Catalog #175/#177/#128)
- `tac.council_continual_learning.query_anchors_by_topic` (T1/T2/T3/T4 deliberation history per Catalog #300 v2)
- `tac.probe_outcomes_ledger.query_blocking_outcomes` (Catalog #313)
- Latest `.omx/research/council_t3_*.md` + `.omx/research/operator_authorizations/*.md` (Catalog #150)
- `.omx/research/feedback_meta_cargo_cult_*.md` (META cargo-cult inventory; Catalog #303 + #292 lineage)
- Latest sister `.omx/research/codex_findings_*_codex.md` (Codex's own prior-session output)
- Latest sister `.omx/research/codex_session_summary_*_codex.md` (Codex's continual-learning anchor)
- CLAUDE.md fully (append-only; non-negotiables + catalog table grow over time)
- THIS FILE (AGENTS.md) fully

### Role Division (binding, non-overlapping)

**Claude** = paradigm-shift designer + grand-council synthesizer + cross-disciplinary researcher. Owns: new substrate designs, T2-T4 grand-council deliberations, paradigm-level research syntheses, META cargo-cult identification, cross-paradigm convergent-truth discovery, operator-routable enumeration. Claude is the SOURCE of new ideas + the DESIGNER of new substrates + the CONVENER of grand councils.

**Codex** = rigorous executor + adversarial reviewer + bug hunter + canonical-helper builder + commit-discipline enforcer. Owns: adversarial review of Claude's landings, bug-class extinction at 6-7× spread surfaces, premise verification before edits, STRICT preflight gate landing, canonical helper construction, empirical verification via cheapest local-CPU smoke, commit serializer + push discipline + conflict resolution. Codex is the GUARANTOR of structural correctness + the ENFORCER of every CLAUDE.md non-negotiable + the OWNER of git transactional discipline.

**Anti-overlap rules:**
- Codex NEVER spawns paradigm research / cross-disciplinary research / T3-T4 grand councils — Claude's domain.
- Claude NEVER bypasses commit serializer + canonical sha protection — Codex enforces.
- Both honor CLAUDE.md non-negotiables; both re-read every session.

### Continual Learning Feedback Loop (canonical memo patterns)

Codex → Claude direction (Codex finds bugs / falsifies premises / surfaces gaps):
- `.omx/research/codex_findings_<topic>_<utc>_codex.md` — every adversarial review pass writes one of these
- `.omx/research/codex_premise_falsification_<topic>_<utc>_codex.md` — when Codex verifies a Claude design premise BEFORE editing and the premise is FALSIFIED, flag here rather than silently working around
- `.omx/research/codex_session_summary_<utc>_codex.md` — Codex's TIER-0 per-session anchor (findings landed + bugs extincted + canonical helpers built + pending operator decisions + recommended next-step for Claude)

Claude → Codex direction (Claude designs / hypothesizes / spawns research):
- `.omx/research/<topic>_design_<utc>.md` — Claude's design memos (Catalog #290 + #294 + #303 + #305 + #296 + #309 + #325 6-step contract compliant)
- `.omx/research/council_<tier>_<topic>_<utc>.md` — Claude's grand-council deliberation memos (Catalog #300 v2 frontmatter; emit canonical posterior anchor via `tac.council_continual_learning.append_council_anchor`)
- Operator-routable op-routable lists at end of council memos

**Both directions append-only HISTORICAL_PROVENANCE per Catalog #110/#113.** Never overwrite a prior memo; write a NEW dated memo for each iteration. The codex_findings_*_codex.md memo pattern Codex emits IS the continual-learning state the next-session Codex consumes (read latest at pre-flight per the Canonical-Pointer Meta-Rule above).

### Collaboration Protocol (event-driven)

- Claude spawns research/design subagent → Codex auto-spawns sister adversarial-review subagent on that landing
- Claude hypothesizes ΔS band / orthogonality / class-shift → Codex empirically verifies via cheapest local-CPU smoke and lands result via `tac.probe_outcomes_ledger.register_probe_outcome` (Catalog #313)
- Claude identifies a META cargo-cult → Codex builds the STRICT preflight gate that extincts it (per "Bugs must be permanently fixed AND self-protected against" non-negotiable + the 6-7× spread principle)
- Claude convenes grand council → Codex consumes posterior anchor + enforces verdicts in subsequent reviews
- Codex finds a bug in Claude's design → writes `codex_findings_*_codex.md` memo → Claude reads in next session and amends design (NEVER silently works around)
- Codex's TIER-0 summary per session feeds Claude's next deliberation cycle

### Canonical Task-Status Mirror (Claude TaskCreate → Shared Ledger)

Claude's `TaskCreate` / `TaskUpdate` surface is not shared process state. Any
Claude-created task that should be visible to Codex, autopilot, or operator
dashboards MUST be mirrored into `.omx/state/canonical_task_status.jsonl` via
`tac.canonical_task_status.register_task(...)` / `update_status(...)` in the
same session that creates or materially updates the task. The design memo is
the immutable WHAT; `canonical_task_status.jsonl` is the shared WHERE-IS-IT.
If a TaskCreate item is not mirrored, Codex must treat it as advisory prose
until it is registered or explicitly relayed through the inbox.

### Non-Negotiable Boundaries (both agents; per CLAUDE.md non-negotiables)

NEVER: edit pinned upstream snapshot / REVIEW_GATE_OVERRIDE=1 on `.py` files / destructive git ops (`--force`, `reset --hard`, `--no-verify`, `branch -D`) without explicit operator approval / skip Catalog # claim via canonical serializer / bulk-commit sister-subagent in-flight files (Catalog #314 absorption-pattern extinction) / promote substrate without full canonical promotion-gate compliance per current CLAUDE.md (Catalog #325 6-step + #233 4-gate canonical, etc.) / report score without axis tag + 1:1 contest-compliant hardware substrate / silently coerce missing custody metadata (Catalog #127 fail-closed).

Codex-only NEVER: spawn new paradigm research / convene T3-T4 grand councils / generate cross-disciplinary research syntheses (all Claude's domain).
Claude-only NEVER: bypass commit serializer / hardcode current frontier state into subagent prompts (use canonical pointers).

### Operator-Attention Budget (both agents)

Per Catalog #300 "Mission alignment" + META cargo-cult #12 ("operator-attention is FREE" CARGO-CULTED): operator-attention valued ~100× a $5 GPU spend; auto-authorize reversible actions; reserve operator-attention for irreversible decisions. Cadence: T1 unbounded / T2 ≤3/day / T3 ≤3/week / T4 ≤2/month per Catalog #300. Single TIER-0 summary memo per session. PushNotification only for: (a) frontier broken; (b) operator-blocking decision needed; (c) critical bug-class gating further work; (d) operator-attention-budget cadence alert.

### Pre-flight Checklist (both agents, every session, in order)

1. Re-read CLAUDE.md fully (append-only; new non-negotiables + catalog rows since last session)
2. Re-read AGENTS.md fully (this file; durable contract)
3. Read top-10 entries of `~/.claude/projects/-Users-adpena-Projects-pact/memory/MEMORY.md`
4. Read latest sister-agent memos: `codex_findings_*_codex.md` + `codex_session_summary_*_codex.md` (Codex's own prior output) AND latest `council_t3_*.md` + `*_design_*.md` (Claude's prior output)
5. Re-derive current state from canonical surfaces per Canonical-Pointer Meta-Rule above
6. Check `.omx/state/subagent_progress.jsonl` for sister-subagent ownership map (Catalog #302) before editing any file
7. Pre-register your lane via `tools/lane_maturity.py add-lane <id> --name <name> --phase <N>` at L0 (Catalog #126)
8. Catalog #206 checkpoint discipline: every 10 tool uses OR every major milestone

## Subagent Coherence-By-Default — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** operator concern (2026-05-09): coherent integration without orchestration layers. Don't add skills/MCP. Engineer correctly + document discipline in CLAUDE.md + AGENTS.md so every subagent honors it via mandatory pre-read.

### Mandatory pre-flight for every subagent

Before starting any work:

1. **Read CLAUDE.md AND AGENTS.md** — both files. Honor every NON-NEGOTIABLE marker.
2. **Check `.omx/state/lane_registry.json`** for in-flight conflicts (per `tools/claim_lane_dispatch.py`).
3. **Check sibling subagents** named in the parent prompt's "running in parallel" section. Do NOT duplicate primary deliverables.
4. **Read top-10 MEMORY.md entries** — recent landings change optimal next-step.
5. **Read all `.omx/research/*_directive_*` files** dated within last 24 hours — operator-routed inter-subagent directives.

### Mandatory wire-in for every landing (no orphaned signals)

Every landing must wire into the unified solver stack OR explicitly tag `research_only=true`. Per `feedback_unified_lagrangian_action_principle_GR_style_20260509.md` (the GR-style action memo):

1. Sensitivity-map contribution (`tac.sensitivity_map.*`)
2. Pareto constraint (`tac.pareto_*` or rationale why non-binding)
3. Bit-allocator hook
4. Cathedral autopilot dispatch hook
5. Continual-learning posterior update on every empirical anchor
6. Probe-disambiguator if 2+ defensible interpretations exist

Silent omission = orphan-work failure mode.

### Anti-duplication: lane registry IS the deduplication layer

Two subagents on the same lane = registry failure. Pre-register every lane (even SKETCH at L0) the moment a name + verdict exists. Subagent prompts MUST cite the registered `lane_id`.

### Anti-fragmentation: unified-Lagrangian action

Migration target: `tac.unified_action.S_total(theta, archive_bytes, hardware)` — ONE scalar action, all track-Lagrangians composed via δS/δθ = 0 (GR-style). Until then: explicit 6-hook wire-in.

### Anti-arbitrariness: probe-disambiguator pattern

When a design choice has 2+ defensible interpretations: ship BOTH modes via callable interface + build `tools/probe_<track>_disambiguator.py`. The probe IS the arbitration. Per `feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md`.

### Background-execution clarification

Operator floated "skills or MCP tools." **Do not pursue.** CLAUDE.md + AGENTS.md non-negotiables ARE the always-on, zero-token orchestration. Adding another layer = kitchen_sink anti-pattern at meta level.

The skill-vs-rule decision: **skills are user-invocable patterns; rules are agent-binding contracts.** Coherence is solved by RULES, not skills.

Cross-ref `feedback_unified_lagrangian_action_principle_GR_style_20260509.md`, `feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md`, CLAUDE.md "Subagent coherence-by-default" section.

## Main-Branch Source Of Truth — NON-NEGOTIABLE

`main` is the sole source of truth for this repository. Do not do production
work, recovery work, public-frontier intake, or contest-custody edits on any
other branch. If another branch name points at the same commit, treat it as
historical metadata only. Do not create, checkout, or promote side branches
unless the operator explicitly asks for branch work in that turn.

Detached public PR clones and recovered/quarantined trees may be inspected as
forensic inputs, but any promoted code, docs, artifacts, or ledgers must land
back on `main` through explicit review. Never let a detached clone, stash,
quarantine directory, provider workspace, or subagent fork become an implicit
source of truth.

## Execution Accountability — NON-NEGOTIABLE

When the user asks to push score, recover state, harden a bug class, or proceed
autonomously, do real work before adding more strategy text. A valid autonomous
turn must produce or advance at least one concrete artifact: code patch, test,
candidate archive, dispatch claim, queued job, harvested eval, ledger entry,
profile artifact, or exact failure classification. Grand-council, research, and
mathematical deliberation are useful only when they directly change the next
build, eval, guard, or dispatch decision.

Avoid narrative-only loops. If blocked, record the blocker as an artifact or
guardrail, choose the next highest-EV unblocked action, and continue.

## Frontier Velocity And Anti-Conservatism — NON-NEGOTIABLE

The default operating mode is aggressive frontier movement, not conservative
local polish. When public submissions, hidden-gem lanes, arithmetic/entropy
coders, HNeRV/NeRV/SIREN-style representations, Cool-Chic/C3, RAFT/ego-motion,
foveation, wavelets, learned atom allocation, or other high-leverage evidence
appears, agents must immediately evaluate whether it changes the next dispatch
or build. Do not continue shaving a saturated local basin when a plausible
larger representation, packer, or exact-replay target is available.

Before spending time on a small slice, ask and answer in the ledger or artifact:

- Is there a live public/archive/blob/source target with a lower claimed score?
- Is there a byte-level repack, arithmetic-code, entropy-code, or payload-layout
  opportunity that can be tested on exact archive bytes?
- Is there a hidden-gem lane already implemented that can be wired into the
  current champion with less wall-clock than another micro-ablation?
- Does this action produce a candidate archive, exact replay, profiler, guard,
  or deconstruction artifact? If not, pick a higher-EV action.

Conservatism is allowed only when it protects contest compliance, custody,
public-release hygiene, or partner worktree state. It is not allowed as a reason
to avoid high-upside deconstruction, exact replay, low-level binary analysis,
or risky but contest-faithful experiments.

## Long-Burn Campaign Default — NON-NEGOTIABLE

When the operator asks for aggressive score lowering, no meat left on the bone,
no holds barred, no budget/time limit, funded reproduction, or frontier escape,
do not collapse that into more meta-review. Treat every plausible floor-breaking
representation family as a managed campaign unless a dated ledger records a
real blocker.

A campaign is stronger than a lane. It must own:

- a `lane_id` and dispatch-claim plan;
- source evidence and hypothesis for score movement;
- a timing-smoke command that measures seconds/epoch or seconds/candidate;
- a full-run command with resumable checkpoint and harvest paths;
- live-rate cost model, not stale hand estimates;
- byte-closed archive/export/inflate plan for promotion;
- stop/continue thresholds at smoke, mid-stage, export, and exact-eval gates.

Budget uncertainty is not a blocker. If cost is unknown, the next action is a
timing smoke that turns uncertainty into measured GPU-hours. Missing final
archive grammar blocks promotion and score claims; it does not by itself block a
clearly tagged non-promotional timing smoke or source-faithful reproduction
probe. Older budget caps, no-dispatch memos, or no-GPU notes are superseded by a
newer explicit operator directive to fund or launch a named campaign, while the
claim lifecycle, provider import probe, artifact custody, and exact-eval axis
rules remain mandatory.

Visible high-EV ideas such as PR95/HNeRV, NeRV-family replacements, SIREN/FINER/
WIRE/BACON, Ballé/CompressAI, Cool-Chic/C3, wavelet residuals, RAFT/ego-motion,
LA-pose/telescopic foveation, SABOR, S2SBS, arithmetic/range/ANS compiler
passes, or scorer-inverse representations must be converted within the same
session into either a campaign ledger plus timing-smoke/launch decision or an
explicit blocker. `research_only=true` is not a resting state for frontier work:
if the signal is promising, the same memo must name the next byte-closed
prototype or campaign gate.

If `.omx/state/RACE_MODE_ACTIVE.flag` exists, campaign actuation outranks
additional grand-council text unless that text directly writes launchable
commands, hardens the actuator, or records a blocker that prevents spend.

## Public Frontier Watch And Intake — NON-NEGOTIABLE

During active contest windows or post-deadline replay windows, keep the public
frontier current. Refresh GitHub PRs and official leaderboard state often
enough that late submissions are not missed while working on internal lanes.
Any PR/archive/title/body/comment that plausibly beats the local exact frontier
must enter an intake queue immediately with:

- PR number, title, author, URL, head SHA, created/updated time.
- Archive URL, local archive path, bytes, SHA-256, member names, member SHA-256s.
- `inflate.sh`, `inflate.py`, `compress.sh`, README/report, training scripts,
  binary blobs, releases, and relevant author-repo links when public.
- Claimed components and recomputed score from public rounded values, tagged
  `external` until exact CUDA replay lands.
- Compliance risks: sidecars, network installs, source-embedded payloads,
  malformed ZIP reliance, non-canonical runtime signatures, dependency gaps.
- Fastest path to exact replay or fail-closed blocker.

Use detached clones/downloads in experiment artifact directories for public PR
forensics. Do not `gh pr checkout` into the dirty shared worktree.

## Apples-To-Apples Evidence Discipline — NON-NEGOTIABLE

Do not classify, promote, retire, or submit HNeRV/public-frontier work from an
inferred equivalence. Score movement must be compared against the matching
baseline on the same evidence axis and runtime contract.

- Decoded tensor, symbol, `state_dict`, or latent parity is parser-consumption
  evidence only. It is not full-frame inflate parity and it is not scorer
  parity. Keep `full_frame_inflate_output_parity_missing` until source and
  candidate `inflate.sh archive_dir output_dir file_list` outputs are compared
  byte-for-byte, or until both packets have exact same-runtime eval artifacts.
- `[contest-CPU]`, `[contest-CUDA]`, `[macOS-CPU advisory]`,
  `[macOS-MLX research-signal]`, MPS, and proxy signals are separate evidence
  spaces. The public HNeRV leaderboard CPU axis can be materially better than
  the CUDA/T4 axis. Never convert CPU to CUDA, CUDA to CPU, MLX to either
  contest axis, or local advisory output to promotion/rank/kill authority.
- There is no universal CPU-better or CUDA-better ordering. Treat the
  CPU/CUDA gap as a per-submission, per-runtime, per-inflate-device, and
  per-scorer-device property. A valid mechanism claim must record archive SHA,
  runtime content SHA, inflate device, evaluate device, inflated raw-output
  aggregate SHA when available, and PoseNet/SegNet component deltas.
- Public PR baselines must use the source archive plus the source runtime that
  actually produced the cited result. If an adapter changes `inflate.py`,
  `inflate.sh`, section constants, dependency closure, or Python invocation,
  compare against the matching original runtime under the same evaluator path.
- A surprising negative from a byte transform that preserves decoded model
  streams is `indeterminate-harness-or-runtime-mismatch` until full-frame
  parity, same-runtime source replay, and component recomputation are checked.
  Do not turn that into a method negative or lane retirement.
- Generated reports must carry inline axis labels near words like
  "medal-band", "rounds to", "submission-ready", "auto-promote", "frontier",
  or "score gap." Missing labels are evidence bugs.

## Bit-Level Deconstruction And Repack Discipline

For archive/packer work, start from bytes, not prose. Every public or internal
archive that can affect the frontier should be inspectable at the lowest useful
level:

- ZIP structure: local/central header parity, member order, compression method,
  timestamps, flags, sizes, CRCs, duplicate names, hidden files, resource forks.
- Payload grammar: magic, fixed sections, length prefixes, section offsets,
  hashes, entropy estimates, decoded tensor shapes, codec families, side
  channels, and no-op/provenance detection.
- Compression opportunities: brotli/zstd/lzma/arithmetic/range/ANS/Huffman,
  tensor grouping, histogram overhead, prefix removal, filename/header savings,
  section-length hardcoding only when contest-compliant, and deterministic
  pack ordering.
- Runtime impact: dependency closure, CUDA availability, inflate budget,
  deterministic decode, scorer-free inflate, and runtime tree SHA.

Arithmetic coding, range coding, and other entropy coders are first-class
optimization lanes, not afterthoughts. If a dense byte stream remains in a
generic compressor, estimate its entropy and compare a real coded payload
before declaring the lane saturated.

## Cross-Agent Dispatch Coordination — NON-NEGOTIABLE (Level 2)

**Before dispatching ANY training, eval, or remote-GPU job (Vast.ai, Modal, Lightning, Azure, etc.), claim the lane with `tools/claim_lane_dispatch.py claim ...`.** The helper takes an exclusive file lock, reads `.omx/state/active_lane_dispatch_claims.md`, inserts the newest row at the top, and refuses active same-`lane_id` conflicts inside the 24-hour TTL unless an operator passes an explicit force flag with notes.

If you find an active conflicting claim:
- Do NOT dispatch
- Coordinate via the file's notes column or pick a different lane

When your dispatch completes (success or fail): append a terminal row with the
same `lane_id` and `instance/job_id` via `tools/claim_lane_dispatch.py claim
--force --status completed_...`, `--status failed_...`,
`--status stopped_...`, `--status refused_dispatch...`, or a precise
`--status stale_superseded...` row. The helper treats a newer terminal row as closing
the matching older nonterminal row for conflict detection. Do not leave
completed jobs as phantom active claims.

Manual table edits are acceptable only for emergency recovery or correcting stale historical rows. This rule exists because 2026-05-01 ~23:50 UTC the user reported a possible Q-FAITHFUL dispatch conflict between Claude (H100 SXM via Vast.ai) and codex (Lightning).

Lightning Studio-backed submitters enforce this rule in `scripts/launch_lightning_batch_job.py`: non-dry-run exact-eval, component-response, and component-sensitivity submissions must have a matching active claim row for the lane/job, unless an auditable `--allow-missing-dispatch-claim-reason` is supplied.

Modal auth-eval submitters enforce the same rule in `experiments/modal_auth_eval.py`
and `experiments/modal_auth_eval_cpu.py`. For long exact-eval runs, detach at
both layers: pass Modal CLI `modal run --detach ...` before the script path, and
pass the wrapper flags `--detach --provider-detach-ack` after the script
arguments. Wrapper `--detach` without provider-level CLI detach is forbidden:
the ephemeral Modal app can stop before the spawned function returns, producing
a blank `RemoteError` and no score artifact. Harvest detached calls through the
canonical `tools/recover_modal_auth_eval.py` path so `contest_auth_eval.json`,
`inflated_outputs_manifest.json`, runtime custody, and terminal claim rows are
preserved consistently across CUDA and CPU axes.

## Operator Gates And Discoverability — NON-NEGOTIABLE

Recovered or newly created tools are not complete until they are discoverable
from normal operator flows. If a guard, profiler, packer, recovery tool, or
submission checker matters, wire it into at least one of: `preflight_all()`,
`tools/all_lanes_preflight.py`, `tools/operator_briefing.py`, a documented
runbook, or a dated `.omx/research/` control ledger. Hidden one-off scripts
are treated as incomplete work.

## Provider Runtime Architecture — NON-NEGOTIABLE

Provider dispatch logic must stay composable, deterministic, and reusable.
Do not bury cloud-provider runtime contracts inside a lane experiment just
because a lane is urgent. Experiment files may be thin actuators that bind a
specific lane label, claim lifecycle, parameter set, and recovery policy, but
shared provider/runtime concerns belong in deploy-layer modules:

- reusable provider contracts: `src/tac/deploy/<provider>/`;
- provider-agnostic path/bootstrap logic: `src/tac/deploy/cloud_*` or
  `src/tac/deploy/base.py`;
- thin operator CLIs: `scripts/`, `tools/`, or `experiments/` only when they
  delegate to the reusable deploy/runtime module.

For Modal specifically, shared scorer/runtime dependency closure belongs in
`src/tac/deploy/modal/`, not in each lane-local Modal script. If a Modal lane
needs `upstream/modules.py`, `tac.scorer`, DALI, `safetensors`,
`segmentation-models-pytorch`, or other contest-scorer imports, use the
canonical Modal runtime helper and add an import probe before GPU training.
Missing scorer/runtime packages are an infrastructure failure
(`failed_scorer_runtime_deps` / `remote_import_probe_failed`), not a model or
lane result.

Every provider actuator must preserve deterministic reproducibility:

- plan-only default path; real spend requires an explicit execution flag;
- lane claim before provider job creation and terminal claim on all outcomes;
- mounted-code or shipped-tarball manifest with git SHA, dirty status, diff
  SHA-256s, mounted files, and runtime dependency contract;
- artifact/result harvest path that records provider job id, command, hardware,
  archive bytes/SHA, runtime-tree SHA, logs, and exact score schema;
- no provider-specific hidden state, account paths, credentials, or transient
  URLs in public artifacts;
- no score or promotion claim from proxy substrates such as Kaggle, MPS, or
  Modal CPU; promote only byte-closed archive/runtime packets through claimed
  exact CUDA eval.

Provider contract registries must not advertise exact-CUDA support for scaffold
providers. A provider is exact-eval capable only after a real lifecycle,
runtime-closure, claim, harvest, and adjudication path exists. Scaffolds such
as AWS/GCP are useful capacity planning targets, but they must stay
`exact_cuda_eval_supported=false` until implementation catches up.

Optimizer/search substrates must share one false-authority contract. Kaggle,
macOS/MPS, Optuna, CMA-ES, and other proxy rows must flow through
`tac.optimization.proxy_candidate_contract` or an equivalent canonical helper
that forces `score_claim=false`, `promotion_eligible=false`,
`rank_or_kill_eligible=false`, and
`ready_for_exact_eval_dispatch=false`. Paid dispatch queues must require an
explicit contest target marker such as `target_modes=["contest_exact_eval"]`;
missing target metadata fails closed.

MLX scorer-response rows are a special local-substrate case: they must flow
through `tac.optimization.scorer_response_dataset` and, when used for spend
triage, `tac.local_acceleration.mlx_score_calibration`. A calibrated
`[macOS-MLX research-signal]` row may select local follow-up or exact-eval
spend candidates only after parity and score-calibration gates pass; it still
cannot claim a score, promote, rank/kill, or skip exact CPU/CUDA auth eval.

If a lane-specific provider script starts accumulating package lists, path
mount rules, import probes, cost tables, timeout policy, or runtime closure
logic, stop and extract that logic to `src/tac/deploy/<provider>/` first. The
experiment script should become a small, reviewable adapter around the shared
provider contract.

## TAC / comma-lab Boundary And Research State Tracking

Keep `tac` clean, but put real reusable Python implementation in `tac`.
`tac` is the reusable Task-Aware Compression library and runtime-contract
surface, and checks that protect archive validity, inflate/runtime compliance,
CUDA-score custody, and package safety are part of that public contract. Codec
primitives, archive grammars, payload parsers, scorer/eval contracts, byte
profilers, planning primitives, visualization primitives, and contest-relevant
algorithms belong in `tac` when they are reusable. Thin CLIs may live in
`experiments/`, `scripts/`, or `tools/`, but they should delegate to `tac`
modules instead of burying implementation in ad hoc entry points.

Claude/OMX/provider/recovery policy should not enter `tac` unless the logic is
genuinely reusable codec, contest-runtime, or contest-preflight functionality.
Research custody, public-frontier intake, provider state, hosted supplements,
dashboards, and recovery audits belong in the comma-lab layer (`src/comma_lab/`,
`tools/`, `docs/`, `.omx/`).

Contest-specific public-submission reverse engineering belongs in
`reverse_engineering/`. Keep that tree clean: curated runbooks, intake
indexes, byte-anatomy notes, adapter boundaries, and small manifests are valid;
raw public PR clones, downloaded archives, provider transcripts, and large
rebuildable artifacts remain in ignored experiment/custody locations with
ledger links. Reusable wire parsers, payload grammars, profilers, atom
planners, and archive builders still belong in `tac`, with
`reverse_engineering/` documenting how they were used. Run
`.venv/bin/python tools/audit_reverse_engineering_tree.py --repo-root .`
before cleanup
or promotion sweeps so orphaned public-runtime copies, recovery specs, and
candidate `tac` modules receive explicit dispositions.

Small durable `.omx/research` ledgers and small structured summaries are
trackable git state. Raw `.omx/state/*.json`, provider logs, raw Modal/
Lightning/Vast transcripts, auto-memory snapshots, generated public-site
bundles, and large rebuildable artifacts are not. Canonicalize useful signal
into a dated `.omx/research` ledger or a `docs/paper/ara` source record; host
large canonical artifacts externally with a committed manifest.

Run `python tools/audit_research_state_tracking.py --repo-root .` when
deciding what to track, summarize, externalize, keep private, or delete after
manifesting. The implementation lives in `src/comma_lab/research_state.py` by
design so `tac` does not become an operator-state junk drawer. The
`src/comma_lab/preflight/strict_checks.py` module is an adapter/catalog surface;
`src/tac/preflight.py` remains the canonical preflight implementation.

Current durable gates:

- `tac.preflight.preflight_all()` runs
  `check_dispatch_cli_shell_hazards(strict=True)`. This blocks repeated
  dispatch bug classes before GPU spend: stale adjudicator flags passed to
  `scripts/launch_lightning_batch_job.py`, known typo flags such as `--rmote`,
  zsh-facing `path` shell variables, and local/macOS `find -printf`.
- `tools/check_dispatch_cli_shell_hazards.py --strict` is the standalone
  scanner and is also run by `tools/all_lanes_preflight.py`.
- `tools/parallel_dispatch_top_k.py` runs the exact-ready live-custody audit
  before provider fan-out. A row with `ready_for_exact_eval_dispatch=true`
  is still refused if its archive/runtime/report/manifest custody is stale, if
  terminal lane-claim evidence already retired the same lane/archive, or if the
  selected row cannot survive the same audit used by `tools/operator_briefing.py`.
  The Vast.ai provider path is dry-run-only until that launcher owns a mandatory
  pre-instance `claim_lane_dispatch.py` claim and terminal claim update. The
  recovered `tools/feedback_loop_sweep.py` scaffold is also dry-run/research
  only; paid feedback-loop work must emit a promoted exact-ready queue first.
- `scripts/pre_submission_compliance_check.py --contest-final --strict ...`
  is the canonical upload-surface gate before a judge-facing packet or public
  release. It validates required files, executable `inflate.sh`, ZIP member
  safety and local/central header parity, duplicate/hidden/resource members,
  packed-payload multiplicity, auth-eval archive identity, component
  recomputation, T4/A++ promotion stamps, runtime-tree custody, archive
  manifest freshness, report custody links, terminal dispatch-claim linkage,
  and public supplement hygiene.

Use these gates before saying a lane is ready, a packet is releasable, or a
bug class is permanently fixed. If a new repeated failure class appears, add a
focused test and wire the guard into the same visible surfaces instead of
leaving a private helper buried in `scripts/` or `tools/`.

## Source-Of-Truth Documents

Use these documents as the research/control plane:

- `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md`
- `.omx/research/council_paradigm_shift_round{1,2,3}_20260430.md`
- `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430_codex_progress.md`
- `.omx/research/contest_grade_all_lane_results_audit_20260430.md`
- `.omx/research/contest_grade_all_lane_results_audit_20260430_codex_progress.md`
- `.omx/research/shannon_floor_execution_readiness_20260430.md`
- `.omx/research/shannon_floor_execution_readiness_20260430_codex_progress.md`
- `.omx/research/shannon_floor_paper_rigor_writeup_blueprint_20260430.md`
- `.omx/research/shannon_floor_claim_matrix_20260430_codex.md`
- `.omx/research/dx_self_protecting_harness_hardening_20260430_codex_progress.md`
- `.omx/research/codex_recursive_adversarial_greenup_review_20260430.md`
- `.omx/research/lightning_pypi_compromise_security_review_20260430_codex.md`
- `.omx/research/owv3_fisher_byte_aware_redesign_spec_20260430_codex.md`
- `.omx/research/alpha_pose_preserving_redesign_spec_20260430_codex.md`
- `.omx/research/kl_distill_hardening_status_20260430_codex.md`
- `.omx/research/component_sensitivity_map_certification_20260501_codex.md`
- `.omx/research/atom_lagrangian_waterfill_sub03_system_20260501_codex.md`
- `.omx/research/contest_faithful_swarm_execution_20260502_codex.md`
- `.omx/research/charged_mask_grammar_ego_foveation_greenup_20260502_codex.md`
- `.omx/research/works_negatives_hardened_stack_20260502_codex.md`

When these documents disagree, prefer the strictest contest-grade evidence
standard and the newest dated progress addendum. Preserve history by appending
supersession notes instead of erasing old research context.

## Codebase Map

- `src/tac/` - training, codecs, archive helpers, profiles, guards, and tests.
- `src/tac/analysis/` - scorer/video/archive telemetry and feature builders:
  hard-pair maps, component traces, foveation fields, LA-POSE-lite motion
  features, byte anatomy, and opportunity manifests. These modules do not
  dispatch jobs or claim scores.
- `src/tac/optimization/` - atom allocation and policy planning:
  meta-Lagrangian ledgers, water-fill rankings, active-subspace policies, and
  stack interaction planning. These modules emit planning artifacts, not score
  claims.
- `experiments/` - canonical training/eval/build entry points and lane tools.
- `scripts/` - remote lane launchers, adjudicators, harvesters, and runbooks.
- `submissions/robust_current/` - contest submission runtime and inflate path.
- `upstream/` - contest evaluator assets. Do not patch scorer files.
- `reverse_engineering/` - clean public-submission deconstruction runbooks,
  indexes, and small manifests. No raw clones, provider logs, or large
  archives.
- `reports/` - derived reports and non-authoritative summaries.
- `.omx/state/` - dispatch state. Treat as advisory when live API or lane-local
  artifacts disagree.
- `.omx/research/` - dated scientific, mathematical, adversarial, and progress
  ledgers.

## Contest Objective

All score claims reduce to the contest formula:

```text
score = 100 * seg_dist
      + sqrt(10 * pose_dist)
      + 25 * archive_bytes / 37,545,489
```

Every scored archive must record exact archive bytes, archive SHA-256,
component distances, sample count, recomputed score, eval command, hardware,
manifest, provenance, and logs.

## Positive And Negative Signal Discipline

What works and what does not work are equally important contest assets. Exact
negative results, harness bugs, preflight blocks, queue anomalies, no-op
controls, and failed scorer-basin probes must be preserved and converted into
guardrails. Do not bury them in chat.

- Every scoped negative should record the exact archive or candidate, bytes,
  SHA-256, component distances when available, hardware, command/logs,
  evidence grade, failure class, and reactivation criteria.
- Every repeated bug class should become one of: a test, strict preflight
  guard, archive validator rule, dispatch-claim rule, manifest requirement, or
  durable AGENTS protocol.
- Byte-only wins that collapse PoseNet/SegNet are valuable boundary probes, not
  failures to forget. Use them to define cliffs, trust regions, and water-fill
  constraints.
- No-op controls are first-class bugs. Format/packer experiments must prove the
  targeted payload changed and provenance must distinguish reuse from
  decode/re-encode from true codec transformation.
- Treat the software stack as a small compiler for contest archives: typed
  components, explicit lowering passes, profile-guided feedback, deterministic
  byte emission, and exact CUDA validation as the final optimizer check.

Returned results must be investigated adversarially before they change lane
status. This applies to exact CUDA positives, exact CUDA negatives, proxy/MPS
signals, byte-only wins, remote failures, queue anomalies, and subagent
findings. The minimum result-review packet is:

- custody: archive/path bytes and SHA-256, runtime tree SHA, command, hardware,
  sample count, structured JSON path, logs, and dispatch-claim status;
- recomputation: formula score recomputed from component fields when present,
  archive byte term checked against `25 * bytes / 37,545,489`, and payload
  closure checked against the scored `inflate.sh`;
- classification: legitimate score movement, measured-config regression,
  harness bug, archive/runtime bug, proxy leakage, no-op, dependency failure,
  component collapse, timeout/infrastructure, or indeterminate;
- adversarial review: engineering, mathematical, scorer-geometry, optimization,
  and contest-compliance explanations considered before status changes;
- reactivation criteria: the exact evidence, implementation change, or
  theoretical proof that would reopen or supersede the conclusion.

Do not collapse "a result came back bad" into "the lane is dead." Bad results
usually update trust regions, dispatch gates, solver priors, byte floors,
or implementation TODOs. They kill only the measured configuration unless the
Kill Discipline section's stronger standard is met.

## Build Discipline — OSS, Paper, Production, Composability

Every lane, codec, training script, decoder, optimizer, and tool must be
designed with FOUR durable constituencies in mind from the first commit.
Optimizing only for the current contest dispatch is forbidden — work must
remain useful past the May 3 deadline.

1. **OSS-readiness.** Code lands with permissive structure (clear module
   boundaries, no hard-coded operator paths, no embedded credentials, MIT/
   Apache-compatible upstream choices). Public-facing API surfaces must have
   docstrings, type hints, and a usage example. README/section in
   `docs/` for every promoted lane. Avoid hidden private assumptions
   (machine-specific paths, undocumented env-var sentinels, magic constants
   without rationale).

2. **Paper-readiness.** Every score, every loss curve, every architectural
   choice ships with a `[evidence:<artifact>]` tag and an entry in the dated
   `.omx/research/` ledger. Hyperparameters, seeds, schedules, and
   ablations must be reproducible from the committed manifest. Any claim
   that lands in a writeup must be backed by a contest-CUDA artifact (per
   the Evidence Grades section). Mathematical derivations get a `[derivation:<source>]`
   citation pointing to the ledger entry; empirical claims get
   `[empirical:<artifact path>]`. Lane memory files contain the council
   deliberation that resolves design tradeoffs — these become paper sections.

3. **comma-ai production-readiness.** Inflate paths, decoders, and
   runtime artifacts must be small, deterministic, and run on the production
   target (T4-equivalent CUDA + 30 min budget). Custom decoders in Rust,
   Zig, or C must compile reproducibly with a documented toolchain (no
   "works-on-my-machine" Cargo features). Submission archives must carry
   their own decoder when one is needed; sidecar dependencies are
   forbidden per Contest Compliance. Provenance JSON must record every
   binary's SHA-256, build host, and reproducible-build instructions.
   Code that can be upstreamed to openpilot's compression path
   (`src/comma_lab/`) belongs there, not in lane-local one-offs.

4. **Stacking and composability.** Every component is designed to compose
   with others in the canonical order
   `representation → prediction → quantization → hyperprior → arithmetic → pack`
   (see memory `project_codec_stacking_composition_canonical_orders_20260429.md`).
   Codecs accept a typed input contract (frames/masks/poses/weights) and
   emit a typed output contract (bytes + decoder + meta), without
   in-place mutations to global state. Cross-stream coordinators (Joint-ADMM,
   water-fill, multi-pass) operate on these contracts, never on hidden
   side-channels. Bolt-on lanes (PD-V2, LCT, Ω-W-V3, NeRV-mask) must
   compose with the deploy champion's archive without re-deriving the
   primary representation. New axes ship with stack-composition
   documentation showing which other axes they're orthogonal to and
   which they conflict with.

When these four constituencies disagree (e.g., a faster contest-only
shortcut conflicts with paper reproducibility), the slower path that
preserves all four wins. The user has stated explicitly that this work
continues past May 3 for paper, OSS release, and production deployment;
short-term contest gains that compromise long-term value are net-negative.

Subagents and codex-spawned helpers must inherit these four constituencies
in their commits — review their output for OSS hygiene, ledger entries,
production fitness, and composability hooks before promoting.

## How To Design A New Substrate (UNIQUE-AND-COMPLETE-PER-METHOD)

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable
(2026-05-15). Default mode for every new substrate / codec / method /
composition is UNIQUE-AND-COMPLETE-PER-METHOD: the question is not "how do I
share with the canonical helper" but "what is the OPTIMAL ENGINEERING for THIS
specific method to achieve the lowest score possible." Canonical helpers are
TOOLS available for use when they serve. They are NOT OBLIGATIONS to extend
or share with by default.

### Why the default flipped

The 18-assumption audit
(`feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md`)
empirically established that 90%+ of substrates share 18 structural
assumptions (EMA 100% / archive.zip 100% / eval_roundtrip 97% / canonical
scorer-preprocess 97% / canonical auth_eval routing 97% / Tier-1 engineering
78-100% / etc.). The variance between substrates IS the variance of the 10%
NOT shared. The 0.196-0.199 cluster was the local-minimum produced by the
SHARED 90% — a flat plateau where every "new substrate" was structurally a
variation of the SAME implementation under different names. The
canonical-helper-share + META-layer-consolidation reflexes were structurally
suppressing substrate-optimal engineering. PR 95 winners did not have a
270-catalog META layer constraining them; they had focused unique-and-complete
implementations bound into single coherent packets reviewable in 30 seconds
(PR101 = 605 LOC total).

### The decision flowchart (the falling-rule list)

```
substrate X considering canonical helper Y?
├── EMPIRICAL: paired-comparison smoke run? ($5-15)
│   ├── YES → adopt the lower-scoring path (canonical OR unique)
│   └── NO ↓
├── PRINCIPLED: does Y's design assumption fit X's mathematical structure?
│   ├── YES, obvious-fit proof → adopt canonical
│   ├── NO (clear mathematical mismatch) → fork to UNIQUE-AND-DISTINCT
│   └── UNCLEAR → run paired-comparison smoke OR fork with explicit rationale
└── COUNCIL: when unsure + cost > $20 design work, summon council per
    CLAUDE.md "Design decisions — non-negotiable"
```

Unclear is not "canonical by default." If there is no empirical win and no
obvious-fit proof, default to a substrate-specific fork or a paired smoke that
measures whether the canonical helper suppresses score signal.

### Size budget guidance per HNeRV parity discipline lesson 7

- **Bolt-ons** share patterns and obey the ≤350 LOC budget per landing.
  Bolt-ons happen many times across substrates.
- **Substrate engineering** unique-ifies and may exceed the bolt-on size
  budget. Substrate engineering happens ONCE per architecture class.
- Treating substrate engineering like a bolt-on is the structural mistake
  the operator's 2026-05-15 retrospective named.

### Mandatory design memo discipline

Every substrate scaffold landing memo dated >= 2026-05-15 MUST include a
literal section header `## Canonical-vs-unique decision per layer` enforced
by Catalog #290. Inside the section, list every canonical helper / META
layer field / engineering pattern adoption decision per the falling-rule
above (one row per layer: scorer-preprocess, auth-eval routing, archive
grammar, inflate runtime, training curriculum, Tier-1 engineering, scorer
routing, score-aware loss). For each row, name the canonical, name the
substrate's choice (ADOPT or FORK), and give a one-line rationale
referencing either an empirical paired smoke OR a principled mathematical
mismatch OR an explicit "UNCLEAR — defaulted to canonical / forked with
council review" tag. Sister to the existing 6-hook wire-in declaration per
Catalog #125.

### Canonical worked example — STC-DASHER scaffold

The STC-DASHER scaffold v1 (`feedback_stc_dasher_scaffold_v1_arithmetic_maximalism_landed_20260515.md`)
is the canonical worked example of the share-vs-unique balance. STC-DASHER
keeps canonical math primitives (arithmetic coding helper, scorer-preprocess,
fcntl-locked state writes) shareable because they obey the falling-rule list's
"obvious-fit" branch — there is no plausible substrate-class reason to fork
fcntl-based ledger writes. STC-DASHER's codec envelope (the syndrome-trellis
encoder + Dasher-style symbol model + Filler's MaxStego loss) is intentionally
unique-and-distinct because the substrate's mathematical structure is steno-
graphic embedding under a contest-CUDA decoder constraint, which the canonical
codec helpers (Ballé hyperprior + STE entropy + cooperative-receiver loss) do
NOT serve. The design memo's `## Canonical-vs-unique decision per layer`
section should make this split explicit row-by-row.

### Cross-references

- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode — NON-NEGOTIABLE,
  HIGHEST EMPHASIS" (the binding contract).
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" (substrate-
  level PR 95 lesson; the META extension lives in the new section above).
- CLAUDE.md FORBIDDEN PATTERNS "Forbidden force-canonical-without-evaluation-
  of-suppression (the canonicalization-trap)".
- Catalog #290 (`check_substrate_design_memo_has_canonical_vs_unique_decision_section`)
  — the structural enforcement.
- `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md`
  — the principle.
- `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md`
  — the historical depth + retrospective acknowledgment.
- `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md`
  — the 18-assumption matrix + 10 NSCS substrate-class shifts.

## Beauty, Simplicity, And Developer Experience

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

## Contest Versus Production Targeting

Every candidate that could route through a dispatcher must declare enough
target intent for tooling to distinguish contest score work from comma-ai /
openpilot production exploration. The default target for exact-eval dispatch
tools is `contest_exact_eval`; production-only candidates belong in planning
ledgers, OSS/prototype benches, or production runbooks until they intentionally
produce a contest archive.

Use explicit metadata when a lane is not purely contest-scoring:

- `target_modes`: for example `["contest_exact_eval"]`,
  `["contest_exact_eval", "openpilot"]`, or `["openpilot_edge"]`.
- `deployment_target`: for example `comma_ai_production`,
  `openpilot_edge`, `t4_contest_runtime`, `desktop_research`, or
  `device_learning_optional`.
- `score_affecting_payload_changed` / `charged_bits_changed` plus old/new
  archive or payload SHA-256s when a lane changes scored bytes.

Self-compression, neural compression, on-device learning, edge-learning,
generated decoders, Rust/Zig/C/assembly kernels, and binary codegen are valid
research and production axes, but they are not valid contest dispatches unless
charged archive bits actually changed and the manifest proves the old/new byte
or SHA boundary. Outside contest mode they must remain optional capabilities
with deterministic fallbacks, explicit toolchain provenance, and no hidden
runtime negotiation. A production target can optimize latency, power, memory,
device portability, maintainability, or upstreamability differently from the
contest objective, but it must not be mislabeled as a contest score-lowering
candidate.

## Cross-Language Conformance Tests

Tests for archive parsers, codec grammars, entropy coders, byte transducers,
generated decoders, and native ports must be written as reusable conformance
assets, not just Python regression checks. Prefer golden vectors and manifests
that another implementation in Rust, Zig, C, C++, assembly, or Python can run
without understanding the original test harness.

For every promoted byte-level primitive, include:

- canonical input bytes, output bytes, SHA-256s, lengths, offsets, CRCs, and
  charged-byte accounting;
- decoded semantic facts such as tensor names, shapes, dtypes, scale factors,
  byte maps, stream boundaries, entropy tables, and padding/tail-bit rules;
- negative vectors for malformed headers, duplicate members, zip-slip names,
  truncated streams, trailing data, bad CRC/SHA, impossible entropy symbols,
  unsupported feature flags, and no-op candidate proofs;
- deterministic output proofs: same input and config must emit byte-identical
  output across repeated runs and across supported platforms;
- explicit oracle status: Python-readable reference, native implementation
  under test, or independently generated third-party vector.

Python may remain the most readable oracle during deconstruction. Native
implementations become authoritative only after they pass the same vectors
byte-for-byte and fail closed on the same invalid inputs. Training loops can be
research/proxy evidence, but fixed inference, inflate, packing, and bitstream
transforms should be reducible to deterministic byte transducers whenever the
wire contract is fully understood.

## Deterministic Submission Packet Compiler

The long-term native/codegen objective is a separate deterministic
submission-packet compiler, not a collection of lane-local rewrite tricks. The
compiler should be able to ingest any contest-compliant submission packet,
deconstruct it into typed streams and conformance vectors, then emit either an
identical packet or an intentionally byte-different packet whose changes are
proven by manifest.

The compiler has at least four legitimate targets:

- `contest_one_video_replay`: overfit to the contest video. This target may
  replace learned inference with deterministic replay, distilled byte
  transducers, generated code, lookup tables, motion atoms, or fixed
  per-frame/per-pair streams derived from the trained model's behavior on the
  one scored video. It is valid only when all replay data and runtime code are
  inside the packet or fixed contest code, the scored inflate path consumes
  them, and exact CUDA auth eval validates the resulting archive.
- `contest_generalized`: contest-compliant, but not one-video replay. This
  target must keep the runtime contract valid for unseen contest-shaped videos
  and must not require per-frame lookup tables or fixed replay data from the
  scored video. Use it when a native/codegen rewrite should remain a normal
  contest submission rather than an overfit replay artifact.
- `production_generalized`: works across videos and product targets. This
  target may use the same deconstruction and native-codegen infrastructure, but
  it must preserve generalization, device portability, maintainability,
  openpilot/comma-ai integration constraints, and deterministic reproducible
  builds across supported platforms.
- `production_edge_adaptive`: production-only, outside contest mode. This
  target may include optional on-device learning or edge adaptation only when
  deterministic fallbacks, reproducible builds, and explicit capability gates
  are present. It is not a contest dispatch target unless a separate
  contest-profile packet with charged-bit proof is emitted.

Minimum contract:

- input: a contest packet with `archive.zip`, `inflate.sh`, runtime files, and
  optional source/report material;
- intake: strict ZIP validation, runtime-tree manifest, payload grammar
  detection, member SHA-256s, charged bytes, exact offsets, decoder/codec
  contracts, and unsupported-feature classification;
- deconstruction: typed streams for masks, renderer weights, latents, poses,
  sidechannels, entropy tables, generated code, and archive metadata whenever
  the packet exposes them;
- rewrite modes:
  - `identity`: re-emit byte-identical output and prove archive/runtime parity;
  - `canonicalize`: change only contest-irrelevant metadata when allowed by
    compliance policy and report every changed byte;
  - `optimize`: change score-affecting bytes only when a decoder/runtime
    contract consumes them and exact old/new SHA plus charged-byte accounting
    is recorded;
- output: deterministic packet, conformance vectors, provenance JSON,
  reproducible build/toolchain manifest for native helpers, and a strict
  compliance report suitable for exact CUDA auth eval.

The compiler must fail closed unless the transformed packet remains
contest-compliant: all score-affecting artifacts inside `archive.zip` or fixed
contest code, no scorer modifications, no hidden sidecars, deterministic
inflate, no ZIP parser-divergence dependency, and no external network or local
state dependency. Identity and canonicalization modes are useful for OSS and
production even when `optimize` has no byte win; only `optimize` may create a
score-lowering candidate, and only with charged-bit proof.

## Public Release Hygiene And Hosted Supplements

The public submission/report/site track may use Lightning.ai notebooks and
Cloudflare Pages for the Apogee supplement, but public hosting is separate from
private custody.

- Do not commit secrets, API tokens, SSH targets, local absolute paths, private
  Lightning Studio app links, Vast job endpoints, raw provider state, or
  operator-specific account metadata to public-facing GitHub/docs/site
  surfaces.
- Use placeholders such as `${LIGHTNING_SUPPLEMENT_URL}`,
  `${CLOUDFLARE_PAGES_URL}`, `${PUBLIC_NOTEBOOK_URL}`, and
  `${APOGEE_RELEASE_MANIFEST}` until a URL has been intentionally published.
  Final public URLs belong in a sanitized release manifest, not in raw
  `.omx/state` or provider transcripts.
- `.omx/state/`, `reports/raw/`, `reports/private/`, harvested manifests, and
  provider job logs are local custody and forensic surfaces unless explicitly
  sanitized and copied into a public artifact directory.
- Public notebooks must strip execution environments, local paths, raw job
  links, and credentials before upload. They should cite artifacts by SHA,
  archive bytes, evidence grade, and relative release-manifest paths.
- Before publishing GitHub/docs/Cloudflare/Lightning supplement content, run
  `check_public_release_hygiene(strict=True, scan_paths=[...])` on the exact
  publish surface. Normal repo preflight calls the same guard warn-only because
  legacy custody ledgers intentionally preserve private local evidence.

## Evidence Grades

Use evidence grades rigorously:

- `A++`: exact 1:1 contest-grade archive evidence. Requires exact archive
  custody, clean manifest, payload closure, canonical `archive.zip ->
  inflate.sh -> upstream/evaluate.py` path, CUDA, full sample count,
  T4/equivalent or official contest-equivalent hardware, inflate budget proof,
  and adversarial review.
- `A`: exact local CUDA score-grade archive evidence with full component
  recomputation and archive custody, but not necessarily contest-equivalent
  hardware.
- `A-negative`: exact archive CUDA evidence showing a measured implementation
  regresses. This supports diagnosis and redesign, not broad family kills.
- `B`: diagnostic CUDA evidence with incomplete custody, schema, or rerun
  proof.
- `contest-CPU`: exact same archive/runtime custody through
  `archive.zip -> inflate.sh -> upstream/evaluate.py --device cpu`, full sample
  count, and component recomputation on 1:1 Linux x86_64 contest-compliant
  hardware. This is authoritative for the public leaderboard / PR-comment CPU
  axis and for CPU-vs-CUDA drift diagnosis. It is not a substitute for the
  CUDA axis, and it must not by itself promote an internal CUDA lane, anchor a
  paper score, kill a family, or retire a method.
- `empirical`: byte, smoke, loss, round-trip, partial, or component evidence.
- `derivation`: formula-only conclusion.
- `prediction`: hypothesis or forecast.
- `external`: outside paper, OSS, or leaderboard intake.
- `invalid`: CPU outside the explicit `contest-CPU` protocol, MPS, proxy,
  stale, no-op, sidecar, missing archive, or unreproducible score evidence.

No lane can promote a CUDA-axis claim, kill a family, retire a method, or
anchor stack math from prediction, byte-only, non-`contest-CPU` CPU, MPS,
proxy, smoke, memory-only, or stale-log evidence. A `contest-CPU` artifact may
rank the official public-leaderboard CPU axis only when its archive/runtime
custody is exact and the missing CUDA/CPU counterpart is recorded instead of
inferred.

## Dual-Axis Auth Eval Truth

MPS, CPU, local proxy scorers, local renderer checks, and non-canonical eval
paths can materially distort SegNet/PoseNet behavior and total score. They are
useful only for development, byte checks, shape checks, smoke tests, and bug
triage.

For any GPU-dependent score or signal claim, the reliable GPU-axis source of
truth is exact CUDA auth eval on the exact archive bytes through:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

Prefer `experiments/contest_auth_eval.py --device cuda` for this path and use
its `contest_auth_eval.json` as the canonical artifact. Local M-series/MPS
output through SegNet/PoseNet scorers must never promote, rank, kill, retire
a method, validate a stack, or anchor paper claims. If a local/MPS result
disagrees with CUDA auth eval, CUDA auth eval wins on the CUDA axis.

The public leaderboard has an additional official CPU axis. Do not assign
global priority to CPU or CUDA until both axes are measured for the same
archive/runtime; report the pair as a two-axis result, not a scalar
extrapolation.

## Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE — NON-NEGOTIABLE

**Every archive that ships in a PR or that we use to claim "medal-band"
or "frontier" status MUST get authoritative auth eval scores on BOTH
`--device cuda` AND `--device cpu` through `upstream/evaluate.py`, AND
both must run on hardware that is 1:1 contest-compliant with the
contest's GitHub Actions CI runner.** The contest leaderboard ranks by
the CPU eval, not the CUDA eval. Verified 2026-05-08 from PR #102 (third
prize) public bot comments:

- PR102 public CUDA: 0.22839 — matches our T4 CUDA replay within 3e-6
- PR102 public CPU: **0.19538** — this is the medal-band score the prize was awarded against
- PR104 public CUDA: 0.23115 — matches our T4 CUDA replay within 1e-5
- Our PR #107 (apogee submission): public CUDA 0.22936; maintainer did not
  publish a CPU comment, but lab GHA Linux x86_64 replay landed
  `0.1966358879` with exact archive/runtime custody.

Earlier guidance "Local M-series/MPS or CPU output must never promote
[etc.]" was a conflation of two different paths:

- **Local CPU forward pass through SegNet/PoseNet scorers** during
  training/proxy: NOISE, similar drift class to MPS. Still forbidden as
  authoritative axis.
- **Contest CPU evaluator** via `upstream/evaluate.py --device cpu` on
  EXACT submission archive bytes: AUTHORITATIVE — this IS the contest
  leaderboard's official scorer.

**1:1 hardware-compliance rule (NON-NEGOTIABLE):**

- Local macOS (M-series ARM, Intel iMac, anywhere on Apple Silicon or
  otherwise) is NEVER a 1:1 axis for CPU auth eval. ARM CPU floating-point
  intrinsics differ from x86_64 in ways that affect SegNet/PoseNet output
  bytes.
- Required CPU substrate: **Linux x86_64** (Ubuntu LTS, matching the
  contest's GitHub Actions `ubuntu-latest` runner family; AMD EPYC or
  Intel Xeon class).
- Required CUDA substrate: **NVIDIA T4 / A100 / 4090 / equivalent** on
  Linux (matching the contest's CUDA runner; T4 is the contest's
  reference for the bot's CUDA comments).
- Both eval paths must use IDENTICAL upstream `evaluate.py` SHA, IDENTICAL
  `public_test_video_names.txt`, IDENTICAL video payloads, IDENTICAL
  `inflate.sh` runtime tree, IDENTICAL archive bytes.

**Where to run CPU auth eval (1:1 contest-compliant):**

- **Modal CPU container** (Linux x86_64; ~$0.06/hr; recommended)
- **Lightning CPU Studio** (Linux x86_64)
- **Vast.ai CPU instance** (Linux x86_64; cheap)
- **GitHub Actions CI workflow** itself (the actual contest hardware)
- **NOT** local M5 Max / Apple Silicon / any macOS as the authoritative axis.
  Local macOS CPU is allowed as a high-throughput advisory/dev-loop signal
  (PR107 M5 Max `0.19664189` vs GHA Linux x86_64 `0.1966358879`, delta
  `6e-6`), but tag it `[macOS-CPU advisory only]`, NEVER `[contest-CPU]`.

**Operational rules:**

1. Dual-eval is mandatory for any submission packet. Produce BOTH a
   `[contest-CUDA]` artifact AND a `[contest-CPU]` artifact on the same
   archive bytes BEFORE PR'ing or before declaring frontier status, BOTH
   on 1:1 contest-compliant hardware.

2. Both tags are authoritative for their axis but not interchangeable.
   Report both; do not extrapolate one from the other. The CUDA−CPU gap
   is per-archive empirical, not a constant; PR102 saw +0.033, but
   architecture/checkpoint drift can shift this.

   Mechanism attribution is not closed. Treat DALI/NVDEC-vs-PyAV decoder
   bytes, CPU/CUDA forward-kernel drift, and pose-head numerics as competing
   hypotheses until a 2x2 decoder/network split lands. Earlier FastViT
   attention/TF32 explanations are not valid for FastViT-T12 on T4.

3. CPU eval discipline: clean CPU-only PyTorch env on Linux x86_64
   (verify `torch.cuda.is_available() == False` and no MPS path). Use
   `--device cpu` directly on `upstream/evaluate.py`. CPU eval on a
   small Vast.ai / Modal CPU instance takes 60-120 min for 600 samples
   (matching the contest GitHub Actions CPU runner).

4. Tag distinctness. `[contest-CPU]` is its own tag, distinct from
   `[contest-CUDA]`, `[MPS-PROXY]`, `[MPS-research-signal]`,
   `[advisory only]`, and `[CPU-prep proxy]`. Never collapse them.

5. Lane Maturity registry must reflect both axes. A lane reaching
   Level 2/3 with a `[contest-CUDA]` anchor but no `[contest-CPU]`
   anchor is incomplete for medal-band ranking purposes — record both
   or record the missing one as a known gap.

6. Existing CUDA-only artifacts are NOT retroactively invalidated. They
   remain `[contest-CUDA]` with their CUDA-axis truth value. The
   dual-eval mandate is forward-looking from this rule's commit.

**Required tooling surfaces:**

- `experiments/contest_auth_eval.py --device cuda` emits the promotable CUDA
  axis when it is full-sample T4-equivalent and marked `A++`.
- `experiments/contest_auth_eval.py --device cpu` emits `contest-CPU` for
  full-sample public leaderboard reproduction. It must set
  `promotion_eligible=false`, `score_claim_valid=false`, and
  `rank_or_kill_eligible=false`.
- `tools/plan_dual_device_auth_eval.py` creates paired CPU/CUDA command plans
  for the same archive/runtime.
- `tools/plan_public_pr_cpu_auth_eval.py` plans or runs public-PR CPU replay
  from the reproduction ledger.
- `tools/public_pr_eval_comment_scorecard.py` extracts host PR-comment CUDA
  and CPU component rows so apparent drift is classified by device, not
  guessed.

Cross-references: CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
section (canonical statement of this rule); the codex drift hypothesis
matrix at `.omx/research/public_replay_drift_hypothesis_20260508_codex.md`
(empirical basis); memory file
`feedback_dual_cpu_cuda_auth_eval_mandatory_20260508.md`.

## Local MPS Research-Signal Harvesting

The local Apple MPS device is an underused free discovery engine. Use it
aggressively for long cheap sweeps that identify curve shapes, code failures,
and candidate priors, but never let it become score evidence.

Allowed MPS uses:

- proxy curve-shape discovery for distortion-vs-bytes and loss-vs-bytes sweeps;
- smoke tests and code-correctness checks;
- candidate generation priors for the meta-Lagrangian, Pareto, bilevel, and
  dispatch-advisor planners;
- early flattening detection before spending CUDA dollars.

Required MPS artifact contract:

- `evidence_grade="MPS-research-signal"` and
  `evidence_semantics="mps_proxy_curve_shape_only"`;
- `score_claim=false`, `promotion_eligible=false`,
  `ready_for_exact_eval_dispatch=false`, and `dispatchable=false`;
- explicit blockers including `mps_proxy_signal_not_score_evidence`,
  `not_cuda_auth_eval`, and `requires_exact_cuda_auth_eval_before_any_score_use`;
- no lane rank, kill, retirement, paper empirical claim, or stack validation
  until the same idea produces byte-closed archive evidence and exact CUDA auth
  eval.

The canonical adapter is
`tools/build_mps_research_signal_manifest.py`, backed by
`tac.optimization.mps_research_signal`. Feed its proxy atoms into planners only
as priors; planners must keep those rows proxy-only and non-dispatchable.

## Contest Compliance

Non-negotiable compliance rules:

- Neural/runtime artifacts required by inflate must be inside `archive.zip` or
  fixed contest code.
- Do not modify upstream scorer files.
- Do not use local renderer shortcuts for score claims.
- Do not load score-affecting sidecars outside the archive.
- Archive manifests must exclude resource forks, hidden files, caches, debug
  payloads, and zip-slip paths.
- Packed renderer archives must contain exactly one packed payload container
  (`p`, `renderer_payload.bin`, or `renderer_payload.bin.br`). If more than
  one container is present, inflate must fail closed before scoring. Exact
  eval provenance must preserve the runtime unpack summary so the logical
  members, bytes, and SHA-256s are auditable after unpack.
- Use deterministic archive construction: fixed member ordering, timestamps,
  permissions, compression settings, and manifest records.
- Strict-score archives must not depend on ZIP parser divergence. The central
  directory and each local file header must name the same nonempty member, and
  duplicate member names are forbidden. Public submissions that rely on
  `unzip` accepting malformed ZIPs may be studied as external/current-workflow
  forensics, but they are not strict custody evidence for our own claims.
- Public fixed-slice single-member payloads must be sliced by their raw wire
  contract, not by whatever order a runtime metadata wrapper reports. For
  PR67/QZS3/QP1-style payloads the raw order is `masks.mkv`, `renderer.bin`,
  then `optimized_poses.bin`; builders and profilers must verify each slice by
  charged bytes, SHA-256, decompression success, and decoded magic. Total
  archive-length heuristics are advisory only and may not dispatch GPU work
  unless decompression/magic validation proves the split.
- Exact eval path is canonical: `archive.zip -> inflate.sh ->
  upstream/evaluate.py`, preferably through `experiments/contest_auth_eval.py`.
- JSON artifacts are authoritative. Do not parse scores from human logs when a
  structured `contest_auth_eval.json` exists.
- Recompute the score from components before claiming any result.
- Exact eval provenance must record the fixed inflate/runtime tree hash, not
  only `archive.zip` SHA-256. Identical archive bytes can score differently
  when repo-local runtime Python changes; `experiments/contest_auth_eval.py`
  records `inflate_runtime_manifest.runtime_tree_sha256` and the runtime file
  list. Any cross-run comparison with identical archive SHA but different
  runtime tree hash is a runtime-custody comparison, not a pure archive
  comparison.
- Exact adjudication may reject on PoseNet/SegNet component gates even when
  the total score is in band. Component collapse is a first-class failure mode.
- Learned codec/corpus lanes must emit deterministic manifests with checkpoint
  paths, sizes, hashes, selected tensors, block counts, and exclusion reasons.
- No dummy/random sensitivity, hard-pair, Fisher, or scorer-side proxy signal
  is allowed in a promotable dispatch. Smoke/debug modes must be explicit and
  non-promotable in provenance.
- Archive and diagnostic ZIP handling must be zip-slip safe: no absolute paths,
  parent traversal, resource forks, or hidden sidecars.
- Any tool that consumes `renderer.bin` or a renderer-like checkpoint for
  pose regeneration, TTO, tracing, sensitivity, component response, or exact
  eval support must content-detect the wire format before loading. Packed
  contest renderer formats (`QZS3`, `MQZ1`, `QFAI`, `OWV2`, `OWV3`, `NWC1`,
  `NWCS1`, `IMPS`, `SCv1`, `SZv1`, `CCh1`, `C3R1`, etc.) must delegate to the
  canonical contest inflate loader or an exactly equivalent reviewed loader.
  Unknown non-pickle magic must fail closed in preflight; never let it fall
  through to `torch.load()`. This is the C-063 pose-regeneration extension of
  the DEN-V2/SHIRAZ loader bug class.
- Renderer-transplant/self-compression candidates that preserve charged
  `masks.mkv` and `optimized_poses.bin` but replace `renderer.bin` must pass
  `experiments/preflight_renderer_transplant_pose_safety.py` against the exact
  source/candidate archive SHA pair before any exact-eval dispatch command is
  considered valid. Byte closure and loader compatibility are insufficient:
  the local runtime output-parity gate must report
  `safe_for_exact_eval_dispatch=true`, and the transplant preflight/readiness
  planner must fail closed when the pose-safety report is missing, failed,
  stale, or mismatched by SHA.
- Any TTO, pose-regeneration, scorer-target, or training-support tool that
  requests only the contest window of a video must push that frame limit into
  the decoder itself. Do not decode a longer video and slice afterward.
  `tac.data.decode_video(..., max_frames=N)` is the guard. This prevents
  hidden wall-clock failures where `load_gt_video(n_frames=1200)` spends CPU
  time materializing frames that cannot affect the archive.
- Pose-regeneration/TTO archive-isolation runs that already provide
  `--gt-pose-targets`, do not use KL distillation, and will immediately run
  exact CUDA auth eval should pass `--skip-proxy-score`. Proxy scores are not
  evidence, and decoding/rendering GT-video proxy comparisons can dominate
  wall-clock without changing archive bytes.
- QZS3/JointFrameGenerator pose-regeneration on CUDA must keep renderer
  micro-batches below PyTorch's Conv2d 32-bit indexing limit. The canonical
  guard is `experiments/optimize_poses.py::apply_renderer_cuda_batch_safety`,
  which caps QZS3 CUDA `--batch-pairs` to 32 before the first renderer call.
  Do not relaunch failed `batch_pairs=100` pose jobs by only changing a driver;
  the guard and regression test must remain in place.
- If a legacy `masks.mkv` archive contains `alpha4_residual_repair.amr1*`,
  the default `inflate_renderer()` path must apply the repair and the exact
  eval inflate log must show `Applied Alpha residual repair`. It is not enough
  for helper/TTO paths to support the member. A scored run with the repair
  member present but no apply log is a no-op harness bug, not method evidence.
- AMR1 repair over half-frame mask streams must preserve tensor metadata such
  as `_half_frame_only`. A repair transform that clones class tensors and drops
  this metadata can silently switch the renderer from 1200-frame reconstruction
  to 600-frame output; strict auth eval catches the wrong raw size, but the
  runtime test must guard it before dispatch.

## Kill Discipline

Do not use broad `KILL`, `dead`, or permanent-retirement language unless the
Grand Council has completed deep adversarial review and reached three clean
consensus passes on scope.

User mandate 2026-05-08: investigate every returned result deeply,
adversarially, and rigorously; falsify or kill only as an absolute last
resort. A single exact CUDA failure can retire a measured archive/config after
custody review, but it cannot retire the method family unless independent exact
evidence or a mathematical impossibility proof survives the review packet
above, research-path exhaustion, and documented reactivation criteria.

For any bad, surprising, or disappointing result:

1. Preserve the exact archive, JSON, logs, manifest, SHA, source provenance,
   environment, and command before cleanup.
2. Recompute score from components and verify device, sample count, archive
   bytes, eval path, and payload closure.
3. Classify failure mode: legitimate regression, harness bug, archive bug,
   no-op/encode-discard bug, config/dead-flag bug, CPU/MPS/proxy leakage,
   sidecar dependency, codec attribution confound, KL/PoseNet collapse, data
   geometry mismatch, timeout/NVDEC infrastructure, or indeterminate.
4. Run engineering, mathematical, geometry, and optimization review before
   drawing a conclusion.
5. Run mitigation and stacking analysis before retirement: hybrid residuals,
   fallback routing, side-info accounting, per-region gating, PFP16/SA/H-V3/
   OWV3-style composition, or other full-stack rescue paths.
6. Run leaderboard reverse-engineering analysis: archive member sizes,
   representation family, raw-output geometry, stream allocation, and likely
   full-stack strategy.
7. Scope any retirement narrowly to the measured implementation/config unless
   independent exact reproductions or a mathematical impossibility proof support
   a broader claim.

Use these status words precisely:

- `run abort`: budget, timeout, smoke, or control threshold. This is not
  scientific failure evidence.
- `measured-implementation retired`: exact artifact/config failed after custody,
  scorer, archive, and harness checks.
- `family/method killed`: only after independent exact evidence or a
  mathematical impossibility argument plus clean Grand Council consensus.

Every negative result should produce redesign options, not just a verdict.

Returned results must also receive a composition review before any retirement
language. For every exact positive, exact negative, surprising proxy, or
byte-only result, record whether it is likely additive, antagonistic,
orthogonal, or redundant with the current champion and adjacent lanes. Review
HStack and VStack forms explicitly: parallel component splits, serial
representation->prediction->quantization->hyperprior->arithmetic->pack stacks,
multi-pass refinement, per-tensor/per-channel routing, residual rescue, and
hybrid fallback. A bad standalone score can still be valuable as a component,
trust-region boundary, sensitivity map, side-info source, or allocator prior.
Do not call a result exhausted until this stacking/synergy/antagonism analysis
is preserved in the result ledger or review packet.

## Scientific And Mathematical Rigor

- Distinguish measurement, derivation, prediction, and external motivation.
- Additive component deltas are not composable until standalone exact evals and
  a stacked exact eval exist.
- Dykstra/ADMM language is a feasibility and projection discipline, not proof
  that sampled nonconvex codec deltas compose.
- Use Dykstra-style intersection constraints: rate budget, SegNet distortion,
  PoseNet distortion, archive compliance, inflate budget, and reproducibility.
- Treat side information as charged bytes inside the archive.
- Require calibration/holdout stability for sensitivity maps and learned
  allocation rules.
- Keep exact eval single-device unless distributed sampling and aggregation are
  audited for no duplicate/missing samples.
- No Shannon-floor attainment claim is allowed without exact contest-grade
  evidence. The broad mandate is to push aggressively toward the floor, not to
  inflate claims.
- Internal shorthand `Yousfi-Fridrich floor` means the contest-task
  MDL/rate-distortion floor for a charged sufficient-statistic program of the
  fixed video and fixed SegNet/PoseNet evaluator. It may sit below generic
  human-video Shannon intuition because the target is machine score, not
  perceptual reconstruction. It does not relax contest compliance: every
  postfilter, learned decoder, GAN-style refiner, latent, pose stream, mask
  grammar, and entropy-code bit must be charged inside `archive.zip` or fixed
  contest code, and only exact CUDA eval can claim progress toward it.
- Every new atom, radius, threshold, loss weight, quantizer, selector, basis,
  foveation field, ego-motion model, decoder knob, and archive-packing rule
  must be grounded before it can drive dispatch. Record which contest score
  term it targets, which domain prior or measured artifact motivates it, what
  hardware/runtime constraint it respects, what bytes are charged, and what
  evidence grade supports it. Ungrounded constants are allowed only as
  explicitly tagged heuristics in planning-only tools.
- Prefer differentiable, learned, or statistically fitted proposal mechanisms
  over hand-tuned grids when the contract permits it: Lagrangian water-filling,
  Fisher/Hessian/Jacobian influence, active subspaces, Gumbel/STE selectors,
  bandit/BO/CEM proposal search, and ego-motion or camera-geometry manifolds
  should replace arbitrary scalar sweeps over time. The final archive remains
  deterministic and byte-closed; the differentiable learner is either charged
  inside the archive or used only to choose charged payload atoms at
  compression time.
- Any hard-coded cutoff that survives into an archive builder must have one of
  three records: exact component-response support, an ablation/sweep manifest,
  or a mathematical feasibility bound. Otherwise it must fail closed as
  `planning_only` and cannot be promoted, stacked, or cited as score evidence.

## Closed-Loop Compiler-Style Optimization

Treat the contest archive pipeline like an optimizing compiler with profile-
guided feedback. Each stage should emit a typed, machine-readable profile
artifact that later stages and the next optimization pass can consume:

- representation/profile facts: tensor shapes, stream contract, decoder
  contract, payload member names, byte counts, SHA-256, runtime, and legality.
- scorer-profile facts: exact component distances, pair/frame/class/component
  deltas, hard-pair opportunity density, confidence, and failure class.
- optimizer facts: selected atoms, rejected atoms, Lagrangian multipliers,
  water-fill budgets, active subspace basis, constraints, and interaction
  assumptions.

These feedback edges are allowed to guide aggressive search, but only exact
CUDA archive evidence can promote, rank, kill, or anchor paper claims. Proxy
losses, component traces, public-submission anatomy, Hessian/Fisher maps,
openpilot/camera/ego-motion priors, and learned selectors are compiler profile
feedback, not score truth. Before a feedback artifact changes a dispatch, it
must record its input archive SHA/bytes, source command, hardware/runtime
environment, evidence grade, and whether it is promotable.

Lane-specific telemetry can become global scorer feedback. For example, Lane W
hard-pair weights are CUDA per-pair scorer telemetry over the contest video;
they must be canonicalized as `src/tac/analysis` inputs and may route any
downstream archive atom family. Do not silo such maps inside their original
lane once they have general scorer meaning.

Optimization passes should explicitly look for positive and negative feedback:
an upstream representation can expose cheaper downstream pose coding; a
renderer/pose change can shift SegNet hard pairs; a packer can make a repair
atom cheap enough to become worthwhile; and a side-channel can improve one
component while damaging another. Do not assume additive deltas compose. Use
exact stacked archives to validate synergies and antagonisms.

Planning artifacts that rank atoms or policies must emit learnable feedback
fields when feasible: rate-score cost, break-even component benefit, bytes per
changed element, bytes per run/component, trust-region membership, no-op
status, and whether the record is positive/negative/neutral training signal.
No-op or source-preserving policies must be penalized or marked
non-dispatchable in the artifact itself. A low byte estimate is not sufficient
for dispatch unless the artifact also shows an archive-relevant state change
and a plausible break-even component path.

### Meta-Lagrangian, Pareto, And Learnable Field Solver

The meta-Lagrangian/Pareto system is a living optimization kernel, not a static
ranking report. Whenever an agent touches scoring, stacking, hidden-gem
deconstruction, public-frontier intake, entropy coding, sensitivity maps,
foveation, pose, categorical labels, or cross-paradigm wiring, it must ask
whether the change should make the planner more complete, more correct, more
learnable, more deterministic, simpler to operate, or faster to solve. If the
answer is yes, update the planner contract or record the explicit blocker in a
dated `.omx/research/` ledger.

Continuous improvement means pushing the planner toward learnable and solvable
theoretical-floor discovery, not preserving a snapshot of today's ranker. No
signal loss: keep exact CUDA outputs, byte custody, runtime-tree hashes,
commands, assumptions, negatives, calibration residuals, and cross-paradigm
rows machine-readable so future agents can reseed the solver instead of
reconstructing intent from prose.

Every stackable or substitutive atom should converge toward a shared field row:
candidate id, family, paradigm(s), role, pareto scope, charged-byte delta,
expected SegNet/PoseNet/rate deltas, confidence/uncertainty, evidence grade,
archive and runtime custody, interaction assumptions, conflicts, Volterra or
higher-order interaction terms when known, KKT/ADMM residuals when applicable,
expected information gain, blockers, and next required proof. Orphaned research
artifacts that can affect score are incomplete until they are either wired into
the cross-paradigm inventory/meta selector or explicitly marked non-actionable
with evidence.

Prefer solvable formulations over arbitrary sweeps. New thresholds, weights,
field parameters, codebook choices, foveation radii, quantizer settings, and
selection heuristics should be derived from at least one of: entropy-rate
decomposition, MDL, Fisher/Hessian/Jacobian sensitivity, Frechet/adjoint
derivatives, Dykstra/ADMM feasibility residuals, Bayesian experimental design,
optimal transport/camera geometry, exact component-response data, or a
documented ablation manifest. If a parameter is still heuristic, tag it as
planning-only and keep it out of dispatch readiness until evidence arrives.

The planner must learn from every exact result and every high-quality negative.
Exact CUDA evals, byte-equivalent no-op controls, component collapses,
decode/re-encode failures, runtime blockers, and successful hidden-gem repacks
should become calibration rows, trust-region updates, Pareto constraints,
interaction terms, or hard guards. The ideal loop is:

```text
formulate objective and constraints
-> emit typed atoms/policies
-> prune by custody, Pareto, KKT/ADMM, and interaction gates
-> select by score delta plus expected information gain
-> build deterministic candidate archive
-> exact CUDA eval
-> reseed calibration and planner weights
```

Keep this system elegant and operational. Remove stale planner fields and dead
adapters when a better contract replaces them; keep JSON schemas deterministic;
add focused tests for every new blocker or objective term; make selection fast
enough for loop use; and preserve cross-platform behavior. A powerful equation
that cannot emit a byte-closed archive, a dispatch packet, or a learnable
feedback row is not yet useful.

Planner recipes, dispatch snippets, and schema migrations must be grounded in
the current tool surfaces. Grep the real argparse/help contract before writing
or invoking any flag, and record a blocker or add a reviewed interface when the
solver needs a capability the tools do not expose. Never invent flags, schema
keys, or evidence fields to make a theoretical plan look executable.

### Hardware, Camera, And openpilot Priors

The contest video was captured by real comma/openpilot hardware and scored on
fixed CUDA hardware. Treat those facts as optimization priors, not as excuses
for unchecked shortcuts.

- Camera calibration, ego-motion, vanishing point, horizon, lane geometry,
  rolling temporal structure, and openpilot-style pose semantics may define
  allocation fields, foveation centers, hard-pair priors, low-dimensional
  motion bases, curriculum weights, and learned proposal distributions.
- These priors are development signals until a concrete archive consumes them
  through charged payload bytes or fixed contest code and passes exact CUDA
  auth eval. A plausible radial zoom, telescope/foveation transform, or
  ego-flow model is invalid if the inflate runtime does not actually consume
  it in the scored path.
- Prefer using ego-motion and camera geometry first as atom-ranking fields and
  active-subspace coordinates. Pixel warps, mask expansion warps, or learned
  geometry decoders require parity tests, payload-closure checks, and exact
  component gates because small geometric errors can catastrophically damage
  PoseNet.
- Hardware-specific speedups are allowed for compression search and inflate
  runtime engineering when they are deterministic and contest-legal. Promotion
  claims still require the canonical `archive.zip -> inflate.sh ->
  upstream/evaluate.py` CUDA path on T4/equivalent hardware, with runtime and
  dependency provenance recorded.

### Yousfi-Fridrich Atom-Field Planning Contract

The atom-field planner is the sanctioned bridge from high-dimensional math to
contest archives. Use `experiments/plan_yousfi_fridrich_field_equations.py`
to convert planning-only atom ledgers into deterministic policy JSON. Use
`experiments/build_cmg3_adaptive_runs_candidate.py --field-policy-json ...`
to consume those policies as archive bytes. This keeps the equation system,
selected atoms, charged byte proxies, and concrete archive builder coupled
without turning proxy math into a score claim.

Two modes are allowed:

- `contest`: the practical low-order projection. It emits archive-builder
  policy candidates over charged atoms and remains `planning_only` until exact
  CUDA auth eval.
- `ideal`: the infinite-compute all-order field equation. It may record
  Taylor/Frechet, Fourier/Walsh, Riemannian, Feynman/CEM, Dykstra/ADMM, and
  learned-control search plans, but it must not dispatch or claim score.

Configuration should be reproducible but fast to operate. CLI flags are the
source of truth in manifests; environment variables are acceptable defaults
for iteration; Python kwargs are acceptable for tests and composable tooling.
Supported field-planner env defaults include `PACT_FIELD_EQUATION_MODE`,
`PACT_FIELD_CANDIDATE_SIZES`, `PACT_FIELD_MAX_SOURCE_ATOMS`,
`PACT_FIELD_INTERACTION_MODEL`, `PACT_FIELD_CURVATURE_STRENGTH`,
`PACT_FIELD_PAIR_ANTAGONISM`, `PACT_FIELD_FRAME_ANTAGONISM`,
`PACT_FIELD_CLASS_SYNERGY`, `PACT_FIELD_LOW_RANK_MODES`,
`PACT_FIELD_POSITIVE_PROXY_ONLY`, and `PACT_FIELD_POLICY_PREFIX`.

For CMG3A archives, always align the planner residual basis with the builder
base. If a ledger was computed against a top-2 row-run candidate, build with
`--base-runs-per-row 2`; if it was computed against top-1, build with
`--base-runs-per-row 1`. The archive manifest must record the selected
policy id, source policy SHA-256, matched/unmatched row-run atoms, and
base-runs semantics.

Field-policy archive builds must fail closed on duplicate selected atoms,
unmatched atoms, and base-run mismatches. Negative field-energy policies should
be filtered by default; emitting them requires an explicit
`--allow-negative-field-energy` cliff-mapping decision and does not justify
remote dispatch by itself.

CMG3A target-body selection must never assume compressed body bytes are
monotonic over a priority prefix. `--target-body-bytes` callers must use the
builder's nonmonotonic search contract and record `body_search` in the
manifest. Exhaustive mode is exact only over every prefix; coarse/auto sampled
mode is a deterministic byte-screen heuristic and must record evaluated and
unevaluated prefix counts. Do not describe sampled body-budget selection as an
exact optimizer, and do not promote a candidate from body-budget proxy alone.

### Local-Minimum Escape Discipline

This project is a nonconvex rate-distortion search. Treat local minima as an
expected failure mode, not an exception. Once a lane reaches diminishing
returns, every additional polishing dispatch must be paired with one of:

- an exact CUDA eval candidate that can plausibly cross a leaderboard threshold;
- an orthogonal representation/decoder/packer family with independent failure
  modes;
- a finite policy search or atom-allocation proof that the remaining local
  neighborhood has been materially exhausted;
- an explicit ledger note explaining why diversification would cost more
  wall-clock than it saves.

For mask grammars, do not freeze a row-fill, draw order, default class,
reconciliation rule, or residual selector because it was convenient in the
first implementation. Search finite policy spaces exhaustively when they are
small, record the search cardinality and winning policy, and keep runtime
decode parity tests for every emitted policy mode. When the finite policy
space becomes too large, emit a deterministic planning ledger with evaluated
and unevaluated counts, then use exact CUDA archive eval to calibrate the next
allocator.

Remote PMG/row-span mask-grammar dispatches must carry a geometry-escape
proof, a learned/pose-safe contract, or an explicit guarded replay marker. The
permanent preflight guard is
`check_pmg_remote_dispatch_requires_geometry_escape`; do not spend GPU on a
byte-only PMG/CMG3 row-span candidate after same-family PoseNet-collapse
evidence unless the command cites why it escapes that measured failure mode.

Multimask and multichannel reconciliation lanes are valid orthogonal escape
paths when every score-affecting bit is charged. They must remain
non-promotable until a concrete archive with the reconciler and all mask/latent
payloads inside `archive.zip` receives exact CUDA auth eval.

### High-Upside Learned-Codec Non-Conservatism

Do not let clearance gates silently become lane retirement. NeRV/HNeRV/INR,
learned renderer, learned mask, learned latent, Muon/AdamW/QAT, and other
large learned-codec families are high-upside floor-breaking candidates. A lane
that is blocked by missing parser support, missing provenance, missing L2
clearance, unstable training, or incomplete runtime closure must have one of:

- an active owner and a concrete unblock artifact;
- a paid or local experiment queued under the dispatch-claim rules;
- an exact negative CUDA artifact that scopes the measured implementation; or
- a mathematical impossibility argument recorded in a dated research ledger.

Absence of clearance is not evidence that a family is low value. Public
breakthrough submissions or credible external claims in these families must be
handled in this order: exact replay of their archive/runtime, byte-and-runtime
deconstruction, local parity/profile extraction, then an owned reproduction or
improvement lane. Do not spend wall-clock on low-EV polish while a newly open
floor-breaking learned codec has not been replayed or assigned.

When a prior lane was delayed by conservatism, preserve that as a process bug
and add a guard, checklist item, or dispatch policy. The correction is not to
lower evidence standards; it is to run higher-upside experiments earlier while
keeping archive custody, charged bits, exact CUDA eval, and deterministic
manifests intact.

## Shannon-Floor Execution Policy

Shortest wall-clock progress comes from parallel independent hypotheses, not
from serial speculation.

Operate these streams in parallel when write sets and scientific failure modes
are independent:

- Alpha representation work: mask/video/latent payloads that preserve scorer
  geometry and temporal information.
- Beta sensitivity work: per-channel/per-region score sensitivity,
  mixed-precision allocation, water-filling, and PoseNet/SegNet protection.
- Renderer compression work: renderer-byte reduction with exact distortion
  gates.
- Pose-byte work: small deterministic archive wins with no scorer side effects.
- Hidden-gem recovery: bugged lanes re-engineered under strict evidence gates.
- Gamma coordination: ADMM/MDL/entropy/hyperprior/range or arithmetic coding
  only after measured components exist.
- Custom overfit decoder work: Rust/Zig/C/static-binary, Python bytecode,
  bitpacked RLE/ANS/range-coded streams, temporal grammars, small learned
  decoders, RL/bandit searched payloads, and unlimited offline compression
  search are in scope when they are contest-compliant. The complete decoder
  contract and all score-affecting payload bits must be inside `archive.zip`
  or fixed submission code; no scorer patches, external sidecars, network
  fetches, host-local files, or nondeterministic runtime generation are
  allowed. Complex lanes have the same priority as small byte-shaving lanes
  when they can run in parallel and preserve exact custody.

Stack experiments wait until component archives have exact evidence. A stack is
its own archive and must pass its own exact eval.

## 0.33-Or-Below Public-Floor Path

The May 3 leaderboard objective requires a `0.33` or lower contest-valid
archive. When current internal archives are far above that band, do not spend
the critical path on incremental OWV3/Alpha polish unless it directly supports
the public-floor basin. The active high-EV route is:

1. Treat public PR #63 `qpose14` and PR #64 `unified_brotli` as the measured
   basin contract, not just inspiration: decoded mask SHA, renderer payload
   family, one-scalar pose manifold, PR64 single-Brotli length-table packing,
   per-pair component traces, archive bytes, and exact CUDA reproduction.
2. Build only charged, contest-closed variants of that contract: deterministic
   repacks, pose residual atoms, renderer/pose/mask packers, and byte-layout
   optimizers. Copying a public archive verbatim is not a scientific result;
   every submitted candidate must have its own provenance and exact eval.
3. Use fast empirical checks to reject out-of-basin candidates before exact
   eval. A candidate that fails the public-floor geometry contract must not be
   promoted just because its archive bytes are small.
4. Use L40S/H100/A100 for triage and T4/equivalent only for promotion-grade
   confirmation. Queue T4 only when formula math plus contract checks make
   `<=0.33` plausible.
5. If rigor must be compressed for wall-clock, sacrifice documentation polish
   and low-value local sweeps first. Never sacrifice charged payload closure,
   exact CUDA score truth, archive SHA/bytes custody, or failure classification.
6. Prefer learned, differentiable, scorer-aligned proposal distributions over
   arbitrary grids when the runtime contract permits it. For pose/byte search,
   gradient-guided or bandit/BO proposal atoms are development-time search
   policies only: the accepted payload must still be charged in the archive,
   the final archive must inflate without scorer access, and the score truth
   remains exact CUDA auth eval of the exact bytes.
7. For public-floor QZS3/QP1/JFG archives, treat pose search as an
   anisotropic manifold problem after scalar-radius gains flatten. The
   preferred order is: accepted scalar checkpoints, sparse/asymmetric
   `--delta-sets`, differentiable `--gradient-delta-sets`, hard-pair temporal
   windows and DCT/spline/jerk modes through `--basis-delta-sets`, then
   charged qpose residual atoms. Every proposal policy is non-promotable until
   it produces a closed archive with exact CUDA auth eval on identical bytes.
   Larger symmetric radii are lower priority once they stop improving the
   rounded archive objective.
8. Public-submission replays that require a public PR's own `inflate.sh` or a
   compatibility adapter are reverse-engineering traces only. They can inform
   mask/pose/model atom design, but they are not our contest evidence and must
   be tagged `external` unless the exact charged archive and fixed submission
   runtime are rebuilt under this repo's contest-faithful payload-closure
   rules. Do not compare a public archive replay directly against our
   leaderboard rank without that attribution boundary.

## Quantizr-Style Five-Stage QAT Protocol

Quantizr's public 0.33 lane demonstrated that five-stage QAT is a competitive
training pattern, but copying the stage names is not enough. In this repo,
five-stage QAT is only production/scientific-grade when each phase is wired to
the same scorer, archive, and packer contracts used at deployment.

Durable requirements for any five-stage QAT or Q-FAITHFUL successor:

- The canonical stage shape is `anchor -> finetune -> joint -> QAT -> final`.
  The stage boundaries, epoch counts, learning rates, seed, batch policy,
  quantization start, and early-stop/kill criteria must be recorded in the
  artifact provenance.
- EMA must run through every stage, including QAT and final consolidation.
  Export and archive builders must declare whether the EMA shadow or live
  weights were packed.
- Training must simulate the scorer input contract: upsample, clamp/round to
  uint8, YUV/resize/downsample behavior, and any pair ordering or mask parity
  used by the inflate path. `eval_roundtrip=True` remains mandatory.
- Pose-conditioning is load-bearing. Any QAT lane with `pose_dim>0` must train
  and export against the exact deployed pose stream or an explicitly recorded
  candidate pose stream; zero-pose fallback is a preflight failure.
- Half-frame Q-FAITHFUL successors with charged zoom/foveation/ego-motion
  geometry must prove both byte preservation and runtime consumption. A
  `zoom_scalars.bin` archive member is not sufficient: the inflate runtime
  must consume it either as renderer `ego_flow` when `use_zoom_flow=True` or
  as a pre-render half-frame mask-expansion warp when the renderer itself has
  no `ego_flow` input. If neither consumed path is proven, exact-screen
  preflight must fail closed with
  `zoom_warp_geometry_not_consumed_by_runtime`.
- Quantization is an optimization variable, not a post-hoc export step. Per-
  tensor scales, grouped bit depth, FP4 codebook choice, stochastic/robust
  scale, entropy model, and packer layout should be treated as atoms with byte
  cost and scorer benefit.
- Hard-pair and water-fill information may enter as Lagrange weights,
  curriculum sampling, atom budgets, or learned selectors. These weights are
  development signals until a concrete archive built from them passes exact
  CUDA auth eval.
- Every QAT snapshot that is harvested for a score attempt must emit a
  deterministic archive through the same `representation -> prediction ->
  quantization -> hyperprior -> arithmetic -> pack` contract as other lanes.
  Byte-only, loss-only, or proxy improvements cannot promote.

The aggressive upgrade over public Quantizr is not one bigger stage. It is a
closed loop: exact component traces and public-floor deltas update QAT sampling
and quantizer allocation; QAT exports update packer/anatomy measurements; exact
archive eval updates the next Lagrangian weights. The loop is valid only when
each feedback artifact is tagged with its evidence grade and provenance.

Lane 12/Alpha NeRV retraining is build-only until explicit L2 clearance.
Production training targets must use decoded baseline archive masks with a
validated `alpha_geo_primitive_contract_v1`; direct `gt_masks_source=segnet`
is forensic/debug only behind an explicit flag. Contract consumption must
validate decoded-mask SHA and shape, record contract SHA and sampling gates,
and preserve weighted sampling provenance for uniform, critical-box,
boundary-band, and transition-endpoint pools. These trainer artifacts are
empirical/no-score until a later canonical CUDA archive eval is run.
Lane 12 remote wrappers must forward full-CUDA training overrides explicitly:
`NERV_STEPS`, `NERV_EVAL_EVERY`, and `NERV_WEIGHT_DTYPE` must reach
`experiments/train_nerv_mask.py` as `--steps`, `--eval-every`, and
`--weight-dtype`, and provenance must record the override values. Do not
dispatch a long NeRV burn through a wrapper that silently falls back to the
shorter profile defaults.

Alpha-Geo-0 stale-pose isolation is a permitted Lane 12/Alpha causal
experiment when it does not retrain the mask codec: keep the exact measured
`masks.nrv`, regenerate `optimized_poses.bin` against the decoded candidate
mask stream, rebuild a deterministic archive, then run CUDA auth eval. Treat
its result as a narrow answer to "stale poses vs incompatible mask geometry";
it does not clear new NeRV retraining unless the L2 clearance packet and three
clean Grand Council passes are recorded.

Alpha sparse-repair archives may replace `masks.mkv` with `grayscale.mkv` plus
an optional `alpha4_residual_repair.amr1` payload, including reviewed compressed
forms `.xz`, `.zlib`, or `.br`. The inflate path must verify decoded-candidate
mask SHA, AMR1 shape, record bounds, and full-repair source SHA when the payload
declares non-partial repair. Archive builders must record repair policy,
selected classes/runs/pixels, compressed and raw SHA/bytes, anchor mask SHA
match, and `score_claim=false` until exact CUDA auth eval adjudication lands.
Any new archive member suffix required by this contract must be admitted in the
canonical auth-eval archive validator and the local smoke whitelist together;
AMR1 suffixes currently allowed are `.amr1`, `.amr1.xz`, `.amr1.zlib`, and
`.amr1.br`. Pair-targeted Alpha repair policies select frames `2*i` and
`2*i+1` for each absolute contest pair index `i`; they must record the source
of the pair list and remain non-promotable until the resulting archive has its
own exact CUDA auth eval.

Predictive charged mask grammar (`CMG2`-class) candidates must pass a
byte-screen and runtime-contract screen before exact-eval spend. A manifest
must compare the candidate against the currently charged mask stream bytes and
record decoded tensor SHA, transform/decoder schema, payload SHA, payload
bytes, and `score_claim=false`. Generic lossless tensor wrappers are planning
signals only unless they beat the charged mask stream and ship with a reviewed
runtime decoder envelope. Byte-regressive lossless probes must not be promoted
to CUDA exact eval; move the main effort to predictive/lossy/scorer-weighted
grammar or learned decoder atoms instead.

CDO1 decoded-mask overlay sidecars (`masks.cdo1`, `.zlib`, `.xz`, `.br`) are
charged overlay payloads, not standalone mask streams. The inflate runtime must
decode the selected base mask payload first, verify
`base_mask_tensor_sha256`, apply sorted non-overlapping overlay runs, preserve
`_half_frame_only` metadata, verify `reconstructed_mask_u8_sha256`, and log
that CDO1 was applied. Builders must record raw/compressed payload SHA/bytes,
base archive SHA, output archive SHA/bytes, and `score_claim=false`. A CDO1
payload or spec is non-promotable until the byte-closed archive gets exact CUDA
auth eval; no-op overlays and spec-only artifacts are method evidence only for
runtime/debugging, not score evidence.

CDO1 remote dispatch also requires a joint byte+geometry preflight, not just a
small overlay payload. Before exact-eval spend, record a planner artifact that
prices both the candidate archive bytes and the residual decoded-mask
disagreement after overlay. Byte-headroom candidates that still fail the
residual geometry gate are lower-bound/economics evidence only; do not dispatch
them unless a specific diagnostic override is recorded in the lane claim.
CDO1 archive builders must support the deployed packed single-member `p`
container, not only expanded `renderer.bin`/`masks.*` archives. A CDO1
candidate that starts from a packed frontier archive must unpack through the
reviewed runtime unpacker, add the charged overlay as a logical runtime member,
and optionally re-emit a single packed payload with recorded payload format,
member name, Brotli settings, expanded-candidate SHA, and final archive SHA.
Tests must cover this packed round trip before any packed CDO1 dispatch.

SJ-KL residual candidates may add an optional charged `sjkl.bin` archive
member to JointFrameGenerator/QZS/QFAI-style renderer archives. The runtime
contract is narrow: `sjkl.bin` contains only packed basis and coefficient
bytes, the inflate path must not load SegNet/PoseNet or scorer modules, and
the payload is applied only inside the q-faithful JointFrameGenerator pair
path. Non-JointFrameGenerator renderers, shape mismatches, absent payloads,
or invalid payload headers must skip/fail closed as appropriate and must not
silently create a score claim. Builders must record basis SHA, coefficient
SHA, coefficient quantization, charged bytes, target frame shape, source
renderer-output SHA, GT/residual tensor SHA, and `score_claim=false` until
the concrete archive receives exact CUDA auth eval. Any SJ-KL exact-eval
dispatch must set `SJKL_REQUIRE_APPLIED=1` unless it is explicitly labelled
forensic/no-op-control; charged `sjkl.bin` bytes that do not affect at least
one renderer pair are a harness failure, not score evidence.

C067-era JointFrameGenerator renderer compression lanes must anchor on the
exact public-floor-style packed runtime contract, not on older Lane G/ASYM
renderer assumptions. IMP, QZS reblocking, Block-FP, or self-compression
builders for this family must first prove deterministic unpack/repack parity
against the current packed payload, record logical member SHA/bytes, and emit
`score_claim=false` archives with exact archive SHA/bytes before any remote
spend. Byte-screen winners are empirical only. L40S/H100 diagnostics may test
distortion quickly, but any frontier or leaderboard claim requires T4/equivalent
CUDA confirmation on identical archive bytes. Remote IMP bridge scripts are
build/byte-screen helpers only unless the emitted archive is separately routed
through the canonical exact CUDA auth-eval path with source manifests and
adjudication.

## Backend Routing

- Lightning AI is the preferred exact-eval home for T4/equivalent
  promotion-grade runs. Use hermetic staged trees and preserve source manifests.
- Modal is approved for build-only, smoke, Fisher/sensitivity, ablation, and
  cheap exploratory work. Modal auth evals are advisory unless the wrapper
  proves CUDA exact eval and all contest gates.
- Modal exact-eval wrappers must call `experiments/contest_auth_eval.py` with
  literal `--device cuda`; they must not call `inflate_renderer.py` directly,
  must not silently fall back to CPU, and must mark harvested artifacts
  non-promotable until JSON adjudication validates custody, device, sample
  count, SHA, bytes, and component gates.
- Vast.AI is useful for cheap parallel training but has host/NVDEC and state
  drift hazards. Trust lane-local artifacts and live API over stale trackers.
- Vast training/output lanes must be single-flight per lane label and output
  directory. A duplicate concurrent trainer writing the same artifact tree
  invalidates custody for that run; harvest logs for forensics, terminate the
  duplicate work, and rerun behind an explicit lock before spending exact eval.
- Single-GPU diagnostic loops on warm Vast/H100 boxes must also be
  single-flight by process signature and output directory. Before launching an
  ad-hoc pose/search/eval process, run `pgrep -af <script-or-output-tag>` and
  inspect `tmux ls` or the lane-local PID/log file. The launch must create the
  output directory and first-byte stdout/stderr log before backgrounding. If a
  duplicate process is found, keep the highest-EV active run, terminate only
  the lower-priority duplicate, and mark the stale dispatch claim
  `failed_<reason>` or `cancelled_duplicate_claim`.
- Local M-series/MPS work is for development, smoke, and byte/round-trip checks
  only. It cannot rank or promote.
- Do not trust remote state ledgers without reconciliation. Use
  `scripts/reconcile_vast_dispatch_state.py` for non-mutating drift reports.
- No new retraining lane should be dispatched before Lane 12/Alpha has an
  explicit L2 unblock packet at `.omx/state/lane12_nerv_l2_clearance.json`.
  Build-only, harvest, and exact-eval-only lanes may continue. Retraining
  dispatches must fail closed unless the packet records
  `cleared_for_retraining_unblock=true`, `lane12_l2=true`,
  `geometry_gate_passed=true`, `grand_council_clean_passes>=3`, and evidence
  paths. The Vast retry launcher enforces this gate.

## Fast Chip Preference — Wall-Clock Optimization Beats $/hr

Time-to-Shannon-floor is the optimization target, not $/hr. Per the user's
2026-05-01 directive ("make sure we are using fast chips because we don't
have time to waste waiting for results"), prefer the fastest available chip
within a 2-minute boot window. $5-10/hr is acceptable when it removes 30+
minutes of wait time.

Every score-affecting experiment matters. New diagnostic exact-eval,
component-response, trace, sweep, or build/eval loop work must use the fastest
available verified CUDA hardware by default. Use T4 only for contest-equivalent
A++ promotion or when the experiment specifically tests T4 runtime behavior.
If a non-T4 diagnostic is in the public-floor band, queue T4 confirmation on
the identical archive bytes immediately and keep the fast-chip loop moving.

Default chip ranking (fastest first), measured by contest-CUDA archive
inflate + evaluate end-to-end on a single owv3-class candidate:

- **H100 SXM (80GB)**: ~5-8× T4 throughput, ~$1.80/hr Vast.ai. New default
  for time-critical chains, sub-frontier sweeps, and any newly-spawned
  iteration loop.
- **H200**: ~6-10× T4, ~$2.50+/hr. Reserve for heaviest training (NeRV mask
  codec, IMP cycles, Joint-ADMM long iterations).
- **A100 SXM4 (40-80GB)**: ~3-4× T4 + extra VRAM, $0.80-1.50/hr. Use when
  VRAM > 24GB needed (Q-FAITHFUL OOM'd on RTX 4090).
- **RTX 5090**: ~4-5× T4, $0.50-1.00/hr. Acceptable when H100/A100 supply
  is tight.
- **RTX 4090**: ~3-4× T4, $0.25-0.30/hr. Use when warm-venv reuse on an
  already-bootstrapped instance amortizes the slower chip cost.
- **Modal A10G**: $0.59/hr, ~2× T4. Use only if Vast.ai supply is
  exhausted or a Modal-specific lane (e.g., `modal_train_lane.py`) requires
  it.
- **T4 / Lightning T4**: 1.0× baseline. ONLY for the final A++ promotion
  run on a deploy candidate (per the contest's T4-equivalent grading).
  NEVER for iteration.

Canonical Vast.ai search filter for new dispatches:

```bash
.venv/bin/vastai search offers \
    'gpu_name in [H100_SXM,H100_NVL,H100_PCIE,H200] reliability>0.95 disk_space>=80 num_gpus=1 dph<3.0' \
    -o 'dph'
```

Fallback to A100 / RTX 5090 / RTX 4090 only if no H100 within budget. The
permanent fix in `scripts/launch_lane_on_vastai.py` should grow a
`--prefer-fast-chip` flag that walks this list automatically; until then,
operators specify the chip explicitly in `vastai create instance`.

Cost discipline: do NOT parallelize on multiple slow chips when one fast
chip clears the queue faster. Idle a slow instance to free up budget for
a fast one. Memory:
`feedback_fast_chip_directive_no_waiting_20260501.md`.

## Remote Bootstrap Canonicalization And Reuse

The 2026-05-01 loop session burned 4 destroyed Vast.ai instances and ~$1.50
chasing 6 sequential bug-class re-discoveries: uv missing, ffmpeg missing,
macOS resource forks in `upstream/`, `.venv` with no pip, system ffmpeg
lacking `in_primaries` scale option, and the cu13-vs-cu124 torch wheel
mismatch. The root cause was duplication: each new chain driver re-implemented
the same install logic inline instead of reusing the canonical wrapper. To
prevent re-learning these lessons, the rules below are now non-negotiable.

- The single canonical bootstrap function is `bootstrap_runtime_deps()` in
  `scripts/remote_archive_only_eval.sh`. It delegates uv install to
  `scripts/ensure_remote_uv.sh`, apt-installs ffmpeg, strips macOS `._*`
  resource forks under `upstream/`, and auto-downloads the BtbN ffmpeg static
  build (with retry on truncated downloads) when the system ffmpeg lacks
  `in_primaries`/`in_color_matrix`/`in_transfer` scale options required by
  `submissions/robust_current/inflate.sh`.
- `scripts/remote_archive_only_eval.sh` must also bootstrap scorer-runtime
  dependencies in the Python interpreter that runs `upstream/evaluate.py`
  (`timm`, `einops`, `segmentation-models-pytorch`, `safetensors`, `av`,
  `tqdm`) and record a probe JSON. Bare CUDA images often have Torch but not
  the scorer stack; failing after archive inflate is avoidable wall-clock loss.
- Compress-time proposal tools that import `upstream/modules.py` directly
  (for example `experiments/line_search_pose_refinement.py`) must run an
  explicit scorer-runtime dependency preflight before importing the upstream
  module, touching `DaliVideoDataset`, or launching paid remote work. Missing
  `timm`, `einops`, `segmentation-models-pytorch`, `safetensors`, or
  `nvidia.dali` is a
  `failed_scorer_runtime_deps` preflight failure, not lane evidence. Do not
  paper over this with ad-hoc imports; install the runtime extra and run
  `scripts/bootstrap_dali_hash_pinned.py` in the runner interpreter so DALI is
  direct-wheel, hash-pinned, and recorded before any GPU spend.
- Compress-time proposal tools that decode the contest video through
  `upstream/frame_utils.DaliVideoDataset` must also preflight
  `nvidia.dali` in the same runner interpreter before paid remote work.
  Missing DALI is a `failed_dali_runtime_deps` preflight failure. Install the
  driver-compatible wheel (`nvidia-dali-cuda130` for CUDA 13 images,
  `nvidia-dali-cuda120` for CUDA 12 images) and record a probe JSON before
  relaunch.
- Archive-only exact-eval chains must clean heavy inflated/eval work after
  preserving canonical `contest_auth_eval.json`, provenance, report, logs, and
  runtime-tooling metadata. They must also remove the per-job
  `UV_PROJECT_ENVIRONMENT` when it lives under the job log dir; this directory
  is reproducible dependency state, not scientific custody. Use an explicit
  keep flag only when a later component trace or forensic inspection requires
  the raw inflated directory.
- Archive-only exact-eval chains must also mirror the exact evaluated
  `archive.zip` into the result directory and write `archive_custody.json`
  with SHA-256 and byte count before invoking the scorer. A score JSON whose
  archive path can be overwritten later is diagnostic-only until custody is
  re-established from a preserved archive copy.
- Archive-only exact-eval chains that intentionally evaluate a non-default
  inflate script must pass that path through the runner's explicit
  `INFLATE_SH`/`--inflate-sh` control, resolve it inside the repo, record its
  SHA-256 in provenance, and fail closed on unsafe or missing paths. The
  default remains `submissions/robust_current/inflate.sh`; public-adapter
  traces are external reverse-engineering evidence unless rebuilt as our own
  charged runtime.
- Every new `scripts/remote_lane_*.sh`, every chain driver, and every ad-hoc
  one-off MUST either invoke `scripts/remote_archive_only_eval.sh` directly or
  `source` its bootstrap function. Inline duplication of `curl ... | sh`,
  `apt-get install -y ffmpeg`, or `find upstream -name '._*' -delete` is
  forbidden. Memory:
  `feedback_remote_archive_only_eval_self_bootstraps_all_deps_20260501.md`.
- `scripts/ensure_remote_uv.sh` is the only sanctioned uv installer. Sister
  pattern for venvs that lack pip: call `python -m ensurepip --upgrade`
  immediately after any `python -m venv` / `uv venv` / `virtualenv` invocation.
  This was the root of the `/workspace/pact/.venv/bin/python: No module named
  pip` failure that broke `scripts/probe_nvdec.sh` on instance 35958897.
- The torch wheel must be pinned by the host driver before any
  `uv run --with torch ...` invocation. The canonical selector lives in
  `scripts/remote_archive_only_eval.sh:88-95`:
  - `nvidia-smi` driver_major < 580 (CUDA 12.x): export
    `INFLATE_TORCH_SPEC=torch==2.5.1+cu124`,
    `UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124`,
    `UV_INDEX_STRATEGY=unsafe-best-match`.
  - driver_major >= 580 (CUDA 13.x): the default `torch==2.11.0` (cu13 wheel)
    works without an extra index.

  Writing `--with torch` unpinned anywhere is forbidden because the cu13 wheel
  silently fails `torch.cuda.init()` on a CUDA 12.x driver and the inflate
  pipeline falls back to CPU, which makes every downstream score
  `[advisory only]` per the MPS-falsification rule. Memory:
  `feedback_vast_cuda_driver_too_old_silent_cpu_fallback_20260501.md`.
- Vast.ai chain dispatches must use `--disk 60` minimum. The contest auth
  eval pipeline writes 3.6GB of inflated raw frames per candidate; a
  six-candidate chain needs ~27GB working set on top of the ~5GB uv torch
  cache. The 30GB default fills before the chain finishes and crashes the
  trailing candidates. The chain driver MUST also
  `rm -rf eval_work/{inflated,extracted,archive.zip}` after every successful
  evaluation. Reference driver:
  `experiments/results/lane_g_v3_owv3_wave3_refinement_20260501/wave3_chain_driver.sh`.
- Vast.ai search filters must include `cuda_vers>=12.4`, `disk_space>=60`,
  `reliability>0.95`, and a unique `--label`. Every successful create must
  immediately register the instance ID into
  `.omx/state/vastai_active_instances.json` so cleanup, harvest, and audit
  scripts can reconcile drift.

When a new bug class appears in this surface area, do not patch it inline in
the failing script. Add the fix to `scripts/remote_archive_only_eval.sh` (or
`scripts/ensure_remote_uv.sh`) and update this section so the next agent
inherits the lesson. "Capture the meta-pattern in memory + AGENTS.md the
FIRST time it bites" — the same rule that already applies to the
dead-flag-wiring trap applies here.

## Lightning And PyTorch Lightning Guidance

Use Lightning AI infrastructure aggressively, but keep the contest path
deterministic.

- Use `scripts/lightning_repro_workspace.py` for Lightning Studio staging.
  One-off `rsync` is acceptable only for emergency debugging and must be
  superseded by a source/artifact manifest before promotion. Generated payloads
  must be passed with explicit `--artifact`; bulky experiment outputs,
  checkpoints, videos, and archives are not source.
- Use `scripts/configure_lightning_ssh.py` to install reproducible Lightning
  SSH aliases on new operator machines. The managed alias uses BatchMode,
  public-key-only auth, bounded connect attempts, SSH keepalives,
  opportunistic ControlMaster reuse, `StrictHostKeyChecking accept-new`, and a
  dedicated known-hosts file. Do not use Lightning UI helper output that
  disables host-key checking or sends known-hosts to `/dev/null` for contest
  custody work.
- The `cloud` optional dependency extra owns Lightning SDK installation for
  reproducible operator environments. Use `uv sync --locked --extra cloud`
  when a fresh machine needs Lightning automation tools; continue to forbid
  the PyPI package named `lightning`.
- Use `scripts/lightning_repro_workspace.py --ssh-check-only --require-cuda`
  before interactive Lightning CUDA work. Plain SSH success only proves the
  Studio is reachable; it does not prove a GPU is attached. If this runtime
  probe reports `torch_cuda_available=false`, do not run or promote
  interactive CUDA work from that Studio shell. Batch Jobs may still run on
  requested GPU machines, but their own `lightning_runner_preflight.json` is
  the authority.
- A Lightning staged tree must preserve a local and remote manifest with file
  count, bytes, SHA-256, source/artifact role, git status, command, and
  environment JSON. Exact eval jobs must cite that manifest in provenance.
- Before non-dry-run Lightning submit, run
  `scripts/launch_lightning_batch_job.py doctor --ssh-target <alias>
  --require-ssh --require-remote-supply-chain --require-machine-inventory`
  and preserve its JSON. Doctor passing is not score evidence, but a failed
  doctor blocks dispatch.
- Component-response perturbation plans used for remote Batch Jobs must not
  contain host-local absolute paths in point archives or per-point eval JSON.
  Plan paths must be relative to the plan file, and the staged source manifest
  must include every resolved file. If a plan contains stale top-level
  `baseline_contest_auth_eval_json`, the explicit CLI
  `--baseline-contest-auth-eval-json` path is the authority and must point
  inside the remote repo.
- Remote Lightning `uv sync` must use copy-mode installs:
  `UV_LINK_MODE=${UV_LINK_MODE:-copy} uv sync --locked --extra runtime`.
  Studio filesystems can fail hardlink installs while materializing Torch; this
  is a permanent DX guard, not an ad hoc workaround.
- Exact-eval Batch Jobs must isolate inflate-side `uv run` environments from
  the scorer environment. Export a per-job `UV_PROJECT_ENVIRONMENT` under the
  job output dir before `contest_auth_eval.py`; otherwise `inflate.sh` can
  recreate the shared repo `.venv` and break `upstream/evaluate.py`.
- Exact-eval Studio submissions must be source-manifest closed over the
  inflate runtime, not only the archive. For `submissions/robust_current`, the
  staged manifest must include both `submissions/robust_current/inflate.sh`
  and its sibling `submissions/robust_current/config.env`; missing
  `config.env` is a submit-time blocker because `contest_auth_eval.py` will
  fail closed before score evidence exists.
- Promotion-sensitive exact evals should also gate the recorded
  `inflate_runtime_manifest.runtime_tree_sha256` by passing
  `expected_runtime_tree_sha256` through queue metadata. Identical archive
  bytes evaluated under different runtime helpers are runtime-custody
  comparisons, not interchangeable promotion evidence.
- T4/g4dn exact-eval submissions must explicitly pin inflate-side Torch for
  the host driver. For CUDA-12 era T4 workers, pass
  `INFLATE_TORCH_SPEC=torch==2.5.1+cu124`,
  `UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124`, and
  `UV_INDEX_STRATEGY=unsafe-best-match` in recorded job env. Do not rely on
  resolver defaults for promotion-grade runs.
- Inflate-side `uv run --with ...` dependencies must be deterministic. Do not
  leave `brotli`, `av`, `torch`, or `numpy` as floating resolver inputs in
  contest runtime scripts. `submissions/robust_current/inflate.sh` exposes
  `INFLATE_BROTLI_SPEC`, `INFLATE_AV_SPEC`, `INFLATE_TORCH_SPEC`, and
  `INFLATE_NUMPY_SPEC`; runners may override these only with recorded
  provenance. Vast/other older-driver archive-only evals must choose a
  driver-compatible Torch spec before inflate, rather than silently falling
  back to CPU rendering from an incompatible latest wheel.
- Exact-eval Batch Jobs that mutate/check the shared `.venv` for DALI/bootstrap
  must hold `.omx/state/lightning_exact_eval_venv.lock` while doing so. Parallel
  jobs may run the scorer concurrently only after this setup phase is complete.
- Lightning Batch Job artifact paths must be derived from
  `lightning_sdk_job_name()` or an explicit output directory. The SDK normalizes
  underscores to hyphens for `/teamspace/jobs/<job>/artifacts`; do not
  hand-compose artifact paths from the local queue name.
- Exact/component/sensitivity Batch submissions that target non-default
  Lightning cloud accounts must pass the explicit `--cloud-account` through
  to the SDK job run. Machine inventory on one cloud account is not evidence
  that the same accelerator slug exists on the account selected by `Job.run`;
  failed machine-name retries are a pre-spend routing bug.
- Studio-backed Lightning Batch Jobs can only run on the cloud account attached
  to that Studio namespace. `list-machines --cloud-account X` proving H100/L40S
  capacity on account X is not dispatch authority unless the selected Studio is
  also on account X. Before claiming/submitting a cross-account accelerator
  hedge, either use a Studio that already belongs to that cloud account, switch
  to a reviewed image-backed/non-Studio submit path, or record a terminal
  `failed_predispatch_cloud_account_mismatch` claim. Do not spend queue time
  retrying a Studio/cloud-account mismatch after the SDK reports it.
- `scripts/launch_lightning_batch_job.py refresh-status` should be run with
  only `--state-path` and `--job-name` when the job was queued locally; it
  infers the SDK job name, teamspace, org, and user from the state record to
  avoid operator drift.
- Lightning SDK status strings are telemetry, not standalone custody. If
  refresh history shows a nonterminal regression such as `Running -> Pending`,
  the local state must record `status_anomalies`,
  `status_reconciliation_required=true`, and full per-refresh snapshots. For
  non-dry-run exact/component-response/sensitivity jobs, nonterminal status
  regressions must fail closed as `REMOTE_STATUS_RECONCILIATION_REQUIRED`
  unless a terminal SDK status supersedes them; terminal artifacts still need
  harvest validation before scientific use.
- Lightning refreshes that only resolve jobs by name must record
  `identity_confidence=name_only` and
  `identity_reconciliation_required=true`. Prefer stable SDK job ids whenever
  the SDK exposes them; null-id name-only refreshes are not enough for
  promotion custody without state-derived artifact validation.
- Non-dry-run Studio-backed Lightning Batch submissions must use
  `--remote-preflight-ssh-target <alias>` unless a specific auditable
  break-glass reason is recorded. This runs
  `scripts/scan_lightning_supply_chain.py --quiet --strict` on the remote
  Studio tree immediately before SDK submission, so stale snapshots with
  compromised `lightning` CLI wrappers fail before spending GPU time.
- Lightning staging and harvest SSH/SCP/rsync operations must use noninteractive
  auth policy: `BatchMode=yes`, password and keyboard-interactive auth
  disabled, and an explicit `ConnectTimeout`. Preflight-only SSH hardening is
  insufficient if the actual copy/harvest commands do not reuse the same
  policy.
- Lightning SSH is transport only. It cannot keep a Studio GPU attached after
  Lightning's machine policy switches the Studio back to CPU. Interactive SSH
  work must probe `nvidia-smi` and `torch.cuda.is_available()` before CUDA
  assumptions; promotion-grade CUDA should prefer Batch Jobs with explicit
  machine selection and per-job runner preflight.
- Studio-backed Lightning jobs persist Studio paths under the SDK artifact
  mirror, e.g. `/teamspace/jobs/<job>/artifacts/pact/...`. Use
  `scripts/launch_lightning_batch_job.py harvest-ssh --job-name ...` for
  terminal harvests; it derives the persisted path from the recorded
  `remote_output_dir`, copies only canonical evidence files, and validates
  archive/adjudication JSON locally. Do not `scp -r` whole eval directories;
  they can contain multi-GB raw frames and break custody.
- Custom score-producing Lightning jobs must use a specific fail-closed role,
  not generic tracking. Alpha-Geo-0 pose-regeneration jobs use
  `scripts/launch_lightning_alpha_geo0_pose_regen.py`, which records role
  `alpha_geo0_exact_eval`, source/artifact manifest custody, remote
  supply-chain preflight, hash-pinned DALI bootstrap, CUDA runner preflight,
  canonical auth eval, and exact adjudication metadata. The final archive SHA
  and bytes are unknown until pose regeneration finishes, so write them into
  `lightning_queue_metadata.json` after deterministic archive construction and
  before harvest validation.
- Official component-response harvests must also be state-derived:
  `scripts/launch_lightning_batch_job.py harvest-component-response-ssh
  --state-path .omx/state/lightning_batch_jobs.json --job-name <job> ...`.
  The wrapper maps recorded Studio output dirs into SDK artifact mirrors and
  validates compact response evidence locally. Do not hand-compose
  `/teamspace/jobs/...` paths for promotion-grade claims.
- Diagnostic component-sensitivity harvests must also be state-derived:
  `scripts/launch_lightning_batch_job.py harvest-component-sensitivity-ssh
  --state-path .omx/state/lightning_batch_jobs.json --job-name <job> ...`.
  These artifacts are non-promotable unless later assembled into a reviewed
  `component_sensitivity_v1` packet through the official CUDA component
  response path.
- After a diagnostic CUDA component-sensitivity harvest, use
  `experiments/build_component_response_plan_from_sensitivity_artifacts.py`
  to validate the harvested artifact directory, build pre-response
  `official_component_response_prediction_deltas_v1`, and emit the deterministic
  official response plan. This remains planning evidence only; score signal
  still requires the subsequent official CUDA component-response Batch Job with
  same-run eps=0 and `--require-passed`.
- Generated Lightning Batch commands must emit option values that may begin
  with `-` using the `--flag=value` form, not `--flag value`. This is required
  for epsilon ladders such as `--response-epsilons=-0.002,...`; otherwise
  argparse can treat the negative value as another option and fail remotely.
- Supply-chain rule: do not install the PyPI package named `lightning` in this
  repo or on remote runners. On 2026-04-30, `lightning==2.6.2` and
  `lightning==2.6.3` were reported compromised with import-time credential
  theft. Use `lightning-sdk` for Lightning AI Batch Jobs/CLI work.
- Run `scripts/scan_lightning_supply_chain.py --strict` before trusting a new
  local or remote runner for exact eval. Preserve the JSON output under
  `.omx/state/` with the runner/date in the filename.
- The Lightning supply-chain scanner must stay current with incident IOCs,
  including reported `router_runtime.js`, `_runtime/start.py`,
  malicious `lightning/__init__.py`, and `lightning` wheel hashes, plus pip/uv
  cache scans for cached `lightning` 2.6.2/2.6.3 artifacts.
- Do not execute `lightning` CLI probes just to discover installation state.
  Inspect package metadata or known console-script targets instead. A poisoned
  `lightning` executable on `PATH` can be an import-time trigger.
- Do not call `.venv/bin/lightning`, bare `lightning`, or `$LIGHTNING` from
  operator scripts. Use SSH-backed wrappers or `lightning-sdk` APIs. The strict
  supply-chain preflight scans `scripts/`, `tools/`, and Lightning deploy
  helpers for these stale console-script paths.
- Treat any environment that installed and imported `lightning==2.6.2` or
  `lightning==2.6.3` as compromised until isolated and credentials are rotated.
- Scan for Mini Shai-Hulud indicators before trusting a runner or repo:
  `.claude/router_runtime.js`, `.claude/setup.mjs`, `.vscode/setup.mjs`,
  `.github/workflows/format-check.yml`, hidden `lightning/_runtime/`, and
  npm `postinstall` hooks that run `setup.mjs`.
- Borrow Lightning Fabric patterns selectively for optional training-lane
  wrappers, seed/rank-zero discipline, callback organization, and optional
  training-state checkpoints.
- Treat Lightning-AI ecosystem repos such as LitModels, lightning-thunder, and
  utilities as research inputs, not promotion dependencies, until they pass the
  local supply-chain scanner, deterministic replay checks, CUDA parity checks,
  and import audit. Copy/adapt small, audited patterns where useful; do not add
  broad dependencies or cloud model-registry custody to contest artifacts.
- LitModels may inform checkpoint/registry ergonomics, but its optional
  Lightning/PyTorch-Lightning integrations are outside promotion environments.
  Do not install extras that pull the PyPI `lightning` package.
- lightning-thunder may be evaluated only behind opt-in profiling/training
  flags. It is not allowed in canonical exact eval or score custody until
  numerical parity, deterministic behavior, compile-cache effects, and CUDA
  runtime provenance are adversarially audited.
- lightning-utilities patterns such as rank-zero logging, import/version
  helpers, and dependency CLI checks may be adapted if they reduce local code
  fragility. Keep local wrappers independent of `lightning` imports.
- When adapting `lightning-utilities` import helpers, do not point any helper
  that imports the target module at `lightning`; use metadata-only package
  inspection for high-risk names. Avoid broad requirement-rewrite helpers for
  Pact because `uv.lock`, upper bounds, and reviewed dependency custody are
  authoritative.
- Do not migrate canonical archive construction or exact eval into a full
  PyTorch Lightning `Trainer` loop.
- Avoid DDP for exact eval unless sampler and aggregation are audited.
- Avoid remote loggers/artifact managers in canonical result custody.
- Keep authoritative artifacts as local JSON, logs, manifests, `.pt` files, and
  ZIP archives harvested into `experiments/results/`.

## Component Sensitivity And OWV3

- Promotion-grade sensitivity artifacts must be CUDA-authored and must separate
  PoseNet, SegNet, and combined scorer signal. A single proxy/Fisher tensor is
  not enough for paper or deployment claims unless the component breakdown,
  calibration split, holdout stability, and response-curve validation are
  recorded.
- The target schema is `component_sensitivity_v1`: manifest, PoseNet map,
  SegNet map, combined map, per-pair metrics, perturbation response curves,
  command, environment, source manifest, input hashes, sample plan, stability
  metrics, and optional exact eval custody.
- Use `experiments/build_component_sensitivity_manifest.py` to assemble
  `component_sensitivity_v1` packets from real CUDA maps, response curves,
  exact eval JSON, and archive artifacts. Do not hand-edit promotable
  sensitivity manifests.
- Current `experiments/profile_component_sensitivity.py` output is
  diagnostic Fisher-proxy evidence only, even on CUDA. It may produce maps,
  response-curve JSON, stability JSON, and sample plans for design/debugging,
  but it deliberately records `promotion_eligible=false` and blocks
  `--manifest-output`. Do not use it as promotion-grade
  `component_sensitivity_v1` evidence until official finite-difference
  component response validation, symmetric/directional response curves, and
  CUDA exact-eval custody are implemented and reviewed. CPU runs additionally
  require `--allow-diagnostic-cpu` and are non-promotable.
- Lightning diagnostic component-sensitivity validation must inspect every map,
  response curve, summary, input-preflight, and run-metadata artifact for
  `score_claim=false`, `promotion_eligible=false`, allowed
  `sensitivity_source`, non-official response status, and
  `canonical_scorer_path=false`. Direct renderer CUDA finite-difference maps
  may be `planning_eligible` and `certification_handoff_eligible`, but remain
  non-promotable until certified with official CUDA component-response
  evidence. Fisher/proxy maps are planning-only and never certification handoff
  eligible.
- Direct renderer finite-difference sensitivity maps are weight-space channel
  response maps. They may rank channels and select perturbation atoms, but
  they must not be projected into archive-byte `predicted_delta` values unless
  a reviewed byte-basis calibration marks the maps
  `archive_byte_prediction_eligible=true`. Response-only archive plans without
  prediction deltas are allowed for calibration/diagnosis, but they are not
  promotion-passed certification evidence.
- Promotion-grade component maps must pass a separate certification stage; do
  not edit or strip diagnostic metadata from source maps. Use
  `experiments/certify_component_sensitivity_maps.py` to copy eligible CUDA
  direct finite-difference tensors into new `tac_score_sensitivity_map_v1`
  files with `component_sensitivity_map_certification_v1` metadata. The
  certification must cite source-map SHA, official response-curve SHA,
  stability SHA, sample-plan SHA, baseline archive SHA/bytes, baseline
  `contest_auth_eval.json` SHA, pre-response prediction-deltas SHA,
  archive-byte perturbation-basis SHA, response gate metrics, stability gate
  metrics, and at least three clean review passes. Fisher/proxy/debug/smoke/
  random maps are never certifiable.
- `experiments/build_component_sensitivity_manifest.py` promotion assembly
  must reject raw diagnostic maps and clean-but-uncertified maps. A promotable
  manifest may only reference certified maps plus official CUDA response curves
  with all promotion gates passed.
- Official component-response jobs that are given an external baseline
  `contest_auth_eval.json` must compare the same-run eps=0 baseline to that
  external JSON. External-baseline component drift is a runner/scorer
  calibration failure and blocks promotion even if same-run zero reproduces
  internally. Local component-response artifact validation and map
  certification must reject or de-promote curves that omit or fail
  `gate_results.external_baseline_repro` when an external baseline was
  supplied.
- Official component-response packets may be non-promotable as calibration
  packets while still containing exact point archives with valid CUDA
  `contest_auth_eval.json` custody. A point archive may be separately
  adjudicated and ranked only when its original archive bytes/SHA, exact
  canonical CUDA eval JSON, provenance, sample count, device, component gates,
  and source plan are preserved. Do not promote the parent response packet
  itself unless its own promotion gates pass.
- Component sensitivity sample plans must identify absolute dataset pair IDs.
  If top-k pair weighting selects a subset, do not record subset-relative
  offsets in calibration/holdout records.
- Fake, random, dummy, CPU, MPS, smoke, debug, proxy-only, or no-holdout
  sensitivity artifacts are non-promotable. They may guide debugging only.
- Official component-response promotion plans must carry pre-response
  prediction deltas from
  `experiments/build_component_response_prediction_deltas.py` in
  `official_component_response_prediction_deltas_v1` format. Ad hoc epsilon
  maps, post-hoc observed-response deltas, copied scorer JSON, or any payload
  containing response-curve/eval leakage are non-promotable and must fail
  closed when `--require-predicted-deltas` is set.
- `experiments/profile_component_sensitivity_official.py --require-passed`
  must use a same-run eps=0 baseline. External baseline JSON may be retained
  as archive custody only; it cannot satisfy zero-repro gates or absorb
  runtime/scorer drift.
- Promotable component-response curves must include explicit finite
  `gate_results` with coverage, same-run zero repro, signal, prediction-error,
  and promotion gates all exactly true, and no nested promotion blockers.
  `experiments/build_component_sensitivity_manifest.py` must preserve those
  gates and `src/tac/component_sensitivity_artifact.py` must reject missing or
  false gates.
- OWV3 Fisher profiling for promotion must include protected Conv2d weights
  with `--include-protected-conv2d`; protected Linear FiLM parameters remain
  excluded unless a new reviewed converter supports them.
- OWV3 Fisher conversion must use `--missing-policy error`. Any protected or
  nonprotected missing Conv2d sensitivity key blocks promotion. The legacy
  protected-missing fallback is smoke/debug only.
- Do not spend exact eval on an OWV3 archive that is larger than the PFP16 A++
  byte frontier unless an exact distortion-reduction justification and review
  tag are present.

## Neural Weight Codec Lanes

- J-NWC and J-NWCS artifacts must be loadable renderer formats, not ad hoc
  concatenated tensor blobs. File magic, schema version, JSON header,
  embedded codec state or explicit loader contract, tensor metadata,
  length-prefixed blobs, and inflate dispatch must be tested end to end before
  promotion.
- NWC/NWCS training seed contracts include codec construction. Set the torch
  seed before constructing codec modules, not only inside the training sampler.
- Corpus manifests must be deterministic and relocatable. Replay may use a
  manifest-relative root, but must always recheck file size, SHA-256, tensor
  shape, dtype, block count, and ordering.
- NWCS sensitivity artifacts must be provenance-anchored for promotion:
  anchor archive SHA-256, anchor renderer SHA-256, corpus manifest SHA-256,
  block size, parameter names, shapes, block counts, and nonnegative finite
  values. Raw shape-only sensitivity dictionaries are debug-only.
- J-NWC/NWCS remote scripts must use zip-slip-safe archive reads, reject hidden
  sidecars, duplicates, absolute paths, traversal, and unexpected members, and
  record SHA-256/bytes for every custody artifact.
- J-NWC/NWCS exact CUDA paths must run `scripts/adjudicate_contest_auth_eval.py`
  after `contest_auth_eval.py`, preserve adjudication provenance and
  adjudicated JSON, and configure component gates against the active frontier.
  Build-only/debug paths must stop before auth eval with `score_claim=false`,
  `promotion_eligible=false`, `auth_eval_skipped=true`, and `result_json=null`.
- NWCS `NWCS1` export heredocs must import and use
  `_infer_asymmetric_config` in the same Python process that writes the
  container metadata. A fallback to `{"tensor_only": true}` is non-promotable.
- `AUTH_EVAL_DEVICE` must be `cuda` for promotable J-NWC/NWCS runs. CPU/MPS
  overrides must fail closed or mark the run explicitly non-promotable before
  any result can be harvested.

## Loss And Training Guardrails

- Primary KL distillation is forbidden for promotion paths unless explicitly
  fenced as forensic. SegNet-only auxiliary KL must be explicitly scoped,
  temperature-plumbed, and promotion-gated by exact PoseNet non-collapse.
- Renderer-training KL/JBL auxiliaries must never activate from
  `kl_distill_weight` alone. Positive `kl_distill_weight` requires
  `kl_distill_scope="segnet_aux"` in CLI/profile, and `primary_scorer` scope
  is blocked in `train_renderer`.
- Legacy `loss_mode="segnet_kl"` is forensic/debug only unless separately
  revalidated; it must set `kl_distill_scope="segnet_aux"` and
  `promotion_eligible=False`.
- `loss_mode="kl_distill"` is never self-explanatory. Promotion-capable KL
  configs must set `kl_distill_scope="segnet_aux"` and record
  `kl_distill_weight`, `kl_distill_temperature`, `eval_roundtrip`, and exact
  component gates. `kl_distill_scope="primary_scorer"` is forensic-only,
  requires `allow_banned_primary_kl_distill=True` and
  `promotion_eligible=False`, and must not be routed through SegMapTrainer.
- Retired/adversarially invalidated formulas or adaptive schemes must remain
  disabled unless a new proof and exact evidence reopens them.
- Loss weights, units, reductions, and temperature factors must be derived or
  empirically justified. No arbitrary constants in promoted lanes.
- Scorer-sensitive objectives need exact post-training archive eval; proxy
  losses do not promote.
- Bad training results are suspected engineering/config/math issues until
  reviewed.
- Component sensitivity promotion requires official CUDA finite-difference
  component response, full 600-pair sample coverage, response-curve gates,
  stability gates, exact contest eval custody, and
  `component_sensitivity_v1` manifest validation. Fisher-proxy sensitivity
  maps are diagnostic only. Use
  `experiments/profile_component_sensitivity.py --promotion-finite-difference`
  only for the promotion path; the default profiler mode must remain
  non-promotable.

## Harness And DX Hardening

- Fix bug classes, not just individual bugs.
- Prefer fail-closed behavior for promotion paths.
- Add preflight checks for known meta-bugs: stale CLI guidance, regex score
  parsing, nondeterministic archives, sidecars, duplicate dispatches, stale
  trackers, non-CUDA evals, and hidden fallbacks.
- Format repackers must distinguish semantic codec changes from archive-layout
  no-ops. If a source payload is already in the target codec family, a changed
  knob such as block size, bit depth, grouping, quantizer, or entropy mode must
  either decode/re-encode the payload and record source/target contract
  provenance, or explicitly mark the candidate as a no-op control.
- Frontier archives may be single charged runtime blobs (`p`,
  `renderer_payload.bin`, or compressed variants). Repackers and analyzers must
  consume the deployed archive contract directly, using the contest runtime
  unpacker when needed, instead of relying on lane-local exploded sidecars.
- Remote scripts must use strict shell mode, deterministic packaging,
  lane-local JSON adjudication, heartbeat/provenance logs, and explicit
  hardware recording.
- Avoid duplicate dispatches: use locks and live-prefix checks. Promotion-capable
  launchers must fail closed when an active process already targets the same
  lane output directory.
- Same-lane parallel dispatch is allowed only as an audited child claim:
  `tools/claim_lane_dispatch.py claim --allow-parallel --child-of <active-job>
  --parallel-reason <why-disjoint>`. This is for bounded cases such as promoting
  a completed H100 diagnostic candidate on T4 while the parent sweep finishes.
  Do not use `--force` for normal same-lane parallel work.
- Keep all warnings and low-severity DX issues on the hardening backlog.
- MCP servers are disabled for this project unless explicitly re-enabled by the
  user; do not depend on MCP tools for routine work.
- MCP is globally disabled for this operator environment as of 2026-05-01:
  known MCP config files, plugin caches, tool-output caches, OAuth/state files,
  and `.playwright-mcp` artifacts have been removed from the local tool homes.
  Do not recreate, install, sync, or enable MCP server/plugin state in Claude,
  Cursor, Gemini, LM Studio, Codex, or project-local config.
- Repo-owned MCP config files must have empty `mcpServers` objects and no
  active `mcp_servers` TOML sections; preflight blocks accidental reactivation.
- If MCP helper processes respawn from an outer app/runtime, kill only the exact
  MCP command patterns and continue without relying on those integrations.
- Already-running MCP helper processes are a preflight failure during
  contest/eval work. Run `check_no_live_mcp_processes(strict=True)` after
  cleanup and before trusting the local execution environment.
- Use `scripts/kill_orphaned_mcp_processes.py --strict` to clean the known MCP
  helper process class (`chrome-devtools-mcp`, `rbx-studio-mcp`,
  `roblox_studio_mcp`, and `model.context`). Prefer this over broad process
  killing. If helpers keep respawning, continue killing exact matches and treat
  the supervisor as external noise unless the user explicitly re-enables MCP.
- MCP cleanup/preflight must distinguish live helpers from audit commands that
  merely mention MCP tokens. Keep regression coverage so `find`, `rg`, `grep`,
  shell audit commands, and Python one-liners containing these strings are not
  killed or reported as live MCP helpers, while direct binaries, `npm exec`,
  `npx`/package launchers, shell-wrapped launches, and `python -m
  model.context` remain blocked.
- Lightning source manifests are part of promotion custody. Any manifest used
  to submit exact-eval or component-response jobs must fail closed on absolute
  paths, `..` traversal, duplicate entries, empty separators, backslashes,
  control characters, hidden files, macOS resource forks, and `__MACOSX`
  entries. Queue metadata that names a baseline JSON must be covered by the
  staged source manifest.
- Lightning SSH harvest targets must be state-derived where possible and must
  use a configured SSH alias or user-qualified target. Not bare ssh.lightning.ai —
  preflight `check_lightning_ssh_static_policy` is FATAL on regressions to the
  bare provider host (it is not acceptable for reproducible artifact custody).
- Lightning SSH commands used by repo wrappers must set noninteractive auth,
  a finite `ConnectTimeout`, client keepalives (`ServerAliveInterval` and
  `ServerAliveCountMax`), `TCPKeepAlive=yes`, and bounded
  `ConnectionAttempts`. Transient provider key-exchange resets such as
  `kex_exchange_identification` or `Connection reset by peer` should be retried
  only as transport failures; public-key auth failures, disabled host-key
  checking, and supply-chain scan failures must still fail closed.
- Modal component-sensitivity fallback work must use the dedicated lightweight
  direct-FD shard launcher, keep the same deterministic shard topology as the
  Lightning wave it backs up, and remain diagnostic/non-promotable until exact
  CUDA archive custody and official response gates are satisfied. Do not use
  broad Modal training mounts for this path when only the PFP16 archive,
  source, and upstream scorer assets are needed.
- Modal wrappers that rely on DALI/NVDEC must pin the DALI CUDA package to the
  audited exact-eval line (`nvidia-dali-cuda120==1.52.0` for CUDA 12) and use
  the NVIDIA package index. Unpinned Modal mirror resolution can change runtime
  DALI behavior and must fail preflight before score work.
- A Modal or Vast CUDA/DALI/NVDEC preflight failure is an infrastructure abort,
  not lane evidence. Do not retry the same backend blindly when the same
  NVDEC/NVML class repeats; reroute the scientific experiment to Lightning or
  another exact-CUDA backend with manifest custody.
- Mask/video decoding helpers must validate candidate `ffmpeg` binaries before
  use. `TAC_FFMPEG` may be either an executable path or a command name on
  `PATH`; an explicit unusable override must fail closed, while a broken
  repo-local `upstream/ffmpeg-new` must be skipped in favor of a validated
  system binary when one is available.
- Remote archive-only eval runners must preflight both `uv` and a
  parity-compatible `FFMPEG_BIN` before spending GPU time. The selected ffmpeg
  must expose the explicit inflate color-contract `scale` options used by
  `submissions/robust_current/inflate.sh` (`in_range`, `out_range`,
  `in_color_matrix`, `in_primaries`, and `in_transfer`). Ubuntu 22.04
  `ffmpeg` is not sufficient unless this preflight proves otherwise.
- Standalone component traces must enforce the same runtime parity as exact
  archive evals before their per-pair profile feedback can drive optimizer
  decisions. `experiments/contest_component_trace.py` must select or reject
  `FFMPEG_BIN` against the explicit color-contract options above, isolate
  inflate-side `uv run` with a job-local `UV_PROJECT_ENVIRONMENT` and
  `UV_LINK_MODE=copy`, emit `component_trace_runtime_env.json`, and remain
  explicitly non-promotable unless cross-checked against exact CUDA
  `contest_auth_eval.json` for identical archive bytes.
- Empirical Alpha diagnostic helpers must be bounded by default and explicitly
  non-promotable. Full-corpus or expensive primitive diagnostics require an
  explicit operator flag, and their outputs remain design signal only until a
  resulting archive passes exact CUDA auth eval.
- J-NWC/NWCS promotion paths must use exact corpus-manifest custody. When a
  prebuilt corpus manifest or `CORPUS_SENSITIVITY_PT` is involved, remote
  scripts must receive the matching `PREBUILT_CORPUS_MANIFEST` and
  `CORPUS_REPLAY_ROOT`, preserve manifest bytes/SHA, and fail closed on
  missing or mismatched replay custody.
- Q-FAITHFUL/JointFrameGenerator training must consume a proven nonzero
  deployed pose stream. Silent zero-pose fallback is forbidden for
  `variant=quantizr_faithful`; training checkpoints and QFAI/QZS3 exports must
  record a `training_pose_contract` with pose dimension, pair count, SHA-256,
  `training_uses_nonzero_pose_stream=true`, and
  `zero_pose_fallback_allowed=false`. Launchers for this family must build
  half-frame mask archives explicitly and deploy runtime-readable QFAI/QZS3
  bytes, not brotli-compressed data under a raw `.bin` member name.
- Trained-renderer export scanners must include harvested Q-FAITHFUL artifact
  trees and classify QFAI/QZS3/MQZ-style exports by evidence state: present
  but raw-needs-packing, present but transplant-preflight-blocked, or
  present-with-exact-negative CUDA evidence. Do not report this family as
  "missing" when non-surrogate artifacts exist, and do not allow
  exact-negative Q-FAITHFUL snapshots to re-enter H100/T4 dispatch through a
  generic renderer self-compression readiness path.

## Verification Before Deployment

Before treating any change as deployable:

```bash
.venv/bin/python -m py_compile <touched python files>
.venv/bin/python -m pytest <focused tests> -q
bash -n <touched shell scripts>
git diff --check
```

For score-affecting changes, also require:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive <candidate archive.zip> \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir upstream \
  --device cuda \
  --keep-work-dir \
  --work-dir <evidence dir>
```

The final deploy packet must include archive, SHA, manifest, JSON, logs,
hardware provenance, command line, source/staged-tree manifest, upstream hash,
and adversarial review status.

## Agent Workflow

- Read the relevant source docs and current progress ledgers before acting.
- Use subagents when explicitly authorized for parallel research, audit, or
  implementation. Give them disjoint scopes and require file paths changed or
  read-only output.
- When a trusted partner agent is working in parallel, treat their lanes and
  artifacts as live shared-worktree scopes. Do not overwrite, revert, or
  reinterpret their work without independent custody review. Handoff artifacts
  from partner agents must pass the same archive SHA/bytes, payload closure,
  CUDA auth-eval, component recomputation, manifest, and adversarial-review
  gates before promotion, ranking, stacking, or paper claims.
- Use online research liberally for current papers, OSS, Lightning AI tooling,
  entropy coders, learned compression, and optimization methods, but never use
  external results as contest evidence.
- Record progress in adjacent dated markdown docs and persistent memory.
- End substantial turns with work landed, ongoing work, roadmap, and next
  steps needed to keep pushing toward the floor.
- Never revert unrelated user/agent changes in a dirty worktree.
- If the work is blocked on user action and the user is not present, use the
  installed `imsg` CLI for brief escalation. Use only when intervention is
  genuinely required to unblock an active experiment or security issue; include
  the run/job name, the blocker, and the exact requested action. Do not store
  private phone numbers, emails, or chat IDs in source; pass them via local
  shell history, environment, or the operator's active command context.
