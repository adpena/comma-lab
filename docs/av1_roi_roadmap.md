# av1 + roi roadmap

This is the canonical roadmap for the speculative AV1+ROI lane.

It exists to keep the lane explicit, honest, and resumable:

- what we know
- what is already live
- what is still missing
- what public evidence suggests
- what must be true before this lane can become authoritative

## status

### current live canonical lane

The live promoted Track B floor is still the flat AV1 lane:

- `524x394`
- `libsvtav1`
- `preset=0`
- `crf=34`
- `film-grain=22`
- `lanczos` downscale
- `lanczos` upscale
- `unsharp=9:9:0.35:9:9:0.0`
- explicit encoded tags: `tv / bt709 / bt709 / bt709`
- explicit decode conversion: `rgb24(pc)`

Measured authoritative local result:

- `current_workflow = 2.12`

Measured local honest accounting estimate:

- `rule_faithful = 2.1418040615200598`

### current av1+roi status

AV1+ROI is **not** a live supported lane.

It is intentionally blocked because the existing ROI machinery is still x265-centric.

Current guardrail:

- `ROI_ENABLE=1` with `VIDEO_CODEC!=libx265` fails fast

That guard exists to avoid a dishonest state where the config looks AV1+ROI-capable but the implementation silently falls back to x265 assumptions.

## why this lane matters

The contest score is:

`100 * segnet_distortion + 25 * rate + sqrt(10 * posenet_distortion)`

That creates a strong incentive to:

- preserve task-relevant structure
- preserve motion / temporal cues
- avoid wasting bits on regions the learned metrics care about less

That is exactly why ROI-style ideas remain attractive:

- keep the driving corridor / semantically important regions cleaner
- allow more aggressive degradation outside them
- keep the total byte count low enough that the rate term still wins

## what public evidence already says

As checked on 2026-04-06, the public leading score is `1.98` (`damir_bearclaw_002`, PR `#30`).

That public submission describes a pipeline that:

- stays in the AV1 family
- uses spatially varying fidelity
- applies a hand-authored ROI mask schedule over time
- preserves the ROI baseline path
- degrades outside the ROI more aggressively

Interpretation:

- the public leaderboard already validates the **non-uniform spatial fidelity** idea
- the winning public path is much closer to ROI-aware degradation than to our current flat AV1 lane

That does **not** mean our repo currently supports an honest equivalent.

It means the idea has external evidence and is worth a disciplined revisit.

## what we have already learned locally

### broad lessons

1. The flat AV1 lane is real and competitive.
2. The rawvideo byte-layout bug was catastrophic and had to be fixed first.
3. Explicit evaluator-facing color handling materially mattered.
4. The local flat frontier is already fairly tight.
5. ROI ideas are plausible, but our earlier tested forms were too expensive or too brittle.

### local negative roi evidence

The repo already contains informative negative ROI results:

- `robust_current-roi-two-pass-cpu-2026-04-04`
- `robust_current-dynamic-main-roi-cpu-2026-04-05`

These runs showed that naive or heavy ROI variants can easily lose once task distortion is counted honestly.

So the lane should not be reopened as “ROI because ROI sounds good.”

It should be reopened only as:

- AV1-aware
- parity-checked
- scorer-measured

## implementation gap

The missing pieces are concrete.

### 1. codec-agnostic roi encode abstraction

The current ROI machinery assumes x265 rate-control knobs and x265 invocation patterns.

What is needed:

- one abstraction for base / roi / roi2 encoding
- codec-specific parameter emission underneath
- no hidden x265-only assumptions at call sites

### 2. av1 params for base / roi / roi2 streams

Need explicit AV1-capable parameter surfaces for:

- base stream
- ROI stream
- optional second ROI stream

Examples of knobs likely needed:

- `preset`
- `crf`
- `film-grain`
- `keyint`
- maybe `sharpness`
- maybe `scd`

### 3. matching av1-aware metadata roi path

The metadata-driven ROI path must be AV1-aware too.

That includes:

- archive construction
- stream naming / manifest semantics
- decode / inflate assumptions
- branch-specific validation

### 4. inflate / smoke / scorer parity checks

This lane must not bypass the rigor that the flat path now has.

Required:

- inflate parity
- exact raw count / geometry checks
- sampled RGB semantic checks
- scorer-backed result

### 5. scorer-backed proof that it actually helps

No promotion without a measured win.

This lane stays speculative until it produces scorer-backed evidence that beats the flat AV1 floor honestly.

## roadmap

### phase 0 — keep the guard

Current rule stays in force:

- fail fast on AV1+ROI instead of silently drifting

This protects compliance and keeps comparisons honest.

### phase 1 — refactor for truthfulness

Goal:

- separate ROI pipeline structure from codec-specific implementation

Deliverables:

- codec-agnostic encode interface
- explicit codec branch selection
- no x265-only assumptions in shared logic

Acceptance:

- x265 ROI path still works
- AV1 path can be wired without fake support

### phase 2 — av1 roi minimum viable path

Goal:

- one honest AV1 two-stream ROI path

Deliverables:

- AV1 base stream encode
- AV1 ROI stream encode
- matching inflate path
- matching smoke pass

Acceptance:

- packages successfully
- inflates successfully
- passes raw count / geometry / semantic smoke checks

### phase 3 — scorer validation

Goal:

- prove whether the lane is worth anything

Process:

1. run smoke
2. run scorer
3. compare against live flat floor
4. if worse, record as a rejection
5. if better, still require promotion review before canonicalization

### phase 4 — only then consider richer roi variants

Only after a minimal AV1 ROI path is honest and measured should we consider:

- dynamic ROI windows
- multi-ROI variants
- teacher-first ROI schedules
- staticness-aware masks

## promotion criteria for this lane

AV1+ROI may be promoted only if all of the following are true:

1. packaging succeeds
2. inflation succeeds
3. smoke passes:
   - file count
   - exact frame count
   - exact geometry bytes
   - sampled RGB semantic sanity
4. scorer result is better than the current flat floor
5. promotion review explicitly confirms:
   - `current_workflow` accounting
   - `rule_faithful` accounting
   - no hidden codec drift
   - no unsupported-path ambiguity

## what not to do

- do not treat public ROI success as proof that our implementation is ready
- do not silently relax the AV1+ROI fail-fast guard
- do not promote ROI because it is fashionable or because the leaderboard leader uses a related idea
- do not confuse BAT00 or other side-lane experiments with authoritative promotion evidence

## velocity policy for this lane

To increase development speed without losing rigor:

- keep one canonical roadmap document instead of spreading lane status across chat memory
- keep one explicit status:
  - `speculative`
  - `experimental`
  - `promotable`
- use cheap smoke checks before scorer time
- keep authoritative claims on the local official-style scorer path
- let side machines help with:
  - profiling
  - implementation
  - surrogate ranking
  - branch experiments
- do not let side machines become the source of authoritative score claims

## current recommendation

Do **not** reopen AV1+ROI as the primary lane yet.

Near-term recommendation:

1. keep squeezing the flat AV1 lane for a sub-`2.1` result
2. keep AV1+ROI explicitly speculative
3. reopen this roadmap only after the next low-hanging-fruit exploration round is complete or the flat lane stalls decisively

## trusted-partner-agent handoff

This section is the operational handoff for any partner agent or side lane.

If an agent picks up AV1+ROI work concurrently, it should be able to do so from this document alone plus the referenced files.

### mission for the lane

Build an **honest, AV1-aware, scorer-measured ROI lane** that can be compared directly against the current flat AV1 floor without ambiguity in codec path, byte accounting, or inflate behavior.

### hard constraints

- do not edit outside the mutation frontier
- do not weaken or remove the AV1+ROI fail-fast guard unless real AV1+ROI support is being added
- do not blur `current_workflow` and `rule_faithful`
- do not use BAT00 / tertiary numbers as authoritative promotion evidence
- do not promote the lane without scorer-backed local evidence

### canonical source files

These are the first files a partner agent should inspect.

#### packaging / eval / smoke

- `src/comma_lab/evaluate.py`
- `src/comma_lab/smoke.py`
- `src/comma_lab/install.py`
- `src/comma_lab/cli.py`

#### submission implementation

- `submissions/robust_current/compress.sh`
- `submissions/robust_current/inflate.sh`
- `submissions/robust_current/analyze_roi.py`
- `submissions/robust_current/config.env`

#### rigor / status docs

- `docs/speculative_lanes.md`
- `docs/compliance_audit.md`
- `reports/scoring_rigor_review.md`
- `reports/ffmpeg_path_review.md`
- `.omx/research/findings.md`
- `.omx/state/current_focus.md`
- `.omx/state/next_experiments.md`

### live config knobs relevant to this lane

#### flat AV1 baseline knobs

Current canonical flat AV1 config:

- `SCALE_W=524`
- `SCALE_H=394`
- `DOWNSCALE_FLAGS=lanczos`
- `UPSCALE_FLAGS=lanczos`
- `VIDEO_CODEC=libsvtav1`
- `SVT_AV1_PRESET=0`
- `SVT_AV1_CRF=34`
- `SVT_AV1_PARAMS=film-grain=22:keyint=180`
- `INFLATE_POSTFILTER=unsharp=9:9:0.35:9:9:0.0`

#### ROI knobs already present

These exist today, but are still x265-oriented in practice:

- `ROI_ENABLE`
- `ROI_METADATA_ENABLE`
- `ROI_X_FRAC`
- `ROI_Y_FRAC`
- `ROI_W_FRAC`
- `ROI_H_FRAC`
- `ROI_BASE_CRF_DELTA`
- `ROI_CRF_DELTA`
- `ROI2_ENABLE`
- `ROI2_X_FRAC`
- `ROI2_Y_FRAC`
- `ROI2_W_FRAC`
- `ROI2_H_FRAC`
- `ROI2_CRF_DELTA`
- `ROI_METADATA_WINDOW_FRAMES`
- `ROI_METADATA_SAMPLE_STEP`
- `ROI_METADATA_TILE_COLS`
- `ROI_METADATA_TILE_ROWS`

#### color / output contract knobs

These must remain explicit on any AV1+ROI path too:

- `SOURCE_W`
- `SOURCE_H`
- `SOURCE_COLOR_RANGE`
- `SOURCE_COLOR_MATRIX`
- `SOURCE_COLOR_PRIMARIES`
- `SOURCE_COLOR_TRC`
- `RGB_OUTPUT_RANGE`

### known good / known bad evidence

#### current flat AV1 authoritative floor

- canonical summary:
  - `reports/raw/robust_current-current_workflow-cpu-summary.json`
- canonical report:
  - `reports/raw/robust_current-current_workflow-cpu-report.txt`

#### fresh smoke evidence

- `reports/raw/2026-04-06-semantic-rigor/robust_current-smoke.json`

This confirms:

- file count pass
- exact frame-count pass
- exact geometry-derived byte-size pass
- sampled RGB semantic sanity pass

#### ROI-specific negative or cautionary evidence

- `reports/raw/2026-04-04-roi-two-pass-prototype/robust_current-roi-two-pass-current_workflow-cpu-summary.json`
- `reports/raw/2026-04-05-dynamic-main-roi/robust_current-dynamic-main-roi-current_workflow-cpu-summary.json`
- `reports/raw/2026-04-06-rigor-pass/roi-rigor-checks.txt`

Use these before re-proposing the lane. They capture earlier failure modes and guardrail checks.

### public comparison snapshot

As of 2026-04-06:

- `#20` (`2.09`) uses AV1 + film grain + bicubic upscale + strong unsharp
- `#23` (`2.08`) is in the same family and adds `sharpness=1`
- `#24` (`2.05`) is in the same family and includes `scd=0`
- `#30` (`1.98`) appears to use a hand-authored ROI mask schedule over time

Interpretation:

- our current flat lane most closely resembles `#20/#23/#24`
- the public leader `#30` is the clearest external validation of the ROI idea
- our repo does **not** yet support an honest AV1-aware equivalent of `#30`

### exact current guardrail state

The current live guard is operationally correct and intentional:

- `ROI_ENABLE=1` with `VIDEO_CODEC!=libx265` fails

That is not a bug to “fix around.”

It is the current honesty boundary.

### what a partner agent may safely do now

#### allowed immediately

- read and map the current ROI implementation
- design a codec-agnostic abstraction
- prepare AV1-specific parameter plumbing
- add narrow tests / smoke helpers
- improve documentation and evidence logging
- use BAT00 / tertiary for profiling, prototyping, or surrogate ranking

#### not allowed without fresh scorer evidence

- promoting AV1+ROI to canonical Track B
- describing AV1+ROI as supported in the current repo
- using side-machine results as authoritative competition claims

### concurrency split for side agents

If multiple agents or machines work on this simultaneously, split the lane like this:

#### lane A — local authoritative lane

Owner:

- main local repo

Responsibilities:

- authoritative smoke
- authoritative scorer
- final promotion decision
- final doc/status updates

#### lane B — implementation / refactor lane

Owner:

- BAT00 or tertiary if available, otherwise a local branch/worktree

Responsibilities:

- codec-agnostic ROI abstraction
- AV1 parameter plumbing
- metadata-path cleanup

Constraint:

- no authoritative score claims from this lane

#### lane C — research / public comparison lane

Owner:

- any side lane

Responsibilities:

- mine public PRs for ideas
- compare method families
- identify candidate knobs worth testing locally

## partner-agent operating contract

This section is the authority model for a trusted concurrent partner agent.

### default posture

A trusted partner agent working on this lane has permission to:

- explore aggressively
- prototype quickly
- refactor hard inside the lane
- run many speculative experiments
- use side machines for throughput
- push branches / commits frequently

That freedom is intentional.

The constraint is not “move slowly.”

The constraint is:

- never lie
- never blur authority boundaries
- never lose observability

### what the partner agent is allowed to change

Inside the mutation frontier, a trusted partner agent may change:

- `src/comma_lab/**`
- `submissions/robust_current/**`
- `docs/**`
- `reports/**`
- `.omx/**`
- `.ralph/**`
- `experiments/**`

provided the changes stay inside the AV1+ROI lane or its required support surfaces.

### what the partner agent must not do

- must not treat AV1+ROI as canonical before the lane clears the promotion gates
- must not weaken the explicit current guardrails for convenience
- must not hide failures or negative results
- must not collapse `current_workflow` and `rule_faithful`
- must not claim side-machine numbers as authoritative scorer results
- must not leave undocumented branch-specific behavior changes

### required observability

Every serious experiment or refactor attempt must leave behind enough signal for another agent to resume without chat history.

Minimum required outputs:

1. **config / command record**
   - exact env or config diff
   - exact command used

2. **artifact record**
   - archive path
   - archive bytes
   - packaging view

3. **verification record**
   - smoke pass/fail
   - scorer pass/fail if run
   - branch-specific caveats

4. **interpretation record**
   - estimate before run
   - result after run
   - short reflection on why it likely helped or hurt

### required documentation surfaces

The partner agent must update at least:

- `.omx/state/current_focus.md`
- `.omx/state/next_experiments.md`
- `.omx/research/findings.md`
- `.ralph/run_log.md`
- `reports/latest.md`

For lane-specific work, also update:

- `docs/speculative_lanes.md`
- this file: `docs/av1_roi_roadmap.md`

### required raw evidence surfaces

Every serious experiment should leave raw evidence under a dated folder, for example:

- `reports/raw/<date>-av1-roi-<candidate>/...`

Suggested contents:

- smoke JSON
- scorer summary JSON
- scorer report TXT
- ad hoc notes or logs as needed

### authority model

#### what the partner agent may decide alone

- branch structure
- local refactors inside the lane
- speculative candidate ordering
- side-machine usage
- whether a candidate is obviously rejected before scorer time

#### what must come back to the authoritative lane

- canonical promotion decision
- public submission decision
- any claim that a result is the new honest floor
- any change that affects packaging-view truthfulness

### branch / commit hygiene

Partner agents should commit often enough that work is resumable, but not so often that history becomes noise.

Commit messages should preserve:

- why the change was made
- what constraint shaped it
- what was tested
- what was not tested

### experiment-hard / document-hard rule

The intended behavior is:

- **experiment hard**
- **document harder**

Fast iteration is good.

Fast iteration without evidence is not.

### preferred partner-agent workflow

1. read this roadmap
2. pick one bounded subproblem
3. implement fast
4. run smoke / checks
5. record evidence immediately
6. commit
7. hand back promotion decisions to the authoritative lane

### what “good concurrent help” looks like

- a branch that adds real capability or real evidence
- a raw evidence folder with exact commands and outputs
- updated durable state
- clear note on whether the work is:
  - infrastructure only
  - research only
  - scorer-ready
  - promotable

### what “bad concurrent help” looks like

- a branch that changes many things but leaves no evidence
- “it should work” without smoke/scorer proof
- undocumented packaging-view ambiguity
- side-machine claims presented as canonical

### recommended execution order for a partner agent

1. read:
   - `compress.sh`
   - `inflate.sh`
   - `analyze_roi.py`
2. confirm where x265 assumptions still leak
3. write down every x265-only dependency
4. design a codec-agnostic abstraction with the narrowest possible surface
5. implement only enough AV1 support to create one honest two-stream path
6. run smoke
7. hand back to the authoritative local lane for scorer evaluation

### minimal acceptance checklist for the first honest AV1+ROI prototype

- [ ] AV1 base stream encode path exists
- [ ] AV1 ROI stream encode path exists
- [ ] inflate path reconstructs correctly
- [ ] smoke passes:
  - [ ] file count
  - [ ] exact frame count
  - [ ] exact geometry bytes
  - [ ] sampled RGB semantic sanity
- [ ] scorer run completes locally
- [ ] results are recorded under both packaging views
- [ ] failure or win is written down honestly

### failure modes to watch for

These are the bugs or ambiguity classes most likely to waste time:

1. **silent codec drift**
   - config says AV1+ROI
   - implementation still encodes ROI pieces with x265

2. **inflate path mismatch**
   - archive structure changes
   - inflate assumptions do not match

3. **byte-accounting ambiguity**
   - side assets not clearly included / excluded under the chosen packaging view

4. **color-contract drift**
   - AV1 ROI path reintroduces implicit ffmpeg defaults

5. **non-authoritative benchmark creep**
   - BAT00 or tertiary numbers start being treated like promotion evidence

### required evidence logging for every AV1+ROI attempt

Every serious run should record:

- exact config / env
- archive bytes
- smoke result
- scorer result if run
- packaging view
- whether the lane is:
  - rejected
  - research-only
  - promotable
- short reflection on what changed and why it likely helped or hurt

### where to record new work

- durable status:
  - `.omx/state/current_focus.md`
  - `.omx/state/next_experiments.md`
- findings:
  - `.omx/research/findings.md`
- iteration history:
  - `.ralph/run_log.md`
- high-level report:
  - `reports/latest.md`
- speculative lane updates:
  - `docs/speculative_lanes.md`
- raw evidence:
  - `reports/raw/<date>-<lane>/...`

### one-sentence decision rule

If the lane is not yet **AV1-aware, parity-checked, and scorer-measured locally**, it is still speculative and must be treated that way in both code and prose.
