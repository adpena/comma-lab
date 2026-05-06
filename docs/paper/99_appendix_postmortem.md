# Appendix A. Postmortem: the May 4 race window

This appendix expands the §7.8 lessons-learned section with the full
medalist LOC table, the five explicit lessons, and per-lesson reactivation
criteria suitable for camera-ready inclusion.

## A.1 The race window

The comma.ai contest deadline was 2026-05-03 11:59pm AOE
(= 2026-05-04 12:00 UTC). PR #95 (`hnerv_muon`, AaronLeslie138) — the seminal
HNeRV-class submission — was published at 2026-05-04 07:47:15 UTC, scoring
0.20. The final top three all landed within a **4 hour 8 minute window**
that followed it.

### A.1.1 Top-three deadline submissions

| Rank | PR | Author | Created (UTC) | Lines added | Files | Score |
|:----:|:----:|:--------:|:-------------:|:------------:|:------:|:------:|
| 🥇 1 | #101 | SajayR | 11:50:13 | 660 | 5 | 0.193 |
| 🥈 2 | #103 | rem2 | 11:55:56 | **241** | **2** | 0.195 |
| 🥉 3 | #102 | EthanYangTW | 11:54:32 | 367 | 7 | 0.195 |

### A.1.2 Other notable deadline-window submissions

| PR | Author | Created (UTC) | Lines added | Files | Score | Notes |
|:----:|:--------:|:-------------:|:------------:|:------:|:------:|:------|
| #95 | AaronLeslie138 | 07:47:15 | (seminal) | n/a | 0.20 | HNeRV release that decided the race |
| #97 | BradyMeighan | 09:40:11 | n/a | n/a | 0.23 | first iteration on HNeRV |
| #99 | BradyMeighan | 11:25:21 | n/a | n/a | 0.197 | second iteration |
| #100 | BradyMeighan | 11:52:47 | n/a | n/a | 0.195 | third iteration (silver-tier) |
| #98 | EthanYangTW | 11:00:12 | n/a | n/a | 0.196 | first iteration |
| #102 | EthanYangTW | 11:54:32 | 367 | 7 | 0.195 | bronze (above) |
| #96 | rem2 | 09:55:00 | n/a | n/a | 0.21 | first iteration |
| #103 | rem2 | 11:55:56 | 241 | 2 | 0.195 | silver (above) |
| #105 | valtterivalo | 11:58:00 | **1,776** | **21** | 0.198 | "kitchen sink" — 1776 LOC, lost to PR #103's 241 |
| #106 | valtterivalo | post-deadline | n/a | n/a | 0.20945673 | exact post-deadline frontier (`belt_and_suspenders`) |
| #107 | adpena (us) | 11:30:00 | n/a | n/a | 0.2293 | our final submission, ~11th place |

The medal-band submissions were small bolt-ons on top of HNeRV: a fine-tune
microcodec (#101), an arithmetic-coded latent variant (#103), an LC + scale-knob
tweak (#102). The silver medal was 241 lines in 2 files, shipped within 4
hours 8 minutes of HNeRV's publication. PR #105 (`kitchen_sink`) — 1,776
lines across 21 files, throwing every available technique at HNeRV — landed
at 0.198 and lost to PR #103's 241-line increment.

Each medalist iterated **publicly**: BradyMeighan shipped #97 → #99 → #100
in 2h12m; rem2 went #96 → #103 in 3h24m; EthanYangTW went #98 → #102 in
2h23m. Each public PR locked in a contest-CUDA score, forced an honest
measurement, and established presence on the leaderboard before competitors
could ship past.

## A.2 Our approach during the same window

We had every primitive needed pre-built: PR #106 byte-deconstructed; the
apogee_intN codec built (int4–int8 archives existed on disk); arithmetic
codec, block-FP, water-filling, sensitivity maps all checked in. PR #107
`apogee` landed at 0.229 (~11th). We did not ship a competitive submission
in the 4-hour window because we used the window to build infrastructure
rather than to dispatch candidates.

The infrastructure built during the race window was substantial and
defensible *as engineering*:

- A meta-Lagrangian search engine with predictor refusal modes
  (`tac.optimizer.MetaLagrangianSearch`)
- A closed-form distortion proxy (`experiments.distortion_proxy_local`)
- A 5-gate predispatch sanity ladder (`tools/predispatch_sanity.py`)
- Four new STRICT preflight checks (PCC9–PCC11 + smoke-promotion gate)
- An OSS extraction of the `tac` library to a public repository
- An adversarial-review subagent + a 7-bug fix subagent

Each item closes a specific bug class surfaced by a specific past incident.
None of them dispatched a candidate to a paid GPU during the window in which
the contest was decided.

## A.3 The planner-without-actuator failure mode

The deeper failure was architectural. The meta-Lagrangian engine was
*conceptually* a parallel-dispatch system: rank N candidates locally in
microseconds (the proxy is closed-form), select top-K, fire K dispatches
**in parallel** to N concurrent paid GPUs (~$0.11 per Lightning T4
dispatch), harvest empirical anchors, reseed the calibration. With 16
concurrent dispatches at $0.11 each, ~$2 buys 16 simultaneous empirical
anchors per cycle, and the loop converges in 2-3 cycles.

What we built was the **ranking layer** (`evaluate_all`, `top_k`, refusal
modes, sanity gate). What we did **not** build was the **actuator**:
`concurrent.futures.ThreadPoolExecutor` over the existing dispatch wrapper,
~150 lines. Without the actuator, the engine produces a ranked list that no
one executes. With the actuator, every loop tick fans out 16 dispatches and
gathers 16 empirical anchors. The architecture was *right design with the
actuator missing*.

A cron job had been firing every 5 minutes during the entire race window
with the prompt *"push to implement all necessary under plan for shannon
theoretical floor in absolute minimum wall clock."* The 5-minute cadence is
the natural cadence for fan-out-and-harvest sweeps when dispatch wall-clock
is shorter than the loop tick. Each tick should have launched the next batch
of 16 dispatches. Instead, the agent translated *"absolute minimum wall
clock"* as *"absolute maximum local rigor"* and used each tick to add
another sequential validation gate. The cron was effectively a
force-multiplier for the wrong subroutine.

## A.4 The five lessons (with reactivation criteria)

These five lessons are the camera-ready externalization of the post-mortem.
Each names a specific structural mechanism and the precise condition under
which the lesson would be **reactivated** (i.e., reinstated as the
recommended default) if invalidated by future evidence.

### Lesson 1 — Build the actuator first, the ranker second

**Mechanism:** A ranker without a parallel actuator is a planner that
produces ranked plans no one executes. When dispatch is paid and ranking is
free, the binding constraint is dispatch fan-out, not ranking quality.

**Reactivation criterion:** If a future contest measurement environment is
free (no paid-GPU constraint) and ranker compute is the binding constraint,
this lesson inverts: rank first, dispatch sequentially. Trigger
condition — empirical evidence that ranking-CPU-cost dominates dispatch-
wall-clock-cost in the target loop.

### Lesson 2 — Strategic-rigor must be explicit in the loop dispatcher prior

**Mechanism:** Pre-leadership-shift, max rigor is correct (every wasted
dispatch is the operator's money). Post-leadership-shift, max velocity is
correct (every minute of additional gating is a competitor shipping ahead).
The transition between these priors must be triggered by a runtime detector.

**Concrete instantiation:** poll the public leaderboard every cycle; when
it has moved within the last 24 hours, narrow top-K, drop sanity gates that
block on proxy-only evidence, prioritize ship-velocity over local rigor.

**Reactivation criterion:** If a future loop has no public leaderboard or
no measurable leadership signal, the runtime detector cannot fire and the
default reverts to max rigor. Trigger condition — explicit absence of a
publicly-observable competitive signal.

### Lesson 3 — Public-PR cadence is itself a competitive primitive

**Mechanism:** Each medalist iterated publicly, locking in a contest-CUDA
score with each PR. We iterated privately. Public PRs force honest
measurement, establish leaderboard presence, and force competitors to ship
past you (rather than around you). The medalist-band submitted N=2-3 PRs
each in the deadline window; we submitted N=1.

**Reactivation criterion:** If a future contest disallows iterative public
submissions (single-PR rule, sealed-bid format, or no public leaderboard
during the window), the lesson inverts: hold private artifacts to maximize
information advantage.

### Lesson 4 — Kitchen-sink loses to focused increment under deadline pressure

**Mechanism:** PR #105 threw 1,776 lines across 21 files at HNeRV and lost
to PR #103's 241 lines in 2 files. Under deadline pressure, the marginal
expected score from focused incremental improvement on a known-good
paradigm exceeds the expected score from architectural breadth, because
breadth costs integration risk per file added.

**Reactivation criterion:** If a future deadline window has *no* known-good
paradigm to incrementally improve (i.e., no equivalent of HNeRV publishing
4 hours pre-deadline), kitchen-sink may dominate. Trigger condition —
absence of a recent-published paradigm that competitors converged on.

### Lesson 5 — Pre-deadline harness work is investment; intra-window harness work is forfeit

**Mechanism:** Every minute of intra-window engineering is a minute not
spent dispatching. The temporal locality of the deadline window means
harness ROI inverts from positive (pre-window) to negative (intra-window).

**Reactivation criterion:** If a future workflow has no hard deadline (e.g.,
internal research with no fixed publication date), the lesson does not
apply — pre-deadline / intra-window distinction is meaningless. Trigger
condition — absence of a binding hard deadline.

## A.5 Post-deadline corrective artifacts

Three artifacts now exist in the repository to extinct this failure mode
structurally:

### `tools/parallel_dispatch_top_k.py`

`concurrent.futures.ThreadPoolExecutor` over `tools/lightning_dispatch_pr106_stack.py`
and `scripts/launch_lane_on_vastai.py`. Per-dispatch and total-cost gating;
per-dispatch timeouts; harvested-JSONL output. Strict-mode rejects
candidates not marked `ready_for_exact_eval_dispatch=true` to honor the
post-int4-falsification safety reflex.

### `tools/harvest_and_reseed.py`

Ingests harvested JSONL, drops any row not tagged `[contest-CUDA]` (per
the repository auth-eval-everywhere rule), cross-verifies the harvested score
against the per-dispatch `contest_auth_eval.json`, appends new empirical
anchors to `.omx/calibration/anchors_*.json`. Closes the prediction →
empirical → updated-prediction feedback loop.

### `tools/feedback_loop_sweep.py`

Single binary that runs the closed loop end-to-end: rank → dispatch
(in-process `ThreadPoolExecutor`) → harvest → reseed → check convergence →
repeat. Configurable `--max-cycles`, `--max-total-cost`,
`--max-cost-per-cycle`, `--top-k`, `--convergence-eps`. The `--race-mode`
flag forces the leadership-shift prior immediately: narrows top-K to 4,
caps per-cycle spend to $0.50, drops sanity gates that block on proxy-only
evidence.

The corresponding repository rule is now explicit: when the work calls for
parallel, high-throughput search, automation, or stacking sweeps, the first
file built must be the actuator that fans out N concurrent paid-GPU
dispatches. A ranker without a parallel actuator is a planner that produces
ranked plans no one executes. Any future PR that adds a new ranking,
predictor, or sanity-gate primitive must explicitly link to the
parallel-actuator file that consumes its output.

## A.6 Cross-references

- The full game-theoretic premature-convergence analysis: §7.7
- The §7.8 in-paper summary of this postmortem (lighter weight)
- The OSS extraction `tac` library: <https://github.com/adpena/tac>
- The empirical incident memo: `feedback_may_4_hnerv_race_postmortem_20260505.md`
  in the parent comma-lab repository's auto-memory store
