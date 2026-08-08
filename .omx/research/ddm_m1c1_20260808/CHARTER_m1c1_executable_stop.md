# ddm_m1c1 — M1 FIRE GATE: BUILD the executable stop chain + the derivation cures

Round 4's three independent review passes ALL returned `FINDINGS_RESET_COUNTER`. You are building
the cures. You are NOT reviewing, and you do NOT fire the burn.

## The finding you are curing (MEASURED, verified at source by MAIN — re-verify, don't trust)

**The M1 ticket's event-driven stopping rule has no executor.** Found independently by two lenses:

- M1R4B-F1: *"the burn is declarative, not executable: the pinned trainer emits no atomic per-eval
  JSONL, calls no trajectory-stop evaluator."*
- M1R4C-F2: `run_mlx_train` appends eval rows to an **in-memory list**, checkpoints every 250 steps,
  and writes the result at process end (`experiments/ddm_mx1_pr130_semantic_renderer.py:3079-3219,4025`).
  The only `_write_jsonl` calls belong to the offline facets analyzer (`:1584-1590,1850-1853`).

MAIN's independent re-derivation (2026-08-08), which you MUST reproduce before building:
- `grep -rn "evaluate_trajectory_stop" src tools experiments --include="*.py"` → exactly ONE
  production caller: `src/tac/optimization/terminal_pose_gn.py:1231` (the terminal POSE solver).
  **Nothing on the M1 seg training path calls it.**
- No `stop_policy` consumer exists near the trainer. The live controls are `--steps 3250` and the
  `safe_run` timeout.

⚠ **Instrument warning, from MAIN's own first attempt:** an unquoted `--include=*.py` under zsh
globs before grep runs and returns a clean, false "0 matches". Quote your globs. A zero is a
measurement; check the instrument before believing it.

## What you build

**B1 — a resume-safe per-eval journal on the REAL training path.** Every eval row is appended
atomically (tmp+rename or O_APPEND with a single write), immediately, to a durable JSONL under the
run dir — not accumulated in memory and flushed at exit. A crash at step N must leave N rows on
disk. Rows carry at minimum: step, the measured objective, wall-clock, and the liveness signal the
confound-immune-system law requires (accepted-batch fraction / `weights_stepped`), so a frozen run
can never read as a converging one.

**B2 — an executable stop controller.** It imports its inputs FROM the ticket (never re-typed
constants), derives a `TrajectoryStopConfig`, consumes the B1 journal, and emits a TYPED decision at
every eligible eval row AND at every boundary. It must provide: ticketed fresh argv, ticketed resume
argv, and durable receipts for BOTH terminal modes (event-stop and cap/timeout).

**REUSE, do not twin.** `tac.optimization.trajectory_stopping` is the canonical surface and
`evaluate_trajectory_stop` already exists there. `CapStopReceipt` already extends it (landed by ca1,
57c87898c2) — the censored-cap receipt surface EXISTS; wire it, don't rebuild it. A parallel
stopping module would be the duplicate-SoT defect.

**B3 — the gate is staircase-blind (M1R4B-F2).** `evaluate_trajectory_stop` treats a flat four-row
tail as `marginal_below_bar`, with no event-gap, sustained-window, or loss/margin input. This
vehicle's descent is *event-punctuated*: it has shown plateau-then-drop. A plateau must not certify
convergence until a re-derived event-free horizon AND a score-relevant facet/loss test are also
flat. Add that, with **both** positive controls: a plateau-then-drop trace (must NOT stop) and a
flat-loss-down trace (must NOT stop).

**B4 — EMA (M1R4B-F4).** The MLX loop has no EMA shadow and checkpoints only live parameters,
against the standing EMA non-negotiable and `ema_decay_run_geometry_v1` (decay follows from
steps/epoch × horizon — NEVER a flat 0.997). There is an existing n32 A/B where the last-eight-
checkpoint mean beat the final checkpoint by 5.8809916e-6 d_seg. Implement and checkpoint a
LawRef-derived EMA shadow; at minimum bind the already-measured tail-average selection protocol.
Compare live/EMA/average on the same CPU facet before adopting either.

**B5 — the schedule constants (M1R4B-F3).** `lr = 2e-7` and the cosine horizon are SOURCE-VERIFIED
from the PR130 repro `train.sh:113` — i.e. verified in ANOTHER vehicle's batch geometry. That makes
them BORROWED CONSTANTS (constants-are-poison; cross-regime transfer is a named recurring defect
here). Re-derive at OUR n120 accumulated-batch geometry, or pre-register a same-object checkpoint
decision that selects a derived schedule. Extension must not induce a horizon-driven LR jump (at the
3250 cap the cosine reaches 2e-9; extending to 6500 would jump it ~50.5×).

## Rules

- **ONE amendment.** Land ALL cures before MAIN re-reviews — any artifact change voids all three
  passes, so trickling cures costs three review rounds each time.
- **Re-derive, don't confirm.** Quote file:line. `UNDETERMINED` with a named missing input beats a
  guess. Label every number MEASURED / DERIVED / INFERRED / ASSUMED.
- **Never invent** CLI flags, API names, or VALUES. Grep `add_argument` / `def` first.
- **Every cure needs BOTH-DIRECTION controls**: it fires on the bad state AND stays silent on the
  good state. A false alarm on the good state trains the reader to ignore the indicator — that
  defect was committed and corrected in this very session.
- **DO NOT FIRE THE BURN.** Do not launch any Metal job. Bounded smokes only, and only if they
  cannot touch the sealed run dir.
- Commit via `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per
  file, tags `[no-triality] [p0-ledger-ok]`, NO Claude/AI attribution and no Co-Authored-By trailer.
  `.py` files need two `review_tracker.py mark-file` passes; `REVIEW_GATE_OVERRIDE=1` is FORBIDDEN
  with `.py`.

## Deliverable

`.omx/research/ddm_m1c1_20260808/M1C1_CURES.md`:

1. Per cure B1–B5: what you built, file:line, the control you ran in BOTH directions, and the
   honesty label on every number.
2. The exact ticket diff (semantic leaf diffs, not a textual diff — `sort_keys` reordering is noise).
3. Anything you could NOT cure, with the named blocker — an honest partial beats a fake complete.
4. `NEXT_IF_RESUMED:` line.

## OPTIMAL FORM

- **Reference form:** a production event-driven training controller — durable append-only journal,
  typed decisions, resume-safe, with the stop criterion derived from the run's own geometry rather
  than inherited from another vehicle's config.
- **Scope reductions (legal):** bounded to the M1 trainer + ticket + the canonical stopping module;
  no new n600 measurement commissioned; EMA/schedule comparisons may use existing banked receipts.
- **Mechanism reductions:** NONE. A stop controller that is wired but never consulted, or a journal
  written at exit, reproduces the exact defect being cured. If you cannot make it executable, say so
  and name the blocker — do not ship a marker.
- **Provenance pins:** ticket `.omx/research/ddm_m1_20260808/launch_ticket_v5_event_driven.json`
  (sha256 `9c8373b5b352cacc…` at review time — re-hash; it WILL change as you cure it, which is
  expected for a builder, unlike the reviewers). Trainer
  `experiments/ddm_mx1_pr130_semantic_renderer.py` (`1ef18faf37e2f171…`). The three round-4 reviews
  are at `.omx/research/ddm_m1r4_20260808/M1R4{A,B,C}_REVIEW.md` — read all three; B and C carry
  detail this charter compresses.
