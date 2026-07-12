# UGC terminal-polish A/B implementation contract — 2026-07-12

`lane_id=lane_ugc_terminal_polish_ab_396_400_20260712` · `research_only=false`

## Objective and authority

Extend `src/tac/through_r/mc_finisher.py`; do not create a second finisher.  The
finite action is a direction-pinned edit list supplied by the existing #400
pair-local diagonal apparatus.  Bit `b_i=1` applies edit `i`; `b_i=0` keeps the
frozen base cell.  All optimizer arms consume the same edit list, seed family,
function-evaluation budget, exact objective, and strict monotone acceptance gate.

The measurement fixture is the completed, STOP-sealed
`experiments/results/click_polish_399_campaign` archive and its 600-pair frozen
CPU-torch authority arrays.  The active
`experiments/results/click_polish_399_import` campaign is read-only and is not a
fixture.  No paid dispatch and no process/run mutation are permitted.

For pair-local edits on distinct pairs, a mask evaluation is the exact n600 score:

`S(b) = 100 mean(d_seg_i(b_i)) + sqrt(10 mean(d_pose_i(b_i))) + 25 B(b)/37_545_489`.

Each base/edit cell is measured through the frozen CPU-torch scorer in canonical
16-pair layout, and `B(b)` is the actual deterministic repack byte count.  This is
the already-proved #400 splice identity, not an additive-score approximation: the
nonlinear Pose term is applied only after the exact 600-pair mean.  Each arm's final
mask is independently re-rendered/re-scored and compared to the cell-composed value.
Rows are `[macOS-CPU advisory]`, `score_claim=false`, and non-promotable; they do not
move the contest pointer.

## Estimators (minimize expected exact S)

Let `p=sigmoid(phi)`, `K` be support size, and `f(b)=S(b)`.

- DisARM: antithetic Bernoulli masks with the unbiased logit-gradient normalization
  `0.5 (f(b)-f(b_tilde)) (-1)^b_tilde 1[b!=b_tilde] max(p,1-p)`.
- bitflip-1: select `j` uniformly, evaluate `b` and `flip_j(b)`, and emit only
  `K p_j(1-p_j) (-1)^b_j (f(flip_j(b))-f(b))` at `j`.
- UGC: coordinatewise use bitflip-1 when
  `min(p_i,1-p_i) < tau`, otherwise DisARM, with paper-standard
  `tau=1/(2K)`.  The selected bitflip coordinate is uniform over all `K`
  coordinates; this preserves unbiasedness when only boundary coordinates consume
  its component.
- RLOO: two independent masks and the two-sample leave-one-out logit gradient
  `0.5 (f(b)-f(b')) (b-b')`.
- Exact enumeration: enumerate all `2^K` masks on the registered small-support
  block and choose the best exact mask.
- Existing control: discrete `(1+1)-ES` mutation of the current mask with the same
  exact ratchet and the same function-evaluation budget.

UGC/DisARM/RLOO convert the stochastic gradient only into a proposed bit flip.  A
proposal changes state only when exact `S(candidate) < S(current)`; estimator scale
never bypasses the ratchet.

## Matched-budget measurement

Use a representative six-pair active block (`K=6`), selected deterministically from
the completed campaign's next unpolished 48-pair block after a shared diagonal
direction sweep.  `2^K=64`, so every arm receives 64 exact objective calls for the
variance receipt and a separate 64 exact objective calls for search.  Report:

1. gradient trace-variance at the shared mixed boundary/interior probability vector;
2. exact `delta_S` after 64 search calls and `-delta_S / 64` improvement per call;
3. wall-clock for variance, search, and independent final-mask verification;
4. exact function-evaluation count and final-mask verification residual.

For `(1+1)-ES`, gradient trace-variance is `N/A` because it has no unbiased gradient
estimator; report its exact proposal-gain variance separately rather than laundering
that statistic into the gradient table.  Exact enumeration has zero estimator
variance by construction.

Verdict scope is this archive, six-pair candidate block, proposal construction,
probability vector, seed set, and 64-call budget.  A UGC loss is an
instance/formulation-scoped result, never a family-level kill.

## Required verification and landing

- Positive, negative, boundary-switch, budget-exhaustion, strict-ratchet,
  determinism, resume, and exact-enumeration tests.
- UGC unbiasedness: Monte Carlo mean against brute-force exact logit gradient on a
  tiny synthetic Bernoulli objective containing interactions.
- Focused pytest, repository ruff, and repository type checker (`ty`; run mypy too
  only if the project config supports it without inventing flags).
- Measured law in `src/tac/canonical_equations/` plus canonical registry event and
  a DAG FEED row.  Add a witness-DSL lever only if UGC wins the scoped default-routing
  criterion against both `(1+1)-ES` and DisARM.
- Mark every edited `.py` with the review tracker and commit exact owned paths using
  `tools/subagent_commit_serializer.py` with post-edit SHA-256 values.

