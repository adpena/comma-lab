# Off-the-shelf lift wave — MAIN element-level fractal audit (2026-08-06)

Operator demand: "sure that nothing is not even toy and that you're digging deep and
recursively." Standard: fractal_audit_standard (single-pass = PARTIAL; element level;
calibration-lineage recursion; self-audit round). Owner MAIN. score_claim=false.

## Leg 1 — hb1 HPAC host chain: FAITHFUL at element level (verified at source)

- Calibration-lineage RECURSION CLOSED: driver flags diffed against repro_repo/scripts/
  e2e.py:1220-1247 — ALL match (e60/b8/eb4/lr .003/lr-exp 2e-4/lr-bits .01/bit-eps 1e-6/
  rate-lambda 1/qat .5/init-bits 8/seed 20260716). Key dissolve: e2e.py:1222 shows THEIR
  self-compress stage warm-starts from hpac_p64 — my archive-extract init IS their
  terminal-stage form, not a deviation. Their random-init is stage 33 (first HPAC train),
  not this stage.
- Named residual deltas (documented, legal): (a) --device cpu — substrate; torch RNG
  streams differ CPU-vs-CUDA so the run is recipe-faithful but not their-draw-faithful
  (irrelevant to a race, relevant to any reproduction claim); (b) payload = OUR labels
  (the race itself); (c) their p64 was trained on their token distribution — a fresh
  stage-33-style random-init control is the OPTIONAL follow-up if the converged number
  is contested. NOT a toy: n600 full population, exact-decode gate, model bytes counted.
- Wall-clock element: epoch 0 emitted at ~0 min, no epoch-2 row after ~25 min ⇒
  >~12 min/epoch CPU ⇒ ~12h+/payload. ACCEPTED (detached, nice-10, resumable every 2
  epochs). If the mx-family MLX port later covers the HPAC trainer, re-race faster.
- Epoch-0 numbers are the trainer's ESTIMATES (estimated_* fields); exact bytes come
  only from stage 3 pack + stage 4 encode/--require-exact decode. No byte claim before.

## Legs 2+3 — mx1/mx2 port arms: toy-seam risks NAMED + amendment placed

CHARTER_AMENDMENT_MAIN.md placed in both receipt dirs, BINDING at landing review:
(1) parity on REAL frames only — synthetic-tensor parity = NO-FAKE #3, will be refused;
(2) #855 default-MLX-conv-adapter 76-argmax-flip hazard must be named + counted;
(3) #903 loss-scalar-parity ≠ gradient-parity — one per-tensor gradient check or scoped
claim; (4) batch-shape instrument pinning in every parity receipt; (5) inits cite e2e.py
lines (no invented defaults). Landing review = the enforcement point (gen=1
review_required; arms may not see mid-flight files).

## Leg 4 — et4 chain: audited this window (batch-seam diagnosis memo); no new elements.

## Self-audit round (what THIS audit might miss)

- The amendments rely on landing-review enforcement, not in-flight consumption — if an
  arm finishes with synthetic parity, the cost is one respawn cycle, not a false verdict.
- e60-at-their-form ≠ e60-optimal-for-our-labels: the race measures their recipe on our
  payload, not the payload's optimum. If HPAC wins the race, a horizon/λ sweep is the
  optimal-form follow-up before any composed-vehicle byte claim cites the number.
- The epoch-0 −12,727 B transfer is init-attributable; only the CONVERGED+packed number
  enters the race table.
