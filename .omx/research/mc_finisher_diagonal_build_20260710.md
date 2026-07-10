# #400 pair-local DIAGONAL mode — BUILD-ONLY landing (the witness click-polish exploit) — 2026-07-10

`research_only=true` · **BUILD-ONLY** (no launch, no measurement — a dry-start/relaunch owns the
machine; the n600 diagonal-polish measurement fires post-launch at the terminal band per deferral
**D27b, ARMED**). Pointer contest-CPU **0.19108282 UNMOVED** (this build is MEANS; only a byte-closed
`upstream/evaluate.py` n600 diagonal-polish row < the pointer moves it).

**STORES CONSULTED:** `.omx/research/clickpolish_to_witness_design_20260710.md` (THE spec — the
unify-into-#396 recommendation §3.2, the per-vehicle clickable-code inventory §1, the pair-locality
proofs with code cites §2, the 4a′/4c′ terminal slots §3.1, the build list §7) · `src/tac/through_r/
mc_finisher.py` (#396 — the exact-gated finisher READ IN FULL; its `FinisherProblem` injected-callable
seam is the seam that makes the diagonal a MODE, not a fork) ·
`.omx/research/ADVISORY_pr128_hnerv_reverse_engineering_sdf_transfer_v753_v8_20260710.md` §9.2 (the J(a)
generalized-finisher contract — 8 points folded) · `src/tac/click_polish.py` (#399 sibling — REFERENCE
ONLY for the diagonal-batch / `verify_pair_locality` / `verify_batch_equivalence` / per-pair-accept +
bisect pattern; NOT edited — different substrate, stays separate per §3 SCOPE NOTE) ·
`src/tac/through_r/harness.py` (`measure_through_r` already returns `per_pair_dseg` — build-item 2 was
ALREADY satisfied; no harness edit) · `tools/levelset_byte_close_and_eval.py` (the 4c′ decode surfaces
`parse_pose_carrier`/`serialize_pose_carrier`/`pose_carrier_confirm` — imported READ-ONLY at committed
HEAD, NOT edited) · `src/tac/canonical_equations/morse_smale_stratified_parallax_dpose_20260708.py` +
`fullstack_home_assignment_20260710.py` + `.omx/research/r1_dxi_shippability_byteclose_20260708.md` +
MEMORY L68 (the banked R1 dxi d_pose 0.001610 / 0.127 contribution — the ξ rollback floor's
value-provenance) · `.omx/state/deferral_ledger.md` D27b (the ARMED terminal-solve queue this tool joins)
· `.omx/research/ADVISORY_sdf_scorer_waterfill_20260710.md` (the coordinator SIGNAL — #400 = the
pair-local tier of the hierarchical waterfill) · CLAUDE.md NO-FAKE + resumability-P0 + value-provenance
ladder.

## What landed (files + surfaces)

- `src/tac/through_r/mc_finisher.py` (+~520 LOC, a MODE appended — the #396 classes untouched):
  - **`PairLocalDiagonalFinisher`** — the driver. Deterministic column/δ diagonal sweep: for each
    `(col, δ)`, apply the click to that column across ALL pairs, ONE CONFIRM render measures the
    per-pair distortion VECTOR (this scores `n_pairs` independent candidates in one render — the
    exploit), each pair records its best improving click; then a JOINT accept (≤1 click/pair) is
    re-measured on the CONFIRM (canonical-layout) authority with a strict monotone-S ratchet + bisect
    salvage. RNG-free ⇒ resume is exact.
  - **`DiagonalProblem` / `DiagonalObjective`** — the injected `render_measure_fn` / `probe_frames_fn` /
    `byte_cost_fn` seam (mirrors #396's `FinisherProblem` decoupling). `DiagonalObjective.axis_s_component`
    computes the controlled S-contribution (`100·d_seg + rate` for the code axis; `√(10·d_pose) + rate`
    for the ξ axis — the OTHER distortion cancels, from canonical `tac.contest_score` terms).
  - **`LocalityGuardError` + `require_locality`/`verify_locality`** — the FAIL-CLOSED 2-pair cross-talk
    probe (perturb pair a, assert pair b's frames byte-identical AND pair a non-vacuously changed on the
    ACTUAL render). Run before any batch is trusted; raises on cross-talk OR a vacuous probe.
  - **`make_through_r_code_measure`** — the d_seg/code axis `render_measure_fn`, wired to
    `tac.through_r.harness.measure_through_r` (its `per_pair_dseg` IS the diagonal vector — NO harness
    change was needed). CPU-locked (the harness refuses MPS/MLX).
  - **`make_byte_close_xi_pose_measure` + `load_byte_close_pose_surfaces`** — the **4c′ ξ-terminal entry
    point**: a READ-ONLY import of the committed-HEAD byte-close decode surfaces
    (`parse_pose_carrier`/`serialize_pose_carrier`/`pose_carrier_confirm`) composed into a per-pair
    d_pose measure via the frozen CPU-torch PoseNet (the authority; never MPS). The heavy full-inflate
    fires post-launch at the terminal band — this factory binds the SEAM only.
  - **`BANKED_R1_DXI_DPOSE_FLOOR = 0.001610` + `load_banked_r1_dxi_dpose_floor`** — the ξ rollback floor
    (never ship worse than the banked pose guarantee). Value-provenance ladder MEASURED-ANCHOR: the
    loader re-reads the n600 authority value from the r1_dxi byte-close memo and asserts agreement
    (drift-guarded); `√(10·floor)=0.1269` == the banked 0.127 contribution. NOT a bare literal.
- `src/tac/through_r/tests/test_mc_finisher_diagonal.py` (NEW, 28 tests, n4 CPU fixtures, ~0.2 s).

## The five discipline requirements (design memo §7 build-item 1) — all realized

| requirement | realization |
|---|---|
| (a) LOCALITY GUARD, fail-closed | `require_locality` — 2-pair probe; raises `LocalityGuardError` on cross-talk (negative test: `_nonlocal_probe` ⇒ refuse) or vacuity |
| (b) accept on EXACT per-pair contribution via canonical re-render | `_accept_joint` re-measures the JOINT candidate through CONFIRM in the canonical layout; the ratchet reads that exact S, NEVER a per-move sum |
| (c) rate-aware, REAL bytes | injected `byte_cost_fn` folded into `DiagonalObjective.archive_bytes`; test asserts `== os.path.getsize(real_encoded_file)` |
| (d) monotone ratchet + rollback vs pinned floor | strict-S ratchet + `_floor_ok` refuses any d_pose accept above `BANKED_R1_DXI_DPOSE_FLOOR` (ACCEPT-LAYER rail — the sweep ranks by distortion, so the floor guards a byte-driven pose sacrifice) |
| (e) resumable accepted-moves JSONL (P0) | `run(log_path=…)` appends per-round rows (fsync'd); `resume_from_ledger` replays them onto the original table; + atomic table npz snapshot |

## Test results (28, all green; ≥18 required)

- **locality guard**: refuses a deliberately-nonlocal problem (CROSS-TALK), passes on a local one,
  refuses a vacuous probe, refuses when no `probe_frames_fn` injected.
- **diagonal ≡ sequential** on n4 (d_seg) + on n4 (d_pose) — one diagonal render == 4 single-pair renders.
- **monotone ratchet** never regresses (d_seg); converges to target; pairs_touched tracked.
- **rollback floor** rejects a byte-win that worsens d_pose below the banked guarantee; without the floor
  the same move is accepted (the guard is the only difference).
- **rate accounting == `os.path.getsize` of the real re-encoded file.**
- **ξ floor value-provenance**: matches the byte-close memo; `√(10·floor)==0.127`; a drifting memo RAISES;
  a missing memo returns the cached constant.
- **resume round-trip** (JSONL replay reconstructs the exact table + round counter); atomic snapshot, no
  leftover tmp; JSONL rows carry per-component deltas + `score_claim=False`/`promotable=False`.
- **confirm authority** asserted when the measure lies (P9); **bisect** salvages the good half; **plateau**
  stop; **determinism** (identical runs); table/objective/problem **validation**; **column subset**
  restricts the sweep; **4c′ seam** (surfaces load, both factories bind a callable).

## Waterfill-contract fold (coordinator SIGNAL, 2026-07-10)

Per `.omx/research/ADVISORY_sdf_scorer_waterfill_20260710.md`: **#400 implements the PAIR-LOCAL TIER
(steps 1-3, and critically step 6) of the hierarchical interaction-aware water-filler.** The JOINT accept
is EXACTLY re-verified through the real CONFIRM and the ratchet reads that exact S — per-move deltas are
NEVER summed as the final claim (the no-local-gains-additivity law). The accepted-moves ledger now records
SEPARATE per-component deltas (axis distortion / rate-term / bytes, each `{before, after, delta}`) so a
future interaction-aware selector (steps 4-5: cross-class interaction estimate + budgeted compatible-set
selection) can consume the ledger without re-deriving. Steps 4-5 are the OUT-OF-SCOPE upper tier.

## Registration + honest gaps

- **TOOL, not a Lever** (per #396's own classification, design memo §7 item 4): it holds no swept trainer
  flag the argparse cannot supply, so the DSL holds NO leg to drift.
- **Activation ledger**: the ledger's event vocab is `{fired, measured, retired}` and its
  `known_levers()` is DSL-lever-scoped (AST over factory names). A never-fired TOOL has no supported row
  (`record_activation` would require a lie or pollute a lever-scoped surface). Honest category: the tracked
  duty-to-measure queue for this tool is **D27b (ARMED)** — its on-disk owner line now points at
  `src/tac/through_r/mc_finisher.py::PairLocalDiagonalFinisher`. (This matches #396's precedent — no ledger
  row.)
- **4c′ dependency**: the ξ measure imports `tools/levelset_byte_close_and_eval.py` at committed HEAD
  (read-only). If the receiver-hardening sibling (#402) is editing that file concurrently, this binds to
  its committed HEAD; the surface contract (`parse_pose_carrier`/`serialize_pose_carrier`/
  `pose_carrier_confirm`) is checked at import and raises a clear reconcile error if it changed. NOT edited
  by this task.
- **Honest gaps (unmeasured, by design — BUILD-ONLY):** every ΔS band in the design memo (§6) is
  ESTIMATED; NO witness diagonal-polish row is measured. The equations leg is DEFERRED-to-first-measured-row
  (candidate `pair_local_diagonal_click_polish_{dseg,dpose}_v1`). The 4c′ byte-close d_pose path binds the
  seam but its per-pair-vector shape (`d_pose_per_pair`) is only present if `pose_carrier_confirm` exposes
  it; today it returns the aggregate + a broadcast fallback — the per-pair vector for the ξ diagonal is an
  owed follow-up at measurement time (the code axis's per-pair vector IS real via the harness). This is
  stated, not hidden.

## Triality legs

- **DAG**: `### FEED-400-diagonal-build` appended to `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **DSL**: N/A-TOOL (stated above — no swept trainer flag; nothing to drift).
- **equations**: DEFERRED-to-first-measured-row (stated; every band ESTIMATED — no anchor mints until a
  byte-closed n600 diagonal-polish row lands).

**Pointer contest-CPU 0.19108282 UNMOVED — BUILD-ONLY, MEANS. The measurement fires post-launch at the
terminal band (D27b).**
