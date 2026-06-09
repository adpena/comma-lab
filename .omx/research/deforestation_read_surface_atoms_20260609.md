# Deforestation read-surface atom proposers (Phase-D1 primitives) — landing memo

- **Date:** 2026-06-09 (UTC)
- **Agent:** swarm V3-DEFOREST
- **Lane:** `lane_swarm_v3_deforest_read_surface_atoms_20260609`
- **Files landed (DISJOINT ownership):**
  - `src/tac/optimization/scorer_read_surface_atoms.py` (NEW)
  - `src/tac/optimization/tests/test_scorer_read_surface_atoms.py` (NEW, 24 tests)
- **Authority:** `[macOS-CPU advisory]` / `planning_control_false_authority`.
  Every atom carries `score_claim=False`, `promotable=False`,
  `promotion_eligible=False`, `rank_or_kill_eligible=False`,
  `ready_for_exact_eval_dispatch=False`. Exact authority is the B2 paired
  upstream `evaluate.py` eval on contest hardware; these atoms are PROPOSALS.

## Operator directive grounding

Operator 2026-06-09: *"give everything what it needs and no more; slash-and-burn
to the skeleton of only what's necessary in the optimal format and
quantization."* The contest scorer reads MINIMALLY; everything it does NOT read
is its NULL SPACE => burnable. This landing turns the three code-grounded
read-surface facts into typed *deforestation atoms*.

**ALL read-surface constants are sourced from `tac.contest_eval_contract`** (which
pins + drift-checks the snippets against live `upstream/modules.py`). Nothing is
hardcoded in this module. Verified at import: `SEQ_LEN=2`,
`PUBLIC_TEST_FRAME_COUNT=1200`, `PUBLIC_TEST_PAIR_COUNT=600`,
`scored_frame_index_within_pair=1`, `unscored_frame_index_within_pair=0`,
`rate_price_per_archive_byte=25/37_545_489`.

## B-BEFORE-D discipline (Hotz council SEAL)

This landing is the Phase-D1 PRIMITIVES + atom proposers ONLY. It does NOT build
the live waterfiller loop — that waits for the B2 base archive. The primitives
also feed B1's score-aware loss weighting (the per-pixel margin tolerance is a
SegNet loss reweighting prior) and the post-B2 waterfiller (the atoms promote to
base-bound `CandidateActionEvaluation`).

## The three deforestation levers (atom proposers)

### Atom 1 — `segnet_argmax_margin_tolerance_map` (per-pixel waterfilling)

Read-surface fact: SegNet distortion = `mean(argmax(gt)!=argmax(comp))` over 5
classes (`upstream/modules.py::SegNet.compute_distortion`; contract snippet
`segnet_argmax_flip`) => argmax-ONLY => per-pixel precision is set by the top-2
logit margin `m = top1 - top2`.

- Computed from REAL SegNet logits via the EXACT contest path:
  `segnet(segnet.preprocess_input(pair))` -> `(B,5,384,512)` logits, the same
  tensor `compute_distortion` argmaxes over `dim=1` (== `scorer.py:410`,
  `tac.scorer` line ~410). The helper takes the logit tensor as input so the
  CALLER controls the path; it does NOT load `mini_scorer` (a 25K-param
  surrogate = proxy, forbidden as authority).
- Classifies each pixel PROTECTED (small margin, near a SegNet decision boundary)
  vs FREE (large margin, class interior). The per-pixel tolerance == the margin.
- `segnet_margin_tolerance_atom(...)` proposes shaving RGB precision from FREE
  pixels of the SCORED (2nd-of-pair) frame; advisory `d_seg` delta is 0.0 by
  construction (FREE-pixel argmax does not flip).

**Deforestation target:** RGB precision on class-interior pixels of scored frames
(the per-pixel quantization waterfill).

### Atom 2 — `seg_scored_frame_mask` (the biggest deforestation)

Read-surface fact: SegNet reads ONLY `x[:, -1]` (`upstream/modules.py`:
`x = x[:, -1, ...] # Use only last frame`; contract snippet
`segnet_last_frame_only`) => the FIRST frame of each pair carries ZERO d_seg
signal.

- For non-overlapping `seq_len==2` batching: SCORED frames = odd global indices
  (1,3,5,...), SEG-FREE frames = even indices (0,2,4,...). **~600 of 1200 frames
  are seg-free.**
- `seg_free_frame_atom(...)` proposes freeing the SegNet-fidelity budget of every
  seg-free frame (advisory `d_seg=0.0` by construction; advisory total score
  delta == the rate-term reduction `-25*bytes_freed/37_545_489`). The only
  remaining constraint on a seg-free frame is PoseNet fidelity (Atom 3).

**Deforestation target:** whole-frame SegNet fidelity on 1st-of-pair frames (the
single biggest burnable lever).

### Atom 3 — `pose_null_projection` (Jacobian null space)

Read-surface fact: PoseNet scores only the first 6 of 12 pose dims
(`upstream/modules.py`: `out1[h.name][..., : h.out // 2] - out2[...]`; contract
snippet `posenet_first_six_dims`) on 2-frame YUV6 => half the head is null space.

- Consumes `tac.scorer_exploits.compute_scorer_jacobian` +
  `project_to_scorer_null_space` READ-ONLY (does NOT reimplement them). Returns
  the component of a candidate RGB perturbation that lives in the scorer
  Jacobian's null space — invisible to the scored pose-6 AND the sampled SegNet
  outputs (a conservative joint null space).
- `pose_null_atom(...)` proposes spending the null-space energy as
  scorer-invisible RGB precision shaving; advisory pose+seg deltas 0.0 by
  construction. Particularly relevant on SEG-FREE frames where pose is the only
  constraint.

**Deforestation target:** RGB directions in the scorer Jacobian null space.

## How each atom composes into the post-B2 waterfiller

Each `DeforestationAtom.to_row()` carries `contract_provenance` naming
`promotes_to = tac.optimization.evaluator_action_waterfill.CandidateActionEvaluation`.
Verified the atom's rate denominator (37_545_489, from the contract) is identical
to `evaluator_action_waterfill.CONTEST_ARCHIVE_RATE_DENOM`, so the units match.

Post-B2 promotion path (per atom):

1. Apply the atom's proposed precision shave / byte free to the B2 base archive
   -> produce a candidate `archive.zip` (`with_action_archive_sha256`).
2. Run the paired upstream eval to get exact `(d_seg, d_pose, bytes)` for base
   AND candidate (`d_seg_base/d_pose_base/bytes_base` +
   `d_seg_with_action/...`).
3. Construct `CandidateActionEvaluation(base_archive_sha256=<B2 sha>, ...)`. The
   rent law (`pays_rent`) admits the atom iff `delta_score_total < 0` AND
   `scorer_effect_survived`. A byte-FREEING atom (Atom 2; `delta_bytes < 0`) that
   holds/lowers score is unconditionally admissible (`value_per_byte == +inf`,
   sorted first).
4. `waterfill_select_actions(...)` ranks survivors by exact value-per-byte.
   Because SegNet/PoseNet are noncommutative, the caller re-measures + re-selects
   after each accepted atom (`requires_recompute_after_accept=True`); accepted
   atoms make remaining evaluations `is_stale_for_base`.

The advisory deltas these atoms carry are the PLANNING-CONTROL prior that ranks
which atoms to evaluate FIRST (cheapest exact-eval budget on the
highest-advisory-value atoms), exactly the anti-drift discipline the waterfill
module enforces.

## B1 score-aware-loss composition (also feeds the loss weighter)

The Atom-1 margin tolerance map is a per-pixel SegNet-loss reweighting prior:
PROTECTED (small-margin) pixels get HIGH loss weight (protect the argmax),
interior FREE pixels get LOW weight (let the renderer spend bits elsewhere). The
seg-free frame mask (Atom 2) tells B1 to set the SegNet loss weight to ZERO on
1st-of-pair frames. These are exposed via the same typed helpers so B1 can
consume them without re-deriving the read surface.

## NO-FAKE test discipline (24 tests, all behavioral)

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" Slot EEE class 2 (a test that still
passes if the function returns canonical markers is FAKE). The causal cores:

- **Atom 1** `test_margin_map_predicts_argmax_flip_exactly`: bumping a pixel's
  2nd-place class by `0.5*margin` NEVER flips the argmax; by `margin+1e-2` ALWAYS
  flips; at a `mid` delta a FREE (large-margin) pixel tolerates it while a
  PROTECTED (small-margin) pixel flips. This is the EXACT mathematical claim the
  tolerance map makes about the contest argmax surface — a constant-margin fake
  fails it.
- **Atom 1 (real path)** `test_margin_map_from_real_segnet_exact_path` +
  `test_margin_map_uses_only_last_frame_of_pair`: real upstream SegNet via the
  exact path produces a non-degenerate margin spread; the winner mask equals the
  contest argmax; mutating pair frame_0 leaves the SegNet logits IDENTICAL
  (frame_0 is seg-free) while frame_1 changes them.
- **Atom 2** parametrized last-of-pair invariant across `seq_len ∈ {1,2,3,4}` (a
  fixed-list fake fails); byte-freeing scales with seg-free count.
- **Atom 3** `test_pose_null_projection_lies_in_scorer_jacobian_null_space`:
  with REAL scorers, `||J @ projected|| = 2.9e-4` vs `||J @ raw|| = 0.38`
  (~1300x smaller) while `||projected|| > 0` (a real direction) and
  `||projected|| <= ||raw||` (a component of the input). A zeros-fake fails the
  nonzero check; a raw-passthrough fake fails the null-space check.
- **False authority** every atom row carries the planning-control markers; the
  advisory rate delta equals exactly `25*bytes/37_545_489`.

`24 passed in ~22s`. ruff clean. Real-scorer tests skip cleanly if upstream
model safetensors are absent.

## 6 unified-solver wire-in hooks (per CLAUDE.md "Subagent coherence-by-default")

1. **Sensitivity-map contribution** — ACTIVE (conceptual). The Atom-1 per-pixel
   margin tolerance IS a SegNet sensitivity surface (boundary pixels = high
   sensitivity); the seg-free frame mask is a per-frame sensitivity gate (zero
   SegNet sensitivity on 1st-of-pair frames). Exposed via typed helpers; a
   `tac.sensitivity_map` consumer can ingest `summarize_read_surface()` +
   `SegnetMarginToleranceMap.margin` directly. (No edit to `tac.sensitivity_map`
   — DISJOINT file ownership; flagged for a downstream wire-in.)
2. **Pareto constraint** — ACTIVE via the rent law. Each atom's
   `advisory_delta_score` / `advisory_delta_rate_score` is a (rate, distortion)
   point the post-B2 waterfiller's `CandidateActionEvaluation` plots against the
   exact contest Pareto frontier; byte-freeing atoms move strictly toward the
   frontier.
3. **Bit-allocator hook** — ACTIVE (PRIMARY). `precision_bits_shaveable` per atom
   is exactly the bit-allocator's input: how many RGB bits the codec can shave
   per pixel/frame/direction. This is the core deliverable.
4. **Cathedral autopilot dispatch hook** — N/A at D1. These are planning
   primitives, not an archive-deployable substrate; the dispatch hook attaches at
   the post-B2 waterfiller, not here.
5. **Continual-learning posterior update** — N/A at D1 (no empirical anchor; all
   outputs are advisory by construction). The posterior update fires when the
   post-B2 paired eval lands the exact ΔS per atom.
6. **Probe-disambiguator** — N/A. The atoms are not 2+ defensible
   interpretations; they are read-surface facts pinned by the contract. The
   `protected_quantile` vs `protected_margin_threshold` knob is a single
   continuous parameter the downstream allocator sweeps, not a disambiguation.

## Notes for sister agents (no edits made to their files)

- `tac.sensitivity_map` could ingest the margin tolerance map + seg-free frame
  mask as a canonical SegNet read-surface sensitivity contribution (Hook 1).
- The post-B2 waterfiller loop should consume `DeforestationAtom` rows, promote
  each to `CandidateActionEvaluation` via the paired eval, and run
  `waterfill_select_actions` with re-measure-after-accept (the atoms are
  noncommutative — Atom-1 precision shaves on a frame interact with Atom-3
  pose-null shaves on the same frame).
