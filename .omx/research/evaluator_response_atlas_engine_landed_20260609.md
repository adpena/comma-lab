# EVALUATOR RESPONSE ATLAS ENGINE — landed 2026-06-09

**Subagent:** `atlas_engine_mlx_jacobian_20260609` (task #36).
**Evidence grade:** `[macOS-CPU advisory]` (scorer forwards) + `[macOS-MLX
research-signal]` (cross-video reduce). Mechanism-only. No score claims; no
dispatch. $0 local, NO cloud, NO paid GPU, NO MPS.
**Frontier at landing** (orphan inventory `da62505aa`): contest-CPU
**0.19198533** (archive `b7106c9b…`, 178,493 B). The atlas does not change the
frontier; it is a *targeting index* downstream actuators (#46 waterfiller,
rate-attack repair, PR110++ selector) consume to spend bytes where the scorer is
fragile/free instead of sweeping uniformly.

## What landed

The **missing per-pair scorer-sensitivity INDEX over the full 600-pair contest
video**. The contest objective is an evaluator quotient
(`100·d_seg + sqrt(10·d_pose) + 25·archive_bytes/37_545_489`), so the binding
question for every byte that touches a frame pair is: *where, across the 600
pairs, does the scorer-fragile mass concentrate, and where is the free budget?*
Answering it requires the per-pair PoseNet pixel-Jacobian + SegNet argmax-flip
margin field for ALL 600 pairs. The #35 cone producer computed those fields for
a single pair on demand; nothing materialised or indexed them across the whole
video. This engine does.

- **Core (the typed INDEX):**
  `src/tac/optimization/evaluator_response_atlas.py` —
  - `AtlasPairRow` (schema `evaluator_response_atlas_pair_row.v1`): one index
    row per pair = `{pair_index, SegMarginFieldStats, PoseJacobianFieldStats,
    JointConeSummary, sensitivity_refs (cone-map path + sha + spectral cell ref +
    exploit-atom families), per_region}`. **Pointers + reduced stats; NO copied
    tensors** — the full per-pixel fields live in the cone-map `.npz` (which the
    LF waterfiller `Frame1ConeMap` already reads).
  - `EvaluatorResponseAtlas` + the **query surface**: `by_pair`,
    `top_budget_pairs(k)`, `most_fragile_pairs(k)`, `pose_bound_pairs(thr)`,
    `by_region_budget(class)`, `by_family(family)` — the consumer API the
    waterfiller / rate-attack / PR110++ selector use to TARGET pairs/regions.
  - JSONL persistence (`to_jsonl_lines` / `from_jsonl_lines`): header (schema +
    headline + provenance) + one line per pair row. The persisted INDEX is small
    (each row < 4 KB; no tensor inlined).
- **MLX compute kernel for the cross-video reduce:**
  `atlas_cross_video_reduce_mlx` (Apple unified-memory column reductions + Gini
  sort) **with the canonical numpy reference** `atlas_cross_video_reduce_numpy`
  (Catalog #383 Backend pattern). MLX-first via `atlas_cross_video_reduce`
  (numpy-portable fallback). The two agree to fp32 tolerance (test
  `test_mlx_and_numpy_reduce_agree_to_fp32_tolerance` ran the real MLX path).
- **CLI / 600-pair runner:** `tools/build_evaluator_response_atlas.py` —
  per-pair cone compute (real CPU-torch scorers + differentiable-YUV6 patch +
  fail-closed-on-zero-Jacobian guard) → atlas row → cone-map `.npz` on the SSD
  tier → cross-video MLX reduce → atlas JSONL index. Durable per-pair progress
  JSONL + `--resume` for crash-recovery; run detached (nohup) per the
  durable-daemon doctrine.
- **Tests:** `src/tac/tests/test_evaluator_response_atlas.py` — 24 fast + 1 slow
  real-scorer NO-FAKE proof. The MLX/numpy parity, query surface, JSONL
  round-trip, fail-closed contracts, and cathedral-bridge wiring are all
  behaviorally verified.

## Which compute path each quantity used (apples-to-apples, NEVER MPS)

| Quantity | Path | Why |
|---|---|---|
| SegNet argmax-flip margin field (per pair) | **CPU-torch** | real SegNet EfficientNet-B2 forward + boundary-slope backward; the only contest-faithful scorer is torch |
| PoseNet frame1-channel pixel-Jacobian field (per pair) | **CPU-torch** | real differentiable PoseNet backward through the patched (differentiable) `rgb_to_yuv6`; fail-closed if the Jacobian is identically zero (severed-gradient signature) |
| joint cone radius / per-region aggregates (per pair) | **CPU-torch** (via #35 producer) | reuses `compute_frame1_joint_safe_cone` |
| cross-video reduction (600 → headline: means/std/Gini/top-K) | **MLX unified memory** | embarrassingly-parallel reduce that saturates the M5 Max 128GB; numpy reference is the portability oracle |

MPS was NEVER used. Per the operator binding, MPS corrupted 95.5% of the
frontier's mode picks; any MPS-derived table is contamination. The atlas row
carries `compute_path="cpu_torch"` so promotion-honesty is structural — a
`compute_path` of `mps`/`cuda` is fail-closed rejected at row construction.

## REUSE (no-duplicative-code; orphan inventory 2026-06-09)

The atlas is an INDEX over existing surfaces, not a new registry (REUSE PLAN
task #36):

- **Per-pair score-effect fields** — `compute_frame1_joint_safe_cone`
  (`tac.optimization.frame1_joint_safe_cone`, #35). The engine orchestrates it
  across 600 pairs and summarises; it does not re-measure the scorer.
- **Atom score-unit row / admission law / vocabulary** — referenced by name:
  the `tac.analysis.action_effect` IR, `tac.optimization.evaluator_action_waterfill`
  admission law, and the `tac.contest_exploits` families (`ATLAS_EXPLOIT_ATOM_FAMILIES`).
  The atlas `by_family()` query JOINs to `tac.contest_exploits.<family>`; it does
  NOT re-author the score row or the law.
- **Spatial budget surface** — the cone-map `.npz` the LF waterfiller
  `Frame1ConeMap` already reads. The atlas references each map by path + sha256.
- **Difficulty-atlas cathedral consumer** — extended, not duplicated. The new
  `consume_evaluator_response_atlas` bridge in
  `tac.cathedral_consumers.per_pair_difficulty_atlas_consumer` emits the same
  `[predicted]` Tier-A payload shape as the existing byte-gradient path, ranking
  pairs by SCORER-FRAGILITY (`fragile_fraction · (1 + pose_binds)`) — the
  orthogonal complement to the byte-gradient `||g_p||` ranking. No mutation of
  the existing path; the 15 existing consumer tests pass unchanged.

## 600-pair headline (`[macOS-CPU advisory]` scorer fields + `[macOS-MLX research-signal]` reduce)

Artifacts (durable SSD; deterministically rebuildable from the CLI):
`/Volumes/VertigoDataTier/pact/evaluator_response_atlas_20260610T001515Z/`
— 600 cone-map `.npz` (2.4 GB; sha-cited manifest), atlas JSONL index (1.9 MB),
per-pair rows index + progress JSONL (crash-resume). 22.6 min wall-clock,
~2.3 s/pair on the M5 Max CPU-torch path; reduce ran `mlx_unified_memory`.

- **video_pose_binds_fraction: 0.7286** (per-pair range 0.629–0.852) — across
  the WHOLE video, the POSE budget (not SegNet) is the binding half-cone for
  ~73% of frame1 pixels. The #35 8-pair estimate (0.731) generalizes to all 600
  pairs: the CLAUDE.md marginal-value flip is a video-wide regime, not a local
  artifact. Pose-targeted lanes dominate the marginal byte everywhere.
- **video_usable_budget_fraction: 0.4655** — ~47% of frame1 pixels carry ≥ ½
  uint8 step of joint perturbation budget on average; **total_free_budget
  77,554,376** (sum of usable cone radii over all 600 pairs, scorer units).
- **Fragile mass is nearly UNIFORM across pairs (Gini 0.0585)** — per-pair
  seg-margin fragile fraction is tightly banded (mean 0.0138, std 0.0015,
  max 0.0222). No single "killer pair": the fragile boundary set is a
  per-pair-local structure, so protection must be spatial (per-pixel cone),
  not pair-exclusion.
- **Budget concentration Gini 0.0956** (pair_budget mean 129,257, std 21,771,
  range 70,801–196,569) — free budget is also broadly distributed but with a
  meaningful spread: the best pair carries 2.8× the budget of the worst.
- **Top-10 highest-budget pairs (the rate-attack spend-first set):**
  `[442, 426, 439, 577, 437, 438, 579, 145, 532, 440]` with budgets
  196,569 → 172,046. **Temporally clustered** (426–442 and 577–579): two
  contiguous video segments where frame1 carries the most joint-safe room —
  coarsen these segments' frame1-touching bytes first.
- **Top-10 most-fragile pairs (the PROTECT set):**
  `[517, 522, 133, 177, 519, 510, 178, 518, 515, 514]` — also temporally
  clustered (510–522 and 133/177–178): segments whose frame1 seg-margins are
  thinnest; no frame1-touching byte should move there.
- **pose_jacobian_l2 per pair: mean 3.88, range 2.20–8.04** — a 3.7× spread in
  total pose sensitivity across pairs; the high-Jacobian pairs are where
  PR110++ frame1-modes must respect the pose budget most tightly.

## NO-FAKE proof (the indexed quantities are measured, not zeros)

`test_real_scorer_atlas_row_indexes_nontrivial_measured_fields` builds one atlas
row from the REAL CPU-torch scorers and asserts: the indexed pose Jacobian L2 is
> 0 (gradient reachable through the differentiable-YUV6 patch — a severed
gradient would index all-zeros and fail), the seg margin field varies
(`max > mean`), the pair budget is a positive reduction, and
`compute_path == "cpu_torch"`. A FAKE engine would index constants/zeros; this
fails closed on all three.

## 6-hook wire-in (Catalog #125)

1. **sensitivity-map** — ACTIVE (PRIMARY). The atlas IS the per-pair
   scorer-sensitivity index (seg margin field stats + pose Jacobian field stats +
   joint cone summary per pair, over 600 pairs).
2. **Pareto constraint** — ACTIVE (advisory). The fragile/pose-bound pairs are
   the feasibility boundary; the rate attack must avoid spending into them.
3. **bit-allocator hook** — ACTIVE. `pair_budget` (integrated usable cone radius)
   is the per-pair byte-spend budget; `top_budget_pairs` ranks where bytes are
   free, `most_fragile_pairs` the protect set.
4. **cathedral autopilot dispatch** — ACTIVE. `consume_evaluator_response_atlas`
   bridge emits the `[predicted]` Tier-A payload the difficulty-atlas cathedral
   consumer presents (auto-discovered per Catalog #335).
5. **continual-learning posterior** — N/A. `[macOS-CPU advisory]`
   non-promotable; the atlas is recomputed per archive, not a static posterior
   anchor (the consumer's `posterior_wire_status` records the canonical-helper
   schema mismatch honestly — no fake empirical mutation).
6. **probe-disambiguator** — ACTIVE. The query surface (`top_budget` vs
   `most_fragile` vs `pose_bound`) IS the regime-conditional disambiguator a
   consumer uses to pick the targeting policy per pair.

## Per-layer canonical-vs-unique decision (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| per-pair scorer fields | ADOPT_CANONICAL (#35 producer) | reuse the verified-real cone producer; same scorer contract + differentiable-YUV6 + fail-closed guard |
| atom row / admission law / vocabulary | ADOPT_CANONICAL (reference by name) | the atlas is an index; the IR + law + exploit families already exist |
| cross-video reduce kernel | FORK_PRINCIPLED (new MLX kernel + numpy ref) | the 600→headline reduction is genuinely new; follows the canonical_kernels Backend numpy-reference contract |
| atlas row + query surface | FORK_PRINCIPLED | genuinely new: the typed join key + the by-pair/region/family/budget consumer API |
| cathedral consumer bridge | ADOPT_CANONICAL (payload shape) + FORK_PRINCIPLED (fragility ranking) | reuses the `[predicted]` Tier-A shape; the scorer-fragility difficulty is the new orthogonal signal |

## Consumers (downstream wiring)

- **#46 LF rate-distortion waterfiller** (`lf_payload_rate_distortion.py`) — the
  atlas `top_budget_pairs` / `by_region_budget` tells THE-LAW waterfill WHICH
  pairs/regions to coarsen first; the cone-map paths the atlas indexes are the
  exact `.npz` the waterfiller's `Frame1ConeMap` reads for the per-pixel budget.
- **rate-attack repair** (`repair_*` campaign) — `most_fragile_pairs` = the
  protect set; `top_budget_pairs` = where free bytes live.
- **PR110++ selector improvement** — `pose_bound_pairs` flags where the
  frame1-mode family must respect the pose budget (the marginal-value flip).

## Single most important consumer wiring to do next

**Wire `EvaluatorResponseAtlas.top_budget_pairs` + `by_region_budget` into the
#46 LF waterfiller's per-pair/per-section dispatch order.** The waterfiller today
applies THE-LAW per section uniformly; the atlas gives it the *cross-video
ranking* of which pairs carry the most joint-safe budget so it coarsens the
highest-budget pairs first and protects the most-fragile pairs — turning "apply
the law everywhere" into "apply the law where the 600-pair index says the bytes
are free", which is the missing targeting layer for the 0.19199 → lower rate
attack.

## Reproduce

```bash
# Full 600-pair sweep (detached; ~22 min M5 Max CPU path):
PYTHONPATH=src:upstream .venv/bin/python tools/build_evaluator_response_atlas.py \
    --num-pairs 600
# Tests:
PYTHONPATH=src:upstream .venv/bin/python -m pytest \
    src/tac/tests/test_evaluator_response_atlas.py -q   # 24 + 1 slow
```
