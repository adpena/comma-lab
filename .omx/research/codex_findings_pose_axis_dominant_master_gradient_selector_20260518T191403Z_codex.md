# Codex Findings - Pose-Axis Master-Gradient Selector + Cathedral Bridge

Date: 2026-05-18 19:14:03 UTC
Author: Codex

## Scope

Canonical task:
`codex_routing_directive_v2_synthesis_followup_null_space_plus_hash_seed_plus_cross_stack_20260518::ITEM_7`.

This is a partial cheap-probe OP-7 closure:

1. a planning-only selector that turns aggregate master-gradient score-axis
   dominance into typed `CandidateModificationSpec` rows; and
2. a Cathedral/autopilot bridge that consumes the existing
   `per_pair_difficulty_atlas` sidecar as planning-only rank signal.

The canonical anchor ledger still does not persist per-byte
`score_axis_dominance` or per-pair custody metadata, so OP-7 is not fully
closed. This landing closes a safe consumer path and records the remaining
packet-builder/custody blocker instead of treating diagnostic byte indices as
raw archive mutation authority.

## Patch

- Added `select_pose_axis_dominant_bytes(...)` to
  `src/tac/master_gradient_consumers.py`.
- The helper computes per-byte score-axis dominance as:
  `abs(gradient_axis) * contest_score_marginal / sum_axes(...)`.
- It returns `CandidateModificationSpec` rows with:
  - `mutation_grain="grammar_aware_operator"`
  - `coordinate_system="grammar_aware_operator_response"`
  - `raw_archive_byte_coordinates_allowed=false`
  - `score_claim=false`
  - `promotion_eligible=false`
  - packet-proof blockers intact
- Optional sidecar persistence writes `score_axis_dominance` diagnostics under
  the canonical master-gradient consumer output root.
- Wired `tools/cathedral_autopilot_autonomous_loop.py` to consume the canonical
  `per_pair_difficulty_atlas_<sha>_*.json` sidecar after the existing Venn and
  sister-817 per-pair sidecar cascade.
- Tightened the `per_pair_difficulty_atlas` sidecar contract so the producer
  emits `score_claim=false`, `promotion_eligible=false`, and
  `ready_for_exact_eval_dispatch=false`, plus optional score-marginal-weighted
  axis fields when an `OperatingPoint` is supplied.
- Cathedral now validates the full payload `archive_sha256` and schema, not
  only the 12-character filename prefix.
- The stronger pose-hard reward requires score-marginal-weighted fields
  (`*_axis_score_l1`). Raw `seg_axis_l1` / `pose_axis_l1` / `rate_axis_l1`
  remain useful diagnostics but cannot decide pose dominance because those
  axes live in different units.
- Hardened the per-pair sidecar reward sign convention. Cathedral sorts
  predicted-score deltas most-negative first, so a reward must multiply a
  negative delta by a factor above 1.0. The previous local comment/constant
  pattern used subunit "reward" factors, which are penalties in this ranker.
  Focused tests now prove the sidecar reward makes the effective delta more
  negative and changes rank ordering in the intended direction.
- xhigh adversarial reviewer `019e3c89-c160-7433-a82c-f3be43a73b80` found the
  raw-axis unit mismatch and filename-prefix authority gap before staging; both
  were patched and retested.

## Live Probe

Current ledger anchor:
`f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd`.

Probe command:
`select_pose_axis_dominant_bytes(sha, top_k=8, axis_dominance_threshold=0.7, write_sidecar=False)`.

Result:

- `8` typed specs emitted.
- First three diagnostic indices: `35781`, `35782`, `113384`.
- All emitted specs are planning-only and non-promotable.

The anchor does not store a persistent `score_axis_dominance` field yet; this
consumer computes the equivalent on demand from the aggregate tensor and
operating point. Future extractor rows still need to persist the field directly
if downstream tools require canonical custody rather than derived diagnostics.

No live `per_pair_difficulty_atlas_<sha>_*.json` sidecar exists for this anchor
in the current ignored state directory, so the Cathedral bridge is currently a
fail-open passthrough for this archive until a producer persists that sidecar.

## Verification

- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_master_gradient_consumers.py]`
  - Result: `41 passed`
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_master_gradient_consumers.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py]`
  - Result: `204 passed`
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check --select F821,I001,F401 src/tac/master_gradient_consumers.py src/tac/tests/test_master_gradient_consumers.py]`
  - Result: `All checks passed`
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check --select F821,I001,F401 tools/cathedral_autopilot_autonomous_loop.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py src/tac/master_gradient_consumers.py src/tac/tests/test_master_gradient_consumers.py]`
  - Result: `All checks passed`
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check --select F821 src/ experiments/ submissions/robust_current/ scripts/ tools/]`
  - Result: `All checks passed`
- `[empirical:git diff --check]`
  - Result: clean

## Residual

This is not a packet mutation builder. The next closure is to resolve these
diagnostic gradient-subject indices into grammar-aware archive coordinates and
prove repack, CRC, inflate success, and byte-consumption closure before any
operator probe or provider dispatch.

The next OP-7 custody closure should persist `score_axis_dominance` and/or
per-pair score-axis dominance in the canonical master-gradient anchor path, then
teach the grammar-aware mutation builder to consume those persisted fields
rather than recomputing from a sidecar-only diagnostic tensor.
