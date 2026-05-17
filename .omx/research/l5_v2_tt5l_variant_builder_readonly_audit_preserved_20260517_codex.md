# L5-v2 TT5L Variant Builder Read-only Audit Preservation

Date: 2026-05-17
Agent: 019e33ac-f69a-76d2-a575-b4a677684571
Scope: preserve read-only subagent signal before pushing `main`
Status: preserved; no files were edited by the subagent

## Context

The operator requested that all work be committed and pushed to `origin/main`
with no signal loss. The read-only L5-v2/TT5L audit returned useful API,
blocker, test, and duplication-risk guidance that had not yet been committed
as durable research state.

## APIs to reuse

- `src/tac/substrates/time_traveler_l5_autonomy/archive.py` is the canonical
  TT5L packet grammar surface: `TT5L_MAGIC`,
  `TT5L_SIDE_INFO_SECTION_WIDTHS`, `pack_archive`, `parse_archive`,
  `parse_tt5l_archive_bytes`, and `side_info_liveness_stats`.
- `src/tac/substrates/time_traveler_l5_autonomy/consumption_proof.py` already
  provides the section-hash and non-target-section proof pattern through
  `build_tt5l_contest_full_frame_sideinfo_consumption_proof` and
  `build_tt5l_inflate_provenance_manifest`.
- `src/tac/optimization/l5_v2_sideinfo_effect_curve.py`,
  `src/tac/optimization/l5_v2_measurement_schedule.py`, and
  `src/tac/optimization/l5_v2_paired_measurement_dispatch_plan.py` remain the
  authority surfaces for effect curves, schedules, and paired dispatch plans.
  The TT5L side-info variant builder should materialize byte-closed variant
  archives and hand off to those surfaces rather than becoming a second
  scheduler or evaluator.

## False-authority blockers

Top-level variant manifests should continue to carry explicit non-promotion
state until paired evidence exists:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `rank_or_kill_eligible=false`

Required custody fields include schema, builder path, command argv, generated
time, source archive path/SHA/bytes/member SHA, runtime submission directory,
runtime tree/content SHA, variant list, and per-variant candidate archive
path/SHA/bytes/member SHA.

Per variant (`zero`, `random_lsb`, `shuffled`, `trained`, `ablated`) the
manifest should record generation rule, seed/source, parsed section offsets,
section lengths, section hashes from `parse_tt5l_archive_bytes`, mutated byte
offsets, non-target section identity, allowed header deltas, and per-pair
side-info liveness.

Preserve blockers until the matching evidence exists:

- `requires_byte_closed_archive_path`
- `requires_archive_sha256`
- `requires_submission_dir_or_inflate_runtime`
- `requires_operator_execute_flag`
- paired CPU/CUDA identity blockers
- runtime identity blockers
- side-info effect-curve missing-cell blockers

Active side-info variants must not be all-zero. The existing trained CUDA cell
in `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json` has
all-zero side-info liveness and should be treated as a negative anchor, not as
a staircase step.

## Test obligations

- Variant builder tests should prove `random_lsb`, `shuffled`, and `trained`
  fail closed on all-zero side-info.
- `zero` and `ablated` may allow all-zero side-info only with checked liveness
  and false-authority flags still false.
- Byte-closure tests should prove generated archives parse via
  `parse_tt5l_archive_bytes`, side-info offsets and hashes match archive bytes,
  and world/action/meta sections remain unchanged except for documented variant
  behavior.
- No-op tests should block expected-changing variants when archive SHA or
  side-info section SHA is unchanged.
- Pair-identity tests should reject CPU/CUDA cells for a variant when archive
  SHA or runtime content/tree SHA differs.
- Integration tests should show a synthetic eligible schedule expands the
  side-info curve into paired work units, while pre-eval builder output cannot
  populate `observed_cells` or pass `validate_l5_v2_sideinfo_effect_curve`.
- Regression coverage should ensure old 25ep/all-zero trained packets cannot
  close trained side-info liveness or architecture-lock gates.

## Duplication risk

The highest risk is reimplementing TT5L packet layout, section parsing,
liveness stats, proof manifests, or dispatch/effect-curve schemas inside the
variant builder. The safe shape is a thin materializer:

1. Generate side-info variants.
2. Call the canonical TT5L archive APIs.
3. Emit custody manifests and false-authority blockers.
4. Hand off to the existing paired dispatch and effect-curve machinery.

## Next concrete action

Harden `src/tac/optimization/tt5l_sideinfo_variant_packets.py` and
`src/tac/tests/test_tt5l_sideinfo_variant_packets.py` against active all-zero
and expected-changing no-op variants, then rerun the focused L5-v2 test slice.

## Second read-only audit findings

Agent: 019e33e7-7a73-7130-b9b5-d308f0c977a5
Status: completed after the initial push; no files were edited by the subagent

### Findings

1. High: the effect-curve validator can accept custody-light hand-written
   curves. `build_l5_v2_sideinfo_effect_curve()` validates exact-eval cells,
   but the shared validator checks shape, flags, pair identity, liveness, and
   blockers without requiring artifact/log paths, hardware/devices, auth-eval
   command, sample count, component fields, raw-output aggregate SHA, or
   inflated-output manifest custody. Patch `_sideinfo_effect_curve_blockers()`
   in `src/tac/optimization/l5_v2_measurement_schedule.py` and add a negative
   test around the minimal fixture in
   `src/tac/tests/test_l5_v2_measurement_schedule.py`.

2. Medium: variant no-op blockers are weaker than this memo requires. The
   builder records section SHAs, but `_variant_blockers()` only compares
   side-info arrays to zero/trained controls, not archive/member SHA or
   `per_pair_side_info_blob` SHA no-ops. Patch `_variant_blockers()` and its
   call site in `src/tac/optimization/tt5l_sideinfo_variant_packets.py`, with
   tests for expected-changing variants blocking unchanged archive/member/side
   section SHA.

3. Low: byte-closure tests do not verify all recorded byte facts. Current
   tests parse generated archives and check non-target section identity, but
   do not assert recorded section offsets, lengths, and SHAs against the actual
   inner bytes or prove `mutated_byte_ranges` align with the side-info section.

4. Low: memo metadata gaps remain. The review memo calls for generation time
   and per-variant generation rule/seed/source. The manifest has
   `command_argv`, source custody, `variant_seed`, and semantics, but no
   generation timestamp and no per-row seed/source. Patch
   `build_tt5l_sideinfo_variant_packets()` and its tests.

### Duplication risk

Parser/layout duplication is mostly controlled: the builder uses canonical
`parse_archive`, `parse_tt5l_archive_bytes`, `TT5L_SIDE_INFO_SECTION_WIDTHS`,
and `side_info_liveness_stats`. The remaining moderate risk is manual TT5L
header repacking in `_replace_sideinfo_blob`, mirrored by consumption-proof
helpers like `_pack_with_world_blob`. A canonical replace-side-info-section
helper in `archive.py` would remove that drift risk.

### Updated next concrete action

Priority order for the next L5-v2 hardening patch:

1. Close the high-severity custody-light effect-curve validator gap in
   `l5_v2_measurement_schedule.py`.
2. Close expected-changing no-op blockers in
   `tt5l_sideinfo_variant_packets.py`.
3. Extend byte-closure tests to assert offsets, lengths, hashes, and mutated
   byte ranges against actual inner TT5L bytes.
4. Add generation timestamp and per-row seed/source metadata.
