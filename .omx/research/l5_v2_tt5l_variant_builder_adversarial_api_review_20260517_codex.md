# L5 v2 TT5L variant-builder adversarial API review

Date: 2026-05-17
Author: Codex, preserving read-only subagent audit output
Scope: L5 v2 Time-Traveler side-info variant packets, byte-closure, false-authority prevention

## APIs to reuse

- TT5L packet grammar is canonical in
  `src/tac/substrates/time_traveler_l5_autonomy/archive.py`:
  `TT5L_MAGIC`, `TT5L_SIDE_INFO_SECTION_WIDTHS`, `pack_archive`,
  `parse_archive`, `parse_tt5l_archive_bytes`, and
  `side_info_liveness_stats`.
- Byte-diff and consumption proof should reuse
  `src/tac/substrates/time_traveler_l5_autonomy/consumption_proof.py`:
  `build_tt5l_contest_full_frame_sideinfo_consumption_proof`,
  `build_tt5l_inflate_provenance_manifest`, and the existing section-hash /
  non-target-section proof pattern.
- L5-v2 authority should stay with:
  `src/tac/optimization/l5_v2_sideinfo_effect_curve.py`,
  `src/tac/optimization/l5_v2_measurement_schedule.py`, and
  `src/tac/optimization/l5_v2_paired_measurement_dispatch_plan.py`.
  Variant builders should materialize byte-closed archives and hand them to
  these surfaces, not create a second scheduler or effect-curve authority.

## False-authority blockers

Top-level manifests need schema, builder tool path, command argv, generation
time, source archive path/SHA/bytes/member SHA, runtime submission dir, runtime
tree/content SHA, variant list, and explicit:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `rank_or_kill_eligible=false`

Each variant (`zero`, `random_lsb`, `shuffled`, `trained`, `ablated`) needs:

- variant id, generation rule, seed/source
- candidate archive path/SHA/bytes
- inner TT5L member SHA
- parsed section offsets/lengths/SHAs from `parse_tt5l_archive_bytes`
- mutated byte offsets
- non-target sections identical
- allowed header delta
- per-pair side-info liveness

Preserve blockers until real custody exists:

- `requires_byte_closed_archive_path`
- `requires_archive_sha256`
- `requires_submission_dir_or_inflate_runtime`
- `requires_operator_execute_flag`
- paired CPU/CUDA identity blockers
- runtime identity blockers
- side-info effect-curve missing-cell blockers

Active side-info variants must not be all-zero. The old trained CUDA cell in
`.omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json` has
all-zero liveness and should remain classified as a negative anchor, not as a
staircase step.

## Test obligations

- Variant builder tests: `random_lsb`, `shuffled`, and `trained` fail closed on
  all-zero side-info; `zero` and `ablated` may be all-zero only when liveness is
  checked and false-authority flags remain false.
- Byte-closure tests: generated archives parse via `parse_tt5l_archive_bytes`;
  side-info offsets/hashes match bytes; world/action/meta sections remain
  unchanged except documented variant behavior.
- No-op tests: expected-changing variants must block if archive SHA or
  side-info section SHA is unchanged.
- Pair identity tests: CPU/CUDA cells for a variant reject mismatched archive
  SHA or runtime content/tree SHA.
- Integration tests: synthetic eligible schedules may expand the side-info
  curve into five paired work units, but pre-eval builder output must not
  populate `observed_cells` or pass `validate_l5_v2_sideinfo_effect_curve`.
- Regression: the old 25ep/all-zero trained packet cannot close trained
  side-info liveness or architecture-lock gates.

## Classification

Duplication risk is high if the packet builder reimplements TT5L layout,
section parsing, liveness stats, proof manifests, or dispatch/effect-curve
schemas. Duplication risk is low if the builder stays a thin materializer:
generate side-info variants, call the existing archive APIs, emit custody
manifests, and hand off to the paired dispatch/effect-curve machinery.

Current gate state remains `architecture_lock=false`: Dykstra planning evidence
exists, but probe, paired-axis, side-info effect-curve, timing-smoke, and anchor
pair evidence are incomplete.
