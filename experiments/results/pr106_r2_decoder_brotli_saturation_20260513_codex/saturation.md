# HNeRV Decoder Brotli Saturation Audit

- planning_only: `true`
- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_exact_eval_dispatch: `false`

## Source

- label: `PR106-R2-lowlevel`
- archive_bytes: `186629`
- archive_sha256: `287e6edc612803a9a9d5de3ce50b421c039704f38bae442a6dcc97a3e8d6ed4d`
- decoder_section_bytes: `170127`
- decoder_section_sha256: `07725c39ff436195e319f258b1e033290de30e259bc3f103b1b487f21a698c5c`
- decoder_section_entropy_bpb: `7.998223575625`

## Grid

- attempts: `576`
- qualities: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`
- lgwins: `[None, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]`
- lgblocks: `[None]`
- modes: `['font', 'generic', 'text']`

## Verdict

- verdict: `no_rate_positive_brotli_grid_candidate_same_size_variants_only`
- minimum_section_bytes_to_beat: `170126`
- best_grid_section_bytes: `170127`
- bytes_short_of_rate_positive: `1`
- best_grid_archive_byte_delta_if_swapped: `0`
- rate_positive_attempt_count: `0`
- same_size_byte_different_attempt_count: `18`

## Best Attempt

| mode | quality | lgwin | lgblock | bytes | delta | sha256 |
|---|---:|---:|---:|---:|---:|---|
| generic | 10 | default | default | 170127 | 0 | `07725c39ff436195e319f258b1e033290de30e259bc3f103b1b487f21a698c5c` |

## Entropy Ranking Anchor

- current_frontier_label: `PR106-R2-lowlevel`
- next_target_section: `None`
- ranking_minimum_section_bytes_to_beat: `170126`
- top_byte_mass_section_bytes: `170127`

Interpretation: this is a bounded local Brotli parameter-grid proof. It is
not an archive preflight result, not a score claim, and not dispatch
authorization.
