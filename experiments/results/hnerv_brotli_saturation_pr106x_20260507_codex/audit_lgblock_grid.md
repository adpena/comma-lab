# HNeRV Decoder Brotli Saturation Audit

- planning_only: `true`
- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_exact_eval_dispatch: `false`

## Source

- label: `PR106x-lowlevel-brotli`
- archive_bytes: `186080`
- archive_sha256: `b0a12549a39e34a0d7f83ea99e05e55fcd01d795a15db2ffb3d92ccc6267e53f`
- decoder_section_bytes: `170127`
- decoder_section_sha256: `07725c39ff436195e319f258b1e033290de30e259bc3f103b1b487f21a698c5c`
- decoder_section_entropy_bpb: `7.998223575625`

## Grid

- attempts: `5760`
- qualities: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`
- lgwins: `[None, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]`
- lgblocks: `[None, 16, 17, 18, 19, 20, 21, 22, 23, 24]`
- modes: `['font', 'generic', 'text']`

## Verdict

- verdict: `rate_positive_brotli_grid_candidate_found`
- minimum_section_bytes_to_beat: `170126`
- best_grid_section_bytes: `170126`
- bytes_short_of_rate_positive: `0`
- best_grid_archive_byte_delta_if_swapped: `-1`
- rate_positive_attempt_count: `18`
- same_size_byte_different_attempt_count: `144`

## Best Attempt

| mode | quality | lgwin | lgblock | bytes | delta | sha256 |
|---|---:|---:|---:|---:|---:|---|
| generic | 10 | default | 16 | 170126 | -1 | `a812f1e837afd0e463a7f133b680ea6c027339ff8816db7012dd41253435afbf` |

## Entropy Ranking Anchor

- current_frontier_label: `PR106x-lowlevel-brotli`
- next_target_section: `decoder_packed_brotli`
- ranking_minimum_section_bytes_to_beat: `170126`
- top_byte_mass_section_bytes: `170127`

Interpretation: this is a bounded local Brotli parameter-grid proof. It is
not an archive preflight result, not a score claim, and not dispatch
authorization.
