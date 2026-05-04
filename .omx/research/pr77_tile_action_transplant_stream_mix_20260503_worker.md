# PR77 Tile-Action Transplant Stream-Mix - Worker - 2026-05-03

Scope: local deterministic PR77 tile-action transplant / stream-mix candidates
only. No remote GPU jobs were dispatched. No score claims are made here; exact
CUDA auth eval remains the only score truth and must be run by the parent after
the dispatch-claim protocol.

## Source And Tooling

- Builder:
  `experiments/build_pr77_tile_action_transplant_candidates.py`
- Output matrix:
  `experiments/results/pr77_tile_action_transplant_stream_mix_20260503_worker/candidate_matrix.json`
- Robust parser:
  `submissions/robust_current/unpack_renderer_payload.py`
- PR77 source archive:
  `experiments/results/top_submission_reverse_engineering_20260503_pr77/archive.zip`
- PR77 archive SHA-256:
  `f90880383c95e14d82704f99db9b20944786ae6452a844348638b06c439972af`
- PR77 payload SHA-256:
  `3be6a58673133db2dd14d9f1f0903d528e452bd2e930d57fb4adb02bf264f8ec`

The builder uses the robust-current unpacker for fixed-slice decode closure and
`tac.archive_byte_profile.profile_archive()` for byte-only ZIP profiles.

## PR77 Action Stream

- Encoded action segment bytes: `325`
- Encoded action segment SHA-256:
  `d8c75e4f3725bbcf608434f0a78f5b37a9ce86bd8177c71092fd727d7e2af75a`
- Expanded runtime action bytes: `588`
- Expanded runtime action SHA-256:
  `8ac9a01caad973096c58b42daf2b1a8e476ad68cf285d443baa4ac94fdb42255`
- Runtime records: `147`
- Pair range: `11..599`
- Tile range: `82..140`
- Action range: `2..107`
- Nondecreasing pair order: `false`
- Duplicate pair/tile count: `0`

The nondecreasing-pair guard blocks treating this exact PR77 action stream as a
P6 pair-delta action stream without a separate semantic/raw-output parity
proof.

## Candidate Matrix

| candidate | archive bytes | archive sha256 | payload bytes | payload sha256 | delta vs target | delta vs PR77 | status |
|---|---:|---|---:|---|---:|---:|---|
| `pr77_actions_on_pr75_minp` | `276551` | `8f60c64b9dff70a0f53387ca108d69c768157d2deec8c8f25bfe953a9a39a360` | `276451` | `3be6a58673133db2dd14d9f1f0903d528e452bd2e930d57fb4adb02bf264f8ec` | `+70` | `0` | `pr77_non_action_streams_identical` |
| `pr77_actions_on_pr75_public` | `276830` | `adc468ddd260184638a415b30aa0e7dc3c66cc004a269d0a3b906c40a23dbbde` | `276730` | `1eef9001f42da9e20e22fe79627627af220d1aec7630b6bc96f17aebbc41daa4` | `+89` | `+279` | `runtime_parse_only_non_action_mismatch` |

Both candidates passed robust fixed-slice parse validation. The first rebuilds
the PR77 payload from PR75/minp non-action streams plus PR77 actions. The ZIP
SHA differs from the public PR77 ZIP because this builder emits a deterministic
stored-member archive with fixed metadata; the payload SHA matches PR77.

The second candidate is runtime-parse compatible, but not semantic evidence:
its renderer stream differs from PR77, so it requires exact CUDA auth eval and
component-gate review before any conclusion. Do not promote it from byte
closure.

## Skipped Targets

- `c089_p6_frontier`: skipped because the source payload is
  `public_pr75_qzs3_qp1_segactions_p6_delta_varint`, not fixed-slice. PR77's
  expanded action order is not globally nondecreasing by pair, so a direct P6
  re-encode would require reordering or a new codec and is unsafe without a
  raw-output parity proof.
- `c089_raw_no_header_probe`: skipped because the robust-current parser cannot
  validate it as a supported fixed-slice PR75-family payload
  (`bad renderer payload magic b'U\\x98h#'; expected b'RPK1'`). Do not dispatch
  PR77 action transplants against this source until a reviewed parser contract
  validates the exact raw boundaries.

## Preflight Guard Recommendation

Before any exact eval dispatch of PR77 action transplants:

1. Claim the lane with `tools/claim_lane_dispatch.py claim ...`.
2. Require candidate manifest fields:
   `score_claim=false`, `promotion_eligible=false`,
   `runtime_parse_validation.payload_format=public_pr75_qzs3_qp1_segactions_fixed_slices`,
   and PR77 action encoded SHA
   `d8c75e4f3725bbcf608434f0a78f5b37a9ce86bd8177c71092fd727d7e2af75a`.
3. Refuse P6/C089 transplants unless the PR77 actions are carried by a reviewed
   fixed-slice parser or a semantic parity proof shows that any reordering or
   new stream codec is output-equivalent.
4. Treat non-action mismatches as runtime-compatible only, not score or
   promotion evidence.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_build_pr77_tile_action_transplant_candidates.py src/tac/tests/test_unpack_renderer_payload_fixedslice.py -q`
  - `14 passed`
- `.venv/bin/python experiments/build_pr77_tile_action_transplant_candidates.py --force`
  - emitted 2 candidates, 2 skipped targets, no score claims.
