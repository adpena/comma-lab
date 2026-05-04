# Top Submission Reverse Engineering Deep Pass - 2026-05-03

Scope: local, contest-compliant reverse-engineering pass on public PR75/PR67
versus our C088/C089 PR75-action candidates. No remote GPU jobs were
dispatched. Existing code was not modified; all generated artifacts live under
`experiments/results/top_submission_reverse_engineering_20260503_deep_codex/`.

## Public Sources

Source summary artifact:
`experiments/results/top_submission_reverse_engineering_20260503_deep_codex/sources/public_sources_summary.json`.

Public PR URLs and source heads:

| PR | URL | State | Head | Archive URL | Body SHA |
|---|---|---:|---|---|---|
| PR67 | https://github.com/commaai/comma_video_compression_challenge/pull/67 | closed/merged 2026-05-03T02:22:08Z | `EthanYangTW:submission/qpose14-r55` `60297a385cdb0a5250be95e6ba2556585ec80bce` | https://github.com/EthanYangTW/comma_video_compression_challenge/releases/download/qpose14-r55-segactions-minp/archive.zip | `cc1dfa346811dade15044910fa0143e73a7a049b98aeec39b96a53481a0788bd` |
| PR75 | https://github.com/commaai/comma_video_compression_challenge/pull/75 | open | `EthanYangTW:submission/qpose14-r55-segactions-minp` `6369f829d647f2a1aff8d457291d903f9801d113` | same release asset | `cc1dfa346811dade15044910fa0143e73a7a049b98aeec39b96a53481a0788bd` |

Downloaded source and PR metadata:

- `sources/pr67_api.json`, `sources/pr75_api.json`
- `sources/pr67_files.json`, `sources/pr75_files.json`
- `sources/pr67.diff`, `sources/pr75.diff`
- `sources/pr75_head_source.tar.gz`
- extracted relevant public runtime under `sources/pr75_head/`

The PR body fields for both PR67 and PR75 report rounded components:
PoseNet `0.00048653`, SegNet `0.00060686`, archive bytes `276741`.
Recompute from rounded fields:

```text
100*0.00060686 + sqrt(10*0.00048653) + 25*276741/37545489
= 0.31470817503416587
```

This is a public-body report, not fresh local A++ evidence.

## Public Archive Anatomy

Fresh download:
`downloads/pr75_pr67_qpose14_r55_segactions_minp_archive.zip`

- Archive bytes: `276741`
- Archive SHA-256:
  `cc1dfa346811dade15044910fa0143e73a7a049b98aeec39b96a53481a0788bd`
- ZIP members: one stored member `p`, size `276641`
- Member `p` SHA-256:
  `959124abd97042983a47163f70c80fd4c0c751dd5e4cf6b5f0f434d9d2fcd66c`
- Tool outputs:
  - `tool_outputs/pr75_unzip_l.txt`
  - `tool_outputs/pr75_zipinfo_v.txt`
  - `tool_outputs/pr75_file.txt`
  - `tool_outputs/pr75_member_p_file.txt`
  - `tool_outputs/pr75_member_p_head_xxd.txt`
  - `tool_outputs/pr75_member_p_tail_xxd.txt`
  - `tool_outputs/pr75_member_p_strings_head.txt`

Public fixed-slice layout:

| Slice | Charged bytes | Charged SHA | Decoded bytes | Decoded SHA |
|---|---:|---|---:|---|
| `masks.mkv.br` | `219472` | `d1ae0d39...` | `223385` | `a5c2b89c...` |
| `renderer.bin.br` | `56034` | `d769fd89...` | `59288` | `2333284a...` |
| `seg_tile_actions.br` | `236` | `f172ef41...` | `268` | `bfd46b2b...` |
| `optimized_poses.bin.br` | `899` | `83767bbd...` | `7200` | `8b2ec594...` |

The decoded renderer magic is `QZS3`. The decoded pose stream is 600 rows of
6 fp16 values (`7200` bytes). The decoded PR75 action stream has `67` 4-byte
records: `u16 pair_index`, `u8 tile_id`, `u8 action_id`; pair range
`33..587`, tile range `83..140`, action range `2..106`.

## Compared Archives

Machine-readable comparison artifacts:

- `comparisons/stream_analysis.json`
- `comparisons/archive_summary.csv`
- `comparisons/action_records.csv`
- `comparisons/action_policy_summary.csv`
- decoded slices for public PR75, C088, and staged C089 under `unpacked/decoded/`
- helper script:
  `experiments/results/top_submission_reverse_engineering_20260503_deep_codex/analyze_top_submission_streams.py`

Summary:

| Label | Archive bytes | SHA | Payload format | Actions | Score evidence |
|---|---:|---|---|---:|---|
| `public_pr75` | `276741` | `cc1dfa34...` | public fixed slices | `67` | public body only, recompute `0.31470817503416587` |
| `legacy_public_pr67` | `276564` | `a5ed8da0...` | `public_pr67_qzs3_qp1_fixed_slices` | `0` | external replay `0.3630208381950889`, not a current PR75 replay |
| `c082_actions_only_p3` | `276460` | `851a61dd...` | P3 | `67` | exact T4 `0.31553274375536466` |
| `c088_top40_p3` | `276386` | `9feef7ff...` | P3 | `40` | exact T4 `0.3155226919767294` |
| `c089_staged_top25_ampminus1_p3` | `276328` | `bdc966be...` | P3 | `25` | staged; RTX PRO diagnostic same bytes scored `0.31530060435788376` |
| `full_pr75_streams_on_c067` | `276751` | `f14f3c8d...` | P3 | `67` | exact T4 `0.31577689189222813` |
| `nextwave_beam_pose2_top55_p3` | `276404` | `e1a589b0...` | P3 | `47` | staged L40S diagnostic lane exists; no local result yet |
| `nextwave_beam_pose4_top55_p3` | `276378` | `bfe4f369...` | P3 | `38` | staged L40S diagnostic lane exists; no local result yet |
| `nextwave_top25_signedposemix1_p3` | `276359` | `b4d55dd8...` | P3 | `50` | staged L40S diagnostic lane exists; no local result yet |

## Stream Deltas

Public PR75 vs C088:

- Archive: public PR75 is `+355` bytes versus C088.
- Decoded masks: byte-identical (`223385` bytes, SHA `a5c2b89c...`).
- Decoded renderer: same length (`59288`), different SHA; `2359` changed bytes,
  first diff at decoded offset `20035`.
- Decoded poses: same length (`7200`), different SHA; `467` changed bytes,
  first diff at offset `0`; fp16 metrics over 3600 values:
  max abs `0.09375`, mean abs `0.0032812499`, RMS `0.0099845259`.
- Decoded actions: public PR75 has `268` bytes / `67` records; C088 has
  `160` bytes / `40` records. C088 exactly overlaps `40` public records and
  drops `27` public records.

C088 vs staged C089 top25 ampminus1:

- Archive: C089 is `58` bytes smaller than C088.
- Decoded masks, renderer, and poses are byte-identical to C088.
- Decoded actions are the only stream change: C088 `40` records/`160` bytes;
  C089 `25` records/`100` bytes.
- Exact action overlap is `0` because ampminus1 shifts action IDs; pair/tile
  overlap is `25`.
- RTX PRO diagnostic for these exact C089 bytes scored
  `0.31530060435788376`, a diagnostic delta of `-0.0002220876188456483`
  versus C088. At pure rate, the same score delta equals about `334` bytes,
  so most of the diagnostic gain is component-side, not rate.

Nextwave beam candidates vs C088:

- `beam_pose2_top55_p3`: `+18` archive bytes, same mask/renderer/pose as C088,
  `47` records, exact action overlap `39`.
- `beam_pose4_top55_p3`: `-8` archive bytes, same mask/renderer/pose as C088,
  `38` records, exact action overlap `31`.
- `top25_signedposemix1_p3`: `-27` archive bytes, same mask/renderer/pose as
  C088, `50` records but only `25` unique pairs due signed duplicate records.

## Runtime Code Path Differences

Public PR75 runtime source:
`sources/pr75_head/submissions/qpose14_r55_segactions_minp/inflate.py`

Relevant tool outputs:

- `tool_outputs/public_pr67_to_pr75_inflate_diff.txt`
- `tool_outputs/pr75_public_vs_robust_current_inflate_diff.txt`
- `tool_outputs/pr75_public_inflate_ast_functions.txt`
- `tool_outputs/pr75_public_inflate_dis_tile_actions.txt`
- `tool_outputs/pr75_public_inflate_py_compile.txt`

Public PR75 adds these runtime mechanics over the older qpose14/QZS3/QP1
path:

1. Fixed public PR75 layout: `mask[219472]`, `model[56034]`,
   `actions[236]`, `pose[899]`.
2. Self-describing P3 layout: `b"P3" + <IHH> + mask + model + actions + pose`.
3. `seg_tile_action_specs`: 9 RGB directions x 6 amplitudes x sign = 108
   fixed action vectors.
4. `load_seg_tile_actions_data`: Brotli-decodes 4-byte or 5-byte action
   records and groups them by generated pair index.
5. Runtime application: before upsampling, apply action deltas to fake2 only,
   over 32x32 tiles on the 512x384 renderer grid, then clamp to `[0, 255]`.
6. If the decoded mask stream has fewer than 600 masks, repeat-interleave to
   600 generated pairs.

Our `robust_current` path supports PR75 actions through a stricter split:

- `unpack_renderer_payload.py` parses public fixed slices, P3, P4, P5, and P6,
  expands charged action streams into `seg_tile_actions.bin`, and writes a
  normal logical archive layout.
- `inflate_renderer.py` loads optional `seg_tile_actions.bin` or
  `seg_tile_actions.br`, validates tile/action bounds, supports custom
  `seg_tile_action_dict.bin`, and applies deltas to pair tensor slot 1 before
  upscale.

Important interpretation: the public PR75 body score is better than C088 by
about `0.0008145`, but our exact T4 `full_pr75_streams_on_c067` replay through
`robust_current` scored worse than C088 (`0.31577689189222813`). That means the
public renderer/pose/action blob is not a simple stream transplant win in the
current robust runtime. The remaining public-body clue is either a runtime-path
parity issue, scorer/eval drift from the public report, or an interaction that
requires PR75's exact public inflate code path. Do not claim public PR75 parity
until a raw-output parity replay or exact public-runtime auth eval proves it.

## Score Math

C088 exact T4 score: `0.3155226919767294`.

- Gap to public PR75 body-field recompute: `0.0008145169425635146`.
- Gap to hard `0.314` threshold: `0.0015226919767293845`.
- Pure rate equivalent for the hard threshold: about `2287` bytes.
- Pure rate equivalent for the user's `>=0.0016` target: about `2403` bytes.

The PR75 action stream itself cannot supply this as bytes: the full public
action slice is only `236` charged bytes. Action policies can help components,
but the missing move must involve renderer/runtime parity, renderer
self-compression, or a geometry-preserving mask/representation byte reduction.

## Ranked Implementable Deltas

1. **Public-runtime raw-output parity harness for PR75.**
   Implement a local comparer that inflates the same public PR75 archive through
   `sources/pr75_head/.../inflate.py` and through `robust_current`, then
   hashes raw output frames before scoring. This is the highest-EV forensic
   step because public body math is `0.0008145` below C088, while robust full
   stream replay regresses. If output differs, port the minimal runtime delta
   rather than guessing new atoms.

2. **Promote or reject C089 top25 ampminus1 on T4, then use it as the next
   action baseline if it transfers.**
   Existing RTX PRO diagnostic score is `0.31530060435788376`, `-0.0002221`
   versus C088. This is not enough alone, but it is the cleanest current action
   improvement because mask/renderer/pose are byte-identical to C088 and only
   the charged action records changed.

3. **Learn a second-generation tile-action dictionary with raw-output parity
   checks, not only prefix pruning.**
   Existing P4/P5 dictionaries are byte-regressive, but the action policy table
   shows non-prefix policies with higher trace sums. A better implementation
   should search custom RGB deltas and record subsets jointly, then emit P3/P6
   unless a compact dictionary actually beats fixed actions after outer ZIP.

4. **Renderer self-compress the C088 renderer with a strict output/parity
   budget.**
   Sub-0.314 needs only about `2287` pure-rate bytes from C088 if components
   hold. That is about `4.1%` of the `55965`-byte charged renderer slice. This
   is much more plausible than action-byte savings, but must pass
   renderer-transplant pose safety and exact CUDA because prior full PR75
   renderer/pose transplant was antagonistic.

5. **Geometry-preserving mask stream byte shave, targeting only 1-2% of the
   `219472` charged mask slice.**
   Public PR75, legacy PR67, C082, C088, C089, and nextwave action candidates
   all share the same decoded mask SHA. Direct lossy mask/grammar attempts have
   collapsed PoseNet, so the implementable version is OBU/mux/lossless or
   near-lossless with raw-output parity gates, not another broad AV1 CRF sweep.

6. **Pose active-subspace continuation as a secondary stackable delta.**
   Public PR75 pose differs from C088 in `467` decoded bytes but direct
   transplant regresses. Use only component-trace/active-subspace pose updates
   that preserve the C088 renderer/mask basin; do not wholesale copy public
   pose bytes.

## Compliance Notes

- No remote GPU jobs were dispatched during this pass.
- No existing source files were modified.
- The new helper script is local to the artifact directory and is not imported
  by production/runtime code.
- Public PR body fields and public archive anatomy are `external`/forensic
  evidence until exact CUDA auth eval with the intended runtime path lands.
