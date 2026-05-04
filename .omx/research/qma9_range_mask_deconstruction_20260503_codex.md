# QMA9 Range-Mask Arithmetic Deconstruction - 2026-05-03

Scope: local-only deconstruction of PR81/QMA9 range-mask arithmetic coding. No
remote GPU dispatch, no scorer invocation, and no score/component claim.

## Custody

- PR81 archive: `experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex/archive.zip`
- Archive bytes/SHA-256: `215960`, `cd01378a52688fe00ee1bfb898c67695aed89a7b3ded602b597eb7fc3031d7fc`
- Stored member `p`: `215860` bytes, SHA-256 `c59524610474c89e5a41433f47d2bb881f878e694f853db1377272699f9eb3e9`
- QMA9 mask segment: `159011` bytes, SHA-256 `4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179`
- QMA9 bitstream: `158991` bytes, SHA-256 `77196f31a8fdfbf8f91c9c30600945473376534f15245bfb7622ddd77d17e3af`
- Decoded mask contract: `600 x 512 x 384 = 117964800` class bytes, classes `0..4`, sentinel `5`

Artifacts:

- [empirical:experiments/results/qma9_range_mask_deconstruction_20260503_codex/qma9_range_mask_bitstream_profile.json]
- [empirical:experiments/results/qma9_range_mask_deconstruction_20260503_codex/qma9_range_mask_cpp_full_profile.json]
- [empirical:experiments/results/qma9_range_mask_deconstruction_20260503_codex/pr81_qma9_cpp_decode_hash_profile.json]
- [empirical:experiments/results/qma9_range_mask_deconstruction_20260503_codex/candidates/qma9_pr81_template_rebuild_byte_screen/manifest.json]

## Verification

Pure-Python prefix/state verification was run on the actual PR81 payload for
the first full frame (`196608` decoded pixels). Checkpoints were recorded after
pixels `0, 1, 2, 4095, 65535, 196607`; prefix decoded SHA-256 is
`cce78dbebb6d2bdea7369c7c6671a1b27f7f04182a30dbe819485f07faadf061`.
The Python prefix self-roundtrip re-encodes and decodes that frame with exact
raw-byte equality. Full pure-Python all-frame re-encode was not run because the
full stream is 117,964,800 pixels; full-stream profiling used the local C++
instrumentation of the same arithmetic model.

Public C++ decode hash check succeeded:
`117964800` decoded bytes, SHA-256
`c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`.

The local C++ full-stream model-bit accounting is tight:

- estimated adaptive-model bits: `1271926.81158`
- charged bitstream bits: `1271928`
- final decoder consumed bits: `1271926` plus final padding

This is strong evidence that the profiler is following the same arithmetic
state and symbol model as the PR81 payload.

## Symbol Model Anatomy

QMA9 uses a 9-neighbor base-6 context:
`prev, left, up, up_left, up_right, prev_right, prev_down, up2, left2`.
For each pixel it encodes a fixed decision tree:

1. binary `same as up`
2. if false, binary `same as left`
3. if false, binary `same as prev frame`
4. if false, 5-way raw class

Measured PR81 symbol outcomes:

- `up`: `117394995` pixels
- `left`: `409373` pixels
- `prev`: `82850` pixels
- `class_fallback`: `77582` pixels

Measured model-bit cost:

- up gate: `1233870.669770` bits
- left gate: `21814.881306` bits
- prev gate: `5284.969128` bits
- class fallback: `10956.291377` bits

Interpretation: the stream is almost entirely a vertical-copy coder. The up
gate predicts `99.5169%` of pixels, but because it is paid on every pixel it
still accounts for about `97%` of modeled bits. Improvements should primarily
try to skip or amortize repeated up-gate decisions, not add more unconditional
fallback branches.

Class distribution:

- class `0`: `27408427`
- class `1`: `690063`
- class `2`: `58413695`
- class `3`: `1459867`
- class `4`: `29992748`

Top contexts are mostly uniform fields:

- all `2`: `56831167` pixels, `12414.608398` bits
- all `4`: `28850653` pixels, `842.639648` bits
- all `0`: `24779297` pixels, `105861.882812` bits
- all `3`: `1164629` pixels, `3079.859619` bits

The all-zero context is disproportionately expensive relative to all-2/all-4,
so context-specific zero-region transitions are a concrete target.

## Cost Hotspots

Top frames by estimated bytes: `517` (`428.01`), `522` (`387.93`), `519`
(`360.14`), `70` (`359.63`), `518` (`358.91`), `74` (`352.97`).

Top row indices by estimated bytes: row `0` (`918.17` total bytes over all
frames), then rows `288`, `320`, `296`, `324`, `304`, `321`, `317`, `318`.
Row `0` is special because `up` is sentinel; it needs a first-row grammar or
row-header/copy treatment rather than the same vertical predictor path.

Horizontal run structure:

- total horizontal runs: `1584702`
- max run length: `229`
- run buckets: `16_31=81858`, `32_63=124081`, `64_plus=840087`

The run count shows a large finite run-escape search space. However, current
up-gate coding is already extremely cheap inside many long runs, so a naive
horizontal RLE is not automatically a win; it needs exact byte-screen rebuilds.

## Candidate Screens

1. `qma9_vertical_copy_block_or_run_escape`
   - Target the dominant cost: every pixel pays an up-gate even when huge
     regions are vertical copies.
   - Concrete rebuild: add deterministic row/block copy flags for spans whose
     current up-gate probability is already saturated; residual pixels fall
     back to the current QMA9 tree.
   - Required proof before dispatch: exact raw mask equality, byte-closed
     decoder/runtime, deterministic archive manifest, then CUDA auth eval.

2. `qma9_first_row_specialization`
   - Row `0` is the hottest row because `up` is sentinel. Encode first rows
     with left-run or previous-frame-row predictors instead of paying the
     standard up-first tree.
   - This is small but well-scoped and contest-compliant if decoder bytes are
     charged or fixed in runtime.

3. `qma9_context_backoff_prune_sparse9`
   - The 9-neighbor model has huge sparse context space. A deterministic
     lower-order backoff for cold contexts may reduce cold-start bits without
     side information.
   - Must be measured by full re-encode; no proxy promotion.

4. `qma9_horizontal_run_escape_len16`
   - Planning screen reports `1046026` horizontal runs of length `>=16` and a
     lower-bound `57999862` tail pixels.
   - This is a search candidate, not a predicted win, because vertical-copy
     up-gate bits are already near zero in many long runs.

5. Extra unconditional fallback gates are rejected as simple PR81 successors.
   - Example: `up_right` matches `60606 / 77582` fallback pixels, but the
     current fallback class branch averages only `0.1412` bits. A naive added
     binary gate costs about `77582` bits and screens at roughly `-8628` bytes.
   - Only context-conditioned extra gates should be considered.

## Commands

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_qma9_range_mask_contract.py \
  src/tac/tests/test_profile_qma9_range_mask_bitstream.py \
  src/tac/tests/test_build_qma9_range_mask_candidate.py \
  src/tac/tests/test_profile_pr81_qma9_range_mask_contract.py -q

.venv/bin/python experiments/profile_qma9_range_mask_bitstream.py \
  --output-dir experiments/results/qma9_range_mask_deconstruction_20260503_codex \
  --output-json experiments/results/qma9_range_mask_deconstruction_20260503_codex/qma9_range_mask_bitstream_profile.json \
  --pure-python-max-pixels 196608 \
  --checkpoint-pixels 0,1,2,4095,65535,196607 \
  --cpp-timeout-seconds 300

.venv/bin/python experiments/profile_pr81_qma9_range_mask_contract.py \
  --try-cpp-decode-hash \
  --output-json experiments/results/qma9_range_mask_deconstruction_20260503_codex/pr81_qma9_cpp_decode_hash_profile.json \
  --cpp-timeout-seconds 300

.venv/bin/python experiments/build_qma9_range_mask_candidate.py \
  --output-dir experiments/results/qma9_range_mask_deconstruction_20260503_codex/candidates \
  --candidate-id qma9_pr81_template_rebuild_byte_screen \
  --template-pr81-archive experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex/archive.zip
```

The template rebuild reproduced archive SHA-256
`cd01378a52688fe00ee1bfb898c67695aed89a7b3ded602b597eb7fc3031d7fc` exactly.

## 2026-05-03 Byte-Search Profiler Addendum

Scope: added a deterministic local byte-search mode to
`experiments/profile_qma9_range_mask_bitstream.py` for PR81/PR84-style QMA9
full-stream mask containers. It remains local-only: no GPU dispatch, no scorer
invocation, no training, and no exact-score claim.

The profiler now accepts PR81 `inflate.py` constants or PR84 profile JSON
split metadata, including PR84 archives where the router/action stream is
absent. With `--run-byte-search --byte-search-frames all`, it decodes the QMA9
mask bytes, records the raw-mask SHA-256, evaluates a finite mode matrix, and
requires exact raw-mask decode parity before a candidate can be marked
`accepted_for_exact_eval_candidate=true`.

Initial finite modes:

- `qma9_reference_reencode`: reference/no-op mode. It records exact bytes and
  SHA-256 but is rejected for candidate selection unless it changes archive
  state and wins bytes, which it should not.
- `qmb1_vertical_block_escape_bw{N}` for a deterministic list of block widths.
  Each candidate records exact payload bytes, SHA-256, raw parity, copied-block
  opportunity stats, projected archive bytes if other streams stay unchanged,
  rate-only delta, and rejection reasons.

Selection guard: a future exact-eval candidate is safe to choose only when the
local manifest shows raw parity, archive-relevant state change, and a local
range-mask byte win. Dispatch still requires runtime archive closure, a lane
claim, and exact CUDA auth eval through the canonical archive path.

Verification:

```bash
.venv/bin/python -m py_compile \
  experiments/profile_qma9_range_mask_bitstream.py \
  src/tac/tests/test_qma9_range_mask_byte_search.py \
  src/tac/tests/test_profile_qma9_range_mask_bitstream.py

c++ -O3 -std=c++17 experiments/qma9_range_mask_cpp_profiler.cpp \
  -o /tmp/qma9_range_mask_cpp_profiler_check

.venv/bin/python -m pytest \
  src/tac/tests/test_qma9_range_mask_byte_search.py \
  src/tac/tests/test_profile_qma9_range_mask_bitstream.py \
  src/tac/tests/test_qma9_range_mask_contract.py -q

.venv/bin/python experiments/profile_qma9_range_mask_bitstream.py \
  --archive experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex/archive.zip \
  --split-constants-py experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex/replay_submission/inflate.py \
  --output-dir /tmp/qma9_byte_search_smoke \
  --output-json /tmp/qma9_byte_search_smoke/profile.json \
  --pure-python-max-pixels 16 \
  --checkpoint-pixels 0 \
  --skip-cpp-full \
  --run-byte-search \
  --byte-search-frames 1 \
  --qmb1-block-widths 16,32

.venv/bin/python experiments/profile_qma9_range_mask_bitstream.py \
  --archive experiments/results/top_submission_reverse_engineering_20260503_pr84/archive.zip \
  --split-constants-py experiments/results/top_submission_reverse_engineering_20260503_pr84/pr84_qma9_semantic_range_mask_profile.json \
  --output-dir /tmp/qma9_pr84_byte_search_smoke \
  --output-json /tmp/qma9_pr84_byte_search_smoke/profile.json \
  --pure-python-max-pixels 16 \
  --checkpoint-pixels 0 \
  --skip-cpp-full \
  --run-byte-search \
  --byte-search-frames 1 \
  --qmb1-block-widths 16
```

Observed smoke result: both PR81 and PR84 prefix byte-searches completed with
`score_claim=False`, `dispatch_performed=False`, and zero accepted local byte
wins. That is a valid negative byte-screen outcome, not a method kill; full
stream search remains opt-in via `--byte-search-frames all`.

## 2026-05-03 Codex QMB1 Vertical Block-Escape Prototype

Implemented a local-only deterministic `QMB1` prototype for
`qma9_vertical_block_escape`: exact row-above block-copy flags are charged in
the range-mask payload, and non-copy blocks fall back to the existing QMA9
pixel decision tree. This is not a contest runtime format and did not touch
inflate, Lightning, scorer, or PR82 files.

Code/artifacts:

- [empirical:src/tac/qma9_range_mask_contract.py]
- [empirical:experiments/build_qma9_vertical_block_escape_candidate.py]
- [empirical:experiments/results/qma9_range_mask_deconstruction_20260503_codex/candidates/qma9_vertical_block_escape_len16_prefix1/manifest.json]
- [empirical:experiments/results/qma9_range_mask_deconstruction_20260503_codex/candidates/qma9_vertical_block_escape_len64_prefix4/manifest.json]

Synthetic encode/decode parity is green for repeated-row masks. The PR81
bounded byte screens are negative:

- `block_width=16`, first PR81 frame: baseline subset QMA9 `359` bytes,
  QMB1 `513` bytes, delta `+154` bytes. The subset has `11761 / 12264`
  eligible blocks copied and `188176 / 196608` pixels copied, but the copy
  flags and colder fallback model still cost more than the already-saturated
  QMA9 up-gate.
- `block_width=8`, first PR81 frame: QMB1 `573` bytes, delta `+214` bytes.
- `block_width=32`, first PR81 frame: QMB1 `463` bytes, delta `+104` bytes.
- `block_width=64`, first PR81 frame: QMB1 `434` bytes, delta `+75` bytes.
- `block_width=64`, first 4 PR81 frames: baseline subset QMA9 `1221` bytes,
  QMB1 `1533` bytes, delta `+312` bytes. The subset has `10573 / 12264`
  eligible blocks copied and `676672 / 786432` pixels copied.

Linear full-stream projections from these bounded prefixes remain
planning-only and are not dispatchable. The best sampled case above
(`block_width=64`, first 4 frames) projects `229950` range-mask bytes versus
the PR81 range-mask segment at `159011` bytes (`+70939` bytes), so this
specific unconditional row-above block escape is a measured-implementation
negative for local follow-up. The result narrows the next useful search toward
context-conditioned copy flags, first-row specialization, or a run escape that
does not cool the QMA9 adaptive model as aggressively.

Commands:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_qma9_range_mask_contract.py \
  src/tac/tests/test_build_qma9_vertical_block_escape_candidate.py \
  src/tac/tests/test_profile_qma9_range_mask_bitstream.py \
  src/tac/tests/test_build_qma9_range_mask_candidate.py -q

.venv/bin/python experiments/build_qma9_vertical_block_escape_candidate.py \
  --output-dir experiments/results/qma9_range_mask_deconstruction_20260503_codex/candidates \
  --candidate-id qma9_vertical_block_escape_len16_prefix1 \
  --frames 1 \
  --block-width 16

.venv/bin/python experiments/build_qma9_vertical_block_escape_candidate.py \
  --output-dir experiments/results/qma9_range_mask_deconstruction_20260503_codex/candidates \
  --candidate-id qma9_vertical_block_escape_len64_prefix4 \
  --frames 4 \
  --block-width 64
```

## Status

Evidence grade: `empirical/planning_only`. No score claim. No dispatch. The
next useful local step is a deterministic re-encoder variant for vertical
copy/block escape and first-row specialization, with raw-mask equality as the
first gate and byte count as the second gate.

Supersession note: the `QMB1` row-above block-copy prototype above completed
the vertical block-escape substep and screened negative on bounded PR81
prefixes. Keep first-row specialization and context-conditioned/run escape
variants open; do not promote unconditional block-copy flags from this screen.

## 2026-05-03 Codex QMC1 Context-Conditioned Run-Escape Screen

Implemented a local-only deterministic `QMC1` planning screen for
context-conditioned row-above run escapes. Unlike `QMB1`, copied runs still
advance the base QMA9 adaptive model with the known copied symbols, so this
screen isolates copy-flag/run overhead from the previous cold-model failure
mode. The screen is bounded-prefix only, emits no runtime format, touches no
Lightning/PR82/scorer files, and is not dispatchable.

Code/artifacts:

- [empirical:experiments/build_qma9_context_run_escape_candidate.py]
- [empirical:src/tac/tests/test_build_qma9_context_run_escape_candidate.py]
- [empirical:experiments/results/qma9_range_mask_deconstruction_20260503_codex/candidates/qma9_context_run_escape_min192_prefix4/manifest.json]
- [empirical:experiments/results/qma9_range_mask_deconstruction_20260503_codex/candidates/qma9_context_run_escape_min224_prefix4/manifest.json]

PR81 bounded byte screens, all against exact subset QMA9 re-encode of the same
decoded raw prefix:

- `min_run_length=64`, require left context, first 4 frames: baseline subset
  QMA9 `1221` bytes, QMC1 `1321` bytes, delta `+100` bytes. It copied
  `2073` runs / `359711` pixels and projects `198150` range-mask bytes versus
  PR81 `159011` (`+39139` bytes).
- `min_run_length=128`, require left context, first 4 frames: QMC1 `1309`
  bytes, delta `+88` bytes, projection `196350` bytes (`+37339`).
- Best nontrivial sampled case: `min_run_length=192`, require left context,
  first 4 frames: QMC1 `1264` bytes, delta `+43` bytes. It copied `538` runs /
  `110966` pixels, updated the QMA9 model for every copied pixel, and projects
  `189600` range-mask bytes (`+30589`).
- `min_run_length>=224` produced `0` copied runs in the prefix and QMC1 `1229`
  bytes, delta `+8` bytes. This is marked non-dispatchable as a no-op except
  for wrapper/header bytes; its lower projection is not an archive-relevant
  improvement.
- Removing the left-context gate worsened bytes (`min64_noleft`: QMC1 `1426`
  bytes, projection `213900`) despite copying more pixels, so broader flags
  are ruled out for this measured implementation.

Conclusion: `QMC1` is a measured-implementation negative for local follow-up.
It rules out a simple high-confidence row-above run flag escape, even when the
adaptive base model is kept hot. The remaining plausible QMA9-native work is
not a broader copy flag: it should target either a true first-row grammar with
negligible side cost, a zero/no-flag deterministic transform with exact
exceptions, or a lower-order context/backoff change measured by full re-encode.

Runtime bridge addendum: PR81 QMA9 is now supported by the robust inflate path
as a typed charged member. `unpack_renderer_payload.py` emits `masks.qma9`,
`inflate_renderer.py` routes QMA9 masks through the fast C++ range-mask decoder
when available, and `submissions/robust_current/range_mask_codec.cpp` is now
part of the robust runtime source. This is an infrastructure greenup, not a new
score claim; the current exact score remains the PR81 T4 replay at
`0.2812078926981712`.

Commands:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_build_qma9_context_run_escape_candidate.py -q

.venv/bin/python experiments/build_qma9_context_run_escape_candidate.py \
  --output-dir experiments/results/qma9_range_mask_deconstruction_20260503_codex/candidates \
  --candidate-id qma9_context_run_escape_min64_prefix4 \
  --frames 4 \
  --min-run-length 64

.venv/bin/python experiments/build_qma9_context_run_escape_candidate.py \
  --output-dir experiments/results/qma9_range_mask_deconstruction_20260503_codex/candidates \
  --candidate-id qma9_context_run_escape_min192_prefix4 \
  --frames 4 \
  --min-run-length 192

.venv/bin/python experiments/build_qma9_context_run_escape_candidate.py \
  --output-dir experiments/results/qma9_range_mask_deconstruction_20260503_codex/candidates \
  --candidate-id qma9_context_run_escape_min224_prefix4 \
  --frames 4 \
  --min-run-length 224
```

## 2026-05-03 Codex QMH1 Horizontal Run-Tail Escape Screen

Implemented a local-only deterministic `QMH1` planning prototype for true
horizontal run-tail escapes. Unlike `QMC1`, this is not a row-above copy flag:
the encoder first pays the normal QMA9 model for a run head, then may encode a
tail escape only immediately after a horizontal transition or row-start pixel.
The default gate requires `up != left`, so it targets horizontal tails where
the dominant QMA9 up predictor is not already the cheap explanation. Accepted
tails still update the base QMA9 adaptive model with the copied class symbols.

Scope: bounded PR84 prefix byte screen only. No runtime files, PR82/packedp
rerun state, Lightning/Vast/Modal/Azure dispatch, scorer path, training, or
exact eval were touched.

Input custody:

- PR84 archive:
  `experiments/results/top_submission_reverse_engineering_20260503_pr84/archive.zip`
- Archive bytes/SHA-256:
  `215735`,
  `a607a6c3ae9b610e6edfb546c3206004ae40fc348ecaef2446b7134a19b8e07f`
- Range-mask payload bytes/SHA-256:
  `159011`,
  `4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179`
- Prefix: first `4` frames, `786432` raw class bytes.

Code/artifacts:

- [empirical:experiments/build_qma9_horizontal_run_escape_candidate.py]
- [empirical:src/tac/tests/test_build_qma9_horizontal_run_escape_candidate.py]
- [empirical:experiments/results/qma9_horizontal_run_escape_20260503_worker/candidates/qma9_horizontal_run_escape_min192_updiff_prefix4/manifest.json]
- [empirical:experiments/results/qma9_horizontal_run_escape_20260503_worker/candidates/qma9_horizontal_run_escape_min192_allowup_prefix4/manifest.json]

Best nontrivial `up != left` sampled case:

- Candidate:
  `qma9_horizontal_run_escape_min192_updiff_prefix4`
- Manifest SHA-256:
  `3d4e928c5e8c218effe6f72279bbdf150b64885b85eac0bd85c8d9c1b2c481ef`
- Candidate payload SHA-256:
  `9fdef79e08129b251e8d37284778f84f90b63ea4ef7c53aba0fe4ef0af74c375`
- Baseline prefix QMA9: `1221` bytes.
- QMH1 prefix: `1233` bytes, delta `+12` bytes.
- Candidate positions: `442`; escaped runs: `4`; copied pixels: `881`.
- Linear projection: `184950` range-mask bytes versus source `159011`
  (`+25939` bytes). This projection is planning-only and non-dispatchable.

Threshold sweep on the same PR84 prefix:

| candidate | qma9 bytes | qmh1 bytes | delta | candidates | escaped runs | copied pixels | state change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `min16_updiff` | 1221 | 1246 | +25 | 739 | 25 | 2004 | true |
| `min32_updiff` | 1221 | 1245 | +24 | 739 | 17 | 1840 | true |
| `min64_updiff` | 1221 | 1240 | +19 | 739 | 12 | 1617 | true |
| `min96_updiff` | 1221 | 1237 | +16 | 733 | 7 | 1184 | true |
| `min128_updiff` | 1221 | 1234 | +13 | 721 | 4 | 881 | true |
| `min192_updiff` | 1221 | 1233 | +12 | 442 | 4 | 881 | true |
| `min224_updiff` | 1221 | 1229 | +8 | 4 | 0 | 0 | false |
| `min256_updiff` | 1221 | 1229 | +8 | 4 | 0 | 0 | false |
| `min192_allowup` | 1221 | 1854 | +633 | 4467 | 627 | 129197 | true |

Interpretation: `QMH1` is a measured local negative for the sampled PR84
prefix. The high-threshold variants converge to a no-op wrapper/header loss,
while allowing `up == left` confirms that broad horizontal tail flags repeat
the earlier copy-flag failure mode. Do not dispatch QMH1 from this screen. The
useful remaining path is a zero-or-near-zero-side-cost model change, likely
compiled full-stream context/backoff or a deterministic transform with sparse
exceptions, not another explicit run flag family.

Dispatch gate:

- `planning_only/no_remote_dispatch`
- `score_claim=false`
- `dispatch_performed=false`
- `gpu_required=false`
- Required before any orchestrator packet could dispatch: full `600`-frame
  QMH1 raw-mask parity, robust runtime/inflate integration that consumes QMH1
  bytes from the archive, deterministic archive closure and runtime manifest,
  lane claim, and exact CUDA auth eval through
  `archive.zip -> inflate.sh -> upstream/evaluate.py`.

Verification:

```bash
.venv/bin/python -m py_compile \
  experiments/build_qma9_horizontal_run_escape_candidate.py \
  experiments/profile_qma9_range_mask_bitstream.py \
  src/tac/tests/test_build_qma9_horizontal_run_escape_candidate.py \
  src/tac/tests/test_profile_qma9_range_mask_bitstream.py \
  src/tac/tests/test_qma9_range_mask_byte_search.py \
  src/tac/tests/test_qma9_range_mask_contract.py

.venv/bin/python -m pytest \
  src/tac/tests/test_build_qma9_horizontal_run_escape_candidate.py \
  src/tac/tests/test_profile_qma9_range_mask_bitstream.py \
  src/tac/tests/test_qma9_range_mask_byte_search.py \
  src/tac/tests/test_qma9_range_mask_contract.py -q
```

Result: `22 passed`.
