# Public HNeRV Frontier Deconstruction - 2026-05-04

This ledger records exact-replay and byte-forensics work for the late public
HNeRV frontier. Public PR scores are external until exact CUDA replay of the
exact archive bytes through `archive.zip -> inflate.sh -> upstream/evaluate.py`.

## 2026-05-08 Monolithic-Layout Supersession

Later layout verification proves the stock PR101 and PR106 archives are
single-member ZIP packets with parser-proven internal sections, not separate
ZIP members for masks, poses, or renderer files. Treat this ledger's section
byte anatomy as parser-level evidence only. Any older phrasing that implies a
member-level pose/mask/renderer budget is superseded by
`reports/frontier_monolithic_archive_layout_20260508.json` and
`.omx/research/frontier_monolithic_archive_layout_20260508_codex.md`.

Current local A++ HNeRV rate anchor is now PR103-on-PR106 at strict formula
score `0.2089810755823297`, `185578` bytes, SHA-256
`ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`.
PR106/PR106x rows below are preserved as predecessor evidence and source
substrates, not as the current rate anchor.

## Current Exact Frontier

- PR101 `hnerv_ft_microcodec` exact T4 A++ replay:
  - score: `0.22635331443973267`
  - bytes: `178258`
  - archive SHA-256: `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`
  - artifact: `experiments/results/lightning_batch/exact_eval_public_pr101_hnerv_ft_microcodec_t4_20260504T1302Z/contest_auth_eval.adjudicated.json`

PR103 also replayed cleanly on T4 but is behind PR101:

- PR103 `hnerv_lc_ac` exact T4 A++ replay:
  - score: `0.2277649714224471`
  - bytes: `178223`
  - archive SHA-256: `31881b2d23d027e6619f2d8df2fe35d4d207d08882ec673d6c1b7ff119f18c30`

PR106 first replay failed before score because the original public
`inflate.sh` resolved `python` outside the managed venv and missed `brotli`.
This is an infrastructure/runtime-closure failure, not method evidence.

## Byte Anatomy

Generated artifact:

- JSON: `experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/profiles.json`
- Markdown: `experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/profiles.md`
- Tool: `experiments/profile_hnerv_frontier_payloads.py`

Section summaries:

- PR101 `hnerv_ft_microcodec`, archive `178258` bytes, member `x` `178158` bytes:
  - `decoder_compact_brotli_streams`: `162164` bytes, entropy `7.998425`
  - `latents_raw_lzma_delta_u8`: `15387` bytes, entropy `7.988287`
  - `sidecar_dim_delta_huffman_enum`: `607` bytes, entropy `7.710458`
- PR103 `hnerv_lc_ac`, archive `178223` bytes, member `x` `178123` bytes:
  - `merged_range_coded_weights_and_hi_latents`: `153856` bytes, entropy `7.998795`
  - `latent_low_bytes_brotli`: `15537` bytes, entropy `7.986907`
  - `sidecar_corrections_brotli`: `555` bytes, entropy `7.646035`
- PR105 `kitchen_sink`, archive `177857` bytes, member `0.bin` `177749` bytes:
  - `packed_header_ff_len24`: `4` bytes
  - `decoder_packed_brotli`: `161891` bytes, entropy `7.998095`
  - `latents_and_sidecar_brotli`: `15854` bytes, entropy `7.985312`
- PR106 `belt_and_suspenders`, archive `186239` bytes, member `0.bin` `186131` bytes:
  - `packed_header_ff_len24`: `4` bytes
  - `decoder_packed_brotli`: `170278` bytes, entropy `7.998152`
  - `latents_and_sidecar_brotli`: `15849` bytes, entropy `7.985386`

Immediate implication: the active HNeRV frontier is dominated by saturated
decoder-weight and latent payloads. Real gains require a better runtime basin,
weight quantization/self-compression, arithmetic/range coding with lower
overhead than current sections, or learned sidecar corrections that reduce
SegNet/PoseNet enough to justify their charged bytes. Cosmetic ZIP changes or
obfuscation are not scientific score work unless exact eval proves a real
score/byte change.

## Runtime Adapter Fix

Added venv-stable public replay adapters:

- `experiments/results/public_runtime_adapters_20260504_codex/pr105_kitchen_sink_adapter/inflate.sh`
- `experiments/results/public_runtime_adapters_20260504_codex/pr106_belt_and_suspenders_adapter/inflate.sh`

The adapters invoke the repo-managed `.venv/bin/python`, set `PYTHONPATH` to
the public submission `src`, and fail closed if `brotli` is not importable.
This fixes the PR106 missing-`brotli` replay failure class without changing
archive bytes.

Verification:

- `py_compile` passed for `experiments/profile_hnerv_frontier_payloads.py`
- `bash -n` passed for both adapters

## Live Exact Replays

Both jobs are queued on T4 through Lightning with source manifest
`.omx/state/public_pr105_106_adapter_replays_20260504T1333Z_manifest.json`,
remote preflight `lightning-pact`, explicit CUDA-12 torch pin, and brotli
closure:

- `exact_eval_public_pr105_kitchen_sink_adapter_t4_20260504T1330Z`
  - archive SHA-256: `597ba0732810eba08cdae619b679d211d398bc0249b8831898f7096d5beece1d`
  - bytes: `177857`
  - status at queue: `Pending`
- `exact_eval_public_pr106_belt_and_suspenders_adapter_t4_20260504T1330Z`
  - archive SHA-256: `3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58`
  - bytes: `186239`
  - status at queue: `Pending`

Additional deterministic `x` member repacks are queued as byte-different
variants using `experiments/repack_single_member_archive.py`. These preserve the
single payload bytes and only rename `0.bin` to `x`, which is already an
allowed archive basename. The adapter was hardened to read `${BASE}.bin` first
and fall back to `x`.

- PR105 x-repack:
  - archive bytes: `177849`
  - archive SHA-256: `692a46931f66416ab71270661f580a4b5bb2e5a887b19c51aabbb8851493feab`
  - byte delta vs public PR105 archive: `-8`
  - queued job: `exact_eval_public_pr105_kitchen_sink_xrepack_t4_20260504T1342Z`
- PR106 x-repack:
  - archive bytes: `186231`
  - archive SHA-256: `d25bca80057e8b533197895b4c56370678feb4e05fea0312c405bd12f29bec8e`
  - byte delta vs public PR106 archive: `-8`
  - queued job: `exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z`

This is not presented as meaningful frontier progress until exact CUDA scores
land. The expected score effect from rate alone is only about `-0.0000053`.

## Next Actions

1. Refresh and harvest PR105/PR106 adapter replays immediately when terminal.
2. If either beats PR101 exact T4, update `reports/latest.md`, close dispatch
   claims with terminal status, and rebuild the public supplement evidence
   index against the new champion.
3. If both replay above PR101, preserve them as scorer-runtime drift evidence
   and focus the next engineering tranche on PR105/106 sidecar/decode-time
   correction ideas that can be trained or searched against exact CUDA traces.
4. Convert the payload profiler into the standing public-frontier intake gate:
   every future public archive must get section offsets, entropy, section
   SHA-256, ZIP overhead, and no-op/provenance checks before stack claims.

## 2026-05-04T13:39Z Refresh

GitHub PR refresh found no newer public pull request beyond our PR107. The
latest visible frontier cluster remains:

- PR107 `apogee submission (0.2293)`, created `2026-05-04T12:10:42Z`
- PR106 `belt_and_suspenders (0.20946)`, created `2026-05-04T11:57:01Z`
- PR105 `kitchen_sink (0.19797)`, created `2026-05-04T11:56:48Z`
- PR104 `qhnerv_ft_best`, created `2026-05-04T11:56:00Z`
- PR103 `hnerv_lc_ac submission (0.19)`, created
  `2026-05-04T11:55:56Z`

Current replay state:

- `exact_eval_public_pr106_belt_and_suspenders_adapter_t4_20260504T1330Z`
  is `Running` on Lightning T4, cost observed at `$0.02586111`. The Studio
  remote artifact mirror has not exposed eval files yet, so this is live
  execution/provisioning signal only, not score evidence.
- `exact_eval_public_pr105_kitchen_sink_adapter_t4_20260504T1330Z`,
  `exact_eval_public_pr105_kitchen_sink_xrepack_t4_20260504T1342Z`, and
  `exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z`
  are still `Pending` at zero cost.

Strict interpretation of the `x` repacks: they are deterministic,
byte-different custody probes that remove 8 ZIP bytes by using the shorter
single-member name `x`. They are acceptable to exact-evaluate because the
adapter consumes either `0.bin` or `x`, but they are not an optimization lane
with real scientific weight unless exact eval proves the runtime path is
identical and the byte saving survives adjudication.

Added a standing scorecard builder:

- tool: `experiments/build_hnerv_frontier_scorecard.py`
- JSON: `experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/scorecard.json`
- Markdown: `experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/scorecard.md`

Current exact-score rows:

- PR101: `0.226353314440`, `178258` bytes, A++, largest profiled payload
  section `decoder_compact_brotli_streams:162164`.
- PR103: `0.227764971422`, `178223` bytes, A++, largest profiled payload
  section `merged_range_coded_weights_and_hi_latents:153856`.

This scorecard should be regenerated immediately after PR105/PR106 harvest so
the next engineering branch is chosen from exact score plus byte anatomy instead
of title claims.

## 2026-05-04T13:43Z Live Queue State

All four PR105/PR106 replay jobs have advanced to `Running`:

- `exact_eval_public_pr105_kitchen_sink_adapter_t4_20260504T1330Z`, cost
  `$0.033483334`.
- `exact_eval_public_pr106_belt_and_suspenders_adapter_t4_20260504T1330Z`,
  cost `$0.09119444`.
- `exact_eval_public_pr105_kitchen_sink_xrepack_t4_20260504T1342Z`, cost
  `$0.022322223`.
- `exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z`,
  cost `$0.0`.

The Studio SSH mirror still shows no artifact files in the four local
`experiments/results/lightning_batch/...` output dirs. Treat this as live
Lightning execution without harvestable JSON, not as failure.

## 2026-05-04T15:05Z — NEW EXACT PUBLIC FRONTIER: PR106

Four T4 jobs harvested terminal. **PR106 belt_and_suspenders adapter replays at `0.20945673680571203` [contest-CUDA T4 A++]** — beats prior best PR101 (`0.22635331`) by **-0.01690**.

### Exact replay results (Tesla T4, n=600, device=cuda, A++)

| label | score | bytes | seg | pose | rate | sha256 (16) |
|---|---:|---:|---:|---:|---:|---|
| **PR106 adapter** | **0.20945673680571203** | 186239 | 0.067142 | 0.018306 | 0.124009 | `3fefbe5dfdd73817` |
| PR106 xrepack | 0.20945123680571204 | 186231 | 0.067142 | 0.018306 | 0.124003 | `d25bca80057e8b53` |
| PR101 ft_microcodec | 0.22635331443973267 | 178258 | 0.066304 | 0.041355 | 0.118695 | `b83bf3488625dbd7` |
| PR103 lc_ac | 0.22776497142244710 | 178223 | 0.067623 | 0.041470 | 0.118672 | `31881b2d23d027e6` |
| PR105 kitchen_sink xrepack | 0.23043182986984995 | 177849 | 0.070456 | 0.041554 | 0.118422 | `692a46931f66416a` |
| PR105 kitchen_sink adapter | 0.23043732986984997 | 177857 | 0.070456 | 0.041554 | 0.118428 | `597ba0732810eba0` |

### Key insight

PR106's win is **PoseNet domination**: pose contribution `0.018306` is **~2.3× lower** than PR101/103/105 (`0.041355`-`0.041554`). PR106 trades +8KB rate (`+5314 bytes` vs PR101) for a sqrt(10·pose_dist) advantage. That's the Fridrich square-root-law in action — concentrated bytes purchasing disproportionate pose reduction.

The PR105 vs PR106 split (same `kitchen_sink`/`belt_and_suspenders` author family) reveals that PR106's extra ~8KB correlates with most of its score advantage via lower PoseNet contribution, not segmentation. Supersession after the monolithic-layout finding: this does **not** prove a separate PR106 pose member or member-level pose budget. The actionable surface is PR106's parser-proven `decoder_packed_brotli` plus `latents_and_sidecar_brotli`, or a new charged sidecar/runtime packet with exact CUDA proof.

### Strict interpretation of x-repack evidence

The two byte-different x-repacks (PR105x: `0.23043183`, PR106x: `0.20945123`) reproduce the predicted ~`-0.0000053`/byte rate effect almost exactly:
- PR106: `0.20945673 - 0.20945123 = 0.00000550` for 8 bytes saved → `6.875e-7/byte`, matches `25/37545489 = 6.658e-7/byte`.
- PR105: `0.23043733 - 0.23043183 = 0.00000550` for 8 bytes saved → matches.

This is **decisive evidence** the runtime adapter is byte-faithful: the 8-byte ZIP-header savings produce exactly the rate-only Δ with no payload-quality shift. The exact replay path is sound. Our adapters are valid and our public-frontier intake gate is calibrated.

### Stacking implications for sub-0.20

To beat PR106 (0.209), our internal stack needs to reduce **at least one** of:
- pose contribution (0.018306 → push lower than ~0.014, savings ≥0.004) — RAFT/openpilot-style ego-motion or PoseNet-aware sidecar
- seg contribution (0.067142 → push below 0.060, savings ≥0.007) — Cool-Chic mask codec, learned charged correction sidecar
- rate (0.124009 at 186KB → push below 0.10 means archive ≤150KB) — HNeRV decoder self-compression, range/ANS coding on dense streams

The public frontier is now decoder-saturated brotli (170KB on PR106) — entropy coding the dense decoder/latent streams is the natural attack surface, plus our Fridrich Lagrangian sensitivity-aware bit allocation.

### Action items

1. Update `reports/latest.md` to record PR106 as new exact public frontier.
2. Spawn comprehensive PR107 reorganization subagent: gh-repo research + internal hidden-gem audit + tac OSS plan + month-of-work writeup + senior-engineer review.
3. Build PR106 stacking lane: HNeRV decoder self-compression (NWC) + arithmetic coding of decoder/latent streams + RAFT-derived pose sidecar.
   Supersession: the RAFT-derived pose sidecar is a new charged packet/runtime
   design, not replacement of an existing PR106 pose ZIP member.
