# PR91 HPM1 First-Symbol Contract Probe - 2026-05-04

Scope: PR91/HPM1 public PR, source, archive, and local replay-contract
forensics only. No remote/GPU job was dispatched, no scorer was loaded, and no
PR91 score claim is made here. The current verified frontier remains the
user-provided PR85+STBM score `0.25369011029397787`.

## Live Public PR Check

Checked live with `gh api` on 2026-05-04 from the public PR:

- PR: `https://github.com/commaai/comma_video_compression_challenge/pull/91`
- title: `Hpac coder hybrid`
- author: `ottokunkel`
- state: `open`
- updated at: `2026-05-04T04:44:01Z`
- head: `ottokunkel/comma_video_compression_challenge@77f958d24e55980d95e01e3e9767b5a94320ed43`
- PR-reported exact score: `0.24879480490416128`
- PR-reported archive bytes: `222404`
- PR files: only `inflate.py`, `inflate.sh`, `pr86_hpac.py`, and
  `range_mask_codec.cpp` under `submissions/hpac_coder_hybrid/`
- PR body says no compression script is included.

The only issue comment was the GitHub Actions welcome/eval-trigger note. No
author-side token-generation recipe or probability trace was present in the PR
discussion.

## Archive And Runtime Contract

Downloaded PR91 archive custody remains:

- archive:
  `experiments/results/public_pr91_intake_20260504_codex/archive.zip`
- archive bytes/SHA-256: `222404`,
  `4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f`
- single stored ZIP member `x`: `222304` bytes,
  `5c213c61cc4d29b62286063bfdcb97e812af6b06c0021aeaecc8bc46644e17bf`
- HPM1 mask segment: `145087` bytes,
  `a4ed57ff0af1d8c914f004de165aeead50ec8dd61e99b0afdfbfa2d1e7fd9fcc`
- HPM1 token stream: `116796` bytes / `29199` uint32 words,
  `541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b`
- HPM1 HPAC PPMd model: `28243` bytes,
  `de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd`
- HPM1 config: `N=600`, `H=384`, `W=512`, `P=32`, `delta=2`,
  `ch=64`, `use_spm=true`, `hpac_d_film=8`, `ppmd_order=4`

PR91's `pr86_hpac.py` is byte-identical to the merged PR86 HPAC inflate source
hash `f86f3067386928478d983817c9f9ee095ce6eb02aee8c0fbb7987cd0af1f9b01`.
PR91 also embeds the same HPAC PPMd model as PR86. The PR91 token stream is not
PR86's token stream: it is `+2896` bytes, with only a `164` byte / `41` uint32
word common prefix.

The submitted HPM1 branch passes `str(device)` to PR86 HPAC decode. A source
comment says HPAC should be forced to CPU, but the code does not force CPU, and
no fallback around HPM1 entropy failure exists in the actual HPM1 branch.

## Corrected PR85 Reference Layout

Important correction: the previous prefix probe compared PR91 decoded symbols
against the PR85 QMA9 token-source file as if it were stored `[N,H,W]`. The PR85
token-source profile records storage order as `frame_major_header_width_by_header_height`
with shape `[600,512,384]`. The render-order comparison must reshape as
`[N,W,H]` and transpose to `[N,H,W]`.

Correct render-order SHA-256:

```text
0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45
```

That is the same render-order token SHA already recorded for PR90 STBM parity
against PR85. Therefore the old `global_symbol=7` PR91-vs-PR85 mismatch was a
probe-layout artifact, not a real token mismatch.

## First-Symbol Probe

Added deterministic local probe:

- `src/tac/pr91_hpm1_codec.py::run_pr91_hpm1_first_symbol_state_probe`
- CLI:
  `experiments/replay_pr91_hpm1_mask.py --first-symbol-state-probe`
- artifact:
  `experiments/results/public_pr91_intake_20260504_codex/diagnostics/pr91_hpm1_first16_symbol_state_all_variants_20260504_codex.json`
- extended artifact:
  `experiments/results/public_pr91_intake_20260504_codex/diagnostics/pr91_hpm1_first64_symbol_state_all_variants_20260504_codex.json`

Command for the first-16 artifact:

```text
.venv/bin/python experiments/replay_pr91_hpm1_mask.py \
  --archive experiments/results/public_pr91_intake_20260504_codex/archive.zip \
  --first-symbol-state-probe \
  --symbol-count 16 \
  --reference-layout qma9_storage_wh_to_render_hw \
  --probability-variants source_float64_perfect_false,source_float32_perfect_false,source_float64_perfect_true,source_float32_perfect_true \
  --json-out experiments/results/public_pr91_intake_20260504_codex/diagnostics/pr91_hpm1_first16_symbol_state_all_variants_20260504_codex.json
```

Result:

- status: `passed` as a prefix diagnostic
- elapsed: `5.341s`
- all four variants decode the first 16 submitted symbols as:
  `[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]`
- corrected PR85 render-order reference first 16:
  `[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]`
- first-16 mismatch: none

The first-64 artifact narrows the first real local source-contract mismatch:

- source variant decoded symbols 0..63:
  `[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4,4,2,2,0,4,4,2,2,0,4,4,4,0,2,2,4,2,4,2,4,4,2,2,2,2,2,2,2,2]`
- PR85 render-order reference symbols 0..63:
  all `2`
- first mismatch: `global_symbol=33`, `frame=0`, `group=0`,
  `symbol_in_group=33`, `pixel=(y=64,x=32)`, decoded `4`, reference `2`

At the original entropy assertion point, the corrected reference symbol is also
the model argmax:

- failure coordinate: `frame=0`, `group=10`, `symbol_in_group=191`,
  `pixel=(y=37,x=480)`
- decoded symbols before failure: `5951`
- corrected PR85 reference symbol at failure: `2`
- source-contract probability row at failure:
  `[0.032410533766438016, 0.00005810932545209993, 0.9667295673977898, 0.0007811914334127637, 0.000020598076907271897]`

## Comparison To PR86 And PR90

PR86 HPAC:

- same HPAC runtime and same HPAC model as PR91
- PR86 and PR91 both fail local source-contract HPAC decode at the same frame 0
  group 10 symbol 191 under `source_float64_perfect_false`
- PR91's HPM1 stream is not a simple PR86 token copy; it diverges at uint32 word
  41 and changes early decoded symbols by global symbol 33
- no merged PR86/PR91 public source includes the exact token-generation trace
  that produced the submitted queue stream

PR90 STBM:

- PR90 uses a deterministic custom range coder and semantic topband/road-boundary
  decomposition, not neural HPAC probabilities
- local STBM analysis proved exact PR85 render-order token parity with SHA
  `0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45`
- PR90/PR85+STBM is therefore a recovered lossless mask-representation contract;
  PR91/HPM1 is still an unrecovered neural entropy contract

## Missing Contract After This Pass

The missing contract is no longer PR85 QMA9 token-source layout for the first
16 symbols. That is fixed in local tooling.

The remaining unrecovered contract is the PR86/PR91 HPAC token-generation state
that maps the submitted uint32 queue stream to the intended token tensor:

1. Why the source-contract decoder emits class `4` at global symbol `33` while
   corrected PR85 render-order reference is class `2`.
2. Why the source-contract decoder later asserts at global symbol `5951` even
   though the corrected reference symbol at failure has probability `0.9667`.
3. Whether the submitted token stream was generated with a different
   constriction version/API, different categorical quantization, different
   previous/current token semantics, different PR85-like token source, or a
   CUDA-vs-CPU probability trace not represented by the public source.

No candidate is dispatchable from this evidence. If parity recovers, the next
candidate gate is local-only first:

1. full 600-frame PR91 HPM1 decode under the recovered contract;
2. byte-exact re-encode of `541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b`;
3. corrected PR85/PR90 render-order token comparison and runtime output-parity
   preflight;
4. only after those pass, claim a lane with `tools/claim_lane_dispatch.py`
   before any remote exact eval.

