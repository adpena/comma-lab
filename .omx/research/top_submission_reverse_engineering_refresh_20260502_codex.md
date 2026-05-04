# Top Submission Reverse Engineering Refresh - 2026-05-02

Evidence grade: `external_plus_empirical_byte_anatomy`.

Score claim: `false`. Public GitHub PR reports and public archive anatomy are
design signals only. C-059 remains the current internal exact A++ frontier until
a repo-built archive passes exact CUDA auth eval on identical bytes.

## Scope And Source Boundary

- Repo inspected: `commaai/comma_video_compression_challenge`
- Live open PR set observed from GitHub API: `#65`, `#67`, `#69`, `#70`, `#71`
- Deterministic local anatomy artifact:
  `experiments/results/top_submission_reverse_engineering_refresh_20260502/archive_anatomy.json`
- Deterministic structured summary:
  `experiments/results/top_submission_reverse_engineering_refresh_20260502/public_pr_summary.json`
- No remote jobs were launched.

## Active Internal Comparator

| Label | Evidence | Score | PoseNet | SegNet | Bytes | SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| C-059 QZS3/QP1 b32 mask-first packed layout | A++ exact T4 CUDA | `0.3157055307844823` | `0.00049637` | `0.00061244` | `276347` | `cf44aa7fdb13b9ab6236aefde1f5f58e915d7dfa99128246235443841396c6ab` |

Source artifact:
`experiments/results/lightning_batch/exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z/contest_auth_eval.json`.

## Public PR Intelligence

| PR | Public state | Public reported score | PoseNet | SegNet | Bytes | Classification |
|---:|---|---:|---:|---:|---:|---|
| #67 `qpose14_qzs3_filmq9g_slsb1_r55` | open | `0.31` rounded; formula from rounded fields `0.314864164052394` | `0.00048597` | `0.00061000` | `276564` | `external_strict_valid_looking` |
| #65 `henosis_qz_n3z_r25_clean` | open | `0.32` rounded; author exact approx `0.31968005`; formula from rounded fields `0.319682427689121` | `0.00035283` | `0.00070896` | `284425` | `external_strict_valid_looking` |
| #71 `tomasdousek` / `ditcher` | open | `0.71` rounded; formula from rounded fields `0.711366091253782` | `0.00358563` | `0.00328412` | `290747` | `external_nonfrontier_validity_unconfirmed` |
| #70 `mask_decoder` | open | `0.19` rounded; formula from rounded fields `0.185435672878816` | `0.00048759` | `0.00077435` | `57329` | `invalid_exploit_quarantine` |
| #69 `houdini` | open | none; maintainer eval requested | n/a | n/a | archive linked, not scored | `external_quarantine_indeterminate` |
| #68 `loophole_v2` | closed | expected around `0.13`, not local-run | n/a | n/a | `22` byte empty zip | `invalid_exploit_quarantine` |
| #64 `unified_brotli` | closed | live PR body now reports `0.34`; formula from rounded fields `0.336126770676108` | `0.00052868` | `0.00072205` | `287165` | `external_historical_valid_looking` |
| #63 `qpose14` | closed | `0.32` rounded; formula from rounded fields `0.324961783081994` | `0.00052154` | `0.00061261` | `287573` | `external_historical_valid_looking` |
| #62 `fp4_mask_gen` | closed | `0.36` rounded; formula from rounded fields `0.359066121006115` | `0.00063958` | `0.00112878` | `249624` | `external_historical_valid_looking` |
| #56 `selfcomp` | closed | `0.36` rounded; formula from rounded fields `0.364116043771751` | `0.00039717` | `0.00115296` | `279036` | `external_historical_valid_looking_cpu_report` |
| #55 `quantizr` | closed | `0.33` rounded; formula from rounded fields `0.332271995510734` | `0.00051010` | `0.00061113` | `299970` | `external_historical_valid_looking` |

Note: `archive_anatomy.json` still records the pinned PR64 archive anatomy from
the local reverse-engineering script, whose hard-coded reported-score fields are
older than the live PR body. Use the live PR body above for current public
reported score/components; use `archive_anatomy.json` for byte anatomy only.

## Architecture Signals

- PR67 is the current public contest-faithful target. Its archive is a single
  zip member `p` with `276464` payload bytes, SHA
  `a64523528fad2adee466f948ad29839e8581ccfa70caf934b5e4b4a6d733f76f`.
  The fixed slices are `219472` Brotli mask bytes, `56093` QZS3 model bytes,
  and `899` QP1 pose bytes. QZS3 decodes to `111` state-dict keys and
  `87836` finite parameters.
- PR65 is a Quantizr-derived, archive-counted variant with member `x`, `284325`
  payload bytes, SHA
  `1adc465971038009a0e0f2c1fc7209302256ef18ad3f52844fbb9832a9cb3cc6`.
  It uses a compact 24-bit length table; the first lengths are `219472`,
  `57074`, `1487`, `1400`, then small post/shift/bias-like payloads.
- Quantizr PR55 established the durable basin: 88k-param depthwise-separable
  FiLM-conditioned JointFrameGenerator, half-rate mask stream, scorer-contract
  training with upsample/round/downsample, and a five-stage QAT-like process.
- Selfcomp PR56 remains relevant for weight compression, not for current score:
  it reports self-compression around `1.017` bits per weight plus an affine
  learned-image pose path, but its SegNet is far weaker than PR67/C-059.
- PR71 is not frontier-relevant today, but it adds a generator that consumes
  masks, inter-pair poses, and intra-pair poses with depthwise-separable
  residual blocks, squeeze-excitation, pixel attention, and boundary-loss
  diagnostics. Treat it as post-contest architecture signal, not a C-059
  replacement.

## Exploit Forensics

- PR68 explicitly demonstrates archive-rate metering failure: the same kind of
  compressed payload is moved into `inflate.py` while `archive.zip` is nearly
  empty. This is invalid for our strict custody standard.
- PR70 publicly states it moved bytes from the archive into `inflate.py`; local
  forensic artifacts also show `inflate.py` at `1299244` bytes and
  `archive.zip` at `57329` bytes. The local PR70 archive has central member
  `m` with `57230` bytes, and the existing provenance classifies the ZIP as
  malformed because Python `zipfile` rejects the empty local-header filename
  while `unzip` extracts `m`.
- PR69 has no reported score and says the data-flow boundary was refactored.
  Its raw `inflate.py` contains an embedded base85 payload literal, so it stays
  quarantined until a strict archive-metered maintainer result exists.

## C-059 Implications

1. PR67 beats C-059 by public rounded-field formula by about
   `0.000841366732089` despite being `217` bytes larger. The gap is component
   quality: PR67 gains about `0.000741982243369` in PoseNet contribution and
   `0.000244000000000` in SegNet contribution, partly offset by a
   `0.000144491392828` worse rate term.
2. More byte shaving alone is low EV unless it is free. C-059 already has the
   rate edge over PR67; the remaining public gap requires charged pose/SegNet
   atoms that keep the existing b32 mask-first packer shape.
3. PR65 is the strongest pose clue: PoseNet `0.00035283` is much better than
   both PR67 and C-059, but SegNet `0.00070896` pays most of it back. Mine
   PR65 for pose residual/manifold ideas, but gate every transfer against the
   C-059 SegNet trust region.
4. Do not promote PR64 as a `0.33` target from stale local notes. The live PR64
   body currently reports `0.34`; its useful transfer is single-stream Brotli
   and velocity-delta packing, not its current score.
5. PR68/69/70 are validator hardening inputs, not score basins. Submission
   packets and archive validators must keep rejecting unmetered script payloads,
   malformed ZIP central/local header divergence, and any dependency on bytes
   outside `archive.zip`.

## Exact URLs

- PR55 Quantizr: https://github.com/commaai/comma_video_compression_challenge/pull/55
- PR56 Selfcomp: https://github.com/commaai/comma_video_compression_challenge/pull/56
- PR62 FP4 mask generator: https://github.com/commaai/comma_video_compression_challenge/pull/62
- PR63 qpose14: https://github.com/commaai/comma_video_compression_challenge/pull/63
- PR64 unified_brotli: https://github.com/commaai/comma_video_compression_challenge/pull/64
- PR65 henosis: https://github.com/commaai/comma_video_compression_challenge/pull/65
- PR67 faithful qpose14/QZS3: https://github.com/commaai/comma_video_compression_challenge/pull/67
- PR68 loophole_v2: https://github.com/commaai/comma_video_compression_challenge/pull/68
- PR69 houdini: https://github.com/commaai/comma_video_compression_challenge/pull/69
- PR70 mask_decoder: https://github.com/commaai/comma_video_compression_challenge/pull/70
- PR71 ditcher: https://github.com/commaai/comma_video_compression_challenge/pull/71

## QPost Bias A-Negative - 2026-05-02T04:40Z

A C-059 archive plus the charged PR65-derived `qpost.bin` bias atom was exact
CUDA screened on H100.

Artifact:

- Local harvest: `experiments/results/vast_harvest/archive_eval_c059_qpost_bias_20260502/contest_auth_eval.json`
- Archive bytes: `276700`
- Archive SHA-256: `b3e5fa6a1333bfdfcb6473a1ff20badbe2c03de57e2cac87e230c7177bfeb5bf`
- Recomputed score: `0.3636174334250468`
- PoseNet: `0.00102822`
- SegNet: `0.00077973`
- Hardware: H100 diagnostic CUDA, not T4 promotion.

Verdict:

- `A-negative` for the measured C-059 plus PR65-bias-qpost implementation.
- The atom is not stackable as-is: both PoseNet and SegNet regress, overwhelming
  the charged-byte perturbation.
- This does not kill qpost or PR65 transfer broadly; it narrows the failure to
  the measured global bias atom and strengthens the requirement for pair-gated
  or low-dimensional manifold-gated transfer.

## Pair-Gated QPost A-Negative - 2026-05-02T05:03Z

The next stricter PR65 transfer test was a charged, pair-filtered qpost archive
on top of C-059. The builder re-encoded the PR65 qpost streams so non-selected
pairs decode to the runtime identity defaults, then exact-screened only the
top-16 C-059 pose-manifold pairs.

Implementation and local byte screen:

- Builder: `experiments/build_qzs3_postprocess_candidate.py`
- Runtime: `submissions/robust_current/apply_qzs3_postprocess.py`
- Test: `src/tac/tests/test_qzs3_postprocess_candidate.py`
- Local candidates: `experiments/results/qzs3_postprocess_c059_pairgated_20260502/summary.json`
- Top16 archive bytes: `276893`
- Top16 archive SHA-256:
  `2080fdcee5dca4cea1ccc22e78c55a2e5c45aac651b64115014a2fe3851f502c`
- Active streams: `post`, `shift`, `frac`, `frac2`, `frac3`, `bias`,
  `region`; `randmulti` intentionally omitted until a sparse pair-safe encoder
  exists.

Exact CUDA diagnostic:

- Local harvest:
  `experiments/results/vast_harvest/archive_eval_c059_pairgated_qpost_top16_20260502/contest_auth_eval.json`
- Recomputed score: `0.3632133637856646`
- PoseNet: `0.00101771`
- SegNet: `0.00077960`
- Hardware: H100 diagnostic CUDA, not T4 promotion.
- Inflate plus qpost runtime completed in `75.1s`; full auth eval completed in
  `150.37s`, inside the 30-minute inflate/eval budget.

Verdict:

- `A-negative` for the measured pair-gated PR65 qpost top16 implementation.
- Pair gating reduced the global qpost blast radius slightly, but still
  regressed both PoseNet and SegNet enough to make top32/top64 not worth H100
  spend under the same stream family.
- Do not kill output-space correction broadly. Retire this exact transfer
  mechanism and require future postprocess atoms to be learned/scorer-aware at
  the pair or region level, with a cheaper encoding than the public PR65 qpost
  stream format.

## Public Leaderboard Refresh - 2026-05-02T12:30Z

Source: https://comma.ai/leaderboard and linked public GitHub PRs.

The visible faithful lossy video compression leaderboard still reports:

```text
0.32 qpose14        PR63
0.33 unified_brotli PR64
0.33 quantizr       PR55
0.37 fp4_mask_gen   PR62
0.38 selfcomp       PR56
```

PR63 qpose14 exact public report fields:

```text
PoseNet: 0.00052154
SegNet:  0.00061261
bytes:   287573
rounded final score: 0.32
recomputed from rounded fields: 0.3249617830819938
```

PR64 unified_brotli public eval fields recompute to
`0.33097206780134475` from rounded components. PR55 quantizr public eval
fields recompute to `0.3326422723352197` from rounded components.

Interpretation:

- The public page rounds scores, so exact leaderboard ordering below the
  visible two decimals must be inferred from PR component reports or local
  exact traces.
- Our current local A++ T4 C-067 frontier remains below the visible public
  faithful floor by measured internal score, but it is not a leaderboard rank
  until packaged, submitted, and accepted by the public workflow.
- The active CMG3 row-span stride1 T4 packet is the immediate threshold
  attempt because its formula-only score crosses 0.30 if components hold.

## Sub-0.30/Sub-0.24 Reverse-Engineering Consolidation - 2026-05-02T23:38Z

Evidence grade: `external_plus_empirical_byte_anatomy` plus cited exact local
negative artifacts. Score claim: `false`. No remote GPU jobs launched. No
SJ-KL or Lane12 implementation files touched.

Primary sources refreshed:

- Official visible leaderboard still lists faithful merged entries as:
  `0.32 qpose14 #63`, `0.33 unified_brotli #64`, `0.33 quantizr #55`,
  `0.37 fp4_mask_gen #62`, `0.38 selfcomp #56`.
- PR #67 is open and reports `qpose14_qzs3_filmq9g_slsb1_r55 (0.31)` with
  CUDA `600`-sample fields: PoseNet `0.00048597`, SegNet `0.00061000`,
  `276564` bytes.
- PR #65 is open and reports `henosis_qz_n3z_r25_clean (0.32)` with CUDA
  `600`-sample fields: PoseNet `0.00035283`, SegNet `0.00070896`,
  `284425` bytes.
- PR #55 Quantizr remains the architecture origin signal: quantizr-style
  JointFrameGenerator, GPU inflate, score `0.33`, and author note that the
  supplied five-stage file was a combined approximation of several training
  scripts.
- PR #56 Selfcomp remains the renderer self-compression signal: score `0.36`,
  PoseNet `0.00039717`, SegNet `0.00115296`, `279036` bytes, AV1 video, and
  reported weight self-compression around `1.017` bits per weight.
- PR #68 and PR #70 are exploit/validator inputs, not faithful score targets:
  PR #68 states the compressed payload was embedded in `inflate.py` instead of
  `archive.zip`; PR #70 states it realized the same issue after moving bytes
  into `inflate.py`.

Local artifact support:

- C067 / Apogee exact T4 anchor:
  `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.adjudicated.json`
  records score `0.31561703078448233`, bytes `276214`, SHA
  `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`,
  PoseNet `0.00049637`, SegNet `0.00061244`.
- Public archive anatomy:
  `experiments/results/top_submission_reverse_engineering_refresh_20260502/archive_anatomy.json`.
  PR67 has single member `p`, payload `276464` bytes, fixed slices:
  `219472` mask OBU+Brotli, `56093` QZS3 renderer+Brotli, `899` QP1
  pose+Brotli. QZS3 decodes to `111` keys and `87836` finite parameters.
  PR65 has single member `x`, payload `284325` bytes, 24-bit length-table
  bundle with leading streams `219472`, `57074`, `1487`, `1400`, `226`,
  `106`, `149`, `154`, `223`, `273`, then a large `randmulti` tail.
- Byte accounting:
  `experiments/results/c067_archive_byte_accounting_20260502/archive_byte_accounting.json`
  shows the C067 streams are already generic-recompression tight:
  mask `219472` bytes, renderer `55965` bytes, pose `677` bytes, no nested
  savings. Unchanged-distortion sub-0.30 needs `23454` bytes removed; sub-0.24
  needs a representation change, not packer polish.
- Negative transfer evidence:
  PR65 qpost global and pair-gated transplants exact-screened on H100 at
  `0.3636174334250468` and `0.3632133637856646`; PMG row-span/residual
  exact screens scored `30.930370939355445` T4 and `28.41411894150047` L40S;
  old Q-FAITHFUL snapshots scored `8.420915711675079` and `22.065520725118258`.

Decision-quality conclusions:

1. PR67/C067 basin is still the only faithful sub-0.33-class architecture with
   local exact custody. Keep the JFG + QZS3 + QP1 + single-blob archive grammar
   as the deploy basin until a different complete archive exact-evals below it.
2. PR65's useful signal is not the measured qpost stream transfer. The useful
   signal is its PoseNet basin: it gets much lower PoseNet while paying SegNet
   and rate. Future PR65 mining must be learned/pair/region gated and
   component-trace calibrated; do not rerun global or top16 qpost transplants.
3. Sub-0.30 cannot come from pose bytes or zip overhead. It requires either
   about `10.7%` of the charged mask stream without component collapse, a real
   renderer self-compression export, or a stacked distortion win large enough
   to offset bytes. Sub-0.24 requires a larger mask/topology/decoder
   representation change.
4. The public exploit submissions invalidate naive "score" comparisons but
   strengthen our validator posture: no script-side payload movement, no
   malformed ZIP central/local divergence, no unmetered constants.

Top 3 implementation moves from this signal:

1. **Mask topology replacement with scorer-trust-region atoms.** Build the
   next mask-side archive around multimask/JFG-native/foveated/learned
   topology atoms, not row-span PMG or plain CRF grayscale. Gate by decoded
   mask SHA, exact component trace, and charged payload bytes. This is the only
   single stream big enough for sub-0.30 and the only plausible stream for
   sub-0.24.
2. **Geometry-closed Q-FAITHFUL/JFG successor export.** Reuse Quantizr/PR67
   architecture and packer lessons, but require nonzero deployed pose training,
   EMA export provenance, full/half-frame geometry parity, QZS3 or Torch-FP4
   deterministic export, and runtime closure before H100 spend. Old collapsed
   snapshots are no-retry unless these gates change.
3. **PR65-informed low-dimensional correction, not PR65 qpost copy.** Mine
   PR65 for pose/residual/manifold bases, then fit/select small charged atoms
   against C067 component traces. The acceptance criterion is exact stacked
   archive R(D), not local image quality or a wholesale public-side-channel
   transplant.

Do-not-rerun list:

- PR68/PR70-style script-payload or malformed ZIP mechanics: exploit
  quarantine only.
- PR69 Houdini until a strict archive-metered maintainer result and payload
  boundary are available.
- Plain CRF grayscale, broad PMG row-span, PMG row-run residual rescue, and
  old Q-FAITHFUL snapshots without geometry/training-contract changes.
- PR65 global qpost and pair-gated top16 qpost transfers as implemented.
- Global QZS block-size scalar sweeps around C067; use mixed/local learned
  allocation only if local byte/runtime contracts prove a real archive change.
