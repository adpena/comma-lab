# Top Submission Delta Reverse Engineering - 2026-05-03

Worker scope: refreshed public top-submission code and archive anatomy against
the C-089 exact A++ frontier. No remote GPU dispatch. No upstream scorer edits.

## Anchor

C-089 exact frontier:

- candidate: `c067_pr75_qp1_top40_p6`
- score: `0.3154707273953505`
- bytes: `276342`
- SHA-256: `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`
- PoseNet: `0.00049601`
- SegNet: `0.00061038`
- evidence: A++ exact T4 CUDA,
  `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/contest_auth_eval.adjudicated.json`

Score gap to hard `0.3140000000000000` is `0.0014707273953505`.
Byte-only closure would require about `2209` fewer archive bytes at
`25 / 37,545,489 = 6.658586e-7` score per byte. To beat the PR67 body score
recomputed from public rounded components, C-089 needs `0.0007625523611846`
score improvement, or about `1146` byte-equivalent.

## Public Sources Refreshed

Source metadata and archives are stored under
`experiments/results/top_submission_delta_reverse_engineering_20260503/`.

Public PR heads as of this pass:

| PR | title | head SHA | archive source |
|---:|---|---|---|
| 55 | `quantizr (0.33)` | `e0b643b0a7c21f62cc93b5d920bcf3fc0d5a33d9` | GitHub user attachment |
| 63 | `add qpose14 submission` | `17a3474eb2a7bbd648b850ec6c8338ef8ba15a65` | GitHub user attachment |
| 64 | `Add unified_brotli submission` | `8e5437d923d664d88382a93be8110a25b1966348` | GitHub release asset |
| 65 | `henosis_qz_n3z_r25_clean (0.32)` | `a8b53b5280ee8f05db65740cd48cf7c321a55497` | Git LFS/raw media |
| 67 | `qpose14_r55_segactions_minp (0.31)` | `60297a385cdb0a5250be95e6ba2556585ec80bce` | GitHub release asset |

Public leaderboard refresh still shows rounded `0.32` for PR67, PR65, PR63
and rounded `0.33` for PR64, PR55.

## Archive Anatomy

Generated artifacts:

- `experiments/results/top_submission_delta_reverse_engineering_20260503/archive_anatomy/public_vs_c089_anatomy.json`
- `experiments/results/top_submission_delta_reverse_engineering_20260503/archive_anatomy/public_vs_c089_anatomy.md`
- `experiments/results/top_submission_delta_reverse_engineering_20260503/comparisons/logical_member_delta_vs_c089.csv`
- `experiments/results/top_submission_delta_reverse_engineering_20260503/comparisons/score_byte_gaps.csv`
- `experiments/results/top_submission_delta_reverse_engineering_20260503/comparisons/top5_concrete_deltas.json`

Score and byte table:

| archive | zip bytes | bytes vs C-089 | score/body score | score vs C-089 |
|---|---:|---:|---:|---:|
| PR55 Quantizr | `299970` | `+23628` | `0.3322719955107341` | `+0.016801268` |
| PR63 qpose14 | `287573` | `+11231` | `0.3249617830819938` | `+0.009491056` |
| PR64 unified_brotli | `287165` | `+10823` | `0.33612677067610813` | `+0.020656043` |
| PR65 Henosis | `284425` | `+8083` | `0.3196824276891214` | `+0.004211700` |
| PR67 current release URL | `276620` | `+278` | body components imply `0.31470817503416587` | `-0.000762552` |
| PR67 body-stated cached archive | `276741` | `+399` | body components imply `0.31470817503416587` | `-0.000762552` |
| C-089 | `276342` | `0` | `0.3154707273953505` | `0` |

Important adversarial source finding:

- PR67 body states SHA
  `cc1dfa346811dade15044910fa0143e73a7a049b98aeec39b96a53481a0788bd`
  and `276,741` bytes.
- The current PR67 release URL downloaded SHA
  `86c8694adf8bf53a09a2f2162285601be51ae3030572c73d97f85f3db04c85b8`
  and `276,620` bytes.
- Both are valid public artifacts, but they must not be conflated. The body
  score belongs to the body-stated artifact unless exact eval proves the
  current release asset's components.

## Member Deltas

The mask stream is identical across PR55, PR63, PR65, PR67, and C-089:
compressed SHA `d1ae0d39c848e5715b74bb0122269066a4a1ab60ba9e12f34e70fd62ac136d87`.
That means the remaining public-top delta is not another broad mask-stream
import. It is renderer, pose, action/postfilter, and exact runtime behavior.

Key C-089 encoded slices:

- mask: `219472` bytes, same public mask
- renderer: `55965` bytes, decodes to `59288` bytes, QZS3
- pose: `677` bytes, decodes to `1140` bytes, QP1
- actions: `116` bytes, decodes to `160` bytes, P6 delta-varint actions
- P6 self-describing header cost: `12` bytes

PR67 body-stated cached artifact:

- mask: same `219472`
- renderer: `56034` bytes, `+69` vs C-089
- actions: `236` bytes, `+120` vs C-089
- pose: `899` bytes, `+222` vs C-089
- no self-describing header

PR67 current release URL:

- mask: same `219472`
- renderer: `55914` bytes, `-51` vs C-089
- actions: `236` bytes, `+120` vs C-089
- pose: `898` bytes, `+221` vs C-089
- no self-describing header
- parser bug class found: our fixed-slice parser knew the older `56034`
  model length but not the current `55914` length.

PR65 Henosis:

- mask: same `219472`
- renderer: `57074`, `+1109` vs C-089
- pose stream: `1487`, with `P1D1` decoded pose representation
- charged postfilter streams: `post=1400`, `shift=226`, `frac=106`,
  `frac2=149`, `frac3=154`, `bias=223`, `region=273`, `randmulti=3731`
- body PoseNet is much better (`0.00035283`), but SegNet/rate are worse.
  Useful signal is selected post/pose atoms, not wholesale PR65 transplant.

PR64:

- `unified_brotli` proves single outer Brotli can reduce qpose14-style ZIP
  overhead by about `408` bytes vs PR63, but its public body SegNet is much
  worse. Treat as a packer idea, not a component win.

## Parser Hardening Landed

Patched:

- `submissions/robust_current/unpack_renderer_payload.py`
- `src/tac/tests/test_unpack_renderer_payload_fixedslice.py`

Fix:

- Added current PR67 release fixed-slice model length `55914`.
- Refactored non-self-describing PR75 fixed-slice parsing to try observed model
  lengths by Brotli/QZS3/QP1/content validation instead of trusting one length.
- Added regression coverage synthesizing `mask=219472`, `model=55914`,
  `actions=236`, `pose=677` fixed-slice payload.

Focused verification:

- `.venv/bin/python -m py_compile submissions/robust_current/unpack_renderer_payload.py src/tac/tests/test_unpack_renderer_payload_fixedslice.py`
- `.venv/bin/python -m pytest src/tac/tests/test_unpack_renderer_payload_fixedslice.py -q`
- result: `9 passed in 2.31s`

## Top 5 Concrete Deltas

1. **Recover PR67/PR75 component edge with C-089 bytes.**
   C-089 is already smaller than PR67, but worse in PoseNet and SegNet. The
   PR67 body component edge is about `+0.000352` SegNet contribution and
   `+0.000676` PoseNet contribution relative to C-089, partly offset by C-089's
   rate win. Required work: raw-output parity trace between C-089, current PR67,
   and body-stated PR67; component trace localized by pair/frame; exact T4 on
   any stacked archive.

2. **PR75/P6 action grammar and dictionary-v2 search.**
   Public fixed actions use `236` encoded bytes for `268` decoded action bytes;
   C-089 P6 uses `116` encoded bytes for `160` decoded bytes. Direct rate
   savings are mostly harvested, but action choices are cheap component levers.
   Required gates: decode action records, prove non-noop runtime application,
   compare action deltas to PR67 public records, exact T4 for top variants.

3. **Renderer QZS3/self-compression transplant without pose collapse.**
   The renderer is still the only stream with multi-KB plausible byte savings.
   C-089's encoded renderer is `55965` bytes; current PR67 release has a valid
   `55914` slice, but that is only `51` bytes. Sub-0.314 needs either a real
   learned/block-FP renderer shrink or a component improvement. Required gates:
   trained renderer transplant preflight, pose-safety preflight, runtime tree
   hash custody, exact T4.

4. **Henosis postfilter atom subset as charged typed qpost.**
   PR65's post/shift/fractional-shift/bias/region/random-multi atoms are exactly
   the kind of scorer-targeted pixel corrections that can improve PoseNet
   without touching masks. Wholesale PR65 is not enough, but selected atoms
   against C-089 hard pairs may be. Required gates: opt-in charged qpost member,
   no-op guard, pair/frame component trace, exact T4.

5. **Pose active-subspace transfer from PR65 without SegNet tax.**
   PR65's PoseNet body value is much lower than C-089, while SegNet is worse.
   A bounded QP1/low-dimensional perturbation may be high EV if it preserves
   C-089's mask/renderer/action basin. Required gates: decode pose trajectories,
   active-subspace builder, pose-loader magic preflight, exact T4.

## Immediate Implementation Candidates

Highest EV next local implementation:

1. Build a three-way raw-output/component trace table:
   C-089 vs PR67 current release vs PR67 body-stated cached archive.
   This should produce exact pair/frame deltas for PoseNet and SegNet and feed
   the action/pose/postfilter selectors.
2. Run the existing PR75 action dictionary-v2 worker outputs through the updated
   fixed-slice parser and exact local no-op guards. Dispatch only if the variant
   changes decoded actions and has a plausible component path.
3. Use PR65 qpost streams as a source dictionary, not a wholesale transplant:
   generate tiny top-k hard-pair qpost subsets against C-089, require raw-output
   delta proof, then exact T4 only for the smallest high-confidence candidate.
4. Keep renderer self-compression as the main sub-0.314 byte lever once Modal
   training artifacts return; run transplant and pose-safety preflights before
   any exact eval.
5. Add a release-asset SHA drift guard to public reverse-engineering tooling:
   public archive URL, PR body SHA, downloaded SHA, and parsed split must all be
   recorded before any public artifact is used as an anchor.

