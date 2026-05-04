# writeup working notes

## current state - 2026-05-04

The writeup is now a contest-faithful semantic-bundle story centered on the
verified PR98 HNeRV/Muon adapter replay frontier. The PR95 stem-permutation
repack, older conservative PR95 repack, PR85+STBM, C-067/PR84/QMA9 rows remain
method history and superseded exact evidence, not the headline.

The current exact frontier is **PR98 HNeRV/Muon adapter replay**, a
contest-signature adapter replay of the public PR98 archive. The score-bearing
evidence is
local exact Tesla T4 auth eval through `archive.zip -> inflate.sh ->
upstream/evaluate.py`:

- score: `0.22933111465960354` `[A++]`
- archive bytes: `178392`
- archive SHA-256:
  `7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb`
- SegNet: `0.00068841`
- PoseNet: `0.00017394`
- samples: `600`
- GPU: Tesla T4, `gpu_t4_match=true`
- score authority:
  `experiments/results/lightning_batch/exact_eval_public_pr98_hnerv_adapter_t4_20260504T0958Z/contest_auth_eval.adjudicated.json`
- runtime tree SHA-256:
  `0232154c17410621325ec1647e0f0723b3310d63b0d4bc4bf7bbb5e9aa2fccd0`
- source attribution is required because the archive and runtime originate
  from public PR #98. The local score claim is only for the exact archive
  bytes plus the adapter runtime tree above.

The superseded PR95 stem-permutation repack is the immediate predecessor:

- local exact T4 replay score: `0.23089404465634825` `[A++]`
- archive bytes/SHA-256:
  `178277`,
  `e40c3f2fb3587b12eccb8707e0a1b7831fde149318f3eb212500c674ccbfbf28`
- score authority:
  `experiments/results/lightning_batch/exact_eval_pr95_hnerv_muon_stemperm_t4_20260504T0906Z/contest_auth_eval.adjudicated.json`
- runtime tree SHA-256:
  `a3f8ab2cfbbdfab53a6d437b0c39f525e0adde2d8bd971765de96aeda4da3dc7`

PR98 improves on PR95 stem-permutation by `0.00156292999674471` score points
despite `115` more archive bytes. This is a component tradeoff: SegNet
improves from `0.00070732` to `0.00068841`, while PoseNet moves from
`0.00017185` to `0.00017394`.

The public PR95 exact T4 replay remains a source anchor:

- local exact T4 replay score: `0.23098329465634826` `[A++]`
- archive bytes/SHA-256:
  `178417`,
  `e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a`
- score authority:
  `experiments/results/lightning_batch/exact_eval_public_pr95_hnerv_muon_t4_fix2_20260504T0848Z/contest_auth_eval.adjudicated.json`
- PR95 source attribution is required because the archive and runtime originate
  from public PR #95. The local score claim is only for the exact repacked
  archive bytes and our replay/eval custody.

PR85 itself is the exact public-source anchor:

- local exact T4 replay score: `0.25806611029397786` `[A++]`
- archive bytes/SHA-256:
  `236328`,
  `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- score authority:
  `experiments/results/lightning_batch/exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_20260503T2332Z/contest_auth_eval.adjudicated.json`
- PR85 source attribution is required because the archive and runtime originate
  from public PR #85. The local score claim is only for the exact downloaded
  archive bytes and our replay/eval custody.

PR85+STBM remains a superseded exact lineage: it improved PR85 by `-6572`
charged archive bytes and `-0.004376` score with unchanged PR85 component
distances at JSON precision. It is now method history and source context, not
the current frontier.

## evidence boundary

Use these terms precisely:

- **Exact evidence:** local or canonical CUDA auth-eval JSON on exact archive
  bytes, with archive SHA/bytes, runtime tree hash, sample count, component
  distances, and recomputed score.
- **External PR claim:** public PR title/body/comment/README values before our
  replay succeeds. These can motivate work but cannot rank or promote.
- **Static anatomy:** local parse/profile/preflight of a public archive or
  runtime. Useful for source attribution and risk, not a score.
- **Invalid pre-score failure:** inflate/runtime/entropy failures that prevent
  `contest_auth_eval.json`. These are harness or replay contract evidence, not
  method scores.

The PR91 title/body self-report says `0.24879480490416128` at `222404` bytes.
It remains external only. Our T4/L40S canonical replay attempts failed before
scoring inside HPM1 entropy decode:

- T4 hedge mirror:
  `experiments/results/lightning_batch/exact_eval_public_pr91_hpac_hybrid_t4_g4dn2x_20260504T0509Z`
- archive bytes/SHA-256:
  `222404`,
  `4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f`
- failure: `AssertionError: Tried to decode from compressed data that is invalid for the employed entropy model.`
- terminal class: `inflate_failure_before_score`
- evidence grade: `invalid`; `score_claim=false`

PR91-derived archives are non-dispatchable until full HPM1 decode/reencode
parity or a recovered PR86/PR91 entropy contract exists.

## PR91 source anatomy

PR91 is useful as source anatomy, not current exact evidence.

- Single stored ZIP member: `x`, `222304` bytes.
- Bundle format: PR85/PR91 v5 micro header with fixed `bias=223` and
  `region=273` lengths.
- Payload slices:
  - `mask`: `145087` bytes, magic `HPM1`
  - `model`: `57074` bytes, Brotli `QH0`
  - `pose`: `1487` bytes, Brotli `P1D1`
  - `post`: `1400` bytes
  - `shift`: `226` bytes
  - `frac`: `106` bytes
  - `frac2`: `149` bytes
  - `frac3`: `154` bytes
  - `bias`: `223` bytes
  - `region`: `273` bytes
  - `randmulti`: `16101` bytes
- HPM1 mask details:
  - tokens: `116796` bytes / `29199` uint32 words,
    SHA-256 `541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b`
  - HPAC model: `28243` bytes,
    SHA-256 `de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd`
  - config: `N=600`, `H=384`, `W=512`, `P=32`, `delta=2`,
    `ch=64`, `use_spm=true`, `hpac_d_film=8`, `ppmd_order=4`

`range_mask_codec.cpp` in the PR91 runtime is live only for the
`QMA6`/`QMA7`/`QMA8`/`QMA9` range-mask branch. It is not the HPM1 entropy
decoder and it does not rescue HPM1 decode failures. The submitted dispatch
order is HPM1 first, then QMA6-QMA9, then Brotli OBU. There is no submitted
runtime fallback from failed HPM1 decode to old PR85 masks.

## method spine

The method should be framed as a compiler for charged sufficient statistics:

- public semantic-bundle intake with strict source attribution;
- typed payload parsing for mask, model, pose, post/shift/frac/bias/region, and
  randmulti streams;
- lossless or component-safe recoding only when decoded output parity is proved;
- exact submitted runtime custody before GPU dispatch;
- CUDA/T4 auth eval as the only score promotion path.

The STBM result is a pure rate win at fixed components. The paper should not
overclaim new SegNet/PoseNet learning from it. The learning is in the archive
contract: a recovered lossless semantic-mask representation can lower bytes
without changing evaluator-facing masks.

## compliance and release gate

The PR98 release packet is:

- submission directory:
  `experiments/results/submission_packet_pr98_adapter_20260504/apogee_pr98_hnerv_adapter`
- packet manifest:
  `experiments/results/submission_packet_pr98_adapter_20260504/submission_packet_manifest.json`
- strict pre-submission compliance JSON:
  `experiments/results/submission_packet_pr98_adapter_20260504/pre_submission_compliance.json`

The provider-agnostic compliance gate passed with `78` checks and no failed or
warning checks. Before any public PR body, supplement, or release packet is
changed again, rerun the gate on the exact packet directory with these
score-custody constants:

- auth eval JSON:
  `experiments/results/lightning_batch/exact_eval_public_pr98_hnerv_adapter_t4_20260504T0958Z/contest_auth_eval.adjudicated.json`
- expected archive SHA-256:
  `7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb`
- expected archive size: `178392`
- expected runtime tree SHA-256:
  `0232154c17410621325ec1647e0f0723b3310d63b0d4bc4bf7bbb5e9aa2fccd0`
- expected source PR: `PR98`

The gate does not score. It fails closed on missing required files, bad ZIP
custody, unsafe members, central/local filename mismatch, non-executable
`inflate.sh`, exact-eval/archive/runtime mismatches, missing T4-equivalent
CUDA evidence, public reference parsing errors, and public-surface secret/path
risk. A passed gate is release readiness evidence, not a new score.

## site/report plan

Public surfaces should show:

1. **Frontier summary:** PR98 adapter replay exact A++ as the current
   frontier.
2. **Exact artifact table:** PR98 adapter replay, PR95 stem-permutation repack, PR95 conservative
   repack, PR95 public replay, PR85+STBM/RMB1, PR85, PR84, PR81, and older
   C-067 rows, each with grade, archive bytes/SHA, components, score JSON,
   runtime custody, and source attribution.
3. **External public context:** PR96 unresolved/body score, PR95 body score,
   PR91 self-report, PR86/PR89/PR87/PR70
   classifications, clearly separated from exact evidence.
4. **Failure ledger:** PR91 HPM1 fail-closed, wrong-runtime STBM pre-score
   failures, QRGB exact negatives, and PR87/PR70 payload-closure violations.
5. **Compliance gate:** pre-submission command, required inputs, and pass/fail
   semantics.
6. **Residual gaps:** PR99/PR98 deconstruction, PR91 HPM1 parity, public release
   hygiene scan on the final site bundle, and final upload URL placeholders.

## what not to write

- Do not headline PR95 stem-permutation, PR85+STBM, C-067, PR81, PR84, or PR85 alone as current.
- Do not treat PR98's public body/CPU score as exact evidence; cite only the
  adapter exact T4 adjudicated JSON for score claims.
- Do not treat PR95's public body/CPU score as exact evidence.
- Do not treat PR96's public score as exact evidence until local CUDA replay
  exists.
- Do not claim PR91's `0.24879480490416128` as exact evidence.
- Do not imply `range_mask_codec.cpp` decodes PR91 HPM1.
- Do not treat PR87/PR70 source-embedded payload submissions as floor evidence.
- Do not expose private provider URLs, raw account/job metadata, local absolute
  paths, or unsanitized state files in public docs.
