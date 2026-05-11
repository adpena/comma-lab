# Public PR95+ mechanism map and non-HNeRV escape routes (2026-05-11)

## Status

This is a mechanism ledger, not a score claim.

- `score_claim=false`
- `dispatch_attempted=false`
- goal: preserve signal from public PR reports/writeups/source and route it
  into exact score-lowering work without HNeRV tunnel vision

## Core correction

The public leaderboard rows and our internal CUDA rows are not one scalar
series. They are a device/runtime/scorer matrix. Do not compare a public
CPU-row report to a Modal T4 CUDA replay without the axis label.

Evidence already in hand:

- PR100 public body reports `device: cpu`, score `0.1954`, bytes `178981`,
  pose `0.00003443`, seg `0.00057654`.
  Source:
  `experiments/results/public_pr_archive_kaggle_mirror/public_pr100_intake_20260505_auto/pr_body.md`
- PR101 public body reports CPU-style score `0.19`, bytes `178258`,
  pose `0.00003286`, seg `0.00056018`.
  Source:
  `experiments/results/public_pr_archive_kaggle_mirror/public_pr101_intake_20260505_auto/pr_body.md`
- PR103 public body reports local CPU `0.19487`, bytes `178223`,
  pose `0.00003443`, seg `0.00057638`.
  Source:
  `experiments/results/public_pr_archive_kaggle_mirror/public_pr103_intake_20260505_auto/pr_body.md`
- PR106 public body reports `device: cuda`, score `0.20946`, bytes `186239`,
  pose `0.00003351`, seg `0.00067142`.
  Source:
  `experiments/results/public_pr_archive_kaggle_mirror/public_pr106_intake_20260505_auto/pr_body.md`

Our exact/device-axis evidence shows the sign is packet-dependent:

- PR106 latent sidecar T4 CUDA exact:
  `experiments/results/modal_auth_eval/pr106_latent_sidecar_20260511T150517Z/contest_auth_eval.adjudicated.json`
  - score `0.20739428085403283`
  - pose `0.00003281`
  - seg `0.00064893`
  - bytes `186808`
- same PR106 latent sidecar Linux CPU diagnostic:
  `experiments/results/modal_auth_eval_cpu/pr106_latent_sidecar_20260511T151955Z/contest_auth_eval.json`
  - score `0.2286802845175232`
  - pose `0.00016424`
  - seg `0.00063766`
  - bytes `186808`
- PR103-on-PR106 paired mechanism analysis:
  `experiments/results/dual_device_auth_eval/pr103_pr106_dual_runtime_mechanism_analysis_20260511T030251Z/analysis.md`
  - CPU score `0.229657686265`
  - CUDA score `0.208983075582`
  - CUDA wins by `0.02067461068280979` on that exact pair

Interpretation: there is no universal CPU-better or CUDA-better rule. The
device effect is a function of raw output, scorer path, inflate path, and
threshold geometry. Treat it as a candidate-specific response surface.

## What PR95+ did that generalizes

### Training discipline

PR95/106 mechanism:

- full RGB renderer, not mask slot
- archive grammar in the loop
- eval-roundtrip and differentiable YUV6 discipline
- staged curriculum:
  CE -> tau-Softplus -> smooth-disagreement -> QAT -> L7/C1a ->
  lambda sweep -> sigma sweep -> Muon fine-tune

Primary evidence:

- `experiments/results/public_pr95_intake_20260504_codex/profile_pr95_hnerv_muon_intake.md`
- `experiments/results/comma_lab_public_export/docs/pr106_vs_pr101_training_recipe_finding_20260504.md`

General principle:

- Train the substrate against the actual scorer geometry and the actual
  export/inflate/eval roundtrip, then apply byte-level coders and sidecars.

Immediate engineering consequence:

- T1/Ballé/HNeRV parity work should not dispatch another renderer unless the
  training loop has eval-roundtrip, YUV6 reachability, score-domain loss, and
  archive export in the same loop.

### Inference-time arithmetic and section grammar

PR103 mechanism:

- lossless byte-level repack of PR100/PR95-family HNeRV payload
- arithmetic/range coding on largest weight tensors and latent-hi stream
- hardcoded section lengths, single-byte ZIP filename, merged AC streams

Primary evidence:

- `experiments/results/public_pr_archive_kaggle_mirror/public_pr103_intake_20260505_auto/pr_body.md`
- `experiments/results/hnerv_pr103_lc_ac_schema_refresh_20260510_codex/manifest.md`

General principle:

- build a deterministic packet compiler that can deconstruct sections, choose
  per-section entropy coders, and prove identity/optimize rewrites.

Immediate engineering consequence:

- PR103 AC should be a reusable `tac` packet-compiler pass, not a one-off
  HNeRV script.

### Per-pair sidecar search

Current best exact CUDA path:

- PR106 latent sidecar exact T4: `0.20739428085403283`
- PR106 latent radius-2 Kaggle P100 diagnostic: `0.2066238854574151`
- local materialized radius-2 archive:
  `experiments/results/pr106_latent_sidecar_r2_from_kaggle_table_20260511_codex/sidecar_archive.zip`
  - bytes `186822`
  - sha256 `7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f`

General principle:

- sidecars are not arbitrary constants; they are a local discrete optimizer
  over a scorer response surface. The right API is score-table production ->
  deterministic materialization -> exact adjudication.

Immediate engineering consequence:

- keep expanding score-table producers, but never promote Kaggle/P100/MPS
  directly. Promote only materialized archives after exact T4/CUDA custody.

Supersession note (`2026-05-11T16:40Z`): the PR106 latent radius-2 table later
was harvested, materialized, and exact-T4 adjudicated at
`0.20664588545741508` with archive SHA-256
`7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f`.
Any "pending R2 exact adjudication" wording in this memo is historical context.
The result is still a measured frontier / constructive upper bound, not a
certified lower-bound floor.

## Non-HNeRV escape routes

These are not alternatives to rigor; they are alternative basis families that
must enter the same archive/export/scorer loop.

| Family | Existing signal | What to do next |
| --- | --- | --- |
| RAFT / ego-motion / telescopic foveation | PR95 intake explicitly recommends replacing part of `600 x 28` free latents with camera-aware bases. | Build a latent-basis sidecar score table: basis coefficients over PR106 decoded residuals, not a new monolithic renderer. |
| Cool-Chic / C3 / wavelets | PR95 intake calls them residual bases missing from current HNeRV archives. | Attach tiny residual programs to HNeRV/PR106 outputs; require byte-closed decoder and no-op proof before eval. |
| Ballé / CompressAI / hyperprior | Phase 1/T1 is pending and should be trained end-to-end, not bolted onto frozen bytes. | Fix runtime/export loop first; dispatch only if eval-roundtrip and score-domain losses are in-loop. |
| Arithmetic/range/ANS/bit packing | PR103 proves byte-level coding still matters at medal band. | Promote into the deterministic submission-packet compiler with cross-language vectors. |
| HPAC / categorical / mask-action codecs | Public PR91 and internal PR85/PR86/PR79 lanes show non-HNeRV grammar families exist. | Re-intake as typed streams in the compiler, then exact replay or fail-closed blocker. |
| Sparse pixel/class atoms | PR95 intake identifies post-inflate sparse disagreements as a residual opportunity. | Use component traces to rank atoms by byte-normalized score benefit; dispatch only after charged-byte proof. |

## Required next experiments

1. Historical pre-supersession item: complete the pending Modal T4 exact
   adjudication for the PR106 latent radius-2 materialized archive. This has
   since completed at `0.20664588545741508`; preserve the line as context for
   why R2 became the measured frontier.
2. For every PR100/101/103/106-family candidate, run a four-cell axis matrix:
   CPU scorer + CPU inflate, CPU scorer + CUDA inflate, CUDA scorer + CPU
   inflate, CUDA scorer + CUDA inflate. Save raw-output aggregate SHA for each.
3. Build a shared input-layer xray for SegNet/PoseNet: same raw output, same
   ground-truth frames, CPU vs CUDA layer drift, final component drift.
4. Promote PR103 arithmetic coding into the packet compiler with identity and
   optimize modes.
5. Build a PR106 residual-basis score table queue for wavelet/foveation/RAFT
   coefficients using the same materialize-then-exact-adjudicate discipline as
   latent sidecars.

## Current grand-council verdict

The thing to beat is not "HNeRV." The thing to beat is the full compiler:

`score-aware substrate training -> export-first archive grammar -> byte-level
entropy compiler -> scorer-axis sidecar optimizer -> exact dual-axis xray`.

HNeRV is only the current best substrate. The score floor moves when we combine
its useful parts with non-HNeRV residual bases and an exact packet compiler,
while treating CPU/CUDA as a measured axis rather than a belief.

## Supersession note - 2026-05-11 PR106 R2 exact T4

The "Current best exact CUDA path" and "Required next experiments" sections
above are stale where they list PR106 latent sidecar `0.20739428085403283` as
best or describe radius-2 exact T4 adjudication as pending. That adjudication
has landed and is now the authoritative internal PR106-family
`[contest-CUDA]` anchor:

- packet: PR106 latent radius-2 sidecar
- canonical score: `0.20664588545741508`
- archive bytes: `186822`
- archive SHA-256:
  `7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f`
- job id: `pr106_latent_sidecar_r2_20260511T160358Z`
- result JSON:
  `experiments/results/modal_auth_eval/pr106_latent_sidecar_r2_20260511T160358Z/contest_auth_eval.json`

The Kaggle P100 `0.2066238854574151` row remains diagnostic only. Future public
frontier/mechanism language should compare exact CUDA candidates against the
R2 T4 score above and keep CPU, Kaggle/P100, and T4 evidence axes separate.
