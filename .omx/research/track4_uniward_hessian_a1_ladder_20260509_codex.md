# Track 4 UNIWARD/Hessian A1 Ladder - 2026-05-09

## Scope

Sequential score-lowering work on the A1 latent-aligned PR101 archive:

- source archive:
  `experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal/harvested_artifacts/finetuned_archive/archive.zip`
- source archive bytes: `178262`
- source archive SHA-256:
  `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`
- source CPU anchor: `0.19284757743677347` `[contest-CPU]`
- source CUDA anchor: `0.2263520234784395` `[contest-CUDA]`

This ledger records byte-candidate construction and macOS CPU advisory screens
only. No score promotion is allowed from this file.

## Code Fixes Landed Before Candidate Use

`tools/build_uniward_stc_hessian_a1_v1.py` was untracked WIP when found. Review
found a score-affecting no-op bug: the draft passed 28 per-tensor Fisher values
to `fisher_information_bit_allocation`, which is a per-parameter allocator, then
clamped every tensor back to 8 bits. That made the default path effectively
non-load-bearing.

Fixes made:

- parameter-weighted per-tensor bit allocation;
- compressed-byte calibration: source decoder compressed size maps to the
  current 8-bit tensor grid, and requested byte cuts scale down from there;
- complete runtime packet output by copying the A1 `submission_dir` beside
  `archive.zip`;
- manifest fields for source/new inner SHA-256, decoder blob bytes/SHA,
  `score_affecting_payload_changed`, codec roundtrip error, source distortion,
  and runtime packet completeness;
- exact-eval wrapper fix: `experiments/contest_auth_eval.py` now defaults
  `PYTHON` to `sys.executable` for inflate subprocesses so PR101/A1-style
  `${PYTHON:-python3}` runtimes use the repo venv during local exact-eval
  screens.

Focused tests:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_build_uniward_stc_hessian_a1_v1.py \
  src/tac/tests/test_contest_auth_eval.py -q
```

Result: `37 passed`.

## Candidate Ladder

All rows are runtime-packet complete and have `score_affecting_payload_changed=true`.
Evidence grade is byte-anchor plus local packet smoke until exact eval lands.

| target bytes | archive bytes | delta vs A1 | global L1 rel err vs A1 | max rel err vs A1 | archive SHA-256 |
|---:|---:|---:|---:|---:|---|
| `176000` | `177668` | `-594` | `0.0022816252261110117` | `0.007874015748031496` | `c0896bee805a1130840e88839c97f35b01396fb11783eda46428ecf5b350f439` |
| `174000` | `177624` | `-638` | `0.004878680875800402` | `0.007874015748031496` | `0335bd14eb8522dcfe48f858505cdaf62c74c42b0ca405baafd3a0143f187e8a` |
| `172000` | `177760` | `-502` | `0.00833529153923029` | `0.007874015748031496` | `db9bd259392f5acce046722208ce81f511d19c68003ecb1eb8263abc9271dd62` |
| `170000` | `177186` | `-1076` | `0.010932347188919682` | `0.007874015748031496` | `7bbe84261c1ee26b9d07fc81f9f4147db35c60dac326f50cb6dd4263c98cba87` |
| `168000` | `174937` | `-3325` | `0.011892827137073321` | `0.01600203674854451` | `33ce715133274324d286e9003334eed46b16f258a681d5a1ba75d19487726690` |
| `166000` | `173017` | `-5245` | `0.013348424729292017` | `0.01600203932649563` | `8e493325de9110693f54e7ba067fc77c2acde446c9fd0d6a5f737789da94b989` |
| `164000` | `171109` | `-7153` | `0.014320841246860767` | `0.016002037854237854` | `5ad48bb9b03ee532677f6a6e000507548c5b59ad67892bd590bfcfa984d23d10` |
| `162000` | `167322` | `-10940` | `0.01630732827411296` | `0.01600203932649563` | `9d756c048fcbb37ca3b1cf7a1566ec1bf02876ed3ece3522a91d298823bf9054` |
| `160000` | `164890` | `-13372` | `0.017279744791681713` | `0.016002037854237854` | `3abf2ac8239ff99cd447165a9066e6aaf527e1a9f1ddb775340206c77a9cd98f` |

## macOS CPU Advisory Screen - target164000

Command:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/track4_uniward_stc_hessian_a1_target164000_20260509_codex/archive.zip \
  --inflate-sh experiments/results/track4_uniward_stc_hessian_a1_target164000_20260509_codex/submission_dir/inflate.sh \
  --upstream-dir upstream \
  --device cpu \
  --work-dir experiments/results/track4_uniward_stc_hessian_a1_target164000_20260509_codex/macos_cpu_eval_work_retry1 \
  --json-out experiments/results/track4_uniward_stc_hessian_a1_target164000_20260509_codex/contest_auth_eval_macos_cpu.json \
  --keep-work-dir
```

Result:

- evidence grade: `[macOS-CPU advisory]`
- canonical score: `0.2605155145868584`
- archive bytes: `171109`
- pose distance: `0.00007057`
- seg distance: `0.00120016`
- rate contribution: `0.1139345`
- inflate elapsed: `36.85752304107882 s`
- evaluate elapsed: `410.396697417018 s`
- runtime tree SHA-256:
  `b7d33a4138e782113e72a2568717667613b7ee7814a4a036f2502ace1769203c`

Classification: measured-config regression. The `-7153 B` rate gain is real,
but scorer distortion dominates: SegNet roughly doubled against the A1 CPU
anchor (`0.00056023 -> 0.00120016`) and pose increased (`0.00003286 ->
0.00007057`). This retires only the `target164000` configuration, not the
Track 4 family.

Reactivation criteria:

- screen smaller-distortion ladder points (`174000`, `176000`) to find the
  trust-region edge;
- replace the magnitude/Fisher proxy with score-domain sensitivity from A1
  gradients or component-response maps;
- prefer bit cuts on tensors whose perturbations are proven scorer-cheap, not
  merely small in weight-domain saliency;
- require paired `[contest-CPU]` and `[contest-CUDA]` before any score
  promotion.

## macOS CPU Advisory Screen - target174000

Command:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/track4_uniward_stc_hessian_a1_target174000_20260509_codex/archive.zip \
  --inflate-sh experiments/results/track4_uniward_stc_hessian_a1_target174000_20260509_codex/submission_dir/inflate.sh \
  --upstream-dir upstream \
  --device cpu \
  --work-dir experiments/results/track4_uniward_stc_hessian_a1_target174000_20260509_codex/macos_cpu_eval_work \
  --json-out experiments/results/track4_uniward_stc_hessian_a1_target174000_20260509_codex/contest_auth_eval_macos_cpu.json \
  --keep-work-dir
```

Result:

- evidence grade: `[macOS-CPU advisory]`
- canonical score: `0.22729521075217057`
- archive bytes: `177624`
- pose distance: `0.00005131`
- seg distance: `0.00086371`
- rate contribution: `0.11827249999999999`
- inflate elapsed: `35.894286582944915 s`
- evaluate elapsed: `411.10497741610743 s`
- runtime tree SHA-256:
  `50de6adadf1ec973882473894f042f385684f62846d81468c158e0b26afd7a2e`

Classification: measured-config regression. This is more important than the
larger `target164000` miss because it shows the naive weight-magnitude proxy is
outside the scorer trust region even at `638 B` saved and `0.00487868` global
L1 weight distortion. The rate gain is only about `-0.0004248` score points,
while SegNet and pose movement cost roughly `+0.0349` versus the A1 CPU anchor.

Implication: do not remote-dispatch the Track 4 magnitude/Hessian ladder as a
score candidate unless the `target176000` edge screen lands cleanly. The next
real score-lowering allocator should consume score-domain sensitivity or
component-response maps.

## macOS CPU Advisory Screen - target176000

Command:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/track4_uniward_stc_hessian_a1_target176000_20260509_codex/archive.zip \
  --inflate-sh experiments/results/track4_uniward_stc_hessian_a1_target176000_20260509_codex/submission_dir/inflate.sh \
  --upstream-dir upstream \
  --device cpu \
  --work-dir experiments/results/track4_uniward_stc_hessian_a1_target176000_20260509_codex/macos_cpu_eval_work \
  --json-out experiments/results/track4_uniward_stc_hessian_a1_target176000_20260509_codex/contest_auth_eval_macos_cpu.json \
  --keep-work-dir
```

Result:

- evidence grade: `[macOS-CPU advisory]`
- canonical score: `0.21300500575810702`
- archive bytes: `177668`
- pose distance: `0.00004233`
- seg distance: `0.00074129`
- rate contribution: `0.11830175`
- inflate elapsed: `36.65886112500448 s`
- evaluate elapsed: `411.0869190840749 s`
- runtime tree SHA-256:
  `2a266ba4f2c1f3c65322a0a5f0d3a1e73bb69f29466fd23e2edb60eb786e18cd`

Classification: measured-config regression. This is the trust-region edge
screen and it still loses: `-594 B` improves the rate term by only about
`0.0003955`, while SegNet rises from the A1 CPU anchor's `0.00056023` to
`0.00074129` and pose rises from `0.00003286` to `0.00004233`. Net score
movement is approximately `+0.02016` versus the A1 Linux CPU anchor.

Decision: do not dispatch this Track 4 magnitude/Hessian ladder remotely as a
score-lowering candidate. The implementation remains useful as a byte-closed
packet compiler and negative control, but the current no-data weight-domain
saliency proxy is score-misaligned even at the smallest measured byte cut.

Reactivation criteria:

- replace `mean(theta^2)` Fisher proxy with real score-gradient or component
  response maps from PoseNet/SegNet;
- include no-op/one-tensor perturbation controls that localize which tensors
  drive the SegNet jump;
- retry only with a smaller exact changed-payload step than `594 B` or with a
  certified score-domain sensitivity map;
- require paired `[contest-CPU]` and `[contest-CUDA]` before any score
  promotion.

## In Flight

`track1_phase_a1_score_gradient_long_lr1e6_20260509T030424Z_modal` was
dispatched to Modal T4 for a gentler/longer A1 score-gradient schedule.

- call id: `fc-01KR5B4JKGXQWZ4452G7JJ23J1`
- params: `epochs=120`, `steps_per_epoch=8`, `lr=1e-6`,
  `aux_kl_weight=0.2`, `aux_pixel_l1_weight=0.01`, `batch_size=4`
- estimated cost: `$2.36`
- predicted ETA: `2026-05-09T07:05:15Z`
- claim: active `track1_phase_a1_score_gradient`

Harvest command:

```bash
.venv/bin/python experiments/modal_phase_a1_score_gradient_pr101.py recover \
  --label track1_phase_a1_score_gradient_long_lr1e6_20260509T030424Z_modal
```

## Custody / Preflight

`experiments/results/` runtime packet source remains raw rebuildable custody;
the durable signal is this ledger plus the committed builder/test/report
surface. After the Track 4 ladder and local advisory evals generated new
runtime packet sources under `experiments/results/`, the strict untracked-source
baseline in `.omx/research/untracked_source_dispositions_20260505_codex.json`
was intentionally rebaselined to:

- algorithm: `pact_runtime_source_set_v1`
- count: `10573`
- sha256:
  `8ad89e999311c5e24bf310a92d53f3853097256099ffb2eef78594e94ae4f6d5`

No raw Track 4 runtime packet, archive, macOS work directory, or generated
result JSON is promoted to git.
