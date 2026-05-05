# 2026-05-01 Shannon-Floor Frontier Reproduction And Lane Triage

Use this as durable operational memory, not as the primary result ledger. The
source of truth remains the repo docs under `.omx/research/`, especially
`shannon_floor_nextwave_telemetry_and_research_20260430_codex.md`.

## Current Frontier-Critical Threads

- Best current exact lower-score OWV3/Fisher signal is the stack archive:
  `experiments/results/lane_g_v3_owv3_fisher_beta_20260501_LANDED/lane_g_v3_owv3_fisher_stack_results/archive_lane_g_v3_owv3.zip`.
- Archive SHA-256:
  `57abe0fdf786d95b38325334b568e7a947143afe097ba189f214f2208492cb8f`,
  bytes `638165`.
- Local exact CUDA score was `1.0160176664836693` on RTX 4090 with full
  600-sample custody.
- Lightning T4 reproduction job
  `exact_eval_owv3_fisher_beta_t4_20260501T1025Z` completed and was harvested.
  Exact T4 score `1.0093151488173409`, bytes `638165`, PoseNet `0.00331601`,
  SegNet `0.00402288`, `600` samples, Tesla T4.
- Adjudication grade is A++ contest T4 evidence, but promotion is blocked by
  a tiny SegNet component-gate violation: relative `1.0040733197556009` versus
  configured `1.004`. Status is `COMPONENT_GATE_REVIEW_REQUIRED`, so do not
  claim a promotable frontier anchor without review or a repaired archive.
- Modal independent OWV3 eval failed inside DALI/NVML after inflate and before
  `contest_auth_eval.json`; this is infrastructure failure, not score evidence.

## R7 Custody

- Lightning T4 job `exact_eval_r7_p2_canonical_t4_20260501T1020Z` completed
  and harvested with A++ T4 custody.
- Result: score `1.036860107631869`, archive SHA-256
  `c00393dda0736edb2b2a25e3108624b7571042c63e6c5b138b5168c1a28f1193`,
  archive bytes `686636`, PoseNet `0.00316055`, SegNet `0.00401878`,
  `600` samples, Tesla T4.
- This is a useful custody improvement but not the strategic large-move path.

## Alpha-Geo-0 State

- Vast Alpha-Geo-0 attempts failed due NVDEC/SCP infrastructure. Classification:
  run abort, no conclusion about NeRV or Alpha.
- Active replacement path is now Lightning Batch via
  `scripts/launch_lightning_alpha_geo0_pose_regen.py` and reusable runner
  `experiments/alpha_geo0_pose_regen.py`. Modal and Vast both produced
  infrastructure aborts.
- The experiment keeps exact Lane 12 `masks.nrv`, regenerates poses, rebuilds
  deterministic archive, then runs canonical CUDA auth eval and adjudication.
- The first Modal dispatch used an unpinned DALI image and failed closed at
  CUDA/DALI/NVDEC preflight with `nvidia_dali_version=2.1.0` and NVML error
  999. Diagnostic only.
- Modal exact-eval/training wrappers were pinned to
  `nvidia-dali-cuda120==1.52.0` with the NVIDIA index, but the pinned
  Alpha-Geo-0 call `fc-01KQHHFXYWRY8184K37992EPXZ` still failed closed at
  CUDA/DALI/NVDEC preflight with NVML error 999. This is a Modal
  infrastructure abort, not lane evidence.
- Lightning Alpha job queued:
  `alpha_geo0_pose_regen_lightning_t4_20260501T103952Z`, SDK job
  `alpha-geo0-pose-regen-lightning-t4-20260501t103952z`, state file
  `.omx/state/lightning_batch_jobs.json`, status `Pending` at
  `2026-05-01T10:40:53Z`, concrete machine `g4dn.xlarge`/reported `T4_SMALL`.
  Source manifest:
  `.omx/state/alpha_geo0_pose_regen_lightning_t4_20260501T103952Z_manifest.json`.
- If PoseNet recovers, stale poses were the main Lane 12 failure. If it does
  not recover, treat the measured `jsonfix40` mask geometry as incompatible
  and redesign Alpha around scorer-preserving mask payloads/residuals.

## Vast Cleanup And Protocol

- Vast live inventory was reconciled and cleaned to `live_count=0`.
- Harvested forensic artifacts under `experiments/results/vast_live_harvest/`.
- Lane 19 snapshot archive had only `renderer.bin`; invalid until payload
  closure is fixed.
- MAE-V duplicate concurrent trainers wrote the same output dir; custody
  invalid. Durable rule added to `AGENTS.md`: Vast/output lanes must be
  single-flight per lane label and artifact directory, with duplicate active
  process detection failing closed.

## Operating Priority

- Do not spend more wall clock on millipoint-only work unless it closes exact
  custody for an existing frontier candidate.
- Highest expected-value large moves remain Alpha mask-payload replacement,
  OWV3/Fisher T4 reproduction and stack anchoring, and hidden-gem recovery
  only after exact archive closure.

## 2026-05-01 Exact Frontier Update

- New best exact T4 rankable candidate:
  `experiments/results/frontier_candidate_direct_fd_m2_pfp16_20260501/`.
- Score `1.035635453539817`, archive bytes `686632`, archive SHA-256
  `d561b8bf1367619e6c6e1d9b9d09213da6bdfdfd894be518866ef5119cba2927`,
  PoseNet `0.00311743`, SegNet `0.00401873`, `600` samples, Tesla T4.
- Adjudication status: `A++ contest T4`, `promotion_eligible=true`,
  `IN_PREDICTED_BAND`, component gates passed against same-run PFP16 T4
  reference.
- Duplicate T4_SMALL job reproduced the same archive SHA with score
  `1.0356379048481845`, also passing gates.
- Important protocol distinction: the parent direct-FD response-only jobs are
  non-promotable calibration packets, but individual point archives inside
  them can become rankable when exact T4 eval JSON, archive custody, provenance,
  and independent adjudication are present.

## Alpha-Geo-0 / FFmpeg Resolver Update

- First Lightning Alpha-Geo-0 job failed before evidence because remote
  `upstream/ffmpeg-new` lacked `libSvtAv1Enc.so.2`.
- `src/tac/mask_codec.py` now validates candidate ffmpeg binaries with
  `-version`, honors command-name overrides like `TAC_FFMPEG=ffmpeg`, skips
  broken upstream binaries, and fails closed on explicit bad overrides.
- Replacement job:
  `alpha_geo0_pose_regen_lightning_t4_ffmpegfix_20260501T110126Z`; status
  `Pending` at last refresh, zero cost.

## Backend Cleanup Update

- Vast instance `35955469` / `owv3_0134_eval_repro` was harvested and
  destroyed; live Vast count is now `0`.
- Best harvested Vast OWV3 score was `1.0134396099014253` on RTX 4090, A-grade
  CUDA only, not A++ T4.

## 2026-05-01T11:20Z Continuation State

- Running Lightning jobs:
  `alpha_geo0_pose_regen_lightning_t4_ffmpegfix_20260501T110126Z`
  (`Running`, T4_SMALL, cost `0.026758334` at `2026-05-01T11:18:13Z`) and
  `exact_eval_owv3_5c110_r7_t4_20260501T1112Z` (`Running`, T4_SMALL at
  `2026-05-01T11:18:27Z`).
- Pending Lightning job:
  `exact_eval_direct_fd_m2_frontier_t4_20260501T1110Z`, standalone T4 repro of
  archive SHA-256
  `d561b8bf1367619e6c6e1d9b9d09213da6bdfdfd894be518866ef5119cba2927`.
- New Vast live instance discovered by reconciliation:
  `35956905` / `owv3_wave3_chain_eval`, RTX 4090, `ssh3.vast.ai:36904`,
  initially `loading`, then `running` by `2026-05-01T11:20:27Z`.
- Vast inspection showed repo and Wave3 archives staged but no eval process
  running. Patched the local and remote Wave3 summary-line formatting bug and
  attempted restart. It exposed missing `uv`, then old Ubuntu ffmpeg `4.4.2`
  missing the `scale` color-contract options required by `inflate.sh`.
- Installed remote `uv 0.11.8` and BtbN master ffmpeg
  `N-124278-gcc3ca17127-20260430`, tarball SHA-256
  `e75caec4d65d9baa84c063e54746d2e08f1bdcd719b187967f34330f7c1486fb`.
  Restarted exactly one Wave3 chain at `2026-05-01T11:27:18Z` with
  `FFMPEG_BIN=/workspace/ffmpeg-btbn/bin/ffmpeg`, remote PID `4494`.
  At `2026-05-01T11:27:37Z`, candidate `owv3_0043` had reached
  `contest_auth_eval.py` and `uv run` inflate. Treat future Wave3 outputs as
  advisory until canonical JSON is harvested and locally adjudicated.
- Alpha local screen now exists at
  `experiments/alpha_frontier_candidate_screen.py`; it is empirical only and
  cannot promote. It was hardened to validate ffmpeg in the Alpha4 path too.
  Tests passed: `src/tac/tests/test_alpha_frontier_candidate_screen.py -q`
  (`7 passed`).
- Capped screen artifact:
  `experiments/results/alpha_frontier_candidate_screen_pfp16_20260501/report_max16.json`.
  First-16-frame non-score bytes: Alpha2 `325906` with agreement `1.0`,
  Alpha3 `54819` with agreement `0.999315579732`, Alpha4 `11989` with
  agreement `0.998299280802`. These are large Alpha byte-move leads, not score
  evidence; repair scorer geometry before exact eval.
- Alpha primitive diagnostics now exist at
  `experiments/alpha_primitive_mask_diagnostics.py`; tests at
  `src/tac/tests/test_alpha_primitive_mask_diagnostics.py` passed locally
  after Codex tightened the CLI default to a bounded `64` frames and added
  `--all-frames` for explicit full-corpus analysis.
- Capped primitive artifact:
  `experiments/results/alpha_primitive_mask_diagnostics_pfp16_20260501/report_max8.json`.
  It is non-promotable empirical geometry signal. First-8-frame summary:
  class-1 total components `727`, max class-1 components in one frame `116`,
  total temporal changed pixels `10598`, mean temporal changed fraction
  `0.007700602214`.

## 2026-05-01T11:45Z Continuation Memory

- Lightning exact eval `exact_eval_direct_fd_m2_frontier_t4_20260501T1110Z`
  was harvested and reconciled to Completed. Canonical local artifact dir:
  `experiments/results/lightning_batch/exact_eval_direct_fd_m2_frontier_t4_20260501T1110Z/`.
  Score `1.0356355862798443`, PoseNet `0.00311747`, SegNet `0.00401872`,
  archive bytes `686632`, SHA-256
  `d561b8bf1367619e6c6e1d9b9d09213da6bdfdfd894be518866ef5119cba2927`,
  600 samples, CUDA, Tesla T4, component gates pass, A++/promotion eligible.
- Lightning exact eval `exact_eval_owv3_5c110_r7_t4_20260501T1112Z` was
  harvested and reconciled to Completed. Artifact dir:
  `experiments/results/lightning_batch/exact_eval_owv3_5c110_r7_t4_20260501T1112Z/`.
  Score `1.0077865870356524`, PoseNet `0.00340903`, SegNet `0.00402679`,
  archive bytes `631473`, SHA-256
  `5c11013539755c6470fb9f55e4d7f2ab6ec1edb2b951a468513d4ed7550f66ef`,
  600 samples, CUDA, Tesla T4. Strict SegNet component gate fails
  (`1.0050492192803802` > `1.004`), so this is exact forensic/no-rank
  evidence until repaired.
- Added Alpha mask candidate builder. Full output:
  `experiments/results/alpha_mask_candidate_builder_pfp16_20260501_full/`.
  It proves full repair can restore class-id agreement to `1.0`, but emitted
  bytes are too large: `grayscale.mkv` `859664` bytes and
  `alpha4_residual_repair.amr1` `3657451` bytes versus original `masks.mkv`
  `421483` bytes. Do not spend exact eval on that artifact as-is; next Alpha
  work must compress/select residuals or change representation.
- Durable DX hardening landed: `submissions/robust_current/inflate.sh` now
  uses deterministic `INFLATE_BROTLI_SPEC`, `INFLATE_AV_SPEC`,
  `INFLATE_TORCH_SPEC`, and `INFLATE_NUMPY_SPEC` rather than floating uv
  resolver inputs. `scripts/remote_archive_only_eval.sh` selects
  `INFLATE_TORCH_SPEC=torch==2.5.1` on older Vast drivers such as 550.120,
  otherwise defaults to `torch==2.11.0`.
- Vast Wave3 v2 instance `35957332` at `ssh9.vast.ai:37332` was re-staged
  with patched `inflate.sh`, `remote_archive_only_eval.sh`, and driver.
  Restarted at `2026-05-01T11:44:46Z`; first candidate logs confirm
  `INFLATE_TORCH_SPEC=torch==2.5.1`, CUDA available, NVDEC OK, BtbN ffmpeg
  parity. No canonical Wave3 JSON harvested yet.
- Alpha-Geo-0 Lightning job
  `alpha_geo0_pose_regen_lightning_t4_ffmpegfix_20260501T110126Z` remains
  Running as of `2026-05-01T11:39:19Z`, cost `0.093258336`; no scored
  artifact harvested yet.
