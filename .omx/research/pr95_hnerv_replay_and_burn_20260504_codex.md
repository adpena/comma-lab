# PR95 HNeRV replay and owned burn ledger - 2026-05-04

This is a dated custody/progress ledger, not a durable protocol. Use `AGENTS.md`
for non-negotiable operating rules.

## Current A++ anchor

- Archive: PR85+STBM1BR+PR92/RMB1 RandMulti stack
- Evidence: `A++` exact T4 CUDA, 600 samples
- Artifact:
  `experiments/results/lightning_batch/exact_eval_pr85_stbm1br_pr92_rmb1_t4_20260504T082220Z/contest_auth_eval.adjudicated.json`
- Score: `0.2535063602939779`
- Bytes: `229480`
- Archive SHA-256:
  `f8d2dff12004fe15bdedefcd3f9574fab97f22c302fa1417a265c325468ad774`
- Component distances: PoseNet `0.0001894`, SegNet `0.00057185`

## PR95 public HNeRV/Muon intake

- Source PR claim: `0.1987048012202245`
- Source archive:
  `experiments/results/public_pr95_intake_20260504_codex/archive.zip`
- Source archive bytes: `178417`
- Source archive SHA-256:
  `e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a`
- Source member: `0.bin`
- Source member bytes: `178309`
- Source member SHA-256:
  `4b8013fb1168e21b12fc7ee6c395e032660d88daded48a0b845085e7d5eb11c4`
- Static runtime checks: public `inflate.sh` shell syntax and PR95 Python
  modules compile locally.

Static PR95 anatomy:

- meta brotli payload: `80` bytes
- decoder brotli payload: `162349` bytes
- latents brotli payload: `15868` bytes
- meta shape: base channels `36`, eval size `[384,512]`, latent dim `28`,
  `600` pairs

## Reversible PR95 repack

Tool:
`experiments/profile_pr95_hnerv_muon_packing.py`

Profile artifact:
`experiments/results/pr95_hnerv_muon_packing_profile_20260504_codex/profile_pr95_hnerv_muon_packing.json`

Candidate archive:
`experiments/results/pr95_hnerv_muon_packing_profile_20260504_codex/archive.pr95_repacked.zip`

- Candidate archive bytes: `178321`
- Candidate archive SHA-256:
  `2b9b471358f5bba97ea809dcca544ecca26b504fde770a9002046830f469368b`
- Candidate member bytes: `178213`
- Candidate member SHA-256:
  `d9fa160d366d0aed105d14458e289f88ec5b71ec4d5de318ddb2ec0d44b50bf5`
- Archive delta vs public PR95: `-96` bytes
- Score claim: none until exact CUDA. If decoded output is identical and PR95
  score validates, the rate-only predicted gain is
  `25 * 96 / 37545489 = 0.0000639231`.

Selected reversible packing choices:

- decoder: record-size ascending order, brotli quality `11`, lgwin `18`,
  compressed `162265` bytes
- latents: original compressed payload, brotli quality `9`, lgwin `18`,
  compressed `15868` bytes
- meta: compact sorted JSON, brotli quality `8`, lgwin `18`, compressed `68`
  bytes

## Exact replay attempts

Modal T4 and Modal H100 exact-eval attempts for both public PR95 and the
96-byte repack reached strict inflate and raw-size validation, then failed
inside evaluator startup with DALI/NVML `999` before `contest_auth_eval.json`.
These are provider/evaluator infrastructure failures, not archive or runtime
failures and not score evidence.

Lightning T4 fix2 jobs are the active score truth path:

- `exact_eval_public_pr95_hnerv_muon_t4_fix2_20260504T0848Z`
  - lane: `public_pr95_hnerv_muon_t4_replay_fix2`
  - expected archive bytes/SHA:
    `178417`,
    `e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a`
  - harvested at `2026-05-04T08:58:58Z`
  - evidence: `A++` exact T4 CUDA, 600 samples, promotion eligible
  - exact score: `0.23098329465634826`
  - component distances: PoseNet `0.00017185`, SegNet `0.00070728`
  - result:
    `experiments/results/lightning_batch/exact_eval_public_pr95_hnerv_muon_t4_fix2_20260504T0848Z/contest_auth_eval.adjudicated.json`
  - delta vs previous A++ PR85+STBM1BR+PR92/RMB1: `-0.022523065637629625`

- `exact_eval_pr95_hnerv_muon_repacked_t4_fix2_20260504T0848Z`
  - lane: `pr95_hnerv_muon_repacked_t4_replay_fix2`
  - expected archive bytes/SHA:
    `178321`,
    `2b9b471358f5bba97ea809dcca544ecca26b504fde770a9002046830f469368b`
  - harvested at `2026-05-04T09:01:16Z`
  - evidence: `A++` exact T4 CUDA, 600 samples, promotion eligible
  - exact score: `0.23091954465634829`
  - component distances: PoseNet `0.00017185`, SegNet `0.00070728`
  - result:
    `experiments/results/lightning_batch/exact_eval_pr95_hnerv_muon_repacked_t4_fix2_20260504T0848Z/contest_auth_eval.adjudicated.json`
  - delta vs previous A++ PR85+STBM1BR+PR92/RMB1: `-0.022586815637629598`
  - delta vs public PR95 exact replay: `-0.00006375`

Important exact-eval finding: public PR95's exact T4 score is materially higher
than the PR body's CPU report. Treat the PR body score as external/static only;
the new confirmed frontier is the deterministic 96-byte repack at
`0.23091954465634829`.

## Owned H100 burns

Two patched Modal H100 PR95/HNeRV full-burn replicas are running and syncing
snapshots to the `comma-train-lane-results` Modal volume.

Replica fix1:

- Label: `pr95_hnerv_muon_full_burn_modal_h100_fix1_20260504T0836Z`
- Modal call: `fc-01KQS22WSZ7YR3ZJYXVPPYE4VB`
- Latest observed snapshot at `2026-05-04T08:48:44Z`
- Snapshot archive bytes/SHA:
  `217634`,
  `dded3613936a45745bf0f5a9940cb50145c822a67910cefacbe7639b6a4fa7f8`
- Latest training log reached epoch `110/3000` in Stage 1. Proxy score at
  epoch 100: `0.7097`, archive `217526` bytes.
- Refresh at `2026-05-04T08:51:44Z`: snapshot archive `217148` bytes,
  SHA-256 `d7e9a55d906de58eadb4818bfffb4e1d8e751dde8d0840ac31b6000dad5d09af`.
  Training log reached epoch `140/3000`; proxy score at epoch 125 was
  `0.6421`, archive `217040` bytes.

Replica fix2:

- Label: `pr95_hnerv_muon_full_burn_modal_h100_fix2_20260504T0838Z`
- Modal call: `fc-01KQS25G854XJWFKWCCMYZTDTT`
- Latest observed snapshot at `2026-05-04T08:49:53Z`
- Snapshot archive bytes/SHA:
  `217304`,
  `8e026ec35c2797d3081e867304abaff76faca894b93351501bc349c998ed380b`
- Latest training log reached epoch `100/3000` in Stage 1. Proxy score at
  epoch 100: `0.6972`, archive `217196` bytes.
- Refresh at `2026-05-04T08:52:53Z`: snapshot archive `216300` bytes,
  SHA-256 `721a90829a491aedff7f0608ddc3444de196ad58d4f35b100a5834c9be203a34`.
  Training log reached epoch `130/3000`; proxy score at epoch 125 was
  `0.6321`, archive `216192` bytes.

## PR91/PR92 replay preflight

The PR91/PR92 worker added a deterministic replay preflight:
`experiments/preflight_pr91_pr92_replay_contracts.py`.

Current classification:

- PR91/HPM1 remains blocked with
  `hpm1_probability_range_contract_mismatch`.
- Failure window: `frame=0 group=10 symbol=191` after `5951` decoded symbols.
- All four tested PR91 variants fail closed, so no PR91-derived remote
  dispatch is allowed from this state.
- PR92/RMB1 replay/stacking is unblocked and validated by the current A++
  PR85+STBM1BR+PR92/RMB1 exact T4 packet.

Verification reported by worker:

- `py_compile` for the preflight and tests.
- `pytest src/tac/tests/test_preflight_pr91_pr92_replay_contracts.py -q`.
- `experiments/preflight_pr91_pr92_replay_contracts.py --stdout`.
- focused PR91/PR92 regression tests and `git diff --check`.

Local re-verification by orchestrator:

- `.venv/bin/python -m py_compile experiments/preflight_pr91_pr92_replay_contracts.py src/tac/tests/test_preflight_pr91_pr92_replay_contracts.py`
  passed.
- `.venv/bin/python -m pytest src/tac/tests/test_preflight_pr91_pr92_replay_contracts.py -q`
  passed: `3 passed in 0.44s`.

## Final submission preflight hardening

The final-submission worker added permanent guardrails in:

- `scripts/pre_submission_compliance_check.py`
- `scripts/build_contest_submission_packet.py`
- `src/tac/preflight.py`

Bug classes now guarded:

- exact archive manifest linkage and manifest/archive drift
- packed-payload container ambiguity among `p`, `renderer_payload.bin`, and
  `renderer_payload.bin.br`
- ZIP CRC failure, duplicate names, zip-slip/resource-fork/hidden names, and
  local-header vs central-directory name mismatch
- final report missing exact archive SHA-256 or byte size
- optional dispatch-claim linkage missing a terminal row
- public release hygiene leaking raw Modal app/call identifiers

Local re-verification by orchestrator:

- `.venv/bin/python -m py_compile src/tac/preflight.py scripts/pre_submission_compliance_check.py scripts/build_contest_submission_packet.py src/tac/tests/test_pre_submission_compliance_check.py src/tac/tests/test_build_contest_submission_packet.py`
  passed.
- `.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py src/tac/tests/test_build_contest_submission_packet.py -q`
  passed: `38 passed, 1 warning in 0.59s`. The warning is the intended
  duplicate-ZIP-member regression fixture.
- `git diff --check` passed for the touched final-submission hardening files.

These snapshots are training signals only. They should not consume T4 exact eval
until proxy score and bytes are plausibly competitive with public PR95 or they
produce a distinct architecture/latent artifact worth diagnosis.

## Active decision rule

1. Harvest either Lightning T4 PR95 replay as soon as terminal.
2. If public PR95 validates, the repack becomes the current best candidate if
   exact decoded output is identical and score drops by rate only.
3. If Lightning fails before score, classify artifact logs precisely and keep
   Modal raw-size success as inflate-valid support, not score evidence.
4. Keep H100 burns running while cheap enough, but do not let early high-proxy
   snapshots distract from validated public PR95 replay and byte/coder lowering.

## Local verification

- `git diff --check -- .omx/research/pr95_hnerv_replay_and_burn_20260504_codex.md`
  passed.
- `.venv/bin/python -m pytest src/tac/tests/test_profile_pr95_hnerv_muon_packing.py src/tac/tests/test_lightning_batch_jobs.py -q`
  passed: `124 passed in 2.17s`.

## 2026-05-04T09:17Z continuation

Confirmed frontier remains the PR95 conservative repack A++ exact T4 packet:

- score: `0.23091954465634829`
- bytes: `178321`
- archive SHA-256:
  `2b9b471358f5bba97ea809dcca544ecca26b504fde770a9002046830f469368b`
- PoseNet: `0.00017185`
- SegNet: `0.00070728`
- runtime tree SHA-256:
  `a3f8ab2cfbbdfab53a6d437b0c39f525e0adde2d8bd971765de96aeda4da3dc7`

Additional dispatches:

- `exact_eval_pr95_hnerv_muon_stemperm_t4_20260504T0906Z` is queued/running
  on Lightning T4 for the stem-permutation repack candidate:
  bytes `178277`, SHA-256
  `e40c3f2fb3587b12eccb8707e0a1b7831fde149318f3eb212500c674ccbfbf28`.
  This is a tiny exact byte win only if CUDA confirms no component drift.
- `exact_eval_pr95_hnerv_muon_repack_component_trace_t4_20260504T0914Z`
  is queued on Lightning T4 to produce a component trace for the current PR95
  repack. This unlocks component-weighted signed latent atom planning from
  `experiments/build_pr95_hnerv_residual_atom_plan.py`.
- `exact_eval_public_pr96_rem2_hnerv_t4_20260504T0916Z` is queued on
  Lightning T4 for public PR96 exact replay. Public PR96 body claims `0.21`,
  but no local score claim exists until this exact CUDA packet lands.

Final packet hardening landed after the strict gate found two concrete release
bugs:

- `scripts/build_contest_submission_packet.py` now normalizes copied release
  modes so `inflate.sh` is executable while `archive.zip` and `report.txt` are
  non-executable. This fixes the PR95 runtime source mode mismatch.
- The packet builder now writes `archive_manifest.json` and appends an exact
  custody section to copied `report.txt` with archive SHA-256, archive bytes,
  score, samples, component distances, dispatch lane id, and dispatch job id.
- `src/tac/public_submission_refs.py` now recognizes PR92, PR94, PR95, and
  PR96 as provenance-only public submission references.

The rebuilt packet
`experiments/results/submission_packet_pr95_repack_20260504/apogee_pr95_repack`
passes strict final compliance:

```text
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir experiments/results/submission_packet_pr95_repack_20260504/apogee_pr95_repack \
  --archive experiments/results/submission_packet_pr95_repack_20260504/apogee_pr95_repack/archive.zip \
  --auth-eval-json experiments/results/lightning_batch/exact_eval_pr95_hnerv_muon_repacked_t4_fix2_20260504T0848Z/contest_auth_eval.adjudicated.json \
  --contest-final --expect-single-member 0.bin \
  --expected-archive-sha256 2b9b471358f5bba97ea809dcca544ecca26b504fde770a9002046830f469368b \
  --expected-archive-size-bytes 178321 --expected-samples 600 \
  --expected-runtime-tree-sha256 a3f8ab2cfbbdfab53a6d437b0c39f525e0adde2d8bd971765de96aeda4da3dc7 \
  --dispatch-claims-md .omx/state/active_lane_dispatch_claims.md \
  --expected-lane-id pr95_hnerv_muon_repacked_t4_replay_fix2 \
  --expected-job-id exact_eval_pr95_hnerv_muon_repacked_t4_fix2_20260504T0848Z \
  --source-prs PR95
```

Status: `passed`.

Additional verification:

- `.venv/bin/python -m pytest src/tac/tests/test_build_contest_submission_packet.py src/tac/tests/test_pre_submission_compliance_check.py src/tac/tests/test_public_submission_refs.py -q`
  passed: `42 passed, 1 warning in 0.61s`.
- `git diff --check` passed for the packet builder, final preflight tests, PR
  refs, PR intake ledger, and this ledger.

## 2026-05-04T09:42Z public frontier escalation

Confirmed A++ local champion after the stem-permutation replay:

- archive: `experiments/results/submission_packet_pr95_stemperm_20260504/apogee_pr95_stemperm/archive.zip`
- score: `0.23089404465634825`
- bytes: `178277`
- SHA-256: `e40c3f2fb3587b12eccb8707e0a1b7831fde149318f3eb212500c674ccbfbf28`
- runtime tree SHA-256:
  `a3f8ab2cfbbdfab53a6d437b0c39f525e0adde2d8bd971765de96aeda4da3dc7`

The strict final packet gate passed for
`experiments/results/submission_packet_pr95_stemperm_20260504/apogee_pr95_stemperm`.

New public frontier intake:

- PR98 `hnerv_muon_finetuned_from_pr95 (0.1963)` was downloaded from the
  author release URL, SHA-256
  `7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb`,
  bytes `178392`, one stored `0.bin` member. Static public replay preflight
  passed with runtime tree SHA-256
  `9b689ea8a242ee4cc84241b2bb1bd1fa0e63e881f2efcace8a7c9ca1bae9e9f0`.
- PR99 `hnerv_muon_lc submission (0.20)` was downloaded from the PR, SHA-256
  `278b1c7a1bd6b03a5bceddafcb3489b2624c558ad22825d9211b701333b6eefb`,
  bytes `178546`, one stored `0.bin` member. Static public replay preflight
  passed with runtime tree SHA-256
  `a47c40ac1db7aa2d58aacc5c488b717e1503b28502ca3861180e0d351d5c11be`.

Exact T4 replay dispatches queued:

- `exact_eval_public_pr98_hnerv_muon_finetuned_t4_20260504T0940Z`
  (`public_pr98_hnerv_muon_finetuned_t4_replay`)
- `exact_eval_public_pr99_hnerv_muon_lc_t4_20260504T0940Z`
  (`public_pr99_hnerv_muon_lc_t4_replay`)

Both were claimed before dispatch, staged through manifest-based Lightning
sync, and submitted with PR provenance metadata. The Studio staging host has no
visible CUDA, so staging was rerun with manifest verification but without the
staging CUDA check; the exact eval jobs still perform CUDA/T4 preflight inside
the allocated Batch Job.

Other active branches:

- PR97 original T4 replay has accrued cost but currently reports Pending. A
  premature harvest copied partial artifacts: inflate completed and evaluation
  had started, but no `contest_auth_eval.json` existed yet. Treat this as
  in-progress/status-anomaly evidence only, not a failure or score claim.
- PR95 signed residual atom exact screen
  `exact_eval_pr95_hnerv_residual_atoms_t4_20260504T0932Z` is running on
  T4_SMALL. Candidate bytes are `178319`, SHA-256
  `3910d65a0107f9d2874b5797d152cb52f45837c1f14a5d03b1f257a7178f44b6`.
- Modal H100 PR95 full-burn fix1/fix2 are still running, but current recovered
  snapshots are about 213.9 KB with proxy scores around 0.44, so no exact eval
  is warranted from those snapshots yet.

Bug classes surfaced and handled in this slice:

- zsh special variable collision: using loop variable `path` corrupted command
  lookup. Use names such as `file_path` in zsh download loops.
- stale adjudicator flags on the Lightning exact-eval parser were rejected by
  strict argparse before dispatch. The submit commands were rerun against the
  actual parser surface.
- public PR provenance registry was missing PR98/PR99. `src/tac/public_submission_refs.py`
  and its focused tests now recognize PR98 and PR99.
- source-manifest closure initially missed PR98 `README.md` and PR99 runtime
  `archive.zip`; both manifests were restaged from complete runtime directory
  walks and passed remote SHA verification before submit.
