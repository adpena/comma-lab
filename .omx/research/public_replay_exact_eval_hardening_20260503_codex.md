# Public Replay Exact-Eval Hardening - 2026-05-03

## Scope

Adversarial hardening review for T4 replay of public PR81/PR82 exact eval after
external inflate dependency fixes. No remote GPU dispatch was performed.

Reviewed bug classes:

- missing optional packages and PATH/python mismatch for public replay inflates
- public-vs-robust inflate misrouting
- exact archive/runtime tree custody for replay jobs
- dependency bootstrap artifact custody
- non-promotable public intake/profile overclaiming
- malformed ZIP/preflight leakage
- hidden local paths/secrets in public-facing replay surfaces

## Finding: wrapper could silently use robust inflate for public replay

`scripts/lightning_exact_eval_repro.py` did not expose `--inflate-sh`, so the
repro wrapper always built `launch_lightning_batch_job.py exact-eval` commands
without an explicit public replay inflate path. For PR81/PR82 T4 replay, that is
a fail-open route into the default `submissions/robust_current/inflate.sh`
runtime rather than the public submission replay runtime. Any result from that
misroute would be a runtime-custody comparison, not public replay evidence.

Hardening added:

- `scripts/lightning_exact_eval_repro.py --inflate-sh`
- queue command forwarding of the exact replay inflate path
- plan metadata recording `inflate_runtime.inflate_sh`
- automatic staging of the inflate runtime directory for non-robust replay
  inflates

Evidence:

- `src/tac/tests/test_public_replay_exact_eval_hardening.py::test_public_replay_repro_plan_forwards_external_inflate_and_stages_runtime`

## Finding: external inflate sibling closure was under-specified

`scripts/launch_lightning_batch_job.py` submit validation required only
`inflate.sh` and `config.env` in the staged manifest. Public PR81/PR82 replay
inflates execute sibling Python sources; PR81 also compiles sibling
`range_mask_codec.cpp`. A staged manifest missing these siblings could pass the
local submit gate and fail only after remote job startup, or worse, run a
different local runtime if a stale file exists remotely.

Hardening added:

- non-default exact-eval inflates now require all local non-hidden sibling files
  in the staged source manifest
- robust-current behavior remains unchanged to avoid disrupting its existing
  separately staged runtime closure

Evidence:

- `src/tac/tests/test_public_replay_exact_eval_hardening.py::test_public_replay_submit_requires_external_inflate_sibling_closure`

## Review Notes

Existing guards already covered several reviewed classes:

- T4/g4dn exact-eval submit requires an explicit `INFLATE_TORCH_SPEC`, with
  `+cu124` requiring the PyTorch cu124 index and `UV_INDEX_STRATEGY`.
- External replay inflates skip the robust-current-only seg-tile action
  preflight, preventing public payload misclassification.
- PR81/PR82 static profilers record `score_claim=false`, `promotion_eligible=false`,
  and planning/static evidence grades.
- Lightning artifact validation uses structured JSON score custody, DALI
  bootstrap artifacts, runner preflight artifacts, supply-chain scan artifacts,
  archive SHA/bytes, device, sample count, and adjudication provenance.

Remaining non-blocking review concern:

- `lightning_inflate_runtime_bootstrap.json` is emitted by exact eval commands,
  but it is still treated as an optional harvested artifact in the broad
  validation contract. For public replay promotion, harvest checklists should
  require that artifact whenever `INFLATE_*_SPEC` overrides were used.

## Verification

RED before patch:

```bash
.venv/bin/python -m pytest src/tac/tests/test_public_replay_exact_eval_hardening.py -q
```

Result: 2 failed. The wrapper rejected `--inflate-sh`, and submit validation did
not reject missing external inflate siblings.

GREEN after patch:

```bash
.venv/bin/python -m pytest src/tac/tests/test_public_replay_exact_eval_hardening.py -q
.venv/bin/python -m pytest src/tac/tests/test_lightning_exact_eval_repro.py src/tac/tests/test_seg_tile_actions_preflight.py -q
.venv/bin/python -m pytest src/tac/tests/test_lightning_batch_jobs.py -q
```

Results:

- 2 passed
- 16 passed
- 104 passed

No remote GPU dispatch was performed.

## 2026-05-04 Addendum: Static Public Replay Intake Preflight

Scope: harden public PR replay/compliance intake before exact eval dispatch.
No remote/GPU work was dispatched.

New tool:

- `experiments/preflight_public_replay_intake.py`

Contract:

- Accepts an explicit `--archive` and `--inflate-sh` pair. It does not read PR
  titles, PR self-reported scores, or leaderboard prose.
- Validates ZIP custody before extraction-class workflows: duplicate names,
  central/local filename mismatch, empty/local malformed names, zip-slip/hidden
  member grammar, and charged-member allowlist.
- Performs local member decode smoke and format-magic inspection. For PR85/
  PR91-family single-member `x` archives, it parses the PR85-family micro
  bundle, records segment lengths, SHAs, QMA9/HPM1 mask contracts, and Brotli
  decoded magics for model/pose/qpost side-channel segments.
- Records the exact inflate runtime tree via the same
  `contest_auth_eval_runtime_dependency_manifest_v1` used by canonical auth
  eval, including `runtime_tree_sha256`, runtime file SHAs, static repo-local
  `tac` import closure, and upstream `evaluate.py` hash.
- Scans external replay runtime source for hidden files and large encoded/
  decompressed payload literals that would move charged bytes out of
  `archive.zip`.
- Emits `evidence_grade=external/local_preflight_non_score_until_cuda`,
  `score_claim=false`, `promotion_eligible=false`,
  `cuda_required_for_score=true`, and `dispatch_performed=false` on every
  report.

Dispatch gate semantics:

- `ready_for_exact_eval_dispatch=true` means only that the static archive/
  runtime pair is byte-closed enough to consider exact CUDA replay. It is not a
  score claim and it does not promote a public PR.
- `--fail-if-not-ready` exits nonzero before any queue/dispatch wrapper can use
  a malformed ZIP, unexpected sidecar member, wrong runtime tree, source-
  embedded payload runtime, bad compressed member, or mismatched expected SHA/
  byte/runtime hash.
- PR91-style HPM1 intake remains non-score evidence until canonical CUDA replay
  or an explicit replay-parity artifact succeeds. Its public title/self-report
  is not consumed by the preflight.

Focused tests:

- `src/tac/tests/test_preflight_public_replay_intake.py::test_public_replay_preflight_accepts_byte_closed_x_archive_and_runtime`
- `src/tac/tests/test_preflight_public_replay_intake.py::test_public_replay_preflight_blocks_duplicate_and_sidecar_members`
- `src/tac/tests/test_preflight_public_replay_intake.py::test_public_replay_preflight_blocks_zip_central_local_name_mismatch`
- `src/tac/tests/test_preflight_public_replay_intake.py::test_public_replay_preflight_blocks_source_embedded_payload_runtime`
- `src/tac/tests/test_preflight_public_replay_intake.py::test_public_replay_preflight_expected_runtime_tree_mismatch_fails_closed`

Local artifacts:

- PR85 static intake report:
  `experiments/results/public_pr85_intake_20260503_codex/public_replay_intake_preflight.json`
- PR91 static intake report:
  `experiments/results/public_pr91_intake_20260504_codex/public_replay_intake_preflight.json`
