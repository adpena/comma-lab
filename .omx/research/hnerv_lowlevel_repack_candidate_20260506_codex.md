# HNeRV Low-Level Repack Candidate - 2026-05-06

Evidence grade: `byte_profile_only`

Score claim: `false`

This tranche turned the HNeRV payload scorecard follow-up into a concrete
archive candidate against the current PR106x exact frontier. The candidate only
recodes the packed HNeRV decoder brotli section and preserves brotli raw
equivalence for the decoded section payload.

## Result

- Source archive: `experiments/results/public_pr106_belt_and_suspenders_xrepack_20260504_codex/archive.zip`
- Source label: `PR106x`
- Source archive bytes: `186231`
- Candidate archive: `experiments/results/hnerv_lowlevel_repack_pr106x_20260506_codex/pr106x_hnerv_brotli_repack_candidate.zip`
- Candidate archive bytes: `186080`
- Archive byte delta: `-151`
- Repacked section: `decoder_packed_brotli`
- Source section bytes: `170278`
- Candidate section bytes: `170127`
- Candidate diff audit blockers: none
- Ready for archive preflight: `true`
- Ready for exact eval dispatch: `false`

The expected rate-only change if SegNet/PoseNet components are bit-identical is
approximately `-0.000100546893` score. This is not a promoted score delta until
the candidate archive passes exact CUDA auth eval through the canonical path.

## Follow-Up

1. Run strict archive manifest/preflight on the candidate archive.
2. Claim the lane before any remote eval dispatch.
3. Run exact CUDA auth eval on T4/equivalent hardware.
4. If exact components are unchanged, promote as a byte-only HNeRV archive
   improvement; if components drift, preserve as a negative payload-custody
   result and investigate runtime/member-name effects.

## 2026-05-06 Custody-V2 Dispatch Update

Evidence grade remains `external/local_preflight_non_score_until_cuda`.

The strict public replay preflight now passes and records the public PR106
runtime source tree as an explicit runtime dependency, not just the one-file
adapter:

- Preflight artifact:
  `experiments/results/hnerv_lowlevel_repack_pr106x_20260506_codex/public_replay_preflight.json`
- Candidate archive SHA-256:
  `b0a12549a39e34a0d7f83ea99e05e55fcd01d795a15db2ffb3d92ccc6267e53f`
- Candidate archive bytes: `186080`
- Adapter SHA-256:
  `9bfa00e3790cbee7687dff8a484fe83a9ca1c82f18b157f93185fab31662c0f0`
- Runtime tree SHA-256 with declared external PR106 source root:
  `06080a260017887f591e304c7cdcbe42ac5215f10fe12ecc00ddb1bc72870534`
- External runtime root:
  `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders`
- External runtime source scan bytes (`.py`/`.sh`): `69896`
- Blockers: none
- Ready for exact eval dispatch: `true`

The first exact-eval submit for
`exact_eval_pr106x_lowlevel_brotli_repack_t4_20260506` was stopped before
running because it was submitted before external runtime dependency roots were
included in the runtime custody hash. The lane claim was closed as
`stopped_superseded_by_runtime_dependency_custody_v2`.

The custody-v2 exact CUDA eval was staged and queued as:

- Job: `exact_eval_pr106x_lowlevel_brotli_repack_custody_v2_t4_20260506`
- Lane: `pr106x_lowlevel_brotli_repack`
- Queue status at `2026-05-06T08:55:51Z`: `Pending`
- Source manifest:
  `.omx/state/exact_eval_pr106x_lowlevel_brotli_repack_custody_v2_t4_20260506_manifest.json`
- Manifest SHA-256:
  `4e241cd66d85bad4bc52046ccc3ec2088b7d501c88282ad73d47557a81dda4e8`
- Staged file count: `1411`
- Staged total bytes: `22029886`

This job is the canonical PR106x low-level brotli repack score run. Do not
promote the candidate from the stopped pre-custody-v2 job or from byte evidence
alone.

## 2026-05-06 Adapter Fail-Closed Hardening

After the custody-v2 job was queued, a read-only adversarial review found two
adapter sharp edges: the adapter could silently prefer `${base}.bin` when both
`${base}.bin` and `x` existed, and it could fall back to ambient `python3` when
the managed `.venv/bin/python` was absent.

These edge cases do not change the queued candidate's semantics because the
candidate archive has exactly one charged member (`x`) and the staged Lightning
workspace has `.venv/bin/python`. The running custody-v2 job remains valid
pending exact CUDA output, but any follow-up or rerun should use the hardened
adapter/preflight:

- Hardened adapter SHA-256:
  `02d5e131790bf4f3f7dbb4e9ae9603ef7bfccf1ad4f4b3be7cd9e06e7be57dfa`
- Hardened runtime tree SHA-256:
  `f402908b2490718c4f7b76987335ec1a496cb12ab71c27e1e1aea4024d5712cb`
- Hardened preflight artifact:
  `experiments/results/hnerv_lowlevel_repack_pr106x_20260506_codex/public_replay_preflight.json`
- Hardened preflight blockers: none

The adapter now fails closed if both possible payload member names exist and
requires a managed executable Python (`${PYTHON}` override or repo
`.venv/bin/python`).

## 2026-05-06 Exact CUDA Result And Diagnostic Failure Class

Evidence grade: `A++ contest T4`

Score claim: `true`

The custody-v2 Lightning exact CUDA run completed the canonical score path:

- Job: `exact_eval_pr106x_lowlevel_brotli_repack_custody_v2_t4_20260506`
- Local artifact directory:
  `experiments/results/lightning_batch/exact_eval_pr106x_lowlevel_brotli_repack_custody_v2_t4_20260506`
- Candidate archive SHA-256:
  `b0a12549a39e34a0d7f83ea99e05e55fcd01d795a15db2ffb3d92ccc6267e53f`
- Candidate archive bytes: `186080`
- Canonical recomputed score: `0.20935073680571203`
- Average PoseNet distortion: `0.00003351`
- Average SegNet distortion: `0.00067142`
- Samples: `600`
- Device: `cuda`
- GPU: `Tesla T4`
- Runtime tree SHA-256 recorded by exact eval:
  `bb6baee66c61781f285fee5862ab499b8eb1fec93edeea046cb3019289638fd3`
- External runtime dependency roots recorded: `1`
- External runtime dependency root file count: `20`
- Baseline PR106x score:
  `0.20945123680571204`
- Score delta vs PR106x baseline:
  `-0.00010050000000000336`
- Archive byte delta vs PR106x baseline: `-151`

Component gates passed against the PR106x baseline with exact measured parity:
PoseNet relative `1.0`, SegNet relative `1.0`. This is a byte-only HNeRV
archive improvement, not a representation or neural-output change.

Lightning marked the job failed after exact scoring because the optional
diagnostic component-trace step crashed:

```text
RuntimeError: tac.scoring.evaluate_archive_per_pair is unavailable; cannot compute diagnostic component trace
```

This diagnostic failure did not affect `archive.zip -> inflate.sh ->
upstream/evaluate.py` scoring. The dispatch claim was closed as
`completed_a_pp_score_harvested_component_trace_optional_failure`.

Follow-up hardening landed in `src/tac/deploy/lightning/batch_jobs.py`:
optional component trace now writes `component_trace_status.json` and cannot
turn a valid exact CUDA score into a failed Lightning job. Diagnostic trace
JSON remains validated if present, but missing/unavailable trace evidence is
recorded as non-score diagnostic status rather than blocking adjudicated score
custody.
