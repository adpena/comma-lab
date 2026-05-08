# Public replay drift hypothesis matrix - Codex - 2026-05-08

Scope: PR102 and PR104 public replay drift versus public comments and local
exact CUDA. No dispatch was performed. No score artifacts, archives, runtime
files, reports, or state files were changed.

Evidence grade: `evidence_audit_no_score`. Score claim: false.

## Bottom line

PR102 drift is best explained by public CPU-vs-CUDA evaluation ambiguity, not
by a broken local replay. The local hardened T4 CUDA replay of the canonical
PR102 archive scored `0.22839372989108092`, matching the public CUDA comment
band (`0.228390831179840` recomputed from rounded public fields) and not the
later public CPU/leaderboard band (`0.195376176526498` recomputed from rounded
public fields).

PR104 does not show meaningful drift against its public CUDA comment. The local
root-staged T4 CUDA replay scored `0.23113446620399658`, within
`0.000010637114003` of the public CUDA comment score recomputed from rounded
fields (`0.231145103318179`).

## Evidence inspected

Local exact artifacts:

- PR102:
  `experiments/results/lightning_batch/pr102-public-exact-replay-hardened-g4dn2-20260508T103725Z/contest_auth_eval.adjudicated.json`
  - archive `178981` bytes,
    `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
  - runtime tree
    `62560a0411dc341286eebfaf6e8ed79564efeb14fb8da5e3e1be026611e7aba1`
  - Tesla T4 CUDA, `n_samples=600`, upstream `evaluate.py`
    `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`
  - score `0.22839372989108092`
- PR104:
  `experiments/results/lightning_batch/pr104-public-exact-replay-rootstaged-g4dn2-20260508T1130Z/contest_auth_eval.adjudicated.json`
  - archive `178637` bytes,
    `6564c32a9edeeaf08abd7f0ea673ba2fda23444605ca207eb4ba794cc66797b8`
  - runtime tree
    `40e47f8677abc9885179c848a4f44096aa024ac0fb1a937ba8d99e29b49147b9`
  - Tesla T4 CUDA, `n_samples=600`, upstream `evaluate.py`
    `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`
  - score `0.23113446620399658`

Public comment refresh via `gh pr view`:

- PR102 `https://github.com/commaai/comma_video_compression_challenge/pull/102`
  - public CUDA comment: device `cuda`, pose `0.00017347`, seg `0.00067565`,
    bytes `178981`, 600 samples, rounded display score `0.23`
  - public CPU comment: device `cpu`, pose `0.00003460`, seg `0.00057599`,
    bytes `178981`, 600 samples, rounded display score `0.20`
  - maintainer prize comment identifies this PR as third prize
- PR104 `https://github.com/commaai/comma_video_compression_challenge/pull/104`
  - public CUDA comment: device `cuda`, pose `0.00017235`, seg `0.00070683`,
    bytes `178637`, 600 samples, rounded display score `0.23`

Local custody and ledgers inspected:

- `.omx/research/public_frontier_drift_adversarial_review_20260508_codex.md`
- `.omx/research/evidence_grade_drift_pr102_pr104_pr106_codex_20260508.md`
- `.omx/research/pr102_exact_replay_readiness_20260508_codex.md`
- `.omx/research/pr104_exact_replay_readiness_20260508_codex.md`
- `.omx/research/pr104_exact_replay_dispatch_status_20260508_codex.md`
- `.omx/research/public_pr102_pr108_reverse_engineering_intake_20260508_codex.md`
- `experiments/results/pr100_107_reproduction_ledger_20260507_codex/ledger.json`
- PR102 canonical archive identity check:
  `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/release_identity_checks/archive_identity_check.json`

## Hypothesis matrix

| Rank | Drift source | Status | Evidence | What would falsify or revive it |
| ---: | --- | --- | --- | --- |
| 1 | Public comment CPU/GPU ambiguity | Strong for PR102; not needed for PR104 | PR102 has both public CUDA and CPU comments over 600 samples and the same byte count. Local T4 CUDA matches the public CUDA comment band within `0.000002899`, while differing from the CPU band by about `0.033018`. PR104 has only a public CUDA result in the refreshed comments and local CUDA matches it within rounded-component noise. | Falsify for PR102 by producing a public, same-archive, same-runtime CUDA record near `0.195376`, or by showing the CPU comment used a different archive/runtime/frame set. Revive for PR104 only if a hidden or later public CPU/leaderboard source claims a materially different same-archive score. |
| 2 | Archive identity / wrong archive | Mostly falsified for current exact PR102 and PR104; historical PR102 hazard remains | `shasum` matches local intake archive to harvested exact archive for both PR102 and PR104. Both exact archives have single member `0.bin`. PR102 identity check shows the canonical comment attachment, EthanYang release, and PR100 release share SHA `afd53348...`, while the stale wrong qpose auto-intake was a different `276481` byte member `p` archive. | Falsify current archive closure only by finding a public authoritative archive SHA different from the harvested one that is tied to the public score. Revive if any downstream ledger or tool still routes PR102 through the stale qpose archive. |
| 3 | Runtime dependency / package drift | Confirmed infrastructure failure class; weak as score-drift cause after hardening | The first PR102 replay failed before structured JSON with `RuntimeError: operator torchvision::nms does not exist`. Hardened PR102 and PR104 replays pinned `torch==2.5.1+cu124` and `torchvision==0.20.1+cu124`, ran on T4 CUDA, and produced structured exact JSON. PR104's first job also exposed a dependency-root staging gap and was stopped before the root-staged relaunch. | Falsify as a score-drift source by rerunning with package pins varied while archive, runtime source, device, frame set, and evaluator are fixed and showing the score remains stable within rounding. Revive if an exact replay reaches a different score solely by changing torch/torchvision/brotli/ffmpeg closure. |
| 4 | Runtime-tree differences | Plausible secondary audit surface; not the main observed drift | PR102 local replay used an adapter/runtime tree SHA `62560a...`; PR104 used adapter/runtime tree SHA `40e47f...` with enumerated external dependency root. Both local CUDA results match public CUDA comments, so runtime-tree differences do not explain PR102 CUDA-vs-CUDA. PR102 still remains a runtime/source atom over PR100-identical archive bytes, not a new archive-byte atom. | Falsify as a remaining concern by building byte/SHA manifests of the public GitHub Actions runtime tree and showing exact parity with local adapter source or a behavior-equivalent no-network adapter. Revive if source-tree delta alone, with archive/device/frame/evaluator fixed, reproduces the CPU/leaderboard band. |
| 5 | Evaluator commit / evaluator code drift | Weak | Local exact provenance records upstream commit `c5e1274e54e47f81b121bc3bf75eaa9a432b1837` and `evaluate.py` SHA `7da71a84...`. Local `upstream/evaluate.py`, PR102 intake source, and PR104 intake source have the same SHA. Current local upstream HEAD differs only in leaderboard/documentation context for this audit; `evaluate.py` content did not change. Public comments do not expose an evaluator commit, so this is not closed absolutely. | Falsify by tying the public GitHub Actions run to the same `evaluate.py` SHA and video-name file hash. Revive if public workflow logs show a different `evaluate.py` or scoring formula commit for the CPU or CUDA comments. |
| 6 | Frame set / video names | Weakest observed source | Public comments and local exact JSONs all say 600 samples. Local exact provenance uses `upstream/public_test_video_names.txt`; the local file, PR102 intake source, and PR104 intake source share SHA `7ff99d08...`. There is no evidence of a different frame set in the public comments. | Falsify by hashing the exact public GitHub Actions `public_test_video_names.txt` and video payloads and matching local custody. Revive if public logs or artifacts show a different video names file, video payload, sample order, or preprocessing path for CPU versus CUDA. |

## Interpretation rules

1. PR102 local exact CUDA is valid custody evidence for the canonical PR102
   archive/runtime pair, but the public leaderboard/CPU score is not local CUDA
   truth.
2. PR104 is now closed as a local exact CUDA replay gap. It is non-frontier
   versus current local anchors, but it is no longer "public comment only".
3. Public comments are process evidence unless tied to archive SHA, runtime
   tree, evaluator hash, frame set, hardware/device, and structured component
   recomputation.
4. The highest-value follow-up is not another same-archive PR102/PR104 replay.
   It is provenance closure: public workflow log capture for evaluator commit,
   video payload hashes, and runtime-source tree hashes if those are needed for
   paper-grade explanation.

## Commands run

- `git branch --show-current && git status --short`
- `rg` over memory, `.omx/research`, `experiments/results`, `reports`, and
  `reverse_engineering` for PR102/PR104 drift artifacts
- `jq` summaries of PR102 and PR104 `contest_auth_eval.adjudicated.json`
- `gh pr view 102 --repo commaai/comma_video_compression_challenge --json ...`
- `gh pr view 104 --repo commaai/comma_video_compression_challenge --json ...`
- `awk` recomputation of public rounded-component scores
- `shasum -a 256` for intake archives, harvested archives, `evaluate.py`, and
  `public_test_video_names.txt`
- `zipinfo -1` for PR102/PR104 archive member names
- `rg` over exact replay logs for dependency/runtime failures
