# Public frontier drift adversarial review - 2026-05-08

Evidence grade: `A++` for the PR102 hardened contest-CUDA replay only;
`empirical`/`external` for public leaderboard and GitHub-comment comparisons.
Score claim: `true` only for the exact PR102 CUDA replay artifact listed below.

## Verdict

The current PR102 experiment/replay is not broken. The first Lightning replay
attempt inflated the corrected archive and then failed inside upstream
evaluation because the remote torch/torchvision install lacked the
`torchvision::nms` operator. That is infrastructure/runtime-package evidence,
not PR102 method evidence.

The hardened replay then scored the same corrected archive on Tesla T4 CUDA:

- Archive: `178981` bytes,
  `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`.
- Runtime tree:
  `62560a0411dc341286eebfaf6e8ed79564efeb14fb8da5e3e1be026611e7aba1`.
- Components: `avg_posenet_dist=0.00017347`,
  `avg_segnet_dist=0.00067568`, `rate_unscaled=0.00476704`.
- Recomputed score: `0.22839372989108092`.
- Hardware: Tesla T4, `device=cuda`, `n_samples=600`,
  contest-equivalent hardware.
- Custody:
  `experiments/results/lightning_batch/pr102-public-exact-replay-hardened-g4dn2-20260508T103725Z/contest_auth_eval.adjudicated.json`.

This agrees with the public PR102 CUDA GitHub Action comment band
(`0.22839083118` from rounded public components; delta about `0.000002898711`)
and disagrees with the public CPU/leaderboard band (`0.195376176526`; delta
about `0.033393729891`). The drift is therefore explained as upstream/public
CPU-vs-CUDA evaluation drift plus leaderboard policy, not a local exact-replay
failure.

## Findings

1. Severity: high. Public leaderboard position is not the CUDA score truth for
   internal promotion.

   The current README and comma.ai leaderboard still show PR101 first at
   `0.193`, PR103 second at `0.195`, and PR102 third at `0.195`. Public PR
   comments show those rows were rescored on CPU after CUDA comments. Local
   promotion/ranking inside this repo must continue to use exact CUDA auth eval,
   so the public leaderboard rows are external/public ranking facts, not local
   CUDA frontier scores.

2. Severity: high. PR102 is A++ exact CUDA evidence, but it is not a new
   archive-byte atom.

   PR102 uses the same archive SHA as PR100:
   `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`.
   The score movement is runtime/source behavior over identical charged bytes,
   including the PR102 scale/r+1 runtime path. Treat PR102 as a replay/runtime
   atom until byte-level decode/re-encode parity and compress-to-archive
   reproduction exist.

3. Severity: high. PR104 remains an evidence hole.

   The reproduction ledger has no same-archive local exact CUDA replay for
   PR104 `qhnerv_ft_best`. Public comments show a CUDA evaluation around
   `0.231145103318` recomputed from rounded components, and public leaderboards
   show `0.231`, but the local ledger row has `exact_cuda_replay_missing`,
   `same_archive_structured_exact_eval_json_missing`, and no research note.
   PR104 cannot be used to support drift classification or stack atom decisions
   until that local exact replay lands.

4. Severity: medium. Log-only replays are custody evidence, but not complete
   structured replay packets.

   PR100, PR101, PR103, and PR107 have same-archive CUDA scores in the ledger
   but `same_archive_structured_exact_eval_json_missing`. They should not be
   cited as A++ structured packets or used for component-driven optimization
   until `contest_auth_eval.json` and adjudicated JSON are present or the same
   archive/runtime pair is rerun to produce them.

5. Severity: medium. No PR100-107 row is ready as a promoted stack atom.

   The reproduction ledger marks `ready_for_stack_atom=false` for all rows.
   Every row still lacks compress-to-archive 1:1 reproduction, and most rows
   also lack decode/re-encode parity. Current binary understanding is mostly
   member-prefix inventory, not a payload grammar sufficient for safe
   recomposition.

6. Severity: medium. PR102 component-trace evidence is diagnostic-missing, not
   negative.

   The PR102 component trace failed because
   `tac.scoring.evaluate_archive_per_pair` was unavailable. This does not
   downgrade the exact score, but it blocks pair-level trust-region and
   water-fill analysis from this replay.

7. Severity: low. PR108 does not change the frontier at this refresh.

   Live refresh still sees PR108 open, absent from the README/comma.ai
   leaderboard, with the local intake classifying it as non-frontier by its own
   CPU report. It remains useful as an AV/ROI reference, not a HNeRV frontier
   drift explanation.

## Replay table

| PR | public row | local same-archive CUDA | drift status | structured JSON | blocker summary |
|---:|---:|---:|---|---:|---|
| 100 | `0.195` | `0.22826947142244708` | CUDA mismatches CPU/leaderboard | no | missing compress entrypoint, compression reproduction, decode/re-encode, structured JSON |
| 101 | `0.193` | `0.22635331443973267` | CUDA mismatches CPU/leaderboard | no | missing compress entrypoint, compression reproduction, decode/re-encode, structured JSON |
| 102 | `0.195` | `0.22839372989108092` | CUDA matches public CUDA comment, mismatches CPU/leaderboard | yes | missing compression reproduction and decode/re-encode parity |
| 103 | `0.195` | `0.2277649714224471` | CUDA mismatches CPU/leaderboard | no | missing compress entrypoint, compression reproduction, structured JSON |
| 104 | `0.231` | missing | no local comparison | no | exact CUDA replay, research note, compression reproduction, decode/re-encode, structured JSON |
| 105 | `0.198` | `0.23043732986984997` | CUDA mismatches CPU/leaderboard | yes | missing compression reproduction, decode/re-encode, research note |
| 106 | `0.209` | `0.20945673680571203` | matches local CUDA within public rounding | yes | missing compression reproduction and decode/re-encode parity |
| 107 | `0.229` | `0.22933111465960354` | matches local CUDA within public rounding | no | missing compress entrypoint, compression reproduction, decode/re-encode, structured JSON |

## Claims to downgrade

- Downgrade "public leaderboard score equals local CUDA frontier score" to
  `external/public leaderboard fact`. The public rows are CPU-band for PR100,
  PR101, PR102, PR103, and PR105.
- Downgrade PR100, PR101, PR103, and PR107 exact-replay references from
  "structured A++ packet" to "same-archive CUDA score observed from logs" until
  JSON custody exists.
- Downgrade any "PR102 byte win" or "PR102 archive atom" language to
  "runtime/source replay atom over PR100-identical archive bytes" until
  compression and decode/re-encode proofs exist.
- Downgrade "PR104 drift resolved" to "unresolved; public CUDA comment only"
  until local same-archive exact CUDA replay exists.
- Downgrade "component-trace-ready PR102" to "exact score ready, diagnostic
  component trace unavailable".

## Exact next actions

1. PR104: create a source-sized replay adapter for
   `experiments/results/public_pr_intake_full/public_pr104_intake_20260505_auto`
   and, only after dispatch is permitted and a lane claim is active, run a
   same-archive exact CUDA replay for archive
   `6564c32a9edeeaf08abd7f0ea673ba2fda23444605ca207eb4ba794cc66797b8`.

2. Structured JSON closure: for PR100, PR101, PR103, and PR107, either recover
   `contest_auth_eval.json`/adjudicated JSON from the replay artifact if present
   but unindexed, or rerun the exact same archive/runtime identity to emit
   structured JSON. Do not promote those rows from log-only evidence.

3. Compression reproduction: for every PR100-107 row, run the public
   `compress.sh` or documented compression entrypoint in an isolated copy with
   external artifacts pinned. Classify each as true compression, archive fetch,
   missing entrypoint, or non-reproducible, and compare output archive SHA and
   ZIP member SHA to the scored archive.

4. Decode/re-encode parity: turn the single-member archive inventories into
   payload grammar notes and no-op controls. A candidate is not a stack atom
   until decode/re-encode parity or a precise non-reversible-runtime reason is
   recorded.

5. PR102 component traces: repair or replace the per-pair trace path requiring
   `tac.scoring.evaluate_archive_per_pair`, then rerun diagnostic tracing
   against the already-harvested PR102 exact replay. This is diagnostic only
   and should not affect the A++ score claim.

6. Public refresh discipline: before any future public-frontier conclusion,
   refresh GitHub PR state, README, and comma.ai leaderboard state. At this
   review, latest PRs seen were PR108 through PR99, with PR108 still open and
   not listed on the public leaderboard.

## Commands run

- `git status --short --branch`
- `rg -n "public HNeRV|public frontier|PR102|PR10[0-8]|HNeRV" /Users/adpena/.codex/memories/MEMORY.md`
- `nl -ba /Users/adpena/.codex/memories/MEMORY.md | sed -n '169,181p'`
- `nl -ba /Users/adpena/.codex/memories/MEMORY.md | sed -n '223,228p'`
- `wc -c` over the specified input files
- `jq` summaries of
  `.omx/research/pr102_hardened_exact_replay_result_20260508_codex.json`
- `sed -n` and `rg -n` over
  `.omx/research/pr102_exact_replay_readiness_20260508_codex.md`
- `sed -n` over
  `.omx/research/pr100_107_reproduction_ledger_20260507_codex.md`
- `jq` summaries of
  `experiments/results/pr100_107_reproduction_ledger_20260507_codex/ledger.json`
- `jq` summaries of
  `reverse_engineering/public_pr102_pr108_intake_20260508/manifest.json`
- `ls -1` and `sed -n` over the PR102 Lightning replay artifact directories
- attempted `find ... -printf` once while checking PR102 artifacts; it produced
  no useful output on this macOS checkout and was replaced with `ls -1`
- `jq` over PR102 `contest_auth_eval.json` and `component_trace_status.json`
- `curl -fsSL` GitHub pulls, README, PR comments for PR100-108, and
  `https://comma.ai/leaderboard`
- `awk` recomputation for the rounded PR104 public CUDA-comment score
