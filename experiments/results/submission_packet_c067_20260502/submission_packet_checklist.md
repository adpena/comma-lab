# C-067 Submission Packet Checklist

**Status**: METADATA-ONLY CUSTODY PACKET. `score_claim=false`, `ranking_claim=false`, `promotion_claim=false`. The active C-067 score authority is `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.adjudicated.json`.

**Score**: `0.31561703078448233` [contest-CUDA T4 A++], 600 samples, Tesla T4 CUDA.
**Archive bytes**: `276,214`.
**Archive SHA-256**: `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`.
**Components**: SegNet `0.00061244`, PoseNet `0.00049637`.

## Validation checks (all PASSED for the source artifact dir)

- [x] Archive bytes recomputed locally: `276,214`
- [x] Archive SHA-256 recomputed locally: `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`
- [x] `contest_auth_eval.adjudicated.json` `n_samples == 600`
- [x] `contest_auth_eval.adjudicated.json` recorded archive SHA matches local recomputation
- [x] `contest_auth_eval.adjudicated.json` recorded archive_size_bytes matches local
- [x] `eval_provenance.json` device == `cuda`
- [x] `eval_provenance.json` gpu_t4_match == `true`
- [x] `component_trace.json` cross-check passed (SegNet/PoseNet match adjudicated JSON within float epsilon)

## External-source attribution (mandatory for C-067)

C-067 is a fixed-slice composite archive composing three charged payload segments:

| Segment | Bytes | Origin | Authorship |
|---|---:|---|---|
| `mask.obu.br` | 219,472 | PR #67 `qpose14_qzs3_filmq9g_slsb1_r55` deployed `archive.zip` | EthanYangTW (MIN-CHUN Yang), comma.ai PR #67 |
| `model.pt.br` | 55,965 | C-059 internal QZS3-grouped variable-bit-depth FP4 packed renderer | This work |
| `pose_q.br` | 677 | C-059 internal QP1 (delta+VLQ first-column) pose codec output | This work |

The PR #67 mask segment is charged in the deployed archive bytes (no script-side payload movement, no sidecar). The local exact CUDA T4 score is A++ evidence for the exact archive bytes; the mask source remains externally attributed per `docs/paper/EXTERNAL_SOURCE_ATTRIBUTION_C067.md`.

Reverse-engineering custody for the PR #67 mask source: `reports/raw/leaderboard_intel_20260501/pr67_archive.zip` (sha `a5ed8da0d9988943c986b231b4cd33cea0ab878a8e1628134341db5f7f41c765`, 276,564 bytes; the PR #67 deployed archive as downloaded 2026-05-01).

## Compliance posture (charged-payload-closure standard)

- [x] All score-affecting bytes live in `archive.zip` (no script-side payload, no host-local sidecar, no scorer patches)
- [x] Deterministic ZIP construction (per `experiments/build_qpose_archive.py` orchestrator)
- [x] Hidden-file/resource-fork exclusion (no `._*` macOS metadata, no `__MACOSX/`)
- [x] Zip-slip rejected (canonical filename `p` only)
- [x] Scorer-load guards (no scorer parameters loaded at inflate time)
- [x] CUDA-only score truth (per CLAUDE.md `MPS auth eval is NOISE`)
- [x] T4-equivalent CUDA promotion verified (Tesla T4, gpu_t4_match=true)
- [x] Inflate-budget evidence: ~84.6s inflate + 21.7s score = ~106s total (well under 30 min T4 budget)
- [x] PR #67 mask attribution documented + bibliographically citable

## Submission package contents

The deploy-ready submission consists of:

- `archive.zip` (276,214 bytes, sha `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`) — the C-067 fixed-slice composite
- `inflate.sh` from `submissions/exact_current/inflate.sh` (canonical contest inflate path)
- `inflate.py` from `submissions/exact_current/inflate.py` (frozen upstream-snapshot-derived)

Inflate runtime contract: `archive.zip → inflate.sh → upstream/evaluate.py` on T4-equivalent CUDA, 30 min budget.

## Cross-references

- Active frontier report: `reports/latest.md` (C-067 frontier table refreshed at commit `fba28721`)
- Codex writeup ledger: `.omx/research/submission_writeup_integration_20260502_codex.md`
- Working notes: `reports/writeup_working.md`
- Paper draft: `docs/paper/04_results.md` (C-067 frontier table)
- Methodology addendum: `docs/paper/methodology_addendum_atomic_decomposition_yf_floor_20260502.md`
- External-source attribution memo: `docs/paper/EXTERNAL_SOURCE_ATTRIBUTION_C067.md`
- Codex's C-059 packet template: `experiments/results/submission_packet_c059_20260502/`
- Reverse-engineering refs: `reports/raw/leaderboard_intel_20260501/`

## Submission action items (deferred until user signal)

1. Verify `submissions/exact_current/inflate.sh` SHA matches the SHA recorded in `eval_provenance.json` (runtime-custody guard)
2. Confirm submission target (PR description, archive upload location, leaderboard form)
3. Pre-submission contest-CUDA dry-run on a fresh T4 instance to verify reproducibility of the 0.31561703 score from these committed bytes
4. PR description draft including the EXTERNAL_SOURCE_ATTRIBUTION_C067.md attribution boilerplate
5. Acknowledgements section mentioning Quantizr (PR #55), EthanYangTW (PR #67), henosis-us (PR #65), szabolcs-cs (PR #56), and the comma.ai contest organizers

## Update protocol

This packet is metadata-only. If a sub-frontier archive lands (e.g., NeRV-stacked-C-067 sub-0.30), generate a new packet at `experiments/results/submission_packet_<new_id>_20260502/` rather than mutating this one. The packet should be append-only across the contest deadline.
</parameter>
</invoke>