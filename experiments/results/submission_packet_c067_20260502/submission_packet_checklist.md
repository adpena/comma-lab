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
- `inflate.sh` from `submissions/robust_current/inflate.sh` (sha `86449a1f52ac6b2be120d47287b8410f915dce7e562c69f480103f6e527c6017`) — VERIFIED 2026-05-02 to match `eval_provenance.json` `inflate_script_sha256`. This is the Track-B (robust_current) inflate path under which C-067 was scored. (Track A `submissions/exact_current/inflate.sh` has a different SHA `d2c3b491a52d09241e0660c0893e173b2411f4e87d8a3e880f91cb33c1bca36d` and was NOT used to score C-067.)

Inflate runtime contract: `archive.zip → submissions/robust_current/inflate.sh → upstream/evaluate.py` on T4-equivalent CUDA, 30 min budget.

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

1. [x] **DONE 2026-05-02**: Verify `submissions/robust_current/inflate.sh` SHA (`86449a1f52ac6b2be120d47287b8410f915dce7e562c69f480103f6e527c6017`) matches the SHA recorded in `eval_provenance.json` `inflate_script_sha256` field. Live file matches eval-recorded SHA exactly. (Note: prior version of this checklist incorrectly named `submissions/exact_current/inflate.sh` as canonical; corrected — the C-067 contest-CUDA score was measured on Track-B `robust_current` inflate path.)
2. Confirm submission target (PR description, archive upload location, leaderboard form)
3. Pre-submission contest-CUDA dry-run on a fresh T4 instance to verify reproducibility of the 0.31561703 score from these committed bytes
4. PR description draft including the EXTERNAL_SOURCE_ATTRIBUTION_C067.md attribution boilerplate
5. Acknowledgements section mentioning Quantizr (PR #55), EthanYangTW (PR #67), henosis-us (PR #65), szabolcs-cs (PR #56), and the comma.ai contest organizers

## Post-deadline maintenance protocol (per user directive 2026-05-02)

Per user mandate "keep pushing outside the deadline window too" + "we will update our writeup and results and submissions online even after the deadline" + "keep it all alive and push all extreme rigor": this packet is NOT the final state. After contest deadline (May 4 06:59 AM CDT), the engineering effort continues:

- **Append-only packet schema**: each new sub-frontier archive lands a new `experiments/results/submission_packet_<id>_<date>/` directory; this C-067 packet remains immutable as the historical-deadline submission record.
- **Post-deadline reactivations**: Council 5/0 SHIP-C-067 verdict for the deadline window does NOT close out the dispatched lanes — Lane 12 NeRV Path B (parser + Alpha-Geo contracts + L2 clearance), Quantizr's recipe #4 (KL-soft-distill on AV1 logits T=2.0), Block-FP transplant, and the other deferred sub-0.3 attempts remain ALIVE post-deadline workstreams under continued extreme-rigor council review.
- **Public-facing artifact updates**: writeup (`docs/paper/`), site, GitHub PR description, leaderboard submission may be continuously updated as the post-deadline measurement cadence produces new sub-frontier results. The leaderboard submission mechanic + cadence is contest-organizer dependent and must be checked for each update.
- **No "lane killed" verdicts** in the registry mean "permanently dead" — all kills are scoped to a deadline window and reactivate post-window unless an empirical-falsification has been recorded with full council review per `feedback_grand_council_imp_permanent_fix_review_20260430` standard.

## Update protocol

This packet is metadata-only. If a sub-frontier archive lands (e.g., NeRV-stacked-C-067 sub-0.30), generate a new packet at `experiments/results/submission_packet_<new_id>_20260502/` rather than mutating this one. The packet should be append-only across the contest deadline.
</parameter>
</invoke>