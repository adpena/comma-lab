# ddm_pk1 — move-40 PR #140 staging packet, nothing published (2026-09-10)

`[no-triality] [p0-ledger-ok]` · research_only=true · new_score_claim=false

The pointer did not move. The requested byte-exact staging copy and report are retained; strict compliance **REFUSED (70 PASS / 6 FAIL / 76 checks, rc=1)**. This is a reviewable staging packet, not a release-ready one-command swap. Every refusal is retained without waiver. Publication remains queued to MAIN/operator.

## Verification

| Obligation | Finding |
|---|---|
| Candidate shippable copy | 47/47 files copied byte-exact into `submissions/_staging_move40_pr140_swap/`; 48 files including regenerated report |
| Archive | 180,233 B, SHA-256 `986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857`; equals current CUDA pointer |
| Receipt | SHA-256 `cd6d5ef5243e26fa1868100a2bbb34efe0d6446b4a7fb9bd1eab7f309f5d3d00`; pinned charter receipt verified |
| Components [contest-CUDA T4 n600], retained | d_seg 0.00010636; d_pose 4.89e-06; exact archive denominator 37,545,489 B |
| Derived score | 100*d_seg + sqrt(10*d_pose) + 25*bytes/37,545,489 = **0.13763861019288715**, exact Python-float equality to charter, pointer and receipt |
| Arithmetic producer | `tac.pointer_move.score_row_from_harvest` and `cross_check_against_report`; direct formula equality independently asserted; packet CLI not applied |
| Precision | Eight-decimal evaluator report components, not unrounded tensors; retained worst-case score rounding bound 4.07599281621511e-06 |
| T4 time, retained | inflation 990.053829427 s; evaluate 45.23132363599984 s; no local timing claim |
| Receiver identity | 44/44 staged .py/.sh/.c files equal corresponding retained T4 runtime-manifest hashes |
| Retracted code | 0 text matches for `tc3`, `tc4`, lane-predictor and pr8 named symbols across all 47 non-archive staged files; no matching filenames. Source identity provides the stronger binding to move40. No claim of a fresh semantic decode |
| Source MANIFEST | 44/46 PASS, 1 stale `inflate.py` hash, 1 missing `compress.py` (outside charter copy list); copied manifest kept unchanged |
| Source README | Still move23/AFR1 score, hash and reproduction text; preserved verbatim per charter, BLOCKS unqualified publication |
| Staged executable mode | `inflate.sh` changed 0644 → 0755; bytes unchanged; source mode unchanged |
| Gitignore | archive.zip ignored, README/source not all ignored; no staging-tree file/binary included in metadata landing |
| Boundaries | Source 48 files and live 40 files hash-unchanged; shared index and upstream evaluator unchanged |
| Commands | Bash and embedded Python syntax checked; copy/commit/push/hosting/PR commands **NOT RUN** |

The charter's move memo path does not exist. The discovered canonical packet is `.omx/research/ddm_sj1_t4_compose39_rp1_union_20260910_pointer_move_40_20260910.md`; it and current pointer agree. The live tree has no `report.txt`; the evaluator's retained report supplies the format. Its evaluation-results block is retained verbatim, with custody/score lines appended and provider configuration paths omitted.

## Strict compliance refusal — exact terminal check details

- `auth_eval_raw_promotion_policy_blockers_absent`: {"promotion_blockers": ["raw_auth_eval_does_not_verify_submission_policy_gates", "cpu_leaderboard_reproduction_not_adjudicated", "pre_submission_compliance_check_not_recorded"], "rank_or_kill_blockers": ["raw_auth_eval_not_rank_or_kill_authority", "requires_adjudicated_cuda_cpu_policy_review"]}
- `contest_cpu_auth_eval_exists`: submissions/_staging_move40_pr140_swap/contest_cpu_auth_eval.json
- `submission_runtime_imports_within_allowlist`: disallowed=['submissions/_staging_move40_pr140_swap/runtime/rc3_shared_mixer.py:experiments', 'submissions/_staging_move40_pr140_swap/runtime/sm1_semantic_mixer.py:experiments']
- `submission_runtime_tree_matches_auth_eval`: submission_candidates=['7cb8e0d0756800e81ff05b1cf682c0b6297c3d719bd749759e67f85bcffa6627', '80e6855fe36c2e0255ac86f8cb6b990aa53347400cf0017fec82a4e7fa1ed142'] auth_eval_candidates={'provenance.inflate_runtime_manifest.runtime_tree_sha256': '27fc92e8ee156d01ffe65ceb2e141ac0c98a0ccd7b9096d12aa18919fcb28a39', 'provenance.inflate_runtime_manifest.runtime_tree_sha256_without_submission_custody_files': '27fc92e8ee156d01ffe65ceb2e141ac0c98a0ccd7b9096d12aa18919fcb28a39', 'provenance.inflate_runtime_manifest.portable_runtime_tree_sha256_without_submission_custody_files': 'a513161cf789dfecece5b7e381b61be97f9a41d0ab30bee25d47b8faed9f0c62'} runtime_equivalence_proof_valid=False
- `archive_manifest_exists`: submissions/_staging_move40_pr140_swap/archive_manifest.json
- `hosted_archive_manifest_supplied`: strict contest-final packets require --hosted-archive-manifest-json

Full machine output: `.omx/research/ddm_pk1_20260910/COMPLIANCE.json`; stdout/stderr and argv are adjacent. R1 (67/9) and R2 (69/7) are preserved. The only subsequent changes were the executable permission and passing the actual competitive answer (with explicit score context) instead of the PR template question to the policy-input flag. Terminal R3 is 70/6. No gate was disabled and no authority flags or receipt blockers were edited.

The two forbidden import hits are the `except ImportError` research fallbacks in rc3/sm1. Their normal local imports exist; this explains a plausible cure, but does not waive the static rule or prove a changed receiver. The tree mismatch is consistent with excluding the evaluated `compress.py` member (45 evaluated code files, 44 staged); per-file equality does not substitute for the checker's required equivalence proof. Source MANIFEST and README failures above are additional packet defects, not included in the six gate reds.

## Disclosure before / after — operator approval pending

Before (exact local posted-body snapshot; pr5/pr6 independently recorded the same sentence):

> I used coding agents (Claude as orchestrator of Codex subagents) extensively as research and engineering tools for the work behind this submission.

After (draft):

> I used automated research and engineering tools extensively for the work behind this submission.

`PR_BODY.md` is a complete move40 draft with `linux-nvidia-t4`, generated receipt values, inherited-vehicle credits, and no named-assistant attribution. The proposed release URL is visibly labeled unhosted/unverified. It does not claim the old compressor rebuilds move40. This arm made no live public-state query and does not claim the prior posted snapshot is freshly verified. `SWAP_COMMANDS.md` requires all applicable readiness/approval conditions and never replaces the existing AFR1 release asset.

## RECALL EVIDENCE

Queries and output paths are recorded in `RECALL_SEARCHES.json`: content search of all `.omx/research/*.md` for `PR #140|PR140|p0_swap_procedure_no_push_without_confirm|hosted_bytes_identical` (123 matching lines); index/DAG search for `PR #140|PR140|p0_swap_procedure|hosted.archive` (0 matching lines in the named index/DAG scope); docs/design search for `PR #140|PR140|p0_swap_procedure|submission.packet` (45 lines); canonical task ledger search for `PR #140|PR140|ddm_pk1|p0_swap_procedure` (3 lines). Canonical equations CLI searched all 483 emitted rows; pointer/custody/exchange laws consulted, no new equation or fit created. Graph-memory coverage is bounded to the index and DAG files in that query, not an unqueried service.

Beyond charter seeds, ps1/ps2 established stale-source README/compressor risk, separate custody-tree proof, new release tags and download/hash verification, and existing queued operator publication gates. The posted-PR memo established the approved public snapshot and fork branch. These changed the handoff: carry explicit stale-document blockers, preserve the source exactly, use a new planned tag, and do not claim whole-tree authority from code-file equality. The current hot-state retracts move41 and supersedes stale common-contract frontier prose. Current directives reinforced T4-only retained timing and the retraction. No training, decode, scorer or GPU work was started, so no scorer slot was consumed.

## Scope, custody and landing

Read-only boundaries: live `submissions/semantic_joint_ctxmix/`, all `upstream/`, both sealed `ddm_sj1_compose39_price` stores, and the three common-contract forbidden files. No `gh pr`, push, hosting upload, Modal/GPU dispatch, pointer mutation, stash, direct index mutation, or cleanup/delete was executed. No agent delegation. The only payload materialized is the copied archive, retained at the user-requested staging path with full hash/size; no measure-and-discard. Cache files were excluded from copying; none were deleted. All evidence paths are durable. The extracted raw auth JSON preserves original provider provenance strings, but evidence consumers use its durable local path, never a provider scratch path.

Metadata landing is via the serializer with post-edit hashes, no co-author trailer, both required tags, and review override only for .md/.json/.txt. The staging binaries and source tree are excluded. `LANDING_STATUS.json` records the actual serializer outcome; a fallback bundle is not a main-branch commit.

All six solver hooks are N/A: this packaging arm changes no sensitivity, Pareto constraint, bit allocator, dispatch implementation, posterior empirical anchor, or probe mechanism. Equations/DAG/DSL are untouched; the retained equation references do not create a new law. Publication/readiness consumers instead receive explicit `FIRE_ORDERS.json` (two QUEUED-WITH-A-FIRE-ORDER rows). The old fs1/fs2 publication packets are historical antecedents, not readiness proof for these bytes.

## LIVE-HYPOTHESES

- A separate release packet can retain the exact move40 decode while clearing packaging checks: all 44 executable-source hashes match the T4 manifest. It still needs the actual checker-accepted custody proof and policy/document fixes.

## DEAD-ENDS

- INSTANCE: using the byte-exact source copy as release-ready is closed by six strict failures plus stale README/MANIFEST. Do not publish on byte identity alone.
- INSTANCE: treating 0.14 as the precise score is closed by component recomputation; 0.13763861019288715 is the retained report-component value.
- FAMILY within this packet: importing move41/tc4 code is excluded by the rule118 retraction, regardless of an attractive byte count.

## NEXT_IF_RESUMED

- QUEUED-WITH-A-FIRE-ORDER; owner MAIN; consumer `.omx/research/ddm_pk1_20260910/FIRE_ORDERS.json`; fire on harvest: resolve the six strict failures and stale README/MANIFEST in a separate reviewed release packet.
- QUEUED-WITH-A-FIRE-ORDER; owner operator + MAIN; consumer `.omx/research/ddm_pk1_20260910/SWAP_COMMANDS.md`; fire after non-hosting clearance and explicit one-line confirm: host/fetchback, obtain strict PASS, then execute the approved swap/PR update.

composition S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600] (move 40) unchanged.

<!-- # FORMALIZATION_PENDING: staging/process memo — recomputes the retained move 40 score from its receipt components and records compliance refusals; no new measured row, no law -->
