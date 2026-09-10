# ddm_swp2 — move43 PR #140 swap packet staged; nothing published (2026-09-10)

`[no-triality] [p0-ledger-ok]` · research_only=true · new_score_claim=false

The exact pointer did not move. The move43 byte-exact staging packet is complete as a preparation artifact; publication is **REFUSED: 85 PASS / 8 FAIL / 93 strict checks, rc 1**. The CPU declaration/refusal form explicitly requested by the charter is rejected by the current checker. This is a concrete handoff, not a release-ready one-command confirm. Nothing was pushed, hosted, dispatched, or edited on PR #140.

## Verification

| Obligation | Measured or retained result |
|---|---|
| Staging | `submissions/_staging_move43_pr140_swap/`: 48/48 source files copied byte-exact, plus generated public-safe report; archive retained |
| Archive | 180,466 B; SHA-256 `7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e` |
| T4 components, retained | d_seg 0.00010345; d_pose 4.59e-06; 600 samples; denominator 37,545,489 B |
| Recomputed S | `100*d_seg + sqrt(10*d_pose) + 25*B/37_545_489` = **0.1372848557085275**, exact Python-float equality to charter and retained receipt |
| Precision | Evaluator eight-decimal printed components, not unrounded tensor values; retained worst-case score rounding bound 4.191067489650264e-06 |
| T4 timing, retained | inflate 1089.633945164 s; evaluate 42.740094623000005 s. No new timing or score run |
| Runtime files | 45/45 evaluated code members match retained T4 per-file hashes; `compress.py` retained, unlike pk1 |
| Move40 comparison | 44/45 evaluated code files equal move40; only `inflate.py` differs. Legacy compressor, including encoded literal, equals move40 |
| Bare venv | rc 0; normal rc3/sm1/ihs2 package imports; no `experiments` available or attempted. 2/2 forced-import-error observer controls detect the actual fallback |
| Retracted receiver | No tc3/tc4/lane-predictor receiver text, identifiers or filenames found. One all-tree case-insensitive hit: `compress.py:1172`, `tC4` inside encoded legacy `residual_pre_dx2.py` source, not receiver code |
| Source manifest | 45/46 listed hashes pass; `inflate.py` is stale. Preserved verbatim in this byte-exact staging copy |
| Source README | Ancestor move23/AFR1 numbers and reproduction narrative; publication defect carried separately from strict checks |
| Permissions | Staged `inflate.sh` set to 0755; bytes and source mode unchanged |
| Protected boundaries | 84 source and 64 live files hash-unchanged; upstream evaluator and shared staged index unchanged |
| Swap procedure | All shell blocks and embedded Python syntax checked; copy/hosting/commit/push/PR commands NOT RUN |

Evidence bundle: `.omx/research/ddm_swp2_20260910/`. `STAGING_MANIFEST.json` identifies every copied file. `SCORE_RECOMPUTATION.json`, `RUNTIME_FILE_IDENTITY.json`, `MOVE40_CODE_COMPARISON.json`, and `BOUNDARY_VERIFICATION.json` support the table. No receiver code was edited; no scorer slot was consumed.

## Strict compliance and comparison to pk1

pk1 was **70/76 PASS, 6 FAIL**. swp2 is **85/93 PASS, 8 FAIL**. The denominator grew because supplying the CPU refusal and archive manifest runs additional checks; raw pass counts are not a same-denominator improvement claim. Runtime-tree identity and archive-manifest checks now pass. Five dispatch-linkage checks pass using the actual job `ddm_sj1_pass6_t4_20260910`, not the lane ID as an assumed job ID. The initial wrong-job invocation (80/93) is retained as `r1_COMPLIANCE*`; terminal output is `COMPLIANCE*`.

| Remaining strict refusal | Cure class | Exact reason / owner |
|---|---|---|
| `auth_eval_raw_promotion_policy_blockers_absent` | declaration | Retain raw receipt unchanged and produce a genuine policy adjudication after required packet checks; do not flip raw receipt flags. Owner: MAIN / policy adjudicator |
| `contest_cpu_auth_eval_score_parseable` | declaration | Add a real checker-supported CUDA-only refusal adjudication path bound to these bytes and receiver; preserve absence of CPU metrics. Existing charter form is refused; do not fabricate a CPU score. Owner: MAIN / compliance-policy owner |
| `contest_cpu_auth_eval_archive_sha_matches` | declaration | Add a real checker-supported CUDA-only refusal adjudication path bound to these bytes and receiver; preserve absence of CPU metrics. Existing charter form is refused; do not fabricate a CPU score. Owner: MAIN / compliance-policy owner |
| `contest_cpu_auth_eval_archive_size_matches` | declaration | Add a real checker-supported CUDA-only refusal adjudication path bound to these bytes and receiver; preserve absence of CPU metrics. Existing charter form is refused; do not fabricate a CPU score. Owner: MAIN / compliance-policy owner |
| `contest_cpu_auth_eval_schema_metric_consistency` | declaration | Add a real checker-supported CUDA-only refusal adjudication path bound to these bytes and receiver; preserve absence of CPU metrics. Existing charter form is refused; do not fabricate a CPU score. Owner: MAIN / compliance-policy owner |
| `contest_cpu_auth_eval_runtime_tree_recorded` | declaration | Add a real checker-supported CUDA-only refusal adjudication path bound to these bytes and receiver; preserve absence of CPU metrics. Existing charter form is refused; do not fabricate a CPU score. Owner: MAIN / compliance-policy owner |
| `submission_runtime_imports_within_allowlist` | receiver change | Operator decides unchanged receiver/static policy adjudication versus removal of fallback imports; any receiver edit requires fresh exact T4 row. Not applied. Owner: operator + MAIN |
| `hosted_archive_manifest_supplied` | manifest | After non-hosting clearance and explicit confirm, host a new move43 release, retain download, hash it and generate the real hosted manifest; then strict PASS required before PR swap. Owner: operator + MAIN |

All exact terminal details are in `BLOCKERS.json` and full checker stdout/stderr/JSON plus argv are retained. No gate, flag, receipt blocker, or metric was waived or forged. The specified `--submission-score-axis contest_cuda --contest-cpu-auth-eval-json .omx/research/ddm_rp1_round2_packet_inputs_20260910/CPU_AXIS_ADJUDICATION.json` was used verbatim. That record describes refusal on move42's same receiver family, not a CPU score on move43. Its five failures are missing score, missing archive SHA, missing archive size, missing metric schema, and missing runtime digest. The checker does not currently interpret that declaration as satisfying these fields. Do not invent CPU metrics to make it green.

The stale README and stale `MANIFEST.sha256` are additional release defects, outside the eight strict reds. Refresh them in a separate reviewed successor and rebind changed custody definitions; preserve the sealed reference and this byte-exact staging copy. The old `compress.py` is included for evaluated-tree custody and explicitly disclaimed as a move43 rebuilder.

## Import-hygiene operator decision

Exact unchanged lines: `runtime/rc3_shared_mixer.py:14-17` (fallback import at 17), and `runtime/sm1_semantic_mixer.py:13-18` (fallback imports at 17-18). Both catch `ImportError` from ordinary package-relative imports and reach into `experiments`.

MEASURED [macOS-CPU import diagnostic]: a newly created isolated venv with only locally installed NumPy 1.26.4, no system-site packages, Python `-I -B`, and only the staged tree added to sys.path imported rc3, sm1, and their real ihs2 consumer successfully. The import observer recorded zero fallback attempts and first verified that `experiments` could not be found. Two separate controls forced the relevant relative import to fail and detected one `experiments` attempt each. Receipts: `BARE_VENV_SMOKE.json`, adjacent stdout/stderr, and `IMPORT_OBSERVER_CONTROLS.json`.

Scope is normal package import resolution of the two flagged modules and ihs2 on this host. This is not a full `inflate.sh`/Torch import, Linux dependency-bootstrap, numerical decode, or universal unreachable-code proof. Broken package imports can still enter the fallback, as the controls show. The packet's answer is **fallback not reached in the shipped-package import smoke**, while the static checker remains red. Operator decision: preserve receiver and obtain a genuine policy adjudication, or remove the fallbacks and obtain fresh exact T4 evidence on that changed receiver. This arm applies neither change nor waiver.

The disposable venv was routed to `/Volumes/VertigoDataTier/pact/ddm_swp2/retained/`, certified with command, wheel hash, full environment file/tree hashes and bytes, then removed only after successful smoke and controls. Certificates and outcomes remain; no candidate payload was discarded.

## Runtime digest definitions remain distinct

| Producer / scope | Digest |
|---|---|
| `tac.candidate_seal.measure_runtime_digest`, source 48 files | `68fae56a0709ebc435e1fb99019beb16d97e2a30948a44a33866f999e6032a8c` |
| Same helper, stage 49 files including added report | `59820a4549b3cdb18bbbe55b58d02d746181fe99d017f73d10a6f225d959750c` |
| Retained auth-eval runtime manifest, 45 code files | `a726739a52452f8824c4eb9f98322f927b8b5d2d4c6ac1717911f61b841a9803` |
| Checker staged tree | `750fc1e4e099db72d138d67db2e9d4028786531b4fbbadfe1587a5bfc8ccb8d7` |
| Checker portable code identity, equal on stage and auth manifest | `5863425ffe98033df7be70716563a7583473981d69349f7d49533a14764027ab` |

`DIGEST_DEFINITIONS.json` retains the helper's source hash and every exact runtime check detail. Retaining `compress.py` closes pk1's omitted-member mismatch through the checker's existing portable definition; no equivalence proof was invented. General digest-definition consolidation remains folded into the existing MAIN/r9m obligation, not claimed complete here.

## Disclosure before / after

Before, quoted from pk1's retained local posted-body snapshot; not freshly checked on the live PR:

> I used coding agents (Claude as orchestrator of Codex subagents) extensively as research and engineering tools for the work behind this submission.

After, neutral draft in `PR_BODY.md`:

> I used automated research and engineering tools extensively for the work behind this submission.

The public-facing draft preserves inherited vehicle credits, declares `linux-nvidia-t4`, uses move43's actual report block, and labels the new release URL unhosted/unverified. No named assistant attribution appears in the proposed public text. It does not claim that the old compressor rebuilds the current bytes. Publication and the disclosure change still require the operator's one-line confirm.

## RECALL EVIDENCE

`RECALL_SEARCHES.json` records exact content searches and output paths: research Markdown for `PR #140|PR140|bare.venv|runtime.tree.digest|p0_swap_procedure` (197 matching lines); docs/design surface for `PR #140|PR140|submission.packet|runtime.tree.digest` (45 lines); canonical task ledger for `PR #140|PR140|ddm_swp2|r9m` (4 lines); named research-index/DAG files for `PR #140|PR140|runtime.tree.digest` (0 lines). Full canonical-equations output is retained in `equations_recall.json`; score/exchange and runtime-custody laws were consulted without adding a law or empirical fit. Graph scope is local index/DAG content, not an unqueried service.

Beyond the charter seeds: ps1's update packet found source README/compressor regressions, so byte identity alone cannot justify publication; pr9's manifest condition establishes that a manifest-only refresh still needs downstream digest rebinding; older dependency-closure receipts distinguish bare-venv import success from full Linux bootstrap. These changed this handoff: keep source documents verbatim as named defects, retain the evaluated compressor member to prove portable runtime identity, and scope the import smoke precisely. Current directives and hot-state exclude retracted move41 code; the specific move43 charter supersedes stale frontier prose in the common contract. No external live leaderboard or PR query was needed or performed.

## Scope and landing

Read-only: live `submissions/semantic_joint_ctxmix/`, all `upstream/`, sealed `/Volumes/*/pact/ddm_sj1_pass6/`, and the common contract's three forbidden files. No `gh pr`, push, upload, Modal/GPU dispatch, scorer, pointer mutation, stash, direct staged-index mutation, or receiver edit occurred. No delegation. The only newly staged score payload is the retained archive copy. Public files are limited to the intended draft/report; internal logs and provider provenance are not a public release surface.

All six solver hooks are N/A: this preparation unit changes no sensitivity, Pareto constraint, bit allocator, dispatch implementation, posterior numerical anchor, or probe mechanism. `research_only=true`; no new equation or DSL/DAG law. Operational consumers receive `FIRE_ORDERS.json` with explicit dispositions, owners and triggers.

The serializer is invoked LAST, once, with post-edit SHA-256 for each metadata/text file, both required tags and `--no-co-author`. No staging binary or Python source is in that landing, so the charter-permitted non-Python review override is applicable. `LANDING_STATUS.json` records the actual exit code and fallback custody after the attempt. rc17 is not a main-branch commit: MAIN lands the verified bundle. The shared index is checked unchanged. Checkpoint identity: `ddm_swp2`.

## NEXT_IF_RESUMED

- QUEUED-WITH-A-FIRE-ORDER; owner MAIN with operator for receiver choice; consumer `.omx/research/ddm_swp2_20260910/BLOCKERS.json`; trigger handoff harvest: resolve declaration/import policy and refresh README/MANIFEST in a reviewed successor, with fresh exact evidence for any receiver edit.
- QUEUED-WITH-A-FIRE-ORDER; owner MAIN; consumer `.omx/research/ddm_swp2_20260910/LANDING_STATUS.json`; trigger serializer rc17 with verified bundle: land exact metadata and verify it.
- QUEUED-WITH-A-FIRE-ORDER; owner operator + MAIN; consumer `.omx/research/ddm_swp2_20260910/SWAP_COMMANDS.md`; trigger non-hosting clearance plus explicit confirm on the reviewed bytes/body: host/fetchback, produce hosted manifest, require strict PASS, then swap/push/update and verify. Restage first if the shipping pointer changes.

composition S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600] (move 43)

## LIVE-HYPOTHESES

- An unchanged-receiver release may clear after genuine declaration and import-policy adjudication: all 45 evaluated code hashes match and normal package imports avoid the fallbacks. Checker acceptance is untested; this is not a waiver.

## DEAD-ENDS

- INSTANCE: passing the existing CPU refusal JSON as an auth-eval score record does not clear the current checker; five exact failures remain. Do not repeat it unchanged expecting PASS.
- INSTANCE: publishing this byte-exact staging copy is blocked by eight strict failures plus stale README/MANIFEST. Hash identity alone is insufficient.
- INSTANCE: excluding the evaluated compressor member caused pk1's runtime mismatch; retaining it here clears portable runtime identity. Do not reintroduce that omission without a real custody proof.
- INSTANCE: treating the import smoke as full Linux/T4 decode or a universal unreachable-code proof is invalid; its scope is normal package imports on this host.
- FORMULATION boundary for this packet: retracted move41/tc4 receiver code and unapproved publication are excluded by charter, regardless of their potential score.

<!-- # FORMALIZATION_PENDING: preparation/process memo; recomputes retained move43 report components and records strict refusals; no new measured score row or law -->
