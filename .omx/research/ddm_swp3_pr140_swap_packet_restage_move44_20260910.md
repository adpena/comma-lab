# ddm_swp3 — move44 PR #140 swap packet prepared, publication refused

`[no-triality] [p0-ledger-ok]` · research_only=true · new_score_claim=false

The pointer did not move. Preparation is complete; release is **REFUSED: 81/93 PASS, 12 FAIL, strict rc 1**, versus swp2/cpx2's 91/93. No publish, push, PR update, hosting, Modal dispatch, receiver edit, or new scorer measurement occurred.

Everything staged lives under `submissions/_staging_move44_pr140_swap/`: `shippable/` contains the candidate tree and public report, `PR_BODY.md` and `SWAP_COMMANDS.md` are sibling drafts, and `_packet/` holds internal evidence. Only `shippable/` is the receiver/upload tree. Internal evidence must never be copied into the public PR. The mandated CPU adjudication additionally lives at `.omx/research/ddm_rlc5_packet_inputs_20260910/CPU_AXIS_ADJUDICATION.json`.

## Verification

| Obligation | Result and scope |
|---|---|
| Copy identity | 51/51 source members byte-exact, plus generated public report; no receiver edits |
| Archive | 180,406 B; SHA-256 `04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e` |
| Retained authority | run5 call fc-01M26NNV3WR2XXXV914S8BTDR4; [contest-CUDA T4 n600]; no new evaluation |
| Recomputed score | **0.1372449041713402**, exact float equality to retained receipt and charter |
| Components | d_seg 0.00010345, d_pose 4.59e-06; denominator 37,545,489 B |
| Arithmetic order | `rate = B / denominator; S = 100*seg + sqrt(10*pose) + 25*rate`, matching upstream/evaluate.py:65,92. Reassociating `25*B/denominator` produces 0.13724490417134017; this is a floating-point ordering difference, not changed components. |
| Precision | Report eight-decimal components, not unrounded tensor precision; retained worst-case score rounding bound 4.191067489650264e-06 |
| Retained timings | inflate 1232.418725255 s; evaluate 50.7263101110002 s |
| Runtime code custody | 48/48 evaluated code hashes match; CPU command/provenance and CUDA share `fb1f6295a3527fdc07b3682d7e8ed42354e14e54dbbb4317deaa9f4f27ba21d4` |
| Source manifest | **49/49** source MANIFEST.sha256 entries verified, so swp2's stale MANIFEST defect is closed here |
| Rider census | All 51 census file hashes match; parsed 60-byte configuration = 1 version + 40 fitted weight bytes + 19 geometry bytes; geometry agrees with retained census |
| Receiver delta | Six changed/added code paths versus staged move43, plus MANIFEST = the seven-path receiver/custody delta (archive/report excluded) |
| Retracted symbols | Zero matching Python identifiers/imports/modules in 48 scanned code files; no retracted tc3/tc4 module filenames. Literal text absence is **not** true: 8 TC3 historical prose/error strings in RLC1 mixer and 2 case-insensitive tc4 substrings inside legacy encoded compressor text. No lane-predictor symbol found. |
| CPU refusal | call `fc-01M26Q7WV0YYF2HE2500ASKF8G`, receipt rc 1 after 8.053126651 s, exact move44 bytes; no CPU metrics |
| Boundaries | 119 previously enumerated protected/candidate/live files hash-unchanged; staged-index diff unchanged; source volumes read-only |
| Swap procedure | 3 shell blocks parse; embedded Python parses; mutation commands NOT RUN |

The rider check used the real staged archive's `read_residual_archive`, then `parse_config`, and retained the extracted config as lossless hex. `runtime/residual_archive.py` passes those counted bytes to `LaneMixer`; weights come from config[1:41], geometry from config[41:]. The CLEAR census is scoped to the RLC1 cure delta; it is not a fresh whole-family certification or a claim that all inherited arithmetic is integer. `RIDER_CENSUS_VERIFICATION.json`, `RUNTIME_FILE_IDENTITY.json`, `RETRACTED_CODE_SCAN.json`, `SOURCE_MANIFEST_CHECK.json`, and `MOVE43_DELTA.json` retain the bounded proof.

## Strict compliance and blockers

Full argv, stdout, stderr and 93 check rows are retained in `_packet/COMPLIANCE_COMMAND.json`, `compliance.stdout.txt`, `compliance.stderr.txt`, and `COMPLIANCE.json`. First diagnostic r1 (77/93) is retained separately: internal evidence was initially inside the submitted tree and a statement included its template heading. Splitting the shippable subtree and correcting the statement cured those staging defects. The terminal packet's portable runtime, archive, public hygiene, and statement checks pass. No gate was waived.

| Refusal group | Count | Cure class / owner |
|---|---:|---|
| CPU typed refusal checks | 6 | Checker declaration, MAIN: cpx2 accepts only `if not torch.cuda.is_available()`, but move44 genuinely uses `if not advisory_cpu and not torch.cuda.is_available()` at line 57. Its actual error/guard is recorded, never rewritten to impersonate the accepted form. Validators return `refusal_guard_does_not_raise_recorded_error` and `refusal_receiver_guard_not_bound`. |
| Raw promotion policy | 1 | Checker policy, MAIN: typed refusal rejection prevents genuine policy adjudication. Raw T4 flags remain unchanged. |
| Latest dispatch terminal status + archive SHA + runtime SHA | 3 | Dispatch custody, MAIN: newest run5 pointer row has an unaccepted status and omits both hashes; it shadows the older accepted harvest row. That older row pins prefire path-sensitive e3d237…; auth-eval records worker path-sensitive e44a61…. Reconcile with actual same-content custody, not invented equivalence. |
| Static import allowlist | 1 | Receiver decision, operator + MAIN: `experiments` fallbacks in rc3/sm1; unchanged inherited blocker. Any receiver edit needs fresh exact evidence. |
| Hosted archive manifest | 1 | At publish, operator + MAIN: host only after non-hosting clearance and explicit one-line confirm; retain fetchback and require strict PASS. |

Every exact failure name and details appears in `_packet/BLOCKERS.json`, with cure class, owner, disposition, consumer and trigger. **10 extra failures versus cpx2**, comprising the six CPU checks, raw-policy check and three dispatch checks. No assertion of 91/93 acceptance was carried from the charter.

Additional release defect: source README still describes the 180,002-byte ancestor and its score/reproduction narrative. Preserve this byte-exact staging reference, refresh README in a reviewed successor and rebind MANIFEST. The included legacy compressor is explicitly disclaimed as a move44 rebuilder. Source MANIFEST itself now verifies; do not blindly carry swp2's stale-manifest blocker forward. The previous bare-venv smoke is historical scoped evidence, not a new move44 full-runtime test.

## Disclosure

The draft preserves the exact swp2 sentence:

> I used automated research and engineering tools extensively for the work behind this submission.

The public draft declares `linux-nvidia-t4`, retains inherited vehicle credits, gives move44 components, and marks the proposed move44 release URL unhosted/unverified. It contains no named-assistant attribution or internal paths. This is prepare-only; the operator's explicit confirm remains mandatory before publishing.

## RECALL EVIDENCE

`_packet/RECALL_SEARCHES.json` contains exact argv, output path and rc for content recall across research memos/receipts, docs/design, canonical task ledger, local research-index/DAG surfaces, and the full canonical-equations export. Initial query was `PR.?140|runtime.tree.digest|CPU.axis.refusal|counted.rider`; corrected queries include `PR #140|PR140|ddm_swp3|rlc5` on `.omx/state/canonical_task_status.jsonl` and `submission.packet|runtime.tree|PR #140|PR140` on docs/index/DAG. The first guessed task-ledger path did not exist (rc 2); the actual canonical store was then searched. No graph service or external PR was queried.

Beyond charter seeds: ddm_ps1's task row already records the same missing terminal ledger hash class; ddm_pr9 and ddm_pr14 require manifest-only downstream rebinding; the real move44 receiver source exposes the advisory-CPU guard missed by the simple cpx2 reference form. These changed the plan: retain new checker/ledger refusals with concrete consumers, verify the current MANIFEST rather than inheriting its stale predecessor status, and scope retracted-code absence to executable names rather than false zero text hits. Docs template establishes release checks but its old frontier numbers are not current. Canonical equation recall preserves score/rate provenance and introduces no new law. All current dated directive files were inspected; move41 retraction and no authority transfer remain binding. No relevant personal-memory hit was found for `swp3|pr140|swap_packet`; no memory-derived factual claim used.

## Provenance pins and landing

| Input | SHA-256 | Bytes |
|---|---|---:|
| t4: `/Volumes/VertigoDataTier/pact/ddm_rlc5_first_measurement_run5/MODAL_REMOTE_RESULT.json` | `b3ec1dd9dbe43debeeeeae075a49c1cd8b2d23507a7fc317e9884590b9c8fe46` | 242556 |
| cpu: `/Volumes/VertigoDataTier/pact/ddm_rlc5_move44_cpu_20260910/MODAL_REMOTE_RESULT.json` | `ef39b9b7aa173f1acf3730d57b1e2dc1c661bfef381c693923c6c7480c612124` | 48262 |
| census: `.omx/research/ddm_rlc5_20260910/LITERAL_CENSUS.json` | `e94b5f4b602d9be1eab8096d45647a368c536a4704548b1d8f53b223bd6a08bd` | 13410 |
| swp2: `.omx/research/ddm_swp2_pr140_swap_packet_restage_move43_20260910.md` | `8e0a3ebe66895315767173b224a5ee70db1cc80c14c9208f57e3a5b4e21819c4` | 15592 |
| pointer_memo: `.omx/research/ddm_rlc5_counted_rider_rebase_move43_first_measurement_20260910_pointer_move_44_20260910.md` | `f7638e1e171e0d19309f0d0b2f2f9f0acf9b5c70c5fb87f14bc029dfbaeaba8b` | 6343 |

The charter's pointer commit is 99625f32f. All evidence references are durable; historical remote paths inside immutable receipts remain original provenance, not local evidence locations. The common contract's stale frontier prose is superseded by the explicit move44 charter and current hot-state pointer. No upstream, source-volume, live-PR, common-contract forbidden-file, or receiver write occurred. No delegation, no stash, no direct index manipulation, no scorer slot or paid dispatch.

All six solver hooks are N/A for this prepare-only unit: no sensitivity, Pareto, bit allocator, dispatch implementation, posterior numerical anchor, or probe mechanism changes. `research_only=true`; no new empirical law or pointer promotion.

Serializer is invoked LAST, once, with post-edit hashes and both required tags; only generated manifests/text are selected, never archive bytes or receiver source. Non-Python override is charter-authorized. Fallback is explicitly routed inside this packet to honor the no-/Volumes-writes boundary. `LANDING_STATUS.json` records the actual rc and bundle custody; rc17 means MAIN must land, not main-branch success. Checkpoint identity is `ddm_swp3`; COMPLETE means preparation and blocker handoff, not release clearance.

## NEXT_IF_RESUMED

- QUEUED-WITH-A-FIRE-ORDER; owner MAIN/compliance-policy owner; consumer `submissions/_staging_move44_pr140_swap/_packet/BLOCKERS.json`; trigger packet harvest: support or adjudicate the actual advisory-CPU guard using this bound refusal, then rerun policy checks without fabricated metrics.
- QUEUED-WITH-A-FIRE-ORDER; owner MAIN/dispatch ledger owner; consumer `.omx/state/active_lane_dispatch_claims.md`; trigger packet harvest: reconcile latest run5 status and archive/runtime custody from retained receipts, then rerun strict checks.
- QUEUED-WITH-A-FIRE-ORDER; owner operator + MAIN; consumer `submissions/_staging_move44_pr140_swap/_packet/BLOCKERS.json`; trigger packet harvest and receiver decision: resolve import hygiene and stale README; fresh exact evidence for receiver edits.
- QUEUED-WITH-A-FIRE-ORDER; owner operator + MAIN; consumer `submissions/_staging_move44_pr140_swap/SWAP_COMMANDS.md`; trigger non-hosting clearance plus explicit confirm: host/fetchback, create hosted manifest, require strict PASS, then swap and verify; restage if pointer changes.
- QUEUED-WITH-A-FIRE-ORDER; owner MAIN; consumer `submissions/_staging_move44_pr140_swap/_packet/LANDING_STATUS.json`; trigger verified serializer rc17 bundle: land exact metadata and verify all hashes.

composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)

## LIVE-HYPOTHESES

- A narrowly reviewed typed-refusal extension could accept the unchanged move44 receiver: real CPU failure, CUDA/CPU code-file digest, and archive binding agree; the unsupported conjunction is the terminal validator difference. This is untested and grants no waiver.
- Correctly reconciled latest dispatch custody could clear three checks without receiver changes: run5's retained receipt and older harvested row supply the facts, while the newest status row omits them. Checker acceptance remains untested.

## DEAD-ENDS

- INSTANCE: the cpx2 simple-guard path does not accept move44's real advisory-CPU conjunction. Repeating it unchanged cannot produce 91/93.
- INSTANCE: mixing internal evidence with the submitted receiver changes its digest and leaks private text into public scans. The packet now uses a separate shippable subtree.
- INSTANCE: claiming zero tc3/tc4 text matches is false; historical strings remain. The executable-name scan and census support the narrower claim.
- INSTANCE: swp2's stale MANIFEST assumption does not transfer: move44 verifies 49/49. Its README is still stale.
- INSTANCE: publishing this packet remains refused by 12 strict checks and stale README; a retained T4 score is not release authorization.

<!-- # FORMALIZATION_PENDING: preparation/custody memo only; retained exact row recomputed, no new score or scientific law -->
