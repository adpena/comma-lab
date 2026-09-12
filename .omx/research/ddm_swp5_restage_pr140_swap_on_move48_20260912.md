# ddm_swp5 — PR #140 swap packet re-staged on pointer move 48; publication still refused; one NEW custody item found

`[no-triality] [p0-ledger-ok]` · research_only=true · new_score_claim=false · $0

**The pointer did not move. The packet did — and the re-stage surfaced a real defect the move-47 stage could not
have seen.** The staged swap now sits on move 48's bytes (`d830edd3…`, 179,111 B, S 0.13638261682704697
`[contest-CUDA T4 n600]`) instead of move 47's 179,359 B. Strict `--contest-final` compliance is **89/93 PASS,
4 FAIL, rc 1** — two are the known open items, and the other two are ONE root cause in the dispatch ledger, not
in the packet. Nothing was pushed, hosted, PR-updated, or published; no dispatch was fired; no receiver file,
sealed tree, live PR tree, `/Volumes` path, or `upstream/` path was written.

Everything lives under `submissions/_staging_move48_pr140_swap/`: `shippable/` is the only candidate release tree
(51 source members + the generated public `report.txt`), `PR_BODY.md` and `SWAP_COMMANDS.md` are sibling drafts,
and `_packet/` holds internal evidence that must never be copied into a public PR.

## The new finding — one ledger row, two failing checks

`dispatch_claim_terminal_runtime_tree_sha_bound` FAILS. The move-48 terminal claim row binds
`runtime_tree_sha256=a8bc13af…`, which is the receipt's `runtime_digests.expected_runtime_tree` under
`tac.decode_wall_clock.measure_t4_runtime_digest`. The contest-final checker requires the terminal row to bind the
scored tree under the auth-eval provenance definition, `provenance.inflate_runtime_manifest.runtime_tree_sha256`
= `61d618ba…`. **Move 47's row bound the provenance value (`5170a798…`) and passed; move 48's row, written by the
first-measurement path's poller, binds the other definition.** This is a ledger custody difference, not a receiver
change and not a packet defect — the archive, the runtime tree and the score are all exactly what the receipt says.

I did not reason about whether some other pin satisfies both checks; I MEASURED it. Re-running the identical argv
with `--expected-runtime-tree-sha256 a8bc13af…` gives **89/93 again**, with the failure merely moving to
`auth_eval_runtime_tree_expected_match` (`a8bc13af…` is not among the auth-eval's runtime candidates). Both runs
are retained (`_packet/COMPLIANCE.json`, `_packet/ALT_DECODE_WALLCLOCK_PIN_COMPLIANCE.json`). **No value of that
flag reaches 91/93 on the current row.** The run of record keeps the canonical pin `61d618ba…`, the same definition
move 47 used.

`auth_eval_raw_promotion_policy_blockers_absent` FAILS as a **consequence, not an independent item**. The checker
records its CUDA-only policy review (`submission_policy_adjudication.v1`) only when every unresolved check lies
inside `POLICY_REVIEW_RELEASE_BLOCKERS` — which is exactly `{submission_runtime_imports_within_allowlist,
hosted_archive_manifest_supplied}` (`src/tac/auth_eval_schema.py:813`). A third unresolved check outside that set
withholds the review, so the raw blockers stay unresolved. Cure the claim row and both clear together: the
expected state is then 91/93 with exactly the two known open items.

I did not touch `.omx/state/active_lane_dispatch_claims.md`. Writing the row the checker wants would be
manufacturing the custody the checker exists to verify. It is MAIN's row and MAIN's fix, append-only.

## Verification

| Obligation | Result and scope |
|---|---|
| Copy identity | 51/51 source members byte-exact against the promoted tree (`_packet/STAGING_MANIFEST.json`); `__pycache__` and AppleDouble `._*` excluded; `inflate.sh` given the executable bit at staging only |
| Archive | 179,111 B, SHA-256 `d830edd371641e1968765ae6be27120a6c55a9b1ca4b3747158e603a43ef149c` — equal to the charter's pin, to the T4 receipt's `expected_archive_sha256`, and to the CPU adjudication's pin |
| Retained authority | call `fc-01M29A1CWQBBDT4XXTQ1K7TG88`, Tesla T4, `gpu_t4_match=true`, n600; no new evaluation was run |
| Recomputed score | **0.13638261682704697**, exact float equality to the receipt's `score_recomputed_from_components`, to its `canonical_score`, and to the charter |
| Components | d_seg 0.00010345, d_pose 4.59e-06, denominator 37,545,489 B; contributions 0.010345 / 0.00677495387438173 / 0.11926266295266524 |
| Arithmetic order | `rate = B/denominator; S = 100*seg + sqrt(10*pose) + 25*rate` (upstream/evaluate.py:65,92). At move 48's values the reassociated `25*B/denominator` order agrees **bit-for-bit**. No component differs from move 47 — only B. |
| Precision | eight-decimal report components plus exact archive bytes; retained worst-case score rounding bound 4.191067489650264e-06 |
| Retained timings | inflate 1023.262 s, evaluate 40.352 s (move 47: 989.298 s / 38.652 s) — this is the move's own measured t4_direct leg, not an inherited one |
| Source manifest | **49/49** `MANIFEST.sha256` entries verified on the staged tree (`_packet/SOURCE_MANIFEST_CHECK.json`); `README.md` is covered by that manifest, `archive.zip` is not |
| Runtime custody | staged portable runtime tree binds to the auth-eval manifest (`submission_runtime_tree_matches_auth_eval` PASS); the staged portable digest `0ff50f28…` equals the receipt's `expected_runtime_content_tree_sha256` |
| Dispatch linkage | lane `ddm_hpr1_comp_even_on_refit_move47_first_measurement_20260911`, job `ddm_hpr1_first_measurement_t4_run1_20260911`; terminal status `completed_contest_cuda_exact_eval_harvested` binding the exact archive sha — but the runtime-tree sha under the other digest definition (see above) |
| CPU axis | `fc-01M29B7NWPT3F1E26DYNJ50P81`, rc 1 after 10.273 s on move 48's own bytes; refusal receipt sha/bytes equal the adjudication's pin; all eight `contest_cpu_auth_eval_*` checks PASS on the first run (the refusal branch sets six of them); no CPU metric exists by design |
| Retracted symbols | 43 scanned Python files: **zero** executable identifiers/imports and zero `tc3`/`tc4` module filenames. Ten literal text occurrences remain (3 distinct: `compress.py:Tc4`, `compress.py:tC4E`, `runtime/rlc1_mixer.py:TC3`) in historical prose and legacy encoded text — the same honest scoping swp3/swp4 recorded, not a zero-text claim. |
| Swap procedure | 3 shell blocks parse under `bash -n` and 3 embedded Python heredocs parse under `ast.parse`; every mutation command is marked NOT RUN and none was executed |
| Boundaries | no tracked repo file modified by this arm before the memo/P0 landing (`git status` showed only MAIN's live `.omx/state` rows); the staging tree is untracked, as swp3's and swp4's are; `/Volumes` was read-only — a `find -newermt` sweep over `ddm_hpr1` shows zero files touched during this arm |

## Member-level diff

`_packet/TREE_DIFF.json` holds byte counts and shas for both comparisons.

**vs swp4's move-47 staging** — 52 vs 52 members, 48 identical, **4 changed, 0 added, 0 removed**:
`archive.zip` (179,111 ← 179,359 B), `inflate.py` (2,735 B both), `MANIFEST.sha256` (4,570 B both), and the
regenerated `report.txt` (584 B both). The two text members are minimal to the line: `diff` on `inflate.py` is
**exactly two lines** (`ARCHIVE_SHA256`, `ARCHIVE_BYTES`) and on `MANIFEST.sha256` **exactly one** (the rebinding
of that `inflate.py` hash). That is what "the only delta is the pointer" looks like at the member level, and it
independently corroborates the packet memo's claim that the receiver is unchanged (behaviour digest `9f6e7168…`).
The charter's framing — "only the `hpac` member and the archive differ" — is the archive-INTERNAL section view
(hpac 12,262 → 11,629 B, RLC1 118,511 → 118,896 B, net −248 B; MEASURED by ddm_hpr1, cited here, not re-measured
by me). At the tree level the archive's two dependent pins and the regenerated report necessarily move with it.

**vs the live PR tree** (`submissions/semantic_joint_ctxmix/`, read only) — 52 vs 40 members, 30 identical,
10 changed, 12 present only in the staged tree (`archive.zip`, `report.txt`, and ten runtime/cpr1 modules the live
tree predates), 0 present only in the live tree. The live tree tracks no `archive.zip`; the asset is hosted
separately, and the archive it describes is the 180,002-byte ancestor at 0.14797617125559104.

## Strict compliance

Argv, stdout, stderr and all 93 rows are retained in `_packet/COMPLIANCE_COMMAND.json`, `compliance.stdout.txt`,
`compliance.stderr.txt` and `COMPLIANCE.json`; the flag-by-flag state is in `_packet/CHECKLIST_93.json`. The two
known open items are unchanged from the reference form and **neither is closed by assumption**:

| Open check | Cure class / owner | Why it stays open |
|---|---|---|
| `submission_runtime_imports_within_allowlist` | receiver change — operator + MAIN | `runtime/rc3_shared_mixer.py` and `runtime/sm1_semantic_mixer.py` carry `experiments` fallback imports. A receiver edit needs fresh exact evidence and is the operator's decision; this arm does not touch receiver bytes. |
| `hosted_archive_manifest_supplied` | at publish — operator + MAIN | Strict contest-final requires `--hosted-archive-manifest-json`. Nothing is hosted, so the manifest cannot honestly exist yet. |
| `dispatch_claim_terminal_runtime_tree_sha_bound` | dispatch-ledger custody row — MAIN (**NEW at move 48**) | The terminal row binds the decode-wall-clock digest, not the scored auth-eval provenance digest. Append-only cure; no receiver or packet change. |
| `auth_eval_raw_promotion_policy_blockers_absent` | consequence of the row above — MAIN | The checker withholds its policy review while any unresolved check sits outside `POLICY_REVIEW_RELEASE_BLOCKERS`. Clears with the row. |

`_packet/BLOCKERS.json` carries all four with owner, disposition and trigger, plus the release debt: the packaged
`README.md` still describes the 180,002-byte ancestor and its score. It is carried byte-exact because
`MANIFEST.sha256` covers it, so refreshing it is a receiver-tree change. `PR_BODY.md` names that staleness in the
body so no stale number can reach a reader silently.

### Two things the reference form taught, and one it could not

swp4's r1 diagnostic cost seven strict checks by hand-transcribing the refusal receipt's `receiver.error` from the
source text instead of deriving it from the folded AST constant. I derived every receiver field from the staged
tree's own AST and then called the validator's own `staged_cpu_refusal_receiver` on the result before writing the
file. All eight `contest_cpu_auth_eval_*` checks passed on the **first** run; no r1 was needed.

The runtime-tree pin is not in my charter's pin list. Rather than inherit move 47's literal, I derived it from this
receipt's `contest_auth_eval.json` through the checker's own `_runtime_tree_candidates`. That derivation is what
exposed the ledger mismatch: an inherited literal would have failed silently in a way that looked like my error.

What the reference form could not teach: a stage that only ever re-runs the previous stage's passing argv learns
nothing when the *upstream producer* changes. Move 48 came through the first-measurement chain rather than the
move-47 chain, and the two chains' pollers write different runtime digests. The defect was invisible until a
packet was actually built on the new chain's output.

## Public disclosure hygiene

`_packet/PUBLIC_HYGIENE_SCAN.json` lists all 53 scanned public files (`PR_BODY.md` + `shippable/**`) and the ten
patterns applied. **Zero hits** for SSD/volume paths, local home paths, Modal volume paths, `.omx` internal state,
Modal call ids, private/Tailscale IPs, assistant attribution, credentials, and provider names. The checker's own
scan agrees (`public_hygiene.hits == []` over the same files). The retained T4 `report.txt` contained a
`/__modal/volumes/vo-…` path in its config block; the public `report.txt` is generated from the evaluation-results
block onward and is asserted free of `/__modal`, `/root` and `/Volumes` before it is written.

One class needs stating rather than suppressing: 117 internal `ddm_*` arm tokens appear inside the receiver source
(module names and comments) — the same 117 the live PR tree already ships. The staged tree adds no new exposure;
`PR_BODY.md` contains none. Internal paths and provider details live only in `_packet/` and `SWAP_COMMANDS.md`,
which are never published.

The disclosure sentence is unchanged and names no assistant:

> I used automated research and engineering tools extensively for the work behind this submission.

## Provenance pins

Every sha below was measured by this arm; `_packet/INPUT_BINDINGS.json` is the authority and nothing here was
retyped from a prior memo.

| Input | SHA-256 | Bytes |
|---|---|---:|
| t4 receipt: `/Volumes/VertigoDataTier/pact/ddm_hpr1_first_measurement/run1/MODAL_REMOTE_RESULT.json` | `9bd402caccc1648c3de44fe59b0364b0ddac0decf51fe2bf2163361685220328` | 242,361 |
| cpu receipt: `/Volumes/VertigoDataTier/pact/ddm_hpr1_first_measurement/cpu1/MODAL_REMOTE_RESULT.json` | `5fb95e8a638752e5fc12700ffb5b5d14c13058ae0f8c0460541ad66a039228bb` | 48,036 |
| cpu adjudication: `.omx/research/ddm_hpr1_packet_inputs_20260911/CPU_AXIS_ADJUDICATION_move48.json` | `39be4dfa24aa34403721e91877e65fffd6d3dedf05b7d1947c3b96ce9d08739c` | 1,066 |
| first-measurement authorization: `.omx/research/ddm_hpr1_20260911/v2/FIRST_MEASUREMENT_AUTHORIZATION.json` | `d02e27c4e14ec80c562c2642a7c4affe65f0c31585671383133c592bb6340a95` | 1,733 |
| seal: `.omx/research/ddm_hpr1_20260911/v2/SEAL_ddm_hpr1_comp_even_on_refit_contest_cuda_v3.json` | `993697eaa6352faf32829c659744094611e3212192a4a24bbfafa14eef01976a` | 17,500 |
| pointer memo (move 48) | `d184feb31415f1982a6fb66d13fe178bd4d99350432c346c880adc63e8b26afe` | 6,610 |
| swp4 memo (equals this charter's pin) | `8062f5c7d35ca6a5dcf2e8fd8666077a30987ee7bbc106eeb251a17160a56f1b` | 12,187 |
| this charter | `ba9bb12efed46e5a31052f20d798e6c6dcc60ab82a7e816918a066f99a18e355` | 4,562 |
| checker: `scripts/pre_submission_compliance_check.py` | `6e47f6c71a29eef85b144208faad0ef94045b5c40690fffd58ce71722074e343` | 142,782 |

The checker sha is byte-identical to the one swp4 measured, so the 93-item count and the 89-vs-91 comparison are
on the same instrument (cpx3's hardened checker), not on an inherited number.

## Boundaries honored

No push, no `gh` invocation, no hosting, no PR edit, no Modal or GPU dispatch, no scorer run, no receiver edit, no
write to any `/Volumes` path, no edit to `upstream/`, the live PR tree, sealed trees, another arm's directory, or
the dispatch-claims ledger. $0. The operator's one-line confirm is the only publish gate and was neither received
nor inferred. All six solver wire-in hooks are N/A for this prepare-only unit: no sensitivity, Pareto,
bit-allocator, dispatch, posterior or probe surface changed.

## NEXT_IF_RESUMED

- QUEUED-WITH-A-FIRE-ORDER; owner MAIN; consumer `.omx/state/active_lane_dispatch_claims.md`; trigger: immediately.
  Land an append-only terminal claim row for lane `ddm_hpr1_comp_even_on_refit_move47_first_measurement_20260911`
  binding the scored `runtime_tree_sha256=61d618ba…`, or reconcile the first-measurement poller's digest definition
  with the checker's. Then re-run the retained argv; expected 91/93.
- STANDING; owner MAIN; trigger: the next first-measurement pointer move. The two dispatch chains write different
  runtime digests into the terminal row. Whichever definition wins, both chains must write the same one, or every
  future packet on the first-measurement chain inherits this failure.
- QUEUED-WITH-A-FIRE-ORDER; owner operator; consumer `submissions/_staging_move48_pr140_swap/SWAP_COMMANDS.md`;
  trigger the one-line publish confirm: host, fetch back, emit the hosted manifest, require a full strict PASS,
  then swap.
- QUEUED-WITH-A-FIRE-ORDER; owner operator + MAIN; consumer `_packet/BLOCKERS.json`; trigger the receiver-hygiene
  decision: resolve the `experiments` fallback imports with fresh exact evidence, or record the accepted exception.
- QUEUED-WITH-A-FIRE-ORDER; owner MAIN; consumer `shippable/README.md` + `MANIFEST.sha256`; trigger a reviewed
  successor: refresh the README off move 48's numbers and rebind the manifest.
- STANDING; owner the next staging arm; trigger any further pointer move: re-stage rather than repoint this fixed
  move-48 packet; derive the refusal receiver fields from the staged tree's AST; and derive the runtime-tree pin
  from the new receipt's own auth-eval provenance rather than inheriting the previous move's literal.

composition S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600] (move 48)

## DEAD-ENDS

- INSTANCE: on the current move-48 claim row there is no `--expected-runtime-tree-sha256` value that satisfies both
  `dispatch_claim_terminal_runtime_tree_sha_bound` and `auth_eval_runtime_tree_expected_match`. Measured, both
  directions, both runs retained. Searching for a third value is closed.
- INSTANCE: `auth_eval_raw_promotion_policy_blockers_absent` cannot be cured on its own. It is a derived check
  whose input is the set of unresolved checks; treating it as an independent item wastes a cycle.
- INSTANCE: a staged packet cannot honestly satisfy `hosted_archive_manifest_supplied` before hosting. It is not a
  defect to cure, it is the publish gate expressed as a check.
- INSTANCE: refreshing the packaged README is not a free text edit — `MANIFEST.sha256` covers it, so it is a
  receiver-tree change with its own review debt.

<!-- # FORMALIZATION_PENDING: preparation and custody memo only; the retained exact row is recomputed, not re-measured, and no new empirical law or canonical equation is introduced -->
