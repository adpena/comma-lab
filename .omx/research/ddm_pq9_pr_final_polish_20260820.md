# ddm_pq9 — PR final polish, final-minus-operator-line

Date: 2026-08-20  
Owner: `ddm_pq9`  
Score claim: no new score  
Publish state: prepared, not published

## Outcome

The generation-5 packet is now final-minus-operator-line for the currently measured jg5 object.
The commit-pinned hosted archive is wired into the PR source draft, the RC2 swap and URL-re-pin
obligations are explicit, the LLM disclosure is an operator-only scaffold, all shipping-runtime
figures are held behind fresh RC2 receipts, and the reviewer appendix is executable from the
submission directory.

This arm did not publish, push, open a pull request, run a scorer, fire Modal, edit `upstream/`, or
edit `submissions/robust_current/jg5_sub015_runtime/`.

## Per-item disposition

| Work item | What changed | Receipt / source row | Verification |
|---|---|---|---|
| 1. Hosting and swap | `PR_BODY_DRAFT.md` now carries the exact commit-pinned raw URL and says mechanics were verified by MAIN. It refuses URL transfer if RC2 replaces jg5. `SWAP_PROCEDURE.md` step `4A REPUBLISH_AND_REPIN_HOSTED_ARCHIVE` requires operator authorization, an exact-byte commit and push, a new 40-character pin, a fresh HTTP-200 download, SHA-256 and byte-count equality, and a freeze receipt. | Charter MAIN receipt: HTTP 200 and downloaded SHA-256 `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e`. Swap source is `SWAP_PROCEDURE.md` step 4A and refusal conditions. | Local staged archive is 180,625 B with the same SHA. The sandbox could not resolve `raw.githubusercontent.com`, so this arm did not claim a second network measurement; it preserved MAIN's supplied verified-download receipt and made the reviewer command perform that comparison afresh. |
| 2. pq5 + pq6 folds | The public body retains pq5's corrected denominator/tone/limits treatment and pq6's terse method, headroom, joint-repair, and priced-but-unbuilt folds. No new mechanism claim was invented. | pq5 provenance is commit `4226017206e5e276d65c252b165d5433c33e2caa`; the named pq5 directory is empty at the current tree, so no nonexistent memo row is cited. pq6 sources: `EVIDENCE.md:18-26` shipped row, `:32-56` headroom, `:62-83` directions, `:113-141` method, `:148-186` repair; routing is `MERGE_PLAN.md:42-50`. | Re-read the public folds against those rows. The PR report block remains byte-identical to `REPORT_PUBLIC.txt`. |
| 3. LLM disclosure | Added a top-level section explicitly labelled `DRAFT ONLY; OPERATOR MUST REWRITE IN THEIR OWN WORDS`. It scaffolds actual tool roles, points to the NO-FAKE #7 borrowed-substrate accounting, and states that mechanism accounting cannot answer code authorship or the policy's “most of the code” test. | `PR_BODY_DRAFT.md`, LLM-disclosure section; `BORROWED_SUBSTRATE_ACCOUNTING.md` is the measured accounting source. | The draft still says it is source material, not text to paste. The final policy judgment remains operator-owned. |
| 4. Runtime figures | Removed shipping declarations that transferred jg5, instrumented-tree, or local-receiver timing. `REPORT_PUBLIC.txt`, its PR mirror, the PR host/build/GPU fields, and `README_PUBLIC.md` now mark composed-object CUDA timing, CPU outcome, and budget verdict `GATED-ON-RC2`, naming the exact fresh receipt fields required. | `ddm_rr8_t4_wallclock_verdict_20260820.md` fire order item 4; `ddm_rc2_composed_clean_decode_seal_20260820.md` timing boundary and authority boundary. | Searches across all three public documents found no `464.559`, `1,283`, `4,369.6`, `1,419.904`, `1,484.803`, or `15.3%` transfer. Object B's local full-n600 timings remain explicitly contended `[macOS-CPU advisory]`, never shipping declarations. |
| 5. Reviewer appendix | Added exact commands for `MANIFEST.sha256`, local-vs-hosted archive SHA, T4-receipt formula recomputation and archive/runtime identity assertions, then the contest evaluator. Corrected the evaluator path to run from the declared submission-directory working directory. | Public `How to verify` sections in `PR_BODY_DRAFT.md` and `README_PUBLIC.md`; T4 custody receipt `gen5_receipts/contest_auth_eval.json`. | `shasum -a 256 -c MANIFEST.sha256` returned 33/33 `OK`; local formula recomputed `0.14839100138338618`; archive SHA is `f3bce5d2…8acb7e`; receipt and measured-byte stager both return runtime tree `2103073d…cb9b`. Source README/report copies equal their staged copies byte-for-byte. |
| 6. Fresh review | One genuinely fresh review read all three public documents and checked their commands, denominators, custody statements, mirrored block, stale figures, private paths, archive/runtime identities, and staged copies. It found and fixed three real defects: the evaluator command used the wrong working-directory model; the README falsely said receipts were inside the public packet; and README/PR body called the 33-row evaluated runtime manifest a 34-file runtime tree. | This memo, “Fresh review verdict” below. The third finding was resolved as “33 manifest runtime files plus the exact archive”, matching `STAGING_RECEIPT.json`. | This was a finding pass, not a clean pass. Counter at exit: **0/5**. |

## Packet and identity verification

- Active staged packet: `/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen5_jg5_waterfill`.
- Canonical stager: `tools/stage_contest_submission_packet.py`; verdict
  `STAGED_TREE_PROVED_IDENTICAL_TO_EVALUATED_TREE`.
- Authority runtime rows: **33/33** verified from freshly measured staged bytes.
- Re-derived runtime tree: `2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b`, unchanged after every document edit.
- Archive: **180,625 B**, SHA-256
  `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e`.
- `MANIFEST.sha256`: **33/33 OK**.
- Final census: **45 declared** = 33 runtime + 12 non-runtime, **0 undeclared**, **2 declared-but-absent** (`GENERATION_RECEIPT.json`, `RECEIVER_PARSEBACK.json`), `CENSUS_CLEAN`; prep and receipt directories also clean. AppleDouble sidecars: **0** after final purge.
- Public-document private-surface search: no `/Volumes/`, `/Users/adpena`, private-key marker, or token-pattern hit in the PR body, README, or report.
- `git diff --check`: clean.

The two pre-review staged trees were moved intact to retained sibling generation directories rather
than overwritten. They are historical doc-state backups, not active candidates.

## Fresh review verdict

The charter's prior-law prediction was supported: the pass found three material truth/execution
defects. All three were repaired and the packet was canonically restaged after each repair batch.
Because the pass had findings, it does not count as clean even though the repaired surfaces now pass
the focused rechecks.

**Five-pass counter at exit: 0/5.** No successor may call this `1/5` without a new independent pass
over the final frozen candidate and all three final documents.

## Strict compliance receipt

Fresh clean-surface receipt:
`/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen5_receipts/pre_submission_compliance.gen5.pq9.r5.json`.
It records **81 GREEN / 6 RED of 87**, scans 44 public files, and uses an explicit substantive
competitive statement rather than misusing the operator-only PR source draft as the statement.

The six reds remain typed rather than cosmetically cleared:

1. `auth_eval_raw_promotion_policy_blockers_absent` — the raw authority receipt still carries its
   expected pre-compliance policy blockers.
2. `contest_cpu_auth_eval_exists` — no current/final composed contest-CPU receipt exists.
3. `submission_runtime_has_no_network_install_or_local_paths` — the declared Brotli bootstrap may
   use the network at decode time; this is publicly disclosed.
4. `hosted_archive_manifest_supplied` — the URL is verified and public, but no checker-format hosted
   archive manifest was supplied.
5. `submission_runtime_imports_within_allowlist` — the current checker world treats declared
   non-runtime `compress.py` as runtime and sees its `tac` import.
6. `submission_runtime_tree_matches_auth_eval` — the current checker walk produces candidates that
   do not equal the authority's 33-row enumerated tree, despite the canonical stager independently
   re-deriving all 33 measured rows to the exact authority hash.

Items 5–6 are new relative to the earlier 83/87 receipt. The checker script itself remains SHA-256
`c4145263…fad9`, but its helper `src/tac/preflight.py` moved from `2f70d4d9…b885` to
`a6184195…d8e3` and the canonical pointer world also refreshed. That is evidence of instrument/world
drift, not permission to waive either red. MAIN's compliance owner must reconcile the current
runtime-walk semantics with the stager's declared non-runtime set or produce a valid equivalence
proof on the final chosen object.

## Remaining operator-only actions

The operator-only list is shorter: current jg5 hosting is closed and is no longer an owed action.

1. Write the final PR description and LLM-policy answer in the operator's own words after inspecting
   the final submitted diff and authorship history, including a direct “most of the code” judgment.
2. Authorize and perform the final publish/open-PR action once the selected object, compliance
   disposition, and review counter satisfy the operator's release bar.
3. Conditional only: if RC2 becomes the selected shipping object, authorize its public-repository
   push so MAIN can execute swap step 4A and re-pin every URL to the new commit.

The RC2 paired authority fire, compliance-instrument reconciliation, hosted-manifest construction,
and independent review passes are not operator-only prose decisions; they remain owned engineering
or routing work.

## RECALL EVIDENCE

The recall pass searched the full `.omx/research/` corpus, arm final messages, design docs and live
state using the content queries `PR final|fresh eyes|review counter|hosted archive|re-pin|LLM
disclosure|runtime figure|cross-regime|GATED-ON-RC2`. It also ran
`tools/list_canonical_equations.py --json` and filtered for `score|rate|pose|runtime|archive|waterfill`,
then searched `CANONICAL_RESEARCH_INDEX*` and the `sub015_DAG_*` FEED blocks for
`PR_BODY_DRAFT|submission packet|hosted archive|runtime budget|review counter|jg5|RC2`.

Beyond the charter seeds, recall found RC2's completed two-run full-n600 real-receiver proof and
axis seals. That changed the wording from “decode pending” to “locally decode-proven and sealed,
authority pending,” without promoting the rate-only score hypothesis. It also found the current
checker-helper and pointer world had moved since the older 83/87 receipt; that forced a fresh strict
receipt and preserved two new reds instead of citing the stale count. The DAG confirmed that the
live exact row is jg5, superseding the stale frontier paragraph in the common contract. No canonical
equation altered the score arithmetic: the exact contest formula remains the only score recompute.

The named pq5 source directory was empty in the current tree. A bounded Git-object lookup found its
actual landing as commit `4226017206e5e276d65c252b165d5433c33e2caa`, whose diff records the 94.5%
denominator correction, qualified headroom line, verification block and consolidated limits. That
commit—not a fabricated memo row—was used for the pq5 fold verification.

## Measurement boundary

This arm measured packet bytes, hashes, manifest closure, census denominators, formula arithmetic,
document equality, checker outcomes and review findings. It did **not** measure a new score, a new
runtime wall, RC2 on contest-CUDA, RC2 on contest-CPU, or a new hosted download. The only authority
score remains the retained jg5 row. The reviewer appendix's network leg could not execute in this
sandbox because DNS resolution was unavailable; MAIN's charter-supplied HTTP-200/downloaded-SHA
receipt is the hosting evidence carried into the draft.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN scorer-lane router; consumer store: `/Volumes/APDataStore/pact/ddm_rc2/{t4_row_r1,cpu_row_r1}/` then `.omx/state/main_hot_state.md`; fire trigger: no active full-n600 scorer/Modal claim, both RC2 axis seals remain valid on unchanged bytes, and their pointer baselines remain unchanged; action: execute RC2's documented sequential CUDA-then-CPU pair and retain every output.
- **CONDITIONAL-SWAP** — owner: operator + MAIN packet owner; consumer store: active generation packet, `PR_BODY_DRAFT.md`, hosted-archive manifest and final freeze receipt; fire trigger: fresh RC2 authority receipts make RC2 the selected exact shipping object and the operator authorizes the public push; action: run `SWAP_PROCEDURE.md` including step 4A, then re-stage and re-buy every identity/compliance receipt.
- **QUEUED-FOR-APPARATUS-ADJUDICATION** — owner: MAIN compliance owner; consumer store: `gen5_receipts/` and `COMPLIANCE_RUNBOOK.md`; fire trigger: final candidate bytes are selected and checker/helper commit is frozen; action: reconcile the two new runtime-walk reds against the canonical 33-row measured-byte stager or supply a valid equivalence proof, without waiving them by prose.
- **QUEUED-FOR-INDEPENDENT-REVIEW** — owner: submission reviewers who did not author the final repair batch; consumer store: the canonical adversarial-review counter and release hold surface; fire trigger: the final selected candidate and all public documents are frozen after the last swap/edit; action: buy genuinely independent clean passes, resetting to 0 on any finding, until the release bar is satisfied.
- **OPERATOR-OWNED** — owner: repository operator; consumer store: final public PR description and contest pull request; fire trigger: chosen packet, authority axes, compliance dispositions and review bar are final; action: write the policy/LLM answer in their own words and explicitly authorize publication.

## LIVE-HYPOTHESES

- RC2 should preserve the jg5 T4 distortion components and lower only the rate term because two full local real-receiver runs preserved same-axis output bytes and the rider archive is 169 B smaller. This makes `S=0.14827847122030854` plausible, but only the fresh T4 row can establish it.
- The two new compliance reds are caused by the current helper treating declared packet documentation, especially `compress.py`, as runtime. This is plausible because the 33-row canonical stager still re-derives the exact authority tree while the older helper produced 83/87; the final helper-level adjudication is untested.
- The composed runtime may fit the 30-minute T4 wall, but the instrumented 464.559 s observation and contended local timings cannot establish that for the clean composed tree.

## DEAD-ENDS

- Transferring any jg5, instrumented-tree, or local-receiver timing into the composed shipping declaration is closed; the object and regime differ.
- Reading evaluator `final_score: 0.15` as the candidate score is closed; the score is recomputed from components and exact archive bytes.
- Reusing jg5's working hosted URL after an RC2 swap is closed; changed bytes require a new commit pin and downloaded identity proof.
- Calling the public packet self-contained in receipts, or calling the authority runtime a 34-row tree, is closed by the final census and stager receipt.
- Counting the finding pass as clean is closed; the five-pass counter is 0/5.
- Using the operator-only PR draft itself as the checker's competitive-policy statement is closed; the checker accepts a concise affirmative evidence statement, while the public policy answer remains operator-authored.
- Publishing, pushing, opening the PR, or firing an eval from this arm is closed by charter and was not done.

OWN-VEHICLE FRONTIER: unchanged at **S=0.14839100138338618 @ 180,625 B [contest-CUDA T4, n600]**, archive `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e`.
