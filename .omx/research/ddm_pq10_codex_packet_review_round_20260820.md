# ddm_pq10 — Codex cross-family packet review round

`date_utc: 2026-08-20` · `owner: ddm_pq10` · `score_claim: false` ·
`publish_action: none` · `counter_at_exit: 0/5`

## Outcome

**FINDING ROUND. The five-pass counter remains 0/5.**

The packet is not the selected shipping object. Every live publication surface still
describes the jg5 archive `f3bce5d2…` at 180,625 B and its 33-row runtime tree
`2103073d…`, while the admitted own-vehicle frontier and selected shipping identity are
the composed rider archive `df7fd266…` at 180,456 B through the 36-row composed runtime
tree `fdd57749…`. The fresh CUDA and CPU receipts already exist, but the public documents
still say `GATED-ON-RC2` and the executable appendix still verifies jg5.

This was not repaired by a document-only edit. A truthful repair requires the typed
candidate swap: archive, runtime, manifest row set, hosted URL, source pin, receipts and
all packet documents must move together. Staging the composed object would change the
runtime tree from the charter's hard-pinned `2103073d…` to the measured `fdd57749…`, while
the charter says to STOP if that hash moves and assigns the runtime/archive landing to the
parallel wave owner. Updating prose alone while leaving the old archive/runtime in the
active packet would be a NO-FAKE identity defect. Therefore this arm left every packet
byte untouched and routed the indivisible swap rather than manufacturing a partial fix.

No publish, push, hosting action, Modal fire, PR opening, scorer launch, `upstream/` edit,
or edit under `submissions/robust_current/jg5_sub015_runtime/` occurred.

## Findings

### F1 — selected-object swap never reached the packet

**Severity: blocking for publication. Class: cross-object identity / incomplete swap.**

Across the prep tree and active generation, searches found the old archive SHA in 17
files, the old byte count in 16 files, the old score in 14 files and the old runtime-tree
pin in 11 files. The selected composed archive SHA, 180,456-byte count, exact score and
`fdd57749…` runtime tree occur in **zero** packet/prep files.

Measured from disk and the retained authority receipt:

| identity field | active packet | selected composed object |
|---|---:|---:|
| archive SHA-256 | `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e` | `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080` |
| archive bytes | 180,625 | 180,456 |
| archive member | `p`, 180,525 B | `p`, 180,356 B |
| runtime rows | 33 | 36 |
| measured runtime tree | `2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b` | `fdd5774921319a317a385a9594489aa97e45cebc0f6f20cdc50fe8aaeb08a7f2` |
| recomputed CUDA score | 0.14839100138338618 | **0.14827847122030852** |

The selected archive on disk is exactly 180,456 B, hashes to `df7fd266…`, contains one
stored 180,356-byte member `p`, and all 36 authority rows re-hash to `fdd57749…`. The
active packet's 33 rows independently re-hash to `2103073d…`. This is not display drift;
it is two different executable objects.

### F2 — fresh RC2 runtime and CPU outcomes remain falsely pending

**Severity: high. Class: stale receipt gate / cross-regime cure not consumed.**

`GATED-ON-RC2` appears 33 times across the three live source documents and their staged
README/report copies. The named fire trigger has already occurred.

The retained CUDA receipt measures, on the selected exact archive/runtime pair:

- `inflate_elapsed_seconds = 458.752594349`
- `evaluate_elapsed_seconds = 39.72359129999995`
- `contest_auth_eval_elapsed_seconds = 508.81608174799993`
- outer Modal function elapsed `513.7945766859999 s`
- Tesla T4 match true, n600, `d_seg = 0.00020139`, `d_pose = 0.00000637`
- score recomputed from components and exact bytes:
  `0.14827847122030852`; receipt display field `0.15` is not quoted as the score.

The retained CPU receipt measures the same archive and the same runtime-content identity
(`runtime_content_tree_sha256 ccd9f7ab…`, `runtime_files_sha256 e8dcbc65…`, 36 rows) on
Linux x86_64 with four Torch threads. Inflation timed out at the exact 1,800-second contest
wall before evaluation, so there is no CPU score. The receiver later emitted a complete
2,850.781-second report with token decode alone at 2,427.166 seconds,
`NativeFreeCorrector`, and decoded-token SHA `cc10a7b0…`; this is a wall failure, not a
decode desynchronization. The packet must declare `linux-nvidia-t4` and CPU
wall-infeasible, not pending.

Two current-method statements are stale consequences of the same unconsumed receipts:
the README says the 183-byte carrier rider was declined and that the native port “does not
ship.” The rider was re-measured on the final body at 169 B and the selected object ships
both the rider and the clean native corrector. The historical 183-byte row may remain only
with an explicit supersession to the 169-byte admitted row.

### F3 — the repo-side archive manifest is stale even on its own named object

**Severity: medium-high. Class: stale machine-readable present-tense sibling.**

`ARCHIVE_MANIFEST.json` still describes packet generation 4 (`35c318d5…`, 177,182 B,
score 0.15710198138050818), while the packet-side `archive_manifest.gen5.json` describes
jg5 and the selected object is now composed RC2. Its `cpu_axis_note` also says no exact CPU
measurement exists and carries an inherited expectation, despite the file's own generation-4
adjudication note in `SWAP_PROCEDURE.md` recording the measured 4,369.6-second CPU wall on
those exact bytes. This is a stale machine-readable claim independent of the RC2 swap.

### F4 — the reviewer appendix executes against the wrong object

**Severity: blocking for a counted pass. Class: asserted verification surface does not
verify the selected submission.**

The appendix commands are syntactically live for jg5, not for the selected composed
candidate. The local/pinned-object form recomputed jg5 exactly and the manifest command
returned 33/33 `OK`, but the URLs, expected archive SHA, expected runtime tree and expected
score all bind the superseded jg5 object. Executing them successfully would therefore prove
the wrong submission.

The network forms were executed as written and failed at DNS resolution for
`raw.githubusercontent.com` in this sandbox. A bounded substitute read the same commit-pinned
Git objects locally: the pinned archive equals the active packet byte-for-byte and the pinned
T4 receipt command recomputes `0.14839100138338618`. This proves the command arithmetic and
the pinned Git content, not live HTTP availability. The final evaluator command was not
launched: this charter does not grant the full-n600 scorer slot, the selected packet is not
staged, and re-evaluating superseded jg5 would duplicate a settled authority row. Its
disposition is **FOLDED** into the existing exact jg5 and RC2 authority receipts; after the
typed swap, a fresh reviewer must execute the appendix on the final staged object.

## Per-item review ledger

| claim checked | method EXECUTED | MEASURED result | verdict |
|---|---|---|---|
| `MANIFEST.sha256` verifies the active tree | Ran `shasum -a 256 -c MANIFEST.sha256` from the active packet; independently re-derived the tree from the 33 staged files | 33/33 `OK`; measured tree `2103073d…` | **CLEAN for jg5; FINDING for selected-object applicability** |
| Hosted archive identity command | Ran the exact shell block; DNS failed. Then read the exact commit-pinned blob through local Git object custody and compared it with staged bytes | pinned Git blob and active archive both `f3bce5d2…`; live HTTP not measured this round | **ENVIRONMENT-BLOCKED network leg; FINDING because URL is the old object** |
| T4 receipt recompute command | Ran exact expression on the locally pinned `MODAL_REMOTE_RESULT.json`; independently recomputed both jg5 and RC2 from components | jg5 `0.14839100138338618`; RC2 `0.14827847122030852`; both receipt display fields `0.15` | **ARITHMETIC CLEAN; selected-object pin FINDING** |
| Final evaluator invocation | Inspected clean pinned upstream `evaluate.sh`/`evaluate.py`; did not launch a duplicate full-n600 job without scorer ownership | upstream checkout clean; `evaluate.py` SHA `7da71a84…`; existing exact receipts consumed | **FOLDED, not counted as executed; round cannot be clean** |
| rv15 F1 denominator cure | Searched all live public surfaces for `15.3%`, old wall pairs and denominator remnants | zero stale occurrences; current public docs no longer mix token-stage and inflate ratios | **CLEAN** |
| rv15 F2 stager mechanism cure | Re-derived old and composed trees from measured files; ran stager+census tests and Ruff | old `2103073d…`, composed `fdd57749…`; 49 tests passed; Ruff clean | **CLEAN mechanism; proves the two objects differ** |
| pq9 three fixes | Re-read evaluator path, receipt-location wording and runtime-row language | evaluator path is correct for a contest checkout; receipts explicitly outside public packet; text says 33 manifest rows plus archive, not 34 runtime rows | **CLEAN** |
| Score/byte custody | Recomputed score formula from receipt components; measured both archives and ZIP members | all quoted jg5 primary numbers reproduce, but they belong to the superseded object; RC2 values above reproduce | **FINDING: correct numbers on wrong selected object** |
| Runtime custody | Parsed nested CUDA/CPU artifacts and joined runtime content/file hashes across axes | CUDA exact timings measured; CPU 1,800-second wall measured; no CPU score | **FINDING: still published as pending** |
| Shipping identity consistency | Exact-content search across prep tree and active generation | 0 files name `df7fd266…`, 180,456 B, RC2 exact score or `fdd57749…`; 71 old-identity references in the core review set | **FINDING** |
| Swap procedure URL re-pin step | Read `SWAP_PROCEDURE.md` step 4A and refusal conditions | exact-byte commit/push, new 40-character pin, fresh HTTP 200, SHA and byte equality are all explicit | **CLEAN; not yet executed** |
| Axis declaration | Compared PR/README/report wording with both axis receipts | `linux-nvidia-t4` is the right requested axis; CPU is wall-infeasible rather than pending | **PARTIAL: runner clean, boundary stale** |
| Borrowed-substrate accounting | Read the append-only live §9 table and PR summary; compared to composed mechanism set | table is itemized and conservative for jg5; it has no composed-rider/native-port amendment | **FINDING for selected object** |
| LLM disclosure | Searched and read the full section | still says `DRAFT ONLY; OPERATOR MUST REWRITE IN THEIR OWN WORDS`; no policy answer was filled | **CLEAN** |

## Reviewer-persona pass

**Contest maintainer.** I would stop before technical review because the proposed body,
download URL, manifest and verification commands describe a different executable object from
the one claimed as the shipping candidate. The old jg5 arithmetic is careful and reproducible,
but it cannot stand in for the composed runtime. I would ask for one final frozen archive/runtime
pair, a URL pinned to those exact bytes, the 36-row manifest from its authority receipt, the
measured T4 times separated from any projection of the full CI setup, and the CPU timeout stated
as a timeout rather than as a missing score.

**Rival submitter.** I would challenge the packet if it retained the jg5 contribution story
while shipping the rider plus native port. The current methods section says the 183-byte rider
was declined and the native port does not ship, while the admitted row is a 169-byte rider
through the native runtime. I would also insist that the borrowed-substrate table add those
mechanisms and keep the PR #138 / PR #135 attribution boundaries explicit before treating the
new total as a publishable competitive claim.

**Compliance auditor.** The operator-only LLM scaffold is correctly unfilled, and the URL-re-pin
procedure is explicit. The release is nevertheless blocked: the current 33-row manifest proves
jg5, the selected authority receipt proves a different 36-row tree, the hosted URL is still
jg5, the strict compliance receipt predates the swap, and the full evaluator appendix has not
been executed on a final staged composed packet. No prose waiver can reconcile those identities;
the swap, hosted-byte proof, compliance re-buy and fresh clean-pass sequence are indivisible.

## Counter verdict

This round found four material defects/classes. Fixing them later will not make this round
clean. **Consecutive clean-pass counter: 0/5.** The next eligible round begins only after the
composed object and every current publication surface are frozen together.

## RECALL EVIDENCE

Searched the full `.omx/research/` corpus, live state, arm final messages, design/spec surfaces,
canonical index/DAG and task/dispatch ledgers with the content queries `PR_BODY_DRAFT`,
`submission packet`, `hosted archive`, `re-pin`, `runtime budget`, `review counter`, `jg5`,
`RC2`, `df7fd266`, `t4_row_r2`, `GATED-ON-RC2`, `OPERATOR MUST REWRITE`, `borrowed substrate`,
`contest maintainer`, `rival submitter` and `compliance auditor`. Ran
`tools/list_canonical_equations.py --json` and filtered for score/rate/pose/runtime/archive/
waterfill. Read pq9 first, then pq8/rv15, RC2 CUDA, the live CPU closure, sw2, the stager and
the complete live PR/README/report/accounting/swap/manifest surfaces.

Beyond the charter seeds, recall found that the RC2 CPU leg had terminally closed as a pure
wall failure, not merely remained live; it also found the repo-side archive manifest was stale
against its own generation-4 CPU adjudication. Those facts changed the plan from a possible
clean confirmation to a finding round and made a prose-only RC2 refresh inadmissible. The
canonical equations did not change the score calculation: the exact contest formula and
object-specific container/runtime laws remain the applicable rows. Did not find a later packet
swap or composed hosted-URL receipt in the searched packet, index, DAG and live-state scope.

## Measurement boundary

Measured exact packet/source bytes, ZIP members, manifest rows, runtime-tree derivations,
component score arithmetic, CUDA receipt timings, CPU timeout evidence, cross-axis runtime
content identity, text denominators, stale-identity occurrence counts, stager tests and lint.
Did not measure live HTTP availability, did not run a new scorer, did not measure a new score,
did not stage the composed packet and did not alter any runtime or packet byte.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN packet owner; consumer store: a new retained composed generation under `/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/` plus every file in `.omx/research/ddm_pq1_submission_packet_prep_20260815/`; fire trigger: MAIN resolves the charter's obsolete `2103073d… UNCHANGED` guard in favour of the selected exact object's measured `fdd57749…` identity and confirms rv17 is not concurrently editing the packet; action: execute `SWAP_PROCEDURE.md` from `VERIFY_SOURCE` through document refresh using the retained RC2 CUDA/CPU receipts, with no hand copy.
- **OPERATOR-GATED** — owner: repository operator plus MAIN packet owner; consumer store: public source commit, commit-pinned archive/receipt URLs and final freeze receipt; fire trigger: the composed generation is frozen and the operator explicitly authorizes the public push; action: run swap step 4A, then prove HTTP 200, SHA-256 and byte-count equality for the exact selected archive and receipt.
- **QUEUED-FOR-COMPLIANCE-REBUY** — owner: MAIN compliance owner; consumer store: composed generation receipts and `COMPLIANCE_RUNBOOK.md`; fire trigger: hosted pins and all checker-scanned surfaces are final; action: rerun both censuses and the strict chain against the 36-row selected object, preserving every red with a typed disposition.
- **QUEUED-FOR-INDEPENDENT-REVIEW** — owner: a reviewer who did not author the swap/fix batch; consumer store: `ADVERSARIAL_REVIEW_SCAFFOLD.md` five-pass counter; fire trigger: composed bytes, URLs, manifests, PR body, README, report and accounting are frozen after the compliance re-buy; action: execute every appendix command and begin a new clean-pass attempt, resetting to 0 on any finding.
- **OPERATOR-OWNED** — owner: repository operator; consumer store: final public PR description and contest pull request; fire trigger: the packet reaches the release bar; action: write the LLM/policy answer in the operator's own words and explicitly authorize publication.

## LIVE-HYPOTHESES

- A fully refreshed composed packet should preserve the jg5 distortion prose and change only the archive/runtime/rate/timing/custody surfaces, because the RC2 T4 row reproduced both reported distortion components exactly. This is plausible from the exact receipt, but every section-level and attribution statement still needs a fresh object-specific audit.
- The full GitHub Actions job should fit the 1,800-second wall on T4 because the measured authority wrapper is 508.816 seconds and prior checkout/dependency estimates leave substantial headroom. It remains a projection until the actual workflow runs; the packet must not promote it to a measured whole-CI time.
- The CPU and CUDA runtime-tree hashes differ because their canonical manifests include host-specific path metadata while their 36 measured file digests and portable content tree match. This is plausible from the equal `runtime_files_sha256` and `runtime_content_tree_sha256`; the final packet should use the authority tree per axis and the portable content identity for the cross-axis join rather than assert one tree hash transfers.

## DEAD-ENDS

verdict_scope: instance — closed repair/packaging routes for the jg5→rc2 swap on THIS packet
generation only; process-route closures, not technique/family negatives.

- A document-only substitution of `df7fd266…` into the current packet is closed: it would leave the old jg5 archive/runtime behind and create a fake identity claim.
- Counting this round as clean after later fixes is closed: findings reset the counter and fixes require a new independent pass.
- Reusing jg5's hosted URL, 33-row manifest, `2103073d…` runtime pin or score recompute for the composed object is closed: the measured objects differ.
- Keeping `GATED-ON-RC2` after both fresh receipts exist is closed except for explicitly named quantities the receipt schema does not measure, such as the complete GitHub Actions setup wall.
- Carrying the historical 183-byte rider or “native port does not ship” statements into the composed body is closed by the final 169-byte admitted row and exact shipping runtime.
- Quoting `final_score: 0.15` as the exact score is closed; both authority rows are recomputed from components and exact archive bytes.
- Re-running superseded jg5 merely to make the appendix look executed is closed: it duplicates settled authority evidence and still verifies the wrong selected object.

OWN-VEHICLE FRONTIER: unchanged at **S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]**, archive `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080`.
