# Exact compliance-chain runbook

This runbook targets the active candidate in `PACKET_TARGET.json`. It executes
the real strict checker; it is not a substitute for a green result.

> **ACTIVE SECTION: "Generation 7" at the bottom of this file.** Generation 7
> is AFR1 (`cbb8d928…`, 180,002 B, 38 runtime rows). All earlier commands and
> receipts are historical and must refuse if aimed at the live packet.

> **ACTIVE SECTION: "Generation 6" at the bottom of this file.** Everything above
> it describes superseded candidates and is retained as lineage. **Generation 6 has
> no compliance receipt yet** — it was deliberately not re-bought at the swap,
> because a receipt is a joint measurement of BYTES × INSTRUMENT × WORLD and all
> three moved. Every receipt named above (gen-5 `pq9.r5` 83/87, gen-4 `r1` 83/87,
> gen-3 `r5` 82/86) is HISTORICAL and none of their arithmetic carries.

## Generation 0 run (2026-08-15) — HISTORICAL, retired e480b candidate

The command block and custody paragraph below WERE the generation-0 invocation
(archive `e3e6f440…`/183,502 B on VertigoDataTier). They are preserved as the
template; every pinned value is superseded. The ACTIVE generation-3 values:
archive `debb025f…`/179,930 B on APDataStore, submission tree `67059c1d…`,
portable content tree `994f8aaa…`, receipt **r5** (`gen3_receipts/pre_submission_compliance.gen3.r5.json`,
sha `6f4f6dc8e3648eb0…` — the CANONICAL terminal receipt; r3 and r4 are superseded,
see "Generation 3" below).

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --contest-final \
  --strict \
  --submission-dir /Volumes/VertigoDataTier/pact/ddm_pq1_submission_packet/submission_dir \
  --archive /Volumes/VertigoDataTier/pact/ddm_pq1_submission_packet/submission_dir/archive.zip \
  --auth-eval-json /Volumes/VertigoDataTier/pact/ddm_pq1_submission_packet/receipts/cuda_auth/contest_auth_eval.json \
  --archive-manifest-json /Volumes/VertigoDataTier/pact/ddm_pq1_submission_packet/submission_dir/archive_manifest.json \
  --submission-score-axis contest_cuda \
  --expect-single-member p \
  --expected-archive-sha256 e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3 \
  --expected-archive-size-bytes 183502 \
  --expected-runtime-tree-sha256 77b94b5c02c6564024265e3692fc4add10b021038367f962103a648c34ca5035 \
  --dispatch-claims-md .omx/state/active_lane_dispatch_claims.md \
  --expected-lane-id lane_ddm_rx2_e480b_hpac_winner_v2_paired_modal_auth_20260815T125117Z_contest_cuda \
  --expected-job-id ddm_rx2_e480b_hpac_winner_v2_paired_modal_auth_20260815T125117Z_cuda \
  --competitive-or-innovative-statement-file .omx/research/ddm_pq1_submission_packet_prep_20260815/README_PUBLIC.md \
  --public-scan-path .omx/research/ddm_pq1_submission_packet_prep_20260815/PR_BODY_DRAFT.md \
  --json-out /Volumes/VertigoDataTier/pact/ddm_pq1_submission_packet/receipts/pre_submission_compliance.final.json
```

The omitted CPU-auth, hosted-archive, and public-source-ref manifest arguments
are intentional typed blockers, not forgotten switches. They can be added only
when their real receipts exist. A future rerun must not point them at
placeholders or hand-authored assertions.

The public `README.md` and `report.txt` are inert documentation and are excluded
from the checker/runtime content hash. The GENERATION-0 final strict run
confirmed (2026-08-15, historical) that that packet's executable runtime
matched its measured CUDA authority tree: full tree `77b94b5c02c6564024265e3692fc4add10b021038367f962103a648c34ca5035`
and portable content tree
`26c7d418ca26d7478e67f958354809503242298b5bf8f08c5ff0902508932a20`
(both GENERATION-0 values; the generation-3 counterparts are `67059c1d…` and
`994f8aaa…` per the r5 paragraph below). The principle carries: adding the
public documents creates no local runtime-custody gap.

## Contribution-convention gate

Before recommending a PR, apply `CONTRIBUTION_ETIQUETTE.md` and verify all of
the following against the active packet generation:

- use the repository template headings, including plain-language eval-host,
  build-cost, changes-from-upstream, and competitive-or-innovative answers;
- keep exactly one PR and one active archive URL, hosted outside the code tree,
  then fetch it back and prove its bytes and SHA-256;
- copy the real `report.txt`, label CPU and CUDA separately, and never transfer
  a score across different archive bytes or hardware axes;
- retain section-level lineage credit and remove whole-vehicle originality
  wording;
- do not alter repository-wide dependency files, leak internal paths, include
  provider transcripts, or move score-bearing payload into code;
- if compression source is offered for merge, require a sanitized, seeded,
  documented reproduction bundle that consumes public/pinned inputs and emits
  the retained archive hash. Until that real proof exists, answer the template
  question “no” and do not imply reproducibility.

**Compression-source gate adjudication (2026-08-18, MAIN): SATISFIED under
the pinned-inputs reading for generation 3.** The bundle exists and is real:
`experiments/ddm_pq2_compress_e2e.py` + `RECIPE_sz1_composed.json` rebuilt the
exact candidate end-to-end from four SHA-verified PINNED retained inputs —
final archive `debb025f…`/179,930 B EXACT, every stage rc=0, fail-closed hash
asserts, determinism repeat byte-identical (receipt `RESULT_pq2_e2e.json`).
The inputs are pinned-retained, not public; the PR body discloses exactly
that scope ("Stage A — provenance (documented, not re-run)") and does not
imply training reproduction. The PR body's carefully-scoped "Yes, and it is
offered for merge" therefore stands; the formerly open gate is closed.

The strict checker does not by itself prove the compression-source item; the
adjudication above closes it for the current packet.

## Generation 3 (sz1 composed) strict run — 2026-08-18

Invocation: same checker, adapted values — submission dir
`/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen3_sz1_composed_split`,
archive sha `debb025f45bb42e3…`/179,930 B, runtime tree `0d0fc008d6a3…`,
auth-eval JSON `ddm_sz1/t4_row_composed/r3_artifacts/contest_auth_eval.json`,
lane `ddm_sz1_composed_t4_n600` / job `sz1_composed_r3`. Receipt:
`generations/gen3_receipts/pre_submission_compliance.gen3.r2.json`.

Result: **80 GREEN / 6 RED of 86** (r1 was 73/13; fixed mechanically: report.txt
staged in, non-standard metadata moved to `gen3_receipts/` so the runtime-tree
hash matches the sealed value, sha-bound supplementary terminal claim row).

The 6 residual reds, each typed with its route:

> **CORRECTION 2026-08-18 (r3 receipt `pre_submission_compliance.gen3.r3.json`,
> 82 GREEN / 4 RED).** The prior routes for reds 1-2 were STALE PROSE: the
> "generation-0 pattern" never existed — gen-0's own final receipt
> (`receipts/pre_submission_compliance.final.json` on Vertigo) is RED on the
> same two checks, and its `receipts/cuda_auth/` JSON has no adjudication
> fields (the adjudicator was never run). Verified at source 2026-08-18:
> (a) red 1 was a vocabulary contradiction between two canonical surfaces —
> `experiments/contest_auth_eval.py:2391` stamps
> `canonical_score_source="report_8dp_components_plus_exact_archive_bytes"`
> (pinned by 2 tests) while `tac/auth_eval_schema.py` demanded the older
> literal; cured at the schema (commit `6449c7cdd5`, accepted-label set +
> 2 tests, unknown labels still refused, numeric formula guard unchanged) —
> now GREEN with the sealed receipt untouched. (b) red 2 is STRUCTURALLY
> UNSATISFIABLE for any payload descended from the raw emitter: the blocker
> trio is stamped unconditionally at `contest_auth_eval.py:2530`, and
> `scripts/adjudicate_contest_auth_eval.py` carries the fields through
> (`result_payload = dict(payload)`) while its `_check_raw_promotion_policy_gate`
> treats their presence as gate-triggered ("must not be laundered") — running
> the adjudicator would DOWNGRADE the payload and flip the currently-green
> `auth_eval_adjudicated_raw_policy_clean` check red. Red 2 is therefore a
> DOCUMENTED STRUCTURAL red, same class as red 3: the blockers' substantive
> content is satisfied by the surrounding packet (compliance recorded =
> gen3.r3 receipt; CPU reproduction adjudicated = the measured-infeasibility
> receipt; submission policy gates = the 82/86 strict chain itself).
> (c) red 6 was a status-vocabulary nit: the terminal claim row's status did
> not start with a `completed_contest_cuda*`-family prefix; cured with a
> conforming sha-bound terminal row appended 2026-08-18 — now GREEN.

1. `auth_eval_schema_metric_consistency` — **GREEN as of r3** (see correction).
2. `auth_eval_raw_promotion_policy_blockers_absent` — **DOCUMENTED STRUCTURAL
   red** (see correction); never converted by editing the receipt or the check.
3. `contest_cpu_auth_eval_exists` — STRUCTURALLY RED for this candidate: the
   CPU axis is MEASURED INFEASIBLE within the 1,800 s budget on THESE EXACT
   BYTES (`ddm_cpu1` 2026-08-20, call `fc-01M0FGBV7547NWJVJWQ8W3YX76`: inflate
   4,369.6 s = 2.43x the whole job wall; the evaluator never ran; receipt in
   PACKET_TARGET cpu_axis). The 3,422.7 s figure previously cited here was
   inherited from gen-3 bytes and understated the cost by 946.9 s. No CPU score
   can legally exist; the packet documents GPU-required as a measured fact.
   Documented waiver — never converted by copying a receipt.
4. `submission_runtime_has_no_network_install_or_local_paths` — inflate.sh:27
   pinned-wheel dependency bootstrap (Brotli, fail-closed exit 69). Standing
   MAIN policy adjudication per the e4/PR100/PR101 declared-dependency
   precedent; the check stays red by design and is never hand-edited.
5. `hosted_archive_manifest_supplied` — OPERATOR-GATED: hosting authorization
   is part of the final one-line confirm.
6. `dispatch_claim_successful_exact_eval_terminal_row` — **GREEN as of r3**
   (conforming `completed_contest_cuda_exact_eval_harvested` terminal row,
   sha-bound to archive `debb025f…` + runtime tree `0d0fc008…`).

**r4 (2026-08-18 09:56Z, superseded by r5) —
`gen3_receipts/pre_submission_compliance.gen3.r4.json`: 82 GREEN / 4 RED,
identical red set to r3.** Re-run with `--strict` after the review-pass-5
fix batch (8 findings, all on custody-excluded surfaces: report.txt text,
manifest portable-tree key, staged docs, deleted `__pycache__`); submission
tree `67059c1d…`, portable content tree `994f8aaa…`, and the 34-file runtime
manifest all byte-identical to r3 — measured proof the fixes were
hash-neutral. SUPERSEDED: round-7's fixes rewrote two checker-scanned files
(PR body, packet README) AFTER r4 was produced — round-8 F2 caught the
staleness (r4's own statement_preview still carried the pre-fix submission
name).

**r5 (2026-08-18, the CANONICAL terminal receipt) —
`gen3_receipts/pre_submission_compliance.gen3.r5.json` (sha `6f4f6dc8e3648eb0…`):
82 GREEN / 4 RED, identical red set to r3/r4, re-bought AFTER the round-7 and
round-8 fix batches so every scanned surface it reads is current** (its
statement_preview carries `sz1_composed_reencode`). The expected runtime-tree
sha for the invocation was DERIVED from the r4 receipt, never hand-typed —
the first r5 attempt used a hand-completed sha prefix and the checker
correctly refused with two tree-mismatch reds, an executed demonstration of
the no-hand-typed-values law. All review passes from round 9 onward count
against r5. STANDING RULE: any fix batch that edits a checker-scanned
surface ENDS by re-running the checker and re-pointing all receipt citations.

---

## Generation 4 (ck1 composed row-prune) strict run — 2026-08-19 — HISTORICAL

Result: **83 GREEN / 4 RED of 87**, strict `--contest-final`. Receipt
`generations/gen4_receipts/pre_submission_compliance.gen4.r1.json`
(sha256 `587af0cf78b67858fb044b98e78ac140c62564375504218ad6941d27213ac59b`).
Checker source sha `c4145263037225337d0edda409f513aa5030191afd740036212862043647fad9`;
38 files scanned; frontier pointer state recorded in the receipt's own
`instrument_and_world` block, `last_refreshed_utc 2026-08-18T23:57:41Z`.

The census clause below was run FIRST and returned rc=0
(`39 declared (32 runtime + 7 non-runtime - 0 in both) | undeclared 0 | CENSUS_CLEAN`
and `prep census: 21 flat document(s) | PREP_CLEAN`). The two `MISSING:` lines
name `GENERATION_RECEIPT.json` and `RECEIVER_PARSEBACK.json` — that is the
round-11 F2 exposure closing by construction, not a defect: the ck1 lineage
never had those files. Note also that the round-11 F3 double-declaration
masking is gone at this generation (`0 in both`), because the two names are no
longer in the runtime manifest for either authority to double-cover.

Invocation (every pinned value derived from a receipt, none hand-typed):

```bash
GEN4=/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen4_ck1_composed
R=/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen4_receipts

.venv/bin/python tools/packet_census_guard.py \
    --packet-dir "$GEN4" \
    --auth-eval-json "$R/contest_auth_eval.json" \
    --prep-dir .omx/research/ddm_pq1_submission_packet_prep_20260815

.venv/bin/python scripts/pre_submission_compliance_check.py \
  --contest-final \
  --strict \
  --submission-dir "$GEN4" \
  --archive "$GEN4/archive.zip" \
  --auth-eval-json "$R/contest_auth_eval.json" \
  --archive-manifest-json "$GEN4/archive_manifest.json" \
  --submission-score-axis contest_cuda \
  --expect-single-member p \
  --expected-archive-sha256 35c318d541d703708ab06c55473c200bb893491e24bea312e37be42f010677e3 \
  --expected-archive-size-bytes 177182 \
  --expected-runtime-tree-sha256 da91e06744b94f77077303b2b760cb259aa84b078d998921fb99e018d52fff6f \
  --dispatch-claims-md .omx/state/active_lane_dispatch_claims.md \
  --expected-lane-id ddm_ck1_composed_r4_t4 \
  --expected-job-id ck1_r4_t4_r3_20260818T233350Z \
  --competitive-or-innovative-statement-file .omx/research/ddm_pq1_submission_packet_prep_20260815/PR_BODY_DRAFT.md \
  --public-scan-path .omx/research/ddm_pq1_submission_packet_prep_20260815/PR_BODY_DRAFT.md \
  --json-out "$R/pre_submission_compliance.gen4.r1.json"
```

The omitted CPU-auth, hosted-archive, and public-source-ref manifest arguments
are intentional typed blockers, not forgotten switches. They can be added only
when their real receipts exist.

### The 4 residual reds, each typed with its route

1. `auth_eval_raw_promotion_policy_blockers_absent` — **DOCUMENTED STRUCTURAL**,
   unchanged in kind from generation 3. The blocker trio is stamped
   unconditionally by the raw emitter (`contest_auth_eval.py:2530`), and
   `scripts/adjudicate_contest_auth_eval.py` carries the fields through while
   treating their presence as gate-triggered, so running the adjudicator would
   DOWNGRADE the payload and flip the currently-green
   `auth_eval_adjudicated_raw_policy_clean`. The blockers' substantive content
   is satisfied by the surrounding packet: compliance recorded = this receipt;
   CPU reproduction adjudicated = the cpu_axis block in `PACKET_TARGET.json`;
   submission policy gates = this 83/87 chain. **Never converted by editing a
   receipt or a check.**
2. `contest_cpu_auth_eval_exists` — **RED, and the reason is weaker than
   generation 3's, deliberately.** Generation 3 could call the CPU axis
   MEASURED INFEASIBLE because a Modal x86_64 CPU row was run on its exact
   bytes. **No such row exists on these bytes.** The generation-3 measurement is
   recorded as an inherited expectation (same token decoder, 3,422.7 s vs the
   1,800 s budget) and explicitly NOT transferred as a measurement. No CPU score
   exists and none is claimed. If a reviewer wants this axis measured rather
   than inherited, the route is a paid CPU row on `35c318d5…` — MAIN's call, and
   the fire order for it is not sealed by this arm.
3. `submission_runtime_has_no_network_install_or_local_paths` — `inflate.sh:27`
   pinned-wheel dependency bootstrap (Brotli, fail-closed exit 69). Standing
   MAIN policy adjudication per the e4/PR100/PR101 declared-dependency
   precedent; the check stays red by design and is never hand-edited.
4. `hosted_archive_manifest_supplied` — OPERATOR-GATED: hosting authorization is
   part of the final one-line confirm.

### What moved from red to green versus generation 3

- `auth_eval_schema_metric_consistency` — GREEN (cured at the schema in
  `6449c7cdd5`; carries over).
- `dispatch_claim_successful_exact_eval_terminal_row` — GREEN. A conforming
  `completed_contest_cuda_exact_eval_harvested` terminal row was appended for
  lane `ddm_ck1_composed_r4_t4` / job `ck1_r4_t4_r3_20260818T233350Z`, sha-bound
  to archive and both tree hashes. The dispatch really had completed (rc=0,
  harvested, `poller.done` present); closing a completed dispatch is mandated
  hygiene, not a converted red.
- `frontier_no_regression_on_submitted_axis` — GREEN. This was round-11 F1's
  fifth red on the generation-3 packet: our own frontier had moved past that
  candidate twice in a day. It is green here because this candidate **is** the
  pointer (`same_archive=True`).
- `public_scan_has_no_private_surface` — GREEN at 38 files scanned. On
  generation 3 the same check went RED with 63 hits once the round-11 F2(b)
  scanner fix landed. See the custody note in `GENERATION_LOG.md`.


---

## Generation 5 (HISTORICAL) — jg5 joint-waterfill candidate, the first sub-0.15 row

> **SUPERSEDED 2026-08-20 by generation 6.** Every command, sha, byte count and receipt
> below pins the jg5 object. Do not run these invocations against the shipping packet:
> the expected-identity flags would refuse, which is the tool behaving correctly. The
> live invocation is in "Generation 6" at the bottom of this file.

Result: **83 GREEN / 4 RED of 87**, strict `--contest-final`. Receipt
`generations/gen5_receipts/pre_submission_compliance.gen5.r2.json`. Checker source
sha `c4145263037225337d0edda409f513aa5030191afd740036212862043647fad9`.
Public hygiene GREEN at **39 files scanned, 0 hits** — the denominator is reported
because a scan that opened nothing cannot certify anything.

### The census clause runs FIRST, and the purge runs before the census

**Ordering law established at this generation.** Writing any file onto the ExFAT
volume causes macOS to create AppleDouble `._*` sidecars. Staging generation 5
produced a clean tree; writing the four public docs into it then created **51**
sidecars across the generations tree, which `packet_census_guard.py` caught with
exact paths. Therefore: **purge, then census, then buy the receipt** — in that
order, with no writes in between.

```bash
GEN5=/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen5_jg5_waterfill
R=/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen5_receipts

find /Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations -name '._*' -delete

.venv/bin/python tools/packet_census_guard.py \
    --packet-dir "$GEN5" \
    --auth-eval-json "$R/contest_auth_eval.json" \
    --prep-dir .omx/research/ddm_pq1_submission_packet_prep_20260815 \
    --receipts-dir "$R"

.venv/bin/python scripts/pre_submission_compliance_check.py \
  --contest-final \
  --strict \
  --submission-dir "$GEN5" \
  --archive "$GEN5/archive.zip" \
  --auth-eval-json "$R/contest_auth_eval.json" \
  --archive-manifest-json "$GEN5/archive_manifest.json" \
  --submission-score-axis contest_cuda \
  --expect-single-member p \
  --expected-archive-sha256 f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e \
  --expected-archive-size-bytes 180625 \
  --expected-runtime-tree-sha256 2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b \
  --dispatch-claims-md .omx/state/active_lane_dispatch_claims.md \
  --expected-lane-id lane_ddm_jg5_waterfill455_t4_20260820 \
  --expected-job-id ddm_jg5_t4_r1 \
  --competitive-or-innovative-statement-file .omx/research/ddm_pq1_submission_packet_prep_20260815/PR_BODY_DRAFT.md \
  --public-scan-path .omx/research/ddm_pq1_submission_packet_prep_20260815/PR_BODY_DRAFT.md \
  --json-out "$R/pre_submission_compliance.gen5.r2.json"
```

Staging itself is now a committed tool rather than an ad-hoc script:

```bash
.venv/bin/python tools/stage_contest_submission_packet.py \
    --auth-eval-json "$R/contest_auth_eval.json" \
    --source-runtime-dir /Volumes/APDataStore/pact/ddm_jg5/candidate_runtime_jg5 \
    --out-dir "$GEN5" \
    --expected-archive-sha256 f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e \
    --expected-archive-size-bytes 180625 \
    --json-out "$R/STAGING_RECEIPT.json"
```

### The four reds, with routes

| # | Check | Why red | Owner | Class |
|---|---|---|---|---|
| 1 | `auth_eval_raw_promotion_policy_blockers_absent` | Blockers stamped unconditionally by the raw emitter; running the adjudicator would DOWNGRADE the payload and flip a currently-green sibling | MAIN | **STRUCTURAL** |
| 2 | `contest_cpu_auth_eval_exists` | No CPU row exists on these bytes; the prior lineage measured the axis infeasible (3,422.7 s vs 1,800 s) and this candidate ships the same token decoder | MAIN | **CURABLE** by a paid CPU row; expected to measure infeasible |
| 3 | `submission_runtime_has_no_network_install_or_local_paths` | `inflate.sh:27` pinned-wheel Brotli bootstrap | MAIN | **BY DESIGN**, e4/PR100/PR101 declared-dependency precedent |
| 4 | `hosted_archive_manifest_supplied` | No hosting authorization | **Operator** | **OPERATOR-GATED** |

### r1 → r2: four reds CURED, not adjudicated

The first receipt (`pre_submission_compliance.gen5.r1.json`) measured **79/87 with
8 reds**. The four extra reds were fixed at source rather than argued away:

- `public_source_repo_link_present` — the rewritten PR body had dropped the public
  source repository link. Restored, together with the pinned 40-char source commit
  and the archive↔runtime sha binding.
- `dispatch_claim_successful_exact_eval_terminal_row` — the existing terminal row's
  status (`completed_modal_auth_eval_harvested`) is not in the checker's accepted
  prefix set for a successful exact CUDA eval.
- `dispatch_claim_terminal_archive_sha_bound` and
  `dispatch_claim_terminal_runtime_tree_sha_bound` — the existing row carried only
  the 8-character archive sha prefix; the checker requires the full 64 characters
  of **both** the archive sha and the runtime tree sha.

All three dispatch rows were cured by appending one conforming terminal claim via
`tools/claim_lane_dispatch.py`, binding both full shas. The dispatch genuinely
completed and was harvested, so the row is the hygiene CLAUDE.md requires — but it
was appended by the arm that benefits from it being green, and is **FLAGGED for
independent verification in round 13** for exactly that reason.

### A reader's caution on this receipt

The checker writes its JSON and exits `1` in strict mode **with no stdout at all**.
An empty log is not a crash. Read the receipt, and read it with the key `passed` —
the rows do not carry an `ok` field, and a reader that looks for one will score the
packet 0/87 and raise a false alarm. That mistake was made and caught during this
generation.


## Generation 6 (ACTIVE) — the composed rider × native port

**Object.** archive `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080`
at 180,456 B, member `p` 180,356 B stored; 36-row runtime tree
`fdd5774921319a317a385a9594489aa97e45cebc0f6f20cdc50fe8aaeb08a7f2`; `[contest-CUDA
T4, n600]` **0.14827847122030852**.

**Receipt status: OWED.** Nothing in this section reports a measured compliance
result, because none exists for these bytes. Reporting generation 5's would be the
stale-receipt defect this runbook already names.

### Run it in this order

Census first — both directories, one invocation, `rc` must be 0. Buying a receipt
over a directory holding files nobody declared certifies the contaminants too.

```bash
GEN6=/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen6_rc2_composed
R=/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen6_receipts

.venv/bin/python tools/packet_census_guard.py \
    --packet-dir "$GEN6" \
    --auth-eval-json "$R/contest_auth_eval.json" \
    --prep-dir .omx/research/ddm_pq1_submission_packet_prep_20260815
```

Then the strict chain:

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --contest-final \
  --strict \
  --submission-dir "$GEN6" \
  --archive "$GEN6/archive.zip" \
  --auth-eval-json "$R/contest_auth_eval.json" \
  --archive-manifest-json "$GEN6/archive_manifest.json" \
  --submission-score-axis contest_cuda \
  --expect-single-member p \
  --expected-archive-sha256 df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080 \
  --expected-archive-size-bytes 180456 \
  --expected-runtime-tree-sha256 fdd5774921319a317a385a9594489aa97e45cebc0f6f20cdc50fe8aaeb08a7f2 \
  --dispatch-claims-md .omx/state/active_lane_dispatch_claims.md \
  --expected-lane-id lane_ddm_rc2_composed_cuda_20260820 \
  --expected-job-id ddm_rc2_composed_cuda_r2 \
  --competitive-or-innovative-statement-file .omx/research/ddm_pq1_submission_packet_prep_20260815/PR_BODY_DRAFT.md \
  --public-scan-path .omx/research/ddm_pq1_submission_packet_prep_20260815/PR_BODY_DRAFT.md \
  --json-out "$R/pre_submission_compliance.gen6.r1.json"
```

Staging, for the record — this is the invocation that produced the tree, and it
re-derives the tree hash from freshly measured staged bytes rather than from the
manifest's own claimed digests:

```bash
.venv/bin/python tools/stage_contest_submission_packet.py \
    --auth-eval-json "$R/contest_auth_eval.json" \
    --source-runtime-dir /Volumes/APDataStore/pact/ddm_rc1/candidate_runtime_composed \
    --out-dir "$GEN6" \
    --expected-archive-sha256 df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080 \
    --expected-archive-size-bytes 180456 \
    --json-out "$R/STAGING_RECEIPT.json"
```

### The reds to expect, and how their status changed

These are PREDICTIONS from the generation-5 red set plus what this object measured;
they are not results. Whoever runs the chain records what actually fires.

| # | Check | Expected state on these bytes | Owner | Class |
|---|---|---|---|---|
| 1 | `auth_eval_raw_promotion_policy_blockers_absent` | Still red. Blockers are stamped unconditionally by the raw emitter; running the adjudicator would DOWNGRADE the payload and flip a currently-green sibling | MAIN | **STRUCTURAL** |
| 2 | `contest_cpu_auth_eval_exists` | Still red, and now for a **measured** reason rather than an inherited expectation: the CPU row on THESE bytes was fired and inflation was killed at the 1800 s wall before the evaluator started | MAIN | **MEASURED INFEASIBLE** — no longer "curable by a paid row"; the row was bought and it settled the question |
| 3 | `submission_runtime_has_no_network_install_or_local_paths` | Still red. `inflate.sh` pinned-wheel Brotli bootstrap | MAIN | **BY DESIGN**, e4/PR100/PR101 declared-dependency precedent |
| 4 | `hosted_archive_manifest_supplied` | Still red until the operator publishes. The archive has never been hosted and the PR body's download field is deliberately blank | **Operator** | **OPERATOR-GATED** |

The dispatch-claim checks should now find their terminal rows: lane
`lane_ddm_rc2_composed_cuda_20260820` closed `completed_harvested` binding archive
`df7fd266…`, and the CPU lane `lane_ddm_rc2_composed_cpu_20260820` closed
`failed_inflate_timeout_cpu_wall` on the same bytes.

### One thing this generation changes about red #2

Generation 5 carried red #2 with the note "expected to measure infeasible", inherited
from an older lineage's 3,422.7 s figure. That inheritance was wrong by 946.9 s when it
was finally measured on ck1's bytes, which is why this generation measured its own:
inflation killed at 1,800 s, receiver report 2,850.781244341 s, token decode alone
2,427.166373672 s, decoded token stream bit-identical to the CUDA axis. The axis is
closed by measurement on the shipping object, and no figure was carried onto it.

## Generation 7 (ACTIVE) — AFR1

Object: archive `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`
at 180,002 B, member `p` 179,902 B, 38-row runtime tree
`6cdfa27dd1e9b46fc2bbbe88774c78d95ed3605fee7a15ba3861f96e24041e58`,
`[contest-CUDA]` T4 n600 S `0.14797617125559104`.

```bash
GEN7=/Volumes/APDataStore/pact/ddm_pq12/generation_7_afr1
R=/Volumes/APDataStore/pact/ddm_pq12/receipts
AUTH=/Volumes/APDataStore/pact/ddm_pq12/afr1_authority_materialized/returned_artifacts/contest_auth_eval.json

.venv/bin/python tools/packet_census_guard.py \
  --packet-dir "$GEN7" \
  --auth-eval-json "$AUTH" \
  --prep-dir .omx/research/ddm_pq1_submission_packet_prep_20260815 \
  --receipts-dir "$R"

.venv/bin/python scripts/pre_submission_compliance_check.py \
  --contest-final --strict \
  --submission-dir "$GEN7" \
  --archive "$GEN7/archive.zip" \
  --auth-eval-json "$AUTH" \
  --archive-manifest-json "$GEN7/archive_manifest.json" \
  --submission-score-axis contest_cuda \
  --expect-single-member p \
  --expected-archive-sha256 cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25 \
  --expected-archive-size-bytes 180002 \
  --expected-runtime-tree-sha256 6cdfa27dd1e9b46fc2bbbe88774c78d95ed3605fee7a15ba3861f96e24041e58 \
  --dispatch-claims-md .omx/state/active_lane_dispatch_claims.md \
  --expected-lane-id ddm_afr1_tile48_groupbin8_cuda_n600_20260831 \
  --expected-job-id modal:ddm_afr1_tile48_groupbin8_cuda_n600_20260831 \
  --competitive-or-innovative-statement-file .omx/research/ddm_pq1_submission_packet_prep_20260815/PR_BODY_DRAFT.md \
  --public-scan-path .omx/research/ddm_pq1_submission_packet_prep_20260815/PR_BODY_DRAFT.md \
  --json-out "$R/pre_submission_compliance.gen7.r1.json"
```

The missing hosted manifest is deliberate: this is the real final-mode checker
run at a freeze-not-publish boundary. Expected residuals before measurement are
the raw-emitter promotion stamp, AFR1 CPU RECORD-WITH-REASON, the pinned Brotli
bootstrap, and hosting. Actual results belong in the pq12 freeze memo.

### Generation-7 measured terminal compliance state

Receipt `pre_submission_compliance.gen7.r2.json`: **80 GREEN / 7 RED of 87**.
The first run was 75/87; five failures were stale checker inputs rather than
candidate failures and were cured before the terminal run: the private
operator-input draft was replaced by the factual README as the checker-only
policy statement, and a canonical harvested terminal ledger row was appended
with the full archive/runtime hashes. The seven terminal reds are:

| Check | Disposition | Owner |
|---|---|---|
| `auth_eval_raw_promotion_policy_blockers_absent` | structural raw-emitter stamp; preserve | MAIN policy adjudicator |
| `contest_cpu_auth_eval_exists` | RECORD-WITH-REASON; no inherited score | MAIN scorer router |
| `submission_runtime_has_no_network_install_or_local_paths` | pinned Brotli bootstrap in evaluated receiver | MAIN/runtime policy |
| `submission_runtime_imports_within_allowlist` | staged non-runtime `compress.py` imports repo `tac`; do not fake it into receiver closure | MAIN/packet policy |
| `submission_runtime_tree_matches_auth_eval` | expected staged-doc widening; enumerated 38-row tree is proven by stager, but no separate equivalence proof was supplied | MAIN/packet policy |
| `public_scan_has_no_private_surface` | authority-row `FX5_BUILD_MANIFEST.json` contains three provider-local paths; changing it would change evaluated runtime bytes | MAIN/runtime policy |
| `hosted_archive_manifest_supplied` | intentionally absent at FREEZE-NOT-PUBLISH | operator |

These are recorded, not waived. The six non-hosting rows require changed bytes,
a separate equivalence proof, a new CPU policy requirement, or checker-policy
adjudication; none is silently converted green. Hosting remains the mechanical
red controlled by the operator's single publication gate.
