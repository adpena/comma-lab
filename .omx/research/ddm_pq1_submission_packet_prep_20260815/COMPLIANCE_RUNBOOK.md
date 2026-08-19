# Exact compliance-chain runbook

This runbook targets the active candidate in `PACKET_TARGET.json`. It executes
the real strict checker; it is not a substitute for a green result.

> **ACTIVE SECTION: "Generation 4" at the bottom of this file.** Everything above
> it describes superseded candidates and is retained as lineage. The active
> receipt is `generations/gen4_receipts/pre_submission_compliance.gen4.r1.json`
> (83 GREEN / 4 RED of 87). The generation-3 r5 receipt and its 82/86 arithmetic
> are HISTORICAL.

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
  --competitive-or-innovative-statement-file .omx/research/ddm_pq1_submission_packet_prep_20260815/PR_BODY_DRAFT.md \
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
   CPU axis is MEASURED INFEASIBLE within the 1,800 s budget (inflate 3,422.7 s,
   receipt in PACKET_TARGET cpu_axis). No CPU score can legally exist; the
   packet documents GPU-required as a measured fact. Documented waiver — never
   converted by copying a receipt.
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

## Generation 4 (ck1 composed row-prune) strict run — 2026-08-19 — ACTIVE

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
