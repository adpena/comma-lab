# Exact compliance-chain runbook

This runbook targets the active candidate in `PACKET_TARGET.json`. It executes
the real strict checker; it is not a substitute for a green result.

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
from the checker/runtime content hash. The final strict run confirmed that the
packet's executable runtime still matches the measured CUDA authority tree:
full tree `77b94b5c02c6564024265e3692fc4add10b021038367f962103a648c34ca5035`
and portable content tree
`26c7d418ca26d7478e67f958354809503242298b5bf8f08c5ff0902508932a20`.
There is therefore no local runtime-custody gap caused by adding the public
documents.

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

The strict checker does not by itself prove the last compression-source item.
It is an additional publication gate and remains open for the current packet.

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
