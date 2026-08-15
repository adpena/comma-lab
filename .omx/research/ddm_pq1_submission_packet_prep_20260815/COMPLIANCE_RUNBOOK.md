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
