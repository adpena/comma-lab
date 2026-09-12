# ddm_rih1 — the PR #140 receiver import-hygiene item: a decision packet for the operator ($0; charter, MAIN 2026-09-12; codex sol medium)

## Why
The PR #140 swap packet is staged on move 48 at 91/93 (`submissions/_staging_move48_pr140_swap/`, swp5 memo
`.omx/research/ddm_swp5_restage_pr140_swap_on_move48_20260912.md`, `_packet/COMPLIANCE_after_row.json`). One of the two open items is
`submission_runtime_imports_within_allowlist` — the shipped receiver carries fallback imports the strict allowlist refuses. Whether to
clean them is the operator's receiver-change decision; this arm makes that decision a one-liner by pricing it exactly.

## Deliverable ($0; no Modal; no edits to the staged tree, the live PR tree, sealed trees, or contract code)
1. From the compliance JSON's detail for that check and `scripts/pre_submission_compliance_check.py` (read the allowlist and the
   detection), enumerate EVERY offending import in the shipped receiver (file, line, symbol, whether the import is reachable on the
   contest decode path or only in a dead/fallback branch — prove reachability by reading the code, cite lines).
2. Draft the MINIMAL diff that would pass the check (in a scratch copy under `/Volumes/VertigoDataTier/pact/ddm_rih1/`, never in the staged
   tree), and determine, with receipts: (a) does it change the receiver BEHAVIOUR digest v2 (`measure_receiver_behavior_digest.v2`,
   9f6e7168…; compute it on the scratch copy with the landed tool — grep for the digest tool's argparse)? (b) does it change the timing
   identity class (pr19 `measured_t4_identity_class_envelope`)? (c) therefore: would the cleaned receiver need a FIRST-MEASUREMENT chain
   (a T4 fire, ~$0.3, ~20 min) or can it inherit move 48's measured leg? (d) the decoded output: prove byte-identical raw on a local cold
   decode of a seeded random n≥120 subset with the scratch receiver (or state precisely why not).
3. Write `.omx/research/ddm_rih1_receiver_import_hygiene_decision_packet_20260912.md`: the import table, the minimal diff (verbatim), the
   digest/timing verdicts, the cost of each option (publish as-is with the item open; clean + inherit; clean + first-measurement fire), and
   ONE recommended line for the operator. `# FORMALIZATION_PENDING:<rationale>`.

## Process (codex arm rules)
Serializer commits only with post-edit shas; `REVIEW_GATE_OVERRIDE=1` for the .md; NO co-author trailer, NO AI attribution anywhere. A
Git-object write denial (rc 17) is NOT a stop: keep files in the working tree, leave the bundle, report; MAIN lands; commit LAST. Checkpoint
as `ddm_rih1`. Read `docs/operating_manual_craft_handoff.md`. Do not touch `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7` (live).

Final message: the import table, the diff size, the digest and timing verdicts, the three options priced, the one-line recommendation, memo
path + sha, commit rc, and `composition S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600] (move 48)` unchanged (NOT published).

<!-- # FORMALIZATION_PENDING: charter, not a finding; the arm memo carries the cite -->
