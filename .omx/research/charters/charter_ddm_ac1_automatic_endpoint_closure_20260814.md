# CHARTER — ddm_ac1_automatic_endpoint_closure (2026-08-14, OPERATOR BINDING "All of this should be automatic and not manual or ad hoc")

CONTEXT (recall, do not re-derive). This session's EC2/bg1 chain is the
empirical anchor: four Modal crash-loop closures took ~10 hand commands
(ledger terminal rows · claim terminal rows · dual-ledger blockers firing
because closure happened after-the-fact); one SKIPPED payload harvest cost a
full arm cycle (bg1 REFUSED-TO-SEAL for want of `modal volume get`); every
git-blocked arm memo needed a hand commit. Every step is deterministic. The
operator has bound the cure: the WHOLE dispatch→endpoint chain is APPARATUS.
Memory: modal-endpoint-closure-must-be-automatic.

## THE WORK (compose existing canonical pieces — build NOTHING that already exists)

1. **Recall first**: `tools/modal_harvest_poller.py` (canonical poller +
   status vocab) · `tools/claim_lane_dispatch.py` (claim rows; flag is
   `--instance-job-id`) · `tac.deploy.modal.call_id_ledger.update_call_id_outcome`
   · `tools/launch_detached_process.py` (detached + done-receipts +
   verify-alive) · the np1 NEXT_IF_RESUMED extractor (commit 499ffd68a1, 2
   wired readers) · `tools/subagent_commit_serializer.py` · the EC2 dispatcher
   family's result schemas (`ddm_ec2_modal_return.v1`, `retained_payloads`/
   `payloads` blocks w/ path+sha256+bytes) · the r4 postmortem chain in
   `.omx/research/` (this session's memos).
2. **BUILD `tools/modal_endpoint_close.py`** — ONE closer, armed DETACHED at
   dispatch time (fire-and-forget via launch_detached_process). Behavior:
   (a) poll the call to terminal (reuse the canonical poller as a library or
   subprocess — do not fork its logic); (b) derive the correct terminal
   status from the result (`training_complete`/rc/error class) and close BOTH
   ledgers — call ledger + claim row — AT terminal, extincting the
   after-the-fact dual-ledger blocker class; (c) pull EVERY payload entry in
   the result manifest (`retained_payloads` + `payloads`) from the named
   volume to a named local store, pre-creating dest dirs (the volume-get
   gotcha), verifying sha256 fail-closed (payload law DEF CON 1000 — a
   mismatch REFUSES and preserves both copies); (d) if the result or an arm
   final-message declares a git-blocked memo path+sha, verify the sha and
   commit via the serializer with the declared message; (e) write a typed
   `ENDPOINT_CLOSURE.receipt.json` (statuses, payload table, memo commit,
   NEXT_IF_RESUMED surface extracted via the np1 reader) + a done-receipt.
   MAIN's endpoint job reduces to reading the receipt and adjudicating.
3. **CLOSURE-MANIFEST convention**: define the small typed block a dispatcher
   result must carry for data-driven closure (volume name · lane id ·
   instance-job id · payload table · optional memo handoff). Retrofit the EC2
   dispatcher family as the reference emitter (additive, no seal-affecting
   fields). Two-landing self-protection: a WARN-ONLY preflight check that new
   `experiments/*_modal_*` dispatchers emit the manifest block (same-line
   waiver for genuinely manifest-free tools; placeholder rationales rejected).
4. **Tests** (≥15): terminal-status derivation table · sha-mismatch refusal ·
   dest-dir pre-creation · dual-ledger both-closed invariant · memo-handoff
   sha gate · receipt schema round-trip · idempotent re-run on an already
   closed endpoint (must be a no-op, never a duplicate terminal row).
5. **Dogfood**: run the closer (dry-run mode where Modal is unreachable from
   the sandbox) against the RETAINED bg1/EC2 receipts as fixtures; MAIN will
   arm it live on the mc35/bg2 endpoints.

## OPTIMAL FORM

Family reference PINS (receipts): the EC2 r1–r4 postmortem chain (memos
2026-08-14) · bg1 REFUSED-TO-SEAL memo ddm_bg1_bilinear_gate_pricing_20260814
· EC2 final_result sha b9df6af2175b2003e8a236e67e3315b60a065c7913792fcd93464c203c321a1b
· harvested payload SHAs 803a1d8755ca…/ffa88ae44787…/9559c2ab5128… · np1
extractor commit 499ffd68a1 · canonical poller `tools/modal_harvest_poller.py`.
MECHANISM reductions = TOY-BRACKET: forking the poller/claims logic instead of
composing it · a closer that only PRINTS what to do (must EXECUTE) · skipping
the idempotency test. Payload law DEF CON 1000. Arms cannot RUN Modal — build
+ test with fixtures/dry-run; MAIN executes live. Git-blocked → declare memo
SHA for MAIN handoff.

## OUTPUT

`tools/modal_endpoint_close.py` + tests + the manifest convention doc in
`.omx/research/ddm_ac1_automatic_endpoint_closure_20260814.md` + the warn-only
gate. Commit via `tools/subagent_commit_serializer.py` (post-edit shas,
`[no-triality] [p0-ledger-ok]`, no co-author trailer). End with
NEXT_IF_RESUMED + LIVE-HYPOTHESES + DEAD-ENDS.
