Implemented and committed automatic Modal endpoint closure.

- [Closer](/Users/adpena/Projects/pact/tools/modal_endpoint_close.py) polls terminal state, closes both ledgers, harvests and SHA-verifies every payload, handles memo commits, extracts `NEXT_IF_RESUMED`, and emits typed receipts.
- [EC2 dispatcher](/Users/adpena/Projects/pact/experiments/ddm_ec2_modal_oriented_adapter_trainer.py) now emits the manifest and automatically arms the detached closer.
- Added the warn-only dispatcher coverage gate and [36 fault-injection tests](/Users/adpena/Projects/pact/tools/tests/test_modal_endpoint_close.py).
- Verification: 131 tests passed, Ruff and compilation passed, and two review-tracker passes completed.
- R4/BG1 dogfood found 5/5 payloads: three verified locally and two correctly classified `would_pull`; both ledgers remained no-ops.
- Full convention, recall, custody, and boundaries are in the [AC1 receipt](/Users/adpena/Projects/pact/.omx/research/ddm_ac1_automatic_endpoint_closure_20260814.md).

Commits: `4f4537d835`, `b4404f9fa3`, `8a3207e10e`, `f204c8fcb6`. Exact-hunk patch mode excluded unrelated existing EC2/preflight edits, which remain untouched.

No Modal job, scorer, training, or evaluation ran. Effective frontier remains CP135 at `S=0.16195513827824176 @ 186,252 B`; own-vehicle frontier remains LC2 at `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN / next MC35-BG2-family Modal dispatcher owner. Consumer store: that dispatch’s SSD `endpoint_closure/` and `retained/endpoint_payloads/` directories. Fire trigger: before its next data-bearing `.spawn()`, emit the closure manifest and arm the closer; MAIN then adjudicates the typed receipt.**

## LIVE-HYPOTHESES

- The complete remote manifest should eliminate skipped-harvest arm cycles; R4 reconstructed all 5/5 payloads, but live provider-backed pulling remains untested.
- Claim-first terminal closure should eliminate the observed dual-ledger blocker; temporary-ledger tests prove local ordering, while live provider timing remains untested.
- Typed memo handoffs should eliminate manual git-blocked memo commits when an arm supplies exact path, SHA, and message metadata.

## DEAD-ENDS

- A second poller was closed because the canonical poller already owns timeout and remote-exception behavior.
- A print-only runbook was closed because it preserves the manual failure class.
- Treating local poll deadlines as remote failures was closed because it falsifies provider terminality.
- Winner-only or size-only payload custody was closed in favor of complete manifest enumeration and SHA verification.
- Retrofiring BG2 or MC35 was closed because both searched units are already terminal.