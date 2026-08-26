Completed #1302 as two verified thin-bundle landings.

- Landing 1 `dff62c8085…`: replaces the 8.4 GiB checkout clone with bounded Git plumbing, preserves rc=17, and adds loud pre-write rc=19 reserve refusal.
- Landing 2 `b2763c5edd…`: strict preflight guard with executed positive and negative controls.
- Evidence `a5eeb5e6e6…`: memo plus canonical registration, in-progress, and completed ledger rows.
- Validation: serializer suite 118/118; focused guard/integration 45/45; Ruff, diff checks, bundle verification, and two-pass Python review policy all green.
- Staged index remains empty; unrelated SR3 files were preserved.
- No scorer or Modal work ran. GB1 remains `S 0.14811799921260607 @ 180,215 B [contest-CUDA T4, n600]`.

Full evidence: [ddm_fc1x_serializer_fatclone_cure_20260826.md](/Users/adpena/Projects/pact/.omx/research/ddm_fc1x_serializer_fatclone_cure_20260826.md)

## NEXT_IF_RESUMED

- `QUEUED-WITH-LANDING-ORDER` — owner: `MAIN repository custodian`; consumer store: `/Users/adpena/Projects/pact` and `.omx/tmp/codex_runs/ddm_fc1x/`; fire trigger: Git-object writes become available. Land `landing1_dff62c8085.bundle`, then `landing2_b2763c5edd.bundle`, then consume evidence commit `a5eeb5e6e6…` from `evidence_a5eeb5e6e6.bundle`.

## LIVE-HYPOTHESES

- A dynamically assembled clone command outside the bounded direct-sister list could evade the guard; this is plausible because Python argv can be constructed indirectly, though no such route was found.
- The recurring object-write denial is managed-sandbox policy rather than filesystem mode; readable `0755` objects still failed with `Operation not permitted`, but FC1X did not test sandbox policy directly.

## DEAD-ENDS

- Full-checkout clone fallback: closed by the BS3 incident, source repair, and executable guard. Do not retry it on either SSD.
- Prevention without repair: closed because it leaves the harmful mechanism live; both landings are required.
- Invented reserves or post-write checks: closed; the repair uses the canonical 40 GiB constant and refuses before SSD mutation.
- Treating public-repository intake clones as serializer recovery: closed by scope; they do not clone the Pact checkout.
- This apparatus work did not move the own-vehicle frontier: GB1 remains `S 0.14811799921260607 @ 180,215 B [contest-CUDA T4, n600]`.

