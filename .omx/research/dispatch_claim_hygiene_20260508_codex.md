# Dispatch Claim Hygiene - Codex - 2026-05-08

scope: local dispatch-claim coordination hygiene only
remote_dispatch: false
score_claim: false

## Action

The adversarial review of the 2026-05-08 swarm found stale nonterminal rows in
`.omx/state/active_lane_dispatch_claims.md`. The claim helper's ledger header
states that stale claims older than 24 hours should be marked
`stale_assumed_dead` before the lane is reused.

I used `tools/claim_lane_dispatch.py claim --force` to append terminal rows for
the `33` stale nonterminal lane/job pairs reported by:

```bash
.venv/bin/python tools/claim_lane_dispatch.py summary
```

No remote job, GPU eval, CPU eval, or training dispatch was attempted.

## Result

Post-cleanup summary:

```text
CLAIM_SUMMARY active=3 stale_nonterminal=0 terminal_latest=516 unparsable_timestamp=0
```

The remaining active rows are current `pending_authorization` prestage rows and
were intentionally left open:

- `apogee_int6_contest_cuda_anchor`
- `pr101_admm_step6_no_dead_k`
- `pr107_apogee_cpu_auth_eval_linux_x86_64`

## Evidence Semantics

- Evidence grade: `coordination`
- Score/rank/promotion impact: none
- Dispatch impact: removes stale conflict noise only
- Reactivation: if any closed provider job unexpectedly resurfaces, append a new
  claim row with the live provider evidence and exact job id.
