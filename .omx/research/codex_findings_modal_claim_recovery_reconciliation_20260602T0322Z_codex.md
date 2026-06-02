# Codex Findings: Modal Claim Recovery Reconciliation

UTC: 2026-06-02T03:22Z
Artifact: `.omx/research/modal_claim_recovery_reconciliation_20260602T0322Z.json`
Axis: dispatch-custody / no score authority

## Verdict

GO for custody repair. NO-GO for exact dispatch, promotion, or score/rank claims.

Current `origin/main` was missing the active PR101 storage-order CPU claim row
even though the canonical recovery summary still reports the Modal call as
pending. I restored a nonterminal claim row through
`tools/claim_lane_dispatch.py`:

- lane: `lane_pr101_storage_order_len24_exact_cpu_20260601`
- call_id: `fc-01KT2BZT54G6CXPMD94SY43MMH`
- recovery status: `pending`
- restored status: `active_modal_cpu_auth_eval_pending_recovery_poll`
- no score claim, no closure, no CUDA launch

This restores the dispatch hold on main while preserving fail-closed authority.

## Z5 Recovery Attempt

I also polled the stale Z5 Wave2a paired CPU/CUDA claims:

- CPU call `fc-01KSZ74F7PDTC0T7N9B7AV30PR`: `pending` at 2026-06-02T03:20:45Z
- CUDA call `fc-01KSZ7438Z9T0WXB22AN6D5H0T`: `pending` at 2026-06-02T03:20:45Z

Because the canonical recovery tool still reports `pending`, I did not
terminalize either Z5 row. Attempting to refresh the stale rows through
`tools/claim_lane_dispatch.py` was refused because stale nonterminal rows require
terminal stale closure first. That refusal is correct; forced refresh would have
papered over the unresolved Modal state.

## Ledger State

After PR101 claim restoration:

- active rows: 1
- stale nonterminal rows: 8
- terminal latest rows: 723
- unparsable timestamps: 0
- invalid lane ids: 0

The active row is PR101 CPU pending recovery. The stale rows include Z5 CPU/CUDA
and six older local/modal lanes that need lane-specific recovery or terminal
classification before exact-eval planning leans on the ledger.

## Authority

This reconciliation is not a score artifact:

- `score_claim=false`
- `frontier_score_claim=false`
- `rank_or_kill_eligible=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `exact_or_full_video_launched=false`

## Next Step

Keep polling PR101 CPU until the canonical recovery tool returns a terminal
artifact. Do not launch new full-video/exact/CUDA work while PR101 is pending.
Poll Z5 CPU/CUDA separately and terminalize only when recovery produces terminal
evidence, or when an operator explicitly authorizes a stale-assumed-dead closure.
