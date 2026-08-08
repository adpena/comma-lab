# ZC1 Leg Receipt: P2A T0 Harness Drain Six Rows

Exit: DONE-with-artifact.
Axis: task-ledger/source receipt read; no scorer run.
Score claim: false.
Promotion eligible: false.
Verdict scope: INSTANCE, the six P2A T0 rows named by OH1/P2A.

## RECALL EVIDENCE

Searches performed:

- `rg -n "#375|#841|#536|#873|#862|#824|P2A|tr1_window_receipt|factor10|canonical_task_status" .omx/research .omx/state tools src`
- Targeted read of `.omx/research/ddm_p2a_task_backlog_drain_20260801.md`.
- Targeted read of `.omx/research/factor10_kkt_waterfill_blocked_receipt_20260718.json`.
- Targeted reads of `.omx/research/ddm_rg5_rate_gradient_sign_20260801.md` and `.omx/research/ddm_sm2_20260805/SM2_RECEIPT.md`.
- Canonical task-status history was queried for #873 and #824.
- Targeted checks of `tools/auto_push_main.py` and `.claude/settings.json` verified the #375 closure surface.
- Targeted reads of `.omx/research/ddm_wi1_wrong_instrument_sweep_20260731.md` verified #841 closure evidence.

Found beyond the charter seed:

- P2A's #873 status is stale relative to current canonical task state: #873 is completed by `ddm_pj2`, with non-promotable n600 evidence and commits recorded in canonical history.
- #824 remains pending under its current canonical owner; a partial `tr1_window_receipt.json` basename is not enough to close it.
- #862's original hold reason was refuted by RG5, but SM2 narrowed the successor interpretation to the marginal-entropy-blind subspace.

What this changed:

- ZC1 drains the six-row bundle as a closing receipt, without mutating the canonical ledger.
- Rows with current owners or successor scopes are queued rather than falsely closed.

## Six-Row Drain

| task | ZC1 disposition | evidence |
| --- | --- | --- |
| `#375` | CLOSED / FOLDED | `tools/auto_push_main.py` exists and `.claude/settings.json` Stop hooks include it. |
| `#841` | CLOSED / FOLDED | WI1 recorded the wrong-instrument sweep and the ceiling reversal beyond roughly k=600. |
| `#536` | REAL-OPEN / QUEUED-WITH-A-FIRE-ORDER | factor10 KKT waterfill receipt is explicitly no scientific measurement, no score claim, no launch, and blocked by governed-claim/receiver-object requirements. |
| `#873` | CURRENT-COMPLETED / FOLDED | Canonical status shows completion by `ddm_pj2`; P2A's open status is stale. |
| `#862` | ORIGINAL-FRAMING FOLDED / SUCCESSOR SCOPED | RG5 refuted the original backwards-gradient hold; SM2 keeps only the marginal-entropy-blind successor scope. |
| `#824` | PENDING / QUEUED-WITH-A-FIRE-ORDER | Current canonical status remains pending; partial receipt basename does not close the diagonal/off-diagonal scope. |

## Follow-On Disposition

- `#375`: FOLDED.
- `#841`: FOLDED.
- `#536`: QUEUED-WITH-A-FIRE-ORDER. Reopen only with governed claim receipt, receiver object, realized uint8 quantum, candidate delta, dimension rate home, and coder owner.
- `#873`: FOLDED against current canonical status; no P2A re-open.
- `#862`: FOLDED for original backwards-gradient framing. QUEUED only for the SM2-scoped marginal-entropy-blind successor.
- `#824`: QUEUED-WITH-A-FIRE-ORDER. Current owner must attach exact receipt path and resolve the off-diagonal/diagonal scope explicitly.

Own-vehicle frontier line: unchanged, `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
