# PQ1 strict-compliance gap report

> **LIVE GENERATION-7 POINTER.** AFR1 is archive `cbb8d928…` / 180,002 B,
> 38-row runtime `6cdfa27d…`, `[contest-CUDA]` T4 n600
> `0.14797617125559104`. The historical gap report below is preserved. Current
> red rows and owners are in the pq12 freeze memo and generation-7 runbook. The
> only publication gate is operator confirmation plus operator-authored text.

> **SUPERSEDED — HISTORICAL (generation-0, the retired e480b candidate,
> archive `e3e6f440…`). Written 2026-08-15; preserved append-only.** The live
> candidate is **generation-6, the composed rider × native port**
> (`df7fd266e1b7488c…` / 180,456 B, 36-row runtime tree `fdd57749…`,
> `[contest-CUDA]` **0.14827847122030852**). That generation has **no compliance
> receipt yet** — it was not re-bought at the swap, because bytes, scanned
> surfaces and the frontier pointer all moved and a receipt is stale when any of
> the three does. Every earlier receipt named anywhere in this file describes a
> superseded object: generation 5 (`f3bce5d2…`/180,625 B, 83/87), generation 4
> (`35c318d5…`/177,182 B, 83/87), generation 3 (`debb025f…`/179,930 B, 82/86).
> The adjudicated red classes are routed in `COMPLIANCE_RUNBOOK.md`.
> Every present-tense gap statement below describes the generation-0 state, not
> the current packet. The one gap this report named OUTSIDE the check set (the
> compression-source reproduction bundle) is ADJUDICATED SATISFIED under the
> pinned-inputs reading — see the runbook's "Compression-source gate
> adjudication" note.

Disposition at time of writing (generation 0): **HOLD — packet prepared, not safe to PR or submit.**

The final real strict check evaluated 86 checks: 78 passed and 8 failed. The
exact candidate archive, member, CUDA authority receipt, executable runtime,
public report, public README, archive manifest, source pin, axis labels, and
competitive statement all passed. The remaining red items are typed below;
none is hidden by a waiver, placeholder, or edited authority receipt.

| Red check | Disposition | Owner | Consumer store | Fire trigger |
|---|---|---|---|---|
| `auth_eval_schema_metric_consistency` | `QUEUED_WITH_A_FIRE_ORDER` | MAIN, authority-adjudication owner | A new blocker-free paired authority receipt beside `receipts/cuda_auth/contest_auth_eval.json` | Exact e480b contest-CPU authority is harvested and the canonical adjudicator can preserve the real component-derived score without rewriting the historical raw receipt |
| `auth_eval_raw_promotion_policy_blockers_absent` | `QUEUED_WITH_A_FIRE_ORDER` | MAIN, submission-policy adjudicator | Paired CPU/CUDA adjudication record in the retained packet receipts | Exact e480b contest-CPU authority and final compliance evidence both exist, so the raw receipt's explicit promotion blockers can be resolved by the real policy review |
| `contest_cpu_auth_eval_exists` | `QUEUED_WITH_A_FIRE_ORDER` | MAIN | `/Volumes/VertigoDataTier/pact/ddm_pq1_submission_packet/receipts/cpu_auth/` and the packet `submission_dir/contest_cpu_auth_eval.json` consumer copy | The e960 packet decision is complete, no conflicting live scorer lane or call exists, the exact archive/runtime hashes revalidate, and MAIN claims the sealed CPU lane in `CPU_AXIS_SEALED_FIRE_ORDER.json` |
| `submission_runtime_has_no_network_install_or_local_paths` | `HOLD_RUNTIME_AUTHORITY` | MAIN, dependency-closure/compliance owner | Accepted dependency-closure equivalence or fresh authority-runtime receipt | The compliance policy explicitly accepts the pinned fail-closed Brotli bootstrap, or a changed dependency-closed runtime receives a fresh exact authority evaluation; do not edit the currently measured receiver in place |
| `hosted_archive_manifest_supplied` | `BLOCKED_ON_OPERATOR_AUTHORITY` | Operator / MAIN after authorization | Hosted archive manifest and PR evidence bundle | The operator authorizes hosting after candidate selection, and the hosted bytes are fetched back and proven SHA-identical to the exact archive |
| `dispatch_claim_successful_exact_eval_terminal_row` | `QUEUED_WITH_A_FIRE_ORDER` | MAIN, lane-claim custodian | `.omx/state/active_lane_dispatch_claims.md` plus a refreshed compliance receipt | The retained CUDA authority receipt is reconciled into a canonical terminal status accepted by the checker; do not manufacture a terminal row from this prep arm |
| `dispatch_claim_terminal_archive_sha_bound` | `QUEUED_WITH_A_FIRE_ORDER` | MAIN, lane-claim custodian | The canonical terminal lane row in `.omx/state/active_lane_dispatch_claims.md` | The real terminal-row reconciliation is performed and its notes bind archive SHA-256 `e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3` |
| `dispatch_claim_terminal_runtime_tree_sha_bound` | `QUEUED_WITH_A_FIRE_ORDER` | MAIN, lane-claim custodian | The canonical terminal lane row in `.omx/state/active_lane_dispatch_claims.md` | The same real reconciliation binds runtime-tree SHA-256 `77b94b5c02c6564024265e3692fc4add10b021038367f962103a648c34ca5035` |

No scorer, paid dispatch, source push, PR, submission, or hosting action was
performed. The active e960 work was not inspected or mutated.

## Additional publication gap outside the 86 strict checks

| Gap | Disposition | Owner | Consumer store | Fire trigger |
|---|---|---|---|---|
| Friendly compression-source bundle | `QUEUED_WITH_A_FIRE_ORDER` | MAIN, public-source owner | Sanitized compression bundle at an immutable public source pin plus its reproduction manifest | The four real RX1/RX2 compression-side scripts have path/config inputs externalized, all RNG and stage checkpoints remain explicit, a clean documented run consumes pinned public inputs, and its retained output proves archive SHA-256 `e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3`; until then the PR answer remains “no” |
| Canonical Git landing | `QUEUED_WITH_A_FIRE_ORDER` | MAIN, Git custodian | Canonical source checkout | Git metadata is writable; verify the retained clean-clone commit/patch against all 14 source artifacts, then land it through the serializer without staging unrelated work |
