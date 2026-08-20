Implemented and landed all four decision-integrity ports in commit `a61c550890`:

- m94 claim-unit scoping extends canonical `verdict_payload()` and refuses over-capacity claims.
- m37 enforces same-parent SHA-256 freshness for fits, maps, selectors, and corrections.
- Activation-ledger CLI emits typed `FIRED`/`FOLDED`/`queued` joins and exits nonzero on missing evidence.
- EG1 exposes parameterized same-parent complete-score `continue`/`handoff`/`stop` decisions without TR1 constants.

Verification: 151 tests passed post-commit; Ruff and diff checks passed; two clean reviews plus policy approval recorded. The real v7.5.2 compiler’s 9 non-default levers joined 9/9 against the live 251-row ledger.

No scorer, evaluator, Modal, GPU, MPS, terminal object, candidate, archive, payload, or score measurement ran. Protected files and unrelated worktree changes were preserved. Full receipt: [ddm_ip1_decision_integrity_ports_20260811.md](/Users/adpena/Projects/pact/.omx/research/ddm_ip1_decision_integrity_ports_20260811.md).

Effective frontier remains cp135 `S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`.  
Own-vehicle frontier remains `LC2 S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, adjudicated, n600]`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: JS1 reseal owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hr1_preflight/instrument_scope/`; fire trigger: before the first terminal-base scorer or realization verdict; action: emit the real m94 scope receipt.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: JS1 content-binding owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hr1_preflight/content_bindings/`; fire trigger: terminal roles bind and before every selected candidate is scored; action: emit freshness receipts for each fit/map/selector/correction.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: JS1 reseal and costate owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hr1_preflight/activation_audit/`; fire trigger: exact terminal DSL compilation completes; action: run the activation join before enabling any realization command.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: JS1 event-controller owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/event_policy/`; fire trigger: before Stage-0 stopping or terminal-finisher admission is sealed; action: derive `StopVerdict` from current same-parent complete-component receipts.

## LIVE-HYPOTHESES

- m94 may prevent false object/family closures because prior reversals were caused by instruments poorer than their claimed objects.
- m37 may catch stale terminal fits after parent selection changes; q43a already exposed this failure shape.
- The terminal activation join may expose a lever omitted by historical summaries because the ledger documents writer-vacuity and namespace drift.
- Complete-score dominance may reject attractive axis-local finishers that lose after all score components are included.

## DEAD-ENDS

- Do not create another verdict surface; m94 belongs in the canonical embeddable producer.
- Do not treat `never_fired()` or a bare ledger count as terminal-config truth.
- Do not infer `FOLDED` or `queued` from free-text reasons; typed events now exist.
- Do not import EG1’s TR1 constants, packet grammar, targets, or stale effect sizes.
- Do not permit parent-hash waivers or force-fit adapters for freshness-bound objects.