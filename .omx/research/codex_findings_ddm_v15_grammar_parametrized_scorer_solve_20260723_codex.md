---
title: Codex findings — DDM V15 grammar-parametrized scorer solve
utc: 2026-07-23T00:06:05Z
tasks: [578, 603, 613]
verdict: FORMULATION_SCOPED_GATE_NOT_MET
verdict_scope: shared 1x1 RGB templates x three row bands x bounded projected-gradient/secant search
research_only: true
execution_allowed: false
score_claim: false
main_landing_review_required: true
---

# Outcome

V15 extends the v14 receiver with a strict six-record shared RGB template bank and solves those
templates encode-side through exact `R` and frozen SegNet. No update survived the hard zero-collateral
gate. Full n600 is `133,941 B / d_seg 0.027470296224 / d_pose 163.061327281443`; Movable is
`0.291615222639` and Lane is `0.435195521828`. The archive fits the 160 KB box, but Movable misses
the preregistered `<=0.05` gate. Do not flag R6. #366 joint predictor/template training remains open.

## What landed

- Strict V15 compiler/parser/receiver schema with counted row-band templates, parse/re-encode
  equality, exact member-home custody, no scorer/GT table at decode, and fail-closed mutation proof.
- Typed local-only n64/n600 runner with Fisher/margin ranking, exact-R gradients, uint8 projection,
  realized secants, zero-collateral admission, reverse-waterfill, immutable stage checkpoints, and
  SHA-bound n64→n600 handoff.
- Canonical admission law, tests, equations note, DAG feed, invalidation record, and receipts.

## Bounded re-derivation

```text
/Users/adpena/Projects/pact/.venv/bin/python tools/measure_ddm_v15_scorer_solved_templates.py --config .omx/research/configs/ddm_v15_scorer_solved_templates_n64_20260722.json --output-directory .omx/research/ddm_v15_scorer_solved_templates_n64_20260723T011500Z
/Users/adpena/Projects/pact/.venv/bin/python tools/measure_ddm_v15_scorer_solved_templates.py --config .omx/research/configs/ddm_v15_scorer_solved_templates_n600_20260722.json --output-directory .omx/research/ddm_v15_scorer_solved_templates_n600_20260723T013000Z
```

Authoritative receipt SHAs are `be679f30d913ced637001548e3a8e5d44ec992c64489ce3ee44bc1c4a1849639`
(n64) and `5ed6f830b3749a51e0d300a9104fda9a77e86bbeb3b81428a20e1ec0d3dcfcb8`
(n600). Producer SHAs are `a39dad6f79a560f3f7d544593d3aa49e9d68f234d83e9263551d2e277d8456d2`
(runner) and `1a3622a64b307c8b5a6b1987f8bdb86d9df441d0159a3a420bd9c283d41f0824`
(receiver/compiler).

## Constraint diagnosis

The n64 solve admitted zero steps. Improving Movable proposals harmed at least 13 baseline-correct
off-target cells; improving Lane proposals harmed at least 23. The emitted six-record/86-byte bank
therefore equals the inherited prototypes. N600 adds 694 exact archive bytes and all 600 receiver
camera outputs are byte-identical to v14 (38 preserved batches, digest
`5d502c1eafe0bd6b3a3e8ea323b02a66573f51939e0a21ebde6e592e04141d7c`). Its scorer row is
`DERIVED_FROM_EXACT_FULL_P_CAMERA_BYTE_IDENTITY`, not independently remeasured or promoted.

## Round-1 adversarial self-review

1. An initial receiver applied templates after all semantic paint; later-role ownership was broken.
   The receipt is invalidated and the receiver now applies templates inside the v14 paint order.
2. A second solve optimized cells later overwritten by higher-priority roles. That receipt is
   invalidated; receiver-visible masks now subtract later semantic supports, and a no-op prototype
   replay is byte-identical.
3. Storage preflight initially persisted live free bytes and an absolute worktree path. Those
   receipts are superseded; the gate still measures capacity fail-closed, while the receipt retains
   only deterministic gate facts. A regression test covers the bug class.
4. This is projected gradient plus realized secant search, not a #549 QP. It evaluates shared 1x1
   templates only. Larger patches, a true constrained QP, and #366 joint training remain untested.
5. Pose was not inside the template optimizer. The n600 pose value is preserved through exact camera
   identity; any non-identity successor must restore the joint Seg/Pose trust region.
6. AR(1) was not guessed: it remains blocked without decoder-free physical-BEV custody.

## Blocker delta versus #603

#603 supplied scorer obligations but no compact RGB realization. V15 supplies a legal counted
template realization and exact encode-side optimizer/receiver closure. The remaining blocker is no
longer missing actuation; it is the measured incompatibility of this bounded shared-template search
with hard zero collateral. #366 is the next formulation, not a claim that the template family is dead.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; `docs/operating_manual_craft_handoff.md`; v7.5/v8 specs;
v13/v14 receipts and memos; #549 joint inverse-solve memo; G4 stationarity equations; frozen n600
target cache; `reports/latest.md`; lane/task/progress state; operator directives dated 2026-07-19.

Pointer `0.1910828242 [contest-CPU]` unchanged. MAIN landing review is required.
