# Contract-First Acquisition Readers - Codex Findings

Timestamp: 2026-05-31T10:30:12Z

## Verdict

The remaining contract-reader findings were not mathematically or contest-space complete before this slice: repair stack search, PR95 observer package extraction, public replay preflight, DQS1 skip metadata, and the bounded runner still had paths that could observe archive-like custody through legacy fields.

This slice moves those paths onto the shared archive-bound candidate contract and makes legacy archive/proof/readiness signals blocker evidence rather than promotion authority.

## What Changed

- Repair stack search now refuses raw archive/proof flags unless a selected shared contract owns candidate archive custody, runtime proof custody, and exact handoff readiness.
- PR95 observer package extraction now recognizes direct shared contracts and nested adapter packages instead of reading family-specific readiness shims.
- Public replay preflight now requires selected shared contract custody and matching archive bytes/SHA before runtime checks continue.
- DQS1 skip metadata now records selected contract validity, key, archive SHA/bytes, and contract blockers for observed candidates.
- Archive-bound contract audit now groups migration-required findings by family, stage, scope, and entropy position, and emits contest-space grounding requirements for every group.
- The autonomous repair floor loop now refuses non-contract candidate paths unless the work is explicitly contract migration or blocker work.

## Verification

- Ruff on all touched Python/test files: passed.
- Focused pytest across contract audit, repair stack search, PR95 observer extraction, public replay preflight, DQS1 skips, and bounded runner behavior: 17 passed.
- Lane maturity validation: 1563 lane(s) validated cleanly.
- Archive-bound contract audit over tracked research/result paths: passed with 0 blocking findings, 9093 migration-required findings grouped into 22 executable backlog groups, and 1 advisory JSON parse finding.

## Remaining Gaps

This is not an optimality proof. It is a custody and acquisition-grounding fix. The remaining 22 backlog groups still need migration into byte-closed contract emitters, and each score-lowering candidate still needs byte archive custody, contest inflate/runtime consumption proof, video/content-tree or runtime-tree custody, SegNet/PoseNet/rate axis evidence, exact CPU/CUDA replay or blocker, and posterior update evidence before spend or promotion.
