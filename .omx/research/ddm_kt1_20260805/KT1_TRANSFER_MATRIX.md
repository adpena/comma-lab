---
arm: ddm_kt1
title: "cross-domain transfer matrix: known in one domain, forgotten in another"
date_utc: 2026-08-05
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
axis: "[scorer-free audit/map]"
tokens: "[no-triality] [p0-ledger-ok]"
---

# KT1 transfer matrix

## Scope and receipts

This arm answers the operator question: **what do we already know in one domain but forget in another?**
It makes no fixes, fires no scorer, changes no frontier pointer, and writes only this map under
`.omx/research/ddm_kt1_20260805/`.

Governing reads completed: `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`,
`.omx/state/main_hot_state.md`, the common contract, and the KT1 charter.
Focused receipts consulted: TJ1, UF1, BD1, NG1, RG1, DRAFT #302/#315, SPEC v8, WF2, SQ1, MS8, DC1,
and `src/tac/negative_verdict_gate.py`. Broad corpus greps were treated as noisy; absence claims below are
bounded to the cited scopes and the seeded KT1 cells, not global non-existence claims.

Current own-vehicle line remains **S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]**.
Borrowed contest pointer remains **0.1910828242 [contest-CPU]**. Pointer delta: **none**.

## Denominators

Matrix denominator: **17 laws x 8 domains = 136 cells**.

| status | cells | percent |
|---|---:|---:|
| CURED | 31 | 22.8% |
| AD_HOC | 70 | 51.5% |
| ABSENT | 35 | 25.7% |

Domain columns: training-loops, terminal-solves, measurement-instruments, rate-coding, receiver/decode,
fleet/apparatus, memory/recall, calibration/authority.

Legend: `C` = cured structurally or close enough to be a reusable law in that domain; `A` = used, but ad hoc
or per-arm; `X` = absent in the bounded scope and owned in `transfer_matrix.jsonl`.

| # | law | train | terminal | measure | rate | receiver | fleet | memory | authority |
|---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | event-driven exits | C | X | A | A | A | A | X | A |
| 2 | trajectory-derived stopping | A | A | C | X | X | X | X | A |
| 3 | freshness at consumption | A | X | A | A | X | C | A | C |
| 4 | EMA at eval | C | X | A | X | X | X | X | A |
| 5 | eval-roundtrip in loop | C | A | C | C | C | A | X | C |
| 6 | positive controls per instrument | A | A | A | A | A | X | X | A |
| 7 | two-landing bug-to-class | C | A | A | A | A | A | A | C |
| 8 | races, not reputation | X | X | A | C | A | X | X | A |
| 9 | waterfill allocation | A | X | A | C | A | X | X | C |
| 10 | denominator reporting | A | A | C | C | C | A | A | C |
| 11 | resumability/checkpoints P0 | C | X | A | A | A | X | A | A |
| 12 | byte identity when inactive | A | A | A | C | C | A | X | A |
| 13 | operating-point thresholds | A | A | A | C | A | X | X | C |
| 14 | per-EDGE decomposition | C | X | A | A | A | X | X | X |
| 15 | verdict-scope ladder | A | A | C | A | A | A | A | C |
| 16 | staleness fail-closed | A | X | A | A | X | C | A | C |
| 17 | exchange-rate vs gap anchoring | A | A | A | C | A | A | X | C |

## Seed-cell verification

- **Resumability law:** training is structurally cured by the P0 checkpoint contract. Terminal solves and fleet
  arms are not: TJ1 has an explicit resume note waiting on SQ2, while the charter's seeded #878-style missing
  `NEXT-IF-RESUMED` problem remains a class to fix, not a single typo.
- **Byte identity:** receiver/rate surfaces are strongest. RG1 proves inactive extension byte identity; MS8 converts
  counted-but-inert receiver manifest bytes into consumed bytes without changing old archives. Training remains
  `AD_HOC`: there are many default-off claims and some trajectory-neutral tests, but I did not verify an all-lever
  default-off trajectory-neutral proof in the bounded source/test search, so I did not rely on the charter's #855
  numeric seed.
- **Per-EDGE:** v8 has the law in the seg carrier design. Pose still mostly reports per-pair/aggregate or per-DOF
  source concepts, not a current per-DOF/per-edge authority table; this is a high-ranked transfer.
- **Races-not-reputation:** rate coding has the cleanest adoption: #940-style per-surface coder races and MS8's
  exhaustive codebook race. Training optimizer and terminal solver choices remain too reputation-driven.
- **Positive controls:** SQ1/MS8/DC1 are good recent exemplars. NG1's audit says the control habit is still uneven,
  so the matrix marks the central/fleet/instrument transfer as `AD_HOC` or `ABSENT`, not cured.
- **Waterfill:** WF2 cured the rate/pricing distinction. Terminal solve iteration depth and fleet attention still
  lack a marginal-S allocator.
- **Two-landing:** bug-to-class exists in training and calibration/freshness. Positive wins and many audit findings
  still often stop at instance receipts.
- **EMA/smoothing:** training has the strict EMA-at-eval contract. Measurement noise and solver-best state need a
  separate "best snapshot versus live iterate" transfer.

## Ranked transfer builds

Ranking heuristic: **receiving-domain S leverage x low build cost**, with exact score movement still requiring a
separate measured row after this map.

| rank | receiving domain | forgotten law | cheapest transfer build | why first |
|---:|---|---|---|---|
| 1 | terminal-solves | event-driven exits + waterfill | wire `trajectory_stopping.py`/`allocate_adaptive_depths` into terminal GN/menu solvers with cap fallback | cheap code path, likely prevents wasted solver depth and routes toward lower exact rows |
| 2 | fleet/apparatus | waterfill allocation | scheduler ranks scorer/review slots by expected marginal S per minute and stale-row penalty | high leverage because live scorer/fleet slots are saturated |
| 3 | fleet/apparatus | positive controls per instrument | central canary registry with minimum positive/negative controls per measurement family | prevents false rows before they consume scarce authority |
| 4 | training optimizers | races-not-reputation | per-stage/per-surface optimizer race from identical checkpoints | Muon-style lessons should not globally transfer without matched stage evidence |
| 5 | calibration/authority | per-EDGE pose decomposition | m66-style pose per-DOF/per-edge table before pose aggregate shelves are priced | pose marginal is nonlinear and stale shelves are already known to drift |
| 6 | terminal-solves/fleet | resumability/checkpoints P0 | terminal solver state + machine-readable `NEXT-IF-RESUMED` before lane/slot ownership | avoids losing expensive partial solve state and prevents ownerless continuation |
| 7 | receiver/decode | freshness/staleness fail-closed | bind receiver proofs to current archive sha, grammar version, consumed sections; queue survival refresh on change | BD1-style receiver-closed proof is not survival proof on a moved candidate |
| 8 | memory/recall | freshness/exchange-rate anchoring | memory numeric claims carry baseline/archive/gap id and refuse stale numeric transfer | stops repeating old prices and thresholds after frontier/archive movement |

## NEXT-IF-RESUMED

1. Convert the top-ranked transfer into a small code patch: terminal solver stop policy adapter using
   `src/tac/optimization/trajectory_stopping.py`, with no scorer invocation and a tiny deterministic fixture.
2. In the same landing, require terminal-solver receipts to persist `NEXT-IF-RESUMED` and consumed baseline/archive
   shas. That closes the highest-overlap KT1 gap without touching scorer ownership.
3. Then build the fleet positive-control registry, seeded from SQ1/MS8/DC1 controls and NG1's negative audit.

## Boundaries

- No scorer forwards.
- No claim that any row below moved the exact pointer.
- No global absence claim beyond bounded scopes in the typed JSONL.
- No fixes landed in this arm by charter; this is the map and handoff.
