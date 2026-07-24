# DDM RD1 λ-continuation frontier — Codex findings

**Lane:** `lane_ddm_rd1_lambda_continuation_frontier_20260724`
**Evidence axis:** `[macOS-CPU frozen-scorer advisory]`
**Authority:** `research_only=true`, `score_claim=false`, `promotion_eligible=false`
**Pointer:** `0.1910828242 [contest-CPU]` **UNCHANGED**
**MAIN landing review:** **REQUIRED — review the complete base..branch diff**

## Outcome first

> **MEASURED, FORMULATION-SCOPED:** `MEASURED_RESTRICTED_N600_LAMBDA_FRONTIER;
> V19C_CURRENT_UNSUPPORTED; KNEE_OUTSIDE_R6_BOX;
> GLOBAL_LATTICE_AND_MENU_ARCHIVE_CLOSURE_OPEN`.

The restricted domain contains 110 own-problem n600 descriptions with exact
skeleton/fiber byte homes: 104 V19C archive-closed points, five Menu1
measurement-harness rows, and one fresh exact C1 archive-closed row. Seven are
Pareto-nondominated, four are supported by `R + λD`, and every one of the ten
continuation checkpoints was independently checked against the full 110-point
rank.

| λ-supported description | bytes | d_seg | d_pose | D |
|---|---:|---:|---:|---:|
| `v19c_admit_0005` | 137,823 | 0.026140611437 | 163.059414695664 | 42.99467712410129 |
| `scalar_gain_bias_12b_frame1` | 137,839 | 0.024444792006 | 159.395332998206 | 42.36882427951228 |
| `statistics_hard_analytic_composed_frame1` | 138,801 | 0.070519231160 | 36.618184778057 | 26.18780166344113 |
| `c1_exact_solved_n600` | 409,526,925 | 0.000151969062 | 0.000101843121 | 0.04710977519640476 |

The discrete crossover duals are 25.565115087888007,
59.452361128556376, and 15660952.118260449 bytes/D. The normalized knee
is the 138,801-byte Menu1 composed description. It is a complete
measurement-harness custody bundle, not a counted contest archive, and it fails
the R6 d_seg/d_pose box.

## Fresh exact anchor

The C1 raw receiver output was replayed through the frozen scorers at n600 with
four Torch threads:

- d_seg = `0.0001519690619574653` = `17,927 / 117,964,800`
- d_pose = `0.00010184312078531729`
- D = `0.04710977519640476`
- strict packet re-encode and both decoded planes are exact

The fresh local row differs from the preserved contest-CPU display by
`9.061957465287488e-09` d_seg and `3.120785317287881e-09` d_pose. This is
recorded as `MEASURED_AXIS_OR_BATCH_GEOMETRY_DRIFT`; the axes are not equated.

## Premise corrections and invalidation

1. V19C admission index is not its trial archive index. The accepted
   `stage_checkpoints/02_n600_decisions` rows are the authoritative mapping.
   Using admission index as a filesystem index was detected and repaired before
   the v2 result.
2. At λ=0, equal-rate ties must include realized distortion in the deterministic
   rank key. The corrector now reaches `v19c_admit_0005`, rather than an
   arbitrary same-byte identifier.
3. The first receipt
   `ddm_rd1_lambda_continuation_frontier_receipt.json`
   (`57f697db8f27c4f662728ca09e4718da49dda602f2ed5011b254c0fa97b13278`)
   is **INVALID/SUPERSEDED** because it persisted volatile exact SSD free-space
   bytes and did not carry the separate typed `S_composed` supplement. It is
   preserved as historical provenance only. The stable authority is
   `ddm_rd1_lambda_continuation_frontier_receipt_v2.json`
   (`cdfa9a400d9633ea7f8f698dee6d55c65ac478ddcfed3ec01e6d2e6cefe6bbae`).
   The latest canonical-equation event points only to v2.

## What landed

- Strict `CodedStream` and `MeasuredDescription` types that refuse donor
  conditioning, false byte partitions, non-n600 rows, or score authority.
- A deterministic finite-domain lower hull, 8–12 point λ ladder,
  neighbor-only continuation, full-rank verifier, crossover duals, and
  normalized-knee implementation.
- An SSD-preflighted, checkpoint-resumable measurement tool. Input hashes,
  scorer batches, λ points, and the knee custody bundle are immutable.
- A pure registered evaluator,
  `ddm_restricted_realized_lambda_continuation_v1`, plus empirical anchor.
- A ten-row typed R(D) supplement with arithmetic-only
  `S_composed = D + 25R/37,545,489`; every row remains `score_claim=false`.
- Regression tests for authority firewalls, two-type byte closure, neighbor
  continuation, evaluator registration, immutable checkpoints, receipt/bundle
  custody, stable storage receipts, and typed S composition.

## Review receipt

Three clean passes are recorded in
`.omx/research/ddm_rd1_lambda_continuation_frontier_20260724T011239Z/ddm_rd1_three_pass_review_receipt.json`.
They include:

1. lint plus seven focused tests and stable-v2 canonical registry lookup;
2. independent recomputation of all 110 distortions, byte partitions,
   nondominance, monotone-chain hull, three duals, ten full ranks, typed S rows,
   and knee ZIP custody;
3. two byte-identical end-to-end resumptions followed by a fresh lint/test and
   source-diff inspection.

The RD1 lane is internally consistent at L2 with only implementation,
real-archive empirical, strict-preflight, and three-clean-review gates marked.
The repository-wide lane validator still reports 110 historical missing-evidence
paths; none names this lane, and no unrelated registry debt was edited.

Stable hashes:

- v2 receipt: `cdfa9a400d9633ea7f8f698dee6d55c65ac478ddcfed3ec01e6d2e6cefe6bbae`
- typed frontier: `7266153f1984e220e69fa8fe04ed674a070cfffe3f176524e7c2985d212c2b1c`
- knee bundle: `0cd45580e778a53f31e52ff5b713b2ee541e62731acba747049b56b3fb76ce83`

## Verdict scope and remaining work

- `V19C_CURRENT_UNSUPPORTED` applies only to weighted-sum scalarization after
  this measured Menu1 pool is admitted. The point remains Pareto-nondominated;
  the V19C family is open.
- Menu1 is measurement-harness receiver-closed, not archive-closed. Its
  composed knee cannot be promoted.
- The full-rank claim covers only the SHA-custodied 110-point domain, not the
  global uint8 lattice.
- DR2b transfers direction only; its coordinates are not imported.
- MS1 was still active during the build and no donor-conditioned artifact was
  consumed. A future point must satisfy own stored-problem and archive closure.
- No paid dispatch, contest exact evaluation, training, frontier mutation, or
  pointer movement occurred.

## Reproduce

```bash
PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python \
  tools/measure_ddm_rd1_lambda_continuation_frontier.py \
  --config .omx/research/configs/ddm_rd1_lambda_continuation_frontier_20260724.json

/Users/adpena/.local/bin/ruff check \
  src/tac/optimization/ddm_lambda_continuation_frontier.py \
  src/tac/canonical_equations/ddm_lambda_continuation_frontier_20260724.py \
  tools/measure_ddm_rd1_lambda_continuation_frontier.py \
  tests/test_ddm_lambda_continuation_frontier.py

/Users/adpena/Projects/pact/.venv/bin/python -m pytest -q \
  tests/test_ddm_lambda_continuation_frontier.py
```

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`
- `.omx/research/ddm_scorer_native_doctrine_and_synthesis_20260723.md`
- `reports/latest.md`, `.omx/state/lane_registry.json`,
  `.omx/state/subagent_progress.jsonl`,
  `.omx/state/master_gradient_anchors.jsonl`,
  `.omx/state/modal_call_id_ledger.jsonl`,
  `.omx/state/cost_band_posterior.jsonl`, and
  `.omx/state/continual_learning_posterior.jsonl`
- latest sister Codex findings/session summary, latest T3 council/design memos,
  current 2026-07-23/24 directives, V19C/Menu1/DR2b/E3/C1 receipts, exact
  archives, stage checkpoints, and target-cache custody named by the typed
  config

MAIN must review the complete branch diff, including the append-only registry
events and the preserved invalid round0 receipt, before any landing.
