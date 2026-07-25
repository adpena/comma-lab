---
title: DDM MR1 independent-approver serial-merge findings
date_utc: 2026-07-25
reviewer: mr1-independent-approver
base_commit: 8a7220238c
main_landing_review_required: true
score_claim: false
pointer_moved: false
---

# Verdict

`MERGE-WORTHY AFTER FIXES; MAIN REVIEW REQUIRED`.

RG4, J12, and MS2RP were independently rederived and merged in the required
serial order on the isolated
`codexwt/ddm_mr1_independent_approver_merge_20260725T193432Z` branch. This
branch is forensic until MAIN reviews and lands it. No source branch was
treated as a source of truth, and no whole-file ancestor copy was used.

# SHA-pinned serial merge custody

| Order | Reviewed source tip | Integration merge | Parents | Disposition |
|---|---|---|---|---|
| RG4 | `cb34bbb0f119f790015c2561e2b57d0470580537` | `e12f832466ea6a3a30c5beff1be5f5643648282d` | `8700f100c706828f8e2433b46fbc9ea3b61ffb03` + `cb34bbb0f119f790015c2561e2b57d0470580537` | merge-worthy after custody fixes |
| J12 | `0fc6659e2e4846ba1966ec129ae7cb860a49bb39` | `02dee7099ff0ba3ebe61dbff0a8420f03ad01a37` | `e12f832466ea6a3a30c5beff1be5f5643648282d` + `0fc6659e2e4846ba1966ec129ae7cb860a49bb39` | merge-worthy after PC1/J12 fail-closed fixes |
| MS2RP | `b7f7557110c6d16e97208378ca1f9ec666464e82` | `ec14320cb9834f851d37f2177f86d664866a0f37` | `02dee7099ff0ba3ebe61dbff0a8420f03ad01a37` + `b7f7557110c6d16e97208378ca1f9ec666464e82` | merge-worthy scoped blocker; RG4 prerequisite is an ancestor |

The custom lane-registry merge driver reported 110 missing evidence paths in
the pre-existing global registry. That is repository-wide apparatus debt, not
a semantic conflict introduced by these arms. Each merge's additive lane row
was preserved. No nontrivial code conflict occurred.

# Per-entity independent verdicts

## J12 and inherited PC1

1. `serialize_pc1_packet`: CLEAN. Typed immutable arrays, bounded packet
   geometry, and canonical zlib bytes rederived.
2. `parse_pc1_packet`: CLEAN. Header/length checks and canonical re-emission
   are exact.
3. `receive_pc1_camera_pairs`: CLEAN AFTER FIX. Inactive uint8 identity and
   deterministic realization are preserved; genuine nonfiniteness refuses.
4. `null_projector_from_full_column_rank_sketch`: CLEAN AFTER FIX. The
   full-column-rank implication is sound and coordinate IDs now require unique
   nonempty strings.
5. `null_projector_from_receiver_gram`: CLEAN AFTER FIX. The complete
   `J.T @ J` eigenspace construction remains symmetric/idempotent and has the
   same coordinate-ID custody.
6. `objective_gate_contradiction`: CLEAN AFTER FIX. Only a literal boolean
   auxiliary decision is admitted and it cannot override the realized joint
   objective.
7. `_measure_jacobians`: CLEAN AFTER FIX. Resumed pair IDs, exact array
   geometry, and finiteness revalidate before digest/Gram accumulation.
8. `_rehomed_endpoint`: CLEAN AFTER FIX. Resumed chunks revalidate schema,
   endpoint/packet/parent/archive identity, pair range, counts, class totals,
   and finite nonnegative Pose SSE.

The PC1 adapter build/parse/receive helpers are also clean after zero-effect
packet type validation became mandatory.

## RG4

- `src/tac/optimization/ddm_rg4_g3_blocks_and_active_tube.py`: CLEAN AFTER
  FIX. It now enforces unique/equal missing/residual key sets, exact
  actuator-magnitude by both-sign probe coverage, valid IDs and SHA custody,
  and refuses an empty active-tube batch.
- `tools/run_ddm_rg4_g3_blocks_and_active_tube.py`: CLEAN AFTER FIX. It now
  refuses malformed, nonfinite, or shape-inconsistent source/scorer batches
  before pricing.
- `src/tac/optimization/tests/test_ddm_rg4_g3_blocks_and_active_tube.py`:
  CLEAN. Adversarial incomplete-grid, duplicate-key, empty-batch, and
  malformed-scorer cases are covered.

The landed receipt revalidates 25 distinct obstruction keys, each with the
exact Cartesian product of its counted magnitudes and both signed
one-quantum directions.

## MS2RP

The JSON/Markdown-only branch is CLEAN at artifact level. All nine input
bindings and the receipt self-content digest validate. Independent
recomputation found:

- RG4: 1,200 assignment rows, 34 `RECOVERED_COMPLETE` incidence rows, and
  1,166 measured-no-event unrecoverable rows;
- PF3: 37 occupied buckets, zero fully materialized occupied buckets, and all
  five required materialization-field counts equal to zero;
- RD1: 0/162 finite same-object prices;
- MS2R-R3: zero measured Task-701 rungs.

Its exact verdict is therefore
`BLOCKED_NO_MATERIALIZABLE_PARTIAL_MEMBER; BOX_FALSIFIER_NOT_REACHED`, scoped
to `PRECONDITION/INSTANCE`. It is not a negative verdict on the describe
line, typed waterfilling, or a representation family.

# Independent-approver credentials

The DuckDB review tracker records the distinct reviewer identity
`mr1-independent-approver`:

- `mr1_rg4_clean_pass_1`, `_2`, `_3`: 47/47 distinct entities each;
- `mr1_j12_clean_pass_1`, `_2`, `_3`: 136/136 distinct entities each;
- total: 549 review events over 183 distinct Python entities.

MS2RP has no Python entities; its three artifact-level clean passes are
recorded in its receipt and dedicated review memo. The RG4 Python prerequisite
was credentialed before MS2RP was merged.

# PC1 warning disposition

The finite PC1 warp warnings were reproduced with `PYTHONWARNINGS=error` at
`rotation.T @ points`: the NumPy/Accelerate backend emitted divide, overflow,
and invalid warnings despite finite inputs and output. A direct `einsum`
control was byte-identical. The fix preserves the original matmul and confines
`np.errstate` to that operation, then fails closed on any nonfinite transform,
SE(3) result, inverse projection, or unrepresentable float32 grid. An extreme
transform regression test exercises the refusal path. This is sanitization of
a proven backend artifact plus explicit postcondition enforcement, not global
warning suppression.

# Verification evidence

- RG4 touched suite under `PYTHONWARNINGS=error`: 5 passed.
- J12/PC1 four-suite group under `PYTHONWARNINGS=error`: 30 passed, zero
  warnings.
- Preserved J12 rehome custody: 95/95 chunks across five endpoints pass the
  new semantic validator without scorer execution or artifact mutation.
- Ruff check/format, `py_compile`, JSON parse checks, source ancestry, merge
  parent order, and `git diff --check`: clean at the independent pass.

# Frontier and authority boundary

Historical imported rows that name `0.1910828242 [contest-CPU]` are
custody-specific local baselines, not the competitive frontier. The operator
correction received at `2026-07-25T19:52:29Z` establishes the current official
leaderboard best (displayed about `0.172`, reconstructed about `0.1721413`) as
the competitive reference. MAIN owns the separate canonical-pointer repair;
this branch deliberately makes no duplicate pointer edit.

No receiver/scorer execution, exact contest evaluation, archive promotion,
training, paid dispatch, reseal, READY/FIRE disposition, campaign fire, or
frontier mutation occurred. J12 merged-main/worst-geometry reseal remains a
separately governed MAIN/j-chain action after landing review.

STORES CONSULTED: delegated authority prompt; `CLAUDE.md`; `AGENTS.md`;
`docs/operating_manual_craft_handoff.md`; canonical lane registry, task status,
subagent progress, frontier/report, equation, probe, and council surfaces;
source-branch configs, receipts, DAG feeds, findings, and preserved J12 chunks;
git object/ancestry graph; DuckDB review tracker; per-arm and fleet inboxes.

# DAG FEED

`base 8a7220238c`
→ independent lane registration `8700f100c7`
→ RG4 source tip `cb34bbb0f1` via merge `e12f832466`
→ J12 source tip `0fc6659e2e` via merge `02dee7099f`
→ MS2RP source tip `b7f7557110` via merge `ec14320cb9`
→ this aggregate credential
→ **MAIN diff review and explicit landing required**
→ separately governed J12 reseal decision.

MAIN should review the merge-parent order, the independent fixes inside the
three merge commits, the review-tracker credentials, the scoped MS2RP blocker,
and integration with MAIN's separate canonical-pointer repair. It must not
infer READY/FIRE or a score claim from this landing.
