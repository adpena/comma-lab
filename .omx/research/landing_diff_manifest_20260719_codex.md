# LandingDiffManifest — task #555 landing memo

**Date:** 2026-07-19
**Lane:** `lane_landing_diff_manifest_20260719`
**Verdict scope:** apparatus-only MEANS; no launch, paid dispatch, archive,
score, pointer, or promotion authority.
**Consumer:** task #555 and the required independent MAIN landing review.

## Verdict

The typed `BASE..HEAD` receipt and landing-disposition integration are built
and locally verified. Every changed Git path defaults to `UNACCOUNTED` and can
only become `merged`, `intentionally-dropped` with a real reason, or `deferred`
with a named consumer through an explicit declaration. Research findings/memos
also require a per-path consumer. Receipts bind resolved base/head commits,
deterministic diff bytes, old/new per-path content hashes, rename/delete state,
HEAD-tree ignore rules, dispositions, and blockers. Serialized receipts reject
duplicate JSON keys, unknown schema fields, contradictory completeness, and
Git-object drift.

`tools/codex_landing_review_gate.py disposition` now re-derives the receipt
before a terminal ledger append, records its SHA-256 and blocker/path details,
and supports explicit strict refusal plus a same-command real-rationale waiver.
The migration default remains **WARN-ONLY**; neither rc=13 nor rc=14 serializer
protection was changed or weakened.

## Required retro live-count

No historical dispositions were invented. Both arms were run with empty
declarations against their actual arm commits:

| Arm | BASE..HEAD | Changed paths | `UNACCOUNTED` / blockers | Receipt SHA-256 |
|---|---|---:|---:|---|
| `integer_plane_vehicle_spec` | `99ca1f1b1362788c82bc281e4e7d2e997e50d905..4db3b50a437a606b824124cd91a8a668fa9c98cf` | 2 | 2 / 2 | `abc32443fcdae43c052da037c3915ca118a462a4c65d2d4fc3914d051c802868` |
| `yhat_native` arm (landed by merge `9b25ba3ce0`) | `e8c9973d6a4f09d26eae5840e2c33c8b1ca9a747..632343535f8ac0785ea6f1f6d2f3892cfb0c8923` | 12 | 12 / 12 | `335051c82ed3d7c6469cb721e4d1a503eb0f8651c02e6ef15ad4d71e3fd11f5a` |

The measured live count is therefore **14**, not zero. Strict-by-default is
not authorized in this landing. The durable receipts are
`.omx/research/landing_diff_manifest_retro_integer_plane_vehicle_spec_20260719.json`
and `.omx/research/landing_diff_manifest_retro_yhat_native_20260719.json`.

## Verification and review

- **MEASURED:** 69 tests passed across the new manifest tests, landing-gate
  regression suite, and consolidation-debt seam test.
- **MEASURED:** Ruff, `py_compile`, and `git diff --check` passed on every
  touched Python surface. (MAIN review correction 2026-07-19: the markdown
  deliverables carried trailing whitespace at landing — the arm's 5-round cap
  fired before a sixth fix round; stripped at MAIN review, `git diff --check`
  now clean repo-wide on this landing.)
- The new/extended behavioral tests cover positive, negative, waiver, strict,
  migration warning, rename, delete, forced gitignored add, empty diff,
  deterministic rebuild, consumer custody, malformed receipt, and tamper
  cases. The task asked for at least 15; 37 new collected test cases
  were added to the focused surface.
- The global lane validator still reports 110 pre-existing missing-evidence
  errors. It reports no error for `lane_landing_diff_manifest_20260719`; the
  ambient debt is not represented as a green global validation claim.

Self-review cap was five rounds:

1. **NOT CLEAN:** HEAD ignore status depended on the current checkout; fixed by
   evaluating committed `.gitignore` bytes in a context-managed scratch repo.
2. **NOT CLEAN:** the disposition ledger did not bind receipt bytes, relative
   paths depended on caller CWD, and duplicate JSON keys could collapse; all
   three custody classes were closed.
3. **NOT CLEAN:** immediate warnings named blocker codes but not affected
   paths; path-bearing details now reach stderr and the ledger.
4. **NOT CLEAN:** the memo undercounted new collected tests as 32; collection
   proved 27 new manifest cases plus 10 new gate cases, and the claim was fixed.
5. **CLEAN:** source, test, receipt, gate seam, and memo re-derivation found no
   further defect. The five-round cap is exhausted.

Every touched `.py` carries both `landing_diff_manifest_pass1_20260719` and
`landing_diff_manifest_pass2_20260719` review-tracker marks. The tracker query
returned both passes for all four files (40/40, 19/19, 25/25, and 34/34 tracked
entities respectively). The exact serializer result is recorded in the final
branch handoff, not pre-claimed here.

## Stores consulted

- Verified delegated authority prompt and live per-arm/broadcast inboxes.
- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, and
  `docs/operating_manual_craft_handoff.md` (especially re-derive from primary
  artifacts, label evidence, and attack the conclusion before handoff).
- `.omx/research/poly_functors_interaction_crosswalk_20260719_codex.md`.
- `tools/codex_harvest_commit.py`, `tools/codex_landing_review_gate.py`, and
  the rc=13/rc=14 guards in `tools/subagent_commit_serializer.py`.
- `codex_arm_disposition_at_done_marker_not_worktree_snapshot_20260718.md` and
  `codex_findings_disposition_is_not_consumption_bug_class_20260717.md` from
  the canonical operator memory store.
- `reports/latest.md`, lane/subagent state, latest sister findings/session
  memos, and the two retro arm commits above.

## Triality, pointer, and custody

This is workflow apparatus, so the commit is explicitly `[no-triality]`: it
does not add a witness DSL lever, scientific DAG edge, canonical equation,
sensitivity signal, Pareto constraint, bit allocation, or dispatch hook. Its
typed receipt is the custody surface consumed by the existing landing gate.

- Pointer before: `0.1910828242 [contest-CPU Linux x86_64]`.
- Pointer after: `0.1910828242 [contest-CPU Linux x86_64]`.
- Delta: exactly zero; no evaluator or score run occurred.
- Sacred run `experiments/results/levelset_n600_witness_20260717T113932Z/` was
  not written.
- Independent MAIN review of the complete branch diff is required before
  merge; branch-local tests and review marks are not promotion authority.
