# Task #555 implementation spec — LandingDiffManifest

Date: 2026-07-19  
Lane: `lane_landing_diff_manifest_20260719` (L0 / phase 1)  
Authority: verified delegated prompt SHA-256
`6e734799f232172cb6e8e3c2b3c8c5a9c9bf58d71bc2fd97dc8f08d0b950d025`.

## Objective and authority boundary

Build a typed, deterministic receipt over an arm's terminal `BASE..HEAD` Git
diff so every path has an explicit landing disposition. Extend the existing
landing-review disposition command; do not fork the landing or harvest
apparatus. This is apparatus-only MEANS work. It has no launch, paid dispatch,
score, archive, pointer, or promotion authority. Pointer
`0.1910828242 [contest-CPU Linux x86_64]` remains unchanged. Never write
`experiments/results/levelset_n600_witness_20260717T113932Z/`.

The implementation follows the `LandingDiffManifest` design in
`.omx/research/poly_functors_interaction_crosswalk_20260719_codex.md`, composes
with `tools/codex_harvest_commit.py` and
`tools/codex_landing_review_gate.py`, and must preserve the rc=13 gitignored
and rc=14 append-doc shrink protections in
`tools/subagent_commit_serializer.py`. The evidence/review discipline is the
one in `docs/operating_manual_craft_handoff.md`: re-derive from Git objects,
label known versus assumed, and attack the result before handoff.

## Owned files

The bounded implementation process owns only:

1. `src/tac/landing_diff_manifest.py` (new)
2. `tools/codex_landing_review_gate.py` (extend)
3. `src/tac/tests/test_landing_diff_manifest.py` (new)
4. `src/tac/tests/test_codex_landing_review_gate.py` (extend)

It must not edit the final research memo, lane registry/state, checkpoint
ledger, serializer, harvest tool, crosswalk, reports, or any experiment path.
It must not commit. The parent performs independent review and commit.

## Typed model and serialized contract

Implement dependency-free stdlib dataclasses/enums in
`tac.landing_diff_manifest`:

- `PathDisposition`: serialized values exactly `merged`,
  `intentionally-dropped`, `deferred`, `UNACCOUNTED`.
- `PathChange`: Git status, destination/current path, optional old path for
  rename/copy, base content SHA-256 when present, head content SHA-256 when
  present, head gitignored flag, disposition, optional reason, optional named
  consumer, and whether the path is a findings/memo artifact.
- `LandingDiffManifest`: schema/version, fully resolved 40-hex base and head
  SHAs, deterministic tracked diff SHA-256, ordered per-path records, blockers,
  completeness boolean, and generation metadata that does not introduce
  nondeterministic receipt bytes. A receipt rebuilt from unchanged Git objects
  and declarations must serialize byte-identically.
- A small typed declaration value for caller-supplied path dispositions is
  acceptable when it keeps parsing strict.

Use Git object bytes, not current file testimony, for tracked content hashes.
Parse `git diff --name-status -z -M -C BASE HEAD --` without lossy whitespace
splitting. Support add, modify, delete, type change, rename, and copy. The
destination path is the classification key for rename/copy; preserve the old
path and both available content hashes. Sort deterministically by path.

Resolve and validate both revisions with Git. Reject a non-repository,
unresolvable revisions, malformed declarations, duplicate normalized paths,
absolute paths, `..` escapes, or a declaration for a path outside the diff.
An empty diff is a complete empty manifest. Do not silently treat uncommitted
worktree state as `HEAD`; the receipt's authority is the committed terminal
range. The API may expose untracked discovery separately, but it must never
launder an untracked/ignored path into the committed range.

Default every changed path to `UNACCOUNTED`; there is no bulk implicit
`merged` default. Enforce:

- `merged`: no reason required.
- `intentionally-dropped`: non-placeholder reason required.
- `deferred`: non-placeholder named consumer required.
- `UNACCOUNTED`: always a blocker.
- A findings/memo artifact requires a non-placeholder named consumer unless
  its disposition is `deferred` with that named consumer. Treat Markdown under
  `.omx/research/` and paths whose basename identifies a findings/memo artifact
  as this class. A global disposition consumer may be copied into each such
  record by an explicit caller option; it must appear per path in the receipt.
- A tracked path ignored by the HEAD checkout's Git rules is recorded and is a
  blocker, preserving the rc=13 sharp-edge signal. Detection must use
  `git check-ignore --no-index` or an equivalently explicit check so a forcibly
  tracked ignored path is not missed.

Expose `to_dict`/`from_dict`, strict JSON load/write helpers with atomic
replacement, and a pure blocker validator. Unknown schema, unknown status,
missing required fields, a path-set mismatch, content-hash mismatch, or a
receipt claiming `complete=true` while blockers exist must fail closed.
Provide a small module CLI able to build a receipt from `--repo`, `--base`,
`--head`, a declarations JSON file, and `--output`; return nonzero when the
result is incomplete while still writing the inspection receipt.

## Landing gate integration

Extend only the existing `disposition` command. Add options equivalent to:

- `--landing-manifest <json>`
- `--landing-manifest-strict` (explicit strict flip)
- `--landing-manifest-waiver <real rationale>`

The precise flag spellings may vary only if tests and help make them obvious.
The default is WARN-ONLY because historical landing rows do not yet carry
these receipts and the required two-arm retro is expected to measure that
gap. Do not silently flip strict in this branch.

For terminal custody states `reviewed_committed` and `closed`:

- missing manifest, invalid/tampered manifest, `UNACCOUNTED` paths,
  consumer-less findings/memos, gitignored changed paths, or other manifest
  blockers produce a loud warning in default mode;
- the same conditions refuse before ledger append in strict mode;
- a same-command waiver with a real non-placeholder rationale permits the
  disposition, but the ledger row must record the waiver and the manifest
  blocker summary;
- the receipt base/head/path summary and validation mode are copied into the
  ledger row so inspection does not depend on a transient CLI line;
- `respawned` and `held_entangled` retain existing semantics and are exempt;
- the existing `--consumed-by` contract is preserved and never weakened.

Use an import arrangement that works when the tool is executed directly from
the repository without installation (add `src` to `sys.path` narrowly if
needed). No subprocess shell strings.

## Tests and acceptance

Add at least 15 new behavioral tests (more is preferred) using temporary Git
repositories and no network. Required coverage:

1. fully merged positive diff;
2. unaccounted negative;
3. dropped-with-reason positive and missing/placeholder reason negative;
4. deferred-with-consumer positive and missing/placeholder consumer negative;
5. findings memo with per-path consumer positive;
6. consumer-less findings memo negative;
7. explicit global consumer copied per path;
8. rename with old/destination path and content hashes;
9. delete with base hash and null head hash;
10. forced-add gitignored path detected and blocked;
11. empty diff complete;
12. declaration outside diff rejected;
13. deterministic JSON/receipt rebuild;
14. malformed/unknown receipt rejected;
15. tampered hash/path-set/complete claim rejected;
16. gate warn-only appends a row with blocker summary;
17. gate strict refuses before append;
18. real waiver allows and is recorded;
19. placeholder waiver refuses;
20. terminal positive manifest succeeds;
21. nonterminal/respawn compatibility;
22. existing landing-gate tests remain green.

Run at minimum:

```text
python3 -m pytest -q \
  src/tac/tests/test_landing_diff_manifest.py \
  src/tac/tests/test_codex_landing_review_gate.py
python3 -m py_compile \
  src/tac/landing_diff_manifest.py \
  tools/codex_landing_review_gate.py \
  src/tac/tests/test_landing_diff_manifest.py \
  src/tac/tests/test_codex_landing_review_gate.py
```

No test may depend on the operator's main worktree or mutate repository state.

## Parent-owned completion after implementation

The parent will independently review the diff, repair defects if necessary,
run the focused and relevant regression suites, and generate two retro receipts
without inventing historical dispositions:

- integer-plane spec: `4db3b50a43^..4db3b50a43`;
- yhat-native arm: `632343535f^..632343535f` (landed via merge
  `9b25ba3ce0`).

Because those historical arms did not emit typed per-path declarations, the
retro default is `UNACCOUNTED`; report the exact path/blocker counts and keep
the live gate WARN-ONLY unless the measured count is actually zero. The parent
then writes `.omx/research/landing_diff_manifest_20260719_codex.md`, performs
two review-tracker passes on every touched Python file, commits through
`tools/subagent_commit_serializer.py` with post-edit content hashes and a
`[no-triality]` message, checkpoints, and requires independent MAIN review.
