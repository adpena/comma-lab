# Delete obsolete `inflate.py` LOC restriction

`lane_id=lane_delete_inflate_loc_restriction_20260721T160814Z`  
`status=COMPLETE_AWAITING_MAIN_REVIEW`  
`research_only=false`  
`apparatus_only=true`  
`score_claim=false`  
`pointer=0.1910828242 [contest-CPU] UNMOVED`  
`MAIN_REVIEW_REQUIRED=true`

## Authority and intent

Delegated authority SHA-256:
`8ae0f039944c75d2674277fde712f4d6fc253025e4b8c867b4ae6d9105932cfa`.

Permanently remove every enforcement path for the former default-100/hard-200
line-count restriction on submission `inflate.py`. Keep line count only as
clearly informational telemetry where it already helps diagnostics.

MAIN's `2026-07-21T16:20:48Z` scope amendment also removes the historical
`inflate_runtime_loc_budget` and `bolt_on_loc_budget` fields from the Catalog
#124 representation-lane promotion gate. The fields remain legal/importable in
the `SubstrateContract`, lane-maturity, and Phase-1 packet data models, but no
longer authorize or block promotion.

The anti-fake boundary is unchanged: rule 118, Catalog #417
receiver-consumption bijection, `embedded_constants_audit`,
`archive_payload_manifest`, and `python_reference_equivalence_test` remain
authoritative.

## Acceptance contract

1. `preflight_all` no longer invokes the line-count check.
2. The compatibility scanner always returns `[]`, including for a 500-line
   `inflate.py`; historical constants remain importable.
3. Bundle, audit, and lifecycle tooling never fails, asserts, or changes exit
   status because of `inflate_py_loc`.
4. Focused tests, import smoke, and `preflight_all` smoke pass.
5. A repository grep finds no surviving LOC enforcement; any retained
   `inflate_py_loc` field is labeled informational.
6. A scoped diff proves the #417 and payload-cleanliness implementation paths
   were not weakened.

## Landing boundary

All edits and commits remain on
`codexwt/delete_inflate_loc_restriction_20260721T160814Z`. MAIN must review the
diff and merge boundary before treating the change as repository truth.

## Landed behavior

- `preflight_all` has no call to
  `check_submission_inflate_py_under_loc_budget`.
- The compatibility preflight function and standalone scanner always return
  `[]`; the legacy audit CLI always exits zero and labels itself retired.
- Submission bundle and lifecycle paths retain LOC only as informational
  compatibility metadata. The linter emits no LOC finding.
- The former formula-extinction row is a permanent zero-valued no-op and does
  not emit an atom even if its historical atom flag is requested.
- Direct source-length assertions across substrate/runtime tests now assert
  source presence or functional behavior. Reintroduction guards exercise
  500-line interpreter source.
- Catalog #124 now requires exactly:
  `archive_grammar`, `parser_section_manifest`, `runtime_dep_closure`,
  `export_format`, `score_aware_loss`, `no_op_detector_planned`.

## Verification receipts

- Modified-file Python compile: PASS.
- `git diff --check`: PASS.
- Expanded changed-test suite with four unrelated missing-custody fixtures
  deselected: `1303 passed, 19 skipped, 4 deselected`.
- The corresponding unfiltered run reached `1303 passed, 19 skipped`; its four
  failures were exclusively absent isolated-worktree fixtures:
  `upstream/videos/0.mkv` (two tests) and `submissions/a1/archive.zip` (two
  tests). No failure touched the changed LOC behavior.
- Core LOC/Catalog #124/bundle/linter suite: `220 passed, 2 skipped`.
- Catalog #417 receiver-bijection parity suites: `18 passed, 1 skipped`.
- `preflight_all(check_codebase=False, verbose=False, use_fs_cache=False,
  wall_clock_budget_s=None)`: PASS.
- Import smoke confirms historical constants `100`/`200`, permanent no-op
  results, and the exact six-field Catalog #124 tuple.
- Review tracker records two clean passes over all 65 modified reviewable
  Python files (`3047` entity marks per pass). Its canonical scan deliberately
  excludes files named `__init__.py`; the one modified package initializer,
  `submission_bundle_builder_consumer/__init__.py`, was manually diff-reviewed,
  exercised by the changed-file suite, and is explicitly held for MAIN review.
- Scoped diffs contain no modification to #417 receiver-bijection or the
  `embedded_constants_audit`, `archive_payload_manifest`, and
  `python_reference_equivalence_test` implementation paths.

Primary post-edit SHA-256 custody:

- `preflight.py`: `8235a82e4892b1dc17d1232de1e1547f6bcbfe8110059445da2ea45b65e9751f`
- `submission_inflate_loc_budget.py`: `a2676c0dc03c5d6276f26d29fa41a666eb323cdacacda5a991d090e402ce5946`
- `submission_packet/builder.py`: `27fe6ad20f23e1a15fe1ae45825208766786704170ea4d646875c54a1ffb64e2`
- `submission_packet/linter.py`: `1da9adeecf8372201cc2da21572fda67059fce62073bcb8a70a0a5bfee3e7fe2`
- `submission_bundle_builder_consumer/__init__.py`:
  `ec2ec6ef04bf42fa9938053abc4bc865512389f89728467dfbf946863b2fa701`
- retired formula compatibility helper:
  `04a83318f86713084bdba16c7ffcf93c4e5987e1832587c06fa5e53eb9251ea5`
- 500-line reintroduction guard:
  `3b7cb0ebeba08339a500522d8df868c53cf299c421dbea5f5eabc32cff35f255`
- Catalog #124 amended tests:
  `76b006aaae7ff6df98453a677312f87dd787b970995184ba0fd1f87311746264`

## Verdict

`MEASURED`: the obsolete `inflate.py` and bolt-on source-line restrictions no
longer participate in preflight, lint, bundle, lifecycle, formula/autopilot, or
representation-lane promotion decisions. Runtime wall-clock, dependency
closure, exact archive bytes, rule 118, Catalog #417 receiver consumption, and
payload-cleanliness gates remain binding. Pointer is unchanged.
