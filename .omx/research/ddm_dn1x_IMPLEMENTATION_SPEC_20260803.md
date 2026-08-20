# ddm_dn1x implementation spec — fail-open instruments (#904 + #899 residual)

## Objective

Land two scorer-free instruments whose positive controls prove that they can emit the negative:

1. #904: an AST/import-graph audit for cross-module declared/set-but-never-read values, fields,
   and argparse flags. The canonical positive control is
   `DirectDescriptionJointDescentMLXModule.margin_targets` in
   `src/tac/optimization/direct_description_joint_descent.py`; it is declared as an init parameter,
   assigned to `self.margin_targets`, externally constructed from multiple tool modules, and never
   loaded. The tool MUST assert this control before it emits any full-sweep result.
2. #899 residual only: current HEAD already closed the JSONL read-path bypass in commits
   `124a35cae4` and `ffbaa63960`. Do not duplicate or edit that fix. Add the still-missing warn-only
   preflight guard against future readers of `.omx/state/required_component_ledger.jsonl` that bypass
   the canonical validated read surface.

This is apparatus only: no score claim, no scorer, no run launch, no pointer mutation.

## Constraints and ownership

- Do NOT edit `.omx/research/ddm_pu2_pose_tail_floor_probe_20260803.md`.
- Do NOT edit `.omx/research/ddm_cr1_composition_row_827_20260801.md`.
- Do NOT edit `src/tac/witness_dsl/guarded_constant.py` or its owned surfaces.
- Do NOT edit `src/tac/witness_dsl/activation_ledger.py` or
  `src/tac/tests/test_build_completeness_grades.py`; #899's runtime/read fix is already committed.
- Preserve all unrelated dirty-worktree changes. Do not stage, commit, or revert anything.
- Do not run a scorer or any paid/heavy job.
- Do not use `REVIEW_GATE_OVERRIDE=1`.
- Use `apply_patch` for edits.

## #904 deliverables

Create:

- `tools/audit_cross_module_declared_never_read.py`
- `tools/tests/test_audit_cross_module_declared_never_read.py`

The production scope is exactly:

- recursive `src/tac/**/*.py`
- recursive `tools/**/*.py`
- top-level `experiments/*.py`
- exclude `experiments/results/**` even under `--no-ignore`

The output must state the dynamic denominator by root, parsed-file count, parse-error count/details,
production-vs-test counts, and excluded-results count. Never silently treat an unreadable/unparseable
file as clean.

Build an AST-based module/import graph, not regex-only search. At minimum support:

- constructor parameters assigned (possibly through a simple expression) to `self.<field>`;
- class/dataclass fields when a production class crosses a module boundary;
- argparse declarations and Namespace loads (`args.x`, `getattr(args, "x", ...)`);
- import aliases/re-exports sufficiently to identify external constructor/call sites;
- production reads separately from test-only reads;
- cross-module evidence: a hit is reportable only when there is at least one external importer,
  constructor/call site, or resolved import path of one or more hops. Single-file dead stores are out
  of scope;
- source locations for declaration, assignment, reads (including test-only), external call sites,
  shortest import-hop evidence, explicit-vs-defaulted constructor bindings, and a deterministic
  blast-radius rank (document the formula).

Be conservative about identity: unrelated classes sharing an attribute name must not count as a read
of each other. Simple type inference from `x = ImportedClass(...)` is sufficient; unresolved dynamic
uses must be reported as limitations/counters, not silently asserted absent.

The CLI must have a controls-only mode and a full JSON-output mode. The default/full path MUST execute
controls before computing or printing the sweep. Abort nonzero without sweep results if any control
fails.

Controls:

- positive: `DirectDescriptionJointDescentMLXModule.margin_targets` MUST be a hit;
- negative: three known-consumed fields from the same class MUST NOT be hits:
  `seg_targets`, `pose_targets`, `margin_hinge_weight` (all have real production loads);
- negative flag controls: choose and pin three argparse declarations with real executable reads in
  production (for example `verdict_batch`, `seed`, and a third verified dest in
  `experiments/train_levelset_witness_realized_through_R_mlx.py`). The test must assert the exact
  declaration and read locations rather than merely their names.

Tests must include: canonical control gate, synthetic two-module positive, consumed negative,
same-file-only exclusion, N-hop/re-export import chain, unrelated same-name attribute non-consumption,
test-only read classification, argparse consumed/unconsumed behavior, results-directory exclusion,
parse-error accounting, deterministic rank/output, and a mutation-style test proving removal or
misclassification of the positive control makes the control gate fail before sweep output.

## #899 residual deliverables

Edit minimally:

- `src/tac/preflight.py`
- add `src/tac/tests/test_check_required_component_jsonl_read_validation.py`

Add a warn-only check named clearly, e.g.
`check_no_unvalidated_required_component_jsonl_readers`. It scans production Python under the same
bounded roots for new raw readers of the required-component store/symbol that do not route through
one of the canonical validated surfaces:

- `read_required_components`
- `required_component_integrity_summary`
- `verify_required_component_row`

The canonical implementation in `tac.witness_dsl.activation_ledger` is exempt because it owns the
store. Detect at least literal `required_component_ledger.jsonl` reads and imports/uses of
`REQUIRED_COMPONENT_PATH` followed by raw `open`, `.open`, `.read_text`, line iteration, or JSON
parsing. AST must drive the decision; text may be a conservative trigger but cannot be the sole
proof. Return a list of violations, print the scanned denominator when verbose, and wire the check
into the normal all/codebase preflight as `strict=False`.

Same-line waiver:

`# REQUIRED_COMPONENT_JSONL_READ_OK:<substantive real rationale>`

The waiver applies only to the violating raw-read line. Reject empty/short rationales and placeholder
`<reason>`/`<rationale>`.

Tests must prove:

- positive control: a synthetic deliberately invalid raw reader is reported/refused when strict;
- canonical helper consumer passes;
- canonical owner passes;
- substantive same-line waiver passes;
- nearby-line waiver does not pass;
- placeholder waiver does not pass;
- verbose denominator is nonzero/non-vacuous;
- warn-only integration returns violations without raising, strict mode raises.

## Acceptance commands

All of these must pass:

```bash
uv run pytest -q tools/tests/test_audit_cross_module_declared_never_read.py
uv run pytest -q src/tac/tests/test_check_required_component_jsonl_read_validation.py
uv run pytest -q src/tac/tests/test_build_completeness_grades.py
uv run python tools/audit_cross_module_declared_never_read.py --controls-only
uv run python tools/audit_cross_module_declared_never_read.py --output-json /tmp/ddm_dn1x_hits.json
```

The full command must produce a nonempty hit list with `margin_targets` present and the six negative
controls absent. Do not write the final research memo; the parent will hand-verify and report the top
10 from the generated full list.
