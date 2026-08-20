# DDM EKH1 Worktree Residue Harvest Receipt

Generated UTC: 2026-08-05T22:12:26Z

Charter: `.omx/tmp/codex_runs/ekh1_prompt.md`
Common contract: `.omx/tmp/codex_runs/_common_contract.md`
Target worktree: `.omx/tmp/codex_worktrees/einstein_kolmogorov_crux_20260719T212159Z`
Target branch: `codexwt/einstein_kolmogorov_crux_20260719T212159Z`

## Verdict

The charter's expected uncommitted-residue state is no longer live.

`git -C .omx/tmp/codex_worktrees/einstein_kolmogorov_crux_20260719T212159Z status --short --untracked-files=all`
returned no status entries. There were no staged, unstaged, or untracked files to copy from that worktree in the normal Git inventory.

The branch is not merged. The live history already contains:

- `57e4c4e52b` - `einstein-kolmogorov: harden canonical producer identity (#876)`.
- `01f2115062` - `ek2 merge-debt CLOSED as superseded-by-main`, explicitly recording this branch as provenance-only and never-merge.

This EKH1 run did one fix-forward action on main: registered the already-landed EK canonical equation in the live canonical-equations registry. Before this run, the source module built, but `tools/list_canonical_equations.py --equation-id einstein_kolmogorov_crux_action_rate_contract_v1` returned no row.

## Inventory

### Normal worktree status

| Scope | Command | Result | Disposition |
|---|---|---|---|
| Target worktree uncommitted files | `git status --short --untracked-files=all` | empty | NO-OP: no live uncommitted files to harvest |
| Target worktree staged diff | `git diff --cached --stat` | empty | NO-OP |
| Target worktree unstaged diff | `git diff --stat` | empty | NO-OP |

### Ignored artifacts observed

These were observed only through `git status --short --untracked-files=all --ignored=matching`. They are ignored run-output residue, not normal uncommitted status entries. EKH1 did not delete, move, cold-store, or commit them.

| Path root | Observed size | Disposition |
|---|---:|---|
| `.omx/research/einstein_kolmogorov_crux_closure_v2_20260720/` in the target worktree | 1.5M | OBSERVED-IGNORED-NOT-HARVESTED |
| `.omx/research/einstein_kolmogorov_crux_runs_final_20260719/` in the target worktree | 28M | OBSERVED-IGNORED-NOT-HARVESTED |

This is a bounded observation, not a deletion or certification claim. The earlier EK2 receipt separately certified two different untracked run-output directories to SSD cold store under `/Volumes/VertigoDataTier/pact/cold_store/ddm_ek2_worktree_harvest_20260810`.

### Branch delta against main

`git diff --name-status main...codexwt/einstein_kolmogorov_crux_20260719T212159Z` still reports 19 branch-delta paths. They are not uncommitted residue. They are branch commits already adjudicated by main commit `01f2115062`.

| Path group | Representative paths | EKH1 disposition |
|---|---|---|
| EK2 disposition package | `.omx/research/ddm_ek2_worktree_harvest_20260810/*` | ALREADY-HARVESTED on main by `01f2115062` |
| Small xi_bridge receipts | `.omx/research/einstein_kolmogorov_xi_bridge_*.json` | ALREADY-HARVESTED on main by `01f2115062` |
| EK research memos | `.omx/research/einstein_kolmogorov_*` memos | SUPERSEDED/ALREADY-MAIN per `01f2115062` |
| Governing/docs/code hardening | `src/tac/canonical_equations/einstein_kolmogorov_crux_20260719.py`, `src/tac/canonical_equations/registry.py`, `src/tac/preflight.py`, three #154/#344/#351 tests, `docs/meta_bug_class_catalog.md`, `CLAUDE.md`, `uv.lock` | SUPERSEDED-BY-MAIN; do not merge the branch |

## Canonical Equation Verification

### Source module

Command:

```text
PYTHONPATH=src .venv/bin/python - <<'PY'
from tac.canonical_equations.einstein_kolmogorov_crux_20260719 import (
    EQUATION_ID,
    build_einstein_kolmogorov_crux_action_rate_contract_v1,
)
eq = build_einstein_kolmogorov_crux_action_rate_contract_v1()
print(eq.equation_id, eq.equation_id == EQUATION_ID)
print(len(eq.empirical_anchors))
print(eq.canonical_producers)
print(eq.canonical_consumers)
print(eq.domain_of_validity)
PY
```

Result summary:

```text
equation_id einstein_kolmogorov_crux_action_rate_contract_v1
expected_id_match True
anchors 2
canonical_producers:
  .omx/research/einstein_kolmogorov_crux_measurement_20260719.json
  .omx/research/einstein_kolmogorov_frontier_magnitude_chart_20260720.json
canonical_consumers:
  tools.probe_einstein_kolmogorov_crux
  tac.optimization.einstein_kolmogorov_crux
research_only True
promotion_eligible False
full_archive_claim False
```

Producer hashes matched the constants in the module:

| Producer | Bytes | SHA-256 | Match |
|---|---:|---|---|
| `.omx/research/einstein_kolmogorov_crux_measurement_20260719.json` | 20,898 | `0b2e02e39601f863d07465bca66e006f7bad503b64c9ab3f901b44bed9637451` | yes |
| `.omx/research/einstein_kolmogorov_frontier_magnitude_chart_20260720.json` | 18,418 | `1c5926d8e899b32a0ef46c13cfd32f0d6f1f9585cc7435cf52a7605720927ae6` | yes |

### Registry state

Before EKH1 registry append:

```text
PYTHONPATH=src .venv/bin/python tools/list_canonical_equations.py --equation-id einstein_kolmogorov_crux_action_rate_contract_v1
(no canonical equations registered yet OR no match for filter)
```

Fix-forward action:

```text
PYTHONPATH=src .venv/bin/python - <<'PY'
from tac.canonical_equations.einstein_kolmogorov_crux_20260719 import (
    populate_einstein_kolmogorov_crux_action_rate_contract_v1,
)
populate_einstein_kolmogorov_crux_action_rate_contract_v1(
    agent="codex",
    subagent_id="ddm_ekh1",
)
PY
```

After EKH1 registry append:

```text
# Canonical equations registry (1 entries)

equation_id: einstein_kolmogorov_crux_action_rate_contract_v1
  anchors:        2
  well-calibrated: True
  last calibrated: 2026-07-20T03:38:39Z
  consumers (2): tools.probe_einstein_kolmogorov_crux, tac.optimization.einstein_kolmogorov_crux
  producers (2): .omx/research/einstein_kolmogorov_crux_measurement_20260719.json, .omx/research/einstein_kolmogorov_frontier_magnitude_chart_20260720.json
  residuals:
    [OK] fixed_byte_palette_no_regression: residual=0.0000
    [OK] frontier_magnitude_chart_arithmetic: residual=0.0000
```

Direct registry query:

```text
registry_latest_count 423
target_count_latest 1
duplicate_latest_ids 0
target_present True
consumer_query tools.probe_einstein_kolmogorov_crux ['einstein_kolmogorov_crux_action_rate_contract_v1']
consumer_query tac.optimization.einstein_kolmogorov_crux ['einstein_kolmogorov_crux_action_rate_contract_v1']
producer_query .omx/research/einstein_kolmogorov_crux_measurement_20260719.json ['einstein_kolmogorov_crux_action_rate_contract_v1']
producer_query .omx/research/einstein_kolmogorov_frontier_magnitude_chart_20260720.json ['einstein_kolmogorov_crux_action_rate_contract_v1']
```

The registry file already had one unrelated uncommitted append (`pose_null_subspace_is_ac_only_v1`). EKH1 preserved it as outside this run's ownership. The intended commit contains the EK registry row only, not the pre-existing `pose_null` row.

Serializer note: the first patch-mode commit attempt failed before commit creation with:

```text
git_apply_rc=128
error: unable to create temporary file: Operation not permitted
error: unable to create backing store for newly created file .omx/state/canonical_equations_registry.jsonl
```

The retry path isolated the registry working tree to `HEAD + EK row`, then attempted normal serializer staging with post-edit SHA declarations. That also failed before commit creation:

```text
git_add rc=128
error: unable to create temporary file: Operation not permitted
error: .omx/state/canonical_equations_registry.jsonl: failed to insert into database
error: unable to index file '.omx/state/canonical_equations_registry.jsonl'
fatal: updating files failed
```

After that failed attempt, EKH1 restored the unrelated `pose_null` append in the working tree. The EK registry row is verified in the working tree, but the serializer commit is blocked by the managed-sandbox Git object/index write failure unless a later writable Git context lands it.

## Tests

Focused canonical-equation suite:

```text
PYTHONPATH=src .venv/bin/python -m pytest src/tac/canonical_equations/tests/test_einstein_kolmogorov_crux_20260719.py
29 passed in 1.93s
```

No `.py` file was edited by EKH1, so no review-tracker pass was required for this run. The prior EK2 receipt records two review-tracker passes for the Python files it edited.

## RECALL EVIDENCE

Required files read or queried:

```text
PROGRAM.md
CLAUDE.md
AGENTS.md
docs/operating_manual_craft_handoff.md
.omx/state/main_hot_state.md
.omx/tmp/codex_runs/ekh1_prompt.md
.omx/tmp/codex_runs/_common_contract.md
```

Memory lookup:

```text
rg -n "ekh1|common_contract|codex_runs|charter|contract" /Users/adpena/.codex/memories/MEMORY.md
```

Repo recall beyond the charter seed:

```text
git log --oneline --decorate --max-count=40 --all --grep='ek2\|einstein\|876\|canonical-equations EK\|worktree residue harvest'
git log --oneline --decorate --max-count=80 | rg -n "ek2|einstein|876|57e4c4e52b|canonical-equations|worktree"
git diff --name-status main...codexwt/einstein_kolmogorov_crux_20260719T212159Z
rg -n "einstein_kolmogorov_crux_action_rate_contract_v1" .omx/state/canonical_equations_registry.jsonl
PYTHONPATH=src .venv/bin/python tools/list_canonical_equations.py --equation-id einstein_kolmogorov_crux_action_rate_contract_v1
```

Findings beyond the charter:

- The target worktree is clean in normal Git status; the original "~17 UNCOMMITTED files" condition is stale.
- The branch's current head is `81dfb0ee68`, whose message says EK2 landed the work on the branch after a sandbox Git-ref blocker.
- Main commit `01f2115062` already closed EK2 merge debt as superseded-by-main and explicitly marked the branch as provenance-only, never-merge.
- The live main module imported and built, but the live canonical-equations registry did not contain the EK equation row until EKH1 appended it.
- The target worktree still contains ignored run-output directories totaling about 29.5M; EKH1 did not move or delete them.

What this changed:

- Do not copy or merge branch code onto main; doing so would fight the already-landed main variant and the explicit never-merge hazard.
- Do append the missing EK canonical-equations registry event so the U2 equations leg is queryable on main.

## Score And Launch Boundary

No scorer run, exact replay, CUDA dispatch, Modal dispatch, paid job, merge, or branch checkout was performed by EKH1.

## Commit Status

No commit was created by EKH1. Both serializer attempts that included `.omx/state/canonical_equations_registry.jsonl` failed before commit creation on Git write operations. A narrower docs-only serializer attempt also failed before commit creation:

```text
git_add rc=128
error: unable to create temporary file: Operation not permitted
error: .omx/research/ddm_ekh1_20260805/.done: failed to insert into database
error: unable to index file '.omx/research/ddm_ekh1_20260805/.done'
fatal: adding files failed
```

This confirms the blocker is not registry-specific. The shared index is still unstaged (`git diff --cached --name-status` empty after the failures).

Working-tree artifacts left for resume:

```text
.omx/state/canonical_equations_registry.jsonl
.omx/research/ddm_ekh1_20260805/RECEIPT.md
.omx/research/ddm_ekh1_20260805/NEXT_IF_RESUMED.md
.omx/research/ddm_ekh1_20260805/.done
```

Own-vehicle frontier line required by the common contract:

```text
S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]
```

Contest pointer remains borrowed/external and unmoved by EKH1. No new `archive.zip` score exists from this run.
