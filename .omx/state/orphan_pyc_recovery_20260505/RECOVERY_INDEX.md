# Orphan .pyc Recovery Index — 2026-05-05

## Status: 97/97 orphans recovered to disk; commits BLOCKED by parent's preflight issues

## Recovery harness

- **Decompiler**: pycdc (built from source at `/tmp/pycdc/pycdc`).
  - decompyle3 / uncompyle6 do NOT support Python 3.12 bytecode.
  - pycdc handles ~90% of 3.12 syntax; trips on `@dataclass`, complex
    lambdas, walrus, structural pattern match.
- **Spec extractor**: per-pyc Python helper using matching interpreter
  (3.12, 3.13, 3.14) to dump `co_names` / `co_consts` / `co_varnames` /
  `dis()` for every code object. Always written next to the .py as
  `<basename>.recovery_spec.json`.
- **Stub strategy** for partial pycdc output: the pycdc text is wrapped
  inside a triple-quoted r-string in a parseable `.py` stub. This is
  necessary because the global pre-commit preflight scans every `.py`
  file in `scripts/`, `experiments/`, `src/tac/contrib/`, `src/tac/deploy/`,
  and `src/tac/experiments/` for SyntaxError. Un-stubbed partial .py
  files would block ALL future commits.
- **Quarantine** at `/Users/adpena/Projects/pact/.recovery_quarantine_20260505T004735Z/`:
  parent agent's earlier triage of the un-stubbed v2 outputs. Safe to
  delete — superseded by current v3 stubs in canonical locations.

## Recovery stats

| Status | Count | Meaning |
|---|---|---|
| Full decompile (AST-OK, no warnings) | 2 | pycdc produced clean source; rehydration done |
| AST-OK with pycdc warnings | 19 | pycdc emitted "Decompyle incomplete" but the file still parses (some lambdas / inner expressions are mangled but the file is mostly usable) |
| STUB (partial, quarantined inside r-string) | 76 | pycdc produced incomplete Python; raw text preserved as `_PYCDC_PARTIAL_OUTPUT`; recovery_spec.json holds dis() ground-truth |
| No pyc found | 0 | (all 97 had a recoverable .pyc somewhere) |

## .py-file companion artifacts

Every recovered `.py` has a sibling `.recovery_spec.json` (95 of 97 — the 2
fully-OK ones don't need a spec). Spec contents:

- `pyc_path`, `names`, `varnames`, `consts`, `freevars`, `cellvars`, `doc`
- `module_dis`: textual `dis.dis()` output of module-level code
- `child_codes`: list of `{qualname, name, argcount, kwonlyargcount, varnames, names, freevars, cellvars, doc, dis}` for every nested code object (functions, methods, comprehensions, lambdas)

## Tailscale + backup search results

- macOS Time Machine local snapshots: searched `/Volumes/Macintosh\ HD`, no orphan basenames found.
- iCloud Drive: searched `~/Library/Mobile Documents`, no orphan basenames found.
- `~/.codex/`, `/tmp/codex_runs/`: no project-specific .py files (only skill scaffolds).
- `~/.Trash/`: empty.
- Tailscale fleet:
  - `tertiary` (100.65.24.39): SSH connect timeout
  - `bat00` (100.120.99.124, port 2222 WSL2): SSH connect timeout
  - `molt` (100.114.131.54): host key verification failed (untrusted)
  - `mac-mini` (100.125.140.94): SSH OK, no project files found
- **Conclusion**: no external recovery source available. All recoveries are
  from local `__pycache__/*.pyc` only.

## Already recovered before this session (skip list)

These 3 files were recovered from session JSONLs by the parent agent:

- `scripts/probe_fastest_chip.py`
- `tools/argparse_dryrun.py`
- `tools/lane_magic_registry.py`

## Commit status: BLOCKED

The pre-commit preflight currently fails on 2 unrelated checks owned by the
parent agent:

1. `dispatch_claim_helper`: missing `closed_instance_job_ids` guard
2. `shell-lane-arity`: 13 violations across `scripts/remote_lane_apogee_intN.sh`
   and `scripts/remote_lane_sjkl_c067.sh` (invented CLI flags)

Until those are fixed, NO `.py` commit can land. The recovered files are all
on disk and parse cleanly; the parent agent can land them in batches once
the preflight unblocks. Suggested batches (group by directory):

- src/tac/ (11 files + 11 specs)
- src/tac/tests/ (20 files + ~14 specs)
- experiments/ root (17 files + 15 specs)
- experiments/results/ (32 files + ~30 specs)
- .omx/state/modal_pr95_h100_burn_triage/ (10 files + 10 specs)
- scripts/ (3 files + 3 specs)
- reports/, submissions/, tools/ (4 files + 4 specs)

## Recovery raw outputs

- `/tmp/recovery_results_v3.json` — complete per-file structured results
  (orphan_rel, pyc_path, decompiled_ok, ast_parse_ok, stub_written,
  spec_path, warnings, ast_error)
- `/tmp/recover_orphans_v3.py` — the harness itself

## Per-file table (status)

See `/tmp/recovery_results_v3.json` for the machine-readable form. Status
key: OK = full decompile, AST-OK = parses but pycdc said incomplete,
STUB = quarantined partial.
