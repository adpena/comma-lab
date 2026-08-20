# `experiments/ddm_pq2_compress_e2e.py` — coherence and optimization verdict

`date_utc: 2026-08-20` · `owner: ddm_pq4` · `axis: byte-exact and scorer-free; no score claim`

## VERDICT

**COHERENT after 7 fixes, with 2 refusal-typed gaps registered.** The script's stage order,
guards and sanitization are sound and its `--help` is accurate. The one material defect was an
**over-promise at the default invocation**: with no flags it resolves the expected archive from
the canonical frontier pointer — which now reads jg5 — and then answered a structurally
impossible request with "pass `--recipe-json`". That advice is correct for a missing recipe and
false for a missing stage, and the packet's own accounting lists this entry point as
`ours-original`, so a reviewer may reasonably try the default invocation first.

## The expressibility boundary, stated

The script rebuilds the **token stream** (optionally plus a declared container repack) and
carries the other seven parsed sections through verbatim. That is the rr4 and sz1 shape, and
for those candidates the byte-close is genuine.

The jg5 candidate's chain also **re-decides content in sections the script copies**:

| Missing stage | Real builder |
|---|---|
| seg token edit solve over 573 pairs (writes the semantic stream) | `experiments/ddm_jg3_joint_solve.py` |
| splice of those edits into the br1 body → the jg4 candidate body | (jg4 composition step) |
| joint edit-admission waterfill, 455 of 573 admitted | `experiments/ddm_jg5_pose_resolve_on_edited_renders.py` |
| pose-carrier re-solve on the candidate's own renders + archive rebuild | `experiments/ddm_up3_carrier_splice.py::build_archive`, solver `experiments/ddm_br1_pose_basis_reorientation.py::gn_solve_pair` |

Chain read at source from `ddm_jg5_pose_resolve_on_edited_renders.py:1-80` and its pinned
constants, not from a memo summary.

## The 7 fixes

1. **Docstring said "THE THREE STAGES" and listed four headings.** Now "THE STAGES. Three
   phases — A, B, C — across four subcommands", with the reason stated: a small incoherence
   makes a reader stop trusting the larger claims.
2. **New docstring section: what the entry point expresses and what it does not**, so the
   boundary is in `--help` rather than discoverable only by running it.
3. **`NOT_EXPRESSIBLE` registry + `RebuildNotExpressible`**, keyed by archive sha256, naming
   the missing STAGES and their real builders with a receipt path. Two candidates registered:
   jg5 (`f3bce5d2…`) and ck1 (`35c318d5…`).
4. **Grammar guard wired BEFORE the cross-pin guard**, so the honest reason is not hidden
   behind a fixable-looking one. Verified: default invocation now refuses jg5 by name at rc=1
   before creating the store directory.
5. **The rc64 comment undercounted the bodies.** It said two distinct bodies wear the filename
   `rc64_backend.c`. `ddm_rv14f` hashed all **241** copies across the three custody roots on
   2026-08-19 and measured **four**
   (`reverse_engineering/rc64_backend_role_registry.json`). The fourth matters: one body is a
   **PR #138 `opal_v1` intake — a third party's source**. A filename-keyed search can reach it,
   and pinning it would silently build against foreign code. The comment now carries the
   measured four-row table with copy counts and that warning.
6. **`RC64_SHIPPED_MEMBER_SHA256` and `RC64_ROLE_REGISTRY` as module constants**, so a recipe
   author pins the measured decoder value instead of retyping a sha out of a comment. Verified
   against the live packet: `gen5_jg5_waterfill/runtime/entropy/rc64_backend.c` hashes to
   `05839d1416e68a49…`, exactly the constant.
7. **`verify_inputs` reports its denominator and refuses an empty spec.** It printed nothing
   before, so a loop over an empty or partly-unpinned spec passed silently — and a silent pass
   reads exactly like a real one. Now: `verified N/M inputs, P sha256-pinned`.

## What I did NOT change, and why

- **The rr4 recipe's `rc64_source_sha256` pin is CORRECT** and stays. My charter said it was
  stale and that the cure was to pin `05839d14`. Measured, that would have been a defect:
  `05839d14` is the shipped **decoder-only** body (5,638 B, 237 copies) and exports no encoder
  symbol, so it can never drive the encode stage. The pin correctly names the **encoder** role
  (`5c75e2c7`, 12,222 B, 1 copy).
- **`rc64_shipped_member_sha256` was NOT added to `RR4_RECIPE`.** Declaring it makes
  `rc64_shipped_member` a required input, which would break every existing caller that does not
  set `TAC_PQ2_RC64_SHIPPED_MEMBER`. It remains optional, and the constant is now available for
  a recipe author who wants both coder roles custodied by name.
- **No behavioural change to any produced byte.** Every edit is a docstring, a comment, a
  constant, a new pre-flight refusal, or a print. The encode, build, split and decode paths are
  untouched, so the rr4 byte-close is unaffected.

## Checks run

| Check | Result |
|---|---|
| `ruff check` | All checks passed |
| `--help` accuracy | Accurate; stage list matches `choices`; `all` correctly documented as excluding `decode` |
| Sanitization: `/Volumes`, `/Users`, fleet IPs, bare `python` | Clean. Child processes use `sys.executable`; the one custody path in a comment is already a `<VertigoDataTier>` placeholder |
| Default invocation (pointer → jg5) | Refuses by name, rc=1, before `store.mkdir` |
| ck1 sha via `--expected-archive-sha256` | Refuses by name |
| Unregistered sha | Falls through to the cross-pin guard, message unchanged |
| rr4 identity | Passes both guards and reaches input resolution — the happy path is intact |
| `--emit-inputs-template` | Unchanged |

## Optimization

**None applied, deliberately.** The charter permits optimization only where measured-safe with
an identity check. The script is an orchestrator: its own runtime is dominated by the two
subprocess stages it shells out to, so there is no hot path here to optimize, and any change
touching the encode path would need a byte-identity re-run to be admissible. The measured
wall-clock problem in this lineage is the **token decode inside inflate** (1,341.5 s = 95.72%
of inflation), which lives in the receiver, not in this script. Optimizing this file would have
been visible work on the wrong object.

## Owed

- The `NOT_EXPRESSIBLE` registry is hand-maintained. A future candidate outside the grammar
  will fall through to the cross-pin guard and get the over-promising message again. The
  structural cure — deriving expressibility from the recipe's declared stage set rather than
  from a sha allowlist — is the right fix and is not built.
- The script has no tests. The guard paths above were verified by execution, not by a suite.
