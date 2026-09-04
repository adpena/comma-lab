# ddm_fm3 — fmtools (ours) as the ADVISORY second lane beside every regex census, and fmtools polished to OSS world-class

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-04 · Cost: $0 (on-device FM)

## Operator directives (2026-09-04, verbatim)
"Remember you can always use fmtools in addition to regex too." · "And we own fmtools and can enhance or extend or harden
or polish fmtools as well as long as in keeping with OSS world class best practices."

## Why (the wave's three regex misses)
(1) Catalog #344's finding token `ratified` matched inside `stratified` — 55% false positives that blocked honest commits
once the gate reached the commit path (eq1). (2) The lineage gate's regexes accept only `.npy`/`.pt` — blind to the
`.npz` PyAV table the born trainer PINS as authority (bh1 finding 2; ~372 historical consumers light up if widened
naively). (3) ql3's provenance-comment scan missed the retired n96 constants living in two harnesses; only a value
fingerprint found them. Each is a prose/code CLASSIFICATION regex cannot make. fmtools (`~/Projects/fmtools`, OURS;
Apple on-device FM via `apple_fm_sdk`; measured 5/6 vs regex 3/6 on triage; ~0.5 s/call; ADVISORY — never a score,
verdict, or blocking authority) is the second lane. Firewall unchanged: fail-open, subprocess under the fmtools venv
(`~/Projects/fmtools/.venv/bin/python`, FM available (True, None) verified 2026-09-04), pre-filtered candidates only,
never per-turn/latency-critical paths.

## Verified at source (VERIFIED-AT-SOURCE LAW — extend with path:line)
- fmtools repo `~/Projects/fmtools` @ `75fa5d9` (its own `CLAUDE.md`/`AGENTS.md` BIND inside it; `pyproject.toml`,
  `tests/`, `CHANGELOG.md`, `docs/`, `examples/`); surface `@local_extract(schema, retries=3)` (async; docstring =
  system instruction; returns a validated `@apple_fm_sdk.generable()` object), `stream_extract`, DSPy adapter, Polars
  `.local_llm.extract()`. Existing pact consumers to REUSE as the pattern: `tools/magnitude_dismissal_detector.py`
  (subprocess-under-fmtools-venv, fail-open, "fmtools confirmation owed" label), `tools/triality_drift_detector.py`,
  `tools/codex_status.py --classify`, `tools/dashboard_fm_events.py`.
- The three regex sites: `src/tac/preflight.py` #344 tokens `_CHECK_344_*` (:85204-85240) and the hook step
  `tools/preflight_hook.py::run_canonical_equation_reference_scan`; the lineage gate regexes `src/tac/preflight.py:2466-2469`
  (WARN-ONLY at :2863); ql3's census memo `.omx/research/ddm_ql3_apparatus_debt_20260904.md` (value-fingerprint hits).

## Deliver — part A (pact): three advisory lanes, each report-only
A1. **#344 second lane**: an fmtools classifier "does this memo STATE a measured empirical finding of its own (vs. a
    review/process memo)?" over the pre-filtered memos the regex flags; emitted as an ADVISORY column in the hook's
    output and in `check_empirical_finding_memo_references_canonical_equation(verbose=True)`; disagreement between the
    lanes is LOGGED (jsonl under `.omx/state/`), never enforced; fail-open. Measure agreement on the 29 eq1 memos
    (regex-after-fix vs FM vs eq1's human table) and report precision/recall of each lane.
A2. **Lineage-consumer census → the widening plan**: enumerate every reader of `gt_n600.npz` / `gt_n96.npz` / `.npz` GT
    tables across `src`, `tools`, `experiments` (AST + grep for the enumeration; fmtools to CLASSIFY each as
    authority-consumer / continuity-frame / advisory-instrument / test-fixture / historical-memo); deliver the table with
    path:line and the live count the widened regex (`.npz` added) would produce per class; then widen the gate's regex
    with the class-aware acceptance the table justifies (authority consumers refuse; continuity/advisory carry a
    same-line lineage label) — landing WARN-ONLY unless the live count of the refusing class is 0.
A3. **Constant-provenance lane**: over ql3's value-fingerprint hits (and any new ones from a re-run), fmtools labels each
    constant's provenance class from its surrounding code/docstring (measured-n600 / measured-prefix / derived / waived /
    unknown) as an advisory column beside the value-fingerprint verdict.

## Deliver — part B (fmtools repo, ours): OSS world-class polish, scoped to what A exercises
B1. A stable **batch classification surface + CLI** for subprocess callers (JSON lines in → validated JSON lines out;
    `--schema`, `--instruction`, `--timeout`, `--retries`, `--max-concurrency`; exit codes documented; fail-open
    semantics explicit) so pact tools stop re-implementing the subprocess dance.
B2. **Determinism + offline tests**: recorded-fixture tests (no FM required in CI), a live-FM test marked/skipped when
    the model is unavailable, latency measured and asserted only as a ceiling; typing clean (ruff + ty as in pact — never
    mypy); docstrings; README section; `CHANGELOG.md` entry; semver bump; CI workflow if absent.
B3. Keep the repo's own conventions (read its `CLAUDE.md`/`AGENTS.md`); commits in that repo under its conventions —
    NO co-author trailers anywhere (operator rule); no scope creep beyond A's needs.

## Constraints
- $0; on-device FM only; never network. In pact: the fmtools venv is the subprocess interpreter — the pact venv gains ZERO
  deps. ng2/ng3 LIVE on the Metal, fs1/gv1/ng4 on the CPU — never touch their custody/claims; `upstream/` and
  `submissions/semantic_joint_ctxmix/` read-only; no /tmp paths. OPTIMAL FORM: reference = the existing fmtools surface
  and the magnitude-dismissal subprocess pattern at pact `32bd236136d1c166cc1e6a1962431fd942c7ccee` / fmtools `75fa5d9`; TOY-BRACKET none (A1's
  agreement study is the measurement).
- Pact commits ONLY via `tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256
  <file>=<post-edit sha>`; tags `[no-triality] [p0-ledger-ok]`; every .py: tests + `tools/review_tracker.py mark-file`
  twice; never REVIEW_GATE_OVERRIDE on .py. Memo `.omx/research/ddm_fm3_fmtools_advisory_lanes_and_polish_20260904.md`
  (MEASURED agreement numbers; the widening plan table; what changed in fmtools with its version). Final message →
  `.omx/research/arm_final_messages/ddm_fm3_final_<utc>.md`, committed; LAST action `touch .omx/tmp/codex_runs/ddm_fm3.done`.
  Read `docs/operating_manual_craft_handoff.md` §labels first.
