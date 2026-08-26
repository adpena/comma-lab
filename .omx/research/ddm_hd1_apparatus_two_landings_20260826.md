# ddm_hd1 apparatus two-landings receipt — 2026-08-26

## Status

Both structural cures are implemented, exercised in both directions, and ready as two
separate intended landings. They are **not commits on `main` from this arm**: the live
serializer hit the exact #1293 Git-object denial while trying to land landing 1. The new
fallback handled that already-broken case and retained a verified bundle, format-patch,
intended-tree patch, per-file hashes, and environment receipt on VertigoDataTier. The same
documented path is required for landing 2. No scorer or Modal call was made, and the frontier
did not move.

## RECALL EVIDENCE

Searched the full required surfaces before editing:

- Code and tests: `rg` queries for `#1293`, `#1237`, `Operation not permitted`,
  `ARCHIVE_SHA256`, `ARCHIVE_BYTES`, `check_pin_consistency`, `repin_receiver`,
  `make_candidate_seal`, and the serializer lock/commit paths.
- Research corpus and receipts: content searches under `.omx/research/`, `.omx/tmp/`, and
  both SSD roots for the same terms plus `HALF-UPDATED PIN`, `dg2`, and `jf2`.
- Canonical equations: `.venv/bin/python tools/list_canonical_equations.py --json`; no
  equation directly governs either apparatus defect.
- Research index/DAG/task surfaces: `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`,
  `.omx/state/main_hot_state.md`, and task-status content for #1237/#1293.
- Real custody: jf2's original retained runtimes, fixed runtime copies, and
  `scorer/runtime_fix/FIX_NOTE.json` on APDataStore.

The material finding beyond the charter seeds was that the canonical pin checker and repair
already existed in `src/tac/candidate_seal.py`; the Modal firer already consumed it, and the
runtime assembler already performed producer-time prevention. That changed the plan from a
new checker/framework to the smallest correct extension: reuse the checker in the two missing
consumers only. The SSD sweep also found four `ddm_po1` mismatches beyond the dg2/jf2 seeds.

## Landing 1 — #1293 serializer auto-bundle fallback

`tools/subagent_commit_serializer.py` now recognizes only a filesystem-permission token
combined with a Git-object-write token. On a match it snapshots the exact intended files or
patch, releases the fcntl lock, authors the intended commit in a throwaway shared clone, and
retains these artifacts on the SSD waterfall:

- `intended-commit.bundle`
- `intended-commit.format-patch`
- `intended-tree.patch`
- `receipts.jsonl` with bundle identity, per-file content hashes, failed Git output, cwd,
  uid/gid, sandbox markers, Git/object paths and mode, and repo/receipt mount lines

Success is structurally distinct: rc=17 plus a printed `BUNDLE_FALLBACK` line. A fallback
construction failure is rc=18 plus `BUNDLE_FALLBACK_FAILED`. Ordinary commit behavior and
return codes are unchanged.

Executed controls:

- Positive, throwaway repo with read-only `.git/objects`: rc=17; live throwaway HEAD stayed
  unchanged; the retained bundle verified and reproduced the intended file bytes in a fresh
  clone.
- Negative, normal throwaway commit: rc=0; no fallback receipt was created.
- Live already-broken case in this checkout: `git add` failed rc=128 with `Operation not
  permitted` / `failed to insert into database`; the cure returned rc=17 and printed the
  bundle line. No simulation or permission change touched the live `.git` directory.

Class population: **4/7 arms** in the bounded 2026-08-26 cohort hit the denial (pc2, hv2,
jf2, d3b); 3/7 did not (d3a, d3c, or1). Scope is the seven named same-day arms recorded by
MAIN, not a claim about all historical serializer calls.

Landing-1 custody receipt:

- Receipt:
  `/Volumes/VertigoDataTier/pact/ddm_hd1_landing1/receipts/commit_serializer_fallbacks/20260826T202050.386739Z-34035/receipts.jsonl`
- Bundle:
  `/Volumes/VertigoDataTier/pact/ddm_hd1_landing1/receipts/commit_serializer_fallbacks/20260826T202050.386739Z-34035/intended-commit.bundle`
- Bundle SHA-256: `ae44b6b4c7a033c059f529698a84e74ae4be8bbf1d81862057480ef48bd959cc`
- Intended commit: `9d89f98dd81cb56fcce9ce1c69a6cd30980a9e10`, based on
  `fc915c771fd50419e86c0ae512df3854712d6ad3`
- Intended file hashes: serializer
  `e3d0febdff7810a90425212ad793d700cc40a7342fcf6d9a52ea1e6c2e3465cc`; test
  `1035f34188eaaab75ff9e197d9f8be69249fc2341893d757dbdc114da9cca220`

## Landing 2 — #1237 pin consistency at every missing consumer

The existing `tac.candidate_seal.check_pin_consistency` is now called by:

- `tools/fire_local_advisory.py`, after the runtime/archive/inflate shell exist and before
  upstream repair, attempt creation, scorer composition, or fire;
- `tools/make_candidate_seal.py`, before archive measurement, pointer work, document build,
  or seal write.

Both paths refuse any missing or inconsistent pin. They require both
`sha256(archive.zip) == ARCHIVE_SHA256` and `stat().st_size == ARCHIVE_BYTES` in `inflate.py`.
No new checker, repair mechanism, or framework was added. Scope reduction: the existing Modal
firer and runtime assembler were already wired and were not changed; other bespoke historical
consumers are represented by the bounded sweep and are not silently claimed cured.

Executed real controls:

- Positive disease input: a retained copy of jf2's original broken k002500 runtime at
  `/Volumes/VertigoDataTier/pact/ddm_hd1_apparatus_two_landings/controls/jf2_k002500_broken_original_copy`.
  The archive is 180,200 B, while `inflate.py` declares 180,368 B. The local firer refused
  rc=2 before an attempt existed; the seal producer refused rc=3 and wrote no seal.
- Negative cured input: jf2's fixed k002500 runtime at
  `/Volumes/APDataStore/pact/ddm_jf2_terminal_diagonal_harvest/scorer/runtime_fix/k002500/candidate_runtime`.
  The local dry-run passed rc=0 without materializing an attempt. The seal producer passed
  rc=0 and its consumer validation returned `SEAL_VALID`.

Class population: in a read-only census of directories named `candidate_runtime` containing
both `archive.zip` and `inflate.py` under VertigoDataTier and APDataStore, excluding symlink
traversal, Git/venv/cache trees, and this arm's copied control, **11/23 runtime directories
were `MISMATCH` and 12/23 were `CONSISTENT`**. The 11 mismatches were seven jf2 runtimes and
four po1 runtimes. Combined with the charter's dg2 evidence, the class has affected at least
three producing arms/families; the exact-directory denominator here is 23, not a global
historical denominator.

The prior-law prediction was confirmed: the sweep found four po1 mismatches beyond the two
known dg2/jf2 instances. This scope did not find any `PIN_ABSENT`, missing-input, or checker
error outcome among the 23 admitted directories.

## Tests and reviews

- Serializer fallback focused controls: 3 passed.
- Full serializer regression suite: 81 passed.
- Seal/pin/local-advisory regressions: 82 passed.
- Ruff on every changed Python file: pass.
- `py_compile` on the two consumers and their changed tests: pass.
- `git diff --check`: pass.
- Every changed `.py` received two genuine review passes through
  `tools/review_tracker.py`; no review override was used.
- Re-introduction detectors remained live after the cure: a changed archive still turns a
  re-pinned runtime red; a normal serializer commit stays on rc=0; non-object Git failures do
  not match the fallback detector.

## Ledger and custody receipts

- Serializer typed receipt and bundle: landing-1 paths above.
- Pin controls, sweep denominator, typed outcomes, artifact hashes, and rcs:
  `/Volumes/VertigoDataTier/pact/ddm_hd1_apparatus_two_landings/receipts/pin_consistency_controls_and_sweep.json`
- Fixed-control self-validating seal:
  `/Volumes/VertigoDataTier/pact/ddm_hd1_apparatus_two_landings/controls/SEAL_jf2_fixed_control.json`,
  1,839 B, SHA-256
  `e1ba3e603070d8dc78171394661dab7c183ed2f45a2c6f9dd094ae0b5cc451ae`.

## Boundaries

This was apparatus-only work. It measured bytes, hashes, return codes, and directory
outcomes. It did **not** run or infer Seg/Pose components, score a candidate, use MPS as
authority, invoke Modal, mutate `upstream/`, repair the 11 historical mismatched runtime
directories, or move the frontier.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: the landing-1 serializer receipt
  above; fire trigger: fetch/verify the bundle and cherry-pick intended commit
  `9d89f98dd81cb56fcce9ce1c69a6cd30980a9e10`, then run the named serializer suites.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: the landing-2 fallback receipt
  printed by this arm's serializer invocation; fire trigger: after landing 1, fetch/verify
  that bundle, cherry-pick its intended commit, and rerun the 82-test consumer suite.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: jf2/po1 runtime custodians; consumer store:
  `pin_consistency_controls_and_sweep.json`; fire trigger: before any of the 11 mismatched
  directories is consumed, either re-pin its receiver from the staged archive and re-run the
  checker or retire the runtime with an explicit custody disposition.

## LIVE-HYPOTHESES

- Bespoke candidate materializers outside `tools/assemble_candidate_runtime.py` are the likely
  source of the remaining half-updated pins: this is plausible because all 11 mismatches are
  clustered in two experiment families while 12 other retained runtimes pass the same check.
- The serializer denial is imposed by per-process sandbox policy rather than ordinary Unix
  mode bits: this is plausible because `.git/objects` reported mode 0755 and repo mount
  `protect`, while the same process could write the workspace and VertigoDataTier.

## DEAD-ENDS

- A new pin checker/framework is closed: the canonical checker, typed verdicts, repair helper,
  Modal consumer, and assembler prevention already existed; duplication would create drift.
- A prevention-only fix is closed: the real jf2 broken copy and live Git denial both exercised
  already-broken inputs and produced the required refusals/custody artifacts.
- Treating the first two pin incidents as a fully drained historical class is closed: 11/23
  current in-scope runtime directories mismatch, including four po1 paths beyond the seeds.
- Repeated direct commit retries are closed for this sandbox: two live serializer attempts hit
  the same object-store denial and both emitted valid fallback custody; MAIN must consume the
  bundles instead of asking this process to retry `.git` writes.

Own-vehicle frontier unchanged: **gb1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4, n600]**.
