# ddm_rvf1 — rv15 findings execution

Date: 2026-08-20  
Owner: ddm_rvf1  
Axis: apparatus, documentation, and scorer-free custody audit  
Score effect: none; no archive bytes changed and rvf1 dispatched no scorer or Modal job.

## Outcome

The batch found and cured **five code defects** (F7, F11, F14, F15, F19), corrected the
F2/F3/F4 publication surfaces, and separated three rv15 overreads from real defects (F10d,
F10e, and the §2.5 resolving-measurement premise). The prior-law prediction of at least three
real code defects and at least two reviewer overreads therefore held.

The worktree implementation is complete and tested, but **not committed**. The required
`tools/commit_autosha.sh` call used 17 explicit files, 17 post-edit SHA-256 guards, label
`ddm_rvf1`, and the required `[no-triality] [p0-ledger-ok]` message. It refused before staging:

```text
git add failed (rc=128):
error: unable to create temporary file: Operation not permitted
error: .omx/research/ddm_cd1_corrector_shipping_axis_decomposition_20260820.md: failed to insert into database
fatal: updating files failed
```

HEAD remained `fa6863305c4ae4c947fc304dfc2b0e6c4876457d`; `git diff --cached --name-only`
remained empty. `verdict_scope: this managed sandbox's Git object/index write surface; the
working-tree fixes and their tests are present, but no landing is claimed.`

## Per-finding adjudication

| Finding | Re-derived-at-source verdict | Cure or refusal | Executed control |
|---|---|---|---|
| F1 | **CONFIRMED_FIXED** | MAIN already landed `d9874dbf9b`; rvf1 did not redo it. | The nine live packet surfaces carry 4,369.6 s per the source review. |
| F2 | **CONFIRMED_FIXED in worktree** | Every live cd1/rr8 citation now carries `2.03–2.22×` frame B and `2.77–3.08×` frame A, with original and re-anchored endpoints named. The rr8 staging manifest keeps its legacy scalar keys and adds explicit band keys. Commit refused by the sandbox as recorded above. | Projection arithmetic re-run: k=2.22 gives 904.1 s charged; k=3.08 gives 788.6 s. `py_compile` passed. |
| F3 | **CONFIRMED_FIXED in worktree** | Sources now state k=2 misses frame B by 7.6 s and d2h removal moves the bar to 1.75×. Bare harness-id routing was replaced by work content. Commit refused by the sandbox. | Source sweep found no remaining live `#1162` routing in cd1. rr8's 4.450× conservative and 6.007× scope-isolated rows clear all modeled endpoints. |
| F4 | **CONFIRMED_FIXED in worktree and superseded by direct evidence** | cd1 now calls 4.4–5.0× an unmatched-instrument upper bound, not a per-core result. MAIN independently landed `ddm_rr8_t4_wallclock_verdict_20260820.md`: 464.558564563 s inflate and 403.698 s token stage on Tesla T4. The latter is below cd1's modeled 423.6 s non-corrector residual, proving the split does not transfer. Commit of the source correction refused by the sandbox. | Read-only verification of MAIN's result: n600, rc=0, Tesla T4, archive `f3bce5d2…` at 180,625 B, score 0.14839100138338618 unchanged. |
| F7 | **CONFIRMED_FIXED in worktree** | `take_census` now fails closed on file `lstat`, directory `lstat`, and `os.walk`/`scandir` errors; it counts directory symlinks without following them. Destination census failure writes a blocker and retains the source. `--apply` also refuses an empty reference surface unless a substantive no-known-references rationale is supplied. Commit refused by the sandbox. | Injected unreadable file and unreadable directory controls raise `CensusError`; directory-symlink control proves no traversal. CLI positive control returns rc=2 when apply has neither references nor rationale. |
| F10a | **CONFIRMED_FIXED in worktree** | vr1 now lists only phases the tool emits; the nonexistent `SOURCE_MANIFEST` phase is removed. | Ledger re-read: pfs1 has 6 phase rows, dqs1 4. |
| F10b | **CONFIRMED_FIXED in worktree** | vr1 now describes `copy_and_hash_one`; `rsync` is retained only as the explicitly superseded aborted design. | Tool source contains the single-pass copy/hash loop and no rsync invocation. |
| F10c | **CONFIRMED_FIXED in worktree** | Fresh tracked-corpus census is 112 files / 157 occurrences for pfs1 and 561 / 22,274 for dqs1, a 141.9× occurrence ratio. The old `728`/`many` labels are removed. | `git grep -l` and `git grep -o` re-run on the final worktree, excluding this execution memo so its report does not perturb its own denominator. |
| F10d | **REFUTED_WITH_EVIDENCE at artifact-validity scope; future apparatus gap confirmed** | Empty `referenced_by` does not falsify the two manifest-equality proofs, so rv15 overreached if read as invalidating the completed moves. It is a real provenance omission for future moves; the apply-time reference/rationale gate closes that narrower defect. `verdict_scope: the two completed byte-equality verdicts only; no general waiver for future moves.` | Both historical PLAN rows read `referenced_by: []`; new missing-reference apply control returns rc=2. |
| F10e | **REFUTED_WITH_EVIDENCE at claim-attribution scope** | The original vr1 memo did not call the ledger “10 complete certificates.” rv15 attributed an implication not present in the source. The memo now makes the denominator explicit anyway: ten phase rows across two artifacts, no single complete-cert row. `verdict_scope: wording attribution only; the ledger remains phase-granular.` | Source-text search plus phase enumeration (6 + 4). |
| F11 | **CONFIRMED_FIXED in worktree** | The actual structural misroute was `_lint_bare_task_ids`: one memo filename anywhere waived every bare id in a charter. It now adjudicates each id-bearing line independently and tells the arm to route by content. No historical queue row was rewritten. Commit refused by the sandbox. | Positive/negative controls prove an anchored line is silent while an unrelated memo cannot launder `#1162`; a second bare line is flagged independently. |
| F14 | **CONFIRMED_FIXED in worktree** | Dry-run tests the generated shim in success-only `TemporaryDirectory` scratch, prints the intended attempt paths, and writes no attempt directory, shim, or manifest. Commit refused by the sandbox. | Controls prove an absent attempt remains absent and an existing empty attempt retains inode, size, mtime, and empty contents. |
| F15 | **CONFIRMED_FIXED in worktree; resolving measurement FIRED** | Canonical certification now flushes/fsyncs its ledger row and invalidates the cache immediately; alternate test ledgers cannot invalidate live state accidentally. The detached resolving scan completed rc=0. Commit refused by the sandbox. | `[byte/custody apparatus, scorer-free]`: 148,170 files scanned in 196.0 s; 5 authored blobs owed, 1 certified, 0 unreadable. Durable `ddm_rvf1_ssd_cache_refresh_20260820/launch/run.log` sha256 `22ab84a094ef189da118ccfc76621c3616a3cda693e9212ba4936a8227082477`; launch manifest sha256 `b701b44b02b5a7393f8597a40077043dd1a5c8350fcdb36cecc79a8caa06f504`. |
| F19 | **CONFIRMED_FIXED in worktree** | The cause was default label `anonymous`, so an agent's own mandatory checkpoint looked like a sister. Checkpoint CLI now records the current Codex/Claude session by default; the serializer infers self only from one fresh, file-covering checkpoint with the same session. A lone sister can never be inferred as self. Commit refused by the sandbox. | Exact reproduction now commits in an isolated repo without `--label`; log label is `ddm_rvf1`. Existing lone-sister controls still refuse rc=8/9. |
| §2.5 | **REFUTED_WITH_EVIDENCE as the specified experiment; no run launched** | The named pinned-vs-unpinned same-tree A/B cannot vary the instrument: the exact cd1 runtime sets all five numeric-thread env keys to `4` inside `inflate.py` and calls `inflate_archive(..., num_threads=4)`. External env settings are overwritten; editing the runtime would violate “same tree.” MAIN's direct T4 row has also replaced the modeled decision. `verdict_scope: this exact same-tree/environment A/B; it does not claim thread count is universally irrelevant.` | Source inspection of the exact archive-matched runtime and sha equality of cd1/jg5 archives. No fake identical A/B was launched. |

## Verification

- Two genuine review-tracker passes were recorded for every changed Python entity; a full scan
  ingested the two newly added symlink/error controls before their two follow-up marks.
- Final focused suite: **176 passed in 11.62 s**.
- All changed Python files passed `py_compile` and Ruff fatal-rule selection
  `E9,F63,F7,F82`.
- `git diff --check` passed for the declared batch.
- The first review pass caught and rejected an unsafe serializer design that would have inferred
  a lone sister as self. The final worktree design requires exact session identity.
- The second review pass found the `os.walk` directory-error surface and directory-symlink census
  gap; both now fail closed and have controls.

## RECALL EVIDENCE

Searched beyond the charter seeds:

- canonical equations via `tools/list_canonical_equations.py --json`, queried for corrector,
  decode, runtime, thread, census, and custody laws;
- `.omx/research/`, `CANONICAL_RESEARCH_INDEX*`, the sub-0.15 DAG FEED blocks, and the harness
  bridge by content for break-even, sync elimination, reference custody, dry-run, serializer,
  and SSD-authored-signal debt;
- `codex_arm_queue.jsonl`, `main_hot_state.md`, and lane claims for actual owners and current rows;
- the exact cd1/rr8 runtime, stager, ledgers, and MAIN's newly landed rr8 T4 verdict.

Beyond the seeds, recall changed the plan in four ways: it exposed the whole-charter memo waiver
inside `codex_arm_queue`; it showed the specified thread A/B is shadowed by hardcoded runtime
settings; it turned F19's ambiguous file-cover inference into a session-identity cure; and it
captured MAIN's completed direct T4 row, which supersedes the F4 model instead of inviting another
local projection.

## MAIN adjudication queue

The five current SSD-authored-signal rows are measured debt, not silently folded into this batch:

- Three rr8 runtime blobs (two `residual_archive.py` variants and one `inflate.sh`) look generated
  by committed stagers but require byte-identity proof before certification. Owner: MAIN/rr8
  custody. Consumer: `.omx/research/ssd_authored_signal_certified.jsonl` or their tracked source
  homes.
- The rr5 retained rider `inflate.py` needs the same generated-copy versus unique-delta decision.
  Owner: MAIN/rr5 custody. Consumer: the same certification ledger or tracked rider source.
- The gt2 `positive_control_repo/tools/undeclared_gt_consumer.py` is likely deliberate injected
  control material, but gt2 is live and must disposition it. Owner: ddm_gt2, then MAIN. Consumer:
  the certification ledger or gt2's cleanup receipt.

## NEXT_IF_RESUMED

- **QUEUED_WITH_OWNER** — owner: MAIN or the next Git-writable rvf1 successor; consumer store: Git `main`; fire trigger: `.git/objects` and the temporary index accept writes. Re-run the 176-test suite, include this memo plus the durable SSD launch manifest and log, and serialize the exact declared rvf1 files with `tools/commit_autosha.sh`.
- **QUEUED_WITH_OWNER** — owner: MAIN/rr8 custody; consumer store: `.omx/research/ssd_authored_signal_certified.jsonl` or tracked source; fire trigger: byte-compare proves each of the three rr8 blobs is a deterministic generated copy, or finds a unique authored delta requiring recovery.
- **QUEUED_WITH_OWNER** — owner: MAIN/rr5 custody; consumer store: `.omx/research/ssd_authored_signal_certified.jsonl` or tracked rider source; fire trigger: byte-compare classifies the retained rider `inflate.py` as generated or unique.
- **QUEUED_WITH_OWNER** — owner: ddm_gt2 then MAIN; consumer store: gt2 cleanup receipt or `.omx/research/ssd_authored_signal_certified.jsonl`; fire trigger: the live gt2 lane finishes using its injected positive-control consumer.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN review coordinator; consumer store: the successor wave-end adversarial-review memo; fire trigger: the rvf1 fix batch lands. Run clean review round 1; do not credit the two review-tracker passes as adversarial clean rounds.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN review coordinator; consumer store: the successor wave-end adversarial-review memo; fire trigger: clean review round 1 lands CLEAN. Run clean review round 2.

## LIVE-HYPOTHESES

- The three rr8 owed blobs are generated staging outputs rather than lost sole-authored code. This
  is plausible because they live under named candidate-runtime trees and the committed rr8 stager
  rewrites those exact runtime surfaces; byte comparison is still required.
- The rr5 rider blob is probably another generated runtime copy. Its path is under a retained rider
  runtime, but no source-identity receipt was found in this batch, so certification remains owed.
- Per-line task-id anchoring closes the observed laundering bug, but a content-resolvable mapping
  would be stronger than filename adjacency. It is plausible because the repo already has task
  status and memo indexes; the two stores need an explicit join rather than another regex waiver.

## DEAD-ENDS

- Re-running F1 is closed: MAIN's `d9874dbf9b` already fixed it.
- A pinned/unpinned environment A/B on the exact cd1 tree is closed: both arms would execute the
  same hardcoded four-thread settings, while changing the runtime violates the matched-tree control.
- Re-firing rr8's T4 row is closed: MAIN already harvested a passing 464.558564563 s exact row.
- Treating one memo filename anywhere in a charter as authority for every bare task id is closed by
  per-line adjudication.
- Treating ten phase rows as ten complete certificates is closed by the explicit two-artifact
  denominator.
- Retrying the same commit in this sandbox is closed until Git write permissions change; the first
  canonical attempt refused before staging and the index remains clean.

**Own-vehicle frontier: S = 0.14839100138338618 @ 180,625 B `[contest-CUDA T4 n600]`, UNMOVED by
rvf1. MAIN's concurrent rr8 row changed wall-clock only; rvf1 changed no archive bytes.**
