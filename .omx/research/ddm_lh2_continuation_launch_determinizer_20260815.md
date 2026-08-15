# DDM LH2 continuation-launch determinizer

Date: 2026-08-15  
Disposition: **LOCAL HPAC CONTINUATION LAUNCH CLASS DETERMINIZED; LIVE CLOSER DOGFOOD ARMED**  
Axis: `[scorer-free local apparatus; no score claim]`

## Outcome

`tools/fire_watched_continuation.py` is now the canonical RX2/WC2 local-continuation entry point.
Given the parent run and a sealed wrapper mode, it composes the continuation without hand-authored
launcher or trainer argv:

- the reference trainer's three fail-closed environment requirements are parsed from its AST at
  composition time and verified byte-for-byte against all `--env` launcher assignments in one pass;
- peak RSS, thread need, walltime, and niceness policy come from the parent
  `launch_manifest.json`;
- the unique highest `qat_stage_end_epoch_*.pt` is selected under the parent checkpoint root, and
  an epoch mismatch or duplicate newest checkpoint refuses composition;
- the parent liveness and quality configs are path-rebased, while the quality bar is derived from
  the highest parent endpoint row (`131,220 B` at epoch 480) and `bar_start_epoch=481`;
- the complete closer-plus-trainer command is sealed to `<run_root>/launcher/launch.sh` before
  firing. The closer launches first, then the trainer fires only through
  `tools/launch_detached_process.py --arm-watchers --done-receipt`;
- the post-fire verifier reads only bytes appended after its launch offset. Its typed outcomes are
  `GATE_REFUSED`, `RECONCILED`, `RESUMED`, `FIRST_EPOCH_ROW`, `DEAD`, or
  `PENDING_BOUNDED`; a bounded local deadline is explicitly not process death.

`tools/local_endpoint_close.py` is the armed-at-launch sibling of the Modal closer. On the exact
source done receipt it runs only the scorer-free HPAC descent-law fitter, retains and SHA-256 binds
the exact source receipt and log, verifies and hashes both the final-best and terminal QAT-stage
checkpoint payloads, then emits `NEXT_FIRE_ORDER.json` for endpoint -> identity race -> micro-edit
recompile -> one composed #1058 T4 row. Every downstream row names a disposition, owner, consumer
store, and fire trigger. The closer itself launches none of those rows and invokes no scorer,
provider, archive compiler, or auth evaluator.

The warn-only hot-state preflight currently reports `CURRENT`: `main_hot_state.md` and
`canonical_frontier_pointer.json` both carry `S=0.1600920261571558`. A mismatch is surfaced as
`WARN_STALE`; it never silently rewrites the gitignored live-state file.

## Executed controls

- Focused incident-shaped suite: **22 passed**. It includes source mutation adding a fourth env
  gate, exact source-to-launcher env equality, real descent-law fitter execution, checkpoint
  ambiguity refusal, protected/unknown override refusal, all three historical gate-cure classes,
  launch-log offset isolation, source-done `DEAD(rc)`, timeout-not-death, receipt corruption,
  exact refit/log binding, retained done-receipt equality, final checkpoint custody, typed fire
  order, and idempotent closure replay.
- Existing watched-launch plus sealed-wrapper compatibility: **17 passed**.
- Ruff, CPython compilation, `git diff --check`, and the payload-retention detector all passed;
  the bounded payload scan found **0 findings across the four LH2 Python files**.
- Both generated real-parent watcher configs passed their own `--validate-only` entry points.
- Two review-tracker passes completed on each of the four Python files after the final fixes.
- The broader Modal closer suite was also run as a sister-family check: 54 passed and 3 failed.
  All three failures are bounded to its fixed `2026-08-14T12:00:00Z` claim fixtures becoming more
  than 24 hours old on 2026-08-15, so the current stale-claim guard refuses them before their older
  assertions. LH2 did not edit that independently owned surface.

The real-parent no-fire positive control is retained at
`/Volumes/VertigoDataTier/pact/ddm_lh2_20260815/positive_control/composed_e960_next/`:

- `composition_manifest.json`: 10,792 B, SHA-256
  `72f766ee275cb6c42468fec203f8694a8a3baa1f279f7bb6b0decd87ff1b265b`;
- `launch.sh`: 3,147 B, SHA-256
  `d81e8ad707ca88f5daaa2617643a5fcb899bebaf7a1da20b5be774305d50697e`;
- selected retained parent checkpoint: 1,099,767 B, SHA-256
  `cd89907b5330bd78f9c1477107504231792c235fa7637b8981698a10948a5a61`.

This control composed but did not fire. It launched no trainer, scorer, Modal call, or provider
work.

## Live dogfood

The closer is armed read-only against
`.omx/tmp/codex_runs/rx2_wc2_full_mps_e960.done`. Launch counter 25 / closer supervisor PID 96666
is retained at
`/Volumes/VertigoDataTier/pact/ddm_lh2_20260815/dogfood_live_e960_v2/launcher/launch_manifest.json`
(3,632 B, SHA-256 `21634685ab38127fad4227d0b97f7254b1e0f0c03fa9a50f2f60ab748aebb267`).
Its `ARMED.json` is 612 B, SHA-256
`80725202a2346504be2034c0925de53c473615024bf5e98f35d1d1d84725be45`, status
`ARMED_WAITING`, with the same-argv restart contract persisted. All dogfood writes are routed to
that separate Vertigo store; the live e960 run directory is read-only to LH2.

The first dogfood arming attempt, counter 24, was fail-closed before the closer child started when
the managed sandbox refused `setpriority(..., 10)`. Its typed manifest and rc=8 done receipt are
retained under `dogfood_live_e960/`; v2 removed only the unnecessary niceness mutation and kept the
same `safe_run` RSS, thread, and walltime envelope.

At the last bounded read, the sacred trainer log had reached epoch 512 after its already-live
counter-23 launch. LH2 did not signal, throttle, stop, relaunch, or modify that trainer or its two
watchers.

## Deduplicated harness census

The ARC-AGI-3/Weng harness crosswalk store is the existing append-only
`.omx/state/harness_failure_ledger.jsonl`, not a new registry. The five e960 hand attempts are
recorded as **one deduplicated incident**:

| class_id | attempt count | deduplicated incidents | initial state | cure state |
|---|---:|---:|---|---|
| `hpac_continuation_local_detached_launch_gate_chain_ad_hoc_typing` | 5 | 1 | `OBSERVED/OPEN` | `VERIFIED_CLOSED/CLOSED` |

The closing row binds the composer plus gate-chain verifier as the cure, the focused tests and
real-parent composition as controls, and dogfood counter 25 as the read-only live arming receipt.
The V2 ledger hygiene preflight passes strictly with zero findings. Reopen only if a future HPAC
continuation again needs hand-written launch argv/polling or a newly added source env gate is absent
from the sealed command.

## RECALL EVIDENCE

The recall pass searched beyond the charter seeds before implementation:

- `.omx/research/`, `.omx/state/`, `docs/`, `tools/`, and `src/tac/` by content for
  `continuation launch|watched continuation|gate-chain|PYTHONHASHSEED|resume lineage|verify-alive`;
- `CANONICAL_RESEARCH_INDEX*`, the complete `sub015_DAG_*` FEED surface, task/P0 ledgers, Git history,
  and `main_hot_state.md` for `#1057|#1058|HPAC|endpoint closure|done-receipt`;
- `.venv/bin/python tools/list_canonical_equations.py --json`, which found the live
  `hpac_mc36_joint_descent_law_v1` producer/consumer chain and no local continuation-launch
  composer equation to duplicate;
- DT1's charter, memo, and strict-cure test to recover its incident-dedup convention;
- the canonical harness-failure ledger and ARC-AGI-3 crosswalk API, which changed the census plan:
  the five attempts are one typed V2 incident with two lifecycle rows, rather than an ad hoc memo
  table or five inflated recurrences;
- the launcher, wrapper, reference trainer, watcher validators, Modal endpoint closer, parent/live
  manifests, exact five-attempt log, WC2 section 5j, and #1058 chain artifacts at source.

Nothing beyond those surfaces supplied an existing local composer, post-3-second gate verifier, or
armed local endpoint closer to reuse. The reusable discoveries were the harness-failure store,
Modal receipt conventions, canonical launcher, canonical watcher validators, and the existing
descent-law fitter; LH2 composed them rather than building parallel substitutes.

## Boundaries

- The reference trainer hash remains
  `8392a9b9f2d303698de59e627fa489a792ab0b0b38170cebd425f9310162059e`; it was read only.
- No scorer, Modal call, training run, archive build, auth evaluation, or paid work was launched by
  LH2. The dogfood process is a receipt watcher and scorer-free fitter only.
- No score row was measured and the exact pointer did not move. This apparatus landing is a means,
  not goal progress.
- Own-vehicle frontier remains `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- **Disposition: FIRED. Owner: local endpoint closer supervisor PID 96666, then MAIN for adjudication. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_lh2_20260815/dogfood_live_e960_v2/closure/` and its typed `NEXT_FIRE_ORDER.json`. Fire trigger: `.omx/tmp/codex_runs/rx2_wc2_full_mps_e960.done` appears with canonical schema and rc=0; action: refit the endpoint, hash both final checkpoints, retain the source receipt, and emit the contained #1058 fire order.**

## LIVE-HYPOTHESES

- Parsing fail-closed env comparisons from the reference trainer will keep future HPAC continuation
  modes synchronized without another sequential refusal, because the live three-gate incident and
  the synthetic fourth-gate control use the same AST seam.
- The endpoint closer will close the live run without hand intervention because its only trigger is
  the canonical done receipt, all post-trigger inputs are retained local files, and the complete
  closure path ran successfully on the synthetic final-run positive control.
- The composer seams should generalize to the next RX2/WC2 continuation mode because trainer-specific
  behavior is isolated to literal `PORT_MODES`, reference argparse/env extraction, and the parent
  manifest; arbitrary trainers remain deliberately out of scope.

## DEAD-ENDS

- Hand-written launcher/trainer argv plus bounded shell polling is closed for this HPAC continuation
  class: it caused the five-attempt incident and is now replaced by one source-derived command and
  typed verifier.
- Hardcoding the three current env variables is closed: a source mutation adding a fourth gate is a
  passing positive control, so the trainer source remains the authority.
- Treating the launcher's three-second window or any later local poll deadline as process death is
  closed: only a canonical done receipt produces `DEAD(rc)`; bounded silence produces
  `PENDING_BOUNDED`.
- A second endpoint poller is closed: the local closer consumes the launcher's canonical done receipt
  and mirrors the landed Modal receipt convention.
- Retrying the dogfood with explicit `--nice 10` in this sandbox is closed: counter 24 proved the
  niceness mutation is denied, while counter 25 armed under the same enforced resource envelope with
  inherited niceness.
