# FX5 — Linux x86_64 wheel closure

Generated: `2026-08-09T16:47:56Z`  
Axis: `[byte-only scorer-free, macOS-CPU custody and dispatch-preflight]`  
Authority boundary: no Linux container executed, no network wheel fetch occurred, no scorer ran,
no contest score was measured, and the pointer did not move.

## 1. Linux x86_64 network bootstrap receipt

**ABSENT — the governed Modal path refused before spawn.** I did not bypass the refusal, use a
different provider, seed a wheel cache, or claim PyPI metadata as execution. No FX5 lane claim,
Modal call ID, provider spend, dependency target, or Linux output was created.

The first refusal is the canonical #513 single-flight guard. This terminal local command ran with
the repository venv's Modal CLI on `PATH`:

```text
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src .venv/bin/python -c \
  'from tac.deploy.modal.single_flight import assert_modal_single_flight; \
   assert_modal_single_flight(label="ddm_fx5_linux_wheel_closure_20260809", \
   lane_id="lane_ddm_fx5_linux_wheel_closure_20260809")'
```

It terminated `rc=1`. Its governing refusal was:

```text
MODAL SINGLE-FLIGHT REFUSAL (operator binding 2026-07-15): a live Modal job /
active claim already exists — HARVEST or close it first; never fire a second.
```

It named two active #906 claims: `ddm_chroma_dali_av_gt_diff/modal-t4-r3` and
`lane_ddm_chroma_dali_av_gt_diff_20260809/dali_av_gt_diff_r2_20260809T013932Z`.
The provider cross-check also timed out and was explicitly skipped by the guard.

The independent dual-ledger reconciler terminated `rc=6` and reported **0 live call IDs / 3
active Modal claims**. Its denominator is all three active claim rows it found:

1. `ddm_chroma_dali_av_gt_diff/modal-t4-r3`, status `active_modal_t4`;
2. `lane_ddm_chroma_dali_av_gt_diff_20260809/dali_av_gt_diff_r2_20260809T013932Z`, status `eval`;
3. `lane_ddm_chroma_dali_av_gt_diff_20260809/pending_spawn_dali_av_gt_diff`, status `spawning`.

Each lacked a live call-ID-ledger row. A later terminal #906 receipt exists and says the successful
successor completed 600 / 600 AV and DALI pairs, but it does not assign a terminal outcome to each
of these three exact job IDs. I therefore did not invent those outcomes or use the later receipt as
an implicit terminal row.

The second refusal is provider reachability. A direct terminal provider check:

```text
/usr/bin/time -p .venv/bin/modal app list --json
```

terminated `rc=1` after `56.56 s` with `Could not connect to the Modal server.` The billing query
was likewise unable to return a report. Therefore neither live provider state nor remaining #381
headroom could be proven. This is an `INSTANCE` blocker for this execution environment, not a claim
that Modal or the Linux wheel is unavailable globally.

The background-output rule was satisfied: no remote call was spawned and no partial remote output
was consumed. The two quoted preflight processes and the reconciler all reached terminal status.

## Whole-job budget

**No current Linux timing exists.** This arm produced neither a dependency-bootstrap segment nor a
whole-job Linux measurement. The authoritative workflow wall remains `1,800 s` for the entire job,
not for `inflate.sh` alone. I do not carry forward FX1's `159.598 s` cross-host projection as a
measurement; it double-counts receiver startup, mixes hosts, and excludes other current job steps.

For antecedent context only, FX1 measured a `17.50 s` cold and `0.51 s` warm dependency-entrypoint
smoke on macOS arm64 with a cached wheel. Those numbers remain scoped to FX1's host and cache. They
do not answer this charter's Linux network-fetch question.

## Measured rate denominator

`[byte-only scorer-free, macOS filesystem measurement]`: the current contest-shaped
`upstream/videos/` tree contains **1 / 1 regular file**:

| path | bytes | SHA-256 |
|---|---:|---|
| `upstream/videos/0.mkv` | **37,545,489** | `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9` |

`upstream/evaluate.py:63-65` was re-read at the live snapshot. It computes
`uncompressed_size` by summing every regular file under `args.uncompressed_dir.rglob('*')`.
The measured dynamic sum on this one-file tree is therefore **37,545,489 B**. The familiar value
is correct here because of the present tree contents, not because it is an evaluator constant.

For the reproduced `191,052 B` archive, the byte-only rate is
`191052 / 37545489 = 0.005088547388475883`, and its score-form rate term is
`25 * 191052 / 37545489 = 0.1272136847118971`. This does not assert that a remote Linux tree was
measured; the blocked Modal job never materialized one.

## Declared dependency versus vendored wheel

**Race not executable under the charter's gate.** The Linux leg did not pass, so the charter's
“if and only if” condition forbids reopening the vendor comparison as a measured race.

A bounded custody search covered the FX1 SSD work root, the user uv cache, and the user cache for a
`constriction-0.5.0` manylinux x86_64 wheel and returned **0 matching paths / 3 searched roots**.
Thus this arm still lacks wheel bytes with which to price extraction/import glue or compare decode
time. FX1's structural preference for vendoring remains an unexecuted hypothesis, not an adopted
result. No source build was allowed, and no arithmetic-decoder substitute was attempted.

## Runtime and intake custody

- FX1 commit pin: `3e616f568ac17a658be76971cb1eb2cd02edf4c6`.
- Intake pin: `e34f31bc4969042c0051ac81aa3c56884419a231`; intake status remained clean.
- The five receiver modules remain byte-identical to intake: **0 changed / 5 compared**.
- The cured entrypoint SHA-256 remains
  `00ca74cef3986d1a01be6d0256e9ba671f4996a19035d5260d37ac4711ca46af`.
- The dependency manifest SHA-256 remains
  `a7cadcda5e169adbfa26f95663c1b72f56c16a994436a35d382b54d87a1dbf1d`.
- The FX1 installed-target manifest pin remains
  `bfe25761e26f32b1dca1f7114a45648fa9b25dc8f98f0fa2e98b199992fd4a4b`.
- The expected Linux wheel pin remains
  `eb7909d0ad4940d3b74696d98f0dc16dec7294e57f9e0797bc06d5ce7b3b1507`.

No receiver code, intake file, upstream file, or numeric fallback changed in FX5.

## Ranked residuals and falsifiers

1. **Dual-ledger inconsistency — HIGH, `INSTANCE`.** Three exact active claim rows have no live
   call-ID row, so #513 refuses. **Falsifier/cure:** recover the exact terminal evidence for each
   named job ID, append terminal rows through `tools/claim_lane_dispatch.py`, and require
   `tools/claim_lane_dispatch.py reconcile` to terminate `rc=0`.
2. **Provider connectivity and #381 visibility — HIGH, `INSTANCE`.** The CLI could not reach Modal,
   so live-app state and budget headroom are unknown. **Falsifier:** `modal app list --json` and a
   billing report both terminate `rc=0`, with no live conflicting app and proved cumulative spend
   plus the proposed CPU ceiling at or below `$20`.
3. **Linux network wheel execution — HIGH, `INSTANCE`.** Still unmeasured. **Falsifier:** a clean
   Linux x86_64 CPython 3.11 venv with `ENABLE_USER_SITE=False` and
   `find_spec('constriction') is None` fetches the pinned wheel over the network, verifies its hash,
   imports `RangeDecoder` and `Categorical`, imports the real receiver, prints
   `PR130_DEPENDENCY_READY`, and exits `rc=0`.
4. **Current whole-job time — HIGH, `INSTANCE`.** No single-host Linux total exists.
   **Falsifier:** one terminal contest-shaped job reports checkout, LFS, environment setup,
   dependency bootstrap, full inflate, and evaluator time inside the same 1,800-second frame.
5. **Vendored-vs-declared winner — MEDIUM, `FORMULATION`.** No race ran. **Falsifier:** after item 3
   passes, run the same wheel bytes as declared-install and vendored extraction/import treatments,
   verify receiver/decode identity, and compare network, extraction, import, and decode time.

## Could not check / why

- Linux x86_64 wheel import and receiver execution: the governed remote path refused before spawn.
- Network resolve, download, and install timing: no Linux process or wheel fetch occurred.
- Provider single-flight truth: the provider API was unreachable; local ledgers are inconsistent.
- Remaining #381 budget: the provider billing query returned no report.
- Full n600 inflate and whole-job time: no remote process started, and this arm owns no scorer job.
- Declared-vs-vendored timing: the Linux-pass prerequisite was false and no matching wheel was in
  the three bounded custody roots.
- Remote Linux denominator: no remote contest tree was materialized; only the local contest-shaped
  tree was measured.

## RECALL EVIDENCE

Searches covered:

- memory registry query `ddm_pr130|fx5|linux wheel|PR130`;
- `.omx/research` content queries for
  `constriction|RangeDecoder|Categorical|self-install|wheel-only|dependency bootstrap|Linux x86_64`;
- `.omx/research/CANONICAL_RESEARCH_INDEX_20260629.md` and
  `.omx/state/canonical_task_status.jsonl` with the same mechanism terms;
- the `sub015_DAG_*` FEED blocks for `constriction|self-bootstrap|dependency`;
- canonical equation registry output from
  `tools/list_canonical_equations.py --json`, selecting
  `pr95_family_l42_lazy_brotli_auto_install_bootstrap_v1`;
- live `main_hot_state.md`, the lane-claim ledger, the Modal call-ID ledger, and the operator P0
  ledger for `Modal|single-flight|#381|#513|PR130`;
- FX1's memo/JSON receipt and the completed FX4/#906 terminal receipt.

Beyond the charter seeds, recall found:

- `ddm_ua2_upstream_defenses_and_budget_surface_20260731.md` had already settled that the 30-minute
  limit is a whole-job wall and that declared dependencies must be priced against vendoring. This
  prevented reporting a dependency step as a whole-job result.
- Canonical equation L42 preserves lazy bootstrap as a real precedent, but its domain is an HNeRV
  Brotli CPU-runner subclass. It supports the boundary pattern, not Linux compatibility for
  `constriction`.
- FEED-603 records an IC2 clean-venv declared-dependency bootstrap, reinforcing the need for actual
  clean-host execution rather than metadata.
- The DAG also records `constriction` losing two real coder races on other payloads. Those results
  are payload-scoped and did **not** license skipping or prejudging this exact PR130 decoder race.
- FX4 proved the later #906 computation terminal, while the dual-ledger audit showed its predecessor
  job IDs remain active. That changed the plan from “dispatch” to “refuse and reconcile exact job
  identities first”; silently assuming the successor closed every predecessor would recreate the
  false-terminality bug the charter warns about.

No relevant row was found in the memory registry or canonical research index for FX5/Linux-wheel
execution. Those are scoped negatives for the queried stores, not global nonexistence claims.

## Follow-on dispositions

- **QUEUED-WITH-A-FIRE-ORDER:** owner = MAIN / resumed FX5; consumer store = this memo, then the
  canonical lane-claim and Modal call-ID ledgers; fire trigger = exact terminal evidence closes all
  three named stale claims, dual-ledger reconciliation is `rc=0`, provider app and billing checks are
  `rc=0`, no live Modal work exists, and cumulative #381 spend plus a short CPU ceiling is `<= $20`.
  Then claim `lane_ddm_fx5_linux_wheel_closure_20260809`, spawn one CPU-only CPython 3.11 job,
  immediately register its call ID, harvest to terminal status, and close both ledgers.
- **QUEUED-WITH-A-FIRE-ORDER:** owner = resumed FX5; consumer store = this memo; fire trigger = the
  Linux network bootstrap passes with the exact wheel hash. Then race declared-install against
  vendored extraction/import on the same host and bytes; do not adopt either arm before the race.
- **FOLDED for this execution:** owner = FX5; consumer store = this memo; fire trigger to reopen =
  the first queued item becomes green. Cached/pre-seeded Linux wheels, a different provider, a source
  build, and metadata-only compatibility remain inadmissible substitutes.

Frontier status: PR130 CPR1 remains `S=0.172141297491896447` at `191,052 B`
`[contest-CUDA, DALI GT, n600]`; this scorer-free blocker audit produced no new row.
