# ddm_rr13 Round 13 Findings

status: NOT-CLEAN
round: 13
clean_pass_counter_after_round: 0/3
axis: apparatus / scorer-free
score_claim: false
frontier_moved: false
tags: [no-triality] [p0-ledger-ok]

## Counts First

| item | count / status |
|---|---:|
| Findings filed | 4 |
| Fixed inline | 1 |
| Routed/folded into mx1g | 3 |
| Fire logs read | 5/5 |
| Metal / scorer / archive launches | 0 |
| Python review tracker passes | 2 per touched .py |

## Findings

### RR13-F1 - HIGH - Fire guard omitted the effective microbatch footprint

Verdict scope: APPARATUS / fire-guard receipt equivalence.

The seeded hypothesis was real. Before this round, `tools/mx1_fire_guard.py` compared mode,
device, pairs, bits, caches, init, soft-memory, and scalar training fields, but not the
post-mx1f chunking footprint. The mem-probe receipt records `argv_config.microbatch_pairs=0`
and a measured effective `train_result_summary.microbatch_plan.microbatch_pairs=4`; a fire
argv changed to `--microbatch-pairs 32` could pass the guard against an n4 chunked receipt.

Fix landed in this round:

- `tools/mx1_fire_guard.py:120-128` derives the effective microbatch footprint with the same
  GPU default rule as the trainer: explicit positive value, else GPU defaults to `min(4,pairs)`,
  else CPU full batch.
- `tools/mx1_fire_guard.py:131-145` records the effective fire-side `microbatch_pairs`.
- `tools/mx1_fire_guard.py:157-180` reads the receipt-side effective microbatch plan from
  `train_result_summary.microbatch_plan` when present, falling back to `argv_config`.
- `tools/mx1_fire_guard.py:276-289` includes `microbatch_pairs` in the named comparison set.
- `tools/tests/test_mx1_fire_guard.py:242-272` mutates a matching ticket to
  `--microbatch-pairs 32` and proves the guard refuses with `receipt_config_mismatch`.

Positive current-ticket check after the fix:

```text
passed fire_guard_passed 4 4
```

That was a scorer-free guard evaluation of
`.omx/research/ddm_mx1e_20260807/launch_ticket_v4_fire_guarded.json` key
`argv_n32_arm_cap`; both fire and receipt effective microbatch footprints were 4.

Disposition: FIRED in this round.

### RR13-F2 - HIGH - Rewrap safety depends on mx1g landing and artifact refresh

Verdict scope: APPARATUS / outer safe_run wrapper projection.

The hand-rewrap pattern is real in the fire corpus. The checked v4 ticket artifact still carries
the old outer safe_run projection shape: `--projected-gib 66.268951` and `--rss-mb 90000`.
The successful fire5 manifest instead used `--projected-gib 15` and `--rss-mb 45000` around
the same inner trainer argv:

- `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/arm_cap_fire5/launch_manifest.json:3-13`
  shows the hand-corrected outer wrapper.
- `.omx/research/ddm_mx1g_20260807/CHARTER.md:24-31` assigns the real cure: derive
  wrapper projection and RSS cap from the passed mem-probe receipt, or emit a
  `REQUIRES_FRESH_MEM_PROBE` sentinel that safe_run refuses.

Current dirty-source inspection found the mx1g-style mechanism already present in
`experiments/ddm_mx1_pr130_semantic_renderer.py`:

- `_wrap_fire_argv` now accepts and emits `--status-receipt` plus `--child-pidfile`.
- `launch_ticket()` now builds receipt-derived safe_run projections and per-key status paths.

Boundary: that source state is not this rr13 edit and the checked v4 ticket artifact on disk is
stale. A future fire must consume the regenerated/serialized mx1g ticket, not the stale v4
artifact or the fire5 hand-rewrap.

Disposition: FOLDED into `ddm_mx1g` deliverable 1; fire order is "land mx1g, regenerate the
ticket from the receipt-derived source, then use only the regenerated detached fire command."

### RR13-F3 - HIGH - Stale `.done` receipt remains an artifact-level watcher hazard

Verdict scope: APPARATUS / detached receipt uniqueness.

The stale `.done` hypothesis is real for the observed fire sequence. Fire4 and fire5 used the
same detached receipt path:

- `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/arm_cap_fire4/launch_manifest.json:58`
  names `.omx/tmp/codex_runs/mx1_arm_cap_fire.done`.
- `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/arm_cap_fire5/launch_manifest.json:58`
  names the same path.
- `.omx/tmp/codex_runs/mx1_arm_cap_fire.done:1` still reads `rc=9 elapsed=0 detached-job`.
- `tools/codex_arm_queue.py:647-659` treats the last line of `<name>.done` as the terminal
  receipt for watcher/status classification.

Current dirty-source inspection found the likely mx1g cure already started: attempt-specific
detached receipt names and status receipt paths are generated in `launch_ticket()`. The checked
v4 artifact has not been regenerated and therefore still cannot be used as proof that the watcher
side is cured.

Disposition: FOLDED into `ddm_mx1g` deliverable 2; fire order is "regenerated ticket must carry
attempt-unique detached done names and safe_run status receipts before any ARM-VEH/n120 fire."

### RR13-F4 - MEDIUM - Current artifact lags the child-pidfile doctrine

Verdict scope: APPARATUS / kill procedure and operator evidence.

The safe_run code side is clean for the doctrine:

- `tools/safe_run.py:109-116` exposes `--child-pidfile`.
- `tools/safe_run.py:204-243` derives/writes the child pidfile and child-only kill command.
- `tools/safe_run.py:406-418` writes `child_pidfile`, `child_only_kill_command`, and an
  operator rule that says never to use argv pattern matching.
- RR12 recorded the test evidence at `.omx/research/ddm_rr12_20260807/ROUND12_FINDINGS.md:39-60`.

The stale v4 ticket artifact and fire5 manifest do not carry those fields, so they do not provide
the operator-facing child-only kill evidence. I did not find a live instruction surface in the
mx1e/mx1f/mx1g/rr9-rr13 scope that still directs `pkill -f` as the procedure; occurrences found
were historical findings or anti-pattern descriptions.

Disposition: FOLDED into `ddm_mx1g` deliverable 2/status-receipt refresh; do not reuse the stale
v4 artifact for kill-on-sight procedure.

## Bounded Clean Checks

### F5 resume-leg soundness

No additional finding beyond the mx1g artifact boundary. In current source, GPU `mlx-train`
with `--resume-from` still enters both gates before MLX setup:

- `experiments/ddm_mx1_pr130_semantic_renderer.py:2514-2517` calls
  `assert_governed_admission(...)` for `mlx-train`, then `_assert_gpu_fire_guard(...)` for GPU.
- `experiments/ddm_mx1_pr130_semantic_renderer.py:2388-2463` reruns
  `tools.mx1_fire_guard.evaluate_guard(...)` in-process and cross-checks the passed verdict.
- Current dirty source emits `argv_*_resume` ticket keys and `mem_probe_resume` receipt paths.

Boundary: `--resume-from` intentionally stays outside `_validate_config_match` because it changes
the training source checkpoint, not the probed memory footprint. The resume fire still needs a
fresh receipt under the resume key because the guard freshness window is <=6h.

Disposition: FOLDED into `ddm_mx1g` deliverable 3; do not hand-append resume flags to the stale
v4 artifact.

### Fire corpus read

Read all charter-named fire logs:

- `arm_cap_fire/run.log`: governor stale/high projection refusal.
- `arm_cap_fire2/run.log`: in-process guard import failed with `ModuleNotFoundError: No module named 'tools'`.
- `arm_cap_fire3/run.log`: in-process guard failed with `NameError: name 'pathlib' is not defined`.
- `arm_cap_fire4/run.log`: guard failed with `mem_probe_receipt_missing`.
- `arm_cap_fire5/run.log`: successful start, then grouped-backward banner.

I did not touch live run directories or fire Metal.

## Assumption Challenge

The assumption "the fire guard already binds every footprint-determining field" was false; F1
fixed the omitted effective microbatch footprint. The assumption "mx1g owns the ticket-generator
cures" is true in the charter and current source direction, but not yet true for the stale v4
artifact on disk. The assumption "child-only kill is available for this fire sequence" is true for
`safe_run.py` source, false for the already-fired artifact manifests.

## RECALL EVIDENCE

| scope | query / source | found beyond charter seeds | changed plan |
|---|---|---|---|
| Governing files | Read `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, charter, and common contract | Hot state says ddm_rr13 is active in the burn-window wave and own-vehicle pointer is `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`. | Kept the round apparatus-only and pointer-honest. |
| Incident memory | Read `concurrent_metal_fires_without_composed_preflight_oomed_the_machine_20260806` | Leg 9 requires child-only kill by pidfile and says free-memory swings were observed-not-receipted after RR10-F3. | Separated safe_run source support from stale fire artifact evidence. |
| Prior rounds | Read rr9, rr10, rr11, rr12 findings plus mx1e/mx1f receipts | RR9 mem-probe-before-fire, RR10 review race, RR11 ticket immutability, RR12 child-pidfile, and mx1f microbatch are the predecessor chain. | Focused patch on microbatch guard equivalence and folded ticket generator cures to mx1g. |
| mx1g surface | Read `.omx/research/ddm_mx1g_20260807/CHARTER.md`; searched for `MX1G_FINDINGS.md` | Only mx1g charter is present under `.omx/research/ddm_mx1g_20260807`; no mx1g findings receipt is present. | Routed projection/done/resume artifact refresh to mx1g instead of duplicating it. |
| Current source/artifacts | Searched/read `tools/mx1_fire_guard.py`, `tools/safe_run.py`, `experiments/ddm_mx1_pr130_semantic_renderer.py`, v4 ticket, fire4/fire5 manifests, `.done` receipt, and focused tests | Current source contains uncommitted mx1g-style ticket changes, while the checked v4 ticket artifact is stale. | Report distinguishes source capability from durable artifact authority. |
| Canonical equations / graph | Ran the canonical-equations registry and targeted research/state searches for `ddm_rr9_mem_probe_fire_protocol`, `ddm_rr8_stage_rc_success_contract`, `mx1`, `safe_run`, `review_interlock`, `microbatch_pairs`, and `resume-from` | Relevant registry anchors reinforce mem-probe-before-fire and stage-rc success semantics; no equation supersedes the live artifacts. | Did not promote any score or dispatch authority from apparatus findings. |

## Verification

Commands run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tools/tests/test_mx1_fire_guard.py -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tools/tests/test_mx1_fire_guard.py experiments/tests/test_ddm_mx1_memory_probe.py experiments/tests/test_ddm_rr12_mx1_ticket_immutability.py -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile tools/mx1_fire_guard.py tools/tests/test_mx1_fire_guard.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check tools/mx1_fire_guard.py tools/tests/test_mx1_fire_guard.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file tools/mx1_fire_guard.py --status reviewed
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file tools/tests/test_mx1_fire_guard.py --status reviewed
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check tools/mx1_fire_guard.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check tools/tests/test_mx1_fire_guard.py
git diff --check -- tools/mx1_fire_guard.py tools/tests/test_mx1_fire_guard.py
```

Results:

- Focused guard tests: `4 passed`.
- Guard + mx1 memory/ticket regression tests: `22 passed`.
- Py compile: passed.
- Ruff: passed.
- Review tracker: 23 guard entities and 5 test entities marked reviewed twice; policy checks had 0 violations.
- Diff check: passed.
- Scorer / archive / evaluator: not run.

## Boundaries

- No Metal, MLX training, scorer slot, archive build, remote dispatch, or `upstream/evaluate.py` run was performed.
- No live run dir was edited.
- I touched only `tools/mx1_fire_guard.py`, `tools/tests/test_mx1_fire_guard.py`, and this findings file.
- Pre-existing dirty work is extensive; I did not revert or stage unrelated files.
- Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved.
