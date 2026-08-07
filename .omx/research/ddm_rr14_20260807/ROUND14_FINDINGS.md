# ddm_rr14 Round 14 Findings

status: NOT-CLEAN
round: 14
clean_pass_counter_after_round: 0/3
axis: apparatus / scorer-free
score_claim: false
frontier_moved: false
tags: [no-triality] [p0-ledger-ok]

## Counts First

| item | count / status |
|---|---:|
| Findings filed | 3 |
| Fixed inline | 3 |
| Clean adjudications | 2 |
| Metal / scorer / archive launches | 0 |
| Python review tracker passes | 2 per touched .py |

## Verdict

Round 14 is NOT-CLEAN. The post-round-13 state had small but real apparatus/test holes on
the fire chain:

- R1's fire-guard verdict schema was a consumer footgun: `reason_code` was canonical, but
  `reason` was absent.
- R2's resume path had only the missing-receipt refusal checked, not the matching-receipt PASS
  half.
- R4's torch-verdict fail-closed claims were true in code, but missing direct negative tests for
  missing/extra NPZ tensors and checkpoint/history step mismatch.

All three were fixed inline. Because this round landed fixes, it does not advance the clean-pass
counter; the next round starts from 0/3 and must review these fixes as new code.

## Findings

### RR14-F1 - LOW - Fire-guard verdicts lacked a `reason` compatibility alias

Verdict scope: APPARATUS / fire-guard verdict schema ergonomics.

The live dry-run symptom was real: `tools.mx1_fire_guard.evaluate_guard(...)` emitted
`reason_code` but not `reason`. I did not find an MX1 fire-guard consumer in the searched code
still using `get('reason')`, and the in-process entrypoint already prints `reason_code`. The
schema still made MAIN's dry-run `v.get('reason')` read as `None`, which is a preventable
operator-facing ambiguity.

Fix landed:

- `tools/mx1_fire_guard.py` now emits `reason` as an alias equal to `reason_code` on every
  verdict while preserving `reason_code` as canonical.
- `tools/tests/test_mx1_fire_guard.py` asserts the alias on both PASS and missing-receipt REFUSE.

Disposition: FIRED inline. This is a compatibility addition, not a launch or score change.

### RR14-F2 - MEDIUM - Resume guard PASS half was untested

Verdict scope: APPARATUS / resume-specific mem-probe receipt binding.

Before this round, the live dry-run had proven the resume argv failed closed for the right reason
when the resume-specific receipt was absent. It had not proven the symmetric PASS path: a fresh
host-correct receipt at the resume path should bind to `argv_n32_arm_cap_resume` and pass without
touching Metal.

Fix landed:

- `tools/tests/test_mx1_fire_guard.py` now builds a synthetic `argv_n32_arm_cap_resume` ticket in
  a pytest temporary directory, writes a matching schema-correct receipt under
  `mem_probe_resume/mem_probe_receipt.json`, runs `tools/mx1_fire_guard.py` through the CLI, and
  asserts `status=passed`, `reason_code=fire_guard_passed`, `reason=fire_guard_passed`,
  `argv_key=argv_n32_arm_cap_resume`, and the resume receipt path.

Boundary: this is a schema-level PASS test. It does not run Metal, MLX training, a scorer, or the
live run directory.

Disposition: FIRED inline.

### RR14-F3 - LOW - Torch-verdict fail-closed custody lacked negative tests

Verdict scope: APPARATUS / MX1H CPU-torch verdict loader and proxy-comparison custody.

The MX1H code was already strict on inspection:

- `_load_mlx_npz_checkpoint_for_torch(...)` compares the complete `param::*` key set against the
  torch model state dict and raises on missing or unexpected tensors.
- `_torch_tensor_from_mlx_param(...)` raises on mapped shape mismatch.
- `run_torch_verdict(...)` obtains `step = checkpoint_meta["step"]` and calls
  `_history_row_at_step(history, step)`, which requires an exact history row with
  `d_seg_batch`.

The gap was test evidence, not mechanism.

Fix landed:

- `experiments/tests/test_ddm_mx1_memory_probe.py` now corrupts a synthetic NPZ by deleting one
  parameter and by adding an unexpected parameter, and asserts the loader refuses both.
- The same test file now asserts `_history_row_at_step(...)` refuses when the checkpoint step has
  no matching history row.

Disposition: FIRED inline.

## Clean Adjudications

### R3 - Projection derivation audit

Clean. The regenerated ticket
`.omx/research/ddm_mx1g_20260807/launch_ticket_mx1g_from_regen2.json` carries the margin rule and
receipt provenance in the ticket:

- receipt sha256:
  `602331d28783365f98e961a04cade6cc555de121b232cb0284a4cfd037e0c0f2`
- measured peak: `13.378453 GiB`
- rule: `projected_gib=max(15, ceil(measured_peak*1.5))`
- derived projection: `projected_gib=21`, `rss_mb=45000`

The other seven argv keys emit `REQUIRES_FRESH_MEM_PROBE` for both `--projected-gib` and
`--rss-mb`. `tools/safe_run.py` parses those flags as `float` / `int`, so the sentinel is
fail-closed before launch rather than a string a later consumer silently ignores.

### R5 - Absorption hygiene

Clean in searched scope. `d7f557bb7c` contains one coherent copy of both intended MX1G and MX1H
changes in `experiments/ddm_mx1_pr130_semantic_renderer.py`:

- MX1G ticket-generator changes are present: receipt-derived safe-run projections,
  attempt-unique status receipts/child pidfiles/done receipts, explicit resume keys, and
  per-key mem-probe receipt paths.
- MX1H torch-verdict changes are present: NPZ loader, CPU-only verdict path, checkpoint pair IDs,
  exact history-step comparison, and receipt schema.

I did not find duplicate function bodies for the reviewed surfaces. The MX1G findings accurately
disclose that the trainer hunks landed in the MX1H commit, and no intended MX1G/MX1H hunk was
lost in the checked HEAD file.

## Assumption Challenge

Shared assumption: the MX1 Row-1 fire chain is still a procedural wrapper around a learned
PR130-derived renderer, and the review is validating apparatus safety rather than challenging
whether that substrate is the breakthrough representation.

Would violating it unlock breakthrough? Not inside this charter. The charter's blast radius is the
~21:00 resume / ARM-VEH / n120 fire chain, so the highest-value action is making that fire chain
fail closed and readable. The substrate-level assumption remains live for strategy, but violating
it here would bypass the requested guard review and would not produce an exact score row.

## RECALL EVIDENCE

| scope | query / source | found beyond charter seeds | changed plan |
|---|---|---|---|
| Governing files | Read `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, charter, and common contract. | Hot state says ARM-CAP is live and the endpoint chain is push-button only if the guard/ticket/verdict chain is proven; own-vehicle pointer is `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`. | Kept this scorer-free and pointer-honest. |
| Memory registry | Searched `MEMORY.md` for `ddm_rr14`, `rr14`, `20260807`, `common_contract`, `main_hot_state`, and frontier/lane surfaces. | Found live-state cautions: re-read `main_hot_state.md`, canonical pointer, task status, and lane receipts before action. No direct ddm_rr14 prior memory. | Re-read live surfaces instead of relying on the common contract's older frontier line. |
| Prior rounds and MX1 corpus | Searched `.omx/research` and state surfaces for `ddm_rr9_mem_probe_fire_protocol`, `safe_run`, `fire_guard`, `torch-verdict`, `CPU-torch`, `REQUIRES_FRESH_MEM_PROBE`, `microbatch_pairs`, `resume-from`, `mx1g`, and `mx1h`. | RR9's canonical equation reinforces that safe_run admission is not a substitute for a passed mem-probe receipt; stale ddm_mx1d/v4 artifacts still show older projection shapes; ddm_mx1g regenerated ticket is the current artifact for this chain. | Tested resume PASS against the current guard/ticket shape and did not treat stale tickets as authority. |
| Canonical equations | Ran `.venv/bin/python tools/list_canonical_equations.py --json` and searched for `ddm_rr9`, `mem_probe`, `safe_run`, `microbatch`, `resume`, and `stage_rc`. | Relevant anchors were `ddm_rr8_stage_rc_success_contract_v1` and `ddm_rr9_mem_probe_fire_protocol_v1`; no equation changed the MX1H torch-verdict implementation. | Preserved score_claim=false and no Metal/scorer use. |
| Current source and artifacts | Read `tools/mx1_fire_guard.py`, `tools/tests/test_mx1_fire_guard.py`, `experiments/ddm_mx1_pr130_semantic_renderer.py`, `experiments/tests/test_ddm_mx1_memory_probe.py`, MX1G ticket, MX1H receipt, ROUND13/MX1G/MX1H findings, and the three commit diffs. | Code was stricter than test evidence on MX1H; guard schema was weaker than operator ergonomics; resume PASS was absent. | Landed narrow tests plus the guard alias. |

Scoped negative: in the searched source/artifact scope, I did not find an MX1 fire-guard consumer
still calling `get('reason')` against the verdict dict. The alias is still retained because the
dry-run showed that human/operator consumers can make that read.

## Verification

Commands run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tools/tests/test_mx1_fire_guard.py experiments/tests/test_ddm_mx1_memory_probe.py -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile tools/mx1_fire_guard.py tools/tests/test_mx1_fire_guard.py experiments/ddm_mx1_pr130_semantic_renderer.py experiments/tests/test_ddm_mx1_memory_probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check tools/mx1_fire_guard.py tools/tests/test_mx1_fire_guard.py experiments/ddm_mx1_pr130_semantic_renderer.py experiments/tests/test_ddm_mx1_memory_probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file tools/mx1_fire_guard.py --status reviewed
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file tools/tests/test_mx1_fire_guard.py --status reviewed
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file experiments/tests/test_ddm_mx1_memory_probe.py --status reviewed
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check tools/mx1_fire_guard.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check tools/tests/test_mx1_fire_guard.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check experiments/tests/test_ddm_mx1_memory_probe.py
git diff --check -- tools/mx1_fire_guard.py tools/tests/test_mx1_fire_guard.py experiments/tests/test_ddm_mx1_memory_probe.py
```

Results:

- Focused pytest: `30 passed`.
- Py compile: passed.
- Ruff: passed.
- Review tracker: 23 guard entities, 5 guard-test entities, and 19 MX1 memory-probe test
  entities marked reviewed twice; policy checks had 0 violations.
- Diff check: passed.

## Boundaries

- No Metal, MLX training, scorer slot, archive build, remote dispatch, or `upstream/evaluate.py`
  run was performed.
- No live run directory was edited.
- I touched only `tools/mx1_fire_guard.py`, `tools/tests/test_mx1_fire_guard.py`,
  `experiments/tests/test_ddm_mx1_memory_probe.py`, and this findings file.
- Pre-existing dirty work is extensive; I did not revert or stage unrelated files.
- Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`;
  contest pointer remains borrowed/unmoved.
