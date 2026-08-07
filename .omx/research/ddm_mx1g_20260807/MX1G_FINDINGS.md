# ddm_mx1g Findings

## Verdict

LANDED implementation path for the MX1 ticket generator. The code/test hunks are present at HEAD
via sibling serializer commit `d7f557bb7c` (`ddm_mx1h torch verdict endpoint`), which also absorbed
the MX1G ticket-generator changes while concurrent arms shared the same hot trainer file.
`launch_ticket()` no longer carries the pre-mx1f 66.268951 GiB projection latch. Each emitted fire
argv now derives its outer `tools/safe_run.py` `--projected-gib` and `--rss-mb` from that argv key's
fresh, passed, guard-validated mem-probe receipt, or emits the non-numeric
`REQUIRES_FRESH_MEM_PROBE` sentinel so the wrapper refuses before launch.

No Metal training, scorer job, upstream eval, archive construction, or frontier move was performed.
This arm is ticket apparatus only.

## Recall Evidence

Sources searched beyond the charter seeds:

- `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`,
  `.omx/state/main_hot_state.md`.
- `.omx/research/`, `.omx/state/`, `docs/`, `experiments/`, `tools/`, and canonical equations via
  queries for `ddm_mx1g`, `ddm_mx1f`, `safe_run`, `child-pidfile`, `done-receipt`, `mem_probe`,
  `projected-gib`, `66.268951`, `REQUIRES_FRESH_MEM_PROBE`, and `microbatch_pairs`.
- `tools/list_canonical_equations.py --json`.

Findings that changed the plan:

- `ddm_rr9_mem_probe_fire_protocol_v1` says safe_run projection is not a Metal load-stage receipt;
  the ticket therefore had to bind safe_run sizing to the actual mem-probe receipt rather than a
  remembered scalar.
- `ddm_rr12_20260807/ROUND12_FINDINGS.md` says `safe_run.py` has `--status-receipt` and
  `--child-pidfile`; the ticket now includes both per argv attempt.
- Fire manifests showed the stale done-receipt bug directly:
  `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/arm_cap_fire4/launch_manifest.json` and
  `.../arm_cap_fire5/launch_manifest.json` both used
  `/Users/adpena/Projects/pact/.omx/tmp/codex_runs/mx1_arm_cap_fire.done`.
- Current `tools/mx1_fire_guard.py` already includes `microbatch_pairs` in `_validate_config_match`
  at HEAD (`d98cf49bfa`, `ddm_rr13: microbatch fire-guard review`). I did not edit the guard; that
  is RR13's landed adjudication, not an MX1G ownership claim.

## Implementation

Deliverable 1: receipt-derived projection.

- Removed the latched `MX1B_MEM_PROBE_RESULT`/`METAL_UNKNOWN_MARGIN_GIB` projection path.
- Added `SAFE_RUN_RECEIPT_SENTINEL = "REQUIRES_FRESH_MEM_PROBE"`.
- The margin rule is recorded in code and ticket JSON:
  `measured_peak=max(peak_rss_gib, peak_mlx_reported_gib, peak_mlx_active_gib+peak_mlx_cache_gib)`;
  `projected_gib=max(15, ceil(measured_peak*1.5))`;
  `rss_mb=max(45000, ceil(projected_gib*1024))`.
- The receipt path and SHA-256 are written into each safe-run projection row.

Deliverable 2: attempt-unique receipts.

- Fire-guard verdict paths are now `<run_dir>/fire_guard/<argv_key>.<attempt_id>.json`.
- safe_run status receipts are now `<run_dir>/safe_run/<argv_key>.<attempt_id>.status.json`.
- safe_run child pidfiles are now `<status_receipt>.child.pid`.
- detached done receipt names are now `mx1_<argv_key>_<attempt_id>`, preventing the fire4/fire5
  stale `.done` alias.

Deliverable 3: first-class resume keys.

- Added `argv_n32_arm_cap_resume`, `argv_n32_arm_veh_resume`, `argv_n120_arm_cap_resume`, and
  `argv_n120_arm_veh_resume`.
- Each resume argv appends `--resume-from <arm run_dir>/mlx.latest.npz`.
- Each resume key gets its own `mem_probe_resume/mem_probe_receipt.json` and ticketed mem-probe
  command with the same chunked microbatch config requirement documented in `resume_protocol`.

## Regenerated Ticket

Artifact:
`.omx/research/ddm_mx1g_20260807/launch_ticket_mx1g_from_regen2.json`

Positive-control receipt-derived row:

- argv key: `argv_n32_arm_cap`
- receipt:
  `.omx/research/ddm_mx1e_20260807/regen2/launch_arm_cap/n32_metal/mem_probe/mem_probe_receipt.json`
- receipt sha256: `602331d28783365f98e961a04cade6cc555de121b232cb0284a4cfd037e0c0f2`
- measured peak: `13.378453 GiB`
- projected safe_run memory: `--projected-gib 21`, `--rss-mb 45000`
- axis: `[load-phase memory telemetry projection; score_claim=false]`

The other seven argv keys were correctly fail-closed because their matching mem-probe receipts were
absent in this scope.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile experiments/ddm_mx1_pr130_semantic_renderer.py experiments/tests/test_ddm_mx1_memory_probe.py`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest experiments/tests/test_ddm_mx1_memory_probe.py experiments/tests/test_ddm_rr12_mx1_ticket_immutability.py -q`
  - result: `24 passed in 0.98s`
- `.venv/bin/python -m ruff check experiments/ddm_mx1_pr130_semantic_renderer.py experiments/tests/test_ddm_mx1_memory_probe.py experiments/tests/test_ddm_rr12_mx1_ticket_immutability.py`
  - result: `All checks passed!`

Generated-ticket command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_mx1_pr130_semantic_renderer.py \
  --mode probe \
  --run-dir .omx/research/ddm_mx1e_20260807/regen2 \
  --out .omx/research/ddm_mx1g_20260807/mx1g_ticket_result_from_regen2.json \
  --launch-ticket-path .omx/research/ddm_mx1g_20260807/launch_ticket_mx1g_from_regen2.json
```

The local MLX probe remained blocked because this sandbox has no accessible Metal device. That is
environmental and not a score or training claim.

## Boundaries

- No `upstream/` edits.
- No protected files edited.
- No scorer slot used.
- No n600 run, archive eval, or contest promotion.
- Pointer unchanged.

Own-vehicle frontier line from current `.omx/state/main_hot_state.md`:
`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
