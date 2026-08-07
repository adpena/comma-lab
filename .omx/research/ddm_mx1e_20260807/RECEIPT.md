# ddm_mx1e Receipt - Software Memory Cap + v4 Fire Guard

Date: 2026-08-07
Axis: structural launch safety / scorer-free
Score claim: false
Promotion eligible: false
Tokens: [no-triality] [p0-ledger-ok]

## What Now Enforces

MLX 0.31.2 has no hard `set_memory_limit(..., relaxed=False)` cap, so Row-1 Metal clearance now rests on a software cap inside the trainer: `mx.get_active_memory() + max(0, process_rss - start_process_rss) <= budget`. The cap is installed before checkpoint/cache/scorer setup, checked after every setup stage, checked after every training step, and raises `MemoryBudgetExceeded` on breach; mem-probe writes an atomic failed receipt before re-raising. `mx.set_memory_limit` is recorded only as a soft guideline, while `mx.set_wired_limit(min(budget, 35% of total memory))` is attempted when available because the wired pool is the host-freeze surface.

## Code Changes

- `experiments/ddm_mx1_pr130_semantic_renderer.py`
  - Added `MemoryBudgetExceeded` and `LoadPhaseMemoryProbe` software-budget accounting.
  - Derives and installs the budget before heavyweight loads; GPU mode refuses if no software budget can be installed.
  - Checks the budget after setup samples and every train/eval step.
  - Replaced the old hard-cap interpretation with `enforcement="software_stage_step_cap"` receipt fields and MLX 0.31.2 soft-limit semantics.
  - Calls `mx.set_wired_limit` where present and records the derivation.
  - GPU `mlx-train` now requires `--fire-guard-verdict`, `--launch-ticket-path`, and `--fire-argv-key`, then re-runs `tools.mx1_fire_guard.evaluate_guard(...)` before MLX setup.
- `tools/mx1_fire_guard.py`
  - Accepts software-cap receipts when the budget summary exists, checks ran, and the last check stayed within budget.
  - Refuses mem-probe receipts older than 6 hours (`mem_probe_receipt_stale`) because host state drifts across reboots.
  - Validates ticket-bound `--launch-ticket-path` and `--fire-argv-key` from the fire argv.
- Focused tests cover software receipts, budget-exceeded receipt+raise, stale receipt refusal, v4 ticket schema, and forged/minimal passed verdict refusal.

## Ticket Artifacts

Generated:

```bash
.venv/bin/python experiments/ddm_mx1_pr130_semantic_renderer.py --mode probe --run-dir .omx/research/ddm_mx1e_20260807/row1_v4_two_arm --out .omx/research/ddm_mx1e_20260807/row1_v4_two_arm_ticket_result.json --launch-ticket-path .omx/research/ddm_mx1e_20260807/launch_ticket_v4_fire_guarded.json
```

Ticket summary:

- Ticket: `.omx/research/ddm_mx1e_20260807/launch_ticket_v4_fire_guarded.json`
- Driver result: `.omx/research/ddm_mx1e_20260807/row1_v4_two_arm_ticket_result.json`
- Schema: `ddm_mx1_row1_launch_ticket.v4_software_cap_fire_guarded`
- `main_fire_sequence`: `guard_precheck -> probe -> gate -> fire`
- Both n32 arms carry `--launch-ticket-path` and `--fire-argv-key`.
- Scheduling remains `SEQUENTIAL one-Metal-fire-at-a-time`.

Local guard precheck:

```bash
.venv/bin/python tools/mx1_fire_guard.py --ticket .omx/research/ddm_mx1e_20260807/launch_ticket_v4_fire_guarded.json --argv-key argv_n32_arm_cap --out .omx/research/ddm_mx1e_20260807/row1_v4_two_arm/launch_arm_cap/n32_metal/fire_guard_verdict.json
```

Observed rc: `4` as expected in the CPU sandbox.

Verdict:

```json
{"schema":"ddm_mx1_fire_guard_verdict.v1","status":"failed","reason_code":"mem_probe_receipt_missing","argv_key":"argv_n32_arm_cap"}
```

This is the correct pre-probe state: no Metal fire clears until MAIN writes a fresh passed mem-probe receipt and reruns the guard to a passed verdict.

## RECALL EVIDENCE

| source searched | query / command | found beyond charter seeds | changed plan |
|---|---|---|---|
| Governing files | Read `mx1e_prompt.md`, `_common_contract.md`, `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | Live board keeps METAL HOLD until mx1d/mx1e hardening plus capped probe; own-vehicle frontier remains advisory and unmoved. | Kept this landing scorer-free, local-only, and pointer-honest. |
| Incident memory | Read `/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/memory/concurrent_metal_fires_without_composed_preflight_oomed_the_machine_20260806.md` | MAIN's measured law says MLX 0.31.2 `set_memory_limit` is soft, `relaxed=` is removed, and only software cap plus wired limit exist. | Removed the impossible hard-cap requirement and made software budget the fire-clearance mechanism. |
| mx1d and rr10 | Read `.omx/research/ddm_mx1d_20260807/RECEIPT.md`, `launch_ticket_v3_fire_guarded.json`, `.omx/research/ddm_rr10_20260807/ROUND10_FINDINGS.md` | RR10-F2 found the entrypoint trusted schema/status only and could be bypassed by stale/forged passed JSON. | Added ticket/key arguments and in-process guard revalidation before MLX setup. |
| Current source/tests | Searched and read `experiments/ddm_mx1_pr130_semantic_renderer.py`, `tools/mx1_fire_guard.py`, and focused tests | Existing receipts had load samples and guard config matching but still expected hard-limit fields. | Converted receipt/guard/tests to software-cap validation plus 6h freshness. |
| Memory registry | `rg -n "mx1e|common_contract|codex_runs|charter|lane registry|main_hot_state|harness_tasklist_bridge|frontier pointer" /Users/adpena/.codex/memories/MEMORY.md` | Reinforced Pact lane, frontier, and queue-custody boundaries; no mx1e-specific prior row. | Did not claim score movement or dispatch. |

## Verification

```bash
.venv/bin/python -m pytest experiments/tests/test_ddm_mx1_memory_probe.py tools/tests/test_mx1_fire_guard.py -q
.venv/bin/python -m py_compile experiments/ddm_mx1_pr130_semantic_renderer.py experiments/tests/test_ddm_mx1_memory_probe.py tools/mx1_fire_guard.py tools/tests/test_mx1_fire_guard.py
.venv/bin/python -m ruff check experiments/ddm_mx1_pr130_semantic_renderer.py experiments/tests/test_ddm_mx1_memory_probe.py tools/mx1_fire_guard.py tools/tests/test_mx1_fire_guard.py
git diff --check -- experiments/ddm_mx1_pr130_semantic_renderer.py experiments/tests/test_ddm_mx1_memory_probe.py tools/mx1_fire_guard.py tools/tests/test_mx1_fire_guard.py .omx/research/ddm_mx1e_20260807/launch_ticket_v4_fire_guarded.json .omx/research/ddm_mx1e_20260807/row1_v4_two_arm_ticket_result.json .omx/research/ddm_mx1e_20260807/row1_v4_two_arm/launch_arm_cap/n32_metal/fire_guard_verdict.json
```

Results:

- Focused pytest: `17 passed`
- Py compile: passed
- Ruff: passed
- Diff check: passed

## Hashes

```text
351f90f196e0f0fb13cfe9cec620cffd0ad44882c47622fc95da663864af20af  experiments/ddm_mx1_pr130_semantic_renderer.py
6e276bd1a304fef3c91be967956368fcf7ea64f972656c1d8cb34c9b61b302e0  experiments/tests/test_ddm_mx1_memory_probe.py
5af5f3ed93a541fe22ed4757475eaa8507dc690faa0be97b72452af429e2f5ef  tools/mx1_fire_guard.py
087b5cafe0e813ca07bbf69b2b916a8728429f22a3aa7d706547f97efe3ebfa1  tools/tests/test_mx1_fire_guard.py
3dde93df4731fd3de0cda29074e5d0b93092bd9a614f032965fd2b2c12074a54  .omx/research/ddm_mx1e_20260807/launch_ticket_v4_fire_guarded.json
bac9f8dbcd03f30b0395ab4dad1b70a0c0447dca78a72661d14435d0c523d5bd  .omx/research/ddm_mx1e_20260807/row1_v4_two_arm_ticket_result.json
8de95533f4fce91a55c0d6df81363c909a2fc44e01250e768db1d3523288bb80  .omx/research/ddm_mx1e_20260807/row1_v4_two_arm/launch_arm_cap/n32_metal/fire_guard_verdict.json
```

## Boundaries

- No Metal training was run here.
- No scorer job, archive build, full-n600 run, remote dispatch, or `upstream/evaluate.py` run was performed.
- The local guard verdict is intentionally failed because the keyed Metal mem-probe receipt is absent.
- The probe-mode local MLX availability check was blocked by sandbox Metal unavailability.
- Own-vehicle frontier unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
