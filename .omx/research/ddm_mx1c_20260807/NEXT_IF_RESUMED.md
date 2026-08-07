# ddm_mx1c Next If Resumed

Date: 2026-08-07
Tokens: [no-triality] [p0-ledger-ok]

## Fire Order

Use `.omx/research/ddm_mx1c_20260807/launch_ticket_v2_two_arm_governed.json` as the current
Row-1 v2 two-arm ticket.

1. Re-read `RECEIPT.md` and the JSON ticket.
2. Before any fire, prove liveness with a successful enumerator. If `pgrep` returns `rc>=2`,
   run `ps axo command`; if `ps` fails or is denied with `rc!=0`, REFUSE. Do not interpret
   denied process enumeration as zero live candidates.
3. Run the ticket's Metal-host `mem_probe_command` and require
   `.omx/research/ddm_mx1c_20260807/row1_v2_two_arm/mem_probe_receipt.json` with `status=passed`
   and the required MLX/load-stage samples before training fire.
4. Fire `argv_n32_arm_cap` first through its existing `safe_run.py` wrapper.
5. Fire `argv_n32_arm_veh` only after cap completes, or after a composed measured-peak projection
   proves host headroom under the 116 GiB ceiling.
6. Do not fire either n120 arm until the two n32 CPU-torch verdicts select the scaled arm.

## Boundaries

- This arm did not run Metal training, a scorer job, archive build, or `upstream/evaluate.py`.
- The local MLX probe is blocked by inaccessible Metal in the sandbox; it is not clearance.
- The #254 static scan still reports 145 historical warn-only violations outside the two lifted
  target entrypoints.
- Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
