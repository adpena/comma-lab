# ZC1 Leg Receipt: P1A TW1 Repriced Knee Live Vehicle

Exit: BLOCKED-with-named-breakage.
Named breakage: `TW1_DRIVER_INTACT_BUT_LIVE_VEHICLE_INPUTS_MISSING`.
Axis: local CPU/source read; no scorer run.
Score claim: false.
Promotion eligible: false.
Verdict scope: INSTANCE, rerunning the existing TW1 driver against the current tq1c or ARM selection vehicle.

## RECALL EVIDENCE

Searches performed:

- `rg -n "tw1|token_waterfill|wr1_cell_records|pfs1|tq1c|cell_records|descent_receipt" .omx/research .omx/state experiments src tools`
- Targeted read of `experiments/ddm_tw1_token_waterfill_state_dependence.py`.
- Targeted read of `.omx/research/ddm_tw1_token_waterfill_state_dependence_20260801.md`.
- Targeted read of `.omx/state/main_hot_state.md` for the current live base.

Found beyond the charter seed:

- The TW1 pricing driver exists and is intact as a local CPU, no-scorer byte-pricing tool.
- Its defaults are hard-bound to the pfs1 archive and wr1 cell-record/receipt surfaces.
- The source receipt's controls reproduce wr1 state-token bytes for the pfs1/wr1 L16 lattice.
- The current live base is tq1c, with a different base and archive lineage.
- No tq1c/current-vehicle equivalent of the wr1 cell records and descent receipt was found in the searched scope.

What this changed:

- Running TW1 with default pfs1/wr1 inputs would produce a stale-object price, not the live-vehicle repricing requested by the charter.
- ZC1 therefore blocks rather than minting a misleading "live" row.

## Verdict

Blocked. The driver is runnable, but the required current-vehicle inputs are missing.

Required inputs before rerun:

- Current tq1c or ARM selected archive/member path.
- Current token lattice compatible with the driver semantics.
- Current cell records with the same field contract as TW1 expects.
- Current descent receipt that provides the driver controls and object id.

## Follow-On Disposition

QUEUED-WITH-A-FIRE-ORDER:

1. Materialize the current tq1c or ARM-CAP/ARM-VEH token lattice, cell records, and descent receipt.
2. Run `experiments/ddm_tw1_token_waterfill_state_dependence.py` with explicit `--archive`, `--cells`, and `--receipt` arguments.
3. Record whether the TW1 state-dependence/knee result still changes #984 byte ordering.
4. Do not use pfs1/wr1 defaults as live evidence.

Own-vehicle frontier line: unchanged, `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
