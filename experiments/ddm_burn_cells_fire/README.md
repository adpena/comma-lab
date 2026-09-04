# Burn-cell fire specs

**The shell fire scripts are GONE (ddm_gov2, 2026-09-04).** A cell is launched by exactly ONE surface:

```bash
.venv/bin/python tools/cell_queue_driver.py fire \
    --queue experiments/ddm_burn_cells_fire/burn_cells_queue.json --cell-id <id> [--dry-run]
```

`burn_cells_queue.json` is the converted, lossless replacement for `fire_ng2_cap_cell.sh`,
`fire_ng3_tau_cell.sh`, `fire_ng4_continuous_cell.sh` and `wait_then_fire_ng4.sh` (git and SSD copies were
byte-identical, sha-verified, and every launcher argv is preserved verbatim in the spec). Why they went: each script
carried its OWN admission rule. ng3's computed reclaimable memory as `free + inactive` — the reclaimable-BLIND basis
that `check_no_raw_virtual_memory_safety_basis` refuses in Python, which escaped the gate only because the gate parses
Python ASTs. ng2 declared `--measured-peak-rss-gib 2.3959503173828125` for a family whose MEASURED cost is
**49.572 GiB** (20.69x). With two such cells live the VM compressor reached 76.978 GiB and 72.0 GiB of swap on a
128 GiB box, and jetsam killed background daemons.

`fire` runs, in order: the seal law (pins verified inside the firing tree), the duplicate-done-receipt check, the
storage waterfall, the MEASURED-PEAK LAW (`"measured_peak_rss_gib": "from_ledger"`; a typed number below the family's
measured row is REFUSED), memory + concurrency admission, the lane claims, the chain driver's authorize, and the
canonical launcher — with the measured peak substituted into its argv.

Enforced by STRICT preflight **Catalog #413** `check_cell_launches_only_through_queue_driver`: any `experiments/**`,
`tools/**` or `scripts/**` file that EXECUTES `launch_detached_process.py` with a `run-config` argument is refused
outside the driver (same-line waiver `# CELL_FIRE_PATH_OK:<rationale>`).

The `authorize_*.py` scripts remain: they bind claim ids through the chain driver's own `authorized_config` /
`write_or_verify_authorized`, never a hand-edited JSON. `cell_queue_driver.authorize()` supersedes them
functionally; they are kept as the historical record of each cell's authorization.

Memo: `.omx/research/ddm_gov2_control_plane_permanence_20260904.md`.
