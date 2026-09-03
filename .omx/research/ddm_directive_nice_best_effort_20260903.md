# DIRECTIVE (MAIN, 2026-09-03 ~21:0xZ) for live arms ddm_gc1 / ddm_gf2 / ddm_cd1 and every later arm — detached launches in a sandbox

`tools/launch_detached_process.py` refuses a launch at rc=8 when `--nice N` cannot be applied (sandboxes cannot
`setpriority`). gc1's detached measure launch was refused this way and the arm ran the compute in-session instead —
strictly worse (default priority beside the governed Metal burn, and subject to the session reaper). The launcher now
accepts `--nice-best-effort`: pass `--nice 10 --nice-best-effort` on every detached launch from an arm; the manifest
records `nice_status=unapplied:best_effort` and the launch proceeds. The strict default is unchanged for MAIN.
