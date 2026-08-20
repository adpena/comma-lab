# CHARTER ADDENDUM (MAIN, post-spawn — operator correction "we have memory guards and governor")

Surface 3(a)'s premise is CORRECTED: do NOT ask "does the class need a structural gate" —
the structure EXISTS (system_memory_governor + admission_guard #254 + safe_run/governed
launchers + memory guard #172 + waterfill #294/#298). Audit the BYPASS instead:
(a1) WHY the #254 assert never fired: the lifted entrypoint never adopted it (opt-in
     contract) + filename evades the train_*.py detection vocabulary + MAIN spawned via
     ungoverned launch_detached_process.py. Verify each leg at file:line; check whether
     admission enforce is armed (.omx/state/admission_enforce.flag / TAC_ADMISSION_ENFORCE).
(a2) Is there a static preflight gate that scans heavy entrypoints for the assert? If yes,
     name its detection vocabulary and the class-fix (pattern → registry/argparse-shape).
(a3) Grade MAIN's ungoverned spawn against the #254 docstring's own contract — the guard's
     stated failure mode is EXACTLY what happened; that is a finding regardless of cures.
Corrected memory: concurrent_metal_fires_without_composed_preflight_oomed_the_machine_20260806.
