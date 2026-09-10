# RLC1 rebuild and retained-state instructions

All commands run from `/Users/adpena/Projects/pact`. All large outputs belong under
`/Volumes/VertigoDataTier/pact/ddm_rlc1_rule118_cure` (8 GiB total). Do not overwrite
completed outputs or source/pointer runtimes. Source archive is move40 sha986d536b…;
its exact runtime digest is in `SOURCE_RUNTIME_BINDING.json` in that SSD store.

1. Build the geometry library using `retained/build_geometry.json`'s exact argv.
2. Run `experiments/ddm_rlc1_run.py encode --resume-from /Volumes/VertigoDataTier/pact/ddm_rlc1_rule118_cure` with `.venv/bin/python -B` through `tools/launch_detached_process.py --nice-best-effort`. This verifies source bindings, resumes at the last immutable frame, retains both candidate streams and the source reconstruction, and keeps every stage checkpoint.
3. Run the same driver with `stage` to retain all 40 predeclared container samples and stage the smallest qualifying archive. No learned content enters receiver code.
4. Run `experiments/ddm_rlc1_public.py --resume-from /Volumes/VertigoDataTier/pact/ddm_rlc1_rule118_cure --threads 1` and again with `--threads 4`, through the governed launcher. These are full literal public paths; the environment varies BLAS while Torch remains four-threaded. At most two computation workers. Every raw output is retained; never reuse another arm's decoded field as proof output.
5. Run `experiments/ddm_rlc1_smoke.py probe --role candidate`, `probe --role frontier`, both `shell` roles, and `collect`. These are explicitly setup/gate smokes, not the identity proof.

Source-bound reruns refuse if code changed. Existing public proof generation g2 is
an actual experiment identifier: initial wrappers failed before decode when `ps`
execution raised PermissionError. Both failures and all initial files are retained.
For a future source change, mint a new bound generation. Never rewrite INPUTS.json
to force a resume. If an output exists without its final receipt, recover the receipt
from the saved log/output bytes; do not discard the raw or rerender blindly.

Timing requires a trusted whole-host concurrency count and matched T4/local calibration.
A blocked `ps` inventory is UNKNOWN, never zero. The current candidate cannot inherit
move40 timing because its receiver code differs. MAIN alone may fire after a valid
seal and pr9 second-family clearance. No Modal dispatch is authorized for this arm.

For an interrupted g2 public run, preserve its native cache and set `CC` to
`/Users/adpena/Projects/pact/experiments/ddm_rlc1_cached_cc.py` and
`RLC1_NATIVE_CACHE` to that run's `retained_native` directory before invoking the
same unchanged public driver. The cache verifies source, flags and native bytes.
A fresh Mach-O build has a different hash and would fail checkpoint binding.
The exact library bytes captured from both original shells are retained. The
separate `ddm_rlc1_resume_smoke.py --resume-from <arm-root>` test exercises a real
frame25-to26 public restart; it never substitutes for either cold n600 proof.
