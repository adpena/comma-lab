# Rebuild and verification

Public code is the seven files in submissions/mrs5. Its archive is the unchanged
move53 payload. Run its documented extraction and inflate.sh commands; generated
libraries stay out of the source delivery. CPU output is an advisory proof only.

All bulk proof stores live in `/Volumes/APDataStore/pact/ddm_mrs5`. Heavy runs use
`tools/launch_detached_process.py --done-receipt <name>`; exact invocations and
storage admission records are retained under `launch_*/launch_manifest.json`.

1. Build a COPY of the public tree using its own `inflate.sh --help`. Run
   `experiments/ddm_mrs5_proof.py --resume-from <fresh parity store>
   --receiver <compiled copy>/inflate.py`. The fixed source-bound mrs2 bank
   supplies 48 stratified pre-pair states, two seeded random pairs per 25-pair
   stratum. All geometry bins, tokens and sampled raws are retained.
2. `experiments/ddm_mrs5_bootstrap.py --resume-from <bootstrap store>` rebuilds
   the bare APFS environment from retained offline wheels. No system site
   packages or user site is included.
3. `experiments/ddm_mrs5_smoke.py --resume-from <fresh native store>` runs the
   actual cold n600 public entrypoint and checks the complete raw hash and all
   600 pair hashes against move53 before certified duplicate-scratch cleanup.
4. The same smoke with `--without-compiler` in a separate store injects stale
   libraries, runs the real shell without cc, verifies all three disappear,
   then decodes/renders the same48 pairs using Python only.
5. After both smokes and bootstrap cleanup, run
   `experiments/ddm_mrs5_timing.py --resume-from <fresh timing store>`. Its A/B/A/B
   order is fixed. A uses the immutable sealed shell and its native builds; B
   uses a source-identical snapshot and its shell. An external PATH shim samples
   the 48 state-restored pairs, using A's original decoder body and each own
   renderer. This is process-cold sampled evidence, NOT cold n600 or an unchanged
   public invocation. The separate cold proof in step3 covers that surface.
   Timings include shell builds, setup and retained-output work; compute-only
   sensitivity excludes state restore and output retention. Common overhead
   can dilute wall ratios. Inspect both and the recorded load drift. Interrupted
   sequences are preserved, and resume starts a new complete four-run sequence.
6. Preserve the unrounded ratio and its spread. Projection is conditional on
   serial-Python scaling, unchanged GPU stage semantics and approximately
   stationary load. If it exceeds1260 seconds, stop before intent production.
   Otherwise pass truthful receipts to the original make_candidate_seal.py
   producer. STOP on its actual typed refusal; never modify the frozen contract,
   add forbidden pins, counterfeit encoder executions or relabel n48 as n600.

The bootstrap `--cleanup` path inventories installed files before reclaiming
its reproducible APFS venv. Retained wheels, sources, exact commands, hashes and
completed stage receipts remain. Retention at handoff is <=2 GiB; the transient
3.66GB raw is certified reproducible duplicate scratch before cleanup.
MAIN lands the exact source tar. Any valid future intent must itself be committed
BEFORE MAIN authorizes a first measurement. There is no dispatch here.
