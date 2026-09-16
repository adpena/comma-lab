# Rebuild and verify

The source delivery contains exactly the six public files plus internal evidence
and the four external proof and audit scripts. Compile and decode through the documented
public `inflate.sh` command. No native library is shipped.

Internal proof order, from the repository root:

1. Use `experiments/ddm_mrs4_proof.py --mode parity --receiver <compiled-copy>/inflate.py
   --resume-from /Volumes/APDataStore/pact/ddm_mrs4/<new-parity-store>` with the
   SHA-bound 48-state predecessor bank. The retained original harness snapshot
   identifies the exact code used for the first proof.
2. Recreate the isolated APFS environment with
   `experiments/ddm_mrs4_bootstrap.py --resume-from /Volumes/APDataStore/pact/ddm_mrs4/bootstrap`.
3. Run `experiments/ddm_mrs4_smoke.py --resume-from /Volumes/APDataStore/pact/ddm_mrs4/public_native`
   for the cold public shell. Completed stages are reusable only with the same
   source binding. The full raw is certified against all 600 oracle pair hashes
   before automatic cleanup of this reproducible scratch.
4. Run the same smoke with `--without-compiler` in a separate output directory;
   both stale libraries must disappear and all 48 fallback pairs must agree.
5. Run the proof script with `--mode profile` and `--mode python` in separate
   stores. The Python profile copy has the range library but no corrector library.
   Nested stage rows overlap their parents and must not be summed.
6. Rehash and compare retained vectors independently with
   `experiments/ddm_mrs4_audit/python_reference_equivalence_test.py
   --resume-from /Volumes/APDataStore/pact/ddm_mrs4/parity`.

All heavy commands use `tools/launch_detached_process.py --done-receipt <name>`.
Every completed pair or subprocess stage has a source-bound disk receipt.
An interrupted public subprocess restarts cold and preserves its partial raw
until a complete replacement is certified. Venv cleanup uses the bootstrap
script's `--cleanup`; the installed-file inventory and wheel hashes survive.
The exact executed commands, compilers, environments and receipt paths are in
the retained launch manifests. Never reuse a proof store after changing a source.
