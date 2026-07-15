# DAG FEED — DSL compile hash enforcement

**Catalog:** #406  
**Lane:** `dsl_hash_enforcement_20260715`  
**Mode:** BUILD + local verification only; no training/provider dispatch  
**Pointer delta:** UNMOVED

## Authority flow

```text
TypedWitnessConfig
  -> WitnessProgram.validate()
  -> WitnessProgram.compile_trainer_argv_with_constants()
  -> #332 ConfigBijectionSnapshot + bijection_hash
  -> resolved LawRef records
  -> canonical dsl_compile_hash
  -> {launch.sh, dsl_provenance.json, launch_manifest.json}
       -> launcher exact-byte reopen + typed-program recompile
       -> durable governor exact-byte reopen + typed-program recompile
       -> trainer sys.argv round-trip
       -> native-dispatch admission binding
       -> ADMIT
```

Every missing edge terminates at rc 8 before spawn. The old manifest-absent,
dry-run advisory, skip, rationale override, and verifier-error fail-open branches
have no authorization effect. A post-compile semantic flag cannot be appended:
it has no #332 Lever owner and therefore has no valid compile binding.

## Hash law

`dsl_compile_hash` uses the existing #332 canonical JSON SHA-256 helper over:

1. canonical typed WitnessProgram spec;
2. canonical resolved trainer argv tuple;
3. canonical #332 flag-to-Lever bijection manifest plus its bijection hash;
4. canonical resolved LawRef provenance.

Run identity (`out_dir`, run-id, label, purpose), observation timestamps, and
observed staleness age are preserved only as non-authoritative context. The
interpreter, semantic tokens, values, ordering, types, ownership, LawRefs, and
consumer locations remain in the binding.

`# NO_EQUATION_NEEDED: cryptographic provenance integrity invariant, not a new scientific witness-dynamics law.`

## Triality and consumers

- **DSL:** `tac.v9_provenance_gates.build_dsl_compile_provenance_document` is
  the sole binding builder and extends, rather than duplicates, #332.
- **DAG:** this FEED preserves the four admission consumers and refusal edges.
- **Equations:** no new scientific equation; LawRef records already identify
  the governing value laws and are themselves hashed/recompiled.
- **Sensitivity/Pareto/bit allocator:** N/A; apparatus-only, score-neutral.
- **Cathedral/autopilot:** indirect active guard: invalid witness launches never
  enter durable execution or emit empirical anchors.
- **Continual learning:** only DSL-custodied run results may become anchors.
- **Probe disambiguator:** exact mutation tests arbitrate valid compile versus
  missing, forged, or post-edited artifacts.

## Evidence and scope

- `v9_cgauge_ideal_mod19` dry-run: compiled 219/219 real flags, emitted all
  three launch artifacts, printed a deterministic hash, and returned rc 0.
- Launcher post-compile `--component-wallclock-telemetry`: REFUSED rc 8.
- Durable governor raw witness trainer argv: REFUSED rc 8 before admission/Popen.
- Pointer and score custody are unchanged; this is LOAD-BEARING MEANS only.
