# ddm_ps135b — GEN-2: clear parity, then FIRE Leg A (the actual pose re-solve)

## Mission (operator 2026-08-10 "Try 2" — the second attempt starts NOW, not at the storage gate)

You are the continuation of ddm_ps135_pose_resolve (gen-1 landed honestly at its fail-closed
boundary, memo `.omx/research/ddm_ps135_pose_resolve_20260810.md`, commits d6ea363904/452ce890b5/
f5615682e8). RECALL FIRST: read that memo + your own NEXT_IF_RESUMED rows + the SSD store
`/Volumes/VertigoDataTier/pact/ddm_ps135_20260810/` — the apparatus (Leg-A runner
`experiments/ddm_ps135_pose_resolve.py` + Stage-C driver `experiments/ddm_ps135_stage_c_mixed_precision.py`)
is BUILT and tested (54 green). Do NOT rebuild; RESUME.

## Ordered work (storage-independent FIRST — do not idle on the gate)

1. **BLOCKED-PARITY (fire immediately, no storage needed):** the pinned Python/Torch runtime
   timed out during import; exact q4 pair-599 master parity is unmeasured (wrong-runtime
   control differed by 12 bytes). Diagnose the import timeout (venv health / torch pin /
   MKL-OMP thread stall — check the si1/bare-python + shared-venv-hijack precedents), make the
   pinned runtime import responsively, produce + validate exact q4 pair-599 parity at the
   required thread geometry. This clears your own Stage-C parity receipt store.
2. **STORAGE GATE WATCH:** a certified move is freeing ~25 GB on Vertigo RIGHT NOW
   (detached job `vertigo_ai1_move`, receipt `.omx/tmp/codex_runs/vertigo_ai1_move.done`,
   manifest at `/Volumes/APDataStore/pact/ddm_ai1_20260809_MOVE_MANIFEST/MOVED.json` when done;
   expected within the hour). Poll cheaply between parity work units. When Vertigo free
   ≥3,000,000,000 B AND your committed preflight passes AND the host process scan is empty AND
   you claim the sole scorer lane → **FIRE Leg A** per your own fire trigger.
3. **Leg A = the pose re-solve itself** per the gen-1 charter + your LIVE-HYPOTHESES:
   LC2-native coefficient re-solve (their re-solve never touched LC2's exact PR130 raw/renderer
   lineage) · radius>1 GN · wrong/global starts (the ±1 identical continuation is CLOSED —
   your own zero-accept receipt). Retain EVERY candidate payload (the law that blocked you is
   the law you honor).
4. **Stage-C compensation** stays owed before ANY rate-archive claim (the 5 parse-backed
   archives 187,226→186,393 B are scorer-free artifacts, never score improvements — your words).

## Grants in force (operator 2026-08-10)

"You can lower anything into rust" + "And otherwise optimize": if the solver's hot loops need
it, native lowering (Rust runtime-rs discipline / C) + any optimization route is AUTHORIZED —
bit-identity vs the Python oracle + rr3 bootstrap proof gate every lowering. See
`.omx/research/charters/ddm_rc64p_native_cpu_decode_charter_20260810.md` ADDENDUM for the full
clause set.

## Boundaries (unchanged)

No Modal dispatch (MAIN fires exact rows). Serializer commits w/ post-edit
--expected-content-sha256, tags [no-triality] [p0-ledger-ok]. Payloads to the SSD store with
sha256+bytes. Durable memo appends to your gen-1 memo (ADDENDUM section, never rewrite).
Honest fail-closed close if a gate cannot be satisfied — name the blocker, as gen-1 did.

## OPTIMAL FORM

Reference = your OWN gen-1 apparatus at its landed form (no mechanism reduction; SCOPE
unchanged: full n600, real carrier dims, MIN_PASSES=8 + DRY_PASSES=3 + JRD lineage).
PRIOR-LAW PREDICTION (from your LIVE-HYPOTHESES, derived fresh at gen-1's landing): LC2-native
re-solve + global starts recovers a measurable share of PR133's pose gain on OUR lineage;
falsifier = zero accepted rows across all start families at radius>1 → the singleton-neighborhood
exhaustion extends to the global space on this lattice, close the family honestly.
