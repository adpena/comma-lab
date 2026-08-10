# ddm_rc64p — NATIVE TOKEN DECODE: buy the CPU axis on OUR vehicle

## Mission (operator 2026-08-10 "Continue with all. We have everything we need to do better")

The #998 verdict: our lc2 bytes are CPU-budget-infeasible — measured 1,958 s total vs the
1,800 s contest budget, with the **token decode = 1,777.6 s** (the constriction-Python loop)
and render only 180.8 s. The named cure is IN CUSTODY: PR135's public `rc64_backend.c`
(compiled with `cc` at decode time, granted off the shelf) plus our own Rust runtime line
(runtime-rs). Kill the 1,777.6 s term. No scorer needed — this arm is timing + symbol identity.

## Two raced routes (races-not-reputation; winner by measured decode seconds at byte parity)

**ROUTE A — native decode of OUR existing ANS payload (byte-identical archive).** Write a
C (their rc64_backend.c build pattern, compile-at-decode) or Rust (runtime-rs crate pattern,
tac-boundary-decode precedent: golden vectors + python_reference_equivalence_test) decoder for
the lc2 cross-state ANS token stream. Archive bytes UNCHANGED; pure wall-clock. Verification:
decoded symbol stream BIT-IDENTICAL to the constriction path on the full n600 payload.

**ROUTE B — RC64-formatted token payload + their C decoder.** Recode our token state into
their RC64 container, decode with their backend. Bytes CHANGE — coordinate with cp135's
same-state race (read `.omx/research/ddm_cp135_*` + its SSD store if landed; fd135 measured
their RC64 at 0.539946 B above its own model ideal, so byte cost on our state is an empirical
question, not a presumption). Only competitive if route A stalls or B's bytes win outright.

## Deliverables

1. lc2-lineage inflate variant with the native decode wired (fail-closed fallback to the
   constriction path on compile/import failure — the e4/rr3 precedent), full parse-back,
   symbol bit-identity receipt on all 600 pairs.
2. MEASURED decode timing: old vs new on this host, single-threaded AND 4-thread (contest CPU
   is 4-core/16GB — report the honest margin and label the hardware; the final authority is a
   Modal CPU row MAIN fires later, not this arm).
3. Bare-venv bootstrap smoke for the new path (the rr3 lesson: prove the compile/bootstrap,
   never assume host toolchain — `cc` presence check + typed exit).
4. ALWAYS KEEP THE PAYLOAD: recoded payloads + timing receipts to
   /Volumes/VertigoDataTier/pact/ddm_rc64p_20260810/ with sha256+bytes.
5. borrowed_substrate_accounting: rc64_backend.c = codexblack/PR135 (granted); the ANS format
   + inflate line = ours; Rust crate patterns = ours.
6. Durable memo `.omx/research/ddm_rc64p_native_cpu_decode_20260810.md`, serializer commit
   (post-edit --expected-content-sha256, tags [no-triality] [p0-ledger-ok]). Checkpoint per
   protocol. Do NOT dispatch Modal; do NOT take the scorer slot (none needed).

## OPTIMAL FORM

Reference: the runtime-rs tac-boundary-decode discipline (Python oracle + golden vectors +
bit-identical parity gate + payload-cleanliness audit) and their rc64_backend compile-at-decode
pattern. SCOPE = the token-decode term only (render 180.8 s already fits); MECHANISM reduction
FORBIDDEN (real full-n600 payload, real timing, no extrapolated seconds). Pins: lc2 archive sha
`f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45`; PR135 runtime custody
`/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr135_src/` (READ-ONLY); #998 receipts in
`.omx/research/ddm_lc2_adjudication_and_cpu_verdict_20260810.md`.
PRIOR-LAW PREDICTION (from F26's reported much-shorter token decode on the same coder class +
the general 50-500× native-vs-Python entropy-loop ratio): route A lands the token decode under
120 s on this host, total under the budget with margin; falsifier = native decode >900 s
single-threaded, which would mean the cost is NOT the Python loop and the #998 mechanism
attribution needs re-measurement.

## Falsifier

Neither route fits the derived budget margin at symbol identity → the CPU axis on this carrier
family is decode-structure-limited, not implementation-limited; record the per-stage timing
decomposition and close honestly (INSTRUMENT/host-scoped, contest-CPU proof still owed).

## ADDENDUM (MAIN, 2026-08-10 ~20:55Z — operator grant; consume at next checkpoint)

Your falsifier FIRED and your receipts prove it: symbol recovery is 1.11s constriction / 1.78s
native (0.08-0.12% of wall); ~99.9% of the #998 "token decode" term is HPAC PROBABILITY
GENERATION. The operator has granted, verbatim: **"You can lower anything into rust"** +
**"And otherwise optimize."** Consequences for this arm:

1. **The HPAC model itself is in scope for native lowering** — not just the entropy coder.
   Rule-118: decoder CODE (the HPAC context-mixing algorithm) is FREE; its learned parameters
   are ALREADY counted archive bytes. Lowering changes ZERO archive bytes — pure wall-clock,
   the route-A byte-identity property preserved.
2. **Full optimization authority, any route**: Rust (runtime-rs crate discipline —
   tac-boundary-decode precedent: Python oracle + golden vectors + bit-identical parity test +
   payload-cleanliness audit, `video_derived_constants_embedded_in_native_source` must stay []) ·
   C via the cc-at-decode pattern (their rc64_backend.c precedent, toolchain-presence proven) ·
   vectorized integer numpy · algorithmic restructuring (batch independent streams/cells;
   probability generation is per-symbol-serial WITHIN a stream but parallel ACROSS independent
   contexts) · 4-thread (contest CPU = 4-core; you already built the --threads {1,4} worker).
3. **Bit-identity is the gate, not a preference**: the integer-HPAC forward must reproduce the
   full 117,964,800-symbol stream sha (c5c7671d…) exactly. Integer arithmetic only in the hot
   path — the "Native eval-time runtime discipline" deterministic-integer-kernel case.
4. **Bootstrap honesty (rr3)**: GitHub Actions ubuntu-latest images ship rustc/cargo AND cc —
   but VERIFY, never assume: toolchain-presence check + fail-closed fallback to the pure-Python
   path + compile time counted inside the 30-min budget. Prefer whichever of {Rust, C} your
   measured bootstrap proves; race if cheap.
5. Deliverable unchanged in shape: measured old-vs-new per-stage timing (1-thread AND 4-thread),
   symbol identity receipt, payloads kept, honest margin vs the 1,800s budget. If even the
   lowered model misses budget, the falsifier close stands — but now measured at the TRUE term.
