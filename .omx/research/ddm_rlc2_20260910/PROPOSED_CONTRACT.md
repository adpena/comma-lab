# Proposed first-fire lifecycle contract — NOT ADOPTED

Owner of decision: MAIN and the second family. This document grants no dispatch permission.

The current cycle is impossible for an unseen receiver: a valid seal requires a completed timing
receipt, but the charter requires a valid seal before the first fire that produces that receipt.
The smallest change is a separate admission state for that first measurement, not a weaker timing
receipt or a relaxation of the completed-seal validator.

1. Add a distinct, versioned **pre-fire intent** schema and typed validation result. Its timing state
   says measurement is required; it carries no measured/projected seconds, no timing clearance, and
   `score_claim=false`. Existing normal seal validation must continue to refuse this object.
2. Freeze and validate all existing non-timing gates: actual archive bytes and digest, full runtime
   and normalized receiver digests with named definitions, receiver pins, manifest, current pointer
   and derived admit bar, both public smoke groups, retained payload paths, and falsifiers. For this
   candidate also require the real twin encode, complete cold n600 raw-identity receipt, and pr9's
   tree-specific literal census. Do not copy ancestor measurements into candidate fields.
3. MAIN's specifically authorized first-fire consumer may accept this separate schema only after
   the second family's prospective rule is committed and hash-frozen. Bind the authorization to the
   intent digest, exact candidate, contest-CUDA T4 axis, cold n600 public entrypoint, lane, retained
   receipt destination, budget and timeout. Record the intent digest in dispatch custody. Retain the
   existing lane, resource and spend guards. Unknown states, missing authorization and identity or
   pointer drift refuse before dispatch. This is measurement permission, never timing clearance.
4. After MAIN harvests that exact fire, run the **unchanged** `build_t4_direct_leg` validator: exact
   archive/runtime/receiver identity, completed successful canonical CUDA n600 path, T4 hardware,
   cold report, and actual inflate time at or below 1,260 seconds. Timeout, missing result, failed
   execution, warm/resumed decode, drift or excessive time must block the transition.
5. Issue a new completed seal that names the immutable intent digest and resulting receipt. Validate
   all gates again, including the current pointer/admit bar. Do not overwrite the intent or mutate a
   historical seal. Promotion remains a separate exact-score decision; rlc2 has no CPU score by this
   process, and no public PR update is authorized.

Implementation surfaces if approved: `src/tac/candidate_seal.py`,
`tools/make_candidate_seal.py`, `tools/fire_modal_auth_eval.py`, their focused tests and dispatch
receipt binding. `src/tac/decode_wall_clock.py` retains the completed `t4_direct` requirements.

Required verification before use: a real producer must emit a complete pre-fire object from a
fully proved candidate; the consumer must accept that object only in its first-measurement mode,
refuse it as a completed seal, and reject pin, pointer, authorization and timing drift. MAIN's real
harvest supplies the positive completed-timing receipt. Handwritten passing fixtures cannot be the
only positive control.

No such schema, CLI option, authorization or transition was implemented by ddm_rlc2.
