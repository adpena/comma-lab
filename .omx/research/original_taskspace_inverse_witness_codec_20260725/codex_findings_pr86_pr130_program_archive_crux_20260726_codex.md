# Codex findings — PR86/PR130 lessons and the program-to-archive crux

Date: 2026-07-26  
Verdict: `IMPLEMENTATION_BLOCKER_CONFIRMED`  
Authority: architecture and adversarial-review finding only; no candidate or score claim

## Canonical state

- Dynamic competitive pointer: strict score below `0.172`.
- Pointer SHA-256:
  `2a61b052be496d3a9a1be1a9c230c8d179a788e61fd03472e50fc85832da94c6`.
- G57 did **not** test the intended selected-preimage codec. It tested a
  direct, resized-source, two-plane x264 control and scored
  `39.30593503092899` on `[macOS-CPU advisory]`.
- G58 now defines a strict G49-program-to-fresh-operand bridge, but it does not
  synthesize the selected-preimage factors.
- G59 recurrently gates `CAMPAIGN_SEAL -> PRE_ENCODE -> ENCODE -> POST_EVAL ->
  PRE_PUBLIC_CLOSURE -> PRE_PROMOTION`. It deliberately refuses the current
  program campaign because the production compiler/config/runner does not yet
  exist.

## Donor facts, not donor payload

No PR86 or PR130 archive, token stream, model, weights, basis, coefficients, or
historical candidate bytes are candidate inputs here. These public sources are
mechanism evidence only.

### PR86

Sources:

- <https://github.com/commaai/comma_video_compression_challenge/pull/86>
- <https://github.com/jas0xf/comma-anr-supplementary/blob/master/writeup.pdf>

Measured lessons:

1. Optimizing adversarial pixels and then asking a conventional codec to
   preserve them failed: the task signal was high-frequency and the codec/rate
   surrogate did not match the real archive.
2. The two frame roles are asymmetric. The last frame must carry SegNet
   decision/boundary state; the first frame only needs to make the pair land in
   the PoseNet preimage.
3. A flat class palette recovered most SegNet decisions; the expensive debt
   concentrated near boundaries. This is evidence for region topology plus
   sparse boundary carriers, not dense photographic reconstruction.
4. The exact semantic token population remained the dominant rate object, so a
   learned causal spatial/temporal entropy model was used instead of a generic
   compressor.

### PR130

Sources:

- <https://github.com/commaai/comma_video_compression_challenge/pull/130>
- <https://fesalfayed.com/blog/semantic-pose-compression/>

Measured lessons:

1. Task factorization reached a `191,052`-byte archive. The public PR reports
   displayed `d_seg=0.00028609`, `d_pose=0.00001967`, and approximately
   `0.1698476624` from displayed components. An independent reproduction in
   the PR reports exact score `0.17076926565506415`.
2. Its byte ledger exposes the remaining structure: about `116,980` bytes for
   the semantic stream, `40,252` raw bytes for the int4 renderer, and `23,054`
   bytes for the pose carrier before the final ZIP overhead/accounting.
3. Integer-lattice deployment, exact symbol gates, and entropy coding matched
   to the latent type mattered. Generic video and generic byte compressors did
   not.
4. The semantic and pose representations were trained separately and joined
   at pair assembly, but PoseNet still evaluates their composition. Our pose
   solve must therefore be conditional on the exact decoded semantic frame.

## Original re-expression in our geometry

PR86/PR130 demonstrate that task factorization is necessary. They do not close
the stronger quotient available to us:

1. We do not need to preserve a particular reference token map. We may select
   any uint8 realization inside the same SegNet argmax cells.
2. We do not need a standalone pose image. We may select any first frame in the
   conditional fiber of PoseNet given the exact selected semantic second frame.
3. The full-lattice teacher is an encoder-side feasibility oracle, not a
   payload. Its dense solution should supervise the selection of the cheapest,
   most shared, most factorizable representative in each evaluator cell.
4. The representation should be the shortest program for the selected
   solution: shared topology, temporal evolution, boundary exceptions,
   conditional pose factors, and only then a trained irreducible quotient.

The required differential controller is:

`dS = 100*d(d_seg) + 5/sqrt(10*d_pose)*d(d_pose) + 25/37_545_489*d(bytes)`

Consequently no fixed independent target for Seg, Pose, or rate is valid.
Every actuator is admitted on the joint same-object score differential and its
real compressed byte price.

## The missing production compiler

The missing executable is not another plane codec. It is a fresh n600 compiler:

`fresh target custody + own semantic compile + full-lattice/costate teacher`

`-> shared semantic topology/region program`

`-> temporal boundary and island evolution`

`-> sparse selected-preimage boundary correction`

`-> conditional Y0 | decoded Y1 pose factors`

`-> optional learned irreducible quotient`

`-> exact G49 packet + exact fresh semantic member`

`-> public generic decoder in inflate.py`

The packet and semantic member are the primary codec. x264 or another standard
codec may be a typed terminal quotient only if the joint allocator proves that
its score-unit gain exceeds its exact byte cost.

## Adversarial review finding

The first independent review of G58 found a release-blocking custody bug:
arbitrary extra ZIP members could be added under neutral names while the proof
claimed that no target or historical payload was embedded. A concrete probe
placed `786,432` bytes of target labels under a neutral member and obtained an
admission.

The fix makes the G58 outer payload a complete partition: exactly the fresh
semantic member and the G49 packet member, no third member. G58 now reads the
archive from one descriptor-stable byte image; the strict production verifier
independently re-enumerates the complete member set. A neutral-name regression
test must remain in every release suite.

The next independent pass found three additional lifecycle defects before
commit:

1. direct-control launchers requested a live program-candidate receipt that
   G59 correctly could never issue;
2. exact-eval staging required `PRE_PUBLIC_CLOSURE`, while that receipt itself
   required the exact-eval row, forming a cycle;
3. full-chain reopening checked links and identities but did not require
   adjacent stages, so a canonically sealed chain could skip boundaries.

The fix keeps direct G52 build/eval staging explicitly research-only, requires
G59 only on promotion/candidate surfaces, and makes every live chain equal the
exact adjacent lifecycle prefix. The future program-residual producer remains
blocked until its typed config and runner exist; this is deliberate and must
not be bypassed by attaching candidate authority to the direct control.

## Executable order

1. Land and commit the G58 complete-member partition and adjacent-prefix G59
   recurrent gate.
2. Implement the production compiler/config schema and one resumable
   `PROGRAM_RESIDUAL_LAYERED` runner. This is the current P0.
3. Compile analytic factors first on five 120-pair stages; measure exact packet
   bytes and decoded selected-plane hashes at every checkpoint.
4. Fit only residual score-valued debt that the analytic program cannot
   represent.
5. Run n2/prefix falsifiers, then the full n600 batch-16 same-object scorer.
6. Let G59 force post-eval integration and public double-decode closure before
   promotion.

## Triality and stores consulted

- DSL: G49 selected-preimage factor roles plus the owed production config
  schema.
- DAG: the six recurrent G59 boundaries above.
- Equations: the exact contest action and its differential controller.
- Stores consulted: `CLAUDE.md`, `AGENTS.md`, canonical frontier pointer,
  G49/G50/G57/G58/G59 artifacts, upstream PR86, PR86 write-up, upstream PR130,
  and the PR130 technical write-up.

Pointer delta: zero. The direct formulation is killed; the original
selected-preimage family is explicitly not killed.
