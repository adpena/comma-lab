# G26 terminal G14 feedback-costate materializer — Codex findings (2026-07-26)

Lane: `lane_g26_g14_feedback_costate_materializer_20260726`  
Status: L1 implementation-complete, `research_only=true`  
Pointer delta: none; no scorer, eval, dispatch, candidate, archive mutation, or pointer write

## Landed contract

`taskspace_feedback_costate_materializer_v1` consumes only canonical terminal
G14 `final_receipt` bytes and a strict serialized C0B `PairPopulation` envelope.
It reopens the population, requires exact ordered source IDs `0..599` across
V9/PBR/IR/V10, derives the population identity, builds frozen G18 feedback, and
derives the complete G19 exact-base binding from those reopened inputs. A caller
population digest is comparison-only and a mismatch refuses; it can never mint
the controller population identity.

The CLI reads only explicit non-symlink regular files and writes one atomic,
write-once-or-equal receipt. It has no live-run, scorer, evaluator, dispatcher,
candidate, or pointer-mutating option. A partial G14 checkpoint/manifest is not
an accepted input type.

## G20/G22 placement closure

G19 now accepts `FullN600ReceiverEqualityClosureV1`, constructible only from
exact canonical G20 receipt bytes, exact canonical final G22 receipt bytes, and
the exact serialized PairPopulation bytes. Construction refuses unless all of
the following match:

- source and selected whole archives and `0.bin` members, SHA-256 and bytes;
- frozen generic decoder runtime, SHA-256 and bytes;
- G20 decoded full-quantized-state equality and exact G20 receipt custody;
- exact ordered pair IDs `0..599` and the same PairPopulation identity;
- 600 pairs / 1,200 frames / all 3,662,409,600 realized uint8 bytes per arm,
  directly compared with equal whole-output hashes;
- immutable run manifest, every checkpoint custody row, pre-cleanup receipt,
  cleanup certificate, cleanup completion, success-only cleanup, and full-n600
  reviewed argv; and
- research-only truth with full receiver replay closed but contest CPU/CUDA
  same-byte evaluation still owed.

The decoded-byte proof dependency hash contains archives, members, runtime,
pair population/order, replay tool, comparison algorithm, and realized output.
It deliberately excludes the competitive pointer. Pointer artifact identity and
semantic target identity are separate observations. A later metadata-only
pointer refresh leaves decode equality valid and does not request admission
rebase; a semantic target change marks only admission rebase and still leaves
decode equality valid.

G20's whole-object placement handoff changes from
`BLOCKED_FULL_N600_REPLAY_OWED_NO_WRITE` to
`READY_RECEIVER_EQUALITY_CLOSED_CONTEST_EVAL_SEPARATELY_OWED_NO_WRITE` only when
that exact typed closure matches the G20 placement. It still does not grant score
or promotion authority.

## Verification

- focused materializer + CLI + G19 suite: `27 passed`;
- composed G18/G19/G20/G22/PairPopulation plus allocator/costate-authority suite:
  `171 passed`;
- Ruff: clean;
- `py_compile`: clean;
- lane registry: `2173 lane(s) validated cleanly`.

The full-n600 acceptance receipt in unit tests is structural parser-contract
coverage only, not an empirical replay. No live G14 partial or final receipt and
no live full-n600 G22 receipt were consumed. Production execution remains with
root after those exact final receipts exist.

## Artifact hashes

- `src/tac/witness_control/taskspace_interaction_costate_bridge_v1.py`:
  `94070f46f23a78fb9e12939bc4c64c832d70d79fab6426215a72c59e7737323b`
- `src/tac/witness_control/taskspace_feedback_costate_materializer_v1.py`:
  `b027f80f1de48113b57057c463d86df864a57316e4343300ba1d11c376cd37cf`
- `src/tac/witness_control/tests/test_taskspace_feedback_costate_materializer_v1.py`:
  `94b074fac0d4e694868e3848d4505dec6dc2ce48549eb3fcaae6f294ed1a5429`
- `tools/materialize_taskspace_feedback_costate.py`:
  `40fd1d051577599aa3c6039b6aab5fd6c05083ee0671031fa985990c01aba7ad`
- `tools/tests/test_materialize_taskspace_feedback_costate.py`:
  `d6c5b0829b1d878e16659b0a386d411097b3d3535ed6720bfa6e1ef829a0a614`

HISTORICAL_PROVENANCE: G26 implementation receipt. This landing extends G19
but does not rewrite or consume G14/G23/G25 owner files.
