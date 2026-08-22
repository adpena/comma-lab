RC1 is built, measured, retained, sealed, and landed in commit `35710b32bd`.

The selected K=2,048 terminal temporal-program representation measured:

- Counted RC1 payload: **59,884 B**
- Complete research shadow archive: **113,006 B**
- Cut versus DX2: **67,362 B**
- Headroom below the 137,986 B ceiling: **24,980 B**
- Token agreement: **98.795970%**

This covers the full 42,382-byte demand. If distortion were unchanged, the arithmetic score would be `0.1033663`, but this is not a score result. The receiver currently stops at reconstructed tokens, and class-1 IoU is only `0.146`; no full-RGB render, scorer, Modal dispatch, or evaluator ran.

Artifacts:

- [Charter memo](/Users/adpena/Projects/pact/.omx/research/ddm_rc1_rate_crush_20260822.md)
- [Representation and receiver](/Users/adpena/Projects/pact/src/tac/optimization/rc1_terminal_program_vq.py)
- [Resumable materializer](/Users/adpena/Projects/pact/experiments/ddm_rc1_rate_crush.py)
- [Canonical retained result](/Volumes/APDataStore/pact/ddm_rc1_rate_crush/measurement_v4/RESULT.json)
- [MAIN fire-order](/Volumes/APDataStore/pact/ddm_rc1_rate_crush/measurement_v4/SEALED_FIRE_ORDER.json), SHA-256 `0d683cd3…eae56`

Verification covered 11 candidates and 451 retained files, every payload/shadow parse-back, all five paid-section corruption controls, deterministic repeats, and every 15×12 coder combination for both 8-bit and 16-bit assignments. Ruff, compilation, diff checks, gitleaks, two review passes, and serializer post-commit hashes passed. Developer preflight remained 18/25; bounded checks found no RC1 involvement in the seven existing failures.

Own-vehicle frontier: **DX2 S 0.14821987563243377 @ 180,368 B `[contest-CUDA T4 n600]`**, unchanged.

## NEXT_IF_RESUMED

- **QUEUED-WITH-FIRE-ORDER, BLOCKED** — owner: MAIN exact-row and unique n600 scorer-lane owner; consumer store: `/Volumes/APDataStore/pact/ddm_rc1_rate_crush/measurement_v4/main_fire/`; fire trigger: after the live joint solve is harvested and MAIN owns an idle lane, integrate payload `eab66bad…e61164` into a fresh full-RGB DX2 receiver, retain exact repeats and all-paid-section mutation controls, then evaluate that exact archive and fold it if recomputed S does not improve the pointer or exceeds 137,986 B.

## LIVE-HYPOTHESES

- A class-balanced or boundary-debt dictionary could preserve rare classes much better at the same K because the current population-weighted objective favors majority classes.
- A multiscale spatial grammar could free codebook bytes: at K=2,048 the assignment map costs only 10,900 B, while the codebook costs 48,920 B.
- The evaluator may tolerate token changes that remain inside the same scorer cells, but only full-RGB receiver integration and exact scoring can determine this.
- RC1’s temporal routing may serve as cheap counted conditioning for NR1’s evaluator-cell quotient.
- The charter’s placement-law prediction remains untested because no smaller DX2 stream exposes a homologous categorical temporal lattice.

## DEAD-ENDS

- Another fixed-RC64 coder/context race cannot supply the required cut; its measured 88 B ceiling is already shipped.
- PR130 memoryless bounds are not transferable DX2 entropy floors.
- MZ2 semantic recoding, post-hoc carrier refit, WD4, and FS2/FS3 remain closed on their measured scopes.
- DC1S sparse-grid Family A remains closed at 388,326 B, 274,549 B worse than the token member.
- Literal C1 planes and PP1/SP1/WS1 shipping streams exceed the byte corridor before realization.
- K=4,096 in this formulation is byte-dead at 105,811 B payload and 158,933 B shadow.
- Overall token agreement cannot be promoted as evaluator evidence; K=2,048’s rare-class collapse makes the exact receiver/scorer leg mandatory.

