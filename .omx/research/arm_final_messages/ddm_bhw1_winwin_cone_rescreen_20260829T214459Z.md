Implemented and committed as `3a9c133b4d`.

Measured on `[macOS-CPU frozen-scorer advisory]`:

- Full LD1 cone: 5,282 edits → 176,913 B, **−3,476 B** versus LD1 and **−3,302 B** versus gb1.
- LD1-specific cone: only 14 new edits → 180,390 B, **+1 B**. This falsifies the cross-family law at its stated 0.1% threshold.
- The full saving is therefore a live composite rate opening, mostly inherited from DX2—not independent LD1 confirmation.
- `d_seg` and `d_pose` remain explicitly unmeasured; no score or promotion claim was made.
- DG2/JF2, OE1, and AE1 remain reduced-scope unknowns because matching final coding-argmax fields were not retained.

Artifacts: [verdict memo](/Users/adpena/Projects/pact/.omx/research/ddm_bhw1_winwin_cone_rescreen_20260829.md), [driver](/Users/adpena/Projects/pact/experiments/ddm_bhw1_winwin_cone_rescreen.py), [machine result](/Volumes/APDataStore/pact/ddm_bhw1_winwin_cone_rescreen/REAL_REENCODE_RESULT.json).

Validation passed: byte-identical n600 control, two real n600 re-encodes, 231-artifact manifest rehash, deterministic summary repeat, two Python review passes, Ruff, compilation, and 4 classifier tests.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_bhw1_winwin_cone_rescreen/scorer_n600/`; fire trigger: fcd3 terminates and MAIN claims the released scorer lane. Score the exact LD1 base and 176,913 B candidate.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN or bhw2 successor; consumer store: `/Volumes/APDataStore/pact/ddm_bhw1_winwin_cone_rescreen/jf2_k060_rescreen/`; fire trigger: the scorer row is terminal and AP has at least 8 GiB free. Regenerate JF2 k060’s matching argmax before screening it.
- **FOLDED** — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_bhw1_winwin_cone_rescreen/oe1_rescreen/`; fire trigger: JF2 independently shows at least 0.1% B with negative real marginal bytes.

## LIVE-HYPOTHESES

- The 176,913 B composite may survive joint scoring because its rate credit is real and its edits move tokens to GT; fcd2’s pose refusal makes this uncertain.
- A JF2 refit model may create a genuinely new cone because it changes the field×model pair, unlike fixed-model LD1.

## DEAD-ENDS

- LD1 independently confirming the law: closed—14/168,159 new B cells and +1 B.
- Treating all 5,282 cells as LD1 evidence: closed—5,268 were inherited from DX2.
- Entropy/additive pricing, substituting DX2 argmax for refit models, or inferring scorer outcomes from token labels: closed.
- AE1’s pre-corrector FS2 argmax is not a valid final coding surface.

Own-vehicle frontier: gb1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4, n600], UNMOVED.