Completed the charter’s research and implementation. The full handoff is in the [RFO2 memo](/Users/adpena/Projects/pact/.omx/research/ddm_rfo2_fresh_eyes_gestalt_synergy_20260815.md).

Key outcome: strict sub-0.15 at unchanged distortion requires saving **15,157 B**, not 15,153 B. No measured combination currently supplies that gap. The best route is mixed precision → carrier rank/refit → nested-width distillation → token drop/coder refit.

Implemented:

- Future joint-byte and top1 regression protection.
- SSD-compatible watcher alert publication.
- Distortion-aware retained-checkpoint selection.
- Endpoint integration with exact log/checkpoint hash verification and legacy-receipt compatibility.

Verification: Ruff passed, **45 focused tests passed**, and two review-tracker passes completed. No scorer, Modal job, training launch, live-process mutation, or new score occurred.

Landing remains blocked: the required serializer failed because the sandbox cannot write Git objects (`Operation not permitted`). The shared index remains empty, HEAD remains `dec54025777c70c1279e5c0849ef73f06c3bec4f`, and all ten deliverable files are intact but uncommitted.

The bounded public check found no post-August-10 score/archive signal in [PR135](https://github.com/commaai/comma_video_compression_challenge/pull/135) or [PR130](https://github.com/commaai/comma_video_compression_challenge/pull/130); private activity remains unknown.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — Run the selector at e960 completion; owner: MAIN #1058 adjudicator; consumer store: e960 endpoint-closure and RX2 identity-race stores; fire trigger: real rc=0 burn receipt and legacy closer completion.
- **QUEUED-WITH-A-FIRE-ORDER** — Receiver-close the retained 182,679 B q3/q4 candidate; owner: MZ2 successor; consumer store: `/Volumes/VertigoDataTier/pact/ddm_mz2_frozen_section_representation/`; fire trigger: retained SHA reverified.
- **QUEUED-WITH-A-FIRE-ORDER** — Materialize retained carrier r10/r8/r6/r4 refits; owner: carrier-structure successor; consumer store: `ddm_mz2_frozen_section_representation/carrier_rank/`; fire trigger: q3/q4 byte/receiver stage becomes terminal.
- **QUEUED-WITH-A-FIRE-ORDER** — Launch nested-width distillation; owner: width-distillation trainer; consumer store: `ddm_mz2_frozen_section_representation/distill/`; fire trigger: e960 releases the trainer slot and storage/governor preflight passes.
- **QUEUED-WITH-A-FIRE-ORDER** — Land this arm through the serializer; owner: operator or Git-enabled MAIN; consumer store: repository history; fire trigger: a session with Git-object write permission.

## LIVE-HYPOTHESES

- Carrier atom/rank reduction plus refitting can save several kilobytes because 22,032 B currently resides in basis and coefficient streams.
- MZ2 q3/q4 may provide a real first rung because it already saves 823 complete-archive bytes, though distortion remains unmeasured.
- Nested-width training can outperform exact recoding because it removes learned channels instead of losslessly representing full-rank tensors.
- Token drop may pay only after representation changes alter the token distribution and HPAC model jointly.
- A sufficiently large carrier-rate reduction can tolerate modest Pose regression because the rate benefit is nonlinear-score competitive.

## DEAD-ENDS

- Same-state lossless recoding is closed on e480b: MZ1’s 8/8 race saved 0 B.
- The supposed 52,566 B serialization gap was a model-versus-wrapper accounting error.
- Naive exact semantic deletion and pointwise low-rank recoding are closed on the tested state.
- Absolute carrier-code coarsening is closed on PZ4A: 500 B gross became 2,232 B net growth.
- Packaging cannot supply the gap; header, residual, and ZIP framing total only 210 B.
- Treating the live token estimate as an 18 KB archive win is closed; through epoch 600, the best advisory joint estimate remains 130,875 B.
- This arm did not move the pointer. Own-vehicle frontier remains **S=0.1600920261571558 @ 183,502 B `[contest-CUDA T4, n600]`**, archive SHA-256 `e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3`.