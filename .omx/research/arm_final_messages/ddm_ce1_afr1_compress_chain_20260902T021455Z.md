Implemented, verified, documented, and committed as `2c3a2153e4599f4a170cf7f69c80f4246677ae53`.

- Two complete chain invocations produced exactly 180,002 B, SHA `cbb8d928…`; retained finals are byte-identical.
- All six intermediate identities matched their pins, including the GB1 pointer/JT21 fork.
- The AFR1 bridge encoded cold twice; earlier stages exercised checkpointed resume.
- No scorer, Modal dispatch, packet mutation, publication, or frontier update occurred.
- Tests: 3 passed; Ruff and `py_compile` passed; two review-tracker passes completed.
- Global developer preflight had one unrelated existing failure in `ddm_dds1_decoder_derivable_born_stats.py`; the commit-scoped hooks passed.
- Sealed packet text was not edited; exact replacement text and proposed cuts are in the receipt.

Artifacts: [receipt](/Users/adpena/Projects/pact/.omx/research/ddm_ce1_afr1_compress_chain_20260901.md), [runner](/Users/adpena/Projects/pact/experiments/ddm_pq2_compress_e2e.py), [tests](/Users/adpena/Projects/pact/experiments/tests/test_ddm_pq2_compress_e2e.py), [machine result](/Volumes/APDataStore/pact/ddm_ce1_afr1_compress_chain/RESULT_pq2_e2e.json).

Own-vehicle frontier remains **S 0.14797617125559104 @ 180,002 B `[contest-CUDA T4 n600]`**.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN; consumer store: `.omx/research/ddm_pq1_submission_packet_prep_20260815/`; fire trigger: commit `2c3a2153e4` is present on MAIN and packet checks remain green; apply the proposed compression answer and retire stale refusal language.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN/public-source owner; consumer store: a new clean-room CE1 reproduction receipt in the packet directory; fire trigger: all 15 inputs are available at immutable sanitized pins; run from an empty work store.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: operator/MAIN; consumer store: task rows 1111 and 1363; fire trigger: operator clears both submission-hold and contest-policy gates.

## LIVE-HYPOTHESES

- A completely empty-store replay will reproduce `cbb8d928…` without checkpoint reuse. This is plausible because both cold AFR1 encodes matched exactly and every predecessor checkpoint completed against the same pinned mechanisms, but it remains untested as one wholly cold invocation.
- Packet prose can safely use abbreviated hashes while machine authorities retain full hashes. The CE1 result and retention manifest now provide centralized full-SHA custody, reducing duplicated drift surfaces.

## DEAD-ENDS

- Standalone `cls_groupbin8` is not JT21: it produced 180,268 B / `bd1c663f…`, first differing offset 14.
- The post-DX2 lineage is not linear: GB1’s 180,215-byte pointer and the 180,192-byte JT21 bank are separate branch outputs.
- An opaque pre-staged AFR1 runtime is not an acceptable stage mechanism; the admitted implementation constructs the encoding runtime from reproduced LB1 bytes plus the identity-tool-pinned corrector.
- Blind resume over FX5’s post-stage repin fails; resume must restore the rc2 pointer body before continuing encoder state.
- JT22 and JT23 are not hidden chain stages: JT22’s −1 B was refused at its bar, and JT23 collected 0 B.