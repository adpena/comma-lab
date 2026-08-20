Implemented and landed HB2.

Measured:
- Reproduced tq1c stage-3 failure: rc 1, `self-compressed round trip changed logits by 0.25`.
- Diagnosed mechanism: one `conv_a` channel at 8 bits emitted `-128`, while the deploy consumer clamps to `weight_bound=127`, restoring it as `-127`.
- Fixed repro repo in commit `e34f31bc4969042c0051ac81aa3c56884419a231`.
- Reran tq1c stage 3: rc 0, `hpac.bin.xz` = 14,116 B, `verified_exact=true`.
- Reran tq1c stage 4 encode/decode: rc 0/0, `tokens.bin` = 97,928 B, decode `verified_exact=true`.

Artifacts:
- [HB2_FINDINGS.md](/Users/adpena/Projects/pact/.omx/research/ddm_hb2_20260808/HB2_FINDINGS.md)
- [hb2_receipts.jsonl](/Users/adpena/Projects/pact/.omx/research/ddm_hb2_20260808/hb2_receipts.jsonl)
- [patch mirror](/Users/adpena/Projects/pact/.omx/research/ddm_hb2_20260808/0001-Fix-HPAC-self-compress-deploy-bounds.patch)
- Pact artifact commit: `896d0df5fd`

Boundaries held: CPU-only, no Metal/MPS/CUDA/scorer slot, no `checkpoints/gt/`, no driver-script edit, no upstream edit, no driver kill/signal. The gt arm inherits the fix at its future stage-3/stage-4 subprocess boundary.

Own-vehicle frontier unchanged: S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory].

