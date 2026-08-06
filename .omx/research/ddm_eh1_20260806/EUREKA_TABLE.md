# EUREKA table

Scope: PR130/PR86 offline harvest, real local archive bytes and source, no new scorer work.
`|Delta S|` values below are projections unless the row says "measured bytes".

## Top Rows

| rank | class | mechanism PR130 has that we do not yet have in this vehicle | projected `|Delta S|` | falsifier | cheapest first measurement | consumer |
|---:|---|---|---:|---|---|---|
| 1 | EUREKA | Train a compact semantic-token to RGB receiver through the exact SegNet/R/uint8 path, then QAT it. This replaces flat/template paint with a learned source-forward renderer. | Up to `0.80084` vs fp1 flat-paint receiver floor: `100*(0.008305 - 0.00029660)`. TY2's local projection says PR130-class d_seg/pose with a small renderer reaches `S=0.157385863 @ 168,892 B`, still 11,092 B above sub-0.15 at same distortions. | A same-token, source-forward renderer fails to beat flat/template paint after parse-back, or renderer bytes erase the d_seg gain. | `$0-$slot-light`: train width-96/4-block renderer on TK1/tq1c labels with exact-R loss on stratified n32, then n120; no n600 claim until scorer slot is assigned. | TK2/TY2 semantic receiver, TR1 successor |
| 2 | EUREKA | Standalone neutral-gray 12-D pose carrier with 24x32 basis, 600x12 int12 coefs, hard-mined PoseNet objective, and exact int12 search. It is trained with the vehicle, not a post-hoc warp fantasy. | `0.069379` on the current own-vehicle pose term if it reaches PR130 d_pose: `sqrt(10*0.000716509) - sqrt(10*0.00002331)`. On pfs1 warp-base this exact class is not proven; pfs1 says post-hoc warp tails remain dead. | Carrier reaches only pfs1-like tail floor, or the int12/gray basis cannot hit `<5e-5` d_pose on our conditioned frames at <=25 KB. | Fit the PR130 carrier family on the current best frame_1 surface or on EH1 row-1's learned semantic renderer; PoseNet-only n32/n120, exact int12 lattice, parse-back carrier bytes. | pose leg, terminal-compose owner |
| 3 | EUREKA | Integer HPAC on semantic labels: masked convs, previous-frame context, patch-64 grouping, exact sparse range decode, self-compressed bit-depths. This is a partition-token coder, not an IX2 latent-stream prior. | Current rate gap to PR130 archive size is `0.111055`; PR130 token stream alone is 116,980 B = `0.077892 S` rate. Semantic+carrier+HPAC+tokens raw sections price at `0.133481 S`. | Counted model + range stream fails to beat the live KT/Brotli/SMEVR streams on exact decode equality, or decode time violates the contest budget. | Byte-only: feed TK1 exact semantic labels or current sub_final token planes to HPAC; require exact token decode equality and counted model bytes before any scorer use. | token coder stack, #869/TK1 |
| 4 | INCREMENT | CPR1 compact carrier repack: canonical Huffman for basis codes plus Rice-coded coefficient deltas, lossless round-trip and archive hash gates. | Measured byte delta from landslide to CPR1: 3,328 B = `0.002216 S` rate. | Compact representation cannot round-trip the carrier exactly or model XZ size does not fall. | Repack any low-rank pose/semantic carrier we own with the same decode-equality gate. | receiver packer / archive grammar |
| 5 | EUREKA-INFRA | 49-stage provenance graph with selected boundaries, exact input/output hashes, skip/resume markers, and final official metric parse. | No direct `Delta S`; reduces fake-row risk and makes full pipelines reproducible. | A stage can be skipped despite changed command/runtime/input/output, or final archive cannot be reproduced byte-for-byte. | Port only the marker/fingerprint discipline to the next local training run; do not port all machinery. | governed launcher / serializer discipline |
| 6 | INCREMENT | Bit-depth self-compression in the HPAC model: per-channel bit depths with model-rate term in the training objective. | Source receipt: raw model 20,179 B, compressed model 15,164 B, zero logit diff. Rate saving inside PR130 model = 5,015 B = `0.003340 S`. | Self-compression changes logits or the bit-depth metadata overhead exceeds savings. | Run on any small counted model section first; require max logit diff 0. | model-section quantization owners |

## Synthesis

The top three are not independent knobs to sum blindly. Row 1 creates the
semantic frame family whose token distribution row 3 compresses, and row 2 is
only PR130-class when trained against a pose-legible vehicle. The concrete
EUREKA is the coupled object:

`semantic labels -> trained RGB receiver -> neutral-gray pose carrier -> HPAC token coder -> CPR1 repack`.

That is structurally different from our current flat-paint, template-paint,
warp-base, IX2 latent-prior, and global-L token-drop negatives.
