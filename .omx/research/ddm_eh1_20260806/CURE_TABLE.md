# Cure table

Required surfaces classified: 7.

Counts:
- Strong cure candidates: 2 (`fp1`, `tk2`)
- Partial / split cure candidates: 2 (`TR1/tb1`, pose carriers)
- Scope reopen / rescope: 1 (`hp1`)
- Byte/scorer-gated rate cure candidates: 2 (`IX2TOK01`, `#869`)

| surface | prior negative / blocker | PR130 mechanism that bears on it | cure verdict | exact next race |
|---|---|---|---|---|
| fp1 flat-paint receiver floor `0.008305` | Perfect GT argmax painted with solved prototype colors still gives d_seg `0.008305`; receiver, not tokens, is the wall. | PR130 trains a semantic-token RGB renderer through exact SegNet/R, QAT, and exact checkpoint selection; official d_seg `0.00029660`. | STRONG CURE CANDIDATE. It refutes "class field cannot be compressed" for trained receivers, not by flat paint. Own vehicle still unmeasured. | Train row-1 semantic receiver on TK1/tq1c labels, n32/n120 first, then n600 when scorer slot is assigned. |
| tk2 C1/C2 template paint | n4 smoke: c0 flat paint `0.008454641`, c1 template `0.010585785`, c2 boundary AA `0.010674795`; C3 TR1 retarget blocked. | PR130's receiver is trained and source-forward, not a static palette/template. | STRONG CURE CANDIDATE. The tk2 negative is toy/template-scoped, not a semantic-renderer family negative. | Fire the TK2 D1 trained receiver discriminator; compare template paint vs trained PR130-style receiver under same token source. |
| TR1/tb1 renderer architecture + burn schedule | TR1 has a real token-grid renderer and event schedule, but current rows are advisory and far from PR130. | PR130 uses width-96 semantic renderer, staged CE/margin/exact/QAT schedule, selected boundaries, and exact source-forward selection. | PARTIAL CURE INPUT. It gives concrete architecture/schedule priors, not a transplant verdict. | Race PR130 semantic renderer schedule against current TR1 schedule at n32/n120 with same tokens and exact-R loss; promote only with parse-back evidence. |
| warp-pose6 / sc1 e_p rank-1 / 194B warp-base pose carriers | pfs1 shows post-hoc warp/e_p tails remain huge on the seg-only warp base; wd1 says retired pose basis would regress. sc1 says cheap e_p field is mandatory but not sufficient. | PR130 uses a neutral-gray pose frame carrier, 12-D basis, int12 code search, and PoseNet objective, coupled to its vehicle. | SPLIT. It cures "pose must be huge" on a trained semantic-pose vehicle, but does not cure pfs1's unconditioned warp-base negative. | Fit PR130 gray carrier on a conditioned base; fail if d_pose does not reach PR130-class at <=25 KB. |
| hp1 learned-AR-prior FAMILY_NEGATIVE latent-stream-scoped | HP1 best learned prior on live `IX2TOK01` stream was 456,166 B, +114,870 B vs shipped, so negative for <=10K static-context learned priors on that latent stream. | PR130 HPAC is not a generic learned prior on IX2 latent codes; it is an integer causal model over semantic labels with sparse exact range decode and counted self-compressed weights. | RESCOPES / REOPENS. HP1 remains negative for IX2 latent stream; semantic-token HPAC is a different live family. | Byte-only HPAC on TK1/tq1c semantic labels; exact decode equality and model bytes counted before scorer. |
| IX2TOK01 LZ-match coding | IX2 found adaptive distribution coders lose to match/LZ structure on the IX2 latent member; bytes were mostly between members or in layout. | PR130 semantic token coder uses causal HPAC logits and range coding on 5-class semantic maps, with previous-frame and spatial contexts. | BYTE CURE CANDIDATE. It does not contradict IX2; it says change the represented stream first, then use HPAC. | Same token payload, same train/test split: compare KT/Brotli/SMEVR/LZ-match vs HPAC range bytes under exact decode equality. |
| #869 adaptive quant map / per-cell L | #869/TW1/TZ1 says static global L is degenerate; adaptive [16,12,8,4] has measured byte side `-113,555 B` but scorer leg is queued. | PR130 self-compressed HPAC bit-depths prove model-channel adaptive depth can be trained and decoded exactly; it is not the same map. | SCORER-GATED CURE CANDIDATE. It supports adaptive allocation as form, but #869 still needs own-vehicle joint d_seg remeasure. | Fire scorer-batch #869 row only after bytes are refreshed on current sub_final; pays iff `Delta d_seg < 7.561e-4`. |

No row in this table is a pointer move. The strongest actionable change is to
stop treating flat/template receiver negatives as evidence against trained
semantic receivers.
