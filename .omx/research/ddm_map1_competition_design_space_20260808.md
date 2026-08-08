# THE COMPETITION DESIGN-SPACE MAP — every public PR as an evaluated design point (#984 arithmetic sheet, FIRST SECTION)

Tags: [no-triality] [p0-ledger-ok]. Owner MAIN, fired at the VEH endpoint boundary per the
#984 metadata binding (the upstream-authority / zero-gravitational-pull law: vehicle selection
answers to the COMPETITION MAP + upstream, never to our incumbent's gravity).
Sources (recall-first, receipts cited, nothing re-measured): pi1
`pr86_pr130_fullstack_intake_20260728.md` · eh1 `ddm_eh1_20260806/EUREKA_TABLE.md` ·
`public_pr129_132_intake_20260725.md` · `pr112_127_intake_20260710.md` +
`pr128_intake_reverse_engineering_20260710.md` · CLAUDE.md L14–L32 (PR95-family anatomy) ·
`ddm_upstream_factor_flattening_20260808.md` · VEH_CAP_N32_VERDICT.md (e9939b1669).
Every external score keeps its axis label; none is our row.

## 1. The design points

Design-point tuple: {token family · coder family · renderer class · pose mechanism · byte
split} → {seg S, pose S, rate S, total S}. S components in contest units.

| PR | token family | coder | renderer | pose mechanism | bytes | S (axis) |
|---|---|---|---|---|---:|---|
| 95 (root) | RGB-INR latent (28-d/pair) | brotli | HNeRV PixelShuffle+sin decoder (weights=payload) | implicit via RGB | ~300K | 0.193 era-best |
| 100 | same | brotli 4-section 0.bin | hnerv_lc_v2 (268 LOC) | implicit + warp | — | 0.195 |
| 101 (gold) | same + per-pair 1-dim correction sidecar | byte-maps + perms + split-brotli + raw-LZMA1 + colex-rank | PR100 substrate | implicit | ~300K | 0.193 |
| 102 (bronze) | same | same-family | same | implicit | — | 0.19538 [contest-CPU] / 0.22839 [contest-CUDA] |
| 103 (silver) | same | + constriction range-AC on 8 tensors | same | implicit | — | 0.195 |
| 110 | same + FEC6 frame-exploit selector | same-family | same | selector modes | — | **0.191083 [contest-CPU]** (= our borrowed pointer base) |
| 112 | same (recode of PR110 payload) | ctx range coder | same | same | — | 0.191126 |
| 118 | — (ANALYSIS) | — | — | — | — | 0.20 repro; **the saturation study: 62% of score = rate, 93% of archive = decoder weights, 4 quantified dead-ends, "only escape = from-scratch arch vs frozen scorer"** |
| 121 | same | +M_GRID/Markov nibble (−56 B) | same | same | — | 0.191088 |
| 123 | same | LZMA + greedy-brotli order | 9-stage Muon retrain (HURT) | 31-mode selector | — | 0.25 |
| 125 | same + full 600×28 QAT gradient polish | same | frozen decoder | same | — | 0.190946 |
| 126/127 | same + exact-grid QAT polish, **boundary seg loss σ(−margin/τ)** | same | frozen | same | — | 0.190506 (fp16-select regresses CPU: 0.191955) |
| 128 | same + **discrete exact-gated click-polish** | same | frozen | same | — | 0.187946 [claim] — the RGB-cluster BEST |
| 129 | same + exactgrid latent polish | PR112 ctx coder | frozen PR95/101 | 35-mode frame_0 pixel selector (+48 B) | 176,337 | **0.190502 [contest-CPU]** |
| 131 | Cool-Chic INR leg | — | — | — | 176,531 | 0.188533 [external self-report, unverified] |
| **86** | **SEMANTIC 5-class token grid 384×512 (=SegNet's own grid)** | **HPAC learned integer AR prior, 0.00773 bpp** | TokenRendererV62 (token→RGB) | slave ShrinkSingleNeRV frame_0 (32,287 B → d_pose 4.57e-4) | 207,579 | **0.273636 [contest-CUDA]** |
| **130 (RANK 1)** | **semantic tokens 116,980 B @ 0.00793 bpp** | **HPAC masked-conv + prev-frame ctx (conv_past) + FiLM, range-coded; int4/int8 models xz 50,914 B; self-compressed bit-depths** | **int4 w96 4-dilated-block token→RGB, 40,252 B** | **CPR1: frame_0 neutral-gray 12-D actuator, 24×32 basis, int12 coefs, 23,054 B → d_pose 2.33e-5** | 191,052 | **0.172141 [contest-CUDA]** = seg 0.02966 + pose 0.01527 + rate 0.12721 |
| 132 | PR130 + 800-step renderer CE fine-tune | identical code | same | same | 191,028 claim | unverified |

## 2. The map's structure — two clusters and a dead zone

1. **RGB-INR cluster (PR95→129):** carries RGB; the archive is mostly DECODER WEIGHTS (93%
   per PR118's own writeup); coders act on weights/latents; pose is implicit. Fifteen design
   points over 3 months converge to a **0.1879–0.1911 plateau** — the cluster's own best
   analyst (PR118) declared it saturated, and its best point (PR128) is a terminal
   discrete-polish of a frozen decoder. **SATURATED, measured by the competition itself.**
2. **SEMANTIC-CARRIAGE cluster (PR86→130→132):** carries the PARTITION ITSELF as tokens +
   a learned prior that prices them + a small trained renderer + a dedicated frame_0 pose
   actuator (exploiting the frame-role asymmetry: SegNet reads last-frame only). TWO evaluated
   points: 0.2736 → **0.1721**. The −0.1015 jump decomposes (pi1 §4.3, measured): pose-leg
   swap ≈ **0.0523** (half the gain) + renderer/model −9,972 B + determinism engineering; the
   token stream itself barely changed (+3,080 B, bpp slightly WORSE). **NOT saturated — the
   only cluster whose frontier moved, and its own token/rate term is still coarsely engineered.**
3. **Dead zone:** classical codecs (AV1/x265, PRs 113–122 various) 1.2–4.0; Cool-Chic leg
   unscored. Off-frontier.

## 3. The vehicle-argmax check (the postmortem cure, executed)

The #984 composed vehicle = semantic carriage (HPAC learned prior + AC) + trained receiver +
joint/dedicated pose leg. **It sits in cluster 2 at the map's argmax — confirmed, not by our
history but by the map.** The VEH/CAP verdict pair (e9939b1669) independently closed the one
routing alternative inside the cluster (decode-side correction of degraded tokens: VEH flat at
input's own error; carry better tokens instead).

**The map's open axes — what NO evaluated design point exploits, with our measured receipts:**

| axis | PR130's position | our banked receipt | consumer |
|---|---|---|---|
| task-lossy TOLERANCE | carries tokens near-exact; scorer forgives cheap flips | fl1 per-class GT-flicker floors · ms2r 7.6× error headroom at box · dr2b rungs | label-stream rate (lx1 term 2) |
| per-EDGE conditional structure | HPAC context is spatial+prev-frame, class-agnostic | m91 one-graph/Road-hub (87.8% of flips) · cr1 edge-conditional −19.2% [byte-only] | token coder (tr2p1 AC-amended race live) |
| ξ-keyed temporal alignment | conv_past = RAW prev-frame context, unaligned | QA39 carried-ξ INTER (rule-118 free warp) · tac.lie SE(3) · am1 | HPAC context expert |
| blind/null/gauge DOF | pays for provably-invisible pixels | #839 four canonical names (22.70% dual-blind · 80.67% resize-nullity · ~52% range(A)-complement) | receiver targets + rate credit |
| exact scorer factorization | empirical frame_0-is-seg-free only | the THEOREM (pz1 shared-D; pi1 E3 "they found the corollary; we have the theorem") | pose leg placement (Q3 law #889) |

**LOCATE-ON-MAP GATE (standing, from the postmortem cures):** any proposed new frontier/vehicle
must place itself as a design point on THIS map — {cluster, tuple, which open axis it exploits,
which evaluated point it must beat} — before absorbing fleet slots. A vehicle that cannot name
its cluster is orbiting an incumbent, not competing.

## 4. Composed arithmetic sheet — every term at its best MEASURED value (skeleton, per-term)

Bar = PR130 0.1721417 (na1: 191,052 B reproduces the published floor). Axis labels binding;
nothing below is a score claim.

| term | PR130 (the bar) | OUR best MEASURED | gap mechanism | improving rows (fire state) |
|---|---|---|---|---|
| label/token stream + prior | tokens 116,980 B + models ~50,914 B (0.00793 bpp) | **hb2: HPAC on OUR tq1c labels = 112,044 B JOINT** (model 14,116 + stream; 2.1% over model ideal) [byte-only] | ours measured on tq1c labels; gt-label price pending | gt-HPAC ep60 (~1-2d, driver 9316) · tr2p1 TROT-vs-CR1 AC race (LIVE) · tolerance rungs UNPRICED on this stream |
| renderer (token→RGB) | int4 w96, 40,252 B, converged n600 → d_seg 2.966e-4 | CAP gt→gt K=8 tail **0.0010862** (n32, 6k-step lr-2e-7 probe) [macOS-CPU advisory] | TRAINING/CONFIG DEBT, not architecture (same class; VEH/CAP verdict) | **n120 reference-form build = next Metal fire** (PR130 form off-the-shelf + flip-targeted losses + m88/m96 stratified sampling) |
| pose leg | CPR1 23,054 B → d_pose 2.33e-5 → 0.0153 S | banked R1 dxi 7.2 KB → 0.127 S (shipped-compact); exact-regime 9.3e-10 (rate-dead) | their carrier is trained WITH the vehicle; ours was post-hoc | eh1 row-2: fit CPR1 family on OUR conditioned frames (int12 lattice, PoseNet-only n32/n120) — queued; jd joint-descent line = the in-loop alternative |
| tolerance (task-lossy) | not exploited | fl1 floors + ms2r box headroom 7.6× + m66 gap equation | never composed into a carriage price | lx1 term-2 rows (arm LIVE) |
| temporal/ξ | conv_past raw | QA39 raced as SMEVR expert; ξ-curve carried | alignment unexploited publicly | HPAC ctx expert race after gt-HPAC lands |
| blind/null | not exploited | #580 projector + 22.70% mask [measured] | free-fill + rate credit unpriced on THIS vehicle | lx1 term-6 rows |
| composed projection | 0.172141 | **TY2 projection: S=0.157385863 @ 168,892 B** at PR130-class distortions w/ small renderer (eh1) — PROJECTION, not a row | 11,092 B above sub-0.15 at those distortions | becomes real only via byte-closed n600 evaluate.py |

**Reading:** the sheet says the composed vehicle beats the bar IF (a) the n120+ receiver closes
the renderer training debt (existence proof: PR130; capacity proof: CAP), (b) the gt-HPAC price
lands at ≈ hb2's 112 KB scale, (c) ONE of {tolerance, per-edge, ξ-context} rungs pays on the
token stream, and (d) a pose leg reaches CPR1-class ≤25 KB (eh1 row-2 or jd-line). Every letter
has a live or queued owner; none requires new invention — the deltas are banked, the base was
the miss.

## 5. Follow-on disposition (no orphans)

| item | state |
|---|---|
| n120 reference-form receiver config + fire | NEXT (owner MAIN, Metal free) |
| gt-HPAC ep60 price | LIVE (driver 9316) |
| tr2p1 TROT race / lx1 crosswalk | LIVE codex arms (#990/#991) |
| eh1 row-2 pose-carrier family fit | QUEUED-W/-FIRE-ORDER (after n120 receiver, PoseNet-only n32 first) |
| rr19 review round | QUEUED (facets_v2 + veh facets + this window; counter 0/3) |
| MEMORY.md consolidation (19,610 B > budget) | OWED at next quiet boundary |
