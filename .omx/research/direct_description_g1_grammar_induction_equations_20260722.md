---
schema: ddm_g1_grammar_induction_equations.v1
date_utc: 2026-07-22
status: CANDIDATE_NOT_REGISTERED
research_only: true
score_claim: false
main_landing_review_required: true
---

# Candidate canonical equations — per-stratum grammar MDL

Let `s` be a semantic stratum, `G_s` a generic decoder grammar, `P(G_s)` its fired production
streams, and `D_s` the video-derived derivation. Under rule 118, generic interpreter code is free
but every derivation and framing byte is counted:

```text
L_MDL(s; G_s, D_s)
  = L_generic(G_s) + L_counted(D_s | G_s)
  = 0 + 5 + sum_{p in P(G_s)} [10 + min_c bytes(c(serialize(D_{s,p})))] .
```

The measured coder set is `c in {Brotli-q11, raw-LZMA1-preset1-dict1MiB, zlib9}`. The five-byte
envelope and ten-byte per-production frames are part of `L_counted`. This equation does not make
video-derived constants free merely because a decoder can parse them.

For exact semantic reconstruction `R_G(D_s) = M_s`, the measured candidate-set optimum is

```text
G_s^lossless = arg min_{G in C_s : R_G(D_s)=M_s} L_MDL(s;G,D_s).
```

For tolerance `tau`, the measured rate-distortion row is

```text
G_s(tau) = arg min_{G in C_s : d_mask(R_G(D_s),M_s)<=tau} L_MDL(s;G,D_s),
d_mask(A,B) = |A xor B| / (600*384*512).
```

This is a mask-corpus law, not evaluator `d_seg`: the latter requires receiver-visible RGB,
realization through `R`, and the frozen scorer. Therefore every reported `d_mask` is named
`dseg_oracle_clean_rest` only as an explicit projection field, never as a score.

For independently decoded Lane and Movable masks, with all other strata assumed exact, the safe
composition used here is the union bound

```text
d_union <= (e_Lane + e_Movable) / N,
N = 600*384*512.
```

The bound row is

```text
B = 27,692 + 29,810 = 57,502 bytes,
e = 583,417 + 33,378 = 616,795,
d_union <= 0.005228635999891493.
```

Thus `B <= 60,000` is true and `d_union <= 0.005` is false. Exact blocker size is
`616,795 - round(0.005*N) = 26,971` mask errors. Relative to the separate 0.00116 box, the bound is
4.5074448275x. Cross-stratum overwrites can make the sum loose; they cannot be assumed beneficial
without an ordered receiver composition.

## Candidate extension law

Any new production `q` should be evaluated by actual incremental complete-stream bytes and
semantic reconstruction before admission:

```text
rho_q = [d_mask(before)-d_mask(after)] / [B(after)-B(before)].
```

This `rho_q` is only a mask-corpus acquisition statistic. Downstream receiver work must replace it
with the registered Fisher/margin metric, corrected inner Jacobian, exact nonlinear Pose term, and
final-ZIP rate; any residual basis must be curvelet/shearlet, not Fourier. Admission still stops at
the registered score-rate break-even `25/37,545,489` after those authority surfaces exist.

## Status and scope

`CANDIDATE_NOT_REGISTERED`: the equation is bound to compact receipt
`.omx/research/direct_description_g1_grammar_induction_20260722.json` and primary receipt SHA-256
`aeeb916f973523d5ffa3389ee8d744901fe9477cc149af7e756726e2ead907f6`. It is not appended to the
canonical equation registry because it has no receiver-visible RGB, Pose, final-ZIP, or contest-axis
anchor. MAIN review is required before registration or consumption.

STORES CONSULTED: paired landing memo and DAG FEED; #596 and #610 representation/grammar anchors;
v12 residual decomposition; real-coder survey; frozen cache and emitted semantic parse-backs;
`docs/operating_manual_craft_handoff.md`; canonical pointer/lane/task/subagent state; delegation
inboxes. Pointer `0.1910828242 [contest-CPU]` unchanged.
