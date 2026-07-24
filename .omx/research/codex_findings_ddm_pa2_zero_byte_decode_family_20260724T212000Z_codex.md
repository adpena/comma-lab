---
title: Codex findings - DDM PA2 zero-byte decode family
date_utc: 2026-07-24T21:20:00Z
lane_id: lane_ddm_pa2_zero_byte_decode_family_20260724
research_only: true
score_claim: false
pointer_moved: false
main_landing_review_required: true
---

# Outcome

The decoded-frame-only PA2 family produced one measured admission:

`IC2 W_seg+PA1 -> frame1 xi-hat` moves the advisory objective from
`28.00173925293584` to `25.244396496399435`, a conditional
`delta_S=-2.7573427565364064`, with the exact same 131,154-byte archive and
SHA-256
`be1989fe16a2983b291c85c5e58f3e2db74edd3123a5ed19add11c7e9800f97e`.

This is `[macOS-CPU frozen-scorer advisory]`, `research_only=true`, and
`score_claim=false`. The contest pointer remains
`0.1910828242 [contest-CPU]`.

# Three-base conditional table

| Base | Member | d_seg | d_pose | S | Conditional delta S | Verdict |
|---|---|---:|---:|---:|---:|---|
| IC1 | spatial stem | 0.07520784166124132 | 26.487295402510874 | 23.883317161086644 | +0.2215250248531042 | reject instance |
| IC1 | frame0 xi-hat | 0.07051923116048177 | 70.97412270530354 | 33.7805071274169 | +10.118714991183356 | reject instance |
| IC1 | frame1 xi-hat | 0.19362428453233507 | 71.84755990390745 | 46.25443865006635 | +22.592646513832808 | reject instance |
| IC2 | spatial stem | 0.024254336886935762 | 78.20705382857086 | 30.478287855584014 | +2.4765486026481724 | reject singleton |
| IC2 | frame0 xi-hat | 0.024124510023328993 | 145.14605295034937 | 40.597819461901096 | +12.596080208965255 | reject singleton |
| IC2 | frame1 xi-hat | 0.07160721672905816 | 32.38684246616016 | 25.244396496399435 | -2.7573427565364064 | admit |
| IC2 after frame1 xi-hat | spatial stem | 0.07597559611002604 | 33.33732282039997 | 25.94340079298668 | +0.6990042965872441 | reject conditional |
| IC2 after frame1 xi-hat | frame0 xi-hat | 0.07160721672905816 | 82.80372292025596 | 36.023687772761576 | +10.779291276362141 | reject conditional |
| MS2R | PA1 | 0.001159998575846354 | 6.64586490644108 | 202.1699400774639 | +7.744379794217423 | reject instance |
| MS2R | spatial stem | 0.0013282267252604167 | 0.0186827167672035 | 194.46678054385436 | +0.041220260607872206 | reject instance |
| MS2R | frame0 xi-hat | 0.001159998575846354 | 8.083889155380108 | 203.00876772083737 | +8.583207437590886 | reject instance |
| MS2R | frame1 xi-hat | 0.008386323716905382 | 6.901931707446569 | 203.04814169200725 | +8.622581408760766 | reject instance |

# Interpretation

The IC2 admission is not a visual-fidelity result. It is a typed trade: Seg
degrades by `+0.04748270670572917`, Pose improves by
`-32.648144663161176`, and the nonlinear joint objective improves. The same
algorithm fails badly on IC1 and MS2R. The result therefore supports
base-conditioned greedy measurement, not a universal temporal doctrine.

PA1 also does not transfer to MS2R under its present decoded-moment
formulation. It leaves Seg unchanged but moves `d_pose` from
`0.01663315390825408` to `6.64586490644108`. The correct verdict is
`INSTANCE REJECTED`, not “PA1 dead.”

# Rule-118 boundary

Free receiver code:

- exact resize/stem geometry;
- decoded-frame luma-gradient centroid;
- integer xi-hat displacement;
- deterministic half blend;
- blind-coordinate mask and generic zero fill;
- PA1 moments derived from decoded YUV6.

Counted if supplied:

- per-pair displacement or orbit tables;
- target labels or class maps;
- learned/video-derived gain, tone, gamma, or residual tables;
- rank-4 feature-to-RGB assignments;
- gauge coefficients or selected orbit positions.

No such counted statistic was hidden in code.

# Typed blockers and preserved family scope

- Gauge orbit: current measured energy is sample-specific and no generic legal
  RGB/uint8 receiver pullback exists.
- Rank-4 tone/gamma: the current artifact has feature prototypes/hashes but no
  frozen class assignment or RGB/uint8 pullback.
- #401: scorer-input identity is proved n600 on all bases, but a pure generator
  has no camera-resolution stored section, so its direct byte saving is zero.

# Receiver and archive custody

| Base | Exact archive bytes | Exact SHA-256 | Selected receiver-output SHA-256 |
|---|---:|---|---|
| IC1 | 131,582 | `aba831de718f7bc5fd264b9334a29ef8a9388052a629941e50eea1b854fe49d9` | `e73e013eb8124d1b3e4f39b04852d3aad3eee8216da89edf9903af664e749782` |
| IC2 | 131,154 | `be1989fe16a2983b291c85c5e58f3e2db74edd3123a5ed19add11c7e9800f97e` | `aa091498a48c718c584c27f8e1199f90e4c3405b1ea52a725cc7ff200f0d1048` |
| MS2R | 291,205,400 | `e3d0581ff4a3f475057e77e530374dad444b640a049b058cd66b37563534773e` | `32a773a23a79c036ca39352b9ca9a048e20c089dc45beaa4c847689083641558` |

Each selected output is 600 pairs, 3,662,409,600 bytes, split into 19
immutable batch-32 stages on `/Volumes/VertigoDataTier/pact`. No local bulk
was created and no destructive cleanup occurred.

# Durable integration

- typed DSL config;
- reusable decoded-frame transform module;
- resumable three-base measurement runner;
- NumPy 2.4 stored-NPY compatibility fix on the canonical zero-copy helper;
- canonical conditional equation
  `ddm_pa2_zero_byte_conditional_greedy_v1`;
- DAG FEED and aggregate receipt;
- regression tests for geometry, identity, blockers, immutable stages, and
  strict conditional admission.

# Stores consulted

The decision consumed the canonical frontier, lane registry, subagent
progress, gradient-anchor, dispatch, posterior, probe-outcome, council/design,
latest Codex findings/session-summary, scorer factorization, PA1, IC1, IC2,
MS2R, #401, #580, and #583 surfaces. PR98/L28 were quarantined as historical
signal only and were not imported.

# Remaining authority debt

MAIN must review and integrate the IC2 receiver composition. Only after that
may the exact archive be replayed on contest CPU and contest CUDA. No contest
score, promotion, or pointer movement is claimed here.
