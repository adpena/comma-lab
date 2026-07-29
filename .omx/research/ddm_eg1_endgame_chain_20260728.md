---
schema: ddm_eg1_endgame_chain.v1
date_utc: 2026-07-29
arm: ddm_eg1_endgame_chain
axis: "[macOS-CPU/MLX research-signal; stale bounded rehearsal only]"
research_only: true
score_claim: false
pointer_moved: false
paid_dispatch: false
n600_scorer_job: false
local_contest_cpu_anchor: 0.1910828242
competitive_effective_frontier: "official displayed 0.172"
main_landing_review_required: true
verdict: BUILD_REHEARSED_NO_CANDIDATE
---

# DDM EG1 — post-burn endgame chain

**Pointer honesty first:** the custody-specific local `0.1910828242
[contest-CPU]` row is **UNMOVED**. The newer official-best-aware competitive
effective frontier is the official displayed **0.172**, not 0.191. This arm
did not launch the burn, run an n600 scorer job, buy compute, create an exact
row, or promote a candidate. Every result below is research-only and requires
MAIN landing review.

**One-line verdict:** EG1 built and bounded-rehearsed the post-burn
byte-close/policy/terminal chain, but produced no candidate or exact score;
the competitive frontier and local pointer are unchanged.

## Scope and authority boundary

EG1 owns the post-burn completion chain only:

1. a deterministic TR1 four-section archive/receiver rehearsal on the stale
   TB1 T2 lottery checkpoint;
2. a typed, operating-point-dependent stop/continue/handoff policy; and
3. bounded terminal QDBS and six-equation pose finishers.

The long burn owns the n600 scorer slot. EG1 therefore uses stale SSD custody
and bounded pairs only. A bounded rehearsal proves mechanism and interfaces,
not an exact score, population result, or near-rail payoff.

## STORES CONSULTED

`CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`; delegated authority
`ddm_eg1_endgame_chain_20260729T000937Z.wrapped.prompt.txt`; live per-arm and
fleet inboxes; `.omx/research/ddm_tb1_renderer_build_20260728.md`;
`.omx/research/SPEC_tr1_trained_partition_renderer_20260728.md`;
`experiments/train_tr1_partition_renderer_mlx.py`; stale TB1 T2 lottery/plain
checkpoint custody under `/Volumes/VertigoDataTier/pact/ddm_tb1_20260728`;
E4/E5A exporter, receiver, adapter, and parse-back receipts;
`tools/levelset_receiver_bijection_gate.py`; EU1 and FD2 memo/receipt surfaces;
`src/tac/optimization/uint8_lattice_feasibility.py`;
`direct_description_joint_descent.py`; `ddm_family_d_gn_description.py`;
`pose_frame0_inverse_solve_probe.py`; the terminal-pose closure memory; #344
`ncde_trajectory.py`; the #216 unified-flow synthesis; the #475 grokking
receipt; registered equation
`ddm_v17_validity_radius_law_20260723`; the #341 basin-finisher receipt; MENU1
prices; PC2 and R1 pose receipts; the pinned `upstream/{evaluate.py,
evaluate.sh,modules.py,frame_utils.py}` closure.

## E1 — R6/TR1 push-button byte-close rehearsal

### Inventory corrections

- **MEASURED:** the stale T2 lottery final checkpoint exists at
  `/Volumes/VertigoDataTier/pact/ddm_tb1_20260728/t2_n600_lotto/checkpoints/
  stage_seg_trunk_tau_final.npz`, 14,963,191 bytes, SHA-256
  `67454bc9ac30ea6f971c29c6c6cea7b8af8d98293ed77e44be3a14b8a8667bc2`.
  It contains EMA base/delta tokens and lottery renderer tensors, but no
  standalone production token packet.
- **MEASURED:** the existing E4/WS1 path is deliberately sealed to exactly two
  WS1 streams and the `W_seg`/`W_joint` candidates. It is not a generic TR1
  adapter. EG1 therefore preserves that seal and adds a focused typed TR1
  grammar using the same deterministic framing/custody discipline.
- **MEASURED correction:** the charter/SPEC reference to “#402 exact
  consumption” is a miscitation. #402 is telemetry liveness. The
  counted-but-inert / receiver-consumption-bijection gate is **#417**.
  Static #417 analysis is not enough for a dynamic packet, so the TR1 parser
  additionally requires fixed section order, explicit lengths and offsets,
  per-section SHA-256, final-cursor equality, and refusal of trailing bytes.

### Counted/free boundary and receiver contract

The packet has exactly four ordered, length-prefixed sections:

| section | counted content | receiver obligation |
|---|---|---|
| token stream | quantized realized EMA pair tokens | reconstruct every pair token consumed by the renderer |
| renderer | lottery supermask and learned modulation/bias state | regenerate the fixed seed bank for free and consume all learned state |
| selector ledger | every video-selected decoder/config choice | configure and validate the receiver; no selector hidden in free code |
| pose stub | explicit typed unresolved stub | consume and validate it while making no pose-mechanism or pose-score claim |

The generic parser, PRNG expansion, renderer algorithm, resize, and packet
interpreter live in free code under rule 118. Learned/video-derived tokens,
mask, modulation/bias state, and selected configuration live in the counted
packet. The export quantizes the **combined** EMA `base + delta` token seen by
the actual forward pass; it does not reuse TB1's former size-only estimate.

### Rehearsal receipt

Receipt
`.omx/research/ddm_eg1_tr1_rehearsal_20260728.json`, SHA-256
`61992dc3249d87da1396b334ae7ff3d60cd15d7ebe67df099768004ab64c4809`,
records:

- a canonical 504,249-byte packet (SHA-256
  `1dd5453a36cf75a6f4be80287961f0e455c8e2e6b35aecc991a174204665f7bf`)
  inside a deterministic stored 504,736-byte ZIP (SHA-256
  `5c33f231339854e929dc0d1ef19abf51abd7ede521ad0e04650f572e8b5c12db`);
- four counted Brotli-Q11 sections: tokens 499,587 bytes, lottery renderer
  3,341 bytes, selector 535 bytes, inert pose declaration 83 bytes;
- exact parse/re-emit, per-section consumption, closed final cursor, and
  trailing-byte refusal;
- same-deployed-value NumPy/MLX parity on stale pair 0: token max error zero,
  pre-R max/mean error `2.288818e-4` / `9.401179e-6`, and camera-byte
  agreement `0.9999793579` (63 mismatches of 3,052,008), above the registered
  0.9997 gate; and
- a byte-identical locked `evaluate.sh`/`evaluate.py` CPU invocation on one
  deterministically derived two-frame sample, exit 0 and one report sample,
  after a 256 MiB SSD storage preflight. The run is explicitly
  `PASS_INTERFACE_ONLY_NONCOMPARABLE`; the report's partial score and rate
  have no scientific comparison authority.

The separately measured source-EMA-fp32 to deployed-fp16 gap is **not**
mislabelled as backend parity. The counted archive contains no checkpoint
path, checkpoint hash, config hash, or provenance-only parameter-set label;
those remain in the external receipt.

The bounded upstream smoke, if green, establishes only that the locked
`evaluate.sh` entrypoint can inflate and consume the packet on the rehearsed
pairs. Its partial distortion/rate output is not a contest score and must not
be compared with 0.172 or 0.15.

## E2 — derived stopping and handoff policy

For an operating state \(x=(d_s,d_p,B)\), the only score authority used by
the policy is

\[
S(x)=100d_s+\sqrt{10d_p}+\frac{25B}{37{,}545{,}489}.
\]

For a same-parent action quote \(a\), let

\[
G_a=S(x)-S(x_a), \qquad \eta_a=G_a/C_a,
\]

where \(C_a\) is measured wall time. A terminal action may replace another
training window only when its conservative score-gain-rate lower bound is
strictly greater than the training window's upper bound. A foreign-parent
quote, soft or unparsed endpoint, non-consumed payload, non-improving endpoint,
missing wall time, missing interval, or overlapping intervals cannot trigger a
handoff.

### Typed decision rules

| evidence at the same operating point | policy output |
|---|---|
| training quote exists; no admissible finisher quote | `CONTINUE_BOUNDED_WINDOW` |
| any required comparison price is absent or intervals overlap | `MEASURE_FINISHER_QUOTE` |
| admissible seg GN dominates and topology is stable with no transitions pending | `HANDOFF_SEG_GN` |
| admissible QDBS dominates | `HANDOFF_QDBS` |
| admissible terminal pose dominates and composition is frozen/stable | `HANDOFF_TERMINAL_POSE` |
| byte-closed candidate has the required hard public closure | `R6_EXACT_EVAL` recommendation only |
| malformed, foreign-parent, or authority-mismatched evidence | `REFUSE_INSUFFICIENT_EVIDENCE` |

Every decision carries the operating-point identity, evidence axis, pair
count, verdict scope, and serialized input quotes. The policy is advisory and
cannot dispatch.

### Why the named trajectory receipts are not absolute triggers

- **#344 NCDE:** shadow-only, `actuation: NONE`; its defaults are provisional
  forecast features, not stop thresholds.
- **#216 saddle-to-saddle:** distinguishes staircase from smooth descent but
  has no calibrated classifier. A staircase can prioritize a QDBS quote; it
  cannot stop training by itself.
- **#475 grokking:** its negative is scoped to a fixed 31-feature quadratic
  chart and explicitly has no witness stage-advance authority.
- **v17 validity radius:** \(\rho=\Delta F_{\rm realized}/\Delta
  F_{\rm predicted}\) adjusts a proposal radius. Positive \(\rho\) does not
  accept; negative \(\rho\) shrinks.
- **#341 GN economics:** the K=8 subset improved locally but worsened n600
  d_seg by 5.1%. Only a full-population, in-trainer, topology-stable successor
  is admissible. Its historical time is a quote prior, not a current-parent
  price.

MENU1, QDBS, PC2, and R1 prices remain parent/formulation scoped. In
particular, PC2 measured `-0.247501 S` for +23 bytes over 1,275.255 seconds on
its own parent; the hoped-for 2 KB / \(2.33\times10^{-5}\) pose endpoint is not
measured on TR1.

### Composed-candidate arithmetic

At the exact rate coefficient \(25/37{,}545{,}489 =
6.6585895312\times10^{-7}\) score/byte:

| corner | \(d_s\) | \(d_p\) | archive bytes | exact derived \(S\) | vs 0.172 | vs 0.15 |
|---|---:|---:|---:|---:|---:|---:|
| TR1 A, decimal 149 KB | 2.97e-4 | 2.33e-5 | 149,000 | 0.144177 | below | below |
| TR1 B | 3e-4 | 2.33e-5 | 196,000 | 0.175773 | above | above |
| TR1 B-prime | 5e-4 | 2.33e-5 | 196,000 | 0.195773 | above | above |
| banked-pose C | 5e-4 | 1.61e-3 | 201,000 | 0.310723 | above | above |
| pp1 conditional | 3e-4 | 2.33e-5 | 215,616.5 | 0.188835 | above | above |
| stale T2 lottery, even pose=0 | 0.013833 | 0 | 534,597 | at least 1.739266 | above | above |

At \(d_s=3\times10^{-4},d_p=2.33\times10^{-5}\), strict integer byte
ceilings are 190,334 for 0.172 and 157,294 for 0.15. At
\(d_s=5\times10^{-4}\), they tighten to 160,297 and 127,257. With
\(d_p=0.001610\), sub-0.15 is impossible already at
\(d_s=3\times10^{-4}\), even at zero bytes. These are exact arithmetic
ceilings, not attainable-price forecasts.

The deterministic arithmetic receipt is
`.omx/research/ddm_eg1_policy_arithmetic_20260728.json`, SHA-256
`cdfe86de9c3d3fc408a2a8429b6a1096721878882b2757bf0ccb869c4a0fd5d7`.
It regenerates byte-identically from `tools/derive_ddm_endgame_policy.py`.
The QDBS cost is deliberately a time bound, not a payoff quote: 49 full
verdicts at the measured 423–514 seconds/verdict imply 20,727–25,186 seconds
(5.76–7.00 hours), while gain remains `null`.

## E3 — terminal finishers

### FD2-QDBS hard terminal search

The QDBS harness generalizes the fresh hard-oracle acceptance discipline of
`repair_with_hard_oracle` to grouped **description** coordinates rather than
pretending a group is one pixel `IntegerMove`. It requires:

- exactly 16 signed singleton proposals and 8 grouped proposals derived from
  no-new-score/scorer signals;
- exactly 24 precommitted matched integer-random controls;
- at most 48 candidate evaluations plus one shared base;
- deterministic, all-distinct proposal/control identities with a reserved
  shared-base identity and fixed seed;
- compile, parse-back, exact-consumption, realized joint-action evaluation for
  every candidate; and
- strict score improvement for selection.

Smooth GN/Fisher output may rank the proposal menu but never accepts a
candidate. Production custody is explicitly self-attested: even a strict
full-n600-shaped improvement can produce only
`REQUIRES_EXTERNAL_GOVERNOR`; the module cannot authorize handoff, promotion,
score, or pointer movement. Production requires an absolute resume path,
binds the parent plus compiler/receiver/evaluator identities, atomically
retains every exact base/candidate archive, and commits a deterministic
idempotency-keyed `PENDING_EVALUATION` intent before invoking the evaluator.
After a crash, the hash-bound durable evaluator protocol receives the same
key; cached evidence is reusable only after the retained archive is reopened
by byte count and SHA-256. The durability protocol and all custody labels
remain self-attested and therefore require external-governor verification.

Receipt `.omx/research/ddm_eg1_qdbs_rehearsal_20260728.json`, SHA-256
`18cff2b2c844bbc114e5b6bd46d3be723d8d791f9666ae20ec139cc339e3e46a`,
uses the real stale 368-coordinate FD2 checkpoint (344 active coordinates)
and its scorer-trained first moment. All 16 signed singletons, 8 groups, and
24 matched controls remained inside that active set; the run executed 48
candidates plus one shared base through the compact test
compile/parse/consume/action chain. Its best test-oracle delta was
`-5.4347826087e-7` score units. That number is a deterministic mechanism
canary only: no production compiler, public receiver, frozen scorer, n600
evaluation, or current-parent payoff was exercised.

### Terminal six-equation pose GN

The pose solver starts only from frozen final composed uint8 frames. It takes a
caller-supplied generic low-frequency basis/render function and frozen Pose6
oracle, measures the actual finite-difference Jacobian after coefficient
quantization, basis normalization, receiver expansion, add/clamp/round, and
relinearizes for a bounded 2–3 iterations. Frame 1 is asserted byte-identical
throughout. Line search accepts only a realized Pose MSE decrease that also
strictly improves the exact joint action including payload bytes.

The receiver packet stores only the basis/seed selector and quantized
per-pair coefficients. It contains no scorer, target, surrogate, or copied
PR130 video-derived artifact. The PR130 lessons used are structural only:
low-frequency fields, bicubic expansion, zero-mean/unit-RMS normalization,
explicit amplitude handling, and per-pair coefficients.

Receipt
`.omx/research/ddm_eg1_pose_gn_rehearsal_20260728.json`, SHA-256
`632d1894c2c30b34ba4d446d71f2ca4289bbeac3ef16c12ce962b378123286f3`,
is the exact output of one `--oracle both` invocation. The pinned frozen
PoseNet one-pair advisory moved its local target MSE from
`5.2816893588` to `1.3504594700`, ending at coefficients
`[0,5,0,1,5,4]`; every realized verdict kept frame 1 byte-identical.
This is not a candidate payoff: `d_seg=0` records only the terminal
frame-1-frozen delta and omits the parent constant, while the 107-byte
artifact is only the terminal section, not an outer archive.

The production API separately binds exact outer-archive and packet bytes,
rechecks evaluator archive size/SHA, types parent/compiler/receiver/evaluator/
upstream custody and contest axis, and requires atomic manifest/per-verdict/
per-iteration/completed resume ledgers. Because that custody is self-attested
inside this module, even a strict full-n600-shaped improvement sets only
`external_governor_review_required`; `governed_handoff_eligible`,
`production_accepted`, `promotion_allowed`, `score_claim`, and
`pointer_moved` remain false.

## Verification, triality, and handoff

| surface | final verification |
|---|---|
| focused tests | 58 passed: TR1 8, policy 19, QDBS 22, pose 9 |
| static checks | Ruff check/format and Python byte-compilation clean on all 12 new Python files; `git diff --check` clean |
| receipt determinism | all four generated JSON receipts reproduced byte-identically; receipt SHA-256 values are recorded above |
| review gate | two explicit review passes recorded for every new/modified Python file; no review-gate override |
| prohibited work | no n600 scorer job, launch, paid dispatch, candidate promotion, score claim, or pointer mutation |

- DSL/control leg: typed TR1 packet, policy, QDBS, and pose contracts.
- DAG leg: this memo and
  `.omx/research/ddm_eg1_endgame_chain_DAG_FEED_20260728.md`.
- Equation leg: reuses the exact contest score and registered v17 validity
  radius law; no new empirical law is claimed.
- Continual-learning: the generated receipts and this memo make the post-burn
  chain restartable without rediscovering E4 sealing, #417 consumption, quote
  scope, or terminal ordering.
- MAIN review must inspect rule-118 byte homes, exact stream consumption,
  stale/deploy parity, bounded evaluator custody, same-parent quote matching,
  full-n600 production gating, uint8-realized pose acceptance, and every
  no-score/no-pointer label before landing.
