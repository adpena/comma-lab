# ddm_ra2 — CALIBRATE THE CARRIER RANK LADDER (the one number that decides 102% of the gap)

You are measuring d_pose as a function of carrier rank on the LIVE frontier archive.
Everything else on this ladder is already MEASURED and retained. One number per rank
closes it.

## What is already MEASURED (do not re-derive; verify the pins, then build on them)

Frontier (VERIFIED against .omx/state/canonical_frontier_pointer.json this turn):
  effective_frontier.score = our_local_frontier_contest_cuda.score = 0.15959729295498598
  archive_sha256 = 80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e
  extra.archive_bytes = 182,759          upstream_leaderboard_snapshot.best_entry = 0.162
  gap to 0.15 = 0.0095973

  COMPONENTS — the pointer stores NO component breakdown, so do not "cite" them; they are
  DERIVED and self-checking, and I re-derived them this turn:
    rate = 25 * 182759 / 37545489 = 0.12169171641365491   (exact, 17 places)
    pose = 0.0082945765                                    (T4 receipt)
    seg  = S - rate - pose = 0.029611000                   (closes to 9 places)
  The triple CLOSING is the staleness check: if any component were superseded the residual
  would not land on 0.029611. Re-run this closure before quoting any component.

ra1 receipt (retained, SHA-pinned custody inside it):
  /Volumes/APDataStore/pact/ddm_ra1_carrier_rank_refit_20260816/retained/CARRIER_RANK_REFIT_PREPROOF.json
  + payloads/rank{01..12}_refit.br  (real coded bytes, shipped CPR1 codec + shipped Brotli q11)
  Tool: experiments/ddm_ra1_carrier_rank_refit_preproof.py  (fires clean, $0, 0.46 s)

The ladder (exact coded bytes; mse = carrier-field MSE in pixel units, signal energy 512.7490):

| rank | coded B | saved B | rate credit S | % of gap | carrier err % | affordable d_pose ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 12 |22,257|     21|1.398e-05|  0.15|  0.000| 1.003x |
| 11 |20,611|  1,667|1.110e-03| 11.57|  0.960| 1.286x |
| 10 |18,655|  3,623|2.412e-03| 25.14|  2.672| 1.666x |
|  9 |16,666|  5,612|3.737e-03| 38.94|  3.943| 2.104x |
|  8 |14,691|  7,587|5.052e-03| 52.64|  7.059| 2.589x |
|  7 |12,914|  9,364|6.235e-03| 64.97| 15.813| 3.068x |
|  6 |11,229| 11,049|7.357e-03| 76.66| 23.224| 3.561x |
|  5 | 9,269| 13,009|8.662e-03| 90.26| 26.109| 4.179x |
|  4 | 7,569| 14,709|9.794e-03|102.05| 30.605| 4.756x |
|  3 | 5,708| 16,570|1.103e-02|114.96| 34.653| 5.430x |
|  2 | 3,884| 18,394|1.225e-02|127.62| 50.662| 6.134x |
|  1 | 1,940| 20,338|1.354e-02|141.10| 66.168| 6.931x |

## Two premises I VERIFIED AT SOURCE this turn (cite these; re-verify, do not assume)

(P1) The carrier renders frame_0 ONLY, so the rank cut is SEG-INVISIBLE BY CONSTRUCTION.
     src/tac/pr130_runtime/fx1_runtime_tree/inflate.py:645-673 —
       output[2*i + 1] = master  <- semantic(tokens), interpolate bilinear   (frame_1)
       output[2*i    ] = slave   <- einsum(coeff, basis) carrier, bicubic    (frame_0)
     upstream/modules.py SegNet consumes x[:, -1, ...] = frame_1 = the master.
     => d_seg CANNOT move with carrier rank. ra1's pre-registered "seg falsifier"
        (frame_1 byte-identity at every rank) is a STRUCTURAL proof, not a measurement.
        You still run it ONCE as a positive control that the implementation matches the
        proof; if frame_1 bytes ever differ, STOP and report — the premise is wrong.
     => PoseNet reads BOTH frames, so the carrier is a PURE POSE actuator.

(P2) The rank-r REFIT c_r = (Br^T Br)^-1 Br^T B c is the least-squares OPTIMUM, hence a
     LOWER BOUND on the reconstruction error of EVERY rank-r carrier that keeps the shipped
     receiver's linear synthesis. A rank that fails here fails under every refit heuristic.

## The gate that is INVALID — do not use it to close anything

The ra1 receipt carries pk2_pregate = {mse: 2.5e-6, min_bytes: 2000}. That gate is
REFUTED BY ITS OWN VEHICLE: the rank-12 FULL-RANK control (exact re-encode, zero rank
loss) realizes int12 MSE 2.4865e-05 — 9.9x the gate. A gate the shipped frontier bytes
themselves fail cannot close this family. Genus: inherited ceilings refuted by their own
arithmetic — see .omx/research/ddm_et1_eta_on_the_priced_band_20260803.md (the band family
died on a MEASURED eta with a rising bar, not on a transferred ceiling). Record this
explicitly in your verdict; do not silently drop it.

The VALID decision rule is break-even, and it needs no d_pose literal:
  pose term = sqrt(10 * d_pose), so a rate credit R is paid for iff
      d_pose(rank) / d_pose(base)  <  ((POSE + R) / POSE)^2      with POSE = 0.0082945765
  i.e. the "affordable d_pose ratio" column above. ADMIT the largest saving whose measured
  ratio is under its affordance, then compose the exact net ΔS.

## What to do

1. VERIFY the four ra1 custody pins (bytes + sha256) and re-run the ra1 tool to reproduce
   the ladder. Expect byte-identical rows. Note: ra1 emits benign RuntimeWarnings from a
   STICKY FP status flag set by an earlier torch/BLAS op and attributed by numpy to the
   next ufunc — I proved this spurious this turn (flag consumed + errstate(all="raise")
   around the solve: 0/12 raise, 0 non-finite in or out, rows identical). Do not re-litigate;
   do not "fix" it by suppressing warnings globally.

2. BUILD the swap harness: substitute payloads/rank{r}_refit.br for the shipped carrier
   section in the hv1 archive, byte-close a real candidate archive per rank, and confirm the
   shipped receiver parses it back. Reuse the mp2 generation harness at
   /Volumes/APDataStore/pact/ddm_mp2_mixed_precision_receiver_close_20260815/generations/hv1_base_control
   — the encoder at src/tac/pr130_runtime/fx1_runtime_tree/carrier_codec.py is VERIFIED to
   re-encode the shipped carrier byte-identically (22,307 B, sha 709ea928c2d73c59...).
   Do NOT hand-roll a second codec.

3. MEASURE d_pose per rank through the REAL decode path on the torch-CPU authority.
   The MLX-PoseNet drift law is MEASURED in
   .omx/research/ddm_pk4_optimal_form_frame0_pose_verdict_20260813.md (0.55% rel drift;
   CPU parity 2.29e-5 vs retained, 3,400x tighter than the MLX leg) — CPU is the default
   and the authority. Do NOT measure d_pose on MLX.
   Advisory n600 is preferred and affordable — the carrier path is frame_0-only and cheap.
   If you must subset for a first read, use SEEDED STRATIFIED-RANDOM, NEVER a prefix
   (m88/m96: pose prefixes measure 2.5-4.2x HARDER than the population; a prefix here is
   exactly the false-negative shape). Report the sampling scheme with the number.
   Run the ranks CHEAPEST-FIRST from rank 11 down only far enough to bracket the knee —
   the map d_pose(carrier MSE) is expected monotone, so 3-4 ranks bracket all 11.

4. ADJUDICATE against the affordance column. Emit, per rank: coded bytes, measured d_pose,
   ratio vs base, affordable ratio, exact net ΔS, ADMIT/REFUSE.

5. If ANY rank is ADMIT, seal a dual-axis T4 fire-order for the best one (MAIN fires; the
   canonical row is only NAMED after the T4 gate). If none, the verdict is
   FORMULATION-scoped by (P2): rank reduction of the shipped 12-dim CPR1 carrier under the
   receiver's linear synthesis cannot pay — and then report the measured d_pose(MSE) curve,
   because it prices every future carrier-fidelity question on this vehicle.

## OPTIMAL FORM

REFERENCE form: the family's optimal form is a rank/precision reduction of a linear
synthesis carrier, judged by the exact scored quantity (d_pose through the real receiver
and the frozen PoseNet), against a real-coder byte count. This charter is AT reference
form on mechanism: real shipped codec, real receiver, real scorer, exact bytes.

Declared deltas:
  - SCOPE reduction (legal): a first read MAY use a seeded stratified-random subset to
    bracket the knee; the ADMIT decision requires n600 on the torch-CPU authority.
  - SCOPE reduction (legal): ranks measured cheapest-first to bracket, not all 11.
  - MECHANISM reduction: NONE. No proxy for d_pose, no surrogate receiver, no MLX PoseNet
    as authority, no synthetic carrier.

Provenance pins:
  archive   182,759 B  sha256 80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e
  carrier.br 22,161 B  sha256 fd14aabcb9daa5f1dd1c9c6e63e745a88f2978766e3129b184dd3a9ac7334de0
  carrier.raw 22,219 B sha256 065fce08fc3d44e49d29ad624561cbef86d01282cc73dcd32533b5d63115bd9f
  outer_carrier 22,242 B sha256 196f0e5136f4d6bfd22c4cf24ad779eee55f6e95a4f5f5994ae09a4fc268b6ef
  tool experiments/ddm_ra1_carrier_rank_refit_preproof.py
  receiver src/tac/pr130_runtime/fx1_runtime_tree/inflate.py

## Binding

ALWAYS KEEP THE PAYLOAD (P0): every rendered frame set / candidate archive you materialize
is PERSISTED with sha256 + byte count in the receipt. Never a scalars-only artifact.
Payloads to /Volumes/APDataStore/pact/ (VertigoDataTier is near-full).
No launches from the arm. NO Claude/co-author attribution on commits.
Report honestly: if the ladder dies, the d_pose(MSE) curve IS the deliverable.

---

## AMENDMENT 1 — 2026-08-16, round-1 recursive adversarial review (CRITICAL, self-caught)

**This amendment BINDS and overrides any conflicting line above.** Caught by MAIN reviewing
its own in-flight ra2 build before a scorer slot was spent. Two defects, one root cause.

### D1 (CRITICAL) — the affordance column is CUDA-denominated; the measurement was routed to CPU

The break-even rule above uses `POSE = 0.008294576541331089`, which is the frontier's
**contest-CUDA** pose contribution (verified: `/incumbent/score_pose_contribution` in
`ddm_hv1_t4_sealed_fire_order_ep0634_20260815.json`, axis `[contest-CUDA] Tesla T4 n600`;
implied `d_pose = POSE²/10 = 6.880e-06`, matching `ddm_pv1_pose_floor_and_admission_bar_20260816.md`
§ "the frontier's d_pose is 6.88e-06"). Step 3 then routes the measurement to the
**macOS-CPU** authority, where the same bytes measure `d_pose = 1.4747e-04` (pv1 line 91,
n600 advisory) — **21.4× larger**, consistent with the settled `#1054` CPU row (pose 21×
CPU-degraded).

Judging a CPU-measured ratio against the CUDA-derived affordance is a cross-instrument
comparison, and the bias runs the wrong way. On the CPU axis the pose contribution is
`√(10·1.4747e-04) = 0.038402`, so the affordance is far tighter:

| rank | saved B | rate credit R (S) | affordance on CUDA | affordance on CPU | bar ratio |
|---:|---:|---:|---:|---:|---:|
| 11 | 1,667 | 1.110e-03 | 1.286× | 1.059× | 1.21× |
| 8 | 7,587 | 5.052e-03 | 2.589× | 1.280× | 2.02× |
| 6 | 11,049 | 7.357e-03 | 3.561× | 1.420× | 2.51× |
| **4** | **14,709** | **9.794e-03** | **4.756×** | **1.575×** | **3.02×** |
| 1 | 20,338 | 1.354e-02 | 6.931× | 1.830× | 3.79× |

At the headline rank the bar is **3.02× too loose** — the false-ADMIT shape. Worse, a CPU
d_pose is dominated by an instrument floor 21× the CUDA signal, so carrier damage is
*diluted* in the CPU ratio: the error is anti-conservative on the pose axis, the same
direction as the `m88`/`m96` prefix-bias law.

Root cause is a **wrong-object transfer**: `ddm_pk4`'s law says *CPU beats MLX* (parity
2.29e-05 vs MLX 0.55% rel drift). It does **not** say a CPU d_pose proxies a CUDA d_pose.
Genus: `cross-regime-constant-transfer-genus-finishing-stage`.

**Binding cure.** CLAUDE.md "Apples-to-apples evidence discipline" rule 2 — CPU and CUDA are
separate evidence spaces, never inferred from each other:
1. The CPU legs are a **SCREEN and knee-bracket ONLY**, never an ADMIT. Line 125's
   "the ADMIT decision requires n600 on the torch-CPU authority" is **SUPERSEDED**: the ADMIT
   requires a **dual-axis T4 row**, because the frontier is a CUDA row.
2. Every CPU screen must be judged against the **CPU-axis affordance** computed from a
   **same-instrument** CPU base d_pose measured in this run — never against the CUDA column,
   and never against a base literal quoted from another instrument.
3. The rank-12 full-rank refit is **not** the base (it saves 21 B, so it is not byte-identical).
   Measure a genuine rank-0 shipped-carrier leg through the identical pipeline.
4. Report every d_pose with its axis tag and its base's axis tag in the same row.

### D2 (CRITICAL) — the render shortcut is a MECHANISM reduction

MAIN's in-flight plan was to skip step 2's archive rebuild and instead synthesize the slave
frame directly (`einsum → 127.5+64·c/√12 → clamp → round → bicubic → round → uint8`),
reasoning that only frame_0 changes so the archive is unnecessary. **That is a second
receiver, and the OPTIMAL FORM block declares MECHANISM reduction: NONE.** It is the same
class the charter already forbids for the codec ("Do NOT hand-roll a second codec").

The path is not benign. `inflate.py:657` sets `pose_batch = 64 if device.type == "cuda" else 1`
— the receiver renders carriers at **batch 64 on CUDA and batch 1 on CPU**. Per the measured
et4 law (*batch shape is part of the forward instrument*; oneDNN flips argmax ties between
batch 1 and 16), a hand-rolled render can differ from the shipped one at `.round()` ties, and
would differ from the T4 render regardless.

**Binding cure.** Step 2 stands as written: build the real candidate archive per rank and
render through the **shipped receiver**. No reimplementation of the synthesis chain is
admissible as a d_pose measurement. If a fast pre-screen is wanted, it may rank ranks but may
not produce any number reported as d_pose.

### What is unchanged

The ra1 ladder, both premises (P1 frame-0-only ⇒ seg-invisible; P2 least-squares lower bound),
the pk2-gate invalidation, and the break-even *form* are untouched. Only the **axis of the
denominator** and the **realization path** are corrected.
