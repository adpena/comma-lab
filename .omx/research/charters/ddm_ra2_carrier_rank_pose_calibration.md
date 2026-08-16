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

---

## AMENDMENT 2 (2026-08-16, round-1 recursive adversarial review, independent reviewer arm)

An independent reviewer re-derived this charter's premises from source. **The rank-4 BYTE
conclusion survives. The rank-4 MSE conclusion does not, and the payloads on disk were
corrupt.** Six binding corrections; two of them correct arguments *this charter* made.

### C1 — CRITICAL, FIXED: the emitted payloads did not decode back (11 of 12 corrupt)

ra1 hand-rolled the codec's zigzag as `((delta << 1) ^ (delta >> 63)) & 0xFFF`. The
receiver's coefficient cumsum is **modular** (`inflate.py:278`, `torch.cumsum(...) & 0xFFF`),
so `delta` must be wrapped into signed 12-bit **before** zigzag. Masking **after** zigzag
truncates the high bit of an already-doubled value and decodes to a different number whenever
`|delta| > 2047` — which happens at 11 of 12 ranks (max |delta| reaches 3,506). The shipped
`carrier_codec._zigzag_signed` **raises** on out-of-range; the hand-rolled copy dropped that
guard. Rank-4 true on-disk MSE was 638.13 against a receipted 156.93.

FIXED at both sites (`67afd3fd83`) with `((delta + 2048) & 0xFFF) - 2048` before zigzag, plus
a mandatory `_assert_round_trip` that replays the receiver's exact reconstruction and refuses
to emit on mismatch. All 12 ra1 ranks and all 44 ra1b payloads regenerated, 12/12 and 44/44
verified. Cost: **+7 B at rank 4**. **ra2 MUST use the regenerated payloads** and must assert
`round_trip_verified` in its own receipt — the pre-fix payloads would have made ra2 measure a
carrier nobody designed and report a false NO-GO.

### C2 — the keep set is EXHAUSTIVELY optimal, not energy-greedy; P2 is REFUTED as published

`ddm_ra1b_exhaustive_keepset_refit.py` searches all C(12,r). Greedy is suboptimal at **10 of
11 ranks**. ra1's greedy keep set ranks **269th of 495** at r=4. P2's claim — "a LOWER BOUND on
the reconstruction error of EVERY rank-r carrier … a rank that fails here fails under every
refit heuristic" — is FALSE as published: the refit is optimal **given** the keep set; the keep
set was a heuristic; the space is exhaustible in seconds. **Two independent derivations agree**
on the r=4 optimum (keep `[1,2,3,8]`, MSE 104.665).

### C3 — the baseline is 22,161 B, not 22,278 B; the rank-4 margin is +248 B, not +295 B

The archive's actual carrier stream is **22,161 B** (the custody pin at the head of this
charter already says so). Every "saved bytes" row was computed against a 22,278 B CPR1
rebaseline and overstated by 117 B. Corrected, at the **exhaustive** rank-4 keep set:

| quantity | value |
|---|---:|
| coded bytes (exhaustive r=4) | 7,499 |
| saved vs the real 22,161 B stream | **14,662** |
| the rung (`RUNG_BYTES`) | 14,414 |
| margin over the rung | **+248 B** |
| rate credit | 0.0097628 S |
| % of the 0.0095973 gap | **101.7%** |
| affordable d_pose ratio | **4.739×** |

Greedy r=4 lands 7,576 B → +171 B margin. Both still clear the rung; the margin is 42%
smaller than this charter first published, so **report the margin, never just "clears."**

### C4 — this charter's pk2 refutation was WRONG; the honest statement is a units error

§"The gate that is INVALID" argued the pk2 gate is refuted because "the rank-12 FULL-RANK
control realizes int12 MSE 2.4865e-05, 9.9× the gate." **Withdrawn.** Rank 12 is *not* a
full-rank control — it re-derives `sub_scale` and requantises, so its residual belongs to the
tool's requantizer, not to the shipped bytes; the shipped codes against themselves are exactly
0.0. The correct finding is narrower and more useful: **pk2 never defined or measured the
quantity its 2.5e-6 gate is denominated in.** It is an undefined-units gate, not a gate the
frontier fails. Do not cite the 9.9× figure. The break-even rule in §"The VALID decision rule"
is unaffected — it needs no gate literal.

### C5 — the shipped receiver HARD-GATES the carrier body length; a container port is in scope

The receiver expects a carrier body of exactly 22,183 B. A rank-reduced carrier cannot be
dropped in as this charter's step 2 describes. Every one of these hardcodes 12 and must be
ported for the swap to parse back: `PACKED_CAP1_SECTION_BYTES`,
`_restore_packed_cap1_metadata`, `_restore_cap1`, `decode_cap1(dimensions=)`, the F0C1 u16
length field, and the Q2C1 overlay. **This is a porting item, not a wall** (the receiver is
ours, rule-118 free code) — but it is REQUIRED work that must be scoped before the swap, not
discovered at runtime.

### C6 — this charter's warnings paragraph was wrong in evidence AND mechanism

Step 1 says "0/12 raise" under `errstate(all="raise")` and instructs the next arm not to
re-litigate. **Measured: 12/12 raise.** My check wrapped `lstsq` alone; the raise comes from
the `matmul` at `ra1:263` (`Grc @ coeff.T`) and is a numpy-1.26 matmul false positive —
data-independent, unrelated to torch, and visible in the re-run output above the ladder. The
*conclusion* survives (the solve is correct: verified to 1.6e-15, cond ≤ 15.71), but the
evidence and mechanism were both wrong, and "do not re-litigate" is withdrawn. Replace it with:
the warnings are a known numpy-matmul false positive at `ra1:263`; do not suppress warnings
globally; do not treat them as a numerical defect.

### What SURVIVED independent re-derivation (cite these freely)

- Custody re-derived byte-identically from the four pins.
- **P1 seg-invisibility HOLDS**, including the adversarial half (aliasing, cross-frame stats).
- The archive is **ONE STORED ZIP member** (compress_type 0, compress_size = file_size =
  182,659, +100 B fixed overhead), so carrier byte changes map **1:1** to archive bytes and
  `archive_bytes_if_adopted` is exact on the byte axis.
- The closed-form MSE is validated against a real render: 156.543 actual vs 156.926 closed
  form — **0.24% conservative**.
- `RUNG_BYTES = 14,414` is correct.

### A strictly better candidate that is still unbuilt

Rotating the rank-r subspace (whitened eigendecomposition of the Gram) is receiver-legal for
free — the basis atoms are stored data, rotation commutes with the bicubic→mean-subtract
chain, and per-atom RMS is absorbable into the coefficients. Closed-form error at r=4 is
**6.00%** (vs 20.41% exhaustive-selection, 30.60% greedy); r=6 is 2.56%. Eigen-sum closes
exactly (512.7490). **Error-only so far — not yet priced in bytes.** If ra2's d_pose(MSE)
curve is steep, this is the rung to price next; a pose-Jacobian-weighted (Fisher) rotation is
better still and also unbuilt.

---

## AMENDMENT 3 (2026-08-16, MAIN) — the ra2a decode path was at the WRONG LAYER; the corrected chain, and why no archive writer is needed

**Status: ra2a is BLOCKED and the blocker is now fully diagnosed.** Three plumbing layers were
peeled in the prior pass (runtime path · receiver identity · fx1 tree path). The fourth is not
plumbing — it is a scope error in this charter, and it is now named exactly.

### C7. ra2a read the archive at the wrong layer

ra2a did `zipfile.read("p")` → `receiver.split_payload(payload)`. That is the CPR1 semantic-pose
receiver's entry point, and it correctly REFUSED ("combined payload has no complete token
section") because the hv1 archive is **F26-wrapped**: `GEN/inflate.py` is the F26 outer wrapper,
and `split_payload` expects the already-unwrapped combined payload. There are TWO receivers,
nested; the charter assumed one.

**The exact chain** (verified at source in `GEN/runtime/f26_inflate.py::inflate_archive`):

```
parts             = read_residual_archive(archive_path)              # F26 section split
carrier_blob, _   = split_frame0_selector_carrier(parts.carrier_blob)
canonical_carrier = materialize_cpr1(carrier_blob, renderer)
semantic_pose     = pack("<II", 40252, len(canonical_carrier)) + bytes(40252) + canonical_carrier
_, basis, coeff   = renderer.unpack_semantic_pose(semantic_pose)
...                                                                   # + compensation overlay
render_video(semantic, basis, coeff, tokens, destination, device)
```

Note that `ddm_ra1_carrier_rank_refit_preproof.py` already reads at the RIGHT layer — it consumes
a pre-extracted `receiver_decode/outer_carrier.bin` and calls `materialize_cpr1` directly. ra1's
coded-byte numbers are therefore unaffected by this defect; only ra2a's decode was wrong.

### C8. `residual_archive` is READ-ONLY — and the pose measurement does not need a writer

There is no archive writer in `runtime/residual_archive.py` (reader + decoders only). So
"re-pack a modified archive and inflate it" is not an available operation on our side.

**It is also not required.** The carrier enters the render at exactly one place
(`inflate.py:660`): `carrier = einsum("bk,kchw->bchw", coeff, basis)` → bicubic → frame_0.
d_pose is a function of the RENDERED frames, not of the container. So the honest measurement is:
unwrap once, perturb `coeff`, render twice, score twice. The container round-trip is only needed
for the RATE column — and the rate column is already exactly known from ra1b's measured coded
lengths through the shipped codec.

**The measurement therefore SPLITS, and each half already has its instrument:**

| column | instrument | status |
|---|---|---|
| rate (bytes returned) | ra1b coded lengths through the shipped CPR1 codec + Brotli cell | MEASURED |
| d_pose(fidelity) | unwrap → perturb coeff → `render_video` ×2 → PoseNet | the remaining build |
| affordance bar | `carrier_rate_credit_pose_affordance_v1` | REGISTERED, evaluator executable |

### C9. The mirror needs a byte-identity control, and it has an exact one

Reproducing the f26 unwrap chain inside ra2a is duplication of a pinned source. That is
acceptable ONLY with a control that proves the mirror is faithful: **at α = 1 (carrier
unmodified) the mirrored chain must reproduce the RETAINED base render byte-identically.** If the
render sha does not match the retained receipt, the mirror is wrong and no ladder row is
admissible. Pin the f26_inflate source sha in the receipt alongside it.

### C10. Fire order, corrected

1. Build the mirrored unwrap + the α=1 byte-identity control. Refuse on sha mismatch.
2. Fire **α = 0 FIRST** (carrier deleted). Per the registered affordance law it returns 153.8% of
   the remaining gap and sits under the loosest bar (7.72× tolerance) — it is one decode, it
   bounds the whole d_pose(fidelity) curve, and it answers affordability in the easiest regime
   rather than the hardest.
3. Only then walk interior rungs, and only if α=0 is NOT affordable.

Ordering interior-rungs-first was the charter's error, and it is the same LEVEL error as ranking
the ladder by Euclidean MSE: both optimize something other than the question being asked.
