# ddm_mc1 — a decoder-derivable MOTION-COMPENSATED previous-field plane for the trained HPAC mixer: CEILING-REFUSED at step 1; the prior-law falsifier FIRED

**Arm:** ddm_mc1 (Fable, re-spawned 2026-09-04 23:20Z on MAIN's charter
`.omx/research/charters/ddm_mc1_motion_compensated_previous_plane_20260903.md`).
**Pointer at spawn and at close:** fs2 — S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600],
archive sha `a8f3a3791499b2b62ee4d16bc67f15f819f454dc9b88e3cce04fe50a30427bb6`. **The exact pointer did not move.**
Demand at held distortion: −41,817.8 B (archive ≤ 138,205.2 B). This arm bought zero of it.

**Typed verdict: `CEILING-REFUSED`.** The best decoder-derivable motion-compensated plane conditions the
shipped coder for **+159.60 B held-out** (pair-level two-fold; the best cell is a one-seed tilt row,
and the three-seed indicator minimum is +138.49 B) against a pre-registered refusal bar of 5,000 B. The charter's falsifier ("closed-form ceiling < 5,000 B") fired.
Steps 2–4 (warm-start retrain, RC64 exact price, fire/ready order) were **not executed**, by the
charter's own rule. No archive was built; no scorer ran; no Modal was spent.

Axis of every number below: `[macOS-CPU advisory / scorer-free EXACT byte measurement]` for the rows
control; `[model-ledger code length on the coder's own rows; REFUSAL-ONLY]` for every ceiling number.
Labels: MEASURED / DERIVED / INFERRED as marked. Craft per `docs/operating_manual_craft_handoff.md`
(§4 re-derive, §5 label, §6 attack your own conclusion, §7 answer first).

---

## 1. The object, and why it was worth a ceiling

The shipped integer HPAC (`submissions/semantic_joint_ctxmix/cpr1/hpac_integer.py`) conditions on the
previous pair's semantic field CO-LOCATED: `prepare_frame_context` one-hots `previous_raw` into
`conv_past` (3×3, 5→64), and the sparse receiver adds that `past` plane to the causal current-frame
features before the two dilated depthwise stages. Between pairs the ego-vehicle moves; the charter's
hypothesis (dc1's learned-receptive-field mechanism) was that a plane ALIGNED to the current pair —
derived only from already-decoded fields, so zero archive bytes — would let the trained mixer save
≥ 5,000 B of the 113,411 B stream at ≤ +1,500 B of model.

The design I priced: motion m_t estimated from field_{t−2} → field_{t−1} by deterministic INTEGER
search (argmax of class agreement on the edge band of field_{t−1}, identity-first tie-break), applied
once more to field_{t−1} (constant velocity); pairs 0–1 co-located. Six decoder-derivable rungs:

| rung | family | parameters searched per pair | decode cost (numpy, M5 Max, 1 thread) MEASURED |
|---|---|---|---|
| `shift` | global integer translation | (dy,dx) ∈ [−8,8]² (289) | 26 ms/pair |
| `zoom` | integer zoom about image centre + shift | s = 1+k/1024, k ∈ {0,2,…,40} × (dy,dx) ∈ [−3,3]² (637) | 134 ms/pair |
| `planar` | ground-plane quadratic flow (dy ∝ k(y−y0)², dx ∝ k(x−x0)(y−y0)) + shift | k ∈ {0,…,40} × y0 ∈ {160,176,192,208} × [−2,2]² (2,100) | 260 ms/pair |
| `block` | per-64×64-block translation (48 blocks) | (dy,dx) ∈ [−8,8]² per block | 47 ms/pair |
| `block_gated` | block, shift kept only if it beat identity by ≥ max(8, 10 % of the block's band) | same | 63 ms/pair |
| `block_median3` | block, component-wise median of the last three transition estimates | same | 55 ms/pair |

Rule 118: the estimator is a generic deterministic algorithm (free); the plane costs zero archive bytes.

## 2. The instrument — mi1's cross-fit on the coder's OWN rows (rows control PASSED)

I re-ran the shipped encoder over the retained exact field (sha `cc10a7b0…`, 117,964,800 B) on the fs2
fire tree (`/Volumes/VertigoDataTier/pact/ddm_fs2_carrier_resolve/fire_runtime_D_alternation/`) and
recorded every coding row the RC64 coder was handed (float32 × 5 classes × all positions).
**Control: the emitted stream is byte-identical to the shipped stream — 113,411 B, sha
`5601d6fd792c60c176e7cb7478e6033c4ed9a7e87404582340ed3f50ed60cfe3` (MEASURED, 1,182.9 s).** So the
rows are the coder's rows, not a model of them.

Ceiling family (mi1's, `ddm_mi1_indicator_model_axis_20260824.md` §5): `q′ = σ(logit(1−pmax) + β_cell)` on
the argmax indicator, one offset per context cell, Newton-fit, 2-fold cross-fitted; live positions =
`pmax < 1.0` in float32 (saturated positions cost 0 for any finite offset). Two splits: PAIR-level
(the decision split; 300/300 pairs, seed 20260824; three seeds on the load-bearing cells) and
position-level (mi1's split, for comparability). Plus a 5-way log-linear TILT generalisation
`q′(c) ∝ p(c)·exp(β_cell,c)` (ridge 1e-3, damped Newton) — the natural family for a NEW categorical
plane, since a plane can move mass toward a specific class, not only recalibrate the argmax. Plus the
charter's literal (b): a bare KT-smoothed categorical of field_t with contexts {coloc} vs {coloc, mc}
over all positions of pairs 2..599 (no coder) — reported, but it is a different baseline and cannot
be compared to the 113,411 B stream.

Nesting controls (identical across models, as they must be): indicator `none` +1.66 B, `coloc` +2.25 B,
`coloc_x_arg` +2.12 B held-out; tilt `none` +2.07 B, `coloc_x_arg` −2.09 B. The instrument's noise
floor on this body is ~2 B.

## 3. Alignment — every derivable plane is WORSE aligned than co-located (MEASURED, all 598 MC pairs)

Mean per-pair IoU against field_t, and agreement on the edge band of field_t (radius 3, 14,160
positions/pair on average):

| plane | Lane | Movable | Road | MyCar | Undrivable | band agreement |
|---|---:|---:|---:|---:|---:|---:|
| **co-located (shipped)** | **0.2495** | 0.8431 | **0.9522** | **0.9931** | **0.9942** | **0.8630** |
| `shift` | 0.2457 | 0.8369 | 0.9508 | 0.9922 | 0.9941 | 0.8580 |
| `zoom` | 0.2423 | 0.8382 | 0.9499 | 0.9914 | 0.9940 | 0.8545 |
| `planar` | 0.2429 | 0.8378 | 0.9499 | 0.9914 | 0.9940 | 0.8544 |
| `block` | 0.2295 | 0.8494 | 0.9456 | 0.9889 | 0.9934 | 0.8402 |
| `block_gated` | 0.2460 | **0.8510** | 0.9498 | 0.9911 | 0.9939 | 0.8554 |
| `block_median3` | 0.2180 | 0.8547 | 0.9482 | 0.9917 | 0.9937 | 0.8480 |

The only gains are on Movable (cars: coherent local motion) — 1.24 % of the area. On Lane (33.56 % of
the model bits) every rung LOSES alignment. The `shift` estimator returned identity on 42.6 % of pairs;
on the 343 non-identity pairs Lane IoU moved −0.0067 and Road −0.0025 — a 1-px global shift is already
noise on this field.

## 4. Why — the ORACLE separates "bad extrapolation" from "not rigid motion" (DIAGNOSTIC, reads field_t)

Estimating the SAME families on field_{t−1} → field_t (not decoder-derivable; never a candidate):

| oracle family | Lane | Movable | Road | band agreement | identity fraction |
|---|---:|---:|---:|---:|---:|
| co-located | 0.2496 | 0.8431 | 0.9522 | 0.8630 | — |
| `shift` | 0.2678 | 0.8471 | 0.9538 | 0.8689 | 0.426 |
| `zoom` | 0.2727 | 0.8514 | 0.9551 | 0.8732 | 0.237 |
| `planar` | 0.2742 | 0.8510 | 0.9552 | 0.8736 | 0.132 |
| `block` | **0.3240** | **0.8899** | **0.9661** | **0.9124** | 0.000 |

Two facts fall out. (a) Even with perfect knowledge, rigid and ground-plane motion buys ≤ +0.025 Lane
IoU: the inter-pair change of this argmax field is mostly NOT rigid motion (dash birth/death, argmax
flicker at boundaries). (b) Local block motion IS real (+0.074 Lane IoU, +0.049 band) — but it is not
predictable from the past. Comparing each pair's derivable block shift (t−2→t−1) with the oracle shift
(t−1→t), per block-row (MEASURED, 598 pairs × 8 blocks):

| block-row (64 px) | mean |shift| px | exact match | corr dy | corr dx |
|---|---:|---:|---:|---:|
| 0–1 (sky/Undrivable) | 0.00 | 1.000 | — | — |
| 2 (horizon/road far) | 2.04 | 0.395 | 0.086 | 0.205 |
| 3 (road mid) | 5.97 | 0.083 | 0.274 | 0.178 |
| 4 (road near) | 4.46 | 0.031 | −0.345 | −0.111 |
| 5 (hood/MyCar) | 0.00 | 1.000 | — | — |

The estimator behaves physically (motion 2–6 px only in the road rows, zero on sky and hood), yet one
transition's block shift is nearly uncorrelated with the next. The gated and median-3 rungs were built
to test exactly this and did not cross co-located either (§3). Carrying the oracle block motion instead
would cost **9,861 B** (zeroth-order entropy of the oracle shifts, 16.5 B/pair; DERIVED from the
persisted oracle parameters) — already twice the prior-law's whole predicted saving.

## 5. The ceiling — what the coder's rows say (MEASURED; REFUSAL-ONLY)

Live positions 50,009,121 (saturated 67,955,679, zero saturated flips — mi1/df1's counts, re-derived);
base indicator 110,909.01 B (97.8 % of the 113,411 B stream — hc1's "one binary question"); base full
row 113,410.84 B. Pair-level two-fold, seed 20260824; the three load-bearing indicator cells also ran
seeds 777 and 31337 and the table shows the **minimum**. Tilt cells ran one seed. Held-out bytes saved
on the coder's own rows; positive = the plane would save bytes.

| plane | Lane IoU | band | ind `mc` | ind `agree` | ind `mc_x_arg` (min 3 seeds) | ind `mc_x_coloc_x_arg` (min 3) | ind `mc_x_arg_x_bd` (min 3) | ind 625-cell | tilt `mc_x_arg` | tilt `mc_x_coloc_x_arg` | tilt 625-cell | best derivable cell | bare Δ (11× baseline) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `shift` | 0.2457 | 0.8580 | +5.01 | +3.74 | -124.84 | -38.41 | -149.37 | -391.71 | -9.68 | -23.01 | -139.17 | ind:mc **+5.01 B** | 59,620 |
| `zoom` | 0.2423 | 0.8545 | +5.20 | +3.06 | +13.77 | +3.03 | -18.06 | -321.33 | +24.66 | -9.34 | -133.08 | tilt:mc_x_arg **+24.66 B** | 74,346 |
| `planar` | 0.2429 | 0.8544 | +5.79 | +3.89 | +16.66 | -13.18 | -17.56 | -91.26 | +32.78 | +2.99 | -78.58 | tilt:mc_x_arg **+32.78 B** | 70,901 |
| `block` | 0.2295 | 0.8402 | +17.02 | +3.56 | +138.49 | -35.08 | +91.07 | -51.53 | +153.72 | +159.60 | +4.74 | tilt:mc_x_coloc_x_arg **+159.60 B** | 143,323 |
| `block_gated` | 0.2460 | 0.8554 | +5.76 | +2.89 | +0.89 | -279.57 | -68.02 | -213.89 | +12.90 | -4.53 | -159.12 | tilt:mc_x_arg **+12.90 B** | 56,755 |
| `block_median3` | 0.2180 | 0.8480 | +8.81 | +3.74 | +80.39 | -71.85 | +0.95 | -37.92 | +92.70 | +109.14 | +14.79 | tilt:mc_x_coloc_x_arg **+109.14 B** | 117,793 |
| `oracle_block` (DIAGNOSTIC) | 0.3240 | 0.9124 | +62.40 | +18.04 | +2666.66 | +3109.82 | +2904.42 | +3224.79 | +2826.79 | +3378.95 | +3420.35 | tilt 625-cell **+3,420.35 B** | 460,216 |

Position-level split (mi1's geometry), best MC cell per plane: shift +13.66, zoom +17.41, planar
+21.72, block +166.20, block_gated +19.56, block_median3 +107.94, oracle_block +3,379.51 B — the two
split geometries agree in magnitude, so the pair-level verdict is not a fold artefact.

What the table says:
- **Every decoder-derivable plane is at or below +159.60 B** — 31× under the 5,000 B bar and 0.14 % of
  the stream. The rigid families sit at the instrument's floor (≤ +33 B). The block plane's value is
  not alignment (it is the WORST-aligned derivable plane on Lane) but the information in its
  DISAGREEMENT with co-located on the road/Movable edges — and it is worth ~150 B.
- **The oracle-mirage signature reproduces on every plane**: the 625-cell rows are negative held-out
  for five of six derivable planes while in-sample climbs to 200–540 B (receipts). Anyone quoting an
  in-sample number for this axis is quoting noise.
- **The ORACLE plane — perfect t−1→t block motion, not decoder-derivable — conditions the coder for
  +3,420.35 B at most** (tilt 625-cell; +3,109.82 B on the 3-seed indicator minimum). That is the
  most any member of this family can deliver through this instrument, and it is below the bar AND
  below its own carriage (9,861 B): net −6,441 B at best. The carried-motion road is closed on
  arithmetic, not on taste.
- **The bare categorical (charter (b) literal)** reports 56,755–143,323 B "saved" by adding MC — against
  a 1,230,327 B baseline (10.8× the shipped stream), so it is a different object and never a rate
  claim. It does carry one corroborating fact: `ctx mc` ALONE is worse than `ctx coloc` ALONE for every
  derivable plane (1,268,748–1,400,764 B vs 1,230,327 B) and better only for the oracle (845,160 B) —
  the alignment result restated in a second instrument.

## 6. Decision rule (pre-registered, charter §4) and the prior-law prediction

- Step 1 refuse rule: best ideal saving < 5,000 B → **CEILING-REFUSED**. Best: `block`, tilt
  `mc_x_coloc_x_arg`, **+159.60 B** held-out (31× under the bar; the 3-seed indicator minimum is
  +138.49 B on `mc_x_arg`). Verdict receipt:
  `/Volumes/APDataStore/pact/ddm_mc1_motion_compensated_previous_plane/ceiling/CEILING_RESULT.json`.
- Steps 2–4: NOT RUN (retrain, RC64 exact price, fire/ready order). Nothing to fire; no
  READY-FOR-T4 candidate exists from this arm.
- Prior-law prediction (dc1 mechanism: ≥ 5,000 B saved at ≤ +1,500 B model): **FALSIFIED** on this
  body. Per the charter, the MC-input door is closed at **FORMULATION scope for the shipped receptive
  field**, and the alignment + oracle evidence (§3–4) says the closure is wider than the formulation:
  on this field the previous plane's information beyond co-located is not a motion-alignment
  problem the mixer could learn to read — the misalignment is unpredictable from the past.

verdict_scope: **formulation** (decoder-derivable constant-velocity / gated / median-3 integer motion,
six families, on the fs2 body's coder) with a measured **diagnostic bound on the family**: the oracle
plane's ceiling (§5, `oracle_block`) is what ANY carried or learned motion could at most deliver
through this instrument, and its carriage alone costs 9,861 B.

## 7. RECALL EVIDENCE (consumed, not re-derived)

- `ddm_xi1_carried_xi_inter_race_20260729.md`: warp context on a count-based coder +12,262 B (context
  dilution). Here the plane fed a recalibration of the TRAINED coder's rows and still returned ≤ 33 B.
- `ddm_d3b_lossless_lane_factorization_20260826.md`: from-zero online mixers with temporal contexts
  358,520 B. Not repeated; the trained rows were used directly.
- `ddm_dds1_decoder_derivable_verdict_20260901.md`: co-located previous-state buckets 0.88–4.13 % of the
  wrong-half gain; its n120 seeded-random screen discipline is why the ceiling here runs on all 600
  pairs with pair-level folds, never a prefix ([[m88]]).
- `ddm_mi1_indicator_model_axis_20260824.md`: the instrument (β-per-cell on the indicator, 2-fold,
  saturated excluded), its 2,162 B whole-conditioning target at z=+11.9, and the oracle-mirage
  signature (in-sample climbs, held-out collapses) — reproduced here on every rich cell.
- `ddm_dc1_decode_budget_conditional_coding_20260816.md`: 21-tap oracle floor 144,167 B; "only
  affordable as learned weights" — the mechanism this charter tested with a NEW input; the input
  carries no aligned information the current-frame causal context does not already hold.
- `ddm_gb1_groupbin8_verdict_20260824.md`: −153 B lossless from decode-scan conditioning; the scale a
  real conditioning gain looks like on this body.
- Naming note: an unrelated 2026-07-24 arm also used the tag `ddm_mc1`
  (`ddm_mc1_hood_static_reassert`); this memo's file name carries the full object to keep them apart.

## 8. Custody (ALWAYS KEEP THE PAYLOAD)

| artifact | path | bytes / sha |
|---|---|---|
| coding rows (float32, 600×196,608×5) | `/Volumes/VertigoDataTier/pact/ddm_mc1_motion_compensated_previous_plane/rows/coding_rows.f32.npy` | see `ROWS_RESULT.json` |
| base argmax, boundary bucket (u8) | same dir, `base_argmax.u8.npy`, `boundary_bucket.u8.npy` | see `ROWS_RESULT.json` |
| control stream (byte-identical to shipped) | `/Volumes/APDataStore/pact/ddm_mc1_motion_compensated_previous_plane/rows/control_stream.bin` | 113,411 B, `5601d6fd…` |
| MC planes, 6 derivable + `oracle_block` (u8, 600×384×512) | `/Volumes/VertigoDataTier/pact/ddm_mc1_motion_compensated_previous_plane/motion/mc_plane_*.u8.npy` | 117,964,800 B each |
| motion parameters per pair (json) | `/Volumes/APDataStore/pact/ddm_mc1_motion_compensated_previous_plane/motion/params_*.json`, `oracle_params_block.json` | — |
| alignment receipts | same dir, `MOTION_*.json`, `ORACLE_*.json`, `iou_*.npy` | — |
| ceiling receipts | `/Volumes/APDataStore/pact/ddm_mc1_motion_compensated_previous_plane/ceiling/CEILING_*.json`, `CEILING_RESULT.json` | — |
| receiver copy of the fs2 tree (untouched; the MC-aware decode was never needed) | `/Volumes/VertigoDataTier/pact/ddm_mc1_motion_compensated_previous_plane/receiver_mc/` | — |
| code | `experiments/ddm_mc1_motion_compensated_previous_plane.py` (stages rows / motion / oracle / ceiling / verdict) | committed |

Reproduce: `.venv/bin/python experiments/ddm_mc1_motion_compensated_previous_plane.py --stage {rows,motion,oracle,ceiling,verdict} [--models …]`.

## 9. What I did NOT do (plainly)

- No retrain, no MC-aware receiver, no RC64 exact price, no archive, no T4 row: the pre-registered
  ceiling refused first, as it was written to.
- No scorer run, no Modal, no Metal.
- The receiver copy under `receiver_mc/` is an untouched copy of the fs2 tree; no MC decode was wired.
- The ceiling ran on the fs2 body's coder (the shipped one); it was NOT run on a retrained model
  (that is step 2, refused).
- Operational note: the 23:47Z rc=143 on my first ceiling launch was my own `kill` (I stopped it to
  fix a wrong in-sample telemetry formula and relaunched); the compressor ramp the watchdog saw was
  three `load_live` phases coinciding, since cured by a two-pass preallocation. From ~00:05Z the
  fleet's `vertigo_certify_move.py` jobs saturated `/Volumes/APDataStore` (ExFAT/FSKit) and my ceiling
  children sat in uninterruptible I/O wait on their receipt/log writes; they completed when the volume
  freed. Nothing was lost; the wall-clock was.

## 10. LIVE-HYPOTHESES / DEAD-ENDS / NEXT_IF_RESUMED

**DEAD-ENDS (with numbers):**
- Decoder-derivable MC plane, any of six integer families: worse-aligned than co-located on Lane/Road
  (§3); coder ceiling ≤ +159.60 B (§5). CLOSED at formulation scope.
- Carried block motion: carriage 9,861 B (§4) vs an oracle-plane ceiling of +3,420.35 B (§5).
  CLOSED on arithmetic.
- Rigid/ground-plane oracle alignment ≤ +0.025 Lane IoU: the field's inter-pair change is not rigid.

**LIVE-HYPOTHESES (not tested here; none is this charter's object):**
- Temporal information beyond co-located, if any, is in dash birth/death and boundary flicker, which
  are current-frame-causal phenomena the 7×7 masked context already sees; a temporal RUN-LENGTH or
  age plane (how many pairs a position has held its class) is a different, cheap plane — mi1's
  `run8` control measured +15 B, so it is likely also drained.
- Movable is the one class where local motion aligns (+0.006 IoU derivable, +0.047 oracle); it carries
  little of the stream. Not a rate lever at this scale.

**NEXT_IF_RESUMED:** nothing on this door. The demand (−41,817.8 B) is elsewhere; see ddm_x012's door map.

## Equations leg (`tac.canonical_equations`)

Registered as `motion_compensated_previous_plane_alignment_gate_v1`
(`src/tac/canonical_equations/motion_compensated_previous_plane_gate_20260904.py`, exported from
`tac.canonical_equations`; re-derivation guards in
`src/tac/tests/test_ddm_mc1_motion_compensated_previous_plane_gate.py`). The law: a temporal-context
plane is admissible for a coder ceiling only if it beats the co-located plane on the bit-carrying
class's IoU AND on edge-band agreement (`plane_passes_alignment_gate`); a plane failing that gate
prices at the instrument floor (`ceiling_refused`); a carried motion plane additionally owes its
carriage against the ORACLE plane's ceiling (`carried_motion_breakeven_open`); constant-velocity
extrapolation of block motion needs a transition-to-transition correlation ≥ 0.5
(`temporal_predictability_supports_extrapolation`). Three empirical anchors, all
VERIFIED_VIA_EMPIRICAL_ANCHOR: alignment (residual 0.0035 IoU), oracle + block consistency (0.274),
coder ceiling (residual 5,000 − 159.60 = 4,840.40 B = the prior-law gap). Predicted vs empirical:
predicted ≥ 5,000 B, empirical +159.60 B (derivable) / +3,420.35 B (oracle).

## Frontier line

fs2 S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600] — unchanged. Candidate line: none (CEILING-REFUSED; advisory only, nothing READY-FOR-T4).
