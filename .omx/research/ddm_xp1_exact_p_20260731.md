# ddm_xp1 — EXACT P on the rung-1 birth ENDPOINT (task #806, MAIN-owed)

**Model: claude-opus-4-8.** The MAIN-owed exact-P measurement named verbatim in the rung-1 endpoint
manifest (`/Volumes/VertigoDataTier/pact/ddm_r1c_20260731/rung1_endpoint_manifest.json`,
`p_remeasure.exact_owed_to_main`): *"QA92 base-pass method on this endpoint render
(experiments/ddm_qa92_carrier_discriminator.py) -> P in S units"* — curing the manifest's own 4-conn
`above_nucleus_erased_estimate = 474` DERIVED-ESTIMATE with the exact 8-conn scorer measurement.

**Pointer honesty FIRST: 0.1910828242 [contest-CPU] UNMOVED.** Every number below is
**[macOS-CPU advisory]**, `score_claim=false`, `research_only`. This unit is the free scorer slot — a $0
measurement, NOT a score mover. No exact row moves here.

## ANSWER (lead)

**P = 0.04401 S** — the exact erased super-nucleus Lane pool remaining at the ep641 birth endpoint (8-conn,
frozen CPU-torch SegNet, n600). The decisive, genuinely-unpredicted read: **140 epochs of birth-arm
continuation (ep499 control_tail parent → ep641 endpoint) LOWERED overall d_seg by 0.000677 but did NOT
recover the erased Lane pool — the pool GREW +0.00212 S (0.04189 → 0.04401) and gained +187 erased
super-nucleus components (4041 → 4228).** The overall d_seg win came entirely from Road (−0.0336 S),
Movable (−0.01778 S) and MyCar (−0.01987 S); the Lane axis — the birth arm's raison d'être — got slightly
WORSE (+0.00151 S). This is a SECOND, orthogonal (frozen-scorer, not tr1-telemetry) confirmation of the
birth gate's own stall signal: the endpoint is a Lane-**stall**, not a Lane-birth-success. Burn-4's
KD-from-birth arm therefore targets a pool that is (a) LARGER than QA92's parent snapshot assumed, and (b)
has NOT moved under the birth arm's own continuation physics.

| quantity (n600, 8-conn, frozen SegNet) | QA92 @ ep499 (parent) | **XP1 @ ep641 (endpoint)** | Δ (endpoint − parent) |
|---|---|---|---|
| ckpt sha256 | a2dc86b8… | **40553db8…** | (child = 140-epoch continuation of parent) |
| base d_seg (mean) | 0.0049411 | **0.004264052** | −0.000677 (better overall) |
| **P = erased super-nucleus Lane pool (S-units)** | 0.04189 | **0.04401** | **+0.00212 (WORSE — pool grew)** |
| n erased super-nucleus (of 9035 GT) | 4041 | **4228** | +187 erased |
| n GT super-nucleus (invariant) | 9035 | **9035** | 0 ✓ (GT is fixed) |
| realized Lane super-nucleus (scorer basis) | — | **6023** | (see §3 basis note) |
| target px total (erased support) | 58761 | **61675** | +2914 px |
| base Lane S-units | 0.12438 | **0.12589** | +0.00151 (worse) |
| base per-class S [Road,Lane,Undriv,Mov,MyCar] | [0.222,0.124,0.054,0.056,0.038] | **[0.188,0.126,0.056,0.038,0.018]** | [−0.034,+0.002,+0.002,−0.018,−0.020] |

Validation: `base_dseg_mean 0.004264052` reproduces the manifest `n600_d_seg 0.00426407708` to ~1e-7 — the
render → deploy-R → uint8 → frozen-SegNet chain is exact (same discipline QA92 used: base d_seg == parent
endpoint). Per-class S sums to the seg term (0.4264 = 100·d_seg) exactly, both endpoints.

## §1 The measurement (all [macOS-CPU advisory]; frozen CPU-torch SegNet = authority, NEVER MPS)

`experiments/ddm_xp1_exact_p.py` (3e588562d1…) is a **small driver that REUSES QA92's exact base/P
computation path** — it imports and calls QA92's `erased_super_nucleus_mask` (scipy 8-conn on GT Lane
`lstars==1`, super-nucleus >5px, a component *erased* iff <50% of its GT-Lane pixels are classified Lane in
the base pass) and its P formula verbatim (`P = 100·Σ base_flip_in_target / total_px`, QA92 `aggregate()`
L353). It runs the **base pass ONLY** — no oracle/flat paint tiers (those were QA92's *discriminator*; MAIN
owes only P) — so it needs only `lstars` (no gt_f1), is memory-light, and finished n600 in **133 s**.
Deploy surfaces reused verbatim: `_torch_R_to_camera_uint8` + `cpu_verdict_d_seg_argmax_batch`
(`train_witness_realized_through_R_mlx`), `render_frame`/`realized_gate` (tr1), `load_frozen_module`
(fp1). Seeded, resumable-per-chunk (chunk 120 / seg_batch 12), atomic writes. The ONLY new logic beyond
QA92's base path is the realized-Lane super-nucleus component count (§3), which the base-pass P does not
itself yield.

## §2 What GREW vs SHRANK (decompose the headline — the birth continuation's win is NOT Lane)

The net d_seg improvement of −0.000677 over the 140-epoch continuation decomposes exactly (per-class ΔS):

```
Road   −0.0336   ┐
Movable−0.01778  ├─ the entire overall win (−0.0692 across these three)
MyCar  −0.01987  ┘
Lane   +0.00151  ┐
Undriv +0.00204  ┘─ +0.00355 REGRESSION (Lane + sky/undrivable)
--------------------------------------------------
net    −0.0677 / 100 = −0.000677  ✓ (matches base_dseg Δ exactly)
```

The birth arm (r1c window_01) was launched to birth Lane structure, yet at its endpoint the Lane axis
regressed and the erased Lane super-nucleus pool grew. The overall d_seg gain is real but is a
Road/Movable/MyCar phenomenon — NOT Lane birth. This is the "d_seg improved but the target axis didn't"
decomposition; the composite headline (−0.000677) would have hidden it.

## §3 Basis reconciliation — the manifest's 474, the birth gate's 985/500, and my exact 4228

Three DIFFERENT bases, stated explicitly (the obligation demanded it):

1. **Manifest DERIVED-ESTIMATE `above_nucleus_erased_estimate = 474`** = `round(erased_count 485 ×
   super_nucleus_area_frac 0.9767)`, where `erased_count 485 = betti0_gt_lane 985 −
   betti0_realized_endpoint 500`. The manifest itself flagged this "out of this $0 producer's scope; EXACT
   super-nucleus-erased COUNT needs one scorer pass (QA91 method_note)". **XP1 IS that scorer pass.**
2. **Birth-gate telemetry basis** (`birth_completion_gate.json`, `evidence_axis: apparatus — telemetry-only,
   no scorer`): `betti0_gt_lane 985`, `betti0_realized_endpoint 500`, window [534,539,531,513,500] over
   ep[624,629,634,639,640]. These are the **tr1 trainer's own realized_gate class-field counts** (NO frozen
   SegNet; the tr1 config runs `grid_downsample 16`, so this betti0 is a coarse-grid/probe metric), which
   is why 985/500 are ~9× smaller than the full-res scorer counts below. NOT comparable to XP1's numbers —
   they are a different instrument on a different resolution.
3. **XP1 exact scorer basis** (frozen CPU-torch SegNet argmax, full 384×512, 8-conn, n600 aggregate):
   **n_super 9035, realized_super 6023, n_erased 4228, P = 0.04401 S.** This is the exact quantity the
   manifest owed. Do NOT read 4228 as "replacing" 474 same-basis — they are different instruments; 4228 is
   the exact scorer-basis erased super-nucleus count, 474 was a coarse-telemetry proxy.

**Orthogonal-basis AGREEMENT on direction:** the birth gate's telemetry betti0 was DECREASING at the
endpoint (534→500, moving away from GT 985; the gate's own OLS slope −10.3±3.7 comp/gate, |t|=2.81,
one-sided α=0.159 — no significant positive birth), and it "fired birth_completion" on slope-FLATTENING,
NOT pool exhaustion. XP1's independent frozen-scorer basis (pool +0.00212 S, +187 erased) CORROBORATES: two
different instruments agree the endpoint is a Lane **stall/slight-regression**, not a completed birth.

## §4 The burn-4 accounting line (what pool remains for KD-from-birth to chase)

**The realistically-attackable Lane pool at the ep641 birth endpoint is P ≤ 0.04401 S, and NO measured
mechanism has moved it:**

- **Continuation physics has STALLED on it.** The 140-epoch birth continuation that produced this endpoint
  did not shrink P — it grew it (+0.00212 S). The birth-completion stop was a stall-detector firing, not a
  recovery. KD-from-birth must do what the plain birth continuation demonstrably could not.
- **Paint-on-texture is net-negative on it (QA92, measured at the parent).** Even the perfect GT-RGB oracle
  recovers only O=0.41 of the pool (P·O = 0.017 S target) while injecting +0.317 S off-target ERF
  collateral → JOINT +0.30 S WORSE. Output-side compositing is dead; burn-4's birth arm defaults to
  KD-from-birth / re-render, not paint (QA92 §3.2).
- **Net:** KD-from-birth's target pool is 0.044 S (slightly BIGGER than QA92's ep499 snapshot 0.042
  assumed), the birth arm's own continuation physics has plateaued/regressed against it, and paint cannot
  cross the receiver-ERF wall. So burn-4 chases an **unrecovered 0.044 S pool with no demonstrated crossing
  mechanism** — a harder starting position than the parent snapshot implied. This is a direct input to the
  burn-4 charter's birth-arm priority; b4s consumes this P line at fire time.

## §5 verdict_scope ledger

- P = 0.04401 S erased super-nucleus Lane pool @ ep641: **MEASURED** (n600, frozen CPU-torch SegNet, exact
  scipy 8-conn, QA92 method reused verbatim). Base d_seg 0.004264052 == manifest n600_d_seg to ~1e-7.
- n_erased 4228 / n_super 9035 (8-conn) / realized_super 6023: **MEASURED**.
- Pool-grew (+0.00212 S) + Lane-axis-regressed (+0.00151 S) over the continuation: **MEASURED** (endpoint −
  parent, both n600 frozen-scorer). Directional, small absolute; the two-instrument agreement (§3) raises
  confidence it is signal not noise.
- Burn-4 accounting (P ≤ 0.044 S, continuation stalled, paint net-negative): **MECHANICAL** read from the
  measured P + QA92's measured paint-negativity + the gate's recorded stall fit. Advisory input to burn-4;
  no score claim.
- verdict_scope: this is a MEASUREMENT, not a family verdict. It re-prices burn-4's birth-arm inputs; it
  does not close or open any carrier family.

## §6 STORES-CONSULTED (recall-first, path+sha)

rung-1 endpoint manifest `/Volumes/VertigoDataTier/pact/ddm_r1c_20260731/rung1_endpoint_manifest.json`
(the obligation + ep641 ckpt sha 40553db8… + n600_d_seg 0.00426407708 + parent baseline 0.0049411) ·
birth gate `/Volumes/VertigoDataTier/pact/ddm_r1c_20260731/window_01/birth_completion_gate.json`
(betti0_gt_lane 985 / realized 500 / erased 485 / area_frac 0.9767 / slope fit; telemetry basis) ·
tr1 config `.../window_01/tr1_config.json` (grid_downsample 16, w_seg 100, variant lotto, epochs 641) ·
QA92 harness `experiments/ddm_qa92_carrier_discriminator.py` (b6920be3d2…; reused `erased_super_nucleus_mask`
+ P L353 + `_per_class_flip_counts` + `load_module`) + verdict
`/Volumes/VertigoDataTier/pact/ddm_qa92_20260731/qa92_verdict.json` (ep499 P 0.04189, 4041/9035, base
0.0049411, per-class, O/F/collateral paint-negative) + memo
`.omx/research/ddm_qa92_carrier_discriminator_20260731.md` (6c23883be6) · fp1/QA91
`.omx/research/ddm_fp1_class_field_projection_20260731.md` (8-conn method_note, super-nucleus inventory,
receiver-is-the-wall) · deploy surfaces `experiments/train_witness_realized_through_R_mlx.py`
(`_torch_R_to_camera_uint8`, `cpu_verdict_d_seg_argmax_batch`) + `experiments/ddm_fp1_class_field_projection.py`
(`load_frozen_module`) · gt cache `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` (cf8d83605d…) ·
frozen-scorer laws (SegNet reads REGIONS, comma10k class order [Road,Lane,Undriv,Movable,MyCar] Lane=1,
ERF r50~85px). Closed forks re-checked, none reopened: paint (b)/(d) stays QA92-net-negative; this is the
scorer-basis P the birth endpoint owed, distinct from the telemetry betti0.

## §7 custody
`/Volumes/VertigoDataTier/pact/ddm_xp1_20260731/`: `xp1_verdict.json` + 5× `chunk_*.npz` (per-pair
accumulators) + `xp1_custody_manifest.json` (ckpt/gt/driver/reused-harness sha256 + deterministic rebuild
command) + `run.log`. Certified-rebuildable (frozen ckpt + gt cache + seeded driver, no RNG; ~133 s n600).
No `/tmp` in evidence.

**Task #806 completion evidence lives here + FEED-xp1. Pointer 0.1910828242 [contest-CPU] UNMOVED.**
[no-triality] [p0-ledger-ok]
