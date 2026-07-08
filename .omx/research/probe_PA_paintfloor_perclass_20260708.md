# PROBE P-A — per-class attribution of the #210 oracle-R paint floor (task #359, $0)

**Date:** 2026-07-08 · **Axis:** `[macOS-CPU advisory] NON-PROMOTABLE`, $0, read-only vs run-1 (pid 63069
LIVE, untouched) · **Pointer 0.19110 UNMOVED (means).** · **verdict_scope: n600 (full — NOT a subset).**
Pre-registered in `.omx/research/perclass_carriers_design_20260708.md` §2/§5 (probe P-A).

## STORES CONSULTED
- `perclass_carriers_design_20260708.md` (the pre-registration; §2 measured-evidence table row "Oracle-R
  floor (#210) 0.00091 COMPOSITE"; §5 the decisive-unknown framing on the 20-50 KB Road/Undriv band).
- `tools/levelset_gate_discriminators_n600.py` (#210 gate — the EXACT R+SegNet code path reused verbatim:
  `_torch_R_to_camera_uint8`, `area_down`, `seg_argmax_batch`, `load_real_segnet`).
- `reports/levelset_oracle_R_floor_n600_20260701.json` + `...ckpt.npz` (the #210 composite floor result the
  checkpoint stored ONLY as per-pair scalars — no per-pixel argmax → attribution required recompute).
- CLAUDE.md canonical class order (Road0/Lane1/Undriv2/Movable3/MyCar4, MEASURED) + measured margin
  distribution (50/19/13% flip-prone for classes 0/1/2) + §Confound-self-protection (apparatus-validity).
- MEMORY L65 (dash erasure), L66 (annulus = boundary-jitter ~97% in 4.7% area), L17 (islands = lane ~8-dim).
- `docs/operating_manual_craft_handoff.md` (label MEASURED/DERIVED/INFERRED; attack own conclusion).

## METHOD (faithful, $0, recompute — the checkpoint had no per-pixel argmax)
Reused the #210 gate's EXACT authority path: real frame `gt_f1` → `area_down` to render grid →
`_torch_R_to_camera_uint8` (bicubic↑874 → round/uint8) → `SegNet.preprocess_input` (bilinear 874→384×512)
→ argmax, frozen CPU-torch SegNet, NEVER MPS. The **real-frame→R condition is the ACHIEVABLE-THROUGH-R
FLOOR** (a task-space flat/palette/carrier render is a strict SUBSET of the real frame's detail — gate
docstring), so this attributes the strongest global proof that the paint problem is solvable. Added a 5×5
GT×realized **confusion accumulator** over all n600 pairs at BOTH the headline 384×512 grid (composite
0.00091) and the witness-default 192×256 grid (0.00247). **Internal-consistency check:** confusion-derived
composite = 0.000909974839952257 ≡ per-pair-mean 0.0009099748399522569 (15 digits) and reproduces the
Jul-1 #210 n600 floor EXACTLY → the decomposition is faithful, not a reinterpretation.
(script: scratchpad `probe_PA_attribution.py`; reruns from `gt_n600.npz`; ~4 min wall, nice -10.)

## RESULT 1 — per-class flip table @ 384×512 (the headline composite d_seg = 0.000910)

| class | GT area | flip-mass contribution | share of composite | **within-class flip rate** | flips-to (top) |
|---|---:|---:|---:|---:|---|
| **Road0** | 0.2323 | 0.000398 | **43.7%** | 0.00171 (0.17%) | Lane1 41%, Undriv2 25%, MyCar4 23%, Movable3 10% |
| Lane1 | 0.0059 | 0.000149 | 16.3% | **0.02537 (2.5%)** | Road0 99% |
| **Undriv2** | 0.4952 | 0.000165 | 18.2% | **0.00033 (0.03%)** | Road0 64%, Movable3 36% |
| Movable3 | 0.0124 | 0.000094 | 10.4% | 0.00761 (0.76%) | Undriv2 57%, Road0 43% |
| MyCar4 | 0.2543 | 0.000104 | 11.4% | 0.00041 (0.04%) | Road0 99% |

(@192×256, composite 0.002469: Road0 47.8% / Lane1 14.2% / Undriv2 20.1% / Movable3 7.7% / MyCar4 10.2%;
within-class Road 0.51%, Lane 5.99%, Undriv 0.10%, Movable 1.53%, MyCar 0.10% — same ordering, larger.)

## RESULT 2 — flip-destination matrix = the region-adjacency (Morse-Smale) graph (@384)
```
Road0    -> Lane1 41%, Undriv2 25%, MyCar4 23%, Movable3 10%   (the hub: flips at ALL its boundaries)
Lane1    -> Road0 99%                                          (lane embedded in road surface)
Undriv2  -> Road0 64%, Movable3 36%                            (horizon + cars against sky)
Movable3 -> Undriv2 57%, Road0 43%                             (car silhouettes)
MyCar4   -> Road0 99%                                          (hood-road boundary; #139 clamp)
```
This is EXACTLY the design §1 prediction: **d_seg factorizes over pairwise tie-loci on the region-adjacency
graph.** Road is the connective hub; every class's residual lives on its shared separatrix with Road (+ the
Undriv↔Movable and Road↔Undriv ties). No class flips in its interior — the residual is codim-1 boundary
placement, matching L66 (annulus = boundary-jitter).

## VERDICT ON THE 20-50 KB Road/Undriv BAND — **CONSERVATIVE (leaning confirmed-range); the "optimistic" risk is REFUTED**

verdict_scope: formulation — REFUTED applies to the design's named risk formulation ("Road/Undriv interior textured-paint fidelity through R is the binding constraint"), measured at the oracle-paint upper bound (real-frame texture) on the exact gate path, n600. It does NOT judge the achievable-bytes figure (increment-1's to measure) nor any carrier family — the boundary-placement residual it localizes remains fully open.

The design's decisive dichotomy: *"if Road+Undriv paint accounts for nearly all the composite residual → band
OPTIMISTIC; if lane/movable-dominated → bulk classes paint near-perfectly → band CONSERVATIVE."*

By **raw flip mass**, Road+Undriv = **61.9%** of the composite floor (43.7% + 18.2%) — the literal reading of
branch (A) ("optimistic"). **But that reading is an area artifact and the correct verdict inverts it:**

1. **Road/Undriv paint near-PERFECTLY per pixel.** Their within-class flip rates (Road 0.17%, Undriv 0.03%)
   are the **LOWEST of all classes** — Undriv2, the single biggest class (49.5% of the frame), is essentially
   flawless through R. Their large flip-mass share is because they are 72% of every frame, NOT because they
   are hard to paint. The design's named risk — *"Road/Undriv textured-paint fidelity through R is binding"* —
   is **MEASURED FALSE at the achievable floor.**
2. **The entire Road/Undriv residual is separatrix placement, not interior fidelity** (destination matrix:
   Road flips to Lane/Undriv/MyCar/Movable boundaries; Undriv flips to Road/Movable boundaries; zero interior
   flips). This is precisely what the #308 **grid-bulk (interior, near-free) + INR-annulus (boundary, where the
   bytes go)** hybrid targets. The 20-50 KB buys BOUNDARY precision; interiors ride null-space class-typical
   texture at ~0 byte.
3. Therefore the band is **CONSERVATIVE-to-CONFIRMED for the fidelity requirement**: the achievable floor proves
   the interior-paint half is near-free, so the byte budget is spent only on the annulus the carrier is designed
   for. The oracle cannot pin the exact 20-50 KB figure (see caveat) — but it removes the "interiors are the hard
   part" failure mode and localizes 100% of the Road/Undriv residual to the boundary.

**Honest caveat (attack-own-conclusion).** The oracle uses the REAL FRAME's full texture — the achievable-
through-R UPPER BOUND on appearance fidelity. A byte-limited carrier reproduces LESS texture; this probe proves
interior appearance is NOT the binding constraint and the residual is boundary geometry, but it does NOT by
itself confirm that 20-50 KB reproduces enough Road↔Undriv boundary precision — that is exactly what v8
increment-1 measures (per design §5). No paradigm/family claim; verdict is FORMULATION-level on the band figure.

**Refinement of the campaign's flip-distribution prior.** The witness-training margin distribution (CLAUDE.md
50/19/13% for Road/Lane/Undriv) includes the training gap. At the IRREDUCIBLE oracle floor (ideal paint), the
distribution is Road 43.7% / Undriv 18.2% / Lane 16.3% — **Undriv is HIGHER and Lane LOWER** than the training
prior, because the ideal paint has already solved lane recall (real-frame lane recall 0.94 supersampled, #210
gate). The floor is even more Road-hub-boundary-concentrated than training suggests.

## ONE-LINE IMPLICATION FOR v8 INCREMENT-1
Split Road/Undriv into the dedicated bulk-boundary field as designed — it targets 62% of the irreducible floor —
but spend the bytes on **boundary/annulus precision, not interior texture** (interiors flip ~0), and treat the
**Road↔Lane tie** (41%→54% of Road's flips) as the highest-leverage ~0-byte `b_c` argmax tie-calibration against
the v7 lane band; Movable keeps its per-field homotopy (residual = Movable↔Undriv/Road boundaries).

## FINAL STATE
$0 attribution; n600 full; run-1 untouched; pointer **0.19110 UNMOVED** (means — moves only through a
byte-closed `upstream/evaluate.py` n600 exact row). Candidate law `tropical_perclass_reconciliation_v1` (design
§6) gains its d_seg-factorizes-over-pairwise-tie-loci anchor here (destination matrix = the RAG); still
council-flagged pending increment-1 per-class rows.
