# WITNESS CAPSTONE — deep-math levers on the non-RGB task-space argmax INR (2026-06-25)

**Evidence grade:** `[macOS-MLX research-signal]` — promotion_eligible=false, score_claim=false,
ready_for_exact_eval_dispatch=false. d_seg = argmax-disagreement RATE vs the FROZEN CPU-torch SegNet
GT argmax (the exact score-native quantity), precomputed targets at `targets_n600` (NEVER MPS as
authority; MLX is the training-gradient device only). NO byte-closed contest-CPU/CUDA exact eval yet
(a LATER unit).

## Unit question
Does applying the deep-math levers DRIVE the witness d_seg DOWN from the lever-B baseline
(0.008257 over 600 pairs, hidden=96/n_hidden=4/n_fourier=16/mod=32, 85.4K params, 63.8KB blob)
toward the sub-0.15 need (~0.001)?

## Vehicle
`ImprovedSegGenerator` (MLX coord-INR): deterministic isotropic Fourier features (+ optional
boundary-proximity scalar + optional oriented/anisotropic Fourier features) -> in_proj -> n_hidden
FiLM-per-pair-modulated activation layers -> 5-class-logit head; per-pair `mod` code. Trained
scorer-only (margin-weighted CE through the frozen SegNet argmax, NO full-RGB recon). numpy-portable
forward verified ARGMAX-parity. Tool: `tools/witness_capstone_deepmath_smoke.py`; tests:
`src/tac/tests/test_witness_capstone_deepmath_smoke.py` (7, pass). Commit: `d893436ce`.

## DECISIVE structural finding (the crux refinement)
The lever-B crux framed the residual as "class-1 lane edges". MEASURED on the GT (targets_n600,
20-pair sample): the small-margin (flip-prone) pixels are **NOT** lane-dominated —
- class-0 (road): 50.0% of small-margin px
- class-1 (lane): 18.9%
- class-2 (sky/bg): 13.2%, class-4: 10.4%, class-3: 7.4%
The flips concentrate at **ALL inter-class boundaries** (a codim-1 curve = the union of every class
edge), of which the lane is a minority. So a LANE-ONLY routing/orientation prior targets the wrong
band. The decisive lever is **ALL-CLASS boundary** orientation.

## Lever sweep (MEASURED d_seg, frozen-scorer authority)

### n96 (96 pairs, 196,608 px each = 18.9M px x pairs), MLX (Apple GPU), 250-400 epochs
| arm | levers | best d_seg | params | bytes | vs control |
|---|---|---:|---:|---:|---:|
| control | none (iso Fourier only) | 0.006539 | 67K | 48.5KB | baseline |
| dir (lane-only) | oriented Fourier, class-1 | 0.006001 | 70K | 50.5KB | -8% |
| dir_cap (lane-only) | + h160/nh6 | 0.006899 | 293K | 203KB | +6% (HURTS) |
| **dir_allcls** | oriented Fourier, ALL-class | 0.003416 | 72K | 49.7KB | **-48%** |
| dirprox_allcls | + boundary-prox scalar | 0.004050 | 72K | 50.0KB | -38% |
| dir_allcls_sharp | + n_dir_freqs=8, across=48 | 0.003212 | 72K | 50.6KB | -51% |
| **dir_allcls_cap** | ALL-class dir + h128/nh5 | **0.002338** | 158K | 113KB | **-64%** |

### n32 (fast iteration), 150 epochs
control 0.005948 · dir(lane) 0.005620 (-5.5%) · cap_big 0.005959 (no help, 4x bytes) ·
prox(lane) 0.007641 (HURTS) · combo 0.008697 (HURTS).

## Lever verdicts (what helped / didn't)
1. **CAPACITY alone — NO** (with the wrong/isotropic basis it does nothing or HURTS + diverges at
   lr=3e-3; n96 dir_cap lane-only = +6%). With the CORRECT all-class directional basis, modest
   capacity (h128/nh5) DOES help (dir_allcls 0.00342 -> dir_allcls_cap 0.00234). Lever INTERACTION:
   capacity only pays once the basis is matched to the manifold. Matches the operator's
   "loss-movable not capacity-bound" finding — the basis is the loss-shaper.
2. **DIRECTIONAL / ANISOTROPIC (curvelet) Fourier — YES, the decisive lever**, but ONLY when
   oriented to the ALL-CLASS boundary (lane-only gives -8%; all-class gives -48%). The curvelet
   inductive bias (high-freq across the edge, low-freq along it) is the right prior for the codim-1
   smooth-curve partition. ~0 byte cost.
3. **BOUNDARY-PROXIMITY scalar input — NO / mild drag** on top of directional (dirprox_allcls
   0.00405 vs dir_allcls 0.00342). The orientation field already encodes the WHERE; the raw scalar
   competes with the iso features. Dropped from the winner.
4. **SHARPER across-freq** — small additional help (-3% over plain all-class dir).

## Best config + projected S-trajectory (n96, advisory)
- **dir_allcls (cheap):** d_seg 0.00342, ~50KB -> 100*d_seg = 0.342; 25*bytes/N: 25*50e3/37.5e6 = 0.033.
- **dir_allcls_cap:** d_seg 0.00234, ~113KB -> 100*d_seg = 0.234; 25*bytes/N = 0.075.
NOTE: these are n96 numbers; the n600 plateau regime is harder (more pairs share the base). The
decisive n600 measurement is in flight (daemons `witcap_n600_dirallcls_cap`, `witcap_n600_dirallcls`,
`witcap_n600_control`). The lever-B baseline at n600 was 0.008257 (100*d_seg = 0.826) — the d_seg
term ALONE must fall ~8x (to ~0.001) for sub-0.15; the levers cut it ~half-to-two-thirds so far.

## Honest gap to ~0.001
Best MEASURED so far is 0.00234 (n96) — still ~2.3x above the ~0.001 sub-0.15 need, and that is at
the EASIER n96 regime; n600 will be higher. The directional-all-class lever is real and large
(-48% to -64%) but does NOT yet close the gap alone. Remaining headroom to probe: per-pair capacity
(bigger mod codes), multi-scale all-class basis, a true step/curvelet activation (gauss tried in
code, not yet swept), and longer training (arms were still descending). The byte cost of the
capacity bump (50KB->113KB) trades against the d_seg gain — the cheap all-class-dir (50KB, -48%) is
the better S point unless n600 shows the capacity gain holds.

## n600 DECISIVE confirmation (full plateau regime, MEASURED, frozen-scorer authority)
The levers TRANSFER to the real 600-pair plateau regime where the lever-B baseline plateaued at
0.008257. Daemons `witcap_n600_*` (durable, survive the session); readout while still descending:

| n600 arm | levers | d_seg (ep) | params | bytes | vs lever-B 0.008257 |
|---|---|---:|---:|---:|---:|
| control | iso only | 0.007828 (ep350, plateaued) | 86K | ~64KB | -5% |
| dirallcls | ALL-class dir | 0.005697 (ep100, descending) | ~90K | ~50KB | **-31%** |
| dirallcls_cap | ALL-class dir + h128/nh5 | 0.004445 (ep100, descending) | ~178K | ~113KB | **-46%** |

dir+cap trajectory: ep50=0.0052 -> ep100=0.0044 (still dropping; daemons run to ep450). At ep100 it
already beats the lever-B baseline by 46% and the iso control by 43%. CONFIRMS the n96 finding at
full scale. NOTE: still `[macOS-MLX research-signal]`, NO byte-closed exact eval yet.

## Honest gap to ~0.001 (UPDATED with n600)
Best n600 MEASURED so far = 0.004445 (ep100, dir+cap, still descending). The sub-0.15 d_seg need is
~0.001 (100*d_seg ~ 0.1). We are ~4.4x above it and the curve is still moving — final n600 (ep450)
will be lower but likely lands ~0.003-0.004 absent a further lever. The all-class directional lever
is large and real but does NOT close the gap alone; the byte side is healthy (50-113KB << HNeRV
177KB). Remaining headroom: larger per-pair mod, multi-scale all-class basis, true step/curvelet
activation (code has `gauss`, unswept), longer training, and a residual sidecar on the witness
argmax (the orphaned closed_spec_boundary_solver, now that the residual is more contiguous).

## Borrowed-substrate accounting
- BORROWED: scipy EDT (Felzenszwalb-Huttenlocher/Maurer); isotropic random Fourier features
  (Tancik 2020); anisotropic/oriented generalization (AFPE Kuckelhaus 2025; steerable filters
  Freeman-Adelson 1991; curvelets Candes-Donoho 2004); FiLM (Perez 2018); HNeRV amortized decoder.
- OURS: the ALL-CLASS-boundary distance/tangent as the orientation field for a small NON-RGB argmax
  witness INR to descend the contest's binding d_seg; the codec-free (0-byte, train-time-only)
  prior; the per-pair oriented PE; the measured finding that flips are all-class (not lane) and that
  capacity only pays after the basis is matched.
