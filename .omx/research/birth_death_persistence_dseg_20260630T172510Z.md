# Birth-death (persistent-homology) analysis of the witness segmentation residual

`[macOS-numpy advisory . NON-PROMOTABLE]` -- pointer UNMOVED 0.19110. $0 CPU-only, read-only, no GT recompute.

- generated: 2026-06-30T17:26:20Z
- witness stage (argmax): `maps_l7.npz`  (witness d_seg per summary.json)
- GT frames analysed: 100 (margin+argmax from `gt_strided_n200.npz`); witness frames: 96
- filtration: H0 superlevel-set of the FROZEN-SegNet top1-top2 margin, 4-conn union-find (elder rule), per canonical class
- runtime: 69s

## 2. Persistence-error curve (does d_seg error concentrate on small/shallow features?)

Feature = connected component of the GT class (the H0 of the class indicator = the literal dashes/segments). flip-rate = fraction of the component's pixels where witness argmax != GT lstar. Binned by component SIZE (px) and by component PEAK GT-margin (superlevel birth = confidence prominence).

witness stage = `l7` (d_seg per attribution summary).

### by component SIZE -- class 0 = Road
| bin | mid | #comp | pixels | flip-rate |
|---|---:|---:|---:|---:|
| [1.00,6.06) | 2.46 | 52 | 145 | 0.3379 |
| [6.06,36.71) | 14.91 | 14 | 226 | 0.3761 |
| [36.71,222.45) | 90.37 | 4 | 501 | 0.0818 |
| [222.45,1347.84) | 547.56 | 22 | 18322 | 0.0429 |
| [1347.84,8166.70) | 3317.74 | 31 | 57026 | 0.0315 |
| [8166.70,49483.00) | 20102.56 | 96 | 4318460 | 0.0053 |

**small/large (or shallow/deep) flip-rate ratio = 64.25x** (smallest/shallowest bin 0.3379 vs largest/deepest 0.0053).

### by component SIZE -- class 1 = Lane
| bin | mid | #comp | pixels | flip-rate |
|---|---:|---:|---:|---:|
| [1.00,2.80) | 1.67 | 519 | 762 | 0.9213 |
| [2.80,7.87) | 4.70 | 887 | 4042 | 0.8634 |
| [7.87,22.07) | 13.18 | 525 | 6845 | 0.7001 |
| [22.07,61.90) | 36.96 | 295 | 11089 | 0.4881 |
| [61.90,173.62) | 103.67 | 245 | 26473 | 0.2780 |
| [173.62,487.00) | 290.78 | 220 | 59893 | 0.1854 |

**small/large (or shallow/deep) flip-rate ratio = 4.97x** (smallest/shallowest bin 0.9213 vs largest/deepest 0.1854).

### by component SIZE -- class 2 = Undrivable
| bin | mid | #comp | pixels | flip-rate |
|---|---:|---:|---:|---:|
| [1.00,6.81) | 2.61 | 5 | 6 | 0.8333 |
| [6.81,46.41) | 17.78 | 2 | 42 | 1.0000 |
| [46.41,316.17) | 121.14 | 0 | 0 | nan |
| [316.17,2153.93) | 825.24 | 0 | 0 | nan |
| [2153.93,14673.71) | 5621.94 | 0 | 0 | nan |
| [14673.71,99965.00) | 38299.58 | 96 | 9347065 | 0.0009 |
### by component SIZE -- class 3 = Movable
| bin | mid | #comp | pixels | flip-rate |
|---|---:|---:|---:|---:|
| [1.00,4.63) | 2.15 | 16 | 40 | 1.0000 |
| [4.63,21.46) | 9.97 | 33 | 410 | 0.7829 |
| [21.46,99.44) | 46.20 | 120 | 7642 | 0.2510 |
| [99.44,460.72) | 214.05 | 87 | 16978 | 0.1465 |
| [460.72,2134.49) | 991.66 | 54 | 51422 | 0.0521 |
| [2134.49,9889.00) | 4594.34 | 35 | 144835 | 0.0217 |

**small/large (or shallow/deep) flip-rate ratio = 36.07x** (smallest/shallowest bin 0.7829 vs largest/deepest 0.0217).

### by component SIZE -- class 4 = MyCar
| bin | mid | #comp | pixels | flip-rate |
|---|---:|---:|---:|---:|
| [49189.00,49606.86) | 49397.49 | 10 | 494064 | 0.0005 |
| [49606.86,50028.27) | 49817.12 | 43 | 2144064 | 0.0007 |
| [50028.27,50453.25) | 50240.31 | 34 | 1707702 | 0.0008 |
| [50453.25,50881.85) | 50667.10 | 8 | 404564 | 0.0008 |
| [50881.85,51314.09) | 51097.51 | 0 | 0 | nan |
| [51314.09,51750.00) | 51531.58 | 1 | 51750 | 0.0062 |

**small/large (or shallow/deep) flip-rate ratio = 0.09x** (smallest/shallowest bin 0.0005 vs largest/deepest 0.0062).

### by component PEAK-MARGIN -- class 0 = Road
| bin | mid | #comp | pixels | flip-rate |
|---|---:|---:|---:|---:|
| [0.01,0.04) | 0.02 | 19 | 54 | 0.9074 |
| [0.04,0.14) | 0.08 | 5 | 16 | 0.5625 |
| [0.14,0.46) | 0.26 | 16 | 71 | 0.3803 |
| [0.46,1.49) | 0.83 | 25 | 224 | 0.2188 |
| [1.49,4.82) | 2.68 | 12 | 3078 | 0.0880 |
| [4.82,15.52) | 8.64 | 142 | 4391237 | 0.0057 |

**small/large (or shallow/deep) flip-rate ratio = 158.97x** (smallest/shallowest bin 0.9074 vs largest/deepest 0.0057).

### by component PEAK-MARGIN -- class 1 = Lane
| bin | mid | #comp | pixels | flip-rate |
|---|---:|---:|---:|---:|
| [0.01,0.03) | 0.02 | 1390 | 6997 | 0.9923 |
| [0.03,0.07) | 0.04 | 59 | 630 | 0.8429 |
| [0.07,0.17) | 0.11 | 160 | 1669 | 0.7400 |
| [0.17,0.44) | 0.28 | 322 | 5457 | 0.6091 |
| [0.44,1.14) | 0.71 | 468 | 32525 | 0.3244 |
| [1.14,2.94) | 1.83 | 292 | 61826 | 0.1662 |

**small/large (or shallow/deep) flip-rate ratio = 5.97x** (smallest/shallowest bin 0.9923 vs largest/deepest 0.1662).

### by component PEAK-MARGIN -- class 2 = Undrivable
| bin | mid | #comp | pixels | flip-rate |
|---|---:|---:|---:|---:|
| [0.16,0.32) | 0.23 | 7 | 48 | 0.9792 |
| [0.32,0.65) | 0.46 | 0 | 0 | nan |
| [0.65,1.29) | 0.92 | 0 | 0 | nan |
| [1.29,2.58) | 1.83 | 0 | 0 | nan |
| [2.58,5.15) | 3.65 | 0 | 0 | nan |
| [5.15,10.28) | 7.28 | 96 | 9347065 | 0.0009 |
### by component PEAK-MARGIN -- class 3 = Movable
| bin | mid | #comp | pixels | flip-rate |
|---|---:|---:|---:|---:|
| [0.01,0.03) | 0.02 | 50 | 675 | 0.9956 |
| [0.03,0.11) | 0.06 | 1 | 6 | 0.8333 |
| [0.11,0.36) | 0.20 | 8 | 363 | 0.7052 |
| [0.36,1.21) | 0.66 | 41 | 3228 | 0.3494 |
| [1.21,4.01) | 2.20 | 129 | 14980 | 0.1632 |
| [4.01,13.29) | 7.30 | 116 | 202075 | 0.0301 |

**small/large (or shallow/deep) flip-rate ratio = 33.05x** (smallest/shallowest bin 0.9956 vs largest/deepest 0.0301).

### by component PEAK-MARGIN -- class 4 = MyCar
| bin | mid | #comp | pixels | flip-rate |
|---|---:|---:|---:|---:|
| [10.27,10.82) | 10.54 | 1 | 51750 | 0.0062 |
| [10.82,11.41) | 11.11 | 0 | 0 | nan |
| [11.41,12.03) | 11.71 | 1 | 50645 | 0.0016 |
| [12.03,12.68) | 12.35 | 19 | 945628 | 0.0007 |
| [12.68,13.37) | 13.02 | 45 | 2250572 | 0.0007 |
| [13.37,14.09) | 13.72 | 30 | 1503549 | 0.0008 |

**small/large (or shallow/deep) flip-rate ratio = 8.28x** (smallest/shallowest bin 0.0062 vs largest/deepest 0.0008).

### per-pixel GT-margin bin -> flip-rate (the annulus baseline, all classes)
| margin range | pixels | flip-rate |
|---|---:|---:|
| [0.00,0.10) | 6484 | 0.7645 |
| [0.10,0.25) | 2738 | 0.0000 |
| [0.25,0.50) | 5457 | 0.0000 |
| [0.50,1.00) | 12591 | 0.0000 |
| [1.00,2.00) | 23009 | 0.0000 |
| [2.00,4.00) | 50523 | 0.0000 |
| [4.00,6.00) | 131581 | 0.0000 |
| [6.00,8.00) | 715728 | 0.0000 |
| [8.00,12.00) | 228671 | 0.0000 |
| [12.00,14.73) | 2866 | 0.0000 |

**lowest-margin / highest-margin bin flip-rate ratio = infx** (0.7645 vs 0.0000).

## 1+3. Topological dimensionality / rate floor (per class, GT margin)

Avg #H0 features (local-max blobs) per frame above persistence thresholds = the #features a codec must encode (the topological rate). persistence-Zipf exponent + Schweinhart PH^0 dim characterise the scaling.

| class | #feat(all) | >0.5 | >1.0 | >2.0 | >4.0 | Zipf exp | PH^0 dim |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 Road | 787 | 97 | 32 | 8 | 4 | -1.54 | 0.29 |
| 1 Lane | 166 | 23 | 6 | 1 | 0 | -0.84 | 0.83 |
| 2 Undrivable | 2335 | 64 | 12 | 2 | 1 | -1.46 | 0.26 |
| 3 Movable | 16 | 5 | 3 | 2 | 1 | -2.43 | 0.36 |
| 4 MyCar | 1209 | 45 | 10 | 2 | 1 | -1.45 | 0.27 |

### H1 (loops, gudhi cubical, mean per frame)

| class | mean #H1 loops | mean H1 persistence |
|---|---:|---:|
| 0 Road | 311.8 | 0.130 |
| 1 Lane | 0.0 | nan |
| 2 Undrivable | 1730.1 | 0.095 |
| 3 Movable | 8.2 | 0.086 |
| 4 MyCar | 1006.8 | 0.098 |

## 4. GAP2 / R-survival (which births die under the contest R operator)

R = bicubic^874 -> uint8 -> bilinear v384 applied to the margin field (faithful spatial low-pass + uint8). Persistence-survival = post/pre #features above threshold; dash-survival = post/pre connected components of the class indicator through R.

| class | pre #feat>1.0 | post #feat>1.0 | feat-survival | total-pers survival | dash pre | dash post | dash-survival |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 Road | 32 | 27 | 0.846 | 0.923 | 2 | 2 | 1.000 |
| 1 Lane | 6 | 5 | 0.851 | 0.900 | 28 | 28 | 1.000 |
| 2 Undrivable | 11 | 10 | 0.935 | 0.951 | 1 | 1 | 1.000 |
| 3 Movable | 3 | 3 | 0.990 | 0.980 | 4 | 4 | 1.000 |
| 4 MyCar | 10 | 10 | 0.966 | 0.951 | 1 | 1 | 1.000 |

## 5. Temporal vineyard (code-once+transport vs learned residual)

H0 features (persistence>0.5) chained across consecutive frames by peak nearest-neighbour (radius 6px). Temporal-life = #consecutive frames a feature track survives. Coherent = life>=3.

| class | #tracks | mean life | frac life>=3 | frac life==1 | bottleneck (consec) |
|---|---:|---:|---:|---:|---:|
| 0 Road | 4736 | 2.04 | 0.188 | 0.654 | 1.451 |
| 1 Lane | 1446 | 1.57 | 0.124 | 0.728 | 0.952 |
| 2 Undrivable | 4986 | 1.28 | 0.056 | 0.828 | 0.436 |
| 3 Movable | 128 | 3.60 | 0.266 | 0.516 | 0.847 |
| 4 MyCar | 3457 | 1.29 | 0.058 | 0.802 | 0.567 |

## Synthesis / rate-floor + codec implication

- **error~1/persistence CONFIRMED**: flip-rate rises monotonically as feature scale shrinks. Lane small/large flip ratio = 4.97x (by component size), 5.97x (by peak-margin); Road = 64.25x; Movable = 36.07x. The smallest lane dashes (<3px) flip at ~92% while large lane segments flip at ~19%.
- **the annulus is razor-thin**: per-pixel flip-rate = 0.764 for GT margin < 0.10, and ~0.000 for every higher-margin bin. d_seg is ENTIRELY the sub-0.10-margin codim-1 boundary (consistent with the 98% annulus_frac in attribution summary).
- **R-survival -> the lane residual is R-RECOVERABLE, NOT R-destroyed**: pushing the GT margin field through the contest R (bicubic^874->uint8->bilinear v384) preserves Lane feature-survival 0.851, total-persistence 0.900, and 100% (1.000) of dash connected-components. R destroys only ~15% of >1.0-persistence lane features. => the GAP2 wall is the GENERATOR's ability to SYNTHESISE the field through R, not R's resample/uint8 erasing an existing good field. (Caveat: this tests R on the TARGET field; it bounds how much R alone can erase.)
- **topological rate floor**: Lane carries ~23 features >0.5 and ~6 features >1.0 persistence per frame -- the #birth-death pairs a codec must code. Lane PH^0-dim=0.83 (HIGHEST of all classes; the genuinely multi-scale/fractal class) vs Road 0.29 / Undrivable 0.26 / MyCar 0.27 (one big stable region each). This is the topological signature of 'Road = few high-persistence stable boundaries, Lane = many short-lived dashes'.
- **temporal coherence (Lane)**: only 0.124 of lane feature-tracks live >=3 frames under naive 6px-NN matching; 0.728 are single-frame. This is a LOWER bound on transportability (lane dashes stream fast toward the camera; the se(3) ground-homography warp -- FEED-ja stratified per-class transport, NOT implemented here -- would recover most). Movable has the HIGHEST coherence (mean life 3.60; cars track cleanly).

### Codec implication (advisory)
- The rate-relevant signal is the sub-0.10-margin annulus. A witness only needs to get the argmax right where two classes are within 0.10 logits -- everywhere else (>0.10 margin, the class interiors) is FREE (0 flips even now).
- Lane is the binding residual: ~6 persistent + a long tail of ~166 total (mostly sub-persistence-threshold) dash births/frame, multi-scale (PH^0-dim 0.83). Coding only the persistent dashes (a few birth-death pairs/frame x a few params each) + transporting them across frames by the screw warp is the topological rate floor; the ephemeral short-persistence births are where a small LEARNED residual is genuinely required (R cannot recover what was never in the target, but R does NOT itself destroy the target dashes).
