# Witness per-stage / per-pixel / per-class d_seg attribution  [macOS-numpy advisory . NON-PROMOTABLE]

- generated: `2026-07-11T00:06:44Z`
- verdict pairs: **24** of 600 (strided `range(0,600,25)[:24]`, identical across all ckpts)
- render: deploy-faithful fp32 ONE-CODEPATH, int8-dequantized EMA shadow, self-orient fixed-point `so_iters=4` (byte-close/inflate authority)
- d_seg = realized through-R frozen CPU-torch SegNet argmax disagreement vs GT `lstars` (NOT a proxy). Pointer 0.19110 UNMOVED; this is advisory.
- class order (CLAUDE.md): 0=Road 1=Lane 2=Undrivable 3=Movable 4=MyCar

## Stage chain + realized d_seg

| stage | epoch | softmax_temp | mean fp-iters | realized d_seg | n_wrong | annulus% of wrong |
|---|--:|--:|--:|--:|--:|--:|
| Best | 50 | 1.0000 | 0.00 | **0.029262** | 138,076 | 54.6% |

## Per-class disagreement per stage  (fraction of each GT class's pixels mislabeled)

| stage | Road | Lane | Undrivable | Movable | MyCar |
|---|---|---|---|---|---|
| Best | 9.41% | 40.62% | 0.92% | 3.33% | 0.02% |

### Flip MASS share per stage  (of ALL wrong pixels, fraction whose GT class is c)

| stage | Road | Lane | Undrivable | Movable | MyCar |
|---|---|---|---|---|---|
| Best | 74.7% | 8.1% | 15.5% | 1.4% | 0.2% |

## Stage transitions  (per-pixel, summed over the verdict pairs)

| transition | net d_seg Delta | corrected | regressed | persist-wrong | PRIMED | STUCK | corrected annulus% | persist-wrong annulus% |
|---|--:|--:|--:|--:|--:|--:|--:|--:|

### Per-transition CORRECTED by class

| transition | Road | Lane | Undrivable | Movable | MyCar |
|---|---|---|---|---|---|

### Per-transition PRIMED by class  (persist-wrong with realized SegNet margin moving toward GT)

| transition | Road | Lane | Undrivable | Movable | MyCar |
|---|---|---|---|---|---|

### Per-transition STUCK by class  (persist-wrong, margin flat/worse -> store/deterministic candidates)

| transition | Road | Lane | Undrivable | Movable | MyCar |
|---|---|---|---|---|---|

