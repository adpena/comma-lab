# Witness per-stage / per-pixel / per-class d_seg attribution  [macOS-numpy advisory . NON-PROMOTABLE]

- generated: `2026-07-19T03:00:08Z`
- verdict pairs: **600** of 600 (strided `range(0,600,1)[:600]`, identical across all ckpts)
- render: deploy-faithful fp32 ONE-CODEPATH, int8-dequantized EMA shadow, self-orient fixed-point `so_iters=4` (byte-close/inflate authority)
- d_seg = realized through-R frozen CPU-torch SegNet argmax disagreement vs GT `lstars` (NOT a proxy). Pointer 0.19110 UNMOVED; this is advisory.
- class order (CLAUDE.md): 0=Road 1=Lane 2=Undrivable 3=Movable 4=MyCar

## Stage chain + realized d_seg

| stage | epoch | softmax_temp | mean fp-iters | realized d_seg | n_wrong | annulus% of wrong |
|---|--:|--:|--:|--:|--:|--:|
| c2best | 725 | 0.2168 | 4.00 | **0.003513** | 414,378 | 96.8% |

## Per-class disagreement per stage  (fraction of each GT class's pixels mislabeled)

| stage | Road | Lane | Undrivable | Movable | MyCar |
|---|---|---|---|---|---|
| c2best | 0.51% | 21.56% | 0.09% | 3.58% | 0.07% |

### Flip MASS share per stage  (of ALL wrong pixels, fraction whose GT class is c)

| stage | Road | Lane | Undrivable | Movable | MyCar |
|---|---|---|---|---|---|
| c2best | 33.4% | 35.9% | 12.7% | 12.6% | 5.3% |

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

