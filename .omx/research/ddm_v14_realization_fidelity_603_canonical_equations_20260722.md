# DDM v14 receiver-realization fidelity — canonical equations note

Date: 2026-07-22  
Equation: `ddm_describe_line_rate_distortion_bracket_v1`  
Axis: `[macOS-CPU frozen-scorer advisory]`; `score_claim=false`

## Receiver map

Let `s_t : Omega_s -> C` be the ordered semantic cell map at scorer resolution, where
`C = {UndrivableBoundary, Road, Lane, Movable, MyCar}` and later classes overwrite earlier ones.
G1 supplies the authoritative Movable support; it does not union with the inherited support.

Let `U_nn` be nearest-neighbor lift from `Omega_s = 384x512` to camera resolution
`Omega_c = 874x1164`, and let `q_c in {0,...,255}^3` be the counted uint8 prototype for class c.
The v14 camera image is

```text
I_t(p) = q_{U_nn(s_t)(p)},       p in Omega_c,
Y_t    = R_bilinear(I_t),        Y_t in uint8^(384x512x3),
z_t    = SegNet(Y_t),
D_seg  = mean_p 1[argmax z_t(p) != g_t(p)].
```

The selected profile uses Movable `q_M = (107,0,114)`, amplitude 255, zero coverage expansion,
and the paint order above. The profile is 23 payload bytes / 85 exact ZIP-home bytes.

For a static rule `r = (a -> b, Omega_r)`, the decoder-derived activation is

```text
A_r,t(p) = 1[p in Omega_r] * 1[s_t(p) = a],
s'_t(p)  = b if A_r,t(p)=1 else s_t(p).
```

The rule never stores a per-frame GT argmax table. The two parametric rules use 12 payload bytes;
the sparse-all aggregate rule uses 4,107 payload bytes and remains research-only.

## Realization transfer and joint objective

For a cell-space forecast `Delta D_cell` and the measured receiver delta `Delta D_receiver`, define

```text
eta_r = - Delta D_receiver / Delta D_cell.
```

Positive `eta_r` means the receiver realizes the forecasted improvement. Measured n600 values:

| Rule | Delta D_cell | Delta D_receiver | eta |
|---|---:|---:|---:|
| Movable midband | +0.001352979872 forecast improvement | +0.001039276123 harm | -0.768138643107 |
| Horizon row | +0.001013403998 forecast improvement | -0.000053575304 gain | 0.052866679113 |
| Sparse-all | +0.007806744046 forecast improvement | +0.002431810167 harm | -0.311501203673 |

For exact archive bytes `B`, the advisory contest-form objective change is

```text
Delta J = 100 Delta D_seg
        + sqrt(10 (D_pose + Delta D_pose)) - sqrt(10 D_pose)
        + 25 Delta B / 37_545_489.
```

The horizon row has `Delta B=508`, `Delta D_seg=-0.000053575304`,
`Delta D_pose=+0.000131379690`, and `Delta J=-0.005003006483`, with score gain per additional
byte 0.000009848438. This is advisory means evidence, not a contest score.

## Free-context non-subtraction law

G4 measures 490,794 B -> 401,633 B for a future context-coded innovation stream, saving 89,161 B
/ 18.166685%. Because these static archives contain no such innovation stream,

```text
B_exact_candidate != B_exact_candidate - 89_161.
```

The honest derived composition is 133,247 + 401,633 = 534,880 B before container overhead. It is
outside the box and is not a candidate.

## Physical-BEV custody precondition

The AR(1)-whitened BEV lane equation is undefined until the receiver owns an independently observed
physical homography/liveCalibration and decoder-free metric pose. Scorer Pose6 does not satisfy this
precondition. Current disposition:
`BLOCKED_NO_DECODER_FREE_PHYSICAL_BEV_CUSTODY`.

## Registered anchor

The canonical evaluator's v14 branch consumes the n64 and n600 fixed receiver rows plus the G4
receiver row, binds the three receipt SHA-256 values, and emits
`ADVISORY_V14_RECEIVER_REALIZATION_REPAIR_PARTIAL_STATIC_CELL_FORECAST_FALSIFIED`. The equation has
six empirical anchors after registration. Recalibration is required on any new receiver prototype,
static-rule grammar, or physical-BEV custody surface.

Receipt bindings:

- n64: `bc9d01c6a3691e1103a580d0e1a34088ff134258b2788b2380930fe85fbae703`
- n600: `82d3249908d42a86575c407ab3d7acdf9b3706b31225f2e46862b2472966e5a9`
- G4 receiver projection: `0b35be44d944bd5a929097bb3967ba7b7c7ce068e2f067d1546ab149cc9e44da`

Pointer: `0.1910828242 [contest-CPU]` — UNMOVED.
