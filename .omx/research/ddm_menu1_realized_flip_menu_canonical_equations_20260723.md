# DDM MENU1 realized-flip menu canonical equations

`research_only=true`

`score_claim=false`

`evidence_axis=[macOS-CPU frozen-scorer advisory]`

## Menu product and count custody

Let `C` be the 2,649 SN1 clusters and `F` the six governed fixes. The compiled
menu is

```text
M = C x F,    |M| = 2,649 x 6 = 15,894.
```

Each row is

```text
m = (
  cluster_id,
  mechanism_bucket,
  composition_pool_id,
  Delta errors_realized,
  Delta B_counted,
  B_free,
  B_null
).
```

The V19C counts are distinct:

```text
E_residual = 2,265,811
E_role = 658,180
E_total = E_residual + E_role = 2,923,991.
```

`E_residual` binds the SN1 residual menu. `E_total` alone enters `d_seg`.

## Authority action and exact transitions

For an exact receiver state `x`,

```text
S(x) = 100 d_seg(x)
     + sqrt(10 d_pose(x))
     + 25 B(x) / 37,545,489.
```

For parent cells `p`, child cells `q`, and target cells `y`,

```text
corrected  = count((p != y) and (q == y))
introduced = count((p == y) and (q != y))

Delta errors_realized = corrected - introduced
errors(q) = errors(p) - Delta errors_realized.
```

Every reported delta is measured after the actual camera receiver, final
uint8, frozen SegNet, frozen PoseNet YUV6 path, and exact payload byte count.

## FREE, NULL, and COUNTED partition

Every row obeys

```text
Delta B = B_COUNTED + B_FREE + B_NULL.
```

`COUNTED` is video-derived payload. `FREE` is a generic deterministic or
scorer-derived constant with a receiver-survival proof. `NULL` is a certified
`ker(A)` component. An unproven scorer-derived constant is
`FREE_candidate`, not `FREE`, and remains waterfill-ineligible.

The AT1X manifest has zero amplitude factors and explicitly requires
through-R uint8 survival before a nonzero amplitude row. Therefore

```text
BN expected-stat table -> camera affine
```

is a named formulation blocker, not a fabricated zero-byte RGB transform.

## Frame1-only affine ladder

All GT-fitted amplitude targets use `gt_f1`; frame 0 is preserved.

For the 12-byte scalar rung,

```text
a = sigma_target / sigma_source
b = mu_target - a mu_source
frame1' = round_clip(a frame1 + b).
```

`a,b` are two float16 values plus an 8-byte typed header.

For temporal segment `k` and RGB channel `c`,

```text
a[k,c] = sigma_target[k,c] / sigma_source[k,c]
b[k,c] = mu_target[k,c] - a[k,c] mu_source[k,c]
k(pair) = floor(16 pair / 600)
frame1'[pair,c] = round_clip(a[k(pair),c] frame1[pair,c] + b[k(pair),c]).
```

The 16-knot payload is

```text
12-byte header + 16 x 6 x 2-byte values = 204 bytes.
```

For class `c`, camera row band `r`, and RGB channel `j`,

```text
a[c,r,j] = sigma_target[c,r,j] / sigma_source[c,r,j]
b[c,r,j] = mu_target[c,r,j] - a[c,r,j] mu_source[c,r,j].
```

The 5-class x 16-band payload is

```text
14-byte header + 5 x 16 x 6 x 2-byte values = 974 bytes.
```

Class ownership comes from the actual V19C decoder layers, with a
nearest-decoder-palette fallback only where no layer owns a scorer cell.

## Geometry composition

For semantic signed-distance fields `phi_c`,

```text
hard(x) = palette[argmax_c phi_c(x)]

coverage_c(x) = clip(1/2 + phi_c(x), 0, 1)
analytic(x) = sum_c coverage_c(x) palette[c] / sum_c coverage_c(x).
```

Hard placement is used where the top-two SDF separation is at least one camera
pixel; analytic coverage is used elsewhere. Only actual decoder-owned frame1
pixels are replaced, after which the local affine is applied. Geometry has
zero additional counted parameters in this fixed formulation.

## Non-additive pool competition

Let

```text
P = {scalar, temporal, local, local+hard+analytic}.
```

All `p in P` are measured independently from exact V19C. The pool winner is

```text
p* = argmin_(p in P union {V19C}) S(p)
     subject to B(p) <= 200,000.
```

Independent pool deltas are never summed. The measured winner is the composed
local+hard+analytic row:

```text
B(p*) = 138,801
d_seg(p*) = 0.07051923116048177
d_pose(p*) = 36.6181847780574
S(p*) = 26.28022355199344.
```

The PT1 186-byte spectrum row is dominated by the PT1 30-byte statistics row
under their shared control. That dominance is retained as cross-control pool
evidence, not imported as a V19C delta.

## Targeted formulation and dual rejection

The top-cluster prototype acts on current predicted Undrivable sites whose
target is Road, whose boundary band is `ANNULUS_2_TO_5`, and whose pair belongs
to the top SN1 cluster support. It emits the generic decoder Road prototype.

The sidecar does not encode the historical `d2`, G3-tail, or semantic-history
axes, so this is a formulation-scoped approximation.

Measured from `p*`,

```text
Delta B = 88,568
B_total = 227,369 > 200,000
Delta errors_realized = -5,069,958
S_targeted = 30.104108909525713 > S(p*).
```

It fails both the byte and strict-joint-gain gates. The next measurement is a
Fisher/margin, corrected-inner-Jacobian targeted actuator, not merely a more
compressed copy of this coarse prototype.

## PA1 pose-amplitude pool

PA1 is a distinct cross-control pool:

```text
pose_amplitude != paint_amplitude.
```

Its frame0-only rows preserve frame1 Seg bytes structurally within PA1, but
their deltas are not V19C prices. The scorer-stat 0-byte row is
`FREE_candidate` until receiver survival. Exact V19C joint composition is
required before either PA1 row can enter this curve.

## Verdict scope

These equations govern the exact local macOS-CPU frozen-scorer advisory
measurement only. They do not establish a contest-CPU/CUDA score, archive
promotion, family or paradigm negative, or pointer movement.
